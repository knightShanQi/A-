from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, build_daily_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DUCKDB_PATH = PROJECT_ROOT / "data" / "openclaw_market_data.duckdb"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".cache" / "next_day_factor_model"
DEFAULT_PRICE_TABLE = "a_share_daily_prices"
MODEL_VERSION = 2
DEFAULT_SYMBOL_PREFIXES = ("0", "3", "6")
DEFAULT_STOCK_BASIC_PATH = PROJECT_ROOT / ".cache" / "baidu_daily_stock" / "stock_basic.csv"
ProgressCallback = Callable[[str], None]


MARKET_CONTEXT_COLUMNS = [
    "market_ret_1",
    "market_ret_5",
    "market_ret_20",
    "market_up_ratio_1",
    "market_up_ratio_5",
    "market_up_ratio_20",
    "market_volatility_20",
    "market_turnover_20",
    "market_activity_ratio_5",
    "market_amount_ratio_5",
    "market_drawdown_20",
]

INDUSTRY_CONTEXT_COLUMNS = [
    "industry_ret_1",
    "industry_ret_5",
    "industry_ret_20",
    "industry_up_ratio_1",
    "industry_up_ratio_5",
    "industry_up_ratio_20",
    "industry_turnover_5",
    "industry_amount_ratio_5",
    "industry_relative_ret_5",
    "industry_relative_ret_20",
    "industry_rank_ret_5",
    "industry_rank_up_ratio_5",
    "industry_rank_amount_ratio_5",
]

SEGMENT_CONTEXT_COLUMNS = [
    "segment_ret_1",
    "segment_ret_5",
    "segment_ret_20",
    "segment_up_ratio_1",
    "segment_up_ratio_5",
    "segment_up_ratio_20",
    "segment_turnover_5",
    "segment_amount_ratio_5",
    "segment_relative_ret_5",
    "segment_relative_ret_20",
    "segment_rank_ret_5",
    "segment_rank_activity_5",
]

MICROSTRUCTURE_COLUMNS = [
    "limit_up_flag",
    "limit_down_flag",
    "near_limit_up_flag",
    "near_limit_down_flag",
    "limit_up_streak_3",
    "limit_down_streak_3",
    "trade_gap_days",
    "recent_resume_flag",
    "market_main_flag",
    "market_chinext_flag",
    "market_star_flag",
]


def _emit(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _context_default(column: str) -> float:
    return 0.5 if "up_ratio" in column or "rank" in column else 0.0


def _split_context_by_key(frame: pd.DataFrame | None, key_column: str) -> dict[str, pd.DataFrame]:
    if not isinstance(frame, pd.DataFrame) or frame.empty or key_column not in frame.columns:
        return {}
    return {str(key): group.copy() for key, group in frame.groupby(key_column, sort=False)}


def _context_slice(context: object, key_column: str, key_value: object) -> pd.DataFrame:
    key = str(key_value)
    if isinstance(context, dict):
        selected = context.get(key)
        return selected if isinstance(selected, pd.DataFrame) else pd.DataFrame()
    if isinstance(context, pd.DataFrame) and not context.empty and key_column in context.columns:
        return context.loc[context[key_column].astype(str).eq(key)].copy()
    return pd.DataFrame()


def _context_available(context: object) -> bool:
    if isinstance(context, dict):
        return bool(context)
    return isinstance(context, pd.DataFrame) and not context.empty


@dataclass(slots=True)
class DateSplits:
    analysis_start: str
    analysis_end: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str


@dataclass(slots=True)
class PipelineResult:
    output_dir: str
    model_path: str
    feature_ranking_path: str
    cluster_summary_path: str
    selected_factors_path: str
    metrics_path: str
    report_path: str
    selected_factors: list[str]
    metrics: dict[str, object]
    date_splits: dict[str, str]
    data_summary: dict[str, object]


def _safe_table_name(table: str) -> str:
    text = str(table or "").strip()
    if not text.replace("_", "").isalnum() or not text[0].isalpha():
        raise ValueError(f"unsafe table name: {table}")
    return text


def _date_string(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return _date_string(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _normalize_scores(values: Sequence[float]) -> np.ndarray:
    series = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if series.nunique(dropna=False) <= 1:
        return np.zeros(len(series), dtype=float)
    return series.rank(method="average", pct=True).to_numpy(dtype=float)


def _prefix_filter(prefixes: Sequence[str]) -> tuple[str, list[str]]:
    clean = [str(prefix).strip() for prefix in prefixes if str(prefix).strip()]
    if not clean:
        clean = list(DEFAULT_SYMBOL_PREFIXES)
    clause = " or ".join("symbol like ?" for _ in clean)
    return f"({clause})", [f"{prefix}%" for prefix in clean]


def _symbol_market_segment(symbol: object, market: object = "") -> str:
    text = str(symbol or "").zfill(6)
    market_text = str(market or "").strip().lower()
    if text.startswith(("300", "301")) or "创业" in market_text or "cyb" in market_text:
        return "chinext"
    if text.startswith(("688", "689")) or "科创" in market_text:
        return "star"
    if text.startswith(("000", "001", "002", "003", "600", "601", "603", "605")):
        return "main"
    if text.startswith(("8", "4", "9")) or "北交" in market_text or "bj" in market_text:
        return "beijing"
    return "other"


def _load_stock_basic_metadata(stock_basic_path: Path | str | None = DEFAULT_STOCK_BASIC_PATH) -> pd.DataFrame:
    path = Path(stock_basic_path) if stock_basic_path else DEFAULT_STOCK_BASIC_PATH
    columns = ["symbol", "name", "industry", "market", "is_st", "market_segment"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if frame.empty:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(index=frame.index)
    if "symbol" in frame.columns:
        result["symbol"] = frame["symbol"].astype(str).str.extract(r"(\d{1,6})", expand=False).fillna("").str.zfill(6)
    elif "code" in frame.columns:
        result["symbol"] = frame["code"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    else:
        return pd.DataFrame(columns=columns)
    result["name"] = (frame["name"] if "name" in frame.columns else pd.Series("", index=frame.index)).fillna("").astype(str)
    result["industry"] = (
        frame["industry"] if "industry" in frame.columns else pd.Series("unknown", index=frame.index)
    ).fillna("unknown").astype(str)
    result["market"] = (frame["market"] if "market" in frame.columns else pd.Series("", index=frame.index)).fillna("").astype(str)
    result["is_st"] = result["name"].str.upper().str.contains("ST|退", regex=True, na=False)
    result["market_segment"] = [
        _symbol_market_segment(symbol, market) for symbol, market in zip(result["symbol"], result["market"])
    ]
    result = result[result["symbol"].str.len().eq(6)].copy()
    return result.drop_duplicates("symbol", keep="last").reindex(columns=columns).reset_index(drop=True)


def inspect_duckdb_coverage(
    duckdb_path: Path | str = DEFAULT_DUCKDB_PATH,
    *,
    price_table: str = DEFAULT_PRICE_TABLE,
    symbol_prefixes: Sequence[str] = DEFAULT_SYMBOL_PREFIXES,
) -> dict[str, object]:
    table = _safe_table_name(price_table)
    clause, params = _prefix_filter(symbol_prefixes)
    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        frame = connection.execute(
            f"""
            select min(trade_date) as min_date,
                   max(trade_date) as max_date,
                   count(*) as rows,
                   count(distinct symbol) as symbols,
                   count(distinct trade_date) as trade_days
            from {table}
            where {clause}
            """,
            params,
        ).fetchdf()
    if frame.empty:
        raise RuntimeError("DuckDB price table is empty for the requested symbol prefixes.")
    row = frame.iloc[0].to_dict()
    return {
        "min_date": _date_string(row.get("min_date")),
        "max_date": _date_string(row.get("max_date")),
        "rows": int(row.get("rows") or 0),
        "symbols": int(row.get("symbols") or 0),
        "trade_days": int(row.get("trade_days") or 0),
        "symbol_prefixes": list(symbol_prefixes),
        "price_table": table,
    }


def _resolve_analysis_dates(
    connection,
    *,
    price_table: str,
    symbol_prefixes: Sequence[str],
    start_date: str | None,
    end_date: str | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    clause, params = _prefix_filter(symbol_prefixes)
    coverage = connection.execute(
        f"""
        select min(trade_date) as min_date, max(trade_date) as max_date
        from {price_table}
        where {clause}
        """,
        params,
    ).fetchdf()
    if coverage.empty or pd.isna(coverage.loc[0, "min_date"]) or pd.isna(coverage.loc[0, "max_date"]):
        raise RuntimeError("DuckDB price table is empty for the requested symbol prefixes.")
    max_date = pd.Timestamp(coverage.loc[0, "max_date"])
    min_date = pd.Timestamp(coverage.loc[0, "min_date"])
    analysis_end = pd.Timestamp(pd.to_datetime(end_date, errors="coerce")) if end_date else max_date
    if pd.isna(analysis_end):
        raise ValueError(f"Invalid end_date: {end_date}")
    if start_date:
        analysis_start = pd.Timestamp(pd.to_datetime(start_date, errors="coerce"))
    else:
        analysis_start = analysis_end - pd.DateOffset(years=20)
    if pd.isna(analysis_start):
        raise ValueError(f"Invalid start_date: {start_date}")
    analysis_start = max(pd.Timestamp(analysis_start), min_date)
    analysis_end = min(pd.Timestamp(analysis_end), max_date)
    if analysis_start >= analysis_end:
        raise ValueError("analysis_start must be earlier than analysis_end")
    return analysis_start.normalize(), analysis_end.normalize()


def _fetch_trade_dates(
    connection,
    *,
    price_table: str,
    symbol_prefixes: Sequence[str],
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
) -> list[pd.Timestamp]:
    clause, params = _prefix_filter(symbol_prefixes)
    frame = connection.execute(
        f"""
        select distinct trade_date
        from {price_table}
        where {clause}
          and trade_date >= cast(? as date)
          and trade_date <= cast(? as date)
        order by trade_date
        """,
        [*params, analysis_start.strftime("%Y-%m-%d"), analysis_end.strftime("%Y-%m-%d")],
    ).fetchdf()
    dates = [pd.Timestamp(value).normalize() for value in frame["trade_date"].tolist()]
    if len(dates) < 120:
        raise RuntimeError("Not enough trade dates for a chronological train/validation/test split.")
    return dates


def build_date_splits(
    connection,
    *,
    price_table: str = DEFAULT_PRICE_TABLE,
    symbol_prefixes: Sequence[str] = DEFAULT_SYMBOL_PREFIXES,
    start_date: str | None = None,
    end_date: str | None = None,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> DateSplits:
    table = _safe_table_name(price_table)
    analysis_start, analysis_end = _resolve_analysis_dates(
        connection,
        price_table=table,
        symbol_prefixes=symbol_prefixes,
        start_date=start_date,
        end_date=end_date,
    )
    dates = _fetch_trade_dates(
        connection,
        price_table=table,
        symbol_prefixes=symbol_prefixes,
        analysis_start=analysis_start,
        analysis_end=analysis_end,
    )
    train_index = max(1, min(len(dates) - 3, int(len(dates) * float(train_fraction)) - 1))
    validation_index = max(
        train_index + 1,
        min(len(dates) - 2, int(len(dates) * float(train_fraction + validation_fraction)) - 1),
    )
    return DateSplits(
        analysis_start=_date_string(analysis_start),
        analysis_end=_date_string(analysis_end),
        train_end=_date_string(dates[train_index]),
        validation_start=_date_string(dates[train_index + 1]),
        validation_end=_date_string(dates[validation_index]),
        test_start=_date_string(dates[validation_index + 1]),
        test_end=_date_string(dates[-1]),
    )


def _fetch_symbols(
    connection,
    *,
    price_table: str,
    symbol_prefixes: Sequence[str],
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
    min_history_rows: int,
    exclude_symbols: Sequence[str] | None = None,
    symbol_limit: int | None = None,
) -> list[str]:
    clause, params = _prefix_filter(symbol_prefixes)
    limit_clause = "" if symbol_limit is None else "limit ?"
    limit_params: list[object] = [] if symbol_limit is None else [int(symbol_limit)]
    exclude = [str(symbol).zfill(6) for symbol in (exclude_symbols or []) if str(symbol).strip()]
    exclude_clause = ""
    exclude_params: list[object] = []
    if exclude:
        exclude_clause = f"and symbol not in ({','.join('?' for _ in exclude)})"
        exclude_params = exclude
    frame = connection.execute(
        f"""
        select symbol
        from {price_table}
        where {clause}
          and trade_date >= cast(? as date)
          and trade_date <= cast(? as date)
          {exclude_clause}
        group by symbol
        having count(*) >= ?
        order by symbol
        {limit_clause}
        """,
        [
            *params,
            analysis_start.strftime("%Y-%m-%d"),
            analysis_end.strftime("%Y-%m-%d"),
            *exclude_params,
            int(min_history_rows),
            *limit_params,
        ],
    ).fetchdf()
    return frame["symbol"].astype(str).str.zfill(6).tolist()


def _iter_chunks(values: Sequence[str], size: int) -> Iterator[list[str]]:
    step = max(int(size), 1)
    for index in range(0, len(values), step):
        yield list(values[index : index + step])


def _fetch_price_batch(
    connection,
    *,
    price_table: str,
    symbols: Sequence[str],
    query_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    placeholders = ",".join("?" for _ in symbols)
    frame = connection.execute(
        f"""
        select symbol, name, trade_date, open, high, low, close,
               pre_close, change, pct_chg, volume, amount, turnover_rate
        from {price_table}
        where symbol in ({placeholders})
          and trade_date >= cast(? as date)
          and trade_date <= cast(? as date)
        order by symbol, trade_date
        """,
        [*symbols, query_start.strftime("%Y-%m-%d"), analysis_end.strftime("%Y-%m-%d")],
    ).fetchdf()
    if frame.empty:
        return frame
    frame["symbol"] = frame["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    for column in ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "volume", "amount", "turnover_rate"]:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    return frame.dropna(subset=["symbol", "trade_date", "close"]).reset_index(drop=True)


def build_market_context_frame(
    duckdb_path: Path | str,
    *,
    price_table: str = DEFAULT_PRICE_TABLE,
    symbol_prefixes: Sequence[str] = DEFAULT_SYMBOL_PREFIXES,
    query_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
) -> pd.DataFrame:
    table = _safe_table_name(price_table)
    clause, params = _prefix_filter(symbol_prefixes)
    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        frame = connection.execute(
            f"""
            select trade_date,
                   avg(
                       case
                           when pct_chg is not null then pct_chg / 100.0
                           when pre_close is not null and pre_close <> 0 then close / pre_close - 1.0
                           else null
                       end
                   ) as market_ret_1,
                   avg(
                       case
                           when pct_chg is not null and pct_chg > 0 then 1.0
                           when pct_chg is not null then 0.0
                           when pre_close is not null and pre_close <> 0 and close > pre_close then 1.0
                           when pre_close is not null and pre_close <> 0 then 0.0
                           else null
                       end
                   ) as market_up_ratio_1,
                   avg(turnover_rate) as market_turnover_1,
                   sum(coalesce(amount, 0.0)) as market_amount_1,
                   count(*) as market_stock_count
            from {table}
            where {clause}
              and trade_date >= cast(? as date)
              and trade_date <= cast(? as date)
            group by trade_date
            order by trade_date
            """,
            [*params, query_start.strftime("%Y-%m-%d"), analysis_end.strftime("%Y-%m-%d")],
        ).fetchdf()
    if frame.empty:
        return pd.DataFrame(columns=["trade_date", *MARKET_CONTEXT_COLUMNS])
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
    for column in ["market_ret_1", "market_up_ratio_1", "market_turnover_1", "market_amount_1", "market_stock_count"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["market_ret_1"] = frame["market_ret_1"].fillna(0.0)
    frame["market_up_ratio_1"] = frame["market_up_ratio_1"].ffill().fillna(0.5)
    frame["market_turnover_1"] = frame["market_turnover_1"].ffill().fillna(0.0)
    frame["market_ret_5"] = frame["market_ret_1"].rolling(5, min_periods=2).mean().fillna(frame["market_ret_1"])
    frame["market_ret_20"] = frame["market_ret_1"].rolling(20, min_periods=5).mean().fillna(frame["market_ret_5"])
    frame["market_up_ratio_5"] = frame["market_up_ratio_1"].rolling(5, min_periods=2).mean().fillna(frame["market_up_ratio_1"])
    frame["market_up_ratio_20"] = frame["market_up_ratio_1"].rolling(20, min_periods=5).mean().fillna(frame["market_up_ratio_5"])
    frame["market_volatility_20"] = frame["market_ret_1"].rolling(20, min_periods=5).std().fillna(0.0)
    frame["market_turnover_20"] = frame["market_turnover_1"].rolling(20, min_periods=5).mean().fillna(frame["market_turnover_1"])
    turnover_base = frame["market_turnover_20"].replace(0, np.nan)
    frame["market_activity_ratio_5"] = (
        frame["market_turnover_1"].rolling(5, min_periods=2).mean() / turnover_base - 1.0
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    amount_5 = frame["market_amount_1"].rolling(5, min_periods=2).mean()
    amount_20 = frame["market_amount_1"].rolling(20, min_periods=5).mean().replace(0, np.nan)
    frame["market_amount_ratio_5"] = (amount_5 / amount_20 - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    market_index = (1.0 + frame["market_ret_1"]).cumprod()
    frame["market_drawdown_20"] = (
        market_index / market_index.rolling(20, min_periods=5).max().replace(0, np.nan) - 1.0
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for column in MARKET_CONTEXT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame[["trade_date", *MARKET_CONTEXT_COLUMNS]].copy()


def build_industry_context_frame(
    duckdb_path: Path | str,
    *,
    stock_basic: pd.DataFrame,
    market_context: pd.DataFrame | None,
    price_table: str = DEFAULT_PRICE_TABLE,
    symbol_prefixes: Sequence[str] = DEFAULT_SYMBOL_PREFIXES,
    query_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
    min_symbols_per_industry: int = 5,
    exclude_st: bool = True,
) -> pd.DataFrame:
    if not isinstance(stock_basic, pd.DataFrame) or stock_basic.empty:
        return pd.DataFrame(columns=["trade_date", "industry", *INDUSTRY_CONTEXT_COLUMNS])
    metadata_columns = ["symbol", "industry", "is_st"] if "is_st" in stock_basic.columns else ["symbol", "industry"]
    metadata = stock_basic[metadata_columns].copy()
    metadata["symbol"] = metadata["symbol"].astype(str).str.zfill(6)
    metadata["industry"] = metadata["industry"].fillna("unknown").astype(str)
    if exclude_st and "is_st" in metadata.columns:
        metadata = metadata.loc[~metadata["is_st"].astype(bool)].copy()
    metadata = metadata.drop(columns=["is_st"], errors="ignore")
    metadata = metadata.loc[metadata["industry"].str.len().gt(0)].drop_duplicates("symbol", keep="last")
    if metadata.empty:
        return pd.DataFrame(columns=["trade_date", "industry", *INDUSTRY_CONTEXT_COLUMNS])

    table = _safe_table_name(price_table)
    clause, params = _prefix_filter(symbol_prefixes)
    prefixed_clause = clause.replace("symbol like", "p.symbol like")
    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        connection.register("stock_basic_meta", metadata)
        frame = connection.execute(
            f"""
            select p.trade_date,
                   m.industry,
                   avg(
                       case
                           when p.pct_chg is not null then p.pct_chg / 100.0
                           when p.pre_close is not null and p.pre_close <> 0 then p.close / p.pre_close - 1.0
                           else null
                       end
                   ) as industry_ret_1,
                   avg(
                       case
                           when p.pct_chg is not null and p.pct_chg > 0 then 1.0
                           when p.pct_chg is not null then 0.0
                           when p.pre_close is not null and p.pre_close <> 0 and p.close > p.pre_close then 1.0
                           when p.pre_close is not null and p.pre_close <> 0 then 0.0
                           else null
                       end
                   ) as industry_up_ratio_1,
                   avg(p.turnover_rate) as industry_turnover_1,
                   sum(coalesce(p.amount, 0.0)) as industry_amount_1,
                   count(*) as industry_stock_count
            from {table} p
            join stock_basic_meta m
              on p.symbol = m.symbol
            where {prefixed_clause}
              and p.trade_date >= cast(? as date)
              and p.trade_date <= cast(? as date)
            group by p.trade_date, m.industry
            having count(*) >= ?
            order by p.trade_date, m.industry
            """,
            [
                *params,
                query_start.strftime("%Y-%m-%d"),
                analysis_end.strftime("%Y-%m-%d"),
                int(min_symbols_per_industry),
            ],
        ).fetchdf()
    if frame.empty:
        return pd.DataFrame(columns=["trade_date", "industry", *INDUSTRY_CONTEXT_COLUMNS])
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["industry"] = frame["industry"].fillna("unknown").astype(str)
    for column in [
        "industry_ret_1",
        "industry_up_ratio_1",
        "industry_turnover_1",
        "industry_amount_1",
        "industry_stock_count",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "industry"]).sort_values(["industry", "trade_date"])
    grouped = frame.groupby("industry", group_keys=False)
    frame["industry_ret_1"] = frame["industry_ret_1"].fillna(0.0)
    frame["industry_up_ratio_1"] = frame["industry_up_ratio_1"].fillna(0.5)
    frame["industry_ret_5"] = grouped["industry_ret_1"].transform(lambda s: s.rolling(5, min_periods=2).mean())
    frame["industry_ret_20"] = grouped["industry_ret_1"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    frame["industry_up_ratio_5"] = grouped["industry_up_ratio_1"].transform(lambda s: s.rolling(5, min_periods=2).mean())
    frame["industry_up_ratio_20"] = grouped["industry_up_ratio_1"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    frame["industry_ret_5"] = frame["industry_ret_5"].fillna(frame["industry_ret_1"])
    frame["industry_ret_20"] = frame["industry_ret_20"].fillna(frame["industry_ret_5"])
    frame["industry_up_ratio_5"] = frame["industry_up_ratio_5"].fillna(frame["industry_up_ratio_1"])
    frame["industry_up_ratio_20"] = frame["industry_up_ratio_20"].fillna(frame["industry_up_ratio_5"])
    frame["industry_turnover_1"] = grouped["industry_turnover_1"].transform(lambda s: s.ffill().fillna(0.0))
    frame["industry_turnover_5"] = grouped["industry_turnover_1"].transform(lambda s: s.rolling(5, min_periods=2).mean())
    industry_amount_5 = grouped["industry_amount_1"].transform(lambda s: s.rolling(5, min_periods=2).mean())
    industry_amount_20 = grouped["industry_amount_1"].transform(lambda s: s.rolling(20, min_periods=5).mean()).replace(0, np.nan)
    frame["industry_amount_ratio_5"] = (industry_amount_5 / industry_amount_20 - 1.0).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    frame["industry_turnover_5"] = frame["industry_turnover_5"].fillna(frame["industry_turnover_1"])
    frame["industry_amount_ratio_5"] = frame["industry_amount_ratio_5"].fillna(0.0)

    if isinstance(market_context, pd.DataFrame) and not market_context.empty:
        wanted = ["trade_date", "market_ret_5", "market_ret_20"]
        if "market_amount_ratio_5" in market_context.columns:
            wanted.append("market_amount_ratio_5")
        market = market_context[wanted].copy()
        frame = frame.merge(market, on="trade_date", how="left")
    else:
        frame["market_ret_5"] = 0.0
        frame["market_ret_20"] = 0.0
        frame["market_amount_ratio_5"] = 0.0
    if "market_amount_ratio_5" not in frame.columns:
        frame["market_amount_ratio_5"] = 0.0
    frame["industry_relative_ret_5"] = frame["industry_ret_5"] - frame["market_ret_5"].fillna(0.0)
    frame["industry_relative_ret_20"] = frame["industry_ret_20"] - frame["market_ret_20"].fillna(0.0)
    frame["industry_rank_ret_5"] = frame.groupby("trade_date")["industry_ret_5"].rank(pct=True).fillna(0.5)
    frame["industry_rank_up_ratio_5"] = frame.groupby("trade_date")["industry_up_ratio_5"].rank(pct=True).fillna(0.5)
    frame["industry_rank_amount_ratio_5"] = (
        frame.groupby("trade_date")["industry_amount_ratio_5"].rank(pct=True).fillna(0.5)
    )
    for column in INDUSTRY_CONTEXT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        frame[column] = frame[column].fillna(_context_default(column))
    return frame[["trade_date", "industry", *INDUSTRY_CONTEXT_COLUMNS]].copy()


def build_segment_context_frame(
    duckdb_path: Path | str,
    *,
    stock_basic: pd.DataFrame,
    market_context: pd.DataFrame | None,
    price_table: str = DEFAULT_PRICE_TABLE,
    symbol_prefixes: Sequence[str] = DEFAULT_SYMBOL_PREFIXES,
    query_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
    min_symbols_per_segment: int = 20,
    exclude_st: bool = True,
) -> pd.DataFrame:
    if not isinstance(stock_basic, pd.DataFrame) or stock_basic.empty:
        return pd.DataFrame(columns=["trade_date", "market_segment", *SEGMENT_CONTEXT_COLUMNS])
    metadata_columns = ["symbol"]
    if "market_segment" in stock_basic.columns:
        metadata_columns.append("market_segment")
    if "is_st" in stock_basic.columns:
        metadata_columns.append("is_st")
    metadata = stock_basic[metadata_columns].copy()
    metadata["symbol"] = metadata["symbol"].astype(str).str.zfill(6)
    if "market_segment" not in metadata.columns:
        metadata["market_segment"] = metadata["symbol"].map(_symbol_market_segment)
    metadata["market_segment"] = metadata["market_segment"].fillna("other").astype(str)
    if exclude_st and "is_st" in metadata.columns:
        metadata = metadata.loc[~metadata["is_st"].astype(bool)].copy()
    metadata = metadata.drop(columns=["is_st"], errors="ignore")
    metadata = metadata.loc[metadata["market_segment"].str.len().gt(0)].drop_duplicates("symbol", keep="last")
    if metadata.empty:
        return pd.DataFrame(columns=["trade_date", "market_segment", *SEGMENT_CONTEXT_COLUMNS])

    table = _safe_table_name(price_table)
    clause, params = _prefix_filter(symbol_prefixes)
    prefixed_clause = clause.replace("symbol like", "p.symbol like")
    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        connection.register("stock_segment_meta", metadata)
        frame = connection.execute(
            f"""
            select p.trade_date,
                   m.market_segment,
                   avg(
                       case
                           when p.pct_chg is not null then p.pct_chg / 100.0
                           when p.pre_close is not null and p.pre_close <> 0 then p.close / p.pre_close - 1.0
                           else null
                       end
                   ) as segment_ret_1,
                   avg(
                       case
                           when p.pct_chg is not null and p.pct_chg > 0 then 1.0
                           when p.pct_chg is not null then 0.0
                           when p.pre_close is not null and p.pre_close <> 0 and p.close > p.pre_close then 1.0
                           when p.pre_close is not null and p.pre_close <> 0 then 0.0
                           else null
                       end
                   ) as segment_up_ratio_1,
                   avg(p.turnover_rate) as segment_turnover_1,
                   sum(coalesce(p.amount, 0.0)) as segment_amount_1,
                   count(*) as segment_stock_count
            from {table} p
            join stock_segment_meta m
              on p.symbol = m.symbol
            where {prefixed_clause}
              and p.trade_date >= cast(? as date)
              and p.trade_date <= cast(? as date)
            group by p.trade_date, m.market_segment
            having count(*) >= ?
            order by p.trade_date, m.market_segment
            """,
            [
                *params,
                query_start.strftime("%Y-%m-%d"),
                analysis_end.strftime("%Y-%m-%d"),
                int(min_symbols_per_segment),
            ],
        ).fetchdf()
    if frame.empty:
        return pd.DataFrame(columns=["trade_date", "market_segment", *SEGMENT_CONTEXT_COLUMNS])
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["market_segment"] = frame["market_segment"].fillna("other").astype(str)
    for column in [
        "segment_ret_1",
        "segment_up_ratio_1",
        "segment_turnover_1",
        "segment_amount_1",
        "segment_stock_count",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "market_segment"]).sort_values(["market_segment", "trade_date"])
    grouped = frame.groupby("market_segment", group_keys=False)
    frame["segment_ret_1"] = frame["segment_ret_1"].fillna(0.0)
    frame["segment_up_ratio_1"] = frame["segment_up_ratio_1"].fillna(0.5)
    frame["segment_turnover_1"] = grouped["segment_turnover_1"].transform(lambda s: s.ffill().fillna(0.0))
    frame["segment_ret_5"] = grouped["segment_ret_1"].transform(lambda s: s.rolling(5, min_periods=2).mean())
    frame["segment_ret_20"] = grouped["segment_ret_1"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    frame["segment_up_ratio_5"] = grouped["segment_up_ratio_1"].transform(lambda s: s.rolling(5, min_periods=2).mean())
    frame["segment_up_ratio_20"] = grouped["segment_up_ratio_1"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    frame["segment_turnover_5"] = grouped["segment_turnover_1"].transform(lambda s: s.rolling(5, min_periods=2).mean())
    segment_amount_5 = grouped["segment_amount_1"].transform(lambda s: s.rolling(5, min_periods=2).mean())
    segment_amount_20 = grouped["segment_amount_1"].transform(lambda s: s.rolling(20, min_periods=5).mean()).replace(0, np.nan)
    frame["segment_amount_ratio_5"] = (segment_amount_5 / segment_amount_20 - 1.0).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    frame["segment_ret_5"] = frame["segment_ret_5"].fillna(frame["segment_ret_1"])
    frame["segment_ret_20"] = frame["segment_ret_20"].fillna(frame["segment_ret_5"])
    frame["segment_up_ratio_5"] = frame["segment_up_ratio_5"].fillna(frame["segment_up_ratio_1"])
    frame["segment_up_ratio_20"] = frame["segment_up_ratio_20"].fillna(frame["segment_up_ratio_5"])
    frame["segment_turnover_5"] = frame["segment_turnover_5"].fillna(frame["segment_turnover_1"])
    frame["segment_amount_ratio_5"] = frame["segment_amount_ratio_5"].fillna(0.0)

    if isinstance(market_context, pd.DataFrame) and not market_context.empty:
        market = market_context[["trade_date", "market_ret_5", "market_ret_20"]].copy()
        frame = frame.merge(market, on="trade_date", how="left")
    else:
        frame["market_ret_5"] = 0.0
        frame["market_ret_20"] = 0.0
    frame["segment_relative_ret_5"] = frame["segment_ret_5"] - frame["market_ret_5"].fillna(0.0)
    frame["segment_relative_ret_20"] = frame["segment_ret_20"] - frame["market_ret_20"].fillna(0.0)
    frame["segment_rank_ret_5"] = frame.groupby("trade_date")["segment_ret_5"].rank(pct=True).fillna(0.5)
    frame["segment_rank_activity_5"] = (
        frame.groupby("trade_date")["segment_amount_ratio_5"].rank(pct=True).fillna(0.5)
    )
    for column in SEGMENT_CONTEXT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        frame[column] = frame[column].fillna(_context_default(column))
    return frame[["trade_date", "market_segment", *SEGMENT_CONTEXT_COLUMNS]].copy()


def _build_symbol_factor_frame(
    daily_raw: pd.DataFrame,
    *,
    analysis_start: pd.Timestamp,
    analysis_end: pd.Timestamp,
    stock_metadata: dict[str, dict[str, object]] | None = None,
    market_context: pd.DataFrame | None = None,
    industry_context: pd.DataFrame | None = None,
    segment_context: pd.DataFrame | None = None,
    include_target: bool = True,
) -> pd.DataFrame:
    if daily_raw.empty:
        return pd.DataFrame()
    daily = daily_raw.copy().sort_values("trade_date")
    daily["date"] = pd.to_datetime(daily["trade_date"], errors="coerce")
    close = pd.to_numeric(daily.get("close"), errors="coerce")
    open_ = pd.to_numeric(daily.get("open"), errors="coerce").fillna(close)
    high = pd.to_numeric(daily.get("high"), errors="coerce")
    low = pd.to_numeric(daily.get("low"), errors="coerce")
    daily["open"] = open_
    daily["high"] = pd.concat([high, open_, close], axis=1).max(axis=1).fillna(close)
    daily["low"] = pd.concat([low, open_, close], axis=1).min(axis=1).fillna(close)
    daily["close"] = close
    raw_turnover = pd.to_numeric(daily.get("turnover_rate"), errors="coerce")
    volume = pd.to_numeric(daily.get("volume"), errors="coerce")
    if volume.notna().sum() < 30:
        if raw_turnover.notna().sum() >= 30:
            volume = raw_turnover.copy()
        else:
            volume = close.pct_change().abs().rolling(5, min_periods=1).mean().fillna(0.0) + 1.0
    else:
        volume = volume.where(volume.notna(), raw_turnover)
    daily["volume"] = volume.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
    amount = pd.to_numeric(daily.get("amount"), errors="coerce")
    if amount.notna().sum() < 30:
        amount = daily["volume"] * close
    else:
        amount = amount.where(amount.notna(), daily["volume"] * close)
    daily["amount"] = amount.replace([np.inf, -np.inf], np.nan).ffill().fillna(close)
    turnover = raw_turnover.copy()
    volume_base = volume.rolling(20, min_periods=5).mean().replace(0, np.nan)
    turnover_proxy = volume / volume_base
    if turnover.notna().sum() < 30:
        turnover = turnover_proxy
    else:
        turnover = turnover.where(turnover.notna(), turnover_proxy)
    daily["turnover"] = turnover.replace([np.inf, -np.inf], np.nan)
    daily = daily.dropna(subset=["date", "open", "high", "low", "close"]).copy()
    if len(daily) < 130:
        return pd.DataFrame()
    symbol = str(daily["symbol"].dropna().iloc[-1]).zfill(6)
    metadata = (stock_metadata or {}).get(symbol, {})
    industry = str(metadata.get("industry") or "unknown")
    segment = str(metadata.get("market_segment") or _symbol_market_segment(symbol))
    is_st = bool(metadata.get("is_st", False))
    pct_change = pd.to_numeric(daily.get("pct_chg"), errors="coerce") / 100.0
    prev_close = pd.to_numeric(daily.get("pre_close"), errors="coerce")
    pct_change = pct_change.where(pct_change.notna(), daily["close"] / prev_close.replace(0, np.nan) - 1.0)
    pct_change = pct_change.where(pct_change.notna(), daily["close"].pct_change())
    if is_st:
        limit_threshold = pd.Series(0.048, index=daily.index, dtype=float)
    elif segment == "star":
        limit_threshold = pd.Series(0.196, index=daily.index, dtype=float)
    elif segment == "chinext":
        reform_date = pd.Timestamp("2020-08-24")
        limit_threshold = pd.Series(
            np.where(pd.to_datetime(daily["date"], errors="coerce").ge(reform_date), 0.196, 0.098),
            index=daily.index,
            dtype=float,
        )
    elif segment == "beijing":
        limit_threshold = pd.Series(0.295, index=daily.index, dtype=float)
    else:
        limit_threshold = pd.Series(0.098, index=daily.index, dtype=float)
    limit_up_flag = pct_change.ge(limit_threshold).astype(float)
    limit_down_flag = pct_change.le(-limit_threshold).astype(float)
    near_limit_up_flag = pct_change.ge(limit_threshold * 0.80).astype(float)
    near_limit_down_flag = pct_change.le(-limit_threshold * 0.80).astype(float)
    trade_gap_days = pd.to_datetime(daily["date"], errors="coerce").diff().dt.days.fillna(1).clip(lower=1, upper=30)

    daily = daily.set_index("date", drop=False)
    features = build_daily_features(daily)
    frame = features.copy()
    neutral_feature_values = {
        "upper_shadow_ratio": 0.0,
        "lower_shadow_ratio": 0.0,
        "body_ratio": 0.0,
        "close_position_day": 0.5,
    }
    for column, value in neutral_feature_values.items():
        if column in frame.columns:
            frame[column] = frame[column].fillna(value)
    frame["trade_date"] = pd.to_datetime(daily["date"], errors="coerce").to_numpy()
    frame["symbol"] = symbol
    frame["name"] = str(daily["name"].dropna().iloc[-1]) if "name" in daily.columns and daily["name"].notna().any() else frame["symbol"].iloc[0]
    frame["industry"] = industry
    frame["market_segment"] = segment
    frame["limit_up_flag"] = limit_up_flag.to_numpy()
    frame["limit_down_flag"] = limit_down_flag.to_numpy()
    frame["near_limit_up_flag"] = near_limit_up_flag.to_numpy()
    frame["near_limit_down_flag"] = near_limit_down_flag.to_numpy()
    frame["limit_up_streak_3"] = limit_up_flag.rolling(3, min_periods=1).sum().clip(upper=3).to_numpy()
    frame["limit_down_streak_3"] = limit_down_flag.rolling(3, min_periods=1).sum().clip(upper=3).to_numpy()
    frame["trade_gap_days"] = trade_gap_days.to_numpy()
    frame["recent_resume_flag"] = trade_gap_days.gt(7).astype(float).to_numpy()
    frame["market_main_flag"] = float(segment == "main")
    frame["market_chinext_flag"] = float(segment == "chinext")
    frame["market_star_flag"] = float(segment == "star")
    if include_target:
        future_return = daily["close"].shift(-1) / daily["close"] - 1.0
        frame["future_return_1d"] = future_return.to_numpy()
        frame["target_up_1d"] = (future_return > 0.0).astype("int8").to_numpy()
        required = [*FEATURE_COLUMNS, "future_return_1d", "target_up_1d"]
    else:
        required = FEATURE_COLUMNS
    frame = frame.loc[
        frame["trade_date"].ge(analysis_start) & frame["trade_date"].le(analysis_end)
    ].copy()
    if isinstance(market_context, pd.DataFrame) and not market_context.empty:
        frame = frame.merge(market_context, on="trade_date", how="left")
        for column in MARKET_CONTEXT_COLUMNS:
            if column in frame.columns:
                default = 0.5 if "up_ratio" in column else 0.0
                frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(default)
    industry_slice = _context_slice(industry_context, "industry", industry)
    if _context_available(industry_context):
        if not industry_slice.empty:
            frame = frame.merge(industry_slice, on=["trade_date", "industry"], how="left")
        for column in INDUSTRY_CONTEXT_COLUMNS:
            if column not in frame.columns:
                frame[column] = _context_default(column)
            else:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(_context_default(column))
    segment_slice = _context_slice(segment_context, "market_segment", segment)
    if _context_available(segment_context):
        if not segment_slice.empty:
            frame = frame.merge(segment_slice, on=["trade_date", "market_segment"], how="left")
        for column in SEGMENT_CONTEXT_COLUMNS:
            if column not in frame.columns:
                frame[column] = _context_default(column)
            else:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(_context_default(column))
    for column in MICROSTRUCTURE_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=required)
    if include_target:
        frame["target_up_1d"] = frame["target_up_1d"].astype("int8")
    return frame.reset_index(drop=True)


def iter_factor_batches(
    duckdb_path: Path | str,
    *,
    symbols: Sequence[str],
    date_splits: DateSplits,
    price_table: str = DEFAULT_PRICE_TABLE,
    batch_symbols: int = 160,
    lookback_calendar_days: int = 260,
    stock_metadata: dict[str, dict[str, object]] | None = None,
    market_context: pd.DataFrame | None = None,
    industry_context: pd.DataFrame | None = None,
    segment_context: pd.DataFrame | None = None,
    include_target: bool = True,
) -> Iterator[pd.DataFrame]:
    table = _safe_table_name(price_table)
    analysis_start = pd.Timestamp(date_splits.analysis_start)
    analysis_end = pd.Timestamp(date_splits.analysis_end)
    query_start = analysis_start - pd.Timedelta(days=max(int(lookback_calendar_days), 160))
    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        for symbol_chunk in _iter_chunks(list(symbols), batch_symbols):
            price_frame = _fetch_price_batch(
                connection,
                price_table=table,
                symbols=symbol_chunk,
                query_start=query_start,
                analysis_end=analysis_end,
            )
            if price_frame.empty:
                continue
            feature_frames: list[pd.DataFrame] = []
            for _, group in price_frame.groupby("symbol", sort=False):
                factor_frame = _build_symbol_factor_frame(
                    group,
                    analysis_start=analysis_start,
                    analysis_end=analysis_end,
                    stock_metadata=stock_metadata,
                    market_context=market_context,
                    industry_context=industry_context,
                    segment_context=segment_context,
                    include_target=include_target,
                )
                if not factor_frame.empty:
                    feature_frames.append(factor_frame)
            if feature_frames:
                yield pd.concat(feature_frames, ignore_index=True, sort=False)


def _estimate_training_rows(
    connection,
    *,
    price_table: str,
    symbol_prefixes: Sequence[str],
    analysis_start: str,
    train_end: str,
) -> int:
    clause, params = _prefix_filter(symbol_prefixes)
    frame = connection.execute(
        f"""
        select count(*) as rows
        from {price_table}
        where {clause}
          and trade_date >= cast(? as date)
          and trade_date <= cast(? as date)
        """,
        [*params, analysis_start, train_end],
    ).fetchdf()
    return int(frame.loc[0, "rows"]) if not frame.empty else 0


def collect_factor_screen_sample(
    duckdb_path: Path | str,
    *,
    symbols: Sequence[str],
    date_splits: DateSplits,
    price_table: str = DEFAULT_PRICE_TABLE,
    symbol_prefixes: Sequence[str] = DEFAULT_SYMBOL_PREFIXES,
    sample_limit: int = 500_000,
    batch_symbols: int = 160,
    stock_metadata: dict[str, dict[str, object]] | None = None,
    market_context: pd.DataFrame | None = None,
    industry_context: pd.DataFrame | None = None,
    segment_context: pd.DataFrame | None = None,
    random_state: int = 42,
    progress: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, int, float]:
    table = _safe_table_name(price_table)
    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        estimated_rows = _estimate_training_rows(
            connection,
            price_table=table,
            symbol_prefixes=symbol_prefixes,
            analysis_start=date_splits.analysis_start,
            train_end=date_splits.train_end,
        )
    sample_fraction = min(1.0, (float(sample_limit) / max(float(estimated_rows), 1.0)) * 1.35)
    samples: list[pd.DataFrame] = []
    seen_rows = 0
    train_end = pd.Timestamp(date_splits.train_end)
    total_batches = max(1, math.ceil(len(symbols) / max(int(batch_symbols), 1)))
    for batch_index, batch in enumerate(
        iter_factor_batches(
            duckdb_path,
            symbols=symbols,
            date_splits=date_splits,
            price_table=table,
            batch_symbols=batch_symbols,
            stock_metadata=stock_metadata,
            market_context=market_context,
            industry_context=industry_context,
            segment_context=segment_context,
            include_target=True,
        ),
        start=1,
    ):
        train_batch = batch.loc[pd.to_datetime(batch["trade_date"]).le(train_end)].copy()
        if train_batch.empty:
            continue
        seen_rows += len(train_batch)
        if sample_fraction < 1.0:
            train_batch = train_batch.sample(
                frac=sample_fraction,
                random_state=random_state + batch_index,
            )
        samples.append(train_batch)
        if batch_index == 1 or batch_index % 10 == 0 or batch_index >= total_batches:
            _emit(
                progress,
                f"factor sample batch {batch_index}/{total_batches}: seen={seen_rows:,}, sampled={sum(len(item) for item in samples):,}",
            )
    if not samples:
        return pd.DataFrame(), seen_rows, sample_fraction
    sample = pd.concat(samples, ignore_index=True, sort=False)
    if len(sample) > sample_limit:
        sample = sample.sample(n=int(sample_limit), random_state=random_state).reset_index(drop=True)
    else:
        sample = sample.reset_index(drop=True)
    return sample, seen_rows, sample_fraction


def _feature_matrix(frame: pd.DataFrame, feature_columns: Sequence[str]) -> tuple[pd.DataFrame, pd.Series]:
    columns = list(feature_columns)
    matrix = frame[columns].replace([np.inf, -np.inf], np.nan).copy()
    medians = matrix.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    matrix = matrix.fillna(medians).astype("float64")
    target = pd.to_numeric(frame["target_up_1d"], errors="coerce").fillna(0).astype(int)
    return matrix, target


def _market_state_cluster_importance(
    x_scaled: np.ndarray,
    y: np.ndarray,
    feature_columns: Sequence[str],
    *,
    n_clusters: int,
    random_state: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    if len(y) < max(200, n_clusters * 20) or len(np.unique(y)) < 2:
        return np.zeros(len(feature_columns), dtype=float), pd.DataFrame()
    clusters = max(2, min(int(n_clusters), max(2, len(y) // 100)))
    kmeans = MiniBatchKMeans(
        n_clusters=clusters,
        random_state=random_state,
        batch_size=min(8192, max(256, len(y))),
        n_init="auto",
    )
    labels = kmeans.fit_predict(x_scaled)
    baseline = float(np.mean(y))
    summary_rows: list[dict[str, float | int]] = []
    weighted_effect = np.zeros(len(feature_columns), dtype=float)
    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        if not np.any(mask):
            continue
        up_rate = float(np.mean(y[mask]))
        lift = up_rate - baseline
        size = int(mask.sum())
        means = np.asarray(x_scaled[mask].mean(axis=0), dtype=float)
        weighted_effect += (size / len(y)) * abs(lift) * np.square(means)
        summary_rows.append(
            {
                "market_state_cluster": int(cluster_id),
                "sample_size": size,
                "up_rate": up_rate,
                "up_rate_lift": lift,
            }
        )
    return weighted_effect, pd.DataFrame(summary_rows)


def rank_next_day_factors(
    sample: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    top_n: int = 10,
    feature_cluster_distance: float = 0.35,
    market_state_clusters: int = 8,
    importance_sample_limit: int = 220_000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    if sample.empty:
        raise RuntimeError("No training sample is available for factor ranking.")
    columns = [column for column in feature_columns if column in sample.columns]
    if len(columns) < top_n:
        raise RuntimeError(f"Need at least {top_n} feature columns, got {len(columns)}.")
    usable = sample.dropna(subset=["target_up_1d"]).copy()
    if len(usable) > importance_sample_limit:
        usable = usable.sample(n=int(importance_sample_limit), random_state=random_state)
    x_frame, y_series = _feature_matrix(usable, columns)
    y = y_series.to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        raise RuntimeError("Training sample contains only one target class.")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_frame.to_numpy(dtype=float))

    mutual_info = mutual_info_classif(x_scaled, y, random_state=random_state)

    trees = ExtraTreesClassifier(
        n_estimators=160,
        max_depth=9,
        min_samples_leaf=80,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    trees.fit(x_frame, y)
    tree_importance = trees.feature_importances_

    logistic = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=random_state,
    )
    logistic.fit(x_scaled, y)
    logistic_importance = np.abs(logistic.coef_[0])

    state_importance, state_summary = _market_state_cluster_importance(
        x_scaled,
        y,
        columns,
        n_clusters=market_state_clusters,
        random_state=random_state,
    )

    corr = x_frame.corr(method="spearman").abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    distance = 1.0 - corr.to_numpy(dtype=float)
    np.fill_diagonal(distance, 0.0)
    if len(columns) == 1:
        feature_clusters = np.array([0], dtype=int)
    else:
        try:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=float(feature_cluster_distance),
                metric="precomputed",
                linkage="average",
            )
        except TypeError:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=float(feature_cluster_distance),
                affinity="precomputed",
                linkage="average",
            )
        feature_clusters = clusterer.fit_predict(distance)

    ranking = pd.DataFrame(
        {
            "feature": columns,
            "feature_cluster": feature_clusters,
            "extra_trees_importance": tree_importance,
            "mutual_info": mutual_info,
            "logistic_abs_coef": logistic_importance,
            "market_state_cluster_importance": state_importance,
        }
    )
    ranking["extra_trees_rank_score"] = _normalize_scores(ranking["extra_trees_importance"])
    ranking["mutual_info_rank_score"] = _normalize_scores(ranking["mutual_info"])
    ranking["logistic_rank_score"] = _normalize_scores(ranking["logistic_abs_coef"])
    ranking["market_state_rank_score"] = _normalize_scores(ranking["market_state_cluster_importance"])
    ranking["composite_importance"] = (
        0.40 * ranking["extra_trees_rank_score"]
        + 0.25 * ranking["mutual_info_rank_score"]
        + 0.20 * ranking["logistic_rank_score"]
        + 0.15 * ranking["market_state_rank_score"]
    )
    ranking = ranking.sort_values("composite_importance", ascending=False).reset_index(drop=True)

    cluster_rows: list[dict[str, object]] = []
    for cluster_id, group in ranking.groupby("feature_cluster", sort=False):
        ordered = group.sort_values("composite_importance", ascending=False)
        cluster_rows.append(
            {
                "feature_cluster": int(cluster_id),
                "cluster_size": int(len(group)),
                "best_feature": str(ordered.iloc[0]["feature"]),
                "best_composite_importance": float(ordered.iloc[0]["composite_importance"]),
                "features": ",".join(ordered["feature"].astype(str).tolist()),
            }
        )
    cluster_summary = pd.DataFrame(cluster_rows).sort_values(
        "best_composite_importance",
        ascending=False,
    )

    selected: list[str] = []
    for _, row in cluster_summary.iterrows():
        feature = str(row["best_feature"])
        if feature not in selected:
            selected.append(feature)
        if len(selected) >= top_n:
            break
    if len(selected) < top_n:
        for feature in ranking["feature"].astype(str):
            if feature not in selected:
                selected.append(feature)
            if len(selected) >= top_n:
                break

    ranking["selected_top10"] = ranking["feature"].isin(selected)
    ranking["selected_rank"] = ranking["feature"].map({feature: index + 1 for index, feature in enumerate(selected)})
    return ranking, cluster_summary, state_summary, selected


def _clean_supervised_arrays(
    frame: pd.DataFrame,
    selected_features: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = [*selected_features, "target_up_1d", "future_return_1d"]
    data = frame[required].replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    if data.empty:
        return np.empty((0, len(selected_features))), np.empty(0, dtype=int), np.empty(0, dtype=float)
    x = data[list(selected_features)].to_numpy(dtype=float)
    y = data["target_up_1d"].astype(int).to_numpy()
    returns = data["future_return_1d"].astype(float).to_numpy()
    return x, y, returns


def _filter_date_window(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    mask = pd.Series(True, index=frame.index)
    if start:
        mask &= dates.ge(pd.Timestamp(start))
    if end:
        mask &= dates.le(pd.Timestamp(end))
    return frame.loc[mask].copy()


def _fit_streaming_scaler(
    duckdb_path: Path | str,
    *,
    symbols: Sequence[str],
    date_splits: DateSplits,
    selected_features: Sequence[str],
    price_table: str,
    batch_symbols: int,
    stock_metadata: dict[str, dict[str, object]] | None = None,
    market_context: pd.DataFrame | None = None,
    industry_context: pd.DataFrame | None = None,
    segment_context: pd.DataFrame | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[StandardScaler, dict[int, int], int]:
    scaler = StandardScaler()
    class_counts = {0: 0, 1: 0}
    sample_count = 0
    total_batches = max(1, math.ceil(len(symbols) / max(int(batch_symbols), 1)))
    for batch_index, batch in enumerate(
        iter_factor_batches(
            duckdb_path,
            symbols=symbols,
            date_splits=date_splits,
            price_table=price_table,
            batch_symbols=batch_symbols,
            stock_metadata=stock_metadata,
            market_context=market_context,
            industry_context=industry_context,
            segment_context=segment_context,
            include_target=True,
        ),
        start=1,
    ):
        train_batch = _filter_date_window(batch, None, date_splits.train_end)
        x, y, _ = _clean_supervised_arrays(train_batch, selected_features)
        if len(y) == 0:
            continue
        scaler.partial_fit(x)
        values, counts = np.unique(y, return_counts=True)
        for value, count in zip(values, counts):
            class_counts[int(value)] = class_counts.get(int(value), 0) + int(count)
        sample_count += int(len(y))
        if batch_index == 1 or batch_index % 10 == 0 or batch_index >= total_batches:
            _emit(progress, f"scaler batch {batch_index}/{total_batches}: train_rows={sample_count:,}")
    if sample_count == 0:
        raise RuntimeError("No training rows available for selected factors.")
    return scaler, class_counts, sample_count


def _fit_streaming_classifier(
    duckdb_path: Path | str,
    *,
    symbols: Sequence[str],
    date_splits: DateSplits,
    selected_features: Sequence[str],
    price_table: str,
    batch_symbols: int,
    stock_metadata: dict[str, dict[str, object]] | None,
    market_context: pd.DataFrame | None,
    industry_context: pd.DataFrame | None,
    segment_context: pd.DataFrame | None,
    scaler: StandardScaler,
    class_counts: dict[int, int],
    train_epochs: int,
    random_state: int,
    progress: ProgressCallback | None = None,
) -> SGDClassifier:
    total = max(sum(class_counts.values()), 1)
    class_weight = {
        0: total / max(2.0 * class_counts.get(0, 1), 1.0),
        1: total / max(2.0 * class_counts.get(1, 1), 1.0),
    }
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=1e-5,
        l1_ratio=0.15,
        class_weight=class_weight,
        average=True,
        random_state=random_state,
    )
    classes = np.array([0, 1], dtype=int)
    initialized = False
    total_batches = max(1, math.ceil(len(symbols) / max(int(batch_symbols), 1)))
    for epoch in range(max(int(train_epochs), 1)):
        trained_rows = 0
        for batch_index, batch in enumerate(
            iter_factor_batches(
                duckdb_path,
                symbols=symbols,
                date_splits=date_splits,
                price_table=price_table,
                batch_symbols=batch_symbols,
                stock_metadata=stock_metadata,
                market_context=market_context,
                industry_context=industry_context,
                segment_context=segment_context,
                include_target=True,
            ),
            start=1,
        ):
            train_batch = _filter_date_window(batch, None, date_splits.train_end)
            x, y, _ = _clean_supervised_arrays(train_batch, selected_features)
            if len(y) == 0:
                continue
            order = np.random.default_rng(random_state + epoch * 100_000 + batch_index).permutation(len(y))
            x_scaled = scaler.transform(x[order])
            classifier.partial_fit(x_scaled, y[order], classes=classes)
            initialized = True
            trained_rows += int(len(y))
            if batch_index == 1 or batch_index % 10 == 0 or batch_index >= total_batches:
                _emit(
                    progress,
                    f"classifier epoch {epoch + 1}/{max(int(train_epochs), 1)} batch {batch_index}/{total_batches}: train_rows={trained_rows:,}",
                )
    if not initialized:
        raise RuntimeError("Streaming classifier was not initialized.")
    return classifier


def _fit_sample_boost_model(
    training_sample: pd.DataFrame | None,
    selected_features: Sequence[str],
    *,
    max_rows: int,
    random_state: int,
    progress: ProgressCallback | None = None,
) -> HistGradientBoostingClassifier | None:
    if training_sample is None or training_sample.empty:
        return None
    sample = training_sample
    if len(sample) > max_rows:
        sample = sample.sample(n=int(max_rows), random_state=random_state)
    x, y, _ = _clean_supervised_arrays(sample, selected_features)
    if len(y) < 1_000 or len(np.unique(y)) < 2:
        return None
    _emit(progress, f"fit nonlinear sample booster on {len(y):,} rows")
    model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.045,
        max_leaf_nodes=31,
        min_samples_leaf=180,
        l2_regularization=0.05,
        random_state=random_state,
    )
    model.fit(x, y)
    return model


def _predict_probability(
    x: np.ndarray,
    *,
    scaler: StandardScaler,
    classifier: SGDClassifier,
    booster: HistGradientBoostingClassifier | None = None,
    booster_weight: float = 0.55,
    probability_flipped: bool = False,
) -> np.ndarray:
    linear_prob = classifier.predict_proba(scaler.transform(x))[:, 1]
    if booster is None:
        probability = linear_prob
    else:
        boost_prob = booster.predict_proba(x)[:, 1]
        weight = float(max(0.0, min(1.0, booster_weight)))
        probability = (1.0 - weight) * linear_prob + weight * boost_prob
    if probability_flipped:
        probability = 1.0 - probability
    return np.clip(probability, 1e-6, 1.0 - 1e-6)


def _select_accuracy_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    min_positive_prediction_rate: float = 0.02,
) -> dict[str, float]:
    if len(y_true) == 0:
        return {"threshold": 0.5, "accuracy": float("nan"), "positive_prediction_rate": 0.0}
    min_rate = float(max(0.0, min(min_positive_prediction_rate, 0.49)))
    quantiles = np.linspace(0.01, 0.99, 99)
    candidates = np.unique(np.concatenate([np.quantile(y_prob, quantiles), np.array([0.5])]))
    best: dict[str, float] | None = None
    fallback: dict[str, float] | None = None
    for threshold in candidates:
        y_pred = (y_prob >= float(threshold)).astype(int)
        positive_rate = float(np.mean(y_pred))
        accuracy = float(accuracy_score(y_true, y_pred))
        record = {
            "threshold": float(threshold),
            "accuracy": accuracy,
            "positive_prediction_rate": positive_rate,
        }
        if fallback is None or (accuracy, -abs(float(threshold) - 0.5)) > (
            fallback["accuracy"],
            -abs(fallback["threshold"] - 0.5),
        ):
            fallback = record
        if min_rate <= positive_rate <= (1.0 - min_rate):
            if best is None or (accuracy, -abs(float(threshold) - 0.5)) > (
                best["accuracy"],
                -abs(best["threshold"] - 0.5),
            ):
                best = record
    return best or fallback or {"threshold": 0.5, "accuracy": float("nan"), "positive_prediction_rate": 0.0}


def _classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    returns: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    if len(y_true) == 0:
        return {
            "sample_size": 0.0,
            "positive_rate": 0.0,
            "decision_threshold": float(threshold),
            "roc_auc": float("nan"),
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "brier": float("nan"),
            "avg_forward_return": float("nan"),
            "top_decile_return": float("nan"),
            "bottom_decile_return": float("nan"),
            "top_decile_up_rate": float("nan"),
        }
    threshold = float(threshold)
    y_pred = (y_prob >= threshold).astype(int)
    positive_rate = float(np.mean(y_true))
    always_down_accuracy = float(1.0 - positive_rate)
    always_up_accuracy = positive_rate
    majority_accuracy = float(max(always_down_accuracy, always_up_accuracy))
    metrics = {
        "sample_size": float(len(y_true)),
        "positive_rate": positive_rate,
        "decision_threshold": threshold,
        "positive_prediction_rate": float(np.mean(y_pred)),
        "always_down_accuracy": always_down_accuracy,
        "always_up_accuracy": always_up_accuracy,
        "majority_class_accuracy": majority_accuracy,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "avg_forward_return": float(np.mean(returns)),
    }
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    order = np.argsort(y_prob)
    bucket = max(1, int(math.ceil(len(y_true) * 0.10)))
    top = order[-bucket:]
    bottom = order[:bucket]
    metrics["top_decile_return"] = float(np.mean(returns[top]))
    metrics["bottom_decile_return"] = float(np.mean(returns[bottom]))
    metrics["top_decile_up_rate"] = float(np.mean(y_true[top]))
    metrics["probability_spread_top_bottom"] = float(np.mean(y_prob[top]) - np.mean(y_prob[bottom]))
    metrics["accuracy_lift_vs_majority"] = float(metrics["accuracy"] - majority_accuracy)
    return metrics


def _evaluate_streaming_model(
    duckdb_path: Path | str,
    *,
    symbols: Sequence[str],
    date_splits: DateSplits,
    selected_features: Sequence[str],
    price_table: str,
    batch_symbols: int,
    stock_metadata: dict[str, dict[str, object]] | None,
    market_context: pd.DataFrame | None,
    industry_context: pd.DataFrame | None,
    segment_context: pd.DataFrame | None,
    scaler: StandardScaler,
    classifier: SGDClassifier,
    booster: HistGradientBoostingClassifier | None,
    booster_weight: float,
    probability_flipped: bool,
    start: str,
    end: str,
    label: str,
    threshold: float = 0.5,
    progress: ProgressCallback | None = None,
    return_predictions: bool = False,
) -> dict[str, float] | tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    y_parts: list[np.ndarray] = []
    prob_parts: list[np.ndarray] = []
    return_parts: list[np.ndarray] = []
    total_batches = max(1, math.ceil(len(symbols) / max(int(batch_symbols), 1)))
    seen_rows = 0
    for batch_index, batch in enumerate(
        iter_factor_batches(
            duckdb_path,
            symbols=symbols,
            date_splits=date_splits,
            price_table=price_table,
            batch_symbols=batch_symbols,
            stock_metadata=stock_metadata,
            market_context=market_context,
            industry_context=industry_context,
            segment_context=segment_context,
            include_target=True,
        ),
        start=1,
    ):
        target_batch = _filter_date_window(batch, start, end)
        x, y, returns = _clean_supervised_arrays(target_batch, selected_features)
        if len(y) == 0:
            continue
        prob = _predict_probability(
            x,
            scaler=scaler,
            classifier=classifier,
            booster=booster,
            booster_weight=booster_weight,
            probability_flipped=probability_flipped,
        )
        y_parts.append(y)
        prob_parts.append(prob)
        return_parts.append(returns)
        seen_rows += int(len(y))
        if batch_index == 1 or batch_index % 10 == 0 or batch_index >= total_batches:
            _emit(progress, f"{label} eval batch {batch_index}/{total_batches}: rows={seen_rows:,}")
    if not y_parts:
        metrics = _classification_metrics(
            np.empty(0, dtype=int),
            np.empty(0),
            np.empty(0),
            threshold=threshold,
        )
        if return_predictions:
            return metrics, np.empty(0, dtype=int), np.empty(0), np.empty(0)
        return metrics
    y_all = np.concatenate(y_parts)
    prob_all = np.concatenate(prob_parts)
    returns_all = np.concatenate(return_parts)
    metrics = _classification_metrics(y_all, prob_all, returns_all, threshold=threshold)
    if return_predictions:
        return metrics, y_all, prob_all, returns_all
    return metrics


def train_selected_factor_model(
    duckdb_path: Path | str,
    *,
    symbols: Sequence[str],
    date_splits: DateSplits,
    selected_features: Sequence[str],
    training_sample: pd.DataFrame | None = None,
    price_table: str = DEFAULT_PRICE_TABLE,
    batch_symbols: int = 160,
    stock_metadata: dict[str, dict[str, object]] | None = None,
    market_context: pd.DataFrame | None = None,
    industry_context: pd.DataFrame | None = None,
    segment_context: pd.DataFrame | None = None,
    train_epochs: int = 2,
    sample_model_max_rows: int = 500_000,
    booster_weight: float = 0.55,
    min_positive_prediction_rate: float = 0.02,
    random_state: int = 42,
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    _emit(progress, "fit scaler on selected top factors")
    scaler, class_counts, train_sample_size = _fit_streaming_scaler(
        duckdb_path,
        symbols=symbols,
        date_splits=date_splits,
        selected_features=selected_features,
        price_table=price_table,
        batch_symbols=batch_symbols,
        stock_metadata=stock_metadata,
        market_context=market_context,
        industry_context=industry_context,
        segment_context=segment_context,
        progress=progress,
    )
    _emit(progress, "fit streaming next-day classifier")
    classifier = _fit_streaming_classifier(
        duckdb_path,
        symbols=symbols,
        date_splits=date_splits,
        selected_features=selected_features,
        price_table=price_table,
        batch_symbols=batch_symbols,
        stock_metadata=stock_metadata,
        market_context=market_context,
        industry_context=industry_context,
        segment_context=segment_context,
        scaler=scaler,
        class_counts=class_counts,
        train_epochs=train_epochs,
        random_state=random_state,
        progress=progress,
    )
    booster = _fit_sample_boost_model(
        training_sample,
        selected_features,
        max_rows=sample_model_max_rows,
        random_state=random_state,
        progress=progress,
    )
    _emit(progress, "evaluate validation window")
    raw_validation_eval = _evaluate_streaming_model(
        duckdb_path,
        symbols=symbols,
        date_splits=date_splits,
        selected_features=selected_features,
        price_table=price_table,
        batch_symbols=batch_symbols,
        stock_metadata=stock_metadata,
        market_context=market_context,
        industry_context=industry_context,
        segment_context=segment_context,
        scaler=scaler,
        classifier=classifier,
        booster=booster,
        booster_weight=booster_weight,
        probability_flipped=False,
        start=date_splits.validation_start,
        end=date_splits.validation_end,
        label="validation",
        progress=progress,
        return_predictions=True,
    )
    raw_validation_metrics, validation_y, validation_prob, validation_returns = raw_validation_eval
    probability_flipped = bool(raw_validation_metrics.get("roc_auc", 0.5) < 0.5)
    if probability_flipped:
        _emit(progress, "validation AUC is below 0.5; flip probability orientation")
        validation_prob = 1.0 - validation_prob
    threshold_selection = _select_accuracy_threshold(
        validation_y,
        validation_prob,
        min_positive_prediction_rate=min_positive_prediction_rate,
    )
    decision_threshold = float(threshold_selection["threshold"])
    validation_metrics = _classification_metrics(
        validation_y,
        validation_prob,
        validation_returns,
        threshold=decision_threshold,
    )
    validation_metrics["threshold_selected_on_validation"] = 1.0
    validation_metrics["threshold_selection_accuracy"] = float(threshold_selection["accuracy"])
    validation_metrics["threshold_selection_positive_prediction_rate"] = float(
        threshold_selection["positive_prediction_rate"]
    )
    _emit(progress, "evaluate test window")
    test_metrics = _evaluate_streaming_model(
        duckdb_path,
        symbols=symbols,
        date_splits=date_splits,
        selected_features=selected_features,
        price_table=price_table,
        batch_symbols=batch_symbols,
        stock_metadata=stock_metadata,
        market_context=market_context,
        industry_context=industry_context,
        segment_context=segment_context,
        scaler=scaler,
        classifier=classifier,
        booster=booster,
        booster_weight=booster_weight,
        probability_flipped=probability_flipped,
        threshold=decision_threshold,
        start=date_splits.test_start,
        end=date_splits.test_end,
        label="test",
        progress=progress,
    )
    bundle = {
        "model_version": MODEL_VERSION,
        "target": "next_trading_day_close_to_close_up",
        "selected_features": list(selected_features),
        "date_splits": asdict(date_splits),
        "scaler": scaler,
        "classifier": classifier,
        "booster": booster,
        "booster_weight": float(booster_weight) if booster is not None else 0.0,
        "probability_flipped": probability_flipped,
        "decision_threshold": decision_threshold,
        "threshold_selection": threshold_selection,
        "min_positive_prediction_rate": float(min_positive_prediction_rate),
        "class_counts": class_counts,
        "train_sample_size": int(train_sample_size),
    }
    metrics = {
        "train_sample_size": int(train_sample_size),
        "train_class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "sample_booster_rows": int(min(len(training_sample), sample_model_max_rows)) if training_sample is not None else 0,
        "booster_weight": float(booster_weight) if booster is not None else 0.0,
        "probability_flipped": probability_flipped,
        "decision_threshold": decision_threshold,
        "threshold_selection": threshold_selection,
        "min_positive_prediction_rate": float(min_positive_prediction_rate),
        "raw_validation": raw_validation_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    return bundle, metrics


def _write_report(
    path: Path,
    *,
    selected_factors: Sequence[str],
    date_splits: DateSplits,
    data_summary: dict[str, object],
    metrics: dict[str, object],
) -> None:
    validation = metrics.get("validation", {})
    test = metrics.get("test", {})
    lines = [
        "# Next-day A-share factor model",
        "",
        f"- Data window: {date_splits.analysis_start} to {date_splits.analysis_end}",
        f"- Symbols: {data_summary.get('eligible_symbols')} eligible / {data_summary.get('coverage_symbols')} covered",
        f"- Target: next trading day close-to-close return > 0",
        f"- Train end: {date_splits.train_end}",
        f"- Validation: {date_splits.validation_start} to {date_splits.validation_end}",
        f"- Test: {date_splits.test_start} to {date_splits.test_end}",
        "",
        "## Selected top 10 factors",
        "",
    ]
    lines.extend(f"{index}. {feature}" for index, feature in enumerate(selected_factors, start=1))
    lines.extend(
        [
            "",
            "## Validation metrics",
            "",
            f"- ROC AUC: {validation.get('roc_auc')}",
            f"- Accuracy: {validation.get('accuracy')}",
            f"- Majority-class accuracy: {validation.get('majority_class_accuracy')}",
            f"- Decision threshold: {validation.get('decision_threshold')}",
            f"- Positive prediction rate: {validation.get('positive_prediction_rate')}",
            f"- Precision: {validation.get('precision')}",
            f"- Recall: {validation.get('recall')}",
            f"- Top decile next-day return: {validation.get('top_decile_return')}",
            "",
            "## Test metrics",
            "",
            f"- ROC AUC: {test.get('roc_auc')}",
            f"- Accuracy: {test.get('accuracy')}",
            f"- Majority-class accuracy: {test.get('majority_class_accuracy')}",
            f"- Decision threshold: {test.get('decision_threshold')}",
            f"- Positive prediction rate: {test.get('positive_prediction_rate')}",
            f"- Precision: {test.get('precision')}",
            f"- Recall: {test.get('recall')}",
            f"- Top decile next-day return: {test.get('top_decile_return')}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_next_day_factor_pipeline(
    *,
    duckdb_path: Path | str = DEFAULT_DUCKDB_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    price_table: str = DEFAULT_PRICE_TABLE,
    symbol_prefixes: Sequence[str] = DEFAULT_SYMBOL_PREFIXES,
    start_date: str | None = None,
    end_date: str | None = None,
    top_n: int = 10,
    analysis_sample_limit: int = 500_000,
    importance_sample_limit: int = 220_000,
    batch_symbols: int = 160,
    min_history_rows: int = 180,
    symbol_limit: int | None = None,
    train_epochs: int = 2,
    sample_model_max_rows: int = 500_000,
    booster_weight: float = 0.55,
    include_market_context: bool = True,
    include_industry_context: bool = True,
    include_segment_context: bool = True,
    include_microstructure: bool = True,
    exclude_st: bool = True,
    stock_basic_path: Path | str | None = DEFAULT_STOCK_BASIC_PATH,
    min_positive_prediction_rate: float = 0.02,
    feature_cluster_distance: float = 0.35,
    market_state_clusters: int = 8,
    random_state: int = 42,
    progress: ProgressCallback | None = None,
) -> PipelineResult:
    duckdb_path = Path(duckdb_path)
    output_dir = Path(output_dir)
    table = _safe_table_name(price_table)
    output_dir.mkdir(parents=True, exist_ok=True)
    stock_basic = _load_stock_basic_metadata(stock_basic_path)
    stock_metadata = (
        stock_basic.set_index("symbol").to_dict("index")
        if isinstance(stock_basic, pd.DataFrame) and not stock_basic.empty
        else {}
    )
    excluded_symbols = (
        stock_basic.loc[stock_basic["is_st"].astype(bool), "symbol"].astype(str).str.zfill(6).tolist()
        if exclude_st and isinstance(stock_basic, pd.DataFrame) and "is_st" in stock_basic.columns
        else []
    )

    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        date_splits = build_date_splits(
            connection,
            price_table=table,
            symbol_prefixes=symbol_prefixes,
            start_date=start_date,
            end_date=end_date,
        )
        coverage = inspect_duckdb_coverage(
            duckdb_path,
            price_table=table,
            symbol_prefixes=symbol_prefixes,
        )
        symbols = _fetch_symbols(
            connection,
            price_table=table,
            symbol_prefixes=symbol_prefixes,
            analysis_start=pd.Timestamp(date_splits.analysis_start),
            analysis_end=pd.Timestamp(date_splits.analysis_end),
            min_history_rows=min_history_rows,
            exclude_symbols=excluded_symbols,
            symbol_limit=symbol_limit,
        )
    if not symbols:
        raise RuntimeError("No eligible A-share symbols found for the requested window.")
    _emit(
        progress,
        f"window {date_splits.analysis_start} -> {date_splits.analysis_end}; eligible symbols={len(symbols):,}",
    )
    market_context = None
    candidate_feature_columns = list(FEATURE_COLUMNS)
    if include_microstructure:
        candidate_feature_columns = [*candidate_feature_columns, *MICROSTRUCTURE_COLUMNS]
    if include_market_context:
        _emit(progress, "build market context features")
        market_context = build_market_context_frame(
            duckdb_path,
            price_table=table,
            symbol_prefixes=symbol_prefixes,
            query_start=pd.Timestamp(date_splits.analysis_start) - pd.Timedelta(days=260),
            analysis_end=pd.Timestamp(date_splits.analysis_end),
        )
        candidate_feature_columns = [*candidate_feature_columns, *MARKET_CONTEXT_COLUMNS]
    industry_context = None
    if include_industry_context and stock_metadata:
        _emit(progress, "build industry rotation features")
        industry_context = build_industry_context_frame(
            duckdb_path,
            stock_basic=stock_basic,
            market_context=market_context,
            price_table=table,
            symbol_prefixes=symbol_prefixes,
            query_start=pd.Timestamp(date_splits.analysis_start) - pd.Timedelta(days=260),
            analysis_end=pd.Timestamp(date_splits.analysis_end),
            exclude_st=exclude_st,
        )
        if isinstance(industry_context, pd.DataFrame) and not industry_context.empty:
            candidate_feature_columns = [*candidate_feature_columns, *INDUSTRY_CONTEXT_COLUMNS]
    segment_context = None
    if include_segment_context and stock_metadata:
        _emit(progress, "build market-segment rotation features")
        segment_context = build_segment_context_frame(
            duckdb_path,
            stock_basic=stock_basic,
            market_context=market_context,
            price_table=table,
            symbol_prefixes=symbol_prefixes,
            query_start=pd.Timestamp(date_splits.analysis_start) - pd.Timedelta(days=260),
            analysis_end=pd.Timestamp(date_splits.analysis_end),
            exclude_st=exclude_st,
        )
        if isinstance(segment_context, pd.DataFrame) and not segment_context.empty:
            candidate_feature_columns = [*candidate_feature_columns, *SEGMENT_CONTEXT_COLUMNS]
    industry_context_for_merge = _split_context_by_key(industry_context, "industry")
    segment_context_for_merge = _split_context_by_key(segment_context, "market_segment")

    _emit(progress, "collect factor-screening sample")
    sample, train_rows_seen, sample_fraction = collect_factor_screen_sample(
        duckdb_path,
        symbols=symbols,
        date_splits=date_splits,
        price_table=table,
        symbol_prefixes=symbol_prefixes,
        sample_limit=analysis_sample_limit,
        batch_symbols=batch_symbols,
        stock_metadata=stock_metadata,
        market_context=market_context,
        industry_context=industry_context_for_merge,
        segment_context=segment_context_for_merge,
        random_state=random_state,
        progress=progress,
    )
    _emit(progress, f"rank factors with clustering; sample rows={len(sample):,}")
    ranking, cluster_summary, state_summary, selected_factors = rank_next_day_factors(
        sample,
        feature_columns=candidate_feature_columns,
        top_n=top_n,
        feature_cluster_distance=feature_cluster_distance,
        market_state_clusters=market_state_clusters,
        importance_sample_limit=importance_sample_limit,
        random_state=random_state,
    )
    _emit(progress, "selected top factors: " + ", ".join(selected_factors))
    bundle, metrics = train_selected_factor_model(
        duckdb_path,
        symbols=symbols,
        date_splits=date_splits,
        selected_features=selected_factors,
        training_sample=sample,
        price_table=table,
        batch_symbols=batch_symbols,
        stock_metadata=stock_metadata,
        market_context=market_context,
        industry_context=industry_context_for_merge,
        segment_context=segment_context_for_merge,
        train_epochs=train_epochs,
        sample_model_max_rows=sample_model_max_rows,
        booster_weight=booster_weight,
        min_positive_prediction_rate=min_positive_prediction_rate,
        random_state=random_state,
        progress=progress,
    )
    bundle["feature_ranking"] = ranking
    bundle["feature_cluster_summary"] = cluster_summary
    bundle["market_state_cluster_summary"] = state_summary

    feature_ranking_path = output_dir / "feature_ranking.csv"
    cluster_summary_path = output_dir / "feature_cluster_summary.csv"
    state_summary_path = output_dir / "market_state_cluster_summary.csv"
    selected_factors_path = output_dir / "selected_top10_factors.csv"
    metrics_path = output_dir / "model_metrics.json"
    model_path = output_dir / "next_day_top10_factor_model.pkl"
    report_path = output_dir / "training_report.md"

    ranking.to_csv(feature_ranking_path, index=False, encoding="utf-8")
    cluster_summary.to_csv(cluster_summary_path, index=False, encoding="utf-8")
    state_summary.to_csv(state_summary_path, index=False, encoding="utf-8")
    ranking.loc[ranking["selected_top10"]].sort_values("selected_rank").to_csv(
        selected_factors_path,
        index=False,
        encoding="utf-8",
    )
    data_summary = {
        "coverage_rows": coverage["rows"],
        "coverage_symbols": coverage["symbols"],
        "coverage_trade_days": coverage["trade_days"],
        "eligible_symbols": len(symbols),
        "factor_screen_train_rows_seen": int(train_rows_seen),
        "factor_screen_sample_rows": int(len(sample)),
        "factor_screen_sample_fraction": float(sample_fraction),
        "include_market_context": bool(include_market_context),
        "include_industry_context": bool(include_industry_context and isinstance(industry_context, pd.DataFrame) and not industry_context.empty),
        "include_segment_context": bool(include_segment_context and isinstance(segment_context, pd.DataFrame) and not segment_context.empty),
        "include_microstructure": bool(include_microstructure),
        "exclude_st": bool(exclude_st),
        "excluded_st_symbols": int(len(excluded_symbols)),
        "stock_basic_rows": int(len(stock_basic)) if isinstance(stock_basic, pd.DataFrame) else 0,
        "candidate_feature_count": int(len(candidate_feature_columns)),
        "min_positive_prediction_rate": float(min_positive_prediction_rate),
        "symbol_prefixes": list(symbol_prefixes),
        "price_table": table,
        "duckdb_path": str(duckdb_path),
    }
    metrics_payload = {
        "model_version": MODEL_VERSION,
        "target": "next_trading_day_close_to_close_up",
        "date_splits": asdict(date_splits),
        "data_summary": data_summary,
        "selected_factors": selected_factors,
        "metrics": metrics,
        "artifacts": {
            "feature_ranking": str(feature_ranking_path),
            "feature_cluster_summary": str(cluster_summary_path),
            "market_state_cluster_summary": str(state_summary_path),
            "selected_top10_factors": str(selected_factors_path),
            "model": str(model_path),
            "report": str(report_path),
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    with model_path.open("wb") as handle:
        pickle.dump(bundle, handle)
    _write_report(
        report_path,
        selected_factors=selected_factors,
        date_splits=date_splits,
        data_summary=data_summary,
        metrics=metrics,
    )
    _emit(progress, f"artifacts written to {output_dir}")
    return PipelineResult(
        output_dir=str(output_dir),
        model_path=str(model_path),
        feature_ranking_path=str(feature_ranking_path),
        cluster_summary_path=str(cluster_summary_path),
        selected_factors_path=str(selected_factors_path),
        metrics_path=str(metrics_path),
        report_path=str(report_path),
        selected_factors=selected_factors,
        metrics=metrics,
        date_splits=asdict(date_splits),
        data_summary=data_summary,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a next-day A-share direction model from clustered top factors.")
    parser.add_argument("--duckdb-path", default=str(DEFAULT_DUCKDB_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--price-table", default=DEFAULT_PRICE_TABLE)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--analysis-sample-limit", type=int, default=500_000)
    parser.add_argument("--importance-sample-limit", type=int, default=220_000)
    parser.add_argument("--batch-symbols", type=int, default=160)
    parser.add_argument("--min-history-rows", type=int, default=180)
    parser.add_argument("--symbol-limit", type=int, default=None)
    parser.add_argument("--train-epochs", type=int, default=2)
    parser.add_argument("--sample-model-max-rows", type=int, default=500_000)
    parser.add_argument("--booster-weight", type=float, default=0.55)
    parser.add_argument("--include-market-context", action="store_true", default=True)
    parser.add_argument("--no-market-context", action="store_true")
    parser.add_argument("--include-industry-context", action="store_true", default=True)
    parser.add_argument("--no-industry-context", action="store_true")
    parser.add_argument("--include-segment-context", action="store_true", default=True)
    parser.add_argument("--no-segment-context", action="store_true")
    parser.add_argument("--no-microstructure", action="store_true")
    parser.add_argument("--include-st", action="store_true")
    parser.add_argument("--stock-basic-path", default=str(DEFAULT_STOCK_BASIC_PATH))
    parser.add_argument("--min-positive-prediction-rate", type=float, default=0.02)
    parser.add_argument("--feature-cluster-distance", type=float, default=0.35)
    parser.add_argument("--market-state-clusters", type=int, default=8)
    parser.add_argument("--symbol-prefixes", default=",".join(DEFAULT_SYMBOL_PREFIXES))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    prefixes = tuple(prefix.strip() for prefix in str(args.symbol_prefixes).split(",") if prefix.strip())
    progress = None if args.quiet else (lambda message: print(f"[next-day] {message}", file=sys.stderr, flush=True))
    result = run_next_day_factor_pipeline(
        duckdb_path=args.duckdb_path,
        output_dir=args.output_dir,
        price_table=args.price_table,
        symbol_prefixes=prefixes,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=args.top_n,
        analysis_sample_limit=args.analysis_sample_limit,
        importance_sample_limit=args.importance_sample_limit,
        batch_symbols=args.batch_symbols,
        min_history_rows=args.min_history_rows,
        symbol_limit=args.symbol_limit,
        train_epochs=args.train_epochs,
        sample_model_max_rows=args.sample_model_max_rows,
        booster_weight=args.booster_weight,
        include_market_context=bool(args.include_market_context and not args.no_market_context),
        include_industry_context=bool(args.include_industry_context and not args.no_industry_context),
        include_segment_context=bool(args.include_segment_context and not args.no_segment_context),
        include_microstructure=not args.no_microstructure,
        exclude_st=not args.include_st,
        stock_basic_path=args.stock_basic_path,
        min_positive_prediction_rate=args.min_positive_prediction_rate,
        feature_cluster_distance=args.feature_cluster_distance,
        market_state_clusters=args.market_state_clusters,
        random_state=args.random_state,
        progress=progress,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
