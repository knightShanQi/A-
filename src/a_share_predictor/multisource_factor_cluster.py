from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import duckdb
import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS
from .next_day_factor_model import (
    DEFAULT_DUCKDB_PATH,
    DEFAULT_PRICE_TABLE,
    DEFAULT_STOCK_BASIC_PATH,
    DEFAULT_SYMBOL_PREFIXES,
    INDUSTRY_CONTEXT_COLUMNS,
    MARKET_CONTEXT_COLUMNS,
    MICROSTRUCTURE_COLUMNS,
    SEGMENT_CONTEXT_COLUMNS,
    DateSplits,
    _fetch_symbols,
    _json_default,
    _load_stock_basic_metadata,
    _safe_table_name,
    build_industry_context_frame,
    build_market_context_frame,
    build_segment_context_frame,
    iter_factor_batches,
    rank_next_day_factors,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".cache" / "one_year_multisource_factor_cluster"
DEFAULT_INTRADAY_TABLE = "a_share_intraday_bars"
DEFAULT_STOCK_FUND_FLOW_TABLE = "a_share_stock_fund_flow"
DEFAULT_CALL_AUCTION_PROXY_TABLE = "a_share_call_auction_proxy"
ProgressCallback = Callable[[str], None]


TRUE_INTRADAY_COLUMNS = [
    "intraday_ret_1m_path",
    "intraday_first30_ret",
    "intraday_tail30_ret",
    "intraday_first30_volume_share",
    "intraday_tail30_volume_share",
    "intraday_opening_volume_ratio",
    "intraday_vwap_gap",
    "intraday_above_vwap_ratio",
    "intraday_realized_volatility",
    "intraday_max_pullback",
    "intraday_close_strength",
    "intraday_high_time_ratio",
    "intraday_low_time_ratio",
    "auction_open_bar_ret",
    "auction_open_bar_volume_share",
    "auction_open_bar_amount_share",
    "auction_open_bar_range_pct",
]

TRUE_FUND_FLOW_FEATURE_COLUMNS = [
    "true_fund_net_inflow",
    "true_fund_main_net_amount",
    "true_fund_main_net_ratio",
    "true_fund_main_net_amount_5",
    "true_fund_main_net_ratio_5",
    "true_fund_main_inflow_streak_5",
]

TRUE_FUND_FLOW_COLUMNS = [
    *TRUE_FUND_FLOW_FEATURE_COLUMNS,
    "true_fund_flow_coverage",
]

DAILY_PROXY_COLUMNS = [
    "auction_open_gap",
    "intraday_close_position_proxy",
    "intraday_body_ratio_proxy",
    "intraday_upper_shadow_proxy",
    "intraday_lower_shadow_proxy",
    "fund_signed_amount_proxy_1",
    "fund_price_volume_confirm_5",
    "fund_amount_ratio_5",
    "fund_turnover_ratio_20",
    "sector_heat_score_5",
    "sector_fund_heat_score_5",
    "segment_heat_score_5",
    "market_trend_score_5",
    "market_risk_appetite_5",
]

MULTISOURCE_FEATURE_COLUMNS = [
    *TRUE_INTRADAY_COLUMNS,
    *TRUE_FUND_FLOW_FEATURE_COLUMNS,
    *DAILY_PROXY_COLUMNS,
]


@dataclass(slots=True)
class MultiSourceClusterResult:
    output_dir: str
    feature_ranking_path: str
    cluster_summary_path: str
    selected_factors_path: str
    coverage_path: str
    report_path: str
    selected_factors: list[str]
    date_splits: dict[str, str]
    data_summary: dict[str, object]


def _emit(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _date_string(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def _time_string(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    return text[:8] if len(text) >= 8 else text


def _prefix_clause(prefixes: Sequence[str], column: str = "symbol") -> tuple[str, list[str]]:
    clean = [str(prefix).strip() for prefix in prefixes if str(prefix).strip()]
    if not clean:
        clean = list(DEFAULT_SYMBOL_PREFIXES)
    clause = " or ".join(f"{column} like ?" for _ in clean)
    return f"({clause})", [f"{prefix}%" for prefix in clean]


def _safe_table_exists(connection, table: str) -> bool:
    safe = _safe_table_name(table)
    rows = connection.execute("show tables").fetchall()
    return safe in {str(row[0]) for row in rows}


def build_one_year_date_splits(
    duckdb_path: Path | str = DEFAULT_DUCKDB_PATH,
    *,
    price_table: str = DEFAULT_PRICE_TABLE,
    symbol_prefixes: Sequence[str] = DEFAULT_SYMBOL_PREFIXES,
    end_date: str | None = None,
    lookback_days: int = 365,
) -> DateSplits:
    table = _safe_table_name(price_table)
    clause, params = _prefix_clause(symbol_prefixes)
    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        if end_date:
            end_row = connection.execute(
                f"""
                select max(trade_date)
                from {table}
                where {clause}
                  and trade_date <= cast(? as date)
                """,
                [*params, end_date],
            ).fetchone()
        else:
            end_row = connection.execute(
                f"""
                select max(trade_date)
                from {table}
                where {clause}
                """,
                params,
            ).fetchone()
    analysis_end = pd.Timestamp(end_row[0]) if end_row and end_row[0] is not None else pd.NaT
    if pd.isna(analysis_end):
        raise RuntimeError("No trade dates are available for one-year clustering.")
    floor = analysis_end - pd.Timedelta(days=max(int(lookback_days), 30))
    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        start_row = connection.execute(
            f"""
            select min(trade_date)
            from {table}
            where {clause}
              and trade_date >= cast(? as date)
              and trade_date <= cast(? as date)
            """,
            [*params, floor.strftime("%Y-%m-%d"), analysis_end.strftime("%Y-%m-%d")],
        ).fetchone()
    analysis_start = pd.Timestamp(start_row[0]) if start_row and start_row[0] is not None else floor
    return DateSplits(
        analysis_start=_date_string(analysis_start),
        analysis_end=_date_string(analysis_end),
        train_end=_date_string(analysis_end),
        validation_start=_date_string(analysis_end),
        validation_end=_date_string(analysis_end),
        test_start=_date_string(analysis_end),
        test_end=_date_string(analysis_end),
    )


def build_intraday_feature_frame(
    duckdb_path: Path | str = DEFAULT_DUCKDB_PATH,
    *,
    intraday_table: str = DEFAULT_INTRADAY_TABLE,
    start_date: str,
    end_date: str,
    interval_minutes: int = 1,
) -> pd.DataFrame:
    table = _safe_table_name(intraday_table)
    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        if not _safe_table_exists(connection, table):
            return pd.DataFrame(columns=["symbol", "trade_date", *TRUE_INTRADAY_COLUMNS, "intraday_bar_count"])
        bars = connection.execute(
            f"""
            select symbol, trade_date, bar_time, open, high, low, close, volume, amount
            from {table}
            where interval_minutes = ?
              and trade_date >= cast(? as date)
              and trade_date <= cast(? as date)
            order by symbol, trade_date, bar_time
            """,
            [int(interval_minutes), start_date, end_date],
        ).fetchdf()
    if bars.empty:
        return pd.DataFrame(columns=["symbol", "trade_date", *TRUE_INTRADAY_COLUMNS, "intraday_bar_count"])
    bars = bars.copy()
    bars["symbol"] = bars["symbol"].astype(str).str.extract(r"(\d{1,6})", expand=False).fillna("").str.zfill(6)
    bars["trade_date"] = pd.to_datetime(bars["trade_date"], errors="coerce")
    bars["bar_time_text"] = bars["bar_time"].astype(str)
    bars["bar_timestamp"] = pd.to_datetime(
        bars["trade_date"].dt.strftime("%Y-%m-%d") + " " + bars["bar_time_text"],
        errors="coerce",
    )
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        bars[column] = pd.to_numeric(bars[column], errors="coerce")
    bars = bars.dropna(subset=["symbol", "trade_date", "bar_timestamp", "close"]).sort_values(
        ["symbol", "trade_date", "bar_timestamp"],
    )
    if bars.empty:
        return pd.DataFrame(columns=["symbol", "trade_date", *TRUE_INTRADAY_COLUMNS, "intraday_bar_count"])

    keys = ["symbol", "trade_date"]
    grouped = bars.groupby(keys, sort=False)
    bars["bar_index"] = grouped.cumcount()
    bars["bar_count"] = grouped["close"].transform("size")
    bars["cum_amount"] = grouped["amount"].cumsum()
    bars["cum_volume"] = grouped["volume"].cumsum()
    bars["vwap"] = bars["cum_amount"] / (bars["cum_volume"] * 100.0).replace(0, np.nan)
    bars["above_vwap"] = bars["close"].ge(bars["vwap"]).astype(float)
    bars["close_cummax"] = grouped["close"].cummax()
    bars["drawdown"] = (1.0 - bars["close"] / bars["close_cummax"].replace(0, np.nan)).clip(lower=0)
    bars["log_return"] = np.log(bars["close"] / grouped["close"].shift(1).replace(0, np.nan))

    summary = grouped.agg(
        first_open=("open", "first"),
        first_close=("close", "first"),
        first_high=("high", "first"),
        first_low=("low", "first"),
        first_volume=("volume", "first"),
        first_amount=("amount", "first"),
        last_close=("close", "last"),
        total_volume=("volume", "sum"),
        total_amount=("amount", "sum"),
        high=("high", "max"),
        low=("low", "min"),
        intraday_bar_count=("close", "size"),
        intraday_above_vwap_ratio=("above_vwap", "mean"),
        intraday_realized_volatility=("log_return", "std"),
        intraday_max_pullback=("drawdown", "max"),
    ).reset_index()

    first30 = bars.loc[bars["bar_timestamp"].dt.time < pd.Timestamp("10:00").time()]
    first30_summary = first30.groupby(keys, sort=False).agg(
        first30_close=("close", "last"),
        first30_volume=("volume", "sum"),
    ).reset_index()
    tail30 = bars.loc[bars["bar_timestamp"].dt.time >= pd.Timestamp("14:30").time()]
    tail30_summary = tail30.groupby(keys, sort=False).agg(
        tail30_open=("open", "first"),
        tail30_close=("close", "last"),
        tail30_volume=("volume", "sum"),
    ).reset_index()

    high_idx = bars.loc[bars.groupby(keys)["high"].idxmax(), [*keys, "bar_index", "bar_count"]].rename(
        columns={"bar_index": "high_bar_index", "bar_count": "high_bar_count"},
    )
    low_idx = bars.loc[bars.groupby(keys)["low"].idxmin(), [*keys, "bar_index", "bar_count"]].rename(
        columns={"bar_index": "low_bar_index", "bar_count": "low_bar_count"},
    )

    result = summary.merge(first30_summary, on=keys, how="left")
    result = result.merge(tail30_summary, on=keys, how="left")
    result = result.merge(high_idx, on=keys, how="left")
    result = result.merge(low_idx, on=keys, how="left")
    result["intraday_ret_1m_path"] = result["last_close"] / result["first_open"].replace(0, np.nan) - 1.0
    result["intraday_first30_ret"] = result["first30_close"] / result["first_open"].replace(0, np.nan) - 1.0
    result["intraday_tail30_ret"] = result["tail30_close"] / result["tail30_open"].replace(0, np.nan) - 1.0
    result["intraday_first30_volume_share"] = result["first30_volume"] / result["total_volume"].replace(0, np.nan)
    result["intraday_tail30_volume_share"] = result["tail30_volume"] / result["total_volume"].replace(0, np.nan)
    result["intraday_opening_volume_ratio"] = result["first30_volume"] / (
        result["total_volume"] - result["tail30_volume"].fillna(0.0)
    ).replace(0, np.nan)
    vwap = result["total_amount"] / (result["total_volume"] * 100.0).replace(0, np.nan)
    result["intraday_vwap_gap"] = result["last_close"] / vwap.replace(0, np.nan) - 1.0
    result["intraday_close_strength"] = (result["last_close"] - result["low"]) / (result["high"] - result["low"]).replace(0, np.nan)
    result["intraday_high_time_ratio"] = result["high_bar_index"] / (result["high_bar_count"] - 1).replace(0, np.nan)
    result["intraday_low_time_ratio"] = result["low_bar_index"] / (result["low_bar_count"] - 1).replace(0, np.nan)
    result["auction_open_bar_ret"] = result["first_close"] / result["first_open"].replace(0, np.nan) - 1.0
    result["auction_open_bar_volume_share"] = result["first_volume"] / result["total_volume"].replace(0, np.nan)
    result["auction_open_bar_amount_share"] = result["first_amount"] / result["total_amount"].replace(0, np.nan)
    result["auction_open_bar_range_pct"] = (result["first_high"] - result["first_low"]) / result["first_open"].replace(0, np.nan)
    for column in TRUE_INTRADAY_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce")
    return result[["symbol", "trade_date", *TRUE_INTRADAY_COLUMNS, "intraday_bar_count"]].copy()


def build_fund_flow_feature_frame(
    duckdb_path: Path | str = DEFAULT_DUCKDB_PATH,
    *,
    fund_flow_table: str = DEFAULT_STOCK_FUND_FLOW_TABLE,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    table = _safe_table_name(fund_flow_table)
    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        if not _safe_table_exists(connection, table):
            return pd.DataFrame(columns=["symbol", "trade_date", *TRUE_FUND_FLOW_COLUMNS])
        frame = connection.execute(
            f"""
            select symbol, trade_date, net_inflow, main_net_amount, main_net_ratio
            from {table}
            where trade_date >= cast(? as date)
              and trade_date <= cast(? as date)
            order by symbol, trade_date
            """,
            [start_date, end_date],
        ).fetchdf()
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "trade_date", *TRUE_FUND_FLOW_COLUMNS])
    frame = frame.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.extract(r"(\d{1,6})", expand=False).fillna("").str.zfill(6)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    for column in ["net_inflow", "main_net_amount", "main_net_ratio"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["symbol", "trade_date"]).sort_values(["symbol", "trade_date"])
    grouped = frame.groupby("symbol", sort=False)
    frame["true_fund_net_inflow"] = frame["net_inflow"]
    frame["true_fund_main_net_amount"] = frame["main_net_amount"]
    frame["true_fund_main_net_ratio"] = frame["main_net_ratio"]
    frame["true_fund_main_net_amount_5"] = grouped["main_net_amount"].transform(lambda s: s.rolling(5, min_periods=2).mean())
    frame["true_fund_main_net_ratio_5"] = grouped["main_net_ratio"].transform(lambda s: s.rolling(5, min_periods=2).mean())
    inflow_flag = frame["main_net_amount"].gt(0).astype(float)
    frame["true_fund_main_inflow_streak_5"] = inflow_flag.groupby(frame["symbol"], sort=False).transform(
        lambda s: s.rolling(5, min_periods=1).sum() / 5.0,
    )
    frame["true_fund_flow_coverage"] = 1.0
    for column in TRUE_FUND_FLOW_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return frame[["symbol", "trade_date", *TRUE_FUND_FLOW_COLUMNS]].copy()


def build_auction_feature_frame(
    duckdb_path: Path | str = DEFAULT_DUCKDB_PATH,
    *,
    auction_table: str = DEFAULT_CALL_AUCTION_PROXY_TABLE,
    start_date: str,
    end_date: str,
    interval_minutes: int = 15,
) -> pd.DataFrame:
    table = _safe_table_name(auction_table)
    columns = [
        "symbol",
        "trade_date",
        "auction_open_gap",
        "auction_open_bar_ret",
        "auction_open_bar_volume_share",
        "auction_open_bar_amount_share",
        "auction_open_bar_range_pct",
        "auction_proxy_available_flag",
    ]
    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        if not _safe_table_exists(connection, table):
            return pd.DataFrame(columns=columns)
        frame = connection.execute(
            f"""
            select symbol,
                   trade_date,
                   auction_open_gap,
                   first_bar_ret as auction_open_bar_ret,
                   first_bar_volume_share as auction_open_bar_volume_share,
                   first_bar_amount_share as auction_open_bar_amount_share,
                   first_bar_range_pct as auction_open_bar_range_pct
            from {table}
            where interval_minutes = ?
              and trade_date >= cast(? as date)
              and trade_date <= cast(? as date)
            """,
            [int(interval_minutes), start_date, end_date],
        ).fetchdf()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame = frame.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.extract(r"(\d{1,6})", expand=False).fillna("").str.zfill(6)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    for column in columns:
        if column not in {"symbol", "trade_date", "auction_proxy_available_flag"} and column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame["auction_proxy_available_flag"] = 1.0
    return frame[columns].dropna(subset=["symbol", "trade_date"]).copy()


def inspect_multisource_coverage(
    duckdb_path: Path | str = DEFAULT_DUCKDB_PATH,
    *,
    price_table: str = DEFAULT_PRICE_TABLE,
    intraday_table: str = DEFAULT_INTRADAY_TABLE,
    fund_flow_table: str = DEFAULT_STOCK_FUND_FLOW_TABLE,
    auction_table: str = DEFAULT_CALL_AUCTION_PROXY_TABLE,
    start_date: str,
    end_date: str,
    stock_basic_path: Path | str | None = DEFAULT_STOCK_BASIC_PATH,
    intraday_interval_minutes: int = 1,
) -> dict[str, object]:
    price = _safe_table_name(price_table)
    intraday = _safe_table_name(intraday_table)
    fund_flow = _safe_table_name(fund_flow_table)
    auction_proxy = _safe_table_name(auction_table)
    coverage: dict[str, object] = {
        "start_date": start_date,
        "end_date": end_date,
        "price_table": price,
        "intraday_table": intraday,
    }
    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
        daily = connection.execute(
            f"""
            select count(*) as rows,
                   count(distinct symbol) as symbols,
                   count(distinct trade_date) as trade_days,
                   min(trade_date) as min_date,
                   max(trade_date) as max_date
            from {price}
            where trade_date >= cast(? as date)
              and trade_date <= cast(? as date)
            """,
            [start_date, end_date],
        ).fetchone()
        coverage["daily"] = {
            "rows": int(daily[0] or 0),
            "symbols": int(daily[1] or 0),
            "trade_days": int(daily[2] or 0),
            "min_date": _date_string(daily[3]),
            "max_date": _date_string(daily[4]),
        }
        if _safe_table_exists(connection, intraday):
            intraday_rows = connection.execute(
                f"""
                select interval_minutes,
                       count(*) as rows,
                       count(distinct symbol) as symbols,
                       count(distinct trade_date) as trade_days,
                       min(trade_date) as min_date,
                       max(trade_date) as max_date
                from {intraday}
                where trade_date >= cast(? as date)
                  and trade_date <= cast(? as date)
                group by interval_minutes
                order by interval_minutes
                """,
                [start_date, end_date],
            ).fetchall()
            selected_bar = connection.execute(
                f"""
                select min(bar_time) as min_time,
                       max(bar_time) as max_time,
                       count(distinct trade_date) filter (where bar_time = cast('09:30:00' as time)) as open_0930_days
                from {intraday}
                where interval_minutes = ?
                  and trade_date >= cast(? as date)
                  and trade_date <= cast(? as date)
                """,
                [int(intraday_interval_minutes), start_date, end_date],
            ).fetchone()
        else:
            intraday_rows = []
            selected_bar = None
        if _safe_table_exists(connection, fund_flow):
            fund_flow_row = connection.execute(
                f"""
                select count(*) as rows,
                       count(distinct symbol) as symbols,
                       count(distinct trade_date) as trade_days,
                       min(trade_date) as min_date,
                       max(trade_date) as max_date
                from {fund_flow}
                where trade_date >= cast(? as date)
                  and trade_date <= cast(? as date)
                """,
                [start_date, end_date],
            ).fetchone()
        else:
            fund_flow_row = None
        if _safe_table_exists(connection, auction_proxy):
            auction_proxy_row = connection.execute(
                f"""
                select count(*) as rows,
                       count(distinct symbol) as symbols,
                       count(distinct trade_date) as trade_days,
                       min(trade_date) as min_date,
                       max(trade_date) as max_date,
                       min(first_bar_time) as min_first_bar_time,
                       max(first_bar_time) as max_first_bar_time
                from {auction_proxy}
                where interval_minutes = ?
                  and trade_date >= cast(? as date)
                  and trade_date <= cast(? as date)
                """,
                [int(intraday_interval_minutes), start_date, end_date],
            ).fetchone()
        else:
            auction_proxy_row = None
        tables = [str(row[0]) for row in connection.execute("show tables").fetchall()]
    stock_basic = _load_stock_basic_metadata(stock_basic_path)
    intraday_summary = [
        {
            "interval_minutes": int(row[0]),
            "rows": int(row[1] or 0),
            "symbols": int(row[2] or 0),
            "trade_days": int(row[3] or 0),
            "min_date": _date_string(row[4]),
            "max_date": _date_string(row[5]),
        }
        for row in intraday_rows
    ]
    daily_days = max(int(coverage["daily"]["trade_days"]), 1)
    one_minute_days = next((item["trade_days"] for item in intraday_summary if item["interval_minutes"] == 1), 0)
    selected_interval_days = next(
        (item["trade_days"] for item in intraday_summary if item["interval_minutes"] == int(intraday_interval_minutes)),
        0,
    )
    coverage["intraday"] = {
        "available": bool(intraday_summary),
        "intervals": intraday_summary,
        "one_minute_trade_day_coverage": float(one_minute_days / daily_days),
        "selected_interval_minutes": int(intraday_interval_minutes),
        "selected_interval_trade_day_coverage": float(selected_interval_days / daily_days),
        "selected_interval_min_time": _time_string(selected_bar[0]) if selected_bar else "",
        "selected_interval_max_time": _time_string(selected_bar[1]) if selected_bar else "",
        "selected_interval_0930_trade_days": int(selected_bar[2] or 0) if selected_bar else 0,
    }
    true_auction_tables = [
        table
        for table in tables
        if "auction" in table.lower()
        and table.lower() != auction_proxy.lower()
        and "proxy" not in table.lower()
    ]
    has_true_auction_table = bool(true_auction_tables)
    selected_has_0930 = bool(selected_bar and int(selected_bar[2] or 0) > 0)
    coverage["auction"] = {
        "true_call_auction_table_available": has_true_auction_table,
        "true_call_auction_tables": true_auction_tables,
        "proxy_table_available": bool(auction_proxy_row and int(auction_proxy_row[0] or 0) > 0),
        "proxy_table": auction_proxy,
        "proxy_rows": int(auction_proxy_row[0] or 0) if auction_proxy_row else 0,
        "proxy_symbols": int(auction_proxy_row[1] or 0) if auction_proxy_row else 0,
        "proxy_trade_days": int(auction_proxy_row[2] or 0) if auction_proxy_row else 0,
        "proxy_min_date": _date_string(auction_proxy_row[3]) if auction_proxy_row else "",
        "proxy_max_date": _date_string(auction_proxy_row[4]) if auction_proxy_row else "",
        "proxy_min_first_bar_time": _time_string(auction_proxy_row[5]) if auction_proxy_row else "",
        "proxy_max_first_bar_time": _time_string(auction_proxy_row[6]) if auction_proxy_row else "",
        "mode": (
            "true_call_auction_table"
            if has_true_auction_table
            else "derived_daily_open_plus_first_intraday_bar"
            if auction_proxy_row and int(auction_proxy_row[0] or 0) > 0
            else "09:30_open_bar_proxy_plus_daily_open_gap"
            if selected_has_0930
            else "first_intraday_bar_proxy_plus_daily_open_gap"
        ),
    }
    true_fund_available = fund_flow_row is not None and int(fund_flow_row[0] or 0) > 0
    coverage["fund_flow"] = {
        "true_fund_flow_table_available": true_fund_available,
        "table": fund_flow,
        "rows": int(fund_flow_row[0] or 0) if fund_flow_row else 0,
        "symbols": int(fund_flow_row[1] or 0) if fund_flow_row else 0,
        "trade_days": int(fund_flow_row[2] or 0) if fund_flow_row else 0,
        "min_date": _date_string(fund_flow_row[3]) if fund_flow_row else "",
        "max_date": _date_string(fund_flow_row[4]) if fund_flow_row else "",
        "mode": "true_stock_main_fund_flow_plus_proxies" if true_fund_available else "amount_turnover_signed_money_proxy",
    }
    coverage["sector"] = {
        "stock_basic_rows": int(len(stock_basic)) if isinstance(stock_basic, pd.DataFrame) else 0,
        "industry_rows": int(stock_basic["industry"].notna().sum()) if isinstance(stock_basic, pd.DataFrame) and "industry" in stock_basic else 0,
        "mode": "industry_and_market_segment_cross_section_heat",
    }
    return coverage


def _clean_feature_columns(frame: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in feature_columns:
        if column not in result.columns:
            result[column] = 0.0
        result[column] = pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return result


def _numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def add_multisource_features(
    frame: pd.DataFrame,
    intraday_features: pd.DataFrame | None = None,
    fund_flow_features: pd.DataFrame | None = None,
    auction_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce")
    result["symbol"] = result["symbol"].astype(str).str.zfill(6)
    if isinstance(intraday_features, pd.DataFrame) and not intraday_features.empty:
        intraday = intraday_features.copy()
        intraday["trade_date"] = pd.to_datetime(intraday["trade_date"], errors="coerce")
        intraday["symbol"] = intraday["symbol"].astype(str).str.zfill(6)
        result = result.merge(intraday, on=["symbol", "trade_date"], how="left")
    result["intraday_available_flag"] = result.get("intraday_bar_count", pd.Series(np.nan, index=result.index)).notna().astype(float)
    if isinstance(fund_flow_features, pd.DataFrame) and not fund_flow_features.empty:
        fund_flow = fund_flow_features.copy()
        fund_flow["trade_date"] = pd.to_datetime(fund_flow["trade_date"], errors="coerce")
        fund_flow["symbol"] = fund_flow["symbol"].astype(str).str.zfill(6)
        result = result.merge(fund_flow, on=["symbol", "trade_date"], how="left")

    result["auction_open_gap"] = _numeric_series(result, "gap_return_1")
    result["intraday_close_position_proxy"] = _numeric_series(result, "close_position_day")
    result["intraday_body_ratio_proxy"] = _numeric_series(result, "body_ratio")
    result["intraday_upper_shadow_proxy"] = _numeric_series(result, "upper_shadow_ratio")
    result["intraday_lower_shadow_proxy"] = _numeric_series(result, "lower_shadow_ratio")
    result["fund_amount_ratio_5"] = _numeric_series(result, "amount_ratio_5")
    result["fund_turnover_ratio_20"] = _numeric_series(result, "turnover_ratio_20")
    result["fund_signed_amount_proxy_1"] = _numeric_series(result, "ret_1") * result["fund_amount_ratio_5"]
    result["fund_price_volume_confirm_5"] = _numeric_series(result, "ret_5") * _numeric_series(result, "volume_ratio_5")
    if isinstance(auction_features, pd.DataFrame) and not auction_features.empty:
        auction = auction_features.copy()
        auction["trade_date"] = pd.to_datetime(auction["trade_date"], errors="coerce")
        auction["symbol"] = auction["symbol"].astype(str).str.zfill(6)
        result = result.merge(auction, on=["symbol", "trade_date"], how="left", suffixes=("", "_auction_proxy"))
        for column in [
            "auction_open_gap",
            "auction_open_bar_ret",
            "auction_open_bar_volume_share",
            "auction_open_bar_amount_share",
            "auction_open_bar_range_pct",
        ]:
            proxy_column = f"{column}_auction_proxy"
            if proxy_column in result.columns:
                proxy_values = pd.to_numeric(result[proxy_column], errors="coerce").astype("float64")
                fallback_values = pd.to_numeric(_numeric_series(result, column, np.nan), errors="coerce").astype("float64")
                result[column] = proxy_values.where(proxy_values.notna(), fallback_values)
                result = result.drop(columns=[proxy_column])
    if "auction_proxy_available_flag" not in result.columns:
        result["auction_proxy_available_flag"] = 0.0
    result["sector_heat_score_5"] = pd.concat(
        [
            _numeric_series(result, "industry_rank_ret_5", 0.5),
            _numeric_series(result, "industry_rank_up_ratio_5", 0.5),
        ],
        axis=1,
    ).mean(axis=1)
    result["sector_fund_heat_score_5"] = _numeric_series(result, "industry_rank_amount_ratio_5", 0.5)
    result["segment_heat_score_5"] = pd.concat(
        [
            _numeric_series(result, "segment_rank_ret_5", 0.5),
            _numeric_series(result, "segment_rank_activity_5", 0.5),
        ],
        axis=1,
    ).mean(axis=1)
    result["market_trend_score_5"] = pd.concat(
        [
            _numeric_series(result, "market_ret_5"),
            _numeric_series(result, "market_up_ratio_5", 0.5) - 0.5,
        ],
        axis=1,
    ).mean(axis=1)
    result["market_risk_appetite_5"] = _numeric_series(result, "market_activity_ratio_5") - _numeric_series(
        result,
        "market_drawdown_20",
    ).abs()
    return _clean_feature_columns(result, MULTISOURCE_FEATURE_COLUMNS)


def _write_report(
    path: Path,
    *,
    selected_factors: Sequence[str],
    data_summary: dict[str, object],
    coverage: dict[str, object],
) -> None:
    lines = [
        "# One-year multi-source factor clustering",
        "",
        "## Selected factors",
        "",
    ]
    lines.extend(f"{index}. `{feature}`" for index, feature in enumerate(selected_factors, start=1))
    lines.extend(
        [
            "",
            "## Data summary",
            "",
            f"- Analysis window: `{data_summary.get('analysis_start')}` to `{data_summary.get('analysis_end')}`",
            f"- Eligible symbols: `{data_summary.get('eligible_symbols')}`",
            f"- Sample rows: `{data_summary.get('sample_rows')}`",
            f"- Candidate features: `{data_summary.get('candidate_feature_count')}`",
            f"- Intraday interval minutes: `{data_summary.get('intraday_interval_minutes')}`",
            f"- Intraday true rows merged: `{data_summary.get('intraday_feature_rows')}`",
            f"- Intraday available sample share: `{data_summary.get('intraday_available_sample_share')}`",
            f"- True intraday columns used in clustering: `{data_summary.get('true_intraday_columns_used')}`",
            f"- True fund-flow rows merged: `{data_summary.get('fund_flow_feature_rows')}`",
            f"- True fund-flow sample share: `{data_summary.get('fund_flow_available_sample_share')}`",
            f"- True fund-flow columns used in clustering: `{data_summary.get('true_fund_flow_columns_used')}`",
            f"- Auction/opening proxy rows merged: `{data_summary.get('auction_feature_rows')}`",
            "",
            "## Coverage notes",
            "",
            f"- Daily trade days: `{coverage.get('daily', {}).get('trade_days')}`",
            f"- 1-minute intraday trade-day coverage: `{coverage.get('intraday', {}).get('one_minute_trade_day_coverage')}`",
            f"- Selected-interval intraday trade-day coverage: `{coverage.get('intraday', {}).get('selected_interval_trade_day_coverage')}`",
            f"- Selected-interval first/last bar time: `{coverage.get('intraday', {}).get('selected_interval_min_time')}` / `{coverage.get('intraday', {}).get('selected_interval_max_time')}`",
            f"- Auction mode: `{coverage.get('auction', {}).get('mode')}`",
            f"- Auction proxy rows/symbols/days: `{coverage.get('auction', {}).get('proxy_rows')}` / `{coverage.get('auction', {}).get('proxy_symbols')}` / `{coverage.get('auction', {}).get('proxy_trade_days')}`",
            f"- Fund-flow mode: `{coverage.get('fund_flow', {}).get('mode')}`",
            f"- True fund-flow rows/symbols/days: `{coverage.get('fund_flow', {}).get('rows')}` / `{coverage.get('fund_flow', {}).get('symbols')}` / `{coverage.get('fund_flow', {}).get('trade_days')}`",
            f"- Sector mode: `{coverage.get('sector', {}).get('mode')}`",
            "",
            "Source coverage is recorded in `data_coverage.json`; columns with insufficient coverage are kept out of the final candidate set by coverage gates.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one_year_multisource_cluster(
    *,
    duckdb_path: Path | str = DEFAULT_DUCKDB_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    price_table: str = DEFAULT_PRICE_TABLE,
    intraday_table: str = DEFAULT_INTRADAY_TABLE,
    fund_flow_table: str = DEFAULT_STOCK_FUND_FLOW_TABLE,
    auction_table: str = DEFAULT_CALL_AUCTION_PROXY_TABLE,
    stock_basic_path: Path | str | None = DEFAULT_STOCK_BASIC_PATH,
    symbol_prefixes: Sequence[str] = DEFAULT_SYMBOL_PREFIXES,
    end_date: str | None = None,
    lookback_days: int = 365,
    top_n: int = 10,
    sample_limit: int = 500_000,
    importance_sample_limit: int = 220_000,
    batch_symbols: int = 240,
    min_history_rows: int = 180,
    symbol_limit: int | None = None,
    min_symbols_per_industry: int = 5,
    min_symbols_per_segment: int = 20,
    feature_cluster_distance: float = 0.35,
    market_state_clusters: int = 8,
    min_true_intraday_sample_share: float = 0.05,
    min_true_fund_flow_sample_share: float = 0.05,
    intraday_interval_minutes: int = 1,
    random_state: int = 42,
    progress: ProgressCallback | None = None,
) -> MultiSourceClusterResult:
    duckdb_path = Path(duckdb_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = _safe_table_name(price_table)

    date_splits = build_one_year_date_splits(
        duckdb_path,
        price_table=table,
        symbol_prefixes=symbol_prefixes,
        end_date=end_date,
        lookback_days=lookback_days,
    )
    stock_basic = _load_stock_basic_metadata(stock_basic_path)
    stock_metadata = (
        stock_basic.set_index("symbol").to_dict("index")
        if isinstance(stock_basic, pd.DataFrame) and not stock_basic.empty
        else {}
    )
    excluded_symbols = (
        stock_basic.loc[stock_basic["is_st"].astype(bool), "symbol"].astype(str).str.zfill(6).tolist()
        if isinstance(stock_basic, pd.DataFrame) and "is_st" in stock_basic.columns
        else []
    )

    with duckdb.connect(str(duckdb_path), read_only=True) as connection:
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
        raise RuntimeError("No eligible symbols found for one-year multi-source clustering.")

    _emit(progress, f"one-year window {date_splits.analysis_start} -> {date_splits.analysis_end}; symbols={len(symbols):,}")
    coverage = inspect_multisource_coverage(
        duckdb_path,
        price_table=table,
        intraday_table=intraday_table,
        fund_flow_table=fund_flow_table,
        auction_table=auction_table,
        start_date=date_splits.analysis_start,
        end_date=date_splits.analysis_end,
        stock_basic_path=stock_basic_path,
        intraday_interval_minutes=intraday_interval_minutes,
    )
    _emit(progress, "build market, industry, and segment context")
    market_context = build_market_context_frame(
        duckdb_path,
        price_table=table,
        symbol_prefixes=symbol_prefixes,
        query_start=pd.Timestamp(date_splits.analysis_start) - pd.Timedelta(days=260),
        analysis_end=pd.Timestamp(date_splits.analysis_end),
    )
    industry_context = build_industry_context_frame(
        duckdb_path,
        stock_basic=stock_basic,
        market_context=market_context,
        price_table=table,
        symbol_prefixes=symbol_prefixes,
        query_start=pd.Timestamp(date_splits.analysis_start) - pd.Timedelta(days=260),
        analysis_end=pd.Timestamp(date_splits.analysis_end),
        min_symbols_per_industry=min_symbols_per_industry,
        exclude_st=True,
    )
    segment_context = build_segment_context_frame(
        duckdb_path,
        stock_basic=stock_basic,
        market_context=market_context,
        price_table=table,
        symbol_prefixes=symbol_prefixes,
        query_start=pd.Timestamp(date_splits.analysis_start) - pd.Timedelta(days=260),
        analysis_end=pd.Timestamp(date_splits.analysis_end),
        min_symbols_per_segment=min_symbols_per_segment,
        exclude_st=True,
    )
    _emit(progress, "build true intraday features where available")
    intraday_features = build_intraday_feature_frame(
        duckdb_path,
        intraday_table=intraday_table,
        start_date=date_splits.analysis_start,
        end_date=date_splits.analysis_end,
        interval_minutes=intraday_interval_minutes,
    )
    _emit(progress, "build true stock fund-flow features where available")
    fund_flow_features = build_fund_flow_feature_frame(
        duckdb_path,
        fund_flow_table=fund_flow_table,
        start_date=date_splits.analysis_start,
        end_date=date_splits.analysis_end,
    )
    _emit(progress, "build call-auction/opening proxy features where available")
    auction_features = build_auction_feature_frame(
        duckdb_path,
        auction_table=auction_table,
        start_date=date_splits.analysis_start,
        end_date=date_splits.analysis_end,
        interval_minutes=intraday_interval_minutes,
    )

    _emit(progress, "collect one-year multi-source factor sample")
    frames: list[pd.DataFrame] = []
    rows_seen = 0
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
        enriched = add_multisource_features(batch, intraday_features, fund_flow_features, auction_features)
        if enriched.empty:
            continue
        rows_seen += int(len(enriched))
        frames.append(enriched)
        if batch_index == 1 or batch_index % 10 == 0 or batch_index >= total_batches:
            _emit(progress, f"sample batch {batch_index}/{total_batches}: rows={rows_seen:,}")
    if not frames:
        raise RuntimeError("No factor rows were collected for multi-source clustering.")
    sample = pd.concat(frames, ignore_index=True, sort=False)
    if len(sample) > sample_limit:
        sample = sample.sample(n=int(sample_limit), random_state=random_state).reset_index(drop=True)
    else:
        sample = sample.reset_index(drop=True)
    intraday_share = float(pd.to_numeric(sample.get("intraday_available_flag"), errors="coerce").fillna(0.0).mean())
    true_intraday_columns_used = bool(intraday_share >= float(min_true_intraday_sample_share))
    fund_flow_share = float(
        pd.to_numeric(
            sample.get("true_fund_flow_coverage", pd.Series(0.0, index=sample.index)),
            errors="coerce",
        )
        .fillna(0.0)
        .mean()
    )
    true_fund_flow_columns_used = bool(fund_flow_share >= float(min_true_fund_flow_sample_share))
    multisource_candidate_columns = [
        *(TRUE_INTRADAY_COLUMNS if true_intraday_columns_used else []),
        *(TRUE_FUND_FLOW_FEATURE_COLUMNS if true_fund_flow_columns_used else []),
        *DAILY_PROXY_COLUMNS,
    ]
    feature_columns = [
        *FEATURE_COLUMNS,
        *MICROSTRUCTURE_COLUMNS,
        *MARKET_CONTEXT_COLUMNS,
        *INDUSTRY_CONTEXT_COLUMNS,
        *SEGMENT_CONTEXT_COLUMNS,
        *multisource_candidate_columns,
    ]
    if not true_intraday_columns_used:
        _emit(
            progress,
            f"true intraday sample share {intraday_share:.4f} is below {float(min_true_intraday_sample_share):.4f}; use audited daily proxies for clustering",
        )
    if not true_fund_flow_columns_used:
        _emit(
            progress,
            f"true fund-flow sample share {fund_flow_share:.4f} is below {float(min_true_fund_flow_sample_share):.4f}; keep amount/turnover proxies for clustering",
        )

    _emit(progress, f"rank and cluster multi-source factors; sample rows={len(sample):,}")
    ranking, cluster_summary, state_summary, selected_factors = rank_next_day_factors(
        sample,
        feature_columns=feature_columns,
        top_n=top_n,
        feature_cluster_distance=feature_cluster_distance,
        market_state_clusters=market_state_clusters,
        importance_sample_limit=importance_sample_limit,
        random_state=random_state,
    )

    feature_ranking_path = output_dir / "feature_ranking.csv"
    cluster_summary_path = output_dir / "feature_cluster_summary.csv"
    state_summary_path = output_dir / "market_state_cluster_summary.csv"
    selected_factors_path = output_dir / "selected_top10_factors.csv"
    coverage_path = output_dir / "data_coverage.json"
    report_path = output_dir / "cluster_report.md"

    ranking.to_csv(feature_ranking_path, index=False, encoding="utf-8")
    cluster_summary.to_csv(cluster_summary_path, index=False, encoding="utf-8")
    state_summary.to_csv(state_summary_path, index=False, encoding="utf-8")
    ranking.loc[ranking["selected_top10"]].sort_values("selected_rank").to_csv(
        selected_factors_path,
        index=False,
        encoding="utf-8",
    )
    data_summary = {
        "analysis_start": date_splits.analysis_start,
        "analysis_end": date_splits.analysis_end,
        "eligible_symbols": int(len(symbols)),
        "rows_seen": int(rows_seen),
        "sample_rows": int(len(sample)),
        "candidate_feature_count": int(len([column for column in feature_columns if column in sample.columns])),
        "intraday_feature_rows": int(len(intraday_features)),
        "intraday_interval_minutes": int(intraday_interval_minutes),
        "intraday_available_sample_share": intraday_share,
        "min_true_intraday_sample_share": float(min_true_intraday_sample_share),
        "true_intraday_columns_used": true_intraday_columns_used,
        "fund_flow_feature_rows": int(len(fund_flow_features)),
        "auction_feature_rows": int(len(auction_features)),
        "fund_flow_available_sample_share": fund_flow_share,
        "min_true_fund_flow_sample_share": float(min_true_fund_flow_sample_share),
        "true_fund_flow_columns_used": true_fund_flow_columns_used,
        "excluded_st_symbols": int(len(excluded_symbols)),
        "price_table": table,
        "intraday_table": intraday_table,
        "fund_flow_table": fund_flow_table,
        "auction_table": auction_table,
        "duckdb_path": str(duckdb_path),
    }
    payload = {
        "target": "next_trading_day_close_to_close_up",
        "date_splits": asdict(date_splits),
        "data_summary": data_summary,
        "coverage": coverage,
        "selected_factors": selected_factors,
        "artifacts": {
            "feature_ranking": str(feature_ranking_path),
            "feature_cluster_summary": str(cluster_summary_path),
            "market_state_cluster_summary": str(state_summary_path),
            "selected_top10_factors": str(selected_factors_path),
            "report": str(report_path),
        },
    }
    coverage_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    _write_report(report_path, selected_factors=selected_factors, data_summary=data_summary, coverage=coverage)
    _emit(progress, "selected factors: " + ", ".join(selected_factors))
    _emit(progress, f"artifacts written to {output_dir}")
    return MultiSourceClusterResult(
        output_dir=str(output_dir),
        feature_ranking_path=str(feature_ranking_path),
        cluster_summary_path=str(cluster_summary_path),
        selected_factors_path=str(selected_factors_path),
        coverage_path=str(coverage_path),
        report_path=str(report_path),
        selected_factors=selected_factors,
        date_splits=asdict(date_splits),
        data_summary=data_summary,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cluster one-year A-share factors with intraday and multi-source context.")
    parser.add_argument("--duckdb-path", default=str(DEFAULT_DUCKDB_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--price-table", default=DEFAULT_PRICE_TABLE)
    parser.add_argument("--intraday-table", default=DEFAULT_INTRADAY_TABLE)
    parser.add_argument("--fund-flow-table", default=DEFAULT_STOCK_FUND_FLOW_TABLE)
    parser.add_argument("--auction-table", default=DEFAULT_CALL_AUCTION_PROXY_TABLE)
    parser.add_argument("--stock-basic-path", default=str(DEFAULT_STOCK_BASIC_PATH))
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--sample-limit", type=int, default=500_000)
    parser.add_argument("--importance-sample-limit", type=int, default=220_000)
    parser.add_argument("--batch-symbols", type=int, default=240)
    parser.add_argument("--min-history-rows", type=int, default=180)
    parser.add_argument("--symbol-limit", type=int, default=None)
    parser.add_argument("--min-symbols-per-industry", type=int, default=5)
    parser.add_argument("--min-symbols-per-segment", type=int, default=20)
    parser.add_argument("--feature-cluster-distance", type=float, default=0.35)
    parser.add_argument("--market-state-clusters", type=int, default=8)
    parser.add_argument("--min-true-intraday-sample-share", type=float, default=0.05)
    parser.add_argument("--min-true-fund-flow-sample-share", type=float, default=0.05)
    parser.add_argument("--intraday-interval-minutes", type=int, default=1)
    parser.add_argument("--symbol-prefixes", default=",".join(DEFAULT_SYMBOL_PREFIXES))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    prefixes = tuple(prefix.strip() for prefix in str(args.symbol_prefixes).split(",") if prefix.strip())
    progress = None if args.quiet else (lambda message: print(f"[multi-source-cluster] {message}", file=sys.stderr, flush=True))
    result = run_one_year_multisource_cluster(
        duckdb_path=args.duckdb_path,
        output_dir=args.output_dir,
        price_table=args.price_table,
        intraday_table=args.intraday_table,
        fund_flow_table=args.fund_flow_table,
        auction_table=args.auction_table,
        stock_basic_path=args.stock_basic_path,
        symbol_prefixes=prefixes,
        end_date=args.end_date,
        lookback_days=args.lookback_days,
        top_n=args.top_n,
        sample_limit=args.sample_limit,
        importance_sample_limit=args.importance_sample_limit,
        batch_symbols=args.batch_symbols,
        min_history_rows=args.min_history_rows,
        symbol_limit=args.symbol_limit,
        min_symbols_per_industry=args.min_symbols_per_industry,
        min_symbols_per_segment=args.min_symbols_per_segment,
        feature_cluster_distance=args.feature_cluster_distance,
        market_state_clusters=args.market_state_clusters,
        min_true_intraday_sample_share=args.min_true_intraday_sample_share,
        min_true_fund_flow_sample_share=args.min_true_fund_flow_sample_share,
        intraday_interval_minutes=args.intraday_interval_minutes,
        random_state=args.random_state,
        progress=progress,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
