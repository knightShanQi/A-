from __future__ import annotations

import datetime as dt
import os
import pickle
import re
import threading
import time
import unicodedata
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from . import database_source, duckdb_source

try:
    from py_mini_racer import MiniRacer
except ImportError:  # pragma: no cover
    MiniRacer = None

try:
    import akshare as ak
except ImportError:  # pragma: no cover
    ak = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / ".cache"
DAILY_HISTORY_CACHE_VERSION = 1
DAILY_HISTORY_INCREMENTAL_LOOKBACK_DAYS = 10
A_SHARE_UNIVERSE_CACHE_VERSION = 1
TUSHARE_API_URL = "https://api.tushare.pro"
TUSHARE_DEFAULT_TOKEN = "1a8c415549c26b9a265ff349ac24a4b799068b13e9319aedf6703675"


def _active_database_source():
    if duckdb_source.is_enabled():
        return duckdb_source
    if database_source.is_enabled():
        return database_source
    return None


DAILY_RENAME = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude_pct",
    "涨跌幅": "change_pct",
    "涨跌额": "change_amount",
    "换手率": "turnover",
    "股票代码": "symbol",
}

SPOT_RENAME = {
    "代码": "symbol",
    "名称": "name",
    "最新价": "latest_price",
    "涨跌幅": "change_pct",
    "涨跌额": "change_amount",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude_pct",
    "最高": "high",
    "最低": "low",
    "今开": "open",
    "昨收": "prev_close",
    "量比": "volume_ratio",
    "换手率": "turnover",
    "市盈率-动态": "pe_ttm",
    "市净率": "pb",
    "总市值": "market_cap",
    "流通市值": "float_market_cap",
    "涨速": "speed",
    "5分钟涨跌": "change_5m",
    "60日涨跌幅": "change_60d",
    "年初至今涨跌幅": "change_ytd",
}

MINUTE_RENAME = {
    "时间": "datetime",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "均价": "avg_price",
}

FALLBACK_WATCHLIST = {
    "600519": "贵州茅台",
    "000333": "美的集团",
    "002594": "比亚迪",
    "600036": "招商银行",
    "000858": "五粮液",
    "601318": "中国平安",
    "300750": "宁德时代",
    "601899": "紫金矿业",
    "600900": "长江电力",
    "002475": "立讯精密",
}

THS_FUND_FLOW_REFERERS = {
    "industry": "http://data.10jqka.com.cn/funds/hyzjl/",
    "concept": "http://data.10jqka.com.cn/funds/gnzjl/",
}

THS_FUND_FLOW_PERIOD_PATHS = {
    "即时": "field/tradezdf/order/desc/page/{page}/ajax/1/free/1/",
    "3日排行": "board/3/field/tradezdf/order/desc/page/{page}/ajax/1/free/1/",
    "5日排行": "board/5/field/tradezdf/order/desc/page/{page}/ajax/1/free/1/",
    "10日排行": "board/10/field/tradezdf/order/desc/page/{page}/ajax/1/free/1/",
    "20日排行": "board/20/field/tradezdf/order/desc/page/{page}/ajax/1/free/1/",
}


def ensure_akshare() -> None:
    if ak is None:
        raise RuntimeError(
            "当前环境未安装 AkShare。请先执行：\n"
            ".\\.venv\\Scripts\\python.exe -m pip install -e ."
        )


def _resolve_tushare_token() -> str:
    token = os.getenv("TUSHARE_TOKEN", TUSHARE_DEFAULT_TOKEN).strip()
    if not token:
        raise RuntimeError("Tushare token is not configured")
    return token


def _call_tushare_api(api_name: str, params: dict | None = None, fields: str = "") -> pd.DataFrame:
    payload = {
        "api_name": api_name,
        "token": _resolve_tushare_token(),
        "params": params or {},
        "fields": fields,
    }
    response = requests.post(TUSHARE_API_URL, json=payload, timeout=15)
    response.raise_for_status()
    body = response.json()
    if int(body.get("code", -1)) != 0:
        raise RuntimeError(str(body.get("msg") or f"Tushare {api_name} failed"))
    data = body.get("data") or {}
    items = data.get("items") or []
    columns = data.get("fields") or []
    return pd.DataFrame(items, columns=columns)


def normalize_symbol(symbol: str) -> str:
    digits = re.sub(r"\D", "", str(symbol))
    if len(digits) != 6:
        raise ValueError(f"无法识别股票代码: {symbol}")
    return digits


def try_normalize_symbol(symbol: str) -> str | None:
    digits = re.sub(r"\D", "", str(symbol))
    if len(digits) == 6:
        return digits
    return None


def normalize_search_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = re.sub(r"\s+", "", normalized)
    return normalized.upper()


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _finalize_daily_history_frame(
    df: pd.DataFrame,
    clean_symbol: str,
    start_date: str,
    end_date: str,
    source: str,
) -> pd.DataFrame:
    if df.empty:
        raise ValueError(f"{clean_symbol} 没有拉到历史数据")

    normalized = df.copy().rename(columns=DAILY_RENAME)
    if "date" not in normalized.columns:
        normalized = normalized.reset_index().rename(columns={"index": "date"})
    if "date" not in normalized.columns:
        raise ValueError(f"{clean_symbol} 缺少日期字段")

    numeric_cols = [
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "amplitude_pct",
        "change_pct",
        "change_amount",
        "turnover",
    ]
    normalized = _coerce_numeric(normalized, numeric_cols)
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized = normalized.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")

    start_ts = pd.to_datetime(start_date, format="%Y%m%d", errors="coerce")
    end_ts = pd.to_datetime(end_date, format="%Y%m%d", errors="coerce")
    if pd.notna(start_ts):
        normalized = normalized[normalized["date"] >= start_ts]
    if pd.notna(end_ts):
        normalized = normalized[normalized["date"] <= end_ts]
    if normalized.empty:
        raise ValueError(f"{clean_symbol} 在指定区间没有可用历史数据")

    if "amount" not in normalized.columns or normalized["amount"].isna().all():
        normalized["amount"] = normalized["close"] * normalized["volume"]
    if "change_amount" not in normalized.columns or normalized["change_amount"].isna().all():
        normalized["change_amount"] = normalized["close"].diff()
    if "change_pct" not in normalized.columns or normalized["change_pct"].isna().all():
        normalized["change_pct"] = normalized["close"].pct_change() * 100
    if "amplitude_pct" not in normalized.columns or normalized["amplitude_pct"].isna().all():
        low_base = normalized["low"].where(normalized["low"].ne(0))
        normalized["amplitude_pct"] = (normalized["high"] / low_base - 1) * 100
    if "turnover" not in normalized.columns:
        normalized["turnover"] = float("nan")
    if "symbol" not in normalized.columns:
        normalized["symbol"] = clean_symbol

    normalized = normalized.set_index("date", drop=False)
    normalized.attrs["data_source"] = source
    return normalized


def _daily_history_cache_path(symbol: str, adjust: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_adjust = re.sub(r"[^0-9A-Za-z_-]", "", str(adjust or "")) or "raw"
    return CACHE_DIR / f"daily_history_v{DAILY_HISTORY_CACHE_VERSION}_{symbol}_{safe_adjust}.pkl"


def _normalize_cached_daily_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    normalized = df.copy()
    if "date" in normalized.columns:
        normalized = normalized.reset_index(drop=True)
    else:
        normalized = normalized.reset_index().rename(columns={"index": "date"})
    if "date" not in normalized.columns:
        return pd.DataFrame()

    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized = (
        normalized.dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    if normalized.empty:
        return normalized
    normalized = normalized.set_index("date", drop=False)
    return normalized


def _read_daily_history_disk_cache(symbol: str, adjust: str) -> tuple[pd.DataFrame | None, dict]:
    cache_path = _daily_history_cache_path(symbol, adjust)
    if not cache_path.exists():
        return None, {}
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None, {}

    meta = payload.get("meta", {})
    df = payload.get("data")
    if meta.get("cache_version") != DAILY_HISTORY_CACHE_VERSION:
        return None, {}
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None, meta

    normalized = _normalize_cached_daily_history(df)
    if normalized.empty:
        return None, meta
    data_source = str(meta.get("data_source") or normalized.attrs.get("data_source") or "")
    if data_source:
        normalized.attrs["data_source"] = data_source
    return normalized, meta


def _write_daily_history_disk_cache(
    symbol: str,
    adjust: str,
    df: pd.DataFrame,
    *,
    requested_start_date: str,
    requested_end_date: str,
) -> None:
    normalized = _normalize_cached_daily_history(df)
    if normalized.empty:
        return

    min_date = normalized["date"].min()
    max_date = normalized["date"].max()
    payload = {
        "meta": {
            "cache_version": DAILY_HISTORY_CACHE_VERSION,
            "symbol": symbol,
            "adjust": adjust,
            "requested_start_date": str(requested_start_date),
            "requested_end_date": str(requested_end_date),
            "cached_start_date": min_date.strftime("%Y%m%d") if pd.notna(min_date) else "",
            "cached_end_date": max_date.strftime("%Y%m%d") if pd.notna(max_date) else "",
            "data_source": str(normalized.attrs.get("data_source", "")),
        },
        "data": normalized,
    }
    with _daily_history_cache_path(symbol, adjust).open("wb") as handle:
        pickle.dump(payload, handle)


def _slice_daily_history(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    sliced = df.copy()
    start_ts = pd.to_datetime(start_date, format="%Y%m%d", errors="coerce")
    end_ts = pd.to_datetime(end_date, format="%Y%m%d", errors="coerce")
    if pd.notna(start_ts):
        sliced = sliced[sliced["date"] >= start_ts]
    if pd.notna(end_ts):
        sliced = sliced[sliced["date"] <= end_ts]
    return sliced.copy()


def _merge_daily_history_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    valid_frames = [frame.copy() for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not valid_frames:
        return pd.DataFrame()

    merged = pd.concat(valid_frames, axis=0, ignore_index=False)
    merged = _normalize_cached_daily_history(merged)
    if merged.empty:
        return merged

    for frame in reversed(valid_frames):
        source = str(frame.attrs.get("data_source", "")).strip()
        if source:
            merged.attrs["data_source"] = source
            break
    return merged


def fetch_market_spot() -> pd.DataFrame:
    ensure_akshare()
    try:
        df = ak.stock_zh_a_spot_em()
        df = df.rename(columns=SPOT_RENAME).copy()
        keep_cols = [
            "symbol",
            "name",
            "latest_price",
            "change_pct",
            "change_amount",
            "volume",
            "amount",
            "amplitude_pct",
            "high",
            "low",
            "open",
            "prev_close",
            "volume_ratio",
            "turnover",
            "pe_ttm",
            "pb",
            "market_cap",
            "float_market_cap",
            "speed",
            "change_5m",
            "change_60d",
            "change_ytd",
        ]
        existing = [col for col in keep_cols if col in df.columns]
        df = df[existing]
        numeric_cols = [col for col in existing if col not in {"symbol", "name"}]
        df = _coerce_numeric(df, numeric_cols)
        df["symbol"] = df["symbol"].astype(str).str.zfill(6)
        if "name" not in df.columns:
            df["name"] = df["symbol"]
        df["data_mode"] = "live"
        return df
    except Exception:
        return _build_fallback_market_spot()


@lru_cache(maxsize=8192)
def _fetch_daily_history_cached(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str,
    timeout: float,
) -> pd.DataFrame:
    ensure_akshare()
    clean_symbol = symbol

    def _with_retry(fetcher, attempts: int = 2, pause_seconds: float = 0.35) -> pd.DataFrame:
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                return fetcher()
            except Exception as exc:  # pragma: no cover - exercised through source-specific tests
                last_exc = exc
                if attempt + 1 < attempts:
                    time.sleep(pause_seconds * float(attempt + 1))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("daily history fetch failed without exception")

    try:
        df = _with_retry(
            lambda: ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
                timeout=timeout,
            )
        )
        return _finalize_daily_history_frame(df, clean_symbol, start_date, end_date, source="eastmoney")
    except Exception as primary_exc:
        try:
            fallback_symbol = f"{market_from_symbol(clean_symbol)}{clean_symbol}"
            df = _with_retry(lambda: ak.stock_zh_a_daily(symbol=fallback_symbol, adjust=adjust))
            return _finalize_daily_history_frame(df, clean_symbol, start_date, end_date, source="sina")
        except Exception:
            raise primary_exc


def fetch_daily_history(
    symbol: str,
    start_date: str = "20220101",
    end_date: str | None = None,
    adjust: str = "qfq",
    timeout: float = 8.0,
) -> pd.DataFrame:
    clean_symbol = normalize_symbol(symbol)
    final_end_date = end_date or dt.date.today().strftime("%Y%m%d")
    using_default_cache_dir = Path(CACHE_DIR).resolve() == (PROJECT_ROOT / ".cache").resolve()
    active_source = _active_database_source()
    if using_default_cache_dir and active_source is not None:
        try:
            database_df = active_source.fetch_daily_history(clean_symbol, start_date, final_end_date)
        except Exception:
            database_df = pd.DataFrame()
        if not database_df.empty:
            database_min = pd.to_datetime(database_df["date"], errors="coerce").min()
            database_max = pd.to_datetime(database_df["date"], errors="coerce").max()
            requested_start = pd.to_datetime(start_date, format="%Y%m%d", errors="coerce")
            requested_end = pd.to_datetime(final_end_date, format="%Y%m%d", errors="coerce")
            start_gap_days = (database_min - requested_start).days if pd.notna(database_min) and pd.notna(requested_start) else 0
            covers_start = pd.isna(requested_start) or (
                pd.notna(database_min)
                and (database_min <= requested_start or 0 <= start_gap_days <= 10)
            )
            covers_end = pd.isna(requested_end) or (pd.notna(database_max) and database_max >= requested_end)
            allow_partial = os.getenv("OPENCLAW_DATABASE_ALLOW_PARTIAL", "").strip().lower() in {"1", "true", "yes"}
            if allow_partial or (covers_start and covers_end):
                return database_df

    start_ts = pd.to_datetime(start_date, format="%Y%m%d", errors="coerce")
    end_ts = pd.to_datetime(final_end_date, format="%Y%m%d", errors="coerce")

    cached_df, cache_meta = _read_daily_history_disk_cache(clean_symbol, adjust)
    if cached_df is not None and not cached_df.empty:
        cached_min = pd.to_datetime(cached_df["date"], errors="coerce").min()
        cached_max = pd.to_datetime(cached_df["date"], errors="coerce").max()
        covers_start = pd.isna(start_ts) or (pd.notna(cached_min) and cached_min <= start_ts)
        # Treat the cache as covering the requested end only when the cached
        # rows themselves actually reach that date. A previous fetch may have
        # requested today's date before the provider finished publishing the
        # close, which leaves requested_end_date ahead of cached_max.
        covers_end = pd.isna(end_ts) or (pd.notna(cached_max) and cached_max >= end_ts)
        if covers_start and covers_end:
            return _slice_daily_history(cached_df, start_date, final_end_date)

    if cached_df is None or cached_df.empty:
        fresh = _fetch_daily_history_cached(clean_symbol, start_date, final_end_date, adjust, float(timeout)).copy()
        _write_daily_history_disk_cache(
            clean_symbol,
            adjust,
            fresh,
            requested_start_date=start_date,
            requested_end_date=final_end_date,
        )
        return fresh

    refreshed = cached_df
    needs_write = False
    cached_min = pd.to_datetime(cached_df["date"], errors="coerce").min()
    cached_max = pd.to_datetime(cached_df["date"], errors="coerce").max()
    requested_start_ts = pd.to_datetime(cache_meta.get("requested_start_date"), format="%Y%m%d", errors="coerce")

    should_refresh_prefix = pd.notna(start_ts) and pd.notna(cached_min) and start_ts < cached_min and (
        pd.isna(requested_start_ts) or start_ts < requested_start_ts
    )
    if should_refresh_prefix:
        prefix_end_ts = cached_min - pd.Timedelta(days=1)
        if prefix_end_ts >= start_ts:
            try:
                prefix_df = _fetch_daily_history_cached(
                    clean_symbol,
                    start_date,
                    prefix_end_ts.strftime("%Y%m%d"),
                    adjust,
                    float(timeout),
                ).copy()
            except Exception:
                prefix_df = pd.DataFrame()
            if not prefix_df.empty:
                refreshed = _merge_daily_history_frames(prefix_df, refreshed)
                needs_write = True

    requested_end_ts = pd.to_datetime(cache_meta.get("requested_end_date"), format="%Y%m%d", errors="coerce")
    should_refresh_suffix = pd.notna(end_ts) and (
        pd.isna(cached_max)
        or cached_max < end_ts
        or pd.isna(requested_end_ts)
        or end_ts > requested_end_ts
    )
    if should_refresh_suffix and pd.notna(cached_max):
        refresh_start_ts = cached_max - pd.Timedelta(days=DAILY_HISTORY_INCREMENTAL_LOOKBACK_DAYS)
        if pd.notna(start_ts):
            refresh_start_ts = max(refresh_start_ts, start_ts)
        try:
            suffix_df = _fetch_daily_history_cached(
                clean_symbol,
                refresh_start_ts.strftime("%Y%m%d"),
                final_end_date,
                adjust,
                float(timeout),
            ).copy()
        except Exception:
            suffix_df = pd.DataFrame()
        if not suffix_df.empty:
            refreshed = _merge_daily_history_frames(refreshed, suffix_df)
            needs_write = True

    if needs_write:
        requested_start_date = cache_meta.get("requested_start_date") or start_date
        cached_requested_start_ts = pd.to_datetime(requested_start_date, format="%Y%m%d", errors="coerce")
        if pd.notna(start_ts) and (pd.isna(cached_requested_start_ts) or start_ts < cached_requested_start_ts):
            requested_start_date = start_date
        requested_end_date = cache_meta.get("requested_end_date") or final_end_date
        cached_requested_end_ts = pd.to_datetime(requested_end_date, format="%Y%m%d", errors="coerce")
        if pd.notna(end_ts) and (pd.isna(cached_requested_end_ts) or end_ts > cached_requested_end_ts):
            requested_end_date = final_end_date
        _write_daily_history_disk_cache(
            clean_symbol,
            adjust,
            refreshed,
            requested_start_date=str(requested_start_date),
            requested_end_date=str(requested_end_date),
        )

    return _slice_daily_history(refreshed, start_date, final_end_date)


def clear_daily_history_cache(*, include_disk: bool = True) -> None:
    _fetch_daily_history_cached.cache_clear()
    if not include_disk:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for path in CACHE_DIR.glob(f"daily_history_v{DAILY_HISTORY_CACHE_VERSION}_*.pkl"):
        try:
            path.unlink()
        except OSError:
            continue


@lru_cache(maxsize=16)
def _fetch_index_daily_history_cached(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    ensure_akshare()
    try:
        df = ak.stock_zh_index_daily(symbol=symbol)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df

    normalized = df.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized = _coerce_numeric(normalized, ["open", "high", "low", "close", "volume", "amount"])
    normalized = normalized.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
    start_ts = pd.to_datetime(start_date, errors="coerce")
    end_ts = pd.to_datetime(end_date, errors="coerce")
    if pd.notna(start_ts):
        normalized = normalized[normalized["date"] >= start_ts]
    if pd.notna(end_ts):
        normalized = normalized[normalized["date"] <= end_ts]
    normalized = normalized.reset_index(drop=True)
    return normalized


def fetch_index_daily_history(
    symbol: str = "sh000300",
    start_date: str = "20220101",
    end_date: str | None = None,
) -> pd.DataFrame:
    final_end_date = end_date or dt.date.today().strftime("%Y%m%d")
    return _fetch_index_daily_history_cached(symbol, start_date, final_end_date).copy()


def _build_fallback_market_spot() -> pd.DataFrame:
    rows: list[dict] = []
    for symbol, name in FALLBACK_WATCHLIST.items():
        try:
            hist = fetch_daily_history(symbol=symbol, start_date="20230101")
        except Exception:
            continue
        latest = hist.iloc[-1]
        prev_close = hist["close"].iloc[-2] if len(hist) >= 2 else latest["close"]
        rows.append(
            {
                "symbol": symbol,
                "name": name,
                "latest_price": float(latest["close"]),
                "change_pct": float(latest.get("change_pct", 0.0)),
                "change_amount": float(latest.get("change_amount", 0.0)),
                "volume": float(latest.get("volume", 0.0)),
                "amount": float(latest.get("amount", 0.0)),
                "amplitude_pct": float(latest.get("amplitude_pct", 0.0)),
                "high": float(latest["high"]),
                "low": float(latest["low"]),
                "open": float(latest["open"]),
                "prev_close": float(prev_close),
                "volume_ratio": 1.0,
                "turnover": float(latest.get("turnover", 0.0)),
                "pe_ttm": float("nan"),
                "pb": float("nan"),
                "market_cap": float("nan"),
                "float_market_cap": float("nan"),
                "speed": 0.0,
                "change_5m": 0.0,
                "change_60d": 0.0,
                "change_ytd": 0.0,
                "data_mode": "fallback",
            }
        )
    return pd.DataFrame(rows)


def _finalize_minute_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    rename_map = dict(MINUTE_RENAME)
    rename_map["day"] = "datetime"
    df = df.rename(columns=rename_map).copy()
    df = _coerce_numeric(df, ["open", "close", "high", "low", "volume", "amount", "avg_price"])
    if "datetime" not in df.columns:
        return pd.DataFrame()

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    if df.empty:
        return df

    latest_session = df["datetime"].dt.normalize().max()
    df = df[df["datetime"].dt.normalize() == latest_session].copy().reset_index(drop=True)
    if df.empty:
        return df

    if "avg_price" not in df.columns or df["avg_price"].isna().all():
        cum_volume = df["volume"].fillna(0).cumsum()
        cum_amount = df["amount"].fillna(0).cumsum()
        avg_price = cum_amount.div(cum_volume.where(cum_volume.ne(0)))
        df["avg_price"] = avg_price.ffill().fillna(df["close"])
    else:
        df["avg_price"] = df["avg_price"].ffill().fillna(df["close"])

    return df


def fetch_minute_history(symbol: str) -> pd.DataFrame:
    clean_symbol = normalize_symbol(symbol)
    active_source = _active_database_source()
    if active_source is not None and hasattr(active_source, "fetch_intraday_history"):
        try:
            df = active_source.fetch_intraday_history(clean_symbol, interval_minutes=1)
            df = _finalize_minute_history(df)
            if not df.empty:
                df["symbol"] = clean_symbol
                df.attrs["data_source"] = "duckdb"
                return df
        except Exception:
            pass

    ensure_akshare()

    try:
        minute_symbol = f"{market_from_symbol(clean_symbol)}{clean_symbol}"
        df = ak.stock_zh_a_minute(symbol=minute_symbol, period="1", adjust="")
        df = _finalize_minute_history(df)
        if not df.empty:
            df["symbol"] = clean_symbol
            return df
    except Exception:
        pass

    today = dt.date.today().strftime("%Y-%m-%d")
    start = f"{today} 09:30:00"
    end = f"{today} 15:00:00"
    try:
        df = ak.stock_zh_a_hist_min_em(
            symbol=clean_symbol,
            start_date=start,
            end_date=end,
            period="1",
            adjust="",
        )
    except Exception:
        return pd.DataFrame()

    df = _finalize_minute_history(df)
    if not df.empty:
        df["symbol"] = clean_symbol
    return df


def parse_watchlist(raw_text: str) -> list[str]:
    if not raw_text.strip():
        return []
    pieces = re.split(r"[\s,，;；]+", raw_text.strip())
    return [normalize_symbol(piece) for piece in pieces if piece.strip()]


def market_clock(now: dt.datetime | None = None) -> dict[str, str]:
    now = now or dt.datetime.now()
    current = now.strftime("%Y-%m-%d %H:%M:%S")
    weekday = now.weekday()
    time_str = now.strftime("%H:%M")
    is_session = weekday < 5 and (
        "09:30" <= time_str <= "11:30" or "13:00" <= time_str <= "15:00"
    )
    if is_session:
        status = "A股交易中"
        note = "分时和行情会随着刷新更新。"
    elif weekday >= 5:
        status = "A股休市中"
        note = "当前为周末，页面展示最新可得交易日数据。"
    else:
        status = "A股未开盘/已收盘"
        note = "当前不是交易时段，页面展示最新可得交易日数据。"
    return {"now": current, "status": status, "note": note}


def _a_share_universe_cache_path() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"a_share_universe_v{A_SHARE_UNIVERSE_CACHE_VERSION}.pkl"


def _normalize_a_share_universe_frame(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=["symbol", "name", "name_normalized"])

    normalized = df.copy().rename(columns={"code": "symbol", "stock_code": "symbol", "stock_name": "name"})
    if "symbol" not in normalized.columns:
        return pd.DataFrame(columns=["symbol", "name", "name_normalized"])
    if "name" not in normalized.columns:
        normalized["name"] = normalized["symbol"]

    normalized["symbol"] = (
        normalized["symbol"]
        .astype(str)
        .str.extract(r"(\d{6})", expand=False)
        .fillna("")
        .astype(str)
    )
    normalized = normalized[normalized["symbol"].str.fullmatch(r"\d{6}", na=False)].copy()
    if normalized.empty:
        return pd.DataFrame(columns=["symbol", "name", "name_normalized"])

    normalized["name"] = normalized["name"].astype(str)
    normalized["name_normalized"] = normalized["name"].map(normalize_search_text)
    keep_cols = ["symbol", "name", "name_normalized"]
    return normalized[keep_cols].drop_duplicates("symbol", keep="first").sort_values("symbol").reset_index(drop=True)


def _read_a_share_universe_disk_cache() -> tuple[pd.DataFrame | None, dict]:
    cache_path = _a_share_universe_cache_path()
    if not cache_path.exists():
        return None, {}
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None, {}

    meta = payload.get("meta", {})
    df = payload.get("data")
    if meta.get("cache_version") != A_SHARE_UNIVERSE_CACHE_VERSION:
        return None, {}
    normalized = _normalize_a_share_universe_frame(df)
    if normalized.empty:
        return None, meta
    return normalized, meta


def _write_a_share_universe_disk_cache(df: pd.DataFrame, *, source: str) -> None:
    normalized = _normalize_a_share_universe_frame(df)
    if normalized.empty:
        return

    payload = {
        "meta": {
            "cache_version": A_SHARE_UNIVERSE_CACHE_VERSION,
            "row_count": int(len(normalized)),
            "source": str(source or ""),
            "cached_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "data": normalized,
    }
    with _a_share_universe_cache_path().open("wb") as handle:
        pickle.dump(payload, handle)


def _build_static_a_share_universe() -> pd.DataFrame:
    return _normalize_a_share_universe_frame(
        pd.DataFrame([{"symbol": symbol, "name": name} for symbol, name in FALLBACK_WATCHLIST.items()])
    )


def _universe_invalid_name_mask(df: pd.DataFrame) -> pd.Series:
    if not isinstance(df, pd.DataFrame) or df.empty or "symbol" not in df.columns:
        return pd.Series(dtype=bool)
    symbols = df["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    names = (
        df["name"].fillna("").astype(str).str.strip()
        if "name" in df.columns
        else pd.Series("", index=df.index, dtype=str)
    )
    return names.eq("") | names.eq(symbols) | names.str.fullmatch(r"\d{6}", na=False)


def _fill_universe_names_from_fallback(primary: pd.DataFrame, fallback: pd.DataFrame | None) -> pd.DataFrame:
    if not isinstance(primary, pd.DataFrame) or primary.empty:
        return pd.DataFrame()
    frame = _normalize_a_share_universe_frame(primary)
    if frame.empty or not isinstance(fallback, pd.DataFrame) or fallback.empty:
        return frame
    fallback_frame = _normalize_a_share_universe_frame(fallback)
    if fallback_frame.empty or not {"symbol", "name"}.issubset(fallback_frame.columns):
        return frame
    fallback_frame = fallback_frame.loc[~_universe_invalid_name_mask(fallback_frame)].copy()
    if fallback_frame.empty:
        return frame
    name_map = fallback_frame.drop_duplicates("symbol").set_index("symbol")["name"]
    invalid_mask = _universe_invalid_name_mask(frame)
    repaired = frame["symbol"].map(name_map).fillna("").astype(str).str.strip()
    frame.loc[invalid_mask & repaired.ne(""), "name"] = repaired[invalid_mask & repaired.ne("")]
    frame["name_normalized"] = frame["name"].map(normalize_search_text)
    return frame


def fetch_a_share_universe() -> pd.DataFrame:
    using_default_cache_dir = Path(CACHE_DIR).resolve() == (PROJECT_ROOT / ".cache").resolve()
    cached_df, _ = _read_a_share_universe_disk_cache()
    active_source = _active_database_source()
    if using_default_cache_dir and active_source is not None:
        try:
            database_universe = _normalize_a_share_universe_frame(active_source.fetch_universe())
        except Exception:
            database_universe = pd.DataFrame()
        if not database_universe.empty:
            database_universe = _fill_universe_names_from_fallback(database_universe, cached_df)
            invalid_ratio = float(_universe_invalid_name_mask(database_universe).mean()) if not database_universe.empty else 1.0
            if invalid_ratio < 0.25 or cached_df is None or cached_df.empty:
                return database_universe

    last_error: Exception | None = None
    if ak is not None:
        for attempt in range(2):
            try:
                live_df = _normalize_a_share_universe_frame(ak.stock_info_a_code_name())
                if not live_df.empty:
                    _write_a_share_universe_disk_cache(live_df, source="akshare")
                    return live_df
                break
            except Exception as exc:
                last_error = exc
                if attempt + 1 < 2:
                    time.sleep(0.35 * float(attempt + 1))
    if cached_df is not None and not cached_df.empty:
        return cached_df.copy()

    try:
        tushare_df = _normalize_a_share_universe_frame(fetch_tushare_stock_basic())
        if not tushare_df.empty:
            _write_a_share_universe_disk_cache(tushare_df, source="tushare")
            return tushare_df
    except Exception as exc:
        if last_error is None:
            last_error = exc

    static_df = _build_static_a_share_universe()
    if not static_df.empty:
        return static_df

    if last_error is not None:
        raise last_error
    return pd.DataFrame(columns=["symbol", "name", "name_normalized"])
    """
        .str.replace("Ａ", "A", regex=False)
        .str.replace("Ｂ", "B", regex=False)
    )
    """



@lru_cache(maxsize=2)
def fetch_tushare_stock_basic() -> pd.DataFrame:
    df = _call_tushare_api(
        "stock_basic",
        params={"exchange": "", "list_status": "L"},
        fields="ts_code,symbol,name,area,industry,market,list_date,delist_date,list_status",
    )
    if df.empty:
        return df
    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["name"] = df["name"].astype(str)
    df["industry"] = (df["industry"] if "industry" in df.columns else pd.Series("", index=df.index)).fillna("").astype(str)
    df["market"] = (df["market"] if "market" in df.columns else pd.Series("", index=df.index)).fillna("").astype(str)
    df["list_status"] = (df["list_status"] if "list_status" in df.columns else pd.Series("L", index=df.index)).fillna("L").astype(str)
    df["list_date"] = (df["list_date"] if "list_date" in df.columns else pd.Series("", index=df.index)).fillna("").astype(str)
    df["delist_date"] = (df["delist_date"] if "delist_date" in df.columns else pd.Series("", index=df.index)).fillna("").astype(str)
    df["name_normalized"] = df["name"].map(normalize_search_text)
    return df.sort_values("symbol").reset_index(drop=True)


@lru_cache(maxsize=2)
def fetch_tushare_stock_basic_all_statuses() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    fields = "ts_code,symbol,name,area,industry,market,list_date,delist_date,list_status"
    for status in ("L", "D", "P"):
        try:
            frame = _call_tushare_api(
                "stock_basic",
                params={"exchange": "", "list_status": status},
                fields=fields,
            )
        except Exception:
            frame = pd.DataFrame()
        if frame.empty:
            try:
                frame = _call_tushare_api(
                    "stock_basic",
                    params={"exchange": "", "list_status": status},
                    fields="ts_code,symbol,name,area,industry,market,list_date",
                )
            except Exception:
                frame = pd.DataFrame()
            if not frame.empty:
                frame["list_status"] = status
                frame["delist_date"] = ""
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return fetch_tushare_stock_basic().copy()
    merged = pd.concat(frames, ignore_index=True)
    merged["symbol"] = merged["symbol"].astype(str).str.zfill(6)
    merged["name"] = merged["name"].astype(str)
    merged["industry"] = (merged["industry"] if "industry" in merged.columns else pd.Series("", index=merged.index)).fillna("").astype(str)
    merged["market"] = (merged["market"] if "market" in merged.columns else pd.Series("", index=merged.index)).fillna("").astype(str)
    merged["list_status"] = (merged["list_status"] if "list_status" in merged.columns else pd.Series("", index=merged.index)).fillna("").astype(str)
    merged["list_date"] = (merged["list_date"] if "list_date" in merged.columns else pd.Series("", index=merged.index)).fillna("").astype(str)
    merged["delist_date"] = (merged["delist_date"] if "delist_date" in merged.columns else pd.Series("", index=merged.index)).fillna("").astype(str)
    merged["name_normalized"] = merged["name"].map(normalize_search_text)
    return merged.drop_duplicates("symbol", keep="first").sort_values("symbol").reset_index(drop=True)


def _parse_tushare_date_series(values: pd.Series | object) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values
    else:
        series = pd.Series(values)
    text = series.fillna("").astype(str).str.replace(r"\D", "", regex=True)
    text = text.where(text.str.len() == 8, "")
    return pd.to_datetime(text, format="%Y%m%d", errors="coerce")


def filter_point_in_time_a_share_universe(stock_basic: pd.DataFrame, market_data_date: str | None) -> pd.DataFrame:
    if not isinstance(stock_basic, pd.DataFrame) or stock_basic.empty:
        return pd.DataFrame()
    frame = stock_basic.copy()
    if "symbol" not in frame.columns:
        return pd.DataFrame()
    if not market_data_date:
        return frame.copy()
    market_ts = pd.to_datetime(str(market_data_date), errors="coerce")
    if pd.isna(market_ts):
        return frame.copy()
    list_dates = _parse_tushare_date_series(frame.get("list_date", pd.Series("", index=frame.index)))
    delist_dates = _parse_tushare_date_series(frame.get("delist_date", pd.Series("", index=frame.index)))
    status = frame.get("list_status", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper()
    active_mask = (
        (list_dates.isna() | (list_dates <= market_ts))
        & (delist_dates.isna() | (delist_dates > market_ts))
        & status.ne("P")
    )
    return frame.loc[active_mask].copy()


def filter_historical_a_share_universe_window(
    stock_basic: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    if not isinstance(stock_basic, pd.DataFrame) or stock_basic.empty:
        return pd.DataFrame()
    frame = stock_basic.copy()
    if "symbol" not in frame.columns:
        return pd.DataFrame()
    start_ts = pd.to_datetime(str(start_date), errors="coerce") if start_date else pd.NaT
    end_ts = pd.to_datetime(str(end_date), errors="coerce") if end_date else pd.NaT
    list_dates = _parse_tushare_date_series(frame.get("list_date", pd.Series("", index=frame.index)))
    delist_dates = _parse_tushare_date_series(frame.get("delist_date", pd.Series("", index=frame.index)))
    status = frame.get("list_status", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper()
    mask = status.ne("P")
    if pd.notna(end_ts):
        mask &= list_dates.isna() | (list_dates <= end_ts)
    if pd.notna(start_ts):
        mask &= delist_dates.isna() | (delist_dates >= start_ts)
    return frame.loc[mask].copy()


@lru_cache(maxsize=8)
def fetch_tushare_recent_trade_dates(end_date: str | None = None, limit: int = 30) -> list[str]:
    active_source = _active_database_source()
    if active_source is not None:
        try:
            dates = active_source.fetch_recent_trade_dates(end_date=end_date, limit=limit)
        except Exception:
            dates = []
        allow_partial = os.getenv("OPENCLAW_DATABASE_ALLOW_PARTIAL", "").strip().lower() in {"1", "true", "yes"}
        if dates and (allow_partial or len(dates) >= int(limit)):
            return dates

    final_end_date = str(end_date or dt.date.today().strftime("%Y%m%d"))
    start_date = (pd.to_datetime(final_end_date, format="%Y%m%d", errors="coerce") - pd.Timedelta(days=60)).strftime("%Y%m%d")
    try:
        df = _call_tushare_api(
            "trade_cal",
            params={"exchange": "", "start_date": start_date, "end_date": final_end_date, "is_open": 1},
            fields="cal_date,is_open",
        )
        if not df.empty and "cal_date" in df.columns:
            dates = sorted(df["cal_date"].astype(str).tolist())
            return dates[-int(limit) :]
    except Exception:
        pass

    # Fallback for tokens without trade_cal permission: probe the daily endpoint
    # backwards until we collect enough open trading dates.
    dates: list[str] = []
    cursor = pd.to_datetime(final_end_date, format="%Y%m%d", errors="coerce")
    remaining_days = max(int(limit) * 6, 30)
    while pd.notna(cursor) and remaining_days > 0 and len(dates) < int(limit):
        candidate = cursor.strftime("%Y%m%d")
        try:
            daily_df = _call_tushare_api(
                "daily",
                params={"trade_date": candidate},
                fields="ts_code,trade_date,close",
            )
        except Exception:
            daily_df = pd.DataFrame()
        if not daily_df.empty:
            dates.append(candidate)
        cursor = cursor - pd.Timedelta(days=1)
        remaining_days -= 1
    return sorted(set(dates))[-int(limit) :]


@lru_cache(maxsize=32)
def fetch_tushare_daily_snapshot(trade_date: str) -> pd.DataFrame:
    active_source = _active_database_source()
    if active_source is not None:
        try:
            snapshot = active_source.fetch_daily_snapshot(trade_date)
        except Exception:
            snapshot = pd.DataFrame()
        if not snapshot.empty:
            return snapshot

    daily_df = _call_tushare_api(
        "daily",
        params={"trade_date": str(trade_date)},
        fields="ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
    )
    if daily_df.empty:
        return daily_df
    try:
        daily_basic_df = _call_tushare_api(
            "daily_basic",
            params={"trade_date": str(trade_date)},
            fields="ts_code,trade_date,turnover_rate,volume_ratio,total_mv,circ_mv",
        )
    except Exception:
        daily_basic_df = pd.DataFrame(columns=["ts_code", "trade_date", "turnover_rate", "volume_ratio", "total_mv", "circ_mv"])
    merged = daily_df.merge(
        daily_basic_df,
        on=["ts_code", "trade_date"],
        how="left",
    )
    try:
        stock_basic = fetch_tushare_stock_basic()[["ts_code", "symbol", "name", "industry", "market"]]
    except Exception:
        stock_basic = pd.DataFrame({
            "ts_code": merged["ts_code"].astype(str),
            "symbol": merged["ts_code"].astype(str).str.split(".").str[0],
            "name": merged["ts_code"].astype(str).str.split(".").str[0],
            "industry": "",
            "market": "",
        }).drop_duplicates("ts_code")
    merged = merged.merge(stock_basic, on="ts_code", how="left")
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
        "turnover_rate",
        "volume_ratio",
        "total_mv",
        "circ_mv",
    ]
    merged = _coerce_numeric(merged, numeric_cols)
    merged["trade_date"] = pd.to_datetime(merged["trade_date"], format="%Y%m%d", errors="coerce")
    merged["symbol"] = merged["symbol"].fillna(merged["ts_code"].astype(str).str.split(".").str[0]).astype(str).str.zfill(6)
    merged["name"] = merged["name"].fillna(merged["symbol"]).astype(str)
    merged["industry"] = merged["industry"].fillna("").astype(str)
    merged["market"] = merged["market"].fillna("").astype(str)
    merged["amount"] = merged["amount"].fillna(0.0) * 1000
    return merged.sort_values("symbol").reset_index(drop=True)


@lru_cache(maxsize=8)
def fetch_tushare_daily_window(end_date: str | None = None, window: int = 20) -> pd.DataFrame:
    trade_dates = fetch_tushare_recent_trade_dates(end_date=end_date, limit=max(int(window), 2))
    if not trade_dates:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for trade_date in trade_dates:
        frame = fetch_tushare_daily_snapshot(trade_date)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    return merged


def search_a_share_universe(universe_df: pd.DataFrame, query: str, limit: int = 30) -> pd.DataFrame:
    if universe_df.empty:
        return universe_df
    clean_query = query.strip()
    if not clean_query:
        return universe_df.head(limit).copy()
    normalized_text = clean_query.replace(" ", "").upper()
    normalized_text = normalize_search_text(clean_query)
    normalized_symbol = try_normalize_symbol(clean_query)
    symbol_series = universe_df["symbol"].astype(str)
    name_series = universe_df["name"].astype(str)
    normalized_name_series = universe_df["name_normalized"].astype(str).str.upper()
    normalized_name_series = universe_df["name_normalized"].astype(str)

    mask = (
        symbol_series.str.contains(clean_query, regex=False, na=False)
        | name_series.str.contains(clean_query, regex=False, na=False)
        | normalized_name_series.str.contains(normalized_text, regex=False, na=False)
    )
    if normalized_symbol:
        mask = mask | symbol_series.str.contains(normalized_symbol, regex=False, na=False)

    result = universe_df.loc[mask].copy()
    exact_code = result["symbol"] == (normalized_symbol or clean_query)
    prefix_code = result["symbol"].str.startswith(normalized_symbol or clean_query, na=False)
    exact_name = result["name"] == clean_query
    exact_normalized_name = result["name_normalized"].str.upper() == normalized_text
    exact_normalized_name = result["name_normalized"].astype(str) == normalized_text
    result["_rank"] = 3
    result.loc[prefix_code, "_rank"] = 2
    result.loc[exact_name, "_rank"] = 1
    result.loc[exact_normalized_name, "_rank"] = 1
    result.loc[exact_code, "_rank"] = 0
    result = result.sort_values(["_rank", "symbol"]).drop(columns="_rank")
    return result.head(limit).reset_index(drop=True)


def normalize_board_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    normalized = re.sub(r"[ⅠⅡⅢⅣⅤVI]+$", "", name.strip())
    normalized = normalized.replace("（", "(").replace("）", ")")
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _build_ths_headers(referer: str) -> dict[str, str]:
    headers = {
        "Accept": "text/html, */*; q=0.01",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Pragma": "no-cache",
        "Referer": referer,
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "X-Requested-With": "XMLHttpRequest",
    }
    if MiniRacer is None or threading.current_thread() is not threading.main_thread():
        return headers
    try:
        from akshare.stock_feature import stock_fund_flow as ths_fund_flow

        js = MiniRacer()
        js.eval(ths_fund_flow._get_file_content_ths("ths.js"))
        v_code = js.call("v")
    except Exception:
        return headers
    headers["Cookie"] = f"v={v_code}"
    headers["hexin-v"] = v_code
    return headers


def _ths_page_count(html: str) -> int:
    match = re.search(r'<span[^>]*class="page_info"[^>]*>\s*\d+\s*/\s*(\d+)\s*</span>', html)
    if not match:
        return 1
    return max(int(match.group(1)), 1)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    flattened: list[str] = []
    for column in df.columns:
        parts = [str(part).strip() for part in column if str(part).strip() and "Unnamed" not in str(part)]
        flattened.append("-".join(parts) if parts else "value")
    result = df.copy()
    result.columns = flattened
    return result


def _parse_percent_like(value) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text or text == "--":
        return float("nan")
    text = text.rstrip("%")
    return pd.to_numeric(text, errors="coerce")


def _parse_amount_value(value, unit: str = "亿") -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().replace(",", "")
    if not text or text == "--":
        return float("nan")

    multiplier = 1.0
    if text.endswith("%"):
        text = text[:-1]
    if text.endswith("亿元"):
        text = text[:-2]
        multiplier = 1.0 if unit == "亿" else 1e8
    elif text.endswith("亿"):
        text = text[:-1]
        multiplier = 1.0 if unit == "亿" else 1e8
    elif text.endswith("万元"):
        text = text[:-2]
        multiplier = 1.0 if unit == "万" else 1e4
    elif text.endswith("万"):
        text = text[:-1]
        multiplier = 1.0 if unit == "万" else 1e4
    elif text.endswith("元"):
        text = text[:-1]
        multiplier = 1.0

    numeric = pd.to_numeric(text, errors="coerce")
    if pd.isna(numeric):
        return float("nan")
    if unit == "亿":
        if multiplier == 1e8:
            return float(numeric)
        if multiplier == 1e4:
            return float(numeric) / 10000
        return float(numeric)
    if unit == "元":
        if multiplier == 1.0:
            return float(numeric)
        return float(numeric) * multiplier
    if unit == "万":
        if multiplier == 1e4:
            return float(numeric)
        if multiplier == 1e8:
            return float(numeric) * 10000
        return float(numeric) / 10000
    return float(numeric)


@lru_cache(maxsize=16)
def _fetch_ths_board_flow_table(kind: str, period: str) -> pd.DataFrame:
    path_key = {"industry": "hyzjl", "concept": "gnzjl"}[kind]
    period_path = THS_FUND_FLOW_PERIOD_PATHS.get(period, THS_FUND_FLOW_PERIOD_PATHS["即时"])
    referer = THS_FUND_FLOW_REFERERS[kind]
    headers = _build_ths_headers(referer)
    first_url = f"http://data.10jqka.com.cn/funds/{path_key}/{period_path.format(page=1)}"
    response = requests.get(first_url, headers=headers, timeout=10)
    response.raise_for_status()
    response.encoding = "gbk"
    page_count = _ths_page_count(response.text)

    frames: list[pd.DataFrame] = []
    for page in range(1, page_count + 1):
        url = f"http://data.10jqka.com.cn/funds/{path_key}/{period_path.format(page=page)}"
        page_response = response if page == 1 else requests.get(url, headers=headers, timeout=10)
        if page != 1:
            page_response.raise_for_status()
            page_response.encoding = "gbk"
        try:
            page_frame = pd.read_html(StringIO(page_response.text))[0]
        except ValueError:
            continue
        frames.append(page_frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _normalize_ths_board_flow_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    view = _flatten_columns(df).copy()
    column_count = len(view.columns)
    if column_count >= 11:
        view = view.iloc[:, :11].copy()
        view.columns = [
            "rank",
            "sector_name",
            "sector_index",
            "change_pct",
            "inflow",
            "outflow",
            "net_inflow",
            "company_count",
            "leader",
            "leader_change_pct",
            "leader_price",
        ]
    elif column_count >= 8:
        view = view.iloc[:, :8].copy()
        view.columns = [
            "rank",
            "sector_name",
            "company_count",
            "sector_index",
            "change_pct",
            "inflow",
            "outflow",
            "net_inflow",
        ]
        view["leader"] = ""
        view["leader_change_pct"] = float("nan")
        view["leader_price"] = float("nan")
    else:
        return pd.DataFrame()

    view["sector_name"] = view["sector_name"].astype(str).str.strip()
    view["sector_index"] = pd.to_numeric(view["sector_index"], errors="coerce")
    view["company_count"] = pd.to_numeric(view["company_count"], errors="coerce")
    view["change_pct"] = view["change_pct"].map(_parse_percent_like)
    view["leader_change_pct"] = view["leader_change_pct"].map(_parse_percent_like)
    for column in ["inflow", "outflow", "net_inflow"]:
        view[column] = view[column].map(lambda value: _parse_amount_value(value, unit="亿"))
    view["leader_price"] = pd.to_numeric(view["leader_price"], errors="coerce")
    view["sector_name_normalized"] = view["sector_name"].map(normalize_board_name)
    return view.dropna(subset=["sector_name"]).reset_index(drop=True)


def _extract_ths_profile_value(text: str, label: str) -> str | None:
    if label not in text:
        return None
    if "：" in text:
        _, value = text.split("：", 1)
    elif ":" in text:
        _, value = text.split(":", 1)
    else:
        return None
    value = value.replace("\xa0", " ").strip()
    if label == "所属申万行业" and "—" in value:
        value = value.split("—")[-1].strip()
    return value or None


def _fetch_ths_basic_profile(symbol: str) -> dict[str, str]:
    url = f"http://basic.10jqka.com.cn/{symbol}/company.html"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    response.raise_for_status()
    response.encoding = "gbk"
    tables = pd.read_html(StringIO(response.text))

    profile = {"股票代码": symbol, "股票简称": symbol, "行业": "未知"}
    title_match = re.search(r"<title>\s*([^<(]+)\(", response.text)
    if title_match:
        profile["股票简称"] = title_match.group(1).strip()

    if tables:
        summary = tables[0].copy()
        for value in summary.astype(str).stack():
            text = str(value).strip()
            if not text or text == "nan":
                continue
            if parsed := _extract_ths_profile_value(text, "公司名称"):
                profile["公司名称"] = parsed
            elif parsed := _extract_ths_profile_value(text, "所属申万行业"):
                profile["行业"] = parsed
            elif parsed := _extract_ths_profile_value(text, "所属地域"):
                profile["所属地域"] = parsed

    try:
        business_df = ak.stock_zyjs_ths(symbol=symbol)
    except Exception:
        business_df = pd.DataFrame()
    if not business_df.empty:
        record = business_df.iloc[0].to_dict()
        for key in ["主营业务", "产品类型", "产品名称", "经营范围"]:
            value = record.get(key)
            if value is not None and str(value).strip():
                profile[key] = str(value).strip()
    return profile


def _parse_stockpage_fund_flow_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    view = _flatten_columns(df).copy()
    rename_map = {
        "日期-日期": "日期",
        "收盘价-收盘价": "收盘价",
        "涨跌幅-涨跌幅": "涨跌幅",
        "资金净流入-资金净流入": "资金净流入",
        "5日主力净额-5日主力净额": "5日主力净额",
        "大单(主力)-净额": "主力净流入-净额",
        "大单(主力)-净占比": "主力净流入-净占比",
        "中单-净额": "中单净流入-净额",
        "中单-净占比": "中单净流入-净占比",
        "小单-净额": "小单净流入-净额",
        "小单-净占比": "小单净流入-净占比",
    }
    view = view.rename(columns=rename_map)
    required = {"日期", "收盘价", "涨跌幅", "主力净流入-净额", "主力净流入-净占比"}
    if not required.issubset(view.columns):
        return pd.DataFrame()

    view["日期"] = pd.to_datetime(view["日期"], format="%Y%m%d", errors="coerce")
    view["收盘价"] = pd.to_numeric(view["收盘价"], errors="coerce")
    view["涨跌幅"] = view["涨跌幅"].map(_parse_percent_like)
    for column in ["资金净流入", "5日主力净额", "主力净流入-净额", "中单净流入-净额", "小单净流入-净额"]:
        if column in view.columns:
            view[column] = pd.to_numeric(view[column], errors="coerce") * 10000
    for column in ["主力净流入-净占比", "中单净流入-净占比", "小单净流入-净占比"]:
        if column in view.columns:
            view[column] = view[column].map(_parse_percent_like)
    view = view.dropna(subset=["日期", "收盘价", "主力净流入-净额"]).sort_values("日期", ascending=False)
    return view.reset_index(drop=True)


def _fetch_stockpage_fund_flow_table(symbol: str) -> pd.DataFrame:
    url = f"https://stockpage.10jqka.com.cn/{symbol}/funds/"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    response.raise_for_status()
    response.encoding = "utf-8"
    tables = pd.read_html(StringIO(response.text))
    for table in tables:
        parsed = _parse_stockpage_fund_flow_table(table)
        if not parsed.empty:
            return parsed
    return pd.DataFrame()


def fetch_stock_profile(symbol: str) -> dict[str, str]:
    ensure_akshare()
    clean_symbol = normalize_symbol(symbol)
    try:
        df = ak.stock_individual_info_em(symbol=clean_symbol)
    except Exception:
        try:
            return _fetch_ths_basic_profile(clean_symbol)
        except Exception:
            return {"股票代码": clean_symbol, "股票简称": clean_symbol, "行业": "未知"}
    if df.empty:
        try:
            return _fetch_ths_basic_profile(clean_symbol)
        except Exception:
            return {"股票代码": clean_symbol, "股票简称": clean_symbol, "行业": "未知"}
    profile = {str(row["item"]): str(row["value"]) for _, row in df.iterrows()}
    profile.setdefault("股票代码", clean_symbol)
    profile.setdefault("股票简称", clean_symbol)
    profile.setdefault("行业", "未知")
    return profile


def fetch_stock_news(symbol: str, limit: int = 10) -> pd.DataFrame:
    ensure_akshare()
    clean_symbol = normalize_symbol(symbol)
    try:
        df = ak.stock_news_em(symbol=clean_symbol)
    except Exception:
        return pd.DataFrame(columns=["关键词", "新闻标题", "新闻内容", "发布时间", "文章来源", "新闻链接"])
    if df.empty:
        return df
    df = df.copy()
    if "发布时间" in df.columns:
        df["发布时间"] = pd.to_datetime(df["发布时间"], errors="coerce")
        df = df.sort_values("发布时间", ascending=False)
    return df.head(limit).reset_index(drop=True)


def _rename_flow_df(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "行业": "sector_name",
        "行业指数": "sector_index",
        "行业-涨跌幅": "change_pct",
        "流入资金": "inflow",
        "流出资金": "outflow",
        "净额": "net_inflow",
        "公司家数": "company_count",
        "领涨股": "leader",
        "领涨股-涨跌幅": "leader_change_pct",
        "当前价": "leader_price",
        "概念": "sector_name",
    }
    renamed = df.rename(columns=rename_map).copy()
    numeric_cols = [
        "sector_index",
        "change_pct",
        "inflow",
        "outflow",
        "net_inflow",
        "company_count",
        "leader_change_pct",
        "leader_price",
    ]
    return _coerce_numeric(renamed, numeric_cols)


def fetch_industry_fund_flow(period: str = "即时") -> pd.DataFrame:
    ensure_akshare()
    try:
        df = ak.stock_fund_flow_industry(symbol=period)
    except Exception:
        try:
            return _normalize_ths_board_flow_table(_fetch_ths_board_flow_table("industry", period))
        except Exception:
            return pd.DataFrame()
    if df.empty:
        try:
            return _normalize_ths_board_flow_table(_fetch_ths_board_flow_table("industry", period))
        except Exception:
            return df
    normalized = _rename_flow_df(df)
    normalized["sector_name_normalized"] = normalized["sector_name"].map(normalize_board_name)
    if normalized.empty or normalized["net_inflow"].isna().all():
        try:
            return _normalize_ths_board_flow_table(_fetch_ths_board_flow_table("industry", period))
        except Exception:
            return normalized
    return normalized


def fetch_concept_fund_flow(period: str = "即时") -> pd.DataFrame:
    ensure_akshare()
    try:
        df = ak.stock_fund_flow_concept(symbol=period)
    except Exception:
        try:
            return _normalize_ths_board_flow_table(_fetch_ths_board_flow_table("concept", period))
        except Exception:
            return pd.DataFrame()
    if df.empty:
        try:
            return _normalize_ths_board_flow_table(_fetch_ths_board_flow_table("concept", period))
        except Exception:
            return df
    normalized = _rename_flow_df(df)
    normalized["sector_name_normalized"] = normalized["sector_name"].map(normalize_board_name)
    if normalized.empty or normalized["net_inflow"].isna().all():
        try:
            return _normalize_ths_board_flow_table(_fetch_ths_board_flow_table("concept", period))
        except Exception:
            return normalized
    return normalized


def market_from_symbol(symbol: str) -> str:
    clean_symbol = normalize_symbol(symbol)
    if clean_symbol.startswith(("600", "601", "603", "605", "688")):
        return "sh"
    if clean_symbol.startswith(("000", "001", "002", "003", "300", "301")):
        return "sz"
    if clean_symbol.startswith(("430", "830", "831", "832", "833", "834", "835", "836", "837", "838", "839", "870", "871", "872", "873", "874", "875", "876", "877", "878", "879")):
        return "bj"
    return "sh"


def fetch_stock_main_fund_flow(symbol: str, limit: int = 10) -> pd.DataFrame:
    ensure_akshare()
    clean_symbol = normalize_symbol(symbol)
    market = market_from_symbol(clean_symbol)
    try:
        df = ak.stock_individual_fund_flow(stock=clean_symbol, market=market)
    except Exception:
        try:
            return _fetch_stockpage_fund_flow_table(clean_symbol).head(limit).reset_index(drop=True)
        except Exception:
            return pd.DataFrame()
    if df.empty:
        try:
            return _fetch_stockpage_fund_flow_table(clean_symbol).head(limit).reset_index(drop=True)
        except Exception:
            return df
    df = df.copy()
    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
        df = df.sort_values("日期", ascending=False)
    return df.head(limit).reset_index(drop=True)


def fetch_macro_calendar(limit: int = 10) -> pd.DataFrame:
    ensure_akshare()
    try:
        df = ak.news_economic_baidu()
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df = df.copy()
    if "日期" in df.columns and "时间" in df.columns:
        dt_series = pd.to_datetime(
            df["日期"].astype(str) + " " + df["时间"].astype(str),
            errors="coerce",
        )
        df["datetime"] = dt_series
        df = df.sort_values("datetime")
    return df.tail(limit).reset_index(drop=True)
