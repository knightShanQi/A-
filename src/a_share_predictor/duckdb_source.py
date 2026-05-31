from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import database_source
from .duckdb_store import (
    DEFAULT_CALENDAR_TABLE,
    DEFAULT_INTRADAY_TABLE,
    DEFAULT_ROW_TABLE,
    DEFAULT_SERIES_TABLE,
    connect_duckdb,
    duckdb_path,
)


ENABLED_SOURCE_VALUES = {"duckdb", "localduckdb", "local-duckdb"}
STORAGE_MODES = {"row", "series"}


def is_enabled() -> bool:
    database_source.load_env_file()
    source = (
        os.getenv("OPENCLAW_MARKET_DATA_SOURCE")
        or os.getenv("A_SHARE_MARKET_DATA_SOURCE")
        or os.getenv("MARKET_DATA_SOURCE")
        or ""
    ).strip().lower()
    return source in ENABLED_SOURCE_VALUES


def _safe_table_name(table: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table or ""):
        raise ValueError(f"unsafe DuckDB table name: {table}")
    return table


def _table_name() -> str:
    default_table = DEFAULT_SERIES_TABLE if _storage_mode() == "series" else DEFAULT_ROW_TABLE
    return _safe_table_name(os.getenv("OPENCLAW_DAILY_PRICE_TABLE", default_table).strip() or default_table)


def _calendar_table_name() -> str:
    return _safe_table_name(os.getenv("OPENCLAW_TRADE_CALENDAR_TABLE", DEFAULT_CALENDAR_TABLE).strip() or DEFAULT_CALENDAR_TABLE)


def _intraday_table_name() -> str:
    return _safe_table_name(os.getenv("OPENCLAW_INTRADAY_BARS_TABLE", DEFAULT_INTRADAY_TABLE).strip() or DEFAULT_INTRADAY_TABLE)


def _storage_mode() -> str:
    mode = (os.getenv("OPENCLAW_DAILY_PRICE_STORAGE") or "row").strip().lower()
    return mode if mode in STORAGE_MODES else "row"


def _query_frame(connection, query: str, params: Iterable[object] = ()) -> pd.DataFrame:
    cursor = connection.execute(query, list(params))
    rows = cursor.fetchall()
    columns = [description[0] for description in cursor.description or []]
    return pd.DataFrame(rows, columns=columns)


def fetch_daily_history(symbol: str, start_date: str = "20220101", end_date: str | None = None) -> pd.DataFrame:
    clean_symbol = database_source._normalize_symbol(symbol)
    start_value = database_source._date_value(start_date)
    end_value = database_source._date_value(end_date, dt.date.today().strftime("%Y%m%d"))
    if _storage_mode() == "row":
        with connect_duckdb(read_only=True) as connection:
            frame = _query_frame(
                connection,
                f"""
                select symbol, name, trade_date, open, high, low, close, pre_close,
                       change, pct_chg, volume, amount, turnover_rate
                from {_table_name()}
                where symbol = ?
                  and (? is null or trade_date >= cast(? as date))
                  and (? is null or trade_date <= cast(? as date))
                order by trade_date
                """,
                (clean_symbol, start_value, start_value, end_value, end_value),
            )
        return database_source.normalize_daily_history_frame(frame, source="duckdb")
    start_year = start_value.year if start_value is not None else 1990
    end_year = end_value.year if end_value is not None else dt.date.today().year
    with connect_duckdb(read_only=True) as connection:
        frame = _query_frame(
            connection,
            f"""
            select symbol, name, year, dates, open, high, low, close,
                   volume, amount, turnover_rate
            from {_table_name()}
            where symbol = ?
              and year between ? and ?
            order by year
            """,
            (clean_symbol, int(start_year), int(end_year)),
        )
    return database_source.normalize_daily_history_frame(
        database_source._series_rows_to_daily_frame(frame, start=start_value, end=end_value),
        source="duckdb",
    )


def fetch_intraday_history(
    symbol: str,
    trade_date: str | None = None,
    interval_minutes: int = 1,
) -> pd.DataFrame:
    clean_symbol = database_source._normalize_symbol(symbol)
    target_value = database_source._date_value(trade_date) if trade_date else None
    with connect_duckdb(read_only=True) as connection:
        if target_value is None:
            latest = connection.execute(
                f"""
                select max(trade_date)
                from {_intraday_table_name()}
                where symbol = ?
                  and interval_minutes = ?
                """,
                [clean_symbol, int(interval_minutes)],
            ).fetchone()[0]
            target_value = latest
        if target_value is None:
            return pd.DataFrame()
        frame = _query_frame(
            connection,
            f"""
            select symbol, trade_date, bar_time, interval_minutes,
                   open, high, low, close, volume, amount
            from {_intraday_table_name()}
            where symbol = ?
              and trade_date = cast(? as date)
              and interval_minutes = ?
            order by bar_time
            """,
            (clean_symbol, target_value, int(interval_minutes)),
        )
    if frame.empty:
        return frame
    result = frame.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce")
    result["datetime"] = pd.to_datetime(
        result["trade_date"].dt.strftime("%Y-%m-%d") + " " + result["bar_time"].astype(str),
        errors="coerce",
    )
    result = result.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result.attrs["data_source"] = "duckdb"
    return result


def fetch_recent_trade_dates(end_date: str | None = None, limit: int = 30) -> list[str]:
    end_value = database_source._date_value(end_date, dt.date.today().strftime("%Y%m%d"))
    with connect_duckdb(read_only=True) as connection:
        frame = _query_frame(
            connection,
            f"""
            select trade_date
            from {_calendar_table_name()}
            where (? is null or trade_date <= cast(? as date))
            order by trade_date desc
            limit ?
            """,
            (end_value, end_value, int(limit)),
        )
    if frame.empty:
        return []
    return sorted(database_source._format_tushare_date(value) for value in frame["trade_date"] if database_source._format_tushare_date(value))


def fetch_daily_snapshot(trade_date: str) -> pd.DataFrame:
    target_value = database_source._date_value(trade_date)
    if target_value is None:
        return pd.DataFrame()
    if _storage_mode() == "row":
        with connect_duckdb(read_only=True) as connection:
            frame = _query_frame(
                connection,
                f"""
                select symbol, name, trade_date, open, high, low, close, pre_close,
                       change, pct_chg, volume, amount, turnover_rate
                from {_table_name()}
                where trade_date = cast(? as date)
                order by symbol
                """,
                (target_value,),
            )
        if frame.empty:
            return frame
        result = _normalize_snapshot_frame(frame)
        result.attrs["data_source"] = "duckdb"
        return result
    with connect_duckdb(read_only=True) as connection:
        series_frame = _query_frame(
            connection,
            f"""
            select symbol, name, year, dates, open, high, low, close,
                   volume, amount, turnover_rate
            from {_table_name()}
            where year = ?
            order by symbol
            """,
            (target_value.year,),
        )
    frame = database_source._series_rows_to_daily_frame(series_frame, start=target_value, end=target_value)
    if frame.empty:
        return frame
    result = _normalize_snapshot_frame(frame)
    result.attrs["data_source"] = "duckdb"
    return result


def _normalize_snapshot_frame(frame: pd.DataFrame) -> pd.DataFrame:
    for column in ["pre_close", "change", "pct_chg"]:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["name"] = frame["name"].where(frame["name"].str.len().gt(0), frame["symbol"])
    frame["industry"] = ""
    frame["market"] = frame["symbol"].map(database_source._market_suffix)
    frame["ts_code"] = frame["symbol"] + "." + frame["market"]
    for column in ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "volume", "amount", "turnover_rate"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["vol"] = pd.to_numeric(frame.get("volume"), errors="coerce")
    close = pd.to_numeric(frame.get("close"), errors="coerce")
    pre_close = pd.to_numeric(frame.get("pre_close"), errors="coerce")
    base = pre_close.where(pre_close.ne(0))
    frame["change"] = frame["change"].where(frame["change"].notna(), close - pre_close)
    frame["pct_chg"] = frame["pct_chg"].where(frame["pct_chg"].notna(), (close / base - 1.0) * 100)
    return frame.reset_index(drop=True)


def fetch_daily_window(end_date: str | None = None, window: int = 20) -> pd.DataFrame:
    dates = fetch_recent_trade_dates(end_date=end_date, limit=max(int(window), 2))
    frames = [fetch_daily_snapshot(trade_date) for trade_date in dates]
    valid = [frame for frame in frames if not frame.empty]
    if not valid:
        return pd.DataFrame()
    result = pd.concat(valid, ignore_index=True).sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    result.attrs["data_source"] = "duckdb"
    return result


def fetch_universe(limit: int | None = None) -> pd.DataFrame:
    limit_clause = "" if limit is None else "limit ?"
    params: tuple[object, ...] = () if limit is None else (int(limit),)
    if _storage_mode() == "row":
        with connect_duckdb(read_only=True) as connection:
            frame = _query_frame(
                connection,
                f"""
                select symbol, max(nullif(name, '')) as name, max(trade_date) as latest_trade_date
                from {_table_name()}
                group by symbol
                order by symbol
                {limit_clause}
                """,
                params,
            )
        if frame.empty:
            return pd.DataFrame(columns=["symbol", "name"])
        frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
        frame["name"] = frame["name"].fillna("").astype(str)
        frame["name"] = frame["name"].where(frame["name"].str.len().gt(0), frame["symbol"])
        result = frame[["symbol", "name"]].reset_index(drop=True)
        result.attrs["data_source"] = "duckdb"
        return result
    with connect_duckdb(read_only=True) as connection:
        frame = _query_frame(
            connection,
            f"""
            select symbol, max(nullif(name, '')) as name, max(year) as latest_year
            from {_table_name()}
            group by symbol
            order by symbol
            {limit_clause}
            """,
            params,
        )
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "name"])
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["name"] = frame["name"].where(frame["name"].str.len().gt(0), frame["symbol"])
    result = frame[["symbol", "name"]].reset_index(drop=True)
    result.attrs["data_source"] = "duckdb"
    return result


def current_path() -> Path:
    return duckdb_path()
