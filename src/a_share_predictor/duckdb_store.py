from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from .database_source import load_env_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DUCKDB_PATH = PROJECT_ROOT / "data" / "openclaw_market_data.duckdb"
DEFAULT_ROW_TABLE = "a_share_daily_prices"
DEFAULT_SERIES_TABLE = "a_share_daily_price_series"
DEFAULT_CALENDAR_TABLE = "a_share_trade_calendar"
DEFAULT_INTRADAY_TABLE = "a_share_intraday_bars"
DEFAULT_STOCK_FUND_FLOW_TABLE = "a_share_stock_fund_flow"
DEFAULT_CALL_AUCTION_PROXY_TABLE = "a_share_call_auction_proxy"
DEFAULT_RESEARCH_BACKTEST_TABLE = "research_market_backtest_runs"
DEFAULT_INTRADAY_RETENTION_DAYS = 365
DEFAULT_INTRADAY_IMPORT_BATCH_FILES = 250
INTRADAY_INTERVALS = (1, 5, 15, 30, 60)
YEARLY_DATE_COL = "\u65e5\u671f"
YEARLY_CLOSE_COL = "\u6536\u76d8\u4ef7"
YEARLY_TURNOVER_COL = "\u6362\u624b\u7387"
DAILY_DATE_COL = "\u4ea4\u6613\u65e5\u671f"
DAILY_NAME_COL = "\u80a1\u7968\u540d\u79f0"
DAILY_CLOSE_COL = "\u6536\u76d8\u4ef7"
DAILY_PCT_CHG_COL = "\u6da8\u8dcc\u5e45"
DAILY_AMOUNT_COL = "\u6210\u4ea4\u989d"
DAILY_TURNOVER_COL = "\u6362\u624b\u7387"
INTRADAY_DATE_COL = "\u65e5\u671f"
INTRADAY_TIME_COL = "\u65f6\u95f4"
INTRADAY_OPEN_COL = "\u5f00\u76d8"
INTRADAY_HIGH_COL = "\u6700\u9ad8"
INTRADAY_LOW_COL = "\u6700\u4f4e"
INTRADAY_CLOSE_COL = "\u6536\u76d8"
INTRADAY_VOLUME_COL = "\u6210\u4ea4\u91cf"
INTRADAY_AMOUNT_COL = "\u6210\u4ea4\u989d"
FUND_DATE_COL = "\u65e5\u671f"
FUND_CLOSE_COL = "\u6536\u76d8\u4ef7"
FUND_CHANGE_PCT_COL = "\u6da8\u8dcc\u5e45"
FUND_NET_INFLOW_COL = "\u8d44\u91d1\u51c0\u6d41\u5165"
FUND_5D_MAIN_NET_COL = "5\u65e5\u4e3b\u529b\u51c0\u989d"
FUND_MAIN_NET_AMOUNT_COL = "\u4e3b\u529b\u51c0\u6d41\u5165-\u51c0\u989d"
FUND_MAIN_NET_RATIO_COL = "\u4e3b\u529b\u51c0\u6d41\u5165-\u51c0\u5360\u6bd4"
FUND_MIDDLE_NET_AMOUNT_COL = "\u4e2d\u5355\u51c0\u6d41\u5165-\u51c0\u989d"
FUND_MIDDLE_NET_RATIO_COL = "\u4e2d\u5355\u51c0\u6d41\u5165-\u51c0\u5360\u6bd4"
FUND_SMALL_NET_AMOUNT_COL = "\u5c0f\u5355\u51c0\u6d41\u5165-\u51c0\u989d"
FUND_SMALL_NET_RATIO_COL = "\u5c0f\u5355\u51c0\u6d41\u5165-\u51c0\u5360\u6bd4"
SERIES_COLUMNS = [
    "symbol",
    "name",
    "year",
    "dates",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover_rate",
    "source_files",
    "source_updated_at",
]


def duckdb_path(path: str | Path | None = None) -> Path:
    load_env_file()
    raw = str(path or os.getenv("OPENCLAW_DUCKDB_PATH") or DEFAULT_DUCKDB_PATH).strip()
    resolved = Path(raw)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def connect_duckdb(path: str | Path | None = None, *, read_only: bool = False):
    import duckdb

    db_path = duckdb_path(path)
    if not read_only:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path), read_only=read_only)


def ensure_schema(connection, *, series_table: str = DEFAULT_SERIES_TABLE, calendar_table: str = DEFAULT_CALENDAR_TABLE) -> None:
    connection.execute(
        f"""
        create table if not exists {series_table} (
            symbol varchar not null,
            name varchar,
            year smallint not null,
            dates date[] not null,
            open real[],
            high real[],
            low real[],
            close real[] not null,
            volume real[],
            amount real[],
            turnover_rate real[],
            source_files varchar[],
            source_updated_at timestamp,
            ingested_at timestamp not null default current_timestamp,
            primary key (symbol, year)
        )
        """
    )
    connection.execute(f"create index if not exists {series_table}_year_idx on {series_table} (year)")
    connection.execute(
        f"""
        create table if not exists {calendar_table} (
            trade_date date primary key,
            ingested_at timestamp not null default current_timestamp
        )
        """
    )


def ensure_row_schema(connection, *, row_table: str = DEFAULT_ROW_TABLE, calendar_table: str = DEFAULT_CALENDAR_TABLE) -> None:
    connection.execute(
        f"""
        create table if not exists {row_table} (
            symbol varchar not null,
            name varchar,
            trade_date date not null,
            open real,
            high real,
            low real,
            close real not null,
            pre_close real,
            change real,
            pct_chg real,
            volume real,
            amount real,
            turnover_rate real,
            source_file varchar,
            ingested_at timestamp not null default current_timestamp
        )
        """
    )
    connection.execute(
        f"""
        create table if not exists {calendar_table} (
            trade_date date primary key,
            ingested_at timestamp not null default current_timestamp
        )
        """
    )


def ensure_row_indexes(connection, *, row_table: str = DEFAULT_ROW_TABLE) -> None:
    connection.execute(f"create index if not exists {row_table}_symbol_date_idx on {row_table} (symbol, trade_date)")
    connection.execute(f"create index if not exists {row_table}_trade_date_idx on {row_table} (trade_date)")


def ensure_intraday_schema(connection, *, intraday_table: str = DEFAULT_INTRADAY_TABLE) -> None:
    connection.execute(
        f"""
        create table if not exists {intraday_table} (
            symbol varchar not null,
            trade_date date not null,
            bar_time time not null,
            interval_minutes smallint not null,
            open real,
            high real,
            low real,
            close real not null,
            volume double,
            amount double,
            source_file varchar,
            ingested_at timestamp not null default current_timestamp
        )
        """
    )


def ensure_intraday_indexes(connection, *, intraday_table: str = DEFAULT_INTRADAY_TABLE) -> None:
    connection.execute(
        f"create index if not exists {intraday_table}_symbol_interval_dt_idx on {intraday_table} (symbol, interval_minutes, trade_date, bar_time)"
    )
    connection.execute(
        f"create index if not exists {intraday_table}_date_interval_idx on {intraday_table} (trade_date, interval_minutes)"
    )


def ensure_stock_fund_flow_schema(connection, *, fund_flow_table: str = DEFAULT_STOCK_FUND_FLOW_TABLE) -> None:
    connection.execute(
        f"""
        create table if not exists {fund_flow_table} (
            symbol varchar not null,
            trade_date date not null,
            close real,
            change_pct real,
            net_inflow double,
            five_day_main_net_amount double,
            main_net_amount double,
            main_net_ratio real,
            middle_net_amount double,
            middle_net_ratio real,
            small_net_amount double,
            small_net_ratio real,
            source varchar,
            ingested_at timestamp not null default current_timestamp
        )
        """
    )


def ensure_stock_fund_flow_indexes(connection, *, fund_flow_table: str = DEFAULT_STOCK_FUND_FLOW_TABLE) -> None:
    connection.execute(f"create index if not exists {fund_flow_table}_symbol_date_idx on {fund_flow_table} (symbol, trade_date)")
    connection.execute(f"create index if not exists {fund_flow_table}_trade_date_idx on {fund_flow_table} (trade_date)")


def ensure_call_auction_proxy_schema(
    connection,
    *,
    auction_table: str = DEFAULT_CALL_AUCTION_PROXY_TABLE,
) -> None:
    connection.execute(
        f"""
        create table if not exists {auction_table} (
            symbol varchar not null,
            trade_date date not null,
            interval_minutes smallint not null,
            open real,
            pre_close real,
            auction_open_gap real,
            first_bar_time time,
            first_bar_ret real,
            first_bar_volume_share real,
            first_bar_amount_share real,
            first_bar_range_pct real,
            source varchar,
            ingested_at timestamp not null default current_timestamp
        )
        """
    )


def ensure_call_auction_proxy_indexes(
    connection,
    *,
    auction_table: str = DEFAULT_CALL_AUCTION_PROXY_TABLE,
) -> None:
    connection.execute(f"create index if not exists {auction_table}_symbol_date_idx on {auction_table} (symbol, trade_date)")
    connection.execute(f"create index if not exists {auction_table}_date_idx on {auction_table} (trade_date)")


def ensure_research_backtest_schema(
    connection,
    *,
    backtest_table: str = DEFAULT_RESEARCH_BACKTEST_TABLE,
) -> None:
    connection.execute(
        f"""
        create table if not exists {backtest_table} (
            run_id varchar primary key,
            created_at timestamp not null default current_timestamp,
            date_from date,
            date_to date,
            horizon_days integer,
            positive_return double,
            strategy_mode varchar,
            top_k integer,
            evaluation_engine varchar,
            annualized_return double,
            max_drawdown double,
            portfolio_trade_count integer,
            summary_json varchar not null,
            summary_path varchar,
            results_path varchar,
            portfolio_nav_path varchar,
            portfolio_trades_path varchar
        )
        """
    )
    connection.execute(f"create index if not exists {backtest_table}_created_idx on {backtest_table} (created_at)")
    connection.execute(f"create index if not exists {backtest_table}_date_idx on {backtest_table} (date_from, date_to)")


def _backtest_run_id(summary: dict[str, object], paths: dict[str, object]) -> str:
    payload = {
        "date_from": summary.get("date_from"),
        "date_to": summary.get("date_to"),
        "horizon_days": summary.get("horizon_days"),
        "positive_return": summary.get("positive_return"),
        "strategy_mode": summary.get("strategy_mode"),
        "top_k": summary.get("top_k"),
        "evaluation_engine": summary.get("evaluation_engine"),
        "summary_path": paths.get("summary_path"),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:24]


def upsert_research_backtest_summary(
    connection,
    *,
    summary: dict[str, object],
    paths: dict[str, object] | None = None,
    backtest_table: str = DEFAULT_RESEARCH_BACKTEST_TABLE,
) -> str:
    ensure_research_backtest_schema(connection, backtest_table=backtest_table)
    clean_paths = dict(paths or {})
    run_id = _backtest_run_id(summary, clean_paths)
    row = (
        run_id,
        pd.to_datetime(summary.get("date_from"), errors="coerce").date()
        if pd.notna(pd.to_datetime(summary.get("date_from"), errors="coerce"))
        else None,
        pd.to_datetime(summary.get("date_to"), errors="coerce").date()
        if pd.notna(pd.to_datetime(summary.get("date_to"), errors="coerce"))
        else None,
        int(summary.get("horizon_days", 0) or 0),
        float(summary.get("positive_return", 0.0) or 0.0),
        str(summary.get("strategy_mode") or ""),
        int(summary.get("top_k", 0) or 0),
        str(summary.get("evaluation_engine") or ""),
        float(summary.get("annualized_return", 0.0) or 0.0),
        float(summary.get("max_drawdown", 0.0) or 0.0),
        int(summary.get("portfolio_trade_count", 0) or 0),
        json.dumps(summary, ensure_ascii=False, sort_keys=True),
        str(clean_paths.get("summary_path") or ""),
        str(clean_paths.get("results_path") or ""),
        str(clean_paths.get("portfolio_nav_path") or ""),
        str(clean_paths.get("portfolio_trades_path") or ""),
    )
    connection.execute(f"delete from {backtest_table} where run_id = ?", [run_id])
    connection.execute(
        f"""
        insert into {backtest_table} (
            run_id, date_from, date_to, horizon_days, positive_return,
            strategy_mode, top_k, evaluation_engine, annualized_return,
            max_drawdown, portfolio_trade_count, summary_json, summary_path,
            results_path, portfolio_nav_path, portfolio_trades_path
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        row,
    )
    return run_id


def _list_or_none(value: object) -> list | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return None
    return [None if pd.isna(item) else item for item in value]


def _timestamp_or_none(value: object) -> dt.datetime | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    stamp = pd.Timestamp(parsed).to_pydatetime()
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return stamp


def _record_tuple(record: dict[str, object]) -> tuple:
    return (
        str(record.get("symbol", "")).zfill(6),
        str(record.get("name") or ""),
        int(record.get("year")),
        _list_or_none(record.get("dates")) or [],
        _list_or_none(record.get("open")),
        _list_or_none(record.get("high")),
        _list_or_none(record.get("low")),
        _list_or_none(record.get("close")) or [],
        _list_or_none(record.get("volume")),
        _list_or_none(record.get("amount")),
        _list_or_none(record.get("turnover_rate")),
        _list_or_none(record.get("source_files")),
        _timestamp_or_none(record.get("source_updated_at")),
    )


def upsert_series_records(connection, records: Iterable[dict[str, object]], *, series_table: str = DEFAULT_SERIES_TABLE) -> int:
    prepared = [_record_tuple(record) for record in records]
    if not prepared:
        return 0
    connection.executemany(
        f"delete from {series_table} where symbol = ? and year = ?",
        [(row[0], row[2]) for row in prepared],
    )
    placeholders = ", ".join(["?"] * len(SERIES_COLUMNS))
    connection.executemany(
        f"""
        insert into {series_table} ({", ".join(SERIES_COLUMNS)})
        values ({placeholders})
        """,
        prepared,
    )
    return len(prepared)


def normalize_stock_fund_flow_frame(symbol: str, frame: pd.DataFrame, *, source: str = "stockpage_10jqka") -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    clean_symbol = re.sub(r"\D", "", str(symbol)).zfill(6)
    normalized = frame.copy()
    rename_map = {
        FUND_DATE_COL: "trade_date",
        FUND_CLOSE_COL: "close",
        FUND_CHANGE_PCT_COL: "change_pct",
        FUND_NET_INFLOW_COL: "net_inflow",
        FUND_5D_MAIN_NET_COL: "five_day_main_net_amount",
        FUND_MAIN_NET_AMOUNT_COL: "main_net_amount",
        FUND_MAIN_NET_RATIO_COL: "main_net_ratio",
        FUND_MIDDLE_NET_AMOUNT_COL: "middle_net_amount",
        FUND_MIDDLE_NET_RATIO_COL: "middle_net_ratio",
        FUND_SMALL_NET_AMOUNT_COL: "small_net_amount",
        FUND_SMALL_NET_RATIO_COL: "small_net_ratio",
    }
    normalized = normalized.rename(columns=rename_map)
    if "trade_date" not in normalized.columns:
        return pd.DataFrame()
    normalized["symbol"] = clean_symbol
    normalized["trade_date"] = pd.to_datetime(normalized["trade_date"], errors="coerce").dt.normalize()
    numeric_columns = [
        "close",
        "change_pct",
        "net_inflow",
        "five_day_main_net_amount",
        "main_net_amount",
        "main_net_ratio",
        "middle_net_amount",
        "middle_net_ratio",
        "small_net_amount",
        "small_net_ratio",
    ]
    for column in numeric_columns:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        else:
            normalized[column] = pd.NA
    normalized["source"] = source
    normalized = normalized.dropna(subset=["symbol", "trade_date"]).copy()
    if normalized.empty:
        return pd.DataFrame()
    return normalized[
        [
            "symbol",
            "trade_date",
            "close",
            "change_pct",
            "net_inflow",
            "five_day_main_net_amount",
            "main_net_amount",
            "main_net_ratio",
            "middle_net_amount",
            "middle_net_ratio",
            "small_net_amount",
            "small_net_ratio",
            "source",
        ]
    ].drop_duplicates(["symbol", "trade_date"], keep="last")


def upsert_stock_fund_flow_frame(
    connection,
    frame: pd.DataFrame,
    *,
    fund_flow_table: str = DEFAULT_STOCK_FUND_FLOW_TABLE,
) -> int:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return 0
    ensure_stock_fund_flow_schema(connection, fund_flow_table=fund_flow_table)
    rows = frame.copy()
    rows["trade_date"] = pd.to_datetime(rows["trade_date"], errors="coerce").dt.date
    rows = rows.dropna(subset=["symbol", "trade_date"]).copy()
    if rows.empty:
        return 0
    temp_table = "tmp_openclaw_stock_fund_flow_import"
    connection.execute(f"drop table if exists {temp_table}")
    connection.register("stock_fund_flow_import_df", rows)
    connection.execute(f"create temp table {temp_table} as select * from stock_fund_flow_import_df")
    connection.execute(
        f"""
        delete from {fund_flow_table}
        where (symbol, trade_date) in (
            select symbol, cast(trade_date as date)
            from {temp_table}
        )
        """
    )
    connection.execute(
        f"""
        insert into {fund_flow_table} (
            symbol, trade_date, close, change_pct, net_inflow,
            five_day_main_net_amount, main_net_amount, main_net_ratio,
            middle_net_amount, middle_net_ratio, small_net_amount,
            small_net_ratio, source
        )
        select symbol, cast(trade_date as date), close, change_pct, net_inflow,
               five_day_main_net_amount, main_net_amount, main_net_ratio,
               middle_net_amount, middle_net_ratio, small_net_amount,
               small_net_ratio, source
        from {temp_table}
        """
    )
    connection.execute(f"drop table if exists {temp_table}")
    ensure_stock_fund_flow_indexes(connection, fund_flow_table=fund_flow_table)
    return int(len(rows))


def rebuild_call_auction_proxy_table(
    connection,
    *,
    price_table: str = DEFAULT_ROW_TABLE,
    intraday_table: str = DEFAULT_INTRADAY_TABLE,
    auction_table: str = DEFAULT_CALL_AUCTION_PROXY_TABLE,
    start_date: str | dt.date,
    end_date: str | dt.date,
    interval_minutes: int = 15,
) -> int:
    ensure_call_auction_proxy_schema(connection, auction_table=auction_table)
    connection.execute(
        f"""
        delete from {auction_table}
        where interval_minutes = ?
          and trade_date >= cast(? as date)
          and trade_date <= cast(? as date)
        """,
        [int(interval_minutes), str(start_date), str(end_date)],
    )
    connection.execute(
        f"""
        insert into {auction_table} (
            symbol, trade_date, interval_minutes, open, pre_close, auction_open_gap,
            first_bar_time, first_bar_ret, first_bar_volume_share,
            first_bar_amount_share, first_bar_range_pct, source
        )
        with daily as (
            select p.symbol,
                   p.trade_date,
                   try_cast(p.open as real) as daily_open,
                   try_cast(p.pre_close as real) as reported_pre_close,
                   try_cast(p.close as real) as close_price,
                   lag(try_cast(p.close as real)) over (
                       partition by p.symbol
                       order by p.trade_date
                   ) as lag_close
            from {price_table} p
            where trade_date >= cast(? as date) - interval 30 day
              and trade_date <= cast(? as date)
        ),
        daily_window as (
            select symbol,
                   trade_date,
                   daily_open,
                   coalesce(reported_pre_close, lag_close) as pre_close
            from daily
            where trade_date >= cast(? as date)
              and trade_date <= cast(? as date)
        ),
        bars as (
            select symbol,
                   trade_date,
                   bar_time,
                   try_cast(open as real) as bar_open,
                   try_cast(high as real) as bar_high,
                   try_cast(low as real) as bar_low,
                   try_cast(close as real) as bar_close,
                   try_cast(volume as double) as bar_volume,
                   try_cast(amount as double) as bar_amount,
                   row_number() over (partition by symbol, trade_date order by bar_time) as rn,
                   sum(try_cast(volume as double)) over (partition by symbol, trade_date) as total_volume,
                   sum(try_cast(amount as double)) over (partition by symbol, trade_date) as total_amount
            from {intraday_table}
            where interval_minutes = ?
              and trade_date >= cast(? as date)
              and trade_date <= cast(? as date)
        ),
        first_bar as (
            select *
            from bars
            where rn = 1
        )
        select d.symbol,
               d.trade_date,
               cast(? as smallint) as interval_minutes,
               coalesce(d.daily_open, f.bar_open) as open,
               d.pre_close,
               cast(coalesce(d.daily_open, f.bar_open) / nullif(d.pre_close, 0) - 1.0 as real) as auction_open_gap,
               f.bar_time as first_bar_time,
               cast(f.bar_close / nullif(f.bar_open, 0) - 1.0 as real) as first_bar_ret,
               cast(f.bar_volume / nullif(f.total_volume, 0) as real) as first_bar_volume_share,
               cast(f.bar_amount / nullif(f.total_amount, 0) as real) as first_bar_amount_share,
               cast((f.bar_high - f.bar_low) / nullif(f.bar_open, 0) as real) as first_bar_range_pct,
               'daily_open_plus_first_intraday_bar' as source
        from daily_window d
        left join first_bar f
          on d.symbol = f.symbol
         and d.trade_date = f.trade_date
        where coalesce(d.daily_open, f.bar_open) is not null
          and d.pre_close is not null
        """,
        [
            str(start_date),
            str(end_date),
            str(start_date),
            str(end_date),
            int(interval_minutes),
            str(start_date),
            str(end_date),
            int(interval_minutes),
        ],
    )
    ensure_call_auction_proxy_indexes(connection, auction_table=auction_table)
    return int(
        connection.execute(
            f"""
            select count(*)
            from {auction_table}
            where interval_minutes = ?
              and trade_date >= cast(? as date)
              and trade_date <= cast(? as date)
            """,
            [int(interval_minutes), str(start_date), str(end_date)],
        ).fetchone()[0]
    )


def rebuild_calendar(connection, *, series_table: str = DEFAULT_SERIES_TABLE, calendar_table: str = DEFAULT_CALENDAR_TABLE) -> int:
    connection.execute(f"delete from {calendar_table}")
    connection.execute(
        f"""
        insert into {calendar_table} (trade_date)
        select distinct unnest(dates) as trade_date
        from {series_table}
        order by trade_date
        """
    )
    return int(connection.execute(f"select count(*) from {calendar_table}").fetchone()[0])


def _postgres_url(value: str | None = None) -> str:
    load_env_file()
    url = (value or os.getenv("DATABASE_URL") or os.getenv("SUPABASE_POSTGRES_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL or SUPABASE_POSTGRES_URL is required to copy Supabase data into DuckDB")
    return url


def sync_from_postgres(
    *,
    postgres_url: str | None = None,
    duckdb_database: str | Path | None = None,
    source_table: str = DEFAULT_SERIES_TABLE,
    target_table: str = DEFAULT_SERIES_TABLE,
    calendar_table: str = DEFAULT_CALENDAR_TABLE,
    replace: bool = False,
    batch_size: int = 1000,
    years: list[int] | None = None,
) -> dict[str, object]:
    import psycopg

    target_path = duckdb_path(duckdb_database)
    url = _postgres_url(postgres_url)
    duck_conn = connect_duckdb(target_path)
    try:
        ensure_schema(duck_conn, series_table=target_table, calendar_table=calendar_table)
        if replace:
            duck_conn.execute(f"delete from {target_table}")
            duck_conn.execute(f"delete from {calendar_table}")

        rows_written = 0
        years_to_copy = years or _postgres_years(url, source_table)
        for year in sorted({int(value) for value in years_to_copy}):
            copied = _copy_postgres_year(
                url,
                duck_conn,
                source_table=source_table,
                target_table=target_table,
                year=year,
                batch_size=int(batch_size),
            )
            rows_written += copied
            print(f"[duckdb] copied {year}: {copied} symbol-year rows", flush=True)
        calendar_rows = rebuild_calendar(duck_conn, series_table=target_table, calendar_table=calendar_table)
        duck_conn.execute("checkpoint")
        return {
            "duckdb_path": str(target_path),
            "series_table": target_table,
            "calendar_table": calendar_table,
            "rows_written": rows_written,
            "calendar_rows": calendar_rows,
            "years": sorted({int(value) for value in years_to_copy}),
        }
    finally:
        duck_conn.close()


def _postgres_years(postgres_url: str, source_table: str) -> list[int]:
    import psycopg

    for attempt in range(1, 4):
        try:
            with psycopg.connect(postgres_url, connect_timeout=30) as pg_conn:
                with pg_conn.cursor() as cursor:
                    cursor.execute(f"select distinct year from {source_table} order by year")
                    return [int(row[0]) for row in cursor.fetchall()]
        except psycopg.Error:
            if attempt >= 3:
                raise
            time.sleep(2.0 * attempt)
    return []


def _copy_postgres_year(
    postgres_url: str,
    duck_conn,
    *,
    source_table: str,
    target_table: str,
    year: int,
    batch_size: int,
) -> int:
    import psycopg

    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            copied = 0
            with psycopg.connect(postgres_url, connect_timeout=30) as pg_conn:
                with pg_conn.cursor() as cursor:
                    cursor.execute(
                        f"""
                        select {", ".join(SERIES_COLUMNS)}
                        from {source_table}
                        where year = %s
                        order by symbol
                        """,
                        (int(year),),
                    )
                    while True:
                        rows = cursor.fetchmany(int(batch_size))
                        if not rows:
                            break
                        records = [dict(zip(SERIES_COLUMNS, row)) for row in rows]
                        copied += upsert_series_records(duck_conn, records, series_table=target_table)
            return copied
        except (psycopg.Error, OSError) as exc:
            last_exc = exc
            if attempt < 3:
                time.sleep(2.0 * attempt)
    if last_exc is not None:
        raise last_exc
    return 0


def parse_years(value: str) -> list[int]:
    years: set[int] = set()
    for part in str(value or "").split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start_year = int(start_text)
            end_year = int(end_text)
            if end_year < start_year:
                start_year, end_year = end_year, start_year
            years.update(range(start_year, end_year + 1))
        else:
            years.add(int(token))
    return sorted(years)


def sync_from_local_baidu_cache(
    *,
    duckdb_database: str | Path | None = None,
    cache_dir: str | Path | None = None,
    target_table: str = DEFAULT_SERIES_TABLE,
    calendar_table: str = DEFAULT_CALENDAR_TABLE,
    years: list[int] | None = None,
    replace: bool = False,
) -> dict[str, object]:
    from . import daily_stock_sync

    target_path = duckdb_path(duckdb_database)
    source_dir = Path(cache_dir or daily_stock_sync.DEFAULT_DOWNLOAD_DIR)
    if not source_dir.is_absolute():
        source_dir = PROJECT_ROOT / source_dir
    years_to_import = years or _local_cache_years(source_dir)

    duck_conn = connect_duckdb(target_path)
    try:
        ensure_schema(duck_conn, series_table=target_table, calendar_table=calendar_table)
        if replace:
            duck_conn.execute(f"delete from {target_table}")
            duck_conn.execute(f"delete from {calendar_table}")

        imported_files = 0
        rows_read = 0
        rows_written = 0
        series_rows = 0
        for year in sorted({int(value) for value in years_to_import}):
            files = _local_year_files(source_dir, year)
            frame, year_imported_files, year_rows_read = daily_stock_sync.load_daily_stock_files(files)
            records = daily_stock_sync.daily_stock_frame_to_series_records(frame)
            year_series_rows = upsert_series_records(duck_conn, records, series_table=target_table)
            imported_files += int(year_imported_files)
            rows_read += int(year_rows_read)
            rows_written += int(len(frame))
            series_rows += int(year_series_rows)
            print(
                f"[duckdb] imported {year}: files={year_imported_files} rows={len(frame)} series={year_series_rows}",
                flush=True,
            )
        calendar_rows = rebuild_calendar(duck_conn, series_table=target_table, calendar_table=calendar_table)
        duck_conn.execute("checkpoint")
        return {
            "duckdb_path": str(target_path),
            "series_table": target_table,
            "calendar_table": calendar_table,
            "years": sorted({int(value) for value in years_to_import}),
            "imported_files": imported_files,
            "rows_read": rows_read,
            "rows_written": rows_written,
            "series_rows": series_rows,
            "calendar_rows": calendar_rows,
        }
    finally:
        duck_conn.close()


def _local_cache_years(source_dir: Path) -> list[int]:
    years: set[int] = set()
    if not source_dir.exists():
        return []
    for path in source_dir.iterdir():
        match = re.fullmatch(r"(20\d{2})", path.stem if path.is_file() else path.name)
        if match:
            years.add(int(match.group(1)))
        elif path.is_file() and re.fullmatch(r"(20\d{2})\d{4}", path.stem):
            years.add(int(path.stem[:4]))
    return sorted(years)


def _local_year_files(source_dir: Path, year: int) -> list[Path]:
    from . import daily_stock_sync

    files: list[Path] = []
    year_dir = source_dir / str(year)
    if year_dir.exists():
        files.extend(daily_stock_sync.iter_data_files(year_dir))
    else:
        archive = source_dir / f"{year}.7z"
        if archive.exists():
            for extracted in daily_stock_sync.extract_supported_archives(archive):
                files.extend(daily_stock_sync.iter_data_files(extracted))
    for path in source_dir.glob(f"{year}*.csv"):
        if re.fullmatch(r"20\d{6}", path.stem):
            files.append(path)
    return sorted(dict.fromkeys(path.resolve() for path in files))


def build_row_database_from_local_cache(
    *,
    duckdb_database: str | Path | None = None,
    cache_dir: str | Path | None = None,
    row_table: str = DEFAULT_ROW_TABLE,
    calendar_table: str = DEFAULT_CALENDAR_TABLE,
    years: list[int] | None = None,
    replace: bool = False,
) -> dict[str, object]:
    from . import daily_stock_sync

    target_path = duckdb_path(duckdb_database)
    source_dir = Path(cache_dir or daily_stock_sync.DEFAULT_DOWNLOAD_DIR)
    if not source_dir.is_absolute():
        source_dir = PROJECT_ROOT / source_dir
    years_to_import = years or _local_cache_years(source_dir)

    duck_conn = connect_duckdb(target_path)
    try:
        ensure_row_schema(duck_conn, row_table=row_table, calendar_table=calendar_table)
        if replace:
            duck_conn.execute(f"drop table if exists {row_table}")
            duck_conn.execute(f"drop table if exists {calendar_table}")
            ensure_row_schema(duck_conn, row_table=row_table, calendar_table=calendar_table)

        rows_written = 0
        for year in sorted({int(value) for value in years_to_import}):
            year_rows = _import_row_year(duck_conn, source_dir=source_dir, row_table=row_table, year=year)
            rows_written += int(year_rows)
            print(f"[duckdb] row import {year}: {year_rows} rows", flush=True)
        daily_rows = _import_root_daily_files(duck_conn, source_dir=source_dir, row_table=row_table, years=years_to_import)
        daily_rows += _import_daily_directories(duck_conn, source_dir=source_dir, row_table=row_table, years=years_to_import)
        rows_written += int(daily_rows)
        ensure_row_indexes(duck_conn, row_table=row_table)
        _rebuild_calendar_from_row_table(duck_conn, row_table=row_table, calendar_table=calendar_table)
        duck_conn.execute("checkpoint")
        summary = duck_conn.execute(
            f"""
            select min(trade_date), max(trade_date), count(*), count(distinct symbol)
            from {row_table}
            """
        ).fetchone()
        calendar_rows = int(duck_conn.execute(f"select count(*) from {calendar_table}").fetchone()[0])
        return {
            "duckdb_path": str(target_path),
            "row_table": row_table,
            "calendar_table": calendar_table,
            "years": sorted({int(value) for value in years_to_import}),
            "rows_written": rows_written,
            "calendar_rows": calendar_rows,
            "min_trade_date": str(summary[0]) if summary and summary[0] is not None else None,
            "max_trade_date": str(summary[1]) if summary and summary[1] is not None else None,
            "row_count": int(summary[2] or 0),
            "symbol_count": int(summary[3] or 0),
        }
    finally:
        duck_conn.close()


def _year_glob(source_dir: Path, year: int) -> str | None:
    year_dir = source_dir / str(year)
    nested = year_dir / str(year)
    if nested.exists():
        return nested.as_posix() + "/*.csv"
    if year_dir.exists():
        return year_dir.as_posix() + "/**/*.csv"
    archive = source_dir / f"{year}.7z"
    if archive.exists():
        from . import daily_stock_sync

        for extracted in daily_stock_sync.extract_supported_archives(archive):
            nested = extracted / str(year)
            if nested.exists():
                return nested.as_posix() + "/*.csv"
            return extracted.as_posix() + "/**/*.csv"
    return None


def _import_row_year(connection, *, source_dir: Path, row_table: str, year: int) -> int:
    source_glob = _year_glob(source_dir, year)
    if not source_glob:
        return 0
    before = int(connection.execute(f"select count(*) from {row_table}").fetchone()[0])
    connection.execute(f"delete from {row_table} where year(trade_date) = ?", [int(year)])
    connection.execute(
        f"""
        insert into {row_table} (
            symbol, name, trade_date, open, high, low, close, pre_close,
            change, pct_chg, volume, amount, turnover_rate, source_file
        )
        select regexp_extract(filename, '([0-9]{{6}})\\.csv$', 1) as symbol,
               null as name,
               try_cast("{YEARLY_DATE_COL}" as date) as trade_date,
               null::real as open,
               null::real as high,
               null::real as low,
               try_cast("{YEARLY_CLOSE_COL}" as real) as close,
               null::real as pre_close,
               null::real as change,
               null::real as pct_chg,
               null::real as volume,
               null::real as amount,
               try_cast("{YEARLY_TURNOVER_COL}" as real) as turnover_rate,
               filename as source_file
        from read_csv_auto(?, filename=true, union_by_name=true)
        where regexp_extract(filename, '([0-9]{{6}})\\.csv$', 1) <> ''
          and try_cast("{YEARLY_DATE_COL}" as date) is not null
          and try_cast("{YEARLY_CLOSE_COL}" as real) is not null
        """,
        [source_glob],
    )
    after = int(connection.execute(f"select count(*) from {row_table}").fetchone()[0])
    return max(after - before, 0)


def _import_root_daily_files(connection, *, source_dir: Path, row_table: str, years: list[int]) -> int:
    total = 0
    year_set = {int(value) for value in years}
    for path in sorted(source_dir.glob("20??????.csv")):
        if len(path.stem) != 8 or int(path.stem[:4]) not in year_set:
            continue
        trade_date = f"{path.stem[:4]}-{path.stem[4:6]}-{path.stem[6:8]}"
        connection.execute(f"delete from {row_table} where trade_date = cast(? as date)", [trade_date])
        raw = pd.read_csv(path, encoding="utf-8-sig")
        frame = pd.DataFrame(
            {
                "symbol": raw["stock_code"].astype(str).str.extract(r"([0-9]{6})", expand=False).fillna(""),
                "name": raw.get(DAILY_NAME_COL, pd.Series("", index=raw.index)).fillna("").astype(str),
                "trade_date": pd.to_datetime(raw[DAILY_DATE_COL], errors="coerce"),
                "open": pd.NA,
                "high": pd.NA,
                "low": pd.NA,
                "close": pd.to_numeric(raw[DAILY_CLOSE_COL], errors="coerce"),
                "pre_close": pd.NA,
                "change": pd.NA,
                "pct_chg": pd.to_numeric(raw.get(DAILY_PCT_CHG_COL), errors="coerce"),
                "volume": pd.NA,
                "amount": pd.to_numeric(raw.get(DAILY_AMOUNT_COL), errors="coerce"),
                "turnover_rate": pd.to_numeric(raw.get(DAILY_TURNOVER_COL), errors="coerce"),
                "source_file": str(path),
            }
        )
        frame = frame.dropna(subset=["trade_date", "close"])
        frame = frame.loc[frame["symbol"].str.len().eq(6)].copy()
        frame["trade_date"] = frame["trade_date"].dt.date
        imported = int(len(frame))
        if imported:
            connection.register("daily_import_frame", frame)
            try:
                connection.execute(
                    f"""
                    insert into {row_table} (
                        symbol, name, trade_date, open, high, low, close, pre_close,
                        change, pct_chg, volume, amount, turnover_rate, source_file
                    )
                    select symbol, name, trade_date, open, high, low, close, pre_close,
                           change, pct_chg, volume, amount, turnover_rate, source_file
                    from daily_import_frame
                    """
                )
            finally:
                connection.unregister("daily_import_frame")
        total += imported
        print(f"[duckdb] daily import {path.name}: {imported} rows", flush=True)
    return total


def _import_daily_directories(connection, *, source_dir: Path, row_table: str, years: list[int]) -> int:
    from . import daily_stock_sync

    total = 0
    year_set = {int(value) for value in years}
    for path in sorted(source_dir.iterdir() if source_dir.exists() else []):
        if not path.is_dir() or not re.fullmatch(r"20\d{6}", path.name) or int(path.name[:4]) not in year_set:
            continue
        files = daily_stock_sync.iter_data_files(path)
        frame, imported_files, _ = daily_stock_sync.load_daily_stock_files(files)
        imported = _insert_normalized_daily_rows(connection, row_table=row_table, frame=frame)
        total += imported
        print(f"[duckdb] minute daily import {path.name}: files={imported_files} rows={imported}", flush=True)
    return total


def _insert_normalized_daily_rows(connection, *, row_table: str, frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    prepared = frame.copy()
    prepared["trade_date"] = pd.to_datetime(prepared["trade_date"], errors="coerce")
    prepared = prepared.dropna(subset=["symbol", "trade_date", "close"]).copy()
    if prepared.empty:
        return 0
    dates = sorted({value.date() for value in prepared["trade_date"] if pd.notna(value)})
    for trade_date in dates:
        connection.execute(f"delete from {row_table} where trade_date = ?", [trade_date])
    prepared["trade_date"] = prepared["trade_date"].dt.date
    if "source_file" not in prepared.columns:
        prepared["source_file"] = ""
    for column in ["name", "open", "high", "low", "pre_close", "change", "pct_chg", "volume", "amount", "turnover_rate"]:
        if column not in prepared.columns:
            prepared[column] = pd.NA
    output = prepared[
        [
            "symbol",
            "name",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "volume",
            "amount",
            "turnover_rate",
            "source_file",
        ]
    ].copy()
    connection.register("normalized_daily_import", output)
    try:
        connection.execute(
            f"""
            insert into {row_table} (
                symbol, name, trade_date, open, high, low, close, pre_close,
                change, pct_chg, volume, amount, turnover_rate, source_file
            )
            select symbol, name, trade_date, open, high, low, close, pre_close,
                   change, pct_chg, volume, amount, turnover_rate, source_file
            from normalized_daily_import
            """
        )
    finally:
        connection.unregister("normalized_daily_import")
    return int(len(output))


def _rebuild_calendar_from_row_table(connection, *, row_table: str, calendar_table: str) -> int:
    connection.execute(f"delete from {calendar_table}")
    connection.execute(
        f"""
        insert into {calendar_table} (trade_date)
        select distinct trade_date
        from {row_table}
        order by trade_date
        """
    )
    return int(connection.execute(f"select count(*) from {calendar_table}").fetchone()[0])


def parse_intraday_intervals(value: str | Iterable[int] | None) -> list[int]:
    if value is None:
        return list(INTRADAY_INTERVALS)
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",") if part.strip()]
    else:
        raw_values = [str(part).strip() for part in value]
    intervals: set[int] = set()
    for raw in raw_values:
        cleaned = raw.lower().removesuffix("min")
        interval = int(cleaned)
        if interval not in INTRADAY_INTERVALS:
            raise ValueError(f"unsupported intraday interval: {raw}")
        intervals.add(interval)
    return sorted(intervals) or list(INTRADAY_INTERVALS)


def intraday_retention_days(value: str | int | None = None) -> int | None:
    load_env_file()
    raw = value if value is not None else os.getenv("OPENCLAW_INTRADAY_RETENTION_DAYS", str(DEFAULT_INTRADAY_RETENTION_DAYS))
    text = str(raw).strip().lower()
    if text in {"", "0", "off", "false", "none", "null"}:
        return None
    days = int(text)
    if days <= 0:
        return None
    return days


def intraday_import_batch_files(value: str | int | None = None) -> int:
    load_env_file()
    raw = value if value is not None else os.getenv(
        "OPENCLAW_INTRADAY_IMPORT_BATCH_FILES",
        str(DEFAULT_INTRADAY_IMPORT_BATCH_FILES),
    )
    try:
        batch_size = int(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"invalid intraday import batch file count: {raw}") from exc
    if batch_size <= 0:
        raise ValueError(f"intraday import batch file count must be positive: {raw}")
    return batch_size


def _iter_batches(items: list[Path], batch_size: int) -> Iterable[list[Path]]:
    for offset in range(0, len(items), batch_size):
        yield items[offset : offset + batch_size]


def _sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _duckdb_csv_sources_expression(files: list[Path]) -> str:
    if len(files) == 1:
        return _sql_string_literal(files[0].as_posix())
    return "[" + ", ".join(_sql_string_literal(file.as_posix()) for file in files) + "]"


def _date_or_today(value: str | dt.date | None = None) -> dt.date:
    if value is None:
        return dt.date.today()
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return dt.date.today()
    return pd.Timestamp(parsed).date()


def purge_intraday_bars(
    connection,
    *,
    intraday_table: str = DEFAULT_INTRADAY_TABLE,
    retention_days: int | None = DEFAULT_INTRADAY_RETENTION_DAYS,
    reference_date: str | dt.date | None = None,
) -> dict[str, object]:
    if retention_days is None or int(retention_days) <= 0:
        return {"retention_days": None, "retention_cutoff": None, "rows_purged": 0}
    cutoff = _date_or_today(reference_date) - dt.timedelta(days=int(retention_days))
    before = int(connection.execute(f"select count(*) from {intraday_table}").fetchone()[0])
    connection.execute(f"delete from {intraday_table} where trade_date < ?", [cutoff])
    after = int(connection.execute(f"select count(*) from {intraday_table}").fetchone()[0])
    return {
        "retention_days": int(retention_days),
        "retention_cutoff": str(cutoff),
        "rows_purged": max(before - after, 0),
    }


def _interval_from_dir_name(path: Path) -> int | None:
    match = re.fullmatch(r"(\d+)min", path.name.lower())
    if not match:
        return None
    interval = int(match.group(1))
    return interval if interval in INTRADAY_INTERVALS else None


def _intraday_date_hint(path: Path) -> dt.date | None:
    for part in reversed(path.parts):
        if re.fullmatch(r"20\d{6}", part):
            return dt.datetime.strptime(part, "%Y%m%d").date()
    return None


def _parse_intraday_date_cell(value: str) -> dt.date | None:
    text = value.strip().strip('"').strip("'")
    if not text:
        return None
    match = re.search(r"(?<!\d)(20\d{2})[-/]?([01]\d)[-/]?([0-3]\d)(?!\d)", text)
    if not match:
        return None
    try:
        return dt.date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def _split_intraday_tail_line(line: str) -> list[str]:
    for separator in (",", "\t", ";", "|"):
        if separator in line:
            return [part.strip() for part in line.split(separator)]
    return line.split()


def _latest_intraday_date_in_file(path: Path) -> dt.date | None:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(size - 65536, 0))
            raw = handle.read()
    except OSError:
        return None
    for encoding in ("utf-8-sig", "gb18030", "utf-16"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="ignore")
    dates: list[dt.date] = []
    for line in reversed(text.splitlines()[-32:]):
        if not line.strip():
            continue
        for cell in _split_intraday_tail_line(line)[:3]:
            parsed = _parse_intraday_date_cell(cell)
            if parsed is not None:
                dates.append(parsed)
                break
    return max(dates) if dates else None


def _latest_intraday_trade_date(interval_dir: Path) -> dt.date | None:
    latest: dt.date | None = None
    for path in interval_dir.rglob("*.csv"):
        if not path.is_file():
            continue
        file_date = _latest_intraday_date_in_file(path)
        if file_date is not None and (latest is None or file_date > latest):
            latest = file_date
    return latest


def _prepare_intraday_roots(source_dir: str | Path) -> list[Path]:
    from . import daily_stock_sync

    root = Path(source_dir)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    if root.is_file() and root.suffix.lower() in {".zip", ".7z"}:
        return daily_stock_sync.extract_supported_archives(root)
    return [root]


def _find_intraday_interval_dirs(
    roots: Iterable[Path],
    *,
    intervals: Iterable[int] | None = None,
    latest_only: bool = False,
) -> list[tuple[int, Path]]:
    requested = set(parse_intraday_intervals(intervals))
    matches: list[tuple[int, Path]] = []
    for root in roots:
        candidates = [root]
        if root.exists() and root.is_dir():
            candidates.extend(path for path in root.rglob("*") if path.is_dir())
        for path in candidates:
            interval = _interval_from_dir_name(path)
            if interval is None or interval not in requested:
                continue
            if not any(path.rglob("*.csv")):
                continue
            matches.append((interval, path.resolve()))
    matches = [
        (interval, path)
        for interval, path in matches
        if not any(
            interval == other_interval and path != other_path and path in other_path.parents
            for other_interval, other_path in matches
        )
    ]
    if latest_only:
        hints = [hint for _, path in matches if (hint := _intraday_date_hint(path)) is not None]
        if hints:
            latest = max(hints)
            matches = [(interval, path) for interval, path in matches if _intraday_date_hint(path) == latest]
    return sorted(dict.fromkeys(matches), key=lambda item: (str(item[1]), item[0]))


def _import_intraday_interval_dir(
    connection,
    *,
    intraday_table: str,
    interval: int,
    interval_dir: Path,
    start_date: str | dt.date | None = None,
    end_date: str | dt.date | None = None,
) -> int:
    source_files = sorted(path.resolve() for path in interval_dir.rglob("*.csv") if path.is_file())
    if not source_files:
        return 0
    temp_table = "tmp_openclaw_intraday_import"
    date_filters = ""
    params: list[object] = [int(interval)]
    if start_date is not None:
        date_filters += f'\n          and try_cast("{INTRADAY_DATE_COL}" as date) >= cast(? as date)'
        params.append(str(start_date))
    if end_date is not None:
        date_filters += f'\n          and try_cast("{INTRADAY_DATE_COL}" as date) <= cast(? as date)'
        params.append(str(end_date))
    imported = 0
    batch_size = intraday_import_batch_files()
    batches = list(_iter_batches(source_files, batch_size))
    for batch_index, batch_files in enumerate(batches, start=1):
        source_expression = _duckdb_csv_sources_expression(batch_files)
        connection.execute(
            f"""
            drop table if exists {temp_table}
            """
        )
        connection.execute(
            f"""
            create temp table {temp_table} as
            select regexp_extract(filename, '([0-9]{{6}})\\.csv$', 1) as symbol,
                   try_cast("{INTRADAY_DATE_COL}" as date) as trade_date,
                   try_cast("{INTRADAY_TIME_COL}" as time) as bar_time,
                   cast(? as smallint) as interval_minutes,
                   try_cast("{INTRADAY_OPEN_COL}" as real) as open,
                   try_cast("{INTRADAY_HIGH_COL}" as real) as high,
                   try_cast("{INTRADAY_LOW_COL}" as real) as low,
                   try_cast("{INTRADAY_CLOSE_COL}" as real) as close,
                   try_cast("{INTRADAY_VOLUME_COL}" as double) as volume,
                   try_cast("{INTRADAY_AMOUNT_COL}" as double) as amount,
                   filename as source_file
            from read_csv_auto({source_expression}, filename=true, union_by_name=true)
            where regexp_extract(filename, '([0-9]{{6}})\\.csv$', 1) <> ''
              and try_cast("{INTRADAY_DATE_COL}" as date) is not null
              and try_cast("{INTRADAY_TIME_COL}" as time) is not null
              and try_cast("{INTRADAY_CLOSE_COL}" as real) is not null
              {date_filters}
            """,
            params,
        )
        batch_imported = int(
            connection.execute(
                f"""
                select count(*)
                from (
                    select row_number() over (
                               partition by symbol, trade_date, bar_time, interval_minutes
                               order by source_file desc
                           ) as rn
                    from {temp_table}
                )
                where rn = 1
                """
            ).fetchone()[0]
        )
        if not batch_imported:
            continue
        connection.execute(
            f"""
            delete from {intraday_table}
            using {temp_table}
            where {intraday_table}.interval_minutes = ?
              and {intraday_table}.symbol = {temp_table}.symbol
              and {intraday_table}.trade_date = {temp_table}.trade_date
            """,
            [int(interval)],
        )
        connection.execute(
            f"""
            insert into {intraday_table} (
                symbol, trade_date, bar_time, interval_minutes, open, high, low,
                close, volume, amount, source_file
            )
            select symbol, trade_date, bar_time, interval_minutes, open, high, low,
                   close, volume, amount, source_file
            from (
                select *,
                       row_number() over (
                           partition by symbol, trade_date, bar_time, interval_minutes
                           order by source_file desc
                       ) as rn
                from {temp_table}
            )
            where rn = 1
            """
        )
        imported += batch_imported
        if len(batches) > 1:
            print(
                f"[duckdb] intraday import {interval}min batch {batch_index}/{len(batches)}: {batch_imported} rows",
                flush=True,
            )
            connection.execute("checkpoint")
    connection.execute(f"drop table if exists {temp_table}")
    return imported


def sync_intraday_bars_from_local_tree(
    *,
    duckdb_database: str | Path | None = None,
    source_dir: str | Path,
    intraday_table: str = DEFAULT_INTRADAY_TABLE,
    intervals: list[int] | None = None,
    latest_only: bool = False,
    start_date: str | dt.date | None = None,
    end_date: str | dt.date | None = None,
    retention_days: int | None = DEFAULT_INTRADAY_RETENTION_DAYS,
    retention_reference_date: str | dt.date | None = None,
) -> dict[str, object]:
    target_path = duckdb_path(duckdb_database)
    roots = _prepare_intraday_roots(source_dir)
    interval_dirs = _find_intraday_interval_dirs(roots, intervals=intervals, latest_only=latest_only)
    duck_conn = connect_duckdb(target_path)
    try:
        ensure_intraday_schema(duck_conn, intraday_table=intraday_table)
        rows_written = 0
        imported_dirs: list[dict[str, object]] = []
        for interval, path in interval_dirs:
            import_start_date = start_date
            import_end_date = end_date
            latest_date: dt.date | None = None
            if latest_only and start_date is None and end_date is None:
                latest_date = _intraday_date_hint(path) or _latest_intraday_trade_date(path)
                if latest_date is not None:
                    import_start_date = latest_date
                    import_end_date = latest_date
            imported = _import_intraday_interval_dir(
                duck_conn,
                intraday_table=intraday_table,
                interval=interval,
                interval_dir=path,
                start_date=import_start_date,
                end_date=import_end_date,
            )
            rows_written += int(imported)
            imported_dirs.append(
                {
                    "interval_minutes": int(interval),
                    "path": str(path),
                    "rows_written": int(imported),
                    "latest_trade_date": str(latest_date) if latest_date is not None else None,
                }
            )
            print(f"[duckdb] intraday import {interval}min {path}: {imported} rows", flush=True)
        retention = purge_intraday_bars(
            duck_conn,
            intraday_table=intraday_table,
            retention_days=retention_days,
            reference_date=retention_reference_date,
        )
        ensure_intraday_indexes(duck_conn, intraday_table=intraday_table)
        duck_conn.execute("checkpoint")
        summary = duck_conn.execute(
            f"""
            select count(*), count(distinct symbol), min(trade_date), max(trade_date)
            from {intraday_table}
            """
        ).fetchone()
        intervals_summary = duck_conn.execute(
            f"""
            select interval_minutes, count(*), count(distinct trade_date), min(trade_date), max(trade_date)
            from {intraday_table}
            group by interval_minutes
            order by interval_minutes
            """
        ).fetchall()
        return {
            "duckdb_path": str(target_path),
            "intraday_table": intraday_table,
            "interval_dirs": imported_dirs,
            "rows_written": int(rows_written),
            **retention,
            "row_count": int(summary[0] or 0),
            "symbol_count": int(summary[1] or 0),
            "min_trade_date": str(summary[2]) if summary and summary[2] is not None else None,
            "max_trade_date": str(summary[3]) if summary and summary[3] is not None else None,
            "intervals": [
                {
                    "interval_minutes": int(row[0]),
                    "row_count": int(row[1]),
                    "trade_dates": int(row[2]),
                    "min_trade_date": str(row[3]) if row[3] is not None else None,
                    "max_trade_date": str(row[4]) if row[4] is not None else None,
                }
                for row in intervals_summary
            ],
        }
    finally:
        duck_conn.close()


def sync_row_database_from_daily_files(
    *,
    duckdb_database: str | Path | None = None,
    row_table: str = DEFAULT_ROW_TABLE,
    calendar_table: str = DEFAULT_CALENDAR_TABLE,
    input_dir: str | Path | None = None,
    share_url: str | None = None,
    download_dir: str | Path | None = None,
    baidu_cookie: str = "",
    baidu_password: str = "",
    skip_download: bool = False,
    backfill_years: list[int] | set[int] | None = None,
) -> dict[str, object]:
    from . import daily_stock_sync

    target_path = duckdb_path(duckdb_database)
    resolved_input_dir = Path(input_dir) if input_dir is not None else None
    resolved_download_dir = Path(download_dir) if download_dir is not None else daily_stock_sync.DEFAULT_DOWNLOAD_DIR
    if resolved_input_dir is not None and not resolved_input_dir.is_absolute():
        resolved_input_dir = PROJECT_ROOT / resolved_input_dir
    if not resolved_download_dir.is_absolute():
        resolved_download_dir = PROJECT_ROOT / resolved_download_dir
    resolved_share_url = share_url if share_url is not None else daily_stock_sync.DEFAULT_BAIDU_SHARE_URL
    year_filter = {int(value) for value in (backfill_years or [])} or None
    files = daily_stock_sync.discover_or_download_inputs(
        input_dir=resolved_input_dir,
        share_url=resolved_share_url,
        download_dir=resolved_download_dir,
        baidu_cookie=baidu_cookie,
        baidu_password=baidu_password,
        skip_download=skip_download,
        backfill_years=year_filter,
    )
    frame, imported_files, rows_read = daily_stock_sync.load_daily_stock_files(files)
    duck_conn = connect_duckdb(target_path)
    try:
        ensure_row_schema(duck_conn, row_table=row_table, calendar_table=calendar_table)
        rows_written = _insert_normalized_daily_rows(duck_conn, row_table=row_table, frame=frame)
        ensure_row_indexes(duck_conn, row_table=row_table)
        calendar_rows = _rebuild_calendar_from_row_table(duck_conn, row_table=row_table, calendar_table=calendar_table)
        duck_conn.execute("checkpoint")
        summary = duck_conn.execute(
            f"""
            select count(*), count(distinct symbol), min(trade_date), max(trade_date)
            from {row_table}
            """
        ).fetchone()
        return {
            "duckdb_path": str(target_path),
            "row_table": row_table,
            "calendar_table": calendar_table,
            "discovered_files": len(files),
            "imported_files": int(imported_files),
            "rows_read": int(rows_read),
            "rows_written": int(rows_written),
            "calendar_rows": int(calendar_rows),
            "row_count": int(summary[0] or 0),
            "symbol_count": int(summary[1] or 0),
            "min_trade_date": str(summary[2]) if summary and summary[2] is not None else None,
            "max_trade_date": str(summary[3]) if summary and summary[3] is not None else None,
        }
    finally:
        duck_conn.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Copy compact A-share history from Supabase/PostgreSQL into DuckDB.")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env.local")
    parser.add_argument("--postgres-url", default=None)
    parser.add_argument("--duckdb-path", default=None)
    parser.add_argument("--source-table", default=DEFAULT_SERIES_TABLE)
    parser.add_argument("--target-table", default=DEFAULT_SERIES_TABLE)
    parser.add_argument("--calendar-table", default=DEFAULT_CALENDAR_TABLE)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--years", default="", help="Optional years/ranges to copy, e.g. 2020-2026 or 2018,2019.")
    parser.add_argument("--replace", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    import json

    args = build_arg_parser().parse_args(argv)
    load_env_file(args.env_file)
    result = sync_from_postgres(
        postgres_url=args.postgres_url,
        duckdb_database=args.duckdb_path,
        source_table=args.source_table,
        target_table=args.target_table,
        calendar_table=args.calendar_table,
        replace=bool(args.replace),
        batch_size=int(args.batch_size),
        years=parse_years(args.years) or None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
