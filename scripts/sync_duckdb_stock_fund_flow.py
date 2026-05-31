from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import pandas as pd

from a_share_predictor.data import _fetch_stockpage_fund_flow_table
from a_share_predictor.database_source import load_env_file
from a_share_predictor.duckdb_store import (
    DEFAULT_ROW_TABLE,
    DEFAULT_STOCK_FUND_FLOW_TABLE,
    PROJECT_ROOT,
    connect_duckdb,
    duckdb_path,
    ensure_stock_fund_flow_schema,
    normalize_stock_fund_flow_frame,
    upsert_stock_fund_flow_frame,
)
from a_share_predictor.next_day_factor_model import DEFAULT_SYMBOL_PREFIXES


def _safe_table_name(name: str) -> str:
    if not str(name).replace("_", "").isalnum():
        raise ValueError(f"unsafe table name: {name}")
    return str(name)


def _prefix_clause(prefixes: tuple[str, ...]) -> tuple[str, list[str]]:
    clean = [prefix for prefix in prefixes if prefix]
    clause = " or ".join("symbol like ?" for _ in clean)
    return f"({clause})", [f"{prefix}%" for prefix in clean]


def _fetch_symbols(
    connection: duckdb.DuckDBPyConnection,
    *,
    price_table: str,
    start_date: str,
    end_date: str,
    prefixes: tuple[str, ...],
    symbol_limit: int | None,
) -> list[str]:
    prefix_sql, params = _prefix_clause(prefixes)
    sql = f"""
        select distinct symbol
        from {price_table}
        where trade_date >= cast(? as date)
          and trade_date <= cast(? as date)
          and {prefix_sql}
        order by symbol
    """
    values: list[object] = [start_date, end_date, *params]
    if symbol_limit is not None:
        sql += " limit ?"
        values.append(int(symbol_limit))
    return [str(row[0]).zfill(6) for row in connection.execute(sql, values).fetchall()]


def _existing_latest_dates(connection: duckdb.DuckDBPyConnection, *, fund_flow_table: str) -> dict[str, str]:
    ensure_stock_fund_flow_schema(connection, fund_flow_table=fund_flow_table)
    rows = connection.execute(
        f"""
        select symbol, max(trade_date)
        from {fund_flow_table}
        group by symbol
        """
    ).fetchall()
    return {str(symbol).zfill(6): str(max_date) for symbol, max_date in rows if max_date is not None}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync true stock main-fund-flow snapshots into DuckDB.")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env.local")
    parser.add_argument("--duckdb-path", default=None)
    parser.add_argument("--price-table", default=DEFAULT_ROW_TABLE)
    parser.add_argument("--fund-flow-table", default=DEFAULT_STOCK_FUND_FLOW_TABLE)
    parser.add_argument("--start-date", default="2025-05-28")
    parser.add_argument("--end-date", default="2026-05-28")
    parser.add_argument("--symbol-prefixes", default=",".join(DEFAULT_SYMBOL_PREFIXES))
    parser.add_argument("--symbol-limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--refresh", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    load_env_file(args.env_file)
    price_table = _safe_table_name(args.price_table)
    fund_flow_table = _safe_table_name(args.fund_flow_table)
    prefixes = tuple(prefix.strip() for prefix in str(args.symbol_prefixes).split(",") if prefix.strip())

    connection = connect_duckdb(args.duckdb_path)
    try:
        ensure_stock_fund_flow_schema(connection, fund_flow_table=fund_flow_table)
        symbols = _fetch_symbols(
            connection,
            price_table=price_table,
            start_date=args.start_date,
            end_date=args.end_date,
            prefixes=prefixes,
            symbol_limit=args.symbol_limit,
        )
        existing_latest = _existing_latest_dates(connection, fund_flow_table=fund_flow_table)
        rows_written = 0
        fetched_symbols = 0
        skipped_symbols = 0
        failed_symbols: list[dict[str, str]] = []
        start_ts = pd.to_datetime(args.start_date)
        end_ts = pd.to_datetime(args.end_date)
        for index, symbol in enumerate(symbols, start=1):
            if not args.refresh and existing_latest.get(symbol, "") >= args.end_date:
                skipped_symbols += 1
                continue
            try:
                raw = _fetch_stockpage_fund_flow_table(symbol)
                normalized = normalize_stock_fund_flow_frame(symbol, raw, source="stockpage_10jqka")
                if not normalized.empty:
                    normalized = normalized[
                        (normalized["trade_date"] >= start_ts)
                        & (normalized["trade_date"] <= end_ts)
                    ].copy()
                written = upsert_stock_fund_flow_frame(
                    connection,
                    normalized,
                    fund_flow_table=fund_flow_table,
                )
                rows_written += written
                fetched_symbols += 1
            except Exception as exc:
                failed_symbols.append({"symbol": symbol, "error": str(exc)[:200]})
            if index % 100 == 0:
                print(
                    f"[fund-flow] {index}/{len(symbols)} fetched={fetched_symbols} rows={rows_written} failed={len(failed_symbols)}",
                    flush=True,
                )
                connection.execute("checkpoint")
            if args.sleep_seconds > 0:
                time.sleep(float(args.sleep_seconds))
        connection.execute("checkpoint")
        summary = connection.execute(
            f"""
            select count(*) as rows,
                   count(distinct symbol) as symbols,
                   count(distinct trade_date) as trade_days,
                   min(trade_date) as min_date,
                   max(trade_date) as max_date
            from {fund_flow_table}
            where trade_date >= cast(? as date)
              and trade_date <= cast(? as date)
            """,
            [args.start_date, args.end_date],
        ).fetchone()
        payload = {
            "duckdb_path": str(duckdb_path(args.duckdb_path)),
            "fund_flow_table": fund_flow_table,
            "symbols_seen": len(symbols),
            "fetched_symbols": fetched_symbols,
            "skipped_symbols": skipped_symbols,
            "failed_symbols": failed_symbols[:50],
            "failed_count": len(failed_symbols),
            "rows_written": rows_written,
            "coverage": {
                "rows": int(summary[0] or 0),
                "symbols": int(summary[1] or 0),
                "trade_days": int(summary[2] or 0),
                "min_date": str(summary[3]) if summary[3] is not None else None,
                "max_date": str(summary[4]) if summary[4] is not None else None,
            },
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
