from __future__ import annotations

import argparse
import json
from pathlib import Path

from a_share_predictor.database_source import load_env_file
from a_share_predictor.duckdb_store import (
    DEFAULT_CALL_AUCTION_PROXY_TABLE,
    DEFAULT_INTRADAY_TABLE,
    DEFAULT_ROW_TABLE,
    PROJECT_ROOT,
    connect_duckdb,
    duckdb_path,
    rebuild_call_auction_proxy_table,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build derived call-auction/opening-pressure proxy table in DuckDB.")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env.local")
    parser.add_argument("--duckdb-path", default=None)
    parser.add_argument("--price-table", default=DEFAULT_ROW_TABLE)
    parser.add_argument("--intraday-table", default=DEFAULT_INTRADAY_TABLE)
    parser.add_argument("--auction-table", default=DEFAULT_CALL_AUCTION_PROXY_TABLE)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--interval-minutes", type=int, default=15)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    load_env_file(args.env_file)
    with connect_duckdb(args.duckdb_path) as connection:
        rows = rebuild_call_auction_proxy_table(
            connection,
            price_table=args.price_table,
            intraday_table=args.intraday_table,
            auction_table=args.auction_table,
            start_date=args.start_date,
            end_date=args.end_date,
            interval_minutes=args.interval_minutes,
        )
        connection.execute("checkpoint")
        summary = connection.execute(
            f"""
            select count(*) as rows,
                   count(distinct symbol) as symbols,
                   count(distinct trade_date) as trade_days,
                   min(trade_date) as min_date,
                   max(trade_date) as max_date,
                   min(first_bar_time) as min_first_bar_time,
                   max(first_bar_time) as max_first_bar_time
            from {args.auction_table}
            where interval_minutes = ?
              and trade_date >= cast(? as date)
              and trade_date <= cast(? as date)
            """,
            [int(args.interval_minutes), args.start_date, args.end_date],
        ).fetchone()
    payload = {
        "duckdb_path": str(duckdb_path(args.duckdb_path)),
        "auction_table": args.auction_table,
        "interval_minutes": int(args.interval_minutes),
        "rows_written": rows,
        "coverage": {
            "rows": int(summary[0] or 0),
            "symbols": int(summary[1] or 0),
            "trade_days": int(summary[2] or 0),
            "min_date": str(summary[3]) if summary[3] is not None else None,
            "max_date": str(summary[4]) if summary[4] is not None else None,
            "min_first_bar_time": str(summary[5]) if summary[5] is not None else None,
            "max_first_bar_time": str(summary[6]) if summary[6] is not None else None,
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
