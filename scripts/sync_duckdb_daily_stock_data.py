from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path

from a_share_predictor.database_source import load_env_file
from a_share_predictor.daily_stock_sync import DEFAULT_BAIDU_SHARE_URL, DEFAULT_DOWNLOAD_DIR, parse_years
from a_share_predictor.duckdb_store import (
    DEFAULT_CALENDAR_TABLE,
    DEFAULT_INTRADAY_TABLE,
    DEFAULT_ROW_TABLE,
    PROJECT_ROOT,
    intraday_retention_days,
    parse_intraday_intervals,
    sync_intraday_bars_from_local_tree,
    sync_row_database_from_daily_files,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync Baidu Pan/local daily A-share files into the local DuckDB database.")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env.local")
    parser.add_argument("--duckdb-path", default=None)
    parser.add_argument("--row-table", default=DEFAULT_ROW_TABLE)
    parser.add_argument("--calendar-table", default=DEFAULT_CALENDAR_TABLE)
    parser.add_argument("--share-url", default=DEFAULT_BAIDU_SHARE_URL)
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--intraday-table", default=DEFAULT_INTRADAY_TABLE)
    parser.add_argument("--intraday-intervals", default="1,5,15,30,60")
    parser.add_argument("--intraday-retention-days", default=None, help="Keep only this many calendar days of intraday bars. Use 0/off to disable.")
    parser.add_argument("--skip-intraday", action="store_true")
    parser.add_argument("--all-intraday-dates", action="store_true")
    parser.add_argument("--baidu-cookie", default=None)
    parser.add_argument("--baidu-password", default=None)
    parser.add_argument("--prompt-baidu-cookie", action="store_true")
    parser.add_argument("--backfill-years", default="", help="Comma-separated years or ranges, e.g. 2026 or 2024-2026.")
    parser.add_argument("--skip-download", action="store_true", help="Only import --input-dir or existing --download-dir files.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    load_env_file(args.env_file)
    input_dir = args.input_dir
    if input_dir is None and args.skip_download:
        input_dir = args.download_dir
    baidu_cookie = args.baidu_cookie if args.baidu_cookie is not None else os.getenv("BAIDU_PAN_COOKIE", "")
    if args.prompt_baidu_cookie:
        baidu_cookie = getpass.getpass("Baidu Pan cookie: ")
    baidu_password = args.baidu_password if args.baidu_password is not None else os.getenv("BAIDU_PAN_PASSWORD", "")
    result = sync_row_database_from_daily_files(
        duckdb_database=args.duckdb_path,
        row_table=args.row_table,
        calendar_table=args.calendar_table,
        input_dir=input_dir,
        share_url=args.share_url,
        download_dir=args.download_dir,
        baidu_cookie=baidu_cookie,
        baidu_password=baidu_password,
        skip_download=bool(args.skip_download),
        backfill_years=parse_years(args.backfill_years) or None,
    )
    if not args.skip_intraday:
        intraday_root = input_dir or args.download_dir
        result["intraday"] = sync_intraday_bars_from_local_tree(
            duckdb_database=args.duckdb_path,
            source_dir=intraday_root,
            intraday_table=args.intraday_table,
            intervals=parse_intraday_intervals(args.intraday_intervals),
            latest_only=not bool(args.all_intraday_dates),
            retention_days=intraday_retention_days(args.intraday_retention_days),
        )
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
