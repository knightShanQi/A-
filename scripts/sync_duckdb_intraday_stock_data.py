from __future__ import annotations

import argparse
import json
from pathlib import Path

from a_share_predictor.database_source import load_env_file
from a_share_predictor.duckdb_store import (
    DEFAULT_INTRADAY_TABLE,
    PROJECT_ROOT,
    intraday_retention_days,
    parse_intraday_intervals,
    sync_intraday_bars_from_local_tree,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import local intraday A-share minute bars into DuckDB.")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env.local")
    parser.add_argument("--duckdb-path", default=None)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--intraday-table", default=DEFAULT_INTRADAY_TABLE)
    parser.add_argument("--intervals", default="1,5,15,30,60", help="Comma-separated minute intervals, e.g. 1,5,15.")
    parser.add_argument("--start-date", default=None, help="Only import bars on or after this date, YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="Only import bars on or before this date, YYYY-MM-DD.")
    parser.add_argument("--retention-days", default=None, help="Keep only this many calendar days of intraday bars. Use 0/off to disable.")
    parser.add_argument("--latest-only", action="store_true", help="When scanning a cache root, import only the latest dated intraday folder.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    load_env_file(args.env_file)
    result = sync_intraday_bars_from_local_tree(
        duckdb_database=args.duckdb_path,
        source_dir=args.input_dir,
        intraday_table=args.intraday_table,
        intervals=parse_intraday_intervals(args.intervals),
        latest_only=bool(args.latest_only),
        start_date=args.start_date,
        end_date=args.end_date,
        retention_days=intraday_retention_days(args.retention_days),
    )
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
