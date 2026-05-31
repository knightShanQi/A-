from __future__ import annotations

import argparse
import json
from pathlib import Path

from a_share_predictor.database_source import load_env_file
from a_share_predictor.duckdb_store import (
    DEFAULT_CALENDAR_TABLE,
    DEFAULT_SERIES_TABLE,
    PROJECT_ROOT,
    parse_years,
    sync_from_local_baidu_cache,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the local DuckDB market database from cached Baidu Pan files.")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env.local")
    parser.add_argument("--duckdb-path", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--target-table", default=DEFAULT_SERIES_TABLE)
    parser.add_argument("--calendar-table", default=DEFAULT_CALENDAR_TABLE)
    parser.add_argument("--years", default="")
    parser.add_argument("--replace", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    load_env_file(args.env_file)
    result = sync_from_local_baidu_cache(
        duckdb_database=args.duckdb_path,
        cache_dir=args.cache_dir,
        target_table=args.target_table,
        calendar_table=args.calendar_table,
        years=parse_years(args.years) or None,
        replace=bool(args.replace),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
