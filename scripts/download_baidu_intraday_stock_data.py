from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path

from a_share_predictor.database_source import load_env_file
from a_share_predictor.daily_stock_sync import (
    DEFAULT_BAIDU_SHARE_URL,
    BaiduPanShareClient,
    extract_supported_archives,
    select_intraday_share_files,
)
from a_share_predictor.duckdb_store import PROJECT_ROOT, parse_intraday_intervals


def _parse_years(value: str) -> set[int] | None:
    years = {int(part.strip()) for part in value.split(",") if part.strip()}
    return years or None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Baidu Pan A-share intraday archives.")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env.local")
    parser.add_argument("--share-url", default=DEFAULT_BAIDU_SHARE_URL)
    parser.add_argument("--download-dir", type=Path, default=PROJECT_ROOT / ".cache" / "baidu_intraday_stock")
    parser.add_argument("--years", default="", help="Comma-separated years to download, e.g. 2026.")
    parser.add_argument("--intervals", default="1", help="Comma-separated intervals, e.g. 1,5,15.")
    parser.add_argument("--include-legacy-rar", action="store_true", help="Also download legacy 2000-2025 RAR archives when matched.")
    parser.add_argument("--extract", action="store_true", help="Extract supported archives after download. RAR is download-only.")
    parser.add_argument("--dry-run", action="store_true", help="List matching remote files without downloading.")
    parser.add_argument("--baidu-cookie", default=None)
    parser.add_argument("--baidu-password", default=None)
    parser.add_argument("--prompt-baidu-cookie", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    load_env_file(args.env_file)
    baidu_cookie = args.baidu_cookie if args.baidu_cookie is not None else os.getenv("BAIDU_PAN_COOKIE", "")
    if args.prompt_baidu_cookie:
        baidu_cookie = getpass.getpass("Baidu Pan cookie: ")
    baidu_password = args.baidu_password if args.baidu_password is not None else os.getenv("BAIDU_PAN_PASSWORD", "")

    client = BaiduPanShareClient(args.share_url, cookie=baidu_cookie, password=baidu_password)
    years = _parse_years(args.years)
    intervals = set(parse_intraday_intervals(args.intervals))
    matched = select_intraday_share_files(
        client.list_files(),
        years=years,
        intervals=intervals,
        include_legacy_rar=bool(args.include_legacy_rar),
    )
    if args.dry_run:
        payload = {
            "matched": [
                {
                    "server_filename": item.get("server_filename"),
                    "path": item.get("path"),
                    "size": item.get("size"),
                }
                for item in matched
            ]
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    downloaded = client._download_file_items(matched, args.download_dir)
    extracted = extract_supported_archives(args.download_dir, strict=True) if args.extract else []
    payload = {
        "download_dir": str(args.download_dir),
        "matched_count": len(matched),
        "downloaded": [str(path) for path in downloaded],
        "extracted": [str(path) for path in extracted],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
