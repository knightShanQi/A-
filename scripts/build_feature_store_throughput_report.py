from __future__ import annotations

import json
import random
import time
from pathlib import Path

import pandas as pd

from a_share_predictor.store import (
    MARKET_SNAPSHOT_HISTORY_STORE_VERSION,
    _build_feature_row_from_group,
    _build_strategy_snapshot_context,
    market_snapshot_history_store_path,
    read_market_snapshot_history_store,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
OUTPUT_DIR = PROJECT_ROOT / ".cache" / "trading_system_attribution"
TARGET_MARKET_DATA_DATE = "2026-05-27"
SAMPLE_SIZE = 24
RNG_SEED = 20260528


def _load_snapshot_history() -> pd.DataFrame:
    history = read_market_snapshot_history_store(TARGET_MARKET_DATA_DATE)
    if history is None or history.empty:
        raise RuntimeError(f"snapshot history store missing for {TARGET_MARKET_DATA_DATE}")
    frame = history.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    frame["date"] = pd.to_datetime(frame.get("date", frame.get("trade_date")), errors="coerce")
    frame = frame.dropna(subset=["symbol", "date"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    return frame


def build_report() -> dict[str, object]:
    snapshot_history = _load_snapshot_history()
    latest_snapshot_df, _previous_snapshot = _build_strategy_snapshot_context(TARGET_MARKET_DATA_DATE)
    latest_snapshot_df = latest_snapshot_df.copy() if isinstance(latest_snapshot_df, pd.DataFrame) else pd.DataFrame()
    if not latest_snapshot_df.empty:
        latest_snapshot_df["symbol"] = latest_snapshot_df["symbol"].astype(str).str.zfill(6)
    meta_by_symbol = {
        str(row.get("symbol", "")).zfill(6): {
            "name": str(row.get("name", "") or ""),
            "industry_name": str(row.get("industry", "") or "").strip(),
            "market": str(row.get("market", "") or "").strip(),
        }
        for row in latest_snapshot_df.to_dict("records")
    }

    grouped = snapshot_history.groupby("symbol", sort=False)
    symbols = list(grouped.groups.keys())
    sample_symbols = symbols[:]
    random.Random(RNG_SEED).shuffle(sample_symbols)
    sample_symbols = sample_symbols[: min(SAMPLE_SIZE, len(sample_symbols))]

    per_symbol_seconds: list[float] = []
    built_rows = 0
    sample_results: list[dict[str, object]] = []
    for symbol in sample_symbols:
        group = grouped.get_group(symbol).copy()
        meta = meta_by_symbol.get(symbol, {})
        started = time.perf_counter()
        row = _build_feature_row_from_group(
            group,
            symbol=symbol,
            name=str(meta.get("name", symbol) or symbol),
            industry_name=str(meta.get("industry_name", "") or ""),
            market=str(meta.get("market", "") or ""),
            snapshot_trade_date=TARGET_MARKET_DATA_DATE,
            pre_normalized=True,
        )
        elapsed = time.perf_counter() - started
        per_symbol_seconds.append(elapsed)
        if row is not None:
            built_rows += 1
        sample_results.append(
            {
                "symbol": symbol,
                "rows": int(len(group)),
                "elapsed_seconds": round(elapsed, 4),
                "built": row is not None,
            }
        )

    average_seconds = sum(per_symbol_seconds) / len(per_symbol_seconds) if per_symbol_seconds else 0.0
    estimated_total_seconds = average_seconds * len(symbols)

    snapshot_path = market_snapshot_history_store_path(TARGET_MARKET_DATA_DATE)
    report = {
        "target_market_data_date": TARGET_MARKET_DATA_DATE,
        "snapshot_history_path": str(snapshot_path),
        "snapshot_history_row_count": int(len(snapshot_history)),
        "unique_symbols_in_snapshot_history": int(len(symbols)),
        "latest_snapshot_symbol_count": int(latest_snapshot_df["symbol"].nunique()) if not latest_snapshot_df.empty else 0,
        "sample_size": int(len(sample_symbols)),
        "sample_rows_built": int(built_rows),
        "average_per_symbol_seconds": average_seconds,
        "estimated_total_seconds_for_snapshot_symbols": estimated_total_seconds,
        "estimated_total_minutes_for_snapshot_symbols": estimated_total_seconds / 60.0,
        "sample_results": sample_results,
    }
    return report


def write_outputs(report: dict[str, object]) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "feature_store_throughput_20260527.json"
    doc_path = DOCS_DIR / "feature_store_throughput_2026-05-28.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Feature Store Throughput 2026-05-28",
        "",
        "## Purpose",
        "",
        "Estimate the cold-path cost of current-date feature-store generation on the supported `h5 / 3%` path, using real snapshot-history data and sampled per-symbol feature construction.",
        "",
        "## Key Findings",
        "",
        f"- Target market date: `{report['target_market_data_date']}`.",
        f"- Snapshot-history rows: `{report['snapshot_history_row_count']}`.",
        f"- Unique snapshot-history symbols: `{report['unique_symbols_in_snapshot_history']}`.",
        f"- Sampled symbols: `{report['sample_size']}`.",
        f"- Average sampled per-symbol feature-build time: `{report['average_per_symbol_seconds']:.4f}` seconds.",
        f"- Estimated full pass over current snapshot-history symbols: `{report['estimated_total_minutes_for_snapshot_symbols']:.1f}` minutes.",
        "",
        "## Interpretation",
        "",
        "- This estimate is only for sampled per-symbol feature construction on already-loaded snapshot history. It does not include every surrounding cold-start cost.",
        "- If the estimated full-pass time is already large, the current feature-store timeout is structurally plausible even before candidate-pool or ranking work begins.",
        "",
        f"JSON: `{json_path}`",
    ]
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    report = build_report()
    write_outputs(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
