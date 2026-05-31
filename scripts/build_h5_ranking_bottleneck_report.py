from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from a_share_predictor.dashboard import FULL_MARKET_MAX_WORKERS, RULE_BASED_CANDIDATE_POOL_SIZE
from a_share_predictor.default_config_migration import (
    SUPPORTED_DEFAULT_HORIZON_DAYS,
    SUPPORTED_DEFAULT_POSITIVE_RETURN,
)
from a_share_predictor.store import (
    DYNAMIC_FALLBACK_ANALYSIS_BUFFER_SIZE,
    MARKET_CANDIDATE_POOL_STORE_VERSION,
    MARKET_DAILY_FEATURE_STORE_VERSION,
    MARKET_DYNAMIC_FALLBACK_STORE_VERSION,
    MARKET_SNAPSHOT_HISTORY_STORE_VERSION,
    market_candidate_pool_store_path,
    market_daily_feature_store_path,
    market_dynamic_fallback_store_path,
    market_snapshot_history_store_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / ".cache"
OUTPUT_DIR = CACHE_DIR / "trading_system_attribution"
DOCS_DIR = PROJECT_ROOT / "docs"


def _latest_cache_date(prefix: str, version: int) -> str | None:
    pattern = f"{prefix}_v{version}_*.pkl"
    candidates = sorted(CACHE_DIR.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    stem = candidates[0].stem
    marker = f"{prefix}_v{version}_"
    if not stem.startswith(marker):
        return None
    return stem[len(marker) :]


def _load_store_meta(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"exists": False}
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return {"exists": True, "readable": False}
    data = payload.get("data")
    meta = payload.get("meta", {})
    return {
        "exists": True,
        "readable": True,
        "row_count": int(len(data)) if isinstance(data, pd.DataFrame) else None,
        "cache_version": meta.get("cache_version"),
        "market_data_date": meta.get("market_data_date"),
        "path": str(path),
    }


def _candidate_analysis_stats() -> dict[str, object]:
    all_files = sorted(CACHE_DIR.glob("candidate_analysis_v*_*.pkl"))
    h5_files = [path for path in all_files if "_h5_" in path.name]
    current_h5_files = [path for path in h5_files if "_20260527_" in path.name]
    return {
        "all_h5_files": len(h5_files),
        "current_h5_files_20260527": len(current_h5_files),
        "latest_h5_file": h5_files[-1].name if h5_files else None,
    }


def build_report() -> dict[str, object]:
    latest_snapshot_date = _latest_cache_date("market_snapshot_history_store", MARKET_SNAPSHOT_HISTORY_STORE_VERSION)
    latest_feature_date = _latest_cache_date("market_daily_feature_store", MARKET_DAILY_FEATURE_STORE_VERSION)
    latest_candidate_pool_date = _latest_cache_date("market_candidate_pool_store", MARKET_CANDIDATE_POOL_STORE_VERSION)
    latest_dynamic_pool_date = _latest_cache_date("market_dynamic_fallback_store", MARKET_DYNAMIC_FALLBACK_STORE_VERSION)

    target_market_data_date = latest_snapshot_date
    snapshot_meta = _load_store_meta(market_snapshot_history_store_path(target_market_data_date))
    feature_meta = _load_store_meta(market_daily_feature_store_path(target_market_data_date))
    candidate_pool_meta = _load_store_meta(market_candidate_pool_store_path(target_market_data_date))
    dynamic_pool_meta = _load_store_meta(market_dynamic_fallback_store_path(target_market_data_date))
    candidate_analysis_meta = _candidate_analysis_stats()

    current_cache_gap = {
        "target_market_data_date": target_market_data_date,
        "snapshot_history_ready": bool(snapshot_meta.get("exists")),
        "feature_store_ready": bool(feature_meta.get("exists")),
        "candidate_pool_ready": bool(candidate_pool_meta.get("exists")),
        "dynamic_pool_ready": bool(dynamic_pool_meta.get("exists")),
        "candidate_analysis_cache_ready": candidate_analysis_meta["current_h5_files_20260527"] > 0,
    }

    bottleneck_signals: list[str] = []
    if current_cache_gap["snapshot_history_ready"] and not current_cache_gap["feature_store_ready"]:
        bottleneck_signals.append("current snapshot history exists, but no same-date feature store has been written yet")
    if not current_cache_gap["candidate_pool_ready"]:
        bottleneck_signals.append("no same-date strategy candidate pool store exists for the target date")
    if not current_cache_gap["candidate_analysis_cache_ready"]:
        bottleneck_signals.append("no current-date h5 candidate-analysis cache files exist, so per-symbol analysis starts cold")
    if FULL_MARKET_MAX_WORKERS <= 1:
        bottleneck_signals.append("full-market candidate analysis is hard-pinned to one worker")

    report = {
        "target_config": {
            "horizon_days": SUPPORTED_DEFAULT_HORIZON_DAYS,
            "positive_return": SUPPORTED_DEFAULT_POSITIVE_RETURN,
        },
        "latest_store_dates": {
            "snapshot_history": latest_snapshot_date,
            "feature_store": latest_feature_date,
            "candidate_pool_store": latest_candidate_pool_date,
            "dynamic_fallback_store": latest_dynamic_pool_date,
        },
        "target_date_store_status": {
            "snapshot_history": snapshot_meta,
            "feature_store": feature_meta,
            "candidate_pool_store": candidate_pool_meta,
            "dynamic_fallback_store": dynamic_pool_meta,
        },
        "candidate_analysis_cache": candidate_analysis_meta,
        "rebuild_structure": {
            "full_market_max_workers": FULL_MARKET_MAX_WORKERS,
            "rule_based_candidate_pool_size": RULE_BASED_CANDIDATE_POOL_SIZE,
            "dynamic_fallback_analysis_buffer_size": DYNAMIC_FALLBACK_ANALYSIS_BUFFER_SIZE,
            "minimum_serial_candidate_analyses_if_cold": RULE_BASED_CANDIDATE_POOL_SIZE,
        },
        "current_cache_gap": current_cache_gap,
        "bottleneck_signals": bottleneck_signals,
    }
    return report


def write_outputs(report: dict[str, object]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "h5_ranking_bottleneck.json"
    md_path = DOCS_DIR / "h5_ranking_bottleneck_2026-05-28.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    lines = [
        "# h5 Ranking Bottleneck 2026-05-28",
        "",
        "## Purpose",
        "",
        "Identify the concrete code-path and cache-state reasons why supported-path `h5 / 3%` regeneration is still too slow to complete inside normal CLI windows.",
        "",
        "## Key Findings",
        "",
        f"- Target supported config remains `h{report['target_config']['horizon_days']} / {report['target_config']['positive_return']:.0%}`.",
        f"- Latest snapshot-history store date is `{report['latest_store_dates']['snapshot_history']}`.",
        f"- Same-date feature store exists: `{report['current_cache_gap']['feature_store_ready']}`.",
        f"- Same-date candidate-pool store exists: `{report['current_cache_gap']['candidate_pool_ready']}`.",
        f"- Same-date dynamic-fallback store exists: `{report['current_cache_gap']['dynamic_pool_ready']}`.",
        f"- Current-date h5 candidate-analysis cache exists: `{report['current_cache_gap']['candidate_analysis_cache_ready']}`.",
        f"- Full-market analysis worker count is hard-pinned to `{report['rebuild_structure']['full_market_max_workers']}`.",
        f"- Cold-path minimum serial candidate analyses implied by current pool sizing is `{report['rebuild_structure']['minimum_serial_candidate_analyses_if_cold']}` symbols.",
        "",
        "## Interpretation",
        "",
    ]
    for item in report["bottleneck_signals"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Practical Consequence",
            "",
            "- The current `h5` migration blocker is no longer just “the command times out”.",
            "- It is structurally a cold rebuild problem: the path has current snapshot history, but it lacks same-date feature-store, candidate-pool, and candidate-analysis cache coverage, while the candidate-analysis phase still runs in a single worker.",
            "- That means annualized-return optimization work should continue to treat supported-path recovery as an engineering throughput problem before it is a scorer-formula problem.",
            "",
            "## Suggested Next Optimization Order",
            "",
            "1. Reduce cold-path candidate-analysis cost before touching score formulas again.",
            "2. Add a resumable or batched warmup path for current-date h5 candidate-analysis caches.",
            "3. Only after h5 ranking artifacts can be regenerated on demand should default-path migration be considered operationally ready.",
            "",
            f"JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    report = build_report()
    write_outputs(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
