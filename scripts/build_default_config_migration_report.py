from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = PROJECT_ROOT / ".cache" / "daily_focus_board_reviews"
OUTPUT_DIR = PROJECT_ROOT / ".cache" / "trading_system_attribution"
DOC_PATH = PROJECT_ROOT / "docs" / "default_config_migration_2026-05-28.md"


def _collect_artifact_window(snapshot_pattern: str, review_pattern: str) -> dict[str, object]:
    snapshot_paths = sorted(REVIEW_DIR.glob(snapshot_pattern))
    review_paths = sorted(REVIEW_DIR.glob(review_pattern))
    snapshot_dates: list[str] = []
    review_dates: list[tuple[str, str]] = []
    for path in snapshot_paths:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        meta = payload.get("meta", {})
        snapshot_dates.append(str(meta.get("board_date") or ""))
    for path in review_paths:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        meta = payload.get("meta", {})
        review_dates.append((str(meta.get("board_date") or ""), str(meta.get("review_date") or "")))
    return {
        "snapshot_files": len(snapshot_paths),
        "review_files": len(review_paths),
        "first_snapshot_date": snapshot_dates[0] if snapshot_dates else "",
        "last_snapshot_date": snapshot_dates[-1] if snapshot_dates else "",
        "first_review_pair": review_dates[0] if review_dates else ("", ""),
        "last_review_pair": review_dates[-1] if review_dates else ("", ""),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    h3 = _collect_artifact_window("snapshot_v10_h3_r1000_*.pkl", "review_v10_h3_r1000_*.pkl")
    h5 = _collect_artifact_window("snapshot_v3_h5_r300_*.pkl", "review_v3_h5_r300_*.pkl")
    cache_root = PROJECT_ROOT / ".cache"
    rows = [
        {
            "config": "h3_r1000_current_default",
            **h3,
            "ranking_cache_present": (cache_root / "market_rankings_v12_h3_r1000.pkl").exists(),
        },
        {
            "config": "h5_r300_proxy_reference",
            **h5,
            "ranking_cache_present": (cache_root / "market_rankings_v12_h5_r300.pkl").exists(),
        },
    ]
    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "default_config_migration_summary.csv"
    json_path = OUTPUT_DIR / "default_config_migration.json"
    df.to_csv(csv_path, index=False)

    findings = [
        f"The current default `h3 / 10%` path has the freshest review stack in the workspace: {h3['snapshot_files']} snapshot files and {h3['review_files']} review files spanning {h3['first_snapshot_date']} to {h3['last_snapshot_date']}.",
        f"The only proxy-backed alternative, `h5 / 3%`, is materially stale in the review stack: {h5['snapshot_files']} snapshots and {h5['review_files']} reviews, with the latest review ending {h5['last_review_pair'][1] or 'n/a'}.",
        f"`h3 / 10%` therefore has operational continuity but no model support, while `h5 / 3%` has proxy-model support but not a current review/ranking artifact chain.",
        "That means the practical default decision is two-stage, not one-step: retire `h3 / 10%` as the strategic default, but do not blindly switch users to `h5 / 3%` until its current ranking/review pipeline is regenerated.",
        "The evidence-backed migration order is: first regenerate fresh `h5 / 3%` ranking/review artifacts under the current cache schema, then switch the default, then resume score-layer optimization on the supported path.",
    ]

    payload = {
        "summary": rows,
        "findings": findings,
        "recommendation": "prepare_h5_then_switch_default",
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    lines = [
        "# Default Config Migration 2026-05-28",
        "",
        "## Purpose",
        "",
        "Decide whether the unsupported default `h3 / 10%` path should be restored, retired, or migrated, using both model-source availability and current artifact freshness.",
        "",
        "## Key Findings",
        "",
    ]
    for finding in findings:
        lines.append(f"- {finding}")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            df.to_markdown(index=False),
            "",
            "## Recommendation",
            "",
            "- Strategic recommendation: retire `h3 / 10%` as the long-lived default because it is unsupported by model artifacts.",
            "- Operational recommendation: do not switch the default immediately to `h5 / 3%` until you regenerate a fresh ranking/review chain for that configuration under the current schema.",
            "- Practical migration sequence: rebuild `h5 / 3%` caches and review history, validate its proxy-backed path, then flip the default and continue formula simplification there.",
            "",
            "## Next Actions",
            "",
            "1. Run the maintained market-wide training and ranking/review maintenance flow for `h5 / 3%` so it becomes the supported path with current artifacts.",
            "2. Only after that should the UI default move away from `h3 / 10%`.",
            "",
            f"Summary CSV: `{csv_path}`",
            f"JSON: `{json_path}`",
            "",
        ]
    )
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
