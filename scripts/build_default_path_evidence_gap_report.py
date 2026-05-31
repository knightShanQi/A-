from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = PROJECT_ROOT / ".cache" / "daily_focus_board_reviews"
OUTPUT_DIR = PROJECT_ROOT / ".cache" / "trading_system_attribution"
DOC_PATH = PROJECT_ROOT / "docs" / "default_path_evidence_gap_2026-05-28.md"

REQUIRED_SNAPSHOT_FIELDS = [
    "selection_score",
    "final_rank_score",
    "predicted_upside_pct",
    "sector_score",
    "fund_score",
    "news_score",
    "launch_window_score",
    "launch_window_confidence_weight",
    "execution_score",
]
REQUIRED_REVIEW_FIELDS = [
    "selection_score",
    "final_rank_score",
    "predicted_upside_pct",
    "sector_score",
    "fund_score",
    "news_score",
    "launch_window_score",
    "launch_window_confidence_weight",
    "execution_score",
]


@dataclass(frozen=True)
class ConfigTarget:
    label: str
    horizon_days: int
    positive_return: float
    board_size: int

    @property
    def slug(self) -> str:
        return f"h{self.horizon_days}_r{int(round(self.positive_return * 10000))}_b{self.board_size}"


TARGETS = [
    ConfigTarget("h3_default_fallback", 3, 0.10, 50),
    ConfigTarget("h5_supported_proxy", 5, 0.03, 50),
]


def _load_payload(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload if isinstance(payload, dict) else {}


def _matching_files(target: ConfigTarget, kind: str) -> list[Path]:
    pattern = f"{kind}_v*_h{target.horizon_days}_r{int(round(target.positive_return * 10000))}_b{target.board_size}_*.pkl"
    return sorted(REVIEW_DIR.glob(pattern))


def _latest_by_board_date(paths: list[Path], date_key: str) -> Path | None:
    latest_path: Path | None = None
    latest_value = ""
    latest_version = -1
    for path in paths:
        payload = _load_payload(path)
        meta = payload.get("meta", {})
        value = str(meta.get(date_key) or "")
        version = int(meta.get("cache_version") or -1)
        if (value, version, path.name) > (latest_value, latest_version, latest_path.name if latest_path else ""):
            latest_path = path
            latest_value = value
            latest_version = version
    return latest_path


def _coverage(required: list[str], actual: list[str]) -> tuple[int, list[str]]:
    missing = [field for field in required if field not in actual]
    return len(required) - len(missing), missing


def _review_summary_stats(paths: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = _load_payload(path)
        meta = payload.get("meta", {})
        summary = payload.get("summary", {})
        if not summary:
            continue
        rows.append(
            {
                "cache_version": int(meta.get("cache_version") or -1),
                "board_date": str(meta.get("board_date") or ""),
                "review_date": str(meta.get("review_date") or ""),
                "review_count": float(summary.get("review_count") or 0.0),
                "direction_hit_rate_pct": float(summary.get("direction_hit_rate_pct") or 0.0),
                "avg_return_pct": float(summary.get("avg_return_pct") or 0.0),
                "avg_target_progress_pct": float(summary.get("avg_target_progress_pct") or 0.0),
                "win_rate_pct": float(summary.get("win_rate_pct") or 0.0),
            }
        )
    if not rows:
        return {
            "review_files": 0,
            "mean_direction_hit_rate_pct": None,
            "mean_avg_return_pct": None,
            "mean_avg_target_progress_pct": None,
            "mean_win_rate_pct": None,
        }
    frame = pd.DataFrame(rows)
    return {
        "review_files": int(len(frame)),
        "mean_direction_hit_rate_pct": round(float(frame["direction_hit_rate_pct"].mean()), 4),
        "mean_avg_return_pct": round(float(frame["avg_return_pct"].mean()), 4),
        "mean_avg_target_progress_pct": round(float(frame["avg_target_progress_pct"].mean()), 4),
        "mean_win_rate_pct": round(float(frame["win_rate_pct"].mean()), 4),
    }


def _build_target_summary(target: ConfigTarget) -> dict[str, Any]:
    snapshot_files = _matching_files(target, "snapshot")
    review_files = _matching_files(target, "review")
    latest_snapshot_path = _latest_by_board_date(snapshot_files, "board_date")
    latest_review_path = _latest_by_board_date(review_files, "review_date")

    latest_snapshot = _load_payload(latest_snapshot_path) if latest_snapshot_path else {}
    latest_review = _load_payload(latest_review_path) if latest_review_path else {}

    latest_snapshot_meta = latest_snapshot.get("meta", {})
    latest_review_meta = latest_review.get("meta", {})
    latest_snapshot_board = latest_snapshot.get("board", pd.DataFrame())
    latest_review_details = latest_review.get("details", pd.DataFrame())

    snapshot_columns = list(latest_snapshot_board.columns) if isinstance(latest_snapshot_board, pd.DataFrame) else []
    review_columns = list(latest_review_details.columns) if isinstance(latest_review_details, pd.DataFrame) else []
    snapshot_present, snapshot_missing = _coverage(REQUIRED_SNAPSHOT_FIELDS, snapshot_columns)
    review_present, review_missing = _coverage(REQUIRED_REVIEW_FIELDS, review_columns)

    summary = {
        "label": target.label,
        "horizon_days": target.horizon_days,
        "positive_return": target.positive_return,
        "board_size": target.board_size,
        "snapshot_files": int(len(snapshot_files)),
        "review_files": int(len(review_files)),
        "latest_snapshot_file": latest_snapshot_path.name if latest_snapshot_path else "",
        "latest_review_file": latest_review_path.name if latest_review_path else "",
        "latest_snapshot_cache_version": int(latest_snapshot_meta.get("cache_version") or -1),
        "latest_review_cache_version": int(latest_review_meta.get("cache_version") or -1),
        "latest_snapshot_board_date": str(latest_snapshot_meta.get("board_date") or ""),
        "latest_review_date": str(latest_review_meta.get("review_date") or ""),
        "latest_snapshot_rows": int(len(latest_snapshot_board)) if isinstance(latest_snapshot_board, pd.DataFrame) else 0,
        "latest_review_rows": int(len(latest_review_details)) if isinstance(latest_review_details, pd.DataFrame) else 0,
        "latest_snapshot_columns": int(len(snapshot_columns)),
        "latest_review_columns": int(len(review_columns)),
        "snapshot_required_fields_present": snapshot_present,
        "review_required_fields_present": review_present,
        "snapshot_missing_fields": snapshot_missing,
        "review_missing_fields": review_missing,
        "latest_model_source_label": str(latest_snapshot_meta.get("model_source_label") or ""),
    }
    summary.update(_review_summary_stats(review_files))
    return summary


def _write_outputs(frame: pd.DataFrame, payload: dict[str, Any]) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "default_path_evidence_gap_summary.csv"
    json_path = OUTPUT_DIR / "default_path_evidence_gap.json"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def _build_doc(frame: pd.DataFrame, payload: dict[str, Any], csv_path: Path, json_path: Path) -> str:
    h3 = payload["targets"]["h3_default_fallback"]
    h5 = payload["targets"]["h5_supported_proxy"]
    lines = [
        "# Default Path Evidence Gap 2026-05-28",
        "",
        "## Purpose",
        "",
        "Quantify why the current default-path audit evidence and the supported-path migration evidence are not yet interchangeable.",
        "",
        "## Key Findings",
        "",
        f"- The current default path `h3 / 10%` is operationally current in the review stack, but its latest artifacts are still generated under an unsupported model-source configuration (`{h3['latest_model_source_label'] or 'n/a'}`).",
        f"- The supported candidate path `h5 / 3%` is not just stale; it is also structurally pre-modern in the audit sense. Its latest snapshot cache version is `v{h5['latest_snapshot_cache_version']}` and latest review cache version is `v{h5['latest_review_cache_version']}`, versus `v{h3['latest_snapshot_cache_version']}` / `v{h3['latest_review_cache_version']}` on the current default path.",
        f"- On the latest snapshot, `h3 / 10%` exposes {h3['snapshot_required_fields_present']} / {len(REQUIRED_SNAPSHOT_FIELDS)} required modern audit fields, while `h5 / 3%` exposes only {h5['snapshot_required_fields_present']} / {len(REQUIRED_SNAPSHOT_FIELDS)}.",
        f"- On the latest review detail, `h3 / 10%` exposes {h3['review_required_fields_present']} / {len(REQUIRED_REVIEW_FIELDS)} required modern audit fields, while `h5 / 3%` exposes only {h5['review_required_fields_present']} / {len(REQUIRED_REVIEW_FIELDS)}.",
        f"- The missing `h5 / 3%` fields are exactly the ones current remediation work depends on: snapshot missing = {', '.join(h5['snapshot_missing_fields']) or 'none'}; review missing = {', '.join(h5['review_missing_fields']) or 'none'}.",
        "- That means a default flip to the old `h5 / 3%` artifacts would improve model support but simultaneously throw away the observability needed for the current formula-level audit and simplification work.",
        "",
        "## Summary",
        "",
        frame.to_markdown(index=False),
        "",
        "## Recommendation",
        "",
        "- Treat `h5 / 3%` regeneration as a schema-and-observability rebuild, not just a cache refresh.",
        "- Do not compare live `h3` review metrics directly against the old `h5` stack as if they were like-for-like strategy evidence; they differ in both horizon and audit schema depth.",
        "- The correct migration order remains: regenerate fresh `h5 / 3%` artifacts under the current cache schema, confirm modern fields are present, then switch the default away from unsupported `h3 / 10%`.",
        "",
        "## Next Actions",
        "",
        "1. Run the supported-path regeneration entrypoint until it produces current-version `h5 / 3%` snapshot and review files with modern fields.",
        "2. Re-run this evidence-gap report after regeneration and only treat the default switch as ready when the field-coverage gap closes.",
        "3. After that, resume score-family pruning on the supported path instead of the unsupported fallback path.",
        "",
        f"Summary CSV: `{csv_path}`",
        f"JSON: `{json_path}`",
    ]
    return "\n".join(lines)


def main() -> None:
    records = [_build_target_summary(target) for target in TARGETS]
    frame = pd.DataFrame(records)
    payload = {"generated_at": pd.Timestamp.now().isoformat(), "targets": {row["label"]: row for row in records}}
    csv_path, json_path = _write_outputs(frame, payload)
    DOC_PATH.write_text(_build_doc(frame, payload, csv_path, json_path), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
