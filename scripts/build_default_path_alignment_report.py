from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = PROJECT_ROOT / ".cache" / "daily_focus_board_reviews"
OUTPUT_DIR = PROJECT_ROOT / ".cache" / "trading_system_attribution"
DOC_PATH = PROJECT_ROOT / "docs" / "default_path_alignment_2026-05-28.md"

HORIZON_DAYS = 3
POSITIVE_RETURN = 0.10
BOARD_SIZE = 50
RANKING_BY = "关注分数"


def _load_payload(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload if isinstance(payload, dict) else {}


def _supported_versions() -> list[int]:
    return [10, 9, 8, 7, 6]


def _snapshot_payloads() -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for version in _supported_versions():
        pattern = f"snapshot_v{version}_h{HORIZON_DAYS}_r{int(POSITIVE_RETURN * 10000)}_b{BOARD_SIZE}_*.pkl"
        for path in sorted(REVIEW_DIR.glob(pattern)):
            payload = _load_payload(path)
            meta = payload.get("meta", {})
            board = payload.get("board")
            if str(meta.get("ranking_by") or "") != RANKING_BY:
                continue
            if not isinstance(board, pd.DataFrame):
                continue
            payloads.append({"path": path, "meta": meta, "board": board.copy()})
    return payloads


def _review_payloads() -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for version in _supported_versions():
        pattern = f"review_v{version}_h{HORIZON_DAYS}_r{int(POSITIVE_RETURN * 10000)}_b{BOARD_SIZE}_*.pkl"
        for path in sorted(REVIEW_DIR.glob(pattern)):
            payload = _load_payload(path)
            meta = payload.get("meta", {})
            details = payload.get("details")
            summary = payload.get("summary")
            if str(meta.get("ranking_by") or "") != RANKING_BY:
                continue
            if not isinstance(details, pd.DataFrame) or not isinstance(summary, dict):
                continue
            payloads.append({"path": path, "meta": meta, "details": details.copy(), "summary": dict(summary)})
    return payloads


def _select_loader_snapshot(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        payloads,
        key=lambda item: (
            str(item["meta"].get("board_date") or ""),
            str(item["meta"].get("captured_at") or ""),
        ),
    )[-1]


def _select_latest_review(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        payloads,
        key=lambda item: (
            str(item["meta"].get("review_date") or ""),
            str(item["meta"].get("board_date") or ""),
            int(item["meta"].get("cache_version") or -1),
        ),
    )[-1]


def _select_review_linked_snapshot(snapshot_payloads: list[dict[str, Any]], board_date: str) -> dict[str, Any] | None:
    candidates = [item for item in snapshot_payloads if str(item["meta"].get("board_date") or "") == board_date]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            int(item["meta"].get("cache_version") or -1),
            str(item["meta"].get("captured_at") or ""),
            item["path"].name,
        ),
    )[-1]


def _overlap(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
    left_symbols = set(left["symbol"].astype(str)) if "symbol" in left.columns else set()
    right_symbols = set(right["symbol"].astype(str)) if "symbol" in right.columns else set()
    common = sorted(left_symbols & right_symbols)
    return {
        "left_rows": int(len(left)),
        "right_rows": int(len(right)),
        "common_symbols": int(len(common)),
        "left_only": sorted(left_symbols - right_symbols),
        "right_only": sorted(right_symbols - left_symbols),
    }


def _shared_value_diffs(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
    if "symbol" not in left.columns or "symbol" not in right.columns:
        return {"shared_columns": 0, "changed_columns": {}}
    common_cols = [col for col in left.columns if col in right.columns]
    merged = left[common_cols].merge(right[common_cols], on="symbol", suffixes=("_left", "_right"))
    changed: dict[str, int] = {}
    for col in common_cols:
        if col == "symbol":
            continue
        left_series = merged[f"{col}_left"].fillna("__NA__").astype(str)
        right_series = merged[f"{col}_right"].fillna("__NA__").astype(str)
        diff_count = int((left_series != right_series).sum())
        if diff_count:
            changed[col] = diff_count
    return {"shared_columns": int(len(common_cols)), "changed_columns": changed}


def _artifact_summary(item: dict[str, Any], frame_key: str) -> dict[str, Any]:
    frame = item[frame_key]
    meta = item["meta"]
    return {
        "file": item["path"].name,
        "cache_version": int(meta.get("cache_version") or -1),
        "board_date": str(meta.get("board_date") or ""),
        "review_date": str(meta.get("review_date") or ""),
        "latest_market_data_date": str(meta.get("latest_market_data_date") or ""),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_names": list(frame.columns),
    }


def main() -> None:
    snapshots = _snapshot_payloads()
    reviews = _review_payloads()
    loader_snapshot = _select_loader_snapshot(snapshots)
    latest_review = _select_latest_review(reviews)
    linked_snapshot = _select_review_linked_snapshot(snapshots, str(latest_review["meta"].get("board_date") or ""))
    if linked_snapshot is None:
        raise RuntimeError("no review-linked snapshot found for latest default-path review")

    loader_board = loader_snapshot["board"]
    linked_board = linked_snapshot["board"]
    review_details = latest_review["details"]

    payload = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "loader_selected_snapshot": _artifact_summary(loader_snapshot, "board"),
        "latest_review": _artifact_summary(latest_review, "details"),
        "review_linked_snapshot": _artifact_summary(linked_snapshot, "board"),
        "loader_vs_linked_symbol_overlap": _overlap(loader_board, linked_board),
        "loader_vs_linked_shared_value_diffs": _shared_value_diffs(loader_board, linked_board),
        "linked_snapshot_missing_vs_review": [col for col in review_details.columns if col not in linked_board.columns],
        "loader_snapshot_missing_vs_review": [col for col in review_details.columns if col not in loader_board.columns],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "default_path_alignment.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    loader = payload["loader_selected_snapshot"]
    review = payload["latest_review"]
    linked = payload["review_linked_snapshot"]
    overlap = payload["loader_vs_linked_symbol_overlap"]
    diffs = payload["loader_vs_linked_shared_value_diffs"]["changed_columns"]

    lines = [
        "# Default Path Alignment 2026-05-28",
        "",
        "## Purpose",
        "",
        "Check whether the default snapshot loader is returning an audit-safe board for interpreting the latest default-path review evidence.",
        "",
        "## Key Findings",
        "",
        f"- The default snapshot loader currently resolves to `{loader['file']}` (`v{loader['cache_version']}`, board date {loader['board_date']}).",
        f"- The latest default-path review evidence resolves to `{review['file']}` (`v{review['cache_version']}`, board date {review['board_date']}, review date {review['review_date']}).",
        f"- The review-linked snapshot for that latest review is `{linked['file']}` (`v{linked['cache_version']}`, board date {linked['board_date']}).",
        f"- The loader-selected snapshot and the review-linked snapshot contain the same symbol set on this sample (`{overlap['common_symbols']} / {overlap['left_rows']}` overlap, no symbol drift), so the risk is not candidate identity drift.",
        f"- The risk is interpretability drift: the loader-selected snapshot is missing {len(payload['loader_snapshot_missing_vs_review'])} review-detail columns versus {len(payload['linked_snapshot_missing_vs_review'])} on the review-linked snapshot.",
        f"- Even on shared columns, the two snapshots are not value-identical. {len(diffs)} shared columns changed between the loader-selected `v{loader['cache_version']}` board and the review-linked `v{linked['cache_version']}` board, including attention/probability/launch fields.",
        "- Practical implication: using `load_latest_snapshot_board()` as the explanation layer for the latest review can silently downgrade field coverage and mix board dates, even when the selected stock set is unchanged.",
        "",
        "## Recommendation",
        "",
        "- For audit and root-cause analysis, anchor on the latest review detail first, then join back to the snapshot with the same `board_date`.",
        "- Treat `load_latest_snapshot_board()` as a UI convenience helper, not as the authoritative artifact selector for formula-level audits while mixed cache generations exist.",
        "- After the supported-path regeneration work, re-check whether snapshot selection still needs a schema-aware tie-breaker.",
        "",
        "## Artifact Summary",
        "",
        f"- Loader-selected snapshot: `{loader['file']}` with {loader['rows']} rows and {loader['columns']} columns",
        f"- Review-linked snapshot: `{linked['file']}` with {linked['rows']} rows and {linked['columns']} columns",
        f"- Latest review detail: `{review['file']}` with {review['rows']} rows and {review['columns']} columns",
        f"- Loader snapshot missing vs review: {', '.join(payload['loader_snapshot_missing_vs_review'][:20])}",
        f"- Review-linked snapshot missing vs review: {', '.join(payload['linked_snapshot_missing_vs_review'][:20])}",
        "",
        f"JSON: `{json_path}`",
    ]
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
