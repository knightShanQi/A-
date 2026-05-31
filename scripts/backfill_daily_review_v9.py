from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from a_share_predictor import daily_review, dashboard


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / ".cache" / "daily_focus_board_reviews"


def _load_snapshot_index() -> dict[tuple[int, int, str, int, str], dict[str, object]]:
    index: dict[tuple[int, int, str, int, str], dict[str, object]] = {}
    for path in sorted(CACHE_DIR.glob("snapshot_v*_*.pkl")):
        payload = daily_review._load_snapshot_payload(path)
        if payload is None:
            continue
        meta = dict(payload.get("meta", {}))
        key = (
            int(meta.get("horizon_days", 0) or 0),
            int(float(meta.get("positive_return", 0.0) or 0.0) * 10000),
            str(meta.get("ranking_by", "")),
            int(meta.get("board_size", 0) or 0),
            str(meta.get("board_date", "")),
        )
        current = index.get(key)
        if current is None or int(meta.get("cache_version", 0) or 0) > int(current["meta"].get("cache_version", 0) or 0):
            index[key] = {"payload": payload, "meta": meta, "path": path}
    return index


def _load_feature_store(board_date: str) -> pd.DataFrame:
    token = board_date.replace("-", "")
    candidates = sorted(PROJECT_ROOT.joinpath(".cache").glob(f"market_daily_feature_store_*_{token}.pkl"))
    if not candidates:
        return pd.DataFrame()
    path = max(candidates, key=lambda p: p.stat().st_size)
    try:
        payload = pd.read_pickle(path)
    except Exception:
        return pd.DataFrame()
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, pd.DataFrame) or data.empty:
        return pd.DataFrame()
    feature_cols = [
        "symbol",
        "stage_code",
        "stage_label",
        "stage_score",
        "launch_score",
        "launch_readiness_score",
        "market_resonance_score",
        "quant_score",
    ]
    available = [col for col in feature_cols if col in data.columns]
    if "symbol" not in available:
        return pd.DataFrame()
    feature_frame = data[available].copy()
    feature_frame["symbol"] = feature_frame["symbol"].astype(str).str.zfill(6)
    return feature_frame


def _enrich_snapshot_board(
    board: pd.DataFrame,
    meta: dict[str, object],
    *,
    launch_window_confidence_weight: float | None = None,
) -> pd.DataFrame:
    enriched = board.copy()
    if "symbol" in enriched.columns:
        enriched["symbol"] = enriched["symbol"].astype(str).str.zfill(6)

    feature_store = _load_feature_store(str(meta.get("board_date", "")))
    if not feature_store.empty:
        existing_cols = [col for col in feature_store.columns if col != "symbol" and col in enriched.columns]
        feature_store = feature_store.drop(columns=existing_cols, errors="ignore")
        enriched = enriched.merge(feature_store, on="symbol", how="left")

    enriched.attrs["market_data_date"] = str(meta.get("board_date", ""))
    enriched.attrs["latest_market_data_date"] = str(meta.get("latest_market_data_date") or meta.get("board_date") or "")
    enriched.attrs["model_source_label"] = str(meta.get("model_source_label") or "")
    enriched.attrs["computed_at"] = str(meta.get("computed_at") or meta.get("captured_at") or "")
    if launch_window_confidence_weight is not None:
        enriched["launch_window_confidence_weight"] = float(launch_window_confidence_weight)

    enriched = dashboard._ensure_launch_window_columns(enriched, force=True)
    action_view = enriched.apply(lambda row: pd.Series(dashboard._evaluate_board_action(row)), axis=1)
    for column in action_view.columns:
        enriched[column] = action_view[column]

    if "stage_score" not in enriched.columns:
        enriched["stage_score"] = 50.0
    if "stage_code" not in enriched.columns:
        enriched["stage_code"] = ""
    if "selection_score" not in enriched.columns:
        enriched["selection_score"] = pd.to_numeric(enriched.get("attention_score"), errors="coerce").fillna(50.0)
    if "execution_score" not in enriched.columns:
        enriched["execution_score"] = 50.0

    return enriched


def backfill_reviews(
    *,
    ranking_by: str | None,
    board_size: int | None,
    force: bool,
    launch_window_confidence_weight: float | None = None,
) -> dict[str, object]:
    snapshot_index = _load_snapshot_index()
    review_paths = sorted(CACHE_DIR.glob("review_v*_*.pkl"))

    processed = 0
    created_reviews = 0
    created_snapshots = 0
    skipped = 0
    failures: list[dict[str, object]] = []

    for review_path in review_paths:
        payload = daily_review._load_review_payload(review_path)
        if payload is None:
            continue
        meta = dict(payload.get("meta", {}))
        cache_version = int(meta.get("cache_version", 0) or 0)
        if cache_version >= daily_review.DAILY_REVIEW_CACHE_VERSION:
            continue
        if ranking_by is not None and str(meta.get("ranking_by", "")) != ranking_by:
            continue
        if board_size is not None and int(meta.get("board_size", 0) or 0) != int(board_size):
            continue

        processed += 1
        horizon_days = int(meta.get("horizon_days", 0) or 0)
        positive_return = float(meta.get("positive_return", 0.0) or 0.0)
        board_size_value = int(meta.get("board_size", 0) or 0)
        ranking_value = str(meta.get("ranking_by", ""))
        board_date = str(meta.get("board_date", ""))
        review_date = str(meta.get("review_date", ""))
        key = (
            horizon_days,
            int(positive_return * 10000),
            ranking_value,
            board_size_value,
            board_date,
        )
        snapshot_record = snapshot_index.get(key)
        if snapshot_record is None:
            failures.append({"review_path": str(review_path), "reason": "matching_snapshot_missing"})
            continue

        target_review_path = daily_review._review_cache_path(
            horizon_days,
            positive_return,
            ranking_value,
            board_size_value,
            board_date,
            review_date,
        )
        if target_review_path.exists() and not force:
            skipped += 1
            continue

        try:
            snapshot_payload = snapshot_record["payload"]
            snapshot_meta = dict(snapshot_payload.get("meta", {}))
            snapshot_board = snapshot_payload["board"]
            enriched_board = _enrich_snapshot_board(
                snapshot_board,
                snapshot_meta,
                launch_window_confidence_weight=launch_window_confidence_weight,
            )

            snapshot_target_path = daily_review.persist_focus_board_snapshot(
                enriched_board,
                horizon_days=horizon_days,
                positive_return=positive_return,
                ranking_by=ranking_value,
                board_size=board_size_value,
            )
            if snapshot_target_path is not None and snapshot_target_path.exists():
                created_snapshots += 1

            summary, details = daily_review._review_snapshot(
                enriched_board,
                board_date=board_date,
                review_date=review_date,
                positive_return=positive_return,
            )
            daily_review._save_review_payload(
                target_review_path,
                summary=summary,
                details=details,
                meta={
                    "board_date": board_date,
                    "review_date": review_date,
                    "horizon_days": horizon_days,
                    "positive_return": positive_return,
                    "ranking_by": ranking_value,
                    "board_size": board_size_value,
                    "backfilled_from_review": str(review_path.name),
                    "backfilled_from_snapshot": str(Path(snapshot_record["path"]).name),
                },
            )
            created_reviews += 1
        except Exception as exc:  # pragma: no cover - operational reporting
            failures.append({"review_path": str(review_path), "reason": repr(exc)})

    return {
        "processed_legacy_reviews": processed,
        "created_current_version_reviews": created_reviews,
        "created_current_version_snapshots": created_snapshots,
        "skipped_existing_current_version_reviews": skipped,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill v9 daily review artifacts with scorer fields.")
    parser.add_argument("--ranking-by", dest="ranking_by", default="关注分数")
    parser.add_argument("--board-size", dest="board_size", type=int, default=50)
    parser.add_argument("--launch-window-confidence-weight", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    payload = backfill_reviews(
        ranking_by=args.ranking_by,
        board_size=args.board_size,
        force=bool(args.force),
        launch_window_confidence_weight=args.launch_window_confidence_weight,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
