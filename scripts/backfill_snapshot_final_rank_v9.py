from __future__ import annotations

import argparse
import json
from pathlib import Path

from a_share_predictor import daily_review


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / ".cache" / "daily_focus_board_reviews"


def backfill_snapshots(*, ranking_by: str | None, board_size: int | None, force: bool) -> dict[str, object]:
    processed = 0
    updated = 0
    skipped = 0
    failures: list[dict[str, object]] = []

    for path in sorted(CACHE_DIR.glob("snapshot_v*.pkl")):
        payload = daily_review._load_snapshot_payload(path)
        if payload is None:
            continue
        meta = dict(payload.get("meta", {}))
        cache_version = int(meta.get("cache_version", 0) or 0)
        if cache_version != daily_review.DAILY_REVIEW_CACHE_VERSION:
            continue
        if ranking_by is not None and str(meta.get("ranking_by", "")) != ranking_by:
            continue
        if board_size is not None and int(meta.get("board_size", 0) or 0) != int(board_size):
            continue

        processed += 1
        board = payload["board"]
        has_final_rank = "final_rank_score" in board.columns and board["final_rank_score"].notna().any()
        if has_final_rank and not force:
            skipped += 1
            continue

        try:
            snapshot_path = daily_review.persist_focus_board_snapshot(
                board,
                horizon_days=int(meta.get("horizon_days", 0) or 0),
                positive_return=float(meta.get("positive_return", 0.0) or 0.0),
                ranking_by=str(meta.get("ranking_by", "")),
                board_size=int(meta.get("board_size", 0) or 0),
            )
            if snapshot_path is not None and snapshot_path.exists():
                updated += 1
            else:
                failures.append({"snapshot_path": str(path), "reason": "snapshot_not_written"})
        except Exception as exc:  # pragma: no cover - operational reporting
            failures.append({"snapshot_path": str(path), "reason": repr(exc)})

    return {
        "processed_v9_snapshots": processed,
        "updated_v9_snapshots": updated,
        "skipped_existing_final_rank": skipped,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill final_rank_score into existing v9 snapshots.")
    parser.add_argument("--ranking-by", dest="ranking_by", default=None)
    parser.add_argument("--board-size", dest="board_size", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    payload = backfill_snapshots(
        ranking_by=args.ranking_by,
        board_size=args.board_size,
        force=bool(args.force),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
