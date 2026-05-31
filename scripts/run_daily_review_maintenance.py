from __future__ import annotations

import argparse
import json

from a_share_predictor.dashboard import DEFAULT_VIEW_PARAMS, _build_focus_board
from a_share_predictor.data import parse_watchlist
from a_share_predictor.daily_review import run_daily_review_maintenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Run daily focus-board review maintenance.")
    parser.add_argument("--board-size", type=int, default=int(DEFAULT_VIEW_PARAMS["board_size"]))
    parser.add_argument("--horizon-days", type=int, default=int(DEFAULT_VIEW_PARAMS["horizon_days"]))
    parser.add_argument("--positive-return", type=float, default=float(DEFAULT_VIEW_PARAMS["positive_return"]))
    parser.add_argument("--ranking-by", type=str, default=str(DEFAULT_VIEW_PARAMS["ranking_by"]))
    parser.add_argument("--watchlist", type=str, default=str(DEFAULT_VIEW_PARAMS["watchlist_text"]))
    parser.add_argument("--rolling-review-days", type=int, default=20)
    parser.add_argument("--launch-window-confidence-weight", type=float, default=None)
    args = parser.parse_args()

    board = _build_focus_board(
        board_size=int(args.board_size),
        custom_watchlist=tuple(parse_watchlist(args.watchlist)),
        horizon_days=int(args.horizon_days),
        positive_return=float(args.positive_return),
        ranking_by=str(args.ranking_by),
    )
    if board.empty:
        raise SystemExit("focus board is empty")
    if args.launch_window_confidence_weight is not None:
        board = board.copy()
        board["launch_window_confidence_weight"] = float(args.launch_window_confidence_weight)

    result = run_daily_review_maintenance(
        board,
        horizon_days=int(args.horizon_days),
        positive_return=float(args.positive_return),
        ranking_by=str(args.ranking_by),
        board_size=int(args.board_size),
        rolling_review_days=int(args.rolling_review_days),
    )
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
