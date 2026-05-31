from __future__ import annotations

import argparse
import json

from a_share_predictor.data import parse_watchlist
from a_share_predictor.default_config_migration import (
    SUPPORTED_DEFAULT_HORIZON_DAYS,
    SUPPORTED_DEFAULT_POSITIVE_RETURN,
    SupportedDefaultConfig,
    regenerate_supported_default_artifacts,
)


def main() -> None:
    defaults = SupportedDefaultConfig()
    parser = argparse.ArgumentParser(
        description="Regenerate fresh ranking and review artifacts for the supported default migration path."
    )
    parser.add_argument("--board-size", type=int, default=defaults.board_size)
    parser.add_argument("--ranking-by", type=str, default=defaults.ranking_by)
    parser.add_argument("--rolling-review-days", type=int, default=defaults.rolling_review_days)
    parser.add_argument("--launch-window-confidence-weight", type=float, default=defaults.launch_window_confidence_weight)
    parser.add_argument("--watchlist", type=str, default="")
    parser.add_argument("--horizon-days", type=int, default=SUPPORTED_DEFAULT_HORIZON_DAYS)
    parser.add_argument("--positive-return", type=float, default=SUPPORTED_DEFAULT_POSITIVE_RETURN)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--rankings-only", action="store_true")
    parser.add_argument("--stores-only", action="store_true")
    parser.add_argument("--store-stage", choices=["snapshot", "features", "pools"], default=None)
    args = parser.parse_args()

    result = regenerate_supported_default_artifacts(
        board_size=int(args.board_size),
        ranking_by=str(args.ranking_by),
        rolling_review_days=int(args.rolling_review_days),
        launch_window_confidence_weight=args.launch_window_confidence_weight,
        watchlist=tuple(parse_watchlist(args.watchlist)),
        horizon_days=int(args.horizon_days),
        positive_return=float(args.positive_return),
        force_refresh=bool(args.force_refresh),
        rankings_only=bool(args.rankings_only),
        stores_only=bool(args.stores_only),
        store_stage=args.store_stage,
    )
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
