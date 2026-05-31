from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from a_share_predictor.data import fetch_a_share_universe
from a_share_predictor.dashboard import (
    DEFAULT_VIEW_PARAMS,
    _latest_market_close_date,
    _build_focus_board,
    _build_ranked_market_snapshot,
    _write_market_rankings_cache,
    load_market_rankings,
)
from a_share_predictor.daily_review import run_daily_review_maintenance
from a_share_predictor.store import (
    build_market_candidate_pool_store,
    build_market_daily_feature_store,
    build_market_dynamic_fallback_pool_store,
    load_incremental_market_snapshot_history,
)

SUPPORTED_DEFAULT_HORIZON_DAYS = 5
SUPPORTED_DEFAULT_POSITIVE_RETURN = 0.03


@dataclass(frozen=True)
class SupportedDefaultConfig:
    board_size: int = int(DEFAULT_VIEW_PARAMS["board_size"])
    ranking_by: str = str(DEFAULT_VIEW_PARAMS["ranking_by"])
    rolling_review_days: int = 20
    launch_window_confidence_weight: float | None = 0.0
    horizon_days: int = SUPPORTED_DEFAULT_HORIZON_DAYS
    positive_return: float = SUPPORTED_DEFAULT_POSITIVE_RETURN


def regenerate_supported_default_artifacts(
    *,
    board_size: int = int(DEFAULT_VIEW_PARAMS["board_size"]),
    ranking_by: str = str(DEFAULT_VIEW_PARAMS["ranking_by"]),
    rolling_review_days: int = 20,
    launch_window_confidence_weight: float | None = 0.0,
    watchlist: tuple[str, ...] = (),
    horizon_days: int = SUPPORTED_DEFAULT_HORIZON_DAYS,
    positive_return: float = SUPPORTED_DEFAULT_POSITIVE_RETURN,
    force_refresh: bool = False,
    rankings_only: bool = False,
    stores_only: bool = False,
    store_stage: str | None = None,
) -> dict[str, object]:
    if store_stage is not None:
        return warm_supported_default_store_stage(stage=str(store_stage), force_rebuild=bool(force_refresh))
    if stores_only:
        return warm_supported_default_stores(force_rebuild=bool(force_refresh))

    rankings = _load_or_refresh_market_rankings(
        int(horizon_days),
        float(positive_return),
        force_refresh=bool(force_refresh),
    )
    if rankings_only:
        return {
            "target_config": {
                "horizon_days": int(horizon_days),
                "positive_return": float(positive_return),
                "board_size": int(board_size),
                "ranking_by": str(ranking_by),
                "rolling_review_days": int(rolling_review_days),
                "force_refresh": bool(force_refresh),
                "rankings_only": True,
                "launch_window_confidence_weight": (
                    None if launch_window_confidence_weight is None else float(launch_window_confidence_weight)
                ),
                "watchlist_size": int(len(watchlist)),
            },
            "rankings": {
                "row_count": int(len(rankings)),
                "market_data_date": str(rankings.attrs.get("market_data_date") or ""),
                "latest_market_data_date": str(rankings.attrs.get("latest_market_data_date") or ""),
                "model_source_label": str(rankings.attrs.get("model_source_label") or ""),
                "cache_stale": bool(rankings.attrs.get("cache_stale")),
            },
            "board": None,
            "maintenance": {"skipped": True, "reason": "rankings_only"},
        }
    board = _build_focus_board(
        board_size=int(board_size),
        custom_watchlist=tuple(watchlist),
        horizon_days=int(horizon_days),
        positive_return=float(positive_return),
        ranking_by=str(ranking_by),
    )
    if board.empty:
        raise ValueError(
            f"focus board is empty for h{int(horizon_days)} / {float(positive_return):.2%}; "
            "regenerate rankings or inspect model-source availability first"
        )
    board = _with_optional_launch_window_weight(board, launch_window_confidence_weight)
    maintenance = run_daily_review_maintenance(
        board,
        horizon_days=int(horizon_days),
        positive_return=float(positive_return),
        ranking_by=str(ranking_by),
        board_size=int(board_size),
        rolling_review_days=int(rolling_review_days),
    )
    market_data_date = str(board.attrs.get("market_data_date") or "")
    latest_market_data_date = str(board.attrs.get("latest_market_data_date") or market_data_date)
    return {
        "target_config": {
            "horizon_days": int(horizon_days),
            "positive_return": float(positive_return),
            "board_size": int(board_size),
            "ranking_by": str(ranking_by),
            "rolling_review_days": int(rolling_review_days),
            "force_refresh": bool(force_refresh),
            "rankings_only": False,
            "launch_window_confidence_weight": (
                None if launch_window_confidence_weight is None else float(launch_window_confidence_weight)
            ),
            "watchlist_size": int(len(watchlist)),
        },
        "rankings": {
            "row_count": int(len(rankings)),
            "market_data_date": str(rankings.attrs.get("market_data_date") or ""),
            "latest_market_data_date": str(rankings.attrs.get("latest_market_data_date") or ""),
            "model_source_label": str(rankings.attrs.get("model_source_label") or ""),
            "cache_stale": bool(rankings.attrs.get("cache_stale")),
        },
        "board": {
            "row_count": int(len(board)),
            "market_data_date": market_data_date,
            "latest_market_data_date": latest_market_data_date,
            "model_source_label": str(board.attrs.get("model_source_label") or ""),
            "symbols": board["symbol"].astype(str).head(10).tolist() if "symbol" in board.columns else [],
        },
        "maintenance": maintenance,
    }


def _with_optional_launch_window_weight(board: pd.DataFrame, weight: float | None) -> pd.DataFrame:
    if weight is None:
        return board
    weighted = board.copy()
    weighted["launch_window_confidence_weight"] = float(weight)
    return weighted


def warm_supported_default_stores(*, force_rebuild: bool = False) -> dict[str, object]:
    universe = fetch_a_share_universe()[["symbol", "name"]].drop_duplicates("symbol").copy()
    market_data_date = _latest_market_close_date()
    feature_store = build_market_daily_feature_store(
        universe,
        market_data_date,
        force_rebuild=bool(force_rebuild),
    )
    candidate_pool = build_market_candidate_pool_store(
        universe,
        market_data_date,
        feature_store=feature_store,
        force_rebuild=bool(force_rebuild),
    )
    dynamic_pool = build_market_dynamic_fallback_pool_store(
        universe,
        market_data_date,
        feature_store=feature_store,
        force_rebuild=bool(force_rebuild),
    )
    return {
        "target_config": {
            "horizon_days": SUPPORTED_DEFAULT_HORIZON_DAYS,
            "positive_return": SUPPORTED_DEFAULT_POSITIVE_RETURN,
            "force_refresh": bool(force_rebuild),
            "stores_only": True,
        },
        "market_data_date": str(market_data_date or ""),
        "feature_store": {
            "row_count": int(len(feature_store)),
        },
        "candidate_pool": {
            "row_count": int(len(candidate_pool)),
        },
        "dynamic_fallback_pool": {
            "row_count": int(len(dynamic_pool)),
        },
    }


def warm_supported_default_store_stage(*, stage: str, force_rebuild: bool = False) -> dict[str, object]:
    normalized_stage = str(stage or "").strip().lower()
    market_data_date = _latest_market_close_date()

    if normalized_stage == "snapshot":
        snapshot_history = load_incremental_market_snapshot_history(
            market_data_date,
            force_rebuild=bool(force_rebuild),
        )
        return {
            "target_config": {
                "horizon_days": SUPPORTED_DEFAULT_HORIZON_DAYS,
                "positive_return": SUPPORTED_DEFAULT_POSITIVE_RETURN,
                "force_refresh": bool(force_rebuild),
                "store_stage": "snapshot",
            },
            "market_data_date": str(market_data_date or ""),
            "snapshot_history": {
                "row_count": int(len(snapshot_history)),
            },
        }

    universe = fetch_a_share_universe()[["symbol", "name"]].drop_duplicates("symbol").copy()

    if normalized_stage == "features":
        feature_store = build_market_daily_feature_store(
            universe,
            market_data_date,
            force_rebuild=bool(force_rebuild),
        )
        return {
            "target_config": {
                "horizon_days": SUPPORTED_DEFAULT_HORIZON_DAYS,
                "positive_return": SUPPORTED_DEFAULT_POSITIVE_RETURN,
                "force_refresh": bool(force_rebuild),
                "store_stage": "features",
            },
            "market_data_date": str(market_data_date or ""),
            "feature_store": {
                "row_count": int(len(feature_store)),
            },
        }

    if normalized_stage == "pools":
        feature_store = build_market_daily_feature_store(
            universe,
            market_data_date,
            force_rebuild=bool(force_rebuild),
        )
        candidate_pool = build_market_candidate_pool_store(
            universe,
            market_data_date,
            feature_store=feature_store,
            force_rebuild=bool(force_rebuild),
        )
        dynamic_pool = build_market_dynamic_fallback_pool_store(
            universe,
            market_data_date,
            feature_store=feature_store,
            force_rebuild=bool(force_rebuild),
        )
        return {
            "target_config": {
                "horizon_days": SUPPORTED_DEFAULT_HORIZON_DAYS,
                "positive_return": SUPPORTED_DEFAULT_POSITIVE_RETURN,
                "force_refresh": bool(force_rebuild),
                "store_stage": "pools",
            },
            "market_data_date": str(market_data_date or ""),
            "feature_store": {
                "row_count": int(len(feature_store)),
            },
            "candidate_pool": {
                "row_count": int(len(candidate_pool)),
            },
            "dynamic_fallback_pool": {
                "row_count": int(len(dynamic_pool)),
            },
        }

    raise ValueError("store_stage must be one of: snapshot, features, pools")


def _load_or_refresh_market_rankings(
    horizon_days: int,
    positive_return: float,
    *,
    force_refresh: bool,
) -> pd.DataFrame:
    if not force_refresh:
        return load_market_rankings(horizon_days, positive_return)

    if hasattr(load_market_rankings, "clear"):
        load_market_rankings.clear()
    ranked, data_mode = _build_ranked_market_snapshot(horizon_days, positive_return)
    if ranked.empty:
        return ranked
    _write_market_rankings_cache(ranked, horizon_days, positive_return, data_mode)
    ranked.attrs["data_mode"] = data_mode
    ranked.attrs["market_data_date"] = str(ranked.attrs.get("market_data_date") or "")
    ranked.attrs["latest_market_data_date"] = str(
        ranked.attrs.get("latest_market_data_date") or ranked.attrs.get("market_data_date") or ""
    )
    ranked.attrs["cache_stale"] = False
    ranked.attrs["horizon_days"] = int(horizon_days)
    ranked.attrs["positive_return"] = float(positive_return)
    return ranked
