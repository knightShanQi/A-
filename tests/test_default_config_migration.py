import pandas as pd
import pytest

import a_share_predictor.default_config_migration as migration


def test_regenerate_supported_default_artifacts_uses_supported_h5_path(monkeypatch):
    rankings = pd.DataFrame([{"symbol": "600001"}])
    rankings.attrs["market_data_date"] = "2026-05-28"
    rankings.attrs["latest_market_data_date"] = "2026-05-28"
    rankings.attrs["model_source_label"] = "市场代理模型"
    rankings.attrs["cache_stale"] = False

    board = pd.DataFrame(
        [
            {"symbol": "600001", "selection_score": 88.0},
            {"symbol": "600002", "selection_score": 84.0},
        ]
    )
    board.attrs["market_data_date"] = "2026-05-28"
    board.attrs["latest_market_data_date"] = "2026-05-28"
    board.attrs["model_source_label"] = "市场代理模型"

    calls: dict[str, object] = {}

    def fake_load_market_rankings(horizon_days, positive_return):
        calls["rankings_args"] = (horizon_days, positive_return)
        return rankings

    def fake_build_focus_board(*, board_size, custom_watchlist, horizon_days, positive_return, ranking_by):
        calls["board_args"] = {
            "board_size": board_size,
            "custom_watchlist": custom_watchlist,
            "horizon_days": horizon_days,
            "positive_return": positive_return,
            "ranking_by": ranking_by,
        }
        return board

    def fake_run_daily_review_maintenance(
        weighted_board,
        *,
        horizon_days,
        positive_return,
        ranking_by,
        board_size,
        rolling_review_days,
    ):
        calls["maintenance_args"] = {
            "horizon_days": horizon_days,
            "positive_return": positive_return,
            "ranking_by": ranking_by,
            "board_size": board_size,
            "rolling_review_days": rolling_review_days,
        }
        calls["maintenance_board"] = weighted_board.copy()
        return {"snapshot_created": True, "new_reviews": 1}

    monkeypatch.setattr(migration, "load_market_rankings", fake_load_market_rankings)
    monkeypatch.setattr(migration, "_build_focus_board", fake_build_focus_board)
    monkeypatch.setattr(migration, "run_daily_review_maintenance", fake_run_daily_review_maintenance)

    result = migration.regenerate_supported_default_artifacts()

    assert calls["rankings_args"] == (5, 0.03)
    assert calls["board_args"] == {
        "board_size": 50,
        "custom_watchlist": (),
        "horizon_days": 5,
        "positive_return": 0.03,
        "ranking_by": "关注分数",
    }
    assert calls["maintenance_args"] == {
        "horizon_days": 5,
        "positive_return": 0.03,
        "ranking_by": "关注分数",
        "board_size": 50,
        "rolling_review_days": 20,
    }
    weighted_board = calls["maintenance_board"]
    assert weighted_board["launch_window_confidence_weight"].tolist() == [0.0, 0.0]
    assert result["target_config"]["horizon_days"] == 5
    assert result["target_config"]["positive_return"] == 0.03
    assert result["board"]["symbols"] == ["600001", "600002"]
    assert result["maintenance"]["new_reviews"] == 1


def test_regenerate_supported_default_artifacts_can_force_refresh_rankings(monkeypatch):
    refreshed = pd.DataFrame([{"symbol": "600001"}])
    refreshed.attrs["market_data_date"] = "2026-05-28"
    refreshed.attrs["latest_market_data_date"] = "2026-05-28"
    refreshed.attrs["model_source_label"] = "市场代理模型"

    board = pd.DataFrame([{"symbol": "600001", "selection_score": 88.0}])
    board.attrs["market_data_date"] = "2026-05-28"
    board.attrs["latest_market_data_date"] = "2026-05-28"
    board.attrs["model_source_label"] = "市场代理模型"

    calls: dict[str, object] = {"load_calls": 0}

    def fake_load_market_rankings(horizon_days, positive_return):
        calls["load_calls"] = int(calls["load_calls"]) + 1
        return pd.DataFrame([{"symbol": "stale"}])

    def fake_build_ranked_market_snapshot(horizon_days, positive_return):
        calls["refresh_args"] = (horizon_days, positive_return)
        return refreshed.copy(), "history"

    def fake_write_market_rankings_cache(ranked, horizon_days, positive_return, data_mode):
        calls["write_args"] = (len(ranked), horizon_days, positive_return, data_mode)

    def fake_build_focus_board(*, board_size, custom_watchlist, horizon_days, positive_return, ranking_by):
        calls["board_args"] = (board_size, custom_watchlist, horizon_days, positive_return, ranking_by)
        return board

    def fake_run_daily_review_maintenance(weighted_board, **kwargs):
        calls["maintenance_board"] = weighted_board.copy()
        return {"snapshot_created": True, "new_reviews": 1}

    monkeypatch.setattr(migration, "load_market_rankings", fake_load_market_rankings)
    monkeypatch.setattr(migration, "_build_ranked_market_snapshot", fake_build_ranked_market_snapshot)
    monkeypatch.setattr(migration, "_write_market_rankings_cache", fake_write_market_rankings_cache)
    monkeypatch.setattr(migration, "_build_focus_board", fake_build_focus_board)
    monkeypatch.setattr(migration, "run_daily_review_maintenance", fake_run_daily_review_maintenance)

    result = migration.regenerate_supported_default_artifacts(force_refresh=True)

    assert calls["load_calls"] == 0
    assert calls["refresh_args"] == (5, 0.03)
    assert calls["write_args"] == (1, 5, 0.03, "history")
    assert calls["board_args"] == (50, (), 5, 0.03, "关注分数")
    assert result["target_config"]["force_refresh"] is True
    assert result["rankings"]["cache_stale"] is False


def test_regenerate_supported_default_artifacts_can_stop_after_rankings(monkeypatch):
    rankings = pd.DataFrame([{"symbol": "600001"}])
    rankings.attrs["market_data_date"] = "2026-05-28"
    rankings.attrs["latest_market_data_date"] = "2026-05-28"
    rankings.attrs["model_source_label"] = "市场代理模型"
    rankings.attrs["cache_stale"] = False

    calls: dict[str, int] = {"board": 0, "maintenance": 0}

    monkeypatch.setattr(migration, "load_market_rankings", lambda horizon_days, positive_return: rankings.copy())

    def fail_build_focus_board(**kwargs):
        calls["board"] += 1
        raise AssertionError("board build should be skipped in rankings-only mode")

    def fail_run_daily_review_maintenance(*args, **kwargs):
        calls["maintenance"] += 1
        raise AssertionError("maintenance should be skipped in rankings-only mode")

    monkeypatch.setattr(migration, "_build_focus_board", fail_build_focus_board)
    monkeypatch.setattr(migration, "run_daily_review_maintenance", fail_run_daily_review_maintenance)

    result = migration.regenerate_supported_default_artifacts(rankings_only=True)

    assert calls["board"] == 0
    assert calls["maintenance"] == 0
    assert result["target_config"]["rankings_only"] is True
    assert result["board"] is None
    assert result["maintenance"] == {"skipped": True, "reason": "rankings_only"}
    assert result["rankings"]["row_count"] == 1


def test_regenerate_supported_default_artifacts_can_warm_stores_only(monkeypatch):
    universe = pd.DataFrame(
        [
            {"symbol": "600001", "name": "A"},
            {"symbol": "600002", "name": "B"},
        ]
    )
    feature_store = pd.DataFrame([{"symbol": "600001"}, {"symbol": "600002"}])
    candidate_pool = pd.DataFrame([{"symbol": "600001"}])
    dynamic_pool = pd.DataFrame([{"symbol": "600002"}])
    calls: dict[str, object] = {}

    monkeypatch.setattr(migration, "fetch_a_share_universe", lambda: universe.copy())
    monkeypatch.setattr(migration, "_latest_market_close_date", lambda: "2026-05-28")

    def fake_build_market_daily_feature_store(input_universe, market_data_date, force_rebuild=False):
        calls["feature"] = (len(input_universe), market_data_date, force_rebuild)
        return feature_store.copy()

    def fake_build_market_candidate_pool_store(input_universe, market_data_date, feature_store=None, force_rebuild=False):
        calls["candidate_pool"] = (len(input_universe), market_data_date, len(feature_store), force_rebuild)
        return candidate_pool.copy()

    def fake_build_market_dynamic_fallback_pool_store(input_universe, market_data_date, feature_store=None, force_rebuild=False):
        calls["dynamic_pool"] = (len(input_universe), market_data_date, len(feature_store), force_rebuild)
        return dynamic_pool.copy()

    monkeypatch.setattr(migration, "build_market_daily_feature_store", fake_build_market_daily_feature_store)
    monkeypatch.setattr(migration, "build_market_candidate_pool_store", fake_build_market_candidate_pool_store)
    monkeypatch.setattr(migration, "build_market_dynamic_fallback_pool_store", fake_build_market_dynamic_fallback_pool_store)

    result = migration.regenerate_supported_default_artifacts(stores_only=True, force_refresh=True)

    assert calls["feature"] == (2, "2026-05-28", True)
    assert calls["candidate_pool"] == (2, "2026-05-28", 2, True)
    assert calls["dynamic_pool"] == (2, "2026-05-28", 2, True)
    assert result["target_config"]["stores_only"] is True
    assert result["target_config"]["force_refresh"] is True
    assert result["feature_store"]["row_count"] == 2
    assert result["candidate_pool"]["row_count"] == 1
    assert result["dynamic_fallback_pool"]["row_count"] == 1


def test_regenerate_supported_default_artifacts_can_warm_snapshot_stage_only(monkeypatch):
    snapshot_history = pd.DataFrame([{"symbol": "600001"}, {"symbol": "600002"}])

    monkeypatch.setattr(migration, "_latest_market_close_date", lambda: "2026-05-28")

    calls: dict[str, object] = {}

    def fake_load_incremental_market_snapshot_history(market_data_date, force_rebuild=False):
        calls["snapshot"] = (market_data_date, force_rebuild)
        return snapshot_history.copy()

    monkeypatch.setattr(migration, "load_incremental_market_snapshot_history", fake_load_incremental_market_snapshot_history)

    result = migration.regenerate_supported_default_artifacts(store_stage="snapshot", force_refresh=True)

    assert calls["snapshot"] == ("2026-05-28", True)
    assert result["target_config"]["store_stage"] == "snapshot"
    assert result["snapshot_history"]["row_count"] == 2


def test_regenerate_supported_default_artifacts_can_warm_feature_stage_only(monkeypatch):
    universe = pd.DataFrame([{"symbol": "600001", "name": "A"}])
    feature_store = pd.DataFrame([{"symbol": "600001"}])
    calls: dict[str, object] = {}

    monkeypatch.setattr(migration, "_latest_market_close_date", lambda: "2026-05-28")
    monkeypatch.setattr(migration, "fetch_a_share_universe", lambda: universe.copy())

    def fake_build_market_daily_feature_store(input_universe, market_data_date, force_rebuild=False):
        calls["feature"] = (len(input_universe), market_data_date, force_rebuild)
        return feature_store.copy()

    monkeypatch.setattr(migration, "build_market_daily_feature_store", fake_build_market_daily_feature_store)

    result = migration.regenerate_supported_default_artifacts(store_stage="features", force_refresh=True)

    assert calls["feature"] == (1, "2026-05-28", True)
    assert result["target_config"]["store_stage"] == "features"
    assert result["feature_store"]["row_count"] == 1


def test_regenerate_supported_default_artifacts_raises_on_empty_board(monkeypatch):
    monkeypatch.setattr(migration, "load_market_rankings", lambda horizon_days, positive_return: pd.DataFrame())
    monkeypatch.setattr(
        migration,
        "_build_focus_board",
        lambda **kwargs: pd.DataFrame(columns=["symbol", "selection_score"]),
    )

    with pytest.raises(ValueError, match="focus board is empty"):
        migration.regenerate_supported_default_artifacts()
