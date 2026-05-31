from types import SimpleNamespace

import pandas as pd

import a_share_predictor.dashboard as dashboard
import a_share_predictor.store as market_store


def test_dashboard_build_strategy_candidate_pool_delegates_to_store(monkeypatch):
    universe = pd.DataFrame([{"symbol": "600001", "name": "A"}, {"symbol": "600002", "name": "B"}])
    expected = pd.DataFrame(
        [
            {"symbol": "600001", "name": "A", "candidate_strategy": "strategy-1"},
            {"symbol": "600002", "name": "B", "candidate_strategy": "strategy-2"},
        ]
    )

    monkeypatch.setattr(dashboard, "build_market_candidate_pool_store", lambda universe, market_data_date: expected.copy())

    result = dashboard._build_strategy_candidate_pool(universe, "2026-04-21")

    assert result.equals(expected)


def test_dashboard_build_dynamic_fallback_candidate_pool_delegates_to_store(monkeypatch):
    universe = pd.DataFrame([{"symbol": "600001", "name": "A"}, {"symbol": "600002", "name": "B"}])
    expected = pd.DataFrame(
        [
            {"symbol": "600001", "name": "A", "candidate_strategy": "dynamic_fallback"},
            {"symbol": "600002", "name": "B", "candidate_strategy": "dynamic_fallback"},
        ]
    )

    monkeypatch.setattr(dashboard, "build_market_dynamic_fallback_pool_store", lambda universe, market_data_date: expected.copy())

    result = dashboard._build_dynamic_fallback_candidate_pool(universe, "2026-04-21")

    assert result.equals(expected)


def test_prepare_symbol_base_analysis_reuses_feature_store_row(monkeypatch):
    dashboard._prepare_symbol_base_analysis.cache_clear()
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-18", "2026-04-20"]),
            "close": [10.0, 10.5],
            "change_pct": [0.0, 5.0],
            "amount": [1.1e8, 2.2e8],
            "turnover": [2.0, 3.5],
        }
    )
    feature_row = {
        "snapshot": {"date": "2026-04-20", "change_pct": 5.0},
        "latest_features": {"signal": 2.0},
        "model_feature_values": {
            "market_regime_label": "trend",
            "launch_readiness": 78.0,
            "market_resonance": 74.0,
        },
        "stage_object": SimpleNamespace(label="trend", priority=2, structure_summary="ok"),
        "stage_label": "trend",
        "stage_priority": 2,
        "stage_summary": "ok",
        "stage_score": 72.0,
        "quant_signal_object": SimpleNamespace(total_score=66.0, primary_signal="trend"),
        "quant_score": 66.0,
        "quant_primary_signal": "trend",
        "rule_context_object": SimpleNamespace(board_label="main", price_limit_label="10%", rule_summary="ok"),
        "board_label": "main",
        "price_limit_label": "10%",
        "rule_summary": "ok",
        "launch_score": 61.0,
        "latest_price": 10.5,
        "change_pct": 5.0,
        "amount": 2.2e8,
        "turnover": 3.5,
        "consecutive_up_days": 1,
        "analysis_date": "2026-04-20",
    }

    monkeypatch.setattr(dashboard, "fetch_daily_history", lambda symbol, start_date=None: daily)
    monkeypatch.setattr(dashboard, "get_market_daily_feature_row", lambda symbol, market_data_date: feature_row)
    monkeypatch.setattr(dashboard, "_read_symbol_base_analysis_disk_cache", lambda symbol, market_data_date: None)
    monkeypatch.setattr(dashboard, "_write_symbol_base_analysis_disk_cache", lambda symbol, market_data_date, data: None)
    monkeypatch.setattr(
        dashboard,
        "build_daily_features",
        lambda daily_df: (_ for _ in ()).throw(AssertionError("should not rebuild features")),
    )
    monkeypatch.setattr(
        dashboard,
        "classify_stage",
        lambda daily_df: (_ for _ in ()).throw(AssertionError("should not rebuild stage")),
    )
    monkeypatch.setattr(
        dashboard,
        "evaluate_quant_signal",
        lambda daily_df, feature_df: (_ for _ in ()).throw(AssertionError("should not rebuild quant")),
    )
    monkeypatch.setattr(
        dashboard,
        "build_trading_rule_context",
        lambda symbol, name: (_ for _ in ()).throw(AssertionError("should not rebuild rules")),
    )

    result = dashboard._prepare_symbol_base_analysis("600001", "A", market_data_date="2026-04-20")

    assert result is not None
    assert result["analysis_date"] == "2026-04-20"
    assert result["stage_label"] == "trend"
    assert result["quant_score"] == 66.0
    assert result["latest_features"]["market_regime_label"] == "trend"
    assert result["launch_readiness_score"] == 78.0
    assert result["market_resonance_score"] == 74.0
    assert result["daily"]["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-04-18", "2026-04-20"]


def test_build_market_candidate_pool_store_selects_both_rule_sets(monkeypatch):
    monkeypatch.setattr(market_store, "_write_market_candidate_pool_store", lambda candidate_pool, market_data_date: None)
    monkeypatch.setattr(market_store, "_load_candidate_replay_profile", lambda: None)
    feature_store = pd.DataFrame(
        [
            {
                "symbol": "600001",
                "name": "A",
                "industry_name": "consumer",
                "analysis_date": "2026-04-21",
                "latest_price": 12.7,
                "change_pct": 3.7,
                "amount": 2.5e8,
                "turnover": 3.8,
                "ret_3d_pct": 4.2,
                "ret_5d_pct": 6.8,
                "ret_10d_pct": 18.0,
                "ret_15d_pct": 22.0,
                "ret_20d_pct": 28.0,
                "ma5": 12.4,
                "ma10": 12.1,
                "ma20": 11.6,
                "high_10": 12.9,
                "distance_to_high_10_pct": 1.6,
                "max_gain_10_pct": 18.0,
                "pullback_days": 4,
                "pullback_volume_decay": True,
                "pullback_kept_ma10": True,
                "industry_ret_2d_pct": 5.2,
                "industry_rank_2d": 2,
                "industry_up_count": 5,
                "industry_top2d_flag": True,
                "consecutive_up_days": 1,
                "stage_label": "trend",
                "stage_priority": "P1",
                "quant_score": 72.0,
                "launch_score": 68.0,
                "board_label": "main",
                "price_limit_label": "10%",
            },
            {
                "symbol": "600002",
                "name": "B",
                "industry_name": "consumer",
                "analysis_date": "2026-04-21",
                "latest_price": 13.7,
                "change_pct": 8.0,
                "amount": 3.8e8,
                "turnover": 6.2,
                "ret_3d_pct": 12.0,
                "ret_5d_pct": 18.0,
                "ret_10d_pct": 25.0,
                "ret_15d_pct": 30.0,
                "ret_20d_pct": 32.0,
                "ma5": 12.2,
                "ma10": 11.8,
                "ma20": 11.2,
                "high_10": 13.7,
                "distance_to_high_10_pct": 0.0,
                "max_gain_10_pct": 24.0,
                "pullback_days": 2,
                "pullback_volume_decay": False,
                "pullback_kept_ma10": True,
                "industry_ret_2d_pct": 5.2,
                "industry_rank_2d": 2,
                "industry_up_count": 5,
                "industry_top2d_flag": True,
                "consecutive_up_days": 2,
                "stage_label": "breakout",
                "stage_priority": "P1",
                "quant_score": 78.0,
                "launch_score": 74.0,
                "board_label": "main",
                "price_limit_label": "10%",
                "snapshot": {
                    "close_vs_ma20": 0.12,
                    "volume_ratio_5": 1.8,
                    "range_position_20": 0.96,
                    "upper_shadow_ratio": 0.08,
                },
            },
            {
                "symbol": "600003",
                "name": "C",
                "industry_name": "consumer",
                "analysis_date": "2026-04-21",
                "latest_price": 11.6,
                "change_pct": 3.1,
                "amount": 2.2e8,
                "turnover": 3.6,
                "ret_3d_pct": 4.2,
                "ret_5d_pct": 7.8,
                "ret_10d_pct": 10.4,
                "ret_15d_pct": 13.5,
                "ret_20d_pct": 15.8,
                "ma5": 11.3,
                "ma10": 11.15,
                "ma20": 10.95,
                "high_10": 11.75,
                "distance_to_high_10_pct": 1.28,
                "max_gain_10_pct": 19.0,
                "pullback_days": 2,
                "pullback_volume_decay": False,
                "pullback_kept_ma10": True,
                "industry_ret_2d_pct": 3.8,
                "industry_rank_2d": 6,
                "industry_up_count": 2,
                "industry_top2d_flag": False,
                "consecutive_up_days": 1,
                "stage_label": "launch",
                "stage_priority": "P2",
                "quant_score": 67.0,
                "launch_score": 70.0,
                "launch_readiness_score": 73.0,
                "market_resonance_score": 66.0,
                "board_label": "main",
                "price_limit_label": "10%",
                "snapshot": {
                    "close_vs_ma20": 0.051,
                    "volume_ratio_5": 1.35,
                    "range_position_20": 0.68,
                    "upper_shadow_ratio": 0.12,
                },
            },
        ]
    )
    feature_store = pd.concat(
        [
            feature_store,
            pd.DataFrame(
                [
                    {
                        **feature_store.iloc[0].to_dict(),
                        "symbol": "300001",
                        "name": "GrowthA",
                        "board_label": "创业板",
                    },
                    {
                        **feature_store.iloc[1].to_dict(),
                        "symbol": "688001",
                        "name": "StarA",
                        "board_label": "科创板",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    universe = feature_store[["symbol", "name"]].copy()

    result = market_store.build_market_candidate_pool_store(
        universe,
        "2026-04-21",
        feature_store=feature_store,
        force_rebuild=True,
    )

    assert set(result["symbol"]) == {"600001", "600002", "600003"}
    assert not result["symbol"].astype(str).str.startswith(("300", "301", "688", "689")).any()
    assert set(result["candidate_strategy"]) == {"策略1", "策略2", "strategy3"}
    assert result["strategy_pass"].all()
    assert (result["strategy_rank"] == result["candidate_priority"]).all()
    assert result.set_index("symbol").loc["600003", "candidate_strategy"] == "strategy3"
    strategy2_reason = result.set_index("symbol").loc["600002", "candidate_reason"]
    assert "板块2日排名" in strategy2_reason
    assert "板块上涨家数" in strategy2_reason
    assert (result["candidate_reason"].astype(str).str.len() > 0).all()


def test_build_market_candidate_pool_store_applies_vetoes_and_launch_labels(monkeypatch):
    monkeypatch.setattr(market_store, "_write_market_candidate_pool_store", lambda candidate_pool, market_data_date: None)
    monkeypatch.setattr(market_store, "_load_candidate_replay_profile", lambda: None)

    strategy1_base = {
        "name": "A",
        "industry_name": "consumer",
        "analysis_date": "2026-04-21",
        "latest_price": 12.7,
        "change_pct": 3.7,
        "amount": 2.5e8,
        "turnover": 3.8,
        "ret_3d_pct": 4.2,
        "ret_5d_pct": 6.8,
        "ret_10d_pct": 18.0,
        "ret_15d_pct": 22.0,
        "ret_20d_pct": 28.0,
        "ma5": 12.4,
        "ma10": 12.1,
        "ma20": 11.6,
        "high_10": 12.9,
        "distance_to_high_10_pct": 1.6,
        "max_gain_10_pct": 18.0,
        "pullback_days": 4,
        "pullback_volume_decay": True,
        "pullback_kept_ma10": True,
        "industry_ret_2d_pct": 5.2,
        "industry_rank_2d": 2,
        "industry_up_count": 5,
        "industry_top2d_flag": True,
        "consecutive_up_days": 1,
        "stage_label": "trend",
        "stage_priority": "P1",
        "quant_score": 72.0,
        "launch_score": 68.0,
        "launch_readiness_score": 72.0,
        "market_resonance_score": 74.0,
        "board_label": "main",
        "price_limit_label": "10%",
        "snapshot": {"volume_ratio_5": 1.05, "range_position_20": 0.62, "upper_shadow_ratio": 0.08, "close_vs_ma20": 0.05},
    }
    strategy2_base = {
        **strategy1_base,
        "latest_price": 13.7,
        "change_pct": 8.0,
        "amount": 3.8e8,
        "turnover": 5.6,
        "ret_3d_pct": 12.0,
        "ret_5d_pct": 18.0,
        "ret_15d_pct": 30.0,
        "ret_20d_pct": 22.0,
        "ma5": 12.2,
        "ma10": 11.8,
        "ma20": 11.2,
        "high_10": 13.7,
        "distance_to_high_10_pct": 0.0,
        "max_gain_10_pct": 18.0,
        "pullback_days": 2,
        "pullback_volume_decay": False,
        "stage_label": "breakout",
        "launch_readiness_score": 84.0,
        "market_resonance_score": 78.0,
        "snapshot": {"volume_ratio_5": 1.45, "range_position_20": 0.65, "upper_shadow_ratio": 0.07, "close_vs_ma20": 0.08},
    }
    feature_store = pd.DataFrame(
        [
            {**strategy1_base, "symbol": "600101"},
            {
                **strategy1_base,
                "symbol": "600102",
                "snapshot": {"volume_ratio_5": 1.90, "range_position_20": 0.64, "upper_shadow_ratio": 0.32, "close_vs_ma20": 0.05},
            },
            {**strategy2_base, "symbol": "600201"},
            {
                **strategy2_base,
                "symbol": "600202",
                "industry_up_count": 3,
                "industry_ret_2d_pct": 0.6,
                "snapshot": {"volume_ratio_5": 1.70, "range_position_20": 0.55, "upper_shadow_ratio": 0.34, "close_vs_ma20": 0.08},
            },
        ]
    )

    result = market_store.build_market_candidate_pool_store(
        feature_store[["symbol", "name"]],
        "2026-04-21",
        feature_store=feature_store,
        force_rebuild=True,
    )

    assert set(result["symbol"]) == {"600101", "600201"}
    assert "600102" not in set(result["symbol"])
    assert "600202" not in set(result["symbol"])
    assert {
        "launch_readiness",
        "breakout_quality",
        "resonance_quality",
        "board_resonance_strength",
        "long_setup_quality",
        "crowding_risk",
        "risk_of_late_entry",
        "launch_phase_label",
    }.issubset(result.columns)
    assert result.set_index("symbol").loc["600201", "launch_phase_label"] == "刚启动"
    assert result.set_index("symbol").loc["600201", "candidate_reason"].find("突破质量") >= 0
    assert "拥挤风险" in result.set_index("symbol").loc["600201", "candidate_reason"]


def test_build_market_candidate_pool_store_penalizes_quant_crowding_without_new_strategy(monkeypatch):
    monkeypatch.setattr(market_store, "_write_market_candidate_pool_store", lambda candidate_pool, market_data_date: None)
    monkeypatch.setattr(market_store, "_load_candidate_replay_profile", lambda: None)

    base = {
        "name": "A",
        "industry_name": "robotics",
        "analysis_date": "2026-04-21",
        "latest_price": 14.0,
        "change_pct": 8.2,
        "amount": 4.2e8,
        "turnover": 5.8,
        "ret_3d_pct": 12.5,
        "ret_5d_pct": 18.0,
        "ret_10d_pct": 20.0,
        "ret_15d_pct": 31.0,
        "ret_20d_pct": 24.0,
        "ma5": 13.1,
        "ma10": 12.6,
        "ma20": 11.8,
        "high_10": 14.0,
        "distance_to_high_10_pct": 0.0,
        "max_gain_10_pct": 18.0,
        "pullback_days": 2,
        "pullback_volume_decay": False,
        "pullback_kept_ma10": True,
        "industry_ret_2d_pct": 5.0,
        "industry_rank_2d": 1,
        "industry_up_count": 7,
        "industry_top2d_flag": True,
        "consecutive_up_days": 1,
        "stage_label": "breakout",
        "stage_priority": "P1",
        "quant_score": 74.0,
        "launch_score": 72.0,
        "launch_readiness_score": 82.0,
        "market_resonance_score": 80.0,
        "board_label": "main",
        "price_limit_label": "10%",
        "snapshot": {"volume_ratio_5": 1.35, "range_position_20": 0.68, "upper_shadow_ratio": 0.06, "close_vs_ma20": 0.08},
    }
    crowded = {
        **base,
        "symbol": "600302",
        "turnover": 10.0,
        "turnover_2d_avg": 10.0,
        "snapshot": {"volume_ratio_5": 1.70, "range_position_20": 0.68, "upper_shadow_ratio": 0.20, "close_vs_ma20": 0.08},
    }
    vetoed = {
        **base,
        "symbol": "600303",
        "turnover": 22.0,
        "turnover_2d_avg": 22.0,
        "snapshot": {"volume_ratio_5": 2.40, "range_position_20": 0.72, "upper_shadow_ratio": 0.21, "close_vs_ma20": 0.08},
    }
    feature_store = pd.DataFrame([{**base, "symbol": "600301"}, crowded, vetoed])

    result = market_store.build_market_candidate_pool_store(
        feature_store[["symbol", "name"]],
        "2026-04-21",
        feature_store=feature_store,
        force_rebuild=True,
    )

    indexed = result.set_index("symbol")
    assert set(indexed.index) == {"600301", "600302"}
    assert "策略3" not in set(result["candidate_strategy"].astype(str))
    assert indexed.loc["600302", "crowding_risk"] > indexed.loc["600301", "crowding_risk"]
    assert indexed.loc["600301", "strategy_rank"] > indexed.loc["600302", "strategy_rank"]
    assert "600303" not in set(result["symbol"])


def test_build_market_candidate_pool_store_applies_replay_strategy_fit(monkeypatch):
    monkeypatch.setattr(market_store, "_write_market_candidate_pool_store", lambda candidate_pool, market_data_date: None)
    monkeypatch.setattr(
        market_store,
        "_load_candidate_replay_profile",
        lambda: {
            "review_days": 5,
            "review_stocks": 80,
            "stage_edges": {"launch": 0.18, "trend": -0.08},
            "stage_supports": {"launch": 40, "trend": 40},
            "quant_bucket_edges": {"70-80": 0.10, "60-70": -0.08},
            "quant_bucket_supports": {"70-80": 40, "60-70": 40},
            "launch_bucket_edges": {"65-80": 0.12, "50-65": -0.06},
            "launch_bucket_supports": {"65-80": 40, "50-65": 40},
            "resonance_bucket_edges": {"60-75": 0.12, "45-60": -0.06},
            "resonance_bucket_supports": {"60-75": 40, "45-60": 40},
        },
    )
    base = {
        "name": "A",
        "industry_name": "consumer",
        "analysis_date": "2026-04-21",
        "latest_price": 11.6,
        "change_pct": 3.1,
        "amount": 2.2e8,
        "turnover": 3.6,
        "ret_3d_pct": 4.2,
        "ret_5d_pct": 7.8,
        "ret_10d_pct": 10.4,
        "ret_15d_pct": 13.5,
        "ret_20d_pct": 15.8,
        "ma5": 11.3,
        "ma10": 11.15,
        "ma20": 10.95,
        "high_10": 11.75,
        "distance_to_high_10_pct": 1.28,
        "max_gain_10_pct": 19.0,
        "pullback_days": 4,
        "pullback_volume_decay": True,
        "pullback_kept_ma10": True,
        "industry_ret_2d_pct": 3.8,
        "industry_rank_2d": 6,
        "industry_up_count": 2,
        "industry_top2d_flag": False,
        "consecutive_up_days": 1,
        "board_label": "main",
        "price_limit_label": "10%",
        "snapshot": {
            "close_vs_ma20": 0.051,
            "volume_ratio_5": 1.35,
            "range_position_20": 0.68,
            "upper_shadow_ratio": 0.12,
        },
    }
    feature_store = pd.DataFrame(
        [
            {
                **base,
                "symbol": "600011",
                "stage_label": "trend",
                "quant_score": 66.0,
                "launch_score": 59.0,
                "launch_readiness_score": 60.0,
                "market_resonance_score": 55.0,
            },
            {
                **base,
                "symbol": "600012",
                "stage_label": "launch",
                "quant_score": 74.0,
                "launch_score": 70.0,
                "launch_readiness_score": 70.0,
                "market_resonance_score": 66.0,
            },
        ]
    )

    result = market_store.build_market_candidate_pool_store(
        feature_store[["symbol", "name"]],
        "2026-04-21",
        feature_store=feature_store,
        force_rebuild=True,
    )

    assert result.iloc[0]["symbol"] == "600012"
    assert result.iloc[0]["strategy_fit_score"] > result.iloc[1]["strategy_fit_score"]
    assert result["replay_profile_applied"].all()


def test_build_market_daily_feature_store_skips_symbols_without_exact_market_date(monkeypatch):
    monkeypatch.setattr(market_store, "_write_market_daily_feature_store", lambda feature_store, market_data_date: None)
    stock_basic = pd.DataFrame(
        [
            {"symbol": "600001", "name": "A", "industry": "consumer", "market": ""},
            {"symbol": "600002", "name": "B", "industry": "consumer", "market": ""},
        ]
    )
    latest_trade_date = pd.Timestamp("2026-04-21")
    previous_trade_date = pd.Timestamp("2026-04-20")
    latest_snapshot = pd.DataFrame(
        [
            {"symbol": "600001", "trade_date": latest_trade_date, "close": 12.0, "pct_chg": 4.0},
            {"symbol": "600002", "trade_date": latest_trade_date, "close": 11.0, "pct_chg": 2.0},
        ]
    )
    previous_snapshot = pd.DataFrame(
        [
            {"symbol": "600001", "trade_date": previous_trade_date, "close": 11.5, "pct_chg": 1.0},
            {"symbol": "600002", "trade_date": previous_trade_date, "close": 10.8, "pct_chg": 0.8},
        ]
    )

    def make_history(symbol: str, *, include_target_day: bool) -> pd.DataFrame:
        dates = pd.bdate_range("2026-03-18", periods=25)
        closes = [10.0 + index * 0.1 for index in range(len(dates))]
        frame = pd.DataFrame(
            {
                "date": dates,
                "close": closes,
                "high": [value * 1.01 for value in closes],
                "low": [value * 0.99 for value in closes],
                "vol": [100000 - index * 1000 for index in range(len(dates))],
                "amount": [2.0e8 + index * 1e6 for index in range(len(dates))],
                "turnover": [3.0 + index * 0.05 for index in range(len(dates))],
                "change_pct": [0.0] + [0.8 for _ in range(len(dates) - 1)],
            }
        )
        if not include_target_day:
            frame = frame.iloc[:-1].reset_index(drop=True)
        return frame

    monkeypatch.setattr(market_store, "fetch_tushare_stock_basic_all_statuses", lambda: pd.DataFrame())
    monkeypatch.setattr(market_store, "fetch_tushare_stock_basic", lambda: stock_basic)
    monkeypatch.setattr(market_store, "fetch_tushare_recent_trade_dates", lambda end_date=None, limit=3: ["20260417", "20260420", "20260421"])
    monkeypatch.setattr(
        market_store,
        "fetch_tushare_daily_snapshot",
        lambda trade_date: latest_snapshot.copy() if str(trade_date) == "20260421" else previous_snapshot.copy(),
    )
    monkeypatch.setattr(
        market_store,
        "fetch_daily_history",
        lambda symbol, start_date=None: make_history(str(symbol), include_target_day=str(symbol) == "600001"),
    )
    monkeypatch.setattr(
        market_store,
        "build_daily_features",
        lambda daily_df: pd.DataFrame(
            {
                "range_position_20": [0.6 for _ in range(len(daily_df))],
                "upper_shadow_ratio": [0.1 for _ in range(len(daily_df))],
            }
        ),
    )
    monkeypatch.setattr(
        market_store,
        "build_latest_snapshot",
        lambda daily_df, feature_df: {
            "date": daily_df["date"].iloc[-1].strftime("%Y-%m-%d"),
            "change_pct": float(daily_df["change_pct"].iloc[-1]),
            "close_vs_ma20": 0.05,
            "ret_20": 0.12,
            "volume_ratio_5": 1.2,
            "breakout_distance_20": 0.01,
        },
    )
    monkeypatch.setattr(
        market_store,
        "classify_stage",
        lambda daily_df: SimpleNamespace(code="trend", label="trend", priority="P1", structure_summary="ok"),
    )
    monkeypatch.setattr(
        market_store,
        "evaluate_quant_signal",
        lambda daily_df, feature_df: SimpleNamespace(total_score=70.0, primary_signal="trend"),
    )
    monkeypatch.setattr(
        market_store,
        "build_trading_rule_context",
        lambda symbol, name: SimpleNamespace(board_label="main", price_limit_label="10%", rule_summary="ok"),
    )
    monkeypatch.setattr(market_store, "main_rise_start_score", lambda latest_features: 62.0)
    monkeypatch.setattr(market_store, "stage_numeric_score", lambda stage, latest_features: 75.0)
    monkeypatch.setattr(
        market_store,
        "_prepare_live_feature_frame",
        lambda daily_df, latest_feature_values=None, symbol=None: pd.DataFrame([{"ret_20": 0.12}]),
    )

    feature_store = market_store.build_market_daily_feature_store(
        stock_basic[["symbol", "name"]],
        "2026-04-21",
        force_rebuild=True,
    )

    assert feature_store["symbol"].tolist() == ["600001"]
    assert feature_store.iloc[0]["analysis_date"] == "2026-04-21"


def test_build_market_daily_feature_store_prefers_batch_snapshot_history(monkeypatch):
    monkeypatch.setattr(market_store, "_write_market_daily_feature_store", lambda feature_store, market_data_date: None)
    monkeypatch.setattr(market_store, "_write_market_snapshot_history_store", lambda snapshot_history, market_data_date: None)
    stock_basic = pd.DataFrame(
        [
            {"symbol": "600001", "name": "A", "industry": "consumer", "market": ""},
            {"symbol": "600002", "name": "B", "industry": "tech", "market": ""},
        ]
    )
    trade_dates = [date.strftime("%Y%m%d") for date in pd.bdate_range("2026-03-18", periods=25)]

    def snapshot_for(trade_date: str) -> pd.DataFrame:
        index = trade_dates.index(str(trade_date))
        base_close_a = 10.0 + index * 0.12
        base_close_b = 8.0 + index * 0.08
        return pd.DataFrame(
            [
                {
                    "symbol": "600001",
                    "name": "A",
                    "industry": "consumer",
                    "market": "",
                    "trade_date": pd.Timestamp(str(trade_date)),
                    "open": base_close_a * 0.99,
                    "high": base_close_a * 1.01,
                    "low": base_close_a * 0.98,
                    "close": base_close_a,
                    "pre_close": base_close_a * 0.995,
                    "change": base_close_a * 0.005,
                    "pct_chg": 1.2,
                    "vol": 120000 + index * 1000,
                    "amount": 2.4e8 + index * 1.0e6,
                },
                {
                    "symbol": "600002",
                    "name": "B",
                    "industry": "tech",
                    "market": "",
                    "trade_date": pd.Timestamp(str(trade_date)),
                    "open": base_close_b * 0.99,
                    "high": base_close_b * 1.01,
                    "low": base_close_b * 0.98,
                    "close": base_close_b,
                    "pre_close": base_close_b * 0.995,
                    "change": base_close_b * 0.005,
                    "pct_chg": 0.9,
                    "vol": 90000 + index * 800,
                    "amount": 1.9e8 + index * 8.0e5,
                },
            ]
        )

    monkeypatch.setattr(market_store, "fetch_tushare_stock_basic_all_statuses", lambda: pd.DataFrame())
    monkeypatch.setattr(market_store, "fetch_tushare_stock_basic", lambda: stock_basic)
    monkeypatch.setattr(market_store, "fetch_tushare_recent_trade_dates", lambda end_date=None, limit=30: trade_dates[-limit:])
    monkeypatch.setattr(market_store, "fetch_tushare_daily_snapshot", snapshot_for)
    monkeypatch.setattr(
        market_store,
        "fetch_daily_history",
        lambda symbol, start_date=None: (_ for _ in ()).throw(AssertionError("should not fetch per-symbol history")),
    )
    monkeypatch.setattr(
        market_store,
        "build_daily_features",
        lambda daily_df: pd.DataFrame(
            {
                "range_position_20": [0.65 for _ in range(len(daily_df))],
                "upper_shadow_ratio": [0.08 for _ in range(len(daily_df))],
            }
        ),
    )
    monkeypatch.setattr(
        market_store,
        "build_latest_snapshot",
        lambda daily_df, feature_df: {
            "date": daily_df["date"].iloc[-1].strftime("%Y-%m-%d"),
            "change_pct": float(daily_df["change_pct"].iloc[-1]),
            "close_vs_ma20": 0.06,
            "ret_20": 0.14,
            "ret_60": 0.18,
            "volume_ratio_5": 1.25,
            "breakout_distance_20": 0.02,
        },
    )
    monkeypatch.setattr(
        market_store,
        "classify_stage",
        lambda daily_df: SimpleNamespace(code="trend", label="trend", priority="P1", structure_summary="ok"),
    )
    monkeypatch.setattr(
        market_store,
        "evaluate_quant_signal",
        lambda daily_df, feature_df: SimpleNamespace(total_score=71.0, primary_signal="trend"),
    )
    monkeypatch.setattr(
        market_store,
        "build_trading_rule_context",
        lambda symbol, name: SimpleNamespace(board_label="main", price_limit_label="10%", rule_summary="ok"),
    )
    monkeypatch.setattr(market_store, "main_rise_start_score", lambda latest_features: 64.0)
    monkeypatch.setattr(market_store, "stage_numeric_score", lambda stage, latest_features: 76.0)
    captured = {"latest_feature_values": None}

    def fake_prepare_live_feature_frame(daily_df, latest_feature_values=None, symbol=None):
        captured["latest_feature_values"] = latest_feature_values
        return pd.DataFrame([{"ret_20": 0.14}])

    monkeypatch.setattr(market_store, "_prepare_live_feature_frame", fake_prepare_live_feature_frame)

    feature_store = market_store.build_market_daily_feature_store(
        stock_basic[["symbol", "name"]],
        "2026-04-21",
        force_rebuild=True,
    )

    assert set(feature_store["symbol"]) == {"600001", "600002"}
    assert feature_store["analysis_date"].nunique() == 1
    assert float(feature_store["turnover"].min()) > 0
    assert captured["latest_feature_values"] is not None


def test_build_market_daily_feature_store_reuses_snapshot_history_when_live_snapshot_context_is_empty(monkeypatch):
    monkeypatch.setattr(market_store, "_write_market_daily_feature_store", lambda feature_store, market_data_date: None)
    stock_basic = pd.DataFrame(
        [
            {"symbol": "600001", "name": "A", "industry": "consumer", "market": ""},
            {"symbol": "600002", "name": "B", "industry": "tech", "market": ""},
        ]
    )
    trade_dates = pd.bdate_range("2026-03-18", periods=25)
    snapshot_history_rows = []
    for index, trade_date in enumerate(trade_dates):
        base_close_a = 10.0 + index * 0.12
        base_close_b = 8.0 + index * 0.08
        snapshot_history_rows.extend(
            [
                {
                    "symbol": "600001",
                    "name": "A",
                    "industry": "consumer",
                    "market": "",
                    "trade_date": trade_date,
                    "date": trade_date,
                    "open": base_close_a * 0.99,
                    "high": base_close_a * 1.01,
                    "low": base_close_a * 0.98,
                    "close": base_close_a,
                    "pre_close": base_close_a * 0.995,
                    "change": base_close_a * 0.005,
                    "pct_chg": 1.2,
                    "change_pct": 1.2,
                    "vol": 120000 + index * 1000,
                    "volume": 120000 + index * 1000,
                    "amount": 2.4e8 + index * 1.0e6,
                    "turnover": 3.0 + index * 0.02,
                },
                {
                    "symbol": "600002",
                    "name": "B",
                    "industry": "tech",
                    "market": "",
                    "trade_date": trade_date,
                    "date": trade_date,
                    "open": base_close_b * 0.99,
                    "high": base_close_b * 1.01,
                    "low": base_close_b * 0.98,
                    "close": base_close_b,
                    "pre_close": base_close_b * 0.995,
                    "change": base_close_b * 0.005,
                    "pct_chg": 0.9,
                    "change_pct": 0.9,
                    "vol": 90000 + index * 800,
                    "volume": 90000 + index * 800,
                    "amount": 1.9e8 + index * 8.0e5,
                    "turnover": 2.5 + index * 0.02,
                },
            ]
        )
    snapshot_history = pd.DataFrame(snapshot_history_rows)

    monkeypatch.setattr(market_store, "fetch_tushare_stock_basic_all_statuses", lambda: pd.DataFrame())
    monkeypatch.setattr(market_store, "fetch_tushare_stock_basic", lambda: stock_basic)
    monkeypatch.setattr(market_store, "_build_strategy_snapshot_context", lambda market_data_date: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(market_store, "load_incremental_market_snapshot_history", lambda *args, **kwargs: snapshot_history.copy())
    monkeypatch.setattr(
        market_store,
        "fetch_daily_history",
        lambda symbol, start_date=None: (_ for _ in ()).throw(AssertionError("should not fetch per-symbol history")),
    )
    monkeypatch.setattr(
        market_store,
        "build_daily_features",
        lambda daily_df: pd.DataFrame(
            {
                "range_position_20": [0.65 for _ in range(len(daily_df))],
                "upper_shadow_ratio": [0.08 for _ in range(len(daily_df))],
            }
        ),
    )
    monkeypatch.setattr(
        market_store,
        "build_latest_snapshot",
        lambda daily_df, feature_df: {
            "date": daily_df["date"].iloc[-1].strftime("%Y-%m-%d"),
            "change_pct": float(daily_df["change_pct"].iloc[-1]),
            "close_vs_ma20": 0.06,
            "ret_20": 0.14,
            "ret_60": 0.18,
            "volume_ratio_5": 1.25,
            "breakout_distance_20": 0.02,
        },
    )
    monkeypatch.setattr(
        market_store,
        "classify_stage",
        lambda daily_df: SimpleNamespace(code="trend", label="trend", priority="P1", structure_summary="ok"),
    )
    monkeypatch.setattr(
        market_store,
        "evaluate_quant_signal",
        lambda daily_df, feature_df: SimpleNamespace(total_score=71.0, primary_signal="trend"),
    )
    monkeypatch.setattr(
        market_store,
        "build_trading_rule_context",
        lambda symbol, name: SimpleNamespace(board_label="main", price_limit_label="10%", rule_summary="ok"),
    )
    monkeypatch.setattr(market_store, "main_rise_start_score", lambda latest_features: 64.0)
    monkeypatch.setattr(market_store, "stage_numeric_score", lambda stage, latest_features: 76.0)
    monkeypatch.setattr(
        market_store,
        "_prepare_live_feature_frame",
        lambda daily_df, latest_feature_values=None, symbol=None: pd.DataFrame([{"ret_20": 0.14}]),
    )

    feature_store = market_store.build_market_daily_feature_store(
        stock_basic[["symbol", "name"]],
        "2026-04-21",
        force_rebuild=True,
    )

    assert set(feature_store["symbol"]) == {"600001", "600002"}
    assert feature_store["analysis_date"].nunique() == 1
    assert float(feature_store["turnover"].min()) > 0


def test_load_incremental_market_snapshot_history_reuses_previous_store(monkeypatch):
    previous_store = pd.DataFrame(
        [
            {
                "symbol": "600001",
                "trade_date": pd.Timestamp("2026-04-28"),
                "date": pd.Timestamp("2026-04-28"),
                "open": 10.0,
                "high": 10.2,
                "low": 9.9,
                "close": 10.1,
                "change_pct": 1.0,
                "volume": 120000,
                "vol": 120000,
                "amount": 2.0e8,
                "turnover": 3.0,
            }
        ]
    )
    latest_snapshot = pd.DataFrame(
        [
            {
                "symbol": "600001",
                "trade_date": pd.Timestamp("2026-04-29"),
                "open": 10.1,
                "high": 10.4,
                "low": 10.0,
                "close": 10.3,
                "pct_chg": 1.8,
                "vol": 130000,
                "amount": 2.2e8,
            }
        ]
    )
    writes: dict[str, object] = {}

    monkeypatch.setattr(
        market_store,
        "read_market_snapshot_history_store",
        lambda market_data_date: previous_store.copy() if str(market_data_date) == "2026-04-28" else None,
    )
    monkeypatch.setattr(
        market_store,
        "fetch_tushare_recent_trade_dates",
        lambda end_date=None, limit=20: ["20260428", "20260429"],
    )
    monkeypatch.setattr(market_store, "fetch_tushare_daily_snapshot", lambda trade_date: latest_snapshot.copy())
    monkeypatch.setattr(
        market_store,
        "_write_market_snapshot_history_store",
        lambda snapshot_history, market_data_date: writes.update({"rows": len(snapshot_history), "market_data_date": market_data_date}),
    )

    result = market_store.load_incremental_market_snapshot_history("2026-04-29", force_rebuild=False)

    assert result["trade_date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-04-28", "2026-04-29"]
    assert writes == {"rows": 2, "market_data_date": "2026-04-29"}


def test_load_incremental_market_snapshot_history_ignores_invalid_cached_store(monkeypatch):
    invalid_cached = pd.DataFrame(
        [
            {
                "symbol": "600001",
                "trade_date": pd.Timestamp("2024-12-09"),
                "date": pd.Timestamp("2024-12-09"),
                "close": 10.1,
            }
        ]
    )
    rebuilt = pd.DataFrame(
        [
            {
                "symbol": "600001",
                "trade_date": pd.Timestamp("2026-05-06"),
                "date": pd.Timestamp("2026-05-06"),
                "close": 10.8,
            }
        ]
    )

    monkeypatch.setattr(
        market_store,
        "read_market_snapshot_history_store",
        lambda market_data_date: invalid_cached.copy() if str(market_data_date) == "2026-05-06" else None,
    )
    monkeypatch.setattr(
        market_store,
        "_load_recent_snapshot_history",
        lambda market_data_date, lookback_sessions=140, progress_callback=None, force_rebuild=False: rebuilt.copy(),
    )

    result = market_store.load_incremental_market_snapshot_history("2026-05-06", force_rebuild=False)

    assert result["trade_date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-05-06"]


def test_build_market_dynamic_fallback_pool_store_uses_close_based_streaks(monkeypatch):
    monkeypatch.setattr(market_store, "_write_market_dynamic_fallback_store", lambda pool_df, market_data_date: None)
    monkeypatch.setattr(market_store, "_load_candidate_replay_profile", lambda: None)
    feature_store = pd.DataFrame(
        [
            {
                "symbol": "600001",
                "name": "A",
                "industry_name": "consumer",
                "analysis_date": "2026-04-21",
                "latest_price": 12.7,
                "change_pct": 3.7,
                "amount": 2.5e8,
                "turnover": 3.8,
                "consecutive_up_days": 4,
                "ret_3d_pct": 4.2,
                "ret_5d_pct": 6.8,
                "stage_label": "trend",
                "stage_priority": "P1",
                "quant_score": 72.0,
                "launch_score": 68.0,
                "board_label": "main",
                "price_limit_label": "10%",
            },
            {
                "symbol": "600002",
                "name": "B",
                "industry_name": "consumer",
                "analysis_date": "2026-04-21",
                "latest_price": 13.7,
                "change_pct": 8.0,
                "amount": 3.8e8,
                "turnover": 6.2,
                "consecutive_up_days": 2,
                "ret_3d_pct": 12.0,
                "ret_5d_pct": 18.0,
                "stage_label": "breakout",
                "stage_priority": "P1",
                "quant_score": 78.0,
                "launch_score": 74.0,
                "board_label": "main",
                "price_limit_label": "10%",
            },
            {
                "symbol": "600003",
                "name": "C",
                "industry_name": "tech",
                "analysis_date": "2026-04-21",
                "latest_price": 9.7,
                "change_pct": 2.2,
                "amount": 1.8e8,
                "turnover": 4.1,
                "consecutive_up_days": 5,
                "ret_3d_pct": 3.1,
                "ret_5d_pct": 5.4,
                "stage_label": "trend",
                "stage_priority": "P2",
                "quant_score": 64.0,
                "launch_score": 59.0,
                "board_label": "main",
                "price_limit_label": "10%",
            },
        ]
    )
    feature_store = pd.concat(
        [
            feature_store,
            pd.DataFrame(
                [
                    {
                        **feature_store.iloc[0].to_dict(),
                        "symbol": "300001",
                        "name": "GrowthA",
                        "board_label": "创业板",
                        "candidate_priority": 999.0,
                    },
                    {
                        **feature_store.iloc[2].to_dict(),
                        "symbol": "688001",
                        "name": "StarA",
                        "board_label": "科创板",
                        "candidate_priority": 999.0,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    result = market_store.build_market_dynamic_fallback_pool_store(
        feature_store[["symbol", "name"]],
        "2026-04-21",
        feature_store=feature_store,
        force_rebuild=True,
    )

    assert result["symbol"].tolist() == ["600003", "600001"]
    assert not result["symbol"].astype(str).str.startswith(("300", "301", "688", "689")).any()
    assert (result["candidate_strategy"] == "dynamic_fallback").all()
    assert result["candidate_priority"].iloc[0] >= result["candidate_priority"].iloc[1]
    assert result["candidate_reason"].str.contains("Close-based fallback").all()


def test_build_ranked_market_snapshot_reuses_candidate_analysis_cache(monkeypatch):
    universe = pd.DataFrame([{"symbol": "600001", "name": "A"}])
    cached_result = {
        "symbol": "600001",
        "name": "A",
        "attention_score": 81.0,
        "probability_up": 63.0,
        "amount": 10.0,
        "turnover": 2.5,
        "latest_price": 12.3,
        "change_pct": 2.1,
        "consecutive_up_days": 3,
        "analysis_date": "2026-04-21",
        "industry_name": "consumer",
        "sector_label": "hot",
        "sector_score": 66.0,
        "quant_score": 72.0,
        "launch_score": 68.0,
        "precision_priority": 0,
        "precision_gate_label": "none",
        "precision_gate_threshold": 1.0,
        "precision_gate_precision": 0.0,
        "precision_gate_support": 0,
        "stage_label": "trend",
        "stage_priority": "P1",
        "stage_summary": "ok",
        "tomorrow_bias": "偏多",
        "tomorrow_setup": "观察",
        "tomorrow_buy_point": "buy",
        "tomorrow_sell_point": "sell",
        "tomorrow_plan_confidence": 60.0,
        "reason": "cached",
        "board_label": "main",
        "price_limit_label": "10%",
        "candidate_strategy": "strategy-1",
        "candidate_reason": "cached",
        "model_source": "market_wide",
        "model_source_label": "market wide",
        "model_result_status": "最新结果",
        "market_ret_5": 0.0,
        "market_ret_20": 0.0,
        "market_close_vs_ma20": 0.0,
        "market_volatility_10": 0.0,
        "market_range_position_20": 0.5,
        "ret_20": 0.12,
        "close_vs_ma20": 0.05,
        "breakout_distance_20": 0.01,
        "range_position_20": 0.6,
        "volume_ratio_5": 1.2,
        "upper_shadow_ratio": 0.1,
        "stretch_risk": 0.0,
        "risk_pressure": 0.0,
    }

    monkeypatch.setattr(dashboard, "load_a_share_universe", lambda: universe)
    monkeypatch.setattr(dashboard, "_latest_market_close_date", lambda *args, **kwargs: "2026-04-21")
    monkeypatch.setattr(dashboard, "_load_market_model_or_none", lambda *args, **kwargs: None)
    monkeypatch.setattr(dashboard, "_load_market_proxy_or_none", lambda *args, **kwargs: None)
    monkeypatch.setattr(dashboard, "build_market_daily_feature_store", lambda universe, market_data_date, progress_callback=None: pd.DataFrame())
    monkeypatch.setattr(
        dashboard,
        "build_market_candidate_pool_store",
        lambda universe, market_data_date, feature_store=None, progress_callback=None: pd.DataFrame(
            [{"symbol": "600001", "name": "A", "industry_name": "consumer", "candidate_strategy": "strategy-1", "candidate_reason": "cached"}]
        ),
    )
    monkeypatch.setattr(dashboard, "_build_market_context", lambda: {"industry_flow": pd.DataFrame()})
    monkeypatch.setattr(dashboard, "_read_candidate_analysis_cache", lambda *args, **kwargs: dict(cached_result))
    monkeypatch.setattr(
        dashboard,
        "_call_analyze_single_base",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not recompute candidate")),
    )

    ranked, mode = dashboard._build_ranked_market_snapshot(3, 0.1)

    assert mode == "strategy_candidate_pool"
    assert ranked["symbol"].tolist() == ["600001"]
    assert ranked.iloc[0]["reason"] == "cached"
