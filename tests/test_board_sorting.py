import pickle
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import a_share_predictor.dashboard as dashboard
from a_share_predictor.dashboard import (
    _build_display_board,
    _detail_display_context,
    _extract_market_data_date,
    _fallback_candidate_pool,
    _filter_focus_candidates,
    _read_market_rankings_cache,
    _sort_focus_board,
)


def test_sort_focus_board_by_attention_score():
    board = pd.DataFrame(
        [
            {"symbol": "000001", "attention_score": 88.0, "probability_up": 61.0, "amount": 10},
            {"symbol": "000002", "attention_score": 93.0, "probability_up": 58.0, "amount": 12},
            {"symbol": "000003", "attention_score": 90.0, "probability_up": 65.0, "amount": 11},
        ]
    )

    result = _sort_focus_board(board, ranking_by="关注分数", board_size=2)

    assert result["symbol"].tolist() == ["000002", "000003"]
    assert result["rank"].tolist() == [1, 2]


def test_sort_focus_board_by_probability_up():
    board = pd.DataFrame(
        [
            {"symbol": "000001", "attention_score": 88.0, "probability_up": 61.0, "amount": 10},
            {"symbol": "000002", "attention_score": 93.0, "probability_up": 58.0, "amount": 12},
            {"symbol": "000003", "attention_score": 90.0, "probability_up": 65.0, "amount": 11},
        ]
    )

    result = _sort_focus_board(board, ranking_by="上涨概率", board_size=2)

    assert result["symbol"].tolist() == ["000003", "000001"]
    assert result["rank"].tolist() == [1, 2]


def test_sort_focus_board_uses_launch_window_as_tie_breaker():
    board = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "attention_score": 88.0,
                "probability_up": 64.0,
                "amount": 10.0,
                "quant_score": 72.0,
                "stage_label": "主升启动",
                "launch_score": 72.0,
                "launch_readiness_score": 79.0,
                "market_resonance_score": 74.0,
                "launch_specialist_score": 82.0,
                "launch_regime_fit_score": 77.0,
                "launch_specialist_confidence": 72.0,
                "close_vs_ma20": 0.04,
                "breakout_distance_20": 0.01,
            },
            {
                "symbol": "000002",
                "attention_score": 88.0,
                "probability_up": 64.0,
                "amount": 12.0,
                "quant_score": 58.0,
                "stage_label": "高位分歧",
                "launch_score": 45.0,
                "launch_readiness_score": 40.0,
                "market_resonance_score": 43.0,
                "launch_specialist_score": 38.0,
                "launch_regime_fit_score": 41.0,
                "launch_specialist_confidence": 44.0,
                "close_vs_ma20": 0.12,
                "breakout_distance_20": 0.09,
            },
        ]
    )

    result = _sort_focus_board(board, ranking_by="鍏虫敞鍒嗘暟", board_size=2)

    assert result["symbol"].tolist() == ["000001", "000002"]
    assert result.loc[result["symbol"] == "000001", "launch_window_score"].iloc[0] > result.loc[
        result["symbol"] == "000002", "launch_window_score"
    ].iloc[0]


def test_sort_focus_board_prioritizes_precision_qualified_signals():
    board = pd.DataFrame(
        [
            {"symbol": "000001", "attention_score": 95.0, "probability_up": 62.0, "amount": 10, "precision_priority": 0},
            {"symbol": "000002", "attention_score": 89.0, "probability_up": 60.0, "amount": 9, "precision_priority": 2},
            {"symbol": "000003", "attention_score": 88.0, "probability_up": 67.0, "amount": 8, "precision_priority": 1},
        ]
    )

    result = _sort_focus_board(board, ranking_by="关注分数", board_size=3)

    assert result["symbol"].tolist() == ["000002", "000003", "000001"]


def test_sort_focus_board_applies_replay_calibration_before_ranking():
    board = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "attention_score": 80.0,
                "enhanced_attention_score": 82.0,
                "probability_up": 58.0,
                "quant_score": 76.0,
                "stage_label": "趋势主升加速",
                "precision_gate_label": "90%精度放行",
                "amount": 9.0,
            },
            {
                "symbol": "000002",
                "attention_score": 80.6,
                "enhanced_attention_score": 80.8,
                "probability_up": 92.0,
                "quant_score": 45.0,
                "stage_label": "高位分歧派发",
                "precision_gate_label": "未达90%精度门槛",
                "amount": 10.0,
            },
        ]
    )
    optimization_profile = {
        "review_days": 4,
        "review_stocks": 96,
        "weights": {
            "attention_score": 0.46,
            "probability_up": 0.30,
            "enhanced_attention_score": 0.16,
            "quant_score": 0.08,
        },
        "stage_edges": {"趋势主升加速": 0.14, "高位分歧派发": -0.12},
        "stage_supports": {"趋势主升加速": 24, "高位分歧派发": 12},
        "precision_segment_edges": {"precision_active": 0.10, "precision_unreached": -0.08},
        "precision_segment_supports": {"precision_active": 20, "precision_unreached": 18},
        "probability_bucket_edges": {"40-60": 0.05, "95-100": -0.10},
        "probability_bucket_supports": {"40-60": 28, "95-100": 16},
        "quant_bucket_edges": {"70-80": 0.08, "0-50": -0.09},
        "quant_bucket_supports": {"70-80": 30, "0-50": 14},
    }

    result = _sort_focus_board(board, ranking_by="关注分数", board_size=2, optimization_profile=optimization_profile)

    risky_row = result[result["symbol"] == "000002"].iloc[0]
    strong_row = result[result["symbol"] == "000001"].iloc[0]

    assert risky_row["raw_probability_up"] == 92.0
    assert risky_row["probability_up"] < risky_row["model_probability_up"]
    assert risky_row["attention_score"] < risky_row["model_attention_score"]
    assert strong_row["attention_score"] > strong_row["model_attention_score"]
    assert "ranking_score" in result.columns
    assert result["replay_calibration_active"].all()


def test_sort_focus_board_can_apply_market_replay_without_short_review():
    board = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "attention_score": 78.0,
                "enhanced_attention_score": 80.0,
                "probability_up": 61.0,
                "quant_score": 72.0,
                "amount": 9.0,
                "market_ret_5": 0.012,
                "market_ret_20": 0.034,
                "market_close_vs_ma20": 0.028,
                "market_volatility_10": 0.011,
                "market_range_position_20": 0.69,
                "ret_20": 0.14,
                "close_vs_ma20": 0.08,
                "breakout_distance_20": -0.01,
                "range_position_20": 0.71,
                "volume_ratio_5": 1.18,
                "upper_shadow_ratio": 0.10,
                "stretch_risk": 8.0,
                "risk_pressure": 148.0,
            },
            {
                "symbol": "000002",
                "attention_score": 81.0,
                "enhanced_attention_score": 83.0,
                "probability_up": 82.0,
                "quant_score": 46.0,
                "amount": 10.0,
                "market_ret_5": -0.018,
                "market_ret_20": -0.024,
                "market_close_vs_ma20": -0.03,
                "market_volatility_10": 0.026,
                "market_range_position_20": 0.38,
                "ret_20": -0.08,
                "close_vs_ma20": 0.06,
                "breakout_distance_20": -0.01,
                "range_position_20": 0.93,
                "volume_ratio_5": 1.01,
                "upper_shadow_ratio": 0.44,
                "stretch_risk": 25.0,
                "risk_pressure": 238.0,
            },
        ]
    )
    optimization_profile = {
        "review_days": 0,
        "review_stocks": 0,
        "weights": {
            "attention_score": 0.46,
            "probability_up": 0.30,
            "enhanced_attention_score": 0.16,
            "quant_score": 0.08,
        },
        "market_replay_days": 64,
        "market_replay_rows": 18000,
        "market_replay_symbols": 260,
        "market_state_edges": {"trend": 0.10, "defense": -0.08},
        "market_state_supports": {"trend": 4200, "defense": 3600},
        "market_stage_proxy_edges": {"trend_drive": 0.12, "distribution_risk": -0.09},
        "market_stage_proxy_supports": {"trend_drive": 2600, "distribution_risk": 2100},
    }

    result = _sort_focus_board(board, ranking_by="关注分数", board_size=2, optimization_profile=optimization_profile)

    assert result["replay_calibration_active"].all()
    assert result.loc[result["symbol"] == "000001", "attention_score"].iloc[0] > 78.0
    assert result.loc[result["symbol"] == "000002", "probability_up"].iloc[0] < 82.0


def test_local_precision_certification_prefers_backtest_gate(monkeypatch):
    dashboard._local_precision_certification.cache_clear()
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=8, freq="B"),
            "close": [10, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.5],
        }
    )

    monkeypatch.setattr(dashboard, "_prepare_symbol_base_analysis", lambda **kwargs: {"daily": daily})
    monkeypatch.setattr(
        dashboard,
        "train_probability_model",
        lambda *args, **kwargs: SimpleNamespace(
            latest_probability=0.923,
            precision_gate_threshold=0.8,
            precision_gate_precision=0.91,
            precision_gate_support=8,
            precision_gate_active=False,
            precision_gate_label="高精度观察",
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "run_daily_strategy_backtest",
        lambda *args, **kwargs: SimpleNamespace(
            achieved_precision=0.94,
            selected_threshold=0.78,
            trade_count=9,
            latest_signal_active=True,
            target_reached=True,
            status_label="达标可交易",
        ),
    )

    result = dashboard._local_precision_certification("600519", "贵州茅台", 5, 0.03, "2026-04-09")

    assert result["certification_ready"] is True
    assert result["precision_priority"] == 3
    assert result["precision_gate_label"] == "90%精度放行"
    assert result["probability_up"] == 92.3
    assert result["backtest_status_label"] == "达标可交易"


def test_enrich_candidate_uses_local_precision_certification(monkeypatch):
    candidate = {
        "symbol": "600519",
        "name": "贵州茅台",
        "attention_score": 78.0,
        "probability_up": 61.0,
        "quant_score": 66.0,
        "analysis_date": "2026-04-09",
    }
    base = {
        "stage": object(),
        "snapshot": {
            "close_vs_ma20": 0.04,
            "ret_20": 0.08,
            "volume_ratio_5": 1.3,
            "breakout_distance_20": 0.02,
        },
        "latest_features": {"close": 11.5},
        "quant_score": 68.0,
        "stage_score": 82.0,
        "stage_priority": "P1",
    }

    monkeypatch.setattr(dashboard, "fetch_stock_profile", lambda symbol: {"行业": "白酒"})
    monkeypatch.setattr(
        dashboard,
        "compute_sector_hot_score",
        lambda industry_name, industry_flow: {"sector_score": 72.0, "sector_label": "板块共振"},
    )
    monkeypatch.setattr(
        dashboard,
        "evaluate_main_fund_signal",
        lambda fund_df: {"fund_score": 74.0, "label": "主力流入", "confidence_score": 80.0},
    )
    monkeypatch.setattr(
        dashboard,
        "evaluate_news_sentiment",
        lambda news_df: {"sentiment_score": 69.0, "label": "消息偏多", "confidence_score": 76.0},
    )
    monkeypatch.setattr(dashboard, "fetch_stock_main_fund_flow", lambda symbol, limit=10: pd.DataFrame())
    monkeypatch.setattr(dashboard, "fetch_stock_news", lambda symbol, limit=8: pd.DataFrame())
    monkeypatch.setattr(dashboard, "_prepare_symbol_base_analysis", lambda **kwargs: base)
    monkeypatch.setattr(
        dashboard,
        "_local_precision_certification",
        lambda **kwargs: {
            "certification_ready": True,
            "probability_up": 92.0,
            "precision_priority": 3,
            "precision_gate_label": "90%精度放行",
            "precision_gate_threshold": 0.78,
            "precision_gate_precision": 94.0,
            "precision_gate_support": 9,
        },
    )
    monkeypatch.setattr(dashboard, "fetch_minute_history", lambda symbol: pd.DataFrame())
    monkeypatch.setattr(dashboard, "evaluate_intraday", lambda minute: {"label": "分时偏强"})
    monkeypatch.setattr(dashboard, "evaluate_intraday_structure_signal", lambda minute: {"signal": "strong"})
    monkeypatch.setattr(
        dashboard,
        "build_tomorrow_plan",
        lambda *args, **kwargs: SimpleNamespace(
            bias="偏多执行",
            setup_label="强势跟随",
            buy_point="分时第一次回踩均价线不破再执行",
            sell_point="均价线失守 5 分钟收不回先减仓",
            confidence=88.0,
        ),
    )

    enriched = dashboard._enrich_candidate(candidate, pd.DataFrame(), 5, 0.03)

    assert enriched["probability_up"] == 92.0
    assert enriched["precision_priority"] == 3
    assert enriched["precision_gate_label"] == "90%精度放行"
    assert enriched["precision_gate_support"] == 9
    assert enriched["attention_score"] != candidate["attention_score"]
    assert enriched["tomorrow_buy_point"] == "分时第一次回踩均价线不破再执行"


def test_analyze_single_base_applies_first_pass_live_context(monkeypatch):
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-20", "2026-04-21"]),
            "close": [10.0, 10.8],
            "high": [10.2, 11.0],
            "low": [9.8, 10.4],
            "amount": [2.0e8, 2.8e8],
            "turnover": [3.2, 4.1],
            "change_pct": [0.5, 8.0],
        }
    )
    base = {
        "daily": daily,
        "stage": SimpleNamespace(code="launch"),
        "snapshot": {
            "close_vs_ma20": 0.04,
            "ret_20": 0.08,
            "volume_ratio_5": 1.3,
            "breakout_distance_20": 0.02,
            "range_position_20": 0.72,
            "upper_shadow_ratio": 0.08,
        },
        "latest_features": {"ret_20": 0.08, "launch_readiness": 72.0, "market_resonance": 68.0},
        "quant_score": 68.0,
        "stage_score": 82.0,
        "stage_label": "主升初启",
        "stage_priority": "P1",
        "stage_summary": "ok",
        "board_label": "main",
        "price_limit_label": "10%",
        "latest_price": 10.8,
        "change_pct": 8.0,
        "amount": 2.8e8,
        "turnover": 4.1,
        "consecutive_up_days": 1,
        "analysis_date": "2026-04-21",
        "quant_primary_signal": "launch",
        "launch_score": 70.0,
        "launch_readiness_score": 72.0,
        "market_resonance_score": 68.0,
    }

    monkeypatch.setattr(dashboard, "_prepare_symbol_base_analysis", lambda **kwargs: base)
    monkeypatch.setattr(dashboard, "predict_latest_probability", lambda *args, **kwargs: 0.62)
    monkeypatch.setattr(
        dashboard,
        "_load_candidate_live_context",
        lambda symbol: {
            "minute": pd.DataFrame({"close": [10.6, 10.8]}),
            "fund_flow": pd.DataFrame({"net": [1.0]}),
            "news": pd.DataFrame({"title": ["positive catalyst"]}),
        },
    )
    monkeypatch.setattr(
        dashboard,
        "build_live_probability_upgrade",
        lambda *args, **kwargs: {
            "base_probability": 0.62,
            "upgraded_probability": 0.71,
            "summary": "live context applied",
            "live_context_score": 76.0,
            "intraday_execution_score": 78.0,
            "temporal_news_score": 66.0,
        },
    )
    monkeypatch.setattr(dashboard, "evaluate_intraday", lambda minute: {"label": "分时承接强", "score": 0.82})
    monkeypatch.setattr(
        dashboard,
        "evaluate_main_fund_signal",
        lambda fund_df: {"fund_score": 74.0, "label": "主力流入", "confidence_score": 80.0},
    )
    monkeypatch.setattr(
        dashboard,
        "evaluate_news_sentiment",
        lambda news_df: {"sentiment_score": 69.0, "label": "消息偏多", "confidence_score": 76.0},
    )
    monkeypatch.setattr(
        dashboard,
        "build_tomorrow_plan",
        lambda *args, **kwargs: SimpleNamespace(
            bias="偏多",
            setup_label="跟随",
            buy_point="回踩确认",
            sell_point="跌破均价",
            confidence=80.0,
        ),
    )
    monkeypatch.setattr(dashboard, "_build_launch_window_view", lambda *args, **kwargs: {})
    monkeypatch.setattr(dashboard, "_apply_candidate_strategy_prediction_profile", lambda row: row)

    result = dashboard._analyze_single_base("600001", "A", 3, 0.10, market_data_date="2026-04-21")

    assert result is not None
    assert result["full_context_prediction_active"] is True
    assert result["probability_up"] == 71.0
    assert result["probability_upgrade_note"] == "live context applied"
    assert result["intraday_label"] == "分时承接强"
    assert result["fund_label"] == "主力流入"
    assert result["news_label"] == "消息偏多"


def test_enrich_candidate_does_not_apply_live_upgrade_twice(monkeypatch):
    candidate = {
        "symbol": "600519",
        "name": "贵州茅台",
        "attention_score": 78.0,
        "probability_up": 71.0,
        "predicted_upside_pct": 12.5,
        "predicted_upside_low_pct": 7.5,
        "predicted_upside_high_pct": 18.0,
        "quant_score": 66.0,
        "analysis_date": "2026-04-09",
        "full_context_prediction_active": True,
        "probability_upgrade_note": "first pass live context",
    }
    base = {
        "stage": object(),
        "snapshot": {"close_vs_ma20": 0.04, "ret_20": 0.08, "volume_ratio_5": 1.3, "breakout_distance_20": 0.02},
        "latest_features": {"close": 11.5},
        "quant_score": 68.0,
        "stage_score": 82.0,
        "stage_priority": "P1",
        "launch_score": 70.0,
    }

    monkeypatch.setattr(dashboard, "fetch_stock_profile", lambda symbol: {"行业": "白酒", "琛屾笟": "白酒"})
    monkeypatch.setattr(dashboard, "compute_sector_hot_score", lambda industry_name, industry_flow: {"sector_score": 72.0, "sector_label": "板块共振"})
    monkeypatch.setattr(dashboard, "fetch_stock_main_fund_flow", lambda symbol, limit=10: pd.DataFrame({"net": [1.0]}))
    monkeypatch.setattr(dashboard, "fetch_stock_news", lambda symbol, limit=8: pd.DataFrame({"title": ["good"]}))
    monkeypatch.setattr(dashboard, "evaluate_main_fund_signal", lambda fund_df: {"fund_score": 74.0, "label": "主力流入", "confidence_score": 80.0})
    monkeypatch.setattr(dashboard, "evaluate_news_sentiment", lambda news_df: {"sentiment_score": 69.0, "label": "消息偏多", "confidence_score": 76.0})
    monkeypatch.setattr(dashboard, "_prepare_symbol_base_analysis", lambda **kwargs: base)
    monkeypatch.setattr(
        dashboard,
        "_local_precision_certification",
        lambda **kwargs: {
            "certification_ready": True,
            "probability_up": 92.0,
            "precision_priority": 3,
            "precision_gate_label": "90%精度放行",
            "precision_gate_threshold": 0.78,
            "precision_gate_precision": 94.0,
            "precision_gate_support": 9,
        },
    )
    monkeypatch.setattr(dashboard, "fetch_minute_history", lambda symbol: pd.DataFrame({"close": [10.0, 10.2]}))
    monkeypatch.setattr(dashboard, "evaluate_intraday", lambda minute: {"label": "分时偏强", "score": 0.8})
    monkeypatch.setattr(dashboard, "evaluate_intraday_structure_signal", lambda minute: {"signal": "strong"})
    monkeypatch.setattr(
        dashboard,
        "build_live_probability_upgrade",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("live upgrade should not run twice")),
    )
    monkeypatch.setattr(
        dashboard,
        "build_tomorrow_plan",
        lambda *args, **kwargs: SimpleNamespace(
            bias="偏多",
            setup_label="跟随",
            buy_point="回踩确认",
            sell_point="跌破均价",
            confidence=80.0,
        ),
    )

    enriched = dashboard._enrich_candidate(candidate, pd.DataFrame(), 5, 0.03)

    assert enriched["probability_up"] == 71.0
    assert enriched["predicted_upside_pct"] == 12.5
    assert enriched["probability_upgrade_note"] == "first pass live context"


def test_fallback_candidate_pool_prefers_dynamic_pool(monkeypatch):
    universe = pd.DataFrame(
        [
            {"symbol": "600519", "name": "A"},
            {"symbol": "000333", "name": "B"},
            {"symbol": "300750", "name": "C"},
            {"symbol": "688981", "name": "D"},
        ]
    )

    monkeypatch.setattr(dashboard, "_latest_market_close_date", lambda: "2026-04-08")
    monkeypatch.setattr(
        dashboard,
        "_build_dynamic_fallback_candidate_pool",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {"symbol": "300750", "name": "C"},
                {"symbol": "688981", "name": "D"},
            ]
        ),
    )

    candidates = _fallback_candidate_pool(universe)

    assert ("300750", "C") in candidates
    assert ("688981", "D") in candidates
    assert ("600519", "A") not in candidates


def test_build_dynamic_fallback_candidate_pool_uses_consecutive_up_rank(monkeypatch):
    universe = pd.DataFrame(
        [
            {"symbol": "301053", "name": "远信工业"},
            {"symbol": "600713", "name": "南京医药"},
            {"symbol": "000001", "name": "平安银行"},
        ]
    )
    expected = pd.DataFrame(
        [
            {
                "symbol": "301053",
                "name": "远信工业",
                "consecutive_up_days": 12,
                "analysis_date": "2026-04-08",
            },
            {
                "symbol": "600713",
                "name": "南京医药",
                "consecutive_up_days": 8,
                "analysis_date": "2026-04-08",
            },
        ]
    )
    monkeypatch.setattr(dashboard, "build_market_dynamic_fallback_pool_store", lambda universe, market_data_date: expected.copy())

    pool = dashboard._build_dynamic_fallback_candidate_pool(universe, "2026-04-08")

    assert pool["symbol"].tolist() == ["301053", "600713"]
    assert pool["consecutive_up_days"].tolist() == [12, 8]
    assert pool["analysis_date"].tolist() == ["2026-04-08", "2026-04-08"]


def test_filter_focus_candidates_keeps_only_three_day_up_streaks():
    board = pd.DataFrame(
        [
            {"symbol": "000001", "consecutive_up_days": 2, "attention_score": 88.0, "probability_up": 61.0, "amount": 10},
            {"symbol": "000002", "consecutive_up_days": 3, "attention_score": 93.0, "probability_up": 58.0, "amount": 12},
            {"symbol": "000003", "consecutive_up_days": 5, "attention_score": 90.0, "probability_up": 65.0, "amount": 11},
        ]
    )

    result = _filter_focus_candidates(board)

    assert result["symbol"].tolist() == ["000002", "000003"]


def test_build_display_board_falls_back_when_no_three_day_up_streak_exists():
    board = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "name": "A",
                "consecutive_up_days": 2,
                "attention_score": 88.0,
                "probability_up": 61.0,
                "amount": 10.0,
                "latest_price": 10.2,
                "change_pct": 1.2,
                "turnover": 2.1,
                "quant_score": 59.0,
                "stage_label": "range",
                "reason": "test",
            },
            {
                "symbol": "000002",
                "name": "B",
                "consecutive_up_days": 1,
                "attention_score": 86.0,
                "probability_up": 58.0,
                "amount": 9.0,
                "latest_price": 9.8,
                "change_pct": 0.8,
                "turnover": 1.8,
                "quant_score": 55.0,
                "stage_label": "range",
                "reason": "test",
            },
        ]
    )

    result = _build_display_board(board, board_size=2, ranking_by="关注分数", data_mode="fallback_watchlist", loading=True)

    assert not result.empty
    assert result.attrs["focus_filter_mode"] == "fallback"
    assert result.iloc[0]["symbol"] == "000001"


def test_build_display_board_supplements_when_three_day_up_streak_count_is_low():
    board = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "name": "A",
                "consecutive_up_days": 4,
                "attention_score": 91.0,
                "probability_up": 66.0,
                "amount": 12.0,
                "latest_price": 10.2,
                "change_pct": 1.2,
                "turnover": 2.1,
                "quant_score": 59.0,
                "stage_label": "trend",
                "reason": "test",
            },
            {
                "symbol": "000002",
                "name": "B",
                "consecutive_up_days": 3,
                "attention_score": 89.0,
                "probability_up": 64.0,
                "amount": 11.0,
                "latest_price": 10.1,
                "change_pct": 1.0,
                "turnover": 2.0,
                "quant_score": 58.0,
                "stage_label": "trend",
                "reason": "test",
            },
            {
                "symbol": "000003",
                "name": "C",
                "consecutive_up_days": 2,
                "attention_score": 88.0,
                "probability_up": 63.0,
                "amount": 10.0,
                "latest_price": 9.9,
                "change_pct": 0.9,
                "turnover": 1.9,
                "quant_score": 57.0,
                "stage_label": "range",
                "reason": "test",
            },
        ]
    )

    result = _build_display_board(board, board_size=10, ranking_by="鍏虫敞鍒嗘暟", data_mode="history", loading=True)

    assert result.attrs["focus_filter_mode"] == "supplemented"
    assert set(result["symbol"].tolist()) == {"000001", "000002", "000003"}


def test_build_display_board_preserves_data_freshness_attrs():
    board = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "name": "A",
                "analysis_date": "2026-04-09",
                "consecutive_up_days": 3,
                "attention_score": 88.0,
                "enhanced_attention_score": 91.0,
                "probability_up": 61.0,
                "amount": 10.0,
                "latest_price": 10.2,
                "change_pct": 1.2,
                "turnover": 2.1,
                "quant_score": 59.0,
                "stage_label": "range",
                "reason": "test",
            }
        ]
    )
    board.attrs["market_data_date"] = "2026-04-09"
    board.attrs["latest_market_data_date"] = "2026-04-10"
    board.attrs["cache_stale"] = True
    board.attrs["model_source_label"] = "快速代理模型"
    board.attrs["computed_at"] = "2026-04-10 09:05:00"

    result = _build_display_board(board, board_size=1, ranking_by="鍏虫敞鍒嗘暟", data_mode="history", loading=False)

    assert result.attrs["market_data_date"] == "2026-04-09"
    assert result.attrs["latest_market_data_date"] == "2026-04-10"
    assert result.attrs["cache_stale"] is True
    assert result.iloc[0]["analysis_date"] == "2026-04-09"
    assert result.iloc[0]["model_result_status"] == "非最新结果(2026-04-09)"
    assert result.iloc[0]["model_source_label"] == "快速代理模型"


def test_build_display_board_adds_action_columns():
    board = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "name": "A",
                "analysis_date": "2026-04-09",
                "consecutive_up_days": 3,
                "attention_score": 88.0,
                "enhanced_attention_score": 91.0,
                "probability_up": 68.0,
                "amount": 10.0,
                "latest_price": 10.2,
                "change_pct": 1.2,
                "turnover": 2.1,
                "quant_score": 65.0,
                "stage_label": "trend",
                "tomorrow_setup": "follow",
                "tomorrow_bias": "bullish",
                "tomorrow_plan_confidence": 72.0,
                "sector_label": "strong",
                "fund_label": "inflow",
                "news_label": "positive",
                "reason": "test",
            }
        ]
    )

    result = _build_display_board(board, board_size=1, ranking_by="关注分数", data_mode="history", loading=False)

    assert "action_label" in result.columns
    assert "action_badge" in result.columns
    assert "action_score" in result.columns
    assert "action_confidence" in result.columns
    assert "selection_score" in result.columns
    assert "execution_label" in result.columns
    assert "execution_score" in result.columns
    assert "reward_risk_label" in result.columns
    assert result.iloc[0]["action_label"] in {"买", "卖", "持", "观察"}
    assert result.iloc[0]["execution_label"] in {"可执行", "临门观察", "等待结构", "暂不执行"}


def test_build_display_board_normalizes_code_like_names(monkeypatch):
    board = pd.DataFrame(
        [
            {
                "symbol": "300308",
                "name": "300308",
                "analysis_date": "2026-04-09",
                "consecutive_up_days": 3,
                "attention_score": 88.0,
                "enhanced_attention_score": 91.0,
                "probability_up": 68.0,
                "amount": 10.0,
                "latest_price": 10.2,
                "change_pct": 1.2,
                "turnover": 2.1,
                "quant_score": 65.0,
                "stage_label": "trend",
                "reason": "test",
            }
        ]
    )
    monkeypatch.setattr(
        dashboard,
        "load_a_share_universe",
        lambda: pd.DataFrame([{"symbol": "300308", "name": "中际旭创"}]),
    )

    result = _build_display_board(board, board_size=1, ranking_by="关注分数", data_mode="history", loading=False)

    assert result.iloc[0]["name"] == "中际旭创"


def test_build_symbol_detail_uses_industry_flow_snapshot_instead_of_full_market_context(monkeypatch):
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2026-03-01", periods=30, freq="B"),
            "close": [10 + idx * 0.1 for idx in range(30)],
        }
    )
    features = pd.DataFrame(
        [
            {
                "close": 12.9,
            }
        ]
    )

    monkeypatch.setattr(dashboard, "fetch_daily_history", lambda **kwargs: daily)
    monkeypatch.setattr(dashboard, "fetch_minute_history", lambda symbol: pd.DataFrame())
    monkeypatch.setattr(dashboard, "build_daily_features", lambda daily_df: features)
    monkeypatch.setattr(
        dashboard,
        "classify_stage",
        lambda daily_df: SimpleNamespace(code="trend", label="趋势上行", intraday_expectation="分时承接为主"),
    )
    monkeypatch.setattr(dashboard, "_market_model_status", lambda *args, **kwargs: {"model_ready": False})
    monkeypatch.setattr(dashboard, "_load_market_model_or_none", lambda *args, **kwargs: None)
    monkeypatch.setattr(dashboard, "_load_market_proxy_or_none", lambda *args, **kwargs: None)
    monkeypatch.setattr(dashboard, "_resolve_model_source", lambda *args, **kwargs: ("local_fast_fallback", "本地快速回退"))
    monkeypatch.setattr(dashboard, "_latest_market_close_date", lambda: "2026-04-11")
    monkeypatch.setattr(
        dashboard,
        "train_probability_model",
        lambda *args, **kwargs: SimpleNamespace(
            latest_probability=0.64,
            metrics={},
            signal_label="偏多",
            backtest_summary="summary",
            strategy_score=72.0,
            agreement_score=68.0,
            quality_label="良好",
            risk_label="中低风险",
            signal_breakdown={},
            model_name="local",
            coefficients={},
            precision_gate_threshold=0.8,
            precision_gate_precision=0.91,
            precision_gate_support=9,
            precision_gate_active=False,
            precision_gate_label="高精度观察",
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "run_daily_strategy_backtest",
        lambda *args, **kwargs: SimpleNamespace(
            achieved_precision=0.91,
            selected_threshold=0.78,
            trade_count=9,
            latest_signal_active=True,
            target_reached=True,
            status_label="达标可交易",
        ),
    )
    monkeypatch.setattr(dashboard, "evaluate_quant_signal", lambda *args, **kwargs: SimpleNamespace(total_score=71.0))
    monkeypatch.setattr(dashboard, "evaluate_intraday", lambda minute: {"label": "分时偏强", "summary": "承接稳定"})
    monkeypatch.setattr(
        dashboard,
        "latest_snapshot",
        lambda *args, **kwargs: {
            "date": "2026-04-11",
            "close": 12.9,
            "close_vs_ma20": 0.04,
            "ret_20": 0.10,
            "volume_ratio_5": 1.2,
            "breakout_distance_20": 0.01,
        },
    )
    monkeypatch.setattr(dashboard, "explain_latest_model_state", lambda *args, **kwargs: {})
    monkeypatch.setattr(dashboard, "stage_numeric_score", lambda *args, **kwargs: 78.0)
    monkeypatch.setattr(dashboard, "fetch_stock_profile", lambda symbol: {"琛屼笟": "白酒"})
    monkeypatch.setattr(
        dashboard,
        "build_trading_rule_context",
        lambda **kwargs: SimpleNamespace(board_label="主板", price_limit_label="10%"),
    )
    monkeypatch.setattr(
        dashboard,
        "load_industry_flow_snapshot",
        lambda: pd.DataFrame({"sector_name": ["白酒"], "net_inflow": [1.0]}),
    )
    monkeypatch.setattr(
        dashboard,
        "compute_sector_hot_score",
        lambda *args, **kwargs: {"sector_score": 73.0, "sector_label": "板块共振", "sector_summary": "hot"},
    )
    monkeypatch.setattr(dashboard, "fetch_stock_main_fund_flow", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(
        dashboard,
        "evaluate_main_fund_signal",
        lambda df: {"fund_score": 74.0, "summary": "流入", "label": "主力流入", "confidence_score": 78.0},
    )
    monkeypatch.setattr(dashboard, "fetch_stock_news", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(
        dashboard,
        "evaluate_news_sentiment",
        lambda df: {"sentiment_score": 66.0, "summary": "偏多", "label": "消息偏多", "confidence_score": 76.0},
    )
    monkeypatch.setattr(
        dashboard,
        "evaluate_temporal_news_pulse",
        lambda df: SimpleNamespace(stronger_window="intraday", summary="pulse"),
    )
    monkeypatch.setattr(
        dashboard,
        "evaluate_intraday_structure_signal",
        lambda minute: SimpleNamespace(label="放量上攻", summary="summary"),
    )
    monkeypatch.setattr(
        dashboard,
        "build_strategy_workbench",
        lambda **kwargs: SimpleNamespace(strategy_score=70.0, summary="summary", style="跟随", entry_window="开盘后"),
    )
    monkeypatch.setattr(
        dashboard,
        "build_tomorrow_plan",
        lambda *args, **kwargs: SimpleNamespace(
            setup_label="强势跟随",
            bias="偏多执行",
            buy_point="分时回踩均价再跟随",
            sell_point="跌回 MA20 先减仓",
            confidence=82.0,
        ),
    )
    monkeypatch.setattr(dashboard, "_build_market_context", lambda: (_ for _ in ()).throw(AssertionError("should not call")))

    detail = dashboard._build_symbol_detail("600519", 5, 0.03)

    assert detail["symbol"] == "600519"
    assert detail["sector_signal"]["sector_score"] == 73.0


def test_detail_display_context_prefers_summary_row_values():
    class _QuantSignal:
        total_score = 66.0

    class _Model:
        latest_probability = 0.42
        predicted_upside_pct = 6.8
        predicted_upside_low_pct = 4.2
        predicted_upside_high_pct = 10.1

    class _TomorrowPlan:
        setup_label = "细节模型"
        bias = "中性"
        buy_point = "detail buy"
        sell_point = "detail sell"
        confidence = 55.0

    detail = {
        "model": _Model(),
        "quant_signal": _QuantSignal(),
        "tomorrow_plan": _TomorrowPlan(),
        "stage": type("Stage", (), {"label": "detail stage"})(),
        "analysis_date": "2026-04-09",
        "latest_market_data_date": "2026-04-09",
        "base_attention_score": 70.0,
        "enhanced_attention_score": 75.0,
        "model_source_label": "全市场模型",
        "snapshot": {"date": "2026-04-09"},
    }
    summary_row = {
        "analysis_date": "2026-04-09",
        "attention_score": 88.0,
        "enhanced_attention_score": 92.0,
        "probability_up": 67.0,
        "predicted_upside_pct": 14.5,
        "predicted_upside_low_pct": 9.1,
        "predicted_upside_high_pct": 19.8,
        "quant_score": 73.0,
        "stage_label": "榜单阶段",
        "tomorrow_setup": "榜单明日形势",
        "tomorrow_bias": "偏多",
        "tomorrow_buy_point": "board buy",
        "tomorrow_sell_point": "board sell",
        "tomorrow_plan_confidence": 81.0,
        "model_result_status": "最新模型结果",
        "model_source_label": "快速代理模型",
    }

    context = _detail_display_context(detail, summary_row)

    assert context["base_attention_score"] == 88.0
    assert context["enhanced_attention_score"] == 92.0
    assert context["probability_up"] == 67.0
    assert context["predicted_upside_pct"] == 14.5
    assert context["quant_score"] == 73.0
    assert context["stage_label"] == "榜单阶段"
    assert context["tomorrow_buy_point"] == "board buy"
    assert context["model_result_status"] == "最新模型结果"
    assert context["model_source_label"] == "快速代理模型"
    assert context["is_aligned_with_board"] is True


def test_detail_display_context_uses_detail_values_for_placeholder_summary():
    class _QuantSignal:
        total_score = 66.0

    class _Model:
        latest_probability = 0.42
        predicted_upside_pct = 6.8
        predicted_upside_low_pct = 4.2
        predicted_upside_high_pct = 10.1

    class _TomorrowPlan:
        setup_label = "细节模型"
        bias = "中性"
        buy_point = "detail buy"
        sell_point = "detail sell"
        confidence = 55.0

    detail = {
        "model": _Model(),
        "quant_signal": _QuantSignal(),
        "tomorrow_plan": _TomorrowPlan(),
        "stage": type("Stage", (), {"label": "detail stage"})(),
        "analysis_date": "2026-04-09",
        "latest_market_data_date": "2026-04-09",
        "base_attention_score": 70.0,
        "enhanced_attention_score": 75.0,
        "model_source_label": "全市场模型",
        "snapshot": {"date": "2026-04-09"},
    }
    summary_row = {
        "analysis_date": "2026-04-09",
        "attention_score": 0.0,
        "enhanced_attention_score": 0.0,
        "probability_up": 0.0,
        "predicted_upside_pct": 0.0,
        "predicted_upside_low_pct": 0.0,
        "predicted_upside_high_pct": 0.0,
        "quant_score": 0.0,
        "stage_label": "详情加载中",
        "tomorrow_setup": "待评估",
        "tomorrow_bias": "详情加载中",
        "tomorrow_buy_point": "loading",
        "tomorrow_sell_point": "loading",
        "tomorrow_plan_confidence": 0.0,
        "detail_placeholder": True,
    }

    context = _detail_display_context(detail, summary_row)

    assert context["base_attention_score"] == 70.0
    assert context["enhanced_attention_score"] == 75.0
    assert context["probability_up"] == 42.0
    assert context["predicted_upside_pct"] == 6.8
    assert context["quant_score"] == 66.0
    assert context["stage_label"] == "detail stage"
    assert context["is_aligned_with_board"] is False


def test_evaluate_symbol_action_covers_four_states():
    def make_detail(
        *,
        probability: float,
        base_attention: float,
        enhanced_attention: float,
        quant_score: float,
        sector_score: float,
        fund_score: float,
        news_score: float,
        close_vs_ma20: float,
        breakout_distance: float,
        intraday_label: str,
        intraday_summary: str,
        intraday_structure_label: str,
        intraday_structure_summary: str,
        tomorrow_setup: str,
        tomorrow_bias: str,
        tomorrow_confidence: float,
        latest_signal_active: bool,
        target_reached: bool,
        achieved_precision: float,
    ):
        detail = {
            "model": SimpleNamespace(
                latest_probability=probability / 100,
                precision_gate_threshold=0.78,
                precision_gate_precision=0.91,
                precision_gate_support=12,
                precision_gate_active=False,
                precision_gate_label="高精度观察",
            ),
            "backtest": SimpleNamespace(
                achieved_precision=achieved_precision,
                selected_threshold=0.78,
                trade_count=12,
                latest_signal_active=latest_signal_active,
                target_reached=target_reached,
                status_label="测试中",
            ),
            "sector_signal": {"sector_score": sector_score, "sector_summary": "板块摘要"},
            "fund_signal": {"fund_score": fund_score, "summary": "资金摘要"},
            "news_signal": {"sentiment_score": news_score, "summary": "消息摘要"},
            "snapshot": {
                "close_vs_ma20": close_vs_ma20,
                "breakout_distance_20": breakout_distance,
            },
            "intraday": {"label": intraday_label, "summary": intraday_summary},
            "intraday_structure_signal": SimpleNamespace(
                label=intraday_structure_label,
                summary=intraday_structure_summary,
            ),
            "tomorrow_plan": SimpleNamespace(confidence=tomorrow_confidence),
        }
        display_context = {
            "probability_up": probability,
            "base_attention_score": base_attention,
            "enhanced_attention_score": enhanced_attention,
            "quant_score": quant_score,
            "tomorrow_setup": tomorrow_setup,
            "tomorrow_bias": tomorrow_bias,
            "tomorrow_plan_confidence": tomorrow_confidence,
            "tomorrow_buy_point": "放量站回分时均价再跟随",
            "tomorrow_sell_point": "跌破分时均价与 MA20 先减仓",
        }
        return detail, display_context

    buy_detail, buy_context = make_detail(
        probability=74.0,
        base_attention=76.0,
        enhanced_attention=82.0,
        quant_score=75.0,
        sector_score=73.0,
        fund_score=78.0,
        news_score=69.0,
        close_vs_ma20=0.05,
        breakout_distance=0.01,
        intraday_label="分时偏强",
        intraday_summary="均价线上方承接稳定",
        intraday_structure_label="放量上攻",
        intraday_structure_summary="早盘回踩后快速修复",
        tomorrow_setup="强势跟随",
        tomorrow_bias="偏多执行",
        tomorrow_confidence=84.0,
        latest_signal_active=True,
        target_reached=True,
        achieved_precision=0.92,
    )
    hold_detail, hold_context = make_detail(
        probability=59.0,
        base_attention=64.0,
        enhanced_attention=66.0,
        quant_score=62.0,
        sector_score=61.0,
        fund_score=58.0,
        news_score=55.0,
        close_vs_ma20=0.02,
        breakout_distance=-0.01,
        intraday_label="分时震荡偏稳",
        intraday_summary="均价线附近反复拉锯",
        intraday_structure_label="窄幅整理",
        intraday_structure_summary="承接尚可但进攻性一般",
        tomorrow_setup="回踩承接",
        tomorrow_bias="偏多但不追高",
        tomorrow_confidence=67.0,
        latest_signal_active=False,
        target_reached=True,
        achieved_precision=0.84,
    )
    sell_detail, sell_context = make_detail(
        probability=34.0,
        base_attention=44.0,
        enhanced_attention=48.0,
        quant_score=40.0,
        sector_score=41.0,
        fund_score=38.0,
        news_score=40.0,
        close_vs_ma20=-0.05,
        breakout_distance=-0.06,
        intraday_label="分时走弱",
        intraday_summary="跌破均价后承接不足",
        intraday_structure_label="破位回落",
        intraday_structure_summary="尾盘弱化明显",
        tomorrow_setup="防守减仓",
        tomorrow_bias="偏空执行",
        tomorrow_confidence=41.0,
        latest_signal_active=False,
        target_reached=False,
        achieved_precision=0.48,
    )
    watch_detail, watch_context = make_detail(
        probability=54.0,
        base_attention=57.0,
        enhanced_attention=58.0,
        quant_score=55.0,
        sector_score=52.0,
        fund_score=50.0,
        news_score=51.0,
        close_vs_ma20=0.01,
        breakout_distance=-0.04,
        intraday_label="分时反复",
        intraday_summary="盘中有回流但持续性一般",
        intraday_structure_label="平台内震荡",
        intraday_structure_summary="等待选择方向",
        tomorrow_setup="观察确认",
        tomorrow_bias="中性",
        tomorrow_confidence=54.0,
        latest_signal_active=False,
        target_reached=False,
        achieved_precision=0.60,
    )

    assert dashboard._evaluate_symbol_action(buy_detail, buy_context)["action_label"] == "买"
    assert dashboard._evaluate_symbol_action(hold_detail, hold_context)["action_label"] == "持"
    assert dashboard._evaluate_symbol_action(sell_detail, sell_context)["action_label"] == "卖"
    assert dashboard._evaluate_symbol_action(watch_detail, watch_context)["action_label"] == "观察"


def test_evaluate_symbol_action_defaults_launch_window_confidence_weight_to_zero():
    detail = {
        "model": SimpleNamespace(
            latest_probability=0.72,
            precision_gate_threshold=0.78,
            precision_gate_precision=0.91,
            precision_gate_support=12,
            precision_gate_active=False,
            precision_gate_label="?????",
        ),
        "backtest": SimpleNamespace(
            achieved_precision=0.86,
            selected_threshold=0.78,
            trade_count=12,
            latest_signal_active=False,
            target_reached=True,
            status_label="???",
        ),
        "sector_signal": {"sector_score": 72.0, "sector_summary": "????"},
        "fund_signal": {"fund_score": 74.0, "summary": "????"},
        "news_signal": {"sentiment_score": 68.0, "summary": "????"},
        "snapshot": {
            "close_vs_ma20": 0.04,
            "breakout_distance_20": 0.01,
        },
        "intraday": {"label": "????", "summary": "?????????"},
        "intraday_structure_signal": SimpleNamespace(
            label="????",
            summary="?????????",
        ),
        "tomorrow_plan": SimpleNamespace(confidence=82.0),
    }
    display_context = {
        "probability_up": 72.0,
        "predicted_upside_pct": 8.0,
        "base_attention_score": 74.0,
        "enhanced_attention_score": 80.0,
        "quant_score": 71.0,
        "tomorrow_setup": "????",
        "tomorrow_bias": "????",
        "tomorrow_plan_confidence": 82.0,
        "candidate_strategy": "??1",
    }

    baseline = dashboard._evaluate_symbol_action(detail, display_context)
    explicit_zero = dashboard._evaluate_symbol_action(
        detail,
        {**display_context, "launch_window_confidence_weight": 0.0},
    )

    assert baseline["selection_score"] == explicit_zero["selection_score"]


def test_evaluate_symbol_action_allows_positive_launch_window_confidence_override():
    detail = {
        "model": SimpleNamespace(
            latest_probability=0.72,
            precision_gate_threshold=0.78,
            precision_gate_precision=0.91,
            precision_gate_support=12,
            precision_gate_active=False,
            precision_gate_label="?????",
        ),
        "backtest": SimpleNamespace(
            achieved_precision=0.86,
            selected_threshold=0.78,
            trade_count=12,
            latest_signal_active=False,
            target_reached=True,
            status_label="???",
        ),
        "sector_signal": {"sector_score": 72.0, "sector_summary": "????"},
        "fund_signal": {"fund_score": 74.0, "summary": "????"},
        "news_signal": {"sentiment_score": 68.0, "summary": "????"},
        "snapshot": {
            "close_vs_ma20": 0.04,
            "breakout_distance_20": 0.01,
        },
        "intraday": {"label": "????", "summary": "?????????"},
        "intraday_structure_signal": SimpleNamespace(
            label="????",
            summary="?????????",
        ),
        "tomorrow_plan": SimpleNamespace(confidence=82.0),
    }
    display_context = {
        "probability_up": 72.0,
        "predicted_upside_pct": 8.0,
        "base_attention_score": 74.0,
        "enhanced_attention_score": 80.0,
        "quant_score": 71.0,
        "tomorrow_setup": "????",
        "tomorrow_bias": "????",
        "tomorrow_plan_confidence": 82.0,
        "candidate_strategy": "??1",
    }

    baseline = dashboard._evaluate_symbol_action(detail, display_context)
    overridden = dashboard._evaluate_symbol_action(
        detail,
        {**display_context, "launch_window_confidence_weight": 0.04},
    )

    assert overridden["selection_score"] >= baseline["selection_score"]
    assert overridden["selection_score"] - baseline["selection_score"] > 0.0


def test_evaluate_symbol_action_defaults_action_score_to_selection_score():
    detail = {
        "analysis": {
            "achieved_precision": 0.68,
            "trade_count": 12,
            "latest_signal_active": False,
            "target_reached": True,
            "status_label": "ok",
        },
        "sector_signal": {"sector_score": 72.0, "sector_summary": "strong"},
        "fund_signal": {"fund_score": 74.0, "summary": "inflow"},
        "news_signal": {"sentiment_score": 68.0, "summary": "positive"},
        "snapshot": {"close_vs_ma20": 0.04, "breakout_distance_20": 0.01},
        "intraday": {"label": "strong", "summary": "healthy"},
        "intraday_structure_signal": SimpleNamespace(label="strong", summary="healthy"),
        "tomorrow_plan": SimpleNamespace(confidence=82.0),
    }
    display_context = {
        "probability_up": 72.0,
        "predicted_upside_pct": 8.0,
        "base_attention_score": 74.0,
        "enhanced_attention_score": 80.0,
        "quant_score": 71.0,
        "tomorrow_setup": "follow",
        "tomorrow_bias": "bullish",
        "tomorrow_plan_confidence": 82.0,
        "candidate_strategy": "策略1",
    }

    result = dashboard._evaluate_symbol_action(detail, display_context)

    assert result["action_execution_weight"] == 0.0
    assert result["action_score"] == result["selection_score"]


def test_evaluate_symbol_action_allows_execution_weight_override():
    detail = {
        "analysis": {
            "achieved_precision": 0.68,
            "trade_count": 12,
            "latest_signal_active": False,
            "target_reached": True,
            "status_label": "ok",
        },
        "sector_signal": {"sector_score": 72.0, "sector_summary": "strong"},
        "fund_signal": {"fund_score": 74.0, "summary": "inflow"},
        "news_signal": {"sentiment_score": 68.0, "summary": "positive"},
        "snapshot": {"close_vs_ma20": 0.04, "breakout_distance_20": 0.01},
        "intraday": {"label": "strong", "summary": "healthy"},
        "intraday_structure_signal": SimpleNamespace(label="strong", summary="healthy"),
        "tomorrow_plan": SimpleNamespace(confidence=82.0),
    }
    display_context = {
        "probability_up": 72.0,
        "predicted_upside_pct": 8.0,
        "base_attention_score": 74.0,
        "enhanced_attention_score": 80.0,
        "quant_score": 71.0,
        "tomorrow_setup": "follow",
        "tomorrow_bias": "bullish",
        "tomorrow_plan_confidence": 82.0,
        "candidate_strategy": "策略1",
        "action_execution_weight": 0.38,
    }

    result = dashboard._evaluate_symbol_action(detail, display_context)

    assert result["action_execution_weight"] == 0.38
    assert result["action_score"] != result["selection_score"]


def test_evaluate_board_action_defaults_launch_window_confidence_weight_to_zero():
    row = {
        "probability_up": 72.0,
        "predicted_upside_pct": 8.0,
        "attention_score": 74.0,
        "enhanced_attention_score": 80.0,
        "quant_score": 71.0,
        "tomorrow_setup": "follow",
        "tomorrow_bias": "bullish",
        "tomorrow_plan_confidence": 82.0,
        "candidate_strategy": "策略1",
        "sector_score": 72.0,
        "fund_score": 74.0,
        "news_score": 68.0,
        "stage_label": "主升启动",
        "intraday_label": "强势",
        "reason": "量价共振",
        "close_vs_ma20": 0.04,
        "breakout_distance_20": 0.01,
    }

    baseline = dashboard._evaluate_board_action(row)
    explicit_zero = dashboard._evaluate_board_action({**row, "launch_window_confidence_weight": 0.0})

    assert baseline["selection_score"] == explicit_zero["selection_score"]


def test_evaluate_board_action_defaults_action_score_to_selection_score():
    row = {
        "probability_up": 72.0,
        "predicted_upside_pct": 8.0,
        "attention_score": 74.0,
        "enhanced_attention_score": 80.0,
        "quant_score": 71.0,
        "tomorrow_setup": "follow",
        "tomorrow_bias": "bullish",
        "tomorrow_plan_confidence": 82.0,
        "candidate_strategy": "策略1",
        "sector_score": 72.0,
        "fund_score": 74.0,
        "news_score": 68.0,
        "stage_label": "主升启动",
        "intraday_label": "强势",
        "reason": "量价共振",
        "close_vs_ma20": 0.04,
        "breakout_distance_20": 0.01,
    }

    result = dashboard._evaluate_board_action(row)

    assert result["action_execution_weight"] == 0.0
    assert result["action_score"] == result["selection_score"]


def test_extract_market_data_date_uses_latest_analysis_date():
    board = pd.DataFrame(
        [
            {"symbol": "000001", "analysis_date": "2026-04-07"},
            {"symbol": "000002", "analysis_date": "2026-04-08"},
        ]
    )

    result = _extract_market_data_date(board)

    assert result == "2026-04-08"


def test_latest_market_close_date_uses_latest_available_reference_symbol(monkeypatch):
    history_map = {
        "600519": pd.DataFrame({"date": pd.to_datetime(["2026-04-18", "2026-04-20"])}),
        "000001": pd.DataFrame({"date": pd.to_datetime(["2026-04-18", "2026-04-21"])}),
        "601398": pd.DataFrame({"date": pd.to_datetime(["2026-04-17", "2026-04-21"])}),
    }

    def fake_fetch_daily_history(symbol, start_date=None, adjust="hfq", timeout=15.0):
        return history_map.get(str(symbol), pd.DataFrame())

    monkeypatch.setattr(dashboard, "fetch_daily_history", fake_fetch_daily_history)

    assert dashboard._latest_market_close_date() == "2026-04-21"


def test_should_auto_force_market_refresh_only_once_per_latest_date():
    assert (
        dashboard._should_auto_force_market_refresh(
            cache_stale=True,
            has_non_latest_results=False,
            custom_watchlist=tuple(),
            cached_market_data_date="2026-04-20",
            latest_market_data_date="2026-04-21",
            last_forced_date="2026-04-20",
        )
        is True
    )
    assert (
        dashboard._should_auto_force_market_refresh(
            cache_stale=True,
            has_non_latest_results=False,
            custom_watchlist=tuple(),
            cached_market_data_date="2026-04-20",
            latest_market_data_date="2026-04-21",
            last_forced_date="2026-04-21",
        )
        is False
    )
    assert (
        dashboard._should_auto_force_market_refresh(
            cache_stale=False,
            has_non_latest_results=False,
            custom_watchlist=tuple(),
            cached_market_data_date="2026-04-21",
            latest_market_data_date="2026-04-21",
            last_forced_date="",
        )
        is False
    )


def test_should_auto_force_market_refresh_when_board_has_non_latest_results():
    assert (
        dashboard._should_auto_force_market_refresh(
            cache_stale=False,
            has_non_latest_results=True,
            custom_watchlist=tuple(),
            cached_market_data_date="2026-04-21",
            latest_market_data_date="2026-04-21",
            last_forced_date="2026-04-20",
        )
        is True
    )


def test_board_has_non_latest_model_results_detects_stale_rows():
    board = pd.DataFrame(
        [
            {"symbol": "000001", "analysis_date": "2026-04-21", "model_result_status": "最新结果"},
            {"symbol": "000002", "analysis_date": "2026-04-20", "model_result_status": "非最新结果(2026-04-20)"},
        ]
    )

    assert dashboard._board_has_non_latest_model_results(board, "2026-04-21") is True


def test_board_has_non_latest_model_results_accepts_latest_rows():
    board = pd.DataFrame(
        [
            {"symbol": "000001", "analysis_date": "2026-04-21", "model_result_status": "最新结果"},
            {"symbol": "000002", "analysis_date": "2026-04-21", "model_result_status": "最新结果"},
        ]
    )

    assert dashboard._board_has_non_latest_model_results(board, "2026-04-21") is False


def test_resolve_market_refresh_request_detects_new_closed_day_while_page_stays_open():
    board = pd.DataFrame([{"symbol": "000001", "analysis_date": "2026-04-21", "model_result_status": "最新结果"}])
    board.attrs["market_data_date"] = "2026-04-21"
    board.attrs["latest_market_data_date"] = "2026-04-21"
    board.attrs["cache_stale"] = False

    state = dashboard._resolve_market_refresh_request(
        board,
        custom_watchlist=tuple(),
        latest_market_data_date="2026-04-22",
        last_forced_date="2026-04-21",
    )

    assert state["cache_stale"] is True
    assert state["refresh_reason"] == "stale_results"
    assert state["should_force"] is True


def test_ensure_market_refresh_task_for_board_starts_async_refresh_for_new_closed_day(monkeypatch):
    board = pd.DataFrame([{"symbol": "000001", "analysis_date": "2026-04-21", "model_result_status": "最新结果"}])
    board.attrs["market_data_date"] = "2026-04-21"
    board.attrs["latest_market_data_date"] = "2026-04-21"
    board.attrs["cache_stale"] = False

    fake_st = SimpleNamespace(session_state={})
    calls: dict[str, tuple] = {}

    monkeypatch.setattr(dashboard, "st", fake_st)
    monkeypatch.setattr(dashboard, "_market_rank_refresh_async_key", lambda horizon_days, positive_return: "refresh-task")
    monkeypatch.setattr(dashboard, "_ensure_async_task", lambda *args: calls.setdefault("args", args))
    monkeypatch.setattr(dashboard, "_get_async_task_progress", lambda task_key: {})

    state = dashboard._ensure_market_refresh_task_for_board(
        board,
        custom_watchlist=tuple(),
        horizon_days=3,
        positive_return=0.10,
        latest_market_data_date="2026-04-22",
    )

    assert state["started_now"] is True
    assert state["task_key"] == "refresh-task"
    assert fake_st.session_state["forced_market_refresh_date"] == "2026-04-22"
    assert fake_st.session_state["forced_market_refresh_reason"] == "stale_results"
    assert calls["args"][0] == "refresh-task"


def test_resolve_market_refresh_request_marks_quick_board_pending():
    board = pd.DataFrame([{"symbol": "000001", "analysis_date": "2026-04-22", "model_result_status": "最新收盘快榜"}])
    board.attrs["market_data_date"] = "2026-04-22"
    board.attrs["latest_market_data_date"] = "2026-04-22"
    board.attrs["cache_stale"] = False
    board.attrs["quick_board_pending"] = True

    state = dashboard._resolve_market_refresh_request(
        board,
        custom_watchlist=tuple(),
        latest_market_data_date="2026-04-22",
        last_forced_date="2026-04-21",
    )

    assert state["quick_board_pending"] is True
    assert state["refresh_reason"] == "quick_board_pending"
    assert state["should_force"] is True


def test_read_market_rankings_cache_invalidates_when_new_close_date_arrives(monkeypatch, tmp_path):
    cache_path = Path(tmp_path) / "market_rankings_v8_h5_r300.pkl"
    board = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "analysis_date": "2026-04-07",
                "consecutive_up_days": 3,
                "attention_score": 88.0,
                "probability_up": 61.0,
                "amount": 10.0,
            }
        ]
    )
    payload = {
        "meta": {
            "cache_version": dashboard.MARKET_RANKING_CACHE_VERSION,
            "cache_date": "2026-04-07",
            "market_data_date": "2026-04-07",
            "horizon_days": 5,
            "positive_return": 0.03,
            "data_mode": "history",
            "row_count": 1,
        },
        "data": board,
    }
    with cache_path.open("wb") as handle:
        pickle.dump(payload, handle)

    monkeypatch.setattr(dashboard, "_ranking_cache_path", lambda *args, **kwargs: cache_path)
    monkeypatch.setattr(dashboard, "_latest_market_close_date", lambda *args, **kwargs: "2026-04-08")

    cached_df, meta = _read_market_rankings_cache(5, 0.03)

    assert cached_df is None
    assert meta == {}


def test_read_market_rankings_cache_can_return_stale_cache_when_requested(monkeypatch, tmp_path):
    cache_path = Path(tmp_path) / "market_rankings_v8_h5_r300.pkl"
    board = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "analysis_date": "2026-04-08",
                "consecutive_up_days": 3,
                "attention_score": 88.0,
                "probability_up": 61.0,
                "amount": 10.0,
            }
        ]
    )
    payload = {
        "meta": {
            "cache_version": dashboard.MARKET_RANKING_CACHE_VERSION,
            "cache_date": "2026-04-08",
            "market_data_date": "2026-04-08",
            "horizon_days": 5,
            "positive_return": 0.03,
            "data_mode": "history",
            "row_count": 1,
        },
        "data": board,
    }
    with cache_path.open("wb") as handle:
        pickle.dump(payload, handle)

    monkeypatch.setattr(dashboard, "_ranking_cache_path", lambda *args, **kwargs: cache_path)
    monkeypatch.setattr(dashboard, "_latest_market_close_date", lambda *args, **kwargs: "2026-04-09")

    cached_df, meta = _read_market_rankings_cache(5, 0.03, allow_stale=True)

    assert cached_df is not None
    assert meta["cache_stale"] is True
    assert meta["market_data_date"] == "2026-04-08"
    assert meta["latest_market_data_date"] == "2026-04-09"


def test_async_task_progress_helpers_round_trip():
    task_key = "market-refresh::test"
    dashboard._clear_async_task_progress(task_key)

    dashboard._set_async_task_progress(task_key, "扫描候选股票", 12, 80, "已完成 12/80")
    progress = dashboard._get_async_task_progress(task_key)

    assert progress == {
        "phase": "扫描候选股票",
        "completed": 12,
        "total": 80,
        "message": "已完成 12/80",
    }

    dashboard._clear_async_task_progress(task_key)
    assert dashboard._get_async_task_progress(task_key) == {}


def test_refresh_market_rankings_cache_task_publishes_completion_progress(monkeypatch):
    task_key = "market-refresh::5::0.0300"
    board = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "analysis_date": "2026-04-09",
                "attention_score": 88.0,
                "probability_up": 61.0,
                "amount": 10.0,
            }
        ]
    )

    monkeypatch.setattr(
        dashboard,
        "_build_ranked_market_snapshot",
        lambda horizon_days, positive_return, progress_callback=None: (board, "history"),
    )
    written: dict[str, object] = {}
    monkeypatch.setattr(
        dashboard,
        "_write_market_rankings_cache",
        lambda df, horizon_days, positive_return, data_mode: written.update({"rows": len(df), "data_mode": data_mode}),
    )

    dashboard._clear_async_task_progress(task_key)
    result = dashboard._refresh_market_rankings_cache_task(task_key, 5, 0.03)
    progress = dashboard._get_async_task_progress(task_key)

    assert result == {"data_mode": "history", "market_data_date": "2026-04-09", "row_count": 1}
    assert written == {"rows": 1, "data_mode": "history"}
    assert progress["phase"] == "写入缓存"
    assert progress["completed"] == 1
    assert progress["total"] == 1


def test_build_ranked_market_snapshot_uses_focus_candidate_pool(monkeypatch):
    universe = pd.DataFrame(
        [
            {"symbol": "000001", "name": "A"},
            {"symbol": "000002", "name": "B"},
            {"symbol": "000003", "name": "C"},
        ]
    )
    analyzed: list[str] = []

    monkeypatch.setattr(dashboard, "load_a_share_universe", lambda: universe)
    monkeypatch.setattr(dashboard, "_load_market_model_or_none", lambda *args, **kwargs: None)
    monkeypatch.setattr(dashboard, "_load_market_proxy_or_none", lambda *args, **kwargs: None)
    monkeypatch.setattr(dashboard, "_latest_market_close_date", lambda *args, **kwargs: "2026-04-09")
    monkeypatch.setattr(dashboard, "fetch_market_spot", lambda: pd.DataFrame())
    monkeypatch.setattr(dashboard, "build_market_daily_feature_store", lambda universe, market_data_date, progress_callback=None: pd.DataFrame())
    monkeypatch.setattr(dashboard, "build_market_candidate_pool_store", lambda universe, market_data_date, feature_store=None, progress_callback=None: pd.DataFrame())
    monkeypatch.setattr(dashboard, "_read_candidate_analysis_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(dashboard, "_write_candidate_analysis_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        dashboard,
        "build_market_dynamic_fallback_pool_store",
        lambda universe, market_data_date, feature_store=None, progress_callback=None: pd.DataFrame(
            [{"symbol": "000002", "name": "B"}, {"symbol": "000003", "name": "C"}]
        ),
    )
    monkeypatch.setattr(dashboard, "_fallback_candidate_pool", lambda _: [("000002", "B"), ("000003", "C")])

    def fake_analyze(symbol, name, horizon_days, positive_return, start_date, market_model, market_proxy_model, market_data_date):
        analyzed.append(symbol)
        return {
            "symbol": symbol,
            "name": name,
            "attention_score": 80.0 if symbol == "000002" else 75.0,
            "probability_up": 60.0,
            "amount": 10.0,
            "consecutive_up_days": 3,
            "analysis_date": market_data_date,
            "latest_price": 10.0,
            "change_pct": 1.0,
            "turnover": 2.0,
            "quant_score": 55.0,
            "stage_label": "test",
            "stage_priority": "P1",
            "stage_summary": "summary",
            "board_label": "主板",
            "price_limit_label": "10%",
            "tomorrow_bias": "偏多",
            "tomorrow_setup": "观察",
            "tomorrow_buy_point": "buy",
            "tomorrow_sell_point": "sell",
            "tomorrow_plan_confidence": 60.0,
            "reason": "test",
        }

    monkeypatch.setattr(dashboard, "_analyze_single_base", fake_analyze)

    ranked, mode = dashboard._build_ranked_market_snapshot(5, 0.03)

    assert analyzed == ["000002", "000003"]
    assert ranked["symbol"].tolist() == ["000002", "000003"]
    assert mode == "dynamic_fallback_pool"


def test_build_market_context_reads_same_day_disk_cache(monkeypatch):
    cached_context = {
        "industry_flow": pd.DataFrame([{"sector_name": "白酒"}]),
        "concept_flow": pd.DataFrame([{"sector_name": "AI"}]),
        "macro_calendar": pd.DataFrame([{"日期": "2026-04-09"}]),
    }

    monkeypatch.setattr(dashboard, "_latest_market_close_date", lambda *args, **kwargs: "2026-04-09")
    monkeypatch.setattr(dashboard, "_read_market_context_cache", lambda market_data_date: cached_context)

    context = dashboard._build_market_context()

    assert context is cached_context


def test_load_history_first_focus_board_falls_back_to_snapshot(monkeypatch):
    snapshot_board = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "name": "A",
                "analysis_date": "2026-04-09",
                "consecutive_up_days": 3,
                "attention_score": 88.0,
                "probability_up": 61.0,
                "quant_score": 55.0,
                "amount": 10.0,
            }
        ]
    )

    monkeypatch.setattr(dashboard, "_read_market_rankings_cache", lambda *args, **kwargs: (None, {}))
    monkeypatch.setattr(dashboard, "load_latest_close_quick_board", lambda **kwargs: (pd.DataFrame(), {}))
    monkeypatch.setattr(
        dashboard,
        "load_latest_snapshot_board",
        lambda **kwargs: (
            snapshot_board,
            {
                "board_date": "2026-04-09",
                "latest_market_data_date": "2026-04-09",
                "horizon_days": 5,
                "positive_return": 0.03,
                "model_source_label": "历史关注榜快照",
                "captured_at": "2026-04-10 09:30:00",
            },
        ),
    )
    monkeypatch.setattr(dashboard, "_latest_market_close_date", lambda *args, **kwargs: "2026-04-10")

    rendered = dashboard._load_history_first_focus_board(
        board_size=50,
        custom_watchlist=tuple(),
        horizon_days=5,
        positive_return=0.03,
        ranking_by="关注分数",
    )

    assert rendered["symbol"].tolist() == ["000001"]
    assert rendered.attrs["market_data_date"] == "2026-04-09"
    assert rendered.attrs["latest_market_data_date"] == "2026-04-10"
    assert rendered.attrs["cache_stale"] is True


def test_load_history_first_focus_board_prefers_latest_close_quick_board(monkeypatch):
    ranked_cache = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "name": "Old",
                "analysis_date": "2026-04-28",
                "consecutive_up_days": 3,
                "attention_score": 70.0,
                "probability_up": 58.0,
                "quant_score": 55.0,
                "amount": 8.0,
            }
        ]
    )
    quick_board = pd.DataFrame(
        [
            {
                "symbol": "000002",
                "name": "New",
                "analysis_date": "2026-04-29",
                "consecutive_up_days": 4,
                "attention_score": 82.0,
                "enhanced_attention_score": 84.0,
                "probability_up": 68.0,
                "raw_probability_up": 68.0,
                "quant_score": 66.0,
                "amount": 12.0,
                "candidate_strategy": "dynamic_fallback",
                "candidate_reason": "最新收盘快榜非正式策略榜兜底",
                "stage_label": "最新收盘快榜",
                "stage_priority": "快榜",
                "reason": "最新收盘快榜非正式策略榜兜底",
                "latest_price": 12.4,
                "change_pct": 3.2,
                "turnover": 4.5,
                "predicted_upside_pct": 6.2,
                "predicted_upside_low_pct": 4.0,
                "predicted_upside_high_pct": 7.4,
                "precision_gate_label": "后台完整版回测中",
                "tomorrow_setup": "最新收盘快榜先行",
                "tomorrow_bias": "偏强观察",
                "tomorrow_buy_point": "等待低吸",
                "tomorrow_sell_point": "跌破 MA20 防守",
                "tomorrow_plan_confidence": 66.0,
                "sector_label": "板块两日强度 3.4",
                "fund_label": "主力资金待完整版补齐",
                "news_label": "消息面待完整版补齐",
                "launch_score": 72.0,
                "launch_readiness_score": 76.0,
                "market_resonance_score": 70.0,
                "snapshot": {
                    "date": "2026-04-29",
                    "change_pct": 3.2,
                    "close_vs_ma20": 0.05,
                    "ret_20": 0.12,
                    "volume_ratio_5": 1.4,
                    "breakout_distance_20": 0.01,
                    "range_position_20": 0.7,
                    "upper_shadow_ratio": 0.12,
                },
            }
        ]
    )
    quick_board.attrs["data_mode"] = "latest_close_quick_board"
    quick_board.attrs["market_data_date"] = "2026-04-29"
    quick_board.attrs["latest_market_data_date"] = "2026-04-29"
    quick_board.attrs["cache_stale"] = False
    quick_board.attrs["quick_board_pending"] = True
    quick_board.attrs["computed_at"] = "2026-04-29 20:30:00"
    quick_board.attrs["horizon_days"] = 3
    quick_board.attrs["positive_return"] = 0.10
    quick_board.attrs["model_source"] = "latest_close_quick_board"
    quick_board.attrs["model_source_label"] = "最新收盘快榜（完整版特征与回测正在后台补齐）"
    quick_board.attrs["model_schema_version"] = dashboard.MODEL_SCHEMA_VERSION

    monkeypatch.setattr(
        dashboard,
        "_read_market_rankings_cache",
        lambda *args, **kwargs: (
            ranked_cache,
            {
                "data_mode": "history",
                "market_data_date": "2026-04-28",
                "latest_market_data_date": "2026-04-29",
                "cache_stale": True,
                "computed_at": "2026-04-28 20:30:00",
                "horizon_days": 3,
                "positive_return": 0.10,
                "model_source": "model",
                "model_source_label": "完整版模型",
                "model_schema_version": dashboard.MODEL_SCHEMA_VERSION,
            },
        ),
    )
    monkeypatch.setattr(dashboard, "_latest_market_close_date", lambda *args, **kwargs: "2026-04-29")
    monkeypatch.setattr(
        dashboard,
        "load_latest_close_quick_board",
        lambda **kwargs: (
            quick_board.copy(),
            {
                "board_date": "2026-04-29",
                "latest_market_data_date": "2026-04-29",
                "captured_at": "2026-04-29 20:30:00",
                "horizon_days": 3,
                "positive_return": 0.10,
                "model_source_label": "最新收盘快榜（完整版特征与回测正在后台补齐）",
                "quick_board_pending": True,
            },
        ),
    )

    rendered = dashboard._load_history_first_focus_board(
        board_size=50,
        custom_watchlist=tuple(),
        horizon_days=3,
        positive_return=0.10,
        ranking_by="关注分数",
    )

    assert rendered["symbol"].tolist() == ["000002"]
    assert rendered.attrs["market_data_date"] == "2026-04-29"
    assert rendered.attrs["latest_market_data_date"] == "2026-04-29"
    assert rendered.attrs["quick_board_pending"] is True
def test_load_latest_close_quick_board_falls_back_to_snapshot_when_current_history_store_missing(monkeypatch):
    latest_snapshot = pd.DataFrame(
        [
            {
                "symbol": "600001",
                "close": 12.3,
                "pct_chg": 3.1,
                "amount": 3.4e8,
                "turnover_rate": 4.5,
                "industry": "科技",
            },
            {
                "symbol": "600002",
                "close": 10.5,
                "pct_chg": 1.8,
                "amount": 2.2e8,
                "turnover_rate": 3.1,
                "industry": "科技",
            },
        ]
    )
    previous_snapshot = pd.DataFrame(
        [
            {"symbol": "600001", "close": 11.9},
            {"symbol": "600002", "close": 10.1},
        ]
    )
    invalid_cached_history = pd.DataFrame(
        [
            {
                "symbol": "600001",
                "trade_date": pd.Timestamp("2024-12-09"),
                "date": pd.Timestamp("2024-12-09"),
                "close": 10.0,
            }
        ]
    )

    if hasattr(dashboard.load_latest_close_quick_board, "clear"):
        dashboard.load_latest_close_quick_board.clear()
    monkeypatch.setattr(dashboard, "_latest_market_close_date", lambda: "2026-05-06")
    monkeypatch.setattr(
        dashboard,
        "load_a_share_universe",
        lambda: pd.DataFrame(
            [
                {"symbol": "600001", "name": "A"},
                {"symbol": "600002", "name": "B"},
            ]
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "_build_strategy_snapshot_context",
        lambda market_data_date: (latest_snapshot.copy(), previous_snapshot.copy()),
    )
    monkeypatch.setattr(
        dashboard,
        "read_market_snapshot_history_store",
        lambda market_data_date: invalid_cached_history.copy(),
    )

    quick_board, meta = dashboard.load_latest_close_quick_board(
        horizon_days=3,
        positive_return=0.10,
        ranking_by="鍏虫敞鍒嗘暟",
        board_size=10,
    )

    assert not quick_board.empty
    assert quick_board.attrs["data_mode"] == "latest_close_quick_board"
    assert quick_board.attrs["quick_board_pending"] is True
    assert meta["board_date"] == "2026-05-06"
    assert set(quick_board["name"].tolist()) == {"A", "B"}


def test_fallback_candidate_pool_prefers_strategy_pool(monkeypatch):
    universe = pd.DataFrame(
        [
            {"symbol": "600001", "name": "A"},
            {"symbol": "600002", "name": "B"},
        ]
    )

    monkeypatch.setattr(dashboard, "_latest_market_close_date", lambda *args, **kwargs: "2026-04-18")
    monkeypatch.setattr(
        dashboard,
        "_build_strategy_candidate_pool",
        lambda universe, market_data_date: pd.DataFrame(
            [{"symbol": "600002", "name": "B", "industry_name": "消费"}]
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "_build_dynamic_fallback_candidate_pool",
        lambda universe, market_data_date: pd.DataFrame([{"symbol": "600001", "name": "A"}]),
    )

    result = dashboard._fallback_candidate_pool(universe)

    assert result == [("600002", "B")]

@pytest.mark.skip(reason="strategy selection logic moved to store module")
def test_build_strategy_candidate_pool_selects_both_rule_sets(monkeypatch):
    trade_dates = pd.bdate_range("2026-03-20", periods=20)
    stock_basic = pd.DataFrame(
        [
            {"ts_code": "600001.SH", "symbol": "600001", "name": "趋势股", "industry": "消费", "market": "主板"},
            {"ts_code": "600002.SH", "symbol": "600002", "name": "突破股", "industry": "消费", "market": "主板"},
            {"ts_code": "600003.SH", "symbol": "600003", "name": "板块股1", "industry": "消费", "market": "主板"},
            {"ts_code": "600004.SH", "symbol": "600004", "name": "板块股2", "industry": "消费", "market": "主板"},
        ]
    )

    def build_rows(symbol, closes, amounts, turnovers):
        rows = []
        prev_close = closes[0]
        for index, (trade_date, close, amount, turnover) in enumerate(zip(trade_dates, closes, amounts, turnovers)):
            pct = 0.0 if index == 0 else (close / prev_close - 1) * 100
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "close": close,
                    "vol": 100000 - index * 1000,
                    "amount": amount,
                    "turnover_rate": turnover,
                    "pct_chg": pct,
                }
            )
            prev_close = close
        return rows

    daily_window = pd.DataFrame(
        build_rows(
            "600001",
            [10.0, 10.1, 10.2, 10.35, 10.5, 10.7, 10.9, 11.1, 11.3, 11.5, 11.8, 12.0, 12.2, 12.4, 12.3, 12.2, 12.15, 12.18, 12.25, 12.7],
            [1.6e8] * 14 + [1.4e8, 1.3e8, 1.2e8, 1.1e8, 1.5e8, 2.5e8],
            [2.5] * 19 + [3.8],
        )
        + build_rows(
            "600002",
            [10.0, 10.05, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 11.0, 11.1, 11.3, 11.5, 11.7, 11.5, 11.7, 11.9, 12.1, 13.7],
            [2.0e8] * 17 + [2.4e8, 2.8e8, 3.8e8],
            [3.0] * 17 + [4.0, 4.8, 6.2],
        )
        + build_rows(
            "600003",
            [9.8, 9.85, 9.9, 9.92, 9.95, 10.0, 10.05, 10.08, 10.12, 10.16, 10.2, 10.25, 10.3, 10.35, 10.42, 10.5, 10.58, 10.66, 10.75, 11.0],
            [1.1e8] * 20,
            [2.0] * 20,
        )
        + build_rows(
            "600004",
            [9.9, 9.92, 9.95, 9.98, 10.0, 10.02, 10.06, 10.1, 10.15, 10.2, 10.24, 10.28, 10.32, 10.36, 10.4, 10.46, 10.52, 10.6, 10.7, 10.95],
            [1.0e8] * 20,
            [2.1] * 20,
        )
    )

    monkeypatch.setattr(dashboard, "fetch_tushare_stock_basic", lambda: stock_basic)
    monkeypatch.setattr(dashboard, "fetch_tushare_daily_window", lambda end_date=None, window=22: daily_window)
    history_by_symbol = {}
    for symbol in ["600001", "600002", "600003", "600004"]:
        history_rows = [row for row in daily_window.to_dict("records") if str(row["symbol"]) == symbol]
        history_frame = pd.DataFrame(history_rows).rename(
            columns={"trade_date": "date", "pct_chg": "change_pct", "turnover_rate": "turnover"}
        )
        history_frame["date"] = pd.to_datetime(history_frame["date"], errors="coerce")
        history_frame["high"] = history_frame["close"] * 1.01
        history_frame["low"] = history_frame["close"] * 0.99
        history_by_symbol[symbol] = history_frame
    monkeypatch.setattr(dashboard, "fetch_daily_history", lambda symbol, start_date=None: history_by_symbol[str(symbol)])

    universe = stock_basic[["symbol", "name"]].copy()
    result = dashboard._build_strategy_candidate_pool(universe, str(trade_dates[-1].date()))

    assert {"600001", "600002"}.issubset(set(result["symbol"]))
    assert result.set_index("symbol").loc["600001", "candidate_strategy"] == "策略1"
    assert result.set_index("symbol").loc["600002", "candidate_strategy"] == "策略2"
def test_build_strategy_candidate_pool_skips_symbols_without_latest_market_date(monkeypatch):
    trade_dates = pd.bdate_range("2026-03-25", periods=20)
    stock_basic = pd.DataFrame(
        [
            {"ts_code": "600001.SH", "symbol": "600001", "name": "A", "industry": "消费", "market": "主板"},
            {"ts_code": "600002.SH", "symbol": "600002", "name": "B", "industry": "消费", "market": "主板"},
            {"ts_code": "600003.SH", "symbol": "600003", "name": "C", "industry": "消费", "market": "主板"},
            {"ts_code": "600004.SH", "symbol": "600004", "name": "D", "industry": "消费", "market": "主板"},
        ]
    )

    def build_rows(symbol, closes, amounts, turnovers):
        rows = []
        prev_close = closes[0]
        for index, (trade_date, close, amount, turnover) in enumerate(zip(trade_dates, closes, amounts, turnovers)):
            pct = 0.0 if index == 0 else (close / prev_close - 1) * 100
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "close": close,
                    "vol": 100000 - index * 1000,
                    "amount": amount,
                    "turnover_rate": turnover,
                    "pct_chg": pct,
                }
            )
            prev_close = close
        return rows

    daily_window = pd.DataFrame(
        build_rows(
            "600001",
            [10.0, 10.1, 10.2, 10.35, 10.5, 10.7, 10.9, 11.1, 11.3, 11.5, 11.8, 12.0, 12.2, 12.4, 12.3, 12.2, 12.15, 12.18, 12.25, 12.7],
            [1.6e8] * 14 + [1.4e8, 1.3e8, 1.2e8, 1.1e8, 1.5e8, 2.5e8],
            [2.5] * 19 + [3.8],
        )
        + build_rows(
            "600002",
            [10.0, 10.05, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 11.0, 11.1, 11.3, 11.5, 11.7, 11.5, 11.7, 11.9, 12.1, 13.7],
            [2.0e8] * 17 + [2.4e8, 2.8e8, 3.8e8],
            [3.0] * 17 + [4.0, 4.8, 6.2],
        )
        + build_rows(
            "600003",
            [9.8, 9.85, 9.9, 9.92, 9.95, 10.0, 10.05, 10.08, 10.12, 10.16, 10.2, 10.25, 10.3, 10.35, 10.42, 10.5, 10.58, 10.66, 10.75, 11.0],
            [1.1e8] * 20,
            [2.0] * 20,
        )
        + build_rows(
            "600004",
            [9.9, 9.92, 9.95, 9.98, 10.0, 10.02, 10.06, 10.1, 10.15, 10.2, 10.24, 10.28, 10.32, 10.36, 10.4, 10.46, 10.52, 10.6, 10.7, 10.95],
            [1.0e8] * 20,
            [2.1] * 20,
        )
    )

    monkeypatch.setattr(dashboard, "fetch_tushare_stock_basic", lambda: stock_basic)
    monkeypatch.setattr(dashboard, "fetch_tushare_daily_window", lambda end_date=None, window=22: daily_window)

    history_by_symbol = {}
    for symbol in ["600001", "600002", "600003", "600004"]:
        history_rows = [row for row in daily_window.to_dict("records") if str(row["symbol"]) == symbol]
        history_frame = pd.DataFrame(history_rows).rename(
            columns={"trade_date": "date", "pct_chg": "change_pct", "turnover_rate": "turnover"}
        )
        history_frame["date"] = pd.to_datetime(history_frame["date"], errors="coerce")
        history_frame["high"] = history_frame["close"] * 1.01
        history_frame["low"] = history_frame["close"] * 0.99
        history_by_symbol[symbol] = history_frame
    history_by_symbol["600002"] = history_by_symbol["600002"].iloc[:-1].reset_index(drop=True)

    monkeypatch.setattr(dashboard, "fetch_daily_history", lambda symbol, start_date=None: history_by_symbol[str(symbol)])

    universe = stock_basic[["symbol", "name"]].copy()
    result = dashboard._build_strategy_candidate_pool(universe, "2026-04-21")

    assert "600001" in set(result["symbol"])
    assert "600002" not in set(result["symbol"])


@pytest.mark.skip(reason="exact-date candidate filtering moved to store module")
def test_build_strategy_candidate_pool_skips_symbols_without_latest_market_date(monkeypatch):
    trade_dates = pd.bdate_range("2026-03-25", periods=20)
    stock_basic = pd.DataFrame(
        [
            {"ts_code": "600001.SH", "symbol": "600001", "name": "A", "industry": "娑堣垂", "market": "涓绘澘"},
            {"ts_code": "600002.SH", "symbol": "600002", "name": "B", "industry": "娑堣垂", "market": "涓绘澘"},
            {"ts_code": "600003.SH", "symbol": "600003", "name": "C", "industry": "娑堣垂", "market": "涓绘澘"},
            {"ts_code": "600004.SH", "symbol": "600004", "name": "D", "industry": "娑堣垂", "market": "涓绘澘"},
        ]
    )

    def build_rows(symbol, closes, amounts, turnovers):
        rows = []
        prev_close = closes[0]
        for index, (trade_date, close, amount, turnover) in enumerate(zip(trade_dates, closes, amounts, turnovers)):
            pct = 0.0 if index == 0 else (close / prev_close - 1) * 100
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "close": close,
                    "vol": 100000 - index * 1000,
                    "amount": amount,
                    "turnover_rate": turnover,
                    "pct_chg": pct,
                }
            )
            prev_close = close
        return rows

    daily_window = pd.DataFrame(
        build_rows(
            "600001",
            [10.0, 10.1, 10.2, 10.35, 10.5, 10.7, 10.9, 11.1, 11.3, 11.5, 11.8, 12.0, 12.2, 12.4, 12.3, 12.2, 12.15, 12.18, 12.25, 12.7],
            [1.6e8] * 14 + [1.4e8, 1.3e8, 1.2e8, 1.1e8, 1.5e8, 2.5e8],
            [2.5] * 19 + [3.8],
        )
        + build_rows(
            "600002",
            [10.0, 10.05, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 11.0, 11.1, 11.3, 11.5, 11.7, 11.5, 11.7, 11.9, 12.1, 13.7],
            [2.0e8] * 17 + [2.4e8, 2.8e8, 3.8e8],
            [3.0] * 17 + [4.0, 4.8, 6.2],
        )
        + build_rows(
            "600003",
            [9.8, 9.85, 9.9, 9.92, 9.95, 10.0, 10.05, 10.08, 10.12, 10.16, 10.2, 10.25, 10.3, 10.35, 10.42, 10.5, 10.58, 10.66, 10.75, 11.0],
            [1.1e8] * 20,
            [2.0] * 20,
        )
        + build_rows(
            "600004",
            [9.9, 9.92, 9.95, 9.98, 10.0, 10.02, 10.06, 10.1, 10.15, 10.2, 10.24, 10.28, 10.32, 10.36, 10.4, 10.46, 10.52, 10.6, 10.7, 10.95],
            [1.0e8] * 20,
            [2.1] * 20,
        )
    )

    monkeypatch.setattr(dashboard, "fetch_tushare_stock_basic", lambda: stock_basic)
    monkeypatch.setattr(dashboard, "fetch_tushare_daily_window", lambda end_date=None, window=22: daily_window)

    history_by_symbol = {}
    for symbol in ["600001", "600002", "600003", "600004"]:
        history_rows = [row for row in daily_window.to_dict("records") if str(row["symbol"]) == symbol]
        history_frame = pd.DataFrame(history_rows).rename(
            columns={"trade_date": "date", "pct_chg": "change_pct", "turnover_rate": "turnover"}
        )
        history_frame["date"] = pd.to_datetime(history_frame["date"], errors="coerce")
        history_frame["high"] = history_frame["close"] * 1.01
        history_frame["low"] = history_frame["close"] * 0.99
        history_by_symbol[symbol] = history_frame
    history_by_symbol["600002"] = history_by_symbol["600002"].iloc[:-1].reset_index(drop=True)

    monkeypatch.setattr(dashboard, "fetch_daily_history", lambda symbol, start_date=None: history_by_symbol[str(symbol)])

    universe = stock_basic[["symbol", "name"]].copy()
    result = dashboard._build_strategy_candidate_pool(universe, "2026-04-21")

    assert "600001" in set(result["symbol"])
    assert "600002" not in set(result["symbol"])


def test_prepare_symbol_base_analysis_truncates_to_market_data_date(monkeypatch):
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-18", "2026-04-20", "2026-04-21"]),
            "close": [10.0, 10.5, 11.5],
            "change_pct": [0.0, 5.0, 9.52],
            "amount": [1.1e8, 2.2e8, 3.3e8],
            "turnover": [2.0, 3.5, 4.8],
        }
    )
    features = pd.DataFrame([{"signal": 1.0}, {"signal": 2.0}, {"signal": 3.0}])

    monkeypatch.setattr(dashboard, "fetch_daily_history", lambda symbol, start_date=None: daily)
    monkeypatch.setattr(dashboard, "get_market_daily_feature_row", lambda symbol, market_data_date: None)
    monkeypatch.setattr(dashboard, "_read_symbol_base_analysis_disk_cache", lambda symbol, market_data_date: None)
    monkeypatch.setattr(dashboard, "_write_symbol_base_analysis_disk_cache", lambda symbol, market_data_date, data: None)
    monkeypatch.setattr(dashboard, "build_daily_features", lambda daily_df: features.iloc[: len(daily_df)].reset_index(drop=True))
    monkeypatch.setattr(
        dashboard,
        "latest_snapshot",
        lambda daily_df, feature_df: {
            "date": daily_df["date"].iloc[-1].strftime("%Y-%m-%d"),
            "change_pct": float(daily_df["change_pct"].iloc[-1]),
        },
    )
    monkeypatch.setattr(
        dashboard,
        "classify_stage",
        lambda daily_df: SimpleNamespace(label="趋势上行", priority=2, structure_summary="ok"),
    )
    monkeypatch.setattr(dashboard, "evaluate_quant_signal", lambda daily_df, feature_df: SimpleNamespace(total_score=66.0, primary_signal="trend"))
    monkeypatch.setattr(
        dashboard,
        "build_trading_rule_context",
        lambda symbol, name: SimpleNamespace(board_label="主板", price_limit_label="10%", rule_summary="ok"),
    )
    monkeypatch.setattr(dashboard, "stage_numeric_score", lambda stage, latest_features: 72.0)
    monkeypatch.setattr(dashboard, "main_rise_start_score", lambda latest_features: 61.0)

    result = dashboard._prepare_symbol_base_analysis("600001", "A", market_data_date="2026-04-20")

    assert result is not None
    assert result["analysis_date"] == "2026-04-20"
    assert result["latest_price"] == 10.5
    assert result["change_pct"] == 5.0


def test_prepare_symbol_base_analysis_requires_exact_market_data_date(monkeypatch):
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-18", "2026-04-20"]),
            "close": [10.0, 10.5],
            "change_pct": [0.0, 5.0],
            "amount": [1.1e8, 2.2e8],
            "turnover": [2.0, 3.5],
        }
    )

    monkeypatch.setattr(dashboard, "fetch_daily_history", lambda symbol, start_date=None: daily)
    monkeypatch.setattr(dashboard, "get_market_daily_feature_row", lambda symbol, market_data_date: None)
    monkeypatch.setattr(dashboard, "_read_symbol_base_analysis_disk_cache", lambda symbol, market_data_date: None)

    result = dashboard._prepare_symbol_base_analysis("600001", "A", market_data_date="2026-04-21")

    assert result is None


def test_align_daily_history_to_market_date_handles_date_index_and_column():
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-18", "2026-04-20", "2026-04-21"]),
            "close": [10.0, 10.5, 11.5],
        }
    ).set_index("date", drop=False)

    aligned = dashboard._align_daily_history_to_market_date(daily, "2026-04-20", require_exact=True)

    assert aligned["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-04-18", "2026-04-20"]
    assert aligned["close"].tolist() == [10.0, 10.5]


def test_build_display_board_keeps_strategy_pool_without_old_streak_filter():
    board = pd.DataFrame(
        [
            {
                "symbol": "600001",
                "name": "A",
                "consecutive_up_days": 1,
                "attention_score": 90.0,
                "probability_up": 70.0,
                "amount": 3e8,
                "latest_price": 12.3,
                "change_pct": 2.8,
                "turnover": 5.2,
                "quant_score": 66.0,
                "stage_label": "策略启动",
                "reason": "策略1入围",
                "candidate_strategy": "策略1",
            }
        ]
    )

    result = dashboard._build_display_board(
        board,
        board_size=10,
        ranking_by="关注分数",
        data_mode="strategy_candidate_pool",
        loading=True,
    )

    assert result["symbol"].tolist() == ["600001"]
    assert result.attrs["focus_filter_mode"] == "strategy_hard_filter"
    assert result.iloc[0]["candidate_strategy_label"] == "策略1·趋势中继"


def test_build_display_board_keeps_latest_close_quick_board_without_old_streak_filter():
    board = pd.DataFrame(
        [
            {
                "symbol": "600001",
                "name": "A",
                "consecutive_up_days": 1,
                "attention_score": 90.0,
                "enhanced_attention_score": 93.0,
                "probability_up": 72.0,
                "amount": 3e8,
                "latest_price": 12.3,
                "change_pct": 2.8,
                "turnover": 5.2,
                "quant_score": 66.0,
                "stage_label": "快榜启动",
                "reason": "最新收盘快榜命中",
                "candidate_strategy": "dynamic_fallback",
            }
        ]
    )

    result = dashboard._build_display_board(
        board,
        board_size=10,
        ranking_by="鍏虫敞鍒嗘暟",
        data_mode="latest_close_quick_board",
        loading=True,
    )

    assert result["symbol"].tolist() == ["600001"]
    assert result.attrs["focus_filter_mode"] == "quick_board_latest_close"


def test_candidate_strategy_profile_differentiates_prediction_modes():
    strategy1 = dashboard._apply_candidate_strategy_prediction_profile(
        {
            "candidate_strategy": "策略1",
            "probability_up": 60.0,
            "predicted_upside_pct": 10.0,
            "predicted_upside_low_pct": 6.0,
            "predicted_upside_high_pct": 14.0,
            "attention_score": 70.0,
            "enhanced_attention_score": 72.0,
        }
    )
    strategy2 = dashboard._apply_candidate_strategy_prediction_profile(
        {
            "candidate_strategy": "策略2",
            "probability_up": 60.0,
            "predicted_upside_pct": 10.0,
            "predicted_upside_low_pct": 6.0,
            "predicted_upside_high_pct": 14.0,
            "attention_score": 70.0,
            "enhanced_attention_score": 72.0,
        }
    )

    assert strategy1["candidate_strategy_label"] == "策略1·趋势中继"
    assert strategy2["candidate_strategy_label"] == "策略2·突破共振"
    assert strategy1["probability_up"] > strategy2["probability_up"]
    assert strategy1["predicted_upside_pct"] < strategy2["predicted_upside_pct"]


def test_candidate_strategy_profile_treats_unknown_strategy_as_generic():
    unknown_strategy = dashboard._apply_candidate_strategy_prediction_profile(
        {
            "candidate_strategy": "策略X",
            "probability_up": 60.0,
            "predicted_upside_pct": 10.0,
            "predicted_upside_low_pct": 6.0,
            "predicted_upside_high_pct": 14.0,
            "attention_score": 70.0,
            "enhanced_attention_score": 72.0,
        }
    )

    assert unknown_strategy["candidate_strategy_label"] == "通用模型"
    assert unknown_strategy["candidate_strategy_short_label"] == "通用"
    assert unknown_strategy["probability_up"] == 60.0
    assert unknown_strategy["predicted_upside_pct"] == 10.0


def test_candidate_strategy_profile_supports_strategy3():
    strategy3 = dashboard._apply_candidate_strategy_prediction_profile(
        {
            "candidate_strategy": "strategy3",
            "probability_up": 60.0,
            "predicted_upside_pct": 10.0,
            "predicted_upside_low_pct": 6.0,
            "predicted_upside_high_pct": 14.0,
            "attention_score": 70.0,
            "enhanced_attention_score": 72.0,
        }
    )

    assert strategy3["candidate_strategy"] == "strategy3"
    assert strategy3["candidate_strategy_label"] == "策略3·多因子主升预备"
    assert strategy3["candidate_strategy_short_label"] == "多因子主升"
