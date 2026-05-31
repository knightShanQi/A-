import math
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import a_share_predictor.modeling as modeling
from a_share_predictor.features import FEATURE_COLUMNS
from a_share_predictor.modeling import (
    GLOBAL_MODEL_TEST_END,
    GLOBAL_MODEL_TEST_START,
    GLOBAL_MODEL_TRAIN_END,
    GLOBAL_MODEL_TRAIN_START,
    MarketProxyModel,
    MODEL_FEATURE_COLUMNS,
    ProbabilityResult,
    _apply_probability_calibrator,
    _apply_incremental_probability_upgrade,
    _build_launch_specialist_frame,
    _build_backtest_metrics,
    _build_context_score_frame,
    _derive_dynamic_ensemble_weights,
    _fit_probability_calibrator,
    _safe_roc_auc_score,
    train_market_wide_model,
)


def test_safe_roc_auc_score_returns_nan_for_single_class_targets():
    score = _safe_roc_auc_score([1, 1, 1], [0.2, 0.4, 0.8])
    assert math.isnan(score)


def test_safe_roc_auc_score_returns_value_for_binary_targets():
    score = _safe_roc_auc_score([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
    assert score == 1.0


def test_predict_latest_probability_matches_full_model(monkeypatch):
    feature_rows = []
    for index in range(150):
        row = {column: float((index % 7) + 1) for column in FEATURE_COLUMNS}
        row["target"] = 1 if index >= 75 else 0
        row["future_return"] = 0.05 if row["target"] else -0.02
        feature_rows.append(row)
    dataset = pd.DataFrame(feature_rows)

    monkeypatch.setattr(modeling, "build_training_frame", lambda *args, **kwargs: dataset.copy())

    latest_probability = modeling.predict_latest_probability(pd.DataFrame(), horizon_days=5, positive_return=0.03)
    full_model = modeling.train_probability_model(pd.DataFrame(), horizon_days=5, positive_return=0.03)

    assert latest_probability == full_model.latest_probability


def test_predict_latest_probability_uses_current_feature_snapshot(monkeypatch):
    training_rows = []
    for index in range(150):
        is_positive = 60 <= index < 149
        feature_value = 4.0 if is_positive else -3.0
        row = {column: feature_value for column in FEATURE_COLUMNS}
        row["target"] = 1 if is_positive else 0
        row["future_return"] = 0.05 if row["target"] else -0.03
        training_rows.append(row)
    dataset = pd.DataFrame(training_rows)

    latest_feature_frame = pd.DataFrame([{column: 4.0 for column in FEATURE_COLUMNS}])

    monkeypatch.setattr(modeling, "build_training_frame", lambda *args, **kwargs: dataset.copy())
    monkeypatch.setattr(modeling, "build_daily_features", lambda *args, **kwargs: latest_feature_frame.copy())

    latest_probability = modeling.predict_latest_probability(pd.DataFrame(), horizon_days=5, positive_return=0.03)
    full_model = modeling.train_probability_model(pd.DataFrame(), horizon_days=5, positive_return=0.03)

    assert latest_probability == full_model.latest_probability
    assert latest_probability > 0.5
    assert full_model.signal_label
    assert 0 <= full_model.strategy_score <= 100
    assert "top_bucket_return" in full_model.metrics


def test_predict_latest_probability_uses_proxy_model_when_available(monkeypatch):
    proxy_model = object()
    called: dict[str, object] = {}

    def fake_score(daily, market_proxy_model, latest_feature_values=None):
        called["market_proxy_model"] = market_proxy_model
        called["latest_feature_values"] = latest_feature_values
        return ProbabilityResult(
            latest_probability=0.77,
            probabilities=pd.Series([0.77], dtype=float),
            metrics={},
            coefficients=[],
        )

    monkeypatch.setattr(modeling, "score_with_market_proxy_model", fake_score)
    monkeypatch.setattr(
        modeling,
        "build_training_frame",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("slow path should not run")),
    )

    latest_probability = modeling.predict_latest_probability(
        pd.DataFrame(),
        horizon_days=5,
        positive_return=0.03,
        market_proxy_model=proxy_model,
        latest_feature_values={"ret_1": 0.01},
    )

    assert latest_probability == 0.77
    assert called["market_proxy_model"] is proxy_model
    assert called["latest_feature_values"] == {"ret_1": 0.01}


def test_predict_latest_probability_fast_fallback_skips_slow_training(monkeypatch):
    monkeypatch.setattr(modeling, "_estimate_fast_latest_probability", lambda *args, **kwargs: 0.66)
    monkeypatch.setattr(
        modeling,
        "build_training_frame",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("slow path should not run")),
    )

    latest_probability = modeling.predict_latest_probability(
        pd.DataFrame(),
        horizon_days=5,
        positive_return=0.03,
        allow_slow_fallback=False,
    )

    assert latest_probability == 0.66


def test_news_snapshot_features_include_research_priors():
    trade_dates = pd.DatetimeIndex(pd.to_datetime(["2026-01-02", "2026-01-05"]))
    news = pd.DataFrame(
        {
            "title": ["new product approval and technology breakthrough"],
            "content": ["patent approval improves product pipeline"],
            "published_at": pd.to_datetime(["2026-01-02 10:00"]),
            "source": ["newswire"],
        }
    )

    features = modeling._build_news_snapshot_features(trade_dates, news)

    assert "news_research_score_3d" in features.columns
    assert "news_research_excess_1d" in features.columns
    assert features.loc[pd.Timestamp("2026-01-02"), "news_research_score_3d"] > 0.0
    assert features.loc[pd.Timestamp("2026-01-02"), "news_research_excess_1d"] > 0.0


def test_build_backtest_metrics_detects_precision_gate():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=int)
    y_prob = np.array([0.20, 0.25, 0.32, 0.41, 0.58, 0.61, 0.72, 0.83, 0.91, 0.95], dtype=float)
    future_return = pd.Series([-0.03, -0.02, -0.01, -0.01, 0.01, 0.02, 0.03, 0.05, 0.08, 0.09], dtype=float)

    metrics = _build_backtest_metrics(y_true, y_prob, future_return)

    assert metrics["precision_target"] == 0.90
    assert metrics["precision_target_reached"] == 1.0
    assert metrics["precision_gate_threshold"] >= 0.55
    assert metrics["precision_gate_precision"] >= 0.90
    assert metrics["precision_gate_support"] >= 4


def test_recent_tail_backtest_profile_uses_latest_window():
    y_true = np.array(([0] * 18) + ([1] * 18), dtype=int)
    y_prob = np.array(([0.15] * 18) + ([0.85] * 18), dtype=float)
    future_return = pd.Series(([-0.02] * 18) + ([0.04] * 18), dtype=float)

    profile = modeling._recent_tail_backtest_profile(y_true, y_prob, future_return, window=12, min_samples=8)

    assert profile["recent_backtest_ready"] == 1.0
    assert profile["recent_backtest_window"] == 12.0
    assert profile["recent_sample_size"] == 12.0


def test_apply_recent_backtest_guard_shrinks_overfit_like_signal():
    metrics = {
        "sample_size": 180.0,
        "positive_rate": 0.42,
        "roc_auc": 0.71,
        "brier": 0.18,
        "top_bucket_return": 0.045,
        "spread_return": 0.06,
        "precision_gate_precision": 0.88,
        "recent_backtest_window": 24.0,
        "recent_backtest_ready": 1.0,
        "recent_roc_auc": 0.56,
        "recent_brier": 0.25,
        "recent_top_bucket_return": 0.008,
        "recent_precision_gate_precision": 0.61,
        "recent_precision_gate_support": 9.0,
    }

    guarded_probability, upside, low, high, guard_info, note = modeling._apply_recent_backtest_guard(
        0.82,
        metrics,
        predicted_upside_pct=16.0,
        predicted_upside_low_pct=10.0,
        predicted_upside_high_pct=22.0,
    )

    assert guarded_probability < 0.82
    assert upside < 16.0
    assert low <= upside
    assert high >= upside
    assert guard_info["recent_guard_active"] == 1.0
    assert note


def test_apply_recent_backtest_guard_stays_idle_without_recent_support():
    guarded_probability, upside, low, high, guard_info, note = modeling._apply_recent_backtest_guard(
        0.67,
        {"sample_size": 50.0, "positive_rate": 0.44, "recent_backtest_window": 0.0, "recent_backtest_ready": 0.0},
        predicted_upside_pct=12.0,
        predicted_upside_low_pct=8.0,
        predicted_upside_high_pct=16.0,
    )

    assert guarded_probability == 0.67
    assert upside == 12.0
    assert low == 8.0
    assert high == 16.0
    assert guard_info["recent_guard_active"] == 0.0
    assert note == ""


def test_dynamic_ensemble_weights_favor_better_component():
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0], dtype=int)
    future_return = pd.Series([0.05 if label else -0.03 for label in y_true], dtype=float)
    component_matrix = np.column_stack(
        [
            np.array([0.10, 0.18, 0.22, 0.76, 0.82, 0.87, 0.91, 0.29, 0.78, 0.14, 0.83, 0.20], dtype=float),
            np.array([0.46, 0.48, 0.51, 0.53, 0.55, 0.57, 0.58, 0.49, 0.54, 0.47, 0.56, 0.50], dtype=float),
            np.array([0.84, 0.80, 0.77, 0.24, 0.19, 0.15, 0.12, 0.74, 0.27, 0.79, 0.18, 0.75], dtype=float),
        ]
    )

    weights = _derive_dynamic_ensemble_weights(y_true, component_matrix, future_return)

    assert weights[0] > weights[1]
    assert weights[0] > weights[2]
    assert math.isclose(float(weights.sum()), 1.0, rel_tol=1e-9)


def test_probability_calibrator_returns_bounded_probabilities():
    raw_probability = np.linspace(0.06, 0.94, 120)
    y_true = np.array([0] * 60 + [1] * 60, dtype=int)
    component_matrix = np.column_stack(
        [
            raw_probability,
            np.clip(raw_probability * 0.92 + 0.03, 0.0, 1.0),
            np.clip(raw_probability * 1.05 - 0.02, 0.0, 1.0),
        ]
    )

    calibrator = _fit_probability_calibrator(y_true, raw_probability, component_matrix)
    calibrated = _apply_probability_calibrator(raw_probability, component_matrix, calibrator)

    assert calibrator is not None
    assert np.all(calibrated > 0.0)
    assert np.all(calibrated < 1.0)


def test_context_score_frame_tracks_launch_and_market_resonance():
    feature_frame = pd.DataFrame(
        [
            {
                "ret_20": 0.05,
                "ret_60": 0.08,
                "close_vs_ma20": 0.03,
                "ma20_slope_5": 0.01,
                "breakout_distance_20": -0.01,
                "range_position_20": 0.64,
                "volume_ratio_5": 1.02,
                "volume_ratio_20": 1.00,
                "amount_ratio_5": 1.00,
                "pullback_to_breakout_20": -0.02,
                "lower_shadow_ratio": 0.06,
                "upper_shadow_ratio": 0.10,
                "body_ratio": 0.40,
                "volatility_10": 0.02,
                "turnover_ratio_20": 1.00,
                "close_position_day": 0.55,
                "volatility_contraction": -0.02,
                "ma_alignment_score": 0.58,
                "momentum_persistence_10": 0.56,
                "efficiency_ratio_10": 0.45,
                "downside_vol_ratio_20": 0.22,
                "relative_strength_5": 0.01,
                "relative_strength_20": 0.02,
                "market_ret_5": 0.003,
                "market_ret_20": 0.01,
                "market_close_vs_ma20": 0.01,
                "market_volatility_10": 0.015,
                "market_regime_score": 0.55,
                "market_regime_risk": 0.18,
                "news_sentiment_7d": 0.12,
                "news_confidence_7d": 0.45,
                "news_positive_ratio_7d": 0.54,
                "fund_ratio_5d": 0.08,
                "fund_net_strength_5d": 0.06,
                "fund_positive_ratio_5d": 0.52,
                "fund_inflow_streak_5d": 0.20,
                "fund_consistency_5d": 0.36,
                "launch_readiness": 52.0,
                "market_resonance": 48.0,
            },
            {
                "ret_20": 0.11,
                "ret_60": 0.17,
                "close_vs_ma20": 0.06,
                "ma20_slope_5": 0.03,
                "breakout_distance_20": 0.02,
                "range_position_20": 0.83,
                "volume_ratio_5": 1.30,
                "volume_ratio_20": 1.18,
                "amount_ratio_5": 1.24,
                "pullback_to_breakout_20": -0.01,
                "lower_shadow_ratio": 0.10,
                "upper_shadow_ratio": 0.05,
                "body_ratio": 0.55,
                "volatility_10": 0.015,
                "turnover_ratio_20": 1.18,
                "close_position_day": 0.78,
                "volatility_contraction": -0.05,
                "ma_alignment_score": 0.74,
                "momentum_persistence_10": 0.71,
                "efficiency_ratio_10": 0.62,
                "downside_vol_ratio_20": 0.16,
                "relative_strength_5": 0.03,
                "relative_strength_20": 0.05,
                "market_ret_5": 0.01,
                "market_ret_20": 0.03,
                "market_close_vs_ma20": 0.025,
                "market_volatility_10": 0.01,
                "market_regime_score": 0.82,
                "market_regime_risk": 0.10,
                "news_sentiment_7d": 0.28,
                "news_confidence_7d": 0.72,
                "news_positive_ratio_7d": 0.66,
                "fund_ratio_5d": 0.18,
                "fund_net_strength_5d": 0.15,
                "fund_positive_ratio_5d": 0.64,
                "fund_inflow_streak_5d": 0.60,
                "fund_consistency_5d": 0.74,
                "launch_readiness": 81.0,
                "market_resonance": 79.0,
            },
        ]
    )

    score_frame = _build_context_score_frame(feature_frame)

    assert "launch_readiness_score" in score_frame.columns
    assert "market_resonance_score" in score_frame.columns
    assert {"launch_readiness", "breakout_quality", "resonance_quality", "risk_of_late_entry", "launch_phase_label"}.issubset(
        score_frame.columns
    )
    assert score_frame.loc[1, "launch_readiness_score"] > score_frame.loc[0, "launch_readiness_score"]
    assert score_frame.loc[1, "market_resonance_score"] > score_frame.loc[0, "market_resonance_score"]
    assert score_frame.loc[1, "breakout_quality"] > score_frame.loc[0, "breakout_quality"]
    assert score_frame.loc[1, "resonance_quality"] > score_frame.loc[0, "resonance_quality"]
    assert score_frame.loc[1, "launch_phase_label"] == "刚启动"
    assert score_frame.loc[1, "context_composite_score"] > score_frame.loc[0, "context_composite_score"]


def test_launch_specialist_frame_rewards_regime_fit_and_launch_shape():
    feature_frame = pd.DataFrame(
        [
            {
                "launch_readiness": 48.0,
                "market_resonance": 44.0,
                "trend_strength": 18.0,
                "breakout_readiness": 12.0,
                "pullback_quality": 8.0,
                "volume_thrust": 4.0,
                "risk_pressure": 96.0,
                "stretch_risk": 18.0,
                "market_regime_score": 0.32,
                "market_regime_risk": 0.58,
                "relative_strength_20": -0.01,
                "close_vs_ma20": 0.00,
                "breakout_distance_20": -0.04,
                "range_position_20": 0.49,
                "market_regime_label": "defense",
            },
            {
                "launch_readiness": 84.0,
                "market_resonance": 80.0,
                "trend_strength": 52.0,
                "breakout_readiness": 46.0,
                "pullback_quality": 32.0,
                "volume_thrust": 28.0,
                "risk_pressure": 42.0,
                "stretch_risk": 6.0,
                "market_regime_score": 0.82,
                "market_regime_risk": 0.14,
                "relative_strength_20": 0.05,
                "close_vs_ma20": 0.06,
                "breakout_distance_20": 0.01,
                "range_position_20": 0.76,
                "market_regime_label": "trend",
            },
        ]
    )

    specialist = _build_launch_specialist_frame(feature_frame)

    assert specialist.loc[1, "launch_specialist_score"] > specialist.loc[0, "launch_specialist_score"]
    assert specialist.loc[1, "launch_regime_fit_score"] > specialist.loc[0, "launch_regime_fit_score"]
    assert specialist.loc[1, "launch_specialist_delta_pct"] > 0
    assert specialist.loc[0, "launch_specialist_delta_pct"] < 0


def test_incremental_probability_upgrade_preserves_base_model_and_applies_small_delta():
    feature_frame = pd.DataFrame(
        [
            {
                **{column: 0.0 for column in MODEL_FEATURE_COLUMNS},
                "ret_20": 0.09,
                "ret_60": 0.18,
                "close_vs_ma20": 0.07,
                "ma20_slope_5": 0.03,
                "breakout_distance_20": 0.02,
                "range_position_20": 0.84,
                "volume_ratio_5": 1.65,
                "volume_ratio_20": 1.35,
                "amount_ratio_5": 1.42,
                "lower_shadow_ratio": 0.03,
                "body_ratio": 0.58,
                "turnover_ratio_20": 1.30,
                "close_position_day": 0.82,
                "ma_alignment_score": 0.92,
                "momentum_persistence_10": 0.88,
                "efficiency_ratio_10": 0.77,
                "relative_strength_5": 0.08,
                "relative_strength_20": 0.11,
                "market_ret_5": 0.02,
                "market_ret_20": 0.05,
                "market_close_vs_ma20": 0.03,
                "market_regime_score": 0.70,
                "news_sentiment_7d": 0.55,
                "news_confidence_7d": 0.72,
                "news_positive_ratio_7d": 0.76,
                "fund_ratio_5d": 0.42,
                "fund_net_strength_5d": 0.30,
                "fund_positive_ratio_5d": 0.74,
                "fund_inflow_streak_5d": 0.80,
                "fund_consistency_5d": 0.68,
            }
        ]
    )

    upgraded, detail = _apply_incremental_probability_upgrade(np.array([0.58], dtype=float), feature_frame)

    assert upgraded[0] > 0.58
    assert upgraded[0] < 0.69
    assert not detail.empty
    assert detail.iloc[0]["context_composite_score"] > 50
    assert abs(detail.iloc[0]["upgrade_delta_pct"]) < 12


def test_apply_live_probability_upgrade_adds_execution_context(monkeypatch):
    latest = pd.DataFrame([{column: 0.0 for column in MODEL_FEATURE_COLUMNS}])
    latest["ret_20"] = 0.06
    latest["ret_60"] = 0.12
    latest["close_vs_ma20"] = 0.04
    latest["ma20_slope_5"] = 0.02
    latest["volume_ratio_5"] = 1.25
    latest["market_ret_20"] = 0.03
    latest["market_regime_score"] = 0.60
    latest["news_confidence_7d"] = 0.65
    latest["fund_consistency_5d"] = 0.62

    monkeypatch.setattr(modeling, "_prepare_live_feature_frame", lambda *args, **kwargs: latest.copy())
    monkeypatch.setattr(modeling, "evaluate_intraday", lambda minute_df: {"score": 0.78, "max_pullback": 0.02})
    monkeypatch.setattr(
        modeling,
        "evaluate_intraday_structure_signal",
        lambda minute_df: SimpleNamespace(
            opening_volume_ratio=0.28,
            first30_volume_share=0.21,
            early_return_pct=0.014,
            label="强",
            summary="强",
        ),
    )
    monkeypatch.setattr(
        modeling,
        "evaluate_temporal_news_pulse",
        lambda news_df: SimpleNamespace(
            intraday_score=68.0,
            overnight_score=72.0,
            next_session_score=75.0,
            stronger_window="隔夜更强",
            summary="隔夜更强",
        ),
    )
    monkeypatch.setattr(modeling, "evaluate_news_sentiment", lambda news_df: {"sentiment_score": 70.0})
    monkeypatch.setattr(modeling, "evaluate_main_fund_signal", lambda fund_df: {"fund_score": 74.0})

    result = ProbabilityResult(
        latest_probability=0.61,
        probabilities=pd.Series([0.61], dtype=float),
        metrics={"precision_target": 0.90},
        coefficients=[],
        signal_breakdown={"trend_score": 62.0, "breakout_score": 58.0, "pullback_score": 54.0, "risk_score": 36.0},
        base_probability=0.58,
        predicted_upside_pct=12.5,
        predicted_upside_low_pct=8.4,
        predicted_upside_high_pct=17.1,
    )

    upgraded = modeling.apply_live_probability_upgrade(
        result,
        pd.DataFrame(),
        latest_feature_values={column: 0.0 for column in MODEL_FEATURE_COLUMNS},
        minute_df=pd.DataFrame({"x": [1]}),
        news_df=pd.DataFrame({"x": [1]}),
        fund_flow_df=pd.DataFrame({"x": [1]}),
        symbol="000001",
    )

    assert upgraded.latest_probability > result.latest_probability
    assert upgraded.signal_breakdown["intraday_execution_score"] > 0
    assert upgraded.signal_breakdown["live_context_score"] > 0
    assert upgraded.predicted_upside_pct > 0
    assert upgraded.predicted_upside_high_pct >= upgraded.predicted_upside_pct
    assert upgraded.upgrade_summary


def test_probability_result_keeps_latest_and_probability_tail_aligned():
    result = ProbabilityResult(
        latest_probability=0.62,
        probabilities=pd.Series([0.41, 0.62], dtype=float),
        metrics={},
        coefficients=[],
        raw_probability=0.55,
        enhanced_probability=0.66,
    )

    assert result.raw_probability == 0.55
    assert result.enhanced_probability == 0.66
    assert result.latest_probability == result.probabilities.iloc[-1]


def test_apply_live_probability_upgrade_supports_legacy_result_objects(monkeypatch):
    latest = pd.DataFrame([{column: 0.0 for column in MODEL_FEATURE_COLUMNS}])
    latest["ret_20"] = 0.05
    latest["close_vs_ma20"] = 0.03
    latest["volume_ratio_5"] = 1.18
    latest["market_ret_20"] = 0.02

    monkeypatch.setattr(modeling, "_prepare_live_feature_frame", lambda *args, **kwargs: latest.copy())
    monkeypatch.setattr(modeling, "evaluate_intraday", lambda minute_df: {"score": 0.70, "max_pullback": 0.01})
    monkeypatch.setattr(
        modeling,
        "evaluate_intraday_structure_signal",
        lambda minute_df: SimpleNamespace(
            opening_volume_ratio=0.22,
            first30_volume_share=0.18,
            early_return_pct=0.01,
            label="绋冲畾",
            summary="绋冲畾",
        ),
    )
    monkeypatch.setattr(
        modeling,
        "evaluate_temporal_news_pulse",
        lambda news_df: SimpleNamespace(
            intraday_score=60.0,
            overnight_score=66.0,
            next_session_score=69.0,
            stronger_window="闅斿",
            summary="闅斿",
        ),
    )
    monkeypatch.setattr(modeling, "evaluate_news_sentiment", lambda news_df: {"sentiment_score": 63.0})
    monkeypatch.setattr(modeling, "evaluate_main_fund_signal", lambda fund_df: {"fund_score": 67.0})

    legacy_result = SimpleNamespace(
        latest_probability=0.57,
        metrics={"precision_target": 0.90},
        signal_breakdown={"trend_score": 58.0, "breakout_score": 55.0, "pullback_score": 52.0, "risk_score": 38.0},
    )

    upgraded = modeling.apply_live_probability_upgrade(
        legacy_result,
        pd.DataFrame(),
        latest_feature_values={column: 0.0 for column in MODEL_FEATURE_COLUMNS},
        minute_df=pd.DataFrame({"x": [1]}),
        news_df=pd.DataFrame({"x": [1]}),
        fund_flow_df=pd.DataFrame({"x": [1]}),
        symbol="000001",
    )

    assert upgraded is legacy_result
    assert upgraded.latest_probability > 0.57
    assert upgraded.base_probability == 0.57
    assert upgraded.signal_breakdown["intraday_execution_score"] > 0
    assert upgraded.predicted_upside_pct > 0
    assert upgraded.predicted_upside_high_pct >= upgraded.predicted_upside_pct
    assert upgraded.upgrade_summary


def test_proxy_validation_split_prefers_global_train_and_test_windows():
    train_dates = pd.bdate_range(GLOBAL_MODEL_TRAIN_START, periods=220, freq="B")
    test_dates = pd.bdate_range(GLOBAL_MODEL_TEST_START, periods=60, freq="B")
    rows = []
    for idx, signal_date in enumerate([*train_dates, *test_dates]):
        target = 1 if idx % 4 in (1, 2) else 0
        row = {column: float((idx % 6) - 2) for column in MODEL_FEATURE_COLUMNS}
        row["signal_date"] = signal_date
        row["target"] = target
        row["future_return"] = 0.04 if target else -0.02
        rows.append(row)
    dataset = pd.DataFrame(rows)

    train_frame, validation_frame, split_label = modeling._proxy_validation_split(
        dataset,
        train_start=GLOBAL_MODEL_TRAIN_START,
        train_end=GLOBAL_MODEL_TRAIN_END,
        test_start=GLOBAL_MODEL_TEST_START,
        test_end=GLOBAL_MODEL_TEST_END,
    )

    assert split_label == "global_train_test_split"
    assert len(train_frame) == len(train_dates)
    assert len(validation_frame) == len(test_dates)
    assert train_frame["signal_date"].min() >= pd.Timestamp(GLOBAL_MODEL_TRAIN_START)
    assert validation_frame["signal_date"].min() >= pd.Timestamp(GLOBAL_MODEL_TEST_START)


def test_prepare_proxy_training_dataset_keeps_future_return():
    dataset = pd.DataFrame(
        {
            "signal_date": pd.bdate_range("2025-01-02", periods=4, freq="B"),
            "target": [0, 1, 0, 1],
            "future_return": [-0.02, 0.05, -0.01, 0.06],
            **{column: [0.1, 0.2, 0.3, 0.4] for column in MODEL_FEATURE_COLUMNS},
        }
    )

    prepared = modeling._prepare_proxy_training_dataset(dataset)

    assert "future_return" in prepared.columns
    assert prepared["future_return"].tolist() == [-0.02, 0.05, -0.01, 0.06]


def test_load_partial_market_dataset_falls_back_to_legacy_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(modeling, "CACHE_DIR", tmp_path)
    legacy_path = (
        tmp_path
        / "global_market_dataset_v2_h5_r300_20250101_20251231_20260101_20260331.partial.pkl"
    )
    legacy_payload = {
        "meta": {"schema_version": 2},
        "data": pd.DataFrame(
            {
                "symbol": ["000001"],
                "signal_date": [pd.Timestamp("2026-01-05")],
                "target": [1],
                "future_return": [0.05],
                MODEL_FEATURE_COLUMNS[0]: [0.1],
            }
        ),
    }
    legacy_path.write_bytes(pickle.dumps(legacy_payload))

    loaded = modeling._load_partial_market_dataset(
        horizon_days=5,
        positive_return=0.03,
        train_start=GLOBAL_MODEL_TRAIN_START,
        train_end=GLOBAL_MODEL_TRAIN_END,
        test_start=GLOBAL_MODEL_TEST_START,
        test_end=GLOBAL_MODEL_TEST_END,
    )

    assert not loaded.empty
    assert loaded.iloc[0]["symbol"] == "000001"


def test_select_market_proxy_candidate_prefers_highest_validation_objective(monkeypatch):
    dataset = pd.DataFrame(
        {
            "signal_date": pd.bdate_range("2025-01-02", periods=12, freq="B"),
            "target": [0, 1] * 6,
            "future_return": [-0.02, 0.04] * 6,
        }
    )

    monkeypatch.setattr(
        modeling,
        "_proxy_validation_split",
        lambda *args, **kwargs: (dataset.iloc[:8].copy(), dataset.iloc[8:].copy(), "tail_holdout_split"),
    )
    monkeypatch.setattr(
        modeling,
        "_proxy_candidate_builders",
        lambda: {
            "weak": lambda: "weak",
            "strong": lambda: "strong",
        },
    )
    monkeypatch.setattr(modeling, "_fit_proxy_candidate_model", lambda train_df, builder: builder())

    def fake_evaluate(model, validation_df):
        if model == "strong":
            return {"objective": 3.5, "roc_auc": 0.64, "precision_gate_precision": 0.78}
        return {"objective": 1.2, "roc_auc": 0.55, "precision_gate_precision": 0.61}

    monkeypatch.setattr(modeling, "_evaluate_proxy_candidate_model", fake_evaluate)

    candidate_name, metrics, split_label = modeling._select_market_proxy_candidate(
        dataset,
        train_start=GLOBAL_MODEL_TRAIN_START,
        train_end=GLOBAL_MODEL_TRAIN_END,
        test_start=GLOBAL_MODEL_TEST_START,
        test_end=GLOBAL_MODEL_TEST_END,
    )

    assert candidate_name == "strong"
    assert split_label == "tail_holdout_split"
    assert metrics["roc_auc"] == 0.64
    assert metrics["proxy_validation_objective"] == 3.5


def test_score_with_market_proxy_model_uses_validation_metrics(monkeypatch):
    latest = pd.DataFrame([{column: 0.1 for column in MODEL_FEATURE_COLUMNS}])

    class DummyProxyEstimator:
        def predict_proba(self, frame):
            return np.array([[0.18, 0.82]], dtype=float)

    monkeypatch.setattr(modeling, "_latest_prediction_feature_frame", lambda *args, **kwargs: latest.copy())

    proxy_model = MarketProxyModel(
        fitted_model=DummyProxyEstimator(),
        sample_size=6400,
        positive_rate=0.26,
        source_label="partial_market_dataset:extra_calibrated",
        validation_metrics={
            "precision_target": 0.90,
            "precision_gate_threshold": 0.80,
            "precision_gate_precision": 0.84,
            "precision_gate_support": 45.0,
            "precision_gate_return": 0.013,
            "precision_target_reached": 0.0,
            "roc_auc": 0.629,
            "brier": 0.223,
            "top_bucket_return": 0.016,
            "top_bucket_win_rate": 0.66,
            "spread_return": 0.029,
            "sample_size": 1200.0,
            "positive_rate": 0.23,
        },
        validation_summary="validated proxy summary",
        candidate_name="extra_calibrated",
    )

    result = modeling.score_with_market_proxy_model(pd.DataFrame(), proxy_model)

    assert result.model_name == "market_proxy_extra_calibrated"
    assert result.metrics["roc_auc"] == 0.629
    assert result.precision_gate_threshold == 0.80
    assert result.precision_gate_precision == 0.84
    assert result.precision_gate_support == 45
    assert result.precision_gate_active is False
    assert result.precision_gate_label.startswith("高精度观察")
    assert result.predicted_upside_pct > 0
    assert result.predicted_upside_low_pct <= result.predicted_upside_pct
    assert result.predicted_upside_high_pct >= result.predicted_upside_pct
    assert result.backtest_summary == "validated proxy summary"


def test_train_market_wide_model_uses_fixed_train_and_test_windows(monkeypatch, tmp_path):
    train_dates = pd.bdate_range(GLOBAL_MODEL_TRAIN_START, GLOBAL_MODEL_TRAIN_END, freq="B")[:80]
    test_dates = pd.bdate_range(GLOBAL_MODEL_TEST_START, GLOBAL_MODEL_TEST_END, freq="B")[:24]
    rows = []
    for symbol in ["000001", "000002"]:
        for idx, signal_date in enumerate([*train_dates, *test_dates]):
            target = 1 if idx % 4 in (2, 3) else 0
            feature_value = 2.5 if target else -2.0
            row = {column: feature_value for column in MODEL_FEATURE_COLUMNS}
            row["symbol"] = symbol
            row["name"] = f"Stock-{symbol}"
            row["signal_date"] = signal_date
            row["target"] = target
            row["future_return"] = 0.06 if target else -0.03
            rows.append(row)
    dataset = pd.DataFrame(rows)

    monkeypatch.setattr(modeling, "build_market_wide_dataset", lambda *args, **kwargs: dataset.copy())
    monkeypatch.setattr(
        modeling,
        "fetch_a_share_universe",
        lambda: pd.DataFrame({"symbol": ["000001", "000002", "000003"], "name": ["A", "B", "C"]}),
    )
    monkeypatch.setattr(
        modeling,
        "_global_model_cache_path",
        lambda *args, **kwargs: Path(tmp_path) / "market_model.pkl",
    )

    result = train_market_wide_model(refresh=True)

    assert result.train_start == GLOBAL_MODEL_TRAIN_START
    assert result.train_end == GLOBAL_MODEL_TRAIN_END
    assert result.test_start == GLOBAL_MODEL_TEST_START
    assert result.test_end == GLOBAL_MODEL_TEST_END
    assert result.train_sample_size == len(dataset[dataset["signal_date"].dt.year == 2025])
    assert result.test_sample_size == len(dataset[dataset["signal_date"].dt.year == 2026])
    assert result.universe_size == 3
    assert result.eligible_symbols == 2
    assert result.metrics["survivorship_bias_risk"] == 1.0
    assert len(result.ensemble_weights) == 3
    assert math.isclose(sum(result.ensemble_weights), 1.0, rel_tol=1e-9)
    assert "2025-01-01" in result.summary
    assert "2026-03-31" in result.summary
    assert "survivorship_bias_risk=True" in result.summary


def test_build_historical_market_universe_uses_all_status_stock_basic(monkeypatch):
    stock_basic = pd.DataFrame(
        [
            {"symbol": "600001", "name": "Active", "list_date": "20200101", "delist_date": "", "list_status": "L"},
            {"symbol": "600002", "name": "Future", "list_date": "20260501", "delist_date": "", "list_status": "L"},
            {"symbol": "600003", "name": "OldGone", "list_date": "20190101", "delist_date": "20240101", "list_status": "D"},
            {"symbol": "600004", "name": "WindowGone", "list_date": "20190101", "delist_date": "20260301", "list_status": "D"},
        ]
    )

    monkeypatch.setattr(modeling, "fetch_tushare_stock_basic_all_statuses", lambda: stock_basic)
    monkeypatch.setattr(
        modeling,
        "fetch_a_share_universe",
        lambda: (_ for _ in ()).throw(AssertionError("should not fallback to current universe")),
    )

    universe, point_in_time = modeling._build_historical_market_universe("2025-01-01", "2026-03-31")

    assert point_in_time is True
    assert set(universe["symbol"]) == {"600001", "600004"}


def test_train_market_wide_model_lowers_survivorship_risk_for_point_in_time_dataset(monkeypatch, tmp_path):
    train_dates = pd.bdate_range(GLOBAL_MODEL_TRAIN_START, GLOBAL_MODEL_TRAIN_END, freq="B")[:80]
    test_dates = pd.bdate_range(GLOBAL_MODEL_TEST_START, GLOBAL_MODEL_TEST_END, freq="B")[:24]
    rows = []
    for symbol in ["000001", "000002"]:
        for idx, signal_date in enumerate([*train_dates, *test_dates]):
            target = 1 if idx % 4 in (2, 3) else 0
            feature_value = 2.5 if target else -2.0
            row = {column: feature_value for column in MODEL_FEATURE_COLUMNS}
            row["symbol"] = symbol
            row["name"] = f"Stock-{symbol}"
            row["signal_date"] = signal_date
            row["target"] = target
            row["future_return"] = 0.06 if target else -0.03
            rows.append(row)
    dataset = pd.DataFrame(rows)
    dataset.attrs["point_in_time_universe"] = True
    dataset.attrs["source_universe_size"] = 4

    monkeypatch.setattr(modeling, "build_market_wide_dataset", lambda *args, **kwargs: dataset.copy())
    monkeypatch.setattr(
        modeling,
        "_global_model_cache_path",
        lambda *args, **kwargs: Path(tmp_path) / "market_model_point_in_time.pkl",
    )

    result = train_market_wide_model(refresh=True)

    assert result.universe_size == 4
    assert result.metrics["point_in_time_universe"] == 1.0
    assert result.metrics["survivorship_bias_risk"] < 1.0
    assert "survivorship_bias_risk=lowered" in result.summary


def test_external_snapshot_features_become_trainable_columns(monkeypatch, tmp_path):
    trade_dates = pd.bdate_range("2026-03-02", periods=6, freq="B")
    daily = pd.DataFrame(
        {
            "date": trade_dates,
            "symbol": ["000001"] * len(trade_dates),
            "open": np.linspace(10.0, 10.5, len(trade_dates)),
            "close": np.linspace(10.1, 10.8, len(trade_dates)),
            "high": np.linspace(10.3, 11.0, len(trade_dates)),
            "low": np.linspace(9.9, 10.4, len(trade_dates)),
            "volume": np.linspace(1_000_000, 1_400_000, len(trade_dates)),
            "amount": np.linspace(1.0e8, 1.4e8, len(trade_dates)),
            "turnover": np.linspace(2.0, 3.2, len(trade_dates)),
        }
    ).set_index("date", drop=False)

    news_df = pd.DataFrame(
        {
            modeling.NEWS_TITLE_KEYS[0]: ["bull buyback", "bull order", "bear warning"],
            modeling.NEWS_BODY_KEYS[0]: ["bull expansion plan", "bull pricing power", "bear selling plan"],
            modeling.NEWS_TIME_KEYS[0]: [
                pd.Timestamp("2026-03-02 09:30"),
                pd.Timestamp("2026-03-04 12:00"),
                pd.Timestamp("2026-03-05 19:20"),
            ],
            modeling.NEWS_SOURCE_KEYS[0]: ["证券时报", "财联社", "东方财富"],
        }
    )
    fund_df = pd.DataFrame(
        {
            "date": trade_dates,
            modeling.FUND_RATIO_KEYS[0]: [3.2, 2.4, 1.8, -0.6, 2.1, 3.4],
            modeling.FUND_NET_KEYS[0]: [2.6e8, 2.1e8, 1.4e8, -0.8e8, 1.8e8, 2.9e8],
        }
    )

    monkeypatch.setattr(modeling, "_external_snapshot_cache_path", lambda symbol: Path(tmp_path) / f"{symbol}.pkl")
    monkeypatch.setattr(modeling, "fetch_stock_news", lambda symbol, limit=120: news_df.copy())
    monkeypatch.setattr(modeling, "fetch_stock_main_fund_flow", lambda symbol, limit=160: fund_df.copy())
    monkeypatch.setattr(modeling, "BULLISH_KEYWORDS", {"bull": 2.0, "buyback": 2.5, "order": 1.6})
    monkeypatch.setattr(modeling, "BEARISH_KEYWORDS", {"bear": -2.0, "warning": -1.2})

    snapshot = modeling._external_snapshot_feature_frame(daily, symbol="000001")

    assert set(modeling.EXTERNAL_SNAPSHOT_COLUMNS).issubset(snapshot.columns)
    assert float(snapshot["news_sentiment_7d"].iloc[-1]) != 0.0
    assert float(snapshot["fund_ratio_5d"].iloc[-1]) > 0.0
    assert float(snapshot["fund_positive_ratio_5d"].iloc[-1]) > 0.0


def test_market_regime_features_create_one_hot_and_schema_version():
    frame = pd.DataFrame(
        {
            "market_ret_5": [0.03, 0.04, 0.002, -0.03],
            "market_ret_20": [0.09, -0.03, 0.004, -0.08],
            "market_close_vs_ma20": [0.05, -0.01, 0.0, -0.06],
            "market_volatility_10": [0.01, 0.012, 0.008, 0.022],
            "market_range_position_20": [0.82, 0.42, 0.50, 0.24],
        }
    )

    enriched = modeling._append_market_regime_features(frame)

    assert modeling.MODEL_SCHEMA_VERSION >= 2
    assert enriched["market_regime_trend"].iloc[0] == 1.0
    assert enriched["market_regime_defense"].iloc[-1] == 1.0
    assert np.allclose(
        enriched[
            [
                "market_regime_trend",
                "market_regime_rebound",
                "market_regime_rotation",
                "market_regime_defense",
            ]
        ].sum(axis=1),
        1.0,
    )
    assert enriched["market_regime_score"].between(0.0, 1.0).all()
    assert enriched["market_regime_risk"].between(0.0, 1.0).all()


def test_explain_latest_model_state_summarizes_regime_and_snapshots(monkeypatch, tmp_path):
    trade_dates = pd.bdate_range("2026-03-02", periods=6, freq="B")
    daily = pd.DataFrame(
        {
            "date": trade_dates,
            "symbol": ["000001"] * len(trade_dates),
            "open": np.linspace(10.0, 10.8, len(trade_dates)),
            "close": np.linspace(10.2, 11.3, len(trade_dates)),
            "high": np.linspace(10.3, 11.5, len(trade_dates)),
            "low": np.linspace(9.9, 10.6, len(trade_dates)),
            "volume": np.linspace(1_000_000, 1_500_000, len(trade_dates)),
            "amount": np.linspace(1.0e8, 1.6e8, len(trade_dates)),
            "turnover": np.linspace(2.0, 3.6, len(trade_dates)),
        }
    ).set_index("date", drop=False)

    monkeypatch.setattr(modeling, "_external_snapshot_cache_path", lambda symbol: Path(tmp_path) / f"{symbol}.pkl")
    monkeypatch.setattr(
        modeling,
        "fetch_stock_news",
        lambda symbol, limit=120: pd.DataFrame(
            {
                modeling.NEWS_TITLE_KEYS[0]: ["bull order", "bull expansion"],
                modeling.NEWS_BODY_KEYS[0]: ["bull catalyst", "bull follow through"],
                modeling.NEWS_TIME_KEYS[0]: [pd.Timestamp("2026-03-04 09:30"), pd.Timestamp("2026-03-05 09:30")],
                modeling.NEWS_SOURCE_KEYS[0]: ["证券时报", "财联社"],
            }
        ),
    )
    monkeypatch.setattr(
        modeling,
        "fetch_stock_main_fund_flow",
        lambda symbol, limit=160: pd.DataFrame(
            {
                "date": trade_dates,
                modeling.FUND_RATIO_KEYS[0]: [1.2, 1.8, 2.4, 2.0, 2.6, 3.0],
                modeling.FUND_NET_KEYS[0]: [1.2e8, 1.6e8, 1.9e8, 1.8e8, 2.2e8, 2.5e8],
            }
        ),
    )
    monkeypatch.setattr(modeling, "BULLISH_KEYWORDS", {"bull": 2.0, "order": 1.5, "expansion": 1.2})
    monkeypatch.setattr(modeling, "BEARISH_KEYWORDS", {"bear": -2.0})
    monkeypatch.setattr(
        modeling,
        "_build_market_feature_frame",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "market_ret_5": [0.01, 0.015, 0.018, 0.02, 0.022, 0.024],
                "market_ret_20": [0.03, 0.04, 0.05, 0.06, 0.065, 0.07],
                "market_close_vs_ma20": [0.01, 0.015, 0.02, 0.025, 0.03, 0.035],
                "market_volatility_10": [0.01, 0.01, 0.011, 0.011, 0.012, 0.012],
                "market_range_position_20": [0.62, 0.66, 0.70, 0.74, 0.78, 0.82],
            },
            index=trade_dates,
        ),
    )

    state = modeling.explain_latest_model_state(daily, symbol="000001")

    assert state["regime_code"] in modeling.REGIME_LABELS
    assert state["regime_label"]
    assert isinstance(state["state_reason_lines"], list)
    assert len(state["state_reason_lines"]) >= 3
    assert state["news_snapshot_score"] > 50
    assert state["fund_snapshot_score"] > 50
