import pandas as pd

from a_share_predictor.features import build_daily_features, latest_snapshot
from a_share_predictor.stages import build_tomorrow_plan, classify_stage, main_rise_start_score, stage_numeric_score


def make_trend_acceleration_daily() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=180, freq="B")
    ramp = pd.Series(range(180), index=dates, dtype=float)
    close = 8 + ramp * 0.12
    daily = pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.06,
            "close": close,
            "high": close + 0.15,
            "low": close - 0.15,
            "volume": 150000 + ramp * 1500,
            "amount": (150000 + ramp * 1500) * close,
            "turnover": 3.0 + ramp * 0.01,
        },
        index=dates,
    )
    return daily


def make_main_rise_start_daily() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=180, freq="B")
    close_values: list[float] = []
    for index in range(180):
        if index < 100:
            close = 9.8 + index * 0.009
        elif index < 150:
            cycle = (index - 100) % 8
            close = 10.72 + (cycle - 3.5) * 0.018
        elif index < 170:
            close = 10.98 + (index - 150) * 0.038
        else:
            tail = [11.72, 11.77, 11.82, 11.79, 11.84, 11.80, 11.86, 11.83, 11.88, 11.85]
            close = tail[index - 170]
        close_values.append(round(close, 3))
    close = pd.Series(close_values, index=dates, dtype=float)
    volume = pd.Series([120000 + (idx % 7) * 1800 for idx in range(180)], index=dates, dtype=float)
    volume.iloc[-8:] = [138000, 141000, 145000, 139000, 146000, 140000, 147000, 142000]
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.995,
            "close": close,
            "high": close * 1.012,
            "low": close * 0.988,
            "volume": volume,
            "amount": volume * close,
            "turnover": 2.2 + pd.Series(range(180), index=dates, dtype=float) * 0.004,
        },
        index=dates,
    )


def test_classify_stage_detects_trend_acceleration():
    daily = make_trend_acceleration_daily()
    stage = classify_stage(daily)
    assert stage.code == "trend_acceleration"
    assert stage.label
    assert stage.description
    assert len(stage.rationale) >= 3
    assert len(stage.focus_points) >= 3


def test_classify_stage_detects_main_rise_start():
    daily = make_main_rise_start_daily()
    stage = classify_stage(daily)
    features = build_daily_features(daily).dropna()

    assert stage.code == "main_rise_start"
    assert stage.label
    assert main_rise_start_score(features.iloc[-1]) >= 72


def test_build_tomorrow_plan_for_trend_stage():
    daily = make_trend_acceleration_daily()
    features = build_daily_features(daily)
    valid_features = features.dropna()
    stage = classify_stage(daily)
    latest_features = valid_features.iloc[-1]
    snapshot = latest_snapshot(daily, features)

    plan = build_tomorrow_plan(
        stage,
        snapshot=snapshot,
        latest_features=latest_features,
        probability_up=0.72,
        quant_score=78.0,
    )

    assert plan.setup_label == "强更强跟随"
    assert "买点" not in plan.setup_label
    assert "均价线" in plan.buy_point
    assert "减仓" in plan.sell_point
    assert plan.confidence > 60
    assert "MA20" in plan.buy_point


def test_build_tomorrow_plan_for_main_rise_start_stage():
    daily = make_main_rise_start_daily()
    features = build_daily_features(daily)
    valid_features = features.dropna()
    stage = classify_stage(daily)
    latest_features = valid_features.iloc[-1]
    snapshot = latest_snapshot(daily, features)

    plan = build_tomorrow_plan(
        stage,
        snapshot=snapshot,
        latest_features=latest_features,
        probability_up=0.68,
        quant_score=74.0,
    )

    assert plan.setup_label == "主升初启右侧"
    assert "突破位" in plan.buy_point
    assert "启动失败" in plan.sell_point
    assert plan.confidence > 60


def test_build_tomorrow_plan_appends_intraday_execution_clause():
    daily = make_trend_acceleration_daily()
    features = build_daily_features(daily)
    valid_features = features.dropna()
    stage = classify_stage(daily)
    latest_features = valid_features.iloc[-1]
    snapshot = latest_snapshot(daily, features)

    plan = build_tomorrow_plan(
        stage,
        snapshot=snapshot,
        latest_features=latest_features,
        probability_up=0.61,
        quant_score=72.0,
        intraday_state={
            "label": "分时偏弱/博弈",
            "score": 0.42,
            "above_avg_ratio": 0.31,
            "max_pullback": 0.052,
        },
        intraday_signal={
            "label": "开盘确认不足",
            "first30_volume_share": 0.12,
            "early_return_pct": -0.011,
        },
    )

    assert "分时条件" in plan.buy_point
    assert "均价线" in plan.buy_point
    assert "5到10 分钟收不回" in plan.sell_point


def test_stage_numeric_score_penalizes_distribution_risk():
    daily = make_trend_acceleration_daily()
    features = build_daily_features(daily).dropna()
    trend_stage = classify_stage(daily)
    trend_score = stage_numeric_score(trend_stage, features.iloc[-1])

    risk_stage = trend_stage.__class__(
        code="distribution_risk",
        label="高位分歧派发",
        description=trend_stage.description,
        structure_summary=trend_stage.structure_summary,
        intraday_expectation=trend_stage.intraday_expectation,
        priority=trend_stage.priority,
        focus_points=trend_stage.focus_points,
        invalidation=trend_stage.invalidation,
        rationale=trend_stage.rationale,
        risk_flags=trend_stage.risk_flags,
    )
    risk_score = stage_numeric_score(risk_stage, features.iloc[-1])

    assert trend_score > risk_score
