import pandas as pd

from a_share_predictor.strategy import (
    assess_execution_readiness,
    assess_launch_window,
    build_strategy_workbench,
    build_trading_rule_context,
    evaluate_intraday_structure_signal,
    evaluate_temporal_news_pulse,
)


def test_build_trading_rule_context_detects_growth_board_limit():
    context = build_trading_rule_context("300750", "宁德时代")

    assert context.board_label == "创业板"
    assert context.price_limit_pct == 20.0
    assert "T+1" in context.t_plus_one_label


def test_build_trading_rule_context_detects_st_limit():
    context = build_trading_rule_context("600001", "*ST样例")

    assert context.price_limit_pct == 5.0
    assert "5%" in context.price_limit_label


def test_evaluate_temporal_news_pulse_splits_intraday_and_overnight():
    news = pd.DataFrame(
        {
            "发布时间": pd.to_datetime(
                [
                    "2026-03-12 10:05:00",
                    "2026-03-12 20:20:00",
                ]
            ),
            "新闻标题": ["公司中标新订单", "股东减持计划披露"],
            "新闻内容": ["中标并签约大单", "存在减持风险提示"],
        }
    )

    pulse = evaluate_temporal_news_pulse(news)

    assert pulse.intraday_score > pulse.overnight_score
    assert pulse.next_session_score != 50.0
    assert pulse.stronger_window


def test_evaluate_intraday_structure_signal_reads_opening_flow():
    minute = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-03-12 09:31:00", periods=120, freq="min"),
            "open": [10.0] * 120,
            "close": [10 + i * 0.003 for i in range(120)],
            "volume": [3000] * 30 + [800] * 90,
        }
    )

    signal = evaluate_intraday_structure_signal(minute)

    assert signal.opening_volume_ratio > 0
    assert signal.first30_volume_share > 0
    assert signal.label


def test_build_strategy_workbench_prefers_trend_confirmation():
    rule_context = build_trading_rule_context("688981", "中芯国际")
    pulse = evaluate_temporal_news_pulse(
        pd.DataFrame(
            {
                "发布时间": pd.to_datetime(["2026-03-12 20:20:00"]),
                "新闻标题": ["公司回购并签约新订单"],
                "新闻内容": ["回购与订单共振"],
            }
        )
    )
    intraday_signal = evaluate_intraday_structure_signal(
        pd.DataFrame(
            {
                "datetime": pd.date_range("2026-03-12 09:31:00", periods=120, freq="min"),
                "open": [20.0] * 120,
                "close": [20 + i * 0.01 for i in range(120)],
                "volume": [2500] * 30 + [900] * 90,
            }
        )
    )

    workbench = build_strategy_workbench(
        stage_code="breakout_confirmation",
        probability_up=0.68,
        quant_score=76.0,
        sector_score=71.0,
        temporal_pulse=pulse,
        intraday_signal=intraday_signal,
        rule_context=rule_context,
    )

    assert workbench.style == "顺势确认"
    assert workbench.strategy_score > 60
    assert "均价线" in workbench.entry_window

def test_assess_launch_window_rewards_early_launch_alignment():
    strong = assess_launch_window(
        stage_code="breakout_confirmation",
        stage_label="主升启动",
        probability_up=74.0,
        predicted_upside_pct=12.0,
        quant_score=76.0,
        sector_score=72.0,
        fund_score=74.0,
        news_score=66.0,
        launch_score=70.0,
        launch_readiness_score=78.0,
        market_resonance_score=73.0,
        launch_specialist_score=82.0,
        launch_regime_fit_score=79.0,
        launch_specialist_confidence=75.0,
        close_vs_ma20=0.04,
        breakout_distance=0.01,
        intraday_bias=2,
    )
    weak = assess_launch_window(
        stage_code="distribution_risk",
        stage_label="高位分歧",
        probability_up=38.0,
        predicted_upside_pct=3.0,
        quant_score=41.0,
        sector_score=43.0,
        fund_score=39.0,
        news_score=45.0,
        launch_score=44.0,
        launch_readiness_score=36.0,
        market_resonance_score=40.0,
        launch_specialist_score=34.0,
        launch_regime_fit_score=37.0,
        launch_specialist_confidence=42.0,
        close_vs_ma20=-0.05,
        breakout_distance=0.09,
        intraday_bias=-2,
    )

    assert strong.window_score > weak.window_score
    assert strong.window_confidence > weak.window_confidence
    assert strong.status in {"黄金启动窗", "启动观察窗"}
    assert weak.status == "高位风险窗"


def test_assess_execution_readiness_separates_quality_from_timing():
    strong = assess_execution_readiness(
        stage_code="breakout_confirmation",
        stage_label="主升启动",
        probability_up=74.0,
        predicted_upside_pct=11.0,
        quant_score=73.0,
        launch_window_score=80.0,
        launch_window_status="黄金启动窗",
        launch_window_confidence=76.0,
        sector_score=69.0,
        fund_score=72.0,
        news_score=64.0,
        close_vs_ma20=0.03,
        breakout_distance=0.01,
        intraday_bias=2,
    )
    weak = assess_execution_readiness(
        stage_code="distribution_risk",
        stage_label="高位分歧",
        probability_up=42.0,
        predicted_upside_pct=2.5,
        quant_score=41.0,
        launch_window_score=38.0,
        launch_window_status="高位风险窗",
        launch_window_confidence=44.0,
        sector_score=43.0,
        fund_score=40.0,
        news_score=46.0,
        close_vs_ma20=-0.05,
        breakout_distance=0.08,
        intraday_bias=-2,
    )

    assert strong.execution_score > weak.execution_score
    assert strong.reward_risk_ratio > weak.reward_risk_ratio
    assert strong.label == "可执行"
    assert weak.label in {"暂不执行", "等待结构"}
