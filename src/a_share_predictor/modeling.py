from __future__ import annotations

import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import (
    fetch_a_share_universe,
    fetch_daily_history,
    fetch_index_daily_history,
    fetch_stock_main_fund_flow,
    fetch_stock_news,
    fetch_tushare_stock_basic_all_statuses,
    filter_historical_a_share_universe_window,
    normalize_symbol,
)
from .features import FEATURE_COLUMNS, build_daily_features, build_training_frame, evaluate_intraday
from .quant import (
    BEARISH_KEYWORDS,
    BULLISH_KEYWORDS,
    FUND_NET_KEYS,
    FUND_RATIO_KEYS,
    NEWS_BODY_KEYS,
    NEWS_SOURCE_KEYS,
    NEWS_TIME_KEYS,
    NEWS_TITLE_KEYS,
    SOURCE_WEIGHTS,
    evaluate_main_fund_signal,
    evaluate_news_sentiment,
)
from .news_impact import build_research_enhanced_news_signal, classify_news_events, score_news_event_with_research
from .strategy import evaluate_intraday_structure_signal, evaluate_temporal_news_pulse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / ".cache"
MODEL_SCHEMA_VERSION = 5
EXTERNAL_SNAPSHOT_CACHE_VERSION = 2
GLOBAL_MODEL_TRAIN_START = "2025-01-01"
GLOBAL_MODEL_TRAIN_END = "2025-12-31"
GLOBAL_MODEL_TEST_START = "2026-01-01"
GLOBAL_MODEL_TEST_END = "2026-03-31"
GLOBAL_MODEL_HISTORY_START = "2024-09-01"
RECENT_BACKTEST_WINDOW = 30
RECENT_BACKTEST_MIN_SAMPLES = 12
GLOBAL_MODEL_HISTORY_END = "2026-04-10"
MARKET_PROXY_SAMPLE_SIZE = 120_000
# AkShare relies on py_mini_racer for some endpoints, which can hard-crash
# the interpreter when initialized concurrently across many threads.
# Building the market-wide dataset is a heavy one-time cache fill, so we
# prefer a stable serial fetch here over an occasionally faster but brittle run.
GLOBAL_MODEL_MAX_WORKERS = 1
GLOBAL_MODEL_CHECKPOINT_EVERY = 40
LINEAR_MODEL_WEIGHT = 0.40
FOREST_MODEL_WEIGHT = 0.34
BOOST_MODEL_WEIGHT = 0.26
ENSEMBLE_WEIGHTS = np.array(
    [
        LINEAR_MODEL_WEIGHT,
        FOREST_MODEL_WEIGHT,
        BOOST_MODEL_WEIGHT,
    ],
    dtype=float,
)
COMPONENT_MODEL_NAMES = ("logistic", "forest", "boost")

DERIVED_FEATURE_COLUMNS = [
    "trend_strength",
    "breakout_readiness",
    "pullback_quality",
    "volume_thrust",
    "risk_pressure",
    "stretch_risk",
    "launch_readiness",
    "market_resonance",
]
MARKET_FEATURE_COLUMNS = [
    "market_ret_5",
    "market_ret_20",
    "market_close_vs_ma20",
    "market_volatility_10",
    "market_range_position_20",
    "relative_strength_5",
    "relative_strength_20",
]
EXTERNAL_SNAPSHOT_COLUMNS = [
    "news_sentiment_3d",
    "news_sentiment_7d",
    "news_confidence_7d",
    "news_volume_3d",
    "news_volume_7d",
    "news_event_shock_3d",
    "news_positive_ratio_7d",
    "news_research_score_3d",
    "news_research_score_7d",
    "news_research_confidence_7d",
    "news_research_excess_1d",
    "news_research_excess_5d",
    "fund_ratio_1d",
    "fund_ratio_5d",
    "fund_net_strength_1d",
    "fund_net_strength_5d",
    "fund_positive_ratio_5d",
    "fund_inflow_streak_5d",
    "fund_trend_delta_5d",
    "fund_consistency_5d",
]
REGIME_LABELS = ("trend", "rebound", "rotation", "defense")
REGIME_DISPLAY_LABELS = {
    "trend": "趋势主导",
    "rebound": "修复反弹",
    "rotation": "轮动震荡",
    "defense": "防守承压",
}
REGIME_FEATURE_COLUMNS = [
    "market_regime_trend",
    "market_regime_rebound",
    "market_regime_rotation",
    "market_regime_defense",
    "market_regime_score",
    "market_regime_risk",
]
MODEL_FEATURE_COLUMNS = [
    *FEATURE_COLUMNS,
    *DERIVED_FEATURE_COLUMNS,
    *MARKET_FEATURE_COLUMNS,
    *EXTERNAL_SNAPSHOT_COLUMNS,
    *REGIME_FEATURE_COLUMNS,
]
META_CALIBRATION_COLUMNS = [
    "trend_strength",
    "breakout_readiness",
    "volume_thrust",
    "risk_pressure",
    "stretch_risk",
    "launch_readiness",
    "market_resonance",
    "ret_120",
    "close_vs_ma120",
    "drawdown_20",
    "efficiency_ratio_10",
    "downside_vol_ratio_20",
    "market_ret_20",
    "market_volatility_10",
    "relative_strength_20",
    "news_sentiment_7d",
    "news_confidence_7d",
    "news_research_score_7d",
    "news_research_confidence_7d",
    "news_research_excess_1d",
    "fund_ratio_5d",
    "fund_positive_ratio_5d",
    "fund_consistency_5d",
    "market_regime_score",
    "market_regime_risk",
]


@dataclass(slots=True)
class ProbabilityResult:
    latest_probability: float
    probabilities: pd.Series
    metrics: dict[str, float]
    coefficients: list[tuple[str, float]]
    out_of_sample_probabilities: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    strategy_score: float = 50.0
    agreement_score: float = 50.0
    signal_label: str = "样本不足"
    risk_label: str = "等待更多数据"
    model_name: str = "本地量化集成"
    quality_label: str = "观望"
    backtest_summary: str = "样本不足，先以阶段结构和分时确认为主。"
    signal_breakdown: dict[str, float] = field(default_factory=dict)
    regime_label: str = "rotation"
    precision_target: float = 0.90
    precision_gate_threshold: float = 1.0
    precision_gate_precision: float = 0.0
    precision_gate_support: int = 0
    precision_gate_active: bool = False
    precision_gate_label: str = "未达90%精度门槛"
    raw_probability: float = 0.0
    enhanced_probability: float = 0.0
    base_probability: float = 0.0
    upgrade_delta: float = 0.0
    upgrade_components: dict[str, float] = field(default_factory=dict)
    upgrade_summary: str = ""
    predicted_upside_pct: float = 0.0
    predicted_upside_low_pct: float = 0.0
    predicted_upside_high_pct: float = 0.0


@dataclass(slots=True)
class MarketWideModel:
    horizon_days: int
    positive_return: float
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    fitted_models: tuple[Pipeline, Pipeline, Pipeline]
    ensemble_weights: tuple[float, float, float]
    calibrator: Pipeline | None
    metrics: dict[str, float]
    coefficients: list[tuple[str, float]]
    train_sample_size: int
    test_sample_size: int
    universe_size: int
    eligible_symbols: int
    quality_label: str
    summary: str
    schema_version: int = MODEL_SCHEMA_VERSION
    regime_calibrators: dict[str, Pipeline] = field(default_factory=dict)
    regime_distribution: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class MarketProxyModel:
    fitted_model: Pipeline
    sample_size: int
    positive_rate: float
    source_label: str
    validation_metrics: dict[str, float] = field(default_factory=dict)
    validation_summary: str = ""
    candidate_name: str = "linear"


def _clip(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return float(max(lower, min(value, upper)))


def _safe_roc_auc_score(y_true, y_score) -> float:
    classes = np.unique(np.asarray(y_true))
    if len(classes) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_brier_score(y_true, y_prob) -> float:
    classes = np.unique(np.asarray(y_true))
    if len(classes) < 2:
        return float("nan")
    return float(brier_score_loss(y_true, y_prob))


def _numeric_series(feature_frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if not isinstance(feature_frame, pd.DataFrame) or feature_frame.empty:
        return pd.Series(dtype=float)
    if column not in feature_frame.columns:
        return pd.Series(float(default), index=feature_frame.index, dtype=float)
    return pd.to_numeric(feature_frame[column], errors="coerce").astype(float).fillna(float(default))


def _clip_probability_array(values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    return np.clip(vector, 1e-4, 1 - 1e-4)


def _launch_readiness_series(feature_frame: pd.DataFrame) -> pd.Series:
    if not isinstance(feature_frame, pd.DataFrame) or feature_frame.empty:
        return pd.Series(dtype=float)

    breakout_distance_20 = _numeric_series(feature_frame, "breakout_distance_20")
    breakout_distance_60 = _numeric_series(feature_frame, "breakout_distance_60")
    close_vs_ma20 = _numeric_series(feature_frame, "close_vs_ma20")
    range_position_20 = _numeric_series(feature_frame, "range_position_20", 0.5)
    consolidation_width_20 = _numeric_series(feature_frame, "consolidation_width_20", 0.25)
    pullback_to_breakout_20 = _numeric_series(feature_frame, "pullback_to_breakout_20")
    volume_ratio_5 = _numeric_series(feature_frame, "volume_ratio_5", 1.0)
    ma_alignment_score = _numeric_series(feature_frame, "ma_alignment_score", 0.5)
    momentum_persistence_10 = _numeric_series(feature_frame, "momentum_persistence_10", 0.5)
    stretch_risk = _numeric_series(feature_frame, "stretch_risk")
    risk_pressure = _numeric_series(feature_frame, "risk_pressure")

    return (
        50
        + breakout_distance_20 * 360
        + breakout_distance_60 * 180
        + close_vs_ma20 * 140
        + (range_position_20 - 0.58) * 56
        + np.maximum(0.30 - consolidation_width_20, 0.0) * 92
        - pullback_to_breakout_20.abs() * 180
        + (volume_ratio_5 - 1.0) * 16
        + (ma_alignment_score - 0.5) * 28
        + (momentum_persistence_10 - 0.5) * 24
        - stretch_risk * 0.40
        - risk_pressure * 0.035
    ).clip(lower=0, upper=100)


def _append_market_resonance_features(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame.copy()

    df = frame.copy()
    relative_strength_5 = _numeric_series(df, "relative_strength_5")
    relative_strength_20 = _numeric_series(df, "relative_strength_20")
    market_ret_5 = _numeric_series(df, "market_ret_5")
    market_ret_20 = _numeric_series(df, "market_ret_20")
    market_close_vs_ma20 = _numeric_series(df, "market_close_vs_ma20")
    market_regime_score = _numeric_series(df, "market_regime_score")
    market_regime_risk = _numeric_series(df, "market_regime_risk")
    news_sentiment_7d = _numeric_series(df, "news_sentiment_7d")
    news_confidence_7d = _numeric_series(df, "news_confidence_7d")
    news_positive_ratio_7d = _numeric_series(df, "news_positive_ratio_7d", 0.5)
    fund_ratio_5d = _numeric_series(df, "fund_ratio_5d")
    fund_net_strength_5d = _numeric_series(df, "fund_net_strength_5d")
    fund_positive_ratio_5d = _numeric_series(df, "fund_positive_ratio_5d", 0.5)
    fund_inflow_streak_5d = _numeric_series(df, "fund_inflow_streak_5d")
    fund_consistency_5d = _numeric_series(df, "fund_consistency_5d")

    df["market_resonance"] = (
        50
        + relative_strength_5 * 220
        + relative_strength_20 * 180
        + market_ret_20 * 96
        + market_ret_5 * 64
        + market_close_vs_ma20 * 128
        + market_regime_score * 18
        - market_regime_risk * 15
        + news_sentiment_7d * 20
        + news_confidence_7d * 8
        + (news_positive_ratio_7d - 0.5) * 20
        + fund_ratio_5d * 18
        + fund_net_strength_5d * 15
        + (fund_positive_ratio_5d - 0.5) * 16
        + fund_inflow_streak_5d * 10
        + fund_consistency_5d * 8
    ).clip(lower=0, upper=100)
    return df.replace([np.inf, -np.inf], np.nan)


def _build_launch_specialist_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(feature_frame, pd.DataFrame) or feature_frame.empty:
        return pd.DataFrame()

    launch_readiness = _numeric_series(feature_frame, "launch_readiness", 50.0)
    market_resonance = _numeric_series(feature_frame, "market_resonance", 50.0)
    trend_strength = _numeric_series(feature_frame, "trend_strength")
    breakout_readiness = _numeric_series(feature_frame, "breakout_readiness")
    pullback_quality = _numeric_series(feature_frame, "pullback_quality")
    volume_thrust = _numeric_series(feature_frame, "volume_thrust")
    risk_pressure = _numeric_series(feature_frame, "risk_pressure")
    stretch_risk = _numeric_series(feature_frame, "stretch_risk")
    market_regime_score = _numeric_series(feature_frame, "market_regime_score")
    market_regime_risk = _numeric_series(feature_frame, "market_regime_risk")
    relative_strength_20 = _numeric_series(feature_frame, "relative_strength_20")
    close_vs_ma20 = _numeric_series(feature_frame, "close_vs_ma20")
    breakout_distance_20 = _numeric_series(feature_frame, "breakout_distance_20")
    range_position_20 = _numeric_series(feature_frame, "range_position_20", 0.5)

    regime_labels = pd.Series(_regime_labels_from_feature_frame(feature_frame), index=feature_frame.index, dtype=str)
    trend_fit = (
        52
        + (trend_strength / 4.5)
        + (breakout_readiness / 5.8)
        + (market_resonance - 50.0) * 0.28
        + np.maximum(volume_thrust, 0.0) * 0.22
        - risk_pressure * 0.09
        - stretch_risk * 0.36
    )
    rebound_fit = (
        50
        + (launch_readiness - 50.0) * 0.40
        + (pullback_quality / 4.6)
        + (market_resonance - 50.0) * 0.18
        + np.maximum(close_vs_ma20, 0.0) * 220
        - stretch_risk * 0.30
        - market_regime_risk * 14.0
    )
    rotation_fit = (
        49
        + (launch_readiness - 50.0) * 0.34
        + (market_resonance - 50.0) * 0.24
        + relative_strength_20 * 180
        + (range_position_20 - 0.58) * 30
        - risk_pressure * 0.07
        - stretch_risk * 0.22
    )
    defense_fit = (
        42
        + (launch_readiness - 50.0) * 0.18
        + (market_resonance - 50.0) * 0.12
        + np.maximum(close_vs_ma20, 0.0) * 110
        - risk_pressure * 0.10
        - stretch_risk * 0.34
        - market_regime_risk * 18.0
    )
    regime_fit_score = np.select(
        [
            regime_labels.eq("trend"),
            regime_labels.eq("rebound"),
            regime_labels.eq("defense"),
        ],
        [trend_fit, rebound_fit, defense_fit],
        default=rotation_fit,
    )
    regime_tailwind = np.select(
        [
            regime_labels.eq("trend"),
            regime_labels.eq("rebound"),
            regime_labels.eq("defense"),
        ],
        [1.0, 0.55, -0.72],
        default=0.28,
    )
    launch_specialist_score = (
        launch_readiness * 0.36
        + market_resonance * 0.18
        + np.clip(regime_fit_score, 0.0, 100.0) * 0.20
        + np.clip(50.0 + breakout_readiness / 4.2, 0.0, 100.0) * 0.12
        + np.clip(50.0 + pullback_quality / 3.8, 0.0, 100.0) * 0.08
        + np.clip(50.0 + np.maximum(volume_thrust, -40.0) / 3.2, 0.0, 100.0) * 0.06
    )
    launch_specialist_score = np.clip(launch_specialist_score, 0.0, 100.0)
    specialist_confidence = np.clip(
        0.46
        + np.minimum(np.maximum(launch_readiness - 50.0, 0.0) / 100.0, 0.16)
        + np.minimum(np.maximum(market_resonance - 50.0, 0.0) / 100.0, 0.14)
        + np.minimum(np.maximum(market_regime_score, 0.0), 1.0) * 0.10
        + np.minimum(np.maximum(0.12 - np.abs(breakout_distance_20), 0.0) * 2.2, 0.08)
        - np.minimum(np.maximum(market_regime_risk - 0.45, 0.0), 0.35) * 0.22,
        0.40,
        0.92,
    )
    specialist_delta_pct = np.clip(
        np.tanh((launch_specialist_score - 50.0) / 18.0) * (2.0 + np.maximum(regime_tailwind, 0.0) * 1.5)
        * specialist_confidence
        + regime_tailwind * 0.7 * specialist_confidence,
        -3.8,
        4.2,
    )
    return pd.DataFrame(
        {
            "launch_specialist_score": launch_specialist_score,
            "launch_regime_fit_score": np.clip(regime_fit_score, 0.0, 100.0),
            "launch_specialist_confidence": specialist_confidence,
            "launch_specialist_delta_pct": specialist_delta_pct,
        },
        index=feature_frame.index,
    )


def _build_context_score_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(feature_frame, pd.DataFrame) or feature_frame.empty:
        return pd.DataFrame()

    ret_20 = _numeric_series(feature_frame, "ret_20")
    ret_60 = _numeric_series(feature_frame, "ret_60")
    close_vs_ma20 = _numeric_series(feature_frame, "close_vs_ma20")
    ma20_slope_5 = _numeric_series(feature_frame, "ma20_slope_5")
    breakout_distance_20 = _numeric_series(feature_frame, "breakout_distance_20")
    range_position_20 = _numeric_series(feature_frame, "range_position_20", 0.5)
    volume_ratio_5 = _numeric_series(feature_frame, "volume_ratio_5", 1.0)
    volume_ratio_20 = _numeric_series(feature_frame, "volume_ratio_20", 1.0)
    amount_ratio_5 = _numeric_series(feature_frame, "amount_ratio_5", 1.0)
    pullback_to_breakout_20 = _numeric_series(feature_frame, "pullback_to_breakout_20")
    lower_shadow_ratio = _numeric_series(feature_frame, "lower_shadow_ratio")
    upper_shadow_ratio = _numeric_series(feature_frame, "upper_shadow_ratio")
    body_ratio = _numeric_series(feature_frame, "body_ratio")
    volatility_10 = _numeric_series(feature_frame, "volatility_10")
    turnover_ratio_20 = _numeric_series(feature_frame, "turnover_ratio_20", 1.0)
    close_position_day = _numeric_series(feature_frame, "close_position_day", 0.5)
    volatility_contraction = _numeric_series(feature_frame, "volatility_contraction")
    stretch_risk = _numeric_series(feature_frame, "stretch_risk")
    ma_alignment_score = _numeric_series(feature_frame, "ma_alignment_score", 0.5)
    momentum_persistence_10 = _numeric_series(feature_frame, "momentum_persistence_10", 0.5)
    efficiency_ratio_10 = _numeric_series(feature_frame, "efficiency_ratio_10", 0.4)
    downside_vol_ratio_20 = _numeric_series(feature_frame, "downside_vol_ratio_20", 0.4)
    relative_strength_5 = _numeric_series(feature_frame, "relative_strength_5")
    relative_strength_20 = _numeric_series(feature_frame, "relative_strength_20")
    market_ret_5 = _numeric_series(feature_frame, "market_ret_5")
    market_ret_20 = _numeric_series(feature_frame, "market_ret_20")
    market_close_vs_ma20 = _numeric_series(feature_frame, "market_close_vs_ma20")
    market_volatility_10 = _numeric_series(feature_frame, "market_volatility_10")
    market_regime_score = _numeric_series(feature_frame, "market_regime_score")
    market_regime_risk = _numeric_series(feature_frame, "market_regime_risk")
    launch_readiness_score = _numeric_series(feature_frame, "launch_readiness", 50.0)
    market_resonance_score = _numeric_series(feature_frame, "market_resonance", 50.0)
    news_sentiment_7d = _numeric_series(feature_frame, "news_sentiment_7d")
    news_confidence_7d = _numeric_series(feature_frame, "news_confidence_7d")
    news_positive_ratio_7d = _numeric_series(feature_frame, "news_positive_ratio_7d", 0.5)
    fund_ratio_5d = _numeric_series(feature_frame, "fund_ratio_5d")
    fund_net_strength_5d = _numeric_series(feature_frame, "fund_net_strength_5d")
    fund_positive_ratio_5d = _numeric_series(feature_frame, "fund_positive_ratio_5d", 0.5)
    fund_inflow_streak_5d = _numeric_series(feature_frame, "fund_inflow_streak_5d")
    fund_consistency_5d = _numeric_series(feature_frame, "fund_consistency_5d")

    trend_score = (
        52
        + ret_20 * 220
        + ret_60 * 150
        + close_vs_ma20 * 240
        + ma20_slope_5 * 820
    ).clip(lower=0, upper=100)
    breakout_score = (
        50
        + breakout_distance_20 * 520
        + range_position_20 * 32
        + (volume_ratio_5 - 1.0) * 28
    ).clip(lower=0, upper=100)
    pullback_score = (
        48
        - pullback_to_breakout_20.abs() * 420
        + lower_shadow_ratio * 680
        + body_ratio * 24
    ).clip(lower=0, upper=100)
    risk_score = (
        26
        + upper_shadow_ratio * 1400
        + volatility_10 * 1900
        + np.maximum(close_vs_ma20 - 0.10, 0.0) * 360
    ).clip(lower=0, upper=100)

    daily_k_score = (
        trend_score * 0.36
        + breakout_score * 0.22
        + pullback_score * 0.14
        + (100 - risk_score) * 0.28
    ).clip(lower=0, upper=100)
    volume_price_score = (
        46
        + (volume_ratio_5 - 1.0) * 18
        + (volume_ratio_20 - 1.0) * 12
        + (amount_ratio_5 - 1.0) * 14
        + (turnover_ratio_20 - 1.0) * 10
        + (close_position_day - 0.5) * 20
        + lower_shadow_ratio * 180
        - upper_shadow_ratio * 150
        - np.maximum(volatility_contraction, 0.0) * 12
        + np.maximum(-volatility_contraction, 0.0) * 8
    ).clip(lower=0, upper=100)
    market_context_score = (
        50
        + market_ret_20 * 170
        + market_ret_5 * 120
        + market_close_vs_ma20 * 160
        + relative_strength_5 * 160
        + relative_strength_20 * 150
        + market_regime_score * 22
        - market_volatility_10 * 240
        - market_regime_risk * 18
    ).clip(lower=0, upper=100)
    news_fund_score = (
        50
        + news_sentiment_7d * 24
        + news_confidence_7d * 10
        + (news_positive_ratio_7d - 0.5) * 26
        + fund_ratio_5d * 20
        + fund_net_strength_5d * 16
        + (fund_positive_ratio_5d - 0.5) * 20
        + fund_inflow_streak_5d * 12
        + fund_consistency_5d * 10
    ).clip(lower=0, upper=100)
    quant_context_score = (
        48
        + (ma_alignment_score - 0.5) * 30
        + (momentum_persistence_10 - 0.5) * 26
        + (efficiency_ratio_10 - 0.4) * 24
        - downside_vol_ratio_20 * 12
        + relative_strength_5 * 180
        + relative_strength_20 * 150
    ).clip(lower=0, upper=100)
    context_composite_score = (
        daily_k_score * 0.28
        + volume_price_score * 0.20
        + market_context_score * 0.16
        + news_fund_score * 0.12
        + quant_context_score * 0.12
        + launch_readiness_score * 0.06
        + market_resonance_score * 0.06
    ).clip(lower=0, upper=100)
    context_confidence = (
        0.56
        + np.minimum(news_confidence_7d, 1.0) * 0.08
        + np.minimum(fund_consistency_5d, 1.0) * 0.08
        + np.minimum(market_regime_score, 1.0) * 0.07
        + np.minimum(np.abs(relative_strength_20) * 2.5, 0.08)
        + np.minimum(np.maximum(launch_readiness_score - 50.0, 0.0) / 100.0, 0.05)
        + np.minimum(np.maximum(market_resonance_score - 50.0, 0.0) / 100.0, 0.05)
    ).clip(lower=0.50, upper=0.92)
    breakout_quality = (
        breakout_score * 0.52
        + volume_price_score * 0.24
        + market_resonance_score * 0.14
        + np.clip(50.0 + relative_strength_5 * 180, 0.0, 100.0) * 0.10
    ).clip(lower=0, upper=100)
    resonance_quality = (
        market_context_score * 0.42
        + market_resonance_score * 0.30
        + news_fund_score * 0.14
        + quant_context_score * 0.14
    ).clip(lower=0, upper=100)
    risk_of_late_entry = (
        risk_score * 0.46
        + np.maximum(ret_20 - 0.22, 0.0) * 160
        + np.maximum(range_position_20 - 0.86, 0.0) * 70
        + np.maximum(volume_ratio_5 - 1.8, 0.0) * 18
        + stretch_risk * 0.30
        - np.maximum(launch_readiness_score - 55.0, 0.0) * 0.20
    ).clip(lower=0, upper=100)
    launch_phase_label = pd.Series(
        np.select(
            [
                risk_of_late_entry >= 68.0,
                (breakout_quality < 48.0) | (resonance_quality < 50.0),
                (launch_readiness_score >= 64.0) & (risk_of_late_entry < 56.0),
            ],
            ["已走远", "伪突破", "刚启动"],
            default="观察",
        ),
        index=feature_frame.index,
        dtype=str,
    )

    return pd.DataFrame(
        {
            "trend_score": trend_score,
            "breakout_score": breakout_score,
            "pullback_score": pullback_score,
            "risk_score": risk_score,
            "launch_readiness_score": launch_readiness_score,
            "market_resonance_score": market_resonance_score,
            "daily_k_score": daily_k_score,
            "volume_price_score": volume_price_score,
            "market_context_score": market_context_score,
            "news_fund_score": news_fund_score,
            "quant_context_score": quant_context_score,
            "context_composite_score": context_composite_score,
            "context_confidence": context_confidence,
            "launch_readiness": launch_readiness_score,
            "breakout_quality": breakout_quality,
            "resonance_quality": resonance_quality,
            "risk_of_late_entry": risk_of_late_entry,
            "launch_phase_label": launch_phase_label,
        },
        index=feature_frame.index,
    )


def _apply_incremental_probability_upgrade(
    raw_probabilities: np.ndarray | list[float] | tuple[float, ...],
    feature_frame: pd.DataFrame | None,
) -> tuple[np.ndarray, pd.DataFrame]:
    clipped_raw = _clip_probability_array(raw_probabilities)
    if not isinstance(feature_frame, pd.DataFrame) or feature_frame.empty:
        return clipped_raw, pd.DataFrame()

    score_frame = _build_context_score_frame(feature_frame)
    if score_frame.empty:
        return clipped_raw, score_frame
    specialist_frame = _build_launch_specialist_frame(feature_frame)

    anchor_score = clipped_raw * 100
    composite_score = score_frame["context_composite_score"].to_numpy(dtype=float)
    confidence = score_frame["context_confidence"].to_numpy(dtype=float)
    same_direction = np.sign(clipped_raw - 0.5) == np.sign(composite_score - 50)
    alignment_scale = np.where(same_direction, 1.0, 0.64)
    score_gap = composite_score - anchor_score
    delta = np.tanh(score_gap / 18.0) * 0.08 * confidence * alignment_scale
    delta += np.tanh((composite_score - 50.0) / 22.0) * 0.015 * confidence
    if not specialist_frame.empty:
        specialist_delta = specialist_frame["launch_specialist_delta_pct"].to_numpy(dtype=float) / 100.0
        specialist_confidence = specialist_frame["launch_specialist_confidence"].to_numpy(dtype=float)
        specialist_alignment = np.where(same_direction, 1.0, 0.72)
        delta += specialist_delta * specialist_alignment * (0.74 + specialist_confidence * 0.26)
    delta = np.clip(delta, -0.13, 0.13)
    upgraded = _clip_probability_array(clipped_raw + delta)

    detail_frame = score_frame.copy()
    if not specialist_frame.empty:
        detail_frame = detail_frame.join(specialist_frame, how="left")
    detail_frame["base_probability_pct"] = anchor_score
    detail_frame["upgraded_probability_pct"] = upgraded * 100
    detail_frame["upgrade_delta_pct"] = delta * 100
    return upgraded, detail_frame


def build_live_probability_upgrade(
    base_probability: float,
    daily: pd.DataFrame,
    *,
    latest_feature_values: pd.Series | dict[str, float] | None = None,
    minute_df: pd.DataFrame | None = None,
    news_df: pd.DataFrame | None = None,
    fund_flow_df: pd.DataFrame | None = None,
    symbol: str | None = None,
) -> dict[str, object]:
    safe_probability = float(np.clip(base_probability, 1e-4, 1 - 1e-4))
    latest = _prepare_live_feature_frame(daily, latest_feature_values=latest_feature_values, symbol=symbol)
    context_frame = _build_context_score_frame(latest)
    specialist_frame = _build_launch_specialist_frame(latest)
    historical_composite = (
        float(context_frame["context_composite_score"].iloc[-1])
        if not context_frame.empty and "context_composite_score" in context_frame.columns
        else safe_probability * 100
    )
    launch_specialist_score = (
        float(specialist_frame["launch_specialist_score"].iloc[-1])
        if not specialist_frame.empty and "launch_specialist_score" in specialist_frame.columns
        else 50.0
    )
    launch_regime_fit_score = (
        float(specialist_frame["launch_regime_fit_score"].iloc[-1])
        if not specialist_frame.empty and "launch_regime_fit_score" in specialist_frame.columns
        else 50.0
    )
    launch_specialist_confidence = (
        float(specialist_frame["launch_specialist_confidence"].iloc[-1])
        if not specialist_frame.empty and "launch_specialist_confidence" in specialist_frame.columns
        else 0.50
    )
    historical_composite = float(np.clip(historical_composite * 0.90 + launch_specialist_score * 0.10, 0.0, 100.0))

    minute_view = minute_df if isinstance(minute_df, pd.DataFrame) else pd.DataFrame()
    news_view = news_df if isinstance(news_df, pd.DataFrame) else pd.DataFrame()
    fund_view = fund_flow_df if isinstance(fund_flow_df, pd.DataFrame) else pd.DataFrame()

    intraday_state = evaluate_intraday(minute_view)
    intraday_structure_signal = evaluate_intraday_structure_signal(minute_view)
    temporal_news_pulse = evaluate_temporal_news_pulse(news_view)
    live_news_signal = build_research_enhanced_news_signal(
        news_view,
        base_signal=evaluate_news_sentiment(news_view),
        symbol=symbol,
    )
    live_fund_signal = evaluate_main_fund_signal(fund_view)
    research_news_score = float(live_news_signal.get("research_impact_score", live_news_signal.get("sentiment_score", 50.0)) or 50.0)
    research_news_excess_1d = float(live_news_signal.get("research_expected_excess_return_1d_pct", 0.0) or 0.0)

    intraday_execution_score = _clip(
        30
        + float(intraday_state.get("score", 0.5)) * 48
        + float(_signal_value(intraday_structure_signal, "first30_volume_share", 0.0)) * 100 * 0.18
        + float(_signal_value(intraday_structure_signal, "opening_volume_ratio", 0.0)) * 100 * 0.12
        + float(_signal_value(intraday_structure_signal, "early_return_pct", 0.0)) * 100 * 0.45
        - float(intraday_state.get("max_pullback", 0.0)) * 100 * 0.30
    )
    live_context_score = _clip(
        historical_composite * 0.54
        + intraday_execution_score * 0.22
        + float(_signal_value(temporal_news_pulse, "next_session_score", 50.0)) * 0.10
        + float(live_news_signal.get("sentiment_score", 50.0)) * 0.05
        + research_news_score * 0.03
        + float(live_fund_signal.get("fund_score", 50.0)) * 0.06
        + float(np.clip(research_news_excess_1d, -4.0, 4.0)) * 0.45
    )
    availability = 0.55
    if not minute_view.empty:
        availability += 0.15
    if not news_view.empty:
        availability += 0.15
    if not fund_view.empty:
        availability += 0.10
    availability = float(np.clip(availability, 0.55, 0.92))
    same_direction = np.sign(safe_probability - 0.5) == np.sign(live_context_score - 50.0)
    alignment_scale = 1.0 if same_direction else 0.68
    live_delta = float(np.tanh((live_context_score - historical_composite) / 16.0) * 0.04 * availability * alignment_scale)
    upgraded_probability = float(np.clip(safe_probability + live_delta, 1e-4, 1 - 1e-4))

    summary = (
        f"原模型概率 {safe_probability * 100:.1f}% ，"
        f"历史增强分 {historical_composite:.1f} ，"
        f"实时执行分 {live_context_score:.1f} ，"
        f"最终修正 {live_delta * 100:+.1f}pct。"
    )
    return {
        "base_probability": safe_probability,
        "upgraded_probability": upgraded_probability,
        "live_upgrade_delta": live_delta,
        "historical_composite_score": round(historical_composite, 2),
        "launch_specialist_score": round(launch_specialist_score, 2),
        "launch_regime_fit_score": round(launch_regime_fit_score, 2),
        "launch_specialist_confidence": round(launch_specialist_confidence * 100, 2),
        "intraday_execution_score": round(intraday_execution_score, 2),
        "live_context_score": round(live_context_score, 2),
        "temporal_news_score": round(float(_signal_value(temporal_news_pulse, "next_session_score", 50.0)), 2),
        "live_news_score": round(float(live_news_signal.get("sentiment_score", 50.0)), 2),
        "research_news_score": round(research_news_score, 2),
        "research_news_excess_1d_pct": round(research_news_excess_1d, 4),
        "research_news_excess_3d_pct": round(float(live_news_signal.get("research_expected_excess_return_3d_pct", 0.0) or 0.0), 4),
        "research_news_excess_5d_pct": round(float(live_news_signal.get("research_expected_excess_return_5d_pct", 0.0) or 0.0), 4),
        "research_news_event_count": int(live_news_signal.get("research_event_count", 0) or 0),
        "research_news_primary_category": str(live_news_signal.get("research_primary_category", "general") or "general"),
        "live_fund_score": round(float(live_fund_signal.get("fund_score", 50.0)), 2),
        "summary": summary,
    }


def build_sector_fund_probability_upgrade(
    base_probability: float,
    *,
    sector_signal: dict[str, object] | None = None,
) -> dict[str, object]:
    safe_probability = float(np.clip(base_probability, 1e-4, 1 - 1e-4))
    signal = sector_signal or {}
    sector_score = float(_signal_value(signal, "sector_score", 50.0) or 50.0)
    sector_label = str(_signal_value(signal, "sector_label", "行业热度未知") or "行业热度未知")
    sector_summary = str(_signal_value(signal, "sector_summary", "") or "")

    sector_gap = sector_score - 50.0
    anchor_gap = sector_score - safe_probability * 100.0
    same_direction = np.sign(safe_probability - 0.5) == np.sign(sector_gap)
    alignment_scale = 1.0 if same_direction else 0.72
    availability = 0.74 if signal else 0.58
    sector_context_score = float(_clip(50.0 + sector_gap * 0.88))
    sector_delta = float(np.tanh(anchor_gap / 18.0) * 0.028 * availability * alignment_scale)
    sector_delta += float(np.tanh(sector_gap / 22.0) * 0.012 * availability)
    sector_delta = float(np.clip(sector_delta, -0.035, 0.035))
    upgraded_probability = float(np.clip(safe_probability + sector_delta, 1e-4, 1 - 1e-4))

    summary_parts = [f"板块资金热度 {sector_label} {sector_score:.1f} 分"]
    if sector_summary:
        summary_parts.append(sector_summary)
    summary_parts.append(f"板块资金修正 {sector_delta * 100:+.1f}pct")
    return {
        "base_probability": safe_probability,
        "upgraded_probability": upgraded_probability,
        "sector_upgrade_delta": sector_delta,
        "sector_fund_score": round(sector_score, 2),
        "sector_context_score": round(sector_context_score, 2),
        "sector_label": sector_label,
        "summary": "；".join(summary_parts),
    }


def apply_sector_fund_probability_upgrade(
    result: ProbabilityResult,
    *,
    sector_signal: dict[str, object] | None = None,
) -> ProbabilityResult:
    try:
        current_probability = float(getattr(result, "latest_probability", 0.5))
    except (TypeError, ValueError):
        current_probability = 0.5
    try:
        base_probability_value = getattr(result, "base_probability", current_probability)
        base_probability = float(base_probability_value)
    except (TypeError, ValueError):
        base_probability = current_probability

    metrics = dict(getattr(result, "metrics", {}) or {})
    upgrade_payload = build_sector_fund_probability_upgrade(
        current_probability,
        sector_signal=sector_signal,
    )
    signal_breakdown = dict(getattr(result, "signal_breakdown", {}) or {})
    signal_breakdown.update(
        {
            "sector_fund_score": round(float(upgrade_payload["sector_fund_score"]), 2),
            "sector_context_score": round(float(upgrade_payload["sector_context_score"]), 2),
            "sector_fund_delta_pct": round(float(upgrade_payload["sector_upgrade_delta"]) * 100, 2),
        }
    )

    upgraded_probability = float(upgrade_payload["upgraded_probability"])
    current_projected_upside = float(getattr(result, "predicted_upside_pct", 0.0) or 0.0)
    current_projected_low = float(getattr(result, "predicted_upside_low_pct", 0.0) or 0.0)
    current_projected_high = float(getattr(result, "predicted_upside_high_pct", 0.0) or 0.0)
    if current_projected_upside > 0:
        probability_shift = upgraded_probability - current_probability
        projected_upside_pct = float(np.clip(current_projected_upside * (1.0 + probability_shift * 0.55), 0.0, 60.0))
        projected_upside_low_pct = float(np.clip(current_projected_low * (1.0 + probability_shift * 0.38), 0.0, projected_upside_pct))
        projected_upside_high_pct = float(
            np.clip(
                max(current_projected_high * (1.0 + probability_shift * 0.70), projected_upside_pct),
                projected_upside_pct,
                80.0,
            )
        )
    else:
        projected_upside_pct, projected_upside_low_pct, projected_upside_high_pct = _estimate_projected_upside(
            upgraded_probability,
            metrics=metrics,
            signal_breakdown=signal_breakdown,
        )
    signal_breakdown.update(
        {
            "predicted_upside_pct": round(projected_upside_pct, 2),
            "predicted_upside_low_pct": round(projected_upside_low_pct, 2),
            "predicted_upside_high_pct": round(projected_upside_high_pct, 2),
        }
    )

    strategy_score = round(
        _clip(
            upgraded_probability * 100 * 0.52
            + float(signal_breakdown.get("trend_score", 50.0)) * 0.14
            + float(signal_breakdown.get("breakout_score", 50.0)) * 0.10
            + float(signal_breakdown.get("pullback_score", 50.0)) * 0.08
            + (100 - float(signal_breakdown.get("risk_score", 50.0))) * 0.08
            + float(signal_breakdown.get("sector_fund_score", 50.0)) * 0.08
        ),
        2,
    )
    signal_label, risk_label = _strategy_labels(
        latest_probability=upgraded_probability,
        strategy_score=strategy_score,
        risk_score=float(signal_breakdown.get("risk_score", 50.0)),
    )
    precision_gate_threshold, precision_gate_precision, precision_gate_support, precision_gate_active, precision_gate_label = (
        _precision_gate_state(metrics, upgraded_probability)
    )
    upgrade_components = dict(getattr(result, "upgrade_components", {}) or {})
    upgrade_components.update(
        {
            "sector_fund_score": float(upgrade_payload["sector_fund_score"]),
            "sector_context_score": float(upgrade_payload["sector_context_score"]),
            "sector_fund_delta_pct": float(upgrade_payload["sector_upgrade_delta"]) * 100,
        }
    )
    current_summary = str(getattr(result, "upgrade_summary", "") or "").strip()
    latest_summary = str(upgrade_payload["summary"]).strip()
    combined_summary = f"{current_summary} {latest_summary}".strip() if current_summary else latest_summary
    update_payload = {
        "latest_probability": upgraded_probability,
        "strategy_score": strategy_score,
        "signal_label": signal_label,
        "risk_label": risk_label,
        "precision_gate_threshold": precision_gate_threshold,
        "precision_gate_precision": precision_gate_precision,
        "precision_gate_support": precision_gate_support,
        "precision_gate_active": precision_gate_active,
        "precision_gate_label": precision_gate_label,
        "signal_breakdown": signal_breakdown,
        "base_probability": base_probability,
        "upgrade_delta": upgraded_probability - base_probability,
        "upgrade_components": upgrade_components,
        "upgrade_summary": combined_summary,
        "predicted_upside_pct": projected_upside_pct,
        "predicted_upside_low_pct": projected_upside_low_pct,
        "predicted_upside_high_pct": projected_upside_high_pct,
    }
    if isinstance(result, ProbabilityResult):
        return replace(result, **update_payload)

    for key, value in update_payload.items():
        setattr(result, key, value)
    return result


def apply_live_probability_upgrade(
    result: ProbabilityResult,
    daily: pd.DataFrame,
    *,
    latest_feature_values: pd.Series | dict[str, float] | None = None,
    minute_df: pd.DataFrame | None = None,
    news_df: pd.DataFrame | None = None,
    fund_flow_df: pd.DataFrame | None = None,
    symbol: str | None = None,
) -> ProbabilityResult:
    try:
        current_probability = float(getattr(result, "latest_probability", 0.5))
    except (TypeError, ValueError):
        current_probability = 0.5
    try:
        base_probability_value = getattr(result, "base_probability", current_probability)
        base_probability = float(base_probability_value)
    except (TypeError, ValueError):
        base_probability = current_probability
    metrics = dict(getattr(result, "metrics", {}) or {})
    live_payload = build_live_probability_upgrade(
        current_probability,
        daily,
        latest_feature_values=latest_feature_values,
        minute_df=minute_df,
        news_df=news_df,
        fund_flow_df=fund_flow_df,
        symbol=symbol,
    )
    signal_breakdown = dict(getattr(result, "signal_breakdown", {}) or {})
    signal_breakdown.update(
        {
            "base_probability_pct": round(float(live_payload["base_probability"]) * 100, 2),
            "upgraded_probability_pct": round(float(live_payload["upgraded_probability"]) * 100, 2),
            "upgrade_delta_pct": round(float(live_payload["live_upgrade_delta"]) * 100, 2),
            "historical_composite_score": round(float(live_payload["historical_composite_score"]), 2),
            "launch_specialist_score": round(float(live_payload["launch_specialist_score"]), 2),
            "launch_regime_fit_score": round(float(live_payload["launch_regime_fit_score"]), 2),
            "launch_specialist_confidence": round(float(live_payload["launch_specialist_confidence"]), 2),
            "intraday_execution_score": round(float(live_payload["intraday_execution_score"]), 2),
            "live_context_score": round(float(live_payload["live_context_score"]), 2),
            "temporal_news_score": round(float(live_payload["temporal_news_score"]), 2),
            "live_news_score": round(float(live_payload["live_news_score"]), 2),
            "research_news_score": round(float(live_payload.get("research_news_score", 50.0)), 2),
            "research_news_excess_1d_pct": round(float(live_payload.get("research_news_excess_1d_pct", 0.0)), 4),
            "research_news_excess_3d_pct": round(float(live_payload.get("research_news_excess_3d_pct", 0.0)), 4),
            "research_news_excess_5d_pct": round(float(live_payload.get("research_news_excess_5d_pct", 0.0)), 4),
            "research_news_event_count": float(live_payload.get("research_news_event_count", 0.0) or 0.0),
            "live_fund_score": round(float(live_payload["live_fund_score"]), 2),
        }
    )
    upgraded_probability = float(live_payload["upgraded_probability"])
    current_projected_upside = float(getattr(result, "predicted_upside_pct", 0.0) or 0.0)
    current_projected_low = float(getattr(result, "predicted_upside_low_pct", 0.0) or 0.0)
    current_projected_high = float(getattr(result, "predicted_upside_high_pct", 0.0) or 0.0)
    if current_projected_upside > 0:
        probability_shift = upgraded_probability - current_probability
        projected_upside_pct = float(np.clip(current_projected_upside * (1.0 + probability_shift * 0.55), 0.0, 60.0))
        projected_upside_low_pct = float(np.clip(current_projected_low * (1.0 + probability_shift * 0.38), 0.0, projected_upside_pct))
        projected_upside_high_pct = float(
            np.clip(
                max(current_projected_high * (1.0 + probability_shift * 0.70), projected_upside_pct),
                projected_upside_pct,
                80.0,
            )
        )
    else:
        projected_upside_pct, projected_upside_low_pct, projected_upside_high_pct = _estimate_projected_upside(
            upgraded_probability,
            metrics=metrics,
            signal_breakdown=signal_breakdown,
        )
    signal_breakdown.update(
        {
            "predicted_upside_pct": round(projected_upside_pct, 2),
            "predicted_upside_low_pct": round(projected_upside_low_pct, 2),
            "predicted_upside_high_pct": round(projected_upside_high_pct, 2),
        }
    )
    strategy_score = round(
        _clip(
            upgraded_probability * 100 * 0.50
            + float(signal_breakdown.get("trend_score", 50.0)) * 0.14
            + float(signal_breakdown.get("breakout_score", 50.0)) * 0.10
            + float(signal_breakdown.get("pullback_score", 50.0)) * 0.08
            + (100 - float(signal_breakdown.get("risk_score", 50.0))) * 0.08
            + float(signal_breakdown.get("sector_fund_score", 50.0)) * 0.05
            + float(signal_breakdown.get("intraday_execution_score", 50.0)) * 0.06
            + float(signal_breakdown.get("temporal_news_score", 50.0)) * 0.03
            + float(signal_breakdown.get("research_news_score", 50.0)) * 0.01
            + float(signal_breakdown.get("launch_specialist_score", 50.0)) * 0.04
        ),
        2,
    )
    signal_label, risk_label = _strategy_labels(
        latest_probability=upgraded_probability,
        strategy_score=strategy_score,
        risk_score=float(signal_breakdown.get("risk_score", 50.0)),
    )
    precision_gate_threshold, precision_gate_precision, precision_gate_support, precision_gate_active, precision_gate_label = (
        _precision_gate_state(metrics, upgraded_probability)
    )
    upgrade_components = dict(getattr(result, "upgrade_components", {}) or {})
    upgrade_components.update(
        {
            "launch_specialist_score": float(live_payload["launch_specialist_score"]),
            "launch_regime_fit_score": float(live_payload["launch_regime_fit_score"]),
            "launch_specialist_confidence": float(live_payload["launch_specialist_confidence"]),
            "intraday_execution_score": float(live_payload["intraday_execution_score"]),
            "live_context_score": float(live_payload["live_context_score"]),
            "temporal_news_score": float(live_payload["temporal_news_score"]),
            "live_news_score": float(live_payload["live_news_score"]),
            "research_news_score": float(live_payload.get("research_news_score", 50.0)),
            "research_news_excess_1d_pct": float(live_payload.get("research_news_excess_1d_pct", 0.0)),
            "research_news_excess_3d_pct": float(live_payload.get("research_news_excess_3d_pct", 0.0)),
            "research_news_excess_5d_pct": float(live_payload.get("research_news_excess_5d_pct", 0.0)),
            "research_news_event_count": float(live_payload.get("research_news_event_count", 0.0) or 0.0),
            "live_fund_score": float(live_payload["live_fund_score"]),
        }
    )
    current_summary = str(getattr(result, "upgrade_summary", "") or "").strip()
    latest_summary = str(live_payload["summary"]).strip()
    combined_summary = f"{current_summary} {latest_summary}".strip() if current_summary else latest_summary
    update_payload = {
        "latest_probability": upgraded_probability,
        "strategy_score": strategy_score,
        "signal_label": signal_label,
        "risk_label": risk_label,
        "precision_gate_threshold": precision_gate_threshold,
        "precision_gate_precision": precision_gate_precision,
        "precision_gate_support": precision_gate_support,
        "precision_gate_active": precision_gate_active,
        "precision_gate_label": precision_gate_label,
        "signal_breakdown": signal_breakdown,
        "base_probability": base_probability,
        "upgrade_delta": upgraded_probability - base_probability,
        "upgrade_components": upgrade_components,
        "upgrade_summary": combined_summary,
        "predicted_upside_pct": projected_upside_pct,
        "predicted_upside_low_pct": projected_upside_low_pct,
        "predicted_upside_high_pct": projected_upside_high_pct,
    }
    if isinstance(result, ProbabilityResult):
        return replace(result, **update_payload)

    for key, value in update_payload.items():
        setattr(result, key, value)
    return result


def _normalized_datetime_index(index_like) -> pd.DatetimeIndex:
    values = pd.to_datetime(pd.Index(index_like), errors="coerce")
    valid = pd.DatetimeIndex(values[~pd.isna(values)])
    if valid.empty:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(valid.unique()).sort_values()


def _first_existing_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    lowered = {str(column).lower(): column for column in frame.columns}
    for candidate in candidates:
        matched = lowered.get(str(candidate).lower())
        if matched is not None:
            return matched
    return None


def _first_value(row: pd.Series, keys: tuple[str, ...], default=None):
    for key in keys:
        if key in row.index:
            value = row.get(key)
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                return value
    return default


def _signal_value(signal: object, key: str, default=None):
    if isinstance(signal, dict):
        return signal.get(key, default)
    return getattr(signal, key, default)


def _keyword_hits(text: str, mapping: dict[str, float]) -> tuple[float, int]:
    score = 0.0
    hit_count = 0
    for keyword, weight in mapping.items():
        if keyword in text:
            score += float(weight)
            hit_count += 1
    return score, hit_count


def _resolve_symbol_from_daily(daily: pd.DataFrame, symbol: str | None = None) -> str | None:
    if symbol:
        return normalize_symbol(symbol)
    if "symbol" in daily.columns:
        candidates = daily["symbol"].dropna()
        if not candidates.empty:
            return normalize_symbol(str(candidates.iloc[-1]))
    attr_symbol = daily.attrs.get("symbol")
    if attr_symbol:
        return normalize_symbol(str(attr_symbol))
    return None


def _neutral_feature_frame(index_like, columns: list[str] | tuple[str, ...]) -> pd.DataFrame:
    index = _normalized_datetime_index(index_like)
    return pd.DataFrame(0.0, index=index, columns=list(columns), dtype=float)


def _external_snapshot_cache_path(symbol: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"external_snapshots_v{EXTERNAL_SNAPSHOT_CACHE_VERSION}_{normalize_symbol(symbol)}.pkl"


def _load_cached_external_snapshot_frame(symbol: str, trade_dates: pd.DatetimeIndex) -> pd.DataFrame | None:
    cache_path = _external_snapshot_cache_path(symbol)
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None

    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    cached = payload.get("data") if isinstance(payload, dict) else None
    if meta.get("cache_version") != EXTERNAL_SNAPSHOT_CACHE_VERSION or not isinstance(cached, pd.DataFrame):
        return None
    if cached.empty:
        return _neutral_feature_frame(trade_dates, EXTERNAL_SNAPSHOT_COLUMNS)

    frame = cached.copy()
    frame.index = _normalized_datetime_index(frame.index)
    if frame.empty:
        return None
    if trade_dates.empty:
        return _neutral_feature_frame(trade_dates, EXTERNAL_SNAPSHOT_COLUMNS)
    if frame.index.min() > trade_dates.min() or frame.index.max() < trade_dates.max():
        return None
    return frame.reindex(trade_dates).reindex(columns=EXTERNAL_SNAPSHOT_COLUMNS, fill_value=0.0).fillna(0.0)


def _write_external_snapshot_cache(symbol: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    normalized = frame.copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    payload = {
        "meta": {
            "cache_version": EXTERNAL_SNAPSHOT_CACHE_VERSION,
            "symbol": normalize_symbol(symbol),
            "start_date": normalized.index.min().strftime("%Y-%m-%d"),
            "end_date": normalized.index.max().strftime("%Y-%m-%d"),
        },
        "data": normalized,
    }
    try:
        with _external_snapshot_cache_path(symbol).open("wb") as handle:
            pickle.dump(payload, handle)
    except Exception:
        return


def _build_news_snapshot_features(trade_dates: pd.DatetimeIndex, news_df: pd.DataFrame) -> pd.DataFrame:
    features = _neutral_feature_frame(
        trade_dates,
        [
            "news_sentiment_3d",
            "news_sentiment_7d",
            "news_confidence_7d",
            "news_volume_3d",
            "news_volume_7d",
            "news_event_shock_3d",
            "news_positive_ratio_7d",
            "news_research_score_3d",
            "news_research_score_7d",
            "news_research_confidence_7d",
            "news_research_excess_1d",
            "news_research_excess_5d",
        ],
    )
    if features.empty or news_df.empty:
        return features

    rows: list[dict[str, object]] = []
    for _, row in news_df.iterrows():
        publish_time = pd.to_datetime(
            _first_value(row, NEWS_TIME_KEYS + ("date", "datetime", "published_at")),
            errors="coerce",
        )
        if pd.isna(publish_time):
            continue
        title = str(_first_value(row, NEWS_TITLE_KEYS, "") or "")
        body = str(_first_value(row, NEWS_BODY_KEYS, "") or "")
        source = str(_first_value(row, NEWS_SOURCE_KEYS, "unknown") or "unknown")

        title_bull_score, title_bull_hits = _keyword_hits(title, BULLISH_KEYWORDS)
        title_bear_score, title_bear_hits = _keyword_hits(title, BEARISH_KEYWORDS)
        body_bull_score, body_bull_hits = _keyword_hits(body, BULLISH_KEYWORDS)
        body_bear_score, body_bear_hits = _keyword_hits(body, BEARISH_KEYWORDS)

        raw_score = (title_bull_score + title_bear_score) * 1.35 + (body_bull_score + body_bear_score) * 0.85
        positive_hits = title_bull_hits + body_bull_hits
        negative_hits = title_bear_hits + body_bear_hits
        hit_total = positive_hits + negative_hits
        contradiction_penalty = 0.85 if positive_hits and negative_hits else 1.0
        source_weight = float(SOURCE_WEIGHTS.get(source, 1.0))
        sentiment = float(np.tanh(raw_score * contradiction_penalty * source_weight / 4.5))
        confidence = float(
            np.clip(
                0.18
                + min(hit_total / 4.0, 0.42)
                + min(abs(raw_score) / 6.0, 0.30)
                + (0.06 if title.strip() else 0.0),
                0.0,
                1.0,
            )
        )
        shock = float(np.sign(sentiment) * min(abs(sentiment) * (0.55 + confidence * 0.45), 1.0))
        rows.append(
            {
                "date": publish_time.normalize(),
                "weighted_score": sentiment * confidence,
                "weight": confidence,
                "shock": shock,
                "headline_count": 1.0,
                "positive_flag": 1.0 if sentiment > 0.05 else 0.0,
                "negative_flag": 1.0 if sentiment < -0.05 else 0.0,
                "source_count": source,
            }
        )

    if not rows:
        return features

    article_frame = pd.DataFrame(rows)
    research_rows: list[dict[str, object]] = []
    try:
        classified_news = classify_news_events(news_df)
    except Exception:
        classified_news = pd.DataFrame()
    if isinstance(classified_news, pd.DataFrame) and not classified_news.empty:
        for _, event in classified_news.iterrows():
            publish_time = pd.to_datetime(event.get("published_at"), errors="coerce")
            if pd.isna(publish_time):
                continue
            research = score_news_event_with_research(event)
            research_confidence = float(research.get("research_confidence", 0.30) or 0.30)
            research_rows.append(
                {
                    "date": publish_time.normalize(),
                    "research_score_gap": (float(research.get("research_adjusted_score", 50.0) or 50.0) - 50.0) / 50.0,
                    "research_confidence": research_confidence,
                    "research_weight": research_confidence,
                    "research_excess_1d": float(research.get("research_excess_return_1d_pct", 0.0) or 0.0),
                    "research_excess_5d": float(research.get("research_excess_return_5d_pct", 0.0) or 0.0),
                }
            )
    daily_news = (
        article_frame.groupby("date")
        .agg(
            weighted_score=("weighted_score", "sum"),
            weight=("weight", "sum"),
            shock=("shock", "sum"),
            headline_count=("headline_count", "sum"),
            positive_count=("positive_flag", "sum"),
            negative_count=("negative_flag", "sum"),
            source_count=("source_count", pd.Series.nunique),
        )
        .sort_index()
    )
    calendar_index = pd.date_range(
        min(trade_dates.min() - pd.Timedelta(days=7), daily_news.index.min()),
        trade_dates.max(),
        freq="D",
    )
    calendar = daily_news.reindex(calendar_index, fill_value=0.0)
    score_3d = calendar["weighted_score"].rolling(3, min_periods=1).sum()
    weight_3d = calendar["weight"].rolling(3, min_periods=1).sum()
    score_7d = calendar["weighted_score"].rolling(7, min_periods=1).sum()
    weight_7d = calendar["weight"].rolling(7, min_periods=1).sum()
    headline_3d = calendar["headline_count"].rolling(3, min_periods=1).sum()
    headline_7d = calendar["headline_count"].rolling(7, min_periods=1).sum()
    positive_7d = calendar["positive_count"].rolling(7, min_periods=1).sum()
    negative_7d = calendar["negative_count"].rolling(7, min_periods=1).sum()
    source_7d = calendar["source_count"].rolling(7, min_periods=1).sum()
    shock_3d = calendar["shock"].rolling(3, min_periods=1).sum()

    features.loc[:, "news_sentiment_3d"] = (
        score_3d.reindex(trade_dates) / weight_3d.reindex(trade_dates).replace(0, np.nan)
    ).fillna(0.0)
    features.loc[:, "news_sentiment_7d"] = (
        score_7d.reindex(trade_dates) / weight_7d.reindex(trade_dates).replace(0, np.nan)
    ).fillna(0.0)
    features.loc[:, "news_confidence_7d"] = np.clip(
        0.22
        + np.minimum(weight_7d.reindex(trade_dates).fillna(0.0) / 5.0, 0.48)
        + np.minimum(source_7d.reindex(trade_dates).fillna(0.0) / 10.0, 0.16)
        + np.minimum(headline_7d.reindex(trade_dates).fillna(0.0) / 14.0, 0.14),
        0.0,
        1.0,
    )
    features.loc[:, "news_volume_3d"] = np.log1p(headline_3d.reindex(trade_dates).fillna(0.0))
    features.loc[:, "news_volume_7d"] = np.log1p(headline_7d.reindex(trade_dates).fillna(0.0))
    features.loc[:, "news_event_shock_3d"] = np.tanh(shock_3d.reindex(trade_dates).fillna(0.0) / 2.5)
    features.loc[:, "news_positive_ratio_7d"] = (
        positive_7d.reindex(trade_dates).fillna(0.0)
        / (positive_7d.reindex(trade_dates).fillna(0.0) + negative_7d.reindex(trade_dates).fillna(0.0)).replace(0, np.nan)
    ).fillna(0.5)
    if research_rows:
        research_frame = pd.DataFrame(research_rows)
        daily_research = (
            research_frame.groupby("date")
            .agg(
                research_score_gap=("research_score_gap", "sum"),
                research_confidence=("research_confidence", "sum"),
                research_weight=("research_weight", "sum"),
                research_excess_1d=("research_excess_1d", "sum"),
                research_excess_5d=("research_excess_5d", "sum"),
            )
            .sort_index()
        )
        research_calendar = daily_research.reindex(calendar_index, fill_value=0.0)
        research_score_3d = research_calendar["research_score_gap"].rolling(3, min_periods=1).sum()
        research_score_7d = research_calendar["research_score_gap"].rolling(7, min_periods=1).sum()
        research_weight_7d = research_calendar["research_weight"].rolling(7, min_periods=1).sum()
        research_confidence_7d = research_calendar["research_confidence"].rolling(7, min_periods=1).sum()
        research_excess_1d_3d = research_calendar["research_excess_1d"].rolling(3, min_periods=1).sum()
        research_excess_5d_7d = research_calendar["research_excess_5d"].rolling(7, min_periods=1).sum()
        features.loc[:, "news_research_score_3d"] = np.tanh(
            research_score_3d.reindex(trade_dates).fillna(0.0) / 2.4
        )
        features.loc[:, "news_research_score_7d"] = np.tanh(
            research_score_7d.reindex(trade_dates).fillna(0.0) / 3.6
        )
        features.loc[:, "news_research_confidence_7d"] = np.clip(
            research_confidence_7d.reindex(trade_dates).fillna(0.0)
            / research_weight_7d.reindex(trade_dates).replace(0, np.nan),
            0.0,
            1.0,
        ).fillna(0.0)
        features.loc[:, "news_research_excess_1d"] = np.tanh(
            research_excess_1d_3d.reindex(trade_dates).fillna(0.0) / 6.0
        )
        features.loc[:, "news_research_excess_5d"] = np.tanh(
            research_excess_5d_7d.reindex(trade_dates).fillna(0.0) / 10.0
        )
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _build_fund_snapshot_features(trade_dates: pd.DatetimeIndex, fund_flow_df: pd.DataFrame) -> pd.DataFrame:
    features = _neutral_feature_frame(
        trade_dates,
        [
            "fund_ratio_1d",
            "fund_ratio_5d",
            "fund_net_strength_1d",
            "fund_net_strength_5d",
            "fund_positive_ratio_5d",
            "fund_inflow_streak_5d",
            "fund_trend_delta_5d",
            "fund_consistency_5d",
        ],
    )
    if features.empty or fund_flow_df.empty:
        return features

    date_column = _first_existing_column(fund_flow_df, ("date", "日期", "鏃ユ湡"))
    ratio_column = _first_existing_column(fund_flow_df, FUND_RATIO_KEYS)
    net_column = _first_existing_column(fund_flow_df, FUND_NET_KEYS)
    if date_column is None or ratio_column is None or net_column is None:
        return features

    daily_fund = fund_flow_df.copy()
    daily_fund["snapshot_date"] = pd.to_datetime(daily_fund[date_column], errors="coerce").dt.normalize()
    daily_fund = daily_fund.dropna(subset=["snapshot_date"])
    if daily_fund.empty:
        return features

    grouped = (
        daily_fund.groupby("snapshot_date")
        .agg(
            ratio=(ratio_column, "mean"),
            net=(net_column, "mean"),
        )
        .sort_index()
        .reindex(trade_dates, fill_value=0.0)
    )
    ratio = pd.to_numeric(grouped["ratio"], errors="coerce").fillna(0.0)
    net = pd.to_numeric(grouped["net"], errors="coerce").fillna(0.0)
    positive = ratio.gt(0).astype(float)

    inflow_streak_values: list[float] = []
    streak = 0
    for value in ratio.to_numpy(dtype=float):
        if value > 0:
            streak = min(streak + 1, 5)
        else:
            streak = 0
        inflow_streak_values.append(streak / 5.0)

    rolling_ratio_5 = ratio.rolling(5, min_periods=1).mean()
    rolling_net_5 = net.rolling(5, min_periods=1).mean()
    short_ratio = ratio.rolling(3, min_periods=1).mean()
    consistency = 1.0 - ratio.rolling(5, min_periods=2).std(ddof=0) / (ratio.abs().rolling(5, min_periods=1).mean() + 1.0)

    features.loc[:, "fund_ratio_1d"] = np.tanh(ratio / 6.0)
    features.loc[:, "fund_ratio_5d"] = np.tanh(rolling_ratio_5 / 6.0)
    features.loc[:, "fund_net_strength_1d"] = np.tanh(net / 3.0e8)
    features.loc[:, "fund_net_strength_5d"] = np.tanh(rolling_net_5 / 3.0e8)
    features.loc[:, "fund_positive_ratio_5d"] = positive.rolling(5, min_periods=1).mean()
    features.loc[:, "fund_inflow_streak_5d"] = inflow_streak_values
    features.loc[:, "fund_trend_delta_5d"] = np.tanh((short_ratio - rolling_ratio_5) / 4.0)
    features.loc[:, "fund_consistency_5d"] = np.clip(consistency.fillna(0.0), 0.0, 1.0)
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _external_snapshot_feature_frame(daily: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    trade_dates = _normalized_datetime_index(daily["date"] if "date" in daily.columns else daily.index)
    base = _neutral_feature_frame(trade_dates, EXTERNAL_SNAPSHOT_COLUMNS)
    if trade_dates.empty:
        return base

    resolved_symbol = _resolve_symbol_from_daily(daily, symbol=symbol)
    if not resolved_symbol:
        return base

    cached = _load_cached_external_snapshot_frame(resolved_symbol, trade_dates)
    if cached is not None:
        return cached

    try:
        news_df = fetch_stock_news(resolved_symbol, limit=120)
    except Exception:
        news_df = pd.DataFrame()
    try:
        fund_flow_df = fetch_stock_main_fund_flow(resolved_symbol, limit=160)
    except Exception:
        fund_flow_df = pd.DataFrame()

    news_features = _build_news_snapshot_features(trade_dates, news_df)
    fund_features = _build_fund_snapshot_features(trade_dates, fund_flow_df)
    merged = base.copy()
    merged.update(news_features.reindex(merged.index))
    merged.update(fund_features.reindex(merged.index))
    merged = merged.reindex(columns=EXTERNAL_SNAPSHOT_COLUMNS, fill_value=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    _write_external_snapshot_cache(resolved_symbol, merged)
    return merged


def _merge_external_snapshot_features(
    frame: pd.DataFrame,
    daily: pd.DataFrame,
    symbol: str | None = None,
) -> pd.DataFrame:
    merged = frame.copy()
    for column in EXTERNAL_SNAPSHOT_COLUMNS:
        if column not in merged.columns:
            merged[column] = 0.0
    if merged.empty:
        return merged

    trade_dates = _normalized_datetime_index(daily["date"] if "date" in daily.columns else daily.index)
    if isinstance(merged.index, pd.DatetimeIndex):
        merged = merged.copy()
    elif len(trade_dates) == len(merged):
        merged = merged.copy()
        merged.index = trade_dates
    elif len(trade_dates) == 1 and len(merged) == 1:
        merged = merged.copy()
        merged.index = trade_dates
    else:
        return merged.fillna(0.0)

    snapshot_features = _external_snapshot_feature_frame(daily, symbol=symbol)
    if snapshot_features.empty:
        return merged.fillna(0.0)

    aligned = snapshot_features.reindex(merged.index).reindex(columns=EXTERNAL_SNAPSHOT_COLUMNS, fill_value=0.0)
    for column in EXTERNAL_SNAPSHOT_COLUMNS:
        merged[column] = aligned[column].to_numpy(dtype=float)
    return merged.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _append_market_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    for column in REGIME_FEATURE_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = 0.0
    if enriched.empty:
        enriched["market_regime_label"] = pd.Series(dtype=str)
        return enriched

    market_ret_5 = pd.to_numeric(enriched.get("market_ret_5", 0.0), errors="coerce").fillna(0.0)
    market_ret_20 = pd.to_numeric(enriched.get("market_ret_20", 0.0), errors="coerce").fillna(0.0)
    market_close_vs_ma20 = pd.to_numeric(enriched.get("market_close_vs_ma20", 0.0), errors="coerce").fillna(0.0)
    market_volatility_10 = pd.to_numeric(enriched.get("market_volatility_10", 0.0), errors="coerce").fillna(0.0)
    market_range_position_20 = pd.to_numeric(enriched.get("market_range_position_20", 0.5), errors="coerce").fillna(0.5)

    trend_score = (
        market_ret_20 * 18.0
        + market_close_vs_ma20 * 26.0
        + (market_range_position_20 - 0.55) * 3.5
        - market_volatility_10 * 8.0
    )
    rebound_score = (
        market_ret_5 * 20.0
        - np.minimum(market_ret_20, 0.0) * 10.0
        - market_close_vs_ma20.abs() * 6.0
        - np.maximum(market_range_position_20 - 0.68, 0.0) * 4.0
    )
    rotation_score = (
        1.8
        - market_ret_20.abs() * 14.0
        - market_close_vs_ma20.abs() * 18.0
        - (market_range_position_20 - 0.5).abs() * 3.0
        - market_volatility_10 * 5.0
    )
    defense_score = (
        -market_ret_20 * 18.0
        - market_close_vs_ma20 * 22.0
        + market_volatility_10 * 10.0
        + np.maximum(0.45 - market_range_position_20, 0.0) * 6.0
    )
    regime_scores = np.column_stack([trend_score, rebound_score, rotation_score, defense_score])
    best_idx = regime_scores.argmax(axis=1)
    sorted_scores = np.sort(regime_scores, axis=1)
    dominance = sorted_scores[:, -1] - sorted_scores[:, -2]
    labels = np.array(REGIME_LABELS, dtype=object)[best_idx]

    enriched["market_regime_trend"] = (labels == "trend").astype(float)
    enriched["market_regime_rebound"] = (labels == "rebound").astype(float)
    enriched["market_regime_rotation"] = (labels == "rotation").astype(float)
    enriched["market_regime_defense"] = (labels == "defense").astype(float)
    enriched["market_regime_score"] = np.clip(0.5 + dominance / 4.0, 0.0, 1.0)
    enriched["market_regime_risk"] = np.clip(
        0.35
        + market_volatility_10 * 8.0
        - market_ret_20 * 2.2
        - market_close_vs_ma20 * 2.4,
        0.0,
        1.0,
    )
    enriched["market_regime_label"] = labels
    return enriched.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def explain_latest_model_state(
    daily: pd.DataFrame,
    *,
    symbol: str | None = None,
    latest_feature_values: pd.Series | dict[str, float] | None = None,
) -> dict[str, object]:
    latest = _prepare_live_feature_frame(daily, latest_feature_values=latest_feature_values, symbol=symbol)
    if latest.empty:
        return {
            "regime_code": "rotation",
            "regime_label": REGIME_DISPLAY_LABELS["rotation"],
            "regime_summary": "当前模型未拿到足够的新特征，先按中性轮动环境看待。",
            "regime_score": 0.0,
            "regime_risk": 0.0,
            "news_snapshot_score": 50.0,
            "news_snapshot_label": "中性",
            "news_snapshot_summary": "历史消息快照暂不可用。",
            "fund_snapshot_score": 50.0,
            "fund_snapshot_label": "中性",
            "fund_snapshot_summary": "历史资金快照暂不可用。",
            "state_reason_lines": [],
            "snapshot_values": {},
        }

    row = latest.iloc[-1]
    regime_code = str(_regime_labels_from_feature_frame(latest)[-1]) if len(latest) else "rotation"
    regime_score = float(row.get("market_regime_score", 0.0))
    regime_risk = float(row.get("market_regime_risk", 0.0))
    market_ret_20 = float(row.get("market_ret_20", 0.0))
    market_ret_5 = float(row.get("market_ret_5", 0.0))
    market_close_vs_ma20 = float(row.get("market_close_vs_ma20", 0.0))
    market_volatility_10 = float(row.get("market_volatility_10", 0.0))
    news_sentiment_7d = float(row.get("news_sentiment_7d", 0.0))
    news_confidence_7d = float(row.get("news_confidence_7d", 0.0))
    news_event_shock_3d = float(row.get("news_event_shock_3d", 0.0))
    news_positive_ratio_7d = float(row.get("news_positive_ratio_7d", 0.5))
    news_research_score_7d = float(row.get("news_research_score_7d", 0.0))
    news_research_confidence_7d = float(row.get("news_research_confidence_7d", 0.0))
    news_research_excess_1d = float(row.get("news_research_excess_1d", 0.0))
    news_research_excess_5d = float(row.get("news_research_excess_5d", 0.0))
    fund_ratio_5d = float(row.get("fund_ratio_5d", 0.0))
    fund_net_strength_5d = float(row.get("fund_net_strength_5d", 0.0))
    fund_positive_ratio_5d = float(row.get("fund_positive_ratio_5d", 0.5))
    fund_inflow_streak_5d = float(row.get("fund_inflow_streak_5d", 0.0))
    fund_consistency_5d = float(row.get("fund_consistency_5d", 0.0))

    if regime_code == "trend":
        regime_summary = "指数中期趋势和位置同步偏强，模型会更重视顺势延续与强者恒强。"
    elif regime_code == "rebound":
        regime_summary = "市场更像是弱势后的修复阶段，模型会更关注短线修复持续性而不是盲目追高。"
    elif regime_code == "defense":
        regime_summary = "指数处于承压或高风险环境，模型会更严格地压低激进信号。"
    else:
        regime_summary = "市场没有形成强单边趋势，模型会更依赖个股结构、消息和资金共振。"

    news_snapshot_score = _clip(
        50
        + news_sentiment_7d * 34
        + news_event_shock_3d * 14
        + (news_positive_ratio_7d - 0.5) * 28
        + news_confidence_7d * 16
        + news_research_score_7d * 18
        + news_research_confidence_7d * 8
        + news_research_excess_1d * 10
        + news_research_excess_5d * 8
    )
    if news_snapshot_score >= 62:
        news_snapshot_label = "偏多"
        news_snapshot_summary = "近 7 日消息快照整体偏正面，而且不是单条新闻脉冲。"
    elif news_snapshot_score <= 38:
        news_snapshot_label = "偏空"
        news_snapshot_summary = "近 7 日消息快照偏弱，模型会降低消息驱动的可信度。"
    else:
        news_snapshot_label = "中性"
        news_snapshot_summary = "近 7 日消息快照没有形成稳定单边倾向。"

    fund_snapshot_score = _clip(
        50
        + fund_ratio_5d * 26
        + fund_net_strength_5d * 18
        + (fund_positive_ratio_5d - 0.5) * 26
        + fund_inflow_streak_5d * 16
        + fund_consistency_5d * 10
    )
    if fund_snapshot_score >= 62:
        fund_snapshot_label = "偏强"
        fund_snapshot_summary = "近 5 日主力资金持续性较好，模型会把资金承接视为加分项。"
    elif fund_snapshot_score <= 38:
        fund_snapshot_label = "偏弱"
        fund_snapshot_summary = "近 5 日主力资金承接不足，模型会提高防守权重。"
    else:
        fund_snapshot_label = "中性"
        fund_snapshot_summary = "近 5 日主力资金没有形成持续一致的方向。"

    state_reason_lines = [
        f"沪深300近20日 {market_ret_20 * 100:.2f}% ，近5日 {market_ret_5 * 100:.2f}%",
        f"指数相对 MA20 偏离 {market_close_vs_ma20 * 100:.2f}% ，10日波动 {market_volatility_10 * 100:.2f}%",
        f"近7日消息情绪 {news_sentiment_7d:.2f} / 置信度 {news_confidence_7d:.2f} / 正面占比 {news_positive_ratio_7d * 100:.0f}%",
        f"近5日主力资金强度 {fund_ratio_5d:.2f} / 连续流入 {fund_inflow_streak_5d * 5:.0f} 天 / 一致性 {fund_consistency_5d * 100:.0f}%",
    ]
    return {
        "regime_code": regime_code,
        "regime_label": REGIME_DISPLAY_LABELS.get(regime_code, regime_code),
        "regime_summary": regime_summary,
        "regime_score": round(regime_score * 100, 1),
        "regime_risk": round(regime_risk * 100, 1),
        "news_snapshot_score": round(news_snapshot_score, 1),
        "news_snapshot_label": news_snapshot_label,
        "news_snapshot_summary": news_snapshot_summary,
        "fund_snapshot_score": round(fund_snapshot_score, 1),
        "fund_snapshot_label": fund_snapshot_label,
        "fund_snapshot_summary": fund_snapshot_summary,
        "state_reason_lines": state_reason_lines,
        "snapshot_values": {
            "market_ret_20": round(market_ret_20 * 100, 2),
            "market_ret_5": round(market_ret_5 * 100, 2),
            "market_close_vs_ma20": round(market_close_vs_ma20 * 100, 2),
            "market_volatility_10": round(market_volatility_10 * 100, 2),
            "news_sentiment_7d": round(news_sentiment_7d, 2),
            "news_confidence_7d": round(news_confidence_7d, 2),
            "news_research_score_7d": round(news_research_score_7d, 2),
            "news_research_confidence_7d": round(news_research_confidence_7d, 2),
            "news_research_excess_1d": round(news_research_excess_1d, 2),
            "news_research_excess_5d": round(news_research_excess_5d, 2),
            "fund_ratio_5d": round(fund_ratio_5d, 2),
            "fund_positive_ratio_5d": round(fund_positive_ratio_5d * 100, 1),
            "fund_consistency_5d": round(fund_consistency_5d * 100, 1),
        },
    }


def _augment_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    df = frame.copy()
    df["trend_strength"] = (
        df["ret_20"].fillna(0.0) * 120
        + df["ret_60"].fillna(0.0) * 80
        + df["ret_120"].fillna(0.0) * 62
        + df["close_vs_ma20"].fillna(0.0) * 180
        + df["close_vs_ma120"].fillna(0.0) * 120
        + df["ma5_slope_3"].fillna(0.0) * 560
        + df["ma20_slope_5"].fillna(0.0) * 520
        + df["ma60_slope_10"].fillna(0.0) * 360
        + (df["ma_alignment_score"].fillna(0.5) - 0.5) * 42
        + (df["momentum_persistence_10"].fillna(0.5) - 0.5) * 34
        + (df["efficiency_ratio_10"].fillna(0.5) - 0.5) * 40
    )
    df["breakout_readiness"] = (
        df["breakout_distance_20"].fillna(0.0) * 280
        + df["breakout_distance_60"].fillna(0.0) * 180
        + df["range_position_20"].fillna(0.5) * 16
        + df["close_near_high_5"].fillna(0.0) * 240
        + (df["volume_ratio_5"].fillna(1.0) - 1.0) * 18
        + (df["close_position_day"].fillna(0.5) - 0.5) * 26
        + (df["up_day_ratio_10"].fillna(0.5) - 0.5) * 30
    )
    df["pullback_quality"] = (
        -df["pullback_to_breakout_20"].abs().fillna(0.0) * 320
        + df["lower_shadow_ratio"].fillna(0.0) * 180
        + df["body_ratio"].fillna(0.0) * 10
        + df["drawdown_20"].fillna(0.0) * 140
        - (df["rsi_14"].fillna(0.5) - 0.58).abs() * 40
    )
    df["volume_thrust"] = (
        (df["volume_ratio_5"].fillna(1.0) - 1.0) * 32
        + (df["volume_ratio_20"].fillna(1.0) - 1.0) * 18
        + (df["amount_ratio_5"].fillna(1.0) - 1.0) * 14
        + df["turnover"].fillna(0.0) * 0.4
        + (df["turnover_ratio_20"].fillna(1.0) - 1.0) * 20
    )
    df["risk_pressure"] = (
        df["upper_shadow_ratio"].fillna(0.0) * 240
        + df["volatility_10"].fillna(0.0) * 420
        + df["volatility_20"].fillna(0.0) * 280
        + df["downside_vol_ratio_20"].fillna(0.0) * 240
        + df["atr_ratio_14"].fillna(0.0) * 360
        + df["gap_return_1"].abs().fillna(0.0) * 240
        + df["drawdown_20"].abs().fillna(0.0) * 180
    )
    df["stretch_risk"] = (
        np.maximum(df["close_vs_ma20"].fillna(0.0) - 0.08, 0.0) * 220
        + np.maximum(df["range_position_20"].fillna(0.5) - 0.88, 0.0) * 110
        + np.maximum(df["close_vs_ma120"].fillna(0.0) - 0.22, 0.0) * 120
        + np.maximum(df["rsi_14"].fillna(0.5) - 0.72, 0.0) * 140
        + np.maximum(df["volatility_contraction"].fillna(0.0), 0.0) * 40
    )
    df["launch_readiness"] = _launch_readiness_series(df)
    df["market_resonance"] = 50.0
    return df.replace([np.inf, -np.inf], np.nan)


def _build_market_feature_frame(
    start_date: str | None = None,
    end_date: str | None = None,
    benchmark_symbol: str = "sh000300",
) -> pd.DataFrame:
    market = fetch_index_daily_history(
        symbol=benchmark_symbol,
        start_date=start_date or "20220101",
        end_date=end_date,
    )
    if market.empty:
        return pd.DataFrame()

    close = market["close"]
    ma20 = close.rolling(20).mean()
    high20 = market["high"].rolling(20).max()
    low20 = market["low"].rolling(20).min()
    width = (high20 - low20).replace(0, np.nan)
    features = pd.DataFrame(index=pd.to_datetime(market["date"], errors="coerce"))
    features["market_ret_5"] = close.pct_change(5).to_numpy()
    features["market_ret_20"] = close.pct_change(20).to_numpy()
    features["market_close_vs_ma20"] = (close / ma20 - 1).to_numpy()
    features["market_volatility_10"] = close.pct_change().rolling(10).std().to_numpy()
    features["market_range_position_20"] = ((close - low20) / width).to_numpy()
    return features.replace([np.inf, -np.inf], np.nan)


def _merge_market_features(frame: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    merged = frame.copy()
    for column in MARKET_FEATURE_COLUMNS:
        if column not in merged.columns:
            merged[column] = 0.0
    if merged.empty or daily.empty:
        return merged

    try:
        start_date = pd.to_datetime(daily["date"], errors="coerce").min()
        end_date = pd.to_datetime(daily["date"], errors="coerce").max()
        market_features = _build_market_feature_frame(
            start_date=start_date.strftime("%Y%m%d") if pd.notna(start_date) else "20220101",
            end_date=end_date.strftime("%Y%m%d") if pd.notna(end_date) else None,
        )
    except Exception:
        for column in MARKET_FEATURE_COLUMNS:
            merged[column] = merged[column].fillna(0.0)
        return merged

    if market_features.empty:
        for column in MARKET_FEATURE_COLUMNS:
            merged[column] = merged[column].fillna(0.0)
        return merged

    merged = merged.join(market_features.reindex(merged.index), how="left", rsuffix="_market")
    merged["relative_strength_5"] = merged["ret_5"].fillna(0.0) - merged["market_ret_5"].fillna(0.0)
    merged["relative_strength_20"] = merged["ret_20"].fillna(0.0) - merged["market_ret_20"].fillna(0.0)
    for column in MARKET_FEATURE_COLUMNS:
        if column not in merged.columns:
            merged[column] = 0.0
        merged[column] = merged[column].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return merged.replace([np.inf, -np.inf], np.nan)


def _prepare_training_dataset(
    daily: pd.DataFrame,
    horizon_days: int,
    positive_return: float,
    *,
    symbol: str | None = None,
) -> pd.DataFrame:
    dataset = build_training_frame(daily, horizon_days=horizon_days, positive_return=positive_return)
    if dataset.empty:
        return dataset
    prepared = _augment_model_features(dataset)
    prepared = _merge_market_features(prepared, daily)
    prepared = _merge_external_snapshot_features(prepared, daily, symbol=symbol)
    prepared = _append_market_regime_features(prepared)
    prepared = _append_market_resonance_features(prepared)
    return prepared.replace([np.inf, -np.inf], np.nan)


def _prepare_live_feature_frame(
    daily: pd.DataFrame,
    latest_feature_values: pd.Series | dict[str, float] | None = None,
    *,
    symbol: str | None = None,
) -> pd.DataFrame:
    if latest_feature_values is not None:
        row = latest_feature_values if isinstance(latest_feature_values, dict) else latest_feature_values.to_dict()
        feature_frame = pd.DataFrame([row])
        trade_dates = _normalized_datetime_index(daily["date"] if "date" in daily.columns else daily.index)
        if len(trade_dates) == 1:
            feature_frame.index = trade_dates
        elif len(trade_dates) > 1:
            feature_frame.index = pd.DatetimeIndex([trade_dates.max()])
        available_feature_count = sum(1 for column in FEATURE_COLUMNS if column in feature_frame.columns)
        if available_feature_count < max(8, len(FEATURE_COLUMNS) // 4):
            try:
                base_features = build_daily_features(daily)
            except Exception:
                base_features = pd.DataFrame()
            if not base_features.empty:
                overlay_row = base_features.tail(1).copy()
                if len(feature_frame.index) == 1:
                    overlay_row.index = feature_frame.index
                for column, value in row.items():
                    overlay_row[column] = value
                feature_frame = overlay_row
        feature_frame = feature_frame.reindex(columns=list(dict.fromkeys([*feature_frame.columns.tolist(), *FEATURE_COLUMNS])))
        prepared = _augment_model_features(feature_frame)
        prepared = _merge_market_features(prepared, daily)
        prepared = _merge_external_snapshot_features(prepared, daily, symbol=symbol)
        prepared = _append_market_regime_features(prepared)
        prepared = _append_market_resonance_features(prepared)
        return _stabilize_proxy_feature_matrix(prepared.reindex(columns=MODEL_FEATURE_COLUMNS))

    try:
        features = build_daily_features(daily)
    except Exception:
        return pd.DataFrame(columns=MODEL_FEATURE_COLUMNS)
    if features.empty:
        return pd.DataFrame(columns=MODEL_FEATURE_COLUMNS)
    prepared = _augment_model_features(features)
    prepared = _merge_market_features(prepared, daily)
    prepared = _merge_external_snapshot_features(prepared, daily, symbol=symbol)
    prepared = _append_market_regime_features(prepared)
    prepared = _append_market_resonance_features(prepared)
    latest = prepared.dropna(subset=FEATURE_COLUMNS, how="any").tail(1)
    if latest.empty:
        return pd.DataFrame(columns=MODEL_FEATURE_COLUMNS)
    return _stabilize_proxy_feature_matrix(latest.reindex(columns=MODEL_FEATURE_COLUMNS))


def _latest_feature_frame(daily: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    features = _prepare_live_feature_frame(daily, symbol=_resolve_symbol_from_daily(daily))
    if not features.empty:
        return features.reindex(columns=MODEL_FEATURE_COLUMNS).copy()

    available_cols = [column for column in MODEL_FEATURE_COLUMNS if column in dataset.columns]
    if available_cols:
        return dataset[available_cols].tail(1).reindex(columns=MODEL_FEATURE_COLUMNS).copy()
    return pd.DataFrame(columns=MODEL_FEATURE_COLUMNS)


def _balanced_sample_weight(y: pd.Series) -> np.ndarray:
    target = y.astype(int)
    counts = target.value_counts()
    if counts.empty or len(counts) < 2:
        return np.ones(len(target), dtype=float)
    total = float(len(target))
    class_weights = {klass: total / (len(counts) * float(count)) for klass, count in counts.items()}
    return target.map(class_weights).to_numpy(dtype=float)


def _recency_sample_weight(length: int, floor: float = 0.35, ceiling: float = 1.75) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=float)
    if length == 1:
        return np.array([1.0], dtype=float)
    base = np.linspace(-2.0, 2.2, length)
    curve = 1 / (1 + np.exp(-base))
    weights = floor + curve * (ceiling - floor)
    return weights.astype(float)


def _build_linear_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def _build_forest_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=160,
                    max_depth=6,
                    min_samples_leaf=6,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )


def _build_boost_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_depth=4,
                    learning_rate=0.06,
                    max_iter=180,
                    min_samples_leaf=18,
                    random_state=42,
                ),
            ),
        ]
    )


def _build_extra_proxy_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                CalibratedClassifierCV(
                    estimator=ExtraTreesClassifier(
                        n_estimators=260,
                        max_depth=8,
                        min_samples_leaf=4,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=1,
                    ),
                    method="sigmoid",
                    cv=3,
                ),
            ),
        ]
    )


def _fit_model_ensemble(train_df: pd.DataFrame) -> tuple[Pipeline, Pipeline, Pipeline]:
    x_train = train_df[MODEL_FEATURE_COLUMNS]
    y_train = train_df["target"]
    class_weight = _balanced_sample_weight(y_train)
    recency_weight = _recency_sample_weight(len(train_df))
    sample_weight = class_weight * recency_weight

    linear = _build_linear_pipeline()
    forest = _build_forest_pipeline()
    boost = _build_boost_pipeline()

    linear.fit(x_train, y_train, model__sample_weight=sample_weight)
    forest.fit(x_train, y_train, model__sample_weight=sample_weight)
    boost.fit(x_train, y_train, model__sample_weight=sample_weight)
    return linear, forest, boost


def _normalize_ensemble_weights(weights: np.ndarray | list[float] | tuple[float, ...] | None) -> np.ndarray:
    if weights is None:
        return ENSEMBLE_WEIGHTS.copy()
    vector = np.asarray(weights, dtype=float).reshape(-1)
    if vector.size != len(COMPONENT_MODEL_NAMES) or not np.isfinite(vector).all():
        return ENSEMBLE_WEIGHTS.copy()
    vector = np.clip(vector, 1e-6, None)
    return vector / vector.sum()


def _component_probability_breakdown(
    models: tuple[Pipeline, Pipeline, Pipeline],
    feature_frame: pd.DataFrame,
) -> dict[str, np.ndarray]:
    linear, forest, boost = models
    return {
        "logistic": linear.predict_proba(feature_frame)[:, 1],
        "forest": forest.predict_proba(feature_frame)[:, 1],
        "boost": boost.predict_proba(feature_frame)[:, 1],
    }


def _component_probability_matrix(component_probabilities: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack(
        [
            np.asarray(component_probabilities[name], dtype=float)
            for name in COMPONENT_MODEL_NAMES
        ]
    )


def _blend_component_matrix(
    component_matrix: np.ndarray,
    weights: np.ndarray | list[float] | tuple[float, ...] | None = None,
) -> np.ndarray:
    matrix = np.asarray(component_matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, len(COMPONENT_MODEL_NAMES))
    weight_vector = _normalize_ensemble_weights(weights)
    return np.average(matrix, axis=1, weights=weight_vector)


def _meta_calibration_matrix(feature_frame: pd.DataFrame | np.ndarray | None) -> np.ndarray:
    if feature_frame is None:
        return np.empty((0, len(META_CALIBRATION_COLUMNS)), dtype=float)
    if isinstance(feature_frame, pd.DataFrame):
        meta = feature_frame.reindex(columns=META_CALIBRATION_COLUMNS).copy()
        return meta.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    matrix = np.asarray(feature_frame, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, len(META_CALIBRATION_COLUMNS))
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def _calibrator_feature_matrix(
    raw_probabilities: np.ndarray,
    component_matrix: np.ndarray,
    meta_features: pd.DataFrame | np.ndarray | None = None,
) -> np.ndarray:
    raw = np.asarray(raw_probabilities, dtype=float).reshape(-1, 1)
    matrix = np.asarray(component_matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, len(COMPONENT_MODEL_NAMES))
    spread = (matrix.max(axis=1) - matrix.min(axis=1)).reshape(-1, 1)
    std = matrix.std(axis=1).reshape(-1, 1)
    meta = _meta_calibration_matrix(meta_features)
    if meta.shape[0] != matrix.shape[0]:
        meta = np.zeros((matrix.shape[0], len(META_CALIBRATION_COLUMNS)), dtype=float)
    return np.column_stack([raw, matrix, spread, std, meta])


def _blend_objective(y_true: np.ndarray, y_prob: np.ndarray, future_return: pd.Series) -> float:
    metrics = _build_backtest_metrics(y_true, y_prob, future_return)
    positive_rate = max(float(metrics.get("positive_rate", 0.0)), 1e-6)
    roc_auc = float(metrics.get("roc_auc", float("nan")))
    brier = float(metrics.get("brier", float("nan")))
    avg_abs_future = max(float(np.nanmean(np.abs(future_return))) if len(future_return) else 0.0, 0.012)

    auc_term = 0.0 if np.isnan(roc_auc) else _clip((roc_auc - 0.5) / 0.20, 0.0, 1.2)
    brier_term = 0.0 if np.isnan(brier) else _clip((0.26 - brier) / 0.12, 0.0, 1.1)
    precision_term = _clip(
        (float(metrics.get("precision", 0.0)) - positive_rate) / max(1.0 - positive_rate, 0.22),
        0.0,
        1.0,
    )
    top_bucket_term = _clip(
        (float(metrics.get("top_bucket_win_rate", 0.0)) - positive_rate) / max(1.0 - positive_rate, 0.22),
        0.0,
        1.0,
    )
    precision_gate_term = _clip(
        (float(metrics.get("precision_gate_precision", 0.0)) - positive_rate) / max(1.0 - positive_rate, 0.22),
        0.0,
        1.2,
    )
    precision_gate_support_term = _clip(float(metrics.get("precision_gate_support", 0.0)) / 12.0, 0.0, 1.0)
    precision_gate_reached_term = float(metrics.get("precision_target_reached", 0.0))
    return_term = _clip(float(metrics.get("top_bucket_return", 0.0)) / avg_abs_future, 0.0, 2.0)
    spread_term = _clip(float(metrics.get("spread_return", 0.0)) / avg_abs_future, 0.0, 2.0)
    return (
        auc_term * 0.24
        + brier_term * 0.18
        + precision_term * 0.18
        + top_bucket_term * 0.10
        + precision_gate_term * 0.16
        + precision_gate_support_term * 0.06
        + precision_gate_reached_term * 0.10
        + return_term * 0.08
        + spread_term * 0.10
    )


def _derive_dynamic_ensemble_weights(
    y_true: np.ndarray,
    component_matrix: np.ndarray,
    future_return: pd.Series,
) -> np.ndarray:
    matrix = np.asarray(component_matrix, dtype=float)
    valid_mask = np.isfinite(matrix).all(axis=1)
    if valid_mask.sum() < 40:
        return ENSEMBLE_WEIGHTS.copy()

    y = np.asarray(y_true, dtype=int)[valid_mask]
    future = future_return.iloc[np.flatnonzero(valid_mask)].reset_index(drop=True)
    matrix = matrix[valid_mask]
    priors = ENSEMBLE_WEIGHTS.copy()

    component_scores = np.array(
        [
            _blend_objective(y, matrix[:, idx], future)
            for idx in range(matrix.shape[1])
        ],
        dtype=float,
    )
    if np.isfinite(component_scores).any() and component_scores.sum() > 0:
        score_weights = component_scores / component_scores.sum()
    else:
        score_weights = priors.copy()

    best_weights = priors.copy()
    best_score = float("-inf")
    grid = np.arange(0.10, 0.81, 0.05)
    for logistic_weight in grid:
        for forest_weight in grid:
            boost_weight = 1.0 - logistic_weight - forest_weight
            if boost_weight < 0.10 or boost_weight > 0.80:
                continue
            weights = np.array([logistic_weight, forest_weight, boost_weight], dtype=float)
            weights = _normalize_ensemble_weights(weights)
            blended = _blend_component_matrix(matrix, weights)
            objective = _blend_objective(y, blended, future)
            objective += float(np.dot(weights, score_weights)) * 0.18
            objective -= float(np.square(weights - priors).sum()) * 0.08
            if objective > best_score:
                best_score = objective
                best_weights = weights

    smoothed = best_weights * 0.64 + score_weights * 0.18 + priors * 0.18
    smoothed = np.clip(smoothed, 0.12, 0.68)
    return _normalize_ensemble_weights(smoothed)


def _fit_probability_calibrator(
    y_true: np.ndarray,
    raw_probabilities: np.ndarray,
    component_matrix: np.ndarray,
    meta_features: pd.DataFrame | np.ndarray | None = None,
) -> Pipeline | None:
    raw = np.asarray(raw_probabilities, dtype=float)
    matrix = np.asarray(component_matrix, dtype=float)
    valid_mask = np.isfinite(raw) & np.isfinite(matrix).all(axis=1)
    if valid_mask.sum() < 80:
        return None

    y = np.asarray(y_true, dtype=int)[valid_mask]
    if np.unique(y).size < 2:
        return None

    meta_matrix = _meta_calibration_matrix(meta_features)
    if meta_matrix.shape[0] != matrix.shape[0]:
        meta_matrix = np.zeros((matrix.shape[0], len(META_CALIBRATION_COLUMNS)), dtype=float)
    features = _calibrator_feature_matrix(
        np.clip(raw[valid_mask], 1e-4, 1 - 1e-4),
        matrix[valid_mask],
        meta_matrix[valid_mask],
    )
    calibrator = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    sample_weight = _balanced_sample_weight(pd.Series(y)) * _recency_sample_weight(len(y), floor=0.55, ceiling=1.40)
    try:
        calibrator.fit(features, y, model__sample_weight=sample_weight)
    except Exception:
        return None
    return calibrator


def _apply_probability_calibrator(
    raw_probabilities: np.ndarray,
    component_matrix: np.ndarray,
    calibrator: Pipeline | None,
    meta_features: pd.DataFrame | np.ndarray | None = None,
    regime_calibrators: dict[str, Pipeline] | None = None,
) -> np.ndarray:
    raw = np.asarray(raw_probabilities, dtype=float)
    clipped_raw = np.clip(raw, 1e-4, 1 - 1e-4)
    if calibrator is None and not regime_calibrators:
        return clipped_raw
    try:
        features = _calibrator_feature_matrix(clipped_raw, component_matrix, meta_features)
        if regime_calibrators:
            labels = _regime_labels_from_feature_frame(meta_features)
            if len(labels) != len(clipped_raw):
                labels = np.full(len(clipped_raw), "rotation", dtype=object)
            calibrated = np.full(len(clipped_raw), np.nan, dtype=float)
            for label in np.unique(labels):
                label_mask = labels == label
                selected = regime_calibrators.get(str(label), calibrator)
                if selected is None:
                    calibrated[label_mask] = clipped_raw[label_mask]
                    continue
                calibrated[label_mask] = selected.predict_proba(features[label_mask])[:, 1]
            calibrated = np.where(np.isfinite(calibrated), calibrated, clipped_raw)
        else:
            calibrated = calibrator.predict_proba(features)[:, 1]
        return np.clip(np.asarray(calibrated, dtype=float), 1e-4, 1 - 1e-4)
    except Exception:
        return clipped_raw


def _regime_labels_from_feature_frame(feature_frame: pd.DataFrame | np.ndarray | None) -> np.ndarray:
    if isinstance(feature_frame, pd.DataFrame):
        if "market_regime_label" in feature_frame.columns:
            labels = feature_frame["market_regime_label"].astype(str).to_numpy(dtype=object)
            valid = np.isin(labels, REGIME_LABELS)
            labels[~valid] = "rotation"
            return labels
        one_hot_columns = [
            "market_regime_trend",
            "market_regime_rebound",
            "market_regime_rotation",
            "market_regime_defense",
        ]
        if all(column in feature_frame.columns for column in one_hot_columns):
            matrix = feature_frame[one_hot_columns].to_numpy(dtype=float)
            return np.array(REGIME_LABELS, dtype=object)[matrix.argmax(axis=1)]
        return np.full(len(feature_frame), "rotation", dtype=object)
    if feature_frame is None:
        return np.array([], dtype=object)
    matrix = np.asarray(feature_frame)
    return np.full(len(matrix), "rotation", dtype=object)


def _fit_regime_calibrators(
    y_true: np.ndarray,
    raw_probabilities: np.ndarray,
    component_matrix: np.ndarray,
    meta_features: pd.DataFrame | np.ndarray | None = None,
) -> dict[str, Pipeline]:
    if not isinstance(meta_features, pd.DataFrame):
        return {}

    labels = _regime_labels_from_feature_frame(meta_features)
    valid_mask = np.isfinite(raw_probabilities) & np.isfinite(component_matrix).all(axis=1)
    calibrators: dict[str, Pipeline] = {}
    for label in REGIME_LABELS:
        regime_mask = valid_mask & (labels == label)
        if regime_mask.sum() < 120:
            continue
        y_subset = np.asarray(y_true, dtype=int)[regime_mask]
        if np.unique(y_subset).size < 2:
            continue
        calibrator = _fit_probability_calibrator(
            y_subset,
            np.asarray(raw_probabilities, dtype=float)[regime_mask],
            np.asarray(component_matrix, dtype=float)[regime_mask],
            meta_features.loc[regime_mask, MODEL_FEATURE_COLUMNS],
        )
        if calibrator is not None:
            calibrators[label] = calibrator
    return calibrators


def _build_oof_component_probabilities(dataset: pd.DataFrame, n_splits: int) -> np.ndarray:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    oof_component_probabilities = np.full((len(dataset), len(COMPONENT_MODEL_NAMES)), np.nan, dtype=float)

    for train_idx, test_idx in splitter.split(dataset):
        train_df = dataset.iloc[train_idx]
        test_df = dataset.iloc[test_idx]
        if train_df["target"].nunique() < 2:
            fallback_probability = float(train_df["target"].mean())
            oof_component_probabilities[test_idx, :] = fallback_probability
            continue
        models = _fit_model_ensemble(train_df)
        fold_breakdown = _component_probability_breakdown(models, test_df[MODEL_FEATURE_COLUMNS])
        fold_matrix = _component_probability_matrix(fold_breakdown)
        oof_component_probabilities[test_idx, :] = fold_matrix
    return oof_component_probabilities


def _ensemble_probability(
    models: tuple[Pipeline, Pipeline, Pipeline],
    feature_frame: pd.DataFrame,
    weights: np.ndarray | list[float] | tuple[float, ...] | None = None,
    calibrator: Pipeline | None = None,
    calibration_feature_frame: pd.DataFrame | np.ndarray | None = None,
    regime_calibrators: dict[str, Pipeline] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    breakdown = _component_probability_breakdown(models, feature_frame)
    component_matrix = _component_probability_matrix(breakdown)
    raw_ensemble = _blend_component_matrix(component_matrix, weights=weights)
    ensemble = _apply_probability_calibrator(
        raw_ensemble,
        component_matrix,
        calibrator,
        calibration_feature_frame,
        regime_calibrators=regime_calibrators,
    )
    return ensemble, breakdown


def _build_signal_breakdown(latest_features: pd.Series | dict[str, float]) -> dict[str, float]:
    row = latest_features if isinstance(latest_features, dict) else latest_features.to_dict()
    trend_score = _clip(
        52
        + float(row.get("ret_20", 0.0)) * 220
        + float(row.get("ret_60", 0.0)) * 150
        + float(row.get("close_vs_ma20", 0.0)) * 240
        + float(row.get("ma20_slope_5", 0.0)) * 820
    )
    breakout_score = _clip(
        50
        + float(row.get("breakout_distance_20", 0.0)) * 520
        + float(row.get("range_position_20", 0.5)) * 32
        + (float(row.get("volume_ratio_5", 1.0)) - 1.0) * 28
    )
    pullback_score = _clip(
        48
        - abs(float(row.get("pullback_to_breakout_20", 0.0))) * 420
        + float(row.get("lower_shadow_ratio", 0.0)) * 680
        + float(row.get("body_ratio", 0.0)) * 24
    )
    risk_score = _clip(
        26
        + float(row.get("upper_shadow_ratio", 0.0)) * 1400
        + float(row.get("volatility_10", 0.0)) * 1900
        + max(float(row.get("close_vs_ma20", 0.0)) - 0.10, 0.0) * 360
    )
    launch_readiness_score = _clip(float(row.get("launch_readiness", 50.0)), 0.0, 100.0)
    market_resonance_score = _clip(float(row.get("market_resonance", 50.0)), 0.0, 100.0)
    breakout_quality = _clip(
        breakout_score * 0.58
        + (50.0 + float(row.get("relative_strength_5", 0.0)) * 180) * 0.12
        + market_resonance_score * 0.16
        + (100.0 - risk_score) * 0.14,
        0.0,
        100.0,
    )
    resonance_quality = _clip(
        market_resonance_score * 0.54
        + (50.0 + float(row.get("market_ret_20", 0.0)) * 170) * 0.20
        + (50.0 + float(row.get("relative_strength_20", 0.0)) * 150) * 0.16
        + (100.0 - risk_score) * 0.10,
        0.0,
        100.0,
    )
    risk_of_late_entry = _clip(
        risk_score * 0.52
        + max(float(row.get("ret_20", 0.0)) - 0.22, 0.0) * 160
        + max(float(row.get("range_position_20", 0.5)) - 0.86, 0.0) * 70
        + max(float(row.get("volume_ratio_5", 1.0)) - 1.8, 0.0) * 18
        - max(launch_readiness_score - 55.0, 0.0) * 0.20,
        0.0,
        100.0,
    )
    if risk_of_late_entry >= 68.0:
        launch_phase_label = "已走远"
    elif breakout_quality < 48.0 or resonance_quality < 50.0:
        launch_phase_label = "伪突破"
    elif launch_readiness_score >= 64.0 and risk_of_late_entry < 56.0:
        launch_phase_label = "刚启动"
    else:
        launch_phase_label = "观察"
    return {
        "trend_score": round(trend_score, 2),
        "breakout_score": round(breakout_score, 2),
        "pullback_score": round(pullback_score, 2),
        "risk_score": round(risk_score, 2),
        "launch_readiness_score": round(launch_readiness_score, 2),
        "market_resonance_score": round(market_resonance_score, 2),
        "launch_readiness": round(launch_readiness_score, 2),
        "breakout_quality": round(breakout_quality, 2),
        "resonance_quality": round(resonance_quality, 2),
        "risk_of_late_entry": round(risk_of_late_entry, 2),
        "launch_phase_label": launch_phase_label,
    }


def _summarize_quality(metrics: dict[str, float]) -> tuple[str, str]:
    sample_size = float(metrics.get("sample_size", 0.0))
    roc_auc = float(metrics.get("roc_auc", float("nan")))
    brier = float(metrics.get("brier", float("nan")))
    top_bucket_return = float(metrics.get("top_bucket_return", 0.0))
    precision_gate_precision = float(metrics.get("precision_gate_precision", 0.0))
    precision_gate_support = float(metrics.get("precision_gate_support", 0.0))
    precision_target_reached = bool(metrics.get("precision_target_reached", 0.0))

    quality_score = 0.0
    if not np.isnan(roc_auc):
        quality_score += _clip((roc_auc - 0.5) * 140, 0, 24)
    if not np.isnan(brier):
        quality_score += _clip((0.28 - brier) * 180, 0, 20)
    quality_score += _clip(top_bucket_return * 220, 0, 18)
    quality_score += _clip(sample_size / 18, 0, 18)
    quality_score += _clip((precision_gate_precision - 0.5) * 50, 0, 24)
    quality_score += _clip(precision_gate_support, 0, 10)
    if precision_target_reached:
        quality_score += 12

    if precision_target_reached and precision_gate_support >= 6:
        return "高精度放行", "模型已经找到历史上能稳定达到 90% 以上命中率的高置信区间，当前更适合作为严格放行器。"
    if quality_score >= 46:
        return "较可靠", "本地回测表现较稳，可以把它当成页面排序和预案的核心参考之一。"
    if quality_score >= 28:
        return "可参考", "本地回测有一定辨识度，但仍建议和阶段、分时、资金流一起确认。"
    return "偏谨慎", "本地样本对未来涨幅的辨识度一般，更适合作为辅助过滤器。"


def _global_dataset_cache_path(
    horizon_days: int,
    positive_return: float,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_return = int(positive_return * 10000)
    return CACHE_DIR / (
        f"global_market_dataset_v{MODEL_SCHEMA_VERSION}_"
        f"h{horizon_days}_r{safe_return}_"
        f"{train_start.replace('-', '')}_{train_end.replace('-', '')}_"
        f"{test_start.replace('-', '')}_{test_end.replace('-', '')}.pkl"
    )


def _global_dataset_partial_path(
    horizon_days: int,
    positive_return: float,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> Path:
    return _global_dataset_cache_path(
        horizon_days,
        positive_return,
        train_start,
        train_end,
        test_start,
        test_end,
    ).with_suffix(".partial.pkl")


def _candidate_market_dataset_paths(
    horizon_days: int,
    positive_return: float,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> list[Path]:
    current_final = _global_dataset_cache_path(horizon_days, positive_return, train_start, train_end, test_start, test_end)
    current_partial = _global_dataset_partial_path(horizon_days, positive_return, train_start, train_end, test_start, test_end)
    paths: list[Path] = [current_final, current_partial]

    safe_return = int(positive_return * 10000)
    suffix = (
        f"h{horizon_days}_r{safe_return}_"
        f"{train_start.replace('-', '')}_{train_end.replace('-', '')}_"
        f"{test_start.replace('-', '')}_{test_end.replace('-', '')}"
    )
    legacy_patterns = (
        f"global_market_dataset_v*_{suffix}.pkl",
        f"global_market_dataset_v*_{suffix}.partial.pkl",
        f"global_market_dataset_{suffix}.pkl",
        f"global_market_dataset_{suffix}.partial.pkl",
    )
    legacy_paths: list[Path] = []
    for pattern in legacy_patterns:
        legacy_paths.extend(CACHE_DIR.glob(pattern))
    legacy_paths = sorted(
        {path for path in legacy_paths if path not in paths},
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )
    paths.extend(legacy_paths)
    return paths


def _global_model_cache_path(
    horizon_days: int,
    positive_return: float,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_return = int(positive_return * 10000)
    return CACHE_DIR / (
        f"global_market_model_v{MODEL_SCHEMA_VERSION}_"
        f"h{horizon_days}_r{safe_return}_"
        f"{train_start.replace('-', '')}_{train_end.replace('-', '')}_"
        f"{test_start.replace('-', '')}_{test_end.replace('-', '')}.pkl"
    )


def _global_proxy_model_cache_path(
    horizon_days: int,
    positive_return: float,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_return = int(positive_return * 10000)
    return CACHE_DIR / (
        f"global_market_proxy_v{MODEL_SCHEMA_VERSION}_"
        f"h{horizon_days}_r{safe_return}_"
        f"{train_start.replace('-', '')}_{train_end.replace('-', '')}_"
        f"{test_start.replace('-', '')}_{test_end.replace('-', '')}.pkl"
    )


def load_cached_market_wide_model(
    horizon_days: int = 5,
    positive_return: float = 0.03,
    train_start: str = GLOBAL_MODEL_TRAIN_START,
    train_end: str = GLOBAL_MODEL_TRAIN_END,
    test_start: str = GLOBAL_MODEL_TEST_START,
    test_end: str = GLOBAL_MODEL_TEST_END,
) -> MarketWideModel | None:
    cache_path = _global_model_cache_path(horizon_days, positive_return, train_start, train_end, test_start, test_end)
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)
        if isinstance(cached, MarketWideModel):
            return cached
    except Exception:
        return None
    return None


def load_cached_market_proxy_model(
    horizon_days: int = 5,
    positive_return: float = 0.03,
    train_start: str = GLOBAL_MODEL_TRAIN_START,
    train_end: str = GLOBAL_MODEL_TRAIN_END,
    test_start: str = GLOBAL_MODEL_TEST_START,
    test_end: str = GLOBAL_MODEL_TEST_END,
) -> MarketProxyModel | None:
    cache_path = _global_proxy_model_cache_path(horizon_days, positive_return, train_start, train_end, test_start, test_end)
    if not cache_path.exists():
        return None
    source_paths = [
        Path(__file__),
        *_candidate_market_dataset_paths(horizon_days, positive_return, train_start, train_end, test_start, test_end),
    ]
    proxy_mtime = cache_path.stat().st_mtime
    for source_path in source_paths:
        if source_path.exists() and source_path.stat().st_mtime > proxy_mtime:
            return None
    try:
        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)
        if isinstance(cached, MarketProxyModel):
            return cached
    except Exception:
        return None
    return None


def get_market_wide_model_status(
    horizon_days: int = 5,
    positive_return: float = 0.03,
    train_start: str = GLOBAL_MODEL_TRAIN_START,
    train_end: str = GLOBAL_MODEL_TRAIN_END,
    test_start: str = GLOBAL_MODEL_TEST_START,
    test_end: str = GLOBAL_MODEL_TEST_END,
) -> dict[str, object]:
    cache_path = _global_model_cache_path(horizon_days, positive_return, train_start, train_end, test_start, test_end)
    partial_path = _global_dataset_partial_path(
        horizon_days,
        positive_return,
        train_start,
        train_end,
        test_start,
        test_end,
    )
    status: dict[str, object] = {
        "model_ready": cache_path.exists(),
        "model_path": str(cache_path),
        "partial_ready": partial_path.exists(),
        "partial_path": str(partial_path),
        "completed_symbol_count": 0,
        "partial_symbol_count": 0,
        "partial_row_count": 0,
    }
    if partial_path.exists():
        try:
            with partial_path.open("rb") as handle:
                payload = pickle.load(handle)
            meta = payload.get("meta", {})
            status["completed_symbol_count"] = int(meta.get("completed_symbol_count", 0))
            status["partial_symbol_count"] = int(meta.get("symbol_count", 0))
            status["partial_row_count"] = int(meta.get("row_count", 0))
        except Exception:
            pass
    return status


def _load_partial_market_dataset(
    horizon_days: int,
    positive_return: float,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> pd.DataFrame:
    for path in _candidate_market_dataset_paths(
        horizon_days,
        positive_return,
        train_start,
        train_end,
        test_start,
        test_end,
    ):
        if not path.exists():
            continue
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception:
            continue
        dataset = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(dataset, pd.DataFrame) and not dataset.empty:
            return dataset.copy()
    return pd.DataFrame()


def _prepare_proxy_training_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return dataset.copy()

    frame = dataset.copy()
    available_columns = [column for column in MODEL_FEATURE_COLUMNS if column in frame.columns]
    required_columns = [*available_columns, "target"]
    if "future_return" in frame.columns:
        required_columns.append("future_return")
    if "signal_date" in frame.columns:
        required_columns.append("signal_date")
    frame = frame[required_columns].copy()
    frame = frame.dropna(subset=["target"])
    if "signal_date" in frame.columns:
        frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce")
        frame = frame.sort_values("signal_date")
    if len(frame) > MARKET_PROXY_SAMPLE_SIZE:
        frame = frame.tail(MARKET_PROXY_SAMPLE_SIZE).copy()
    return frame.reset_index(drop=True)


def _stabilize_proxy_feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    stabilized = frame.copy().replace([np.inf, -np.inf], np.nan)
    for column in stabilized.columns:
        if stabilized[column].isna().all():
            stabilized[column] = 0.0
    return stabilized


def _proxy_candidate_builders() -> dict[str, callable]:
    return {
        "linear": _build_linear_pipeline,
        "forest": _build_forest_pipeline,
        "boost": _build_boost_pipeline,
        "extra_calibrated": _build_extra_proxy_pipeline,
    }


def _proxy_validation_split(
    dataset: pd.DataFrame,
    *,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if dataset.empty or "target" not in dataset.columns:
        empty = dataset.iloc[0:0].copy()
        return empty, empty, "empty"

    frame = dataset.copy()
    if "signal_date" in frame.columns:
        frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce")
        frame = frame.dropna(subset=["signal_date"]).sort_values("signal_date").reset_index(drop=True)
        global_train = frame.loc[_date_slice_mask(frame["signal_date"], train_start, train_end)].copy()
        global_test = frame.loc[_date_slice_mask(frame["signal_date"], test_start, test_end)].copy()
        if (
            len(global_train) >= 200
            and len(global_test) >= 40
            and global_train["target"].nunique() >= 2
            and global_test["target"].nunique() >= 2
        ):
            return (
                global_train.reset_index(drop=True),
                global_test.reset_index(drop=True),
                "global_train_test_split",
            )
    else:
        frame = frame.reset_index(drop=True)

    if len(frame) < 160:
        empty = frame.iloc[0:0].copy()
        return frame.reset_index(drop=True), empty, "insufficient_validation_split"

    split_idx = int(len(frame) * 0.80)
    split_idx = max(80, min(split_idx, len(frame) - 40))
    train_frame = frame.iloc[:split_idx].copy()
    validation_frame = frame.iloc[split_idx:].copy()
    if (
        train_frame.empty
        or validation_frame.empty
        or train_frame["target"].nunique() < 2
        or validation_frame["target"].nunique() < 2
    ):
        empty = frame.iloc[0:0].copy()
        return frame.reset_index(drop=True), empty, "insufficient_validation_split"
    return train_frame.reset_index(drop=True), validation_frame.reset_index(drop=True), "tail_holdout_split"


def _fit_proxy_candidate_model(train_df: pd.DataFrame, builder) -> Pipeline:
    x_train = _stabilize_proxy_feature_matrix(train_df.reindex(columns=MODEL_FEATURE_COLUMNS))
    y_train = train_df["target"].astype(int)
    sample_weight = _balanced_sample_weight(y_train) * _recency_sample_weight(len(train_df))
    model = builder()
    try:
        model.fit(x_train, y_train, model__sample_weight=sample_weight)
    except TypeError:
        model.fit(x_train, y_train)
    return model


def _evaluate_proxy_candidate_model(model: Pipeline, validation_df: pd.DataFrame) -> dict[str, float]:
    if validation_df.empty or "target" not in validation_df.columns or validation_df["target"].nunique() < 2:
        return {
            "sample_size": float(len(validation_df)),
            "positive_rate": float(validation_df["target"].mean()) if "target" in validation_df else 0.0,
            "roc_auc": float("nan"),
            "brier": float("nan"),
            "top_bucket_return": 0.0,
            "spread_return": 0.0,
            "precision_gate_threshold": 1.0,
            "precision_gate_precision": 0.0,
            "precision_gate_support": 0.0,
            "precision_target": 0.90,
            "precision_target_reached": 0.0,
            "objective": float("-inf"),
        }

    x_validation = _stabilize_proxy_feature_matrix(validation_df.reindex(columns=MODEL_FEATURE_COLUMNS))
    y_validation = validation_df["target"].astype(int).to_numpy(dtype=int)
    future_return = (
        validation_df["future_return"].astype(float).reset_index(drop=True)
        if "future_return" in validation_df.columns
        else pd.Series(np.zeros(len(validation_df), dtype=float))
    )
    probability = model.predict_proba(x_validation)[:, 1]
    probability, _ = _apply_incremental_probability_upgrade(probability, validation_df.reindex(columns=MODEL_FEATURE_COLUMNS))
    metrics = _build_backtest_metrics(y_validation, probability, future_return)
    metrics.update(_recent_tail_backtest_profile(y_validation, probability, future_return))
    metrics["objective"] = float(_blend_objective(y_validation, probability, future_return))
    return metrics


def _select_market_proxy_candidate(
    proxy_dataset: pd.DataFrame,
    *,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> tuple[str, dict[str, float], str]:
    train_frame, validation_frame, split_label = _proxy_validation_split(
        proxy_dataset,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )
    candidate_builders = _proxy_candidate_builders()
    best_name = "boost"
    best_metrics: dict[str, float] = {}
    best_objective = float("-inf")

    for name, builder in candidate_builders.items():
        try:
            model = _fit_proxy_candidate_model(train_frame, builder)
            metrics = _evaluate_proxy_candidate_model(model, validation_frame)
        except Exception:
            continue
        objective = float(metrics.get("objective", float("-inf")))
        if not best_metrics or objective > best_objective:
            best_name = name
            best_metrics = metrics
            best_objective = objective

    if not best_metrics:
        best_metrics = {
            "sample_size": float(len(validation_frame)),
            "positive_rate": float(validation_frame["target"].mean()) if not validation_frame.empty else 0.0,
            "roc_auc": float("nan"),
            "brier": float("nan"),
            "top_bucket_return": 0.0,
            "top_bucket_win_rate": 0.0,
            "low_bucket_return": 0.0,
            "spread_return": 0.0,
            "precision_target": 0.90,
            "precision_gate_threshold": 1.0,
            "precision_gate_precision": 0.0,
            "precision_gate_support": 0.0,
            "precision_gate_return": 0.0,
            "precision_target_reached": 0.0,
            "objective": 0.0,
        }
    best_metrics["proxy_validation_objective"] = float(best_objective if np.isfinite(best_objective) else 0.0)
    return best_name, best_metrics, split_label


def _proxy_validation_summary(candidate_name: str, metrics: dict[str, float], split_label: str) -> str:
    candidate_labels = {
        "linear": "线性代理",
        "forest": "随机森林代理",
        "boost": "梯度提升代理",
        "extra_calibrated": "校准 ExtraTrees 代理",
    }
    split_labels = {
        "global_train_test_split": "2025 训练 / 2026Q1 验证",
        "tail_holdout_split": "最近 20% 时序验证",
        "insufficient_validation_split": "有限样本快速训练",
        "empty": "空样本",
    }
    roc_auc = float(metrics.get("roc_auc", float("nan")))
    top_bucket_return = float(metrics.get("top_bucket_return", 0.0))
    precision_gate_precision = float(metrics.get("precision_gate_precision", 0.0))
    precision_gate_support = int(metrics.get("precision_gate_support", 0.0) or 0)
    auc_text = "--" if np.isnan(roc_auc) else f"{roc_auc:.3f}"
    return (
        f"代理模型采用 {candidate_labels.get(candidate_name, candidate_name)}，基于"
        f"{split_labels.get(split_label, split_label)} 进行回放筛选；"
        f"验证 AUC {auc_text}，高分桶平均未来收益 {top_bucket_return * 100:.2f}%，"
        f"精度门槛区间命中率 {precision_gate_precision * 100:.1f}%（样本 {precision_gate_support}）。"
    )


@lru_cache(maxsize=4)
def load_market_proxy_model(
    horizon_days: int = 5,
    positive_return: float = 0.03,
    train_start: str = GLOBAL_MODEL_TRAIN_START,
    train_end: str = GLOBAL_MODEL_TRAIN_END,
    test_start: str = GLOBAL_MODEL_TEST_START,
    test_end: str = GLOBAL_MODEL_TEST_END,
) -> MarketProxyModel | None:
    cached = load_cached_market_proxy_model(
        horizon_days=horizon_days,
        positive_return=positive_return,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )
    if cached is not None:
        return cached

    dataset = _load_partial_market_dataset(
        horizon_days=horizon_days,
        positive_return=positive_return,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )
    if dataset.empty or "target" not in dataset.columns or dataset["target"].nunique() < 2:
        return None

    proxy_dataset = _prepare_proxy_training_dataset(dataset)
    if proxy_dataset.empty or proxy_dataset["target"].nunique() < 2:
        return None

    candidate_name, validation_metrics, split_label = _select_market_proxy_candidate(
        proxy_dataset,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )
    proxy_model = _fit_proxy_candidate_model(proxy_dataset, _proxy_candidate_builders()[candidate_name])
    y_train = proxy_dataset["target"].astype(int)
    metrics_for_cache = {
        key: float(value)
        for key, value in validation_metrics.items()
        if isinstance(value, (int, float, np.floating, np.integer))
    }
    result = MarketProxyModel(
        fitted_model=proxy_model,
        sample_size=int(len(proxy_dataset)),
        positive_rate=float(y_train.mean()),
        source_label=f"partial_market_dataset:{candidate_name}",
        validation_metrics=metrics_for_cache,
        validation_summary=_proxy_validation_summary(candidate_name, metrics_for_cache, split_label),
        candidate_name=candidate_name,
    )
    cache_path = _global_proxy_model_cache_path(
        horizon_days,
        positive_return,
        train_start,
        train_end,
        test_start,
        test_end,
    )
    try:
        with cache_path.open("wb") as handle:
            pickle.dump(result, handle)
    except Exception:
        pass
    return result


def _date_slice_mask(index_like, start: str, end: str) -> np.ndarray:
    dates = pd.to_datetime(index_like, errors="coerce")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return np.asarray((dates >= start_ts) & (dates <= end_ts), dtype=bool)


def _market_history_end_for_test(horizon_days: int, test_end: str) -> str:
    end_ts = pd.Timestamp(test_end) + pd.offsets.BDay(max(horizon_days + 1, 5))
    upper_bound = pd.Timestamp(GLOBAL_MODEL_HISTORY_END)
    return min(end_ts, upper_bound).strftime("%Y-%m-%d")


def _build_market_symbol_frame(
    symbol: str,
    name: str,
    horizon_days: int,
    positive_return: float,
    signal_start: str,
    signal_end: str,
) -> pd.DataFrame | None:
    try:
        daily = fetch_daily_history(
            symbol=symbol,
            start_date=GLOBAL_MODEL_HISTORY_START.replace("-", ""),
            end_date=_market_history_end_for_test(horizon_days, signal_end).replace("-", ""),
        )
    except Exception:
        return None
    if daily.empty or len(daily) < 140:
        return None

    dataset = _prepare_training_dataset(
        daily,
        horizon_days=horizon_days,
        positive_return=positive_return,
        symbol=symbol,
    )
    if dataset.empty:
        return None

    mask = _date_slice_mask(dataset.index, signal_start, signal_end)
    dataset = dataset.loc[mask].copy()
    if dataset.empty:
        return None

    dataset["symbol"] = symbol
    dataset["name"] = name
    dataset["signal_date"] = pd.to_datetime(dataset.index, errors="coerce")
    dataset = dataset.dropna(subset=["signal_date", "target", "future_return"])
    if dataset.empty:
        return None

    float_columns = [column for column in MODEL_FEATURE_COLUMNS + ["future_return"] if column in dataset.columns]
    for column in float_columns:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce").astype("float32")
    dataset["target"] = dataset["target"].astype("int8")
    return dataset.reset_index(drop=True)


def _build_historical_market_universe(signal_start: str, signal_end: str) -> tuple[pd.DataFrame, bool]:
    try:
        stock_basic = fetch_tushare_stock_basic_all_statuses()
        historical = filter_historical_a_share_universe_window(stock_basic, signal_start, signal_end)
        if not historical.empty and {"symbol", "name"}.issubset(historical.columns):
            universe = historical[["symbol", "name"]].drop_duplicates("symbol").copy()
            universe["symbol"] = universe["symbol"].astype(str).str.zfill(6)
            universe["name"] = universe["name"].astype(str)
            return universe.sort_values("symbol").reset_index(drop=True), True
    except Exception:
        pass
    universe = fetch_a_share_universe()[["symbol", "name"]].drop_duplicates("symbol").copy()
    universe["symbol"] = universe["symbol"].astype(str).str.zfill(6)
    universe["name"] = universe["name"].astype(str)
    return universe.sort_values("symbol").reset_index(drop=True), False


def build_market_wide_dataset(
    horizon_days: int = 5,
    positive_return: float = 0.03,
    train_start: str = GLOBAL_MODEL_TRAIN_START,
    train_end: str = GLOBAL_MODEL_TRAIN_END,
    test_start: str = GLOBAL_MODEL_TEST_START,
    test_end: str = GLOBAL_MODEL_TEST_END,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    cache_path = _global_dataset_cache_path(horizon_days, positive_return, train_start, train_end, test_start, test_end)
    partial_path = _global_dataset_partial_path(
        horizon_days,
        positive_return,
        train_start,
        train_end,
        test_start,
        test_end,
    )
    if not refresh and cache_path.exists():
        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            dataset = payload.get("data")
            if isinstance(dataset, pd.DataFrame) and not dataset.empty:
                meta = payload.get("meta", {})
                dataset.attrs["point_in_time_universe"] = bool(meta.get("point_in_time_universe", False))
                dataset.attrs["universe_source"] = str(meta.get("universe_source", ""))
                dataset.attrs["source_universe_size"] = int(meta.get("source_universe_size", 0) or 0)
                return dataset
        except Exception:
            pass

    signal_start = train_start
    signal_end = test_end
    universe, point_in_time_universe = _build_historical_market_universe(signal_start, signal_end)
    frames: list[pd.DataFrame] = []
    completed_symbols: set[str] = set()
    if not refresh and partial_path.exists():
        try:
            with partial_path.open("rb") as handle:
                partial_payload = pickle.load(handle)
            partial_dataset = partial_payload.get("data")
            partial_symbols = partial_payload.get("completed_symbols", [])
            if isinstance(partial_dataset, pd.DataFrame) and not partial_dataset.empty:
                frames.append(partial_dataset)
            completed_symbols = {str(symbol) for symbol in partial_symbols}
        except Exception:
            frames = []
            completed_symbols = set()

    def _checkpoint_dataset(current_frames: list[pd.DataFrame], current_completed_symbols: set[str]) -> None:
        if not current_frames:
            return
        dataset_so_far = pd.concat(current_frames, ignore_index=True)
        payload = {
            "meta": {
                "horizon_days": horizon_days,
                "positive_return": positive_return,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "row_count": int(len(dataset_so_far)),
                "symbol_count": int(dataset_so_far["symbol"].nunique()) if "symbol" in dataset_so_far.columns else 0,
                "completed_symbol_count": int(len(current_completed_symbols)),
                "source_universe_size": int(len(universe)),
                "point_in_time_universe": bool(point_in_time_universe),
                "universe_source": "tushare_stock_basic_all_statuses" if point_in_time_universe else "current_a_share_universe_fallback",
            },
            "completed_symbols": sorted(current_completed_symbols),
            "data": dataset_so_far,
        }
        with partial_path.open("wb") as handle:
            pickle.dump(payload, handle)

    if GLOBAL_MODEL_MAX_WORKERS <= 1:
        processed = 0
        for row in universe.itertuples(index=False):
            symbol = str(row.symbol)
            if symbol in completed_symbols:
                continue
            frame = _build_market_symbol_frame(
                symbol,
                str(row.name),
                horizon_days,
                positive_return,
                signal_start,
                signal_end,
            )
            if frame is not None and not frame.empty:
                frames.append(frame)
            completed_symbols.add(symbol)
            processed += 1
            if processed % GLOBAL_MODEL_CHECKPOINT_EVERY == 0:
                _checkpoint_dataset(frames, completed_symbols)
    else:
        with ThreadPoolExecutor(max_workers=GLOBAL_MODEL_MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    _build_market_symbol_frame,
                    str(row.symbol),
                    str(row.name),
                    horizon_days,
                    positive_return,
                    signal_start,
                    signal_end,
                ): str(row.symbol)
                for row in universe.itertuples(index=False)
            }
            for future in as_completed(futures):
                frame = future.result()
                if frame is not None and not frame.empty:
                    frames.append(frame)

    if not frames:
        return pd.DataFrame()

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.sort_values(["signal_date", "symbol"]).reset_index(drop=True)
    dataset.attrs["point_in_time_universe"] = bool(point_in_time_universe)
    dataset.attrs["universe_source"] = "tushare_stock_basic_all_statuses" if point_in_time_universe else "current_a_share_universe_fallback"
    dataset.attrs["source_universe_size"] = int(len(universe))
    payload = {
        "meta": {
            "horizon_days": horizon_days,
            "positive_return": positive_return,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "row_count": int(len(dataset)),
            "symbol_count": int(dataset["symbol"].nunique()),
            "source_universe_size": int(len(universe)),
            "point_in_time_universe": bool(point_in_time_universe),
            "universe_source": str(dataset.attrs["universe_source"]),
        },
        "data": dataset,
    }
    with cache_path.open("wb") as handle:
        pickle.dump(payload, handle)
    if partial_path.exists():
        try:
            partial_path.unlink()
        except OSError:
            pass
    return dataset


def _select_precision_gate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    future_return: pd.Series,
    *,
    target_precision: float = 0.90,
) -> dict[str, float]:
    if len(y_prob) == 0:
        return {
            "precision_target": target_precision,
            "precision_gate_threshold": 1.0,
            "precision_gate_precision": 0.0,
            "precision_gate_support": 0.0,
            "precision_gate_return": 0.0,
            "precision_target_reached": 0.0,
        }

    support_floor = max(4, min(30, len(y_prob) // 25 if len(y_prob) >= 25 else 4))
    threshold_rows: list[dict[str, float]] = []
    for threshold in np.arange(0.55, 0.96, 0.05):
        mask = y_prob >= float(threshold)
        support = int(mask.sum())
        if support == 0:
            continue
        precision = float(np.mean(y_true[mask]))
        avg_return = float(future_return[mask].mean()) if support else 0.0
        threshold_rows.append(
            {
                "threshold": float(round(float(threshold), 2)),
                "precision": precision,
                "support": float(support),
                "avg_return": avg_return,
            }
        )

    if not threshold_rows:
        return {
            "precision_target": target_precision,
            "precision_gate_threshold": 1.0,
            "precision_gate_precision": 0.0,
            "precision_gate_support": 0.0,
            "precision_gate_return": 0.0,
            "precision_target_reached": 0.0,
        }

    table = pd.DataFrame(threshold_rows)
    eligible = table[(table["precision"] >= target_precision) & (table["support"] >= support_floor)]
    if not eligible.empty:
        selected = eligible.sort_values(
            ["support", "precision", "avg_return", "threshold"],
            ascending=[False, False, False, True],
        ).iloc[0]
        reached = 1.0
    else:
        fallback = table[table["support"] >= support_floor]
        if fallback.empty:
            fallback = table
        selected = fallback.sort_values(
            ["precision", "support", "avg_return", "threshold"],
            ascending=[False, False, False, True],
        ).iloc[0]
        reached = 0.0

    return {
        "precision_target": target_precision,
        "precision_gate_threshold": float(selected["threshold"]),
        "precision_gate_precision": float(selected["precision"]),
        "precision_gate_support": float(selected["support"]),
        "precision_gate_return": float(selected["avg_return"]),
        "precision_target_reached": reached,
    }


def _build_backtest_metrics(y_true: np.ndarray, y_prob: np.ndarray, future_return: pd.Series) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(_safe_roc_auc_score(y_true, y_prob)),
        "brier": float(_safe_brier_score(y_true, y_prob)),
        "sample_size": float(len(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else 0.0,
    }

    if len(y_prob) >= 5:
        high_cutoff = float(np.nanquantile(y_prob, 0.80))
        low_cutoff = float(np.nanquantile(y_prob, 0.20))
    else:
        high_cutoff = 0.60
        low_cutoff = 0.40

    high_mask = y_prob >= high_cutoff
    low_mask = y_prob <= low_cutoff
    metrics["top_bucket_return"] = float(future_return[high_mask].mean()) if high_mask.any() else 0.0
    metrics["top_bucket_win_rate"] = float(np.mean(y_true[high_mask])) if high_mask.any() else 0.0
    metrics["low_bucket_return"] = float(future_return[low_mask].mean()) if low_mask.any() else 0.0
    metrics["spread_return"] = metrics["top_bucket_return"] - metrics["low_bucket_return"]
    metrics.update(_select_precision_gate(y_true, y_prob, future_return))
    return metrics


def _recent_tail_backtest_profile(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    future_return: pd.Series | np.ndarray,
    *,
    window: int = RECENT_BACKTEST_WINDOW,
    min_samples: int = RECENT_BACKTEST_MIN_SAMPLES,
) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=int)
    y_prob_array = np.asarray(y_prob, dtype=float)
    future_series = pd.Series(future_return).astype(float).reset_index(drop=True)
    if len(y_true_array) < min_samples or len(y_prob_array) != len(y_true_array) or len(future_series) != len(y_true_array):
        return {
            "recent_backtest_window": 0.0,
            "recent_backtest_ready": 0.0,
            "recent_sample_size": 0.0,
            "recent_positive_rate": 0.0,
            "recent_roc_auc": float("nan"),
            "recent_brier": float("nan"),
            "recent_top_bucket_return": 0.0,
            "recent_spread_return": 0.0,
            "recent_precision_gate_precision": 0.0,
            "recent_precision_gate_support": 0.0,
        }
    effective_window = min(int(window), len(y_true_array))
    if effective_window < min_samples:
        effective_window = len(y_true_array)
    tail_slice = slice(len(y_true_array) - effective_window, len(y_true_array))
    recent_metrics = _build_backtest_metrics(
        y_true_array[tail_slice],
        y_prob_array[tail_slice],
        future_series.iloc[tail_slice].reset_index(drop=True),
    )
    return {
        "recent_backtest_window": float(effective_window),
        "recent_backtest_ready": 1.0,
        "recent_sample_size": float(recent_metrics.get("sample_size", effective_window)),
        "recent_positive_rate": float(recent_metrics.get("positive_rate", 0.0)),
        "recent_roc_auc": float(recent_metrics.get("roc_auc", float("nan"))),
        "recent_brier": float(recent_metrics.get("brier", float("nan"))),
        "recent_top_bucket_return": float(recent_metrics.get("top_bucket_return", 0.0)),
        "recent_spread_return": float(recent_metrics.get("spread_return", 0.0)),
        "recent_precision_gate_precision": float(recent_metrics.get("precision_gate_precision", 0.0)),
        "recent_precision_gate_support": float(recent_metrics.get("precision_gate_support", 0.0)),
    }


def _apply_recent_backtest_guard(
    latest_probability: float,
    metrics: dict[str, float] | None,
    *,
    predicted_upside_pct: float = 0.0,
    predicted_upside_low_pct: float = 0.0,
    predicted_upside_high_pct: float = 0.0,
) -> tuple[float, float, float, float, dict[str, float], str]:
    metrics_map = dict(metrics or {})
    recent_window = int(metrics_map.get("recent_backtest_window", 0) or 0)
    if recent_window < RECENT_BACKTEST_MIN_SAMPLES or not bool(metrics_map.get("recent_backtest_ready", 0.0)):
        guard_info = {
            "recent_guard_strength": 0.0,
            "recent_guard_confidence": 0.0,
            "recent_guard_active": 0.0,
            "recent_guard_probability_delta_pct": 0.0,
            "recent_guard_upside_scale": 1.0,
        }
        return latest_probability, predicted_upside_pct, predicted_upside_low_pct, predicted_upside_high_pct, guard_info, ""

    sample_size = max(float(metrics_map.get("sample_size", 0.0) or 0.0), 1.0)
    positive_anchor = float(metrics_map.get("positive_rate", latest_probability) or latest_probability)
    recent_auc = float(metrics_map.get("recent_roc_auc", float("nan")))
    overall_auc = float(metrics_map.get("roc_auc", float("nan")))
    recent_brier = float(metrics_map.get("recent_brier", float("nan")))
    overall_brier = float(metrics_map.get("brier", float("nan")))
    recent_top_return = float(metrics_map.get("recent_top_bucket_return", 0.0) or 0.0)
    overall_top_return = float(metrics_map.get("top_bucket_return", 0.0) or 0.0)
    recent_gate_precision = float(metrics_map.get("recent_precision_gate_precision", 0.0) or 0.0)
    overall_gate_precision = float(metrics_map.get("precision_gate_precision", 0.0) or 0.0)
    recent_gate_support = float(metrics_map.get("recent_precision_gate_support", 0.0) or 0.0)
    average_move = max(
        abs(overall_top_return),
        abs(float(metrics_map.get("spread_return", 0.0) or 0.0)),
        abs(recent_top_return),
        0.012,
    )

    auc_gap = 0.0 if np.isnan(recent_auc) or np.isnan(overall_auc) else _clip((overall_auc - recent_auc) / 0.16, 0.0, 1.8)
    brier_gap = 0.0 if np.isnan(recent_brier) or np.isnan(overall_brier) else _clip((recent_brier - overall_brier) / 0.08, 0.0, 1.8)
    return_gap = _clip((overall_top_return - recent_top_return) / average_move, 0.0, 1.8)
    gate_gap = _clip((overall_gate_precision - recent_gate_precision) / 0.20, 0.0, 1.8)
    support_confidence = min(recent_gate_support / 10.0, 1.0) * 0.45 + min(recent_window / 24.0, 1.0) * 0.55
    history_confidence = min(sample_size / 220.0, 1.0) * 0.35 + min(recent_window / float(RECENT_BACKTEST_WINDOW), 1.0) * 0.65
    guard_confidence = _clip(support_confidence * 0.62 + history_confidence * 0.38, 0.0, 1.0)
    raw_gap = auc_gap * 0.32 + brier_gap * 0.24 + return_gap * 0.24 + gate_gap * 0.20
    guard_strength = _clip(raw_gap * guard_confidence * 0.22, 0.0, 0.28)

    guarded_probability = positive_anchor + (float(latest_probability) - positive_anchor) * (1.0 - guard_strength)
    guarded_probability = float(np.clip(guarded_probability, 1e-4, 1 - 1e-4))
    upside_scale = 1.0 - guard_strength * 1.18
    guarded_upside = max(float(predicted_upside_pct) * upside_scale, 0.0)
    guarded_low = max(min(float(predicted_upside_low_pct) * upside_scale, guarded_upside), 0.0)
    guarded_high = max(float(predicted_upside_high_pct) * max(0.74, 1.0 - guard_strength * 0.92), guarded_upside)
    probability_delta_pct = (guarded_probability - float(latest_probability)) * 100.0

    note = ""
    if guard_strength >= 0.01:
        note = (
            f"最近 {recent_window} 个时序回测样本较整体表现偏弱，"
            f"已对最新概率收缩 {probability_delta_pct:.2f}pct，并同步压缩预测涨幅，避免短期过拟合。"
        )
    guard_info = {
        "recent_guard_strength": round(float(guard_strength), 4),
        "recent_guard_confidence": round(float(guard_confidence), 4),
        "recent_guard_active": float(1.0 if guard_strength >= 0.01 else 0.0),
        "recent_guard_probability_delta_pct": round(float(probability_delta_pct), 2),
        "recent_guard_upside_scale": round(float(upside_scale), 4),
    }
    return guarded_probability, guarded_upside, guarded_low, guarded_high, guard_info, note


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if len(values) == 0:
        return 0.0
    safe_weights = np.asarray(weights, dtype=float)
    safe_values = np.asarray(values, dtype=float)
    valid_mask = np.isfinite(safe_values) & np.isfinite(safe_weights) & (safe_weights > 0)
    if not valid_mask.any():
        return float(np.nanquantile(safe_values, quantile))
    ordered_idx = np.argsort(safe_values[valid_mask])
    ordered_values = safe_values[valid_mask][ordered_idx]
    ordered_weights = safe_weights[valid_mask][ordered_idx]
    cumulative = np.cumsum(ordered_weights)
    cutoff = float(np.clip(quantile, 0.0, 1.0)) * float(cumulative[-1])
    position = int(np.searchsorted(cumulative, cutoff, side="left"))
    position = max(0, min(position, len(ordered_values) - 1))
    return float(ordered_values[position])


def _estimate_projected_upside(
    latest_probability: float,
    *,
    metrics: dict[str, float] | None = None,
    signal_breakdown: dict[str, float] | None = None,
    future_return: pd.Series | np.ndarray | None = None,
    reference_probabilities: pd.Series | np.ndarray | None = None,
) -> tuple[float, float, float]:
    safe_probability = float(np.clip(latest_probability, 1e-4, 1 - 1e-4))
    signal_map = dict(signal_breakdown or {})
    metrics_map = dict(metrics or {})

    projected_return = 0.0
    low_return = 0.0
    high_return = 0.0
    support_count = 0

    if future_return is not None and reference_probabilities is not None:
        future_array = np.asarray(pd.Series(future_return).astype(float), dtype=float)
        probability_array = np.asarray(pd.Series(reference_probabilities).astype(float), dtype=float)
        valid_mask = np.isfinite(future_array) & np.isfinite(probability_array)
        future_array = future_array[valid_mask]
        probability_array = probability_array[valid_mask]
        if len(future_array):
            distance = np.abs(probability_array - safe_probability)
            nearest_count = max(18, min(len(future_array), max(36, len(future_array) // 10)))
            nearest_idx = np.argsort(distance)[:nearest_count]
            local_future = future_array[nearest_idx]
            local_distance = distance[nearest_idx]
            local_weights = 1.0 / np.maximum(local_distance + 0.02, 0.02)
            support_count = int(len(local_future))
            mean_return = float(np.average(local_future, weights=local_weights))
            positive_mask = local_future > 0
            if positive_mask.any():
                positive_return = float(np.average(local_future[positive_mask], weights=local_weights[positive_mask]))
            else:
                positive_return = max(mean_return, 0.0)
            projected_return = max(mean_return * 0.52 + positive_return * 0.48, 0.0)
            low_return = max(_weighted_quantile(local_future, local_weights, 0.35), 0.0)
            high_return = max(_weighted_quantile(local_future, local_weights, 0.80), projected_return)

    if projected_return <= 0.0 and high_return <= 0.0:
        top_bucket_return = float(metrics_map.get("top_bucket_return", 0.0) or 0.0)
        precision_gate_return = float(metrics_map.get("precision_gate_return", 0.0) or 0.0)
        spread_return = float(metrics_map.get("spread_return", 0.0) or 0.0)
        positive_rate = float(metrics_map.get("positive_rate", safe_probability) or safe_probability)
        mean_anchor = max(top_bucket_return * (0.42 + safe_probability * 0.78), 0.0)
        precision_anchor = max(precision_gate_return, 0.0) * (0.18 + safe_probability * 0.62)
        spread_anchor = max(spread_return, 0.0) * 0.24
        projected_return = max(mean_anchor * 0.72 + precision_anchor * 0.28 + spread_anchor, 0.0)
        low_return = max(projected_return * (0.46 + positive_rate * 0.18), 0.0)
        high_return = max(projected_return * (1.18 + safe_probability * 0.16), projected_return)

    structure_tilt = (
        (float(signal_map.get("trend_score", 50.0)) - 50.0) * 0.00055
        + (float(signal_map.get("breakout_score", 50.0)) - 50.0) * 0.00045
        + (float(signal_map.get("pullback_score", 50.0)) - 50.0) * 0.00022
        - (float(signal_map.get("risk_score", 50.0)) - 50.0) * 0.00060
    )
    probability_scale = 0.52 + safe_probability * 0.96
    projected_return = max((projected_return + structure_tilt) * probability_scale, 0.0)
    low_return = max((low_return + structure_tilt * 0.4) * max(0.30, probability_scale * 0.82), 0.0)
    high_return = max((high_return + structure_tilt * 0.9) * max(0.55, probability_scale * 1.04), projected_return)

    if support_count and support_count < 24:
        sample_scale = 0.80 + support_count / 120.0
        projected_return *= sample_scale
        low_return *= sample_scale
        high_return *= sample_scale

    projected_pct = float(np.clip(projected_return * 100, 0.0, 60.0))
    low_pct = float(np.clip(min(low_return, projected_return) * 100, 0.0, projected_pct))
    high_pct = float(np.clip(max(high_return, projected_return) * 100, projected_pct, 80.0))
    return round(projected_pct, 2), round(low_pct, 2), round(high_pct, 2)


def _precision_gate_state(
    metrics: dict[str, float],
    latest_probability: float,
    *,
    target_precision: float = 0.90,
) -> tuple[float, float, int, bool, str]:
    threshold = float(metrics.get("precision_gate_threshold", 1.0) or 1.0)
    precision = float(metrics.get("precision_gate_precision", 0.0) or 0.0)
    support = int(metrics.get("precision_gate_support", 0.0) or 0)
    reached = bool(metrics.get("precision_target_reached", 0.0))
    active = reached and support > 0 and float(latest_probability) >= threshold
    if active:
        label = f"{int(target_precision * 100)}%精度放行"
    elif reached:
        label = f"历史达标，当前未入阈值({threshold:.2f})"
    elif support > 0 and precision >= target_precision - 0.08:
        label = f"高精度观察({precision * 100:.1f}%)"
    else:
        label = "未达90%精度门槛"
    return threshold, precision, support, active, label


def _baseline_probability_result(dataset: pd.DataFrame, latest_features: pd.DataFrame | None = None) -> ProbabilityResult:
    baseline = float(dataset["target"].mean()) if not dataset.empty else 0.5
    probabilities = pd.Series(baseline, index=dataset.index, name="probability")
    metrics = {
        "accuracy": baseline,
        "precision": baseline,
        "recall": baseline,
        "f1": baseline,
        "roc_auc": float("nan"),
        "brier": float("nan"),
        "sample_size": float(len(dataset)),
        "positive_rate": baseline,
        "top_bucket_return": float(dataset["future_return"].mean()) if "future_return" in dataset and not dataset.empty else 0.0,
        "top_bucket_win_rate": baseline,
        "low_bucket_return": float(dataset["future_return"].mean()) if "future_return" in dataset and not dataset.empty else 0.0,
        "spread_return": 0.0,
        "precision_target": 0.90,
        "precision_gate_threshold": 1.0,
        "precision_gate_precision": baseline if len(dataset) else 0.0,
        "precision_gate_support": 0.0,
        "precision_gate_return": 0.0,
        "precision_target_reached": 0.0,
    }
    latest_row = {}
    if latest_features is not None and not latest_features.empty:
        latest_row = latest_features.iloc[-1].to_dict()
    breakdown = _build_signal_breakdown(latest_row) if latest_row else {}
    strategy_score = round(
        _clip(
            baseline * 100 * 0.54
            + float(breakdown.get("trend_score", 50.0)) * 0.18
            + float(breakdown.get("breakout_score", 50.0)) * 0.12
            + float(breakdown.get("pullback_score", 50.0)) * 0.08
            + (100 - float(breakdown.get("risk_score", 50.0))) * 0.08
            + float(breakdown.get("launch_readiness_score", 50.0)) * 0.03
            + float(breakdown.get("market_resonance_score", 50.0)) * 0.03
        ),
        2,
    )
    return ProbabilityResult(
        latest_probability=baseline,
        probabilities=probabilities,
        out_of_sample_probabilities=probabilities.copy(),
        metrics=metrics,
        coefficients=[],
        strategy_score=strategy_score,
        agreement_score=50.0,
        signal_label="样本不足",
        risk_label="历史样本不足，先以 K 线阶段、分时承接和主力资金确认为主。",
        quality_label="偏谨慎",
        backtest_summary="本地样本不足或只有单边标签，暂时退回到基础概率。",
        signal_breakdown=breakdown,
        regime_label=str(_regime_labels_from_feature_frame(latest_features)[-1]) if latest_features is not None and not latest_features.empty else "rotation",
        precision_target=0.90,
        precision_gate_threshold=1.0,
        precision_gate_precision=float(metrics["precision_gate_precision"]),
        precision_gate_support=int(metrics["precision_gate_support"]),
        precision_gate_active=False,
        precision_gate_label="未达90%精度门槛",
    )


def _strategy_labels(latest_probability: float, strategy_score: float, risk_score: float) -> tuple[str, str]:
    probability_pct = latest_probability * 100 if latest_probability <= 1 else latest_probability
    if probability_pct >= 68 and strategy_score >= 70 and risk_score <= 45:
        return "趋势延续优先", "更适合做顺势确认，优先找均价线之上的第一次承接。"
    if probability_pct >= 58 and strategy_score >= 62:
        return "回踩确认可做", "适合等支撑位与分时共振，不适合直接在急拉段追高。"
    if risk_score >= 62 or probability_pct <= 42:
        return "防守等待", "高波动或高分歧阶段，先管仓位和卖点，再谈新开仓。"
    return "边界观察", "信号还没完全统一，等待平台边界或均价线方向更清晰。"


def _baseline_probability_result(dataset: pd.DataFrame, latest_features: pd.DataFrame | None = None) -> ProbabilityResult:
    baseline = float(dataset["target"].mean()) if not dataset.empty else 0.5
    probabilities = pd.Series(baseline, index=dataset.index, name="probability")
    metrics = {
        "accuracy": baseline,
        "precision": baseline,
        "recall": baseline,
        "f1": baseline,
        "roc_auc": float("nan"),
        "brier": float("nan"),
        "sample_size": float(len(dataset)),
        "positive_rate": baseline,
        "top_bucket_return": float(dataset["future_return"].mean()) if "future_return" in dataset.columns and not dataset.empty else 0.0,
        "top_bucket_win_rate": baseline,
        "low_bucket_return": float(dataset["future_return"].mean()) if "future_return" in dataset.columns and not dataset.empty else 0.0,
        "spread_return": 0.0,
        "precision_target": 0.90,
        "precision_gate_threshold": 1.0,
        "precision_gate_precision": baseline if len(dataset) else 0.0,
        "precision_gate_support": 0.0,
        "precision_gate_return": 0.0,
        "precision_target_reached": 0.0,
    }
    latest_row = {}
    if latest_features is not None and not latest_features.empty:
        latest_row = latest_features.iloc[-1].to_dict()
    breakdown = _build_signal_breakdown(latest_row) if latest_row else {}
    predicted_upside_pct, predicted_upside_low_pct, predicted_upside_high_pct = _estimate_projected_upside(
        baseline,
        metrics=metrics,
        signal_breakdown=breakdown,
        future_return=dataset["future_return"] if "future_return" in dataset.columns else None,
        reference_probabilities=probabilities if not probabilities.empty else None,
    )
    strategy_score = round(
        _clip(
            baseline * 100 * 0.54
            + float(breakdown.get("trend_score", 50.0)) * 0.18
            + float(breakdown.get("breakout_score", 50.0)) * 0.12
            + float(breakdown.get("pullback_score", 50.0)) * 0.08
            + (100 - float(breakdown.get("risk_score", 50.0))) * 0.08
            + float(breakdown.get("launch_readiness_score", 50.0)) * 0.03
            + float(breakdown.get("market_resonance_score", 50.0)) * 0.03
        ),
        2,
    )
    return ProbabilityResult(
        latest_probability=baseline,
        probabilities=probabilities,
        out_of_sample_probabilities=probabilities.copy(),
        metrics=metrics,
        coefficients=[],
        strategy_score=strategy_score,
        agreement_score=50.0,
        signal_label="样本不足",
        risk_label="历史样本不足，先以阶段结构、分时承接和资金确认为主。",
        quality_label="偏谨慎",
        backtest_summary="本地样本不足或只有单边标签，暂时退回到基础概率。",
        signal_breakdown=breakdown,
        regime_label=str(_regime_labels_from_feature_frame(latest_features)[-1]) if latest_features is not None and not latest_features.empty else "rotation",
        precision_target=0.90,
        precision_gate_threshold=1.0,
        precision_gate_precision=float(metrics["precision_gate_precision"]),
        precision_gate_support=int(metrics["precision_gate_support"]),
        precision_gate_active=False,
        precision_gate_label="未达90%精度门槛",
        predicted_upside_pct=predicted_upside_pct,
        predicted_upside_low_pct=predicted_upside_low_pct,
        predicted_upside_high_pct=predicted_upside_high_pct,
    )


def train_market_wide_model(
    horizon_days: int = 5,
    positive_return: float = 0.03,
    train_start: str = GLOBAL_MODEL_TRAIN_START,
    train_end: str = GLOBAL_MODEL_TRAIN_END,
    test_start: str = GLOBAL_MODEL_TEST_START,
    test_end: str = GLOBAL_MODEL_TEST_END,
    *,
    refresh: bool = False,
) -> MarketWideModel:
    cache_path = _global_model_cache_path(horizon_days, positive_return, train_start, train_end, test_start, test_end)
    if not refresh and cache_path.exists():
        try:
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            if isinstance(cached, MarketWideModel):
                cached.metrics.setdefault("survivorship_bias_risk", 1.0)
                return cached
        except Exception:
            pass

    dataset = build_market_wide_dataset(
        horizon_days=horizon_days,
        positive_return=positive_return,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        refresh=refresh,
    )
    if dataset.empty:
        raise RuntimeError("全市场样本构建失败，无法训练全市场模型。")

    train_mask = _date_slice_mask(dataset["signal_date"], train_start, train_end)
    test_mask = _date_slice_mask(dataset["signal_date"], test_start, test_end)
    train_df = dataset.loc[train_mask].copy()
    test_df = dataset.loc[test_mask].copy()
    if train_df.empty or test_df.empty or train_df["target"].nunique() < 2 or test_df["target"].nunique() < 2:
        raise RuntimeError("全市场样本不足以完成 2025 训练 / 2026Q1 测试。")

    train_splits = min(5, max(2, len(train_df) // 120))
    train_oof_components = _build_oof_component_probabilities(train_df, n_splits=train_splits)
    train_valid_mask = np.isfinite(train_oof_components).all(axis=1)
    ensemble_weights = _derive_dynamic_ensemble_weights(
        train_df["target"].to_numpy(dtype=int),
        train_oof_components,
        train_df["future_return"].astype(float).reset_index(drop=True),
    )
    train_raw_probabilities = _blend_component_matrix(train_oof_components[train_valid_mask], weights=ensemble_weights)
    calibrator = _fit_probability_calibrator(
        train_df.loc[train_valid_mask, "target"].to_numpy(dtype=int),
        train_raw_probabilities,
        train_oof_components[train_valid_mask],
        train_df.loc[train_valid_mask],
    )
    regime_calibrators = _fit_regime_calibrators(
        train_df.loc[train_valid_mask, "target"].to_numpy(dtype=int),
        train_raw_probabilities,
        train_oof_components[train_valid_mask],
        train_df.loc[train_valid_mask],
    )

    fitted_models = _fit_model_ensemble(train_df)
    test_probabilities, _ = _ensemble_probability(
        fitted_models,
        test_df[MODEL_FEATURE_COLUMNS],
        weights=ensemble_weights,
        calibrator=calibrator,
        calibration_feature_frame=test_df,
        regime_calibrators=regime_calibrators,
    )
    test_probabilities, _ = _apply_incremental_probability_upgrade(test_probabilities, test_df[MODEL_FEATURE_COLUMNS])
    metrics = _build_backtest_metrics(
        test_df["target"].to_numpy(dtype=int),
        test_probabilities,
        test_df["future_return"].astype(float),
    )
    metrics.update(
        _recent_tail_backtest_profile(
            test_df["target"].to_numpy(dtype=int),
            test_probabilities,
            test_df["future_return"].astype(float),
        )
    )
    metrics["enhancement_layer_applied"] = 1.0
    metrics["train_sample_size"] = float(len(train_df))
    metrics["test_sample_size"] = float(len(test_df))
    metrics["train_symbol_count"] = float(train_df["symbol"].nunique())
    metrics["test_symbol_count"] = float(test_df["symbol"].nunique())
    metrics["ensemble_weight_logistic"] = float(ensemble_weights[0])
    metrics["ensemble_weight_forest"] = float(ensemble_weights[1])
    metrics["ensemble_weight_boost"] = float(ensemble_weights[2])
    metrics["calibration_used"] = float(1.0 if calibrator is not None else 0.0)
    metrics["regime_calibrator_count"] = float(len(regime_calibrators))
    point_in_time_universe = bool(dataset.attrs.get("point_in_time_universe", False))
    source_universe_size = int(dataset.attrs.get("source_universe_size", 0) or 0)
    metrics["point_in_time_universe"] = float(1.0 if point_in_time_universe else 0.0)
    metrics["survivorship_bias_risk"] = 0.25 if point_in_time_universe else 1.0

    logistic_model = fitted_models[0].named_steps["model"]
    coefficients = sorted(
        [(name, float(weight)) for name, weight in zip(MODEL_FEATURE_COLUMNS, logistic_model.coef_[0])],
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:12]
    quality_label, quality_summary = _summarize_quality(metrics)
    regime_distribution = {
        label: int((train_df["market_regime_label"] == label).sum())
        for label in REGIME_LABELS
        if "market_regime_label" in train_df.columns
    }
    universe_bias_note = (
        f" 已使用 Tushare 全状态基础表还原历史股票池，source_universe_size={source_universe_size}，"
        "survivorship_bias_risk=lowered。"
        if point_in_time_universe
        else " 当前训练集仍由现时可见股票池回看构造，已标记 survivorship_bias_risk=True。"
    )
    summary = (
        f"全市场模型使用 {train_start} 到 {train_end} 的 A 股样本训练，"
        f"并用 {test_start} 到 {test_end} 做严格时间切分测试；"
        f"测试 AUC {metrics['roc_auc']:.3f}，高置信样本平均未来收益 {metrics['top_bucket_return'] * 100:.2f}%。"
        f" {quality_summary}{universe_bias_note}"
    )
    if metrics.get("precision_target_reached", 0.0):
        summary = (
            f"{summary} 历史上存在 `>= {metrics['precision_gate_threshold']:.2f}` 的高置信区间，"
            f"其上涨命中率约 `{metrics['precision_gate_precision'] * 100:.1f}%`，"
            f"样本数 `{int(metrics['precision_gate_support'])}`。"
        )
    result = MarketWideModel(
        horizon_days=horizon_days,
        positive_return=positive_return,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        fitted_models=fitted_models,
        ensemble_weights=tuple(float(weight) for weight in ensemble_weights),
        calibrator=calibrator,
        metrics=metrics,
        coefficients=coefficients,
        train_sample_size=int(len(train_df)),
        test_sample_size=int(len(test_df)),
        universe_size=source_universe_size or int(fetch_a_share_universe()["symbol"].nunique()),
        eligible_symbols=int(dataset["symbol"].nunique()),
        quality_label=quality_label,
        summary=summary,
        regime_calibrators=regime_calibrators,
        regime_distribution=regime_distribution,
    )
    with cache_path.open("wb") as handle:
        pickle.dump(result, handle)
    return result


@lru_cache(maxsize=4)
def load_market_wide_model(
    horizon_days: int = 5,
    positive_return: float = 0.03,
    train_start: str = GLOBAL_MODEL_TRAIN_START,
    train_end: str = GLOBAL_MODEL_TRAIN_END,
    test_start: str = GLOBAL_MODEL_TEST_START,
    test_end: str = GLOBAL_MODEL_TEST_END,
) -> MarketWideModel:
    return train_market_wide_model(
        horizon_days=horizon_days,
        positive_return=positive_return,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )


def clear_market_wide_model_cache() -> None:
    load_market_wide_model.cache_clear()
    load_market_proxy_model.cache_clear()


def _latest_prediction_feature_frame(
    daily: pd.DataFrame,
    latest_feature_values: pd.Series | dict[str, float] | None = None,
) -> pd.DataFrame:
    return _prepare_live_feature_frame(
        daily,
        latest_feature_values=latest_feature_values,
        symbol=_resolve_symbol_from_daily(daily),
    )


def score_with_market_wide_model(
    daily: pd.DataFrame,
    market_model: MarketWideModel,
    latest_feature_values: pd.Series | dict[str, float] | None = None,
) -> ProbabilityResult:
    latest = _latest_prediction_feature_frame(daily, latest_feature_values=latest_feature_values)
    if latest.empty:
        return _baseline_probability_result(pd.DataFrame(), latest)

    latest_probability, latest_breakdown = _ensemble_probability(
        market_model.fitted_models,
        latest[MODEL_FEATURE_COLUMNS],
        weights=market_model.ensemble_weights,
        calibrator=market_model.calibrator,
        calibration_feature_frame=latest,
        regime_calibrators=market_model.regime_calibrators,
    )
    enhanced_probability, enhancement_frame = _apply_incremental_probability_upgrade(latest_probability, latest)
    latest_row = latest.iloc[-1]
    signal_breakdown = _build_signal_breakdown(latest_row)
    if not enhancement_frame.empty:
        enhancement_row = enhancement_frame.iloc[-1]
        signal_breakdown.update(
            {
                "base_probability_pct": round(float(enhancement_row.get("base_probability_pct", 0.0)), 2),
                "upgraded_probability_pct": round(float(enhancement_row.get("upgraded_probability_pct", 0.0)), 2),
                "upgrade_delta_pct": round(float(enhancement_row.get("upgrade_delta_pct", 0.0)), 2),
                "daily_k_score": round(float(enhancement_row.get("daily_k_score", 50.0)), 2),
                "volume_price_score": round(float(enhancement_row.get("volume_price_score", 50.0)), 2),
                "market_context_score": round(float(enhancement_row.get("market_context_score", 50.0)), 2),
                "news_fund_score": round(float(enhancement_row.get("news_fund_score", 50.0)), 2),
                "quant_context_score": round(float(enhancement_row.get("quant_context_score", 50.0)), 2),
                "launch_readiness_score": round(float(enhancement_row.get("launch_readiness_score", 50.0)), 2),
                "market_resonance_score": round(float(enhancement_row.get("market_resonance_score", 50.0)), 2),
                "launch_specialist_score": round(float(enhancement_row.get("launch_specialist_score", 50.0)), 2),
                "launch_regime_fit_score": round(float(enhancement_row.get("launch_regime_fit_score", 50.0)), 2),
                "launch_specialist_confidence": round(float(enhancement_row.get("launch_specialist_confidence", 0.5)) * 100, 2),
                "context_composite_score": round(float(enhancement_row.get("context_composite_score", 50.0)), 2),
            }
        )
    agreement_score = round(
        _clip(
            100
            - float(
                np.std(
                    [
                        float(latest_breakdown["logistic"][-1]),
                        float(latest_breakdown["forest"][-1]),
                        float(latest_breakdown["boost"][-1]),
                    ]
                )
            )
            * 260
        ),
        2,
    )
    strategy_score = round(
        _clip(
            float(enhanced_probability[-1]) * 100 * 0.50
            + float(signal_breakdown["trend_score"]) * 0.16
            + float(signal_breakdown["breakout_score"]) * 0.12
            + float(signal_breakdown["pullback_score"]) * 0.10
            + (100 - float(signal_breakdown["risk_score"])) * 0.10
            + float(signal_breakdown.get("launch_readiness_score", 50.0)) * 0.04
            + float(signal_breakdown.get("market_resonance_score", 50.0)) * 0.04
            + float(signal_breakdown.get("launch_specialist_score", 50.0)) * 0.03
            + float(signal_breakdown.get("launch_regime_fit_score", 50.0)) * 0.02
            + float(signal_breakdown.get("context_composite_score", 50.0)) * 0.02
        ),
        2,
    )
    signal_label, risk_label = _strategy_labels(
        latest_probability=float(enhanced_probability[-1]),
        strategy_score=strategy_score,
        risk_score=float(signal_breakdown["risk_score"]),
    )
    projected_upside_pct, projected_upside_low_pct, projected_upside_high_pct = _estimate_projected_upside(
        float(enhanced_probability[-1]),
        metrics=market_model.metrics,
        signal_breakdown=signal_breakdown,
    )
    (
        guarded_probability,
        projected_upside_pct,
        projected_upside_low_pct,
        projected_upside_high_pct,
        guard_info,
        guard_note,
    ) = _apply_recent_backtest_guard(
        float(enhanced_probability[-1]),
        market_model.metrics,
        predicted_upside_pct=projected_upside_pct,
        predicted_upside_low_pct=projected_upside_low_pct,
        predicted_upside_high_pct=projected_upside_high_pct,
    )
    signal_breakdown.update(guard_info)
    if guard_note:
        signal_breakdown["recent_guard_note"] = guard_note
    strategy_score = round(
        _clip(
            guarded_probability * 100 * 0.50
            + float(signal_breakdown["trend_score"]) * 0.16
            + float(signal_breakdown["breakout_score"]) * 0.12
            + float(signal_breakdown["pullback_score"]) * 0.10
            + (100 - float(signal_breakdown["risk_score"])) * 0.10
            + float(signal_breakdown.get("launch_readiness_score", 50.0)) * 0.04
            + float(signal_breakdown.get("market_resonance_score", 50.0)) * 0.04
            + float(signal_breakdown.get("launch_specialist_score", 50.0)) * 0.03
            + float(signal_breakdown.get("launch_regime_fit_score", 50.0)) * 0.02
            + float(signal_breakdown.get("context_composite_score", 50.0)) * 0.02
        ),
        2,
    )
    signal_label, risk_label = _strategy_labels(
        latest_probability=guarded_probability,
        strategy_score=strategy_score,
        risk_score=float(signal_breakdown["risk_score"]),
    )
    precision_gate_threshold, precision_gate_precision, precision_gate_support, precision_gate_active, precision_gate_label = (
        _precision_gate_state(
            market_model.metrics,
            guarded_probability,
        )
    )
    return ProbabilityResult(
        latest_probability=guarded_probability,
        probabilities=pd.Series(float(guarded_probability), index=latest.index, name="probability"),
        metrics={
            **market_model.metrics,
            **guard_info,
            "train_sample_size": float(market_model.train_sample_size),
            "test_sample_size": float(market_model.test_sample_size),
        },
        coefficients=market_model.coefficients,
        out_of_sample_probabilities=pd.Series(dtype=float),
        strategy_score=strategy_score,
        agreement_score=agreement_score,
        signal_label=signal_label,
        risk_label=risk_label,
        model_name="全市场 2025->2026Q1 模型",
        quality_label=market_model.quality_label,
        backtest_summary=f"{market_model.summary} {guard_note}".strip(),
        signal_breakdown=signal_breakdown,
        regime_label=str(_regime_labels_from_feature_frame(latest)[-1]) if len(latest) else "rotation",
        precision_target=float(market_model.metrics.get("precision_target", 0.90)),
        precision_gate_threshold=precision_gate_threshold,
        precision_gate_precision=precision_gate_precision,
        precision_gate_support=precision_gate_support,
        precision_gate_active=precision_gate_active,
        precision_gate_label=precision_gate_label,
        raw_probability=float(latest_probability[-1]),
        enhanced_probability=float(enhanced_probability[-1]),
        predicted_upside_pct=projected_upside_pct,
        predicted_upside_low_pct=projected_upside_low_pct,
        predicted_upside_high_pct=projected_upside_high_pct,
        base_probability=float(latest_probability[-1]),
        upgrade_delta=float(guarded_probability - latest_probability[-1]),
        upgrade_components={
            "daily_k_score": float(signal_breakdown.get("daily_k_score", 50.0)),
            "volume_price_score": float(signal_breakdown.get("volume_price_score", 50.0)),
            "market_context_score": float(signal_breakdown.get("market_context_score", 50.0)),
            "news_fund_score": float(signal_breakdown.get("news_fund_score", 50.0)),
            "quant_context_score": float(signal_breakdown.get("quant_context_score", 50.0)),
            "launch_readiness_score": float(signal_breakdown.get("launch_readiness_score", 50.0)),
            "market_resonance_score": float(signal_breakdown.get("market_resonance_score", 50.0)),
            "launch_specialist_score": float(signal_breakdown.get("launch_specialist_score", 50.0)),
            "launch_regime_fit_score": float(signal_breakdown.get("launch_regime_fit_score", 50.0)),
            "launch_specialist_confidence": float(signal_breakdown.get("launch_specialist_confidence", 50.0)),
            "context_composite_score": float(signal_breakdown.get("context_composite_score", 50.0)),
            "recent_guard_strength": float(guard_info.get("recent_guard_strength", 0.0)),
        },
        upgrade_summary=(
            f"基础概率 {float(latest_probability[-1]) * 100:.1f}% ，"
            f"历史增强后 {float(enhanced_probability[-1]) * 100:.1f}% 。"
        ),
    )


def score_with_market_proxy_model(
    daily: pd.DataFrame,
    proxy_model: MarketProxyModel,
    latest_feature_values: pd.Series | dict[str, float] | None = None,
) -> ProbabilityResult:
    latest = _latest_prediction_feature_frame(daily, latest_feature_values=latest_feature_values)
    if latest.empty:
        return _baseline_probability_result(pd.DataFrame(), latest)

    latest_probability = proxy_model.fitted_model.predict_proba(latest[MODEL_FEATURE_COLUMNS])[:, 1]
    enhanced_probability, enhancement_frame = _apply_incremental_probability_upgrade(latest_probability, latest)
    latest_row = latest.iloc[-1]
    signal_breakdown = _build_signal_breakdown(latest_row)
    if not enhancement_frame.empty:
        enhancement_row = enhancement_frame.iloc[-1]
        signal_breakdown.update(
            {
                "base_probability_pct": round(float(enhancement_row.get("base_probability_pct", 0.0)), 2),
                "upgraded_probability_pct": round(float(enhancement_row.get("upgraded_probability_pct", 0.0)), 2),
                "upgrade_delta_pct": round(float(enhancement_row.get("upgrade_delta_pct", 0.0)), 2),
                "daily_k_score": round(float(enhancement_row.get("daily_k_score", 50.0)), 2),
                "volume_price_score": round(float(enhancement_row.get("volume_price_score", 50.0)), 2),
                "market_context_score": round(float(enhancement_row.get("market_context_score", 50.0)), 2),
                "news_fund_score": round(float(enhancement_row.get("news_fund_score", 50.0)), 2),
                "quant_context_score": round(float(enhancement_row.get("quant_context_score", 50.0)), 2),
                "launch_readiness_score": round(float(enhancement_row.get("launch_readiness_score", 50.0)), 2),
                "market_resonance_score": round(float(enhancement_row.get("market_resonance_score", 50.0)), 2),
                "launch_specialist_score": round(float(enhancement_row.get("launch_specialist_score", 50.0)), 2),
                "launch_regime_fit_score": round(float(enhancement_row.get("launch_regime_fit_score", 50.0)), 2),
                "launch_specialist_confidence": round(float(enhancement_row.get("launch_specialist_confidence", 0.5)) * 100, 2),
                "context_composite_score": round(float(enhancement_row.get("context_composite_score", 50.0)), 2),
            }
        )
    strategy_score = round(
        _clip(
            float(enhanced_probability[-1]) * 100 * 0.50
            + float(signal_breakdown["trend_score"]) * 0.18
            + float(signal_breakdown["breakout_score"]) * 0.12
            + float(signal_breakdown["pullback_score"]) * 0.08
            + (100 - float(signal_breakdown["risk_score"])) * 0.08
            + float(signal_breakdown.get("launch_readiness_score", 50.0)) * 0.04
            + float(signal_breakdown.get("market_resonance_score", 50.0)) * 0.04
            + float(signal_breakdown.get("launch_specialist_score", 50.0)) * 0.03
            + float(signal_breakdown.get("launch_regime_fit_score", 50.0)) * 0.02
            + float(signal_breakdown.get("context_composite_score", 50.0)) * 0.04
        ),
        2,
    )
    signal_label, risk_label = _strategy_labels(
        latest_probability=float(enhanced_probability[-1]),
        strategy_score=strategy_score,
        risk_score=float(signal_breakdown["risk_score"]),
    )
    validation_metrics = proxy_model.validation_metrics or {}
    proxy_metrics = {
        "sample_size": float(proxy_model.sample_size),
        "positive_rate": float(proxy_model.positive_rate),
        "proxy_model": 1.0,
        "precision_target": float(validation_metrics.get("precision_target", 0.90) or 0.90),
        "precision_gate_threshold": float(validation_metrics.get("precision_gate_threshold", 1.0) or 1.0),
        "precision_gate_precision": float(validation_metrics.get("precision_gate_precision", 0.0) or 0.0),
        "precision_gate_support": float(validation_metrics.get("precision_gate_support", 0.0) or 0.0),
        "precision_gate_return": float(validation_metrics.get("precision_gate_return", 0.0) or 0.0),
        "precision_target_reached": float(validation_metrics.get("precision_target_reached", 0.0) or 0.0),
        "roc_auc": float(validation_metrics.get("roc_auc", float("nan"))),
        "brier": float(validation_metrics.get("brier", float("nan"))),
        "top_bucket_return": float(validation_metrics.get("top_bucket_return", 0.0) or 0.0),
        "top_bucket_win_rate": float(validation_metrics.get("top_bucket_win_rate", 0.0) or 0.0),
        "low_bucket_return": float(validation_metrics.get("low_bucket_return", 0.0) or 0.0),
        "spread_return": float(validation_metrics.get("spread_return", 0.0) or 0.0),
        "validation_sample_size": float(validation_metrics.get("sample_size", 0.0) or 0.0),
        "validation_positive_rate": float(validation_metrics.get("positive_rate", 0.0) or 0.0),
        "proxy_validation_objective": float(validation_metrics.get("proxy_validation_objective", 0.0) or 0.0),
    }
    projected_upside_pct, projected_upside_low_pct, projected_upside_high_pct = _estimate_projected_upside(
        float(enhanced_probability[-1]),
        metrics=proxy_metrics,
        signal_breakdown=signal_breakdown,
    )
    (
        guarded_probability,
        projected_upside_pct,
        projected_upside_low_pct,
        projected_upside_high_pct,
        guard_info,
        guard_note,
    ) = _apply_recent_backtest_guard(
        float(enhanced_probability[-1]),
        proxy_metrics,
        predicted_upside_pct=projected_upside_pct,
        predicted_upside_low_pct=projected_upside_low_pct,
        predicted_upside_high_pct=projected_upside_high_pct,
    )
    proxy_metrics.update(guard_info)
    signal_breakdown.update(guard_info)
    if guard_note:
        signal_breakdown["recent_guard_note"] = guard_note
    strategy_score = round(
        _clip(
            guarded_probability * 100 * 0.50
            + float(signal_breakdown["trend_score"]) * 0.18
            + float(signal_breakdown["breakout_score"]) * 0.12
            + float(signal_breakdown["pullback_score"]) * 0.08
            + (100 - float(signal_breakdown["risk_score"])) * 0.08
            + float(signal_breakdown.get("launch_readiness_score", 50.0)) * 0.04
            + float(signal_breakdown.get("market_resonance_score", 50.0)) * 0.04
            + float(signal_breakdown.get("launch_specialist_score", 50.0)) * 0.03
            + float(signal_breakdown.get("launch_regime_fit_score", 50.0)) * 0.02
            + float(signal_breakdown.get("context_composite_score", 50.0)) * 0.04
        ),
        2,
    )
    signal_label, risk_label = _strategy_labels(
        latest_probability=guarded_probability,
        strategy_score=strategy_score,
        risk_score=float(signal_breakdown["risk_score"]),
    )
    (
        precision_gate_threshold,
        precision_gate_precision,
        precision_gate_support,
        precision_gate_active,
        precision_gate_label,
    ) = _precision_gate_state(proxy_metrics, latest_probability=guarded_probability)
    candidate_name = str(proxy_model.candidate_name or "proxy")
    return ProbabilityResult(
        latest_probability=guarded_probability,
        probabilities=pd.Series(float(guarded_probability), index=latest.index, name="probability"),
        metrics=proxy_metrics,
        coefficients=[],
        out_of_sample_probabilities=pd.Series(dtype=float),
        strategy_score=strategy_score,
        agreement_score=61.0,
        signal_label=signal_label,
        risk_label=risk_label,
        model_name=f"market_proxy_{candidate_name}",
        quality_label="proxy_validated" if validation_metrics else "proxy_ready",
        backtest_summary=(proxy_model.validation_summary or "Using a validated proxy model trained from the cached partial market dataset for fast ranking refreshes.")
        + (f" {guard_note}" if guard_note else ""),
        signal_breakdown=signal_breakdown,
        regime_label=str(_regime_labels_from_feature_frame(latest)[-1]) if len(latest) else "rotation",
        precision_target=0.90,
        precision_gate_threshold=precision_gate_threshold,
        precision_gate_precision=precision_gate_precision,
        precision_gate_support=precision_gate_support,
        precision_gate_active=precision_gate_active,
        precision_gate_label=precision_gate_label,
        raw_probability=float(latest_probability[-1]),
        enhanced_probability=float(enhanced_probability[-1]),
        predicted_upside_pct=projected_upside_pct,
        predicted_upside_low_pct=projected_upside_low_pct,
        predicted_upside_high_pct=projected_upside_high_pct,
        base_probability=float(latest_probability[-1]),
        upgrade_delta=float(guarded_probability - latest_probability[-1]),
        upgrade_components={
            "daily_k_score": float(signal_breakdown.get("daily_k_score", 50.0)),
            "volume_price_score": float(signal_breakdown.get("volume_price_score", 50.0)),
            "market_context_score": float(signal_breakdown.get("market_context_score", 50.0)),
            "news_fund_score": float(signal_breakdown.get("news_fund_score", 50.0)),
            "quant_context_score": float(signal_breakdown.get("quant_context_score", 50.0)),
            "launch_readiness_score": float(signal_breakdown.get("launch_readiness_score", 50.0)),
            "market_resonance_score": float(signal_breakdown.get("market_resonance_score", 50.0)),
            "launch_specialist_score": float(signal_breakdown.get("launch_specialist_score", 50.0)),
            "launch_regime_fit_score": float(signal_breakdown.get("launch_regime_fit_score", 50.0)),
            "launch_specialist_confidence": float(signal_breakdown.get("launch_specialist_confidence", 50.0)),
            "context_composite_score": float(signal_breakdown.get("context_composite_score", 50.0)),
        },
        upgrade_summary=(
            f"基础概率 {float(latest_probability[-1]) * 100:.1f}% ，"
            f"历史增强后 {float(enhanced_probability[-1]) * 100:.1f}% 。"
        ),
    )


def _estimate_fast_latest_probability(
    daily: pd.DataFrame,
    latest_feature_values: pd.Series | dict[str, float] | None = None,
) -> float:
    latest = _latest_prediction_feature_frame(daily, latest_feature_values=latest_feature_values)
    if latest.empty:
        return 0.5
    latest_row = latest.iloc[-1]
    breakdown = _build_signal_breakdown(latest_row)
    raw_score = (
        (float(breakdown["trend_score"]) - 50.0) * 0.050
        + (float(breakdown["breakout_score"]) - 50.0) * 0.040
        + (float(breakdown["pullback_score"]) - 50.0) * 0.028
        - (float(breakdown["risk_score"]) - 35.0) * 0.048
        + (float(breakdown.get("launch_readiness_score", 50.0)) - 50.0) * 0.032
        + (float(breakdown.get("market_resonance_score", 50.0)) - 50.0) * 0.028
        + float(latest_row.get("relative_strength_5", 0.0)) * 5.5
        + float(latest_row.get("relative_strength_20", 0.0)) * 4.0
        + float(latest_row.get("close_vs_ma20", 0.0)) * 3.8
        + float(latest_row.get("breakout_distance_20", 0.0)) * 2.4
    )
    specialist_frame = _build_launch_specialist_frame(latest)
    if not specialist_frame.empty:
        raw_score += float(specialist_frame["launch_specialist_delta_pct"].iloc[-1]) * 0.42
    probability = 1.0 / (1.0 + np.exp(-raw_score / 6.0))
    base_probability = float(np.clip(probability, 0.06, 0.94))
    enhanced_probability, _ = _apply_incremental_probability_upgrade(
        np.array([base_probability], dtype=float),
        latest.reindex(columns=MODEL_FEATURE_COLUMNS, fill_value=0.0),
    )
    return float(enhanced_probability[-1])


def predict_latest_probability(
    daily: pd.DataFrame,
    horizon_days: int = 5,
    positive_return: float = 0.03,
    market_model: MarketWideModel | None = None,
    market_proxy_model: MarketProxyModel | None = None,
    latest_feature_values: pd.Series | dict[str, float] | None = None,
    allow_slow_fallback: bool = True,
) -> float:
    if market_model is not None:
        return score_with_market_wide_model(
            daily,
            market_model,
            latest_feature_values=latest_feature_values,
        ).latest_probability
    if market_proxy_model is not None:
        return score_with_market_proxy_model(
            daily,
            market_proxy_model,
            latest_feature_values=latest_feature_values,
        ).latest_probability
    if not allow_slow_fallback:
        return _estimate_fast_latest_probability(daily, latest_feature_values=latest_feature_values)
    dataset = _prepare_training_dataset(
        daily,
        horizon_days=horizon_days,
        positive_return=positive_return,
        symbol=_resolve_symbol_from_daily(daily),
    )
    latest_features = _latest_feature_frame(daily, dataset)
    if latest_features.empty:
        return _baseline_probability_result(dataset, latest_features).latest_probability
    if len(dataset) < 120 or dataset["target"].nunique() < 2:
        return _baseline_probability_result(dataset, latest_features).latest_probability

    n_splits = min(4, max(2, len(dataset) // 70))
    oof_components = _build_oof_component_probabilities(dataset, n_splits=n_splits)
    ensemble_weights = _derive_dynamic_ensemble_weights(
        dataset["target"].to_numpy(dtype=int),
        oof_components,
        dataset["future_return"].astype(float).reset_index(drop=True),
    )
    valid_mask = np.isfinite(oof_components).all(axis=1)
    raw_probabilities = _blend_component_matrix(oof_components[valid_mask], weights=ensemble_weights)
    calibrator = _fit_probability_calibrator(
        dataset.loc[valid_mask, "target"].to_numpy(dtype=int),
        raw_probabilities,
        oof_components[valid_mask],
        dataset.loc[valid_mask],
    )
    regime_calibrators = _fit_regime_calibrators(
        dataset.loc[valid_mask, "target"].to_numpy(dtype=int),
        raw_probabilities,
        oof_components[valid_mask],
        dataset.loc[valid_mask],
    )
    models = _fit_model_ensemble(dataset)
    latest_probability, _ = _ensemble_probability(
        models,
        latest_features[MODEL_FEATURE_COLUMNS],
        weights=ensemble_weights,
        calibrator=calibrator,
        calibration_feature_frame=latest_features,
        regime_calibrators=regime_calibrators,
    )
    enhanced_probability, _ = _apply_incremental_probability_upgrade(
        latest_probability,
        latest_features[MODEL_FEATURE_COLUMNS],
    )
    return float(enhanced_probability[-1])


def train_probability_model(
    daily: pd.DataFrame,
    horizon_days: int = 5,
    positive_return: float = 0.03,
    market_model: MarketWideModel | None = None,
) -> ProbabilityResult:
    if market_model is not None:
        return score_with_market_wide_model(daily, market_model)
    dataset = _prepare_training_dataset(
        daily,
        horizon_days=horizon_days,
        positive_return=positive_return,
        symbol=_resolve_symbol_from_daily(daily),
    )
    latest_features = _latest_feature_frame(daily, dataset)
    if len(dataset) < 120 or dataset["target"].nunique() < 2 or latest_features.empty:
        return _baseline_probability_result(dataset, latest_features)

    n_splits = min(4, max(2, len(dataset) // 70))
    oof_components = _build_oof_component_probabilities(dataset, n_splits=n_splits)
    valid_mask = np.isfinite(oof_components).all(axis=1)
    ensemble_weights = _derive_dynamic_ensemble_weights(
        dataset["target"].to_numpy(dtype=int),
        oof_components,
        dataset["future_return"].astype(float).reset_index(drop=True),
    )
    oof_probabilities = np.full(len(dataset), np.nan, dtype=float)
    raw_probabilities = _blend_component_matrix(oof_components[valid_mask], weights=ensemble_weights)
    calibrator = _fit_probability_calibrator(
        dataset.loc[valid_mask, "target"].to_numpy(dtype=int),
        raw_probabilities,
        oof_components[valid_mask],
        dataset.loc[valid_mask],
    )
    regime_calibrators = _fit_regime_calibrators(
        dataset.loc[valid_mask, "target"].to_numpy(dtype=int),
        raw_probabilities,
        oof_components[valid_mask],
        dataset.loc[valid_mask],
    )
    oof_probabilities[valid_mask] = _apply_probability_calibrator(
        raw_probabilities,
        oof_components[valid_mask],
        calibrator,
        dataset.loc[valid_mask],
        regime_calibrators=regime_calibrators,
    )
    enhanced_oof, _ = _apply_incremental_probability_upgrade(
        oof_probabilities[valid_mask],
        dataset.loc[valid_mask, MODEL_FEATURE_COLUMNS],
    )
    oof_probabilities[valid_mask] = enhanced_oof

    metric_mask = ~np.isnan(oof_probabilities)
    y_true = dataset.loc[metric_mask, "target"].to_numpy(dtype=int)
    future_return = dataset.loc[metric_mask, "future_return"].astype(float)
    y_prob = oof_probabilities[metric_mask]
    metrics = _build_backtest_metrics(y_true, y_prob, future_return)
    metrics.update(_recent_tail_backtest_profile(y_true, y_prob, future_return))
    metrics["ensemble_weight_logistic"] = float(ensemble_weights[0])
    metrics["ensemble_weight_forest"] = float(ensemble_weights[1])
    metrics["ensemble_weight_boost"] = float(ensemble_weights[2])
    metrics["calibration_used"] = float(1.0 if calibrator is not None else 0.0)
    metrics["regime_calibrator_count"] = float(len(regime_calibrators))

    final_models = _fit_model_ensemble(dataset)
    fitted_probabilities, breakdown_arrays = _ensemble_probability(
        final_models,
        dataset[MODEL_FEATURE_COLUMNS],
        weights=ensemble_weights,
        calibrator=calibrator,
        calibration_feature_frame=dataset,
        regime_calibrators=regime_calibrators,
    )
    fitted_probabilities, _ = _apply_incremental_probability_upgrade(
        fitted_probabilities,
        dataset[MODEL_FEATURE_COLUMNS],
    )
    latest_probability, latest_breakdown = _ensemble_probability(
        final_models,
        latest_features[MODEL_FEATURE_COLUMNS],
        weights=ensemble_weights,
        calibrator=calibrator,
        calibration_feature_frame=latest_features,
        regime_calibrators=regime_calibrators,
    )
    enhanced_latest_probability, latest_enhancement_frame = _apply_incremental_probability_upgrade(
        latest_probability,
        latest_features[MODEL_FEATURE_COLUMNS],
    )

    logistic_model = final_models[0].named_steps["model"]
    coefficients = sorted(
        [(name, float(weight)) for name, weight in zip(MODEL_FEATURE_COLUMNS, logistic_model.coef_[0])],
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:10]

    latest_row = latest_features.iloc[-1]
    breakdown = _build_signal_breakdown(latest_row)
    if not latest_enhancement_frame.empty:
        enhancement_row = latest_enhancement_frame.iloc[-1]
        breakdown.update(
            {
                "base_probability_pct": round(float(enhancement_row.get("base_probability_pct", 0.0)), 2),
                "upgraded_probability_pct": round(float(enhancement_row.get("upgraded_probability_pct", 0.0)), 2),
                "upgrade_delta_pct": round(float(enhancement_row.get("upgrade_delta_pct", 0.0)), 2),
                "daily_k_score": round(float(enhancement_row.get("daily_k_score", 50.0)), 2),
                "volume_price_score": round(float(enhancement_row.get("volume_price_score", 50.0)), 2),
                "market_context_score": round(float(enhancement_row.get("market_context_score", 50.0)), 2),
                "news_fund_score": round(float(enhancement_row.get("news_fund_score", 50.0)), 2),
                "quant_context_score": round(float(enhancement_row.get("quant_context_score", 50.0)), 2),
                "launch_readiness_score": round(float(enhancement_row.get("launch_readiness_score", 50.0)), 2),
                "market_resonance_score": round(float(enhancement_row.get("market_resonance_score", 50.0)), 2),
                "launch_specialist_score": round(float(enhancement_row.get("launch_specialist_score", 50.0)), 2),
                "launch_regime_fit_score": round(float(enhancement_row.get("launch_regime_fit_score", 50.0)), 2),
                "launch_specialist_confidence": round(float(enhancement_row.get("launch_specialist_confidence", 0.5)) * 100, 2),
                "context_composite_score": round(float(enhancement_row.get("context_composite_score", 50.0)), 2),
            }
        )
    latest_component_probs = np.array(
        [
            float(latest_breakdown["logistic"][-1]),
            float(latest_breakdown["forest"][-1]),
            float(latest_breakdown["boost"][-1]),
        ],
        dtype=float,
    )
    agreement_score = round(_clip(100 - float(np.std(latest_component_probs)) * 260), 2)
    strategy_score = round(
        _clip(
            float(enhanced_latest_probability[-1]) * 100 * 0.50
            + float(breakdown["trend_score"]) * 0.16
            + float(breakdown["breakout_score"]) * 0.12
            + float(breakdown["pullback_score"]) * 0.10
            + (100 - float(breakdown["risk_score"])) * 0.10
            + float(breakdown.get("launch_readiness_score", 50.0)) * 0.04
            + float(breakdown.get("market_resonance_score", 50.0)) * 0.04
            + float(breakdown.get("launch_specialist_score", 50.0)) * 0.03
            + float(breakdown.get("launch_regime_fit_score", 50.0)) * 0.02
            + float(breakdown.get("context_composite_score", 50.0)) * 0.02
        ),
        2,
    )
    signal_label, risk_label = _strategy_labels(
        latest_probability=float(enhanced_latest_probability[-1]),
        strategy_score=strategy_score,
        risk_score=float(breakdown["risk_score"]),
    )
    projected_upside_pct, projected_upside_low_pct, projected_upside_high_pct = _estimate_projected_upside(
        float(enhanced_latest_probability[-1]),
        metrics=metrics,
        signal_breakdown=breakdown,
        future_return=future_return,
        reference_probabilities=y_prob,
    )
    (
        guarded_probability,
        projected_upside_pct,
        projected_upside_low_pct,
        projected_upside_high_pct,
        guard_info,
        guard_note,
    ) = _apply_recent_backtest_guard(
        float(enhanced_latest_probability[-1]),
        metrics,
        predicted_upside_pct=projected_upside_pct,
        predicted_upside_low_pct=projected_upside_low_pct,
        predicted_upside_high_pct=projected_upside_high_pct,
    )
    metrics.update(guard_info)
    strategy_score = round(
        _clip(
            guarded_probability * 100 * 0.50
            + float(breakdown["trend_score"]) * 0.16
            + float(breakdown["breakout_score"]) * 0.12
            + float(breakdown["pullback_score"]) * 0.10
            + (100 - float(breakdown["risk_score"])) * 0.10
            + float(breakdown.get("launch_readiness_score", 50.0)) * 0.04
            + float(breakdown.get("market_resonance_score", 50.0)) * 0.04
            + float(breakdown.get("launch_specialist_score", 50.0)) * 0.03
            + float(breakdown.get("launch_regime_fit_score", 50.0)) * 0.02
            + float(breakdown.get("context_composite_score", 50.0)) * 0.02
        ),
        2,
    )
    signal_label, risk_label = _strategy_labels(
        latest_probability=guarded_probability,
        strategy_score=strategy_score,
        risk_score=float(breakdown["risk_score"]),
    )
    quality_label, quality_summary = _summarize_quality(metrics)
    precision_gate_threshold, precision_gate_precision, precision_gate_support, precision_gate_active, precision_gate_label = (
        _precision_gate_state(
            metrics,
            guarded_probability,
        )
    )
    backtest_summary = (
        f"本地时间序列回测样本 {int(metrics['sample_size'])} 条，"
        f"AUC {metrics['roc_auc']:.3f}，"
        f"高置信样本平均未来收益 {metrics['top_bucket_return'] * 100:.2f}%。"
    )
    if np.isnan(metrics["roc_auc"]):
        backtest_summary = (
            f"本地时间序列回测样本 {int(metrics['sample_size'])} 条，"
            f"当前标签分布单边，先参考高置信样本未来收益 {metrics['top_bucket_return'] * 100:.2f}%。"
        )
    if metrics.get("precision_target_reached", 0.0):
        backtest_summary = (
            f"{backtest_summary} 历史上当概率不低于 {precision_gate_threshold:.2f} 时，"
            f"上涨命中率约 {precision_gate_precision * 100:.1f}% ，样本 {precision_gate_support} 条。"
        )
    if guard_note:
        backtest_summary = f"{backtest_summary} {guard_note}"
    backtest_summary = f"{backtest_summary} {quality_summary}"
    breakdown.update(guard_info)
    if guard_note:
        breakdown["recent_guard_note"] = guard_note

    probabilities = pd.Series(fitted_probabilities, index=dataset.index, name="probability")
    if not probabilities.empty:
        probabilities.iloc[-1] = float(guarded_probability)
    return ProbabilityResult(
        latest_probability=guarded_probability,
        probabilities=probabilities,
        out_of_sample_probabilities=pd.Series(oof_probabilities, index=dataset.index, name="oof_probability"),
        metrics=metrics,
        coefficients=coefficients,
        strategy_score=strategy_score,
        agreement_score=agreement_score,
        signal_label=signal_label,
        risk_label=risk_label,
        quality_label=quality_label,
        backtest_summary=backtest_summary,
        signal_breakdown=breakdown,
        regime_label=str(_regime_labels_from_feature_frame(latest_features)[-1]) if len(latest_features) else "rotation",
        precision_target=float(metrics.get("precision_target", 0.90)),
        precision_gate_threshold=precision_gate_threshold,
        precision_gate_precision=precision_gate_precision,
        precision_gate_support=precision_gate_support,
        precision_gate_active=precision_gate_active,
        precision_gate_label=precision_gate_label,
        raw_probability=float(latest_probability[-1]),
        enhanced_probability=float(enhanced_latest_probability[-1]),
        predicted_upside_pct=projected_upside_pct,
        predicted_upside_low_pct=projected_upside_low_pct,
        predicted_upside_high_pct=projected_upside_high_pct,
        base_probability=float(latest_probability[-1]),
        upgrade_delta=float(guarded_probability - latest_probability[-1]),
        upgrade_components={
            "daily_k_score": float(breakdown.get("daily_k_score", 50.0)),
            "volume_price_score": float(breakdown.get("volume_price_score", 50.0)),
            "market_context_score": float(breakdown.get("market_context_score", 50.0)),
            "news_fund_score": float(breakdown.get("news_fund_score", 50.0)),
            "quant_context_score": float(breakdown.get("quant_context_score", 50.0)),
            "launch_readiness_score": float(breakdown.get("launch_readiness_score", 50.0)),
            "market_resonance_score": float(breakdown.get("market_resonance_score", 50.0)),
            "launch_specialist_score": float(breakdown.get("launch_specialist_score", 50.0)),
            "launch_regime_fit_score": float(breakdown.get("launch_regime_fit_score", 50.0)),
            "launch_specialist_confidence": float(breakdown.get("launch_specialist_confidence", 50.0)),
            "context_composite_score": float(breakdown.get("context_composite_score", 50.0)),
            "recent_guard_strength": float(guard_info.get("recent_guard_strength", 0.0)),
        },
        upgrade_summary=(
            f"基础概率 {float(latest_probability[-1]) * 100:.1f}% ，"
            f"历史增强后 {float(enhanced_latest_probability[-1]) * 100:.1f}% 。"
        ),
    )
