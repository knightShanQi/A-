from __future__ import annotations

import json
import hashlib
import os
from datetime import date, datetime
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any
import time

import numpy as np
import pandas as pd

from .daily_review import (
    DEFAULT_ROLLING_REVIEW_DAYS,
    load_daily_lightweight_backtest_model,
    load_latest_review_bundle,
    load_latest_review_details,
    load_latest_review_summary,
    load_review_battle_panels,
    run_daily_review_maintenance,
)
from .dashboard import (
    _build_display_board,
    _build_enhanced_focus_board,
    _build_focus_board,
    _build_symbol_detail,
    _detail_display_context,
    _get_async_task_progress,
    _latest_market_close_date,
    _read_market_rankings_cache,
    _refresh_market_rankings_cache_task,
    load_a_share_universe,
    load_latest_close_quick_board,
    load_latest_snapshot_board,
    make_daily_chart,
    make_minute_chart,
)
from .data import parse_watchlist, search_a_share_universe, try_normalize_symbol
from .market_backtest_runner import _normalize_strategy_mode
from .news_impact import analyze_symbol_news_impact
from .services import BacktestService
from .task_registry import TaskRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
DEFAULT_API_PARAMS = {
    "ranking_by": "attention",
    "board_size": 50,
    "horizon_days": 3,
    "positive_return_pct": 10.0,
    "watchlist_text": "",
    "security_scope": "main_board",
}
API_MODEL_CONTRACT_VERSION = "api-probability-launch-v2"
API_TASK_EXECUTOR = ThreadPoolExecutor(max_workers=2)
API_TASK_FUTURES: dict[str, Future] = {}
DEFAULT_BACKTEST_SERVICE = BacktestService()
DEFAULT_TASK_REGISTRY = TaskRegistry(PROJECT_ROOT / ".cache" / "api_task_records.json")
API_ROLLING_REVIEW_TASK_PREFIX = "api-rolling-review"
API_REVIEW_PANEL_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
API_REVIEW_PANEL_CACHE_TTL_SECONDS = 180.0
FOCUS_BOARD_ALLOWED_PREFIXES = ("000", "001", "002", "003", "600", "601", "603", "605")
FOCUS_BOARD_EXCLUDED_PREFIXES = ("300", "301", "688", "689")
DEFAULT_LAUNCH_WINDOW_EXECUTION_WEIGHT = 0.22


def normalize_api_params(
    *,
    ranking_by: str = "attention",
    board_size: int = 50,
    horizon_days: int = 3,
    positive_return_pct: float = 10.0,
    watchlist_text: str = "",
    security_scope: str = "main_board",
) -> dict[str, Any]:
    ranking_token = str(ranking_by or "").strip().lower()
    normalized_ranking = "上涨概率" if ranking_token in {"上涨概率", "probability", "probability_up"} else "关注分数"
    normalized_board_size = int(max(10, min(int(board_size), 100)))
    normalized_horizon = int(horizon_days) if int(horizon_days) in {3, 5, 10} else DEFAULT_API_PARAMS["horizon_days"]
    normalized_positive_return_pct = float(max(5.0, min(float(positive_return_pct), 50.0)))
    normalized_watchlist_text = str(watchlist_text or "")
    scope_token = str(security_scope or "").strip().lower()
    normalized_security_scope = "all" if scope_token in {"all", "raw", "original", "full_market"} else "main_board"
    return {
        "ranking_by": normalized_ranking,
        "board_size": normalized_board_size,
        "horizon_days": normalized_horizon,
        "positive_return_pct": normalized_positive_return_pct,
        "positive_return": normalized_positive_return_pct / 100.0,
        "watchlist_text": normalized_watchlist_text,
        "custom_watchlist": tuple(parse_watchlist(normalized_watchlist_text)),
        "security_scope": normalized_security_scope,
    }


def _clean_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean_value(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return dataframe_records(value)
    if isinstance(value, pd.Series):
        return {str(key): _clean_value(item) for key, item in value.to_dict().items()}
    if isinstance(value, np.ndarray):
        return [_clean_value(item) for item in value.tolist()]
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        if np.isnan(numeric) or np.isinf(numeric):
            return None
        return numeric
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if value is None:
        return None
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "__dict__"):
        return {
            str(key): _clean_value(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return value


def dataframe_records(df: pd.DataFrame, *, limit: int | None = None) -> list[dict[str, Any]]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    working = df.copy()
    if limit is not None:
        working = working.head(limit).copy()
    working = working.replace([np.inf, -np.inf], np.nan)
    return [
        {str(key): _clean_value(value) for key, value in row.items()}
        for row in working.to_dict(orient="records")
    ]


def _filter_focus_board_security_scope(board: pd.DataFrame) -> pd.DataFrame:
    if board.empty or "symbol" not in board.columns:
        return board.copy()
    filtered = board.copy()
    original_attrs = dict(filtered.attrs)
    symbols = filtered["symbol"].astype(str).str.strip().str.zfill(6)
    mask = symbols.str.startswith(FOCUS_BOARD_ALLOWED_PREFIXES) & ~symbols.str.startswith(FOCUS_BOARD_EXCLUDED_PREFIXES)
    if "name" in filtered.columns:
        names = filtered["name"].fillna("").astype(str).str.upper()
        mask &= ~names.str.startswith(("ST", "*ST"))
        mask &= ~names.str.contains("ST", na=False)
    if "board_label" in filtered.columns:
        board_labels = filtered["board_label"].fillna("").astype(str)
        mask &= ~board_labels.str.contains("创业|科创|CHINEXT|STAR|鍒涗笟|绉戝垱", case=False, regex=True, na=False)
    filtered = filtered.loc[mask].copy().reset_index(drop=True)
    if not filtered.empty and "rank" in filtered.columns:
        filtered["rank"] = range(1, len(filtered) + 1)
    filtered.attrs.update(original_attrs)
    filtered.attrs["security_scope"] = "main_board_non_st_ex_growth_star"
    filtered.attrs["excluded_board_prefixes"] = ",".join(FOCUS_BOARD_EXCLUDED_PREFIXES)
    filtered.attrs["raw_row_count"] = int(len(board))
    filtered.attrs["filtered_row_count"] = int(len(filtered))
    filtered.attrs["excluded_row_count"] = int(max(len(board) - len(filtered), 0))
    return filtered


def build_freshness_contract(
    meta: dict[str, Any] | None,
    params: dict[str, Any] | None,
    *,
    source: str,
) -> dict[str, Any]:
    safe_meta = meta or {}
    safe_params = params or {}
    data_date = str(
        safe_meta.get("market_data_date")
        or safe_meta.get("board_date")
        or safe_meta.get("analysis_date")
        or ""
    )
    latest_date = str(safe_meta.get("latest_market_data_date") or data_date or "")
    computed_at = str(safe_meta.get("computed_at") or safe_meta.get("captured_at") or "")
    cache_stale = bool(safe_meta.get("cache_stale", False))
    is_latest_data = bool(data_date and latest_date and data_date == latest_date)
    model_schema_version = str(
        safe_meta.get("model_schema_version")
        or safe_meta.get("model_version")
        or API_MODEL_CONTRACT_VERSION
    )
    model_source_label = str(safe_meta.get("model_source_label") or safe_meta.get("model_source") or "unified-api")
    freshness_label = "latest" if is_latest_data and not cache_stale else "stale"
    if not data_date:
        freshness_label = "unknown"
    consistency_key = "|".join(
        [
            data_date or "unknown-date",
            str(safe_params.get("horizon_days", "")),
            f'{float(safe_params.get("positive_return", 0.0) or 0.0):.4f}',
            str(safe_params.get("ranking_by", "")),
            model_schema_version,
        ]
    )
    return _clean_value(
        {
            "source": source,
            "data_date": data_date or None,
            "latest_market_data_date": latest_date or None,
            "computed_at": computed_at or None,
            "cache_stale": cache_stale,
            "is_latest_data": is_latest_data,
            "is_latest_model_result": bool(is_latest_data and not cache_stale),
            "freshness_label": freshness_label,
            "model_schema_version": model_schema_version,
            "model_source_label": model_source_label,
            "consistency_key": consistency_key,
        }
    )


def apply_probability_contract(
    df: pd.DataFrame,
    *,
    launch_window_execution_weight: float = DEFAULT_LAUNCH_WINDOW_EXECUTION_WEIGHT,
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    normalized_execution_weight = float(max(0.0, min(float(launch_window_execution_weight), 1.0)))
    board = df.copy()

    def numeric_series(column: str, default: float = 0.0) -> pd.Series:
        if column in board.columns:
            return pd.to_numeric(board[column], errors="coerce").fillna(default)
        return pd.Series(float(default), index=board.index, dtype=float)

    market_state = board.get("market_state_label", board.get("market_regime_label", "unknown"))
    if isinstance(market_state, pd.Series):
        board["market_state_label"] = market_state.fillna("unknown").astype(str).replace("", "unknown")
    else:
        board["market_state_label"] = "unknown"

    state_prior = board["market_state_label"].map(
        {
            "trend": 57.0,
            "rebound": 53.0,
            "rotation": 50.0,
            "defense": 46.0,
            "unknown": 51.0,
        }
    ).fillna(51.0)
    probability_confidence = numeric_series("probability_confidence", np.nan)
    if probability_confidence.isna().all():
        probability_confidence = numeric_series("selection_confidence", np.nan)
    if probability_confidence.isna().all():
        probability_confidence = numeric_series("action_confidence", np.nan)
    probability_confidence = probability_confidence.fillna(0.62)
    probability_confidence = pd.Series(
        np.where(probability_confidence > 1.0, probability_confidence / 100.0, probability_confidence),
        index=board.index,
        dtype=float,
    ).clip(lower=0.25, upper=0.92)

    explicit_probability = False
    if "calibrated_probability_up" in board.columns:
        calibrated = numeric_series("calibrated_probability_up")
        explicit_probability = True
    elif "display_probability_up" in board.columns:
        calibrated = numeric_series("display_probability_up")
        explicit_probability = True
    else:
        raw_for_calibration = numeric_series("probability_up", np.nan)
        if raw_for_calibration.isna().all():
            raw_for_calibration = numeric_series("model_probability_up", np.nan)
        raw_for_calibration = raw_for_calibration.fillna(state_prior)
        calibrated = raw_for_calibration * probability_confidence + state_prior * (1.0 - probability_confidence)
    raw = numeric_series("raw_probability_up", np.nan)
    if raw.isna().all():
        raw = numeric_series("model_probability_up", np.nan)
    if raw.isna().all():
        raw = numeric_series("probability_up", np.nan)
    raw = raw.fillna(calibrated)
    board["raw_probability_up"] = raw.clip(lower=0.0, upper=100.0)
    enhanced = numeric_series("enhanced_probability_up", np.nan)
    if enhanced.isna().all():
        enhanced = numeric_series("pre_replay_probability_up", np.nan)
    if enhanced.isna().all():
        enhanced = numeric_series("model_probability_up", np.nan)
    if enhanced.isna().all():
        enhanced = numeric_series("probability_up", np.nan)
    enhanced = enhanced.fillna(board["raw_probability_up"])
    board["enhanced_probability_up"] = enhanced.clip(lower=0.0, upper=100.0)
    if explicit_probability:
        board["calibration_method"] = "explicit"
    else:
        board["calibration_method"] = "market_state_shrinkage"
    board["probability_confidence"] = (probability_confidence * 100.0).clip(lower=25.0, upper=92.0)
    board["probability_prior_up"] = state_prior.clip(lower=0.0, upper=100.0)
    board["calibrated_probability_up"] = calibrated.clip(lower=32.0, upper=88.0)
    board["probability_up"] = board["calibrated_probability_up"]
    uncertainty = (1.0 - probability_confidence) * 18.0 + 3.0
    board["probability_band_low"] = (board["calibrated_probability_up"] - uncertainty).clip(lower=0.0, upper=100.0)
    board["probability_band_high"] = (board["calibrated_probability_up"] + uncertainty).clip(lower=0.0, upper=100.0)
    expected_return = numeric_series("expected_return_pct", np.nan)
    if expected_return.isna().all():
        expected_return = numeric_series("predicted_upside_pct")
    expected_return = expected_return.fillna(0.0)
    drawdown_risk = numeric_series("drawdown_risk_pct", np.nan)
    if drawdown_risk.isna().all():
        downside_proxy = numeric_series("predicted_upside_low_pct")
        drawdown_risk = (
            (100.0 - board["calibrated_probability_up"]) * 0.08
            + (10.0 - expected_return).clip(lower=0.0) * 0.22
            + downside_proxy.clip(upper=0.0).abs() * 0.55
        )
    market_state = board.get("market_state_label", board.get("market_regime_label", "unknown"))
    if isinstance(market_state, pd.Series):
        board["market_state_label"] = market_state.fillna("unknown").astype(str).replace("", "unknown")
    else:
        board["market_state_label"] = "unknown"
    board["market_state_display"] = board["market_state_label"].map(
        {
            "trend": "趋势扩散",
            "rebound": "修复反弹",
            "rotation": "轮动震荡",
            "defense": "防守退潮",
            "unknown": "状态待定",
        }
    ).fillna(board["market_state_label"])
    board["p_hit"] = board["calibrated_probability_up"]
    board["expected_return_pct"] = expected_return.clip(lower=-20.0, upper=60.0)
    board["drawdown_risk_pct"] = drawdown_risk.fillna(0.0).clip(lower=0.0, upper=35.0)
    final_rank = numeric_series("final_rank_score", np.nan)
    if final_rank.isna().all():
        final_rank = numeric_series("ranking_score", np.nan)
    if final_rank.isna().all():
        final_rank = numeric_series("enhanced_attention_score", np.nan)
    if final_rank.isna().all():
        final_rank = numeric_series("attention_score", np.nan)
    board["final_rank_score"] = final_rank.fillna(0.0).clip(lower=0.0, upper=100.0)

    intraday_score = numeric_series("intraday_execution_score", np.nan)
    if intraday_score.isna().all():
        intraday_score = numeric_series("intraday_score", np.nan)
    if intraday_score.isna().all():
        intraday_score = numeric_series("intraday_structure_score", np.nan)
    intraday_score = intraday_score.fillna(50.0)
    intraday_score = pd.Series(
        np.where((intraday_score >= 0.0) & (intraday_score <= 1.0), intraday_score * 100.0, intraday_score),
        index=board.index,
        dtype=float,
    ).clip(lower=0.0, upper=100.0)
    market_resonance = numeric_series("market_resonance_score", 50.0).clip(lower=0.0, upper=100.0)
    sector_strength = numeric_series("sector_strength_score", np.nan)
    if sector_strength.isna().all():
        sector_strength = numeric_series("sector_score", np.nan)
    if sector_strength.isna().all():
        sector_strength = numeric_series("sector_fund_score", np.nan)
    sector_strength = sector_strength.fillna(market_resonance).clip(lower=0.0, upper=100.0)
    relative_intraday_alpha = (intraday_score - sector_strength).clip(lower=-50.0, upper=50.0)
    alignment_bonus = pd.Series(
        np.select(
            [
                (intraday_score >= 55.0) & (sector_strength >= 55.0),
                (intraday_score <= 45.0) & (sector_strength <= 45.0),
            ],
            [8.0, -8.0],
            default=-relative_intraday_alpha.abs() * 0.12,
        ),
        index=board.index,
        dtype=float,
    )
    sector_follow_score = (
        50.0
        + (sector_strength - 50.0) * 0.50
        + (intraday_score - 50.0) * 0.25
        - relative_intraday_alpha.abs() * 0.10
    ).clip(lower=0.0, upper=100.0)
    board["intraday_strength_score"] = intraday_score
    board["sector_strength_score"] = sector_strength
    board["relative_intraday_alpha"] = relative_intraday_alpha
    board["sector_follow_score"] = sector_follow_score
    board["intraday_sector_sync_score"] = (
        50.0
        + (intraday_score - 50.0) * 0.38
        + (sector_strength - 50.0) * 0.38
        + (market_resonance - 50.0) * 0.12
        + alignment_bonus
    ).clip(lower=0.0, upper=100.0)
    board["intraday_sector_state"] = np.select(
        [
            (board["intraday_sector_sync_score"] >= 68.0) & (relative_intraday_alpha >= -6.0),
            (sector_strength >= 62.0) & (intraday_score < 55.0),
            (intraday_score >= 62.0) & (sector_strength < 52.0),
            (relative_intraday_alpha.abs() >= 18.0),
        ],
        ["confirmed_sync", "sector_lead_wait", "stock_lead_watch", "divergence_risk"],
        default="neutral_sync",
    )
    board["intraday_sector_display"] = board["intraday_sector_state"].map(
        {
            "confirmed_sync": "分时板块共振",
            "sector_lead_wait": "板块强个股待跟随",
            "stock_lead_watch": "个股领先待板块确认",
            "divergence_risk": "分时板块背离",
            "neutral_sync": "联动中性",
        }
    )
    board["intraday_sector_note"] = np.select(
        [
            board["intraday_sector_state"] == "confirmed_sync",
            board["intraday_sector_state"] == "sector_lead_wait",
            board["intraday_sector_state"] == "stock_lead_watch",
            board["intraday_sector_state"] == "divergence_risk",
        ],
        [
            "个股分时和板块热度同向增强，可提高启动确认权重。",
            "板块先走强但个股分时承接不足，等待跟随确认。",
            "个股分时先于板块异动，需要观察板块扩散。",
            "个股分时与板块方向背离，降低追涨优先级。",
        ],
        default="分时与板块暂未形成明确信号。",
    )
    crowding_risk = numeric_series("crowding_risk", np.nan)
    if crowding_risk.isna().all():
        turnover = numeric_series("turnover", 0.0).clip(lower=0.0, upper=22.0)
        risk_of_late_entry = numeric_series("risk_of_late_entry", 50.0).clip(lower=0.0, upper=100.0)
        crowding_risk = (
            16.0
            + turnover * 1.10
            + board["drawdown_risk_pct"].clip(lower=0.0, upper=35.0) * 0.65
            + (100.0 - board["intraday_sector_sync_score"]).clip(lower=0.0, upper=100.0) * 0.10
            + risk_of_late_entry * 0.28
            - sector_strength.clip(lower=0.0, upper=100.0) * 0.08
        )
    board["crowding_risk"] = crowding_risk.fillna(50.0).clip(lower=0.0, upper=100.0)
    board["crowding_risk_label"] = np.select(
        [
            board["crowding_risk"] >= 76.0,
            board["crowding_risk"] >= 58.0,
            board["crowding_risk"] <= 38.0,
        ],
        ["量化拥挤", "偏拥挤", "拥挤低"],
        default="正常",
    )
    board_resonance_strength = numeric_series("board_resonance_strength", np.nan)
    if board_resonance_strength.isna().all():
        board_resonance_strength = (sector_strength * 0.62 + board["intraday_sector_sync_score"] * 0.38)
    board["board_resonance_strength"] = board_resonance_strength.fillna(50.0).clip(lower=0.0, upper=100.0)
    long_setup_quality = numeric_series("long_setup_quality", np.nan)
    if long_setup_quality.isna().all():
        launch_context_score = numeric_series("launch_window_score", np.nan)
        if launch_context_score.isna().all():
            launch_context_score = numeric_series("launch_score", 50.0)
        launch_context_score = launch_context_score.fillna(50.0).clip(lower=0.0, upper=100.0)
        long_setup_quality = (
            board["board_resonance_strength"] * 0.25
            + launch_context_score * 0.22
            + board["p_hit"] * 0.16
            + numeric_series("quant_score", 50.0).clip(lower=0.0, upper=100.0) * 0.12
            + board["intraday_sector_sync_score"] * 0.15
            - board["crowding_risk"] * 0.10
        )
    board["long_setup_quality"] = long_setup_quality.fillna(50.0).clip(lower=0.0, upper=100.0)
    fallback_rank_score = (
        board["p_hit"] * 0.35
        + board["expected_return_pct"].clip(lower=0.0, upper=25.0)
        + numeric_series("selection_score", np.nan).fillna(numeric_series("enhanced_attention_score", 50.0)) * 0.20
        + market_resonance * 0.08
        + board["intraday_sector_sync_score"] * 0.08
        + board["long_setup_quality"] * 0.10
        - board["drawdown_risk_pct"] * 0.45
        - board["crowding_risk"] * 0.08
    )
    rank_score = numeric_series("rank_score", np.nan)
    if rank_score.isna().all():
        rank_score = numeric_series("ranking_score", np.nan)
    rank_score = rank_score.fillna(fallback_rank_score)
    board["rank_score"] = rank_score.clip(lower=0.0, upper=100.0)

    launch_window_score = numeric_series("launch_window_score", np.nan)
    if launch_window_score.isna().all():
        launch_window_score = numeric_series("launch_score", np.nan)
    launch_window_score = launch_window_score.fillna(
        (
            board["p_hit"] * 0.28
            + board["expected_return_pct"].clip(lower=0.0, upper=25.0) * 1.15
            + numeric_series("execution_score", 50.0) * normalized_execution_weight
            - board["drawdown_risk_pct"] * 0.35
        )
    )
    board["launch_window_score"] = launch_window_score.clip(lower=0.0, upper=100.0)

    launch_specialist_score = numeric_series("launch_specialist_score", np.nan)
    if launch_specialist_score.isna().all():
        launch_specialist_score = (
            board["launch_window_score"] * 0.36
            + board["rank_score"] * 0.28
            + numeric_series("volume_price_score", np.nan).fillna(numeric_series("quant_score", 50.0)) * 0.18
            + numeric_series("sector_strength_score", np.nan).fillna(numeric_series("market_resonance_score", 50.0)) * 0.18
        )
    board["launch_specialist_score"] = launch_specialist_score.clip(lower=0.0, upper=100.0)

    launch_signal_score = numeric_series("launch_signal_score", np.nan)
    if launch_signal_score.isna().all():
        launch_signal_score = (
            board["launch_specialist_score"] * 0.48
            + board["p_hit"] * 0.24
            + board["expected_return_pct"].clip(lower=0.0, upper=25.0) * 0.90
            - board["drawdown_risk_pct"] * 0.40
        )
    board["launch_signal_score"] = launch_signal_score.clip(lower=0.0, upper=100.0)
    board["launch_signal_label"] = np.select(
        [
            board["launch_signal_score"] >= 78.0,
            board["launch_signal_score"] >= 66.0,
            board["launch_signal_score"] >= 54.0,
        ],
        ["breakout", "ready", "watch"],
        default="wait",
    )
    board["launch_signal_display"] = board["launch_signal_label"].map(
        {
            "breakout": "突破确认",
            "ready": "启动就绪",
            "watch": "观察蓄势",
            "wait": "等待确认",
        }
    )
    volume_price = numeric_series("volume_price_score", np.nan).fillna(numeric_series("quant_score", 50.0))
    board["launch_phase_label"] = np.select(
        [
            board["crowding_risk"] >= 76.0,
            (board["launch_signal_score"] >= 78.0) & (board["expected_return_pct"] >= 10.0),
            (board["launch_signal_score"] >= 66.0) & (volume_price >= 58.0),
            (board["launch_signal_score"] >= 54.0) & (board["drawdown_risk_pct"] <= 8.0),
        ],
        ["crowded", "breakout_confirmed", "pre_launch", "pullback_setup"],
        default="wait_confirm",
    )
    board["launch_phase_display"] = board["launch_phase_label"].map(
        {
            "crowded": "量化拥挤期",
            "breakout_confirmed": "突破确认期",
            "pre_launch": "主升预备期",
            "pullback_setup": "缩量回踩期",
            "wait_confirm": "等待确认期",
        }
    )
    reasons: list[str] = []
    for idx in board.index:
        row_reasons: list[str] = []
        if board.at[idx, "p_hit"] >= 60.0:
            row_reasons.append("达标概率占优")
        if board.at[idx, "expected_return_pct"] >= 8.0:
            row_reasons.append("期望涨幅达标")
        if board.at[idx, "drawdown_risk_pct"] <= 8.0:
            row_reasons.append("回撤风险可控")
        if float(sector_strength.loc[idx]) >= 60.0:
            row_reasons.append("板块共振")
        if float(board.at[idx, "intraday_sector_sync_score"]) >= 65.0:
            row_reasons.append("分时联动确认")
        if float(volume_price.loc[idx]) >= 58.0:
            row_reasons.append("量价结构支持")
        if float(board.at[idx, "crowding_risk"]) >= 76.0:
            row_reasons.append("量化拥挤需降权")
        reasons.append(" / ".join(row_reasons[:3]) if row_reasons else "等待量价和板块确认")
    board["launch_reason_text"] = reasons
    board["risk_level_label"] = np.select(
        [
            (board["drawdown_risk_pct"] <= 5.0) & (board["p_hit"] >= 62.0),
            board["drawdown_risk_pct"] <= 10.0,
            (board["drawdown_risk_pct"] <= 18.0) & (board["intraday_sector_sync_score"] >= 38.0),
        ],
        ["low", "medium", "high"],
        default="extreme",
    )
    board["risk_level_display"] = board["risk_level_label"].map(
        {
            "low": "低风险",
            "medium": "中性风险",
            "high": "高风险",
            "extreme": "极高风险",
        }
    )
    board["suggested_position_pct"] = np.select(
        [
            (board["risk_level_label"] == "low") & (board["launch_signal_label"].isin(["breakout", "ready"])),
            board["risk_level_label"].isin(["low", "medium"]),
            board["risk_level_label"] == "high",
        ],
        [30.0, 20.0, 10.0],
        default=0.0,
    )
    board["stop_loss_pct"] = np.select(
        [
            board["risk_level_label"] == "low",
            board["risk_level_label"] == "medium",
            board["risk_level_label"] == "high",
        ],
        [4.0, 6.0, 8.0],
        default=0.0,
    )
    board["take_profit_pct"] = board["expected_return_pct"].clip(lower=5.0, upper=30.0)
    board["risk_control_note"] = np.select(
        [
            board["suggested_position_pct"] <= 0.0,
            board["risk_level_label"] == "high",
            board["risk_level_label"] == "medium",
        ],
        ["只观察，不开仓", "轻仓试错，跌破失效位先处理", "分批参与，等待分时确认"],
        default="可按计划小仓试错",
    )
    return board


def _board_meta(board: pd.DataFrame, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    strategy_counts = {"strategy1": 0, "strategy2": 0, "strategy3": 0}
    strategy_notice = ""
    if isinstance(board, pd.DataFrame) and not board.empty and "candidate_strategy" in board.columns:
        labels = board["candidate_strategy"].fillna("").astype(str)
        lower = labels.str.lower()
        strategy_counts = {
            "strategy1": int((labels.str.contains("策略1", regex=False) | lower.str.contains("strategy1", regex=False)).sum()),
            "strategy2": int((labels.str.contains("策略2", regex=False) | lower.str.contains("strategy2", regex=False)).sum()),
            "strategy3": int((labels.str.contains("策略3", regex=False) | lower.str.contains("strategy3", regex=False)).sum()),
        }
        missing = []
        if strategy_counts["strategy1"] <= 0:
            missing.append("策略1")
        if strategy_counts["strategy2"] <= 0:
            missing.append("策略2")
        if missing:
            strategy_notice = (
                f"今日{'、'.join(missing)}未筛选出股票；当前关注榜由策略3多因子主升预备池补位，"
                "请按策略3口径观察，等待策略1/2硬条件重新出现。"
            )
    meta = {
        "data_mode": board.attrs.get("data_mode"),
        "market_data_date": board.attrs.get("market_data_date"),
        "latest_market_data_date": board.attrs.get("latest_market_data_date"),
        "cache_stale": bool(board.attrs.get("cache_stale", False)),
        "computed_at": board.attrs.get("computed_at"),
        "horizon_days": board.attrs.get("horizon_days"),
        "positive_return": board.attrs.get("positive_return"),
        "model_source": board.attrs.get("model_source"),
        "model_source_label": board.attrs.get("model_source_label"),
        "model_schema_version": board.attrs.get("model_schema_version"),
        "ranking_by": board.attrs.get("ranking_by"),
        "security_scope": board.attrs.get("security_scope"),
        "excluded_board_prefixes": board.attrs.get("excluded_board_prefixes"),
        "raw_row_count": board.attrs.get("raw_row_count"),
        "filtered_row_count": board.attrs.get("filtered_row_count"),
        "excluded_row_count": board.attrs.get("excluded_row_count"),
        "row_count": int(len(board)),
        "strategy_counts": strategy_counts,
        "strategy_notice": strategy_notice,
    }
    if extra:
        meta.update(extra)
    return _clean_value(meta)


def serialize_board(
    board: pd.DataFrame,
    *,
    extra_meta: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    source: str = "board",
) -> dict[str, Any]:
    requested_scope = str((params or {}).get("security_scope") or "main_board")
    if requested_scope == "all":
        scoped_board = board.copy()
        scoped_board.attrs.update(getattr(board, "attrs", {}))
        scoped_board.attrs["security_scope"] = "all"
        scoped_board.attrs["excluded_board_prefixes"] = ""
        scoped_board.attrs["raw_row_count"] = int(len(board))
        scoped_board.attrs["filtered_row_count"] = int(len(scoped_board))
        scoped_board.attrs["excluded_row_count"] = 0
    else:
        scoped_board = _filter_focus_board_security_scope(board)
    normalized_board = apply_probability_contract(scoped_board)
    normalized_board.attrs.update(scoped_board.attrs)
    rows = dataframe_records(normalized_board)
    meta = _board_meta(normalized_board, extra=extra_meta)
    freshness = build_freshness_contract(meta, params, source=source)
    return {
        "rows": rows,
        "top_row": rows[0] if rows else None,
        "meta": meta,
        "freshness": freshness,
    }


def _rolling_review_task_id(params: dict[str, Any], board: pd.DataFrame) -> str:
    data_date = str(board.attrs.get("market_data_date") or board.attrs.get("latest_market_data_date") or "unknown")
    ranking_token = hashlib.sha1(str(params.get("ranking_by") or "").encode("utf-8")).hexdigest()[:8]
    return (
        f'{API_ROLLING_REVIEW_TASK_PREFIX}::{params["horizon_days"]}::'
        f'{float(params["positive_return"]):.4f}::{params["board_size"]}::{ranking_token}::{data_date}'
    )


def start_rolling_review_profile_task(params: dict[str, Any], board: pd.DataFrame) -> dict[str, Any]:
    if params.get("custom_watchlist"):
        return {"task_id": "", "status": "skipped", "reason": "custom_watchlist"}
    if str(params.get("security_scope") or "main_board") != "main_board":
        return {"task_id": "", "status": "skipped", "reason": "raw_universe_scope"}
    if not isinstance(board, pd.DataFrame) or board.empty:
        return {"task_id": "", "status": "skipped", "reason": "empty_board"}

    review_board = _filter_focus_board_security_scope(board)
    if len(review_board) < 10:
        return {"task_id": "", "status": "skipped", "reason": "insufficient_board_rows"}

    task_id = _rolling_review_task_id(params, review_board)
    future = API_TASK_FUTURES.get(task_id)
    if future is None:
        review_board = review_board.copy()
        API_TASK_FUTURES[task_id] = API_TASK_EXECUTOR.submit(
            run_daily_review_maintenance,
            review_board,
            horizon_days=params["horizon_days"],
            positive_return=params["positive_return"],
            ranking_by=params["ranking_by"],
            board_size=params["board_size"],
            rolling_review_days=DEFAULT_ROLLING_REVIEW_DAYS,
        )
    return get_task_status(task_id)


def serialize_market_context(market_context: dict[str, pd.DataFrame]) -> dict[str, Any]:
    return {
        "industry_flow": dataframe_records(market_context.get("industry_flow", pd.DataFrame()), limit=12),
        "concept_flow": dataframe_records(market_context.get("concept_flow", pd.DataFrame()), limit=12),
        "macro_calendar": dataframe_records(market_context.get("macro_calendar", pd.DataFrame()), limit=10),
    }


def serialize_review_panels(
    *,
    horizon_days: int,
    positive_return: float,
    ranking_by: str,
    board_size: int,
) -> dict[str, Any]:
    panels = load_review_battle_panels(
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
        board_size=board_size,
    )
    return {
        "strategy_panel": dataframe_records(panels.get("strategy_panel", pd.DataFrame())),
        "short_market_state_panel": dataframe_records(panels.get("short_market_state_panel", pd.DataFrame())),
        "long_market_state_panel": dataframe_records(panels.get("long_market_state_panel", pd.DataFrame())),
        "combo_panel": dataframe_records(panels.get("combo_panel", pd.DataFrame())),
        "meta": _clean_value(panels.get("meta", {})),
    }


def serialize_daily_lightweight_model(params: dict[str, Any]) -> dict[str, Any]:
    model = load_daily_lightweight_backtest_model(
        horizon_days=params["horizon_days"],
        positive_return=params["positive_return"],
        ranking_by=params["ranking_by"],
        board_size=params["board_size"],
    )
    if not model:
        return {
            "status": "missing",
            "sample_count": 0,
            "review_days": 0,
            "model_parameter_update_allowed": False,
            "calibration_scope": "independent_daily_lightweight_backtest_only",
            "panels": {},
        }
    return _clean_value(model)


def load_cached_review_panels(params: dict[str, Any]) -> dict[str, Any]:
    cache_key = "|".join(
        [
            str(params.get("horizon_days")),
            f'{float(params.get("positive_return", 0.0) or 0.0):.4f}',
            str(params.get("ranking_by")),
            str(params.get("board_size")),
        ]
    )
    now = time.monotonic()
    cached = API_REVIEW_PANEL_CACHE.get(cache_key)
    if cached and now - cached[0] <= API_REVIEW_PANEL_CACHE_TTL_SECONDS:
        payload = dict(cached[1])
        meta = dict(payload.get("meta", {}))
        meta["cache_hit"] = True
        payload["meta"] = meta
        return payload
    payload = serialize_review_panels(
        horizon_days=params["horizon_days"],
        positive_return=params["positive_return"],
        ranking_by=params["ranking_by"],
        board_size=params["board_size"],
    )
    meta = dict(payload.get("meta", {}))
    meta["cache_hit"] = False
    payload["meta"] = meta
    API_REVIEW_PANEL_CACHE[cache_key] = (now, payload)
    return payload


def build_review_health(
    review_summary: dict[str, Any] | None,
    review_panels: dict[str, Any] | None,
    params: dict[str, Any] | None,
    lightweight_model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = review_summary or {}
    panels = review_panels or {}
    lightweight = lightweight_model or {}
    target_precision_pct = 90.0
    win_rate_pct = float(summary.get("win_rate_pct") or summary.get("direction_hit_rate_pct") or 0.0)
    target_hit_rate_pct = float(summary.get("target_hit_rate_pct") or 0.0)
    avg_return_pct = float(summary.get("avg_return_pct") or 0.0)
    calibration_gap_pct = float(summary.get("calibration_gap_pct") or 0.0)
    review_count = int(summary.get("review_count") or 0)
    panel_sample_count = 0
    for panel_name in ("strategy_panel", "short_market_state_panel", "long_market_state_panel", "combo_panel"):
        for row in panels.get(panel_name, []) or []:
            try:
                panel_sample_count += int(row.get("sample_count") or 0)
            except Exception:
                continue
    sample_count = max(review_count, panel_sample_count)
    if sample_count < 10:
        status_label = "insufficient_sample"
        status_display = "样本不足"
    elif win_rate_pct >= target_precision_pct:
        status_label = "target_met"
        status_display = "达到目标"
    elif win_rate_pct >= 60.0 and avg_return_pct > 0.0:
        status_label = "usable"
        status_display = "可辅助参考"
    else:
        status_label = "needs_review"
        status_display = "需要复盘优化"
    health_score = (
        win_rate_pct * 0.48
        + target_hit_rate_pct * 0.22
        + max(avg_return_pct, 0.0) * 3.0
        - calibration_gap_pct * 0.30
        + min(sample_count, 200) * 0.04
    )
    return _clean_value(
        {
            "status_label": status_label,
            "status_display": status_display,
            "target_precision_pct": target_precision_pct,
            "win_rate_pct": round(win_rate_pct, 2),
            "target_hit_rate_pct": round(target_hit_rate_pct, 2),
            "avg_return_pct": round(avg_return_pct, 2),
            "calibration_gap_pct": round(calibration_gap_pct, 2),
            "precision_gap_to_target_pct": round(target_precision_pct - win_rate_pct, 2),
            "sample_count": sample_count,
            "health_score": round(float(np.clip(health_score, 0.0, 100.0)), 2),
            "horizon_days": (params or {}).get("horizon_days"),
            "positive_return_pct": round(float((params or {}).get("positive_return", 0.0) or 0.0) * 100.0, 2),
            "daily_lightweight_status": str(lightweight.get("status") or "missing"),
            "daily_lightweight_sample_count": int(lightweight.get("sample_count") or 0),
            "daily_lightweight_review_days": int(lightweight.get("review_days") or 0),
            "daily_lightweight_model_independent": bool(lightweight)
            and not bool(lightweight.get("model_parameter_update_allowed", True)),
        }
    )


def build_model_snapshot(
    params: dict[str, Any] | None,
    freshness: dict[str, Any] | None,
    review_health: dict[str, Any] | None = None,
    *,
    source: str,
) -> dict[str, Any]:
    safe_params = params or {}
    safe_freshness = freshness or {}
    safe_health = review_health or {}
    version_payload = {
        "contract": API_MODEL_CONTRACT_VERSION,
        "source": source,
        "data_date": safe_freshness.get("data_date"),
        "horizon_days": safe_params.get("horizon_days"),
        "positive_return_pct": round(float(safe_params.get("positive_return", 0.0) or 0.0) * 100.0, 2),
        "ranking_by": safe_params.get("ranking_by"),
        "board_size": safe_params.get("board_size"),
        "health_score": safe_health.get("health_score"),
        "win_rate_pct": safe_health.get("win_rate_pct"),
    }
    encoded = json.dumps(version_payload, ensure_ascii=False, sort_keys=True, default=str)
    signature = hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]
    return _clean_value(
        {
            "model_version_id": f"{API_MODEL_CONTRACT_VERSION}-{signature}",
            "model_schema_version": API_MODEL_CONTRACT_VERSION,
            "source": source,
            "data_date": safe_freshness.get("data_date"),
            "latest_market_data_date": safe_freshness.get("latest_market_data_date"),
            "horizon_days": version_payload["horizon_days"],
            "positive_return_pct": version_payload["positive_return_pct"],
            "ranking_by": version_payload["ranking_by"],
            "board_size": version_payload["board_size"],
            "health_score": safe_health.get("health_score"),
            "win_rate_pct": safe_health.get("win_rate_pct"),
            "target_precision_pct": safe_health.get("target_precision_pct", 90.0),
            "status_label": safe_health.get("status_label"),
            "status_display": safe_health.get("status_display"),
            "signature": signature,
            "comparison_key": encoded,
            "is_comparable": bool(safe_freshness.get("data_date") and version_payload["horizon_days"]),
        }
    )


def resolve_symbol(symbol_or_query: str) -> str:
    normalized = try_normalize_symbol(symbol_or_query)
    if normalized:
        return normalized
    universe = load_a_share_universe()
    matches = search_a_share_universe(universe, symbol_or_query, limit=1)
    if matches.empty:
        raise ValueError(f"未找到股票：{symbol_or_query}")
    return str(matches.iloc[0]["symbol"]).zfill(6)


def _lookup_symbol_name(symbol: str) -> str:
    try:
        universe = load_a_share_universe()
        if not universe.empty and {"symbol", "name"}.issubset(universe.columns):
            matched = universe[universe["symbol"].astype(str).str.zfill(6) == str(symbol).zfill(6)]
            if not matched.empty:
                return str(matched.iloc[0].get("name") or symbol)
    except Exception:
        pass
    return str(symbol)


def _find_cached_board_row(symbol: str, params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = load_quick_board_payload(params)
    rows = payload.get("board", {}).get("rows", []) or []
    resolved = str(symbol).zfill(6)
    for row in rows:
        if str(row.get("symbol", "")).zfill(6) == resolved:
            return dict(row), payload
    return {}, payload


def _build_safe_symbol_detail_payload(
    resolved_symbol: str,
    params: dict[str, Any],
    *,
    reason: str = "live_detail_disabled",
) -> dict[str, Any]:
    cached_row, board_payload = _find_cached_board_row(resolved_symbol, params)
    name = str(cached_row.get("name") or _lookup_symbol_name(resolved_symbol))
    display_context = dict(cached_row)
    display_context.setdefault("symbol", resolved_symbol)
    display_context.setdefault("name", name)
    display_context.setdefault("action_label", cached_row.get("action_label") or "观察")
    display_context.setdefault("probability_up", cached_row.get("probability_up", cached_row.get("p_hit", 50.0)))
    display_context.setdefault("raw_probability_up", cached_row.get("raw_probability_up", display_context["probability_up"]))
    display_context.setdefault("enhanced_probability_up", cached_row.get("enhanced_probability_up", display_context["probability_up"]))
    display_context.setdefault("calibrated_probability_up", cached_row.get("calibrated_probability_up", display_context["probability_up"]))
    display_context.setdefault("display_probability_up", cached_row.get("display_probability_up", display_context["probability_up"]))
    display_context.setdefault("attention_score", cached_row.get("attention_score", cached_row.get("rank_score", 50.0)))
    display_context.setdefault("enhanced_attention_score", cached_row.get("enhanced_attention_score", display_context["attention_score"]))
    display_context.setdefault("rank_score", cached_row.get("rank_score", cached_row.get("ranking_score", display_context["attention_score"])))
    display_context.setdefault("final_rank_score", cached_row.get("final_rank_score", display_context["rank_score"]))
    display_context.setdefault("p_hit", cached_row.get("p_hit", display_context["probability_up"]))
    display_context.setdefault("expected_return_pct", cached_row.get("expected_return_pct", 0.0))
    display_context.setdefault("drawdown_risk_pct", cached_row.get("drawdown_risk_pct", 0.0))
    display_context.setdefault("probability_confidence", cached_row.get("probability_confidence", 62.0))
    display_context.setdefault("market_state_label", cached_row.get("market_state_label", "unknown"))
    display_context.setdefault("quant_score", cached_row.get("quant_score", 50.0))
    display_context.setdefault("launch_score", cached_row.get("launch_score", 50.0))
    display_context.setdefault("market_resonance_score", cached_row.get("market_resonance_score", 50.0))
    display_context.setdefault("launch_signal_label", cached_row.get("launch_signal_label") or "wait")
    display_context.setdefault("launch_signal_display", cached_row.get("launch_signal_display") or "等待确认")
    display_context.setdefault("risk_level_display", cached_row.get("risk_level_display") or "等待确认")
    display_context.setdefault("detail_mode", "safe_cached")
    display_context.setdefault("detail_fallback_reason", reason)
    display_context.setdefault("detail_note", "实时 K 线/分时数据暂未拉取，本卡片先展示缓存榜单里的模型评分与交易状态。")
    contract_context = apply_probability_contract(pd.DataFrame([display_context]))
    if not contract_context.empty:
        contract_row = contract_context.iloc[0].to_dict()
        for field in (
            "raw_probability_up",
            "enhanced_probability_up",
            "calibrated_probability_up",
            "probability_up",
            "calibration_method",
            "probability_confidence",
            "probability_prior_up",
            "probability_band_low",
            "probability_band_high",
            "p_hit",
            "expected_return_pct",
            "drawdown_risk_pct",
            "final_rank_score",
            "rank_score",
            "market_state_display",
            "intraday_strength_score",
            "sector_strength_score",
            "relative_intraday_alpha",
            "sector_follow_score",
            "intraday_sector_sync_score",
            "intraday_sector_state",
            "intraday_sector_display",
            "intraday_sector_note",
            "board_resonance_strength",
            "long_setup_quality",
            "crowding_risk",
            "crowding_risk_label",
            "launch_window_score",
            "launch_specialist_score",
            "launch_signal_score",
            "launch_signal_label",
            "launch_signal_display",
            "launch_phase_label",
            "launch_phase_display",
            "launch_reason_text",
            "risk_level_label",
            "risk_level_display",
            "suggested_position_pct",
            "stop_loss_pct",
            "take_profit_pct",
            "risk_control_note",
        ):
            display_context[field] = contract_row.get(field, display_context.get(field))
    freshness = (
        board_payload.get("freshness")
        or board_payload.get("board", {}).get("freshness")
        or build_freshness_contract({}, params, source="symbol_detail_safe")
    )
    freshness = dict(freshness)
    freshness["source"] = "symbol_detail_safe"
    model_snapshot = build_model_snapshot(params, freshness, None, source="symbol_detail_safe")
    display_context["freshness"] = freshness
    display_context["model_snapshot"] = model_snapshot
    return {
        "symbol": resolved_symbol,
        "params": _clean_value(params),
        "freshness": _clean_value(freshness),
        "model_snapshot": _clean_value(model_snapshot),
        "display_context": _clean_value(display_context),
        "hero": {
            "name": name,
            "industry": str(cached_row.get("industry") or cached_row.get("sector") or "待补充"),
            "board_label": str(cached_row.get("board_label") or "A股"),
            "price_limit_label": str(cached_row.get("price_limit_label") or "--"),
            "analysis_date": freshness.get("data_date"),
            "latest_market_data_date": freshness.get("latest_market_data_date"),
            "intraday_label": "安全缓存详情：实时分时走势后台补齐后展示。",
            "stage_label": str(cached_row.get("stage_label") or cached_row.get("launch_phase_display") or "等待确认"),
        },
        "charts": {
            "daily": None,
            "minute": None,
        },
        "snapshot": {},
        "intraday": {
            "label": "实时分时暂未加载",
            "score": display_context.get("intraday_strength_score", display_context.get("intraday_execution_score", 0.0)),
        },
        "signals": {
            "stage": {
                "label": str(display_context.get("launch_phase_display") or "等待确认"),
                "summary": str(display_context.get("launch_reason_text") or display_context.get("detail_note")),
                "intraday_expectation": str(display_context.get("intraday_sector_note") or "等待分时与板块联动数据补齐。"),
            },
            "sector": {
                "sector_score": display_context.get("sector_strength_score", display_context.get("sector_score", 0.0)),
                "sector_summary": display_context.get("intraday_sector_display") or "等待板块资金增强。",
            },
            "fund": {"fund_score": display_context.get("fund_score", 0.0), "summary": "等待主力资金增强。"},
            "news": {"news_score": display_context.get("news_score", 0.0), "summary": "等待消息面增强。"},
            "quant": {
                "score": display_context.get("quant_score", 0.0),
                "primary_signal": display_context.get("launch_signal_display") or "等待确认",
                "summary": display_context.get("launch_reason_text") or "缓存模型评分已展示。",
            },
            "model": {
                "signal_label": display_context.get("action_label") or "观察",
                "backtest_summary": "使用快榜缓存结果，后台回测完成后自动增强。",
                "strategy_score": display_context.get("rank_score", 0.0),
                "agreement_score": display_context.get("intraday_sector_sync_score", 0.0),
                "quality_label": "safe_cached",
                "risk_label": display_context.get("risk_level_display") or "--",
            },
            "backtest": {
                "status_label": "cached",
                "summary": "详情实时源不可用时启用安全缓存降级。",
                "target_precision": 0.9,
                "selection_summary": "",
                "achieved_precision": 0.0,
                "trade_count": 0,
                "latest_signal_active": False,
            },
            "tomorrow_plan": {
                "setup_label": display_context.get("tomorrow_setup") or display_context.get("launch_reason_text") or "等待确认",
                "bias": display_context.get("action_label") or "观察",
                "buy_point": display_context.get("tomorrow_buy_point") or "等待分时承接确认。",
                "sell_point": display_context.get("tomorrow_sell_point") or "跌破关键支撑或量能转弱先处理。",
                "confidence": display_context.get("action_confidence", display_context.get("probability_confidence", 0.0)),
            },
        },
        "fund_flow": [],
        "news": [],
    }


def load_quick_board_payload(params: dict[str, Any]) -> dict[str, Any]:
    board, quick_meta = load_latest_close_quick_board(
        horizon_days=params["horizon_days"],
        positive_return=params["positive_return"],
        ranking_by=params["ranking_by"],
        board_size=params["board_size"],
    )
    if not board.empty:
        board = _build_display_board(
            board,
            params["board_size"],
            params["ranking_by"],
            str(board.attrs.get("data_mode", "latest_close_quick_board")),
            loading=True,
        )
    elif params["custom_watchlist"]:
        # A custom watchlist is bounded by the user input, so this path does not
        # trigger a full-market rebuild.
        board = _build_focus_board(
            board_size=params["board_size"],
            custom_watchlist=params["custom_watchlist"],
            horizon_days=params["horizon_days"],
            positive_return=params["positive_return"],
            ranking_by=params["ranking_by"],
        )
        quick_meta = {}
    else:
        cached_board, cache_meta = _read_market_rankings_cache(
            params["horizon_days"],
            params["positive_return"],
            allow_stale=True,
        )
        if cached_board is not None and not cached_board.empty:
            cached_board = cached_board.copy()
            for key, value in cache_meta.items():
                cached_board.attrs[key] = value
            cached_board.attrs["data_mode"] = str(cache_meta.get("data_mode", "history"))
            board = _build_display_board(
                cached_board,
                params["board_size"],
                params["ranking_by"],
                str(cached_board.attrs.get("data_mode", "history")),
                loading=True,
            )
            quick_meta = {"quick_source": "cached_market_ranking", **cache_meta}
        else:
            snapshot_board, snapshot_meta = load_latest_snapshot_board(
                horizon_days=params["horizon_days"],
                positive_return=params["positive_return"],
                ranking_by=params["ranking_by"],
                board_size=params["board_size"],
            )
            if not snapshot_board.empty:
                snapshot_board = snapshot_board.copy()
                snapshot_board.attrs["data_mode"] = "snapshot_history"
                snapshot_board.attrs["market_data_date"] = str(
                    snapshot_meta.get("board_date") or snapshot_meta.get("latest_market_data_date") or ""
                )
                snapshot_board.attrs["latest_market_data_date"] = str(
                    snapshot_meta.get("latest_market_data_date") or snapshot_board.attrs["market_data_date"]
                )
                snapshot_board.attrs["computed_at"] = str(snapshot_meta.get("captured_at") or "")
                snapshot_board.attrs["cache_stale"] = bool(snapshot_meta.get("cache_stale", False))
                board = _build_display_board(
                    snapshot_board,
                    params["board_size"],
                    params["ranking_by"],
                    "snapshot_history",
                    loading=True,
                )
                quick_meta = {"quick_source": "snapshot_history", **snapshot_meta}
    review_summary = load_latest_review_summary(
        horizon_days=params["horizon_days"],
        positive_return=params["positive_return"],
        ranking_by=params["ranking_by"],
        board_size=params["board_size"],
    )
    review_summary_clean = _clean_value(review_summary or {})
    daily_lightweight_model = serialize_daily_lightweight_model(params)
    rolling_review_task = start_rolling_review_profile_task(params, board)
    board_payload = serialize_board(board, extra_meta=_clean_value(quick_meta), params=params, source="quick_board")
    review_health = build_review_health(review_summary_clean, {}, params, daily_lightweight_model)
    model_snapshot = build_model_snapshot(params, board_payload["freshness"], review_health, source="quick_board")
    return {
        "params": _clean_value(params),
        "latest_market_data_date": _latest_market_close_date(),
        "freshness": board_payload["freshness"],
        "model_snapshot": model_snapshot,
        "board": board_payload,
        "review_summary": review_summary_clean,
        "review_health": review_health,
        "daily_lightweight_model": daily_lightweight_model,
        "rolling_review_task": _clean_value(rolling_review_task),
        "universe_size": int(len(load_a_share_universe())),
    }


def start_rebuild_ranking_task(params: dict[str, Any]) -> dict[str, Any]:
    task_id = f'api-market-refresh::{params["horizon_days"]}::{params["positive_return"]:.4f}'
    future = API_TASK_FUTURES.get(task_id)
    if future is None or future.done():
        DEFAULT_TASK_REGISTRY.record_submitted(task_id, task_type="market_ranking_rebuild", params=_clean_value(params))
        future = API_TASK_EXECUTOR.submit(
            _refresh_market_rankings_cache_task,
            task_id,
            params["horizon_days"],
            params["positive_return"],
        )
        API_TASK_FUTURES[task_id] = future
    return get_task_status(task_id)


def get_task_status(task_id: str) -> dict[str, Any]:
    future = API_TASK_FUTURES.get(task_id)
    progress = _get_async_task_progress(task_id)
    if future is None:
        DEFAULT_TASK_REGISTRY.record_status(task_id, status="missing")
        return {
            "task_id": task_id,
            "status": "missing",
            "progress": _clean_value(progress),
            "result": None,
            "error": "",
        }
    if not future.done():
        DEFAULT_TASK_REGISTRY.record_status(task_id, status="running")
        return {
            "task_id": task_id,
            "status": "running",
            "progress": _clean_value(progress),
            "result": None,
            "error": "",
        }
    try:
        result = future.result()
    except Exception as exc:  # pragma: no cover
        DEFAULT_TASK_REGISTRY.record_status(task_id, status="failed", error=str(exc))
        return {
            "task_id": task_id,
            "status": "failed",
            "progress": _clean_value(progress),
            "result": None,
            "error": str(exc),
        }
    DEFAULT_TASK_REGISTRY.record_status(task_id, status="completed")
    return {
        "task_id": task_id,
        "status": "completed",
        "progress": _clean_value(progress),
        "result": _clean_value(result or {}),
        "error": "",
    }


def _market_backtest_task_id(params: dict[str, Any]) -> str:
    return "|".join(
        [
            "api-market-backtest",
            str(params.get("date_from")),
            str(params.get("date_to")),
            str(params.get("horizon_days")),
            f'{float(params.get("positive_return", 0.0) or 0.0):.4f}',
            str(params.get("strategy_mode") or "all"),
            str(params.get("top_k") or 50),
        ]
    )


def run_market_backtest(**kwargs: Any) -> dict[str, Any]:
    return DEFAULT_BACKTEST_SERVICE.run_market_backtest(**kwargs)


def load_latest_market_backtest(*, result_limit: int = 50) -> dict[str, Any]:
    return DEFAULT_BACKTEST_SERVICE.load_latest_market_backtest(result_limit=result_limit)


def start_market_backtest_task(
    *,
    date_from: str,
    date_to: str,
    horizon_days: int = 3,
    positive_return_pct: float = 10.0,
    strategy_mode: str = "all",
    top_k: int = 50,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    params = {
        "date_from": str(date_from),
        "date_to": str(date_to),
        "horizon_days": int(horizon_days),
        "positive_return": float(positive_return_pct) / 100.0,
        "strategy_mode": _normalize_strategy_mode(strategy_mode),
        "top_k": int(max(1, min(int(top_k), 300))),
        "force_rebuild": bool(force_rebuild),
    }
    task_id = _market_backtest_task_id(params)
    future = API_TASK_FUTURES.get(task_id)
    if future is None or future.done():
        DEFAULT_TASK_REGISTRY.record_submitted(task_id, task_type="market_backtest", params=_clean_value(params))
        future = API_TASK_EXECUTOR.submit(
            run_market_backtest,
            date_from=params["date_from"],
            date_to=params["date_to"],
            horizon_days=params["horizon_days"],
            positive_return=params["positive_return"],
            strategy_mode=params["strategy_mode"],
            top_k=params["top_k"],
            force_rebuild=params["force_rebuild"],
            max_workers=12,
            fast_strategy_backtest=True,
        )
        API_TASK_FUTURES[task_id] = future
    return get_task_status(task_id)


def load_market_backtest_payload(result_limit: int = 50) -> dict[str, Any]:
    payload = load_latest_market_backtest(result_limit=result_limit)
    if not payload:
        return {
            "status": "missing",
            "summary": {},
            "results": [],
            "summary_path": "",
            "results_path": "",
            "portfolio_nav_path": "",
            "portfolio_trades_path": "",
            "portfolio_daily_nav": [],
            "portfolio_trades": [],
        }
    return {
        "status": "ready",
        "summary": _clean_value(payload.get("summary", {})),
        "summary_path": str(payload.get("summary_path") or ""),
        "results_path": str(payload.get("results_path") or ""),
        "results": dataframe_records(payload.get("results", pd.DataFrame()), limit=result_limit),
        "portfolio_nav_path": str(payload.get("portfolio_nav_path") or ""),
        "portfolio_trades_path": str(payload.get("portfolio_trades_path") or ""),
        "portfolio_daily_nav": dataframe_records(payload.get("portfolio_daily_nav", pd.DataFrame()), limit=result_limit),
        "portfolio_trades": dataframe_records(payload.get("portfolio_trades", pd.DataFrame()), limit=result_limit),
    }


def load_enhanced_board_payload(params: dict[str, Any]) -> dict[str, Any]:
    # Keep the page-load enhanced path crash-safe. Some live AkShare/JS-engine
    # calls can terminate the Python process on Windows, so the API response
    # must be cache-first and non-destructive. Heavy rebuilds belong in the
    # explicit background task, not in a browser refresh request.
    quick_payload = load_quick_board_payload(params)
    review_bundle = load_latest_review_bundle(
        horizon_days=params["horizon_days"],
        positive_return=params["positive_return"],
        ranking_by=params["ranking_by"],
        board_size=params["board_size"],
    )
    review_summary = review_bundle.get("summary")
    review_details = review_bundle.get("details", pd.DataFrame())
    review_snapshot_board = review_bundle.get("snapshot_board", pd.DataFrame())
    review_snapshot_meta = review_bundle.get("snapshot_meta", {})
    review_summary_clean = _clean_value(review_summary or {})
    review_panels = load_cached_review_panels(params)
    daily_lightweight_model = serialize_daily_lightweight_model(params)
    board_payload = quick_payload.get("board", serialize_board(pd.DataFrame(), params=params, source="enhanced_board"))
    board_payload = dict(board_payload)
    board_meta = dict(board_payload.get("meta", {}))
    board_meta["quick_source"] = board_meta.get("quick_source", "safe_cached_enhanced")
    board_meta["enhanced_mode"] = "safe_cached"
    board_payload["meta"] = board_meta
    review_health = build_review_health(review_summary_clean, review_panels, params, daily_lightweight_model)
    model_snapshot = build_model_snapshot(params, board_payload["freshness"], review_health, source="enhanced_board")
    return {
        "params": _clean_value(params),
        "latest_market_data_date": _latest_market_close_date(),
        "freshness": board_payload["freshness"],
        "model_snapshot": model_snapshot,
        "board": board_payload,
        "market_context": serialize_market_context({}),
        "review_summary": review_summary_clean,
        "review_health": review_health,
        "review_panels": review_panels,
        "daily_lightweight_model": daily_lightweight_model,
        "review_details": dataframe_records(review_details, limit=24),
        "review_snapshot_meta": _clean_value(review_snapshot_meta),
        "review_snapshot_board": dataframe_records(review_snapshot_board, limit=24),
    }


def _live_symbol_detail_enabled() -> bool:
    explicit_disable = os.environ.get("A_SHARE_DISABLE_LIVE_DETAIL", "").strip().lower()
    if explicit_disable in {"1", "true", "yes", "on"}:
        return False
    explicit_enable = os.environ.get("A_SHARE_ENABLE_LIVE_DETAIL")
    if explicit_enable is None:
        return True
    return explicit_enable.strip().lower() not in {"0", "false", "no", "off"}


def load_symbol_detail_payload(symbol: str, params: dict[str, Any]) -> dict[str, Any]:
    resolved_symbol = resolve_symbol(symbol)
    if not _live_symbol_detail_enabled():
        return _build_safe_symbol_detail_payload(resolved_symbol, params)
    try:
        detail = _build_symbol_detail(
            resolved_symbol,
            params["horizon_days"],
            params["positive_return"],
        )
    except Exception as exc:
        return _build_safe_symbol_detail_payload(resolved_symbol, params, reason=str(exc))
    display_context = _detail_display_context(detail)
    calibrated_probability = float(display_context.get("probability_up", 0.0) or 0.0)
    raw_probability = float(display_context.get("raw_probability_up", calibrated_probability) or calibrated_probability)
    display_context["raw_probability_up"] = max(0.0, min(raw_probability, 100.0))
    display_context["calibrated_probability_up"] = max(0.0, min(calibrated_probability, 100.0))
    display_context["rank_score"] = max(
        0.0,
        min(
            float(
                display_context.get(
                    "rank_score",
                    display_context.get("action_score", display_context.get("enhanced_attention_score", 0.0)),
                )
                or 0.0
            ),
            100.0,
        ),
    )
    display_context["p_hit"] = display_context["calibrated_probability_up"]
    display_context["expected_return_pct"] = float(
        display_context.get("expected_return_pct", display_context.get("predicted_upside_pct", 0.0)) or 0.0
    )
    display_context["drawdown_risk_pct"] = float(display_context.get("drawdown_risk_pct", 0.0) or 0.0)
    if display_context["drawdown_risk_pct"] <= 0:
        display_context["drawdown_risk_pct"] = max(
            0.0,
            min(
                (100.0 - display_context["p_hit"]) * 0.08
                + max(0.0, 10.0 - display_context["expected_return_pct"]) * 0.22,
                35.0,
            ),
        )
    display_context["market_state_label"] = str(
        display_context.get("market_state_label")
        or detail.get("latest_features", {}).get("market_regime_label")
        or "unknown"
    )
    intraday_context = detail.get("intraday", {}) or {}
    sector_context = detail.get("sector_signal", {}) or {}
    if "intraday_score" not in display_context and "intraday_execution_score" not in display_context:
        display_context["intraday_score"] = float(intraday_context.get("score", 0.5) or 0.5)
    if "sector_strength_score" not in display_context and "sector_score" not in display_context:
        display_context["sector_score"] = float(sector_context.get("sector_score", 50.0) or 50.0)
    contract_context = apply_probability_contract(pd.DataFrame([display_context]))
    if not contract_context.empty:
        contract_row = contract_context.iloc[0].to_dict()
        for field in (
            "raw_probability_up",
            "calibrated_probability_up",
            "probability_up",
            "calibration_method",
            "probability_confidence",
            "probability_prior_up",
            "probability_band_low",
            "probability_band_high",
            "p_hit",
            "expected_return_pct",
            "drawdown_risk_pct",
            "rank_score",
            "market_state_display",
            "intraday_strength_score",
            "sector_strength_score",
            "relative_intraday_alpha",
            "sector_follow_score",
            "intraday_sector_sync_score",
            "intraday_sector_state",
            "intraday_sector_display",
            "intraday_sector_note",
            "board_resonance_strength",
            "long_setup_quality",
            "crowding_risk",
            "crowding_risk_label",
            "launch_window_score",
            "launch_specialist_score",
            "launch_signal_score",
            "launch_signal_label",
            "launch_signal_display",
            "launch_phase_label",
            "launch_phase_display",
            "launch_reason_text",
            "risk_level_label",
            "risk_level_display",
            "suggested_position_pct",
            "stop_loss_pct",
            "take_profit_pct",
            "risk_control_note",
        ):
            display_context[field] = contract_row.get(field, display_context.get(field))
    detail_meta = {
        "market_data_date": detail.get("analysis_date") or detail.get("latest_market_data_date"),
        "latest_market_data_date": detail.get("latest_market_data_date") or detail.get("analysis_date"),
        "computed_at": display_context.get("computed_at"),
        "cache_stale": bool(display_context.get("cache_stale", False)),
        "model_schema_version": display_context.get("model_schema_version") or API_MODEL_CONTRACT_VERSION,
        "model_source_label": display_context.get("model_source_label") or "single-symbol-api",
    }
    freshness = build_freshness_contract(detail_meta, params, source="symbol_detail")
    model_snapshot = build_model_snapshot(params, freshness, None, source="symbol_detail")
    display_context["freshness"] = freshness
    display_context["model_snapshot"] = model_snapshot
    daily_chart = json.loads(make_daily_chart(detail["daily"]).to_json())
    minute_chart = json.loads(make_minute_chart(detail["minute"]).to_json())
    profile = detail.get("profile", {}) or {}
    rule_context = detail.get("rule_context")
    quant_signal = detail.get("quant_signal")
    model = detail.get("model")
    backtest = detail.get("backtest")
    tomorrow_plan = detail.get("tomorrow_plan")
    stage = detail.get("stage")
    return {
        "symbol": resolved_symbol,
        "params": _clean_value(params),
        "freshness": freshness,
        "model_snapshot": model_snapshot,
        "display_context": _clean_value(display_context),
        "hero": {
            "name": str(profile.get("股票简称", resolved_symbol)),
            "industry": str(profile.get("行业", "未知")),
            "board_label": str(getattr(rule_context, "board_label", "主板")),
            "price_limit_label": str(getattr(rule_context, "price_limit_label", "10%")),
            "analysis_date": _clean_value(detail.get("analysis_date")),
            "latest_market_data_date": _clean_value(detail.get("latest_market_data_date")),
            "intraday_label": _clean_value(detail.get("intraday", {}).get("label")),
            "stage_label": _clean_value(getattr(stage, "label", "")),
        },
        "charts": {
            "daily": daily_chart,
            "minute": minute_chart,
        },
        "snapshot": _clean_value(detail.get("snapshot", {})),
        "intraday": _clean_value(detail.get("intraday", {})),
        "signals": {
            "stage": {
                "label": str(getattr(stage, "label", "")),
                "summary": str(getattr(stage, "summary", "")),
                "intraday_expectation": str(getattr(stage, "intraday_expectation", "")),
            },
            "sector": _clean_value(detail.get("sector_signal", {})),
            "fund": _clean_value(detail.get("fund_signal", {})),
            "news": _clean_value(detail.get("news_signal", {})),
            "quant": {
                "score": float(getattr(quant_signal, "total_score", 0.0) or 0.0),
                "primary_signal": str(getattr(quant_signal, "primary_signal", "")),
                "summary": str(getattr(quant_signal, "summary", "")),
            },
            "model": {
                "signal_label": str(getattr(model, "signal_label", "")),
                "backtest_summary": str(getattr(model, "backtest_summary", "")),
                "strategy_score": float(getattr(model, "strategy_score", 0.0) or 0.0),
                "agreement_score": float(getattr(model, "agreement_score", 0.0) or 0.0),
                "quality_label": str(getattr(model, "quality_label", "")),
                "risk_label": str(getattr(model, "risk_label", "")),
            },
            "backtest": {
                "status_label": str(getattr(backtest, "status_label", "")),
                "summary": str(getattr(backtest, "summary", "")),
                "target_precision": float(getattr(backtest, "target_precision", 0.0) or 0.0),
                "selection_summary": str(getattr(backtest, "selection_summary", "")),
                "achieved_precision": float(getattr(backtest, "achieved_precision", 0.0) or 0.0),
                "trade_count": int(getattr(backtest, "trade_count", 0) or 0),
                "latest_signal_active": bool(getattr(backtest, "latest_signal_active", False)),
            },
            "tomorrow_plan": {
                "setup_label": str(getattr(tomorrow_plan, "setup_label", "")),
                "bias": str(getattr(tomorrow_plan, "bias", "")),
                "buy_point": str(getattr(tomorrow_plan, "buy_point", "")),
                "sell_point": str(getattr(tomorrow_plan, "sell_point", "")),
                "confidence": float(getattr(tomorrow_plan, "confidence", 0.0) or 0.0),
            },
        },
        "fund_flow": dataframe_records(detail.get("fund_flow_df", pd.DataFrame()), limit=20),
        "news": dataframe_records(detail.get("news_df", pd.DataFrame()), limit=20),
    }


def search_symbols(query: str, *, limit: int = 20) -> list[dict[str, Any]]:
    universe = load_a_share_universe()
    results = search_a_share_universe(universe, query, limit=limit)
    return dataframe_records(results[["symbol", "name"]], limit=limit) if not results.empty else []


def load_news_impact_payload(
    symbol: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    news_limit: int = 120,
    horizons: tuple[int, ...] = (1, 3, 5),
    include_disclosures: bool = True,
) -> dict[str, Any]:
    result = analyze_symbol_news_impact(
        symbol,
        start_date=start_date,
        end_date=end_date,
        news_limit=news_limit,
        horizons=horizons,
        include_disclosures=include_disclosures,
    )
    return {
        "symbol": result["symbol"],
        "params": {
            "start_date": start_date,
            "end_date": end_date,
            "news_limit": news_limit,
            "horizons": list(result["horizons"]),
            "include_disclosures": include_disclosures,
        },
        "event_count": int(result["event_count"]),
        "impact_sample_count": int(result["impact_sample_count"]),
        "latest_signal": _clean_value(result["latest_signal"]),
        "category_summary": dataframe_records(result["category_summary"], limit=50),
        "event_impacts": dataframe_records(result["event_impacts"], limit=100),
        "events": dataframe_records(result["events"], limit=100),
    }


def frontend_dist_available() -> bool:
    return (FRONTEND_DIST_DIR / "index.html").exists()
