from __future__ import annotations

import hashlib
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import fetch_daily_history
from .modeling import (
    GLOBAL_MODEL_TEST_END,
    GLOBAL_MODEL_TEST_START,
    GLOBAL_MODEL_TRAIN_END,
    GLOBAL_MODEL_TRAIN_START,
    _load_partial_market_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / ".cache"
DAILY_REVIEW_CACHE_VERSION = 10
LEGACY_DAILY_REVIEW_CACHE_VERSIONS: tuple[int, ...] = (9, 8, 7, 6)
MAX_PROFILE_REVIEW_DAYS = 20
DEFAULT_ROLLING_REVIEW_DAYS = 20
DEFAULT_PROFILE_WEIGHTS = {
    "attention_score": 0.33,
    "probability_up": 0.23,
    "enhanced_attention_score": 0.13,
    "quant_score": 0.07,
    "launch_score": 0.06,
    "market_resonance_score": 0.05,
    "intraday_sector_sync_score": 0.06,
    "launch_specialist_score": 0.05,
    "launch_regime_fit_score": 0.03,
    "launch_window_score": 0.05,
}
MARKET_REPLAY_CACHE_VERSION = 1
DAILY_LIGHTWEIGHT_MODEL_VERSION = 1
MAX_MARKET_REPLAY_TRADING_DAYS = 180
REPLAY_PROFILE_MIN_REVIEW_DAYS = 2
REPLAY_PROFILE_MIN_REVIEW_STOCKS = 12
MARKET_REPLAY_MIN_DAYS = 20
MARKET_REPLAY_MIN_ROWS = 1_500
REPLAY_PROBABILITY_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.0, 20.0, "0-20"),
    (20.0, 40.0, "20-40"),
    (40.0, 60.0, "40-60"),
    (60.0, 80.0, "60-80"),
    (80.0, 90.0, "80-90"),
    (90.0, 95.0, "90-95"),
    (95.0, 100.01, "95-100"),
)
REPLAY_QUANT_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.0, 50.0, "0-50"),
    (50.0, 60.0, "50-60"),
    (60.0, 70.0, "60-70"),
    (70.0, 80.0, "70-80"),
    (80.0, 100.01, "80-100"),
)
REPLAY_LAUNCH_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.0, 50.0, "0-50"),
    (50.0, 65.0, "50-65"),
    (65.0, 80.0, "65-80"),
    (80.0, 100.01, "80-100"),
)
REPLAY_RESONANCE_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.0, 45.0, "0-45"),
    (45.0, 60.0, "45-60"),
    (60.0, 75.0, "60-75"),
    (75.0, 100.01, "75-100"),
)
REPLAY_LAUNCH_WINDOW_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.0, 45.0, "0-45"),
    (45.0, 60.0, "45-60"),
    (60.0, 75.0, "60-75"),
    (75.0, 100.01, "75-100"),
)
MARKET_STATE_LABELS: tuple[str, ...] = ("trend", "rebound", "rotation", "defense")
MARKET_STAGE_PROXY_LABELS: tuple[str, ...] = (
    "trend_drive",
    "main_rise_start",
    "breakout_confirm",
    "range_watch",
    "distribution_risk",
)
MARKET_STATE_DISPLAY_LABELS: dict[str, str] = {
    "trend": "趋势扩散",
    "rebound": "修复反弹",
    "rotation": "轮动震荡",
    "defense": "防守退潮",
    "unknown": "状态待定",
}
MARKET_STAGE_PROXY_DISPLAY_LABELS: dict[str, str] = {
    "trend_drive": "趋势驱动",
    "main_rise_start": "主升初启",
    "breakout_confirm": "突破确认",
    "range_watch": "区间观察",
    "distribution_risk": "派发风险",
    "unknown": "阶段待定",
}
STRATEGY_DISPLAY_LABELS: dict[str, str] = {
    "策略1": "策略1·趋势中继",
    "策略2": "策略2·突破共振",
    "dynamic_fallback": "动态补位",
    "fallback_watchlist": "兜底股票池",
    "": "通用模型",
}


@dataclass(slots=True)
class FocusBoardReviewSummary:
    board_date: str
    review_date: str
    review_count: int
    avg_return_pct: float
    win_rate_pct: float
    top_probability_return_pct: float
    top_attention_return_pct: float
    target_hit_rate_pct: float
    direction_hit_rate_pct: float
    calibration_gap_pct: float
    direction_brier_score: float
    avg_target_progress_pct: float
    optimization_note: str


def _review_cache_dir() -> Path:
    path = CACHE_DIR / "daily_focus_board_reviews"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _supported_review_cache_versions() -> tuple[int, ...]:
    versions = [DAILY_REVIEW_CACHE_VERSION]
    for version in LEGACY_DAILY_REVIEW_CACHE_VERSIONS:
        if int(version) not in versions:
            versions.append(int(version))
    return tuple(versions)


def _safe_text_token(value: str) -> str:
    ascii_token = re.sub(r"[^0-9A-Za-z_-]+", "_", str(value))
    ascii_token = ascii_token.strip("_")
    if ascii_token:
        return ascii_token
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:10]
    return f"u{digest}"


def _snapshot_cache_path(
    horizon_days: int,
    positive_return: float,
    ranking_by: str,
    board_size: int,
    board_date: str,
) -> Path:
    return _review_cache_dir() / (
        f"snapshot_v{DAILY_REVIEW_CACHE_VERSION}_"
        f"h{horizon_days}_r{int(positive_return * 10000)}_"
        f"b{board_size}_{_safe_text_token(ranking_by)}_{board_date.replace('-', '')}.pkl"
    )


def _review_cache_path(
    horizon_days: int,
    positive_return: float,
    ranking_by: str,
    board_size: int,
    board_date: str,
    review_date: str,
) -> Path:
    return _review_cache_dir() / (
        f"review_v{DAILY_REVIEW_CACHE_VERSION}_"
        f"h{horizon_days}_r{int(positive_return * 10000)}_"
        f"b{board_size}_{_safe_text_token(ranking_by)}_{board_date.replace('-', '')}_{review_date.replace('-', '')}.pkl"
    )


def _profile_cache_path(
    horizon_days: int,
    positive_return: float,
    ranking_by: str,
    board_size: int,
) -> Path:
    return _review_cache_dir() / (
        f"profile_v{DAILY_REVIEW_CACHE_VERSION}_"
        f"h{horizon_days}_r{int(positive_return * 10000)}_"
        f"b{board_size}_{_safe_text_token(ranking_by)}.pkl"
    )


def _daily_lightweight_model_cache_path(
    horizon_days: int,
    positive_return: float,
    ranking_by: str,
    board_size: int,
) -> Path:
    return _review_cache_dir() / (
        f"daily_lightweight_model_v{DAILY_LIGHTWEIGHT_MODEL_VERSION}_"
        f"h{horizon_days}_r{int(positive_return * 10000)}_"
        f"b{board_size}_{_safe_text_token(ranking_by)}.pkl"
    )


def _market_replay_profile_cache_path(
    horizon_days: int,
    positive_return: float,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> Path:
    return _review_cache_dir() / (
        f"market_replay_profile_v{MARKET_REPLAY_CACHE_VERSION}_"
        f"h{horizon_days}_r{int(positive_return * 10000)}_"
        f"{train_start.replace('-', '')}_{train_end.replace('-', '')}_"
        f"{test_start.replace('-', '')}_{test_end.replace('-', '')}.pkl"
    )


def _default_profile(summary: str) -> dict[str, object]:
    return {
        "weights": dict(DEFAULT_PROFILE_WEIGHTS),
        "generated_at": "",
        "review_days": 0,
        "review_stocks": 0,
        "rolling_review_days": int(DEFAULT_ROLLING_REVIEW_DAYS),
        "rolling_review_mode": "recent_n_trading_days",
        "avg_return_pct": 0.0,
        "win_rate_pct": 0.0,
        "factor_edges": {key: 0.0 for key in DEFAULT_PROFILE_WEIGHTS},
        "stage_edges": {},
        "stage_supports": {},
        "stage_stats": {},
        "strategy_edges": {},
        "strategy_supports": {},
        "strategy_stats": {},
        "best_strategy": "",
        "weakest_strategy": "",
        "precision_segment_edges": {},
        "precision_segment_supports": {},
        "probability_bucket_edges": {},
        "probability_bucket_supports": {},
        "probability_bucket_stats": {},
        "quant_bucket_edges": {},
        "quant_bucket_supports": {},
        "quant_bucket_stats": {},
        "launch_bucket_edges": {},
        "launch_bucket_supports": {},
        "launch_bucket_stats": {},
        "resonance_bucket_edges": {},
        "resonance_bucket_supports": {},
        "resonance_bucket_stats": {},
        "launch_window_bucket_edges": {},
        "launch_window_bucket_supports": {},
        "launch_window_bucket_stats": {},
        "launch_window_status_edges": {},
        "launch_window_status_supports": {},
        "launch_window_status_stats": {},
        "market_replay_days": 0,
        "market_replay_rows": 0,
        "market_replay_symbols": 0,
        "market_replay_start_date": "",
        "market_replay_end_date": "",
        "market_state_edges": {},
        "market_state_supports": {},
        "market_state_stats": {},
        "market_stage_proxy_edges": {},
        "market_stage_proxy_supports": {},
        "market_stage_proxy_stats": {},
        "market_replay_summary": "",
        "calibration_scope": "rank_score_and_risk_overlay_only",
        "model_parameter_update_allowed": False,
        "allowed_calibration_targets": ["ranking_weight_micro_adjustment", "strategy_fit_score", "high_risk_pattern_suppression"],
        "profile_summary": summary,
    }


def _normalize_short_profile(profile: dict[str, object] | None) -> dict[str, object]:
    source = dict(profile or {})
    summary = str(source.get("profile_summary") or "鑷€傚簲浼樺寲鏆傛椂浣跨敤榛樿閰嶇疆銆?")
    normalized = _default_profile(summary)
    normalized.update(source)
    merged_weights = dict(DEFAULT_PROFILE_WEIGHTS)
    if isinstance(source.get("weights"), dict):
        merged_weights.update({key: float(value) for key, value in source["weights"].items() if key in merged_weights})
    normalized["weights"] = merged_weights
    for key in (
        "factor_edges",
        "stage_edges",
        "stage_supports",
        "stage_stats",
        "strategy_edges",
        "strategy_supports",
        "strategy_stats",
        "precision_segment_edges",
        "precision_segment_supports",
        "probability_bucket_edges",
        "probability_bucket_supports",
        "probability_bucket_stats",
        "quant_bucket_edges",
        "quant_bucket_supports",
        "quant_bucket_stats",
        "launch_bucket_edges",
        "launch_bucket_supports",
        "launch_bucket_stats",
        "resonance_bucket_edges",
        "resonance_bucket_supports",
        "resonance_bucket_stats",
        "launch_window_bucket_edges",
        "launch_window_bucket_supports",
        "launch_window_bucket_stats",
        "launch_window_status_edges",
        "launch_window_status_supports",
        "launch_window_status_stats",
    ):
        base_map = dict(normalized.get(key, {}) or {})
        if isinstance(source.get(key), dict):
            base_map.update(source.get(key, {}) or {})
        normalized[key] = base_map
    return normalized


def _clip(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(float(value), upper)))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _weighted_average(values: pd.Series, weights: pd.Series | None = None, default: float = 0.0) -> float:
    series = pd.to_numeric(values, errors="coerce").astype(float)
    if weights is None:
        valid = series.dropna()
        return float(valid.mean()) if not valid.empty else float(default)
    weight_series = pd.to_numeric(weights, errors="coerce").astype(float).reindex(series.index).fillna(0.0)
    valid_mask = series.notna() & weight_series.gt(0)
    if not valid_mask.any():
        return float(default)
    return float(np.average(series.loc[valid_mask], weights=weight_series.loc[valid_mask]))


def _weighted_corr(x: pd.Series, y: pd.Series, weights: pd.Series | None = None) -> float:
    x_series = pd.to_numeric(x, errors="coerce").astype(float)
    y_series = pd.to_numeric(y, errors="coerce").astype(float)
    if weights is None:
        corr = x_series.corr(y_series)
        return 0.0 if pd.isna(corr) else float(corr)
    weight_series = pd.to_numeric(weights, errors="coerce").astype(float).reindex(x_series.index).fillna(0.0)
    valid_mask = x_series.notna() & y_series.notna() & weight_series.gt(0)
    if not valid_mask.any():
        return 0.0
    x_valid = x_series.loc[valid_mask]
    y_valid = y_series.loc[valid_mask]
    w_valid = weight_series.loc[valid_mask]
    x_mean = float(np.average(x_valid, weights=w_valid))
    y_mean = float(np.average(y_valid, weights=w_valid))
    cov = float(np.average((x_valid - x_mean) * (y_valid - y_mean), weights=w_valid))
    var_x = float(np.average((x_valid - x_mean) ** 2, weights=w_valid))
    var_y = float(np.average((y_valid - y_mean) ** 2, weights=w_valid))
    if var_x <= 0 or var_y <= 0:
        return 0.0
    return float(cov / np.sqrt(var_x * var_y))


def _support_scale(count: int, floor: float = 6.0) -> float:
    return float(count) / (float(count) + float(floor)) if count > 0 else 0.0


def _normalize_rolling_review_days(value: int | None = None) -> int:
    if value is None:
        return int(DEFAULT_ROLLING_REVIEW_DAYS)
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = int(DEFAULT_ROLLING_REVIEW_DAYS)
    return int(max(1, min(numeric, MAX_PROFILE_REVIEW_DAYS)))


def _bucket_label(value: float, buckets: tuple[tuple[float, float, str], ...], default: str) -> str:
    numeric_value = _safe_float(value)
    for lower, upper, label in buckets:
        if numeric_value >= lower and numeric_value < upper:
            return label
    return default


def _probability_bucket_label(probability_up: float) -> str:
    return _bucket_label(probability_up, REPLAY_PROBABILITY_BUCKETS, "unknown")


def _quant_bucket_label(quant_score: float) -> str:
    return _bucket_label(quant_score, REPLAY_QUANT_BUCKETS, "unknown")


def _launch_bucket_label(launch_score: float) -> str:
    return _bucket_label(launch_score, REPLAY_LAUNCH_BUCKETS, "unknown")


def _resonance_bucket_label(market_resonance_score: float) -> str:
    return _bucket_label(market_resonance_score, REPLAY_RESONANCE_BUCKETS, "unknown")


def _launch_window_bucket_label(launch_window_score: float) -> str:
    return _bucket_label(launch_window_score, REPLAY_LAUNCH_WINDOW_BUCKETS, "unknown")


def _launch_window_status_label(launch_window_status: str) -> str:
    label = str(launch_window_status or "").strip()
    if not label:
        return "unknown"
    if label in {"黄金启动窗", "启动观察窗", "强势延续", "非启动窗", "高位风险窗"}:
        return label
    return "unknown"


def _market_state_display_label(label: str) -> str:
    key = str(label or "").strip()
    return MARKET_STATE_DISPLAY_LABELS.get(key, key or MARKET_STATE_DISPLAY_LABELS["unknown"])


def _market_stage_proxy_display_label(label: str) -> str:
    key = str(label or "").strip()
    return MARKET_STAGE_PROXY_DISPLAY_LABELS.get(key, key or MARKET_STAGE_PROXY_DISPLAY_LABELS["unknown"])


def _strategy_display_label(strategy_code: str, strategy_label: str = "") -> str:
    label = str(strategy_label or "").strip()
    combined = f"{strategy_code or ''} {label}".lower()
    if "策略3" in combined or "strategy3" in combined:
        return STRATEGY_DISPLAY_LABELS[""]
    if label:
        return label
    code = str(strategy_code or "").strip()
    return STRATEGY_DISPLAY_LABELS.get(code, code or STRATEGY_DISPLAY_LABELS[""])


def _precision_segment_label(precision_gate_label: str) -> str:
    label = str(precision_gate_label or "").strip()
    if not label:
        return "unknown"
    if "90%精度放行" in label:
        return "precision_active"
    if "高精度观察" in label:
        return "precision_watch"
    if "历史达标" in label:
        return "precision_history"
    if "代理模型未提供90%精度认证" in label:
        return "proxy_unverified"
    if "未达90%精度门槛" in label:
        return "precision_unreached"
    return "other"


def _performance_edge(
    *,
    avg_return: float,
    win_rate: float,
    hit_rate: float,
    overall_avg_return: float,
    overall_win_rate: float,
    overall_hit_rate: float,
    count: int,
) -> float:
    return_scale = max(abs(float(overall_avg_return)), 0.015)
    return_term = np.clip((float(avg_return) - float(overall_avg_return)) / return_scale, -1.5, 1.5)
    win_term = np.clip(float(win_rate) - float(overall_win_rate), -0.40, 0.40)
    hit_term = np.clip(float(hit_rate) - float(overall_hit_rate), -0.40, 0.40)
    edge = (return_term * 0.09 + win_term * 0.22 + hit_term * 0.14) * _support_scale(int(count))
    return _clip(edge, -0.18, 0.18)


def _derive_segment_edges(details: pd.DataFrame, labels: pd.Series) -> tuple[dict[str, float], dict[str, int]]:
    if details.empty or labels.empty:
        return {}, {}

    working = details.copy()
    sample_weight = (
        pd.to_numeric(working["sample_weight"], errors="coerce").astype(float).fillna(1.0)
        if "sample_weight" in working.columns
        else pd.Series(1.0, index=working.index, dtype=float)
    )
    working["_segment"] = labels.astype(str).fillna("unknown")
    overall_avg_return = _weighted_average(working["next_day_return"], sample_weight, 0.0)
    overall_win_rate = _weighted_average(working["win"], sample_weight, 0.0)
    overall_hit_rate = _weighted_average(working["hit_target"], sample_weight, 0.0)
    edges: dict[str, float] = {}
    supports: dict[str, int] = {}
    for segment, group in working.groupby("_segment"):
        count = int(len(group))
        group_weight = sample_weight.reindex(group.index).fillna(1.0)
        supports[str(segment)] = count
        edges[str(segment)] = round(
            _performance_edge(
                avg_return=_weighted_average(group["next_day_return"], group_weight, 0.0),
                win_rate=_weighted_average(group["win"], group_weight, 0.0),
                hit_rate=_weighted_average(group["hit_target"], group_weight, 0.0),
                overall_avg_return=overall_avg_return,
                overall_win_rate=overall_win_rate,
                overall_hit_rate=overall_hit_rate,
                count=count,
            ),
            4,
        )
    return edges, supports


def _derive_probability_bucket_edges(details: pd.DataFrame) -> tuple[dict[str, float], dict[str, int]]:
    if details.empty:
        return {}, {}

    working = details.copy()
    sample_weight = (
        pd.to_numeric(working["sample_weight"], errors="coerce").astype(float).fillna(1.0)
        if "sample_weight" in working.columns
        else pd.Series(1.0, index=working.index, dtype=float)
    )
    working["_segment"] = working["probability_up"].apply(_probability_bucket_label)
    overall_avg_return = _weighted_average(working["next_day_return"], sample_weight, 0.0)
    edges: dict[str, float] = {}
    supports: dict[str, int] = {}
    for segment, group in working.groupby("_segment"):
        count = int(len(group))
        group_weight = sample_weight.reindex(group.index).fillna(1.0)
        supports[str(segment)] = count
        avg_return = _weighted_average(group["next_day_return"], group_weight, 0.0)
        avg_probability = _weighted_average(group["probability_up"], group_weight, 0.0) / 100.0
        win_rate = _weighted_average(group["win"], group_weight, 0.0)
        calibration_term = np.clip(win_rate - avg_probability, -0.45, 0.45)
        return_scale = max(abs(float(overall_avg_return)), 0.015)
        return_term = np.clip((avg_return - overall_avg_return) / return_scale, -1.5, 1.5)
        edge = (calibration_term * 0.32 + return_term * 0.08) * _support_scale(count)
        edges[str(segment)] = round(_clip(edge, -0.18, 0.18), 4)
    return edges, supports


def _derive_segment_statistics(details: pd.DataFrame, labels: pd.Series) -> dict[str, dict[str, float]]:
    if details.empty or labels.empty:
        return {}

    working = details.copy()
    sample_weight = (
        pd.to_numeric(working["sample_weight"], errors="coerce").astype(float).fillna(1.0)
        if "sample_weight" in working.columns
        else pd.Series(1.0, index=working.index, dtype=float)
    )
    working["_segment"] = labels.astype(str).fillna("unknown")
    stats: dict[str, dict[str, float]] = {}
    for segment, group in working.groupby("_segment"):
        count = int(len(group))
        if count <= 0:
            continue
        group_weight = sample_weight.reindex(group.index).fillna(1.0)
        avg_return = _weighted_average(group["next_day_return"], group_weight, 0.0)
        intraday_return = _weighted_average(
            group["intraday_high_return"] if "intraday_high_return" in group.columns else group["next_day_return"],
            group_weight,
            avg_return,
        )
        win_rate = _weighted_average(group["win"], group_weight, 0.0)
        hit_rate = _weighted_average(group["hit_target"], group_weight, 0.0)
        stats[str(segment)] = {
            "avg_return_pct": round(avg_return * 100, 2),
            "intraday_high_return_pct": round(intraday_return * 100, 2),
            "win_rate_pct": round(win_rate * 100, 2),
            "hit_rate_pct": round(hit_rate * 100, 2),
            "support": count,
        }
    return stats


def _derive_probability_bucket_statistics(details: pd.DataFrame) -> dict[str, dict[str, float]]:
    if details.empty:
        return {}
    labels = details["probability_up"].apply(_probability_bucket_label)
    return _derive_segment_statistics(details, labels)


def _strategy_performance_summary(details: pd.DataFrame) -> dict[str, object]:
    if details.empty:
        return {
            "best_strategy": "",
            "weakest_strategy": "",
            "strategy_stats": {},
            "strategy_count": 0,
        }
    strategy_labels = [
        _strategy_display_label(code, label)
        for code, label in zip(
            details.get("candidate_strategy", pd.Series("", index=details.index)).fillna("").astype(str).tolist(),
            details.get("candidate_strategy_label", pd.Series("", index=details.index)).fillna("").astype(str).tolist(),
        )
    ]
    stats = _derive_segment_statistics(details, pd.Series(strategy_labels, index=details.index, dtype=str))
    if not stats:
        return {
            "best_strategy": "",
            "weakest_strategy": "",
            "strategy_stats": {},
            "strategy_count": 0,
        }

    def sort_key(item: tuple[str, dict[str, float]]) -> tuple[float, float, float, float]:
        _, value = item
        return (
            _safe_float(value.get("win_rate_pct"), 0.0),
            _safe_float(value.get("avg_return_pct"), 0.0),
            _safe_float(value.get("hit_rate_pct"), 0.0),
            _safe_float(value.get("support"), 0.0),
        )

    ordered = sorted(stats.items(), key=sort_key, reverse=True)
    return {
        "best_strategy": ordered[0][0],
        "weakest_strategy": ordered[-1][0],
        "strategy_stats": stats,
        "strategy_count": int(len(stats)),
    }


def _numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.Series(dtype=float)
    if column not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").astype(float).fillna(float(default))


def _derive_market_state_labels(date_frame: pd.DataFrame) -> pd.Series:
    if not isinstance(date_frame, pd.DataFrame) or date_frame.empty:
        return pd.Series(dtype=str)

    breadth_ma20 = _numeric_series(date_frame, "breadth_ma20", 0.5)
    breadth_ret20 = _numeric_series(date_frame, "breadth_ret20", 0.5)
    mean_ret5 = _numeric_series(date_frame, "mean_ret5")
    mean_ret20 = _numeric_series(date_frame, "mean_ret20")
    mean_risk = _numeric_series(date_frame, "mean_risk")

    breadth_high = max(float(breadth_ma20.quantile(0.70)), 0.58)
    breadth_low = min(float(breadth_ma20.quantile(0.30)), 0.42)
    breadth_ret_high = max(float(breadth_ret20.quantile(0.70)), 0.48)
    ret20_high = max(float(mean_ret20.quantile(0.70)), 0.018)
    ret20_low = min(float(mean_ret20.quantile(0.30)), -0.008)
    ret5_high = max(float(mean_ret5.quantile(0.60)), 0.004)
    risk_high = max(float(mean_risk.quantile(0.70)), 188.0)

    trend_mask = (
        (breadth_ma20 >= breadth_high)
        & (breadth_ret20 >= breadth_ret_high)
        & (mean_ret20 >= ret20_high)
        & (mean_ret5 >= ret5_high)
    )
    defense_mask = (
        ((breadth_ma20 <= breadth_low) & (mean_ret20 <= ret20_low))
        | ((breadth_ma20 <= 0.45) & (mean_risk >= risk_high) & (mean_ret5 <= 0.0))
    )
    rebound_mask = (
        ~trend_mask
        & ~defense_mask
        & (mean_ret5 >= ret5_high)
        & (breadth_ma20 >= min(breadth_low + 0.05, 0.40))
        & (mean_ret20 <= max(ret20_high, 0.035))
    )

    labels = np.full(len(date_frame), "rotation", dtype=object)
    labels[rebound_mask.to_numpy(dtype=bool)] = "rebound"
    labels[defense_mask.to_numpy(dtype=bool)] = "defense"
    labels[trend_mask.to_numpy(dtype=bool)] = "trend"

    if len(set(labels.tolist())) < 3:
        trend_mask = (breadth_ma20 >= breadth_high) & (mean_ret20 >= ret20_high)
        defense_mask = (breadth_ma20 <= breadth_low) & (mean_ret20 <= ret20_low)
        rebound_mask = ~trend_mask & ~defense_mask & (mean_ret5 >= ret5_high)
        labels = np.full(len(date_frame), "rotation", dtype=object)
        labels[rebound_mask.to_numpy(dtype=bool)] = "rebound"
        labels[defense_mask.to_numpy(dtype=bool)] = "defense"
        labels[trend_mask.to_numpy(dtype=bool)] = "trend"

    return pd.Series(labels, index=date_frame.index, dtype=str)


def _derive_market_stage_proxy_labels(feature_frame: pd.DataFrame) -> pd.Series:
    if not isinstance(feature_frame, pd.DataFrame) or feature_frame.empty:
        return pd.Series(dtype=str)

    close_vs_ma20 = _numeric_series(feature_frame, "close_vs_ma20")
    ret_20 = _numeric_series(feature_frame, "ret_20")
    breakout_distance_20 = _numeric_series(feature_frame, "breakout_distance_20")
    range_position_20 = _numeric_series(feature_frame, "range_position_20", 0.5)
    consolidation_width_20 = _numeric_series(feature_frame, "consolidation_width_20", 0.25)
    volume_ratio_5 = _numeric_series(feature_frame, "volume_ratio_5", 1.0)
    upper_shadow_ratio = _numeric_series(feature_frame, "upper_shadow_ratio")
    stretch_risk = _numeric_series(feature_frame, "stretch_risk")
    risk_pressure = _numeric_series(feature_frame, "risk_pressure")

    distribution_mask = (
        (
            (
                (close_vs_ma20 >= 0.05)
                & (
                    (upper_shadow_ratio >= 0.34)
                    | (stretch_risk >= 18.0)
                    | (risk_pressure >= 205.0)
                    | (range_position_20 >= 0.88)
                )
            )
            | (
                (range_position_20 >= 0.88)
                & (
                    (upper_shadow_ratio >= 0.34)
                    | (stretch_risk >= 18.0)
                    | (risk_pressure >= 205.0)
                )
            )
        )
    )
    trend_mask = (
        (close_vs_ma20 >= 0.045)
        & (ret_20 >= 0.10)
        & (breakout_distance_20 >= -0.02)
        & (range_position_20 >= 0.62)
        & (volume_ratio_5 >= 0.95)
        & ~distribution_mask
    )
    launch_mask = (
        (ret_20 >= 0.025)
        & (ret_20 <= 0.18)
        & (close_vs_ma20 >= 0.0)
        & (close_vs_ma20 <= 0.055)
        & (breakout_distance_20 >= -0.02)
        & (breakout_distance_20 <= 0.05)
        & (range_position_20 >= 0.54)
        & (range_position_20 <= 0.82)
        & (consolidation_width_20 <= 0.28)
        & (upper_shadow_ratio <= 0.025)
        & ~distribution_mask
        & ~trend_mask
    )
    breakout_mask = (
        (close_vs_ma20 >= -0.005)
        & (breakout_distance_20 >= -0.02)
        & (range_position_20 >= 0.58)
        & (volume_ratio_5 >= 0.85)
        & ~distribution_mask
        & ~launch_mask
        & ~trend_mask
    )

    labels = np.full(len(feature_frame), "range_watch", dtype=object)
    labels[breakout_mask.to_numpy(dtype=bool)] = "breakout_confirm"
    labels[launch_mask.to_numpy(dtype=bool)] = "main_rise_start"
    labels[trend_mask.to_numpy(dtype=bool)] = "trend_drive"
    labels[distribution_mask.to_numpy(dtype=bool)] = "distribution_risk"
    return pd.Series(labels, index=feature_frame.index, dtype=str)


def _prepare_market_replay_frame(dataset: pd.DataFrame, positive_return: float) -> pd.DataFrame:
    if (
        not isinstance(dataset, pd.DataFrame)
        or dataset.empty
        or "signal_date" not in dataset.columns
        or "future_return" not in dataset.columns
    ):
        return pd.DataFrame()

    frame = dataset.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce")
    frame = frame.dropna(subset=["signal_date"]).sort_values("signal_date").reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame()

    unique_dates = sorted(frame["signal_date"].drop_duplicates().tolist())
    if len(unique_dates) > MAX_MARKET_REPLAY_TRADING_DAYS:
        cutoff = unique_dates[-MAX_MARKET_REPLAY_TRADING_DAYS]
        frame = frame.loc[frame["signal_date"] >= cutoff].reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame()

    frame["future_return"] = pd.to_numeric(frame["future_return"], errors="coerce").astype(float)
    frame = frame.dropna(subset=["future_return"]).reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame()

    for column, default in {
        "close_vs_ma20": 0.0,
        "ret_20": 0.0,
        "ret_5": 0.0,
        "risk_pressure": 0.0,
        "breakout_distance_20": 0.0,
        "range_position_20": 0.5,
        "consolidation_width_20": 0.25,
        "volume_ratio_5": 1.0,
        "upper_shadow_ratio": 0.0,
        "stretch_risk": 0.0,
    }.items():
        if column not in frame.columns:
            frame[column] = float(default)

    if "target" in frame.columns:
        frame["hit_target"] = pd.to_numeric(frame["target"], errors="coerce").astype(float).fillna(0.0)
    else:
        frame["hit_target"] = (frame["future_return"] >= float(positive_return)).astype(float)
    frame["win"] = (frame["future_return"] > 0).astype(float)
    frame["next_day_return"] = frame["future_return"].astype(float)
    frame["market_stage_proxy"] = _derive_market_stage_proxy_labels(frame)

    date_stats = (
        frame.groupby("signal_date")
        .agg(
            breadth_ma20=("close_vs_ma20", lambda values: float((pd.to_numeric(values, errors="coerce") > 0).mean())),
            breadth_ret20=("ret_20", lambda values: float((pd.to_numeric(values, errors="coerce") > 0).mean())),
            mean_ret5=("ret_5", "mean"),
            mean_ret20=("ret_20", "mean"),
            mean_risk=("risk_pressure", "mean"),
            stock_count=("symbol", "nunique"),
        )
        .reset_index()
    )
    date_stats["market_state_label"] = _derive_market_state_labels(date_stats)
    frame = frame.merge(
        date_stats[["signal_date", "market_state_label", "stock_count"]],
        on="signal_date",
        how="left",
    )
    return frame


def _default_market_replay_profile() -> dict[str, object]:
    return {
        "generated_at": "",
        "market_replay_days": 0,
        "market_replay_rows": 0,
        "market_replay_symbols": 0,
        "market_replay_start_date": "",
        "market_replay_end_date": "",
        "market_state_edges": {},
        "market_state_supports": {},
        "market_state_stats": {},
        "market_stage_proxy_edges": {},
        "market_stage_proxy_supports": {},
        "market_stage_proxy_stats": {},
        "market_replay_summary": "",
    }


def _derive_market_replay_profile_from_dataset(dataset: pd.DataFrame, positive_return: float) -> dict[str, object]:
    frame = _prepare_market_replay_frame(dataset, positive_return)
    if frame.empty:
        return _default_market_replay_profile()

    market_state_edges, market_state_supports = _derive_segment_edges(frame, frame["market_state_label"])
    market_stage_edges, market_stage_supports = _derive_segment_edges(frame, frame["market_stage_proxy"])
    market_state_stats = _derive_segment_statistics(frame, frame["market_state_label"])
    market_stage_stats = _derive_segment_statistics(frame, frame["market_stage_proxy"])
    replay_days = int(frame["signal_date"].nunique())
    replay_rows = int(len(frame))
    replay_symbols = int(frame["symbol"].nunique()) if "symbol" in frame.columns else 0
    avg_return = float(frame["next_day_return"].mean())
    win_rate = float(frame["win"].mean())
    strongest_state = max(market_state_edges, key=market_state_edges.get) if market_state_edges else "unknown"
    weakest_state = min(market_state_edges, key=market_state_edges.get) if market_state_edges else "unknown"
    strongest_stage = max(market_stage_edges, key=market_stage_edges.get) if market_stage_edges else "unknown"
    summary = (
        f"?????????? {replay_days} ?????{replay_symbols} ????{replay_rows} ????"
        f"?????? {avg_return * 100:.2f}%????? {win_rate * 100:.1f}%?"
        f"????????????? `{strongest_state}`????????? `{strongest_stage}`??????? `{weakest_state}`?"
    )
    return {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market_replay_days": replay_days,
        "market_replay_rows": replay_rows,
        "market_replay_symbols": replay_symbols,
        "market_replay_start_date": frame["signal_date"].min().strftime("%Y-%m-%d"),
        "market_replay_end_date": frame["signal_date"].max().strftime("%Y-%m-%d"),
        "market_state_edges": market_state_edges,
        "market_state_supports": market_state_supports,
        "market_state_stats": market_state_stats,
        "market_stage_proxy_edges": market_stage_edges,
        "market_stage_proxy_supports": market_stage_supports,
        "market_stage_proxy_stats": market_stage_stats,
        "market_replay_summary": summary,
    }


def load_market_replay_profile(
    *,
    horizon_days: int,
    positive_return: float,
    train_start: str = GLOBAL_MODEL_TRAIN_START,
    train_end: str = GLOBAL_MODEL_TRAIN_END,
    test_start: str = GLOBAL_MODEL_TEST_START,
    test_end: str = GLOBAL_MODEL_TEST_END,
) -> dict[str, object]:
    path = _market_replay_profile_cache_path(
        horizon_days,
        positive_return,
        train_start,
        train_end,
        test_start,
        test_end,
    )
    if path.exists():
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            if payload.get("cache_version") == MARKET_REPLAY_CACHE_VERSION and isinstance(payload.get("profile"), dict):
                return payload["profile"]
        except Exception:
            pass

    dataset = _load_partial_market_dataset(
        horizon_days=horizon_days,
        positive_return=positive_return,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )
    profile = _derive_market_replay_profile_from_dataset(dataset, positive_return)
    try:
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "cache_version": MARKET_REPLAY_CACHE_VERSION,
                    "meta": {
                        "horizon_days": int(horizon_days),
                        "positive_return": float(positive_return),
                        "train_start": train_start,
                        "train_end": train_end,
                        "test_start": test_start,
                        "test_end": test_end,
                    },
                    "profile": profile,
                },
                handle,
            )
    except Exception:
        pass
    return profile


def _merge_profile_layers(short_profile: dict[str, object], market_profile: dict[str, object]) -> dict[str, object]:
    merged = _normalize_short_profile(short_profile)
    if not market_profile:
        return merged
    for key in (
        "market_replay_days",
        "market_replay_rows",
        "market_replay_symbols",
        "market_replay_start_date",
        "market_replay_end_date",
        "market_state_edges",
        "market_state_supports",
        "market_state_stats",
        "market_stage_proxy_edges",
        "market_stage_proxy_supports",
        "market_stage_proxy_stats",
        "market_replay_summary",
    ):
        merged[key] = market_profile.get(key, _default_market_replay_profile().get(key))

    short_summary = str(merged.get("profile_summary") or "").strip()
    market_summary = str(market_profile.get("market_replay_summary") or "").strip()
    if short_summary and market_summary:
        merged["profile_summary"] = f"{short_summary} {market_summary}"
    elif market_summary:
        merged["profile_summary"] = market_summary
    return merged


def _resolve_market_state_label(payload: dict[str, object]) -> str:
    explicit = str(payload.get("market_state_label") or "").strip()
    if explicit in MARKET_STATE_LABELS:
        return explicit
    regime_label = str(payload.get("market_regime_label") or "").strip()
    if regime_label in MARKET_STATE_LABELS:
        return regime_label

    market_ret_5 = _safe_float(payload.get("market_ret_5"), 0.0)
    market_ret_20 = _safe_float(payload.get("market_ret_20"), 0.0)
    market_close_vs_ma20 = _safe_float(payload.get("market_close_vs_ma20"), 0.0)
    market_volatility_10 = _safe_float(payload.get("market_volatility_10"), 0.0)
    market_range_position_20 = _safe_float(payload.get("market_range_position_20"), 0.5)
    if max(
        abs(market_ret_5),
        abs(market_ret_20),
        abs(market_close_vs_ma20),
        abs(market_volatility_10),
        abs(market_range_position_20 - 0.5),
    ) <= 1e-6:
        return "unknown"

    trend_score = (
        market_ret_20 * 18.0
        + market_close_vs_ma20 * 26.0
        + (market_range_position_20 - 0.55) * 3.5
        - market_volatility_10 * 8.0
    )
    rebound_score = (
        market_ret_5 * 20.0
        - min(market_ret_20, 0.0) * 10.0
        - abs(market_close_vs_ma20) * 6.0
        - max(market_range_position_20 - 0.68, 0.0) * 4.0
    )
    rotation_score = (
        1.8
        - abs(market_ret_20) * 14.0
        - abs(market_close_vs_ma20) * 18.0
        - abs(market_range_position_20 - 0.5) * 3.0
        - market_volatility_10 * 5.0
    )
    defense_score = (
        -market_ret_20 * 18.0
        - market_close_vs_ma20 * 22.0
        + market_volatility_10 * 10.0
        + max(0.45 - market_range_position_20, 0.0) * 6.0
    )
    scores = {
        "trend": trend_score,
        "rebound": rebound_score,
        "rotation": rotation_score,
        "defense": defense_score,
    }
    return max(scores, key=scores.get)


def _resolve_market_stage_proxy(payload: dict[str, object]) -> str:
    explicit = str(payload.get("market_stage_proxy") or "").strip()
    if explicit in MARKET_STAGE_PROXY_LABELS:
        return explicit
    proxy_frame = pd.DataFrame([payload])
    if proxy_frame.empty:
        return "unknown"
    labels = _derive_market_stage_proxy_labels(proxy_frame)
    return str(labels.iloc[0]) if not labels.empty else "unknown"


def _profile_has_replay_overlay(profile: dict[str, object]) -> bool:
    required = (
        "stage_edges",
        "stage_supports",
        "precision_segment_edges",
        "precision_segment_supports",
        "probability_bucket_edges",
        "probability_bucket_supports",
        "quant_bucket_edges",
        "quant_bucket_supports",
        "launch_bucket_edges",
        "launch_bucket_supports",
        "resonance_bucket_edges",
        "resonance_bucket_supports",
        "launch_window_bucket_edges",
        "launch_window_bucket_supports",
        "launch_window_status_edges",
        "launch_window_status_supports",
    )
    return all(key in profile for key in required)


def _snapshot_columns(board: pd.DataFrame) -> list[str]:
    preferred = [
        "symbol",
        "name",
        "rank",
        "analysis_date",
        "attention_score",
        "enhanced_attention_score",
        "raw_probability_up",
        "enhanced_probability_up",
        "probability_up",
        "predicted_upside_pct",
        "predicted_upside_low_pct",
        "predicted_upside_high_pct",
        "final_rank_score",
        "quant_score",
        "sector_score",
        "fund_score",
        "news_score",
        "launch_score",
        "stage_score",
        "stage_code",
        "launch_readiness_score",
        "launch_readiness",
        "breakout_quality",
        "resonance_quality",
        "board_resonance_strength",
        "long_setup_quality",
        "crowding_risk",
        "crowding_risk_label",
        "risk_of_late_entry",
        "launch_phase_label",
        "market_resonance_score",
        "launch_specialist_score",
        "launch_regime_fit_score",
        "launch_specialist_confidence",
        "launch_window_label",
        "launch_window_status",
        "launch_window_summary",
        "launch_window_score",
        "launch_window_confidence",
        "launch_window_confidence_weight",
        "selection_score",
        "selection_confidence",
        "technical_adjustment",
        "intraday_adjustment",
        "backtest_adjustment",
        "execution_label",
        "execution_window",
        "execution_score",
        "execution_confidence",
        "candidate_strategy",
        "candidate_strategy_label",
        "candidate_strategy_short_label",
        "action_label",
        "precision_gate_label",
        "precision_gate_precision",
        "precision_gate_support",
        "amount",
        "consecutive_up_days",
        "stage_label",
        "market_state_label",
        "market_stage_proxy",
        "market_regime_label",
        "ret_20",
        "close_vs_ma20",
        "breakout_distance_20",
        "range_position_20",
        "volume_ratio_5",
        "upper_shadow_ratio",
        "stretch_risk",
        "risk_pressure",
        "market_ret_5",
        "market_ret_20",
        "market_close_vs_ma20",
        "market_volatility_10",
        "market_range_position_20",
        "reason",
        "tomorrow_setup",
        "tomorrow_bias",
        "tomorrow_buy_point",
        "tomorrow_sell_point",
        "tomorrow_plan_confidence",
    ]
    return [column for column in preferred if column in board.columns]


def _with_snapshot_score_fallbacks(board: pd.DataFrame) -> pd.DataFrame:
    if board.empty:
        return board.copy()
    snapshot = board.copy()
    if "final_rank_score" not in snapshot.columns:
        final_rank = pd.Series(pd.NA, index=snapshot.index, dtype="Float64")
        for column in ("ranking_score", "enhanced_attention_score", "attention_score"):
            if column in snapshot.columns:
                final_rank = final_rank.fillna(pd.to_numeric(snapshot[column], errors="coerce"))
        snapshot["final_rank_score"] = final_rank.fillna(0.0)
    return snapshot


def persist_focus_board_snapshot(
    board: pd.DataFrame,
    *,
    horizon_days: int,
    positive_return: float,
    ranking_by: str,
    board_size: int,
) -> Path | None:
    if board.empty:
        return None
    board_date = str(board.attrs.get("market_data_date") or "")
    if not board_date:
        return None

    snapshot_path = _snapshot_cache_path(horizon_days, positive_return, ranking_by, board_size, board_date)
    board_with_fallbacks = _with_snapshot_score_fallbacks(board)
    snapshot = board_with_fallbacks[_snapshot_columns(board_with_fallbacks)].copy()
    payload = {
        "meta": {
            "cache_version": DAILY_REVIEW_CACHE_VERSION,
            "board_date": board_date,
            "latest_market_data_date": str(board.attrs.get("latest_market_data_date") or board_date),
            "ranking_by": ranking_by,
            "board_size": int(board_size),
            "horizon_days": int(horizon_days),
            "positive_return": float(positive_return),
            "model_source_label": str(board.attrs.get("model_source_label") or ""),
            "computed_at": str(board.attrs.get("computed_at") or ""),
            "captured_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "board": snapshot,
    }
    with snapshot_path.open("wb") as handle:
        pickle.dump(payload, handle)
    return snapshot_path


def _load_snapshot_payload(path: Path) -> dict | None:
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None

    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    board = payload.get("board") if isinstance(payload, dict) else None
    if int(meta.get("cache_version", -1) or -1) not in _supported_review_cache_versions() or not isinstance(board, pd.DataFrame):
        return None
    return {"meta": meta, "board": board.copy()}


def _save_review_payload(
    review_path: Path,
    *,
    summary: dict[str, object],
    details: pd.DataFrame,
    meta: dict[str, object],
) -> None:
    payload = {
        "meta": {"cache_version": DAILY_REVIEW_CACHE_VERSION, **meta},
        "summary": summary,
        "details": details,
    }
    with review_path.open("wb") as handle:
        pickle.dump(payload, handle)


def _load_review_payload(path: Path) -> dict | None:
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None

    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    summary = payload.get("summary") if isinstance(payload, dict) else None
    details = payload.get("details") if isinstance(payload, dict) else None
    if int(meta.get("cache_version", -1) or -1) not in _supported_review_cache_versions():
        return None
    if not isinstance(summary, dict) or not isinstance(details, pd.DataFrame):
        return None
    return {"meta": meta, "summary": summary, "details": details.copy()}


def _meta_date_sort_key(meta: dict[str, object], *keys: str) -> tuple[pd.Timestamp, ...]:
    values: list[pd.Timestamp] = []
    for key in keys:
        values.append(pd.Timestamp(pd.to_datetime(meta.get(key), errors="coerce") or pd.Timestamp.min))
    return tuple(values)


def _matching_snapshot_payloads(
    *,
    horizon_days: int,
    positive_return: float,
    ranking_by: str,
    board_size: int,
) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for cache_version in _supported_review_cache_versions():
        pattern = f"snapshot_v{cache_version}_h{horizon_days}_r{int(positive_return * 10000)}_b{board_size}_*.pkl"
        for path in sorted(_review_cache_dir().glob(pattern)):
            payload = _load_snapshot_payload(path)
            if payload is None:
                continue
            meta = payload.get("meta", {})
            if str(meta.get("ranking_by") or "") != str(ranking_by):
                continue
            if int(meta.get("board_size", 0) or 0) != int(board_size):
                continue
            payloads.append(payload)
    payloads.sort(
        key=lambda item: _meta_date_sort_key(
            item.get("meta", {}),
            "board_date",
            "latest_market_data_date",
            "captured_at",
        )
    )
    return payloads


def _matching_review_payloads(
    *,
    horizon_days: int,
    positive_return: float,
    ranking_by: str,
    board_size: int,
) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for cache_version in _supported_review_cache_versions():
        pattern = f"review_v{cache_version}_h{horizon_days}_r{int(positive_return * 10000)}_b{board_size}_*.pkl"
        for path in sorted(_review_cache_dir().glob(pattern)):
            payload = _load_review_payload(path)
            if payload is None:
                continue
            meta = payload.get("meta", {})
            if str(meta.get("ranking_by") or "") != str(ranking_by):
                continue
            if int(meta.get("board_size", 0) or 0) != int(board_size):
                continue
            payloads.append(payload)
    payloads.sort(key=lambda item: _meta_date_sort_key(item.get("meta", {}), "review_date", "board_date"))
    return payloads


def _reference_trading_days(
    start_date: str,
    end_date: str,
    reference_symbol: str = "600519",
) -> pd.DatetimeIndex:
    daily = fetch_daily_history(
        reference_symbol,
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
    )
    if daily.empty:
        return pd.DatetimeIndex([])
    if "date" in daily.columns:
        dates = pd.to_datetime(daily["date"], errors="coerce").dropna()
    else:
        dates = pd.to_datetime(daily.index, errors="coerce").dropna()
    return pd.DatetimeIndex(pd.Series(dates).drop_duplicates().sort_values())


def _next_review_date(board_date: str, latest_market_data_date: str) -> str | None:
    trading_days = _reference_trading_days(board_date, latest_market_data_date)
    if trading_days.empty:
        return None
    board_ts = pd.Timestamp(board_date)
    future_days = trading_days[trading_days > board_ts]
    if future_days.empty:
        return None
    review_ts = future_days.min()
    if review_ts > pd.Timestamp(latest_market_data_date):
        return None
    return review_ts.strftime("%Y-%m-%d")


def _review_snapshot(
    snapshot_board: pd.DataFrame,
    *,
    board_date: str,
    review_date: str,
    positive_return: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    board_ts = pd.Timestamp(board_date)
    review_ts = pd.Timestamp(review_date)

    for row in snapshot_board.itertuples(index=False):
        symbol = str(getattr(row, "symbol"))
        try:
            daily = fetch_daily_history(
                symbol,
                start_date=board_ts.strftime("%Y%m%d"),
                end_date=review_ts.strftime("%Y%m%d"),
            )
        except Exception:
            continue
        if daily.empty:
            continue

        view = daily.copy()
        if "date" in view.columns:
            view = view.reset_index(drop=True)
        else:
            index_name = view.index.name or "index"
            view = view.reset_index().rename(columns={index_name: "date"})
        view["date"] = pd.to_datetime(view["date"], errors="coerce")
        view = view.dropna(subset=["date"]).sort_values("date")
        board_row = view.loc[view["date"] == board_ts]
        review_row = view.loc[view["date"] == review_ts]
        if board_row.empty or review_row.empty:
            continue

        entry_close = float(board_row["close"].iloc[-1])
        review_close = float(review_row["close"].iloc[-1])
        review_high = float(review_row["high"].iloc[-1]) if "high" in review_row.columns else review_close
        if not entry_close:
            continue

        next_day_return = review_close / entry_close - 1
        intraday_high_return = review_high / entry_close - 1
        probability_pct = float(getattr(row, "probability_up", 0.0))
        probability = float(np.clip(probability_pct / 100.0, 0.0, 1.0))
        attention_score = float(getattr(row, "attention_score", 0.0))
        enhanced_attention_score = float(getattr(row, "enhanced_attention_score", attention_score))
        quant_score = float(getattr(row, "quant_score", 0.0))
        direction_hit = float((probability >= 0.50) == bool(next_day_return > 0))
        direction_error_pct = float(abs(probability - float(next_day_return > 0)) * 100)
        target_progress_pct = float(next_day_return / positive_return * 100) if positive_return > 0 else 0.0
        intraday_target_progress_pct = float(intraday_high_return / positive_return * 100) if positive_return > 0 else 0.0
        rows.append(
            {
                "symbol": symbol,
                "name": str(getattr(row, "name", symbol)),
                "board_date": board_date,
                "review_date": review_date,
                "rank": int(getattr(row, "rank", len(rows) + 1)),
                "probability_up": probability_pct,
                "predicted_upside_pct": float(getattr(row, "predicted_upside_pct", 0.0)),
                "predicted_upside_low_pct": float(getattr(row, "predicted_upside_low_pct", 0.0)),
                "predicted_upside_high_pct": float(getattr(row, "predicted_upside_high_pct", 0.0)),
                "raw_probability_up": float(getattr(row, "raw_probability_up", probability_pct)),
                "enhanced_probability_up": float(getattr(row, "enhanced_probability_up", probability_pct)),
                "final_rank_score": float(
                    getattr(row, "final_rank_score", getattr(row, "ranking_score", enhanced_attention_score))
                ),
                "attention_score": attention_score,
                "enhanced_attention_score": enhanced_attention_score,
                "quant_score": quant_score,
                "sector_score": float(getattr(row, "sector_score", 50.0)),
                "fund_score": float(getattr(row, "fund_score", 50.0)),
                "news_score": float(getattr(row, "news_score", 50.0)),
                "launch_score": float(getattr(row, "launch_score", _safe_float(getattr(row, "launch_readiness_score", 50.0)))),
                "launch_readiness_score": float(getattr(row, "launch_readiness_score", 50.0)),
                "launch_readiness": float(getattr(row, "launch_readiness", getattr(row, "launch_readiness_score", 50.0))),
                "breakout_quality": float(getattr(row, "breakout_quality", 50.0)),
                "resonance_quality": float(getattr(row, "resonance_quality", 50.0)),
                "board_resonance_strength": float(getattr(row, "board_resonance_strength", 50.0)),
                "long_setup_quality": float(getattr(row, "long_setup_quality", 50.0)),
                "crowding_risk": float(getattr(row, "crowding_risk", 50.0)),
                "crowding_risk_label": str(getattr(row, "crowding_risk_label", "")),
                "risk_of_late_entry": float(getattr(row, "risk_of_late_entry", 50.0)),
                "launch_phase_label": str(getattr(row, "launch_phase_label", "")),
                "market_resonance_score": float(getattr(row, "market_resonance_score", 50.0)),
                "intraday_sector_sync_score": float(getattr(row, "intraday_sector_sync_score", 50.0)),
                "relative_intraday_alpha": float(getattr(row, "relative_intraday_alpha", 0.0)),
                "sector_follow_score": float(getattr(row, "sector_follow_score", 50.0)),
                "intraday_sector_state": str(getattr(row, "intraday_sector_state", "")),
                "launch_specialist_score": float(getattr(row, "launch_specialist_score", 50.0)),
                "launch_regime_fit_score": float(getattr(row, "launch_regime_fit_score", 50.0)),
                "launch_specialist_confidence": float(getattr(row, "launch_specialist_confidence", 50.0)),
                "launch_window_label": str(getattr(row, "launch_window_label", "")),
                "launch_window_status": str(getattr(row, "launch_window_status", "")),
                "launch_window_summary": str(getattr(row, "launch_window_summary", "")),
                "launch_window_score": float(getattr(row, "launch_window_score", 50.0)),
                "launch_window_confidence": float(getattr(row, "launch_window_confidence", 50.0)),
                "launch_window_confidence_weight": float(getattr(row, "launch_window_confidence_weight", 0.04)),
                "candidate_strategy": str(getattr(row, "candidate_strategy", "")),
                "candidate_strategy_label": _strategy_display_label(
                    str(getattr(row, "candidate_strategy", "")),
                    str(getattr(row, "candidate_strategy_label", "")),
                ),
                "candidate_strategy_short_label": str(
                    getattr(
                        row,
                        "candidate_strategy_short_label",
                        _strategy_display_label(str(getattr(row, "candidate_strategy", "")), str(getattr(row, "candidate_strategy_label", ""))),
                    )
                ),
                "action_label": str(getattr(row, "action_label", "")),
                "market_state_label": str(getattr(row, "market_state_label", "")),
                "market_stage_proxy": str(getattr(row, "market_stage_proxy", "")),
                "stage_code": str(getattr(row, "stage_code", "")),
                "stage_score": float(getattr(row, "stage_score", 50.0)),
                "next_day_return": float(next_day_return),
                "intraday_high_return": float(intraday_high_return),
                "next_day_return_pct": round(float(next_day_return * 100), 2),
                "intraday_high_return_pct": round(float(intraday_high_return * 100), 2),
                "win": float(next_day_return > 0),
                "hit_target": float(next_day_return >= positive_return),
                "direction_hit": direction_hit,
                "direction_error_pct": round(direction_error_pct, 2),
                "target_progress_pct": round(target_progress_pct, 2),
                "intraday_target_progress_pct": round(intraday_target_progress_pct, 2),
                "precision_gate_label": str(getattr(row, "precision_gate_label", "")),
                "precision_gate_precision": float(getattr(row, "precision_gate_precision", 0.0)),
                "precision_gate_support": int(getattr(row, "precision_gate_support", 0) or 0),
                "stage_label": str(getattr(row, "stage_label", "")),
                "selection_score": float(getattr(row, "selection_score", attention_score)),
                "selection_confidence": float(getattr(row, "selection_confidence", 50.0)),
                "tomorrow_plan_confidence": float(getattr(row, "tomorrow_plan_confidence", 50.0)),
                "technical_adjustment": float(getattr(row, "technical_adjustment", 0.0)),
                "intraday_adjustment": float(getattr(row, "intraday_adjustment", 0.0)),
                "backtest_adjustment": float(getattr(row, "backtest_adjustment", 0.0)),
                "execution_label": str(getattr(row, "execution_label", "")),
                "execution_window": str(getattr(row, "execution_window", "")),
                "execution_score": float(getattr(row, "execution_score", 50.0)),
                "execution_confidence": float(getattr(row, "execution_confidence", 50.0)),
                "reason": str(getattr(row, "reason", "")),
            }
        )

    details = pd.DataFrame(rows)
    if details.empty:
        summary = FocusBoardReviewSummary(
            board_date=board_date,
            review_date=review_date,
            review_count=0,
            avg_return_pct=0.0,
            win_rate_pct=0.0,
            top_probability_return_pct=0.0,
            top_attention_return_pct=0.0,
            target_hit_rate_pct=0.0,
            direction_hit_rate_pct=0.0,
            calibration_gap_pct=0.0,
            direction_brier_score=0.0,
            avg_target_progress_pct=0.0,
            optimization_note="上一交易日关注榜暂时没有形成可评估样本，系统会先保留默认排序权重。",
        )
        summary_payload = asdict(summary)
        summary_payload.update(_strategy_performance_summary(details))
        return summary_payload, details

    top_n = max(3, min(10, len(details) // 5 if len(details) >= 10 else 3))
    probability_ranked = details.sort_values(["probability_up", "rank"], ascending=[False, True])
    attention_ranked = details.sort_values(["attention_score", "rank"], ascending=[False, True])
    top_probability_return = float(probability_ranked.head(top_n)["next_day_return"].mean())
    top_attention_return = float(attention_ranked.head(top_n)["next_day_return"].mean())
    avg_return = float(details["next_day_return"].mean())
    win_rate = float(details["win"].mean())
    target_hit_rate = float(details["hit_target"].mean())
    avg_probability = float(details["probability_up"].mean()) / 100.0
    direction_hit_rate = float(details["direction_hit"].mean())
    calibration_gap_pct = float(abs(avg_probability - win_rate) * 100)
    direction_brier_score = float(((details["probability_up"] / 100.0 - details["win"]) ** 2).mean())
    avg_target_progress_pct = float(details["target_progress_pct"].mean())

    note = (
        f"上一交易日关注榜共回测 {len(details)} 只，次日平均收益 {avg_return * 100:.2f}%，"
        f"上涨胜率 {win_rate * 100:.1f}%。"
    )
    if top_probability_return > top_attention_return + 0.002:
        note += " 最近上涨概率排序的前排反馈更强，系统会适度提高概率因子的排序权重。"
    elif top_attention_return > top_probability_return + 0.002:
        note += " 最近关注分数排序的前排反馈更稳，系统会适度提高结构分的排序权重。"
    else:
        note += " 最近概率与结构分的前排反馈接近，系统继续保持均衡权重。"

    summary = FocusBoardReviewSummary(
        board_date=board_date,
        review_date=review_date,
        review_count=int(len(details)),
        avg_return_pct=round(avg_return * 100, 2),
        win_rate_pct=round(win_rate * 100, 2),
        top_probability_return_pct=round(top_probability_return * 100, 2),
        top_attention_return_pct=round(top_attention_return * 100, 2),
        target_hit_rate_pct=round(target_hit_rate * 100, 2),
        direction_hit_rate_pct=round(direction_hit_rate * 100, 2),
        calibration_gap_pct=round(calibration_gap_pct, 2),
        direction_brier_score=round(direction_brier_score, 4),
        avg_target_progress_pct=round(avg_target_progress_pct, 2),
        optimization_note=note,
    )
    summary_payload = asdict(summary)
    strategy_summary = _strategy_performance_summary(details)
    summary_payload.update(strategy_summary)
    if strategy_summary.get("best_strategy"):
        summary_payload["optimization_note"] = (
            f"{summary_payload['optimization_note']} 分策略看，当前更有效的是"
            f"{strategy_summary.get('best_strategy')}，相对较弱的是{strategy_summary.get('weakest_strategy')}。"
        )
    return summary_payload, details


def _derive_adaptive_profile(
    review_payloads: list[dict[str, object]],
    *,
    rolling_review_days: int | None = None,
) -> dict[str, object]:
    rolling_limit = _normalize_rolling_review_days(rolling_review_days)
    if not review_payloads:
        return _default_profile("自动复盘已启用，当前还没有足够的上一交易日样本，先使用默认排序权重。")

    recent_payloads = review_payloads[-rolling_limit:]
    detail_frames: list[pd.DataFrame] = []
    payload_count = max(len(recent_payloads), 1)
    for index, payload in enumerate(recent_payloads):
        details = payload["details"]
        if details.empty:
            continue
        weighted_details = details.copy()
        recency_weight = 0.58 + 0.42 * ((index + 1) / payload_count)
        if "rank" in weighted_details.columns:
            rank_series = pd.to_numeric(weighted_details["rank"], errors="coerce").astype(float)
            rank_denom = max(float(rank_series.max()), 1.0)
            rank_weight = (1.18 - ((rank_series - 1.0) / rank_denom).clip(lower=0.0, upper=1.0) * 0.32).fillna(1.0)
        else:
            rank_weight = pd.Series(1.0, index=weighted_details.index, dtype=float)
        weighted_details["sample_weight"] = recency_weight * rank_weight.astype(float)
        detail_frames.append(weighted_details)
    if not detail_frames:
        return _default_profile("最近没有可用的复盘样本，系统会先沿用默认排序权重。")

    details = pd.concat(detail_frames, ignore_index=True)
    sample_weight = (
        pd.to_numeric(details["sample_weight"], errors="coerce").astype(float).fillna(1.0)
        if "sample_weight" in details.columns
        else pd.Series(1.0, index=details.index, dtype=float)
    )
    target_rank = details["next_day_return"].rank(pct=True)
    factor_edges: dict[str, float] = {}
    for column in DEFAULT_PROFILE_WEIGHTS:
        if column not in details.columns or details[column].nunique() < 2:
            factor_edges[column] = 0.0
            continue
        corr = _weighted_corr(details[column].rank(pct=True), target_rank, sample_weight)
        factor_edges[column] = float(np.clip(corr, -0.30, 0.30))

    raw_weights = {
        column: max(0.05, base_weight * (1.0 + factor_edges[column] * 1.35))
        for column, base_weight in DEFAULT_PROFILE_WEIGHTS.items()
    }
    total_weight = sum(raw_weights.values()) or 1.0
    normalized_weights = {column: round(weight / total_weight, 4) for column, weight in raw_weights.items()}
    stage_edges, stage_supports = _derive_segment_edges(
        details,
        details["stage_label"] if "stage_label" in details.columns else pd.Series(dtype=str),
    )
    stage_stats = _derive_segment_statistics(
        details,
        details["stage_label"] if "stage_label" in details.columns else pd.Series(dtype=str),
    )
    strategy_labels = pd.Series(
        [
            _strategy_display_label(code, label)
            for code, label in zip(
                details.get("candidate_strategy", pd.Series("", index=details.index)).fillna("").astype(str).tolist(),
                details.get("candidate_strategy_label", pd.Series("", index=details.index)).fillna("").astype(str).tolist(),
            )
        ],
        index=details.index,
        dtype=str,
    )
    strategy_edges, strategy_supports = _derive_segment_edges(details, strategy_labels)
    strategy_stats = _derive_segment_statistics(details, strategy_labels)
    precision_edges, precision_supports = _derive_segment_edges(
        details,
        (
            details["precision_gate_label"].apply(_precision_segment_label)
            if "precision_gate_label" in details.columns
            else pd.Series(dtype=str)
        ),
    )
    probability_bucket_edges, probability_bucket_supports = _derive_probability_bucket_edges(details)
    probability_bucket_stats = _derive_probability_bucket_statistics(details)
    quant_bucket_edges, quant_bucket_supports = _derive_segment_edges(
        details,
        details["quant_score"].apply(_quant_bucket_label) if "quant_score" in details.columns else pd.Series(dtype=str),
    )
    quant_bucket_stats = _derive_segment_statistics(
        details,
        details["quant_score"].apply(_quant_bucket_label) if "quant_score" in details.columns else pd.Series(dtype=str),
    )
    launch_bucket_edges, launch_bucket_supports = _derive_segment_edges(
        details,
        details["launch_score"].apply(_launch_bucket_label) if "launch_score" in details.columns else pd.Series(dtype=str),
    )
    launch_bucket_stats = _derive_segment_statistics(
        details,
        details["launch_score"].apply(_launch_bucket_label) if "launch_score" in details.columns else pd.Series(dtype=str),
    )
    resonance_bucket_edges, resonance_bucket_supports = _derive_segment_edges(
        details,
        (
            details["market_resonance_score"].apply(_resonance_bucket_label)
            if "market_resonance_score" in details.columns
            else pd.Series(dtype=str)
        ),
    )
    resonance_bucket_stats = _derive_segment_statistics(
        details,
        (
            details["market_resonance_score"].apply(_resonance_bucket_label)
            if "market_resonance_score" in details.columns
            else pd.Series(dtype=str)
        ),
    )
    launch_window_bucket_edges, launch_window_bucket_supports = _derive_segment_edges(
        details,
        (
            details["launch_window_score"].apply(_launch_window_bucket_label)
            if "launch_window_score" in details.columns
            else pd.Series(dtype=str)
        ),
    )
    launch_window_bucket_stats = _derive_segment_statistics(
        details,
        (
            details["launch_window_score"].apply(_launch_window_bucket_label)
            if "launch_window_score" in details.columns
            else pd.Series(dtype=str)
        ),
    )
    launch_window_status_edges, launch_window_status_supports = _derive_segment_edges(
        details,
        (
            details["launch_window_status"].apply(_launch_window_status_label)
            if "launch_window_status" in details.columns
            else pd.Series(dtype=str)
        ),
    )
    launch_window_status_stats = _derive_segment_statistics(
        details,
        (
            details["launch_window_status"].apply(_launch_window_status_label)
            if "launch_window_status" in details.columns
            else pd.Series(dtype=str)
        ),
    )

    avg_return = _weighted_average(details["next_day_return"], sample_weight, 0.0)
    win_rate = _weighted_average(details["win"], sample_weight, 0.0)
    best_factor = max(factor_edges, key=factor_edges.get)
    weakest_factor = min(factor_edges, key=factor_edges.get)
    best_stage = max(stage_edges, key=stage_edges.get) if stage_edges else "unknown"
    weakest_stage = min(stage_edges, key=stage_edges.get) if stage_edges else "unknown"
    best_strategy = max(strategy_edges, key=strategy_edges.get) if strategy_edges else ""
    weakest_strategy = min(strategy_edges, key=strategy_edges.get) if strategy_edges else ""
    best_launch_window = (
        max(launch_window_status_edges, key=launch_window_status_edges.get)
        if launch_window_status_edges
        else "unknown"
    )
    summary = (
        f"最近 {len(recent_payloads)} 个复盘日、{len(details)} 只样本的次日平均收益为 {avg_return * 100:.2f}%，"
        f"上涨胜率 {win_rate * 100:.1f}%。当前最有效的排序因子偏向 `{best_factor}`，"
        f"相对较弱的是 `{weakest_factor}`。回放样本中更强的状态偏向 `{best_stage}`，"
        f"更需要谨慎的是 `{weakest_stage}`。"
    )
    if best_strategy:
        summary += f" 分策略看，最近更有效的是 `{best_strategy}`，相对较弱的是 `{weakest_strategy}`。"
    return {
        "weights": normalized_weights,
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "review_days": int(len(recent_payloads)),
        "review_stocks": int(len(details)),
        "rolling_review_days": int(rolling_limit),
        "rolling_review_mode": "recent_n_trading_days",
        "avg_return_pct": round(avg_return * 100, 2),
        "win_rate_pct": round(win_rate * 100, 2),
        "factor_edges": {key: round(value, 4) for key, value in factor_edges.items()},
        "stage_edges": stage_edges,
        "stage_supports": stage_supports,
        "stage_stats": stage_stats,
        "strategy_edges": strategy_edges,
        "strategy_supports": strategy_supports,
        "strategy_stats": strategy_stats,
        "best_strategy": best_strategy,
        "weakest_strategy": weakest_strategy,
        "precision_segment_edges": precision_edges,
        "precision_segment_supports": precision_supports,
        "probability_bucket_edges": probability_bucket_edges,
        "probability_bucket_supports": probability_bucket_supports,
        "probability_bucket_stats": probability_bucket_stats,
        "quant_bucket_edges": quant_bucket_edges,
        "quant_bucket_supports": quant_bucket_supports,
        "quant_bucket_stats": quant_bucket_stats,
        "launch_bucket_edges": launch_bucket_edges,
        "launch_bucket_supports": launch_bucket_supports,
        "launch_bucket_stats": launch_bucket_stats,
        "resonance_bucket_edges": resonance_bucket_edges,
        "resonance_bucket_supports": resonance_bucket_supports,
        "resonance_bucket_stats": resonance_bucket_stats,
        "launch_window_bucket_edges": launch_window_bucket_edges,
        "launch_window_bucket_supports": launch_window_bucket_supports,
        "launch_window_bucket_stats": launch_window_bucket_stats,
        "launch_window_status_edges": launch_window_status_edges,
        "launch_window_status_supports": launch_window_status_supports,
        "launch_window_status_stats": launch_window_status_stats,
        "calibration_scope": "rank_score_and_risk_overlay_only",
        "model_parameter_update_allowed": False,
        "allowed_calibration_targets": [
            "ranking_weight_micro_adjustment",
            "strategy_fit_score",
            "high_risk_pattern_suppression",
        ],
        "profile_summary": summary,
    }


def load_adaptive_rank_profile(
    *,
    horizon_days: int,
    positive_return: float,
    ranking_by: str,
    board_size: int,
) -> dict[str, object]:
    market_profile = load_market_replay_profile(
        horizon_days=horizon_days,
        positive_return=positive_return,
    )
    path = _profile_cache_path(horizon_days, positive_return, ranking_by, board_size)
    candidate_paths = [path] if path.exists() else []
    if not candidate_paths:
        candidate_paths = []
        for cache_version in _supported_review_cache_versions():
            pattern = f"profile_v{cache_version}_h{horizon_days}_r{int(positive_return * 10000)}_b{board_size}_*.pkl"
            candidate_paths.extend(sorted(_review_cache_dir().glob(pattern)))
    for candidate_path in reversed(candidate_paths):
        try:
            with candidate_path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception:
            continue
        if int(payload.get("cache_version", -1) or -1) not in _supported_review_cache_versions():
            continue
        meta = payload.get("meta", {})
        if meta and str(meta.get("ranking_by") or "") not in {"", str(ranking_by)}:
            continue
        profile = payload.get("profile")
        if not isinstance(profile, dict):
            continue
        profile = _normalize_short_profile(profile)
        if _profile_has_replay_overlay(profile):
            return _merge_profile_layers(profile, market_profile)
        review_payloads = _matching_review_payloads(
            horizon_days=horizon_days,
            positive_return=positive_return,
            ranking_by=ranking_by,
            board_size=board_size,
        )
        if review_payloads:
            return _merge_profile_layers(_derive_adaptive_profile(review_payloads), market_profile)
        return _merge_profile_layers(profile, market_profile)
    return _merge_profile_layers(
        _default_profile("自适应优化尚未形成有效样本，当前先使用默认排序权重。"),
        market_profile,
    )


def compute_replay_calibrated_scores(payload: dict[str, object] | pd.Series, profile: dict[str, object] | None) -> dict[str, object]:
    row = payload if isinstance(payload, dict) else payload.to_dict()
    probability_up = _safe_float(row.get("probability_up"), 0.0)
    attention_score = _safe_float(row.get("attention_score"), 0.0)
    enhanced_attention_score = _safe_float(row.get("enhanced_attention_score"), attention_score)
    quant_score = _safe_float(row.get("quant_score"), 0.0)
    launch_score = _safe_float(row.get("launch_score"), _safe_float(row.get("launch_readiness_score"), 50.0))
    market_resonance_score = _safe_float(row.get("market_resonance_score"), 50.0)
    launch_window_score = _safe_float(row.get("launch_window_score"), 50.0)
    launch_window_status = _launch_window_status_label(str(row.get("launch_window_status") or ""))
    stage_label = str(row.get("stage_label") or "unknown")
    precision_gate_label = str(row.get("precision_gate_label") or "")
    precision_segment = _precision_segment_label(precision_gate_label)
    probability_bucket = _probability_bucket_label(probability_up)
    quant_bucket = _quant_bucket_label(quant_score)
    launch_bucket = _launch_bucket_label(launch_score)
    resonance_bucket = _resonance_bucket_label(market_resonance_score)
    launch_window_bucket = _launch_window_bucket_label(launch_window_score)
    market_state_label = _resolve_market_state_label(row)
    market_stage_proxy = _resolve_market_stage_proxy(row)

    base_result = {
        "probability_up": round(probability_up, 2),
        "attention_score": round(attention_score, 2),
        "enhanced_attention_score": round(enhanced_attention_score, 2),
        "probability_delta_pct": 0.0,
        "attention_delta": 0.0,
        "enhanced_attention_delta": 0.0,
        "replay_calibration_confidence": 0.0,
        "replay_calibration_note": "复盘样本还不够，当前先保持基础模型分数。",
        "replay_calibration_active": False,
        "replay_probability_bucket": probability_bucket,
        "replay_quant_bucket": quant_bucket,
        "replay_launch_bucket": launch_bucket,
        "replay_resonance_bucket": resonance_bucket,
        "replay_launch_window_bucket": launch_window_bucket,
        "replay_launch_window_status": launch_window_status,
        "replay_precision_segment": precision_segment,
        "replay_market_state": market_state_label,
        "replay_market_stage_proxy": market_stage_proxy,
    }
    if not profile:
        return base_result

    review_days = int(profile.get("review_days", 0) or 0)
    review_stocks = int(profile.get("review_stocks", 0) or 0)
    market_replay_days = int(profile.get("market_replay_days", 0) or 0)
    market_replay_rows = int(profile.get("market_replay_rows", 0) or 0)
    market_replay_symbols = int(profile.get("market_replay_symbols", 0) or 0)
    short_ready = review_days >= REPLAY_PROFILE_MIN_REVIEW_DAYS and review_stocks >= REPLAY_PROFILE_MIN_REVIEW_STOCKS
    market_ready = market_replay_days >= MARKET_REPLAY_MIN_DAYS and market_replay_rows >= MARKET_REPLAY_MIN_ROWS
    if not short_ready and not market_ready:
        base_result["replay_calibration_note"] = "短窗口关注榜与长窗口全市场样本都还不够，当前先保留基础模型结果。"
        return base_result

    stage_edges = dict(profile.get("stage_edges", {}) or {})
    precision_edges = dict(profile.get("precision_segment_edges", {}) or {})
    probability_edges = dict(profile.get("probability_bucket_edges", {}) or {})
    quant_edges = dict(profile.get("quant_bucket_edges", {}) or {})
    launch_edges = dict(profile.get("launch_bucket_edges", {}) or {})
    resonance_edges = dict(profile.get("resonance_bucket_edges", {}) or {})
    launch_window_bucket_edges = dict(profile.get("launch_window_bucket_edges", {}) or {})
    launch_window_status_edges = dict(profile.get("launch_window_status_edges", {}) or {})
    stage_supports = dict(profile.get("stage_supports", {}) or {})
    precision_supports = dict(profile.get("precision_segment_supports", {}) or {})
    probability_supports = dict(profile.get("probability_bucket_supports", {}) or {})
    quant_supports = dict(profile.get("quant_bucket_supports", {}) or {})
    launch_supports = dict(profile.get("launch_bucket_supports", {}) or {})
    resonance_supports = dict(profile.get("resonance_bucket_supports", {}) or {})
    launch_window_bucket_supports = dict(profile.get("launch_window_bucket_supports", {}) or {})
    launch_window_status_supports = dict(profile.get("launch_window_status_supports", {}) or {})
    market_state_edges = dict(profile.get("market_state_edges", {}) or {})
    market_state_supports = dict(profile.get("market_state_supports", {}) or {})
    market_stage_edges = dict(profile.get("market_stage_proxy_edges", {}) or {})
    market_stage_supports = dict(profile.get("market_stage_proxy_supports", {}) or {})
    stage_stats = dict(profile.get("stage_stats", {}) or {})
    probability_bucket_stats = dict(profile.get("probability_bucket_stats", {}) or {})
    quant_bucket_stats = dict(profile.get("quant_bucket_stats", {}) or {})
    launch_bucket_stats = dict(profile.get("launch_bucket_stats", {}) or {})
    resonance_bucket_stats = dict(profile.get("resonance_bucket_stats", {}) or {})
    launch_window_bucket_stats = dict(profile.get("launch_window_bucket_stats", {}) or {})
    launch_window_status_stats = dict(profile.get("launch_window_status_stats", {}) or {})
    market_state_stats = dict(profile.get("market_state_stats", {}) or {})
    market_stage_stats = dict(profile.get("market_stage_proxy_stats", {}) or {})

    stage_edge = _safe_float(stage_edges.get(stage_label), 0.0) if short_ready else 0.0
    precision_edge = _safe_float(precision_edges.get(precision_segment), 0.0) if short_ready else 0.0
    probability_edge = _safe_float(probability_edges.get(probability_bucket), 0.0) if short_ready else 0.0
    quant_edge = _safe_float(quant_edges.get(quant_bucket), 0.0) if short_ready else 0.0
    launch_edge = _safe_float(launch_edges.get(launch_bucket), 0.0) if short_ready else 0.0
    resonance_edge = _safe_float(resonance_edges.get(resonance_bucket), 0.0) if short_ready else 0.0
    launch_window_bucket_edge = _safe_float(launch_window_bucket_edges.get(launch_window_bucket), 0.0) if short_ready else 0.0
    launch_window_status_edge = _safe_float(launch_window_status_edges.get(launch_window_status), 0.0) if short_ready else 0.0
    market_state_edge = _safe_float(market_state_edges.get(market_state_label), 0.0) if market_ready else 0.0
    market_stage_edge = _safe_float(market_stage_edges.get(market_stage_proxy), 0.0) if market_ready else 0.0

    short_support_values = [
        _support_scale(int(stage_supports.get(stage_label, 0) or 0), floor=5.0),
        _support_scale(int(precision_supports.get(precision_segment, 0) or 0), floor=5.0),
        _support_scale(int(probability_supports.get(probability_bucket, 0) or 0), floor=5.0),
        _support_scale(int(quant_supports.get(quant_bucket, 0) or 0), floor=5.0),
        _support_scale(int(launch_supports.get(launch_bucket, 0) or 0), floor=5.0),
        _support_scale(int(resonance_supports.get(resonance_bucket, 0) or 0), floor=5.0),
        _support_scale(int(launch_window_bucket_supports.get(launch_window_bucket, 0) or 0), floor=5.0),
        _support_scale(int(launch_window_status_supports.get(launch_window_status, 0) or 0), floor=5.0),
    ]
    short_history_scale = min(review_days / 8.0, 1.0) * 0.45 + min(review_stocks / 180.0, 1.0) * 0.55
    short_support_scale = float(np.mean(short_support_values)) if short_support_values else 0.0
    short_layer_confidence = short_history_scale * 0.58 + short_support_scale * 0.42 if short_ready else 0.0

    market_support_values = [
        _support_scale(int(market_state_supports.get(market_state_label, 0) or 0), floor=260.0),
        _support_scale(int(market_stage_supports.get(market_stage_proxy, 0) or 0), floor=260.0),
    ]
    market_history_scale = min(market_replay_days / 90.0, 1.0) * 0.45 + min(market_replay_rows / 18000.0, 1.0) * 0.55
    market_support_scale = float(np.mean(market_support_values)) if market_support_values else 0.0
    market_layer_confidence = market_history_scale * 0.60 + market_support_scale * 0.40 if market_ready else 0.0

    layer_total = short_layer_confidence + market_layer_confidence
    if layer_total <= 0:
        short_weight = 1.0 if short_ready else 0.0
        market_weight = 1.0 if market_ready and not short_ready else 0.0
    else:
        short_weight = short_layer_confidence / layer_total
        market_weight = market_layer_confidence / layer_total

    calibration_confidence = _clip(
        36.0 + (short_layer_confidence * 0.55 + market_layer_confidence * 0.45) * 54.0,
        36.0,
        94.0,
    )
    confidence_scale = calibration_confidence / 100.0

    def _append_empirical_anchor(
        anchors: list[tuple[float, float]],
        stats_map: dict[str, object],
        key: str,
        metric_key: str,
        *,
        floor: float,
        layer_weight: float,
    ) -> None:
        stats = stats_map.get(key)
        if not isinstance(stats, dict):
            return
        metric_value = _safe_float(stats.get(metric_key), 0.0)
        support = int(stats.get("support", 0) or 0)
        if support <= 0:
            return
        weight = _support_scale(support, floor=floor) * max(layer_weight, 0.0)
        if weight <= 0:
            return
        anchors.append((metric_value, weight))

    probability_anchors: list[tuple[float, float]] = []
    upside_anchors: list[tuple[float, float]] = []
    intraday_upside_anchors: list[tuple[float, float]] = []
    hit_rate_anchors: list[tuple[float, float]] = []

    _append_empirical_anchor(probability_anchors, stage_stats, stage_label, "win_rate_pct", floor=5.0, layer_weight=short_weight)
    _append_empirical_anchor(probability_anchors, probability_bucket_stats, probability_bucket, "win_rate_pct", floor=5.0, layer_weight=short_weight)
    _append_empirical_anchor(probability_anchors, quant_bucket_stats, quant_bucket, "win_rate_pct", floor=5.0, layer_weight=short_weight * 0.75)
    _append_empirical_anchor(probability_anchors, launch_bucket_stats, launch_bucket, "win_rate_pct", floor=5.0, layer_weight=short_weight * 0.85)
    _append_empirical_anchor(probability_anchors, resonance_bucket_stats, resonance_bucket, "win_rate_pct", floor=5.0, layer_weight=short_weight * 0.85)
    _append_empirical_anchor(probability_anchors, launch_window_bucket_stats, launch_window_bucket, "win_rate_pct", floor=5.0, layer_weight=short_weight)
    _append_empirical_anchor(probability_anchors, launch_window_status_stats, launch_window_status, "win_rate_pct", floor=5.0, layer_weight=short_weight)
    _append_empirical_anchor(probability_anchors, market_state_stats, market_state_label, "win_rate_pct", floor=260.0, layer_weight=market_weight)
    _append_empirical_anchor(probability_anchors, market_stage_stats, market_stage_proxy, "win_rate_pct", floor=260.0, layer_weight=market_weight)

    _append_empirical_anchor(upside_anchors, stage_stats, stage_label, "avg_return_pct", floor=5.0, layer_weight=short_weight)
    _append_empirical_anchor(upside_anchors, probability_bucket_stats, probability_bucket, "avg_return_pct", floor=5.0, layer_weight=short_weight)
    _append_empirical_anchor(upside_anchors, launch_bucket_stats, launch_bucket, "avg_return_pct", floor=5.0, layer_weight=short_weight * 0.85)
    _append_empirical_anchor(upside_anchors, resonance_bucket_stats, resonance_bucket, "avg_return_pct", floor=5.0, layer_weight=short_weight * 0.85)
    _append_empirical_anchor(upside_anchors, launch_window_bucket_stats, launch_window_bucket, "avg_return_pct", floor=5.0, layer_weight=short_weight)
    _append_empirical_anchor(upside_anchors, launch_window_status_stats, launch_window_status, "avg_return_pct", floor=5.0, layer_weight=short_weight)
    _append_empirical_anchor(upside_anchors, market_state_stats, market_state_label, "avg_return_pct", floor=260.0, layer_weight=market_weight)
    _append_empirical_anchor(upside_anchors, market_stage_stats, market_stage_proxy, "avg_return_pct", floor=260.0, layer_weight=market_weight)

    _append_empirical_anchor(intraday_upside_anchors, probability_bucket_stats, probability_bucket, "intraday_high_return_pct", floor=5.0, layer_weight=short_weight)
    _append_empirical_anchor(intraday_upside_anchors, launch_window_bucket_stats, launch_window_bucket, "intraday_high_return_pct", floor=5.0, layer_weight=short_weight)
    _append_empirical_anchor(intraday_upside_anchors, launch_window_status_stats, launch_window_status, "intraday_high_return_pct", floor=5.0, layer_weight=short_weight)
    _append_empirical_anchor(intraday_upside_anchors, market_state_stats, market_state_label, "intraday_high_return_pct", floor=260.0, layer_weight=market_weight)
    _append_empirical_anchor(intraday_upside_anchors, market_stage_stats, market_stage_proxy, "intraday_high_return_pct", floor=260.0, layer_weight=market_weight)

    _append_empirical_anchor(hit_rate_anchors, probability_bucket_stats, probability_bucket, "hit_rate_pct", floor=5.0, layer_weight=short_weight)
    _append_empirical_anchor(hit_rate_anchors, launch_window_status_stats, launch_window_status, "hit_rate_pct", floor=5.0, layer_weight=short_weight)

    empirical_probability_pct = (
        _weighted_average(
            pd.Series([value for value, _ in probability_anchors], dtype=float),
            pd.Series([weight for _, weight in probability_anchors], dtype=float),
            probability_up,
        )
        if probability_anchors
        else probability_up
    )
    empirical_upside_pct = (
        _weighted_average(
            pd.Series([value for value, _ in upside_anchors], dtype=float),
            pd.Series([weight for _, weight in upside_anchors], dtype=float),
            0.0,
        )
        if upside_anchors
        else 0.0
    )
    empirical_intraday_upside_pct = (
        _weighted_average(
            pd.Series([value for value, _ in intraday_upside_anchors], dtype=float),
            pd.Series([weight for _, weight in intraday_upside_anchors], dtype=float),
            empirical_upside_pct,
        )
        if intraday_upside_anchors
        else empirical_upside_pct
    )
    empirical_hit_rate_pct = (
        _weighted_average(
            pd.Series([value for value, _ in hit_rate_anchors], dtype=float),
            pd.Series([weight for _, weight in hit_rate_anchors], dtype=float),
            0.0,
        )
        if hit_rate_anchors
        else 0.0
    )

    short_probability_delta = (
        stage_edge * 18.0
        + precision_edge * 16.0
        + probability_edge * 22.0
        + quant_edge * 14.0
        + launch_edge * 16.0
        + resonance_edge * 18.0
        + launch_window_bucket_edge * 18.0
        + launch_window_status_edge * 20.0
    )
    short_attention_delta = (
        stage_edge * 10.0
        + precision_edge * 8.0
        + probability_edge * 6.0
        + quant_edge * 16.0
        + launch_edge * 14.0
        + resonance_edge * 12.0
        + launch_window_bucket_edge * 14.0
        + launch_window_status_edge * 15.0
    )
    short_enhanced_delta = (
        stage_edge * 8.0
        + precision_edge * 10.0
        + probability_edge * 8.0
        + quant_edge * 14.0
        + launch_edge * 12.0
        + resonance_edge * 14.0
        + launch_window_bucket_edge * 13.0
        + launch_window_status_edge * 14.0
    )
    market_probability_delta = market_state_edge * 12.0 + market_stage_edge * 14.0
    market_attention_delta = market_state_edge * 7.0 + market_stage_edge * 10.0
    market_enhanced_delta = market_state_edge * 8.0 + market_stage_edge * 11.0

    probability_delta = _clip(
        (short_probability_delta * short_weight + market_probability_delta * market_weight) * confidence_scale,
        -7.5,
        7.5,
    )
    attention_delta = _clip(
        (short_attention_delta * short_weight + market_attention_delta * market_weight) * confidence_scale,
        -5.5,
        5.5,
    )
    enhanced_delta = _clip(
        (short_enhanced_delta * short_weight + market_enhanced_delta * market_weight) * confidence_scale,
        -6.5,
        6.5,
    )

    if stage_label == "高位分歧派发" and probability_up >= 80:
        probability_delta = _clip(probability_delta - 2.0, -8.5, 7.5)
        enhanced_delta = _clip(enhanced_delta - 1.4, -7.0, 6.5)
    if stage_label == "趋势主升加速" and quant_score >= 70:
        probability_delta = _clip(probability_delta + 1.2, -7.5, 8.5)
        attention_delta = _clip(attention_delta + 0.8, -5.5, 6.0)
    if precision_segment == "precision_active" and quant_score >= 70:
        probability_delta = _clip(probability_delta + 0.8, -7.5, 8.5)
    if precision_segment == "precision_history" and probability_up < 70:
        probability_delta = _clip(probability_delta - 0.8, -8.0, 8.0)

    calibrated_probability = _clip(probability_up + probability_delta, 0.0, 99.99)
    probability_anchor_blend = 0.14 + confidence_scale * 0.20
    if probability_anchors:
        calibrated_probability = _clip(
            calibrated_probability * (1.0 - probability_anchor_blend)
            + empirical_probability_pct * probability_anchor_blend,
            0.0,
            99.99,
        )
    calibrated_attention = _clip(attention_score + attention_delta, 0.0, 100.0)
    calibrated_enhanced = _clip(enhanced_attention_score + enhanced_delta, 0.0, 100.0)
    probability_delta = calibrated_probability - probability_up

    base_upside_pct = _safe_float(
        row.get("predicted_upside_pct"),
        max((probability_up - 40.0) * 0.18, 0.0),
    )
    base_low_pct = _safe_float(row.get("predicted_upside_low_pct"), max(base_upside_pct * 0.72, 0.0))
    base_high_pct = _safe_float(row.get("predicted_upside_high_pct"), max(base_upside_pct * 1.22, base_upside_pct))
    empirical_move_anchor_pct = max(
        empirical_upside_pct,
        empirical_intraday_upside_pct * 0.74,
        empirical_hit_rate_pct * 0.045,
    )
    if empirical_move_anchor_pct <= 0:
        empirical_move_anchor_pct = max((empirical_probability_pct - 42.0) * 0.17, 0.0)
    upside_anchor_blend = 0.16 + confidence_scale * 0.18
    probability_ratio = (
        calibrated_probability / max(probability_up, 1e-6)
        if probability_up > 0
        else max(calibrated_probability / 50.0, 0.4)
    )
    calibrated_upside_pct = base_upside_pct
    if base_upside_pct <= 0:
        calibrated_upside_pct = empirical_move_anchor_pct
    else:
        calibrated_upside_pct = (
            base_upside_pct * (1.0 - upside_anchor_blend)
            + empirical_move_anchor_pct * upside_anchor_blend
        )
    calibrated_upside_pct = _clip(calibrated_upside_pct * (0.82 + probability_ratio * 0.18), 0.2, 30.0)
    calibrated_low_pct = _clip(
        max(base_low_pct * (0.86 + probability_ratio * 0.14), calibrated_upside_pct * 0.70, empirical_upside_pct * 0.92),
        0.1,
        calibrated_upside_pct,
    )
    calibrated_high_pct = _clip(
        max(base_high_pct * (0.80 + probability_ratio * 0.20), calibrated_upside_pct * 1.18, empirical_intraday_upside_pct * 1.04),
        calibrated_upside_pct,
        45.0,
    )

    contributors = {
        "短窗K线阶段": stage_edge * short_weight,
        "短窗精度状态": precision_edge * short_weight,
        "短窗概率分层": probability_edge * short_weight,
        "短窗量化强弱": quant_edge * short_weight,
        "长窗市场状态": market_state_edge * market_weight,
        "长窗结构阶段": market_stage_edge * market_weight,
    }
    contributors["short_launch_bucket"] = launch_edge * short_weight
    contributors["short_resonance_bucket"] = resonance_edge * short_weight
    contributors["short_launch_window_bucket"] = launch_window_bucket_edge * short_weight
    contributors["short_launch_window_status"] = launch_window_status_edge * short_weight
    strongest_name, strongest_value = max(contributors.items(), key=lambda item: abs(item[1]))
    contributor_text = "正向影响" if strongest_value > 0 else "负向影响" if strongest_value < 0 else "中性影响"
    layer_parts: list[str] = []
    if short_ready:
        layer_parts.append(f"短窗关注榜 {review_days} 日/{review_stocks} 股")
    if market_ready:
        layer_parts.append(f"长窗全市场 {market_replay_days} 日/{market_replay_symbols} 股/{market_replay_rows} 条")
    note = (
        f"基于{' + '.join(layer_parts)}进行回放校准，"
        f"当前主要由 {strongest_name} 提供{contributor_text}，最终概率调整 {probability_delta:+.2f}pct。"
    )
    return {
        "probability_up": round(calibrated_probability, 2),
        "attention_score": round(calibrated_attention, 2),
        "enhanced_attention_score": round(calibrated_enhanced, 2),
        "probability_delta_pct": round(probability_delta, 2),
        "attention_delta": round(attention_delta, 2),
        "enhanced_attention_delta": round(enhanced_delta, 2),
        "predicted_upside_pct": round(calibrated_upside_pct, 2),
        "predicted_upside_low_pct": round(calibrated_low_pct, 2),
        "predicted_upside_high_pct": round(calibrated_high_pct, 2),
        "replay_empirical_probability_pct": round(empirical_probability_pct, 2),
        "replay_empirical_upside_pct": round(empirical_upside_pct, 2),
        "replay_empirical_intraday_upside_pct": round(empirical_intraday_upside_pct, 2),
        "replay_empirical_hit_rate_pct": round(empirical_hit_rate_pct, 2),
        "replay_calibration_confidence": round(calibration_confidence, 2),
        "replay_calibration_note": note,
        "replay_calibration_active": bool(
            abs(probability_delta) >= 0.01 or abs(attention_delta) >= 0.01 or abs(enhanced_delta) >= 0.01
        ),
        "replay_probability_bucket": probability_bucket,
        "replay_quant_bucket": quant_bucket,
        "replay_launch_bucket": launch_bucket,
        "replay_resonance_bucket": resonance_bucket,
        "replay_launch_window_bucket": launch_window_bucket,
        "replay_launch_window_status": launch_window_status,
        "replay_precision_segment": precision_segment,
        "replay_market_state": market_state_label,
        "replay_market_stage_proxy": market_stage_proxy,
    }


def load_latest_review_summary(
    *,
    horizon_days: int,
    positive_return: float,
    ranking_by: str,
    board_size: int,
) -> dict[str, object] | None:
    payloads = _matching_review_payloads(
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
        board_size=board_size,
    )
    if not payloads:
        return None
    payload = sorted(
        payloads,
        key=lambda item: (
            str(item.get("meta", {}).get("review_date") or ""),
            str(item.get("meta", {}).get("board_date") or ""),
            int(item.get("meta", {}).get("cache_version") or -1),
        ),
    )[-1]
    return payload["summary"]


def load_latest_review_details(
    *,
    horizon_days: int,
    positive_return: float,
    ranking_by: str,
    board_size: int,
) -> pd.DataFrame:
    payloads = _matching_review_payloads(
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
        board_size=board_size,
    )
    if not payloads:
        return pd.DataFrame()
    payload = sorted(
        payloads,
        key=lambda item: (
            str(item.get("meta", {}).get("review_date") or ""),
            str(item.get("meta", {}).get("board_date") or ""),
            int(item.get("meta", {}).get("cache_version") or -1),
        ),
    )[-1]
    details = payload.get("details")
    if not isinstance(details, pd.DataFrame):
        return pd.DataFrame()
    return details.copy()


def load_latest_review_bundle(
    *,
    horizon_days: int,
    positive_return: float,
    ranking_by: str,
    board_size: int,
) -> dict[str, object]:
    review_payloads = _matching_review_payloads(
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
        board_size=board_size,
    )
    if not review_payloads:
        return {
            "summary": None,
            "details": pd.DataFrame(),
            "review_meta": {},
            "snapshot_board": pd.DataFrame(),
            "snapshot_meta": {},
        }
    review_payload = sorted(
        review_payloads,
        key=lambda item: (
            str(item.get("meta", {}).get("review_date") or ""),
            str(item.get("meta", {}).get("board_date") or ""),
            int(item.get("meta", {}).get("cache_version") or -1),
        ),
    )[-1]
    review_meta = dict(review_payload.get("meta", {}))
    review_board_date = str(review_meta.get("board_date") or "")
    summary = dict(review_payload.get("summary", {}))
    details = review_payload.get("details")
    if not isinstance(details, pd.DataFrame):
        details = pd.DataFrame()
    else:
        details = details.copy()

    snapshot_payloads = _matching_snapshot_payloads(
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
        board_size=board_size,
    )
    linked_snapshot_candidates = [
        item for item in snapshot_payloads if str(item.get("meta", {}).get("board_date") or "") == review_board_date
    ]
    snapshot_meta: dict[str, object] = {}
    snapshot_board = pd.DataFrame()
    if linked_snapshot_candidates:
        snapshot_payload = sorted(
            linked_snapshot_candidates,
            key=lambda item: (
                int(item.get("meta", {}).get("cache_version") or -1),
                str(item.get("meta", {}).get("captured_at") or ""),
            ),
        )[-1]
        snapshot_meta = dict(snapshot_payload.get("meta", {}))
        board = snapshot_payload.get("board")
        if isinstance(board, pd.DataFrame):
            snapshot_board = board.copy()

    return {
        "summary": summary or None,
        "details": details,
        "review_meta": review_meta,
        "snapshot_board": snapshot_board,
        "snapshot_meta": snapshot_meta,
    }


def _weighted_recent_review_details(
    review_payloads: list[dict[str, object]],
    *,
    rolling_review_days: int | None = None,
) -> pd.DataFrame:
    if not review_payloads:
        return pd.DataFrame()

    rolling_limit = _normalize_rolling_review_days(rolling_review_days)
    recent_payloads = review_payloads[-rolling_limit:]
    detail_frames: list[pd.DataFrame] = []
    payload_count = max(len(recent_payloads), 1)
    for index, payload in enumerate(recent_payloads):
        details = payload.get("details")
        if not isinstance(details, pd.DataFrame) or details.empty:
            continue
        weighted = details.copy()
        recency_weight = 0.58 + 0.42 * ((index + 1) / payload_count)
        if "rank" in weighted.columns:
            rank_series = pd.to_numeric(weighted["rank"], errors="coerce").astype(float)
            rank_denom = max(float(rank_series.max()), 1.0)
            rank_weight = (1.16 - ((rank_series - 1.0) / rank_denom).clip(lower=0.0, upper=1.0) * 0.30).fillna(1.0)
        else:
            rank_weight = pd.Series(1.0, index=weighted.index, dtype=float)
        weighted["sample_weight"] = recency_weight * rank_weight.astype(float)
        detail_frames.append(weighted)
    if not detail_frames:
        return pd.DataFrame()

    merged = pd.concat(detail_frames, ignore_index=True)
    merged["candidate_strategy"] = merged.get("candidate_strategy", pd.Series("", index=merged.index)).fillna("").astype(str)
    merged["candidate_strategy_label"] = [
        _strategy_display_label(code, label)
        for code, label in zip(
            merged["candidate_strategy"].tolist(),
            merged.get("candidate_strategy_label", pd.Series("", index=merged.index)).fillna("").astype(str).tolist(),
        )
    ]
    merged["candidate_strategy_short_label"] = (
        merged.get("candidate_strategy_short_label", merged["candidate_strategy_label"])
        .fillna("")
        .astype(str)
        .replace("", pd.NA)
        .fillna(merged["candidate_strategy_label"])
    )
    merged["market_state_label"] = merged.get("market_state_label", pd.Series("unknown", index=merged.index)).fillna("unknown").astype(str)
    merged["market_state_display"] = merged["market_state_label"].apply(_market_state_display_label)
    merged["market_stage_proxy"] = merged.get("market_stage_proxy", pd.Series("unknown", index=merged.index)).fillna("unknown").astype(str)
    merged["market_stage_display"] = merged["market_stage_proxy"].apply(_market_stage_proxy_display_label)
    return merged


def _aggregate_review_performance(
    details: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    if details.empty:
        return pd.DataFrame()

    working = details.copy()
    for column in group_columns:
        if column not in working.columns:
            working[column] = "未知"
        working[column] = working[column].fillna("未知").astype(str)
    sample_weight = (
        pd.to_numeric(working["sample_weight"], errors="coerce").astype(float).fillna(1.0)
        if "sample_weight" in working.columns
        else pd.Series(1.0, index=working.index, dtype=float)
    )
    rows: list[dict[str, object]] = []
    for group_key, group in working.groupby(group_columns, dropna=False):
        keys = list(group_key) if isinstance(group_key, tuple) else [group_key]
        group_weight = sample_weight.reindex(group.index).fillna(1.0)
        avg_probability_pct = _weighted_average(group["probability_up"], group_weight, 0.0)
        avg_return = _weighted_average(group["next_day_return"], group_weight, 0.0)
        intraday_high_return = _weighted_average(group["intraday_high_return"], group_weight, avg_return)
        win_rate = _weighted_average(group["win"], group_weight, 0.0)
        target_hit_rate = _weighted_average(group["hit_target"], group_weight, 0.0)
        direction_hit_rate = _weighted_average(group["direction_hit"], group_weight, 0.0)
        avg_rank = _weighted_average(group["rank"], group_weight, 0.0) if "rank" in group.columns else 0.0
        row = {
            "sample_count": int(len(group)),
            "avg_rank": round(avg_rank, 2),
            "avg_probability_pct": round(avg_probability_pct, 2),
            "avg_return_pct": round(avg_return * 100, 2),
            "intraday_high_return_pct": round(intraday_high_return * 100, 2),
            "win_rate_pct": round(win_rate * 100, 2),
            "target_hit_rate_pct": round(target_hit_rate * 100, 2),
            "direction_hit_rate_pct": round(direction_hit_rate * 100, 2),
            "calibration_gap_pct": round(abs(avg_probability_pct - win_rate * 100), 2),
        }
        for column, value in zip(group_columns, keys):
            row[column] = str(value)
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    sort_columns = [
        column
        for column in ["win_rate_pct", "avg_return_pct", "target_hit_rate_pct", "sample_count"]
        if column in frame.columns
    ]
    return frame.sort_values(sort_columns, ascending=[False] * len(sort_columns)).reset_index(drop=True)


def _market_state_replay_panel(profile: dict[str, object]) -> pd.DataFrame:
    stats_map = dict(profile.get("market_state_stats", {}) or {})
    edges_map = dict(profile.get("market_state_edges", {}) or {})
    if not stats_map:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for state_key, stats in stats_map.items():
        if not isinstance(stats, dict):
            continue
        rows.append(
            {
                "market_state_label": str(state_key),
                "market_state_display": _market_state_display_label(str(state_key)),
                "sample_count": int(stats.get("support", 0) or 0),
                "avg_return_pct": round(_safe_float(stats.get("avg_return_pct"), 0.0), 2),
                "intraday_high_return_pct": round(_safe_float(stats.get("intraday_high_return_pct"), 0.0), 2),
                "win_rate_pct": round(_safe_float(stats.get("win_rate_pct"), 0.0), 2),
                "target_hit_rate_pct": round(_safe_float(stats.get("hit_rate_pct"), 0.0), 2),
                "state_edge": round(_safe_float(edges_map.get(state_key), 0.0), 4),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["win_rate_pct", "avg_return_pct", "sample_count"], ascending=False).reset_index(drop=True)


def load_review_battle_panels(
    *,
    horizon_days: int,
    positive_return: float,
    ranking_by: str,
    board_size: int,
) -> dict[str, object]:
    review_payloads = _matching_review_payloads(
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
        board_size=board_size,
    )
    short_details = _weighted_recent_review_details(review_payloads)
    strategy_panel = _aggregate_review_performance(short_details, ["candidate_strategy_label"])
    short_market_state_panel = _aggregate_review_performance(short_details, ["market_state_display"])
    combo_panel = _aggregate_review_performance(short_details, ["market_state_display", "candidate_strategy_label"])
    market_profile = load_market_replay_profile(
        horizon_days=horizon_days,
        positive_return=positive_return,
    )
    long_market_state_panel = _market_state_replay_panel(market_profile)

    best_strategy = (
        str(strategy_panel.iloc[0]["candidate_strategy_label"])
        if not strategy_panel.empty and "candidate_strategy_label" in strategy_panel.columns
        else ""
    )
    best_market_state = (
        str(short_market_state_panel.iloc[0]["market_state_display"])
        if not short_market_state_panel.empty and "market_state_display" in short_market_state_panel.columns
        else str(long_market_state_panel.iloc[0]["market_state_display"])
        if not long_market_state_panel.empty and "market_state_display" in long_market_state_panel.columns
        else ""
    )
    weakest_strategy = (
        str(strategy_panel.iloc[-1]["candidate_strategy_label"])
        if not strategy_panel.empty and "candidate_strategy_label" in strategy_panel.columns
        else ""
    )
    strategy_effectiveness_summary = (
        f"最近复盘中 {best_strategy} 更有效，{weakest_strategy} 相对较弱。"
        if best_strategy and weakest_strategy and best_strategy != weakest_strategy
        else f"最近复盘中 {best_strategy} 暂时领先。"
        if best_strategy
        else "最近暂无足够分策略复盘样本。"
    )
    return {
        "strategy_panel": strategy_panel,
        "short_market_state_panel": short_market_state_panel,
        "long_market_state_panel": long_market_state_panel,
        "combo_panel": combo_panel,
        "meta": {
            "review_days": int(min(len(review_payloads), MAX_PROFILE_REVIEW_DAYS)),
            "review_rows": int(len(short_details)) if not short_details.empty else 0,
            "review_symbols": int(short_details["symbol"].nunique()) if not short_details.empty and "symbol" in short_details.columns else 0,
            "market_replay_days": int(market_profile.get("market_replay_days", 0) or 0),
            "market_replay_rows": int(market_profile.get("market_replay_rows", 0) or 0),
            "market_replay_symbols": int(market_profile.get("market_replay_symbols", 0) or 0),
            "best_strategy": best_strategy,
            "weakest_strategy": weakest_strategy,
            "strategy_effectiveness_summary": strategy_effectiveness_summary,
            "best_market_state": best_market_state,
        },
    }


def load_latest_snapshot_board(
    *,
    horizon_days: int,
    positive_return: float,
    ranking_by: str,
    board_size: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    payloads = _matching_snapshot_payloads(
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
        board_size=board_size,
    )
    if not payloads:
        return pd.DataFrame(), {}
    payload = sorted(
        payloads,
        key=lambda item: (
            str(item.get("meta", {}).get("board_date") or ""),
            int(item.get("meta", {}).get("cache_version") or -1),
            str(item.get("meta", {}).get("captured_at") or ""),
        ),
    )[-1]
    board = payload.get("board")
    meta = dict(payload.get("meta", {}))
    if not isinstance(board, pd.DataFrame):
        return pd.DataFrame(), {}
    return board.copy(), meta


def compute_adaptive_rank_score(board_df: pd.DataFrame, profile: dict[str, object] | None) -> pd.Series:
    if board_df.empty:
        return pd.Series(dtype=float)

    weights = dict(DEFAULT_PROFILE_WEIGHTS)
    if profile and isinstance(profile.get("weights"), dict):
        for column, value in profile["weights"].items():
            if column in weights:
                weights[column] = float(value)

    enhanced = board_df["enhanced_attention_score"] if "enhanced_attention_score" in board_df.columns else board_df["attention_score"]
    quant_score = (
        board_df["quant_score"]
        if "quant_score" in board_df.columns
        else pd.Series(0.0, index=board_df.index, dtype=float)
    )
    launch_score = (
        board_df["launch_score"]
        if "launch_score" in board_df.columns
        else pd.Series(50.0, index=board_df.index, dtype=float)
    )
    market_resonance_score = (
        board_df["market_resonance_score"]
        if "market_resonance_score" in board_df.columns
        else pd.Series(50.0, index=board_df.index, dtype=float)
    )
    intraday_sector_sync_score = (
        board_df["intraday_sector_sync_score"]
        if "intraday_sector_sync_score" in board_df.columns
        else pd.Series(50.0, index=board_df.index, dtype=float)
    )
    launch_specialist_score = (
        board_df["launch_specialist_score"]
        if "launch_specialist_score" in board_df.columns
        else pd.Series(50.0, index=board_df.index, dtype=float)
    )
    launch_regime_fit_score = (
        board_df["launch_regime_fit_score"]
        if "launch_regime_fit_score" in board_df.columns
        else pd.Series(50.0, index=board_df.index, dtype=float)
    )
    launch_window_score = (
        board_df["launch_window_score"]
        if "launch_window_score" in board_df.columns
        else pd.Series(50.0, index=board_df.index, dtype=float)
    )
    long_setup_quality = (
        board_df["long_setup_quality"]
        if "long_setup_quality" in board_df.columns
        else pd.Series(50.0, index=board_df.index, dtype=float)
    )
    score = (
        pd.to_numeric(board_df["attention_score"], errors="coerce").fillna(0.0) * weights["attention_score"]
        + pd.to_numeric(board_df["probability_up"], errors="coerce").fillna(0.0) * weights["probability_up"]
        + pd.to_numeric(enhanced, errors="coerce").fillna(0.0) * weights["enhanced_attention_score"]
        + pd.to_numeric(quant_score, errors="coerce").fillna(0.0) * weights["quant_score"]
        + pd.to_numeric(launch_score, errors="coerce").fillna(50.0) * weights["launch_score"]
        + pd.to_numeric(market_resonance_score, errors="coerce").fillna(50.0) * weights["market_resonance_score"]
        + pd.to_numeric(intraday_sector_sync_score, errors="coerce").fillna(50.0) * weights["intraday_sector_sync_score"]
        + pd.to_numeric(launch_specialist_score, errors="coerce").fillna(50.0) * weights["launch_specialist_score"]
        + pd.to_numeric(launch_regime_fit_score, errors="coerce").fillna(50.0) * weights["launch_regime_fit_score"]
        + pd.to_numeric(launch_window_score, errors="coerce").fillna(50.0) * weights["launch_window_score"]
        + pd.to_numeric(long_setup_quality, errors="coerce").fillna(50.0) * 0.05
    )
    risk_of_late_entry = (
        pd.to_numeric(board_df["risk_of_late_entry"], errors="coerce").fillna(50.0)
        if "risk_of_late_entry" in board_df.columns
        else pd.Series(50.0, index=board_df.index, dtype=float)
    )
    launch_phase = (
        board_df["launch_phase_label"].fillna("").astype(str)
        if "launch_phase_label" in board_df.columns
        else pd.Series("", index=board_df.index, dtype=str)
    )
    crowding_risk = (
        pd.to_numeric(board_df["crowding_risk"], errors="coerce").fillna(50.0)
        if "crowding_risk" in board_df.columns
        else pd.Series(50.0, index=board_df.index, dtype=float)
    )
    risk_penalty = (risk_of_late_entry - 60.0).clip(lower=0.0, upper=40.0) * 0.08
    risk_penalty = risk_penalty + (crowding_risk - 55.0).clip(lower=0.0, upper=45.0) * 0.06
    risk_penalty = risk_penalty + launch_phase.isin(["已走远", "伪突破", "量化拥挤", "crowded"]).astype(float) * 2.0
    return (score - risk_penalty).round(4)


def _bucket_probability(value: object) -> str:
    probability = _safe_float(value, 0.0)
    for lower, upper, label in REPLAY_PROBABILITY_BUCKETS:
        if lower <= probability < upper:
            return label
    return "unknown"


def _summarize_lightweight_group(details: pd.DataFrame, group_columns: list[str], base_win_rate: float) -> pd.DataFrame:
    if details.empty or not group_columns:
        return pd.DataFrame()
    available = [column for column in group_columns if column in details.columns]
    if len(available) != len(group_columns):
        return pd.DataFrame()
    grouped = (
        details.groupby(available, dropna=False)
        .agg(
            sample_count=("symbol", "count"),
            win_rate=("direction_hit", "mean"),
            target_hit_rate=("hit_target", "mean"),
            avg_return=("next_day_return", "mean"),
            avg_target_progress=("target_progress_pct", "mean"),
        )
        .reset_index()
    )
    if grouped.empty:
        return grouped
    grouped["edge_vs_base"] = grouped["win_rate"].fillna(0.0) - float(base_win_rate)
    grouped["win_rate_pct"] = (grouped["win_rate"].fillna(0.0) * 100).round(2)
    grouped["target_hit_rate_pct"] = (grouped["target_hit_rate"].fillna(0.0) * 100).round(2)
    grouped["avg_return_pct"] = (grouped["avg_return"].fillna(0.0) * 100).round(2)
    grouped["edge_vs_base_pct"] = (grouped["edge_vs_base"].fillna(0.0) * 100).round(2)
    grouped["avg_target_progress_pct"] = grouped["avg_target_progress"].fillna(0.0).round(2)
    return grouped.sort_values(["edge_vs_base", "avg_return", "sample_count"], ascending=False).reset_index(drop=True)


def build_daily_lightweight_backtest_model(
    *,
    horizon_days: int,
    positive_return: float,
    ranking_by: str,
    board_size: int,
    rolling_review_days: int | None = None,
    persist: bool = True,
) -> dict[str, object]:
    review_payloads = _matching_review_payloads(
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
        board_size=board_size,
    )
    details = _weighted_recent_review_details(review_payloads, rolling_review_days=rolling_review_days)
    if details.empty:
        model = {
            "model_version": DAILY_LIGHTWEIGHT_MODEL_VERSION,
            "status": "insufficient_samples",
            "review_days": 0,
            "sample_count": 0,
            "base_win_rate": 0.0,
            "base_avg_return": 0.0,
            "panels": {},
            "model_parameter_update_allowed": False,
        }
    else:
        frame = details.copy()
        if "direction_hit" not in frame.columns:
            frame["direction_hit"] = pd.to_numeric(frame.get("win", 0.0), errors="coerce").fillna(0.0)
        if "hit_target" not in frame.columns:
            frame["hit_target"] = 0.0
        if "next_day_return" not in frame.columns:
            frame["next_day_return"] = 0.0
        if "target_progress_pct" not in frame.columns:
            frame["target_progress_pct"] = 0.0
        if "probability_bucket" not in frame.columns:
            frame["probability_bucket"] = frame.get("probability_up", pd.Series(0.0, index=frame.index)).map(_bucket_probability)
        if "candidate_strategy_label" not in frame.columns:
            frame["candidate_strategy_label"] = frame.get("candidate_strategy", pd.Series("", index=frame.index)).map(_strategy_display_label)
        if "market_state_display" not in frame.columns:
            frame["market_state_display"] = frame.get("market_state_label", pd.Series("unknown", index=frame.index)).map(_market_state_display_label)
        base_win_rate = float(pd.to_numeric(frame["direction_hit"], errors="coerce").fillna(0.0).mean())
        base_avg_return = float(pd.to_numeric(frame["next_day_return"], errors="coerce").fillna(0.0).mean())
        panels = {
            "probability_bucket": _summarize_lightweight_group(frame, ["probability_bucket"], base_win_rate),
            "strategy": _summarize_lightweight_group(frame, ["candidate_strategy_label"], base_win_rate),
            "market_state": _summarize_lightweight_group(frame, ["market_state_display"], base_win_rate),
            "strategy_market_state": _summarize_lightweight_group(
                frame,
                ["candidate_strategy_label", "market_state_display"],
                base_win_rate,
            ),
            "launch_phase": _summarize_lightweight_group(frame, ["launch_phase_label"], base_win_rate),
        }
        best_panel = panels["strategy_market_state"]
        model = {
            "model_version": DAILY_LIGHTWEIGHT_MODEL_VERSION,
            "status": "ready" if len(frame) >= 6 else "limited_samples",
            "review_days": int(min(len(review_payloads), _normalize_rolling_review_days(rolling_review_days))),
            "sample_count": int(len(frame)),
            "base_win_rate": round(base_win_rate, 4),
            "base_avg_return": round(base_avg_return, 6),
            "best_context": (
                best_panel.iloc[0].to_dict()
                if isinstance(best_panel, pd.DataFrame) and not best_panel.empty
                else {}
            ),
            "panels": panels,
            "model_parameter_update_allowed": False,
            "calibration_scope": "independent_daily_lightweight_backtest_only",
        }
    if persist:
        with _daily_lightweight_model_cache_path(horizon_days, positive_return, ranking_by, board_size).open("wb") as handle:
            pickle.dump(
                {
                    "cache_version": DAILY_LIGHTWEIGHT_MODEL_VERSION,
                    "meta": {
                        "horizon_days": int(horizon_days),
                        "positive_return": float(positive_return),
                        "ranking_by": ranking_by,
                        "board_size": int(board_size),
                        "rolling_review_days": int(_normalize_rolling_review_days(rolling_review_days)),
                        "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    "model": model,
                },
                handle,
            )
    return model


def load_daily_lightweight_backtest_model(
    *,
    horizon_days: int,
    positive_return: float,
    ranking_by: str,
    board_size: int,
) -> dict[str, object]:
    path = _daily_lightweight_model_cache_path(horizon_days, positive_return, ranking_by, board_size)
    if not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return {}
    if int(payload.get("cache_version", -1) or -1) != DAILY_LIGHTWEIGHT_MODEL_VERSION:
        return {}
    model = payload.get("model")
    return dict(model) if isinstance(model, dict) else {}


def run_daily_review_maintenance(
    board: pd.DataFrame,
    *,
    horizon_days: int,
    positive_return: float,
    ranking_by: str,
    board_size: int,
    rolling_review_days: int | None = None,
) -> dict[str, object]:
    rolling_limit = _normalize_rolling_review_days(rolling_review_days)
    board_date = str(board.attrs.get("market_data_date") or "")
    existing_snapshot_path = (
        _snapshot_cache_path(horizon_days, positive_return, ranking_by, board_size, board_date)
        if board_date
        else None
    )
    snapshot_existed = bool(existing_snapshot_path and existing_snapshot_path.exists())
    snapshot_path = persist_focus_board_snapshot(
        board,
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
        board_size=board_size,
    )

    latest_market_data_date = str(board.attrs.get("latest_market_data_date") or board.attrs.get("market_data_date") or "")
    if not latest_market_data_date:
        latest_market_data_date = str(board.attrs.get("market_data_date") or "")

    completed_reviews = 0
    new_reviews = 0
    latest_summary: dict[str, object] | None = None

    matching_snapshots = _matching_snapshot_payloads(
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
        board_size=board_size,
    )
    eligible_snapshots: list[dict[str, object]] = []
    for payload in matching_snapshots:
        snapshot_meta = payload["meta"]
        snapshot_board_date = str(snapshot_meta.get("board_date") or "")
        if not snapshot_board_date or not latest_market_data_date or snapshot_board_date >= latest_market_data_date:
            continue
        eligible_snapshots.append(payload)

    for payload in eligible_snapshots[-rolling_limit:]:
        snapshot_board = payload["board"]
        snapshot_meta = payload["meta"]
        snapshot_board_date = str(snapshot_meta.get("board_date") or "")
        review_date = _next_review_date(snapshot_board_date, latest_market_data_date)
        if not review_date:
            continue
        review_path = _review_cache_path(
            horizon_days,
            positive_return,
            ranking_by,
            board_size,
            snapshot_board_date,
            review_date,
        )
        if review_path.exists():
            completed_reviews += 1
            review_payload = _load_review_payload(review_path)
            if review_payload is not None:
                latest_summary = review_payload["summary"]
            continue

        summary, details = _review_snapshot(
            snapshot_board,
            board_date=snapshot_board_date,
            review_date=review_date,
            positive_return=positive_return,
        )
        _save_review_payload(
            review_path,
            summary=summary,
            details=details,
            meta={
                "board_date": snapshot_board_date,
                "review_date": review_date,
                "horizon_days": int(horizon_days),
                "positive_return": float(positive_return),
                "ranking_by": ranking_by,
                "board_size": int(board_size),
            },
        )
        completed_reviews += 1
        new_reviews += 1
        latest_summary = summary

    previous_profile = load_adaptive_rank_profile(
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
        board_size=board_size,
    )
    review_payloads = _matching_review_payloads(
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
        board_size=board_size,
    )
    profile = _derive_adaptive_profile(review_payloads, rolling_review_days=rolling_limit)
    profile["rolling_review_days"] = int(rolling_limit)
    profile["rolling_review_mode"] = "recent_n_trading_days"
    profile["rolling_review_updated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    profile_changed = (
        dict(previous_profile.get("weights", {})) != dict(profile.get("weights", {}))
        or int(previous_profile.get("review_days", 0)) != int(profile.get("review_days", 0))
        or int(previous_profile.get("review_stocks", 0)) != int(profile.get("review_stocks", 0))
    )
    with _profile_cache_path(horizon_days, positive_return, ranking_by, board_size).open("wb") as handle:
        pickle.dump(
            {
                "cache_version": DAILY_REVIEW_CACHE_VERSION,
                "meta": {
                    "ranking_by": ranking_by,
                    "board_size": int(board_size),
                    "horizon_days": int(horizon_days),
                    "positive_return": float(positive_return),
                    "rolling_review_days": int(rolling_limit),
                    "rolling_review_mode": "recent_n_trading_days",
                },
                "profile": profile,
            },
            handle,
        )
    daily_lightweight_model = build_daily_lightweight_backtest_model(
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
        board_size=board_size,
        rolling_review_days=rolling_limit,
        persist=True,
    )

    if latest_summary is None:
        latest_summary = load_latest_review_summary(
            horizon_days=horizon_days,
            positive_return=positive_return,
            ranking_by=ranking_by,
            board_size=board_size,
        )

    return {
        "snapshot_path": str(snapshot_path) if snapshot_path else "",
        "snapshot_created": bool(snapshot_path) and not snapshot_existed,
        "latest_summary": latest_summary,
        "completed_reviews": int(completed_reviews),
        "new_reviews": int(new_reviews),
        "profile": profile,
        "profile_changed": bool(profile_changed),
        "rolling_review_days": int(rolling_limit),
        "eligible_snapshot_count": int(len(eligible_snapshots)),
        "processed_snapshot_count": int(min(len(eligible_snapshots), rolling_limit)),
        "daily_lightweight_model": daily_lightweight_model,
    }
