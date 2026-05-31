from __future__ import annotations

import argparse
import json
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from a_share_predictor.market_backtest_runner import (
    _add_fast_industry_metrics,
    _main_board_non_st_mask,
)


warnings.filterwarnings("ignore", category=FutureWarning)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_SOURCE = PROJECT_ROOT / ".cache" / "ten_year_model_strategy_validation"
DEFAULT_V3_DIR = PROJECT_ROOT / ".cache" / "ten_year_market_regime_v3"
DEFAULT_BULL_PATH = PROJECT_ROOT / ".cache" / "ten_year_bull_market_rank_score" / "bull_bear_regime_daily.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".cache" / "hard_filter_plan_comparison"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "docs" / "hard_filter_plan_backtest_comparison_2026-05-31.md"


@dataclass(frozen=True)
class Rule:
    strategy_mode: str
    market_filter: str
    score_threshold: float
    priority_threshold: float | None
    top_n: int
    sort_mode: str

    @property
    def name(self) -> str:
        prio = "none" if self.priority_threshold is None else f"{self.priority_threshold:g}"
        return (
            f"{self.strategy_mode}__{self.market_filter}__score{self.score_threshold:g}"
            f"__prio{prio}__top{self.top_n}__{self.sort_mode}"
        )


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if np.isnan(numeric) or np.isinf(numeric):
        return float(default)
    return numeric


def _format_date(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def _load_universe_names() -> pd.DataFrame:
    path = PROJECT_ROOT / ".cache" / "a_share_universe_v1.pkl"
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "name"])
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return pd.DataFrame(columns=["symbol", "name"])
    data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, pd.DataFrame) or data.empty:
        return pd.DataFrame(columns=["symbol", "name"])
    frame = data.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    frame["name"] = frame.get("name", frame["symbol"]).fillna(frame["symbol"]).astype(str)
    return frame[["symbol", "name"]].drop_duplicates("symbol")


def _normalize_raw_daily(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    if "symbol" not in frame.columns:
        frame["symbol"] = frame["ts_code"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    else:
        frame["symbol"] = frame["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    names = _load_universe_names()
    if not names.empty:
        frame = frame.merge(names, on="symbol", how="left")
    if "name" not in frame.columns:
        frame["name"] = frame["symbol"]
    frame["name"] = frame["name"].fillna(frame["symbol"]).astype(str)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    for column in ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["change_pct"] = frame["pct_chg"]
    frame["volume"] = frame["vol"]
    frame["amount"] = frame["amount"].fillna(0.0) * 1000.0
    frame["market"] = ""
    frame["industry"] = ""
    frame = frame.dropna(subset=["symbol", "trade_date", "close"]).copy()
    frame = frame.sort_values(["symbol", "trade_date"]).drop_duplicates(["symbol", "trade_date"], keep="last")
    return frame.reset_index(drop=True)


def _load_raw_daily_range(date_from: str, date_to: str, *, lookback_days: int = 260, forward_days: int = 8) -> pd.DataFrame:
    start = pd.to_datetime(date_from) - pd.Timedelta(days=int(lookback_days))
    end = pd.to_datetime(date_to) + pd.Timedelta(days=int(forward_days))
    start_key = start.strftime("%Y%m%d")
    end_key = end.strftime("%Y%m%d")
    files = sorted((PROJECT_ROOT / ".cache").glob("tushare_daily_range_fast_*.pkl"))
    frames: list[pd.DataFrame] = []
    for index, path in enumerate(files, start=1):
        frame = pd.read_pickle(path)
        if frame.empty or "trade_date" not in frame.columns:
            continue
        date_key = frame["trade_date"].astype(str)
        frame = frame.loc[date_key.between(start_key, end_key)].copy()
        if not frame.empty:
            frames.append(frame)
        if index == 1 or index % 25 == 0:
            print(f"[daily_cache] {index}/{len(files)} files, accumulated_parts={len(frames)}", flush=True)
    if not frames:
        return pd.DataFrame()
    return _normalize_raw_daily(pd.concat(frames, ignore_index=True, sort=False))


def _add_research_metrics(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history
    frame = history.sort_values(["symbol", "trade_date"]).copy()
    grouped = frame.groupby("symbol", group_keys=False, sort=False)
    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    amount = pd.to_numeric(frame["amount"], errors="coerce").fillna(0.0)

    for window in (5, 10, 20):
        frame[f"ma{window}"] = grouped["close"].transform(lambda s, w=window: pd.to_numeric(s, errors="coerce").rolling(w, min_periods=w).mean())
    for days, shift in ((3, 2), (5, 4), (10, 9), (15, 14), (20, 19)):
        base = grouped["close"].shift(shift)
        frame[f"ret_{days}d_pct"] = (close / pd.to_numeric(base, errors="coerce").replace(0.0, np.nan) - 1.0) * 100.0

    frame["high_10"] = grouped["high"].transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(10, min_periods=10).max())
    frame["low_10"] = grouped["low"].transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(10, min_periods=10).min())
    frame["distance_to_high_10_pct"] = (frame["high_10"] - close) / frame["high_10"].replace(0.0, np.nan) * 100.0
    frame["max_gain_10_pct"] = (frame["high_10"] / frame["low_10"].replace(0.0, np.nan) - 1.0) * 100.0

    day_range = (high - low).replace(0.0, np.nan)
    frame["close_position_day"] = ((close - low) / day_range).clip(lower=0.0, upper=1.0).fillna(0.5)
    frame["upper_shadow_ratio"] = ((high - close) / day_range).clip(lower=0.0, upper=1.0).fillna(0.0)
    rolling_volume = grouped["volume"].transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(5, min_periods=3).mean())
    frame["volume_ratio_5"] = (volume / rolling_volume.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    if "turnover" in frame.columns:
        turnover = pd.to_numeric(frame["turnover"], errors="coerce")
    else:
        turnover = pd.Series(np.nan, index=frame.index, dtype=float)
    rolling_amount = grouped["amount"].transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(20, min_periods=5).mean())
    if turnover.dropna().empty or float(turnover.fillna(0.0).abs().sum()) <= 0:
        turnover = (amount / rolling_amount.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0) * 4.0
    frame["turnover"] = turnover.fillna(0.0).astype(float)
    frame["amount_ma20"] = rolling_amount
    frame["amount_ratio20"] = (amount / rolling_amount.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    for window in (3, 5):
        frame[f"ma5_slope{window}"] = pd.to_numeric(frame["ma5"], errors="coerce") / grouped["ma5"].shift(window).replace(0.0, np.nan) - 1.0
        frame[f"ma10_slope{window}"] = pd.to_numeric(frame["ma10"], errors="coerce") / grouped["ma10"].shift(window).replace(0.0, np.nan) - 1.0

    # Lightweight pullback approximation for research comparison. The production path has
    # a slower detailed pullback detector; this keeps the ten-year experiment tractable.
    distance = pd.to_numeric(frame["distance_to_high_10_pct"], errors="coerce")
    change_pct = pd.to_numeric(frame["change_pct"], errors="coerce").fillna(0.0)
    frame["pullback_days"] = np.select(
        [
            distance.between(0.0, 1.0, inclusive="both"),
            distance.between(1.0, 6.0, inclusive="right"),
            distance.between(6.0, 12.0, inclusive="right"),
        ],
        [1, 3, 7],
        default=0,
    )
    frame["pullback_volume_decay"] = frame["volume_ratio_5"].le(1.25)
    frame["pullback_kept_ma10"] = close.ge(pd.to_numeric(frame["ma10"], errors="coerce") * 0.98)
    frame["recent_reclaim"] = close.ge(pd.to_numeric(frame["ma10"], errors="coerce") * 0.985) & change_pct.gt(0)
    return frame.replace([np.inf, -np.inf], np.nan)


def _prepared_history_cache_path(date_from: str, date_to: str) -> Path:
    safe_from = str(date_from).replace("-", "")
    safe_to = str(date_to).replace("-", "")
    return DEFAULT_OUTPUT_DIR / f"prepared_fast_history_v1_{safe_from}_{safe_to}.pkl"


def load_or_build_prepared_history(date_from: str, date_to: str, *, force: bool = False) -> pd.DataFrame:
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _prepared_history_cache_path(date_from, date_to)
    if cache_path.exists() and not force:
        print(f"[prepare] reuse {cache_path}", flush=True)
        return pd.read_pickle(cache_path)
    raw = _load_raw_daily_range(date_from, date_to)
    if raw.empty:
        raise RuntimeError("No raw daily cache rows were available for the requested window.")
    print(f"[prepare] raw rows={len(raw)} symbols={raw['symbol'].nunique()}", flush=True)
    history = _add_research_metrics(raw)
    history = _add_fast_industry_metrics(history)
    history = history.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    history = history.replace([np.inf, -np.inf], np.nan)
    history.to_pickle(cache_path)
    print(f"[prepare] saved {cache_path}", flush=True)
    return history


def load_regime(v3_dir: Path, bull_path: Path, date_from: str, date_to: str) -> pd.DataFrame:
    v3 = pd.read_csv(v3_dir / "market_regime_daily.csv", encoding="utf-8-sig", parse_dates=["market_date"])
    bull = pd.read_csv(bull_path, encoding="utf-8-sig", parse_dates=["market_date"])
    keep_v3 = [
        "market_date",
        "market_ret",
        "up_ratio",
        "above_ma20_ratio",
        "limit_up_count",
        "limit_down_count",
        "amount_ma5_ma20",
        "amount_ma20_ma60",
        "up_amount_ratio",
        "strong_amount_ratio",
        "trend_score",
        "flow_score",
        "trend_green",
        "flow_green",
        "internal_green",
        "market_green",
        "v3_full_green",
        "v3_yellow",
        "market_state",
    ]
    keep_bull = [
        "market_date",
        "sse_close",
        "bull_score",
        "bull_bear_state",
        "is_bull_strict",
        "is_bull_loose",
    ]
    frame = v3[[column for column in keep_v3 if column in v3.columns]].merge(
        bull[[column for column in keep_bull if column in bull.columns]],
        on="market_date",
        how="left",
    )
    start = pd.to_datetime(date_from)
    end = pd.to_datetime(date_to)
    frame = frame.loc[frame["market_date"].between(start, end, inclusive="both")].copy()
    for column in [
        "trend_green",
        "flow_green",
        "internal_green",
        "market_green",
        "v3_full_green",
        "v3_yellow",
        "is_bull_strict",
        "is_bull_loose",
    ]:
        if column in frame.columns:
            frame[column] = frame[column].astype("boolean").fillna(False).astype(bool)
    return frame.sort_values("market_date").reset_index(drop=True)


def _regime_bucket(row: pd.Series) -> str:
    bull_score = _safe_float(row.get("bull_score"), 0.0)
    v3_full_green = bool(row.get("v3_full_green", False))
    market_green = bool(row.get("market_green", False))
    v3_yellow = bool(row.get("v3_yellow", False))
    state = str(row.get("market_state", "")).lower()
    up_ratio = _safe_float(row.get("up_ratio"), 0.0)
    above_ma20 = _safe_float(row.get("above_ma20_ratio"), 0.0)
    if v3_full_green and bull_score >= 7 and up_ratio >= 0.52:
        return "strong_trend"
    if v3_full_green or market_green or (bull_score >= 6 and above_ma20 >= 0.50):
        return "strong_range"
    if v3_yellow or state == "yellow" or bull_score >= 4 or up_ratio >= 0.45:
        return "weak_range"
    return "weak"


def _num(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _bool_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    return frame[column].fillna(default).astype(bool)


def _build_plan_candidates_for_day(day: pd.DataFrame, market_date: str, regime_row: pd.Series | None) -> pd.DataFrame:
    if day.empty:
        return pd.DataFrame()
    day = day.loc[_main_board_non_st_mask(day)].copy()
    if day.empty:
        return pd.DataFrame()
    regime_row = regime_row if regime_row is not None else pd.Series(dtype=object)
    bucket = _regime_bucket(regime_row)
    thresholds = {
        "strong_trend": (64.0, 66.0, 58.0),
        "strong_range": (61.0, 68.0, 58.0),
        "weak_range": (63.0, 72.0, 61.0),
        "weak": (68.0, 75.0, 65.0),
    }[bucket]
    s1_threshold, s2_threshold, s3_threshold = thresholds
    base_liquidity = 1.5e8 if bucket == "weak" else 1.0e8
    s2_liquidity = 2.5e8 if bucket in {"weak", "weak_range"} else 2.0e8
    s3_liquidity = 1.2e8 if bucket in {"weak", "weak_range"} else 8.0e7
    ret20_cap = 40.0 if bucket in {"weak", "weak_range"} else 45.0

    close = _num(day, "close")
    ma5 = _num(day, "ma5", np.nan)
    ma10 = _num(day, "ma10", np.nan)
    ma20 = _num(day, "ma20", np.nan)
    amount = _num(day, "amount")
    turnover = _num(day, "turnover")
    change_pct = _num(day, "change_pct")
    ret3 = _num(day, "ret_3d_pct", np.nan)
    ret5 = _num(day, "ret_5d_pct", np.nan)
    ret15 = _num(day, "ret_15d_pct", np.nan)
    ret20 = _num(day, "ret_20d_pct", np.nan)
    max_gain10 = _num(day, "max_gain_10_pct", np.nan)
    distance_high = _num(day, "distance_to_high_10_pct", np.nan)
    high10 = _num(day, "high_10", np.nan)
    close_pos = _num(day, "close_position_day", 0.5)
    upper = _num(day, "upper_shadow_ratio")
    volume_ratio = _num(day, "volume_ratio_5", 1.0)
    amount_ratio20 = _num(day, "amount_ratio20", 1.0)
    ma5_slope3 = _num(day, "ma5_slope3")
    ma10_slope3 = _num(day, "ma10_slope3")
    pullback_days = _num(day, "pullback_days")
    pullback_volume_decay = _bool_series(day, "pullback_volume_decay")
    pullback_kept_ma10 = _bool_series(day, "pullback_kept_ma10")
    industry_ret = _num(day, "industry_ret_2d_pct")
    industry_up = _num(day, "industry_up_count")
    industry_top = _bool_series(day, "industry_top2d_flag")

    market_ret = _safe_float(regime_row.get("market_ret"), 0.0)
    market_ret_pct = market_ret * 100.0 if abs(market_ret) <= 1.0 else market_ret
    up_ratio = _safe_float(regime_row.get("up_ratio"), 0.0)
    amount_ma5_ma20 = _safe_float(regime_row.get("amount_ma5_ma20"), 1.0)
    strong_amount_ratio = _safe_float(regime_row.get("strong_amount_ratio"), 1.0)
    v3_full_green = bool(regime_row.get("v3_full_green", False))
    flow_green = bool(regime_row.get("flow_green", False))

    relative_strength = change_pct - market_ret_pct
    late_risk = (
        ret20.clip(lower=0, upper=60).fillna(0.0) * 0.55
        + max_gain10.clip(lower=0, upper=65).fillna(0.0) * 0.70
        + upper.clip(0, 0.70) * 45.0
        + turnover.clip(0, 18) * 0.55
        + (1.0 - close_pos.clip(0, 1)) * 10.0
    ).clip(0.0, 100.0)
    terminal_risk = (
        (ret20 > 55.0)
        | (max_gain10 > 65.0)
        | ((ret20 > 38.0) & (upper > 0.28) & (volume_ratio > 1.6))
        | ((turnover > 12.0) & (upper > 0.30) & (close_pos < 0.55))
    )
    common = (
        (close > 0)
        & (ma20 > 0)
        & (close >= ma20 * 0.97)
        & (ret20 < 55.0)
        & (max_gain10 < 62.0)
        & (~terminal_risk)
    )

    trend_score = (
        32.0
        + (close >= ma10).astype(float) * 10.0
        + (close >= ma20).astype(float) * 9.0
        + (ma5 >= ma10 * 0.995).astype(float) * 9.0
        + (ma10 >= ma20 * 0.985).astype(float) * 8.0
        + (ma5_slope3 >= -0.002).astype(float) * 5.0
        + (ma10_slope3 >= -0.003).astype(float) * 4.0
        + ret15.between(6.0, 28.0, inclusive="both").fillna(False).astype(float) * 8.0
        + ret20.between(-4.0, 35.0, inclusive="both").fillna(False).astype(float) * 7.0
        + (distance_high <= 5.0).fillna(False).astype(float) * 5.0
    ).clip(0.0, 100.0)
    pullback_score = (
        34.0
        + pullback_days.between(1.0, 2.0, inclusive="both").fillna(False).astype(float) * 8.0
        + pullback_days.between(3.0, 6.0, inclusive="both").fillna(False).astype(float) * 13.0
        + pullback_days.between(7.0, 10.0, inclusive="both").fillna(False).astype(float) * 5.0
        + pullback_volume_decay.astype(float) * 8.0
        + pullback_kept_ma10.astype(float) * 8.0
        + (volume_ratio <= 1.35).astype(float) * 5.0
        + (close_pos >= 0.45).astype(float) * 5.0
    ).clip(0.0, 100.0)
    sector_score = (
        42.0
        + industry_ret.clip(-4, 12).fillna(0.0) * 2.3
        + industry_up.clip(0, 12).fillna(0.0) * 1.7
        + industry_top.astype(float) * 7.0
        + up_ratio * 9.0
        + max(amount_ma5_ma20 - 0.90, 0.0) * 8.0
        + max(strong_amount_ratio - 1.0, 0.0) * 3.0
    ).clip(0.0, 100.0)
    launch_score = (
        42.0
        + change_pct.clip(-2, 9).fillna(0.0) * 2.0
        + close_pos.clip(0, 1) * 16.0
        + (volume_ratio.clip(0.5, 2.4) - 1.0) * 7.0
        + (amount_ratio20.clip(0.5, 3.0) - 1.0) * 6.0
        - upper.clip(0, 0.6) * 18.0
    ).clip(0.0, 100.0)
    breakout_score = (
        38.0
        + relative_strength.clip(-3, 10).fillna(0.0) * 2.5
        + ret3.clip(-4, 16).fillna(0.0) * 0.8
        + ret5.clip(-4, 20).fillna(0.0) * 0.7
        + (distance_high <= 3.0).fillna(False).astype(float) * 8.0
        + (close >= high10).fillna(False).astype(float) * 4.0
        + close_pos.clip(0, 1) * 10.0
        - late_risk * 0.18
    ).clip(0.0, 100.0)

    s1_score = (trend_score * 0.32 + pullback_score * 0.24 + launch_score * 0.22 + sector_score * 0.22 - late_risk * 0.16).clip(0.0, 100.0)
    s2_score = (breakout_score * 0.34 + sector_score * 0.26 + launch_score * 0.25 + trend_score * 0.15 - late_risk * 0.18).clip(0.0, 100.0)
    s3_score = (trend_score * 0.30 + launch_score * 0.25 + sector_score * 0.25 + breakout_score * 0.20 - late_risk * 0.14).clip(0.0, 100.0)

    s1 = (
        common
        & (amount >= base_liquidity)
        & ((close >= ma10 * 0.985) | ((close >= ma20 * 0.97) & (ma5_slope3 >= -0.004)))
        & (ma5 >= ma10 * 0.985)
        & (ma10 >= ma20 * 0.970)
        & (ret20 < ret20_cap)
        & (upper <= 0.34)
        & (s1_score >= s1_threshold)
        & (late_risk < 48.0)
    )
    resonance_ok = (
        (sector_score >= 55.0)
        | ((up_ratio >= 0.55) & (v3_full_green | flow_green))
        | ((industry_ret >= 0.8) & (industry_up >= 2.0))
    )
    s2 = (
        common
        & (amount >= s2_liquidity)
        & (relative_strength >= (3.0 if bucket in {"weak", "weak_range"} else 2.0))
        & (close_pos >= 0.55)
        & (((close >= high10) | (distance_high <= 3.0)).fillna(False))
        & resonance_ok
        & (s2_score >= s2_threshold)
        & (late_risk < 52.0)
    )
    s3 = (
        common
        & (amount >= s3_liquidity)
        & (((close > ma10) | ((close > ma20) & (ma5_slope3 >= -0.003))).fillna(False))
        & ret5.between(0.0, 18.0, inclusive="both").fillna(False)
        & ret20.between(-8.0, 35.0, inclusive="both").fillna(False)
        & (max_gain10 < 45.0)
        & (upper <= 0.34)
        & (close_pos >= 0.42)
        & (s3_score >= s3_threshold)
        & (late_risk < 58.0)
        & (~s1)
        & (~s2)
    )

    labeled: list[pd.DataFrame] = []
    for label, mask, score in [
        ("strategy1_plan_p1", s1, s1_score + 3.0),
        ("strategy2_plan_p1", s2, s2_score + 5.0),
        ("strategy3_plan_p1", s3, s3_score),
    ]:
        if not bool(mask.any()):
            continue
        part = day.loc[mask].copy()
        part["candidate_strategy"] = label
        part["candidate_priority"] = score.loc[part.index].round(4)
        labeled.append(part)
    if not labeled:
        return pd.DataFrame()
    candidates = pd.concat(labeled, ignore_index=True, sort=False)
    candidates["market_date"] = pd.to_datetime(market_date)
    candidates["strategy_rank"] = pd.to_numeric(candidates["candidate_priority"], errors="coerce").fillna(0.0)
    candidates["latest_price"] = pd.to_numeric(candidates["close"], errors="coerce").fillna(0.0)
    candidates["plan_market_bucket"] = bucket
    columns = [
        "market_date",
        "symbol",
        "name",
        "candidate_strategy",
        "candidate_priority",
        "strategy_rank",
        "latest_price",
        "change_pct",
        "amount",
        "turnover",
        "industry_name",
        "industry_ret_2d_pct",
        "industry_up_count",
        "plan_market_bucket",
    ]
    candidates = candidates[[column for column in columns if column in candidates.columns]].copy()
    candidates = candidates.sort_values(["candidate_priority", "amount"], ascending=False)
    candidates = candidates.drop_duplicates(["market_date", "symbol"], keep="first")
    return candidates.reset_index(drop=True)


def build_plan_candidate_pool(history: pd.DataFrame, regime: pd.DataFrame, date_from: str, date_to: str) -> pd.DataFrame:
    start = pd.to_datetime(date_from)
    end = pd.to_datetime(date_to)
    dates = sorted(
        pd.to_datetime(history.loc[history["trade_date"].between(start, end, inclusive="both"), "trade_date"], errors="coerce")
        .dropna()
        .dt.strftime("%Y-%m-%d")
        .unique()
        .tolist()
    )
    regime_lookup = {pd.Timestamp(row.market_date).strftime("%Y-%m-%d"): row for row in regime.itertuples(index=False)}
    frames: list[pd.DataFrame] = []
    total = len(dates)
    for index, market_date in enumerate(dates, start=1):
        if index == 1 or index % 50 == 0:
            print(f"[plan_candidates] {index}/{total} {market_date}", flush=True)
        day = history.loc[history["trade_date"].eq(pd.to_datetime(market_date))].copy()
        row = regime_lookup.get(market_date)
        regime_row = pd.Series(row._asdict()) if row is not None else None
        candidates = _build_plan_candidates_for_day(day, market_date, regime_row)
        if not candidates.empty:
            frames.append(candidates)
    if not frames:
        return pd.DataFrame()
    pool = pd.concat(frames, ignore_index=True, sort=False)
    pool["symbol"] = pool["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    pool["market_date"] = pd.to_datetime(pool["market_date"], errors="coerce").dt.normalize()
    pool = pool.dropna(subset=["market_date", "symbol"]).copy()
    return pool.sort_values(["market_date", "candidate_priority", "amount"], ascending=[True, False, False]).reset_index(drop=True)


def load_old_combined_candidates(source_dir: Path, date_from: str, date_to: str) -> pd.DataFrame:
    columns = {
        "symbol",
        "name",
        "candidate_strategy",
        "candidate_priority",
        "strategy_rank",
        "latest_price",
        "change_pct",
        "amount",
        "turnover",
        "industry_name",
        "industry_ret_2d_pct",
        "industry_up_count",
        "market_date",
        "model_probability",
        "model_score",
        "hold_3d_return",
        "max_high_return",
        "max_drawdown",
        "entry_price",
    }
    frames: list[pd.DataFrame] = []
    start = pd.to_datetime(date_from)
    end = pd.to_datetime(date_to)
    for path in sorted((source_dir / "chunks").glob("*/combined_candidates.csv")):
        frame = pd.read_csv(path, encoding="utf-8-sig", usecols=lambda column: column in columns, parse_dates=["market_date"])
        frame = frame.loc[frame["market_date"].between(start, end, inclusive="both")].copy()
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True, sort=False)
    data["symbol"] = data["symbol"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(6)
    return data.dropna(subset=["market_date", "symbol", "model_score", "hold_3d_return"]).copy()


def attach_model_scores(candidates: pd.DataFrame, model_scores_path: Path) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    local = candidates.copy()
    local["market_date"] = pd.to_datetime(local["market_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    local["symbol"] = local["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    local["join_key"] = local["market_date"] + "|" + local["symbol"]
    needed = set(local["join_key"].tolist())
    usecols = [
        "market_date",
        "symbol",
        "name",
        "model_probability",
        "model_score",
        "hold_3d_return",
        "max_high_return",
        "max_drawdown",
    ]
    frames: list[pd.DataFrame] = []
    for index, chunk in enumerate(pd.read_csv(model_scores_path, encoding="utf-8-sig", usecols=usecols, chunksize=750_000), start=1):
        chunk["market_date"] = pd.to_datetime(chunk["market_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        chunk["symbol"] = chunk["symbol"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(6)
        chunk["join_key"] = chunk["market_date"] + "|" + chunk["symbol"]
        matched = chunk.loc[chunk["join_key"].isin(needed)].copy()
        if not matched.empty:
            frames.append(matched)
        if index == 1 or index % 10 == 0:
            print(f"[model_join] chunks={index} matched_parts={len(frames)}", flush=True)
    if not frames:
        return pd.DataFrame()
    scored = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates("join_key", keep="last")
    merged = local.merge(
        scored.drop(columns=["name"], errors="ignore"),
        on=["join_key", "market_date", "symbol"],
        how="inner",
    )
    merged["market_date"] = pd.to_datetime(merged["market_date"], errors="coerce")
    for column in ["model_probability", "model_score", "hold_3d_return", "max_high_return", "max_drawdown"]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    return merged.dropna(subset=["market_date", "symbol", "model_score", "hold_3d_return"]).copy()


def enrich_candidates(frame: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    local = frame.copy()
    local["market_date"] = pd.to_datetime(local["market_date"], errors="coerce").dt.normalize()
    enriched = local.merge(regime, on="market_date", how="left")
    for column in [
        "trend_green",
        "flow_green",
        "internal_green",
        "market_green",
        "v3_full_green",
        "v3_yellow",
        "is_bull_strict",
        "is_bull_loose",
    ]:
        if column in enriched.columns:
            enriched[column] = enriched[column].astype("boolean").fillna(False).astype(bool)
    text = enriched["candidate_strategy"].fillna("").astype(str).str.lower()
    enriched["strategy_family"] = np.select(
        [
            text.str.contains("strategy1", regex=False) | text.str.contains("1", regex=False),
            text.str.contains("strategy2", regex=False) | text.str.contains("2", regex=False),
            text.str.contains("strategy3", regex=False) | text.str.contains("3", regex=False),
        ],
        ["strategy1", "strategy2", "strategy3"],
        default="unknown",
    )
    enriched["priority_score"] = pd.to_numeric(enriched.get("candidate_priority"), errors="coerce").fillna(0.0).clip(0.0, 100.0)
    enriched["model_priority_80_20"] = pd.to_numeric(enriched["model_score"], errors="coerce") * 0.80 + enriched["priority_score"] * 0.20
    return enriched


def _strategy_mask(frame: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "all":
        return pd.Series(True, index=frame.index)
    if mode == "old12":
        return frame["strategy_family"].isin(["strategy1", "strategy2"])
    return frame["strategy_family"].eq(mode)


def _market_mask(frame: pd.DataFrame, mode: str) -> pd.Series:
    true = pd.Series(True, index=frame.index)
    if mode == "none":
        return true
    if mode == "market_green":
        return frame["market_green"].fillna(False)
    if mode == "v3_full_green":
        return frame["v3_full_green"].fillna(False)
    if mode == "bull7":
        return frame["is_bull_strict"].fillna(False)
    if mode == "bull6":
        return frame["is_bull_loose"].fillna(False)
    if mode == "bull7_v3_full_green":
        return frame["is_bull_strict"].fillna(False) & frame["v3_full_green"].fillna(False)
    if mode == "bull6_v3_full_green":
        return frame["is_bull_loose"].fillna(False) & frame["v3_full_green"].fillna(False)
    raise ValueError(f"Unknown market filter: {mode}")


def select_top(frame: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    data = frame.loc[
        _strategy_mask(frame, rule.strategy_mode)
        & _market_mask(frame, rule.market_filter)
        & pd.to_numeric(frame["model_score"], errors="coerce").ge(float(rule.score_threshold))
    ].copy()
    if rule.priority_threshold is not None:
        data = data.loc[pd.to_numeric(data["priority_score"], errors="coerce").ge(float(rule.priority_threshold))].copy()
    if data.empty:
        return data
    sort_columns = ["market_date", rule.sort_mode]
    ascending = [True, False]
    if rule.sort_mode != "model_score":
        sort_columns.append("model_score")
        ascending.append(False)
    sort_columns.append("priority_score")
    ascending.append(False)
    data = data.sort_values(sort_columns, ascending=ascending).drop_duplicates(["market_date", "symbol"], keep="first")
    data["daily_rule_rank"] = data.groupby("market_date").cumcount() + 1
    selected = data.loc[data["daily_rule_rank"].le(int(rule.top_n))].copy()
    counts = selected.groupby("market_date").size()
    full_dates = counts.loc[counts.ge(int(rule.top_n))].index
    return selected.loc[selected["market_date"].isin(full_dates)].copy()


def _build_curve_with_cost(selected: pd.DataFrame, calendar: pd.DataFrame, cost_bps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = selected.copy()
    if not frame.empty:
        frame["hold_3d_return"] = pd.to_numeric(frame["hold_3d_return"], errors="coerce")
        frame = frame.dropna(subset=["market_date", "hold_3d_return"]).copy()
        frame["adjusted_return"] = frame["hold_3d_return"] - float(cost_bps) / 10_000.0
    if frame.empty:
        daily = pd.DataFrame(columns=["market_date", "selected", "avg_return", "avg_model_score", "avg_priority_score"])
    else:
        daily = (
            frame.groupby("market_date", as_index=False)
            .agg(
                selected=("symbol", "count"),
                avg_return=("adjusted_return", "mean"),
                avg_model_score=("model_score", "mean"),
                avg_priority_score=("priority_score", "mean"),
            )
            .sort_values("market_date")
        )
    curve = calendar[["market_date"]].merge(daily, on="market_date", how="left")
    curve["selected"] = curve["selected"].fillna(0).astype(int)
    curve["avg_return"] = curve["avg_return"].fillna(0.0)
    curve["equity"] = (1.0 + curve["avg_return"]).cumprod()
    curve["running_max"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["running_max"].replace(0.0, np.nan) - 1.0
    return curve, frame


def _period_metrics(curve: pd.DataFrame, selected: pd.DataFrame) -> dict[str, object]:
    if curve.empty:
        return {}
    ending = _safe_float(curve["equity"].iloc[-1], 1.0)
    annualized = ending ** (252.0 / len(curve)) - 1.0 if len(curve) and ending > 0 else 0.0
    max_drawdown = _safe_float(curve["drawdown"].min(), 0.0)
    active_daily = pd.to_numeric(curve.loc[curve["selected"] > 0, "avg_return"], errors="coerce").dropna()
    returns = pd.to_numeric(selected.get("adjusted_return", pd.Series(dtype=float)), errors="coerce").dropna()
    raw_returns = pd.to_numeric(selected.get("hold_3d_return", pd.Series(dtype=float)), errors="coerce").dropna()
    return {
        "calendar_days": int(len(curve)),
        "active_days": int((curve["selected"] > 0).sum()),
        "coverage_pct": round(float((curve["selected"] > 0).mean()) * 100.0, 2),
        "selected_rows": int(len(selected)),
        "annualized_return": round(float(annualized), 6),
        "max_drawdown": round(float(max_drawdown), 6),
        "ending_equity": round(float(ending), 6),
        "return_drawdown_ratio": round(float(annualized / abs(max_drawdown)), 6) if max_drawdown < 0 else None,
        "active_daily_return": round(float(active_daily.mean()), 6) if not active_daily.empty else None,
        "active_daily_win_rate": round(float((active_daily > 0).mean()), 4) if not active_daily.empty else None,
        "avg_trade_return": round(float(returns.mean()), 6) if not returns.empty else None,
        "trade_win_rate": round(float((returns > 0).mean()), 4) if not returns.empty else None,
        "target_hit_rate": round(float((raw_returns >= 0.03).mean()), 4) if not raw_returns.empty else None,
    }


def evaluate_rules(frame: pd.DataFrame, calendar: pd.DataFrame, rules: list[Rule], source: str, cost_bps_values: Iterable[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    selected_samples: list[pd.DataFrame] = []
    for index, rule in enumerate(rules, start=1):
        selected = select_top(frame, rule)
        if index == 1 or index % 50 == 0:
            print(f"[evaluate:{source}] {index}/{len(rules)} {rule.name} selected={len(selected)}", flush=True)
        for cost_bps in cost_bps_values:
            curve, adjusted = _build_curve_with_cost(selected, calendar, float(cost_bps))
            row = _period_metrics(curve, adjusted)
            row.update(
                {
                    "source": source,
                    "rule": rule.name,
                    "cost_bps": float(cost_bps),
                    "strategy_mode": rule.strategy_mode,
                    "market_filter": rule.market_filter,
                    "score_threshold": float(rule.score_threshold),
                    "priority_threshold": rule.priority_threshold,
                    "top_n": int(rule.top_n),
                    "sort_mode": rule.sort_mode,
                }
            )
            rows.append(row)
            if rule.name in {
                "strategy3__bull7_v3_full_green__score68__prionone__top3__model_score",
                "all__bull7_v3_full_green__score68__prionone__top3__model_score",
                "all__v3_full_green__score68__prionone__top3__model_score",
            } and float(cost_bps) in {0.0, 20.0}:
                selected_samples.append(adjusted.assign(source=source, rule=rule.name, cost_bps=float(cost_bps)))
    summary = pd.DataFrame(rows)
    samples = pd.concat(selected_samples, ignore_index=True, sort=False) if selected_samples else pd.DataFrame()
    if not summary.empty:
        summary["cost_adjusted_score"] = (
            pd.to_numeric(summary["annualized_return"], errors="coerce").fillna(0.0) * 100.0
            - pd.to_numeric(summary["max_drawdown"], errors="coerce").abs().fillna(0.0) * 10.0
        )
        summary = summary.sort_values(
            ["source", "cost_bps", "cost_adjusted_score", "return_drawdown_ratio", "annualized_return"],
            ascending=[True, True, False, False, False],
        )
    return summary, samples


def candidate_count_summary(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    local = frame.copy()
    local["market_date"] = pd.to_datetime(local["market_date"], errors="coerce")
    daily = local.groupby("market_date").agg(total_candidates=("symbol", "nunique")).reset_index()
    strategy = (
        local.groupby(["market_date", "strategy_family"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    result = daily.merge(strategy, on="market_date", how="left")
    result["source"] = source
    return result


def build_rules() -> list[Rule]:
    rules: list[Rule] = []
    strategy_modes = ["all", "strategy1", "strategy2", "strategy3"]
    market_filters = ["none", "v3_full_green", "bull7_v3_full_green", "bull6_v3_full_green", "market_green", "bull7"]
    score_thresholds = [66.0, 68.0, 70.0]
    priority_thresholds: list[float | None] = [None, 60.0, 65.0]
    sort_modes = ["model_score", "model_priority_80_20"]
    for strategy_mode in strategy_modes:
        for market_filter in market_filters:
            for score_threshold in score_thresholds:
                for priority_threshold in priority_thresholds:
                    for sort_mode in sort_modes:
                        rules.append(
                            Rule(
                                strategy_mode=strategy_mode,
                                market_filter=market_filter,
                                score_threshold=score_threshold,
                                priority_threshold=priority_threshold,
                                top_n=3,
                                sort_mode=sort_mode,
                            )
                        )
    return rules


def _markdown_pct(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.2f}%"


def write_report(
    *,
    output_dir: Path,
    report_path: Path,
    old_counts: pd.DataFrame,
    plan_counts: pd.DataFrame,
    comparison: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = [
        "# Hard Filter Plan Backtest Comparison 2026-05-31",
        "",
        "## Scope",
        "",
        f"- Window: `{metadata['date_from']}` to `{metadata['date_to']}`.",
        "- New strategy implementation: P1 proxy from `strategy_hard_filter_optimization_plan_2026-05-31.md` using available ten-year daily cache features.",
        "- Ranking and labels: existing ten-year `model_scores.csv`; no model retraining in this run.",
        "- Caveat: ten-year cache does not include true ten-year intraday, fund-flow or news features, so those plan terms are not included here.",
        "",
        "## Candidate Breadth",
        "",
    ]
    for label, counts in [("old_strict", old_counts), ("plan_p1", plan_counts)]:
        if counts.empty:
            continue
        active = counts[counts["total_candidates"] > 0]
        rows.extend(
            [
                f"- `{label}` active days: `{len(active)}`; avg daily candidates: `{active['total_candidates'].mean():.1f}`; "
                f"median: `{active['total_candidates'].median():.1f}`; p90: `{active['total_candidates'].quantile(0.90):.1f}`.",
            ]
        )
    rows.extend(["", "## Key Same-Rule Comparison", ""])
    key_rules = [
        "strategy3__bull7_v3_full_green__score68__prionone__top3__model_score",
        "all__bull7_v3_full_green__score68__prionone__top3__model_score",
        "all__v3_full_green__score68__prionone__top3__model_score",
    ]
    table = comparison.loc[
        comparison["rule"].isin(key_rules) & comparison["cost_bps"].isin([0.0, 20.0])
    ].sort_values(["rule", "cost_bps", "source"])
    rows.append("| rule | cost_bps | source | active_days | selected | ann_return | max_dd | win_rate | avg_trade |")
    rows.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in table.to_dict("records"):
        rows.append(
            "| "
            + " | ".join(
                [
                    str(row.get("rule", "")),
                    f"{float(row.get('cost_bps', 0.0)):.0f}",
                    str(row.get("source", "")),
                    str(int(row.get("active_days", 0) or 0)),
                    str(int(row.get("selected_rows", 0) or 0)),
                    _markdown_pct(row.get("annualized_return")),
                    _markdown_pct(row.get("max_drawdown")),
                    _markdown_pct(row.get("trade_win_rate")),
                    _markdown_pct(row.get("avg_trade_return")),
                ]
            )
            + " |"
        )
    rows.extend(["", "## Best Plan Rules By Cost", ""])
    best_plan = comparison.loc[comparison["source"].eq("plan_p1")].groupby("cost_bps", as_index=False).head(5)
    rows.append("| cost_bps | rule | active_days | ann_return | max_dd | score |")
    rows.append("|---:|---|---:|---:|---:|---:|")
    for row in best_plan.to_dict("records"):
        rows.append(
            "| "
            + " | ".join(
                [
                    f"{float(row.get('cost_bps', 0.0)):.0f}",
                    str(row.get("rule", "")),
                    str(int(row.get("active_days", 0) or 0)),
                    _markdown_pct(row.get("annualized_return")),
                    _markdown_pct(row.get("max_drawdown")),
                    f"{float(row.get('cost_adjusted_score', 0.0)):.3f}",
                ]
            )
            + " |"
        )
    rows.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Summary JSON: `{output_dir / 'summary.json'}`",
            f"- Rule comparison: `{output_dir / 'rule_comparison.csv'}`",
            f"- Candidate counts: `{output_dir / 'candidate_counts.csv'}`",
            f"- Selected samples: `{output_dir / 'selected_samples.csv'}`",
        ]
    )
    report_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def run_comparison(
    *,
    date_from: str,
    date_to: str,
    output_dir: Path,
    model_source: Path,
    v3_dir: Path,
    bull_path: Path,
    report_path: Path,
    force_history: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    regime = load_regime(v3_dir, bull_path, date_from, date_to)
    calendar = regime[["market_date"]].drop_duplicates().sort_values("market_date").reset_index(drop=True)

    print("[old] loading existing strict combined candidates", flush=True)
    old = enrich_candidates(load_old_combined_candidates(model_source, date_from, date_to), regime)

    print("[plan] building P1 candidate pool", flush=True)
    history = load_or_build_prepared_history(date_from, date_to, force=force_history)
    plan_pool = build_plan_candidate_pool(history, regime, date_from, date_to)
    plan_pool_path = output_dir / "plan_candidate_pool.csv"
    plan_pool.to_csv(plan_pool_path, index=False, encoding="utf-8-sig")
    print(f"[plan] candidate rows={len(plan_pool)} days={plan_pool['market_date'].nunique() if not plan_pool.empty else 0}", flush=True)
    plan_scored = enrich_candidates(attach_model_scores(plan_pool, model_source / "model_scores.csv"), regime)

    rules = build_rules()
    costs = [0.0, 10.0, 20.0, 30.0]
    old_summary, old_samples = evaluate_rules(old, calendar, rules, "old_strict", costs)
    plan_summary, plan_samples = evaluate_rules(plan_scored, calendar, rules, "plan_p1", costs)
    comparison = pd.concat([old_summary, plan_summary], ignore_index=True, sort=False)
    comparison = comparison.sort_values(
        ["source", "cost_bps", "cost_adjusted_score", "return_drawdown_ratio", "annualized_return"],
        ascending=[True, True, False, False, False],
    )
    old_counts = candidate_count_summary(old, "old_strict")
    plan_counts = candidate_count_summary(plan_scored, "plan_p1")
    counts = pd.concat([old_counts, plan_counts], ignore_index=True, sort=False)
    samples = pd.concat([old_samples, plan_samples], ignore_index=True, sort=False) if not old_samples.empty or not plan_samples.empty else pd.DataFrame()

    comparison_path = output_dir / "rule_comparison.csv"
    counts_path = output_dir / "candidate_counts.csv"
    samples_path = output_dir / "selected_samples.csv"
    scored_path = output_dir / "plan_scored_candidates.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    counts.to_csv(counts_path, index=False, encoding="utf-8-sig")
    samples.to_csv(samples_path, index=False, encoding="utf-8-sig")
    plan_scored.to_csv(scored_path, index=False, encoding="utf-8-sig")

    metadata = {
        "date_from": date_from,
        "date_to": date_to,
        "old_candidate_rows": int(len(old)),
        "plan_candidate_rows": int(len(plan_scored)),
        "calendar_days": int(len(calendar)),
        "rule_count": int(len(rules)),
        "cost_bps_values": costs,
        "comparison_path": str(comparison_path),
        "candidate_counts_path": str(counts_path),
        "selected_samples_path": str(samples_path),
        "plan_candidate_pool_path": str(plan_pool_path),
        "plan_scored_candidates_path": str(scored_path),
        "report_path": str(report_path),
        "best_by_source_cost": comparison.groupby(["source", "cost_bps"], as_index=False).head(5).replace({np.nan: None}).to_dict("records"),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(
        output_dir=output_dir,
        report_path=report_path,
        old_counts=old_counts,
        plan_counts=plan_counts,
        comparison=comparison,
        metadata=metadata,
    )
    return metadata


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare old hard filters with the 2026-05-31 hard-filter optimization plan.")
    parser.add_argument("--date-from", default="2016-05-27")
    parser.add_argument("--date-to", default="2026-05-26")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-source", default=str(DEFAULT_MODEL_SOURCE))
    parser.add_argument("--v3-dir", default=str(DEFAULT_V3_DIR))
    parser.add_argument("--bull-path", default=str(DEFAULT_BULL_PATH))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--force-history", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    payload = run_comparison(
        date_from=args.date_from,
        date_to=args.date_to,
        output_dir=Path(args.output_dir),
        model_source=Path(args.model_source),
        v3_dir=Path(args.v3_dir),
        bull_path=Path(args.bull_path),
        report_path=Path(args.report_path),
        force_history=bool(args.force_history),
    )
    print(
        json.dumps(
            {
                "date_from": payload["date_from"],
                "date_to": payload["date_to"],
                "old_candidate_rows": payload["old_candidate_rows"],
                "plan_candidate_rows": payload["plan_candidate_rows"],
                "calendar_days": payload["calendar_days"],
                "comparison_path": payload["comparison_path"],
                "report_path": payload["report_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
