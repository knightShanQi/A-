from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .database_source import load_env_file

load_env_file()

from .data import (
    _call_tushare_api,
    fetch_a_share_universe,
    fetch_daily_history,
    fetch_tushare_recent_trade_dates,
    fetch_tushare_stock_basic_all_statuses,
)
from .portfolio_backtester import PortfolioBacktestConfig, simulate_portfolio_from_candidates
from .store import (
    build_market_candidate_pool_store,
    build_market_daily_feature_store,
    load_incremental_market_snapshot_history,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".cache" / "market_full_backtests"
DEFAULT_COMPARE_OUTPUT_DIR = PROJECT_ROOT / ".cache" / "market_strategy_comparison"
ProgressCallback = Callable[[str, int, int, str], None]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_strategy_mode(value: str) -> str:
    mode = str(value or "all").strip().lower()
    if mode in {"old", "legacy", "old_strategies", "strategy1_2", "s12"}:
        return "old"
    if mode in {"new", "new_strategy", "strategy3", "s3", "3"}:
        return "strategy3"
    if mode in {"1", "strategy1", "s1"}:
        return "strategy1"
    if mode in {"2", "strategy2", "s2"}:
        return "strategy2"
    return "all"


def _strategy_mask(frame: pd.DataFrame, strategy_mode: str) -> pd.Series:
    if frame.empty or strategy_mode == "all" or "candidate_strategy" not in frame.columns:
        return pd.Series(True, index=frame.index)
    labels = frame["candidate_strategy"].fillna("").astype(str)
    lower = labels.str.lower()
    if strategy_mode == "old":
        return labels.str.contains("1", regex=False) | labels.str.contains("2", regex=False)
    if strategy_mode == "strategy1":
        return labels.str.contains("1", regex=False)
    if strategy_mode == "strategy2":
        return labels.str.contains("2", regex=False)
    if strategy_mode == "strategy3":
        return lower.str.contains("strategy3", regex=False) | labels.str.contains("策略3", regex=False)
    return pd.Series(True, index=frame.index)


def _trade_dates_between(date_from: str, date_to: str) -> list[str]:
    start_ts = pd.to_datetime(str(date_from), errors="coerce")
    end_ts = pd.to_datetime(str(date_to), errors="coerce")
    start_safe = start_ts.strftime("%Y%m%d") if pd.notna(start_ts) else str(date_from).replace("-", "")
    end_safe = end_ts.strftime("%Y%m%d") if pd.notna(end_ts) else str(date_to).replace("-", "")
    requested: list[str] = []
    try:
        calendar = _call_tushare_api(
            "trade_cal",
            params={"exchange": "", "start_date": start_safe, "end_date": end_safe, "is_open": 1},
            fields="cal_date,is_open",
        )
        if not calendar.empty and "cal_date" in calendar.columns:
            requested = sorted(calendar["cal_date"].astype(str).tolist())
    except Exception:
        requested = []
    if not requested:
        calendar_dates = []
        if pd.notna(start_ts) and pd.notna(end_ts):
            calendar_dates = [ts.strftime("%Y%m%d") for ts in pd.date_range(start_ts, end_ts, freq="D")]
        daily = _fetch_fast_daily_snapshots_for_dates(calendar_dates)
        if not daily.empty and "trade_date" in daily.columns:
            requested = sorted(pd.to_datetime(daily["trade_date"], errors="coerce").dropna().dt.strftime("%Y%m%d").unique().tolist())
    if not requested and getattr(fetch_tushare_recent_trade_dates, "__module__", "") != "a_share_predictor.data":
        requested = fetch_tushare_recent_trade_dates(end_date=end_safe, limit=180)
    dates: list[str] = []
    for value in requested:
        ts = pd.to_datetime(str(value), format="%Y%m%d", errors="coerce")
        if pd.isna(ts):
            continue
        if pd.notna(start_ts) and ts < start_ts:
            continue
        if pd.notna(end_ts) and ts > end_ts:
            continue
        dates.append(ts.strftime("%Y-%m-%d"))
    return dates


def _fetch_tushare_daily_range_paginated(
    start_date: str,
    end_date: str,
    *,
    fields: str,
    throttle_seconds: float = 1.25,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    cache_key = hashlib.sha1(f"{start_date}:{end_date}:{fields}".encode("utf-8")).hexdigest()[:16]
    cache_path = PROJECT_ROOT / ".cache" / f"tushare_daily_range_fast_{cache_key}.pkl"
    if cache_path.exists():
        try:
            cached = pd.read_pickle(cache_path)
            if isinstance(cached, pd.DataFrame) and not cached.empty:
                return cached.copy()
        except Exception:
            pass
    frames: list[pd.DataFrame] = []
    offset = 0
    limit = 6000
    request_count = 0
    completed = False
    while True:
        params = {
            "start_date": str(start_date).replace("-", ""),
            "end_date": str(end_date).replace("-", ""),
            "limit": limit,
            "offset": offset,
        }
        frame = pd.DataFrame()
        failed = False
        for attempt in range(1, 4):
            try:
                frame = _call_tushare_api("daily", params=params, fields=fields)
                break
            except Exception as exc:
                if attempt >= 3:
                    if frames and offset > 0 and "查询数据失败" in str(exc):
                        completed = True
                        if progress_callback is None:
                            print(
                                f"[fast_snapshot] stop {start_date}-{end_date} offset={offset}: {exc}",
                                flush=True,
                            )
                    else:
                        failed = True
                    if failed and progress_callback is None:
                        print(
                            f"[fast_snapshot] failed {start_date}-{end_date} offset={offset}: {exc}",
                            flush=True,
                        )
                    break
                time.sleep(1.5 * attempt)
        if failed:
            break
        request_count += 1
        if frame.empty:
            completed = True
            break
        frames.append(frame)
        if progress_callback is not None and (request_count == 1 or request_count % 10 == 0):
            progress_callback("fast_snapshot", request_count, 0, f"Loaded {request_count} paged Tushare daily chunks")
        elif progress_callback is None and (request_count == 1 or request_count % 5 == 0 or len(frame) < limit):
            print(
                f"[fast_snapshot] {start_date}-{end_date} page={request_count} rows={len(frame)} offset={offset}",
                flush=True,
            )
        if len(frame) < limit:
            completed = True
            break
        offset += limit
        time.sleep(max(float(throttle_seconds), 0.0))
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    if completed:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_pickle(cache_path)
        except Exception:
            pass
    return result


def _normalize_snapshot_history_for_fast_backtest(snapshot_history: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(snapshot_history, pd.DataFrame) or snapshot_history.empty:
        return pd.DataFrame()
    frame = snapshot_history.copy()
    if "symbol" not in frame.columns:
        return pd.DataFrame()
    frame["symbol"] = frame["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    frame["trade_date"] = pd.to_datetime(frame.get("trade_date", frame.get("date")), errors="coerce")
    frame = frame.dropna(subset=["symbol", "trade_date"]).copy()
    if frame.empty:
        return pd.DataFrame()
    rename_map = {
        "pct_chg": "change_pct",
        "turnover_rate": "turnover",
        "vol": "volume",
    }
    for source, target in rename_map.items():
        if target not in frame.columns and source in frame.columns:
            frame[target] = frame[source]
    for column in ["open", "high", "low", "close", "amount", "turnover", "change_pct", "volume"]:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["amount"] = frame["amount"].fillna(0.0)
    if "name" not in frame.columns:
        frame["name"] = frame["symbol"]
    if "industry" not in frame.columns:
        frame["industry"] = ""
    if "market" not in frame.columns:
        frame["market"] = ""
    frame["name"] = frame["name"].fillna(frame["symbol"]).astype(str)
    frame["industry"] = frame["industry"].fillna("").astype(str)
    frame["market"] = frame["market"].fillna("").astype(str)
    frame = frame.sort_values(["symbol", "trade_date"]).drop_duplicates(["symbol", "trade_date"], keep="last")
    return frame.reset_index(drop=True)


def _add_fast_rolling_metrics(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history
    frame = history.sort_values(["symbol", "trade_date"]).copy()
    grouped = frame.groupby("symbol", group_keys=False)
    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")

    for window in (5, 10, 20):
        frame[f"ma{window}"] = grouped["close"].transform(lambda s, w=window: s.rolling(w, min_periods=w).mean())
    for days, shift in ((3, 2), (5, 4), (10, 9), (15, 14), (20, 19)):
        base = grouped["close"].shift(shift)
        frame[f"ret_{days}d_pct"] = (close / base - 1.0) * 100.0
    frame["high_10"] = grouped["high"].transform(lambda s: s.rolling(10, min_periods=10).max())
    frame["low_10"] = grouped["low"].transform(lambda s: s.rolling(10, min_periods=10).min())
    frame["distance_to_high_10_pct"] = (frame["high_10"] - close) / frame["high_10"] * 100.0
    frame["max_gain_10_pct"] = (frame["high_10"] / frame["low_10"] - 1.0) * 100.0
    day_range = (high - low).replace(0, np.nan)
    frame["close_position_day"] = ((close - low) / day_range).clip(lower=0.0, upper=1.0).fillna(0.5)
    frame["upper_shadow_ratio"] = ((high - close) / day_range).clip(lower=0.0, upper=1.0).fillna(0.0)
    rolling_volume = grouped["volume"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    frame["volume_ratio_5"] = (volume / rolling_volume.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    amount = pd.to_numeric(frame["amount"], errors="coerce").fillna(0.0)
    turnover = pd.to_numeric(frame.get("turnover"), errors="coerce")
    if turnover.dropna().empty or float(turnover.fillna(0.0).abs().sum()) <= 0:
        rolling_amount = grouped["amount"].transform(lambda s: s.rolling(20, min_periods=5).mean())
        turnover = (amount / rolling_amount.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0) * 4.0
    frame["turnover"] = turnover.fillna(0.0).astype(float)

    pullback_days = pd.Series(0, index=frame.index, dtype=int)
    pullback_volume_decay = pd.Series(False, index=frame.index, dtype=bool)
    pullback_kept_ma10 = pd.Series(False, index=frame.index, dtype=bool)
    for _, group in frame.groupby("symbol", sort=False):
        indexes = group.index.to_list()
        closes = pd.to_numeric(group["close"], errors="coerce").to_numpy()
        vols = pd.to_numeric(group["volume"], errors="coerce").fillna(0.0).to_numpy()
        ma10 = pd.to_numeric(group["ma10"], errors="coerce").to_numpy()
        for offset, frame_index in enumerate(indexes):
            start = max(0, offset - 9)
            window_close = closes[start : offset + 1]
            if len(window_close) < 7 or np.isnan(window_close[:-1]).all():
                continue
            pre_today = window_close[:-1]
            peak_offset = int(np.nanargmax(pre_today))
            pullback_start = start + peak_offset + 1
            days = offset - pullback_start
            if days <= 0:
                continue
            volume_window = vols[pullback_start:offset]
            rise_volume = vols[start : pullback_start]
            ma_window = ma10[pullback_start:offset]
            close_window = closes[pullback_start:offset]
            pullback_days.loc[frame_index] = int(days)
            if len(volume_window) >= 3 and len(rise_volume) > 0:
                pullback_volume_decay.loc[frame_index] = bool(
                    volume_window[-1] <= volume_window[0]
                    and np.nanmean(volume_window) <= np.nanmean(rise_volume)
                )
            if len(ma_window) > 0:
                ma_gap = close_window - ma_window
                ma_gap = ma_gap[~np.isnan(ma_gap)]
                pullback_kept_ma10.loc[frame_index] = bool(len(ma_gap) > 0 and np.min(ma_gap) >= 0)
    frame["pullback_days"] = pullback_days
    frame["pullback_volume_decay"] = pullback_volume_decay
    frame["pullback_kept_ma10"] = pullback_kept_ma10
    return frame


def _fetch_fast_daily_snapshots_for_dates(
    trade_dates: list[str],
    *,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    if not trade_dates:
        return pd.DataFrame()
    fields = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
    safe_dates = sorted({str(value).replace("-", "") for value in trade_dates if str(value).strip()})
    if not safe_dates:
        return pd.DataFrame()
    start_ts = pd.to_datetime(safe_dates[0], format="%Y%m%d", errors="coerce")
    end_ts = pd.to_datetime(safe_dates[-1], format="%Y%m%d", errors="coerce")
    ranges: list[tuple[str, str]] = []
    if pd.notna(start_ts) and pd.notna(end_ts):
        cursor = pd.Timestamp(start_ts.year, start_ts.month, 1)
        while cursor <= end_ts:
            month_start = max(cursor, start_ts)
            month_end = min(cursor + pd.offsets.MonthEnd(0), end_ts)
            ranges.append((month_start.strftime("%Y%m%d"), month_end.strftime("%Y%m%d")))
            cursor = cursor + pd.offsets.MonthBegin(1)
    frames: list[pd.DataFrame] = []
    for index, (range_start, range_end) in enumerate(ranges or [(safe_dates[0], safe_dates[-1])], start=1):
        if progress_callback is not None:
            progress_callback("fast_snapshot_range", index, max(len(ranges), 1), f"Loading Tushare daily range {range_start}-{range_end}")
        frame = _fetch_tushare_daily_range_paginated(
            range_start,
            range_end,
            fields=fields,
            throttle_seconds=2.4,
            progress_callback=progress_callback,
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    if merged.empty:
        return pd.DataFrame()
    merged = merged[merged["trade_date"].astype(str).isin(set(safe_dates))].copy()
    if merged.empty:
        return pd.DataFrame()
    try:
        stock_basic = fetch_tushare_stock_basic_all_statuses()[["ts_code", "symbol", "name", "industry", "market"]].copy()
    except Exception:
        stock_basic = pd.DataFrame()
    if not stock_basic.empty:
        merged = merged.merge(stock_basic.drop_duplicates("ts_code"), on="ts_code", how="left")
    else:
        merged["symbol"] = merged["ts_code"].astype(str).str.split(".").str[0]
        merged["name"] = merged["symbol"]
        merged["industry"] = ""
        merged["market"] = ""
    merged["symbol"] = merged["symbol"].fillna(merged["ts_code"].astype(str).str.split(".").str[0]).astype(str).str.zfill(6)
    merged["name"] = merged["name"].fillna(merged["symbol"]).astype(str)
    merged["industry"] = merged["industry"].fillna("").astype(str)
    merged["market"] = merged["market"].fillna("").astype(str)
    merged["trade_date"] = pd.to_datetime(merged["trade_date"], format="%Y%m%d", errors="coerce")
    for column in ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]:
        merged[column] = pd.to_numeric(merged.get(column), errors="coerce")
    merged["change_pct"] = merged["pct_chg"]
    merged["volume"] = merged["vol"]
    merged["amount"] = merged["amount"].fillna(0.0) * 1000.0
    return merged.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def _add_fast_industry_metrics(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty or "industry" not in history.columns:
        return history
    frame = history.copy()
    frame["industry_name"] = frame["industry"].fillna("").astype(str).str.strip()
    scoped = frame[frame["industry_name"].ne("")].copy()
    if scoped.empty:
        frame["industry_ret_2d_pct"] = 0.0
        frame["industry_up_count"] = 0
        frame["industry_stock_count"] = 0
        frame["industry_rank_2d"] = np.nan
        frame["industry_top2d_flag"] = False
        return frame
    stats = (
        scoped.groupby(["trade_date", "industry_name"], as_index=False)
        .agg(
            industry_ret_2d_pct=("change_pct", lambda s: float(pd.to_numeric(s, errors="coerce").dropna().median()) if not pd.to_numeric(s, errors="coerce").dropna().empty else 0.0),
            industry_up_count=("change_pct", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
            industry_stock_count=("symbol", "count"),
        )
        .sort_values(["trade_date", "industry_ret_2d_pct"], ascending=[True, False])
    )
    stats["industry_rank_2d"] = stats.groupby("trade_date").cumcount() + 1
    stats["industry_top2d_flag"] = stats["industry_rank_2d"] <= 10
    return frame.merge(stats, on=["trade_date", "industry_name"], how="left")


def _main_board_non_st_mask(frame: pd.DataFrame) -> pd.Series:
    symbols = frame.get("symbol", pd.Series("", index=frame.index)).astype(str).str.zfill(6)
    names = frame.get("name", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper()
    market = frame.get("market", pd.Series("", index=frame.index)).fillna("").astype(str)
    symbol_main = symbols.str.match(r"^(000|001|002|003|600|601|603|605)")
    market_main = market.eq("") | market.str.contains("主板|Main|SSE|SZSE", case=False, regex=True)
    non_st = ~names.str.contains("ST|退", regex=True, na=False)
    return symbol_main & market_main & non_st


def _build_fast_strategy_candidates(history: pd.DataFrame, market_date: str, strategy_mode: str, top_k: int) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    market_ts = pd.to_datetime(market_date, errors="coerce")
    day = history[history["trade_date"].eq(market_ts)].copy()
    if day.empty:
        return pd.DataFrame()
    day = day[_main_board_non_st_mask(day)].copy()
    if day.empty:
        return pd.DataFrame()

    industry_up_count = pd.to_numeric(day.get("industry_up_count"), errors="coerce").fillna(0.0)
    industry_ret = pd.to_numeric(day.get("industry_ret_2d_pct"), errors="coerce").fillna(0.0)
    industry_top = day.get("industry_top2d_flag", pd.Series(False, index=day.index)).fillna(False).astype(bool)
    board_resonance = industry_top & (industry_up_count >= 3)
    close_pos = pd.to_numeric(day.get("close_position_day"), errors="coerce").fillna(0.5)
    upper_shadow = pd.to_numeric(day.get("upper_shadow_ratio"), errors="coerce").fillna(0.0)
    volume_ratio = pd.to_numeric(day.get("volume_ratio_5"), errors="coerce").fillna(1.0)
    amount = pd.to_numeric(day.get("amount"), errors="coerce").fillna(0.0)
    turnover = pd.to_numeric(day.get("turnover"), errors="coerce").fillna(0.0)
    change_pct = pd.to_numeric(day.get("change_pct"), errors="coerce").fillna(0.0)
    ret_3d = pd.to_numeric(day.get("ret_3d_pct"), errors="coerce")
    ret_5d = pd.to_numeric(day.get("ret_5d_pct"), errors="coerce")
    ret_15d = pd.to_numeric(day.get("ret_15d_pct"), errors="coerce")
    ret_20d = pd.to_numeric(day.get("ret_20d_pct"), errors="coerce")
    max_gain_10 = pd.to_numeric(day.get("max_gain_10_pct"), errors="coerce")
    distance_high = pd.to_numeric(day.get("distance_to_high_10_pct"), errors="coerce")

    late_risk = (
        ret_20d.clip(lower=0, upper=55).fillna(0) * 0.7
        + max_gain_10.clip(lower=0, upper=55).fillna(0) * 0.9
        + upper_shadow * 60
        + turnover.clip(lower=0, upper=18) * 0.8
    )
    resonance_score = (45 + industry_ret.clip(-4, 12) * 3.0 + industry_up_count.clip(0, 10) * 3.0 + industry_top.astype(float) * 8).clip(0, 100)
    launch_score = (
        48
        + ret_5d.clip(-4, 18).fillna(0) * 1.4
        + close_pos * 18
        + (1.6 - upper_shadow.clip(0, 0.6)) * 10
        + (volume_ratio.clip(0.6, 2.4) - 1.0) * 8
        - late_risk * 0.25
    ).clip(0, 100)
    quality_score = (
        launch_score * 0.32
        + resonance_score * 0.28
        + change_pct.clip(0, 9) * 2.2
        + ret_5d.clip(0, 18).fillna(0) * 1.0
        - late_risk * 0.20
    ).clip(0, 100)

    strategy1 = (
        ret_15d.between(10, 30, inclusive="both")
        & (day["close"] > day["ma5"])
        & (day["ma5"] > day["ma10"])
        & (day["ma10"] > day["ma20"])
        & pd.to_numeric(day.get("pullback_days"), errors="coerce").between(3, 6, inclusive="both")
        & day.get("pullback_volume_decay", pd.Series(False, index=day.index)).fillna(False).astype(bool)
        & day.get("pullback_kept_ma10", pd.Series(False, index=day.index)).fillna(False).astype(bool)
        & (change_pct > 2)
        & (amount > 2e8)
        & (turnover > 3)
        & (ret_20d < 35)
    )
    strategy2 = (
        (change_pct > 5)
        & (ret_3d > 10)
        & (ret_5d > 15)
        & (amount > 3e8)
        & (turnover > 5)
        & ((day["close"] >= day["high_10"]) | (distance_high < 2))
        & (max_gain_10 < 40)
        & board_resonance
    )
    strategy3 = (
        (day["close"] > day["ma10"])
        & (day["ma5"] >= day["ma20"] * 0.995)
        & ret_5d.between(2, 18, inclusive="both")
        & ret_20d.between(-5, 32, inclusive="both")
        & (max_gain_10 < 34)
        & (amount >= 1.2e8)
        & (turnover >= 1.8)
        & (change_pct > 0.8)
        & (upper_shadow <= 0.26)
        & (close_pos >= 0.50)
        & ((resonance_score >= 56) | ((industry_ret >= 0.8) & (industry_up_count >= 2)) | (launch_score >= 62))
        & (quality_score >= 56)
        & (late_risk < 70)
        & (~strategy1)
        & (~strategy2)
    )

    labeled: list[pd.DataFrame] = []
    if strategy1.any():
        s1 = day.loc[strategy1].copy()
        s1["candidate_strategy"] = "策略1"
        s1["candidate_priority"] = (quality_score.loc[s1.index] + 6).round(4)
        labeled.append(s1)
    if strategy2.any():
        s2 = day.loc[strategy2].copy()
        s2["candidate_strategy"] = "策略2"
        s2["candidate_priority"] = (quality_score.loc[s2.index] + 8).round(4)
        labeled.append(s2)
    if strategy3.any():
        s3 = day.loc[strategy3].copy()
        s3["candidate_strategy"] = "strategy3"
        s3["candidate_priority"] = quality_score.loc[s3.index].round(4)
        labeled.append(s3)
    if not labeled:
        return pd.DataFrame()
    candidates = pd.concat(labeled, ignore_index=True)
    candidates = candidates.loc[_strategy_mask(candidates, strategy_mode)].copy()
    if candidates.empty:
        return candidates
    candidates["latest_price"] = pd.to_numeric(candidates["close"], errors="coerce").fillna(0.0)
    candidates["strategy_rank"] = pd.to_numeric(candidates["candidate_priority"], errors="coerce").fillna(0.0)
    columns = [
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
    ]
    candidates = candidates[[column for column in columns if column in candidates.columns]].copy()
    return candidates.sort_values(["candidate_priority", "amount"], ascending=False).head(max(int(top_k), 1)).reset_index(drop=True)


def _prepare_fast_snapshot_history(
    date_from: str,
    date_to: str,
    trade_dates: list[str],
    *,
    force_rebuild: bool,
    progress_callback: ProgressCallback | None,
) -> pd.DataFrame:
    if not trade_dates:
        return pd.DataFrame()
    lookback_sessions = max(len(trade_dates) + 45, 80)
    if progress_callback is not None:
        progress_callback("fast_prepare", 0, len(trade_dates), f"Loading {lookback_sessions} market snapshots once")
    start_floor = pd.to_datetime(date_from, errors="coerce") - pd.Timedelta(days=max(90, min(160, lookback_sessions + 35)))
    end_ts = pd.to_datetime(date_to, errors="coerce")
    preload_dates: list[str] = []
    if pd.notna(start_floor) and pd.notna(end_ts):
        preload_dates = [
            ts.strftime("%Y%m%d")
            for ts in pd.date_range(start_floor, end_ts, freq="D")
        ]
    snapshot_history = _fetch_fast_daily_snapshots_for_dates(
        preload_dates,
        progress_callback=progress_callback,
    )
    if snapshot_history.empty:
        snapshot_history = load_incremental_market_snapshot_history(
            date_to,
            lookback_sessions=lookback_sessions,
            force_rebuild=force_rebuild,
            progress_callback=progress_callback,
        )
    history = _normalize_snapshot_history_for_fast_backtest(snapshot_history)
    if history.empty:
        return history
    start_floor = pd.to_datetime(date_from, errors="coerce") - pd.Timedelta(days=80)
    end_ts = pd.to_datetime(date_to, errors="coerce")
    if pd.notna(start_floor):
        history = history[history["trade_date"] >= start_floor].copy()
    if pd.notna(end_ts):
        history = history[history["trade_date"] <= end_ts].copy()
    history = _add_fast_rolling_metrics(history)
    history = _add_fast_industry_metrics(history)
    return history.reset_index(drop=True)


def _evaluate_forward_return(symbol: str, signal_date: str, horizon_days: int) -> dict[str, object]:
    try:
        daily = fetch_daily_history(symbol=symbol, start_date="20240101")
    except Exception:
        daily = pd.DataFrame()
    if daily.empty or "date" not in daily.columns:
        return {"forward_available": False}
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    signal_ts = pd.to_datetime(signal_date, errors="coerce")
    if pd.isna(signal_ts):
        return {"forward_available": False}
    signal_positions = frame.index[frame["date"] <= signal_ts].tolist()
    if not signal_positions:
        return {"forward_available": False}
    signal_pos = int(signal_positions[-1])
    entry_pos = signal_pos + 1
    exit_pos = signal_pos + max(int(horizon_days), 1)
    if entry_pos >= len(frame):
        return {"forward_available": False}
    entry_price = _safe_float(frame.loc[entry_pos, "open"], _safe_float(frame.loc[entry_pos, "close"], 0.0))
    if entry_price <= 0:
        return {"forward_available": False}
    holding_returns: dict[str, object] = {}
    for hold_days in (1, 3, 5):
        hold_exit_pos = signal_pos + hold_days
        if hold_exit_pos >= len(frame):
            holding_returns[f"hold_{hold_days}d_available"] = False
            holding_returns[f"hold_{hold_days}d_return"] = float("nan")
            continue
        hold_exit_price = _safe_float(frame.loc[hold_exit_pos, "close"], 0.0)
        holding_returns[f"hold_{hold_days}d_available"] = bool(hold_exit_price > 0)
        holding_returns[f"hold_{hold_days}d_return"] = (
            hold_exit_price / entry_price - 1.0 if hold_exit_price > 0 else float("nan")
        )
    if exit_pos >= len(frame):
        return {
            "forward_available": False,
            "entry_date": pd.Timestamp(frame.loc[entry_pos, "date"]).strftime("%Y-%m-%d"),
            "entry_price": round(entry_price, 4),
            **holding_returns,
        }
    exit_price = _safe_float(frame.loc[exit_pos, "close"], 0.0)
    if exit_price <= 0:
        return {
            "forward_available": False,
            "entry_date": pd.Timestamp(frame.loc[entry_pos, "date"]).strftime("%Y-%m-%d"),
            "entry_price": round(entry_price, 4),
            **holding_returns,
        }
    window = frame.iloc[entry_pos : exit_pos + 1]
    max_high = _safe_float(window["high"].max() if "high" in window.columns else window["close"].max(), exit_price)
    min_low = _safe_float(window["low"].min() if "low" in window.columns else window["close"].min(), exit_price)
    return {
        "forward_available": True,
        "entry_date": pd.Timestamp(frame.loc[entry_pos, "date"]).strftime("%Y-%m-%d"),
        "exit_date": pd.Timestamp(frame.loc[exit_pos, "date"]).strftime("%Y-%m-%d"),
        "entry_price": round(entry_price, 4),
        "exit_price": round(exit_price, 4),
        "forward_return": exit_price / entry_price - 1.0,
        "max_high_return": max_high / entry_price - 1.0,
        "max_drawdown": min_low / entry_price - 1.0,
        **holding_returns,
    }


def _build_fast_history_lookup(history: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if not isinstance(history, pd.DataFrame) or history.empty:
        return {}
    lookup: dict[str, pd.DataFrame] = {}
    keep_columns = [column for column in ["trade_date", "open", "high", "low", "close"] if column in history.columns]
    for symbol, group in history[["symbol", *keep_columns]].drop_duplicates(["symbol", "trade_date"]).groupby("symbol", sort=False):
        lookup[str(symbol).zfill(6)] = group.sort_values("trade_date").reset_index(drop=True)
    return lookup


def _evaluate_forward_return_from_lookup(
    history_lookup: dict[str, pd.DataFrame],
    symbol: str,
    signal_date: str,
    horizon_days: int,
) -> dict[str, object]:
    frame = history_lookup.get(str(symbol).zfill(6), pd.DataFrame())
    if frame.empty:
        return {"forward_available": False}
    signal_ts = pd.to_datetime(signal_date, errors="coerce")
    if pd.isna(signal_ts):
        return {"forward_available": False}
    signal_positions = frame.index[pd.to_datetime(frame["trade_date"], errors="coerce") <= signal_ts].tolist()
    if not signal_positions:
        return {"forward_available": False}
    signal_pos = int(signal_positions[-1])
    entry_pos = signal_pos + 1
    exit_pos = signal_pos + max(int(horizon_days), 1)
    if entry_pos >= len(frame):
        return {"forward_available": False}
    entry_price = _safe_float(frame.loc[entry_pos, "open"], _safe_float(frame.loc[entry_pos, "close"], 0.0))
    if entry_price <= 0:
        return {"forward_available": False}
    holding_returns: dict[str, object] = {}
    for hold_days in (1, 3, 5):
        hold_exit_pos = signal_pos + hold_days
        if hold_exit_pos >= len(frame):
            holding_returns[f"hold_{hold_days}d_available"] = False
            holding_returns[f"hold_{hold_days}d_return"] = float("nan")
            continue
        hold_exit_price = _safe_float(frame.loc[hold_exit_pos, "close"], 0.0)
        holding_returns[f"hold_{hold_days}d_available"] = bool(hold_exit_price > 0)
        holding_returns[f"hold_{hold_days}d_return"] = (
            hold_exit_price / entry_price - 1.0 if hold_exit_price > 0 else float("nan")
        )
    if exit_pos >= len(frame):
        return {
            "forward_available": False,
            "entry_date": pd.Timestamp(frame.loc[entry_pos, "trade_date"]).strftime("%Y-%m-%d"),
            "entry_price": round(entry_price, 4),
            **holding_returns,
        }
    exit_price = _safe_float(frame.loc[exit_pos, "close"], 0.0)
    if exit_price <= 0:
        return {
            "forward_available": False,
            "entry_date": pd.Timestamp(frame.loc[entry_pos, "trade_date"]).strftime("%Y-%m-%d"),
            "entry_price": round(entry_price, 4),
            **holding_returns,
        }
    window = frame.iloc[entry_pos : exit_pos + 1]
    max_high = _safe_float(window["high"].max() if "high" in window.columns else window["close"].max(), exit_price)
    min_low = _safe_float(window["low"].min() if "low" in window.columns else window["close"].min(), exit_price)
    return {
        "forward_available": True,
        "entry_date": pd.Timestamp(frame.loc[entry_pos, "trade_date"]).strftime("%Y-%m-%d"),
        "exit_date": pd.Timestamp(frame.loc[exit_pos, "trade_date"]).strftime("%Y-%m-%d"),
        "entry_price": round(entry_price, 4),
        "exit_price": round(exit_price, 4),
        "forward_return": exit_price / entry_price - 1.0,
        "max_high_return": max_high / entry_price - 1.0,
        "max_drawdown": min_low / entry_price - 1.0,
        **holding_returns,
    }


def _average_holding_returns(results: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {}
    for hold_days in (1, 3, 5):
        column = f"hold_{hold_days}d_return"
        if column not in results.columns:
            summary[f"avg_hold_{hold_days}d_return"] = 0.0
            summary[f"hold_{hold_days}d_sample_count"] = 0
            continue
        values = pd.to_numeric(results[column], errors="coerce").dropna()
        summary[f"avg_hold_{hold_days}d_return"] = round(float(values.mean()), 6) if not values.empty else 0.0
        summary[f"hold_{hold_days}d_sample_count"] = int(len(values))
    return summary


def _build_portfolio_history_frame(
    results: pd.DataFrame,
    *,
    fast_history: pd.DataFrame,
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    if not fast_history.empty:
        keep_columns = [column for column in ["trade_date", "symbol", "open", "high", "low", "close"] if column in fast_history.columns]
        return fast_history[keep_columns].drop_duplicates(["symbol", "trade_date"]).copy()

    frames: list[pd.DataFrame] = []
    for symbol in sorted(results["symbol"].astype(str).str.zfill(6).unique().tolist()):
        try:
            history = fetch_daily_history(symbol)
        except Exception:
            continue
        if not isinstance(history, pd.DataFrame) or history.empty:
            continue
        frame = history.copy()
        trade_date = frame.get("trade_date", frame.get("date"))
        if trade_date is None:
            continue
        frame["trade_date"] = pd.to_datetime(trade_date, errors="coerce")
        frame["symbol"] = str(symbol).zfill(6)
        for column in ["open", "high", "low", "close"]:
            frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
        frames.append(frame[[column for column in ["trade_date", "symbol", "open", "high", "low", "close"] if column in frame.columns]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(["symbol", "trade_date"]).reset_index(drop=True)


def _portfolio_summary(result: object) -> dict[str, object]:
    summary = getattr(result, "summary", {}) if result is not None else {}
    return {
        "ending_equity": round(float(summary.get("ending_equity", 0.0)), 6),
        "cumulative_return": round(float(summary.get("cumulative_return", 0.0)), 6),
        "annualized_return": round(float(summary.get("annualized_return", 0.0)), 6),
        "max_drawdown": round(float(summary.get("max_drawdown", 0.0)), 6),
        "portfolio_trade_count": int(summary.get("trade_count", 0)),
        "portfolio_win_rate": round(float(summary.get("win_rate", 0.0)), 4),
        "portfolio_avg_net_return": round(float(summary.get("avg_net_return", 0.0)), 6),
    }


def _summarize_results(results: pd.DataFrame, positive_return: float) -> dict[str, object]:
    if results.empty:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "target_hit_rate": 0.0,
            "avg_forward_return": 0.0,
            "avg_max_high_return": 0.0,
            "avg_max_drawdown": 0.0,
            **_average_holding_returns(results),
            "strategy_breakdown": [],
        }
    available = results[results["forward_available"].astype(bool)].copy()
    if available.empty:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "target_hit_rate": 0.0,
            "avg_forward_return": 0.0,
            "avg_max_high_return": 0.0,
            "avg_max_drawdown": 0.0,
            **_average_holding_returns(results),
            "strategy_breakdown": [],
        }
    available["win"] = pd.to_numeric(available["forward_return"], errors="coerce").fillna(0.0) > 0
    available["target_hit"] = pd.to_numeric(available["max_high_return"], errors="coerce").fillna(0.0) >= float(positive_return)
    strategy_breakdown = []
    if "candidate_strategy" in available.columns:
        grouped = available.groupby("candidate_strategy", dropna=False)
        for strategy, group in grouped:
            strategy_breakdown.append(
                {
                    "candidate_strategy": str(strategy),
                    "trade_count": int(len(group)),
                    "win_rate": round(float(group["win"].mean()), 4),
                    "target_hit_rate": round(float(group["target_hit"].mean()), 4),
                    "avg_forward_return": round(float(group["forward_return"].mean()), 6),
                }
            )
    return {
        "trade_count": int(len(available)),
        "win_rate": round(float(available["win"].mean()), 4),
        "target_hit_rate": round(float(available["target_hit"].mean()), 4),
        "avg_forward_return": round(float(available["forward_return"].mean()), 6),
        "avg_max_high_return": round(float(available["max_high_return"].mean()), 6),
        "avg_max_drawdown": round(float(available["max_drawdown"].mean()), 6),
        **_average_holding_returns(results),
        "strategy_breakdown": strategy_breakdown,
    }


def run_full_market_backtest(
    *,
    date_from: str,
    date_to: str,
    horizon_days: int = 3,
    positive_return: float = 0.10,
    strategy_mode: str = "all",
    top_k: int = 50,
    output_dir: str | Path | None = None,
    force_rebuild: bool = False,
    progress_callback: ProgressCallback | None = None,
    max_workers: int = 8,
    fast_strategy_backtest: bool = False,
) -> dict[str, object]:
    output_path = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)
    strategy_mode = _normalize_strategy_mode(strategy_mode)
    universe = fetch_a_share_universe()[["symbol", "name"]].drop_duplicates("symbol").copy()
    rows: list[dict[str, object]] = []
    trade_dates = _trade_dates_between(date_from, date_to)
    total_dates = max(len(trade_dates), 1)
    fast_history = (
        _prepare_fast_snapshot_history(
            date_from,
            date_to,
            trade_dates,
            force_rebuild=force_rebuild,
            progress_callback=progress_callback,
        )
        if fast_strategy_backtest
        else pd.DataFrame()
    )
    fast_history_lookup = _build_fast_history_lookup(fast_history) if not fast_history.empty else {}
    if progress_callback is not None:
        progress_callback("prepare", 0, total_dates, f"Loaded {len(universe)} symbols and {len(trade_dates)} trade dates")
    for date_index, market_date in enumerate(trade_dates, start=1):
        if not fast_history.empty:
            if progress_callback is not None:
                progress_callback("candidate_pool", date_index - 1, total_dates, f"Fast screening candidates for {market_date}")
            candidates = _build_fast_strategy_candidates(fast_history, market_date, strategy_mode, top_k)
        else:
            if progress_callback is not None:
                progress_callback("feature_store", date_index - 1, total_dates, f"Building feature store for {market_date}")
            feature_store = build_market_daily_feature_store(
                universe,
                market_date,
                force_rebuild=force_rebuild,
            )
            if progress_callback is not None:
                progress_callback("candidate_pool", date_index - 1, total_dates, f"Screening candidates for {market_date}")
            candidates = build_market_candidate_pool_store(
                universe,
                market_date,
                feature_store=feature_store,
                force_rebuild=force_rebuild,
            )
        if candidates.empty:
            continue
        candidates = candidates.loc[_strategy_mask(candidates, strategy_mode)].copy()
        if candidates.empty:
            continue
        rank_columns = [column for column in ["candidate_priority", "strategy_rank", "quant_score", "amount"] if column in candidates.columns]
        if rank_columns:
            candidates = candidates.sort_values(rank_columns, ascending=False)
        candidates = candidates.head(max(int(top_k), 1)).copy()
        candidate_records = candidates.to_dict("records")
        if fast_history_lookup:
            evaluated = [
                (
                    row,
                    _evaluate_forward_return_from_lookup(
                        fast_history_lookup,
                        str(row.get("symbol", "")).zfill(6),
                        market_date,
                        horizon_days,
                    ),
                )
                for row in candidate_records
            ]
        else:
            worker_count = max(1, min(int(max_workers), len(candidate_records) or 1))
            evaluated = []
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(_evaluate_forward_return, str(row.get("symbol", "")).zfill(6), market_date, horizon_days): row
                    for row in candidate_records
                }
                for future in as_completed(futures):
                    row = futures[future]
                    try:
                        forward = future.result()
                    except Exception:
                        forward = {"forward_available": False}
                    evaluated.append((row, forward))
        for row, forward in evaluated:
            symbol = str(row.get("symbol", "")).zfill(6)
            rows.append(
                {
                    "market_date": market_date,
                    "symbol": symbol,
                    "name": str(row.get("name", symbol)),
                    "candidate_strategy": str(row.get("candidate_strategy", "")),
                    "candidate_priority": _safe_float(row.get("candidate_priority"), 0.0),
                    "latest_price": _safe_float(row.get("latest_price"), 0.0),
                    "change_pct": _safe_float(row.get("change_pct"), 0.0),
                    "amount": _safe_float(row.get("amount"), 0.0),
                    **forward,
                }
            )
        if progress_callback is not None:
            progress_callback("forward_eval", date_index, total_dates, f"Finished {market_date}, accumulated {len(rows)} rows")
    result_frame = pd.DataFrame(rows)
    portfolio_history = _build_portfolio_history_frame(result_frame, fast_history=fast_history)
    portfolio_result = simulate_portfolio_from_candidates(
        result_frame,
        portfolio_history,
        config=PortfolioBacktestConfig(
            max_positions=max(int(top_k), 1),
            holding_days=max(int(horizon_days), 1),
        ),
    )
    summary = {
        "date_from": str(date_from),
        "date_to": str(date_to),
        "horizon_days": int(horizon_days),
        "positive_return": float(positive_return),
        "strategy_mode": strategy_mode,
        "top_k": int(top_k),
        "fast_strategy_backtest": bool(fast_strategy_backtest and not fast_history.empty),
        "trading_day_count": int(len(set(result_frame["market_date"])) if not result_frame.empty else 0),
        **_summarize_results(result_frame, positive_return),
        **_portfolio_summary(portfolio_result),
    }
    if not result_frame.empty:
        result_frame = result_frame.sort_values(["market_date", "candidate_priority"], ascending=[True, False]).reset_index(drop=True)
    results_path = output_path / "trade_like_results.csv"
    portfolio_nav_path = output_path / "portfolio_daily_nav.csv"
    portfolio_trades_path = output_path / "portfolio_trades.csv"
    summary_path = output_path / "summary.json"
    result_frame.to_csv(results_path, index=False, encoding="utf-8-sig")
    portfolio_result.daily_nav.to_csv(portfolio_nav_path, index=False, encoding="utf-8-sig")
    portfolio_result.trades.to_csv(portfolio_trades_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if progress_callback is not None:
        progress_callback("write_outputs", total_dates, total_dates, f"Saved summary to {summary_path}")
    return {
        "summary": summary,
        "results_path": str(results_path),
        "portfolio_nav_path": str(portfolio_nav_path),
        "portfolio_trades_path": str(portfolio_trades_path),
        "summary_path": str(summary_path),
        "results": result_frame,
        "portfolio_daily_nav": portfolio_result.daily_nav,
        "portfolio_trades": portfolio_result.trades,
    }


def load_latest_full_market_backtest(
    output_dir: str | Path | None = None,
    *,
    result_limit: int = 50,
) -> dict[str, object]:
    output_path = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    summary_path = output_path / "summary.json"
    results_path = output_path / "trade_like_results.csv"
    portfolio_nav_path = output_path / "portfolio_daily_nav.csv"
    portfolio_trades_path = output_path / "portfolio_trades.csv"
    if not summary_path.exists():
        return {}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    results = pd.DataFrame()
    portfolio_daily_nav = pd.DataFrame()
    portfolio_trades = pd.DataFrame()
    if results_path.exists():
        try:
            results = pd.read_csv(results_path, encoding="utf-8-sig")
        except Exception:
            results = pd.DataFrame()
    if portfolio_nav_path.exists():
        try:
            portfolio_daily_nav = pd.read_csv(portfolio_nav_path, encoding="utf-8-sig")
        except Exception:
            portfolio_daily_nav = pd.DataFrame()
    if portfolio_trades_path.exists():
        try:
            portfolio_trades = pd.read_csv(portfolio_trades_path, encoding="utf-8-sig")
        except Exception:
            portfolio_trades = pd.DataFrame()
    if isinstance(results, pd.DataFrame) and not results.empty and result_limit > 0:
        results = results.head(int(result_limit)).copy()
    return {
        "summary": summary,
        "summary_path": str(summary_path),
        "results_path": str(results_path),
        "portfolio_nav_path": str(portfolio_nav_path),
        "portfolio_trades_path": str(portfolio_trades_path),
        "results": results,
        "portfolio_daily_nav": portfolio_daily_nav,
        "portfolio_trades": portfolio_trades,
    }


def run_strategy_comparison_backtest(
    *,
    date_from: str,
    date_to: str,
    horizon_days: int = 5,
    positive_return: float = 0.10,
    top_k: int = 50,
    output_dir: str | Path | None = None,
    force_rebuild: bool = False,
    progress_callback: ProgressCallback | None = None,
    max_workers: int = 12,
    fast_strategy_backtest: bool = True,
) -> dict[str, object]:
    output_path = Path(output_dir) if output_dir is not None else DEFAULT_COMPARE_OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)
    modes = [("old", "old_strategy1_2"), ("strategy3", "new_strategy3")]
    summaries: list[dict[str, object]] = []
    result_frames: list[pd.DataFrame] = []
    for index, (mode, folder_name) in enumerate(modes, start=1):
        if progress_callback is not None:
            progress_callback("compare", index - 1, len(modes), f"Running {mode} comparison backtest")
        payload = run_full_market_backtest(
            date_from=date_from,
            date_to=date_to,
            horizon_days=horizon_days,
            positive_return=positive_return,
            strategy_mode=mode,
            top_k=top_k,
            output_dir=output_path / folder_name,
            force_rebuild=force_rebuild if index == 1 else False,
            progress_callback=progress_callback,
            max_workers=max_workers,
            fast_strategy_backtest=fast_strategy_backtest,
        )
        summary = dict(payload["summary"])
        summary["comparison_label"] = folder_name
        summaries.append(summary)
        results = payload.get("results")
        if isinstance(results, pd.DataFrame) and not results.empty:
            copied = results.copy()
            copied["strategy_mode"] = mode
            copied["comparison_label"] = folder_name
            result_frames.append(copied)

    comparison = pd.DataFrame(summaries)
    combined = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    comparison_path = output_path / "strategy_comparison_summary.csv"
    combined_path = output_path / "strategy_comparison_trades.csv"
    summary_path = output_path / "summary.json"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    payload = {
        "date_from": str(date_from),
        "date_to": str(date_to),
        "horizon_days": int(horizon_days),
        "positive_return": float(positive_return),
        "top_k": int(top_k),
        "summaries": summaries,
        "comparison_path": str(comparison_path),
        "combined_results_path": str(combined_path),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if progress_callback is not None:
        progress_callback("compare_done", len(modes), len(modes), f"Saved comparison to {comparison_path}")
    return {
        **payload,
        "summary_path": str(summary_path),
        "comparison": comparison,
        "results": combined,
    }


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-market strategy backtest for A-share focus board rules.")
    parser.add_argument("--date-from", required=True)
    parser.add_argument("--date-to", required=True)
    parser.add_argument("--horizon-days", type=int, default=3)
    parser.add_argument("--positive-return", type=float, default=0.10)
    parser.add_argument(
        "--strategy-mode",
        default="all",
        choices=["all", "old", "legacy", "strategy1", "strategy2", "strategy3", "new", "s1", "s2", "s3", "1", "2", "3"],
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--fast-strategy-backtest", action="store_true")
    parser.add_argument("--compare-old-new", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)

    def print_progress(phase: str, completed: int, total: int, message: str) -> None:
        print(f"[{phase}] {completed}/{total} {message}", flush=True)

    if bool(args.compare_old_new):
        payload = run_strategy_comparison_backtest(
            date_from=args.date_from,
            date_to=args.date_to,
            horizon_days=args.horizon_days,
            positive_return=args.positive_return,
            top_k=args.top_k,
            output_dir=args.output_dir,
            force_rebuild=bool(args.force_rebuild),
            max_workers=int(args.max_workers),
            fast_strategy_backtest=True if not bool(args.fast_strategy_backtest) else bool(args.fast_strategy_backtest),
            progress_callback=print_progress,
        )
        print(json.dumps({"summaries": payload["summaries"], "summary_path": payload["summary_path"]}, ensure_ascii=False, indent=2))
        return
    payload = run_full_market_backtest(
        date_from=args.date_from,
        date_to=args.date_to,
        horizon_days=args.horizon_days,
        positive_return=args.positive_return,
        strategy_mode=args.strategy_mode,
        top_k=args.top_k,
        output_dir=args.output_dir,
        force_rebuild=bool(args.force_rebuild),
        max_workers=int(args.max_workers),
        fast_strategy_backtest=bool(args.fast_strategy_backtest),
        progress_callback=print_progress,
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
