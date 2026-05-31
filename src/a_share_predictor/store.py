from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .data import (
    fetch_daily_history,
    fetch_tushare_daily_snapshot,
    fetch_tushare_recent_trade_dates,
    fetch_tushare_stock_basic,
    fetch_tushare_stock_basic_all_statuses,
    filter_point_in_time_a_share_universe,
)
from .features import build_daily_features, latest_snapshot as build_latest_snapshot
from .modeling import MODEL_FEATURE_COLUMNS, _prepare_live_feature_frame
from .quant import evaluate_quant_signal
from .stages import classify_stage, main_rise_start_score, stage_numeric_score
from .strategy import build_trading_rule_context


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / ".cache"
MARKET_SNAPSHOT_HISTORY_STORE_VERSION = 1
MARKET_DAILY_FEATURE_STORE_VERSION = 4
MARKET_CANDIDATE_POOL_STORE_VERSION = 9
MARKET_DYNAMIC_FALLBACK_STORE_VERSION = 4
FULL_MARKET_HISTORY_START = "20240701"
MARKET_FEATURE_SNAPSHOT_LOOKBACK = 140
RULE_BASED_CANDIDATE_POOL_SIZE = 120
DYNAMIC_FALLBACK_POOL_SIZE = 50
DYNAMIC_FALLBACK_ANALYSIS_BUFFER_SIZE = 80
MIN_FOCUS_CONSECUTIVE_UP_DAYS = 3
DEFAULT_REPLAY_HORIZON_DAYS = 3
DEFAULT_REPLAY_POSITIVE_RETURN = 0.10
DEFAULT_REPLAY_RANKING_BY = "关注分数"
DEFAULT_REPLAY_BOARD_SIZE = 50
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

ProgressCallback = Callable[[str, int, int, str], None]
FEATURE_BUILD_PROGRESS_UPDATES = 60
SNAPSHOT_LOAD_PROGRESS_UPDATES = 24


def _should_report_progress(
    index: int,
    total: int,
    *,
    desired_updates: int,
    min_interval: int = 1,
) -> bool:
    if total <= 0:
        return False
    if index in {1, total}:
        return True
    interval = max(min_interval, total // max(desired_updates, 1))
    if total <= desired_updates:
        return True
    return index % interval == 0


def _store_cache_path(prefix: str, version: int, market_data_date: str | None) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_date = str(market_data_date or "unknown").replace("-", "")
    return CACHE_DIR / f"{prefix}_v{version}_{safe_date}.pkl"


def _read_dataframe_store(path: Path, *, version: int, market_data_date: str | None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    meta = payload.get("meta", {})
    data = payload.get("data")
    if not isinstance(data, pd.DataFrame) or data.empty:
        return None
    if meta.get("cache_version") != version:
        return None
    if meta.get("market_data_date") != market_data_date:
        return None
    return data


@lru_cache(maxsize=16)
def _read_dataframe_store_cached(path_str: str, version: int, market_data_date: str | None) -> pd.DataFrame | None:
    return _read_dataframe_store(Path(path_str), version=version, market_data_date=market_data_date)


def _clear_store_read_caches() -> None:
    _read_dataframe_store_cached.cache_clear()
    _get_market_daily_feature_lookup_cached.cache_clear()
    _get_market_daily_feature_row_cached.cache_clear()


def _write_dataframe_store(path: Path, *, version: int, market_data_date: str | None, data: pd.DataFrame) -> None:
    payload = {
        "meta": {
            "cache_version": version,
            "market_data_date": market_data_date,
            "row_count": int(len(data)),
        },
        "data": data,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    _clear_store_read_caches()


def market_daily_feature_store_path(market_data_date: str | None) -> Path:
    return _store_cache_path("market_daily_feature_store", MARKET_DAILY_FEATURE_STORE_VERSION, market_data_date)


def market_snapshot_history_store_path(market_data_date: str | None) -> Path:
    return _store_cache_path("market_snapshot_history_store", MARKET_SNAPSHOT_HISTORY_STORE_VERSION, market_data_date)


def market_candidate_pool_store_path(market_data_date: str | None) -> Path:
    return _store_cache_path("market_candidate_pool_store", MARKET_CANDIDATE_POOL_STORE_VERSION, market_data_date)


def market_dynamic_fallback_store_path(market_data_date: str | None) -> Path:
    return _store_cache_path("market_dynamic_fallback_store", MARKET_DYNAMIC_FALLBACK_STORE_VERSION, market_data_date)


def read_market_daily_feature_store(market_data_date: str | None) -> pd.DataFrame | None:
    data = _read_dataframe_store_cached(
        str(market_daily_feature_store_path(market_data_date)),
        version=MARKET_DAILY_FEATURE_STORE_VERSION,
        market_data_date=market_data_date,
    )
    return None if data is None else data.copy()


def read_market_snapshot_history_store(market_data_date: str | None) -> pd.DataFrame | None:
    data = _read_dataframe_store_cached(
        str(market_snapshot_history_store_path(market_data_date)),
        version=MARKET_SNAPSHOT_HISTORY_STORE_VERSION,
        market_data_date=market_data_date,
    )
    return None if data is None else data.copy()


def read_market_candidate_pool_store(market_data_date: str | None) -> pd.DataFrame | None:
    data = _read_dataframe_store_cached(
        str(market_candidate_pool_store_path(market_data_date)),
        version=MARKET_CANDIDATE_POOL_STORE_VERSION,
        market_data_date=market_data_date,
    )
    return None if data is None else data.copy()


def read_market_dynamic_fallback_store(market_data_date: str | None) -> pd.DataFrame | None:
    data = _read_dataframe_store_cached(
        str(market_dynamic_fallback_store_path(market_data_date)),
        version=MARKET_DYNAMIC_FALLBACK_STORE_VERSION,
        market_data_date=market_data_date,
    )
    return None if data is None else data.copy()


def _write_market_daily_feature_store(feature_store: pd.DataFrame, market_data_date: str | None) -> None:
    _write_dataframe_store(
        market_daily_feature_store_path(market_data_date),
        version=MARKET_DAILY_FEATURE_STORE_VERSION,
        market_data_date=market_data_date,
        data=feature_store,
    )


def _write_market_snapshot_history_store(snapshot_history: pd.DataFrame, market_data_date: str | None) -> None:
    _write_dataframe_store(
        market_snapshot_history_store_path(market_data_date),
        version=MARKET_SNAPSHOT_HISTORY_STORE_VERSION,
        market_data_date=market_data_date,
        data=snapshot_history,
    )


def _write_market_candidate_pool_store(candidate_pool: pd.DataFrame, market_data_date: str | None) -> None:
    _write_dataframe_store(
        market_candidate_pool_store_path(market_data_date),
        version=MARKET_CANDIDATE_POOL_STORE_VERSION,
        market_data_date=market_data_date,
        data=candidate_pool,
    )


def _write_market_dynamic_fallback_store(pool_df: pd.DataFrame, market_data_date: str | None) -> None:
    _write_dataframe_store(
        market_dynamic_fallback_store_path(market_data_date),
        version=MARKET_DYNAMIC_FALLBACK_STORE_VERSION,
        market_data_date=market_data_date,
        data=pool_df,
    )


@lru_cache(maxsize=16)
def _get_market_daily_feature_lookup_cached(market_data_date: str | None) -> dict[str, dict[str, object]]:
    data = _read_dataframe_store_cached(
        str(market_daily_feature_store_path(market_data_date)),
        version=MARKET_DAILY_FEATURE_STORE_VERSION,
        market_data_date=market_data_date,
    )
    if data is None or data.empty or "symbol" not in data.columns:
        return {}
    lookup: dict[str, dict[str, object]] = {}
    normalized = data.copy()
    normalized["symbol"] = normalized["symbol"].astype(str).str.zfill(6)
    for row in normalized.to_dict("records"):
        lookup[str(row.get("symbol", "")).zfill(6)] = row
    return lookup


@lru_cache(maxsize=16384)
def _get_market_daily_feature_row_cached(symbol: str, market_data_date: str | None) -> dict[str, object] | None:
    return _get_market_daily_feature_lookup_cached(market_data_date).get(str(symbol).zfill(6))


def get_market_daily_feature_row(symbol: str, market_data_date: str | None) -> dict[str, object] | None:
    row = _get_market_daily_feature_row_cached(str(symbol).zfill(6), market_data_date)
    return None if row is None else dict(row)


MAIN_BOARD_PREFIXES = ("000", "001", "002", "003", "600", "601", "603", "605")
EXCLUDED_GROWTH_BOARD_PREFIXES = ("300", "301", "688", "689")


def _is_main_board_security(symbol: str, market: str = "") -> bool:
    clean_symbol = str(symbol or "").strip()
    market_text = str(market or "").strip()
    if market_text:
        if "??" in market_text or "??" in market_text or "??" in market_text:
            return False
        if "??" in market_text or "???" in market_text:
            return True
    return clean_symbol.startswith(MAIN_BOARD_PREFIXES)


def _is_strategy_eligible_security(
    symbol: str,
    *,
    name: str = "",
    market: str = "",
    board_label: str = "",
) -> bool:
    clean_symbol = str(symbol or "").strip().zfill(6)
    if clean_symbol.startswith(EXCLUDED_GROWTH_BOARD_PREFIXES):
        return False
    name_text = str(name or "").strip().upper()
    if name_text.startswith(("ST", "*ST")) or "ST" in name_text:
        return False
    board_text = f"{market} {board_label}".upper()
    if any(token in board_text for token in ("创业", "科创")):
        return False
    if any(token in board_text for token in ("创业", "科创", "CHINEXT", "STAR")):
        return False
    return clean_symbol.startswith(MAIN_BOARD_PREFIXES)


def _filter_strategy_eligible_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "symbol" not in frame.columns:
        return frame.copy()
    names = frame["name"] if "name" in frame.columns else pd.Series("", index=frame.index)
    markets = frame["market"] if "market" in frame.columns else pd.Series("", index=frame.index)
    boards = frame["board_label"] if "board_label" in frame.columns else pd.Series("", index=frame.index)
    mask = [
        _is_strategy_eligible_security(symbol, name=name, market=market, board_label=board)
        for symbol, name, market, board in zip(frame["symbol"], names, markets, boards)
    ]
    return frame.loc[mask].copy()


def _fallback_stock_basic_from_universe(universe: pd.DataFrame) -> pd.DataFrame:
    if universe.empty:
        return pd.DataFrame(columns=["symbol", "name", "industry", "market"])
    base = universe.copy()
    if "symbol" not in base.columns:
        return pd.DataFrame(columns=["symbol", "name", "industry", "market"])
    base["symbol"] = base["symbol"].astype(str).str.zfill(6)
    if "name" not in base.columns:
        base["name"] = base["symbol"]
    base["name"] = base["name"].astype(str)
    base["industry"] = base["industry"].fillna("").astype(str) if "industry" in base.columns else ""
    base["market"] = base["market"].fillna("").astype(str) if "market" in base.columns else ""
    return base[["symbol", "name", "industry", "market"]].drop_duplicates("symbol").reset_index(drop=True)


def _local_daily_history_cached_symbols(adjust: str = "qfq") -> set[str]:
    pattern = f"daily_history_v1_*_{adjust}.pkl"
    symbols: set[str] = set()
    for path in CACHE_DIR.glob(pattern):
        stem = path.stem
        prefix = f"daily_history_v1_"
        suffix = f"_{adjust}"
        if not stem.startswith(prefix) or not stem.endswith(suffix):
            continue
        symbol = stem[len(prefix):-len(suffix)]
        if symbol.isdigit() and len(symbol) == 6:
            symbols.add(symbol)
    return symbols


def _build_market_feature_rows_from_history(
    eligible: pd.DataFrame,
    market_data_date: str | None,
    *,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total = max(len(eligible), 1)
    for index, row in enumerate(eligible.to_dict("records"), start=1):
        symbol = str(row.get("symbol", "")).zfill(6)
        name = str(row.get("name", symbol))
        try:
            daily = fetch_daily_history(symbol=symbol, start_date=FULL_MARKET_HISTORY_START)
        except Exception:
            daily = pd.DataFrame()
        group = align_daily_history_to_market_date(daily, market_data_date, require_exact=True)
        feature_row = _build_feature_row_from_group(
            group,
            symbol=symbol,
            name=name,
            industry_name=str(row.get("industry", "") or "").strip(),
            market=str(row.get("market", "") or "").strip(),
            snapshot_trade_date=market_data_date,
        )
        if feature_row is not None:
            rows.append(feature_row)
        if progress_callback is not None and _should_report_progress(
            index,
            total,
            desired_updates=FEATURE_BUILD_PROGRESS_UPDATES,
            min_interval=10,
        ):
            progress_callback(
                "构建特征",
                index,
                total,
                f"已完成 {index}/{total} 只股票的历史特征整理",
            )

    feature_store = pd.DataFrame(rows)
    if feature_store.empty:
        return feature_store

    full_industry_scope = feature_store[feature_store["industry_name"].ne("")].copy()
    if not full_industry_scope.empty:
        industry_stats = (
            full_industry_scope.groupby("industry_name", as_index=False)
            .agg(
                industry_ret_2d_pct=(
                    "ret_3d_pct",
                    lambda s: float(pd.Series(s).dropna().median()) if not pd.Series(s).dropna().empty else float("nan"),
                ),
                industry_up_count=(
                    "change_pct",
                    lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum()),
                ),
                industry_stock_count=("symbol", "count"),
            )
            .sort_values("industry_ret_2d_pct", ascending=False)
            .reset_index(drop=True)
        )
        if not industry_stats.empty:
            industry_stats["industry_rank_2d"] = range(1, len(industry_stats) + 1)
            industry_stats["industry_top2d_flag"] = industry_stats["industry_rank_2d"] <= min(10, len(industry_stats))
        feature_store = feature_store.merge(industry_stats, on="industry_name", how="left")
    return feature_store


def _build_market_feature_rows_from_snapshot_history(
    eligible: pd.DataFrame,
    snapshot_history: pd.DataFrame,
    market_data_date: str | None,
    *,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    if eligible.empty or snapshot_history.empty:
        return pd.DataFrame()

    history = snapshot_history.copy()
    history["symbol"] = history["symbol"].astype(str).str.zfill(6)
    eligible_symbols = set(eligible["symbol"].astype(str).str.zfill(6))
    history = history[history["symbol"].isin(eligible_symbols)].copy()
    if history.empty:
        return pd.DataFrame()

    base_lookup = eligible[["symbol", "name", "industry", "market"]].drop_duplicates("symbol").copy()
    base_lookup["symbol"] = base_lookup["symbol"].astype(str).str.zfill(6)
    history = history.merge(base_lookup, on="symbol", how="left", suffixes=("", "_base"))
    if "name_base" in history.columns:
        history["name"] = history["name_base"].fillna(history.get("name")).astype(str)
        history = history.drop(columns=["name_base"], errors="ignore")
    if "industry_base" in history.columns:
        history["industry"] = history["industry_base"].fillna(history.get("industry")).astype(str)
        history = history.drop(columns=["industry_base"], errors="ignore")
    if "market_base" in history.columns:
        history["market"] = history["market_base"].fillna(history.get("market")).astype(str)
        history = history.drop(columns=["market_base"], errors="ignore")

    history = _normalize_history_frame(history)
    if history.empty:
        return pd.DataFrame()
    market_ts = pd.to_datetime(market_data_date, errors="coerce")
    if pd.notna(market_ts):
        history = history[history["date"] <= market_ts].copy()
        if history.empty:
            return pd.DataFrame()
        latest_trade_by_symbol = history.groupby("symbol")["date"].transform("max")
        history = history[latest_trade_by_symbol.eq(market_ts)].copy()
        if history.empty:
            return pd.DataFrame()

    grouped = history.groupby("symbol", sort=False)
    rows: list[dict[str, object]] = []
    total = max(grouped.ngroups, 1)
    for index, (symbol, group) in enumerate(grouped, start=1):
        latest_row = group.iloc[-1]
        feature_row = _build_feature_row_from_group(
            group,
            symbol=str(symbol).zfill(6),
            name=str(latest_row.get("name", symbol) or symbol),
            industry_name=str(latest_row.get("industry", "") or "").strip(),
            market=str(latest_row.get("market", "") or "").strip(),
            snapshot_trade_date=latest_row.get("trade_date"),
            pre_normalized=True,
        )
        if feature_row is not None:
            rows.append(feature_row)
        if progress_callback is not None and _should_report_progress(
            index,
            total,
            desired_updates=FEATURE_BUILD_PROGRESS_UPDATES,
            min_interval=10,
        ):
            progress_callback(
                "构建特征",
                index,
                total,
                f"已完成 {index}/{total} 只股票的批量特征整理",
            )

    feature_store = pd.DataFrame(rows)
    if feature_store.empty:
        return feature_store

    full_industry_scope = feature_store[feature_store["industry_name"].ne("")].copy()
    if not full_industry_scope.empty:
        industry_stats = (
            full_industry_scope.groupby("industry_name", as_index=False)
            .agg(
                industry_ret_2d_pct=(
                    "ret_3d_pct",
                    lambda s: float(pd.Series(s).dropna().median()) if not pd.Series(s).dropna().empty else float("nan"),
                ),
                industry_up_count=(
                    "change_pct",
                    lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum()),
                ),
                industry_stock_count=("symbol", "count"),
            )
            .sort_values("industry_ret_2d_pct", ascending=False)
            .reset_index(drop=True)
        )
        if not industry_stats.empty:
            industry_stats["industry_rank_2d"] = range(1, len(industry_stats) + 1)
            industry_stats["industry_top2d_flag"] = industry_stats["industry_rank_2d"] <= min(10, len(industry_stats))
        feature_store = feature_store.merge(industry_stats, on="industry_name", how="left")
    return feature_store


def _consecutive_up_days(daily: pd.DataFrame) -> int:
    if daily.empty or "close" not in daily.columns:
        return 0
    close = pd.to_numeric(daily["close"], errors="coerce")
    if close.dropna().shape[0] < 2:
        return 0
    daily_change = close.diff()
    streak = 0
    for value in reversed(daily_change.fillna(0.0).tolist()):
        if float(value) > 0:
            streak += 1
            continue
        break
    return int(streak)


def _numeric_series_or_default(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").astype(float).fillna(float(default))


def _replay_bucket_label(value: float, buckets: tuple[tuple[float, float, str], ...]) -> str:
    try:
        numeric = float(value)
    except Exception:
        numeric = 0.0
    for lower, upper, label in buckets:
        if numeric >= lower and numeric < upper:
            return label
    return "unknown"


def _load_candidate_replay_profile() -> dict[str, object] | None:
    try:
        from .daily_review import load_adaptive_rank_profile

        profile = load_adaptive_rank_profile(
            horizon_days=DEFAULT_REPLAY_HORIZON_DAYS,
            positive_return=DEFAULT_REPLAY_POSITIVE_RETURN,
            ranking_by=DEFAULT_REPLAY_RANKING_BY,
            board_size=DEFAULT_REPLAY_BOARD_SIZE,
        )
        return profile if isinstance(profile, dict) else None
    except Exception:
        return None


def _map_edge(labels: pd.Series, edges: dict[str, object], supports: dict[str, object], *, support_floor: float) -> pd.Series:
    if labels.empty or not edges:
        return pd.Series(0.0, index=labels.index, dtype=float)

    def value_for(label: object) -> float:
        key = str(label or "unknown")
        try:
            edge = float(edges.get(key, 0.0) or 0.0)
        except Exception:
            edge = 0.0
        try:
            support = float(supports.get(key, 0.0) or 0.0)
        except Exception:
            support = 0.0
        support_scale = max(0.25, min(support / max(float(support_floor), 1.0), 1.0))
        return edge * support_scale

    return labels.map(value_for).astype(float).fillna(0.0)


def _apply_replay_strategy_fit(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()
    profile = _load_candidate_replay_profile()
    scored = metrics.copy()
    if not profile:
        scored["strategy_fit_score"] = 50.0
        scored["replay_edge_score"] = 0.0
        scored["replay_profile_applied"] = False
        return scored

    stage_labels = scored.get("stage_label", pd.Series("unknown", index=scored.index)).fillna("unknown").astype(str)
    quant_labels = _numeric_series_or_default(scored, "quant_score", 50.0).map(
        lambda value: _replay_bucket_label(value, REPLAY_QUANT_BUCKETS)
    )
    launch_labels = _numeric_series_or_default(scored, "launch_score", 50.0).map(
        lambda value: _replay_bucket_label(value, REPLAY_LAUNCH_BUCKETS)
    )
    resonance_labels = _numeric_series_or_default(scored, "market_resonance_score", 50.0).map(
        lambda value: _replay_bucket_label(value, REPLAY_RESONANCE_BUCKETS)
    )

    stage_edge = _map_edge(
        stage_labels,
        dict(profile.get("stage_edges", {}) or {}),
        dict(profile.get("stage_supports", {}) or {}),
        support_floor=8.0,
    )
    quant_edge = _map_edge(
        quant_labels,
        dict(profile.get("quant_bucket_edges", {}) or {}),
        dict(profile.get("quant_bucket_supports", {}) or {}),
        support_floor=8.0,
    )
    launch_edge = _map_edge(
        launch_labels,
        dict(profile.get("launch_bucket_edges", {}) or {}),
        dict(profile.get("launch_bucket_supports", {}) or {}),
        support_floor=8.0,
    )
    resonance_edge = _map_edge(
        resonance_labels,
        dict(profile.get("resonance_bucket_edges", {}) or {}),
        dict(profile.get("resonance_bucket_supports", {}) or {}),
        support_floor=8.0,
    )
    replay_edge = (
        stage_edge * 0.28
        + quant_edge * 0.18
        + launch_edge * 0.26
        + resonance_edge * 0.28
    ).clip(lower=-0.22, upper=0.22)
    scored["replay_edge_score"] = replay_edge.round(4)
    scored["strategy_fit_score"] = (50.0 + replay_edge * 140.0).clip(lower=0.0, upper=100.0).round(2)
    scored["replay_profile_applied"] = True
    scored["replay_profile_review_days"] = int(profile.get("review_days", 0) or 0)
    scored["replay_profile_market_days"] = int(profile.get("market_replay_days", 0) or 0)
    scored["candidate_priority"] = (
        pd.to_numeric(scored["candidate_priority"], errors="coerce").fillna(0.0)
        + (scored["strategy_fit_score"] - 50.0) * 0.26
    )
    return scored


def _rolling_pullback_metrics(window_df: pd.DataFrame) -> dict[str, object]:
    if window_df.empty or len(window_df) < 8:
        return {
            "pullback_days": 0,
            "pullback_volume_decay": False,
            "pullback_kept_ma10": False,
        }

    recent = window_df.tail(7).reset_index(drop=True)
    pre_today = recent.iloc[:-1].copy()
    if pre_today.empty:
        return {
            "pullback_days": 0,
            "pullback_volume_decay": False,
            "pullback_kept_ma10": False,
        }

    peak_idx = int(pre_today["close"].astype(float).idxmax())
    pullback = pre_today.loc[peak_idx + 1 :].copy()
    pullback_days = int(len(pullback))
    if pullback_days == 0:
        return {
            "pullback_days": 0,
            "pullback_volume_decay": False,
            "pullback_kept_ma10": False,
        }

    volume_col = "vol" if "vol" in pullback.columns else "volume" if "volume" in pullback.columns else ""
    if volume_col:
        volume_decay = bool(
            pullback_days >= 3
            and float(pullback[volume_col].iloc[-1]) <= float(pullback[volume_col].iloc[0])
            and float(pullback[volume_col].mean()) <= float(pre_today[volume_col].iloc[: peak_idx + 1].mean() or 0.0)
        )
    else:
        volume_decay = False
    kept_ma10 = bool((pullback["close"] >= pullback["ma10"]).fillna(False).all())
    return {
        "pullback_days": pullback_days,
        "pullback_volume_decay": volume_decay,
        "pullback_kept_ma10": kept_ma10,
    }


def _build_strategy_snapshot_context(market_data_date: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    safe_date = str(market_data_date or "").replace("-", "")
    try:
        trade_dates = fetch_tushare_recent_trade_dates(end_date=safe_date or None, limit=3)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()
    if not trade_dates:
        return pd.DataFrame(), pd.DataFrame()

    latest_trade_date = trade_dates[-1]
    previous_trade_date = trade_dates[-2] if len(trade_dates) >= 2 else latest_trade_date
    try:
        latest_snapshot = fetch_tushare_daily_snapshot(latest_trade_date)
        previous_snapshot = fetch_tushare_daily_snapshot(previous_trade_date)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()
    return latest_snapshot, previous_snapshot


def _normalize_market_data_date(value: str | None) -> str:
    market_ts = pd.to_datetime(value, errors="coerce")
    if pd.notna(market_ts):
        return market_ts.strftime("%Y-%m-%d")
    return str(value or "").strip()


def _coerce_trade_date_series(values: pd.Series | object) -> pd.Series:
    if isinstance(values, pd.Series) and pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, errors="coerce")
    parsed = pd.to_datetime(values, format="%Y%m%d", errors="coerce")
    if isinstance(parsed, pd.Series) and parsed.notna().any():
        return parsed
    return pd.to_datetime(values, errors="coerce")


def _snapshot_history_covers_market_date(snapshot_history: pd.DataFrame | None, market_data_date: str | None) -> bool:
    if not isinstance(snapshot_history, pd.DataFrame) or snapshot_history.empty:
        return False
    market_ts = pd.to_datetime(market_data_date, errors="coerce")
    if pd.isna(market_ts):
        return True
    trade_dates = _coerce_trade_date_series(snapshot_history.get("trade_date"))
    latest_trade_date = trade_dates.max()
    return pd.notna(latest_trade_date) and latest_trade_date == market_ts


def _prepare_snapshot_history_frame(snapshot: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(snapshot, pd.DataFrame) or snapshot.empty:
        return pd.DataFrame()
    frame = snapshot.copy()
    frame["trade_date"] = _coerce_trade_date_series(frame.get("trade_date"))
    frame = frame.dropna(subset=["trade_date"]).copy()
    if frame.empty:
        return pd.DataFrame()
    frame["date"] = frame["trade_date"]
    if "volume" not in frame.columns:
        frame["volume"] = pd.to_numeric(frame.get("vol"), errors="coerce")
    frame["vol"] = pd.to_numeric(frame.get("vol", frame.get("volume")), errors="coerce")
    frame["volume"] = pd.to_numeric(frame.get("volume", frame.get("vol")), errors="coerce")
    frame["turnover"] = pd.to_numeric(frame.get("turnover_rate", frame.get("turnover")), errors="coerce")
    frame["change_pct"] = pd.to_numeric(frame.get("pct_chg", frame.get("change_pct")), errors="coerce")
    keep_cols = [
        "symbol",
        "name",
        "industry",
        "market",
        "trade_date",
        "date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "change_pct",
        "volume",
        "vol",
        "amount",
        "turnover",
    ]
    return frame[[column for column in keep_cols if column in frame.columns]].copy()


def load_incremental_market_snapshot_history(
    market_data_date: str | None,
    *,
    lookback_sessions: int = MARKET_FEATURE_SNAPSHOT_LOOKBACK,
    progress_callback: ProgressCallback | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    normalized_market_data_date = _normalize_market_data_date(market_data_date)
    cached = None if force_rebuild else read_market_snapshot_history_store(normalized_market_data_date)
    if _snapshot_history_covers_market_date(cached, normalized_market_data_date):
        return cached.copy()
    if cached is not None and not cached.empty:
        return _load_recent_snapshot_history(
            normalized_market_data_date,
            lookback_sessions=lookback_sessions,
            progress_callback=progress_callback,
            force_rebuild=force_rebuild,
        )

    safe_date = normalized_market_data_date.replace("-", "")
    try:
        trade_dates = fetch_tushare_recent_trade_dates(
            end_date=safe_date or None,
            limit=max(int(lookback_sessions), 20),
        )
    except Exception:
        trade_dates = []
    if not trade_dates:
        return pd.DataFrame()

    latest_trade_date = str(trade_dates[-1])
    previous_trade_date = str(trade_dates[-2]) if len(trade_dates) >= 2 else ""
    previous_cache_date = _normalize_market_data_date(previous_trade_date)
    previous_cached = None if force_rebuild else read_market_snapshot_history_store(previous_cache_date)
    if _snapshot_history_covers_market_date(previous_cached, previous_cache_date):
        try:
            latest_snapshot = fetch_tushare_daily_snapshot(latest_trade_date)
        except Exception:
            latest_snapshot = pd.DataFrame()
        latest_frame = _prepare_snapshot_history_frame(latest_snapshot)
        if not latest_frame.empty:
            allowed_trade_dates = set(
                pd.to_datetime(trade_dates[-max(int(lookback_sessions), 20):], format="%Y%m%d", errors="coerce")
                .dropna()
                .tolist()
            )
            snapshot_history = (
                pd.concat([previous_cached, latest_frame], ignore_index=True)
                .dropna(subset=["symbol", "trade_date"])
                .sort_values(["symbol", "trade_date"])
                .drop_duplicates(["symbol", "trade_date"], keep="last")
                .reset_index(drop=True)
            )
            if allowed_trade_dates:
                snapshot_history = snapshot_history[snapshot_history["trade_date"].isin(allowed_trade_dates)].reset_index(drop=True)
            if not snapshot_history.empty:
                _write_market_snapshot_history_store(snapshot_history, normalized_market_data_date)
                return snapshot_history

    return _load_recent_snapshot_history(
        normalized_market_data_date,
        lookback_sessions=lookback_sessions,
        progress_callback=progress_callback,
        force_rebuild=force_rebuild,
    )


def _load_recent_snapshot_history(
    market_data_date: str | None,
    *,
    lookback_sessions: int = MARKET_FEATURE_SNAPSHOT_LOOKBACK,
    progress_callback: ProgressCallback | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    cached = None if force_rebuild else read_market_snapshot_history_store(market_data_date)
    if cached is not None and not cached.empty:
        return cached.copy()

    safe_date = str(market_data_date or "").replace("-", "")
    try:
        trade_dates = fetch_tushare_recent_trade_dates(
            end_date=safe_date or None,
            limit=max(int(lookback_sessions), 20),
        )
    except Exception:
        return pd.DataFrame()
    if not trade_dates:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    total = max(len(trade_dates), 1)
    for index, trade_date in enumerate(trade_dates, start=1):
        try:
            snapshot = fetch_tushare_daily_snapshot(trade_date)
        except Exception:
            snapshot = pd.DataFrame()
        frame = _prepare_snapshot_history_frame(snapshot)
        if frame.empty:
            continue
        frames.append(frame)
        if progress_callback is not None and _should_report_progress(
            index,
            total,
            desired_updates=SNAPSHOT_LOAD_PROGRESS_UPDATES,
            min_interval=3,
        ):
            progress_callback(
                "批量快照",
                index,
                total,
                f"正在装载近端交易快照 {index}/{total}",
            )

    if not frames:
        return pd.DataFrame()
    snapshot_history = (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["symbol", "trade_date"])
        .sort_values(["symbol", "trade_date"])
        .drop_duplicates(["symbol", "trade_date"], keep="last")
        .reset_index(drop=True)
    )
    _write_market_snapshot_history_store(snapshot_history, market_data_date)
    return snapshot_history


def _normalize_history_frame(daily: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(daily, pd.DataFrame) or daily.empty:
        return pd.DataFrame()
    frame = daily.copy()
    if "date" not in frame.columns and "trade_date" in frame.columns:
        frame["date"] = frame["trade_date"]
    if "date" not in frame.columns:
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if frame.empty:
        return frame

    numeric_cols = ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "change_pct", "amount", "volume", "vol", "turnover"]
    for column in numeric_cols:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "volume" not in frame.columns and "vol" in frame.columns:
        frame["volume"] = pd.to_numeric(frame["vol"], errors="coerce")
    if "vol" not in frame.columns and "volume" in frame.columns:
        frame["vol"] = pd.to_numeric(frame["volume"], errors="coerce")
    if "change_pct" not in frame.columns and "pct_chg" in frame.columns:
        frame["change_pct"] = pd.to_numeric(frame["pct_chg"], errors="coerce")
    if "amount" not in frame.columns:
        frame["amount"] = float("nan")

    turnover_series = pd.to_numeric(frame.get("turnover"), errors="coerce")
    if turnover_series.dropna().shape[0] < min(20, len(frame)):
        amount_series = pd.to_numeric(frame.get("amount"), errors="coerce")
        amount_baseline = amount_series.rolling(20, min_periods=1).mean().replace(0, pd.NA)
        pseudo_turnover = (amount_series / amount_baseline).clip(lower=0).fillna(1.0) * 4.0
        turnover_series = pseudo_turnover if turnover_series.dropna().empty else turnover_series.fillna(pseudo_turnover)
    frame["turnover"] = turnover_series.astype(float)
    return frame


def _build_feature_row_from_group(
    group: pd.DataFrame,
    *,
    symbol: str,
    name: str,
    industry_name: str,
    market: str,
    industry_2d_change_pct: float = float("nan"),
    snapshot_trade_date: object | None = None,
    pre_normalized: bool = False,
) -> dict[str, object] | None:
    normalized = group.reset_index(drop=True).copy() if pre_normalized else _normalize_history_frame(group)
    if normalized.empty or len(normalized) < 20:
        return None

    features = build_daily_features(normalized)
    valid_features = features.dropna()
    if valid_features.empty:
        return None
    latest_features = valid_features.iloc[-1]
    latest_feature_map = latest_features.to_dict()
    stage = classify_stage(normalized)
    quant_signal = evaluate_quant_signal(normalized, features)
    rule_context = build_trading_rule_context(symbol=symbol, name=name)
    launch_score = float(main_rise_start_score(latest_features))
    stage_score = float(stage_numeric_score(stage, latest_features))

    normalized = normalized.copy()
    normalized["ma5"] = normalized["close"].rolling(5, min_periods=5).mean()
    normalized["ma10"] = normalized["close"].rolling(10, min_periods=10).mean()
    normalized["ma20"] = normalized["close"].rolling(20, min_periods=20).mean()
    latest_hist = normalized.iloc[-1]
    close = float(latest_hist["close"])
    close_3 = float(normalized["close"].iloc[-3]) if len(normalized) >= 3 else float("nan")
    close_5 = float(normalized["close"].iloc[-5]) if len(normalized) >= 5 else float("nan")
    close_10 = float(normalized["close"].iloc[-10]) if len(normalized) >= 10 else float("nan")
    close_15 = float(normalized["close"].iloc[-15]) if len(normalized) >= 15 else float("nan")
    close_20 = float(normalized["close"].iloc[-20]) if len(normalized) >= 20 else float("nan")
    high_10 = float(normalized["high"].tail(10).max()) if "high" in normalized.columns else float(normalized["close"].tail(10).max())
    low_10 = float(normalized["low"].tail(10).min()) if "low" in normalized.columns else float(normalized["close"].tail(10).min())
    max_gain_10 = (high_10 / low_10 - 1) * 100 if low_10 > 0 else float("nan")
    pullback = _rolling_pullback_metrics(normalized.tail(10))
    snapshot = build_latest_snapshot(normalized, features)
    snapshot.update(
        {
            "range_position_20": float(latest_features.get("range_position_20", 0.5)),
            "upper_shadow_ratio": float(latest_features.get("upper_shadow_ratio", 0.0)),
        }
    )
    model_feature_frame = _prepare_live_feature_frame(
        normalized,
        latest_feature_values=latest_features,
        symbol=symbol,
    )
    model_feature_values = (
        model_feature_frame.iloc[-1].to_dict()
        if isinstance(model_feature_frame, pd.DataFrame) and not model_feature_frame.empty
        else {}
    )
    launch_readiness_score = float(model_feature_values.get("launch_readiness", 50.0) or 50.0)
    market_resonance_score = float(model_feature_values.get("market_resonance", 50.0) or 50.0)
    analysis_date = (
        pd.to_datetime(latest_hist.get("date"), errors="coerce").strftime("%Y-%m-%d")
        if pd.notna(pd.to_datetime(latest_hist.get("date"), errors="coerce"))
        else str(snapshot.get("date", "") or "")
    )

    return {
        "symbol": symbol,
        "name": name,
        "analysis_date": analysis_date,
        "industry_name": industry_name,
        "market": market,
        "latest_price": round(close, 2),
        "change_pct": round(float(latest_hist.get("change_pct", 0.0) or 0.0), 2),
        "amount": round(float(latest_hist.get("amount", 0.0) or 0.0), 2),
        "turnover": round(float(latest_hist.get("turnover", 0.0) or 0.0), 2),
        "consecutive_up_days": _consecutive_up_days(normalized),
        "ret_3d_pct": (close / close_3 - 1) * 100 if close_3 > 0 else float("nan"),
        "ret_5d_pct": (close / close_5 - 1) * 100 if close_5 > 0 else float("nan"),
        "ret_10d_pct": (close / close_10 - 1) * 100 if close_10 > 0 else float("nan"),
        "ret_15d_pct": (close / close_15 - 1) * 100 if close_15 > 0 else float("nan"),
        "ret_20d_pct": (close / close_20 - 1) * 100 if close_20 > 0 else float("nan"),
        "ma5": float(latest_hist.get("ma5", float("nan"))),
        "ma10": float(latest_hist.get("ma10", float("nan"))),
        "ma20": float(latest_hist.get("ma20", float("nan"))),
        "high_10": high_10,
        "distance_to_high_10_pct": (high_10 - close) / high_10 * 100 if high_10 > 0 else float("nan"),
        "max_gain_10_pct": max_gain_10,
        "pullback_days": int(pullback["pullback_days"]),
        "pullback_volume_decay": bool(pullback["pullback_volume_decay"]),
        "pullback_kept_ma10": bool(pullback["pullback_kept_ma10"]),
        "industry_2d_change_pct": float(industry_2d_change_pct),
        "snapshot_trade_date": pd.to_datetime(snapshot_trade_date if snapshot_trade_date is not None else analysis_date, errors="coerce"),
        "snapshot": snapshot,
        "latest_features": latest_feature_map,
        "model_feature_values": model_feature_values,
        "stage_object": stage,
        "stage_code": str(stage.code),
        "stage_label": str(stage.label),
        "stage_priority": str(stage.priority),
        "stage_summary": str(stage.structure_summary),
        "stage_score": stage_score,
        "quant_signal_object": quant_signal,
        "quant_score": float(quant_signal.total_score),
        "quant_primary_signal": str(quant_signal.primary_signal),
        "launch_score": launch_score,
        "launch_readiness_score": launch_readiness_score,
        "market_resonance_score": market_resonance_score,
        "rule_context_object": rule_context,
        "board_label": str(rule_context.board_label),
        "price_limit_label": str(rule_context.price_limit_label),
        "rule_summary": str(rule_context.rule_summary),
    }


def align_daily_history_to_market_date(
    daily: pd.DataFrame,
    market_data_date: str | None,
    *,
    require_exact: bool = True,
) -> pd.DataFrame:
    if not isinstance(daily, pd.DataFrame) or daily.empty:
        return pd.DataFrame()
    if "date" in daily.columns:
        aligned = daily.reset_index(drop=True).copy()
    else:
        aligned = daily.reset_index().rename(columns={"index": "date"}).copy()
    if "date" not in aligned.columns:
        return pd.DataFrame()
    aligned["date"] = pd.to_datetime(aligned["date"], errors="coerce")
    aligned = aligned.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if aligned.empty or not market_data_date:
        return aligned
    market_ts = pd.to_datetime(market_data_date, errors="coerce")
    if pd.isna(market_ts):
        return aligned
    aligned = aligned[aligned["date"] <= market_ts].reset_index(drop=True)
    if aligned.empty:
        return pd.DataFrame()
    if require_exact and aligned["date"].max() != market_ts:
        return pd.DataFrame()
    return aligned


def build_market_daily_feature_store(
    universe: pd.DataFrame,
    market_data_date: str | None,
    *,
    progress_callback: ProgressCallback | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    if universe.empty:
        return pd.DataFrame()
    cached = None if force_rebuild else read_market_daily_feature_store(market_data_date)
    if cached is not None and not cached.empty:
        return cached.copy()

    point_in_time_universe = False
    try:
        stock_basic = fetch_tushare_stock_basic_all_statuses()
        point_in_time_universe = not stock_basic.empty
    except Exception:
        stock_basic = pd.DataFrame()
    if stock_basic.empty:
        try:
            stock_basic = fetch_tushare_stock_basic()
        except Exception:
            stock_basic = pd.DataFrame()
    latest_snapshot_df, previous_snapshot = _build_strategy_snapshot_context(market_data_date)

    if stock_basic.empty:
        stock_basic = _fallback_stock_basic_from_universe(universe)
    if stock_basic.empty:
        return pd.DataFrame()

    base_universe = filter_point_in_time_a_share_universe(stock_basic, market_data_date)
    if base_universe.empty:
        base_universe = stock_basic.copy()
    if {"symbol", "name"}.issubset(universe.columns):
        base_universe = base_universe.merge(
            universe[["symbol", "name"]].drop_duplicates("symbol"),
            on="symbol",
            how="left",
            suffixes=("", "_universe"),
        )
        if "name_universe" in base_universe.columns:
            base_universe["name"] = base_universe["name_universe"].fillna(base_universe["name"])
            base_universe = base_universe.drop(columns=["name_universe"], errors="ignore")

    eligible = _filter_strategy_eligible_frame(base_universe)
    if eligible.empty:
        return pd.DataFrame()
    eligible["point_in_time_universe"] = bool(point_in_time_universe)

    def finalize_feature_store(feature_store: pd.DataFrame) -> pd.DataFrame:
        if feature_store.empty:
            return feature_store
        if "industry_top2d_flag" not in feature_store.columns:
            feature_store["industry_top2d_flag"] = False
        if "industry_up_count" not in feature_store.columns:
            feature_store["industry_up_count"] = 0
        if "industry_stock_count" not in feature_store.columns:
            feature_store["industry_stock_count"] = 0
        if "industry_rank_2d" not in feature_store.columns:
            feature_store["industry_rank_2d"] = float("nan")
        feature_store["point_in_time_universe"] = bool(point_in_time_universe)
        feature_store = feature_store.sort_values(["symbol"]).reset_index(drop=True)
        _write_market_daily_feature_store(feature_store, market_data_date)
        return feature_store

    if latest_snapshot_df.empty:
        snapshot_history = load_incremental_market_snapshot_history(
            market_data_date,
            progress_callback=progress_callback,
            force_rebuild=force_rebuild,
        )
        if _snapshot_history_covers_market_date(snapshot_history, market_data_date):
            feature_store = _build_market_feature_rows_from_snapshot_history(
                eligible.rename(columns={"industry": "industry", "market": "market"}),
                snapshot_history,
                market_data_date,
                progress_callback=progress_callback,
            )
            if not feature_store.empty:
                return finalize_feature_store(feature_store)

        cached_symbols = _local_daily_history_cached_symbols()
        if cached_symbols:
            eligible = eligible[eligible["symbol"].isin(cached_symbols)].copy()
        feature_store = _build_market_feature_rows_from_history(
            eligible,
            market_data_date,
            progress_callback=progress_callback,
        )
        return finalize_feature_store(feature_store)

    snapshot_history = load_incremental_market_snapshot_history(
        market_data_date,
        progress_callback=progress_callback,
        force_rebuild=force_rebuild,
    )
    feature_store = _build_market_feature_rows_from_snapshot_history(
        eligible.rename(columns={"industry": "industry", "market": "market"}),
        snapshot_history,
        market_data_date,
        progress_callback=progress_callback,
    )
    if feature_store.empty:
        feature_store = _build_market_feature_rows_from_history(
            eligible,
            market_data_date,
            progress_callback=progress_callback,
        )
    if feature_store.empty:
        return feature_store

    latest_snapshot_df = latest_snapshot_df[latest_snapshot_df["symbol"].isin(eligible["symbol"])].copy()
    if not latest_snapshot_df.empty:
        latest_snapshot_df = latest_snapshot_df.merge(
            eligible[["symbol", "name", "industry", "market"]],
            on="symbol",
            how="left",
            suffixes=("", "_basic"),
        )
        latest_snapshot_df["industry_name"] = latest_snapshot_df["industry"].fillna("").astype(str).str.strip()
        previous_lookup = pd.DataFrame()
        if not previous_snapshot.empty:
            previous_lookup = previous_snapshot[["symbol", "close"]].rename(columns={"close": "prev_trade_close"}).copy()
            latest_snapshot_df = latest_snapshot_df.merge(previous_lookup, on="symbol", how="left")
        latest_snapshot_df["industry_change_base"] = latest_snapshot_df["prev_trade_close"].where(
            pd.to_numeric(latest_snapshot_df.get("prev_trade_close"), errors="coerce").gt(0)
        )
        latest_snapshot_df["industry_2d_change_pct"] = (
            latest_snapshot_df["close"] / latest_snapshot_df["industry_change_base"] - 1
        ) * 100

        full_industry_scope = latest_snapshot_df[latest_snapshot_df["industry_name"].ne("")].copy()
        if not full_industry_scope.empty:
            full_industry_scope["snapshot_change_pct"] = pd.to_numeric(full_industry_scope.get("pct_chg"), errors="coerce")
            industry_stats = (
                full_industry_scope.groupby("industry_name", as_index=False)
                .agg(
                    industry_ret_2d_pct=(
                        "industry_2d_change_pct",
                        lambda s: float(pd.Series(s).dropna().median()) if not pd.Series(s).dropna().empty else float("nan"),
                    ),
                    industry_up_count=(
                        "snapshot_change_pct",
                        lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum()),
                    ),
                    industry_stock_count=("symbol", "count"),
                )
                .sort_values("industry_ret_2d_pct", ascending=False)
                .reset_index(drop=True)
            )
            if not industry_stats.empty:
                industry_stats["industry_rank_2d"] = range(1, len(industry_stats) + 1)
                industry_stats["industry_top2d_flag"] = industry_stats["industry_rank_2d"] <= min(10, len(industry_stats))
            feature_store = feature_store.drop(
                columns=[column for column in ["industry_ret_2d_pct", "industry_up_count", "industry_stock_count", "industry_rank_2d", "industry_top2d_flag"] if column in feature_store.columns],
                errors="ignore",
            )
            feature_store = feature_store.merge(industry_stats, on="industry_name", how="left")

    if "industry_top2d_flag" not in feature_store.columns:
        feature_store["industry_top2d_flag"] = False
    if "industry_up_count" not in feature_store.columns:
        feature_store["industry_up_count"] = 0
    if "industry_stock_count" not in feature_store.columns:
        feature_store["industry_stock_count"] = 0
    if "industry_rank_2d" not in feature_store.columns:
        feature_store["industry_rank_2d"] = float("nan")
    return finalize_feature_store(feature_store)


def build_market_candidate_pool_store(
    universe: pd.DataFrame,
    market_data_date: str | None,
    *,
    feature_store: pd.DataFrame | None = None,
    progress_callback: ProgressCallback | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    if universe.empty:
        return pd.DataFrame()
    cached = None if force_rebuild else read_market_candidate_pool_store(market_data_date)
    if cached is not None and not cached.empty:
        return _filter_strategy_eligible_frame(cached)

    store = feature_store
    if store is None or store.empty:
        store = build_market_daily_feature_store(
            universe,
            market_data_date,
            progress_callback=progress_callback,
            force_rebuild=force_rebuild,
        )
    if store.empty:
        return pd.DataFrame()

    metrics = _filter_strategy_eligible_frame(store.copy())
    if metrics.empty:
        return pd.DataFrame()
    snapshot_series = (
        metrics["snapshot"]
        if "snapshot" in metrics.columns
        else pd.Series([{} for _ in range(len(metrics))], index=metrics.index, dtype=object)
    )
    snapshot_records = [value if isinstance(value, dict) else {} for value in snapshot_series.tolist()]
    metrics["snapshot_close_vs_ma20"] = [float(record.get("close_vs_ma20", float("nan"))) for record in snapshot_records]
    metrics["snapshot_volume_ratio_5"] = [float(record.get("volume_ratio_5", float("nan"))) for record in snapshot_records]
    metrics["snapshot_range_position_20"] = [float(record.get("range_position_20", float("nan"))) for record in snapshot_records]
    metrics["snapshot_upper_shadow_ratio"] = [float(record.get("upper_shadow_ratio", float("nan"))) for record in snapshot_records]
    launch_score_series = _numeric_series_or_default(metrics, "launch_readiness_score", float("nan"))
    launch_score_series = launch_score_series.where(
        launch_score_series.notna(),
        _numeric_series_or_default(metrics, "launch_score", 50.0),
    ).clip(lower=0.0, upper=100.0)
    resonance_score_series = _numeric_series_or_default(metrics, "market_resonance_score", 50.0).clip(lower=0.0, upper=100.0)
    distance_to_high = _numeric_series_or_default(metrics, "distance_to_high_10_pct", 0.0)
    industry_ret_2d = _numeric_series_or_default(metrics, "industry_ret_2d_pct", 0.0)
    industry_up_count = _numeric_series_or_default(metrics, "industry_up_count", 0.0)
    snapshot_volume_ratio = _numeric_series_or_default(metrics, "snapshot_volume_ratio_5", 1.0)
    snapshot_upper_shadow = _numeric_series_or_default(metrics, "snapshot_upper_shadow_ratio", 0.0)
    snapshot_range_position = _numeric_series_or_default(metrics, "snapshot_range_position_20", 0.5)
    snapshot_close_vs_ma20 = _numeric_series_or_default(metrics, "snapshot_close_vs_ma20", 0.0)
    close_position_day = _numeric_series_or_default(metrics, "close_position_day", float("nan"))
    close_position_day = close_position_day.where(close_position_day.notna(), (1.0 - snapshot_upper_shadow).clip(0.0, 1.0))
    high_turnover_2d = _numeric_series_or_default(metrics, "turnover_2d_avg", float("nan"))
    high_turnover_2d = high_turnover_2d.where(high_turnover_2d.notna(), _numeric_series_or_default(metrics, "turnover", 0.0))
    volume_price_quality = (
        50.0
        + (_numeric_series_or_default(metrics, "change_pct", 0.0).clip(lower=0.0, upper=12.0) * 2.2)
        + (_numeric_series_or_default(metrics, "ret_3d_pct", 0.0).clip(lower=0.0, upper=24.0) * 0.9)
        + (2.0 - distance_to_high.abs().clip(upper=8.0)) * 3.0
        + (snapshot_volume_ratio.clip(lower=0.6, upper=2.4) - 1.0) * 12.0
        - snapshot_upper_shadow.clip(lower=0.0, upper=0.6) * 42.0
    )
    metrics["launch_readiness"] = launch_score_series
    metrics["breakout_quality"] = volume_price_quality.clip(lower=0.0, upper=100.0)
    metrics["resonance_quality"] = (
        42.0
        + resonance_score_series * 0.22
        + industry_ret_2d.clip(lower=-4.0, upper=12.0) * 2.4
        + industry_up_count.clip(lower=0.0, upper=10.0) * 3.2
        + metrics["industry_top2d_flag"].fillna(False).astype(float) * 8.0
    ).clip(lower=0.0, upper=100.0)
    metrics["risk_of_late_entry"] = (
        18.0
        + _numeric_series_or_default(metrics, "max_gain_10_pct", 0.0).clip(lower=0.0, upper=55.0) * 1.05
        + _numeric_series_or_default(metrics, "ret_20d_pct", 0.0).clip(lower=0.0, upper=55.0) * 0.55
        + snapshot_range_position.clip(lower=0.0, upper=1.0) * 18.0
        + snapshot_upper_shadow.clip(lower=0.0, upper=0.6) * 55.0
        + _numeric_series_or_default(metrics, "turnover", 0.0).clip(lower=0.0, upper=20.0) * 0.85
        - launch_score_series * 0.22
    ).clip(lower=0.0, upper=100.0)
    intraday_rush_fade = _numeric_series_or_default(metrics, "intraday_rush_fade_score", float("nan"))
    intraday_rush_fade = intraday_rush_fade.where(
        intraday_rush_fade.notna(),
        _numeric_series_or_default(metrics, "opening_rush_fade_score", 0.0),
    )
    stock_lead_board_lag = (
        (_numeric_series_or_default(metrics, "change_pct", 0.0) >= 5.0)
        & ((industry_up_count < 3.0) | (industry_ret_2d < 1.0))
    )
    volume_price_stall = (
        (snapshot_volume_ratio >= 1.55)
        & (close_position_day <= 0.58)
        & (snapshot_upper_shadow >= 0.18)
    )
    tail_withdrawal = (snapshot_upper_shadow >= 0.24) | (close_position_day <= 0.42)
    metrics["crowding_risk"] = (
        12.0
        + _numeric_series_or_default(metrics, "turnover", 0.0).clip(lower=0.0, upper=22.0) * 1.25
        + high_turnover_2d.clip(lower=0.0, upper=22.0) * 0.55
        + (snapshot_volume_ratio.clip(lower=1.0, upper=3.0) - 1.0) * 18.0
        + snapshot_upper_shadow.clip(lower=0.0, upper=0.6) * 62.0
        + (0.62 - close_position_day.clip(lower=0.0, upper=1.0)).clip(lower=0.0) * 44.0
        + _numeric_series_or_default(metrics, "max_gain_10_pct", 0.0).clip(lower=0.0, upper=45.0) * 0.32
        + _numeric_series_or_default(metrics, "ret_20d_pct", 0.0).clip(lower=0.0, upper=45.0) * 0.18
        + pd.Series(np.where(stock_lead_board_lag, 9.0, 0.0), index=metrics.index)
        + pd.Series(np.where(volume_price_stall, 10.0, 0.0), index=metrics.index)
        + pd.Series(np.where(tail_withdrawal, 6.0, 0.0), index=metrics.index)
        + intraday_rush_fade.clip(lower=0.0, upper=100.0) * 0.10
        - metrics["resonance_quality"].clip(lower=0.0, upper=100.0) * 0.12
    ).clip(lower=0.0, upper=100.0)
    pullback_acceptance_quality = (
        50.0
        + pd.Series(np.where(metrics["pullback_volume_decay"], 12.0, -8.0), index=metrics.index)
        + pd.Series(np.where(metrics["pullback_kept_ma10"], 12.0, -14.0), index=metrics.index)
        + (snapshot_range_position.clip(lower=0.0, upper=1.0) - 0.45) * 28.0
        - snapshot_upper_shadow.clip(lower=0.0, upper=0.5) * 42.0
        - (snapshot_volume_ratio.clip(lower=1.0, upper=2.5) - 1.0) * 7.0
    ).clip(lower=0.0, upper=100.0)
    board_leader_quality = (
        48.0
        + industry_ret_2d.clip(lower=-2.0, upper=12.0) * 2.8
        + industry_up_count.clip(lower=0.0, upper=10.0) * 3.4
        - distance_to_high.abs().clip(lower=0.0, upper=8.0) * 1.8
        + metrics["industry_top2d_flag"].fillna(False).astype(float) * 8.0
    ).clip(lower=0.0, upper=100.0)
    metrics["board_resonance_strength"] = board_leader_quality
    metrics["long_setup_quality"] = (
        metrics["resonance_quality"] * 0.22
        + metrics["breakout_quality"] * 0.18
        + pullback_acceptance_quality * 0.20
        + launch_score_series * 0.20
        + _numeric_series_or_default(metrics, "quant_score", 50.0).clip(lower=0.0, upper=100.0) * 0.10
        + board_leader_quality * 0.15
        - metrics["crowding_risk"] * 0.15
    ).clip(lower=0.0, upper=100.0)
    metrics["launch_phase_label"] = np.select(
        [
            metrics["risk_of_late_entry"] >= 68.0,
            metrics["crowding_risk"] >= 76.0,
            (metrics["breakout_quality"] < 48.0) | (metrics["resonance_quality"] < 50.0),
            (metrics["launch_readiness"] >= 64.0) & (metrics["risk_of_late_entry"] < 56.0),
        ],
        ["已走远", "量化拥挤", "伪突破", "刚启动"],
        default="观察",
    )

    strategy1_core_mask = (
        metrics["ret_15d_pct"].between(10, 30, inclusive="both")
        & (metrics["latest_price"] > metrics["ma5"])
        & (metrics["ma5"] > metrics["ma10"])
        & (metrics["ma10"] > metrics["ma20"])
        & metrics["pullback_days"].between(3, 6, inclusive="both")
        & metrics["pullback_volume_decay"]
        & metrics["pullback_kept_ma10"]
        & (metrics["change_pct"] > 2)
        & (metrics["amount"] > 2e8)
        & (metrics["turnover"] > 3)
        & (metrics["ret_20d_pct"] < 35)
    )
    key_bull_center_5 = _numeric_series_or_default(metrics, "key_bull_center_5", float("nan"))
    failed_breakouts_20 = _numeric_series_or_default(metrics, "failed_breakouts_20", 0.0)
    failed_breakouts_20 = failed_breakouts_20.where(
        failed_breakouts_20.notna(),
        _numeric_series_or_default(metrics, "failed_high_attempts_20", 0.0),
    )
    strategy1_veto_mask = (
        ((snapshot_volume_ratio >= 1.65) & (snapshot_upper_shadow >= 0.24))
        | (key_bull_center_5.notna() & (metrics["latest_price"] < key_bull_center_5))
        | (failed_breakouts_20 >= 2)
        | (metrics["crowding_risk"] >= 82.0)
        | volume_price_stall
    )
    strategy1_bonus_score = (
        ((industry_ret_2d >= 2.0) & (metrics["resonance_quality"] >= 62.0)).astype(float) * 4.0
        + ((snapshot_range_position >= 0.55) & (snapshot_upper_shadow <= 0.16)).astype(float) * 4.0
        + (metrics["launch_phase_label"].eq("刚启动")).astype(float) * 5.0
        + (metrics["crowding_risk"] <= 38.0).astype(float) * 3.0
    )
    strategy1_mask = strategy1_core_mask & (~strategy1_veto_mask)
    industry_available = metrics.get("industry_name", pd.Series("", index=metrics.index)).astype(str).str.strip().ne("")
    board_resonance_mask = (
        industry_available
        & metrics["industry_top2d_flag"].fillna(False)
        & (metrics["industry_up_count"].fillna(0) >= 3)
    )
    strategy2_core_mask = (
        (metrics["change_pct"] > 5)
        & (metrics["ret_3d_pct"] > 10)
        & (metrics["ret_5d_pct"] > 15)
        & (metrics["amount"] > 3e8)
        & (metrics["turnover"] > 5)
        & ((metrics["latest_price"] >= metrics["high_10"]) | (metrics["distance_to_high_10_pct"] < 2))
        & (metrics["max_gain_10_pct"] < 40)
        & board_resonance_mask
    )
    strategy2_veto_mask = (
        ((snapshot_upper_shadow >= 0.30) | (close_position_day.notna() & (close_position_day <= 0.42)))
        | ((industry_up_count <= 3) & (industry_ret_2d <= 1.0))
        | ((high_turnover_2d >= 12.0) & (snapshot_upper_shadow >= 0.22) & (snapshot_range_position < 0.68))
        | (metrics["risk_of_late_entry"] >= 76.0)
        | (metrics["crowding_risk"] >= 78.0)
        | volume_price_stall
    )
    strategy2_bonus_score = (
        (metrics["breakout_quality"].clip(lower=50.0, upper=90.0) - 50.0) * 0.08
        + (metrics["resonance_quality"].clip(lower=50.0, upper=90.0) - 50.0) * 0.10
        + (board_leader_quality.clip(lower=50.0, upper=90.0) - 50.0) * 0.08
        + (metrics["launch_phase_label"].eq("刚启动")).astype(float) * 5.0
        + (metrics["crowding_risk"] <= 40.0).astype(float) * 3.0
        - (metrics["risk_of_late_entry"].clip(lower=55.0, upper=90.0) - 55.0) * 0.05
        - (metrics["crowding_risk"].clip(lower=45.0, upper=90.0) - 45.0) * 0.06
    )
    strategy2_mask = strategy2_core_mask & (~strategy2_veto_mask)
    strategy3_core_score = (
        metrics["long_setup_quality"].fillna(50.0) * 0.24
        + metrics["resonance_quality"].fillna(50.0) * 0.18
        + metrics["launch_readiness"].fillna(50.0) * 0.18
        + metrics["breakout_quality"].fillna(50.0) * 0.14
        + metrics["board_resonance_strength"].fillna(50.0) * 0.12
        + _numeric_series_or_default(metrics, "quant_score", 50.0).clip(lower=0.0, upper=100.0) * 0.10
        - metrics["crowding_risk"].fillna(50.0) * 0.10
        - metrics["risk_of_late_entry"].fillna(50.0) * 0.06
    ).clip(lower=0.0, upper=100.0)
    strategy3_structure_mask = (
        (metrics["latest_price"] > metrics["ma10"])
        & (metrics["ma5"] >= metrics["ma20"] * 0.995)
        & (_numeric_series_or_default(metrics, "ret_5d_pct", 0.0).between(2.0, 18.0, inclusive="both"))
        & (_numeric_series_or_default(metrics, "ret_20d_pct", 0.0).between(-5.0, 32.0, inclusive="both"))
        & (_numeric_series_or_default(metrics, "max_gain_10_pct", 0.0) < 34.0)
        & (metrics["amount"] >= 1.2e8)
        & (metrics["turnover"] >= 1.8)
        & (metrics["change_pct"] > 0.8)
        & (snapshot_upper_shadow <= 0.26)
        & (close_position_day >= 0.50)
    )
    strategy3_resonance_mask = (
        (board_resonance_mask & (metrics["resonance_quality"] >= 56.0))
        | ((industry_ret_2d >= 0.8) & (industry_up_count >= 2.0) & (metrics["market_resonance_score"].fillna(50.0) >= 54.0))
        | ((metrics["launch_readiness"] >= 62.0) & (metrics["long_setup_quality"] >= 58.0))
    )
    strategy3_veto_mask = (
        (metrics["crowding_risk"] >= 72.0)
        | (metrics["risk_of_late_entry"] >= 70.0)
        | volume_price_stall
        | ((snapshot_upper_shadow >= 0.22) & (snapshot_volume_ratio >= 1.75))
        | ((metrics["ret_20d_pct"] >= 28.0) & (metrics["max_gain_10_pct"] >= 28.0) & (metrics["change_pct"] >= 7.0))
    )
    strategy3_mask = (
        strategy3_structure_mask
        & strategy3_resonance_mask
        & (strategy3_core_score >= 56.0)
        & (~strategy3_veto_mask)
        & (~strategy1_mask)
        & (~strategy2_mask)
    )
    results: list[pd.DataFrame] = []
    if strategy1_mask.any():
        s1 = metrics.loc[strategy1_mask].copy()
        s1["candidate_strategy"] = "策略1"
        s1["strategy_pass"] = True
        s1["strategy_veto_reason"] = ""
        s1["strategy_bonus_reason"] = s1.apply(
            lambda row: (
                f"加分：行业2日强度 {float(row.get('industry_ret_2d_pct', 0.0)):.1f}，"
                f"启动状态 {row.get('launch_phase_label', '观察')}，"
                f"承接/共振质量 {float(row.get('resonance_quality', 50.0)):.1f}，"
                f"拥挤风险 {float(row.get('crowding_risk', 50.0)):.1f}。"
            ),
            axis=1,
        )
        s1["candidate_reason"] = s1.apply(
            lambda row: (
                f"策略1：15日涨幅 {float(row.get('ret_15d_pct', 0.0)):.1f}%，"
                f"回调 {int(row.get('pullback_days', 0) or 0)} 日且量缩守住MA10，"
                f"当日涨幅 {float(row.get('change_pct', 0.0)):.1f}%，"
                f"成交额 {float(row.get('amount', 0.0)) / 1e8:.2f} 亿，"
                f"换手 {float(row.get('turnover', 0.0)):.1f}%；"
                f"做多质量 {float(row.get('long_setup_quality', 50.0)):.1f}，"
                f"拥挤风险 {float(row.get('crowding_risk', 50.0)):.1f}，"
                f"主升标签 {row.get('launch_phase_label', '观察')}。"
            ),
            axis=1,
        )
        s1["strategy_rank"] = (
            pullback_acceptance_quality.loc[s1.index].fillna(50.0) * 0.25
            + _numeric_series_or_default(s1, "resonance_quality", 50.0) * 0.20
            + _numeric_series_or_default(s1, "long_setup_quality", 50.0) * 0.20
            + _numeric_series_or_default(s1, "launch_readiness", 50.0) * 0.15
            + _numeric_series_or_default(s1, "quant_score", 50.0) * 0.10
            - _numeric_series_or_default(s1, "crowding_risk", 50.0) * 0.10
            + strategy1_bonus_score.loc[s1.index].fillna(0.0)
        )
        s1["candidate_priority"] = s1["strategy_rank"]
        results.append(s1)
    if strategy2_mask.any():
        s2 = metrics.loc[strategy2_mask].copy()
        s2["candidate_strategy"] = "策略2"
        s2["strategy_pass"] = True
        s2["strategy_veto_reason"] = ""
        s2["strategy_bonus_reason"] = s2.apply(
            lambda row: (
                f"加分：突破质量 {float(row.get('breakout_quality', 50.0)):.1f}，"
                f"共振质量 {float(row.get('resonance_quality', 50.0)):.1f}，"
                f"板块前排质量 {float(row.get('board_resonance_strength', 50.0)):.1f}，"
                f"启动状态 {row.get('launch_phase_label', '观察')}，"
                f"拥挤风险 {float(row.get('crowding_risk', 50.0)):.1f}。"
            ),
            axis=1,
        )
        s2["candidate_reason"] = s2.apply(
            lambda row: (
                f"策略2：当日涨幅 {float(row.get('change_pct', 0.0)):.1f}%，"
                f"3/5日涨幅 {float(row.get('ret_3d_pct', 0.0)):.1f}%/{float(row.get('ret_5d_pct', 0.0)):.1f}%，"
                f"距10日高点 {float(row.get('distance_to_high_10_pct', 0.0)):.1f}%，"
                f"板块2日排名 {int(row.get('industry_rank_2d', 0) or 0)}，"
                f"板块上涨家数 {int(row.get('industry_up_count', 0) or 0)}；"
                f"突破质量 {float(row.get('breakout_quality', 50.0)):.1f}，"
                f"共振质量 {float(row.get('resonance_quality', 50.0)):.1f}，"
                f"板块前排质量 {float(row.get('board_resonance_strength', 50.0)):.1f}，"
                f"做多质量 {float(row.get('long_setup_quality', 50.0)):.1f}，"
                f"拥挤风险 {float(row.get('crowding_risk', 50.0)):.1f}，"
                f"主升标签 {row.get('launch_phase_label', '观察')}。"
            ),
            axis=1,
        )
        s2["strategy_rank"] = (
            _numeric_series_or_default(s2, "resonance_quality", 50.0) * 0.25
            + _numeric_series_or_default(s2, "breakout_quality", 50.0) * 0.25
            + _numeric_series_or_default(s2, "board_resonance_strength", 50.0) * 0.15
            + _numeric_series_or_default(s2, "launch_readiness", 50.0) * 0.15
            + _numeric_series_or_default(s2, "quant_score", 50.0) * 0.10
            - _numeric_series_or_default(s2, "crowding_risk", 50.0) * 0.10
            + strategy2_bonus_score.loc[s2.index].fillna(0.0)
        )
        s2["candidate_priority"] = s2["strategy_rank"]
        results.append(s2)
    if strategy3_mask.any():
        s3 = metrics.loc[strategy3_mask].copy()
        s3["candidate_strategy"] = "strategy3"
        s3["strategy_pass"] = True
        s3["strategy_veto_reason"] = ""
        s3["strategy_bonus_reason"] = s3.apply(
            lambda row: (
                f"strategy3 multi-factor: setup {float(row.get('long_setup_quality', 50.0)):.1f}, "
                f"resonance {float(row.get('resonance_quality', 50.0)):.1f}, "
                f"launch {float(row.get('launch_readiness', 50.0)):.1f}, "
                f"crowding {float(row.get('crowding_risk', 50.0)):.1f}."
            ),
            axis=1,
        )
        s3["candidate_reason"] = s3.apply(
            lambda row: (
                f"strategy3: broader multi-factor main-rise setup, "
                f"5d {float(row.get('ret_5d_pct', 0.0)):.1f}%, "
                f"20d {float(row.get('ret_20d_pct', 0.0)):.1f}%, "
                f"amount {float(row.get('amount', 0.0)) / 1e8:.2f}B CNY, "
                f"turnover {float(row.get('turnover', 0.0)):.1f}%, "
                f"sector up-count {int(row.get('industry_up_count', 0) or 0)}, "
                f"setup {float(row.get('long_setup_quality', 50.0)):.1f}, "
                f"crowding {float(row.get('crowding_risk', 50.0)):.1f}."
            ),
            axis=1,
        )
        s3["strategy_rank"] = strategy3_core_score.loc[s3.index].fillna(0.0)
        s3["candidate_priority"] = s3["strategy_rank"]
        results.append(s3)
    if not results:
        return pd.DataFrame()
    candidate_pool = pd.concat(results, ignore_index=True)
    candidate_pool = _apply_replay_strategy_fit(candidate_pool)
    candidate_pool = candidate_pool.sort_values(
        ["candidate_strategy", "strategy_rank", "strategy_fit_score", "amount", "turnover", "change_pct"],
        ascending=[True, False, False, False, False, False],
    )
    candidate_pool = candidate_pool.drop_duplicates("symbol", keep="first").head(
        max(RULE_BASED_CANDIDATE_POOL_SIZE, DYNAMIC_FALLBACK_ANALYSIS_BUFFER_SIZE)
    )
    keep_cols = [
        "symbol",
        "name",
        "industry_name",
        "analysis_date",
        "latest_price",
        "change_pct",
        "amount",
        "turnover",
        "candidate_strategy",
        "candidate_reason",
        "strategy_pass",
        "strategy_rank",
        "strategy_veto_reason",
        "strategy_bonus_reason",
        "candidate_priority",
        "strategy_fit_score",
        "replay_edge_score",
        "replay_profile_applied",
        "replay_profile_review_days",
        "replay_profile_market_days",
        "industry_ret_2d_pct",
        "industry_rank_2d",
        "industry_up_count",
        "consecutive_up_days",
        "stage_label",
        "stage_priority",
        "quant_score",
        "launch_score",
        "launch_readiness_score",
        "market_resonance_score",
        "launch_readiness",
        "breakout_quality",
        "resonance_quality",
        "board_resonance_strength",
        "long_setup_quality",
        "crowding_risk",
        "risk_of_late_entry",
        "launch_phase_label",
        "board_label",
        "price_limit_label",
    ]
    candidate_pool = candidate_pool[[column for column in keep_cols if column in candidate_pool.columns]].reset_index(drop=True)
    _write_market_candidate_pool_store(candidate_pool, market_data_date)
    return candidate_pool


def build_market_dynamic_fallback_pool_store(
    universe: pd.DataFrame,
    market_data_date: str | None,
    *,
    feature_store: pd.DataFrame | None = None,
    progress_callback: ProgressCallback | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    if universe.empty:
        return pd.DataFrame()
    cached = None if force_rebuild else read_market_dynamic_fallback_store(market_data_date)
    if cached is not None and not cached.empty:
        return _filter_strategy_eligible_frame(cached)

    store = feature_store
    if store is None or store.empty:
        store = build_market_daily_feature_store(
            universe,
            market_data_date,
            progress_callback=progress_callback,
            force_rebuild=force_rebuild,
        )
    if store.empty:
        return pd.DataFrame()

    metrics = _filter_strategy_eligible_frame(store.copy())
    if metrics.empty:
        return pd.DataFrame()
    metrics["consecutive_up_days"] = pd.to_numeric(metrics.get("consecutive_up_days"), errors="coerce").fillna(0).astype(int)
    metrics = metrics[metrics["consecutive_up_days"] >= int(MIN_FOCUS_CONSECUTIVE_UP_DAYS)].copy()
    if metrics.empty:
        return pd.DataFrame()

    metrics["candidate_strategy"] = "dynamic_fallback"
    metrics["candidate_reason"] = metrics.apply(
        lambda row: (
            f"Close-based fallback: {int(row.get('consecutive_up_days', 0) or 0)} up days, "
            f"5d {float(row.get('ret_5d_pct', 0.0) or 0.0):.1f}%, "
            f"stage {str(row.get('stage_label', '') or '')}, "
            f"quant {float(row.get('quant_score', 0.0) or 0.0):.1f}"
        ),
        axis=1,
    )
    metrics["candidate_priority"] = (
        metrics["consecutive_up_days"].fillna(0) * 8.0
        + metrics["launch_score"].fillna(0) * 0.32
        + _numeric_series_or_default(metrics, "launch_readiness_score", 50.0) * 0.18
        + _numeric_series_or_default(metrics, "market_resonance_score", 50.0) * 0.14
        + metrics["quant_score"].fillna(0) * 0.18
        + metrics["ret_5d_pct"].fillna(0) * 0.90
        + metrics["amount"].fillna(0) / 1e8 * 0.55
        + metrics["change_pct"].fillna(0) * 0.45
    )
    metrics = _apply_replay_strategy_fit(metrics)
    metrics = metrics.sort_values(
        ["candidate_priority", "strategy_fit_score", "consecutive_up_days", "launch_score", "quant_score", "amount", "change_pct"],
        ascending=False,
    )
    metrics = metrics.drop_duplicates("symbol", keep="first").head(
        max(DYNAMIC_FALLBACK_POOL_SIZE, DYNAMIC_FALLBACK_ANALYSIS_BUFFER_SIZE)
    )
    keep_cols = [
        "symbol",
        "name",
        "industry_name",
        "analysis_date",
        "latest_price",
        "change_pct",
        "amount",
        "turnover",
        "candidate_strategy",
        "candidate_reason",
        "candidate_priority",
        "strategy_fit_score",
        "replay_edge_score",
        "replay_profile_applied",
        "replay_profile_review_days",
        "replay_profile_market_days",
        "consecutive_up_days",
        "ret_3d_pct",
        "ret_5d_pct",
        "stage_label",
        "stage_priority",
        "quant_score",
        "launch_score",
        "launch_readiness_score",
        "market_resonance_score",
        "board_label",
        "price_limit_label",
    ]
    dynamic_pool = metrics[[column for column in keep_cols if column in metrics.columns]].reset_index(drop=True)
    _write_market_dynamic_fallback_store(dynamic_pool, market_data_date)
    return dynamic_pool


__all__ = [
    "MARKET_DAILY_FEATURE_STORE_VERSION",
    "MARKET_CANDIDATE_POOL_STORE_VERSION",
    "MARKET_DYNAMIC_FALLBACK_STORE_VERSION",
    "align_daily_history_to_market_date",
    "build_market_candidate_pool_store",
    "build_market_daily_feature_store",
    "build_market_dynamic_fallback_pool_store",
    "get_market_daily_feature_row",
    "load_incremental_market_snapshot_history",
    "market_candidate_pool_store_path",
    "market_daily_feature_store_path",
    "market_dynamic_fallback_store_path",
    "read_market_snapshot_history_store",
    "read_market_candidate_pool_store",
    "read_market_daily_feature_store",
    "read_market_dynamic_fallback_store",
]
