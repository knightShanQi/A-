from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from a_share_predictor.database_source import _connect, _query_frame, _table_name, load_env_file

load_env_file()

from a_share_predictor.data import _call_tushare_api, fetch_tushare_recent_trade_dates, fetch_tushare_stock_basic_all_statuses
from a_share_predictor.features import FEATURE_COLUMNS, build_daily_features
from a_share_predictor.market_backtest_runner import (
    _add_fast_industry_metrics,
    _add_fast_rolling_metrics,
    _build_fast_history_lookup,
    _build_fast_strategy_candidates,
    _evaluate_forward_return_from_lookup,
    _fetch_fast_daily_snapshots_for_dates,
    _main_board_non_st_mask,
    _normalize_snapshot_history_for_fast_backtest,
)
from a_share_predictor.modeling import (
    EXTERNAL_SNAPSHOT_COLUMNS,
    MARKET_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    _append_market_regime_features,
    _append_market_resonance_features,
    _apply_incremental_probability_upgrade,
    _augment_model_features,
    _ensemble_probability,
    load_cached_market_wide_model,
    load_market_proxy_model,
)
from a_share_predictor.portfolio_backtester import (
    PortfolioBacktestConfig,
    PortfolioBacktestResult,
    simulate_portfolio_from_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOP10_OUTPUT_DIR = PROJECT_ROOT / ".cache" / "strategy_model_top10_backtest"
DAILY_BY_DATE_CACHE_DIR = PROJECT_ROOT / ".cache" / "tushare_daily_by_date_v1"


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


def _market_suffix(symbol: str) -> str:
    text = str(symbol).zfill(6)
    if text.startswith(("600", "601", "603", "605", "688", "689")):
        return "SH"
    if text.startswith(("000", "001", "002", "003", "300", "301")):
        return "SZ"
    if text.startswith(("430", "830", "831", "832", "833", "834", "835", "836", "837", "838", "839", "870", "871", "872", "873", "874", "875", "876", "877", "878", "879", "920")):
        return "BJ"
    return ""


def _latest_available_trade_date() -> str:
    try:
        recent = fetch_tushare_recent_trade_dates(end_date=dt.date.today().strftime("%Y%m%d"), limit=1)
        if recent:
            latest = _format_date(recent[-1])
            if latest:
                return latest
    except Exception:
        pass
    table = _table_name()
    with _connect() as connection:
        try:
            frame = _query_frame(connection, f"select max(trade_date) as trade_date from {table}")
        except Exception:
            frame = _query_frame(connection, f"select max(day_value) as trade_date from {table}, unnest(dates) as day_value")
    if frame.empty:
        raise RuntimeError("No trade dates found in database.")
    latest = _format_date(frame.loc[0, "trade_date"])
    if not latest:
        raise RuntimeError("Latest database trade date is invalid.")
    return latest


def _resolve_window(date_from: str | None, date_to: str | None, months: int) -> tuple[str, str]:
    end = pd.to_datetime(date_to or _latest_available_trade_date(), errors="coerce")
    if pd.isna(end):
        raise ValueError(f"Invalid date_to: {date_to}")
    if date_from:
        start = pd.to_datetime(date_from, errors="coerce")
        if pd.isna(start):
            raise ValueError(f"Invalid date_from: {date_from}")
    else:
        start = end - pd.DateOffset(months=int(months))
    return pd.Timestamp(start).strftime("%Y-%m-%d"), pd.Timestamp(end).strftime("%Y-%m-%d")


def _fetch_database_history(date_from: str, date_to: str, *, lookback_days: int, forward_days: int) -> pd.DataFrame:
    start = pd.to_datetime(date_from, errors="coerce") - pd.Timedelta(days=max(int(lookback_days), 180))
    end = pd.to_datetime(date_to, errors="coerce") + pd.Timedelta(days=max(int(forward_days), 10))
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Invalid history date window.")
    table = _table_name()
    chunk_days = 10
    chunk_cursor = pd.Timestamp(start)
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    while chunk_cursor <= end:
        chunk_start = pd.Timestamp(chunk_cursor)
        chunk_end = min(end, chunk_start + pd.Timedelta(days=chunk_days - 1))
        ranges.append((pd.Timestamp(chunk_start), pd.Timestamp(chunk_end)))
        chunk_cursor = chunk_end + pd.Timedelta(days=1)

    frames: list[pd.DataFrame] = []
    for index, (chunk_start, chunk_end) in enumerate(ranges, start=1):
        print(f"[fetch] {index}/{len(ranges)} {chunk_start:%Y-%m-%d} -> {chunk_end:%Y-%m-%d}", flush=True)
        chunk = pd.DataFrame()
        for attempt in range(1, 4):
            try:
                with _connect() as connection:
                    chunk = _query_frame(
                        connection,
                        f"""
                        select symbol, name, trade_date, open, high, low, close, pre_close,
                               change, pct_chg, volume, amount, turnover_rate
                        from {table}
                        where trade_date >= %s::date
                          and trade_date <= %s::date
                        """,
                        (chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")),
                    )
                break
            except Exception:
                if attempt >= 3:
                    raise
                time.sleep(2.0 * attempt)
        if not chunk.empty:
            frames.append(chunk)
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["symbol", "trade_date", "close"]).copy()
    frame["name"] = frame.get("name", "").fillna("").astype(str)
    frame["name"] = frame["name"].where(frame["name"].str.len().gt(0), frame["symbol"])
    for column in ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "volume", "amount", "turnover_rate"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    close = pd.to_numeric(frame.get("close"), errors="coerce")
    pre_close = pd.to_numeric(frame.get("pre_close"), errors="coerce")
    base = pre_close.where(pre_close.ne(0))
    frame["change"] = pd.to_numeric(frame.get("change"), errors="coerce").where(
        pd.to_numeric(frame.get("change"), errors="coerce").notna(),
        close - pre_close,
    )
    frame["change_pct"] = pd.to_numeric(frame.get("pct_chg"), errors="coerce").where(
        pd.to_numeric(frame.get("pct_chg"), errors="coerce").notna(),
        (close / base - 1.0) * 100.0,
    )
    frame["turnover"] = pd.to_numeric(frame.get("turnover_rate"), errors="coerce")
    frame["vol"] = pd.to_numeric(frame.get("volume"), errors="coerce")
    frame["market"] = ""
    frame["industry"] = ""
    return frame.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def _normalize_tushare_daily_raw(raw: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame()
    merged = raw.copy().drop_duplicates().reset_index(drop=True)
    try:
        stock_basic = fetch_tushare_stock_basic_all_statuses()[["ts_code", "symbol", "name", "industry", "market"]].copy()
    except Exception:
        stock_basic = pd.DataFrame()
    if not stock_basic.empty and "ts_code" in merged.columns:
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
    return merged.dropna(subset=["trade_date"]).sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def _fetch_tushare_history(date_from: str, date_to: str, *, lookback_days: int, forward_days: int) -> pd.DataFrame:
    start = pd.to_datetime(date_from, errors="coerce") - pd.Timedelta(days=max(int(lookback_days), 180))
    end = pd.to_datetime(date_to, errors="coerce") + pd.Timedelta(days=max(int(forward_days), 10))
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Invalid Tushare history date window.")
    trade_dates = _fetch_tushare_trade_calendar(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
    if not trade_dates:
        trade_dates = [ts.strftime("%Y%m%d") for ts in pd.date_range(start, end, freq="B")]

    fast_history = pd.DataFrame()
    try:
        print(f"[fetch_fast] monthly range load {trade_dates[0]} -> {trade_dates[-1]} ({len(trade_dates)} trade dates)", flush=True)
        fast_history = _fetch_fast_daily_snapshots_for_dates(trade_dates)
    except Exception as exc:
        print(f"[fetch_fast] failed: {exc}", flush=True)

    expected = list(dict.fromkeys(str(value).replace("-", "") for value in trade_dates))
    missing_dates = expected
    frames: list[pd.DataFrame] = []
    if isinstance(fast_history, pd.DataFrame) and not fast_history.empty and "trade_date" in fast_history.columns:
        fast_history = fast_history.copy()
        fast_history["trade_date"] = pd.to_datetime(fast_history["trade_date"], errors="coerce")
        fast_history = fast_history.dropna(subset=["trade_date"]).copy()
        date_key = fast_history["trade_date"].dt.strftime("%Y%m%d")
        date_counts = date_key.value_counts()
        missing_dates = [value for value in expected if int(date_counts.get(value, 0)) < 2500]
        covered = len(expected) - len(missing_dates)
        print(f"[fetch_fast] covered {covered}/{len(expected)} dates; fill {len(missing_dates)}", flush=True)
        frames.append(fast_history)

    if missing_dates:
        raw_frames: list[pd.DataFrame] = []
        total = len(missing_dates)
        for index, trade_date in enumerate(missing_dates, start=1):
            if index == 1 or index % 50 == 0:
                print(f"[fetch_daily_fill] {index}/{total} {trade_date}", flush=True)
            frame = _fetch_tushare_daily_by_date(trade_date)
            if not frame.empty:
                raw_frames.append(frame)
        filled = _normalize_tushare_daily_raw(pd.concat(raw_frames, ignore_index=True)) if raw_frames else pd.DataFrame()
        if not filled.empty:
            frames.append(filled)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["trade_date"] = pd.to_datetime(combined["trade_date"], errors="coerce")
    combined["symbol"] = combined["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    combined = combined.dropna(subset=["trade_date"]).copy()
    combined = combined.drop_duplicates(["symbol", "trade_date"], keep="last")
    return combined.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def _fetch_tushare_trade_calendar(start_date: str, end_date: str) -> list[str]:
    cache_path = DAILY_BY_DATE_CACHE_DIR / f"trade_cal_open_{start_date}_{end_date}.json"
    if cache_path.exists():
        try:
            values = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(values, list) and values:
                return [str(value) for value in values]
        except Exception:
            pass
    try:
        frame = _call_tushare_api(
            "trade_cal",
            params={"exchange": "", "start_date": start_date, "end_date": end_date, "is_open": 1},
            fields="cal_date,is_open",
        )
    except Exception:
        return []
    if frame.empty or "cal_date" not in frame.columns:
        return []
    if "is_open" in frame.columns:
        frame = frame.loc[pd.to_numeric(frame["is_open"], errors="coerce").fillna(0).astype(int).eq(1)].copy()
    if frame.empty:
        return []
    values = sorted(frame["cal_date"].astype(str).unique().tolist())
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(values), encoding="utf-8")
    except Exception:
        pass
    return values


def _fetch_tushare_daily_by_date(trade_date: str) -> pd.DataFrame:
    safe_date = str(trade_date).replace("-", "")
    cache_path = DAILY_BY_DATE_CACHE_DIR / safe_date[:4] / f"{safe_date}.pkl"
    if cache_path.exists():
        try:
            cached = pd.read_pickle(cache_path)
            if isinstance(cached, pd.DataFrame) and not cached.empty:
                return cached.copy()
        except Exception:
            pass
        try:
            cache_path.unlink(missing_ok=True)
        except Exception:
            pass
    fields = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
    frame = pd.DataFrame()
    for attempt in range(1, 4):
        try:
            if attempt > 1:
                time.sleep(1.5 * attempt)
            frame = _call_tushare_api("daily", params={"trade_date": safe_date}, fields=fields)
            if not frame.empty:
                break
        except Exception:
            if attempt >= 3:
                return pd.DataFrame()
    if frame.empty:
        return pd.DataFrame()
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(cache_path)
    except Exception:
        pass
    time.sleep(0.04)
    return frame.copy()


def _prepare_strategy_history(raw_history: pd.DataFrame) -> pd.DataFrame:
    history = _normalize_snapshot_history_for_fast_backtest(raw_history)
    if history.empty:
        return history
    history = _add_fast_rolling_metrics(history)
    history = _add_fast_industry_metrics(history)
    return history.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def _candidate_pool_for_dates(
    history: pd.DataFrame,
    trade_dates: list[str],
    *,
    candidate_pool_limit: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    total = len(trade_dates)
    for index, market_date in enumerate(trade_dates, start=1):
        if index == 1 or index % 20 == 0:
            print(f"[candidate_pool] {index}/{total} {market_date}", flush=True)
        candidates = _build_fast_strategy_candidates(
            history,
            market_date,
            strategy_mode="all",
            top_k=max(int(candidate_pool_limit), 1),
        )
        if candidates.empty:
            continue
        candidates = candidates.copy()
        candidates["market_date"] = market_date
        frames.append(candidates)
    if not frames:
        return pd.DataFrame()
    pool = pd.concat(frames, ignore_index=True)
    pool["symbol"] = pool["symbol"].astype(str).str.zfill(6)
    pool["market_date"] = pd.to_datetime(pool["market_date"], errors="coerce")
    pool = pool.dropna(subset=["market_date", "symbol"]).copy()
    pool = pool.sort_values(["market_date", "symbol", "candidate_priority"], ascending=[True, True, False])
    pool = pool.drop_duplicates(["market_date", "symbol"], keep="first").reset_index(drop=True)
    return pool


def _build_feature_frame(history: pd.DataFrame, signal_dates: pd.DatetimeIndex) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    signal_dates = pd.DatetimeIndex(pd.to_datetime(signal_dates, errors="coerce").dropna().unique())
    if signal_dates.empty:
        return pd.DataFrame()
    signal_start = signal_dates.min()
    signal_end = signal_dates.max()
    main_history = history.loc[_main_board_non_st_mask(history)].copy()
    total_symbols = int(main_history["symbol"].nunique())
    for index, (symbol, group) in enumerate(main_history.groupby("symbol", sort=True), start=1):
        if index == 1 or index % 500 == 0:
            print(f"[features] {index}/{total_symbols} symbols", flush=True)
        daily = group.sort_values("trade_date").copy()
        daily["date"] = pd.to_datetime(daily["trade_date"], errors="coerce")
        daily = daily.dropna(subset=["date", "open", "high", "low", "close"]).copy()
        if len(daily) < 80:
            continue
        for column in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
            if column not in daily.columns:
                daily[column] = np.nan
            daily[column] = pd.to_numeric(daily[column], errors="coerce")
        daily["volume"] = daily["volume"].fillna(0.0)
        daily["amount"] = daily["amount"].fillna(0.0)
        daily = daily.set_index("date", drop=False)
        try:
            features = build_daily_features(daily)
        except Exception:
            continue
        if features.empty:
            continue
        features = features.loc[(features.index >= signal_start) & (features.index <= signal_end)].copy()
        if features.empty:
            continue
        features["symbol"] = str(symbol).zfill(6)
        features["trade_date"] = pd.to_datetime(features.index, errors="coerce")
        features["name"] = str(daily["name"].dropna().iloc[-1]) if "name" in daily.columns and daily["name"].notna().any() else str(symbol).zfill(6)
        frames.append(features.reset_index(drop=True))
    if not frames:
        return pd.DataFrame()
    feature_frame = pd.concat(frames, ignore_index=True)
    feature_frame = feature_frame.dropna(subset=["symbol", "trade_date"]).copy()
    return _attach_fast_context_features(feature_frame)


def _attach_fast_context_features(feature_frame: pd.DataFrame) -> pd.DataFrame:
    enriched = _augment_model_features(feature_frame)
    enriched["trade_date"] = pd.to_datetime(enriched["trade_date"], errors="coerce")
    date_stats = (
        enriched.groupby("trade_date", dropna=True)
        .agg(
            market_ret_5=("ret_5", "mean"),
            market_ret_20=("ret_20", "mean"),
            market_close_vs_ma20=("close_vs_ma20", "mean"),
            market_volatility_10=("volatility_10", "mean"),
            market_range_position_20=("range_position_20", "mean"),
        )
        .reset_index()
    )
    enriched = enriched.merge(date_stats, on="trade_date", how="left")
    enriched["relative_strength_5"] = pd.to_numeric(enriched.get("ret_5"), errors="coerce").fillna(0.0) - pd.to_numeric(
        enriched.get("market_ret_5"), errors="coerce"
    ).fillna(0.0)
    enriched["relative_strength_20"] = pd.to_numeric(enriched.get("ret_20"), errors="coerce").fillna(0.0) - pd.to_numeric(
        enriched.get("market_ret_20"), errors="coerce"
    ).fillna(0.0)
    for column in MARKET_FEATURE_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = 0.0
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(0.0)
    for column in EXTERNAL_SNAPSHOT_COLUMNS:
        if column in {"news_positive_ratio_7d", "fund_positive_ratio_5d"}:
            enriched[column] = 0.5
        else:
            enriched[column] = 0.0
    enriched = _append_market_regime_features(enriched)
    enriched = _append_market_resonance_features(enriched)
    return enriched.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _legacy_proxy_paths(horizon_days: int, positive_return: float) -> list[Path]:
    safe_return = int(float(positive_return) * 10000)
    return sorted(
        (PROJECT_ROOT / ".cache").glob(f"global_market_proxy*_h{int(horizon_days)}_r{safe_return}_*.pkl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _load_rank_model(horizon_days: int, positive_return: float):
    wide_model = load_cached_market_wide_model(horizon_days=horizon_days, positive_return=positive_return)
    if wide_model is not None:
        return "market_wide", wide_model
    proxy_model = load_market_proxy_model(horizon_days=horizon_days, positive_return=positive_return)
    if proxy_model is not None:
        return f"market_proxy:{getattr(proxy_model, 'candidate_name', 'proxy')}", proxy_model
    for path in _legacy_proxy_paths(horizon_days, positive_return):
        try:
            with path.open("rb") as handle:
                model = pickle.load(handle)
        except Exception:
            continue
        if hasattr(model, "fitted_model"):
            return f"legacy_market_proxy:{getattr(model, 'candidate_name', path.stem)}", model
    raise RuntimeError("No market-wide or proxy model cache is available for ranking.")


def _align_features_for_estimator(model, features: pd.DataFrame) -> pd.DataFrame:
    estimator = getattr(model, "fitted_model", model)
    feature_names = None
    try:
        feature_names = getattr(estimator.named_steps.get("imputer"), "feature_names_in_", None)
    except Exception:
        feature_names = None
    if feature_names is None:
        feature_names = getattr(estimator, "feature_names_in_", None)
    if feature_names is None:
        return features
    return features.reindex(columns=list(feature_names), fill_value=0.0)


def _score_candidates(scored_frame: pd.DataFrame, *, horizon_days: int, positive_return: float) -> tuple[pd.DataFrame, str]:
    model_kind, model = _load_rank_model(horizon_days, positive_return)
    features = scored_frame.reindex(columns=MODEL_FEATURE_COLUMNS, fill_value=0.0)
    if model_kind == "market_wide":
        probabilities, _ = _ensemble_probability(
            model.fitted_models,
            features,
            weights=model.ensemble_weights,
            calibrator=model.calibrator,
            calibration_feature_frame=scored_frame,
            regime_calibrators=model.regime_calibrators,
        )
    else:
        proxy_features = _align_features_for_estimator(model, features)
        probabilities = model.fitted_model.predict_proba(proxy_features)[:, 1]
    probabilities, enhancement = _apply_incremental_probability_upgrade(probabilities, features)
    scored = scored_frame.copy()
    scored["model_probability"] = probabilities
    scored["model_score"] = scored["model_probability"] * 100.0
    if not enhancement.empty and "context_composite_score" in enhancement.columns:
        scored["context_composite_score"] = enhancement["context_composite_score"].to_numpy(dtype=float)
    return scored, model_kind


def _top_n_by_day(scored: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    ranked = scored.sort_values(
        ["market_date", "model_probability", "candidate_priority", "amount"],
        ascending=[True, False, False, False],
    )
    selected = ranked.groupby("market_date", group_keys=False).head(max(int(top_n), 1)).copy()
    selected["daily_rank"] = selected.groupby("market_date").cumcount() + 1
    return selected.reset_index(drop=True)


def _evaluate_top_selection(top_selection: pd.DataFrame, history: pd.DataFrame, hold_days: int) -> pd.DataFrame:
    lookup = _build_fast_history_lookup(history)
    rows: list[dict[str, object]] = []
    records = top_selection.to_dict("records")
    total = len(records)
    for index, row in enumerate(records, start=1):
        if index == 1 or index % 1000 == 0:
            print(f"[forward] {index}/{total}", flush=True)
        market_date = _format_date(row.get("market_date"))
        forward = _evaluate_forward_return_from_lookup(
            lookup,
            str(row.get("symbol", "")).zfill(6),
            market_date,
            int(hold_days),
        )
        rows.append({**row, **forward})
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    result["market_date"] = pd.to_datetime(result["market_date"], errors="coerce")
    for column in ["hold_1d_return", "hold_3d_return", "hold_5d_return", "forward_return", "max_high_return", "max_drawdown"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _daily_portfolio_frame(results: pd.DataFrame, hold_days: int, positive_return: float) -> pd.DataFrame:
    return_column = f"hold_{int(hold_days)}d_return"
    available_column = f"hold_{int(hold_days)}d_available"
    if results.empty or return_column not in results.columns:
        return pd.DataFrame()
    available = results.copy()
    if available_column in available.columns:
        available = available[available[available_column].astype(bool)].copy()
    available = available.dropna(subset=[return_column])
    if available.empty:
        return pd.DataFrame()
    daily = (
        available.groupby("market_date", as_index=False)
        .agg(
            selected=("symbol", "count"),
            avg_model_probability=("model_probability", "mean"),
            avg_model_score=("model_score", "mean"),
            avg_hold_return=(return_column, "mean"),
            win_rate=(return_column, lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
            target_hit_rate=(return_column, lambda s: float((pd.to_numeric(s, errors="coerce") >= float(positive_return)).mean())),
            avg_intrahold_drawdown=("max_drawdown", "mean"),
        )
        .sort_values("market_date")
        .reset_index(drop=True)
    )
    daily["equity"] = (1.0 + daily["avg_hold_return"].fillna(0.0)).cumprod()
    daily["running_max"] = daily["equity"].cummax()
    daily["drawdown"] = daily["equity"] / daily["running_max"].replace(0.0, np.nan) - 1.0
    return daily


def _summarize(
    *,
    date_from: str,
    date_to: str,
    history: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    scored_candidates: pd.DataFrame,
    results: pd.DataFrame,
    daily: pd.DataFrame,
    top_n: int,
    hold_days: int,
    model_horizon_days: int,
    positive_return: float,
    model_kind: str,
) -> dict[str, object]:
    return_column = f"hold_{int(hold_days)}d_return"
    available_column = f"hold_{int(hold_days)}d_available"
    available = results.copy()
    if not available.empty and available_column in available.columns:
        available = available[available[available_column].astype(bool)].copy()
    if not available.empty and return_column in available.columns:
        available = available.dropna(subset=[return_column])
    trade_returns = pd.to_numeric(available.get(return_column, pd.Series(dtype=float)), errors="coerce").dropna()
    equity_end = _safe_float(daily["equity"].iloc[-1], 1.0) if not daily.empty else 1.0
    trading_day_count = int(len(daily))
    annualized_return = equity_end ** (252.0 / trading_day_count) - 1.0 if trading_day_count > 0 and equity_end > 0 else 0.0
    max_drawdown = _safe_float(daily["drawdown"].min(), 0.0) if not daily.empty else 0.0
    daily_win_rate = float((daily["avg_hold_return"] > 0).mean()) if not daily.empty else 0.0

    strategy_breakdown: list[dict[str, object]] = []
    if not available.empty and "candidate_strategy" in available.columns:
        for strategy, group in available.groupby("candidate_strategy", dropna=False):
            values = pd.to_numeric(group[return_column], errors="coerce").dropna()
            if values.empty:
                continue
            strategy_breakdown.append(
                {
                    "candidate_strategy": str(strategy),
                    "trade_count": int(len(values)),
                    "avg_hold_return": round(float(values.mean()), 6),
                    "win_rate": round(float((values > 0).mean()), 4),
                    "target_hit_rate": round(float((values >= float(positive_return)).mean()), 4),
                }
            )

    signal_dates = pd.to_datetime(results.get("market_date", pd.Series(dtype="datetime64[ns]")), errors="coerce")
    evaluated_dates = pd.to_datetime(daily.get("market_date", pd.Series(dtype="datetime64[ns]")), errors="coerce")
    latest_history_date = _format_date(history["trade_date"].max()) if not history.empty else ""
    return {
        "date_from": str(date_from),
        "date_to": str(date_to),
        "latest_history_date": latest_history_date,
        "evaluated_signal_start": _format_date(evaluated_dates.min()) if len(evaluated_dates) else "",
        "evaluated_signal_end": _format_date(evaluated_dates.max()) if len(evaluated_dates) else "",
        "top_n": int(top_n),
        "hold_days": int(hold_days),
        "positive_return": float(positive_return),
        "model_horizon_days": int(model_horizon_days),
        "model_kind": model_kind,
        "history_rows": int(len(history)),
        "history_symbols": int(history["symbol"].nunique()) if not history.empty else 0,
        "candidate_rows": int(len(candidate_pool)),
        "candidate_days": int(candidate_pool["market_date"].nunique()) if not candidate_pool.empty else 0,
        "scored_candidate_rows": int(len(scored_candidates)),
        "selected_rows": int(len(results)),
        "selected_signal_days": int(signal_dates.nunique()) if len(signal_dates) else 0,
        "evaluated_trade_count": int(len(trade_returns)),
        "evaluated_trading_day_count": trading_day_count,
        "avg_model_probability": round(float(available["model_probability"].mean()), 6) if not available.empty else 0.0,
        "avg_model_score": round(float(available["model_score"].mean()), 4) if not available.empty else 0.0,
        "avg_hold_return": round(float(trade_returns.mean()), 6) if not trade_returns.empty else 0.0,
        "median_hold_return": round(float(trade_returns.median()), 6) if not trade_returns.empty else 0.0,
        "daily_avg_hold_return": round(float(daily["avg_hold_return"].mean()), 6) if not daily.empty else 0.0,
        "model_win_rate": round(float((trade_returns > 0).mean()), 4) if not trade_returns.empty else 0.0,
        "daily_batch_win_rate": round(daily_win_rate, 4),
        "target_hit_rate": round(float((trade_returns >= float(positive_return)).mean()), 4) if not trade_returns.empty else 0.0,
        "annualized_return": round(float(annualized_return), 6),
        "max_drawdown": round(float(max_drawdown), 6),
        "ending_equity": round(float(equity_end), 6),
        "avg_intrahold_drawdown": round(float(pd.to_numeric(available.get("max_drawdown"), errors="coerce").mean()), 6)
        if not available.empty and "max_drawdown" in available.columns
        else 0.0,
        "worst_trade_return": round(float(trade_returns.min()), 6) if not trade_returns.empty else 0.0,
        "best_trade_return": round(float(trade_returns.max()), 6) if not trade_returns.empty else 0.0,
        "strategy_breakdown": strategy_breakdown,
    }


def _daily_from_portfolio_result(result: PortfolioBacktestResult) -> pd.DataFrame:
    daily = result.daily_nav.copy()
    if daily.empty:
        return daily
    daily = daily.rename(columns={"trade_date": "market_date"})
    daily["market_date"] = pd.to_datetime(daily["market_date"], errors="coerce")
    daily["running_max"] = pd.to_numeric(daily["equity"], errors="coerce").cummax()
    daily["drawdown"] = pd.to_numeric(daily["equity"], errors="coerce") / daily["running_max"].replace(0.0, np.nan) - 1.0
    return daily


def _merge_portfolio_and_forward_diagnostics(
    portfolio_trades: pd.DataFrame,
    forward_results: pd.DataFrame,
    hold_days: int,
) -> pd.DataFrame:
    if portfolio_trades.empty:
        return portfolio_trades.copy()
    trades = portfolio_trades.copy()
    diagnostics = forward_results.copy()
    hold_column = f"hold_{int(hold_days)}d_return"
    available_column = f"hold_{int(hold_days)}d_available"
    join_columns = [column for column in ["symbol", "market_date", "entry_date", "exit_date"] if column in trades.columns and column in diagnostics.columns]
    if join_columns:
        extra_columns = [
            column
            for column in [
                hold_column,
                available_column,
                "hold_1d_return",
                "hold_3d_return",
                "hold_5d_return",
                "forward_return",
                "max_high_return",
                "max_drawdown",
            ]
            if column in diagnostics.columns
        ]
        diagnostics = diagnostics[join_columns + extra_columns].drop_duplicates(join_columns)
        trades = trades.merge(diagnostics, on=join_columns, how="left")
    if hold_column not in trades.columns:
        trades[hold_column] = trades.get("gross_return")
    if "forward_return" not in trades.columns:
        trades["forward_return"] = trades.get("gross_return")
    return trades


def _summarize_with_portfolio_engine(
    *,
    date_from: str,
    date_to: str,
    history: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    scored_candidates: pd.DataFrame,
    top_selection: pd.DataFrame,
    portfolio_result: PortfolioBacktestResult,
    diagnostic_results: pd.DataFrame,
    daily: pd.DataFrame,
    top_n: int,
    hold_days: int,
    model_horizon_days: int,
    positive_return: float,
    model_kind: str,
) -> dict[str, object]:
    trades = portfolio_result.trades.copy()
    gross_returns = pd.to_numeric(trades.get("gross_return", pd.Series(dtype=float)), errors="coerce").dropna()
    net_returns = pd.to_numeric(trades.get("net_return", pd.Series(dtype=float)), errors="coerce").dropna()
    signal_dates = pd.to_datetime(top_selection.get("market_date", pd.Series(dtype="datetime64[ns]")), errors="coerce")
    evaluated_dates = pd.to_datetime(daily.get("market_date", pd.Series(dtype="datetime64[ns]")), errors="coerce")
    latest_history_date = _format_date(history["trade_date"].max()) if not history.empty else ""

    target_values = pd.to_numeric(
        diagnostic_results.get(f"hold_{int(hold_days)}d_return", diagnostic_results.get("forward_return", pd.Series(dtype=float))),
        errors="coerce",
    ).dropna()
    strategy_breakdown: list[dict[str, object]] = []
    if not trades.empty and "candidate_strategy" in trades.columns:
        for strategy, group in trades.groupby("candidate_strategy", dropna=False):
            values = pd.to_numeric(group.get("net_return", pd.Series(dtype=float)), errors="coerce").dropna()
            if values.empty:
                continue
            strategy_breakdown.append(
                {
                    "candidate_strategy": str(strategy),
                    "trade_count": int(len(values)),
                    "avg_hold_return": round(float(values.mean()), 6),
                    "win_rate": round(float((values > 0).mean()), 4),
                    "target_hit_rate": round(float((values >= float(positive_return)).mean()), 4),
                }
            )

    return {
        "date_from": str(date_from),
        "date_to": str(date_to),
        "latest_history_date": latest_history_date,
        "evaluated_signal_start": _format_date(evaluated_dates.min()) if len(evaluated_dates) else "",
        "evaluated_signal_end": _format_date(evaluated_dates.max()) if len(evaluated_dates) else "",
        "top_n": int(top_n),
        "hold_days": int(hold_days),
        "positive_return": float(positive_return),
        "model_horizon_days": int(model_horizon_days),
        "model_kind": model_kind,
        "history_rows": int(len(history)),
        "history_symbols": int(history["symbol"].nunique()) if not history.empty else 0,
        "candidate_rows": int(len(candidate_pool)),
        "candidate_days": int(candidate_pool["market_date"].nunique()) if not candidate_pool.empty else 0,
        "scored_candidate_rows": int(len(scored_candidates)),
        "selected_rows": int(len(top_selection)),
        "selected_signal_days": int(signal_dates.nunique()) if len(signal_dates) else 0,
        "evaluated_trade_count": int(len(trades)),
        "evaluated_trading_day_count": int(len(daily)),
        "avg_model_probability": round(float(pd.to_numeric(trades.get("model_probability"), errors="coerce").mean()), 6)
        if not trades.empty and "model_probability" in trades.columns
        else 0.0,
        "avg_model_score": round(float(pd.to_numeric(trades.get("model_score"), errors="coerce").mean()), 4)
        if not trades.empty and "model_score" in trades.columns
        else 0.0,
        "avg_hold_return": round(float(net_returns.mean()), 6) if not net_returns.empty else 0.0,
        "median_hold_return": round(float(net_returns.median()), 6) if not net_returns.empty else 0.0,
        "daily_avg_hold_return": round(float(pd.to_numeric(daily.get("equity"), errors="coerce").pct_change().dropna().mean()), 6)
        if not daily.empty
        else 0.0,
        "model_win_rate": round(float((net_returns > 0).mean()), 4) if not net_returns.empty else 0.0,
        "daily_batch_win_rate": round(float((pd.to_numeric(daily.get("equity"), errors="coerce").pct_change().dropna() > 0).mean()), 4)
        if not daily.empty and len(daily) > 1
        else 0.0,
        "target_hit_rate": round(float((target_values >= float(positive_return)).mean()), 4) if not target_values.empty else 0.0,
        "annualized_return": round(float(portfolio_result.summary.get("annualized_return", 0.0)), 6),
        "max_drawdown": round(float(portfolio_result.summary.get("max_drawdown", 0.0)), 6),
        "ending_equity": round(float(portfolio_result.summary.get("ending_equity", 0.0)), 6),
        "cumulative_return": round(float(portfolio_result.summary.get("cumulative_return", 0.0)), 6),
        "avg_intrahold_drawdown": round(float(pd.to_numeric(diagnostic_results.get("max_drawdown"), errors="coerce").mean()), 6)
        if not diagnostic_results.empty and "max_drawdown" in diagnostic_results.columns
        else 0.0,
        "worst_trade_return": round(float(net_returns.min()), 6) if not net_returns.empty else 0.0,
        "best_trade_return": round(float(net_returns.max()), 6) if not net_returns.empty else 0.0,
        "avg_gross_return": round(float(gross_returns.mean()), 6) if not gross_returns.empty else 0.0,
        "trade_count": int(portfolio_result.summary.get("trade_count", 0)),
        "win_rate": round(float(portfolio_result.summary.get("win_rate", 0.0)), 4),
        "avg_net_return": round(float(portfolio_result.summary.get("avg_net_return", 0.0)), 6),
        "strategy_breakdown": strategy_breakdown,
    }


def run_strategy_model_top10_backtest(
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    months: int = 6,
    hold_days: int = 3,
    top_n: int = 10,
    candidate_pool_limit: int = 5000,
    model_horizon_days: int = 5,
    positive_return: float = 0.03,
    output_dir: str | Path = DEFAULT_TOP10_OUTPUT_DIR,
) -> dict[str, object]:
    resolved_from, resolved_to = _resolve_window(date_from, date_to, months)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[prepare] window {resolved_from} -> {resolved_to}", flush=True)
    raw_history = _fetch_tushare_history(
        resolved_from,
        resolved_to,
        lookback_days=260,
        forward_days=int(hold_days) + 8,
    )
    if raw_history.empty:
        raise RuntimeError("No database daily history found for the requested window.")
    history = _prepare_strategy_history(raw_history)
    if history.empty:
        raise RuntimeError("No normalized market history available after preparation.")
    signal_start = pd.to_datetime(resolved_from)
    signal_end = pd.to_datetime(resolved_to)
    trade_dates = sorted(
        pd.to_datetime(
            history.loc[history["trade_date"].between(signal_start, signal_end, inclusive="both"), "trade_date"],
            errors="coerce",
        )
        .dropna()
        .dt.strftime("%Y-%m-%d")
        .unique()
        .tolist()
    )
    if not trade_dates:
        raise RuntimeError("No trade dates found inside the requested signal window.")
    candidate_pool = _candidate_pool_for_dates(
        history,
        trade_dates,
        candidate_pool_limit=candidate_pool_limit,
    )
    if candidate_pool.empty:
        raise RuntimeError("No strategy 1/2/3 candidates were found in the requested window.")
    feature_frame = _build_feature_frame(history, pd.DatetimeIndex(candidate_pool["market_date"].unique()))
    if feature_frame.empty:
        raise RuntimeError("No model feature rows were built for strategy candidates.")
    scored_input = candidate_pool.merge(
        feature_frame,
        left_on=["symbol", "market_date"],
        right_on=["symbol", "trade_date"],
        how="inner",
        suffixes=("", "_feature"),
    )
    if scored_input.empty:
        raise RuntimeError("No strategy candidates could be aligned with model features.")
    scored_candidates, model_kind = _score_candidates(
        scored_input,
        horizon_days=int(model_horizon_days),
        positive_return=float(positive_return),
    )
    top_selection = _top_n_by_day(scored_candidates, top_n)
    diagnostic_results = _evaluate_top_selection(top_selection, history, int(hold_days))
    portfolio_result = simulate_portfolio_from_candidates(
        top_selection,
        history,
        config=PortfolioBacktestConfig(
            max_positions=int(top_n),
            holding_days=int(hold_days),
        ),
    )
    results = _merge_portfolio_and_forward_diagnostics(portfolio_result.trades, diagnostic_results, int(hold_days))
    daily = _daily_from_portfolio_result(portfolio_result)
    summary = _summarize_with_portfolio_engine(
        date_from=resolved_from,
        date_to=resolved_to,
        history=history,
        candidate_pool=candidate_pool,
        scored_candidates=scored_candidates,
        top_selection=top_selection,
        portfolio_result=portfolio_result,
        diagnostic_results=diagnostic_results,
        daily=daily,
        top_n=int(top_n),
        hold_days=int(hold_days),
        model_horizon_days=int(model_horizon_days),
        positive_return=float(positive_return),
        model_kind=model_kind,
    )

    results_path = output_path / "top10_trades.csv"
    daily_path = output_path / "daily_returns.csv"
    equity_path = output_path / "equity_curve.csv"
    summary_path = output_path / "summary.json"
    candidate_path = output_path / "scored_candidates.csv"
    export_cols = [
        "market_date",
        "daily_rank",
        "symbol",
        "name",
        "candidate_strategy",
        "candidate_priority",
        "model_probability",
        "model_score",
        "context_composite_score",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "hold_3d_return",
        "gross_return",
        "net_return",
        "forward_return",
        "max_high_return",
        "max_drawdown",
    ]
    results[[column for column in export_cols if column in results.columns]].to_csv(results_path, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    daily[[column for column in ["market_date", "equity", "running_max", "drawdown"] if column in daily.columns]].to_csv(
        equity_path,
        index=False,
        encoding="utf-8-sig",
    )
    scored_candidates[
        [
            column
            for column in [
                "market_date",
                "symbol",
                "name",
                "candidate_strategy",
                "candidate_priority",
                "model_probability",
                "model_score",
                "context_composite_score",
            ]
            if column in scored_candidates.columns
        ]
    ].to_csv(candidate_path, index=False, encoding="utf-8-sig")
    summary.update(
        {
            "summary_path": str(summary_path),
            "results_path": str(results_path),
            "daily_returns_path": str(daily_path),
            "equity_curve_path": str(equity_path),
            "scored_candidates_path": str(candidate_path),
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "summary": summary,
        "results": results,
        "daily": daily,
        "summary_path": str(summary_path),
        "results_path": str(results_path),
        "daily_returns_path": str(daily_path),
        "equity_curve_path": str(equity_path),
        "scored_candidates_path": str(candidate_path),
    }


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest strategy 1/2/3 filtered A-share main-board top-N model rankings.")
    parser.add_argument("--date-from", default="")
    parser.add_argument("--date-to", default="")
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument("--hold-days", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--candidate-pool-limit", type=int, default=5000)
    parser.add_argument("--model-horizon-days", type=int, default=5)
    parser.add_argument("--positive-return", type=float, default=0.03)
    parser.add_argument("--output-dir", default=str(DEFAULT_TOP10_OUTPUT_DIR))
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    payload = run_strategy_model_top10_backtest(
        date_from=args.date_from or None,
        date_to=args.date_to or None,
        months=int(args.months),
        hold_days=int(args.hold_days),
        top_n=int(args.top_n),
        candidate_pool_limit=int(args.candidate_pool_limit),
        model_horizon_days=int(args.model_horizon_days),
        positive_return=float(args.positive_return),
        output_dir=args.output_dir,
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
