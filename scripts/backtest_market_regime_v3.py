from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from a_share_predictor.portfolio_backtest_adapter import portfolio_summary_fields, simulate_selected_portfolio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / ".cache" / "ten_year_model_strategy_validation"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".cache" / "ten_year_market_regime_v3"


def _format_date(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def _main_board_mask(symbols: pd.Series) -> pd.Series:
    text = symbols.astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    return text.str.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))


def _safe_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _load_range_cache_history(date_from: str, date_to: str, *, lookback_days: int = 90) -> pd.DataFrame:
    start = pd.to_datetime(date_from) - pd.Timedelta(days=int(lookback_days))
    end = pd.to_datetime(date_to)
    frames: list[pd.DataFrame] = []
    cache_files = sorted((PROJECT_ROOT / ".cache").glob("tushare_daily_range_fast_*.pkl"))
    if not cache_files:
        raise RuntimeError("No Tushare daily range cache files found.")

    columns = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "pct_chg", "amount", "vol"]
    for index, path in enumerate(cache_files, start=1):
        if index == 1 or index % 25 == 0:
            print(f"[market_cache] {index}/{len(cache_files)} {path.name}", flush=True)
        try:
            frame = pd.read_pickle(path)
        except Exception:
            continue
        if not isinstance(frame, pd.DataFrame) or frame.empty or "trade_date" not in frame.columns:
            continue
        keep = [column for column in columns if column in frame.columns]
        frame = frame[keep].copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        frame = frame.loc[frame["trade_date"].between(start, end, inclusive="both")].copy()
        if frame.empty:
            continue
        if "ts_code" not in frame.columns:
            continue
        frame["symbol"] = frame["ts_code"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
        frame = frame.loc[_main_board_mask(frame["symbol"])].copy()
        if frame.empty:
            continue
        frames.append(frame.drop(columns=["ts_code"], errors="ignore"))

    if not frames:
        raise RuntimeError("No usable market history found in range caches.")
    history = pd.concat(frames, ignore_index=True, sort=False)
    history = _safe_numeric(history, ["open", "high", "low", "close", "pre_close", "pct_chg", "amount", "vol"])
    history = history.dropna(subset=["symbol", "trade_date", "close"]).copy()
    history = history.drop_duplicates(["symbol", "trade_date"], keep="last")
    history["ret"] = history["pct_chg"] / 100.0
    fallback = history["close"] / history["pre_close"].replace(0.0, np.nan) - 1.0
    history["ret"] = history["ret"].where(history["ret"].notna(), fallback)
    history["amount"] = history["amount"].fillna(0.0)
    history = history.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    return history


def build_market_regime(
    *,
    source_dir: Path,
    output_dir: Path,
    date_from: str,
    date_to: str,
    force: bool = False,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    regime_path = output_dir / "market_regime_daily.csv"
    if regime_path.exists() and not force:
        return pd.read_csv(regime_path, encoding="utf-8-sig", parse_dates=["market_date"])

    history = _load_range_cache_history(date_from, date_to)
    print(f"[market] history rows={len(history)} dates={history['trade_date'].nunique()} symbols={history['symbol'].nunique()}", flush=True)
    history["ma20"] = history.groupby("symbol", sort=False)["close"].transform(lambda s: s.rolling(20, min_periods=15).mean())
    history["above_ma20"] = history["close"] > history["ma20"]
    history["up_amount"] = history["amount"].where(history["ret"] > 0, 0.0)
    history["down_amount"] = history["amount"].where(history["ret"] < 0, 0.0)
    history["strong_up_amount"] = history["amount"].where(history["ret"] >= 0.03, 0.0)
    history["strong_down_amount"] = history["amount"].where(history["ret"] <= -0.03, 0.0)
    history["limit_up"] = history["ret"] >= 0.095
    history["limit_down"] = history["ret"] <= -0.095

    daily = (
        history.groupby("trade_date", as_index=False)
        .agg(
            stock_count=("symbol", "nunique"),
            market_ret=("ret", "mean"),
            amount_total=("amount", "sum"),
            up_ratio=("ret", lambda s: float((s > 0).mean())),
            above_ma20_ratio=("above_ma20", "mean"),
            up_amount=("up_amount", "sum"),
            down_amount=("down_amount", "sum"),
            strong_up_amount=("strong_up_amount", "sum"),
            strong_down_amount=("strong_down_amount", "sum"),
            limit_up_count=("limit_up", "sum"),
            limit_down_count=("limit_down", "sum"),
        )
        .rename(columns={"trade_date": "market_date"})
        .sort_values("market_date")
    )
    daily["market_index"] = (1.0 + daily["market_ret"].fillna(0.0)).cumprod()
    daily["market_ma20"] = daily["market_index"].rolling(20, min_periods=15).mean()
    daily["market_ma60"] = daily["market_index"].rolling(60, min_periods=45).mean()
    daily["market_ma20_slope5"] = daily["market_ma20"] / daily["market_ma20"].shift(5) - 1.0
    daily["market_ret_5d"] = daily["market_index"] / daily["market_index"].shift(5) - 1.0
    daily["drawdown_20d"] = daily["market_index"] / daily["market_index"].rolling(20, min_periods=10).max() - 1.0
    daily["amount_ma5"] = daily["amount_total"].rolling(5, min_periods=3).mean()
    daily["amount_ma20"] = daily["amount_total"].rolling(20, min_periods=10).mean()
    daily["amount_ma60"] = daily["amount_total"].rolling(60, min_periods=30).mean()
    daily["amount_ma5_ma20"] = daily["amount_ma5"] / daily["amount_ma20"].replace(0.0, np.nan)
    daily["amount_ma20_ma60"] = daily["amount_ma20"] / daily["amount_ma60"].replace(0.0, np.nan)
    daily["up_amount_ratio"] = daily["up_amount"] / daily["amount_total"].replace(0.0, np.nan)
    strong_ratio = daily["strong_up_amount"] / daily["strong_down_amount"].replace(0.0, np.nan)
    strong_ratio = strong_ratio.replace([np.inf, -np.inf], np.nan)
    daily["strong_amount_ratio"] = np.where(
        strong_ratio.notna(),
        strong_ratio,
        np.where(daily["strong_down_amount"].eq(0) & daily["strong_up_amount"].gt(0), 9.99, 0.0),
    )

    score_counts = _daily_model_score_counts(source_dir / "model_scores.csv")
    candidate_counts = _daily_strategy_candidate_counts(source_dir / "strategy_candidates.csv")
    daily = daily.merge(score_counts, on="market_date", how="left")
    daily = daily.merge(candidate_counts, on="market_date", how="left")
    for column in ["score_ge68_count", "score_ge70_count", "strategy_candidate_count"]:
        daily[column] = daily[column].fillna(0).astype(int)

    trend_checks = [
        daily["market_index"] > daily["market_ma20"],
        daily["market_ma20"] > daily["market_ma60"],
        daily["market_ma20_slope5"] > 0,
        daily["market_ret_5d"] >= -0.04,
        daily["drawdown_20d"] >= -0.08,
    ]
    flow_checks = [
        daily["amount_ma5_ma20"] >= 1.05,
        daily["up_amount_ratio"] >= 0.52,
        daily["strong_amount_ratio"] >= 1.20,
        daily["up_ratio"] >= 0.45,
        daily["above_ma20_ratio"] >= 0.45,
    ]
    daily["trend_score"] = np.column_stack([check.fillna(False).to_numpy(bool) for check in trend_checks]).sum(axis=1)
    daily["flow_score"] = np.column_stack([check.fillna(False).to_numpy(bool) for check in flow_checks]).sum(axis=1)
    daily["trend_green"] = daily["trend_score"] >= 4
    daily["flow_green"] = daily["flow_score"] >= 3
    daily["internal_green"] = (
        daily["strategy_candidate_count"].between(40, 100, inclusive="both") & daily["score_ge68_count"].ge(10)
    )
    daily["market_green"] = daily["trend_green"] & daily["flow_green"]
    daily["v3_full_green"] = daily["market_green"] & daily["internal_green"]
    daily["v3_yellow"] = (
        daily["trend_score"].ge(3)
        & daily["flow_score"].ge(2)
        & daily["score_ge70_count"].ge(1)
        & ~daily["v3_full_green"]
    )
    daily["market_state"] = np.select(
        [daily["v3_full_green"], daily["market_green"], daily["v3_yellow"]],
        ["full_green", "market_green", "yellow"],
        default="red",
    )

    start = pd.to_datetime(date_from)
    end = pd.to_datetime(date_to)
    daily = daily.loc[daily["market_date"].between(start, end, inclusive="both")].copy()
    daily.to_csv(regime_path, index=False, encoding="utf-8-sig")
    return daily


def _daily_model_score_counts(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, encoding="utf-8-sig", usecols=["market_date", "model_score"], chunksize=500_000):
        chunk["market_date"] = pd.to_datetime(chunk["market_date"], errors="coerce")
        chunk["model_score"] = pd.to_numeric(chunk["model_score"], errors="coerce")
        chunk = chunk.dropna(subset=["market_date", "model_score"])
        grouped = chunk.groupby("market_date").agg(
            score_ge68_count=("model_score", lambda s: int((s >= 68).sum())),
            score_ge70_count=("model_score", lambda s: int((s >= 70).sum())),
        )
        rows.append(grouped.reset_index())
    counts = pd.concat(rows, ignore_index=True)
    return counts.groupby("market_date", as_index=False)[["score_ge68_count", "score_ge70_count"]].sum()


def _load_model_calendar(path: Path) -> pd.DataFrame:
    dates: set[pd.Timestamp] = set()
    for chunk in pd.read_csv(path, encoding="utf-8-sig", usecols=["market_date"], chunksize=500_000):
        parsed = pd.to_datetime(chunk["market_date"], errors="coerce").dropna().dt.normalize()
        dates.update(pd.Timestamp(value) for value in parsed.unique())
    return pd.DataFrame({"market_date": sorted(dates)})


def _daily_strategy_candidate_counts(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", usecols=["market_date"])
    frame["market_date"] = pd.to_datetime(frame["market_date"], errors="coerce")
    return frame.dropna().groupby("market_date").size().rename("strategy_candidate_count").reset_index()


def enrich_strategy_candidates(source_dir: Path, output_dir: Path, *, force: bool = False) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "strategy_model_candidates.csv"
    if path.exists() and not force:
        return pd.read_csv(path, encoding="utf-8-sig", parse_dates=["market_date"])

    strategy = pd.read_csv(source_dir / "strategy_candidates.csv", encoding="utf-8-sig")
    strategy["market_date"] = pd.to_datetime(strategy["market_date"], errors="coerce")
    strategy["symbol"] = strategy["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    strategy = strategy.dropna(subset=["market_date", "symbol"]).copy()
    strategy["join_key"] = strategy["market_date"].dt.strftime("%Y-%m-%d") + "|" + strategy["symbol"]
    keys = set(strategy["join_key"].tolist())

    score_parts: list[pd.DataFrame] = []
    usecols = ["market_date", "symbol", "model_probability", "model_score"]
    for index, chunk in enumerate(pd.read_csv(source_dir / "model_scores.csv", encoding="utf-8-sig", usecols=usecols, chunksize=500_000), start=1):
        if index == 1 or index % 5 == 0:
            print(f"[join_scores] chunk={index}", flush=True)
        chunk["market_date"] = pd.to_datetime(chunk["market_date"], errors="coerce")
        chunk["symbol"] = chunk["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
        chunk["join_key"] = chunk["market_date"].dt.strftime("%Y-%m-%d") + "|" + chunk["symbol"]
        part = chunk.loc[chunk["join_key"].isin(keys), ["join_key", "model_probability", "model_score"]].copy()
        if not part.empty:
            score_parts.append(part)
    scores = pd.concat(score_parts, ignore_index=True).drop_duplicates("join_key", keep="last")
    merged = strategy.merge(scores, on="join_key", how="inner").drop(columns=["join_key"])
    merged["model_score"] = pd.to_numeric(merged["model_score"], errors="coerce")
    merged["model_probability"] = pd.to_numeric(merged["model_probability"], errors="coerce")
    for column in ["hold_3d_return", "max_high_return", "max_drawdown", "candidate_priority"]:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged = merged.dropna(subset=["market_date", "symbol", "model_score", "hold_3d_return"]).copy()
    merged.to_csv(path, index=False, encoding="utf-8-sig")
    return merged


def _select_rule(candidates: pd.DataFrame, regime: pd.DataFrame, *, rule: str) -> pd.DataFrame:
    frame = candidates.merge(regime, on="market_date", how="left")
    frame = frame.copy()
    if rule == "v2_score68_top3":
        mask = frame["model_score"].ge(68)
        top_n = 3
    elif rule == "v3_trend_top3":
        mask = frame["trend_green"].fillna(False) & frame["model_score"].ge(68)
        top_n = 3
    elif rule == "v3_trend_flow_top3":
        mask = frame["market_green"].fillna(False) & frame["model_score"].ge(68)
        top_n = 3
    elif rule == "v3_full_green_top3":
        mask = frame["v3_full_green"].fillna(False) & frame["model_score"].ge(68)
        top_n = 3
    elif rule == "v3_full_green_top3_cand40_100":
        mask = (
            frame["trend_green"].fillna(False)
            & frame["flow_green"].fillna(False)
            & frame["strategy_candidate_count"].between(40, 100, inclusive="both")
            & frame["score_ge68_count"].ge(10)
            & frame["model_score"].ge(68)
        )
        top_n = 3
    elif rule == "v3_dynamic_yellow_top1_green_top3":
        green = frame["v3_full_green"].fillna(False) & frame["model_score"].ge(68)
        yellow = frame["v3_yellow"].fillna(False) & frame["model_score"].ge(70)
        frame = pd.concat(
            [
                _take_top(frame.loc[green].copy(), 3, require_full=True),
                _take_top(frame.loc[yellow].copy(), 1),
            ],
            ignore_index=True,
        )
        frame["rule"] = rule
        return frame
    else:
        raise ValueError(f"Unknown rule: {rule}")
    selected = _take_top(frame.loc[mask].copy(), top_n, require_full=True)
    selected["rule"] = rule
    return selected


def _take_top(frame: pd.DataFrame, top_n: int, *, require_full: bool = False) -> pd.DataFrame:
    if frame.empty:
        return frame
    sort_columns = ["market_date", "model_score"]
    ascending = [True, False]
    if "candidate_priority" in frame.columns:
        sort_columns.append("candidate_priority")
        ascending.append(False)
    frame = frame.sort_values(sort_columns, ascending=ascending)
    frame["daily_rule_rank"] = frame.groupby("market_date").cumcount() + 1
    selected = frame.loc[frame["daily_rule_rank"].le(int(top_n))].copy()
    if require_full:
        counts = selected.groupby("market_date").size()
        full_dates = counts.loc[counts.ge(int(top_n))].index
        selected = selected.loc[selected["market_date"].isin(full_dates)].copy()
    return selected


def _summarize_rule(
    selected: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    rule: str,
    history: pd.DataFrame,
    top_n: int,
    holding_days: int = 3,
    positive_return: float = 0.03,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    selected = selected.copy()
    selected["market_date"] = pd.to_datetime(selected["market_date"], errors="coerce")
    selected["hold_3d_return"] = pd.to_numeric(selected["hold_3d_return"], errors="coerce")
    selected = selected.dropna(subset=["market_date", "hold_3d_return"])
    daily = (
        selected.groupby("market_date", as_index=False)
        .agg(selected=("symbol", "count"), avg_return=("hold_3d_return", "mean"))
        .sort_values("market_date")
    )
    curve = calendar.merge(daily, on="market_date", how="left")
    curve["selected"] = curve["selected"].fillna(0).astype(int)
    curve["avg_return"] = curve["avg_return"].fillna(0.0)
    curve["equity"] = (1.0 + curve["avg_return"]).cumprod()
    curve["running_max"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["running_max"].replace(0.0, np.nan) - 1.0
    ending = float(curve["equity"].iloc[-1]) if not curve.empty else 1.0
    annualized = ending ** (252.0 / len(curve)) - 1.0 if len(curve) and ending > 0 else -1.0
    max_drawdown = float(curve["drawdown"].min()) if not curve.empty else 0.0
    trade_returns = selected["hold_3d_return"].dropna()
    active_returns = daily["avg_return"].dropna()
    portfolio_result = simulate_selected_portfolio(
        selected,
        history,
        max_positions=max(int(top_n), 1),
        holding_days=max(int(holding_days), 1),
    )
    summary = {
        "rule": rule,
        "active_days": int((curve["selected"] > 0).sum()),
        "coverage_pct": round(float((curve["selected"] > 0).mean()) * 100.0, 2) if not curve.empty else 0.0,
        "selected_rows": int(len(selected)),
        "avg_trade_return": round(float(trade_returns.mean()), 6) if not trade_returns.empty else 0.0,
        "trade_win_rate": round(float((trade_returns > 0).mean()), 4) if not trade_returns.empty else 0.0,
        "target_hit_rate": round(float((trade_returns >= float(positive_return)).mean()), 4) if not trade_returns.empty else 0.0,
        "active_daily_return": round(float(active_returns.mean()), 6) if not active_returns.empty else 0.0,
        "active_daily_win_rate": round(float((active_returns > 0).mean()), 4) if not active_returns.empty else 0.0,
        "annualized_return": round(float(annualized), 6),
        "max_drawdown": round(float(max_drawdown), 6),
        "ending_equity": round(float(ending), 6),
        "return_drawdown_ratio": round(float(annualized / abs(max_drawdown)), 6) if max_drawdown else None,
    }
    summary.update(portfolio_summary_fields(portfolio_result))
    nav = portfolio_result.daily_nav.copy()
    if not nav.empty:
        nav["rule"] = rule
    trades = portfolio_result.trades.copy()
    if not trades.empty:
        trades["rule"] = rule
    return summary, nav, trades


def _apply_pause(selected: pd.DataFrame, calendar: pd.DataFrame, *, pause_drawdown: float = -0.06, pause_days: int = 10) -> pd.DataFrame:
    daily_map = (
        selected.groupby("market_date")
        .agg(avg_return=("hold_3d_return", "mean"))
        .sort_index()["avg_return"]
        .to_dict()
    )
    allowed_dates: set[pd.Timestamp] = set()
    equity_values: list[float] = []
    equity = 1.0
    pause_left = 0
    for value in calendar["market_date"]:
        date = pd.Timestamp(value)
        if pause_left > 0:
            daily_return = 0.0
            pause_left -= 1
        else:
            allowed_dates.add(date)
            daily_return = float(daily_map.get(date, 0.0))
        equity *= 1.0 + daily_return
        equity_values.append(equity)
        recent = pd.Series(equity_values[-20:])
        if pause_left == 0 and len(recent) >= 10:
            recent_drawdown = float(recent.iloc[-1] / recent.max() - 1.0)
            if recent_drawdown <= float(pause_drawdown):
                pause_left = int(pause_days)
    paused = selected.loc[selected["market_date"].isin(allowed_dates)].copy()
    return paused


def evaluate_existing_selection_overlay(
    source_dir: Path,
    regime: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    history: pd.DataFrame,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    path = source_dir / "combined_selected_trades.csv"
    selected = pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=["market_date", "symbol", "name", "rule", "model_score", "hold_3d_return"],
        parse_dates=["market_date"],
    )
    selected["hold_3d_return"] = pd.to_numeric(selected["hold_3d_return"], errors="coerce")
    selected["model_score"] = pd.to_numeric(selected["model_score"], errors="coerce")
    base = selected.loc[selected["rule"].eq("combined_top3_score_ge_68_full")].copy()
    env_cols = [
        "market_date",
        "trend_green",
        "market_green",
        "v3_full_green",
        "market_state",
        "strategy_candidate_count",
        "score_ge68_count",
    ]
    base = base.merge(regime[env_cols], on="market_date", how="left")
    for column in ["trend_green", "market_green", "v3_full_green"]:
        base[column] = base[column].fillna(False).astype(bool)
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    masks = {
        "existing_v2_top3_score68_full": pd.Series(True, index=base.index),
        "existing_v3_trend_top3": base["trend_green"],
        "existing_v3_trend_flow_top3": base["market_green"],
        "existing_v3_full_green_top3": base["v3_full_green"],
        "existing_v3_trend_flow_cand40_100_top3": base["market_green"]
        & base["strategy_candidate_count"].between(40, 100, inclusive="both"),
    }
    for rule, mask in masks.items():
        frame = base.loc[mask].copy()
        frame["rule"] = rule
        frames.append(frame)
        summary, _, _ = _summarize_rule(
            frame,
            calendar,
            rule=rule,
            history=history,
            top_n=3,
        )
        summaries.append(summary)
    all_selected = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    summaries = (
        pd.DataFrame(summaries)
        .sort_values(["return_drawdown_ratio", "annualized_return"], ascending=[False, False])
        .replace({np.nan: None})
        .to_dict("records")
    )
    return summaries, all_selected


def run_v3_backtest(
    *,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    date_from: str = "2016-05-27",
    date_to: str = "2026-05-26",
    force: bool = False,
) -> dict[str, object]:
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    regime = build_market_regime(source_dir=source_path, output_dir=output_path, date_from=date_from, date_to=date_to, force=force)
    candidates = enrich_strategy_candidates(source_path, output_path, force=force)
    history = _load_range_cache_history(date_from, date_to)
    start = pd.to_datetime(date_from)
    end = pd.to_datetime(date_to)
    candidates = candidates.loc[candidates["market_date"].between(start, end, inclusive="both")].copy()
    calendar = _load_model_calendar(source_path / "model_scores.csv")
    calendar = calendar.loc[calendar["market_date"].between(start, end, inclusive="both")].copy()

    rules = [
        "v2_score68_top3",
        "v3_trend_top3",
        "v3_trend_flow_top3",
        "v3_full_green_top3",
        "v3_full_green_top3_cand40_100",
        "v3_dynamic_yellow_top1_green_top3",
    ]
    selected_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    portfolio_nav_frames: list[pd.DataFrame] = []
    portfolio_trade_frames: list[pd.DataFrame] = []
    for rule in rules:
        selected = _select_rule(candidates, regime, rule=rule)
        selected_frames.append(selected)
        summary_row, nav, trades = _summarize_rule(
            selected,
            calendar,
            rule=rule,
            history=history,
            top_n=3,
        )
        summaries.append(summary_row)
        if not nav.empty:
            portfolio_nav_frames.append(nav)
        if not trades.empty:
            portfolio_trade_frames.append(trades)
        if rule in {"v3_full_green_top3", "v3_dynamic_yellow_top1_green_top3"}:
            paused = _apply_pause(selected, calendar)
            paused["rule"] = f"{rule}_pause6_10d"
            selected_frames.append(paused)
            paused_summary, nav, trades = _summarize_rule(
                paused,
                calendar,
                rule=f"{rule}_pause6_10d",
                history=history,
                top_n=3,
            )
            summaries.append(paused_summary)
            if not nav.empty:
                portfolio_nav_frames.append(nav)
            if not trades.empty:
                portfolio_trade_frames.append(trades)

    existing_overlay_rules, existing_overlay_selected = evaluate_existing_selection_overlay(
        source_path,
        regime,
        calendar,
        history=history,
    )
    selected_all = pd.concat(selected_frames, ignore_index=True, sort=False) if selected_frames else pd.DataFrame()
    rules_frame = pd.DataFrame(summaries).sort_values(["return_drawdown_ratio", "annualized_return"], ascending=[False, False])
    portfolio_nav_all = pd.concat(portfolio_nav_frames, ignore_index=True, sort=False) if portfolio_nav_frames else pd.DataFrame()
    portfolio_trades_all = pd.concat(portfolio_trade_frames, ignore_index=True, sort=False) if portfolio_trade_frames else pd.DataFrame()
    regime_summary = (
        regime.groupby("market_state")
        .size()
        .rename("days")
        .reset_index()
        .sort_values("days", ascending=False)
        .to_dict("records")
    )
    summary = {
        "date_from": date_from,
        "date_to": date_to,
        "calendar_days": int(len(calendar)),
        "market_state_days": regime_summary,
        "rules": rules_frame.replace({np.nan: None}).to_dict("records"),
        "best_rules": rules_frame.head(10).replace({np.nan: None}).to_dict("records"),
        "existing_overlay_rules": existing_overlay_rules,
        "summary_path": str(output_path / "v3_summary.json"),
        "market_regime_path": str(output_path / "market_regime_daily.csv"),
        "selected_trades_path": str(output_path / "v3_selected_trades.csv"),
        "rules_path": str(output_path / "v3_rules.csv"),
        "portfolio_nav_path": str(output_path / "v3_portfolio_daily_nav.csv"),
        "portfolio_trades_path": str(output_path / "v3_portfolio_trades.csv"),
        "existing_overlay_rules_path": str(output_path / "v3_existing_overlay_rules.csv"),
        "existing_overlay_selected_path": str(output_path / "v3_existing_overlay_selected_trades.csv"),
        "notes": [
            "annualized_return/max_drawdown/ending_equity remain legacy average-selected-return metrics for compatibility.",
            "portfolio_* fields use the unified portfolio backtest engine and should be preferred for trading-system evaluation.",
        ],
    }
    selected_all.to_csv(output_path / "v3_selected_trades.csv", index=False, encoding="utf-8-sig")
    rules_frame.to_csv(output_path / "v3_rules.csv", index=False, encoding="utf-8-sig")
    portfolio_nav_all.to_csv(output_path / "v3_portfolio_daily_nav.csv", index=False, encoding="utf-8-sig")
    portfolio_trades_all.to_csv(output_path / "v3_portfolio_trades.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(existing_overlay_rules).to_csv(output_path / "v3_existing_overlay_rules.csv", index=False, encoding="utf-8-sig")
    existing_overlay_selected.to_csv(output_path / "v3_existing_overlay_selected_trades.csv", index=False, encoding="utf-8-sig")
    (output_path / "v3_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest V3 market-regime and fund-flow filters.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--date-from", default="2016-05-27")
    parser.add_argument("--date-to", default="2026-05-26")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = run_v3_backtest(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        date_from=args.date_from,
        date_to=args.date_to,
        force=bool(args.force),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
