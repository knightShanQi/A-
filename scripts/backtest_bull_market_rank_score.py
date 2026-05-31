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
DEFAULT_V3_DIR = PROJECT_ROOT / ".cache" / "ten_year_market_regime_v3"
DEFAULT_SSE_PATH = PROJECT_ROOT / ".cache" / "ten_year_sse3500_filter" / "sse_000001_daily.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".cache" / "ten_year_bull_market_rank_score"


def _main_board_mask(symbols: pd.Series) -> pd.Series:
    text = symbols.astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    return text.str.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))


def _load_range_cache_history(date_from: str, date_to: str, *, lookback_days: int = 90) -> pd.DataFrame:
    start = pd.to_datetime(date_from) - pd.Timedelta(days=int(lookback_days))
    end = pd.to_datetime(date_to)
    frames: list[pd.DataFrame] = []
    cache_files = sorted((PROJECT_ROOT / ".cache").glob("tushare_daily_range_fast_*.pkl"))
    columns = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "pct_chg", "amount", "vol"]
    for path in cache_files:
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
        if frame.empty or "ts_code" not in frame.columns:
            continue
        frame["symbol"] = frame["ts_code"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
        frame = frame.loc[_main_board_mask(frame["symbol"])].copy()
        if frame.empty:
            continue
        frames.append(frame.drop(columns=["ts_code"], errors="ignore"))
    if not frames:
        return pd.DataFrame()
    history = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates(["symbol", "trade_date"], keep="last")
    for column in ["open", "high", "low", "close", "pre_close", "pct_chg", "amount", "vol"]:
        if column in history.columns:
            history[column] = pd.to_numeric(history[column], errors="coerce")
    return history.dropna(subset=["trade_date", "symbol", "close"]).sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def _load_model_calendar(path: Path, date_from: str, date_to: str) -> pd.DataFrame:
    dates: set[pd.Timestamp] = set()
    for chunk in pd.read_csv(path, encoding="utf-8-sig", usecols=["market_date"], chunksize=500_000):
        parsed = pd.to_datetime(chunk["market_date"], errors="coerce").dropna().dt.normalize()
        dates.update(pd.Timestamp(value) for value in parsed.unique())
    start = pd.to_datetime(date_from)
    end = pd.to_datetime(date_to)
    calendar = pd.DataFrame({"market_date": sorted(dates)})
    return calendar.loc[calendar["market_date"].between(start, end, inclusive="both")].copy()


def _num(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(float(default), index=frame.index, dtype=float)


def _rebuild_page_rank_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Rebuild dashboard quick-board rank score from cached historical candidate fields."""
    index = frame.index
    change_pct = _num(frame, "change_pct")
    amount = _num(frame, "amount")
    turnover = _num(frame, "turnover")
    industry_strength = _num(frame, "industry_ret_2d_pct")
    industry_up_count = _num(frame, "industry_up_count")
    price_vs_prev = change_pct / 100.0

    quant_score = (
        50.0 + change_pct * 2.2 + turnover * 1.6 + industry_strength * 1.2 + price_vs_prev * 100.0 * 0.8
    ).clip(lower=0.0, upper=100.0)
    launch_score = (
        48.0 + change_pct * 2.6 + turnover * 1.2 + industry_strength * 1.8 + industry_up_count.clip(upper=10) * 0.9
    ).clip(lower=0.0, upper=100.0)
    launch_readiness = (
        launch_score.fillna(50.0)
        + np.where(change_pct >= 2.0, 6.0, 0.0)
        + industry_up_count.clip(upper=10) * 1.1
    ).clip(lower=0.0, upper=100.0)
    market_resonance = (
        50.0 + industry_strength * 4.2 + industry_up_count.clip(upper=10) * 1.7 + change_pct * 0.9
    ).clip(lower=0.0, upper=100.0)

    strategy = frame.get("candidate_strategy", pd.Series("", index=index)).fillna("").astype(str).str.lower()
    strategy1 = strategy.str.contains("strategy1", regex=False) | strategy.str.contains("1", regex=False)
    strategy2 = strategy.str.contains("strategy2", regex=False) | strategy.str.contains("2", regex=False)
    strategy3 = strategy.str.contains("strategy3", regex=False) | strategy.str.contains("3", regex=False)
    consecutive_up_days = pd.Series(
        np.select([strategy1 | strategy2, strategy3, change_pct > 0.0], [3, 2, 1], default=0),
        index=index,
    ).astype(int)
    trend_strength = industry_strength * 1.3 + change_pct * 0.7 + price_vs_prev * 100.0 * 0.5

    composite = (
        (launch_readiness.fillna(50.0) - 50.0) / 11.5
        + (market_resonance.fillna(50.0) - 50.0) / 14.0
        + (quant_score.fillna(50.0) - 50.0) / 14.5
        + change_pct.fillna(0.0).clip(lower=-8.0, upper=12.0) * 0.20
        + turnover.fillna(0.0).clip(lower=0.0, upper=15.0) * 0.10
        + (amount.fillna(0.0) / 1e8).clip(lower=0.0, upper=25.0) * 0.05
        + trend_strength.fillna(0.0).clip(lower=-20.0, upper=35.0) * 0.05
        + consecutive_up_days.fillna(0).clip(lower=0, upper=8) * 0.20
    )
    # The live page ranks relative probability within its board input. Full-market historical
    # page scores were not cached, so this reconstruction ranks within each day's strategy pool.
    signal_rank = composite.groupby(frame["market_date"]).rank(pct=True, method="average").fillna(0.5)
    logistic = 1.0 / (1.0 + np.exp(-((composite - 4.0) / 1.6)))
    absolute_probability = 16.0 + logistic * 74.0
    relative_probability = 24.0 + signal_rank * 58.0
    probability_up = (absolute_probability * 0.45 + relative_probability * 0.55).clip(lower=8.0, upper=92.0)

    attention_score = (
        40.0
        + probability_up.fillna(50.0) * 0.36
        + change_pct * 1.4
        + turnover * 0.7
        + industry_strength * 1.6
    ).clip(lower=0.0, upper=100.0)
    enhanced_attention_score = (
        attention_score.fillna(50.0) + industry_strength * 1.2 + industry_up_count.clip(upper=10) * 0.7
    ).clip(lower=0.0, upper=100.0)
    predicted_upside_pct = (
        (probability_up.fillna(50.0) - 45.0).clip(lower=-10.0, upper=40.0) * 0.12
        + change_pct.clip(lower=-5.0, upper=10.0) * 0.12
        + trend_strength.clip(lower=-15.0, upper=30.0) * 0.05
        + turnover.clip(lower=0.0, upper=12.0) * 0.08
    ).clip(lower=0.6, upper=18.0)

    return pd.DataFrame(
        {
            "probability_up_rebuilt": probability_up,
            "attention_score_rebuilt": attention_score,
            "enhanced_attention_score_rebuilt": enhanced_attention_score,
            "rank_score_rebuilt": enhanced_attention_score,
            "predicted_upside_pct_rebuilt": predicted_upside_pct,
            "quant_score_rebuilt": quant_score,
            "launch_score_rebuilt": launch_score,
            "market_resonance_score_rebuilt": market_resonance,
        },
        index=index,
    )


def _load_combined_candidates(source_dir: Path, date_from: str, date_to: str) -> pd.DataFrame:
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
        chunk = pd.read_csv(path, encoding="utf-8-sig", usecols=lambda column: column in columns, parse_dates=["market_date"])
        chunk = chunk.loc[chunk["market_date"].between(start, end, inclusive="both")].copy()
        if not chunk.empty:
            frames.append(chunk)
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame["symbol"] = frame["symbol"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(6)
    for column in [
        "candidate_priority",
        "model_probability",
        "model_score",
        "hold_3d_return",
        "max_high_return",
        "max_drawdown",
        "change_pct",
        "amount",
        "turnover",
        "industry_ret_2d_pct",
        "industry_up_count",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return pd.concat([frame, _rebuild_page_rank_scores(frame)], axis=1)


def _build_bull_bear_regime(sse: pd.DataFrame, market_regime: pd.DataFrame) -> pd.DataFrame:
    sse = sse.sort_values("market_date").copy()
    for window in (20, 60, 120, 250):
        sse[f"sse_ma{window}"] = sse["sse_close"].rolling(window, min_periods=max(5, int(window * 0.6))).mean()
    sse["sse_ma60_slope20"] = sse["sse_ma60"] / sse["sse_ma60"].shift(20) - 1.0
    sse["sse_ret_60d"] = sse["sse_close"] / sse["sse_close"].shift(60) - 1.0
    sse["sse_drawdown_120d"] = sse["sse_close"] / sse["sse_close"].rolling(120, min_periods=60).max() - 1.0

    keep = [
        "market_date",
        "above_ma20_ratio",
        "amount_ma20_ma60",
        "up_amount_ratio",
        "trend_score",
        "flow_score",
        "market_state",
        "v3_full_green",
    ]
    regime = sse.merge(market_regime[[column for column in keep if column in market_regime.columns]], on="market_date", how="left")
    checks = {
        "close_gt_ma250": regime["sse_close"] > regime["sse_ma250"],
        "ma60_gt_ma250": regime["sse_ma60"] > regime["sse_ma250"],
        "ma20_gt_ma60": regime["sse_ma20"] > regime["sse_ma60"],
        "ma60_slope_pos": regime["sse_ma60_slope20"] > 0.0,
        "ret60_pos": regime["sse_ret_60d"] > 0.0,
        "dd120_ok": regime["sse_drawdown_120d"] >= -0.15,
        "above_ma20_ge50": regime["above_ma20_ratio"] >= 0.50,
        "amount_20_60_ge1": regime["amount_ma20_ma60"] >= 1.0,
        "up_amount_ge50": regime["up_amount_ratio"] >= 0.50,
        "trend_flow_green": (regime["trend_score"] >= 4) & (regime["flow_score"] >= 3),
    }
    for name, values in checks.items():
        regime[name] = values.fillna(False).astype(bool)
    regime["bull_score"] = regime[list(checks)].sum(axis=1).astype(int)
    regime["bull_bear_state"] = np.select(
        [regime["bull_score"] >= 7, regime["bull_score"] <= 4],
        ["bull", "bear"],
        default="transition",
    )
    regime["is_bull_strict"] = regime["bull_score"] >= 7
    regime["is_bull_loose"] = regime["bull_score"] >= 6
    return regime


def _take_top(frame: pd.DataFrame, top_n: int, sort_column: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    sort_columns = ["market_date", sort_column]
    ascending = [True, False]
    if sort_column != "model_score" and "model_score" in frame.columns:
        sort_columns.append("model_score")
        ascending.append(False)
    if "candidate_priority" in frame.columns:
        sort_columns.append("candidate_priority")
        ascending.append(False)
    selected = frame.sort_values(sort_columns, ascending=ascending).copy()
    selected["daily_rule_rank"] = selected.groupby("market_date").cumcount() + 1
    selected = selected.loc[selected["daily_rule_rank"].le(int(top_n))].copy()
    counts = selected.groupby("market_date").size()
    full_dates = counts.loc[counts.ge(int(top_n))].index
    return selected.loc[selected["market_date"].isin(full_dates)].copy()


def _summarize(
    selected: pd.DataFrame,
    calendar: pd.DataFrame,
    rule: str,
    *,
    history: pd.DataFrame,
    top_n: int,
    holding_days: int = 3,
    positive_return: float = 0.03,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = selected.copy()
    if not frame.empty:
        frame["market_date"] = pd.to_datetime(frame["market_date"], errors="coerce")
        frame["hold_3d_return"] = pd.to_numeric(frame["hold_3d_return"], errors="coerce")
        frame = frame.dropna(subset=["market_date", "hold_3d_return"]).copy()
    if frame.empty:
        daily = pd.DataFrame(columns=["market_date", "selected", "avg_return", "avg_rank_score", "avg_model_score"])
    else:
        daily = (
            frame.groupby("market_date", as_index=False)
            .agg(
                selected=("symbol", "count"),
                avg_return=("hold_3d_return", "mean"),
                avg_rank_score=("rank_score_rebuilt", "mean"),
                avg_model_score=("model_score", "mean"),
            )
            .sort_values("market_date")
        )
    curve = calendar[["market_date"]].merge(daily, on="market_date", how="left")
    curve["selected"] = curve["selected"].fillna(0).astype(int)
    curve["avg_return"] = curve["avg_return"].fillna(0.0)
    curve["equity"] = (1.0 + curve["avg_return"]).cumprod()
    curve["running_max"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["running_max"].replace(0.0, np.nan) - 1.0

    trade_returns = frame["hold_3d_return"].dropna() if not frame.empty else pd.Series(dtype=float)
    active_daily_returns = curve.loc[curve["selected"].gt(0), "avg_return"]
    ending = float(curve["equity"].iloc[-1]) if not curve.empty else 1.0
    annualized = ending ** (252.0 / len(curve)) - 1.0 if len(curve) and ending > 0 else 0.0
    max_drawdown = float(curve["drawdown"].min()) if not curve.empty else 0.0
    portfolio_result = simulate_selected_portfolio(
        frame,
        history,
        max_positions=max(int(top_n), 1),
        holding_days=max(int(holding_days), 1),
    )
    summary = {
        "rule": rule,
        "calendar_days": int(len(calendar)),
        "active_days": int((curve["selected"] > 0).sum()),
        "coverage_pct": round(float((curve["selected"] > 0).mean()) * 100.0, 2) if not curve.empty else 0.0,
        "selected_rows": int(len(frame)),
        "avg_trade_return": round(float(trade_returns.mean()), 6) if not trade_returns.empty else None,
        "trade_win_rate": round(float((trade_returns > 0).mean()), 4) if not trade_returns.empty else None,
        "target_hit_rate": round(float((trade_returns >= float(positive_return)).mean()), 4) if not trade_returns.empty else None,
        "active_daily_return": round(float(active_daily_returns.mean()), 6) if not active_daily_returns.empty else None,
        "active_daily_win_rate": round(float((active_daily_returns > 0).mean()), 4) if not active_daily_returns.empty else None,
        "annualized_return": round(float(annualized), 6),
        "max_drawdown": round(float(max_drawdown), 6),
        "ending_equity": round(float(ending), 6),
        "return_drawdown_ratio": round(float(annualized / abs(max_drawdown)), 6) if max_drawdown < 0 else None,
        "avg_selected_rank_score": round(float(frame["rank_score_rebuilt"].mean()), 4) if not frame.empty else None,
        "avg_selected_model_score": round(float(frame["model_score"].mean()), 4) if not frame.empty else None,
    }
    summary.update(portfolio_summary_fields(portfolio_result))
    nav = portfolio_result.daily_nav.copy()
    if not nav.empty:
        nav["rule"] = rule
    trades = portfolio_result.trades.copy()
    if not trades.empty:
        trades["rule"] = rule
    return summary, curve, nav, trades


def run_backtest(
    *,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    v3_dir: str | Path = DEFAULT_V3_DIR,
    sse_path: str | Path = DEFAULT_SSE_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    date_from: str = "2016-05-27",
    date_to: str = "2026-05-26",
    rank_threshold: float = 80.0,
    top_n: int = 3,
) -> dict[str, object]:
    source_path = Path(source_dir)
    v3_path = Path(v3_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    calendar = _load_model_calendar(source_path / "model_scores.csv", date_from, date_to)
    history = _load_range_cache_history(date_from, date_to)
    sse = pd.read_csv(sse_path, encoding="utf-8-sig", parse_dates=["market_date"])
    market_regime = pd.read_csv(v3_path / "market_regime_daily.csv", encoding="utf-8-sig", parse_dates=["market_date"])
    bull_regime = _build_bull_bear_regime(sse, market_regime)
    bull_regime = bull_regime.loc[
        bull_regime["market_date"].between(pd.to_datetime(date_from), pd.to_datetime(date_to), inclusive="both")
    ].copy()

    candidates = _load_combined_candidates(source_path, date_from, date_to)
    candidates = candidates.merge(
        bull_regime[["market_date", "bull_score", "bull_bear_state", "is_bull_strict", "is_bull_loose"]],
        on="market_date",
        how="left",
    )
    candidates = candidates.merge(
        market_regime[["market_date", "v3_full_green"]],
        on="market_date",
        how="left",
    )
    candidates["v3_full_green"] = candidates["v3_full_green"].fillna(False).astype(bool)
    candidates["is_bull_strict"] = candidates["is_bull_strict"].fillna(False).astype(bool)
    candidates["is_bull_loose"] = candidates["is_bull_loose"].fillna(False).astype(bool)

    calendar_with_regime = calendar.merge(
        bull_regime[["market_date", "bull_score", "bull_bear_state", "is_bull_strict", "is_bull_loose"]],
        on="market_date",
        how="left",
    )
    strict_calendar = calendar_with_regime.loc[calendar_with_regime["is_bull_strict"].fillna(False)].copy()
    loose_calendar = calendar_with_regime.loc[calendar_with_regime["is_bull_loose"].fillna(False)].copy()

    rows: list[dict[str, object]] = []
    curves: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    portfolio_nav_frames: list[pd.DataFrame] = []
    portfolio_trade_frames: list[pd.DataFrame] = []
    scenario_configs = [
        ("no_bull_filter", pd.Series(True, index=candidates.index), calendar, calendar),
        ("bull7_strict", candidates["is_bull_strict"], calendar, strict_calendar),
        ("bull6_loose", candidates["is_bull_loose"], calendar, loose_calendar),
    ]
    for scenario, scenario_mask, full_calendar, conditional_calendar in scenario_configs:
        base = candidates.loc[
            scenario_mask
            & candidates["v3_full_green"]
            & candidates["rank_score_rebuilt"].ge(float(rank_threshold))
        ].copy()
        for sort_name, sort_column in [("rank_sorted", "rank_score_rebuilt"), ("model_sorted", "model_score")]:
            selected = _take_top(base, int(top_n), sort_column)
            rule = f"{scenario}_v3_full_green_top{int(top_n)}_rank_score_ge_{float(rank_threshold):g}_{sort_name}"
            selected = selected.assign(rule=rule)
            selected_frames.append(selected)
            summary, curve, nav, trades = _summarize(
                selected,
                full_calendar,
                f"{rule}_as_cash",
                history=history,
                top_n=int(top_n),
            )
            rows.append({"calendar_mode": "full_calendar_as_cash", "scenario": scenario, "sort_mode": sort_name, **summary})
            curves.append(curve.assign(rule=summary["rule"], scenario=scenario, sort_mode=sort_name))
            if not nav.empty:
                portfolio_nav_frames.append(nav.assign(calendar_mode="full_calendar_as_cash", scenario=scenario, sort_mode=sort_name))
            if not trades.empty:
                portfolio_trade_frames.append(trades.assign(calendar_mode="full_calendar_as_cash", scenario=scenario, sort_mode=sort_name))
            if not conditional_calendar.empty:
                conditional_summary, conditional_curve, _, _ = _summarize(
                    selected,
                    conditional_calendar,
                    f"{rule}_conditional",
                    history=history,
                    top_n=int(top_n),
                )
                rows.append(
                    {
                        "calendar_mode": "conditional_market_calendar",
                        "scenario": scenario,
                        "sort_mode": sort_name,
                        **conditional_summary,
                    }
                )
                curves.append(conditional_curve.assign(rule=conditional_summary["rule"], scenario=scenario, sort_mode=sort_name))

    result = pd.DataFrame(rows).sort_values(["calendar_mode", "scenario", "sort_mode"]).reset_index(drop=True)
    selected_all = pd.concat(selected_frames, ignore_index=True, sort=False) if selected_frames else pd.DataFrame()
    curves_all = pd.concat(curves, ignore_index=True, sort=False) if curves else pd.DataFrame()
    portfolio_nav_all = pd.concat(portfolio_nav_frames, ignore_index=True, sort=False) if portfolio_nav_frames else pd.DataFrame()
    portfolio_trades_all = pd.concat(portfolio_trade_frames, ignore_index=True, sort=False) if portfolio_trade_frames else pd.DataFrame()
    result.to_csv(output_path / "bull_market_rank_score80_rules.csv", index=False, encoding="utf-8-sig")
    selected_all.to_csv(output_path / "bull_market_rank_score80_selected.csv", index=False, encoding="utf-8-sig")
    curves_all.to_csv(output_path / "bull_market_rank_score80_curves.csv", index=False, encoding="utf-8-sig")
    portfolio_nav_all.to_csv(output_path / "bull_market_rank_score80_portfolio_daily_nav.csv", index=False, encoding="utf-8-sig")
    portfolio_trades_all.to_csv(output_path / "bull_market_rank_score80_portfolio_trades.csv", index=False, encoding="utf-8-sig")
    bull_regime.to_csv(output_path / "bull_bear_regime_daily.csv", index=False, encoding="utf-8-sig")

    latest = bull_regime.sort_values("market_date").tail(1).iloc[0].to_dict() if not bull_regime.empty else {}
    summary = {
        "date_from": date_from,
        "date_to": date_to,
        "rank_threshold": float(rank_threshold),
        "top_n": int(top_n),
        "calendar_days": int(len(calendar)),
        "bull_strict_days": int(calendar_with_regime["is_bull_strict"].fillna(False).sum()),
        "bull_loose_days": int(calendar_with_regime["is_bull_loose"].fillna(False).sum()),
        "bear_days": int(calendar_with_regime["bull_bear_state"].eq("bear").sum()),
        "transition_days": int(calendar_with_regime["bull_bear_state"].eq("transition").sum()),
        "latest_available_market_state": {
            "market_date": str(pd.to_datetime(latest.get("market_date")).date()) if latest else None,
            "sse_close": round(float(latest.get("sse_close", np.nan)), 4) if latest else None,
            "bull_score": int(latest.get("bull_score", 0)) if latest else None,
            "bull_bear_state": latest.get("bull_bear_state") if latest else None,
            "is_bull_strict": bool(latest.get("is_bull_strict", False)) if latest else False,
            "is_bull_loose": bool(latest.get("is_bull_loose", False)) if latest else False,
            "above_ma20_ratio": round(float(latest.get("above_ma20_ratio", np.nan)), 4) if latest else None,
            "amount_ma20_ma60": round(float(latest.get("amount_ma20_ma60", np.nan)), 4) if latest else None,
            "up_amount_ratio": round(float(latest.get("up_amount_ratio", np.nan)), 4) if latest else None,
        },
        "bull_rule": {
            "strict_bull": "bull_score >= 7",
            "loose_bull_sensitivity": "bull_score >= 6",
            "transition": "5 <= bull_score <= 6 under strict classification",
            "bear": "bull_score <= 4",
            "checks": [
                "SSE close > MA250",
                "SSE MA60 > MA250",
                "SSE MA20 > MA60",
                "SSE MA60 slope over 20 trading days > 0",
                "SSE 60-day return > 0",
                "SSE 120-day drawdown >= -15%",
                "main-board above-MA20 ratio >= 50%",
                "market amount MA20/MA60 >= 1.0",
                "up amount ratio >= 50%",
                "V3 trend_score >= 4 and flow_score >= 3",
            ],
        },
        "rank_score_rebuild_note": (
            "rank_score_rebuilt follows the dashboard quick-board enhanced_attention_score/final_rank_score formula "
            "using cached historical combined_candidates fields. Full-market historical page rank_score was not cached, "
            "so relative probability is ranked within each day's strategy-candidate pool."
        ),
        "rules": result.replace({np.nan: None}).to_dict("records"),
        "summary_path": str(output_path / "bull_market_rank_score80_summary.json"),
        "rules_path": str(output_path / "bull_market_rank_score80_rules.csv"),
        "selected_path": str(output_path / "bull_market_rank_score80_selected.csv"),
        "curves_path": str(output_path / "bull_market_rank_score80_curves.csv"),
        "portfolio_nav_path": str(output_path / "bull_market_rank_score80_portfolio_daily_nav.csv"),
        "portfolio_trades_path": str(output_path / "bull_market_rank_score80_portfolio_trades.csv"),
        "regime_path": str(output_path / "bull_bear_regime_daily.csv"),
        "notes": [
            "annualized_return/max_drawdown/ending_equity remain legacy average-selected-return metrics for compatibility.",
            "portfolio_* fields use the unified portfolio backtest engine and should be preferred for trading-system evaluation.",
            "conditional_market_calendar remains a compressed research view; portfolio_* metrics are only exported for full_calendar_as_cash paths.",
        ],
    }
    (output_path / "bull_market_rank_score80_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest bull-market-only V3 full_green rank-score strategy.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--v3-dir", default=str(DEFAULT_V3_DIR))
    parser.add_argument("--sse-path", default=str(DEFAULT_SSE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--date-from", default="2016-05-27")
    parser.add_argument("--date-to", default="2026-05-26")
    parser.add_argument("--rank-threshold", type=float, default=80.0)
    parser.add_argument("--top-n", type=int, default=3)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = run_backtest(
        source_dir=args.source_dir,
        v3_dir=args.v3_dir,
        sse_path=args.sse_path,
        output_dir=args.output_dir,
        date_from=args.date_from,
        date_to=args.date_to,
        rank_threshold=float(args.rank_threshold),
        top_n=int(args.top_n),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
