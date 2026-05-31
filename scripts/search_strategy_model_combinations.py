from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / ".cache" / "ten_year_model_strategy_validation"
DEFAULT_V3_DIR = PROJECT_ROOT / ".cache" / "ten_year_market_regime_v3"
DEFAULT_BULL_PATH = PROJECT_ROOT / ".cache" / "ten_year_bull_market_rank_score" / "bull_bear_regime_daily.csv"
DEFAULT_SSE_PATH = PROJECT_ROOT / ".cache" / "ten_year_sse3500_filter" / "sse_000001_daily.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".cache" / "ten_year_strategy_search"


@dataclass(frozen=True)
class SearchRule:
    strategy_mode: str
    market_filter: str
    score_threshold: float
    priority_threshold: float | None
    top_n: int
    sort_mode: str
    pause_dd: float | None
    pause_days: int

    @property
    def name(self) -> str:
        priority = "none" if self.priority_threshold is None else f"{self.priority_threshold:g}"
        pause = "none" if self.pause_dd is None or self.pause_days <= 0 else f"{abs(self.pause_dd):.0%}_{self.pause_days}d"
        return (
            f"{self.strategy_mode}__{self.market_filter}__score{self.score_threshold:g}"
            f"__prio{priority}__top{self.top_n}__{self.sort_mode}__pause{pause}"
        )


def _load_model_calendar(path: Path, date_from: str, date_to: str) -> pd.DataFrame:
    dates: set[pd.Timestamp] = set()
    for chunk in pd.read_csv(path, encoding="utf-8-sig", usecols=["market_date"], chunksize=500_000):
        parsed = pd.to_datetime(chunk["market_date"], errors="coerce").dropna().dt.normalize()
        dates.update(pd.Timestamp(value) for value in parsed.unique())
    start = pd.to_datetime(date_from)
    end = pd.to_datetime(date_to)
    calendar = pd.DataFrame({"market_date": sorted(dates)})
    return calendar.loc[calendar["market_date"].between(start, end, inclusive="both")].copy()


def _load_candidates(source_dir: Path, date_from: str, date_to: str) -> pd.DataFrame:
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
        "strategy_rank",
        "latest_price",
        "change_pct",
        "amount",
        "turnover",
        "industry_ret_2d_pct",
        "industry_up_count",
        "model_probability",
        "model_score",
        "hold_3d_return",
        "max_high_return",
        "max_drawdown",
        "entry_price",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    text = frame["candidate_strategy"].fillna("").astype(str).str.lower()
    frame["strategy_family"] = np.select(
        [
            text.str.contains("strategy1", regex=False) | text.str.contains("策略1", regex=False),
            text.str.contains("strategy2", regex=False) | text.str.contains("策略2", regex=False),
            text.str.contains("strategy3", regex=False) | text.str.contains("策略3", regex=False),
        ],
        ["strategy1", "strategy2", "strategy3"],
        default="unknown",
    )
    return frame.dropna(subset=["market_date", "symbol", "model_score", "hold_3d_return"]).copy()


def _enrich_candidates(candidates: pd.DataFrame, v3_dir: Path, bull_path: Path, sse_path: Path) -> pd.DataFrame:
    regime = pd.read_csv(v3_dir / "market_regime_daily.csv", encoding="utf-8-sig", parse_dates=["market_date"])
    bull = pd.read_csv(bull_path, encoding="utf-8-sig", parse_dates=["market_date"])
    sse = pd.read_csv(sse_path, encoding="utf-8-sig", parse_dates=["market_date"])
    keep_regime = [
        "market_date",
        "trend_green",
        "flow_green",
        "market_green",
        "internal_green",
        "v3_full_green",
        "v3_yellow",
        "market_state",
        "market_ret",
        "up_ratio",
        "limit_up_count",
        "limit_down_count",
        "drawdown_20d",
        "amount_ma5_ma20",
        "trend_score",
        "flow_score",
        "above_ma20_ratio",
        "amount_ma20_ma60",
        "up_amount_ratio",
        "strong_amount_ratio",
        "strategy_candidate_count",
        "score_ge68_count",
        "score_ge70_count",
    ]
    keep_bull = ["market_date", "bull_score", "bull_bear_state", "is_bull_strict", "is_bull_loose"]
    keep_sse = ["market_date", "sse_close"]
    frame = candidates.merge(regime[[column for column in keep_regime if column in regime.columns]], on="market_date", how="left")
    frame = frame.merge(bull[[column for column in keep_bull if column in bull.columns]], on="market_date", how="left")
    frame = frame.merge(sse[[column for column in keep_sse if column in sse.columns]], on="market_date", how="left")
    for column in ["trend_green", "flow_green", "market_green", "internal_green", "v3_full_green", "v3_yellow"]:
        if column in frame.columns:
            frame[column] = frame[column].astype("boolean").fillna(False).astype(bool)
    for column in ["is_bull_strict", "is_bull_loose"]:
        if column in frame.columns:
            frame[column] = frame[column].astype("boolean").fillna(False).astype(bool)
    frame["priority_score"] = pd.to_numeric(frame.get("candidate_priority"), errors="coerce").fillna(0.0).clip(0.0, 100.0)
    frame["model_priority_80_20"] = frame["model_score"] * 0.80 + frame["priority_score"] * 0.20
    frame["model_priority_70_30"] = frame["model_score"] * 0.70 + frame["priority_score"] * 0.30
    return frame


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
    if mode == "trend_green":
        return frame["trend_green"].fillna(False)
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
    if mode == "sse3500":
        return pd.to_numeric(frame["sse_close"], errors="coerce").ge(3500.0)
    if mode == "sse3500_v3_full_green":
        return pd.to_numeric(frame["sse_close"], errors="coerce").ge(3500.0) & frame["v3_full_green"].fillna(False)
    if mode == "bull7_sse3500_v3_full_green":
        return (
            frame["is_bull_strict"].fillna(False)
            & pd.to_numeric(frame["sse_close"], errors="coerce").ge(3500.0)
            & frame["v3_full_green"].fillna(False)
        )
    raise ValueError(f"Unknown market filter: {mode}")


def _select_top(frame: pd.DataFrame, rule: SearchRule) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    data = frame.loc[
        _strategy_mask(frame, rule.strategy_mode)
        & _market_mask(frame, rule.market_filter)
        & frame["model_score"].ge(float(rule.score_threshold))
    ].copy()
    if rule.priority_threshold is not None:
        data = data.loc[data["priority_score"].ge(float(rule.priority_threshold))].copy()
    if data.empty:
        return data
    sort_column = rule.sort_mode
    sort_columns = ["market_date", sort_column]
    ascending = [True, False]
    if sort_column != "model_score":
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


def _summarize(selected: pd.DataFrame, calendar: pd.DataFrame, rule_name: str, *, pause_dd: float | None, pause_days: int) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    frame = selected.copy()
    if not frame.empty:
        frame["hold_3d_return"] = pd.to_numeric(frame["hold_3d_return"], errors="coerce")
        frame = frame.dropna(subset=["market_date", "hold_3d_return"]).copy()
    if frame.empty:
        daily = pd.DataFrame(columns=["market_date", "selected", "avg_return", "avg_model_score", "avg_priority_score"])
    else:
        daily = (
            frame.groupby("market_date", as_index=False)
            .agg(
                selected=("symbol", "count"),
                avg_return=("hold_3d_return", "mean"),
                avg_model_score=("model_score", "mean"),
                avg_priority_score=("priority_score", "mean"),
            )
            .sort_values("market_date")
        )
    curve = calendar[["market_date"]].merge(daily, on="market_date", how="left")
    curve["selected"] = curve["selected"].fillna(0).astype(int)
    curve["raw_selected"] = curve["selected"]
    curve["raw_avg_return"] = curve["avg_return"].fillna(0.0)
    curve["avg_return"] = curve["raw_avg_return"]
    curve["paused"] = False

    if pause_dd is not None and int(pause_days) > 0:
        equity = 1.0
        peak = 1.0
        pause_remaining = 0
        allowed_dates: list[pd.Timestamp] = []
        returns: list[float] = []
        selected_counts: list[int] = []
        paused_flags: list[bool] = []
        for row in curve.itertuples(index=False):
            date_value = pd.Timestamp(row.market_date)
            if pause_remaining > 0:
                returns.append(0.0)
                selected_counts.append(0)
                paused_flags.append(True)
                pause_remaining -= 1
                continue
            day_return = float(row.raw_avg_return or 0.0)
            day_selected = int(row.raw_selected or 0)
            returns.append(day_return)
            selected_counts.append(day_selected)
            paused_flags.append(False)
            if day_selected > 0:
                allowed_dates.append(date_value)
            equity *= 1.0 + day_return
            peak = max(peak, equity)
            drawdown = equity / peak - 1.0 if peak else 0.0
            if drawdown <= float(pause_dd):
                pause_remaining = int(pause_days)
        curve["avg_return"] = returns
        curve["selected"] = selected_counts
        curve["paused"] = paused_flags
        frame = frame.loc[frame["market_date"].isin(set(allowed_dates))].copy()

    curve["equity"] = (1.0 + curve["avg_return"].fillna(0.0)).cumprod()
    curve["running_max"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["running_max"].replace(0.0, np.nan) - 1.0
    trade_returns = frame["hold_3d_return"].dropna() if not frame.empty else pd.Series(dtype=float)
    active_daily = curve.loc[curve["selected"].gt(0), "avg_return"]
    ending = float(curve["equity"].iloc[-1]) if not curve.empty else 1.0
    annualized = ending ** (252.0 / len(curve)) - 1.0 if len(curve) and ending > 0 else 0.0
    max_drawdown = float(curve["drawdown"].min()) if not curve.empty else 0.0
    summary = {
        "rule": rule_name,
        "calendar_days": int(len(curve)),
        "active_days": int((curve["selected"] > 0).sum()),
        "coverage_pct": round(float((curve["selected"] > 0).mean()) * 100.0, 2) if not curve.empty else 0.0,
        "selected_rows": int(len(frame)),
        "avg_trade_return": round(float(trade_returns.mean()), 6) if not trade_returns.empty else None,
        "trade_win_rate": round(float((trade_returns > 0).mean()), 4) if not trade_returns.empty else None,
        "target_hit_rate": round(float((trade_returns >= 0.03).mean()), 4) if not trade_returns.empty else None,
        "active_daily_return": round(float(active_daily.mean()), 6) if not active_daily.empty else None,
        "active_daily_win_rate": round(float((active_daily > 0).mean()), 4) if not active_daily.empty else None,
        "annualized_return": round(float(annualized), 6),
        "max_drawdown": round(float(max_drawdown), 6),
        "ending_equity": round(float(ending), 6),
        "return_drawdown_ratio": round(float(annualized / abs(max_drawdown)), 6) if max_drawdown < 0 else None,
        "avg_selected_model_score": round(float(frame["model_score"].mean()), 4) if not frame.empty else None,
        "avg_selected_priority_score": round(float(frame["priority_score"].mean()), 4) if not frame.empty else None,
        "paused_days": int(curve["paused"].sum()),
    }
    return summary, curve, frame


def _recent_metrics(curve: pd.DataFrame, selected: pd.DataFrame, date_from: pd.Timestamp) -> dict[str, object]:
    recent_curve = curve.loc[curve["market_date"].ge(date_from)].copy()
    recent_selected = selected.loc[selected["market_date"].ge(date_from)].copy() if not selected.empty else selected
    if recent_curve.empty:
        return {}
    recent_curve["equity"] = (1.0 + recent_curve["avg_return"].fillna(0.0)).cumprod()
    recent_curve["running_max"] = recent_curve["equity"].cummax()
    recent_curve["drawdown"] = recent_curve["equity"] / recent_curve["running_max"].replace(0.0, np.nan) - 1.0
    ending = float(recent_curve["equity"].iloc[-1])
    annualized = ending ** (252.0 / len(recent_curve)) - 1.0 if ending > 0 else 0.0
    max_drawdown = float(recent_curve["drawdown"].min()) if not recent_curve.empty else 0.0
    trade_returns = recent_selected["hold_3d_return"].dropna() if not recent_selected.empty else pd.Series(dtype=float)
    active_daily = recent_curve.loc[recent_curve["selected"].gt(0), "avg_return"]
    return {
        "recent_calendar_days": int(len(recent_curve)),
        "recent_active_days": int((recent_curve["selected"] > 0).sum()),
        "recent_selected_rows": int(len(recent_selected)),
        "recent_annualized_return": round(float(annualized), 6),
        "recent_max_drawdown": round(float(max_drawdown), 6),
        "recent_active_daily_win_rate": round(float((active_daily > 0).mean()), 4) if not active_daily.empty else None,
        "recent_trade_win_rate": round(float((trade_returns > 0).mean()), 4) if not trade_returns.empty else None,
        "recent_avg_trade_return": round(float(trade_returns.mean()), 6) if not trade_returns.empty else None,
    }


def _build_search_rules(preset: str = "focused") -> list[SearchRule]:
    rules: list[SearchRule] = []
    if preset == "full":
        strategy_modes = ["all", "strategy1", "strategy2", "strategy3", "old12"]
        market_filters = [
            "none",
            "trend_green",
            "market_green",
            "v3_full_green",
            "bull7",
            "bull6",
            "bull7_v3_full_green",
            "bull6_v3_full_green",
            "sse3500_v3_full_green",
            "bull7_sse3500_v3_full_green",
        ]
        score_thresholds = [60.0, 64.0, 66.0, 68.0, 70.0, 72.0, 74.0]
        priority_thresholds: list[float | None] = [None, 60.0, 65.0, 70.0]
        top_ns = [1, 2, 3]
        sort_modes = ["model_score", "model_priority_80_20", "model_priority_70_30"]
        pause_configs: list[tuple[float | None, int]] = [(None, 0), (-0.08, 5), (-0.10, 10)]
    elif preset == "focused":
        strategy_modes = ["all", "strategy3"]
        market_filters = [
            "none",
            "market_green",
            "v3_full_green",
            "bull7_v3_full_green",
            "bull6_v3_full_green",
            "sse3500_v3_full_green",
            "bull7_sse3500_v3_full_green",
        ]
        score_thresholds = [66.0, 68.0, 70.0, 72.0]
        priority_thresholds = [None, 60.0, 65.0]
        top_ns = [1, 2, 3]
        sort_modes = ["model_score", "model_priority_80_20"]
        pause_configs = [(None, 0), (-0.08, 5), (-0.10, 10)]
    else:
        raise ValueError(f"Unknown preset: {preset}")
    for strategy_mode in strategy_modes:
        for market_filter in market_filters:
            for score_threshold in score_thresholds:
                for priority_threshold in priority_thresholds:
                    for top_n in top_ns:
                        for sort_mode in sort_modes:
                            for pause_dd, pause_days in pause_configs:
                                rules.append(
                                    SearchRule(
                                        strategy_mode=strategy_mode,
                                        market_filter=market_filter,
                                        score_threshold=score_threshold,
                                        priority_threshold=priority_threshold,
                                        top_n=top_n,
                                        sort_mode=sort_mode,
                                        pause_dd=pause_dd,
                                        pause_days=pause_days,
                                    )
                                )
    return rules


def run_search(
    *,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    v3_dir: str | Path = DEFAULT_V3_DIR,
    bull_path: str | Path = DEFAULT_BULL_PATH,
    sse_path: str | Path = DEFAULT_SSE_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    date_from: str = "2016-05-27",
    date_to: str = "2026-05-26",
    min_active_days: int = 80,
    preset: str = "focused",
) -> dict[str, object]:
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    calendar = _load_model_calendar(source_path / "model_scores.csv", date_from, date_to)
    candidates = _load_candidates(source_path, date_from, date_to)
    candidates = _enrich_candidates(candidates, Path(v3_dir), Path(bull_path), Path(sse_path))

    rows: list[dict[str, object]] = []
    best_payloads: list[tuple[SearchRule, pd.DataFrame, pd.DataFrame, dict[str, object]]] = []
    recent_start = pd.to_datetime(date_to) - pd.DateOffset(years=1)
    rules = _build_search_rules(preset=preset)
    for index, rule in enumerate(rules, start=1):
        if index == 1 or index % 1000 == 0:
            print(f"[search] {index}/{len(rules)}", flush=True)
        selected = _select_top(candidates, rule)
        if selected.empty:
            continue
        summary, curve, allowed_selected = _summarize(
            selected,
            calendar,
            rule.name,
            pause_dd=rule.pause_dd,
            pause_days=rule.pause_days,
        )
        if summary["active_days"] <= 0:
            continue
        summary.update(
            {
                "strategy_mode": rule.strategy_mode,
                "market_filter": rule.market_filter,
                "score_threshold": rule.score_threshold,
                "priority_threshold": rule.priority_threshold,
                "top_n": rule.top_n,
                "sort_mode": rule.sort_mode,
                "pause_dd": rule.pause_dd,
                "pause_days": rule.pause_days,
            }
        )
        summary.update(_recent_metrics(curve, allowed_selected, recent_start))
        rows.append(summary)
        if int(summary["active_days"]) >= int(min_active_days):
            best_payloads.append((rule, curve, allowed_selected, summary))

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No valid search rules produced trades.")
    result["meets_sample_floor"] = result["active_days"].ge(int(min_active_days))
    result["risk_score"] = (
        result["annualized_return"].fillna(0.0) / result["max_drawdown"].abs().replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    result["selection_score"] = (
        result["annualized_return"].fillna(0.0) * 100.0
        + result["risk_score"].fillna(0.0) * 12.0
        + result["active_daily_win_rate"].fillna(0.0) * 8.0
        - result["max_drawdown"].abs().fillna(1.0) * 18.0
    )
    result = result.sort_values(
        ["meets_sample_floor", "selection_score", "risk_score", "annualized_return"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    result.to_csv(output_path / "strategy_search_results.csv", index=False, encoding="utf-8-sig")
    top = result.head(100).copy()
    top.to_csv(output_path / "strategy_search_top100.csv", index=False, encoding="utf-8-sig")

    top_rules = set(top["rule"].head(25))
    curves = []
    selected_frames = []
    for rule, curve, selected, summary in best_payloads:
        if summary["rule"] not in top_rules:
            continue
        curves.append(curve.assign(rule=summary["rule"]))
        selected_frames.append(selected.assign(rule=summary["rule"]))
    if curves:
        pd.concat(curves, ignore_index=True, sort=False).to_csv(
            output_path / "strategy_search_top_curves.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if selected_frames:
        pd.concat(selected_frames, ignore_index=True, sort=False).to_csv(
            output_path / "strategy_search_top_selected.csv",
            index=False,
            encoding="utf-8-sig",
        )

    robust = result.loc[
        result["meets_sample_floor"]
        & result["annualized_return"].gt(0.0)
        & result["max_drawdown"].gt(-0.25)
        & result["active_daily_win_rate"].ge(0.50)
    ].copy()
    robust.to_csv(output_path / "strategy_search_robust_candidates.csv", index=False, encoding="utf-8-sig")

    summary = {
        "date_from": date_from,
        "date_to": date_to,
        "calendar_days": int(len(calendar)),
        "candidate_rows": int(len(candidates)),
        "candidate_days": int(candidates["market_date"].nunique()),
        "rules_evaluated": int(len(rules)),
        "preset": preset,
        "rules_with_trades": int(len(result)),
        "min_active_days": int(min_active_days),
        "best_overall": result.head(20).replace({np.nan: None}).to_dict("records"),
        "best_robust": robust.head(20).replace({np.nan: None}).to_dict("records"),
        "paths": {
            "results": str(output_path / "strategy_search_results.csv"),
            "top100": str(output_path / "strategy_search_top100.csv"),
            "robust_candidates": str(output_path / "strategy_search_robust_candidates.csv"),
            "top_curves": str(output_path / "strategy_search_top_curves.csv"),
            "top_selected": str(output_path / "strategy_search_top_selected.csv"),
            "summary": str(output_path / "strategy_search_summary.json"),
        },
    }
    (output_path / "strategy_search_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search ten-year strategy/model/market-filter combinations.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--v3-dir", default=str(DEFAULT_V3_DIR))
    parser.add_argument("--bull-path", default=str(DEFAULT_BULL_PATH))
    parser.add_argument("--sse-path", default=str(DEFAULT_SSE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--date-from", default="2016-05-27")
    parser.add_argument("--date-to", default="2026-05-26")
    parser.add_argument("--min-active-days", type=int, default=80)
    parser.add_argument("--preset", choices=["focused", "full"], default="focused")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = run_search(
        source_dir=args.source_dir,
        v3_dir=args.v3_dir,
        bull_path=args.bull_path,
        sse_path=args.sse_path,
        output_dir=args.output_dir,
        date_from=args.date_from,
        date_to=args.date_to,
        min_active_days=int(args.min_active_days),
        preset=str(args.preset),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
