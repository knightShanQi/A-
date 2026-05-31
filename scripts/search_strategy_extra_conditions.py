from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from search_strategy_model_combinations import (
    DEFAULT_BULL_PATH,
    DEFAULT_SOURCE_DIR,
    DEFAULT_SSE_PATH,
    DEFAULT_V3_DIR,
    SearchRule,
    _enrich_candidates,
    _load_candidates,
    _load_model_calendar,
    _select_top,
)
from validate_strategy_recommendation_robustness import _build_curve_with_cost, _period_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".cache" / "ten_year_strategy_extra_conditions"


@dataclass(frozen=True)
class ExtraCondition:
    name: str
    stock: str
    market: str


BASE_RULES: tuple[SearchRule, ...] = (
    SearchRule("strategy3", "bull7_v3_full_green", 68.0, None, 3, "model_score", None, 0),
    SearchRule("strategy3", "bull7_v3_full_green", 68.0, None, 2, "model_score", None, 0),
    SearchRule("all", "bull7_v3_full_green", 68.0, None, 3, "model_score", None, 0),
    SearchRule("all", "v3_full_green", 68.0, None, 2, "model_score", None, 0),
    SearchRule("all", "bull7_v3_full_green", 68.0, 65.0, 2, "model_score", None, 0),
)


STOCK_CONDITIONS = [
    "none",
    "avoid_chase_8p5",
    "avoid_chase_7",
    "positive_day",
    "mild_positive_no_chase",
    "pullback_or_mild",
    "liquid_100m",
    "liquid_200m",
    "turnover_1_20",
    "turnover_2_20",
    "liquid_turnover_1_20",
    "liquid_no_chase",
    "industry_positive",
    "industry_strong_1pct",
    "industry_positive_liquid",
]

MARKET_CONDITIONS = [
    "none",
    "breadth50",
    "breadth60",
    "amount110",
    "up_amount50",
    "strong_amount_gt1",
    "market_ret_nonnegative",
    "drawdown20_gt_minus6",
    "limitup_lt80",
    "score68_abundant50",
    "candidate_count_ge20",
    "breadth50_amount100_up50",
]


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def _stock_mask(frame: pd.DataFrame, name: str) -> pd.Series:
    true = pd.Series(True, index=frame.index)
    change = _num(frame, "change_pct")
    amount = _num(frame, "amount")
    turnover = _num(frame, "turnover")
    industry_ret = _num(frame, "industry_ret_2d_pct")
    if name == "none":
        return true
    if name == "avoid_chase_8p5":
        return change.le(8.5) | change.isna()
    if name == "avoid_chase_7":
        return change.le(7.0) | change.isna()
    if name == "positive_day":
        return change.ge(0.0)
    if name == "mild_positive_no_chase":
        return change.between(0.0, 8.5, inclusive="both")
    if name == "pullback_or_mild":
        return change.between(-2.0, 5.0, inclusive="both")
    if name == "liquid_100m":
        return amount.ge(100_000_000.0)
    if name == "liquid_200m":
        return amount.ge(200_000_000.0)
    if name == "turnover_1_20":
        return turnover.between(1.0, 20.0, inclusive="both")
    if name == "turnover_2_20":
        return turnover.between(2.0, 20.0, inclusive="both")
    if name == "liquid_turnover_1_20":
        return amount.ge(100_000_000.0) & turnover.between(1.0, 20.0, inclusive="both")
    if name == "liquid_no_chase":
        return amount.ge(100_000_000.0) & change.le(8.5)
    if name == "industry_positive":
        return industry_ret.ge(0.0)
    if name == "industry_strong_1pct":
        return industry_ret.ge(1.0)
    if name == "industry_positive_liquid":
        return industry_ret.ge(0.0) & amount.ge(100_000_000.0)
    raise ValueError(f"Unknown stock condition: {name}")


def _market_mask(frame: pd.DataFrame, name: str) -> pd.Series:
    true = pd.Series(True, index=frame.index)
    above = _num(frame, "above_ma20_ratio")
    amount_ratio = _num(frame, "amount_ma20_ma60")
    up_amount = _num(frame, "up_amount_ratio")
    strong_amount = _num(frame, "strong_amount_ratio")
    market_ret = _num(frame, "market_ret")
    drawdown20 = _num(frame, "drawdown_20d")
    limit_up = _num(frame, "limit_up_count")
    score68 = _num(frame, "score_ge68_count")
    candidate_count = _num(frame, "strategy_candidate_count")
    if name == "none":
        return true
    if name == "breadth50":
        return above.ge(0.50)
    if name == "breadth60":
        return above.ge(0.60)
    if name == "amount110":
        return amount_ratio.ge(1.10)
    if name == "up_amount50":
        return up_amount.ge(0.50)
    if name == "strong_amount_gt1":
        return strong_amount.ge(1.0)
    if name == "market_ret_nonnegative":
        return market_ret.ge(0.0)
    if name == "drawdown20_gt_minus6":
        return drawdown20.ge(-0.06)
    if name == "limitup_lt80":
        return limit_up.lt(80.0)
    if name == "score68_abundant50":
        return score68.ge(50.0)
    if name == "candidate_count_ge20":
        return candidate_count.ge(20.0)
    if name == "breadth50_amount100_up50":
        return above.ge(0.50) & amount_ratio.ge(1.0) & up_amount.ge(0.50)
    raise ValueError(f"Unknown market condition: {name}")


def _condition_mask(frame: pd.DataFrame, condition: ExtraCondition) -> pd.Series:
    return _stock_mask(frame, condition.stock) & _market_mask(frame, condition.market)


def _year_stability(curve: pd.DataFrame, selected: pd.DataFrame) -> dict[str, object]:
    if curve.empty:
        return {"active_years": 0, "negative_active_years": 0, "worst_active_year_ann": None, "worst_year_dd": None}
    local_curve = curve.copy()
    local_selected = selected.copy()
    local_curve["year"] = local_curve["market_date"].dt.year
    if not local_selected.empty:
        local_selected["year"] = local_selected["market_date"].dt.year
    active_years = 0
    negative_years = 0
    worst_ann: float | None = None
    worst_dd: float | None = None
    for year, year_curve in local_curve.groupby("year", sort=True):
        if int((year_curve["selected"] > 0).sum()) <= 0:
            continue
        year_selected = local_selected.loc[local_selected["year"].eq(year)].copy() if not local_selected.empty else local_selected
        metrics = _period_metrics(year_curve.drop(columns=["year"]), year_selected)
        active_years += 1
        ann = float(metrics["annualized_return"])
        dd = float(metrics["max_drawdown"])
        if ann < 0.0:
            negative_years += 1
        worst_ann = ann if worst_ann is None else min(worst_ann, ann)
        worst_dd = dd if worst_dd is None else min(worst_dd, dd)
    return {
        "active_years": int(active_years),
        "negative_active_years": int(negative_years),
        "worst_active_year_ann": round(float(worst_ann), 6) if worst_ann is not None else None,
        "worst_year_dd": round(float(worst_dd), 6) if worst_dd is not None else None,
    }


def _build_conditions() -> list[ExtraCondition]:
    conditions: list[ExtraCondition] = []
    for stock in STOCK_CONDITIONS:
        for market in MARKET_CONDITIONS:
            if stock == "none" and market == "none":
                name = "none"
            elif stock == "none":
                name = f"market_{market}"
            elif market == "none":
                name = f"stock_{stock}"
            else:
                name = f"stock_{stock}__market_{market}"
            conditions.append(ExtraCondition(name=name, stock=stock, market=market))
    return conditions


def run_search(
    *,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    v3_dir: str | Path = DEFAULT_V3_DIR,
    bull_path: str | Path = DEFAULT_BULL_PATH,
    sse_path: str | Path = DEFAULT_SSE_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    date_from: str = "2016-05-27",
    date_to: str = "2026-05-26",
    cost_bps: float = 20.0,
    min_active_days: int = 70,
) -> dict[str, object]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    source_path = Path(source_dir)
    calendar = _load_model_calendar(source_path / "model_scores.csv", date_from, date_to)
    candidates = _load_candidates(source_path, date_from, date_to)
    candidates = _enrich_candidates(candidates, Path(v3_dir), Path(bull_path), Path(sse_path))

    rows: list[dict[str, object]] = []
    selected_frames: list[pd.DataFrame] = []
    conditions = _build_conditions()
    for rule in BASE_RULES:
        for condition in conditions:
            filtered = candidates.loc[_condition_mask(candidates, condition)].copy()
            selected = _select_top(filtered, rule)
            if selected.empty:
                continue
            curve, adjusted = _build_curve_with_cost(selected, calendar, cost_bps)
            metrics = _period_metrics(curve, adjusted)
            metrics.update(_year_stability(curve, adjusted))
            if int(metrics["active_days"]) < int(min_active_days):
                continue
            metrics.update(
                {
                    "base_rule": rule.name,
                    "strategy_mode": rule.strategy_mode,
                    "market_filter": rule.market_filter,
                    "score_threshold": rule.score_threshold,
                    "priority_threshold": rule.priority_threshold,
                    "top_n": rule.top_n,
                    "sort_mode": rule.sort_mode,
                    "condition": condition.name,
                    "stock_condition": condition.stock,
                    "market_condition": condition.market,
                    "cost_bps": float(cost_bps),
                }
            )
            metrics["extra_condition_score"] = (
                float(metrics["annualized_return"]) * 100.0
                + float(metrics["return_drawdown_ratio"] or 0.0) * 10.0
                + float(metrics["active_daily_win_rate"] or 0.0) * 5.0
                - abs(float(metrics["max_drawdown"])) * 12.0
                - int(metrics["negative_active_years"]) * 1.5
            )
            rows.append(metrics)
            if condition.name in {"none", "stock_liquid_100m", "stock_liquid_no_chase"}:
                selected_frames.append(adjusted.assign(base_rule=rule.name, condition=condition.name, cost_bps=float(cost_bps)))

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No extra-condition rules produced enough trades.")
    result = result.sort_values(
        ["negative_active_years", "extra_condition_score", "annualized_return", "return_drawdown_ratio"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    result.to_csv(output_path / "extra_condition_search_results.csv", index=False, encoding="utf-8-sig")
    result.head(100).to_csv(output_path / "extra_condition_search_top100.csv", index=False, encoding="utf-8-sig")
    if selected_frames:
        pd.concat(selected_frames, ignore_index=True, sort=False).to_csv(
            output_path / "extra_condition_selected_samples.csv",
            index=False,
            encoding="utf-8-sig",
        )
    summary = {
        "date_from": date_from,
        "date_to": date_to,
        "calendar_days": int(len(calendar)),
        "candidate_rows": int(len(candidates)),
        "cost_bps": float(cost_bps),
        "min_active_days": int(min_active_days),
        "base_rules": [rule.name for rule in BASE_RULES],
        "conditions_evaluated": int(len(BASE_RULES) * len(conditions)),
        "rules_with_enough_trades": int(len(result)),
        "best": result.head(20).replace({np.nan: None}).to_dict("records"),
        "paths": {
            "results": str(output_path / "extra_condition_search_results.csv"),
            "top100": str(output_path / "extra_condition_search_top100.csv"),
            "selected_samples": str(output_path / "extra_condition_selected_samples.csv"),
            "summary": str(output_path / "extra_condition_search_summary.json"),
        },
    }
    (output_path / "extra_condition_search_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search additional stock and market conditions for strategy combinations.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--v3-dir", default=str(DEFAULT_V3_DIR))
    parser.add_argument("--bull-path", default=str(DEFAULT_BULL_PATH))
    parser.add_argument("--sse-path", default=str(DEFAULT_SSE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--date-from", default="2016-05-27")
    parser.add_argument("--date-to", default="2026-05-26")
    parser.add_argument("--cost-bps", type=float, default=20.0)
    parser.add_argument("--min-active-days", type=int, default=70)
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
        cost_bps=float(args.cost_bps),
        min_active_days=int(args.min_active_days),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
