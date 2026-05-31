from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from search_strategy_model_combinations import (
    DEFAULT_BULL_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_DIR,
    DEFAULT_SSE_PATH,
    DEFAULT_V3_DIR,
    SearchRule,
    _enrich_candidates,
    _load_candidates,
    _load_model_calendar,
    _select_top,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBUSTNESS_DIR = PROJECT_ROOT / ".cache" / "ten_year_strategy_robustness"


@dataclass(frozen=True)
class CandidateRule:
    label: str
    rule: SearchRule


DEFAULT_RULES: tuple[CandidateRule, ...] = (
    CandidateRule(
        "recommended_strategy3_bull7_v3_score68_top3",
        SearchRule("strategy3", "bull7_v3_full_green", 68.0, None, 3, "model_score", None, 0),
    ),
    CandidateRule(
        "balanced_all_bull7_v3_score68_top3",
        SearchRule("all", "bull7_v3_full_green", 68.0, None, 3, "model_score", None, 0),
    ),
    CandidateRule(
        "quality_all_bull7_v3_score68_prio60_top3",
        SearchRule("all", "bull7_v3_full_green", 68.0, 60.0, 3, "model_score", None, 0),
    ),
    CandidateRule(
        "loose_bull_all_bull6_v3_score68_top3",
        SearchRule("all", "bull6_v3_full_green", 68.0, None, 3, "model_score", None, 0),
    ),
    CandidateRule(
        "aggressive_all_bull7_v3_score68_prio65_top2",
        SearchRule("all", "bull7_v3_full_green", 68.0, 65.0, 2, "model_score", None, 0),
    ),
    CandidateRule(
        "aggressive_all_v3_score68_top2",
        SearchRule("all", "v3_full_green", 68.0, None, 2, "model_score", None, 0),
    ),
    CandidateRule(
        "baseline_all_v3_score68_top3",
        SearchRule("all", "v3_full_green", 68.0, None, 3, "model_score", None, 0),
    ),
)


def _period_metrics(curve: pd.DataFrame, selected: pd.DataFrame) -> dict[str, object]:
    if curve.empty:
        return {
            "calendar_days": 0,
            "active_days": 0,
            "selected_rows": 0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "ending_equity": 1.0,
            "active_daily_win_rate": None,
            "trade_win_rate": None,
            "avg_trade_return": None,
        }
    local_curve = curve.copy()
    local_curve["equity"] = (1.0 + local_curve["avg_return"].fillna(0.0)).cumprod()
    local_curve["running_max"] = local_curve["equity"].cummax()
    local_curve["drawdown"] = local_curve["equity"] / local_curve["running_max"].replace(0.0, np.nan) - 1.0
    ending = float(local_curve["equity"].iloc[-1])
    annualized = ending ** (252.0 / len(local_curve)) - 1.0 if ending > 0 and len(local_curve) else 0.0
    max_drawdown = float(local_curve["drawdown"].min()) if not local_curve.empty else 0.0
    active_daily = local_curve.loc[local_curve["selected"].gt(0), "avg_return"].dropna()
    trade_returns = selected["adjusted_return"].dropna() if not selected.empty else pd.Series(dtype=float)
    return {
        "calendar_days": int(len(local_curve)),
        "active_days": int((local_curve["selected"] > 0).sum()),
        "selected_rows": int(len(selected)),
        "coverage_pct": round(float((local_curve["selected"] > 0).mean()) * 100.0, 2),
        "annualized_return": round(float(annualized), 6),
        "max_drawdown": round(float(max_drawdown), 6),
        "ending_equity": round(float(ending), 6),
        "return_drawdown_ratio": round(float(annualized / abs(max_drawdown)), 6) if max_drawdown < 0 else None,
        "active_daily_return": round(float(active_daily.mean()), 6) if not active_daily.empty else None,
        "active_daily_win_rate": round(float((active_daily > 0).mean()), 4) if not active_daily.empty else None,
        "avg_trade_return": round(float(trade_returns.mean()), 6) if not trade_returns.empty else None,
        "trade_win_rate": round(float((trade_returns > 0).mean()), 4) if not trade_returns.empty else None,
        "target_hit_rate": round(float((trade_returns >= 0.03).mean()), 4) if not trade_returns.empty else None,
    }


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
    curve["equity"] = (1.0 + curve["avg_return"].fillna(0.0)).cumprod()
    curve["running_max"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["running_max"].replace(0.0, np.nan) - 1.0
    return curve, frame


def _yearly_metrics(label: str, rule_name: str, cost_bps: float, curve: pd.DataFrame, selected: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if curve.empty:
        return rows
    curve = curve.copy()
    selected = selected.copy()
    curve["year"] = curve["market_date"].dt.year
    selected["year"] = selected["market_date"].dt.year if not selected.empty else []
    for year, year_curve in curve.groupby("year", sort=True):
        year_selected = selected.loc[selected["year"].eq(year)].copy() if not selected.empty else selected
        row = _period_metrics(year_curve.drop(columns=["year"]), year_selected)
        row.update({"label": label, "rule": rule_name, "cost_bps": float(cost_bps), "year": int(year)})
        rows.append(row)
    return rows


def _market_state_trade_metrics(label: str, rule_name: str, cost_bps: float, selected: pd.DataFrame) -> list[dict[str, object]]:
    if selected.empty:
        return []
    rows: list[dict[str, object]] = []
    for column in ["bull_bear_state", "market_state"]:
        if column not in selected.columns:
            continue
        for state, frame in selected.groupby(column, dropna=False, sort=True):
            returns = frame["adjusted_return"].dropna()
            rows.append(
                {
                    "label": label,
                    "rule": rule_name,
                    "cost_bps": float(cost_bps),
                    "state_type": column,
                    "state": str(state),
                    "selected_rows": int(len(frame)),
                    "avg_trade_return": round(float(returns.mean()), 6) if not returns.empty else None,
                    "trade_win_rate": round(float((returns > 0).mean()), 4) if not returns.empty else None,
                    "target_hit_rate": round(float((returns >= 0.03).mean()), 4) if not returns.empty else None,
                }
            )
    return rows


def run_robustness(
    *,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    v3_dir: str | Path = DEFAULT_V3_DIR,
    bull_path: str | Path = DEFAULT_BULL_PATH,
    sse_path: str | Path = DEFAULT_SSE_PATH,
    output_dir: str | Path = DEFAULT_ROBUSTNESS_DIR,
    date_from: str = "2016-05-27",
    date_to: str = "2026-05-26",
    cost_bps_values: Iterable[float] = (0.0, 10.0, 20.0, 30.0),
) -> dict[str, object]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    source_path = Path(source_dir)
    calendar = _load_model_calendar(source_path / "model_scores.csv", date_from, date_to)
    candidates = _load_candidates(source_path, date_from, date_to)
    candidates = _enrich_candidates(candidates, Path(v3_dir), Path(bull_path), Path(sse_path))

    summary_rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []
    market_rows: list[dict[str, object]] = []
    selected_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []

    for candidate in DEFAULT_RULES:
        selected = _select_top(candidates, candidate.rule)
        for cost_bps in cost_bps_values:
            curve, cost_selected = _build_curve_with_cost(selected, calendar, float(cost_bps))
            row = _period_metrics(curve, cost_selected)
            row.update(
                {
                    "label": candidate.label,
                    "rule": candidate.rule.name,
                    "cost_bps": float(cost_bps),
                    "strategy_mode": candidate.rule.strategy_mode,
                    "market_filter": candidate.rule.market_filter,
                    "score_threshold": candidate.rule.score_threshold,
                    "priority_threshold": candidate.rule.priority_threshold,
                    "top_n": candidate.rule.top_n,
                    "sort_mode": candidate.rule.sort_mode,
                }
            )
            summary_rows.append(row)
            yearly_rows.extend(_yearly_metrics(candidate.label, candidate.rule.name, float(cost_bps), curve, cost_selected))
            market_rows.extend(_market_state_trade_metrics(candidate.label, candidate.rule.name, float(cost_bps), cost_selected))
            selected_frames.append(cost_selected.assign(label=candidate.label, rule=candidate.rule.name, cost_bps=float(cost_bps)))
            curve_frames.append(curve.assign(label=candidate.label, rule=candidate.rule.name, cost_bps=float(cost_bps)))

    summary = pd.DataFrame(summary_rows)
    yearly = pd.DataFrame(yearly_rows)
    market = pd.DataFrame(market_rows)
    summary["cost_adjusted_score"] = (
        summary["annualized_return"].fillna(0.0) * 100.0
        + summary["return_drawdown_ratio"].fillna(0.0) * 10.0
        + summary["active_daily_win_rate"].fillna(0.0) * 6.0
        - summary["max_drawdown"].abs().fillna(1.0) * 16.0
    )
    summary = summary.sort_values(
        ["cost_bps", "cost_adjusted_score", "return_drawdown_ratio", "annualized_return"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    summary.to_csv(output_path / "robustness_cost_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output_path / "robustness_yearly.csv", index=False, encoding="utf-8-sig")
    market.to_csv(output_path / "robustness_market_state_trades.csv", index=False, encoding="utf-8-sig")
    if selected_frames:
        pd.concat(selected_frames, ignore_index=True, sort=False).to_csv(
            output_path / "robustness_selected_samples.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if curve_frames:
        pd.concat(curve_frames, ignore_index=True, sort=False).to_csv(
            output_path / "robustness_curves.csv",
            index=False,
            encoding="utf-8-sig",
        )

    payload = {
        "date_from": date_from,
        "date_to": date_to,
        "calendar_days": int(len(calendar)),
        "candidate_rows": int(len(candidates)),
        "rules": [{"label": item.label, "rule": asdict(item.rule)} for item in DEFAULT_RULES],
        "cost_bps_values": [float(value) for value in cost_bps_values],
        "best_by_cost": summary.groupby("cost_bps", as_index=False).head(5).replace({np.nan: None}).to_dict("records"),
        "paths": {
            "summary": str(output_path / "robustness_cost_summary.csv"),
            "yearly": str(output_path / "robustness_yearly.csv"),
            "market_state_trades": str(output_path / "robustness_market_state_trades.csv"),
            "curves": str(output_path / "robustness_curves.csv"),
            "selected_samples": str(output_path / "robustness_selected_samples.csv"),
            "json": str(output_path / "robustness_summary.json"),
        },
    }
    (output_path / "robustness_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate shortlisted strategy combinations under cost and regime stress.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--v3-dir", default=str(DEFAULT_V3_DIR))
    parser.add_argument("--bull-path", default=str(DEFAULT_BULL_PATH))
    parser.add_argument("--sse-path", default=str(DEFAULT_SSE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_ROBUSTNESS_DIR))
    parser.add_argument("--date-from", default="2016-05-27")
    parser.add_argument("--date-to", default="2026-05-26")
    parser.add_argument("--cost-bps", default="0,10,20,30")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    cost_bps_values = [float(value.strip()) for value in str(args.cost_bps).split(",") if value.strip()]
    summary = run_robustness(
        source_dir=args.source_dir,
        v3_dir=args.v3_dir,
        bull_path=args.bull_path,
        sse_path=args.sse_path,
        output_dir=args.output_dir,
        date_from=args.date_from,
        date_to=args.date_to,
        cost_bps_values=cost_bps_values,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
