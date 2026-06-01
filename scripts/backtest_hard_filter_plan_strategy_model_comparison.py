from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from backtest_hard_filter_plan_comparison import (
    DEFAULT_BULL_PATH,
    DEFAULT_MODEL_SOURCE,
    DEFAULT_V3_DIR,
    Rule,
    enrich_candidates,
    evaluate_rules,
    load_old_combined_candidates,
    load_regime,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN_CANDIDATES_PATH = PROJECT_ROOT / ".cache" / "hard_filter_plan_comparison_full" / "plan_scored_candidates.csv"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / ".cache" / "hard_filter_plan_strategy_models" / "strategy_model_predictions.csv"
DEFAULT_MODEL_METRICS_PATH = PROJECT_ROOT / ".cache" / "hard_filter_plan_strategy_models" / "strategy_model_metrics.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".cache" / "hard_filter_plan_strategy_model_portfolio"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "docs" / "hard_filter_plan_strategy_model_backtest_comparison_2026-06-01.md"


def _json_default(value: object) -> object:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        if np.isnan(numeric) or np.isinf(numeric):
            return None
        return numeric
    if isinstance(value, Path):
        return str(value)
    return value


def _pct(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.2f}%"


def _num(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def normalize_symbol_series(values: pd.Series) -> pd.Series:
    extracted = values.astype(str).str.extract(r"(\d+)", expand=False).fillna("")
    return extracted.str[-6:].str.zfill(6)


def _weighted_available(frame: pd.DataFrame, weights: list[tuple[str, float]]) -> pd.Series:
    numerator = pd.Series(0.0, index=frame.index, dtype=float)
    denominator = pd.Series(0.0, index=frame.index, dtype=float)
    for column, weight in weights:
        values = _num(frame, column)
        available = values.notna()
        numerator = numerator + values.fillna(0.0) * float(weight)
        denominator = denominator + available.astype(float) * float(weight)
    return numerator / denominator.replace(0.0, np.nan)


def load_strategy_predictions(path: Path) -> tuple[pd.DataFrame, str, str]:
    predictions = pd.read_csv(path, encoding="utf-8-sig", dtype={"symbol": "string"}, parse_dates=["market_date"])
    if predictions.empty:
        raise RuntimeError(f"No strategy model predictions found at {path}")
    predictions["symbol"] = normalize_symbol_series(predictions["symbol"])
    predictions["market_date"] = pd.to_datetime(predictions["market_date"], errors="coerce").dt.normalize()
    predictions["strategy_family"] = predictions["strategy_family"].astype(str)
    predictions["horizon_days"] = pd.to_numeric(predictions["horizon_days"], errors="coerce").astype("Int64")
    predictions["model_score"] = pd.to_numeric(predictions["model_score"], errors="coerce")
    predictions["calibrated_probability"] = pd.to_numeric(predictions["calibrated_probability"], errors="coerce")

    wide = predictions.pivot_table(
        index=["market_date", "symbol", "strategy_family"],
        columns="horizon_days",
        values=["model_score", "calibrated_probability"],
        aggfunc="last",
    )
    wide.columns = [
        f"{'score' if metric == 'model_score' else 'prob'}_{int(horizon)}d"
        for metric, horizon in wide.columns.to_flat_index()
    ]
    wide = wide.reset_index()
    start = pd.to_datetime(predictions["market_date"].min()).strftime("%Y-%m-%d")
    end = pd.to_datetime(predictions["market_date"].max()).strftime("%Y-%m-%d")
    return wide, start, end


def load_plan_with_new_algorithm(
    plan_candidates_path: Path,
    predictions_path: Path,
    regime: pd.DataFrame,
) -> pd.DataFrame:
    plan = pd.read_csv(plan_candidates_path, encoding="utf-8-sig", dtype={"symbol": "string"}, parse_dates=["market_date"])
    if plan.empty:
        return plan
    plan["symbol"] = normalize_symbol_series(plan["symbol"])
    plan["market_date"] = pd.to_datetime(plan["market_date"], errors="coerce").dt.normalize()
    if "strategy_family" not in plan.columns:
        plan = enrich_candidates(plan, regime)
    else:
        plan["strategy_family"] = plan["strategy_family"].astype(str)
    plan = plan.dropna(subset=["market_date", "symbol"]).copy()

    predictions, test_start, test_end = load_strategy_predictions(predictions_path)
    plan = plan.loc[plan["market_date"].between(pd.to_datetime(test_start), pd.to_datetime(test_end), inclusive="both")].copy()
    merged = plan.merge(predictions, on=["market_date", "symbol", "strategy_family"], how="inner")
    if merged.empty:
        raise RuntimeError("No overlap between plan candidates and strategy model predictions.")

    merged = merged.rename(
        columns={
            "model_score": "legacy_model_score",
            "model_probability": "legacy_model_probability",
            "model_priority_80_20": "legacy_model_priority_80_20",
        }
    )
    s1 = merged["strategy_family"].eq("strategy1")
    s2 = merged["strategy_family"].eq("strategy2")
    s3 = merged["strategy_family"].eq("strategy3")
    merged["new_algo_score"] = np.nan
    merged.loc[s1, "new_algo_score"] = _weighted_available(merged.loc[s1], [("score_3d", 0.65), ("score_5d", 0.35)])
    merged.loc[s2, "new_algo_score"] = _weighted_available(merged.loc[s2], [("score_3d", 0.60), ("score_1d", 0.40)])
    merged.loc[s3, "new_algo_score"] = _weighted_available(merged.loc[s3], [("score_3d", 0.65), ("score_5d", 0.35)])
    merged["new_algo_score"] = pd.to_numeric(merged["new_algo_score"], errors="coerce")
    merged = merged.dropna(subset=["new_algo_score", "hold_3d_return"]).copy()
    merged["model_score"] = merged["new_algo_score"].clip(0.0, 100.0)
    merged["model_probability"] = merged["model_score"] / 100.0
    merged["priority_score"] = pd.to_numeric(merged.get("candidate_priority"), errors="coerce").fillna(
        pd.to_numeric(merged.get("priority_score"), errors="coerce")
    )
    merged["model_priority_80_20"] = merged["model_score"] * 0.80 + merged["priority_score"].fillna(0.0).clip(0.0, 100.0) * 0.20
    regime_columns = [column for column in regime.columns if column != "market_date"]
    merged = merged.drop(columns=[column for column in regime_columns if column in merged.columns], errors="ignore")
    return enrich_candidates(merged, regime)


def build_old_rules() -> list[Rule]:
    rules: list[Rule] = []
    for top_n in (1, 3):
        for strategy_mode in ("all", "strategy1", "strategy2", "strategy3"):
            for market_filter in ("none", "v3_full_green", "bull7_v3_full_green", "bull6_v3_full_green", "market_green", "bull7"):
                for score_threshold in (66.0, 68.0, 70.0):
                    for priority_threshold in (None, 60.0, 65.0):
                        for sort_mode in ("model_score", "model_priority_80_20"):
                            rules.append(
                                Rule(
                                    strategy_mode=strategy_mode,
                                    market_filter=market_filter,
                                    score_threshold=score_threshold,
                                    priority_threshold=priority_threshold,
                                    top_n=top_n,
                                    sort_mode=sort_mode,
                                )
                            )
    return rules


def build_new_rules() -> list[Rule]:
    rules: list[Rule] = []
    for top_n in (1, 3):
        for strategy_mode in ("all", "strategy1", "strategy2", "strategy3"):
            for market_filter in ("none", "v3_full_green", "bull7_v3_full_green", "bull6_v3_full_green", "market_green", "bull7"):
                for score_threshold in (25.0, 30.0, 32.5, 35.0, 37.5, 40.0, 45.0, 50.0):
                    for priority_threshold in (None, 60.0, 65.0, 70.0):
                        for sort_mode in ("model_score", "model_priority_80_20"):
                            rules.append(
                                Rule(
                                    strategy_mode=strategy_mode,
                                    market_filter=market_filter,
                                    score_threshold=score_threshold,
                                    priority_threshold=priority_threshold,
                                    top_n=top_n,
                                    sort_mode=sort_mode,
                                )
                            )
    return rules


def _best_by_source_cost(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return comparison
    ordered = comparison.sort_values(
        ["source", "cost_bps", "top_n", "cost_adjusted_score", "return_drawdown_ratio", "annualized_return"],
        ascending=[True, True, True, False, False, False],
    )
    return ordered.groupby(["source", "cost_bps", "top_n"], as_index=False).head(1).reset_index(drop=True)


def _top_rules(frame: pd.DataFrame, source: str, cost_bps: float, top_n: int, n: int = 8) -> pd.DataFrame:
    local = frame.loc[
        frame["source"].eq(source)
        & pd.to_numeric(frame["cost_bps"], errors="coerce").eq(float(cost_bps))
        & pd.to_numeric(frame["top_n"], errors="coerce").eq(int(top_n))
    ].copy()
    if local.empty:
        return local
    return local.sort_values(["cost_adjusted_score", "return_drawdown_ratio", "annualized_return"], ascending=[False, False, False]).head(n)


def _append_metric_table(rows: list[str], title: str, frame: pd.DataFrame) -> None:
    rows.extend(["", f"## {title}", ""])
    if frame.empty:
        rows.append("No rows.")
        return
    rows.append("| source | top_n | cost | rule | active_days | annualized | max_dd | trade_win | avg_trade | target_hit |")
    rows.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in frame.to_dict("records"):
        rows.append(
            "| "
            + " | ".join(
                [
                    str(row.get("source", "")),
                    str(int(float(row.get("top_n", 0) or 0))),
                    f"{float(row.get('cost_bps', 0.0) or 0.0):.0f}bp",
                    str(row.get("rule", "")),
                    str(int(float(row.get("active_days", 0) or 0))),
                    _pct(row.get("annualized_return")),
                    _pct(row.get("max_drawdown")),
                    _pct(row.get("trade_win_rate")),
                    _pct(row.get("avg_trade_return")),
                    _pct(row.get("target_hit_rate")),
                ]
            )
            + " |"
        )


def write_report(
    *,
    report_path: Path,
    output_dir: Path,
    metadata: dict[str, object],
    model_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    best: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "# Hard Filter Plan Strategy Model Portfolio Comparison",
        "",
        "## Scope",
        "",
        f"- OOS window: `{metadata['date_from']}` to `{metadata['date_to']}`.",
        f"- New strategy: P1 hard-filter/soft-score proxy from `{metadata['strategy_plan']}`.",
        "- New algorithm: P2 per-strategy calibrated model score. Strategy1/3 use 3d+5d; strategy2 uses 1d+3d; portfolio return is still measured by 3-day hold.",
        "- Baseline: original candidate strategies plus original ten-year model score, cut to the same OOS dates.",
        "- Search grid is fixed in this script; new model score thresholds use calibrated probability scale, so they are not numerically comparable with the old 66/68/70 score thresholds.",
        f"- Artifacts: `{output_dir}`.",
        "",
        "## Model Accuracy Snapshot",
        "",
        "| strategy | horizon | test | AUC | Brier | avg_return | top_bucket_return | top_bucket_win_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in model_metrics.sort_values(["strategy_family", "horizon_days"]).to_dict("records"):
        rows.append(
            "| "
            + " | ".join(
                [
                    str(row.get("strategy_family", "")),
                    str(int(row.get("horizon_days", 0) or 0)),
                    str(int(row.get("test_sample_count", 0) or 0)),
                    f"{float(row.get('auc', float('nan'))):.3f}" if not pd.isna(row.get("auc")) else "",
                    f"{float(row.get('brier', float('nan'))):.4f}" if not pd.isna(row.get("brier")) else "",
                    _pct(row.get("avg_return")),
                    _pct(row.get("top_bucket_return")),
                    _pct(row.get("top_bucket_win_rate")),
                ]
            )
            + " |"
        )

    _append_metric_table(rows, "Best By Source Cost And TopN", best)
    for top_n in (3, 1):
        rows.extend(["", f"## 20bp Top{top_n} Leaderboard", ""])
        for source in ("old_strategy_old_algo_oos", "new_strategy_new_algo_oos"):
            top = _top_rules(comparison, source=source, cost_bps=20.0, top_n=top_n, n=5)
            rows.append(f"### {source}")
            if top.empty:
                rows.append("No rows.")
                continue
            rows.append("| rule | active_days | annualized | max_dd | trade_win | avg_trade | target_hit |")
            rows.append("|---|---:|---:|---:|---:|---:|---:|")
            for row in top.to_dict("records"):
                rows.append(
                    "| "
                    + " | ".join(
                        [
                            str(row.get("rule", "")),
                            str(int(float(row.get("active_days", 0) or 0))),
                            _pct(row.get("annualized_return")),
                            _pct(row.get("max_drawdown")),
                            _pct(row.get("trade_win_rate")),
                            _pct(row.get("avg_trade_return")),
                            _pct(row.get("target_hit_rate")),
                        ]
                    )
                    + " |"
                )
            rows.append("")

    rows.extend(
        [
            "## Files",
            "",
            f"- Comparison CSV: `{output_dir / 'portfolio_rule_comparison.csv'}`",
            f"- Best summary CSV: `{output_dir / 'portfolio_best_by_source_cost.csv'}`",
            f"- New strategy candidates with new scores: `{output_dir / 'new_strategy_new_algo_candidates.csv'}`",
            f"- Summary JSON: `{output_dir / 'summary.json'}`",
        ]
    )
    report_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def run_portfolio_comparison(
    *,
    plan_candidates_path: Path,
    predictions_path: Path,
    model_metrics_path: Path,
    old_model_source: Path,
    v3_dir: Path,
    bull_path: Path,
    output_dir: Path,
    report_path: Path,
    cost_bps_values: Iterable[float],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_index, date_from, date_to = load_strategy_predictions(predictions_path)
    del prediction_index
    regime = load_regime(v3_dir, bull_path, date_from, date_to)
    calendar = regime[["market_date"]].drop_duplicates().sort_values("market_date").reset_index(drop=True)
    print(f"[scope] {date_from} -> {date_to} calendar_days={len(calendar)}", flush=True)

    old_candidates = load_old_combined_candidates(old_model_source, date_from, date_to)
    old_candidates = enrich_candidates(old_candidates, regime)
    print(f"[old] candidates={len(old_candidates)}", flush=True)

    new_candidates = load_plan_with_new_algorithm(plan_candidates_path, predictions_path, regime)
    print(f"[new] candidates={len(new_candidates)}", flush=True)
    new_candidates_path = output_dir / "new_strategy_new_algo_candidates.csv"
    new_candidates.to_csv(new_candidates_path, index=False, encoding="utf-8-sig")

    old_summary, old_samples = evaluate_rules(
        old_candidates,
        calendar,
        build_old_rules(),
        source="old_strategy_old_algo_oos",
        cost_bps_values=cost_bps_values,
    )
    new_summary, new_samples = evaluate_rules(
        new_candidates,
        calendar,
        build_new_rules(),
        source="new_strategy_new_algo_oos",
        cost_bps_values=cost_bps_values,
    )
    comparison = pd.concat([old_summary, new_summary], ignore_index=True, sort=False)
    best = _best_by_source_cost(comparison)

    comparison_path = output_dir / "portfolio_rule_comparison.csv"
    best_path = output_dir / "portfolio_best_by_source_cost.csv"
    samples_path = output_dir / "portfolio_selected_samples.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    best.to_csv(best_path, index=False, encoding="utf-8-sig")
    samples = pd.concat([old_samples, new_samples], ignore_index=True, sort=False) if not old_samples.empty or not new_samples.empty else pd.DataFrame()
    samples.to_csv(samples_path, index=False, encoding="utf-8-sig")

    model_metrics = pd.read_csv(model_metrics_path, encoding="utf-8-sig")
    metadata = {
        "date_from": date_from,
        "date_to": date_to,
        "strategy_plan": str(PROJECT_ROOT / "docs" / "strategy_hard_filter_optimization_plan_2026-05-31.md"),
        "old_candidate_rows": int(len(old_candidates)),
        "new_candidate_rows": int(len(new_candidates)),
        "calendar_days": int(len(calendar)),
        "comparison_path": str(comparison_path),
        "best_path": str(best_path),
        "selected_samples_path": str(samples_path),
        "new_candidates_path": str(new_candidates_path),
        "report_path": str(report_path),
        "best_by_source_cost": best.to_dict("records"),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    write_report(
        report_path=report_path,
        output_dir=output_dir,
        metadata=metadata,
        model_metrics=model_metrics,
        comparison=comparison,
        best=best,
    )
    return metadata


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare original strategy/model with the hard-filter plan strategy-specific model portfolio.")
    parser.add_argument("--plan-candidates-path", default=str(DEFAULT_PLAN_CANDIDATES_PATH))
    parser.add_argument("--predictions-path", default=str(DEFAULT_PREDICTIONS_PATH))
    parser.add_argument("--model-metrics-path", default=str(DEFAULT_MODEL_METRICS_PATH))
    parser.add_argument("--old-model-source", default=str(DEFAULT_MODEL_SOURCE))
    parser.add_argument("--v3-dir", default=str(DEFAULT_V3_DIR))
    parser.add_argument("--bull-path", default=str(DEFAULT_BULL_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--cost-bps", default="0,10,20,30")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    payload = run_portfolio_comparison(
        plan_candidates_path=Path(args.plan_candidates_path),
        predictions_path=Path(args.predictions_path),
        model_metrics_path=Path(args.model_metrics_path),
        old_model_source=Path(args.old_model_source),
        v3_dir=Path(args.v3_dir),
        bull_path=Path(args.bull_path),
        output_dir=Path(args.output_dir),
        report_path=Path(args.report_path),
        cost_bps_values=[float(value) for value in str(args.cost_bps).split(",") if value.strip()],
    )
    print(
        json.dumps(
            {
                "date_from": payload["date_from"],
                "date_to": payload["date_to"],
                "old_candidate_rows": payload["old_candidate_rows"],
                "new_candidate_rows": payload["new_candidate_rows"],
                "comparison_path": payload["comparison_path"],
                "report_path": payload["report_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
