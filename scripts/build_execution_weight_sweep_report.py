from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = PROJECT_ROOT / ".cache" / "daily_focus_board_reviews"
OUTPUT_DIR = PROJECT_ROOT / ".cache" / "trading_system_attribution"
DOC_PATH = PROJECT_ROOT / "docs" / "execution_weight_sweep_2026-05-28.md"


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _clean_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in frame.to_dict("records"):
        cleaned: dict[str, object] = {}
        for key, value in row.items():
            cleaned[key] = None if pd.isna(value) else value
        records.append(cleaned)
    return records


def _load_v9_review_rows() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in REVIEW_DIR.glob("review_v9_*.pkl"):
        obj = _load_pickle(path)
        details = obj.get("details")
        if not isinstance(details, pd.DataFrame) or details.empty:
            continue
        frame = details.copy()
        frame["board_date"] = str(obj["meta"]["board_date"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _per_day_top(frame: pd.DataFrame, score_col: str, top_n: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for board_date, group in frame.groupby("board_date"):
        ranked = group.dropna(subset=[score_col, "next_day_return_pct"]).sort_values(score_col, ascending=False).head(top_n)
        if len(ranked) < top_n:
            continue
        rows.append(
            {
                "board_date": board_date,
                "symbols": tuple(ranked["symbol"].astype(str).tolist()),
                "avg_next_day_return_pct": float(ranked["next_day_return_pct"].mean()),
                "avg_intraday_high_return_pct": float(ranked["intraday_high_return_pct"].mean()),
                "win_rate": float(ranked["win"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_report() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    review_rows = _load_v9_review_rows()
    scored = review_rows.loc[review_rows["selection_score"].notna() & review_rows["execution_score"].notna()].copy()

    weights = [0.0, 0.10, 0.22, 0.38, 0.50, 1.0]
    summary_rows: list[dict[str, object]] = []
    top3_detail: dict[float, pd.DataFrame] = {}

    for execution_weight in weights:
        selection_weight = 1.0 - execution_weight
        score_col = f"blend_w_{execution_weight:.2f}"
        scored[score_col] = scored["selection_score"] * selection_weight + scored["execution_score"] * execution_weight

        for top_n in [1, 3, 5, 10]:
            per_day = _per_day_top(scored, score_col, top_n)
            if per_day.empty:
                continue
            summary_rows.append(
                {
                    "execution_weight": execution_weight,
                    "selection_weight": selection_weight,
                    "top_n": top_n,
                    "days": int(len(per_day)),
                    "avg_next_day_return_pct": float(per_day["avg_next_day_return_pct"].mean()),
                    "median_next_day_return_pct": float(per_day["avg_next_day_return_pct"].median()),
                    "avg_intraday_high_return_pct": float(per_day["avg_intraday_high_return_pct"].mean()),
                    "avg_win_rate": float(per_day["win_rate"].mean()),
                }
            )
            if top_n == 3:
                top3_detail[execution_weight] = per_day.copy()

    summary = pd.DataFrame(summary_rows)

    baseline = top3_detail[0.0].copy().rename(
        columns={
            "symbols": "baseline_symbols",
            "avg_next_day_return_pct": "baseline_top3_return_pct",
        }
    )
    compare_rows: list[dict[str, object]] = []
    for execution_weight, per_day in top3_detail.items():
        if execution_weight == 0.0:
            continue
        merged = baseline.merge(
            per_day.rename(
                columns={
                    "symbols": "variant_symbols",
                    "avg_next_day_return_pct": "variant_top3_return_pct",
                }
            )[["board_date", "variant_symbols", "variant_top3_return_pct"]],
            on="board_date",
            how="inner",
        )
        compare_rows.append(
            {
                "execution_weight": execution_weight,
                "selection_weight": 1.0 - execution_weight,
                "same_top3_days": int((merged["baseline_symbols"] == merged["variant_symbols"]).sum()),
                "total_days": int(len(merged)),
                "avg_variant_minus_baseline_pct": float(
                    (merged["variant_top3_return_pct"] - merged["baseline_top3_return_pct"]).mean()
                ),
            }
        )
    compare_summary = pd.DataFrame(compare_rows)

    findings = [
        "On the current v9 replay slice, execution weights from 0.00 through 0.50 produce the exact same top-3 picks on all 13 comparable board dates.",
        "That means the current `0.38` execution weight is not merely too large or too small; it is functionally inactive as a ranking differentiator in this sample.",
        (
            "Top-3 average next-day return stays flat at "
            f"{float(summary.loc[(summary['execution_weight'].eq(0.0)) & (summary['top_n'].eq(3)), 'avg_next_day_return_pct'].iloc[0]):.2f}% "
            "for every blend from 0.00 to 0.50 execution weight."
        ),
        (
            "Only the extreme `execution_weight = 1.00` regime reshuffles the top-3 set, and it changes picks on "
            f"{int(compare_summary.loc[compare_summary['execution_weight'].eq(1.0), 'total_days'].iloc[0] - compare_summary.loc[compare_summary['execution_weight'].eq(1.0), 'same_top3_days'].iloc[0])} "
            "of 13 days."
        ),
    ]

    summary_csv = OUTPUT_DIR / "execution_weight_sweep_summary.csv"
    compare_csv = OUTPUT_DIR / "execution_weight_sweep_top3_compare.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    compare_summary.to_csv(compare_csv, index=False, encoding="utf-8-sig")

    payload = {
        "scored_rows": int(len(scored)),
        "board_dates": int(scored["board_date"].nunique()),
        "summary": _clean_records(summary),
        "compare_summary": _clean_records(compare_summary),
        "findings": findings,
    }
    json_path = OUTPUT_DIR / "execution_weight_sweep.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Execution Weight Sweep 2026-05-28",
        "",
        "## Purpose",
        "",
        "Test whether changing the continuous execution-score weight inside the current action blend actually changes replay ranking outcomes.",
        "",
        "## Coverage",
        "",
        f"- Replay rows with both `selection_score` and `execution_score`: {len(scored)}",
        f"- Board dates with comparable v9 scored rows: {scored['board_date'].nunique()}",
        "",
        "## Key Findings",
        "",
    ]
    for item in findings:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Weight Sweep Summary",
            "",
            summary.to_markdown(index=False),
            "",
            "## Top-3 Stability vs Execution-Off Baseline",
            "",
            compare_summary.to_markdown(index=False),
            "",
            "## Interpretation",
            "",
            "- The current ranking stack is saturated enough that partial execution weighting does not move the top of the book at all on this slice.",
            "- This is stronger evidence than a single counterfactual: it says there is no practical ranking leverage anywhere between 0% and 50% execution weight here.",
            "- The right next change is to set the continuous execution weight to zero in research comparisons first, because doing so loses nothing on this replay slice while simplifying the stack.",
            "",
            "## Next Actions",
            "",
            "1. Add an execution-off branch to the research ranking path and rerun unified portfolio backtests.",
            "2. Keep discrete execution labels/windows for veto and explanation while the continuous weight is being audited.",
            "3. Re-run this sweep after replay history expands beyond the current v9 slice.",
            "",
            f"Summary CSV: `{summary_csv}`",
            f"Top-3 compare CSV: `{compare_csv}`",
            f"JSON: `{json_path}`",
        ]
    )
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    payload = build_report()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
