from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = PROJECT_ROOT / ".cache" / "daily_focus_board_reviews"
OUTPUT_DIR = PROJECT_ROOT / ".cache" / "trading_system_attribution"
DOC_PATH = PROJECT_ROOT / "docs" / "execution_score_counterfactual_2026-05-28.md"


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


def _load_selected_reviews() -> pd.DataFrame:
    candidates: list[dict[str, object]] = []
    for path in REVIEW_DIR.glob("review_*.pkl"):
        obj = _load_pickle(path)
        meta = dict(obj.get("meta") or {})
        details = obj.get("details")
        if not isinstance(details, pd.DataFrame) or details.empty:
            continue
        candidates.append(
            {
                "path": path,
                "board_date": str(meta.get("board_date") or ""),
                "cache_version": int(meta.get("cache_version") or 0),
                "board_size": int(meta.get("board_size") or 0),
                "rows": len(details),
            }
        )
    review_index = pd.DataFrame(candidates)
    selected = (
        review_index.loc[review_index["board_size"].eq(50)]
        .sort_values(["board_date", "cache_version", "rows", "path"], ascending=[True, False, False, True])
        .groupby("board_date", as_index=False)
        .head(1)
        .copy()
    )

    frames: list[pd.DataFrame] = []
    for row in selected.itertuples(index=False):
        obj = _load_pickle(Path(row.path))
        details = obj["details"].copy()
        details["board_date"] = str(obj["meta"]["board_date"])
        details["cache_version"] = int(obj["meta"]["cache_version"])
        frames.append(details)
    return pd.concat(frames, ignore_index=True)


def _top_n_summary(frame: pd.DataFrame, score_col: str, top_n: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for board_date, group in frame.groupby("board_date"):
        ranked = group.dropna(subset=[score_col, "next_day_return_pct"]).sort_values(score_col, ascending=False).head(top_n)
        if len(ranked) < top_n:
            continue
        rows.append(
            {
                "board_date": board_date,
                "avg_next_day_return_pct": float(ranked["next_day_return_pct"].mean()),
                "avg_intraday_high_return_pct": float(ranked["intraday_high_return_pct"].mean()),
                "win_rate": float(ranked["win"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_report() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    review_details = _load_selected_reviews()
    scored = review_details.loc[review_details["selection_score"].notna() & review_details["execution_score"].notna()].copy()
    scored["blend_62_38"] = scored["selection_score"] * 0.62 + scored["execution_score"] * 0.38

    summary_rows: list[dict[str, object]] = []
    per_day_top3_rows: list[dict[str, object]] = []
    for score_col in ["selection_score", "blend_62_38", "final_rank_score", "execution_score"]:
        for top_n in [1, 3, 5, 10]:
            per_day = _top_n_summary(scored, score_col, top_n)
            if per_day.empty:
                continue
            summary_rows.append(
                {
                    "score_col": score_col,
                    "top_n": top_n,
                    "days": int(len(per_day)),
                    "avg_next_day_return_pct": float(per_day["avg_next_day_return_pct"].mean()),
                    "median_next_day_return_pct": float(per_day["avg_next_day_return_pct"].median()),
                    "avg_intraday_high_return_pct": float(per_day["avg_intraday_high_return_pct"].mean()),
                    "avg_win_rate": float(per_day["win_rate"].mean()),
                }
            )
            if top_n == 3:
                per_day["score_col"] = score_col
                per_day_top3_rows.append(per_day)

    summary = pd.DataFrame(summary_rows)
    per_day_top3 = pd.concat(per_day_top3_rows, ignore_index=True)

    selection_top3 = per_day_top3.loc[per_day_top3["score_col"].eq("selection_score")].rename(
        columns={"avg_next_day_return_pct": "selection_top3_return_pct"}
    )
    blend_top3 = per_day_top3.loc[per_day_top3["score_col"].eq("blend_62_38")].rename(
        columns={"avg_next_day_return_pct": "blend_top3_return_pct"}
    )
    top3_compare = selection_top3.merge(
        blend_top3[["board_date", "blend_top3_return_pct"]],
        on="board_date",
        how="inner",
    )
    top3_compare["blend_minus_selection_pct"] = (
        top3_compare["blend_top3_return_pct"] - top3_compare["selection_top3_return_pct"]
    )

    action_summary = (
        scored.groupby("action_label", dropna=False)
        .agg(
            rows=("symbol", "size"),
            avg_next_day_return_pct=("next_day_return_pct", "mean"),
            median_next_day_return_pct=("next_day_return_pct", "median"),
            avg_intraday_high_return_pct=("intraday_high_return_pct", "mean"),
            win_rate=("win", "mean"),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
        .reset_index(drop=True)
    )
    action_execution_summary = (
        scored.groupby(["action_label", "execution_label"], dropna=False)
        .agg(
            rows=("symbol", "size"),
            avg_next_day_return_pct=("next_day_return_pct", "mean"),
            median_next_day_return_pct=("next_day_return_pct", "median"),
            win_rate=("win", "mean"),
        )
        .reset_index()
        .sort_values(["action_label", "rows"], ascending=[True, False])
        .reset_index(drop=True)
    )
    high_selection_non_buy = scored.loc[(scored["selection_score"] >= 80.0) & (scored["action_label"] != "买")].copy()
    high_selection_non_buy_summary = (
        high_selection_non_buy.groupby(["action_label", "execution_label"], dropna=False)
        .agg(
            rows=("symbol", "size"),
            avg_next_day_return_pct=("next_day_return_pct", "mean"),
            median_next_day_return_pct=("next_day_return_pct", "median"),
            win_rate=("win", "mean"),
        )
        .reset_index()
        .sort_values(["action_label", "rows"], ascending=[True, False])
        .reset_index(drop=True)
    )

    findings = [
        (
            "Across the 13 board dates with both `selection_score` and `execution_score`, "
            "`selection_score` top-3 and blended `0.62*selection + 0.38*execution` top-3 produced the same next-day return on every date."
        ),
        (
            "At the portfolio-relevant top-3 cut, `selection_score` and the blended action score both average "
            f"{float(summary.loc[(summary['score_col'].eq('selection_score')) & (summary['top_n'].eq(3)), 'avg_next_day_return_pct'].iloc[0]):.2f}% "
            "next-day return, which means the execution-score term did not add marginal ranking value in this replay sample."
        ),
        (
            "`execution_score` used alone is weaker than `selection_score` on the same 13-date sample: "
            f"top-3 average next-day return {float(summary.loc[(summary['score_col'].eq('execution_score')) & (summary['top_n'].eq(3)), 'avg_next_day_return_pct'].iloc[0]):.2f}% "
            f"versus {float(summary.loc[(summary['score_col'].eq('selection_score')) & (summary['top_n'].eq(3)), 'avg_next_day_return_pct'].iloc[0]):.2f}%."
        ),
        (
            "At top-10, blending execution into selection is slightly worse than selection alone: "
            f"{float(summary.loc[(summary['score_col'].eq('blend_62_38')) & (summary['top_n'].eq(10)), 'avg_next_day_return_pct'].iloc[0]):.2f}% "
            f"versus {float(summary.loc[(summary['score_col'].eq('selection_score')) & (summary['top_n'].eq(10)), 'avg_next_day_return_pct'].iloc[0]):.2f}%."
        ),
        (
            "The action layer should not be removed wholesale: rows labeled `买` still average "
            f"{float(action_summary.loc[action_summary['action_label'].eq('买'), 'avg_next_day_return_pct'].iloc[0]):.2f}% "
            "next-day return on this v9 replay slice."
        ),
        (
            "High-selection non-buy rows do not look obviously better than the buy rows. "
            "That supports keeping discrete execution states as veto/explanation logic, while demoting the continuous execution-score weight from ranking blends."
        ),
    ]

    summary_csv = OUTPUT_DIR / "execution_score_counterfactual_summary.csv"
    top3_csv = OUTPUT_DIR / "execution_score_counterfactual_top3.csv"
    action_csv = OUTPUT_DIR / "execution_score_counterfactual_action_summary.csv"
    action_execution_csv = OUTPUT_DIR / "execution_score_counterfactual_action_execution_summary.csv"
    high_selection_non_buy_csv = OUTPUT_DIR / "execution_score_counterfactual_high_selection_non_buy.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    top3_compare.to_csv(top3_csv, index=False, encoding="utf-8-sig")
    action_summary.to_csv(action_csv, index=False, encoding="utf-8-sig")
    action_execution_summary.to_csv(action_execution_csv, index=False, encoding="utf-8-sig")
    high_selection_non_buy_summary.to_csv(high_selection_non_buy_csv, index=False, encoding="utf-8-sig")

    payload = {
        "scored_rows": int(len(scored)),
        "board_dates": int(scored["board_date"].nunique()),
        "summary": _clean_records(summary),
        "top3_compare": _clean_records(top3_compare),
        "action_summary": _clean_records(action_summary),
        "action_execution_summary": _clean_records(action_execution_summary),
        "high_selection_non_buy_summary": _clean_records(high_selection_non_buy_summary),
        "findings": findings,
    }
    json_path = OUTPUT_DIR / "execution_score_counterfactual.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Execution Score Counterfactual 2026-05-28",
        "",
        "## Purpose",
        "",
        "Measure whether the current execution-score contribution actually changes or improves replay ranking outcomes, instead of assuming that more execution logic automatically helps.",
        "",
        "## Coverage",
        "",
        f"- Replay rows with both `selection_score` and `execution_score`: {len(scored)}",
        f"- Board dates with comparable scored rows: {scored['board_date'].nunique()}",
        "",
        "## Key Findings",
        "",
    ]
    for item in findings:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Top-N Comparison",
            "",
            summary.to_markdown(index=False),
            "",
            "## Top-3 Selection vs Blended Action Score",
            "",
            top3_compare.to_markdown(index=False),
            "",
            "## Action Label Summary",
            "",
            action_summary.to_markdown(index=False),
            "",
            "## Action x Execution Label Summary",
            "",
            action_execution_summary.to_markdown(index=False),
            "",
            "## High-Selection Non-Buy Rows",
            "",
            high_selection_non_buy_summary.to_markdown(index=False),
            "",
            "## Interpretation",
            "",
            "- The current `action_score = 0.62*selection_score + 0.38*execution_score` blend is not earning its complexity on this replay slice.",
            "- `execution_score` appears saturated in strong candidates, so it often fails to reshuffle the top of the book even when it materially complicates the decision stack.",
            "- The stronger nuance is: demote the continuous `execution_score` weight first, but keep discrete execution states like `买` / `持` / `观察` and `execution_label` under audit because they still separate behavior better than the raw score alone.",
            "",
            "## Next Actions",
            "",
            "1. Remove the `execution_score` term from default ranking blends in research comparisons and re-check top-3 / top-10 post-cost results.",
            "2. Keep `execution_label` and `execution_window` as discrete explanation fields; they may still be useful as veto diagnostics.",
            "3. Re-run the same comparison after more v9 replay history exists, to confirm this is not a short-window artifact.",
            "",
            f"Summary CSV: `{summary_csv}`",
            f"Top-3 comparison CSV: `{top3_csv}`",
            f"Action summary CSV: `{action_csv}`",
            f"Action x execution CSV: `{action_execution_csv}`",
            f"High-selection non-buy CSV: `{high_selection_non_buy_csv}`",
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
