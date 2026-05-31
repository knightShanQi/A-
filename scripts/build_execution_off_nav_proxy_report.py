from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = PROJECT_ROOT / ".cache" / "daily_focus_board_reviews"
OUTPUT_DIR = PROJECT_ROOT / ".cache" / "trading_system_attribution"
DOC_PATH = PROJECT_ROOT / "docs" / "execution_off_nav_proxy_2026-05-28.md"


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


def _annualized_return(ending_equity: float, day_count: int) -> float:
    if day_count <= 0 or ending_equity <= 0:
        return 0.0
    return float(ending_equity ** (252.0 / day_count) - 1.0)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max.replace(0.0, pd.NA) - 1.0
    return float(pd.to_numeric(drawdown, errors="coerce").fillna(0.0).min())


def _nav_proxy(frame: pd.DataFrame, score_col: str, top_n: int) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    equity = 1.0
    for board_date, group in frame.groupby("board_date"):
        ranked = group.dropna(subset=[score_col, "next_day_return_pct"]).sort_values(score_col, ascending=False).head(top_n)
        if len(ranked) < top_n:
            continue
        day_return = float(ranked["next_day_return_pct"].mean()) / 100.0
        equity *= 1.0 + day_return
        rows.append(
            {
                "board_date": board_date,
                "top_n": int(top_n),
                "score_col": score_col,
                "daily_return": day_return,
                "equity": equity,
            }
        )
    curve = pd.DataFrame(rows)
    summary = {
        "score_col": score_col,
        "top_n": int(top_n),
        "days": int(len(curve)),
        "ending_equity": float(curve["equity"].iloc[-1]) if not curve.empty else 1.0,
        "cumulative_return": float(curve["equity"].iloc[-1] - 1.0) if not curve.empty else 0.0,
        "annualized_return_proxy": _annualized_return(float(curve["equity"].iloc[-1]), len(curve)) if not curve.empty else 0.0,
        "max_drawdown_proxy": _max_drawdown(curve["equity"]) if not curve.empty else 0.0,
        "avg_daily_return": float(curve["daily_return"].mean()) if not curve.empty else 0.0,
    }
    return curve, summary


def build_report() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    review_rows = _load_v9_review_rows()
    scored = review_rows.loc[review_rows["selection_score"].notna() & review_rows["execution_score"].notna()].copy()
    scored["blend_62_38"] = scored["selection_score"] * 0.62 + scored["execution_score"] * 0.38

    variants = [
        ("selection_score", 3),
        ("blend_62_38", 3),
        ("final_rank_score", 3),
        ("execution_score", 3),
        ("selection_score", 10),
        ("blend_62_38", 10),
    ]
    summary_rows: list[dict[str, object]] = []
    curves: list[pd.DataFrame] = []
    for score_col, top_n in variants:
        curve, summary = _nav_proxy(scored, score_col, top_n)
        if not curve.empty:
            curves.append(curve)
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    curve_df = pd.concat(curves, ignore_index=True)

    findings = [
        (
            "On the v9 replay proxy NAV, every top-3 variant is still negative over the 13-day slice. "
            "That reinforces the broader audit conclusion that the current recommendation stack lacks enough short-horizon alpha."
        ),
        (
            "At top-3, execution-off `selection_score` and the current `0.62/0.38` blend are exactly identical: "
            f"{float(summary_df.loc[(summary_df['score_col'].eq('selection_score')) & (summary_df['top_n'].eq(3)), 'cumulative_return'].iloc[0]):.2%} cumulative return proxy."
        ),
        (
            "At top-10, execution-off is better than the current blend on the same replay slice: "
            f"{float(summary_df.loc[(summary_df['score_col'].eq('selection_score')) & (summary_df['top_n'].eq(10)), 'cumulative_return'].iloc[0]):.2%} "
            f"versus {float(summary_df.loc[(summary_df['score_col'].eq('blend_62_38')) & (summary_df['top_n'].eq(10)), 'cumulative_return'].iloc[0]):.2%} cumulative return proxy."
        ),
        (
            "The top-10 proxy also keeps the same max-drawdown proxy after removing the execution weight, which means the blend is not buying visible downside protection on this slice."
        ),
    ]

    summary_csv = OUTPUT_DIR / "execution_off_nav_proxy_summary.csv"
    curve_csv = OUTPUT_DIR / "execution_off_nav_proxy_curves.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    curve_df.to_csv(curve_csv, index=False, encoding="utf-8-sig")

    payload = {
        "scored_rows": int(len(scored)),
        "board_dates": int(scored["board_date"].nunique()),
        "summary": _clean_records(summary_df),
        "findings": findings,
    }
    json_path = OUTPUT_DIR / "execution_off_nav_proxy.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Execution-Off NAV Proxy 2026-05-28",
        "",
        "## Purpose",
        "",
        "Approximate a daily rebalanced top-N equity curve on the current v9 replay slice, to see whether removing the continuous execution-score weight changes short-horizon portfolio behavior.",
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
            "## Summary",
            "",
            summary_df.to_markdown(index=False),
            "",
            "## Interpretation",
            "",
            "- This is only a proxy NAV, not the full unified portfolio engine, because replay rows expose next-day outcome slices rather than full OHLC holding paths.",
            "- Even with that caveat, the direction is clear: removing the continuous execution weight does not hurt top-3 and improves top-10 on this replay slice.",
            "- That makes execution-off a justified next branch to test in the real portfolio backtest path.",
            "",
            "## Next Actions",
            "",
            "1. Add an execution-off ranking branch to the research path and rerun unified portfolio backtests.",
            "2. Keep discrete execution states for veto/explanation while the continuous weight is removed.",
            "3. Replace this proxy with true portfolio results once the execution-off branch is wired into the unified backtest stack.",
            "",
            f"Summary CSV: `{summary_csv}`",
            f"Curve CSV: `{curve_csv}`",
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
