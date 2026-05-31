from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = PROJECT_ROOT / ".cache" / "daily_focus_board_reviews"
OUTPUT_DIR = PROJECT_ROOT / ".cache" / "trading_system_attribution"
DOC_PATH = PROJECT_ROOT / "docs" / "quant_gate_longer_window_2026-05-28.md"


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _clean_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in frame.to_dict("records"):
        cleaned: dict[str, object] = {}
        for key, value in row.items():
            if isinstance(value, (np.floating, float)) and pd.isna(value):
                cleaned[key] = None
            else:
                cleaned[key] = value
        records.append(cleaned)
    return records


def _load_v9_review_details() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(REVIEW_DIR.glob("review_v9_*.pkl")):
        payload = _load_pickle(path)
        meta = dict(payload.get("meta", {}))
        details = payload.get("details")
        if not isinstance(details, pd.DataFrame) or details.empty:
            continue
        if not {"selection_score", "quant_score", "next_day_return"}.issubset(details.columns):
            continue
        frame = details.copy()
        frame["board_date"] = str(meta.get("board_date") or "")
        frame["review_date"] = str(meta.get("review_date") or "")
        frame["review_path"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _select_topn(frame: pd.DataFrame, *, top_n: int, quant_min: float | None = None, quant_max: float | None = None) -> pd.DataFrame:
    working = frame.copy()
    working["selection_score"] = pd.to_numeric(working["selection_score"], errors="coerce")
    working["quant_score"] = pd.to_numeric(working["quant_score"], errors="coerce")
    working["next_day_return"] = pd.to_numeric(working["next_day_return"], errors="coerce")
    if quant_min is not None:
        working = working.loc[working["quant_score"].ge(float(quant_min))].copy()
    if quant_max is not None:
        working = working.loc[working["quant_score"].le(float(quant_max))].copy()
    working = working.dropna(subset=["board_date", "selection_score", "next_day_return"]).sort_values(
        ["board_date", "selection_score", "rank"],
        ascending=[True, False, True],
    )
    return working.groupby("board_date", group_keys=False).head(max(int(top_n), 1)).copy()


def _summarize_proxy(selected: pd.DataFrame, *, variant: str) -> tuple[dict[str, object], pd.DataFrame]:
    if selected.empty:
        daily = pd.DataFrame(columns=["board_date", "selected_count", "avg_next_day_return", "equity", "drawdown"])
        return {
            "variant": variant,
            "board_dates": 0,
            "candidate_rows": 0,
            "avg_selected_quant_score": 0.0,
            "avg_selected_selection_score": 0.0,
            "cumulative_return": 0.0,
            "annualized_return_proxy": 0.0,
            "max_drawdown_proxy": 0.0,
            "win_rate": 0.0,
            "avg_next_day_return": 0.0,
        }, daily

    daily = (
        selected.groupby("board_date")
        .agg(
            selected_count=("symbol", "count"),
            avg_next_day_return=("next_day_return", "mean"),
            avg_quant_score=("quant_score", "mean"),
            avg_selection_score=("selection_score", "mean"),
        )
        .reset_index()
        .sort_values("board_date")
        .reset_index(drop=True)
    )
    daily["equity"] = (1.0 + daily["avg_next_day_return"]).cumprod()
    daily["peak"] = daily["equity"].cummax()
    daily["drawdown"] = daily["equity"] / daily["peak"] - 1.0
    periods = int(len(daily))
    cumulative_return = float(daily["equity"].iloc[-1] - 1.0) if periods else 0.0
    annualized_proxy = float((daily["equity"].iloc[-1] ** (252.0 / periods)) - 1.0) if periods else 0.0
    summary = {
        "variant": variant,
        "board_dates": periods,
        "candidate_rows": int(len(selected)),
        "avg_selected_quant_score": float(selected["quant_score"].mean()),
        "avg_selected_selection_score": float(selected["selection_score"].mean()),
        "cumulative_return": cumulative_return,
        "annualized_return_proxy": annualized_proxy,
        "max_drawdown_proxy": float(daily["drawdown"].min()) if periods else 0.0,
        "win_rate": float((daily["avg_next_day_return"] > 0).mean()) if periods else 0.0,
        "avg_next_day_return": float(daily["avg_next_day_return"].mean()) if periods else 0.0,
    }
    return summary, daily


def build_report() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    details = _load_v9_review_details()
    variants = [
        ("selection_top3", None, None),
        ("selection_top3_quant_ge_58", 58.0, None),
        ("selection_top3_quant_ge_68", 68.0, None),
        ("selection_top3_quant_le_45", None, 45.0),
    ]

    summary_rows: list[dict[str, object]] = []
    daily_frames: list[pd.DataFrame] = []
    for variant, quant_min, quant_max in variants:
        selected = _select_topn(details, top_n=3, quant_min=quant_min, quant_max=quant_max)
        summary, daily = _summarize_proxy(selected, variant=variant)
        summary_rows.append(summary)
        if not daily.empty:
            daily = daily.copy()
            daily["variant"] = variant
            daily_frames.append(daily)

    summary_df = pd.DataFrame(summary_rows)
    daily_df = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()

    baseline = summary_df.loc[summary_df["variant"].eq("selection_top3")].iloc[0]
    mild_gate = summary_df.loc[summary_df["variant"].eq("selection_top3_quant_ge_58")].iloc[0]
    strict_gate = summary_df.loc[summary_df["variant"].eq("selection_top3_quant_ge_68")].iloc[0]

    payload = {
        "review_rows": int(len(details)),
        "review_files": int(details["review_path"].nunique()) if not details.empty else 0,
        "summary": _clean_records(summary_df),
        "findings": [
            f"Across the longer v9 review slice, baseline `selection_top3` proxy annualized return is about {float(baseline['annualized_return_proxy']):.2%}.",
            f"The mild gate `quant>=58` proxy annualized return is about {float(mild_gate['annualized_return_proxy']):.2%} with drawdown proxy {float(mild_gate['max_drawdown_proxy']):.2%}.",
            f"The stricter gate `quant>=68` proxy annualized return is about {float(strict_gate['annualized_return_proxy']):.2%}, showing whether the mild-gate edge survives stricter filtering.",
            "This longer-window proxy does not replace the unified portfolio engine, but it is the right next confidence check before treating quant>=58 as a candidate research rule.",
        ],
    }

    summary_csv = OUTPUT_DIR / "quant_gate_longer_window_summary.csv"
    daily_csv = OUTPUT_DIR / "quant_gate_longer_window_daily.csv"
    json_path = OUTPUT_DIR / "quant_gate_longer_window.json"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    if not daily_df.empty:
        daily_df.to_csv(daily_csv, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Quant Gate Longer Window Report 2026-05-28",
        "",
        "## Purpose",
        "",
        "Retest the promising `quant_score >= 58` gate on a longer v9 review-history slice using a daily next-day-return proxy.",
        "",
        "## Coverage",
        "",
        f"- Review rows: {len(details)}",
        f"- Review files: {details['review_path'].nunique() if not details.empty else 0}",
        "",
        "## Key Findings",
        "",
    ]
    for item in payload["findings"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Proxy Summary",
            "",
            summary_df.to_markdown(index=False),
            "",
            "## Interpretation",
            "",
            "- This is a longer-window replay proxy, not a replacement for the unified portfolio engine. It is useful for checking whether the mild quant gate is directionally stable beyond the 13 snapshot days.",
            "- If the mild gate still helps here while the strict gate degrades, that supports the idea that quant is useful mainly for removing obvious weak candidates rather than for aggressive overfiltering.",
            "- If the mild gate collapses here, then the short-slice improvement should be treated as fragile and not upgraded into research defaults yet.",
            "",
            "## Next Actions",
            "",
            "1. If quant>=58 still looks directionally positive here, run it next on a longer unified-portfolio candidate source before adopting it.",
            "2. If it does not, demote quant back to a lower-priority ingredient and move on to execution-score source-level cleanup.",
            "",
            f"Summary CSV: `{summary_csv}`",
            f"Daily CSV: `{daily_csv}`",
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
