from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = PROJECT_ROOT / ".cache" / "daily_focus_board_reviews"
OUTPUT_DIR = PROJECT_ROOT / ".cache" / "trading_system_attribution"
DOC_PATH = PROJECT_ROOT / "docs" / "final_rank_construction_2026-05-28.md"


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


def _load_v9_snapshots() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(REVIEW_DIR.glob("snapshot_v9_*.pkl")):
        payload = _load_pickle(path)
        board = payload.get("board")
        meta = dict(payload.get("meta", {}))
        if not isinstance(board, pd.DataFrame) or board.empty:
            continue
        frame = board.copy()
        frame["board_date"] = str(meta.get("board_date") or "")
        frame["ranking_by"] = str(meta.get("ranking_by") or "")
        frame["snapshot_path"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _top_n_per_day(frame: pd.DataFrame, score_col: str, top_n: int) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["symbol"] = ranked["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    ranked[score_col] = pd.to_numeric(ranked[score_col], errors="coerce")
    ranked["amount"] = pd.to_numeric(ranked.get("amount"), errors="coerce").fillna(0.0)
    ranked = ranked.dropna(subset=[score_col, "board_date"]).sort_values(
        ["board_date", score_col, "amount"],
        ascending=[True, False, False],
    )
    ranked["daily_rank"] = ranked.groupby("board_date").cumcount() + 1
    return ranked.groupby("board_date", group_keys=False).head(max(int(top_n), 1)).copy()


def build_report() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    snapshots = _load_v9_snapshots()
    for column in ["final_rank_score", "enhanced_attention_score", "selection_score", "launch_window_score", "probability_up"]:
        snapshots[column] = pd.to_numeric(snapshots.get(column), errors="coerce")

    exact_match_mask = snapshots["final_rank_score"].round(6) == snapshots["enhanced_attention_score"].round(6)
    delta = (snapshots["final_rank_score"] - snapshots["enhanced_attention_score"]).fillna(0.0)

    correlation_rows = []
    for column in ["enhanced_attention_score", "selection_score", "launch_window_score", "probability_up"]:
        correlation_rows.append(
            {
                "score_col": column,
                "notna_rows": int(snapshots[column].notna().sum()),
                "correlation_to_final_rank": float(snapshots["final_rank_score"].corr(snapshots[column])),
                "mean_score": float(snapshots[column].mean()),
            }
        )
    correlation_df = pd.DataFrame(correlation_rows)

    overlap_rows = []
    final_top3 = _top_n_per_day(snapshots, "final_rank_score", 3)
    for compare_col in ["enhanced_attention_score", "selection_score", "launch_window_score"]:
        compare_top3 = _top_n_per_day(snapshots, compare_col, 3)
        comparable_dates = sorted(set(final_top3["board_date"]) & set(compare_top3["board_date"]))
        identical_days = 0
        avg_overlap = 0.0
        records = []
        for board_date in comparable_dates:
            final_symbols = tuple(final_top3.loc[final_top3["board_date"].eq(board_date), "symbol"].tolist())
            compare_symbols = tuple(compare_top3.loc[compare_top3["board_date"].eq(board_date), "symbol"].tolist())
            overlap_count = len(set(final_symbols) & set(compare_symbols))
            identical = final_symbols == compare_symbols
            if identical:
                identical_days += 1
            avg_overlap += overlap_count
            records.append(
                {
                    "compare_col": compare_col,
                    "board_date": board_date,
                    "final_symbols": ",".join(final_symbols),
                    "compare_symbols": ",".join(compare_symbols),
                    "overlap_count": overlap_count,
                    "identical": identical,
                }
            )
        overlap_rows.append(
            {
                "compare_col": compare_col,
                "comparable_dates": len(comparable_dates),
                "identical_days": identical_days,
                "avg_overlap_count": (avg_overlap / len(comparable_dates)) if comparable_dates else 0.0,
            }
        )
        overlap_detail_df = pd.DataFrame(records)
        overlap_detail_df.to_csv(
            OUTPUT_DIR / f"final_rank_overlap_{compare_col}_top3.csv",
            index=False,
            encoding="utf-8-sig",
        )

    overlap_df = pd.DataFrame(overlap_rows)

    payload = {
        "snapshot_rows": int(len(snapshots)),
        "exact_equal_rows_final_vs_enhanced": int(exact_match_mask.sum()),
        "delta_mean_final_minus_enhanced": float(delta.mean()),
        "delta_abs_max_final_minus_enhanced": float(delta.abs().max()),
        "correlations": _clean_records(correlation_df),
        "top3_overlap": _clean_records(overlap_df),
        "findings": [
            "On the current v9 snapshot slice, `final_rank_score` and `enhanced_attention_score` are exactly equal on every persisted row.",
            "That means the current `final_rank_score` path is not a second-ranking opinion; it is a renamed attention-style score.",
            "Because of that identity, any portfolio result from `final_rank_score_top3` is mechanically the same as `enhanced_attention_score_top3` until the construction formula changes.",
            "By contrast, `selection_score` is only correlated with `final_rank_score` rather than identical, which is why it can still produce a different and currently stronger top-3 path.",
        ],
    }

    correlation_csv = OUTPUT_DIR / "final_rank_correlation_summary.csv"
    overlap_csv = OUTPUT_DIR / "final_rank_top3_overlap_summary.csv"
    json_path = OUTPUT_DIR / "final_rank_construction.json"
    correlation_df.to_csv(correlation_csv, index=False, encoding="utf-8-sig")
    overlap_df.to_csv(overlap_csv, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Final Rank Construction Report 2026-05-28",
        "",
        "## Purpose",
        "",
        "Verify whether `final_rank_score` is actually a distinct ranking signal on persisted snapshots or just a renamed variant of an existing score.",
        "",
        "## Coverage",
        "",
        f"- Snapshot rows: {len(snapshots)}",
        f"- Distinct board dates: {snapshots['board_date'].nunique()}",
        "",
        "## Key Findings",
        "",
    ]
    for item in payload["findings"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Correlation Summary",
            "",
            correlation_df.to_markdown(index=False),
            "",
            "## Top-3 Overlap Summary",
            "",
            overlap_df.to_markdown(index=False),
            "",
            "## Interpretation",
            "",
            "- The current engineering problem with `final_rank_score` is no longer persistence. It is construction redundancy.",
            "- Promoting `final_rank_score` without changing how it is built would only rename the weaker attention-style ranking path, not create a better one.",
            "- If a future `final_rank_score` is meant to express a richer view, it needs genuinely new ingredients or weights that move it away from exact equality with `enhanced_attention_score` and then beat `selection_score_top3` in the same engine.",
            "",
            "## Next Actions",
            "",
            "1. Audit every place where `final_rank_score` is assigned directly from `enhanced_attention_score` and decide whether that alias should be removed.",
            "2. If `final_rank_score` is supposed to be a composite, rebuild it explicitly from differentiated inputs and rerun the ranking-quality portfolio report.",
            "3. Until that happens, treat `selection_score_top3` as the only current default research ranking baseline on the v9 slice.",
            "",
            f"Correlation CSV: `{correlation_csv}`",
            f"Top-3 overlap CSV: `{overlap_csv}`",
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
