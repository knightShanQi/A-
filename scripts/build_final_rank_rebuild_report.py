from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from a_share_predictor.portfolio_backtester import PortfolioBacktestConfig, simulate_portfolio_from_candidates


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = PROJECT_ROOT / ".cache" / "daily_focus_board_reviews"
HISTORY_DIR = PROJECT_ROOT / ".cache"
OUTPUT_DIR = PROJECT_ROOT / ".cache" / "trading_system_attribution"
DOC_PATH = PROJECT_ROOT / "docs" / "final_rank_rebuild_2026-05-28.md"


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
        frame["snapshot_path"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _load_history_for_symbols(symbols: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for symbol in sorted({str(value).zfill(6) for value in symbols if str(value).strip()}):
        path = HISTORY_DIR / f"daily_history_v1_{symbol}_qfq.pkl"
        if not path.exists():
            continue
        obj = _load_pickle(path)
        data = obj.get("data") if isinstance(obj, dict) else None
        if not isinstance(data, pd.DataFrame) or data.empty:
            continue
        frame = data.reset_index(drop=True).copy()
        if "date" in frame.columns:
            frame = frame.rename(columns={"date": "trade_date"})
        frame["symbol"] = frame.get("symbol", symbol)
        frames.append(frame[[column for column in ["trade_date", "symbol", "open", "high", "low", "close"] if column in frame.columns]])
    if not frames:
        return pd.DataFrame(columns=["trade_date", "symbol", "open", "high", "low", "close"])
    return pd.concat(frames, ignore_index=True)


def _zscore_per_day(frame: pd.DataFrame, column: str) -> pd.Series:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    grouped = numeric.groupby(frame["board_date"])
    means = grouped.transform("mean")
    stds = grouped.transform("std").replace(0.0, np.nan)
    return (((numeric - means) / stds).fillna(0.0)).astype(float)


def _build_candidates(snapshot_rows: pd.DataFrame, score_col: str, top_n: int) -> pd.DataFrame:
    ranked = snapshot_rows.copy()
    ranked["market_date"] = pd.to_datetime(ranked["board_date"], errors="coerce")
    ranked["symbol"] = ranked["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    ranked[score_col] = pd.to_numeric(ranked[score_col], errors="coerce")
    ranked["amount"] = pd.to_numeric(ranked.get("amount"), errors="coerce").fillna(0.0)
    ranked["candidate_priority"] = ranked[score_col].fillna(0.0)
    ranked = ranked.dropna(subset=["market_date", score_col]).sort_values(
        ["market_date", "candidate_priority", "amount"],
        ascending=[True, False, False],
    )
    ranked["daily_rank"] = ranked.groupby("market_date").cumcount() + 1
    selected = ranked.groupby("market_date", group_keys=False).head(max(int(top_n), 1)).copy()
    selected["candidate_strategy"] = f"{score_col}_top{int(top_n)}"
    selected["model_score"] = selected["candidate_priority"]
    selected["name"] = selected.get("name", "").astype(str)
    return selected


def build_report() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    snapshots = _load_v9_snapshots()
    snapshots["symbol"] = snapshots["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    for column in [
        "selection_score",
        "probability_up",
        "quant_score",
        "launch_window_score",
        "enhanced_attention_score",
        "final_rank_score",
    ]:
        snapshots[column] = pd.to_numeric(snapshots.get(column), errors="coerce")

    snapshots["rebuild_balanced"] = (
        snapshots["selection_score"] * 0.55
        + snapshots["probability_up"] * 0.20
        + snapshots["quant_score"] * 0.15
        + snapshots["launch_window_score"] * 0.10
    )
    snapshots["rebuild_launch_guard"] = (
        snapshots["selection_score"] * 0.60
        + snapshots["launch_window_score"] * 0.25
        + snapshots["probability_up"] * 0.15
    )
    snapshots["rebuild_quant_prob"] = (
        snapshots["selection_score"] * 0.65
        + snapshots["quant_score"] * 0.20
        + snapshots["probability_up"] * 0.15
    )
    snapshots["rebuild_zmix"] = (
        _zscore_per_day(snapshots, "selection_score") * 0.50
        + _zscore_per_day(snapshots, "probability_up") * 0.20
        + _zscore_per_day(snapshots, "quant_score") * 0.15
        + _zscore_per_day(snapshots, "launch_window_score") * 0.15
    ) * 10.0 + 50.0

    history = _load_history_for_symbols(snapshots["symbol"].dropna().unique().tolist())

    variants = [
        "selection_score",
        "final_rank_score",
        "rebuild_balanced",
        "rebuild_launch_guard",
        "rebuild_quant_prob",
        "rebuild_zmix",
    ]
    summary_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []

    base_top3 = _build_candidates(snapshots.copy(), "selection_score", 3)
    for score_col in variants:
        candidates = _build_candidates(snapshots.copy(), score_col, 3)
        result = simulate_portfolio_from_candidates(
            candidates,
            history,
            config=PortfolioBacktestConfig(max_positions=3, holding_days=3),
        )
        summary_rows.append(
            {
                "score_col": score_col,
                "candidate_rows": int(len(candidates)),
                "avg_score": float(candidates[score_col].mean()) if not candidates.empty else 0.0,
                "median_score": float(candidates[score_col].median()) if not candidates.empty else 0.0,
                **result.summary,
            }
        )

        comparable_dates = sorted(set(base_top3["market_date"]) & set(candidates["market_date"]))
        identical_days = 0
        avg_overlap = 0.0
        for market_date in comparable_dates:
            base_symbols = tuple(base_top3.loc[base_top3["market_date"].eq(market_date), "symbol"].tolist())
            variant_symbols = tuple(candidates.loc[candidates["market_date"].eq(market_date), "symbol"].tolist())
            overlap = len(set(base_symbols) & set(variant_symbols))
            if base_symbols == variant_symbols:
                identical_days += 1
            avg_overlap += overlap
        overlap_rows.append(
            {
                "score_col": score_col,
                "comparable_dates": len(comparable_dates),
                "identical_days_vs_selection": identical_days,
                "avg_overlap_count_vs_selection": (avg_overlap / len(comparable_dates)) if comparable_dates else 0.0,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    overlap_df = pd.DataFrame(overlap_rows)

    selection_row = summary_df.loc[summary_df["score_col"].eq("selection_score")].iloc[0]
    final_rank_row = summary_df.loc[summary_df["score_col"].eq("final_rank_score")].iloc[0]
    best_rebuild_row = summary_df.loc[summary_df["score_col"].str.startswith("rebuild_")].sort_values(
        ["annualized_return", "ending_equity"], ascending=[False, False]
    ).iloc[0]

    payload = {
        "snapshot_rows": int(len(snapshots)),
        "history_rows": int(len(history)),
        "summary": _clean_records(summary_df),
        "overlap": _clean_records(overlap_df),
        "findings": [
            f"`selection_score_top3` remains the baseline to beat at {float(selection_row['annualized_return']):.2%} annualized.",
            f"`final_rank_score_top3` remains weak at {float(final_rank_row['annualized_return']):.2%} annualized because it is still the attention-style alias path.",
            f"The best lightweight rebuild on current persisted fields is `{best_rebuild_row['score_col']}` at {float(best_rebuild_row['annualized_return']):.2%} annualized.",
            "If no rebuild beats selection_score, the immediate optimization focus should stay on improving selection_score construction rather than inventing another ranking label.",
        ],
    }

    summary_csv = OUTPUT_DIR / "final_rank_rebuild_summary.csv"
    overlap_csv = OUTPUT_DIR / "final_rank_rebuild_overlap.csv"
    json_path = OUTPUT_DIR / "final_rank_rebuild.json"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    overlap_df.to_csv(overlap_csv, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Final Rank Rebuild Report 2026-05-28",
        "",
        "## Purpose",
        "",
        "Test a few lightweight rebuilt final-rank formulas on persisted v9 snapshot fields under the same unified portfolio engine.",
        "",
        "## Coverage",
        "",
        f"- Snapshot rows: {len(snapshots)}",
        f"- Cached history rows: {len(history)}",
        "",
        "## Key Findings",
        "",
    ]
    for item in payload["findings"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Portfolio Summary",
            "",
            summary_df.to_markdown(index=False),
            "",
            "## Top-3 Overlap Vs Selection",
            "",
            overlap_df.to_markdown(index=False),
            "",
            "## Interpretation",
            "",
            "- These are not production formula recommendations. They are direction tests on the fields that currently survive into snapshots.",
            "- If even the best lightweight rebuild cannot beat `selection_score_top3`, then the path of least regret is to improve `selection_score` itself rather than creating a second ranking brand.",
            "- If a rebuild gets close but still loses, it can still help identify which ingredients are directionally useful for future model-aware ranking work.",
            "",
            "## Next Actions",
            "",
            "1. Keep `selection_score_top3` as the default research baseline unless a rebuilt final-rank formula clearly beats it in the same engine.",
            "2. Use the best rebuild only as an ingredient clue, not as a production replacement, until it is validated on longer history.",
            "3. Avoid adding execution-style or attention-style terms back into a rebuild unless they show distinct portfolio lift.",
            "",
            f"Summary CSV: `{summary_csv}`",
            f"Overlap CSV: `{overlap_csv}`",
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
