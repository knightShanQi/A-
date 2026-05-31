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
DOC_PATH = PROJECT_ROOT / "docs" / "selection_score_family_compression_2026-05-28.md"


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
        obj = _load_pickle(path)
        board = obj.get("board")
        meta = dict(obj.get("meta") or {})
        if not isinstance(board, pd.DataFrame) or board.empty:
            continue
        frame = board.copy()
        frame["board_date"] = str(meta.get("board_date") or "")
        frame["snapshot_path"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["trade_date", "symbol", "open", "high", "low", "close"])


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


def _top3_overlap(frame: pd.DataFrame, score_col: str) -> tuple[float, int]:
    ranked = frame.copy()
    ranked["market_date"] = pd.to_datetime(ranked["board_date"], errors="coerce")
    ranked["symbol"] = ranked["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    ranked["selection_score"] = pd.to_numeric(ranked["selection_score"], errors="coerce")
    ranked[score_col] = pd.to_numeric(ranked[score_col], errors="coerce")
    base = (
        ranked.dropna(subset=["market_date", "selection_score"])
        .sort_values(["market_date", "selection_score", "symbol"], ascending=[True, False, True])
        .groupby("market_date", group_keys=False)
        .head(3)
    )
    comp = (
        ranked.dropna(subset=["market_date", score_col])
        .sort_values(["market_date", score_col, "symbol"], ascending=[True, False, True])
        .groupby("market_date", group_keys=False)
        .head(3)
    )
    overlaps: list[int] = []
    for market_date in sorted(set(base["market_date"]) & set(comp["market_date"])):
        left = set(base.loc[base["market_date"].eq(market_date), "symbol"])
        right = set(comp.loc[comp["market_date"].eq(market_date), "symbol"])
        overlaps.append(len(left & right))
    return (float(np.mean(overlaps)) if overlaps else 0.0, int(sum(value == 3 for value in overlaps)))


def build_report() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    snapshots = _load_v9_snapshots()
    if snapshots.empty:
        raise RuntimeError("No snapshot_v9 artifacts found for selection-score family compression report.")

    for column in [
        "selection_score",
        "attention_score",
        "enhanced_attention_score",
        "probability_up",
        "launch_window_score",
        "launch_window_confidence",
        "tomorrow_plan_confidence",
    ]:
        snapshots[column] = pd.to_numeric(snapshots.get(column), errors="coerce")

    snapshots["selection_minus_attention_layer"] = snapshots["selection_score"] - snapshots["attention_score"] * 0.10
    snapshots["selection_minus_enhanced_attention_layer"] = snapshots["selection_score"] - snapshots["enhanced_attention_score"] * 0.22
    snapshots["selection_minus_launch_confidence"] = snapshots["selection_score"] - (snapshots["launch_window_confidence"] - 50.0) * 0.04
    snapshots["selection_minus_launch_family"] = snapshots["selection_score"] - (snapshots["launch_window_score"] - 50.0) * 0.24 - (snapshots["launch_window_confidence"] - 50.0) * 0.04
    snapshots["selection_minus_tomorrow_confidence"] = snapshots["selection_score"] - snapshots["tomorrow_plan_confidence"] * 0.06

    history = _load_history_for_symbols(snapshots["symbol"].dropna().unique().tolist())

    score_columns = [
        "selection_score",
        "selection_minus_attention_layer",
        "selection_minus_enhanced_attention_layer",
        "selection_minus_launch_confidence",
        "selection_minus_launch_family",
        "selection_minus_tomorrow_confidence",
    ]

    summary_rows: list[dict[str, object]] = []
    nav_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    for score_col in score_columns:
        candidates = _build_candidates(snapshots.copy(), score_col, 3)
        result = simulate_portfolio_from_candidates(
            candidates,
            history,
            config=PortfolioBacktestConfig(max_positions=3, holding_days=3),
        )
        avg_overlap, exact_days = _top3_overlap(snapshots, score_col)
        summary_rows.append(
            {
                "score_col": score_col,
                "candidate_rows": int(len(candidates)),
                "avg_score": float(candidates[score_col].mean()) if not candidates.empty else 0.0,
                "avg_top3_overlap_vs_baseline": avg_overlap,
                "exact_top3_match_days": exact_days,
                **result.summary,
            }
        )
        if not result.daily_nav.empty:
            nav = result.daily_nav.copy()
            nav["score_col"] = score_col
            nav_frames.append(nav)
        if not result.trades.empty:
            trades = result.trades.copy()
            trades["score_col"] = score_col
            trade_frames.append(trades)

    summary_df = pd.DataFrame(summary_rows).sort_values("annualized_return", ascending=False, ignore_index=True)
    nav_df = pd.concat(nav_frames, ignore_index=True) if nav_frames else pd.DataFrame()
    trades_df = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()

    baseline_row = summary_df.loc[summary_df["score_col"].eq("selection_score")].iloc[0]
    best_compressed = summary_df.loc[summary_df["score_col"].ne("selection_score")].iloc[0]
    safest_compressed = summary_df.loc[summary_df["score_col"].ne("selection_score")].sort_values(
        ["avg_top3_overlap_vs_baseline", "annualized_return"],
        ascending=[False, False],
    ).iloc[0]

    findings = [
        (
            f"Baseline `selection_score_top3` remains the reference at {float(baseline_row['annualized_return']):.2%} annualized "
            f"with {float(baseline_row['max_drawdown']):.2%} max drawdown."
        ),
        (
            f"The strongest compression variant on the current v9 slice is `{str(best_compressed['score_col'])}` at "
            f"{float(best_compressed['annualized_return']):.2%} annualized with {float(best_compressed['max_drawdown']):.2%} drawdown."
        ),
        (
            f"The safest structural simplification by overlap is `{str(safest_compressed['score_col'])}`, which still keeps "
            f"{float(safest_compressed['avg_top3_overlap_vs_baseline']):.2f} / 3 baseline names on average."
        ),
        "This experiment is a ranking-key counterfactual, not a full formula rewrite: it asks which explicit restated layer can be removed first with the least damage on the current replayable slice.",
    ]

    summary_csv = OUTPUT_DIR / "selection_score_family_compression_summary.csv"
    nav_csv = OUTPUT_DIR / "selection_score_family_compression_nav.csv"
    trades_csv = OUTPUT_DIR / "selection_score_family_compression_trades.csv"
    json_path = OUTPUT_DIR / "selection_score_family_compression.json"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    if not nav_df.empty:
        nav_df.to_csv(nav_csv, index=False, encoding="utf-8-sig")
    if not trades_df.empty:
        trades_df.to_csv(trades_csv, index=False, encoding="utf-8-sig")

    payload = {
        "snapshot_rows": int(len(snapshots)),
        "history_rows": int(len(history)),
        "summary": _clean_records(summary_df),
        "findings": findings,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Selection Score Family Compression Report 2026-05-28",
        "",
        "## Purpose",
        "",
        "Test which explicit attention/launch/tomorrow restatement layers inside `selection_score` can be removed first with the least portfolio damage on the current replayable slice.",
        "",
        "## Coverage",
        "",
        f"- Snapshot rows: {len(snapshots)}",
        f"- Cached history rows: {len(history)}",
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
            summary_df[
                [
                    "score_col",
                    "candidate_rows",
                    "avg_top3_overlap_vs_baseline",
                    "exact_top3_match_days",
                    "annualized_return",
                    "max_drawdown",
                    "ending_equity",
                    "trade_count",
                    "win_rate",
                    "avg_net_return",
                ]
            ].to_markdown(index=False),
            "",
            "## Interpretation",
            "",
            "- If a subtraction variant stays close to or above the baseline, that layer is a good simplification candidate because it is behaving more like restatement than unique alpha.",
            "- If removing a layer immediately collapses annualized return or top-3 overlap, that layer is still doing real ranking work on this slice even if it is conceptually redundant.",
            "- Because these variants start from the persisted baseline score and subtract one explicit formula layer, they are directly useful for sequencing architecture cleanup.",
            "",
            "## Next Actions",
            "",
            "1. Demote the least-damaging subtraction layer first in the real formula or in a research-only branch.",
            "2. After v10 artifacts accumulate, rerun the same report with the newly persisted source terms to confirm the simplification survives better evidence.",
            "",
            f"Summary CSV: `{summary_csv}`",
            f"NAV CSV: `{nav_csv}`",
            f"Trades CSV: `{trades_csv}`",
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
