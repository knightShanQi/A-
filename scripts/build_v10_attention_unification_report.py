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
DOC_PATH = PROJECT_ROOT / "docs" / "v10_attention_unification_2026-05-28.md"


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_snapshots(version: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(REVIEW_DIR.glob(f"snapshot_v{version}_*.pkl")):
        payload = _load_pickle(path)
        meta = dict(payload.get("meta", {}))
        board = payload.get("board")
        if not isinstance(board, pd.DataFrame) or board.empty:
            continue
        frame = board.copy()
        frame["snapshot_path"] = str(path)
        frame["board_date"] = str(meta.get("board_date") or "")
        frame["cache_version"] = version
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshots = _load_snapshots(10)
    if snapshots.empty:
        raise RuntimeError("No snapshot_v10 artifacts found for v10 attention unification report.")

    for column in ["selection_score", "attention_score", "enhanced_attention_score"]:
        snapshots[column] = pd.to_numeric(snapshots.get(column), errors="coerce")

    snapshots["attention_delta"] = snapshots["enhanced_attention_score"] - snapshots["attention_score"]
    snapshots["selection_unify_to_base_attention"] = (
        snapshots["selection_score"] - snapshots["enhanced_attention_score"] * 0.22 + snapshots["attention_score"] * 0.22
    )
    snapshots["selection_unify_to_enhanced_attention"] = (
        snapshots["selection_score"] - snapshots["attention_score"] * 0.10 + snapshots["enhanced_attention_score"] * 0.10
    )
    snapshots["selection_mean_attention_representation"] = (
        snapshots["selection_score"]
        - snapshots["enhanced_attention_score"] * 0.22
        - snapshots["attention_score"] * 0.10
        + ((snapshots["attention_score"] + snapshots["enhanced_attention_score"]) / 2.0) * 0.32
    )

    history = _load_history_for_symbols(snapshots["symbol"].astype(str).tolist())
    score_columns = [
        "selection_score",
        "selection_unify_to_base_attention",
        "selection_unify_to_enhanced_attention",
        "selection_mean_attention_representation",
    ]

    rows: list[dict[str, object]] = []
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
        rows.append(
            {
                "score_col": score_col,
                "candidate_rows": int(len(candidates)),
                "avg_top3_overlap_vs_baseline": avg_overlap,
                "exact_top3_match_days": exact_days,
                **result.summary,
            }
        )
        nav = result.daily_nav.copy()
        nav["score_col"] = score_col
        nav_frames.append(nav)
        trades = result.trades.copy()
        if not trades.empty:
            trades["score_col"] = score_col
            trade_frames.append(trades)

    summary_df = pd.DataFrame(rows).sort_values("annualized_return", ascending=False, ignore_index=True)
    baseline = summary_df.loc[summary_df["score_col"].eq("selection_score")].iloc[0]
    unify_base = summary_df.loc[summary_df["score_col"].eq("selection_unify_to_base_attention")].iloc[0]
    unify_enh = summary_df.loc[summary_df["score_col"].eq("selection_unify_to_enhanced_attention")].iloc[0]
    mean_rep = summary_df.loc[summary_df["score_col"].eq("selection_mean_attention_representation")].iloc[0]

    delta_series = snapshots["attention_delta"].dropna()
    zero_rows = int((delta_series.abs() < 1e-9).sum())
    nonzero_rows = int((delta_series.abs() >= 1e-9).sum())

    findings = [
        f"`attention_score` and `enhanced_attention_score` are identical on {zero_rows} / {int(len(delta_series))} persisted `v10` rows, with correlation {float(snapshots[['attention_score', 'enhanced_attention_score']].corr().iloc[0,1]):.4f}.",
        f"Replacing both weights with the base-attention representation (`selection_unify_to_base_attention`) yields {float(unify_base['annualized_return']):.2%} annualized and {float(unify_base['max_drawdown']):.2%} drawdown versus baseline {float(baseline['annualized_return']):.2%} / {float(baseline['max_drawdown']):.2%}.",
        f"Replacing both weights with the enhanced-attention representation (`selection_unify_to_enhanced_attention`) yields {float(unify_enh['annualized_return']):.2%} annualized and {float(unify_enh['max_drawdown']):.2%} drawdown.",
        f"The mean-attention consolidation variant lands at {float(mean_rep['annualized_return']):.2%} annualized with {float(mean_rep['max_drawdown']):.2%} drawdown, which shows whether a single blended attention representation can survive without full dual-layer stacking.",
        f"Non-zero enhanced-minus-base attention deltas exist on {nonzero_rows} rows, so the right cleanup target is not field deletion but score-family consolidation around one surviving attention representation.",
    ]

    summary_csv = OUTPUT_DIR / "v10_attention_unification_summary.csv"
    nav_csv = OUTPUT_DIR / "v10_attention_unification_nav.csv"
    trades_csv = OUTPUT_DIR / "v10_attention_unification_trades.csv"
    json_path = OUTPUT_DIR / "v10_attention_unification.json"
    summary_df.to_csv(summary_csv, index=False)
    pd.concat(nav_frames, ignore_index=True).to_csv(nav_csv, index=False)
    pd.concat(trade_frames, ignore_index=True).to_csv(trades_csv, index=False)

    payload = {
        "snapshot_rows": int(len(snapshots)),
        "history_rows": int(len(history)),
        "zero_attention_delta_rows": zero_rows,
        "nonzero_attention_delta_rows": nonzero_rows,
        "summary": summary_df.to_dict(orient="records"),
        "findings": findings,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    lines = [
        "# v10 Attention Unification 2026-05-28",
        "",
        "## Purpose",
        "",
        "Test source-level consolidation variants for the duplicated attention family instead of bluntly subtracting one layer from the persisted `selection_score`.",
        "",
        "## Coverage",
        "",
        f"- Snapshot rows: {int(len(snapshots))}",
        f"- Cached history rows: {int(len(history))}",
        f"- Rows where `enhanced_attention_score == attention_score`: {zero_rows}",
        f"- Rows where the two attention fields differ: {nonzero_rows}",
        "",
        "## Key Findings",
        "",
    ]
    for finding in findings:
        lines.append(f"- {finding}")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            summary_df.to_markdown(index=False),
            "",
            "## Interpretation",
            "",
            "- A unification variant is more faithful to the actual architecture choice than a pure subtraction variant because it asks whether one attention representation can replace two, not whether attention should disappear.",
            "- If one unified representation stays close to baseline, the next engineering move should be consolidating the formula inputs and persistence around that single representation.",
            "- If every unification variant still degrades materially, the immediate optimization target should move away from attention-family cleanup and toward other score families or model quality.",
            "",
            "## Next Actions",
            "",
            "1. Prefer the least-damaging unification variant as the next source-level simplification candidate.",
            "2. If that variant is still materially weaker, keep both attention layers for now and postpone consolidation until the upstream formulas are rebuilt.",
            "",
            f"Summary CSV: `{summary_csv}`",
            f"NAV CSV: `{nav_csv}`",
            f"Trades CSV: `{trades_csv}`",
            f"JSON: `{json_path}`",
            "",
        ]
    )
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
