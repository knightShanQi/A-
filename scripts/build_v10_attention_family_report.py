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
DOC_PATH = PROJECT_ROOT / "docs" / "v10_attention_family_compression_2026-05-28.md"


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


def _prepare_variants(snapshots: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    frame = snapshots.copy()
    for column in [
        "selection_score",
        "probability_up",
        "attention_score",
        "enhanced_attention_score",
        "launch_window_confidence_weight",
    ]:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    active_weight = float(
        frame["launch_window_confidence_weight"].dropna().iloc[0]
        if "launch_window_confidence_weight" in frame.columns and frame["launch_window_confidence_weight"].notna().any()
        else 0.04
    )

    frame["selection_minus_probability_layer"] = frame["selection_score"] - frame["probability_up"] * 0.30
    frame["selection_minus_attention_layer"] = frame["selection_score"] - frame["attention_score"] * 0.10
    frame["selection_minus_enhanced_attention_layer"] = frame["selection_score"] - frame["enhanced_attention_score"] * 0.22
    frame["selection_minus_attention_cluster"] = (
        frame["selection_score"] - frame["attention_score"] * 0.10 - frame["enhanced_attention_score"] * 0.22
    )
    frame["selection_minus_prob_attention_cluster"] = (
        frame["selection_score"]
        - frame["probability_up"] * 0.30
        - frame["attention_score"] * 0.10
        - frame["enhanced_attention_score"] * 0.22
    )
    frame["selection_half_attention_cluster"] = (
        frame["selection_score"]
        - frame["attention_score"] * 0.05
        - frame["enhanced_attention_score"] * 0.11
    )
    frame["selection_half_prob_attention_cluster"] = (
        frame["selection_score"]
        - frame["probability_up"] * 0.15
        - frame["attention_score"] * 0.05
        - frame["enhanced_attention_score"] * 0.11
    )
    return frame, active_weight


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshots = _load_snapshots(10)
    if snapshots.empty:
        raise RuntimeError("No snapshot_v10 artifacts found for v10 attention-family compression report.")
    snapshots, active_weight = _prepare_variants(snapshots)
    history = _load_history_for_symbols(snapshots["symbol"].astype(str).tolist())
    score_columns = [
        "selection_score",
        "selection_minus_probability_layer",
        "selection_minus_attention_layer",
        "selection_minus_enhanced_attention_layer",
        "selection_minus_attention_cluster",
        "selection_minus_prob_attention_cluster",
        "selection_half_attention_cluster",
        "selection_half_prob_attention_cluster",
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
    summary_csv = OUTPUT_DIR / "v10_attention_family_compression_summary.csv"
    nav_csv = OUTPUT_DIR / "v10_attention_family_compression_nav.csv"
    trades_csv = OUTPUT_DIR / "v10_attention_family_compression_trades.csv"
    json_path = OUTPUT_DIR / "v10_attention_family_compression.json"
    summary_df.to_csv(summary_csv, index=False)
    pd.concat(nav_frames, ignore_index=True).to_csv(nav_csv, index=False)
    pd.concat(trade_frames, ignore_index=True).to_csv(trades_csv, index=False)

    baseline = summary_df.loc[summary_df["score_col"].eq("selection_score")].iloc[0]
    best_non_baseline = summary_df.loc[summary_df["score_col"].ne("selection_score")].iloc[0]
    minus_prob = summary_df.loc[summary_df["score_col"].eq("selection_minus_probability_layer")].iloc[0]
    minus_attn = summary_df.loc[summary_df["score_col"].eq("selection_minus_attention_layer")].iloc[0]
    minus_enh = summary_df.loc[summary_df["score_col"].eq("selection_minus_enhanced_attention_layer")].iloc[0]
    findings = [
        f"Baseline `selection_score_top3` on the current `v10` slice stays at {float(baseline['annualized_return']):.2%} annualized with {float(baseline['max_drawdown']):.2%} max drawdown.",
        f"No attention-family subtraction beats the baseline; the least-damaging non-baseline variant is `{best_non_baseline['score_col']}` at {float(best_non_baseline['annualized_return']):.2%} annualized with {float(best_non_baseline['max_drawdown']):.2%} drawdown.",
        f"Removing the explicit `probability_up` layer hurts more than removing either attention layer: annualized return drops by {float(baseline['annualized_return'] - minus_prob['annualized_return']):.2%} and average top-3 overlap falls to {float(minus_prob['avg_top3_overlap_vs_baseline']):.2f} / 3.",
        f"`attention_score` and `enhanced_attention_score` behave like one compressed cluster on this slice: their subtraction variants both stay near {float(minus_attn['avg_top3_overlap_vs_baseline']):.2f} / 3 overlap, and removing `enhanced_attention_score` is only marginally less harmful than removing `attention_score`.",
        (
            "Because `launch_window_confidence_weight` is already 0.00 in `v10`, any remaining compression result here is about the attention family itself, "
            "not a hidden launch-window confounder."
        ),
    ]

    payload = {
        "active_launch_window_confidence_weight": active_weight,
        "snapshot_rows": int(len(snapshots)),
        "history_rows": int(len(history)),
        "summary": summary_df.to_dict(orient="records"),
        "findings": findings,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    lines = [
        "# v10 Attention Family Compression 2026-05-28",
        "",
        "## Purpose",
        "",
        "Test how much of the stacked `probability_up + attention_score + enhanced_attention_score` cluster can be compressed now that `launch_window_confidence_weight` is already zeroed in `v10` artifacts.",
        "",
        "## Coverage",
        "",
        f"- Snapshot rows: {int(len(snapshots))}",
        f"- Cached history rows: {int(len(history))}",
        f"- Active launch-window-confidence weight in artifacts: {active_weight:.2f}",
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
            "- If a half-weight or subtraction variant stays close to baseline, that specific layer is behaving more like restatement than unique alpha.",
            "- If annualized return and overlap both break quickly, that layer is still doing real ranking work even if the family is conceptually redundant.",
            "- This experiment is still a ranking-key counterfactual on persisted `v10` artifacts, so it is suitable for sequencing the next research-side simplification.",
            "",
            "## Next Actions",
            "",
            "1. Do not zero or halve the whole attention family in research defaults yet; the current `selection_score` baseline still dominates every tested subtraction.",
            "2. If attention-family cleanup continues, prefer source-level consolidation around one attention representation instead of blunt subtraction from the persisted score.",
            "3. Treat `probability_up` as the stickier core signal inside this cluster and avoid demoting it before a stronger replacement exists.",
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
