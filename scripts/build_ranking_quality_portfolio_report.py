from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from a_share_predictor.portfolio_backtester import PortfolioBacktestConfig, simulate_portfolio_from_candidates


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = PROJECT_ROOT / ".cache" / "daily_focus_board_reviews"
HISTORY_DIR = PROJECT_ROOT / ".cache"
OUTPUT_DIR = PROJECT_ROOT / ".cache" / "trading_system_attribution"
DOC_PATH = PROJECT_ROOT / "docs" / "ranking_quality_portfolio_2026-05-28.md"


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
        frame["latest_market_data_date"] = str(meta.get("latest_market_data_date") or "")
        frame["cache_version"] = int(meta.get("cache_version") or 0)
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


def _top_ns_for_score(score_col: str) -> list[int]:
    if score_col == "selection_score":
        return [3, 10]
    return [3]


def build_report() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    snapshots = _load_v9_snapshots()
    snapshots["symbol"] = snapshots["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    for column in ["selection_score", "launch_window_score", "enhanced_attention_score", "final_rank_score"]:
        snapshots[column] = pd.to_numeric(snapshots.get(column), errors="coerce")
    history = _load_history_for_symbols(snapshots["symbol"].dropna().unique().tolist())

    score_columns = ["selection_score", "launch_window_score", "enhanced_attention_score", "final_rank_score"]
    summary_rows: list[dict[str, object]] = []
    nav_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []

    for score_col in score_columns:
        for top_n in _top_ns_for_score(score_col):
            candidates = _build_candidates(snapshots.copy(), score_col, top_n)
            result = simulate_portfolio_from_candidates(
                candidates,
                history,
                config=PortfolioBacktestConfig(max_positions=int(top_n), holding_days=3),
            )
            summary_rows.append(
                {
                    "score_col": score_col,
                    "top_n": int(top_n),
                    "candidate_rows": int(len(candidates)),
                    "score_coverage_rows": int(snapshots[score_col].notna().sum()) if score_col in snapshots.columns else 0,
                    "avg_score": float(candidates[score_col].mean()) if not candidates.empty else 0.0,
                    "median_score": float(candidates[score_col].median()) if not candidates.empty else 0.0,
                    **result.summary,
                }
            )
            if not result.daily_nav.empty:
                nav = result.daily_nav.copy()
                nav["score_col"] = score_col
                nav["top_n"] = int(top_n)
                nav_frames.append(nav)
            if not result.trades.empty:
                trades = result.trades.copy()
                trades["score_col"] = score_col
                trades["top_n"] = int(top_n)
                trade_frames.append(trades)

    summary_df = pd.DataFrame(summary_rows)
    nav_df = pd.concat(nav_frames, ignore_index=True) if nav_frames else pd.DataFrame()
    trades_df = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()

    selection_top3 = summary_df.loc[(summary_df["score_col"].eq("selection_score")) & (summary_df["top_n"].eq(3))].iloc[0]
    selection_top10 = summary_df.loc[(summary_df["score_col"].eq("selection_score")) & (summary_df["top_n"].eq(10))].iloc[0]
    launch_top3 = summary_df.loc[(summary_df["score_col"].eq("launch_window_score")) & (summary_df["top_n"].eq(3))].iloc[0]
    attention_top3 = summary_df.loc[(summary_df["score_col"].eq("enhanced_attention_score")) & (summary_df["top_n"].eq(3))].iloc[0]
    final_rank_top3 = summary_df.loc[(summary_df["score_col"].eq("final_rank_score")) & (summary_df["top_n"].eq(3))].iloc[0]

    findings = [
        (
            f"On the current v9 unified-portfolio slice, `selection_score_top3` is the strongest realized ranking path at "
            f"{float(selection_top3['annualized_return']):.2%} annualized with {float(selection_top3['max_drawdown']):.2%} max drawdown."
        ),
        (
            f"`launch_window_score_top3` is directionally useful but weaker as a pure ranking key: "
            f"{float(launch_top3['annualized_return']):.2%} annualized vs {float(selection_top3['annualized_return']):.2%} for selection."
        ),
        (
            f"`enhanced_attention_score_top3` is currently the weakest of the persisted top-3 rankers on this slice at "
            f"{float(attention_top3['annualized_return']):.2%} annualized and {float(attention_top3['max_drawdown']):.2%} max drawdown."
        ),
        (
            f"`final_rank_score_top3` now has same-engine evidence on `{int(final_rank_top3['candidate_rows'])}` candidate rows and "
            f"currently matches `enhanced_attention_score_top3` exactly at {float(final_rank_top3['annualized_return']):.2%} annualized."
        ),
        (
            f"Even the broader `selection_score_top10` path only reaches {float(selection_top10['annualized_return']):.2%} annualized "
            f"with {float(selection_top10['max_drawdown']):.2%} drawdown, so widening the basket still does not solve the deeper alpha problem."
        ),
    ]

    summary_csv = OUTPUT_DIR / "ranking_quality_portfolio_summary.csv"
    nav_csv = OUTPUT_DIR / "ranking_quality_portfolio_nav.csv"
    trades_csv = OUTPUT_DIR / "ranking_quality_portfolio_trades.csv"
    json_path = OUTPUT_DIR / "ranking_quality_portfolio.json"

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
        "# Ranking Quality Portfolio Report 2026-05-28",
        "",
        "## Purpose",
        "",
        "Compare the persisted ranking paths on the current v9 candidate slice under the same unified portfolio engine instead of using replay buckets alone.",
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
                    "top_n",
                    "candidate_rows",
                    "score_coverage_rows",
                    "avg_score",
                    "median_score",
                    "ending_equity",
                    "cumulative_return",
                    "annualized_return",
                    "max_drawdown",
                    "trade_count",
                    "win_rate",
                    "avg_net_return",
                ]
            ].to_markdown(index=False),
            "",
            "## Interpretation",
            "",
            "- On current persisted evidence, `selection_score` is still the best ranking key among the scorers that actually survive into snapshots and can be replayed through the real portfolio engine.",
            "- `launch_window_score` appears useful as a supporting filter or overlay, but not as the primary ranker.",
            "- `enhanced_attention_score` is weaker than `selection_score` on this slice, which supports the earlier recommendation to keep the candidate stack simple and focus on ranking quality instead of adding more narrative overlays.",
            "- After snapshot backfill, `final_rank_score` no longer has an observability gap on this slice; it currently behaves the same as `enhanced_attention_score`, so it still does not justify promotion over `selection_score`.",
            "",
            "## Next Actions",
            "",
            "1. Keep `selection_score_top3` as the current default research ranking baseline.",
            "2. Treat `launch_window_score` as filter/overlay research, not as the main ranker, until it beats `selection_score_top3` in the same engine.",
            "3. If `final_rank_score` is intended to be more than a rename of `enhanced_attention_score`, change its construction first and rerun this exact portfolio A/B before promoting it.",
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
