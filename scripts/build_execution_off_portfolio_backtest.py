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
DOC_PATH = PROJECT_ROOT / "docs" / "execution_off_portfolio_backtest_2026-05-28.md"


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
    ranked["candidate_priority"] = pd.to_numeric(ranked[score_col], errors="coerce").fillna(0.0)
    ranked = ranked.dropna(subset=["market_date"]).sort_values(["market_date", "candidate_priority", "amount"], ascending=[True, False, False])
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
    snapshots["selection_score"] = pd.to_numeric(snapshots["selection_score"], errors="coerce")
    snapshots["execution_score"] = pd.to_numeric(snapshots["execution_score"], errors="coerce")
    snapshots["final_rank_score"] = pd.to_numeric(snapshots.get("final_rank_score"), errors="coerce")
    snapshots["blend_62_38"] = snapshots["selection_score"] * 0.62 + snapshots["execution_score"] * 0.38

    history = _load_history_for_symbols(snapshots["symbol"].dropna().unique().tolist())

    variants = [
        ("selection_score", 3),
        ("blend_62_38", 3),
        ("final_rank_score", 3),
        ("selection_score", 10),
        ("blend_62_38", 10),
    ]
    summary_rows: list[dict[str, object]] = []
    nav_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    for score_col, top_n in variants:
        candidates = _build_candidates(snapshots.dropna(subset=[score_col]).copy(), score_col, top_n)
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

    findings = [
        (
            "On the real unified portfolio backtester fed by v9 snapshots plus cached daily history, "
            "execution-off `selection_score` and the current `0.62/0.38` blend are identical at top-3."
        ),
        (
            "At top-10, execution-off improves the real backtest summary versus the current blend if the reported annualized return, ending equity, or average net return is higher without increasing max drawdown."
        ),
    ]
    if not summary_df.empty:
        sel3 = summary_df.loc[(summary_df["score_col"].eq("selection_score")) & (summary_df["top_n"].eq(3))]
        blend3 = summary_df.loc[(summary_df["score_col"].eq("blend_62_38")) & (summary_df["top_n"].eq(3))]
        sel10 = summary_df.loc[(summary_df["score_col"].eq("selection_score")) & (summary_df["top_n"].eq(10))]
        blend10 = summary_df.loc[(summary_df["score_col"].eq("blend_62_38")) & (summary_df["top_n"].eq(10))]
        if not sel3.empty and not blend3.empty:
            findings[0] = (
                "On the real unified portfolio backtester fed by v9 snapshots plus cached daily history, "
                f"top-3 execution-off ends at {float(sel3.iloc[0]['ending_equity']):.0f} and the current blend ends at {float(blend3.iloc[0]['ending_equity']):.0f}."
            )
        if not sel10.empty and not blend10.empty:
            findings[1] = (
                "At top-10, execution-off vs current blend: "
                f"annualized {float(sel10.iloc[0]['annualized_return']):.2%} vs {float(blend10.iloc[0]['annualized_return']):.2%}, "
                f"ending equity {float(sel10.iloc[0]['ending_equity']):.0f} vs {float(blend10.iloc[0]['ending_equity']):.0f}, "
                f"max drawdown {float(sel10.iloc[0]['max_drawdown']):.2%} vs {float(blend10.iloc[0]['max_drawdown']):.2%}."
            )

    summary_csv = OUTPUT_DIR / "execution_off_portfolio_backtest_summary.csv"
    nav_csv = OUTPUT_DIR / "execution_off_portfolio_backtest_nav.csv"
    trades_csv = OUTPUT_DIR / "execution_off_portfolio_backtest_trades.csv"
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
    json_path = OUTPUT_DIR / "execution_off_portfolio_backtest.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Execution-Off Portfolio Backtest 2026-05-28",
        "",
        "## Purpose",
        "",
        "Run the unified portfolio backtester directly on v9 snapshot candidates and cached daily history, comparing execution-off ranking against the current blended execution weighting.",
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
            summary_df.to_markdown(index=False),
            "",
            "## Interpretation",
            "",
            "- This is the closest experiment so far to the real question: does execution-off help once trades are run through the unified portfolio engine instead of just replay averages.",
            "- If top-10 improves here without worse drawdown, the next step is to wire execution-off into the main research ranking path by default for A/B backtests.",
            "",
            "## Next Actions",
            "",
            "1. Use the new execution-off parameter in the real research branch and compare against the existing default in longer backtests.",
            "2. Keep discrete execution states for veto/explanation while the continuous execution weight is being removed.",
            "3. Extend this from v9 snapshots to a longer historical candidate source once the experiment shape is accepted.",
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
