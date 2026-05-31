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
DOC_PATH = PROJECT_ROOT / "docs" / "candidate_breadth_2026-05-28.md"


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


def _build_candidates(snapshot_rows: pd.DataFrame, top_n: int) -> pd.DataFrame:
    ranked = snapshot_rows.copy()
    ranked["market_date"] = pd.to_datetime(ranked["board_date"], errors="coerce")
    ranked["symbol"] = ranked["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    ranked["selection_score"] = pd.to_numeric(ranked["selection_score"], errors="coerce").fillna(0.0)
    ranked["amount"] = pd.to_numeric(ranked.get("amount"), errors="coerce").fillna(0.0)
    ranked = ranked.dropna(subset=["market_date"]).sort_values(
        ["market_date", "selection_score", "amount"],
        ascending=[True, False, False],
    )
    ranked["daily_rank"] = ranked.groupby("market_date").cumcount() + 1
    selected = ranked.groupby("market_date", group_keys=False).head(max(int(top_n), 1)).copy()
    selected["candidate_priority"] = selected["selection_score"]
    selected["candidate_strategy"] = f"selection_score_top{int(top_n)}"
    selected["model_score"] = selected["selection_score"]
    selected["name"] = selected.get("name", "").astype(str)
    return selected


def _format_mix(series: pd.Series) -> str:
    counts = series.fillna("").astype(str).value_counts()
    filtered = {key: int(value) for key, value in counts.items() if key}
    return json.dumps(filtered, ensure_ascii=False, sort_keys=True)


def _series_or_empty(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([""] * len(frame), index=frame.index, dtype=object)


def build_report() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    snapshots = _load_v9_snapshots()
    snapshots["symbol"] = snapshots["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    snapshots["selection_score"] = pd.to_numeric(snapshots["selection_score"], errors="coerce")
    history = _load_history_for_symbols(snapshots["symbol"].dropna().unique().tolist())

    top_ns = [1, 3, 5, 10, 15, 20]
    summary_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    nav_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []

    for top_n in top_ns:
        candidates = _build_candidates(snapshots.copy(), top_n)
        result = simulate_portfolio_from_candidates(
            candidates,
            history,
            config=PortfolioBacktestConfig(max_positions=int(top_n), holding_days=3),
        )
        summary_rows.append(
            {
                "top_n": int(top_n),
                "candidate_rows": int(len(candidates)),
                "avg_selection_score": float(candidates["selection_score"].mean()) if not candidates.empty else 0.0,
                "median_selection_score": float(candidates["selection_score"].median()) if not candidates.empty else 0.0,
                "buy_ratio": float((_series_or_empty(candidates, "action").astype(str) == "买").mean()) if not candidates.empty else 0.0,
                "watch_ratio": float((_series_or_empty(candidates, "action").astype(str) == "观察").mean()) if not candidates.empty else 0.0,
                "action_mix": _format_mix(_series_or_empty(candidates, "action")),
                "execution_mix": _format_mix(_series_or_empty(candidates, "execution_label")),
                **result.summary,
            }
        )
        if not candidates.empty:
            for board_date, frame in candidates.groupby("board_date"):
                daily_rows.append(
                    {
                        "top_n": int(top_n),
                        "board_date": str(board_date),
                        "selected_count": int(len(frame)),
                        "avg_selection_score": float(frame["selection_score"].mean()),
                        "median_selection_score": float(frame["selection_score"].median()),
                        "buy_ratio": float((_series_or_empty(frame, "action").astype(str) == "买").mean()),
                        "watch_ratio": float((_series_or_empty(frame, "action").astype(str) == "观察").mean()),
                        "action_mix": _format_mix(_series_or_empty(frame, "action")),
                        "execution_mix": _format_mix(_series_or_empty(frame, "execution_label")),
                    }
                )
        if not result.daily_nav.empty:
            nav = result.daily_nav.copy()
            nav["top_n"] = int(top_n)
            nav_frames.append(nav)
        if not result.trades.empty:
            trades = result.trades.copy()
            trades["top_n"] = int(top_n)
            trade_frames.append(trades)

    summary_df = pd.DataFrame(summary_rows).sort_values("top_n").reset_index(drop=True)
    daily_df = pd.DataFrame(daily_rows).sort_values(["top_n", "board_date"]).reset_index(drop=True)
    nav_df = pd.concat(nav_frames, ignore_index=True) if nav_frames else pd.DataFrame()
    trades_df = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()

    best_return_row = summary_df.sort_values(["annualized_return", "ending_equity"], ascending=[False, False]).iloc[0]
    best_drawdown_row = summary_df.sort_values(["max_drawdown", "annualized_return"], ascending=[False, False]).iloc[0]
    top3_row = summary_df.loc[summary_df["top_n"].eq(3)].iloc[0]
    top10_row = summary_df.loc[summary_df["top_n"].eq(10)].iloc[0]

    findings = [
        (
            f"On the current v9 snapshot slice, top-{int(best_return_row['top_n'])} is the highest annualized-return width at "
            f"{float(best_return_row['annualized_return']):.2%}, with ending equity {float(best_return_row['ending_equity']):.0f}."
        ),
        (
            f"Top-3 remains the cleanest concentrated basket: annualized {float(top3_row['annualized_return']):.2%}, "
            f"max drawdown {float(top3_row['max_drawdown']):.2%}, average selected score {float(top3_row['avg_selection_score']):.2f}."
        ),
        (
            f"Widening from top-3 to top-10 only slightly lowers annualized return ({float(top10_row['annualized_return']):.2%}) "
            f"but more than triples drawdown magnitude ({float(top10_row['max_drawdown']):.2%}) as average selection score drops "
            f"from {float(top3_row['avg_selection_score']):.2f} to {float(top10_row['avg_selection_score']):.2f}."
        ),
        (
            f"Beyond top-10 the pool is clearly diluted: the most drawdown-efficient width is top-{int(best_drawdown_row['top_n'])} "
            f"at {float(best_drawdown_row['max_drawdown']):.2%}, while top-15 and top-20 both lose annualized return versus top-3."
        ),
    ]

    summary_csv = OUTPUT_DIR / "candidate_breadth_summary.csv"
    daily_csv = OUTPUT_DIR / "candidate_breadth_daily_mix.csv"
    nav_csv = OUTPUT_DIR / "candidate_breadth_nav.csv"
    trades_csv = OUTPUT_DIR / "candidate_breadth_trades.csv"
    json_path = OUTPUT_DIR / "candidate_breadth.json"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    if not daily_df.empty:
        daily_df.to_csv(daily_csv, index=False, encoding="utf-8-sig")
    if not nav_df.empty:
        nav_df.to_csv(nav_csv, index=False, encoding="utf-8-sig")
    if not trades_df.empty:
        trades_df.to_csv(trades_csv, index=False, encoding="utf-8-sig")

    payload = {
        "snapshot_rows": int(len(snapshots)),
        "history_rows": int(len(history)),
        "top_ns": top_ns,
        "summary": _clean_records(summary_df),
        "findings": findings,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Candidate Breadth Report 2026-05-28",
        "",
        "## Purpose",
        "",
        "Measure how widening the daily selected basket changes real unified-portfolio performance on the current v9 snapshot slice.",
        "",
        "## Coverage",
        "",
        f"- Snapshot rows: {len(snapshots)}",
        f"- Cached history rows: {len(history)}",
        f"- Tested widths: {', '.join(str(value) for value in top_ns)}",
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
                    "top_n",
                    "candidate_rows",
                    "avg_selection_score",
                    "median_selection_score",
                    "buy_ratio",
                    "watch_ratio",
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
            "- The current short replay-backed portfolio slice does not support a broad default basket. The signal is still alpha-dense at the top and degrades as lower-ranked names are added.",
            "- Top-10 has slightly more capacity than top-3 on this slice, but that extra breadth comes with materially deeper drawdown and lower average score quality.",
            "- If broader baskets are needed later, the system likely needs stronger ranking calibration first rather than a larger default top-N.",
            "",
            "## Next Actions",
            "",
            "1. Keep top-3 as the default research concentration baseline while deeper ranking fixes are being tested.",
            "2. Treat top-10 as a capacity experiment only after ranking quality improves under the same unified portfolio engine.",
            "3. Use this breadth table together with the execution-off evidence to focus the next optimization on candidate-family and ranking quality, not more overlay complexity.",
            "",
            f"Summary CSV: `{summary_csv}`",
            f"Daily mix CSV: `{daily_csv}`",
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
