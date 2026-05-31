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
DOC_PATH = PROJECT_ROOT / "docs" / "launch_window_confidence_sweep_2026-05-28.md"

CURRENT_WEIGHT = 0.04


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
    baseline = (
        ranked.dropna(subset=["market_date", "selection_score"])
        .sort_values(["market_date", "selection_score", "symbol"], ascending=[True, False, True])
        .groupby("market_date", group_keys=False)
        .head(3)
    )
    compared = (
        ranked.dropna(subset=["market_date", score_col])
        .sort_values(["market_date", score_col, "symbol"], ascending=[True, False, True])
        .groupby("market_date", group_keys=False)
        .head(3)
    )
    overlaps: list[int] = []
    for market_date in sorted(set(baseline["market_date"]) & set(compared["market_date"])):
        left = set(baseline.loc[baseline["market_date"].eq(market_date), "symbol"])
        right = set(compared.loc[compared["market_date"].eq(market_date), "symbol"])
        overlaps.append(len(left & right))
    return (float(sum(overlaps) / len(overlaps)) if overlaps else 0.0, int(sum(value == 3 for value in overlaps)))


def build_report() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    snapshots = _load_v9_snapshots()
    if snapshots.empty:
        raise RuntimeError("No snapshot_v9 artifacts found for launch-window-confidence sweep.")

    snapshots["selection_score"] = pd.to_numeric(snapshots.get("selection_score"), errors="coerce")
    snapshots["launch_window_confidence"] = pd.to_numeric(snapshots.get("launch_window_confidence"), errors="coerce")

    weights = [0.00, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08]
    for weight in weights:
        label = f"selection_launch_conf_w_{weight:.2f}".replace(".", "_")
        delta = (weight - CURRENT_WEIGHT) * (snapshots["launch_window_confidence"] - 50.0)
        snapshots[label] = snapshots["selection_score"] + delta

    history = _load_history_for_symbols(snapshots["symbol"].dropna().unique().tolist())

    summary_rows: list[dict[str, object]] = []
    nav_frames: list[pd.DataFrame] = []
    for weight in weights:
        score_col = f"selection_launch_conf_w_{weight:.2f}".replace(".", "_")
        candidates = _build_candidates(snapshots.copy(), score_col, 3)
        result = simulate_portfolio_from_candidates(
            candidates,
            history,
            config=PortfolioBacktestConfig(max_positions=3, holding_days=3),
        )
        avg_overlap, exact_days = _top3_overlap(snapshots.assign(**{score_col: snapshots[score_col]}), score_col)
        summary_rows.append(
            {
                "launch_window_confidence_weight": weight,
                "score_col": score_col,
                "candidate_rows": int(len(candidates)),
                "avg_top3_overlap_vs_baseline": avg_overlap,
                "exact_top3_match_days": exact_days,
                **result.summary,
            }
        )
        if not result.daily_nav.empty:
            nav = result.daily_nav.copy()
            nav["launch_window_confidence_weight"] = weight
            nav_frames.append(nav)

    summary_df = pd.DataFrame(summary_rows).sort_values("launch_window_confidence_weight", ignore_index=True)
    nav_df = pd.concat(nav_frames, ignore_index=True) if nav_frames else pd.DataFrame()

    baseline_row = summary_df.loc[summary_df["launch_window_confidence_weight"].eq(CURRENT_WEIGHT)].iloc[0]
    zero_row = summary_df.loc[summary_df["launch_window_confidence_weight"].eq(0.0)].iloc[0]
    best_row = summary_df.sort_values("annualized_return", ascending=False).iloc[0]

    findings = [
        (
            f"Current weight `{CURRENT_WEIGHT:.2f}` yields about {float(baseline_row['annualized_return']):.2%} annualized "
            f"with {float(baseline_row['max_drawdown']):.2%} max drawdown on the current v9 slice."
        ),
        (
            f"Zeroing the layer (`0.00`) yields about {float(zero_row['annualized_return']):.2%} annualized "
            f"with {float(zero_row['max_drawdown']):.2%} drawdown and {float(zero_row['avg_top3_overlap_vs_baseline']):.2f} / 3 average top-3 overlap."
        ),
        (
            f"The best tested weight on this slice is `{float(best_row['launch_window_confidence_weight']):.2f}` at "
            f"{float(best_row['annualized_return']):.2%} annualized."
        ),
        "This sweep tests the marginal ranking value of the launch-window-confidence layer rather than the broader launch context family.",
    ]

    summary_csv = OUTPUT_DIR / "launch_window_confidence_sweep_summary.csv"
    nav_csv = OUTPUT_DIR / "launch_window_confidence_sweep_nav.csv"
    json_path = OUTPUT_DIR / "launch_window_confidence_sweep.json"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    if not nav_df.empty:
        nav_df.to_csv(nav_csv, index=False, encoding="utf-8-sig")

    payload = {
        "snapshot_rows": int(len(snapshots)),
        "history_rows": int(len(history)),
        "summary": _clean_records(summary_df),
        "findings": findings,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Launch Window Confidence Sweep 2026-05-28",
        "",
        "## Purpose",
        "",
        "Measure the marginal portfolio value of the `launch_window_confidence` layer inside `selection_score` by sweeping its explicit weight around the current `0.04` setting.",
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
                    "launch_window_confidence_weight",
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
            "- If weights from `0.00` through the current setting are flat or improve as they decrease, the layer is effectively dead complexity and should be demoted or removed.",
            "- If a smaller positive weight wins but zero is materially worse, the right recommendation is weight reduction rather than full removal.",
            "",
            "## Next Actions",
            "",
            "1. If zero or near-zero is best, remove this layer from the research formula first.",
            "2. If only a smaller weight wins, demote the current `0.04` toward that smaller value and retest on future v10 artifacts.",
            "",
            f"Summary CSV: `{summary_csv}`",
            f"NAV CSV: `{nav_csv}`",
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
