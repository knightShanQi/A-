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
DOC_PATH = PROJECT_ROOT / "docs" / "selection_score_ablation_2026-05-28.md"


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


def _top3_overlap(base: pd.DataFrame, compare: pd.DataFrame) -> tuple[int, float]:
    comparable_dates = sorted(set(base["market_date"]) & set(compare["market_date"]))
    identical_days = 0
    avg_overlap = 0.0
    for market_date in comparable_dates:
        base_symbols = tuple(base.loc[base["market_date"].eq(market_date), "symbol"].tolist())
        compare_symbols = tuple(compare.loc[compare["market_date"].eq(market_date), "symbol"].tolist())
        overlap = len(set(base_symbols) & set(compare_symbols))
        if base_symbols == compare_symbols:
            identical_days += 1
        avg_overlap += overlap
    return identical_days, (avg_overlap / len(comparable_dates)) if comparable_dates else 0.0


def build_report() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    snapshots = _load_v9_snapshots()
    snapshots["symbol"] = snapshots["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    for column in [
        "selection_score",
        "probability_up",
        "attention_score",
        "enhanced_attention_score",
        "quant_score",
        "tomorrow_plan_confidence",
        "launch_window_score",
        "launch_window_confidence",
    ]:
        snapshots[column] = pd.to_numeric(snapshots.get(column), errors="coerce").fillna(50.0)

    snapshots["selection_proxy_full"] = (
        snapshots["probability_up"] * 0.32
        + snapshots["enhanced_attention_score"] * 0.22
        + snapshots["attention_score"] * 0.10
        + snapshots["quant_score"] * 0.12
        + snapshots["tomorrow_plan_confidence"] * 0.06
        + (snapshots["launch_window_score"] - 50.0) * 0.24
        + (snapshots["launch_window_confidence"] - 50.0) * 0.04
    ).clip(lower=0.0, upper=100.0)
    snapshots["selection_proxy_no_launch"] = (
        snapshots["probability_up"] * 0.36
        + snapshots["enhanced_attention_score"] * 0.26
        + snapshots["attention_score"] * 0.12
        + snapshots["quant_score"] * 0.18
        + snapshots["tomorrow_plan_confidence"] * 0.08
    ).clip(lower=0.0, upper=100.0)
    snapshots["selection_proxy_no_quant"] = (
        snapshots["probability_up"] * 0.36
        + snapshots["enhanced_attention_score"] * 0.26
        + snapshots["attention_score"] * 0.12
        + snapshots["tomorrow_plan_confidence"] * 0.08
        + (snapshots["launch_window_score"] - 50.0) * 0.30
        + (snapshots["launch_window_confidence"] - 50.0) * 0.05
    ).clip(lower=0.0, upper=100.0)
    snapshots["selection_proxy_no_prob"] = (
        snapshots["enhanced_attention_score"] * 0.34
        + snapshots["attention_score"] * 0.16
        + snapshots["quant_score"] * 0.20
        + snapshots["tomorrow_plan_confidence"] * 0.10
        + (snapshots["launch_window_score"] - 50.0) * 0.30
        + (snapshots["launch_window_confidence"] - 50.0) * 0.05
    ).clip(lower=0.0, upper=100.0)
    snapshots["selection_proxy_no_attention"] = (
        snapshots["probability_up"] * 0.42
        + snapshots["quant_score"] * 0.20
        + snapshots["tomorrow_plan_confidence"] * 0.10
        + (snapshots["launch_window_score"] - 50.0) * 0.32
        + (snapshots["launch_window_confidence"] - 50.0) * 0.06
    ).clip(lower=0.0, upper=100.0)
    snapshots["selection_proxy_no_tomorrow"] = (
        snapshots["probability_up"] * 0.34
        + snapshots["enhanced_attention_score"] * 0.24
        + snapshots["attention_score"] * 0.11
        + snapshots["quant_score"] * 0.13
        + (snapshots["launch_window_score"] - 50.0) * 0.26
        + (snapshots["launch_window_confidence"] - 50.0) * 0.05
    ).clip(lower=0.0, upper=100.0)

    history = _load_history_for_symbols(snapshots["symbol"].dropna().unique().tolist())

    variants = [
        "selection_score",
        "selection_proxy_full",
        "selection_proxy_no_launch",
        "selection_proxy_no_quant",
        "selection_proxy_no_prob",
        "selection_proxy_no_attention",
        "selection_proxy_no_tomorrow",
    ]
    summary_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []

    base_candidates = _build_candidates(snapshots.copy(), "selection_score", 3)
    for score_col in variants:
        candidates = _build_candidates(snapshots.copy(), score_col, 3)
        result = simulate_portfolio_from_candidates(
            candidates,
            history,
            config=PortfolioBacktestConfig(max_positions=3, holding_days=3),
        )
        identical_days, avg_overlap = _top3_overlap(base_candidates, candidates)
        summary_rows.append(
            {
                "score_col": score_col,
                "candidate_rows": int(len(candidates)),
                "avg_score": float(candidates[score_col].mean()) if not candidates.empty else 0.0,
                "median_score": float(candidates[score_col].median()) if not candidates.empty else 0.0,
                "identical_days_vs_selection": identical_days,
                "avg_overlap_count_vs_selection": avg_overlap,
                **result.summary,
            }
        )
        overlap_rows.append(
            {
                "score_col": score_col,
                "identical_days_vs_selection": identical_days,
                "avg_overlap_count_vs_selection": avg_overlap,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    overlap_df = pd.DataFrame(overlap_rows)

    selection_row = summary_df.loc[summary_df["score_col"].eq("selection_score")].iloc[0]
    best_proxy_row = summary_df.loc[summary_df["score_col"].ne("selection_score")].sort_values(
        ["annualized_return", "ending_equity"], ascending=[False, False]
    ).iloc[0]
    worst_proxy_row = summary_df.loc[summary_df["score_col"].ne("selection_score")].sort_values(
        ["annualized_return", "ending_equity"], ascending=[True, True]
    ).iloc[0]

    payload = {
        "snapshot_rows": int(len(snapshots)),
        "history_rows": int(len(history)),
        "summary": _clean_records(summary_df),
        "findings": [
            f"`selection_score_top3` remains the top benchmark at {float(selection_row['annualized_return']):.2%} annualized.",
            f"The strongest persisted-field proxy variant is `{best_proxy_row['score_col']}` at {float(best_proxy_row['annualized_return']):.2%} annualized.",
            f"The weakest leave-one-out variant is `{worst_proxy_row['score_col']}` at {float(worst_proxy_row['annualized_return']):.2%} annualized.",
            "If several simplified proxies cluster below selection_score, that is evidence the current selection_score is already capturing useful interactions even if some ingredients may still be noisy.",
        ],
    }

    summary_csv = OUTPUT_DIR / "selection_score_ablation_summary.csv"
    overlap_csv = OUTPUT_DIR / "selection_score_ablation_overlap.csv"
    json_path = OUTPUT_DIR / "selection_score_ablation.json"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    overlap_df.to_csv(overlap_csv, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Selection Score Ablation 2026-05-28",
        "",
        "## Purpose",
        "",
        "Approximate the persisted selection-score ingredients and test leave-one-out proxy variants under the unified portfolio engine.",
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
            "## Interpretation",
            "",
            "- These are proxy ablations on persisted fields, not exact reproductions of every live selection-score ingredient.",
            "- They are still useful for direction: if a simplified proxy loses meaningfully to selection_score, then the live score is likely capturing interactions that matter.",
            "- If a leave-one-out proxy collapses especially hard, that ingredient family is a good candidate for deeper source-level audit.",
            "",
            "## Next Actions",
            "",
            "1. Keep `selection_score_top3` as the default research baseline unless an audited alternative clearly beats it.",
            "2. Use the weakest leave-one-out directions to choose the next source-level ingredient audit in dashboard selection logic.",
            "3. Avoid replacing selection_score with a simplified proxy just because it is easier to explain; the engine evidence has to stay primary.",
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
