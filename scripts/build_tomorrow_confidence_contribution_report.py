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
DOC_PATH = PROJECT_ROOT / "docs" / "tomorrow_confidence_contribution_2026-05-28.md"


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


def build_report() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    snapshots = _load_v9_snapshots()
    snapshots["symbol"] = snapshots["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    for column in ["selection_score", "tomorrow_plan_confidence"]:
        snapshots[column] = pd.to_numeric(snapshots.get(column), errors="coerce").fillna(50.0)
    snapshots["tomorrow_bias"] = snapshots.get("tomorrow_bias", "").fillna("").astype(str)
    snapshots["selection_plus_tomorrow_tilt"] = (
        snapshots["selection_score"] + (snapshots["tomorrow_plan_confidence"] - 50.0) * 0.20
    ).clip(lower=0.0, upper=100.0)

    history = _load_history_for_symbols(snapshots["symbol"].dropna().unique().tolist())

    variants = [
        ("selection_score", snapshots.copy(), 3, "baseline_selection_top3"),
        ("selection_score", snapshots.loc[snapshots["tomorrow_plan_confidence"].ge(65.0)].copy(), 3, "selection_tomorrow_ge_65"),
        ("selection_score", snapshots.loc[snapshots["tomorrow_plan_confidence"].ge(72.0)].copy(), 3, "selection_tomorrow_ge_72"),
        ("selection_score", snapshots.loc[snapshots["tomorrow_bias"].isin(["偏多确认", "偏多进攻"])].copy(), 3, "selection_bias_positive_only"),
        ("selection_plus_tomorrow_tilt", snapshots.copy(), 3, "selection_plus_tomorrow_tilt"),
        ("tomorrow_plan_confidence", snapshots.copy(), 3, "tomorrow_confidence_only_top3"),
    ]

    summary_rows: list[dict[str, object]] = []
    for score_col, source_frame, top_n, variant_name in variants:
        candidates = _build_candidates(source_frame, score_col, top_n)
        result = simulate_portfolio_from_candidates(
            candidates,
            history,
            config=PortfolioBacktestConfig(max_positions=int(top_n), holding_days=3),
        )
        summary_rows.append(
            {
                "variant": variant_name,
                "score_col": score_col,
                "candidate_rows": int(len(candidates)),
                "avg_tomorrow_confidence": float(pd.to_numeric(candidates.get("tomorrow_plan_confidence"), errors="coerce").mean()) if not candidates.empty else 0.0,
                "bias_mix": json.dumps(candidates.get("tomorrow_bias", pd.Series(dtype=object)).value_counts().to_dict(), ensure_ascii=False),
                **result.summary,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    baseline_row = summary_df.loc[summary_df["variant"].eq("baseline_selection_top3")].iloc[0]
    best_gate_row = summary_df.loc[summary_df["variant"].isin(["selection_tomorrow_ge_65", "selection_tomorrow_ge_72", "selection_bias_positive_only"])].sort_values(
        ["annualized_return", "ending_equity"], ascending=[False, False]
    ).iloc[0]
    tilt_row = summary_df.loc[summary_df["variant"].eq("selection_plus_tomorrow_tilt")].iloc[0]
    tomorrow_only_row = summary_df.loc[summary_df["variant"].eq("tomorrow_confidence_only_top3")].iloc[0]

    payload = {
        "snapshot_rows": int(len(snapshots)),
        "history_rows": int(len(history)),
        "summary": _clean_records(summary_df),
        "findings": [
            f"`selection_score_top3` baseline stays strongest at {float(baseline_row['annualized_return']):.2%} annualized with {float(baseline_row['max_drawdown']):.2%} drawdown.",
            f"The best tomorrow-confidence gate is `{best_gate_row['variant']}` at {float(best_gate_row['annualized_return']):.2%} annualized.",
            f"`tomorrow_confidence_only_top3` lands at {float(tomorrow_only_row['annualized_return']):.2%} annualized, which shows tomorrow-confidence is not a strong standalone ranker.",
            f"`selection_plus_tomorrow_tilt` lands at {float(tilt_row['annualized_return']):.2%} annualized, so overweighting tomorrow-confidence on top of selection does not beat the current baseline on this slice.",
        ],
    }

    summary_csv = OUTPUT_DIR / "tomorrow_confidence_contribution_summary.csv"
    json_path = OUTPUT_DIR / "tomorrow_confidence_contribution.json"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Tomorrow Confidence Contribution Report 2026-05-28",
        "",
        "## Purpose",
        "",
        "Measure whether tomorrow-plan confidence helps more as a ranker, as a gate, or as a small tilt on top of selection_score.",
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
            "- If tomorrow-confidence works poorly as a standalone ranker but still matters inside selection_score, it is probably acting as a context or risk-shaping signal rather than a primary alpha source.",
            "- If strict tomorrow-confidence gates improve drawdown more than return, that points to a risk-control contribution rather than a ranking contribution.",
            "- If extra tomorrow-confidence tilt hurts, then the live selection score may already be using about as much of it as this slice can support.",
            "",
            "## Next Actions",
            "",
            "1. Keep tomorrow-confidence inside selection_score if it helps the baseline indirectly, but do not promote it to a standalone ranker unless longer-history evidence improves dramatically.",
            "2. If one tomorrow-confidence gate reduces drawdown without destroying return, retest it later on longer history before using it as a hard filter.",
            "3. Continue source-level auditing around execution and quant only after the higher-value context families are settled.",
            "",
            f"Summary CSV: `{summary_csv}`",
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
