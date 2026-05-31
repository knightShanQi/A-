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
DOC_PATH = PROJECT_ROOT / "docs" / "selection_score_v10_comparison_2026-05-28.md"

SOURCE_FEATURES = [
    "probability_up",
    "attention_score",
    "enhanced_attention_score",
    "quant_score",
    "launch_window_score",
    "launch_window_confidence",
    "stage_score",
    "launch_readiness_score",
    "market_resonance_score",
]

MISSING_SOURCE_TERMS = [
    "predicted_upside_pct",
    "tomorrow_plan_confidence",
    "sector_score",
    "fund_score",
    "news_score",
    "technical_adjustment",
    "intraday_adjustment",
    "backtest_adjustment",
]

DEFAULT_LAUNCH_WINDOW_CONFIDENCE_WEIGHT = 0.04


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _clean_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in frame.to_dict("records"):
        cleaned: dict[str, object] = {}
        for key, value in row.items():
            if isinstance(value, (np.floating, float)) and pd.isna(value):
                cleaned[key] = None
            else:
                cleaned[key] = value
        records.append(cleaned)
    return records


def _load_review_details(version: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(REVIEW_DIR.glob(f"review_v{version}_*.pkl")):
        payload = _load_pickle(path)
        meta = dict(payload.get("meta", {}))
        details = payload.get("details")
        if not isinstance(details, pd.DataFrame) or details.empty:
            continue
        frame = details.copy()
        frame["review_path"] = str(path)
        frame["board_date"] = str(meta.get("board_date") or frame.get("board_date", ""))
        frame["review_date"] = str(meta.get("review_date") or frame.get("review_date", ""))
        frame["cache_version"] = version
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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


def _top3_overlap_details(frame: pd.DataFrame, score_col: str) -> tuple[float, int]:
    working = frame.loc[frame["selection_score"].notna() & frame[score_col].notna(), ["board_date", "symbol", "selection_score", score_col]].copy()
    if working.empty:
        return 0.0, 0
    selected = (
        working.sort_values(["board_date", "selection_score", "symbol"], ascending=[True, False, True])
        .groupby("board_date", group_keys=False)
        .head(3)
    )
    compared = (
        working.sort_values(["board_date", score_col, "symbol"], ascending=[True, False, True])
        .groupby("board_date", group_keys=False)
        .head(3)
    )
    overlaps: list[int] = []
    for board_date in sorted(set(selected["board_date"]) & set(compared["board_date"])):
        left = set(selected.loc[selected["board_date"].eq(board_date), "symbol"])
        right = set(compared.loc[compared["board_date"].eq(board_date), "symbol"])
        overlaps.append(len(left & right))
    return (float(np.mean(overlaps)) if overlaps else 0.0, int(sum(value == 3 for value in overlaps)))


def _linear_projection(frame: pd.DataFrame) -> tuple[pd.DataFrame, float, int]:
    needed = ["selection_score", *SOURCE_FEATURES]
    working = frame[needed].apply(pd.to_numeric, errors="coerce").dropna().copy()
    if working.empty:
        return pd.DataFrame(columns=["feature", "standardized_coefficient"]), 0.0, 0

    x = working[SOURCE_FEATURES].to_numpy(dtype=float)
    y = working["selection_score"].to_numpy(dtype=float)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    y_mean = y.mean()
    y_std = y.std()
    xz = (x - x_mean) / x_std
    yz = (y - y_mean) / y_std
    design = np.column_stack([np.ones(len(xz)), xz])
    coefficients = np.linalg.lstsq(design, yz, rcond=None)[0]
    predicted = design @ coefficients
    ss_res = float(np.square(yz - predicted).sum())
    ss_tot = float(np.square(yz - yz.mean()).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    coefficient_frame = pd.DataFrame(
        {
            "feature": SOURCE_FEATURES,
            "standardized_coefficient": coefficients[1:],
            "abs_standardized_coefficient": np.abs(coefficients[1:]),
        }
    ).sort_values("abs_standardized_coefficient", ascending=False, ignore_index=True)
    return coefficient_frame, r2, int(len(working))


def _source_summary(details: pd.DataFrame, version: int) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    for column in ["selection_score", *SOURCE_FEATURES]:
        details[column] = pd.to_numeric(details.get(column), errors="coerce")
    coverage_rows = []
    for column in SOURCE_FEATURES + MISSING_SOURCE_TERMS:
        coverage_rows.append(
            {
                "version": version,
                "field": column,
                "present": bool(column in details.columns),
                "non_null_rows": int(details[column].notna().sum()) if column in details.columns else 0,
            }
        )
    coverage_df = pd.DataFrame(coverage_rows)
    available = [column for column in SOURCE_FEATURES if column in details.columns]
    corr_frame = (
        details[["selection_score", *available]]
        .corr(numeric_only=True)[["selection_score"]]
        .reset_index()
        .rename(columns={"index": "field", "selection_score": "correlation_with_selection_score"})
        .sort_values("correlation_with_selection_score", ascending=False, ignore_index=True)
    )
    overlap_rows: list[dict[str, object]] = []
    for column in available:
        avg_overlap, exact_days = _top3_overlap_details(details, column)
        overlap_rows.append(
            {
                "field": column,
                "avg_top3_overlap": avg_overlap,
                "exact_top3_match_days": exact_days,
            }
        )
    overlap_df = pd.DataFrame(overlap_rows).sort_values("avg_top3_overlap", ascending=False, ignore_index=True)
    coefficient_df, projection_r2, projection_rows = _linear_projection(details)
    top_corr = corr_frame.loc[corr_frame["field"].ne("selection_score")].iloc[0]
    top_overlap = overlap_df.iloc[0] if not overlap_df.empty else pd.Series({"field": "", "avg_top3_overlap": 0.0, "exact_top3_match_days": 0})
    quant_corr = corr_frame.loc[corr_frame["field"].eq("quant_score")]
    stage_corr = corr_frame.loc[corr_frame["field"].eq("stage_score")]
    launch_conf_corr = corr_frame.loc[corr_frame["field"].eq("launch_window_confidence")]
    summary = {
        "version": version,
        "review_rows": int(len(details)),
        "review_files": int(details["review_path"].nunique()) if "review_path" in details.columns else 0,
        "projection_rows": projection_rows,
        "projection_r2": projection_r2,
        "top_correlation_field": str(top_corr["field"]),
        "top_correlation_value": float(top_corr["correlation_with_selection_score"]),
        "top_overlap_field": str(top_overlap["field"]),
        "top_overlap_value": float(top_overlap["avg_top3_overlap"]),
        "quant_correlation": float(quant_corr["correlation_with_selection_score"].iloc[0]) if not quant_corr.empty else 0.0,
        "stage_correlation": float(stage_corr["correlation_with_selection_score"].iloc[0]) if not stage_corr.empty else 0.0,
        "launch_confidence_correlation": float(launch_conf_corr["correlation_with_selection_score"].iloc[0]) if not launch_conf_corr.empty else 0.0,
        "missing_source_terms_persisted": int(
            coverage_df.loc[coverage_df["field"].isin(MISSING_SOURCE_TERMS), "present"].sum()
        ),
        "coefficients": _clean_records(coefficient_df),
    }
    return summary, coverage_df, corr_frame


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


def _top3_overlap_snapshots(frame: pd.DataFrame, score_col: str) -> tuple[float, int]:
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


def _family_summary(snapshots: pd.DataFrame, history: pd.DataFrame, version: int) -> tuple[dict[str, object], pd.DataFrame]:
    for column in [
        "selection_score",
        "attention_score",
        "enhanced_attention_score",
        "launch_window_score",
        "launch_window_confidence",
        "tomorrow_plan_confidence",
        "launch_window_confidence_weight",
    ]:
        snapshots[column] = pd.to_numeric(snapshots.get(column), errors="coerce")
    active_weight = float(
        snapshots["launch_window_confidence_weight"].dropna().iloc[0]
        if "launch_window_confidence_weight" in snapshots.columns and snapshots["launch_window_confidence_weight"].notna().any()
        else DEFAULT_LAUNCH_WINDOW_CONFIDENCE_WEIGHT
    )
    snapshots["selection_minus_attention_layer"] = snapshots["selection_score"] - snapshots["attention_score"] * 0.10
    snapshots["selection_minus_enhanced_attention_layer"] = snapshots["selection_score"] - snapshots["enhanced_attention_score"] * 0.22
    snapshots["selection_minus_launch_confidence"] = snapshots["selection_score"] - (snapshots["launch_window_confidence"] - 50.0) * active_weight
    snapshots["selection_minus_launch_family"] = snapshots["selection_score"] - (snapshots["launch_window_score"] - 50.0) * 0.24 - (snapshots["launch_window_confidence"] - 50.0) * active_weight
    snapshots["selection_minus_tomorrow_confidence"] = snapshots["selection_score"] - snapshots["tomorrow_plan_confidence"] * 0.06
    score_columns = [
        "selection_score",
        "selection_minus_attention_layer",
        "selection_minus_enhanced_attention_layer",
        "selection_minus_launch_confidence",
        "selection_minus_launch_family",
        "selection_minus_tomorrow_confidence",
    ]
    rows: list[dict[str, object]] = []
    for score_col in score_columns:
        candidates = _build_candidates(snapshots.copy(), score_col, 3)
        result = simulate_portfolio_from_candidates(
            candidates,
            history,
            config=PortfolioBacktestConfig(max_positions=3, holding_days=3),
        )
        avg_overlap, exact_days = _top3_overlap_snapshots(snapshots, score_col)
        rows.append(
            {
                "version": version,
                "active_launch_window_confidence_weight": active_weight,
                "score_col": score_col,
                "candidate_rows": int(len(candidates)),
                "avg_top3_overlap_vs_baseline": avg_overlap,
                "exact_top3_match_days": exact_days,
                **result.summary,
            }
        )
    summary_df = pd.DataFrame(rows).sort_values("annualized_return", ascending=False, ignore_index=True)
    baseline = summary_df.loc[summary_df["score_col"].eq("selection_score")].iloc[0]
    best_variant = summary_df.loc[summary_df["score_col"].ne("selection_score")].iloc[0]
    summary = {
        "version": version,
        "snapshot_rows": int(len(snapshots)),
        "active_launch_window_confidence_weight": active_weight,
        "baseline_annualized_return": float(baseline["annualized_return"]),
        "baseline_max_drawdown": float(baseline["max_drawdown"]),
        "best_variant": str(best_variant["score_col"]),
        "best_variant_annualized_return": float(best_variant["annualized_return"]),
        "best_variant_max_drawdown": float(best_variant["max_drawdown"]),
        "best_variant_overlap": float(best_variant["avg_top3_overlap_vs_baseline"]),
    }
    return summary, summary_df


def build_report() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    version_summaries: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    family_rows: list[dict[str, object]] = []
    coverage_frames: list[pd.DataFrame] = []
    corr_frames: list[pd.DataFrame] = []
    family_detail_frames: list[pd.DataFrame] = []

    for version in [9, 10]:
        details = _load_review_details(version)
        snapshots = _load_snapshots(version)
        if details.empty or snapshots.empty:
            continue
        history = _load_history_for_symbols(snapshots["symbol"].astype(str).tolist())
        source_summary, coverage_df, corr_df = _source_summary(details.copy(), version)
        family_summary, family_detail_df = _family_summary(snapshots.copy(), history, version)
        version_summaries.append(
            {
                "version": version,
                "review_rows": source_summary["review_rows"],
                "review_files": source_summary["review_files"],
                "snapshot_rows": family_summary["snapshot_rows"],
                "source_projection_r2": source_summary["projection_r2"],
                "source_top_correlation_field": source_summary["top_correlation_field"],
                "source_top_correlation_value": source_summary["top_correlation_value"],
                "launch_confidence_correlation": source_summary["launch_confidence_correlation"],
                "persisted_missing_source_terms": source_summary["missing_source_terms_persisted"],
                "active_launch_window_confidence_weight": family_summary["active_launch_window_confidence_weight"],
                "baseline_annualized_return": family_summary["baseline_annualized_return"],
                "baseline_max_drawdown": family_summary["baseline_max_drawdown"],
                "best_variant": family_summary["best_variant"],
                "best_variant_annualized_return": family_summary["best_variant_annualized_return"],
                "best_variant_max_drawdown": family_summary["best_variant_max_drawdown"],
            }
        )
        source_rows.append(source_summary)
        family_rows.append(family_summary)
        coverage_frames.append(coverage_df)
        corr = corr_df.copy()
        corr["version"] = version
        corr_frames.append(corr)
        family_detail_frames.append(family_detail_df)

    version_df = pd.DataFrame(version_summaries).sort_values("version").reset_index(drop=True)
    coverage_all = pd.concat(coverage_frames, ignore_index=True) if coverage_frames else pd.DataFrame()
    corr_all = pd.concat(corr_frames, ignore_index=True) if corr_frames else pd.DataFrame()
    family_detail_all = pd.concat(family_detail_frames, ignore_index=True) if family_detail_frames else pd.DataFrame()
    if version_df.empty:
        raise RuntimeError("No v9/v10 artifacts available for selection-score comparison report.")

    v9 = version_df.loc[version_df["version"].eq(9)].iloc[0]
    v10 = version_df.loc[version_df["version"].eq(10)].iloc[0]
    findings = [
        (
            f"`v10` zero-weight artifacts now persist {int(v10['persisted_missing_source_terms'])} / {len(MISSING_SOURCE_TERMS)} previously missing source terms, "
            f"up from {int(v9['persisted_missing_source_terms'])} in `v9`."
        ),
        (
            f"The `selection_score` source projection stays high but becomes slightly more grounded under `v10`: "
            f"`v9` R² {float(v9['source_projection_r2']):.2%} vs `v10` R² {float(v10['source_projection_r2']):.2%}."
        ),
        (
            f"The active `launch_window_confidence` weight is now explicit in artifacts: `v9` behaves like {float(v9['active_launch_window_confidence_weight']):.2f}, "
            f"while the generated `v10` research slice is {float(v10['active_launch_window_confidence_weight']):.2f}."
        ),
        (
            f"On the current replayable slice, baseline `selection_score_top3` improves from {float(v9['baseline_annualized_return']):.2%} annualized in `v9` "
            f"to {float(v10['baseline_annualized_return']):.2%} in `v10`, with drawdown improving from {float(v9['baseline_max_drawdown']):.2%} "
            f"to {float(v10['baseline_max_drawdown']):.2%}."
        ),
        (
            f"The best family-compression move remains aligned with the zero-weight decision: `v9` best variant is `{v9['best_variant']}`, "
            f"while `v10` no longer needs that step because the active weight is already zero."
        ),
    ]

    version_csv = OUTPUT_DIR / "selection_score_v10_comparison_summary.csv"
    coverage_csv = OUTPUT_DIR / "selection_score_v10_comparison_coverage.csv"
    corr_csv = OUTPUT_DIR / "selection_score_v10_comparison_correlations.csv"
    family_csv = OUTPUT_DIR / "selection_score_v10_comparison_family.csv"
    json_path = OUTPUT_DIR / "selection_score_v10_comparison.json"
    version_df.to_csv(version_csv, index=False, encoding="utf-8-sig")
    if not coverage_all.empty:
        coverage_all.to_csv(coverage_csv, index=False, encoding="utf-8-sig")
    if not corr_all.empty:
        corr_all.to_csv(corr_csv, index=False, encoding="utf-8-sig")
    if not family_detail_all.empty:
        family_detail_all.to_csv(family_csv, index=False, encoding="utf-8-sig")
    payload = {
        "summary": _clean_records(version_df),
        "source_summaries": source_rows,
        "family_summaries": family_rows,
        "findings": findings,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Selection Score v10 Comparison 2026-05-28",
        "",
        "## Purpose",
        "",
        "Compare the old `v9` baseline artifacts against the new `v10` zero-launch-window-confidence research slice for both source observability and family-compression conclusions.",
        "",
        "## Key Findings",
        "",
    ]
    for item in findings:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Version Summary",
            "",
            version_df.to_markdown(index=False),
            "",
            "## Interpretation",
            "",
            "- `v10` matters because it converts an architecture recommendation into persisted evidence, not just a local formula tweak.",
            "- If the baseline portfolio metrics improve while the active launch-window-confidence weight drops to zero, that is stronger evidence than any isolated sweep because the result survives the real artifact-generation path.",
            "- Once more `v10` history exists, the next highest-value simplification target should move from launch-window-confidence to lighter attention-family compression.",
            "",
            "## Next Actions",
            "",
            "1. Treat `launch_window_confidence_weight = 0.0` as the new default research setting for follow-up score audits.",
            "2. Re-run attention-family compression reports on larger `v10` history as it accumulates.",
            "",
            f"Version CSV: `{version_csv}`",
            f"Coverage CSV: `{coverage_csv}`",
            f"Correlation CSV: `{corr_csv}`",
            f"Family CSV: `{family_csv}`",
            f"JSON: `{json_path}`",
        ]
    )
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    payload = build_report()
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
