from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = PROJECT_ROOT / ".cache" / "daily_focus_board_reviews"
OUTPUT_DIR = PROJECT_ROOT / ".cache" / "trading_system_attribution"
DOC_PATH = PROJECT_ROOT / "docs" / "selection_score_source_audit_2026-05-28.md"


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


def _load_review_details() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(REVIEW_DIR.glob("review_v9_*.pkl")):
        payload = _load_pickle(path)
        meta = dict(payload.get("meta", {}))
        details = payload.get("details")
        if not isinstance(details, pd.DataFrame) or details.empty:
            continue
        frame = details.copy()
        frame["review_path"] = str(path)
        frame["board_date"] = str(meta.get("board_date") or frame.get("board_date", ""))
        frame["review_date"] = str(meta.get("review_date") or frame.get("review_date", ""))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _top3_overlap(frame: pd.DataFrame, score_col: str) -> tuple[float, int]:
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


def build_report() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    details = _load_review_details()
    if details.empty:
        raise RuntimeError("No review_v9 detail files found for selection-score source audit.")

    for column in ["selection_score", *SOURCE_FEATURES]:
        details[column] = pd.to_numeric(details.get(column), errors="coerce")

    coverage_rows = []
    for column in SOURCE_FEATURES + MISSING_SOURCE_TERMS:
        coverage_rows.append(
            {
                "field": column,
                "present_in_v9_review": bool(column in details.columns),
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
        avg_overlap, exact_days = _top3_overlap(details, column)
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

    payload = {
        "review_rows": int(len(details)),
        "review_files": int(details["review_path"].nunique()) if "review_path" in details.columns else 0,
        "source_sites": [
            {
                "file": str(PROJECT_ROOT / "src" / "a_share_predictor" / "dashboard.py"),
                "line": 1887,
                "note": "full display-context selection_score construction",
            },
            {
                "file": str(PROJECT_ROOT / "src" / "a_share_predictor" / "dashboard.py"),
                "line": 2093,
                "note": "board replay selection_score construction",
            },
        ],
        "coverage": _clean_records(coverage_df),
        "correlations": _clean_records(corr_frame),
        "top3_overlap": _clean_records(overlap_df),
        "linear_projection": {
            "rows": projection_rows,
            "r2": projection_r2,
            "coefficients": _clean_records(coefficient_df),
        },
        "findings": [
            f"The strongest persisted same-slice linear relationship to `selection_score` is `{str(top_corr['field'])}` at correlation {float(top_corr['correlation_with_selection_score']):.3f}.",
            f"The highest top-3 overlap with `selection_score` among persisted source-family fields is `{str(top_overlap['field'])}` at {float(top_overlap['avg_top3_overlap']):.2f} / 3 names on average.",
            f"The persisted source-family projection explains about {projection_r2:.2%} of `selection_score` variance on {projection_rows} rows, but coefficient sign-flips indicate heavy multicollinearity rather than clean independent evidence.",
            f"`quant_score` remains only a secondary driver in persisted evidence (correlation {float(quant_corr['correlation_with_selection_score'].iloc[0]) if not quant_corr.empty else 0.0:.3f}), while `stage_score` remains weak (correlation {float(stage_corr['correlation_with_selection_score'].iloc[0]) if not stage_corr.empty else 0.0:.3f}).",
            "Several exact formula inputs are still not persisted into v9 review artifacts, so source-level optimization is partly bottlenecked by observability, not just by weak alpha.",
        ],
    }

    coverage_csv = OUTPUT_DIR / "selection_score_source_coverage.csv"
    corr_csv = OUTPUT_DIR / "selection_score_source_correlations.csv"
    overlap_csv = OUTPUT_DIR / "selection_score_source_overlap.csv"
    coeff_csv = OUTPUT_DIR / "selection_score_source_projection_coefficients.csv"
    json_path = OUTPUT_DIR / "selection_score_source_audit.json"
    coverage_df.to_csv(coverage_csv, index=False, encoding="utf-8-sig")
    corr_frame.to_csv(corr_csv, index=False, encoding="utf-8-sig")
    overlap_df.to_csv(overlap_csv, index=False, encoding="utf-8-sig")
    coefficient_df.to_csv(coeff_csv, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Selection Score Source Audit 2026-05-28",
        "",
        "## Purpose",
        "",
        "Audit the live `selection_score` construction path as a system-design problem: where the score is authored, which persisted inputs are most redundant, and which missing inputs still block a cleaner optimization pass.",
        "",
        "## Authoritative Source Sites",
        "",
        f"- [`src/a_share_predictor/dashboard.py:1887`]({PROJECT_ROOT / 'src' / 'a_share_predictor' / 'dashboard.py'}:1887): full display-context score construction with probability, attention, quant, launch-window, tomorrow-confidence, sector/fund/news, and manual adjustments.",
        f"- [`src/a_share_predictor/dashboard.py:2093`]({PROJECT_ROOT / 'src' / 'a_share_predictor' / 'dashboard.py'}:2093): replay-side simplified score construction used for board-level evaluation.",
        "",
        "## Coverage",
        "",
        f"- Review rows: {len(details)}",
        f"- Review files: {int(details['review_path'].nunique()) if 'review_path' in details.columns else 0}",
        "",
        coverage_df.to_markdown(index=False),
        "",
        "## Correlation With Selection Score",
        "",
        corr_frame.to_markdown(index=False),
        "",
        "## Top-3 Overlap Against Selection Score",
        "",
        overlap_df.to_markdown(index=False),
        "",
        "## Linear Projection",
        "",
        f"- Rows with full persisted source-family coverage: {projection_rows}",
        f"- R² from persisted source-family projection: {projection_r2:.2%}",
        "",
        coefficient_df.to_markdown(index=False),
        "",
        "## Key Findings",
        "",
    ]
    for item in payload["findings"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The live `selection_score` is not a clean stack of independent evidence. The persisted source-family terms already explain almost all score variance, but they do so with obvious redundancy and coefficient instability.",
            "- `probability_up`, `attention_score`, and `enhanced_attention_score` form an especially stacked attention family. They are all strongly tied to `selection_score`, which means adding more weight inside that family is likely to reshuffle the same narrative signal rather than add new alpha.",
            "- `launch_window_score` and `launch_window_confidence` are structurally close to the realized top-3 picks, but earlier realized-return tests showed launch-window works better as context than as a standalone ranker. That combination is exactly what an over-layered gate family looks like.",
            "- `quant_score` is not driving the score nearly as strongly as the attention and launch families, which fits the earlier conclusion that quant is better treated as mild confirmation or damage control than as the main sorting engine.",
            "- `stage_score` remains too weak in persisted evidence to justify prominent default weight inside the ranking path.",
            "",
            "## Optimization Guidance",
            "",
            "1. Collapse the attention family into a cleaner core path before adding any new overlays. At minimum, treat `probability_up`, `attention_score`, and `enhanced_attention_score` as one correlated cluster, not as three independent alpha sources.",
            "2. Keep launch-window as structure/risk context, but stop trying to improve the system by adding more launch weight. The source audit and the realized-return experiments already say it is close to saturation.",
            "3. Do not promote `quant_score` or `stage_score` into primary ranking roles. If quant is used, keep it as a mild filter candidate; if stage remains this weak, demote it from default weighting until stronger evidence appears.",
            "4. Close the observability gap before any deeper source-level optimization: persist `predicted_upside_pct`, `tomorrow_plan_confidence`, `sector_score`, `fund_score`, `news_score`, and the adjustment terms into v9 review artifacts so future ablations can audit the real formula instead of a partial shadow.",
            "",
            f"Coverage CSV: `{coverage_csv}`",
            f"Correlation CSV: `{corr_csv}`",
            f"Overlap CSV: `{overlap_csv}`",
            f"Projection CSV: `{coeff_csv}`",
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
