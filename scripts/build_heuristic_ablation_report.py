from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_VALIDATION_DIR = PROJECT_ROOT / ".cache" / "ten_year_model_strategy_validation"
V3_DIR = PROJECT_ROOT / ".cache" / "ten_year_market_regime_v3"
BULL_DIR = PROJECT_ROOT / ".cache" / "ten_year_bull_market_rank_score"
OUTPUT_DIR = PROJECT_ROOT / ".cache" / "trading_system_attribution"
DOC_PATH = PROJECT_ROOT / "docs" / "heuristic_ablation_2026-05-28.md"


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_rule(records: list[dict[str, object]], rule: str) -> dict[str, object]:
    for row in records:
        if str(row.get("rule")) == rule:
            return dict(row)
    raise KeyError(f"Rule not found: {rule}")


def _clean_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in frame.to_dict("records"):
        cleaned: dict[str, object] = {}
        for key, value in row.items():
            cleaned[key] = None if pd.isna(value) else value
        records.append(cleaned)
    return records


def _metric(row: dict[str, object], name: str, fallback: str | None = None) -> float | None:
    value = row.get(name)
    if value is None and fallback is not None:
        value = row.get(fallback)
    if value is None:
        return None
    return float(value)


def _row(
    *,
    family: str,
    stage: str,
    parent_stage: str | None,
    rule: str,
    source_path: Path,
    row: dict[str, object],
    note: str,
) -> dict[str, object]:
    return {
        "family": family,
        "stage": stage,
        "parent_stage": parent_stage,
        "rule": rule,
        "source_path": str(source_path),
        "active_days": row.get("active_days"),
        "coverage_pct": row.get("coverage_pct"),
        "selected_rows": row.get("selected_rows"),
        "portfolio_trade_count": row.get("portfolio_trade_count", row.get("evaluated_trade_count")),
        "portfolio_win_rate": row.get("portfolio_win_rate", row.get("trade_win_rate")),
        "portfolio_avg_net_return": _metric(row, "portfolio_avg_net_return", "avg_trade_return"),
        "portfolio_cumulative_return": _metric(row, "portfolio_cumulative_return"),
        "portfolio_annualized_return": _metric(row, "portfolio_annualized_return", "annualized_return"),
        "portfolio_max_drawdown": _metric(row, "portfolio_max_drawdown", "max_drawdown"),
        "note": note,
    }


def build_report() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    validation_summary = _load_json(MODEL_VALIDATION_DIR / "summary.json")
    v3_summary = _load_json(V3_DIR / "v3_summary.json")
    bull_summary = _load_json(BULL_DIR / "bull_market_rank_score80_summary.json")

    rows: list[dict[str, object]] = [
        _row(
            family="raw_signal_baselines",
            stage="model_only_top3",
            parent_stage=None,
            rule="model_only_top3",
            source_path=MODEL_VALIDATION_DIR / "summary.json",
            row=_find_rule(validation_summary["model_rules"], "model_only_top3"),
            note="Pure model ranking with no heuristic candidate filter.",
        ),
        _row(
            family="raw_signal_baselines",
            stage="strategy_only_top3",
            parent_stage=None,
            rule="strategy_only_top3",
            source_path=MODEL_VALIDATION_DIR / "summary.json",
            row=_find_rule(validation_summary["strategy_rules"], "strategy_only_top3"),
            note="Pure heuristic candidate priority with no model rescue.",
        ),
        _row(
            family="combined_overlay_path",
            stage="combined_baseline",
            parent_stage=None,
            rule="existing_v2_top3_score68_full",
            source_path=V3_DIR / "v3_summary.json",
            row=_find_rule(v3_summary["existing_overlay_rules"], "existing_v2_top3_score68_full"),
            note="Historical combined baseline after realistic portfolio simulation.",
        ),
        _row(
            family="combined_overlay_path",
            stage="combined_trend_gate",
            parent_stage="combined_baseline",
            rule="existing_v3_trend_top3",
            source_path=V3_DIR / "v3_summary.json",
            row=_find_rule(v3_summary["existing_overlay_rules"], "existing_v3_trend_top3"),
            note="Baseline candidates gated by trend-only market state.",
        ),
        _row(
            family="combined_overlay_path",
            stage="combined_trend_flow_gate",
            parent_stage="combined_baseline",
            rule="existing_v3_trend_flow_top3",
            source_path=V3_DIR / "v3_summary.json",
            row=_find_rule(v3_summary["existing_overlay_rules"], "existing_v3_trend_flow_top3"),
            note="Baseline candidates gated by trend+flow market state.",
        ),
        _row(
            family="combined_overlay_path",
            stage="combined_full_green_gate",
            parent_stage="combined_baseline",
            rule="existing_v3_full_green_top3",
            source_path=V3_DIR / "v3_summary.json",
            row=_find_rule(v3_summary["existing_overlay_rules"], "existing_v3_full_green_top3"),
            note="Baseline candidates gated by strict full-green state.",
        ),
        _row(
            family="native_candidate_path",
            stage="native_v2_score68",
            parent_stage=None,
            rule="v2_score68_top3",
            source_path=V3_DIR / "v3_summary.json",
            row=_find_rule(v3_summary["rules"], "v2_score68_top3"),
            note="Native V2 candidate family without extra state tightening.",
        ),
        _row(
            family="native_candidate_path",
            stage="native_trend_gate",
            parent_stage="native_v2_score68",
            rule="v3_trend_top3",
            source_path=V3_DIR / "v3_summary.json",
            row=_find_rule(v3_summary["rules"], "v3_trend_top3"),
            note="Native candidate family under trend-only state.",
        ),
        _row(
            family="native_candidate_path",
            stage="native_trend_flow_gate",
            parent_stage="native_v2_score68",
            rule="v3_trend_flow_top3",
            source_path=V3_DIR / "v3_summary.json",
            row=_find_rule(v3_summary["rules"], "v3_trend_flow_top3"),
            note="Native candidate family under trend+flow state.",
        ),
        _row(
            family="native_candidate_path",
            stage="native_full_green_gate",
            parent_stage="native_v2_score68",
            rule="v3_full_green_top3",
            source_path=V3_DIR / "v3_summary.json",
            row=_find_rule(v3_summary["rules"], "v3_full_green_top3"),
            note="Native candidate family under strict full-green state.",
        ),
        _row(
            family="native_candidate_path",
            stage="native_full_green_plus_pause",
            parent_stage="native_full_green_gate",
            rule="v3_full_green_top3_pause6_10d",
            source_path=V3_DIR / "v3_summary.json",
            row=_find_rule(v3_summary["rules"], "v3_full_green_top3_pause6_10d"),
            note="Same strict full-green path plus pause overlay.",
        ),
        _row(
            family="bull_rank_research",
            stage="bull_rank_sorted",
            parent_stage=None,
            rule="no_bull_filter_v3_full_green_top3_rank_score_ge_80_rank_sorted_as_cash",
            source_path=BULL_DIR / "bull_market_rank_score80_summary.json",
            row=_find_rule(
                bull_summary["rules"],
                "no_bull_filter_v3_full_green_top3_rank_score_ge_80_rank_sorted_as_cash",
            ),
            note="Research-only rank overlay sorted by rebuilt rank score.",
        ),
        _row(
            family="bull_rank_research",
            stage="bull_model_sorted",
            parent_stage="bull_rank_sorted",
            rule="no_bull_filter_v3_full_green_top3_rank_score_ge_80_model_sorted_as_cash",
            source_path=BULL_DIR / "bull_market_rank_score80_summary.json",
            row=_find_rule(
                bull_summary["rules"],
                "no_bull_filter_v3_full_green_top3_rank_score_ge_80_model_sorted_as_cash",
            ),
            note="Same research path sorted by model score instead of rebuilt rank score.",
        ),
    ]

    frame = pd.DataFrame(rows)
    metrics = [
        "coverage_pct",
        "selected_rows",
        "portfolio_trade_count",
        "portfolio_win_rate",
        "portfolio_avg_net_return",
        "portfolio_cumulative_return",
        "portfolio_annualized_return",
        "portfolio_max_drawdown",
    ]
    for col in metrics:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    parent_lookup = frame.set_index("stage")
    frame["delta_vs_parent_annualized"] = frame.apply(
        lambda row: row["portfolio_annualized_return"] - parent_lookup.loc[row["parent_stage"], "portfolio_annualized_return"]
        if pd.notna(row["portfolio_annualized_return"]) and isinstance(row["parent_stage"], str) and row["parent_stage"] in parent_lookup.index
        else None,
        axis=1,
    )
    frame["delta_vs_parent_drawdown"] = frame.apply(
        lambda row: row["portfolio_max_drawdown"] - parent_lookup.loc[row["parent_stage"], "portfolio_max_drawdown"]
        if pd.notna(row["portfolio_max_drawdown"]) and isinstance(row["parent_stage"], str) and row["parent_stage"] in parent_lookup.index
        else None,
        axis=1,
    )
    frame["delta_vs_parent_trade_count"] = frame.apply(
        lambda row: row["portfolio_trade_count"] - parent_lookup.loc[row["parent_stage"], "portfolio_trade_count"]
        if pd.notna(row["portfolio_trade_count"]) and isinstance(row["parent_stage"], str) and row["parent_stage"] in parent_lookup.index
        else None,
        axis=1,
    )
    frame["annualized_per_100_trades"] = (
        frame["portfolio_annualized_return"] / frame["portfolio_trade_count"] * 100.0
    )

    csv_path = OUTPUT_DIR / "heuristic_ablation.csv"
    json_path = OUTPUT_DIR / "heuristic_ablation.json"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(_clean_records(frame), ensure_ascii=False, indent=2), encoding="utf-8")

    combined_full_green = frame.loc[frame["stage"].eq("combined_full_green_gate")].iloc[0]
    native_full_green = frame.loc[frame["stage"].eq("native_full_green_gate")].iloc[0]
    native_pause = frame.loc[frame["stage"].eq("native_full_green_plus_pause")].iloc[0]
    bull_rank = frame.loc[frame["stage"].eq("bull_rank_sorted")].iloc[0]
    bull_model = frame.loc[frame["stage"].eq("bull_model_sorted")].iloc[0]
    model_only = frame.loc[frame["stage"].eq("model_only_top3")].iloc[0]
    strategy_only = frame.loc[frame["stage"].eq("strategy_only_top3")].iloc[0]

    findings = [
        (
            "Raw model ranking and raw heuristic ranking both fail badly before market-state filtering, "
            f"with portfolio annualized returns of {model_only['portfolio_annualized_return'] * 100:.2f}% and "
            f"{strategy_only['portfolio_annualized_return'] * 100:.2f}% respectively."
        ),
        (
            "Adding the strict full-green gate to the historical combined baseline improves annualized return by "
            f"{combined_full_green['delta_vs_parent_annualized'] * 100:.2f} percentage points and reduces trade count by "
            f"{abs(int(combined_full_green['delta_vs_parent_trade_count']))}, but only reaches "
            f"{combined_full_green['portfolio_annualized_return'] * 100:.2f}% annualized."
        ),
        (
            "Switching from the historical combined candidate family to the native V3 full-green candidate family adds another "
            f"{(native_full_green['portfolio_annualized_return'] - combined_full_green['portfolio_annualized_return']) * 100:.2f} percentage points "
            "of annualized return, which is larger than the gain from most overlay tweaks."
        ),
        (
            "The pause overlay barely changes the native strict-gate result: annualized return moves by "
            f"{native_pause['delta_vs_parent_annualized'] * 100:.2f} percentage points."
        ),
        (
            "Within the bull/rank research path, rebuilt rank-score sorting beats model-score sorting by "
            f"{(bull_rank['portfolio_annualized_return'] - bull_model['portfolio_annualized_return']) * 100:.2f} percentage points, "
            "which is evidence that the current model ordering is not the strongest signal even inside the same gated pool."
        ),
    ]

    lines = [
        "# Heuristic Ablation 2026-05-28",
        "",
        "## Purpose",
        "",
        "Quantify which heuristic layers actually improve the unified portfolio NAV and which ones mainly add complexity.",
        "",
        "## Key Findings",
        "",
    ]
    for item in findings:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Family Comparison",
            "",
            frame[
                [
                    "family",
                    "stage",
                    "parent_stage",
                    "portfolio_trade_count",
                    "portfolio_annualized_return",
                    "portfolio_max_drawdown",
                    "delta_vs_parent_annualized",
                    "delta_vs_parent_trade_count",
                ]
            ].to_markdown(index=False),
            "",
            "## Interpretation",
            "",
            "- The first-order gain comes from shrinking exposure and candidate-family quality, not from stacking more heuristic scorers on top of the same pool.",
            "- The strict full-green gate is useful, but it is a damage-control layer, not a sufficient alpha source by itself.",
            "- Candidate-family replacement matters more than pause-style micro-overlays in current evidence.",
            "- The strategy heuristic layer remains the weakest standalone component and should not be allowed to dominate ranking.",
            "",
            "## Actionable Order",
            "",
            "1. Keep `native_v3_full_green_top3` as the live research baseline.",
            "2. Retire `combined_top3_score_ge_68_full` from default comparison and stop threshold-tuning it.",
            "3. Treat pause overlays as optional polish only after the base family and market-state gate are proven.",
            "4. Rebuild model training around cross-sectional ranking because even inside the same gated pool, model sorting loses to rebuilt rank sorting.",
            "",
            f"CSV: `{csv_path}`",
            f"JSON: `{json_path}`",
        ]
    )
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "doc_path": str(DOC_PATH),
        "rows": _clean_records(frame),
        "findings": findings,
    }
    (OUTPUT_DIR / "heuristic_ablation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    payload = build_report()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
