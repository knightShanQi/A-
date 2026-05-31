from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_VALIDATION_DIR = PROJECT_ROOT / ".cache" / "ten_year_model_strategy_validation"
V3_DIR = PROJECT_ROOT / ".cache" / "ten_year_market_regime_v3"
BULL_DIR = PROJECT_ROOT / ".cache" / "ten_year_bull_market_rank_score"
TOP10_DIR = PROJECT_ROOT / ".cache" / "strategy_model_top10_backtest"
OUTPUT_DIR = PROJECT_ROOT / ".cache" / "trading_system_attribution"
DOC_PATH = PROJECT_ROOT / "docs" / "trading_system_attribution_2026-05-28.md"


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
        cleaned_row: dict[str, object] = {}
        for key, value in row.items():
            cleaned_row[key] = None if pd.isna(value) else value
        records.append(cleaned_row)
    return records


def _row(
    *,
    strategy_id: str,
    family: str,
    source_summary: str,
    date_from: str,
    date_to: str,
    sample_note: str,
    row: dict[str, object],
    legacy_fields: bool = True,
) -> dict[str, object]:
    result = {
        "strategy_id": strategy_id,
        "family": family,
        "source_summary": source_summary,
        "date_from": date_from,
        "date_to": date_to,
        "sample_note": sample_note,
        "rule": row.get("rule", strategy_id),
        "coverage_pct": row.get("coverage_pct"),
        "selected_rows": row.get("selected_rows"),
        "portfolio_trade_count": row.get("portfolio_trade_count", row.get("evaluated_trade_count")),
        "portfolio_win_rate": row.get("portfolio_win_rate", row.get("model_win_rate")),
        "portfolio_avg_net_return": row.get("portfolio_avg_net_return", row.get("avg_hold_return")),
        "portfolio_cumulative_return": row.get("portfolio_cumulative_return"),
        "portfolio_annualized_return": row.get("portfolio_annualized_return", row.get("annualized_return")),
        "portfolio_max_drawdown": row.get("portfolio_max_drawdown", row.get("max_drawdown")),
        "portfolio_ending_equity": row.get("portfolio_ending_equity", row.get("ending_equity")),
    }
    if legacy_fields:
        result["legacy_annualized_return"] = row.get("annualized_return")
        result["legacy_max_drawdown"] = row.get("max_drawdown")
        result["legacy_ending_equity"] = row.get("ending_equity")
    return result


def build_attribution() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    validation_summary = _load_json(MODEL_VALIDATION_DIR / "summary.json")
    v3_summary = _load_json(V3_DIR / "v3_summary.json")
    bull_summary = _load_json(BULL_DIR / "bull_market_rank_score80_summary.json")
    top10_summary = _load_json(TOP10_DIR / "summary.json")

    rows: list[dict[str, object]] = []
    rows.append(
        _row(
            strategy_id="combined_baseline_top3_score68",
            family="combined_baseline",
            source_summary=str(V3_DIR / "v3_summary.json"),
            date_from=str(v3_summary["date_from"]),
            date_to=str(v3_summary["date_to"]),
            sample_note="Ten-year baseline combined selection, re-evaluated with unified portfolio engine.",
            row=_find_rule(v3_summary["existing_overlay_rules"], "existing_v2_top3_score68_full"),
        )
    )
    rows.append(
        _row(
            strategy_id="combined_plus_market_regime",
            family="market_regime_overlay",
            source_summary=str(V3_DIR / "v3_summary.json"),
            date_from=str(v3_summary["date_from"]),
            date_to=str(v3_summary["date_to"]),
            sample_note="Same combined baseline after V3 full_green market filter.",
            row=_find_rule(v3_summary["existing_overlay_rules"], "existing_v3_full_green_top3"),
        )
    )
    rows.append(
        _row(
            strategy_id="native_v3_full_green_top3",
            family="native_market_regime",
            source_summary=str(V3_DIR / "v3_summary.json"),
            date_from=str(v3_summary["date_from"]),
            date_to=str(v3_summary["date_to"]),
            sample_note="Native V3 candidate generation plus full_green top3.",
            row=_find_rule(v3_summary["rules"], "v3_full_green_top3"),
        )
    )
    rows.append(
        _row(
            strategy_id="native_v3_full_green_top3_pause",
            family="native_market_regime",
            source_summary=str(V3_DIR / "v3_summary.json"),
            date_from=str(v3_summary["date_from"]),
            date_to=str(v3_summary["date_to"]),
            sample_note="Native V3 full_green top3 with pause overlay.",
            row=_find_rule(v3_summary["rules"], "v3_full_green_top3_pause6_10d"),
        )
    )
    rows.append(
        _row(
            strategy_id="bull_rank_sorted_as_cash",
            family="bull_rank_overlay",
            source_summary=str(BULL_DIR / "bull_market_rank_score80_summary.json"),
            date_from=str(bull_summary["date_from"]),
            date_to=str(bull_summary["date_to"]),
            sample_note="Bull/rank overlay, full calendar as cash, rebuilt rank score sort.",
            row=_find_rule(bull_summary["rules"], "no_bull_filter_v3_full_green_top3_rank_score_ge_80_rank_sorted_as_cash"),
        )
    )
    rows.append(
        _row(
            strategy_id="bull_model_sorted_as_cash",
            family="bull_rank_overlay",
            source_summary=str(BULL_DIR / "bull_market_rank_score80_summary.json"),
            date_from=str(bull_summary["date_from"]),
            date_to=str(bull_summary["date_to"]),
            sample_note="Bull/rank overlay, full calendar as cash, model-score sort.",
            row=_find_rule(bull_summary["rules"], "no_bull_filter_v3_full_green_top3_rank_score_ge_80_model_sorted_as_cash"),
        )
    )
    rows.append(
        _row(
            strategy_id="recent_top10_6m",
            family="recent_top10",
            source_summary=str(TOP10_DIR / "summary.json"),
            date_from=str(top10_summary["date_from"]),
            date_to=str(top10_summary["date_to"]),
            sample_note="Six-month recent sample only; not directly comparable to ten-year studies.",
            row={
                "rule": "strategy_model_top10_recent",
                "coverage_pct": None,
                "selected_rows": top10_summary.get("selected_rows"),
                "evaluated_trade_count": top10_summary.get("evaluated_trade_count"),
                "model_win_rate": top10_summary.get("model_win_rate"),
                "avg_hold_return": top10_summary.get("avg_hold_return"),
                "annualized_return": top10_summary.get("annualized_return"),
                "max_drawdown": top10_summary.get("max_drawdown"),
                "ending_equity": top10_summary.get("ending_equity"),
            },
            legacy_fields=False,
        )
    )

    frame = pd.DataFrame(rows)
    baseline = frame.loc[frame["strategy_id"].eq("combined_baseline_top3_score68")].iloc[0]
    frame["delta_vs_baseline_annualized"] = pd.to_numeric(frame["portfolio_annualized_return"], errors="coerce") - float(
        baseline["portfolio_annualized_return"]
    )
    frame["delta_vs_baseline_drawdown"] = pd.to_numeric(frame["portfolio_max_drawdown"], errors="coerce") - float(
        baseline["portfolio_max_drawdown"]
    )
    frame["delta_vs_baseline_cumulative"] = pd.to_numeric(frame["portfolio_cumulative_return"], errors="coerce") - float(
        baseline["portfolio_cumulative_return"]
    )

    csv_path = OUTPUT_DIR / "cross_strategy_attribution.csv"
    json_path = OUTPUT_DIR / "cross_strategy_attribution.json"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(_clean_records(frame), ensure_ascii=False, indent=2), encoding="utf-8")

    key_findings = [
        f"Baseline combined rule `combined_top3_score_ge_68_full` drops from legacy `+3.44%` annualized to portfolio `{float(baseline['portfolio_annualized_return']) * 100:.2f}%`, with portfolio max drawdown `{abs(float(baseline['portfolio_max_drawdown'])) * 100:.2f}%`.",
        f"Applying the V3 market filter to the same combined baseline (`existing_v3_full_green_top3`) lifts portfolio annualized to `{float(frame.loc[frame['strategy_id'].eq('combined_plus_market_regime'), 'portfolio_annualized_return'].iloc[0]) * 100:.2f}%` and cuts max drawdown to `{abs(float(frame.loc[frame['strategy_id'].eq('combined_plus_market_regime'), 'portfolio_max_drawdown'].iloc[0])) * 100:.2f}%`, mainly by reducing trades.",
        f"Native V3 candidate generation (`v3_full_green_top3`) is the best ten-year migrated path in current evidence at `{float(frame.loc[frame['strategy_id'].eq('native_v3_full_green_top3'), 'portfolio_annualized_return'].iloc[0]) * 100:.2f}%` annualized with `{abs(float(frame.loc[frame['strategy_id'].eq('native_v3_full_green_top3'), 'portfolio_max_drawdown'].iloc[0])) * 100:.2f}%` max drawdown.",
        f"The rebuilt bull/rank path still looks strong under legacy averaging (`+10.99%` annualized), but under unified portfolio NAV it is `{float(frame.loc[frame['strategy_id'].eq('bull_rank_sorted_as_cash'), 'portfolio_annualized_return'].iloc[0]) * 100:.2f}%`, which is effectively flat to slightly negative.",
        f"The recent top10 path shows `{float(frame.loc[frame['strategy_id'].eq('recent_top10_6m'), 'portfolio_annualized_return'].iloc[0]) * 100:.2f}%` annualized, but it only covers `{frame.loc[frame['strategy_id'].eq('recent_top10_6m'), 'date_from'].iloc[0]} to {frame.loc[frame['strategy_id'].eq('recent_top10_6m'), 'date_to'].iloc[0]}` and should be treated as a short-window observation, not ten-year proof.",
    ]

    lines = [
        "# Trading System Attribution 2026-05-28",
        "",
        "## Unified Portfolio Attribution",
        "",
        "This addendum compares representative strategy paths under the unified portfolio backtest engine wherever available.",
        "",
        "## Key Findings",
        "",
    ]
    for item in key_findings:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Attribution Table",
            "",
            frame[
                [
                    "strategy_id",
                    "family",
                    "date_from",
                    "date_to",
                    "portfolio_trade_count",
                    "portfolio_annualized_return",
                    "portfolio_max_drawdown",
                    "portfolio_cumulative_return",
                    "sample_note",
                ]
            ].to_markdown(index=False),
            "",
            "## Interpretation",
            "",
            "- The largest drop happens when legacy average-selected-return curves are replaced by capital-constrained portfolio NAV. This is strongest in the baseline combined rule and in the rebuilt bull/rank path.",
            "- Market-state filtering does help, but mostly through exposure control and drawdown compression, not through strong per-trade alpha.",
            "- Native V3 full_green selection currently dominates other ten-year migrated paths, but its real portfolio annualized return is still only low-single-digit, which confirms the audit conclusion that the system's core alpha is weak.",
            "- The six-month top10 path is promising but not yet comparable. It needs a longer-window rerun under the same engine before it should influence architectural conclusions.",
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
        "key_findings": key_findings,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    payload = build_attribution()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
