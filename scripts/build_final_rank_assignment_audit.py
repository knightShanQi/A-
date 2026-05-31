from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / ".cache" / "trading_system_attribution"
DOC_PATH = PROJECT_ROOT / "docs" / "final_rank_assignment_audit_2026-05-28.md"


SITE_ROWS = [
    {
        "file": r"E:\openclaw\src\a_share_predictor\dashboard.py",
        "line": 5004,
        "site_type": "assignment_alias",
        "impact": "decision_source",
        "summary": "Quick-board fallback path assigns final_rank_score directly from enhanced_attention_score.",
    },
    {
        "file": r"E:\openclaw\src\a_share_predictor\dashboard.py",
        "line": 5377,
        "site_type": "assignment_alias",
        "impact": "decision_source",
        "summary": "Latest market quick-board path assigns final_rank_score directly from enhanced_attention_score.",
    },
    {
        "file": r"E:\openclaw\src\a_share_predictor\dashboard.py",
        "line": 1261,
        "site_type": "display_fallback",
        "impact": "presentation",
        "summary": "Merged display payload backfills final_rank_score from ranking_score / enhanced_attention_score for UI use.",
    },
    {
        "file": r"E:\openclaw\src\a_share_predictor\dashboard.py",
        "line": 1606,
        "site_type": "display_fallback",
        "impact": "presentation",
        "summary": "Symbol detail payload derives final_rank_score from ranking_score with enhanced_attention fallback.",
    },
    {
        "file": r"E:\openclaw\src\a_share_predictor\api_service.py",
        "line": 347,
        "site_type": "normalization_fallback",
        "impact": "decision_passthrough",
        "summary": "API probability contract uses provided final_rank_score or falls back to ranking_score / enhanced_attention_score / attention_score.",
    },
    {
        "file": r"E:\openclaw\src\a_share_predictor\api_service.py",
        "line": 989,
        "site_type": "detail_passthrough",
        "impact": "presentation",
        "summary": "Safe symbol detail payload defaults final_rank_score from cached row rank_score when missing.",
    },
    {
        "file": r"E:\openclaw\src\a_share_predictor\daily_review.py",
        "line": 1150,
        "site_type": "snapshot_fallback",
        "impact": "persistence",
        "summary": "Snapshot persistence backfills final_rank_score from ranking_score -> enhanced_attention_score -> attention_score.",
    },
    {
        "file": r"E:\openclaw\src\a_share_predictor\daily_review.py",
        "line": 1402,
        "site_type": "review_passthrough",
        "impact": "persistence",
        "summary": "Review detail persistence carries final_rank_score from snapshot row or ranking_score fallback.",
    },
    {
        "file": r"E:\openclaw\src\a_share_predictor\dashboard.py",
        "line": 6285,
        "site_type": "display_usage",
        "impact": "presentation",
        "summary": "Focus-board summary table exposes final_rank_score as a displayed metric.",
    },
    {
        "file": r"E:\openclaw\src\a_share_predictor\dashboard.py",
        "line": 6291,
        "site_type": "display_usage",
        "impact": "presentation",
        "summary": "Focus-board detail table exposes final_rank_score alongside ranking_score for inspection.",
    },
]


def build_report() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sites_df = pd.DataFrame(SITE_ROWS)
    summary_df = (
        sites_df.groupby(["site_type", "impact"], dropna=False)
        .size()
        .reset_index(name="site_count")
        .sort_values(["site_count", "site_type"], ascending=[False, True])
        .reset_index(drop=True)
    )

    alias_sources = sites_df.loc[sites_df["site_type"].eq("assignment_alias")]
    payload = {
        "site_rows": SITE_ROWS,
        "summary": sites_df.groupby("impact").size().to_dict(),
        "findings": [
            "The only true source assignments for final_rank_score in the current quick-board generation paths are direct aliases from enhanced_attention_score.",
            "Everything downstream in api_service and daily_review mostly preserves or backfills that value rather than creating a differentiated score.",
            "Current decision risk therefore sits upstream in dashboard quick-board construction, not in API normalization or snapshot persistence.",
            "As long as those upstream alias assignments remain, any downstream tuning of final_rank_score is structurally incapable of producing a distinct ranking path.",
        ],
        "alias_assignment_count": int(len(alias_sources)),
    }

    summary_csv = OUTPUT_DIR / "final_rank_assignment_summary.csv"
    sites_csv = OUTPUT_DIR / "final_rank_assignment_sites.csv"
    json_path = OUTPUT_DIR / "final_rank_assignment_audit.json"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    sites_df.to_csv(sites_csv, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Final Rank Assignment Audit 2026-05-28",
        "",
        "## Purpose",
        "",
        "Locate where `final_rank_score` is truly created versus where it is only displayed, normalized, or persisted.",
        "",
        "## Key Findings",
        "",
    ]
    for item in payload["findings"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Site Summary",
            "",
            summary_df.to_markdown(index=False),
            "",
            "## Site Inventory",
            "",
            sites_df.to_markdown(index=False),
            "",
            "## Interpretation",
            "",
            "- The engineering bottleneck is concentrated in the upstream quick-board builders, where `final_rank_score` is still assigned as a plain alias of `enhanced_attention_score`.",
            "- Downstream consumers mostly preserve that alias, so removing or rebuilding `final_rank_score` should start at the source assignments rather than at display tables or persistence hooks.",
            "- This also explains why the final-rank construction audit showed exact equality on every persisted snapshot row.",
            "",
            "## Next Actions",
            "",
            "1. Remove or rename the alias assignments in the quick-board builders if `final_rank_score` is not intended to be a distinct signal.",
            "2. If it is intended to be distinct, replace those assignments with an explicit composite formula and rerun the construction and portfolio audits.",
            "3. Avoid using downstream fallback presence as evidence that final_rank_score has independent alpha; current code shows it does not.",
            "",
            f"Summary CSV: `{summary_csv}`",
            f"Sites CSV: `{sites_csv}`",
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
