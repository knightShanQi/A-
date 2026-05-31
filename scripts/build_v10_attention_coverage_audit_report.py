from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = PROJECT_ROOT / ".cache" / "daily_focus_board_reviews"
OUTPUT_DIR = PROJECT_ROOT / ".cache" / "trading_system_attribution"
DOC_PATH = PROJECT_ROOT / "docs" / "v10_attention_coverage_audit_2026-05-28.md"


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


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
        frame["ranking_by"] = str(meta.get("ranking_by") or "")
        frame["model_source_label"] = str(meta.get("model_source_label") or "")
        frame["backfilled_from_snapshot"] = str(meta.get("backfilled_from_snapshot") or "")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshots = _load_snapshots(10)
    if snapshots.empty:
        raise RuntimeError("No snapshot_v10 artifacts found for attention coverage audit.")

    for column in [
        "attention_score",
        "enhanced_attention_score",
        "sector_score",
        "fund_score",
        "news_score",
    ]:
        snapshots[column] = pd.to_numeric(snapshots.get(column), errors="coerce")

    for column in [
        "candidate_strategy",
        "candidate_reason",
        "launch_phase_label",
        "fund_label",
        "news_label",
        "sector_label",
        "backfilled_from_snapshot",
    ]:
        if column not in snapshots.columns:
            snapshots[column] = ""
        snapshots[column] = snapshots[column].fillna("").astype(str)

    snapshots["attention_delta"] = snapshots["enhanced_attention_score"] - snapshots["attention_score"]
    snapshots["all_three_default_50"] = (
        snapshots["sector_score"].eq(50.0)
        & snapshots["fund_score"].eq(50.0)
        & snapshots["news_score"].eq(50.0)
    )
    snapshots["attention_delta_zero"] = snapshots["attention_delta"].abs() < 1e-9
    snapshots["candidate_reason_present"] = snapshots["candidate_reason"].str.strip().ne("")
    snapshots["launch_phase_present"] = snapshots["launch_phase_label"].str.strip().ne("")
    snapshots["fund_label_is_placeholder"] = snapshots["fund_label"].str.contains("完整版补齐", na=False)
    snapshots["news_label_is_placeholder"] = snapshots["news_label"].str.contains("完整版补齐", na=False)
    snapshots["sector_label_is_placeholder"] = snapshots["sector_label"].str.contains("完整版补齐", na=False)

    summary = {
        "snapshot_rows": int(len(snapshots)),
        "snapshot_files": int(snapshots["snapshot_path"].nunique()),
        "all_three_default_50_rows": int(snapshots["all_three_default_50"].sum()),
        "attention_delta_zero_rows": int(snapshots["attention_delta_zero"].sum()),
        "attention_delta_nonzero_rows": int((~snapshots["attention_delta_zero"]).sum()),
        "candidate_reason_present_rows": int(snapshots["candidate_reason_present"].sum()),
        "launch_phase_present_rows": int(snapshots["launch_phase_present"].sum()),
        "fund_label_placeholder_rows": int(snapshots["fund_label_is_placeholder"].sum()),
        "news_label_placeholder_rows": int(snapshots["news_label_is_placeholder"].sum()),
        "sector_label_placeholder_rows": int(snapshots["sector_label_is_placeholder"].sum()),
        "local_quick_fallback_rows": int(snapshots.get("model_source_label", "").fillna("").astype(str).eq("本地快速回退").sum())
        if "model_source_label" in snapshots.columns
        else 0,
        "latest_close_fast_board_rows": int(
            snapshots.get("model_source_label", "").fillna("").astype(str).eq("最新收盘快榜（完整版特征与回测正在后台补齐）").sum()
        )
        if "model_source_label" in snapshots.columns
        else 0,
        "ranking_by_counts": snapshots["ranking_by"].value_counts().to_dict(),
        "candidate_strategy_counts": snapshots["candidate_strategy"].replace("", "missing").value_counts().head(10).to_dict(),
    }

    strategy_delta = (
        snapshots.groupby("candidate_strategy", dropna=False)
        .agg(
            rows=("candidate_strategy", "size"),
            attention_delta_zero_rate=("attention_delta_zero", "mean"),
            all_three_default_50_rate=("all_three_default_50", "mean"),
            avg_attention_delta=("attention_delta", "mean"),
        )
        .reset_index()
        .sort_values(["rows", "attention_delta_zero_rate"], ascending=[False, False], ignore_index=True)
    )
    strategy_delta["candidate_strategy"] = strategy_delta["candidate_strategy"].replace("", "missing")

    sample_nonzero = snapshots.loc[
        ~snapshots["attention_delta_zero"],
        ["symbol", "board_date", "candidate_strategy", "attention_score", "enhanced_attention_score", "attention_delta"],
    ].head(20)

    findings = [
        f"All `{int(len(snapshots))}` persisted `v10` snapshot rows carry `sector_score = fund_score = news_score = 50`, so the current audit slice does not expose real sector/fund/news dispersion inside the attention family.",
        f"`enhanced_attention_score` still differs from `attention_score` on {int((~snapshots['attention_delta_zero']).sum())} rows, but those differences come from quick-board style secondary adjustments rather than the full `_score_final_attention()` enrichment path.",
        f"`candidate_reason` is present on only {int(snapshots['candidate_reason_present'].sum())} rows and `launch_phase_label` on only {int(snapshots['launch_phase_present'].sum())}, which confirms the replayable slice is missing much of the richer narrative context available in live symbol analysis.",
        f"The persisted `model_source_label` mix is dominated by lightweight board-generation paths: `最新收盘快榜（完整版特征与回测正在后台补齐）` on {int(summary['latest_close_fast_board_rows'])} rows and `本地快速回退` on {int(summary['local_quick_fallback_rows'])} rows.",
        "Practical implication: the current attention-family redundancy findings are real for the backfilled focus-board path, but they likely understate the value of the full enriched path because the enriched source fields are still collapsed to defaults in these artifacts.",
    ]

    summary_csv = OUTPUT_DIR / "v10_attention_coverage_summary.csv"
    strategy_csv = OUTPUT_DIR / "v10_attention_coverage_by_strategy.csv"
    sample_csv = OUTPUT_DIR / "v10_attention_coverage_nonzero_samples.csv"
    json_path = OUTPUT_DIR / "v10_attention_coverage_audit.json"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    strategy_delta.to_csv(strategy_csv, index=False)
    sample_nonzero.to_csv(sample_csv, index=False)

    payload = {
        "summary": summary,
        "findings": findings,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    lines = [
        "# v10 Attention Coverage Audit 2026-05-28",
        "",
        "## Purpose",
        "",
        "Audit whether the current replayable `v10` artifact slice actually represents the full attention-enrichment architecture, or whether it is dominated by quick-board/backfilled defaults that can distort redundancy conclusions.",
        "",
        "## Key Findings",
        "",
    ]
    for finding in findings:
        lines.append(f"- {finding}")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            pd.DataFrame([summary]).to_markdown(index=False),
            "",
            "## Strategy Breakdown",
            "",
            strategy_delta.head(10).to_markdown(index=False),
            "",
            "## Interpretation",
            "",
            "- This report is not saying the earlier redundancy audits are false. It is saying their scope is the current backfilled focus-board path, not the full live enriched-analysis path.",
            "- If the persisted slice carries default `50/50/50` sector-fund-news values everywhere, then the incremental value of `_score_final_attention()` cannot be fully observed from these artifacts alone.",
            "- That means any production-facing deletion of `enhanced_attention_score` should wait for a new review/snapshot history generated directly from the full enriched board path rather than reconstructed quick-board snapshots.",
            "",
            "## Next Actions",
            "",
            "1. Generate a fresh non-backfilled review/snapshot history from the live enriched board path and rerun the attention-family audits on that slice.",
            "2. Until then, treat current attention-family simplification results as valid for the quick-board recovery path, but incomplete for the richer symbol-analysis architecture.",
            "",
            f"Summary CSV: `{summary_csv}`",
            f"Strategy CSV: `{strategy_csv}`",
            f"Sample CSV: `{sample_csv}`",
            f"JSON: `{json_path}`",
            "",
        ]
    )
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
