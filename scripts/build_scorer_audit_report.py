from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = PROJECT_ROOT / ".cache" / "daily_focus_board_reviews"
OUTPUT_DIR = PROJECT_ROOT / ".cache" / "trading_system_attribution"
DOC_PATH = PROJECT_ROOT / "docs" / "scorer_audit_2026-05-28.md"


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


def _score_bucket(series: pd.Series, labels: list[str]) -> pd.Series:
    valid = pd.to_numeric(series, errors="coerce")
    if valid.dropna().shape[0] < len(labels):
        return pd.Series(["insufficient"] * len(series), index=series.index, dtype="object")
    ranked = valid.rank(method="first")
    try:
        buckets = pd.qcut(ranked, q=len(labels), labels=labels)
    except ValueError:
        return pd.Series(["insufficient"] * len(series), index=series.index, dtype="object")
    return pd.Series(buckets.astype("object"), index=series.index)


def _bucket_summary(frame: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(bucket_col, dropna=False)
        .agg(
            sample_rows=("symbol", "size"),
            avg_next_day_return_pct=("next_day_return_pct", "mean"),
            avg_intraday_high_return_pct=("intraday_high_return_pct", "mean"),
            win_rate=("win", "mean"),
            direction_hit_rate=("direction_hit", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values(bucket_col).reset_index(drop=True)


def _label_summary(frame: pd.DataFrame, label_col: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(label_col, dropna=False)
        .agg(
            sample_rows=("symbol", "size"),
            avg_next_day_return_pct=("next_day_return_pct", "mean"),
            avg_intraday_high_return_pct=("intraday_high_return_pct", "mean"),
            win_rate=("win", "mean"),
            direction_hit_rate=("direction_hit", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values(["sample_rows", label_col], ascending=[False, True]).reset_index(drop=True)


def _pick_first_present(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for name in candidates:
        if name in frame.columns:
            return frame[name]
    return pd.Series([pd.NA] * len(frame), index=frame.index, dtype="object")


def _load_selected_reviews() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates: list[dict[str, object]] = []
    for path in REVIEW_DIR.glob("review_*.pkl"):
        obj = _load_pickle(path)
        meta = dict(obj.get("meta") or {})
        details = obj.get("details")
        if not isinstance(details, pd.DataFrame) or details.empty:
            continue
        candidates.append(
            {
                "path": path,
                "board_date": str(meta.get("board_date") or ""),
                "cache_version": int(meta.get("cache_version") or 0),
                "board_size": int(meta.get("board_size") or 0),
                "ranking_by": str(meta.get("ranking_by") or ""),
                "rows": len(details),
            }
        )
    review_index = pd.DataFrame(candidates)
    filtered = review_index.loc[
        review_index["board_size"].eq(50) & review_index["ranking_by"].eq("关注分数")
    ].copy()
    filtered = filtered.sort_values(
        ["board_date", "cache_version", "rows", "path"],
        ascending=[True, False, False, True],
    )
    selected = filtered.groupby("board_date", as_index=False).head(1).copy()
    selected["path"] = selected["path"].astype(str)

    detail_frames: list[pd.DataFrame] = []
    for row in selected.itertuples(index=False):
        obj = _load_pickle(Path(row.path))
        details = obj["details"].copy()
        details["board_date"] = str(obj["meta"]["board_date"])
        details["review_date"] = str(obj["meta"]["review_date"])
        details["cache_version"] = int(obj["meta"]["cache_version"])
        details["source_path"] = str(row.path)
        detail_frames.append(details)
    return selected.reset_index(drop=True), pd.concat(detail_frames, ignore_index=True)


def _load_market_feature_scores(board_dates: list[str]) -> pd.DataFrame:
    by_date: dict[str, Path] = {}
    for path in PROJECT_ROOT.joinpath(".cache").glob("market_daily_feature_store_*.pkl"):
        date_token = path.stem.split("_")[-1]
        current = by_date.get(date_token)
        if current is None or path.stat().st_size > current.stat().st_size:
            by_date[date_token] = path

    frames: list[pd.DataFrame] = []
    for board_date in board_dates:
        token = board_date.replace("-", "")
        path = by_date.get(token)
        if path is None:
            continue
        obj = _load_pickle(path)
        data = obj.get("data") if isinstance(obj, dict) else None
        if not isinstance(data, pd.DataFrame) or data.empty:
            continue
        subset = data[["symbol", "stage_code", "stage_label", "stage_score", "launch_score", "quant_score"]].copy()
        subset["symbol"] = subset["symbol"].astype(str).str.zfill(6)
        subset["board_date"] = board_date
        subset["feature_store_path"] = str(path)
        frames.append(subset)
    if not frames:
        return pd.DataFrame(
            columns=["symbol", "stage_code", "stage_label", "stage_score", "launch_score", "quant_score", "board_date", "feature_store_path"]
        )
    return pd.concat(frames, ignore_index=True)


def build_report() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_reviews, review_details = _load_selected_reviews()
    review_details["symbol"] = review_details["symbol"].astype(str).str.zfill(6)

    launch_details = review_details.loc[review_details["launch_window_score"].notna()].copy()
    launch_details["launch_window_bucket"] = _score_bucket(
        launch_details["launch_window_score"],
        labels=["low", "mid", "high"],
    )
    launch_summary = _bucket_summary(launch_details, "launch_window_bucket")

    rank_details = review_details.loc[review_details["final_rank_score"].notna()].copy()
    rank_details["final_rank_bucket"] = _score_bucket(
        rank_details["final_rank_score"],
        labels=["low", "mid", "high"],
    )
    rank_summary = _bucket_summary(rank_details, "final_rank_bucket")

    execution_details = review_details.loc[review_details["execution_score"].notna()].copy()
    execution_details["execution_score_bucket"] = _score_bucket(
        execution_details["execution_score"],
        labels=["low", "mid", "high"],
    )
    execution_summary = _bucket_summary(execution_details, "execution_score_bucket")
    execution_label_summary = pd.DataFrame()
    if "execution_label" in execution_details.columns:
        execution_label_summary = _label_summary(execution_details, "execution_label")

    selection_details = review_details.loc[review_details["selection_score"].notna()].copy()
    selection_details["selection_score_bucket"] = _score_bucket(
        selection_details["selection_score"],
        labels=["low", "mid", "high"],
    )
    selection_summary = _bucket_summary(selection_details, "selection_score_bucket")

    market_scores = _load_market_feature_scores(sorted(selected_reviews["board_date"].unique().tolist()))
    joined_stage = review_details.merge(
        market_scores,
        on=["board_date", "symbol"],
        how="inner",
        suffixes=("_review", "_feature"),
    )
    stage_summary = pd.DataFrame()
    stage_code_summary = pd.DataFrame()
    if not joined_stage.empty:
        joined_stage["stage_score_effective"] = pd.to_numeric(
            _pick_first_present(
                joined_stage,
                ["stage_score_review", "stage_score", "stage_score_feature"],
            ),
            errors="coerce",
        )
        joined_stage["stage_code_effective"] = _pick_first_present(
            joined_stage,
            ["stage_code_review", "stage_code", "stage_code_feature"],
        )
        joined_stage["stage_label_effective"] = _pick_first_present(
            joined_stage,
            ["stage_label_review", "stage_label", "stage_label_feature"],
        )
        joined_stage["stage_score_bucket"] = _score_bucket(
            joined_stage["stage_score_effective"],
            labels=["low", "mid", "high"],
        )
        stage_summary = _bucket_summary(joined_stage, "stage_score_bucket")
        stage_code_summary = (
            joined_stage.groupby(["stage_code_effective", "stage_label_effective"], dropna=False)
            .agg(
                sample_rows=("symbol", "size"),
                avg_next_day_return_pct=("next_day_return_pct", "mean"),
                avg_intraday_high_return_pct=("intraday_high_return_pct", "mean"),
                win_rate=("win", "mean"),
            )
            .reset_index()
            .sort_values(["sample_rows", "avg_next_day_return_pct"], ascending=[False, False])
            .reset_index(drop=True)
        )
        stage_code_summary = stage_code_summary.rename(
            columns={
                "stage_code_effective": "stage_code",
                "stage_label_effective": "stage_label",
            }
        )

    observability = {
        "selected_review_files": int(len(selected_reviews)),
        "selected_review_rows": int(len(review_details)),
        "review_rows_with_launch_window_score": int(review_details["launch_window_score"].notna().sum()) if "launch_window_score" in review_details.columns else 0,
        "review_rows_with_final_rank_score": int(review_details["final_rank_score"].notna().sum()) if "final_rank_score" in review_details.columns else 0,
        "review_rows_with_execution_score": int(review_details["execution_score"].notna().sum()) if "execution_score" in review_details.columns else 0,
        "review_rows_with_selection_score": int(review_details["selection_score"].notna().sum()) if "selection_score" in review_details.columns else 0,
        "review_rows_with_stage_score": int(review_details["stage_score"].notna().sum()) if "stage_score" in review_details.columns else 0,
        "stage_score_join_rows": int(len(joined_stage)),
    }

    findings: list[str] = []
    if not launch_summary.empty and {"low", "high"}.issubset(set(launch_summary["launch_window_bucket"].astype(str))):
        low = launch_summary.loc[launch_summary["launch_window_bucket"].astype(str).eq("low")].iloc[0]
        high = launch_summary.loc[launch_summary["launch_window_bucket"].astype(str).eq("high")].iloc[0]
        findings.append(
            "In replay samples with launch-window data, the high launch-window bucket averages "
            f"{float(high['avg_next_day_return_pct']):.2f}% next-day return versus {float(low['avg_next_day_return_pct']):.2f}% "
            "for the low bucket."
        )
    if not rank_summary.empty and {"low", "high"}.issubset(set(rank_summary["final_rank_bucket"].astype(str))):
        low = rank_summary.loc[rank_summary["final_rank_bucket"].astype(str).eq("low")].iloc[0]
        high = rank_summary.loc[rank_summary["final_rank_bucket"].astype(str).eq("high")].iloc[0]
        findings.append(
            "Where final-rank replay data exists, the high final-rank bucket averages "
            f"{float(high['avg_next_day_return_pct']):.2f}% next-day return versus {float(low['avg_next_day_return_pct']):.2f}% "
            "for the low bucket."
        )
    if not stage_summary.empty and {"low", "high"}.issubset(set(stage_summary["stage_score_bucket"].astype(str))):
        low = stage_summary.loc[stage_summary["stage_score_bucket"].astype(str).eq("low")].iloc[0]
        high = stage_summary.loc[stage_summary["stage_score_bucket"].astype(str).eq("high")].iloc[0]
        findings.append(
            "On dates where stage-score joins are available, the high stage-score bucket averages "
            f"{float(high['avg_next_day_return_pct']):.2f}% next-day return versus {float(low['avg_next_day_return_pct']):.2f}% "
            "for the low bucket."
        )
    if not execution_summary.empty and {"low", "high"}.issubset(set(execution_summary["execution_score_bucket"].astype(str))):
        low = execution_summary.loc[execution_summary["execution_score_bucket"].astype(str).eq("low")].iloc[0]
        high = execution_summary.loc[execution_summary["execution_score_bucket"].astype(str).eq("high")].iloc[0]
        findings.append(
            "In replay samples with execution-score data, the high execution-score bucket averages "
            f"{float(high['avg_next_day_return_pct']):.2f}% next-day return versus {float(low['avg_next_day_return_pct']):.2f}% "
            "for the low bucket."
        )
    if not selection_summary.empty and {"low", "high"}.issubset(set(selection_summary["selection_score_bucket"].astype(str))):
        low = selection_summary.loc[selection_summary["selection_score_bucket"].astype(str).eq("low")].iloc[0]
        high = selection_summary.loc[selection_summary["selection_score_bucket"].astype(str).eq("high")].iloc[0]
        findings.append(
            "In replay samples with selection-score data, the high selection-score bucket averages "
            f"{float(high['avg_next_day_return_pct']):.2f}% next-day return versus {float(low['avg_next_day_return_pct']):.2f}% "
            "for the low bucket."
        )
    if observability["selected_review_rows"] > 0:
        findings.append(
            "Replay-detail observability now covers "
            f"`execution_score` on {observability['review_rows_with_execution_score']} / {observability['selected_review_rows']} rows, "
            f"`selection_score` on {observability['review_rows_with_selection_score']} / {observability['selected_review_rows']} rows, and "
            f"`stage_score` on {observability['review_rows_with_stage_score']} / {observability['selected_review_rows']} rows."
        )
    if observability["review_rows_with_execution_score"] < observability["selected_review_rows"]:
        findings.append(
            "Execution-level scoring is still only partially auditable from replay caches, so mixed-version history can bias any scorer-level conclusion."
        )
    if observability["review_rows_with_stage_score"] < observability["selected_review_rows"]:
        findings.append(
            "Stage-level scores are still incomplete in replay details, so the end-to-end evidence trail remains mixed until more v9 history is regenerated."
        )

    launch_csv = OUTPUT_DIR / "scorer_launch_window_summary.csv"
    rank_csv = OUTPUT_DIR / "scorer_final_rank_summary.csv"
    execution_csv = OUTPUT_DIR / "scorer_execution_score_summary.csv"
    execution_label_csv = OUTPUT_DIR / "scorer_execution_label_summary.csv"
    selection_csv = OUTPUT_DIR / "scorer_selection_score_summary.csv"
    stage_csv = OUTPUT_DIR / "scorer_stage_score_summary.csv"
    stage_code_csv = OUTPUT_DIR / "scorer_stage_code_summary.csv"
    review_csv = OUTPUT_DIR / "selected_review_files.csv"

    selected_reviews.to_csv(review_csv, index=False, encoding="utf-8-sig")
    launch_summary.to_csv(launch_csv, index=False, encoding="utf-8-sig")
    rank_summary.to_csv(rank_csv, index=False, encoding="utf-8-sig")
    execution_summary.to_csv(execution_csv, index=False, encoding="utf-8-sig")
    execution_label_summary.to_csv(execution_label_csv, index=False, encoding="utf-8-sig")
    selection_summary.to_csv(selection_csv, index=False, encoding="utf-8-sig")
    stage_summary.to_csv(stage_csv, index=False, encoding="utf-8-sig")
    stage_code_summary.to_csv(stage_code_csv, index=False, encoding="utf-8-sig")

    payload = {
        "observability": observability,
        "selected_reviews": _clean_records(selected_reviews),
        "launch_summary": _clean_records(launch_summary),
        "rank_summary": _clean_records(rank_summary),
        "execution_summary": _clean_records(execution_summary),
        "execution_label_summary": _clean_records(execution_label_summary),
        "selection_summary": _clean_records(selection_summary),
        "stage_summary": _clean_records(stage_summary),
        "stage_code_summary": _clean_records(stage_code_summary),
        "findings": findings,
    }
    json_path = OUTPUT_DIR / "scorer_audit.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Scorer Audit 2026-05-28",
        "",
        "## Purpose",
        "",
        "Validate whether stage/launch/rank scorers have replay evidence behind them, and identify where the system is still making score-driven decisions without durable post-trade observability.",
        "",
        "## Coverage",
        "",
        f"- Selected review files: {observability['selected_review_files']}",
        f"- Selected review rows: {observability['selected_review_rows']}",
        f"- Rows with `launch_window_score`: {observability['review_rows_with_launch_window_score']}",
        f"- Rows with `final_rank_score`: {observability['review_rows_with_final_rank_score']}",
        f"- Rows with `execution_score`: {observability['review_rows_with_execution_score']}",
        f"- Rows with `selection_score`: {observability['review_rows_with_selection_score']}",
        f"- Rows with `stage_score` directly in review details: {observability['review_rows_with_stage_score']}",
        f"- Rows where `stage_score` could be joined from feature-store snapshots: {observability['stage_score_join_rows']}",
        "",
        "## Key Findings",
        "",
    ]
    for item in findings:
        lines.append(f"- {item}")
    if not launch_summary.empty:
        lines.extend(
            [
                "",
                "## Launch Window Buckets",
                "",
                launch_summary.to_markdown(index=False),
            ]
        )
    if not rank_summary.empty:
        lines.extend(
            [
                "",
                "## Final Rank Buckets",
                "",
                rank_summary.to_markdown(index=False),
            ]
        )
    if not execution_summary.empty:
        lines.extend(
            [
                "",
                "## Execution Score Buckets",
                "",
                execution_summary.to_markdown(index=False),
            ]
        )
    if not selection_summary.empty:
        lines.extend(
            [
                "",
                "## Selection Score Buckets",
                "",
                selection_summary.to_markdown(index=False),
            ]
        )
    if not stage_summary.empty:
        lines.extend(
            [
                "",
                "## Stage Score Buckets",
                "",
                stage_summary.to_markdown(index=False),
            ]
        )
    if not stage_code_summary.empty:
        lines.extend(
            [
                "",
                "## Stage Code Breakdown",
                "",
                stage_code_summary.head(12).to_markdown(index=False),
            ]
        )
    if not execution_label_summary.empty:
        lines.extend(
            [
                "",
                "## Execution Label Breakdown",
                "",
                execution_label_summary.to_markdown(index=False),
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `launch_window_score` now has enough replay coverage to be judged as a real ranking aid rather than a pure narrative field.",
            "- `final_rank_score` has partial replay coverage only in the newest review subset, so evidence is directional but not yet durable.",
            "- `execution_score` and `selection_score` now have enough replay rows for first-pass bucket analysis, but their distributions are saturated near 100, so monotonicity needs to be read cautiously.",
            "- `execution_score`, `selection_score`, and `stage_score` are now persisted in v9 review artifacts, but historical evidence remains mixed until more legacy windows are regenerated.",
            "- `stage_score` can now be audited from either replay details or joined feature snapshots, which removes the prior hard observability gap but does not by itself prove positive alpha.",
            "",
            "## Next Actions",
            "",
            "1. Keep regenerating v9 review history so scorer coverage is dominated by post-persistence artifacts rather than mixed legacy caches.",
            "2. Extend replay summaries to segment by `execution_window`, `execution_label`, and confidence fields, not just raw score buckets.",
            "3. Remove any scorer from default gating/ranking if it cannot show monotonic replay improvement after v9 coverage is broad enough.",
            "",
            f"Review file index: `{review_csv}`",
            f"Launch summary: `{launch_csv}`",
            f"Final-rank summary: `{rank_csv}`",
            f"Execution-score summary: `{execution_csv}`",
            f"Execution-label summary: `{execution_label_csv}`",
            f"Selection-score summary: `{selection_csv}`",
            f"Stage-score summary: `{stage_csv}`",
            f"Stage-code summary: `{stage_code_csv}`",
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
