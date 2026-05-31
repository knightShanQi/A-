from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from validate_model_strategy_combinations import run_validation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".cache" / "ten_year_model_strategy_validation"


def _format_date(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def _window_chunks(date_from: str, date_to: str) -> list[tuple[str, str]]:
    start = pd.to_datetime(date_from, errors="coerce")
    end = pd.to_datetime(date_to, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Invalid date window.")
    chunks: list[tuple[str, str]] = []
    cursor = pd.Timestamp(start)
    while cursor <= end:
        year_end = pd.Timestamp(cursor.year, 12, 31)
        chunk_end = min(year_end, pd.Timestamp(end))
        chunks.append((cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cursor = chunk_end + pd.Timedelta(days=1)
    return chunks


def _auc(y_true: pd.Series, score: pd.Series) -> float | None:
    valid = pd.DataFrame({"y": y_true, "score": score}).dropna()
    if valid.empty or valid["y"].nunique() < 2:
        return None
    return float(roc_auc_score(valid["y"].astype(int), valid["score"].astype(float)))


def _load_csvs(paths: list[Path], columns: list[str] | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists() or path.stat().st_size <= 0:
            continue
        frame = pd.read_csv(path, encoding="utf-8-sig", usecols=columns)
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _summarize_model_scores(model_scores: pd.DataFrame, hold_days: int, positive_return: float) -> dict[str, object]:
    return_column = f"hold_{int(hold_days)}d_return"
    if model_scores.empty:
        return {}
    frame = model_scores.copy()
    frame["market_date"] = pd.to_datetime(frame["market_date"], errors="coerce")
    frame["model_score"] = pd.to_numeric(frame["model_score"], errors="coerce")
    frame[return_column] = pd.to_numeric(frame[return_column], errors="coerce")
    frame = frame.dropna(subset=["market_date", "model_score", return_column])
    auc_positive = _auc((frame[return_column] > 0).astype(int), frame["model_score"])
    auc_target = _auc((frame[return_column] >= float(positive_return)).astype(int), frame["model_score"])
    return {
        "scored_rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()) if "symbol" in frame.columns else 0,
        "days": int(frame["market_date"].nunique()),
        "overall_avg_return": round(float(frame[return_column].mean()), 6),
        "overall_win_rate": round(float((frame[return_column] > 0).mean()), 4),
        "overall_target_hit_rate": round(float((frame[return_column] >= float(positive_return)).mean()), 4),
        "auc_positive_return": None if auc_positive is None else round(float(auc_positive), 4),
        "auc_target_return": None if auc_target is None else round(float(auc_target), 4),
    }


def _summarize_strategy_candidates(strategy_candidates: pd.DataFrame, hold_days: int, positive_return: float) -> dict[str, object]:
    return_column = f"hold_{int(hold_days)}d_return"
    if strategy_candidates.empty:
        return {}
    frame = strategy_candidates.copy()
    frame["market_date"] = pd.to_datetime(frame["market_date"], errors="coerce")
    frame[return_column] = pd.to_numeric(frame[return_column], errors="coerce")
    frame = frame.dropna(subset=["market_date", return_column])
    breakdown: list[dict[str, object]] = []
    if "candidate_strategy" in frame.columns:
        for strategy, group in frame.groupby("candidate_strategy", dropna=False):
            returns = pd.to_numeric(group[return_column], errors="coerce").dropna()
            if returns.empty:
                continue
            breakdown.append(
                {
                    "candidate_strategy": str(strategy),
                    "trade_count": int(len(returns)),
                    "avg_return": round(float(returns.mean()), 6),
                    "win_rate": round(float((returns > 0).mean()), 4),
                    "target_hit_rate": round(float((returns >= float(positive_return)).mean()), 4),
                }
            )
    return {
        "candidate_rows": int(len(frame)),
        "candidate_days": int(frame["market_date"].nunique()),
        "overall_avg_return": round(float(frame[return_column].mean()), 6),
        "overall_win_rate": round(float((frame[return_column] > 0).mean()), 4),
        "overall_target_hit_rate": round(float((frame[return_column] >= float(positive_return)).mean()), 4),
        "strategy_breakdown": breakdown,
    }


def _summarize_selected_rules(selected_frames: list[pd.DataFrame], all_dates: list[pd.Timestamp], hold_days: int, positive_return: float) -> list[dict[str, object]]:
    return_column = f"hold_{int(hold_days)}d_return"
    selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    if selected.empty:
        return []
    selected["market_date"] = pd.to_datetime(selected["market_date"], errors="coerce")
    selected[return_column] = pd.to_numeric(selected[return_column], errors="coerce")
    selected = selected.dropna(subset=["rule", "market_date", return_column]).copy()
    summaries: list[dict[str, object]] = []
    calendar_base = pd.DataFrame({"market_date": pd.to_datetime(sorted(set(all_dates)))})
    for rule, group in selected.groupby("rule", dropna=False):
        daily = (
            group.groupby("market_date", as_index=False)
            .agg(
                selected=("symbol", "count"),
                avg_return=(return_column, "mean"),
            )
            .sort_values("market_date")
        )
        calendar = calendar_base.merge(daily, on="market_date", how="left")
        calendar["avg_return"] = calendar["avg_return"].fillna(0.0)
        calendar["selected"] = calendar["selected"].fillna(0).astype(int)
        calendar["equity"] = (1.0 + calendar["avg_return"]).cumprod()
        calendar["running_max"] = calendar["equity"].cummax()
        calendar["drawdown"] = calendar["equity"] / calendar["running_max"].replace(0.0, np.nan) - 1.0
        trade_returns = pd.to_numeric(group[return_column], errors="coerce").dropna()
        active_returns = pd.to_numeric(daily["avg_return"], errors="coerce").dropna()
        ending_equity = float(calendar["equity"].iloc[-1]) if not calendar.empty else 1.0
        annualized = ending_equity ** (252.0 / len(calendar)) - 1.0 if len(calendar) and ending_equity > 0 else 0.0
        max_drawdown = float(calendar["drawdown"].min()) if not calendar.empty else 0.0
        summaries.append(
            {
                "rule": str(rule),
                "active_days": int((calendar["selected"] > 0).sum()),
                "coverage_pct": round(float((calendar["selected"] > 0).mean()) * 100.0, 2) if not calendar.empty else 0.0,
                "selected_rows": int(len(group)),
                "evaluated_trade_count": int(len(trade_returns)),
                "avg_trade_return": round(float(trade_returns.mean()), 6) if not trade_returns.empty else 0.0,
                "trade_win_rate": round(float((trade_returns > 0).mean()), 4) if not trade_returns.empty else 0.0,
                "target_hit_rate": round(float((trade_returns >= float(positive_return)).mean()), 4) if not trade_returns.empty else 0.0,
                "active_daily_return": round(float(active_returns.mean()), 6) if not active_returns.empty else 0.0,
                "active_daily_win_rate": round(float((active_returns > 0).mean()), 4) if not active_returns.empty else 0.0,
                "calendar_daily_return": round(float(calendar["avg_return"].mean()), 6) if not calendar.empty else 0.0,
                "annualized_return": round(float(annualized), 6),
                "max_drawdown": round(float(max_drawdown), 6),
                "ending_equity": round(float(ending_equity), 6),
            }
        )
    result = pd.DataFrame(summaries)
    if result.empty:
        return []
    result["return_drawdown_ratio"] = result["annualized_return"] / result["max_drawdown"].abs().replace(0.0, np.nan)
    result = result.sort_values(["return_drawdown_ratio", "annualized_return"], ascending=[False, False])
    return result.replace({np.nan: None}).to_dict("records")


def _run_validation_chunk_worker(payload: dict[str, object]) -> dict[str, object]:
    chunk_start = str(payload["chunk_start"])
    chunk_end = str(payload["chunk_end"])
    chunk_dir = Path(str(payload["chunk_dir"]))
    summary_path = chunk_dir / "summary.json"
    if bool(payload.get("skip_existing")) and summary_path.exists():
        print(
            f"[chunk] {payload['index']}/{payload['total']} reuse {chunk_start} -> {chunk_end}",
            flush=True,
        )
        return json.loads(summary_path.read_text(encoding="utf-8"))
    print(
        f"[chunk] {payload['index']}/{payload['total']} run {chunk_start} -> {chunk_end}",
        flush=True,
    )
    return run_validation(
        date_from=chunk_start,
        date_to=chunk_end,
        hold_days=int(payload["hold_days"]),
        model_horizon_days=int(payload["model_horizon_days"]),
        positive_return=float(payload["positive_return"]),
        output_dir=chunk_dir,
    )


def run_ten_year_validation(
    *,
    date_from: str,
    date_to: str,
    hold_days: int = 3,
    model_horizon_days: int = 5,
    positive_return: float = 0.03,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    skip_existing: bool = True,
    max_workers: int = 1,
) -> dict[str, object]:
    output_path = Path(output_dir)
    chunk_root = output_path / "chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    chunks = _window_chunks(date_from, date_to)
    def run_one_chunk(payload: tuple[int, str, str]) -> dict[str, object]:
        index, chunk_start, chunk_end = payload
        chunk_dir = chunk_root / f"{chunk_start.replace('-', '')}_{chunk_end.replace('-', '')}"
        summary_path = chunk_dir / "summary.json"
        if skip_existing and summary_path.exists():
            print(f"[chunk] {index}/{len(chunks)} reuse {chunk_start} -> {chunk_end}", flush=True)
            return json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"[chunk] {index}/{len(chunks)} run {chunk_start} -> {chunk_end}", flush=True)
        return run_validation(
            date_from=chunk_start,
            date_to=chunk_end,
            hold_days=int(hold_days),
            model_horizon_days=int(model_horizon_days),
            positive_return=float(positive_return),
            output_dir=chunk_dir,
        )

    indexed_chunks = [(index, chunk_start, chunk_end) for index, (chunk_start, chunk_end) in enumerate(chunks, start=1)]
    chunk_summaries: list[dict[str, object]] = []
    worker_count = max(1, int(max_workers))
    if worker_count == 1:
        for payload in indexed_chunks:
            chunk_summaries.append(run_one_chunk(payload))
    else:
        print(f"[parallel] running {len(indexed_chunks)} chunks with max_workers={worker_count}", flush=True)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(_run_validation_chunk_worker, payload): payload for payload in [
                {
                    "index": index,
                    "total": len(chunks),
                    "chunk_start": chunk_start,
                    "chunk_end": chunk_end,
                    "chunk_dir": str(chunk_root / f"{chunk_start.replace('-', '')}_{chunk_end.replace('-', '')}"),
                    "skip_existing": bool(skip_existing),
                    "hold_days": int(hold_days),
                    "model_horizon_days": int(model_horizon_days),
                    "positive_return": float(positive_return),
                }
                for index, chunk_start, chunk_end in indexed_chunks
            ]}
            for future in as_completed(futures):
                payload = futures[future]
                try:
                    summary = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Chunk {payload['index']}/{payload['total']} {payload['chunk_start']} -> {payload['chunk_end']} failed"
                    ) from exc
                print(
                    f"[parallel] done {payload['index']}/{payload['total']} {payload['chunk_start']} -> {payload['chunk_end']}",
                    flush=True,
                )
                chunk_summaries.append(summary)

    model_score_paths = [Path(item["model_scores_path"]) for item in chunk_summaries if item.get("model_scores_path")]
    strategy_candidate_paths = [Path(item["strategy_candidates_path"]) for item in chunk_summaries if item.get("strategy_candidates_path")]
    selected_paths_by_key = {
        "model": [Path(item["model_selected_path"]) for item in chunk_summaries if item.get("model_selected_path")],
        "strategy": [Path(item["strategy_selected_path"]) for item in chunk_summaries if item.get("strategy_selected_path")],
        "combined": [Path(item["combined_selected_path"]) for item in chunk_summaries if item.get("combined_selected_path")],
    }

    model_scores = _load_csvs(model_score_paths)
    strategy_candidates = _load_csvs(strategy_candidate_paths)
    date_values = pd.to_datetime(model_scores["market_date"], errors="coerce").dropna().dt.normalize().unique().tolist() if not model_scores.empty else []

    selected_summaries: dict[str, list[dict[str, object]]] = {}
    selected_frames: dict[str, pd.DataFrame] = {}
    for key, paths in selected_paths_by_key.items():
        frame = _load_csvs(paths)
        selected_frames[key] = frame
        selected_summaries[f"{key}_rules"] = _summarize_selected_rules(
            [frame],
            all_dates=[pd.Timestamp(value) for value in date_values],
            hold_days=int(hold_days),
            positive_return=float(positive_return),
        )

    summary = {
        "date_from": str(date_from),
        "date_to": str(date_to),
        "hold_days": int(hold_days),
        "positive_return": float(positive_return),
        "chunk_count": int(len(chunks)),
        "evaluable_days": int(len(set(date_values))),
        "model_accuracy": _summarize_model_scores(model_scores, int(hold_days), float(positive_return)),
        "strategy_accuracy": _summarize_strategy_candidates(strategy_candidates, int(hold_days), float(positive_return)),
        **selected_summaries,
    }
    combined_rules = pd.DataFrame(summary.get("combined_rules", []))
    best_rules = combined_rules
    if not best_rules.empty:
        best_rules = best_rules.sort_values(["return_drawdown_ratio", "annualized_return"], ascending=[False, False])
    summary["best_combined_rules"] = best_rules.head(10).replace({np.nan: None}).to_dict("records") if not best_rules.empty else []

    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    model_scores.to_csv(output_path / "model_scores.csv", index=False, encoding="utf-8-sig")
    strategy_candidates.to_csv(output_path / "strategy_candidates.csv", index=False, encoding="utf-8-sig")
    for key, frame in selected_frames.items():
        frame.to_csv(output_path / f"{key}_selected_trades.csv", index=False, encoding="utf-8-sig")
    for key in ("model_rules", "strategy_rules", "combined_rules"):
        pd.DataFrame(summary.get(key, [])).to_csv(output_path / f"{key}.csv", index=False, encoding="utf-8-sig")
    summary.update(
        {
            "summary_path": str(summary_path),
            "model_scores_path": str(output_path / "model_scores.csv"),
            "strategy_candidates_path": str(output_path / "strategy_candidates.csv"),
            "model_selected_path": str(output_path / "model_selected_trades.csv"),
            "strategy_selected_path": str(output_path / "strategy_selected_trades.csv"),
            "combined_selected_path": str(output_path / "combined_selected_trades.csv"),
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ten-year chunked model/strategy validation.")
    parser.add_argument("--date-from", required=True)
    parser.add_argument("--date-to", required=True)
    parser.add_argument("--hold-days", type=int, default=3)
    parser.add_argument("--model-horizon-days", type=int, default=5)
    parser.add_argument("--positive-return", type=float, default=0.03)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = run_ten_year_validation(
        date_from=args.date_from,
        date_to=args.date_to,
        hold_days=int(args.hold_days),
        model_horizon_days=int(args.model_horizon_days),
        positive_return=float(args.positive_return),
        output_dir=args.output_dir,
        skip_existing=not bool(args.force),
        max_workers=int(args.max_workers),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
