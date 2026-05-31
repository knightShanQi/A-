from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from a_share_predictor.database_source import load_env_file

load_env_file()

from backtest_strategy_model_top10 import (
    _build_feature_frame,
    _candidate_pool_for_dates,
    _fetch_tushare_history,
    _format_date,
    _prepare_strategy_history,
    _resolve_window,
    _score_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".cache" / "model_strategy_validation"


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if np.isnan(numeric) or np.isinf(numeric):
        return float(default)
    return numeric


def _main_board_non_st_mask(frame: pd.DataFrame) -> pd.Series:
    symbols = frame.get("symbol", pd.Series("", index=frame.index)).astype(str).str.zfill(6)
    names = frame.get("name", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper()
    symbol_main = symbols.str.match(r"^(000|001|002|003|600|601|603|605)")
    non_st = ~names.str.contains("ST|退", regex=True, na=False)
    return symbol_main & non_st


def _attach_forward_returns(history: pd.DataFrame, hold_days: int) -> pd.DataFrame:
    frame = history.sort_values(["symbol", "trade_date"]).copy()
    grouped = frame.groupby("symbol", group_keys=False)
    entry_open = grouped["open"].shift(-1)
    exit_close = grouped["close"].shift(-int(hold_days))
    max_high = grouped["high"].shift(-1)
    min_low = grouped["low"].shift(-1)
    for offset in range(2, int(hold_days) + 1):
        max_high = np.maximum(max_high, grouped["high"].shift(-offset))
        min_low = np.minimum(min_low, grouped["low"].shift(-offset))
    frame["entry_price"] = pd.to_numeric(entry_open, errors="coerce")
    frame[f"hold_{int(hold_days)}d_return"] = pd.to_numeric(exit_close, errors="coerce") / frame["entry_price"] - 1.0
    frame["max_high_return"] = pd.to_numeric(max_high, errors="coerce") / frame["entry_price"] - 1.0
    frame["max_drawdown"] = pd.to_numeric(min_low, errors="coerce") / frame["entry_price"] - 1.0
    return frame


def _auc(y_true: pd.Series, score: pd.Series) -> float | None:
    valid = pd.DataFrame({"y": y_true, "score": score}).dropna()
    if valid.empty or valid["y"].nunique() < 2:
        return None
    return float(roc_auc_score(valid["y"].astype(int), valid["score"].astype(float)))


def _evaluate_daily_selection(
    frame: pd.DataFrame,
    *,
    rule_name: str,
    all_dates: list[pd.Timestamp],
    rank_column: str,
    top_n: int,
    min_model_score: float | None = None,
    require_full: bool = True,
    hold_days: int = 3,
    target_return: float = 0.03,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    data = frame.copy()
    if min_model_score is not None and "model_score" in data.columns:
        data = data[pd.to_numeric(data["model_score"], errors="coerce") >= float(min_model_score)].copy()
    if data.empty:
        return (
            {
                "rule": rule_name,
                "active_days": 0,
                "coverage_pct": 0.0,
                "selected_rows": 0,
                "evaluated_trade_count": 0,
                "avg_trade_return": 0.0,
                "trade_win_rate": 0.0,
                "target_hit_rate": 0.0,
                "active_daily_return": 0.0,
                "active_daily_win_rate": 0.0,
                "calendar_daily_return": 0.0,
                "annualized_return": 0.0,
                "max_drawdown": 0.0,
                "ending_equity": 1.0,
            },
            pd.DataFrame(),
            pd.DataFrame(),
        )

    return_column = f"hold_{int(hold_days)}d_return"
    sort_columns = ["market_date", rank_column]
    ascending = [True, False]
    if rank_column != "model_score" and "model_score" in data.columns:
        sort_columns.append("model_score")
        ascending.append(False)
    data = data.sort_values(sort_columns, ascending=ascending)
    selected = data.groupby("market_date", group_keys=False).head(max(int(top_n), 1)).copy()
    selected["daily_rule_rank"] = selected.groupby("market_date").cumcount() + 1
    if require_full:
        counts = selected.groupby("market_date")["symbol"].count()
        full_dates = counts[counts >= int(top_n)].index
        selected = selected[selected["market_date"].isin(full_dates)].copy()
    selected = selected.dropna(subset=[return_column])
    active_daily = (
        selected.groupby("market_date", as_index=False)
        .agg(
            selected=("symbol", "count"),
            avg_return=(return_column, "mean"),
            win_rate=(return_column, lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
            target_hit_rate=(return_column, lambda s: float((pd.to_numeric(s, errors="coerce") >= float(target_return)).mean())),
            avg_model_score=("model_score", "mean") if "model_score" in selected.columns else (return_column, "size"),
        )
        .sort_values("market_date")
        .reset_index(drop=True)
    )
    calendar = pd.DataFrame({"market_date": pd.to_datetime(all_dates)})
    calendar = calendar.merge(active_daily[["market_date", "avg_return", "selected"]], on="market_date", how="left")
    calendar["avg_return"] = calendar["avg_return"].fillna(0.0)
    calendar["selected"] = calendar["selected"].fillna(0).astype(int)
    calendar["equity"] = (1.0 + calendar["avg_return"]).cumprod()
    calendar["running_max"] = calendar["equity"].cummax()
    calendar["drawdown"] = calendar["equity"] / calendar["running_max"].replace(0.0, np.nan) - 1.0
    trade_returns = pd.to_numeric(selected[return_column], errors="coerce").dropna()
    active_returns = pd.to_numeric(active_daily.get("avg_return", pd.Series(dtype=float)), errors="coerce").dropna()
    ending_equity = _safe_float(calendar["equity"].iloc[-1], 1.0) if not calendar.empty else 1.0
    annualized = ending_equity ** (252.0 / len(calendar)) - 1.0 if len(calendar) and ending_equity > 0 else 0.0
    summary = {
        "rule": rule_name,
        "top_n": int(top_n),
        "score_threshold": None if min_model_score is None else float(min_model_score),
        "active_days": int((calendar["selected"] > 0).sum()),
        "coverage_pct": round(float((calendar["selected"] > 0).mean()) * 100.0, 2) if not calendar.empty else 0.0,
        "selected_rows": int(len(selected)),
        "evaluated_trade_count": int(len(trade_returns)),
        "avg_trade_return": round(float(trade_returns.mean()), 6) if not trade_returns.empty else 0.0,
        "trade_win_rate": round(float((trade_returns > 0).mean()), 4) if not trade_returns.empty else 0.0,
        "target_hit_rate": round(float((trade_returns >= float(target_return)).mean()), 4) if not trade_returns.empty else 0.0,
        "active_daily_return": round(float(active_returns.mean()), 6) if not active_returns.empty else 0.0,
        "active_daily_win_rate": round(float((active_returns > 0).mean()), 4) if not active_returns.empty else 0.0,
        "calendar_daily_return": round(float(calendar["avg_return"].mean()), 6) if not calendar.empty else 0.0,
        "annualized_return": round(float(annualized), 6),
        "max_drawdown": round(float(calendar["drawdown"].min()), 6) if not calendar.empty else 0.0,
        "ending_equity": round(float(ending_equity), 6),
    }
    return summary, selected, calendar


def _strategy_breakdown(frame: pd.DataFrame, hold_days: int, target_return: float) -> list[dict[str, object]]:
    return_column = f"hold_{int(hold_days)}d_return"
    rows: list[dict[str, object]] = []
    if "candidate_strategy" not in frame.columns:
        return rows
    for strategy, group in frame.groupby("candidate_strategy", dropna=False):
        returns = pd.to_numeric(group[return_column], errors="coerce").dropna()
        if returns.empty:
            continue
        rows.append(
            {
                "candidate_strategy": str(strategy),
                "trade_count": int(len(returns)),
                "avg_return": round(float(returns.mean()), 6),
                "win_rate": round(float((returns > 0).mean()), 4),
                "target_hit_rate": round(float((returns >= float(target_return)).mean()), 4),
            }
        )
    return rows


def run_validation(
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    months: int = 6,
    hold_days: int = 3,
    model_horizon_days: int = 5,
    positive_return: float = 0.03,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    resolved_from, resolved_to = _resolve_window(date_from, date_to, months)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[prepare] validation window {resolved_from} -> {resolved_to}", flush=True)
    raw_history = _fetch_tushare_history(resolved_from, resolved_to, lookback_days=260, forward_days=int(hold_days) + 8)
    if raw_history.empty:
        raise RuntimeError("No complete OHLCV market history is available for validation.")
    history = _prepare_strategy_history(raw_history)
    history = _attach_forward_returns(history, int(hold_days))

    start = pd.to_datetime(resolved_from)
    end = pd.to_datetime(resolved_to)
    trade_dates = sorted(
        pd.to_datetime(history.loc[history["trade_date"].between(start, end, inclusive="both"), "trade_date"], errors="coerce")
        .dropna()
        .dt.normalize()
        .unique()
        .tolist()
    )
    evaluable_dates = []
    for value in trade_dates:
        day = pd.Timestamp(value)
        day_rows = history[(history["trade_date"].eq(day)) & history[f"hold_{int(hold_days)}d_return"].notna()]
        if not day_rows.empty:
            evaluable_dates.append(day)
    print(f"[prepare] evaluable dates {len(evaluable_dates)} / raw trade dates {len(trade_dates)}", flush=True)

    feature_frame = _build_feature_frame(history, pd.DatetimeIndex(trade_dates))
    feature_frame["market_date"] = pd.to_datetime(feature_frame["trade_date"], errors="coerce").dt.normalize()
    feature_frame["candidate_priority"] = 0.0
    feature_frame["amount"] = 0.0
    forward_cols = ["symbol", "trade_date", f"hold_{int(hold_days)}d_return", "max_high_return", "max_drawdown", "entry_price"]
    feature_frame = feature_frame.merge(
        history[forward_cols].rename(columns={"trade_date": "market_date"}),
        on=["symbol", "market_date"],
        how="left",
    )
    feature_frame = feature_frame[feature_frame["market_date"].isin(evaluable_dates)].copy()
    model_scored, model_kind = _score_candidates(
        feature_frame,
        horizon_days=int(model_horizon_days),
        positive_return=float(positive_return),
    )
    print(f"[model] scored rows {len(model_scored)}", flush=True)

    candidate_pool = _candidate_pool_for_dates(history, [_format_date(value) for value in trade_dates], candidate_pool_limit=5000)
    candidate_pool["market_date"] = pd.to_datetime(candidate_pool["market_date"], errors="coerce").dt.normalize()
    candidate_pool = candidate_pool[candidate_pool["market_date"].isin(evaluable_dates)].copy()
    strategy_eval = candidate_pool.merge(
        history[forward_cols].rename(columns={"trade_date": "market_date"}),
        on=["symbol", "market_date"],
        how="left",
    )
    strategy_eval = strategy_eval.dropna(subset=[f"hold_{int(hold_days)}d_return"]).copy()
    print(f"[strategy] candidate rows {len(strategy_eval)}", flush=True)

    combined_input = candidate_pool.merge(
        model_scored[
            [
                "symbol",
                "market_date",
                "model_probability",
                "model_score",
                f"hold_{int(hold_days)}d_return",
                "max_high_return",
                "max_drawdown",
                "entry_price",
            ]
        ],
        on=["symbol", "market_date"],
        how="inner",
    )
    print(f"[combined] rows {len(combined_input)}", flush=True)

    return_column = f"hold_{int(hold_days)}d_return"
    model_accuracy = {
        "model_kind": model_kind,
        "scored_rows": int(len(model_scored)),
        "symbols": int(model_scored["symbol"].nunique()) if not model_scored.empty else 0,
        "days": int(model_scored["market_date"].nunique()) if not model_scored.empty else 0,
        "auc_positive_return": None,
        "auc_target_return": None,
    }
    model_valid = model_scored.dropna(subset=[return_column, "model_score"]).copy()
    if not model_valid.empty:
        model_accuracy.update(
            {
                "overall_avg_return": round(float(model_valid[return_column].mean()), 6),
                "overall_win_rate": round(float((model_valid[return_column] > 0).mean()), 4),
                "overall_target_hit_rate": round(float((model_valid[return_column] >= float(positive_return)).mean()), 4),
                "auc_positive_return": None
                if _auc((model_valid[return_column] > 0).astype(int), model_valid["model_score"]) is None
                else round(_auc((model_valid[return_column] > 0).astype(int), model_valid["model_score"]), 4),
                "auc_target_return": None
                if _auc((model_valid[return_column] >= float(positive_return)).astype(int), model_valid["model_score"]) is None
                else round(_auc((model_valid[return_column] >= float(positive_return)).astype(int), model_valid["model_score"]), 4),
            }
        )

    model_rules = []
    model_selected_frames = []
    for top_n in (1, 3, 5, 10):
        summary, selected, calendar = _evaluate_daily_selection(
            model_scored,
            rule_name=f"model_only_top{top_n}",
            all_dates=evaluable_dates,
            rank_column="model_score",
            top_n=top_n,
            hold_days=int(hold_days),
            target_return=float(positive_return),
        )
        model_rules.append(summary)
        if top_n in (1, 3, 5, 10):
            model_selected_frames.append(selected.assign(rule=summary["rule"]))

    strategy_accuracy = {
        "candidate_rows": int(len(strategy_eval)),
        "candidate_days": int(strategy_eval["market_date"].nunique()) if not strategy_eval.empty else 0,
        "overall_avg_return": round(float(strategy_eval[return_column].mean()), 6) if not strategy_eval.empty else 0.0,
        "overall_win_rate": round(float((strategy_eval[return_column] > 0).mean()), 4) if not strategy_eval.empty else 0.0,
        "overall_target_hit_rate": round(float((strategy_eval[return_column] >= float(positive_return)).mean()), 4) if not strategy_eval.empty else 0.0,
        "strategy_breakdown": _strategy_breakdown(strategy_eval, int(hold_days), float(positive_return)),
    }
    strategy_rules = []
    strategy_selected_frames = []
    for top_n in (1, 3, 5, 10):
        summary, selected, calendar = _evaluate_daily_selection(
            strategy_eval,
            rule_name=f"strategy_only_top{top_n}",
            all_dates=evaluable_dates,
            rank_column="candidate_priority",
            top_n=top_n,
            hold_days=int(hold_days),
            target_return=float(positive_return),
        )
        strategy_rules.append(summary)
        strategy_selected_frames.append(selected.assign(rule=summary["rule"]))

    combined_rules = []
    combined_selected_frames = []
    for top_n in (1, 3, 5, 10):
        summary, selected, calendar = _evaluate_daily_selection(
            combined_input,
            rule_name=f"combined_top{top_n}",
            all_dates=evaluable_dates,
            rank_column="model_score",
            top_n=top_n,
            hold_days=int(hold_days),
            target_return=float(positive_return),
        )
        combined_rules.append(summary)
        combined_selected_frames.append(selected.assign(rule=summary["rule"]))
    for top_n, threshold in [(1, 68), (2, 68), (3, 68), (5, 66), (10, 64), (10, 62)]:
        summary, selected, calendar = _evaluate_daily_selection(
            combined_input,
            rule_name=f"combined_top{top_n}_score_ge_{threshold}_full",
            all_dates=evaluable_dates,
            rank_column="model_score",
            top_n=top_n,
            min_model_score=float(threshold),
            require_full=True,
            hold_days=int(hold_days),
            target_return=float(positive_return),
        )
        combined_rules.append(summary)
        combined_selected_frames.append(selected.assign(rule=summary["rule"]))

    all_rules = pd.DataFrame([*model_rules, *strategy_rules, *combined_rules])
    if not all_rules.empty:
        all_rules["return_drawdown_ratio"] = all_rules["annualized_return"] / all_rules["max_drawdown"].abs().replace(0.0, np.nan)
        all_rules = all_rules.sort_values(["return_drawdown_ratio", "annualized_return"], ascending=[False, False])

    summary = {
        "date_from": resolved_from,
        "date_to": resolved_to,
        "hold_days": int(hold_days),
        "positive_return": float(positive_return),
        "history_rows": int(len(history)),
        "history_symbols": int(history["symbol"].nunique()) if not history.empty else 0,
        "evaluable_days": int(len(evaluable_dates)),
        "model_accuracy": model_accuracy,
        "strategy_accuracy": strategy_accuracy,
        "model_rules": model_rules,
        "strategy_rules": strategy_rules,
        "combined_rules": combined_rules,
        "best_rules": all_rules.head(10).replace({np.nan: None}).to_dict("records") if not all_rules.empty else [],
    }

    paths = {
        "summary_path": output_path / "summary.json",
        "rule_sensitivity_path": output_path / "rule_sensitivity.csv",
        "model_scores_path": output_path / "model_scores.csv",
        "strategy_candidates_path": output_path / "strategy_candidates.csv",
        "model_selected_path": output_path / "model_selected_trades.csv",
        "strategy_selected_path": output_path / "strategy_selected_trades.csv",
        "combined_selected_path": output_path / "combined_selected_trades.csv",
        "combined_candidates_path": output_path / "combined_candidates.csv",
    }
    paths["summary_path"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    all_rules.to_csv(paths["rule_sensitivity_path"], index=False, encoding="utf-8-sig")
    model_score_columns = [
        column
        for column in [
            "market_date",
            "symbol",
            "name",
            "model_probability",
            "model_score",
            f"hold_{int(hold_days)}d_return",
            "max_high_return",
            "max_drawdown",
        ]
        if column in model_scored.columns
    ]
    model_scored[model_score_columns].to_csv(paths["model_scores_path"], index=False, encoding="utf-8-sig")
    strategy_candidate_columns = [
        column
        for column in [
            "market_date",
            "symbol",
            "name",
            "candidate_strategy",
            "candidate_priority",
            f"hold_{int(hold_days)}d_return",
            "max_high_return",
            "max_drawdown",
        ]
        if column in strategy_eval.columns
    ]
    strategy_eval[strategy_candidate_columns].to_csv(paths["strategy_candidates_path"], index=False, encoding="utf-8-sig")
    pd.concat(model_selected_frames, ignore_index=True).to_csv(paths["model_selected_path"], index=False, encoding="utf-8-sig")
    pd.concat(strategy_selected_frames, ignore_index=True).to_csv(paths["strategy_selected_path"], index=False, encoding="utf-8-sig")
    pd.concat(combined_selected_frames, ignore_index=True).to_csv(paths["combined_selected_path"], index=False, encoding="utf-8-sig")
    combined_input.to_csv(paths["combined_candidates_path"], index=False, encoding="utf-8-sig")
    summary.update({key: str(value) for key, value in paths.items()})
    paths["summary_path"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate model-only, strategy-only, and combined model+strategy selection rules.")
    parser.add_argument("--date-from", default="")
    parser.add_argument("--date-to", default="")
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument("--hold-days", type=int, default=3)
    parser.add_argument("--model-horizon-days", type=int, default=5)
    parser.add_argument("--positive-return", type=float, default=0.03)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = run_validation(
        date_from=args.date_from or None,
        date_to=args.date_to or None,
        months=int(args.months),
        hold_days=int(args.hold_days),
        model_horizon_days=int(args.model_horizon_days),
        positive_return=float(args.positive_return),
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
