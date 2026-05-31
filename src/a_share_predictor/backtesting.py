from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import build_training_frame
from .modeling import (
    ProbabilityResult,
    _augment_model_features,
    _build_signal_breakdown,
    train_probability_model,
)
from .stages import main_rise_start_score


@dataclass(slots=True)
class BacktestResult:
    target_precision: float
    selected_threshold: float
    selected_quality_threshold: float
    achieved_precision: float
    target_reached: bool
    trade_count: int
    avg_trade_return: float
    cumulative_return: float
    annualized_return: float
    max_drawdown: float
    latest_signal_probability: float
    latest_signal_quality: float
    latest_signal_active: bool
    status_label: str
    selection_summary: str
    summary: str
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    threshold_table: pd.DataFrame


def _clip(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return float(max(lower, min(value, upper)))


def _daily_dates(daily: pd.DataFrame) -> pd.DatetimeIndex:
    if "date" in daily.columns:
        return pd.to_datetime(daily["date"], errors="coerce")
    return pd.to_datetime(daily.index, errors="coerce")


def _empty_backtest(target_precision: float) -> BacktestResult:
    empty_curve = pd.DataFrame(columns=["date", "equity"])
    empty_trades = pd.DataFrame(
        columns=[
            "signal_date",
            "entry_date",
            "exit_date",
            "probability",
            "threshold",
            "quality_threshold",
            "entry_price",
            "exit_price",
            "gross_return",
            "net_return",
            "holding_days",
            "win",
        ]
    )
    empty_thresholds = pd.DataFrame(
        columns=["threshold", "quality_threshold", "trade_count", "win_rate", "cumulative_return", "max_drawdown"]
    )
    return BacktestResult(
        target_precision=target_precision,
        selected_threshold=0.0,
        selected_quality_threshold=0.0,
        achieved_precision=0.0,
        target_reached=False,
        trade_count=0,
        avg_trade_return=0.0,
        cumulative_return=0.0,
        annualized_return=0.0,
        max_drawdown=0.0,
        latest_signal_probability=0.0,
        latest_signal_quality=0.0,
        latest_signal_active=False,
        status_label="样本不足",
        selection_summary="历史样本或可交易信号不足，当前回测先不输出交易结论。",
        summary="尚未形成可用的日频回测样本。",
        equity_curve=empty_curve,
        trades=empty_trades,
        threshold_table=empty_thresholds,
    )


def _build_signal_frame(
    daily: pd.DataFrame,
    horizon_days: int,
    positive_return: float,
    min_train_size: int,
    model_result: ProbabilityResult | None = None,
) -> pd.DataFrame:
    dataset = _augment_model_features(
        build_training_frame(daily, horizon_days=horizon_days, positive_return=positive_return)
    )
    if len(dataset) < min_train_size + 10 or dataset["target"].nunique() < 2:
        return pd.DataFrame()

    result = model_result or train_probability_model(daily, horizon_days=horizon_days, positive_return=positive_return)
    probabilities = result.out_of_sample_probabilities.copy()
    if probabilities.empty:
        probabilities = pd.Series(dtype=float)
    probabilities = probabilities.reindex(dataset.index)
    valid_probabilities = probabilities.dropna()
    if len(valid_probabilities) < 10:
        return pd.DataFrame()

    rows: list[dict[str, float | pd.Timestamp]] = []
    for signal_date, probability in valid_probabilities.items():
        idx = int(dataset.index.get_loc(signal_date))
        latest_features = dataset.iloc[idx]
        breakdown = _build_signal_breakdown(latest_features)
        rows.append(
            {
                "signal_date": pd.Timestamp(signal_date),
                "probability": float(probability),
                "trend_score": float(breakdown["trend_score"]),
                "breakout_score": float(breakdown["breakout_score"]),
                "pullback_score": float(breakdown["pullback_score"]),
                "risk_score": float(breakdown["risk_score"]),
                "setup_score": float(
                    breakdown["trend_score"] * 0.32
                    + breakdown["breakout_score"] * 0.24
                    + breakdown["pullback_score"] * 0.18
                    + (100 - breakdown["risk_score"]) * 0.26
                ),
                "quality_gate": float(
                    float(probability) * 100 * 0.58
                    + (
                        breakdown["trend_score"] * 0.32
                        + breakdown["breakout_score"] * 0.24
                        + breakdown["pullback_score"] * 0.18
                        + (100 - breakdown["risk_score"]) * 0.26
                    )
                    * 0.42
                ),
                "ma_alignment_score": float(latest_features.get("ma_alignment_score", 0.5)),
                "momentum_persistence_10": float(latest_features.get("momentum_persistence_10", 0.5)),
                "efficiency_ratio_10": float(latest_features.get("efficiency_ratio_10", 0.4)),
                "volatility_contraction": float(latest_features.get("volatility_contraction", 0.0)),
                "turnover_ratio_20": float(latest_features.get("turnover_ratio_20", 1.0)),
                "drawdown_20": float(latest_features.get("drawdown_20", -0.03)),
                "downside_vol_ratio_20": float(latest_features.get("downside_vol_ratio_20", 0.4)),
                "close_near_high_5": float(latest_features.get("close_near_high_5", -0.02)),
                "ret_20": float(latest_features.get("ret_20", 0.0)),
                "breakout_distance_20": float(latest_features.get("breakout_distance_20", -0.03)),
                "range_position_20": float(latest_features.get("range_position_20", 0.5)),
                "consolidation_width_20": float(latest_features.get("consolidation_width_20", 0.25)),
                "close_vs_ma20": float(latest_features.get("close_vs_ma20", 0.0)),
                "close_vs_ma60": float(latest_features.get("close_vs_ma60", 0.0)),
                "close_vs_ma120": float(latest_features.get("close_vs_ma120", 0.0)),
                "launch_readiness": float(main_rise_start_score(latest_features)),
                "future_return": float(latest_features.get("future_return", 0.0)),
            }
        )
    signal_frame = pd.DataFrame(rows)
    if signal_frame.empty:
        return signal_frame

    base_setup_score = signal_frame["setup_score"].astype(float).copy()
    base_quality_gate = signal_frame["quality_gate"].astype(float).copy()
    trend_quality = (
        signal_frame["ma_alignment_score"].fillna(0.5) * 10
        + signal_frame["momentum_persistence_10"].fillna(0.5) * 8
        + signal_frame["efficiency_ratio_10"].fillna(0.4) * 6
    )
    contraction_bonus = (-signal_frame["volatility_contraction"].clip(upper=0).fillna(0.0)) * 6
    breakout_alignment = (
        (0.05 - signal_frame["close_near_high_5"].abs().fillna(0.02).clip(upper=0.12)) * 35
    )
    launch_alignment = (
        signal_frame["launch_readiness"].fillna(50.0)
        + (0.06 - signal_frame["breakout_distance_20"].sub(0.008).abs().fillna(0.05).clip(upper=0.08)) * 120
        + (0.18 - signal_frame["consolidation_width_20"].fillna(0.25).clip(upper=0.30)) * 42
        + (0.22 - signal_frame["range_position_20"].sub(0.68).abs().fillna(0.18).clip(upper=0.24)) * 24
    )
    turnover_support = (signal_frame["turnover_ratio_20"].fillna(1.0).clip(lower=0.6, upper=2.2) - 1.0) * 5
    extension_penalty = (
        (signal_frame["close_vs_ma20"].clip(lower=0.05).fillna(0.0) - 0.05).clip(lower=0) * 140
        + (signal_frame["close_vs_ma60"].clip(lower=0.10).fillna(0.0) - 0.10).clip(lower=0) * 190
        + (signal_frame["close_vs_ma120"].clip(lower=0.18).fillna(0.0) - 0.18).clip(lower=0) * 210
    )
    expansion_penalty = signal_frame["volatility_contraction"].clip(lower=0).fillna(0.0) * 20
    downside_penalty = signal_frame["downside_vol_ratio_20"].fillna(0.4) * 8
    quality_adjustment = (
        trend_quality * 0.16
        + contraction_bonus * 0.24
        + breakout_alignment
        + (launch_alignment - 50.0) * 0.14
        + turnover_support * 0.35
        - expansion_penalty * 0.10
        - extension_penalty * 0.08
        - downside_penalty * 0.08
    )

    signal_frame["setup_score"] = (base_setup_score + (signal_frame["launch_readiness"].fillna(50.0) - 50.0) * 0.10).clip(lower=0, upper=100)
    signal_frame["quality_gate"] = (
        base_quality_gate
        + quality_adjustment
    ).clip(lower=0, upper=100)
    return signal_frame


def _max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    rolling_peak = equity_curve.cummax()
    drawdown = equity_curve / rolling_peak - 1
    return float(drawdown.min())


def _simulate_trades(
    daily: pd.DataFrame,
    signals: pd.DataFrame,
    threshold: float,
    quality_threshold: float,
    horizon_days: int,
    transaction_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if signals.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["date", "equity"])

    daily_view = daily.copy().reset_index(drop=True)
    daily_view["date"] = _daily_dates(daily_view)
    daily_view = daily_view.dropna(subset=["date", "open", "close"]).copy()
    if daily_view.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["date", "equity"])

    position_lookup = {pd.Timestamp(date): idx for idx, date in enumerate(daily_view["date"])}
    trades: list[dict[str, object]] = []
    next_free_entry = 0

    for row in signals.itertuples(index=False):
        signal_date = pd.Timestamp(row.signal_date)
        signal_pos = position_lookup.get(signal_date)
        if signal_pos is None:
            continue
        entry_pos = signal_pos + 1
        exit_pos = signal_pos + horizon_days
        if entry_pos >= len(daily_view) or exit_pos >= len(daily_view):
            continue
        if entry_pos < next_free_entry:
            continue
        if float(row.probability) < threshold:
            continue
        setup_score = float(
            getattr(
                row,
                "setup_score",
                float(row.trend_score) * 0.32
                + float(row.breakout_score) * 0.24
                + float(row.pullback_score) * 0.18
                + (100 - float(row.risk_score)) * 0.26,
            )
        )
        quality_gate = float(getattr(row, "quality_gate", float(row.probability) * 100 * 0.58 + setup_score * 0.42))
        if setup_score < 54 or quality_gate < quality_threshold:
            continue
        if float(row.risk_score) >= 92 and float(row.probability) < max(threshold + 0.12, 0.82):
            continue
        if float(row.risk_score) >= 86 and float(row.probability) < max(threshold + 0.08, 0.76):
            continue
        if float(row.risk_score) >= 70 and float(row.probability) < max(threshold + 0.06, 0.74):
            continue
        close_vs_ma120 = float(getattr(row, "close_vs_ma120", 0.0))
        if close_vs_ma120 > 0.30 and float(row.risk_score) > 68 and float(row.probability) < max(threshold + 0.10, 0.78):
            continue

        entry_price = float(daily_view.loc[entry_pos, "open"])
        exit_price = float(daily_view.loc[exit_pos, "close"])
        if entry_price <= 0:
            continue
        gross_return = exit_price / entry_price - 1
        net_return = gross_return - float(transaction_cost)
        trades.append(
            {
                "signal_date": signal_date,
                "entry_date": pd.Timestamp(daily_view.loc[entry_pos, "date"]),
                "exit_date": pd.Timestamp(daily_view.loc[exit_pos, "date"]),
                "probability": float(row.probability),
                "threshold": float(threshold),
                "quality_threshold": float(quality_threshold),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": float(gross_return),
                "net_return": float(net_return),
                "holding_days": int(exit_pos - entry_pos + 1),
                "win": bool(net_return > 0),
            }
        )
        next_free_entry = exit_pos + 1

    trades_df = pd.DataFrame(trades)
    equity_curve = pd.DataFrame({"date": daily_view["date"], "equity": 1.0})
    if trades_df.empty:
        return trades_df, equity_curve

    equity_value = 1.0
    for trade in trades:
        exit_date = pd.Timestamp(trade["exit_date"])
        exit_mask = equity_curve["date"] >= exit_date
        equity_value *= 1 + float(trade["net_return"])
        equity_curve.loc[exit_mask, "equity"] = equity_value

    return trades_df, equity_curve


def _summarize_trade_set(trades_df: pd.DataFrame, equity_curve: pd.DataFrame) -> dict[str, float]:
    if trades_df.empty:
        return {
            "trade_count": 0.0,
            "win_rate": 0.0,
            "avg_trade_return": 0.0,
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
        }

    trade_count = float(len(trades_df))
    win_rate = float(trades_df["win"].mean())
    avg_trade_return = float(trades_df["net_return"].mean())
    cumulative_return = float(equity_curve["equity"].iloc[-1] - 1)
    max_drawdown = float(_max_drawdown(equity_curve["equity"]))
    elapsed_days = max((pd.Timestamp(trades_df["exit_date"].max()) - pd.Timestamp(trades_df["entry_date"].min())).days, 1)
    annualized_return = float((1 + cumulative_return) ** (365 / elapsed_days) - 1) if cumulative_return > -1 else -1.0
    return {
        "trade_count": trade_count,
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_return,
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
    }


def _select_threshold(threshold_table: pd.DataFrame, target_precision: float, min_trades: int) -> pd.Series | None:
    if threshold_table.empty:
        return None

    eligible = threshold_table[
        (threshold_table["win_rate"] >= target_precision) & (threshold_table["trade_count"] >= min_trades)
    ]
    if not eligible.empty:
        return eligible.sort_values(
            ["cumulative_return", "win_rate", "trade_count", "threshold", "quality_threshold"],
            ascending=[False, False, False, True, True],
        ).iloc[0]

    fallback = threshold_table[threshold_table["trade_count"] >= min_trades]
    if not fallback.empty:
        return fallback.sort_values(
            ["win_rate", "cumulative_return", "trade_count", "threshold", "quality_threshold"],
            ascending=[False, False, False, True, True],
        ).iloc[0]

    non_empty = threshold_table[threshold_table["trade_count"] > 0]
    if not non_empty.empty:
        return non_empty.sort_values(
            ["win_rate", "trade_count", "threshold", "quality_threshold"],
            ascending=[False, False, True, True],
        ).iloc[0]
    return threshold_table.iloc[0]


def run_daily_strategy_backtest(
    daily: pd.DataFrame,
    horizon_days: int = 5,
    positive_return: float = 0.03,
    *,
    target_precision: float = 0.90,
    min_train_size: int = 120,
    min_trades: int = 6,
    transaction_cost: float = 0.0015,
    model_result: ProbabilityResult | None = None,
) -> BacktestResult:
    signals = _build_signal_frame(
        daily=daily,
        horizon_days=horizon_days,
        positive_return=positive_return,
        min_train_size=min_train_size,
        model_result=model_result,
    )
    if signals.empty:
        return _empty_backtest(target_precision)

    threshold_rows: list[dict[str, float]] = []
    simulations: dict[tuple[float, float], tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]] = {}
    thresholds = [round(value, 2) for value in np.arange(0.55, 0.91, 0.05)]
    quality_thresholds = [56.0, 60.0, 64.0, 68.0, 72.0]
    for threshold in thresholds:
        for quality_threshold in quality_thresholds:
            trades_df, equity_curve = _simulate_trades(
                daily=daily,
                signals=signals,
                threshold=threshold,
                quality_threshold=quality_threshold,
                horizon_days=horizon_days,
                transaction_cost=transaction_cost,
            )
            metrics = _summarize_trade_set(trades_df, equity_curve)
            threshold_rows.append(
                {
                    "threshold": float(threshold),
                    "quality_threshold": float(quality_threshold),
                    **metrics,
                }
            )
            simulations[(float(threshold), float(quality_threshold))] = (trades_df, equity_curve, metrics)

    threshold_table = pd.DataFrame(threshold_rows)
    selected = _select_threshold(threshold_table, target_precision=target_precision, min_trades=min_trades)
    if selected is None:
        return _empty_backtest(target_precision)

    selected_threshold = float(selected["threshold"])
    selected_quality_threshold = float(selected["quality_threshold"])
    trades_df, equity_curve, metrics = simulations[(selected_threshold, selected_quality_threshold)]
    achieved_precision = float(metrics["win_rate"])
    target_reached = bool(achieved_precision >= target_precision and metrics["trade_count"] >= min_trades)
    latest_signal_probability = float(signals["probability"].iloc[-1])
    latest_signal_quality = float(signals["quality_gate"].iloc[-1]) if "quality_gate" in signals.columns else 0.0
    latest_signal_active = bool(
        target_reached
        and latest_signal_probability >= selected_threshold
        and latest_signal_quality >= selected_quality_threshold
    )

    if target_reached:
        status_label = "达标可交易"
        selection_summary = (
            f"在至少 {min_trades} 笔交易的前提下，阈值 `{selected_threshold:.2f}` 的历史命中率达到 "
            f"`{achieved_precision * 100:.1f}%`，并通过质量门槛 `{selected_quality_threshold:.0f}`，满足你设定的 90% 目标。"
        )
    else:
        status_label = "未达标先观察"
        selection_summary = (
            f"当前样本下，没有任何阈值在至少 {min_trades} 笔交易时达到 90% 命中率；"
            f"已退回到最优历史组合 `概率阈值 {selected_threshold:.2f} / 质量门槛 {selected_quality_threshold:.0f}`，"
            f"其命中率为 `{achieved_precision * 100:.1f}%`。"
        )

    summary = (
        f"共回测 {int(metrics['trade_count'])} 笔交易，平均单笔收益 `{metrics['avg_trade_return'] * 100:.2f}%`，"
        f"累计收益 `{metrics['cumulative_return'] * 100:.2f}%`，最大回撤 `{abs(metrics['max_drawdown']) * 100:.2f}%`。"
    )

    return BacktestResult(
        target_precision=target_precision,
        selected_threshold=selected_threshold,
        selected_quality_threshold=selected_quality_threshold,
        achieved_precision=achieved_precision,
        target_reached=target_reached,
        trade_count=int(metrics["trade_count"]),
        avg_trade_return=float(metrics["avg_trade_return"]),
        cumulative_return=float(metrics["cumulative_return"]),
        annualized_return=float(metrics["annualized_return"]),
        max_drawdown=float(metrics["max_drawdown"]),
        latest_signal_probability=latest_signal_probability,
        latest_signal_quality=latest_signal_quality,
        latest_signal_active=latest_signal_active,
        status_label=status_label,
        selection_summary=selection_summary,
        summary=summary,
        equity_curve=equity_curve,
        trades=trades_df,
        threshold_table=threshold_table,
    )
