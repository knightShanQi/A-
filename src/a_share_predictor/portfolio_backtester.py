from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .evaluation import CANONICAL_EVALUATION_ENGINE, CANONICAL_PRIMARY_METRIC, CANONICAL_PRIMARY_SOURCE


@dataclass(slots=True)
class PortfolioBacktestConfig:
    initial_capital: float = 1_000_000.0
    max_positions: int = 5
    holding_days: int = 3
    transaction_cost_rate: float = 0.0015
    slippage_rate: float = 0.001
    min_lot: int = 100
    price_limit_pct: float | None = 0.10


@dataclass(slots=True)
class PortfolioBacktestResult:
    daily_nav: pd.DataFrame
    trades: pd.DataFrame
    summary: dict[str, object]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if pd.isna(numeric):
        return float(default)
    return float(numeric)


def _normalize_history_frame(frame: pd.DataFrame) -> pd.DataFrame:
    history = frame.copy()
    history["trade_date"] = pd.to_datetime(history["trade_date"], errors="coerce")
    history["symbol"] = history["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    for column in ["open", "close", "high", "low", "pre_close"]:
        if column not in history.columns:
            history[column] = pd.NA
        history[column] = pd.to_numeric(history[column], errors="coerce")
    history = history.dropna(subset=["trade_date", "symbol", "close"]).copy()
    history = history.sort_values(["symbol", "trade_date"]).drop_duplicates(["symbol", "trade_date"], keep="last")
    return history.reset_index(drop=True)


def _build_history_lookup(history: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], list[pd.Timestamp]]:
    normalized = _normalize_history_frame(history)
    lookup: dict[str, pd.DataFrame] = {}
    for symbol, group in normalized.groupby("symbol", sort=False):
        lookup[str(symbol).zfill(6)] = group.reset_index(drop=True)
    calendar = sorted(pd.Timestamp(value) for value in normalized["trade_date"].dropna().unique().tolist())
    return lookup, calendar


def _next_trade_row(frame: pd.DataFrame, signal_date: pd.Timestamp) -> pd.Series | None:
    rows = frame.loc[frame["trade_date"].gt(signal_date)]
    if rows.empty:
        return None
    return rows.iloc[0]


def _exit_trade_row(frame: pd.DataFrame, signal_date: pd.Timestamp, holding_days: int) -> pd.Series | None:
    rows = frame.loc[frame["trade_date"].gt(signal_date)]
    if len(rows) < max(int(holding_days), 1):
        return None
    return rows.iloc[max(int(holding_days), 1) - 1]


def _annualized_return(ending_equity: float, day_count: int) -> float:
    if day_count <= 0 or ending_equity <= 0:
        return 0.0
    return float(ending_equity ** (252.0 / day_count) - 1.0)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max.replace(0.0, pd.NA) - 1.0
    return float(pd.to_numeric(drawdown, errors="coerce").fillna(0.0).min())


def simulate_portfolio_from_candidates(
    candidates: pd.DataFrame,
    history: pd.DataFrame,
    *,
    config: PortfolioBacktestConfig | None = None,
) -> PortfolioBacktestResult:
    cfg = config or PortfolioBacktestConfig()
    if candidates.empty or history.empty:
        empty_daily = pd.DataFrame(columns=["trade_date", "cash", "market_value", "equity", "open_positions"])
        empty_trades = pd.DataFrame()
        return PortfolioBacktestResult(
            daily_nav=empty_daily,
            trades=empty_trades,
            summary={
                "evaluation_engine": CANONICAL_EVALUATION_ENGINE,
                "evaluation_primary_metric": CANONICAL_PRIMARY_METRIC,
                "evaluation_primary_source": CANONICAL_PRIMARY_SOURCE,
                "initial_capital": float(cfg.initial_capital),
                "ending_equity": float(cfg.initial_capital),
                "cumulative_return": 0.0,
                "annualized_return": 0.0,
                "max_drawdown": 0.0,
                "trade_count": 0,
                "win_rate": 0.0,
                "avg_net_return": 0.0,
                "blocked_entry_count": 0,
            },
        )

    history_lookup, calendar = _build_history_lookup(history)
    signals = candidates.copy()
    signals["market_date"] = pd.to_datetime(signals["market_date"], errors="coerce")
    signals["symbol"] = signals["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    signals = signals.dropna(subset=["market_date", "symbol"]).copy()
    priority_columns = [column for column in ["market_date", "candidate_priority", "strategy_rank", "amount"] if column in signals.columns]
    if priority_columns:
        ascending = [True] + [False] * (len(priority_columns) - 1)
        signals = signals.sort_values(priority_columns, ascending=ascending)

    pending_by_entry_date: dict[pd.Timestamp, list[dict[str, object]]] = {}
    for row in signals.to_dict("records"):
        symbol = str(row.get("symbol", "")).zfill(6)
        frame = history_lookup.get(symbol)
        if frame is None or frame.empty:
            continue
        signal_date = pd.Timestamp(row["market_date"])
        entry_row = _next_trade_row(frame, signal_date)
        exit_row = _exit_trade_row(frame, signal_date, int(cfg.holding_days))
        if entry_row is None or exit_row is None:
            continue
        entry_date = pd.Timestamp(entry_row["trade_date"])
        pending_by_entry_date.setdefault(entry_date, []).append(
            {
                **row,
                "entry_date": entry_date,
                "planned_exit_date": pd.Timestamp(exit_row["trade_date"]),
                "entry_open": _safe_float(entry_row.get("open"), _safe_float(entry_row.get("close"), 0.0)),
                "entry_pre_close": _safe_float(entry_row.get("pre_close"), 0.0),
                "planned_exit_close": _safe_float(exit_row.get("close"), 0.0),
            }
        )

    cash = float(cfg.initial_capital)
    open_positions: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    nav_rows: list[dict[str, object]] = []
    blocked_entry_count = 0

    for trade_date in calendar:
        current_date = pd.Timestamp(trade_date)

        exiting = [position for position in open_positions if pd.Timestamp(position["exit_date"]) == current_date]
        remaining_positions = [position for position in open_positions if pd.Timestamp(position["exit_date"]) != current_date]
        for position in exiting:
            frame = history_lookup[position["symbol"]]
            day_rows = frame.loc[frame["trade_date"].eq(current_date)]
            if day_rows.empty:
                continue
            close_price = _safe_float(day_rows.iloc[0].get("close"), 0.0)
            if close_price <= 0:
                continue
            sell_price = close_price * (1.0 - float(cfg.slippage_rate))
            gross_proceeds = float(position["shares"]) * sell_price
            fees = gross_proceeds * float(cfg.transaction_cost_rate)
            net_proceeds = gross_proceeds - fees
            cash += net_proceeds
            trade_rows.append(
                {
                    "symbol": position["symbol"],
                    "market_date": position.get("market_date"),
                    "signal_date": position["signal_date"],
                    "entry_date": position["entry_date"],
                    "exit_date": current_date,
                    "name": position.get("name", ""),
                    "candidate_strategy": position.get("candidate_strategy", ""),
                    "candidate_priority": position.get("candidate_priority"),
                    "strategy_rank": position.get("strategy_rank"),
                    "daily_rank": position.get("daily_rank"),
                    "model_probability": position.get("model_probability"),
                    "model_score": position.get("model_score"),
                    "context_composite_score": position.get("context_composite_score"),
                    "entry_price": float(position["entry_price"]),
                    "exit_price": close_price,
                    "shares": int(position["shares"]),
                    "invested_capital": float(position["invested_capital"]),
                    "gross_return": close_price / float(position["entry_price"]) - 1.0,
                    "net_return": net_proceeds / float(position["invested_capital"]) - 1.0,
                }
            )
        open_positions = remaining_positions

        entries = pending_by_entry_date.get(current_date, [])
        available_slots = max(int(cfg.max_positions) - len(open_positions), 0)
        if entries and available_slots > 0 and cash > 0:
            selected_entries = entries[:available_slots]
            target_notional = cash / max(len(selected_entries), 1)
            for entry in selected_entries:
                entry_price = float(entry["entry_open"])
                if entry_price <= 0:
                    continue
                previous_close = float(entry.get("entry_pre_close") or 0.0)
                if cfg.price_limit_pct is not None and previous_close > 0:
                    limit_up_price = previous_close * (1.0 + float(cfg.price_limit_pct))
                    if entry_price >= limit_up_price * 0.999:
                        blocked_entry_count += 1
                        continue
                buy_price = entry_price * (1.0 + float(cfg.slippage_rate))
                raw_shares = int(target_notional / max(buy_price * (1.0 + float(cfg.transaction_cost_rate)), 1e-9))
                shares = (raw_shares // max(int(cfg.min_lot), 1)) * max(int(cfg.min_lot), 1)
                if shares <= 0:
                    continue
                gross_cost = float(shares) * buy_price
                fees = gross_cost * float(cfg.transaction_cost_rate)
                total_cost = gross_cost + fees
                if total_cost > cash:
                    continue
                cash -= total_cost
                open_positions.append(
                    {
                        "symbol": str(entry["symbol"]).zfill(6),
                        "signal_date": pd.Timestamp(entry["market_date"]),
                        "market_date": pd.Timestamp(entry["market_date"]),
                        "entry_date": current_date,
                        "exit_date": pd.Timestamp(entry["planned_exit_date"]),
                        "name": entry.get("name", ""),
                        "candidate_strategy": entry.get("candidate_strategy", ""),
                        "candidate_priority": entry.get("candidate_priority"),
                        "strategy_rank": entry.get("strategy_rank"),
                        "daily_rank": entry.get("daily_rank"),
                        "model_probability": entry.get("model_probability"),
                        "model_score": entry.get("model_score"),
                        "context_composite_score": entry.get("context_composite_score"),
                        "entry_price": entry_price,
                        "shares": shares,
                        "invested_capital": total_cost,
                    }
                )

        market_value = 0.0
        for position in open_positions:
            frame = history_lookup[position["symbol"]]
            day_rows = frame.loc[frame["trade_date"].eq(current_date)]
            if day_rows.empty:
                last_rows = frame.loc[frame["trade_date"].lt(current_date)]
                if last_rows.empty:
                    continue
                mark_price = _safe_float(last_rows.iloc[-1].get("close"), 0.0)
            else:
                mark_price = _safe_float(day_rows.iloc[0].get("close"), 0.0)
            market_value += float(position["shares"]) * mark_price

        nav_rows.append(
            {
                "trade_date": current_date,
                "cash": float(cash),
                "market_value": float(market_value),
                "equity": float(cash + market_value),
                "open_positions": int(len(open_positions)),
            }
        )

    daily_nav = pd.DataFrame(nav_rows)
    trades = pd.DataFrame(trade_rows)
    ending_equity = float(daily_nav["equity"].iloc[-1]) if not daily_nav.empty else float(cfg.initial_capital)
    cumulative_return = ending_equity / float(cfg.initial_capital) - 1.0 if cfg.initial_capital > 0 else 0.0
    net_returns = pd.to_numeric(trades.get("net_return", pd.Series(dtype=float)), errors="coerce").dropna()
    summary = {
        "evaluation_engine": CANONICAL_EVALUATION_ENGINE,
        "evaluation_primary_metric": CANONICAL_PRIMARY_METRIC,
        "evaluation_primary_source": CANONICAL_PRIMARY_SOURCE,
        "initial_capital": float(cfg.initial_capital),
        "ending_equity": ending_equity,
        "cumulative_return": float(cumulative_return),
        "annualized_return": _annualized_return(ending_equity / float(cfg.initial_capital), len(daily_nav))
        if cfg.initial_capital > 0
        else 0.0,
        "max_drawdown": _max_drawdown(daily_nav["equity"]) if not daily_nav.empty else 0.0,
        "trade_count": int(len(trades)),
        "win_rate": float((net_returns > 0).mean()) if not net_returns.empty else 0.0,
        "avg_net_return": float(net_returns.mean()) if not net_returns.empty else 0.0,
        "blocked_entry_count": int(blocked_entry_count),
    }
    return PortfolioBacktestResult(daily_nav=daily_nav, trades=trades, summary=summary)
