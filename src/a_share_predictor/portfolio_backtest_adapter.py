from __future__ import annotations

import pandas as pd

from .portfolio_backtester import PortfolioBacktestConfig, PortfolioBacktestResult, simulate_portfolio_from_candidates


def simulate_selected_portfolio(
    selected: pd.DataFrame,
    history: pd.DataFrame,
    *,
    max_positions: int,
    holding_days: int,
    initial_capital: float = 1_000_000.0,
) -> PortfolioBacktestResult:
    if selected.empty or history.empty:
        return simulate_portfolio_from_candidates(
            pd.DataFrame(columns=["market_date", "symbol"]),
            pd.DataFrame(columns=["trade_date", "symbol", "open", "high", "low", "close"]),
            config=PortfolioBacktestConfig(
                initial_capital=float(initial_capital),
                max_positions=max(int(max_positions), 1),
                holding_days=max(int(holding_days), 1),
            ),
        )
    candidates = selected.copy()
    candidates["market_date"] = pd.to_datetime(candidates["market_date"], errors="coerce")
    candidates["symbol"] = candidates["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)
    candidates = candidates.dropna(subset=["market_date", "symbol"]).copy()
    return simulate_portfolio_from_candidates(
        candidates,
        history,
        config=PortfolioBacktestConfig(
            initial_capital=float(initial_capital),
            max_positions=max(int(max_positions), 1),
            holding_days=max(int(holding_days), 1),
        ),
    )


def portfolio_summary_fields(result: PortfolioBacktestResult, *, prefix: str = "portfolio_") -> dict[str, object]:
    summary = result.summary
    return {
        f"{prefix}evaluation_engine": str(summary.get("evaluation_engine", "")),
        f"{prefix}evaluation_primary_metric": str(summary.get("evaluation_primary_metric", "")),
        f"{prefix}evaluation_primary_source": str(summary.get("evaluation_primary_source", "")),
        f"{prefix}trade_count": int(summary.get("trade_count", 0)),
        f"{prefix}win_rate": round(float(summary.get("win_rate", 0.0)), 4),
        f"{prefix}avg_net_return": round(float(summary.get("avg_net_return", 0.0)), 6),
        f"{prefix}blocked_entry_count": int(summary.get("blocked_entry_count", 0)),
        f"{prefix}ending_equity": round(float(summary.get("ending_equity", 0.0)), 6),
        f"{prefix}cumulative_return": round(float(summary.get("cumulative_return", 0.0)), 6),
        f"{prefix}annualized_return": round(float(summary.get("annualized_return", 0.0)), 6),
        f"{prefix}max_drawdown": round(float(summary.get("max_drawdown", 0.0)), 6),
    }
