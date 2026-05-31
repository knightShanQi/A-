from __future__ import annotations

import pandas as pd

from a_share_predictor.portfolio_backtester import PortfolioBacktestConfig, simulate_portfolio_from_candidates


def _make_history() -> pd.DataFrame:
    rows = []
    for symbol, base in [("000001", 10.0), ("000002", 20.0), ("000003", 30.0)]:
        dates = pd.bdate_range("2026-01-05", periods=6)
        for offset, trade_date in enumerate(dates):
            open_price = base + offset
            close_price = open_price + 0.5
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "open": open_price,
                    "close": close_price,
                    "high": close_price + 0.2,
                    "low": open_price - 0.2,
                }
            )
    return pd.DataFrame(rows)


def test_simulate_portfolio_from_candidates_limits_positions_and_marks_nav():
    history = _make_history()
    candidates = pd.DataFrame(
        [
            {"market_date": "2026-01-05", "symbol": "000001", "candidate_priority": 90},
            {"market_date": "2026-01-05", "symbol": "000002", "candidate_priority": 80},
            {"market_date": "2026-01-05", "symbol": "000003", "candidate_priority": 70},
        ]
    )

    result = simulate_portfolio_from_candidates(
        candidates,
        history,
        config=PortfolioBacktestConfig(
            initial_capital=100_000.0,
            max_positions=2,
            holding_days=2,
            transaction_cost_rate=0.0,
            slippage_rate=0.0,
            min_lot=100,
        ),
    )

    assert result.summary["trade_count"] == 2
    assert result.daily_nav["open_positions"].max() == 2
    assert result.summary["ending_equity"] > 100_000.0
    assert set(result.trades["symbol"]) == {"000001", "000002"}


def test_simulate_portfolio_from_candidates_applies_costs():
    history = _make_history()
    candidates = pd.DataFrame([{"market_date": "2026-01-05", "symbol": "000001", "candidate_priority": 90}])

    no_cost = simulate_portfolio_from_candidates(
        candidates,
        history,
        config=PortfolioBacktestConfig(
            initial_capital=50_000.0,
            max_positions=1,
            holding_days=2,
            transaction_cost_rate=0.0,
            slippage_rate=0.0,
            min_lot=100,
        ),
    )
    with_cost = simulate_portfolio_from_candidates(
        candidates,
        history,
        config=PortfolioBacktestConfig(
            initial_capital=50_000.0,
            max_positions=1,
            holding_days=2,
            transaction_cost_rate=0.002,
            slippage_rate=0.001,
            min_lot=100,
        ),
    )

    assert no_cost.summary["ending_equity"] > with_cost.summary["ending_equity"]
    assert no_cost.trades.iloc[0]["net_return"] > with_cost.trades.iloc[0]["net_return"]
