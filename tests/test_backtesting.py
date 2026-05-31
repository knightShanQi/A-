import pandas as pd

import a_share_predictor.backtesting as backtesting


def make_backtest_daily() -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=14, freq="B")
    close = [10.0, 10.1, 11.0, 10.2, 10.1, 11.2, 10.3, 10.2, 11.1, 10.4, 10.3, 9.5, 10.0, 10.1]
    return pd.DataFrame(
        {
            "date": dates,
            "open": [10.0] * len(dates),
            "close": close,
            "high": [value + 0.2 for value in close],
            "low": [value - 0.2 for value in close],
            "volume": [100000] * len(dates),
            "amount": [100000 * value for value in close],
            "turnover": [2.5] * len(dates),
        }
    )


def test_run_daily_strategy_backtest_hits_target(monkeypatch):
    daily = make_backtest_daily()
    signals = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2026-01-02", "2026-01-07", "2026-01-12", "2026-01-15"]),
            "probability": [0.92, 0.88, 0.86, 0.72],
            "trend_score": [75.0, 72.0, 74.0, 60.0],
            "breakout_score": [70.0, 68.0, 69.0, 58.0],
            "pullback_score": [66.0, 64.0, 65.0, 54.0],
            "risk_score": [30.0, 34.0, 32.0, 40.0],
            "close_vs_ma20": [0.05, 0.04, 0.03, 0.01],
            "future_return": [0.08, 0.09, 0.08, -0.05],
        }
    )

    monkeypatch.setattr(backtesting, "_build_signal_frame", lambda *args, **kwargs: signals.copy())

    result = backtesting.run_daily_strategy_backtest(
        daily,
        horizon_days=2,
        positive_return=0.03,
        target_precision=0.90,
        min_trades=3,
        transaction_cost=0.0,
    )

    assert result.target_reached is True
    assert result.selected_threshold == 0.55
    assert result.selected_quality_threshold >= 68.0
    assert result.trade_count == 3
    assert result.achieved_precision == 1.0
    assert result.latest_signal_active is False


def test_run_daily_strategy_backtest_downgrades_when_target_not_met(monkeypatch):
    daily = make_backtest_daily()
    signals = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2026-01-02", "2026-01-07", "2026-01-12", "2026-01-15"]),
            "probability": [0.92, 0.88, 0.84, 0.80],
            "trend_score": [74.0, 74.0, 74.0, 74.0],
            "breakout_score": [68.0, 68.0, 68.0, 68.0],
            "pullback_score": [64.0, 64.0, 64.0, 64.0],
            "risk_score": [28.0, 28.0, 28.0, 28.0],
            "close_vs_ma20": [0.05, 0.04, 0.02, 0.01],
            "future_return": [0.08, -0.07, 0.08, -0.05],
        }
    )

    monkeypatch.setattr(backtesting, "_build_signal_frame", lambda *args, **kwargs: signals.copy())

    result = backtesting.run_daily_strategy_backtest(
        daily,
        horizon_days=2,
        positive_return=0.03,
        target_precision=0.90,
        min_trades=3,
        transaction_cost=0.0,
    )

    assert result.target_reached is False
    assert result.trade_count >= 3
    assert result.achieved_precision < 0.90


def test_simulate_trades_skips_extreme_extension_with_weak_probability():
    daily = make_backtest_daily()
    signals = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2026-01-02", "2026-01-07"]),
            "probability": [0.66, 0.78],
            "trend_score": [78.0, 78.0],
            "breakout_score": [70.0, 70.0],
            "pullback_score": [62.0, 62.0],
            "risk_score": [70.0, 64.0],
            "setup_score": [66.0, 66.0],
            "quality_gate": [68.0, 68.0],
            "close_vs_ma20": [0.08, 0.04],
            "close_vs_ma60": [0.18, 0.06],
            "close_vs_ma120": [0.34, 0.08],
            "volatility_contraction": [0.18, -0.12],
            "turnover_ratio_20": [0.82, 1.05],
            "downside_vol_ratio_20": [0.60, 0.30],
            "future_return": [-0.05, 0.08],
        }
    )

    trades, _equity_curve = backtesting._simulate_trades(
        daily,
        signals,
        threshold=0.55,
        quality_threshold=56.0,
        horizon_days=2,
        transaction_cost=0.0,
    )

    assert len(trades) == 1
    assert trades.iloc[0]["signal_date"] == pd.Timestamp("2026-01-07")
