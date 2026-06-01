from __future__ import annotations

from a_share_predictor.services.backtest_service import BacktestService


def test_backtest_service_delegates_market_backtest(monkeypatch):
    captured = {}

    def fake_run_full_market_backtest(**kwargs):
        captured.update(kwargs)
        return {"summary": {"evaluation_engine": "unified_portfolio_nav_v1"}}

    monkeypatch.setattr("a_share_predictor.services.backtest_service.run_full_market_backtest", fake_run_full_market_backtest)

    payload = BacktestService().run_market_backtest(
        date_from="2026-04-01",
        date_to="2026-04-30",
        horizon_days=5,
        positive_return=0.03,
        strategy_mode="strategy1",
        top_k=10,
    )

    assert payload["summary"]["evaluation_engine"] == "unified_portfolio_nav_v1"
    assert captured["horizon_days"] == 5
    assert captured["strategy_mode"] == "strategy1"
