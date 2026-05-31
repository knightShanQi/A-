import pandas as pd

import a_share_predictor.market_backtest_runner as runner


def test_run_full_market_backtest_reports_progress(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "fetch_a_share_universe", lambda: pd.DataFrame([{"symbol": "600001", "name": "A"}]))
    monkeypatch.setattr(runner, "fetch_tushare_recent_trade_dates", lambda end_date=None, limit=420: ["20260421"])
    monkeypatch.setattr(runner, "build_market_daily_feature_store", lambda *args, **kwargs: pd.DataFrame({"symbol": ["600001"]}))
    monkeypatch.setattr(
        runner,
        "build_market_candidate_pool_store",
        lambda *args, **kwargs: pd.DataFrame([{"symbol": "600001", "name": "A", "candidate_strategy": "strategy1"}]),
    )
    monkeypatch.setattr(
        runner,
        "fetch_daily_history",
        lambda symbol, start_date=None: pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-04-21", "2026-04-22", "2026-04-23", "2026-04-24"]),
                "open": [10.0, 10.0, 10.0, 10.0],
                "close": [10.0, 10.2, 10.4, 10.6],
                "high": [10.0, 10.3, 10.5, 10.7],
                "low": [9.9, 10.0, 10.1, 10.2],
            }
        ),
    )
    progress = []

    runner.run_full_market_backtest(
        date_from="2026-04-21",
        date_to="2026-04-21",
        output_dir=tmp_path,
        progress_callback=lambda phase, completed, total, message: progress.append((phase, completed, total, message)),
    )
    latest = runner.load_latest_full_market_backtest(tmp_path)

    assert progress
    assert progress[-1][0] == "write_outputs"
    assert latest["summary"]["trade_count"] == 1
