import json

import pandas as pd

import a_share_predictor.market_backtest_runner as runner
from a_share_predictor.duckdb_store import connect_duckdb


def test_run_full_market_backtest_writes_summary_and_results(monkeypatch, tmp_path):
    universe = pd.DataFrame(
        [
            {"symbol": "600001", "name": "A"},
            {"symbol": "600002", "name": "B"},
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "symbol": "600001",
                "name": "A",
                "candidate_strategy": "策略1",
                "candidate_priority": 90.0,
                "latest_price": 10.0,
                "change_pct": 3.0,
                "amount": 3.0e8,
            },
            {
                "symbol": "600002",
                "name": "B",
                "candidate_strategy": "策略2",
                "candidate_priority": 88.0,
                "latest_price": 20.0,
                "change_pct": 6.0,
                "amount": 4.0e8,
            },
        ]
    )
    histories = {
        "600001": pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-04-21", "2026-04-22", "2026-04-23", "2026-04-24"]),
                "open": [10.0, 10.2, 10.6, 10.8],
                "close": [10.0, 10.5, 11.0, 11.2],
                "high": [10.1, 10.7, 11.3, 11.4],
                "low": [9.9, 10.1, 10.5, 10.7],
            }
        ),
        "600002": pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-04-21", "2026-04-22", "2026-04-23", "2026-04-24"]),
                "open": [20.0, 20.2, 19.8, 19.6],
                "close": [20.0, 19.8, 19.4, 19.2],
                "high": [20.1, 20.3, 19.9, 19.7],
                "low": [19.8, 19.6, 19.2, 19.0],
            }
        ),
    }

    monkeypatch.setattr(runner, "fetch_a_share_universe", lambda: universe.copy())
    monkeypatch.setattr(runner, "fetch_tushare_recent_trade_dates", lambda end_date=None, limit=420: ["20260421"])
    monkeypatch.setattr(runner, "build_market_daily_feature_store", lambda *args, **kwargs: pd.DataFrame({"symbol": ["600001", "600002"]}))
    monkeypatch.setattr(runner, "build_market_candidate_pool_store", lambda *args, **kwargs: candidates.copy())
    monkeypatch.setattr(runner, "fetch_daily_history", lambda symbol, start_date=None: histories[str(symbol)].copy())

    payload = runner.run_full_market_backtest(
        date_from="2026-04-21",
        date_to="2026-04-21",
        horizon_days=2,
        positive_return=0.05,
        strategy_mode="all",
        top_k=2,
        output_dir=tmp_path,
        force_rebuild=True,
        persist_research=True,
        duckdb_database=tmp_path / "research.duckdb",
    )

    assert payload["summary"]["trade_count"] == 2
    assert payload["summary"]["win_rate"] == 0.5
    assert payload["summary"]["target_hit_rate"] == 0.5
    assert round(payload["summary"]["avg_hold_1d_return"], 6) == 0.004805
    assert round(payload["summary"]["avg_hold_3d_return"], 6) == 0.024267
    assert payload["summary"]["hold_5d_sample_count"] == 0
    assert "annualized_return" in payload["summary"]
    assert "ending_equity" in payload["summary"]
    assert "max_drawdown" in payload["summary"]
    assert payload["summary"]["evaluation_engine"] == "unified_portfolio_nav_v1"
    assert payload["summary"]["evaluation_primary_metric"] == "annualized_return"
    assert payload["summary"]["evaluation_primary_source"] == "portfolio_daily_nav"
    assert "avg_forward_return" in payload["summary"]["diagnostic_metric_keys"]
    assert payload["summary"]["research_run_id"]
    assert payload["summary"]["portfolio_trade_count"] == 2
    assert len(payload["results"]) == 2
    assert "hold_1d_return" in payload["results"].columns
    assert "hold_3d_return" in payload["results"].columns
    assert "hold_5d_return" in payload["results"].columns
    assert (tmp_path / "trade_like_results.csv").exists()
    assert (tmp_path / "portfolio_daily_nav.csv").exists()
    assert (tmp_path / "portfolio_trades.csv").exists()
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["trade_count"] == 2
    latest = runner.load_latest_full_market_backtest(tmp_path)
    assert latest["summary"]["trade_count"] == 2
    assert len(latest["results"]) == 2
    assert not latest["portfolio_daily_nav"].empty
    assert len(latest["portfolio_trades"]) == 2
    with connect_duckdb(tmp_path / "research.duckdb", read_only=True) as connection:
        stored = connection.execute("select evaluation_engine, portfolio_trade_count from research_market_backtest_runs").fetchone()
    assert stored == ("unified_portfolio_nav_v1", 2)


def test_run_full_market_backtest_can_filter_strategy(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "fetch_a_share_universe", lambda: pd.DataFrame([{"symbol": "600001", "name": "A"}]))
    monkeypatch.setattr(runner, "fetch_tushare_recent_trade_dates", lambda end_date=None, limit=420: ["20260421"])
    monkeypatch.setattr(runner, "build_market_daily_feature_store", lambda *args, **kwargs: pd.DataFrame({"symbol": ["600001"]}))
    monkeypatch.setattr(
        runner,
        "build_market_candidate_pool_store",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {"symbol": "600001", "name": "A", "candidate_strategy": "策略1", "candidate_priority": 90.0},
                {"symbol": "600002", "name": "B", "candidate_strategy": "策略2", "candidate_priority": 91.0},
            ]
        ),
    )
    monkeypatch.setattr(
        runner,
        "fetch_daily_history",
        lambda symbol, start_date=None: pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-04-21", "2026-04-22", "2026-04-23"]),
                "open": [10.0, 10.0, 10.0],
                "close": [10.0, 10.5, 10.8],
                "high": [10.1, 10.6, 10.9],
                "low": [9.9, 10.0, 10.3],
            }
        ),
    )

    payload = runner.run_full_market_backtest(
        date_from="2026-04-21",
        date_to="2026-04-21",
        horizon_days=2,
        strategy_mode="strategy1",
        output_dir=tmp_path,
    )

    assert payload["results"]["candidate_strategy"].tolist() == ["策略1"]
    assert payload["summary"]["portfolio_trade_count"] == 1
