from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(relative_path: str, module_name: str):
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _sample_selected() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "market_date": "2026-04-21",
                "symbol": "600001",
                "name": "A",
                "candidate_priority": 90.0,
                "model_score": 71.0,
                "rank_score_rebuilt": 82.0,
                "hold_3d_return": 0.10,
            },
            {
                "market_date": "2026-04-21",
                "symbol": "600002",
                "name": "B",
                "candidate_priority": 88.0,
                "model_score": 69.0,
                "rank_score_rebuilt": 81.0,
                "hold_3d_return": -0.05,
            },
        ]
    )


def _sample_history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"trade_date": "2026-04-21", "symbol": "600001", "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0},
            {"trade_date": "2026-04-22", "symbol": "600001", "open": 10.2, "high": 10.5, "low": 10.1, "close": 10.4},
            {"trade_date": "2026-04-23", "symbol": "600001", "open": 10.5, "high": 10.9, "low": 10.4, "close": 10.8},
            {"trade_date": "2026-04-24", "symbol": "600001", "open": 10.8, "high": 11.3, "low": 10.7, "close": 11.2},
            {"trade_date": "2026-04-21", "symbol": "600002", "open": 20.0, "high": 20.1, "low": 19.9, "close": 20.0},
            {"trade_date": "2026-04-22", "symbol": "600002", "open": 20.1, "high": 20.2, "low": 19.8, "close": 19.9},
            {"trade_date": "2026-04-23", "symbol": "600002", "open": 19.8, "high": 19.9, "low": 19.4, "close": 19.5},
            {"trade_date": "2026-04-24", "symbol": "600002", "open": 19.6, "high": 19.7, "low": 19.1, "close": 19.2},
        ]
    )


def _sample_calendar() -> pd.DataFrame:
    return pd.DataFrame({"market_date": pd.to_datetime(["2026-04-21", "2026-04-22", "2026-04-23", "2026-04-24"])})


def test_market_regime_v3_summary_exposes_portfolio_metrics():
    module = _load_script_module("scripts/backtest_market_regime_v3.py", "backtest_market_regime_v3_test")
    summary, nav, trades = module._summarize_rule(
        _sample_selected(),
        _sample_calendar(),
        rule="unit_rule",
        history=_sample_history(),
        top_n=2,
    )

    assert "portfolio_annualized_return" in summary
    assert "portfolio_max_drawdown" in summary
    assert summary["portfolio_trade_count"] == 2
    assert len(nav) == 4
    assert len(trades) == 2


def test_bull_rank_score_summary_exposes_portfolio_metrics():
    module = _load_script_module("scripts/backtest_bull_market_rank_score.py", "backtest_bull_market_rank_score_test")
    summary, curve, nav, trades = module._summarize(
        _sample_selected(),
        _sample_calendar(),
        "unit_rule",
        history=_sample_history(),
        top_n=2,
    )

    assert "portfolio_annualized_return" in summary
    assert "portfolio_max_drawdown" in summary
    assert summary["portfolio_trade_count"] == 2
    assert len(curve) == 4
    assert len(nav) == 4
    assert len(trades) == 2
