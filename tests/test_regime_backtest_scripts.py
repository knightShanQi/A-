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


def test_hard_filter_plan_strategy_family_normalization_and_labels():
    module = _load_script_module(
        "scripts/train_hard_filter_plan_strategy_models.py",
        "train_hard_filter_plan_strategy_models_test_labels",
    )
    assert module.normalize_strategy_family("strategy1_plan_p1") == "strategy1"
    assert module.normalize_strategy_family("strategy2_plan_p1") == "strategy2"
    assert module.normalize_strategy_family("strategy3_plan_p1") == "strategy3"

    candidates = pd.DataFrame(
        [
            {"market_date": "2026-01-02", "symbol": "600001", "candidate_strategy": "strategy1_plan_p1"},
        ]
    )
    history = pd.DataFrame(
        [
            {"market_date": "2026-01-02", "symbol": "600001", "close": 10.0, "high": 10.2, "low": 9.8},
            {"market_date": "2026-01-05", "symbol": "600001", "close": 10.5, "high": 10.7, "low": 10.1},
            {"market_date": "2026-01-06", "symbol": "600001", "close": 10.8, "high": 11.0, "low": 10.4},
            {"market_date": "2026-01-07", "symbol": "600001", "close": 10.6, "high": 10.9, "low": 10.2},
            {"market_date": "2026-01-08", "symbol": "600001", "close": 11.0, "high": 11.2, "low": 10.5},
            {"market_date": "2026-01-09", "symbol": "600001", "close": 11.2, "high": 11.4, "low": 10.8},
        ]
    )

    candidates["strategy_family"] = candidates["candidate_strategy"].map(module.normalize_strategy_family)
    labeled = module.attach_forward_labels(candidates, history)

    assert round(float(labeled.loc[0, "forward_return_1d"]), 6) == 0.05
    assert round(float(labeled.loc[0, "forward_return_3d"]), 6) == 0.06
    assert round(float(labeled.loc[0, "max_high_return_5d"]), 6) == 0.14


def test_hard_filter_plan_strategy_model_retraining_writes_artifacts(tmp_path):
    module = _load_script_module(
        "scripts/train_hard_filter_plan_strategy_models.py",
        "train_hard_filter_plan_strategy_models_test_run",
    )
    rows: list[dict[str, object]] = []
    dates = pd.bdate_range("2025-01-02", periods=90)
    strategies = ["strategy1_plan_p1", "strategy2_plan_p1", "strategy3_plan_p1"]
    for strategy_index, strategy in enumerate(strategies):
        for day_index, market_date in enumerate(dates):
            signal = 1.0 if (day_index + strategy_index) % 4 in {0, 1} else 0.0
            base_return = 0.055 * signal - 0.018 * (1.0 - signal)
            rows.append(
                {
                    "market_date": market_date,
                    "symbol": f"600{strategy_index}{day_index % 10:02d}",
                    "name": f"S{strategy_index}",
                    "candidate_strategy": strategy,
                    "candidate_priority": 58.0 + signal * 25.0 + strategy_index,
                    "strategy_rank": 58.0 + signal * 25.0,
                    "latest_price": 10.0 + strategy_index,
                    "change_pct": -1.0 + signal * 5.0,
                    "amount": 1.5e8 + signal * 1.0e8,
                    "turnover": 3.0 + signal * 2.0,
                    "industry_ret_2d_pct": signal * 2.0,
                    "industry_up_count": 2.0 + signal * 4.0,
                    "model_probability": 0.42 + signal * 0.25,
                    "model_score": 42.0 + signal * 25.0,
                    "priority_score": 58.0 + signal * 25.0,
                    "model_priority_80_20": 45.2 + signal * 25.0,
                    "market_ret": 0.002 * signal,
                    "up_ratio": 0.45 + signal * 0.18,
                    "above_ma20_ratio": 0.40 + signal * 0.15,
                    "limit_up_count": 8 + int(signal * 10),
                    "limit_down_count": 1,
                    "amount_ma5_ma20": 0.9 + signal * 0.4,
                    "amount_ma20_ma60": 0.95 + signal * 0.2,
                    "up_amount_ratio": 0.45 + signal * 0.2,
                    "strong_amount_ratio": 1.0 + signal * 0.4,
                    "trend_score": 45 + signal * 35,
                    "flow_score": 48 + signal * 30,
                    "sse_close": 3100 + day_index,
                    "bull_score": 4 + signal * 3,
                    "plan_market_bucket": "strong_range" if signal else "weak_range",
                    "market_state": "green" if signal else "red",
                    "bull_bear_state": "bull" if signal else "bear",
                    "trend_green": bool(signal),
                    "flow_green": bool(signal),
                    "internal_green": bool(signal),
                    "market_green": bool(signal),
                    "v3_full_green": bool(signal),
                    "v3_yellow": not bool(signal),
                    "is_bull_strict": bool(signal),
                    "is_bull_loose": bool(signal),
                    "forward_return_1d": base_return * 0.45,
                    "forward_return_3d": base_return,
                    "forward_return_5d": base_return * 1.2,
                    "max_high_return_1d": base_return * 0.65 + 0.01,
                    "max_high_return_3d": base_return + 0.025,
                    "max_high_return_5d": base_return * 1.2 + 0.03,
                    "max_drawdown_1d": -0.01 - (1.0 - signal) * 0.02,
                    "max_drawdown_3d": -0.02 - (1.0 - signal) * 0.04,
                    "max_drawdown_5d": -0.025 - (1.0 - signal) * 0.05,
                }
            )
    candidates = pd.DataFrame(rows)
    candidate_path = tmp_path / "plan_scored_candidates.csv"
    candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    report_path = tmp_path / "report.md"

    payload = module.run_strategy_model_retraining(
        candidates_path=candidate_path,
        history_path=None,
        strategy_plan_path=tmp_path / "strategy_hard_filter_optimization_plan_2026-05-31.md",
        output_dir=tmp_path / "models",
        report_path=report_path,
        min_samples=30,
    )

    metrics = pd.read_csv(payload["metrics_path"], encoding="utf-8-sig")
    assert payload["trained_model_count"] == 6
    assert set(metrics["strategy_family"]) == {"strategy1", "strategy2", "strategy3"}
    assert metrics["auc"].dropna().ge(0.5).all()
    assert (tmp_path / "models" / "strategy_model_bundle.pkl").exists()
    assert report_path.exists()
