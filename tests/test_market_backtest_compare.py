import json

import pandas as pd

import a_share_predictor.market_backtest_runner as runner


def test_trade_dates_between_falls_back_to_fast_snapshots(monkeypatch):
    requested_ranges = []

    def fail_trade_calendar(*args, **kwargs):
        raise RuntimeError("trade_cal unavailable")

    def fake_fast_snapshots(trade_dates, **kwargs):
        requested_ranges.append(list(trade_dates))
        return pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2026-04-21", "2026-04-22", "2026-04-24"]),
            }
        )

    monkeypatch.setattr(runner, "_call_tushare_api", fail_trade_calendar)
    monkeypatch.setattr(runner, "_fetch_fast_daily_snapshots_for_dates", fake_fast_snapshots)
    monkeypatch.setattr(
        runner,
        "fetch_tushare_recent_trade_dates",
        lambda end_date=None, limit=180: ["20260425"],
    )

    dates = runner._trade_dates_between("2026-04-21", "2026-04-23")

    assert requested_ranges == [["20260421", "20260422", "20260423"]]
    assert dates == ["2026-04-21", "2026-04-22"]


def test_run_strategy_comparison_backtest_writes_combined_outputs(monkeypatch, tmp_path):
    calls = []

    def fake_run_full_market_backtest(**kwargs):
        calls.append(kwargs)
        mode = kwargs["strategy_mode"]
        return {
            "summary": {
                "strategy_mode": mode,
                "trade_count": 1,
                "win_rate": 1.0 if mode == "old" else 0.0,
            },
            "results": pd.DataFrame(
                [
                    {
                        "market_date": "2026-04-21",
                        "symbol": "600001" if mode == "old" else "600002",
                        "candidate_strategy": mode,
                    }
                ]
            ),
        }

    monkeypatch.setattr(runner, "run_full_market_backtest", fake_run_full_market_backtest)

    payload = runner.run_strategy_comparison_backtest(
        date_from="2026-04-21",
        date_to="2026-04-30",
        output_dir=tmp_path,
        force_rebuild=True,
        fast_strategy_backtest=True,
    )

    assert [call["strategy_mode"] for call in calls] == ["old", "strategy3"]
    assert calls[0]["force_rebuild"] is True
    assert calls[1]["force_rebuild"] is False
    assert payload["summaries"][0]["comparison_label"] == "old_strategy1_2"
    assert payload["summaries"][1]["comparison_label"] == "new_strategy3"
    assert len(payload["results"]) == 2
    assert payload["results"]["strategy_mode"].tolist() == ["old", "strategy3"]
    assert (tmp_path / "strategy_comparison_summary.csv").exists()
    assert (tmp_path / "strategy_comparison_trades.csv").exists()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert [item["comparison_label"] for item in summary["summaries"]] == ["old_strategy1_2", "new_strategy3"]
