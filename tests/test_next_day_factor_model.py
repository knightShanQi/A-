from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from a_share_predictor.next_day_factor_model import (
    build_industry_context_frame,
    build_market_context_frame,
    build_segment_context_frame,
    run_next_day_factor_pipeline,
)


def _write_synthetic_duckdb(path: Path) -> None:
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2020-01-02", periods=420)
    symbols = ["000001", "000002", "300001", "600001"]
    rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(symbols):
        close = 10.0 + symbol_index
        for day_index, trade_date in enumerate(dates):
            seasonal = 0.006 * np.sin(day_index / 11.0 + symbol_index)
            noise = rng.normal(0.0, 0.018)
            daily_return = seasonal + noise
            open_price = close * (1.0 + rng.normal(0.0, 0.004))
            close = max(1.0, close * (1.0 + daily_return))
            high = max(open_price, close) * (1.0 + abs(rng.normal(0.003, 0.004)))
            low = min(open_price, close) * (1.0 - abs(rng.normal(0.003, 0.004)))
            pre_close = close / (1.0 + daily_return)
            rows.append(
                {
                    "symbol": symbol,
                    "name": symbol,
                    "trade_date": trade_date.date(),
                    "open": float(open_price),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close),
                    "pre_close": float(pre_close),
                    "change": float(close - pre_close),
                    "pct_chg": float(daily_return * 100.0),
                    "volume": float(1_000_000 + 20_000 * day_index + rng.normal(0, 50_000)),
                    "amount": float(close * 1_000_000),
                    "turnover_rate": float(1.0 + 0.2 * np.sin(day_index / 17.0)),
                    "source_file": "synthetic",
                    "ingested_at": pd.Timestamp("2026-01-01"),
                }
            )
    prices = pd.DataFrame(rows)
    calendar = pd.DataFrame({"trade_date": [value.date() for value in dates], "ingested_at": pd.Timestamp("2026-01-01")})
    with duckdb.connect(str(path)) as connection:
        connection.register("prices_df", prices)
        connection.execute("create table a_share_daily_prices as select * from prices_df")
        connection.register("calendar_df", calendar)
        connection.execute("create table a_share_trade_calendar as select * from calendar_df")


def test_run_next_day_factor_pipeline_writes_top10_artifacts(tmp_path):
    duckdb_path = tmp_path / "market.duckdb"
    output_dir = tmp_path / "next_day_model"
    _write_synthetic_duckdb(duckdb_path)

    result = run_next_day_factor_pipeline(
        duckdb_path=duckdb_path,
        output_dir=output_dir,
        analysis_sample_limit=1_000,
        importance_sample_limit=1_000,
        batch_symbols=2,
        min_history_rows=180,
        train_epochs=1,
        random_state=7,
    )

    assert len(result.selected_factors) == 10
    assert Path(result.model_path).exists()
    assert Path(result.feature_ranking_path).exists()
    assert Path(result.cluster_summary_path).exists()
    assert Path(result.selected_factors_path).exists()
    assert Path(result.metrics_path).exists()
    assert result.metrics["test"]["sample_size"] > 0

    selected = pd.read_csv(result.selected_factors_path)
    assert selected["selected_top10"].all()
    assert selected["selected_rank"].tolist() == list(range(1, 11))


def test_cross_section_context_frames_include_rotation_and_flow_proxies(tmp_path):
    duckdb_path = tmp_path / "market.duckdb"
    _write_synthetic_duckdb(duckdb_path)
    stock_basic = pd.DataFrame(
        {
            "symbol": ["000001", "000002", "300001", "600001"],
            "industry": ["bank", "bank", "software", "industry"],
            "market_segment": ["main", "main", "chinext", "main"],
            "is_st": [False, False, False, False],
        }
    )
    query_start = pd.Timestamp("2020-01-02")
    analysis_end = pd.Timestamp("2021-08-12")

    market_context = build_market_context_frame(
        duckdb_path,
        query_start=query_start,
        analysis_end=analysis_end,
    )
    industry_context = build_industry_context_frame(
        duckdb_path,
        stock_basic=stock_basic,
        market_context=market_context,
        query_start=query_start,
        analysis_end=analysis_end,
        min_symbols_per_industry=1,
    )
    segment_context = build_segment_context_frame(
        duckdb_path,
        stock_basic=stock_basic,
        market_context=market_context,
        query_start=query_start,
        analysis_end=analysis_end,
        min_symbols_per_segment=1,
    )

    assert "market_amount_ratio_5" in market_context.columns
    assert {"industry_rank_ret_5", "industry_amount_ratio_5"}.issubset(industry_context.columns)
    assert {"segment_rank_ret_5", "segment_amount_ratio_5"}.issubset(segment_context.columns)
    assert set(segment_context["market_segment"]) == {"main", "chinext"}
