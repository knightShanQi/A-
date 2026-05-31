from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from a_share_predictor.multisource_factor_cluster import (
    build_auction_feature_frame,
    build_fund_flow_feature_frame,
    build_intraday_feature_frame,
    inspect_multisource_coverage,
    run_one_year_multisource_cluster,
)


def _write_multisource_duckdb(path: Path) -> Path:
    rng = np.random.default_rng(321)
    dates = pd.bdate_range("2025-01-02", periods=330)
    symbols = ["000001", "000002", "000003", "300001", "600001", "600002"]
    rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(symbols):
        close = 8.0 + symbol_index
        for day_index, trade_date in enumerate(dates):
            daily_return = 0.004 * np.sin(day_index / 9.0 + symbol_index) + rng.normal(0.0, 0.015)
            pre_close = close
            open_price = max(1.0, pre_close * (1.0 + rng.normal(0.0, 0.004)))
            close = max(1.0, pre_close * (1.0 + daily_return))
            high = max(open_price, close) * (1.0 + abs(rng.normal(0.003, 0.003)))
            low = min(open_price, close) * (1.0 - abs(rng.normal(0.003, 0.003)))
            volume = float(1_000_000 + symbol_index * 20_000 + day_index * 1_000)
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
                    "volume": volume,
                    "amount": float(close * volume),
                    "turnover_rate": float(1.0 + 0.1 * np.sin(day_index / 13.0)),
                    "source_file": "synthetic",
                    "ingested_at": pd.Timestamp("2026-01-01"),
                }
            )
    prices = pd.DataFrame(rows)
    calendar = pd.DataFrame({"trade_date": [value.date() for value in dates], "ingested_at": pd.Timestamp("2026-01-01")})

    intraday_rows: list[dict[str, object]] = []
    intraday_date = dates[-3].date()
    for symbol_index, symbol in enumerate(symbols):
        base = float(prices.loc[(prices["symbol"] == symbol) & (prices["trade_date"] == intraday_date), "open"].iloc[0])
        for minute in range(40):
            stamp = pd.Timestamp.combine(intraday_date, pd.Timestamp("09:30").time()) + pd.Timedelta(minutes=minute)
            price = base * (1.0 + 0.001 * np.sin(minute / 5.0 + symbol_index))
            volume = float(100 + minute + symbol_index)
            intraday_rows.append(
                {
                    "symbol": symbol,
                    "trade_date": intraday_date,
                    "bar_time": stamp.time(),
                    "interval_minutes": 1,
                    "open": price,
                    "high": price * 1.001,
                    "low": price * 0.999,
                    "close": price,
                    "volume": volume,
                    "amount": price * volume * 100.0,
                    "source_file": "synthetic_intraday",
                    "ingested_at": pd.Timestamp("2026-01-01"),
                }
            )
    intraday = pd.DataFrame(intraday_rows)
    fund_rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(symbols):
        for trade_date in dates[-30:]:
            fund_rows.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date.date(),
                    "close": 10.0 + symbol_index,
                    "change_pct": 0.5,
                    "net_inflow": float((symbol_index + 1) * 1_000_000),
                    "five_day_main_net_amount": float((symbol_index + 1) * 2_000_000),
                    "main_net_amount": float((symbol_index + 1) * 500_000),
                    "main_net_ratio": float(symbol_index + 1),
                    "middle_net_amount": 0.0,
                    "middle_net_ratio": 0.0,
                    "small_net_amount": 0.0,
                    "small_net_ratio": 0.0,
                    "source": "synthetic_fund",
                    "ingested_at": pd.Timestamp("2026-01-01"),
                }
            )
    fund_flow = pd.DataFrame(fund_rows)
    with duckdb.connect(str(path)) as connection:
        connection.register("prices_df", prices)
        connection.execute("create table a_share_daily_prices as select * from prices_df")
        connection.register("calendar_df", calendar)
        connection.execute("create table a_share_trade_calendar as select * from calendar_df")
        connection.register("intraday_df", intraday)
        connection.execute("create table a_share_intraday_bars as select * from intraday_df")
        connection.register("fund_flow_df", fund_flow)
        connection.execute("create table a_share_stock_fund_flow as select * from fund_flow_df")
    return Path(intraday_date.strftime("%Y-%m-%d"))


def _write_stock_basic(path: Path) -> None:
    pd.DataFrame(
        {
            "symbol": ["000001", "000002", "000003", "300001", "600001", "600002"],
            "name": ["A", "B", "C", "D", "E", "F"],
            "industry": ["bank", "bank", "software", "software", "industry", "industry"],
            "market": ["主板", "主板", "主板", "创业板", "主板", "主板"],
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")


def test_build_intraday_feature_frame_extracts_minute_structure(tmp_path):
    duckdb_path = tmp_path / "market.duckdb"
    intraday_date = _write_multisource_duckdb(duckdb_path)

    frame = build_intraday_feature_frame(
        duckdb_path,
        start_date=str(intraday_date),
        end_date=str(intraday_date),
    )

    assert len(frame) == 6
    assert "intraday_vwap_gap" in frame.columns
    assert frame["intraday_bar_count"].min() == 40
    assert frame["auction_open_bar_volume_share"].between(0.0, 1.0).all()


def test_build_fund_flow_feature_frame_extracts_true_flow(tmp_path):
    duckdb_path = tmp_path / "market.duckdb"
    _write_multisource_duckdb(duckdb_path)

    frame = build_fund_flow_feature_frame(
        duckdb_path,
        start_date="2026-01-01",
        end_date="2026-12-31",
    )

    assert len(frame) == 180
    assert "true_fund_main_net_ratio_5" in frame.columns
    assert frame["true_fund_flow_coverage"].eq(1.0).all()


def test_build_auction_feature_frame_extracts_interval_specific_proxy_rows(tmp_path):
    duckdb_path = tmp_path / "market.duckdb"
    intraday_date = _write_multisource_duckdb(duckdb_path)
    with duckdb.connect(str(duckdb_path)) as connection:
        connection.execute(
            """
            create table a_share_call_auction_proxy as
            select 'sz000001' as symbol,
                   cast(? as date) as trade_date,
                   15::smallint as interval_minutes,
                   0.03::real as auction_open_gap,
                   0.02::real as first_bar_ret,
                   0.25::real as first_bar_volume_share,
                   0.30::real as first_bar_amount_share,
                   0.04::real as first_bar_range_pct
            union all
            select '600001',
                   cast(? as date),
                   1::smallint,
                   0.01::real,
                   0.005::real,
                   0.10::real,
                   0.12::real,
                   0.02::real
            """,
            [str(intraday_date), str(intraday_date)],
        )

    frame = build_auction_feature_frame(
        duckdb_path,
        start_date=str(intraday_date),
        end_date=str(intraday_date),
        interval_minutes=15,
    )

    assert frame["symbol"].tolist() == ["000001"]
    assert frame["auction_proxy_available_flag"].tolist() == [1.0]
    assert frame.loc[0, "auction_open_gap"] == 0.03
    assert frame.loc[0, "auction_open_bar_amount_share"] == 0.30


def test_inspect_multisource_coverage_labels_proxy_auction_source(tmp_path):
    duckdb_path = tmp_path / "market.duckdb"
    intraday_date = _write_multisource_duckdb(duckdb_path)
    with duckdb.connect(str(duckdb_path)) as connection:
        connection.execute(
            """
            create table a_share_call_auction_proxy as
            select '000001' as symbol,
                   cast(? as date) as trade_date,
                   15::smallint as interval_minutes,
                   10.0::real as open,
                   9.9::real as pre_close,
                   0.01::real as auction_open_gap,
                   time '09:45:00' as first_bar_time,
                   0.02::real as first_bar_ret,
                   0.20::real as first_bar_volume_share,
                   0.22::real as first_bar_amount_share,
                   0.03::real as first_bar_range_pct,
                   'daily_open_plus_first_intraday_bar' as source
            """,
            [str(intraday_date)],
        )

    coverage = inspect_multisource_coverage(
        duckdb_path,
        start_date=str(intraday_date),
        end_date=str(intraday_date),
        intraday_interval_minutes=15,
    )

    assert coverage["auction"]["true_call_auction_table_available"] is False
    assert coverage["auction"]["proxy_table_available"] is True
    assert coverage["auction"]["mode"] == "derived_daily_open_plus_first_intraday_bar"


def test_run_one_year_multisource_cluster_writes_artifacts(tmp_path):
    duckdb_path = tmp_path / "market.duckdb"
    _write_multisource_duckdb(duckdb_path)
    stock_basic_path = tmp_path / "stock_basic.csv"
    _write_stock_basic(stock_basic_path)
    output_dir = tmp_path / "cluster"

    result = run_one_year_multisource_cluster(
        duckdb_path=duckdb_path,
        output_dir=output_dir,
        stock_basic_path=stock_basic_path,
        symbol_limit=6,
        sample_limit=1_500,
        importance_sample_limit=1_200,
        batch_symbols=3,
        min_history_rows=180,
        min_symbols_per_industry=1,
        min_symbols_per_segment=1,
        random_state=11,
    )

    assert len(result.selected_factors) == 10
    assert Path(result.feature_ranking_path).exists()
    assert Path(result.selected_factors_path).exists()
    assert Path(result.coverage_path).exists()
    assert result.data_summary["candidate_feature_count"] >= 100
    assert result.data_summary["intraday_feature_rows"] == 6
    assert result.data_summary["fund_flow_feature_rows"] == 180


def test_run_one_year_multisource_cluster_disables_low_coverage_true_sources(tmp_path):
    duckdb_path = tmp_path / "market.duckdb"
    _write_multisource_duckdb(duckdb_path)
    stock_basic_path = tmp_path / "stock_basic.csv"
    _write_stock_basic(stock_basic_path)
    progress_messages: list[str] = []

    result = run_one_year_multisource_cluster(
        duckdb_path=duckdb_path,
        output_dir=tmp_path / "cluster_low_coverage",
        stock_basic_path=stock_basic_path,
        symbol_limit=6,
        sample_limit=1_500,
        importance_sample_limit=1_200,
        batch_symbols=3,
        min_history_rows=180,
        min_symbols_per_industry=1,
        min_symbols_per_segment=1,
        min_true_intraday_sample_share=0.5,
        min_true_fund_flow_sample_share=0.5,
        random_state=11,
        progress=progress_messages.append,
    )

    assert result.data_summary["true_intraday_columns_used"] is False
    assert result.data_summary["true_fund_flow_columns_used"] is False
    assert any("true intraday sample share" in message for message in progress_messages)
    assert any("true fund-flow sample share" in message for message in progress_messages)
