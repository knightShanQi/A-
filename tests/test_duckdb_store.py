from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from a_share_predictor.duckdb_store import (
    connect_duckdb,
    normalize_stock_fund_flow_frame,
    rebuild_call_auction_proxy_table,
    sync_intraday_bars_from_local_tree,
    sync_row_database_from_daily_files,
    upsert_stock_fund_flow_frame,
)


def test_sync_row_database_from_daily_files_upserts_local_csv(tmp_path):
    input_dir = tmp_path / "daily"
    input_dir.mkdir()
    price_file = input_dir / "prices.csv"
    price_file.write_text(
        "symbol,trade_date,close,turnover_rate\n"
        "000001,2026-05-26,10.1,0.8\n"
        "600001,2026-05-26,20.2,1.2\n",
        encoding="utf-8",
    )
    duckdb_path = tmp_path / "market.duckdb"

    first = sync_row_database_from_daily_files(
        duckdb_database=duckdb_path,
        input_dir=input_dir,
        skip_download=True,
    )

    assert first["rows_written"] == 2
    assert first["calendar_rows"] == 1
    with connect_duckdb(duckdb_path, read_only=True) as connection:
        assert connection.execute("select count(*) from a_share_daily_prices").fetchone()[0] == 2

    price_file.write_text(
        "symbol,trade_date,close,turnover_rate\n"
        "000001,2026-05-26,10.3,0.9\n",
        encoding="utf-8",
    )

    second = sync_row_database_from_daily_files(
        duckdb_database=duckdb_path,
        input_dir=input_dir,
        skip_download=True,
    )

    assert second["rows_written"] == 1
    with connect_duckdb(duckdb_path, read_only=True) as connection:
        rows = connection.execute(
            "select symbol, close, turnover_rate from a_share_daily_prices order by symbol"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "000001"
    assert rows[0][1] == pytest.approx(10.3)
    assert rows[0][2] == pytest.approx(0.9)


def test_sync_intraday_bars_from_local_tree_imports_interval_dirs(tmp_path):
    root = tmp_path / "20260526"
    interval_dir = root / "1min"
    interval_dir.mkdir(parents=True)
    (interval_dir / "sz000001.csv").write_text(
        "日期,时间,开盘,最高,最低,收盘,成交量,成交额\n"
        "2026-05-26,09:30,10.0,10.2,9.9,10.1,100,1000\n"
        "2026-05-26,09:31,10.1,10.3,10.0,10.2,120,1224\n",
        encoding="utf-8",
    )
    duckdb_path = tmp_path / "market.duckdb"

    result = sync_intraday_bars_from_local_tree(
        duckdb_database=duckdb_path,
        source_dir=root,
        intervals=[1],
    )

    assert result["rows_written"] == 2
    assert result["intervals"][0]["interval_minutes"] == 1
    with connect_duckdb(duckdb_path, read_only=True) as connection:
        rows = connection.execute(
            """
            select symbol, trade_date, left(cast(bar_time as varchar), 5), interval_minutes, close
            from a_share_intraday_bars
            order by bar_time
            """
        ).fetchall()
    assert rows[0][0] == "000001"
    assert str(rows[0][1]) == "2026-05-26"
    assert rows[0][2] == "09:30"
    assert rows[0][3] == 1
    assert rows[0][4] == pytest.approx(10.1)
    assert rows[1][4] == pytest.approx(10.2)


def test_sync_intraday_bars_file_batches_do_not_overwrite_other_symbols(tmp_path, monkeypatch):
    root = tmp_path / "20260526"
    interval_dir = root / "1min"
    interval_dir.mkdir(parents=True)
    for symbol, close in [("sz000001", 10.1), ("sz000002", 20.2), ("sh600001", 30.3)]:
        interval_dir.joinpath(f"{symbol}.csv").write_text(
            "日期,时间,开盘,最高,最低,收盘,成交量,成交额\n"
            f"2026-05-26,09:30,{close - 0.1},{close + 0.1},{close - 0.2},{close},100,1000\n",
            encoding="utf-8",
        )
    duckdb_path = tmp_path / "market.duckdb"
    monkeypatch.setenv("OPENCLAW_INTRADAY_IMPORT_BATCH_FILES", "1")

    result = sync_intraday_bars_from_local_tree(
        duckdb_database=duckdb_path,
        source_dir=root,
        intervals=[1],
    )

    assert result["rows_written"] == 3
    with connect_duckdb(duckdb_path, read_only=True) as connection:
        rows = connection.execute(
            """
            select symbol, close
            from a_share_intraday_bars
            order by symbol
            """
        ).fetchall()
    assert rows == [("000001", pytest.approx(10.1)), ("000002", pytest.approx(20.2)), ("600001", pytest.approx(30.3))]


def test_sync_intraday_bars_applies_retention_window(tmp_path):
    root = tmp_path / "bars"
    old_dir = root / "20240520" / "1min"
    recent_dir = root / "20260528" / "1min"
    old_dir.mkdir(parents=True)
    recent_dir.mkdir(parents=True)
    old_dir.joinpath("sz000001.csv").write_text(
        "日期,时间,开盘,最高,最低,收盘,成交量,成交额\n"
        "2024-05-20,09:30,10.0,10.1,9.9,10.0,100,1000\n",
        encoding="utf-8",
    )
    recent_dir.joinpath("sz000001.csv").write_text(
        "日期,时间,开盘,最高,最低,收盘,成交量,成交额\n"
        "2026-05-28,09:30,10.0,10.2,9.9,10.1,100,1000\n",
        encoding="utf-8",
    )
    duckdb_path = tmp_path / "market.duckdb"

    result = sync_intraday_bars_from_local_tree(
        duckdb_database=duckdb_path,
        source_dir=root,
        intervals=[1],
        retention_days=365,
        retention_reference_date=dt.date(2026, 5, 28),
    )

    assert result["retention_cutoff"] == "2025-05-28"
    assert result["rows_purged"] == 1
    with connect_duckdb(duckdb_path, read_only=True) as connection:
        rows = connection.execute("select distinct trade_date from a_share_intraday_bars").fetchall()
    assert [str(row[0]) for row in rows] == ["2026-05-28"]


def test_sync_intraday_bars_filters_import_date_range(tmp_path):
    root = tmp_path / "legacy" / "15min"
    root.mkdir(parents=True)
    root.joinpath("sz000001.csv").write_text(
        "日期,时间,开盘,最高,最低,收盘,成交量,成交额\n"
        "2025-05-27,09:30,10.0,10.1,9.9,10.0,100,1000\n"
        "2025-05-28,09:45,10.0,10.2,9.9,10.1,100,1000\n"
        "2026-01-02,10:00,10.0,10.3,9.9,10.2,100,1000\n",
        encoding="utf-8",
    )
    duckdb_path = tmp_path / "market.duckdb"

    result = sync_intraday_bars_from_local_tree(
        duckdb_database=duckdb_path,
        source_dir=tmp_path / "legacy",
        intervals=[15],
        start_date="2025-05-28",
        end_date="2025-12-31",
        retention_days=None,
    )

    assert result["rows_written"] == 1
    with connect_duckdb(duckdb_path, read_only=True) as connection:
        rows = connection.execute(
            "select trade_date, left(cast(bar_time as varchar), 5), interval_minutes from a_share_intraday_bars"
        ).fetchall()
    assert [(str(row[0]), row[1], row[2]) for row in rows] == [("2025-05-28", "09:45", 15)]


def test_upsert_stock_fund_flow_frame_normalizes_true_flow(tmp_path):
    raw = pd.DataFrame(
        [
            {
                "日期": "20260528",
                "收盘价": 10.5,
                "涨跌幅": "1.2",
                "资金净流入": 2.1e8,
                "5日主力净额": 3.2e8,
                "主力净流入-净额": 1.2e8,
                "主力净流入-净占比": 4.5,
                "中单净流入-净额": -1.0e7,
                "中单净流入-净占比": -0.4,
                "小单净流入-净额": -2.0e7,
                "小单净流入-净占比": -0.7,
            }
        ]
    )
    frame = normalize_stock_fund_flow_frame("000001", raw)
    duckdb_path = tmp_path / "market.duckdb"

    with connect_duckdb(duckdb_path) as connection:
        written = upsert_stock_fund_flow_frame(connection, frame)
        rows = connection.execute(
            """
            select symbol, trade_date, main_net_amount, main_net_ratio, source
            from a_share_stock_fund_flow
            """
        ).fetchall()

    assert written == 1
    assert rows == [("000001", dt.date(2026, 5, 28), 1.2e8, pytest.approx(4.5), "stockpage_10jqka")]


def test_rebuild_call_auction_proxy_table_uses_daily_open_and_first_bar(tmp_path):
    duckdb_path = tmp_path / "market.duckdb"
    with connect_duckdb(duckdb_path) as connection:
        connection.execute(
            """
            create table a_share_daily_prices as
            select '000001' as symbol, date '2026-05-27' as trade_date,
                   10.0::real as open, null::real as pre_close, 10.0::real as close
            union all
            select '000001' as symbol, date '2026-05-28' as trade_date,
                   10.5::real as open, null::real as pre_close, 10.7::real as close
            """
        )
        connection.execute(
            """
            create table a_share_intraday_bars as
            select '000001' as symbol, date '2026-05-28' as trade_date,
                   time '09:45:00' as bar_time, 15::smallint as interval_minutes,
                   10.5::real as open, 10.8::real as high, 10.4::real as low,
                   10.7::real as close, 100.0::double as volume, 107000.0::double as amount
            union all
            select '000001', date '2026-05-28', time '10:00:00', 15::smallint,
                   10.7::real, 10.9::real, 10.6::real, 10.8::real, 300.0::double, 324000.0::double
            """
        )

        rows_written = rebuild_call_auction_proxy_table(
            connection,
            start_date="2026-05-28",
            end_date="2026-05-28",
            interval_minutes=15,
        )
        row = connection.execute(
            """
            select auction_open_gap, first_bar_time, first_bar_ret, first_bar_volume_share
            from a_share_call_auction_proxy
            """
        ).fetchone()

    assert rows_written == 1
    assert row[0] == pytest.approx(0.05)
    assert str(row[1]) == "09:45:00"
    assert row[2] == pytest.approx(10.7 / 10.5 - 1.0)
    assert row[3] == pytest.approx(0.25)
