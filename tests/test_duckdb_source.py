from __future__ import annotations

import pandas as pd
import pytest

from a_share_predictor.duckdb_source import fetch_daily_history, fetch_daily_snapshot, fetch_recent_trade_dates
from a_share_predictor.duckdb_store import connect_duckdb, ensure_row_schema


def _seed_row_store(path):
    with connect_duckdb(path) as connection:
        ensure_row_schema(connection)
        connection.execute(
            """
            insert into a_share_daily_prices (
                symbol, name, trade_date, open, high, low, close, pre_close,
                change, pct_chg, volume, amount, turnover_rate, source_file
            )
            values
                ('000001', '', '2026-05-26', 10.0, 10.6, 9.8, 10.5, 10.0, null, null, 1200, 12600, 0.8, 'seed.csv'),
                ('600001', '浦发银行', '2026-05-27', 20.0, 20.4, 19.8, 20.2, 20.0, 0.2, 1.0, 3400, 68680, 1.1, 'seed.csv')
            """
        )
        connection.execute(
            """
            insert into a_share_trade_calendar (trade_date)
            values ('2026-05-26'), ('2026-05-27'), ('2026-05-28')
            """
        )


def test_fetch_daily_snapshot_row_mode_fills_derived_columns(monkeypatch, tmp_path):
    duckdb_path = tmp_path / "market.duckdb"
    _seed_row_store(duckdb_path)
    monkeypatch.setenv("OPENCLAW_DUCKDB_PATH", str(duckdb_path))
    monkeypatch.setenv("OPENCLAW_DAILY_PRICE_STORAGE", "row")

    result = fetch_daily_snapshot("20260526")

    assert list(result["symbol"]) == ["000001"]
    row = result.iloc[0]
    assert row["name"] == "000001"
    assert row["market"] == "SZ"
    assert row["ts_code"] == "000001.SZ"
    assert row["change"] == 0.5
    assert row["pct_chg"] == pytest.approx(5.0)
    assert row["vol"] == 1200
    assert result.attrs["data_source"] == "duckdb"


def test_fetch_daily_history_and_recent_trade_dates_use_row_store(monkeypatch, tmp_path):
    duckdb_path = tmp_path / "market.duckdb"
    _seed_row_store(duckdb_path)
    monkeypatch.setenv("OPENCLAW_DUCKDB_PATH", str(duckdb_path))
    monkeypatch.setenv("OPENCLAW_DAILY_PRICE_STORAGE", "row")

    history = fetch_daily_history("000001", start_date="20260501", end_date="20260527")
    recent_dates = fetch_recent_trade_dates(end_date="20260527", limit=2)

    assert list(history["symbol"]) == ["000001"]
    assert pd.Timestamp(history.iloc[0]["date"]) == pd.Timestamp("2026-05-26")
    assert history.iloc[0]["turnover"] == pytest.approx(0.8)
    assert history.attrs["data_source"] == "duckdb"
    assert recent_dates == ["20260526", "20260527"]
