import pandas as pd

import a_share_predictor.data as data
import a_share_predictor.database_source as database_source


def test_fetch_daily_history_prefers_database_source(monkeypatch):
    expected = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-26"]),
            "symbol": ["000001"],
            "open": [10.0],
            "high": [10.5],
            "low": [9.9],
            "close": [10.2],
            "volume": [1000.0],
            "amount": [100000.0],
        }
    ).set_index("date", drop=False)

    monkeypatch.setattr(data.duckdb_source, "is_enabled", lambda: False)
    monkeypatch.setattr(data.database_source, "is_enabled", lambda: True)
    monkeypatch.setattr(data.database_source, "fetch_daily_history", lambda symbol, start_date, end_date: expected.copy())
    monkeypatch.setattr(
        data,
        "_fetch_daily_history_cached",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not fetch live provider")),
    )

    result = data.fetch_daily_history("000001", start_date="20260526", end_date="20260526")

    assert result.equals(expected)


def test_fetch_daily_history_accepts_database_when_start_is_market_holiday(monkeypatch):
    expected = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05"]),
            "symbol": ["000001"],
            "close": [10.2],
        }
    ).set_index("date", drop=False)

    monkeypatch.setattr(data.duckdb_source, "is_enabled", lambda: False)
    monkeypatch.setattr(data.database_source, "is_enabled", lambda: True)
    monkeypatch.setattr(data.database_source, "fetch_daily_history", lambda symbol, start_date, end_date: expected.copy())
    monkeypatch.setattr(
        data,
        "_fetch_daily_history_cached",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not fetch live provider")),
    )

    result = data.fetch_daily_history("000001", start_date="20260101", end_date="20260105")

    assert result.equals(expected)


def test_fetch_daily_history_prefers_duckdb_source(monkeypatch):
    expected = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-26"]),
            "symbol": ["000001"],
            "close": [10.2],
        }
    ).set_index("date", drop=False)

    monkeypatch.setattr(data.duckdb_source, "is_enabled", lambda: True)
    monkeypatch.setattr(data.duckdb_source, "fetch_daily_history", lambda symbol, start_date, end_date: expected.copy())
    monkeypatch.setattr(data.database_source, "is_enabled", lambda: True)
    monkeypatch.setattr(
        data.database_source,
        "fetch_daily_history",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should prefer duckdb")),
    )

    result = data.fetch_daily_history("000001", start_date="20260526", end_date="20260526")

    assert result.equals(expected)


def test_fetch_minute_history_prefers_duckdb_source(monkeypatch):
    expected = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-05-28 09:30", "2026-05-28 09:31"]),
            "open": [10.0, 10.1],
            "high": [10.2, 10.3],
            "low": [9.9, 10.0],
            "close": [10.1, 10.2],
            "volume": [100.0, 120.0],
            "amount": [1000.0, 1224.0],
        }
    )

    monkeypatch.setattr(data.duckdb_source, "is_enabled", lambda: True)
    monkeypatch.setattr(data.duckdb_source, "fetch_intraday_history", lambda symbol, interval_minutes=1: expected.copy())
    monkeypatch.setattr(data, "ensure_akshare", lambda: (_ for _ in ()).throw(AssertionError("should prefer duckdb")))

    result = data.fetch_minute_history("000001")

    assert list(result["symbol"]) == ["000001", "000001"]
    assert list(result["close"]) == [10.1, 10.2]
    assert result.attrs["data_source"] == "duckdb"


def test_tushare_snapshot_prefers_database_source(monkeypatch):
    expected = pd.DataFrame([{"symbol": "000001", "trade_date": pd.Timestamp("2026-05-26"), "close": 10.2}])

    data.fetch_tushare_daily_snapshot.cache_clear()
    monkeypatch.setattr(data.duckdb_source, "is_enabled", lambda: False)
    monkeypatch.setattr(data.database_source, "is_enabled", lambda: True)
    monkeypatch.setattr(data.database_source, "fetch_daily_snapshot", lambda trade_date: expected.copy())
    monkeypatch.setattr(
        data,
        "_call_tushare_api",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not call tushare")),
    )

    result = data.fetch_tushare_daily_snapshot("20260526")

    assert result.equals(expected)


def test_trade_dates_prefers_database_source(monkeypatch):
    data.fetch_tushare_recent_trade_dates.cache_clear()
    monkeypatch.setenv("OPENCLAW_DATABASE_ALLOW_PARTIAL", "1")
    monkeypatch.setattr(data.duckdb_source, "is_enabled", lambda: False)
    monkeypatch.setattr(data.database_source, "is_enabled", lambda: True)
    monkeypatch.setattr(data.database_source, "fetch_recent_trade_dates", lambda end_date=None, limit=30: ["20260526"])
    monkeypatch.setattr(
        data,
        "_call_tushare_api",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not call tushare")),
    )

    result = data.fetch_tushare_recent_trade_dates(end_date="20260526", limit=3)

    assert result == ["20260526"]


def test_database_env_loader_strips_utf8_bom(monkeypatch, tmp_path):
    env_file = tmp_path / ".env.local"
    env_file.write_text("\ufeffDATABASE_URL=postgresql://example\n", encoding="utf-8")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    loaded = database_source.load_env_file(env_file)

    assert loaded["DATABASE_URL"] == "postgresql://example"


def test_series_rows_expand_to_daily_history():
    rows = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "name": "",
                "year": 2022,
                "dates": [pd.Timestamp("2022-01-04").date(), pd.Timestamp("2022-01-05").date()],
                "open": None,
                "high": None,
                "low": None,
                "close": [10.5, 10.8],
                "volume": None,
                "amount": None,
                "turnover_rate": [1.2, 1.5],
            }
        ]
    )

    expanded = database_source._series_rows_to_daily_frame(
        rows,
        start=pd.Timestamp("2022-01-05").date(),
        end=pd.Timestamp("2022-01-05").date(),
    )
    result = database_source.normalize_daily_history_frame(expanded)

    assert list(result["symbol"]) == ["000001"]
    assert list(result["close"]) == [10.8]
    assert list(result["turnover"]) == [1.5]
