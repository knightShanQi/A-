import pandas as pd

import a_share_predictor.data as data


RAW_DATE = next(key for key, value in data.DAILY_RENAME.items() if value == "date")
RAW_OPEN = next(key for key, value in data.DAILY_RENAME.items() if value == "open")
RAW_CLOSE = next(key for key, value in data.DAILY_RENAME.items() if value == "close")
RAW_HIGH = next(key for key, value in data.DAILY_RENAME.items() if value == "high")
RAW_LOW = next(key for key, value in data.DAILY_RENAME.items() if value == "low")
RAW_VOLUME = next(key for key, value in data.DAILY_RENAME.items() if value == "volume")
RAW_AMOUNT = next(key for key, value in data.DAILY_RENAME.items() if value == "amount")


def build_hist_row(date: str, open_price: str, close_price: str, high_price: str, low_price: str, volume: str, amount: str) -> dict:
    return {
        RAW_DATE: date,
        RAW_OPEN: open_price,
        RAW_CLOSE: close_price,
        RAW_HIGH: high_price,
        RAW_LOW: low_price,
        RAW_VOLUME: volume,
        RAW_AMOUNT: amount,
    }


def test_fetch_daily_history_reuses_cached_result(monkeypatch, tmp_path):
    class DummyAk:
        calls = 0

        @classmethod
        def stock_zh_a_hist(cls, **kwargs) -> pd.DataFrame:
            cls.calls += 1
            return pd.DataFrame(
                [
                    build_hist_row("2026-03-10", "10", "10.2", "10.3", "9.9", "100", "1020"),
                    build_hist_row("2026-03-11", "10.2", "10.4", "10.5", "10.1", "120", "1248"),
                ]
            )

    monkeypatch.setattr(data, "ak", DummyAk)
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    data.clear_daily_history_cache()

    first = data.fetch_daily_history("600519", start_date="20260301", end_date="20260311")
    second = data.fetch_daily_history("600519", start_date="20260301", end_date="20260311")

    assert DummyAk.calls == 1
    assert first is not second
    assert first["close"].tolist() == second["close"].tolist()

    data.clear_daily_history_cache()
    data.fetch_daily_history("600519", start_date="20260301", end_date="20260311")
    assert DummyAk.calls == 2


def test_fetch_daily_history_disk_cache_survives_memory_clear(monkeypatch, tmp_path):
    class DummyAk:
        calls = 0

        @classmethod
        def stock_zh_a_hist(cls, **kwargs) -> pd.DataFrame:
            cls.calls += 1
            return pd.DataFrame(
                [
                    build_hist_row("2026-03-10", "10", "10.2", "10.3", "9.9", "100", "1020"),
                    build_hist_row("2026-03-11", "10.2", "10.4", "10.5", "10.1", "120", "1248"),
                ]
            )

    monkeypatch.setattr(data, "ak", DummyAk)
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    data.clear_daily_history_cache()

    first = data.fetch_daily_history("600519", start_date="20260301", end_date="20260311")
    data.clear_daily_history_cache(include_disk=False)
    second = data.fetch_daily_history("600519", start_date="20260301", end_date="20260311")

    assert DummyAk.calls == 1
    assert first["close"].tolist() == second["close"].tolist()


def test_fetch_daily_history_incrementally_refreshes_latest_window(monkeypatch, tmp_path):
    class DummyAk:
        calls: list[dict] = []

        @classmethod
        def stock_zh_a_hist(cls, **kwargs) -> pd.DataFrame:
            cls.calls.append(dict(kwargs))
            end_date = kwargs["end_date"]
            if end_date == "20260320":
                dates = pd.date_range("2026-03-10", "2026-03-20", freq="D")
            else:
                dates = pd.date_range("2026-03-10", "2026-03-21", freq="D")
            rows = []
            for index, date in enumerate(dates, start=1):
                close = 10 + index * 0.1
                rows.append(
                    build_hist_row(
                        date.strftime("%Y-%m-%d"),
                        f"{close - 0.1:.2f}",
                        f"{close:.2f}",
                        f"{close + 0.1:.2f}",
                        f"{close - 0.2:.2f}",
                        "100",
                        f"{close * 100:.2f}",
                    )
                )
            return pd.DataFrame(rows)

    monkeypatch.setattr(data, "ak", DummyAk)
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    data.clear_daily_history_cache()

    first = data.fetch_daily_history("600519", start_date="20260301", end_date="20260320")
    data.clear_daily_history_cache(include_disk=False)
    second = data.fetch_daily_history("600519", start_date="20260301", end_date="20260321")

    assert len(DummyAk.calls) == 2
    assert DummyAk.calls[0]["start_date"] == "20260301"
    assert DummyAk.calls[1]["start_date"] == "20260310"
    assert len(first) == 11
    assert len(second) == 12
    assert second["date"].max().strftime("%Y-%m-%d") == "2026-03-21"


def test_fetch_daily_history_refreshes_when_cache_requested_today_but_rows_stop_yesterday(monkeypatch, tmp_path):
    class DummyAk:
        calls: list[dict] = []

        @classmethod
        def stock_zh_a_hist(cls, **kwargs) -> pd.DataFrame:
            cls.calls.append(dict(kwargs))
            if len(cls.calls) == 1:
                dates = pd.date_range("2026-04-21", "2026-04-22", freq="D")
            else:
                dates = pd.date_range("2026-04-21", "2026-04-23", freq="D")
            rows = []
            for index, date in enumerate(dates, start=1):
                close = 10 + index * 0.1
                rows.append(
                    build_hist_row(
                        date.strftime("%Y-%m-%d"),
                        f"{close - 0.1:.2f}",
                        f"{close:.2f}",
                        f"{close + 0.1:.2f}",
                        f"{close - 0.2:.2f}",
                        "100",
                        f"{close * 100:.2f}",
                    )
                )
            return pd.DataFrame(rows)

    monkeypatch.setattr(data, "ak", DummyAk)
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    data.clear_daily_history_cache()

    first = data.fetch_daily_history("600519", start_date="20260421", end_date="20260423")
    assert first["date"].max().strftime("%Y-%m-%d") == "2026-04-22"

    data.clear_daily_history_cache(include_disk=False)
    second = data.fetch_daily_history("600519", start_date="20260421", end_date="20260423")

    assert len(DummyAk.calls) == 2
    assert second["date"].max().strftime("%Y-%m-%d") == "2026-04-23"


def test_fetch_daily_history_falls_back_to_sina_daily(monkeypatch, tmp_path):
    class DummyAk:
        hist_calls = 0
        daily_calls = 0

        @classmethod
        def stock_zh_a_hist(cls, **kwargs):
            cls.hist_calls += 1
            raise RuntimeError("eastmoney blocked")

        @classmethod
        def stock_zh_a_daily(cls, **kwargs):
            cls.daily_calls += 1
            return pd.DataFrame(
                [
                    {"date": "2026-03-10", "open": "10", "high": "10.3", "low": "9.9", "close": "10.2", "volume": "100", "amount": "1020", "turnover": "1.2"},
                    {"date": "2026-03-11", "open": "10.2", "high": "10.5", "low": "10.1", "close": "10.4", "volume": "120", "amount": "1248", "turnover": "1.3"},
                ]
            )

    monkeypatch.setattr(data, "ak", DummyAk)
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    data.clear_daily_history_cache()

    result = data.fetch_daily_history("600519", start_date="20260301", end_date="20260311")

    assert DummyAk.hist_calls == 2
    assert DummyAk.daily_calls == 1
    assert result["close"].tolist() == [10.2, 10.4]
    assert result.attrs["data_source"] == "sina"


def test_fetch_daily_history_keeps_cached_rows_when_prefix_refresh_fails(monkeypatch, tmp_path):
    class DummyAk:
        hist_calls: list[dict] = []
        daily_calls = 0

        @classmethod
        def stock_zh_a_hist(cls, **kwargs):
            cls.hist_calls.append(dict(kwargs))
            if kwargs["start_date"] == "20260408":
                return pd.DataFrame(
                    [
                        build_hist_row("2026-04-08", "10", "10.2", "10.3", "9.9", "100", "1020"),
                        build_hist_row("2026-04-09", "10.2", "10.4", "10.5", "10.1", "120", "1248"),
                    ]
                )
            raise RuntimeError("older prefix unavailable")

        @classmethod
        def stock_zh_a_daily(cls, **kwargs):
            cls.daily_calls += 1
            return pd.DataFrame(
                [
                    {"date": "2026-04-08", "open": "10", "high": "10.3", "low": "9.9", "close": "10.2", "volume": "100", "amount": "1020", "turnover": "1.2"},
                    {"date": "2026-04-09", "open": "10.2", "high": "10.5", "low": "10.1", "close": "10.4", "volume": "120", "amount": "1248", "turnover": "1.3"},
                ]
            )

    monkeypatch.setattr(data, "ak", DummyAk)
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    data.clear_daily_history_cache()

    first = data.fetch_daily_history("001325", start_date="20260408", end_date="20260409")
    data.clear_daily_history_cache(include_disk=False)
    second = data.fetch_daily_history("001325", start_date="20240101", end_date="20260409")

    assert len(first) == 2
    assert len(second) == 2
    assert second["date"].min().strftime("%Y-%m-%d") == "2026-04-08"
    assert second["date"].max().strftime("%Y-%m-%d") == "2026-04-09"
