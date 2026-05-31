import pandas as pd
import pytest

import a_share_predictor.data as data


def test_fetch_minute_history_prefers_sina_and_builds_avg_price(monkeypatch):
    class DummyAk:
        @staticmethod
        def stock_zh_a_minute(symbol: str, period: str, adjust: str) -> pd.DataFrame:
            assert symbol == "sh600519"
            assert period == "1"
            assert adjust == ""
            return pd.DataFrame(
                [
                    {
                        "day": "2026-03-11 14:59:00",
                        "open": "9.90",
                        "high": "10.00",
                        "low": "9.80",
                        "close": "10.00",
                        "volume": "100",
                        "amount": "1000",
                    },
                    {
                        "day": "2026-03-12 09:31:00",
                        "open": "10.00",
                        "high": "10.20",
                        "low": "9.90",
                        "close": "10.10",
                        "volume": "100",
                        "amount": "1010",
                    },
                    {
                        "day": "2026-03-12 09:32:00",
                        "open": "10.10",
                        "high": "10.40",
                        "low": "10.00",
                        "close": "10.30",
                        "volume": "200",
                        "amount": "2060",
                    },
                ]
            )

        @staticmethod
        def stock_zh_a_hist_min_em(**kwargs) -> pd.DataFrame:
            raise AssertionError("eastmoney fallback should not be used when sina minute data is available")

    monkeypatch.setattr(data, "ak", DummyAk())
    monkeypatch.setattr(data, "_active_database_source", lambda: None)

    result = data.fetch_minute_history("600519")

    assert result["symbol"].unique().tolist() == ["600519"]
    assert result["datetime"].dt.strftime("%Y-%m-%d").unique().tolist() == ["2026-03-12"]
    assert result["avg_price"].iloc[0] == pytest.approx(10.10)
    assert result["avg_price"].iloc[1] == pytest.approx((1010 + 2060) / 300)


def test_fetch_minute_history_falls_back_to_eastmoney(monkeypatch):
    class DummyAk:
        @staticmethod
        def stock_zh_a_minute(symbol: str, period: str, adjust: str) -> pd.DataFrame:
            raise RuntimeError("sina minute data unavailable")

        @staticmethod
        def stock_zh_a_hist_min_em(**kwargs) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "时间": "2026-03-12 09:31:00",
                        "开盘": "10.00",
                        "收盘": "10.05",
                        "最高": "10.08",
                        "最低": "9.99",
                        "成交量": "100",
                        "成交额": "1005",
                        "均价": "10.05",
                    },
                    {
                        "时间": "2026-03-12 09:32:00",
                        "开盘": "10.05",
                        "收盘": "10.10",
                        "最高": "10.12",
                        "最低": "10.01",
                        "成交量": "120",
                        "成交额": "1212",
                        "均价": "10.08",
                    },
                ]
            )

    monkeypatch.setattr(data, "ak", DummyAk())
    monkeypatch.setattr(data, "_active_database_source", lambda: None)

    result = data.fetch_minute_history("000333")

    assert result["symbol"].unique().tolist() == ["000333"]
    assert result["close"].tolist() == pytest.approx([10.05, 10.10])
    assert result["avg_price"].tolist() == pytest.approx([10.05, 10.08])
