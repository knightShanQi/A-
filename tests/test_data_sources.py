import pandas as pd

import a_share_predictor.data as data


def test_filter_point_in_time_universe_excludes_future_and_delisted_names():
    stock_basic = pd.DataFrame(
        [
            {"symbol": "600001", "name": "Active", "list_date": "20200101", "delist_date": "", "list_status": "L"},
            {"symbol": "600002", "name": "Future", "list_date": "20260501", "delist_date": "", "list_status": "L"},
            {"symbol": "600003", "name": "Gone", "list_date": "20190101", "delist_date": "20260401", "list_status": "D"},
            {"symbol": "600004", "name": "WasLive", "list_date": "20190101", "delist_date": "20260430", "list_status": "D"},
            {"symbol": "600005", "name": "Pending", "list_date": "20260401", "delist_date": "", "list_status": "P"},
        ]
    )

    result = data.filter_point_in_time_a_share_universe(stock_basic, "2026-04-21")

    assert set(result["symbol"]) == {"600001", "600004"}


def test_filter_historical_universe_window_keeps_symbols_alive_during_window():
    stock_basic = pd.DataFrame(
        [
            {"symbol": "600001", "name": "Active", "list_date": "20200101", "delist_date": "", "list_status": "L"},
            {"symbol": "600002", "name": "Future", "list_date": "20260501", "delist_date": "", "list_status": "L"},
            {"symbol": "600003", "name": "GoneBefore", "list_date": "20190101", "delist_date": "20240101", "list_status": "D"},
            {"symbol": "600004", "name": "DelistedInside", "list_date": "20190101", "delist_date": "20260301", "list_status": "D"},
            {"symbol": "600005", "name": "Pending", "list_date": "20250101", "delist_date": "", "list_status": "P"},
        ]
    )

    result = data.filter_historical_a_share_universe_window(stock_basic, "2025-01-01", "2026-03-31")

    assert set(result["symbol"]) == {"600001", "600004"}


def test_normalize_ths_board_flow_table_handles_current_page_shape():
    raw = pd.DataFrame(
        [
            {
                "序号": 1,
                "行业": "贵金属",
                "行业指数": 6418.02,
                "涨跌幅": "9.37%",
                "流入资金(亿)": 139.32,
                "流出资金(亿)": 107.44,
                "净额(亿)": 31.88,
                "公司家数": 12,
                "领涨股": "晓程科技",
                "涨跌幅.1": "16.83%",
                "当前价(元)": 61.3,
            }
        ]
    )

    result = data._normalize_ths_board_flow_table(raw)

    assert result.loc[0, "sector_name"] == "贵金属"
    assert result.loc[0, "net_inflow"] == 31.88
    assert result.loc[0, "change_pct"] == 9.37
    assert result.loc[0, "leader"] == "晓程科技"
    assert result.loc[0, "sector_name_normalized"] == "贵金属"


def test_parse_stockpage_fund_flow_table_normalizes_to_yuan():
    raw = pd.DataFrame(
        [[20260408, 1465.02, "1.74%", 55948.05, 47423.31, 42496.07, "8.59%", 13455.66, "2.72%", -3.68, "-0.00%"]],
        columns=pd.MultiIndex.from_tuples(
            [
                ("日期", "日期"),
                ("收盘价", "收盘价"),
                ("涨跌幅", "涨跌幅"),
                ("资金净流入", "资金净流入"),
                ("5日主力净额", "5日主力净额"),
                ("大单(主力)", "净额"),
                ("大单(主力)", "净占比"),
                ("中单", "净额"),
                ("中单", "净占比"),
                ("小单", "净额"),
                ("小单", "净占比"),
            ]
        ),
    )

    result = data._parse_stockpage_fund_flow_table(raw)

    assert result.loc[0, "主力净流入-净额"] == 424960700.0
    assert result.loc[0, "资金净流入"] == 559480500.0
    assert result.loc[0, "主力净流入-净占比"] == 8.59


def test_fetch_industry_fund_flow_falls_back_to_ths(monkeypatch):
    class DummyAk:
        @staticmethod
        def stock_fund_flow_industry(symbol: str):
            raise RuntimeError("wrapper broken")

    raw = pd.DataFrame(
        [
            {
                "序号": 1,
                "行业": "白酒",
                "行业指数": 1234.0,
                "涨跌幅": "3.20%",
                "流入资金(亿)": 8.2,
                "流出资金(亿)": 3.1,
                "净额(亿)": 5.1,
                "公司家数": 20,
                "领涨股": "贵州茅台",
                "涨跌幅.1": "2.50%",
                "当前价(元)": 1500.0,
            }
        ]
    )

    monkeypatch.setattr(data, "ak", DummyAk)
    monkeypatch.setattr(data, "_fetch_ths_board_flow_table", lambda kind, period: raw)

    result = data.fetch_industry_fund_flow("即时")

    assert result.loc[0, "sector_name"] == "白酒"
    assert result.loc[0, "net_inflow"] == 5.1


def test_fetch_stock_main_fund_flow_falls_back_to_stockpage(monkeypatch):
    class DummyAk:
        @staticmethod
        def stock_individual_fund_flow(**kwargs):
            raise RuntimeError("eastmoney blocked")

    fallback_df = pd.DataFrame(
        [
            {
                "日期": pd.Timestamp("2026-04-08"),
                "收盘价": 10.2,
                "涨跌幅": 1.2,
                "主力净流入-净额": 1.5e8,
                "主力净流入-净占比": 6.8,
            }
        ]
    )

    monkeypatch.setattr(data, "ak", DummyAk)
    monkeypatch.setattr(data, "_fetch_stockpage_fund_flow_table", lambda symbol: fallback_df)

    result = data.fetch_stock_main_fund_flow("600519", limit=5)

    assert result.loc[0, "主力净流入-净额"] == 1.5e8
    assert result.loc[0, "主力净流入-净占比"] == 6.8


def test_fetch_stock_profile_falls_back_to_ths_profile(monkeypatch):
    class DummyAk:
        @staticmethod
        def stock_individual_info_em(symbol: str):
            raise RuntimeError("eastmoney blocked")

    monkeypatch.setattr(data, "ak", DummyAk)
    monkeypatch.setattr(
        data,
        "_fetch_ths_basic_profile",
        lambda symbol: {
            "股票代码": symbol,
            "股票简称": "贵州茅台",
            "行业": "白酒Ⅱ",
            "主营业务": "白酒销售",
        },
    )

    result = data.fetch_stock_profile("600519")

    assert result["股票简称"] == "贵州茅台"
    assert result["行业"] == "白酒Ⅱ"
def test_fetch_tushare_daily_snapshot_merges_basic_fields(monkeypatch):
    def fake_call(api_name, params=None, fields=""):
        if api_name == "daily":
            return pd.DataFrame(
                [
                    {
                        "ts_code": "600001.SH",
                        "trade_date": "20260418",
                        "open": 10.0,
                        "high": 10.5,
                        "low": 9.9,
                        "close": 10.3,
                        "pre_close": 9.8,
                        "change": 0.5,
                        "pct_chg": 5.1,
                        "vol": 120000,
                        "amount": 250000,
                    }
                ]
            )
        if api_name == "daily_basic":
            return pd.DataFrame(
                [
                    {
                        "ts_code": "600001.SH",
                        "trade_date": "20260418",
                        "turnover_rate": 6.8,
                        "volume_ratio": 1.5,
                        "total_mv": 1230000,
                        "circ_mv": 950000,
                    }
                ]
            )
        raise AssertionError(api_name)

    monkeypatch.setattr(data, "_call_tushare_api", fake_call)
    monkeypatch.setattr(
        data,
        "fetch_tushare_stock_basic",
        lambda: pd.DataFrame(
            [{"ts_code": "600001.SH", "symbol": "600001", "name": "测试股", "industry": "消费", "market": "主板"}]
        ),
    )

    result = data.fetch_tushare_daily_snapshot("20260418")

    assert result.loc[0, "symbol"] == "600001"
    assert result.loc[0, "name"] == "测试股"
    assert result.loc[0, "industry"] == "消费"
    assert result.loc[0, "amount"] == 250000000
    assert result.loc[0, "turnover_rate"] == 6.8


def test_fetch_a_share_universe_uses_disk_cache_when_live_fetch_fails(monkeypatch, tmp_path):
    cached = pd.DataFrame(
        [
            {"symbol": "600001", "name": "CacheA", "name_normalized": "CACHEA"},
            {"symbol": "000001", "name": "CacheB", "name_normalized": "CACHEB"},
        ]
    )

    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    data._write_a_share_universe_disk_cache(cached, source="test-cache")

    class DummyAk:
        @staticmethod
        def stock_info_a_code_name():
            raise RuntimeError("universe fetch failed")

    monkeypatch.setattr(data, "ak", DummyAk)
    monkeypatch.setattr(
        data,
        "fetch_tushare_stock_basic",
        lambda: (_ for _ in ()).throw(AssertionError("should not hit tushare when disk cache is available")),
    )

    result = data.fetch_a_share_universe()

    assert result["symbol"].tolist() == ["000001", "600001"]
    assert result["name"].tolist() == ["CacheB", "CacheA"]


def test_build_ths_headers_skips_mini_racer_off_main_thread(monkeypatch):
    worker_thread = object()
    main_thread = object()

    monkeypatch.setattr(data, "MiniRacer", object())
    monkeypatch.setattr(data.threading, "current_thread", lambda: worker_thread)
    monkeypatch.setattr(data.threading, "main_thread", lambda: main_thread)

    headers = data._build_ths_headers("http://example.com")

    assert headers["Referer"] == "http://example.com"
    assert "Cookie" not in headers
    assert "hexin-v" not in headers
