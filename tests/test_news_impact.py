import numpy as np
import pandas as pd

from a_share_predictor import news_impact


def test_classify_news_events_detects_event_categories():
    raw = pd.DataFrame(
        {
            "symbol": ["000001", "000001", "000001"],
            "title": [
                "\u516c\u53f8\u56de\u8d2d\u8ba1\u5212\u843d\u5730",
                "\u516c\u53f8\u88ab\u7acb\u6848\u8c03\u67e5\u5e76\u53ef\u80fd\u5904\u7f5a",
                "new order contract signed",
            ],
            "content": [
                "\u8463\u4e8b\u4f1a\u901a\u8fc7\u80a1\u4efd\u56de\u8d2d\u65b9\u6848",
                "\u516c\u53f8\u63d0\u793a\u76d1\u7ba1\u98ce\u9669",
                "large contract improves backlog",
            ],
            "published_at": pd.to_datetime(
                ["2026-01-02 09:10", "2026-01-02 11:00", "2026-01-02 13:30"]
            ),
            "source": ["cninfo", "cninfo", "newswire"],
        }
    )

    events = news_impact.classify_news_events(raw)
    by_title = events.set_index("title")

    assert by_title.loc["\u516c\u53f8\u56de\u8d2d\u8ba1\u5212\u843d\u5730", "event_category"] == "shareholder_action"
    assert by_title.loc["\u516c\u53f8\u56de\u8d2d\u8ba1\u5212\u843d\u5730", "event_direction"] == "bullish"
    assert by_title.loc["\u516c\u53f8\u88ab\u7acb\u6848\u8c03\u67e5\u5e76\u53ef\u80fd\u5904\u7f5a", "event_category"] == "regulatory_risk"
    assert by_title.loc["\u516c\u53f8\u88ab\u7acb\u6848\u8c03\u67e5\u5e76\u53ef\u80fd\u5904\u7f5a", "event_direction"] == "bearish"
    assert by_title.loc["new order contract signed", "event_category"] == "contract_order"
    assert events["expected_impact_score"].between(0, 100).all()


def test_normalize_news_frame_reads_cninfo_disclosure_time():
    raw = pd.DataFrame(
        {
            "代码": ["000001"],
            "公告标题": ["2026年一季度报告"],
            "公告时间": ["2026-04-25"],
            "公告链接": ["https://example.test/report"],
        }
    )

    normalized = news_impact.normalize_news_frame(raw)

    assert normalized.loc[0, "symbol"] == "000001"
    assert normalized.loc[0, "title"] == "2026年一季度报告"
    assert normalized.loc[0, "published_at"] == pd.Timestamp("2026-04-25")


def test_classify_routine_governance_disclosure_as_neutral_category():
    raw = pd.DataFrame(
        {
            "symbol": ["000001"],
            "title": ["\u5173\u4e8e\u53ec\u5f002026\u5e74\u7b2c\u4e00\u6b21\u4e34\u65f6\u80a1\u4e1c\u5927\u4f1a\u7684\u6cd5\u5f8b\u610f\u89c1\u4e66"],
            "content": ["\u672c\u6b21\u80a1\u4e1c\u5927\u4f1a\u7684\u53ec\u96c6\u548c\u8868\u51b3\u7a0b\u5e8f\u7b26\u5408\u516c\u53f8\u7ae0\u7a0b"],
            "published_at": pd.to_datetime(["2026-01-02 18:00"]),
            "source": ["cninfo"],
        }
    )

    events = news_impact.classify_news_events(raw)

    assert events.loc[0, "event_category"] == "routine_governance"
    assert events.loc[0, "event_direction"] == "neutral"
    assert events.loc[0, "neutral_hits"] >= 1


def test_classify_fund_flow_market_heat_not_regulatory_risk():
    raw = pd.DataFrame(
        {
            "symbol": ["000001"],
            "title": ["\u9f99\u864e\u699c\uff1a\u4e3b\u529b\u8d44\u91d1\u51c0\u6d41\u51fa\u4e14\u878d\u8d44\u507f\u8fd8\u589e\u52a0"],
            "content": ["\u5e02\u573a\u8d44\u91d1\u4ea4\u6613\u6570\u636e\u663e\u793a\u77ed\u7ebf\u70ed\u5ea6\u56de\u843d"],
            "published_at": pd.to_datetime(["2026-01-02 11:00"]),
            "source": ["eastmoney"],
        }
    )

    events = news_impact.classify_news_events(raw)

    assert events.loc[0, "event_category"] == "fund_flow_market_heat"
    assert events.loc[0, "event_direction"] == "bearish"


def test_build_event_impact_dataset_maps_after_close_to_next_trade_day():
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
            "open": [9.8, 10.2, 10.7],
            "close": [10.0, 10.6, 11.0],
            "high": [10.1, 10.8, 11.2],
            "low": [9.7, 10.1, 10.5],
            "volume": [1000, 1200, 1400],
        }
    )
    events = news_impact.classify_news_events(
        pd.DataFrame(
            {
                "symbol": ["000001"],
                "title": ["share buyback approved"],
                "content": ["company will repurchase shares"],
                "published_at": pd.to_datetime(["2026-01-02 18:00"]),
                "source": ["cninfo"],
            }
        )
    )

    impact = news_impact.build_event_impact_dataset(events, prices, horizons=(1, 2))

    assert impact.loc[0, "impact_trade_date"] == pd.Timestamp("2026-01-05")
    assert impact.loc[0, "baseline_trade_date"] == pd.Timestamp("2026-01-02")
    assert np.isclose(impact.loc[0, "open_gap_pct"], (10.2 / 10.0 - 1.0) * 100)
    assert np.isclose(impact.loc[0, "return_1d_pct"], (10.6 / 10.0 - 1.0) * 100)
    assert impact.loc[0, "direction_hit_1d"] == 1.0


def test_build_event_impact_dataset_keeps_close_return_when_open_missing():
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
            "close": [10.0, 10.6, 11.0],
        }
    )
    events = news_impact.classify_news_events(
        pd.DataFrame(
            {
                "symbol": ["000001"],
                "title": ["share buyback approved"],
                "content": ["company will repurchase shares"],
                "published_at": pd.to_datetime(["2026-01-05 10:00"]),
                "source": ["cninfo"],
            }
        )
    )

    impact = news_impact.build_event_impact_dataset(events, prices, horizons=(1, 2))

    assert len(impact) == 1
    assert np.isclose(impact.loc[0, "return_1d_pct"], (10.6 / 10.0 - 1.0) * 100)
    assert np.isclose(impact.loc[0, "open_gap_pct"], (10.6 / 10.0 - 1.0) * 100)


def test_build_event_impact_dataset_accepts_named_trade_date_index():
    prices = pd.DataFrame(
        {
            "open": [9.8, 10.2, 10.7],
            "close": [10.0, 10.6, 11.0],
            "high": [10.1, 10.8, 11.2],
            "low": [9.7, 10.1, 10.5],
            "volume": [1000, 1200, 1400],
        },
        index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
    )
    prices.index.name = "trade_date"
    events = news_impact.classify_news_events(
        pd.DataFrame(
            {
                "symbol": ["000001"],
                "title": ["share buyback approved"],
                "content": ["company will repurchase shares"],
                "published_at": pd.to_datetime(["2026-01-02 18:00"]),
                "source": ["cninfo"],
            }
        )
    )

    impact = news_impact.build_event_impact_dataset(events, prices, horizons=(1,))

    assert impact.loc[0, "impact_trade_date"] == pd.Timestamp("2026-01-05")
    assert impact.loc[0, "baseline_trade_date"] == pd.Timestamp("2026-01-02")
    assert np.isclose(impact.loc[0, "return_1d_pct"], (10.6 / 10.0 - 1.0) * 100)


def test_summarize_category_impact_reports_direction_hit_rate():
    impact = pd.DataFrame(
        {
            "event_category": ["contract_order", "contract_order", "regulatory_risk"],
            "event_direction": ["bullish", "bullish", "bearish"],
            "event_sentiment": [0.5, 0.4, -0.6],
            "event_confidence": [0.8, 0.7, 0.9],
            "expected_impact_score": [72, 68, 25],
            "open_gap_pct": [1.0, 0.5, -1.5],
            "return_1d_pct": [3.0, 2.0, -4.0],
            "direction_hit_1d": [1.0, 1.0, 1.0],
        }
    )

    summary = news_impact.summarize_category_impact(impact, horizons=(1,))
    contract = summary.set_index("event_category").loc["contract_order"]
    risk = summary.set_index("event_category").loc["regulatory_risk"]

    assert contract["event_count"] == 2
    assert contract["avg_return_1d_pct"] == 2.5
    assert contract["direction_hit_rate_1d"] == 1.0
    assert risk["avg_return_1d_pct"] == -4.0
    assert risk["direction_hit_rate_1d"] == 1.0


def test_research_enhanced_news_signal_uses_large_sample_priors():
    raw = pd.DataFrame(
        {
            "symbol": ["000001"],
            "title": ["new product approval and technology breakthrough"],
            "content": ["patent approval improves product pipeline"],
            "published_at": pd.to_datetime(["2026-01-02 10:00"]),
            "source": ["newswire"],
        }
    )

    signal = news_impact.build_research_enhanced_news_signal(
        raw,
        base_signal={"sentiment_score": 50.0, "confidence_score": 30.0},
        symbol="000001",
    )

    assert signal["research_prior_version"] == news_impact.NEWS_RESEARCH_PRIOR_VERSION
    assert signal["research_impact_score"] > 60.0
    assert signal["sentiment_score"] > 50.0
    assert signal["research_expected_excess_return_1d_pct"] > 0.0
    assert signal["research_primary_category"] == "product_technology"


def test_analyze_symbol_news_impact_uses_fetchers(monkeypatch):
    raw_news = pd.DataFrame(
        {
            "title": ["new order contract signed"],
            "content": ["large contract improves backlog"],
            "published_at": pd.to_datetime(["2026-01-02 10:00"]),
            "source": ["newswire"],
        }
    )
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"]),
            "open": [9.8, 10.0, 10.8],
            "close": [10.0, 10.5, 11.0],
            "high": [10.1, 10.7, 11.1],
            "low": [9.7, 9.9, 10.6],
            "volume": [1000, 1300, 1500],
        }
    ).set_index("date", drop=False)
    monkeypatch.setattr(news_impact, "fetch_stock_news", lambda symbol, limit=120: raw_news.copy())
    monkeypatch.setattr(news_impact, "fetch_daily_history", lambda *args, **kwargs: prices.copy())

    result = news_impact.analyze_symbol_news_impact("000001", include_disclosures=False)

    assert result["symbol"] == "000001"
    assert result["event_count"] == 1
    assert result["impact_sample_count"] == 1
    assert not result["category_summary"].empty
    assert result["latest_signal"]["label"] in {"bullish", "neutral"}


def test_load_news_impact_payload_serializes_component(monkeypatch):
    from a_share_predictor import api_service

    def fake_analyze(*args, **kwargs):
        return {
            "symbol": "000001",
            "event_count": 1,
            "impact_sample_count": 1,
            "horizons": (1, 3),
            "latest_signal": {"score": 66.0, "label": "bullish"},
            "category_summary": pd.DataFrame(
                [{"event_category": "contract_order", "event_count": 1, "avg_return_1d_pct": 3.2}]
            ),
            "event_impacts": pd.DataFrame(
                [{"title": "order", "impact_trade_date": pd.Timestamp("2026-01-02"), "return_1d_pct": 3.2}]
            ),
            "events": pd.DataFrame(
                [{"title": "order", "published_at": pd.Timestamp("2026-01-01 10:00"), "event_category": "contract_order"}]
            ),
        }

    monkeypatch.setattr(api_service, "analyze_symbol_news_impact", fake_analyze)

    payload = api_service.load_news_impact_payload("000001", horizons=(1, 3))

    assert payload["symbol"] == "000001"
    assert payload["latest_signal"]["label"] == "bullish"
    assert payload["category_summary"][0]["event_category"] == "contract_order"
    assert payload["event_impacts"][0]["impact_trade_date"].startswith("2026-01-02")
