import pandas as pd

from a_share_predictor.features import build_daily_features
from a_share_predictor.quant import evaluate_main_fund_signal, evaluate_news_sentiment, evaluate_quant_signal


def make_trend_daily() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=220, freq="B")
    base = pd.Series(range(220), index=dates, dtype=float)
    close = 12 + base * 0.1
    daily = pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.05,
            "close": close,
            "high": close + 0.15,
            "low": close - 0.15,
            "volume": 100000 + base * 1000,
            "amount": (100000 + base * 1000) * close,
            "turnover": 2.0 + base * 0.01,
        },
        index=dates,
    )
    return daily


def test_news_sentiment_prefers_bullish_headlines():
    news_df = pd.DataFrame(
        {
            "新闻标题": ["公司发布回购计划并签订大订单", "业绩预增超预期"],
            "新闻内容": ["董事会通过回购方案", "新签订单推动盈利改善"],
            "发布时间": pd.to_datetime(["2026-03-12 09:00:00", "2026-03-11 18:00:00"]),
            "文章来源": ["证券时报", "东方财富"],
        }
    )

    result = evaluate_news_sentiment(news_df)

    assert result["sentiment_score"] > 50
    assert result["confidence_score"] > 40
    assert result["positive_hits"] > result["negative_hits"]


def test_main_fund_signal_prefers_consistent_inflow():
    fund_df = pd.DataFrame(
        {
            "主力净流入-净占比": [8.2, 6.4, 5.8, 2.1, -0.5],
            "主力净流入-净额": [4.2e8, 3.6e8, 2.7e8, 1.2e8, -0.3e8],
        }
    )

    result = evaluate_main_fund_signal(fund_df)

    assert result["fund_score"] > 55
    assert result["confidence_score"] > 45
    assert result["inflow_streak"] >= 3


def test_quant_signal_recognizes_trending_series():
    daily = make_trend_daily()
    features = build_daily_features(daily)
    signal = evaluate_quant_signal(daily, features)

    assert signal.total_score > 55
