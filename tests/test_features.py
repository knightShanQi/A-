import pandas as pd

from a_share_predictor.features import build_daily_features, build_training_frame, evaluate_intraday


def make_sample_daily() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    base = pd.Series(range(120), index=dates, dtype=float)
    close = 10 + base * 0.08
    daily = pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.05,
            "close": close,
            "high": close + 0.12,
            "low": close - 0.12,
            "volume": 100000 + base * 1200,
            "amount": (100000 + base * 1200) * close,
            "turnover": 2.0 + base * 0.01,
        },
        index=dates,
    )
    return daily


def test_build_daily_features_has_expected_columns():
    daily = make_sample_daily()
    features = build_daily_features(daily)
    assert "close_vs_ma20" in features.columns
    assert "breakout_distance_20" in features.columns
    assert "rsi_14" in features.columns
    assert "atr_ratio_14" in features.columns
    assert "ma_alignment_score" in features.columns
    assert features.dropna().shape[0] > 0


def test_training_frame_has_target():
    daily = make_sample_daily()
    dataset = build_training_frame(daily, horizon_days=5, positive_return=0.02)
    assert "target" in dataset.columns
    assert dataset["target"].isin([0, 1]).all()


def test_evaluate_intraday_handles_minute_drawdown_series():
    minute = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-03-12 09:31:00", periods=4, freq="min"),
            "close": [10.0, 10.3, 10.1, 10.4],
            "avg_price": [10.0, 10.1, 10.12, 10.18],
            "volume": [100, 120, 110, 130],
        }
    )

    result = evaluate_intraday(minute)

    assert result["label"]
    assert 0 <= result["score"] <= 1
    assert result["max_pullback"] >= 0
