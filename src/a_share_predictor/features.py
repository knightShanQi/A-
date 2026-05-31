from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_60",
    "ret_120",
    "close_vs_ma5",
    "close_vs_ma20",
    "close_vs_ma60",
    "close_vs_ma120",
    "ma5_slope_3",
    "ma20_slope_5",
    "ma60_slope_10",
    "volume_ratio_5",
    "volume_ratio_20",
    "turnover",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "body_ratio",
    "range_position_20",
    "range_position_60",
    "breakout_distance_20",
    "breakout_distance_60",
    "pullback_to_breakout_20",
    "consolidation_width_10",
    "consolidation_width_20",
    "volatility_10",
    "volatility_20",
    "amount_ratio_5",
    "rsi_14",
    "atr_ratio_14",
    "gap_return_1",
    "close_position_day",
    "up_day_ratio_10",
    "ma_alignment_score",
    "volatility_contraction",
    "turnover_ratio_20",
    "momentum_persistence_10",
    "drawdown_20",
    "efficiency_ratio_10",
    "downside_vol_ratio_20",
    "close_near_high_5",
]


def _rolling_position(close: pd.Series, low_roll: pd.Series, high_roll: pd.Series) -> pd.Series:
    width = (high_roll - low_roll).replace(0, np.nan)
    return (close - low_roll) / width


def build_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    close = daily["close"]
    high = daily["high"]
    low = daily["low"]
    open_ = daily["open"]
    volume = daily["volume"]
    amount = daily["amount"]
    turnover = daily["turnover"] if "turnover" in daily.columns else pd.Series(np.nan, index=daily.index)

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120, min_periods=60).mean()
    prev_close = close.shift(1)
    high5 = high.rolling(5).max()
    high20_prev = high.rolling(20).max().shift(1)
    high60_prev = high.rolling(60).max().shift(1)
    low20 = low.rolling(20).min()
    low60 = low.rolling(60).min()
    high20 = high.rolling(20).max()
    high60 = high.rolling(60).max()
    turnover_ma20 = turnover.rolling(20).mean()

    body = (close - open_).abs()
    candle_range = (high - low).replace(0, np.nan)
    upper_shadow = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_shadow = pd.concat([open_, close], axis=1).min(axis=1) - low
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = true_range.rolling(14).mean()
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi14 = 100 - (100 / (1 + rs))
    rsi14 = rsi14.where(avg_loss.ne(0), 100.0)
    rsi14 = rsi14.where(~(avg_gain.eq(0) & avg_loss.eq(0)), 50.0)
    ma_alignment = (
        (close > ma5).astype(float)
        + (ma5 > ma20).astype(float)
        + (ma20 > ma60).astype(float)
    ) / 3
    positive_day = close.pct_change().gt(0).astype(float)
    abs_directional_move_10 = close.diff().abs().rolling(10).sum().replace(0, np.nan)
    efficiency_ratio_10 = close.diff(10).abs() / abs_directional_move_10
    downside_returns = close.pct_change().where(close.pct_change() < 0, 0.0)
    volatility_20 = close.pct_change().rolling(20).std()
    downside_vol_ratio_20 = downside_returns.rolling(20).std() / volatility_20.replace(0, np.nan)

    feats = pd.DataFrame(index=daily.index)
    feats["ret_1"] = close.pct_change()
    feats["ret_3"] = close.pct_change(3)
    feats["ret_5"] = close.pct_change(5)
    feats["ret_10"] = close.pct_change(10)
    feats["ret_20"] = close.pct_change(20)
    feats["ret_60"] = close.pct_change(60)
    feats["ret_120"] = close.pct_change(120).fillna(close.pct_change(90))
    feats["close_vs_ma5"] = close / ma5 - 1
    feats["close_vs_ma20"] = close / ma20 - 1
    feats["close_vs_ma60"] = close / ma60 - 1
    feats["close_vs_ma120"] = close / ma120 - 1
    feats["ma5_slope_3"] = ma5 / ma5.shift(3) - 1
    feats["ma20_slope_5"] = ma20 / ma20.shift(5) - 1
    feats["ma60_slope_10"] = ma60 / ma60.shift(10) - 1
    feats["volume_ratio_5"] = volume / volume.rolling(5).mean()
    feats["volume_ratio_20"] = volume / volume.rolling(20).mean()
    feats["turnover"] = turnover
    feats["upper_shadow_ratio"] = upper_shadow / close
    feats["lower_shadow_ratio"] = lower_shadow / close
    feats["body_ratio"] = body / candle_range
    feats["range_position_20"] = _rolling_position(close, low20, high20)
    feats["range_position_60"] = _rolling_position(close, low60, high60)
    feats["breakout_distance_20"] = close / high20_prev - 1
    feats["breakout_distance_60"] = close / high60_prev - 1
    feats["pullback_to_breakout_20"] = low / high20_prev - 1
    feats["consolidation_width_10"] = (high.rolling(10).max() - low.rolling(10).min()) / close
    feats["consolidation_width_20"] = (high20 - low20) / close
    feats["volatility_10"] = feats["ret_1"].rolling(10).std()
    feats["volatility_20"] = volatility_20
    feats["amount_ratio_5"] = amount / amount.rolling(5).mean()
    feats["rsi_14"] = rsi14 / 100
    feats["atr_ratio_14"] = atr14 / close
    feats["gap_return_1"] = open_ / prev_close - 1
    feats["close_position_day"] = (close - low) / candle_range
    feats["up_day_ratio_10"] = positive_day.rolling(10).mean()
    feats["ma_alignment_score"] = ma_alignment
    feats["volatility_contraction"] = feats["volatility_10"] / feats["volatility_20"].replace(0, np.nan) - 1
    feats["turnover_ratio_20"] = turnover / turnover_ma20.replace(0, np.nan)
    feats["momentum_persistence_10"] = close.gt(ma5).astype(float).rolling(10).mean()
    feats["drawdown_20"] = close / high20.replace(0, np.nan) - 1
    feats["efficiency_ratio_10"] = efficiency_ratio_10
    feats["downside_vol_ratio_20"] = downside_vol_ratio_20
    feats["close_near_high_5"] = close / high5.replace(0, np.nan) - 1
    return feats


def build_training_frame(
    daily: pd.DataFrame,
    horizon_days: int = 5,
    positive_return: float = 0.03,
) -> pd.DataFrame:
    features = build_daily_features(daily)
    future_return = daily["close"].shift(-horizon_days) / daily["close"] - 1
    dataset = features.copy()
    dataset["future_return"] = future_return
    dataset["target"] = (future_return >= positive_return).astype(float)
    dataset = dataset.dropna(subset=FEATURE_COLUMNS + ["future_return", "target"]).copy()
    dataset["target"] = dataset["target"].astype(int)
    return dataset


def latest_snapshot(daily: pd.DataFrame, features: pd.DataFrame) -> dict[str, float]:
    latest_daily = daily.iloc[-1]
    latest_feat = features.dropna().iloc[-1]
    return {
        "date": latest_daily["date"].strftime("%Y-%m-%d"),
        "close": float(latest_daily["close"]),
        "change_pct": float(latest_daily.get("change_pct", np.nan)),
        "turnover": float(latest_daily.get("turnover", np.nan)),
        "close_vs_ma20": float(latest_feat["close_vs_ma20"]),
        "close_vs_ma60": float(latest_feat["close_vs_ma60"]),
        "volume_ratio_5": float(latest_feat["volume_ratio_5"]),
        "breakout_distance_20": float(latest_feat["breakout_distance_20"]),
        "ret_20": float(latest_feat["ret_20"]),
        "ret_60": float(latest_feat["ret_60"]),
    }


def evaluate_intraday(minute_df: pd.DataFrame) -> dict[str, float | str]:
    if minute_df.empty:
        return {
            "label": "待开盘/暂无分时",
            "score": 0.45,
            "summary": "当前没有可用的当日 1 分钟数据，暂用日 K 阶段判断。",
        }

    above_avg_ratio = float((minute_df["close"] >= minute_df["avg_price"]).mean())
    latest_close = float(minute_df["close"].iloc[-1])
    latest_avg = float(minute_df["avg_price"].iloc[-1])
    final_above_avg = latest_close >= latest_avg
    pullback = (1 - minute_df["close"] / minute_df["close"].cummax()).clip(lower=0)
    max_pullback = float(pullback.max())
    morning_volume = float(minute_df["volume"].head(30).sum()) or 1.0
    tail_volume = float(minute_df["volume"].tail(30).sum())
    tail_ratio = tail_volume / morning_volume

    score = 0.0
    score += min(above_avg_ratio, 1.0) * 0.45
    score += (0.30 if final_above_avg else 0.08)
    score += max(0.0, 0.20 - min(max_pullback, 0.20))
    score += min(tail_ratio, 1.0) * 0.05
    score = float(min(max(score, 0.0), 1.0))

    if score >= 0.72:
        label = "分时偏强"
        summary = "大部分时间站在均价线之上，回踩不深，资金控制力较强。"
    elif score >= 0.55:
        label = "分时承接尚可"
        summary = "均价线附近有承接，但还没有强到纯逼空。"
    else:
        label = "分时偏弱/博弈"
        summary = "均价线压制或回落较深，说明当天分歧更大。"

    return {
        "label": label,
        "score": round(score, 4),
        "summary": summary,
        "above_avg_ratio": round(above_avg_ratio, 4),
        "max_pullback": round(max_pullback, 4),
    }
