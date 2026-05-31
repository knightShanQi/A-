from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


NEWS_TITLE_KEYS = ("新闻标题", "鏂伴椈鏍囬")
NEWS_BODY_KEYS = ("新闻内容", "鏂伴椈鍐呭")
NEWS_TIME_KEYS = ("发布时间", "鍙戝竷鏃堕棿")
NEWS_SOURCE_KEYS = ("文章来源", "来源", "鏂囩珷鏉ユ簮", "鏉ユ簮", "source")
FUND_RATIO_KEYS = ("主力净流入-净占比", "涓诲姏鍑€娴佸叆-鍑€鍗犳瘮")
FUND_NET_KEYS = ("主力净流入-净额", "涓诲姏鍑€娴佸叆-鍑€棰?")


def _clip_score(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return float(max(lower, min(value, upper)))


def _first_value(row: pd.Series, keys: tuple[str, ...], default=None):
    for key in keys:
        if key in row.index:
            value = row.get(key)
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                return value
    return default


def _keyword_hits(text: str, mapping: dict[str, float]) -> tuple[float, int]:
    score = 0.0
    hit_count = 0
    for keyword, weight in mapping.items():
        if keyword in text:
            score += weight
            hit_count += 1
    return score, hit_count


@dataclass(slots=True)
class QuantSignal:
    total_score: float
    primary_signal: str
    momentum_score: float
    breakout_score: float
    pullback_score: float
    risk_score: float
    summary: str


def evaluate_quant_signal(daily: pd.DataFrame, features: pd.DataFrame) -> QuantSignal:
    latest = features.dropna().iloc[-1]
    momentum_score = _clip_score(
        50
        + latest["ret_20"] * 220
        + latest["ret_60"] * 120
        + latest["ret_120"] * 70
        + latest["close_vs_ma20"] * 180
        + latest["close_vs_ma120"] * 120
        + latest["ma20_slope_5"] * 800
        + latest["ma5_slope_3"] * 650
        + (latest["efficiency_ratio_10"] - 0.5) * 36
    )
    breakout_score = _clip_score(
        45
        + latest["breakout_distance_20"] * 700
        + latest["range_position_20"] * 25
        + latest["close_near_high_5"] * 260
        + (latest["volume_ratio_5"] - 1.0) * 22
    )
    pullback_score = _clip_score(
        55
        - abs(latest["pullback_to_breakout_20"]) * 900
        + latest["close_vs_ma20"] * 90
        + latest["lower_shadow_ratio"] * 550
        + latest["drawdown_20"] * 120
    )
    risk_score = _clip_score(
        35
        + max(latest["upper_shadow_ratio"], 0) * 900
        + max(latest["volatility_10"], 0) * 1400
        + max(latest["downside_vol_ratio_20"], 0) * 260
        + abs(min(latest["drawdown_20"], 0)) * 120
        + max(latest["close_vs_ma20"] - 0.12, 0) * 300
    )

    total_score = _clip_score(
        momentum_score * 0.34
        + breakout_score * 0.28
        + pullback_score * 0.20
        + (100 - risk_score) * 0.18
    )

    signals = {
        "趋势跟随": momentum_score,
        "突破追踪": breakout_score,
        "回踩确认": pullback_score,
    }
    primary_signal = max(signals, key=signals.get)

    if total_score >= 70:
        summary = "量化辅助偏多，趋势、突破或回踩确认里至少有一类在发力。"
    elif total_score >= 55:
        summary = "量化辅助中性偏多，可以作为技术观察的辅助，但不宜单独决策。"
    else:
        summary = "量化辅助偏谨慎，当前更适合结合消息面和资金流做二次确认。"

    return QuantSignal(
        total_score=round(total_score, 2),
        primary_signal=primary_signal,
        momentum_score=round(momentum_score, 2),
        breakout_score=round(breakout_score, 2),
        pullback_score=round(pullback_score, 2),
        risk_score=round(risk_score, 2),
        summary=summary,
    )


BULLISH_KEYWORDS = {
    "回购": 2.6,
    "增持": 2.4,
    "中标": 2.2,
    "订单": 2.0,
    "合作": 1.6,
    "预增": 2.8,
    "超预期": 2.6,
    "分红": 1.8,
    "扩产": 1.5,
    "突破": 1.2,
    "创新高": 1.8,
    "签约": 1.4,
    "提价": 1.8,
    "改善": 1.2,
    "能力提升": 1.6,
    "放量": 1.3,
}

BEARISH_KEYWORDS = {
    "减持": -2.6,
    "立案": -2.8,
    "处罚": -2.8,
    "亏损": -2.2,
    "风险": -1.4,
    "诉讼": -1.9,
    "质押": -1.2,
    "终止": -1.8,
    "下调": -1.5,
    "暴跌": -2.3,
    "跌停": -2.2,
    "ST": -3.0,
    "监管": -1.4,
    "问题": -1.2,
    "下滑": -1.4,
    "不及预期": -2.0,
    "回落": -1.2,
}

SOURCE_WEIGHTS = {
    "证券时报": 1.10,
    "中国证券报": 1.08,
    "上海证券报": 1.06,
    "第一财经": 1.05,
    "财联社": 1.04,
    "东方财富": 1.02,
}


def evaluate_news_sentiment(news_df: pd.DataFrame) -> dict[str, float | str]:
    if news_df.empty:
        return {
            "sentiment_score": 50.0,
            "confidence_score": 25.0,
            "headline_count": 0,
            "label": "消息面中性",
            "summary": "当前没有抓到最近新闻，消息面对评分不形成明显偏置。",
            "source_count": 0,
            "positive_hits": 0,
            "negative_hits": 0,
        }

    score = 0.0
    total_weight = 0.0
    confidence_sum = 0.0
    positive_hits = 0
    negative_hits = 0
    sources: set[str] = set()
    now = pd.Timestamp.now()

    for _, row in news_df.iterrows():
        title = str(_first_value(row, NEWS_TITLE_KEYS, "") or "")
        body = str(_first_value(row, NEWS_BODY_KEYS, "") or "")
        title_bull_score, title_bull_hits = _keyword_hits(title, BULLISH_KEYWORDS)
        title_bear_score, title_bear_hits = _keyword_hits(title, BEARISH_KEYWORDS)
        body_bull_score, body_bull_hits = _keyword_hits(body, BULLISH_KEYWORDS)
        body_bear_score, body_bear_hits = _keyword_hits(body, BEARISH_KEYWORDS)

        title_score = title_bull_score + title_bear_score
        body_score = body_bull_score + body_bear_score
        item_score = title_score * 1.35 + body_score * 0.85
        hit_total = title_bull_hits + title_bear_hits + body_bull_hits + body_bear_hits
        contradiction_penalty = 0.88 if (title_bull_hits + body_bull_hits) and (title_bear_hits + body_bear_hits) else 1.0

        publish_time = pd.to_datetime(_first_value(row, NEWS_TIME_KEYS), errors="coerce")
        age_hours = 72.0 if pd.isna(publish_time) else max((now - publish_time).total_seconds() / 3600, 0.0)
        recency_weight = float(np.clip(np.exp(-age_hours / 72), 0.28, 1.0))

        source = str(_first_value(row, NEWS_SOURCE_KEYS, "未知来源") or "未知来源")
        source_weight = SOURCE_WEIGHTS.get(source, 1.0)
        direction_strength = abs(item_score) / max(abs(title_score) + abs(body_score), 1.0)
        confidence_piece = min(1.0, 0.32 + direction_strength * 0.48 + min(hit_total / 4, 0.20))
        item_weight = recency_weight * source_weight

        score += item_score * contradiction_penalty * item_weight
        total_weight += item_weight
        confidence_sum += confidence_piece * item_weight
        positive_hits += title_bull_hits + body_bull_hits
        negative_hits += title_bear_hits + body_bear_hits
        sources.add(source)

    avg = score / total_weight if total_weight else 0.0
    headline_depth = min(len(news_df) / 8, 1.0)
    source_diversity = min(len(sources) / max(len(news_df), 1), 1.0)
    confidence_score = _clip_score(
        34
        + headline_depth * 18
        + source_diversity * 12
        + (confidence_sum / max(total_weight, 1e-6)) * 22
        + min(abs(avg) * 6, 12)
    )
    confidence_scale = 0.50 + confidence_score / 200
    normalized = _clip_score(50 + avg * 9.5 * confidence_scale)

    if normalized >= 62:
        label = "消息面偏多"
        summary = "最近新闻显示利多事件占优，系统已对来源可信度、发布时间和前后矛盾信息做了降噪处理。"
    elif normalized <= 38:
        label = "消息面偏空"
        summary = "最近新闻以风险、减持或不及预期类事件为主，短线情绪约束更强。"
    else:
        label = "消息面中性"
        summary = "最近新闻没有形成高一致性的单边倾向，执行上更应该依赖价格和资金确认。"

    return {
        "sentiment_score": round(normalized, 2),
        "confidence_score": round(confidence_score, 2),
        "headline_count": int(len(news_df)),
        "label": label,
        "summary": summary,
        "source_count": int(len(sources)),
        "positive_hits": int(positive_hits),
        "negative_hits": int(negative_hits),
    }


def evaluate_main_fund_signal(fund_flow_df: pd.DataFrame) -> dict[str, float | str]:
    if fund_flow_df.empty:
        return {
            "fund_score": 50.0,
            "confidence_score": 25.0,
            "label": "主力资金中性",
            "summary": "当前没有抓到个股主力资金流数据。",
            "inflow_streak": 0,
            "positive_day_ratio": 0.0,
        }

    recent = fund_flow_df.head(5).copy()
    ratio_series = pd.to_numeric(
        recent[_first_existing_column(recent, FUND_RATIO_KEYS)],
        errors="coerce",
    ).fillna(0.0)
    net_series = pd.to_numeric(
        recent[_first_existing_column(recent, FUND_NET_KEYS)],
        errors="coerce",
    ).fillna(0.0)

    weights = np.array([1.0, 0.86, 0.74, 0.62, 0.50], dtype=float)[: len(recent)]
    weights = weights / weights.sum()
    weighted_ratio = float(np.average(ratio_series.to_numpy(dtype=float), weights=weights))
    weighted_net = float(np.average(net_series.to_numpy(dtype=float), weights=weights))
    positive_day_ratio = float((ratio_series > 0).mean())
    short_trend = float(ratio_series.head(min(3, len(ratio_series))).mean())
    medium_trend = float(ratio_series.tail(max(len(ratio_series) - 2, 1)).mean())
    trend_delta = short_trend - medium_trend
    inflow_streak = 0
    for value in ratio_series.tolist():
        if float(value) > 0:
            inflow_streak += 1
            continue
        break

    ratio_stability = 1.0 - min(float(ratio_series.std(ddof=0)) / max(abs(weighted_ratio) + 0.8, 1.2), 1.0)
    score = _clip_score(
        50
        + weighted_ratio * 3.6
        + np.sign(weighted_net) * min(abs(weighted_net) / 2e8, 12)
        + (positive_day_ratio - 0.5) * 22
        + trend_delta * 1.8
        + inflow_streak * 2.6
    )
    confidence_score = _clip_score(
        32
        + len(recent) * 7
        + positive_day_ratio * 18
        + ratio_stability * 24
        + min(abs(weighted_ratio) * 1.6, 10)
    )

    if score >= 62:
        label = "主力资金偏强"
        summary = "主力资金不只当日偏强，而且有连续性和一定一致性，短线关注价值更高。"
    elif score <= 38:
        label = "主力资金偏弱"
        summary = "最近几日主力资金偏出或缺乏承接，短线更适合等待二次确认。"
    else:
        label = "主力资金中性"
        summary = "主力资金还没有形成明显的连续偏向，更适合继续看其是否与行业和价格形成共振。"

    return {
        "fund_score": round(score, 2),
        "confidence_score": round(confidence_score, 2),
        "label": label,
        "summary": summary,
        "main_ratio": round(weighted_ratio, 2),
        "main_net": round(weighted_net, 2),
        "inflow_streak": int(inflow_streak),
        "positive_day_ratio": round(positive_day_ratio, 2),
        "trend_delta": round(trend_delta, 2),
    }


def _first_existing_column(frame: pd.DataFrame, keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in frame.columns:
            return key
    return keys[0]


def compute_sector_hot_score(industry_name: str, industry_flow_df: pd.DataFrame) -> dict[str, float | str]:
    if industry_flow_df.empty:
        return {
            "sector_score": 50.0,
            "sector_label": "行业热度未知",
            "sector_summary": "当前没有抓到行业资金流数据。",
        }

    normalized = industry_name.strip()
    normalized = normalized.replace("Ⅰ", "").replace("Ⅱ", "").replace("Ⅲ", "")
    normalized = normalized.replace("（", "(").replace("）", ")").replace(" ", "")
    df = industry_flow_df.copy()
    match = df[df["sector_name_normalized"] == normalized]
    if match.empty:
        match = df[df["sector_name_normalized"].str.contains(normalized[:2], na=False)]
    if match.empty:
        return {
            "sector_score": 50.0,
            "sector_label": "行业热度未知",
            "sector_summary": f"没有在当日行业资金流榜里匹配到 `{industry_name}`。",
        }

    row = match.iloc[0]
    rank = int(row.name) + 1
    rank_score = max(0.0, 100 - rank * 2.2)
    net_inflow = float(row.get("net_inflow", 0.0))
    change_pct = float(row.get("change_pct", 0.0))
    score = _clip_score(rank_score * 0.55 + (50 + net_inflow * 0.18) * 0.30 + (50 + change_pct * 8) * 0.15)

    if score >= 68:
        label = "行业热度较高"
    elif score >= 55:
        label = "行业热度中上"
    else:
        label = "行业热度一般"

    summary = f"{industry_name} 在行业资金流榜大致位于前 {rank}，净流入 {net_inflow:.2f} 亿。"
    return {
        "sector_score": round(score, 2),
        "sector_label": label,
        "sector_summary": summary,
        "matched_sector": str(row.get("sector_name", industry_name)),
    }
