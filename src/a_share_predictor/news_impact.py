from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import data as data_module
from .data import fetch_daily_history, fetch_stock_news, normalize_symbol
from .quant import (
    BEARISH_KEYWORDS,
    BULLISH_KEYWORDS,
    NEWS_BODY_KEYS,
    NEWS_SOURCE_KEYS,
    NEWS_TIME_KEYS,
    NEWS_TITLE_KEYS,
    SOURCE_WEIGHTS,
)


TITLE_COLUMNS = (
    *NEWS_TITLE_KEYS,
    "title",
    "headline",
    "news_title",
    "\u65b0\u95fb\u6807\u9898",
    "\u6807\u9898",
    "\u516c\u544a\u6807\u9898",
)
BODY_COLUMNS = (
    *NEWS_BODY_KEYS,
    "content",
    "body",
    "summary",
    "abstract",
    "\u65b0\u95fb\u5185\u5bb9",
    "\u5185\u5bb9",
    "\u516c\u544a\u5185\u5bb9",
)
TIME_COLUMNS = (
    *NEWS_TIME_KEYS,
    "published_at",
    "publish_time",
    "datetime",
    "date",
    "\u53d1\u5e03\u65f6\u95f4",
    "\u516c\u544a\u65e5\u671f",
    "\u516c\u544a\u65f6\u95f4",
    "\u62ab\u9732\u65f6\u95f4",
)
SOURCE_COLUMNS = (
    *NEWS_SOURCE_KEYS,
    "source",
    "media",
    "\u6587\u7ae0\u6765\u6e90",
    "\u6765\u6e90",
    "\u4fe1\u606f\u6765\u6e90",
)
URL_COLUMNS = (
    "url",
    "link",
    "article_url",
    "news_url",
    "\u65b0\u95fb\u94fe\u63a5",
    "\u516c\u544a\u94fe\u63a5",
    "\u94fe\u63a5",
)
KEYWORD_COLUMNS = (
    "keyword",
    "keywords",
    "\u5173\u952e\u8bcd",
    "\u80a1\u7968\u7b80\u79f0",
    "\u7b80\u79f0",
)
SYMBOL_COLUMNS = (
    "symbol",
    "code",
    "stock_code",
    "security_code",
    "\u80a1\u7968\u4ee3\u7801",
    "\u8bc1\u5238\u4ee3\u7801",
    "\u4ee3\u7801",
)

NEWS_CANONICAL_COLUMNS = [
    "symbol",
    "title",
    "content",
    "published_at",
    "source",
    "url",
    "keyword",
]
CLASSIFIED_NEWS_COLUMNS = [
    *NEWS_CANONICAL_COLUMNS,
    "event_category",
    "event_label",
    "event_direction",
    "event_sentiment",
    "event_confidence",
    "expected_impact_score",
    "event_strength",
    "positive_hits",
    "negative_hits",
    "neutral_hits",
    "matched_keywords",
    "session_bucket",
    "impact_horizon_days",
    "source_weight",
    "dedupe_key",
]
IMPACT_COLUMNS = [
    *CLASSIFIED_NEWS_COLUMNS,
    "impact_trade_date",
    "baseline_trade_date",
    "baseline_close",
    "impact_open",
    "impact_close",
    "open_gap_pct",
    "same_day_return_pct",
]


@dataclass(frozen=True, slots=True)
class NewsEventRule:
    category: str
    label: str
    positive_keywords: tuple[str, ...]
    negative_keywords: tuple[str, ...] = ()
    neutral_keywords: tuple[str, ...] = ()
    base_weight: float = 1.0
    horizon_days: int = 3


EVENT_RULES: tuple[NewsEventRule, ...] = (
    NewsEventRule(
        category="earnings",
        label="earnings",
        positive_keywords=(
            "profit growth",
            "beat",
            "turnaround",
            "\u9884\u589e",
            "\u626d\u4e8f",
            "\u589e\u957f",
            "\u5927\u589e",
            "\u8d85\u9884\u671f",
            "\u4e1a\u7ee9\u5feb\u62a5",
            "\u4e1a\u7ee9\u9884\u544a",
            "\u5229\u6da6\u589e",
        ),
        negative_keywords=(
            "loss",
            "miss",
            "\u9884\u51cf",
            "\u4e8f\u635f",
            "\u4e0b\u6ed1",
            "\u4e0d\u53ca\u9884\u671f",
            "\u51cf\u503c",
            "\u8ba1\u63d0",
        ),
        neutral_keywords=(
            "\u5e74\u5ea6\u62a5\u544a",
            "\u4e00\u5b63\u5ea6\u62a5\u544a",
            "\u534a\u5e74\u5ea6\u62a5\u544a",
            "\u4e09\u5b63\u5ea6\u62a5\u544a",
            "\u5b63\u5ea6\u62a5\u544a",
            "\u4e1a\u7ee9\u8bf4\u660e\u4f1a",
            "\u4e1a\u7ee9\u5feb\u62a5",
            "\u4e1a\u7ee9\u9884\u544a",
        ),
        base_weight=1.25,
        horizon_days=5,
    ),
    NewsEventRule(
        category="contract_order",
        label="contract / order",
        positive_keywords=(
            "contract",
            "order",
            "tender",
            "\u4e2d\u6807",
            "\u8ba2\u5355",
            "\u5408\u540c",
            "\u7b7e\u7ea6",
            "\u5927\u5355",
            "\u6846\u67b6\u534f\u8bae",
        ),
        base_weight=1.18,
        horizon_days=3,
    ),
    NewsEventRule(
        category="shareholder_action",
        label="shareholder action",
        positive_keywords=(
            "buyback",
            "repurchase",
            "increase holding",
            "\u56de\u8d2d",
            "\u589e\u6301",
            "\u5458\u5de5\u6301\u80a1",
        ),
        negative_keywords=(
            "share reduction",
            "pledge",
            "\u51cf\u6301",
            "\u8d28\u62bc",
            "\u89e3\u9664\u8d28\u62bc",
            "\u88ab\u52a8\u51cf\u6301",
            "\u89e3\u7981",
            "\u9650\u552e\u80a1\u4e0a\u5e02\u6d41\u901a",
        ),
        base_weight=1.22,
        horizon_days=3,
    ),
    NewsEventRule(
        category="dividend_distribution",
        label="dividend / distribution",
        positive_keywords=(
            "\u5206\u7ea2",
            "\u6d3e\u606f",
            "\u73b0\u91d1\u7ea2\u5229",
            "\u6743\u76ca\u5206\u6d3e",
            "\u5229\u6da6\u5206\u914d",
            "\u8f6c\u589e",
            "\u9001\u80a1",
        ),
        neutral_keywords=(
            "\u9664\u6743\u9664\u606f",
            "\u5206\u914d\u65b9\u6848",
        ),
        base_weight=0.84,
        horizon_days=3,
    ),
    NewsEventRule(
        category="financing_mna",
        label="financing / M&A",
        positive_keywords=(
            "merger",
            "acquisition",
            "restructuring",
            "\u5e76\u8d2d",
            "\u6536\u8d2d",
            "\u91cd\u7ec4",
            "\u6ce8\u5165",
            "\u5b9a\u589e",
            "\u6218\u6295",
        ),
        negative_keywords=(
            "terminated",
            "\u7ec8\u6b62",
            "\u5931\u8d25",
            "\u6682\u505c",
        ),
        base_weight=1.12,
        horizon_days=5,
    ),
    NewsEventRule(
        category="policy_sector",
        label="policy / sector",
        positive_keywords=(
            "policy",
            "subsidy",
            "stimulus",
            "\u653f\u7b56",
            "\u89c4\u5212",
            "\u8865\u8d34",
            "\u56fd\u5e38\u4f1a",
            "\u53d1\u6539\u59d4",
            "\u5de5\u4fe1\u90e8",
            "\u4ea7\u4e1a",
        ),
        negative_keywords=(
            "restriction",
            "\u9650\u5236",
            "\u6574\u6cbb",
            "\u538b\u964d",
        ),
        base_weight=0.96,
        horizon_days=5,
    ),
    NewsEventRule(
        category="fund_flow_market_heat",
        label="fund flow / market heat",
        positive_keywords=(
            "\u51c0\u6d41\u5165",
            "\u4e3b\u529b\u51c0\u6d41\u5165",
            "\u878d\u8d44\u4e70\u5165",
            "\u83b7\u878d\u8d44",
            "\u5317\u5411\u8d44\u91d1\u4e70\u5165",
            "\u51c0\u4e70\u5165",
        ),
        negative_keywords=(
            "\u51c0\u6d41\u51fa",
            "\u4e3b\u529b\u51c0\u6d41\u51fa",
            "\u8d44\u91d1\u51fa\u9003",
            "\u4e3b\u529b\u51fa\u9003",
            "\u878d\u8d44\u507f\u8fd8",
            "\u51c0\u5356\u51fa",
        ),
        neutral_keywords=(
            "\u9f99\u864e\u699c",
            "\u4e3b\u529b\u8d44\u91d1",
            "\u878d\u8d44\u878d\u5238",
            "\u878d\u8d44\u5ba2",
            "\u5317\u5411\u8d44\u91d1",
            "\u6caa\u80a1\u901a",
            "\u6df1\u80a1\u901a",
        ),
        base_weight=0.72,
        horizon_days=2,
    ),
    NewsEventRule(
        category="regulatory_risk",
        label="regulatory risk",
        positive_keywords=(),
        negative_keywords=(
            "investigation",
            "penalty",
            "lawsuit",
            "delisting",
            "\u7acb\u6848",
            "\u7acb\u6848\u8c03\u67e5",
            "\u5904\u7f5a",
            "\u884c\u653f\u5904\u7f5a",
            "\u76d1\u7ba1\u51fd",
            "\u95ee\u8be2\u51fd",
            "\u8b66\u793a\u51fd",
            "\u8bc9\u8bbc",
            "\u4ef2\u88c1",
            "\u8fdd\u89c4",
            "\u9000\u5e02",
            "\u88ab\u6267\u884c",
            "\u51bb\u7ed3",
            "ST",
        ),
        base_weight=1.35,
        horizon_days=5,
    ),
    NewsEventRule(
        category="product_technology",
        label="product / technology",
        positive_keywords=(
            "new product",
            "approval",
            "patent",
            "breakthrough",
            "\u65b0\u4ea7\u54c1",
            "\u4e13\u5229",
            "\u8ba4\u8bc1",
            "\u83b7\u6279",
            "\u7a81\u7834",
            "\u4e34\u5e8a",
            "\u9996\u53d1",
        ),
        negative_keywords=(
            "recall",
            "\u53ec\u56de",
            "\u5931\u8d25",
            "\u672a\u83b7\u6279",
        ),
        base_weight=1.02,
        horizon_days=5,
    ),
    NewsEventRule(
        category="market_opinion",
        label="market opinion",
        positive_keywords=(
            "buy rating",
            "upgrade",
            "target price",
            "\u7814\u62a5",
            "\u4e70\u5165",
            "\u4e0a\u8c03",
            "\u76ee\u6807\u4ef7",
            "\u9996\u4e88",
        ),
        negative_keywords=(
            "downgrade",
            "sell rating",
            "\u4e0b\u8c03",
            "\u5356\u51fa",
            "\u770b\u7a7a",
        ),
        base_weight=0.78,
        horizon_days=2,
    ),
    NewsEventRule(
        category="accident_risk",
        label="accident / operation risk",
        positive_keywords=(),
        negative_keywords=(
            "accident",
            "shutdown",
            "fire",
            "\u4e8b\u6545",
            "\u505c\u4ea7",
            "\u706b\u707e",
            "\u5b89\u5168",
            "\u73af\u4fdd",
            "\u53ec\u56de",
        ),
        base_weight=1.18,
        horizon_days=3,
    ),
    NewsEventRule(
        category="routine_governance",
        label="routine governance disclosure",
        positive_keywords=(),
        negative_keywords=(),
        neutral_keywords=(
            "\u80a1\u4e1c\u5927\u4f1a",
            "\u80a1\u4e1c\u4f1a",
            "\u8463\u4e8b\u4f1a\u51b3\u8bae",
            "\u76d1\u4e8b\u4f1a\u51b3\u8bae",
            "\u6cd5\u5f8b\u610f\u89c1\u4e66",
            "\u72ec\u7acb\u8463\u4e8b",
            "\u8ff0\u804c\u62a5\u544a",
            "\u5185\u90e8\u63a7\u5236",
            "\u516c\u53f8\u7ae0\u7a0b",
            "\u4f1a\u8bae\u8d44\u6599",
            "\u5ba1\u8ba1\u62a5\u544a",
            "\u5236\u5ea6",
        ),
        base_weight=0.55,
        horizon_days=1,
    ),
)

GENERIC_POSITIVE_KEYWORDS: Mapping[str, float] = {
    "bull": 1.0,
    "good": 0.8,
    "positive": 0.8,
    "growth": 0.8,
    "beat": 1.2,
    "\u5229\u597d": 1.0,
    "\u6539\u5584": 0.8,
    "\u589e\u957f": 0.8,
    "\u521b\u65b0\u9ad8": 1.0,
    "\u653e\u91cf": 0.6,
}
GENERIC_NEGATIVE_KEYWORDS: Mapping[str, float] = {
    "bear": -1.0,
    "bad": -0.8,
    "negative": -0.8,
    "risk": -0.8,
    "warning": -1.0,
    "\u5229\u7a7a": -1.0,
    "\u98ce\u9669": -0.8,
    "\u4e0b\u6ed1": -0.9,
    "\u66b4\u8dcc": -1.2,
    "\u8dcc\u505c": -1.3,
}
SOURCE_RELIABILITY_ALIASES: Mapping[str, float] = {
    "cninfo": 1.08,
    "disclosure": 1.08,
    "\u5de8\u6f6e": 1.08,
    "\u8bc1\u5238\u65f6\u62a5": 1.10,
    "\u4e2d\u56fd\u8bc1\u5238\u62a5": 1.08,
    "\u4e0a\u6d77\u8bc1\u5238\u62a5": 1.06,
    "\u8d22\u8054\u793e": 1.05,
    "\u4e1c\u65b9\u8d22\u5bcc": 1.02,
}

NEWS_RESEARCH_PRIOR_VERSION = "a_share_news_impact_1083_symbols_20260528"
NEWS_RESEARCH_NEUTRAL_BENCHMARK = {
    "mean_return_1d_pct": 0.3537,
    "mean_return_3d_pct": 0.3047,
    "mean_return_5d_pct": 0.3476,
}
NEWS_CATEGORY_RESEARCH_PRIORS: Mapping[str, Mapping[str, float | int]] = {
    "product_technology": {
        "event_count": 519,
        "mean_return_1d_pct": 3.6117,
        "mean_return_3d_pct": 2.5254,
        "mean_return_5d_pct": 2.8712,
        "excess_return_1d_pct": 3.2580,
        "excess_return_3d_pct": 2.2207,
        "excess_return_5d_pct": 2.5236,
        "positive_return_rate_1d": 0.7360,
    },
    "financing_mna": {
        "event_count": 212,
        "mean_return_1d_pct": 0.6060,
        "mean_return_3d_pct": 2.6226,
        "mean_return_5d_pct": 2.7946,
        "excess_return_1d_pct": 0.2523,
        "excess_return_3d_pct": 2.3178,
        "excess_return_5d_pct": 2.4470,
        "positive_return_rate_1d": 0.4858,
    },
    "accident_risk": {
        "event_count": 175,
        "mean_return_1d_pct": 0.5236,
        "mean_return_3d_pct": -1.0412,
        "mean_return_5d_pct": -0.8658,
        "excess_return_1d_pct": 0.1699,
        "excess_return_3d_pct": -1.3459,
        "excess_return_5d_pct": -1.2134,
        "positive_return_rate_1d": 0.4743,
    },
    "routine_governance": {
        "event_count": 1728,
        "mean_return_1d_pct": 0.3178,
        "mean_return_3d_pct": 0.4276,
        "mean_return_5d_pct": 0.1849,
        "excess_return_1d_pct": -0.0358,
        "excess_return_3d_pct": 0.1229,
        "excess_return_5d_pct": -0.1626,
        "positive_return_rate_1d": 0.4688,
    },
    "general": {
        "event_count": 6733,
        "mean_return_1d_pct": 0.2906,
        "mean_return_3d_pct": 0.1484,
        "mean_return_5d_pct": 0.2541,
        "excess_return_1d_pct": -0.0630,
        "excess_return_3d_pct": -0.1564,
        "excess_return_5d_pct": -0.0935,
        "positive_return_rate_1d": 0.4768,
    },
    "contract_order": {
        "event_count": 175,
        "mean_return_1d_pct": 0.1073,
        "mean_return_3d_pct": -0.7758,
        "mean_return_5d_pct": -1.3041,
        "excess_return_1d_pct": -0.2464,
        "excess_return_3d_pct": -1.0806,
        "excess_return_5d_pct": -1.6517,
        "positive_return_rate_1d": 0.4857,
    },
    "earnings": {
        "event_count": 1785,
        "mean_return_1d_pct": 0.0852,
        "mean_return_3d_pct": 0.6127,
        "mean_return_5d_pct": 1.0164,
        "excess_return_1d_pct": -0.2685,
        "excess_return_3d_pct": 0.3080,
        "excess_return_5d_pct": 0.6688,
        "positive_return_rate_1d": 0.4796,
    },
    "regulatory_risk": {
        "event_count": 763,
        "mean_return_1d_pct": -0.0561,
        "mean_return_3d_pct": -0.8486,
        "mean_return_5d_pct": -0.6992,
        "excess_return_1d_pct": -0.4098,
        "excess_return_3d_pct": -1.1533,
        "excess_return_5d_pct": -1.0467,
        "positive_return_rate_1d": 0.4325,
    },
    "dividend_distribution": {
        "event_count": 407,
        "mean_return_1d_pct": -0.1781,
        "mean_return_3d_pct": -0.9197,
        "mean_return_5d_pct": -1.2415,
        "excess_return_1d_pct": -0.5318,
        "excess_return_3d_pct": -1.2245,
        "excess_return_5d_pct": -1.5891,
        "positive_return_rate_1d": 0.4226,
    },
    "policy_sector": {
        "event_count": 422,
        "mean_return_1d_pct": -0.2695,
        "mean_return_3d_pct": -1.2645,
        "mean_return_5d_pct": -1.9636,
        "excess_return_1d_pct": -0.6232,
        "excess_return_3d_pct": -1.5693,
        "excess_return_5d_pct": -2.3112,
        "positive_return_rate_1d": 0.4194,
    },
    "fund_flow_market_heat": {
        "event_count": 2227,
        "mean_return_1d_pct": -0.3610,
        "mean_return_3d_pct": -0.8399,
        "mean_return_5d_pct": -0.6361,
        "excess_return_1d_pct": -0.7147,
        "excess_return_3d_pct": -1.1446,
        "excess_return_5d_pct": -0.9837,
        "positive_return_rate_1d": 0.4149,
    },
    "shareholder_action": {
        "event_count": 505,
        "mean_return_1d_pct": -0.5505,
        "mean_return_3d_pct": -0.7668,
        "mean_return_5d_pct": -1.2414,
        "excess_return_1d_pct": -0.9042,
        "excess_return_3d_pct": -1.0715,
        "excess_return_5d_pct": -1.5889,
        "positive_return_rate_1d": 0.3822,
    },
    "market_opinion": {
        "event_count": 236,
        "mean_return_1d_pct": -0.8658,
        "mean_return_3d_pct": -1.4869,
        "mean_return_5d_pct": -2.7262,
        "excess_return_1d_pct": -1.2195,
        "excess_return_3d_pct": -1.7916,
        "excess_return_5d_pct": -3.0737,
        "positive_return_rate_1d": 0.3856,
    },
}
NEWS_CATEGORY_DIRECTION_RESEARCH_PRIORS: Mapping[tuple[str, str], Mapping[str, float | int]] = {
    ("product_technology", "bullish"): {
        "event_count": 518,
        "excess_return_1d_pct": 3.2648,
        "excess_return_3d_pct": 2.2314,
        "excess_return_5d_pct": 2.5301,
        "positive_return_rate_1d": 0.7355,
    },
    ("financing_mna", "bullish"): {
        "event_count": 142,
        "excess_return_1d_pct": 0.7026,
        "excess_return_3d_pct": 3.4676,
        "excess_return_5d_pct": 3.7230,
        "positive_return_rate_1d": 0.5282,
    },
    ("shareholder_action", "bullish"): {
        "event_count": 274,
        "excess_return_1d_pct": -0.0377,
        "excess_return_3d_pct": 0.1542,
        "excess_return_5d_pct": -0.1148,
        "positive_return_rate_1d": 0.4745,
    },
    ("earnings", "bullish"): {
        "event_count": 628,
        "excess_return_1d_pct": -0.0691,
        "excess_return_3d_pct": 0.3809,
        "excess_return_5d_pct": 0.5425,
        "positive_return_rate_1d": 0.5032,
    },
    ("regulatory_risk", "bearish"): {
        "event_count": 756,
        "excess_return_1d_pct": -0.3708,
        "excess_return_3d_pct": -1.0963,
        "excess_return_5d_pct": -0.9792,
        "positive_return_rate_1d": 0.4365,
    },
    ("fund_flow_market_heat", "neutral"): {
        "event_count": 2002,
        "excess_return_1d_pct": -0.6674,
        "excess_return_3d_pct": -1.0120,
        "excess_return_5d_pct": -0.8727,
        "positive_return_rate_1d": 0.4176,
    },
    ("fund_flow_market_heat", "bearish"): {
        "event_count": 185,
        "excess_return_1d_pct": -1.0067,
        "excess_return_3d_pct": -2.1804,
        "excess_return_5d_pct": -2.6054,
        "positive_return_rate_1d": 0.4000,
    },
    ("policy_sector", "bearish"): {
        "event_count": 226,
        "excess_return_1d_pct": -1.0240,
        "excess_return_3d_pct": -2.6729,
        "excess_return_5d_pct": -3.3927,
        "positive_return_rate_1d": 0.4115,
    },
    ("shareholder_action", "bearish"): {
        "event_count": 218,
        "excess_return_1d_pct": -1.8546,
        "excess_return_3d_pct": -2.7472,
        "excess_return_5d_pct": -3.7198,
        "positive_return_rate_1d": 0.2890,
    },
    ("market_opinion", "bullish"): {
        "event_count": 209,
        "excess_return_1d_pct": -1.1611,
        "excess_return_3d_pct": -1.3896,
        "excess_return_5d_pct": -2.8767,
        "positive_return_rate_1d": 0.3876,
    },
}


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(float(value), upper)))


def _clip_score(value: float) -> float:
    return float(max(0.0, min(float(value), 100.0)))


def _dedupe(items: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = str(item).strip()
        if not key:
            continue
        folded = key.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        result.append(key)
    return tuple(result)


def _first_existing_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    if frame.empty and len(frame.columns) == 0:
        return None
    lookup = {str(column).strip().casefold(): column for column in frame.columns}
    for candidate in candidates:
        key = str(candidate).strip().casefold()
        if key in lookup:
            return lookup[key]
    return None


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def _keyword_hits(text: str, keywords: Sequence[str]) -> tuple[int, tuple[str, ...]]:
    folded = text.casefold()
    hits: list[str] = []
    for keyword in _dedupe(keywords):
        if keyword.casefold() in folded:
            hits.append(keyword)
    return len(hits), tuple(hits)


def _weighted_keyword_score(text: str, mapping: Mapping[str, float]) -> tuple[float, int, tuple[str, ...]]:
    folded = text.casefold()
    score = 0.0
    hits: list[str] = []
    for keyword, weight in mapping.items():
        if str(keyword).casefold() in folded:
            score += float(weight)
            hits.append(str(keyword))
    return score, len(hits), tuple(hits)


def _source_weight(source: object) -> float:
    text = _safe_text(source)
    if not text:
        return 1.0
    if text in SOURCE_WEIGHTS:
        return float(SOURCE_WEIGHTS[text])
    folded = text.casefold()
    for token, weight in SOURCE_RELIABILITY_ALIASES.items():
        if str(token).casefold() in folded:
            return float(weight)
    return 1.0


def _session_bucket(published_at: object) -> str:
    timestamp = pd.to_datetime(published_at, errors="coerce")
    if pd.isna(timestamp):
        return "unknown"
    clock = timestamp.time()
    if dt.time(9, 30) <= clock <= dt.time(11, 30):
        return "morning_session"
    if dt.time(11, 30) < clock < dt.time(13, 0):
        return "lunch_break"
    if dt.time(13, 0) <= clock <= dt.time(15, 0):
        return "afternoon_session"
    if dt.time(15, 0) < clock <= dt.time(23, 59, 59):
        return "after_close"
    return "pre_open"


def _direction_from_sentiment(sentiment: float) -> str:
    if sentiment >= 0.12:
        return "bullish"
    if sentiment <= -0.12:
        return "bearish"
    return "neutral"


def normalize_news_frame(news_df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    if not isinstance(news_df, pd.DataFrame) or news_df.empty:
        return pd.DataFrame(columns=NEWS_CANONICAL_COLUMNS)

    frame = news_df.copy()
    normalized = pd.DataFrame(index=frame.index)

    column_map = {
        "title": TITLE_COLUMNS,
        "content": BODY_COLUMNS,
        "published_at": TIME_COLUMNS,
        "source": SOURCE_COLUMNS,
        "url": URL_COLUMNS,
        "keyword": KEYWORD_COLUMNS,
        "symbol": SYMBOL_COLUMNS,
    }
    for target, candidates in column_map.items():
        column = _first_existing_column(frame, _dedupe(candidates))
        if column is not None:
            normalized[target] = frame[column]
        elif target == "symbol" and symbol is not None:
            normalized[target] = normalize_symbol(symbol)
        elif target == "source":
            normalized[target] = "unknown"
        elif target == "published_at":
            normalized[target] = pd.NaT
        else:
            normalized[target] = ""

    if symbol is not None:
        normalized["symbol"] = normalize_symbol(symbol)
    else:
        normalized["symbol"] = normalized["symbol"].map(lambda value: data_module.try_normalize_symbol(value) or "")
    normalized["title"] = normalized["title"].map(_safe_text)
    normalized["content"] = normalized["content"].map(_safe_text)
    normalized["source"] = normalized["source"].map(lambda value: _safe_text(value) or "unknown")
    normalized["url"] = normalized["url"].map(_safe_text)
    normalized["keyword"] = normalized["keyword"].map(_safe_text)
    normalized["published_at"] = pd.to_datetime(normalized["published_at"], errors="coerce")
    normalized = normalized.loc[(normalized["title"] != "") | (normalized["content"] != "")].copy()
    if normalized.empty:
        return pd.DataFrame(columns=NEWS_CANONICAL_COLUMNS)
    normalized = normalized.drop_duplicates(subset=["symbol", "title", "published_at", "source"], keep="first")
    normalized = normalized.sort_values("published_at", ascending=False, na_position="last").reset_index(drop=True)
    return normalized.reindex(columns=NEWS_CANONICAL_COLUMNS)


def classify_news_event(row: pd.Series | Mapping[str, object]) -> dict[str, object]:
    title = _safe_text(row.get("title", "") if isinstance(row, Mapping) else row.get("title", ""))
    content = _safe_text(row.get("content", "") if isinstance(row, Mapping) else row.get("content", ""))
    keyword = _safe_text(row.get("keyword", "") if isinstance(row, Mapping) else row.get("keyword", ""))
    source = row.get("source", "unknown") if isinstance(row, Mapping) else row.get("source", "unknown")
    text = f"{title} {content} {keyword}"

    best_rule: NewsEventRule | None = None
    best_score = 0.0
    best_strength = -1.0
    best_evidence = 0
    category_scores: list[float] = []
    positive_hits = 0
    negative_hits = 0
    neutral_hits = 0
    matched_keywords: list[str] = []

    for rule in EVENT_RULES:
        pos_count, pos_matches = _keyword_hits(text, rule.positive_keywords)
        neg_count, neg_matches = _keyword_hits(text, rule.negative_keywords)
        neutral_count, neutral_matches = _keyword_hits(text, rule.neutral_keywords)
        raw = (pos_count - neg_count) * rule.base_weight
        evidence = pos_count + neg_count + neutral_count
        if evidence:
            category_scores.append(raw)
            positive_hits += pos_count
            negative_hits += neg_count
            neutral_hits += neutral_count
            matched_keywords.extend(pos_matches)
            matched_keywords.extend(neg_matches)
            matched_keywords.extend(neutral_matches)
            strength = abs(raw) if raw else min(neutral_count * rule.base_weight * 0.34, 0.95)
            if (
                best_rule is None
                or strength > best_strength
                or (strength == best_strength and evidence > best_evidence)
            ):
                best_rule = rule
                best_score = raw
                best_strength = strength
                best_evidence = evidence

    imported_positive = {str(key): float(value) for key, value in BULLISH_KEYWORDS.items()}
    imported_negative = {str(key): float(value) for key, value in BEARISH_KEYWORDS.items()}
    generic_positive = {**GENERIC_POSITIVE_KEYWORDS, **imported_positive}
    generic_negative = {**GENERIC_NEGATIVE_KEYWORDS, **imported_negative}
    positive_score, extra_pos_hits, pos_extra = _weighted_keyword_score(text, generic_positive)
    negative_score, extra_neg_hits, neg_extra = _weighted_keyword_score(text, generic_negative)
    positive_hits += extra_pos_hits
    negative_hits += extra_neg_hits
    matched_keywords.extend(pos_extra)
    matched_keywords.extend(neg_extra)

    evidence_count = positive_hits + negative_hits + neutral_hits
    source_weight = _source_weight(source)
    category_score = float(np.sum(category_scores)) * 0.7 if category_scores else 0.0
    raw_score = (category_score + positive_score + negative_score) * source_weight
    sentiment = float(np.tanh(raw_score / 4.8)) if evidence_count else 0.0
    body_bonus = 0.04 if content else 0.0
    confidence = _clip(
        0.16
        + min(evidence_count * 0.075, 0.42)
        + min(abs(raw_score) * 0.035, 0.22)
        + min(abs(source_weight - 1.0), 0.08)
        + body_bonus,
        0.08,
        0.98,
    )
    expected_impact_score = _clip_score(50.0 + sentiment * (28.0 + confidence * 22.0))
    event_strength = _clip(abs(sentiment) * (0.5 + confidence * 0.5), 0.0, 1.0)

    if best_rule is None:
        best_rule = NewsEventRule(
            category="general",
            label="general news",
            positive_keywords=(),
            negative_keywords=(),
            base_weight=0.7,
            horizon_days=3,
        )

    published_at = row.get("published_at") if isinstance(row, Mapping) else row.get("published_at")
    dedupe_parts = [
        _safe_text(row.get("symbol", "") if isinstance(row, Mapping) else row.get("symbol", "")),
        title.casefold(),
        str(pd.to_datetime(published_at, errors="coerce")),
        _safe_text(source).casefold(),
    ]
    return {
        "event_category": best_rule.category,
        "event_label": best_rule.label,
        "event_direction": _direction_from_sentiment(sentiment),
        "event_sentiment": round(sentiment, 4),
        "event_confidence": round(confidence, 4),
        "expected_impact_score": round(expected_impact_score, 2),
        "event_strength": round(event_strength, 4),
        "positive_hits": int(positive_hits),
        "negative_hits": int(negative_hits),
        "neutral_hits": int(neutral_hits),
        "matched_keywords": "|".join(_dedupe(matched_keywords)),
        "session_bucket": _session_bucket(published_at),
        "impact_horizon_days": int(best_rule.horizon_days),
        "source_weight": round(float(source_weight), 4),
        "dedupe_key": "|".join(dedupe_parts),
    }


def classify_news_events(news_df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    normalized = normalize_news_frame(news_df, symbol=symbol)
    if normalized.empty:
        return pd.DataFrame(columns=CLASSIFIED_NEWS_COLUMNS)
    classified_rows = [classify_news_event(row) for _, row in normalized.iterrows()]
    classified = pd.concat([normalized.reset_index(drop=True), pd.DataFrame(classified_rows)], axis=1)
    classified = classified.drop_duplicates(subset=["dedupe_key"], keep="first")
    return classified.reindex(columns=CLASSIFIED_NEWS_COLUMNS).reset_index(drop=True)


def _normalize_disclosure_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=NEWS_CANONICAL_COLUMNS)
    view = frame.copy()
    view["source"] = "cninfo_disclosure"
    if "symbol" not in view.columns:
        view["symbol"] = symbol
    return normalize_news_frame(view, symbol=symbol)


def fetch_symbol_news_events(
    symbol: str,
    *,
    limit: int = 80,
    include_disclosures: bool = True,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    clean_symbol = normalize_symbol(symbol)
    frames: list[pd.DataFrame] = []
    try:
        stock_news = fetch_stock_news(clean_symbol, limit=limit)
        if not stock_news.empty:
            frames.append(normalize_news_frame(stock_news, symbol=clean_symbol))
    except Exception:
        pass

    if include_disclosures:
        end = end_date or dt.date.today().strftime("%Y%m%d")
        start = start_date or (pd.to_datetime(end, format="%Y%m%d", errors="coerce") - pd.Timedelta(days=180)).strftime(
            "%Y%m%d"
        )
        try:
            data_module.ensure_akshare()
            disclosure_frame = data_module.ak.stock_zh_a_disclosure_report_cninfo(
                symbol=clean_symbol,
                market="\u6caa\u6df1\u4eac",
                start_date=start,
                end_date=end,
            )
            if not disclosure_frame.empty:
                frames.append(_normalize_disclosure_frame(disclosure_frame, clean_symbol))
        except Exception:
            pass

    if not frames:
        return pd.DataFrame(columns=CLASSIFIED_NEWS_COLUMNS)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values("published_at", ascending=False, na_position="last").head(limit).reset_index(drop=True)
    return classify_news_events(merged, symbol=clean_symbol)


def fetch_market_news_events(limit: int = 80) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    try:
        data_module.ensure_akshare()
    except Exception:
        return pd.DataFrame(columns=CLASSIFIED_NEWS_COLUMNS)

    fetchers = (
        ("cls_global", lambda: data_module.ak.stock_info_global_cls(symbol="\u5168\u90e8")),
        ("ths_global", lambda: data_module.ak.stock_info_global_ths()),
        ("futu_global", lambda: data_module.ak.stock_info_global_futu()),
    )
    for source, fetcher in fetchers:
        try:
            frame = fetcher()
        except Exception:
            continue
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            view = frame.copy()
            if _first_existing_column(view, SOURCE_COLUMNS) is None:
                view["source"] = source
            frames.append(normalize_news_frame(view))

    if not frames:
        return pd.DataFrame(columns=CLASSIFIED_NEWS_COLUMNS)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values("published_at", ascending=False, na_position="last").head(limit).reset_index(drop=True)
    return classify_news_events(merged)


def _normalize_price_frame(daily_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(daily_df, pd.DataFrame) or daily_df.empty:
        return pd.DataFrame(columns=["date", "open", "close", "high", "low", "volume"])
    frame = daily_df.copy()
    if "date" in frame.columns:
        frame = frame.reset_index(drop=True)
    else:
        index_name = frame.index.name or "index"
        frame = frame.reset_index().rename(columns={index_name: "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    for column in ("open", "close", "high", "low", "volume", "amount", "turnover"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "open" not in frame.columns:
        frame["open"] = np.nan
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")
    frame["open"] = frame["open"].fillna(frame["close"])
    return frame.reset_index(drop=True)


def _resolve_impact_position(published_at: object, trade_dates: pd.Series) -> int | None:
    timestamp = pd.to_datetime(published_at, errors="coerce")
    if pd.isna(timestamp) or trade_dates.empty:
        return None
    event_date = timestamp.normalize()
    after_close = timestamp.time() > dt.time(15, 0)
    if after_close:
        candidates = trade_dates[trade_dates > event_date]
    else:
        candidates = trade_dates[trade_dates >= event_date]
    if candidates.empty:
        return None
    impact_date = candidates.iloc[0]
    positions = trade_dates.index[trade_dates == impact_date].tolist()
    return int(positions[0]) if positions else None


def _direction_hit(direction: str, actual_return: float | int | None) -> float:
    if actual_return is None or pd.isna(actual_return):
        return float("nan")
    if direction == "bullish":
        return 1.0 if float(actual_return) > 0 else 0.0
    if direction == "bearish":
        return 1.0 if float(actual_return) < 0 else 0.0
    return float("nan")


def build_event_impact_dataset(
    news_events: pd.DataFrame,
    daily_prices: pd.DataFrame,
    *,
    horizons: Sequence[int] = (1, 3, 5),
) -> pd.DataFrame:
    if not isinstance(news_events, pd.DataFrame) or news_events.empty:
        return pd.DataFrame(columns=IMPACT_COLUMNS)
    prices = _normalize_price_frame(daily_prices)
    if prices.empty:
        return pd.DataFrame(columns=IMPACT_COLUMNS)
    events = news_events.copy()
    if "event_category" not in events.columns:
        events = classify_news_events(events)
    if events.empty:
        return pd.DataFrame(columns=IMPACT_COLUMNS)

    trade_dates = prices["date"]
    rows: list[dict[str, object]] = []
    sorted_horizons = tuple(sorted({int(horizon) for horizon in horizons if int(horizon) >= 1}))
    for _, event in events.iterrows():
        impact_pos = _resolve_impact_position(event.get("published_at"), trade_dates)
        if impact_pos is None or impact_pos <= 0:
            continue
        baseline_pos = impact_pos - 1
        baseline_close = float(prices.iloc[baseline_pos]["close"])
        if baseline_close <= 0 or math.isnan(baseline_close):
            continue
        impact_open = float(prices.iloc[impact_pos]["open"])
        impact_close = float(prices.iloc[impact_pos]["close"])
        output = event.to_dict()
        output.update(
            {
                "impact_trade_date": prices.iloc[impact_pos]["date"],
                "baseline_trade_date": prices.iloc[baseline_pos]["date"],
                "baseline_close": baseline_close,
                "impact_open": impact_open,
                "impact_close": impact_close,
                "open_gap_pct": (impact_open / baseline_close - 1.0) * 100.0,
                "same_day_return_pct": (impact_close / baseline_close - 1.0) * 100.0,
            }
        )
        direction = str(event.get("event_direction", "neutral"))
        for horizon in sorted_horizons:
            target_pos = impact_pos + horizon - 1
            return_column = f"return_{horizon}d_pct"
            hit_column = f"direction_hit_{horizon}d"
            if target_pos >= len(prices):
                output[return_column] = np.nan
                output[hit_column] = np.nan
                continue
            target_close = float(prices.iloc[target_pos]["close"])
            horizon_return = (target_close / baseline_close - 1.0) * 100.0
            output[return_column] = horizon_return
            output[hit_column] = _direction_hit(direction, horizon_return)
        rows.append(output)

    if not rows:
        dynamic_columns = [f"return_{horizon}d_pct" for horizon in sorted_horizons]
        return pd.DataFrame(columns=[*IMPACT_COLUMNS, *dynamic_columns])
    return pd.DataFrame(rows).sort_values("published_at", ascending=False, na_position="last").reset_index(drop=True)


def summarize_category_impact(
    impact_df: pd.DataFrame,
    *,
    horizons: Sequence[int] = (1, 3, 5),
    min_events: int = 1,
) -> pd.DataFrame:
    if not isinstance(impact_df, pd.DataFrame) or impact_df.empty:
        return pd.DataFrame()

    frame = impact_df.copy()
    if "event_category" not in frame.columns:
        return pd.DataFrame()
    sorted_horizons = tuple(sorted({int(horizon) for horizon in horizons if int(horizon) >= 1}))
    rows: list[dict[str, object]] = []
    for category, group in frame.groupby("event_category", dropna=False):
        event_count = int(len(group))
        if event_count < int(min_events):
            continue
        row: dict[str, object] = {
            "event_category": str(category),
            "event_count": event_count,
            "bullish_count": int((group.get("event_direction") == "bullish").sum()),
            "bearish_count": int((group.get("event_direction") == "bearish").sum()),
            "neutral_count": int((group.get("event_direction") == "neutral").sum()),
            "avg_sentiment": round(float(pd.to_numeric(group.get("event_sentiment"), errors="coerce").mean()), 4),
            "avg_confidence": round(float(pd.to_numeric(group.get("event_confidence"), errors="coerce").mean()), 4),
            "avg_expected_impact_score": round(
                float(pd.to_numeric(group.get("expected_impact_score"), errors="coerce").mean()), 2
            ),
            "avg_open_gap_pct": round(float(pd.to_numeric(group.get("open_gap_pct"), errors="coerce").mean()), 4),
        }
        first_horizon_score = 0.0
        first_hit_rate = 0.5
        for horizon in sorted_horizons:
            return_column = f"return_{horizon}d_pct"
            hit_column = f"direction_hit_{horizon}d"
            returns = pd.to_numeric(group.get(return_column), errors="coerce")
            if hit_column in group.columns:
                hits = pd.to_numeric(group[hit_column], errors="coerce")
            else:
                hits = group.apply(lambda item: _direction_hit(str(item.get("event_direction", "neutral")), item.get(return_column)), axis=1)
            row[f"avg_return_{horizon}d_pct"] = round(float(returns.mean()), 4) if returns.notna().any() else np.nan
            row[f"median_return_{horizon}d_pct"] = round(float(returns.median()), 4) if returns.notna().any() else np.nan
            row[f"positive_return_rate_{horizon}d"] = (
                round(float((returns > 0).mean()), 4) if returns.notna().any() else np.nan
            )
            row[f"direction_hit_rate_{horizon}d"] = round(float(hits.mean()), 4) if hits.notna().any() else np.nan
            if horizon == sorted_horizons[0] and returns.notna().any():
                first_horizon_score = abs(float(returns.mean()))
                first_hit_rate = float(hits.mean()) if hits.notna().any() else 0.5
        row["impact_rank_score"] = round(first_horizon_score * math.log1p(event_count) * (0.75 + first_hit_rate), 4)
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["impact_rank_score", "event_count"], ascending=False).reset_index(drop=True)


def get_news_research_prior(category: object, direction: object | None = None) -> dict[str, object]:
    category_key = _safe_text(category) or "general"
    direction_key = _safe_text(direction)
    category_prior = dict(NEWS_CATEGORY_RESEARCH_PRIORS.get(category_key, NEWS_CATEGORY_RESEARCH_PRIORS["general"]))
    direction_prior = NEWS_CATEGORY_DIRECTION_RESEARCH_PRIORS.get((category_key, direction_key))
    if direction_prior:
        merged = {**category_prior, **dict(direction_prior)}
        merged["prior_scope"] = "category_direction"
    else:
        merged = category_prior
        merged["prior_scope"] = "category"
    merged["event_category"] = category_key
    merged["event_direction"] = direction_key or ""
    merged["research_prior_version"] = NEWS_RESEARCH_PRIOR_VERSION
    return merged


def _research_prior_score(prior: Mapping[str, object]) -> tuple[float, float]:
    excess_1d = float(prior.get("excess_return_1d_pct", 0.0) or 0.0)
    excess_3d = float(prior.get("excess_return_3d_pct", excess_1d) or excess_1d)
    excess_5d = float(prior.get("excess_return_5d_pct", excess_3d) or excess_3d)
    blended_excess = excess_1d * 0.52 + excess_3d * 0.30 + excess_5d * 0.18
    score = _clip_score(50.0 + math.tanh(blended_excess / 2.8) * 38.0)
    return score, blended_excess


def score_news_event_with_research(row: pd.Series | Mapping[str, object]) -> dict[str, object]:
    category = row.get("event_category", "general") if isinstance(row, Mapping) else row.get("event_category", "general")
    direction = row.get("event_direction", "neutral") if isinstance(row, Mapping) else row.get("event_direction", "neutral")
    prior = get_news_research_prior(category, direction)
    research_score, blended_excess = _research_prior_score(prior)
    event_score = float(row.get("expected_impact_score", 50.0) if isinstance(row, Mapping) else row.get("expected_impact_score", 50.0) or 50.0)
    sample_count = int(prior.get("event_count", 0) or 0)
    sample_weight = _clip(math.log1p(max(sample_count, 0)) / math.log1p(2200), 0.20, 1.0)
    research_weight = _clip(0.52 + sample_weight * 0.28, 0.52, 0.82)
    adjusted_score = _clip_score(research_score * research_weight + event_score * (1.0 - research_weight))
    return {
        "research_prior_version": NEWS_RESEARCH_PRIOR_VERSION,
        "research_prior_scope": str(prior.get("prior_scope", "category")),
        "research_sample_count": sample_count,
        "research_score": round(research_score, 2),
        "research_adjusted_score": round(adjusted_score, 2),
        "research_confidence": round(_clip(0.30 + sample_weight * 0.55 + min(abs(blended_excess) / 5.0, 0.15)), 4),
        "research_blended_excess_pct": round(blended_excess, 4),
        "research_excess_return_1d_pct": round(float(prior.get("excess_return_1d_pct", 0.0) or 0.0), 4),
        "research_excess_return_3d_pct": round(float(prior.get("excess_return_3d_pct", 0.0) or 0.0), 4),
        "research_excess_return_5d_pct": round(float(prior.get("excess_return_5d_pct", 0.0) or 0.0), 4),
        "research_positive_return_rate_1d": round(float(prior.get("positive_return_rate_1d", 0.5) or 0.5), 4),
    }


def build_research_news_impact_signal(news_events: pd.DataFrame, *, window_days: int = 7) -> dict[str, object]:
    events = news_events.copy() if isinstance(news_events, pd.DataFrame) else pd.DataFrame()
    if events.empty:
        return {
            "score": 50.0,
            "label": "neutral",
            "confidence_score": 25.0,
            "summary": "No recent A-share news sample is available.",
            "event_count": 0,
            "expected_excess_return_1d_pct": 0.0,
            "expected_excess_return_3d_pct": 0.0,
            "expected_excess_return_5d_pct": 0.0,
            "top_categories": [],
            "research_prior_version": NEWS_RESEARCH_PRIOR_VERSION,
        }
    if "event_category" not in events.columns:
        events = classify_news_events(events)
    if events.empty:
        return build_research_news_impact_signal(pd.DataFrame(), window_days=window_days)
    events["published_at"] = pd.to_datetime(events.get("published_at"), errors="coerce")
    if events["published_at"].notna().any():
        anchor = events["published_at"].max()
        events = events.loc[events["published_at"] >= anchor - pd.Timedelta(days=int(window_days))].copy()
    if events.empty:
        return {
            "score": 50.0,
            "label": "neutral",
            "confidence_score": 25.0,
            "summary": "No news remained after the recency filter.",
            "event_count": 0,
            "expected_excess_return_1d_pct": 0.0,
            "expected_excess_return_3d_pct": 0.0,
            "expected_excess_return_5d_pct": 0.0,
            "top_categories": [],
            "research_prior_version": NEWS_RESEARCH_PRIOR_VERSION,
        }

    research_rows = [score_news_event_with_research(row) for _, row in events.iterrows()]
    research_frame = pd.DataFrame(research_rows, index=events.index)
    merged = pd.concat([events, research_frame], axis=1)
    event_confidence = pd.to_numeric(merged.get("event_confidence"), errors="coerce").fillna(0.20)
    research_confidence = pd.to_numeric(merged["research_confidence"], errors="coerce").fillna(0.30)
    if merged["published_at"].notna().any():
        age_days = (merged["published_at"].max() - merged["published_at"]).dt.total_seconds().div(86400).fillna(window_days)
        recency = np.exp(-age_days / max(float(window_days), 1.0))
    else:
        recency = pd.Series(1.0, index=merged.index)
    weights = np.clip((event_confidence * 0.35 + research_confidence * 0.65) * recency, 0.05, 1.0)
    score_series = pd.to_numeric(merged["research_adjusted_score"], errors="coerce").fillna(50.0)
    aggregate_score = float((score_series * weights).sum() / max(float(weights.sum()), 1e-9))
    excess_1d = float((pd.to_numeric(merged["research_excess_return_1d_pct"], errors="coerce").fillna(0.0) * weights).sum() / max(float(weights.sum()), 1e-9))
    excess_3d = float((pd.to_numeric(merged["research_excess_return_3d_pct"], errors="coerce").fillna(0.0) * weights).sum() / max(float(weights.sum()), 1e-9))
    excess_5d = float((pd.to_numeric(merged["research_excess_return_5d_pct"], errors="coerce").fillna(0.0) * weights).sum() / max(float(weights.sum()), 1e-9))
    confidence_score = _clip_score(25.0 + min(float(len(merged)) / 10.0, 1.0) * 18.0 + float(np.average(research_confidence, weights=weights)) * 57.0)
    if aggregate_score >= 58:
        label = "bullish"
    elif aggregate_score <= 42:
        label = "bearish"
    else:
        label = "neutral"
    category_frame = (
        merged.assign(weight=weights)
        .groupby("event_category")
        .agg(
            event_count=("event_category", "size"),
            avg_research_score=("research_adjusted_score", "mean"),
            expected_excess_return_1d_pct=("research_excess_return_1d_pct", "mean"),
            weight=("weight", "sum"),
        )
        .sort_values(["weight", "event_count"], ascending=False)
        .head(5)
        .reset_index()
    )
    top_categories = category_frame.drop(columns=["weight"], errors="ignore").to_dict(orient="records")
    primary = str(top_categories[0]["event_category"]) if top_categories else "general"
    return {
        "score": round(aggregate_score, 2),
        "label": label,
        "confidence_score": round(confidence_score, 2),
        "summary": (
            f"{len(merged)} recent events; research-adjusted score {aggregate_score:.1f}; "
            f"expected excess return {excess_1d:+.2f}%/1d, {excess_3d:+.2f}%/3d."
        ),
        "event_count": int(len(merged)),
        "primary_category": primary,
        "expected_excess_return_1d_pct": round(excess_1d, 4),
        "expected_excess_return_3d_pct": round(excess_3d, 4),
        "expected_excess_return_5d_pct": round(excess_5d, 4),
        "top_categories": top_categories,
        "research_prior_version": NEWS_RESEARCH_PRIOR_VERSION,
    }


def build_research_enhanced_news_signal(
    news_df: pd.DataFrame,
    *,
    base_signal: Mapping[str, object] | None = None,
    symbol: str | None = None,
    window_days: int = 7,
) -> dict[str, object]:
    base = dict(base_signal or {})
    if not base:
        base = {
            "sentiment_score": 50.0,
            "confidence_score": 25.0,
            "label": "news neutral",
            "summary": "No keyword sentiment signal supplied.",
        }
    try:
        events = classify_news_events(news_df, symbol=symbol) if isinstance(news_df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        events = pd.DataFrame()
    research = build_research_news_impact_signal(events, window_days=window_days)
    base_score = float(base.get("sentiment_score", 50.0) or 50.0)
    base_confidence = float(base.get("confidence_score", 25.0) or 25.0)
    research_score = float(research.get("score", 50.0) or 50.0)
    research_confidence = float(research.get("confidence_score", 25.0) or 25.0)
    if int(research.get("event_count", 0) or 0) <= 0:
        blended_score = base_score
        blended_confidence = base_confidence
    else:
        research_weight = _clip(0.45 + research_confidence / 100.0 * 0.30, 0.45, 0.76)
        blended_score = _clip_score(research_score * research_weight + base_score * (1.0 - research_weight))
        blended_confidence = _clip_score(max(base_confidence, research_confidence) * 0.74 + min(base_confidence, research_confidence) * 0.26)
    if blended_score >= 58:
        label = "bullish"
    elif blended_score <= 42:
        label = "bearish"
    else:
        label = "neutral"
    enhanced = {
        **base,
        "sentiment_score": round(blended_score, 2),
        "confidence_score": round(blended_confidence, 2),
        "label": label,
        "summary": str(research.get("summary") or base.get("summary") or ""),
        "keyword_sentiment_score": round(base_score, 2),
        "keyword_confidence_score": round(base_confidence, 2),
        "research_impact_score": round(research_score, 2),
        "research_impact_confidence_score": round(research_confidence, 2),
        "research_expected_excess_return_1d_pct": research.get("expected_excess_return_1d_pct", 0.0),
        "research_expected_excess_return_3d_pct": research.get("expected_excess_return_3d_pct", 0.0),
        "research_expected_excess_return_5d_pct": research.get("expected_excess_return_5d_pct", 0.0),
        "research_primary_category": research.get("primary_category", "general"),
        "research_event_count": int(research.get("event_count", 0) or 0),
        "research_prior_version": NEWS_RESEARCH_PRIOR_VERSION,
        "research_top_categories": research.get("top_categories", []),
    }
    return enhanced


def build_latest_news_impact_signal(news_events: pd.DataFrame, *, window_days: int = 7) -> dict[str, object]:
    events = news_events.copy() if isinstance(news_events, pd.DataFrame) else pd.DataFrame()
    if events.empty:
        return {
            "score": 50.0,
            "label": "neutral",
            "summary": "No recent A-share news sample is available.",
            "event_count": 0,
            "top_categories": [],
        }
    if "event_category" not in events.columns:
        events = classify_news_events(events)
    events["published_at"] = pd.to_datetime(events["published_at"], errors="coerce")
    if events["published_at"].notna().any():
        anchor = events["published_at"].max()
        events = events.loc[events["published_at"] >= anchor - pd.Timedelta(days=int(window_days))].copy()
    if events.empty:
        return {
            "score": 50.0,
            "label": "neutral",
            "summary": "No news remained after the recency filter.",
            "event_count": 0,
            "top_categories": [],
        }

    sentiment = pd.to_numeric(events["event_sentiment"], errors="coerce").fillna(0.0)
    confidence = pd.to_numeric(events["event_confidence"], errors="coerce").fillna(0.2)
    if events["published_at"].notna().any():
        age_days = (events["published_at"].max() - events["published_at"]).dt.total_seconds().div(86400).fillna(window_days)
        recency = np.exp(-age_days / max(float(window_days), 1.0))
    else:
        recency = pd.Series(1.0, index=events.index)
    weights = np.clip(confidence * recency, 0.05, 1.0)
    aggregate_sentiment = float((sentiment * weights).sum() / max(float(weights.sum()), 1e-9))
    score = _clip_score(50.0 + aggregate_sentiment * 45.0)
    if score >= 62:
        label = "bullish"
    elif score <= 38:
        label = "bearish"
    else:
        label = "neutral"
    category_frame = (
        events.groupby("event_category")
        .agg(
            event_count=("event_category", "size"),
            avg_sentiment=("event_sentiment", "mean"),
            avg_expected_impact_score=("expected_impact_score", "mean"),
        )
        .sort_values(["event_count", "avg_expected_impact_score"], ascending=False)
        .head(5)
        .reset_index()
    )
    top_categories = category_frame.to_dict(orient="records")
    research_signal = build_research_news_impact_signal(events, window_days=window_days)
    return {
        "score": round(score, 2),
        "label": label,
        "summary": f"{len(events)} recent events, weighted sentiment {aggregate_sentiment:.2f}.",
        "event_count": int(len(events)),
        "top_categories": top_categories,
        "research_score": research_signal["score"],
        "research_label": research_signal["label"],
        "research_confidence_score": research_signal["confidence_score"],
        "research_expected_excess_return_1d_pct": research_signal["expected_excess_return_1d_pct"],
        "research_expected_excess_return_3d_pct": research_signal["expected_excess_return_3d_pct"],
        "research_expected_excess_return_5d_pct": research_signal["expected_excess_return_5d_pct"],
        "research_prior_version": NEWS_RESEARCH_PRIOR_VERSION,
    }


def analyze_symbol_news_impact(
    symbol: str,
    *,
    news_df: pd.DataFrame | None = None,
    daily_df: pd.DataFrame | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    news_limit: int = 120,
    horizons: Sequence[int] = (1, 3, 5),
    include_disclosures: bool = True,
) -> dict[str, object]:
    clean_symbol = normalize_symbol(symbol)
    final_end = end_date or dt.date.today().strftime("%Y%m%d")
    if news_df is None:
        events = fetch_symbol_news_events(
            clean_symbol,
            limit=news_limit,
            include_disclosures=include_disclosures,
            start_date=start_date,
            end_date=final_end,
        )
    else:
        events = classify_news_events(news_df, symbol=clean_symbol)
    if daily_df is None:
        if start_date is None and not events.empty and pd.to_datetime(events["published_at"], errors="coerce").notna().any():
            first_news_date = pd.to_datetime(events["published_at"], errors="coerce").min() - pd.Timedelta(days=20)
            resolved_start = first_news_date.strftime("%Y%m%d")
        else:
            resolved_start = start_date or (dt.date.today() - dt.timedelta(days=420)).strftime("%Y%m%d")
        prices = fetch_daily_history(clean_symbol, start_date=resolved_start, end_date=final_end)
    else:
        prices = daily_df.copy()

    impact = build_event_impact_dataset(events, prices, horizons=horizons)
    summary = summarize_category_impact(impact, horizons=horizons)
    latest_signal = build_latest_news_impact_signal(events)
    return {
        "symbol": clean_symbol,
        "events": events,
        "event_impacts": impact,
        "category_summary": summary,
        "latest_signal": latest_signal,
        "event_count": int(len(events)),
        "impact_sample_count": int(len(impact)),
        "horizons": tuple(int(horizon) for horizon in horizons),
    }


def _records(frame: pd.DataFrame, *, limit: int | None = None) -> list[dict[str, object]]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    view = frame.head(limit).copy() if limit is not None else frame.copy()
    return json.loads(view.to_json(orient="records", date_format="iso", force_ascii=False))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze A-share news categories and following price impact.")
    parser.add_argument("symbol", help="Six-digit A-share symbol, for example 000001.")
    parser.add_argument("--start-date", default=None, help="Price/news start date in YYYYMMDD.")
    parser.add_argument("--end-date", default=None, help="Price/news end date in YYYYMMDD.")
    parser.add_argument("--limit", type=int, default=120, help="Maximum news events to fetch.")
    parser.add_argument("--horizons", default="1,3,5", help="Comma-separated holding horizons in trading days.")
    parser.add_argument("--no-disclosures", action="store_true", help="Skip CNInfo disclosure fetches.")
    parser.add_argument("--output-dir", default="", help="Optional directory for event and summary CSV outputs.")
    args = parser.parse_args(argv)

    horizons = tuple(int(part.strip()) for part in args.horizons.split(",") if part.strip())
    result = analyze_symbol_news_impact(
        args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        news_limit=args.limit,
        horizons=horizons,
        include_disclosures=not args.no_disclosures,
    )
    payload = {
        "symbol": result["symbol"],
        "event_count": result["event_count"],
        "impact_sample_count": result["impact_sample_count"],
        "horizons": list(result["horizons"]),
        "latest_signal": result["latest_signal"],
        "category_summary": _records(result["category_summary"]),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result["events"].to_csv(output_dir / f"{result['symbol']}_news_events.csv", index=False, encoding="utf-8-sig")
        result["event_impacts"].to_csv(
            output_dir / f"{result['symbol']}_news_event_impacts.csv", index=False, encoding="utf-8-sig"
        )
        result["category_summary"].to_csv(
            output_dir / f"{result['symbol']}_news_category_summary.csv", index=False, encoding="utf-8-sig"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
