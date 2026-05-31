from __future__ import annotations

import datetime as dt
import hashlib
import inspect
import pickle
import re
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from functools import lru_cache
from html import escape
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

from .database_source import load_env_file

try:
    import akshare as ak
except ImportError:  # pragma: no cover
    ak = None

from .backtesting import run_daily_strategy_backtest
from .data import (
    FALLBACK_WATCHLIST,
    clear_daily_history_cache,
    fetch_a_share_universe,
    fetch_concept_fund_flow,
    fetch_daily_history,
    fetch_industry_fund_flow,
    fetch_macro_calendar,
    fetch_market_spot,
    fetch_minute_history,
    fetch_stock_main_fund_flow,
    fetch_stock_news,
    fetch_stock_profile,
    fetch_tushare_daily_snapshot,
    fetch_tushare_daily_window,
    fetch_tushare_recent_trade_dates,
    fetch_tushare_stock_basic,
    market_clock,
    normalize_search_text,
    parse_watchlist,
    search_a_share_universe,
    try_normalize_symbol,
)
from .daily_review import (
    compute_adaptive_rank_score,
    compute_replay_calibrated_scores,
    load_daily_lightweight_backtest_model,
    load_adaptive_rank_profile,
    load_review_battle_panels,
    load_latest_review_details,
    load_latest_snapshot_board,
    load_latest_review_summary,
    persist_focus_board_snapshot,
    run_daily_review_maintenance,
)
from .features import build_daily_features, evaluate_intraday, latest_snapshot
from .market_backtest_runner import load_latest_full_market_backtest, run_full_market_backtest
from .modeling import (
    GLOBAL_MODEL_TEST_END,
    GLOBAL_MODEL_TEST_START,
    GLOBAL_MODEL_TRAIN_END,
    GLOBAL_MODEL_TRAIN_START,
    MODEL_SCHEMA_VERSION,
    apply_live_probability_upgrade,
    apply_sector_fund_probability_upgrade,
    build_sector_fund_probability_upgrade,
    build_live_probability_upgrade,
    clear_market_wide_model_cache,
    explain_latest_model_state,
    get_market_wide_model_status,
    load_cached_market_wide_model,
    load_market_proxy_model,
    load_market_wide_model,
    predict_latest_probability,
    score_with_market_proxy_model,
    score_with_market_wide_model,
    train_probability_model,
)
from .quant import (
    compute_sector_hot_score,
    evaluate_main_fund_signal,
    evaluate_news_sentiment,
    evaluate_quant_signal,
)
from .news_impact import build_research_enhanced_news_signal
from .stages import build_tomorrow_plan, classify_stage, main_rise_start_score, stage_numeric_score
from .strategy import (
    assess_execution_readiness,
    assess_launch_window,
    build_strategy_workbench,
    build_trading_rule_context,
    evaluate_intraday_structure_signal,
    evaluate_temporal_news_pulse,
)
from .store import (
    align_daily_history_to_market_date as store_align_daily_history_to_market_date,
    build_market_candidate_pool_store,
    get_market_daily_feature_row,
    build_market_daily_feature_store,
    build_market_dynamic_fallback_pool_store,
    load_incremental_market_snapshot_history,
    read_market_snapshot_history_store,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / ".cache"
FULL_MARKET_HISTORY_START = "20240701"
# AkShare relies on py_mini_racer for several A-share endpoints. When we fan
# out many daily-history requests in parallel, the embedded runtime can crash
# the whole refresh task before the ranking cache is written. Keep the market
# ranking scan stable and let the UI-level async wrapper provide responsiveness.
FULL_MARKET_MAX_WORKERS = 1
DEFAULT_SELECTION_LAUNCH_WINDOW_CONFIDENCE_WEIGHT = 0.0
DEFAULT_ACTION_EXECUTION_WEIGHT = 0.0


def _selection_launch_window_confidence_weight(payload: dict[str, object] | pd.Series | None) -> float:
    if payload is None:
        return DEFAULT_SELECTION_LAUNCH_WINDOW_CONFIDENCE_WEIGHT
    raw_value = None
    if isinstance(payload, pd.Series):
        raw_value = payload.get("launch_window_confidence_weight")
    elif isinstance(payload, dict):
        raw_value = payload.get("launch_window_confidence_weight")
    weight = _safe_float(raw_value, DEFAULT_SELECTION_LAUNCH_WINDOW_CONFIDENCE_WEIGHT)
    return float(max(0.0, min(weight, 1.0)))


def _action_execution_weight(payload: dict[str, object] | pd.Series | None) -> float:
    if payload is None:
        return DEFAULT_ACTION_EXECUTION_WEIGHT
    raw_value = None
    if isinstance(payload, pd.Series):
        raw_value = payload.get("action_execution_weight")
    elif isinstance(payload, dict):
        raw_value = payload.get("action_execution_weight")
    weight = _safe_float(raw_value, DEFAULT_ACTION_EXECUTION_WEIGHT)
    return float(max(0.0, min(weight, 1.0)))
MARKET_RANKING_CACHE_VERSION = 14
MARKET_CANDIDATE_ANALYSIS_CACHE_VERSION = 2
DYNAMIC_FALLBACK_CACHE_VERSION = 4
STRATEGY_CANDIDATE_CACHE_VERSION = 2
DYNAMIC_FALLBACK_POOL_SIZE = 50
DYNAMIC_FALLBACK_ANALYSIS_BUFFER_SIZE = 80
RULE_BASED_CANDIDATE_POOL_SIZE = 120
FALLBACK_CANDIDATE_SYMBOLS = (
    "600519",
    "000333",
    "002594",
    "600036",
    "000858",
    "601318",
    "300750",
    "601899",
    "600900",
    "002475",
    "600276",
    "601012",
    "600030",
    "600309",
    "002415",
    "601888",
    "000001",
    "000651",
    "600031",
    "600887",
    "601166",
    "601398",
    "601988",
    "601816",
    "000568",
    "300059",
    "002230",
    "600690",
    "601633",
    "601225",
    "600660",
    "603259",
    "600809",
    "603288",
    "601111",
    "600585",
    "601601",
    "600426",
    "600406",
    "601668",
    "600104",
    "601669",
    "600438",
    "603501",
    "002371",
    "002027",
    "002714",
    "300308",
    "688981",
    "601728",
)
ASYNC_UI_EXECUTOR = ThreadPoolExecutor(max_workers=4)
MARKET_CONTEXT_PREFETCH_EXECUTOR = ThreadPoolExecutor(max_workers=1)
ASYNC_UI_FUTURES: dict[str, Future] = {}
ASYNC_UI_PROGRESS: dict[str, dict[str, object]] = {}
ASYNC_UI_PROGRESS_LOCK = Lock()
DEFAULT_VIEW_PARAMS = {
    "refresh_seconds": 0,
    "ranking_by": "关注分数",
    "board_size": 50,
    "horizon_days": 3,
    "positive_return": 0.10,
    "watchlist_text": "",
}
MIN_FOCUS_CONSECUTIVE_UP_DAYS = 3
MIN_FOCUS_BOARD_CANDIDATES = 10
LATEST_RESULT_PROBE_SECONDS = 20


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #1d1d1f;
            --muted: #6e6e73;
            --soft-ink: #424245;
            --panel: rgba(255, 255, 255, 0.76);
            --panel-strong: rgba(255, 255, 255, 0.9);
            --line: rgba(15, 23, 42, 0.08);
            --shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
            --shadow-strong: 0 24px 70px rgba(15, 23, 42, 0.14);
            --accent: #0071e3;
            --accent-soft: rgba(0, 113, 227, 0.12);
            --surface-blue: rgba(225, 238, 255, 0.56);
            --surface-ice: rgba(244, 248, 255, 0.72);
            --up: #ff5f57;
            --down: #30b0a0;
            --volume: #9aa8b6;
            --glass: saturate(180%) blur(26px);
        }
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "PingFang SC", "Helvetica Neue", "Noto Sans SC", sans-serif;
        }
        .stApp {
            background:
                radial-gradient(circle at 0% 0%, rgba(0, 113, 227, 0.14), transparent 28%),
                radial-gradient(circle at 100% 10%, rgba(120, 120, 135, 0.08), transparent 22%),
                radial-gradient(circle at 50% 100%, rgba(255, 255, 255, 0.88), transparent 35%),
                linear-gradient(180deg, #f5f5f7 0%, #eef3f8 58%, #fbfbfd 100%);
        }
        .block-container,
        [data-testid="stAppViewBlockContainer"] {
            max-width: min(1880px, 98vw) !important;
            padding-top: 1.1rem;
            padding-bottom: 3rem;
            padding-left: clamp(1rem, 2.4vw, 2.6rem);
            padding-right: clamp(1rem, 2.4vw, 2.6rem);
        }
        [data-testid="stMainBlockContainer"] {
            max-width: min(1880px, 98vw) !important;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.76), rgba(248, 250, 253, 0.82));
            backdrop-filter: var(--glass);
            border-right: 1px solid rgba(15, 23, 42, 0.05);
        }
        [data-testid="stSidebar"] > div:first-child {
            background: transparent;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--ink);
            letter-spacing: -0.02em;
            font-weight: 650;
        }
        [data-baseweb="input"],
        [data-baseweb="select"],
        textarea,
        .stTextArea textarea {
            border-radius: 18px !important;
            border: 1px solid rgba(15, 23, 42, 0.08) !important;
            background: rgba(255, 255, 255, 0.82) !important;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.4);
        }
        .stButton > button,
        .stDownloadButton > button {
            border-radius: 999px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background: rgba(255, 255, 255, 0.82);
            color: var(--ink);
            min-height: 2.75rem;
            font-weight: 620;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
            transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 16px 28px rgba(15, 23, 42, 0.1);
            background: rgba(255, 255, 255, 0.94);
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #1d1d1f 0%, #31353a 100%);
            color: #fff;
            border-color: transparent;
            box-shadow: 0 16px 28px rgba(29, 29, 31, 0.24);
        }
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #121214 0%, #25272c 100%);
        }
        .stRadio [role="radiogroup"] {
            gap: 0.4rem;
            padding: 0.35rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.7);
            border: 1px solid rgba(15, 23, 42, 0.06);
        }
        .hero-card {
            position: relative;
            overflow: hidden;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.86), rgba(247, 250, 255, 0.76)),
                linear-gradient(180deg, rgba(255,255,255,0.55), rgba(255,255,255,0.28));
            border: 1px solid rgba(255, 255, 255, 0.8);
            border-radius: 36px;
            padding: 1.85rem 1.95rem;
            box-shadow: var(--shadow-strong);
            backdrop-filter: var(--glass);
        }
        .hero-card::after {
            content: "";
            position: absolute;
            right: -72px;
            top: -72px;
            width: 320px;
            height: 320px;
            background: radial-gradient(circle, rgba(0, 113, 227, 0.16), transparent 68%);
        }
        .hero-card::before {
            content: "";
            position: absolute;
            left: -80px;
            bottom: -120px;
            width: 360px;
            height: 280px;
            background: radial-gradient(circle, rgba(255, 255, 255, 0.9), transparent 70%);
        }
        .hero-grid {
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.9fr);
            gap: 1.45rem;
            align-items: stretch;
        }
        .hero-copy {
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .hero-kicker {
            font-size: 0.8rem;
            font-weight: 650;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: rgba(0, 113, 227, 0.88);
            margin-bottom: 0.7rem;
        }
        .hero-title {
            margin: 0;
            color: var(--ink);
            font-size: clamp(2.35rem, 4vw, 4.3rem);
            line-height: 0.98;
            letter-spacing: -0.05em;
            font-weight: 720;
        }
        .hero-description {
            margin: 0.95rem 0 0;
            max-width: 980px;
            color: var(--soft-ink);
            font-size: 1.04rem;
            line-height: 1.7;
        }
        .hero-side {
            position: relative;
            display: flex;
            align-items: stretch;
        }
        .hero-panel {
            width: 100%;
            background: rgba(255, 255, 255, 0.54);
            border: 1px solid rgba(255, 255, 255, 0.84);
            border-radius: 28px;
            padding: 1.2rem;
            backdrop-filter: var(--glass);
            box-shadow: 0 18px 36px rgba(15, 23, 42, 0.08);
        }
        .hero-panel-title {
            margin: 0;
            color: var(--ink);
            font-size: 0.88rem;
            font-weight: 620;
            letter-spacing: 0.02em;
        }
        .hero-panel-note {
            margin: 0.35rem 0 0.95rem;
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.55;
        }
        .hero-stat-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
        }
        .hero-stat-card {
            background: rgba(255, 255, 255, 0.74);
            border: 1px solid rgba(15, 23, 42, 0.05);
            border-radius: 22px;
            padding: 0.9rem 1rem;
            min-height: 112px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        }
        .hero-stat-label {
            color: var(--muted);
            font-size: 0.82rem;
            letter-spacing: 0.02em;
        }
        .hero-stat-value {
            margin-top: 0.38rem;
            color: var(--ink);
            font-size: 1.18rem;
            font-weight: 680;
            letter-spacing: -0.03em;
            line-height: 1.15;
        }
        .hero-stat-note {
            margin-top: 0.4rem;
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.5;
        }
        .hero-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 1.15rem;
        }
        .hero-pill, .brief-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.42rem 0.85rem;
            border-radius: 999px;
            font-size: 0.84rem;
            color: var(--soft-ink);
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(15, 23, 42, 0.06);
            backdrop-filter: var(--glass);
        }
        .section-banner {
            background: linear-gradient(180deg, rgba(255,255,255,0.74), rgba(255,255,255,0.56));
            border: 1px solid rgba(255, 255, 255, 0.72);
            border-radius: 28px;
            padding: 1.1rem 1.2rem;
            box-shadow: var(--shadow);
            backdrop-filter: var(--glass);
            margin: 1.3rem 0 0.95rem;
        }
        .section-eyebrow {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: rgba(0, 113, 227, 0.82);
            margin-bottom: 0.28rem;
            font-weight: 640;
        }
        .section-title {
            font-size: 1.36rem;
            font-weight: 690;
            color: var(--ink);
            margin: 0;
            letter-spacing: -0.03em;
        }
        .section-note {
            margin-top: 0.38rem;
            color: var(--muted);
            font-size: 0.96rem;
            line-height: 1.6;
        }
        .metric-card {
            position: relative;
            overflow: hidden;
            background: linear-gradient(180deg, rgba(255,255,255,0.8), rgba(255,255,255,0.62));
            border: 1px solid rgba(255, 255, 255, 0.84);
            border-radius: 26px;
            padding: 1rem 1.08rem;
            min-height: 138px;
            box-shadow: 0 16px 32px rgba(15, 23, 42, 0.07);
            backdrop-filter: var(--glass);
        }
        .metric-card::after {
            content: "";
            position: absolute;
            top: 0;
            left: 1rem;
            right: 1rem;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(0, 113, 227, 0.2), transparent);
        }
        .card-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            color: #6f7480;
            letter-spacing: 0.11em;
            margin-bottom: 0.4rem;
            font-weight: 620;
        }
        .card-value {
            font-size: 1.75rem;
            font-weight: 690;
            color: var(--ink);
            line-height: 1.1;
            letter-spacing: -0.04em;
        }
        .card-note {
            font-size: 0.92rem;
            color: var(--muted);
            margin-top: 0.5rem;
            line-height: 1.55;
        }
        .section-tag {
            display: inline-block;
            padding: 0.26rem 0.68rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: rgba(0, 113, 227, 0.92);
            font-size: 0.8rem;
            margin-bottom: 0.6rem;
            font-weight: 620;
        }
        .focus-card {
            position: relative;
            overflow: hidden;
            background: linear-gradient(180deg, rgba(255,255,255,0.82), rgba(248,250,253,0.68));
            border: 1px solid rgba(255, 255, 255, 0.84);
            border-radius: 30px;
            padding: 1.15rem 1.15rem 1.05rem;
            box-shadow: 0 18px 36px rgba(15, 23, 42, 0.08);
            backdrop-filter: var(--glass);
            min-height: 270px;
        }
        .focus-card::after {
            content: "";
            position: absolute;
            right: -80px;
            top: -80px;
            width: 220px;
            height: 220px;
            background: radial-gradient(circle, rgba(0, 113, 227, 0.1), transparent 72%);
        }
        .focus-head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.8rem;
            position: relative;
            z-index: 1;
        }
        .focus-rank {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 3rem;
            height: 3rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(0, 113, 227, 0.16), rgba(0, 113, 227, 0.08));
            color: var(--accent);
            font-size: 1rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.45);
        }
        .focus-symbol {
            margin: 0;
            color: var(--ink);
            font-size: 1.3rem;
            font-weight: 680;
            letter-spacing: -0.03em;
            line-height: 1.15;
        }
        .focus-symbol span {
            display: block;
            margin-top: 0.2rem;
            color: var(--muted);
            font-size: 0.92rem;
            font-weight: 540;
            letter-spacing: 0.01em;
        }
        .focus-subtitle {
            margin-top: 0.32rem;
            color: var(--soft-ink);
            font-size: 0.93rem;
        }
        .focus-score-block {
            margin-top: 1rem;
            position: relative;
            z-index: 1;
        }
        .focus-score-value {
            color: var(--ink);
            font-size: 2.4rem;
            line-height: 0.95;
            letter-spacing: -0.08em;
            font-weight: 720;
        }
        .focus-score-label {
            margin-top: 0.3rem;
            color: var(--muted);
            font-size: 0.88rem;
        }
        .focus-meta {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.55rem;
            margin-top: 1rem;
            position: relative;
            z-index: 1;
        }
        .focus-stat {
            padding: 0.72rem 0.78rem;
            border-radius: 18px;
            background: rgba(255,255,255,0.7);
            border: 1px solid rgba(15, 23, 42, 0.05);
        }
        .focus-stat-label {
            display: block;
            color: var(--muted);
            font-size: 0.78rem;
            margin-bottom: 0.2rem;
        }
        .focus-stat-value {
            color: var(--ink);
            font-size: 0.98rem;
            font-weight: 650;
        }
        .focus-reason {
            margin-top: 0.95rem;
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.6;
            position: relative;
            z-index: 1;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        .focus-status-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.85rem;
            position: relative;
            z-index: 1;
        }
        .focus-status-row .overview-action-chip {
            border-color: rgba(15, 23, 42, 0.08);
            box-shadow: none;
        }
        .focus-plan {
            margin-top: 0.9rem;
            padding-top: 0.85rem;
            border-top: 1px solid rgba(15, 23, 42, 0.06);
            position: relative;
            z-index: 1;
        }
        .focus-plan-label {
            color: var(--accent);
            font-size: 0.8rem;
            font-weight: 650;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .focus-plan-text {
            margin-top: 0.34rem;
            color: var(--soft-ink);
            font-size: 0.94rem;
            line-height: 1.6;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        .insight-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 1rem 0 1.25rem;
        }
        .insight-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.78), rgba(248,251,255,0.68));
            border: 1px solid rgba(255,255,255,0.84);
            border-radius: 24px;
            padding: 1rem 1.05rem;
            box-shadow: 0 14px 28px rgba(15, 23, 42, 0.06);
            backdrop-filter: var(--glass);
        }
        .insight-label {
            color: var(--muted);
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 650;
        }
        .insight-value {
            margin-top: 0.36rem;
            color: var(--ink);
            font-size: 1.7rem;
            line-height: 1;
            letter-spacing: -0.05em;
            font-weight: 700;
        }
        .insight-note {
            margin-top: 0.42rem;
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.55;
        }
        .overview-banner {
            position: relative;
            overflow: hidden;
            background: linear-gradient(135deg, rgba(20, 23, 28, 0.94), rgba(36, 41, 48, 0.88));
            border-radius: 32px;
            padding: 1.35rem 1.4rem;
            color: #f4f8fb;
            box-shadow: 0 22px 42px rgba(17, 24, 39, 0.22);
            margin-bottom: 1.1rem;
        }
        .overview-banner::after {
            content: "";
            position: absolute;
            right: -60px;
            top: -80px;
            width: 260px;
            height: 260px;
            background: radial-gradient(circle, rgba(0, 113, 227, 0.2), transparent 68%);
        }
        .overview-main {
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: minmax(0, 1.25fr) minmax(220px, 0.75fr);
            gap: 1rem;
            align-items: start;
        }
        .overview-title {
            margin: 0;
            font-size: clamp(1.9rem, 3vw, 3rem);
            font-weight: 700;
            color: #f7fafc;
            letter-spacing: -0.05em;
        }
        .overview-subtitle {
            margin-top: 0.45rem;
            color: rgba(237, 242, 247, 0.76);
            font-size: 0.96rem;
            line-height: 1.6;
        }
        .overview-score-card {
            background: rgba(255,255,255,0.09);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 24px;
            padding: 1rem 1.05rem;
            backdrop-filter: blur(18px);
        }
        .overview-score-label {
            color: rgba(237, 242, 247, 0.68);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }
        .overview-score-value {
            margin-top: 0.4rem;
            color: #ffffff;
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.07em;
            line-height: 0.96;
        }
        .overview-score-note {
            margin-top: 0.42rem;
            color: rgba(237, 242, 247, 0.72);
            font-size: 0.9rem;
            line-height: 1.5;
        }
        .overview-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
            position: relative;
            z-index: 1;
        }
        .overview-meta .brief-pill {
            background: rgba(255, 255, 255, 0.12);
            color: #eff6fa;
            border-color: rgba(255, 255, 255, 0.08);
        }
        .overview-action-row {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.55rem;
            margin-top: 0.95rem;
            position: relative;
            z-index: 1;
        }
        .overview-action-chip {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.58rem 0.95rem;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 0.88rem;
            font-weight: 690;
            letter-spacing: 0.02em;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
        }
        .overview-action-chip.buy {
            background: rgba(255, 95, 87, 0.18);
            color: #ffe7e4;
        }
        .overview-action-chip.hold {
            background: rgba(10, 132, 255, 0.18);
            color: #dff0ff;
        }
        .overview-action-chip.sell {
            background: rgba(48, 176, 160, 0.2);
            color: #dcfffb;
        }
        .overview-action-chip.watch {
            background: rgba(255, 214, 10, 0.18);
            color: #fff5c7;
        }
        .overview-action-note {
            margin-top: 0.62rem;
            color: rgba(237, 242, 247, 0.8);
            font-size: 0.94rem;
            line-height: 1.62;
            max-width: 78rem;
            position: relative;
            z-index: 1;
        }
        .overview-action-row + .overview-action-note + .overview-action-row,
        .overview-action-row + .overview-action-note + .overview-action-row + .overview-action-note {
            display: none;
        }
        .news-item {
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(255, 255, 255, 0.8);
            border-radius: 22px;
            padding: 0.95rem 1rem;
            margin-bottom: 0.72rem;
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.06);
            backdrop-filter: var(--glass);
        }
        .news-meta {
            font-size: 0.82rem;
            color: #7b818b;
            margin-bottom: 0.4rem;
        }
        .news-title, .news-title a {
            font-size: 1rem;
            font-weight: 650;
            color: var(--ink);
            text-decoration: none;
            line-height: 1.55;
        }
        .news-title a:hover {
            color: var(--accent);
        }
        .news-source {
            margin-top: 0.42rem;
            color: #667580;
            font-size: 0.85rem;
        }
        .freshness-banner {
            background: linear-gradient(180deg, rgba(255,255,255,0.82), rgba(248,251,255,0.74));
            border: 1px solid rgba(255, 255, 255, 0.86);
            border-radius: 24px;
            padding: 0.95rem 1.05rem;
            margin: 0.4rem 0 1rem;
            box-shadow: 0 14px 26px rgba(15, 23, 42, 0.06);
            backdrop-filter: var(--glass);
        }
        .freshness-title {
            font-size: 0.82rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #6d7480;
            margin-bottom: 0.68rem;
            font-weight: 700;
        }
        .freshness-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .freshness-pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.46rem 0.82rem;
            font-size: 0.86rem;
            color: var(--ink);
            background: rgba(255,255,255,0.84);
            border: 1px solid rgba(15, 23, 42, 0.08);
        }
        .freshness-pill.positive {
            background: rgba(48, 176, 160, 0.12);
            color: #0f766e;
            border-color: rgba(15, 118, 110, 0.14);
        }
        .freshness-pill.warning {
            background: rgba(255, 159, 10, 0.14);
            color: #9a6700;
            border-color: rgba(154, 103, 0, 0.12);
        }
        .freshness-note {
            color: #5d6673;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        .chart-caption {
            margin: 0 0 0.52rem;
            color: #707783;
            font-size: 0.92rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.48rem;
            background: rgba(255,255,255,0.62);
            border: 1px solid rgba(255, 255, 255, 0.78);
            padding: 0.34rem;
            border-radius: 18px;
            backdrop-filter: var(--glass);
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border: 1px solid transparent;
            border-radius: 14px;
            padding: 0.45rem 0.95rem;
            color: var(--soft-ink);
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(29, 29, 31, 0.96), rgba(52, 57, 63, 0.94));
            color: #f3f8fb !important;
            box-shadow: 0 10px 20px rgba(15, 23, 42, 0.14);
        }
        div[data-testid="stDataFrame"] {
            background: rgba(255,255,255,0.62);
            border: 1px solid rgba(255, 255, 255, 0.82);
            border-radius: 24px;
            padding: 0.35rem;
            box-shadow: 0 16px 30px rgba(15, 23, 42, 0.06);
            backdrop-filter: var(--glass);
        }
        div[data-testid="stExpander"] {
            border-radius: 24px;
            border-color: rgba(255, 255, 255, 0.82);
            background: rgba(255,255,255,0.62);
            backdrop-filter: var(--glass);
        }
        div[data-testid="stPlotlyChart"] {
            background: rgba(255,255,255,0.56);
            border: 1px solid rgba(255, 255, 255, 0.78);
            border-radius: 28px;
            padding: 0.3rem 0.4rem;
            box-shadow: 0 16px 32px rgba(15, 23, 42, 0.06);
            backdrop-filter: var(--glass);
        }
        div[data-testid="stAlert"] {
            border-radius: 22px;
            border: 1px solid rgba(255, 255, 255, 0.8);
            backdrop-filter: var(--glass);
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.05);
        }
        @media (max-width: 900px) {
            .hero-card {
                padding: 1.2rem 1.05rem;
                border-radius: 28px;
            }
            .hero-grid,
            .overview-main,
            .insight-strip {
                grid-template-columns: 1fr;
            }
            .metric-card {
                min-height: 122px;
            }
            .focus-meta,
            .hero-stat-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _section_header(eyebrow: str, title: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="section-banner">
            <div class="section-eyebrow">{escape(eyebrow)}</div>
            <p class="section-title">{escape(title)}</p>
            <div class="section-note">{escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_short_amount(amount: float | int | None) -> str:
    if amount is None or pd.isna(amount):
        return "--"
    amount = float(amount)
    if abs(amount) >= 1e8:
        return f"{amount / 1e8:.2f}亿"
    if abs(amount) >= 1e4:
        return f"{amount / 1e4:.1f}万"
    return f"{amount:.0f}"


def _format_pct(value: float | None, *, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "--"
    prefix = "+" if signed and float(value) > 0 else ""
    return f"{prefix}{float(value):.2f}%"


def _build_intraday_snapshot(minute: pd.DataFrame) -> dict[str, float | str]:
    if minute.empty:
        return {
            "session_label": "暂无分时",
            "last_price": float("nan"),
            "session_change_pct": float("nan"),
            "vwap_gap_pct": float("nan"),
            "range_pct": float("nan"),
            "max_drawdown_pct": float("nan"),
            "tail_above_avg_pct": float("nan"),
            "total_amount": float("nan"),
            "bars": 0,
        }

    view = minute.sort_values("datetime").reset_index(drop=True)
    open_price = float(view["open"].iloc[0]) if pd.notna(view["open"].iloc[0]) else float("nan")
    last_price = float(view["close"].iloc[-1]) if pd.notna(view["close"].iloc[-1]) else float("nan")
    latest_avg = float(view["avg_price"].iloc[-1]) if pd.notna(view["avg_price"].iloc[-1]) else float("nan")
    high_price = float(view["high"].max()) if "high" in view.columns else float("nan")
    low_price = float(view["low"].min()) if "low" in view.columns else float("nan")
    running_peak = view["close"].cummax().replace(0, pd.NA)
    drawdown = (1 - view["close"] / running_peak).clip(lower=0)
    tail_view = view.tail(min(30, len(view)))
    tail_above_avg = float((tail_view["close"] >= tail_view["avg_price"]).mean()) if not tail_view.empty else float("nan")

    session_change_pct = (last_price / open_price - 1) * 100 if open_price and pd.notna(open_price) else float("nan")
    vwap_gap_pct = (last_price / latest_avg - 1) * 100 if latest_avg and pd.notna(latest_avg) else float("nan")
    range_pct = (high_price / low_price - 1) * 100 if low_price and pd.notna(low_price) else float("nan")

    return {
        "session_label": view["datetime"].iloc[-1].strftime("%Y-%m-%d"),
        "last_price": last_price,
        "session_change_pct": session_change_pct,
        "vwap_gap_pct": vwap_gap_pct,
        "range_pct": range_pct,
        "max_drawdown_pct": float(drawdown.max() * 100) if not drawdown.empty else float("nan"),
        "tail_above_avg_pct": tail_above_avg * 100 if pd.notna(tail_above_avg) else float("nan"),
        "total_amount": float(view["amount"].sum()) if "amount" in view.columns else float("nan"),
        "bars": int(len(view)),
    }


def _score_base_attention(
    probability: float,
    stage_score: float,
    snapshot: dict[str, float],
    quant_score: float,
    launch_score: float = 50.0,
) -> float:
    trend_component = 50 + snapshot["close_vs_ma20"] * 220 + snapshot["ret_20"] * 70
    volume_component = 45 + (snapshot["volume_ratio_5"] - 1.0) * 35
    breakout_component = 50 + snapshot["breakout_distance_20"] * 280
    launch_component = 50 + (float(launch_score) - 50.0) * 0.75
    score = (
        probability * 100 * 0.29
        + stage_score * 0.22
        + quant_score * 0.19
        + max(0.0, min(trend_component, 100.0)) * 0.13
        + max(0.0, min(volume_component, 100.0)) * 0.05
        + max(0.0, min(breakout_component, 100.0)) * 0.05
        + max(0.0, min(launch_component, 100.0)) * 0.07
    )
    return float(max(0.0, min(score, 100.0)))


def _precision_priority_from_model_result(model_result) -> tuple[int, str, float, float, int]:
    if model_result is None:
        return 0, "未做90%精度认证", 1.0, 0.0, 0
    threshold = float(getattr(model_result, "precision_gate_threshold", 1.0) or 1.0)
    precision = float(getattr(model_result, "precision_gate_precision", 0.0) or 0.0)
    support = int(getattr(model_result, "precision_gate_support", 0) or 0)
    active = bool(getattr(model_result, "precision_gate_active", False))
    label = str(getattr(model_result, "precision_gate_label", "未达90%精度门槛"))
    if active:
        return 2, label, threshold, precision, support
    if precision >= 0.90 and support > 0:
        return 1, label, threshold, precision, support
    return 0, label, threshold, precision, support


def _precision_priority_from_certification(model_result, backtest=None) -> tuple[int, str, float, float, int]:
    priority, label, threshold, precision, support = _precision_priority_from_model_result(model_result)
    if backtest is None:
        return priority, label, threshold, precision, support

    backtest_precision = float(getattr(backtest, "achieved_precision", 0.0) or 0.0)
    backtest_threshold = float(getattr(backtest, "selected_threshold", threshold) or threshold)
    backtest_support = int(getattr(backtest, "trade_count", support) or support)
    latest_signal_active = bool(getattr(backtest, "latest_signal_active", False))
    target_reached = bool(getattr(backtest, "target_reached", False))

    if latest_signal_active:
        return 3, "90%精度放行", backtest_threshold, backtest_precision, backtest_support
    if target_reached:
        return 2, "历史回测达标", backtest_threshold, backtest_precision, backtest_support
    return priority, label, threshold, precision, support


@lru_cache(maxsize=4096)
def _local_precision_certification(
    symbol: str,
    name: str,
    horizon_days: int,
    positive_return: float,
    market_data_date: str | None,
) -> dict[str, object]:
    default_result = {
        "certification_ready": False,
        "probability_up": None,
        "predicted_upside_pct": 0.0,
        "predicted_upside_low_pct": 0.0,
        "predicted_upside_high_pct": 0.0,
        "precision_priority": 0,
        "precision_gate_label": "未做90%精度认证",
        "precision_gate_threshold": 1.0,
        "precision_gate_precision": 0.0,
        "precision_gate_support": 0,
        "backtest_status_label": "回测准备中",
        "backtest_precision_pct": 0.0,
        "backtest_trade_count": 0,
        "backtest_latest_signal_active": False,
        "backtest_target_reached": False,
    }
    try:
        base = _prepare_symbol_base_analysis(
            symbol=symbol,
            name=name,
            market_data_date=market_data_date,
        )
        if base is None:
            return default_result
        daily = base.get("daily")
        if not isinstance(daily, pd.DataFrame) or daily.empty:
            return default_result
        model_result = train_probability_model(
            daily,
            horizon_days=horizon_days,
            positive_return=positive_return,
        )
        backtest = run_daily_strategy_backtest(
            daily,
            horizon_days=horizon_days,
            positive_return=positive_return,
            model_result=model_result,
        )
        priority, label, threshold, precision, support = _precision_priority_from_certification(model_result, backtest)
        return {
            "certification_ready": True,
            "probability_up": round(float(model_result.latest_probability) * 100, 2),
            "predicted_upside_pct": round(float(getattr(model_result, "predicted_upside_pct", 0.0) or 0.0), 2),
            "predicted_upside_low_pct": round(float(getattr(model_result, "predicted_upside_low_pct", 0.0) or 0.0), 2),
            "predicted_upside_high_pct": round(float(getattr(model_result, "predicted_upside_high_pct", 0.0) or 0.0), 2),
            "precision_priority": int(priority),
            "precision_gate_label": str(label),
            "precision_gate_threshold": round(float(threshold), 2),
            "precision_gate_precision": round(float(precision) * 100, 2),
            "precision_gate_support": int(support),
            "backtest_status_label": str(getattr(backtest, "status_label", "未回测")),
            "backtest_precision_pct": round(float(getattr(backtest, "achieved_precision", 0.0)) * 100, 2),
            "backtest_trade_count": int(getattr(backtest, "trade_count", 0) or 0),
            "backtest_latest_signal_active": bool(getattr(backtest, "latest_signal_active", False)),
            "backtest_target_reached": bool(getattr(backtest, "target_reached", False)),
        }
    except Exception:
        return default_result


def _confidence_adjusted_score(score: float, confidence: float | None) -> float:
    confidence_value = float(confidence if confidence is not None else 55.0)
    confidence_scale = 0.45 + max(0.0, min(confidence_value, 100.0)) / 100 * 0.55
    return float(max(0.0, min(50 + (score - 50) * confidence_scale, 100.0)))


def _score_final_attention(
    base_score: float,
    sector_score: float,
    fund_score: float,
    news_score: float,
    *,
    fund_confidence: float | None = None,
    news_confidence: float | None = None,
) -> float:
    adjusted_fund_score = _confidence_adjusted_score(fund_score, fund_confidence)
    adjusted_news_score = _confidence_adjusted_score(news_score, news_confidence)
    synergy_bonus = 0.0
    if base_score >= 60 and adjusted_fund_score >= 60 and adjusted_news_score >= 60:
        synergy_bonus += 2.0
    if adjusted_fund_score <= 42 or adjusted_news_score <= 42:
        synergy_bonus -= 1.5
    score = base_score * 0.58 + sector_score * 0.18 + adjusted_fund_score * 0.14 + adjusted_news_score * 0.10 + synergy_bonus
    return float(max(0.0, min(score, 100.0)))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _apply_replay_calibration_to_row(row: dict[str, object], optimization_profile: dict[str, object] | None) -> dict[str, object]:
    if not optimization_profile:
        return row
    calibrated = compute_replay_calibrated_scores(row, optimization_profile)
    merged = dict(row)
    merged.setdefault("raw_probability_up", _safe_float(row.get("probability_up")))
    merged.setdefault("raw_attention_score", _safe_float(row.get("attention_score")))
    merged.setdefault(
        "raw_enhanced_attention_score",
        _safe_float(row.get("enhanced_attention_score"), _safe_float(row.get("attention_score"))),
    )
    merged["pre_replay_probability_up"] = _safe_float(row.get("probability_up"))
    merged["pre_replay_attention_score"] = _safe_float(row.get("attention_score"))
    merged["pre_replay_enhanced_attention_score"] = _safe_float(
        row.get("enhanced_attention_score"),
        _safe_float(row.get("attention_score")),
    )
    merged.setdefault("enhanced_probability_up", merged["pre_replay_probability_up"])
    merged.setdefault("model_probability_up", _safe_float(row.get("probability_up")))
    merged.setdefault("model_attention_score", _safe_float(row.get("attention_score")))
    merged.setdefault(
        "model_enhanced_attention_score",
        _safe_float(row.get("enhanced_attention_score"), _safe_float(row.get("attention_score"))),
    )
    merged["probability_up"] = float(calibrated["probability_up"])
    merged["attention_score"] = float(calibrated["attention_score"])
    merged["enhanced_attention_score"] = float(calibrated["enhanced_attention_score"])
    merged["display_probability_up"] = float(calibrated["probability_up"])
    merged["display_attention_score"] = float(calibrated["attention_score"])
    merged["display_enhanced_attention_score"] = float(calibrated["enhanced_attention_score"])
    merged["replay_probability_delta_pct"] = float(calibrated["probability_delta_pct"])
    merged["replay_attention_delta"] = float(calibrated["attention_delta"])
    merged["replay_enhanced_attention_delta"] = float(calibrated["enhanced_attention_delta"])
    merged["final_rank_score"] = _safe_float(
        merged.get("ranking_score"),
        _safe_float(merged.get("enhanced_attention_score")),
    )
    if "predicted_upside_pct" in calibrated:
        merged["predicted_upside_pct"] = float(calibrated["predicted_upside_pct"])
    if "predicted_upside_low_pct" in calibrated:
        merged["predicted_upside_low_pct"] = float(calibrated["predicted_upside_low_pct"])
    if "predicted_upside_high_pct" in calibrated:
        merged["predicted_upside_high_pct"] = float(calibrated["predicted_upside_high_pct"])
    if "replay_empirical_probability_pct" in calibrated:
        merged["replay_empirical_probability_pct"] = float(calibrated["replay_empirical_probability_pct"])
    if "replay_empirical_upside_pct" in calibrated:
        merged["replay_empirical_upside_pct"] = float(calibrated["replay_empirical_upside_pct"])
    if "replay_empirical_intraday_upside_pct" in calibrated:
        merged["replay_empirical_intraday_upside_pct"] = float(calibrated["replay_empirical_intraday_upside_pct"])
    if "replay_empirical_hit_rate_pct" in calibrated:
        merged["replay_empirical_hit_rate_pct"] = float(calibrated["replay_empirical_hit_rate_pct"])
    merged["replay_calibration_confidence"] = float(calibrated["replay_calibration_confidence"])
    merged["replay_calibration_note"] = str(calibrated["replay_calibration_note"])
    merged["replay_calibration_active"] = bool(calibrated["replay_calibration_active"])
    merged["replay_probability_bucket"] = str(calibrated["replay_probability_bucket"])
    merged["replay_quant_bucket"] = str(calibrated["replay_quant_bucket"])
    merged["replay_precision_segment"] = str(calibrated["replay_precision_segment"])
    merged["replay_market_state"] = str(calibrated.get("replay_market_state", "unknown"))
    merged["replay_market_stage_proxy"] = str(calibrated.get("replay_market_stage_proxy", "unknown"))
    return merged


def _apply_replay_calibration_to_board(
    board_df: pd.DataFrame,
    optimization_profile: dict[str, object] | None,
) -> pd.DataFrame:
    if board_df.empty or not optimization_profile:
        return board_df.copy()
    has_short_window = int(optimization_profile.get("review_days", 0) or 0) > 0
    has_market_window = int(optimization_profile.get("market_replay_days", 0) or 0) > 0
    if not has_short_window and not has_market_window:
        return board_df.copy()
    calibrated_rows = [
        _apply_replay_calibration_to_row(row, optimization_profile)
        for row in board_df.to_dict("records")
    ]
    return pd.DataFrame(calibrated_rows)


def _signal_value(signal: object, key: str, default: object = None) -> object:
    if isinstance(signal, dict):
        return signal.get(key, default)
    return getattr(signal, key, default)


def _text_signal_bias(text: object) -> int:
    content = str(text or "").strip()
    if not content:
        return 0
    positive_keywords = (
        "强",
        "偏多",
        "多头",
        "突破",
        "放量",
        "承接",
        "修复",
        "回流",
        "站上",
        "走强",
        "跟随",
        "流入",
        "共振",
        "新高",
        "进攻",
    )
    negative_keywords = (
        "弱",
        "偏空",
        "空头",
        "破位",
        "回落",
        "走弱",
        "跌破",
        "分歧",
        "减仓",
        "风险",
        "流出",
        "兑现",
        "背离",
        "承压",
    )
    positive_hits = sum(1 for keyword in positive_keywords if keyword in content)
    negative_hits = sum(1 for keyword in negative_keywords if keyword in content)
    return positive_hits - negative_hits


def _unique_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for line in lines:
        text = str(line or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _build_launch_window_view(
    payload: dict[str, object] | pd.Series,
    *,
    stage_code: str | None = None,
    stage_label: str | None = None,
    intraday_bias: int | None = None,
) -> dict[str, object]:
    row = dict(payload) if isinstance(payload, (dict, pd.Series)) else {}
    snapshot = row.get("snapshot")
    probability_up = _safe_float(row.get("probability_up"))
    predicted_upside_pct = _safe_float(row.get("predicted_upside_pct"))
    if predicted_upside_pct <= 0:
        predicted_upside_pct = max((probability_up - 40.0) * 0.18, 0.0)
    resolved_stage_label = str(stage_label or row.get("stage_label") or _signal_value(row.get("stage"), "label", ""))
    resolved_stage_code = str(stage_code or row.get("stage_code") or _signal_value(row.get("stage"), "code", resolved_stage_label))
    resolved_intraday_bias = intraday_bias
    if resolved_intraday_bias is None:
        intraday_text = " ".join(
            part
            for part in [
                str(row.get("intraday_label", "")),
                str(_signal_value(row.get("intraday"), "label", "")),
                str(_signal_value(row.get("intraday"), "summary", "")),
                str(_signal_value(row.get("intraday_structure_signal"), "label", "")),
                str(_signal_value(row.get("intraday_structure_signal"), "summary", "")),
                str(row.get("tomorrow_setup", "")),
                str(row.get("tomorrow_bias", "")),
            ]
            if str(part).strip()
        )
        resolved_intraday_bias = _text_signal_bias(intraday_text)

    assessment = assess_launch_window(
        stage_code=resolved_stage_code,
        stage_label=resolved_stage_label,
        probability_up=probability_up,
        predicted_upside_pct=predicted_upside_pct,
        quant_score=_safe_float(row.get("quant_score"), 50.0),
        sector_score=_safe_float(row.get("sector_score"), _safe_float(_signal_value(row.get("sector_signal"), "sector_score"), 50.0)),
        fund_score=_safe_float(row.get("fund_score"), _safe_float(_signal_value(row.get("fund_signal"), "fund_score"), 50.0)),
        news_score=_safe_float(row.get("news_score"), _safe_float(_signal_value(row.get("news_signal"), "sentiment_score"), 50.0)),
        launch_score=_safe_float(row.get("launch_score"), 50.0),
        launch_readiness_score=_safe_float(row.get("launch_readiness_score"), 50.0),
        market_resonance_score=_safe_float(row.get("market_resonance_score"), 50.0),
        launch_specialist_score=_safe_float(row.get("launch_specialist_score"), 50.0),
        launch_regime_fit_score=_safe_float(row.get("launch_regime_fit_score"), 50.0),
        launch_specialist_confidence=_safe_float(row.get("launch_specialist_confidence"), 50.0),
        close_vs_ma20=_safe_float(row.get("close_vs_ma20"), _safe_float(_signal_value(snapshot, "close_vs_ma20"), 0.0)),
        breakout_distance=_safe_float(
            row.get("breakout_distance_20"),
            _safe_float(_signal_value(snapshot, "breakout_distance_20"), 0.0),
        ),
        intraday_bias=int(resolved_intraday_bias),
    )
    return {
        "launch_window_label": str(assessment.label),
        "launch_window_status": str(assessment.status),
        "launch_window_summary": str(assessment.summary),
        "launch_window_score": float(assessment.window_score),
        "launch_window_confidence": float(assessment.window_confidence),
        "launch_window_drivers": " | ".join(str(item) for item in assessment.drivers if str(item).strip()),
    }


def _ensure_launch_window_columns(board_df: pd.DataFrame, *, force: bool = False) -> pd.DataFrame:
    if board_df.empty:
        return board_df.copy()
    board = board_df.copy()
    required_columns = {
        "launch_window_label",
        "launch_window_status",
        "launch_window_summary",
        "launch_window_score",
        "launch_window_confidence",
        "launch_window_drivers",
    }
    needs_fill = any(column not in board.columns for column in required_columns)
    if not needs_fill:
        numeric_cols = {"launch_window_score", "launch_window_confidence"}
        text_cols = required_columns - numeric_cols
        numeric_missing = any(board[column].isna().any() for column in numeric_cols if column in board.columns)
        text_missing = any((board[column].astype(str).str.strip() == "").any() for column in text_cols if column in board.columns)
        needs_fill = numeric_missing or text_missing
    if not force and not needs_fill:
        return board

    launch_view = board.apply(lambda row: pd.Series(_build_launch_window_view(row)), axis=1)
    for column in launch_view.columns:
        if force:
            board[column] = launch_view[column]
        elif column not in board.columns:
            board[column] = launch_view[column]
        elif pd.api.types.is_numeric_dtype(launch_view[column]):
            board[column] = pd.to_numeric(board[column], errors="coerce").fillna(launch_view[column])
        else:
            existing = board[column].astype(str)
            fallback = launch_view[column].astype(str)
            board[column] = existing.where(existing.str.strip() != "", fallback)
    return board


def _candidate_strategy_profile(candidate_strategy: str | None) -> dict[str, object]:
    strategy = str(candidate_strategy or "").strip()
    strategy_token = strategy.lower()
    if "策略1" in strategy or "strategy1" in strategy_token or strategy_token in {"s1", "1"}:
        return {
            "candidate_strategy": "策略1",
            "candidate_strategy_label": "策略1·趋势中继",
            "candidate_strategy_short_label": "趋势中继",
            "candidate_strategy_forecast_bias": "更重视延续胜率与回调承接",
            "candidate_strategy_note": "回调转强型，更强调稳健续涨与平台承接，不主打极限弹性。",
            "probability_multiplier": 1.03,
            "probability_offset_pct": 0.8,
            "upside_multiplier": 0.94,
            "upside_low_multiplier": 1.06,
            "upside_high_multiplier": 0.96,
            "attention_bonus": 1.6,
            "buy_threshold_probability": 58.0,
            "buy_threshold_upside": 4.8,
            "hold_threshold_probability": 50.0,
            "hold_threshold_upside": 2.5,
        }
    if "策略2" in strategy or "strategy2" in strategy_token or strategy_token in {"s2", "2"}:
        return {
            "candidate_strategy": "策略2",
            "candidate_strategy_label": "策略2·突破共振",
            "candidate_strategy_short_label": "突破共振",
            "candidate_strategy_forecast_bias": "更重视爆发弹性与板块共振，但同时抬高追高风控",
            "candidate_strategy_note": "强势突破型，更强调板块共振后的加速空间，但对追高回撤更敏感。",
            "probability_multiplier": 0.985,
            "probability_offset_pct": -1.2,
            "upside_multiplier": 1.12,
            "upside_low_multiplier": 0.92,
            "upside_high_multiplier": 1.18,
            "attention_bonus": 2.4,
            "buy_threshold_probability": 62.0,
            "buy_threshold_upside": 7.2,
            "hold_threshold_probability": 54.0,
            "hold_threshold_upside": 3.8,
        }
    if "strategy3" in strategy_token or strategy_token in {"s3", "3"}:
        return {
            "candidate_strategy": "strategy3",
            "candidate_strategy_label": "策略3·多因子主升预备",
            "candidate_strategy_short_label": "多因子主升",
            "candidate_strategy_forecast_bias": "更重视综合胜率、板块共振和低拥挤启动质量",
            "candidate_strategy_note": "中宽口径的主升预备策略，覆盖未达到策略1/2硬阈值但多因子质量较好的候选。",
            "probability_multiplier": 1.015,
            "probability_offset_pct": 0.4,
            "upside_multiplier": 1.03,
            "upside_low_multiplier": 1.00,
            "upside_high_multiplier": 1.06,
            "attention_bonus": 1.9,
            "buy_threshold_probability": 60.0,
            "buy_threshold_upside": 5.8,
            "hold_threshold_probability": 52.0,
            "hold_threshold_upside": 3.0,
        }
    if "dynamic_fallback" in strategy_token or "fallback" in strategy_token or "兜底" in strategy:
        return {
            "candidate_strategy": "dynamic_fallback",
            "candidate_strategy_label": "非正式兜底池",
            "candidate_strategy_short_label": "兜底池",
            "candidate_strategy_forecast_bias": "非正式策略榜兜底，仅用于避免首屏空白",
            "candidate_strategy_note": "策略1/策略2没有足够候选时的显式兜底，不作为正式策略入选依据。",
            "probability_multiplier": 1.0,
            "probability_offset_pct": 0.0,
            "upside_multiplier": 1.0,
            "upside_low_multiplier": 1.0,
            "upside_high_multiplier": 1.0,
            "attention_bonus": 0.0,
            "buy_threshold_probability": 60.0,
            "buy_threshold_upside": 6.0,
            "hold_threshold_probability": 52.0,
            "hold_threshold_upside": 3.0,
        }
    if "策略3" in strategy or "strategy3" in strategy_token:
        strategy = ""
    return {
        "candidate_strategy": strategy,
        "candidate_strategy_label": "通用模型",
        "candidate_strategy_short_label": "通用",
        "candidate_strategy_forecast_bias": "按统一口径评估",
        "candidate_strategy_note": "当前未命中特定硬筛选策略，按统一预测口径处理。",
        "probability_multiplier": 1.0,
        "probability_offset_pct": 0.0,
        "upside_multiplier": 1.0,
        "upside_low_multiplier": 1.0,
        "upside_high_multiplier": 1.0,
        "attention_bonus": 0.0,
        "buy_threshold_probability": 60.0,
        "buy_threshold_upside": 6.0,
        "hold_threshold_probability": 52.0,
        "hold_threshold_upside": 3.0,
    }


def _apply_candidate_strategy_prediction_profile(payload: dict[str, object]) -> dict[str, object]:
    if bool(payload.get("candidate_strategy_profile_applied")):
        return payload
    profile = _candidate_strategy_profile(payload.get("candidate_strategy"))
    probability_pct = _safe_float(payload.get("probability_up"))
    upside_pct = max(_safe_float(payload.get("predicted_upside_pct")), 0.0)
    upside_low_pct = max(_safe_float(payload.get("predicted_upside_low_pct")), 0.0)
    upside_high_pct = max(_safe_float(payload.get("predicted_upside_high_pct")), upside_pct)
    attention_score = _safe_float(payload.get("attention_score"))
    enhanced_attention_score = _safe_float(payload.get("enhanced_attention_score"), attention_score)
    payload.setdefault("raw_probability_up", round(probability_pct, 2))
    payload.setdefault("enhanced_probability_up", round(probability_pct, 2))
    payload.setdefault("raw_attention_score", round(attention_score, 2))
    payload.setdefault("raw_enhanced_attention_score", round(enhanced_attention_score, 2))
    raw_probability_pct = _safe_float(payload.get("raw_probability_up"), probability_pct)
    raw_attention_score = _safe_float(payload.get("raw_attention_score"), attention_score)
    raw_enhanced_attention_score = _safe_float(payload.get("raw_enhanced_attention_score"), enhanced_attention_score)

    probability_pct = float(
        np.clip(
            probability_pct * float(profile["probability_multiplier"]) + float(profile["probability_offset_pct"]),
            0.0,
            99.9,
        )
    )
    upside_pct = float(np.clip(upside_pct * float(profile["upside_multiplier"]), 0.0, 60.0))
    upside_low_pct = float(np.clip(min(upside_low_pct * float(profile["upside_low_multiplier"]), upside_pct), 0.0, upside_pct))
    upside_high_pct = float(
        np.clip(
            max(upside_high_pct * float(profile["upside_high_multiplier"]), upside_pct),
            upside_pct,
            80.0,
        )
    )
    attention_bonus = float(profile["attention_bonus"])
    payload["probability_up"] = round(probability_pct, 2)
    payload["display_probability_up"] = payload["probability_up"]
    payload["predicted_upside_pct"] = round(upside_pct, 2)
    payload["predicted_upside_low_pct"] = round(upside_low_pct, 2)
    payload["predicted_upside_high_pct"] = round(upside_high_pct, 2)
    payload["attention_score"] = round(min(max(attention_score + attention_bonus, 0.0), 100.0), 2)
    payload["enhanced_attention_score"] = round(min(max(enhanced_attention_score + attention_bonus, 0.0), 100.0), 2)
    payload["final_rank_score"] = round(_safe_float(payload.get("ranking_score"), payload["enhanced_attention_score"]), 2)
    payload["strategy_probability_delta_pct"] = round(payload["probability_up"] - raw_probability_pct, 2)
    payload["strategy_attention_delta"] = round(payload["attention_score"] - raw_attention_score, 2)
    payload["strategy_enhanced_attention_delta"] = round(
        payload["enhanced_attention_score"] - raw_enhanced_attention_score,
        2,
    )
    payload["candidate_strategy"] = str(profile["candidate_strategy"])
    payload["candidate_strategy_label"] = str(profile["candidate_strategy_label"])
    payload["candidate_strategy_short_label"] = str(profile["candidate_strategy_short_label"])
    payload["candidate_strategy_forecast_bias"] = str(profile["candidate_strategy_forecast_bias"])
    payload["candidate_strategy_note"] = str(profile["candidate_strategy_note"])
    payload["candidate_strategy_profile_applied"] = True
    return payload


def _build_execution_view(
    payload: dict[str, object] | pd.Series,
    *,
    launch_window_view: dict[str, object] | None = None,
    stage_code: str | None = None,
    stage_label: str | None = None,
    intraday_bias: int | None = None,
) -> dict[str, object]:
    row = dict(payload) if isinstance(payload, (dict, pd.Series)) else {}
    snapshot = row.get("snapshot")
    probability_up = _safe_float(row.get("probability_up"))
    predicted_upside_pct = _safe_float(row.get("predicted_upside_pct"))
    if predicted_upside_pct <= 0:
        predicted_upside_pct = max((probability_up - 40.0) * 0.18, 0.0)

    resolved_stage_label = str(stage_label or row.get("stage_label") or _signal_value(row.get("stage"), "label", ""))
    resolved_stage_code = str(stage_code or row.get("stage_code") or _signal_value(row.get("stage"), "code", resolved_stage_label))
    resolved_intraday_bias = intraday_bias
    if resolved_intraday_bias is None:
        intraday_text = " ".join(
            part
            for part in [
                str(row.get("intraday_label", "")),
                str(_signal_value(row.get("intraday"), "label", "")),
                str(_signal_value(row.get("intraday"), "summary", "")),
                str(_signal_value(row.get("intraday_structure_signal"), "label", "")),
                str(_signal_value(row.get("intraday_structure_signal"), "summary", "")),
                str(row.get("tomorrow_setup", "")),
                str(row.get("tomorrow_bias", "")),
            ]
            if str(part).strip()
        )
        resolved_intraday_bias = _text_signal_bias(intraday_text)

    resolved_launch_window_view = (
        launch_window_view
        if isinstance(launch_window_view, dict) and launch_window_view
        else _build_launch_window_view(
            row,
            stage_code=resolved_stage_code,
            stage_label=resolved_stage_label,
            intraday_bias=int(resolved_intraday_bias),
        )
    )
    assessment = assess_execution_readiness(
        stage_code=resolved_stage_code,
        stage_label=resolved_stage_label,
        probability_up=probability_up,
        predicted_upside_pct=predicted_upside_pct,
        quant_score=_safe_float(row.get("quant_score"), 50.0),
        launch_window_score=_safe_float(resolved_launch_window_view.get("launch_window_score"), 50.0),
        launch_window_status=str(resolved_launch_window_view.get("launch_window_status", "非启动窗")),
        launch_window_confidence=_safe_float(resolved_launch_window_view.get("launch_window_confidence"), 50.0),
        sector_score=_safe_float(row.get("sector_score"), _safe_float(_signal_value(row.get("sector_signal"), "sector_score"), 50.0)),
        fund_score=_safe_float(row.get("fund_score"), _safe_float(_signal_value(row.get("fund_signal"), "fund_score"), 50.0)),
        news_score=_safe_float(row.get("news_score"), _safe_float(_signal_value(row.get("news_signal"), "sentiment_score"), 50.0)),
        close_vs_ma20=_safe_float(row.get("close_vs_ma20"), _safe_float(_signal_value(snapshot, "close_vs_ma20"), 0.0)),
        breakout_distance=_safe_float(
            row.get("breakout_distance_20"),
            _safe_float(_signal_value(snapshot, "breakout_distance_20"), 0.0),
        ),
        intraday_bias=int(resolved_intraday_bias),
    )
    return {
        "execution_label": str(assessment.label),
        "execution_window": str(assessment.window),
        "execution_summary": str(assessment.summary),
        "execution_score": float(assessment.execution_score),
        "execution_confidence": float(assessment.execution_confidence),
        "execution_entry_zone": str(assessment.entry_zone),
        "execution_invalidation_rule": str(assessment.invalidation_rule),
        "reward_risk_label": str(assessment.reward_risk_label),
        "expected_return_pct": float(assessment.expected_return_pct),
        "drawdown_risk_pct": float(assessment.drawdown_risk_pct),
        "reward_risk_ratio": float(assessment.reward_risk_ratio),
        "chase_risk_label": str(assessment.chase_risk_label),
        "execution_drivers": " | ".join(str(item) for item in assessment.drivers if str(item).strip()),
    }


def _evaluate_symbol_action(detail: dict, display_context: dict[str, object]) -> dict[str, object]:
    probability_up = _safe_float(display_context.get("probability_up"))
    predicted_upside_pct = _safe_float(display_context.get("predicted_upside_pct"))
    if predicted_upside_pct <= 0:
        predicted_upside_pct = max((probability_up - 40.0) * 0.18, 0.0)
    base_attention_score = _safe_float(display_context.get("base_attention_score"))
    enhanced_attention_score = _safe_float(display_context.get("enhanced_attention_score"))
    quant_score = _safe_float(display_context.get("quant_score"))
    tomorrow_confidence = _safe_float(
        display_context.get("tomorrow_plan_confidence"),
        _safe_float(_signal_value(detail.get("tomorrow_plan"), "confidence"), 50.0),
    )
    strategy_profile = _candidate_strategy_profile(display_context.get("candidate_strategy"))
    sector_score = _safe_float(_signal_value(detail.get("sector_signal"), "sector_score"), 50.0)
    fund_score = _safe_float(_signal_value(detail.get("fund_signal"), "fund_score"), 50.0)
    news_score = _safe_float(_signal_value(detail.get("news_signal"), "sentiment_score"), 50.0)

    snapshot = detail.get("snapshot", {}) or {}
    close_vs_ma20 = _safe_float(snapshot.get("close_vs_ma20"))
    breakout_distance = _safe_float(snapshot.get("breakout_distance_20"))

    intraday_text = " ".join(
        part
        for part in [
            str(_signal_value(detail.get("intraday"), "label", "")),
            str(_signal_value(detail.get("intraday"), "summary", "")),
            str(_signal_value(detail.get("intraday_structure_signal"), "label", "")),
            str(_signal_value(detail.get("intraday_structure_signal"), "summary", "")),
            str(display_context.get("tomorrow_setup", "")),
            str(display_context.get("tomorrow_bias", "")),
        ]
        if str(part).strip()
    )
    intraday_tone = _text_signal_bias(intraday_text)

    backtest = detail.get("backtest")
    model = detail.get("model")
    precision_priority, precision_label, _, precision_precision, precision_support = _precision_priority_from_certification(
        model,
        backtest,
    )
    achieved_precision_pct = _safe_float(getattr(backtest, "achieved_precision", 0.0)) * 100
    latest_signal_active = bool(getattr(backtest, "latest_signal_active", False))
    target_reached = bool(getattr(backtest, "target_reached", False))

    positive_lines: list[str] = []
    negative_lines: list[str] = []

    if probability_up >= 65:
        positive_lines.append(f"未来上涨概率 {probability_up:.1f}% 处于高位")
    elif probability_up <= 42:
        negative_lines.append(f"未来上涨概率仅 {probability_up:.1f}%，偏弱")

    if enhanced_attention_score >= 75:
        positive_lines.append(f"增强分数 {enhanced_attention_score:.1f} 进入强关注区")
    elif enhanced_attention_score <= 55:
        negative_lines.append(f"增强分数 {enhanced_attention_score:.1f} 尚未进入强关注区")

    if quant_score >= 68:
        positive_lines.append(f"量化辅助 {quant_score:.1f}，趋势与结构配合较好")
    elif quant_score <= 45:
        negative_lines.append(f"量化辅助仅 {quant_score:.1f}，结构优势不足")

    if sector_score >= 65:
        positive_lines.append("板块热度对个股有正向共振")
    elif sector_score <= 42:
        negative_lines.append("板块热度支撑偏弱")

    if fund_score >= 65:
        positive_lines.append("主力资金承接偏强")
    elif fund_score <= 42:
        negative_lines.append("主力资金承接偏弱")

    if news_score >= 65:
        positive_lines.append("消息面偏暖，催化方向偏正面")
    elif news_score <= 42:
        negative_lines.append("消息面偏谨慎，缺少强催化")

    technical_adjustment = 0.0
    if close_vs_ma20 >= 0.05:
        technical_adjustment += 4.5
        positive_lines.append("当前收盘显著站上 MA20")
    elif close_vs_ma20 >= 0.02:
        technical_adjustment += 2.5
        positive_lines.append("当前收盘站上 MA20")
    elif close_vs_ma20 <= -0.05:
        technical_adjustment -= 5.5
        negative_lines.append("当前收盘明显跌破 MA20")
    elif close_vs_ma20 <= -0.02:
        technical_adjustment -= 3.0
        negative_lines.append("当前收盘回到 MA20 下方")

    if -0.015 <= breakout_distance <= 0.03:
        technical_adjustment += 3.5
        positive_lines.append("价格位于平台突破位或前高附近")
    elif breakout_distance > 0.03:
        technical_adjustment += 1.2
        positive_lines.append("价格已经站上平台突破位")
    elif breakout_distance <= -0.05:
        technical_adjustment -= 3.5
        negative_lines.append("距离平台突破位仍偏远")
    elif breakout_distance <= -0.02:
        technical_adjustment -= 1.5
        negative_lines.append("尚未回到平台突破位附近")

    intraday_adjustment = 0.0
    if intraday_tone >= 2:
        intraday_adjustment += 6.0
        positive_lines.append("分时承接与执行结构偏强")
    elif intraday_tone == 1:
        intraday_adjustment += 3.0
        positive_lines.append("分时仍有承接")
    elif intraday_tone <= -2:
        intraday_adjustment -= 7.0
        negative_lines.append("分时走势偏弱，盘中承接不足")
    elif intraday_tone == -1:
        intraday_adjustment -= 3.0
        negative_lines.append("分时强度一般，暂未形成强上攻")

    launch_window_view = _build_launch_window_view(
        {
            **display_context,
            "snapshot": snapshot,
            "sector_score": sector_score,
            "fund_score": fund_score,
            "news_score": news_score,
            "launch_score": _safe_float(detail.get("launch_score"), 50.0),
            "launch_readiness_score": _safe_float(
                display_context.get("launch_readiness_score"),
                _safe_float(detail.get("launch_readiness_score"), 50.0),
            ),
            "market_resonance_score": _safe_float(
                display_context.get("market_resonance_score"),
                _safe_float(detail.get("market_resonance_score"), 50.0),
            ),
            "launch_specialist_score": _safe_float(
                display_context.get("launch_specialist_score"),
                _safe_float(detail.get("launch_specialist_score"), 50.0),
            ),
            "launch_regime_fit_score": _safe_float(
                display_context.get("launch_regime_fit_score"),
                _safe_float(detail.get("launch_regime_fit_score"), 50.0),
            ),
            "launch_specialist_confidence": _safe_float(
                display_context.get("launch_specialist_confidence"),
                _safe_float(detail.get("launch_specialist_confidence"), 50.0),
            ),
            "stage": detail.get("stage"),
            "stage_label": display_context.get("stage_label", _signal_value(detail.get("stage"), "label", "")),
        },
        stage_code=str(_signal_value(detail.get("stage"), "code", "")),
        intraday_bias=intraday_tone,
    )
    launch_window_score = _safe_float(launch_window_view.get("launch_window_score"), 50.0)
    launch_window_confidence = _safe_float(launch_window_view.get("launch_window_confidence"), 50.0)
    launch_window_confidence_weight = _selection_launch_window_confidence_weight(display_context)
    launch_window_status = str(launch_window_view.get("launch_window_status", "非启动窗"))

    if launch_window_score >= 72:
        positive_lines.append(f"{launch_window_status}，更像主升刚启动或刚确认")
    elif launch_window_score >= 62:
        positive_lines.append(f"{launch_window_status}，启动结构正在靠拢")
    elif launch_window_score <= 40:
        negative_lines.append(f"{launch_window_status}，位置与结构更偏风险防守")

    backtest_adjustment = 0.0
    if latest_signal_active:
        backtest_adjustment += 7.0
        positive_lines.append("历史回测最新信号已放行")
    elif target_reached:
        backtest_adjustment += 3.5
        positive_lines.append("历史回测达到目标精度")
    else:
        backtest_adjustment -= 4.0
        negative_lines.append("历史回测暂未达到稳定放行门槛")

    if precision_priority >= 3:
        backtest_adjustment += 3.0
        positive_lines.append(f"{precision_label}，样本 {precision_support} 笔")
    elif precision_priority == 2:
        backtest_adjustment += 1.5
        positive_lines.append(f"{precision_label}，命中率 {precision_precision * 100:.1f}%")
    elif achieved_precision_pct and achieved_precision_pct < 55:
        backtest_adjustment -= 2.0
        negative_lines.append(f"历史回测命中率仅 {achieved_precision_pct:.1f}%")

    selection_score = (
        probability_up * 0.30
        + min(predicted_upside_pct, 30.0) * 0.10
        + enhanced_attention_score * 0.22
        + base_attention_score * 0.10
        + quant_score * 0.12
        + sector_score * 0.07
        + fund_score * 0.08
        + news_score * 0.05
        + tomorrow_confidence * 0.06
        + technical_adjustment
        + intraday_adjustment
        + (launch_window_score - 50.0) * 0.24
        + (launch_window_confidence - 50.0) * launch_window_confidence_weight
        + backtest_adjustment
        + float(strategy_profile["attention_bonus"]) * 1.2
    )
    selection_score = float(max(0.0, min(selection_score, 100.0)))
    selection_confidence = float(
        max(
            35.0,
            min(
                98.0,
                46.0 + abs(selection_score - 50.0) * 0.9 + max(0, precision_priority - 1) * 6.0 + abs(intraday_tone) * 3.0,
            ),
        )
    )
    execution_view = _build_execution_view(
        {
            **display_context,
            "snapshot": snapshot,
            "sector_score": sector_score,
            "fund_score": fund_score,
            "news_score": news_score,
            "stage": detail.get("stage"),
            "stage_code": str(_signal_value(detail.get("stage"), "code", "")),
            "stage_label": display_context.get("stage_label", _signal_value(detail.get("stage"), "label", "")),
            "intraday": detail.get("intraday"),
            "intraday_structure_signal": detail.get("intraday_structure_signal"),
        },
        launch_window_view=launch_window_view,
        stage_code=str(_signal_value(detail.get("stage"), "code", "")),
        stage_label=display_context.get("stage_label", _signal_value(detail.get("stage"), "label", "")),
        intraday_bias=intraday_tone,
    )
    execution_score = _safe_float(execution_view.get("execution_score"), 50.0)
    execution_confidence = _safe_float(execution_view.get("execution_confidence"), 50.0)
    execution_label = str(execution_view.get("execution_label", "等待结构"))
    execution_window = str(execution_view.get("execution_window", "信号未合流"))
    action_execution_weight = _action_execution_weight(display_context)
    action_selection_weight = 1.0 - action_execution_weight
    action_score = float(
        max(0.0, min(selection_score * action_selection_weight + execution_score * action_execution_weight, 100.0))
    )
    action_confidence = float(
        max(
            35.0,
            min(98.0, selection_confidence * 0.54 + execution_confidence * 0.46 + max(0, precision_priority - 1) * 2.5),
        )
    )

    can_open_new_position = latest_signal_active or target_reached or precision_priority >= 2
    is_buy = (
        selection_score >= 70
        and execution_score >= 68
        and action_score >= 72
        and probability_up >= float(strategy_profile["buy_threshold_probability"])
        and predicted_upside_pct >= float(strategy_profile["buy_threshold_upside"])
        and enhanced_attention_score >= 68
        and quant_score >= 58
        and close_vs_ma20 >= -0.005
        and intraday_tone >= 0
        and launch_window_score >= 62
        and launch_window_status != "高位风险窗"
        and execution_label == "可执行"
        and can_open_new_position
    )
    is_sell = (
        action_score <= 42
        or execution_score <= 36
        or (probability_up <= 40 and enhanced_attention_score <= 55 and intraday_tone < 0)
        or (close_vs_ma20 <= -0.03 and quant_score <= 45 and fund_score <= 45)
        or (launch_window_status == "高位风险窗" and launch_window_score <= 45)
        or (execution_label == "暂不执行" and execution_score <= 40 and probability_up <= 45)
    )
    is_hold = (
        not is_buy
        and not is_sell
        and selection_score >= 58
        and action_score >= 58
        and probability_up >= float(strategy_profile["hold_threshold_probability"])
        and predicted_upside_pct >= float(strategy_profile["hold_threshold_upside"])
        and enhanced_attention_score >= 60
        and intraday_tone > -2
        and launch_window_score >= 54
        and execution_score >= 52
    )

    reason_lines: list[str]
    if is_buy:
        action_label = "买"
        action_css_class = "buy"
        reason_lines = _unique_lines(
            positive_lines[:4]
            + [f"启动窗口：{launch_window_status}（{launch_window_score:.1f}）"]
            + [f"执行窗口：{execution_window}（{execution_score:.1f}）"]
            + [str(execution_view.get("execution_entry_zone", ""))]
            + [f'盈亏比：{str(execution_view.get("reward_risk_label", ""))} / 预期收益 {float(execution_view.get("expected_return_pct", 0.0)):.1f}%']
            + [str(strategy_profile["candidate_strategy_note"])]
            + ["等待明日买点触发后再执行，避免脱离平台位置追价"]
        )
        action_reason = f'{strategy_profile["candidate_strategy_short_label"]}口径下，既是强候选，也进入了可执行区。'
    elif is_sell:
        action_label = "卖"
        action_css_class = "sell"
        reason_lines = _unique_lines(
            negative_lines[:4]
            + [f"启动窗口：{launch_window_status}（{launch_window_score:.1f}）"]
            + [f"执行层：{execution_label} / {execution_window}"]
            + [str(execution_view.get("execution_invalidation_rule", ""))]
            + ["优先按防守位和卖点脚本处理，先控制回撤"]
        )
        action_reason = "候选质量与执行条件同时转弱，当前更适合防守而不是继续进攻。"
    elif is_hold:
        action_label = "持"
        action_css_class = "hold"
        reason_lines = _unique_lines(
            positive_lines[:3]
            + negative_lines[:1]
            + [f"启动窗口：{launch_window_status}（{launch_window_score:.1f}）"]
            + [f"执行层：{execution_label} / {execution_window}"]
            + [str(strategy_profile["candidate_strategy_note"])]
            + ["趋势尚未被破坏，更适合已有仓位继续跟踪，不宜激进追高"]
        )
        action_reason = f'{strategy_profile["candidate_strategy_short_label"]}口径下，选股质量仍在，但执行优势没有强到必须加仓。'
    else:
        action_label = "观察"
        action_css_class = "watch"
        reason_lines = _unique_lines(
            positive_lines[:2]
            + negative_lines[:2]
            + [f"启动窗口：{launch_window_status}（{launch_window_score:.1f}）"]
            + [f"执行层：{execution_label} / {execution_window}"]
            + [str(execution_view.get("execution_entry_zone", ""))]
            + [str(strategy_profile["candidate_strategy_note"])]
            + ["等待分时承接、平台突破或资金回流进一步确认"]
        )
        action_reason = f'{strategy_profile["candidate_strategy_short_label"]}口径下，这只票值得盯，但今天还不算理想执行点。'

    return {
        "action_label": action_label,
        "action_css_class": action_css_class,
        "selection_score": selection_score,
        "selection_confidence": selection_confidence,
        "predicted_upside_pct": predicted_upside_pct,
        "tomorrow_plan_confidence": tomorrow_confidence,
        "sector_score": sector_score,
        "fund_score": fund_score,
        "news_score": news_score,
        "technical_adjustment": technical_adjustment,
        "intraday_adjustment": intraday_adjustment,
        "backtest_adjustment": backtest_adjustment,
        "action_score": action_score,
        "action_confidence": action_confidence,
        "action_badge": f"{action_label} 路 {action_score:.1f}",
        "action_execution_weight": action_execution_weight,
        "action_reason": action_reason,
        "action_reason_lines": reason_lines[:5],
        "launch_window_label": str(launch_window_view.get("launch_window_label", "等待确认")),
        "launch_window_status": launch_window_status,
        "launch_window_summary": str(launch_window_view.get("launch_window_summary", "")),
        "launch_window_score": launch_window_score,
        "launch_window_confidence": launch_window_confidence,
        "launch_window_drivers": str(launch_window_view.get("launch_window_drivers", "")),
        **execution_view,
    }


def _board_precision_priority(row: dict[str, object]) -> int:
    priority = int(_safe_float(row.get("precision_priority"), 0))
    label_text = str(row.get("precision_gate_label", "") or "")
    if "90%" in label_text or "放行" in label_text:
        return max(priority, 3)
    if "回测" in label_text or "高精度" in label_text or "达标" in label_text:
        return max(priority, 2)
    return priority


def _evaluate_board_action(row: dict | pd.Series) -> dict[str, object]:
    payload = dict(row) if isinstance(row, (dict, pd.Series)) else {}
    probability_up = _safe_float(payload.get("probability_up"))
    predicted_upside_pct = _safe_float(payload.get("predicted_upside_pct"))
    if predicted_upside_pct <= 0:
        predicted_upside_pct = max((probability_up - 40.0) * 0.18, 0.0)
    base_attention_score = _safe_float(payload.get("attention_score"))
    enhanced_attention_score = _safe_float(payload.get("enhanced_attention_score"), base_attention_score)
    quant_score = _safe_float(payload.get("quant_score"))
    tomorrow_confidence = _safe_float(payload.get("tomorrow_plan_confidence"), 50.0)
    sector_score = _safe_float(payload.get("sector_score"), 50.0)
    fund_score = _safe_float(payload.get("fund_score"), 50.0)
    news_score = _safe_float(payload.get("news_score"), 50.0)
    precision_priority = _board_precision_priority(payload)
    strategy_profile = _candidate_strategy_profile(payload.get("candidate_strategy"))
    tone_text = " ".join(
        str(payload.get(key, ""))
        for key in (
            "stage_label",
            "tomorrow_setup",
            "tomorrow_bias",
            "sector_label",
            "fund_label",
            "news_label",
            "intraday_label",
            "reason",
        )
        if str(payload.get(key, "")).strip()
    )
    tone = _text_signal_bias(tone_text)
    launch_window_view = _build_launch_window_view(payload, intraday_bias=tone)
    launch_window_score = _safe_float(launch_window_view.get("launch_window_score"), 50.0)
    launch_window_confidence = _safe_float(launch_window_view.get("launch_window_confidence"), 50.0)
    launch_window_confidence_weight = _selection_launch_window_confidence_weight(payload)
    launch_window_status = str(launch_window_view.get("launch_window_status", "非启动窗"))
    selection_score = (
        probability_up * 0.32
        + min(predicted_upside_pct, 30.0) * 0.10
        + enhanced_attention_score * 0.22
        + base_attention_score * 0.10
        + quant_score * 0.12
        + tomorrow_confidence * 0.06
        + (launch_window_score - 50.0) * 0.24
        + (launch_window_confidence - 50.0) * launch_window_confidence_weight
        + precision_priority * 4.2
        + tone * 3.5
        + float(strategy_profile["attention_bonus"]) * 1.3
    )
    selection_score = float(max(0.0, min(selection_score, 100.0)))
    selection_confidence = float(
        max(
            35.0,
            min(97.0, 45.0 + abs(selection_score - 50.0) * 0.86 + precision_priority * 4.5 + abs(tone) * 2.2),
        )
    )
    execution_view = _build_execution_view(
        payload,
        launch_window_view=launch_window_view,
        intraday_bias=tone,
    )
    execution_score = _safe_float(execution_view.get("execution_score"), 50.0)
    execution_confidence = _safe_float(execution_view.get("execution_confidence"), 50.0)
    execution_label = str(execution_view.get("execution_label", "等待结构"))
    execution_window = str(execution_view.get("execution_window", "信号未合流"))
    action_execution_weight = _action_execution_weight(payload)
    action_selection_weight = 1.0 - action_execution_weight
    action_score = float(
        max(0.0, min(selection_score * action_selection_weight + execution_score * action_execution_weight, 100.0))
    )
    action_confidence = float(
        max(
            35.0,
            min(97.0, selection_confidence * 0.56 + execution_confidence * 0.44 + precision_priority * 1.8),
        )
    )

    if (
        selection_score >= 70
        and execution_score >= 66
        and action_score >= 72
        and probability_up >= float(strategy_profile["buy_threshold_probability"])
        and predicted_upside_pct >= float(strategy_profile["buy_threshold_upside"])
        and enhanced_attention_score >= 66
        and quant_score >= 58
        and tone >= 0
        and launch_window_score >= 62
        and launch_window_status != "高位风险窗"
        and precision_priority >= 1
        and execution_label == "可执行"
    ):
        action_label = "买"
        action_css_class = "buy"
        action_reason = f"候选质量与执行条件同时转强，当前属于{launch_window_status}，且处于{execution_window}。"
    elif (
        action_score <= 42
        or execution_score <= 36
        or (probability_up <= 40 and enhanced_attention_score <= 55)
        or (quant_score <= 45 and tone < 0)
        or (launch_window_status == "高位风险窗" and launch_window_score <= 45)
        or (execution_label == "暂不执行" and probability_up <= 45)
    ):
        action_label = "卖"
        action_css_class = "sell"
        action_reason = f"候选质量或执行层明显转弱，当前更像{launch_window_status}，宜按防守思路处理。"
    elif (
        selection_score >= 58
        and execution_score >= 52
        and action_score >= 58
        and probability_up >= float(strategy_profile["hold_threshold_probability"])
        and predicted_upside_pct >= float(strategy_profile["hold_threshold_upside"])
        and enhanced_attention_score >= 60
        and tone > -2
        and launch_window_score >= 54
    ):
        action_label = "持"
        action_css_class = "hold"
        action_reason = f'{strategy_profile["candidate_strategy_short_label"]}口径下，候选质量仍在，当前属于{launch_window_status}，但执行优势更偏持有跟踪。'
    else:
        action_label = "观察"
        action_css_class = "watch"
        action_reason = f'{strategy_profile["candidate_strategy_short_label"]}口径下，这只票值得盯，但当前更接近{execution_window}，先等执行信号合流。'

    return {
        "action_label": action_label,
        "action_css_class": action_css_class,
        "selection_score": selection_score,
        "selection_confidence": selection_confidence,
        "predicted_upside_pct": predicted_upside_pct,
        "tomorrow_plan_confidence": tomorrow_confidence,
        "sector_score": sector_score,
        "fund_score": fund_score,
        "news_score": news_score,
        "technical_adjustment": 0.0,
        "intraday_adjustment": 0.0,
        "backtest_adjustment": 0.0,
        "action_score": action_score,
        "action_confidence": action_confidence,
        "action_badge": f"{action_label} 路 {action_score:.1f}",
        "action_execution_weight": action_execution_weight,
        "action_reason": action_reason,
        "launch_window_label": str(launch_window_view.get("launch_window_label", "等待确认")),
        "launch_window_status": launch_window_status,
        "launch_window_summary": str(launch_window_view.get("launch_window_summary", "")),
        "launch_window_score": launch_window_score,
        "launch_window_confidence": launch_window_confidence,
        "launch_window_drivers": str(launch_window_view.get("launch_window_drivers", "")),
        **execution_view,
    }


def _build_backtest_status_from_model(model_result) -> SimpleNamespace:
    precision_gate_precision = float(getattr(model_result, "precision_gate_precision", 0.0) or 0.0)
    precision_gate_threshold = float(getattr(model_result, "precision_gate_threshold", 1.0) or 1.0)
    precision_gate_support = int(getattr(model_result, "precision_gate_support", 0) or 0)
    precision_gate_active = bool(getattr(model_result, "precision_gate_active", False))
    precision_gate_label = str(getattr(model_result, "precision_gate_label", "未做90%精度认证") or "未做90%精度认证")
    target_reached = precision_gate_active or (precision_gate_precision >= 0.90 and precision_gate_support > 0)
    status_label = "统一模型代理认证"
    if precision_gate_active:
        status_label = "高精度放行"
    elif target_reached:
        status_label = "历史精度达标"
    return SimpleNamespace(
        achieved_precision=precision_gate_precision,
        selected_threshold=precision_gate_threshold,
        trade_count=precision_gate_support,
        latest_signal_active=precision_gate_active,
        target_reached=target_reached,
        status_label=status_label,
        selection_summary=precision_gate_label,
        summary=str(getattr(model_result, "backtest_summary", "") or precision_gate_label),
    )


def _format_reason(stage_label: str, snapshot: dict[str, float], quant_signal_name: str) -> str:
    reasons = [stage_label, quant_signal_name]
    if snapshot["close_vs_ma20"] > 0:
        reasons.append("站上20日线")
    if snapshot["volume_ratio_5"] > 1.15:
        reasons.append("量能放大")
    return " / ".join(reasons[:4])


def _consecutive_up_days(daily: pd.DataFrame) -> int:
    if daily.empty or "close" not in daily.columns:
        return 0
    close = pd.to_numeric(daily["close"], errors="coerce")
    if close.dropna().shape[0] < 2:
        return 0
    daily_change = close.diff()
    streak = 0
    for value in reversed(daily_change.fillna(0.0).tolist()):
        if float(value) > 0:
            streak += 1
            continue
        break
    return int(streak)


def _filter_focus_candidates(
    board_df: pd.DataFrame,
    min_consecutive_up_days: int = MIN_FOCUS_CONSECUTIVE_UP_DAYS,
) -> pd.DataFrame:
    if board_df.empty or "consecutive_up_days" not in board_df.columns:
        return board_df.copy()
    filtered = board_df[board_df["consecutive_up_days"] >= int(min_consecutive_up_days)].copy()
    return filtered.reset_index(drop=True)


def _fallback_focus_candidates(board_df: pd.DataFrame) -> pd.DataFrame:
    if board_df.empty or "consecutive_up_days" not in board_df.columns:
        return board_df.copy()
    return (
        board_df.sort_values(
            ["consecutive_up_days", "attention_score", "probability_up", "amount"],
            ascending=False,
        )
        .reset_index(drop=True)
        .copy()
    )


def _is_main_board_security(symbol: str, market: str = "") -> bool:
    clean_symbol = str(symbol or "").strip()
    market_text = str(market or "").strip()
    if market_text:
        if "创业" in market_text or "科创" in market_text or "北交" in market_text:
            return False
        if "涓绘澘" in market_text or "中小板" in market_text:
            return True
    return clean_symbol.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))


def _rolling_pullback_metrics(window_df: pd.DataFrame) -> dict[str, object]:
    if window_df.empty or len(window_df) < 8:
        return {
            "pullback_days": 0,
            "pullback_volume_decay": False,
            "pullback_kept_ma10": False,
        }

    recent = window_df.tail(7).reset_index(drop=True)
    pre_today = recent.iloc[:-1].copy()
    if pre_today.empty:
        return {
            "pullback_days": 0,
            "pullback_volume_decay": False,
            "pullback_kept_ma10": False,
        }

    peak_idx = int(pre_today["close"].astype(float).idxmax())
    pullback = pre_today.loc[peak_idx + 1 :].copy()
    pullback_days = int(len(pullback))
    if pullback_days == 0:
        return {
            "pullback_days": 0,
            "pullback_volume_decay": False,
            "pullback_kept_ma10": False,
        }

    volume_decay = bool(
        pullback_days >= 3
        and float(pullback["vol"].iloc[-1]) <= float(pullback["vol"].iloc[0])
        and float(pullback["vol"].mean()) <= float(pre_today["vol"].iloc[: peak_idx + 1].mean() or 0.0)
    )
    kept_ma10 = bool((pullback["close"] >= pullback["ma10"]).fillna(False).all())
    return {
        "pullback_days": pullback_days,
        "pullback_volume_decay": volume_decay,
        "pullback_kept_ma10": kept_ma10,
    }


def _build_strategy_snapshot_context(market_data_date: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    safe_date = str(market_data_date or "").replace("-", "")
    try:
        trade_dates = fetch_tushare_recent_trade_dates(end_date=safe_date or None, limit=3)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()
    if not trade_dates:
        return pd.DataFrame(), pd.DataFrame()

    latest_trade_date = trade_dates[-1]
    previous_trade_date = trade_dates[-2] if len(trade_dates) >= 2 else latest_trade_date
    try:
        latest_snapshot = fetch_tushare_daily_snapshot(latest_trade_date)
        previous_snapshot = fetch_tushare_daily_snapshot(previous_trade_date)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()
    return latest_snapshot, previous_snapshot


def _align_daily_history_to_market_date(
    daily: pd.DataFrame,
    market_data_date: str | None,
    *,
    require_exact: bool = True,
) -> pd.DataFrame:
    if not isinstance(daily, pd.DataFrame) or daily.empty:
        return pd.DataFrame()
    if "date" in daily.columns:
        aligned = daily.reset_index(drop=True).copy()
    else:
        aligned = daily.reset_index().rename(columns={"index": "date"}).copy()
    if "date" not in aligned.columns:
        return pd.DataFrame()
    aligned["date"] = pd.to_datetime(aligned["date"], errors="coerce")
    aligned = aligned.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if aligned.empty or not market_data_date:
        return aligned
    market_ts = pd.to_datetime(market_data_date, errors="coerce")
    if pd.isna(market_ts):
        return aligned
    aligned = aligned[aligned["date"] <= market_ts].reset_index(drop=True)
    if aligned.empty:
        return pd.DataFrame()
    if require_exact and aligned["date"].max() != market_ts:
        return pd.DataFrame()
    return aligned


def _build_strategy_candidate_pool(universe: pd.DataFrame, market_data_date: str | None) -> pd.DataFrame:
    if universe.empty:
        return pd.DataFrame()
    try:
        candidate_pool = build_market_candidate_pool_store(universe, market_data_date)
    except Exception:
        return pd.DataFrame()
    if candidate_pool is None or candidate_pool.empty:
        return pd.DataFrame()
    return candidate_pool.copy()


def _build_symbol_base_analysis_from_feature_store_row(
    feature_row: dict[str, object],
    *,
    symbol: str,
    name: str,
    daily: pd.DataFrame,
) -> dict[str, object]:
    snapshot = dict(feature_row.get("snapshot") or {})
    latest_features = {
        **dict(feature_row.get("latest_features") or {}),
        **dict(feature_row.get("model_feature_values") or {}),
    }
    stage = feature_row.get("stage_object")
    quant_signal = feature_row.get("quant_signal_object")
    rule_context = feature_row.get("rule_context_object")
    latest_daily = daily.iloc[-1]
    return {
        "symbol": symbol,
        "name": name,
        "daily": daily,
        "snapshot": snapshot,
        "stage": stage,
        "stage_label": feature_row.get("stage_label", ""),
        "stage_priority": feature_row.get("stage_priority", ""),
        "stage_summary": feature_row.get("stage_summary", ""),
        "stage_score": float(feature_row.get("stage_score", 0.0) or 0.0),
        "launch_score": float(feature_row.get("launch_score", 50.0) or 50.0),
        "launch_readiness_score": float(feature_row.get("launch_readiness_score", latest_features.get("launch_readiness", 50.0)) or 50.0),
        "market_resonance_score": float(feature_row.get("market_resonance_score", latest_features.get("market_resonance", 50.0)) or 50.0),
        "latest_features": latest_features,
        "quant_signal": quant_signal,
        "quant_score": float(feature_row.get("quant_score", 0.0) or 0.0),
        "quant_primary_signal": str(feature_row.get("quant_primary_signal", "") or ""),
        "board_label": str(feature_row.get("board_label", "") or ""),
        "price_limit_label": str(feature_row.get("price_limit_label", "") or ""),
        "rule_summary": str(feature_row.get("rule_summary", "") or ""),
        "rule_context": rule_context,
        "latest_price": round(float(feature_row.get("latest_price", latest_daily.get("close", 0.0)) or 0.0), 2),
        "change_pct": round(float(feature_row.get("change_pct", latest_daily.get("change_pct", snapshot.get("change_pct", 0.0))) or 0.0), 2),
        "amount": round(float(feature_row.get("amount", latest_daily.get("amount", 0.0)) or 0.0), 2),
        "turnover": round(float(feature_row.get("turnover", latest_daily.get("turnover", 0.0)) or 0.0), 2),
        "consecutive_up_days": int(feature_row.get("consecutive_up_days", 0) or 0),
        "analysis_date": str(feature_row.get("analysis_date", snapshot.get("date", "")) or ""),
    }


def _symbol_base_analysis_cache_path(symbol: str, market_data_date: str | None) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_date = str(market_data_date or "unknown").replace("-", "")
    return CACHE_DIR / f"symbol_base_analysis_{safe_date}_{symbol}.pkl"


def _read_symbol_base_analysis_disk_cache(symbol: str, market_data_date: str | None) -> dict | None:
    cache_path = _symbol_base_analysis_cache_path(symbol, market_data_date)
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    if payload.get("market_data_date") != market_data_date:
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    return data


def _write_symbol_base_analysis_disk_cache(symbol: str, market_data_date: str | None, data: dict) -> None:
    payload = {
        "market_data_date": market_data_date,
        "data": data,
    }
    with _symbol_base_analysis_cache_path(symbol, market_data_date).open("wb") as handle:
        pickle.dump(payload, handle)


@lru_cache(maxsize=8192)
def _prepare_symbol_base_analysis(
    symbol: str,
    name: str,
    start_date: str = FULL_MARKET_HISTORY_START,
    market_data_date: str | None = None,
) -> dict | None:
    if market_data_date:
        cached = _read_symbol_base_analysis_disk_cache(symbol, market_data_date)
        if cached is not None:
            return cached
    feature_row = get_market_daily_feature_row(symbol, market_data_date) if market_data_date else None
    raw_daily = fetch_daily_history(symbol=symbol, start_date=start_date)
    daily = store_align_daily_history_to_market_date(raw_daily, market_data_date, require_exact=bool(market_data_date))
    if daily.empty:
        return None
    if feature_row is not None:
        result = _build_symbol_base_analysis_from_feature_store_row(
            feature_row,
            symbol=symbol,
            name=name,
            daily=daily,
        )
        if market_data_date and str(result["analysis_date"]) == str(market_data_date):
            _write_symbol_base_analysis_disk_cache(symbol, market_data_date, result)
        return result
    features = build_daily_features(daily)
    valid_features = features.dropna()
    if valid_features.empty:
        return None

    snapshot = latest_snapshot(daily, features)
    stage = classify_stage(daily)
    quant_signal = evaluate_quant_signal(daily, features)
    rule_context = build_trading_rule_context(symbol=symbol, name=name)
    latest_features = valid_features.iloc[-1]
    launch_score = main_rise_start_score(latest_features)
    latest_daily = daily.iloc[-1]
    result = {
        "symbol": symbol,
        "name": name,
        "daily": daily,
        "snapshot": snapshot,
        "stage": stage,
        "stage_label": stage.label,
        "stage_priority": stage.priority,
        "stage_summary": stage.structure_summary,
        "stage_score": stage_numeric_score(stage, latest_features),
        "launch_score": float(launch_score),
        "latest_features": latest_features.to_dict(),
        "quant_score": float(quant_signal.total_score),
        "quant_primary_signal": quant_signal.primary_signal,
        "board_label": rule_context.board_label,
        "price_limit_label": rule_context.price_limit_label,
        "rule_summary": rule_context.rule_summary,
        "latest_price": round(float(latest_daily["close"]), 2),
        "change_pct": round(float(latest_daily.get("change_pct", snapshot["change_pct"])), 2),
        "amount": round(float(latest_daily.get("amount", 0.0)), 2),
        "turnover": round(float(latest_daily.get("turnover", 0.0)), 2),
        "consecutive_up_days": _consecutive_up_days(daily),
        "analysis_date": snapshot["date"],
    }
    if market_data_date and str(snapshot["date"]) == str(market_data_date):
        _write_symbol_base_analysis_disk_cache(symbol, market_data_date, result)
    return result


def _clear_symbol_base_analysis_cache() -> None:
    _prepare_symbol_base_analysis.cache_clear()
    _local_precision_certification.cache_clear()


def _load_market_model_or_none(horizon_days: int, positive_return: float):
    try:
        return load_cached_market_wide_model(
            horizon_days=horizon_days,
            positive_return=positive_return,
            train_start=GLOBAL_MODEL_TRAIN_START,
            train_end=GLOBAL_MODEL_TRAIN_END,
            test_start=GLOBAL_MODEL_TEST_START,
            test_end=GLOBAL_MODEL_TEST_END,
        )
    except Exception:
        return None


def _load_market_proxy_or_none(horizon_days: int, positive_return: float):
    try:
        return load_market_proxy_model(
            horizon_days=horizon_days,
            positive_return=positive_return,
            train_start=GLOBAL_MODEL_TRAIN_START,
            train_end=GLOBAL_MODEL_TRAIN_END,
            test_start=GLOBAL_MODEL_TEST_START,
            test_end=GLOBAL_MODEL_TEST_END,
        )
    except Exception:
        return None


def _market_model_status(horizon_days: int, positive_return: float) -> dict[str, object]:
    try:
        return get_market_wide_model_status(
            horizon_days=horizon_days,
            positive_return=positive_return,
            train_start=GLOBAL_MODEL_TRAIN_START,
            train_end=GLOBAL_MODEL_TRAIN_END,
            test_start=GLOBAL_MODEL_TEST_START,
            test_end=GLOBAL_MODEL_TEST_END,
        )
    except Exception:
        return {"model_ready": False, "partial_ready": False}


def _resolve_model_source(market_model=None, market_proxy_model=None) -> tuple[str, str]:
    if market_model is not None:
        return "market_wide", "全市场模型"
    if market_proxy_model is not None:
        return "market_proxy", "快速代理模型"
    return "local_fast_fallback", "本地快速回退"


def _build_model_result_status(
    analysis_date: str | None,
    latest_market_data_date: str | None,
    *,
    cache_stale: bool = False,
) -> str:
    analysis = str(analysis_date or "").strip()
    latest = str(latest_market_data_date or "").strip()
    if analysis and latest and analysis == latest and not cache_stale:
        return "最新结果"
    if analysis and latest and analysis != latest:
        return f"非最新结果({analysis})"
    if analysis:
        return f"已按 {analysis} 计算"
    return "结果日期待确认"


def _load_candidate_live_context(symbol: str) -> dict[str, pd.DataFrame]:
    context: dict[str, pd.DataFrame] = {
        "minute": pd.DataFrame(),
        "fund_flow": pd.DataFrame(),
        "news": pd.DataFrame(),
    }
    fetchers = {
        "minute": lambda: fetch_minute_history(symbol),
        "fund_flow": lambda: fetch_stock_main_fund_flow(symbol, limit=10),
        "news": lambda: fetch_stock_news(symbol, limit=8),
    }
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(fetcher): key for key, fetcher in fetchers.items()}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    value = future.result()
                except Exception:
                    value = pd.DataFrame()
                context[key] = value if isinstance(value, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return context
    return context


def _analyze_single_base(
    symbol: str,
    name: str,
    horizon_days: int,
    positive_return: float,
    start_date: str = FULL_MARKET_HISTORY_START,
    market_model=None,
    market_proxy_model=None,
    market_data_date: str | None = None,
    market_context_future: Future | None = None,
    candidate_industry_name: str | None = None,
    candidate_strategy: str | None = None,
    candidate_reason: str | None = None,
) -> dict | None:
    try:
        model_source_key, model_source_label = _resolve_model_source(market_model, market_proxy_model)
        base = _prepare_symbol_base_analysis(
            symbol=symbol,
            name=name,
            start_date=start_date,
            market_data_date=market_data_date,
        )
        if base is None:
            return None
        model_result = None
        if market_model is not None:
            model_result = score_with_market_wide_model(
                base["daily"],
                market_model,
                latest_feature_values=base["latest_features"],
            )
        elif market_proxy_model is not None:
            model_result = score_with_market_proxy_model(
                base["daily"],
                market_proxy_model,
                latest_feature_values=base["latest_features"],
            )
        latest_probability = (
            float(model_result.latest_probability)
            if model_result is not None
            else predict_latest_probability(
                base["daily"],
                horizon_days=horizon_days,
                positive_return=positive_return,
                market_model=market_model,
                market_proxy_model=market_proxy_model,
                latest_feature_values=base["latest_features"],
                allow_slow_fallback=False,
            )
        )
        predicted_upside_pct = (
            float(getattr(model_result, "predicted_upside_pct", 0.0) or 0.0)
            if model_result is not None
            else round(max(positive_return * 100 * (0.55 + latest_probability), 0.0), 2)
        )
        predicted_upside_low_pct = (
            float(getattr(model_result, "predicted_upside_low_pct", 0.0) or 0.0)
            if model_result is not None
            else round(max(predicted_upside_pct * 0.62, 0.0), 2)
        )
        predicted_upside_high_pct = (
            float(getattr(model_result, "predicted_upside_high_pct", 0.0) or 0.0)
            if model_result is not None
            else round(max(predicted_upside_pct * 1.28, predicted_upside_pct), 2)
        )
        industry_name = str(candidate_industry_name or "").strip()
        sector_signal = {
            "sector_score": 50.0,
            "sector_label": "行业热度待补充",
            "sector_summary": "基础榜单正在等待板块资金上下文。",
        }
        if industry_name:
            prefetched_context = _resolve_prefetched_market_context(market_context_future)
            industry_flow = prefetched_context.get("industry_flow") if isinstance(prefetched_context, dict) else None
            if isinstance(industry_flow, pd.DataFrame) and not industry_flow.empty:
                sector_signal = compute_sector_hot_score(industry_name, industry_flow)
                if model_result is not None:
                    model_result = apply_sector_fund_probability_upgrade(model_result, sector_signal=sector_signal)
                    latest_probability = float(model_result.latest_probability)
                    predicted_upside_pct = float(getattr(model_result, "predicted_upside_pct", predicted_upside_pct) or predicted_upside_pct)
                    predicted_upside_low_pct = float(
                        getattr(model_result, "predicted_upside_low_pct", predicted_upside_low_pct) or predicted_upside_low_pct
                    )
                    predicted_upside_high_pct = float(
                        getattr(model_result, "predicted_upside_high_pct", predicted_upside_high_pct) or predicted_upside_high_pct
                    )
                else:
                    sector_upgrade = build_sector_fund_probability_upgrade(
                        latest_probability,
                        sector_signal=sector_signal,
                    )
                    sector_upgraded_probability = float(sector_upgrade["upgraded_probability"])
                    probability_shift = sector_upgraded_probability - float(sector_upgrade["base_probability"])
                    latest_probability = sector_upgraded_probability
                    predicted_upside_pct = max(min(predicted_upside_pct * (1.0 + probability_shift * 0.55), 60.0), 0.0)
                    predicted_upside_low_pct = max(
                        min(predicted_upside_low_pct * (1.0 + probability_shift * 0.38), predicted_upside_pct),
                        0.0,
                    )
                    predicted_upside_high_pct = max(
                        min(max(predicted_upside_high_pct * (1.0 + probability_shift * 0.70), predicted_upside_pct), 80.0),
                        predicted_upside_pct,
                    )
        full_context_prediction_active = False
        probability_upgrade_note = ""
        intraday_state = {"label": "鍒嗘椂寰呮洿鏂?", "score": 0.5}
        fund_signal = {"label": "涓诲姏璧勯噾寰呮洿鏂?", "fund_score": 50.0, "confidence_score": 55.0}
        news_signal = {"label": "娑堟伅闈㈠緟鏇存柊", "sentiment_score": 50.0, "confidence_score": 55.0}
        live_context_score = 50.0
        intraday_execution_score = 50.0
        temporal_news_score = 50.0
        try:
            live_context = _load_candidate_live_context(symbol)
            minute_df = live_context.get("minute", pd.DataFrame())
            fund_flow_df = live_context.get("fund_flow", pd.DataFrame())
            news_df = live_context.get("news", pd.DataFrame())
            has_live_context = any(
                isinstance(frame, pd.DataFrame) and not frame.empty
                for frame in (minute_df, fund_flow_df, news_df)
            )
            intraday_state = evaluate_intraday(minute_df)
            fund_signal = evaluate_main_fund_signal(fund_flow_df)
            news_signal = build_research_enhanced_news_signal(
                news_df,
                base_signal=evaluate_news_sentiment(news_df),
                symbol=symbol,
            )
            if has_live_context:
                if model_result is not None:
                    model_result = apply_live_probability_upgrade(
                        model_result,
                        base["daily"],
                        latest_feature_values=base.get("latest_features"),
                        minute_df=minute_df,
                        news_df=news_df,
                        fund_flow_df=fund_flow_df,
                        symbol=symbol,
                    )
                    latest_probability = float(model_result.latest_probability)
                    predicted_upside_pct = float(getattr(model_result, "predicted_upside_pct", predicted_upside_pct) or predicted_upside_pct)
                    predicted_upside_low_pct = float(
                        getattr(model_result, "predicted_upside_low_pct", predicted_upside_low_pct) or predicted_upside_low_pct
                    )
                    predicted_upside_high_pct = float(
                        getattr(model_result, "predicted_upside_high_pct", predicted_upside_high_pct) or predicted_upside_high_pct
                    )
                    probability_upgrade_note = str(getattr(model_result, "upgrade_summary", "") or "")
                    live_context_score = _safe_float(getattr(model_result, "upgrade_components", {}).get("live_context_score"), 50.0)
                    intraday_execution_score = _safe_float(
                        getattr(model_result, "upgrade_components", {}).get("intraday_execution_score"),
                        50.0,
                    )
                    temporal_news_score = _safe_float(getattr(model_result, "upgrade_components", {}).get("temporal_news_score"), 50.0)
                else:
                    base_probability_before_live = latest_probability
                    live_upgrade = build_live_probability_upgrade(
                        latest_probability,
                        base["daily"],
                        latest_feature_values=base.get("latest_features"),
                        minute_df=minute_df,
                        news_df=news_df,
                        fund_flow_df=fund_flow_df,
                        symbol=symbol,
                    )
                    latest_probability = float(live_upgrade.get("upgraded_probability", latest_probability))
                    probability_shift = latest_probability - float(live_upgrade.get("base_probability", base_probability_before_live))
                    predicted_upside_pct = max(min(predicted_upside_pct * (1.0 + probability_shift * 0.55), 60.0), 0.0)
                    predicted_upside_low_pct = max(
                        min(predicted_upside_low_pct * (1.0 + probability_shift * 0.38), predicted_upside_pct),
                        0.0,
                    )
                    predicted_upside_high_pct = max(
                        min(max(predicted_upside_high_pct * (1.0 + probability_shift * 0.70), predicted_upside_pct), 80.0),
                        predicted_upside_pct,
                    )
                    probability_upgrade_note = str(live_upgrade.get("summary", "") or "")
                    live_context_score = _safe_float(live_upgrade.get("live_context_score"), 50.0)
                    intraday_execution_score = _safe_float(live_upgrade.get("intraday_execution_score"), 50.0)
                    temporal_news_score = _safe_float(live_upgrade.get("temporal_news_score"), 50.0)
                full_context_prediction_active = True
        except Exception:
            full_context_prediction_active = False

        base_attention_score = _score_base_attention(
            latest_probability,
            float(base["stage_score"]),
            base["snapshot"],
            float(base["quant_score"]),
            float(base.get("launch_score", 50.0) or 50.0),
        )
        precision_priority, precision_gate_label, precision_gate_threshold, precision_gate_precision, precision_gate_support = (
            _precision_priority_from_model_result(model_result)
        )
        tomorrow_plan = build_tomorrow_plan(
            base["stage"],
            base["snapshot"],
            base["latest_features"],
            latest_probability,
            float(base["quant_score"]),
        )
        latest_feature_map = dict(base.get("latest_features", {}) or {})
        snapshot = dict(base.get("snapshot", {}) or {})
        signal_breakdown = dict(getattr(model_result, "signal_breakdown", {}) or {})
        launch_readiness_score = _safe_float(
            signal_breakdown.get("launch_readiness_score"),
            _safe_float(base.get("launch_readiness_score"), _safe_float(latest_feature_map.get("launch_readiness"), 50.0)),
        )
        market_resonance_score = _safe_float(
            signal_breakdown.get("market_resonance_score"),
            _safe_float(base.get("market_resonance_score"), _safe_float(latest_feature_map.get("market_resonance"), 50.0)),
        )
        launch_specialist_score = _safe_float(signal_breakdown.get("launch_specialist_score"), 50.0)
        launch_regime_fit_score = _safe_float(signal_breakdown.get("launch_regime_fit_score"), 50.0)
        launch_specialist_confidence = _safe_float(signal_breakdown.get("launch_specialist_confidence"), 50.0)
        result = {
            "symbol": symbol,
            "name": name,
            "board_label": base["board_label"],
            "price_limit_label": base["price_limit_label"],
            "latest_price": base["latest_price"],
            "change_pct": base["change_pct"],
            "amount": base["amount"],
            "turnover": base["turnover"],
            "industry_name": industry_name or "unknown",
            "sector_label": str(sector_signal.get("sector_label", "sector pending")),
            "sector_score": round(float(sector_signal.get("sector_score", 50.0) or 50.0), 2),
            "fund_label": str(fund_signal.get("label", "fund pending")),
            "fund_score": round(_safe_float(fund_signal.get("fund_score"), 50.0), 2),
            "news_label": str(news_signal.get("label", "news pending")),
            "news_score": round(_safe_float(news_signal.get("sentiment_score"), 50.0), 2),
            "news_impact_score": round(_safe_float(news_signal.get("research_impact_score"), 50.0), 2),
            "news_impact_confidence_score": round(_safe_float(news_signal.get("research_impact_confidence_score"), 25.0), 2),
            "news_expected_excess_return_1d_pct": round(
                _safe_float(news_signal.get("research_expected_excess_return_1d_pct"), 0.0),
                4,
            ),
            "news_expected_excess_return_3d_pct": round(
                _safe_float(news_signal.get("research_expected_excess_return_3d_pct"), 0.0),
                4,
            ),
            "news_expected_excess_return_5d_pct": round(
                _safe_float(news_signal.get("research_expected_excess_return_5d_pct"), 0.0),
                4,
            ),
            "news_research_primary_category": str(news_signal.get("research_primary_category", "general")),
            "news_research_event_count": int(news_signal.get("research_event_count", 0) or 0),
            "intraday_label": str(intraday_state.get("label", "intraday pending")),
            "intraday_score": round(_safe_float(intraday_state.get("score"), 0.5) * 100, 2),
            "live_context_score": round(live_context_score, 2),
            "intraday_execution_score": round(intraday_execution_score, 2),
            "temporal_news_score": round(temporal_news_score, 2),
            "full_context_prediction_active": bool(full_context_prediction_active),
            "probability_upgrade_note": probability_upgrade_note,
            "consecutive_up_days": int(base["consecutive_up_days"]),
            "attention_score": round(base_attention_score, 2),
            "probability_up": round(float(latest_probability * 100), 2),
            "predicted_upside_pct": round(predicted_upside_pct, 2),
            "predicted_upside_low_pct": round(predicted_upside_low_pct, 2),
            "predicted_upside_high_pct": round(predicted_upside_high_pct, 2),
            "quant_score": round(float(base["quant_score"]), 2),
            "launch_score": round(float(base.get("launch_score", 50.0) or 50.0), 2),
            "launch_readiness_score": round(launch_readiness_score, 2),
            "market_resonance_score": round(market_resonance_score, 2),
            "launch_specialist_score": round(launch_specialist_score, 2),
            "launch_regime_fit_score": round(launch_regime_fit_score, 2),
            "launch_specialist_confidence": round(launch_specialist_confidence, 2),
            "precision_priority": int(precision_priority),
            "precision_gate_label": precision_gate_label,
            "precision_gate_threshold": round(float(precision_gate_threshold), 2),
            "precision_gate_precision": round(float(precision_gate_precision * 100), 2),
            "precision_gate_support": int(precision_gate_support),
            "stage_label": base["stage_label"],
            "stage_priority": base["stage_priority"],
            "stage_summary": base["stage_summary"],
            "tomorrow_bias": tomorrow_plan.bias,
            "tomorrow_setup": tomorrow_plan.setup_label,
            "tomorrow_buy_point": tomorrow_plan.buy_point,
            "tomorrow_sell_point": tomorrow_plan.sell_point,
            "tomorrow_plan_confidence": tomorrow_plan.confidence,
            "reason": _format_reason(base["stage_label"], base["snapshot"], str(base["quant_primary_signal"])),
            "candidate_strategy": str(candidate_strategy or ""),
            "candidate_reason": str(candidate_reason or ""),
            "analysis_date": base["analysis_date"],
            "model_source": model_source_key,
            "model_source_label": model_source_label,
            "model_result_status": _build_model_result_status(base["analysis_date"], market_data_date),
            "market_regime_label": str(latest_feature_map.get("market_regime_label", "")),
            "market_ret_5": round(_safe_float(latest_feature_map.get("market_ret_5")), 6),
            "market_ret_20": round(_safe_float(latest_feature_map.get("market_ret_20")), 6),
            "market_close_vs_ma20": round(_safe_float(latest_feature_map.get("market_close_vs_ma20")), 6),
            "market_volatility_10": round(_safe_float(latest_feature_map.get("market_volatility_10")), 6),
            "market_range_position_20": round(_safe_float(latest_feature_map.get("market_range_position_20"), 0.5), 6),
            "ret_20": round(_safe_float(latest_feature_map.get("ret_20")), 6),
            "close_vs_ma20": round(_safe_float(snapshot.get("close_vs_ma20")), 6),
            "breakout_distance_20": round(_safe_float(snapshot.get("breakout_distance_20")), 6),
            "range_position_20": round(_safe_float(snapshot.get("range_position_20"), 0.5), 6),
            "volume_ratio_5": round(_safe_float(snapshot.get("volume_ratio_5"), 1.0), 6),
            "upper_shadow_ratio": round(_safe_float(snapshot.get("upper_shadow_ratio")), 6),
            "stretch_risk": round(_safe_float(latest_feature_map.get("stretch_risk")), 6),
            "risk_pressure": round(_safe_float(latest_feature_map.get("risk_pressure")), 6),
        }
        result.update(
            _build_launch_window_view(
                result,
                stage_code=str(getattr(base["stage"], "code", "")),
                stage_label=str(base["stage_label"]),
            )
        )
        return _apply_candidate_strategy_prediction_profile(result)
    except Exception:
        return None


def _build_symbol_model_result(
    daily: pd.DataFrame,
    *,
    horizon_days: int,
    positive_return: float,
    latest_feature_values: pd.Series | dict[str, float] | None = None,
):
    market_model = _load_market_model_or_none(horizon_days, positive_return)
    market_proxy_model = None if market_model is not None else _load_market_proxy_or_none(horizon_days, positive_return)
    model_source_key, model_source_label = _resolve_model_source(market_model, market_proxy_model)
    if market_model is not None:
        model_result = score_with_market_wide_model(
            daily,
            market_model,
            latest_feature_values=latest_feature_values,
        )
    elif market_proxy_model is not None:
        model_result = score_with_market_proxy_model(
            daily,
            market_proxy_model,
            latest_feature_values=latest_feature_values,
        )
    else:
        model_result = train_probability_model(
            daily,
            horizon_days=horizon_days,
            positive_return=positive_return,
        )
    return model_result, market_model, market_proxy_model, model_source_key, model_source_label


def _enrich_candidate(
    candidate: dict,
    industry_flow: pd.DataFrame,
    horizon_days: int,
    positive_return: float,
) -> dict:
    enriched = candidate.copy()
    symbol = candidate["symbol"]
    try:
        profile = fetch_stock_profile(symbol)
        industry_name = str(profile.get("琛屼笟", "鏈煡"))
        sector_signal = compute_sector_hot_score(industry_name, industry_flow)
        fund_flow_df = fetch_stock_main_fund_flow(symbol, limit=10)
        fund_signal = evaluate_main_fund_signal(fund_flow_df)
        news_df = fetch_stock_news(symbol, limit=8)
        news_signal = build_research_enhanced_news_signal(
            news_df,
            base_signal=evaluate_news_sentiment(news_df),
            symbol=symbol,
        )
        market_data_date = str(candidate.get("analysis_date", "") or "")
        base = _prepare_symbol_base_analysis(
            symbol=symbol,
            name=str(candidate.get("name", symbol)),
            market_data_date=market_data_date,
        )
        local_certification = _local_precision_certification(
            symbol=symbol,
            name=str(candidate.get("name", symbol)),
            horizon_days=horizon_days,
            positive_return=positive_return,
            market_data_date=market_data_date,
        )
        minute = fetch_minute_history(symbol)
        intraday = evaluate_intraday(minute)
        intraday_signal = evaluate_intraday_structure_signal(minute)
        probability_up = float(local_certification.get("probability_up") or candidate.get("probability_up", 0.0))
        predicted_upside_pct = float(
            local_certification.get("predicted_upside_pct") or candidate.get("predicted_upside_pct", 0.0) or 0.0
        )
        predicted_upside_low_pct = float(
            local_certification.get("predicted_upside_low_pct") or candidate.get("predicted_upside_low_pct", 0.0) or 0.0
        )
        predicted_upside_high_pct = float(
            local_certification.get("predicted_upside_high_pct") or candidate.get("predicted_upside_high_pct", 0.0) or 0.0
        )
        live_already_applied = bool(candidate.get("full_context_prediction_active", False))
        if live_already_applied:
            probability_up = float(candidate.get("probability_up", probability_up) or probability_up)
            predicted_upside_pct = float(candidate.get("predicted_upside_pct", predicted_upside_pct) or predicted_upside_pct)
            predicted_upside_low_pct = float(candidate.get("predicted_upside_low_pct", predicted_upside_low_pct) or predicted_upside_low_pct)
            predicted_upside_high_pct = float(candidate.get("predicted_upside_high_pct", predicted_upside_high_pct) or predicted_upside_high_pct)
        base_attention_score = float(candidate["attention_score"])
        precision_payload: dict[str, object] = {}
        if base is not None and bool(local_certification.get("certification_ready")):
            if "daily" in base and not live_already_applied:
                base_probability_up = probability_up
                live_upgrade = build_live_probability_upgrade(
                    probability_up / 100,
                    base["daily"],
                    latest_feature_values=base.get("latest_features"),
                    minute_df=minute,
                    news_df=news_df,
                    fund_flow_df=fund_flow_df,
                    symbol=symbol,
                )
                probability_up = float(live_upgrade["upgraded_probability"]) * 100
                if predicted_upside_pct > 0 and base_probability_up > 0:
                    probability_ratio = probability_up / max(base_probability_up, 1e-6)
                    predicted_upside_pct = round(float(max(predicted_upside_pct * (0.78 + probability_ratio * 0.22), 0.0)), 2)
                    predicted_upside_low_pct = round(
                        float(max(predicted_upside_low_pct * (0.82 + probability_ratio * 0.18), 0.0)),
                        2,
                    )
                    predicted_upside_high_pct = round(
                        float(max(predicted_upside_high_pct * (0.72 + probability_ratio * 0.28), predicted_upside_pct)),
                        2,
                    )
            else:
                live_upgrade = {"summary": str(candidate.get("probability_upgrade_note", "") or "")}
            base_attention_score = _score_base_attention(
                probability_up / 100,
                float(base["stage_score"]),
                dict(base["snapshot"]),
                float(base["quant_score"]),
                float(base.get("launch_score", 50.0) or 50.0),
            )
            precision_payload = {
                "attention_score": round(base_attention_score, 2),
                "probability_up": round(probability_up, 2),
                "predicted_upside_pct": round(predicted_upside_pct, 2),
                "predicted_upside_low_pct": round(predicted_upside_low_pct, 2),
                "predicted_upside_high_pct": round(predicted_upside_high_pct, 2),
                "launch_score": round(float(base.get("launch_score", 50.0) or 50.0), 2),
                "precision_priority": int(local_certification.get("precision_priority", 0) or 0),
                "precision_gate_label": str(local_certification.get("precision_gate_label", "未做90%精度认证")),
                "precision_gate_threshold": round(float(local_certification.get("precision_gate_threshold", 1.0) or 1.0), 2),
                "precision_gate_precision": round(float(local_certification.get("precision_gate_precision", 0.0) or 0.0), 2),
                "precision_gate_support": int(local_certification.get("precision_gate_support", 0) or 0),
                "probability_upgrade_note": str(live_upgrade.get("summary", "")),
            }
        enhanced_attention_score = _score_final_attention(
            base_attention_score,
            float(sector_signal["sector_score"]),
            float(fund_signal["fund_score"]),
            float(news_signal["sentiment_score"]),
            fund_confidence=float(fund_signal.get("confidence_score", 55.0)),
            news_confidence=float(news_signal.get("confidence_score", 55.0)),
        )
        tomorrow_payload: dict[str, object] = {}
        if base is not None:
            probability = probability_up / 100
            tomorrow_plan = build_tomorrow_plan(
                base["stage"],
                base["snapshot"],
                base["latest_features"],
                probability,
                float(candidate.get("quant_score", base["quant_score"])),
                intraday_state=intraday,
                intraday_signal=intraday_signal,
            )
            tomorrow_payload = {
                "stage_priority": base["stage_priority"],
                "tomorrow_bias": tomorrow_plan.bias,
                "tomorrow_setup": tomorrow_plan.setup_label,
                "tomorrow_buy_point": tomorrow_plan.buy_point,
                "tomorrow_sell_point": tomorrow_plan.sell_point,
                "tomorrow_plan_confidence": tomorrow_plan.confidence,
            }
        enriched.update(
            {
                "industry_name": industry_name,
                "sector_label": str(sector_signal["sector_label"]),
                "fund_label": str(fund_signal["label"]),
                "news_label": str(news_signal["label"]),
                "news_score": round(_safe_float(news_signal.get("sentiment_score"), 50.0), 2),
                "news_impact_score": round(_safe_float(news_signal.get("research_impact_score"), 50.0), 2),
                "news_impact_confidence_score": round(_safe_float(news_signal.get("research_impact_confidence_score"), 25.0), 2),
                "news_expected_excess_return_1d_pct": round(
                    _safe_float(news_signal.get("research_expected_excess_return_1d_pct"), 0.0),
                    4,
                ),
                "news_expected_excess_return_5d_pct": round(
                    _safe_float(news_signal.get("research_expected_excess_return_5d_pct"), 0.0),
                    4,
                ),
                "news_research_primary_category": str(news_signal.get("research_primary_category", "general")),
                "news_research_event_count": int(news_signal.get("research_event_count", 0) or 0),
                "enhanced_attention_score": round(enhanced_attention_score, 2),
                "predicted_upside_pct": round(predicted_upside_pct, 2),
                "predicted_upside_low_pct": round(predicted_upside_low_pct, 2),
                "predicted_upside_high_pct": round(predicted_upside_high_pct, 2),
                "intraday_label": str(intraday.get("label", "分时待更新")),
                **precision_payload,
                **tomorrow_payload,
            }
        )
    except Exception:
        enriched.update(
            {
                "industry_name": "未知",
                "sector_label": "行业热度未知",
                "fund_label": "主力资金中性",
                "news_label": "消息面中性",
                "enhanced_attention_score": candidate["attention_score"],
            }
        )
    return enriched


def _ranking_cache_path(horizon_days: int, positive_return: float) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"market_rankings_v{MARKET_RANKING_CACHE_VERSION}_h{horizon_days}_r{int(positive_return * 10000)}.pkl"


def _extract_market_data_date(board_df: pd.DataFrame) -> str | None:
    if board_df.empty or "analysis_date" not in board_df.columns:
        return None
    analysis_dates = pd.to_datetime(board_df["analysis_date"], errors="coerce").dropna()
    if analysis_dates.empty:
        return None
    return analysis_dates.max().strftime("%Y-%m-%d")


def _latest_market_close_date(reference_symbol: str = "600519") -> str | None:
    reference_symbols = list(
        dict.fromkeys(
            [
                str(reference_symbol or "").strip(),
                "000001",
                "600519",
                "601398",
                "000333",
                "300750",
            ]
        )
    )
    latest_dates: list[pd.Timestamp] = []
    for symbol in reference_symbols:
        if not symbol:
            continue
        try:
            daily = fetch_daily_history(symbol=symbol, start_date="20250101")
        except Exception:
            continue
        if daily.empty:
            continue
        parsed = pd.to_datetime(daily["date"], errors="coerce").dropna()
        if parsed.empty:
            continue
        latest_dates.append(parsed.max())
    if not latest_dates:
        return None
    return max(latest_dates).strftime("%Y-%m-%d")


def _dynamic_fallback_candidate_cache_path(market_data_date: str | None) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_date = str(market_data_date or "unknown").replace("-", "")
    return CACHE_DIR / f"dynamic_fallback_candidates_v{DYNAMIC_FALLBACK_CACHE_VERSION}_{safe_date}.pkl"


def _read_dynamic_fallback_candidate_cache(market_data_date: str | None) -> pd.DataFrame | None:
    cache_path = _dynamic_fallback_candidate_cache_path(market_data_date)
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    meta = payload.get("meta", {})
    df = payload.get("data")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if meta.get("cache_version") != DYNAMIC_FALLBACK_CACHE_VERSION:
        return None
    if meta.get("market_data_date") != market_data_date:
        return None
    return df


def _write_dynamic_fallback_candidate_cache(pool_df: pd.DataFrame, market_data_date: str | None) -> None:
    payload = {
        "meta": {
            "cache_version": DYNAMIC_FALLBACK_CACHE_VERSION,
            "market_data_date": market_data_date,
            "row_count": int(len(pool_df)),
        },
        "data": pool_df,
    }
    with _dynamic_fallback_candidate_cache_path(market_data_date).open("wb") as handle:
        pickle.dump(payload, handle)


def _strategy_candidate_cache_path(market_data_date: str | None) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_date = str(market_data_date or "unknown").replace("-", "")
    return CACHE_DIR / f"strategy_candidates_v{STRATEGY_CANDIDATE_CACHE_VERSION}_{safe_date}.pkl"


def _read_strategy_candidate_cache(market_data_date: str | None) -> pd.DataFrame | None:
    cache_path = _strategy_candidate_cache_path(market_data_date)
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    meta = payload.get("meta", {})
    df = payload.get("data")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if meta.get("cache_version") != STRATEGY_CANDIDATE_CACHE_VERSION:
        return None
    if meta.get("market_data_date") != market_data_date:
        return None
    return df


def _write_strategy_candidate_cache(pool_df: pd.DataFrame, market_data_date: str | None) -> None:
    payload = {
        "meta": {
            "cache_version": STRATEGY_CANDIDATE_CACHE_VERSION,
            "market_data_date": market_data_date,
            "row_count": int(len(pool_df)),
        },
        "data": pool_df,
    }
    with _strategy_candidate_cache_path(market_data_date).open("wb") as handle:
        pickle.dump(payload, handle)


def _candidate_analysis_cache_path(
    symbol: str,
    market_data_date: str | None,
    horizon_days: int,
    positive_return: float,
    model_source: str,
    candidate_industry_name: str | None,
) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_date = str(market_data_date or "unknown").replace("-", "")
    safe_symbol = str(symbol).zfill(6)
    safe_model_source = "".join(ch for ch in str(model_source or "unknown") if ch.isalnum() or ch in ("_", "-")) or "unknown"
    safe_positive_return = f"{float(positive_return):.4f}".replace(".", "_")
    industry_digest = hashlib.md5(str(candidate_industry_name or "").strip().encode("utf-8")).hexdigest()[:10]
    filename = (
        f"candidate_analysis_v{MARKET_CANDIDATE_ANALYSIS_CACHE_VERSION}_"
        f"{safe_date}_{safe_symbol}_h{int(horizon_days)}_r{safe_positive_return}_{safe_model_source}_{industry_digest}.pkl"
    )
    return CACHE_DIR / filename


def _read_candidate_analysis_cache(
    symbol: str,
    market_data_date: str | None,
    horizon_days: int,
    positive_return: float,
    model_source: str,
    candidate_industry_name: str | None,
) -> dict | None:
    cache_path = _candidate_analysis_cache_path(
        symbol,
        market_data_date,
        horizon_days,
        positive_return,
        model_source,
        candidate_industry_name,
    )
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    meta = payload.get("meta", {})
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    if meta.get("cache_version") != MARKET_CANDIDATE_ANALYSIS_CACHE_VERSION:
        return None
    if meta.get("market_data_date") != market_data_date:
        return None
    if int(meta.get("horizon_days", -1)) != int(horizon_days):
        return None
    if float(meta.get("positive_return", -1.0)) != float(positive_return):
        return None
    if str(meta.get("model_source") or "") != str(model_source or ""):
        return None
    if int(meta.get("model_schema_version", -1)) != int(MODEL_SCHEMA_VERSION):
        return None
    return data


def _write_candidate_analysis_cache(
    symbol: str,
    market_data_date: str | None,
    horizon_days: int,
    positive_return: float,
    model_source: str,
    candidate_industry_name: str | None,
    data: dict,
) -> None:
    payload = {
        "meta": {
            "cache_version": MARKET_CANDIDATE_ANALYSIS_CACHE_VERSION,
            "market_data_date": market_data_date,
            "horizon_days": int(horizon_days),
            "positive_return": float(positive_return),
            "model_source": str(model_source or ""),
            "model_schema_version": int(MODEL_SCHEMA_VERSION),
        },
        "data": data,
    }
    with _candidate_analysis_cache_path(
        symbol,
        market_data_date,
        horizon_days,
        positive_return,
        model_source,
        candidate_industry_name,
    ).open("wb") as handle:
        pickle.dump(payload, handle)


def _read_market_rankings_cache(
    horizon_days: int,
    positive_return: float,
    *,
    allow_stale: bool = False,
) -> tuple[pd.DataFrame | None, dict]:
    cache_path = _ranking_cache_path(horizon_days, positive_return)
    if not cache_path.exists():
        return None, {}
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None, {}

    meta = payload.get("meta", {})
    df = payload.get("data")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None, {}
    if meta.get("cache_version") != MARKET_RANKING_CACHE_VERSION:
        return None, {}
    if meta.get("horizon_days") != horizon_days or meta.get("positive_return") != positive_return:
        return None, {}
    cached_market_data_date = str(meta.get("market_data_date") or _extract_market_data_date(df) or "")
    latest_market_data_date = _latest_market_close_date()
    if latest_market_data_date and cached_market_data_date and cached_market_data_date != latest_market_data_date:
        if not allow_stale:
            return None, {}
        meta["cache_stale"] = True
    else:
        meta["cache_stale"] = False
    meta["market_data_date"] = cached_market_data_date or latest_market_data_date
    meta["latest_market_data_date"] = latest_market_data_date or cached_market_data_date
    meta["computed_at"] = str(meta.get("computed_at") or "")
    meta["model_source"] = str(meta.get("model_source") or "")
    meta["model_source_label"] = str(meta.get("model_source_label") or "")
    return df, meta


def _write_market_rankings_cache(
    df: pd.DataFrame,
    horizon_days: int,
    positive_return: float,
    data_mode: str,
) -> None:
    market_data_date = _extract_market_data_date(df)
    payload = {
        "meta": {
            "cache_version": MARKET_RANKING_CACHE_VERSION,
            "cache_date": dt.date.today().isoformat(),
            "market_data_date": market_data_date,
            "horizon_days": horizon_days,
            "positive_return": positive_return,
            "data_mode": data_mode,
            "row_count": int(len(df)),
            "computed_at": str(df.attrs.get("computed_at") or ""),
            "model_source": str(df.attrs.get("model_source") or ""),
            "model_source_label": str(df.attrs.get("model_source_label") or ""),
            "model_schema_version": int(df.attrs.get("model_schema_version", MODEL_SCHEMA_VERSION)),
        },
        "data": df,
    }
    with _ranking_cache_path(horizon_days, positive_return).open("wb") as handle:
        pickle.dump(payload, handle)


def _market_context_cache_path(market_data_date: str | None) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_date = str(market_data_date or "unknown").replace("-", "")
    return CACHE_DIR / f"market_context_{safe_date}.pkl"


def _read_market_context_cache(market_data_date: str | None) -> dict[str, pd.DataFrame] | None:
    cache_path = _market_context_cache_path(market_data_date)
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    if payload.get("market_data_date") != market_data_date:
        return None
    context = payload.get("context")
    if not isinstance(context, dict):
        return None
    if not all(isinstance(value, pd.DataFrame) for value in context.values()):
        return None
    return context


def _write_market_context_cache(market_data_date: str | None, context: dict[str, pd.DataFrame]) -> None:
    payload = {
        "market_data_date": market_data_date,
        "context": context,
    }
    with _market_context_cache_path(market_data_date).open("wb") as handle:
        pickle.dump(payload, handle)


def _merge_live_market_snapshot(rank_df: pd.DataFrame, spot_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if spot_df.empty or "data_mode" not in spot_df.columns:
        return rank_df, "history"
    data_mode = str(spot_df["data_mode"].iloc[0])
    if data_mode != "live" or len(spot_df) < 1000:
        return rank_df, "history"

    live_view = spot_df[["symbol", "latest_price", "change_pct", "amount", "turnover"]].drop_duplicates("symbol")
    merged = rank_df.drop(columns=["latest_price", "change_pct", "amount", "turnover"], errors="ignore").merge(
        live_view,
        on="symbol",
        how="left",
    )
    for column in ["latest_price", "change_pct", "amount", "turnover"]:
        merged[column] = merged[column].fillna(rank_df[column] if column in rank_df.columns else 0.0)
    return merged, "live"


def _normalize_board_names(board_df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(board_df, pd.DataFrame) or board_df.empty or "symbol" not in board_df.columns:
        return board_df

    board = board_df.copy()
    board["symbol"] = board["symbol"].astype(str).str.zfill(6)
    if "name" not in board.columns:
        board["name"] = board["symbol"]

    raw_name = board["name"].fillna("").astype(str).str.strip()
    invalid_name_mask = (
        raw_name.eq("")
        | raw_name.eq(board["symbol"])
        | raw_name.str.fullmatch(r"\d{6}(?:\.[A-Z]{2})?", na=False)
    )
    if not bool(invalid_name_mask.any()):
        return board

    universe = load_a_share_universe()
    if not isinstance(universe, pd.DataFrame) or universe.empty or not {"symbol", "name"}.issubset(universe.columns):
        board["name"] = raw_name.where(~invalid_name_mask, board["symbol"])
        return board

    name_lookup = universe[["symbol", "name"]].drop_duplicates("symbol").copy()
    name_lookup["symbol"] = name_lookup["symbol"].astype(str).str.zfill(6)
    name_lookup["name"] = name_lookup["name"].fillna("").astype(str).str.strip()
    name_map = name_lookup.set_index("symbol")["name"]
    fallback_name = board["symbol"].map(name_map).fillna("").astype(str).str.strip()
    board["name"] = raw_name.where(~invalid_name_mask, fallback_name)
    board["name"] = board["name"].fillna("").astype(str).str.strip().replace("", pd.NA).fillna(board["symbol"])
    return board


def _sort_focus_board(
    board_df: pd.DataFrame,
    ranking_by: str,
    board_size: int,
    optimization_profile: dict[str, object] | None = None,
) -> pd.DataFrame:
    if board_df.empty:
        return board_df.copy()
    board = _apply_replay_calibration_to_board(board_df, optimization_profile)
    board = _ensure_launch_window_columns(board, force=True)
    has_precision_priority = "precision_priority" in board.columns
    if ranking_by == "上涨概率":
        board["ranking_score"] = pd.to_numeric(board["probability_up"], errors="coerce").fillna(0.0)
        sort_cols = ["ranking_score", "launch_window_score", "attention_score", "amount"]
        if has_precision_priority:
            sort_cols.insert(0, "precision_priority")
    else:
        has_adaptive_profile = bool(optimization_profile) and int(optimization_profile.get("review_days", 0) or 0) > 0
        if has_adaptive_profile:
            board["adaptive_attention_score"] = compute_adaptive_rank_score(board, optimization_profile)
            board["ranking_score"] = pd.to_numeric(board["adaptive_attention_score"], errors="coerce").fillna(0.0)
            sort_cols = ["ranking_score", "launch_window_score", "attention_score", "probability_up", "amount"]
        else:
            board["ranking_score"] = pd.to_numeric(board["attention_score"], errors="coerce").fillna(0.0)
            sort_cols = ["ranking_score", "launch_window_score", "probability_up", "amount"]
        if has_precision_priority:
            sort_cols.insert(0, "precision_priority")
    board = board.sort_values(sort_cols, ascending=False).head(board_size).reset_index(drop=True).copy()
    board = board.drop(columns=["rank"], errors="ignore")
    board.insert(0, "rank", range(1, len(board) + 1))
    return board


def _build_dynamic_fallback_candidate_pool(universe: pd.DataFrame, market_data_date: str | None) -> pd.DataFrame:
    try:
        pool_df = build_market_dynamic_fallback_pool_store(universe, market_data_date)
    except Exception:
        return pd.DataFrame()
    if pool_df is None or pool_df.empty:
        return pd.DataFrame()
    return pool_df.copy()


def _fallback_candidate_pool(universe: pd.DataFrame) -> list[tuple[str, str]]:
    market_data_date = _latest_market_close_date()
    strategy_pool = _build_strategy_candidate_pool(universe, market_data_date)
    if not strategy_pool.empty:
        return list(strategy_pool[["symbol", "name"]].itertuples(index=False, name=None))
    dynamic_pool = _build_dynamic_fallback_candidate_pool(universe, market_data_date)
    if not dynamic_pool.empty:
        return list(dynamic_pool[["symbol", "name"]].itertuples(index=False, name=None))

    fallback_symbols = list(dict.fromkeys([*FALLBACK_CANDIDATE_SYMBOLS, *FALLBACK_WATCHLIST.keys()]))
    pool = universe[universe["symbol"].isin(fallback_symbols)][["symbol", "name"]].drop_duplicates("symbol").copy()
    if pool.empty:
        pool = universe.head(DYNAMIC_FALLBACK_POOL_SIZE)[["symbol", "name"]].copy()
    return list(pool.itertuples(index=False, name=None))


def _call_analyze_single_base(
    symbol: str,
    name: str,
    horizon_days: int,
    positive_return: float,
    start_date: str,
    market_model,
    market_proxy_model,
    market_data_date: str | None,
    market_context_future: Future | None = None,
    candidate_industry_name: str | None = None,
    candidate_strategy: str | None = None,
    candidate_reason: str | None = None,
):
    try:
        parameters = inspect.signature(_analyze_single_base).parameters
    except (TypeError, ValueError):
        parameters = {}
    if (
        "market_context_future" in parameters
        or "candidate_industry_name" in parameters
        or "candidate_strategy" in parameters
        or "candidate_reason" in parameters
        or len(parameters) >= 10
    ):
        return _analyze_single_base(
            symbol,
            name,
            horizon_days,
            positive_return,
            start_date,
            market_model,
            market_proxy_model,
            market_data_date,
            market_context_future,
            candidate_industry_name,
            candidate_strategy,
            candidate_reason,
        )
    return _analyze_single_base(
        symbol,
        name,
        horizon_days,
        positive_return,
        start_date,
        market_model,
        market_proxy_model,
        market_data_date,
    )


def _load_or_compute_candidate_analysis(
    symbol: str,
    name: str,
    horizon_days: int,
    positive_return: float,
    start_date: str,
    market_model,
    market_proxy_model,
    market_data_date: str | None,
    market_context_future: Future | None = None,
    candidate_industry_name: str | None = None,
    candidate_strategy: str | None = None,
    candidate_reason: str | None = None,
) -> dict | None:
    model_source_key, _ = _resolve_model_source(market_model, market_proxy_model)
    cached = _read_candidate_analysis_cache(
        symbol,
        market_data_date,
        horizon_days,
        positive_return,
        model_source_key,
        candidate_industry_name,
    )
    if cached is not None:
        return dict(cached)

    result = _call_analyze_single_base(
        symbol,
        name,
        horizon_days,
        positive_return,
        start_date,
        market_model,
        market_proxy_model,
        market_data_date,
        market_context_future,
        candidate_industry_name,
        candidate_strategy,
        candidate_reason,
    )
    if result is None:
        return None
    if market_data_date and str(result.get("analysis_date", "") or "") == str(market_data_date):
        _write_candidate_analysis_cache(
            symbol,
            market_data_date,
            horizon_days,
            positive_return,
            model_source_key,
            candidate_industry_name,
            result,
        )
    return result


def _build_ranked_market_snapshot(
    horizon_days: int,
    positive_return: float,
    *,
    progress_callback=None,
) -> tuple[pd.DataFrame, str]:
    if progress_callback is not None:
        progress_callback("准备候选池", 0, 1, "正在读取股票池与交易日数据")
    universe = load_a_share_universe()[["symbol", "name"]].drop_duplicates("symbol").copy()
    market_model = _load_market_model_or_none(horizon_days, positive_return)
    market_proxy_model = None if market_model is not None else _load_market_proxy_or_none(horizon_days, positive_return)
    model_source_key, model_source_label = _resolve_model_source(market_model, market_proxy_model)
    market_data_date = _latest_market_close_date()
    market_context_future = MARKET_CONTEXT_PREFETCH_EXECUTOR.submit(_build_market_context)
    if progress_callback is not None:
        progress_callback("沉淀市场特征", 0, 1, "正在装载全市场日线特征 store")
    feature_store = build_market_daily_feature_store(
        universe,
        market_data_date,
        progress_callback=progress_callback,
    )
    if progress_callback is not None:
        progress_callback(
            "沉淀市场特征",
            1,
            1,
            f"全市场特征 store 已就绪，覆盖 {len(feature_store) if isinstance(feature_store, pd.DataFrame) else 0} 只股票",
        )
    # Rebuild rankings from close-based data only. The user explicitly does not
    # need real-time spot updates here, and the full-market snapshot endpoint is
    # both slow and unstable on this machine.
    spot = pd.DataFrame()
    strategy_pool = build_market_candidate_pool_store(
        universe,
        market_data_date,
        feature_store=feature_store,
        progress_callback=progress_callback,
    )
    candidates = (
        list(strategy_pool[["symbol", "name"]].itertuples(index=False, name=None))
        if not strategy_pool.empty
        else _fallback_candidate_pool(universe)
    )
    dynamic_pool = (
        build_market_dynamic_fallback_pool_store(
            universe,
            market_data_date,
            feature_store=feature_store,
            progress_callback=progress_callback,
        )
        if candidates and strategy_pool.empty
        else pd.DataFrame()
    )
    selected_pool = strategy_pool if not strategy_pool.empty else dynamic_pool
    candidate_context_by_symbol = (
        {
            str(row["symbol"]): {
                "industry_name": str(row.get("industry_name", "") or "").strip(),
                "candidate_strategy": str(row.get("candidate_strategy", "") or "").strip(),
                "candidate_reason": str(row.get("candidate_reason", "") or "").strip(),
            }
            for row in selected_pool.to_dict("records")
        }
        if not selected_pool.empty
        else {}
    )
    ranking_mode = "strategy_candidate_pool" if not strategy_pool.empty else "dynamic_fallback_pool" if not dynamic_pool.empty else "fallback_watchlist"

    if progress_callback is not None:
        progress_callback(
            "扫描候选股票",
            0,
            max(len(candidates), 1),
            f"正在分析 {len(candidates)} 只候选股票",
        )

    rows: list[dict] = []
    total_count = max(len(candidates), 1)
    if FULL_MARKET_MAX_WORKERS <= 1:
        for completed_count, (symbol, name) in enumerate(candidates, start=1):
            result = _load_or_compute_candidate_analysis(
                symbol,
                name,
                horizon_days,
                positive_return,
                FULL_MARKET_HISTORY_START,
                market_model,
                market_proxy_model,
                market_data_date,
                market_context_future,
                candidate_context_by_symbol.get(symbol, {}).get("industry_name"),
                candidate_context_by_symbol.get(symbol, {}).get("candidate_strategy"),
                candidate_context_by_symbol.get(symbol, {}).get("candidate_reason"),
            )
            if result is not None:
                context = candidate_context_by_symbol.get(str(result.get("symbol", "")), {})
                if context:
                    result["candidate_strategy"] = str(context.get("candidate_strategy", "") or "")
                    result["candidate_reason"] = str(context.get("candidate_reason", "") or "")
                    result = _apply_candidate_strategy_prediction_profile(result)
                rows.append(result)
            if progress_callback is not None:
                progress_callback(
                    "扫描候选股票",
                    completed_count,
                    total_count,
                    f"已完成 {completed_count}/{total_count} 只候选股票分析",
                )
    else:
        with ThreadPoolExecutor(max_workers=FULL_MARKET_MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    _load_or_compute_candidate_analysis,
                    symbol,
                    name,
                    horizon_days,
                    positive_return,
                    FULL_MARKET_HISTORY_START,
                    market_model,
                    market_proxy_model,
                    market_data_date,
                    market_context_future,
                    candidate_context_by_symbol.get(symbol, {}).get("industry_name"),
                    candidate_context_by_symbol.get(symbol, {}).get("candidate_strategy"),
                    candidate_context_by_symbol.get(symbol, {}).get("candidate_reason"),
                ): (symbol, name)
                for symbol, name in candidates
            }
            completed_count = 0
            total_count = max(len(futures), 1)
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    context = candidate_context_by_symbol.get(str(result.get("symbol", "")), {})
                    if context:
                        result["candidate_strategy"] = str(context.get("candidate_strategy", "") or "")
                        result["candidate_reason"] = str(context.get("candidate_reason", "") or "")
                        result = _apply_candidate_strategy_prediction_profile(result)
                    rows.append(result)
                completed_count += 1
                if progress_callback is not None:
                    progress_callback(
                        "扫描候选股票",
                        completed_count,
                        total_count,
                        f"已完成 {completed_count}/{total_count} 只候选股票分析",
                    )

    if not rows:
        return pd.DataFrame(), ranking_mode

    if progress_callback is not None:
        progress_callback("写入缓存", 0, 1, "正在整理排序结果并写入缓存")
    ranked = pd.DataFrame(rows)
    ranked, data_mode = _merge_live_market_snapshot(ranked, spot)
    if ranking_mode != "history":
        data_mode = ranking_mode
    ranked = ranked.sort_values(["attention_score", "probability_up", "amount"], ascending=False).reset_index(drop=True)
    ranked.attrs["horizon_days"] = int(horizon_days)
    ranked.attrs["positive_return"] = float(positive_return)
    ranked.attrs["model_source"] = model_source_key
    ranked.attrs["model_source_label"] = model_source_label
    ranked.attrs["model_schema_version"] = MODEL_SCHEMA_VERSION
    ranked.attrs["computed_at"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if progress_callback is not None:
        progress_callback("写入缓存", 1, 1, f"榜单整理完成，共 {len(ranked)} 只股票")
    return ranked, data_mode


@st.cache_data(ttl=300, show_spinner="正在准备全市场 A 股排名，首次加载可能需要几分钟...")
def load_market_rankings(horizon_days: int, positive_return: float) -> pd.DataFrame:
    cached_df, meta = _read_market_rankings_cache(horizon_days, positive_return, allow_stale=True)
    stale_cache_df: pd.DataFrame | None = None
    if cached_df is not None:
        cached_df = cached_df.copy()
        cached_df.attrs["data_mode"] = meta.get("data_mode", "history")
        cached_df.attrs["market_data_date"] = meta.get("market_data_date")
        cached_df.attrs["latest_market_data_date"] = meta.get("latest_market_data_date")
        cached_df.attrs["cache_stale"] = bool(meta.get("cache_stale"))
        cached_df.attrs["computed_at"] = meta.get("computed_at")
        cached_df.attrs["horizon_days"] = int(meta.get("horizon_days", horizon_days))
        cached_df.attrs["positive_return"] = float(meta.get("positive_return", positive_return))
        cached_df.attrs["model_source"] = meta.get("model_source")
        cached_df.attrs["model_source_label"] = meta.get("model_source_label")
        cached_df.attrs["model_schema_version"] = int(meta.get("model_schema_version", MODEL_SCHEMA_VERSION))
        if not bool(meta.get("cache_stale")):
            return cached_df
        stale_cache_df = cached_df

    ranked, data_mode = _build_ranked_market_snapshot(horizon_days, positive_return)
    if ranked.empty:
        return stale_cache_df if stale_cache_df is not None else ranked
    _write_market_rankings_cache(ranked, horizon_days, positive_return, data_mode)
    ranked.attrs["data_mode"] = data_mode
    ranked.attrs["market_data_date"] = _extract_market_data_date(ranked)
    ranked.attrs["latest_market_data_date"] = ranked.attrs["market_data_date"]
    ranked.attrs["cache_stale"] = False
    ranked.attrs["horizon_days"] = int(horizon_days)
    ranked.attrs["positive_return"] = float(positive_return)
    ranked.attrs["model_schema_version"] = MODEL_SCHEMA_VERSION
    return ranked


def _build_market_context() -> dict[str, pd.DataFrame]:
    market_data_date = _latest_market_close_date()
    cached = _read_market_context_cache(market_data_date)
    if cached is not None:
        return cached

    context = {
        "industry_flow": fetch_industry_fund_flow("鍗虫椂"),
        "concept_flow": fetch_concept_fund_flow("鍗虫椂"),
        "macro_calendar": fetch_macro_calendar(limit=10),
    }
    _write_market_context_cache(market_data_date, context)
    return context


@st.cache_data(ttl=300, show_spinner=False)
def load_industry_flow_snapshot() -> pd.DataFrame:
    market_data_date = _latest_market_close_date()
    cached_context = _read_market_context_cache(market_data_date)
    if cached_context is not None:
        industry_flow = cached_context.get("industry_flow")
        if isinstance(industry_flow, pd.DataFrame):
            return industry_flow.copy()
    return fetch_industry_fund_flow("鍗虫椂")


def _resolve_prefetched_market_context(market_context_future: Future | None) -> dict[str, pd.DataFrame]:
    if market_context_future is not None:
        try:
            payload = market_context_future.result()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            return payload
    try:
        return _build_market_context()
    except Exception:
        return {
            "industry_flow": pd.DataFrame(),
            "concept_flow": pd.DataFrame(),
            "macro_calendar": pd.DataFrame(),
        }


def _build_display_board(
    board_df: pd.DataFrame,
    board_size: int,
    ranking_by: str,
    data_mode: str,
    *,
    loading: bool,
) -> pd.DataFrame:
    if board_df.empty:
        return pd.DataFrame()

    source_attrs = dict(board_df.attrs)
    focus_filter_mode = "strict"
    source_board_df = board_df.copy()
    if data_mode in {"strategy_candidate_pool", "latest_close_quick_board"}:
        board_df = source_board_df
        focus_filter_mode = (
            "strategy_hard_filter" if data_mode == "strategy_candidate_pool" else "quick_board_latest_close"
        )
    else:
        filtered_board_df = _filter_focus_candidates(source_board_df)
        if filtered_board_df.empty:
            board_df = _fallback_focus_candidates(source_board_df)
            focus_filter_mode = "fallback"
        else:
            board_df = filtered_board_df
            if len(filtered_board_df) < MIN_FOCUS_BOARD_CANDIDATES:
                fallback_board_df = _fallback_focus_candidates(source_board_df)
                fallback_board_df = fallback_board_df[
                    ~fallback_board_df["symbol"].isin(filtered_board_df["symbol"])
                ].copy()
                if not fallback_board_df.empty:
                    board_df = pd.concat([filtered_board_df, fallback_board_df], ignore_index=True)
                    focus_filter_mode = "supplemented"
    if board_df.empty:
        return pd.DataFrame()

    board_df = _normalize_board_names(board_df)
    optimization_profile = load_adaptive_rank_profile(
        horizon_days=int(source_attrs.get("horizon_days", DEFAULT_VIEW_PARAMS["horizon_days"])),
        positive_return=float(source_attrs.get("positive_return", DEFAULT_VIEW_PARAMS["positive_return"])),
        ranking_by=ranking_by,
        board_size=board_size,
    )
    board = _sort_focus_board(
        board_df,
        ranking_by=ranking_by,
        board_size=board_size,
        optimization_profile=optimization_profile,
    )
    if "board_label" not in board.columns or "price_limit_label" not in board.columns:
        contexts = board.apply(
            lambda row: build_trading_rule_context(symbol=str(row.get("symbol", "")), name=str(row.get("name", ""))),
            axis=1,
        )
        if "board_label" not in board.columns:
            board["board_label"] = [context.board_label for context in contexts]
        if "price_limit_label" not in board.columns:
            board["price_limit_label"] = [context.price_limit_label for context in contexts]

    loading_defaults = {
        "board_label": "A股",
        "price_limit_label": "涨跌幅规则待更新",
        "industry_name": "热度补充中",
        "sector_label": "板块热度补充中",
        "fund_label": "主力资金补充中",
        "news_label": "消息面补充中",
        "enhanced_attention_score": board["attention_score"],
        "predicted_upside_pct": 0.0,
        "predicted_upside_low_pct": 0.0,
        "predicted_upside_high_pct": 0.0,
        "precision_gate_label": "未做90%精度认证",
        "precision_gate_precision": 0.0,
        "precision_gate_threshold": 1.0,
        "precision_gate_support": 0,
        "consecutive_up_days": 0,
        "stage_priority": "待补充",
        "tomorrow_setup": "待更新",
        "tomorrow_bias": "待更新",
        "tomorrow_buy_point": "正在准备明日买点",
        "tomorrow_sell_point": "正在准备明日卖点",
        "tomorrow_plan_confidence": 0.0,
    }
    fallback_defaults = {
        "board_label": "A股",
        "price_limit_label": "涨跌幅规则未知",
        "industry_name": "未知",
        "sector_label": "行业热度未知",
        "fund_label": "主力资金中性",
        "news_label": "消息面中性",
        "enhanced_attention_score": board["attention_score"],
        "predicted_upside_pct": 0.0,
        "predicted_upside_low_pct": 0.0,
        "predicted_upside_high_pct": 0.0,
        "precision_gate_label": "未做90%精度认证",
        "precision_gate_precision": 0.0,
        "precision_gate_threshold": 1.0,
        "precision_gate_support": 0,
        "consecutive_up_days": 0,
        "stage_priority": "未知",
        "tomorrow_setup": "暂无结论",
        "tomorrow_bias": "中性观察",
        "tomorrow_buy_point": "详情加载完成后显示",
        "tomorrow_sell_point": "详情加载完成后显示",
        "tomorrow_plan_confidence": 0.0,
    }
    defaults = loading_defaults if loading else fallback_defaults
    for column, default in defaults.items():
        if column not in board.columns:
            board[column] = default
    structural_defaults = {
        "latest_price": 0.0,
        "change_pct": 0.0,
        "turnover": 0.0,
        "quant_score": 50.0,
        "stage_label": "观察",
        "reason": "策略硬筛选入围，等待增强链路补充解释",
        "candidate_strategy": "",
        "candidate_strategy_label": "通用模型",
        "candidate_strategy_short_label": "通用",
        "candidate_strategy_forecast_bias": "按统一口径评估",
        "candidate_strategy_note": "当前未命中特定硬筛选策略，按统一预测口径处理。",
        "raw_probability_up": board["probability_up"] if "probability_up" in board.columns else 0.0,
        "raw_attention_score": board["attention_score"] if "attention_score" in board.columns else 0.0,
        "raw_enhanced_attention_score": (
            board["enhanced_attention_score"]
            if "enhanced_attention_score" in board.columns
            else board["attention_score"] if "attention_score" in board.columns else 0.0
        ),
        "display_probability_up": board["probability_up"] if "probability_up" in board.columns else 0.0,
        "display_attention_score": board["attention_score"] if "attention_score" in board.columns else 0.0,
        "display_enhanced_attention_score": (
            board["enhanced_attention_score"]
            if "enhanced_attention_score" in board.columns
            else board["attention_score"] if "attention_score" in board.columns else 0.0
        ),
        "ranking_score": board["attention_score"] if "attention_score" in board.columns else 0.0,
        "launch_specialist_score": 50.0,
        "launch_regime_fit_score": 50.0,
        "launch_specialist_confidence": 50.0,
        "board_resonance_strength": 50.0,
        "long_setup_quality": 50.0,
        "crowding_risk": 50.0,
        "crowding_risk_label": "正常",
        "breakout_quality": 50.0,
        "resonance_quality": 50.0,
        "risk_of_late_entry": 50.0,
        "launch_phase_label": "观察",
        "launch_window_label": "等待确认",
        "launch_window_status": "非启动窗",
        "launch_window_summary": "结构与共振尚未形成清晰启动窗口。",
        "launch_window_score": 50.0,
        "launch_window_confidence": 50.0,
        "launch_window_drivers": "",
        "selection_score": 50.0,
        "selection_confidence": 50.0,
        "execution_label": "等待结构",
        "execution_window": "信号未合流",
        "execution_summary": "先看候选质量，再等执行结构和分时承接进一步确认。",
        "execution_score": 50.0,
        "execution_confidence": 50.0,
        "execution_entry_zone": "等待平台位、均线位和分时回踩后的低风险切入点。",
        "execution_invalidation_rule": "若跌回 MA20 下方且量价失衡，应取消计划。",
        "reward_risk_label": "等待更多结构确认",
        "expected_return_pct": 0.0,
        "drawdown_risk_pct": 0.0,
        "reward_risk_ratio": 0.0,
        "chase_risk_label": "先观察",
        "execution_drivers": "",
    }
    for column, default in structural_defaults.items():
        if column not in board.columns:
            board[column] = default
    board = board.apply(lambda row: pd.Series(_apply_candidate_strategy_prediction_profile(dict(row))), axis=1)
    board = _ensure_launch_window_columns(board, force=True)

    market_data_date = str(source_attrs.get("market_data_date") or _extract_market_data_date(board) or "")
    latest_market_data_date = str(source_attrs.get("latest_market_data_date") or market_data_date)
    cache_stale = bool(source_attrs.get("cache_stale", False))
    model_source_label = str(source_attrs.get("model_source_label") or "")
    if "analysis_date" not in board.columns:
        board["analysis_date"] = market_data_date
    if "model_source_label" not in board.columns:
        board["model_source_label"] = model_source_label or "模型来源待确认"
    if "model_result_status" not in board.columns:
        board["model_result_status"] = board["analysis_date"].apply(
            lambda value: _build_model_result_status(str(value or ""), latest_market_data_date, cache_stale=cache_stale)
        )
    action_view = board.apply(_evaluate_board_action, axis=1, result_type="expand")
    for column in action_view.columns:
        board[column] = action_view[column]

    keep_cols = [
        "rank",
        "symbol",
        "name",
        "action_label",
        "action_badge",
        "action_score",
        "action_confidence",
        "selection_score",
        "selection_confidence",
        "analysis_date",
        "model_result_status",
        "model_source_label",
        "board_label",
        "price_limit_label",
        "industry_name",
        "latest_price",
        "change_pct",
        "amount",
        "turnover",
        "consecutive_up_days",
        "attention_score",
        "enhanced_attention_score",
        "raw_attention_score",
        "raw_enhanced_attention_score",
        "candidate_strategy",
        "candidate_strategy_label",
        "candidate_strategy_short_label",
        "candidate_strategy_forecast_bias",
        "candidate_strategy_note",
        "probability_up",
        "raw_probability_up",
        "display_probability_up",
        "ranking_score",
        "predicted_upside_pct",
        "predicted_upside_low_pct",
        "predicted_upside_high_pct",
        "sector_score",
        "fund_score",
        "news_score",
        "precision_gate_label",
        "precision_gate_precision",
        "precision_gate_threshold",
        "precision_gate_support",
        "quant_score",
        "launch_specialist_score",
        "launch_regime_fit_score",
        "launch_specialist_confidence",
        "launch_window_label",
        "launch_window_status",
        "launch_window_summary",
        "launch_window_score",
        "launch_window_confidence",
        "launch_window_drivers",
        "technical_adjustment",
        "intraday_adjustment",
        "backtest_adjustment",
        "execution_label",
        "execution_window",
        "execution_summary",
        "execution_score",
        "execution_confidence",
        "execution_entry_zone",
        "execution_invalidation_rule",
        "reward_risk_label",
        "expected_return_pct",
        "drawdown_risk_pct",
        "reward_risk_ratio",
        "chase_risk_label",
        "execution_drivers",
        "stage_label",
        "stage_priority",
        "tomorrow_setup",
        "tomorrow_bias",
        "tomorrow_buy_point",
        "tomorrow_sell_point",
        "tomorrow_plan_confidence",
        "sector_label",
        "fund_label",
        "news_label",
        "reason",
    ]
    board = board[keep_cols]
    board.attrs.update(source_attrs)
    board.attrs["data_mode"] = data_mode
    board.attrs["ranking_by"] = ranking_by
    board.attrs["loading"] = loading
    board.attrs["focus_filter_mode"] = focus_filter_mode
    board.attrs["market_data_date"] = market_data_date or source_attrs.get("market_data_date")
    board.attrs["latest_market_data_date"] = latest_market_data_date or source_attrs.get("latest_market_data_date")
    board.attrs["cache_stale"] = cache_stale
    board.attrs["horizon_days"] = int(source_attrs.get("horizon_days", DEFAULT_VIEW_PARAMS["horizon_days"]))
    board.attrs["positive_return"] = float(source_attrs.get("positive_return", DEFAULT_VIEW_PARAMS["positive_return"]))
    board.attrs["model_schema_version"] = int(source_attrs.get("model_schema_version", MODEL_SCHEMA_VERSION))
    return board


def _build_enhanced_focus_board(
    base_board: pd.DataFrame,
    ranking_by: str,
    data_mode: str,
    horizon_days: int,
    positive_return: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    market_context = _build_market_context()
    industry_flow = market_context["industry_flow"]
    rows = base_board.drop(columns=["rank"], errors="ignore").to_dict("records")
    enriched_rows: list[dict] = []
    worker_count = min(10, max(1, len(rows)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_enrich_candidate, row, industry_flow, horizon_days, positive_return): row["symbol"]
            for row in rows
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                enriched_rows.append(result)

    if enriched_rows:
        board = pd.DataFrame(enriched_rows)
    else:
        board = base_board.copy()
    rendered = _build_display_board(board, len(base_board), ranking_by, data_mode, loading=False)
    for attr_name, attr_value in base_board.attrs.items():
        rendered.attrs[attr_name] = attr_value
    return rendered, market_context


def _async_task_ready(task_key: str) -> bool:
    future = ASYNC_UI_FUTURES.get(task_key)
    return future is not None and future.done()


def _set_async_task_progress(
    task_key: str,
    phase: str,
    completed: int,
    total: int,
    message: str,
) -> None:
    with ASYNC_UI_PROGRESS_LOCK:
        ASYNC_UI_PROGRESS[task_key] = {
            "phase": str(phase),
            "completed": max(int(completed), 0),
            "total": max(int(total), 1),
            "message": str(message),
        }


def _get_async_task_progress(task_key: str) -> dict[str, object]:
    with ASYNC_UI_PROGRESS_LOCK:
        progress = ASYNC_UI_PROGRESS.get(task_key, {}).copy()
    return progress


def _clear_async_task_progress(task_key: str) -> None:
    with ASYNC_UI_PROGRESS_LOCK:
        ASYNC_UI_PROGRESS.pop(task_key, None)


def _ensure_async_task(task_key: str, fn, *args) -> None:
    future = ASYNC_UI_FUTURES.get(task_key)
    if future is not None:
        return
    ASYNC_UI_FUTURES[task_key] = ASYNC_UI_EXECUTOR.submit(fn, *args)


def _consume_async_task(task_key: str):
    future = ASYNC_UI_FUTURES.get(task_key)
    if future is None or not future.done():
        return False, None, None
    ASYNC_UI_FUTURES.pop(task_key, None)
    try:
        return True, future.result(), None
    except Exception as exc:  # pragma: no cover
        return True, None, exc


def _board_async_key(board: pd.DataFrame, ranking_by: str, data_mode: str) -> str:
    symbols = ",".join(board["symbol"].astype(str).tolist())
    return f"board::{ranking_by}::{data_mode}::{symbols}"


def _market_rank_refresh_async_key(horizon_days: int, positive_return: float) -> str:
    return f"market-refresh::{horizon_days}::{positive_return:.4f}"


def _detail_async_key(symbol: str, horizon_days: int, positive_return: float) -> str:
    return f"detail::{symbol}::{horizon_days}::{positive_return:.4f}"


def _daily_review_async_key(
    board: pd.DataFrame,
    *,
    ranking_by: str,
    horizon_days: int,
    positive_return: float,
    board_size: int,
) -> str:
    market_data_date = str(board.attrs.get("market_data_date") or _extract_market_data_date(board) or "unknown")
    return f"daily-review::{market_data_date}::{ranking_by}::{horizon_days}::{positive_return:.4f}::{board_size}"


def _latest_daily_review_is_current(
    board: pd.DataFrame,
    *,
    ranking_by: str,
    horizon_days: int,
    positive_return: float,
    board_size: int,
) -> bool:
    latest_market_data_date = str(board.attrs.get("latest_market_data_date") or board.attrs.get("market_data_date") or "")
    if not latest_market_data_date:
        return False
    summary = load_latest_review_summary(
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
        board_size=board_size,
    )
    if not summary:
        return False
    return str(summary.get("review_date") or "") == latest_market_data_date


def _run_daily_review_maintenance_task(
    task_key: str,
    board: pd.DataFrame,
    horizon_days: int,
    positive_return: float,
    ranking_by: str,
    board_size: int,
) -> dict[str, object]:
    _set_async_task_progress(task_key, "保存快照", 0, 3, "正在固化今日关注榜快照")
    _set_async_task_progress(task_key, "复盘上一交易日", 1, 3, "正在回测上一交易日关注榜的次日表现")
    result = run_daily_review_maintenance(
        board,
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
        board_size=board_size,
    )
    new_reviews = int(result.get("new_reviews", 0) or 0)
    review_label = "已完成新增复盘并更新排序画像" if new_reviews else "今日没有新增复盘样本，已保存关注榜快照"
    _set_async_task_progress(task_key, "更新排序画像", 3, 3, review_label)
    return result


def _refresh_market_rankings_cache_task(task_key: str, horizon_days: int, positive_return: float) -> dict[str, object] | None:
    _set_async_task_progress(task_key, "准备候选池", 0, 1, "正在读取股票池与交易日数据")
    ranked, data_mode = _build_ranked_market_snapshot(
        horizon_days,
        positive_return,
        progress_callback=lambda phase, completed, total, message: _set_async_task_progress(
            task_key,
            phase,
            completed,
            total,
            message,
        ),
    )
    if ranked.empty:
        return None
    _write_market_rankings_cache(ranked, horizon_days, positive_return, data_mode)
    _set_async_task_progress(task_key, "写入缓存", 1, 1, f"榜单已刷新，共 {len(ranked)} 只股票")
    return {
        "data_mode": data_mode,
        "market_data_date": _extract_market_data_date(ranked),
        "row_count": int(len(ranked)),
    }


def _normalize_view_params(view_params: dict) -> dict:
    return {
        "refresh_seconds": int(view_params.get("refresh_seconds", DEFAULT_VIEW_PARAMS["refresh_seconds"])),
        "ranking_by": str(view_params.get("ranking_by", DEFAULT_VIEW_PARAMS["ranking_by"])),
        "board_size": int(view_params.get("board_size", DEFAULT_VIEW_PARAMS["board_size"])),
        "horizon_days": int(view_params.get("horizon_days", DEFAULT_VIEW_PARAMS["horizon_days"])),
        "positive_return": float(view_params.get("positive_return", DEFAULT_VIEW_PARAMS["positive_return"])),
        "watchlist_text": str(view_params.get("watchlist_text", DEFAULT_VIEW_PARAMS["watchlist_text"])),
    }


def _focus_board_request_key(view_params: dict) -> str:
    params = _normalize_view_params(view_params)
    watchlist = ",".join(parse_watchlist(params["watchlist_text"]))
    return (
        "focus-board::"
        f'{params["board_size"]}::'
        f'{params["horizon_days"]}::'
        f'{params["positive_return"]:.4f}::'
        f'{params["ranking_by"]}::'
        f"{watchlist}"
    )


def _resolve_search_candidate(universe_df: pd.DataFrame, query: str) -> tuple[pd.DataFrame, str | None, str | None]:
    clean_query = str(query).strip()
    if not clean_query:
        return pd.DataFrame(), None, None

    matches = search_a_share_universe(universe_df, clean_query, limit=30)
    normalized_symbol = try_normalize_symbol(clean_query)
    exact_match = pd.DataFrame()
    if not matches.empty and normalized_symbol:
        exact_match = matches[matches["symbol"] == normalized_symbol]
    if exact_match.empty and not matches.empty:
        exact_match = matches[matches["name"] == clean_query]
    if exact_match.empty and not matches.empty and "name_normalized" in matches.columns:
        exact_match = matches[matches["name_normalized"].astype(str) == normalize_search_text(clean_query)]

    if not exact_match.empty:
        row = exact_match.iloc[0]
        return matches, str(row["symbol"]), str(row["name"])
    if len(matches) == 1:
        row = matches.iloc[0]
        return matches, str(row["symbol"]), str(row["name"])
    if normalized_symbol and not universe_df[universe_df["symbol"] == normalized_symbol].empty:
        return matches, normalized_symbol, normalized_symbol
    return matches, None, None


def _view_params_summary(view_params: dict) -> str:
    params = _normalize_view_params(view_params)
    return f'{params["horizon_days"]}日窗口 / 阈值 {params["positive_return"] * 100:.1f}% / 榜单 {params["board_size"]} 只'


def _clear_pending_view_update() -> None:
    task_key = st.session_state.pop("pending_board_update_task_key", None)
    if task_key:
        ASYNC_UI_FUTURES.pop(task_key, None)
    st.session_state.pop("pending_view_params", None)


def _clear_active_board_override() -> None:
    st.session_state.pop("active_board_override_key", None)
    st.session_state.pop("active_board_override", None)


def _clear_heat_data_caches() -> None:
    load_market_context.clear()
    load_industry_flow_snapshot.clear()
    load_focus_board.clear()
    load_symbol_detail.clear()
    for path in CACHE_DIR.glob("market_context_*.pkl"):
        try:
            path.unlink()
        except OSError:
            continue


def _adopt_completed_market_refresh(task_key: str) -> bool:
    ready, payload, error = _consume_async_task(task_key)
    if not ready:
        return False

    _clear_async_task_progress(task_key)
    if error is not None or payload is None:
        st.session_state["market_refresh_notice"] = "后台重算全市场榜单失败，当前先保留已有结果。"
        return True

    load_market_rankings.clear()
    load_focus_board.clear()
    _clear_active_board_override()
    st.session_state["market_refresh_notice"] = (
        f'全市场榜单已刷新到 {payload.get("market_data_date", "--")}，'
        f'当前可用股票数 {int(payload.get("row_count", 0))}。'
    )
    return True


def _board_has_non_latest_model_results(
    board_df: pd.DataFrame,
    latest_market_data_date: str | None,
) -> bool:
    if not isinstance(board_df, pd.DataFrame) or board_df.empty:
        return False
    latest = str(latest_market_data_date or "").strip()
    if not latest:
        return False
    if "analysis_date" in board_df.columns:
        analysis_dates = (
            pd.to_datetime(board_df["analysis_date"], errors="coerce")
            .dropna()
            .dt.strftime("%Y-%m-%d")
        )
        if not analysis_dates.empty and (analysis_dates != latest).any():
            return True
    if "model_result_status" in board_df.columns:
        statuses = board_df["model_result_status"].fillna("").astype(str).str.strip()
        if statuses.str.startswith("非最新结果").any():
            return True
    return False


def _should_auto_force_market_refresh(
    *,
    cache_stale: bool,
    has_non_latest_results: bool,
    quick_board_pending: bool = False,
    custom_watchlist: tuple[str, ...],
    cached_market_data_date: str | None,
    latest_market_data_date: str | None,
    last_forced_date: str | None,
) -> bool:
    cached = str(cached_market_data_date or "").strip()
    latest = str(latest_market_data_date or "").strip()
    forced = str(last_forced_date or "").strip()
    return (
        not custom_watchlist
        and (cache_stale or has_non_latest_results or quick_board_pending)
        and bool(latest)
        and (latest != cached or has_non_latest_results or quick_board_pending)
        and latest != forced
    )


def _resolve_market_refresh_request(
    board: pd.DataFrame,
    *,
    custom_watchlist: tuple[str, ...],
    latest_market_data_date: str | None = None,
    last_forced_date: str | None = None,
) -> dict[str, object]:
    cached_market_data_date = str(board.attrs.get("market_data_date") or "")
    quick_board_pending = bool(board.attrs.get("quick_board_pending", False))
    latest_market_data_date = str(
        latest_market_data_date
        or board.attrs.get("latest_market_data_date")
        or cached_market_data_date
        or ""
    )
    has_non_latest_results = _board_has_non_latest_model_results(board, latest_market_data_date)
    cache_stale = bool(
        board.attrs.get("cache_stale", False)
        or (
            bool(latest_market_data_date)
            and bool(cached_market_data_date)
            and latest_market_data_date != cached_market_data_date
        )
    )
    refresh_reason = (
        "quick_board_pending"
        if quick_board_pending
        else "stale_results"
        if has_non_latest_results
        else "stale_cache"
        if cache_stale
        else ""
    )
    should_force = _should_auto_force_market_refresh(
        cache_stale=cache_stale,
        has_non_latest_results=has_non_latest_results,
        quick_board_pending=quick_board_pending,
        custom_watchlist=custom_watchlist,
        cached_market_data_date=cached_market_data_date,
        latest_market_data_date=latest_market_data_date,
        last_forced_date=last_forced_date,
    )
    return {
        "cache_stale": cache_stale,
        "cached_market_data_date": cached_market_data_date,
        "latest_market_data_date": latest_market_data_date,
        "has_non_latest_results": has_non_latest_results,
        "quick_board_pending": quick_board_pending,
        "refresh_reason": refresh_reason,
        "should_force": should_force,
    }


def _ensure_market_refresh_task_for_board(
    board: pd.DataFrame,
    *,
    custom_watchlist: tuple[str, ...],
    horizon_days: int,
    positive_return: float,
    latest_market_data_date: str | None = None,
) -> dict[str, object]:
    forced_market_refresh_date = str(st.session_state.get("forced_market_refresh_date", "") or "")
    state = _resolve_market_refresh_request(
        board,
        custom_watchlist=custom_watchlist,
        latest_market_data_date=latest_market_data_date,
        last_forced_date=forced_market_refresh_date,
    )
    market_refresh_task_key = _market_rank_refresh_async_key(horizon_days, positive_return)
    state["task_key"] = None
    state["started_now"] = False

    if (
        custom_watchlist
        or (
            not bool(state["cache_stale"])
            and not bool(state["has_non_latest_results"])
            and not bool(state.get("quick_board_pending"))
            and str(state["cached_market_data_date"] or "") == str(state["latest_market_data_date"] or "")
        )
    ):
        st.session_state.pop("forced_market_refresh_date", None)
        st.session_state.pop("forced_market_refresh_reason", None)
        return state

    if (bool(state["cache_stale"]) or bool(state["has_non_latest_results"])) and not custom_watchlist:
        if bool(state["should_force"]):
            st.session_state["forced_market_refresh_date"] = str(state["latest_market_data_date"] or "")
            st.session_state["forced_market_refresh_reason"] = str(state["refresh_reason"] or "")
            _ensure_async_task(
                market_refresh_task_key,
                _refresh_market_rankings_cache_task,
                market_refresh_task_key,
                horizon_days,
                positive_return,
            )
            state["task_key"] = market_refresh_task_key
            state["started_now"] = True
            return state
        if market_refresh_task_key in ASYNC_UI_FUTURES or _get_async_task_progress(market_refresh_task_key):
            state["task_key"] = market_refresh_task_key
            return state
        return state

    return state


def _clear_ranking_cache_files() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for pattern in ("market_rankings_*.pkl", "dynamic_fallback_candidates_*.pkl", "strategy_candidates_*.pkl"):
        for path in CACHE_DIR.glob(pattern):
            try:
                path.unlink()
            except OSError:
                continue


def _clear_market_ranking_caches() -> None:
    load_market_rankings.clear()
    _clear_symbol_base_analysis_cache()
    clear_market_wide_model_cache()
    clear_daily_history_cache(include_disk=False)
    _clear_active_board_override()
    _clear_ranking_cache_files()


def _clear_async_ui_state() -> None:
    ASYNC_UI_FUTURES.clear()
    with ASYNC_UI_PROGRESS_LOCK:
        ASYNC_UI_PROGRESS.clear()
    for key in [
        "enhanced_board_context_key",
        "enhanced_board",
        "market_context_context_key",
        "market_context_async",
        "detail_context_key",
        "detail_async",
    ]:
        st.session_state.pop(key, None)


@st.cache_data(ttl=300, show_spinner=False)
def load_market_context() -> dict[str, pd.DataFrame]:
    return _build_market_context()


@st.cache_data(ttl=86400, show_spinner=False)
def load_a_share_universe() -> pd.DataFrame:
    return fetch_a_share_universe()


def _build_focus_board(
    board_size: int,
    custom_watchlist: tuple[str, ...],
    horizon_days: int,
    positive_return: float,
    ranking_by: str,
) -> pd.DataFrame:
    market_model = _load_market_model_or_none(horizon_days, positive_return)
    market_proxy_model = None if market_model is not None else _load_market_proxy_or_none(horizon_days, positive_return)
    model_source_key, model_source_label = _resolve_model_source(market_model, market_proxy_model)
    market_data_date = _latest_market_close_date()
    if custom_watchlist:
        pool = load_a_share_universe()
        pool = pool[pool["symbol"].isin(custom_watchlist)][["symbol", "name"]].drop_duplicates("symbol").copy()
        if pool.empty:
            return pd.DataFrame()
        base_rows: list[dict] = []
        worker_count = min(8, max(1, len(pool)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _analyze_single_base,
                    symbol,
                    name,
                    horizon_days,
                    positive_return,
                    FULL_MARKET_HISTORY_START,
                    market_model,
                    market_proxy_model,
                    market_data_date,
                ): (symbol, name)
                for symbol, name in pool.itertuples(index=False, name=None)
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    base_rows.append(result)
        if not base_rows:
            return pd.DataFrame()
        ranked = pd.DataFrame(base_rows)
        data_mode = "custom"
        ranked.attrs["market_data_date"] = market_data_date
        ranked.attrs["latest_market_data_date"] = market_data_date
        ranked.attrs["cache_stale"] = False
        ranked.attrs["computed_at"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ranked.attrs["horizon_days"] = int(horizon_days)
        ranked.attrs["positive_return"] = float(positive_return)
        ranked.attrs["model_source"] = model_source_key
        ranked.attrs["model_source_label"] = model_source_label
        ranked.attrs["model_schema_version"] = MODEL_SCHEMA_VERSION
    else:
        ranked = load_market_rankings(horizon_days=horizon_days, positive_return=positive_return)
        data_mode = ranked.attrs.get("data_mode", "history") if not ranked.empty else "history"

    if ranked.empty:
        return pd.DataFrame()
    return _build_display_board(ranked, board_size, ranking_by, data_mode, loading=True)


@st.cache_data(ttl=60, show_spinner=False)
def load_focus_board(
    board_size: int,
    custom_watchlist: tuple[str, ...],
    horizon_days: int,
    positive_return: float,
    ranking_by: str,
) -> pd.DataFrame:
    return _build_focus_board(
        board_size=board_size,
        custom_watchlist=custom_watchlist,
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
    )


def _build_focus_board_payload(view_params: dict) -> dict:
    params = _normalize_view_params(view_params)
    custom_watchlist = tuple(parse_watchlist(params["watchlist_text"]))
    board = _build_focus_board(
        board_size=params["board_size"],
        custom_watchlist=custom_watchlist,
        horizon_days=params["horizon_days"],
        positive_return=params["positive_return"],
        ranking_by=params["ranking_by"],
    )
    return {
        "view_params": params,
        "board": board,
        "board_key": _focus_board_request_key(params),
    }


def _latest_positive_streak_from_flags(flags: pd.Series) -> int:
    streak = 0
    for value in reversed(pd.to_numeric(flags, errors="coerce").fillna(0).astype(int).tolist()):
        if value > 0:
            streak += 1
        else:
            break
    return streak


def _quick_board_probability_from_scores(
    *,
    launch_readiness: pd.Series,
    market_resonance: pd.Series,
    quant_score: pd.Series,
    change_pct: pd.Series,
    turnover: pd.Series,
    amount: pd.Series,
    trend_strength: pd.Series | float,
    streak_days: pd.Series | float,
) -> pd.Series:
    trend_series = (
        trend_strength
        if isinstance(trend_strength, pd.Series)
        else pd.Series(float(trend_strength), index=launch_readiness.index, dtype=float)
    )
    streak_series = (
        streak_days
        if isinstance(streak_days, pd.Series)
        else pd.Series(float(streak_days), index=launch_readiness.index, dtype=float)
    )
    composite_signal = (
        (launch_readiness.fillna(50.0) - 50.0) / 11.5
        + (market_resonance.fillna(50.0) - 50.0) / 14.0
        + (quant_score.fillna(50.0) - 50.0) / 14.5
        + change_pct.fillna(0.0).clip(lower=-8.0, upper=12.0) * 0.20
        + turnover.fillna(0.0).clip(lower=0.0, upper=15.0) * 0.10
        + (amount.fillna(0.0) / 1e8).clip(lower=0.0, upper=25.0) * 0.05
        + trend_series.fillna(0.0).clip(lower=-20.0, upper=35.0) * 0.05
        + streak_series.fillna(0.0).clip(lower=0.0, upper=8.0) * 0.20
    )
    logistic = 1.0 / (1.0 + np.exp(-((composite_signal - 4.0) / 1.6)))
    absolute_probability = 16.0 + logistic * 74.0
    signal_rank = composite_signal.rank(pct=True, method="average").fillna(0.5)
    relative_probability = 24.0 + signal_rank * 58.0
    return (absolute_probability * 0.45 + relative_probability * 0.55).clip(lower=8.0, upper=92.0)


def _quick_board_upside_from_probability(
    *,
    probability_up: pd.Series,
    change_pct: pd.Series,
    trend_strength: pd.Series | float,
    turnover: pd.Series,
) -> pd.Series:
    trend_series = (
        trend_strength
        if isinstance(trend_strength, pd.Series)
        else pd.Series(float(trend_strength), index=probability_up.index, dtype=float)
    )
    projected = (
        (probability_up.fillna(50.0) - 45.0).clip(lower=-10.0, upper=40.0) * 0.12
        + change_pct.fillna(0.0).clip(lower=-5.0, upper=10.0) * 0.12
        + trend_series.fillna(0.0).clip(lower=-15.0, upper=30.0) * 0.05
        + turnover.fillna(0.0).clip(lower=0.0, upper=12.0) * 0.08
    )
    return projected.clip(lower=0.6, upper=18.0)


def _build_latest_close_quick_board_from_snapshots(
    latest_snapshot_df: pd.DataFrame,
    previous_snapshot_df: pd.DataFrame,
    *,
    universe: pd.DataFrame | None,
    latest_market_data_date: str,
    horizon_days: int,
    positive_return: float,
    ranking_by: str,
    board_size: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if latest_snapshot_df.empty:
        return pd.DataFrame(), {}

    frame = latest_snapshot_df.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
    if isinstance(universe, pd.DataFrame) and not universe.empty and {"symbol", "name"}.issubset(universe.columns):
        name_lookup = universe[["symbol", "name"]].drop_duplicates("symbol").copy()
        name_lookup["symbol"] = name_lookup["symbol"].astype(str).str.zfill(6)
        frame = frame.merge(name_lookup, on="symbol", how="left", suffixes=("", "_universe"))
        snapshot_name = frame.get("name")
        universe_name = frame.get("name_universe")
        if snapshot_name is not None or universe_name is not None:
            snapshot_series = (
                snapshot_name.fillna("").astype(str).str.strip()
                if snapshot_name is not None
                else pd.Series("", index=frame.index, dtype=str)
            )
            universe_series = (
                universe_name.fillna("").astype(str).str.strip()
                if universe_name is not None
                else pd.Series("", index=frame.index, dtype=str)
            )
            invalid_snapshot_name = snapshot_series.eq("") | snapshot_series.eq(frame["symbol"])
            frame["name"] = snapshot_series.where(~invalid_snapshot_name, universe_series)
        frame = frame.drop(columns=["name_universe"], errors="ignore")
    if "name" not in frame.columns:
        frame["name"] = frame["symbol"]
    frame["name"] = frame["name"].fillna("").astype(str).str.strip().replace("", pd.NA).fillna(frame["symbol"])
    frame["latest_price"] = pd.to_numeric(frame.get("close"), errors="coerce")
    frame["change_pct"] = pd.to_numeric(frame.get("pct_chg", frame.get("change_pct")), errors="coerce")
    frame["amount"] = pd.to_numeric(frame.get("amount"), errors="coerce")
    frame["turnover"] = pd.to_numeric(frame.get("turnover_rate", frame.get("turnover")), errors="coerce")
    if frame["turnover"].dropna().empty:
        frame["turnover"] = (
            frame["amount"].rank(pct=True, method="average").fillna(0.0) * 8.0
        ).clip(lower=0.5, upper=12.0)

    previous_lookup = (
        previous_snapshot_df[["symbol", "close"]].rename(columns={"close": "prev_trade_close"}).copy()
        if not previous_snapshot_df.empty and {"symbol", "close"}.issubset(previous_snapshot_df.columns)
        else pd.DataFrame(columns=["symbol", "prev_trade_close"])
    )
    if not previous_lookup.empty:
        previous_lookup["symbol"] = previous_lookup["symbol"].astype(str).str.zfill(6)
        frame = frame.merge(previous_lookup, on="symbol", how="left")
    else:
        frame["prev_trade_close"] = float("nan")

    frame["industry_name"] = frame.get("industry", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
    frame["industry_2d_change_pct"] = (
        frame["latest_price"] / pd.to_numeric(frame["prev_trade_close"], errors="coerce").replace(0, np.nan) - 1
    ) * 100

    industry_scope = frame[frame["industry_name"].ne("")].copy()
    if not industry_scope.empty:
        industry_scope["industry_up_flag"] = (pd.to_numeric(industry_scope["change_pct"], errors="coerce") > 0).astype(int)
        industry_stats = (
            industry_scope.groupby("industry_name", as_index=False)
            .agg(
                industry_ret_2d_pct=("industry_2d_change_pct", "median"),
                industry_up_count=("industry_up_flag", "sum"),
                industry_stock_count=("symbol", "count"),
            )
            .sort_values("industry_ret_2d_pct", ascending=False)
            .reset_index(drop=True)
        )
        industry_stats["industry_rank_2d"] = range(1, len(industry_stats) + 1)
        industry_stats["industry_top2d_flag"] = industry_stats["industry_rank_2d"] <= min(10, len(industry_stats))
        frame = frame.drop(
            columns=[column for column in ["industry_ret_2d_pct", "industry_up_count", "industry_stock_count", "industry_rank_2d", "industry_top2d_flag"] if column in frame.columns],
            errors="ignore",
        )
        frame = frame.merge(industry_stats, on="industry_name", how="left")
    else:
        frame["industry_ret_2d_pct"] = float("nan")
        frame["industry_up_count"] = 0
        frame["industry_stock_count"] = 0
        frame["industry_rank_2d"] = float("nan")
        frame["industry_top2d_flag"] = False

    change_pct = frame["change_pct"].fillna(0.0)
    amount = frame["amount"].fillna(0.0)
    turnover = frame["turnover"].fillna(0.0)
    industry_strength = frame["industry_ret_2d_pct"].fillna(0.0)
    industry_up_count = frame["industry_up_count"].fillna(0)
    price_vs_prev = (
        frame["latest_price"] / pd.to_numeric(frame["prev_trade_close"], errors="coerce").replace(0, np.nan) - 1
    ).fillna(0.0)

    frame["quant_score"] = (
        50.0
        + change_pct * 2.2
        + turnover * 1.6
        + industry_strength * 1.2
        + price_vs_prev * 100 * 0.8
    ).clip(lower=0.0, upper=100.0)
    frame["launch_score"] = (
        48.0
        + change_pct * 2.6
        + turnover * 1.2
        + industry_strength * 1.8
        + industry_up_count.clip(upper=10) * 0.9
    ).clip(lower=0.0, upper=100.0)
    frame["launch_readiness_score"] = (
        frame["launch_score"].fillna(50.0)
        + np.where(change_pct >= 2.0, 6.0, 0.0)
        + industry_up_count.clip(upper=10) * 1.1
    ).clip(lower=0.0, upper=100.0)
    frame["market_resonance_score"] = (
        50.0
        + industry_strength * 4.2
        + industry_up_count.clip(upper=10) * 1.7
        + change_pct * 0.9
    ).clip(lower=0.0, upper=100.0)

    industry_available = frame["industry_name"].astype(str).str.strip().ne("")
    strategy1_mask = (
        (change_pct > 2.0)
        & (amount > 2e8)
        & (turnover > 3.0)
        & ((industry_up_count >= 2) | (~industry_available))
    )
    strategy2_mask = (
        (change_pct > 5.0)
        & (amount > 3e8)
        & (turnover > 5.0)
        & industry_available
        & frame["industry_top2d_flag"].fillna(False)
        & (industry_up_count >= 3)
    )
    strategy3_mask = (
        (~strategy1_mask)
        & (~strategy2_mask)
        & (change_pct > 0.8)
        & (amount >= 1.2e8)
        & (turnover >= 1.8)
        & (
            ((industry_strength >= 0.8) & (industry_up_count >= 2))
            | (frame["launch_readiness_score"].fillna(50.0) >= 56.0)
            | (frame["quant_score"].fillna(50.0) >= 58.0)
        )
    )
    frame["candidate_strategy"] = np.select(
        [strategy2_mask, strategy1_mask, strategy3_mask],
        ["策略2", "策略1", "strategy3"],
        default="dynamic_fallback",
    )
    frame["consecutive_up_days"] = np.select(
        [strategy2_mask | strategy1_mask, strategy3_mask, change_pct > 0],
        [3, 2, 1],
        default=0,
    ).astype(int)
    frame["candidate_priority"] = (
        frame["launch_readiness_score"].fillna(50.0) * 0.32
        + frame["market_resonance_score"].fillna(50.0) * 0.22
        + frame["quant_score"].fillna(50.0) * 0.16
        + change_pct * 1.8
        + turnover * 0.8
        + amount / 1e8 * 0.4
    )
    trend_strength = industry_strength * 1.3 + change_pct * 0.7 + price_vs_prev * 100 * 0.5
    frame["probability_up"] = _quick_board_probability_from_scores(
        launch_readiness=frame["launch_readiness_score"],
        market_resonance=frame["market_resonance_score"],
        quant_score=frame["quant_score"],
        change_pct=change_pct,
        turnover=turnover,
        amount=amount,
        trend_strength=trend_strength,
        streak_days=frame["consecutive_up_days"],
    )
    frame["attention_score"] = (
        40.0
        + frame["probability_up"].fillna(50.0) * 0.36
        + change_pct * 1.4
        + turnover * 0.7
        + industry_strength * 1.6
    ).clip(lower=0.0, upper=100.0)
    frame["enhanced_attention_score"] = (
        frame["attention_score"].fillna(50.0)
        + industry_strength * 1.2
        + industry_up_count.clip(upper=10) * 0.7
    ).clip(lower=0.0, upper=100.0)
    frame["predicted_upside_pct"] = _quick_board_upside_from_probability(
        probability_up=frame["probability_up"],
        change_pct=change_pct,
        trend_strength=trend_strength,
        turnover=turnover,
    )

    frame["analysis_date"] = latest_market_data_date
    frame["raw_probability_up"] = frame["probability_up"]
    frame["enhanced_probability_up"] = frame["probability_up"]
    frame["raw_attention_score"] = frame["attention_score"]
    frame["raw_enhanced_attention_score"] = frame["enhanced_attention_score"]
    frame["strategy_pass"] = strategy1_mask | strategy2_mask | strategy3_mask
    frame["strategy_rank"] = frame["candidate_priority"]
    frame["final_rank_score"] = frame["enhanced_attention_score"]
    frame["predicted_upside_low_pct"] = (frame["predicted_upside_pct"] * 0.78).clip(lower=0.4)
    frame["predicted_upside_high_pct"] = (frame["predicted_upside_pct"] * 1.20).clip(lower=0.8)
    frame["precision_gate_label"] = "后台完整版回测中"
    frame["precision_gate_precision"] = 0.0
    frame["precision_gate_threshold"] = 1.0
    frame["precision_gate_support"] = 0
    frame["stage_label"] = np.select(
        [strategy2_mask, strategy1_mask, strategy3_mask],
        ["突破共振快榜", "趋势中继快榜", "多因子主升预备快榜"],
        default="最新收盘快榜",
    )
    frame["stage_priority"] = "快榜"
    frame["candidate_reason"] = np.select(
        [strategy1_mask, strategy2_mask, strategy3_mask],
        [
            "最新收盘日快榜命中策略1，趋势延续强于普通个股。",
            "最新收盘日快榜命中策略2，强势突破与板块共振优先。",
            "最新收盘日快榜命中策略3，多因子主升预备与低拥挤启动质量优先。",
        ],
        default="最新收盘日快榜按当日强度、成交额和板块扩散做轻量筛选。",
    )
    frame["reason"] = frame["candidate_reason"]
    frame["tomorrow_setup"] = "最新收盘快榜先行，等待完整版特征与回测复核"
    frame["tomorrow_bias"] = np.where(change_pct >= 0, "偏强观察", "中性观察")
    frame["tomorrow_buy_point"] = "优先看开盘回踩承接或均价线上方再介入"
    frame["tomorrow_sell_point"] = "若承接消失或跌破预设防守位，先减仓处理"
    frame["tomorrow_plan_confidence"] = (frame["probability_up"].fillna(50.0) * 0.70).clip(lower=0.0, upper=100.0)
    frame["sector_label"] = [
        f"板块两日强度 {float(value):.1f}" if pd.notna(value) else "板块热度待完整版补齐"
        for value in frame["industry_ret_2d_pct"].tolist()
    ]
    frame["fund_label"] = "主力资金待完整版补齐"
    frame["news_label"] = "消息面待完整版补齐"
    frame["snapshot"] = [
        {
            "date": latest_market_data_date,
            "change_pct": float(row_change or 0.0),
            "close_vs_ma20": 0.04 if strategy == "策略2" else 0.01,
            "ret_20": float(row_change or 0.0) / 100.0,
            "volume_ratio_5": 1.4 if strategy == "策略2" else 1.0,
            "breakout_distance_20": 0.01 if strategy == "策略2" else 0.0,
            "range_position_20": 0.72 if strategy == "策略2" else 0.55,
            "upper_shadow_ratio": 0.12 if strategy == "策略2" else 0.18,
        }
        for row_change, strategy in zip(frame["change_pct"].tolist(), frame["candidate_strategy"].tolist())
    ]

    candidate_mask = strategy1_mask | strategy2_mask | strategy3_mask
    quick_board = frame[candidate_mask].copy()
    if quick_board.empty:
        quick_board = frame.copy()
        quick_board["candidate_strategy"] = "strategy3"
        quick_board["candidate_reason"] = "最新收盘日快榜未命中策略1/2硬阈值，按策略3多因子主升预备口径补位。"
        quick_board["strategy_pass"] = True
    quick_board = quick_board.sort_values(
        ["candidate_priority", "enhanced_attention_score", "probability_up", "amount"],
        ascending=False,
    ).drop_duplicates("symbol", keep="first")
    quick_board = quick_board.head(max(board_size, 80)).reset_index(drop=True)
    quick_board.attrs["data_mode"] = "latest_close_quick_board"
    quick_board.attrs["market_data_date"] = latest_market_data_date
    quick_board.attrs["latest_market_data_date"] = latest_market_data_date
    quick_board.attrs["cache_stale"] = False
    quick_board.attrs["quick_board_pending"] = True
    quick_board.attrs["computed_at"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    quick_board.attrs["horizon_days"] = int(horizon_days)
    quick_board.attrs["positive_return"] = float(positive_return)
    quick_board.attrs["model_source"] = "latest_close_quick_board"
    quick_board.attrs["model_source_label"] = "最新收盘快榜（完整版特征与回测正在后台补齐）"
    quick_board.attrs["model_schema_version"] = MODEL_SCHEMA_VERSION
    quick_board.attrs["ranking_by"] = ranking_by
    return quick_board, {
        "board_date": latest_market_data_date,
        "latest_market_data_date": latest_market_data_date,
        "captured_at": quick_board.attrs["computed_at"],
        "horizon_days": int(horizon_days),
        "positive_return": float(positive_return),
        "model_source_label": quick_board.attrs["model_source_label"],
        "quick_board_pending": True,
    }


@st.cache_data(ttl=180, show_spinner=False)
def load_latest_close_quick_board(
    *,
    horizon_days: int,
    positive_return: float,
    ranking_by: str,
    board_size: int,
    rolling_review_days: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    latest_market_data_date = _latest_market_close_date()
    if not latest_market_data_date:
        return pd.DataFrame(), {}

    universe = load_a_share_universe()[["symbol", "name"]].drop_duplicates("symbol").copy()
    if universe.empty:
        return pd.DataFrame(), {}
    universe["symbol"] = universe["symbol"].astype(str).str.zfill(6)

    latest_snapshot_df, previous_snapshot_df = _build_strategy_snapshot_context(latest_market_data_date)
    snapshot_history = read_market_snapshot_history_store(latest_market_data_date)
    snapshot_trade_dates = pd.to_datetime(snapshot_history.get("trade_date"), errors="coerce") if isinstance(snapshot_history, pd.DataFrame) and not snapshot_history.empty else pd.Series(dtype="datetime64[ns]")
    snapshot_history_current = (
        isinstance(snapshot_history, pd.DataFrame)
        and not snapshot_history.empty
        and pd.notna(snapshot_trade_dates.max())
        and snapshot_trade_dates.max() == pd.to_datetime(latest_market_data_date, errors="coerce")
    )
    if not snapshot_history_current:
        if latest_snapshot_df.empty:
            return pd.DataFrame(), {}
        return _build_latest_close_quick_board_from_snapshots(
            latest_snapshot_df,
            previous_snapshot_df,
            universe=universe,
            latest_market_data_date=latest_market_data_date,
            horizon_days=horizon_days,
            positive_return=positive_return,
            ranking_by=ranking_by,
            board_size=board_size,
        )

    history = snapshot_history.copy()
    history["symbol"] = history["symbol"].astype(str).str.zfill(6)
    history = history[history["symbol"].isin(set(universe["symbol"]))].copy()
    if history.empty:
        return pd.DataFrame(), {}

    history["date"] = pd.to_datetime(history.get("date"), errors="coerce")
    history = history.dropna(subset=["date"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    market_ts = pd.to_datetime(latest_market_data_date, errors="coerce")
    if pd.notna(market_ts):
        history = history[history["date"] <= market_ts].copy()
        if history.empty:
            return pd.DataFrame(), {}
        latest_by_symbol = history.groupby("symbol")["date"].transform("max")
        history = history[latest_by_symbol.eq(market_ts)].copy()
        if history.empty:
            return pd.DataFrame(), {}

    numeric_columns = ["open", "high", "low", "close", "change_pct", "amount", "turnover", "volume", "vol"]
    for column in numeric_columns:
        if column in history.columns:
            history[column] = pd.to_numeric(history[column], errors="coerce")
    if "volume" not in history.columns and "vol" in history.columns:
        history["volume"] = pd.to_numeric(history["vol"], errors="coerce")
    if "vol" not in history.columns and "volume" in history.columns:
        history["vol"] = pd.to_numeric(history["volume"], errors="coerce")

    grouped = history.groupby("symbol", sort=False)
    close_group = grouped["close"]
    high_group = grouped["high"] if "high" in history.columns else grouped["close"]
    low_group = grouped["low"] if "low" in history.columns else grouped["close"]
    volume_group = grouped["volume"]

    history["ret_3d_pct"] = close_group.pct_change(3) * 100
    history["ret_5d_pct"] = close_group.pct_change(5) * 100
    history["ret_10d_pct"] = close_group.pct_change(10) * 100
    history["ret_15d_pct"] = close_group.pct_change(15) * 100
    history["ret_20d_pct"] = close_group.pct_change(20) * 100
    history["ma5"] = close_group.rolling(5, min_periods=5).mean().reset_index(level=0, drop=True)
    history["ma10"] = close_group.rolling(10, min_periods=10).mean().reset_index(level=0, drop=True)
    history["ma20"] = close_group.rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    history["volume_ma5"] = volume_group.rolling(5, min_periods=5).mean().reset_index(level=0, drop=True)
    history["high_10"] = high_group.rolling(10, min_periods=1).max().reset_index(level=0, drop=True)
    history["low_10"] = low_group.rolling(10, min_periods=1).min().reset_index(level=0, drop=True)
    range_low_20 = low_group.rolling(20, min_periods=10).min().reset_index(level=0, drop=True)
    range_high_20 = high_group.rolling(20, min_periods=10).max().reset_index(level=0, drop=True)
    history["close_vs_ma20"] = history["close"] / history["ma20"] - 1
    history["volume_ratio_5"] = history["volume"] / history["volume_ma5"].replace(0, np.nan)
    history["distance_to_high_10_pct"] = (history["high_10"] - history["close"]) / history["high_10"].replace(0, np.nan) * 100
    history["max_gain_10_pct"] = (history["high_10"] / history["low_10"].replace(0, np.nan) - 1) * 100
    history["range_position_20"] = (
        (history["close"] - range_low_20) / (range_high_20 - range_low_20).replace(0, np.nan)
    ).clip(lower=0.0, upper=1.0)
    history["upper_shadow_ratio"] = (
        (history["high"] - history[["open", "close"]].max(axis=1))
        / (history["high"] - history["low"]).replace(0, np.nan)
    ).clip(lower=0.0, upper=1.0)
    history["up_flag"] = (history["change_pct"] > 0).astype(int)
    history["prev_trade_close"] = close_group.shift(1)

    latest_rows = grouped.tail(1).copy().reset_index(drop=True)
    streaks = grouped["up_flag"].apply(_latest_positive_streak_from_flags).rename("consecutive_up_days")
    latest_rows = latest_rows.merge(streaks, on="symbol", how="left")
    latest_rows = latest_rows.merge(universe, on="symbol", how="left", suffixes=("", "_universe"))
    if "name_universe" in latest_rows.columns:
        latest_rows["name"] = latest_rows["name_universe"].fillna(latest_rows.get("name")).astype(str)
        latest_rows = latest_rows.drop(columns=["name_universe"], errors="ignore")
    latest_rows["industry_name"] = latest_rows.get("industry", pd.Series("", index=latest_rows.index)).fillna("").astype(str).str.strip()
    latest_rows["industry_2d_change_pct"] = (
        latest_rows["close"] / latest_rows["prev_trade_close"].replace(0, np.nan) - 1
    ) * 100

    industry_scope = latest_rows[latest_rows["industry_name"].ne("")].copy()
    if not industry_scope.empty:
        industry_scope["industry_up_flag"] = (pd.to_numeric(industry_scope["change_pct"], errors="coerce") > 0).astype(int)
        industry_stats = (
            industry_scope.groupby("industry_name", as_index=False)
            .agg(
                industry_ret_2d_pct=("industry_2d_change_pct", "median"),
                industry_up_count=("industry_up_flag", "sum"),
                industry_stock_count=("symbol", "count"),
            )
            .sort_values("industry_ret_2d_pct", ascending=False)
            .reset_index(drop=True)
        )
        industry_stats["industry_rank_2d"] = range(1, len(industry_stats) + 1)
        industry_stats["industry_top2d_flag"] = industry_stats["industry_rank_2d"] <= min(10, len(industry_stats))
        latest_rows = latest_rows.drop(
            columns=[column for column in ["industry_ret_2d_pct", "industry_up_count", "industry_stock_count", "industry_rank_2d", "industry_top2d_flag"] if column in latest_rows.columns],
            errors="ignore",
        )
        latest_rows = latest_rows.merge(industry_stats, on="industry_name", how="left")
    else:
        latest_rows["industry_ret_2d_pct"] = float("nan")
        latest_rows["industry_up_count"] = 0
        latest_rows["industry_stock_count"] = 0
        latest_rows["industry_rank_2d"] = float("nan")
        latest_rows["industry_top2d_flag"] = False

    latest_rows["consecutive_up_days"] = pd.to_numeric(latest_rows["consecutive_up_days"], errors="coerce").fillna(0).astype(int)
    latest_rows["quant_score"] = (
        50.0
        + latest_rows["ret_5d_pct"].fillna(0.0) * 1.05
        + latest_rows["close_vs_ma20"].fillna(0.0) * 210.0
        + (latest_rows["volume_ratio_5"].fillna(1.0) - 1.0) * 16.0
        + (latest_rows["range_position_20"].fillna(0.5) - 0.5) * 18.0
        - latest_rows["upper_shadow_ratio"].fillna(0.0) * 24.0
    ).clip(lower=0.0, upper=100.0)
    latest_rows["launch_score"] = (
        48.0
        + latest_rows["ret_10d_pct"].fillna(0.0) * 0.85
        + latest_rows["change_pct"].fillna(0.0) * 1.9
        + latest_rows["close_vs_ma20"].fillna(0.0) * 185.0
        + (3.0 - latest_rows["distance_to_high_10_pct"].abs().clip(upper=3.0)).fillna(0.0) * 4.0
    ).clip(lower=0.0, upper=100.0)
    latest_rows["launch_readiness_score"] = (
        latest_rows["launch_score"].fillna(50.0)
        + latest_rows["consecutive_up_days"].fillna(0) * 2.4
        + latest_rows["industry_ret_2d_pct"].fillna(0.0) * 1.3
    ).clip(lower=0.0, upper=100.0)
    latest_rows["market_resonance_score"] = (
        50.0
        + latest_rows["industry_ret_2d_pct"].fillna(0.0) * 3.8
        + latest_rows["industry_up_count"].fillna(0).clip(upper=10) * 1.8
        + latest_rows["change_pct"].fillna(0.0) * 0.9
    ).clip(lower=0.0, upper=100.0)

    industry_available = latest_rows["industry_name"].astype(str).str.strip().ne("")
    strategy1_mask = (
        latest_rows["ret_15d_pct"].between(10, 30, inclusive="both")
        & (latest_rows["close"] > latest_rows["ma5"])
        & (latest_rows["ma5"] > latest_rows["ma10"])
        & (latest_rows["ma10"] > latest_rows["ma20"])
        & (latest_rows["change_pct"] > 2)
        & (latest_rows["amount"] > 2e8)
        & (latest_rows["turnover"] > 3)
        & (latest_rows["ret_20d_pct"] < 35)
        & (latest_rows["consecutive_up_days"] >= 3)
    )
    strategy2_mask = (
        (latest_rows["change_pct"] > 5)
        & (latest_rows["ret_3d_pct"] > 10)
        & (latest_rows["ret_5d_pct"] > 15)
        & (latest_rows["amount"] > 3e8)
        & (latest_rows["turnover"] > 5)
        & ((latest_rows["close"] >= latest_rows["high_10"]) | (latest_rows["distance_to_high_10_pct"] < 2))
        & (latest_rows["max_gain_10_pct"] < 40)
        & industry_available
        & latest_rows["industry_top2d_flag"].fillna(False)
        & (latest_rows["industry_up_count"].fillna(0) >= 3)
    )
    strategy3_mask = (
        (~strategy1_mask)
        & (~strategy2_mask)
        & (latest_rows["close"] > latest_rows["ma10"])
        & (latest_rows["ma5"] >= latest_rows["ma20"] * 0.995)
        & latest_rows["ret_5d_pct"].between(2.0, 18.0, inclusive="both")
        & latest_rows["ret_20d_pct"].between(-5.0, 32.0, inclusive="both")
        & (latest_rows["max_gain_10_pct"] < 34.0)
        & (latest_rows["amount"] >= 1.2e8)
        & (latest_rows["turnover"] >= 1.8)
        & (latest_rows["change_pct"] > 0.8)
        & (latest_rows["upper_shadow_ratio"].fillna(0.0) <= 0.26)
        & (latest_rows["range_position_20"].fillna(0.5) >= 0.50)
        & (
            (
                (latest_rows["industry_ret_2d_pct"].fillna(0.0) >= 0.8)
                & (latest_rows["industry_up_count"].fillna(0) >= 2)
            )
            | (latest_rows["launch_readiness_score"].fillna(50.0) >= 62.0)
            | (latest_rows["market_resonance_score"].fillna(50.0) >= 56.0)
        )
    )
    latest_rows["candidate_strategy"] = np.select(
        [strategy2_mask, strategy1_mask, strategy3_mask],
        ["策略2", "策略1", "strategy3"],
        default="dynamic_fallback",
    )
    latest_rows["candidate_priority"] = (
        latest_rows["launch_readiness_score"].fillna(50.0) * 0.34
        + latest_rows["market_resonance_score"].fillna(50.0) * 0.20
        + latest_rows["quant_score"].fillna(50.0) * 0.18
        + latest_rows["ret_5d_pct"].fillna(0.0) * 0.90
        + latest_rows["change_pct"].fillna(0.0) * 0.70
        + latest_rows["consecutive_up_days"].fillna(0) * 5.0
        + (latest_rows["amount"].fillna(0.0) / 1e8) * 0.55
    )
    trend_strength = (
        latest_rows["ret_5d_pct"].fillna(0.0) * 0.8
        + latest_rows["ret_10d_pct"].fillna(0.0) * 0.45
        + latest_rows["close_vs_ma20"].fillna(0.0) * 140.0
        + latest_rows["industry_ret_2d_pct"].fillna(0.0) * 1.1
    )
    latest_rows["probability_up"] = _quick_board_probability_from_scores(
        launch_readiness=latest_rows["launch_readiness_score"],
        market_resonance=latest_rows["market_resonance_score"],
        quant_score=latest_rows["quant_score"],
        change_pct=latest_rows["change_pct"].fillna(0.0),
        turnover=latest_rows["turnover"].fillna(0.0),
        amount=latest_rows["amount"].fillna(0.0),
        trend_strength=trend_strength,
        streak_days=latest_rows["consecutive_up_days"],
    )
    latest_rows["attention_score"] = (
        36.0
        + latest_rows["probability_up"].fillna(50.0) * 0.34
        + latest_rows["launch_score"].fillna(50.0) * 0.16
        + latest_rows["ret_10d_pct"].fillna(0.0) * 0.58
        + latest_rows["consecutive_up_days"].fillna(0) * 4.2
        + (latest_rows["amount"].fillna(0.0) / 1e8) * 0.7
    ).clip(lower=0.0, upper=100.0)
    latest_rows["enhanced_attention_score"] = (
        latest_rows["attention_score"].fillna(50.0)
        + (latest_rows["industry_ret_2d_pct"].fillna(0.0) * 1.8)
        + latest_rows["industry_up_count"].fillna(0).clip(upper=10) * 0.9
    ).clip(lower=0.0, upper=100.0)
    latest_rows["predicted_upside_pct"] = _quick_board_upside_from_probability(
        probability_up=latest_rows["probability_up"],
        change_pct=latest_rows["change_pct"].fillna(0.0),
        trend_strength=trend_strength,
        turnover=latest_rows["turnover"].fillna(0.0),
    )

    candidate_reason = np.select(
        [strategy1_mask, strategy2_mask, strategy3_mask],
        [
            "最新收盘日快榜命中策略1，趋势中继结构优先。",
            "最新收盘日快榜命中策略2，强势突破与板块共振优先。",
            "最新收盘日快榜命中策略3，多因子主升预备与低拥挤启动质量优先。",
        ],
        default="最新收盘日快榜按连涨、量价与板块共振进行轻量补位。",
    )
    latest_rows["candidate_reason"] = candidate_reason

    latest_rows["stage_label"] = np.select(
        [strategy2_mask, strategy1_mask, strategy3_mask, latest_rows["consecutive_up_days"] >= 3],
        ["突破共振快榜", "趋势中继快榜", "多因子主升预备快榜", "强势延续快榜"],
        default="收盘快榜待复核",
    )
    latest_rows["stage_priority"] = "快榜"
    latest_rows["reason"] = latest_rows["candidate_reason"]
    latest_rows["analysis_date"] = latest_market_data_date
    latest_rows["latest_price"] = latest_rows["close"].round(2)
    latest_rows["raw_probability_up"] = latest_rows["probability_up"]
    latest_rows["enhanced_probability_up"] = latest_rows["probability_up"]
    latest_rows["raw_attention_score"] = latest_rows["attention_score"]
    latest_rows["raw_enhanced_attention_score"] = latest_rows["enhanced_attention_score"]
    latest_rows["strategy_pass"] = strategy1_mask | strategy2_mask | strategy3_mask
    latest_rows["strategy_rank"] = latest_rows["candidate_priority"]
    latest_rows["final_rank_score"] = latest_rows["enhanced_attention_score"]
    latest_rows["predicted_upside_low_pct"] = (latest_rows["predicted_upside_pct"] * 0.75).clip(lower=0.5)
    latest_rows["predicted_upside_high_pct"] = (latest_rows["predicted_upside_pct"] * 1.22).clip(lower=0.8)
    latest_rows["precision_gate_label"] = "后台完整版回测中"
    latest_rows["precision_gate_precision"] = 0.0
    latest_rows["precision_gate_threshold"] = 1.0
    latest_rows["precision_gate_support"] = 0
    latest_rows["tomorrow_setup"] = "最新收盘快榜先行，等待完整版特征与回测复核"
    latest_rows["tomorrow_bias"] = np.where(latest_rows["change_pct"].fillna(0.0) >= 0, "偏强观察", "中性观察")
    latest_rows["tomorrow_plan_confidence"] = (
        latest_rows["probability_up"].fillna(50.0) * 0.72 + latest_rows["quant_score"].fillna(50.0) * 0.18
    ).clip(lower=0.0, upper=100.0)
    latest_rows["tomorrow_buy_point"] = [
        f"优先看 {float(close):.2f} 上方回踩不破，或 MA20 附近承接后再跟。"
        for close in latest_rows["close"].fillna(0.0).tolist()
    ]
    latest_rows["tomorrow_sell_point"] = [
        f"若跌回 MA20 或前一日低点下方，先按防守位处理。"
        for _ in range(len(latest_rows))
    ]
    latest_rows["sector_label"] = [
        f"板块两日强度 {float(value):.1f}" if pd.notna(value) else "板块热度待完整版补齐"
        for value in latest_rows["industry_ret_2d_pct"].tolist()
    ]
    latest_rows["fund_label"] = "主力资金待完整版补齐"
    latest_rows["news_label"] = "消息面待完整版补齐"
    latest_rows["snapshot"] = [
        {
            "date": latest_market_data_date,
            "change_pct": float(change_pct or 0.0),
            "close_vs_ma20": float(close_vs_ma20 or 0.0),
            "ret_20": float(ret_20 or 0.0) / 100.0,
            "volume_ratio_5": float(volume_ratio_5 or 1.0),
            "breakout_distance_20": float(distance_to_high or 0.0) / 100.0,
            "range_position_20": float(range_position or 0.5),
            "upper_shadow_ratio": float(upper_shadow or 0.0),
        }
        for change_pct, close_vs_ma20, ret_20, volume_ratio_5, distance_to_high, range_position, upper_shadow in zip(
            latest_rows["change_pct"].tolist(),
            latest_rows["close_vs_ma20"].tolist(),
            latest_rows["ret_20d_pct"].tolist(),
            latest_rows["volume_ratio_5"].tolist(),
            latest_rows["distance_to_high_10_pct"].tolist(),
            latest_rows["range_position_20"].tolist(),
            latest_rows["upper_shadow_ratio"].tolist(),
        )
    ]

    candidate_mask = strategy1_mask | strategy2_mask | strategy3_mask
    quick_board = latest_rows[candidate_mask].copy()
    if quick_board.empty:
        quick_board = latest_rows.copy()
        quick_board["candidate_strategy"] = "strategy3"
        quick_board["candidate_reason"] = "最新收盘日快榜未命中策略1/2硬阈值，按策略3多因子主升预备口径补位。"
        quick_board["strategy_pass"] = True
    quick_board = quick_board.sort_values(
        ["candidate_priority", "enhanced_attention_score", "probability_up", "amount"],
        ascending=False,
    ).drop_duplicates("symbol", keep="first")
    quick_board = quick_board.head(max(board_size, 80)).reset_index(drop=True)
    quick_board.attrs["data_mode"] = "latest_close_quick_board"
    quick_board.attrs["market_data_date"] = latest_market_data_date
    quick_board.attrs["latest_market_data_date"] = latest_market_data_date
    quick_board.attrs["cache_stale"] = False
    quick_board.attrs["quick_board_pending"] = True
    quick_board.attrs["computed_at"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    quick_board.attrs["horizon_days"] = int(horizon_days)
    quick_board.attrs["positive_return"] = float(positive_return)
    quick_board.attrs["model_source"] = "latest_close_quick_board"
    quick_board.attrs["model_source_label"] = "最新收盘快榜（完整版特征与回测正在后台补齐）"
    quick_board.attrs["model_schema_version"] = MODEL_SCHEMA_VERSION
    quick_board.attrs["ranking_by"] = ranking_by
    return quick_board, {
        "board_date": latest_market_data_date,
        "latest_market_data_date": latest_market_data_date,
        "captured_at": quick_board.attrs["computed_at"],
        "horizon_days": int(horizon_days),
        "positive_return": float(positive_return),
        "model_source_label": quick_board.attrs["model_source_label"],
        "quick_board_pending": True,
    }


def _load_history_first_focus_board(
    *,
    board_size: int,
    custom_watchlist: tuple[str, ...],
    horizon_days: int,
    positive_return: float,
    ranking_by: str,
) -> pd.DataFrame:
    if custom_watchlist:
        return pd.DataFrame()

    ranked_cache, meta = _read_market_rankings_cache(horizon_days, positive_return, allow_stale=True)
    latest_market_data_date = _latest_market_close_date()
    if ranked_cache is not None and not ranked_cache.empty:
        ranked_cache = ranked_cache.copy()
        ranked_cache.attrs["data_mode"] = meta.get("data_mode", "history")
        ranked_cache.attrs["market_data_date"] = meta.get("market_data_date")
        ranked_cache.attrs["latest_market_data_date"] = meta.get("latest_market_data_date")
        ranked_cache.attrs["cache_stale"] = bool(meta.get("cache_stale"))
        ranked_cache.attrs["computed_at"] = meta.get("computed_at")
        ranked_cache.attrs["horizon_days"] = int(meta.get("horizon_days", horizon_days))
        ranked_cache.attrs["positive_return"] = float(meta.get("positive_return", positive_return))
        ranked_cache.attrs["model_source"] = meta.get("model_source")
        ranked_cache.attrs["model_source_label"] = meta.get("model_source_label")
        ranked_cache.attrs["model_schema_version"] = int(meta.get("model_schema_version", MODEL_SCHEMA_VERSION))
        cached_market_data_date = str(ranked_cache.attrs.get("market_data_date") or "")
        latest_market_data_date = str(latest_market_data_date or ranked_cache.attrs.get("latest_market_data_date") or "")
        if latest_market_data_date and cached_market_data_date and latest_market_data_date != cached_market_data_date:
            quick_board, quick_meta = load_latest_close_quick_board(
                horizon_days=horizon_days,
                positive_return=positive_return,
                ranking_by=ranking_by,
                board_size=board_size,
            )
            quick_board_date = str(quick_meta.get("board_date") or "")
            if not quick_board.empty and quick_board_date == latest_market_data_date:
                return _build_display_board(
                    quick_board,
                    board_size,
                    ranking_by,
                    str(quick_board.attrs.get("data_mode", "latest_close_quick_board")),
                    loading=True,
                )
            return pd.DataFrame()
        return _build_display_board(
            ranked_cache,
            board_size,
            ranking_by,
            str(ranked_cache.attrs.get("data_mode", "history")),
            loading=True,
        )

    quick_board, quick_meta = load_latest_close_quick_board(
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
        board_size=board_size,
        rolling_review_days=20,
    )
    quick_board_date = str(quick_meta.get("board_date") or "")
    if not quick_board.empty and quick_board_date == str(latest_market_data_date or ""):
        return _build_display_board(
            quick_board,
            board_size,
            ranking_by,
            str(quick_board.attrs.get("data_mode", "latest_close_quick_board")),
            loading=True,
        )

    snapshot_board, snapshot_meta = load_latest_snapshot_board(
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
        board_size=board_size,
    )
    if snapshot_board.empty:
        return pd.DataFrame()

    snapshot_board = snapshot_board.copy()
    board_date = str(snapshot_meta.get("board_date") or snapshot_meta.get("latest_market_data_date") or "")
    snapshot_stale = bool(latest_market_data_date and board_date and latest_market_data_date != board_date)
    snapshot_board.attrs["data_mode"] = "history"
    snapshot_board.attrs["market_data_date"] = board_date
    snapshot_board.attrs["latest_market_data_date"] = latest_market_data_date or board_date
    snapshot_board.attrs["cache_stale"] = snapshot_stale
    snapshot_board.attrs["computed_at"] = str(snapshot_meta.get("computed_at") or snapshot_meta.get("captured_at") or "")
    snapshot_board.attrs["horizon_days"] = int(snapshot_meta.get("horizon_days", horizon_days))
    snapshot_board.attrs["positive_return"] = float(snapshot_meta.get("positive_return", positive_return))
    snapshot_board.attrs["model_source"] = ""
    snapshot_board.attrs["model_source_label"] = str(snapshot_meta.get("model_source_label") or "历史关注榜快照")
    snapshot_board.attrs["model_schema_version"] = MODEL_SCHEMA_VERSION
    history_defaults = {
        "latest_price": 0.0,
        "change_pct": 0.0,
        "turnover": 0.0,
        "predicted_upside_pct": 0.0,
        "predicted_upside_low_pct": 0.0,
        "predicted_upside_high_pct": 0.0,
        "stage_label": "历史快照待补充",
        "reason": "历史关注榜快照优先展示，最新解读正在后台刷新",
    }
    for column, default in history_defaults.items():
        if column not in snapshot_board.columns:
            snapshot_board[column] = default
    return _build_display_board(snapshot_board, board_size, ranking_by, "history", loading=True)


def _build_symbol_detail(symbol: str, horizon_days: int, positive_return: float) -> dict:
    with ThreadPoolExecutor(max_workers=6) as executor:
        daily_future = executor.submit(fetch_daily_history, symbol=symbol, start_date="20220101")
        minute_future = executor.submit(fetch_minute_history, symbol)
        profile_future = executor.submit(fetch_stock_profile, symbol)
        industry_flow_future = executor.submit(load_industry_flow_snapshot)
        fund_flow_future = executor.submit(fetch_stock_main_fund_flow, symbol, 10)
        news_future = executor.submit(fetch_stock_news, symbol, 12)

        daily = daily_future.result()
        minute = minute_future.result()
        profile = profile_future.result()
        industry_flow = industry_flow_future.result()
        fund_flow_df = fund_flow_future.result()
        news_df = news_future.result()
    features = build_daily_features(daily)
    stage = classify_stage(daily)
    market_model_status = _market_model_status(horizon_days, positive_return)
    latest_market_data_date = _latest_market_close_date()
    latest_features = features.dropna().iloc[-1]
    model, market_model, market_proxy_model, model_source_key, model_source_label = _build_symbol_model_result(
        daily,
        horizon_days=horizon_days,
        positive_return=positive_return,
        latest_feature_values=latest_features,
    )
    backtest_model = model
    if market_model is not None or market_proxy_model is not None:
        backtest = _build_backtest_status_from_model(model)
    else:
        backtest = run_daily_strategy_backtest(
            daily,
            horizon_days=horizon_days,
            positive_return=positive_return,
            model_result=backtest_model,
        )
    quant_signal = evaluate_quant_signal(daily, features)
    intraday = evaluate_intraday(minute)
    snapshot = latest_snapshot(daily, features)
    model_state = explain_latest_model_state(daily, symbol=symbol, latest_feature_values=latest_features)
    stage_score = stage_numeric_score(stage, latest_features)
    rule_context = build_trading_rule_context(symbol=symbol, name=str(profile.get("股票简称", symbol)), profile=profile)
    industry_name = str(profile.get("行业", "未知"))
    sector_signal = compute_sector_hot_score(industry_name, industry_flow)
    fund_signal = evaluate_main_fund_signal(fund_flow_df)
    news_signal = build_research_enhanced_news_signal(
        news_df,
        base_signal=evaluate_news_sentiment(news_df),
        symbol=symbol,
    )
    temporal_news_pulse = evaluate_temporal_news_pulse(news_df)
    intraday_structure_signal = evaluate_intraday_structure_signal(minute)
    model = apply_live_probability_upgrade(
        model,
        daily,
        latest_feature_values=latest_features.to_dict(),
        minute_df=minute,
        news_df=news_df,
        fund_flow_df=fund_flow_df,
        symbol=symbol,
    )
    model_state["state_reason_lines"] = list(model_state.get("state_reason_lines", [])) + [str(model.upgrade_summary)]
    base_attention_score = _score_base_attention(
        model.latest_probability,
        stage_score,
        snapshot,
        quant_signal.total_score,
        main_rise_start_score(latest_features),
    )
    raw_base_attention_score = float(base_attention_score)
    attention_score = _score_final_attention(
        base_attention_score,
        float(sector_signal["sector_score"]),
        float(fund_signal["fund_score"]),
        float(news_signal["sentiment_score"]),
        fund_confidence=float(fund_signal.get("confidence_score", 55.0)),
        news_confidence=float(news_signal.get("confidence_score", 55.0)),
    )
    raw_enhanced_attention_score = float(attention_score)
    optimization_profile = load_adaptive_rank_profile(
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by="关注分数",
        board_size=50,
    )
    replay_overlay = compute_replay_calibrated_scores(
        {
            "probability_up": float(model.latest_probability) * 100,
            "attention_score": base_attention_score,
            "enhanced_attention_score": attention_score,
            "quant_score": quant_signal.total_score,
            "launch_score": main_rise_start_score(latest_features),
            "launch_readiness_score": _safe_float(model.signal_breakdown.get("launch_readiness_score"), _safe_float(latest_features.get("launch_readiness"), 50.0)),
            "market_resonance_score": _safe_float(model.signal_breakdown.get("market_resonance_score"), _safe_float(latest_features.get("market_resonance"), 50.0)),
            "stage_label": stage.label,
            "precision_gate_label": getattr(model, "precision_gate_label", ""),
            "market_regime_label": str(latest_features.get("market_regime_label", "")),
            "market_ret_5": _safe_float(latest_features.get("market_ret_5")),
            "market_ret_20": _safe_float(latest_features.get("market_ret_20")),
            "market_close_vs_ma20": _safe_float(latest_features.get("market_close_vs_ma20")),
            "market_volatility_10": _safe_float(latest_features.get("market_volatility_10")),
            "market_range_position_20": _safe_float(latest_features.get("market_range_position_20"), 0.5),
            "ret_20": _safe_float(latest_features.get("ret_20")),
            "close_vs_ma20": _safe_float(snapshot.get("close_vs_ma20")),
            "breakout_distance_20": _safe_float(snapshot.get("breakout_distance_20")),
            "range_position_20": _safe_float(snapshot.get("range_position_20"), 0.5),
            "volume_ratio_5": _safe_float(snapshot.get("volume_ratio_5"), 1.0),
            "upper_shadow_ratio": _safe_float(snapshot.get("upper_shadow_ratio")),
            "stretch_risk": _safe_float(latest_features.get("stretch_risk")),
            "risk_pressure": _safe_float(latest_features.get("risk_pressure")),
        },
        optimization_profile,
    )
    if replay_overlay.get("replay_calibration_active"):
        model.latest_probability = float(replay_overlay["probability_up"]) / 100.0
        model.signal_breakdown = dict(model.signal_breakdown)
        model.signal_breakdown["replay_probability_delta_pct"] = float(replay_overlay["probability_delta_pct"])
        model.signal_breakdown["replay_calibration_confidence"] = float(replay_overlay["replay_calibration_confidence"])
        model.signal_breakdown["replay_attention_delta"] = float(replay_overlay["attention_delta"])
        model.signal_breakdown["replay_enhanced_attention_delta"] = float(replay_overlay["enhanced_attention_delta"])
        model.upgrade_summary = (
            f"{model.upgrade_summary} {str(replay_overlay['replay_calibration_note'])}".strip()
        )
        model_state["state_reason_lines"] = list(model_state.get("state_reason_lines", [])) + [
            str(replay_overlay["replay_calibration_note"])
        ]
    base_attention_score = float(replay_overlay["attention_score"])
    attention_score = float(replay_overlay["enhanced_attention_score"])
    strategy_workbench = build_strategy_workbench(
        stage_code=stage.code,
        probability_up=model.latest_probability,
        quant_score=quant_signal.total_score,
        sector_score=float(sector_signal["sector_score"]),
        temporal_pulse=temporal_news_pulse,
        intraday_signal=intraday_structure_signal,
        rule_context=rule_context,
    )
    tomorrow_plan = build_tomorrow_plan(
        stage,
        snapshot,
        latest_features,
        model.latest_probability,
        quant_signal.total_score,
        intraday_state=intraday,
        intraday_signal=intraday_structure_signal,
    )
    detail_payload = {
        "symbol": symbol,
        "daily": daily,
        "minute": minute,
        "stage": stage,
        "stage_score": stage_score,
        "model": model,
        "backtest_model": backtest_model,
        "market_model": market_model,
        "market_proxy_model": market_proxy_model,
        "market_model_status": market_model_status,
        "backtest": backtest,
        "quant_signal": quant_signal,
        "intraday": intraday,
        "snapshot": snapshot,
        "latest_features": latest_features.to_dict(),
        "model_state": model_state,
        "profile": profile,
        "rule_context": rule_context,
        "sector_signal": sector_signal,
        "fund_signal": fund_signal,
        "fund_flow_df": fund_flow_df,
        "news_signal": news_signal,
        "news_df": news_df,
        "temporal_news_pulse": temporal_news_pulse,
        "intraday_structure_signal": intraday_structure_signal,
        "strategy_workbench": strategy_workbench,
        "tomorrow_plan": tomorrow_plan,
        "analysis_date": snapshot["date"],
        "latest_market_data_date": latest_market_data_date,
        "model_source": model_source_key,
        "model_source_label": model_source_label,
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "base_attention_score": base_attention_score,
        "enhanced_attention_score": attention_score,
        "attention_score": attention_score,
        "launch_score": float(main_rise_start_score(latest_features)),
        "launch_readiness_score": float(
            _safe_float(model.signal_breakdown.get("launch_readiness_score"), _safe_float(latest_features.get("launch_readiness"), 50.0))
        ),
        "market_resonance_score": float(
            _safe_float(model.signal_breakdown.get("market_resonance_score"), _safe_float(latest_features.get("market_resonance"), 50.0))
        ),
        "raw_probability_up": float(getattr(model, "base_probability", model.latest_probability) or model.latest_probability) * 100,
        "raw_attention_score": raw_base_attention_score,
        "raw_enhanced_attention_score": raw_enhanced_attention_score,
        "launch_specialist_score": float(_safe_float(model.signal_breakdown.get("launch_specialist_score"), 50.0)),
        "launch_regime_fit_score": float(_safe_float(model.signal_breakdown.get("launch_regime_fit_score"), 50.0)),
        "launch_specialist_confidence": float(_safe_float(model.signal_breakdown.get("launch_specialist_confidence"), 50.0)),
        "optimization_profile": optimization_profile,
        "replay_overlay": replay_overlay,
    }
    detail_payload.update(
        _build_launch_window_view(
            detail_payload,
            stage_code=str(getattr(stage, "code", "")),
            stage_label=str(getattr(stage, "label", "")),
        )
    )
    return detail_payload


@st.cache_data(ttl=180, show_spinner=False)
def load_symbol_detail(symbol: str, horizon_days: int, positive_return: float) -> dict:
    return _build_symbol_detail(symbol, horizon_days, positive_return)


def make_daily_chart(daily: pd.DataFrame) -> go.Figure:
    view = daily.tail(120).copy()
    ma5 = view["close"].rolling(5).mean()
    ma20 = view["close"].rolling(20).mean()
    ma60 = view["close"].rolling(60).mean()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.72, 0.28],
    )
    fig.add_trace(
        go.Candlestick(
            x=view["date"],
            open=view["open"],
            high=view["high"],
            low=view["low"],
            close=view["close"],
            increasing_line_color="#ff5f57",
            decreasing_line_color="#30b0a0",
            increasing_fillcolor="rgba(255, 95, 87, 0.82)",
            decreasing_fillcolor="rgba(48, 176, 160, 0.72)",
            name="日K",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=view["date"], y=ma5, name="MA5", line=dict(color="#0071e3", width=1.7)), row=1, col=1)
    fig.add_trace(go.Scatter(x=view["date"], y=ma20, name="MA20", line=dict(color="#5e5ce6", width=1.9)), row=1, col=1)
    fig.add_trace(go.Scatter(x=view["date"], y=ma60, name="MA60", line=dict(color="#7d8590", width=1.6)), row=1, col=1)
    fig.add_trace(
        go.Bar(
            x=view["date"],
            y=view["volume"],
            name="成交量",
            marker_color="rgba(154, 168, 182, 0.75)",
            opacity=0.8,
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        height=620,
        margin=dict(l=10, r=10, t=22, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.74)",
        legend=dict(orientation="h", y=1.03, x=0.01, bgcolor="rgba(255,255,255,0.0)"),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        font=dict(
            family='-apple-system, BlinkMacSystemFont, "SF Pro Display", "PingFang SC", "Helvetica Neue", sans-serif',
            color="#1d1d1f",
        ),
    )
    fig.update_xaxes(showgrid=False, tickformat="%m-%d")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.16)", zeroline=False)
    return fig


def make_minute_chart(minute: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.74, 0.26],
    )
    if minute.empty:
        fig.update_layout(
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.74)",
            annotations=[
                dict(
                    text="当前暂无当日 1 分钟数据",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=18, color="#465866"),
                )
            ],
        )
        return fig

    fig.add_trace(
        go.Scatter(
            x=minute["datetime"],
            y=minute["close"],
            mode="lines",
            name="分时",
            line=dict(color="#0071e3", width=2.7),
            fill="tozeroy",
            fillcolor="rgba(0, 113, 227, 0.12)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=minute["datetime"],
            y=minute["avg_price"],
            mode="lines",
            name="鍧囦环绾",
            line=dict(color="#30b0a0", width=1.95),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=minute["datetime"],
            y=minute["volume"],
            name="分钟量",
            marker_color="rgba(154, 168, 182, 0.76)",
            opacity=0.82,
        ),
        row=2,
        col=1,
    )
    session_open = float(minute["open"].iloc[0]) if pd.notna(minute["open"].iloc[0]) else None
    if session_open is not None:
        fig.add_hline(
            y=session_open,
            row=1,
            col=1,
            line=dict(color="rgba(29, 29, 31, 0.18)", width=1, dash="dot"),
        )
    fig.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.74)",
        legend=dict(orientation="h", y=1.03, x=0.01, bgcolor="rgba(255,255,255,0.0)"),
        hovermode="x unified",
        font=dict(
            family='-apple-system, BlinkMacSystemFont, "SF Pro Display", "PingFang SC", "Helvetica Neue", sans-serif',
            color="#1d1d1f",
        ),
    )
    fig.update_xaxes(showgrid=False, tickformat="%H:%M")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.16)", zeroline=False)
    return fig


def make_fund_flow_chart(fund_flow_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if fund_flow_df.empty or "日期" not in fund_flow_df.columns or "主力净流入-净额" not in fund_flow_df.columns:
        fig.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.64)",
            annotations=[
                dict(
                    text="当前暂无可绘制的主力资金数据",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16, color="#465866"),
                )
            ],
        )
        return fig

    view = fund_flow_df.copy()
    view["日期"] = pd.to_datetime(view["日期"], errors="coerce")
    view["主力净流入-净额"] = pd.to_numeric(view["主力净流入-净额"], errors="coerce")
    view = view.dropna(subset=["日期", "主力净流入-净额"]).sort_values("日期").tail(10)
    colors = ["#ff5f57" if value >= 0 else "#30b0a0" for value in view["主力净流入-净额"]]
    fig.add_bar(
        x=view["日期"],
        y=view["主力净流入-净额"] / 1e8,
        marker_color=colors,
        name="主力净流入(亿)",
        opacity=0.88,
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.74)",
        showlegend=False,
        font=dict(
            family='-apple-system, BlinkMacSystemFont, "SF Pro Display", "PingFang SC", "Helvetica Neue", sans-serif',
            color="#1d1d1f",
        ),
    )
    fig.update_xaxes(showgrid=False, tickformat="%m-%d")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.16)", zeroline=False)
    return fig


def make_backtest_chart(backtest) -> go.Figure:
    fig = go.Figure()
    equity_curve = backtest.equity_curve.copy()
    if equity_curve.empty or "date" not in equity_curve.columns or "equity" not in equity_curve.columns:
        fig.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.64)",
            annotations=[
                dict(
                    text="当前暂无可展示的回测权益曲线",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16, color="#465866"),
                )
            ],
        )
        return fig

    equity_curve["date"] = pd.to_datetime(equity_curve["date"], errors="coerce")
    equity_curve = equity_curve.dropna(subset=["date", "equity"]).copy()
    if equity_curve.empty:
        return fig

    fig.add_trace(
        go.Scatter(
            x=equity_curve["date"],
            y=equity_curve["equity"],
            mode="lines",
            name="策略净值",
            line=dict(color="#0071e3", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(0, 113, 227, 0.12)",
        )
    )
    fig.add_hline(y=1.0, line=dict(color="rgba(29, 29, 31, 0.18)", width=1, dash="dot"))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=18, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.74)",
        showlegend=False,
        hovermode="x unified",
        font=dict(
            family='-apple-system, BlinkMacSystemFont, "SF Pro Display", "PingFang SC", "Helvetica Neue", sans-serif',
            color="#1d1d1f",
        ),
    )
    fig.update_xaxes(showgrid=False, tickformat="%Y-%m")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.16)", zeroline=False)
    return fig


def _metric_card(title: str, value: str, note: str) -> None:
    safe_title = escape(str(title))
    safe_value = escape(str(value))
    safe_note = escape(str(note))
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="card-title">{safe_title}</div>
            <div class="card-value">{safe_value}</div>
            <div class="card-note">{safe_note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _describe_data_mode(data_mode: str, custom_watchlist: tuple[str, ...] = ()) -> tuple[str, str]:
    headline = "自定义股票池" if custom_watchlist else "全市场 A 股统一排名"
    if data_mode == "live":
        note = "实时快照可用，结合在线热度增强做当日更新。"
    elif data_mode == "strategy_candidate_pool":
        note = "优先使用策略1/策略2的正式硬筛选结果，连续上涨池只做显式兜底。"
    elif data_mode == "dynamic_fallback_pool":
        note = "实时快照不可用时，改用最新收盘日生成的动态上涨兜底池。"
    elif data_mode == "fallback_watchlist":
        note = "主数据源受限，当前使用核心股票池做应急展示。"
    else:
        note = "优先展示可用快榜单，再异步补齐增强信息。"
    return headline, note


def _render_market_hero(
    board: pd.DataFrame,
    *,
    clock: dict[str, str],
    ranking_by: str,
    custom_watchlist: tuple[str, ...],
    data_mode: str,
) -> None:
    context = _board_freshness_context(board)
    headline, mode_note = _describe_data_mode(data_mode, custom_watchlist)
    top_row = board.iloc[0] if not board.empty else {}
    top_name = escape(str(top_row.get("name", "--")))
    top_symbol = escape(str(top_row.get("symbol", "--")))
    top_action = escape(str(top_row.get("action_label", "观察")))
    top_stage = escape(str(top_row.get("stage_label", "观察")))
    top_strategy = escape(str(top_row.get("candidate_strategy_label", "通用模型")))
    top_prob = f'{float(top_row.get("probability_up", 0.0) or 0.0):.1f}%'
    top_upside = f'{float(top_row.get("predicted_upside_pct", 0.0) or 0.0):.1f}%'
    top_attention = f'{float(top_row.get("enhanced_attention_score", top_row.get("attention_score", 0.0)) or 0.0):.1f}'
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-grid">
                <div class="hero-copy">
                    <div class="hero-kicker">A股信号台</div>
                    <h1 class="hero-title">今日关注榜与交易工作台</h1>
                    <p class="hero-description">
                        先用全市场统一排序秒出快榜，再补齐消息面、主力资金、板块热度和个股详情。
                        当前按 <strong>{escape(ranking_by)}</strong> 排序，帮助你先锁定最值得深看的一批股票。
                    </p>
                    <div class="hero-meta">
                        <span class="hero-pill">北京时间 {escape(str(clock.get("now", "--")))}</span>
                        <span class="hero-pill">市场状态 {escape(str(clock.get("status", "--")))}</span>
                        <span class="hero-pill">数据基准日 {escape(str(context["market_data_date"]))}</span>
                        <span class="hero-pill">最新收盘日 {escape(str(context["latest_market_data_date"]))}</span>
                        <span class="hero-pill">模型引擎 {escape(str(context["model_source_label"]))}</span>
                    </div>
                </div>
                <div class="hero-side">
                    <div class="hero-panel">
                        <p class="hero-panel-title">当前模式</p>
                        <p class="hero-panel-note">{escape(headline)}。{escape(mode_note)}</p>
                        <div class="hero-stat-grid">
                            <div class="hero-stat-card">
                                <div class="hero-stat-label">榜首个股</div>
                                <div class="hero-stat-value">{top_name} {top_symbol}</div>
                                <div class="hero-stat-note">{top_strategy}</div>
                            </div>
                            <div class="hero-stat-card">
                                <div class="hero-stat-label">当前动作</div>
                                <div class="hero-stat-value">{top_action}</div>
                                <div class="hero-stat-note">{top_stage}</div>
                            </div>
                            <div class="hero-stat-card">
                                <div class="hero-stat-label">上涨概率</div>
                                <div class="hero-stat-value">{top_prob}</div>
                                <div class="hero-stat-note">基于榜首个股的当前主值</div>
                            </div>
                            <div class="hero-stat-card">
                                <div class="hero-stat-label">预测涨幅 / 关注分</div>
                                <div class="hero-stat-value">{top_upside} / {top_attention}</div>
                                <div class="hero-stat-note">融合热度、量价和模型结果</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_board_insight_strip(
    board: pd.DataFrame,
    *,
    ranking_by: str,
    board_size: int,
    horizon_days: int,
) -> None:
    if board.empty:
        return
    display_scores = pd.to_numeric(
        board.get("enhanced_attention_score", board.get("attention_score", pd.Series(dtype=float))),
        errors="coerce",
    )
    probabilities = pd.to_numeric(board.get("probability_up", pd.Series(dtype=float)), errors="coerce")
    launch_scores = pd.to_numeric(board.get("launch_window_score", pd.Series(dtype=float)), errors="coerce")
    action_labels = board.get("action_label", pd.Series(dtype=object)).fillna("")
    display_count = int(len(board))
    positive_actions = int(action_labels.isin(["买", "持"]).sum())
    launch_ready_count = int((launch_scores >= 75).sum()) if not launch_scores.empty else 0
    top_strategy = str(
        board.iloc[0].get("candidate_strategy_short_label")
        or board.iloc[0].get("candidate_strategy_label")
        or "通用"
    )
    avg_score_text = f'{float(display_scores.mean()):.1f}' if not display_scores.dropna().empty else "--"
    avg_prob_text = f'{float(probabilities.mean()):.1f}%' if not probabilities.dropna().empty else "--"
    launch_ratio_text = f"{launch_ready_count}/{display_count}" if display_count else "0/0"
    st.markdown(
        f"""
        <div class="insight-strip">
            <div class="insight-card">
                <div class="insight-label">Board Coverage</div>
                <div class="insight-value">{display_count}</div>
                <div class="insight-note">当前展示前 {min(display_count, int(board_size))} 支，按 {escape(ranking_by)} 排序。</div>
            </div>
            <div class="insight-card">
                <div class="insight-label">Average Score</div>
                <div class="insight-value">{escape(avg_score_text)}</div>
                <div class="insight-note">榜内平均增强分，用来看当前候选整体质量。</div>
            </div>
            <div class="insight-card">
                <div class="insight-label">Average Prob</div>
                <div class="insight-value">{escape(avg_prob_text)}</div>
                <div class="insight-note">对应未来 {int(horizon_days)} 个交易日达标概率的榜内均值。</div>
            </div>
            <div class="insight-card">
                <div class="insight-label">Launch / Action</div>
                <div class="insight-value">{escape(launch_ratio_text)}</div>
                <div class="insight-note">启动窗强候选 {launch_ready_count} 支，买/持信号 {positive_actions} 支，榜首策略 {escape(top_strategy)}。</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _top_focus_cards(board: pd.DataFrame) -> None:
    cols = st.columns(min(3, len(board)))
    for col, (_, row) in zip(cols, board.head(3).iterrows()):
        display_score = float(row.get("enhanced_attention_score", row["attention_score"]))
        stock_name = escape(str(row["name"]))
        industry_name = escape(str(row.get("industry_name", "行业待补充")))
        stage_label = escape(str(row.get("stage_label", "观察")))
        strategy_label = escape(str(row.get("candidate_strategy_label", row.get("candidate_strategy", "通用模型"))))
        strategy_note = escape(str(row.get("candidate_strategy_forecast_bias", "按统一口径评估")))
        reason = escape(str(row.get("reason", "")))
        tomorrow_setup = escape(str(row.get("tomorrow_setup", "待评估")))
        tomorrow_buy_point = escape(str(row.get("tomorrow_buy_point", "详情补充中")))
        launch_window_status = escape(str(row.get("launch_window_status", "非启动窗")))
        launch_window_score = float(row.get("launch_window_score", 50.0) or 50.0)
        execution_label = escape(str(row.get("execution_label", "等待结构")))
        execution_window = escape(str(row.get("execution_window", "信号未合流")))
        execution_score = float(row.get("execution_score", 50.0) or 50.0)
        selection_score = float(row.get("selection_score", row.get("attention_score", 50.0)) or 50.0)
        consecutive_up_days = int(row.get("consecutive_up_days", 0) or 0)
        probability_text = f'{float(row.get("probability_up", 0.0)):.1f}%'
        predicted_upside_text = f'{float(row.get("predicted_upside_pct", 0.0)):.1f}%'
        quant_text = f'{float(row.get("quant_score", 0.0)):.1f}'
        change_text = _format_pct(row.get("change_pct"), signed=True)
        action_badge = escape(str(row.get("action_badge", row.get("action_label", "观察"))))
        action_css_class = escape(str(row.get("action_css_class", "watch")))
        action_confidence_text = f'{float(row.get("action_confidence", 0.0)):.1f}'
        reward_risk_label = escape(str(row.get("reward_risk_label", "等待更多结构确认")))
        with col:
            st.markdown(
                f"""
                <div class="focus-card">
                    <div class="focus-head">
                        <div>
                            <div class="focus-symbol">{stock_name}<span>{row["symbol"]}</span></div>
                            <div class="focus-subtitle">{industry_name} | {stage_label} | {strategy_label}</div>
                        </div>
                        <div class="focus-rank">#{int(row.get("rank", 0) or 0)}</div>
                    </div>
                    <div class="focus-score-block">
                        <div class="focus-score-value">{display_score:.1f}</div>
                        <div class="focus-score-label">综合关注分</div>
                    </div>
                    <div class="focus-meta">
                        <div class="focus-stat">
                            <span class="focus-stat-label">上涨概率</span>
                            <span class="focus-stat-value">{probability_text}</span>
                        </div>
                        <div class="focus-stat">
                            <span class="focus-stat-label">量化辅助</span>
                            <span class="focus-stat-value">{quant_text}</span>
                        </div>
                        <div class="focus-stat">
                            <span class="focus-stat-label">选股分</span>
                            <span class="focus-stat-value">{selection_score:.1f}</span>
                        </div>
                        <div class="focus-stat">
                            <span class="focus-stat-label">执行分</span>
                            <span class="focus-stat-value">{execution_score:.1f}</span>
                        </div>
                        <div class="focus-stat">
                            <span class="focus-stat-label">预测涨幅</span>
                            <span class="focus-stat-value">{predicted_upside_text}</span>
                        </div>
                        <div class="focus-stat">
                            <span class="focus-stat-label">当日涨跌</span>
                            <span class="focus-stat-value">{change_text}</span>
                        </div>
                        <div class="focus-stat">
                            <span class="focus-stat-label">连涨天数</span>
                            <span class="focus-stat-value">{consecutive_up_days} 天</span>
                        </div>
                    </div>
                    <div class="focus-reason">入选逻辑：{reason}</div>
                    <div class="focus-reason">策略口径：{strategy_note} | 明日形势：{tomorrow_setup}</div>
                    <div class="focus-status-row">
                        <span class="overview-action-chip {action_css_class}">{action_badge}</span>
                        <span class="brief-pill">置信度 {action_confidence_text}</span>
                        <span class="brief-pill">启动窗口 {launch_window_status} {launch_window_score:.1f}</span>
                        <span class="brief-pill">执行层 {execution_label}</span>
                    </div>
                    <div class="focus-reason">执行窗口：{execution_window} | 盈亏比：{reward_risk_label}</div>
                    <div class="focus-plan">
                        <div class="focus-plan-label">计划买点</div>
                        <div class="focus-plan-text">{tomorrow_buy_point}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_focus_board_tables(rendered_board: pd.DataFrame) -> None:
    summary_columns = [
        "rank", "symbol", "name", "candidate_strategy_label", "action_label", "launch_window_status",
        "selection_score", "execution_label", "execution_score",
        "attention_score", "enhanced_attention_score", "raw_probability_up", "enhanced_probability_up",
        "probability_up", "final_rank_score", "long_setup_quality", "crowding_risk", "launch_phase_label", "predicted_upside_pct",
    ]
    detail_columns = [
        "rank", "symbol", "name", "analysis_date", "model_result_status", "model_source_label",
        "candidate_strategy_label", "candidate_strategy_forecast_bias", "attention_score", "enhanced_attention_score",
        "selection_score", "selection_confidence", "raw_probability_up", "enhanced_probability_up",
        "probability_up", "final_rank_score", "ranking_score", "launch_phase_label",
        "breakout_quality", "resonance_quality", "board_resonance_strength", "long_setup_quality",
        "crowding_risk", "crowding_risk_label", "risk_of_late_entry", "predicted_upside_pct", "precision_gate_label",
        "quant_score", "launch_window_status", "launch_window_score", "launch_window_confidence",
        "execution_label", "execution_window", "execution_score", "execution_confidence",
        "reward_risk_label", "reward_risk_ratio", "execution_entry_zone", "execution_invalidation_rule",
        "tomorrow_plan_confidence", "latest_price", "change_pct", "amount", "board_label", "price_limit_label",
        "industry_name", "stage_label", "tomorrow_setup", "tomorrow_bias", "tomorrow_buy_point",
        "tomorrow_sell_point", "sector_label", "fund_label", "news_label", "reason", "launch_window_summary", "execution_summary",
    ]
    summary_view = rendered_board[[column for column in summary_columns if column in rendered_board.columns]].copy()
    detail_view = rendered_board[[column for column in detail_columns if column in rendered_board.columns]].copy()

    st.caption("核心榜单用于快速扫盘，研判明细会补充模型来源、次日计划、热度解释和买卖点。")
    tab_summary, tab_detail = st.tabs(["核心榜单", "研判明细"])
    with tab_summary:
        st.dataframe(
            summary_view,
            width="stretch",
            hide_index=True,
            column_config={
                "rank": st.column_config.NumberColumn("排名", format="%d"),
                "symbol": "代码",
                "name": "名称",
                "candidate_strategy_label": "策略",
                "action_label": "状态",
                "launch_window_status": "启动窗口",
                "selection_score": st.column_config.ProgressColumn("选股分", min_value=0, max_value=100, format="%.1f"),
                "execution_label": "执行状态",
                "execution_score": st.column_config.ProgressColumn("执行分", min_value=0, max_value=100, format="%.1f"),
                "attention_score": st.column_config.ProgressColumn("关注分", min_value=0, max_value=100, format="%.1f"),
                "enhanced_attention_score": st.column_config.ProgressColumn("增强分", min_value=0, max_value=100, format="%.1f"),
                "raw_probability_up": st.column_config.ProgressColumn("原始概率(%)", min_value=0, max_value=100, format="%.1f"),
                "enhanced_probability_up": st.column_config.ProgressColumn("增强后概率(%)", min_value=0, max_value=100, format="%.1f"),
                "probability_up": st.column_config.ProgressColumn("最终概率(%)", min_value=0, max_value=100, format="%.1f"),
                "final_rank_score": st.column_config.ProgressColumn("最终排序分", min_value=0, max_value=100, format="%.1f"),
                "long_setup_quality": st.column_config.ProgressColumn("做多质量", min_value=0, max_value=100, format="%.1f"),
                "crowding_risk": st.column_config.ProgressColumn("拥挤风险", min_value=0, max_value=100, format="%.1f"),
                "launch_phase_label": "主升标签",
                "predicted_upside_pct": st.column_config.NumberColumn("预测涨幅(%)", format="%.1f"),
            },
        )
    with tab_detail:
        st.dataframe(
            detail_view,
            width="stretch",
            hide_index=True,
            column_config={
                "rank": st.column_config.NumberColumn("排名", format="%d"),
                "symbol": "代码",
                "name": "名称",
                "analysis_date": "基准日",
                "model_result_status": "结果状态",
                "model_source_label": "模型引擎",
                "candidate_strategy_label": "策略",
                "candidate_strategy_forecast_bias": "预测口径",
                "attention_score": st.column_config.ProgressColumn("关注分", min_value=0, max_value=100, format="%.1f"),
                "enhanced_attention_score": st.column_config.ProgressColumn("增强分", min_value=0, max_value=100, format="%.1f"),
                "selection_score": st.column_config.ProgressColumn("选股分", min_value=0, max_value=100, format="%.1f"),
                "selection_confidence": st.column_config.ProgressColumn("选股置信度", min_value=0, max_value=100, format="%.1f"),
                "raw_probability_up": st.column_config.ProgressColumn("原始概率(%)", min_value=0, max_value=100, format="%.1f"),
                "enhanced_probability_up": st.column_config.ProgressColumn("增强后概率(%)", min_value=0, max_value=100, format="%.1f"),
                "probability_up": st.column_config.ProgressColumn("最终概率(%)", min_value=0, max_value=100, format="%.1f"),
                "final_rank_score": st.column_config.ProgressColumn("最终排序分", min_value=0, max_value=100, format="%.1f"),
                "ranking_score": st.column_config.ProgressColumn("排序分", min_value=0, max_value=100, format="%.1f"),
                "launch_phase_label": "主升标签",
                "breakout_quality": st.column_config.ProgressColumn("突破质量", min_value=0, max_value=100, format="%.1f"),
                "resonance_quality": st.column_config.ProgressColumn("共振质量", min_value=0, max_value=100, format="%.1f"),
                "board_resonance_strength": st.column_config.ProgressColumn("板块前排", min_value=0, max_value=100, format="%.1f"),
                "long_setup_quality": st.column_config.ProgressColumn("做多质量", min_value=0, max_value=100, format="%.1f"),
                "crowding_risk": st.column_config.ProgressColumn("拥挤风险", min_value=0, max_value=100, format="%.1f"),
                "crowding_risk_label": "拥挤状态",
                "risk_of_late_entry": st.column_config.ProgressColumn("追高风险", min_value=0, max_value=100, format="%.1f"),
                "predicted_upside_pct": st.column_config.NumberColumn("预测涨幅(%)", format="%.1f"),
                "quant_score": st.column_config.ProgressColumn("量化辅助", min_value=0, max_value=100, format="%.1f"),
                "launch_window_status": "启动窗口",
                "launch_window_score": st.column_config.ProgressColumn("启动分", min_value=0, max_value=100, format="%.1f"),
                "launch_window_confidence": st.column_config.ProgressColumn("启动置信度", min_value=0, max_value=100, format="%.1f"),
                "execution_label": "执行状态",
                "execution_window": "执行窗口",
                "execution_score": st.column_config.ProgressColumn("执行分", min_value=0, max_value=100, format="%.1f"),
                "execution_confidence": st.column_config.ProgressColumn("执行置信度", min_value=0, max_value=100, format="%.1f"),
                "reward_risk_label": "盈亏比",
                "reward_risk_ratio": st.column_config.NumberColumn("盈亏比值", format="%.2f"),
                "execution_entry_zone": "计划买点",
                "execution_invalidation_rule": "失效位",
                "tomorrow_plan_confidence": st.column_config.ProgressColumn("计划置信度", min_value=0, max_value=100, format="%.1f"),
                "latest_price": st.column_config.NumberColumn("最新价", format="%.2f"),
                "change_pct": st.column_config.NumberColumn("涨跌幅(%)", format="%.2f"),
                "amount": st.column_config.NumberColumn("成交额", format="%.0f"),
                "board_label": "板块",
                "price_limit_label": "涨跌幅制度",
                "industry_name": "行业",
                "stage_label": "阶段",
                "tomorrow_setup": "明日形势",
                "tomorrow_bias": "明日偏向",
                "tomorrow_buy_point": "明日买点",
                "tomorrow_sell_point": "明日卖点",
                "sector_label": "板块热度",
                "fund_label": "主力资金",
                "news_label": "消息面",
                "reason": "入选逻辑",
                "launch_window_summary": "启动窗说明",
                "execution_summary": "执行说明",
            },
        )

def _render_market_context(market_context: dict[str, pd.DataFrame]) -> None:
    industry_df = market_context["industry_flow"]
    concept_df = market_context["concept_flow"]
    macro_df = market_context["macro_calendar"]

    top_industry = industry_df.iloc[0] if not industry_df.empty else None
    top_concept = concept_df.iloc[0] if not concept_df.empty else None

    summary_cols = st.columns(4)
    with summary_cols[0]:
        title = str(top_industry["sector_name"]) if top_industry is not None else "--"
        note = (
            f'净流入 {float(top_industry["net_inflow"]):.2f} 亿 / 领涨 {top_industry["leader"]}'
            if top_industry is not None
            else "当前没有可用的行业资金数据"
        )
        _metric_card("行业热度第一", title, note)
    with summary_cols[1]:
        title = str(top_concept["sector_name"]) if top_concept is not None else "--"
        note = (
            f'净流入 {float(top_concept["net_inflow"]):.2f} 亿 / 领涨 {top_concept["leader"]}'
            if top_concept is not None
            else "当前没有可用的概念资金数据"
        )
        _metric_card("概念热度第一", title, note)
    with summary_cols[2]:
        rising_count = int((industry_df["change_pct"] > 0).sum()) if not industry_df.empty else 0
        _metric_card("上涨行业数量", str(rising_count), "行业涨跌幅大于 0 的板块数量")
    with summary_cols[3]:
        _metric_card("宏观事件条数", str(len(macro_df)), "当天已抓取的宏观事件与时间点")

    tab_industry, tab_concept, tab_macro = st.tabs(["行业资金", "概念资金", "宏观日历"])
    with tab_industry:
        if industry_df.empty:
            st.info("当前没有抓到行业资金流数据。")
        else:
            st.dataframe(
                industry_df[["sector_name", "net_inflow", "change_pct", "leader", "leader_change_pct"]].head(12),
                width="stretch",
                hide_index=True,
                column_config={
                    "sector_name": "行业",
                    "net_inflow": st.column_config.NumberColumn("净流入(亿)", format="%.2f"),
                    "change_pct": st.column_config.NumberColumn("涨跌幅(%)", format="%.2f"),
                    "leader": "领涨股",
                    "leader_change_pct": st.column_config.NumberColumn("领涨股涨跌幅(%)", format="%.2f"),
                },
            )
    with tab_concept:
        if concept_df.empty:
            st.info("当前没有抓到概念资金流数据。")
        else:
            st.dataframe(
                concept_df[["sector_name", "net_inflow", "change_pct", "leader", "leader_change_pct"]].head(12),
                width="stretch",
                hide_index=True,
                column_config={
                    "sector_name": "概念",
                    "net_inflow": st.column_config.NumberColumn("净流入(亿)", format="%.2f"),
                    "change_pct": st.column_config.NumberColumn("涨跌幅(%)", format="%.2f"),
                    "leader": "领涨股",
                    "leader_change_pct": st.column_config.NumberColumn("领涨股涨跌幅(%)", format="%.2f"),
                },
            )
    with tab_macro:
        if macro_df.empty:
            st.info("当前没有抓到宏观日历数据。")
        else:
            show_cols = [col for col in ["日期", "时间", "地区", "事件", "重要性"] if col in macro_df.columns]
            st.dataframe(macro_df[show_cols], width="stretch", hide_index=True)

def _render_news(news_df: pd.DataFrame) -> None:
    if news_df.empty:
        st.info("当前没有抓到该股最近新闻。")
        return
    for _, row in news_df.head(8).iterrows():
        publish = row.get("发布时间")
        publish_text = publish.strftime("%Y-%m-%d %H:%M") if pd.notna(publish) else "未知时间"
        title = escape(str(row.get("新闻标题", "未命名新闻")))
        source = escape(str(row.get("文章来源", "未知来源")))
        url = str(row.get("新闻链接", "")).strip()
        if url.startswith("http"):
            title_html = f'<a href="{escape(url, quote=True)}" target="_blank">{title}</a>'
        else:
            title_html = title
        st.markdown(
            f"""
            <div class="news-item">
                <div class="news-meta">{escape(publish_text)}</div>
                <div class="news-title">{title_html}</div>
                <div class="news-source">来源：{source}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_fund_flow_panel(fund_flow_df: pd.DataFrame) -> None:
    if fund_flow_df.empty:
        st.info("当前没有抓到个股主力资金流数据。")
        return

    st.plotly_chart(make_fund_flow_chart(fund_flow_df), width="stretch", config={"displayModeBar": False})
    show_cols = [col for col in ["日期", "收盘价", "涨跌幅", "主力净流入-净额", "主力净流入-净占比"] if col in fund_flow_df.columns]
    if show_cols:
        st.dataframe(fund_flow_df[show_cols], width="stretch", hide_index=True)


def _board_freshness_context(board: pd.DataFrame) -> dict[str, object]:
    market_data_date = str(board.attrs.get("market_data_date") or _extract_market_data_date(board) or "--")
    latest_market_data_date = str(board.attrs.get("latest_market_data_date") or market_data_date or "--")
    cache_stale = bool(board.attrs.get("cache_stale", False))
    computed_at = str(board.attrs.get("computed_at") or "--")
    model_source_label = str(board.attrs.get("model_source_label") or "模型来源待确认")
    is_latest = (
        market_data_date not in {"", "--"}
        and latest_market_data_date not in {"", "--"}
        and market_data_date == latest_market_data_date
        and not cache_stale
    )
    status_label = "最新模型结果" if is_latest else "缓存结果待刷新"
    status_note = (
        "当前关注榜已经按最新收盘数据完成模型计算。"
        if is_latest
        else f"当前先展示 {market_data_date} 的可用结果，最新收盘日是 {latest_market_data_date}。"
    )
    return {
        "market_data_date": market_data_date,
        "latest_market_data_date": latest_market_data_date,
        "computed_at": computed_at,
        "model_source_label": model_source_label,
        "model_schema_version": int(board.attrs.get("model_schema_version", MODEL_SCHEMA_VERSION)),
        "status_label": status_label,
        "status_note": status_note,
        "is_latest": is_latest,
    }


def _render_board_freshness_banner(board: pd.DataFrame) -> None:
    context = _board_freshness_context(board)
    status_class = "freshness-pill positive" if context["is_latest"] else "freshness-pill warning"
    st.markdown(
        f"""
        <div class="freshness-banner">
            <div class="freshness-title">榜单基准说明</div>
            <div class="freshness-grid">
                <span class="freshness-pill">数据基准日 {escape(str(context["market_data_date"]))}</span>
                <span class="freshness-pill">最新收盘日 {escape(str(context["latest_market_data_date"]))}</span>
                <span class="{status_class}">{escape(str(context["status_label"]))}</span>
                <span class="freshness-pill">模型引擎 {escape(str(context["model_source_label"]))}</span>
                <span class="freshness-pill">计算时间 {escape(str(context["computed_at"]))}</span>
            </div>
            <div class="freshness-note">{escape(str(context["status_note"]))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_daily_review_banner(
    board: pd.DataFrame,
    *,
    horizon_days: int,
    positive_return: float,
    ranking_by: str,
    board_size: int,
) -> None:
    latest_market_data_date = str(board.attrs.get("latest_market_data_date") or board.attrs.get("market_data_date") or "--")
    summary = load_latest_review_summary(
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
        board_size=board_size,
    )
    profile = load_adaptive_rank_profile(
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
        board_size=board_size,
    )
    weights = dict(profile.get("weights", {}))
    weight_labels = {
        "attention_score": "关注分数",
        "probability_up": "上涨概率",
        "enhanced_attention_score": "增强分数",
        "quant_score": "量化辅助",
    }
    weight_badges = "".join(
        f'<span class="freshness-pill">{escape(weight_labels.get(key, key))} {float(value) * 100:.1f}%</span>'
        for key, value in weights.items()
        if key in weight_labels
    )
    market_replay_badges = ""
    if int(profile.get("market_replay_days", 0) or 0) > 0:
        market_replay_badges = (
            f'<span class="freshness-pill">长窗回放 {int(profile.get("market_replay_days", 0) or 0)} 日</span>'
            f'<span class="freshness-pill">全市场样本 {int(profile.get("market_replay_symbols", 0) or 0)} 股</span>'
            f'<span class="freshness-pill">回放记录 {int(profile.get("market_replay_rows", 0) or 0)} 条</span>'
        )

    if summary:
        board_date = str(summary.get("board_date") or "--")
        review_date = str(summary.get("review_date") or "--")
        status_label = "已回测到最新交易日" if review_date == latest_market_data_date else "等待最新收盘日复盘"
        status_class = "freshness-pill positive" if review_date == latest_market_data_date else "freshness-pill warning"
        note = str(summary.get("optimization_note") or profile.get("profile_summary") or "")
        metrics_html = (
            f'<span class="freshness-pill">复盘榜单日 {escape(board_date)}</span>'
            f'<span class="freshness-pill">验证交易日 {escape(review_date)}</span>'
            f'<span class="freshness-pill">样本 {int(summary.get("review_count", 0) or 0)} 只</span>'
            f'<span class="freshness-pill">次日均收益 {float(summary.get("avg_return_pct", 0.0) or 0.0):.2f}%</span>'
            f'<span class="freshness-pill">上涨胜率 {float(summary.get("win_rate_pct", 0.0) or 0.0):.2f}%</span>'
            f'<span class="freshness-pill">达标命中率 {float(summary.get("target_hit_rate_pct", 0.0) or 0.0):.2f}%</span>'
        )
    else:
        status_label = "自动复盘已启用"
        status_class = "freshness-pill warning"
        note = (
            f"当前已保存 {board.attrs.get('market_data_date', '--')} 的关注榜快照。"
            "下一交易日收盘数据到位后，系统会自动回测前一日榜单并更新排序权重。"
        )
        metrics_html = (
            f'<span class="freshness-pill">最新收盘日 {escape(latest_market_data_date)}</span>'
            f'<span class="freshness-pill">复盘样本日 {int(profile.get("review_days", 0) or 0)} 个</span>'
            f'<span class="freshness-pill">累计样本 {int(profile.get("review_stocks", 0) or 0)} 只</span>'
        )

    profile_note = str(profile.get("profile_summary") or "")
    generated_at = str(profile.get("generated_at") or "--")
    st.markdown(
        f"""
        <div class="freshness-banner">
            <div class="freshness-title">自动复盘与排序优化</div>
            <div class="freshness-grid">
                <span class="{status_class}">{escape(status_label)}</span>
                <span class="freshness-pill">画像更新时间 {escape(generated_at)}</span>
                {metrics_html}
                {market_replay_badges}
                {weight_badges}
            </div>
            <div class="freshness-note">{escape(note)}</div>
            <div class="freshness-note">{escape(profile_note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_review_battle_panels(
    *,
    horizon_days: int,
    positive_return: float,
    ranking_by: str,
    board_size: int,
) -> None:
    battle = load_review_battle_panels(
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
        board_size=board_size,
    )
    strategy_panel = battle.get("strategy_panel", pd.DataFrame())
    short_market_state_panel = battle.get("short_market_state_panel", pd.DataFrame())
    long_market_state_panel = battle.get("long_market_state_panel", pd.DataFrame())
    combo_panel = battle.get("combo_panel", pd.DataFrame())
    meta = dict(battle.get("meta", {}) or {})
    if (
        not isinstance(strategy_panel, pd.DataFrame)
        or strategy_panel.empty
    ) and (
        not isinstance(short_market_state_panel, pd.DataFrame)
        or short_market_state_panel.empty
    ) and (
        not isinstance(long_market_state_panel, pd.DataFrame)
        or long_market_state_panel.empty
    ):
        return

    _section_header(
        "战绩面板",
        "实战战绩面板",
        "把复盘结果拆成市场状态、策略和状态×策略组合，直接观察哪类环境与哪套方法最近更容易兑现。",
    )
    metric_cols = st.columns(4)
    with metric_cols[0]:
        _metric_card(
            "短窗复盘日",
            f'{int(meta.get("review_days", 0) or 0)}',
            f'样本 {int(meta.get("review_rows", 0) or 0)} 条 / 股票 {int(meta.get("review_symbols", 0) or 0)} 只',
        )
    with metric_cols[1]:
        _metric_card(
            "最强策略",
            str(meta.get("best_strategy") or "等待样本"),
            "按最近复盘样本的胜率、平均收益和命中率综合排序",
        )
    with metric_cols[2]:
        _metric_card(
            "最强市场状态",
            str(meta.get("best_market_state") or "等待样本"),
            "优先看短窗复盘；若样本不足则退回长窗全市场状态回放",
        )
    with metric_cols[3]:
        _metric_card(
            "长窗状态样本",
            f'{int(meta.get("market_replay_days", 0) or 0)} 日',
            f'记录 {int(meta.get("market_replay_rows", 0) or 0)} 条 / 股票 {int(meta.get("market_replay_symbols", 0) or 0)} 只',
        )

    tabs = st.tabs(["策略短窗", "市场状态", "状态×策略"])

    with tabs[0]:
        if not isinstance(strategy_panel, pd.DataFrame) or strategy_panel.empty:
            st.info("当前还没有足够的分策略复盘样本，等上一交易日榜单继续沉淀后会自动补齐。")
        else:
            show_cols = [
                column
                for column in [
                    "candidate_strategy_label",
                    "sample_count",
                    "avg_rank",
                    "avg_probability_pct",
                    "avg_return_pct",
                    "intraday_high_return_pct",
                    "win_rate_pct",
                    "target_hit_rate_pct",
                    "direction_hit_rate_pct",
                    "calibration_gap_pct",
                ]
                if column in strategy_panel.columns
            ]
            st.dataframe(
                strategy_panel[show_cols],
                width="stretch",
                hide_index=True,
                column_config={
                    "candidate_strategy_label": "策略",
                    "sample_count": st.column_config.NumberColumn("样本数", format="%d"),
                    "avg_rank": st.column_config.NumberColumn("平均排名", format="%.1f"),
                    "avg_probability_pct": st.column_config.NumberColumn("平均预测概率(%)", format="%.2f"),
                    "avg_return_pct": st.column_config.NumberColumn("次日平均收益(%)", format="%.2f"),
                    "intraday_high_return_pct": st.column_config.NumberColumn("日内高点收益(%)", format="%.2f"),
                    "win_rate_pct": st.column_config.ProgressColumn("上涨胜率(%)", min_value=0, max_value=100, format="%.1f"),
                    "target_hit_rate_pct": st.column_config.ProgressColumn("达标命中率(%)", min_value=0, max_value=100, format="%.1f"),
                    "direction_hit_rate_pct": st.column_config.ProgressColumn("方向命中率(%)", min_value=0, max_value=100, format="%.1f"),
                    "calibration_gap_pct": st.column_config.NumberColumn("概率偏差(%)", format="%.2f"),
                },
            )

    with tabs[1]:
        state_left, state_right = st.columns(2, gap="large")
        with state_left:
            st.caption("最近关注榜短窗复盘")
            if not isinstance(short_market_state_panel, pd.DataFrame) or short_market_state_panel.empty:
                st.info("短窗市场状态样本仍在积累。")
            else:
                show_cols = [
                    column
                    for column in [
                        "market_state_display",
                        "sample_count",
                        "avg_probability_pct",
                        "avg_return_pct",
                        "intraday_high_return_pct",
                        "win_rate_pct",
                        "target_hit_rate_pct",
                        "direction_hit_rate_pct",
                        "calibration_gap_pct",
                    ]
                    if column in short_market_state_panel.columns
                ]
                st.dataframe(
                    short_market_state_panel[show_cols],
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "market_state_display": "市场状态",
                        "sample_count": st.column_config.NumberColumn("样本数", format="%d"),
                        "avg_probability_pct": st.column_config.NumberColumn("平均预测概率(%)", format="%.2f"),
                        "avg_return_pct": st.column_config.NumberColumn("次日平均收益(%)", format="%.2f"),
                        "intraday_high_return_pct": st.column_config.NumberColumn("日内高点收益(%)", format="%.2f"),
                        "win_rate_pct": st.column_config.ProgressColumn("上涨胜率(%)", min_value=0, max_value=100, format="%.1f"),
                        "target_hit_rate_pct": st.column_config.ProgressColumn("达标命中率(%)", min_value=0, max_value=100, format="%.1f"),
                        "direction_hit_rate_pct": st.column_config.ProgressColumn("方向命中率(%)", min_value=0, max_value=100, format="%.1f"),
                        "calibration_gap_pct": st.column_config.NumberColumn("概率偏差(%)", format="%.2f"),
                    },
                )
        with state_right:
            st.caption("更长窗口全市场状态回放")
            if not isinstance(long_market_state_panel, pd.DataFrame) or long_market_state_panel.empty:
                st.info("长窗全市场状态回放样本还没有完成沉淀。")
            else:
                show_cols = [
                    column
                    for column in [
                        "market_state_display",
                        "sample_count",
                        "avg_return_pct",
                        "intraday_high_return_pct",
                        "win_rate_pct",
                        "target_hit_rate_pct",
                        "state_edge",
                    ]
                    if column in long_market_state_panel.columns
                ]
                st.dataframe(
                    long_market_state_panel[show_cols],
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "market_state_display": "市场状态",
                        "sample_count": st.column_config.NumberColumn("样本数", format="%d"),
                        "avg_return_pct": st.column_config.NumberColumn("平均收益(%)", format="%.2f"),
                        "intraday_high_return_pct": st.column_config.NumberColumn("高点收益(%)", format="%.2f"),
                        "win_rate_pct": st.column_config.ProgressColumn("上涨胜率(%)", min_value=0, max_value=100, format="%.1f"),
                        "target_hit_rate_pct": st.column_config.ProgressColumn("达标命中率(%)", min_value=0, max_value=100, format="%.1f"),
                        "state_edge": st.column_config.NumberColumn("状态边际", format="%.4f"),
                    },
                )

    with tabs[2]:
        if not isinstance(combo_panel, pd.DataFrame) or combo_panel.empty:
            st.info("状态×策略组合样本还不足，后续会随每日复盘自动累积。")
        else:
            combo_view = combo_panel.head(16).copy()
            show_cols = [
                column
                for column in [
                    "market_state_display",
                    "candidate_strategy_label",
                    "sample_count",
                    "avg_rank",
                    "avg_return_pct",
                    "intraday_high_return_pct",
                    "win_rate_pct",
                    "target_hit_rate_pct",
                    "calibration_gap_pct",
                ]
                if column in combo_view.columns
            ]
            st.dataframe(
                combo_view[show_cols],
                width="stretch",
                hide_index=True,
                column_config={
                    "market_state_display": "市场状态",
                    "candidate_strategy_label": "策略",
                    "sample_count": st.column_config.NumberColumn("样本数", format="%d"),
                    "avg_rank": st.column_config.NumberColumn("平均排名", format="%.1f"),
                    "avg_return_pct": st.column_config.NumberColumn("次日平均收益(%)", format="%.2f"),
                    "intraday_high_return_pct": st.column_config.NumberColumn("日内高点收益(%)", format="%.2f"),
                    "win_rate_pct": st.column_config.ProgressColumn("上涨胜率(%)", min_value=0, max_value=100, format="%.1f"),
                    "target_hit_rate_pct": st.column_config.ProgressColumn("达标命中率(%)", min_value=0, max_value=100, format="%.1f"),
                    "calibration_gap_pct": st.column_config.NumberColumn("概率偏差(%)", format="%.2f"),
                },
            )


def _render_daily_lightweight_model_panel(
    *,
    horizon_days: int,
    positive_return: float,
    ranking_by: str,
    board_size: int,
) -> None:
    model = load_daily_lightweight_backtest_model(
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
        board_size=board_size,
    )
    if not model:
        _section_header(
            "日度轻量模型",
            "\u72ec\u7acb\u65e5\u5ea6\u8f7b\u91cf\u56de\u6d4b\u6a21\u578b",
            "\u6bcf\u65e5\u590d\u76d8\u6837\u672c\u4f1a\u72ec\u7acb\u6c89\u6dc0\u4e3a\u8f7b\u91cf\u753b\u50cf\uff0c\u4ec5\u7528\u4e8e\u8bc4\u4f30\u4e0e\u6821\u9a8c\uff0c\u4e0d\u76f4\u63a5\u53cd\u5411\u6539\u5199\u4e3b\u6a21\u578b\u53c2\u6570\u3002",
        )
        st.info("\u5c1a\u672a\u627e\u5230\u5f53\u524d\u53c2\u6570\u5bf9\u5e94\u7684\u65e5\u5ea6\u8f7b\u91cf\u56de\u6d4b\u6a21\u578b\uff0c\u4e0b\u4e00\u6b21\u81ea\u52a8\u590d\u76d8\u540e\u4f1a\u81ea\u52a8\u751f\u6210\u3002")
        return

    panels = model.get("panels", {}) if isinstance(model.get("panels", {}), dict) else {}
    best_context = model.get("best_context", {}) if isinstance(model.get("best_context", {}), dict) else {}
    _section_header(
        "日度轻量模型",
        "\u72ec\u7acb\u65e5\u5ea6\u8f7b\u91cf\u56de\u6d4b\u6a21\u578b",
        "\u7528\u6700\u8fd1\u590d\u76d8\u6837\u672c\u62c6\u89e3\u6982\u7387\u5206\u5c42\u3001\u7b56\u7565\u3001\u5e02\u573a\u72b6\u6001\u4e0e\u4e3b\u5347\u9636\u6bb5\uff0c\u5e2e\u52a9\u5224\u65ad\u4eca\u65e5\u5173\u6ce8\u699c\u7684\u53ef\u4fe1\u5ea6\u3002",
    )

    metric_cols = st.columns(4)
    with metric_cols[0]:
        _metric_card(
            "\u6a21\u578b\u72b6\u6001",
            str(model.get("status") or "missing"),
            "\u72ec\u7acb\u8f7b\u91cf\u6821\u9a8c\uff0c\u4e0d\u53c2\u4e0e\u8fc7\u62df\u5408\u5f0f\u53c2\u6570\u53cd\u8bad",
        )
    with metric_cols[1]:
        _metric_card(
            "\u590d\u76d8\u6837\u672c",
            f'{int(model.get("sample_count", 0) or 0)}',
            f'{int(model.get("review_days", 0) or 0)} \u4e2a\u590d\u76d8\u65e5',
        )
    with metric_cols[2]:
        _metric_card(
            "\u57fa\u51c6\u80dc\u7387",
            f'{float(model.get("base_win_rate", 0.0) or 0.0) * 100:.1f}%',
            "\u65b9\u5411\u547d\u4e2d\u7387\uff0c\u4ec5\u505a\u8f85\u52a9\u89c2\u5bdf",
        )
    with metric_cols[3]:
        _metric_card(
            "\u57fa\u51c6\u6536\u76ca",
            f'{float(model.get("base_avg_return", 0.0) or 0.0) * 100:.2f}%',
            "\u6700\u8fd1\u590d\u76d8\u6837\u672c\u5e73\u5747\u6b21\u65e5\u8868\u73b0",
        )

    if best_context:
        st.caption(
            "\u5f53\u524d\u6700\u4f18\u7b56\u7565\u00d7\u5e02\u573a\u7ec4\u5408: "
            + " / ".join(
                str(best_context.get(key) or "")
                for key in ("candidate_strategy_label", "market_state_display")
                if best_context.get(key)
            )
        )

    tab_labels = [
        "\u6982\u7387\u5206\u5c42",
        "\u7b56\u7565\u753b\u50cf",
        "\u7b56\u7565\u00d7\u5e02\u573a",
        "\u4e3b\u5347\u9636\u6bb5",
    ]
    tabs = st.tabs(tab_labels)
    panel_keys = ["probability_bucket", "strategy", "strategy_market_state", "launch_phase"]
    for tab, panel_key in zip(tabs, panel_keys):
        with tab:
            panel = panels.get(panel_key, pd.DataFrame())
            if not isinstance(panel, pd.DataFrame) or panel.empty:
                st.info("\u8be5\u5206\u5c42\u6682\u65e0\u8db3\u591f\u6837\u672c\u3002")
            else:
                st.dataframe(panel.head(20), width="stretch", hide_index=True)


def _manual_market_backtest_task_key(params: dict[str, object]) -> str:
    raw = "|".join(
        [
            str(params.get("date_from") or ""),
            str(params.get("date_to") or ""),
            str(params.get("horizon_days") or ""),
            f'{float(params.get("positive_return", 0.0) or 0.0):.4f}',
            str(params.get("strategy_mode") or "all"),
            str(params.get("top_k") or 50),
            str(bool(params.get("force_rebuild", False))),
        ]
    )
    digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"manual-market-backtest::{digest}"


def _run_manual_market_backtest_task(task_key: str, params: dict[str, object]) -> dict[str, object]:
    def progress_callback(phase: str, completed: int, total: int, message: str) -> None:
        _set_async_task_progress(task_key, phase, completed, total, message)

    progress_callback("prepare", 0, 1, "准备全市场回测")
    payload = run_full_market_backtest(
        date_from=str(params["date_from"]),
        date_to=str(params["date_to"]),
        horizon_days=int(params["horizon_days"]),
        positive_return=float(params["positive_return"]),
        strategy_mode=str(params["strategy_mode"]),
        top_k=int(params["top_k"]),
        force_rebuild=bool(params.get("force_rebuild", False)),
        progress_callback=progress_callback,
    )
    summary = dict(payload.get("summary", {}) or {})
    return {
        "summary": summary,
        "summary_path": str(payload.get("summary_path") or ""),
        "results_path": str(payload.get("results_path") or ""),
    }


def _manual_backtest_strategy_label(value: object) -> str:
    token = str(value or "").lower()
    if token in {"all", ""}:
        return "全部策略"
    if "strategy1" in token or "策略1" in token:
        return "策略1"
    if "strategy2" in token or "策略2" in token:
        return "策略2"
    return str(value)


def _manual_backtest_progress_text(phase: object, message: object) -> str:
    phase_text = {
        "prepare": "准备数据",
        "feature_store": "构建特征库",
        "candidate_pool": "筛选候选池",
        "forward_eval": "验证未来收益",
        "write_outputs": "写入结果",
        "running": "运行中",
    }.get(str(phase or ""), str(phase or "运行中"))
    raw_message = str(message or "")
    translated = raw_message
    match = re.match(r"^Loaded (\d+) symbols and (\d+) trade dates$", raw_message)
    if match:
        translated = f"已加载 {match.group(1)} 只股票和 {match.group(2)} 个交易日"
    match = re.match(r"^Building feature store for (.+)$", raw_message)
    if match:
        translated = f"正在构建 {match.group(1)} 的特征库"
    match = re.match(r"^Screening candidates for (.+)$", raw_message)
    if match:
        translated = f"正在筛选 {match.group(1)} 的候选池"
    match = re.match(r"^Finished (.+), accumulated (\d+) rows$", raw_message)
    if match:
        translated = f"已完成 {match.group(1)}，累计 {match.group(2)} 条记录"
    match = re.match(r"^Saved summary to (.+)$", raw_message)
    if match:
        translated = f"摘要已保存到 {match.group(1)}"
    return f"{phase_text}：{translated or '--'}"


@st.fragment(run_every=2)
def _watch_manual_market_backtest_task(task_key: str) -> None:
    progress_state = _get_async_task_progress(task_key)
    ready, payload, error = _consume_async_task(task_key)
    if ready:
        _clear_async_task_progress(task_key)
        st.session_state.pop("manual_market_backtest_task_key", None)
        if error is not None or payload is None:
            st.error(f"手动全市场回测失败：{error}")
            return
        summary = dict(payload.get("summary", {}) or {})
        st.success(
            "手动全市场回测完成："
            f'{int(summary.get("trade_count", 0) or 0)} 笔交易，'
            f'胜率 {float(summary.get("win_rate", 0.0) or 0.0) * 100:.1f}%，'
            f'达标率 {float(summary.get("target_hit_rate", 0.0) or 0.0) * 100:.1f}%。'
        )
        st.caption(f'结果文件：{payload.get("results_path", "")}')
        st.rerun()
        return

    completed = int(progress_state.get("completed", 0) or 0)
    total = max(int(progress_state.get("total", 1) or 1), 1)
    phase = str(progress_state.get("phase", "running"))
    message = str(progress_state.get("message", "全市场回测运行中"))
    progress_value = min(max(int(round(completed / total * 100)), 0), 100)
    st.progress(progress_value, text=_manual_backtest_progress_text(phase, message))


def _render_manual_market_backtest_panel(
    *,
    latest_market_data_date: str,
    horizon_days: int,
    positive_return: float,
) -> None:
    _section_header(
        "手动回测",
        "\u624b\u52a8\u5168\u5e02\u573a\u6570\u636e\u56de\u6d4b",
        "\u6309\u6307\u5b9a\u4ea4\u6613\u65e5\u533a\u95f4\u91cd\u5efa\u6bcf\u65e5\u5019\u9009\u6c60\uff0c\u518d\u7528\u590d\u6743\u5386\u53f2\u884c\u60c5\u8bc4\u4f30\u672a\u6765\u6536\u76ca\uff1bTushare \u4ec5\u7528\u4e8e\u975e\u590d\u6743\u622a\u9762\u8865\u5145\u3002",
    )
    latest_ts = pd.to_datetime(latest_market_data_date, errors="coerce")
    default_to = latest_ts.date() if pd.notna(latest_ts) else dt.date.today()
    default_from = default_to - dt.timedelta(days=30)
    with st.expander("\u624b\u52a8\u6267\u884c\u5168\u5e02\u573a\u56de\u6d4b", expanded=False):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            date_from = st.date_input("开始日期", value=default_from, key="manual_market_backtest_date_from")
        with col_b:
            date_to = st.date_input("结束日期", value=default_to, key="manual_market_backtest_date_to")
        with col_c:
            strategy_mode = st.selectbox(
                "策略范围",
                options=["all", "strategy1", "strategy2"],
                format_func=_manual_backtest_strategy_label,
                index=0,
                key="manual_market_backtest_strategy_mode",
            )
        col_d, col_e, col_f = st.columns(3)
        with col_d:
            top_k = st.slider("每日取前 K 只", min_value=10, max_value=300, value=50, step=10)
        with col_e:
            backtest_horizon = st.selectbox(
                "预测周期",
                options=[3, 5, 10],
                format_func=lambda value: f"{value}日",
                index=[3, 5, 10].index(horizon_days) if horizon_days in {3, 5, 10} else 0,
                key="manual_market_backtest_horizon_days",
            )
        with col_f:
            backtest_target = st.slider(
                "达标涨幅(%)",
                min_value=5.0,
                max_value=50.0,
                value=float(positive_return) * 100.0,
                step=1.0,
                key="manual_market_backtest_positive_return",
            )
        force_rebuild = st.checkbox("强制重建缓存", value=False, key="manual_market_backtest_force_rebuild")
        params = {
            "date_from": pd.Timestamp(date_from).strftime("%Y-%m-%d"),
            "date_to": pd.Timestamp(date_to).strftime("%Y-%m-%d"),
            "horizon_days": int(backtest_horizon),
            "positive_return": float(backtest_target) / 100.0,
            "strategy_mode": str(strategy_mode),
            "top_k": int(top_k),
            "force_rebuild": bool(force_rebuild),
        }
        if st.button("运行全市场回测", type="primary", width="stretch"):
            task_key = _manual_market_backtest_task_key(params)
            st.session_state["manual_market_backtest_task_key"] = task_key
            _ensure_async_task(task_key, _run_manual_market_backtest_task, task_key, params)
            st.rerun()

        task_key = st.session_state.get("manual_market_backtest_task_key")
        if task_key:
            _watch_manual_market_backtest_task(str(task_key))

    latest_payload = load_latest_full_market_backtest(result_limit=20)
    if not latest_payload:
        st.info("暂无手动全市场回测结果。可以从上方面板运行一次，也可以使用命令行任务生成。")
        return
    summary = dict(latest_payload.get("summary", {}) or {})
    portfolio_metric_cols = st.columns(4)
    with portfolio_metric_cols[0]:
        _metric_card(
            "组合成交数",
            str(int(summary.get("portfolio_trade_count", summary.get("trade_count", 0)) or 0)),
            f'{summary.get("date_from", "--")} 至{summary.get("date_to", "--")}',
        )
    with portfolio_metric_cols[1]:
        _metric_card(
            "组合年化",
            f'{float(summary.get("annualized_return", 0.0) or 0.0) * 100:.2f}%',
            f'策略 {_manual_backtest_strategy_label(summary.get("strategy_mode", "all"))}',
        )
    with portfolio_metric_cols[2]:
        _metric_card(
            "最大回撤",
            f'{abs(float(summary.get("max_drawdown", 0.0) or 0.0)) * 100:.2f}%',
            f'{int(summary.get("trading_day_count", 0) or 0)} 个交易日',
        )
    with portfolio_metric_cols[3]:
        _metric_card(
            "期末权益",
            f'{float(summary.get("ending_equity", 0.0) or 0.0):,.0f}',
            f'累计 {float(summary.get("cumulative_return", 0.0) or 0.0) * 100:.2f}%',
        )
    st.caption("上方为统一组合净值口径；下方保留原筛股远期收益诊断指标。")
    metric_cols = st.columns(4)
    with metric_cols[0]:
        _metric_card("交易次数", str(int(summary.get("trade_count", 0) or 0)), f'{summary.get("date_from", "--")} 至 {summary.get("date_to", "--")}')
    with metric_cols[1]:
        _metric_card("胜率", f'{float(summary.get("win_rate", 0.0) or 0.0) * 100:.1f}%', f'{int(summary.get("trading_day_count", 0) or 0)} 个交易日')
    with metric_cols[2]:
        _metric_card("达标率", f'{float(summary.get("target_hit_rate", 0.0) or 0.0) * 100:.1f}%', f'目标 {float(summary.get("positive_return", 0.0) or 0.0) * 100:.1f}%')
    with metric_cols[3]:
        _metric_card("平均收益", f'{float(summary.get("avg_forward_return", 0.0) or 0.0) * 100:.2f}%', f'策略 {_manual_backtest_strategy_label(summary.get("strategy_mode", "all"))}')
    holding_cols = st.columns(3)
    with holding_cols[0]:
        _metric_card(
            "持有1日均值",
            f'{float(summary.get("avg_hold_1d_return", 0.0) or 0.0) * 100:.2f}%',
            f'{int(summary.get("hold_1d_sample_count", 0) or 0)} 个样本',
        )
    with holding_cols[1]:
        _metric_card(
            "持有3日均值",
            f'{float(summary.get("avg_hold_3d_return", 0.0) or 0.0) * 100:.2f}%',
            f'{int(summary.get("hold_3d_sample_count", 0) or 0)} 个样本',
        )
    with holding_cols[2]:
        _metric_card(
            "持有5日均值",
            f'{float(summary.get("avg_hold_5d_return", 0.0) or 0.0) * 100:.2f}%',
            f'{int(summary.get("hold_5d_sample_count", 0) or 0)} 个样本',
        )
    st.caption(f'最新输出：{latest_payload.get("results_path", "")}')
    portfolio_nav = latest_payload.get("portfolio_daily_nav", pd.DataFrame())
    if isinstance(portfolio_nav, pd.DataFrame) and not portfolio_nav.empty:
        st.caption(f'组合净值：{latest_payload.get("portfolio_nav_path", "")}')
    portfolio_trades = latest_payload.get("portfolio_trades", pd.DataFrame())
    if isinstance(portfolio_trades, pd.DataFrame) and not portfolio_trades.empty:
        st.caption(f'组合成交：{latest_payload.get("portfolio_trades_path", "")}')
        display_portfolio_trades = portfolio_trades.copy()
        if "candidate_strategy" in display_portfolio_trades.columns:
            display_portfolio_trades["candidate_strategy"] = display_portfolio_trades["candidate_strategy"].map(_manual_backtest_strategy_label)
        st.dataframe(
            display_portfolio_trades,
            width="stretch",
            hide_index=True,
            column_config={
                "symbol": "代码",
                "name": "名称",
                "candidate_strategy": "策略",
                "signal_date": "信号日",
                "entry_date": "买入日",
                "exit_date": "卖出日",
                "entry_price": st.column_config.NumberColumn("买入价", format="%.4f"),
                "exit_price": st.column_config.NumberColumn("卖出价", format="%.4f"),
                "shares": st.column_config.NumberColumn("股数", format="%d"),
                "gross_return": st.column_config.NumberColumn("毛收益", format="%.4f"),
                "net_return": st.column_config.NumberColumn("净收益", format="%.4f"),
            },
        )
    results = latest_payload.get("results", pd.DataFrame())
    if isinstance(results, pd.DataFrame) and not results.empty:
        display_results = results.copy()
        if "candidate_strategy" in display_results.columns:
            display_results["candidate_strategy"] = display_results["candidate_strategy"].map(_manual_backtest_strategy_label)
        st.dataframe(
            display_results,
            width="stretch",
            hide_index=True,
            column_config={
                "market_date": "交易日",
                "symbol": "代码",
                "name": "名称",
                "candidate_strategy": "策略",
                "candidate_priority": st.column_config.NumberColumn("候选排序分", format="%.2f"),
                "forward_return": st.column_config.NumberColumn("周期收益", format="%.4f"),
                "hit_target": "是否达标",
                "win": "是否上涨",
                "hold_1d_return": st.column_config.NumberColumn("持有1日收益", format="%.4f"),
                "hold_3d_return": st.column_config.NumberColumn("持有3日收益", format="%.4f"),
                "hold_5d_return": st.column_config.NumberColumn("持有5日收益", format="%.4f"),
            },
        )


def _render_daily_review_comparison(
    *,
    horizon_days: int,
    positive_return: float,
    ranking_by: str,
    board_size: int,
) -> None:
    summary = load_latest_review_summary(
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
        board_size=board_size,
    )
    details = load_latest_review_details(
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
        board_size=board_size,
    )
    if not summary or details.empty:
        return

    board_date = str(summary.get("board_date") or "--")
    review_date = str(summary.get("review_date") or "--")
    _section_header(
        "预测复盘",
        "昨日预测与今日兑现",
        f"对比 {board_date} 的榜单预测与 {review_date} 的实际走势。上涨概率仍按完整 {horizon_days} 日窗口定义；此表只展示次日兑现情况。",
    )

    metric_cols = st.columns(4)
    with metric_cols[0]:
        _metric_card("方向命中", f'{float(summary.get("direction_hit_rate_pct", 0.0) or 0.0):.2f}%', "昨日看多信号是否匹配今日方向")
    with metric_cols[1]:
        _metric_card("校准差距", f'{float(summary.get("calibration_gap_pct", 0.0) or 0.0):.2f}%', "平均预测概率与实际胜率之间的差距")
    with metric_cols[2]:
        _metric_card("Brier分数", f'{float(summary.get("direction_brier_score", 0.0) or 0.0):.4f}', "越低代表概率校准越好")
    with metric_cols[3]:
        _metric_card("目标进度", f'{float(summary.get("avg_target_progress_pct", 0.0) or 0.0):.2f}%', f'预测窗口内目标涨幅为 {positive_return * 100:.1f}%')

    comparison_view = details.copy()
    for column in ["next_day_return_pct", "intraday_high_return_pct", "target_progress_pct", "direction_error_pct"]:
        if column not in comparison_view.columns:
            comparison_view[column] = 0.0
    if "direction_hit" in comparison_view.columns:
        comparison_view["direction_hit"] = comparison_view["direction_hit"].map(lambda value: "是" if bool(value) else "否")
    if "hit_target" in comparison_view.columns:
        comparison_view["hit_target"] = comparison_view["hit_target"].map(lambda value: "是" if bool(value) else "否")

    show_cols = [
        column for column in [
            "rank", "symbol", "name", "probability_up", "next_day_return_pct", "intraday_high_return_pct",
            "target_progress_pct", "direction_hit", "direction_error_pct", "hit_target",
            "precision_gate_label", "stage_label",
        ] if column in comparison_view.columns
    ]
    if show_cols:
        st.dataframe(
            comparison_view[show_cols],
            width="stretch",
            hide_index=True,
            column_config={
                "rank": st.column_config.NumberColumn("昨日排名", format="%d"),
                "symbol": "代码",
                "name": "名称",
                "probability_up": st.column_config.NumberColumn("昨日上涨概率(%)", format="%.2f"),
                "next_day_return_pct": st.column_config.NumberColumn("收盘涨跌(%)", format="%.2f"),
                "intraday_high_return_pct": st.column_config.NumberColumn("日内高点涨幅(%)", format="%.2f"),
                "target_progress_pct": st.column_config.NumberColumn("目标进度(%)", format="%.2f"),
                "direction_hit": "方向命中",
                "direction_error_pct": st.column_config.NumberColumn("概率误差(%)", format="%.2f"),
                "hit_target": "是否达标",
                "precision_gate_label": "精度门槛",
                "stage_label": "阶段",
            },
        )

def _detail_display_context(detail: dict, summary_row: dict | pd.Series | None = None) -> dict[str, object]:
    summary = dict(summary_row) if isinstance(summary_row, (dict, pd.Series)) else {}
    detail_probability_pct = float(detail["model"].latest_probability) * 100
    detail_raw_probability_pct = float(
        getattr(detail["model"], "base_probability", detail["model"].latest_probability) or detail["model"].latest_probability
    ) * 100
    detail_predicted_upside_pct = float(getattr(detail["model"], "predicted_upside_pct", 0.0) or 0.0)
    detail_predicted_upside_low_pct = float(getattr(detail["model"], "predicted_upside_low_pct", 0.0) or 0.0)
    detail_predicted_upside_high_pct = float(getattr(detail["model"], "predicted_upside_high_pct", 0.0) or 0.0)
    detail_quant_score = float(getattr(detail["quant_signal"], "total_score", 0.0))
    detail_analysis_date = str(detail.get("analysis_date") or detail.get("snapshot", {}).get("date", "") or "")
    summary_analysis_date = str(summary.get("analysis_date") or "")
    prefer_summary_values = bool(summary) and not bool(summary.get("detail_placeholder", False))
    base_attention_score = float(
        summary.get("attention_score", detail.get("base_attention_score", detail.get("attention_score", 0.0)))
        if prefer_summary_values
        else detail.get("base_attention_score", detail.get("attention_score", 0.0))
    )
    enhanced_attention_score = float(
        summary.get("enhanced_attention_score", detail.get("enhanced_attention_score", detail.get("attention_score", 0.0)))
        if prefer_summary_values
        else detail.get("enhanced_attention_score", detail.get("attention_score", 0.0))
    )
    raw_probability_up = float(summary.get("raw_probability_up", detail.get("raw_probability_up", detail_raw_probability_pct)) if prefer_summary_values else detail.get("raw_probability_up", detail_raw_probability_pct))
    raw_attention_score = float(
        summary.get("raw_attention_score", detail.get("raw_attention_score", detail.get("base_attention_score", detail.get("attention_score", 0.0))))
        if prefer_summary_values
        else detail.get("raw_attention_score", detail.get("base_attention_score", detail.get("attention_score", 0.0)))
    )
    probability_up = float(summary.get("probability_up", detail_probability_pct) if prefer_summary_values else detail_probability_pct)
    predicted_upside_pct = float(
        summary.get("predicted_upside_pct", detail_predicted_upside_pct) if prefer_summary_values else detail_predicted_upside_pct
    )
    predicted_upside_low_pct = float(
        summary.get("predicted_upside_low_pct", detail_predicted_upside_low_pct)
        if prefer_summary_values
        else detail_predicted_upside_low_pct
    )
    predicted_upside_high_pct = float(
        summary.get("predicted_upside_high_pct", detail_predicted_upside_high_pct)
        if prefer_summary_values
        else detail_predicted_upside_high_pct
    )
    tomorrow_plan = detail["tomorrow_plan"]
    latest_market_data_date = str(detail.get("latest_market_data_date") or detail_analysis_date or "")
    context = {
        "base_attention_score": base_attention_score,
        "enhanced_attention_score": enhanced_attention_score,
        "candidate_strategy": str(summary.get("candidate_strategy", detail.get("candidate_strategy", ""))),
        "candidate_strategy_label": str(summary.get("candidate_strategy_label", detail.get("candidate_strategy_label", "通用模型"))),
        "candidate_strategy_short_label": str(
            summary.get("candidate_strategy_short_label", detail.get("candidate_strategy_short_label", "通用"))
        ),
        "candidate_strategy_forecast_bias": str(
            summary.get("candidate_strategy_forecast_bias", detail.get("candidate_strategy_forecast_bias", "按统一口径评估"))
        ),
        "candidate_strategy_note": str(
            summary.get("candidate_strategy_note", detail.get("candidate_strategy_note", "当前未命中特定硬筛选策略，按统一预测口径处理。"))
        ),
        "raw_probability_up": raw_probability_up,
        "raw_attention_score": raw_attention_score,
        "probability_up": probability_up,
        "predicted_upside_pct": predicted_upside_pct,
        "predicted_upside_low_pct": predicted_upside_low_pct,
        "predicted_upside_high_pct": predicted_upside_high_pct,
        "quant_score": float(summary.get("quant_score", detail_quant_score) if prefer_summary_values else detail_quant_score),
        "stage_label": str(summary.get("stage_label", detail["stage"].label) if prefer_summary_values else detail["stage"].label),
        "tomorrow_setup": str(summary.get("tomorrow_setup", tomorrow_plan.setup_label) if prefer_summary_values else tomorrow_plan.setup_label),
        "tomorrow_bias": str(summary.get("tomorrow_bias", tomorrow_plan.bias) if prefer_summary_values else tomorrow_plan.bias),
        "tomorrow_buy_point": str(summary.get("tomorrow_buy_point", tomorrow_plan.buy_point) if prefer_summary_values else tomorrow_plan.buy_point),
        "tomorrow_sell_point": str(
            summary.get("tomorrow_sell_point", tomorrow_plan.sell_point) if prefer_summary_values else tomorrow_plan.sell_point
        ),
        "tomorrow_plan_confidence": float(
            summary.get("tomorrow_plan_confidence", tomorrow_plan.confidence)
            if prefer_summary_values
            else tomorrow_plan.confidence
        ),
        "analysis_date": summary_analysis_date if prefer_summary_values and summary_analysis_date else detail_analysis_date or "--",
        "latest_market_data_date": latest_market_data_date or "--",
        "model_source_label": str(summary.get("model_source_label", detail.get("model_source_label", "模型来源待确认"))),
        "model_result_status": str(
            summary.get(
                "model_result_status",
                _build_model_result_status(detail_analysis_date, latest_market_data_date),
            )
        ),
        "is_aligned_with_board": prefer_summary_values and bool(summary_analysis_date) and summary_analysis_date == detail_analysis_date,
        "launch_score": float(summary.get("launch_score", detail.get("launch_score", 50.0)) if prefer_summary_values else detail.get("launch_score", 50.0)),
        "launch_readiness_score": float(
            summary.get("launch_readiness_score", detail.get("launch_readiness_score", 50.0))
            if prefer_summary_values
            else detail.get("launch_readiness_score", 50.0)
        ),
        "market_resonance_score": float(
            summary.get("market_resonance_score", detail.get("market_resonance_score", 50.0))
            if prefer_summary_values
            else detail.get("market_resonance_score", 50.0)
        ),
        "launch_specialist_score": float(
            summary.get("launch_specialist_score", detail.get("launch_specialist_score", 50.0))
            if prefer_summary_values
            else detail.get("launch_specialist_score", 50.0)
        ),
        "launch_regime_fit_score": float(
            summary.get("launch_regime_fit_score", detail.get("launch_regime_fit_score", 50.0))
            if prefer_summary_values
            else detail.get("launch_regime_fit_score", 50.0)
        ),
        "launch_specialist_confidence": float(
            summary.get("launch_specialist_confidence", detail.get("launch_specialist_confidence", 50.0))
            if prefer_summary_values
            else detail.get("launch_specialist_confidence", 50.0)
        ),
    }
    context.update(
        _build_launch_window_view(
            {
                **context,
                "snapshot": detail.get("snapshot", {}),
                "stage": detail.get("stage"),
                "intraday": detail.get("intraday"),
                "intraday_structure_signal": detail.get("intraday_structure_signal"),
                "sector_signal": detail.get("sector_signal"),
                "fund_signal": detail.get("fund_signal"),
                "news_signal": detail.get("news_signal"),
            },
            stage_code=str(_signal_value(detail.get("stage"), "code", "")),
            stage_label=str(context["stage_label"]),
        )
    )
    context.update(_evaluate_symbol_action(detail, context))
    return context


def _render_detail_freshness_banner(detail: dict, summary_row: dict | pd.Series | None = None) -> dict[str, object]:
    context = _detail_display_context(detail, summary_row)
    status_class = "freshness-pill positive" if context["is_aligned_with_board"] else "freshness-pill warning"
    alignment_text = (
        "当前个股详情与关注榜使用的是同一交易日和同一套模型结果。"
        if context["is_aligned_with_board"]
        else "当前个股详情是单股补算结果，可能比榜单主值更晚更新。"
    )
    st.markdown(
        f"""
        <div class="freshness-banner">
            <div class="freshness-title">个股详情数据说明</div>
            <div class="freshness-grid">
                <span class="freshness-pill">数据基准日 {escape(str(context["analysis_date"]))}</span>
                <span class="freshness-pill">最新收盘日 {escape(str(context["latest_market_data_date"]))}</span>
                <span class="{status_class}">{escape(str(context["model_result_status"]))}</span>
                <span class="freshness-pill">模型引擎 {escape(str(context["model_source_label"]))}</span>
            </div>
            <div class="freshness-note">{escape(alignment_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return context

def _render_intraday_panel(detail: dict) -> None:
    minute = detail["minute"]
    intraday = detail["intraday"]
    snapshot = _build_intraday_snapshot(minute)
    session_label = str(snapshot["session_label"])
    _section_header("分时工作台", "分时工作台", "分时优先，用均价线、回撤深度和尾盘承接检查当日走势是否兑现预期。")
    st.markdown(
        f'<p class="chart-caption">数据来源：最新交易日 1 分钟线 | 交易日：{escape(session_label)}</p>',
        unsafe_allow_html=True,
    )

    top_metrics = st.columns(3)
    with top_metrics[0]:
        _metric_card("分时状态", str(intraday["label"]), str(intraday["summary"]))
    with top_metrics[1]:
        _metric_card("日内涨跌", _format_pct(snapshot["session_change_pct"], signed=True), "最新价相对开盘价")
    with top_metrics[2]:
        _metric_card("均价线偏离", _format_pct(snapshot["vwap_gap_pct"], signed=True), "最新价相对均价线")

    bottom_metrics = st.columns(3)
    with bottom_metrics[0]:
        _metric_card("最大回撤", _format_pct(snapshot["max_drawdown_pct"]), "从盘中最高点回落幅度")
    with bottom_metrics[1]:
        _metric_card("尾盘承接", _format_pct(snapshot["tail_above_avg_pct"]), "近 30 分钟站上均价比例")
    with bottom_metrics[2]:
        note = f'{snapshot["bars"]} 根 1 分钟线'
        _metric_card("分时成交额", _format_short_amount(snapshot["total_amount"]), note)

    st.plotly_chart(make_minute_chart(minute), width="stretch", config={"displayModeBar": False})


def _render_symbol_banner(
    detail: dict,
    symbol: str,
    horizon_days: int,
    display_context: dict[str, object],
) -> None:
    profile = detail["profile"]
    intraday = detail["intraday"]
    rule_context = detail["rule_context"]
    stock_name = escape(str(profile.get("股票简称", symbol)))
    industry_name = escape(str(profile.get("行业", "未知")))
    board_label = escape(str(rule_context.board_label))
    price_limit_label = escape(str(rule_context.price_limit_label))
    probability_text = f'{float(display_context["probability_up"]):.1f}%'
    predicted_upside_text = f'{float(display_context.get("predicted_upside_pct", 0.0)):.1f}%'
    action_label = escape(str(display_context.get("action_label", "观察")))
    action_reason = escape(str(display_context.get("action_reason", "")))
    action_css_class = escape(str(display_context.get("action_css_class", "watch")))
    launch_window_status = escape(str(display_context.get("launch_window_status", "非启动窗")))
    launch_window_score = float(display_context.get("launch_window_score", 50.0) or 50.0)
    execution_label = escape(str(display_context.get("execution_label", "等待结构")))
    execution_window = escape(str(display_context.get("execution_window", "信号未合流")))
    selection_score = float(display_context.get("selection_score", display_context.get("base_attention_score", 50.0)) or 50.0)
    execution_score = float(display_context.get("execution_score", 50.0) or 50.0)
    st.markdown(
        f"""
        <div class="overview-banner">
            <div class="overview-main">
                <div>
                    <div class="section-tag" style="background:rgba(255,255,255,0.14);color:#eff5f8;border:1px solid rgba(255,255,255,0.08);">
                        交易工作台
                    </div>
                    <h2 class="overview-title">{stock_name} <span style="font-size:1.02rem;color:rgba(232,241,246,0.7);">{symbol}</span></h2>
                    <div class="overview-action-row">
                        <span class="overview-action-chip {action_css_class}">综合状态 {action_label}</span>
                        <span class="brief-pill">综合决策分 {float(display_context.get("action_score", 0.0)):.1f}</span>
                        <span class="brief-pill">状态置信度 {float(display_context.get("action_confidence", 0.0)):.1f}</span>
                    </div>
                    <div class="overview-action-note">{action_reason}</div>
                    <div class="overview-subtitle">{industry_name} | {board_label} | {price_limit_label} | 预测窗口 {horizon_days} 个交易日</div>
                </div>
                <div class="overview-score-card">
                    <div class="overview-score-label">未来上涨概率</div>
                    <div class="overview-score-value">{probability_text}</div>
                    <div class="overview-score-note">分时优先，日K、资金与消息做二次确认</div>
                </div>
            </div>
            <div class="overview-meta">
                <span class="brief-pill">关注分数 {float(display_context["base_attention_score"]):.1f}</span>
                <span class="brief-pill">增强分数 {float(display_context["enhanced_attention_score"]):.1f}</span>
                <span class="brief-pill">选股分 {selection_score:.1f}</span>
                <span class="brief-pill">执行分 {execution_score:.1f}</span>
                <span class="brief-pill">预测涨幅 {predicted_upside_text}</span>
                <span class="brief-pill">启动窗口 {launch_window_status} {launch_window_score:.1f}</span>
                <span class="brief-pill">执行层 {execution_label} / {execution_window}</span>
                <span class="brief-pill">K线阶段 {escape(str(display_context["stage_label"]))}</span>
                <span class="brief-pill">分时状态 {escape(str(intraday["label"]))}</span>
                <span class="brief-pill">明日形势 {escape(str(display_context["tomorrow_setup"]))}</span>
                <span class="brief-pill">数据基准日 {escape(str(display_context["analysis_date"]))}</span>
                <span class="brief-pill">模型状态 {escape(str(display_context["model_result_status"]))}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_symbol_stub(summary_row: pd.Series, horizon_days: int) -> None:
    left, right = st.columns([1.65, 1.0], gap="large")
    with left:
        st.info("榜首个股详情补充中：分时工作台、日K结构、主力资金和消息面正在后台加载。")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "股票": f'{summary_row["name"]} {summary_row["symbol"]}',
                        "数据基准日": summary_row.get("analysis_date", "--"),
                        "模型结果": summary_row.get("model_result_status", "待确认"),
                        "入选策略": summary_row.get("candidate_strategy_label", summary_row.get("candidate_strategy", "通用模型")),
                        "预测口径": summary_row.get("candidate_strategy_forecast_bias", "按统一口径评估"),
                        "板块": summary_row.get("board_label", "A股"),
                        "涨跌幅制度": summary_row.get("price_limit_label", "待更新"),
                        "关注分数": summary_row["attention_score"],
                        "增强分数": summary_row.get("enhanced_attention_score", summary_row["attention_score"]),
                        "上涨概率(%)": summary_row["probability_up"],
                        "预测涨幅(%)": summary_row.get("predicted_upside_pct", 0.0),
                        "量化辅助": summary_row["quant_score"],
                        "K线阶段": summary_row["stage_label"],
                        "明日形势": summary_row.get("tomorrow_setup", "待更新"),
                        "明日买点": summary_row.get("tomorrow_buy_point", "详情加载中"),
                        "明日卖点": summary_row.get("tomorrow_sell_point", "详情加载中"),
                        "关注理由": summary_row["reason"],
                    }
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    with right:
        _metric_card("数据基准日", str(summary_row.get("analysis_date", "--")), str(summary_row.get("model_result_status", "待确认")))
        _metric_card("当前关注分数", f'{summary_row["attention_score"]:.1f}', "已基于全市场统一排序先输出前 50 快榜单。")
        _metric_card("增强分数", f'{float(summary_row.get("enhanced_attention_score", summary_row["attention_score"])):.1f}', "板块热度、主力资金和消息面补齐后的融合分数")
        _metric_card("入选策略", str(summary_row.get("candidate_strategy_short_label", summary_row.get("candidate_strategy", "通用"))), str(summary_row.get("candidate_strategy_forecast_bias", "按统一口径评估")))
        _metric_card("交易制度", str(summary_row.get("board_label", "A股")), str(summary_row.get("price_limit_label", "待更新")))
        _metric_card("未来上涨概率", f'{summary_row["probability_up"]:.1f}%', f"{horizon_days} 个交易日窗口")
        _metric_card("预测涨幅", f'{float(summary_row.get("predicted_upside_pct", 0.0)):.1f}%', f'区间 {float(summary_row.get("predicted_upside_low_pct", 0.0)):.1f}% - {float(summary_row.get("predicted_upside_high_pct", 0.0)):.1f}%')
        _metric_card("K线阶段", str(summary_row["stage_label"]), "完整热度、资金和消息面正在后台补充。")
        _metric_card("明日形势", str(summary_row.get("tomorrow_setup", "待更新")), str(summary_row.get("tomorrow_bias", "详情加载中")))
        _metric_card("当前状态", "后台加载中", "分时图、资金图和个股解读准备好后会自动刷新。")


def _render_symbol_detail(
    detail: dict,
    symbol: str,
    horizon_days: int,
    summary_row: dict | pd.Series | None = None,
) -> None:
    display_context = _render_detail_freshness_banner(detail, summary_row)
    _render_symbol_banner(detail, symbol=symbol, horizon_days=horizon_days, display_context=display_context)
    left, right = st.columns([1.65, 1.0], gap="large")
    with left:
        _render_intraday_panel(detail)
        _section_header("日线结构", "日K结构", "用最近 120 个交易日的趋势、均线和成交量确认当前所处阶段。")
        st.markdown('<p class="chart-caption">观察 MA5 / MA20 / MA60 与成交量结构，判断趋势延续、突破还是回踩确认。</p>', unsafe_allow_html=True)
        st.plotly_chart(make_daily_chart(detail["daily"]), width="stretch", config={"displayModeBar": False})
        _section_header("资金与消息", "资金与消息面", "先看主力资金方向，再看最新消息有没有形成强化催化或风险提示。")
        tab_fund, tab_news = st.tabs(["主力资金", "最新消息面"])
        with tab_fund:
            _render_fund_flow_panel(detail["fund_flow_df"])
        with tab_news:
            _render_news(detail["news_df"])
    with right:
        stage = detail["stage"]
        intraday = detail["intraday"]
        model = detail["model"]
        backtest = detail["backtest"]
        quant_signal = detail["quant_signal"]
        sector_signal = detail["sector_signal"]
        fund_signal = detail["fund_signal"]
        news_signal = detail["news_signal"]
        _section_header("融合面板", "融合结论", "把技术结构、分时状态、板块热度、主力资金和消息面放在一起看，避免单点判断。")
        _metric_card("关注分数", f'{float(display_context["base_attention_score"]):.1f}', "这是关注榜排序使用的基础结构分")
        _metric_card("增强分数", f'{float(display_context["enhanced_attention_score"]):.1f}', "这是叠加板块热度、主力资金和消息面后的融合分数")
        _metric_card("选股分", f'{float(display_context.get("selection_score", display_context["base_attention_score"])):.1f}', f'选股置信度 {float(display_context.get("selection_confidence", 50.0)):.1f}')
        _metric_card("执行分", f'{float(display_context.get("execution_score", 50.0)):.1f}', f'执行置信度 {float(display_context.get("execution_confidence", 50.0)):.1f}')
        _metric_card("原始模型概率", f'{float(display_context.get("raw_probability_up", display_context["probability_up"])):.1f}%', "未叠加策略偏置与回放校准的统一模型输出")
        _metric_card("未来上涨概率", f'{float(display_context["probability_up"]):.1f}%', f"{horizon_days} 个交易日窗口")
        _metric_card("预测涨幅", f'{float(display_context.get("predicted_upside_pct", 0.0)):.1f}%', f'区间 {float(display_context.get("predicted_upside_low_pct", 0.0)):.1f}% - {float(display_context.get("predicted_upside_high_pct", 0.0)):.1f}%')
        _metric_card("K线阶段", str(display_context["stage_label"]), stage.intraday_expectation)
        _metric_card("分时状态", intraday["label"], intraday["summary"])
        _metric_card("启动窗口", f'{str(display_context.get("launch_window_status", "非启动窗"))} / {float(display_context.get("launch_window_score", 50.0)):.1f}', str(display_context.get("launch_window_summary", "等待结构与共振进一步确认")))
        _metric_card("执行窗口", str(display_context.get("execution_window", "信号未合流")), str(display_context.get("execution_summary", "等待分时、平台位和量能进一步确认")))
        _metric_card("计划买点", str(display_context.get("execution_entry_zone", "等待更低风险的切入位置")), str(display_context.get("chase_risk_label", "先避免追高")))
        _metric_card("失效条件", str(display_context.get("execution_invalidation_rule", "等待结构确认")), f'盈亏比 {str(display_context.get("reward_risk_label", "待确认"))} / 比值 {float(display_context.get("reward_risk_ratio", 0.0)):.2f}')
        _metric_card("综合状态", str(display_context["action_label"]), f'综合决策分 {float(display_context["action_score"]):.1f} / 状态置信度 {float(display_context["action_confidence"]):.1f}')
        if display_context.get("action_reason_lines"):
            st.markdown("#### 状态依据")
            st.markdown("\n".join(f"- {line}" for line in display_context["action_reason_lines"]))
        metric_left, metric_right = st.columns(2)
        with metric_left:
            _metric_card("行业热度", f'{sector_signal["sector_score"]:.1f}', str(sector_signal["sector_summary"]))
            _metric_card("主力资金", f'{fund_signal["fund_score"]:.1f}', str(fund_signal["summary"]))
        with metric_right:
            _metric_card("消息面情绪", f'{news_signal["sentiment_score"]:.1f}', str(news_signal["summary"]))
            _metric_card("量化辅助", f"{quant_signal.total_score:.1f}", f"{quant_signal.primary_signal} / {quant_signal.summary}")
        _section_header("本地模型", "本地量化模型", "本地部署的时序集成模型会用历史样本训练，再对当前最新结构给出方向和风险判断。")
        _metric_card("模型信号", model.signal_label, model.backtest_summary)
        _metric_card("策略分数", f"{model.strategy_score:.1f}", f"模型一致性 {model.agreement_score:.1f}")
        _metric_card("模型质量", model.quality_label, model.risk_label)
        _section_header("日频回测", "日频回测", "按时间顺序训练和模拟交易，检验这套策略在历史每日更新数据下能否稳定工作。")
        _metric_card("策略状态", backtest.status_label, backtest.summary)
        _metric_card("目标命中率", f"{backtest.target_precision * 100:.0f}%", backtest.selection_summary)
        _metric_card("历史命中率", f"{float(getattr(backtest, 'achieved_precision', 0.0)) * 100:.1f}%", f"样本 {int(getattr(backtest, 'trade_count', 0) or 0)} 笔")
        _metric_card("最新信号状态", "放行" if bool(getattr(backtest, "latest_signal_active", False)) else "观察", "以滚动回测结果衡量当前信号是否满足实战放行门槛")


def _render_async_sections(
    *,
    board_task_key: str,
    detail_task_key: str,
    board_context_key: str,
    detail_context_key: str,
    summary_row: dict | pd.Series,
    symbol: str,
    horizon_days: int,
) -> None:
    board_error_message: str | None = None
    detail_error_message: str | None = None
    rerun_required = False

    board_ready, board_payload, board_error = _consume_async_task(board_task_key)
    if board_ready:
        if board_error is None and isinstance(board_payload, tuple) and len(board_payload) == 2:
            enhanced_board, market_context = board_payload
            if isinstance(enhanced_board, pd.DataFrame):
                st.session_state["enhanced_board_context_key"] = board_context_key
                st.session_state["enhanced_board"] = enhanced_board
            if isinstance(market_context, dict):
                st.session_state["market_context_context_key"] = board_context_key
                st.session_state["market_context_async"] = market_context
            rerun_required = True
        else:
            board_error_message = "榜单增强补充失败，当前先保留快榜单。"

    detail_ready, detail_payload, detail_error = _consume_async_task(detail_task_key)
    if detail_ready:
        if detail_error is None and isinstance(detail_payload, dict):
            st.session_state["detail_context_key"] = detail_context_key
            st.session_state["detail_async"] = detail_payload
            rerun_required = True
        else:
            detail_error_message = f"{symbol} 详情补充失败，当前先保留榜单概览。"

    if rerun_required:
        st.rerun()

    _section_header(
        "市场脉搏",
        "市场热度与宏观",
        "行业资金、概念资金和宏观日历会在榜单就绪后异步补齐，避免首页等待过久。",
    )
    market_context = None
    if st.session_state.get("market_context_context_key") == board_context_key:
        cached_market_context = st.session_state.get("market_context_async")
        if isinstance(cached_market_context, dict):
            market_context = cached_market_context

    if market_context is not None:
        _render_market_context(market_context)
    elif board_error_message:
        st.warning(board_error_message)
    else:
        st.info("行业资金、概念资金和宏观日历正在后台补充，完成后会自动刷新。")

    detail_result = None
    if st.session_state.get("detail_context_key") == detail_context_key:
        cached_detail = st.session_state.get("detail_async")
        if isinstance(cached_detail, dict):
            detail_result = cached_detail

    if detail_result is not None:
        _render_symbol_detail(detail_result, symbol=symbol, horizon_days=horizon_days, summary_row=summary_row)
    elif detail_error_message:
        st.warning(detail_error_message)
        _render_symbol_stub(pd.Series(summary_row), horizon_days)
    else:
        _render_symbol_stub(pd.Series(summary_row), horizon_days)


@st.fragment(run_every=2)
@st.fragment(run_every=2)
@st.fragment(run_every=2)
def _watch_pending_view_update() -> None:
    pending_view_params = st.session_state.get("pending_view_params")
    task_key = st.session_state.get("pending_board_update_task_key")
    if not pending_view_params or not task_key:
        return

    ready, payload, error = _consume_async_task(task_key)
    if ready:
        _clear_pending_view_update()
        if error is not None or payload is None:
            st.error("参数切换失败，当前先保留上一版榜单。")
            return

        st.session_state["active_view_params"] = payload["view_params"]
        st.session_state["active_board_override_key"] = payload["board_key"]
        st.session_state["active_board_override"] = payload["board"]
        _clear_async_ui_state()
        st.rerun()

    st.info(
        "参数切换计算中："
        f'{_view_params_summary(pending_view_params)}，完成后会自动刷新页面。'
    )

def _watch_latest_closed_market_result_probe(
    board: pd.DataFrame,
    *,
    custom_watchlist: tuple[str, ...],
    horizon_days: int,
    positive_return: float,
    refresh_seconds: int,
) -> None:
    if refresh_seconds or custom_watchlist or board.empty:
        return
    latest_market_data_date = _latest_market_close_date()
    state = _ensure_market_refresh_task_for_board(
        board,
        custom_watchlist=custom_watchlist,
        horizon_days=horizon_days,
        positive_return=positive_return,
        latest_market_data_date=latest_market_data_date,
    )
    if bool(state.get("started_now")):
        st.rerun()


@st.fragment(run_every=2)
@st.fragment(run_every=2)
@st.fragment(run_every=2)
def _watch_market_ranking_refresh(
    task_key: str,
    cached_market_data_date: str | None,
    latest_market_data_date: str | None,
    refresh_reason: str = "",
) -> None:
    progress_state = _get_async_task_progress(task_key)
    ready, payload, error = _consume_async_task(task_key)
    if ready:
        _clear_async_task_progress(task_key)
        if error is not None or payload is None:
            st.warning("全市场榜单后台重算失败，当前先保留已有结果。")
            return

        load_market_rankings.clear()
        load_focus_board.clear()
        _clear_active_board_override()
        _clear_async_ui_state()
        st.session_state["market_refresh_notice"] = (
            f'全市场榜单已刷新到 {payload.get("market_data_date", latest_market_data_date or "--")}，'
            f'当前可用股票数 {int(payload.get("row_count", 0))}。'
        )
        st.rerun()

    reason_suffix = ""
    if refresh_reason == "stale_results":
        reason_suffix = " 检测到榜单结果不是最新交易日，系统正在后台重算。"
    elif refresh_reason == "stale_cache":
        reason_suffix = " 检测到缓存过期，系统正在后台重算。"
    elif refresh_reason == "quick_board_pending":
        reason_suffix = " 当前已先展示最新收盘快榜，系统正在后台补齐批量特征、完整版预测和回测。"

    st.info(
        f'后台正在重算全市场榜单：最新收盘日 `{latest_market_data_date or "--"}`，'
        f'当前缓存基准日 `{cached_market_data_date or "--"}`。{reason_suffix}'
    )
    completed = int(progress_state.get("completed", 0) or 0)
    total = max(int(progress_state.get("total", 1) or 1), 1)
    progress_value = min(max(int(round(completed / total * 100)), 0), 100)
    phase = str(progress_state.get("phase", "后台计算"))
    message = str(progress_state.get("message", "正在处理最新交易日榜单"))
    st.progress(progress_value, text=f"{phase}: {message}")
    st.caption(f"已完成 {completed}/{total}")

@st.fragment(run_every=2)
def _watch_daily_review_maintenance(task_key: str) -> None:
    progress_state = _get_async_task_progress(task_key)
    ready, payload, error = _consume_async_task(task_key)
    if ready:
        _clear_async_task_progress(task_key)
        if error is not None or payload is None:
            st.warning("上一交易日榜单复盘失败，当前先保留已有校准结果。")
            return

        new_reviews = int(payload.get("new_reviews", 0) or 0)
        latest_summary = payload.get("latest_summary") or {}
        if new_reviews > 0:
            review_date = str(latest_summary.get("review_date") or "--")
            st.session_state["daily_review_notice"] = f"上一交易日榜单复盘已完成，最新验证交易日：{review_date}。"
            load_focus_board.clear()
            _clear_active_board_override()
            st.rerun()
        return

    completed = int(progress_state.get("completed", 0) or 0)
    total = max(int(progress_state.get("total", 1) or 1), 1)
    if completed > 0:
        st.caption(f"上一交易日榜单复盘进行中：{completed}/{total}")

def main() -> None:
    load_env_file()
    if "active_view_params" not in st.session_state:
        st.session_state["active_view_params"] = dict(DEFAULT_VIEW_PARAMS)
    active_view_params = dict(st.session_state["active_view_params"])

    
    with st.sidebar:
        st.header("页面参数")
        refresh_seconds = st.select_slider("自动刷新（秒）", options=[0, 30, 60, 120], value=0)
        ranking_by = st.radio("关注榜排序方式", options=["关注分数", "上涨概率"], index=0)
        board_size = st.slider("关注榜展示数量", min_value=10, max_value=100, value=50, step=10)
        horizon_days = st.selectbox("预测周期（交易日）", options=[3, 5, 10], index=0)
        positive_return = st.slider("定义“看涨”的涨幅阈值", min_value=5.0, max_value=50.0, value=10.0, step=1.0) / 100
        watchlist_text = st.text_area(
            "自定义股票池（可空）",
            value="",
            help="支持逗号、空格、换行分隔，例如：600519 000333 002594",
        )
        st.caption("留空时默认展示全市场关注榜前 50 和榜首个股详情；如果填写，则只在自定义股票池内排名。")
        refresh_heat = st.button("刷新热度数据", width="stretch")
        rebuild_rankings = st.button("重算全市场榜单", width="stretch")
        st.caption("刷新热度数据会更新行业资金、消息面、主力资金和榜单增强；重算全市场榜单会重新扫描全市场最新日线并更新缓存。")

    apply_params = st.sidebar.button("应用参数", width="stretch", type="primary")
    st.sidebar.caption("调整阈值或预测周期后，先点“应用参数”再重算，避免拖动滑块时反复刷新。")

    if apply_params:
        pending_view_params = _normalize_view_params(
            {
                "refresh_seconds": int(refresh_seconds),
                "ranking_by": ranking_by,
                "board_size": int(board_size),
                "horizon_days": int(horizon_days),
                "positive_return": float(positive_return),
                "watchlist_text": watchlist_text,
            }
        )
        if pending_view_params != _normalize_view_params(active_view_params):
            task_key = _focus_board_request_key(pending_view_params)
            st.session_state["pending_view_params"] = pending_view_params
            st.session_state["pending_board_update_task_key"] = task_key
            _ensure_async_task(task_key, _build_focus_board_payload, pending_view_params)
            st.rerun()
        st.session_state["active_view_params"] = pending_view_params
        _clear_pending_view_update()
        _clear_active_board_override()
        _clear_async_ui_state()
        st.rerun()

    if refresh_heat:
        _clear_pending_view_update()
        _clear_async_ui_state()
        _clear_heat_data_caches()
        st.session_state.pop("forced_market_refresh_date", None)
        st.rerun()

    if rebuild_rankings:
        _clear_pending_view_update()
        _clear_async_ui_state()
        _clear_heat_data_caches()
        _clear_market_ranking_caches()
        st.session_state.pop("forced_market_refresh_date", None)
        st.rerun()

    active_view_params = dict(st.session_state["active_view_params"])
    refresh_seconds = int(active_view_params["refresh_seconds"])
    ranking_by = str(active_view_params["ranking_by"])
    board_size = int(active_view_params["board_size"])
    horizon_days = int(active_view_params["horizon_days"])
    positive_return = float(active_view_params["positive_return"])
    watchlist_text = str(active_view_params["watchlist_text"])

    if refresh_seconds:
        st_autorefresh(interval=refresh_seconds * 1000, key="refresh")

    custom_watchlist = tuple(parse_watchlist(watchlist_text))
    clock = market_clock()
    if not custom_watchlist:
        completed_refresh_adopted = _adopt_completed_market_refresh(
            _market_rank_refresh_async_key(horizon_days, positive_return)
        )
        if completed_refresh_adopted:
            load_focus_board.clear()
    active_board_key = _focus_board_request_key(active_view_params)
    board = st.session_state.get("active_board_override")
    if st.session_state.get("active_board_override_key") != active_board_key or board is None:
        board = _load_history_first_focus_board(
            board_size=board_size,
            custom_watchlist=custom_watchlist,
            horizon_days=horizon_days,
            positive_return=positive_return,
            ranking_by=ranking_by,
        )
        if board.empty:
            board = load_focus_board(
                board_size=board_size,
                custom_watchlist=custom_watchlist,
                horizon_days=horizon_days,
                positive_return=positive_return,
                ranking_by=ranking_by,
            )
    if board.empty:
        st.warning("褰撳墠娌℃湁绛涘嚭鍙睍绀虹殑鑲＄エ锛岃璋冩暣鑲＄エ姹犳垨绋嶅悗鍐嶈瘯銆")
        return

    data_mode = board.attrs.get("data_mode", "unknown")
    market_refresh_state = _ensure_market_refresh_task_for_board(
        board,
        custom_watchlist=custom_watchlist,
        horizon_days=horizon_days,
        positive_return=positive_return,
    )
    cache_stale = bool(market_refresh_state.get("cache_stale", False))
    cached_market_data_date = market_refresh_state.get("cached_market_data_date")
    latest_market_data_date = market_refresh_state.get("latest_market_data_date")
    refresh_reason = str(market_refresh_state.get("refresh_reason", "") or "")
    market_refresh_task_key = market_refresh_state.get("task_key")
    board_task_key = _board_async_key(board, ranking_by=ranking_by, data_mode=data_mode)
    rendered_board = board
    if st.session_state.get("enhanced_board_context_key") == board_task_key and st.session_state.get("enhanced_board") is not None:
        rendered_board = st.session_state["enhanced_board"]
    else:
        _ensure_async_task(
            board_task_key,
            _build_enhanced_focus_board,
            board,
            ranking_by,
            data_mode,
            horizon_days,
            positive_return,
        )
    try:
        persist_focus_board_snapshot(
            rendered_board,
            horizon_days=horizon_days,
            positive_return=positive_return,
            ranking_by=ranking_by,
            board_size=board_size,
        )
    except Exception:
        pass

    daily_review_task_key = None
    if not custom_watchlist and not _latest_daily_review_is_current(
        rendered_board,
        ranking_by=ranking_by,
        horizon_days=horizon_days,
        positive_return=positive_return,
        board_size=board_size,
    ):
        daily_review_task_key = _daily_review_async_key(
            rendered_board,
            ranking_by=ranking_by,
            horizon_days=horizon_days,
            positive_return=positive_return,
            board_size=board_size,
        )
        _ensure_async_task(
            daily_review_task_key,
            _run_daily_review_maintenance_task,
            daily_review_task_key,
            rendered_board,
            horizon_days,
            positive_return,
            ranking_by,
            board_size,
        )

    top_left, top_mid, top_right = st.columns(3)
    with top_left:
        _metric_card("北京时间", clock["now"], clock["status"])
    with top_mid:
        refresh_mode = "自动刷新" if refresh_seconds else "按需刷新"
        refresh_note = clock["note"] if refresh_seconds else "榜单先快速展示，热度增强和榜首详情在后台自动补齐。"
        _metric_card("刷新机制", refresh_mode, refresh_note)
    with top_right:
        watched = "自定义股票池" if custom_watchlist else "全市场 A 股统一排名"
        if data_mode == "live":
            mode_note = "实时快照 + 在线热度增强"
        elif data_mode == "strategy_candidate_pool":
            mode_note = "按策略1/策略2硬筛选构建正式关注榜，连续上涨榜仅作为空结果时的显式兜底"
        elif data_mode == "dynamic_fallback_pool":
            mode_note = "按最新收盘日动态筛出的连续上涨兜底池"
        elif data_mode == "fallback_watchlist":
            mode_note = "网络受限，已切换静态核心股票池兜底"
        else:
            mode_note = "快榜单 + 最新日线统一排名"
        _metric_card("当前模式", watched, mode_note)

    st.caption(f"已应用参数：{_view_params_summary(active_view_params)}")
    refresh_notice = st.session_state.pop("market_refresh_notice", None)
    if refresh_notice:
        st.success(refresh_notice)
    daily_review_notice = st.session_state.pop("daily_review_notice", None)
    if daily_review_notice:
        st.success(daily_review_notice)
    _watch_latest_closed_market_result_probe(
        rendered_board,
        custom_watchlist=custom_watchlist,
        horizon_days=horizon_days,
        positive_return=positive_return,
        refresh_seconds=refresh_seconds,
    )
    _watch_pending_view_update()
    if market_refresh_task_key:
        _watch_market_ranking_refresh(
            market_refresh_task_key,
            str(cached_market_data_date or ""),
            str(latest_market_data_date or ""),
            refresh_reason,
        )
    if daily_review_task_key:
        _watch_daily_review_maintenance(daily_review_task_key)

    if "未开盘" in clock["status"] or "收盘" in clock["status"] or "休市" in clock["status"]:
        st.info("当前不是 A 股交易时段，所以“当日分钟”分时图可能为空；关注榜、日 K、新闻、行业资金和主力资金仍会显示最新可得数据。")
    if data_mode == "strategy_candidate_pool":
        st.info("当前数据来源：关注榜已切换为策略1/策略2正式硬筛选，不是连续上涨榜；只有策略结果为空时才会退回显式连续上涨兜底池。")
    if data_mode == "dynamic_fallback_pool":
        st.info("当前全市场实时快照不可用，首页已改用按最新收盘日动态筛出的连续上涨兜底池；搜索任意 A 股详情仍可正常查看。")
    elif data_mode == "fallback_watchlist":
        st.warning("当前全市场日线主数据源不可用，首页已自动回退到核心股票池兜底模式；搜索任意 A 股详情仍可正常查看。")

    _section_header(
        "关注榜",
        "全市场关注榜",
        f"先对全市场 A 股统一计算 {ranking_by}，再截取前 {board_size} 名；页面会先秒出快榜单，再后台补齐热度增强与榜首详情。",
    )
    _render_board_freshness_banner(rendered_board)
    _render_daily_review_banner(
        rendered_board,
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
        board_size=board_size,
    )
    _render_review_battle_panels(
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
        board_size=board_size,
    )
    _render_daily_lightweight_model_panel(
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
        board_size=board_size,
    )
    _render_manual_market_backtest_panel(
        latest_market_data_date=str(latest_market_data_date or rendered_board.attrs.get("latest_market_data_date") or ""),
        horizon_days=horizon_days,
        positive_return=positive_return,
    )
    _render_daily_review_comparison(
        horizon_days=horizon_days,
        positive_return=positive_return,
        ranking_by=ranking_by,
        board_size=board_size,
    )
    st.caption(f"当前关注榜模型架构：v{MODEL_SCHEMA_VERSION}。关注榜会优先读取新的全市场 v2 缓存，并在缓存完成后自动切换。")
    if rendered_board.attrs.get("loading", False):
        st.info("快榜单已就绪，板块热度、主力资金、消息面和榜首详情正在后台补充。")

    if rendered_board.attrs.get("focus_filter_mode") == "fallback":
        st.warning("当前缓存候选里暂无满足连续 3 天以上上涨的股票，首页先展示最近最接近条件的候补榜，避免首屏空白。你仍然可以手动刷新或搜索任意 A 股查看详情。")
    elif rendered_board.attrs.get("focus_filter_mode") == "supplemented":
        st.info("当前满足连续 3 天以上上涨的股票不足 10 支，系统已自动补充最接近条件的候选，避免关注榜过窄。")

    _top_focus_cards(rendered_board)
    _render_focus_board_tables(rendered_board)

    _section_header("个股工作台", "个股工作台", "默认打开榜首个股，也支持搜索任意 A 股进入详情页。")
    st.caption("搜索范围：全市场 A 股，可按代码或名称检索，支持 `600519`、`sh600519`、`600519.SH`、`贵州茅台`。")
    detail_mode = st.radio(
        "选择查看方式",
        options=["从关注榜进入", "搜索任意A股"],
        index=0,
        horizontal=True,
    )
    symbol = str(rendered_board.iloc[0]["symbol"])
    selected_name = str(rendered_board.iloc[0]["name"])

    if detail_mode == "从关注榜进入":
        symbol_map = {f'{row["name"]} {row["symbol"]}': row["symbol"] for _, row in rendered_board.iterrows()}
        choice = st.selectbox("查看个股详情", options=list(symbol_map.keys()), index=0)
        symbol = symbol_map[choice]
        selected_name = choice.rsplit(" ", 1)[0]
    else:
        a_share_universe = load_a_share_universe()
        query = st.text_input(
            "搜索任意 A 股代码或名称",
            value="",
            placeholder="例如：600519 / 贵州茅台 / 宁德时代",
        )
        if not query.strip():
            st.info("输入股票代码或名称后，会在全市场 A 股股票池中搜索并展示详情；当前先默认显示关注榜第一名。")
        else:
            matches = search_a_share_universe(a_share_universe, query, limit=30)
            if matches.empty:
                st.warning("没有匹配到股票，当前先默认显示关注榜第一名，请换一个代码或名称再试。")
            else:
                search_map = {f'{row["name"]} {row["symbol"]}': row["symbol"] for _, row in matches.iterrows()}
                st.caption(f"当前匹配到 {len(matches)} 只股票，结果按精确代码/名称优先排序。")
                choice = st.selectbox("搜索结果", options=list(search_map.keys()), index=0)
                symbol = search_map[choice]
                selected_name = choice.rsplit(" ", 1)[0]

    st.markdown("#### 任意 A 股直达")
    direct_search_col, direct_board_col = st.columns([1.5, 1.0], gap="large")
    with direct_search_col:
        direct_query = st.text_input(
            "任意 A 股搜索",
            value="",
            key="direct_symbol_query",
            placeholder="例如：600519 / sh600519 / 600519.SH / 贵州茅台",
        )
        direct_matches = pd.DataFrame()
        if direct_query.strip():
            direct_matches, direct_symbol, direct_name = _resolve_search_candidate(load_a_share_universe(), direct_query)
            if direct_matches.empty and direct_symbol is None:
                st.warning("当前没有匹配到 A 股股票，请换一个代码或名称再试。")
            else:
                if direct_symbol is not None:
                    symbol = direct_symbol
                    selected_name = direct_name or direct_symbol
                if len(direct_matches) > 1:
                    direct_map = {
                        f'{row["name"]} {row["symbol"]}': row["symbol"] for _, row in direct_matches.iterrows()
                    }
                    default_index = 0
                    for idx, match_symbol in enumerate(direct_map.values()):
                        if match_symbol == symbol:
                            default_index = idx
                            break
                    direct_choice = st.selectbox(
                        "搜索结果",
                        options=list(direct_map.keys()),
                        index=default_index,
                        key="direct_symbol_choice",
                    )
                    symbol = direct_map[direct_choice]
                    selected_name = direct_choice.rsplit(" ", 1)[0]
                elif not direct_matches.empty:
                    direct_row = direct_matches.iloc[0]
                    symbol = str(direct_row["symbol"])
                    selected_name = str(direct_row["name"])
                st.caption(f"已切换到 {selected_name} {symbol}")

    with direct_board_col:
        st.caption("不搜索时，详情区仍可按关注榜快速切换。")
        quick_symbol_map = {f'{row["name"]} {row["symbol"]}': row["symbol"] for _, row in rendered_board.iterrows()}
        quick_choice = st.selectbox(
            "关注榜快速切换",
            options=list(quick_symbol_map.keys()),
            index=0,
            key="direct_board_symbol_choice",
        )
        if not st.session_state.get("direct_symbol_query", "").strip():
            symbol = quick_symbol_map[quick_choice]
            selected_name = quick_choice.rsplit(" ", 1)[0]

    summary_match = rendered_board.loc[rendered_board["symbol"] == symbol]
    if summary_match.empty:
        summary_row = {
            "symbol": symbol,
            "name": selected_name,
            "detail_placeholder": True,
            "analysis_date": rendered_board.attrs.get("market_data_date", "--"),
            "model_result_status": _build_model_result_status(
                rendered_board.attrs.get("market_data_date", ""),
                rendered_board.attrs.get("latest_market_data_date", ""),
                cache_stale=bool(rendered_board.attrs.get("cache_stale", False)),
            ),
            "model_source_label": rendered_board.attrs.get("model_source_label", "模型来源待确认"),
            "board_label": "A股",
            "price_limit_label": "涨跌幅制度待识别",
                "attention_score": 0.0,
                "enhanced_attention_score": 0.0,
                "candidate_strategy": "",
                "candidate_strategy_label": "通用模型",
                "candidate_strategy_short_label": "通用",
                "candidate_strategy_forecast_bias": "按统一口径评估",
                "candidate_strategy_note": "当前未命中特定硬筛选策略，按统一预测口径处理。",
                "probability_up": 0.0,
            "predicted_upside_pct": 0.0,
            "predicted_upside_low_pct": 0.0,
            "predicted_upside_high_pct": 0.0,
            "quant_score": 0.0,
            "stage_label": "详情加载中",
            "tomorrow_setup": "待评估",
            "tomorrow_bias": "详情加载中",
            "tomorrow_buy_point": "后台正在准备明日买点。",
            "tomorrow_sell_point": "后台正在准备明日卖点。",
            "reason": "该股票不在当前榜单中，正在后台准备完整详情。",
        }
    else:
        summary_row = summary_match.iloc[0].to_dict()
        summary_row["detail_placeholder"] = False

    if detail_mode == "搜索任意A股" or summary_match.empty:
        st.caption(f"当前个股详情补充仅计算 {selected_name} {symbol} 这单只个股，不会重算关注榜其他股票。")

    detail_task_key = _detail_async_key(symbol, horizon_days, positive_return)
    if not (
        st.session_state.get("detail_context_key") == detail_task_key and st.session_state.get("detail_async") is not None
    ):
        _ensure_async_task(detail_task_key, _build_symbol_detail, symbol, horizon_days, positive_return)

    _render_async_sections(
        board_task_key=board_task_key,
        detail_task_key=detail_task_key,
        board_context_key=board_task_key,
        detail_context_key=detail_task_key,
        summary_row=summary_row,
        symbol=symbol,
        horizon_days=horizon_days,
    )

    _section_header("使用说明", "现在这套页面如何使用", "先看主线，再看个股阶段与分时，最后用资金和消息做确认。")
    usage_left, usage_right = st.columns(2)
    with usage_left:
        st.markdown(
            """
            - 先看“行业资金”和“概念资金”，知道今天市场抱团在哪
            - 再看关注榜，优先挑 `板块热度`、`消息面`、`量化辅助` 同时不差的票
            - 个股详情里先看 K 线阶段，再看分时是否兑现阶段预期
            - 主力资金和消息面用来做最后一道过滤
            """
        )
    with usage_right:
        st.markdown(
            """
            - `量化辅助` 是辅助，不代替交易纪律
            - `消息面情绪` 只做轻量打分，避免被单条新闻带偏
            - `行业热度` 反映当日资金关注方向，不代表每只股票都适合追
            - 非交易时段页面依旧可看，但分时会缺失，重心回到日 K、消息和资金流
            """
        )


if __name__ == "__main__":
    main()
