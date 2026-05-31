from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from .quant import evaluate_news_sentiment


def _clip(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return float(max(lower, min(value, upper)))


def _pct(value: float | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value) * 100:.{digits}f}%"


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


@dataclass(slots=True)
class TradingRuleContext:
    board_label: str
    price_limit_pct: float
    price_limit_label: str
    t_plus_one_label: str
    auction_rule: str
    closing_rule: str
    execution_warning: str
    rule_summary: str


@dataclass(slots=True)
class TemporalNewsPulse:
    intraday_score: float
    overnight_score: float
    next_session_score: float
    stronger_window: str
    summary: str


@dataclass(slots=True)
class IntradayStructureSignal:
    opening_volume_ratio: float
    first30_volume_share: float
    early_return_pct: float
    label: str
    summary: str


@dataclass(slots=True)
class StrategyWorkbench:
    strategy_score: float
    style: str
    entry_window: str
    exit_rule: str
    summary: str
    drivers: tuple[str, ...]


@dataclass(slots=True)
class LaunchWindowAssessment:
    window_score: float
    window_confidence: float
    label: str
    status: str
    summary: str
    drivers: tuple[str, ...]


@dataclass(slots=True)
class ExecutionAssessment:
    execution_score: float
    execution_confidence: float
    label: str
    window: str
    summary: str
    entry_zone: str
    invalidation_rule: str
    reward_risk_label: str
    expected_return_pct: float
    drawdown_risk_pct: float
    reward_risk_ratio: float
    chase_risk_label: str
    drivers: tuple[str, ...]


def _board_from_symbol(symbol: str) -> str:
    normalized = (symbol or "").strip()
    if normalized.startswith(("688", "689")):
        return "科创板"
    if normalized.startswith(("300", "301")) or normalized.startswith("30"):
        return "创业板"
    if normalized.startswith(("600", "601", "603", "605")):
        return "沪主板"
    if normalized.startswith(("000", "001", "002", "003")):
        return "深主板"
    return "A股"


def build_trading_rule_context(symbol: str, name: str = "", profile: Mapping | None = None) -> TradingRuleContext:
    board_label = _board_from_symbol(symbol)
    profile_name = ""
    if profile is not None:
        profile_name = str(profile.get("股票简称") or profile.get("证券简称") or "")
    name_text = f"{name} {profile_name}".upper()
    is_st = "ST" in name_text

    if is_st:
        price_limit_pct = 5.0
    elif board_label in {"创业板", "科创板"}:
        price_limit_pct = 20.0
    else:
        price_limit_pct = 10.0

    price_limit_label = f"日涨跌幅限制 {price_limit_pct:.0f}%"
    t_plus_one_label = "股票按 T+1 交易，买入当日不能卖出"
    auction_rule = "9:15-9:25 集合竞价；9:20-9:25 不能撤单"
    closing_rule = "14:57-15:00 收盘集合竞价，委托单不能撤单"
    execution_warning = "尽量回避 9:20-9:25 和 14:57-15:00 的情绪化追单"
    rule_summary = f"{board_label} | {price_limit_label} | {t_plus_one_label}"

    return TradingRuleContext(
        board_label=board_label,
        price_limit_pct=price_limit_pct,
        price_limit_label=price_limit_label,
        t_plus_one_label=t_plus_one_label,
        auction_rule=auction_rule,
        closing_rule=closing_rule,
        execution_warning=execution_warning,
        rule_summary=rule_summary,
    )


def evaluate_temporal_news_pulse(news_df: pd.DataFrame) -> TemporalNewsPulse:
    if news_df.empty or "发布时间" not in news_df.columns:
        return TemporalNewsPulse(
            intraday_score=50.0,
            overnight_score=50.0,
            next_session_score=50.0,
            stronger_window="暂无时段优势",
            summary="没有足够的分时段消息样本，明日预案仍以量价和阶段结构为主。",
        )

    news = news_df.copy()
    news["发布时间"] = pd.to_datetime(news["发布时间"], errors="coerce")
    intraday_mask = news["发布时间"].dt.time.between(dt.time(9, 30), dt.time(14, 59, 59), inclusive="both")
    intraday_news = news[intraday_mask.fillna(False)]
    overnight_news = news[~intraday_mask.fillna(False)]

    intraday_signal = evaluate_news_sentiment(intraday_news)
    overnight_signal = evaluate_news_sentiment(overnight_news)
    intraday_score = _safe_float(intraday_signal.get("sentiment_score"), 50.0)
    overnight_score = _safe_float(overnight_signal.get("sentiment_score"), 50.0)
    next_session_score = round(_clip(overnight_score * 0.6 + intraday_score * 0.4), 2)

    intraday_edge = abs(intraday_score - 50.0)
    overnight_edge = abs(overnight_score - 50.0)
    if overnight_edge > intraday_edge + 3:
        stronger_window = "隔夜消息更强"
        summary = "隔夜/盘后消息偏向更明显，明日开盘更值得先看竞价和前 15 分钟承接。"
    elif intraday_edge > overnight_edge + 3:
        stronger_window = "盘中消息更强"
        summary = "盘中消息驱动更强，说明当前热度更多来自当日资金反馈，明日要防止隔夜预期回落。"
    else:
        stronger_window = "时段差异不大"
        summary = "消息情绪没有形成明显的盘中或隔夜优势，执行上更应依赖价格确认。"

    return TemporalNewsPulse(
        intraday_score=round(intraday_score, 2),
        overnight_score=round(overnight_score, 2),
        next_session_score=next_session_score,
        stronger_window=stronger_window,
        summary=summary,
    )


def evaluate_intraday_structure_signal(minute_df: pd.DataFrame) -> IntradayStructureSignal:
    if minute_df.empty or "datetime" not in minute_df.columns:
        return IntradayStructureSignal(
            opening_volume_ratio=0.0,
            first30_volume_share=0.0,
            early_return_pct=0.0,
            label="暂无分时结构",
            summary="当前没有分钟级数据，无法用开盘半小时结构辅助判断尾盘与次日延续性。",
        )

    minute = minute_df.copy()
    minute["datetime"] = pd.to_datetime(minute["datetime"], errors="coerce")
    minute = minute.dropna(subset=["datetime", "close", "volume"]).sort_values("datetime")
    if minute.empty:
        return IntradayStructureSignal(
            opening_volume_ratio=0.0,
            first30_volume_share=0.0,
            early_return_pct=0.0,
            label="暂无分时结构",
            summary="分钟数据清洗后为空，暂不做开盘结构判断。",
        )

    first_30_mask = minute["datetime"].dt.time < dt.time(10, 0)
    before_last_half_hour_mask = minute["datetime"].dt.time < dt.time(14, 30)
    first_30 = minute[first_30_mask]
    before_last_half_hour = minute[before_last_half_hour_mask]

    first_30_volume = _safe_float(first_30["volume"].sum(), 0.0)
    early_volume_base = _safe_float(before_last_half_hour["volume"].sum(), 0.0)
    total_volume = _safe_float(minute["volume"].sum(), 0.0)
    opening_volume_ratio = first_30_volume / early_volume_base if early_volume_base else 0.0
    first30_volume_share = first_30_volume / total_volume if total_volume else 0.0

    early_close = _safe_float(first_30["close"].iloc[-1] if not first_30.empty else minute["close"].iloc[min(29, len(minute) - 1)], 0.0)
    session_open = _safe_float(minute["open"].iloc[0] if "open" in minute.columns else minute["close"].iloc[0], 0.0)
    early_return_pct = early_close / session_open - 1 if session_open else 0.0

    if opening_volume_ratio >= 0.22 and early_return_pct > 0:
        label = "开盘先手较强"
        summary = "开盘半小时放量且价格先走强，说明早盘资金先手更积极，次日更适合看强度延续。"
    elif opening_volume_ratio <= 0.12:
        label = "开盘确认不足"
        summary = "开盘半小时在日内成交中的占比较低，说明真正方向可能要等上午后半段或午后再确认。"
    else:
        label = "开盘结构中性"
        summary = "开盘量能和价格都没有形成极端特征，执行上更适合等待均价线和关键价位二次确认。"

    return IntradayStructureSignal(
        opening_volume_ratio=round(opening_volume_ratio, 4),
        first30_volume_share=round(first30_volume_share, 4),
        early_return_pct=round(early_return_pct, 4),
        label=label,
        summary=summary,
    )


def build_strategy_workbench(
    stage_code: str,
    probability_up: float,
    quant_score: float,
    sector_score: float,
    temporal_pulse: TemporalNewsPulse,
    intraday_signal: IntradayStructureSignal,
    rule_context: TradingRuleContext,
) -> StrategyWorkbench:
    probability_pct = probability_up * 100 if probability_up <= 1 else probability_up
    strategy_score = (
        probability_pct * 0.34
        + float(quant_score) * 0.24
        + float(sector_score) * 0.16
        + temporal_pulse.next_session_score * 0.16
        + _clip(intraday_signal.first30_volume_share * 300, 0, 100) * 0.10
    )

    if rule_context.price_limit_pct <= 5:
        strategy_score -= 6
    elif rule_context.price_limit_pct >= 20:
        strategy_score += 2
    strategy_score = round(_clip(strategy_score), 2)

    if stage_code in {"trend_acceleration", "breakout_confirmation"} and probability_pct >= 60 and quant_score >= 65:
        style = "顺势确认"
        entry_window = "优先看 9:35-10:15 的承接，确认均价线与关键价位同步站稳后再跟。"
        exit_rule = "若冲高后量价背离、或回落失守均价线且无法修复，先减仓。"
        summary = "适合做强势确认，不适合抢竞价和无条件追高。"
    elif stage_code in {"pullback_retest", "second_attack_attempt"}:
        style = "回踩再上"
        entry_window = "先等第一次回踩结束，再看 10:00 后是否重新站回均价线。"
        exit_rule = "支撑位被跌破且午前收不回，优先撤退等待下一次结构。"
        summary = "适合做支撑确认和二次转强，不适合在高位横盘中间反复追。"
    elif stage_code in {"distribution_risk", "weak_repair"} or probability_pct < 45:
        style = "防守优先"
        entry_window = "少做预判，尽量等午后或次日再确认，不在开盘情绪波动段重仓出手。"
        exit_rule = "一旦出现放量回落和均价线失守，先保住仓位安全。"
        summary = "当前更像风险管理场景，重点是先卖好再考虑买点。"
    else:
        style = "边界等待"
        entry_window = "等箱体边界或关键均线被明确突破后再动手。"
        exit_rule = "没有方向确认前，冲高不追，破位先退。"
        summary = "当前边际优势不够，等待确认比预判更重要。"

    drivers = (
        f"交易约束：{rule_context.board_label} / {rule_context.price_limit_label}",
        f"隔夜情绪：{temporal_pulse.overnight_score:.1f}",
        f"盘中情绪：{temporal_pulse.intraday_score:.1f}",
        f"开盘量能占比：{_pct(intraday_signal.first30_volume_share)}",
    )
    return StrategyWorkbench(
        strategy_score=strategy_score,
        style=style,
        entry_window=entry_window,
        exit_rule=exit_rule,
        summary=summary,
        drivers=drivers,
    )


def assess_launch_window(
    *,
    stage_code: str,
    stage_label: str,
    probability_up: float,
    predicted_upside_pct: float,
    quant_score: float,
    sector_score: float = 50.0,
    fund_score: float = 50.0,
    news_score: float = 50.0,
    launch_score: float = 50.0,
    launch_readiness_score: float = 50.0,
    market_resonance_score: float = 50.0,
    launch_specialist_score: float = 50.0,
    launch_regime_fit_score: float = 50.0,
    launch_specialist_confidence: float = 50.0,
    close_vs_ma20: float = 0.0,
    breakout_distance: float = 0.0,
    intraday_bias: int = 0,
) -> LaunchWindowAssessment:
    probability_pct = probability_up * 100 if probability_up <= 1 else probability_up
    predicted_upside_pct = max(float(predicted_upside_pct or 0.0), 0.0)
    stage_text = f"{stage_code} {stage_label}".lower()

    stage_bonus = 0.0
    if any(keyword in stage_text for keyword in ("trend_acceleration", "breakout_confirmation", "主升", "突破", "加速")):
        stage_bonus += 8.0
    elif any(keyword in stage_text for keyword in ("pullback_retest", "second_attack_attempt", "回踩", "二次", "回调")):
        stage_bonus += 5.5
    elif any(keyword in stage_text for keyword in ("distribution_risk", "weak_repair", "派发", "弱修复", "高位")):
        stage_bonus -= 11.0

    structure_bonus = 0.0
    if 0.0 <= close_vs_ma20 <= 0.10:
        structure_bonus += 4.5
    elif close_vs_ma20 > 0.10:
        structure_bonus += 1.0
    else:
        structure_bonus -= min(abs(close_vs_ma20) * 120.0, 8.0)

    if -0.015 <= breakout_distance <= 0.03:
        structure_bonus += 6.0
    elif breakout_distance < -0.05:
        structure_bonus -= 6.0
    elif breakout_distance > 0.06:
        structure_bonus -= 4.0
    elif breakout_distance > 0.03:
        structure_bonus += 1.5

    launch_edge = (
        (float(launch_score) - 50.0) * 0.10
        + (float(launch_readiness_score) - 50.0) * 0.28
        + (float(market_resonance_score) - 50.0) * 0.18
        + (float(launch_specialist_score) - 50.0) * 0.26
        + (float(launch_regime_fit_score) - 50.0) * 0.16
        + (float(launch_specialist_confidence) - 50.0) * 0.08
    )
    execution_edge = (
        (float(quant_score) - 50.0) * 0.12
        + (float(sector_score) - 50.0) * 0.07
        + (float(fund_score) - 50.0) * 0.07
        + (float(news_score) - 50.0) * 0.04
        + float(intraday_bias) * 3.2
    )
    probability_edge = (probability_pct - 55.0) * 0.26 + min(predicted_upside_pct, 18.0) * 0.55

    window_score = _clip(50.0 + stage_bonus + structure_bonus + launch_edge + execution_edge + probability_edge)
    window_confidence = _clip(
        42.0
        + abs(window_score - 50.0) * 0.72
        + abs(float(intraday_bias)) * 3.5
        + max(float(launch_specialist_confidence) - 50.0, 0.0) * 0.16,
        35.0,
        96.0,
    )

    if window_score >= 78 and launch_readiness_score >= 62 and launch_specialist_confidence >= 58:
        status = "黄金启动窗"
        label = "主升初启"
        summary = "结构、共振和执行信号已经靠拢，更像主升刚启动或刚确认。"
    elif window_score >= 66 and launch_readiness_score >= 56:
        status = "启动观察窗"
        label = "启动确认"
        summary = "已经进入启动跟踪区，下一步更看分时承接和突破确认。"
    elif window_score >= 58:
        status = "强势延续"
        label = "趋势延续"
        summary = "更像强势延续而不是最早启动点，适合跟踪强弱切换。"
    elif window_score <= 40:
        status = "高位风险窗"
        label = "风险防守"
        summary = "当前位置更偏高位风险或结构失配，优先防守而不是抢启动。"
    else:
        status = "非启动窗"
        label = "等待确认"
        summary = "尚未形成清晰的启动窗口，需要继续等位置、量能或共振确认。"

    drivers = (
        f"主升准备度 {float(launch_readiness_score):.1f}",
        f"市场共振 {float(market_resonance_score):.1f}",
        f"专项模型 {float(launch_specialist_score):.1f} / 适配 {float(launch_regime_fit_score):.1f}",
        f"位置结构 close_vs_ma20={float(close_vs_ma20):+.2%} / breakout={float(breakout_distance):+.2%}",
    )
    return LaunchWindowAssessment(
        window_score=round(window_score, 2),
        window_confidence=round(window_confidence, 2),
        label=label,
        status=status,
        summary=summary,
        drivers=drivers,
    )


def assess_execution_readiness(
    *,
    stage_code: str,
    stage_label: str,
    probability_up: float,
    predicted_upside_pct: float,
    quant_score: float,
    launch_window_score: float,
    launch_window_status: str,
    launch_window_confidence: float = 50.0,
    sector_score: float = 50.0,
    fund_score: float = 50.0,
    news_score: float = 50.0,
    close_vs_ma20: float = 0.0,
    breakout_distance: float = 0.0,
    intraday_bias: int = 0,
) -> ExecutionAssessment:
    probability_pct = probability_up * 100 if probability_up <= 1 else probability_up
    upside_pct = max(float(predicted_upside_pct or 0.0), 0.0)
    stage_text = f"{stage_code} {stage_label}".lower()

    structure_edge = 0.0
    if 0.0 <= close_vs_ma20 <= 0.08:
        structure_edge += 8.0
    elif close_vs_ma20 > 0.08:
        structure_edge += 2.0
    else:
        structure_edge -= min(abs(close_vs_ma20) * 140.0, 12.0)

    if -0.015 <= breakout_distance <= 0.02:
        structure_edge += 9.0
    elif 0.02 < breakout_distance <= 0.06:
        structure_edge += 2.0
    elif breakout_distance > 0.06:
        structure_edge -= min((breakout_distance - 0.06) * 220.0 + 5.0, 15.0)
    elif breakout_distance < -0.03:
        structure_edge -= min(abs(breakout_distance) * 180.0, 12.0)

    stage_edge = 0.0
    if any(keyword in stage_text for keyword in ("breakout_confirmation", "trend_acceleration", "主升", "突破", "加速")):
        stage_edge += 5.5
    elif any(keyword in stage_text for keyword in ("pullback_retest", "second_attack_attempt", "回踩", "二次")):
        stage_edge += 3.0
    elif any(keyword in stage_text for keyword in ("distribution_risk", "weak_repair", "高位", "派发")):
        stage_edge -= 10.0

    resonance_edge = (
        (float(launch_window_score) - 50.0) * 0.32
        + (float(launch_window_confidence) - 50.0) * 0.08
        + (float(sector_score) - 50.0) * 0.10
        + (float(fund_score) - 50.0) * 0.10
        + (float(news_score) - 50.0) * 0.05
    )
    execution_edge = (
        (float(quant_score) - 50.0) * 0.18
        + (probability_pct - 55.0) * 0.18
        + min(upside_pct, 18.0) * 0.70
        + float(intraday_bias) * 5.2
    )

    execution_score = _clip(50.0 + stage_edge + structure_edge + resonance_edge + execution_edge)
    execution_confidence = _clip(
        40.0
        + abs(execution_score - 50.0) * 0.72
        + abs(float(intraday_bias)) * 4.0
        + max(float(launch_window_confidence) - 50.0, 0.0) * 0.10,
        35.0,
        96.0,
    )

    drawdown_risk_pct = round(
        max(
            1.8,
            min(
                12.0,
                2.4
                + max(float(close_vs_ma20) - 0.05, 0.0) * 28.0
                + max(float(breakout_distance), 0.0) * 42.0
                + max(50.0 - float(fund_score), 0.0) * 0.04
                + max(-float(intraday_bias), 0.0) * 0.8
                + (6.0 if "高位风险" in str(launch_window_status) else 0.0),
            ),
        ),
        2,
    )
    expected_return_pct = round(
        max(
            1.2,
            min(
                max(upside_pct * 1.12, 3.0),
                upside_pct * (0.62 + execution_score / 170.0)
                + max(probability_pct - 50.0, 0.0) * 0.05
                + max(float(quant_score) - 60.0, 0.0) * 0.03,
            ),
        ),
        2,
    )
    reward_risk_ratio = round(expected_return_pct / max(drawdown_risk_pct, 0.8), 2)

    if reward_risk_ratio >= 2.4:
        reward_risk_label = "盈亏比优"
    elif reward_risk_ratio >= 1.6:
        reward_risk_label = "盈亏比可接受"
    else:
        reward_risk_label = "盈亏比一般"

    if breakout_distance > 0.06 or close_vs_ma20 > 0.10:
        chase_risk_label = "追价风险高"
    elif breakout_distance > 0.02 or close_vs_ma20 > 0.05:
        chase_risk_label = "追价风险中等"
    else:
        chase_risk_label = "追价风险可控"

    if launch_window_status == "高位风险窗" or execution_score <= 42:
        label = "暂不执行"
        window = "防守等待型"
        summary = "选股逻辑可能仍有看点，但当前位置与执行结构不适合直接出手。"
        entry_zone = "优先等回踩重新建立支撑，或等下一次结构重组后再评估。"
        invalidation_rule = "若继续放量走弱、或收盘重新跌回关键均线下方，本轮执行逻辑直接失效。"
    elif execution_score >= 78 and intraday_bias >= 1 and -0.015 <= breakout_distance <= 0.03:
        label = "可执行"
        if breakout_distance <= 0.005:
            window = "平台确认型"
            entry_zone = "优先等平台位或前高附近回踩不破，再配合均价线重新抬头执行。"
        else:
            window = "突破确认型"
            entry_zone = "优先等突破后第一次回踩承接确认，不脱离平台位追价。"
        summary = "这类信号更接近实战可执行区，关键是按确认点而不是按情绪追价。"
        invalidation_rule = "若回踩平台位后不能收回，或收盘跌回 MA20 下方，本轮执行逻辑失效。"
    elif execution_score >= 64:
        label = "临门观察"
        if breakout_distance < -0.015:
            window = "回踩等待型"
            entry_zone = "先等重新回到平台突破位附近，再看分时承接和均价线修复。"
        else:
            window = "承接确认型"
            entry_zone = "等分时承接更明确后再执行，尽量避免在中段拉升时追进去。"
        summary = "值得跟踪，但更适合等确认，而不是立刻执行。"
        invalidation_rule = "若分时承接持续转弱，或午前仍无法回到关键位上方，先放弃本轮尝试。"
    else:
        label = "等待结构"
        window = "信号未合流"
        entry_zone = "先等位置、量能和分时承接至少有两项同步改善后再考虑。"
        summary = "当前更像研究候选，不像成熟执行点。"
        invalidation_rule = "若后续继续远离平台位或跌破关键支撑，本轮观察价值下降。"

    drivers = (
        f"启动窗口 {float(launch_window_score):.1f} / 置信度 {float(launch_window_confidence):.1f}",
        f"位置结构 close_vs_ma20={float(close_vs_ma20):+.2%} / breakout={float(breakout_distance):+.2%}",
        f"板块/资金/消息 {float(sector_score):.1f}/{float(fund_score):.1f}/{float(news_score):.1f}",
        f"量化 {float(quant_score):.1f} / 分时偏向 {int(intraday_bias):+d}",
    )
    return ExecutionAssessment(
        execution_score=round(execution_score, 2),
        execution_confidence=round(execution_confidence, 2),
        label=label,
        window=window,
        summary=summary,
        entry_zone=entry_zone,
        invalidation_rule=invalidation_rule,
        reward_risk_label=reward_risk_label,
        expected_return_pct=expected_return_pct,
        drawdown_risk_pct=drawdown_risk_pct,
        reward_risk_ratio=reward_risk_ratio,
        chase_risk_label=chase_risk_label,
        drivers=drivers,
    )
