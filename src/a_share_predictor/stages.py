from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from .features import build_daily_features


def _clip(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return float(max(lower, min(value, upper)))


def _feature_dict(values: Mapping[str, float] | pd.Series | None) -> dict[str, float]:
    if values is None:
        return {}
    items = values.items() if not isinstance(values, pd.Series) else values.items()
    numeric: dict[str, float] = {}
    for key, val in items:
        if val is None or pd.isna(val):
            continue
        try:
            numeric[str(key)] = float(val)
        except (TypeError, ValueError):
            continue
    return numeric


def _pct(value: float | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value) * 100:.{digits}f}%"


def _num(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def _ma_level(close_price: float, close_vs_ma: float | None) -> float | None:
    if close_vs_ma is None or pd.isna(close_vs_ma) or close_vs_ma <= -0.95:
        return None
    return float(close_price / (1 + float(close_vs_ma)))


def _level_from_distance(close_price: float | None, distance: float | None) -> float | None:
    if close_price is None or pd.isna(close_price):
        return None
    if distance is None or pd.isna(distance) or distance <= -0.95:
        return None
    return float(close_price / (1 + float(distance)))


def _range_lower_level(
    close_price: float | None,
    range_position: float | None,
    upper_level: float | None,
) -> float | None:
    if close_price is None or upper_level is None:
        return None
    if range_position is None or pd.isna(range_position) or not 0 < float(range_position) < 0.98:
        return None
    position = float(range_position)
    denominator = 1 - position
    if denominator <= 0:
        return None
    return float((close_price - position * upper_level) / denominator)


@dataclass(slots=True)
class StageAssessment:
    code: str
    label: str
    description: str
    structure_summary: str
    intraday_expectation: str
    priority: str
    focus_points: tuple[str, ...]
    invalidation: str
    rationale: tuple[str, ...]
    risk_flags: tuple[str, ...] = ()


@dataclass(slots=True)
class TomorrowPlan:
    setup_label: str
    bias: str
    summary: str
    buy_point: str
    sell_point: str
    avoid_point: str
    confidence: float


def _intraday_payload(values) -> dict[str, object]:
    if values is None:
        return {}
    if isinstance(values, Mapping):
        return dict(values)
    payload: dict[str, object] = {}
    for field in [
        "label",
        "summary",
        "score",
        "above_avg_ratio",
        "max_pullback",
        "opening_volume_ratio",
        "first30_volume_share",
        "early_return_pct",
    ]:
        if hasattr(values, field):
            payload[field] = getattr(values, field)
    return payload


def _intraday_safe_float(payload: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = payload.get(key, default)
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _merge_plan_text(base_text: str, extra_text: str) -> str:
    extra = extra_text.strip()
    if not extra:
        return base_text
    return f"{base_text} 分时条件：{extra}"


def _intraday_entry_clause(
    intraday_state: Mapping[str, object],
    intraday_signal: Mapping[str, object],
) -> str:
    if not intraday_state and not intraday_signal:
        return ""
    label_text = f'{intraday_state.get("label", "")} {intraday_signal.get("label", "")}'
    if "暂无" in label_text or "待开盘" in label_text:
        return "开盘后先看均价线方向和第一次回踩，不抢竞价。"
    score = _intraday_safe_float(intraday_state, "score", 0.5)
    above_avg_ratio = _intraday_safe_float(intraday_state, "above_avg_ratio", 0.5)
    max_pullback = _intraday_safe_float(intraday_state, "max_pullback", 0.0)
    first30_share = _intraday_safe_float(intraday_signal, "first30_volume_share", 0.0)
    early_return = _intraday_safe_float(intraday_signal, "early_return_pct", 0.0)

    if score >= 0.7 and above_avg_ratio >= 0.6:
        return "优先等分时始终运行在均价线上方，第一次回踩均价线不破再执行。"
    if score >= 0.58 and early_return >= 0:
        return "至少等 9:45 之后重新站回均价线，并确认回踩不再跌破开盘价。"
    if first30_share >= 0.34 and early_return > 0:
        return "开盘量能已经先透支，尽量等第一次冲高回落后再次收回均价线再考虑。"
    if max_pullback >= 0.035:
        return "即使触及静态买点，也要先等分时止跌并重新翻上均价线，避免抄在下坠途中。"
    return "没有分时转强前先观察，至少看到均价线拐头向上再动手。"


def _intraday_exit_clause(
    intraday_state: Mapping[str, object],
    intraday_signal: Mapping[str, object],
) -> str:
    if not intraday_state and not intraday_signal:
        return ""
    label_text = f'{intraday_state.get("label", "")} {intraday_signal.get("label", "")}'
    if "暂无" in label_text or "待开盘" in label_text:
        return "开盘后若迟迟收不回均价线，卖点执行优先于主观等待。"
    score = _intraday_safe_float(intraday_state, "score", 0.5)
    max_pullback = _intraday_safe_float(intraday_state, "max_pullback", 0.0)
    early_return = _intraday_safe_float(intraday_signal, "early_return_pct", 0.0)
    first30_share = _intraday_safe_float(intraday_signal, "first30_volume_share", 0.0)

    if score <= 0.48 or max_pullback >= 0.045:
        return "一旦分时均价线失守且 5到10 分钟收不回，或日内回撤继续扩大，就优先执行卖点。"
    if early_return > 0 and first30_share >= 0.32:
        return "若早盘冲高后回落并跌破均价线，说明先手资金开始兑现，卖点要前置。"
    return "若分时跌破均价线后反抽无力，卖点执行优先于主观等待。"


def _common_risk_flags(latest: dict[str, float]) -> tuple[str, ...]:
    flags: list[str] = []
    if latest.get("upper_shadow_ratio", 0.0) >= 0.022:
        flags.append("上影线偏长，说明高位抛压开始出现。")
    if latest.get("volatility_10", 0.0) >= 0.038:
        flags.append("短线波动率升高，明日更容易先拉后砸。")
    if latest.get("volume_ratio_5", 1.0) >= 1.8:
        flags.append("量能突然放大较多，需警惕情绪过热后的分歧。")
    return tuple(flags)


def main_rise_start_score(latest_features: Mapping[str, float] | pd.Series | None) -> float:
    latest = _feature_dict(latest_features)
    close_vs_ma20 = latest.get("close_vs_ma20", 0.0)
    ret_20 = latest.get("ret_20", 0.0)
    ret_60 = latest.get("ret_60", 0.0)
    breakout_distance_20 = latest.get("breakout_distance_20", -0.05)
    range_position_20 = latest.get("range_position_20", 0.5)
    consolidation_width_20 = latest.get("consolidation_width_20", 0.35)
    ma20_slope_5 = latest.get("ma20_slope_5", 0.0)
    volume_ratio_5 = latest.get("volume_ratio_5", 1.0)
    volatility_contraction = latest.get("volatility_contraction", 0.0)
    upper_shadow_ratio = latest.get("upper_shadow_ratio", 0.0)
    downside_vol_ratio_20 = latest.get("downside_vol_ratio_20", 0.4)
    drawdown_20 = latest.get("drawdown_20", -0.06)
    close_near_high_5 = latest.get("close_near_high_5", -0.03)

    score = 46.0
    score += _clip((0.06 - abs(close_vs_ma20 - 0.022)) * 210, -8, 14)
    score += _clip((0.16 - abs(ret_20 - 0.08)) * 70, -6, 10)
    score += _clip((0.22 - abs(ret_60 - 0.14)) * 38, -5, 8)
    score += _clip((0.05 - abs(breakout_distance_20 - 0.008)) * 220, -8, 12)
    score += _clip((0.22 - abs(range_position_20 - 0.68)) * 42, -6, 8)
    score += _clip((0.24 - consolidation_width_20) * 36, -4, 8)
    score += _clip(ma20_slope_5 * 1500, -6, 10)
    score += _clip((0.14 - abs(volume_ratio_5 - 1.08)) * 30, -4, 6)
    score += _clip((-volatility_contraction) * 14, -3, 6)
    score += _clip((0.04 - abs(close_near_high_5)) * 85, -3, 6)
    score -= _clip(upper_shadow_ratio * 420, 0, 10)
    score -= _clip(max(downside_vol_ratio_20 - 0.55, 0.0) * 24, 0, 8)
    score -= _clip(abs(min(drawdown_20, 0.0)) * 60, 0, 6)
    return round(_clip(score), 2)


def _range_reference_text(daily: pd.DataFrame) -> tuple[str, str]:
    recent = daily.tail(20)
    if recent.empty:
        return "近20日高点", "近20日低点"
    high_text = f"近20日高点 {float(recent['high'].max()):.2f}"
    low_text = f"近20日低点 {float(recent['low'].min()):.2f}"
    return high_text, low_text


def _build_trend_acceleration(daily: pd.DataFrame, latest: dict[str, float]) -> StageAssessment:
    close_price = float(daily["close"].iloc[-1])
    ma20_level = _ma_level(close_price, latest.get("close_vs_ma20"))
    high_text, _ = _range_reference_text(daily)
    return StageAssessment(
        code="trend_acceleration",
        label="趋势主升加速",
        description="价格持续运行在中期均线之上，20日线抬升明显，量能没有明显掉队，属于强趋势中的主升阶段。",
        structure_summary="这类结构的关键不是追逐单根大阳线，而是确认趋势斜率、均线支撑和量价延续是否仍在。",
        intraday_expectation="更适合看强更强和回踩承接，分时若长期站在均价线上方，往往说明主升结构还没被破坏。",
        priority="高优先级",
        focus_points=(
            "早盘若高开，先看开盘价和分时均价线能否同步站稳，再决定是否跟随。",
            "盘中回踩时重点看 MA20 附近和昨日强势K线中枢是否有缩量承接。",
            "若冲高后量能明显放大但价格不再扩展，优先防范主升中的第一次大分歧。",
        ),
        invalidation=(
            f"若后续收盘有效跌回 MA20 附近{f'({ma20_level:.2f})' if ma20_level else ''}下方，"
            "并伴随放量回落，主升加速阶段就要降级看待。"
        ),
        rationale=(
            f"60日涨幅 {_pct(latest.get('ret_60'))}，说明中周期趋势已经打开。",
            f"价格高于 MA20 {_pct(latest.get('close_vs_ma20'))}，20日线5日斜率 {_pct(latest.get('ma20_slope_5'))}。",
            f"20日区间位置 {_pct(latest.get('range_position_20'), digits=0)}，接近上沿运行，强势特征明显。",
            f"5日量比 {_num(latest.get('volume_ratio_5'))}，说明量能对趋势仍有支持。参考 {high_text}。",
        ),
        risk_flags=_common_risk_flags(latest),
    )


def _build_breakout_confirmation(daily: pd.DataFrame, latest: dict[str, float]) -> StageAssessment:
    high_text, _ = _range_reference_text(daily)
    return StageAssessment(
        code="breakout_confirmation",
        label="平台突破确认",
        description="股价刚刚摆脱横盘或收敛平台，正在做突破后的确认，最重要的是确认突破位不能快速丢失。",
        structure_summary="这是从蓄势转向趋势的过渡段，真正的买点通常不是平台内，而是突破后回踩不破的再确认。",
        intraday_expectation="更适合观察前平台上沿和分时均价线的承接，盘中一旦跌回平台内部，强度就会明显打折。",
        priority="高优先级",
        focus_points=(
            "关注前平台上沿是否从压力切换为支撑。",
            "突破后的第一次回踩若缩量且分时重新站回均价线，质量通常更高。",
            "放量冲高但量价背离时，不急于追第二次脉冲。",
        ),
        invalidation=f"若价格再次跌回前平台内部，且无法重新站回 {high_text} 附近，突破确认就失败了。",
        rationale=(
            f"20日突破距离 {_pct(latest.get('breakout_distance_20'))}，说明价格就在突破位附近做确认。",
            f"20日涨幅 {_pct(latest.get('ret_20'))}，结构已从震荡切向上攻。",
            f"20日整理宽度 {_pct(latest.get('consolidation_width_20'))}，平台边界相对清晰。",
            f"5日量比 {_num(latest.get('volume_ratio_5'))}，突破需要量能支持而不是单纯脉冲。",
        ),
        risk_flags=_common_risk_flags(latest),
    )


def _build_main_rise_start(daily: pd.DataFrame, latest: dict[str, float]) -> StageAssessment:
    close_price = float(daily["close"].iloc[-1])
    ma20_level = _ma_level(close_price, latest.get("close_vs_ma20"))
    high_text, _ = _range_reference_text(daily)
    return StageAssessment(
        code="main_rise_start",
        label="主升初启确认",
        description="价格刚从整理区向上脱离，20日线开始拐头，涨幅尚未过度扩张，更接近主升启动或刚启动的右侧临界点。",
        structure_summary="这类结构最重要的不是已经涨了多少，而是趋势斜率刚转强、突破位附近承接是否稳定，以及量价是否处在主升前的健康扩张区。",
        intraday_expectation="更适合观察分时均价线之上的首次回踩承接，若早盘放量站稳突破位，往往意味着主升启动的执行质量较高。",
        priority="高优先级",
        focus_points=(
            "重点看前平台上沿与 MA20 是否形成双支撑，而不是单纯追已经拉开的阳线。",
            "理想状态是放量突破后缩量回踩不破，再次站回均价线确认启动。",
            "如果刚突破就反复跌回平台内，说明主升启动还没有完成确认。",
        ),
        invalidation=(
            f"若后续收盘重新跌回前平台 {high_text} 下方，或跌破 MA20 附近{f'({ma20_level:.2f})' if ma20_level else ''}且承接不足，"
            "则主升初启判断需要降级。"
        ),
        rationale=(
            f"20日涨幅 {_pct(latest.get('ret_20'))}，说明刚从整理转入上行，而不是已经过度扩张。",
            f"价格相对 MA20 {_pct(latest.get('close_vs_ma20'))}，20日线5日斜率 {_pct(latest.get('ma20_slope_5'))}，斜率开始转强。",
            f"20日突破距离 {_pct(latest.get('breakout_distance_20'))}，更接近主升右侧确认区而不是远离成本区。",
            f"20日整理宽度 {_pct(latest.get('consolidation_width_20'))}，5日量比 {_num(latest.get('volume_ratio_5'))}，用于确认启动前的收敛与放量节奏。",
            f"静态主升初启分数 {main_rise_start_score(latest):.1f}，可辅助区分‘刚启动’与‘已走远’。",
        ),
        risk_flags=_common_risk_flags(latest),
    )


def _build_pullback_retest(daily: pd.DataFrame, latest: dict[str, float]) -> StageAssessment:
    close_price = float(daily["close"].iloc[-1])
    ma20_level = _ma_level(close_price, latest.get("close_vs_ma20"))
    return StageAssessment(
        code="pullback_retest",
        label="强势回踩确认",
        description="前一段上攻已经完成，当前进入回踩和换手阶段，关键是确认支撑是否真实有效。",
        structure_summary="回踩不是弱，而是强势股从快节奏拉升切换到筹码再平衡；看的是缩量、下影和均线承接。",
        intraday_expectation="分时允许有下探，但理想状态是回踩后逐步收回均价线，尾盘承接增强。",
        priority="中高优先级",
        focus_points=(
            "重点盯 MA20 和前期突破位是否形成双重支撑。",
            "观察下探时是否缩量、下影是否明显，确认筹码是否愿意在支撑位承接。",
            "若回踩结束后重新放量站回均价线，通常是更稳妥的再介入点。",
        ),
        invalidation=(
            f"若后续跌破 MA20 附近{f'({ma20_level:.2f})' if ma20_level else ''}且收不回，"
            "说明这次回踩可能从确认演变为转弱。"
        ),
        rationale=(
            f"60日涨幅 {_pct(latest.get('ret_60'))}，说明大趋势仍在上行通道中。",
            f"价格相对 MA20 {_pct(latest.get('close_vs_ma20'))}，已回到支撑检验区而非完全脱离趋势。",
            f"回踩突破位幅度 {_pct(latest.get('pullback_to_breakout_20'))}，决定这次回踩是健康换手还是失守。",
            f"下影占比 {_pct(latest.get('lower_shadow_ratio'))}、5日量比 {_num(latest.get('volume_ratio_5'))}，可用于确认承接质量。",
        ),
        risk_flags=_common_risk_flags(latest),
    )


def _build_second_attack_attempt(daily: pd.DataFrame, latest: dict[str, float]) -> StageAssessment:
    high_text, _ = _range_reference_text(daily)
    return StageAssessment(
        code="second_attack_attempt",
        label="二波蓄势再攻",
        description="第一段上攻后并未直接走弱，而是在相对高位做缩量整理，属于等待第二次放量的预备态。",
        structure_summary="这类结构更像高位平台，而不是底部震荡。买点不在震荡中间，而在二次放量确认的时候。",
        intraday_expectation="盘中更适合等放量过前高，而不是在横向整理中间来回追单。",
        priority="中高优先级",
        focus_points=(
            "看高位整理是否保持在强势区间内，不能频繁跌回箱体中下沿。",
            "重点确认二次放量是否真的越过前高，而不是假突破。",
            "若盘中多次冲高不过前高，反而要防高位震荡延长。",
        ),
        invalidation=f"若价格持续回落并跌破高位整理下沿，且始终不能重新靠近 {high_text}，二波预期就要取消。",
        rationale=(
            f"60日涨幅 {_pct(latest.get('ret_60'))}，说明前一段主升已经打出辨识度。",
            f"60日区间位置 {_pct(latest.get('range_position_60'), digits=0)}，仍处于相对高位区域。",
            f"20日整理宽度 {_pct(latest.get('consolidation_width_20'))}，说明当前更多是高位蓄势而非彻底转空。",
            f"价格相对 MA20 {_pct(latest.get('close_vs_ma20'))}，决定二波蓄势是否还站在强势一侧。",
        ),
        risk_flags=_common_risk_flags(latest),
    )


def _build_distribution_risk(daily: pd.DataFrame, latest: dict[str, float]) -> StageAssessment:
    _, low_text = _range_reference_text(daily)
    return StageAssessment(
        code="distribution_risk",
        label="高位分歧派发",
        description="价格仍在高位，但上影、波动和放量同时出现，说明短线资金开始在高位博弈出货。",
        structure_summary="这不是简单的强势震荡，而是高位资金分歧在放大。核心任务从进攻切换为保利润和等结构重新稳定。",
        intraday_expectation="分时容易出现冲高回落和均价线反复失守，盘中越急拉越要注意兑现而不是追高。",
        priority="风险优先",
        focus_points=(
            "观察冲高后的回落速度，如果脱离均价线后回不去，说明抛压偏重。",
            "高位大成交若没有带来实体扩展，更多是换手而不是继续推升。",
            "先看能否缩量稳住，再决定后面还有没有二次攻击的可能。",
        ),
        invalidation=f"若后续连收缩量整理并重新站稳均价体系，同时不再跌破 {low_text}，分歧风险才会缓解。",
        rationale=(
            f"20日涨幅 {_pct(latest.get('ret_20'))}，处于高位后更容易出现兑现需求。",
            f"上影占比 {_pct(latest.get('upper_shadow_ratio'))}，说明盘中抛压开始主动释放。",
            f"5日量比 {_num(latest.get('volume_ratio_5'))}，高位放量但价格扩展不足时要先当作风险信号。",
            f"10日波动率 {_pct(latest.get('volatility_10'))}，波动放大往往意味着筹码稳定性下降。",
        ),
        risk_flags=_common_risk_flags(latest) + ("优先保利润，不把高位分歧误判成新的上攻起点。",),
    )


def _build_weak_repair(daily: pd.DataFrame, latest: dict[str, float]) -> StageAssessment:
    return StageAssessment(
        code="weak_repair",
        label="弱转强修复",
        description="此前处于弱势或超跌段，最近开始出现止跌和修复，但尚未真正回到主升结构。",
        structure_summary="这个阶段更像试错区，不是趋势确认区。参与重点是仓位控制和等右侧信号。",
        intraday_expectation="分时更看重是否能放量收复均价线和短均线，否则修复很容易演变成反抽。",
        priority="观察优先",
        focus_points=(
            "先看修复有没有量能和承接，没有量的阳线持续性通常偏弱。",
            "确认是否能收复 MA5 / MA10，而不是只看单日反弹幅度。",
            "修复阶段更适合小仓跟踪，不适合当作确定性主升处理。",
        ),
        invalidation="若再次跌破近期止跌低点，说明修复失败，仍按弱势股处理。",
        rationale=(
            f"20日涨幅 {_pct(latest.get('ret_20'))}，说明此前经历过明显回撤。",
            f"价格相对 MA20 {_pct(latest.get('close_vs_ma20'))}，当前仍处在修复而非主升位置。",
            f"20日区间位置 {_pct(latest.get('range_position_20'), digits=0)}，如果还在中下部，右侧确认仍不充分。",
            f"下影占比 {_pct(latest.get('lower_shadow_ratio'))} 与 5日量比 {_num(latest.get('volume_ratio_5'))} 可判断止跌质量。",
        ),
        risk_flags=_common_risk_flags(latest),
    )


def _build_range_monitor(daily: pd.DataFrame, latest: dict[str, float]) -> StageAssessment:
    high_text, low_text = _range_reference_text(daily)
    return StageAssessment(
        code="range_monitor",
        label="区间震荡观察",
        description="当前没有形成清晰主升、突破或回踩确认，仍处于震荡区间内，更多是等待方向选择。",
        structure_summary="这类结构最容易让人频繁交易。与其预测，不如先等箱体边界和量能把方向走出来。",
        intraday_expectation="分时常见来回拉扯，除非出现放量越过区间边界，否则更适合耐心观察。",
        priority="中性观察",
        focus_points=(
            "优先识别区间上沿和下沿，不在箱体中间频繁追涨杀跌。",
            "如果向上突破，必须配合量能和分时均价线共振。",
            "如果向下破位，要先做风险控制，再等待新的结构形成。",
        ),
        invalidation=f"放量突破 {high_text} 或跌破 {low_text} 后，当前震荡阶段判断就要切换。",
        rationale=(
            f"20日整理宽度 {_pct(latest.get('consolidation_width_20'))}，说明价格仍在区间内来回波动。",
            f"20日区间位置 {_pct(latest.get('range_position_20'), digits=0)}，没有持续贴着上沿运行。",
            f"20日突破距离 {_pct(latest.get('breakout_distance_20'))}，还没有明确脱离震荡区。",
            f"5日量比 {_num(latest.get('volume_ratio_5'))}，量能暂时不足以支撑方向选择。",
        ),
        risk_flags=_common_risk_flags(latest),
    )


def classify_stage(daily: pd.DataFrame) -> StageAssessment:
    features = build_daily_features(daily).dropna()
    if features.empty:
        return StageAssessment(
            code="range_monitor",
            label="区间震荡观察",
            description="历史数据不足，暂按中性震荡处理。",
            structure_summary="先补足日线样本，再判断趋势和阶段。",
            intraday_expectation="分时信号参考意义有限。",
            priority="观察优先",
            focus_points=("先观察，不急于给方向结论。",),
            invalidation="历史样本不足，等待更多数据。",
            rationale=("当前样本不足以做稳定的阶段识别。",),
        )

    latest = _feature_dict(features.iloc[-1])

    is_distribution_risk = (
        latest.get("ret_20", 0.0) > 0.10
        and latest.get("upper_shadow_ratio", 0.0) > 0.022
        and latest.get("volume_ratio_5", 0.0) > 1.12
        and latest.get("range_position_20", 0.0) > 0.75
    )
    is_trend_acceleration = (
        latest.get("ret_60", 0.0) > 0.22
        and latest.get("close_vs_ma20", 0.0) > 0.035
        and latest.get("close_vs_ma60", 0.0) > 0.10
        and latest.get("ma20_slope_5", 0.0) > 0.010
        and latest.get("range_position_20", 0.0) > 0.72
        and latest.get("volume_ratio_5", 0.0) > 0.95
    )
    launch_score = main_rise_start_score(latest)
    is_main_rise_start = (
        launch_score >= 72
        and 0.025 <= latest.get("ret_20", 0.0) <= 0.18
        and 0.06 <= latest.get("ret_60", 0.0) <= 0.26
        and 0.0 <= latest.get("close_vs_ma20", -1.0) <= 0.055
        and latest.get("ma20_slope_5", 0.0) >= 0.0015
        and -0.02 <= latest.get("breakout_distance_20", -1.0) <= 0.05
        and 0.54 <= latest.get("range_position_20", 0.0) <= 0.82
        and latest.get("consolidation_width_20", 1.0) <= 0.28
        and latest.get("upper_shadow_ratio", 0.0) <= 0.025
    )
    is_breakout_confirmation = (
        latest.get("ret_20", 0.0) > 0.06
        and -0.02 <= latest.get("breakout_distance_20", -1.0) <= 0.07
        and latest.get("range_position_20", 0.0) > 0.68
        and latest.get("consolidation_width_20", 1.0) < 0.30
        and latest.get("volume_ratio_5", 0.0) > 1.02
    )
    is_pullback_retest = (
        latest.get("ret_60", 0.0) > 0.12
        and latest.get("close_vs_ma60", 0.0) > 0.03
        and -0.04 <= latest.get("close_vs_ma20", -1.0) <= 0.035
        and latest.get("pullback_to_breakout_20", -1.0) > -0.06
        and latest.get("ma20_slope_5", 0.0) >= -0.002
    )
    is_second_attack_attempt = (
        latest.get("ret_60", 0.0) > 0.16
        and latest.get("range_position_60", 0.0) > 0.58
        and latest.get("consolidation_width_20", 1.0) < 0.24
        and 0.0 <= latest.get("close_vs_ma20", -1.0) <= 0.07
        and -0.03 <= latest.get("breakout_distance_60", -1.0) <= 0.05
        and latest.get("ma20_slope_5", 0.0) < 0.020
    )
    is_weak_repair = (
        latest.get("ret_20", 0.0) < -0.05
        and latest.get("close_vs_ma20", 0.0) < 0.0
        and latest.get("range_position_20", 0.0) > 0.40
        and latest.get("lower_shadow_ratio", 0.0) >= latest.get("upper_shadow_ratio", 0.0)
    )

    if is_distribution_risk:
        return _build_distribution_risk(daily, latest)
    if is_trend_acceleration:
        return _build_trend_acceleration(daily, latest)
    if is_main_rise_start:
        return _build_main_rise_start(daily, latest)
    if is_breakout_confirmation:
        return _build_breakout_confirmation(daily, latest)
    if is_pullback_retest:
        return _build_pullback_retest(daily, latest)
    if is_second_attack_attempt:
        return _build_second_attack_attempt(daily, latest)
    if is_weak_repair:
        return _build_weak_repair(daily, latest)
    return _build_range_monitor(daily, latest)


def stage_numeric_score(stage: StageAssessment, latest_features: Mapping[str, float] | pd.Series) -> float:
    latest = _feature_dict(latest_features)
    base = {
        "trend_acceleration": 84.0,
        "main_rise_start": 80.0,
        "breakout_confirmation": 78.0,
        "pullback_retest": 70.0,
        "second_attack_attempt": 72.0,
        "range_monitor": 55.0,
        "weak_repair": 50.0,
        "distribution_risk": 38.0,
    }.get(stage.code, 55.0)

    score = base
    score += _clip(latest.get("close_vs_ma20", 0.0) * 180, -10, 12)
    score += _clip(latest.get("ma20_slope_5", 0.0) * 900, -8, 10)
    score += _clip((latest.get("volume_ratio_5", 1.0) - 1.0) * 14, -6, 7)
    score += _clip((latest.get("range_position_20", 0.5) - 0.5) * 16, -8, 8)
    score += _clip((main_rise_start_score(latest) - 50.0) * 0.16, -5, 8)
    score -= _clip(latest.get("upper_shadow_ratio", 0.0) * 520, 0, 12)
    score -= _clip(latest.get("volatility_10", 0.0) * 520, 0, 8)
    if stage.code == "distribution_risk":
        score -= 8
    if stage.code == "weak_repair":
        score -= 4
    if stage.code == "main_rise_start":
        score += 3
    return round(_clip(score), 2)


def build_tomorrow_plan(
    stage: StageAssessment,
    snapshot: Mapping[str, float] | pd.Series | None,
    latest_features: Mapping[str, float] | pd.Series | None,
    probability_up: float,
    quant_score: float,
    intraday_state: Mapping[str, object] | object | None = None,
    intraday_signal: Mapping[str, object] | object | None = None,
) -> TomorrowPlan:
    snap = _feature_dict(snapshot)
    latest = _feature_dict(latest_features)
    latest = {**snap, **latest}
    probability = float(probability_up)
    if probability > 1:
        probability /= 100.0

    stage_bonus = {
        "trend_acceleration": 12.0,
        "main_rise_start": 9.0,
        "breakout_confirmation": 8.0,
        "pullback_retest": 6.0,
        "second_attack_attempt": 5.0,
        "range_monitor": 0.0,
        "weak_repair": -4.0,
        "distribution_risk": -12.0,
    }.get(stage.code, 0.0)

    intraday_state_payload = _intraday_payload(intraday_state)
    intraday_signal_payload = _intraday_payload(intraday_signal)
    intraday_entry = _intraday_entry_clause(intraday_state_payload, intraday_signal_payload)
    intraday_exit = _intraday_exit_clause(intraday_state_payload, intraday_signal_payload)
    intraday_score = _intraday_safe_float(intraday_state_payload, "score", 0.5)
    intraday_drawdown = _intraday_safe_float(intraday_state_payload, "max_pullback", 0.0)
    early_return = _intraday_safe_float(intraday_signal_payload, "early_return_pct", 0.0)

    confidence = probability * 100 * 0.56 + float(quant_score) * 0.30 + 20 + stage_bonus
    confidence += (intraday_score - 0.5) * 12
    confidence += max(min(early_return * 140, 4.0), -4.0)
    confidence -= max(min(intraday_drawdown * 120, 6.0), 0.0)
    confidence -= _clip(latest.get("upper_shadow_ratio", 0.0) * 500, 0, 10)
    confidence -= _clip(max(latest.get("volatility_10", 0.0) - 0.03, 0.0) * 600, 0, 8)
    confidence = round(_clip(confidence), 2)

    close_price = latest.get("close")
    ma20_level = _ma_level(close_price, latest.get("close_vs_ma20"))
    ma60_level = _ma_level(close_price, latest.get("close_vs_ma60"))
    breakout20_level = _level_from_distance(close_price, latest.get("breakout_distance_20"))
    breakout60_level = _level_from_distance(close_price, latest.get("breakout_distance_60"))
    range_upper = breakout20_level or breakout60_level
    range_lower = _range_lower_level(close_price, latest.get("range_position_20"), range_upper)

    close_text = _num(close_price) if close_price is not None else "当前价"
    ma20_text = _num(ma20_level) if ma20_level is not None else "MA20"
    ma60_text = _num(ma60_level) if ma60_level is not None else "MA60"
    breakout20_text = _num(breakout20_level) if breakout20_level is not None else "前平台上沿"
    breakout60_text = _num(breakout60_level) if breakout60_level is not None else "前高"
    range_upper_text = _num(range_upper) if range_upper is not None else "箱体上沿"
    range_lower_text = _num(range_lower) if range_lower is not None else "箱体下沿"

    if stage.code == "trend_acceleration":
        return TomorrowPlan(
            setup_label="强更强跟随",
            bias="偏多进攻",
            summary="优先看高开承接和分时均价线之上的二次发力，核心是顺趋势而不是逆势抄回落。",
            buy_point=_merge_plan_text(
                f"优先盯昨收 {close_text} 与 MA20 {ma20_text} 上方的承接，高开或平开后 5到15 分钟不破开盘价并重新贴着均价线上行，再考虑分批跟随。",
                intraday_entry,
            ),
            sell_point=_merge_plan_text(
                f"若冲高放量却始终站不稳 {close_text} 上方，或跌破 MA20 {ma20_text} 后迟迟收不回，先减仓锁定利润。",
                intraday_exit,
            ),
            avoid_point="若一字式高开过大或开盘后快速跌回均价线下，避免情绪化追高。",
            confidence=confidence,
        )
    if stage.code == "main_rise_start":
        return TomorrowPlan(
            setup_label="主升初启右侧",
            bias="偏多确认",
            summary="更适合围绕平台突破位和 MA20 做右侧确认，核心不是追已经拉开的价格，而是等主升真正启动时的承接信号。",
            buy_point=_merge_plan_text(
                f"优先看前平台/突破位 {breakout20_text} 与 MA20 {ma20_text} 的共振支撑，若早盘放量站稳后第一次回踩不破并重新收回均价线，再分批跟随。",
                intraday_entry,
            ),
            sell_point=_merge_plan_text(
                f"若冲高后重新跌回突破位 {breakout20_text} 下方，或 MA20 {ma20_text} 失守后迟迟收不回，优先减仓，避免把‘启动失败’误判成正常洗盘。",
                intraday_exit,
            ),
            avoid_point="不在平台中部追单，也不在放量冲高后远离突破位的位置盲目追加。",
            confidence=confidence,
        )
    if stage.code == "breakout_confirmation":
        return TomorrowPlan(
            setup_label="突破回踩接力",
            bias="偏多确认",
            summary="真正的机会往往在突破后的第一次回踩确认，不在已经脱离成本区后的盲目追涨。",
            buy_point=_merge_plan_text(
                f"优先看前平台上沿 {breakout20_text} 附近的回踩确认，早盘不破该位且重新站回均价线时再接，追单尽量晚于第一次回踩。",
                intraday_entry,
            ),
            sell_point=_merge_plan_text(
                f"若跌回平台内部并失守 {breakout20_text}，或冲高后不能稳住突破位，优先兑现，避免把突破失败拖成被动持有。",
                intraday_exit,
            ),
            avoid_point="平台确认前不要在箱体中部追单，避免买在真假突破分界处。",
            confidence=confidence,
        )
    if stage.code == "pullback_retest":
        return TomorrowPlan(
            setup_label="回踩低吸确认",
            bias="偏多低吸",
            summary="更看重支撑位附近的缩量企稳和分时二次转强，属于等确认后的低风险切入。",
            buy_point=_merge_plan_text(
                f"更优先观察 MA20 {ma20_text} 与前突破位 {breakout20_text} 的共振支撑，若缩量下探后重新站回均价线，可小仓分批试错。",
                intraday_entry,
            ),
            sell_point=_merge_plan_text(
                f"若跌破 MA20 {ma20_text} 和突破位 {breakout20_text} 后回拉无力先退；若反弹接近 {close_text} 上方强压但量价不配合，也可以先做减法。",
                intraday_exit,
            ),
            avoid_point="不要在下跌途中直接抄底，必须等支撑位先出现承接证据。",
            confidence=confidence,
        )
    if stage.code == "second_attack_attempt":
        return TomorrowPlan(
            setup_label="二次放量再攻",
            bias="偏多等待",
            summary="关键不是提前猜第二波，而是等它真的放量越过前高后再跟，做确认而不是做预判。",
            buy_point=_merge_plan_text(
                f"只有放量越过前高 {breakout60_text} 或整理上沿 {breakout20_text}，且分时均价线同步抬升时再考虑跟随；未越前高先观察。",
                intraday_entry,
            ),
            sell_point=_merge_plan_text(
                f"若冲高不过 {breakout60_text}，或放量回落跌回整理平台并失守 MA20 {ma20_text}，优先减仓，防止高位横久转弱。",
                intraday_exit,
            ),
            avoid_point="高位整理中间区域不反复追单，避免被来回震荡消耗。",
            confidence=confidence,
        )
    if stage.code == "distribution_risk":
        return TomorrowPlan(
            setup_label="冲高兑现优先",
            bias="偏空防守",
            summary="短线任务以保护利润为主，除非重新缩量稳住，否则明天更像卖点管理而不是新开仓窗口。",
            buy_point=_merge_plan_text(
                f"原则上不主动开新仓，只有深度回踩至 MA20 {ma20_text} 或 MA60 {ma60_text} 一带后缩量企稳，并重新站回均价线，才考虑轻仓观察。",
                intraday_entry,
            ),
            sell_point=_merge_plan_text(
                f"冲高不能重新站上 {close_text} 附近强压、或分时失守均价线并伴随放量回落时，优先兑现和降仓。",
                intraday_exit,
            ),
            avoid_point="不要把高位长上影和放量分歧误当成新的突破启动。",
            confidence=confidence,
        )
    if stage.code == "weak_repair":
        return TomorrowPlan(
            setup_label="修复试错观察",
            bias="中性偏谨慎",
            summary="可以关注弱转强信号，但应把它当作试错单而不是确定性趋势单。",
            buy_point=_merge_plan_text(
                f"只有在放量收复 MA20 {ma20_text} 或前期反弹强压 {breakout20_text}，且分时均价线保持承接时，才考虑小仓位试错。",
                intraday_entry,
            ),
            sell_point=_merge_plan_text(
                f"若修复无量、冲高回落，或再次跌回 MA60 {ma60_text} 一带时，及时退出，避免反抽变反套。",
                intraday_exit,
            ),
            avoid_point="修复阶段最怕重仓预判，宁可错过，也不要在弱势里重仓赌拐点。",
            confidence=confidence,
        )
    return TomorrowPlan(
        setup_label="区间边界观察",
        bias="中性等待",
        summary="当前更适合按照区间上沿和下沿处理，不适合在箱体中部做方向性押注。",
        buy_point=_merge_plan_text(
            f"更适合等价格靠近箱体下沿 {range_lower_text} 止跌，或放量突破箱体上沿 {range_upper_text} 并站稳后，再考虑介入。",
            intraday_entry,
        ),
        sell_point=_merge_plan_text(
            f"若接近箱体上沿 {range_upper_text} 但放量不足时先减；若向下跌破箱体下沿 {range_lower_text}，先退出等待下一结构。",
            intraday_exit,
        ),
        avoid_point="区间中段没有优势，不为几根小阳线提前透支容错空间。",
        confidence=confidence,
    )
