import PlotlyChart from "./PlotlyChart";

function score(value, suffix = "") {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(1)}${suffix}` : "--";
}

function text(value, fallback = "--") {
  return value === undefined || value === null || value === "" ? fallback : String(value);
}

const LAUNCH_LABEL = {
  breakout: "突破确认",
  ready: "启动就绪",
  watch: "观察蓄势",
  wait: "等待确认",
};

function simpleRows(rows, columns) {
  if (!rows?.length) {
    return <div className="empty-state small">暂无数据</div>;
  }

  const inferredColumns =
    columns ||
    Object.keys(rows[0] || {})
      .slice(0, 3)
      .map((key) => ({ key, label: `${key}: ` }));

  return (
    <div className="simple-list">
      {rows.slice(0, 8).map((row, index) => (
        <div className="simple-list-item" key={index}>
          {inferredColumns.map((column) => (
            <span key={column.key}>
              <strong>{column.label}</strong>
              {column.render ? column.render(row[column.key], row) : text(row[column.key])}
            </span>
          ))}
        </div>
      ))}
    </div>
  );
}

function Metric({ label, value, note }) {
  return (
    <div className="metric-card">
      <span className="metric-label">{label}</span>
      <strong className="metric-value">{value}</strong>
      <span className="metric-note">{note}</span>
    </div>
  );
}

function launchDisplay(context) {
  const label = context.launch_signal_label || "wait";
  return context.launch_signal_display || LAUNCH_LABEL[label] || "等待确认";
}

const NEWS_DIRECTION_LABEL = {
  bullish: "偏多",
  bearish: "偏空",
  neutral: "中性",
};

const NEWS_CATEGORY_LABEL = {
  earnings: "业绩",
  contract_order: "订单合同",
  shareholder_action: "股东行为",
  financing_mna: "融资并购",
  policy_sector: "政策产业",
  regulatory_risk: "监管风险",
  product_technology: "产品技术",
  market_opinion: "市场观点",
  accident_risk: "经营风险",
  general: "综合消息",
};

function directionLabel(value) {
  return NEWS_DIRECTION_LABEL[value] || text(value, "中性");
}

function categoryLabel(value) {
  return NEWS_CATEGORY_LABEL[value] || text(value, "综合消息");
}

function directionClass(value) {
  if (value === "bullish") {
    return "positive";
  }
  if (value === "bearish") {
    return "negative";
  }
  return "neutral";
}

function formatDate(value) {
  if (!value) {
    return "--";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value).slice(0, 16);
  }
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")} ${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
}

function NewsImpactPanel({ newsImpact, loading, error }) {
  if (loading && !newsImpact) {
    return (
      <section className="panel news-impact-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">消息影响</p>
            <h3>消息影响分析</h3>
          </div>
        </div>
        <div className="empty-state small">消息影响加载中...</div>
      </section>
    );
  }

  if (error && !newsImpact) {
    return (
      <section className="panel news-impact-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">消息影响</p>
            <h3>消息影响分析</h3>
          </div>
        </div>
        <div className="empty-state small">{error}</div>
      </section>
    );
  }

  if (!newsImpact) {
    return null;
  }

  const signal = newsImpact.latest_signal || {};
  const categories = newsImpact.category_summary || [];
  const events = newsImpact.event_impacts?.length ? newsImpact.event_impacts : newsImpact.events || [];
  const topCategory = categories[0] || signal.top_categories?.[0] || null;

  return (
    <section className="panel news-impact-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">消息影响</p>
          <h3>消息影响分析</h3>
        </div>
        <span className={`pill ${directionClass(signal.label)}`}>{directionLabel(signal.label)}</span>
      </div>

      <div className="news-impact-metrics">
        <Metric label="最新冲击" value={score(signal.score)} note={signal.summary || "等待样本沉淀"} />
        <Metric label="事件样本" value={newsImpact.impact_sample_count || 0} note={`抓取 ${newsImpact.event_count || 0} 条`} />
        <Metric label="主导类别" value={categoryLabel(topCategory?.event_category)} note={`样本 ${topCategory?.event_count || 0}`} />
      </div>

      <div className="news-impact-grid">
        <div>
          <h4>分类后验</h4>
          <div className="simple-list">
            {categories.length ? (
              categories.slice(0, 5).map((row) => (
                <div className="simple-list-item news-category-row" key={row.event_category}>
                  <span>
                    <strong>{categoryLabel(row.event_category)}</strong>
                    样本 {row.event_count || 0}
                  </span>
                  <span>1日均值 {score(row.avg_return_1d_pct, "%")} / 命中 {score((row.direction_hit_rate_1d || 0) * 100, "%")}</span>
                  <span>开盘缺口 {score(row.avg_open_gap_pct, "%")} / 预期 {score(row.avg_expected_impact_score)}</span>
                </div>
              ))
            ) : (
              <div className="empty-state small">暂无分类样本</div>
            )}
          </div>
        </div>
        <div>
          <h4>事件样本</h4>
          <div className="simple-list">
            {events.length ? (
              events.slice(0, 5).map((row, index) => (
                <div className="simple-list-item news-event-row" key={`${row.dedupe_key || row.title}-${index}`}>
                  <span>
                    <strong>{categoryLabel(row.event_category)}</strong>
                    <span className={`pill tiny ${directionClass(row.event_direction)}`}>{directionLabel(row.event_direction)}</span>
                  </span>
                  <span>{text(row.title, "未命名消息")}</span>
                  <span>{formatDate(row.published_at)} / 1日 {score(row.return_1d_pct, "%")}</span>
                </div>
              ))
            ) : (
              <div className="empty-state small">暂无事件样本</div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

export default function DetailPanel({ detail, loading, newsImpact, newsImpactLoading, newsImpactError }) {
  if (loading && !detail) {
    return (
      <aside className="detail-stack">
        <section className="panel detail-panel">
          <div className="empty-state">个股详情加载中...</div>
        </section>
        <NewsImpactPanel newsImpact={newsImpact} loading={newsImpactLoading} error={newsImpactError} />
      </aside>
    );
  }

  if (!detail) {
    return (
      <aside className="detail-stack">
        <section className="panel detail-panel">
          <div className="empty-state">从关注榜或搜索结果里选择一只股票</div>
        </section>
        <NewsImpactPanel newsImpact={newsImpact} loading={newsImpactLoading} error={newsImpactError} />
      </aside>
    );
  }

  const hero = detail.hero || {};
  const context = detail.display_context || {};
  const signals = detail.signals || {};
  const freshness = detail.freshness || context.freshness || {};
  const modelSnapshot = detail.model_snapshot || context.model_snapshot || {};

  return (
    <aside className="detail-stack">
      <section className="panel detail-panel">
        <div className="detail-hero">
          <div>
            <p className="eyebrow">信号详情</p>
            <h2>{hero.name || detail.symbol}</h2>
            <p className="detail-subline">
              <span>{detail.symbol}</span>
              <span>{hero.industry || "未知行业"}</span>
              <span>{context.action_label || "观察"}</span>
              <span>{launchDisplay(context)}</span>
            </p>
          </div>
          <div className="hero-badge-block">
            <span className="data-chip">{hero.analysis_date || "--"}</span>
            <span className="data-chip">{freshness.is_latest_model_result ? "最新模型结果" : "缓存/历史结果"}</span>
            <span className="data-chip">模型 {modelSnapshot.signature || "--"}</span>
            <span className="data-chip">{hero.board_label || "主板"}</span>
          </div>
        </div>

        <div className="metrics-grid">
          <Metric
            label="达标概率"
            value={score(context.p_hit ?? context.calibrated_probability_up ?? context.probability_up, "%")}
            note={`置信 ${score(context.probability_confidence, "%")} / 区间 ${score(context.probability_band_low, "%")}-${score(context.probability_band_high, "%")}`}
          />
          <Metric
            label="期望涨幅"
            value={score(context.expected_return_pct ?? context.predicted_upside_pct, "%")}
            note={`区间 ${score(context.predicted_upside_low_pct, "%")} - ${score(context.predicted_upside_high_pct, "%")}`}
          />
          <Metric label="启动专项分" value={score(context.launch_specialist_score)} note={`${launchDisplay(context)} / ${context.launch_phase_display || "阶段待确认"}`} />
          <Metric label="分时板块联动" value={score(context.intraday_sector_sync_score)} note={`${context.intraday_sector_display || "--"} / 相对强弱 ${score(context.relative_intraday_alpha)}`} />
          <Metric label="交易决策" value={context.action_label || "观察"} note={`决策分 ${score(context.action_score)} / 置信 ${score(context.action_confidence)}`} />
          <Metric label="风控等级" value={context.risk_level_display || "--"} note={`建议仓位 ${score(context.suggested_position_pct, "%")} / ${context.risk_control_note || "等待确认"}`} />
          <Metric label="止损止盈" value={`${score(context.stop_loss_pct, "%")} / ${score(context.take_profit_pct, "%")}`} note="参考止损 / 参考止盈，不替代人工决策" />
          <Metric label="综合排名分" value={score(context.rank_score)} note={`原始概率 ${score(context.raw_probability_up, "%")} / 回撤 ${score(context.drawdown_risk_pct, "%")}`} />
        </div>

        <div className="detail-copy-grid">
          <div className="story-card">
            <h3>明日交易计划</h3>
            <p>{signals.tomorrow_plan?.setup_label || context.tomorrow_setup || "等待下一交易日确认结构。"}</p>
            <p>买点：{signals.tomorrow_plan?.buy_point || context.tomorrow_buy_point || "等待分时承接"}</p>
            <p>卖点：{signals.tomorrow_plan?.sell_point || context.tomorrow_sell_point || "跌破失效位先处理"}</p>
          </div>
          <div className="story-card">
            <h3>启动逻辑</h3>
            <p>启动窗口: {score(context.launch_window_score)} / {launchDisplay(context)}</p>
            <p>阶段判断: {context.launch_phase_display || "等待确认"} / {context.launch_reason_text || "等待量价和板块确认"}</p>
            <p>分时联动: {context.intraday_sector_note || "等待分时与板块联动确认"}</p>
            <p>执行窗口: {context.execution_window || context.execution_label || "等待结构确认"}</p>
            <p>风报比: {context.reward_risk_label || "--"} / {score(context.reward_risk_ratio)}</p>
          </div>
        </div>
      </section>

      <section className="panel chart-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">分时结构</p>
            <h3>分时工作台</h3>
          </div>
          <div className="panel-note">{hero.intraday_label || "关注均价线、尾盘承接与量能变化"}</div>
        </div>
        <PlotlyChart figure={detail.charts?.minute} />
      </section>

      <section className="panel chart-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">日线结构</p>
            <h3>日 K 趋势结构</h3>
          </div>
          <div className="panel-note">{signals.stage?.label || "结构待确认"}</div>
        </div>
        <PlotlyChart figure={detail.charts?.daily} />
      </section>

      <details className="advanced-section">
        <summary>高级分析：板块、资金、消息与回测</summary>
        <section className="signal-triad">
          <div className="panel story-card">
            <h3>市场、板块与量化</h3>
            <p>行业热度: {score(signals.sector?.sector_score)} / {signals.sector?.sector_summary || "待补齐"}</p>
            <p>主力资金: {score(signals.fund?.fund_score)} / {signals.fund?.summary || "待补齐"}</p>
            <p>量化辅助: {score(signals.quant?.score)} / {signals.quant?.primary_signal || "统一模型"}</p>
          </div>
          <div className="panel story-card">
            <h3>模型与回测</h3>
            <p>模型信号: {signals.model?.signal_label || "--"}</p>
            <p>策略分 / 一致性: {score(signals.model?.strategy_score)} / {score(signals.model?.agreement_score)}</p>
            <p>历史命中率: {score((signals.backtest?.achieved_precision || 0) * 100, "%")} / 样本 {signals.backtest?.trade_count || 0}</p>
          </div>
        </section>

        <NewsImpactPanel newsImpact={newsImpact} loading={newsImpactLoading} error={newsImpactError} />

        <section className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">资金与消息</p>
              <h3>资金与消息</h3>
            </div>
          </div>
          <div className="detail-copy-grid">
            <div>
              <h4>主力资金</h4>
              {simpleRows(detail.fund_flow)}
            </div>
            <div>
              <h4>最新消息</h4>
              {simpleRows(detail.news)}
            </div>
          </div>
        </section>
      </details>
    </aside>
  );
}
