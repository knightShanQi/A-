function pct(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(1)}%` : "--";
}

function score(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(1) : "--";
}

function freshnessText(freshness) {
  if (!freshness) {
    return "状态待确认";
  }
  if (freshness.is_latest_model_result) {
    return "最新模型结果";
  }
  if (freshness.cache_stale) {
    return "缓存结果";
  }
  if (freshness.freshness_label === "unknown") {
    return "数据日待确认";
  }
  return "非最新收盘日";
}

export default function HeroPanel({ boardPayload, enhancedReady, loading }) {
  const topRow = boardPayload?.board?.top_row || {};
  const boardMeta = boardPayload?.board?.meta || {};
  const freshness = boardPayload?.freshness || boardPayload?.board?.freshness || {};
  const summary = boardPayload?.review_summary || {};
  const modelSnapshot = boardPayload?.model_snapshot || {};
  const scopeLabel = boardMeta.security_scope === "all" ? "原始全市场榜单" : "主板非ST，排除创业板/科创板";
  const scopeCountLabel =
    boardMeta.security_scope === "all"
      ? `原始 ${boardMeta.raw_row_count ?? boardMeta.row_count ?? "--"} 只`
      : `过滤 ${boardMeta.excluded_row_count ?? 0} 只 / 保留 ${boardMeta.filtered_row_count ?? boardMeta.row_count ?? "--"} 只`;

  return (
    <section className="hero-surface">
      <div className="hero-main panel">
        <div className="hero-copy">
          <p className="eyebrow">A股信号台</p>
          <h1>独立前端交易工作台</h1>
          <p className="hero-text">
            首屏优先展示最新可用榜单，再异步补齐板块资金、消息面、回测战绩和个股详情。
            当前模型版本、数据日期和回测健康度会同步锁定，方便比较每次迭代是否真的变好。
          </p>
          <div className="chip-row">
            <span className="data-chip">榜单数据日 {freshness.data_date || boardMeta.market_data_date || "--"}</span>
            <span className="data-chip">最新收盘日 {freshness.latest_market_data_date || boardMeta.latest_market_data_date || "--"}</span>
            <span className="data-chip">{freshnessText(freshness)}</span>
            <span className="data-chip">{scopeLabel}</span>
            <span className="data-chip">{scopeCountLabel}</span>
            <span className="data-chip">{enhancedReady ? "增强结果已就绪" : loading ? "快榜加载中" : "快榜先行"}</span>
          </div>
        </div>

        <div className="hero-focus">
          <div className="focus-label">榜首个股</div>
          <div className="focus-name">{topRow.name || topRow.symbol || "--"}</div>
          <div className="focus-symbol">{topRow.symbol || "--"}</div>
          <div className="focus-grid">
            <div>
              <span>达标概率</span>
              <strong>{pct(topRow.p_hit ?? topRow.calibrated_probability_up ?? topRow.probability_up)}</strong>
            </div>
            <div>
              <span>期望涨幅</span>
              <strong>{pct(topRow.expected_return_pct ?? topRow.predicted_upside_pct)}</strong>
            </div>
            <div>
              <span>排名分</span>
              <strong>{score(topRow.rank_score ?? topRow.attention_score)}</strong>
            </div>
            <div>
              <span>启动信号</span>
              <strong>{topRow.launch_signal_display || topRow.action_label || "观察"}</strong>
            </div>
          </div>
        </div>
      </div>

      <div className="hero-side">
        <div className="panel side-stat-card">
          <span className="side-stat-label">模型版本</span>
          <strong>{modelSnapshot.signature || "--"}</strong>
          <p>{modelSnapshot.model_version_id || "等待 API 返回模型版本快照。"}</p>
        </div>
        <div className="panel side-stat-card">
          <span className="side-stat-label">实战表现</span>
          <strong>{pct(modelSnapshot.win_rate_pct ?? summary.win_rate_pct)}</strong>
          <p>
            健康分 {score(modelSnapshot.health_score)}，
            目标准确率 {pct(modelSnapshot.target_precision_pct || 90)}。
          </p>
        </div>
      </div>
    </section>
  );
}
