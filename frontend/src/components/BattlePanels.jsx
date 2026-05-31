function pct(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(1)}%` : "--";
}

function score(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(1) : "--";
}

function value(value) {
  return value === undefined || value === null || value === "" ? "--" : value;
}

function sampleTone(count) {
  const number = Number(count);
  if (!Number.isFinite(number) || number < 20) {
    return "样本偏少";
  }
  if (number < 80) {
    return "可做辅助";
  }
  return "样本充分";
}

function healthClass(status) {
  if (status === "target_met") {
    return "pill positive";
  }
  if (status === "usable") {
    return "pill neutral";
  }
  return "pill";
}

function HealthPanel({ health }) {
  if (!health) {
    return null;
  }

  return (
    <section className="panel battle-panel battle-health-panel">
        <div className="panel-header">
          <div>
          <p className="eyebrow">实战准确率</p>
          <h3>实战健康度</h3>
        </div>
        <span className={healthClass(health.status_label)}>
          {health.status_display || "状态待确认"}
        </span>
      </div>

      <div className="health-grid">
        <div>
          <span>健康分</span>
          <strong>{score(health.health_score)}</strong>
        </div>
        <div>
          <span>上涨胜率</span>
          <strong>{pct(health.win_rate_pct)}</strong>
        </div>
        <div>
          <span>目标差距</span>
          <strong>{pct(health.precision_gap_to_target_pct)}</strong>
        </div>
        <div>
          <span>样本数</span>
          <strong>{value(health.sample_count)}</strong>
        </div>
      </div>

      <p className="panel-note">
        目标上涨准确率 {pct(health.target_precision_pct)}，当前平均收益 {pct(health.avg_return_pct)}，
        校准误差 {pct(health.calibration_gap_pct)}。这个面板用于判断模型是否真的在复盘中变好。
      </p>
    </section>
  );
}

function renderPanelTable(title, subtitle, rows, columns) {
  const safeRows = Array.isArray(rows) ? rows : [];
  return (
    <section className="panel battle-panel" key={title}>
      <div className="panel-header">
        <div>
          <p className="eyebrow">复盘验证</p>
          <h3>{title}</h3>
        </div>
        <div className="panel-note">{subtitle}</div>
      </div>
      <div className="mini-table-shell">
        <table className="mini-table">
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column.key}>{column.label}</th>
              ))}
              <th>可信度</th>
            </tr>
          </thead>
          <tbody>
            {safeRows.length ? (
              safeRows.slice(0, 6).map((row, index) => (
                <tr key={`${title}-${index}`}>
                  {columns.map((column) => (
                    <td key={column.key}>
                      {column.render ? column.render(row[column.key], row) : value(row[column.key])}
                    </td>
                  ))}
                  <td>{sampleTone(row.sample_count)}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={columns.length + 1} className="empty-cell">
                  暂无足够复盘样本，后台完成回测后会自动补齐。
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default function BattlePanels({ reviewPanels, reviewHealth }) {
  const marketRows = reviewPanels?.long_market_state_panel?.length
    ? reviewPanels.long_market_state_panel
    : reviewPanels?.short_market_state_panel || [];

  return (
    <>
      <HealthPanel health={reviewHealth} />
      {reviewPanels ? (
        <div className="battle-grid">
          {renderPanelTable(
            "分策略实战战绩",
            "观察每种选股策略在近期复盘中的命中率和收益质量",
            reviewPanels.strategy_panel || [],
            [
              { key: "candidate_strategy_label", label: "策略" },
              { key: "win_rate_pct", label: "上涨率", render: pct },
              { key: "avg_return_pct", label: "平均收益", render: pct },
              { key: "sample_count", label: "样本数" },
            ],
          )}
          {renderPanelTable(
            "市场状态分层",
            "同一模型在趋势、轮动、防守环境下分开评估",
            marketRows,
            [
              { key: "market_state_display", label: "市场状态" },
              { key: "win_rate_pct", label: "上涨率", render: pct },
              { key: "avg_return_pct", label: "平均收益", render: pct },
              { key: "sample_count", label: "样本数" },
            ],
          )}
          {renderPanelTable(
            "状态 x 策略组合",
            "更接近真实交易口径，用于判断策略适用场景",
            reviewPanels.combo_panel || [],
            [
              { key: "market_state_display", label: "市场状态" },
              { key: "candidate_strategy_label", label: "策略" },
              { key: "win_rate_pct", label: "上涨率", render: pct },
              { key: "avg_return_pct", label: "平均收益", render: pct },
              { key: "sample_count", label: "样本数" },
            ],
          )}
        </div>
      ) : null}
    </>
  );
}
