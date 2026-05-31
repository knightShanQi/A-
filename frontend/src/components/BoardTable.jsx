function formatPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "--";
  }
  return `${number.toFixed(1)}%`;
}

function formatScore(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "--";
  }
  return number.toFixed(1);
}

const ACTION_CLASS = {
  买: "pill positive",
  卖: "pill negative",
  持: "pill neutral",
  观察: "pill",
};

const LAUNCH_LABEL = {
  breakout: "突破确认",
  ready: "启动就绪",
  watch: "观察蓄势",
  wait: "等待确认",
};

function strategyText(row) {
  const label = String(row.candidate_strategy_label || row.candidate_strategy_short_label || "").trim();
  const strategy = String(row.candidate_strategy || "").trim().toLowerCase();
  const combined = `${label} ${strategy}`.toLowerCase();
  if (combined.includes("策略3") || combined.includes("strategy3")) {
    return label || "策略3·多因子主升预备";
  }
  if (label) {
    return label;
  }
  if (strategy.includes("策略1") || strategy.includes("strategy1")) {
    return "策略1·趋势中继";
  }
  if (strategy.includes("策略2") || strategy.includes("strategy2")) {
    return "策略2·突破共振";
  }
  if (strategy.includes("dynamic_fallback") || strategy.includes("fallback")) {
    return "非正式兜底池";
  }
  return "通用模型";
}

function strategyClass(row) {
  const strategy = `${row.candidate_strategy || ""} ${row.candidate_strategy_label || ""}`.toLowerCase();
  if (strategy.includes("策略3") || strategy.includes("strategy3")) {
    return "pill strategy-pill strategy-three";
  }
  if (strategy.includes("策略1") || strategy.includes("strategy1")) {
    return "pill strategy-pill strategy-one";
  }
  if (strategy.includes("策略2") || strategy.includes("strategy2")) {
    return "pill strategy-pill strategy-two";
  }
  if (strategy.includes("fallback") || strategy.includes("兜底")) {
    return "pill strategy-pill strategy-fallback";
  }
  return "pill strategy-pill";
}

function actionClass(action) {
  return ACTION_CLASS[String(action || "")] || "pill";
}

function launchClass(label) {
  if (label === "breakout" || label === "ready") {
    return "pill positive";
  }
  if (label === "watch") {
    return "pill neutral";
  }
  return "pill";
}

export default function BoardTable({ rows = [], selectedSymbol, onSelect, loading }) {
  return (
    <div className="panel board-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">关注榜</p>
          <h2>全市场动态关注榜</h2>
        </div>
        <div className="panel-note">
          {loading ? "后台正在补齐板块、资金与回测增强" : "已切换到增强榜单"}
        </div>
      </div>

      <div className="board-table-shell">
        <table className="board-table">
          <thead>
            <tr>
              <th>排名</th>
              <th>股票</th>
              <th>策略来源</th>
              <th>决策</th>
              <th>达标概率</th>
              <th>期望涨幅</th>
              <th>排名分</th>
              <th>启动信号</th>
              <th>联动</th>
              <th>风险</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const symbol = String(row.symbol || "");
              const active = symbol === selectedSymbol;
              const launchLabel = row.launch_signal_label || "wait";
              return (
                <tr
                  key={symbol}
                  className={active ? "is-active" : ""}
                  onClick={() => onSelect?.(symbol)}
                >
                  <td>{row.rank || "--"}</td>
                  <td>
                    <div className="stock-cell">
                      <strong>{row.name || symbol}</strong>
                      <span>{symbol}</span>
                    </div>
                  </td>
                  <td>
                    <span className={strategyClass(row)}>{strategyText(row)}</span>
                  </td>
                  <td>
                    <span className={actionClass(row.action_label)}>{row.action_label || "观察"}</span>
                  </td>
                  <td>{formatPercent(row.p_hit ?? row.calibrated_probability_up ?? row.probability_up)}</td>
                  <td>{formatPercent(row.expected_return_pct ?? row.predicted_upside_pct)}</td>
                  <td>{formatScore(row.rank_score)}</td>
                  <td>
                    <span className={launchClass(launchLabel)}>
                      {row.launch_signal_display || LAUNCH_LABEL[launchLabel] || "等待确认"}
                    </span>
                  </td>
                  <td>{formatScore(row.intraday_sector_sync_score)}</td>
                  <td>{row.risk_level_display || "--"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
