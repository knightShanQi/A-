import { useEffect, useState } from "react";

function pctRatio(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(2)}%` : "--";
}

function pct(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(1)}%` : "--";
}

function value(value) {
  return value === undefined || value === null || value === "" ? "--" : value;
}

function strategyText(value) {
  const token = String(value || "").toLowerCase();
  if (token.includes("strategy1") || token.includes("策略1")) {
    return "策略1";
  }
  if (token.includes("strategy2") || token.includes("策略2")) {
    return "策略2";
  }
  if (token.includes("strategy3") || token.includes("策略3")) {
    return "策略3";
  }
  if (token === "all") {
    return "全部策略";
  }
  return value === undefined || value === null || value === "" ? "--" : String(value);
}

function progressPhaseText(value) {
  const token = String(value || "");
  const phaseMap = {
    prepare: "准备数据",
    feature_store: "构建特征库",
    candidate_pool: "筛选候选池",
    forward_eval: "验证未来收益",
    write_outputs: "写入结果",
  };
  return phaseMap[token] || value || "--";
}

function progressMessageText(message) {
  const text = String(message || "");
  if (!text) {
    return "--";
  }
  const loaded = text.match(/^Loaded (\d+) symbols and (\d+) trade dates$/);
  if (loaded) {
    return `已加载 ${loaded[1]} 只股票和 ${loaded[2]} 个交易日`;
  }
  const building = text.match(/^Building feature store for (.+)$/);
  if (building) {
    return `正在构建 ${building[1]} 的特征库`;
  }
  const screening = text.match(/^Screening candidates for (.+)$/);
  if (screening) {
    return `正在筛选 ${screening[1]} 的候选池`;
  }
  const finished = text.match(/^Finished (.+), accumulated (\d+) rows$/);
  if (finished) {
    return `已完成 ${finished[1]}，累计 ${finished[2]} 条记录`;
  }
  const saved = text.match(/^Saved summary to (.+)$/);
  if (saved) {
    return `摘要已保存到 ${saved[1]}`;
  }
  return text;
}

function isoDate(value) {
  const date = value ? new Date(value) : new Date();
  if (Number.isNaN(date.getTime())) {
    return new Date().toISOString().slice(0, 10);
  }
  return date.toISOString().slice(0, 10);
}

function defaultStartDate(endDate) {
  const date = new Date(endDate);
  date.setDate(date.getDate() - 30);
  return isoDate(date);
}

export default function MarketBacktestPanel({
  latestMarketDate,
  horizonDays,
  positiveReturnPct,
  latestBacktest,
  taskStatus,
  taskRunning,
  error,
  onRun,
  onRefresh,
}) {
  const endDate = isoDate(latestMarketDate);
  const [form, setForm] = useState({
    date_from: defaultStartDate(endDate),
    date_to: endDate,
    horizon_days: Number(horizonDays) || 3,
    positive_return_pct: Number(positiveReturnPct) || 10,
    strategy_mode: "all",
    top_k: 50,
    force_rebuild: false,
  });

  useEffect(() => {
    setForm((current) => ({
      ...current,
      date_to: current.date_to || endDate,
      horizon_days: Number(horizonDays) || current.horizon_days || 3,
      positive_return_pct: Number(positiveReturnPct) || current.positive_return_pct || 10,
    }));
  }, [endDate, horizonDays, positiveReturnPct]);

  const summary = latestBacktest?.summary || {};
  const rows = Array.isArray(latestBacktest?.results) ? latestBacktest.results : [];
  const progress = taskStatus?.progress || {};
  const completed = Number(progress.completed || 0);
  const total = Math.max(Number(progress.total || 1), 1);
  const progressPct = Math.min(Math.max(Math.round((completed / total) * 100), 0), 100);

  function update(key, nextValue) {
    setForm((current) => ({ ...current, [key]: nextValue }));
  }

  function submit(event) {
    event.preventDefault();
    onRun({
      ...form,
      horizon_days: Number(form.horizon_days) || 3,
      positive_return_pct: Number(form.positive_return_pct) || 10,
      top_k: Number(form.top_k) || 50,
      force_rebuild: Boolean(form.force_rebuild),
    });
  }

  return (
    <section className="panel market-backtest-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">手动回测</p>
          <h3>全市场复盘</h3>
        </div>
        <button className="ghost-button" type="button" onClick={onRefresh}>
          刷新结果
        </button>
      </div>

      <p className="panel-note">
        重建每日全市场候选池，并评估持有 1、3、5 个交易日的平均收益。
        历史结构仍使用项目本地数据路径，Tushare 只作为未复权横截面补充。
      </p>

      <form className="backtest-form" onSubmit={submit}>
        <label>
          开始日期
          <input type="date" value={form.date_from} onChange={(event) => update("date_from", event.target.value)} />
        </label>
        <label>
          结束日期
          <input type="date" value={form.date_to} onChange={(event) => update("date_to", event.target.value)} />
        </label>
        <label>
          策略
          <select value={form.strategy_mode} onChange={(event) => update("strategy_mode", event.target.value)}>
            <option value="all">全部策略</option>
            <option value="strategy1">策略1</option>
            <option value="strategy2">策略2</option>
            <option value="strategy3">策略3·多因子主升预备</option>
          </select>
        </label>
        <label>
          预测周期
          <select value={form.horizon_days} onChange={(event) => update("horizon_days", Number(event.target.value))}>
            <option value="3">3日</option>
            <option value="5">5日</option>
            <option value="10">10日</option>
          </select>
        </label>
        <label>
          达标涨幅
          <input
            type="number"
            min="5"
            max="50"
            step="1"
            value={form.positive_return_pct}
            onChange={(event) => update("positive_return_pct", Number(event.target.value))}
          />
        </label>
        <label>
          每日取前 K 只
          <input
            type="number"
            min="1"
            max="300"
            step="10"
            value={form.top_k}
            onChange={(event) => update("top_k", Number(event.target.value))}
          />
        </label>
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={form.force_rebuild}
            onChange={(event) => update("force_rebuild", event.target.checked)}
          />
          强制重建
        </label>
        <button className="primary-button" type="submit" disabled={taskRunning}>
          {taskRunning ? "运行中..." : "运行全市场回测"}
        </button>
      </form>

      {taskRunning ? (
        <div className="backtest-progress">
          <div className="progress-bar">
            <span style={{ width: `${progressPct}%` }} />
          </div>
          <p>{progressPhaseText(progress.phase)}：{progressMessageText(progress.message)}</p>
        </div>
      ) : null}

      {error ? <div className="error-banner">{error}</div> : null}

      <div className="holding-return-grid">
        <div className="metric-card">
          <span className="metric-label">持有1日均值</span>
          <strong className="metric-value">{pctRatio(summary.avg_hold_1d_return)}</strong>
          <span className="metric-note">{value(summary.hold_1d_sample_count)} 个样本</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">持有3日均值</span>
          <strong className="metric-value">{pctRatio(summary.avg_hold_3d_return)}</strong>
          <span className="metric-note">{value(summary.hold_3d_sample_count)} 个样本</span>
        </div>
        <div className="metric-card">
          <span className="metric-label">持有5日均值</span>
          <strong className="metric-value">{pctRatio(summary.avg_hold_5d_return)}</strong>
          <span className="metric-note">{value(summary.hold_5d_sample_count)} 个样本</span>
        </div>
      </div>

      <div className="backtest-summary-grid">
        <div>
          <span>交易次数</span>
          <strong>{value(summary.trade_count)}</strong>
        </div>
        <div>
          <span>胜率</span>
          <strong>{pctRatio(summary.win_rate)}</strong>
        </div>
        <div>
          <span>达标率</span>
          <strong>{pctRatio(summary.target_hit_rate)}</strong>
        </div>
        <div>
          <span>周期平均收益</span>
          <strong>{pctRatio(summary.avg_forward_return)}</strong>
        </div>
      </div>

      {latestBacktest?.results_path ? <p className="panel-note">最新文件：{latestBacktest.results_path}</p> : null}

      <div className="mini-table-shell">
        <table className="mini-table">
          <thead>
            <tr>
              <th>日期</th>
              <th>代码</th>
              <th>策略</th>
              <th>持有1日</th>
              <th>持有3日</th>
              <th>持有5日</th>
            </tr>
          </thead>
          <tbody>
            {rows.length ? (
              rows.slice(0, 8).map((row, index) => (
                <tr key={`${row.market_date}-${row.symbol}-${index}`}>
                  <td>{value(row.market_date)}</td>
                  <td>{value(row.symbol)}</td>
                  <td>{strategyText(row.candidate_strategy)}</td>
                  <td>{pctRatio(row.hold_1d_return)}</td>
                  <td>{pctRatio(row.hold_3d_return)}</td>
                  <td>{pctRatio(row.hold_5d_return)}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={6} className="empty-cell">
                  暂无手动全市场回测结果。
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}
