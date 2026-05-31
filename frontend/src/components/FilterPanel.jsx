export default function FilterPanel({
  draftParams,
  onChange,
  onApply,
  searchText,
  onSearchChange,
  searchResults,
  onSelectSearchResult,
}) {
  return (
    <section className="command-strip">
      <div className="panel filter-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">筛选命令</p>
            <h2>筛选与搜索</h2>
          </div>
          <div className="panel-note">先展示快榜，再补齐增强榜</div>
        </div>
        <div className="filter-grid">
          <label>
            股票池口径
            <select value={draftParams.security_scope || "main_board"} onChange={(event) => onChange("security_scope", event.target.value)}>
              <option value="main_board">筛选后：主板非ST</option>
              <option value="all">原始榜单：全市场</option>
            </select>
          </label>
          <label>
            排序口径
            <select value={draftParams.ranking_by} onChange={(event) => onChange("ranking_by", event.target.value)}>
              <option value="attention">关注分数</option>
              <option value="probability">上涨概率</option>
            </select>
          </label>
          <label>
            榜单数量
            <input
              type="number"
              min="10"
              max="100"
              step="10"
              value={draftParams.board_size}
              onChange={(event) => onChange("board_size", Number(event.target.value))}
            />
          </label>
          <label>
            预测周期
            <select value={draftParams.horizon_days} onChange={(event) => onChange("horizon_days", Number(event.target.value))}>
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
              value={draftParams.positive_return_pct}
              onChange={(event) => onChange("positive_return_pct", Number(event.target.value))}
            />
          </label>
        </div>
        <label className="watchlist-box">
          自选股票
          <textarea
            value={draftParams.watchlist_text}
            onChange={(event) => onChange("watchlist_text", event.target.value)}
            placeholder="600519 000333 002594"
            rows={3}
          />
        </label>
        <button className="primary-button" onClick={onApply}>应用筛选</button>
      </div>

      <div className="panel search-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">股票搜索</p>
            <h2>搜索任意A股</h2>
          </div>
        </div>
        <div className="search-shell">
          <input
            type="text"
            value={searchText}
            onChange={(event) => onSearchChange(event.target.value)}
            placeholder="688981 / 中芯国际"
          />
          {searchText && (
            <div className="search-results">
              {searchResults.length ? (
                searchResults.map((item) => (
                  <button key={item.symbol} className="search-result-item" onClick={() => onSelectSearchResult(item)}>
                    <strong>{item.name}</strong>
                    <span>{item.symbol}</span>
                  </button>
                ))
              ) : (
                <div className="empty-state small">没有匹配的股票</div>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
