const API_BASE = import.meta.env.VITE_API_BASE || "";

function buildQuery(params = {}) {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null || value === "") {
      return;
    }
    query.set(key, String(value));
  });
  return query.toString();
}

async function request(path, params = {}, options = {}) {
  const query = buildQuery(params);
  const response = await fetch(`${API_BASE}${path}${query ? `?${query}` : ""}`, options);
  if (!response.ok) {
    let message = `请求失败：${response.status}`;
    try {
      const payload = await response.json();
      if (payload?.detail) {
        message = String(payload.detail);
      }
    } catch {
      // Ignore JSON parse failures and preserve the HTTP status message.
    }
    throw new Error(message);
  }
  return response.json();
}

export function fetchQuickBoard(params) {
  return request("/api/board/quick", params);
}

export function fetchEnhancedBoard(params) {
  return request("/api/board/enhanced", params);
}

export function fetchSymbolDetail(symbol, params) {
  return request(`/api/symbol/${encodeURIComponent(symbol)}`, params);
}

export function fetchSymbolNewsImpact(symbol, params) {
  return request(`/api/symbol/${encodeURIComponent(symbol)}/news-impact`, params);
}

export function searchSymbols(query) {
  return request("/api/search", { q: query, limit: 12 });
}

export function fetchLatestMarketBacktest(resultLimit = 50) {
  return request("/api/backtests/market/latest", { result_limit: resultLimit });
}

export function startMarketBacktest(params) {
  return request("/api/tasks/market-backtest", params, { method: "POST" });
}

export function fetchTaskStatus(taskId) {
  return request(`/api/tasks/${encodeURIComponent(taskId)}`);
}
