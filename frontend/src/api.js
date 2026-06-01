const API_BASE = import.meta.env.VITE_API_BASE || "";
const DEFAULT_TIMEOUT_MS = 30000;

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
  const { timeoutMs = DEFAULT_TIMEOUT_MS, signal, ...fetchOptions } = options;
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
  const abortFromCaller = () => controller.abort();
  if (signal) {
    if (signal.aborted) {
      controller.abort();
    } else {
      signal.addEventListener("abort", abortFromCaller, { once: true });
    }
  }

  let response;
  try {
    response = await fetch(`${API_BASE}${path}${query ? `?${query}` : ""}`, {
      ...fetchOptions,
      signal: controller.signal,
    });
  } catch (error) {
    if (controller.signal.aborted) {
      throw new Error("Request timed out or was cancelled.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeoutId);
    signal?.removeEventListener?.("abort", abortFromCaller);
  }

  if (!response.ok) {
    let message = `Request failed: ${response.status}`;
    try {
      const payload = await response.json();
      if (payload?.detail) {
        message = String(payload.detail);
      }
    } catch {
      // Preserve the HTTP status message when the error body is not JSON.
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
