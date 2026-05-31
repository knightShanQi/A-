import { useEffect, useRef, useState } from "react";
import {
  fetchEnhancedBoard,
  fetchLatestMarketBacktest,
  fetchQuickBoard,
  fetchSymbolDetail,
  fetchSymbolNewsImpact,
  fetchTaskStatus,
  searchSymbols,
  startMarketBacktest,
} from "./api";
import BattlePanels from "./components/BattlePanels";
import BoardTable from "./components/BoardTable";
import DetailPanel from "./components/DetailPanel";
import FilterPanel from "./components/FilterPanel";
import HeroPanel from "./components/HeroPanel";
import MarketBacktestPanel from "./components/MarketBacktestPanel";

const DEFAULT_PARAMS = {
  ranking_by: "attention",
  board_size: 50,
  horizon_days: 3,
  positive_return_pct: 10,
  watchlist_text: "",
  security_scope: "main_board",
};

export default function App() {
  const [draftParams, setDraftParams] = useState(DEFAULT_PARAMS);
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const [quickPayload, setQuickPayload] = useState(null);
  const [enhancedPayload, setEnhancedPayload] = useState(null);
  const [selectedSymbol, setSelectedSymbol] = useState("");
  const [detail, setDetail] = useState(null);
  const [newsImpact, setNewsImpact] = useState(null);
  const [searchText, setSearchText] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [quickLoading, setQuickLoading] = useState(true);
  const [enhancedLoading, setEnhancedLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [newsImpactLoading, setNewsImpactLoading] = useState(false);
  const [newsImpactError, setNewsImpactError] = useState("");
  const [marketBacktest, setMarketBacktest] = useState(null);
  const [marketBacktestTask, setMarketBacktestTask] = useState(null);
  const [marketBacktestError, setMarketBacktestError] = useState("");
  const [error, setError] = useState("");
  const [softNotice, setSoftNotice] = useState("");
  const quickRequestRef = useRef(0);
  const detailRequestRef = useRef(0);
  const newsImpactRequestRef = useRef(0);

  useEffect(() => {
    let canceled = false;
    let quickSucceeded = false;
    const requestId = quickRequestRef.current + 1;
    quickRequestRef.current = requestId;
    setQuickLoading(true);
    setEnhancedLoading(false);
    setError("");
    setSoftNotice("");
    setEnhancedPayload(null);
    setDetail(null);

    fetchQuickBoard(params)
      .then((payload) => {
        if (canceled || requestId !== quickRequestRef.current) {
          return null;
        }
        setQuickPayload(payload);
        quickSucceeded = true;
        const topSymbol = payload?.board?.top_row?.symbol || "";
        setSelectedSymbol(topSymbol);
        setQuickLoading(false);
        setEnhancedLoading(true);
        return fetchEnhancedBoard(params);
      })
      .then((payload) => {
        if (!payload || canceled || requestId !== quickRequestRef.current) {
          return;
        }
        setEnhancedPayload(payload);
        setEnhancedLoading(false);
        const topSymbol = payload?.board?.top_row?.symbol || "";
        setSelectedSymbol((current) => current || topSymbol);
      })
      .catch((requestError) => {
        if (canceled || requestId !== quickRequestRef.current) {
          return;
        }
        setQuickLoading(false);
        setEnhancedLoading(false);
        if (quickSucceeded) {
          setSoftNotice("增强数据暂时不可用，已保留快榜结果。");
        } else {
          setError(requestError.message || "榜单加载失败");
        }
      });

    return () => {
      canceled = true;
    };
  }, [params]);

  useEffect(() => {
    if (!selectedSymbol) {
      return undefined;
    }
    let canceled = false;
    const requestId = detailRequestRef.current + 1;
    detailRequestRef.current = requestId;
    setDetailLoading(true);
    fetchSymbolDetail(selectedSymbol, {
      horizon_days: params.horizon_days,
      positive_return_pct: params.positive_return_pct,
      security_scope: params.security_scope,
    })
      .then((payload) => {
        if (canceled || requestId !== detailRequestRef.current) {
          return;
        }
        setDetail(payload);
        setDetailLoading(false);
      })
      .catch((requestError) => {
        if (canceled || requestId !== detailRequestRef.current) {
          return;
        }
        setDetailLoading(false);
        setSoftNotice(requestError.message || "个股详情暂时不可用，榜单仍可正常查看。");
      });
    return () => {
      canceled = true;
    };
  }, [selectedSymbol, params.horizon_days, params.positive_return_pct, params.security_scope]);

  useEffect(() => {
    if (!selectedSymbol) {
      setNewsImpact(null);
      setNewsImpactLoading(false);
      return undefined;
    }
    let canceled = false;
    const requestId = newsImpactRequestRef.current + 1;
    newsImpactRequestRef.current = requestId;
    setNewsImpactLoading(true);
    setNewsImpact(null);
    setNewsImpactError("");
    fetchSymbolNewsImpact(selectedSymbol, {
      news_limit: 80,
      horizons: "1,3,5",
      include_disclosures: true,
    })
      .then((payload) => {
        if (canceled || requestId !== newsImpactRequestRef.current) {
          return;
        }
        setNewsImpact(payload);
        setNewsImpactLoading(false);
      })
      .catch((requestError) => {
        if (canceled || requestId !== newsImpactRequestRef.current) {
          return;
        }
        setNewsImpact(null);
        setNewsImpactLoading(false);
        setNewsImpactError(requestError.message || "消息影响分析暂时不可用");
      });
    return () => {
      canceled = true;
    };
  }, [selectedSymbol]);

  useEffect(() => {
    if (!searchText.trim()) {
      setSearchResults([]);
      return undefined;
    }
    const timer = window.setTimeout(() => {
      searchSymbols(searchText)
        .then((payload) => {
          setSearchResults(payload.results || []);
        })
        .catch(() => {
          setSearchResults([]);
        });
    }, 180);
    return () => {
      window.clearTimeout(timer);
    };
  }, [searchText]);

  useEffect(() => {
    fetchLatestMarketBacktest(40)
      .then((payload) => {
        setMarketBacktest(payload);
      })
      .catch(() => {
        setMarketBacktest(null);
      });
  }, []);

  useEffect(() => {
    if (!marketBacktestTask?.task_id || marketBacktestTask.status !== "running") {
      return undefined;
    }
    const timer = window.setInterval(() => {
      fetchTaskStatus(marketBacktestTask.task_id)
        .then((payload) => {
          setMarketBacktestTask(payload);
          if (payload.status === "completed") {
            fetchLatestMarketBacktest(40).then((latestPayload) => {
              setMarketBacktest(latestPayload);
            });
          }
          if (payload.status === "failed") {
            setMarketBacktestError(payload.error || "全市场回测失败");
          }
        })
        .catch((requestError) => {
          setMarketBacktestError(requestError.message || "全市场回测状态暂时不可用");
        });
    }, 2500);
    return () => {
      window.clearInterval(timer);
    };
  }, [marketBacktestTask]);

  const boardPayload = enhancedPayload || quickPayload;
  const boardRows = boardPayload?.board?.rows || [];
  const reviewPanels = enhancedPayload?.review_panels || null;
  const reviewHealth = boardPayload?.review_health || null;
  const latestMarketDate = boardPayload?.freshness?.latest_market_data_date || boardPayload?.board?.meta?.latest_market_data_date;
  const strategyNotice = boardPayload?.board?.meta?.strategy_notice || boardPayload?.board?.strategy_notice || "";
  const pipelineSteps = [
    {
      label: "快榜",
      status: quickLoading ? "running" : quickPayload ? "done" : "waiting",
      note: quickPayload ? "已展示最新可用榜单" : "读取本地缓存与最新收盘日",
    },
    {
      label: "增强",
      status: enhancedLoading ? "running" : enhancedPayload ? "done" : quickPayload ? "waiting" : "pending",
      note: enhancedPayload ? "板块资金与回测已补齐" : "后台补齐资金、消息和复盘",
    },
    {
      label: "详情",
      status: detailLoading ? "running" : detail ? "done" : selectedSymbol ? "waiting" : "pending",
      note: selectedSymbol ? `当前 ${selectedSymbol}` : "等待选择个股",
    },
    {
      label: "消息影响",
      status: newsImpactLoading ? "running" : newsImpact ? "done" : selectedSymbol ? "waiting" : "pending",
      note: newsImpact ? `样本 ${newsImpact.impact_sample_count || 0}` : "分类消息与后验收益",
    },
  ];

  function handleParamChange(key, value) {
    setDraftParams((current) => ({ ...current, [key]: value }));
  }

  function handleApply() {
    setParams({
      ranking_by: draftParams.ranking_by,
      board_size: Number(draftParams.board_size) || 50,
      horizon_days: Number(draftParams.horizon_days) || 3,
      positive_return_pct: Number(draftParams.positive_return_pct) || 10,
      watchlist_text: draftParams.watchlist_text || "",
      security_scope: draftParams.security_scope || "main_board",
    });
  }

  function handleSelectSearchResult(item) {
    setSelectedSymbol(item.symbol);
    setSearchText("");
    setSearchResults([]);
  }

  function handleRefreshMarketBacktest() {
    fetchLatestMarketBacktest(40)
      .then((payload) => {
        setMarketBacktest(payload);
        setMarketBacktestError("");
      })
      .catch((requestError) => {
        setMarketBacktestError(requestError.message || "最新全市场回测暂时不可用");
      });
  }

  function handleRunMarketBacktest(backtestParams) {
    setMarketBacktestError("");
    startMarketBacktest(backtestParams)
      .then((payload) => {
        setMarketBacktestTask(payload);
        if (payload.status === "completed") {
          fetchLatestMarketBacktest(40).then((latestPayload) => {
            setMarketBacktest(latestPayload);
          });
        }
      })
      .catch((requestError) => {
        setMarketBacktestError(requestError.message || "全市场回测启动失败");
      });
  }

  return (
    <div className="app-shell">
      <div className="background-orb background-orb-a" />
      <div className="background-orb background-orb-b" />
      <div className="app-frame">
        <HeroPanel boardPayload={boardPayload} enhancedReady={Boolean(enhancedPayload)} loading={quickLoading || enhancedLoading} />

        <section className="pipeline-strip panel">
          {pipelineSteps.map((step) => (
            <div className={`pipeline-step is-${step.status}`} key={step.label}>
              <span>{step.label}</span>
              <strong>{step.status === "done" ? "完成" : step.status === "running" ? "加载中" : "等待"}</strong>
              <p>{step.note}</p>
            </div>
          ))}
        </section>

        <FilterPanel
          draftParams={draftParams}
          onChange={handleParamChange}
          onApply={handleApply}
          searchText={searchText}
          onSearchChange={setSearchText}
          searchResults={searchResults}
          onSelectSearchResult={handleSelectSearchResult}
        />

        {error ? <div className="error-banner">{error}</div> : null}
        {softNotice ? <div className="notice-banner">{softNotice}</div> : null}
        {strategyNotice ? <div className="notice-banner strategy-notice">{strategyNotice}</div> : null}

        <div className="workspace-grid">
          <main className="workspace-main">
            <BoardTable
              rows={boardRows}
              selectedSymbol={selectedSymbol}
              onSelect={setSelectedSymbol}
              loading={enhancedLoading}
            />
            <details className="advanced-section">
              <summary>实战回测与模型健康度</summary>
              <BattlePanels reviewPanels={reviewPanels} reviewHealth={reviewHealth} />
              <MarketBacktestPanel
                latestMarketDate={latestMarketDate}
                horizonDays={params.horizon_days}
                positiveReturnPct={params.positive_return_pct}
                latestBacktest={marketBacktest}
                taskStatus={marketBacktestTask}
                taskRunning={marketBacktestTask?.status === "running"}
                error={marketBacktestError}
                onRun={handleRunMarketBacktest}
                onRefresh={handleRefreshMarketBacktest}
              />
            </details>
          </main>
          <DetailPanel
            detail={detail}
            loading={detailLoading}
            newsImpact={newsImpact}
            newsImpactLoading={newsImpactLoading}
            newsImpactError={newsImpactError}
          />
        </div>
      </div>
    </div>
  );
}
