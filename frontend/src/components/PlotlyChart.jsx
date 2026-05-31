import { useEffect, useRef, useState } from "react";

let plotlyLoader;

function loadPlotly() {
  if (!plotlyLoader) {
    plotlyLoader = import("plotly.js-dist-min");
  }
  return plotlyLoader;
}

export default function PlotlyChart({ figure, className = "" }) {
  const containerRef = useRef(null);
  const [status, setStatus] = useState("idle");

  useEffect(() => {
    const container = containerRef.current;
    if (!container || !figure) {
      setStatus("empty");
      return undefined;
    }

    let canceled = false;
    let plotly = null;
    setStatus("loading");

    loadPlotly()
      .then((module) => {
        if (canceled || !containerRef.current) {
          return;
        }
        plotly = module.default || module;
        return plotly.react(
          containerRef.current,
          figure.data || [],
          figure.layout || {},
          {
            displayModeBar: false,
            responsive: true,
          },
        );
      })
      .then(() => {
        if (!canceled) {
          setStatus("ready");
        }
      })
      .catch(() => {
        if (!canceled) {
          setStatus("error");
        }
      });

    return () => {
      canceled = true;
      if (plotly && container) {
        plotly.purge(container);
      }
    };
  }, [figure]);

  return (
    <div ref={containerRef} className={`plot-shell ${className}`.trim()}>
      {status === "empty" ? <span className="chart-loading">图表数据暂未加载，榜单评分已先展示</span> : null}
      {status === "loading" ? <span className="chart-loading">图表加载中...</span> : null}
      {status === "error" ? <span className="chart-loading">图表加载失败，请刷新重试</span> : null}
    </div>
  );
}
