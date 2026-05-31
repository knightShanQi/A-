import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles.css";

function renderFatalError(error) {
  const root = document.getElementById("root");
  if (!root) {
    return;
  }
  root.innerHTML = `
    <div style="max-width: 960px; margin: 48px auto; padding: 24px; border-radius: 24px; background: rgba(255,255,255,.92); font-family: sans-serif; color: #122033; box-shadow: 0 24px 80px rgba(15,23,42,.12);">
      <p style="margin: 0 0 8px; color: #f65753; font-weight: 700;">前端启动失败</p>
      <h1 style="margin: 0 0 12px; font-size: 28px;">A股信号台没有成功渲染</h1>
      <pre style="white-space: pre-wrap; color: #63758a;">${String(error?.stack || error?.message || error)}</pre>
    </div>
  `;
}

window.addEventListener("error", (event) => {
  renderFatalError(event.error || event.message);
});

window.addEventListener("unhandledrejection", (event) => {
  renderFatalError(event.reason);
});

try {
  const root = document.getElementById("root");
  if (!root) {
    throw new Error("缺少 #root 挂载节点");
  }

  ReactDOM.createRoot(root).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  );
} catch (error) {
  renderFatalError(error);
}
