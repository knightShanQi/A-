# 大A关注雷达

这是一个围绕你那份“炒股学习计划”搭起来的本地实时页面项目。它把你学习里最核心的几件事，直接变成了可刷新、可观察、可筛选的页面：

- 用 `K 线阶段` 判断股票处在什么位置
- 用 `分时 + 均价线` 判断当天有没有被资金认真对待
- 用 `量价结构 + 简单概率模型 + 量化辅助信号` 给出当日关注优先级
- 结合 `最新个股新闻`、`板块资金流热度`、`个股主力资金流`
- 用 `日K + 1分钟` 两张图把“为什么值得看”展示出来

## 页面里有什么

- 今日关注榜：自动从大A活跃股票里筛出值得盯的标的
- 任意A股搜索：可直接按代码或名称搜索全市场股票，不受关注榜限制
- 个股详情：展示日K、均线、成交量、1分钟分时和均价线
- 市场热度总览：行业资金、概念资金、宏观日历
- 消息面与资金面：个股新闻情绪、主力资金强弱、行业热度
- 学习映射：把“博弈型 / 逼空型 / 确认型”这些阶段逻辑直接翻译成观察点和证伪条件
- 自动刷新：开盘后可按秒级间隔刷新页面

## 技术栈

- 数据源：DuckDB / Supabase PostgreSQL / AkShare
- 页面：Streamlit
- 图表：Plotly
- 预测：scikit-learn Logistic Regression

## 快速启动

```powershell
cd E:\openclaw
C:\Users\ASUS\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e .
.\scripts\start_dashboard.ps1
```

启动后打开：

- [http://localhost:8501](http://localhost:8501)

如果你想手动指定命令，建议使用项目虚拟环境的 Python 而不是 `streamlit.exe`：
```powershell
.\.venv\Scripts\python.exe scripts\run_streamlit_app.py
```

## 页面使用建议

- 北京时间 `09:30-11:30`、`13:00-15:00` 打开，分时信息最有意义
- 非交易时段页面仍可用，但会展示最新可得交易日数据
- 默认从活跃股票里筛选，你也可以在侧边栏输入自定义股票池

## 与你学习内容的对应关系

项目把分享对话里的知识落成了 4 个程序层：

1. `阶段识别`
   把“二次进攻尝试 / 趋势加速 / 突破确认”做成规则引擎。
2. `分时观察`
   用均价线、回踩、量能节奏，判断当天是强承接还是纯噪音。
3. `交易计划结构`
   页面会给每只票输出关注理由、观察点、证伪条件。
4. `复盘与筛选`
   用注意力分数和上涨概率把“今天先看谁”排出来。
5. `消息和资金联合过滤`
   用个股新闻、行业资金流和主力资金流做二次确认。

## 风险说明

这个项目是分析与观察工具，不是收益承诺，也不是确定性荐股系统。A 股数据接口偶尔会出现延迟、限流或字段变化，页面里已经做了异常兜底，但仍建议结合自己的交易纪律使用。

## Manual full-market backtest

Use this entry when you want a full-market replay outside the Streamlit page. The backtest uses the original project data path for historical structure and only treats Tushare as a non-adjusted cross-section supplement.

You can run it in three ways:

- Streamlit page: open `http://localhost:8501`, then use `Manual Backtest / 手动全市场数据回测`.
- API task: `POST http://127.0.0.1:8000/api/tasks/market-backtest?date_from=2026-04-01&date_to=2026-04-21&horizon_days=3&positive_return_pct=10&strategy_mode=all&top_k=50`.
- CLI command:

```powershell
cd E:\openclaw
.\.venv\Scripts\python.exe -m a_share_predictor.market_backtest_runner --date-from 2026-04-01 --date-to 2026-04-21 --horizon-days 3 --positive-return 0.10 --strategy-mode all --top-k 50
```

Outputs are written to:

- `.cache\market_full_backtests\summary.json`
- `.cache\market_full_backtests\trade_like_results.csv`

The summary includes average holding returns for 1, 3, and 5 trading days: `avg_hold_1d_return`, `avg_hold_3d_return`, and `avg_hold_5d_return`.

## Sync daily stock files

The daily stock sync imports CSV/TXT/TSV/Excel/Parquet/JSON files and normalizes common Chinese and English column names.

Use DuckDB as the local strategy/model data source:

```powershell
cd E:\openclaw
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\build_duckdb_from_baidu_cache.py --replace --duckdb-path data\openclaw_market_data.duckdb --cache-dir .cache\baidu_daily_stock --years 2000-2026
```

Keep these values in `.env.local`:

```env
OPENCLAW_MARKET_DATA_SOURCE=duckdb
OPENCLAW_DAILY_PRICE_STORAGE=row
OPENCLAW_DAILY_PRICE_TABLE=a_share_daily_prices
OPENCLAW_DUCKDB_PATH=E:\openclaw\data\openclaw_market_data.duckdb
OPENCLAW_TRADE_CALENDAR_TABLE=a_share_trade_calendar
OPENCLAW_INTRADAY_BARS_TABLE=a_share_intraday_bars
OPENCLAW_INTRADAY_RETENTION_DAYS=365
```

For unattended DuckDB daily refreshes:

```powershell
.\scripts\install_duckdb_daily_stock_sync_task.ps1 -DailyAt "17:20"
```

The task runs `scripts\run_duckdb_daily_stock_sync.ps1`, downloads/parses the latest Baidu Pan files when credentials are present, upserts daily rows into `a_share_daily_prices`, upserts 1/5/15/30/60 minute bars into `a_share_intraday_bars`, keeps only the latest `OPENCLAW_INTRADAY_RETENTION_DAYS` calendar days of intraday bars, and writes logs under `.cache\sync_logs`.

Supabase/PostgreSQL remains available as a remote store. The PostgreSQL sync upserts rows by `(symbol, trade_date)`:

```powershell
cd E:\openclaw
.\.venv\Scripts\python.exe -m pip install -e .
$env:DATABASE_URL="postgresql://postgres.vrddzvdrbzffynmacbua:<真实密码>@aws-1-ap-northeast-2.pooler.supabase.com:6543/postgres?sslmode=require"
$env:BAIDU_PAN_COOKIE="<浏览器里复制的百度网盘 Cookie>"
.\.venv\Scripts\python.exe scripts\sync_daily_stock_data.py
```

If the Baidu Pan web API requires manual login or verification, download the files into a local folder and run the same import path without the network step:

```powershell
.\.venv\Scripts\python.exe scripts\sync_daily_stock_data.py --skip-download --input-dir E:\daily_stock_files
```

For the local DuckDB store, use:

```powershell
.\.venv\Scripts\python.exe scripts\sync_duckdb_daily_stock_data.py --skip-download --input-dir E:\daily_stock_files
```

To import only already-downloaded intraday bars:

```powershell
.\.venv\Scripts\python.exe scripts\sync_duckdb_intraday_stock_data.py --input-dir .cache\baidu_daily_stock\20260528 --duckdb-path data\openclaw_market_data.duckdb
```

Default storage is row mode with target table `a_share_daily_prices`. For full Baidu Pan history on small Supabase projects, use compact series mode (`OPENCLAW_DAILY_PRICE_STORAGE=series`), which stores one row per stock and year in `a_share_daily_price_series`.

For unattended PostgreSQL daily runs, create `E:\openclaw\.env.local` from `.env.example`, then install the Windows scheduled task:

```powershell
.\scripts\install_daily_stock_sync_task.ps1 -DailyAt "17:00"
```

The task runs `scripts\run_daily_stock_sync.ps1` and writes logs under `.cache\sync_logs`.

To use Supabase/PostgreSQL as the strategy and model data source, keep these values in `.env.local`:

```env
OPENCLAW_MARKET_DATA_SOURCE=supabase
OPENCLAW_DAILY_PRICE_STORAGE=series
OPENCLAW_DAILY_PRICE_TABLE=a_share_daily_price_series
DATABASE_URL=postgresql://postgres.vrddzvdrbzffynmacbua:<真实密码>@aws-1-ap-northeast-2.pooler.supabase.com:6543/postgres?sslmode=require
```

When DuckDB or PostgreSQL fully covers the requested date range, `fetch_daily_history`, market snapshots, and trade-date queries read from that database. DuckDB is preferred when `OPENCLAW_MARKET_DATA_SOURCE=duckdb`; otherwise the project can fall back to Supabase/PostgreSQL or the original providers.
