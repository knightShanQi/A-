# One-Year Multi-Source Factor Clustering

This run extends the next-day A-share factor search with intraday bars,
opening-pressure/call-auction proxies, sector heat, stock-level fund flow, and
broad-market context over the latest one-year window. It is a factor-discovery
pass for the T+1 close-to-close direction label, not a final production trading
system.

## Data Window

- DuckDB: `data/openclaw_market_data.duckdb`
- Daily table: `a_share_daily_prices`
- Intraday table: `a_share_intraday_bars`
- Fund-flow table: `a_share_stock_fund_flow`
- Auction proxy table: `a_share_call_auction_proxy`
- Analysis window: `2025-05-28` to `2026-05-28`
- Daily rows in window: `1,313,402`
- Daily trade days: `243`
- Eligible non-ST symbols: `4,967`
- Excluded ST symbols: `179`
- Rows seen by the clustering pipeline: `1,188,138`
- Sample rows used for ranking: `500,000`
- Candidate features: `127`

## Source Coverage

The database contains real intraday bars with `open`, `high`, `low`, `close`,
`volume`, and `amount`:

| Interval | Rows | Symbols | Trade Days | Date Range |
| ---: | ---: | ---: | ---: | --- |
| 1 minute | 2,653,410 | 5,507 | 2 | `2026-05-26` to `2026-05-28` |
| 5 minutes | 528,480 | 5,507 | 2 | `2026-05-26` to `2026-05-28` |
| 15 minutes | 21,326,681 | 5,791 | 243 | `2025-05-28` to `2026-05-28` |
| 30 minutes | 88,080 | 5,507 | 2 | `2026-05-26` to `2026-05-28` |
| 60 minutes | 38,716 | 5,507 | 2 | `2026-05-26` to `2026-05-28` |

The full run uses the supplemented 15-minute source because it covers all 243
trade days in the one-year window. Its merged true-intraday sample share is
`0.999852`, above the `0.05` coverage gate, so true intraday price, volume, and
amount-derived columns are admitted into the final feature ranking. One-minute
coverage is only `0.00823045267489712` of daily trade days, so one-minute
history is still incomplete for a one-year model.

Standalone historical 09:25 call-auction data is still not present in DuckDB.
The supplemented table `a_share_call_auction_proxy` is therefore explicitly
treated as a derived proxy, not as true auction tick/order-book data:

- Rows: `1,312,653`
- Symbols: `5,547`
- Trade days: `243`
- Date range: `2025-05-28` to `2026-05-28`
- First available selected-interval bar: `09:45:00`
- Mode: `derived_daily_open_plus_first_intraday_bar`

The proxy combines daily open/pre-close information with first available
intraday bar return, volume share, amount share, and range. It captures
opening pressure, but it does not replace real 09:25 auction data.

True stock-level main-fund-flow data is present with partial one-year coverage:

- Rows: `151,168`
- Covered symbols: `5,215`
- Covered trade days: `91`
- Date range: `2025-05-28` to `2026-05-28`
- Sample coverage in the clustering run: `0.117698`

True fund-flow columns pass the `0.05` sample-coverage gate and are included in
the candidate set. They do not enter the final top 10 in this run.

Sector and broad-market context are derived cross-sectionally from daily
constituents:

- Sector heat: industry return, up-ratio, turnover, and amount-ratio ranks.
- Segment heat: market-board return, turnover, and liquidity ranks.
- Broad market: all-A-share momentum, turnover, activity, risk appetite, and
  market-state cluster behavior.
- Trading constraints: ST exclusion, trade-gap/resumption flags, and limit-up
  flags are included.

## Selected Top 10 Factors

| Rank | Factor | Interpretation |
| ---: | --- | --- |
| 1 | `intraday_close_strength` | Same-day close location inside the intraday range. |
| 2 | `trade_gap_days` | Suspension/resumption gap pressure and stale-price effect. |
| 3 | `recent_resume_flag` | Recent resumption state after interrupted trading. |
| 4 | `limit_up_flag` | Current-day limit-up state and next-day constraint behavior. |
| 5 | `intraday_low_time_ratio` | Timing of the intraday low, separating early flush from late weakness. |
| 6 | `close_vs_ma20` | Position of close versus the 20-day trend baseline. |
| 7 | `ma_alignment_score` | Multi-horizon moving-average trend alignment. |
| 8 | `intraday_tail30_ret` | Late-session intraday return and closing pressure. |
| 9 | `market_turnover_20` | Broad-market turnover regime over 20 sessions. |
| 10 | `segment_turnover_5` | Board-level liquidity heat and rotation. |

These factors were selected by a composite ranking that combines ExtraTrees
importance, mutual information with the T+1 label, standardized logistic
coefficient magnitude, and market-state cluster lift. Correlated candidates are
clustered so the selected list is not only near-duplicates of the same signal.

## Artifacts

- `.cache/one_year_multisource_factor_cluster_15min_fundflow_auction_proxy/feature_ranking.csv`
- `.cache/one_year_multisource_factor_cluster_15min_fundflow_auction_proxy/feature_cluster_summary.csv`
- `.cache/one_year_multisource_factor_cluster_15min_fundflow_auction_proxy/market_state_cluster_summary.csv`
- `.cache/one_year_multisource_factor_cluster_15min_fundflow_auction_proxy/selected_top10_factors.csv`
- `.cache/one_year_multisource_factor_cluster_15min_fundflow_auction_proxy/data_coverage.json`
- `.cache/one_year_multisource_factor_cluster_15min_fundflow_auction_proxy/cluster_report.md`

## Reproduce

From `E:\openclaw`, build the opening-pressure proxy table:

```powershell
.\.venv\Scripts\python.exe scripts\build_duckdb_call_auction_proxy.py `
  --start-date 2025-05-28 `
  --end-date 2026-05-28 `
  --interval-minutes 15
```

Then run the one-year clustering pass:

```powershell
.\.venv\Scripts\python.exe scripts\run_one_year_multisource_cluster.py `
  --output-dir .cache\one_year_multisource_factor_cluster_15min_fundflow_auction_proxy `
  --sample-limit 500000 `
  --importance-sample-limit 220000 `
  --batch-symbols 240 `
  --intraday-interval-minutes 15
```

True stock-level main-fund-flow data was supplemented from the stockpage
10jqka source and written into DuckDB with:

```powershell
.\.venv\Scripts\python.exe scripts\sync_duckdb_stock_fund_flow.py `
  --start-date 2025-05-28 `
  --end-date 2026-05-28 `
  --sleep-seconds 0.02
```

The 2026 15-minute source was supplemented from the Baidu Pan intraday archive
and imported into DuckDB:

```powershell
.\.venv\Scripts\python.exe scripts\download_baidu_intraday_stock_data.py `
  --years 2026 `
  --intervals 15 `
  --download-dir .cache\baidu_intraday_stock\2026 `
  --extract

.\.venv\Scripts\python.exe scripts\sync_duckdb_intraday_stock_data.py `
  --input-dir .cache\baidu_intraday_stock\2026 `
  --intervals 15
```

The 2025 part of the window came from the legacy 2000-2025 RAR archive:

```powershell
.\.venv\Scripts\python.exe scripts\download_baidu_intraday_stock_data.py `
  --years 2025 `
  --intervals 15 `
  --include-legacy-rar `
  --download-dir .cache\baidu_intraday_stock\legacy_2000_2025 `
  --extract

.\.venv\Scripts\python.exe scripts\sync_duckdb_intraday_stock_data.py `
  --input-dir .cache\baidu_intraday_stock\legacy_2000_2025 `
  --intervals 15 `
  --start-date 2025-05-28 `
  --end-date 2025-12-31 `
  --retention-days off
```

## Remaining Data Gaps

The remaining material gaps are true standalone historical 09:25 call-auction
data, deeper historical true fund-flow coverage, and full one-minute depth.
The current clustering result does fully use the available one-year 15-minute
intraday source, the derived opening-pressure proxy table, partial true
stock-level fund flow, sector heat, market trend, ST filtering, resumption
flags, and limit-up constraints.
