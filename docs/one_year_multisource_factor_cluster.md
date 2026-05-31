# One-Year Multi-Source Factor Clustering

This run extends the next-day A-share factor search with intraday, call-auction,
sector, fund-flow, and broad-market candidates over the latest one-year window.
It is designed as a factor-discovery pass, not a final production classifier.

## Data Window

- DuckDB: `data/openclaw_market_data.duckdb`
- Daily table: `a_share_daily_prices`
- Intraday table: `a_share_intraday_bars`
- Analysis window: `2025-05-28` to `2026-05-28`
- Daily rows in window: `1,313,402`
- Daily trade days: `243`
- Eligible non-ST symbols: `4,967`
- Excluded ST symbols: `179`
- Rows seen by the clustering pipeline: `1,188,138`
- Sample rows used for ranking: `500,000`
- Candidate features: `127`

## Source Coverage

The database now contains real minute bars, including volume and amount:

| Interval | Rows | Symbols | Trade Days | Date Range |
| ---: | ---: | ---: | ---: | --- |
| 1 minute | 2,653,410 | 5,507 | 2 | `2026-05-26` to `2026-05-28` |
| 5 minutes | 528,480 | 5,507 | 2 | `2026-05-26` to `2026-05-28` |
| 15 minutes | 21,326,681 | 5,791 | 243 | `2025-05-28` to `2026-05-28` |
| 30 minutes | 88,080 | 5,507 | 2 | `2026-05-26` to `2026-05-28` |
| 60 minutes | 38,716 | 5,507 | 2 | `2026-05-26` to `2026-05-28` |

The latest full run uses the supplemented 15-minute intraday source. Its
selected-interval trade-day coverage is `1.0`, and the merged true intraday
sample share is `0.999852`, above the `0.05` coverage gate. True intraday
columns were therefore admitted into the final one-year feature ranking.
One-minute coverage remains only `0.00823045267489712`, so the 1-minute columns
should still be treated as incomplete until the larger historical archive is
imported.

The local DuckDB now also contains a true per-stock main-fund-flow table:

- Fund-flow table: `a_share_stock_fund_flow`
- Rows in the one-year window: `151,168`
- Covered symbols: `5,215`
- Covered trade days: `91`
- Date range: `2025-05-28` to `2026-05-28`
- Sample coverage in the clustering run: `0.117698`

True fund-flow columns passed the `0.05` sample-coverage gate and were included
in the candidate set. They did not enter the final top 10 in this run; the best
true fund-flow candidate was `true_fund_main_inflow_streak_5`.

True standalone call-auction tables are still not present in the local DuckDB.
The current run therefore uses explicit opening-pressure proxies:

- Call auction/opening pressure: `auction_open_gap` plus first available
  intraday open-bar features. In the full-year 15-minute run the selected
  interval starts at `09:45:00`, so these are opening-segment proxies rather
  than a standalone 09:25 call-auction table.
- Fund flow: true stock-level main-fund-flow columns plus amount, turnover, and
  signed price-volume confirmation proxies.
- Sector heat: industry and board-segment cross-sectional return, turnover, and
  amount-ratio ranks from daily constituents.
- Broad market: all-A-share momentum, turnover, activity, and risk-appetite
  proxies.

The source audit is stored in
`.cache/one_year_multisource_factor_cluster_15min_full_year/data_coverage.json`.
The latest run with true fund-flow candidates is stored in
`.cache/one_year_multisource_factor_cluster_15min_fundflow_v2/data_coverage.json`.

## Selected Top 10 Factors

| Rank | Factor | Interpretation |
| ---: | --- | --- |
| 1 | `intraday_close_strength` | Same-day close location inside the intraday range. |
| 2 | `close_vs_ma20` | Position of close versus the 20-day trend baseline. |
| 3 | `segment_turnover_5` | Board-level heat and rotation liquidity. |
| 4 | `ma_alignment_score` | Multi-horizon moving-average trend alignment. |
| 5 | `market_turnover_20` | Broad-market turnover regime over 20 sessions. |
| 6 | `intraday_tail30_ret` | Late-session intraday return and closing pressure. |
| 7 | `intraday_low_time_ratio` | When the intraday low occurs, capturing early flush versus late weakness. |
| 8 | `limit_up_flag` | Current-day limit-up state and next-day constraint behavior. |
| 9 | `market_risk_appetite_5` | Broad participation versus recent downside pressure. |
| 10 | `close_near_high_5` | Short-term close strength near recent highs. |

These factors were selected by a composite ranking that combines ExtraTrees
importance, mutual information with the T+1 label, standardized logistic
coefficient magnitude, and market-state cluster lift. Correlated candidates are
clustered so the selected list is not just several near-duplicates of the same
signal.

## Artifacts

- `.cache/one_year_multisource_factor_cluster_15min_fundflow_v2/feature_ranking.csv`
- `.cache/one_year_multisource_factor_cluster_15min_fundflow_v2/feature_cluster_summary.csv`
- `.cache/one_year_multisource_factor_cluster_15min_fundflow_v2/market_state_cluster_summary.csv`
- `.cache/one_year_multisource_factor_cluster_15min_fundflow_v2/selected_top10_factors.csv`
- `.cache/one_year_multisource_factor_cluster_15min_fundflow_v2/data_coverage.json`
- `.cache/one_year_multisource_factor_cluster_15min_fundflow_v2/cluster_report.md`

## Reproduce

From `E:\openclaw`:

```powershell
.\.venv\Scripts\python.exe scripts\run_one_year_multisource_cluster.py `
  --output-dir .cache\one_year_multisource_factor_cluster_15min_fundflow_v2 `
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

The 2026 15-minute source was supplemented from the Baidu Pan intraday archive with:

```powershell
.\.venv\Scripts\python.exe scripts\download_baidu_intraday_stock_data.py `
  --years 2026 `
  --intervals 15 `
  --download-dir .cache\baidu_intraday_stock\2026 `
  --extract
```

Then it was imported into DuckDB with:

```powershell
.\.venv\Scripts\python.exe scripts\sync_duckdb_intraday_stock_data.py `
  --input-dir .cache\baidu_intraday_stock\2026 `
  --intervals 15
```

The 2025 portion of the one-year window came from the legacy 2000-2025 RAR
archive:

```powershell
.\.venv\Scripts\python.exe scripts\download_baidu_intraday_stock_data.py `
  --years 2025 `
  --intervals 15 `
  --include-legacy-rar `
  --download-dir .cache\baidu_intraday_stock\legacy_2000_2025 `
  --extract
```

Only the needed 2025 date range was imported:

```powershell
.\.venv\Scripts\python.exe scripts\sync_duckdb_intraday_stock_data.py `
  --input-dir .cache\baidu_intraday_stock\legacy_2000_2025 `
  --intervals 15 `
  --start-date 2025-05-28 `
  --end-date 2025-12-31 `
  --retention-days off
```

## Next Data Gap

The remaining material gaps are true standalone call-auction tables, deeper
historical true fund-flow coverage, and full one-minute depth. The 15-minute
intraday source now covers the entire one-year clustering window, and true
main-fund-flow data is present with audited partial coverage.
