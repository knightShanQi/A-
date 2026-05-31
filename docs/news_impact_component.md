# A-share News Impact Component

This component lives in `a_share_predictor.news_impact`.

## Purpose

It turns raw A-share news and announcements into a structured event stream, then checks how each event category mapped to following price moves.

The component is intentionally usable in three modes:

- As a Python module for model features and research notebooks.
- As a FastAPI endpoint for the web terminal.
- As a CLI command for one-off symbol analysis.

## Data Flow

1. Fetch symbol news with `fetch_stock_news()` and, when enabled, CNInfo disclosures through AkShare.
2. Normalize provider-specific columns to:
   `symbol`, `title`, `content`, `published_at`, `source`, `url`, `keyword`.
3. Classify every item into an event category:
   `earnings`, `contract_order`, `shareholder_action`, `financing_mna`,
   `policy_sector`, `regulatory_risk`, `product_technology`,
   `market_opinion`, `accident_risk`, or `general`.
4. Score direction, confidence, expected impact, session bucket, and likely impact horizon.
5. Align every event to the effective trading day:
   intraday/pre-open news maps to the same trading day when available;
   after-close news maps to the next trading day.
6. Measure open gap and 1/3/5-day forward returns against the previous close.
7. Aggregate category-level hit rate, average return, open gap, and rank score.

## Public Functions

- `fetch_symbol_news_events(symbol, limit=80, include_disclosures=True)`
- `fetch_market_news_events(limit=80)`
- `normalize_news_frame(news_df, symbol=None)`
- `classify_news_events(news_df, symbol=None)`
- `build_event_impact_dataset(news_events, daily_prices, horizons=(1, 3, 5))`
- `summarize_category_impact(impact_df, horizons=(1, 3, 5))`
- `build_latest_news_impact_signal(news_events, window_days=7)`
- `analyze_symbol_news_impact(symbol, ...)`

## API

```text
GET /api/symbol/{symbol}/news-impact
```

Query parameters:

- `start_date`: optional `YYYYMMDD`
- `end_date`: optional `YYYYMMDD`
- `news_limit`: default `120`, max `300`
- `horizons`: comma-separated trading-day horizons, default `1,3,5`
- `include_disclosures`: default `true`

The response includes:

- `latest_signal`
- `category_summary`
- `event_impacts`
- `events`

## CLI

```powershell
.\.venv\Scripts\python.exe -m a_share_predictor.news_impact 000001 --horizons 1,3,5 --limit 120
```

After installing the package in editable mode, the script entry point is:

```powershell
a-share-news-impact 000001 --horizons 1,3,5 --limit 120
```

Use `--output-dir .cache\news_impact\000001` to export events, event impacts, and category summaries as CSV files.

## Interpretation Notes

The output is evidence for research and triage, not a deterministic trading signal. A strong category hit rate is meaningful only when sample size, source diversity, and current price structure agree. The existing model should treat these fields as auxiliary variables beside price structure, fund flow, sector heat, and market regime.
