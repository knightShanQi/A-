# Feature Store Throughput 2026-05-28

## Purpose

Estimate the cold-path cost of current-date feature-store generation on the supported `h5 / 3%` path, using real snapshot-history data and sampled per-symbol feature construction.

## Key Findings

- Target market date: `2026-05-27`.
- Snapshot-history rows: `751253`.
- Unique snapshot-history symbols: `5531`.
- Sampled symbols: `24`.
- Average sampled per-symbol feature-build time: `0.0444` seconds.
- Estimated full pass over current snapshot-history symbols: `4.1` minutes.

## Interpretation

- This estimate is only for sampled per-symbol feature construction on already-loaded snapshot history. It does not include every surrounding cold-start cost.
- If the estimated full-pass time is already large, the current feature-store timeout is structurally plausible even before candidate-pool or ranking work begins.

JSON: `E:\openclaw\.cache\trading_system_attribution\feature_store_throughput_20260527.json`
