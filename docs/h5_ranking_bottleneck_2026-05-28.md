# h5 Ranking Bottleneck 2026-05-28

## Purpose

Identify the concrete code-path and cache-state reasons why supported-path `h5 / 3%` regeneration is still too slow to complete inside normal CLI windows.

## Key Findings

- Target supported config remains `h5 / 3%`.
- Latest snapshot-history store date is `20260527`.
- Same-date feature store exists: `False`.
- Same-date candidate-pool store exists: `False`.
- Same-date dynamic-fallback store exists: `False`.
- Current-date h5 candidate-analysis cache exists: `False`.
- Full-market analysis worker count is hard-pinned to `1`.
- Cold-path minimum serial candidate analyses implied by current pool sizing is `120` symbols.
- Stage-isolation evidence now sharpens the bottleneck:
  - `scripts/regenerate_supported_default_artifacts.py --force-refresh --store-stage snapshot` completes in about 40 seconds and rewrites `market_snapshot_history_store_v1_20260527.pkl`
  - `scripts/regenerate_supported_default_artifacts.py --force-refresh --store-stage features` still times out after 10 minutes and does not produce `market_daily_feature_store_v4_20260527.pkl`
  - `scripts/regenerate_supported_default_artifacts.py --store-stage features` also times out after 10 minutes even when it reuses the warmed snapshot-history store and does not rewrite that snapshot file
  - code now explicitly reuses snapshot-history when live snapshot context is empty, so the `features` timeout is no longer explainable by an avoidable fallback to per-symbol daily-history fetch alone
  - this isolates the current cold-start blocker to feature-store generation itself, before candidate-pool construction or per-symbol candidate-analysis even start

## Interpretation

- current snapshot history exists, but no same-date feature store has been written yet
- no same-date strategy candidate pool store exists for the target date
- no current-date h5 candidate-analysis cache files exist, so per-symbol analysis starts cold
- full-market candidate analysis is hard-pinned to one worker
- snapshot-history regeneration is not the dominant blocker anymore; the heavier bottleneck begins at feature-store generation

## Practical Consequence

- The current `h5` migration blocker is no longer just “the command times out”.
- It is structurally a cold rebuild problem centered on feature-store generation: snapshot history can be refreshed quickly, but same-date feature-store output still does not materialize inside a 10-minute stage-isolated window.
- Downstream cold costs still matter after that, because candidate-pool and candidate-analysis caches are also absent, but the first hard wall is now clearly before those stages.
- That means annualized-return optimization work should continue to treat supported-path recovery as an engineering throughput problem before it is a scorer-formula problem.

## Suggested Next Optimization Order

1. Reduce current-date feature-store cold-start cost before touching score formulas again.
2. Keep the staged recovery order explicit: `snapshot` -> `features` -> `pools` -> `rankings`.
3. Add a resumable or batched warmup path for current-date h5 candidate-analysis caches after feature-store generation is healthy.
4. Only after h5 ranking artifacts can be regenerated on demand should default-path migration be considered operationally ready.

JSON: `E:\openclaw\.cache\trading_system_attribution\h5_ranking_bottleneck.json`
