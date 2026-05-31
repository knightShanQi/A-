# Default Config Migration 2026-05-28

## Purpose

Decide whether the unsupported default `h3 / 10%` path should be restored, retired, or migrated, using both model-source availability and current artifact freshness.

## Key Findings

- The current default `h3 / 10%` path has the freshest review stack in the workspace: 13 snapshot files and 13 review files spanning 2026-04-24 to 2026-05-26.
- The only proxy-backed alternative, `h5 / 3%`, is materially stale in the review stack: 7 snapshots and 4 reviews, with the latest review ending 2026-04-16.
- `h3 / 10%` therefore has operational continuity but no model support, while `h5 / 3%` has proxy-model support but not a current review/ranking artifact chain.
- That means the practical default decision is two-stage, not one-step: retire `h3 / 10%` as the strategic default, but do not blindly switch users to `h5 / 3%` until its current ranking/review pipeline is regenerated.
- The evidence-backed migration order is: first regenerate fresh `h5 / 3%` ranking/review artifacts under the current cache schema, then switch the default, then resume score-layer optimization on the supported path.
- Operational bottleneck is now directly confirmed, not inferred:
  - a real `--force-refresh` full regeneration attempt on 2026-05-28 ran for 15 minutes without producing any new `h5` review/snapshot artifacts
  - a follow-up `--force-refresh --rankings-only` run also timed out after 10 minutes without producing any new `h5` ranking/review artifacts
  - this isolates the current supported-path bottleneck to the ranking rebuild stage itself, before review maintenance becomes relevant
- The new bottleneck audit sharpens that further:
  - current-date snapshot history is already present for `2026-05-27`
  - but there is still no same-date feature store, candidate-pool store, dynamic-fallback store, or current-date `h5` candidate-analysis cache
  - the rebuild therefore starts from a cold store layer and then enters a single-worker per-symbol analysis path
- Stage-isolated recovery evidence now narrows the first hard blocker:
  - `--store-stage snapshot` completes in about 40 seconds for `2026-05-27`
  - `--store-stage features` still times out after 10 minutes and does not write `market_daily_feature_store_v4_20260527.pkl`
  - the same `--store-stage features` timeout reproduces even without `--force-refresh`, so the blocker is not snapshot-history rebuild reuse; it is feature-store generation itself
  - so the first operational wall is feature-store generation, not snapshot-history generation

## Summary

| config                   |   snapshot_files |   review_files | first_snapshot_date   | last_snapshot_date   | first_review_pair            | last_review_pair             | ranking_cache_present   |
|:-------------------------|-----------------:|---------------:|:----------------------|:---------------------|:-----------------------------|:-----------------------------|:------------------------|
| h3_r1000_current_default |               13 |             13 | 2026-04-24            | 2026-05-26           | ('2026-04-24', '2026-04-27') | ('2026-05-26', '2026-05-27') | True                    |
| h5_r300_proxy_reference  |                7 |              4 | 2026-04-13            | 2026-04-13           | ('2026-04-09', '2026-04-10') | ('2026-04-15', '2026-04-16') | False                   |

## Recommendation

- Strategic recommendation: retire `h3 / 10%` as the long-lived default because it is unsupported by model artifacts.
- Operational recommendation: do not switch the default immediately to `h5 / 3%` until you regenerate a fresh ranking/review chain for that configuration under the current schema.
- Practical migration sequence: rebuild `h5 / 3%` caches and review history, validate its proxy-backed path, then flip the default and continue formula simplification there.

## Next Actions

1. Use [`E:\openclaw\scripts\regenerate_supported_default_artifacts.py`](E:\openclaw\scripts\regenerate_supported_default_artifacts.py) as the single-entry regeneration command for the supported `h5 / 3%` path; it now supports `--force-refresh`, which bypasses the non-stale ranking cache, rewrites the market ranking artifact, builds the focus board, and runs daily-review maintenance under one configuration.
2. Expect the full-market run to take materially longer than a short CLI smoke test. Local execution evidence now includes:
   - an earlier 180-second timeout on the monolithic regeneration command
   - a later 15-minute timeout on `--force-refresh`
   - a later 10-minute timeout even on `--force-refresh --rankings-only`
   This means the current `h5` migration blocker is ranking rebuild latency, not just review maintenance.
3. Use `--rankings-only` first when diagnosing or scheduling supported-path recovery; it now separates ranking rebuild from board/review maintenance and makes the bottleneck explicit.
4. Use the staged store recovery path when recovering `h5`:
   - `--store-stage snapshot` first
   - then `--store-stage features`
   - then `--store-stage pools`
   - only after those succeed should `--rankings-only` or full regeneration be attempted
5. Only after fresh `h5 / 3%` ranking/review artifacts exist should the UI default move away from `h3 / 10%`.

Summary CSV: `E:\openclaw\.cache\trading_system_attribution\default_config_migration_summary.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\default_config_migration.json`
