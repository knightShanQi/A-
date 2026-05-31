# v10 Attention Coverage Audit 2026-05-28

## Purpose

Audit whether the current replayable `v10` artifact slice actually represents the full attention-enrichment architecture, or whether it is dominated by quick-board/backfilled defaults that can distort redundancy conclusions.

## Key Findings

- All `527` persisted `v10` snapshot rows carry `sector_score = fund_score = news_score = 50`, so the current audit slice does not expose real sector/fund/news dispersion inside the attention family.
- `enhanced_attention_score` still differs from `attention_score` on 210 rows, but those differences come from quick-board style secondary adjustments rather than the full `_score_final_attention()` enrichment path.
- `candidate_reason` is present on only 0 rows and `launch_phase_label` on only 0, which confirms the replayable slice is missing much of the richer narrative context available in live symbol analysis.
- The persisted `model_source_label` mix is dominated by lightweight board-generation paths: `最新收盘快榜（完整版特征与回测正在后台补齐）` on 427 rows and `本地快速回退` on 100 rows.
- Practical implication: the current attention-family redundancy findings are real for the backfilled focus-board path, but they likely understate the value of the full enriched path because the enriched source fields are still collapsed to defaults in these artifacts.

## Summary

|   snapshot_rows |   snapshot_files |   all_three_default_50_rows |   attention_delta_zero_rows |   attention_delta_nonzero_rows |   candidate_reason_present_rows |   launch_phase_present_rows |   fund_label_placeholder_rows |   news_label_placeholder_rows |   sector_label_placeholder_rows |   local_quick_fallback_rows |   latest_close_fast_board_rows | ranking_by_counts   | candidate_strategy_counts                                                                  |
|----------------:|-----------------:|----------------------------:|----------------------------:|-------------------------------:|--------------------------------:|----------------------------:|------------------------------:|------------------------------:|--------------------------------:|----------------------------:|-------------------------------:|:--------------------|:-------------------------------------------------------------------------------------------|
|             527 |               13 |                         527 |                         317 |                            210 |                               0 |                           0 |                             0 |                             0 |                               0 |                         100 |                            427 | {'关注分数': 527}       | {'missing': 200, 'dynamic_fallback': 200, '策略1': 56, '策略2': 41, 'strategy3': 29, '策略3': 1} |

## Strategy Breakdown

| candidate_strategy   |   rows |   attention_delta_zero_rate |   all_three_default_50_rate |   avg_attention_delta |
|:---------------------|-------:|----------------------------:|----------------------------:|----------------------:|
| dynamic_fallback     |    200 |                    0.75     |                           1 |            -1.54135   |
| missing              |    200 |                    0.6      |                           1 |            -0.0479    |
| 策略1                  |     56 |                    0.553571 |                           1 |            -0.0244643 |
| 策略2                  |     41 |                    0.365854 |                           1 |            -0.144146  |
| strategy3            |     29 |                    0        |                           1 |            -0.0210345 |
| 策略3                  |      1 |                    1        |                           1 |             0         |

## Interpretation

- This report is not saying the earlier redundancy audits are false. It is saying their scope is the current backfilled focus-board path, not the full live enriched-analysis path.
- If the persisted slice carries default `50/50/50` sector-fund-news values everywhere, then the incremental value of `_score_final_attention()` cannot be fully observed from these artifacts alone.
- That means any production-facing deletion of `enhanced_attention_score` should wait for a new review/snapshot history generated directly from the full enriched board path rather than reconstructed quick-board snapshots.

## Next Actions

1. Generate a fresh non-backfilled review/snapshot history from the live enriched board path and rerun the attention-family audits on that slice.
2. Until then, treat current attention-family simplification results as valid for the quick-board recovery path, but incomplete for the richer symbol-analysis architecture.

Summary CSV: `E:\openclaw\.cache\trading_system_attribution\v10_attention_coverage_summary.csv`
Strategy CSV: `E:\openclaw\.cache\trading_system_attribution\v10_attention_coverage_by_strategy.csv`
Sample CSV: `E:\openclaw\.cache\trading_system_attribution\v10_attention_coverage_nonzero_samples.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\v10_attention_coverage_audit.json`
