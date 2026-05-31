# Selection Score Family Compression Report 2026-05-28

## Purpose

Test which explicit attention/launch/tomorrow restatement layers inside `selection_score` can be removed first with the least portfolio damage on the current replayable slice.

## Coverage

- Snapshot rows: 527
- Cached history rows: 186540

## Key Findings

- Baseline `selection_score_top3` remains the reference at 1.54% annualized with -3.54% max drawdown.
- The strongest compression variant on the current v9 slice is `selection_minus_launch_confidence` at 1.55% annualized with -3.37% drawdown.
- The safest structural simplification by overlap is `selection_minus_enhanced_attention_layer`, which still keeps 2.23 / 3 baseline names on average.
- This experiment is a ranking-key counterfactual, not a full formula rewrite: it asks which explicit restated layer can be removed first with the least damage on the current replayable slice.

## Summary

| score_col                                |   candidate_rows |   avg_top3_overlap_vs_baseline |   exact_top3_match_days |   annualized_return |   max_drawdown |   ending_equity |   trade_count |   win_rate |   avg_net_return |
|:-----------------------------------------|-----------------:|-------------------------------:|------------------------:|--------------------:|---------------:|----------------:|--------------:|-----------:|-----------------:|
| selection_minus_launch_confidence        |               39 |                        2.07692 |                       6 |          0.0155339  |     -0.03365   |     1.46413e+06 |             9 |   0.666667 |        0.057223  |
| selection_score                          |               39 |                        3       |                      13 |          0.0154165  |     -0.0353602 |     1.45995e+06 |            11 |   0.727273 |        0.0686472 |
| selection_minus_launch_family            |               39 |                        1.92308 |                       6 |          0.0152273  |     -0.0782483 |     1.45324e+06 |             9 |   0.555556 |        0.0502407 |
| selection_minus_enhanced_attention_layer |               39 |                        2.23077 |                       8 |          0.0131149  |     -0.0798746 |     1.38027e+06 |            10 |   0.6      |        0.0596126 |
| selection_minus_attention_layer          |               39 |                        2.23077 |                       8 |          0.0130842  |     -0.0798746 |     1.37923e+06 |             9 |   0.555556 |        0.0435601 |
| selection_minus_tomorrow_confidence      |               39 |                        1.84615 |                       6 |          0.00563435 |     -0.0830169 |     1.14909e+06 |            10 |   0.5      |        0.0223093 |

## Interpretation

- If a subtraction variant stays close to or above the baseline, that layer is a good simplification candidate because it is behaving more like restatement than unique alpha.
- If removing a layer immediately collapses annualized return or top-3 overlap, that layer is still doing real ranking work on this slice even if it is conceptually redundant.
- Because these variants start from the persisted baseline score and subtract one explicit formula layer, they are directly useful for sequencing architecture cleanup.

## Next Actions

1. Demote the least-damaging subtraction layer first in the real formula or in a research-only branch.
2. After v10 artifacts accumulate, rerun the same report with the newly persisted source terms to confirm the simplification survives better evidence.

Summary CSV: `E:\openclaw\.cache\trading_system_attribution\selection_score_family_compression_summary.csv`
NAV CSV: `E:\openclaw\.cache\trading_system_attribution\selection_score_family_compression_nav.csv`
Trades CSV: `E:\openclaw\.cache\trading_system_attribution\selection_score_family_compression_trades.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\selection_score_family_compression.json`