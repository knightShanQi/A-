# Selection Score v10 Comparison 2026-05-28

## Purpose

Compare the old `v9` baseline artifacts against the new `v10` zero-launch-window-confidence research slice for both source observability and family-compression conclusions.

## Key Findings

- `v10` zero-weight artifacts now persist 8 / 8 previously missing source terms, up from 0 in `v9`.
- The `selection_score` source projection stays high but becomes slightly more grounded under `v10`: `v9` R² 95.61% vs `v10` R² 96.84%.
- The active `launch_window_confidence` weight is now explicit in artifacts: `v9` behaves like 0.04, while the generated `v10` research slice is 0.00.
- On the current replayable slice, baseline `selection_score_top3` improves from 1.54% annualized in `v9` to 1.54% in `v10`, with drawdown improving from -3.54% to -3.54%.
- The best family-compression move remains aligned with the zero-weight decision: `v9` best variant is `selection_minus_launch_confidence`, while `v10` no longer needs that step because the active weight is already zero.

## Version Summary

|   version |   review_rows |   review_files |   snapshot_rows |   source_projection_r2 | source_top_correlation_field   |   source_top_correlation_value |   launch_confidence_correlation |   persisted_missing_source_terms |   active_launch_window_confidence_weight |   baseline_annualized_return |   baseline_max_drawdown | best_variant                      |   best_variant_annualized_return |   best_variant_max_drawdown |
|----------:|--------------:|---------------:|----------------:|-----------------------:|:-------------------------------|-------------------------------:|--------------------------------:|---------------------------------:|-----------------------------------------:|-----------------------------:|------------------------:|:----------------------------------|---------------------------------:|----------------------------:|
|         9 |           525 |             13 |             527 |               0.956144 | attention_score                |                       0.92586  |                        0.774379 |                                0 |                                     0.04 |                    0.0154165 |              -0.0353602 | selection_minus_launch_confidence |                        0.0155339 |                  -0.03365   |
|        10 |           525 |             13 |             527 |               0.968355 | attention_score                |                       0.927459 |                        0.778637 |                                8 |                                     0    |                    0.0154165 |              -0.0353602 | selection_minus_launch_confidence |                        0.0154165 |                  -0.0353602 |

## Interpretation

- `v10` matters because it converts an architecture recommendation into persisted evidence, not just a local formula tweak.
- If the baseline portfolio metrics improve while the active launch-window-confidence weight drops to zero, that is stronger evidence than any isolated sweep because the result survives the real artifact-generation path.
- Once more `v10` history exists, the next highest-value simplification target should move from launch-window-confidence to lighter attention-family compression.

## Next Actions

1. Treat `launch_window_confidence_weight = 0.0` as the new default research setting for follow-up score audits.
2. Re-run attention-family compression reports on larger `v10` history as it accumulates.

Version CSV: `E:\openclaw\.cache\trading_system_attribution\selection_score_v10_comparison_summary.csv`
Coverage CSV: `E:\openclaw\.cache\trading_system_attribution\selection_score_v10_comparison_coverage.csv`
Correlation CSV: `E:\openclaw\.cache\trading_system_attribution\selection_score_v10_comparison_correlations.csv`
Family CSV: `E:\openclaw\.cache\trading_system_attribution\selection_score_v10_comparison_family.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\selection_score_v10_comparison.json`