# Selection Score Ablation 2026-05-28

## Purpose

Approximate the persisted selection-score ingredients and test leave-one-out proxy variants under the unified portfolio engine.

## Coverage

- Snapshot rows: 527
- Cached history rows: 186540

## Key Findings

- `selection_score_top3` remains the top benchmark at 1.54% annualized.
- The strongest persisted-field proxy variant is `selection_proxy_no_quant` at 1.01% annualized.
- The weakest leave-one-out variant is `selection_proxy_no_launch` at 0.46% annualized.
- If several simplified proxies cluster below selection_score, that is evidence the current selection_score is already capturing useful interactions even if some ingredients may still be noisy.

## Portfolio Summary

| score_col                    |   candidate_rows |   avg_score |   median_score |   identical_days_vs_selection |   avg_overlap_count_vs_selection |   initial_capital |   ending_equity |   cumulative_return |   annualized_return |   max_drawdown |   trade_count |   win_rate |   avg_net_return |
|:-----------------------------|-----------------:|------------:|---------------:|------------------------------:|---------------------------------:|------------------:|----------------:|--------------------:|--------------------:|---------------:|--------------:|-----------:|-----------------:|
| selection_score              |               39 |     97.4325 |       100      |                            13 |                          3       |             1e+06 |     1.45995e+06 |            0.459955 |          0.0154165  |     -0.0353602 |            11 |   0.727273 |        0.0686472 |
| selection_proxy_full         |               39 |     80.9796 |        82.8963 |                             5 |                          1.76923 |             1e+06 |     1.26243e+06 |            0.262429 |          0.00946622 |     -0.148028  |             9 |   0.666667 |        0.0278685 |
| selection_proxy_no_launch    |               39 |     86.0954 |        90.4313 |                             5 |                          1.61538 |             1e+06 |     1.11979e+06 |            0.119793 |          0.0045849  |     -0.121387  |             9 |   0.666667 |        0.0386945 |
| selection_proxy_no_quant     |               39 |     82.7458 |        84.007  |                             5 |                          1.84615 |             1e+06 |     1.28272e+06 |            0.282717 |          0.0101171  |     -0.120184  |             9 |   0.555556 |        0.015423  |
| selection_proxy_no_prob      |               39 |     85.394  |        88.8762 |                             5 |                          1.76923 |             1e+06 |     1.26243e+06 |            0.262429 |          0.00946622 |     -0.148028  |             9 |   0.666667 |        0.0278685 |
| selection_proxy_no_attention |               39 |     73.3652 |        73.5397 |                             5 |                          1.76923 |             1e+06 |     1.27103e+06 |            0.271032 |          0.00974345 |     -0.146749  |            10 |   0.7      |        0.0454902 |
| selection_proxy_no_tomorrow  |               39 |     83.0232 |        84.7259 |                             2 |                          1.69231 |             1e+06 |     1.1581e+06  |            0.158103 |          0.00595207 |     -0.219613  |            10 |   0.6      |        0.0167756 |

## Interpretation

- These are proxy ablations on persisted fields, not exact reproductions of every live selection-score ingredient.
- They are still useful for direction: if a simplified proxy loses meaningfully to selection_score, then the live score is likely capturing interactions that matter.
- If a leave-one-out proxy collapses especially hard, that ingredient family is a good candidate for deeper source-level audit.

## Next Actions

1. Keep `selection_score_top3` as the default research baseline unless an audited alternative clearly beats it.
2. Use the weakest leave-one-out directions to choose the next source-level ingredient audit in dashboard selection logic.
3. Avoid replacing selection_score with a simplified proxy just because it is easier to explain; the engine evidence has to stay primary.

Summary CSV: `E:\openclaw\.cache\trading_system_attribution\selection_score_ablation_summary.csv`
Overlap CSV: `E:\openclaw\.cache\trading_system_attribution\selection_score_ablation_overlap.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\selection_score_ablation.json`