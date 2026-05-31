# v10 Attention Family Compression 2026-05-28

## Purpose

Test how much of the stacked `probability_up + attention_score + enhanced_attention_score` cluster can be compressed now that `launch_window_confidence_weight` is already zeroed in `v10` artifacts.

## Coverage

- Snapshot rows: 527
- Cached history rows: 186540
- Active launch-window-confidence weight in artifacts: 0.00

## Key Findings

- Baseline `selection_score_top3` on the current `v10` slice stays at 1.54% annualized with -3.54% max drawdown.
- No attention-family subtraction beats the baseline; the least-damaging non-baseline variant is `selection_minus_enhanced_attention_layer` at 1.31% annualized with -7.99% drawdown.
- Removing the explicit `probability_up` layer hurts more than removing either attention layer: annualized return drops by 0.63% and average top-3 overlap falls to 1.77 / 3.
- `attention_score` and `enhanced_attention_score` behave like one compressed cluster on this slice: their subtraction variants both stay near 2.15 / 3 overlap, and removing `enhanced_attention_score` is only marginally less harmful than removing `attention_score`.
- Because `launch_window_confidence_weight` is already 0.00 in `v10`, any remaining compression result here is about the attention family itself, not a hidden launch-window confounder.

## Summary

| score_col                                |   candidate_rows |   avg_top3_overlap_vs_baseline |   exact_top3_match_days |   initial_capital |   ending_equity |   cumulative_return |   annualized_return |   max_drawdown |   trade_count |   win_rate |   avg_net_return |
|:-----------------------------------------|-----------------:|-------------------------------:|------------------------:|------------------:|----------------:|--------------------:|--------------------:|---------------:|--------------:|-----------:|-----------------:|
| selection_score                          |               39 |                        3       |                      13 |             1e+06 |     1.45995e+06 |            0.459955 |          0.0154165  |     -0.0353602 |            11 |   0.727273 |        0.0686472 |
| selection_minus_enhanced_attention_layer |               39 |                        2.15385 |                       8 |             1e+06 |     1.38027e+06 |            0.380266 |          0.0131149  |     -0.0798746 |            10 |   0.6      |        0.0596126 |
| selection_half_attention_cluster         |               39 |                        2.15385 |                       8 |             1e+06 |     1.38027e+06 |            0.380266 |          0.0131149  |     -0.0798746 |            10 |   0.6      |        0.0596126 |
| selection_minus_attention_cluster        |               39 |                        2.15385 |                       8 |             1e+06 |     1.38027e+06 |            0.380266 |          0.0131149  |     -0.0798746 |            10 |   0.6      |        0.0596126 |
| selection_minus_attention_layer          |               39 |                        2.15385 |                       8 |             1e+06 |     1.37923e+06 |            0.379232 |          0.0130842  |     -0.0798746 |             9 |   0.555556 |        0.0435601 |
| selection_minus_probability_layer        |               39 |                        1.76923 |                       6 |             1e+06 |     1.25305e+06 |            0.253052 |          0.00916199 |     -0.0793218 |             9 |   0.555556 |        0.034017  |
| selection_minus_prob_attention_cluster   |               39 |                        1.69231 |                       6 |             1e+06 |     1.25305e+06 |            0.253052 |          0.00916199 |     -0.0793218 |             9 |   0.555556 |        0.034017  |
| selection_half_prob_attention_cluster    |               39 |                        1.76923 |                       6 |             1e+06 |     1.25305e+06 |            0.253052 |          0.00916199 |     -0.0793218 |             9 |   0.555556 |        0.034017  |

## Interpretation

- If a half-weight or subtraction variant stays close to baseline, that specific layer is behaving more like restatement than unique alpha.
- If annualized return and overlap both break quickly, that layer is still doing real ranking work even if the family is conceptually redundant.
- This experiment is still a ranking-key counterfactual on persisted `v10` artifacts, so it is suitable for sequencing the next research-side simplification.

## Next Actions

1. Do not zero or halve the whole attention family in research defaults yet; the current `selection_score` baseline still dominates every tested subtraction.
2. If attention-family cleanup continues, prefer source-level consolidation around one attention representation instead of blunt subtraction from the persisted score.
3. Treat `probability_up` as the stickier core signal inside this cluster and avoid demoting it before a stronger replacement exists.

Summary CSV: `E:\openclaw\.cache\trading_system_attribution\v10_attention_family_compression_summary.csv`
NAV CSV: `E:\openclaw\.cache\trading_system_attribution\v10_attention_family_compression_nav.csv`
Trades CSV: `E:\openclaw\.cache\trading_system_attribution\v10_attention_family_compression_trades.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\v10_attention_family_compression.json`
