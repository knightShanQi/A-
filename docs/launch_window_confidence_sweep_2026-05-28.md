# Launch Window Confidence Sweep 2026-05-28

## Purpose

Measure the marginal portfolio value of the `launch_window_confidence` layer inside `selection_score` by sweeping its explicit weight around the current `0.04` setting.

## Coverage

- Snapshot rows: 527
- Cached history rows: 186540

## Key Findings

- Current weight `0.04` yields about 1.54% annualized with -3.54% max drawdown on the current v9 slice.
- Zeroing the layer (`0.00`) yields about 1.55% annualized with -3.37% drawdown and 2.08 / 3 average top-3 overlap.
- The best tested weight on this slice is `0.00` at 1.55% annualized.
- This sweep tests the marginal ranking value of the launch-window-confidence layer rather than the broader launch context family.

## Summary

|   launch_window_confidence_weight |   candidate_rows |   avg_top3_overlap_vs_baseline |   exact_top3_match_days |   annualized_return |   max_drawdown |   ending_equity |   trade_count |   win_rate |   avg_net_return |
|----------------------------------:|-----------------:|-------------------------------:|------------------------:|--------------------:|---------------:|----------------:|--------------:|-----------:|-----------------:|
|                              0    |               39 |                        2.07692 |                       6 |          0.0155339  |     -0.03365   |     1.46413e+06 |             9 |   0.666667 |        0.057223  |
|                              0.01 |               39 |                        2.07692 |                       6 |          0.0155339  |     -0.03365   |     1.46413e+06 |             9 |   0.666667 |        0.057223  |
|                              0.02 |               39 |                        2.07692 |                       6 |          0.0155339  |     -0.03365   |     1.46413e+06 |             9 |   0.666667 |        0.057223  |
|                              0.03 |               39 |                        2.07692 |                       6 |          0.0155339  |     -0.03365   |     1.46413e+06 |             9 |   0.666667 |        0.057223  |
|                              0.04 |               39 |                        3       |                      13 |          0.0154165  |     -0.0353602 |     1.45995e+06 |            11 |   0.727273 |        0.0686472 |
|                              0.06 |               39 |                        2       |                       6 |          0.00974345 |     -0.146749  |     1.27103e+06 |            10 |   0.7      |        0.0454902 |
|                              0.08 |               39 |                        2       |                       6 |          0.00974345 |     -0.146749  |     1.27103e+06 |            10 |   0.7      |        0.0454902 |

## Interpretation

- If weights from `0.00` through the current setting are flat or improve as they decrease, the layer is effectively dead complexity and should be demoted or removed.
- If a smaller positive weight wins but zero is materially worse, the right recommendation is weight reduction rather than full removal.

## Next Actions

1. If zero or near-zero is best, remove this layer from the research formula first.
2. If only a smaller weight wins, demote the current `0.04` toward that smaller value and retest on future v10 artifacts.

Summary CSV: `E:\openclaw\.cache\trading_system_attribution\launch_window_confidence_sweep_summary.csv`
NAV CSV: `E:\openclaw\.cache\trading_system_attribution\launch_window_confidence_sweep_nav.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\launch_window_confidence_sweep.json`