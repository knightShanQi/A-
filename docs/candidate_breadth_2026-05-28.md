# Candidate Breadth Report 2026-05-28

## Purpose

Measure how widening the daily selected basket changes real unified-portfolio performance on the current v9 snapshot slice.

## Coverage

- Snapshot rows: 527
- Cached history rows: 186540
- Tested widths: 1, 3, 5, 10, 15, 20

## Key Findings

- On the current v9 snapshot slice, top-3 is the highest annualized-return width at 1.54%, with ending equity 1459955.
- Top-3 remains the cleanest concentrated basket: annualized 1.54%, max drawdown -3.54%, average selected score 97.43.
- Widening from top-3 to top-10 only slightly lowers annualized return (1.39%) but more than triples drawdown magnitude (-11.83%) as average selection score drops from 97.43 to 95.98.
- Beyond top-10 the pool is clearly diluted: the most drawdown-efficient width is top-3 at -3.54%, while top-15 and top-20 both lose annualized return versus top-3.

## Summary

|   top_n |   candidate_rows |   avg_selection_score |   median_selection_score |   buy_ratio |   watch_ratio |   ending_equity |   cumulative_return |   annualized_return |   max_drawdown |   trade_count |   win_rate |   avg_net_return |
|--------:|-----------------:|----------------------:|-------------------------:|------------:|--------------:|----------------:|--------------------:|--------------------:|---------------:|--------------:|-----------:|-----------------:|
|       1 |               13 |               97.6973 |                 100      |           0 |             0 |     1.40646e+06 |            0.406461 |          0.0138852  |     -0.0376319 |             4 |   0.75     |        0.0948482 |
|       3 |               39 |               97.4325 |                 100      |           0 |             0 |     1.45995e+06 |            0.459955 |          0.0154165  |     -0.0353602 |            11 |   0.727273 |        0.0686472 |
|       5 |               65 |               96.8094 |                 100      |           0 |             0 |     1.31637e+06 |            0.316368 |          0.0111752  |     -0.0997461 |            18 |   0.666667 |        0.0467598 |
|      10 |              130 |               95.9766 |                  99.1315 |           0 |             0 |     1.40784e+06 |            0.407842 |          0.0139255  |     -0.118269  |            41 |   0.804878 |        0.0708143 |
|      15 |              186 |               95.1914 |                  99.1118 |           0 |             0 |     1.25037e+06 |            0.250371 |          0.00907461 |     -0.103009  |            57 |   0.701754 |        0.0456007 |
|      20 |              241 |               94.5022 |                  99.0686 |           0 |             0 |     1.19183e+06 |            0.191828 |          0.00712022 |     -0.0772071 |            70 |   0.685714 |        0.0381605 |

## Interpretation

- The current short replay-backed portfolio slice does not support a broad default basket. The signal is still alpha-dense at the top and degrades as lower-ranked names are added.
- Top-10 has slightly more capacity than top-3 on this slice, but that extra breadth comes with materially deeper drawdown and lower average score quality.
- If broader baskets are needed later, the system likely needs stronger ranking calibration first rather than a larger default top-N.

## Next Actions

1. Keep top-3 as the default research concentration baseline while deeper ranking fixes are being tested.
2. Treat top-10 as a capacity experiment only after ranking quality improves under the same unified portfolio engine.
3. Use this breadth table together with the execution-off evidence to focus the next optimization on candidate-family and ranking quality, not more overlay complexity.

Summary CSV: `E:\openclaw\.cache\trading_system_attribution\candidate_breadth_summary.csv`
Daily mix CSV: `E:\openclaw\.cache\trading_system_attribution\candidate_breadth_daily_mix.csv`
NAV CSV: `E:\openclaw\.cache\trading_system_attribution\candidate_breadth_nav.csv`
Trades CSV: `E:\openclaw\.cache\trading_system_attribution\candidate_breadth_trades.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\candidate_breadth.json`