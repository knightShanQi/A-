# Execution Weight Sweep 2026-05-28

## Purpose

Test whether changing the continuous execution-score weight inside the current action blend actually changes replay ranking outcomes.

## Coverage

- Replay rows with both `selection_score` and `execution_score`: 525
- Board dates with comparable v9 scored rows: 13

## Key Findings

- On the current v9 replay slice, execution weights from 0.00 through 0.50 produce the exact same top-3 picks on all 13 comparable board dates.
- That means the current `0.38` execution weight is not merely too large or too small; it is functionally inactive as a ranking differentiator in this sample.
- Top-3 average next-day return stays flat at -0.53% for every blend from 0.00 to 0.50 execution weight.
- Only the extreme `execution_weight = 1.00` regime reshuffles the top-3 set, and it changes picks on 6 of 13 days.

## Weight Sweep Summary

|   execution_weight |   selection_weight |   top_n |   days |   avg_next_day_return_pct |   median_next_day_return_pct |   avg_intraday_high_return_pct |   avg_win_rate |
|-------------------:|-------------------:|--------:|-------:|--------------------------:|-----------------------------:|-------------------------------:|---------------:|
|               0    |               1    |       1 |     13 |                -0.336923  |                     0.49     |                        4.43333 |       0.538462 |
|               0    |               1    |       3 |     13 |                -0.525897  |                     0.493333 |                        6.39556 |       0.461538 |
|               0    |               1    |       5 |     13 |                 0.424615  |                     0.832    |                        8.4565  |       0.523077 |
|               0    |               1    |      10 |     13 |                 0.493615  |                     0.149    |                        7.4134  |       0.492308 |
|               0.1  |               0.9  |       1 |     13 |                -0.336923  |                     0.49     |                        4.43333 |       0.538462 |
|               0.1  |               0.9  |       3 |     13 |                -0.525897  |                     0.493333 |                        6.39556 |       0.461538 |
|               0.1  |               0.9  |       5 |     13 |                 0.424615  |                     0.832    |                        8.4565  |       0.523077 |
|               0.1  |               0.9  |      10 |     13 |                 0.417385  |                     0.031    |                        7.4134  |       0.484615 |
|               0.22 |               0.78 |       1 |     13 |                -0.336923  |                     0.49     |                        4.43333 |       0.538462 |
|               0.22 |               0.78 |       3 |     13 |                -0.525897  |                     0.493333 |                        6.39556 |       0.461538 |
|               0.22 |               0.78 |       5 |     13 |                 0.424615  |                     0.832    |                        8.4565  |       0.523077 |
|               0.22 |               0.78 |      10 |     13 |                 0.417385  |                     0.031    |                        7.4134  |       0.484615 |
|               0.38 |               0.62 |       1 |     13 |                -0.336923  |                     0.49     |                        4.43333 |       0.538462 |
|               0.38 |               0.62 |       3 |     13 |                -0.525897  |                     0.493333 |                        6.39556 |       0.461538 |
|               0.38 |               0.62 |       5 |     13 |                 0.424615  |                     0.832    |                        8.4565  |       0.523077 |
|               0.38 |               0.62 |      10 |     13 |                 0.417385  |                     0.031    |                        7.4134  |       0.484615 |
|               0.5  |               0.5  |       1 |     13 |                -0.336923  |                     0.49     |                        4.43333 |       0.538462 |
|               0.5  |               0.5  |       3 |     13 |                -0.525897  |                     0.493333 |                        6.39556 |       0.461538 |
|               0.5  |               0.5  |       5 |     13 |                 0.424615  |                     0.832    |                        8.4565  |       0.523077 |
|               0.5  |               0.5  |      10 |     13 |                 0.417385  |                     0.031    |                        7.4134  |       0.484615 |
|               1    |               0    |       1 |     13 |                -0.559231  |                     0.49     |                        1.65    |       0.538462 |
|               1    |               0    |       3 |     13 |                -0.261026  |                    -1.4      |                        7.78222 |       0.512821 |
|               1    |               0    |       5 |     13 |                -0.0661538 |                    -0.338    |                        7.942   |       0.523077 |
|               1    |               0    |      10 |     13 |                 0.184769  |                     0.175    |                        7.63225 |       0.476923 |

## Top-3 Stability vs Execution-Off Baseline

|   execution_weight |   selection_weight |   same_top3_days |   total_days |   avg_variant_minus_baseline_pct |
|-------------------:|-------------------:|-----------------:|-------------:|---------------------------------:|
|               0.1  |               0.9  |               13 |           13 |                         0        |
|               0.22 |               0.78 |               13 |           13 |                         0        |
|               0.38 |               0.62 |               13 |           13 |                         0        |
|               0.5  |               0.5  |               13 |           13 |                         0        |
|               1    |               0    |                7 |           13 |                         0.264872 |

## Interpretation

- The current ranking stack is saturated enough that partial execution weighting does not move the top of the book at all on this slice.
- This is stronger evidence than a single counterfactual: it says there is no practical ranking leverage anywhere between 0% and 50% execution weight here.
- The right next change is to set the continuous execution weight to zero in research comparisons first, because doing so loses nothing on this replay slice while simplifying the stack.

## Next Actions

1. Add an execution-off branch to the research ranking path and rerun unified portfolio backtests.
2. Keep discrete execution labels/windows for veto and explanation while the continuous weight is being audited.
3. Re-run this sweep after replay history expands beyond the current v9 slice.

Summary CSV: `E:\openclaw\.cache\trading_system_attribution\execution_weight_sweep_summary.csv`
Top-3 compare CSV: `E:\openclaw\.cache\trading_system_attribution\execution_weight_sweep_top3_compare.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\execution_weight_sweep.json`