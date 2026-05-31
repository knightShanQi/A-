# Execution Score Counterfactual 2026-05-28

## Purpose

Measure whether the current execution-score contribution actually changes or improves replay ranking outcomes, instead of assuming that more execution logic automatically helps.

## Coverage

- Replay rows with both `selection_score` and `execution_score`: 525
- Board dates with comparable scored rows: 13

## Key Findings

- Across the 13 board dates with both `selection_score` and `execution_score`, `selection_score` top-3 and blended `0.62*selection + 0.38*execution` top-3 produced the same next-day return on every date.
- At the portfolio-relevant top-3 cut, `selection_score` and the blended action score both average -0.53% next-day return, which means the execution-score term did not add marginal ranking value in this replay sample.
- `execution_score` used alone is weaker than `selection_score` on the same 13-date sample: top-3 average next-day return -0.26% versus -0.53%.
- At top-10, blending execution into selection is slightly worse than selection alone: 0.42% versus 0.49%.
- The action layer should not be removed wholesale: rows labeled `买` still average 1.28% next-day return on this v9 replay slice.
- High-selection non-buy rows do not look obviously better than the buy rows. That supports keeping discrete execution states as veto/explanation logic, while demoting the continuous execution-score weight from ranking blends.

## Top-N Comparison

| score_col        |   top_n |   days |   avg_next_day_return_pct |   median_next_day_return_pct |   avg_intraday_high_return_pct |   avg_win_rate |
|:-----------------|--------:|-------:|--------------------------:|-----------------------------:|-------------------------------:|---------------:|
| selection_score  |       1 |     13 |                -0.336923  |                     0.49     |                        4.43333 |       0.538462 |
| selection_score  |       3 |     13 |                -0.525897  |                     0.493333 |                        6.39556 |       0.461538 |
| selection_score  |       5 |     13 |                 0.424615  |                     0.832    |                        8.4565  |       0.523077 |
| selection_score  |      10 |     13 |                 0.493615  |                     0.149    |                        7.4134  |       0.492308 |
| blend_62_38      |       1 |     13 |                -0.336923  |                     0.49     |                        4.43333 |       0.538462 |
| blend_62_38      |       3 |     13 |                -0.525897  |                     0.493333 |                        6.39556 |       0.461538 |
| blend_62_38      |       5 |     13 |                 0.424615  |                     0.832    |                        8.4565  |       0.523077 |
| blend_62_38      |      10 |     13 |                 0.417385  |                     0.031    |                        7.4134  |       0.484615 |
| final_rank_score |       1 |     13 |                -0.600769  |                     0.59     |                        1.65    |       0.538462 |
| final_rank_score |       3 |     13 |                -0.311026  |                    -1.2      |                        7.78222 |       0.512821 |
| final_rank_score |       5 |     13 |                -0.520769  |                    -0.53     |                        8.05533 |       0.476923 |
| final_rank_score |      10 |     13 |                 0.0300769 |                     0.467    |                        6.4098  |       0.469231 |
| execution_score  |       1 |     13 |                -0.559231  |                     0.49     |                        1.65    |       0.538462 |
| execution_score  |       3 |     13 |                -0.261026  |                    -1.4      |                        7.78222 |       0.512821 |
| execution_score  |       5 |     13 |                -0.0661538 |                    -0.338    |                        7.942   |       0.523077 |
| execution_score  |      10 |     13 |                 0.184769  |                     0.175    |                        7.63225 |       0.476923 |

## Top-3 Selection vs Blended Action Score

| board_date   |   selection_top3_return_pct |   avg_intraday_high_return_pct |   win_rate | score_col       |   blend_top3_return_pct |   blend_minus_selection_pct |
|:-------------|----------------------------:|-------------------------------:|-----------:|:----------------|------------------------:|----------------------------:|
| 2026-04-24   |                    0.493333 |                     nan        |   0.333333 | selection_score |                0.493333 |                           0 |
| 2026-04-27   |                   -1.4      |                     nan        |   0        | selection_score |               -1.4      |                           0 |
| 2026-05-06   |                    1.23333  |                     nan        |   1        | selection_score |                1.23333  |                           0 |
| 2026-05-07   |                    2.92333  |                      10        |   0.666667 | selection_score |                2.92333  |                           0 |
| 2026-05-08   |                    0.876667 |                     nan        |   0.333333 | selection_score |                0.876667 |                           0 |
| 2026-05-11   |                   -1.89333  |                     nan        |   0.333333 | selection_score |               -1.89333  |                           0 |
| 2026-05-12   |                    9.33667  |                     nan        |   1        | selection_score |                9.33667  |                           0 |
| 2026-05-13   |                    3.19     |                     nan        |   0.333333 | selection_score |                3.19     |                           0 |
| 2026-05-20   |                   -5.87     |                     nan        |   0.333333 | selection_score |               -5.87     |                           0 |
| 2026-05-21   |                   -2.35     |                     nan        |   0.333333 | selection_score |               -2.35     |                           0 |
| 2026-05-22   |                   -6.54333  |                     nan        |   0.333333 | selection_score |               -6.54333  |                           0 |
| 2026-05-25   |                   -9.64333  |                       0.233333 |   0.333333 | selection_score |               -9.64333  |                           0 |
| 2026-05-26   |                    2.81     |                       8.95333  |   0.666667 | selection_score |                2.81     |                           0 |

## Action Label Summary

| action_label   |   rows |   avg_next_day_return_pct |   median_next_day_return_pct |   avg_intraday_high_return_pct |   win_rate |
|:---------------|-------:|--------------------------:|-----------------------------:|-------------------------------:|-----------:|
| 买              |    330 |                  1.2753   |                        0.495 |                        3.48982 |   0.545455 |
| 持              |    118 |                 -0.394322 |                       -0.675 |                        3.37097 |   0.457627 |
| 观察             |     40 |                  0.5305   |                       -0.56  |                        1.52    |   0.4      |
| 卖              |     37 |                  0.91     |                       -0.07  |                      nan       |   0.405405 |

## Action x Execution Label Summary

| action_label   | execution_label   |   rows |   avg_next_day_return_pct |   median_next_day_return_pct |   win_rate |
|:---------------|:------------------|-------:|--------------------------:|-----------------------------:|-----------:|
| 买              | 可执行               |    330 |                  1.2753   |                        0.495 |   0.545455 |
| 卖              | 等待结构              |     18 |                 -0.188333 |                       -0.355 |   0.222222 |
| 卖              | 临门观察              |     15 |                  0.618    |                       -0.4   |   0.466667 |
| 卖              | 可执行               |      4 |                  6.9475   |                        6.76  |   1        |
| 持              | 可执行               |    114 |                 -0.465877 |                       -0.81  |   0.447368 |
| 持              | 临门观察              |      4 |                  1.645    |                        1.235 |   0.75     |
| 观察             | 可执行               |     32 |                  0.128125 |                       -0.73  |   0.375    |
| 观察             | 临门观察              |      8 |                  2.14     |                        0.425 |   0.5      |

## High-Selection Non-Buy Rows

| action_label   | execution_label   |   rows |   avg_next_day_return_pct |   median_next_day_return_pct |   win_rate |
|:---------------|:------------------|-------:|--------------------------:|-----------------------------:|-----------:|
| 持              | 可执行               |     82 |                 -0.122927 |                       -0.665 |   0.463415 |
| 持              | 临门观察              |      1 |                 -0.31     |                       -0.31  |   0        |
| 观察             | 可执行               |      8 |                 -2.05625  |                       -2.86  |   0.25     |

## Interpretation

- The current `action_score = 0.62*selection_score + 0.38*execution_score` blend is not earning its complexity on this replay slice.
- `execution_score` appears saturated in strong candidates, so it often fails to reshuffle the top of the book even when it materially complicates the decision stack.
- The stronger nuance is: demote the continuous `execution_score` weight first, but keep discrete execution states like `买` / `持` / `观察` and `execution_label` under audit because they still separate behavior better than the raw score alone.

## Next Actions

1. Remove the `execution_score` term from default ranking blends in research comparisons and re-check top-3 / top-10 post-cost results.
2. Keep `execution_label` and `execution_window` as discrete explanation fields; they may still be useful as veto diagnostics.
3. Re-run the same comparison after more v9 replay history exists, to confirm this is not a short-window artifact.

Summary CSV: `E:\openclaw\.cache\trading_system_attribution\execution_score_counterfactual_summary.csv`
Top-3 comparison CSV: `E:\openclaw\.cache\trading_system_attribution\execution_score_counterfactual_top3.csv`
Action summary CSV: `E:\openclaw\.cache\trading_system_attribution\execution_score_counterfactual_action_summary.csv`
Action x execution CSV: `E:\openclaw\.cache\trading_system_attribution\execution_score_counterfactual_action_execution_summary.csv`
High-selection non-buy CSV: `E:\openclaw\.cache\trading_system_attribution\execution_score_counterfactual_high_selection_non_buy.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\execution_score_counterfactual.json`