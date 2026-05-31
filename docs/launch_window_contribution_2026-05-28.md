# Launch Window Contribution Report 2026-05-28

## Purpose

Measure whether launch-window information helps more as a ranker, as a gate, or as a small tilt on top of selection_score.

## Coverage

- Snapshot rows: 527
- Cached history rows: 186540

## Key Findings

- `selection_score_top3` baseline stays strongest at 1.54% annualized with -3.54% drawdown.
- The best pure launch-window gate is `selection_exclude_non_launch` at 1.54% annualized.
- `launch_window_only_top3` is much weaker at 0.62% annualized, which argues against using launch-window as the main ranker.
- `selection_plus_launch_tilt` lands at 0.50% annualized, so overweighting launch terms on top of selection does not beat the current baseline on this slice.

## Portfolio Summary

| variant                      | score_col                  |   candidate_rows |   avg_launch_window_score | status_mix               |   initial_capital |   ending_equity |   cumulative_return |   annualized_return |   max_drawdown |   trade_count |   win_rate |   avg_net_return |
|:-----------------------------|:---------------------------|-----------------:|--------------------------:|:-------------------------|------------------:|----------------:|--------------------:|--------------------:|---------------:|--------------:|-----------:|-----------------:|
| baseline_selection_top3      | selection_score            |               39 |                   89.3208 | {"强势延续": 32, "启动观察窗": 7} |             1e+06 |     1.45995e+06 |            0.459955 |          0.0154165  |     -0.0353602 |            11 |   0.727273 |        0.0686472 |
| selection_exclude_non_launch | selection_score            |               39 |                   89.3208 | {"强势延续": 32, "启动观察窗": 7} |             1e+06 |     1.45995e+06 |            0.459955 |          0.0154165  |     -0.0353602 |            11 |   0.727273 |        0.0686472 |
| selection_launch_ge_62       | selection_score            |               39 |                   89.3208 | {"强势延续": 32, "启动观察窗": 7} |             1e+06 |     1.45995e+06 |            0.459955 |          0.0154165  |     -0.0353602 |            11 |   0.727273 |        0.0686472 |
| selection_strong_trend_only  | selection_score            |               39 |                   88.011  | {"强势延续": 39}             |             1e+06 |     1.44143e+06 |            0.441427 |          0.0148924  |     -0.0434995 |            13 |   0.692308 |        0.0655301 |
| selection_plus_launch_tilt   | selection_plus_launch_tilt |               39 |                   87.2072 | {"强势延续": 34, "启动观察窗": 5} |             1e+06 |     1.1317e+06  |            0.131705 |          0.00501473 |     -0.0680299 |            12 |   0.666667 |        0.0467369 |
| launch_window_only_top3      | launch_window_score        |               39 |                   89.8956 | {"强势延续": 32, "启动观察窗": 7} |             1e+06 |     1.16596e+06 |            0.165955 |          0.00622696 |     -0.2158    |            11 |   0.636364 |        0.0338038 |

## Interpretation

- If a launch-only ranker is much weaker than the baseline, launch-window is probably better understood as a context/gating input than as the primary alpha source.
- If a modest launch tilt also fails to beat the baseline, then the current live selection_score may already be using roughly the right amount of launch information on this slice.
- If a launch gate mainly reduces exposure but does not improve annualized return, that is useful as a risk-control clue but not as a full replacement ranking logic.

## Next Actions

1. Keep launch-window inputs inside selection_score, but treat them primarily as a structure/risk-control family rather than a standalone ranker.
2. If one gate variant materially improves drawdown without killing return, test it later on a longer history before adopting it.
3. Continue source-level auditing around tomorrow-confidence after launch-window, because those two families look more important than quant on the current persisted slice.

Summary CSV: `E:\openclaw\.cache\trading_system_attribution\launch_window_contribution_summary.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\launch_window_contribution.json`