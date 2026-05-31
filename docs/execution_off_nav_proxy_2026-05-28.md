# Execution-Off NAV Proxy 2026-05-28

## Purpose

Approximate a daily rebalanced top-N equity curve on the current v9 replay slice, to see whether removing the continuous execution-score weight changes short-horizon portfolio behavior.

## Coverage

- Replay rows with both `selection_score` and `execution_score`: 525
- Board dates with comparable v9 scored rows: 13

## Key Findings

- On the v9 replay proxy NAV, every top-3 variant is still negative over the 13-day slice. That reinforces the broader audit conclusion that the current recommendation stack lacks enough short-horizon alpha.
- At top-3, execution-off `selection_score` and the current `0.62/0.38` blend are exactly identical: -8.01% cumulative return proxy.
- At top-10, execution-off is better than the current blend on the same replay slice: 6.27% versus 5.22% cumulative return proxy.
- The top-10 proxy also keeps the same max-drawdown proxy after removing the execution weight, which means the blend is not buying visible downside protection on this slice.

## Summary

| score_col        |   top_n |   days |   ending_equity |   cumulative_return |   annualized_return_proxy |   max_drawdown_proxy |   avg_daily_return |
|:-----------------|--------:|-------:|----------------:|--------------------:|--------------------------:|---------------------:|-------------------:|
| selection_score  |       3 |     13 |        0.919921 |          -0.0800793 |                 -0.801703 |           -0.223805  |        -0.00525897 |
| blend_62_38      |       3 |     13 |        0.919921 |          -0.0800793 |                 -0.801703 |           -0.223805  |        -0.00525897 |
| final_rank_score |       3 |     13 |        0.92666  |          -0.0733396 |                 -0.771562 |           -0.287563  |        -0.00311026 |
| execution_score  |       3 |     13 |        0.932867 |          -0.067133  |                 -0.740004 |           -0.287563  |        -0.00261026 |
| selection_score  |      10 |     13 |        1.06273  |           0.0627332 |                  2.25255  |           -0.0589819 |         0.00493615 |
| blend_62_38      |      10 |     13 |        1.05222  |           0.0522172 |                  1.68228  |           -0.0589819 |         0.00417385 |

## Interpretation

- This is only a proxy NAV, not the full unified portfolio engine, because replay rows expose next-day outcome slices rather than full OHLC holding paths.
- Even with that caveat, the direction is clear: removing the continuous execution weight does not hurt top-3 and improves top-10 on this replay slice.
- That makes execution-off a justified next branch to test in the real portfolio backtest path.

## Next Actions

1. Add an execution-off ranking branch to the research path and rerun unified portfolio backtests.
2. Keep discrete execution states for veto/explanation while the continuous weight is removed.
3. Replace this proxy with true portfolio results once the execution-off branch is wired into the unified backtest stack.

Summary CSV: `E:\openclaw\.cache\trading_system_attribution\execution_off_nav_proxy_summary.csv`
Curve CSV: `E:\openclaw\.cache\trading_system_attribution\execution_off_nav_proxy_curves.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\execution_off_nav_proxy.json`