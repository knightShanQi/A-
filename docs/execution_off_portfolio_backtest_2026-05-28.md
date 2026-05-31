# Execution-Off Portfolio Backtest 2026-05-28

## Purpose

Run the unified portfolio backtester directly on v9 snapshot candidates and cached daily history, comparing execution-off ranking against the current blended execution weighting.

## Coverage

- Snapshot rows: 527
- Cached history rows: 186540

## Key Findings

- On the real unified portfolio backtester fed by v9 snapshots plus cached daily history, top-3 execution-off ends at 1459955 and the current blend ends at 1459955.
- At top-10, execution-off vs current blend: annualized 1.39% vs 1.39%, ending equity 1407842 vs 1406876, max drawdown -11.83% vs -11.79%.

## Summary

| score_col        |   top_n |   candidate_rows |   initial_capital |   ending_equity |   cumulative_return |   annualized_return |   max_drawdown |   trade_count |   win_rate |   avg_net_return |
|:-----------------|--------:|-----------------:|------------------:|----------------:|--------------------:|--------------------:|---------------:|--------------:|-----------:|-----------------:|
| selection_score  |       3 |               39 |             1e+06 |     1.45995e+06 |            0.459955 |           0.0154165 |     -0.0353602 |            11 |   0.727273 |        0.0686472 |
| blend_62_38      |       3 |               39 |             1e+06 |     1.45995e+06 |            0.459955 |           0.0154165 |     -0.0353602 |            11 |   0.727273 |        0.0686472 |
| final_rank_score |       3 |                0 |             1e+06 |     1e+06       |            0        |           0         |      0         |             0 |   0        |        0         |
| selection_score  |      10 |              130 |             1e+06 |     1.40784e+06 |            0.407842 |           0.0139255 |     -0.118269  |            41 |   0.804878 |        0.0708143 |
| blend_62_38      |      10 |              130 |             1e+06 |     1.40688e+06 |            0.406876 |           0.0138973 |     -0.117894  |            41 |   0.780488 |        0.0705368 |

## Interpretation

- This is the closest experiment so far to the real question: does execution-off help once trades are run through the unified portfolio engine instead of just replay averages.
- If top-10 improves here without worse drawdown, the next step is to wire execution-off into the main research ranking path by default for A/B backtests.

## Next Actions

1. Use the new execution-off parameter in the real research branch and compare against the existing default in longer backtests.
2. Keep discrete execution states for veto/explanation while the continuous execution weight is being removed.
3. Extend this from v9 snapshots to a longer historical candidate source once the experiment shape is accepted.

Summary CSV: `E:\openclaw\.cache\trading_system_attribution\execution_off_portfolio_backtest_summary.csv`
NAV CSV: `E:\openclaw\.cache\trading_system_attribution\execution_off_portfolio_backtest_nav.csv`
Trades CSV: `E:\openclaw\.cache\trading_system_attribution\execution_off_portfolio_backtest_trades.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\execution_off_portfolio_backtest.json`