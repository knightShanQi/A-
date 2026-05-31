# Tomorrow Confidence Contribution Report 2026-05-28

## Purpose

Measure whether tomorrow-plan confidence helps more as a ranker, as a gate, or as a small tilt on top of selection_score.

## Coverage

- Snapshot rows: 527
- Cached history rows: 186540

## Key Findings

- `selection_score_top3` baseline stays strongest at 1.54% annualized with -3.54% drawdown.
- The best tomorrow-confidence gate is `selection_tomorrow_ge_65` at 0.53% annualized.
- `tomorrow_confidence_only_top3` lands at 0.95% annualized, which shows tomorrow-confidence is not a strong standalone ranker.
- `selection_plus_tomorrow_tilt` lands at 0.50% annualized, so overweighting tomorrow-confidence on top of selection does not beat the current baseline on this slice.

## Portfolio Summary

| variant                       | score_col                    |   candidate_rows |   avg_tomorrow_confidence | bias_mix                                      |   initial_capital |   ending_equity |   cumulative_return |   annualized_return |   max_drawdown |   trade_count |   win_rate |   avg_net_return |
|:------------------------------|:-----------------------------|-----------------:|--------------------------:|:----------------------------------------------|------------------:|----------------:|--------------------:|--------------------:|---------------:|--------------:|-----------:|-----------------:|
| baseline_selection_top3       | selection_score              |               39 |                   71.4995 | {"偏强观察": 30, "偏多进攻": 5, "偏多确认": 3, "中性等待": 1} |             1e+06 |     1.45995e+06 |           0.459955  |          0.0154165  |     -0.0353602 |            11 |   0.727273 |        0.0686472 |
| selection_tomorrow_ge_65      | selection_score              |               21 |                   82.9886 | {"偏强观察": 12, "偏多进攻": 5, "偏多确认": 3, "中性等待": 1} |             1e+06 |     1.13907e+06 |           0.13907   |          0.00527835 |     -0.0214365 |             8 |   0.75     |        0.0540002 |
| selection_tomorrow_ge_72      | selection_score              |               21 |                   82.9886 | {"偏强观察": 12, "偏多进攻": 5, "偏多确认": 3, "中性等待": 1} |             1e+06 |     1.13907e+06 |           0.13907   |          0.00527835 |     -0.0214365 |             8 |   0.75     |        0.0540002 |
| selection_bias_positive_only  | selection_score              |                8 |                   89.0162 | {"偏多进攻": 5, "偏多确认": 3}                        |             1e+06 |     1.02735e+06 |           0.0273505 |          0.00109152 |     -0.0379587 |             5 |   0.6      |        0.0224156 |
| selection_plus_tomorrow_tilt  | selection_plus_tomorrow_tilt |               39 |                   70.9046 | {"偏强观察": 30, "偏多进攻": 4, "偏多确认": 3, "中性等待": 2} |             1e+06 |     1.1317e+06  |           0.131705  |          0.00501473 |     -0.0680299 |            12 |   0.666667 |        0.0467369 |
| tomorrow_confidence_only_top3 | tomorrow_plan_confidence     |               39 |                   72.2458 | {"偏强观察": 30, "偏多进攻": 5, "偏多确认": 3, "中性等待": 1} |             1e+06 |     1.26243e+06 |           0.262429  |          0.00946622 |     -0.148028  |             9 |   0.666667 |        0.0278685 |

## Interpretation

- If tomorrow-confidence works poorly as a standalone ranker but still matters inside selection_score, it is probably acting as a context or risk-shaping signal rather than a primary alpha source.
- If strict tomorrow-confidence gates improve drawdown more than return, that points to a risk-control contribution rather than a ranking contribution.
- If extra tomorrow-confidence tilt hurts, then the live selection score may already be using about as much of it as this slice can support.

## Next Actions

1. Keep tomorrow-confidence inside selection_score if it helps the baseline indirectly, but do not promote it to a standalone ranker unless longer-history evidence improves dramatically.
2. If one tomorrow-confidence gate reduces drawdown without destroying return, retest it later on longer history before using it as a hard filter.
3. Continue source-level auditing around execution and quant only after the higher-value context families are settled.

Summary CSV: `E:\openclaw\.cache\trading_system_attribution\tomorrow_confidence_contribution_summary.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\tomorrow_confidence_contribution.json`