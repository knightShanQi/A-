# Quant Score Contribution Report 2026-05-28

## Purpose

Measure whether quant-score information helps more as a ranker, as a gate, or as a small tilt on top of selection_score.

## Coverage

- Snapshot rows: 527
- Cached history rows: 186540

## Key Findings

- `selection_score_top3` baseline stays strongest at 1.54% annualized with -3.54% drawdown.
- The best quant gate is `selection_quant_ge_58` at 1.60% annualized.
- `quant_only_top3` lands at 0.07% annualized, which shows quant is not a strong standalone ranker on this slice.
- `selection_plus_quant_tilt` lands at 0.50% annualized, so overweighting quant on top of selection does not beat the current baseline on this slice.

## Portfolio Summary

| variant                   | score_col                 |   candidate_rows |   avg_quant_score |   initial_capital |    ending_equity |   cumulative_return |   annualized_return |   max_drawdown |   trade_count |   win_rate |   avg_net_return |
|:--------------------------|:--------------------------|-----------------:|------------------:|------------------:|-----------------:|--------------------:|--------------------:|---------------:|--------------:|-----------:|-----------------:|
| baseline_selection_top3   | selection_score           |               39 |           91.2638 |             1e+06 |      1.45995e+06 |         0.459955    |         0.0154165   |    -0.0353602  |            11 |   0.727273 |       0.0686472  |
| selection_quant_ge_58     | selection_score           |               38 |           92.3449 |             1e+06 |      1.48159e+06 |         0.48159     |         0.0160206   |    -0.0296908  |            11 |   0.818182 |       0.0804002  |
| selection_quant_ge_68     | selection_score           |               36 |           94.5438 |             1e+06 |      1.41687e+06 |         0.416869    |         0.0141875   |    -0.0610976  |            12 |   0.666667 |       0.0666239  |
| selection_quant_le_45     | selection_score           |                6 |           41.36   |             1e+06 | 999393           |        -0.000606829 |        -2.45412e-05 |    -0.00556244 |             3 |   0.333333 |      -0.00142279 |
| selection_plus_quant_tilt | selection_plus_quant_tilt |               39 |           90.9316 |             1e+06 |      1.1317e+06  |         0.131705    |         0.00501473  |    -0.0680299  |            12 |   0.666667 |       0.0467369  |
| quant_only_top3           | quant_score               |               39 |           92.4478 |             1e+06 |      1.01726e+06 |         0.0172608   |         0.000692138 |    -0.125296   |            13 |   0.538462 |       0.00578604 |

## Interpretation

- If quant works poorly as a standalone ranker but gates or tilts can still help, it is probably a secondary confirmation feature rather than a primary alpha source.
- If quant gates do not improve the baseline materially, the current live score may already contain about the right amount of quant input for this slice.
- If extra quant tilt hurts, then simple weight increases are unlikely to be the right optimization path.

## Next Actions

1. Keep quant inside selection_score only if it supports the baseline indirectly; do not promote it into a separate ranking track without stronger evidence.
2. If one quant gate looks useful on drawdown-adjusted terms, test it later on longer history before adopting it.
3. Once quant and tomorrow/launch context roles are all characterized, the next remaining weak family to challenge is execution score usage in live selection logic.

Summary CSV: `E:\openclaw\.cache\trading_system_attribution\quant_score_contribution_summary.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\quant_score_contribution.json`