# Final Rank Rebuild Report 2026-05-28

## Purpose

Test a few lightweight rebuilt final-rank formulas on persisted v9 snapshot fields under the same unified portfolio engine.

## Coverage

- Snapshot rows: 527
- Cached history rows: 186540

## Key Findings

- `selection_score_top3` remains the baseline to beat at 1.54% annualized.
- `final_rank_score_top3` remains weak at -0.13% annualized because it is still the attention-style alias path.
- The best lightweight rebuild on current persisted fields is `rebuild_balanced` at 0.97% annualized.
- If no rebuild beats selection_score, the immediate optimization focus should stay on improving selection_score construction rather than inventing another ranking label.

## Portfolio Summary

| score_col            |   candidate_rows |   avg_score |   median_score |   initial_capital |    ending_equity |   cumulative_return |   annualized_return |   max_drawdown |   trade_count |   win_rate |   avg_net_return |
|:---------------------|-----------------:|------------:|---------------:|------------------:|-----------------:|--------------------:|--------------------:|---------------:|--------------:|-----------:|-----------------:|
| selection_score      |               39 |     97.4325 |       100      |             1e+06 |      1.45995e+06 |           0.459955  |          0.0154165  |     -0.0353602 |            11 |   0.727273 |      0.0686472   |
| final_rank_score     |               39 |     92.3759 |       100      |             1e+06 | 969089           |          -0.0309107 |         -0.00126863 |     -0.127035  |            12 |   0.583333 |     -0.000977181 |
| rebuild_balanced     |               39 |     92.1903 |        93.7015 |             1e+06 |      1.27103e+06 |           0.271032  |          0.00974345 |     -0.146749  |            10 |   0.7      |      0.0454902   |
| rebuild_launch_guard |               39 |     92.7605 |        95.0985 |             1e+06 |      1.27103e+06 |           0.271032  |          0.00974345 |     -0.146749  |            10 |   0.7      |      0.0454902   |
| rebuild_quant_prob   |               39 |     93.6198 |        96.2062 |             1e+06 |      1.27103e+06 |           0.271032  |          0.00974345 |     -0.146749  |            10 |   0.7      |      0.0454902   |
| rebuild_zmix         |               39 |     59.0713 |        59.4064 |             1e+06 |      1.27103e+06 |           0.271032  |          0.00974345 |     -0.146749  |            10 |   0.7      |      0.0454902   |

## Top-3 Overlap Vs Selection

| score_col            |   comparable_dates |   identical_days_vs_selection |   avg_overlap_count_vs_selection |
|:---------------------|-------------------:|------------------------------:|---------------------------------:|
| selection_score      |                 13 |                            13 |                          3       |
| final_rank_score     |                 13 |                             2 |                          1.07692 |
| rebuild_balanced     |                 13 |                             6 |                          1.76923 |
| rebuild_launch_guard |                 13 |                             6 |                          1.76923 |
| rebuild_quant_prob   |                 13 |                             6 |                          1.76923 |
| rebuild_zmix         |                 13 |                             5 |                          1.76923 |

## Interpretation

- These are not production formula recommendations. They are direction tests on the fields that currently survive into snapshots.
- If even the best lightweight rebuild cannot beat `selection_score_top3`, then the path of least regret is to improve `selection_score` itself rather than creating a second ranking brand.
- If a rebuild gets close but still loses, it can still help identify which ingredients are directionally useful for future model-aware ranking work.

## Next Actions

1. Keep `selection_score_top3` as the default research baseline unless a rebuilt final-rank formula clearly beats it in the same engine.
2. Use the best rebuild only as an ingredient clue, not as a production replacement, until it is validated on longer history.
3. Avoid adding execution-style or attention-style terms back into a rebuild unless they show distinct portfolio lift.

Summary CSV: `E:\openclaw\.cache\trading_system_attribution\final_rank_rebuild_summary.csv`
Overlap CSV: `E:\openclaw\.cache\trading_system_attribution\final_rank_rebuild_overlap.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\final_rank_rebuild.json`