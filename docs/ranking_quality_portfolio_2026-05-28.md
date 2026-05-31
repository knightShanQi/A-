# Ranking Quality Portfolio Report 2026-05-28

## Purpose

Compare the persisted ranking paths on the current v9 candidate slice under the same unified portfolio engine instead of using replay buckets alone.

## Coverage

- Snapshot rows: 527
- Cached history rows: 186540

## Key Findings

- On the current v9 unified-portfolio slice, `selection_score_top3` is the strongest realized ranking path at 1.54% annualized with -3.54% max drawdown.
- `launch_window_score_top3` is directionally useful but weaker as a pure ranking key: 0.62% annualized vs 1.54% for selection.
- `enhanced_attention_score_top3` is currently the weakest of the persisted top-3 rankers on this slice at -0.13% annualized and -12.70% max drawdown.
- `final_rank_score_top3` now has same-engine evidence on `39` candidate rows and currently matches `enhanced_attention_score_top3` exactly at -0.13% annualized.
- Even the broader `selection_score_top10` path only reaches 1.39% annualized with -11.83% drawdown, so widening the basket still does not solve the deeper alpha problem.

## Summary

| score_col                |   top_n |   candidate_rows |   score_coverage_rows |   avg_score |   median_score |    ending_equity |   cumulative_return |   annualized_return |   max_drawdown |   trade_count |   win_rate |   avg_net_return |
|:-------------------------|--------:|-----------------:|----------------------:|------------:|---------------:|-----------------:|--------------------:|--------------------:|---------------:|--------------:|-----------:|-----------------:|
| selection_score          |       3 |               39 |                   527 |     97.4325 |       100      |      1.45995e+06 |           0.459955  |          0.0154165  |     -0.0353602 |            11 |   0.727273 |      0.0686472   |
| selection_score          |      10 |              130 |                   527 |     95.9766 |        99.1315 |      1.40784e+06 |           0.407842  |          0.0139255  |     -0.118269  |            41 |   0.804878 |      0.0708143   |
| launch_window_score      |       3 |               39 |                   527 |     89.8956 |        93.7    |      1.16596e+06 |           0.165955  |          0.00622696 |     -0.2158    |            11 |   0.636364 |      0.0338038   |
| enhanced_attention_score |       3 |               39 |                   527 |     92.3759 |       100      | 969089           |          -0.0309107 |         -0.00126863 |     -0.127035  |            12 |   0.583333 |     -0.000977181 |
| final_rank_score         |       3 |               39 |                   527 |     92.3759 |       100      | 969089           |          -0.0309107 |         -0.00126863 |     -0.127035  |            12 |   0.583333 |     -0.000977181 |

## Interpretation

- On current persisted evidence, `selection_score` is still the best ranking key among the scorers that actually survive into snapshots and can be replayed through the real portfolio engine.
- `launch_window_score` appears useful as a supporting filter or overlay, but not as the primary ranker.
- `enhanced_attention_score` is weaker than `selection_score` on this slice, which supports the earlier recommendation to keep the candidate stack simple and focus on ranking quality instead of adding more narrative overlays.
- After snapshot backfill, `final_rank_score` no longer has an observability gap on this slice; it currently behaves the same as `enhanced_attention_score`, so it still does not justify promotion over `selection_score`.

## Next Actions

1. Keep `selection_score_top3` as the current default research ranking baseline.
2. Treat `launch_window_score` as filter/overlay research, not as the main ranker, until it beats `selection_score_top3` in the same engine.
3. If `final_rank_score` is intended to be more than a rename of `enhanced_attention_score`, change its construction first and rerun this exact portfolio A/B before promoting it.

Summary CSV: `E:\openclaw\.cache\trading_system_attribution\ranking_quality_portfolio_summary.csv`
NAV CSV: `E:\openclaw\.cache\trading_system_attribution\ranking_quality_portfolio_nav.csv`
Trades CSV: `E:\openclaw\.cache\trading_system_attribution\ranking_quality_portfolio_trades.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\ranking_quality_portfolio.json`