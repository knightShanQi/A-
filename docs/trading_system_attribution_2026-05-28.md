# Trading System Attribution 2026-05-28

## Unified Portfolio Attribution

This addendum compares representative strategy paths under the unified portfolio backtest engine wherever available.

## Key Findings

- Baseline combined rule `combined_top3_score_ge_68_full` drops from legacy `+3.44%` annualized to portfolio `-7.54%`, with portfolio max drawdown `68.03%`.
- Applying the V3 market filter to the same combined baseline (`existing_v3_full_green_top3`) lifts portfolio annualized to `0.46%` and cuts max drawdown to `36.40%`, mainly by reducing trades.
- Native V3 candidate generation (`v3_full_green_top3`) is the best ten-year migrated path in current evidence at `2.66%` annualized with `22.53%` max drawdown.
- The rebuilt bull/rank path still looks strong under legacy averaging (`+10.99%` annualized), but under unified portfolio NAV it is `-0.10%`, which is effectively flat to slightly negative.
- The recent top10 path shows `461.32%` annualized, but it only covers `2025-11-26 to 2026-05-26` and should be treated as a short-window observation, not ten-year proof.

## Attribution Table

| strategy_id                     | family                | date_from   | date_to    |   portfolio_trade_count |   portfolio_annualized_return |   portfolio_max_drawdown |   portfolio_cumulative_return | sample_note                                                                       |
|:--------------------------------|:----------------------|:------------|:-----------|------------------------:|------------------------------:|-------------------------:|------------------------------:|:----------------------------------------------------------------------------------|
| combined_baseline_top3_score68  | combined_baseline     | 2016-05-27  | 2026-05-26 |                     294 |                     -0.075361 |                -0.680279 |                     -0.531116 | Ten-year baseline combined selection, re-evaluated with unified portfolio engine. |
| combined_plus_market_regime     | market_regime_overlay | 2016-05-27  | 2026-05-26 |                     137 |                      0.004563 |                -0.363986 |                      0.044991 | Same combined baseline after V3 full_green market filter.                         |
| native_v3_full_green_top3       | native_market_regime  | 2016-05-27  | 2026-05-26 |                     147 |                      0.026602 |                -0.225315 |                      0.288905 | Native V3 candidate generation plus full_green top3.                              |
| native_v3_full_green_top3_pause | native_market_regime  | 2016-05-27  | 2026-05-26 |                     138 |                      0.026353 |                -0.226591 |                      0.285889 | Native V3 full_green top3 with pause overlay.                                     |
| bull_rank_sorted_as_cash        | bull_rank_overlay     | 2016-05-27  | 2026-05-26 |                     447 |                     -0.001036 |                -0.535015 |                     -0.009968 | Bull/rank overlay, full calendar as cash, rebuilt rank score sort.                |
| bull_model_sorted_as_cash       | bull_rank_overlay     | 2016-05-27  | 2026-05-26 |                     446 |                     -0.059728 |                -0.626341 |                     -0.448619 | Bull/rank overlay, full calendar as cash, model-score sort.                       |
| recent_top10_6m                 | recent_top10          | 2025-11-26  | 2026-05-26 |                    1031 |                      4.61316  |                -0.269391 |                    nan        | Six-month recent sample only; not directly comparable to ten-year studies.        |

## Interpretation

- The largest drop happens when legacy average-selected-return curves are replaced by capital-constrained portfolio NAV. This is strongest in the baseline combined rule and in the rebuilt bull/rank path.
- Market-state filtering does help, but mostly through exposure control and drawdown compression, not through strong per-trade alpha.
- Native V3 full_green selection currently dominates other ten-year migrated paths, but its real portfolio annualized return is still only low-single-digit, which confirms the audit conclusion that the system's core alpha is weak.
- The six-month top10 path is promising but not yet comparable. It needs a longer-window rerun under the same engine before it should influence architectural conclusions.

CSV: `E:\openclaw\.cache\trading_system_attribution\cross_strategy_attribution.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\cross_strategy_attribution.json`