# v10 Attention Unification 2026-05-28

## Purpose

Test source-level consolidation variants for the duplicated attention family instead of bluntly subtracting one layer from the persisted `selection_score`.

## Coverage

- Snapshot rows: 527
- Cached history rows: 186540
- Rows where `enhanced_attention_score == attention_score`: 317
- Rows where the two attention fields differ: 210

## Key Findings

- `attention_score` and `enhanced_attention_score` are identical on 317 / 527 persisted `v10` rows, with correlation 0.9875.
- Replacing both weights with the base-attention representation (`selection_unify_to_base_attention`) yields 1.39% annualized and -4.84% drawdown versus baseline 1.54% / -3.54%.
- Replacing both weights with the enhanced-attention representation (`selection_unify_to_enhanced_attention`) yields 1.21% annualized and -7.23% drawdown.
- The mean-attention consolidation variant lands at 1.26% annualized with -4.92% drawdown, which shows whether a single blended attention representation can survive without full dual-layer stacking.
- Non-zero enhanced-minus-base attention deltas exist on 210 rows, so the right cleanup target is not field deletion but score-family consolidation around one surviving attention representation.

## Summary

| score_col                               |   candidate_rows |   avg_top3_overlap_vs_baseline |   exact_top3_match_days |   initial_capital |   ending_equity |   cumulative_return |   annualized_return |   max_drawdown |   trade_count |   win_rate |   avg_net_return |
|:----------------------------------------|-----------------:|-------------------------------:|------------------------:|------------------:|----------------:|--------------------:|--------------------:|---------------:|--------------:|-----------:|-----------------:|
| selection_score                         |               39 |                        3       |                      13 |             1e+06 |     1.45995e+06 |            0.459955 |           0.0154165 |     -0.0353602 |            11 |   0.727273 |        0.0686472 |
| selection_unify_to_base_attention       |               39 |                        2.30769 |                       8 |             1e+06 |     1.40544e+06 |            0.40544  |           0.0138555 |     -0.0484225 |            10 |   0.7      |        0.0749231 |
| selection_mean_attention_representation |               39 |                        2.38462 |                       8 |             1e+06 |     1.36174e+06 |            0.361744 |           0.0125617 |     -0.0491669 |            10 |   0.7      |        0.0556431 |
| selection_unify_to_enhanced_attention   |               39 |                        2.30769 |                       8 |             1e+06 |     1.34508e+06 |            0.345076 |           0.0120576 |     -0.0723144 |            12 |   0.75     |        0.0422398 |

## Interpretation

- A unification variant is more faithful to the actual architecture choice than a pure subtraction variant because it asks whether one attention representation can replace two, not whether attention should disappear.
- If one unified representation stays close to baseline, the next engineering move should be consolidating the formula inputs and persistence around that single representation.
- If every unification variant still degrades materially, the immediate optimization target should move away from attention-family cleanup and toward other score families or model quality.

## Next Actions

1. Prefer the least-damaging unification variant as the next source-level simplification candidate.
2. If that variant is still materially weaker, keep both attention layers for now and postpone consolidation until the upstream formulas are rebuilt.

Summary CSV: `E:\openclaw\.cache\trading_system_attribution\v10_attention_unification_summary.csv`
NAV CSV: `E:\openclaw\.cache\trading_system_attribution\v10_attention_unification_nav.csv`
Trades CSV: `E:\openclaw\.cache\trading_system_attribution\v10_attention_unification_trades.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\v10_attention_unification.json`
