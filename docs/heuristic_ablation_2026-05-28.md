# Heuristic Ablation 2026-05-28

## Purpose

Quantify which heuristic layers actually improve the unified portfolio NAV and which ones mainly add complexity.

## Key Findings

- Raw model ranking and raw heuristic ranking both fail badly before market-state filtering, with portfolio annualized returns of -51.72% and -79.87% respectively.
- Adding the strict full-green gate to the historical combined baseline improves annualized return by 7.99 percentage points and reduces trade count by 157, but only reaches 0.46% annualized.
- Switching from the historical combined candidate family to the native V3 full-green candidate family adds another 2.20 percentage points of annualized return, which is larger than the gain from most overlay tweaks.
- The pause overlay barely changes the native strict-gate result: annualized return moves by -0.02 percentage points.
- Within the bull/rank research path, rebuilt rank-score sorting beats model-score sorting by 5.87 percentage points, which is evidence that the current model ordering is not the strongest signal even inside the same gated pool.

## Family Comparison

| family                | stage                        | parent_stage           |   portfolio_trade_count |   portfolio_annualized_return |   portfolio_max_drawdown |   delta_vs_parent_annualized |   delta_vs_parent_trade_count |
|:----------------------|:-----------------------------|:-----------------------|------------------------:|------------------------------:|-------------------------:|-----------------------------:|------------------------------:|
| raw_signal_baselines  | model_only_top3              |                        |                    7227 |                     -0.517242 |                -0.999454 |                   nan        |                           nan |
| raw_signal_baselines  | strategy_only_top3           |                        |                    7008 |                     -0.798741 |                -1        |                   nan        |                           nan |
| combined_overlay_path | combined_baseline            |                        |                     294 |                     -0.075361 |                -0.680279 |                   nan        |                           nan |
| combined_overlay_path | combined_trend_gate          | combined_baseline      |                     271 |                     -0.076695 |                -0.667265 |                    -0.001334 |                           -23 |
| combined_overlay_path | combined_trend_flow_gate     | combined_baseline      |                     234 |                     -0.026073 |                -0.497729 |                     0.049288 |                           -60 |
| combined_overlay_path | combined_full_green_gate     | combined_baseline      |                     137 |                      0.004563 |                -0.363986 |                     0.079924 |                          -157 |
| native_candidate_path | native_v2_score68            |                        |                     261 |                     -0.006781 |                -0.355111 |                   nan        |                           nan |
| native_candidate_path | native_trend_gate            | native_v2_score68      |                     240 |                     -0.012335 |                -0.38699  |                    -0.005554 |                           -21 |
| native_candidate_path | native_trend_flow_gate       | native_v2_score68      |                     216 |                      0.009961 |                -0.305031 |                     0.016742 |                           -45 |
| native_candidate_path | native_full_green_gate       | native_v2_score68      |                     147 |                      0.026602 |                -0.225315 |                     0.033383 |                          -114 |
| native_candidate_path | native_full_green_plus_pause | native_full_green_gate |                     138 |                      0.026353 |                -0.226591 |                    -0.000249 |                            -9 |
| bull_rank_research    | bull_rank_sorted             |                        |                     447 |                     -0.001036 |                -0.535015 |                   nan        |                           nan |
| bull_rank_research    | bull_model_sorted            | bull_rank_sorted       |                     446 |                     -0.059728 |                -0.626341 |                    -0.058692 |                            -1 |

## Interpretation

- The first-order gain comes from shrinking exposure and candidate-family quality, not from stacking more heuristic scorers on top of the same pool.
- The strict full-green gate is useful, but it is a damage-control layer, not a sufficient alpha source by itself.
- Candidate-family replacement matters more than pause-style micro-overlays in current evidence.
- The strategy heuristic layer remains the weakest standalone component and should not be allowed to dominate ranking.

## Actionable Order

1. Keep `native_v3_full_green_top3` as the live research baseline.
2. Retire `combined_top3_score_ge_68_full` from default comparison and stop threshold-tuning it.
3. Treat pause overlays as optional polish only after the base family and market-state gate are proven.
4. Rebuild model training around cross-sectional ranking because even inside the same gated pool, model sorting loses to rebuilt rank sorting.

CSV: `E:\openclaw\.cache\trading_system_attribution\heuristic_ablation.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\heuristic_ablation.json`