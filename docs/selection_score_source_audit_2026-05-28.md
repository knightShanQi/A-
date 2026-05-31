# Selection Score Source Audit 2026-05-28

## Purpose

Audit the live `selection_score` construction path as a system-design problem: where the score is authored, which persisted inputs are most redundant, and which missing inputs still block a cleaner optimization pass.

## Authoritative Source Sites

- [`src/a_share_predictor/dashboard.py:1887`](E:\openclaw\src\a_share_predictor\dashboard.py:1887): full display-context score construction with probability, attention, quant, launch-window, tomorrow-confidence, sector/fund/news, and manual adjustments.
- [`src/a_share_predictor/dashboard.py:2093`](E:\openclaw\src\a_share_predictor\dashboard.py:2093): replay-side simplified score construction used for board-level evaluation.

## Coverage

- Review rows: 525
- Review files: 13

| field                    | present_in_v9_review   |   non_null_rows |
|:-------------------------|:-----------------------|----------------:|
| probability_up           | True                   |             525 |
| attention_score          | True                   |             525 |
| enhanced_attention_score | True                   |             525 |
| quant_score              | True                   |             525 |
| launch_window_score      | True                   |             525 |
| launch_window_confidence | True                   |             525 |
| stage_score              | True                   |             450 |
| launch_readiness_score   | True                   |             450 |
| market_resonance_score   | True                   |             450 |
| predicted_upside_pct     | False                  |               0 |
| tomorrow_plan_confidence | False                  |               0 |
| sector_score             | False                  |               0 |
| fund_score               | False                  |               0 |
| news_score               | False                  |               0 |
| technical_adjustment     | False                  |               0 |
| intraday_adjustment      | False                  |               0 |
| backtest_adjustment      | False                  |               0 |

## Correlation With Selection Score

| field                    |   correlation_with_selection_score |
|:-------------------------|-----------------------------------:|
| selection_score          |                           1        |
| attention_score          |                           0.92586  |
| enhanced_attention_score |                           0.906663 |
| probability_up           |                           0.867549 |
| launch_window_score      |                           0.858601 |
| quant_score              |                           0.791668 |
| launch_window_confidence |                           0.774379 |
| stage_score              |                           0.115204 |
| launch_readiness_score   |                           0.11072  |
| market_resonance_score   |                           0.100659 |

## Top-3 Overlap Against Selection Score

| field                    |   avg_top3_overlap |   exact_top3_match_days |
|:-------------------------|-------------------:|------------------------:|
| launch_window_confidence |            1.84615 |                       4 |
| launch_window_score      |            1.76923 |                       3 |
| probability_up           |            1.53846 |                       4 |
| launch_readiness_score   |            1.46154 |                       3 |
| market_resonance_score   |            1.46154 |                       4 |
| attention_score          |            1.38462 |                       3 |
| enhanced_attention_score |            1.38462 |                       3 |
| stage_score              |            1.38462 |                       3 |
| quant_score              |            1.30769 |                       3 |

## Linear Projection

- Rows with full persisted source-family coverage: 450
- R² from persisted source-family projection: 95.61%

| feature                  |   standardized_coefficient |   abs_standardized_coefficient |
|:-------------------------|---------------------------:|-------------------------------:|
| launch_window_confidence |                  1.02494   |                      1.02494   |
| launch_window_score      |                 -0.728612  |                      0.728612  |
| probability_up           |                  0.478929  |                      0.478929  |
| attention_score          |                  0.27487   |                      0.27487   |
| enhanced_attention_score |                  0.205251  |                      0.205251  |
| quant_score              |                  0.146448  |                      0.146448  |
| stage_score              |                  0.118625  |                      0.118625  |
| market_resonance_score   |                  0.0947862 |                      0.0947862 |
| launch_readiness_score   |                 -0.0473349 |                      0.0473349 |

## Key Findings

- The strongest persisted same-slice linear relationship to `selection_score` is `attention_score` at correlation 0.926.
- The highest top-3 overlap with `selection_score` among persisted source-family fields is `launch_window_confidence` at 1.85 / 3 names on average.
- The persisted source-family projection explains about 95.61% of `selection_score` variance on 450 rows, but coefficient sign-flips indicate heavy multicollinearity rather than clean independent evidence.
- `quant_score` remains only a secondary driver in persisted evidence (correlation 0.792), while `stage_score` remains weak (correlation 0.115).
- Several exact formula inputs are still not persisted into v9 review artifacts, so source-level optimization is partly bottlenecked by observability, not just by weak alpha.

## Interpretation

- The live `selection_score` is not a clean stack of independent evidence. The persisted source-family terms already explain almost all score variance, but they do so with obvious redundancy and coefficient instability.
- `probability_up`, `attention_score`, and `enhanced_attention_score` form an especially stacked attention family. They are all strongly tied to `selection_score`, which means adding more weight inside that family is likely to reshuffle the same narrative signal rather than add new alpha.
- `launch_window_score` and `launch_window_confidence` are structurally close to the realized top-3 picks, but earlier realized-return tests showed launch-window works better as context than as a standalone ranker. That combination is exactly what an over-layered gate family looks like.
- `quant_score` is not driving the score nearly as strongly as the attention and launch families, which fits the earlier conclusion that quant is better treated as mild confirmation or damage control than as the main sorting engine.
- `stage_score` remains too weak in persisted evidence to justify prominent default weight inside the ranking path.

## Optimization Guidance

1. Collapse the attention family into a cleaner core path before adding any new overlays. At minimum, treat `probability_up`, `attention_score`, and `enhanced_attention_score` as one correlated cluster, not as three independent alpha sources.
2. Keep launch-window as structure/risk context, but stop trying to improve the system by adding more launch weight. The source audit and the realized-return experiments already say it is close to saturation.
3. Do not promote `quant_score` or `stage_score` into primary ranking roles. If quant is used, keep it as a mild filter candidate; if stage remains this weak, demote it from default weighting until stronger evidence appears.
4. Close the observability gap before any deeper source-level optimization: persist `predicted_upside_pct`, `tomorrow_plan_confidence`, `sector_score`, `fund_score`, `news_score`, and the adjustment terms into v9 review artifacts so future ablations can audit the real formula instead of a partial shadow.

Coverage CSV: `E:\openclaw\.cache\trading_system_attribution\selection_score_source_coverage.csv`
Correlation CSV: `E:\openclaw\.cache\trading_system_attribution\selection_score_source_correlations.csv`
Overlap CSV: `E:\openclaw\.cache\trading_system_attribution\selection_score_source_overlap.csv`
Projection CSV: `E:\openclaw\.cache\trading_system_attribution\selection_score_source_projection_coefficients.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\selection_score_source_audit.json`