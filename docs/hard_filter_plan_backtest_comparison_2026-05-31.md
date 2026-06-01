# Hard Filter Plan Backtest Comparison 2026-05-31

## Scope

- Window: `2016-05-27` to `2026-05-26`.
- New strategy implementation: P1 proxy from `strategy_hard_filter_optimization_plan_2026-05-31.md` using available ten-year daily cache features.
- P1 candidate-pool cap: top `150` by strategy soft score per day with per-strategy quota.
- Baseline ranking in this comparison uses existing ten-year `model_scores.csv`.
- Per-strategy retraining/calibration for this plan is handled by `scripts/train_hard_filter_plan_strategy_models.py`.
- Latest retraining artifacts: `.cache/hard_filter_plan_strategy_models/`.
- Caveat: ten-year cache does not include true ten-year intraday, fund-flow or news features, so those plan terms are not included here.

## Candidate Breadth

- `old_strict` active days: `2352`; avg daily candidates: `22.3`; median: `18.0`; p90: `42.0`.
- `plan_p1` active days: `2372`; avg daily candidates: `111.7`; median: `149.0`; p90: `150.0`.

## Key Same-Rule Comparison

| rule | cost_bps | source | active_days | selected | ann_return | max_dd | win_rate | avg_trade |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| all__bull7_v3_full_green__score68__prionone__top3__model_score | 0 | old_strict | 103 | 309 | 7.57% | -20.70% | 46.60% | 0.79% |
| all__bull7_v3_full_green__score68__prionone__top3__model_score | 0 | plan_p1 | 124 | 372 | 2.67% | -46.05% | 46.24% | 0.33% |
| all__bull7_v3_full_green__score68__prionone__top3__model_score | 20 | old_strict | 103 | 309 | 5.25% | -22.46% | 45.95% | 0.59% |
| all__bull7_v3_full_green__score68__prionone__top3__model_score | 20 | plan_p1 | 124 | 372 | 0.00% | -53.64% | 44.62% | 0.13% |
| all__v3_full_green__score68__prionone__top3__model_score | 0 | old_strict | 112 | 336 | 6.17% | -31.34% | 46.43% | 0.63% |
| all__v3_full_green__score68__prionone__top3__model_score | 0 | plan_p1 | 140 | 420 | 2.97% | -46.95% | 46.67% | 0.32% |
| all__v3_full_green__score68__prionone__top3__model_score | 20 | old_strict | 112 | 336 | 3.69% | -34.09% | 45.83% | 0.43% |
| all__v3_full_green__score68__prionone__top3__model_score | 20 | plan_p1 | 140 | 420 | -0.05% | -53.28% | 45.24% | 0.12% |
| strategy3__bull7_v3_full_green__score68__prionone__top3__model_score | 0 | old_strict | 102 | 306 | 8.12% | -20.18% | 47.71% | 0.85% |
| strategy3__bull7_v3_full_green__score68__prionone__top3__model_score | 0 | plan_p1 | 2 | 6 | -0.32% | -3.40% | 16.67% | -1.46% |
| strategy3__bull7_v3_full_green__score68__prionone__top3__model_score | 20 | old_strict | 102 | 306 | 5.82% | -21.96% | 47.06% | 0.65% |
| strategy3__bull7_v3_full_green__score68__prionone__top3__model_score | 20 | plan_p1 | 2 | 6 | -0.36% | -3.60% | 16.67% | -1.66% |

## Best Plan Rules By Cost

| cost_bps | rule | active_days | ann_return | max_dd | score |
|---:|---|---:|---:|---:|---:|
| 0 | strategy2__bull6_v3_full_green__score68__prionone__top3__model_priority_80_20 | 101 | 7.68% | -19.67% | 5.717 |
| 0 | strategy2__bull6_v3_full_green__score68__prio60__top3__model_priority_80_20 | 101 | 7.68% | -19.67% | 5.717 |
| 0 | strategy2__bull6_v3_full_green__score68__prio65__top3__model_priority_80_20 | 101 | 7.68% | -19.67% | 5.717 |
| 0 | strategy2__v3_full_green__score68__prionone__top3__model_priority_80_20 | 105 | 7.09% | -19.67% | 5.128 |
| 0 | strategy2__v3_full_green__score68__prio60__top3__model_priority_80_20 | 105 | 7.09% | -19.67% | 5.128 |
| 10 | strategy2__bull6_v3_full_green__score68__prionone__top3__model_priority_80_20 | 101 | 6.54% | -20.49% | 4.493 |
| 10 | strategy2__bull6_v3_full_green__score68__prio60__top3__model_priority_80_20 | 101 | 6.54% | -20.49% | 4.493 |
| 10 | strategy2__bull6_v3_full_green__score68__prio65__top3__model_priority_80_20 | 101 | 6.54% | -20.49% | 4.493 |
| 10 | strategy2__v3_full_green__score68__prionone__top3__model_priority_80_20 | 105 | 5.91% | -21.28% | 3.785 |
| 10 | strategy2__v3_full_green__score68__prio60__top3__model_priority_80_20 | 105 | 5.91% | -21.28% | 3.785 |
| 20 | strategy2__bull6_v3_full_green__score68__prionone__top3__model_priority_80_20 | 101 | 5.41% | -21.30% | 3.281 |
| 20 | strategy2__bull6_v3_full_green__score68__prio60__top3__model_priority_80_20 | 101 | 5.41% | -21.30% | 3.281 |
| 20 | strategy2__bull6_v3_full_green__score68__prio65__top3__model_priority_80_20 | 101 | 5.41% | -21.30% | 3.281 |
| 20 | strategy2__v3_full_green__score68__prionone__top3__model_priority_80_20 | 105 | 4.74% | -24.09% | 2.334 |
| 20 | strategy2__v3_full_green__score68__prio60__top3__model_priority_80_20 | 105 | 4.74% | -24.09% | 2.334 |
| 30 | strategy2__bull6_v3_full_green__score68__prionone__top3__model_priority_80_20 | 101 | 4.29% | -22.72% | 2.018 |
| 30 | strategy2__bull6_v3_full_green__score68__prio60__top3__model_priority_80_20 | 101 | 4.29% | -22.72% | 2.018 |
| 30 | strategy2__bull6_v3_full_green__score68__prio65__top3__model_priority_80_20 | 101 | 4.29% | -22.72% | 2.018 |
| 30 | strategy1__v3_full_green__score70__prionone__top3__model_score | 6 | 1.58% | 0.00% | 1.583 |
| 30 | strategy1__v3_full_green__score70__prio60__top3__model_score | 6 | 1.58% | 0.00% | 1.583 |

## Artifacts

- Summary JSON: `.cache\hard_filter_plan_comparison_full\summary.json`
- Rule comparison: `.cache\hard_filter_plan_comparison_full\rule_comparison.csv`
- Candidate counts: `.cache\hard_filter_plan_comparison_full\candidate_counts.csv`
- Selected samples: `.cache\hard_filter_plan_comparison_full\selected_samples.csv`
- Strategy model calibration report: `docs\hard_filter_plan_strategy_model_calibration_2026-06-01.md`
