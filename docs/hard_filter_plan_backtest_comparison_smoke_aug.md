# Hard Filter Plan Backtest Comparison 2026-05-31

## Scope

- Window: `2025-08-01` to `2025-08-31`.
- New strategy implementation: P1 proxy from `strategy_hard_filter_optimization_plan_2026-05-31.md` using available ten-year daily cache features.
- Ranking and labels: existing ten-year `model_scores.csv`; no model retraining in this run.
- Caveat: ten-year cache does not include true ten-year intraday, fund-flow or news features, so those plan terms are not included here.

## Candidate Breadth

- `old_strict` active days: `21`; avg daily candidates: `44.0`; median: `44.0`; p90: `66.0`.
- `plan_p1` active days: `19`; avg daily candidates: `589.2`; median: `595.0`; p90: `1007.4`.

## Key Same-Rule Comparison

| rule | cost_bps | source | active_days | selected | ann_return | max_dd | win_rate | avg_trade |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| all__bull7_v3_full_green__score68__prionone__top3__model_score | 0 | old_strict | 10 | 30 | 139.06% | -7.42% | 36.67% | 0.75% |
| all__bull7_v3_full_green__score68__prionone__top3__model_score | 0 | plan_p1 | 11 | 33 | 148.31% | -11.03% | 36.36% | 0.75% |
| all__bull7_v3_full_green__score68__prionone__top3__model_score | 20 | old_strict | 10 | 30 | 83.59% | -7.80% | 36.67% | 0.55% |
| all__bull7_v3_full_green__score68__prionone__top3__model_score | 20 | plan_p1 | 11 | 33 | 85.68% | -11.58% | 36.36% | 0.55% |
| all__v3_full_green__score68__prionone__top3__model_score | 0 | old_strict | 10 | 30 | 139.06% | -7.42% | 36.67% | 0.75% |
| all__v3_full_green__score68__prionone__top3__model_score | 0 | plan_p1 | 11 | 33 | 148.31% | -11.03% | 36.36% | 0.75% |
| all__v3_full_green__score68__prionone__top3__model_score | 20 | old_strict | 10 | 30 | 83.59% | -7.80% | 36.67% | 0.55% |
| all__v3_full_green__score68__prionone__top3__model_score | 20 | plan_p1 | 11 | 33 | 85.68% | -11.58% | 36.36% | 0.55% |
| strategy3__bull7_v3_full_green__score68__prionone__top3__model_score | 0 | old_strict | 10 | 30 | 139.06% | -7.42% | 36.67% | 0.75% |
| strategy3__bull7_v3_full_green__score68__prionone__top3__model_score | 0 | plan_p1 | 0 | 0 | 0.00% | 0.00% |  |  |
| strategy3__bull7_v3_full_green__score68__prionone__top3__model_score | 20 | old_strict | 10 | 30 | 83.59% | -7.80% | 36.67% | 0.55% |
| strategy3__bull7_v3_full_green__score68__prionone__top3__model_score | 20 | plan_p1 | 0 | 0 | 0.00% | 0.00% |  |  |

## Best Plan Rules By Cost

| cost_bps | rule | active_days | ann_return | max_dd | score |
|---:|---|---:|---:|---:|---:|
| 0 | all__none__score66__prionone__top3__model_priority_80_20 | 19 | 2433.89% | -6.09% | 2433.284 |
| 0 | all__none__score66__prio60__top3__model_priority_80_20 | 19 | 2433.89% | -6.09% | 2433.284 |
| 0 | all__none__score66__prio65__top3__model_priority_80_20 | 19 | 2433.89% | -6.09% | 2433.284 |
| 0 | all__bull7__score66__prionone__top3__model_priority_80_20 | 19 | 2433.89% | -6.09% | 2433.284 |
| 0 | all__bull7__score66__prio60__top3__model_priority_80_20 | 19 | 2433.89% | -6.09% | 2433.284 |
| 10 | all__none__score66__prionone__top3__model_priority_80_20 | 19 | 1875.10% | -6.37% | 1874.465 |
| 10 | all__none__score66__prio60__top3__model_priority_80_20 | 19 | 1875.10% | -6.37% | 1874.465 |
| 10 | all__none__score66__prio65__top3__model_priority_80_20 | 19 | 1875.10% | -6.37% | 1874.465 |
| 10 | all__bull7__score66__prionone__top3__model_priority_80_20 | 19 | 1875.10% | -6.37% | 1874.465 |
| 10 | all__bull7__score66__prio60__top3__model_priority_80_20 | 19 | 1875.10% | -6.37% | 1874.465 |
| 20 | all__none__score66__prionone__top3__model_priority_80_20 | 19 | 1439.16% | -6.66% | 1438.493 |
| 20 | all__none__score66__prio60__top3__model_priority_80_20 | 19 | 1439.16% | -6.66% | 1438.493 |
| 20 | all__none__score66__prio65__top3__model_priority_80_20 | 19 | 1439.16% | -6.66% | 1438.493 |
| 20 | all__bull7__score66__prionone__top3__model_priority_80_20 | 19 | 1439.16% | -6.66% | 1438.493 |
| 20 | all__bull7__score66__prio60__top3__model_priority_80_20 | 19 | 1439.16% | -6.66% | 1438.493 |
| 30 | all__none__score66__prionone__top3__model_priority_80_20 | 19 | 1099.14% | -6.95% | 1098.446 |
| 30 | all__none__score66__prio60__top3__model_priority_80_20 | 19 | 1099.14% | -6.95% | 1098.446 |
| 30 | all__none__score66__prio65__top3__model_priority_80_20 | 19 | 1099.14% | -6.95% | 1098.446 |
| 30 | all__bull7__score66__prionone__top3__model_priority_80_20 | 19 | 1099.14% | -6.95% | 1098.446 |
| 30 | all__bull7__score66__prio60__top3__model_priority_80_20 | 19 | 1099.14% | -6.95% | 1098.446 |

## Artifacts

- Summary JSON: `.cache\hard_filter_plan_comparison_smoke_aug\summary.json`
- Rule comparison: `.cache\hard_filter_plan_comparison_smoke_aug\rule_comparison.csv`
- Candidate counts: `.cache\hard_filter_plan_comparison_smoke_aug\candidate_counts.csv`
- Selected samples: `.cache\hard_filter_plan_comparison_smoke_aug\selected_samples.csv`
