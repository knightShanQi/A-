# Hard Filter Plan Backtest Comparison 2026-05-31

## Scope

- Window: `2025-01-02` to `2025-01-31`.
- New strategy implementation: P1 proxy from `strategy_hard_filter_optimization_plan_2026-05-31.md` using available ten-year daily cache features.
- Ranking and labels: existing ten-year `model_scores.csv`; no model retraining in this run.
- Caveat: ten-year cache does not include true ten-year intraday, fund-flow or news features, so those plan terms are not included here.

## Candidate Breadth

- `old_strict` active days: `18`; avg daily candidates: `14.6`; median: `14.0`; p90: `23.2`.
- `plan_p1` active days: `18`; avg daily candidates: `137.9`; median: `79.5`; p90: `300.5`.

## Key Same-Rule Comparison

| rule | cost_bps | source | active_days | selected | ann_return | max_dd | win_rate | avg_trade |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| all__bull7_v3_full_green__score68__prionone__top3__model_score | 0 | old_strict | 0 | 0 | 0.00% | 0.00% |  |  |
| all__bull7_v3_full_green__score68__prionone__top3__model_score | 0 | plan_p1 | 0 | 0 | 0.00% | 0.00% |  |  |
| all__bull7_v3_full_green__score68__prionone__top3__model_score | 20 | old_strict | 0 | 0 | 0.00% | 0.00% |  |  |
| all__bull7_v3_full_green__score68__prionone__top3__model_score | 20 | plan_p1 | 0 | 0 | 0.00% | 0.00% |  |  |
| all__v3_full_green__score68__prionone__top3__model_score | 0 | old_strict | 0 | 0 | 0.00% | 0.00% |  |  |
| all__v3_full_green__score68__prionone__top3__model_score | 0 | plan_p1 | 0 | 0 | 0.00% | 0.00% |  |  |
| all__v3_full_green__score68__prionone__top3__model_score | 20 | old_strict | 0 | 0 | 0.00% | 0.00% |  |  |
| all__v3_full_green__score68__prionone__top3__model_score | 20 | plan_p1 | 0 | 0 | 0.00% | 0.00% |  |  |
| strategy3__bull7_v3_full_green__score68__prionone__top3__model_score | 0 | old_strict | 0 | 0 | 0.00% | 0.00% |  |  |
| strategy3__bull7_v3_full_green__score68__prionone__top3__model_score | 0 | plan_p1 | 0 | 0 | 0.00% | 0.00% |  |  |
| strategy3__bull7_v3_full_green__score68__prionone__top3__model_score | 20 | old_strict | 0 | 0 | 0.00% | 0.00% |  |  |
| strategy3__bull7_v3_full_green__score68__prionone__top3__model_score | 20 | plan_p1 | 0 | 0 | 0.00% | 0.00% |  |  |

## Best Plan Rules By Cost

| cost_bps | rule | active_days | ann_return | max_dd | score |
|---:|---|---:|---:|---:|---:|
| 0 | all__v3_full_green__score66__prionone__top3__model_score | 0 | 0.00% | 0.00% | 0.000 |
| 0 | all__v3_full_green__score66__prionone__top3__model_priority_80_20 | 0 | 0.00% | 0.00% | 0.000 |
| 0 | all__v3_full_green__score66__prio60__top3__model_score | 0 | 0.00% | 0.00% | 0.000 |
| 0 | all__v3_full_green__score66__prio60__top3__model_priority_80_20 | 0 | 0.00% | 0.00% | 0.000 |
| 0 | all__v3_full_green__score66__prio65__top3__model_score | 0 | 0.00% | 0.00% | 0.000 |
| 10 | all__v3_full_green__score66__prionone__top3__model_score | 0 | 0.00% | 0.00% | 0.000 |
| 10 | all__v3_full_green__score66__prionone__top3__model_priority_80_20 | 0 | 0.00% | 0.00% | 0.000 |
| 10 | all__v3_full_green__score66__prio60__top3__model_score | 0 | 0.00% | 0.00% | 0.000 |
| 10 | all__v3_full_green__score66__prio60__top3__model_priority_80_20 | 0 | 0.00% | 0.00% | 0.000 |
| 10 | all__v3_full_green__score66__prio65__top3__model_score | 0 | 0.00% | 0.00% | 0.000 |
| 20 | all__v3_full_green__score66__prionone__top3__model_score | 0 | 0.00% | 0.00% | 0.000 |
| 20 | all__v3_full_green__score66__prionone__top3__model_priority_80_20 | 0 | 0.00% | 0.00% | 0.000 |
| 20 | all__v3_full_green__score66__prio60__top3__model_score | 0 | 0.00% | 0.00% | 0.000 |
| 20 | all__v3_full_green__score66__prio60__top3__model_priority_80_20 | 0 | 0.00% | 0.00% | 0.000 |
| 20 | all__v3_full_green__score66__prio65__top3__model_score | 0 | 0.00% | 0.00% | 0.000 |
| 30 | all__v3_full_green__score66__prionone__top3__model_score | 0 | 0.00% | 0.00% | 0.000 |
| 30 | all__v3_full_green__score66__prionone__top3__model_priority_80_20 | 0 | 0.00% | 0.00% | 0.000 |
| 30 | all__v3_full_green__score66__prio60__top3__model_score | 0 | 0.00% | 0.00% | 0.000 |
| 30 | all__v3_full_green__score66__prio60__top3__model_priority_80_20 | 0 | 0.00% | 0.00% | 0.000 |
| 30 | all__v3_full_green__score66__prio65__top3__model_score | 0 | 0.00% | 0.00% | 0.000 |

## Artifacts

- Summary JSON: `.cache\hard_filter_plan_comparison_smoke_jan\summary.json`
- Rule comparison: `.cache\hard_filter_plan_comparison_smoke_jan\rule_comparison.csv`
- Candidate counts: `.cache\hard_filter_plan_comparison_smoke_jan\candidate_counts.csv`
- Selected samples: `.cache\hard_filter_plan_comparison_smoke_jan\selected_samples.csv`
