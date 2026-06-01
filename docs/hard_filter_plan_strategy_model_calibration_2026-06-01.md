# Hard Filter Plan Strategy Model Retraining And Calibration

## Scope

- Strategy plan: `E:\openclaw\docs\strategy_hard_filter_optimization_plan_2026-05-31.md`
- Candidate source: `.cache\hard_filter_plan_comparison_full\plan_scored_candidates.csv`
- History source: `.cache\hard_filter_plan_comparison\prepared_fast_history_v1_20160527_20260526.pkl`
- Candidate rows after labels: `264971`
- Training mode: per-strategy logistic meta-model plus Platt probability calibration.
- Time split: train, calibration, and test are separated by market date.

## Per-Strategy Metrics

| strategy | horizon | samples | test | pos_rate | AUC | Brier | calibration_gap | avg_return | top_bucket_return | top_bucket_win_rate | p05/p50/p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| strategy1 | 3 | 174521 | 33188 | 0.266 | 0.587 | 0.2017 | 0.0282 | 0.27% | 1.87% | 39.65% | 0.189/0.246/0.360 |
| strategy1 | 5 | 174417 | 33096 | 0.236 | 0.573 | 0.1929 | 0.0413 | 0.34% | 1.44% | 33.87% | 0.163/0.213/0.295 |
| strategy2 | 1 | 28292 | 6802 | 0.520 | 0.594 | 0.2422 | 0.0072 | 1.30% | 2.93% | 69.90% | 0.425/0.542/0.610 |
| strategy2 | 3 | 28292 | 6802 | 0.308 | 0.559 | 0.2259 | 0.0407 | 1.96% | 2.60% | 39.21% | 0.213/0.314/0.378 |
| strategy3 | 3 | 62158 | 13060 | 0.270 | 0.576 | 0.2018 | 0.0305 | 0.35% | 1.63% | 38.90% | 0.182/0.243/0.356 |
| strategy3 | 5 | 62103 | 13018 | 0.242 | 0.575 | 0.1919 | 0.0342 | 0.50% | 1.35% | 35.02% | 0.169/0.219/0.304 |

## Calibration Buckets

The full decile table is written to `strategy_calibration_bins.csv`.

| strategy | horizon | top_bucket_samples | avg_probability | realized_rate | avg_return |
|---|---:|---:|---:|---:|---:|
| strategy1 | 3 | 3319 | 0.372 | 0.397 | 1.87% |
| strategy1 | 5 | 3310 | 0.310 | 0.339 | 1.44% |
| strategy2 | 1 | 681 | 0.614 | 0.699 | 2.93% |
| strategy2 | 3 | 681 | 0.382 | 0.392 | 2.60% |
| strategy3 | 3 | 1306 | 0.366 | 0.389 | 1.63% |
| strategy3 | 5 | 1302 | 0.316 | 0.350 | 1.35% |

## Artifacts

- Summary JSON: `.cache\hard_filter_plan_strategy_models\summary.json`
- Metrics: `.cache\hard_filter_plan_strategy_models\strategy_model_metrics.csv`
- Calibration bins: `.cache\hard_filter_plan_strategy_models\strategy_calibration_bins.csv`
- Test predictions: `.cache\hard_filter_plan_strategy_models\strategy_model_predictions.csv`
- Model bundle: `.cache\hard_filter_plan_strategy_models\strategy_model_bundle.pkl`
