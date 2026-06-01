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
| strategy1 | 3 | 174521 | 33188 | 0.267 | 0.578 | 0.2018 | 0.0318 | 0.15% | 0.73% | 36.25% | 0.189/0.243/0.335 |
| strategy1 | 5 | 92889 | 18321 | 0.229 | 0.581 | 0.1859 | 0.0281 | 0.27% | 1.47% | 34.10% | 0.162/0.215/0.299 |
| strategy2 | 1 | 13689 | 3367 | 0.517 | 0.595 | 0.2414 | 0.0107 | 1.26% | 2.69% | 65.88% | 0.428/0.544/0.615 |
| strategy2 | 3 | 28292 | 6802 | 0.286 | 0.531 | 0.2163 | 0.0389 | 1.12% | 0.63% | 30.98% | 0.209/0.279/0.330 |
| strategy3 | 3 | 62158 | 13060 | 0.272 | 0.571 | 0.2010 | 0.0305 | 0.29% | 0.77% | 36.22% | 0.184/0.242/0.340 |
| strategy3 | 5 | 31921 | 6916 | 0.239 | 0.564 | 0.1890 | 0.0306 | 0.34% | 0.65% | 33.67% | 0.178/0.217/0.287 |

## Calibration Buckets

The full decile table is written to `strategy_calibration_bins.csv`.

| strategy | horizon | top_bucket_samples | avg_probability | realized_rate | avg_return |
|---|---:|---:|---:|---:|---:|
| strategy1 | 3 | 3319 | 0.348 | 0.362 | 0.73% |
| strategy1 | 5 | 1832 | 0.312 | 0.341 | 1.47% |
| strategy2 | 1 | 337 | 0.618 | 0.659 | 2.69% |
| strategy2 | 3 | 681 | 0.332 | 0.310 | 0.63% |
| strategy3 | 3 | 1306 | 0.352 | 0.362 | 0.77% |
| strategy3 | 5 | 692 | 0.296 | 0.337 | 0.65% |

## Artifacts

- Summary JSON: `.cache\hard_filter_plan_strategy_models\summary.json`
- Metrics: `.cache\hard_filter_plan_strategy_models\strategy_model_metrics.csv`
- Calibration bins: `.cache\hard_filter_plan_strategy_models\strategy_calibration_bins.csv`
- Test predictions: `.cache\hard_filter_plan_strategy_models\strategy_model_predictions.csv`
- Model bundle: `.cache\hard_filter_plan_strategy_models\strategy_model_bundle.pkl`
