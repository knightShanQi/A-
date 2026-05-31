# Next-Day A-Share Factor Model

This pipeline uses the local DuckDB daily-price store to build a point-in-time
next-trading-day direction model for A-shares.

## Data And Target

- Source: `data/openclaw_market_data.duckdb`
- Table: `a_share_daily_prices`
- Symbol scope: common A-share prefixes `0`, `3`, and `6`
- Analysis window in the latest full run: `2006-05-26` to `2026-05-26`
- Eligible symbols in the latest full cross-section run: `5226`
- Target: next trading day close-to-close return is greater than `0`

Historical rows in this local store are partly close-only. The feature builder
therefore uses close-derived range proxies when open/high/low are absent and
uses turnover or rolling activity proxies when volume is absent. This keeps the
20-year window usable without leaking future information.

## Method

1. Build daily technical factors per symbol from historical price paths.
2. Create the T+1 binary label from the next trading day's close-to-close return.
3. Use the chronological split below:
   - Train: up to `2020-05-18`
   - Validation: `2020-05-19` to `2023-05-18`
   - Test: `2023-05-19` to `2026-05-26`
4. Rank factors with a composite score:
   - ExtraTrees impurity importance
   - mutual information with the T+1 label
   - absolute logistic coefficient on standardized factors
   - market-state cluster lift from MiniBatchKMeans
5. Cluster correlated factors with agglomerative clustering and select the best
   representative from each cluster until 10 factors are chosen.
6. Train a streaming logistic model with `SGDClassifier` on the selected 10
   factors. If validation AUC is below 0.5, the saved model records a probability
   orientation flip so higher output remains aligned with higher T+1 up odds.
7. Select the decision threshold on the validation window to maximize accuracy
   while preserving at least a small positive-prediction rate. The saved bundle
   records this threshold and the majority-class baseline for audit.

The current enhanced run adds cross-sectional candidates by default:

- market/index proxy momentum and activity from all A-share constituents
- industry rotation and industry activity/amount-ratio ranks from
  `.cache/baidu_daily_stock/stock_basic.csv`
- main-board/ChiNext/STAR segment rotation and activity ranks
- ST exclusion by default
- suspension/resumption proxies from trade-date gaps
- board-aware limit-up/limit-down flags and short limit streaks

The local DuckDB store does not contain historical index futures or true
northbound/main-fund-flow tables, so the current implementation uses available
amount, turnover, and cross-sectional activity proxies instead of pretending
those feeds exist.

## Cross-Section Enhanced Top 10 Factors

1. `limit_up_flag`
2. `limit_down_flag`
3. `market_activity_ratio_5`
4. `market_turnover_20`
5. `segment_rank_activity_5`
6. `range_position_20`
7. `trade_gap_days`
8. `recent_resume_flag`
9. `turnover_ratio_20`
10. `limit_up_streak_3`

## Cross-Section Enhanced Metrics

- Train samples: `7,300,243`
- Validation samples: `3,023,949`
- Test samples: `3,547,078`
- Candidate features: `90`
- Excluded ST symbols: `179`
- Validation ROC AUC: `0.500021236366942`
- Test ROC AUC: `0.4905663699256392`
- Validation decision threshold: `0.5777728351180096`
- Test accuracy: `0.5230065422863551`
- Test majority-class baseline: `0.5220293999737249`
- Test accuracy lift vs majority: `0.0009771423126302414`
- Previous threshold-only test accuracy: `0.521703735185426`
- Test top-decile next-day return: `0.000013578505647095488`

The enhanced model is now the best run by the requested classification accuracy
metric, beating both the previous threshold-tuned run and its own majority-class
baseline. It should still be treated as a direction classifier rather than a
return-ranking engine: the AUC is below 0.5 after validation-time probability
orientation handling, and the test top-decile return is not compelling.

## Previous Threshold-Only Top 10 Factors

1. `range_position_20`
2. `ma_alignment_score`
3. `volume_ratio_5`
4. `ret_5`
5. `close_near_high_5`
6. `ret_1`
7. `efficiency_ratio_10`
8. `consolidation_width_20`
9. `up_day_ratio_10`
10. `close_vs_ma120`

## Previous Threshold-Only Metrics

- Train samples: `7,648,065`
- Validation samples: `3,147,268`
- Test samples: `3,675,266`
- Validation ROC AUC: `0.5045319516808409`
- Test ROC AUC: `0.5049014711640788`
- Validation decision threshold: `0.6820098115056004`
- Test accuracy: `0.521703735185426`
- Test majority-class baseline: `0.5222843734303857`
- Previous 0.5-threshold test accuracy: `0.504906039454015`
- Test top-decile next-day return: `0.000552165532986129`

The previous threshold-tuned model improved raw 0.5-threshold accuracy by about
1.68 percentage points, but it did not beat the majority-class baseline on the
test window.

## Reproduce

From `E:\openclaw`:

```powershell
.\.venv\Scripts\python.exe scripts\train_next_day_factor_model.py `
  --output-dir .cache\next_day_factor_model_cross_section `
  --analysis-sample-limit 500000 `
  --importance-sample-limit 220000 `
  --batch-symbols 240 `
  --train-epochs 1 `
  --sample-model-max-rows 0 `
  --min-positive-prediction-rate 0.02
```

After reinstalling the package, the console entry point is also available:

```powershell
.\.venv\Scripts\a-share-next-day-factor-model.exe --output-dir .cache\next_day_factor_model
```

The latest cross-section enhanced run is stored in
`.cache\next_day_factor_model_cross_section`. The previous threshold-only run is
stored in `.cache\next_day_factor_model_threshold_only`, and the earlier
market-context ablation is stored in `.cache\next_day_factor_model_optimized`.

The newer one-year multi-source clustering pass, including intraday coverage
audit and selected factors, is documented in
`docs\one_year_multisource_factor_cluster.md`.

## Artifacts

- `.cache\next_day_factor_model_cross_section\selected_top10_factors.csv`
- `.cache\next_day_factor_model_cross_section\feature_ranking.csv`
- `.cache\next_day_factor_model_cross_section\feature_cluster_summary.csv`
- `.cache\next_day_factor_model_cross_section\market_state_cluster_summary.csv`
- `.cache\next_day_factor_model_cross_section\next_day_top10_factor_model.pkl`
- `.cache\next_day_factor_model_cross_section\model_metrics.json`
- `.cache\next_day_factor_model_cross_section\training_report.md`
