# Default Path Evidence Gap 2026-05-28

## Purpose

Quantify why the current default-path audit evidence and the supported-path migration evidence are not yet interchangeable.

## Key Findings

- The current default path `h3 / 10%` is operationally current in the review stack, but its latest artifacts are still generated under an unsupported model-source configuration (`最新收盘快榜（完整版特征与回测正在后台补齐）`).
- The supported candidate path `h5 / 3%` is not just stale; it is also structurally pre-modern in the audit sense. Its latest snapshot cache version is `v3` and latest review cache version is `v3`, versus `v8` / `v10` on the current default path.
- On the latest snapshot, `h3 / 10%` exposes 1 / 9 required modern audit fields, while `h5 / 3%` exposes only 0 / 9.
- On the latest review detail, `h3 / 10%` exposes 9 / 9 required modern audit fields, while `h5 / 3%` exposes only 0 / 9.
- The missing `h5 / 3%` fields are exactly the ones current remediation work depends on: snapshot missing = selection_score, final_rank_score, predicted_upside_pct, sector_score, fund_score, news_score, launch_window_score, launch_window_confidence_weight, execution_score; review missing = selection_score, final_rank_score, predicted_upside_pct, sector_score, fund_score, news_score, launch_window_score, launch_window_confidence_weight, execution_score.
- That means a default flip to the old `h5 / 3%` artifacts would improve model support but simultaneously throw away the observability needed for the current formula-level audit and simplification work.

## Summary

| label               |   horizon_days |   positive_return |   board_size |   snapshot_files |   review_files | latest_snapshot_file                              | latest_review_file                                        |   latest_snapshot_cache_version |   latest_review_cache_version | latest_snapshot_board_date   | latest_review_date   |   latest_snapshot_rows |   latest_review_rows |   latest_snapshot_columns |   latest_review_columns |   snapshot_required_fields_present |   review_required_fields_present | snapshot_missing_fields                                                                                                                                                                  | review_missing_fields                                                                                                                                                                    | latest_model_source_label   |   mean_direction_hit_rate_pct |   mean_avg_return_pct |   mean_avg_target_progress_pct |   mean_win_rate_pct |
|:--------------------|---------------:|------------------:|-------------:|-----------------:|---------------:|:--------------------------------------------------|:----------------------------------------------------------|--------------------------------:|------------------------------:|:-----------------------------|:---------------------|-----------------------:|---------------------:|--------------------------:|------------------------:|-----------------------------------:|---------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------|------------------------------:|----------------------:|-------------------------------:|--------------------:|
| h3_default_fallback |              3 |              0.1  |           50 |               59 |             63 | snapshot_v8_h3_r1000_b50_u3b971cb9f1_20260527.pkl | review_v10_h3_r1000_b50_u3b971cb9f1_20260526_20260527.pkl |                               8 |                            10 | 2026-05-27                   | 2026-05-27           |                     29 |                   29 |                        33 |                      76 |                                  1 |                                9 | ['selection_score', 'final_rank_score', 'predicted_upside_pct', 'sector_score', 'fund_score', 'news_score', 'launch_window_confidence_weight', 'execution_score']                        | []                                                                                                                                                                                       | 最新收盘快榜（完整版特征与回测正在后台补齐）      |                       53.4241 |                0.5184 |                         5.1973 |             49.1346 |
| h5_supported_proxy  |              5 |              0.03 |           50 |                9 |              4 | snapshot_v3_h5_r300_b50_u3b971cb9f1_20260415.pkl  | review_v3_h5_r300_b50_u3b971cb9f1_20260415_20260416.pkl   |                               3 |                             3 | 2026-04-15                   | 2026-04-16           |                     50 |                   50 |                        20 |                      24 |                                  0 |                                0 | ['selection_score', 'final_rank_score', 'predicted_upside_pct', 'sector_score', 'fund_score', 'news_score', 'launch_window_score', 'launch_window_confidence_weight', 'execution_score'] | ['selection_score', 'final_rank_score', 'predicted_upside_pct', 'sector_score', 'fund_score', 'news_score', 'launch_window_score', 'launch_window_confidence_weight', 'execution_score'] | 快速代理模型                      |                       46.9225 |                1.79   |                        59.5675 |             59.225  |

## Recommendation

- Treat `h5 / 3%` regeneration as a schema-and-observability rebuild, not just a cache refresh.
- Do not compare live `h3` review metrics directly against the old `h5` stack as if they were like-for-like strategy evidence; they differ in both horizon and audit schema depth.
- The correct migration order remains: regenerate fresh `h5 / 3%` artifacts under the current cache schema, confirm modern fields are present, then switch the default away from unsupported `h3 / 10%`.

## Next Actions

1. Run the supported-path regeneration entrypoint until it produces current-version `h5 / 3%` snapshot and review files with modern fields.
2. Re-run this evidence-gap report after regeneration and only treat the default switch as ready when the field-coverage gap closes.
3. After that, resume score-family pruning on the supported path instead of the unsupported fallback path.

Summary CSV: `E:\openclaw\.cache\trading_system_attribution\default_path_evidence_gap_summary.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\default_path_evidence_gap.json`