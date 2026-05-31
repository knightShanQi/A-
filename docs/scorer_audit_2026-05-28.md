# Scorer Audit 2026-05-28

## Purpose

Validate whether stage/launch/rank scorers have replay evidence behind them, and identify where the system is still making score-driven decisions without durable post-trade observability.

## Coverage

- Selected review files: 23
- Selected review rows: 989
- Rows with `launch_window_score`: 525
- Rows with `final_rank_score`: 525
- Rows with `execution_score`: 525
- Rows with `selection_score`: 525
- Rows with `stage_score` directly in review details: 450
- Rows where `stage_score` could be joined from feature-store snapshots: 75

## Key Findings

- In replay samples with launch-window data, the high launch-window bucket averages 0.65% next-day return versus 0.07% for the low bucket.
- Where final-rank replay data exists, the high final-rank bucket averages 1.17% next-day return versus -0.01% for the low bucket.
- On dates where stage-score joins are available, the high stage-score bucket averages -0.10% next-day return versus 0.11% for the low bucket.
- In replay samples with execution-score data, the high execution-score bucket averages 0.20% next-day return versus 0.23% for the low bucket.
- In replay samples with selection-score data, the high selection-score bucket averages 0.77% next-day return versus 0.22% for the low bucket.
- Replay-detail observability now covers `execution_score` on 525 / 989 rows, `selection_score` on 525 / 989 rows, and `stage_score` on 450 / 989 rows.
- Execution-level scoring is still only partially auditable from replay caches, so mixed-version history can bias any scorer-level conclusion.
- Stage-level scores are still incomplete in replay details, so the end-to-end evidence trail remains mixed until more v9 history is regenerated.

## Launch Window Buckets

| launch_window_bucket   |   sample_rows |   avg_next_day_return_pct |   avg_intraday_high_return_pct |   win_rate |   direction_hit_rate |
|:-----------------------|--------------:|--------------------------:|-------------------------------:|-----------:|---------------------:|
| high                   |           175 |                 0.649829  |                        3.69333 |   0.531429 |             0.531429 |
| low                    |           175 |                 0.0653143 |                        3.33813 |   0.417143 |             0.502857 |
| mid                    |           175 |                 1.73749   |                        4.9575  |   0.565714 |             0.565714 |

## Final Rank Buckets

| final_rank_bucket   |   sample_rows |   avg_next_day_return_pct |   avg_intraday_high_return_pct |   win_rate |   direction_hit_rate |
|:--------------------|--------------:|--------------------------:|-------------------------------:|-----------:|---------------------:|
| high                |           175 |                 1.16914   |                        2.82071 |   0.531429 |             0.531429 |
| low                 |           175 |                -0.0119429 |                        3.31312 |   0.405714 |             0.491429 |
| mid                 |           175 |                 1.29543   |                        4.1837  |   0.577143 |             0.577143 |

## Execution Score Buckets

| execution_score_bucket   |   sample_rows |   avg_next_day_return_pct |   avg_intraday_high_return_pct |   win_rate |   direction_hit_rate |
|:-------------------------|--------------:|--------------------------:|-------------------------------:|-----------:|---------------------:|
| high                     |           175 |                  0.199029 |                        3.45627 |   0.462857 |             0.462857 |
| low                      |           175 |                  0.229943 |                        3.31312 |   0.451429 |             0.508571 |
| mid                      |           175 |                  2.02366  |                        3.9175  |   0.6      |             0.628571 |

## Selection Score Buckets

| selection_score_bucket   |   sample_rows |   avg_next_day_return_pct |   avg_intraday_high_return_pct |   win_rate |   direction_hit_rate |
|:-------------------------|--------------:|--------------------------:|-------------------------------:|-----------:|---------------------:|
| high                     |           175 |                  0.770686 |                       12.08    |   0.52     |             0.52     |
| low                      |           175 |                  0.2152   |                        3.50361 |   0.411429 |             0.497143 |
| mid                      |           175 |                  1.46674  |                        2.50333 |   0.582857 |             0.582857 |

## Stage Score Buckets

| stage_score_bucket   |   sample_rows |   avg_next_day_return_pct |   avg_intraday_high_return_pct |   win_rate |   direction_hit_rate |
|:---------------------|--------------:|--------------------------:|-------------------------------:|-----------:|---------------------:|
| high                 |            25 |                   -0.104  |                           0.54 |       0.36 |                 0.36 |
| low                  |            25 |                    0.1132 |                          10    |       0.48 |                 0.44 |
| mid                  |            25 |                   -2.2076 |                         nan    |       0.2  |                 0.2  |

## Stage Code Breakdown

| stage_code            | stage_label   |   sample_rows |   avg_next_day_return_pct |   avg_intraday_high_return_pct |   win_rate |
|:----------------------|:--------------|--------------:|--------------------------:|-------------------------------:|-----------:|
| breakout_confirmation | 平台突破确认        |            34 |                 -1.58235  |                           0.54 |   0.264706 |
| range_monitor         | 强势延续快榜        |            12 |                 -0.199167 |                          10    |   0.333333 |
| range_monitor         | 区间震荡观察        |            12 |                 -0.305833 |                         nan    |   0.416667 |
| trend_acceleration    | 强势延续快榜        |             9 |                  1.08444  |                         nan    |   0.444444 |
| trend_acceleration    | 趋势主升加速        |             4 |                 -1.9625   |                         nan    |   0.25     |
| distribution_risk     | 高位分歧派发        |             2 |                  1.235    |                         nan    |   1        |
| distribution_risk     | 强势延续快榜        |             1 |                  1.66     |                         nan    |   1        |
| breakout_confirmation | 强势延续快榜        |             1 |                 -1.14     |                         nan    |   0        |

## Execution Label Breakdown

| execution_label   |   sample_rows |   avg_next_day_return_pct |   avg_intraday_high_return_pct |   win_rate |   direction_hit_rate |
|:------------------|--------------:|--------------------------:|-------------------------------:|-----------:|---------------------:|
| 可执行               |           480 |                  0.832562 |                        3.42483 |   0.514583 |             0.525    |
| 临门观察              |            27 |                  1.22111  |                      nan       |   0.518519 |             0.518519 |
| 等待结构              |            18 |                 -0.188333 |                      nan       |   0.222222 |             0.777778 |

## Interpretation

- `launch_window_score` now has enough replay coverage to be judged as a real ranking aid rather than a pure narrative field.
- `final_rank_score` has partial replay coverage only in the newest review subset, so evidence is directional but not yet durable.
- `execution_score` and `selection_score` now have enough replay rows for first-pass bucket analysis, but their distributions are saturated near 100, so monotonicity needs to be read cautiously.
- `execution_score`, `selection_score`, and `stage_score` are now persisted in v9 review artifacts, but historical evidence remains mixed until more legacy windows are regenerated.
- `stage_score` can now be audited from either replay details or joined feature snapshots, which removes the prior hard observability gap but does not by itself prove positive alpha.

## Next Actions

1. Keep regenerating v9 review history so scorer coverage is dominated by post-persistence artifacts rather than mixed legacy caches.
2. Extend replay summaries to segment by `execution_window`, `execution_label`, and confidence fields, not just raw score buckets.
3. Remove any scorer from default gating/ranking if it cannot show monotonic replay improvement after v9 coverage is broad enough.

Review file index: `E:\openclaw\.cache\trading_system_attribution\selected_review_files.csv`
Launch summary: `E:\openclaw\.cache\trading_system_attribution\scorer_launch_window_summary.csv`
Final-rank summary: `E:\openclaw\.cache\trading_system_attribution\scorer_final_rank_summary.csv`
Execution-score summary: `E:\openclaw\.cache\trading_system_attribution\scorer_execution_score_summary.csv`
Execution-label summary: `E:\openclaw\.cache\trading_system_attribution\scorer_execution_label_summary.csv`
Selection-score summary: `E:\openclaw\.cache\trading_system_attribution\scorer_selection_score_summary.csv`
Stage-score summary: `E:\openclaw\.cache\trading_system_attribution\scorer_stage_score_summary.csv`
Stage-code summary: `E:\openclaw\.cache\trading_system_attribution\scorer_stage_code_summary.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\scorer_audit.json`