# Final Rank Assignment Audit 2026-05-28

## Purpose

Locate where `final_rank_score` is truly created versus where it is only displayed, normalized, or persisted.

## Key Findings

- The only true source assignments for final_rank_score in the current quick-board generation paths are direct aliases from enhanced_attention_score.
- Everything downstream in api_service and daily_review mostly preserves or backfills that value rather than creating a differentiated score.
- Current decision risk therefore sits upstream in dashboard quick-board construction, not in API normalization or snapshot persistence.
- As long as those upstream alias assignments remain, any downstream tuning of final_rank_score is structurally incapable of producing a distinct ranking path.

## Site Summary

| site_type              | impact               |   site_count |
|:-----------------------|:---------------------|-------------:|
| assignment_alias       | decision_source      |            2 |
| display_fallback       | presentation         |            2 |
| display_usage          | presentation         |            2 |
| detail_passthrough     | presentation         |            1 |
| normalization_fallback | decision_passthrough |            1 |
| review_passthrough     | persistence          |            1 |
| snapshot_fallback      | persistence          |            1 |

## Site Inventory

| file                                              |   line | site_type              | impact               | summary                                                                                                                              |
|:--------------------------------------------------|-------:|:-----------------------|:---------------------|:-------------------------------------------------------------------------------------------------------------------------------------|
| E:\openclaw\src\a_share_predictor\dashboard.py    |   5004 | assignment_alias       | decision_source      | Quick-board fallback path assigns final_rank_score directly from enhanced_attention_score.                                           |
| E:\openclaw\src\a_share_predictor\dashboard.py    |   5377 | assignment_alias       | decision_source      | Latest market quick-board path assigns final_rank_score directly from enhanced_attention_score.                                      |
| E:\openclaw\src\a_share_predictor\dashboard.py    |   1261 | display_fallback       | presentation         | Merged display payload backfills final_rank_score from ranking_score / enhanced_attention_score for UI use.                          |
| E:\openclaw\src\a_share_predictor\dashboard.py    |   1606 | display_fallback       | presentation         | Symbol detail payload derives final_rank_score from ranking_score with enhanced_attention fallback.                                  |
| E:\openclaw\src\a_share_predictor\api_service.py  |    347 | normalization_fallback | decision_passthrough | API probability contract uses provided final_rank_score or falls back to ranking_score / enhanced_attention_score / attention_score. |
| E:\openclaw\src\a_share_predictor\api_service.py  |    989 | detail_passthrough     | presentation         | Safe symbol detail payload defaults final_rank_score from cached row rank_score when missing.                                        |
| E:\openclaw\src\a_share_predictor\daily_review.py |   1150 | snapshot_fallback      | persistence          | Snapshot persistence backfills final_rank_score from ranking_score -> enhanced_attention_score -> attention_score.                   |
| E:\openclaw\src\a_share_predictor\daily_review.py |   1402 | review_passthrough     | persistence          | Review detail persistence carries final_rank_score from snapshot row or ranking_score fallback.                                      |
| E:\openclaw\src\a_share_predictor\dashboard.py    |   6285 | display_usage          | presentation         | Focus-board summary table exposes final_rank_score as a displayed metric.                                                            |
| E:\openclaw\src\a_share_predictor\dashboard.py    |   6291 | display_usage          | presentation         | Focus-board detail table exposes final_rank_score alongside ranking_score for inspection.                                            |

## Interpretation

- The engineering bottleneck is concentrated in the upstream quick-board builders, where `final_rank_score` is still assigned as a plain alias of `enhanced_attention_score`.
- Downstream consumers mostly preserve that alias, so removing or rebuilding `final_rank_score` should start at the source assignments rather than at display tables or persistence hooks.
- This also explains why the final-rank construction audit showed exact equality on every persisted snapshot row.

## Next Actions

1. Remove or rename the alias assignments in the quick-board builders if `final_rank_score` is not intended to be a distinct signal.
2. If it is intended to be distinct, replace those assignments with an explicit composite formula and rerun the construction and portfolio audits.
3. Avoid using downstream fallback presence as evidence that final_rank_score has independent alpha; current code shows it does not.

Summary CSV: `E:\openclaw\.cache\trading_system_attribution\final_rank_assignment_summary.csv`
Sites CSV: `E:\openclaw\.cache\trading_system_attribution\final_rank_assignment_sites.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\final_rank_assignment_audit.json`