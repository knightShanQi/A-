# Final Rank Construction Report 2026-05-28

## Purpose

Verify whether `final_rank_score` is actually a distinct ranking signal on persisted snapshots or just a renamed variant of an existing score.

## Coverage

- Snapshot rows: 527
- Distinct board dates: 13

## Key Findings

- On the current v9 snapshot slice, `final_rank_score` and `enhanced_attention_score` are exactly equal on every persisted row.
- That means the current `final_rank_score` path is not a second-ranking opinion; it is a renamed attention-style score.
- Because of that identity, any portfolio result from `final_rank_score_top3` is mechanically the same as `enhanced_attention_score_top3` until the construction formula changes.
- By contrast, `selection_score` is only correlated with `final_rank_score` rather than identical, which is why it can still produce a different and currently stronger top-3 path.

## Correlation Summary

| score_col                |   notna_rows |   correlation_to_final_rank |   mean_score |
|:-------------------------|-------------:|----------------------------:|-------------:|
| enhanced_attention_score |          527 |                    1        |      86.0658 |
| selection_score          |          527 |                    0.906747 |      90.8511 |
| launch_window_score      |          527 |                    0.657927 |      83.072  |
| probability_up           |          527 |                    0.873202 |      71.2227 |

## Top-3 Overlap Summary

| compare_col              |   comparable_dates |   identical_days |   avg_overlap_count |
|:-------------------------|-------------------:|-----------------:|--------------------:|
| enhanced_attention_score |                 13 |               13 |             3       |
| selection_score          |                 13 |                2 |             1.07692 |
| launch_window_score      |                 13 |                1 |             1       |

## Interpretation

- The current engineering problem with `final_rank_score` is no longer persistence. It is construction redundancy.
- Promoting `final_rank_score` without changing how it is built would only rename the weaker attention-style ranking path, not create a better one.
- If a future `final_rank_score` is meant to express a richer view, it needs genuinely new ingredients or weights that move it away from exact equality with `enhanced_attention_score` and then beat `selection_score_top3` in the same engine.

## Next Actions

1. Audit every place where `final_rank_score` is assigned directly from `enhanced_attention_score` and decide whether that alias should be removed.
2. If `final_rank_score` is supposed to be a composite, rebuild it explicitly from differentiated inputs and rerun the ranking-quality portfolio report.
3. Until that happens, treat `selection_score_top3` as the only current default research ranking baseline on the v9 slice.

Correlation CSV: `E:\openclaw\.cache\trading_system_attribution\final_rank_correlation_summary.csv`
Top-3 overlap CSV: `E:\openclaw\.cache\trading_system_attribution\final_rank_top3_overlap_summary.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\final_rank_construction.json`