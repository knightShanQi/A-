# Model Source Availability 2026-05-28

## Purpose

Audit whether the audited trading paths are actually running on the same model-source architecture, or whether some configurations are forced into fallback mode before any ranking/scoring logic is applied.

## Key Findings

- The current focus-board audit configuration (`h3 / 10%`) has no market-wide model and no proxy model available: market model ready = False, proxy loaded = False.
- The older default-style configuration (`h5 / 3%`) still has no full market-wide model ready, but it does load a proxy model = True.
- That means the current short-horizon audit path is structurally forced into `local_fast_fallback`, while the older longer-horizon path can at least retain a proxy-model-based ranking layer.
- This is not just a modeling-quality issue. It is a model-availability issue: different horizon/target settings are exercising different ranking architectures in production.
- Practical implication: part of the low annualized return and weak explanation quality on the current short-horizon path may come from running the weakest fallback architecture, not only from the score formulas layered on top.

## Summary

| label                        |   horizon_days |   positive_return | market_model_ready   | partial_dataset_ready   | market_model_loaded   | proxy_model_loaded   | model_path                                                                                 | partial_path                                                                                         |   completed_symbol_count |   partial_symbol_count |   partial_row_count |
|:-----------------------------|---------------:|------------------:|:---------------------|:------------------------|:----------------------|:---------------------|:-------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------|-------------------------:|-----------------------:|--------------------:|
| current_focus_board_h3_r1000 |              3 |              0.1  | False                | False                   | False                 | False                | E:\openclaw\.cache\global_market_model_v4_h3_r1000_20250101_20251231_20260101_20260331.pkl | E:\openclaw\.cache\global_market_dataset_v4_h3_r1000_20250101_20251231_20260101_20260331.partial.pkl |                        0 |                      0 |                   0 |
| legacy_default_h5_r300       |              5 |              0.03 | False                | False                   | False                 | True                 | E:\openclaw\.cache\global_market_model_v4_h5_r300_20250101_20251231_20260101_20260331.pkl  | E:\openclaw\.cache\global_market_dataset_v4_h5_r300_20250101_20251231_20260101_20260331.partial.pkl  |                        0 |                      0 |                   0 |

## Interpretation

- If the short-horizon focus-board path has neither a market-wide model nor a proxy model, then score-layer audits on that path are partly auditing a degraded architecture, not the intended full stack.
- Comparing `h3 / 10%` and `h5 / 3%` directly also becomes an architecture comparison, not just a target-window comparison, because they are using different upstream model availability states.
- Any optimization plan for annualized return therefore has to separate two questions: whether the ranking formulas are weak, and whether the strongest intended model source is even present for the audited configuration.

## Next Actions

1. Treat `h3 / 10%` model-source unavailability as a first-class bottleneck in the current trading stack.
2. Before deleting more score layers on that path, decide whether the intended fix is to train/restore a compatible market model or proxy for `h3 / 10%`, or to explicitly retire that configuration.

Summary CSV: `E:\openclaw\.cache\trading_system_attribution\model_source_availability_summary.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\model_source_availability.json`
