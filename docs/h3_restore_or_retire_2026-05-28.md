# h3 Restore Or Retire 2026-05-28

## Purpose

Turn the model-source availability evidence into a concrete configuration decision for the current default short-horizon trading path.

## Key Findings

- Current UI default = `h3 / 10%`, from `DEFAULT_VIEW_PARAMS`.
- Default path availability: market model loaded = False, proxy loaded = False.
- Reference `h5 / 3%` availability: market model loaded = False, proxy loaded = True.
- Recommendation: `retire_default_until_model_restored`.
- The current default path (`h3 / 10%`) has neither a market-wide model nor a proxy model available, so the UI default is structurally forced into local fallback mode.
- The only matching maintained training entrypoint in the repo (`scripts/train_market_wide_model.py`) hardcodes `h5 / 3%`, which means the workspace currently optimizes and refreshes a different configuration from the default UI path.
- An optimization program that continues pruning score layers on `h3 / 10%` before restoring model-source support is mostly auditing a degraded fallback architecture.

## Summary

| label                     |   horizon_days |   positive_return | market_model_ready   | partial_dataset_ready   | market_model_loaded   | proxy_model_loaded   | model_path                                                                                 | partial_path                                                                                         |
|:--------------------------|---------------:|------------------:|:---------------------|:------------------------|:----------------------|:---------------------|:-------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------|
| current_default           |              3 |              0.1  | False                | False                   | False                 | False                | E:\openclaw\.cache\global_market_model_v4_h3_r1000_20250101_20251231_20260101_20260331.pkl | E:\openclaw\.cache\global_market_dataset_v4_h3_r1000_20250101_20251231_20260101_20260331.partial.pkl |
| available_proxy_reference |              5 |              0.03 | False                | False                   | False                 | True                 | E:\openclaw\.cache\global_market_model_v4_h5_r300_20250101_20251231_20260101_20260331.pkl  | E:\openclaw\.cache\global_market_dataset_v4_h5_r300_20250101_20251231_20260101_20260331.partial.pkl  |

## Recommendation

- Retire `h3 / 10%` as the default production trading configuration for now.
- If you want to preserve it as a research track, do so only after restoring a compatible market-wide or proxy model for that exact horizon/target setting.
- Until then, route default trading analysis to a configuration that actually has model support, such as `h5 / 3%`, or explicitly label the current path as fallback-only.

## Next Actions

1. Either train/restore a compatible `h3 / 10%` market model or proxy, or change the default configuration away from that unsupported path.
2. Only after that decision should further score-layer simplification on the default path be treated as high-priority optimization work.

Summary CSV: `E:\openclaw\.cache\trading_system_attribution\h3_restore_or_retire_summary.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\h3_restore_or_retire.json`
