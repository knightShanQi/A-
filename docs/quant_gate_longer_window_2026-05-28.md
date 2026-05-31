# Quant Gate Longer Window Report 2026-05-28

## Purpose

Retest the promising `quant_score >= 58` gate on a longer v9 review-history slice using a daily next-day-return proxy.

## Coverage

- Review rows: 525
- Review files: 13

## Key Findings

- Across the longer v9 review slice, baseline `selection_top3` proxy annualized return is about -75.47%.
- The mild gate `quant>=58` proxy annualized return is about -64.49% with drawdown proxy -22.49%.
- The stricter gate `quant>=68` proxy annualized return is about -80.33%, showing whether the mild-gate edge survives stricter filtering.
- This longer-window proxy does not replace the unified portfolio engine, but it is the right next confidence check before treating quant>=58 as a candidate research rule.

## Proxy Summary

| variant                    |   board_dates |   candidate_rows |   avg_selected_quant_score |   avg_selected_selection_score |   cumulative_return |   annualized_return_proxy |   max_drawdown_proxy |   win_rate |   avg_next_day_return |
|:---------------------------|--------------:|-----------------:|---------------------------:|-------------------------------:|--------------------:|--------------------------:|---------------------:|-----------:|----------------------:|
| selection_top3             |            13 |               39 |                    91.9277 |                        97.4325 |          -0.0699331 |                 -0.754721 |           -0.224936  |   0.538462 |           -0.00437385 |
| selection_top3_quant_ge_58 |            13 |               38 |                    93.0263 |                        97.3946 |          -0.0520104 |                 -0.644904 |           -0.224936  |   0.538462 |           -0.00288417 |
| selection_top3_quant_ge_68 |            12 |               36 |                    94.7036 |                        97.5014 |          -0.0745183 |                 -0.803336 |           -0.224936  |   0.5      |           -0.0051512  |
| selection_top3_quant_le_45 |             2 |                6 |                    41.36   |                        81.9963 |          -0.0297623 |                 -0.977785 |           -0.0210968 |   0        |           -0.0149745  |

## Interpretation

- This is a longer-window replay proxy, not a replacement for the unified portfolio engine. It is useful for checking whether the mild quant gate is directionally stable beyond the 13 snapshot days.
- If the mild gate still helps here while the strict gate degrades, that supports the idea that quant is useful mainly for removing obvious weak candidates rather than for aggressive overfiltering.
- If the mild gate collapses here, then the short-slice improvement should be treated as fragile and not upgraded into research defaults yet.

## Next Actions

1. If quant>=58 still looks directionally positive here, run it next on a longer unified-portfolio candidate source before adopting it.
2. If it does not, demote quant back to a lower-priority ingredient and move on to execution-score source-level cleanup.

Summary CSV: `E:\openclaw\.cache\trading_system_attribution\quant_gate_longer_window_summary.csv`
Daily CSV: `E:\openclaw\.cache\trading_system_attribution\quant_gate_longer_window_daily.csv`
JSON: `E:\openclaw\.cache\trading_system_attribution\quant_gate_longer_window.json`