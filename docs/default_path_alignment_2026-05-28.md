# Default Path Alignment 2026-05-28

## Purpose

Check whether the default snapshot loader is returning an audit-safe board for interpreting the latest default-path review evidence.

## Key Findings

- The default snapshot loader currently resolves to `snapshot_v8_h3_r1000_b50_u3b971cb9f1_20260527.pkl` (`v8`, board date 2026-05-27).
- The latest default-path review evidence resolves to `review_v10_h3_r1000_b50_u3b971cb9f1_20260526_20260527.pkl` (`v10`, board date 2026-05-26, review date 2026-05-27).
- The review-linked snapshot for that latest review is `snapshot_v10_h3_r1000_b50_u3b971cb9f1_20260526.pkl` (`v10`, board date 2026-05-26).
- The loader-selected snapshot and the review-linked snapshot contain the same symbol set on this sample (`29 / 29` overlap, no symbol drift), so the risk is not candidate identity drift.
- The risk is interpretability drift: the loader-selected snapshot is missing 50 review-detail columns versus 33 on the review-linked snapshot.
- Even on shared columns, the two snapshots are not value-identical. 15 shared columns changed between the loader-selected `v8` board and the review-linked `v10` board, including attention/probability/launch fields.
- Practical implication: using `load_latest_snapshot_board()` as the explanation layer for the latest review can silently downgrade field coverage and mix board dates, even when the selected stock set is unchanged.

## Recommendation

- For audit and root-cause analysis, anchor on the latest review detail first, then join back to the snapshot with the same `board_date`.
- Code-level access path now exists for that rule: use `load_latest_review_bundle()` rather than separately calling `load_latest_review_details()` plus `load_latest_snapshot_board()` when you need audit-safe latest evidence.
- Treat `load_latest_snapshot_board()` as a UI convenience helper, not as the authoritative artifact selector for formula-level audits while mixed cache generations exist.
- After the supported-path regeneration work, re-check whether snapshot selection still needs a schema-aware tie-breaker.

## Artifact Summary

- Loader-selected snapshot: `snapshot_v8_h3_r1000_b50_u3b971cb9f1_20260527.pkl` with 29 rows and 33 columns
- Review-linked snapshot: `snapshot_v10_h3_r1000_b50_u3b971cb9f1_20260526.pkl` with 29 rows and 50 columns
- Latest review detail: `review_v10_h3_r1000_b50_u3b971cb9f1_20260526_20260527.pkl` with 29 rows and 76 columns
- Loader snapshot missing vs review: board_date, review_date, predicted_upside_pct, predicted_upside_low_pct, predicted_upside_high_pct, enhanced_probability_up, final_rank_score, sector_score, fund_score, news_score, launch_score, launch_readiness_score, launch_readiness, breakout_quality, resonance_quality, board_resonance_strength, long_setup_quality, crowding_risk, crowding_risk_label, risk_of_late_entry
- Review-linked snapshot missing vs review: board_date, review_date, predicted_upside_low_pct, predicted_upside_high_pct, enhanced_probability_up, launch_score, launch_readiness_score, launch_readiness, breakout_quality, resonance_quality, board_resonance_strength, long_setup_quality, crowding_risk, crowding_risk_label, risk_of_late_entry, launch_phase_label, market_resonance_score, intraday_sector_sync_score, relative_intraday_alpha, sector_follow_score

JSON: `E:\openclaw\.cache\trading_system_attribution\default_path_alignment.json`
