# Trading System Issue Register 2026-05-28

## Purpose

Collapse the current audit evidence into one issue register that answers four questions at once:

1. What is actually wrong with the trading system?
2. Which issues are already proven by evidence rather than intuition?
3. Which issues hurt annualized return the most?
4. What should be fixed first?

Companion execution matrix:

- [`trading_system_leverage_matrix_2026-05-28.md`](E:\openclaw\docs\trading_system_leverage_matrix_2026-05-28.md)
  - use this when you need impact/confidence/effort ordering rather than just the issue list

## Overall Judgment

The system's low annualized return is not caused by "insufficient factors". It is caused by a stack mismatch:

- the default production path is running on an unsupported short-horizon model configuration
- the historical research stack mixes realistic and unrealistic backtest notions
- the current candidate/ranking logic contains several non-contributing or duplicated scorer layers
- the best surviving ten-year path still has only low-single-digit annualized return after realistic portfolio simulation

The most important strategic conclusion is:

- do not optimize the unsupported default path further as if it were a healthy baseline
- do not trust any score-layer conclusion unless it is tied to the unified portfolio engine or the audit-safe review-linked evidence path

## Priority Matrix

| Priority | Issue | Layer | Proven Evidence | Why it suppresses annualized return | Recommended action |
|:--|:--|:--|:--|:--|:--|
| P0 | Unsupported default configuration | Upstream model source | [`model_source_availability_2026-05-28.md`](E:\openclaw\docs\model_source_availability_2026-05-28.md), [`h3_restore_or_retire_2026-05-28.md`](E:\openclaw\docs\h3_restore_or_retire_2026-05-28.md) | Default `h3 / 10%` path has neither market-wide model nor proxy model, so current short-horizon ranking is being judged on a degraded fallback architecture. | Retire `h3 / 10%` as default unless restored; regenerate modern `h5 / 3%` artifacts first, then switch default. |
| P0 | Legacy backtest inflation | Evaluation / attribution | [`trading_system_attribution_2026-05-28.md`](E:\openclaw\docs\trading_system_attribution_2026-05-28.md) | Legacy averaged-return views overstated performance. Example: baseline combined rule falls from `+3.44%` to `-7.54%` annualized under unified portfolio NAV; bull/rank falls from `+10.99%` to `-0.10%`. | Use only unified portfolio NAV for architecture decisions; treat legacy averaged-return outputs as diagnostic only. |
| P0 | Weak core alpha | Model / candidate family | [`trading_system_audit_2026-05-28.md`](E:\openclaw\docs\trading_system_audit_2026-05-28.md), [`heuristic_ablation_2026-05-28.md`](E:\openclaw\docs\heuristic_ablation_2026-05-28.md) | Ten-year best migrated path is only `+2.66%` annualized with `-22.53%` max drawdown; baseline combined path is negative. The system simply does not start with enough per-trade edge. | Move the training/evaluation target to cross-sectional ranking and post-cost top-K portfolio returns. |
| P1 | Heuristic strategy layer is negative expectancy | Candidate generation | [`trading_system_audit_2026-05-28.md`](E:\openclaw\docs\trading_system_audit_2026-05-28.md), [`heuristic_ablation_2026-05-28.md`](E:\openclaw\docs\heuristic_ablation_2026-05-28.md) | Strategy-only top-N paths are catastrophic under realistic NAV, meaning the handcrafted post-model rules are selecting worse trades, not better ones. | Remove heuristic strategy layers from default ranking/gating until each term proves incremental portfolio value. |
| P1 | Execution continuous score is dead complexity | Scorer / ranking blend | [`execution_score_counterfactual_2026-05-28.md`](E:\openclaw\docs\execution_score_counterfactual_2026-05-28.md), [`execution_weight_sweep_2026-05-28.md`](E:\openclaw\docs\execution_weight_sweep_2026-05-28.md), [`execution_off_portfolio_backtest_2026-05-28.md`](E:\openclaw\docs\execution_off_portfolio_backtest_2026-05-28.md) | Continuous `execution_score` adds no marginal ranking value across the current replayable slice; in the best case it is inert, in wider baskets it is slightly harmful. | Keep discrete action states for explanation or veto; remove continuous execution weight from ranking research paths. |
| P1 | Final-rank score is an alias, not a real signal | Ranking construction | [`final_rank_construction_2026-05-28.md`](E:\openclaw\docs\final_rank_construction_2026-05-28.md), [`final_rank_assignment_audit_2026-05-28.md`](E:\openclaw\docs\final_rank_assignment_audit_2026-05-28.md), [`final_rank_rebuild_2026-05-28.md`](E:\openclaw\docs\final_rank_rebuild_2026-05-28.md) | `final_rank_score` equals `enhanced_attention_score` on the persisted slice and still underperforms `selection_score_top3`; this adds naming complexity without distinct alpha. | Stop treating `final_rank_score` as a primary research ranker until its upstream formula is materially different. |
| P1 | Selection score is strong but internally redundant | Primary ranking stack | [`selection_score_source_audit_2026-05-28.md`](E:\openclaw\docs\selection_score_source_audit_2026-05-28.md), [`selection_score_family_compression_2026-05-28.md`](E:\openclaw\docs\selection_score_family_compression_2026-05-28.md), [`v10_attention_unification_2026-05-28.md`](E:\openclaw\docs\v10_attention_unification_2026-05-28.md) | `selection_score_top3` is the current best ranking baseline, but it is built from highly collinear probability/attention/launch families. That makes future tuning unstable and easy to overfit. | Preserve `selection_score` as the current baseline, but simplify it source-first: remove inert launch confidence, then consolidate duplicated attention representations. |
| P2 | Launch-window confidence is still positively weighted despite no benefit | Selection-score micro layer | [`launch_window_confidence_sweep_2026-05-28.md`](E:\openclaw\docs\launch_window_confidence_sweep_2026-05-28.md) | Weights `0.00` to `0.03` all behave the same and slightly better than the current `0.04`, while heavier weights are clearly harmful. | Keep `launch_window_confidence_weight = 0.0` as the research default and stop spending tuning effort on this term. |
| P2 | Attention family is architecturally duplicated but not yet deletable | Selection-score source family | [`v10_attention_family_compression_2026-05-28.md`](E:\openclaw\docs\v10_attention_family_compression_2026-05-28.md), [`v10_attention_coverage_audit_2026-05-28.md`](E:\openclaw\docs\v10_attention_coverage_audit_2026-05-28.md) | Attention layers are highly redundant, but the replayable `v10` slice is dominated by quick-board paths and does not yet exercise the full enriched path. Premature deletion risks removing real signal. | Do source-level unification, not blunt deletion, and only finalize after a fresh enriched-history slice exists. |
| P2 | Evidence alignment was previously unsafe | Audit / observability | [`default_path_alignment_2026-05-28.md`](E:\openclaw\docs\default_path_alignment_2026-05-28.md), [`default_path_evidence_gap_2026-05-28.md`](E:\openclaw\docs\default_path_evidence_gap_2026-05-28.md) | Snapshot and review evidence were previously easy to mismatch by date/schema, which can contaminate scorer audits and root-cause explanations. | Use `load_latest_review_bundle()` as the authoritative audit-safe entrypoint; treat latest snapshot helpers as UI-oriented only. |
| P3 | Quant gate may help, but evidence is not stable enough | Secondary filter | [`quant_score_contribution_2026-05-28.md`](E:\openclaw\docs\quant_score_contribution_2026-05-28.md), [`quant_gate_longer_window_2026-05-28.md`](E:\openclaw\docs\quant_gate_longer_window_2026-05-28.md) | A mild `quant >= 58` gate looks helpful on one unified-portfolio slice, but only "less bad" on the longer review proxy. It is not yet strong enough to promote. | Retest only after the supported-path artifact chain is rebuilt. |

## What Is Already Good Enough To Keep

- The unified portfolio engine is now the correct evaluation backbone.
- `selection_score_top3` is currently the best short-horizon ranking baseline on the replayable slice.
- `native_v3_full_green_top3` is currently the least misleading ten-year baseline for same-engine comparison.
- The new audit-safe review-linked access path (`load_latest_review_bundle()`) is the correct evidence entrypoint for future scorer work.

## What Should Stop Immediately

- Stop using `combined_top3_score_ge_68_full` as the narrative baseline.
- Stop using legacy averaged-return annualization to justify architecture decisions.
- Stop treating `execution_score` continuous weight and `final_rank_score` as if they were independently validated alpha sources.
- Stop using old `h5 / 3%` artifacts as a drop-in replacement for the default path; they are schema-poor, not just stale.

## Recommended Fix Order

1. Repair the supported path, not the unsupported default:
   use [`regenerate_supported_default_artifacts.py`](E:\openclaw\scripts\regenerate_supported_default_artifacts.py) until modern `h5 / 3%` ranking/review artifacts exist.

2. Keep all future scoring audits review-first:
   use `load_latest_review_bundle()` to avoid date/schema mismatches.

3. Hold the current ranking baseline steady:
   keep `selection_score_top3` as the short-horizon benchmark while removing only clearly inert terms.

4. Remove dead complexity before inventing new scores:
   execution continuous weight, launch-window-confidence weight, and alias-style final-rank logic should not survive into the next research baseline.

5. Rebuild the real alpha target:
   train and evaluate against cross-sectional top-K portfolio outcomes instead of binary next-day direction.

## Final Professional Conclusion

The system is underperforming because it is currently solving the wrong optimization problem on the wrong default path with too much duplicated scorer complexity.

The most dangerous false belief would be:

- "the system just needs a better threshold or a few more factors"

The evidence now says the opposite:

- first fix the supported path
- then keep only the simplest ranking stack that survives unified portfolio scrutiny
- then rebuild the alpha target around ranking and portfolio return, not binary direction

That is the shortest path to materially higher and more believable annualized return.
