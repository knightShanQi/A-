# Trading System Leverage Matrix 2026-05-28

## Purpose

Turn the issue register into an execution-grade optimization matrix:

- expected impact on annualized return / drawdown credibility
- confidence level from current evidence
- engineering effort
- whether the action is safe to do now or should wait

This is the bridge between "what is wrong" and "what to change next".

## Scoring Legend

- Impact:
  - `Very High`: likely to change the truthfulness of the whole system evaluation or default decision
  - `High`: likely to change the effective research baseline or remove a known negative contributor
  - `Medium`: useful but secondary; unlikely to rescue the system alone
  - `Low`: cleanup or optional follow-up
- Confidence:
  - `High`: repeatedly supported by unified portfolio or review-linked evidence
  - `Medium`: supported on one slice but still needs stronger historical confirmation
  - `Low`: plausible but not yet strong enough to govern the roadmap
- Effort:
  - `Low`, `Medium`, `High`

## Leverage Matrix

| Action | Impact | Confidence | Effort | Why it matters | Do now? | Main evidence |
|:--|:--|:--|:--|:--|:--|:--|
| Regenerate a modern `h5 / 3%` artifact chain and prepare default migration | Very High | High | High | Current default `h3 / 10%` path is unsupported by market/proxy models, so many short-horizon conclusions are being produced on degraded architecture. | Yes | [`model_source_availability_2026-05-28.md`](E:\openclaw\docs\model_source_availability_2026-05-28.md), [`default_config_migration_2026-05-28.md`](E:\openclaw\docs\default_config_migration_2026-05-28.md), [`default_path_evidence_gap_2026-05-28.md`](E:\openclaw\docs\default_path_evidence_gap_2026-05-28.md) |
| Keep unified portfolio NAV as the only architecture decision metric | Very High | High | Low | This removes legacy backtest inflation immediately and prevents further optimization on false positive results. | Yes | [`trading_system_attribution_2026-05-28.md`](E:\openclaw\docs\trading_system_attribution_2026-05-28.md) |
| Keep all scorer/root-cause audits review-first via `load_latest_review_bundle()` | High | High | Low | This prevents date/schema mismatch and keeps later optimization evidence trustworthy. | Yes | [`default_path_alignment_2026-05-28.md`](E:\openclaw\docs\default_path_alignment_2026-05-28.md) |
| Remove continuous `execution_score` weight from research ranking paths | High | High | Low | Execution continuous weight is inert at top-3, slightly harmful at wider baskets, and adds dead complexity. The dashboard action/ranking path now defaults this blend weight to `0.0` while still allowing explicit override for A/B. | Yes, landed | [`execution_score_counterfactual_2026-05-28.md`](E:\openclaw\docs\execution_score_counterfactual_2026-05-28.md), [`execution_weight_sweep_2026-05-28.md`](E:\openclaw\docs\execution_weight_sweep_2026-05-28.md), [`execution_off_portfolio_backtest_2026-05-28.md`](E:\openclaw\docs\execution_off_portfolio_backtest_2026-05-28.md) |
| Keep `selection_score_top3` as the short-horizon ranking baseline | High | High | Low | It is currently the strongest replayable ranking baseline; changing away from it too early would likely add noise rather than edge. | Yes | [`ranking_quality_portfolio_2026-05-28.md`](E:\openclaw\docs\ranking_quality_portfolio_2026-05-28.md), [`candidate_breadth_2026-05-28.md`](E:\openclaw\docs\candidate_breadth_2026-05-28.md) |
| Set `launch_window_confidence_weight = 0.0` in research paths | Medium | High | Low | This is one of the few local score edits already shown to be non-destructive and slightly better than the live setting. | Yes | [`launch_window_confidence_sweep_2026-05-28.md`](E:\openclaw\docs\launch_window_confidence_sweep_2026-05-28.md) |
| Demote `final_rank_score` from primary research use | Medium | High | Low | It is an alias of `enhanced_attention_score` on current evidence and still weaker than `selection_score`. | Yes | [`final_rank_construction_2026-05-28.md`](E:\openclaw\docs\final_rank_construction_2026-05-28.md), [`final_rank_assignment_audit_2026-05-28.md`](E:\openclaw\docs\final_rank_assignment_audit_2026-05-28.md) |
| Remove heuristic strategy candidate layer from default decision path | High | High | Medium | Same-engine evidence says the handcrafted strategy layer is negative expectancy. | Yes | [`heuristic_ablation_2026-05-28.md`](E:\openclaw\docs\heuristic_ablation_2026-05-28.md), [`trading_system_audit_2026-05-28.md`](E:\openclaw\docs\trading_system_audit_2026-05-28.md) |
| Rebuild the model target around cross-sectional ranking and top-K post-cost return | Very High | High | High | Weak core alpha is the deepest bottleneck; without this, all later score cleanup only improves a weak baseline. | Yes, but after path repair | [`trading_system_audit_2026-05-28.md`](E:\openclaw\docs\trading_system_audit_2026-05-28.md), [`trading_system_issue_register_2026-05-28.md`](E:\openclaw\docs\trading_system_issue_register_2026-05-28.md) |
| Source-level simplification of `selection_score` (attention/launch dedup) | High | Medium | High | The current best ranker is also internally redundant; simplifying it is likely necessary for robustness, but premature deletion can remove real signal. | Yes, but only source-first | [`selection_score_source_audit_2026-05-28.md`](E:\openclaw\docs\selection_score_source_audit_2026-05-28.md), [`v10_attention_unification_2026-05-28.md`](E:\openclaw\docs\v10_attention_unification_2026-05-28.md), [`v10_attention_coverage_audit_2026-05-28.md`](E:\openclaw\docs\v10_attention_coverage_audit_2026-05-28.md) |
| Promote quant as a mild gate (`quant >= 58`) | Medium | Medium | Low | It may help, but the evidence is not strong enough yet and can easily become a false optimization if adopted too early. | Not yet | [`quant_score_contribution_2026-05-28.md`](E:\openclaw\docs\quant_score_contribution_2026-05-28.md), [`quant_gate_longer_window_2026-05-28.md`](E:\openclaw\docs\quant_gate_longer_window_2026-05-28.md) |
| Delete `enhanced_attention_score` outright from production logic | Medium | Low | Medium | Architecture redundancy is real, but current replayable evidence is too quick-board dominated to justify a production deletion. | No | [`v10_attention_family_compression_2026-05-28.md`](E:\openclaw\docs\v10_attention_family_compression_2026-05-28.md), [`v10_attention_coverage_audit_2026-05-28.md`](E:\openclaw\docs\v10_attention_coverage_audit_2026-05-28.md) |

## Recommended Sequencing

### Phase 1: Stop false optimization

Do immediately:

1. Use only unified portfolio NAV for decision-quality comparison.
2. Keep all current scorer audits review-first through `load_latest_review_bundle()`.
3. Remove continuous `execution_score` weight from research ranking.
4. Keep `selection_score_top3` as the short-horizon baseline.
5. Hold `launch_window_confidence_weight = 0.0` in research.

Expected benefit:

- cleaner evidence
- fewer false-positive score tweaks
- less time wasted on inert or duplicated layers

### Phase 2: Repair the supported path

Do next:

1. Regenerate modern `h5 / 3%` ranking/review artifacts.
2. Confirm that the new `h5` chain has current schema depth.
3. Move the default off unsupported `h3 / 10%`.

Expected benefit:

- puts research and production on a path that actually has model support
- makes subsequent ranking conclusions more trustworthy

### Phase 3: Rebuild the real alpha engine

Do after path repair:

1. Retrain around cross-sectional ranking and top-K post-cost return.
2. Re-run same-engine comparisons against the current `selection_score_top3` baseline.
3. Only then simplify the surviving attention/launch source families.

Expected benefit:

- this is the only phase likely to move the ten-year annualized return materially rather than cosmetically

## What This Matrix Says Professionally

The system does not need more ideas first.

It needs stricter ordering:

- first remove false confidence
- then repair the supported path
- then rebuild alpha on the right target

If that order is violated, the team will keep making local scorer adjustments on top of an unsupported path with weak core alpha, and the annualized return will stay low even if the surface-level diagnostics look better.
