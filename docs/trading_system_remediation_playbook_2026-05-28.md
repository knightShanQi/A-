# Trading System Remediation Playbook 2026-05-28

## Objective

Turn the attribution results into a concrete removal and rebuild order so the next iterations stop optimizing negative-expectancy layers.

Companion consolidated register:

- [`trading_system_issue_register_2026-05-28.md`](E:\openclaw\docs\trading_system_issue_register_2026-05-28.md)
  - use this as the single-page priority matrix for what is broken, why it matters, and what to fix first

## Evidence Baseline

- Unified attribution source: `E:\openclaw\.cache\trading_system_attribution\cross_strategy_attribution.csv`
- Supporting interpretation: `E:\openclaw\docs\trading_system_attribution_2026-05-28.md`

Key facts from the same-engine comparison:

- `combined_top3_score_ge_68_full` is negative after realistic portfolio simulation: `portfolio_annualized_return = -7.54%`, `portfolio_max_drawdown = -68.03%`.
- Adding the V3 full-green filter to the same combined baseline only lifts it to `+0.46%` annualized while sharply reducing exposure.
- The best ten-year migrated path is currently `v3_full_green_top3` at only `+2.66%` annualized with `-22.53%` max drawdown.
- The rebuilt bull/rank overlay falls from legacy `+10.99%` annualized to about `-0.10%` under unified portfolio NAV.

That means the current bottleneck is not missing filters. The bottleneck is that multiple heuristic layers are manufacturing confidence without manufacturing enough tradable alpha.

## Module-Level Diagnosis

### 1. Heuristic score stacking is too deep

The current stack compounds several hand-tuned scores before execution:

- `src/a_share_predictor/strategy.py:244`
  - `build_strategy_workbench()` mixes `probability_up`, `quant_score`, `sector_score`, temporal news, and intraday volume share into `strategy_score`.
- `src/a_share_predictor/strategy.py:305`
  - `assess_launch_window()` adds stage labels, structure heuristics, launch readiness, regime fit, specialist confidence, and upside assumptions into `window_score`.
- `src/a_share_predictor/strategy.py:418`
  - `assess_execution_readiness()` adds another stage/structure/resonance layer into `execution_score` and then derives expected return and reward/risk labels from that synthetic score.

Problem:

- The same underlying signal is effectively counted multiple times.
- These scores are not calibrated against realized portfolio PnL.
- The system produces persuasive labels and confidence numbers even when the ten-year portfolio result is near zero or negative.

Action:

- Treat `strategy_score`, `window_score`, and `execution_score` as UI diagnostics only.
- Stop using them as gating or ranking inputs for research decisions until each score is revalidated against post-cost top-K returns.

### 2. Stage classification is a hard-threshold opinion layer

`src/a_share_predictor/stages.py` currently uses fixed feature thresholds to assign a single market stage:

- `src/a_share_predictor/stages.py:193`
  - `main_rise_start_score()` is a large hand-built weighted formula.
- `src/a_share_predictor/stages.py:449`
  - `classify_stage()` selects one stage from chained boolean threshold blocks.
- `src/a_share_predictor/stages.py:538`
  - `stage_numeric_score()` converts the chosen label back into another numeric score.

Problem:

- The stage layer is effectively a discrete prior placed on top of already noisy features.
- Because the classifier is threshold-based, small feature moves can flip the stage and therefore flip downstream launch/execution judgments.
- The current best ten-year path is still low-single-digit annualized even after these stage-aware filters, which implies the stage system is not generating enough edge to justify its complexity.

Action:

- Keep stage labels for explanation and chart annotation.
- Remove stage-derived numeric scores from any selection logic unless a direct ablation proves positive portfolio contribution.
- Prefer a simpler market-state gate like `v3_full_green` over deep per-stock stage narratives in the research path.

### 3. The baseline combined selection rule should not remain the default

The attribution table is decisive here:

- Baseline combined rule: `-7.54%` annualized, `-68.03%` drawdown.
- Same candidates plus V3 gate: `+0.46%` annualized.
- Native V3 generation: `+2.66%` annualized.

Problem:

- The baseline combined layer is not a weak positive edge. It is a strong drag once realistic capital deployment is enforced.

Action:

- Retire `combined_top3_score_ge_68_full` as a default reference strategy.
- Do not optimize its thresholds further until a clean ablation says the candidate family itself has positive post-cost expectancy.

### 4. Bull/rank overlays are research-only until proven again

The rebuilt bull/rank path looked attractive under the old averaging logic and failed under unified NAV.

Problem:

- This path is currently evidence of backtest-measurement error, not evidence of scalable alpha.

Action:

- Keep `scripts/backtest_bull_market_rank_score.py` for research comparison only.
- Do not use it to justify production architecture or ranking weights.

## Priority Order

### Priority 0: Freeze misleading defaults

Do now:

- Stop presenting `combined_top3_score_ge_68_full` as the main baseline.
- Mark bull/rank overlay outputs as research-only in analysis discussions.
- Prefer `portfolio_*` metrics over legacy averaged-return metrics everywhere.

### Priority 1: Shrink the decision stack

Do next:

- In the research path, rank and select from the simplest surviving family first:
  - `v3_full_green`
  - top-N selection
  - unified portfolio backtest
- Remove or ignore extra heuristic re-scorers unless they beat that baseline in the same engine.

Expected benefit:

- Fewer false-positive degrees of freedom.
- Easier attribution when a change helps or hurts.

### Priority 2: Rebuild the model target around ranking, not binary direction

Current audit evidence says next-day classification quality is weak. The model should be judged by whether it improves top-K portfolio returns, not by whether it slightly predicts up/down.

Do next:

- Use cross-sectional ranking objectives and evaluation:
  - rank IC
  - top-minus-median spread
  - top-K post-cost return
  - hit rate conditional on being selected into the portfolio
- Keep ROC AUC as a secondary health metric only.

### Priority 3: Reintroduce heuristics only through ablation

If a heuristic layer returns, it should return one piece at a time.

Required standard:

- Same candidate universe
- Same portfolio engine
- Same date window
- Same cost/slippage assumptions
- Same top-N and holding period

Add a layer back only if it improves:

- annualized return
- max drawdown
- or return per unit drawdown

without merely collapsing exposure.

## Concrete Refactor Sequence

1. Default research baseline

- Promote `native_v3_full_green_top3` to the main ten-year comparison baseline.
- Demote `combined_top3_score_ge_68_full` to a historical reference only.

2. Selection-path simplification

- Audit where `stage_numeric_score`, `strategy_score`, `window_score`, and `execution_score` are used in candidate ranking or filtering.
- For each use, run a same-engine ablation against the simpler `v3_full_green_top3` baseline.

3. Model retraining objective

- Replace "will next day close up" optimization with portfolio-aware ranking optimization.
- Evaluate on rolling windows using post-cost top-K return, not classification lift alone.

4. UI / explanation cleanup

- Preserve human-readable labels from `strategy.py` and `stages.py`.
- Explicitly separate "narrative/explanation" from "selection/ranking signal".

## Keep / Remove / Research

Keep:

- Unified portfolio engine in `src/a_share_predictor/portfolio_backtester.py`
- Native `v3_full_green` family as the strongest current ten-year base path
- Portfolio-level attribution artifacts

Remove from default decision path:

- `combined_top3_score_ge_68_full`
- any default reliance on legacy averaged-return annualization
- any claim that bull/rank overlay is a validated edge

Research only:

- `strategy_score`
- `window_score`
- `execution_score`
- `stage_numeric_score`
- rebuilt bull/rank overlays

## What To Build Next

The first ablation harness now exists:

- script: `E:\openclaw\scripts\build_heuristic_ablation_report.py`
- report: `E:\openclaw\docs\heuristic_ablation_2026-05-28.md`
- data: `E:\openclaw\.cache\trading_system_attribution\heuristic_ablation.csv`

The first scorer audit now also exists:

- script: `E:\openclaw\scripts\build_scorer_audit_report.py`
- report: `E:\openclaw\docs\scorer_audit_2026-05-28.md`
- data: `E:\openclaw\.cache\trading_system_attribution\scorer_audit.json`

The first execution-score counterfactual now also exists:

- script: `E:\openclaw\scripts\build_execution_score_counterfactual_report.py`
- report: `E:\openclaw\docs\execution_score_counterfactual_2026-05-28.md`
- data: `E:\openclaw\.cache\trading_system_attribution\execution_score_counterfactual.json`

The first execution-weight sweep now also exists:

- script: `E:\openclaw\scripts\build_execution_weight_sweep_report.py`
- report: `E:\openclaw\docs\execution_weight_sweep_2026-05-28.md`
- data: `E:\openclaw\.cache\trading_system_attribution\execution_weight_sweep.json`

The first execution-off NAV proxy now also exists:

- script: `E:\openclaw\scripts\build_execution_off_nav_proxy_report.py`
- report: `E:\openclaw\docs\execution_off_nav_proxy_2026-05-28.md`
- data: `E:\openclaw\.cache\trading_system_attribution\execution_off_nav_proxy.json`

The first real execution-off unified portfolio backtest now also exists:

- script: `E:\openclaw\scripts\build_execution_off_portfolio_backtest.py`
- report: `E:\openclaw\docs\execution_off_portfolio_backtest_2026-05-28.md`
- data: `E:\openclaw\.cache\trading_system_attribution\execution_off_portfolio_backtest.json`

The first candidate-breadth concentration study now also exists:

- script: `E:\openclaw\scripts\build_candidate_breadth_report.py`
- report: `E:\openclaw\docs\candidate_breadth_2026-05-28.md`
- data: `E:\openclaw\.cache\trading_system_attribution\candidate_breadth.json`

The first same-engine ranking-quality portfolio comparison now also exists:

- script: `E:\openclaw\scripts\build_ranking_quality_portfolio_report.py`
- report: `E:\openclaw\docs\ranking_quality_portfolio_2026-05-28.md`
- data: `E:\openclaw\.cache\trading_system_attribution\ranking_quality_portfolio.json`

The first final-rank construction audit now also exists:

- script: `E:\openclaw\scripts\build_final_rank_construction_report.py`
- report: `E:\openclaw\docs\final_rank_construction_2026-05-28.md`
- data: `E:\openclaw\.cache\trading_system_attribution\final_rank_construction.json`

The first final-rank assignment-site audit now also exists:

- script: `E:\openclaw\scripts\build_final_rank_assignment_audit.py`
- report: `E:\openclaw\docs\final_rank_assignment_audit_2026-05-28.md`
- data: `E:\openclaw\.cache\trading_system_attribution\final_rank_assignment_audit.json`

The first lightweight final-rank rebuild comparison now also exists:

- script: `E:\openclaw\scripts\build_final_rank_rebuild_report.py`
- report: `E:\openclaw\docs\final_rank_rebuild_2026-05-28.md`
- data: `E:\openclaw\.cache\trading_system_attribution\final_rank_rebuild.json`

The first selection-score proxy ablation now also exists:

- script: `E:\openclaw\scripts\build_selection_score_ablation_report.py`
- report: `E:\openclaw\docs\selection_score_ablation_2026-05-28.md`
- data: `E:\openclaw\.cache\trading_system_attribution\selection_score_ablation.json`

The first launch-window contribution audit now also exists:

- script: `E:\openclaw\scripts\build_launch_window_contribution_report.py`
- report: `E:\openclaw\docs\launch_window_contribution_2026-05-28.md`
- data: `E:\openclaw\.cache\trading_system_attribution\launch_window_contribution.json`

The first tomorrow-confidence contribution audit now also exists:

- script: `E:\openclaw\scripts\build_tomorrow_confidence_contribution_report.py`
- report: `E:\openclaw\docs\tomorrow_confidence_contribution_2026-05-28.md`
- data: `E:\openclaw\.cache\trading_system_attribution\tomorrow_confidence_contribution.json`

The first quant-score contribution audit now also exists:

- script: `E:\openclaw\scripts\build_quant_score_contribution_report.py`
- report: `E:\openclaw\docs\quant_score_contribution_2026-05-28.md`
- data: `E:\openclaw\.cache\trading_system_attribution\quant_score_contribution.json`

The codebase now also has a safe execution-off experiment hook:

- `E:\openclaw\src\a_share_predictor\api_service.py`
- `apply_probability_contract(..., launch_window_execution_weight=...)`
- default stays at `0.22`, but research callers can now set it to `0.0` without forking the whole normalization path

It already proves three important points:

- strict `full_green` gating is useful mainly because it removes bad exposure
- candidate-family quality matters more than pause-style overlays
- model ordering is still weaker than rebuilt rank ordering inside the same gated pool

The scorer audit adds three more:

- `launch_window_score` has replay evidence and does correlate with better next-day outcomes in the recent review sample
- `selection_score` has a weak positive replay gradient so far: high bucket about `+0.77%` next-day vs low bucket about `+0.22%`, but the middle bucket is stronger than the high bucket, so it is not yet a clean monotonic signal
- `execution_score` does **not** currently justify itself as a ranking input: high bucket about `+0.20%` next-day vs low bucket about `+0.23%`, with the middle bucket doing best, which is consistent with saturation/noise rather than a calibrated edge
- the stronger same-day counterfactual is even worse for `execution_score`: across the 13 board dates with both scores present, `selection_score` top-3 and the blended `0.62*selection + 0.38*execution` top-3 are identical on every date, so the execution term adds no marginal ranking benefit in that replay slice
- at top-10, the blend is slightly worse than `selection_score` alone (`+0.42%` vs `+0.49%` average next-day return), which makes `execution_score` the clearest immediate candidate to remove from default ranking blends
- the weight sweep makes the result even harder to dismiss: on the current v9 replay slice, every execution weight from `0.00` through `0.50` produces the exact same top-3 picks and the same `-0.53%` average next-day return, so the continuous execution term is functionally inactive as a ranking differentiator in that range
- the proxy NAV points the same way: execution-off and the current blend are identical at top-3 (`-8.01%` cumulative over the 13-day replay slice), while execution-off is better at top-10 (`+6.27%` vs `+5.22%`) with the same drawdown proxy
- the real unified portfolio backtest says the same thing in a less fragile form: top-3 execution-off and the current blend are exactly identical on the current v9 snapshot slice, while top-10 execution-off is only marginally better (`1.3925%` vs `1.3897%` annualized, slightly higher ending equity, nearly identical drawdown)
- but the action layer still contains signal in discrete form: `买` rows average about `+1.28%` next-day on the v9 replay slice, while high-selection non-buy rows are not obviously better, so the immediate target is the continuous execution weight, not the entire execution-state layer
- this execution-weight cleanup has now landed in code on the dashboard action/ranking path: `action_execution_weight` defaults to `0.0`, and focused tests prove the default path now collapses `action_score` onto `selection_score` unless an explicit override is passed
- a code-level experiment seam now exists for the API normalization path, so the next execution-off A/B can be run by parameterizing weight instead of copying and drifting the normalization logic
- `stage_score` does not yet show convincing monotonic evidence in the small joined sample
- replay persistence for `execution_score`, `selection_score`, and direct `stage_score` has now landed in code and review cache version `v9`
- backfilled `v9` review history now raises replay-detail coverage to `525 / 989` rows for `execution_score` and `selection_score`, and `450 / 989` rows for direct `stage_score`
- that is enough to remove the old hard observability gap, but not enough to treat scorer-level results as fully stable because the selected history is still mixed-version

So the next engineering task is no longer "build a harness". It is:

- keep regenerating `v9` review history until scorer coverage is dominated by post-persistence artifacts
- prioritize scorer usage sites in this order:
  - `execution_score`: first candidate to demote from default ranking/gating because bucket analysis, same-day counterfactual ranking, the 0.00-0.50 weight sweep, and the proxy NAV all say it is not adding marginal value
  - `selection_score`: keep as research-only until it shows monotonic benefit against simpler baselines
  - `stage_numeric_score`: keep under audit because current joined sample still looks negative/non-monotonic
- compare "used as ranking/gating" vs "ignored, explanation-only" under the same portfolio engine or replay set
- remove any scorer from the default selection path unless it improves annualized return or drawdown-adjusted return on the same date range

Important nuance:

- execution-off is a cleanup step, not a silver bullet
- it reduces dead complexity and slightly helps or stays flat in every experiment so far
- but it does not solve the deeper problem that the short-horizon candidate stack itself still lacks enough alpha, especially in concentrated top-3 mode
- the new breadth study says widening the basket is not the immediate answer either:
  - top-3 is the best current width on the v9 unified-portfolio slice at about `+1.54%` annualized with about `-3.54%` max drawdown
  - top-10 is only slightly lower return at about `+1.39%`, but drawdown more than triples to about `-11.83%`
  - top-15 and top-20 both lose more annualized return, which means lower-ranked names are currently diluting the edge faster than they add capacity
- the ranking-quality portfolio A/B narrows the next bottleneck further:
  - among scorers that actually persist into the current snapshot artifacts, `selection_score_top3` is strongest at about `+1.54%` annualized
  - `launch_window_score_top3` is weaker at about `+0.62%`, so it looks more like a supporting overlay than a primary ranker
  - `enhanced_attention_score_top3` is negative on this slice (about `-0.13%` annualized), so the older attention-style path should not be treated as the main research baseline
  - `final_rank_score` is no longer missing from the current snapshot slice after backfill, but the new same-engine evidence is not encouraging: it matches `enhanced_attention_score_top3` exactly and is also about `-0.13%` annualized, so it currently adds no distinct alpha case over the older attention-style ranker
- the final-rank construction audit tightens that one step further:
  - on all `527` persisted `v9` snapshot rows, `final_rank_score` equals `enhanced_attention_score` exactly, not approximately
  - top-3 selections from `final_rank_score` and `enhanced_attention_score` are identical on all `13/13` comparable board dates
  - this means `final_rank_score` is currently an alias, not a differentiated composite signal; improving it now requires changing its formula, not tuning its thresholds or persistence
- the final-rank assignment audit localizes the engineering root cause:
  - there are only `2` true source assignments, both in `dashboard.py`, and both assign `final_rank_score` directly from `enhanced_attention_score`
  - downstream `api_service.py` and `daily_review.py` sites are mostly fallback, passthrough, or persistence hooks rather than real score-construction sites
  - that means any meaningful `final_rank_score` rebuild should start by changing or deleting those two upstream assignments, not by polishing downstream consumers
- the lightweight rebuild sweep makes the optimization tradeoff practical:
  - simple rebuilds from persisted fields like `selection_score + probability_up + quant_score + launch_window_score` can recover `final_rank_score` from negative to about `+0.97%` annualized
  - but none of the tested rebuilds beat `selection_score_top3` at about `+1.54%` annualized, and all carry much deeper drawdown than the current selection baseline
  - that means the most pragmatic near-term path is still to improve `selection_score` construction directly rather than introducing a second ranking label that remains weaker even after a sensible rebuild
- the first selection-score proxy ablation makes that guidance more concrete:
  - none of the persisted-field proxy variants beat the live `selection_score_top3` baseline at about `+1.54%` annualized
  - the strongest simplified proxy is the `no_quant` variant at about `+1.01%`, which implies the persisted quant term may be less essential than the current live score suggests on this slice
  - the weakest simplified proxy is the `no_launch` variant at about `+0.46%`, which implies the launch-window family is a higher-priority ingredient to preserve and audit carefully
  - removing the `tomorrow_plan_confidence` family also hurts badly and produces the deepest drawdown among tested proxies, so that component family should not be casually stripped without a source-level check
- the launch-window contribution audit sharpens the launch conclusion:
  - `launch_window_score` as a standalone top-3 ranker is weak at about `+0.62%` annualized with very poor drawdown, so it should not be promoted to primary ranking logic
  - explicit launch-only gates on the current `selection_score_top3` baseline do not improve the result because the baseline winners already mostly satisfy the launch criteria on this slice
  - adding extra launch tilt on top of `selection_score` makes things worse, which implies launch-window is probably already represented at roughly the right strength inside the live score
  - the practical role of launch-window is therefore closer to structure/risk-control context than to independent alpha source
- the tomorrow-confidence contribution audit points in a similar but slightly more defensive direction:
  - `tomorrow_plan_confidence` as a standalone top-3 ranker is not strong enough at about `+0.95%` annualized
  - strict tomorrow-confidence gates cut annualized return sharply (best gate only about `+0.53%`) but do reduce drawdown, which suggests this family behaves more like a risk-shaping filter than a primary alpha engine
  - extra tomorrow-confidence tilt on top of `selection_score` also hurts, so the live score is probably already using about as much of it as the current slice can support
  - the practical implication is to keep tomorrow-confidence as an internal context/risk-control component, not to elevate it into its own standalone ranking track
- the quant-score contribution audit is the first one to show a potentially actionable mild gate:
  - `quant_score` as a standalone top-3 ranker is almost useless at about `+0.07%` annualized, so it is clearly not an independent alpha engine
  - blindly overweighting quant on top of `selection_score` is also harmful, dropping annualized return to about `+0.50%`
  - but a moderate gate `quant_score >= 58` improves both annualized return (to about `+1.60%`) and max drawdown (to about `-2.97%`) on the current slice, while a stricter gate `>= 68` overfilters and becomes worse
  - the longer-window `v9` review proxy retest is directionally consistent but not strong enough to promote: `quant>=58` is less bad than the ungated baseline on that slice (about `-64.49%` vs `-75.47%` annualized proxy), but both are still negative, so quant currently looks more like a mild damage-control filter than a confirmed return enhancer
  - that means quant still deserves one more check on a longer unified-portfolio candidate source before adoption, but it should no longer be treated as a near-certain upgrade candidate
- the new `selection_score` source audit reframes the deeper optimization target:
  - the authoritative score construction still lives in two `dashboard.py` sites, and both versions layer the same families repeatedly: probability, attention, launch context, and then smaller quant/tomorrow/adjustment terms
  - on the persisted `v9` review slice, `attention_score`, `enhanced_attention_score`, `probability_up`, and `launch_window_score` all correlate strongly with `selection_score`, while `stage_score` remains very weak
  - those persisted source-family terms explain about `95.61%` of `selection_score` variance on `450` rows, but the sign-flipped projection coefficients show the system is carrying heavy multicollinearity rather than clean independent evidence
  - this means the next meaningful architecture cleanup is not “add another ranking overlay”, but “collapse the attention family into a smaller core representation and reduce redundant launch-layer restatement”
  - that source-audit observability gap has now been closed in code for the next cache generation: `predicted_upside_pct`, `tomorrow_plan_confidence`, `sector_score`, `fund_score`, `news_score`, and the adjustment terms are now written into new review/snapshot artifacts
  - but the historical audit evidence is still dominated by pre-upgrade `v9` files, so the next engineering step is to regenerate or backfill a meaningful `v10` history before trusting formula-level ablations too strongly
  - from an engineering ROI perspective, the next safest improvement path is: regenerate `v10` evidence, then ablate and simplify the clustered attention/launch families before trying to tune thresholds again
- the first family-compression counterfactual now narrows the simplification order:
  - removing the tiny `launch_window_confidence` layer is the cleanest immediate candidate so far: on the current unified-portfolio `v9` slice, `selection_minus_launch_confidence` slightly outperforms the baseline (`~1.55%` vs `~1.54%` annualized) and slightly improves drawdown
  - removing the whole launch family is still survivable but clearly rougher, with drawdown worsening to about `-7.82%`, so the right interpretation is “trim launch gently first”, not “delete launch context wholesale”
  - removing either attention restatement layer (`attention_score` or `enhanced_attention_score`) degrades realized portfolio performance more noticeably, even though top-3 overlap stays around `2.23 / 3`; that means the attention family is redundant in architecture terms but not yet safely removable in one cut on current evidence
  - removing `tomorrow_plan_confidence` is the clearest negative simplification, collapsing annualized return to about `+0.56%`, so that family should stay in place until stronger `v10` evidence says otherwise
  - the practical simplification order is now: first demote `launch_window_confidence`, then retest lighter attention-family compression, while leaving `tomorrow_plan_confidence` intact
- the dedicated `launch_window_confidence` weight sweep makes that first step much firmer:
  - every tested weight from `0.00` through `0.03` produces the exact same portfolio result on the current `v9` slice, which means the layer is functionally inert across that whole range
  - the current live/replay weight `0.04` is actually slightly worse than `0.00` to `0.03`, and heavier weights (`0.06`, `0.08`) degrade badly, dropping annualized return to about `+0.97%` and deepening drawdown to about `-14.67%`
  - the correct engineering interpretation is no longer “reduce this weight a bit”; it is “set it to zero first in research paths, because the tested near-zero corridor is flat and the current positive weight is already on the wrong side of the edge”
  - this is now operationally easy to test in code as well: the selection-score evaluation path accepts an explicit `launch_window_confidence_weight`, and the research scripts `scripts/run_daily_review_maintenance.py` plus `scripts/backfill_daily_review_v9.py` can both pass `--launch-window-confidence-weight 0.0` without rewriting the full formula or changing default UI behavior

Until those scorer-level ablations exist, threshold tuning inside those layers is still mostly noise-fitting.

## Latest v10 Compression Update

- The new `v10` zero-launch slice confirms that attention-family cleanup is harder than launch-window-confidence cleanup:
  - on the same unified portfolio engine, no tested attention-family subtraction beats the live `selection_score_top3` baseline of about `+1.54%` annualized with about `-3.54%` max drawdown
  - removing either explicit attention layer drops annualized return by about `0.23` percentage points and roughly doubles drawdown depth, even though average top-3 overlap still stays around `2.15 / 3`
  - removing the `probability_up` layer is worse, dropping annualized return by about `0.63` percentage points and pushing overlap down to about `1.77 / 3`
  - removing the whole probability-plus-attention cluster is no better than removing `probability_up` alone, which implies probability is the stickier core signal while the two attention layers mostly travel together as one compressed cluster
- Practical implication:
  - keep `launch_window_confidence_weight = 0.0` as the current research default
  - do not do a blunt attention-family deletion next
  - if attention cleanup continues, it should be a source-level consolidation that reduces duplicate attention representations, not a subtraction pass that weakens the surviving ranking core
- The first source-level unification check on the same `v10` slice narrows that further:
  - `attention_score` and `enhanced_attention_score` are literally identical on `317 / 527` persisted rows and correlate at about `0.9875`, so keeping both as first-class ranking layers is hard to justify architecturally
  - if forced to keep only one attention representation today, `attention_score` is the less damaging survivor: the `selection_unify_to_base_attention` counterfactual still reaches about `+1.39%` annualized with about `-4.84%` drawdown
  - keeping only the enhanced representation is worse at about `+1.21%` annualized with about `-7.23%` drawdown
  - but neither unified variant beats the live dual-layer baseline, so the next step is not immediate deletion; it is to rebuild the upstream attention formulas so one representation can replace two without losing ranking quality
- The new coverage audit adds an important scope warning before any production deletion:
  - all `527` replayable `v10` rows currently carry `sector_score = fund_score = news_score = 50`, so the persisted slice is not exercising true sector/fund/news dispersion
  - `candidate_reason` and `launch_phase_label` are also absent on all current rows, which means the replayable slice is missing much of the richer narrative context present in live symbol analysis
  - the `model_source_label` mix is dominated by lightweight generation paths (`427` rows from “最新收盘快榜（完整版特征与回测正在后台补齐）”, `100` rows from “本地快速回退”), so current redundancy findings are strongest for the quick-board recovery path, not for the full enriched-analysis architecture
  - practical consequence: do not delete `enhanced_attention_score` from production-facing logic until a fresh non-backfilled enriched-history slice exists and repeats the same conclusion
- A deeper upstream bottleneck is now confirmed at the model-source layer:
  - the current focus-board audit configuration `h3 / 10%` has neither a market-wide model nor a proxy model available, so it is structurally forced into `local_fast_fallback`
  - by contrast, the older `h5 / 3%` path still lacks a full market-wide model, but it does load a proxy model, meaning the two horizons are not even running the same upstream architecture
  - this means some of the current short-horizon underperformance is likely architecture degradation, not only weak ranking formulas
  - practical consequence: before deleting more score layers on the `h3 / 10%` path, decide whether to train or restore a compatible market/proxy model for that configuration, or explicitly retire it as a trading configuration
- The restore-vs-retire assessment now sharpens that into an actionable default choice:
  - current UI defaults still point users to `h3 / 10%`
  - the maintained training entrypoint in the repo (`scripts/train_market_wide_model.py`) only refreshes `h5 / 3%`
  - therefore the default user path and the maintained model-refresh path are misaligned
  - recommendation from current evidence: retire `h3 / 10%` as the default production trading configuration until a compatible market-wide or proxy model is restored for that exact setting
  - if you want to keep `h3 / 10%`, keep it explicitly as a research track; if you want a production default now, point it to a configuration that actually has model support, such as `h5 / 3%`
  - The migration-readiness audit adds one more operational constraint:
    - `h3 / 10%` currently has the freshest review and snapshot chain, but no model support
    - `h5 / 3%` has proxy-model support, but its ranking/review artifacts are stale and sparse under the current schema
    - therefore the correct move is not an immediate default flip; it is a two-stage migration:
      1. regenerate fresh `h5 / 3%` ranking/review artifacts under the current schema
      2. then switch the default away from unsupported `h3 / 10%`
    - there is now a dedicated regeneration entrypoint for that first step: [`E:\openclaw\scripts\regenerate_supported_default_artifacts.py`](E:\openclaw\scripts\regenerate_supported_default_artifacts.py)
    - that entrypoint now also supports an explicit `--force-refresh` path, so supported-path regeneration is no longer dependent on the ranking cache first deciding it is stale; this closes an important migration-evidence gap for `h5 / 3%`
    - the latest operational evidence narrows the real blocker further:
      - a 15-minute `--force-refresh` run still produced no new `h5` review/snapshot artifacts
      - even `--force-refresh --rankings-only` timed out after 10 minutes without creating any new `h5` ranking artifacts
      - so the immediate engineering bottleneck is no longer "missing migration code"; it is the cost of the full `h5` ranking rebuild itself
      - the dedicated bottleneck audit narrows that one layer deeper:
        - current-date snapshot history already exists for `2026-05-27`
        - but there is no same-date feature store, candidate pool, dynamic fallback pool, or current-date `h5` candidate-analysis cache
        - combined with `FULL_MARKET_MAX_WORKERS = 1` and a rule-based candidate pool size of `120`, the supported path currently enters a cold serial rebuild
      - there is now a matching staged recovery hook in code as well:
        - `scripts/regenerate_supported_default_artifacts.py --stores-only` warms the store layer without immediately paying the full candidate-analysis cost
        - `--rankings-only` remains the next diagnostic step if store warmup succeeds but full ranking still stalls
      - the latest stage-isolated run makes the first hard wall explicit:
        - `--store-stage snapshot` completes in about 40 seconds
        - `--store-stage features` still times out after 10 minutes and does not write a same-date `market_daily_feature_store`
        - that `features` timeout reproduces even when snapshot-history is already warm and no snapshot store rewrite occurs
        - code now avoids one obvious slow fallback by reusing snapshot-history when live snapshot context is empty, but the real `features` stage still times out; the remaining bottleneck is deeper inside feature-store generation
        - so the current supported-path recovery order should be read as `snapshot -> features -> pools -> rankings`, and the primary engineering target is now feature-store cold-start cost
    - initial real execution shows the task is operationally heavy rather than logically blocked: the full-market run exceeded a 180-second CLI timeout, so it should be scheduled and treated as a long-running maintenance job
    - the new evidence-gap audit adds a stricter requirement before any default switch:
      - the old `h5 / 3%` stack is not only stale, it is schema-poor for current audit work (`v3` snapshot + `v3` review, with none of the modern persisted fields such as `selection_score`, `final_rank_score`, `launch_window_score`, `execution_score`, and the source-term persistence family)
      - the current `h3 / 10%` path is also not fully schema-clean on the snapshot side: the latest board-date snapshot is still `v8` while the latest review is `v10`, so recent evidence on the default path itself is mixed across cache generations
      - practical consequence: the `h5` regeneration step must be treated as a schema-and-observability rebuild, not merely a cache refresh, and any near-term score audit should prefer current review-detail evidence over snapshot-only evidence when those two generations disagree
      - the loader audit now sharpens that one step further:
        - the review loaders have been corrected to prefer the highest cache version when `board_date/review_date` tie, so the latest review evidence now correctly resolves to `v10`
        - the remaining mismatch is snapshot-only: `load_latest_snapshot_board()` still picks the latest board date, which currently means a `v8` fast-board snapshot on `2026-05-27` instead of the richer `v10` review-linked snapshot on `2026-05-26`
        - on the current sample this does not change the selected symbol set, but it does materially degrade interpretability by dropping fields such as `selection_score`, `final_rank_score`, `sector/fund/news`, `stage_score`, and `execution_score`
        - practical consequence: for formula/root-cause audits, anchor on the latest review detail first and join back to the snapshot with the same `board_date`; treat `load_latest_snapshot_board()` as a UI helper until the snapshot chain itself is modernized
        - this audit-safe access path is now available in code through `daily_review.load_latest_review_bundle()`, and the enhanced API payload now exposes its linked review snapshot instead of forcing downstream callers to reconstruct it
    - until that migration is complete, additional formula tuning on the default short-horizon path should stay below configuration repair in the priority order
