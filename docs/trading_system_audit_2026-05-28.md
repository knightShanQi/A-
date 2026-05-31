# Trading System Audit 2026-05-28

## Executive Summary

The current system has a clear architecture and a large amount of historical validation work, but the trading stack still has four structural problems:

1. The predictive alpha is weak.
2. The discretionary strategy layer destroys rather than amplifies alpha.
3. Several backtest paths measure "selection hit rate" more than true portfolio capital growth.
4. Execution realism is not strong enough for A-share trading constraints.

This is why the system can occasionally produce seemingly acceptable filtered annualized results, while the broad strategy/model outputs remain low-return or negative.

The most important conclusion is this:

- Your main bottleneck is not lack of indicators.
- Your bottleneck is mismatch between objective, ranking logic, and portfolio construction.

## Evidence Snapshot

### 1. Model alpha is close to random

Source: `.cache/ten_year_model_strategy_validation/summary.json`

- `model_accuracy.overall_avg_return = 0.001617`
- `model_accuracy.overall_win_rate = 0.4833`
- `auc_positive_return = 0.5103`
- `auc_target_return = 0.5916`

Interpretation:

- For binary up/down prediction, AUC near `0.50` means the model has very limited direction discrimination power.
- Even the target-return AUC of `0.5916` is only modest and not strong enough to support aggressive daily ranking over the whole universe.

Source: `.cache/next_day_factor_model/model_metrics.json`

- Validation/test `roc_auc` is about `0.5045` / `0.5049`.
- Test `top_decile_return = 0.000552`, which is very small.

Source: `.cache/next_day_factor_model_optimized/model_metrics.json`

- Validation threshold is pushed to `0.6427`.
- `positive_prediction_rate` is compressed to about `2%`.
- Test `roc_auc = 0.4945`.
- `accuracy_lift_vs_majority < 0`.

Interpretation:

- The "optimized" version improves apparent classification stability by predicting far less often, not by learning stronger edge.
- Once thresholding becomes this aggressive, the model is functioning more like a sparse filter than a true ranking engine.

### 2. Strategy layer is negative expectancy

Source: `.cache/ten_year_model_strategy_validation/summary.json`

- `strategy_accuracy.overall_avg_return = -0.002494`
- `strategy_accuracy.overall_win_rate = 0.4283`
- `strategy_only_top1 annualized_return = -0.916319`
- `strategy_only_top3 annualized_return = -0.798741`

Interpretation:

- The handcrafted strategy candidate layer is not selecting better entries than the raw universe. It is actively selecting worse entries.
- This means your post-model business rules are currently introducing adverse selection.

### 3. Full-system combined strategy is only weakly positive after hard filtering

Source: `.cache/ten_year_model_strategy_validation/summary.json`

- Best combined rule: `combined_top3_score_ge_68_full`
- `avg_trade_return = 0.002641`
- `trade_win_rate = 0.4627`
- `annualized_return = 0.034405`
- `max_drawdown = -0.359647`

Interpretation:

- The only surviving positive rule is extremely selective and still only produces low annualized return with large drawdown.
- This is not a capital-efficient production strategy yet.

### 4. Regime and bull-market overlays help, but mostly by reducing exposure

Source: `.cache/ten_year_market_regime_v3/v3_summary.json`

- `v2_score68_top3 annualized_return = 0.094702`, `max_drawdown = -0.216513`
- `v3_full_green_top3 annualized_return = 0.058951`, `max_drawdown = -0.164038`
- `full_green` days are only `187 / 2423`

Source: `.cache/ten_year_bull_market_rank_score/bull_market_rank_score80_summary.json`

- `no_bull_filter + rank_sorted annualized_return = 0.109895`, `max_drawdown = -0.473159`
- `no_bull_filter + model_sorted annualized_return = 0.003638`, `max_drawdown = -0.388086`

Interpretation:

- Regime filters are improving results mainly by trading less and concentrating exposure.
- The rank-based rebuilt score is contributing more than the model score itself, which is a warning sign that the model is not the real source of edge.

## Technical Problems By Layer

### A. Objective mismatch: model is trained for direction, strategy is monetized on ranking

Files:

- `src/a_share_predictor/modeling.py`
- `src/a_share_predictor/next_day_factor_model.py`

What is happening:

- The system is largely framed as next-day up/down probability prediction.
- The actual trading system uses those scores to rank stocks and construct top-N candidate sets.

Why this hurts annualized return:

- A model can be slightly useful for classification but still poor for cross-sectional ranking.
- Your own evidence shows this. The model has low AUC, but the rank-rebuilt heuristic can outperform model-sorted portfolios.

Root cause:

- Training target and portfolio objective are misaligned.
- You are optimizing for binary discrimination, while the portfolio needs cross-sectional excess return ranking after costs.

### B. Handcrafted score stack is too dense and too weakly identified

Files:

- `src/a_share_predictor/strategy.py`
- `src/a_share_predictor/stages.py`

Examples:

- `strategy_score` is built from manually weighted inputs in `strategy.py`.
- `window_score` and `execution_score` are also manually weighted.

Why this hurts:

- The scoring stack contains many heuristics, but little evidence that each term contributes stable out-of-sample PnL.
- When a score becomes a weighted average of many correlated soft signals, it often creates a smooth ranking that looks plausible but carries weak incremental edge.

Observed symptom:

- The strategy layer is negative even before realistic execution penalties are strengthened.

### C. Backtest logic is not portfolio-first

Files:

- `src/a_share_predictor/backtesting.py`
- `src/a_share_predictor/market_backtest_runner.py`
- `scripts/backtest_market_regime_v3.py`
- `scripts/backtest_bull_market_rank_score.py`

Specific issues:

1. `backtesting.py` uses a sequential single-position gate:
   - `if entry_pos < next_free_entry: continue`
   - This forces one trade to block later signals.

2. It annualizes using calendar elapsed days:
   - `annualized_return = (1 + cumulative_return) ** (365 / elapsed_days) - 1`

3. `market_backtest_runner.py` summarizes average forward return and average max drawdown:
   - `avg_forward_return`
   - `avg_max_high_return`
   - `avg_max_drawdown`

4. The regime scripts construct equity from daily mean selected returns, not from a capital-constrained execution book.

Why this hurts:

- Some paths understate opportunity because of single-position serial blocking.
- Other paths overstate quality because averaging selected returns is not the same as compounding a realistic portfolio with turnover, overlaps, and cash drag.
- This creates inconsistent optimization pressure: one module optimizes for hit rate, another for average return, another for sparse exposure.

### D. Tradability assumptions are too optimistic for A-share reality

Files:

- `src/a_share_predictor/market_backtest_runner.py`

Examples:

- Entry assumes next open is available.
- Exit evaluates future close and also `max_high_return`.
- `target_hit_rate` is based on whether future max high exceeded threshold.

Why this hurts:

- In A-shares, gap opens, one-word boards, weak liquidity, and T+1 constraints matter.
- `max_high_return` is not directly monetizable unless you define a realizable intraday exit policy.
- A strategy that "touches target intraday" can still be hard to execute profitably.

### E. Rebuilt ranking score risks historical inconsistency

File:

- `scripts/backtest_bull_market_rank_score.py`

Problem:

- The script explicitly rebuilds `rank_score_rebuilt` because historical live rank score was not cached.
- It also ranks within each day's candidate pool using `signal_rank`.

Why this matters:

- This is useful for research, but it is not identical to the historical live decision surface.
- If the rebuilt rank is the main driver of positive performance while the model score is weak, then the strategy may be leaning on a synthetic backtest-only ranking artifact.

## Why Annualized Return Is Not High

From a professional trading-system perspective, your annualized return is low because all three multipliers are weak:

`annualized return = edge per trade x usable trade frequency x capital efficiency`

### 1. Edge per trade is weak

- Model alpha is weak.
- Strategy candidate alpha is negative.
- Combined filtered edge is only slightly positive.

### 2. Usable trade frequency is low after filtering

- Positive rules survive only under strong gating such as score thresholds and full-green regimes.
- `v3_full_green` occurs on very few days.

### 3. Capital efficiency is poor

- The best rules often trade only a small fraction of days.
- Idle cash periods are large.
- Drawdown remains high relative to annualized return.

### 4. Ranking signal is unstable

- Model-sorted results are materially worse than rank-sorted rebuilt results.
- This means the system does not yet have a stable primary ranking signal that survives across market states.

### 5. Portfolio construction is primitive

- Top-N equal treatment dominates.
- No evidence of risk budgeting by regime, volatility, liquidity, or correlation cluster.
- No evidence of portfolio turnover penalty or crowding penalty in the main evaluation path.

## Optimization Plan

## Priority 1: Rebuild the objective around cross-sectional return ranking

Do this first.

- Replace the core training objective from "next day up/down" to one of:
  - cross-sectional rank IC
  - top-decile excess return
  - future risk-adjusted return
  - pairwise ranking loss / LambdaMART-style ranking
- Train by market date slices, not only pooled binary labels.
- Measure:
  - daily rank IC
  - top-minus-bottom spread
  - top-K portfolio return after cost
  - regime-conditional IC

Expected impact:

- This aligns the model with the portfolio construction problem.

## Priority 2: Remove weak heuristic layers and validate incremental value

- Break the strategy score stack into atomic factors.
- For each factor, measure incremental contribution:
  - conditional return uplift
  - conditional drawdown reduction
  - interaction with regime
- Delete any feature/rule that does not improve out-of-sample portfolio metrics.

Practical rule:

- Every handcrafted filter must justify its existence by improving `top-K post-cost return` or reducing drawdown without killing too much coverage.

## Priority 3: Unify all backtests into one portfolio simulator

Build one authoritative simulator with:

- position budget
- max concurrent holdings
- cash drag
- T+1 constraint
- slippage
- transaction cost
- limit-up / limit-down tradability rules
- overlap handling
- daily portfolio NAV
- turnover and capacity statistics

Stop using multiple inconsistent notions of success:

- average selected return
- active-day return
- max-high target hit
- calendar-as-cash annualization

All research modules should eventually optimize against the same portfolio NAV engine.

## Priority 4: Rebuild the entry/exit logic with realizable execution

- Replace `max_high_return` evaluation with explicit executable exits:
  - next-day open to fixed holding close
  - next-day open to trailing stop
  - VWAP-style simulated exit
  - open + intraday stop + end-of-day exit
- Add tradability filters:
  - open gap > x
  - one-word limit boards
  - amount threshold
  - turnover threshold
  - free-float proxy if available

Expected impact:

- Reported return will likely decrease at first, but real deployability will increase sharply.

## Priority 5: Regime should control exposure, not define alpha

Current state:

- Regime filters are helping mostly by avoiding bad days.

Upgrade path:

- Keep regime as exposure controller:
  - gross exposure
  - max position count
  - threshold adjustment
  - holding horizon adjustment
- Do not rely on regime rules to manufacture alpha that the stock model does not have.

## Priority 6: Build a proper portfolio ranking stack

Recommended layered ranking:

1. Base alpha score
   - model-predicted expected return

2. Quality adjustment
   - stability of signal
   - calibration confidence
   - recent model degradation

3. Execution adjustment
   - liquidity
   - gap risk
   - turnover
   - board-specific trading constraints

4. Diversification penalty
   - industry concentration
   - correlation cluster overlap
   - crowding

This is much more robust than a large handcrafted blended score.

## Priority 7: Tighten validation standards

A candidate strategy should not proceed unless it passes all of:

- out-of-sample top-K annualized return > benchmark
- max drawdown within explicit budget
- positive performance in at least 3 market subperiods
- stable IC or spread by year
- performance survives transaction-cost stress
- performance survives delayed entry stress
- performance survives threshold perturbation

## Recommended 6-Week Refactor Sequence

### Week 1

- Freeze current artifacts.
- Create one authoritative portfolio backtest engine.
- Re-run existing top rules under realistic execution assumptions.

### Week 2

- Replace binary target reports with ranking reports:
  - daily IC
  - top-K spread
  - turnover-adjusted spread

### Week 3

- Audit all handcrafted strategy filters.
- Keep only filters with positive incremental contribution.

### Week 4

- Train cross-sectional ranking model with rolling walk-forward splits.
- Report by regime and by year.

### Week 5

- Add portfolio construction:
  - max 3 to 10 names
  - volatility scaling
  - sector cap
  - liquidity cap

### Week 6

- Rebuild bull/regime overlays as exposure controllers on top of the new ranker.
- Compare to:
  - raw model top-K
  - filtered model top-K
  - risk-budgeted top-K

## Highest-Value Immediate Actions

If you only do three things, do these:

1. Replace binary classification success metrics with cross-sectional portfolio metrics.
2. Build one realistic unified portfolio simulator and retire average-return-only evaluation.
3. Remove or retrain the current heuristic strategy candidate layer, because it is presently negative expectancy.

## Final Judgment

From a professional quant/architecture standpoint:

- The system is not failing because it lacks complexity.
- It is failing because complexity is concentrated in heuristic scoring and fragmented validation, while the core alpha and portfolio objective are not tightly aligned.

You already have enough infrastructure to move to the next stage.
The next stage is not "add more factors".
The next stage is "make one truthful portfolio engine, one aligned ranking target, and one evidence-based execution layer."
