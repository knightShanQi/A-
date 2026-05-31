# 2026-04-13 Latest Quant Model Research Notes

This note summarizes recent time-series and financial forecasting research that is most relevant to this A-share project.

## What matters most for this project

### 1. Few-shot adaptation is valuable when local history is short or unstable

- Google Research showed that TimesFM can be extended from zero-shot forecasting to few-shot forecasting by teaching the model to use relevant in-context examples at inference time.
- Practical implication for this project:
  - Do not rely only on one stock's own history.
  - When a stock has unstable recent behavior, compare it with nearby context such as peer stocks, recent sector leaders, and the stock's own recent analogue windows.
  - This supports keeping peer-context and sector confirmation as auxiliary evidence rather than treating every name as fully independent.

### 2. Covariates are not optional in production forecasting

- Amazon's Chronos-2 emphasizes that multivariate and covariate-informed forecasting materially improves real-world performance.
- Practical implication for this project:
  - Keep fund flow, sector heat, macro calendar, and news pulse as explicit auxiliary variables.
  - Do not interpret price-only signals as sufficient when external drivers disagree.
  - Treat "price confirms + covariates confirm" as a higher-quality state than price-only breakout.

### 3. Mixture-of-experts is useful because market regimes are heterogeneous

- Moirai-MoE and Time-MoE both argue that human-defined heuristics such as frequency alone are too coarse, while sparse expert routing can specialize automatically and keep inference efficient.
- Practical implication for this project:
  - Continue using regime-aware gating and avoid one global threshold for all states.
  - Let trend, rebound, rotation, and defense states influence ranking and release criteria.
  - Prefer high-precision release rules that can behave differently across market states instead of forcing one universal cutoff.

### 4. Pretrained time-series models help most when data is noisy or limited

- A 2025 study on multivariate financial time series forecasting found that pretrained TSFMs improved performance by 25-50% on limited-data tasks and needed materially fewer years of data to reach similar performance.
- Practical implication for this project:
  - For newly active stocks, short trading histories, and unstable sub-regimes, transfer-style priors are valuable.
  - Use calibration and confidence filtering aggressively when sample support is low.
  - Do not treat small-sample backtests as equally trustworthy as large-sample ones.

### 5. Multimodal fusion helps, but only if the inputs are aligned and high quality

- FinMultiTime and FinZero both point toward the same direction: combining price series with news, charts, tables, and reasoning signals can improve prediction quality, especially in high-confidence groups.
- Practical implication for this project:
  - Keep multimodal confirmation, but make it secondary to alignment and confidence.
  - Penalize stale or low-confidence news and noisy fund-flow snapshots.
  - Prefer "price + minute structure + news/fund agreement" over raw feature stacking.

## How these findings are used right now

- Prediction review now compares previous-trading-day probabilities with the latest realized next-day move.
- Review metrics now include direction hit rate, calibration gap, Brier score, and target progress.
- The focus board keeps using precision-gated release logic so only historically stronger signals rise to the top.
- Intraday, news, sector, and fund-flow confirmation remain auxiliary covariates instead of decorative data.

## Follow-on ideas

- Add peer-series retrieval so each stock can borrow context from same-sector leaders before scoring.
- Add uncertainty intervals for board release, not only point probabilities.
- Split release thresholds by market regime instead of sharing a single gate.
- Version multimodal inputs by freshness and confidence to avoid overreacting to noisy external data.

## Source links

- Google Research, "Time series foundation models can be few-shot learners"
  - https://research.google/blog/time-series-foundation-models-can-be-few-shot-learners/
- Amazon Science, "Introducing Chronos-2: From univariate to universal forecasting"
  - https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting
- PMLR 2025, "Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts"
  - https://proceedings.mlr.press/v267/liu25an.html
- arXiv 2025, "Time Series Foundation Models for Multivariate Financial Time Series Forecasting"
  - https://arxiv.org/abs/2507.07296
- arXiv 2025, "FinZero: Launching Multi-modal Financial Time Series Forecast with Large Reasoning Model"
  - https://arxiv.org/abs/2509.08742
