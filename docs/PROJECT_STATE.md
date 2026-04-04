# Project State

## Thesis
Bachelor thesis at HSE, Economics and Statistics.

Current working title:
Development of an Investment Strategy Based on the Extended All-Weather Portfolio Concept

## Final empirical scope
The final empirical version of the thesis focuses on a Russian multi-asset portfolio with cryptocurrency inclusion.

Digital financial assets (DFAs / CFA) are NOT included in the empirical model because of limited data availability, short history, low comparability, and weak suitability for a consistent long-horizon portfolio backtest. They will be discussed briefly in limitations and future research.

## Asset universe actually used in the empirical model
- MOEX equity index
- OFZ / Russian government bonds
- Gold in RUB
- RUONIA as money market / short defensive proxy
- Bitcoin in RUB
- Ethereum in RUB

Important clarification:
- There is no separate long OFZ and short OFZ empirical implementation in the final dataset.
- RUONIA is used as the short defensive / cash-like leg.
- Any outdated text mentioning separate long/short OFZ segments must be corrected.

## Data frequency and sample
- Weekly data
- Period: 2016–2025
- Currency: RUB
- Returns: log returns

## Portfolio framework
- Rolling window: 156 weeks
- Rebalancing: quarterly
- Long-only
- Crypto cap: max 10% per cryptocurrency
- Out-of-sample evaluation

## Core strategies
- Equal Weight
- Minimum Variance
- Risk Parity
- All-Weather
- Hybrid All-Weather + Risk Parity

## Current priorities
1. Verify `data/final_dataset.csv`
2. Debug and validate `results/oos_returns.csv`
3. Recompute metrics from a single correct source of truth
4. Add missing statistical inference and robustness
5. Align thesis text strictly with actual empirical implementation
6. Final cleanup of formatting, tables, formulas, and figures

## Rules for this project
- Do not assume that the thesis text is correct if it conflicts with code or csv outputs.
- Code and reproducible outputs are the source of truth.
- Every new result must be written into `RESULTS_LOG.md`.
- Any change in model scope must be reflected here immediately.