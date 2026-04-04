# Current Issues

## Critical issues

### 1. Scope inconsistency in thesis text
The thesis text still mentions:
- digital financial assets / CFA in the empirical model
- separate long OFZ and short OFZ structure

But the actual merged dataset currently contains only:
- BTC
- ETH
- Gold
- RUONIA
- MOEX
- OFZ

This inconsistency must be fixed later in the thesis text.

### 2. Out-of-sample returns may be incorrect
There is a serious suspicion that `results/oos_returns.csv` can contain misleading values for some strategies if the return construction or interpretation is wrong (historical note: fixed in code; keep validating).

Observed anomaly:
- some minimum variance strategy outputs appear unrealistically strong
- possible symptoms: all-positive weeks, zero drawdown, implausibly high Sharpe ratio, or mismatch with metrics tables

### 3. Thesis claims exceed implemented evidence
The text mentions or implies:
- Ledoit-Wolf Sharpe difference test
- block bootstrap confidence intervals
- diversification ratio
- effective number of bets
- marginal risk contributions
- marginal CVaR contributions
- robustness and crisis comparisons

Not all of these are currently implemented, verified, or exported.

### 4. Repeated and inconsistent text
There are duplicated paragraphs, unfinished sections, broken table of contents, and old wording that no longer matches the actual model.

## Immediate operational goal
Before doing any text revision, verify:
1. whether `data/final_dataset.csv` is valid
2. whether `results/oos_returns.csv` truly contains correct portfolio out-of-sample log returns
3. whether summary metrics match direct recomputation from `results/oos_returns.csv`

## Working hypothesis
The highest priority bug is in the OOS pipeline, not in the thesis wording.