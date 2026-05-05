# PATCH Validation Report

## Files changed
- `src/rpallw.py`
- `export_thesis_artifacts.py`

## Applied fixes
- Hybrid unconstrained consistency: `compute_hybrid_weights(..., regime="unconstrained")` now uses the same RP initialization path as `rp_sample` (default RP init).
- RUONIA cap reporting tolerance: exceedance uses `weight > 0.250001`; cap binding uses `abs(weight - 0.25) <= 1e-6`.
- Crisis/non-crisis thesis tables: `DiversificationRatio` renamed to `FullSampleDiversificationRatio`.
- Thesis table display: `Sortino = inf` exported as `N/A` in thesis tables only.
- ENB wording clarified as risk-contribution-based ENB proxy (`ENB_RC_Proxy` label in best/worst table; note retained in reports).

## Before/after: hybrid_unconstrained vs rp_sample (max abs differences)
### base
- Weights max abs: before `0.9016170074` -> after `0.0000000000`
- OOS returns max abs: before `0.0299529238` -> after `0.0000000000`
- DR path max abs: before `0.2886401399` -> after `0.0000000000`
- ENB path max abs: before `0.5109588628` -> after `0.0000000000`

### crypto
- Weights max abs: before `0.0000006502` -> after `0.0000000000`
- OOS returns max abs: before `0.0000000581` -> after `0.0000000000`
- DR path max abs: before `0.0000002749` -> after `0.0000000000`
- ENB path max abs: before `0.0000005365` -> after `0.0000000000`

## Changed metrics for affected strategies
Metrics below show `hybrid_unconstrained_base - rp_sample_base` delta before vs after:
- `CAGR`: before `0.00166000`, after `0.00000000`, change `-0.00166000`
- `Volatility`: before `0.00670000`, after `0.00000000`, change `-0.00670000`
- `Sharpe`: before `-0.06224000`, after `0.00000000`, change `0.06224000`
- `Sortino`: before `-0.04129000`, after `0.00000000`, change `0.04129000`
- `CVaR_5%`: before `-0.00123000`, after `0.00000000`, change `0.00123000`
- `DiversificationRatio`: before `-0.04154000`, after `0.00000000`, change `0.04154000`
- `DiversificationRatioMean`: before `-0.04154000`, after `0.00000000`, change `0.04154000`
- `DiversificationRatioMedian`: before `-0.01274000`, after `0.00000000`, change `0.01274000`

## RUONIA cap exceedance after tolerance fix
- `hybrid_ruonia_capped_base`: share_dates_ruonia_gt_25=`0.0`, direct recompute (>0.250001)=`0.0`, bind_share=`0.5185185185185185`, max_ruonia_weight=`0.25`
- `hybrid_ruonia_capped_crypto`: share_dates_ruonia_gt_25=`0.0`, direct recompute (>0.250001)=`0.0`, bind_share=`0.2592592592592592`, max_ruonia_weight=`0.25`

## Sanity checks
- weights sum to 1: PASS (value=4.440892098500626e-16)
- no negative weights: PASS (value=0)
- rc_share sums to 1: PASS (value=5.551115123125783e-16)

## Strategy list and shapes stability
- Strategy list unchanged: True
- Shapes unchanged (OOS/weights/DR/ENB): True
- OOS shape: before [342, 20], after [342, 20]
- weights_history shape: before [2700, 5], after [2700, 5]
- diversification_ratio shape: before [540, 3], after [540, 3]
- enb shape: before [540, 3], after [540, 3]

## Additional confirmations
- `table_metrics_crisis.csv` contains `FullSampleDiversificationRatio`: True
- `table_metrics_non_crisis.csv` contains `FullSampleDiversificationRatio`: True
- Sortino `N/A` count in crisis thesis table: 2
- Sortino `N/A` count in non-crisis thesis table: 2

## Remaining caveats for thesis wording
- ENB should be described as a risk-contribution-based ENB proxy, not classical Meucci ENB.
- `FullSampleDiversificationRatio` in crisis/non-crisis tables is full-sample by construction; regime interpretation should use `DiversificationRatioMean`/`DiversificationRatioMedian`.
- `ew_crypto` remains intentionally uncapped naive benchmark (unchanged).