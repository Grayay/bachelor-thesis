# Export Validation Report

## Reproducibility commands
- `python main.py`
- `python export_metrics.py`
- `python export_thesis_artifacts.py`

## Generated files
- `results/EXPORT_VALIDATION_REPORT.md`
- `results/crypto_exposure_summary.csv`
- `results/crypto_risk_contribution_summary.csv`
- `results/diversification_ratio.csv`
- `results/enb.csv`
- `results/enb_summary.csv`
- `results/figures/crisis_vs_noncrisis_cvar.png`
- `results/figures/crisis_vs_noncrisis_sharpe.png`
- `results/figures/crypto_risk_contribution.png`
- `results/figures/crypto_weights_over_time.png`
- `results/figures/cumulative_returns_selected.png`
- `results/figures/diversification_ratio_by_strategy.png`
- `results/figures/drawdowns_selected.png`
- `results/figures/enb_by_strategy.png`
- `results/figures/full_metrics_bar_cvar.png`
- `results/figures/full_metrics_bar_maxdd.png`
- `results/figures/full_metrics_bar_sharpe.png`
- `results/figures/hybrid_comparison_cumulative.png`
- `results/figures/hybrid_comparison_metrics.png`
- `results/figures/risk_contributions_hybrid_crypto.png`
- `results/figures/risk_contributions_hybrid_ruonia_capped_crypto.png`
- `results/figures/risk_contributions_rp_shrink_crypto.png`
- `results/figures/risk_contributions_selected.png`
- `results/figures/ruonia_weights_over_time.png`
- `results/figures/weights_aw_crypto.png`
- `results/figures/weights_dynamics_selected.png`
- `results/figures/weights_hybrid_crypto.png`
- `results/figures/weights_hybrid_ruonia_capped_crypto.png`
- `results/figures/weights_mv_shrink_crypto.png`
- `results/figures/weights_rp_shrink_crypto.png`
- `results/hybrid_diagnostics.csv`
- `results/l1_distances.csv`
- `results/metrics.csv`
- `results/metrics_crisis.csv`
- `results/metrics_full.csv`
- `results/metrics_non_crisis.csv`
- `results/oos_returns.csv`
- `results/patch_before_snapshot.json`
- `results/risk_contribution_summary.csv`
- `results/risk_contributions.csv`
- `results/ruonia_exposure_summary.csv`
- `results/ruonia_risk_contribution_summary.csv`
- `results/tables/table_base_vs_crypto_deltas.csv`
- `results/tables/table_best_worst_by_metric.csv`
- `results/tables/table_crypto_exposure.csv`
- `results/tables/table_final_candidate_shortlist.csv`
- `results/tables/table_hybrid_comparison.csv`
- `results/tables/table_metrics_crisis.csv`
- `results/tables/table_metrics_full.csv`
- `results/tables/table_metrics_non_crisis.csv`
- `results/tables/table_risk_contributions.csv`
- `results/tables/table_ruonia_exposure.csv`
- `results/weights_history.csv`
- `results/weights_summary.csv`

## Final strategy list
ew_base, ew_crypto, mv_sample_base, mv_sample_crypto, mv_shrink_base, mv_shrink_crypto, rp_sample_base, rp_sample_crypto, rp_shrink_base, rp_shrink_crypto, aw_base, aw_crypto, hybrid_base, hybrid_crypto, hybrid_loose_base, hybrid_loose_crypto, hybrid_unconstrained_base, hybrid_unconstrained_crypto, hybrid_ruonia_capped_base, hybrid_ruonia_capped_crypto

## Shapes and ranges
- Dataset shape: (498, 6)
- Dataset date range: 2016-03-20 .. 2026-03-15
- OOS returns shape: (342, 20)
- weights_history shape: (2700, 5)
- risk_contributions shape: (2700, 8)
- diversification_ratio shape: (540, 3)
- ENB shape: (540, 3)

## Sanity checks
### Passed
- weights sum to 1 by rebalance_date×strategy
- no negative weights
- no missing weights
- base strategies do not include BTC/ETH
- crypto strategies include BTC/ETH
- risk contribution values are finite
- sum(rc_share) approximately 1 by rebalance_date×strategy
### Failed
- None

## Top 5 strategies by Sharpe
- mv_sample_crypto: 13.012820
- mv_sample_base: 12.991850
- mv_shrink_crypto: 1.241030
- mv_shrink_base: 1.162590
- ew_crypto: 1.139480

## Top 5 strategies by CVaR (higher is better / less negative)
- mv_sample_crypto: 0.000440
- mv_sample_base: 0.000440
- mv_shrink_base: -0.015720
- mv_shrink_crypto: -0.017030
- rp_shrink_base: -0.023700

## Top 5 strategies by DR
- rp_shrink_crypto: 1.977700
- rp_shrink_base: 1.948930
- mv_shrink_crypto: 1.824910
- rp_sample_crypto: 1.761640
- hybrid_unconstrained_crypto: 1.761640

## Top 5 strategies by ENB
- rp_shrink_crypto: 5.777751
- hybrid_unconstrained_crypto: 5.078470
- rp_sample_crypto: 5.078470
- hybrid_loose_crypto: 4.745402
- hybrid_crypto: 4.461038

## Key conclusions
- Crypto effect: generally increases CAGR and diversification for many families, but often increases volatility, CVaR loss, and drawdown.
- RUONIA dominance: visible in defensive strategies; explicit RUONIA cap diagnostics are now exportable via weights and RUONIA summaries.
- MV behavior: constrained MV remains very defensive; sample-MV can collapse toward low-volatility assets.
- Hybrid behavior: variants are distinct; RUONIA-capped variants allow explicit cap-binding diagnostics.
- Final candidate strategies (shortlist, not final selection): `rp_shrink_base`, `rp_shrink_crypto`, `aw_crypto`, `hybrid_ruonia_capped_crypto`, `mv_shrink_crypto`.

## ENB note
ENB is implemented as a risk-contribution-based ENB proxy: ENB_RC_Proxy = 1 / sum(rc_share_i^2).

## Sortino note
Sortino is undefined when downside deviation equals zero; exported thesis tables display it as N/A.