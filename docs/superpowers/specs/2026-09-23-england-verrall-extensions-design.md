# England & Verrall Extensions — Design

**Date:** 2026-09-23
**Source reviewed:** https://github.com/DrPeterEngland/StochasticReserving (Python_Examples), MIT licence, Peter England / EMC Actuarial and Analytics Ltd.
**Papers:** England & Verrall (2002) *Stochastic Claims Reserving in General Insurance*; England & Verrall (2006) *Predictive distributions of outstanding liabilities in general insurance*; England, Verrall & Wüthrich (2019) *On the lifetime and one-year views of reserve risk, with application to IFRS 17 and Solvency II risk margins*.

## Goal

Bring the reserve-risk toolkit from the England repository into `bayesianchainladder` so that every estimator in the package (Bayesian GLM, CSR, Mack, ODP bootstraps, BF/CC) can feed a common set of downstream analyses: one-year Claims Development Result, discounted cash flows, cost-of-capital risk margins, IFRS 17 risk measures, fan charts, influence analysis, scaling to target ultimates. Add the two link-ratio bootstraps (Mack, Negative Binomial) that chainladder-python lacks, non-constant scale parameters, an analytic ODP oracle for tests, England's two sample triangles, and a runnable 15-step *modus operandi* notebook built on our API.

## Requirements

### R1. Per-cell simulated triangles (structural prerequisite)
Every `BaseStochasticReserve` subclass that produces simulations must populate `full_cumulative_posterior_`, an `xr.DataArray` with dims `("origin", "dev", "sample")` holding a complete simulated **cumulative** triangle per sample. Observed cells repeat the observed value across samples. `reserves_posterior_` must equal (last dev cumulative − latest observed) derived from it, within floating tolerance. `MackChainLadder` (normal approximation) leaves it `None` and downstream functions raise a clear error.

### R2. Tail summary statistics
A `summary_statistics()` method reporting mean, SD, CoV, min, max and the quantiles 0.5, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5 per origin and total, for reserves or ultimates. `MethodSummary` gains 99.5th percentile, min, max (defaulted so existing positional construction still works).

### R3. Risk measures and discounting (`riskmeasures.py`)
Pure functions: `value_at_risk`, `tail_value_at_risk`, `proportional_hazards_transform`, `discount_factors`, `discounted_reserves`, `future_reserve_profile`, `capital_profile`, `cost_of_capital_risk_margin`, `equivalent_risk_tolerance`. VaR uses `np.quantile`, not England's index formula. Discounting assumes origin grain equals development grain (square annual triangle); otherwise raise.

### R4. Claims Development Result (`cdr.py`)
`claims_development_result(model, future_periods=None, var_level=0.995, drop=None)` implementing England's *CDR_Full_Picture*: for each future period, re-reserve each simulated triangle with volume-weighted chain ladder on the cells known at that date; CDR_t = U_{t−1} − U_t. Return per-period, per-origin, per-sample arrays plus total, cumulative and reverse-cumulative views and a summary table (SD, VaR). Identity to test: the sum of CDRs over all periods equals the deterministic chain-ladder ultimate minus the simulated ultimate, exactly.

### R5. Non-constant scale and user-defined process variance
`CorrelatedBootstrapODPSample` gains `scale="constant" | "nonconstant"` (per-development-period φ_j with England's rules: last = min of previous two, carry forward when n_j ≤ 1, zero where the cumulative factor is 1) and `process_scale=None | array` to override φ at the forecast stage without changing the mean.

### R6. Link-ratio bootstraps (`linkratio.py`)
`MackBootstrap` and `NegativeBinomialBootstrap` estimators (England's *Main_Mack_Bstrap* / *Main_NegBin_Bstrap*): `bootstrap_dist` and `forecast_dist` each in `{"nonparametric", "gamma", "lognormal"}`, `drop` exclusions in chainladder `(origin_label, dev_months)` syntax, `process_sigma` override, `random_seed`. Normal fallback when a mean is non-positive. Populate R1. Bootstrap mean within 2% of chain ladder and SD within 10% of Mack's analytic SE on Taylor & Ashe.

### R7. Analytic ODP oracle (`analytic.py`)
`odp_analytic_rmsep(triangle, scale)` via numpy Poisson IRLS and the GLM prediction covariance (England's *ODP_ChainLadder*), plus `mack_analytic_rmsep` wrapping chainladder. Used in tests as the oracle for bootstrap SDs.

### R8. Influence analysis (`sensitivity.py`)
`link_ratio_sensitivity(triangle, drop=None)` leaves each link ratio out in turn, re-runs analytic Mack, and returns a ranked DataFrame of changes in total reserve, SD and CoV. `top_influential(result, n, by)` returns a chainladder-style `drop` list. On England's liability triangle the top SD-ranked ratio must be origin 3, development 6→7, i.e. `("2003", 72)`.

### R9. Scaling and incurred-to-paid
`BaseStochasticReserve.scale_to_target(target_ultimates, method)` with `"additive"` (preserves SD) or `"multiplicative"` (preserves CoV), per origin or global, returning a `ReserveSamples` container that exposes the shared summary interface. `incurred_to_paid(model, paid_triangle)` converts an incurred-basis ultimate distribution to a reserve distribution.

### R10. Plots
`plot_fan_chart`, `plot_scaled_residuals` (by origin / dev / calendar with sigma overlay), `plot_sensitivity_heatmap`, `plot_capital_profiles`.

### R11. Bayesian path additions (`models.py`, `estimators.py`)
`build_quasi_poisson_model` (Potential-based quasi-likelihood with scalar or per-dev φ plug-in), `build_link_ratio_model` (Normal on link ratios, Mack or NegBin variance, log or log-log link), and a `BayesianMackChainLadder` estimator that feeds posterior factor draws through the R6 forecasting machinery. Under flat priors, posterior mean factors must match volume-weighted chain ladder factors within 2%.

### R12. Data and documentation
Ship England's two triangles as package data with a README attributing the source and licence, plus `load_england_sample(name)`. Build the 15-step modus operandi as an executable notebook under `docs/notebooks/`, citing the source repository, using our estimators. Update README.md and CLAUDE.md.

## Reference values (chainladder 0.9.1, computed 2026-09-23)

| Triangle | CL total reserve | chainladder Mack total SE | England Mack SE |
|---|---|---|---|
| liability (England, 10×10) | 331,038 | 72,448 | 69,077 |
| liability, drop ("2003", 72) | 289,946 | 38,076 | — |
| Taylor & Ashe (= `genins`) | 18,680,856 | 2,441,364 | — |

The Mack SE gap on the liability triangle comes from the last-period sigma rule: chainladder extrapolates log-linearly (σ₉ = 10.1), England takes min(σ₇, σ₈) = 6.43.

## Out of scope
Tail factors / curve fitting; the Stan/CmdStanPy path; England's matplotlib table renderers; Merz–Wüthrich closed-form one-year formula.
