# Bayesian Chain Ladder

A Python package for Bayesian stochastic loss reserving using generalized linear models, built on [Bambi](https://bambinos.github.io/bambi/) and [PyMC](https://www.pymc.io/).

## Overview

This package implements Bayesian versions of the cross-classified chain ladder model for actuarial loss reserving. It integrates with the [chainladder-python](https://chainladder-python.readthedocs.io/) package for data handling and follows the methodology described in Taylor & McGuire (2016).

Key features:

- **Bayesian GLM framework**: Full posterior distributions for reserves and ultimates
- **Flexible formulas**: Patsy-style formulas for model specification
- **Multiple families**: Support for Negative Binomial, Poisson, Gamma distributions
- **Calendar year effects**: Optional calendar period trends
- **Exposure offsets**: Support for exposure-based methods
- **ArviZ integration**: Comprehensive diagnostics and visualization

## Installation

```bash
# Using uv (recommended)
uv sync

# Using pip
pip install bayesianchainladder
```

## Quick Start

```python
import chainladder as cl
from bayesianchainladder import BayesianChainLadderGLM

# Load a sample triangle
triangle = cl.load_sample("clrd").loc[("New Jersey Manufacturers Grp", "wkcomp"), "CumPaidLoss"]

# Fit the Bayesian chain ladder model
model = BayesianChainLadderGLM(
    formula="incremental ~ 1 + C(origin) + C(dev)",
    family="negativebinomial",
    draws=2000,
    tune=1000,
)
model.fit(triangle)

# Get reserve summary
print(model.summary())

# Access posterior samples
reserve_samples = model.sample_reserves(n_samples=1000)
```

## Model Specification

### The Cross-Classified Chain Ladder Model

The standard ODP (Over-Dispersed Poisson) cross-classified chain ladder model is:

```
log(μ_kj) = intercept + α_k + β_j
```

where:
- `μ_kj` is the expected incremental loss for origin `k`, development period `j`
- `α_k` are origin (accident year) effects
- `β_j` are development period effects

### Formulas

The package uses Bambi/Patsy-style formulas. Common specifications:

```python
# Standard cross-classified model
formula = "incremental ~ 1 + C(origin) + C(dev)"

# With calendar year effects
formula = "incremental ~ 1 + C(origin) + C(dev) + C(calendar)"

# Continuous trends (if supported by data)
formula = "incremental ~ 1 + C(origin) + dev"
```

### Distribution Families

Supported distribution families:

- `"negativebinomial"` (default): Overdispersed Poisson-like, suitable for count data with extra variance
- `"poisson"`: Standard Poisson for count data
- `"gamma"`: For positive continuous data

### Exposure-Based Methods

To include exposure as an offset term:

```python
import chainladder as cl
from bayesianchainladder import BayesianChainLadderGLM

# Load triangle with exposure information
triangle = cl.load_sample("clrd")["CumPaidLoss"]
exposure = cl.load_sample("clrd")["EarnedPremDIR"]

model = BayesianChainLadderGLM(
    formula="incremental ~ 1 + C(origin) + C(dev)",
    exposure="exposure",  # Name of exposure column
)
model.fit(triangle, exposure_triangle=exposure)
```

## Diagnostics and Visualization

The package provides comprehensive diagnostic plots via ArviZ:

```python
from bayesianchainladder import (
    plot_trace,
    plot_ppc,
    plot_reserve_distribution,
    plot_residuals,
    plot_heatmap_residuals,
)

# Trace plots for convergence diagnostics
fig, ax = plot_trace(model)

# Posterior predictive checks
fig, ax = plot_ppc(model)

# Reserve distribution by origin
fig, ax = plot_reserve_distribution(model, by="origin")

# Total reserve distribution
fig, ax = plot_reserve_distribution(model, by="total")

# Residual plots
fig, ax = plot_residuals(model, by="dev")
fig, ax = plot_residuals(model, by="calendar")

# Heat map of actual/expected ratios
fig, ax = plot_heatmap_residuals(model)
```

## Output Attributes

After fitting, the model provides several attributes:

```python
# Full posterior samples (ArviZ InferenceData)
model.idata

# Fitted values for observed cells
model.fitted_

# Reserve summary by origin year
model.ibnr_

# Ultimate loss summary by origin year
model.ultimate_

# Full posterior samples of reserves
model.reserves_posterior_
```

## Model Comparison

Use WAIC or LOO-CV for model comparison:

```python
from bayesianchainladder import compute_waic, compute_loo

# WAIC
waic = compute_waic(model.idata)

# LOO-CV
loo = compute_loo(model.idata)
```

## Reserve risk toolkit (England & Verrall extensions)

Every estimator that simulates future cells exposes `full_cumulative_posterior_`
(dims `origin, dev, sample`). The following build on it and therefore work for the
Bayesian GLM, CSR, the ODP bootstraps and the link-ratio bootstraps alike.

```python
from bayesianchainladder import (
    MackBootstrap, NegativeBinomialBootstrap, BayesianMackChainLadder,
    claims_development_result, discounted_reserves, future_reserve_profile,
    cost_of_capital_risk_margin, value_at_risk, tail_value_at_risk,
    proportional_hazards_transform, equivalent_risk_tolerance,
    link_ratio_sensitivity, top_influential, mack_analytic_rmsep, odp_analytic_rmsep,
    plot_fan_chart, plot_scaled_residuals, load_england_sample,
)

tri = load_england_sample("liability")

# Influence analysis, then a Mack bootstrap excluding the top-3 ratios
sens = link_ratio_sensitivity(tri)
boot = MackBootstrap(n_sims=10_000, drop=top_influential(sens, 3), random_seed=1).fit(tri)
boot.summary_statistics("reserves")          # mean, sd, cov, min, 0.5%..99.5%, max
plot_fan_chart(boot, origin=2004)

# One-year view and Solvency II / IFRS 17 quantities
cdr = claims_development_result(boot)         # CDR per future period, origin, sample
disc = discounted_reserves(boot, rate=0.03)   # discounted cash flows per origin
profile = future_reserve_profile(boot, 0.03).mean("sample").values
rm = cost_of_capital_risk_margin(cdr.summary().query("origin == 'Total' and future_period == 1")["var"].iloc[0],
                                 profile / profile[0], coc_rate=0.06, discount_rate=0.03)

# Scale to booked ultimates, preserving CoV
paid = boot._paid_to_date().reindex(boot.reserves_posterior_.origin.values)
target_ultimates = paid + 1.1 * boot.ibnr_["mean"]   # e.g. booked ultimates
scaled = boot.scale_to_target(target_ultimates, method="multiplicative")
```

| Feature | Function / class | Source |
|---|---|---|
| Mack / NegBin bootstraps (nonparametric, Gamma, Lognormal) | `MackBootstrap`, `NegativeBinomialBootstrap` | England & Verrall (2002, 2006) |
| Bayesian link-ratio model | `BayesianMackChainLadder`, `build_link_ratio_model` | England & Verrall (2006) §6 |
| Quasi-Poisson likelihood with per-dev dispersion | `build_quasi_poisson_model` | England & Verrall (2006) |
| Non-constant scale, user-defined process variance | `CorrelatedBootstrapChainLadder(scale=, process_scale=)` — `scale="nonconstant"` changes the process-variance (forecast) stage only, equivalent to `scale="constant", process_scale=sampler.scale_by_dev_`; the resampling stage still uses the globally pooled residuals | England & Verrall (2006) |
| One-year Claims Development Result | `claims_development_result` | England, Verrall & Wüthrich (2019) |
| Discounting, capital profiles, cost-of-capital margin, VaR/TVaR/PHT | `bayesianchainladder.riskmeasures` | England, Verrall & Wüthrich (2019) |
| Influential link ratios | `link_ratio_sensitivity`, `top_influential` | England, *Modus Operandi* |
| Analytic RMSEP oracles | `mack_analytic_rmsep`, `odp_analytic_rmsep` | England & Verrall (2002) |
| Scaling / incurred-to-paid | `scale_to_target`, `incurred_to_paid` | England, *Modus Operandi* |
| Sample data | `load_england_sample("taylor_ashe")` / `load_england_sample("liability")` | England's repository |

A worked 15-step example is in `docs/notebooks/modus_operandi.ipynb`. These
features are adapted from Peter England's
[StochasticReserving](https://github.com/DrPeterEngland/StochasticReserving)
repository (MIT licence).

## Advanced Usage

### Custom Priors

Specify custom priors using Bambi syntax:

```python
import bambi as bmb

priors = {
    "Intercept": bmb.Prior("Normal", mu=10, sigma=5),
    "C(origin)": bmb.Prior("Normal", mu=0, sigma=1),
    "C(dev)": bmb.Prior("Normal", mu=0, sigma=1),
}

model = BayesianChainLadderGLM(
    formula="incremental ~ 1 + C(origin) + C(dev)",
    priors=priors,
)
```

### Accessing Parameters

```python
# Get parameter summary
params = model.get_parameter_summary()

# Origin effects
origin_effects = model.get_origin_effects()

# Development effects
dev_effects = model.get_development_effects()
```

### PyMC Backend

For more control, you can use the PyMC model builder directly:

```python
from bayesianchainladder import build_pymc_model, triangle_to_dataframe

data = triangle_to_dataframe(triangle)
pymc_model = build_pymc_model(
    data,
    response_col="incremental",
    origin_col="origin",
    dev_col="dev",
    family="negativebinomial",
)

# Customize further with PyMC
with pymc_model:
    # Add custom components
    pass
```

## API Reference

### Main Classes

- `BayesianChainLadderGLM`: Main estimator class
- `MackBootstrap`: Nonparametric/Gamma/Lognormal bootstrap of the Mack chain ladder
- `NegativeBinomialBootstrap`: Nonparametric/Gamma/Lognormal bootstrap of a NegBin link-ratio model
- `BayesianMackChainLadder`: Bayesian link-ratio chain ladder (NUTS)
- `ReserveSamples`: Container for per-cell/per-origin reserve posterior samples

### Model Building

- `build_bambi_model()`: Create a Bambi model
- `build_pymc_model()`: Create a PyMC model directly
- `fit_model()`: Fit a model using MCMC

### Utilities

- `triangle_to_dataframe()`: Convert chainladder Triangle to DataFrame
- `prepare_model_data()`: Prepare data for modeling
- `validate_triangle()`: Validate triangle input

### Plotting

- `plot_trace()`: MCMC trace plots
- `plot_ppc()`: Posterior predictive checks
- `plot_reserve_distribution()`: Reserve distributions
- `plot_residuals()`: Residual plots
- `plot_heatmap_residuals()`: Heat map of A/E ratios
- `plot_development_pattern()`: Development pattern visualization

### Reserve risk toolkit (England & Verrall extensions)

- `bayesianchainladder.analytic`: Analytic ODP and Mack RMSEP oracles
- `bayesianchainladder.cdr`: One-year and multi-year Claims Development Result
- `bayesianchainladder.datasets`: England's Taylor-Ashe and liability sample triangles
- `bayesianchainladder.linkratio`: Mack/NegBin bootstraps and the Bayesian link-ratio model
- `bayesianchainladder.riskmeasures`: Discounting, capital profiles, cost-of-capital margin, VaR/TVaR/PHT
- `bayesianchainladder.sensitivity`: Leave-one-ratio-out influence analysis

## References

- Taylor, G. and McGuire, G. (2016). *Stochastic Loss Reserving Using Generalized Linear Models*. CAS Monograph Series Number 3.
- England, P.D. and Verrall, R.J. (2002). Stochastic Claims Reserving in General Insurance. *British Actuarial Journal*, 8(3), 443-518.
- England, P.D. and Verrall, R.J. (2006). Predictive Distributions of Outstanding Liabilities in General Insurance. *Annals of Actuarial Science*, 1(2), 221-270.
- England, P.D., Verrall, R.J. and Wüthrich, M.V. (2019). On the lifetime and one-year views of reserve risk, with application to IFRS 17 and Solvency II risk margins. *Insurance: Mathematics and Economics*, 85, 74-88.

## License

MIT License
