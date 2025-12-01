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
# Using Poetry (recommended)
poetry install

# Using pip
pip install bayesianchainladder
```

## Quick Start

```python
import chainladder as cl
from bayesianchainladder import BayesianChainLadderGLM

# Load a sample triangle
triangle = cl.load_sample("GenIns")

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

## References

- Taylor, G. and McGuire, G. (2016). *Stochastic Loss Reserving Using Generalized Linear Models*. CAS Monograph Series Number 3.
- England, P.D. and Verrall, R.J. (2002). Stochastic Claims Reserving in General Insurance. *British Actuarial Journal*, 8(3), 443-518.

## License

MIT License
