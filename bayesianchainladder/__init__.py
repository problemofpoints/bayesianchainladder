"""
Bayesian Chain Ladder - Stochastic Loss Reserving Using GLMs.

This package implements Bayesian stochastic loss reserving methods using
generalized linear models, built on top of Bambi and PyMC. It integrates
with the chainladder-python package for input data handling.

The main estimator class is `BayesianChainLadderGLM`, which implements
a cross-classified chain ladder model with MCMC sampling for uncertainty
quantification.

Example
-------
>>> import chainladder as cl
>>> from bayesianchainladder import BayesianChainLadderGLM
>>>
>>> # Load a sample triangle
>>> triangle = cl.load_sample("raa")
>>>
>>> # Fit the Bayesian chain ladder model
>>> model = BayesianChainLadderGLM(
...     formula="incremental ~ 1 + C(origin) + C(dev)",
...     family="negativebinomial",
...     draws=2000,
...     tune=1000,
... )
>>> model.fit(triangle)
>>>
>>> # Get reserve summary
>>> print(model.summary())
>>>
>>> # Access posterior samples
>>> reserve_samples = model.sample_reserves(n_samples=1000)

References
----------
Taylor, G. and McGuire, G. (2016). Stochastic Loss Reserving Using
Generalized Linear Models. CAS Monograph Series Number 3.
"""

from importlib.metadata import PackageNotFoundError, version

# Version
try:
    __version__ = version("bayesianchainladder")
except PackageNotFoundError:
    __version__ = "0.1.0"

# Main estimators
# Base contract
from .base import (
    DEFAULT_QUANTILES,
    BaseStochasticReserve,
    MethodSummary,
    ReserveSamples,
)

# Frequentist estimators
from .bootstrap import (
    BootstrapODPBornhuetterFerguson,
    BootstrapODPCapeCod,
    BootstrapODPChainLadder,
    CorrelatedBootstrapChainLadder,
    CorrelatedBootstrapODPBornhuetterFerguson,
    CorrelatedBootstrapODPCapeCod,
    CorrelatedBootstrapODPSample,
    MackChainLadder,
)

# Data
from .datasets import load_england_sample
from .estimators import BayesianChainLadderGLM, BayesianCSR

# Model building functions
from .models import (
    build_bambi_model,
    build_csr_model,
    build_pymc_model,
    compute_loo,
    compute_prior_predictive_summary,
    compute_waic,
    extract_parameter_summary,
    fit_model,
    posterior_predictive_check,
    predict_posterior,
    sample_prior_predictive,
)

# Plotting functions
from .plots import (
    create_summary_table,
    plot_actual_vs_fitted,
    plot_development_pattern,
    plot_energy,
    plot_forest,
    plot_heatmap_residuals,
    plot_posterior,
    plot_ppc,
    plot_prior_predictive,
    plot_prior_predictive_by_origin,
    plot_prior_predictive_development,
    plot_prior_predictive_reserves,
    plot_prior_predictive_summary,
    plot_prior_predictive_triangle,
    plot_prior_vs_posterior,
    plot_rank,
    plot_reserve_distribution,
    plot_residuals,
    plot_trace,
)

# Risk measures
from .riskmeasures import (
    capital_profile,
    cash_flow_periods,
    cost_of_capital_risk_margin,
    discount_factors,
    discounted_reserves,
    equivalent_risk_tolerance,
    future_reserve_profile,
    proportional_hazards_transform,
    tail_value_at_risk,
    value_at_risk,
)

# Utility functions
from .utils import (
    add_categorical_columns,
    compute_log_exposure_offset,
    create_design_info,
    get_future_dataframe,
    long_to_triangle,
    prepare_csr_data,
    prepare_model_data,
    triangle_to_dataframe,
    validate_triangle,
)

__all__ = [
    # Version
    "__version__",
    # Main estimators
    "BayesianChainLadderGLM",
    "BayesianCSR",
    # Base contract
    "BaseStochasticReserve",
    "MethodSummary",
    "ReserveSamples",
    "DEFAULT_QUANTILES",
    # Frequentist estimators
    "MackChainLadder",
    "BootstrapODPChainLadder",
    "BootstrapODPBornhuetterFerguson",
    "BootstrapODPCapeCod",
    "CorrelatedBootstrapChainLadder",
    "CorrelatedBootstrapODPSample",
    "CorrelatedBootstrapODPBornhuetterFerguson",
    "CorrelatedBootstrapODPCapeCod",
    # Data
    "load_england_sample",
    # Model functions
    "build_bambi_model",
    "build_csr_model",
    "build_pymc_model",
    "fit_model",
    "predict_posterior",
    "posterior_predictive_check",
    "sample_prior_predictive",
    "compute_prior_predictive_summary",
    "compute_waic",
    "compute_loo",
    "extract_parameter_summary",
    # Plotting functions - MCMC diagnostics
    "plot_trace",
    "plot_posterior",
    "plot_ppc",
    "plot_energy",
    "plot_rank",
    "plot_forest",
    # Plotting functions - Reserve analysis
    "plot_reserve_distribution",
    "plot_residuals",
    "plot_actual_vs_fitted",
    "plot_development_pattern",
    "plot_heatmap_residuals",
    "create_summary_table",
    # Plotting functions - Prior predictive checks
    "plot_prior_predictive",
    "plot_prior_predictive_by_origin",
    "plot_prior_predictive_development",
    "plot_prior_predictive_reserves",
    "plot_prior_predictive_triangle",
    "plot_prior_predictive_summary",
    "plot_prior_vs_posterior",
    # Utility functions
    "triangle_to_dataframe",
    "get_future_dataframe",
    "prepare_model_data",
    "prepare_csr_data",
    "add_categorical_columns",
    "compute_log_exposure_offset",
    "create_design_info",
    "validate_triangle",
    "long_to_triangle",
    # Risk measures
    "value_at_risk",
    "tail_value_at_risk",
    "proportional_hazards_transform",
    "cash_flow_periods",
    "discount_factors",
    "discounted_reserves",
    "future_reserve_profile",
    "capital_profile",
    "cost_of_capital_risk_margin",
    "equivalent_risk_tolerance",
]
