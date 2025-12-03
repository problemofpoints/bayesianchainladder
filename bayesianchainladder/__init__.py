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
from .estimators import BayesianChainLadderGLM, BayesianCSR

# Model building functions
from .models import (
    build_bambi_model,
    build_csr_model,
    build_pymc_model,
    compute_loo,
    compute_waic,
    extract_parameter_summary,
    fit_model,
    posterior_predictive_check,
    predict_posterior,
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
    plot_rank,
    plot_reserve_distribution,
    plot_residuals,
    plot_trace,
)

# Utility functions
from .utils import (
    add_categorical_columns,
    compute_log_exposure_offset,
    create_design_info,
    get_future_dataframe,
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
    # Model functions
    "build_bambi_model",
    "build_csr_model",
    "build_pymc_model",
    "fit_model",
    "predict_posterior",
    "posterior_predictive_check",
    "compute_waic",
    "compute_loo",
    "extract_parameter_summary",
    # Plotting functions
    "plot_trace",
    "plot_posterior",
    "plot_ppc",
    "plot_energy",
    "plot_rank",
    "plot_forest",
    "plot_reserve_distribution",
    "plot_residuals",
    "plot_actual_vs_fitted",
    "plot_development_pattern",
    "plot_heatmap_residuals",
    "create_summary_table",
    # Utility functions
    "triangle_to_dataframe",
    "get_future_dataframe",
    "prepare_model_data",
    "prepare_csr_data",
    "add_categorical_columns",
    "compute_log_exposure_offset",
    "create_design_info",
    "validate_triangle",
]
