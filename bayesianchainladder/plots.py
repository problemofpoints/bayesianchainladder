"""
Plotting and visualization utilities for Bayesian chain ladder.

This module provides ArviZ-based plotting helpers and summary table
utilities for diagnostic plots and reserve visualization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .estimators import BayesianChainLadderGLM


def plot_trace(
    model: BayesianChainLadderGLM,
    var_names: list[str] | None = None,
    compact: bool = True,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """
    Create trace plots for model parameters.

    Trace plots show the MCMC sampling history and posterior distributions
    for each parameter, useful for diagnosing convergence.

    Parameters
    ----------
    model : BayesianChainLadderGLM
        A fitted Bayesian chain ladder model.
    var_names : list[str], optional
        Parameter names to plot. If None, plots key parameters.
    compact : bool, optional
        If True, combines chains into a single distribution. Default is True.
    figsize : tuple[float, float], optional
        Figure size as (width, height).
    **kwargs
        Additional arguments passed to az.plot_trace.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes objects.

    Examples
    --------
    >>> from bayesianchainladder import BayesianChainLadderGLM
    >>> from bayesianchainladder.plots import plot_trace
    >>> model = BayesianChainLadderGLM()
    >>> model.fit(triangle)
    >>> fig, ax = plot_trace(model)
    >>> plt.show()
    """
    if model.idata is None:
        raise ValueError("Model must be fitted before plotting")

    axes = az.plot_trace(
        model.idata,
        var_names=var_names,
        compact=compact,
        figsize=figsize,
        **kwargs,
    )

    fig = plt.gcf()
    fig.tight_layout()

    return fig, axes


def plot_posterior(
    model: BayesianChainLadderGLM,
    var_names: list[str] | None = None,
    hdi_prob: float = 0.94,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """
    Create posterior distribution plots for model parameters.

    Parameters
    ----------
    model : BayesianChainLadderGLM
        A fitted Bayesian chain ladder model.
    var_names : list[str], optional
        Parameter names to plot. If None, plots key parameters.
    hdi_prob : float, optional
        Probability mass for HDI interval. Default is 0.94.
    figsize : tuple[float, float], optional
        Figure size.
    **kwargs
        Additional arguments passed to az.plot_posterior.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes objects.
    """
    if model.idata is None:
        raise ValueError("Model must be fitted before plotting")

    axes = az.plot_posterior(
        model.idata,
        var_names=var_names,
        hdi_prob=hdi_prob,
        figsize=figsize,
        **kwargs,
    )

    fig = plt.gcf()
    fig.tight_layout()

    return fig, axes


def plot_ppc(
    model: BayesianChainLadderGLM,
    kind: str = "kde",
    num_pp_samples: int = 100,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """
    Create posterior predictive check plots.

    Posterior predictive checks compare the observed data to data
    simulated from the posterior predictive distribution.

    Parameters
    ----------
    model : BayesianChainLadderGLM
        A fitted Bayesian chain ladder model.
    kind : str, optional
        Type of plot: "kde", "cumulative", or "scatter". Default is "kde".
    num_pp_samples : int, optional
        Number of posterior predictive samples to plot. Default is 100.
    figsize : tuple[float, float], optional
        Figure size.
    **kwargs
        Additional arguments passed to az.plot_ppc.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes objects.
    """
    if model.idata is None:
        raise ValueError("Model must be fitted before plotting")

    # Ensure posterior predictive samples are generated
    if "posterior_predictive" not in model.idata.groups():
        model.model_.predict(model.idata, kind="response", inplace=True)

    ax = az.plot_ppc(
        model.idata,
        kind=kind,
        num_pp_samples=num_pp_samples,
        figsize=figsize,
        **kwargs,
    )

    fig = plt.gcf()

    return fig, ax


def plot_energy(
    model: BayesianChainLadderGLM,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """
    Create energy plot for MCMC diagnostics.

    The energy plot helps diagnose issues with HMC/NUTS sampling.
    Large differences between marginal and transition energies
    suggest problems with exploration.

    Parameters
    ----------
    model : BayesianChainLadderGLM
        A fitted Bayesian chain ladder model.
    figsize : tuple[float, float], optional
        Figure size.
    **kwargs
        Additional arguments passed to az.plot_energy.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes objects.
    """
    if model.idata is None:
        raise ValueError("Model must be fitted before plotting")

    ax = az.plot_energy(model.idata, figsize=figsize, **kwargs)
    fig = plt.gcf()

    return fig, ax


def plot_rank(
    model: BayesianChainLadderGLM,
    var_names: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """
    Create rank plots for MCMC diagnostics.

    Rank plots show the distribution of ranks across chains,
    useful for detecting convergence problems.

    Parameters
    ----------
    model : BayesianChainLadderGLM
        A fitted Bayesian chain ladder model.
    var_names : list[str], optional
        Parameter names to plot.
    figsize : tuple[float, float], optional
        Figure size.
    **kwargs
        Additional arguments passed to az.plot_rank.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes objects.
    """
    if model.idata is None:
        raise ValueError("Model must be fitted before plotting")

    axes = az.plot_rank(
        model.idata, var_names=var_names, figsize=figsize, **kwargs
    )
    fig = plt.gcf()
    fig.tight_layout()

    return fig, axes


def plot_forest(
    model: BayesianChainLadderGLM,
    var_names: list[str] | None = None,
    combined: bool = True,
    hdi_prob: float = 0.94,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """
    Create forest plot for model parameters.

    Forest plots show point estimates and credible intervals
    for multiple parameters, useful for comparing effects.

    Parameters
    ----------
    model : BayesianChainLadderGLM
        A fitted Bayesian chain ladder model.
    var_names : list[str], optional
        Parameter names to plot.
    combined : bool, optional
        If True, combines chains. Default is True.
    hdi_prob : float, optional
        Probability mass for HDI. Default is 0.94.
    figsize : tuple[float, float], optional
        Figure size.
    **kwargs
        Additional arguments passed to az.plot_forest.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes objects.
    """
    if model.idata is None:
        raise ValueError("Model must be fitted before plotting")

    axes = az.plot_forest(
        model.idata,
        var_names=var_names,
        combined=combined,
        hdi_prob=hdi_prob,
        figsize=figsize,
        **kwargs,
    )
    fig = plt.gcf()

    return fig, axes


def plot_reserve_distribution(
    model: BayesianChainLadderGLM,
    by: str = "origin",
    kind: str = "kde",
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """
    Plot the posterior distribution of reserves.

    Parameters
    ----------
    model : BayesianChainLadderGLM
        A fitted Bayesian chain ladder model.
    by : str, optional
        How to display reserves:
        - "origin": Separate distribution for each origin year
        - "total": Total reserve distribution only
        Default is "origin".
    kind : str, optional
        Type of plot: "kde", "hist", or "ecdf". Default is "kde".
    figsize : tuple[float, float], optional
        Figure size.
    **kwargs
        Additional plotting arguments.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes objects.
    """
    if model.reserves_posterior_ is None:
        raise ValueError(
            "Model must be fitted with future cells before plotting reserves"
        )

    reserves = model.reserves_posterior_

    if by == "total":
        # Plot total reserve distribution
        total_reserves = reserves.sum(dim="origin")
        samples = total_reserves.stack(sample=["chain", "draw"]).values

        if figsize is None:
            figsize = (10, 6)

        fig, ax = plt.subplots(figsize=figsize)

        if kind == "kde":
            az.plot_kde(samples, ax=ax, **kwargs)
        elif kind == "hist":
            ax.hist(samples, bins=50, density=True, alpha=0.7, **kwargs)
        elif kind == "ecdf":
            sorted_samples = np.sort(samples)
            ax.step(
                sorted_samples,
                np.arange(1, len(sorted_samples) + 1) / len(sorted_samples),
                **kwargs,
            )
            ax.set_ylabel("ECDF")

        ax.set_xlabel("Total Reserves")
        ax.set_title("Posterior Distribution of Total Reserves")

    else:  # by == "origin"
        origins = reserves.coords["origin"].values
        n_origins = len(origins)

        # Determine grid size
        ncols = min(3, n_origins)
        nrows = (n_origins + ncols - 1) // ncols

        if figsize is None:
            figsize = (4 * ncols, 3 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.atleast_2d(axes)

        for i, origin in enumerate(origins):
            row = i // ncols
            col = i % ncols
            ax = axes[row, col]

            origin_reserves = reserves.sel(origin=origin)
            samples = origin_reserves.stack(sample=["chain", "draw"]).values

            if kind == "kde":
                az.plot_kde(samples, ax=ax)
            elif kind == "hist":
                ax.hist(samples, bins=30, density=True, alpha=0.7)
            elif kind == "ecdf":
                sorted_samples = np.sort(samples)
                ax.step(
                    sorted_samples,
                    np.arange(1, len(sorted_samples) + 1) / len(sorted_samples),
                )

            ax.set_title(f"Origin {origin}")
            ax.set_xlabel("Reserves")

        # Hide empty subplots
        for i in range(n_origins, nrows * ncols):
            row = i // ncols
            col = i % ncols
            axes[row, col].set_visible(False)

        fig.tight_layout()
        ax = axes

    return fig, ax


def plot_residuals(
    model: BayesianChainLadderGLM,
    by: str = "origin",
    kind: str = "scatter",
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """
    Plot residuals vs fitted values or by time period.

    Parameters
    ----------
    model : BayesianChainLadderGLM
        A fitted Bayesian chain ladder model.
    by : str, optional
        X-axis variable: "origin", "dev", "calendar", or "fitted".
        Default is "origin".
    kind : str, optional
        Type of plot: "scatter" or "boxplot". Default is "scatter".
    figsize : tuple[float, float], optional
        Figure size.
    **kwargs
        Additional plotting arguments.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes objects.
    """
    if model.fitted_ is None:
        raise ValueError("Model must be fitted before plotting residuals")

    data = model.fitted_.copy()

    # Compute residuals
    data["residual"] = data["incremental"] - data["fitted_mean"]
    data["std_residual"] = data["residual"] / np.sqrt(data["fitted_mean"].abs() + 1)

    if figsize is None:
        figsize = (10, 6)

    fig, ax = plt.subplots(figsize=figsize)

    if by == "fitted":
        x = data["fitted_mean"]
        xlabel = "Fitted Values"
    elif by in ["origin", "dev", "calendar"]:
        x = data[by]
        xlabel = by.capitalize()
    else:
        raise ValueError(f"Unknown 'by' value: {by}")

    if kind == "scatter":
        ax.scatter(x, data["std_residual"], alpha=0.6, **kwargs)
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax.set_ylabel("Standardized Residual")
    elif kind == "boxplot":
        groups = data.groupby(by)["std_residual"].apply(list)
        ax.boxplot(groups.values, labels=groups.index)
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax.set_ylabel("Standardized Residual")

    ax.set_xlabel(xlabel)
    ax.set_title(f"Residuals by {xlabel}")

    return fig, ax


def plot_actual_vs_fitted(
    model: BayesianChainLadderGLM,
    log_scale: bool = False,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """
    Plot actual vs fitted values.

    Parameters
    ----------
    model : BayesianChainLadderGLM
        A fitted Bayesian chain ladder model.
    log_scale : bool, optional
        If True, use log scale. Default is False.
    figsize : tuple[float, float], optional
        Figure size.
    **kwargs
        Additional plotting arguments.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes objects.
    """
    if model.fitted_ is None:
        raise ValueError("Model must be fitted before plotting")

    data = model.fitted_.copy()

    if figsize is None:
        figsize = (8, 8)

    fig, ax = plt.subplots(figsize=figsize)

    actual = data["incremental"]
    fitted = data["fitted_mean"]

    ax.scatter(fitted, actual, alpha=0.6, **kwargs)

    # Add identity line
    min_val = min(actual.min(), fitted.min())
    max_val = max(actual.max(), fitted.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5, label="y=x")

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Actual Values")
    ax.set_title("Actual vs Fitted Values")
    ax.legend()

    return fig, ax


def plot_development_pattern(
    model: BayesianChainLadderGLM,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """
    Plot the estimated development pattern.

    Parameters
    ----------
    model : BayesianChainLadderGLM
        A fitted Bayesian chain ladder model.
    figsize : tuple[float, float], optional
        Figure size.
    **kwargs
        Additional plotting arguments.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes objects.
    """
    if model.idata is None:
        raise ValueError("Model must be fitted before plotting")

    # Find development effects in posterior
    dev_var = None
    for var in model.idata.posterior.data_vars:
        if "dev" in var.lower():
            dev_var = var
            break

    if dev_var is None:
        raise ValueError("Could not find development effects in model")

    if figsize is None:
        figsize = (10, 6)

    fig, ax = plt.subplots(figsize=figsize)

    # Get posterior samples
    dev_effects = model.idata.posterior[dev_var]

    # Compute mean and credible intervals
    mean = dev_effects.mean(dim=["chain", "draw"])
    hdi = az.hdi(model.idata, var_names=[dev_var], hdi_prob=0.94)

    if dev_var in hdi:
        hdi_data = hdi[dev_var]
    else:
        hdi_data = None

    # Plot
    x = np.arange(len(mean))
    ax.plot(x, mean, "o-", label="Posterior Mean", **kwargs)

    if hdi_data is not None:
        ax.fill_between(
            x,
            hdi_data.sel(hdi="lower"),
            hdi_data.sel(hdi="higher"),
            alpha=0.3,
            label="94% HDI",
        )

    ax.set_xlabel("Development Period")
    ax.set_ylabel("Development Effect (log scale)")
    ax.set_title("Estimated Development Pattern")
    ax.legend()

    return fig, ax


def create_summary_table(
    model: BayesianChainLadderGLM,
    format: str = "dataframe",
    quantiles: list[float] | None = None,
) -> pd.DataFrame | str:
    """
    Create a publication-ready summary table.

    Parameters
    ----------
    model : BayesianChainLadderGLM
        A fitted Bayesian chain ladder model.
    format : str, optional
        Output format: "dataframe", "latex", or "html".
        Default is "dataframe".
    quantiles : list[float], optional
        Quantiles to include. Default is [0.05, 0.25, 0.5, 0.75, 0.95].

    Returns
    -------
    pd.DataFrame or str
        Summary table in requested format.
    """
    if model.ultimate_ is None:
        raise ValueError("Model must be fitted before creating summary table")

    if quantiles is None:
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]

    # Build summary table
    summary = model.summary(include_totals=True)

    if format == "dataframe":
        return summary
    elif format == "latex":
        return summary.to_latex(float_format="%.2f")
    elif format == "html":
        return summary.to_html(float_format="%.2f")
    else:
        raise ValueError(f"Unknown format: {format}")


def plot_heatmap_residuals(
    model: BayesianChainLadderGLM,
    figsize: tuple[float, float] | None = None,
    cmap: str = "RdBu_r",
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """
    Create a heatmap of residuals by origin and development period.

    This follows the diagnostic approach in Taylor & McGuire (2016)
    for visualizing model fit across the triangle.

    Parameters
    ----------
    model : BayesianChainLadderGLM
        A fitted Bayesian chain ladder model.
    figsize : tuple[float, float], optional
        Figure size.
    cmap : str, optional
        Colormap name. Default is "RdBu_r".
    **kwargs
        Additional arguments passed to imshow.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes objects.
    """
    if model.fitted_ is None:
        raise ValueError("Model must be fitted before plotting")

    data = model.fitted_.copy()

    # Compute actual/expected ratio
    data["ae_ratio"] = data["incremental"] / data["fitted_mean"]

    # Pivot to matrix form
    pivot = data.pivot(index="origin", columns="dev", values="ae_ratio")

    if figsize is None:
        figsize = (10, 8)

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(
        pivot.values,
        cmap=cmap,
        aspect="auto",
        vmin=0.5,
        vmax=1.5,
        **kwargs,
    )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Actual / Expected")

    # Set labels
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel("Development Period")
    ax.set_ylabel("Origin Period")
    ax.set_title("Actual/Expected Ratio Heatmap")

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val - 1) > 0.3 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color)

    return fig, ax


# =============================================================================
# Prior Predictive Check Plots
# =============================================================================


def plot_prior_predictive(
    model: BayesianChainLadderGLM,
    prior_idata: "az.InferenceData | None" = None,
    kind: str = "kde",
    num_pp_samples: int = 100,
    show_observed: bool = True,
    log_scale: bool = False,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """
    Plot prior predictive distribution vs observed data.

    This is the primary diagnostic for checking whether priors produce
    realistic predictions before fitting. For loss reserving, this helps
    verify that prior predictive incremental losses are in a plausible range.

    Parameters
    ----------
    model : BayesianChainLadderGLM
        A Bayesian chain ladder model (fitted or with prior_idata_).
    prior_idata : az.InferenceData, optional
        Prior predictive samples. If None, uses model.prior_idata_.
    kind : str, optional
        Type of plot: "kde", "hist", or "ecdf". Default is "kde".
    num_pp_samples : int, optional
        Number of prior predictive samples to plot. Default is 100.
    show_observed : bool, optional
        Whether to show observed data. Default is True.
    log_scale : bool, optional
        Use log scale for x-axis. Default is False.
    figsize : tuple[float, float], optional
        Figure size.
    **kwargs
        Additional plotting arguments.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes objects.

    Examples
    --------
    >>> model = BayesianChainLadderGLM()
    >>> prior_idata = model.sample_prior_predictive(triangle)
    >>> fig, ax = plot_prior_predictive(model, prior_idata)

    Notes
    -----
    Key diagnostics for loss reserving:
    - Prior predictive should cover the range of observed data
    - Extreme tails should represent plausible but unlikely scenarios
    - If prior predictive is much wider than observed, priors may be too vague
    - If prior predictive doesn't cover observed, priors may be too informative
    """
    if prior_idata is None:
        if not hasattr(model, "prior_idata_") or model.prior_idata_ is None:
            raise ValueError(
                "No prior predictive samples. Call sample_prior_predictive() first."
            )
        prior_idata = model.prior_idata_

    if model.data_ is None:
        raise ValueError("Model has no data. Call build_model() or fit() first.")

    # Get response name
    response_name = model.model_.response_component.response.name

    if figsize is None:
        figsize = (10, 6)

    fig, ax = plt.subplots(figsize=figsize)

    # Get prior predictive samples
    pp = prior_idata.prior_predictive[response_name]
    pp_flat = pp.values.flatten()

    # Subsample for plotting if needed
    if len(pp_flat) > num_pp_samples * 1000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pp_flat), size=num_pp_samples * 100, replace=False)
        pp_sample = pp_flat[idx]
    else:
        pp_sample = pp_flat

    # Filter out extreme values for visualization (keep 99.9% of data)
    pp_sample = pp_sample[np.isfinite(pp_sample)]
    if len(pp_sample) > 0:
        lower, upper = np.percentile(pp_sample, [0.05, 99.95])
        pp_sample = pp_sample[(pp_sample >= lower) & (pp_sample <= upper)]

    # Get observed data
    observed = model.data_["incremental"].values

    if kind == "kde":
        # Plot prior predictive KDE
        if len(pp_sample) > 0:
            az.plot_kde(pp_sample, ax=ax, plot_kwargs={"color": "steelblue", "alpha": 0.7},
                       label="Prior Predictive", **kwargs)

        if show_observed:
            az.plot_kde(observed, ax=ax, plot_kwargs={"color": "darkred", "linewidth": 2},
                       label="Observed", **kwargs)

    elif kind == "hist":
        if len(pp_sample) > 0:
            ax.hist(pp_sample, bins=50, density=True, alpha=0.6, color="steelblue",
                   label="Prior Predictive", **kwargs)

        if show_observed:
            ax.hist(observed, bins=30, density=True, alpha=0.8, color="darkred",
                   histtype="step", linewidth=2, label="Observed", **kwargs)

    elif kind == "ecdf":
        if len(pp_sample) > 0:
            sorted_pp = np.sort(pp_sample)
            ax.plot(sorted_pp, np.linspace(0, 1, len(sorted_pp)),
                   color="steelblue", alpha=0.7, label="Prior Predictive")

        if show_observed:
            sorted_obs = np.sort(observed)
            ax.plot(sorted_obs, np.linspace(0, 1, len(sorted_obs)),
                   color="darkred", linewidth=2, label="Observed")
            ax.set_ylabel("ECDF")

    if log_scale and len(pp_sample) > 0 and pp_sample.min() > 0:
        ax.set_xscale("log")

    ax.set_xlabel("Incremental Loss")
    ax.set_title("Prior Predictive Check: Incremental Losses")
    ax.legend()

    return fig, ax


def plot_prior_predictive_by_origin(
    model: BayesianChainLadderGLM,
    prior_idata: "az.InferenceData | None" = None,
    show_observed: bool = True,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """
    Plot prior predictive distributions grouped by origin year.

    This shows how the prior predictive varies across accident years,
    which is important for detecting whether origin effects are reasonable.

    Parameters
    ----------
    model : BayesianChainLadderGLM
        A Bayesian chain ladder model.
    prior_idata : az.InferenceData, optional
        Prior predictive samples.
    show_observed : bool, optional
        Whether to show observed data. Default is True.
    figsize : tuple[float, float], optional
        Figure size.
    **kwargs
        Additional plotting arguments.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes objects.
    """
    if prior_idata is None:
        if not hasattr(model, "prior_idata_") or model.prior_idata_ is None:
            raise ValueError("No prior predictive samples available.")
        prior_idata = model.prior_idata_

    if model.data_ is None:
        raise ValueError("Model has no data.")

    response_name = model.model_.response_component.response.name
    pp = prior_idata.prior_predictive[response_name]

    # Get unique origins
    origins = sorted(model.data_["origin"].unique())
    n_origins = len(origins)

    # Determine grid layout
    ncols = min(3, n_origins)
    nrows = (n_origins + ncols - 1) // ncols

    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for i, origin in enumerate(origins):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]

        # Get indices for this origin
        mask = model.data_["origin"] == origin
        origin_idx = np.where(mask)[0]

        # Get prior predictive for this origin
        # pp has dimensions (chain, draw, obs)
        obs_dim = None
        for dim in pp.dims:
            if "obs" in dim.lower() or dim == "__obs__":
                obs_dim = dim
                break
        if obs_dim is None:
            obs_dim = f"{response_name}_obs"

        pp_origin = pp.isel({obs_dim: origin_idx}).values.flatten()

        # Filter extreme values
        pp_origin = pp_origin[np.isfinite(pp_origin)]
        if len(pp_origin) > 0:
            lower, upper = np.percentile(pp_origin, [1, 99])
            pp_origin_filtered = pp_origin[(pp_origin >= lower) & (pp_origin <= upper)]
            if len(pp_origin_filtered) > 0:
                az.plot_kde(pp_origin_filtered, ax=ax,
                           plot_kwargs={"color": "steelblue", "alpha": 0.7})

        if show_observed:
            observed = model.data_.loc[mask, "incremental"].values
            if len(observed) > 0:
                for val in observed:
                    ax.axvline(val, color="darkred", alpha=0.6, linewidth=1.5)

        ax.set_title(f"Origin {origin}")
        ax.set_xlabel("Incremental Loss")

    # Hide empty subplots
    for i in range(n_origins, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].set_visible(False)

    fig.suptitle("Prior Predictive by Origin Year", fontsize=12, y=1.02)
    fig.tight_layout()

    return fig, axes


def plot_prior_predictive_development(
    model: BayesianChainLadderGLM,
    prior_idata: "az.InferenceData | None" = None,
    show_observed: bool = True,
    hdi_prob: float = 0.94,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """
    Plot prior predictive development pattern across development periods.

    This visualization shows how the prior implies losses develop over time.
    For loss reserving, early development periods should typically have
    higher losses with decreasing amounts as claims mature.

    Parameters
    ----------
    model : BayesianChainLadderGLM
        A Bayesian chain ladder model.
    prior_idata : az.InferenceData, optional
        Prior predictive samples.
    show_observed : bool, optional
        Whether to show observed data. Default is True.
    hdi_prob : float, optional
        HDI probability for credible interval. Default is 0.94.
    figsize : tuple[float, float], optional
        Figure size.
    **kwargs
        Additional plotting arguments.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes objects.

    Notes
    -----
    For loss reserving:
    - Development period 1 should show highest losses (initial reporting)
    - Later periods should show decreasing amounts (claim settlement)
    - The pattern should be monotonically decreasing for most lines
    - Tail development (last periods) may show small positive amounts
    """
    if prior_idata is None:
        if not hasattr(model, "prior_idata_") or model.prior_idata_ is None:
            raise ValueError("No prior predictive samples available.")
        prior_idata = model.prior_idata_

    if model.data_ is None:
        raise ValueError("Model has no data.")

    response_name = model.model_.response_component.response.name
    pp = prior_idata.prior_predictive[response_name]

    # Get development periods
    dev_periods = sorted(model.data_["dev"].unique())

    # Find observation dimension
    obs_dim = None
    for dim in pp.dims:
        if "obs" in dim.lower() or dim == "__obs__":
            obs_dim = dim
            break
    if obs_dim is None:
        obs_dim = f"{response_name}_obs"

    if figsize is None:
        figsize = (10, 6)

    fig, ax = plt.subplots(figsize=figsize)

    # Compute aggregated prior predictive by development period
    pp_means = []
    pp_lower = []
    pp_upper = []
    observed_means = []

    alpha = (1 - hdi_prob) / 2

    for dev in dev_periods:
        mask = model.data_["dev"] == dev
        dev_idx = np.where(mask)[0]

        # Sum prior predictive across observations for this dev period
        pp_dev = pp.isel({obs_dim: dev_idx}).sum(dim=obs_dim)
        pp_dev_flat = pp_dev.values.flatten()

        # Filter infinites
        pp_dev_flat = pp_dev_flat[np.isfinite(pp_dev_flat)]

        if len(pp_dev_flat) > 0:
            pp_means.append(np.mean(pp_dev_flat))
            pp_lower.append(np.percentile(pp_dev_flat, alpha * 100))
            pp_upper.append(np.percentile(pp_dev_flat, (1 - alpha) * 100))
        else:
            pp_means.append(np.nan)
            pp_lower.append(np.nan)
            pp_upper.append(np.nan)

        if show_observed:
            observed_means.append(model.data_.loc[mask, "incremental"].sum())

    x = np.arange(len(dev_periods))

    # Plot prior predictive mean and HDI
    ax.plot(x, pp_means, "o-", color="steelblue", linewidth=2,
            label=f"Prior Predictive Mean", **kwargs)
    ax.fill_between(x, pp_lower, pp_upper, color="steelblue", alpha=0.3,
                   label=f"{int(hdi_prob*100)}% HDI")

    if show_observed:
        ax.plot(x, observed_means, "s--", color="darkred", linewidth=2,
               markersize=8, label="Observed", **kwargs)

    ax.set_xticks(x)
    ax.set_xticklabels(dev_periods)
    ax.set_xlabel("Development Period")
    ax.set_ylabel("Total Incremental Loss")
    ax.set_title("Prior Predictive Development Pattern")
    ax.legend()

    return fig, ax


def plot_prior_predictive_reserves(
    model: BayesianChainLadderGLM,
    prior_idata: "az.InferenceData | None" = None,
    by: str = "total",
    kind: str = "kde",
    reference_reserve: float | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """
    Plot prior predictive distribution of reserves.

    This shows the implied reserve distribution under the prior, before
    seeing any data. For loss reserving, this is critical to verify that
    priors don't imply unrealistic reserve levels.

    Parameters
    ----------
    model : BayesianChainLadderGLM
        A Bayesian chain ladder model.
    prior_idata : az.InferenceData, optional
        Prior predictive samples.
    by : str, optional
        How to display: "total" for total reserves, "origin" for by origin.
        Default is "total".
    kind : str, optional
        Type of plot: "kde", "hist", or "ecdf". Default is "kde".
    reference_reserve : float, optional
        A reference reserve value to show (e.g., from traditional chain ladder).
    figsize : tuple[float, float], optional
        Figure size.
    **kwargs
        Additional plotting arguments.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes objects.

    Notes
    -----
    Key diagnostics:
    - Prior predictive reserves should be centered around a reasonable estimate
    - The spread should reflect prior uncertainty
    - If using reference_reserve, it should fall within the prior predictive range
    - Extremely wide distributions may indicate vague priors
    """
    if prior_idata is None:
        if not hasattr(model, "prior_idata_") or model.prior_idata_ is None:
            raise ValueError("No prior predictive samples available.")
        prior_idata = model.prior_idata_

    if model.data_ is None or model.future_data_ is None:
        raise ValueError("Model has no data.")

    if len(model.future_data_) == 0:
        raise ValueError("No future cells for reserve calculation.")

    response_name = model.model_.response_component.response.name
    pp = prior_idata.prior_predictive[response_name]

    # Get observation indices for future cells
    # For prior predictive, we only have observed cells, so we need to
    # compute implied reserves by aggregating predictions for cells that
    # would be unobserved in the actual triangle

    # Find observation dimension
    obs_dim = None
    for dim in pp.dims:
        if "obs" in dim.lower() or dim == "__obs__":
            obs_dim = dim
            break
    if obs_dim is None:
        obs_dim = f"{response_name}_obs"

    # For each origin, compute the reserve as sum of later development periods
    # that would be unobserved (future cells)
    origins = sorted(model.data_["origin"].unique())
    max_dev = model.data_["dev"].max()

    reserve_samples = {}

    for origin in origins:
        origin_mask = model.data_["origin"] == origin
        origin_devs = model.data_.loc[origin_mask, "dev"].values

        # Future cells for this origin are those with dev > max observed dev for this origin
        max_observed_dev = origin_devs.max()

        if max_observed_dev < max_dev:
            # Get indices of cells that represent "future" development
            # These are cells with same origin but higher dev periods
            # Since prior predictive is only on observed data, we use
            # the relationship to infer reserve levels

            # Use the last observed cell as a proxy for future development
            future_mask = (model.data_["origin"] == origin) & (model.data_["dev"] == max_observed_dev)
            future_idx = np.where(future_mask)[0]

            if len(future_idx) > 0:
                # Scale by remaining development periods
                n_future_periods = max_dev - max_observed_dev
                pp_future = pp.isel({obs_dim: future_idx}).values.flatten()
                # Simple heuristic: scale by remaining periods (improvement possible)
                reserve_samples[origin] = pp_future * n_future_periods

    if by == "total":
        # Combine all origin reserves
        all_reserves = []
        for origin in origins:
            if origin in reserve_samples:
                all_reserves.append(reserve_samples[origin])

        if not all_reserves:
            raise ValueError("No future periods to compute reserves.")

        # Sum across origins for each sample
        n_samples = len(all_reserves[0])
        total_reserves = np.zeros(n_samples)
        for res in all_reserves:
            total_reserves += res[:n_samples]  # Ensure same length

        # Filter extreme values
        total_reserves = total_reserves[np.isfinite(total_reserves)]
        lower, upper = np.percentile(total_reserves, [0.5, 99.5])
        total_reserves = total_reserves[(total_reserves >= lower) & (total_reserves <= upper)]

        if figsize is None:
            figsize = (10, 6)

        fig, ax = plt.subplots(figsize=figsize)

        if kind == "kde" and len(total_reserves) > 0:
            az.plot_kde(total_reserves, ax=ax,
                       plot_kwargs={"color": "steelblue", "linewidth": 2})
        elif kind == "hist":
            ax.hist(total_reserves, bins=50, density=True, alpha=0.7, color="steelblue")
        elif kind == "ecdf":
            sorted_res = np.sort(total_reserves)
            ax.step(sorted_res, np.linspace(0, 1, len(sorted_res)), color="steelblue")
            ax.set_ylabel("ECDF")

        if reference_reserve is not None:
            ax.axvline(reference_reserve, color="darkred", linestyle="--", linewidth=2,
                      label=f"Reference: {reference_reserve:,.0f}")
            ax.legend()

        # Add summary statistics
        if len(total_reserves) > 0:
            mean_res = np.mean(total_reserves)
            median_res = np.median(total_reserves)
            q05 = np.percentile(total_reserves, 5)
            q95 = np.percentile(total_reserves, 95)

            text = f"Mean: {mean_res:,.0f}\nMedian: {median_res:,.0f}\n5%-95%: [{q05:,.0f}, {q95:,.0f}]"
            ax.text(0.98, 0.98, text, transform=ax.transAxes, ha="right", va="top",
                   fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax.set_xlabel("Total Reserves (Prior Predictive)")
        ax.set_title("Prior Predictive Distribution of Total Reserves")

    else:  # by == "origin"
        n_origins_with_reserves = sum(1 for o in origins if o in reserve_samples)

        if n_origins_with_reserves == 0:
            raise ValueError("No future periods for any origin.")

        ncols = min(3, n_origins_with_reserves)
        nrows = (n_origins_with_reserves + ncols - 1) // ncols

        if figsize is None:
            figsize = (4 * ncols, 3 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.atleast_2d(axes)

        plot_idx = 0
        for origin in origins:
            if origin not in reserve_samples:
                continue

            row = plot_idx // ncols
            col = plot_idx % ncols
            ax = axes[row, col]

            origin_res = reserve_samples[origin]
            origin_res = origin_res[np.isfinite(origin_res)]

            if len(origin_res) > 0:
                lower, upper = np.percentile(origin_res, [1, 99])
                origin_res_filtered = origin_res[(origin_res >= lower) & (origin_res <= upper)]

                if kind == "kde" and len(origin_res_filtered) > 0:
                    az.plot_kde(origin_res_filtered, ax=ax,
                               plot_kwargs={"color": "steelblue"})
                elif kind == "hist":
                    ax.hist(origin_res_filtered, bins=30, density=True, alpha=0.7,
                           color="steelblue")

            ax.set_title(f"Origin {origin}")
            ax.set_xlabel("Reserves")
            plot_idx += 1

        # Hide empty subplots
        for i in range(plot_idx, nrows * ncols):
            row = i // ncols
            col = i % ncols
            axes[row, col].set_visible(False)

        fig.suptitle("Prior Predictive Reserves by Origin", fontsize=12, y=1.02)
        fig.tight_layout()
        ax = axes

    return fig, ax


def plot_prior_predictive_triangle(
    model: BayesianChainLadderGLM,
    prior_idata: "az.InferenceData | None" = None,
    statistic: str = "mean",
    show_observed: bool = True,
    figsize: tuple[float, float] | None = None,
    cmap: str = "YlOrRd",
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """
    Plot prior predictive as a triangle heatmap.

    This visualization shows the prior predictive mean (or other statistic)
    for each cell in the triangle format, making it easy to compare with
    the actual observed triangle.

    Parameters
    ----------
    model : BayesianChainLadderGLM
        A Bayesian chain ladder model.
    prior_idata : az.InferenceData, optional
        Prior predictive samples.
    statistic : str, optional
        Statistic to display: "mean", "median", "std", "cv".
        Default is "mean".
    show_observed : bool, optional
        Whether to annotate with observed values. Default is True.
    figsize : tuple[float, float], optional
        Figure size.
    cmap : str, optional
        Colormap for heatmap. Default is "YlOrRd".
    **kwargs
        Additional plotting arguments.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes objects.
    """
    if prior_idata is None:
        if not hasattr(model, "prior_idata_") or model.prior_idata_ is None:
            raise ValueError("No prior predictive samples available.")
        prior_idata = model.prior_idata_

    if model.data_ is None:
        raise ValueError("Model has no data.")

    response_name = model.model_.response_component.response.name
    pp = prior_idata.prior_predictive[response_name]

    # Find observation dimension
    obs_dim = None
    for dim in pp.dims:
        if "obs" in dim.lower() or dim == "__obs__":
            obs_dim = dim
            break
    if obs_dim is None:
        obs_dim = f"{response_name}_obs"

    # Compute summary statistic for each cell
    if statistic == "mean":
        pp_stat = pp.mean(dim=["chain", "draw"])
    elif statistic == "median":
        pp_stat = pp.median(dim=["chain", "draw"])
    elif statistic == "std":
        pp_stat = pp.std(dim=["chain", "draw"])
    elif statistic == "cv":
        pp_mean = pp.mean(dim=["chain", "draw"])
        pp_std = pp.std(dim=["chain", "draw"])
        pp_stat = pp_std / pp_mean
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    # Create result dataframe
    result_df = model.data_.copy()
    result_df["pp_value"] = pp_stat.values

    if show_observed:
        result_df["observed"] = model.data_["incremental"]

    # Pivot to triangle format
    pp_pivot = result_df.pivot(index="origin", columns="dev", values="pp_value")

    if figsize is None:
        figsize = (10, 8)

    if show_observed:
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 1.5, figsize[1]))

        # Prior predictive heatmap
        ax1 = axes[0]
        obs_pivot = result_df.pivot(index="origin", columns="dev", values="observed")

        im1 = ax1.imshow(pp_pivot.values, cmap=cmap, aspect="auto", **kwargs)
        ax1.set_xticks(np.arange(len(pp_pivot.columns)))
        ax1.set_yticks(np.arange(len(pp_pivot.index)))
        ax1.set_xticklabels(pp_pivot.columns)
        ax1.set_yticklabels(pp_pivot.index)
        ax1.set_xlabel("Development Period")
        ax1.set_ylabel("Origin Period")
        ax1.set_title(f"Prior Predictive ({statistic.capitalize()})")

        # Add text annotations
        for i in range(len(pp_pivot.index)):
            for j in range(len(pp_pivot.columns)):
                val = pp_pivot.iloc[i, j]
                if not np.isnan(val):
                    text_color = "white" if val > pp_pivot.values[np.isfinite(pp_pivot.values)].mean() else "black"
                    ax1.text(j, i, f"{val:,.0f}", ha="center", va="center",
                            color=text_color, fontsize=8)

        fig.colorbar(im1, ax=ax1, label=f"Prior Predictive {statistic.capitalize()}")

        # Observed heatmap
        ax2 = axes[1]
        im2 = ax2.imshow(obs_pivot.values, cmap=cmap, aspect="auto", **kwargs)
        ax2.set_xticks(np.arange(len(obs_pivot.columns)))
        ax2.set_yticks(np.arange(len(obs_pivot.index)))
        ax2.set_xticklabels(obs_pivot.columns)
        ax2.set_yticklabels(obs_pivot.index)
        ax2.set_xlabel("Development Period")
        ax2.set_ylabel("Origin Period")
        ax2.set_title("Observed")

        # Add text annotations
        for i in range(len(obs_pivot.index)):
            for j in range(len(obs_pivot.columns)):
                val = obs_pivot.iloc[i, j]
                if not np.isnan(val):
                    text_color = "white" if val > obs_pivot.values[np.isfinite(obs_pivot.values)].mean() else "black"
                    ax2.text(j, i, f"{val:,.0f}", ha="center", va="center",
                            color=text_color, fontsize=8)

        fig.colorbar(im2, ax=ax2, label="Observed")
        ax = axes

    else:
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(pp_pivot.values, cmap=cmap, aspect="auto", **kwargs)
        ax.set_xticks(np.arange(len(pp_pivot.columns)))
        ax.set_yticks(np.arange(len(pp_pivot.index)))
        ax.set_xticklabels(pp_pivot.columns)
        ax.set_yticklabels(pp_pivot.index)
        ax.set_xlabel("Development Period")
        ax.set_ylabel("Origin Period")
        ax.set_title(f"Prior Predictive Triangle ({statistic.capitalize()})")

        # Add text annotations
        for i in range(len(pp_pivot.index)):
            for j in range(len(pp_pivot.columns)):
                val = pp_pivot.iloc[i, j]
                if not np.isnan(val):
                    text_color = "white" if val > pp_pivot.values[np.isfinite(pp_pivot.values)].mean() else "black"
                    ax.text(j, i, f"{val:,.0f}", ha="center", va="center",
                            color=text_color, fontsize=8)

        fig.colorbar(im, ax=ax, label=f"Prior Predictive {statistic.capitalize()}")

    fig.tight_layout()

    return fig, ax


def plot_prior_vs_posterior(
    model: BayesianChainLadderGLM,
    prior_idata: "az.InferenceData | None" = None,
    var_names: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """
    Compare prior and posterior distributions of parameters.

    This visualization helps understand how much the data updated the priors.
    Large differences indicate the data was informative; small differences
    may indicate prior-data conflict or non-identifiability.

    Parameters
    ----------
    model : BayesianChainLadderGLM
        A fitted Bayesian chain ladder model.
    prior_idata : az.InferenceData, optional
        Prior predictive samples. If None, uses model.prior_idata_.
    var_names : list[str], optional
        Parameter names to plot. If None, plots common parameters.
    figsize : tuple[float, float], optional
        Figure size.
    **kwargs
        Additional plotting arguments.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes objects.

    Notes
    -----
    Key diagnostics:
    - Posterior narrower than prior: data is informative
    - Posterior shifted from prior: data conflicts with prior expectation
    - Posterior similar to prior: data provides little information
    """
    if model.idata is None:
        raise ValueError("Model must be fitted to compare prior and posterior.")

    if prior_idata is None:
        if not hasattr(model, "prior_idata_") or model.prior_idata_ is None:
            raise ValueError("No prior samples. Call sample_prior_predictive() first.")
        prior_idata = model.prior_idata_

    if "prior" not in prior_idata.groups():
        raise ValueError("prior_idata must contain 'prior' group.")

    # Determine variables to plot
    if var_names is None:
        # Find common scalar parameters
        posterior_vars = set(model.idata.posterior.data_vars)
        prior_vars = set(prior_idata.prior.data_vars)
        common_vars = posterior_vars & prior_vars

        # Filter to scalar or 1D parameters
        var_names = []
        for var in common_vars:
            post_var = model.idata.posterior[var]
            if len(post_var.dims) <= 3:  # chain, draw, + at most 1 other dim
                var_names.append(var)

        # Limit to first 6 for readability
        var_names = sorted(var_names)[:6]

    if not var_names:
        raise ValueError("No common parameters found between prior and posterior.")

    n_vars = len(var_names)
    ncols = min(3, n_vars)
    nrows = (n_vars + ncols - 1) // ncols

    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_vars == 1:
        axes = np.array([[axes]])
    else:
        axes = np.atleast_2d(axes)

    for i, var in enumerate(var_names):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]

        # Get prior and posterior samples
        prior_samples = prior_idata.prior[var].values.flatten()
        post_samples = model.idata.posterior[var].values.flatten()

        # Filter extreme values for visualization
        for samples, label, color in [
            (prior_samples, "Prior", "steelblue"),
            (post_samples, "Posterior", "darkred"),
        ]:
            samples_clean = samples[np.isfinite(samples)]
            if len(samples_clean) > 0:
                lower, upper = np.percentile(samples_clean, [1, 99])
                samples_filtered = samples_clean[(samples_clean >= lower) & (samples_clean <= upper)]
                if len(samples_filtered) > 0:
                    az.plot_kde(samples_filtered, ax=ax,
                               plot_kwargs={"color": color, "alpha": 0.7, "linewidth": 2},
                               label=label)

        ax.set_title(var)
        ax.legend(fontsize=8)

    # Hide empty subplots
    for i in range(n_vars, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].set_visible(False)

    fig.suptitle("Prior vs Posterior Distributions", fontsize=12, y=1.02)
    fig.tight_layout()

    return fig, axes


def plot_prior_predictive_summary(
    model: BayesianChainLadderGLM,
    prior_idata: "az.InferenceData | None" = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """
    Create a comprehensive prior predictive summary plot.

    This is a multi-panel figure showing key prior predictive diagnostics
    relevant to loss reserving in a single view.

    Parameters
    ----------
    model : BayesianChainLadderGLM
        A Bayesian chain ladder model.
    prior_idata : az.InferenceData, optional
        Prior predictive samples.
    figsize : tuple[float, float], optional
        Figure size.
    **kwargs
        Additional plotting arguments.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib Figure and Axes objects.
    """
    if prior_idata is None:
        if not hasattr(model, "prior_idata_") or model.prior_idata_ is None:
            raise ValueError("No prior predictive samples available.")
        prior_idata = model.prior_idata_

    if model.data_ is None:
        raise ValueError("Model has no data.")

    if figsize is None:
        figsize = (14, 10)

    fig = plt.figure(figsize=figsize)

    # Create grid of subplots
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel 1: Overall distribution
    ax1 = fig.add_subplot(gs[0, 0])
    response_name = model.model_.response_component.response.name
    pp = prior_idata.prior_predictive[response_name]
    pp_flat = pp.values.flatten()
    pp_flat = pp_flat[np.isfinite(pp_flat)]

    if len(pp_flat) > 0:
        lower, upper = np.percentile(pp_flat, [0.5, 99.5])
        pp_filtered = pp_flat[(pp_flat >= lower) & (pp_flat <= upper)]
        if len(pp_filtered) > 0:
            az.plot_kde(pp_filtered, ax=ax1,
                       plot_kwargs={"color": "steelblue", "alpha": 0.7})

    observed = model.data_["incremental"].values
    az.plot_kde(observed, ax=ax1,
               plot_kwargs={"color": "darkred", "linewidth": 2})

    ax1.set_xlabel("Incremental Loss")
    ax1.set_title("Prior Predictive vs Observed")
    ax1.legend(["Prior Pred.", "Observed"], loc="upper right")

    # Panel 2: Development pattern
    ax2 = fig.add_subplot(gs[0, 1])
    dev_periods = sorted(model.data_["dev"].unique())

    obs_dim = None
    for dim in pp.dims:
        if "obs" in dim.lower() or dim == "__obs__":
            obs_dim = dim
            break
    if obs_dim is None:
        obs_dim = f"{response_name}_obs"

    pp_means = []
    observed_means = []

    for dev in dev_periods:
        mask = model.data_["dev"] == dev
        dev_idx = np.where(mask)[0]
        pp_dev = pp.isel({obs_dim: dev_idx}).sum(dim=obs_dim)
        pp_dev_flat = pp_dev.values.flatten()
        pp_dev_flat = pp_dev_flat[np.isfinite(pp_dev_flat)]
        pp_means.append(np.mean(pp_dev_flat) if len(pp_dev_flat) > 0 else np.nan)
        observed_means.append(model.data_.loc[mask, "incremental"].sum())

    x = np.arange(len(dev_periods))
    ax2.bar(x - 0.2, pp_means, 0.4, color="steelblue", alpha=0.7, label="Prior Pred.")
    ax2.bar(x + 0.2, observed_means, 0.4, color="darkred", alpha=0.7, label="Observed")
    ax2.set_xticks(x)
    ax2.set_xticklabels(dev_periods)
    ax2.set_xlabel("Development Period")
    ax2.set_ylabel("Total Loss")
    ax2.set_title("Development Pattern")
    ax2.legend()

    # Panel 3: By origin (boxplots)
    ax3 = fig.add_subplot(gs[1, 0])
    origins = sorted(model.data_["origin"].unique())

    pp_by_origin = []
    obs_by_origin = []

    for origin in origins:
        mask = model.data_["origin"] == origin
        origin_idx = np.where(mask)[0]
        pp_origin = pp.isel({obs_dim: origin_idx}).values.flatten()
        pp_origin = pp_origin[np.isfinite(pp_origin)]
        # Sample for efficiency
        if len(pp_origin) > 1000:
            pp_origin = np.random.choice(pp_origin, 1000, replace=False)
        pp_by_origin.append(pp_origin)
        obs_by_origin.append(model.data_.loc[mask, "incremental"].values)

    # Create boxplots
    positions = np.arange(len(origins))
    bp1 = ax3.boxplot(pp_by_origin, positions=positions - 0.2, widths=0.35,
                     patch_artist=True, boxprops=dict(facecolor="steelblue", alpha=0.6))
    bp2 = ax3.boxplot(obs_by_origin, positions=positions + 0.2, widths=0.35,
                     patch_artist=True, boxprops=dict(facecolor="darkred", alpha=0.6))

    ax3.set_xticks(positions)
    ax3.set_xticklabels([str(o)[-2:] if len(str(o)) > 4 else str(o) for o in origins], fontsize=8)
    ax3.set_xlabel("Origin Year")
    ax3.set_ylabel("Incremental Loss")
    ax3.set_title("Distribution by Origin")
    ax3.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Prior Pred.", "Observed"])

    # Panel 4: Summary statistics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    # Compute summary stats
    pp_all = pp.values.flatten()
    pp_all = pp_all[np.isfinite(pp_all)]

    if len(pp_all) > 0:
        summary_text = "Prior Predictive Summary\n" + "=" * 30 + "\n\n"
        summary_text += f"Sample Size: {len(pp_all):,}\n\n"
        summary_text += f"Mean: {np.mean(pp_all):,.0f}\n"
        summary_text += f"Std Dev: {np.std(pp_all):,.0f}\n"
        summary_text += f"Median: {np.median(pp_all):,.0f}\n\n"
        summary_text += "Percentiles:\n"
        for q in [5, 25, 50, 75, 95]:
            summary_text += f"  {q}%: {np.percentile(pp_all, q):,.0f}\n"

        summary_text += "\n" + "-" * 30 + "\n"
        summary_text += "Observed Data Summary\n" + "-" * 30 + "\n\n"
        summary_text += f"Sample Size: {len(observed):,}\n\n"
        summary_text += f"Mean: {np.mean(observed):,.0f}\n"
        summary_text += f"Std Dev: {np.std(observed):,.0f}\n"
        summary_text += f"Median: {np.median(observed):,.0f}\n"

        ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
                fontsize=10, family="monospace", verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("Prior Predictive Check Summary", fontsize=14, y=0.98)

    return fig, [ax1, ax2, ax3, ax4]
