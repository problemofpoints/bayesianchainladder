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
