"""
Risk margin calculation for loss reserve triangles.

This module implements risk margin calculations following the methodology
described in Glenn Meyers' "Stochastic Loss Reserving Using Bayesian MCMC
Models" (CAS Monograph Series, 2015), Section 11.

Risk margins are calculated by:
1. Projecting capital requirements over the run-off period
2. Computing the cost of holding required capital at a "risky" rate
3. The difference between invested capital and present value of returns
   at the risky rate represents the risk margin

Two time horizons are supported:
- Ultimate: Capital is determined by TVaR of ultimate loss estimates
- One-year: Capital is determined by TVaR of one-year-ahead estimates

References
----------
Meyers, G. (2015). Stochastic Loss Reserving Using Bayesian MCMC Models.
CAS Monograph Series Number 1, Section 11.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, Literal
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

if TYPE_CHECKING:
    from .estimators import BayesianChainLadderGLM, BayesianCSR


class LossReserveModel(Protocol):
    """Protocol defining the interface for loss reserve models."""

    idata: Any
    data_: pd.DataFrame
    future_data_: pd.DataFrame
    reserves_posterior_: xr.DataArray | None

    def _check_is_fitted(self) -> None:
        ...


@dataclass
class RiskMarginResult:
    """
    Container for risk margin calculation results.

    Attributes
    ----------
    num_samples : int
        Number of MCMC samples used in the calculation.
    expected_ultimate : float
        Mean of ultimate loss estimates across all scenarios.
    best_estimate : float
        Present value of expected future payouts (best estimate liability).
    pred_mean : np.ndarray
        Posterior mean of ultimate loss by calendar year, shape (num_samples, n_periods).
    pred_assets : np.ndarray
        TVaR (required assets) by calendar year, shape (num_samples, n_periods).
    pred_capital : np.ndarray
        Required capital (TVaR - Mean) by calendar year, shape (num_samples, n_periods).
    capital_release : np.ndarray
        Capital released each period, shape (num_samples, n_periods).
    risk_margin : np.ndarray
        Risk margin for each scenario, shape (num_samples,).
    risk_margin_pct : np.ndarray
        Risk margin as percentage of initial capital, shape (num_samples,).
    fixed_rate : float
        Risk-free discount rate used.
    risky_rate : float
        Cost of capital rate used.
    tvar_percentile : float
        TVaR percentile used (e.g., 0.97 for 97% TVaR).
    time_horizon : str
        Time horizon used: "ultimate" or "one_year".
    """

    num_samples: int
    expected_ultimate: float
    best_estimate: float
    pred_mean: np.ndarray
    pred_assets: np.ndarray
    pred_capital: np.ndarray
    capital_release: np.ndarray
    risk_margin: np.ndarray
    risk_margin_pct: np.ndarray
    fixed_rate: float
    risky_rate: float
    tvar_percentile: float
    time_horizon: str
    ultimate_samples: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> pd.DataFrame:
        """
        Return summary statistics of risk margin calculations.

        Returns
        -------
        pd.DataFrame
            Summary statistics including mean, std, and quantiles.
        """
        return pd.DataFrame(
            {
                "Expected Ultimate": [self.expected_ultimate],
                "Best Estimate": [self.best_estimate],
                "Initial Capital (Mean)": [np.mean(self.pred_capital[:, 0])],
                "Risk Margin (Mean)": [np.mean(self.risk_margin)],
                "Risk Margin (Std)": [np.std(self.risk_margin)],
                "Risk Margin (5%)": [np.percentile(self.risk_margin, 5)],
                "Risk Margin (Median)": [np.median(self.risk_margin)],
                "Risk Margin (95%)": [np.percentile(self.risk_margin, 95)],
                "Risk Margin % (Mean)": [np.mean(self.risk_margin_pct)],
                "Risk Margin % (Std)": [np.std(self.risk_margin_pct)],
            },
            index=["Value"],
        ).T


class RiskMarginCalculator:
    """
    Calculator for risk margins using Bayesian loss reserve models.

    This class computes risk margins by projecting capital requirements
    over the run-off period of a loss reserve triangle. It supports both
    ultimate and one-year time horizons.

    The calculation follows Meyers (2015) methodology:
    1. Generate posterior predictive samples of future loss developments
    2. For each scenario, compute capital requirements using TVaR
    3. Track capital releases as reserves run off
    4. Compute risk margin as initial capital minus PV of releases at risky rate

    Parameters
    ----------
    model : BayesianChainLadderGLM or BayesianCSR
        A fitted Bayesian loss reserve model with posterior samples.
    fixed_rate : float, optional
        Risk-free discount rate for present value calculations.
        Default is 0.04 (4%).
    risky_rate : float, optional
        Cost of capital rate (hurdle rate for holding capital).
        Default is 0.10 (10%).
    tvar_percentile : float, optional
        Percentile for TVaR calculation (e.g., 0.97 means 97% TVaR,
        using the worst 3% of scenarios). Default is 0.97.
    random_seed : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    model : LossReserveModel
        The fitted loss reserve model.
    fixed_rate : float
        Risk-free discount rate.
    risky_rate : float
        Cost of capital rate.
    tvar_percentile : float
        TVaR percentile.
    random_seed : int or None
        Random seed.

    Examples
    --------
    >>> import chainladder as cl
    >>> from bayesianchainladder import BayesianCSR
    >>> from bayesianchainladder.risk_margin import RiskMarginCalculator
    >>>
    >>> # Load data and fit model
    >>> triangle = cl.load_sample("GenIns")
    >>> model = BayesianCSR(draws=2000, tune=1000)
    >>> model.fit(triangle, premium_value=10000)
    >>>
    >>> # Calculate risk margin at ultimate time horizon
    >>> calc = RiskMarginCalculator(model, fixed_rate=0.04, risky_rate=0.10)
    >>> result = calc.calculate_ultimate_horizon()
    >>>
    >>> # View summary
    >>> print(result.summary())

    References
    ----------
    Meyers, G. (2015). Stochastic Loss Reserving Using Bayesian MCMC Models.
    CAS Monograph Series Number 1, Section 11.
    """

    def __init__(
        self,
        model: "BayesianChainLadderGLM | BayesianCSR",
        fixed_rate: float = 0.04,
        risky_rate: float = 0.10,
        tvar_percentile: float = 0.97,
        random_seed: int | None = None,
    ):
        self.model = model
        self.fixed_rate = fixed_rate
        self.risky_rate = risky_rate
        self.tvar_percentile = tvar_percentile
        self.random_seed = random_seed

        # Validate model is fitted
        model._check_is_fitted()

        # Extract model dimensions
        self._setup_model_info()

    def _setup_model_info(self) -> None:
        """Extract key information from the fitted model."""
        data = self.model.data_
        self.origins = sorted(data["origin"].unique())
        self.n_origins = len(self.origins)
        self.dev_periods = sorted(data["dev"].unique())
        self.n_dev = len(self.dev_periods)

        # Get number of MCMC samples
        posterior = self.model.idata.posterior
        n_chains = posterior.dims["chain"]
        n_draws = posterior.dims["draw"]
        self.num_samples = n_chains * n_draws

        # Get premium information if available (for CSR model)
        self.logprem = None
        self.premium = None
        if "logprem" in data.columns:
            self.logprem = data.groupby("origin")["logprem"].first().values
            self.premium = np.exp(self.logprem)

    def _extract_csr_parameters(self) -> dict[str, np.ndarray]:
        """Extract and flatten CSR model parameters."""
        posterior = self.model.idata.posterior

        # Extract all parameters and flatten (chain, draw) -> (sample,)
        alpha = posterior["alpha"].values.reshape(self.num_samples, -1)
        beta = posterior["beta"].values.reshape(self.num_samples, -1)
        speedup = posterior["speedup"].values.reshape(self.num_samples, -1)
        logelr = posterior["logelr"].values.flatten()
        sig = posterior["sig"].values.reshape(self.num_samples, -1)

        # Handle gamma parameter for speedup
        if "gamma" in posterior:
            gamma = posterior["gamma"].values.flatten()
        else:
            gamma = np.zeros(self.num_samples)

        return {
            "alpha": alpha,
            "beta": beta,
            "speedup": speedup,
            "logelr": logelr,
            "sig": sig,
            "gamma": gamma,
        }

    def _get_paid_triangle(self) -> np.ndarray:
        """
        Get paid loss triangle from observed data.

        Returns
        -------
        np.ndarray
            Cumulative paid losses, shape (n_origins, n_dev).
        """
        data = self.model.data_

        # Check if we have cumulative data
        if "cumulative" in data.columns:
            value_col = "cumulative"
        elif "logloss" in data.columns:
            # CSR model uses log cumulative
            data = data.copy()
            data["cumulative"] = np.exp(data["logloss"])
            value_col = "cumulative"
        else:
            # Compute cumulative from incremental
            data = data.copy()
            data = data.sort_values(["origin", "dev"])
            data["cumulative"] = data.groupby("origin")["incremental"].cumsum()
            value_col = "cumulative"

        # Pivot to triangle format
        pivot = data.pivot(index="origin", columns="dev", values=value_col)
        return pivot.values

    def _compute_best_estimate(
        self, params: dict[str, np.ndarray], trpaid: np.ndarray
    ) -> np.ndarray:
        """
        Compute best estimate (present value of expected future payouts).

        For each MCMC sample, projects the expected cumulative paid loss
        for each origin to ultimate, then computes present value.

        Parameters
        ----------
        params : dict
            Model parameters from _extract_csr_parameters().
        trpaid : np.ndarray
            Observed paid triangle, shape (n_origins, n_dev).

        Returns
        -------
        np.ndarray
            Best estimate for each sample, shape (num_samples,).
        """
        alpha = params["alpha"]
        beta = params["beta"]
        speedup = params["speedup"]
        logelr = params["logelr"]
        sig = params["sig"]

        best_estimates = np.zeros(self.num_samples)

        for i in range(self.num_samples):
            # Create a copy of observed triangle for this sample
            tr = trpaid.copy()

            # Fill in future cells with expected values
            # For each future calendar year
            pv = 0.0
            for cy in range(1, self.n_dev):  # cy = 1 to n_dev-1
                for w in range(cy, self.n_origins):  # origin year (0-indexed)
                    d = self.n_dev - 1 + cy - w  # development period (0-indexed)

                    if d < self.n_dev and w < self.n_origins:
                        # Expected cumulative paid at (w, d)
                        # mu = logprem + logelr + alpha[w] + beta[d] * speedup[w]
                        mu = (
                            self.logprem[w]
                            + logelr[i]
                            + alpha[i, w]
                            + beta[i, d] * speedup[i, w]
                        )
                        # E[exp(X)] = exp(mu + sig^2/2) for lognormal
                        expected = np.exp(mu + sig[i, d] ** 2 / 2)
                        tr[w, d] = expected

            # Compute present value of incremental payouts
            for cy in range(1, self.n_dev):
                for w in range(cy, self.n_origins):
                    d = self.n_dev - 1 + cy - w
                    if d > 0 and d < self.n_dev:
                        incremental = tr[w, d] - tr[w, d - 1]
                        # Discount to present (cy - 0.5 to account for mid-year)
                        discount = (1 + self.fixed_rate) ** (cy - 0.5)
                        pv += incremental / discount

            best_estimates[i] = pv

        return best_estimates

    def _simulate_future_losses(
        self, params: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate log cumulative losses for future (lower triangle) cells.

        Parameters
        ----------
        params : dict
            Model parameters.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (logloss_p, mu_p) - Simulated log losses and mu values,
            each of shape (num_samples, n_origins, n_dev).
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        alpha = params["alpha"]
        beta = params["beta"]
        logelr = params["logelr"]
        sig = params["sig"]

        # Initialize arrays for mu and simulated losses
        mu_p = np.zeros((self.num_samples, self.n_origins, self.n_dev))
        logloss_p = np.zeros((self.num_samples, self.n_origins, self.n_dev))

        # Generate simulated losses for future cells (lower triangle)
        # R code: for (d in 2:10) for (w in (12-d):10)
        # In 0-indexed: d = 1 to n_dev-1, w = n_dev-d to n_dev-1
        for d in range(1, self.n_dev):  # dev period 1 to n_dev-1 (2 to 10 in R)
            for w in range(self.n_dev - d, self.n_origins):  # origin years
                # mu = logprem + logelr + alpha[w] + beta[d]
                mu_p[:, w, d] = self.logprem[w] + logelr + alpha[:, w] + beta[:, d]
                logloss_p[:, w, d] = np.random.normal(mu_p[:, w, d], sig[:, d])

        return logloss_p, mu_p

    def _compute_ultimate_by_origin(
        self, params: dict[str, np.ndarray], trpaid: np.ndarray
    ) -> np.ndarray:
        """
        Compute unconditional ultimate loss estimates by origin.

        Parameters
        ----------
        params : dict
            Model parameters.
        trpaid : np.ndarray
            Observed paid triangle.

        Returns
        -------
        np.ndarray
            Ultimate estimates, shape (num_samples, n_origins).
        """
        alpha = params["alpha"]
        logelr = params["logelr"]
        sig = params["sig"]

        # Ultimate development index (last column)
        d_ult = self.n_dev - 1

        mean_ult = np.zeros((self.num_samples, self.n_origins))

        # First origin is fully developed
        mean_ult[:, 0] = trpaid[0, d_ult]

        # Other origins need projection
        for w in range(1, self.n_origins):
            # E[loss] = premium * ELR * exp(alpha) * exp(sig^2/2)
            mean_ult[:, w] = self.premium[w] * np.exp(
                logelr + alpha[:, w] + sig[:, d_ult] ** 2 / 2
            )

        return mean_ult

    def _compute_tvar(
        self, samples: np.ndarray, percentile: float | None = None
    ) -> float:
        """
        Compute Tail Value at Risk (TVaR) / Conditional VaR.

        TVaR is the expected value in the tail beyond the VaR percentile.

        Parameters
        ----------
        samples : np.ndarray
            Loss samples.
        percentile : float, optional
            Percentile for VaR (e.g., 0.97). Default uses self.tvar_percentile.

        Returns
        -------
        float
            TVaR value.
        """
        if percentile is None:
            percentile = self.tvar_percentile

        n = len(samples)
        sorted_samples = np.sort(samples)

        # TVaR is mean of worst (1-percentile) samples
        tail_start = int(percentile * n)
        tail_samples = sorted_samples[tail_start:]

        return np.mean(tail_samples)

    def _compute_posterior_assets(
        self,
        posterior_weights: np.ndarray,
        ultimate_samples: np.ndarray,
        n_resample: int = 10000,
    ) -> tuple[float, float]:
        """
        Compute posterior mean and TVaR given posterior weights.

        Parameters
        ----------
        posterior_weights : np.ndarray
            Weights for resampling (posterior probabilities).
        ultimate_samples : np.ndarray
            Ultimate loss samples.
        n_resample : int
            Number of resamples to draw.

        Returns
        -------
        tuple[float, float]
            (posterior_mean, posterior_tvar)
        """
        # Resample according to posterior weights
        indices = np.random.choice(
            len(ultimate_samples),
            size=n_resample,
            replace=True,
            p=posterior_weights,
        )
        resampled = ultimate_samples[indices]

        mean = np.mean(resampled)
        tvar = self._compute_tvar(resampled)

        return mean, tvar

    def _log_likelihood(
        self,
        simulated_losses: np.ndarray,
        mu_p: np.ndarray,
        sig: np.ndarray,
        calendar_year: int,
    ) -> np.ndarray:
        """
        Compute log likelihood of simulated losses for a calendar year.

        Parameters
        ----------
        simulated_losses : np.ndarray
            Simulated log losses for one scenario, shape (n_origins, n_dev).
        mu_p : np.ndarray
            Expected log losses, shape (num_samples, n_origins, n_dev).
        sig : np.ndarray
            Standard deviations by dev period, shape (num_samples, n_dev).
        calendar_year : int
            Calendar year (1-indexed) being observed.

        Returns
        -------
        np.ndarray
            Log likelihoods, shape (num_samples,).
        """
        ll = np.zeros(self.num_samples)

        for w in range(calendar_year, self.n_origins):
            d = self.n_dev - 1 + calendar_year - w  # Development period
            if d < self.n_dev:
                x = simulated_losses[w, d]
                mu = mu_p[:, w, d]
                sigma = sig[:, d]
                ll += stats.norm.logpdf(x, mu, sigma)

        return ll

    def calculate_ultimate_horizon(
        self,
        n_parallel_scenarios: int | None = None,
        progress_callback: Any | None = None,
    ) -> RiskMarginResult:
        """
        Calculate risk margin at ultimate time horizon.

        At the ultimate time horizon, capital requirements are based on
        the TVaR of the ultimate loss estimates. As calendar years progress
        and new data becomes available, the posterior distribution of
        ultimate losses is updated using Bayesian methods.

        Parameters
        ----------
        n_parallel_scenarios : int, optional
            Number of scenarios to process in parallel. If None, processes
            all scenarios sequentially.
        progress_callback : callable, optional
            Function called with (current, total) to report progress.

        Returns
        -------
        RiskMarginResult
            Container with all risk margin calculation results.

        Notes
        -----
        This method implements the following algorithm:

        1. Generate MCMC samples of model parameters
        2. For each sample, simulate future loss developments
        3. Starting from calendar year 0, progressively observe simulated data
        4. Update posterior distribution using likelihood of observed data
        5. Compute capital requirements (TVaR - Mean) at each calendar year
        6. Calculate capital releases: C_t * (1+r_f) - C_{t+1}
        7. Compute risk margin: C_0 - PV(releases at risky rate)
        """
        # Get model parameters
        params = self._extract_csr_parameters()
        trpaid = self._get_paid_triangle()

        # Compute best estimate
        best_estimates = self._compute_best_estimate(params, trpaid)
        best_estimate = np.mean(best_estimates)

        # Compute unconditional ultimate loss by origin
        mean_ult = self._compute_ultimate_by_origin(params, trpaid)
        ultall = mean_ult.sum(axis=1)  # Total ultimate by sample
        expected_ultimate = np.mean(ultall)

        # Simulate future losses and get mu_p for likelihood calculation
        logloss_p, mu_p = self._simulate_future_losses(params)

        sig = params["sig"]

        # Initialize output arrays
        n_periods = self.n_dev
        pred_mean = np.zeros((self.num_samples, n_periods))
        pred_assets = np.zeros((self.num_samples, n_periods))

        if self.random_seed is not None:
            np.random.seed(self.random_seed + 1)

        # Process each scenario - following R code structure
        # R: for (i in 1:num.mcmc) { x_p = logloss_p[i,,]; ... }
        for i in range(self.num_samples):
            if progress_callback is not None:
                progress_callback(i, self.num_samples)

            # Get simulated losses for this scenario
            x_p = logloss_p[i]  # Shape (n_origins, n_dev)

            # Initial (unconditional) estimates
            # R: p.mean[1]=mean(ultall); p.assets[1]=mean(sort(ultall)[TVaR.Range])
            pred_mean[i, 0] = np.mean(ultall)
            pred_assets[i, 0] = self._compute_tvar(ultall)

            # Accumulate log likelihood as we observe more calendar years
            # R: loglike=rep(0,num.mcmc)
            loglike = np.zeros(self.num_samples)

            # R: for each calendar year, accumulate likelihood and update posterior
            for cy in range(1, n_periods):
                # Add likelihood of observing calendar year cy data
                # R: loglike=loglike+llike(x_p,cy,num.mcmc)
                # R llike: for (w in (1+cy):10) ll=ll+dnorm(x_p[w,11+cy-w],...)
                # In 0-indexed: w from cy to n_origins-1, d = n_dev-1+cy-w
                for w in range(cy, self.n_origins):
                    d = self.n_dev - 1 + cy - w
                    # Access x_p[w, d] and compare to mu_p[:, w, d]
                    x = x_p[w, d]
                    mu = mu_p[:, w, d]
                    sigma = sig[:, d]
                    loglike += stats.norm.logpdf(x, mu, sigma)

                # Compute posterior weights (stabilized)
                # R: loglike2=loglike-max(loglike); postint=sum(exp(loglike2));
                #    posterior=exp(loglike2)/postint
                loglike_shifted = loglike - np.max(loglike)
                weights = np.exp(loglike_shifted)
                weights = weights / np.sum(weights)

                # Compute posterior mean and TVaR
                # R: call.pa=post_assets(posterior,ultall)
                mean, tvar = self._compute_posterior_assets(weights, ultall)
                pred_mean[i, cy] = mean
                pred_assets[i, cy] = tvar

        # Compute required capital
        pred_capital = pred_assets - pred_mean

        # Compute capital release
        capital_release = np.zeros((self.num_samples, n_periods))
        for cy in range(n_periods - 1):
            capital_release[:, cy] = (
                pred_capital[:, cy] * (1 + self.fixed_rate) - pred_capital[:, cy + 1]
            )
        capital_release[:, n_periods - 1] = pred_capital[:, n_periods - 1] * (
            1 + self.fixed_rate
        )

        # Compute risk margin
        risk_margin = pred_capital[:, 0].copy()
        for cy in range(1, n_periods):
            risk_margin -= capital_release[:, cy - 1] / (1 + self.risky_rate) ** cy

        # Risk margin as percentage of initial capital
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            risk_margin_pct = 100 * risk_margin / pred_capital[0, 0]

        return RiskMarginResult(
            num_samples=self.num_samples,
            expected_ultimate=expected_ultimate,
            best_estimate=best_estimate,
            pred_mean=pred_mean,
            pred_assets=pred_assets,
            pred_capital=pred_capital,
            capital_release=capital_release,
            risk_margin=risk_margin,
            risk_margin_pct=risk_margin_pct,
            fixed_rate=self.fixed_rate,
            risky_rate=self.risky_rate,
            tvar_percentile=self.tvar_percentile,
            time_horizon="ultimate",
            ultimate_samples=ultall,
        )

    def calculate_one_year_horizon(
        self,
        n_averaging: int = 12,
        progress_callback: Any | None = None,
    ) -> RiskMarginResult:
        """
        Calculate risk margin at one-year time horizon.

        At the one-year horizon, capital requirements are based on the TVaR
        of the one-year-ahead estimate of ultimate losses. This approach is
        consistent with Solvency II requirements.

        Parameters
        ----------
        n_averaging : int, optional
            Number of independent simulation runs to average for stable
            conditional estimates. Default is 12.
        progress_callback : callable, optional
            Function called with (current, total) to report progress.

        Returns
        -------
        RiskMarginResult
            Container with all risk margin calculation results.

        Notes
        -----
        The key difference from ultimate horizon is that at each future
        calendar year, capital is based on the TVaR of what the estimate
        will be one year later, not on the TVaR of ultimate losses.

        This requires computing E[Ultimate | Data through CY+1] for
        each scenario, which involves nested simulation.
        """
        # Get model parameters
        params = self._extract_csr_parameters()
        trpaid = self._get_paid_triangle()

        # Compute best estimate
        best_estimates = self._compute_best_estimate(params, trpaid)
        best_estimate = np.mean(best_estimates)

        # Compute unconditional ultimate loss by origin
        mean_ult = self._compute_ultimate_by_origin(params, trpaid)
        ultall = mean_ult.sum(axis=1)
        expected_ultimate = np.mean(ultall)

        sig = params["sig"]
        n_periods = self.n_dev

        # Estimate one-year-ahead expected ultimate for each calendar year
        # This requires multiple simulation runs for averaging
        est1yr = np.zeros((self.num_samples, n_periods - 1))

        if self.random_seed is not None:
            np.random.seed(self.random_seed + 100)

        for m in range(n_averaging):
            if progress_callback is not None:
                progress_callback(m, n_averaging)

            # Simulate new future losses
            logloss_p, mu_p = self._simulate_future_losses(params)

            # For each scenario, compute conditional expectations
            for i in range(self.num_samples):
                x_p = logloss_p[i]
                loglike = np.zeros(self.num_samples)

                for cy in range(1, n_periods):
                    # Add likelihood for this calendar year
                    for w in range(cy, self.n_origins):
                        d = self.n_dev - 1 + cy - w
                        x = x_p[w, d]
                        mu = mu_p[:, w, d]
                        sigma = sig[:, d]
                        loglike += stats.norm.logpdf(x, mu, sigma)

                    # Compute posterior weights
                    loglike_shifted = loglike - np.max(loglike)
                    weights = np.exp(loglike_shifted)
                    weights = weights / np.sum(weights)

                    # Posterior mean (one-year-ahead estimate)
                    est1yr[i, cy - 1] += np.sum(weights * ultall)

        est1yr /= n_averaging

        # Now compute capital requirements based on one-year estimates
        if self.random_seed is not None:
            np.random.seed(self.random_seed + 200)

        pred_mean = np.zeros((self.num_samples, n_periods))
        pred_assets = np.zeros((self.num_samples, n_periods))

        # Simulate fresh losses for capital calculation
        logloss_p, mu_p = self._simulate_future_losses(params)

        for i in range(self.num_samples):
            x_p = logloss_p[i]

            # Initial estimates based on one-year-ahead distribution
            pred_mean[i, 0] = np.mean(est1yr[:, 0])
            pred_assets[i, 0] = self._compute_tvar(est1yr[:, 0])

            loglike = np.zeros(self.num_samples)

            for cy in range(1, n_periods):
                # Add likelihood for this calendar year
                for w in range(cy, self.n_origins):
                    d = self.n_dev - 1 + cy - w
                    x = x_p[w, d]
                    mu = mu_p[:, w, d]
                    sigma = sig[:, d]
                    loglike += stats.norm.logpdf(x, mu, sigma)

                # Compute posterior weights
                loglike_shifted = loglike - np.max(loglike)
                weights = np.exp(loglike_shifted)
                weights = weights / np.sum(weights)

                # Use one-year-ahead estimates for this calendar year
                if cy < n_periods - 1:
                    target = est1yr[:, cy]
                else:
                    target = est1yr[:, -1]  # Use last available

                # Resample according to posterior
                indices = np.random.choice(
                    self.num_samples,
                    size=10000,
                    replace=True,
                    p=weights,
                )
                resampled = target[indices]

                pred_mean[i, cy] = np.mean(resampled)
                pred_assets[i, cy] = self._compute_tvar(resampled)

        # Compute required capital
        pred_capital = pred_assets - pred_mean

        # Compute capital release
        capital_release = np.zeros((self.num_samples, n_periods))
        for cy in range(n_periods - 1):
            capital_release[:, cy] = (
                pred_capital[:, cy] * (1 + self.fixed_rate) - pred_capital[:, cy + 1]
            )
        capital_release[:, n_periods - 1] = pred_capital[:, n_periods - 1] * (
            1 + self.fixed_rate
        )

        # Compute risk margin
        risk_margin = pred_capital[:, 0].copy()
        for cy in range(1, n_periods):
            risk_margin -= capital_release[:, cy - 1] / (1 + self.risky_rate) ** cy

        # Risk margin as percentage of initial capital
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            risk_margin_pct = 100 * risk_margin / pred_capital[0, 0]

        return RiskMarginResult(
            num_samples=self.num_samples,
            expected_ultimate=expected_ultimate,
            best_estimate=best_estimate,
            pred_mean=pred_mean,
            pred_assets=pred_assets,
            pred_capital=pred_capital,
            capital_release=capital_release,
            risk_margin=risk_margin,
            risk_margin_pct=risk_margin_pct,
            fixed_rate=self.fixed_rate,
            risky_rate=self.risky_rate,
            tvar_percentile=self.tvar_percentile,
            time_horizon="one_year",
            ultimate_samples=ultall,
        )


def compute_tvar(
    samples: np.ndarray,
    percentile: float = 0.97,
) -> float:
    """
    Compute Tail Value at Risk (TVaR) for a sample.

    TVaR (also known as Conditional VaR or Expected Shortfall) is the
    expected value of losses exceeding the VaR threshold.

    Parameters
    ----------
    samples : np.ndarray
        Array of loss samples.
    percentile : float, optional
        Percentile for VaR threshold. Default is 0.97 (97%).

    Returns
    -------
    float
        TVaR value.

    Examples
    --------
    >>> samples = np.random.normal(100, 20, 10000)
    >>> tvar = compute_tvar(samples, percentile=0.95)
    """
    n = len(samples)
    sorted_samples = np.sort(samples)
    tail_start = int(percentile * n)
    return np.mean(sorted_samples[tail_start:])


def compute_var(
    samples: np.ndarray,
    percentile: float = 0.97,
) -> float:
    """
    Compute Value at Risk (VaR) for a sample.

    VaR is the loss threshold at the specified percentile.

    Parameters
    ----------
    samples : np.ndarray
        Array of loss samples.
    percentile : float, optional
        Percentile for VaR. Default is 0.97 (97%).

    Returns
    -------
    float
        VaR value.

    Examples
    --------
    >>> samples = np.random.normal(100, 20, 10000)
    >>> var = compute_var(samples, percentile=0.95)
    """
    return np.percentile(samples, percentile * 100)
