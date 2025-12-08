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
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
import warnings

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from .estimators import BayesianChainLadderGLM, BayesianCSR


@runtime_checkable
class RiskMarginModel(Protocol):
    """
    Protocol defining the interface for models supporting risk margin calculations.

    Any model implementing these methods can be used with RiskMarginCalculator.
    """

    def get_model_dimensions(self) -> dict[str, int]:
        """
        Get model dimensions.

        Returns dict with keys: n_origins, n_dev, num_samples
        """
        ...

    def get_risk_margin_params(self) -> dict[str, np.ndarray]:
        """Get model parameters needed for risk margin calculations."""
        ...

    def get_paid_triangle(self) -> np.ndarray:
        """Get observed cumulative paid loss triangle."""
        ...

    def simulate_future_losses(
        self, random_seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate future losses. Returns (logloss_p, mu_p)."""
        ...

    def compute_unconditional_ultimate(self) -> np.ndarray:
        """Compute unconditional ultimate loss estimates by origin."""
        ...

    def compute_best_estimate(self, fixed_rate: float = 0.04) -> np.ndarray:
        """Compute best estimate (PV of expected future payouts)."""
        ...

    def compute_log_likelihood(
        self,
        simulated_losses: np.ndarray,
        mu_p: np.ndarray,
        calendar_year: int,
    ) -> np.ndarray:
        """Compute log likelihood for Bayesian updating."""
        ...

    def _check_is_fitted(self) -> None:
        """Check if model is fitted."""
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
        The model must implement the RiskMarginModel protocol.
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

        # Validate model implements required interface
        self._validate_model_interface()

        # Extract model dimensions
        dims = model.get_model_dimensions()
        self.n_origins = dims["n_origins"]
        self.n_dev = dims["n_dev"]
        self.num_samples = dims["num_samples"]

    def _validate_model_interface(self) -> None:
        """Validate that the model implements the required interface."""
        required_methods = [
            "get_model_dimensions",
            "get_risk_margin_params",
            "get_paid_triangle",
            "simulate_future_losses",
            "compute_unconditional_ultimate",
            "compute_best_estimate",
            "compute_log_likelihood",
        ]

        missing = []
        for method in required_methods:
            if not hasattr(self.model, method) or not callable(
                getattr(self.model, method)
            ):
                missing.append(method)

        if missing:
            raise TypeError(
                f"Model must implement risk margin interface. "
                f"Missing methods: {', '.join(missing)}. "
                f"Consider using BayesianCSR which implements the full interface."
            )

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

    def calculate_ultimate_horizon(
        self,
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
        # Get data from model
        params = self.model.get_risk_margin_params()
        sig = params["sig"]

        # Compute best estimate
        best_estimates = self.model.compute_best_estimate(self.fixed_rate)
        best_estimate = np.mean(best_estimates)

        # Compute unconditional ultimate loss by origin
        mean_ult = self.model.compute_unconditional_ultimate()
        ultall = mean_ult.sum(axis=1)  # Total ultimate by sample
        expected_ultimate = np.mean(ultall)

        # Simulate future losses and get mu_p for likelihood calculation
        logloss_p, mu_p = self.model.simulate_future_losses(self.random_seed)

        # Initialize output arrays
        n_periods = self.n_dev
        pred_mean = np.zeros((self.num_samples, n_periods))
        pred_assets = np.zeros((self.num_samples, n_periods))

        if self.random_seed is not None:
            np.random.seed(self.random_seed + 1)

        # Process each scenario - following R code structure
        for i in range(self.num_samples):
            if progress_callback is not None:
                progress_callback(i, self.num_samples)

            # Get simulated losses for this scenario
            x_p = logloss_p[i]  # Shape (n_origins, n_dev)

            # Initial (unconditional) estimates
            pred_mean[i, 0] = np.mean(ultall)
            pred_assets[i, 0] = self._compute_tvar(ultall)

            # Accumulate log likelihood as we observe more calendar years
            loglike = np.zeros(self.num_samples)

            # For each calendar year, accumulate likelihood and update posterior
            for cy in range(1, n_periods):
                # Add likelihood of observing calendar year cy data
                for w in range(cy, self.n_origins):
                    d = self.n_dev - 1 + cy - w
                    x = x_p[w, d]
                    mu = mu_p[:, w, d]
                    sigma = sig[:, d]
                    loglike += stats.norm.logpdf(x, mu, sigma)

                # Compute posterior weights (stabilized)
                loglike_shifted = loglike - np.max(loglike)
                weights = np.exp(loglike_shifted)
                weights = weights / np.sum(weights)

                # Compute posterior mean and TVaR
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
        # Get data from model
        params = self.model.get_risk_margin_params()
        sig = params["sig"]

        # Compute best estimate
        best_estimates = self.model.compute_best_estimate(self.fixed_rate)
        best_estimate = np.mean(best_estimates)

        # Compute unconditional ultimate loss by origin
        mean_ult = self.model.compute_unconditional_ultimate()
        ultall = mean_ult.sum(axis=1)
        expected_ultimate = np.mean(ultall)

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
            logloss_p, mu_p = self.model.simulate_future_losses(None)

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
        logloss_p, mu_p = self.model.simulate_future_losses(None)

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
