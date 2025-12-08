"""
High-level estimator classes for Bayesian chain ladder.

This module provides scikit-learn/chainladder-style estimator classes
for Bayesian stochastic loss reserving.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from .models import build_bambi_model, build_csr_model, fit_model, sample_prior_predictive
from .utils import (
    add_categorical_columns,
    get_future_dataframe,
    prepare_csr_data,
    prepare_model_data,
    triangle_to_dataframe,
    validate_triangle,
)

if TYPE_CHECKING:
    import chainladder as cl


class BayesianChainLadderGLM:
    """
    Bayesian cross-classified chain ladder model using GLM.

    This estimator implements a Bayesian version of the stochastic
    cross-classified chain ladder model described in Taylor & McGuire (2016).
    It uses Bambi (or PyMC directly) to fit GLM models with MCMC sampling.

    The model follows the standard ODP (Over-Dispersed Poisson) cross-classified
    structure:

        log(μ_kj) = intercept + α_k + β_j [+ γ_{k+j-1}] [+ log(exposure)]

    where:
    - μ_kj is the expected incremental loss for origin k, development j
    - α_k are origin (accident year) effects
    - β_j are development period effects
    - γ_{k+j-1} are optional calendar period effects
    - exposure is an optional offset term

    Parameters
    ----------
    formula : str, optional
        Bambi/Patsy-style formula for the model.
        Default is "incremental ~ 1 + C(origin) + C(dev)" for the standard
        cross-classified model without calendar effects.
        Use "incremental ~ 1 + C(origin) + C(dev) + C(calendar)" to include
        calendar year effects.
    family : str, optional
        Response distribution family. Options:
        - "negativebinomial" (default): Overdispersed Poisson-like
        - "poisson": Standard Poisson
        - "gamma": Gamma distribution for positive continuous
        Default is "negativebinomial".
    link : str, optional
        Link function. Default is None (uses family default, typically "log").
    exposure : str, optional
        Name of exposure column for offset term (e.g., "earned_premium").
        If provided, log(exposure) is used as an offset in the linear predictor.
    priors : dict, optional
        Dictionary of prior specifications for model parameters.
    draws : int, optional
        Number of posterior samples per chain. Default is 2000.
    tune : int, optional
        Number of tuning samples. Default is 1000.
    chains : int, optional
        Number of MCMC chains. Default is 4.
    target_accept : float, optional
        Target acceptance probability for NUTS sampler. Default is 0.9.
    random_seed : int, optional
        Random seed for reproducibility.
    backend : str, optional
        Modeling backend. Currently only "bambi" is fully supported.
        Default is "bambi".

    Attributes
    ----------
    model_ : bmb.Model
        The fitted Bambi model.
    idata : az.InferenceData
        ArviZ InferenceData object with posterior samples and predictions.
    data_ : pd.DataFrame
        The observed data in long format.
    future_data_ : pd.DataFrame
        The future/prediction data in long format.
    fitted_ : pd.DataFrame
        Fitted values for observed cells.
    ultimate_ : pd.DataFrame
        Posterior summary of ultimate losses by origin.
    ibnr_ : pd.DataFrame
        Posterior summary of IBNR reserves by origin.
    reserves_posterior_ : xr.DataArray
        Full posterior samples of reserves by origin.

    Examples
    --------
    >>> import chainladder as cl
    >>> from bayesianchainladder import BayesianChainLadderGLM
    >>>
    >>> # Load sample triangle
    >>> tri = cl.load_sample("GenIns")
    >>>
    >>> # Fit Bayesian chain ladder
    >>> model = BayesianChainLadderGLM(
    ...     formula="incremental ~ 1 + C(origin) + C(dev)",
    ...     draws=1000,
    ...     tune=500,
    ... )
    >>> model.fit(tri)
    >>>
    >>> # Get reserve summary
    >>> print(model.summary())

    See Also
    --------
    chainladder.Development : Traditional chain ladder development
    chainladder.Chainladder : Traditional chain ladder reserving

    References
    ----------
    Taylor, G. and McGuire, G. (2016). Stochastic Loss Reserving Using
    Generalized Linear Models. CAS Monograph Series Number 3.
    """

    def __init__(
        self,
        formula: str = "incremental ~ 1 + C(origin) + C(dev)",
        family: str = "negativebinomial",
        link: str | None = None,
        exposure: str | None = None,
        priors: dict[str, Any] | None = None,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int | None = None,
        backend: str = "bambi",
    ):
        self.formula = formula
        self.family = family
        self.link = link
        self.exposure = exposure
        self.priors = priors
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept
        self.random_seed = random_seed
        self.backend = backend

        # Fitted attributes (set by fit())
        self.model_: bmb.Model | None = None
        self.idata: az.InferenceData | None = None
        self.data_: pd.DataFrame | None = None
        self.future_data_: pd.DataFrame | None = None
        self.triangle_: cl.Triangle | None = None
        self.fitted_: pd.DataFrame | None = None
        self.ultimate_: pd.DataFrame | None = None
        self.ibnr_: pd.DataFrame | None = None
        self.reserves_posterior_: xr.DataArray | None = None
        self._is_fitted: bool = False

    def fit(
        self,
        triangle: cl.Triangle,
        exposure_triangle: cl.Triangle | None = None,
        sample_weight: pd.Series | np.ndarray | None = None,
    ) -> "BayesianChainLadderGLM":
        """
        Fit the Bayesian chain ladder model to a triangle.

        Parameters
        ----------
        triangle : chainladder.Triangle
            The claims triangle (cumulative or incremental).
        exposure_triangle : chainladder.Triangle, optional
            Optional exposure triangle (e.g., earned premium by origin).
        sample_weight : array-like, optional
            Sample weights for observations.

        Returns
        -------
        self
            The fitted estimator.
        """
        # Validate input
        validate_triangle(triangle)

        self.triangle_ = triangle.copy()

        # Convert triangle to long format
        self.data_, self.future_data_ = prepare_model_data(
            triangle,
            exposure_triangle=exposure_triangle,
            exposure_column=self.exposure if self.exposure else "exposure",
        )

        # Add categorical encoding (spline columns stay numeric)
        self.data_ = add_categorical_columns(self.data_, formula=self.formula)
        self.future_data_ = add_categorical_columns(self.future_data_, formula=self.formula)

        # Validate data compatibility with chosen family
        self._validate_data_family_compatibility()

        # Set up default priors based on data scale and family/link
        priors = self.priors
        if priors is None:
            import bambi as bmb

            # Compute data-adaptive intercept prior
            response_values = self.data_["incremental"].values
            response_mean = response_values.mean()
            response_std = response_values.std()

            # For log link (negativebinomial, poisson, gamma), use log scale
            # For identity link (gaussian), use original scale
            if self.family.lower() in ("gaussian", "normal"):
                # Identity link - intercept is on original scale
                intercept_mu = response_mean
                intercept_sigma = max(response_std, abs(response_mean) * 0.5)
            else:
                # Log link - intercept is on log scale
                # Use positive values only for computing mean on log scale
                positive_values = response_values[response_values > 0]
                if len(positive_values) > 0:
                    positive_mean = positive_values.mean()
                    intercept_mu = np.log(positive_mean)
                else:
                    intercept_mu = 0.0
                intercept_sigma = 2.0

            priors = {
                "Intercept": bmb.Prior("Normal", mu=intercept_mu, sigma=intercept_sigma),
            }

        # Build the model
        offset = self.exposure if self.exposure else None
        self.model_ = build_bambi_model(
            data=self.data_,
            formula=self.formula,
            family=self.family,
            link=self.link,
            priors=priors,
            offset=offset,
        )

        # Fit the model
        self.idata = fit_model(
            self.model_,
            draws=self.draws,
            tune=self.tune,
            chains=self.chains,
            target_accept=self.target_accept,
            random_seed=self.random_seed,
        )

        # Generate predictions
        self._compute_predictions()

        self._is_fitted = True
        return self

    def _compute_predictions(self) -> None:
        """Compute fitted values and future predictions."""
        # Fitted values for observed data
        # Use kind="response_params" as "mean" is deprecated in newer Bambi
        try:
            self.model_.predict(
                self.idata, data=self.data_, kind="response_params", inplace=True
            )
        except (TypeError, ValueError):
            # Fall back to "mean" for older Bambi versions
            self.model_.predict(self.idata, data=self.data_, kind="mean", inplace=True)

        # Get response name - Bambi stores predictions under 'mu' for mean
        # or in posterior_predictive for 'response' kind
        response_name = self.model_.response_component.response.name

        # Try to find the mean predictions - Bambi may use different names
        mean_name = f"{response_name}_mean"
        if mean_name in self.idata.posterior:
            posterior_data = self.idata.posterior
        elif "mu" in self.idata.posterior:
            mean_name = "mu"
            posterior_data = self.idata.posterior
        else:
            # Try posterior predictive
            mean_name = response_name
            posterior_data = self.idata.posterior_predictive

        # Extract fitted values
        fitted_mean = posterior_data[mean_name].mean(dim=["chain", "draw"])
        self.fitted_ = self.data_.copy()
        self.fitted_["fitted_mean"] = fitted_mean.values

        # Predict future cells
        if len(self.future_data_) > 0:
            try:
                self.model_.predict(
                    self.idata, data=self.future_data_, kind="response_params", inplace=True
                )
            except (TypeError, ValueError):
                self.model_.predict(
                    self.idata, data=self.future_data_, kind="mean", inplace=True
                )

            # Get the future predictions using same name discovery
            if mean_name in self.idata.posterior:
                future_mean = self.idata.posterior[mean_name]
            else:
                future_mean = self.idata.posterior_predictive[mean_name]

            # Compute reserves by origin
            self._compute_reserves(future_mean)

    def _compute_reserves(self, future_predictions: xr.DataArray) -> None:
        """Compute reserve distributions from future predictions."""
        # Find the observation dimension name
        response_name = self.model_.response_component.response.name
        obs_dim = None
        for dim in future_predictions.dims:
            if "obs" in dim.lower() or dim == "__obs__":
                obs_dim = dim
                break

        if obs_dim is None:
            # Fall back to expected name
            obs_dim = f"{response_name}_obs"

        # Get unique origins
        origins = sorted(self.future_data_["origin"].unique())

        # Initialize arrays for reserve samples
        reserve_samples = {}

        # The future predictions may include all data or just future cells
        # We need to select only the future cells (last N observations)
        n_future = len(self.future_data_)
        n_total_obs = future_predictions.sizes.get(obs_dim, n_future)

        # Future observations are the last n_future items
        future_start = n_total_obs - n_future

        for origin in origins:
            # Get indices for this origin in future_data_
            mask = self.future_data_["origin"] == origin
            origin_future_idx = np.where(mask)[0]

            if len(origin_future_idx) > 0:
                # Map to positions in the full prediction array
                pos = [future_start + i for i in origin_future_idx]

                # Sum predictions for this origin across all future cells
                origin_preds = future_predictions.isel({obs_dim: pos})
                reserve_samples[origin] = origin_preds.sum(dim=obs_dim)

        # Create DataArray with reserves by origin
        if reserve_samples:
            reserves_list = []
            for origin in origins:
                if origin in reserve_samples:
                    reserves_list.append(reserve_samples[origin])

            self.reserves_posterior_ = xr.concat(
                reserves_list, dim=pd.Index(origins, name="origin")
            )

            # Compute ultimate and IBNR summaries
            self._compute_reserve_summaries()

    def _compute_reserve_summaries(self) -> None:
        """Compute summary statistics for reserves."""
        if self.reserves_posterior_ is None:
            return

        # Get paid to date for each origin
        paid_to_date = self.data_.groupby("origin", observed=True)["incremental"].sum()

        # Compute summary statistics
        reserves_flat = self.reserves_posterior_.stack(sample=["chain", "draw"])

        summary_data = []
        for origin in self.reserves_posterior_.coords["origin"].values:
            origin_reserves = reserves_flat.sel(origin=origin)

            ibnr_mean = float(origin_reserves.mean())
            ibnr_std = float(origin_reserves.std())
            ibnr_median = float(origin_reserves.median())
            ibnr_q05 = float(origin_reserves.quantile(0.05))
            ibnr_q25 = float(origin_reserves.quantile(0.25))
            ibnr_q75 = float(origin_reserves.quantile(0.75))
            ibnr_q95 = float(origin_reserves.quantile(0.95))

            paid = paid_to_date.get(origin, 0)
            ultimate_mean = paid + ibnr_mean
            ultimate_std = ibnr_std
            ultimate_median = paid + ibnr_median
            ultimate_q05 = paid + ibnr_q05
            ultimate_q25 = paid + ibnr_q25
            ultimate_q75 = paid + ibnr_q75
            ultimate_q95 = paid + ibnr_q95

            summary_data.append(
                {
                    "origin": origin,
                    "paid_to_date": paid,
                    "ibnr_mean": ibnr_mean,
                    "ibnr_std": ibnr_std,
                    "ibnr_median": ibnr_median,
                    "ibnr_5%": ibnr_q05,
                    "ibnr_25%": ibnr_q25,
                    "ibnr_75%": ibnr_q75,
                    "ibnr_95%": ibnr_q95,
                    "ultimate_mean": ultimate_mean,
                    "ultimate_std": ultimate_std,
                    "ultimate_median": ultimate_median,
                    "ultimate_5%": ultimate_q05,
                    "ultimate_25%": ultimate_q25,
                    "ultimate_75%": ultimate_q75,
                    "ultimate_95%": ultimate_q95,
                }
            )

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.set_index("origin")

        # Split into ibnr_ and ultimate_
        self.ibnr_ = summary_df[
            [
                "ibnr_mean",
                "ibnr_std",
                "ibnr_median",
                "ibnr_5%",
                "ibnr_25%",
                "ibnr_75%",
                "ibnr_95%",
            ]
        ].copy()
        self.ibnr_.columns = ["mean", "std", "median", "5%", "25%", "75%", "95%"]

        self.ultimate_ = summary_df[
            [
                "paid_to_date",
                "ultimate_mean",
                "ultimate_std",
                "ultimate_median",
                "ultimate_5%",
                "ultimate_25%",
                "ultimate_75%",
                "ultimate_95%",
            ]
        ].copy()
        self.ultimate_.columns = [
            "paid_to_date",
            "mean",
            "std",
            "median",
            "5%",
            "25%",
            "75%",
            "95%",
        ]

    def predict(
        self,
        triangle: cl.Triangle | None = None,
        kind: Literal["mean", "response"] = "mean",
    ) -> pd.DataFrame:
        """
        Generate predictions for a triangle.

        Parameters
        ----------
        triangle : chainladder.Triangle, optional
            Triangle to predict. If None, uses the fitted triangle's
            future cells.
        kind : {"mean", "response"}, optional
            Type of prediction:
            - "mean": Expected values
            - "response": Samples from posterior predictive
            Default is "mean".

        Returns
        -------
        pd.DataFrame
            DataFrame with predictions.
        """
        self._check_is_fitted()

        # Find the mean variable name
        response_name = self.model_.response_component.response.name
        mean_name = f"{response_name}_mean"
        if mean_name not in self.idata.posterior:
            mean_name = "mu"

        if triangle is None:
            # Return predictions for future cells of fitted triangle
            if len(self.future_data_) == 0:
                return pd.DataFrame()

            future_mean = self.idata.posterior[mean_name].mean(dim=["chain", "draw"])

            result = self.future_data_.copy()
            result["predicted_mean"] = future_mean.values[-len(self.future_data_) :]

            return result

        else:
            # Predict for new triangle
            new_data, new_future = prepare_model_data(triangle)
            combined = pd.concat([new_data, new_future], ignore_index=True)
            combined = add_categorical_columns(combined)

            self.model_.predict(self.idata, data=combined, kind=kind, inplace=True)

            pred_mean = self.idata.posterior[mean_name].mean(dim=["chain", "draw"])
            combined["predicted_mean"] = pred_mean.values[-len(combined) :]

            return combined

    def summary(
        self,
        include_totals: bool = True,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Return summary table of reserves and ultimates.

        Parameters
        ----------
        include_totals : bool, optional
            Whether to include total row. Default is True.
        quantiles : list[float], optional
            Quantiles to include. Default is [0.05, 0.25, 0.5, 0.75, 0.95].

        Returns
        -------
        pd.DataFrame
            Summary table with reserve statistics by origin.
        """
        self._check_is_fitted()

        if self.ultimate_ is None:
            raise ValueError("No reserve summary available. Model may not have future cells.")

        result = pd.concat(
            [
                self.ultimate_[["paid_to_date", "mean", "std", "median"]],
                self.ibnr_[["mean", "std", "median"]],
            ],
            axis=1,
            keys=["Ultimate", "IBNR"],
        )

        if include_totals:
            # Compute total reserves
            total_paid = self.ultimate_["paid_to_date"].sum()

            # Get total reserve distribution
            total_reserves = self.reserves_posterior_.sum(dim="origin")
            total_flat = total_reserves.stack(sample=["chain", "draw"])

            total_ibnr_mean = float(total_flat.mean())
            total_ibnr_std = float(total_flat.std())
            total_ibnr_median = float(total_flat.median())

            total_ult_mean = total_paid + total_ibnr_mean
            total_ult_std = total_ibnr_std
            total_ult_median = total_paid + total_ibnr_median

            total_row = pd.DataFrame(
                {
                    ("Ultimate", "paid_to_date"): [total_paid],
                    ("Ultimate", "mean"): [total_ult_mean],
                    ("Ultimate", "std"): [total_ult_std],
                    ("Ultimate", "median"): [total_ult_median],
                    ("IBNR", "mean"): [total_ibnr_mean],
                    ("IBNR", "std"): [total_ibnr_std],
                    ("IBNR", "median"): [total_ibnr_median],
                },
                index=["Total"],
            )

            result = pd.concat([result, total_row])

        return result

    def get_parameter_summary(
        self,
        var_names: list[str] | None = None,
        filter_vars: str | None = None,
        hdi_prob: float = 0.94,
    ) -> pd.DataFrame:
        """
        Get summary statistics for model parameters.

        Parameters
        ----------
        var_names : list[str], optional
            Parameter names to include. If None, includes all.
        hdi_prob : float, optional
            Probability mass for HDI. Default is 0.94.

        Returns
        -------
        pd.DataFrame
            Parameter summary table.
        """
        self._check_is_fitted()
        return az.summary(self.idata, var_names=var_names, filter_vars=filter_vars, hdi_prob=hdi_prob)

    def get_origin_effects(self) -> pd.DataFrame:
        """
        Extract posterior summary of origin (accident year) effects.

        Returns
        -------
        pd.DataFrame
            Summary of origin effects.
        """
        self._check_is_fitted()

        # Find origin effect variable in posterior
        for var in self.idata.posterior.data_vars:
            if "origin" in var.lower():
                return az.summary(self.idata, var_names=[var])

        raise ValueError("Could not find origin effects in model")

    def get_development_effects(self) -> pd.DataFrame:
        """
        Extract posterior summary of development period effects.

        Returns
        -------
        pd.DataFrame
            Summary of development effects.
        """
        self._check_is_fitted()

        # Find dev effect variable in posterior
        for var in self.idata.posterior.data_vars:
            if "dev" in var.lower():
                return az.summary(self.idata, var_names=[var])

        raise ValueError("Could not find development effects in model")

    def sample_reserves(
        self,
        n_samples: int = 1000,
        random_seed: int | None = None,
    ) -> np.ndarray:
        """
        Draw samples from the reserve distribution.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to draw. Default is 1000.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Array of reserve samples (shape: n_samples).
        """
        self._check_is_fitted()

        if self.reserves_posterior_ is None:
            raise ValueError("No reserve posterior available")

        # Get total reserves
        total_reserves = self.reserves_posterior_.sum(dim="origin")
        all_samples = total_reserves.stack(sample=["chain", "draw"]).values

        if random_seed is not None:
            np.random.seed(random_seed)

        # Sample with replacement if needed
        if n_samples <= len(all_samples):
            indices = np.random.choice(len(all_samples), size=n_samples, replace=False)
        else:
            indices = np.random.choice(len(all_samples), size=n_samples, replace=True)

        return all_samples[indices]

    def build_model(
        self,
        triangle: cl.Triangle,
        exposure_triangle: cl.Triangle | None = None,
    ) -> "BayesianChainLadderGLM":
        """
        Build the Bayesian model without fitting.

        This is useful for prior predictive checks before committing to
        a full MCMC fit. Build the model, examine prior predictive samples,
        adjust priors if needed, then call fit().

        Parameters
        ----------
        triangle : chainladder.Triangle
            The claims triangle (cumulative or incremental).
        exposure_triangle : chainladder.Triangle, optional
            Optional exposure triangle (e.g., earned premium by origin).

        Returns
        -------
        self
            The estimator with model_ attribute set.

        Examples
        --------
        >>> model = BayesianChainLadderGLM(formula="incremental ~ 1 + C(origin) + C(dev)")
        >>> model.build_model(triangle)
        >>> prior_idata = model.sample_prior_predictive(draws=500)
        >>> # Examine prior predictions, adjust priors if needed
        >>> model.fit(triangle)  # Full MCMC fit
        """
        # Validate input
        validate_triangle(triangle)

        self.triangle_ = triangle.copy()

        # Convert triangle to long format
        self.data_, self.future_data_ = prepare_model_data(
            triangle,
            exposure_triangle=exposure_triangle,
            exposure_column=self.exposure if self.exposure else "exposure",
        )

        # Add categorical encoding
        self.data_ = add_categorical_columns(self.data_, formula=self.formula)
        self.future_data_ = add_categorical_columns(self.future_data_, formula=self.formula)

        # Validate data compatibility with chosen family
        self._validate_data_family_compatibility()

        # Set up default priors based on data scale and family/link
        priors = self.priors
        if priors is None:
            import bambi as bmb

            # Compute data-adaptive intercept prior
            response_values = self.data_["incremental"].values
            response_mean = response_values.mean()
            response_std = response_values.std()

            if self.family.lower() in ("gaussian", "normal"):
                intercept_mu = response_mean
                intercept_sigma = max(response_std, abs(response_mean) * 0.5)
            else:
                positive_values = response_values[response_values > 0]
                if len(positive_values) > 0:
                    positive_mean = positive_values.mean()
                    intercept_mu = np.log(positive_mean)
                else:
                    intercept_mu = 0.0
                intercept_sigma = 2.0

            priors = {
                "Intercept": bmb.Prior("Normal", mu=intercept_mu, sigma=intercept_sigma),
            }

        # Build the model
        offset = self.exposure if self.exposure else None
        self.model_ = build_bambi_model(
            data=self.data_,
            formula=self.formula,
            family=self.family,
            link=self.link,
            priors=priors,
            offset=offset,
        )

        return self

    def sample_prior_predictive(
        self,
        triangle: cl.Triangle | None = None,
        exposure_triangle: cl.Triangle | None = None,
        draws: int = 500,
        random_seed: int | None = None,
    ) -> az.InferenceData:
        """
        Sample from the prior predictive distribution.

        Prior predictive checks are critical for Bayesian loss reserving models.
        They help verify that priors produce predictions consistent with domain
        knowledge before fitting to data. This is especially important because:

        1. Loss reserves can span many orders of magnitude
        2. Development patterns should show realistic payment patterns
        3. Priors that are too vague can produce unrealistic predictions
        4. Priors that are too tight may not allow the data to speak

        Parameters
        ----------
        triangle : chainladder.Triangle, optional
            Triangle to use for predictions. If None, uses the previously
            built triangle (requires calling build_model() first).
        exposure_triangle : chainladder.Triangle, optional
            Optional exposure triangle.
        draws : int, optional
            Number of prior predictive samples. Default is 500.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        az.InferenceData
            InferenceData with prior and prior_predictive groups.

        Examples
        --------
        >>> import chainladder as cl
        >>> from bayesianchainladder import BayesianChainLadderGLM
        >>> from bayesianchainladder.plots import plot_prior_predictive
        >>>
        >>> triangle = cl.load_sample("raa")
        >>> model = BayesianChainLadderGLM()
        >>>
        >>> # Sample from prior predictive
        >>> prior_idata = model.sample_prior_predictive(triangle, draws=500)
        >>>
        >>> # Visualize prior predictions
        >>> fig, ax = plot_prior_predictive(model, prior_idata)

        Notes
        -----
        Key things to check in prior predictive samples for loss reserving:

        1. **Incremental loss magnitudes**: Are predicted losses in a
           reasonable range? (e.g., not billions for a small portfolio)

        2. **Development pattern**: Do early periods show higher losses
           with decreasing amounts over time?

        3. **Total reserves**: Is the implied reserve distribution
           consistent with portfolio expectations?

        4. **Tail behavior**: Are extreme predictions (95th percentile)
           plausible worst-case scenarios?
        """
        if triangle is not None:
            # Build model with provided triangle
            self.build_model(triangle, exposure_triangle)
        elif self.model_ is None:
            raise ValueError(
                "No triangle provided and model not yet built. "
                "Either pass a triangle or call build_model() first."
            )

        # Sample from prior predictive using the model
        prior_idata = sample_prior_predictive(
            self.model_,
            draws=draws,
            random_seed=random_seed if random_seed is not None else self.random_seed,
        )

        # Store prior predictive data and model data for later analysis
        self.prior_idata_ = prior_idata

        return prior_idata

    def get_prior_predictive_summary(
        self,
        prior_idata: az.InferenceData | None = None,
        by: str | None = None,
    ) -> pd.DataFrame:
        """
        Get summary statistics of prior predictive samples.

        Parameters
        ----------
        prior_idata : az.InferenceData, optional
            Prior predictive samples. If None, uses stored prior_idata_.
        by : str, optional
            Group summary by "origin", "dev", or "calendar".
            If None, returns per-observation summary.

        Returns
        -------
        pd.DataFrame
            Summary statistics of prior predictive samples.
        """
        if prior_idata is None:
            if not hasattr(self, "prior_idata_") or self.prior_idata_ is None:
                raise ValueError(
                    "No prior predictive samples available. "
                    "Call sample_prior_predictive() first."
                )
            prior_idata = self.prior_idata_

        if "prior_predictive" not in prior_idata.groups():
            raise ValueError("InferenceData must contain prior_predictive group")

        # Get response name from model
        response_name = self.model_.response_component.response.name

        # Get prior predictive samples
        pp = prior_idata.prior_predictive[response_name]

        # Stack chains and draws
        pp_flat = pp.stack(sample=["chain", "draw"])

        # Compute summary
        quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
        summary_data = {
            "mean": pp_flat.mean(dim="sample").values,
            "std": pp_flat.std(dim="sample").values,
        }
        for q in quantiles:
            summary_data[f"{q*100:.1f}%"] = pp_flat.quantile(q, dim="sample").values

        summary_df = pd.DataFrame(summary_data)

        # Add observation metadata
        if self.data_ is not None:
            summary_df["origin"] = self.data_["origin"].values
            summary_df["dev"] = self.data_["dev"].values
            if "calendar" in self.data_.columns:
                summary_df["calendar"] = self.data_["calendar"].values

        # Aggregate by group if requested
        if by is not None and by in ["origin", "dev", "calendar"]:
            if by not in summary_df.columns:
                raise ValueError(f"Column '{by}' not found in data")

            # Sum means across group (for aggregate statistics)
            agg_summary = summary_df.groupby(by).agg({
                "mean": "sum",
                "std": lambda x: np.sqrt((x**2).sum()),  # Sum variances, take sqrt
                "2.5%": "sum",
                "25.0%": "sum",
                "50.0%": "sum",
                "75.0%": "sum",
                "97.5%": "sum",
            })
            return agg_summary

        return summary_df

    def _check_is_fitted(self) -> None:
        """Check if the model has been fitted."""
        if not self._is_fitted:
            raise ValueError(
                "Model has not been fitted. Call fit() before using this method."
            )

    def _validate_data_family_compatibility(self) -> None:
        """Validate that data is compatible with the chosen distribution family."""
        response = self.data_["incremental"].values
        family_lower = self.family.lower()

        # Count distributions require non-negative values
        count_families = ("negativebinomial", "negative_binomial", "negbinom", "poisson")
        # Positive continuous distributions require positive values
        positive_families = ("gamma",)

        min_val = response.min()
        has_negative = min_val < 0
        has_zero_or_negative = min_val <= 0

        if family_lower in count_families:
            if has_negative:
                negative_count = (response < 0).sum()
                raise ValueError(
                    f"The '{self.family}' family requires non-negative values, but the "
                    f"incremental data contains {negative_count} negative values "
                    f"(min={min_val:.2f}). Consider using:\n"
                    f"  - family='gaussian' for data with negative values\n"
                    f"  - A triangle with only positive incremental values\n"
                    f"  - Transforming or truncating negative values before fitting"
                )
            # Check for non-integer values (warning only for count distributions)
            if not np.allclose(response, np.round(response)):
                import warnings
                warnings.warn(
                    f"The '{self.family}' family is intended for count (integer) data, "
                    f"but the incremental data contains non-integer values. "
                    f"Consider using family='gamma' for positive continuous data.",
                    UserWarning,
                )

        elif family_lower in positive_families:
            if has_zero_or_negative:
                nonpositive_count = (response <= 0).sum()
                raise ValueError(
                    f"The '{self.family}' family requires strictly positive values, but the "
                    f"incremental data contains {nonpositive_count} non-positive values "
                    f"(min={min_val:.2f}). Consider using:\n"
                    f"  - family='gaussian' for data with negative or zero values\n"
                    f"  - Adding a small constant to shift values positive"
                )

    # =========================================================================
    # Risk Margin Support Methods
    # =========================================================================

    def get_model_dimensions(self) -> dict[str, int]:
        """
        Get model dimensions for risk margin calculations.

        Returns
        -------
        dict[str, int]
            Dictionary with:
            - n_origins: Number of origin years
            - n_dev: Number of development periods
            - num_samples: Total number of MCMC samples
        """
        self._check_is_fitted()

        origins = sorted(self.data_["origin"].unique())
        dev_periods = sorted(self.data_["dev"].unique())

        posterior = self.idata.posterior
        n_chains = posterior.dims["chain"]
        n_draws = posterior.dims["draw"]

        return {
            "n_origins": len(origins),
            "n_dev": len(dev_periods),
            "num_samples": n_chains * n_draws,
        }

    def get_risk_margin_params(self) -> dict[str, np.ndarray]:
        """
        Extract model parameters needed for risk margin calculations.

        For the GLM model, this returns the fitted linear predictor components.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing flattened posterior samples of model
            parameters. Keys depend on the formula used.
        """
        self._check_is_fitted()

        posterior = self.idata.posterior
        n_chains = posterior.dims["chain"]
        n_draws = posterior.dims["draw"]
        num_samples = n_chains * n_draws

        params = {}
        for var_name in posterior.data_vars:
            values = posterior[var_name].values
            # Flatten chain and draw dimensions
            if values.ndim == 2:
                params[var_name] = values.flatten()
            else:
                params[var_name] = values.reshape(num_samples, -1)

        return params

    def get_paid_triangle(self) -> np.ndarray:
        """
        Get the observed cumulative paid loss triangle.

        Note: The GLM model works on incremental losses internally,
        but this returns cumulative values for consistency.

        Returns
        -------
        np.ndarray
            Cumulative paid losses, shape (n_origins, n_dev).
            NaN values indicate unobserved (future) cells.
        """
        self._check_is_fitted()

        # Convert incremental to cumulative if needed
        df = self.data_.copy()

        # Check if cumulative column exists, if not compute it
        if "cumulative" not in df.columns:
            df = df.sort_values(["origin", "dev"])
            df["cumulative"] = df.groupby("origin")["incremental"].cumsum()

        # Pivot to triangle format
        pivot = df.pivot(index="origin", columns="dev", values="cumulative")
        return pivot.values

    def get_incremental_triangle(self) -> np.ndarray:
        """
        Get the observed incremental loss triangle.

        Returns
        -------
        np.ndarray
            Incremental losses, shape (n_origins, n_dev).
            NaN values indicate unobserved (future) cells.
        """
        self._check_is_fitted()
        pivot = self.data_.pivot(index="origin", columns="dev", values="incremental")
        return pivot.values

    def get_premium(self) -> np.ndarray | None:
        """
        Get premium (exposure) values by origin if available.

        Returns
        -------
        np.ndarray or None
            Premium for each origin, shape (n_origins,), or None
            if exposure was not provided.
        """
        self._check_is_fitted()

        if self.exposure is None or self.exposure not in self.data_.columns:
            return None

        return self.data_.groupby("origin")[self.exposure].first().values

    def get_log_premium(self) -> np.ndarray | None:
        """
        Get log premium values by origin if available.

        Returns
        -------
        np.ndarray or None
            Log premium for each origin, shape (n_origins,), or None
            if exposure was not provided.
        """
        premium = self.get_premium()
        if premium is None:
            return None
        return np.log(premium)

    def simulate_future_losses(
        self,
        random_seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate future incremental losses for lower triangle cells.

        For GLM models, this samples from the posterior predictive distribution.

        Parameters
        ----------
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (incremental_samples, mean_samples) - Simulated incremental losses
            and expected values, each of shape (num_samples, n_origins, n_dev).
        """
        self._check_is_fitted()

        if random_seed is not None:
            np.random.seed(random_seed)

        dims = self.get_model_dimensions()
        num_samples = dims["num_samples"]
        n_origins = dims["n_origins"]
        n_dev = dims["n_dev"]

        # Get mean predictions for future cells
        response_name = self.model_.response_component.response.name
        mean_name = f"{response_name}_mean"
        if mean_name not in self.idata.posterior:
            mean_name = "mu"

        if len(self.future_data_) == 0:
            return (
                np.zeros((num_samples, n_origins, n_dev)),
                np.zeros((num_samples, n_origins, n_dev)),
            )

        # Get predicted means for future cells
        future_means = self.idata.posterior[mean_name]

        # Stack chains and draws
        future_means_flat = future_means.stack(sample=["chain", "draw"])

        # Initialize output arrays
        mean_samples = np.zeros((num_samples, n_origins, n_dev))
        incremental_samples = np.zeros((num_samples, n_origins, n_dev))

        # Map future data back to triangle positions
        n_future = len(self.future_data_)
        n_total = future_means_flat.sizes.get(
            future_means_flat.dims[0], n_future
        )
        future_start = n_total - n_future

        for idx, row in self.future_data_.reset_index(drop=True).iterrows():
            origin_idx = int(row["origin"]) - 1  # Assuming 1-indexed
            dev_idx = int(row["dev"]) - 1

            if origin_idx < n_origins and dev_idx < n_dev:
                obs_idx = future_start + idx
                means = future_means_flat.isel(
                    {future_means_flat.dims[0]: obs_idx}
                ).values

                mean_samples[:, origin_idx, dev_idx] = means
                # For negative binomial, sample from the distribution
                # For simplicity, we'll use the mean as the sample
                # A full implementation would sample from NB/Poisson/Gamma
                incremental_samples[:, origin_idx, dev_idx] = means

        return incremental_samples, mean_samples

    def compute_unconditional_ultimate(self) -> np.ndarray:
        """
        Compute unconditional ultimate loss estimates by origin.

        This sums observed losses with expected future losses.

        Returns
        -------
        np.ndarray
            Ultimate estimates, shape (num_samples, n_origins).
        """
        self._check_is_fitted()

        dims = self.get_model_dimensions()
        num_samples = dims["num_samples"]
        n_origins = dims["n_origins"]

        # Get observed cumulative losses at latest diagonal
        observed_cumulative = self.data_.groupby("origin")["incremental"].sum()

        # Get reserve posterior
        if self.reserves_posterior_ is None:
            return np.tile(
                observed_cumulative.values, (num_samples, 1)
            )

        # Ultimate = Observed + IBNR
        reserves_flat = self.reserves_posterior_.stack(sample=["chain", "draw"])

        ultimate = np.zeros((num_samples, n_origins))
        for i, origin in enumerate(sorted(observed_cumulative.index)):
            paid = observed_cumulative[origin]
            if origin in reserves_flat.coords["origin"].values:
                ibnr = reserves_flat.sel(origin=origin).values
                ultimate[:, i] = paid + ibnr
            else:
                ultimate[:, i] = paid

        return ultimate

    def compute_best_estimate(
        self,
        fixed_rate: float = 0.04,
    ) -> np.ndarray:
        """
        Compute best estimate (present value of expected future payouts).

        Parameters
        ----------
        fixed_rate : float, optional
            Risk-free discount rate. Default is 0.04.

        Returns
        -------
        np.ndarray
            Best estimate for each posterior sample, shape (num_samples,).
        """
        self._check_is_fitted()

        dims = self.get_model_dimensions()
        num_samples = dims["num_samples"]
        n_dev = dims["n_dev"]

        if self.reserves_posterior_ is None or len(self.future_data_) == 0:
            return np.zeros(num_samples)

        # Get mean predictions for future cells
        response_name = self.model_.response_component.response.name
        mean_name = f"{response_name}_mean"
        if mean_name not in self.idata.posterior:
            mean_name = "mu"

        future_means = self.idata.posterior[mean_name]
        future_means_flat = future_means.stack(sample=["chain", "draw"])

        n_future = len(self.future_data_)
        n_total = future_means_flat.sizes.get(
            future_means_flat.dims[0], n_future
        )
        future_start = n_total - n_future

        best_estimates = np.zeros(num_samples)

        # Discount each future cell by calendar year
        for idx, row in self.future_data_.reset_index(drop=True).iterrows():
            origin = int(row["origin"])
            dev = int(row["dev"])

            # Calendar year determines discount period
            # Assuming origin 1 is the oldest year
            calendar_year = origin + dev - 1
            latest_observed = n_dev  # Approximation

            future_cy = calendar_year - latest_observed
            if future_cy > 0:
                discount = (1 + fixed_rate) ** (future_cy - 0.5)
                obs_idx = future_start + idx
                expected_payout = future_means_flat.isel(
                    {future_means_flat.dims[0]: obs_idx}
                ).values
                best_estimates += expected_payout / discount

        return best_estimates

    def compute_log_likelihood(
        self,
        simulated_losses: np.ndarray,
        mu_p: np.ndarray,
        calendar_year: int,
    ) -> np.ndarray:
        """
        Compute log likelihood of simulated losses for a calendar year.

        Used for Bayesian updating in risk margin calculations.
        Uses the actual family from the fitted model (NB, Poisson, Gamma, Gaussian).

        Parameters
        ----------
        simulated_losses : np.ndarray
            Simulated losses for one scenario, shape (n_origins, n_dev).
        mu_p : np.ndarray
            Expected losses, shape (num_samples, n_origins, n_dev).
        calendar_year : int
            Calendar year (1-indexed) being observed.

        Returns
        -------
        np.ndarray
            Log likelihoods for each posterior sample, shape (num_samples,).
        """
        self._check_is_fitted()
        from scipy import stats

        dims = self.get_model_dimensions()
        num_samples = dims["num_samples"]
        n_origins = dims["n_origins"]
        n_dev = dims["n_dev"]

        ll = np.zeros(num_samples)

        # Get family-specific dispersion parameters from posterior
        family_lower = self.family.lower()
        posterior = self.idata.posterior

        # Extract dispersion/scale parameters based on family
        if family_lower in ("negativebinomial", "negative_binomial", "negbinom"):
            # Bambi stores NB dispersion as 'alpha' or '{response}_alpha'
            response_name = self.model_.response_component.response.name
            alpha_name = f"{response_name}_alpha"
            if alpha_name in posterior:
                alpha = posterior[alpha_name].values.flatten()
            elif "alpha" in posterior:
                alpha = posterior["alpha"].values.flatten()
            else:
                # Fall back to a default if not found
                alpha = np.ones(num_samples) * 1.0

        elif family_lower in ("gamma",):
            # Gamma shape parameter
            response_name = self.model_.response_component.response.name
            alpha_name = f"{response_name}_alpha"
            if alpha_name in posterior:
                shape = posterior[alpha_name].values.flatten()
            elif "alpha" in posterior:
                shape = posterior["alpha"].values.flatten()
            else:
                shape = np.ones(num_samples) * 1.0

        elif family_lower in ("gaussian", "normal"):
            # Gaussian sigma
            response_name = self.model_.response_component.response.name
            sigma_name = f"{response_name}_sigma"
            if sigma_name in posterior:
                sigma = posterior[sigma_name].values.flatten()
            elif "sigma" in posterior:
                sigma = posterior["sigma"].values.flatten()
            else:
                sigma = np.ones(num_samples) * 1.0

        # Compute log-likelihood for each cell in the calendar year diagonal
        for w in range(calendar_year, n_origins):
            d = n_dev - 1 + calendar_year - w
            if 0 <= d < n_dev:
                observed = simulated_losses[w, d]
                expected = mu_p[:, w, d]

                with np.errstate(divide="ignore", invalid="ignore"):
                    if family_lower in ("negativebinomial", "negative_binomial", "negbinom"):
                        # Negative binomial: parameterized by n (alpha) and p
                        # mean = n * (1-p) / p, so p = alpha / (alpha + mu)
                        p = alpha / (alpha + expected)
                        ll += stats.nbinom.logpmf(int(max(0, observed)), n=alpha, p=p)

                    elif family_lower == "poisson":
                        ll += stats.poisson.logpmf(int(max(0, observed)), expected)

                    elif family_lower == "gamma":
                        # Gamma: shape=alpha, scale=mu/alpha
                        if observed > 0:
                            scale = expected / shape
                            ll += stats.gamma.logpdf(observed, a=shape, scale=scale)
                        else:
                            ll += -1e10  # Gamma doesn't support zero

                    elif family_lower in ("gaussian", "normal"):
                        ll += stats.norm.logpdf(observed, expected, sigma)

                    else:
                        # Default to Poisson for unknown families
                        ll += stats.poisson.logpmf(int(max(0, observed)), expected)

                    ll = np.nan_to_num(ll, nan=-1e10, neginf=-1e10)

        return ll

    def __repr__(self) -> str:
        fitted_str = "fitted" if self._is_fitted else "not fitted"
        return (
            f"BayesianChainLadderGLM(\n"
            f"    formula='{self.formula}',\n"
            f"    family='{self.family}',\n"
            f"    draws={self.draws},\n"
            f"    tune={self.tune},\n"
            f"    status={fitted_str}\n"
            f")"
        )


class BayesianCSR:
    """
    Bayesian Changing Settlement Rate (CSR) model for stochastic loss reserving.

    This estimator implements the CSR stochastic loss reserving method from
    Glenn Meyers' "Stochastic Loss Reserving Using Bayesian MCMC Models" (2015).
    The CSR model allows for changing development patterns over time, where newer
    accident years may settle at different rates than older ones.

    The model works on cumulative paid loss triangles with premium as an offset.

    Model Structure
    ---------------
    The mean structure is:

        E[log(loss)] = log(premium) + logelr + alpha[origin] + beta[dev] * speedup[origin]

    where:
    - premium is earned premium (exposure)
    - logelr is the log expected loss ratio
    - alpha[origin] are origin year effects (first constrained to 0)
    - beta[dev] are development effects (last constrained to 0)
    - speedup[origin] is a cumulative factor: speedup[1]=1, speedup[i]=speedup[i-1]*(1-gamma)

    The variance structure allows for heteroscedasticity across development periods,
    with variance typically decreasing as claims mature.

    Parameters
    ----------
    priors : dict, optional
        Dictionary of prior specifications for model parameters. Keys can include:
        - "alpha": dict with "sigma" for origin effects prior
        - "beta": dict with "sigma" for development effects prior
        - "logelr": dict with "mu" and "sigma" for log ELR prior
        - "gamma": dict with "mu" and "sigma" for speedup parameter prior
        - "a_ig": dict with "alpha" and "beta" for inverse gamma prior on variance
    draws : int, optional
        Number of posterior samples per chain. Default is 2000.
    tune : int, optional
        Number of tuning samples. Default is 1000.
    chains : int, optional
        Number of MCMC chains. Default is 4.
    target_accept : float, optional
        Target acceptance probability for NUTS sampler. Default is 0.9.
    random_seed : int, optional
        Random seed for reproducibility.
    include_process_variance : bool, optional
        Whether to include process variance in reserve predictions.
        If True (default), samples from the full posterior predictive
        distribution (including both parameter and process uncertainty).
        If False, uses only parameter uncertainty (mean prediction on
        original scale using the lognormal correction exp(mu + sigma²/2)).

    Attributes
    ----------
    model_ : pm.Model
        The fitted PyMC model.
    idata : az.InferenceData
        ArviZ InferenceData object with posterior samples and predictions.
    data_ : pd.DataFrame
        The observed data in long format.
    future_data_ : pd.DataFrame
        The future/prediction data in long format.
    ultimate_ : pd.DataFrame
        Posterior summary of ultimate losses by origin.
    ibnr_ : pd.DataFrame
        Posterior summary of IBNR reserves by origin.
    reserves_posterior_ : xr.DataArray
        Full posterior samples of reserves by origin.
    elr_posterior_ : xr.DataArray
        Full posterior samples of expected loss ratio.
    gamma_posterior_ : xr.DataArray
        Full posterior samples of speedup parameter.

    Examples
    --------
    >>> import chainladder as cl
    >>> from bayesianchainladder import BayesianCSR
    >>>
    >>> # Load sample triangle
    >>> tri = cl.load_sample("GenIns")
    >>>
    >>> # Fit CSR model
    >>> model = BayesianCSR(
    ...     draws=1000,
    ...     tune=500,
    ... )
    >>> model.fit(tri, premium_value=10000)
    >>>
    >>> # Get reserve summary
    >>> print(model.summary())

    References
    ----------
    Meyers, G. (2015). Stochastic Loss Reserving Using Bayesian MCMC Models.
    CAS Monograph Series Number 1.

    See Also
    --------
    BayesianChainLadderGLM : Cross-classified chain ladder using Bambi/GLM framework
    """

    def __init__(
        self,
        priors: dict[str, Any] | None = None,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int | None = None,
        include_process_variance: bool = True,
    ):
        self.priors = priors
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept
        self.random_seed = random_seed
        self.include_process_variance = include_process_variance

        # Fitted attributes (set by fit())
        self.model_: pm.Model | None = None
        self.idata: az.InferenceData | None = None
        self.data_: pd.DataFrame | None = None
        self.future_data_: pd.DataFrame | None = None
        self.triangle_: cl.Triangle | None = None
        self.ultimate_: pd.DataFrame | None = None
        self.ibnr_: pd.DataFrame | None = None
        self.reserves_posterior_: xr.DataArray | None = None
        self.elr_posterior_: xr.DataArray | None = None
        self.gamma_posterior_: xr.DataArray | None = None
        self._is_fitted: bool = False

    def fit(
        self,
        triangle: cl.Triangle,
        premium_triangle: cl.Triangle | None = None,
        premium_value: float | None = None,
    ) -> "BayesianCSR":
        """
        Fit the Bayesian CSR model to a triangle.

        Parameters
        ----------
        triangle : chainladder.Triangle
            The claims triangle (cumulative paid loss). If incremental, will be
            converted to cumulative.
        premium_triangle : chainladder.Triangle, optional
            Premium triangle (earned premium by origin). If provided, premium values
            are extracted and matched to each origin year.
        premium_value : float, optional
            Single premium value to use for all origin years (if premium_triangle
            is not provided). Required if premium_triangle is None.

        Returns
        -------
        self
            The fitted estimator.

        Raises
        ------
        ValueError
            If neither premium_triangle nor premium_value is provided.
        """
        # Validate input
        validate_triangle(triangle)

        self.triangle_ = triangle.copy()

        # Prepare data for CSR model (cumulative loss, log transforms)
        self.data_, self.future_data_ = prepare_csr_data(
            triangle,
            premium_triangle=premium_triangle,
            premium_value=premium_value,
        )

        # Build the PyMC model
        self.model_ = build_csr_model(
            data=self.data_,
            logprem_col="logprem",
            logloss_col="logloss",
            origin_col="origin",
            dev_col="dev",
            priors=self.priors,
        )

        # Fit the model
        self.idata = fit_model(
            self.model_,
            draws=self.draws,
            tune=self.tune,
            chains=self.chains,
            target_accept=self.target_accept,
            random_seed=self.random_seed,
        )

        # Extract key posteriors
        self.elr_posterior_ = np.exp(self.idata.posterior["logelr"])
        self.gamma_posterior_ = self.idata.posterior["gamma"]

        # Compute reserve predictions
        self._compute_predictions()

        self._is_fitted = True
        return self

    def _compute_predictions(self) -> None:
        """Compute reserve predictions from the fitted model.

        The CSR model predicts log(cumulative loss) ~ Normal(mu, sigma).
        For reserve estimation:
        - loss = exp(logloss) follows a lognormal distribution
        - E[loss] = exp(mu + sigma²/2) for the expected (mean) cumulative loss
        - For posterior predictive: sample logloss ~ Normal(mu, sigma), then exp()

        We compute the posterior predictive distribution by sampling from
        Normal(mu, sigma) for each posterior parameter sample, which gives
        the full distribution including both parameter and process uncertainty.

        Note: For fully developed origins (no future cells), Ultimate = Paid,
        StdErr = 0, and IBNR = 0 with no uncertainty.
        """
        # Get posterior samples of model parameters
        posterior = self.idata.posterior

        # Extract parameter arrays
        alpha = posterior["alpha"].values  # (chains, draws, n_origin)
        beta = posterior["beta"].values  # (chains, draws, n_dev)
        speedup = posterior["speedup"].values  # (chains, draws, n_origin)
        logelr = posterior["logelr"].values  # (chains, draws)
        sig = posterior["sig"].values  # (chains, draws, n_dev)

        n_chains, n_draws = alpha.shape[:2]
        n_samples = n_chains * n_draws

        # Get coordinate mappings
        origin_levels = list(self.model_.coords["origin"])
        dev_levels = list(self.model_.coords["dev"])

        # The ultimate development period is the maximum dev in the triangle
        ultimate_dev = max(dev_levels)
        ultimate_dev_idx = dev_levels.index(ultimate_dev)

        # Get origins with future cells
        origins_with_future = set()
        if len(self.future_data_) > 0:
            origins_with_future = set(self.future_data_["origin"].unique())

        # All origins from observed data
        all_origins = sorted(self.data_["origin"].unique())

        # For each origin, compute predicted cumulative loss
        all_predictions = {}

        # Set random seed for reproducibility if provided
        if self.random_seed is not None:
            np.random.seed(self.random_seed + 1000)  # Offset to differ from sampling

        for origin in all_origins:
            origin_idx = origin_levels.index(origin)

            # Get observed data for this origin
            origin_observed = self.data_[self.data_["origin"] == origin]
            if len(origin_observed) == 0:
                continue

            logprem = origin_observed["logprem"].iloc[0]

            # Get the last observed cumulative loss for this origin
            last_observed_dev = origin_observed["dev"].max()
            last_observed_logloss = origin_observed[
                origin_observed["dev"] == last_observed_dev
            ]["logloss"].iloc[0]
            last_observed_cumulative = np.exp(last_observed_logloss)

            if origin in origins_with_future:
                # Origin has future cells - compute posterior predictive
                # Compute mu at ultimate development period
                # mu = logprem + logelr + alpha[origin] + beta[ultimate_dev] * speedup[origin]
                mu_ultimate = (
                    logprem
                    + logelr
                    + alpha[:, :, origin_idx]
                    + beta[:, :, ultimate_dev_idx] * speedup[:, :, origin_idx]
                )

                # Get sigma at ultimate development period
                sig_ultimate = sig[:, :, ultimate_dev_idx]

                # Generate predictions for ultimate cumulative loss
                if self.include_process_variance:
                    # Sample from Normal(mu, sigma) and exponentiate for lognormal
                    # This includes both parameter uncertainty and process variance
                    logloss_samples = mu_ultimate + sig_ultimate * np.random.standard_normal(
                        mu_ultimate.shape
                    )
                    ultimate_cumulative = np.exp(logloss_samples)
                else:
                    # Use expected value without process variance
                    # For lognormal: E[exp(X)] = exp(mu + sigma²/2)
                    ultimate_cumulative = np.exp(mu_ultimate + 0.5 * sig_ultimate**2)

                # IBNR = Ultimate - Paid to date
                ibnr = ultimate_cumulative - last_observed_cumulative

                all_predictions[origin] = {
                    "paid_to_date": last_observed_cumulative,
                    "ultimate_samples": ultimate_cumulative,
                    "ibnr_samples": ibnr,
                }
            else:
                # Fully developed origin - Ultimate = Paid, no uncertainty
                # Create constant arrays for consistency in downstream processing
                ultimate_cumulative = np.full(n_samples, last_observed_cumulative)
                ibnr = np.zeros(n_samples)

                all_predictions[origin] = {
                    "paid_to_date": last_observed_cumulative,
                    "ultimate_samples": ultimate_cumulative,
                    "ibnr_samples": ibnr,
                }

        # Create reserve summaries
        if all_predictions:
            self._compute_reserve_summaries(all_predictions)

    def _compute_reserve_summaries(
        self, future_predictions: dict[Any, dict[str, np.ndarray]]
    ) -> None:
        """Compute summary statistics for reserves."""
        origins = sorted(future_predictions.keys())

        summary_data = []
        ibnr_samples_list = []

        for origin in origins:
            pred = future_predictions[origin]
            paid = pred["paid_to_date"]
            ultimate_samples = pred["ultimate_samples"].flatten()
            ibnr_samples = pred["ibnr_samples"].flatten()

            ibnr_samples_list.append(ibnr_samples)

            ibnr_mean = float(np.mean(ibnr_samples))
            ibnr_std = float(np.std(ibnr_samples))
            ibnr_median = float(np.median(ibnr_samples))
            ibnr_q05 = float(np.percentile(ibnr_samples, 5))
            ibnr_q25 = float(np.percentile(ibnr_samples, 25))
            ibnr_q75 = float(np.percentile(ibnr_samples, 75))
            ibnr_q95 = float(np.percentile(ibnr_samples, 95))

            ultimate_mean = float(np.mean(ultimate_samples))
            ultimate_std = float(np.std(ultimate_samples))
            ultimate_median = float(np.median(ultimate_samples))
            ultimate_q05 = float(np.percentile(ultimate_samples, 5))
            ultimate_q25 = float(np.percentile(ultimate_samples, 25))
            ultimate_q75 = float(np.percentile(ultimate_samples, 75))
            ultimate_q95 = float(np.percentile(ultimate_samples, 95))

            summary_data.append(
                {
                    "origin": origin,
                    "paid_to_date": paid,
                    "ibnr_mean": ibnr_mean,
                    "ibnr_std": ibnr_std,
                    "ibnr_median": ibnr_median,
                    "ibnr_5%": ibnr_q05,
                    "ibnr_25%": ibnr_q25,
                    "ibnr_75%": ibnr_q75,
                    "ibnr_95%": ibnr_q95,
                    "ultimate_mean": ultimate_mean,
                    "ultimate_std": ultimate_std,
                    "ultimate_median": ultimate_median,
                    "ultimate_5%": ultimate_q05,
                    "ultimate_25%": ultimate_q25,
                    "ultimate_75%": ultimate_q75,
                    "ultimate_95%": ultimate_q95,
                }
            )

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.set_index("origin")

        # Split into ibnr_ and ultimate_
        self.ibnr_ = summary_df[
            [
                "ibnr_mean",
                "ibnr_std",
                "ibnr_median",
                "ibnr_5%",
                "ibnr_25%",
                "ibnr_75%",
                "ibnr_95%",
            ]
        ].copy()
        self.ibnr_.columns = ["mean", "std", "median", "5%", "25%", "75%", "95%"]

        self.ultimate_ = summary_df[
            [
                "paid_to_date",
                "ultimate_mean",
                "ultimate_std",
                "ultimate_median",
                "ultimate_5%",
                "ultimate_25%",
                "ultimate_75%",
                "ultimate_95%",
            ]
        ].copy()
        self.ultimate_.columns = [
            "paid_to_date",
            "mean",
            "std",
            "median",
            "5%",
            "25%",
            "75%",
            "95%",
        ]

        # Create reserves posterior DataArray
        ibnr_array = np.stack(ibnr_samples_list, axis=0)
        n_origins, n_samples = ibnr_array.shape

        # Reshape to (n_origins, n_chains * n_draws) for consistency
        self.reserves_posterior_ = xr.DataArray(
            ibnr_array,
            dims=["origin", "sample"],
            coords={
                "origin": origins,
                "sample": np.arange(n_samples),
            },
        )

    def summary(
        self,
        include_totals: bool = True,
    ) -> pd.DataFrame:
        """
        Return summary table of reserves and ultimates.

        Parameters
        ----------
        include_totals : bool, optional
            Whether to include total row. Default is True.

        Returns
        -------
        pd.DataFrame
            Summary table with reserve statistics by origin.
        """
        self._check_is_fitted()

        if self.ultimate_ is None:
            raise ValueError("No reserve summary available. Model may not have future cells.")

        result = pd.concat(
            [
                self.ultimate_[["paid_to_date", "mean", "std", "median"]],
                self.ibnr_[["mean", "std", "median"]],
            ],
            axis=1,
            keys=["Ultimate", "IBNR"],
        )

        if include_totals:
            # Compute total reserves
            total_paid = self.ultimate_["paid_to_date"].sum()

            # Get total reserve distribution
            total_reserves = self.reserves_posterior_.sum(dim="origin")

            total_ibnr_mean = float(total_reserves.mean())
            total_ibnr_std = float(total_reserves.std())
            total_ibnr_median = float(np.median(total_reserves.values))

            total_ult_mean = total_paid + total_ibnr_mean
            total_ult_std = total_ibnr_std
            total_ult_median = total_paid + total_ibnr_median

            total_row = pd.DataFrame(
                {
                    ("Ultimate", "paid_to_date"): [total_paid],
                    ("Ultimate", "mean"): [total_ult_mean],
                    ("Ultimate", "std"): [total_ult_std],
                    ("Ultimate", "median"): [total_ult_median],
                    ("IBNR", "mean"): [total_ibnr_mean],
                    ("IBNR", "std"): [total_ibnr_std],
                    ("IBNR", "median"): [total_ibnr_median],
                },
                index=["Total"],
            )

            result = pd.concat([result, total_row])

        return result

    def get_parameter_summary(
        self,
        var_names: list[str] | None = None,
        filter_vars: str | None = None,
        hdi_prob: float = 0.94,
    ) -> pd.DataFrame:
        """
        Get summary statistics for model parameters.

        Parameters
        ----------
        var_names : list[str], optional
            Parameter names to include. If None, includes all.
        filter_vars : str, optional
            Filter for variable names (e.g., "like" or "regex").
        hdi_prob : float, optional
            Probability mass for HDI. Default is 0.94.

        Returns
        -------
        pd.DataFrame
            Parameter summary table.
        """
        self._check_is_fitted()
        return az.summary(
            self.idata, var_names=var_names, filter_vars=filter_vars, hdi_prob=hdi_prob
        )

    def get_expected_loss_ratio(self) -> pd.DataFrame:
        """
        Get posterior summary of the expected loss ratio.

        Returns
        -------
        pd.DataFrame
            Summary statistics for the expected loss ratio.
        """
        self._check_is_fitted()

        elr_flat = self.elr_posterior_.values.flatten()

        return pd.DataFrame(
            {
                "mean": [np.mean(elr_flat)],
                "std": [np.std(elr_flat)],
                "median": [np.median(elr_flat)],
                "5%": [np.percentile(elr_flat, 5)],
                "95%": [np.percentile(elr_flat, 95)],
            },
            index=["ELR"],
        )

    def get_speedup_parameter(self) -> pd.DataFrame:
        """
        Get posterior summary of the speedup (gamma) parameter.

        The gamma parameter controls how quickly the settlement pattern
        changes across accident years. gamma > 0 means faster settlement
        for newer years.

        Returns
        -------
        pd.DataFrame
            Summary statistics for the gamma parameter.
        """
        self._check_is_fitted()

        gamma_flat = self.gamma_posterior_.values.flatten()

        return pd.DataFrame(
            {
                "mean": [np.mean(gamma_flat)],
                "std": [np.std(gamma_flat)],
                "median": [np.median(gamma_flat)],
                "5%": [np.percentile(gamma_flat, 5)],
                "95%": [np.percentile(gamma_flat, 95)],
            },
            index=["gamma"],
        )

    def sample_reserves(
        self,
        n_samples: int = 1000,
        random_seed: int | None = None,
    ) -> np.ndarray:
        """
        Draw samples from the reserve distribution.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to draw. Default is 1000.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Array of total reserve samples (shape: n_samples).
        """
        self._check_is_fitted()

        if self.reserves_posterior_ is None:
            raise ValueError("No reserve posterior available")

        # Get total reserves
        total_reserves = self.reserves_posterior_.sum(dim="origin").values

        if random_seed is not None:
            np.random.seed(random_seed)

        # Sample with replacement if needed
        if n_samples <= len(total_reserves):
            indices = np.random.choice(len(total_reserves), size=n_samples, replace=False)
        else:
            indices = np.random.choice(len(total_reserves), size=n_samples, replace=True)

        return total_reserves[indices]

    # =========================================================================
    # Risk Margin Support Methods
    # =========================================================================

    def get_risk_margin_params(self) -> dict[str, np.ndarray]:
        """
        Extract model parameters needed for risk margin calculations.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing flattened posterior samples:
            - alpha: Origin effects, shape (num_samples, n_origins)
            - beta: Development effects, shape (num_samples, n_dev)
            - speedup: Speedup factors, shape (num_samples, n_origins)
            - logelr: Log expected loss ratio, shape (num_samples,)
            - sig: Standard deviations by dev, shape (num_samples, n_dev)
            - gamma: Speedup parameter, shape (num_samples,)
        """
        self._check_is_fitted()

        posterior = self.idata.posterior
        n_chains = posterior.dims["chain"]
        n_draws = posterior.dims["draw"]
        num_samples = n_chains * n_draws

        # Extract and flatten (chain, draw) -> (sample,)
        alpha = posterior["alpha"].values.reshape(num_samples, -1)
        beta = posterior["beta"].values.reshape(num_samples, -1)
        speedup = posterior["speedup"].values.reshape(num_samples, -1)
        logelr = posterior["logelr"].values.flatten()
        sig = posterior["sig"].values.reshape(num_samples, -1)

        if "gamma" in posterior:
            gamma = posterior["gamma"].values.flatten()
        else:
            gamma = np.zeros(num_samples)

        return {
            "alpha": alpha,
            "beta": beta,
            "speedup": speedup,
            "logelr": logelr,
            "sig": sig,
            "gamma": gamma,
        }

    def get_paid_triangle(self) -> np.ndarray:
        """
        Get the observed cumulative paid loss triangle.

        Returns
        -------
        np.ndarray
            Cumulative paid losses, shape (n_origins, n_dev).
            NaN values indicate unobserved (future) cells.
        """
        self._check_is_fitted()

        # Pivot the observed data to triangle format
        pivot = self.data_.pivot(index="origin", columns="dev", values="cumulative")
        return pivot.values

    def get_log_premium(self) -> np.ndarray:
        """
        Get log premium values by origin.

        Returns
        -------
        np.ndarray
            Log premium for each origin, shape (n_origins,).
        """
        self._check_is_fitted()
        return self.data_.groupby("origin")["logprem"].first().values

    def get_premium(self) -> np.ndarray:
        """
        Get premium values by origin.

        Returns
        -------
        np.ndarray
            Premium for each origin, shape (n_origins,).
        """
        self._check_is_fitted()
        return self.data_.groupby("origin")["premium"].first().values

    def get_model_dimensions(self) -> dict[str, int]:
        """
        Get model dimensions for risk margin calculations.

        Returns
        -------
        dict[str, int]
            Dictionary with:
            - n_origins: Number of origin years
            - n_dev: Number of development periods
            - num_samples: Total number of MCMC samples
        """
        self._check_is_fitted()

        origins = sorted(self.data_["origin"].unique())
        dev_periods = sorted(self.data_["dev"].unique())

        posterior = self.idata.posterior
        n_chains = posterior.dims["chain"]
        n_draws = posterior.dims["draw"]

        return {
            "n_origins": len(origins),
            "n_dev": len(dev_periods),
            "num_samples": n_chains * n_draws,
        }

    def simulate_future_losses(
        self,
        random_seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate log cumulative losses for future (lower triangle) cells.

        Parameters
        ----------
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (logloss_p, mu_p) - Simulated log losses and expected values,
            each of shape (num_samples, n_origins, n_dev).
        """
        self._check_is_fitted()

        if random_seed is not None:
            np.random.seed(random_seed)

        params = self.get_risk_margin_params()
        dims = self.get_model_dimensions()
        logprem = self.get_log_premium()

        alpha = params["alpha"]
        beta = params["beta"]
        logelr = params["logelr"]
        sig = params["sig"]

        num_samples = dims["num_samples"]
        n_origins = dims["n_origins"]
        n_dev = dims["n_dev"]

        # Initialize arrays
        mu_p = np.zeros((num_samples, n_origins, n_dev))
        logloss_p = np.zeros((num_samples, n_origins, n_dev))

        # Generate simulated losses for future cells (lower triangle)
        # R code: for (d in 2:10) for (w in (12-d):10)
        # In 0-indexed: d = 1 to n_dev-1, w = n_dev-d to n_origins-1
        for d in range(1, n_dev):
            for w in range(n_dev - d, n_origins):
                mu_p[:, w, d] = logprem[w] + logelr + alpha[:, w] + beta[:, d]
                logloss_p[:, w, d] = np.random.normal(mu_p[:, w, d], sig[:, d])

        return logloss_p, mu_p

    def compute_unconditional_ultimate(self) -> np.ndarray:
        """
        Compute unconditional ultimate loss estimates by origin.

        This computes E[Ultimate | parameters] for each posterior sample,
        without conditioning on future observed data.

        Returns
        -------
        np.ndarray
            Ultimate estimates, shape (num_samples, n_origins).
        """
        self._check_is_fitted()

        params = self.get_risk_margin_params()
        dims = self.get_model_dimensions()
        trpaid = self.get_paid_triangle()
        premium = self.get_premium()

        alpha = params["alpha"]
        logelr = params["logelr"]
        sig = params["sig"]

        num_samples = dims["num_samples"]
        n_origins = dims["n_origins"]
        n_dev = dims["n_dev"]

        # Ultimate development index (last column)
        d_ult = n_dev - 1

        mean_ult = np.zeros((num_samples, n_origins))

        # First origin is fully developed - use observed value
        mean_ult[:, 0] = trpaid[0, d_ult]

        # Other origins need projection
        # E[loss] = premium * ELR * exp(alpha) * exp(sig^2/2)
        for w in range(1, n_origins):
            mean_ult[:, w] = premium[w] * np.exp(
                logelr + alpha[:, w] + sig[:, d_ult] ** 2 / 2
            )

        return mean_ult

    def compute_best_estimate(
        self,
        fixed_rate: float = 0.04,
    ) -> np.ndarray:
        """
        Compute best estimate (present value of expected future payouts).

        Parameters
        ----------
        fixed_rate : float, optional
            Risk-free discount rate. Default is 0.04.

        Returns
        -------
        np.ndarray
            Best estimate for each posterior sample, shape (num_samples,).
        """
        self._check_is_fitted()

        params = self.get_risk_margin_params()
        dims = self.get_model_dimensions()
        trpaid = self.get_paid_triangle()
        logprem = self.get_log_premium()

        alpha = params["alpha"]
        beta = params["beta"]
        speedup = params["speedup"]
        logelr = params["logelr"]
        sig = params["sig"]

        num_samples = dims["num_samples"]
        n_origins = dims["n_origins"]
        n_dev = dims["n_dev"]

        best_estimates = np.zeros(num_samples)

        for i in range(num_samples):
            tr = trpaid.copy()
            pv = 0.0

            # Fill in future cells with expected values
            for cy in range(1, n_dev):
                for w in range(cy, n_origins):
                    d = n_dev - 1 + cy - w

                    if d < n_dev and w < n_origins:
                        mu = (
                            logprem[w]
                            + logelr[i]
                            + alpha[i, w]
                            + beta[i, d] * speedup[i, w]
                        )
                        expected = np.exp(mu + sig[i, d] ** 2 / 2)
                        tr[w, d] = expected

            # Compute present value of incremental payouts
            for cy in range(1, n_dev):
                for w in range(cy, n_origins):
                    d = n_dev - 1 + cy - w
                    if d > 0 and d < n_dev:
                        incremental = tr[w, d] - tr[w, d - 1]
                        discount = (1 + fixed_rate) ** (cy - 0.5)
                        pv += incremental / discount

            best_estimates[i] = pv

        return best_estimates

    def compute_log_likelihood(
        self,
        simulated_losses: np.ndarray,
        mu_p: np.ndarray,
        calendar_year: int,
    ) -> np.ndarray:
        """
        Compute log likelihood of simulated losses for a calendar year.

        Used for Bayesian updating in risk margin calculations.

        Parameters
        ----------
        simulated_losses : np.ndarray
            Simulated log losses for one scenario, shape (n_origins, n_dev).
        mu_p : np.ndarray
            Expected log losses, shape (num_samples, n_origins, n_dev).
        calendar_year : int
            Calendar year (1-indexed) being observed.

        Returns
        -------
        np.ndarray
            Log likelihoods for each posterior sample, shape (num_samples,).
        """
        self._check_is_fitted()
        from scipy import stats

        params = self.get_risk_margin_params()
        dims = self.get_model_dimensions()
        sig = params["sig"]

        num_samples = dims["num_samples"]
        n_origins = dims["n_origins"]
        n_dev = dims["n_dev"]

        ll = np.zeros(num_samples)

        for w in range(calendar_year, n_origins):
            d = n_dev - 1 + calendar_year - w
            if d < n_dev:
                x = simulated_losses[w, d]
                mu = mu_p[:, w, d]
                sigma = sig[:, d]
                ll += stats.norm.logpdf(x, mu, sigma)

        return ll

    def _check_is_fitted(self) -> None:
        """Check if the model has been fitted."""
        if not self._is_fitted:
            raise ValueError(
                "Model has not been fitted. Call fit() before using this method."
            )

    def __repr__(self) -> str:
        fitted_str = "fitted" if self._is_fitted else "not fitted"
        return (
            f"BayesianCSR(\n"
            f"    draws={self.draws},\n"
            f"    tune={self.tune},\n"
            f"    chains={self.chains},\n"
            f"    include_process_variance={self.include_process_variance},\n"
            f"    status={fitted_str}\n"
            f")"
        )
