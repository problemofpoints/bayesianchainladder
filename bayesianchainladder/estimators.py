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
import xarray as xr

from .models import build_bambi_model, fit_model
from .utils import (
    add_categorical_columns,
    get_future_dataframe,
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

        # Add categorical encoding
        self.data_ = add_categorical_columns(self.data_)
        self.future_data_ = add_categorical_columns(self.future_data_)

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
