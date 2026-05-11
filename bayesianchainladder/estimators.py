"""
High-level estimator classes for Bayesian chain ladder.

This module provides scikit-learn/chainladder-style estimator classes
for Bayesian stochastic loss reserving.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from .base import BaseStochasticReserve
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


class BayesianChainLadderGLM(BaseStochasticReserve):
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
    response_per_exposure : bool, optional
        If True, divide the response column by the exposure column at fit time
        and model the resulting loss-ratio-incremental as the response. This is
        the recommended approach for identity-link or t families that allow
        negative responses. When True, no log offset is added — the model is on
        loss-ratio scale directly. Caller must still supply ``exposure=...``.
        Default is False.
    include_process_variance : bool, optional
        Whether to include process variance in reserve predictions.
        If True (default), samples from the full posterior predictive
        distribution (``kind="response"``) for future cells, adding draws from
        the response distribution (e.g. Gamma(α, μ/α)) on top of the
        parameter uncertainty in μ.  This gives statistically correct
        prediction intervals.
        If False, uses only parameter uncertainty: the posterior of the
        conditional mean μ is summed across future cells without any
        within-cell sampling noise.
        Default is True.

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
        response_per_exposure: bool = False,
        include_process_variance: bool = True,
        init_priors_from_chainladder: bool = False,
        chainladder_prior_sd: float = 0.5,
    ):
        super().__init__()
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
        self.response_per_exposure = response_per_exposure
        self.include_process_variance = include_process_variance
        self.init_priors_from_chainladder = init_priors_from_chainladder
        self.chainladder_prior_sd = chainladder_prior_sd

        # GLM-specific fitted attributes (not in base)
        self.model_: bmb.Model | None = None
        self.idata: az.InferenceData | None = None
        self.data_: pd.DataFrame | None = None
        self.future_data_: pd.DataFrame | None = None
        self.fitted_: pd.DataFrame | None = None
        self._original_exposure_col: str | None = None

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

        # Align categorical levels and drop future rows with unseen levels.
        #
        # When a triangle has fewer observed dev periods than the full 10-period
        # range, future_data_ may contain dev values (e.g., 96, 108, 120) that
        # never appeared in data_.  Bambi's C(dev) term would raise:
        #   ValueError: The levels (120, 108) in 'C(dev)' are not present in
        #   the original data set.
        # Fix: for columns wrapped in C(...) in the formula (i.e., the only
        # columns Bambi validates for categorical levels), restrict future_data_
        # to levels seen in training.  Dropped future cells are those where the
        # youngest origins require dev periods beyond the training horizon; those
        # origins will receive a *partial* IBNR estimate (up to the last observed
        # dev level), which slightly understates their reserves but avoids a crash.
        if len(self.future_data_) > 0:
            # Identify columns that appear as C(col) in the formula
            c_wrapped_cols = set(re.findall(r'\bC\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)', self.formula))
            drop_mask = pd.Series(False, index=self.future_data_.index)
            for col in c_wrapped_cols:
                if col not in self.future_data_.columns or col not in self.data_.columns:
                    continue
                train_levels = set(
                    self.data_[col].cat.categories
                    if hasattr(self.data_[col], "cat")
                    else self.data_[col].unique()
                )
                unseen = set(self.future_data_[col].unique()) - train_levels
                if unseen:
                    drop_mask |= self.future_data_[col].isin(unseen)
            if drop_mask.any():
                n_dropped = int(drop_mask.sum())
                import warnings
                warnings.warn(
                    f"BayesianChainLadderGLM: dropped {n_dropped} future cell(s) "
                    f"whose categorical levels were not observed in training data. "
                    f"This typically affects the youngest origins at late dev periods. "
                    f"IBNR for those origins will be a partial (lower-bound) estimate.",
                    UserWarning,
                    stacklevel=3,
                )
                self.future_data_ = self.future_data_[~drop_mask].reset_index(drop=True)

        # If response_per_exposure=True, divide the response by exposure in the
        # observed data so the model is on loss-ratio scale rather than dollar scale.
        # No log offset is appended in this mode.
        if self.response_per_exposure:
            if not self.exposure:
                raise ValueError(
                    "response_per_exposure=True requires exposure= to be set."
                )
            self._original_exposure_col = self.exposure
            response_col = self.formula.split("~")[0].strip()
            if response_col not in self.data_.columns:
                raise ValueError(
                    f"Response column '{response_col}' (LHS of formula) not found in data."
                )
            exp_vals = np.asarray(self.data_[self.exposure].values, dtype=np.float64)
            self.data_ = self.data_.copy()
            response_vals = np.asarray(self.data_[response_col].values, dtype=np.float64)
            self.data_[response_col] = response_vals / exp_vals
            # Clear the exposure so build_bambi_model does not append a log offset.
            self.exposure = None

        # Validate data compatibility with chosen family
        self._validate_data_family_compatibility()

        priors = self._build_default_priors()

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

        # Restore exposure if it was temporarily cleared for response_per_exposure mode.
        if self.response_per_exposure and self._original_exposure_col is not None:
            self.exposure = self._original_exposure_col

        self._is_fitted = True
        return self

    def _compute_predictions(self) -> None:
        """Compute fitted values and future predictions."""
        # When an exposure offset is used, build_bambi_model adds a `logoffset`
        # column to the training copy of data.  Bambi's predict() needs that
        # same column in whatever DataFrame we pass.  We compute it here so
        # that both observed and future frames have it.
        obs_data = self.data_.copy()
        fut_data = self.future_data_.copy() if len(self.future_data_) > 0 else self.future_data_
        if self.exposure and self.exposure in obs_data.columns:
            obs_data["logoffset"] = np.log(
                np.asarray(obs_data[self.exposure].values, dtype=np.float64)
            )
        if self.exposure and len(fut_data) > 0 and self.exposure in fut_data.columns:
            fut_data["logoffset"] = np.log(
                np.asarray(fut_data[self.exposure].values, dtype=np.float64)
            )

        # Fitted values for observed data
        # Use kind="response_params" as "mean" is deprecated in newer Bambi
        try:
            self.model_.predict(
                self.idata, data=obs_data, kind="response_params", inplace=True
            )
        except (TypeError, ValueError):
            # Fall back to "mean" for older Bambi versions
            self.model_.predict(self.idata, data=obs_data, kind="mean", inplace=True)

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
            if self.include_process_variance:
                # kind="response" samples from the full posterior predictive
                # distribution (e.g. Gamma(α, μ/α) for gamma family), adding
                # process variance on top of parameter uncertainty.
                # On success the draws land in idata.posterior_predictive.
                # Fall back to parameter-only if the call fails.
                _pv_success = False
                try:
                    self.model_.predict(
                        self.idata, data=fut_data, kind="response", inplace=True,
                        sample_new_groups=True,
                    )
                    _pv_success = True
                except Exception:
                    pass

                if _pv_success and hasattr(self.idata, "posterior_predictive"):
                    pp = self.idata.posterior_predictive
                    if response_name in pp:
                        self._compute_reserves(pp[response_name])
                        return
                    # Otherwise fall through to parameter-only path below.

            # Parameter-only path (include_process_variance=False, or fallback).
            try:
                self.model_.predict(
                    self.idata, data=fut_data, kind="response_params", inplace=True,
                    sample_new_groups=True,
                )
            except (TypeError, ValueError):
                self.model_.predict(
                    self.idata, data=fut_data, kind="mean", inplace=True,
                    sample_new_groups=True,
                )

            # Get the future predictions using same name discovery
            if mean_name in self.idata.posterior:
                future_mean = self.idata.posterior[mean_name]
            else:
                future_mean = self.idata.posterior_predictive[mean_name]

            # Compute reserves by origin
            self._compute_reserves(future_mean)

    def _compute_reserves(self, future_predictions: xr.DataArray) -> None:
        """Compute reserve distributions from future predictions.

        When ``response_per_exposure=True`` the model predicts on loss-ratio
        scale (incremental / EP).  This method back-transforms those
        predictions to dollar scale by multiplying each future cell's predicted
        loss ratio by the corresponding EP before summing within each origin.
        """
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

        # When response_per_exposure=True, predictions are on loss-ratio scale.
        # Recover the per-row EP from future_data_ to back-transform to dollars.
        ep_col = self._original_exposure_col if self.response_per_exposure else None
        if ep_col is not None and ep_col not in self.future_data_.columns:
            # Exposure column was renamed or missing — fall back to raw ratio
            ep_col = None

        for origin in origins:
            # Get indices for this origin in future_data_
            mask = self.future_data_["origin"] == origin
            origin_future_idx = np.where(mask)[0]

            if len(origin_future_idx) > 0:
                # Map to positions in the full prediction array
                pos = [future_start + i for i in origin_future_idx]

                # Sum predictions for this origin across all future cells
                origin_preds = future_predictions.isel({obs_dim: pos})

                if ep_col is not None:
                    # Back-transform loss-ratio predictions to dollar scale:
                    # predicted_dollars[cell] = predicted_lr[cell] * EP[cell]
                    ep_vals = np.asarray(
                        self.future_data_.iloc[origin_future_idx][ep_col].values,
                        dtype=np.float64,
                    )
                    # ep_vals has shape (n_cells_for_origin,); broadcast over
                    # the (chain, draw) dims by creating an xr.DataArray.
                    ep_da = xr.DataArray(ep_vals, dims=[obs_dim])
                    origin_preds = origin_preds * ep_da

                reserve_samples[origin] = origin_preds.sum(dim=obs_dim)

        # Create DataArray with reserves by origin
        if reserve_samples:
            reserves_list = []
            for origin in origins:
                if origin in reserve_samples:
                    reserves_list.append(reserve_samples[origin])

            reserves_posterior = xr.concat(
                reserves_list, dim=pd.Index(origins, name="origin")
            )
            # Standardize to (origin, sample) for the base class helper
            self.reserves_posterior_ = (
                reserves_posterior
                .stack(sample=["chain", "draw"])
                .reset_index("sample", drop=True)
            )

            # Compute ultimate and IBNR summaries (helper now lives in the base)
            self._build_reserve_summaries()

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

            # Mirror the _compute_predictions fix: when an exposure offset is
            # configured, Bambi expects a `logoffset` column in whatever DataFrame
            # we pass to predict().  Build it here on a copy so we never mutate
            # the caller's data.
            if self.exposure and self.exposure in combined.columns:
                combined = combined.copy()
                combined["logoffset"] = np.log(
                    np.asarray(combined[self.exposure].values, dtype=np.float64)
                )

            self.model_.predict(self.idata, data=combined, kind=kind, inplace=True)

            pred_mean = self.idata.posterior[mean_name].mean(dim=["chain", "draw"])
            combined["predicted_mean"] = pred_mean.values[-len(combined) :]

            return combined

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

        priors = self._build_default_priors()

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

    def _build_cl_informed_priors(self) -> dict[str, Any]:
        """Build chain-ladder-informed prior centers for GLM formula terms.

        Runs a deterministic ``cl.Chainladder()`` on the training triangle and
        extracts per-origin ultimates and per-dev incremental fractions.  These
        become the prior *centers* (mu) for each formula term; SDs are set to
        ``self.chainladder_prior_sd`` uniformly.

        Supported formula components
        ----------------------------
        * ``C(origin)``   — log-contrast priors relative to reference origin
        * ``C(dev)``      — log-contrast priors relative to reference dev
        * ``bs(dev_idx, df=N)`` — least-squares projection of CL log-incremental
          pattern onto the spline basis (fallback: skip / leave defaults)
        * ``(1 | origin)`` — HalfNormal hyperprior on RE sigma scaled to the
          empirical SD of log-ultimates
        * ``(1 | calendar)`` — not informed from CL; left to defaults

        For identity-link (``response_per_exposure=True``) models the contrasts
        are on loss-ratio scale rather than log scale.

        Returns
        -------
        dict
            Bambi-compatible priors dict with informed entries for recognised
            terms.  Entries for unrecognised / unsupported terms are omitted
            (Bambi will use its own defaults for those).
        """
        import chainladder as cl
        import warnings

        priors: dict[str, Any] = {}
        tri = self.triangle_
        sd = float(self.chainladder_prior_sd)

        # -------------------------------------------------------------------
        # Detect effective link: log vs identity
        # -------------------------------------------------------------------
        from .models import _get_default_link as _gdl
        family_lower = self.family.lower()
        canonical = {
            "t": "t", "student_t": "t", "studentt": "t",
            "gaussian": "gaussian", "normal": "gaussian",
        }.get(family_lower, family_lower)
        effective_link = self.link if self.link else _gdl(canonical)
        is_log_link = (effective_link == "log")
        # If response_per_exposure is True the transformed response is on
        # loss-ratio scale (identity link effective).
        is_identity = not is_log_link

        # -------------------------------------------------------------------
        # Run chain ladder on the training triangle
        # -------------------------------------------------------------------
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cl_fit = cl.Chainladder().fit(tri)
        except Exception as e:
            import warnings as _w
            _w.warn(
                f"BayesianChainLadderGLM: could not fit chain ladder for "
                f"init_priors_from_chainladder ({e}); using default priors.",
                UserWarning,
                stacklevel=5,
            )
            return priors

        # Ultimate and incremental-fraction extraction
        try:
            ult_frame = cl_fit.ultimate_.to_frame()
            ult_arr = np.asarray(ult_frame.values, dtype=float).flatten()
            cdf_frame = cl_fit.cdf_.to_frame()
            cdf_arr = np.asarray(cdf_frame.values, dtype=float).flatten()
        except Exception:
            return priors

        # Number of dev periods in the full triangle
        n_devs = len(tri.development)
        cdf_for_devs = cdf_arr[:n_devs]

        # Cumulative pct at each dev = 1 / CDF (CDF is from that dev to ult).
        # CDF values < 1 mean we're still developing; CDF = 1 means done.
        cum_pct = 1.0 / np.maximum(cdf_for_devs, 1e-8)  # (n_devs,)
        cum_pct = np.clip(cum_pct, 0.0, 1.0)

        # Incremental pct = diff of cumulative pct (first cell = cum_pct[0])
        incr_pct = np.diff(np.concatenate([[0.0], cum_pct]))  # (n_devs,)
        incr_pct = np.maximum(incr_pct, 1e-10)  # guard against ≤ 0

        # Reference cells: first origin, first dev
        ref_ult = float(ult_arr[0]) if ult_arr.size > 0 else 1.0
        ref_ult = max(ref_ult, 1e-8)
        ref_frac = float(incr_pct[0])
        ref_frac = max(ref_frac, 1e-10)

        # -------------------------------------------------------------------
        # Per-origin EP (needed for identity/loss-ratio scale)
        # -------------------------------------------------------------------
        ep_by_origin: dict[int, float] | None = None
        if is_identity and self.data_ is not None:
            # _original_exposure_col holds the exposure col name before it was cleared
            exp_col = self._original_exposure_col if self._original_exposure_col else self.exposure
            if exp_col and exp_col in self.data_.columns:
                ep_by_origin = (
                    self.data_.groupby("origin")[exp_col].first().to_dict()
                )

        # -------------------------------------------------------------------
        # C(origin) priors — log-contrast or linear-contrast
        # -------------------------------------------------------------------
        c_origin_match = re.search(r'\bC\s*\(\s*origin\s*\)', self.formula)
        if c_origin_match and self.data_ is not None and len(ult_arr) >= 2:
            # Number of contrasts = n_origins - 1 (treatment coding, ref=first)
            n_origins = len(ult_arr)
            n_contrasts_origin = n_origins - 1

            if is_log_link:
                # Prior means = log(ult[k] / ult[ref]) for k > 0
                origin_mus = np.log(np.maximum(ult_arr[1:], 1e-8)) - np.log(ref_ult)
            else:
                # Identity link: prior on loss-ratio-scale origin effects
                # LR[k] = ult[k] / EP[k]; contrast mu_k = LR[k]*frac[0] - LR[0]*frac[0]
                if ep_by_origin is not None:
                    origins_sorted = sorted(ep_by_origin.keys())
                    ep_vals = np.array([ep_by_origin.get(o, 1.0) for o in origins_sorted], dtype=float)
                    ep_vals = np.maximum(ep_vals, 1e-8)
                    lr_arr = ult_arr[:len(ep_vals)] / ep_vals
                    ref_lr = float(lr_arr[0]) if lr_arr.size > 0 else 0.05
                    origin_mus = (lr_arr[1:] - ref_lr) * ref_frac
                else:
                    origin_mus = np.zeros(n_contrasts_origin)

            origin_mus = origin_mus[:n_contrasts_origin]
            if len(origin_mus) > 0:
                priors["C(origin)"] = bmb.Prior(
                    "Normal",
                    mu=origin_mus.astype(float),
                    sigma=np.full(len(origin_mus), sd, dtype=float),
                )

        # -------------------------------------------------------------------
        # C(dev) priors — log-contrast or linear-contrast
        # -------------------------------------------------------------------
        c_dev_match = re.search(r'\bC\s*\(\s*dev\s*\)', self.formula)
        if c_dev_match and n_devs >= 2:
            n_contrasts_dev = n_devs - 1

            if is_log_link:
                # Prior means = log(frac[j] / frac[0]) for j > 0
                dev_mus = np.log(np.maximum(incr_pct[1:], 1e-10)) - np.log(ref_frac)
            else:
                # Identity link: incremental LR deviation from reference dev
                # ref origin LR * (frac[j] - frac[0])
                if ep_by_origin is not None:
                    origins_sorted = sorted(ep_by_origin.keys())
                    ep_vals = np.array([ep_by_origin.get(o, 1.0) for o in origins_sorted], dtype=float)
                    ep_vals = np.maximum(ep_vals, 1e-8)
                    lr_ref = float(ult_arr[0]) / float(ep_vals[0])
                    dev_mus = lr_ref * (incr_pct[1:] - ref_frac)
                else:
                    dev_mus = np.zeros(n_contrasts_dev)

            dev_mus = dev_mus[:n_contrasts_dev]
            if len(dev_mus) > 0:
                priors["C(dev)"] = bmb.Prior(
                    "Normal",
                    mu=dev_mus.astype(float),
                    sigma=np.full(len(dev_mus), sd, dtype=float),
                )

        # -------------------------------------------------------------------
        # bs(dev_idx, df=N) spline priors — project CL pattern onto basis
        # -------------------------------------------------------------------
        spline_match = re.search(r'\bbs\s*\(\s*dev_idx\s*,\s*df\s*=\s*(\d+)\s*\)', self.formula)
        if spline_match and self.data_ is not None:
            df_spline = int(spline_match.group(1))
            try:
                # Build the same spline basis that formulae/patsy will build.
                # dev_idx is 1-based integer index: 1, 2, ..., n_devs.
                dev_idx_vals = np.arange(1, n_devs + 1, dtype=float)

                # Build B-spline basis via scipy (same knots as df=N default).
                from scipy.interpolate import BSpline, make_interp_spline
                from scipy.linalg import lstsq as sp_lstsq

                # Build basis manually using patsy-style: evenly-spaced interior
                # knots with cubic (degree=3) B-splines, augmented boundary knots.
                # df = n_interior_knots + degree + 1 - include_intercept
                # For df=4, degree=3: n_interior_knots = 0  (no interior knots)
                degree = 3
                n_interior = df_spline - degree - 1  # = 0 for df=4
                n_interior = max(n_interior, 0)

                x = dev_idx_vals
                x_min, x_max = float(x.min()), float(x.max())
                # Interior knots evenly spaced
                if n_interior > 0:
                    interior_knots = np.linspace(x_min, x_max, n_interior + 2)[1:-1]
                else:
                    interior_knots = np.array([])
                # Full knot vector: boundary repeated (degree+1) times each
                knots = np.concatenate([
                    np.repeat(x_min, degree + 1),
                    interior_knots,
                    np.repeat(x_max, degree + 1),
                ])
                # Build design matrix: one column per basis function
                n_basis = len(knots) - degree - 1
                B = np.zeros((len(x), n_basis))
                for k in range(n_basis):
                    c = np.zeros(n_basis)
                    c[k] = 1.0
                    spl = BSpline(knots, c, degree)
                    B[:, k] = spl(x)

                # If matrix is wrong size, skip
                if B.shape[1] != df_spline:
                    raise ValueError(
                        f"Spline basis shape mismatch: got {B.shape[1]} columns, "
                        f"expected {df_spline}"
                    )

                # Target: log of incremental fractions (CL pattern on log scale)
                if is_log_link:
                    y_target = np.log(np.maximum(incr_pct, 1e-10))
                else:
                    # Identity link: raw incr_pct as target (roughly)
                    y_target = incr_pct.copy()

                # Least-squares fit: B @ coef ≈ y_target
                coef, _, _, _ = np.linalg.lstsq(B, y_target, rcond=None)

                # Sanity check: coefficients should be finite and not huge
                if np.all(np.isfinite(coef)) and np.all(np.abs(coef) < 50.0):
                    spline_key = f"bs(dev_idx, df={df_spline})"
                    priors[spline_key] = bmb.Prior(
                        "Normal",
                        mu=coef.astype(float),
                        sigma=np.full(df_spline, sd, dtype=float),
                    )
                # else: skip spline priors; fall back to Bambi defaults
            except Exception:
                # Spline projection failed; leave defaults (no entry added)
                pass

        # -------------------------------------------------------------------
        # (1 | origin) random intercept — HalfNormal on sigma
        # -------------------------------------------------------------------
        re_origin_match = re.search(r'\(1\s*\|\s*origin\s*\)', self.formula)
        if re_origin_match:
            # Empirical SD of log(ultimates) across origins
            valid_ults = ult_arr[ult_arr > 0]
            if valid_ults.size >= 2:
                log_ult_sd = float(np.std(np.log(valid_ults), ddof=1))
                # Wide enough to let the data determine the RE SD, but informed
                # by the inter-origin variability: use 2× empirical SD.
                hn_sigma = max(2.0 * log_ult_sd, 0.01)
                priors["1|origin"] = bmb.Prior(
                    "Normal",
                    mu=0,
                    sigma=bmb.Prior("HalfNormal", sigma=hn_sigma),
                )

        # (1 | calendar) — not directly informed by chain ladder; leave defaults.

        return priors

    def _build_default_priors(self) -> dict[str, Any]:
        """Build data-adaptive default priors, with user priors layered on top.

        Bambi's auto-priors scale sigma with sd(y), which is appropriate for
        identity-link models but produces sigmas of 6-19+ for one-hot
        categorical effects under a log link. Those wide priors generate
        prior predictive draws of exp(linear_predictor) that overflow the
        valid parameter range of NegativeBinomial / Poisson, so prior
        predictive sampling crashes with "n too large or p too small".

        For log-link families we therefore set:
          - Intercept: Normal(log(mean) - sigma^2/2, sigma=1.0). The
            -sigma^2/2 lognormal correction keeps E[exp(Intercept)] equal
            to the data mean rather than inflating it by exp(sigma^2/2).
          - Each C(...) categorical term: Normal(0, sigma=1.0), giving a
            95% prior on group multipliers of roughly [0.14, 7.4].

        User-supplied priors via ``self.priors`` always override the defaults.
        """
        # Detect effective link: use explicit self.link if set, else fall back to
        # the family default.  t / gaussian / normal are identity-link by default.
        from .models import _get_default_link as _gdl
        family_lower = self.family.lower()
        canonical = {
            "t": "t", "student_t": "t", "studentt": "t",
            "gaussian": "gaussian", "normal": "gaussian",
        }.get(family_lower, family_lower)
        effective_link = self.link if self.link else _gdl(canonical)
        is_log_link = effective_link == "log"

        # Determine the actual response column name (LHS of formula).
        response_col = self.formula.split("~")[0].strip()
        if response_col not in self.data_.columns:
            response_col = "incremental"
        response_values = self.data_[response_col].values

        if is_log_link:
            intercept_sigma = 1.0
            positive_values = response_values[response_values > 0]
            if len(positive_values) > 0:
                # Lognormal correction: target E[exp(Intercept)] = positive_mean
                intercept_loc = float(np.log(positive_values.mean()))
                # If an exposure offset is being applied, the intercept lives on
                # log(response / exposure) scale, so subtract log(mean(exposure))
                # from the prior location to keep the prior weakly informative.
                if self.exposure is not None and self.exposure in self.data_.columns:
                    mean_ep = float(np.nanmean(self.data_[self.exposure].values))
                    if mean_ep > 0:
                        intercept_loc = intercept_loc - float(np.log(mean_ep))
                intercept_mu = intercept_loc - intercept_sigma**2 / 2
            else:
                intercept_mu = 0.0
        else:
            response_mean = float(response_values.mean())
            response_std = float(response_values.std())
            intercept_mu = response_mean
            intercept_sigma = max(response_std, abs(response_mean) * 0.5)

        defaults: dict[str, Any] = {
            "Intercept": bmb.Prior("Normal", mu=intercept_mu, sigma=intercept_sigma),
        }

        if is_log_link:
            for term in re.findall(
                r"C\(\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\)", self.formula
            ):
                key = re.sub(r"\s+", "", term)
                defaults[key] = bmb.Prior("Normal", mu=0.0, sigma=1.0)

        # Layer chain-ladder-informed priors on top of defaults (before user priors)
        if self.init_priors_from_chainladder and self.triangle_ is not None:
            cl_priors = self._build_cl_informed_priors()
            defaults.update(cl_priors)

        if self.priors:
            defaults.update(self.priors)

        return defaults

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
            f"    include_process_variance={self.include_process_variance},\n"
            f"    init_priors_from_chainladder={self.init_priors_from_chainladder},\n"
            f"    status={fitted_str}\n"
            f")"
        )


class BayesianCSR(BaseStochasticReserve):
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
        super().__init__()
        self.priors = priors
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept
        self.random_seed = random_seed
        self.include_process_variance = include_process_variance

        # CSR-specific fitted attributes (not in base)
        self.model_: pm.Model | None = None
        self.idata: az.InferenceData | None = None
        self.data_: pd.DataFrame | None = None
        self.future_data_: pd.DataFrame | None = None
        self.elr_posterior_: xr.DataArray | None = None
        self.gamma_posterior_: xr.DataArray | None = None

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
        """Build reserves_posterior_ from per-origin samples and delegate
        to the base class for ibnr_/ultimate_ assembly."""
        origins = sorted(future_predictions.keys())
        ibnr_samples_list = [
            future_predictions[origin]["ibnr_samples"].flatten() for origin in origins
        ]
        ibnr_array = np.stack(ibnr_samples_list, axis=0)  # (n_origin, n_samples)

        self.reserves_posterior_ = xr.DataArray(
            ibnr_array,
            dims=["origin", "sample"],
            coords={
                "origin": origins,
                "sample": np.arange(ibnr_array.shape[1]),
            },
        )

        # Base class builds ibnr_ / ultimate_ tables
        self._build_reserve_summaries()

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
