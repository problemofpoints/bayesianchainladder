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
    force_positive_response : bool, optional
        When True (default), automatically shift the response column by
        ``|min_value| + 1.0`` before fitting if the family requires strictly
        positive values (``gamma``) and the data contain non-positive values.
        A UserWarning is emitted describing the shift magnitude.  Reserve
        estimates are back-shifted by the same amount per future cell so the
        output is on the original data scale.  Set to False to disable the
        shift and allow the legacy ``ValueError`` from
        ``_validate_data_family_compatibility`` to surface instead.
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
        force_positive_response: bool = True,
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
        self.force_positive_response = force_positive_response

        # GLM-specific fitted attributes (not in base)
        self.model_: bmb.Model | None = None
        self.idata: az.InferenceData | None = None
        self.data_: pd.DataFrame | None = None
        self.future_data_: pd.DataFrame | None = None
        self.fitted_: pd.DataFrame | None = None
        self._original_exposure_col: str | None = None
        self._response_shift: float = 0.0
        self._n_dummy_rows: int = 0
        # Set of dev values that were added as dummy rows (no real observations).
        # Used by _build_cl_informed_priors to assign tighter σ to those levels.
        self._dummy_dev_levels: set[int] = set()
        # Origins dropped for having zero observations (all NaN in training data).
        self._dropped_zero_obs_origins: list = []

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

        # Drop origins that have ZERO observations in training data.
        # An origin with all NaN values contributes no likelihood information.
        # Such origins never appear in data_ (triangle_to_dataframe drops NaN cells),
        # but they DO appear in future_data_ because get_future_dataframe generates
        # predictions for every origin row.  We detect them as origins present in
        # future_data_ but absent from data_, then also remove them from future_data_.
        # Retaining them would produce purely-prior-driven predictions for that
        # origin — an unanchored posterior that can generate extreme estimates.
        #
        # Additionally, origins may have rows in data_ but with ALL NaN response
        # values (e.g. if add_categorical_columns leaves them).  We catch both cases.
        response_col_for_drop = self.formula.split("~")[0].strip()
        if response_col_for_drop not in self.data_.columns:
            response_col_for_drop = "incremental"
        data_origins_set = set(self.data_["origin"].unique().tolist())
        future_origins_set = set(self.future_data_["origin"].unique().tolist())
        # Case 1: origins in future_data_ but not in data_ at all (all-NaN origins)
        absent_origins = future_origins_set - data_origins_set
        # Case 2: origins in data_ but with zero non-NaN response values
        zero_resp_origins: list = []
        if response_col_for_drop in self.data_.columns:
            obs_counts = self.data_.groupby("origin")[response_col_for_drop].count()
            zero_resp_origins = obs_counts[obs_counts == 0].index.tolist()
        zero_obs_origins = list(absent_origins) + zero_resp_origins
        if zero_obs_origins:
            self._dropped_zero_obs_origins = zero_obs_origins
            self.data_ = self.data_[
                ~self.data_["origin"].isin(zero_obs_origins)
            ].reset_index(drop=True)
            self.future_data_ = self.future_data_[
                ~self.future_data_["origin"].isin(zero_obs_origins)
            ].reset_index(drop=True)
        else:
            self._dropped_zero_obs_origins = []

        # Ensure every C(...) level that appears in future_data_ is also present
        # in data_.  The formulae library (used internally by Bambi) derives the
        # categorical level set from np.unique() of the actual training values —
        # it ignores pd.Categorical dtype categories.  If a dev level (e.g., 108)
        # appears only in future_data_, predict() raises:
        #   ValueError: The levels (108) in 'C(dev)' are not present in the original data set.
        #
        # Fix: for each C(...) column, add one dummy training row per missing
        # level.  Dummy rows use:
        #   - the first observed origin (arbitrary; origin effect is independent)
        #   - a small positive response value (1.0) so count families are happy
        #   - _obs_weight = 0  (real rows carry _obs_weight = 1)
        # The _obs_weight column is carried through so _compute_predictions can
        # strip dummy rows from fitted_.  The dummy rows have negligible
        # likelihood influence when real loss values are thousands to millions.
        self._n_dummy_rows = 0
        self._dummy_dev_levels = set()
        if len(self.future_data_) > 0:
            self.data_, self.future_data_, self._n_dummy_rows, self._dummy_dev_levels = (
                self._pad_missing_dev_levels(
                    self.data_, self.future_data_, self.formula
                )
            )

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

        # Auto-shift response for positive-support families when negatives present.
        #
        # Gamma (and other strictly-positive-support families) cannot handle
        # non-positive incremental values.  When force_positive_response=True
        # (the default), we automatically add a constant shift to the response
        # so every cell is positive before fitting.  After fitting we subtract
        # the same shift from each predicted future cell when aggregating
        # reserves, restoring the predictions to the original scale.
        #
        # The shift = |min_value| + 1.0 guarantees all values are ≥ 1.0 after
        # shifting, with a buffer that keeps the gamma distribution well-behaved.
        self._response_shift = 0.0
        if self.force_positive_response and not self.response_per_exposure:
            positive_families = ("gamma",)
            family_lower = self.family.lower()
            if family_lower in positive_families:
                response_col = self.formula.split("~")[0].strip()
                if response_col not in self.data_.columns:
                    response_col = "incremental"
                resp_vals = np.asarray(self.data_[response_col].values, dtype=np.float64)
                min_val = float(np.nanmin(resp_vals))
                if min_val <= 0:
                    import warnings as _w
                    shift = abs(min_val) + 1.0
                    _w.warn(
                        f"BayesianChainLadderGLM: response column '{response_col}' "
                        f"contains non-positive values (min={min_val:.4g}) which are "
                        f"incompatible with the '{self.family}' family. "
                        f"Automatically shifting response by +{shift:.4g} to make all "
                        f"values strictly positive. "
                        f"Reserve estimates are back-shifted by the same amount per "
                        f"future cell to restore original scale. "
                        f"Set force_positive_response=False to disable this behavior.",
                        UserWarning,
                        stacklevel=3,
                    )
                    self._response_shift = shift
                    self.data_ = self.data_.copy()
                    self.data_[response_col] = resp_vals + shift

        # Validate data compatibility with chosen family
        self._validate_data_family_compatibility()

        priors = self._build_default_priors()

        # Build a potential that zeroes out dummy-row likelihood contributions.
        # Dummy rows are inserted by _pad_missing_dev_levels to expose unseen
        # categorical levels to Bambi's design matrix.  Without masking, their
        # placeholder response values conflict with tight CL-informed priors on
        # the corresponding dev-level contrasts, pushing the intercept far from
        # its prior and producing extreme predictions.
        dummy_potentials = self._build_dummy_row_potential()

        # Build the model
        offset = self.exposure if self.exposure else None
        self.model_ = build_bambi_model(
            data=self.data_,
            formula=self.formula,
            family=self.family,
            link=self.link,
            priors=priors,
            offset=offset,
            potentials=dummy_potentials,
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

    @staticmethod
    def _pad_missing_dev_levels(
        data: pd.DataFrame,
        future_data: pd.DataFrame,
        formula: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, int, set[int]]:
        """Append one dummy row per C(...) level present in future_data but not in data.

        The formulae library (Bambi's design-matrix engine) determines categorical
        levels from ``np.unique()`` of the actual training values — it ignores
        pd.Categorical dtype categories.  If a level appears only in future_data,
        Bambi raises ``ValueError: The levels (...) are not present in the original
        data set`` at predict-time.

        This method guarantees every future level is seen during training by
        inserting one lightweight dummy row per missing level.  Dummy rows carry:

        * ``_obs_weight = 0``  (real rows get ``_obs_weight = 1``)
        * response value = small positive constant (1.0), safe for all families
        * ``origin`` = first observed origin (its effect is independent of ``dev``)
        * other numeric columns copied from the first real row

        The caller (``fit``) stores the dummy-row count in ``self._n_dummy_rows``
        so that ``_compute_predictions`` can strip those rows from ``fitted_``.

        Both ``data`` and ``future_data`` are returned with unified pd.Categorical
        level sets on every C(...) column so that dtype comparisons work correctly.

        Parameters
        ----------
        data : pd.DataFrame
            Training data (observed cells).
        future_data : pd.DataFrame
            Future (prediction) data.
        formula : str
            Bambi formula string used to identify C(...) columns.

        Returns
        -------
        padded_data : pd.DataFrame
            Training data extended with dummy rows (or unchanged if none needed).
        updated_future : pd.DataFrame
            Future data with unified pd.Categorical levels on C(...) columns.
        n_dummy : int
            Number of dummy rows added.
        dummy_dev_levels : set[int]
            Set of ``dev`` column values that were added exclusively as dummy
            rows (no real observations at those dev levels).  Used by
            ``_build_cl_informed_priors`` to assign tighter priors to those
            levels.
        """
        c_wrapped_cols = re.findall(r'\bC\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)', formula)
        if not c_wrapped_cols:
            # Mark real rows and return unchanged (no C(...) terms in formula)
            data = data.copy()
            data["_obs_weight"] = 1.0
            return data, future_data, 0, set()

        dummy_rows: list[pd.DataFrame] = []
        # Track dev values that are added exclusively via dummy rows.
        dummy_dev_levels: set[int] = set()

        # Determine the response column (LHS of formula)
        response_col = formula.split("~")[0].strip()
        if response_col not in data.columns:
            response_col = "incremental"

        # Reference row: first real row — copy all columns from here for dummy rows
        ref_row = data.iloc[[0]].copy()

        for col in c_wrapped_cols:
            if col not in future_data.columns or col not in data.columns:
                continue
            train_vals = set(data[col].dropna().unique().tolist())
            # Cast to the same type for comparison (handles int/str mix in Categorical)
            try:
                future_vals = set(int(v) for v in future_data[col].dropna().unique())
                train_vals_cast = set(int(v) for v in train_vals)
            except (TypeError, ValueError):
                future_vals = set(future_data[col].dropna().unique().tolist())
                train_vals_cast = train_vals

            missing = sorted(future_vals - train_vals_cast)
            for level in missing:
                dummy = ref_row.copy()
                dummy[col] = level
                # Small positive response value — valid for all supported families.
                # With Meyers-scale data (thousands to millions), a single row with
                # value 1.0 has negligible likelihood influence.
                if response_col in dummy.columns:
                    dummy[response_col] = 1.0
                dummy["_obs_weight"] = 0.0
                dummy_rows.append(dummy)
                # Track dummy dev levels so callers can tighten priors on them.
                if col == "dev":
                    try:
                        dummy_dev_levels.add(int(level))
                    except (TypeError, ValueError):
                        pass

        # Mark real rows regardless of whether dummy rows were added
        data = data.copy()
        data["_obs_weight"] = 1.0

        if dummy_rows:
            padded = pd.concat([data] + dummy_rows, ignore_index=True)
            n_dummy = len(dummy_rows)
        else:
            padded = data
            n_dummy = 0

        # Re-apply unified pd.Categorical level set on every C(...) column.
        # pd.concat loses Categorical dtype; we also want train/future to agree.
        future_data = future_data.copy()
        for col in c_wrapped_cols:
            if col not in padded.columns:
                continue
            try:
                all_levels = sorted(
                    set(int(v) for v in padded[col].dropna().unique())
                    | set(int(v) for v in future_data[col].dropna().unique()
                          if col in future_data.columns)
                )
            except (TypeError, ValueError):
                all_levels = sorted(
                    set(padded[col].dropna().unique().tolist())
                    | (set(future_data[col].dropna().unique().tolist())
                       if col in future_data.columns else set())
                )
            padded[col] = pd.Categorical(padded[col], categories=all_levels)
            if col in future_data.columns:
                future_data[col] = pd.Categorical(future_data[col], categories=all_levels)

        return padded, future_data, n_dummy, dummy_dev_levels

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

        # Extract fitted values — strip dummy rows (added by _pad_missing_dev_levels)
        # before storing fitted_, so it aligns with the original observed data.
        fitted_mean = posterior_data[mean_name].mean(dim=["chain", "draw"])
        fitted_data = self.data_.copy()
        fitted_data["fitted_mean"] = fitted_mean.values
        if self._n_dummy_rows > 0:
            # Dummy rows were appended at the END of data_; keep only real rows.
            fitted_data = fitted_data.iloc[: len(fitted_data) - self._n_dummy_rows].copy()
        self.fitted_ = fitted_data

        # Predict future cells.
        # Use formulae's "silent" mode so that any C(...) level in fut_data that
        # is not in the training data is treated as the reference level (all-zero
        # contrast) rather than raising a ValueError.  The dummy-row padding above
        # ensures every future level IS present in training, so this guard only
        # fires if some edge case slips through (e.g. C(calendar) extrapolation).
        import formulae as _formulae
        _prev_unseen = _formulae.config["EVAL_UNSEEN_CATEGORIES"]
        _formulae.config["EVAL_UNSEEN_CATEGORIES"] = "silent"
        try:
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
        finally:
            _formulae.config["EVAL_UNSEEN_CATEGORIES"] = _prev_unseen

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
                n_cells_for_origin = len(origin_future_idx)

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

                origin_total = origin_preds.sum(dim=obs_dim)

                # Back-shift: each future cell's predicted mean is inflated by
                # _response_shift (applied before fitting).  Subtract the total
                # shift for this origin to restore the original scale.
                if self._response_shift != 0.0:
                    origin_total = origin_total - self._response_shift * n_cells_for_origin

                reserve_samples[origin] = origin_total

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

            self._store_full_posterior(future_predictions, obs_dim, future_start)

    def _store_full_posterior(
        self, future_predictions: xr.DataArray, obs_dim: str, future_start: int
    ) -> None:
        """Assemble complete simulated cumulative triangles (origin, dev,
        sample) from observed incrementals plus per-cell future predictions,
        applying the same loss-ratio back-transform and response shift as
        ``_compute_reserves`` so both views agree cell by cell."""
        from ._triangle_ops import cumulative_array, cumulative_to_incremental

        cum, origins, devs = cumulative_array(self.triangle_)
        incr_obs = cumulative_to_incremental(cum)
        observed = ~np.isnan(incr_obs)

        fut = (
            future_predictions.isel({obs_dim: slice(future_start, None)})
            .stack(sample=["chain", "draw"])
            .transpose(obs_dim, "sample")
            .values
        )
        n_samples = fut.shape[1]
        incr = np.repeat(np.where(observed, incr_obs, 0.0)[..., None], n_samples, axis=-1)
        valid = observed.copy()

        ep_col = self._original_exposure_col if self.response_per_exposure else None
        if ep_col is not None and ep_col not in self.future_data_.columns:
            ep_col = None

        fut_origin = self.future_data_["origin"].values
        fut_dev = self.future_data_["dev"].values
        for k in range(len(self.future_data_)):
            o, d = int(fut_origin[k]), int(fut_dev[k])
            if o not in origins or d not in devs:
                continue
            i, j = origins.index(o), devs.index(d)
            cell = np.asarray(fut[k], dtype=float)
            if ep_col is not None:
                cell = cell * float(self.future_data_.iloc[k][ep_col])
            if self._response_shift != 0.0:
                cell = cell - self._response_shift
            incr[i, j, :] = cell
            valid[i, j] = True

        full = np.cumsum(incr, axis=1)
        full[~valid] = np.nan
        self._set_full_cumulative_posterior(full, origins, devs)

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

    def _build_dummy_row_potential(self) -> list[tuple] | None:
        """Build a Bambi potential that zeroes out dummy rows' likelihood contribution.

        Dummy rows are added by ``_pad_missing_dev_levels`` to force missing
        categorical levels (e.g. dev=120 when only dev=12..108 are observed)
        into the Bambi design matrix.  Without this potential those rows
        participate in the likelihood with their placeholder response values
        (1.0 by default, or shifted), which creates severe tension with any
        CL-informed prior on the corresponding dev-level contrast.  The tension
        forces the intercept to drift far from its prior, producing extreme
        predictions for ALL future cells.

        The potential adds ``sum((mask - 1) * logp_per_obs)`` to the log-
        likelihood, where ``mask[i] = 0`` for dummy rows and ``1`` for real
        rows.  For dummy rows this subtracts their logp contribution, making
        them effectively zero-weight observations.  Real rows are unaffected.

        Family support
        --------------
        * gamma      — uses Gamma(alpha, mu) parameterization
        * gaussian   — uses Normal(mu, sigma) parameterization
        * All others — returns None (no potential; dummy rows contribute
                       negligibly when real losses >> 1)

        Returns None when there are no dummy rows or the family is unsupported.
        """
        if self._n_dummy_rows == 0:
            return None

        family_lower = self.family.lower()
        if family_lower not in ("gamma",):
            # Only gamma models are affected because they have tight priors on
            # dev contrasts that cause severe intercept drift.  Other families
            # (gaussian, negativebinomial, poisson) don't exhibit the same
            # pathology with response values of order ~1 relative to the data.
            return None

        # Determine the response column name
        response_col = self.formula.split("~")[0].strip()
        if response_col not in self.data_.columns:
            response_col = "incremental"

        # Build the per-observation mask: 0 for dummy rows, 1 for real rows
        obs_weight_col = "_obs_weight"
        if obs_weight_col in self.data_.columns:
            obs_mask = np.asarray(self.data_[obs_weight_col].values, dtype=np.float64)
        else:
            # Fall back: dummy rows were appended at the END; mark last n_dummy as 0
            obs_mask = np.ones(len(self.data_), dtype=np.float64)
            obs_mask[-self._n_dummy_rows :] = 0.0

        # Capture response values and mask as constants for use in the closure
        y_vals = np.asarray(self.data_[response_col].values, dtype=np.float64)
        mask_vals = obs_mask.copy()

        if family_lower == "gamma":
            # Capture required values and imports in the closure explicitly
            _y_vals = y_vals
            _mask_vals = mask_vals

            def _gamma_mask(mu_val: "pt.TensorVariable", alpha_val: "pt.TensorVariable") -> "pt.TensorVariable":
                """Subtract dummy-row gamma logp from the joint log-likelihood."""
                import pytensor.tensor as _pt
                import pytensor.tensor.special as _pts

                y = _pt.as_tensor_variable(_y_vals)
                w = _pt.as_tensor_variable(_mask_vals)

                # Gamma logp: shape=alpha, rate=alpha/mu
                # logp(y; alpha, mu) = alpha*log(alpha/mu) + (alpha-1)*log(y)
                #                     - (alpha/mu)*y - lgamma(alpha)
                b = alpha_val / mu_val  # rate parameter, shape (n_obs,)
                logp_each = (
                    alpha_val * _pt.log(b)
                    + (alpha_val - 1.0) * _pt.log(y)
                    - b * y
                    - _pts.gammaln(alpha_val)
                )
                # (w - 1) == -1 for dummy rows, 0 for real rows
                return _pt.sum((w - 1.0) * logp_each)

            return [(("mu", "alpha"), _gamma_mask)]

        return None  # unreachable for supported families above

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

        # -------------------------------------------------------------------
        # Build lookup maps: origin/dev value → index in triangle arrays.
        # These are used to filter prior arrays to ONLY the levels that
        # actually have training observations — Bambi drops levels with no
        # rows from the design matrix, so the prior array length must match.
        # -------------------------------------------------------------------
        def _period_to_int(p) -> int:
            """Convert a chainladder period (Timestamp/Period/int) to int year."""
            if hasattr(p, "year"):
                return int(p.year)
            if hasattr(p, "days"):
                return max(1, round(p.days / 365))
            return int(p)

        tri_origin_vals = [_period_to_int(o) for o in tri.origin]   # list[int]
        tri_dev_vals    = [_period_to_int(d) for d in tri.development]  # list[int]
        origin_val_to_idx: dict[int, int] = {v: i for i, v in enumerate(tri_origin_vals)}
        dev_val_to_idx:    dict[int, int] = {v: i for i, v in enumerate(tri_dev_vals)}

        # Observed levels = values that actually appear in the training data.
        # Bambi's treatment-coding reference = first observed level (sorted asc).
        if self.data_ is not None:
            obs_origins_sorted = sorted(int(v) for v in self.data_["origin"].dropna().unique())
            obs_devs_sorted    = sorted(int(v) for v in self.data_["dev"].dropna().unique())
        else:
            obs_origins_sorted = tri_origin_vals
            obs_devs_sorted    = tri_dev_vals

        # Reference cells: first *observed* origin/dev (Bambi treatment reference)
        ref_origin_idx = origin_val_to_idx.get(obs_origins_sorted[0], 0) if obs_origins_sorted else 0
        ref_dev_idx    = dev_val_to_idx.get(obs_devs_sorted[0], 0)       if obs_devs_sorted    else 0

        ref_ult   = float(ult_arr[ref_origin_idx]) if ult_arr.size > ref_origin_idx else 1.0
        ref_ult   = max(ref_ult, 1e-8)
        ref_frac  = float(incr_pct[ref_dev_idx]) if incr_pct.size > ref_dev_idx else 1e-10
        ref_frac  = max(ref_frac, 1e-10)

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
        if c_origin_match and self.data_ is not None and len(obs_origins_sorted) >= 2:
            # Bambi treatment coding: contrasts for obs_origins_sorted[1:] vs [0].
            # Only observed origins get contrast columns — levels with no training
            # rows are silently dropped from the design matrix by formulae/Bambi.
            contrast_origins = obs_origins_sorted[1:]  # exclude reference
            n_contrasts_origin = len(contrast_origins)

            if is_log_link:
                # Prior means = log(ult[k] / ult[ref]) for each contrast origin k
                origin_mus = np.array([
                    np.log(max(float(ult_arr[origin_val_to_idx[o]]), 1e-8)) - np.log(ref_ult)
                    if o in origin_val_to_idx and origin_val_to_idx[o] < len(ult_arr)
                    else 0.0
                    for o in contrast_origins
                ], dtype=float)
            else:
                # Identity link: prior on loss-ratio-scale origin effects
                # LR[k] = ult[k] / EP[k]; contrast mu_k = (LR[k] - LR[ref]) * frac[ref]
                if ep_by_origin is not None:
                    origin_mus = np.array([
                        (
                            float(ult_arr[origin_val_to_idx[o]]) / max(ep_by_origin.get(o, 1.0), 1e-8)
                            - ref_ult / max(ep_by_origin.get(obs_origins_sorted[0], 1.0), 1e-8)
                        ) * ref_frac
                        if o in origin_val_to_idx and origin_val_to_idx[o] < len(ult_arr)
                        else 0.0
                        for o in contrast_origins
                    ], dtype=float)
                else:
                    origin_mus = np.zeros(n_contrasts_origin)

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
        if c_dev_match and len(obs_devs_sorted) >= 2:
            # Bambi treatment coding: contrasts for obs_devs_sorted[1:] vs [0].
            # Only dev levels that have at least one observed training row are
            # included in the design matrix by Bambi — match that here.
            contrast_devs = obs_devs_sorted[1:]  # exclude reference
            n_contrasts_dev = len(contrast_devs)

            if is_log_link:
                # Prior means = log(frac[j] / frac[ref]) for each contrast dev j.
                #
                # Shift adjustment: when the response is shifted by _response_shift
                # (to handle negative incrementals for gamma families), the model
                # fits log(E[incremental + shift]) rather than log(E[incremental]).
                # The CL-informed prior for C(dev)[j] must reflect the *shifted*
                # incremental scale; otherwise it may be astronomically negative for
                # near-zero or negative dev periods (e.g. last dev with tiny/negative
                # actual incremental), forcing the intercept to drift and producing
                # extreme predictions.
                #
                # For a log-link model with exposure EP, the expected response per
                # unit EP at dev j (shifted scale) is:
                #   incr_j_shifted_per_EP = incr_pct[j] * mean_LR + shift / mean_EP
                # where mean_LR = mean(ult/EP) and mean_EP = mean earned premium.
                #
                # Shift-adjusted prior: log(incr_j_shifted_per_EP / incr_ref_shifted_per_EP)
                response_shift = getattr(self, "_response_shift", 0.0)
                shift_per_ep = 0.0  # default: no adjustment
                if response_shift > 0.0 and self.data_ is not None:
                    exp_col = self.exposure
                    if exp_col and exp_col in self.data_.columns:
                        ep_vals = np.asarray(
                            self.data_[exp_col].dropna().values, dtype=float
                        )
                        mean_ep_val = float(np.mean(ep_vals[ep_vals > 0])) if ep_vals.size > 0 else 1.0
                        # loss ratio = mean(ult / EP) across origins
                        lr_vals = []
                        for oi, ou in zip(tri_origin_vals, ult_arr):
                            ep_origin = float(
                                np.mean(
                                    ep_vals[
                                        (self.data_["origin"].values == oi)
                                        if "origin" in self.data_.columns
                                        else np.ones(len(ep_vals), dtype=bool)
                                    ]
                                )
                            ) if len(ep_vals) > 0 else 1.0
                            if ep_origin > 0 and ou > 0:
                                lr_vals.append(float(ou) / ep_origin)
                        mean_lr = float(np.mean(lr_vals)) if lr_vals else 1.0
                        mean_lr = max(mean_lr, 1e-8)
                        mean_ep_val = max(mean_ep_val, 1.0)
                        # shift contribution per unit EP
                        shift_per_ep = float(response_shift) / mean_ep_val
                    else:
                        # No EP column available; use a rough scale based on mean incremental
                        resp_col = self.formula.split("~")[0].strip()
                        if resp_col in self.data_.columns:
                            resp_vals = np.asarray(self.data_[resp_col].values, dtype=float)
                            mean_incr = float(np.nanmean(resp_vals[resp_vals > 0])) if resp_vals.size > 0 else 1.0
                            mean_lr = 1.0  # dimensionless
                            shift_per_ep = float(response_shift) / max(mean_incr, 1.0)
                        else:
                            mean_lr = 1.0
                else:
                    mean_lr = 1.0

                def _shifted_frac(pct: float) -> float:
                    """Compute (pct * mean_lr + shift_per_ep), floored at 1e-10."""
                    return max(pct * mean_lr + shift_per_ep, 1e-10)

                ref_frac_shifted = _shifted_frac(ref_frac)

                dev_mus = np.array([
                    np.log(_shifted_frac(float(incr_pct[dev_val_to_idx[d]]))) - np.log(ref_frac_shifted)
                    if d in dev_val_to_idx and dev_val_to_idx[d] < len(incr_pct)
                    else 0.0
                    for d in contrast_devs
                ], dtype=float)
            else:
                # Identity link: incremental LR deviation from reference dev
                # ref origin LR * (frac[j] - frac[ref])
                if ep_by_origin is not None and obs_origins_sorted:
                    ref_ep = max(ep_by_origin.get(obs_origins_sorted[0], 1.0), 1e-8)
                    lr_ref = ref_ult / ref_ep
                    dev_mus = np.array([
                        lr_ref * (
                            float(incr_pct[dev_val_to_idx[d]]) - ref_frac
                            if d in dev_val_to_idx and dev_val_to_idx[d] < len(incr_pct)
                            else 0.0
                        )
                        for d in contrast_devs
                    ], dtype=float)
                else:
                    dev_mus = np.zeros(n_contrasts_dev)

            if len(dev_mus) > 0:
                # Use a tight sigma (0.1) for dev levels that were added as
                # dummy rows (zero real observations).  Those levels have no
                # likelihood support, so a wide prior would let the posterior
                # wander freely and produce extreme predictions.  σ=0.1 pins
                # the coefficient close to the chain-ladder informed mean while
                # still allowing small Bayesian updates.
                dummy_devs = getattr(self, "_dummy_dev_levels", set())
                dev_sigmas = np.array([
                    0.1 if (d in dummy_devs) else sd
                    for d in contrast_devs
                ], dtype=float)
                priors["C(dev)"] = bmb.Prior(
                    "Normal",
                    mu=dev_mus.astype(float),
                    sigma=dev_sigmas,
                )

        # -------------------------------------------------------------------
        # bs(dev_idx, df=N) spline priors — project CL pattern onto basis
        # -------------------------------------------------------------------
        spline_match = re.search(r'\bbs\s*\(\s*dev_idx\s*,\s*df\s*=\s*(\d+)\s*\)', self.formula)
        if spline_match and self.data_ is not None:
            df_spline = int(spline_match.group(1))
            try:
                # Use the ACTUAL dev_idx values present in the training data.
                # This is the 1-based integer index column added by
                # add_categorical_columns for the ``dev`` column.  It may cover
                # fewer periods than n_devs when the triangle has partial coverage
                # (e.g., only dev=12..96 observed, not 12..120).  Building B and
                # y_target from the same set of periods guarantees compatible shapes.
                if "dev_idx" in self.data_.columns:
                    dev_idx_train = np.array(
                        sorted(self.data_["dev_idx"].unique()), dtype=float
                    )
                else:
                    # Fallback: construct 1-based indices for all triangle dev periods
                    dev_idx_train = np.arange(1, n_devs + 1, dtype=float)

                n_train_devs = len(dev_idx_train)

                # CL incremental fractions indexed by dev_idx (1-based).
                # incr_pct has length n_devs (full triangle); select only the
                # subset that corresponds to dev_idx_train.
                dev_idx_int = dev_idx_train.astype(int)
                valid_mask = (dev_idx_int >= 1) & (dev_idx_int <= n_devs)
                if not np.all(valid_mask):
                    raise ValueError("dev_idx values out of range of CL dev periods")
                # 0-based indexing into incr_pct
                incr_pct_train = incr_pct[dev_idx_int[valid_mask] - 1]

                # Build B-spline basis via scipy (same knots as df=N default).
                from scipy.interpolate import BSpline

                # Build basis manually using patsy-style: evenly-spaced interior
                # knots with cubic (degree=3) B-splines, augmented boundary knots.
                # df = n_interior_knots + degree + 1 - include_intercept
                # For df=4, degree=3: n_interior_knots = 0  (no interior knots)
                degree = 3
                n_interior = df_spline - degree - 1  # = 0 for df=4
                n_interior = max(n_interior, 0)

                x = dev_idx_train
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
                B = np.zeros((n_train_devs, n_basis))
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
                # B and y_target are now both length n_train_devs.
                if is_log_link:
                    y_target = np.log(np.maximum(incr_pct_train, 1e-10))
                else:
                    # Identity link: raw incr_pct as target (roughly)
                    y_target = incr_pct_train.copy()

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
            # Empirical SD of log(ultimates) across origins.
            # Use exactly 1× the empirical SD as the HalfNormal scale: the
            # random-effects pooling structure already allows the posterior to
            # shrink or expand, so there is no need to inflate by 2×.  Tighter
            # hyperprior reduces posterior width on per-origin RE, which in turn
            # reduces combined spline + RE prediction uncertainty.
            valid_ults = ult_arr[ult_arr > 0]
            if valid_ults.size >= 2:
                log_ult_sd = float(np.std(np.log(valid_ults), ddof=1))
                hn_sigma = max(log_ult_sd, 0.01)
                priors["1|origin"] = bmb.Prior(
                    "Normal",
                    mu=0,
                    sigma=bmb.Prior("HalfNormal", sigma=hn_sigma),
                )

        # -------------------------------------------------------------------
        # (1 | calendar) random intercept — weakly informative HalfNormal
        # -------------------------------------------------------------------
        # Calendar effects are typically small (< 20% relative impact).
        # HalfNormal(0.2) gives a 95th percentile of ~0.4 on log scale,
        # corresponding to roughly a ±40% calendar-year swing — weakly
        # informative but much tighter than Bambi's sd(y)-scaled default.
        re_cal_match = re.search(r'\(1\s*\|\s*calendar\s*\)', self.formula)
        if re_cal_match:
            priors["1|calendar"] = bmb.Prior(
                "Normal",
                mu=0,
                sigma=bmb.Prior("HalfNormal", sigma=0.2),
            )

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
            f"    force_positive_response={self.force_positive_response},\n"
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
        from ._triangle_ops import cumulative_array

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

        # Per-cell cumulative posterior paths (England & Verrall "Complete
        # Cumulatives"): start from the observed diagonal, broadcast across
        # samples, and fill in future cells origin-by-origin below.
        cum_obs, tri_origins, tri_devs = cumulative_array(self.triangle_)
        n_samples_total = int(np.prod(alpha.shape[:2]))
        full_paths = np.repeat(cum_obs[..., None], n_samples_total, axis=-1)

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
                # comonotonic paths: one standard-normal shock per (chain,
                # draw) is shared across every future development period for
                # this origin, so the simulated triangle is internally
                # consistent cell-to-cell (England & Verrall "Complete
                # Cumulatives"), while the ultimate cell reproduces exactly
                # the distribution the original single-cell formula gives.
                last_dev_idx = dev_levels.index(last_observed_dev)
                future_dev_idx = list(range(last_dev_idx + 1, ultimate_dev_idx + 1))
                z = np.random.standard_normal(alpha.shape[:2])  # one shock per (chain, draw)
                path_cells = {}
                for k in future_dev_idx:
                    mu_k = (
                        logprem
                        + logelr
                        + alpha[:, :, origin_idx]
                        + beta[:, :, k] * speedup[:, :, origin_idx]
                    )
                    sig_k = sig[:, :, k]
                    if self.include_process_variance:
                        path_cells[k] = np.exp(mu_k + sig_k * z)
                    else:
                        path_cells[k] = np.exp(mu_k + 0.5 * sig_k**2)
                ultimate_cumulative = path_cells[ultimate_dev_idx]

                tri_i = tri_origins.index(int(origin))
                for k, cells in path_cells.items():
                    tri_j = tri_devs.index(int(dev_levels[k]))
                    full_paths[tri_i, tri_j, :] = cells.reshape(-1)

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

        self._set_full_cumulative_posterior(full_paths, tri_origins, tri_devs)

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
