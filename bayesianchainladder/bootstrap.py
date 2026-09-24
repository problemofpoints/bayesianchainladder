"""Stochastic reserve wrappers around chainladder bootstrap/Mack methods.

This module provides wrapper estimators that share the
:class:`bayesianchainladder.base.BaseStochasticReserve` interface:

- :class:`MackChainLadder`: wraps ``chainladder.MackChainladder`` and exposes
  reserve uncertainty as a normal approximation calibrated to Mack's
  ``total_mack_std_err_``.
- :class:`BootstrapODPChainLadder`: wraps ``chainladder.BootstrapODPSample``
  + ``chainladder.Chainladder`` and exposes the bootstrap reserve samples
  directly.
- :class:`BootstrapODPBornhuetterFerguson`: ODP bootstrap residuals with
  ``chainladder.BornhuetterFerguson`` applied per resample.
- :class:`BootstrapODPCapeCod`: ODP bootstrap residuals with
  ``chainladder.CapeCod`` applied per resample.
- :class:`CorrelatedBootstrapChainLadder`: wraps
  :class:`CorrelatedBootstrapODPSample` (Clark/Ding/Zhou 2022) — the same
  bootstrap with calendar-year correlation between cells via a Gaussian
  copula.
- :class:`CorrelatedBootstrapODPBornhuetterFerguson`: correlated bootstrap
  residuals with ``chainladder.BornhuetterFerguson`` applied per resample.
- :class:`CorrelatedBootstrapODPCapeCod`: correlated bootstrap residuals
  with ``chainladder.CapeCod`` applied per resample.

The low-level :class:`CorrelatedBootstrapODPSample` lives here too so the
file is self-contained.
"""

from __future__ import annotations

import types
from warnings import warn

import chainladder as cl
import numpy as np
import pandas as pd
import xarray as xr
from chainladder.development import Development, DevelopmentBase
from chainladder.methods.chainladder import Chainladder
from scipy import stats
from scipy.linalg import cholesky

from .base import BaseStochasticReserve, MethodSummary
from .utils import origin_labels, validate_triangle


class MackChainLadder(BaseStochasticReserve):
    """Mack chain ladder wrapped with the shared stochastic reserve interface.

    Mack only produces per-origin mean + standard error; there is no native
    sample distribution. We populate ``reserves_posterior_`` by drawing
    independent ``Normal(mean_i, stderr_i)`` samples per origin, which gives
    correct *per-origin* marginals. The total of these per-origin draws,
    however, will understate the true total uncertainty because Mack's
    cross-origin covariance is not exposed by chainladder's public API.

    To compensate, ``sample_reserves()`` and ``total_summary()`` are
    overridden to draw from ``Normal(total_mean, total_mack_std_err_)``,
    where ``total_mack_std_err_`` is the calibrated total stderr that
    accounts for cross-origin correlation. The total row of ``summary()``
    is similarly recomputed using the calibrated total.

    Parameters
    ----------
    n_periods : int, default -1
        Forwarded to ``chainladder.Development`` (which Mack uses internally).
        ``-1`` uses all origins.
    random_seed : int, optional
        Seed used when drawing the per-origin Normal samples that populate
        ``reserves_posterior_``. ``sample_reserves()`` accepts its own seed.
    n_samples : int, default 5000
        Number of per-origin samples drawn into ``reserves_posterior_``.
    """

    def __init__(
        self,
        n_periods: int = -1,
        random_seed: int | None = None,
        n_samples: int = 5000,
    ) -> None:
        super().__init__()
        self.n_periods = n_periods
        self.random_seed = random_seed
        self.n_samples = n_samples
        # Calibrated totals, populated by fit(). Use _check_is_fitted() before reading.
        self.total_reserve_mean_: float = 0.0
        self.total_reserve_stddev_: float = 0.0

    def fit(self, triangle):
        validate_triangle(triangle)
        self.triangle_ = triangle.copy()

        # n_periods goes to Development, not MackChainladder directly
        dev = cl.Development(n_periods=self.n_periods).fit_transform(triangle)
        mack = cl.MackChainladder().fit(dev)

        # Per-origin IBNR (sum across development) as a Triangle, then to ndarray
        ibnr_tri = mack.ibnr_.sum("development")
        ibnr_per_origin = np.asarray(ibnr_tri.values).flatten()

        # Per-origin Mack std error — use latest_diagonal to get one value per
        # origin (mack_std_err_ is a full triangle shape, not per-origin vector)
        std_per_origin = np.asarray(
            mack.mack_std_err_.latest_diagonal.values
        ).flatten()

        # Calibrated totals (used to override sample_reserves / total_summary)
        self.total_reserve_mean_ = float(np.nansum(ibnr_per_origin))
        self.total_reserve_stddev_ = float(
            np.asarray(mack.total_mack_std_err_).flatten()[0]
        )

        # Origin labels aligned with the per-origin arrays (same encoding as
        # the GLM data: year for annual grain, YYYYMM otherwise)
        origins = origin_labels(ibnr_tri)

        # Draw independent Normal samples per origin to populate
        # reserves_posterior_. These give correct per-origin marginals.
        rng = np.random.default_rng(self.random_seed)
        samples = np.empty((len(origins), self.n_samples))
        for i, (mean, std) in enumerate(zip(ibnr_per_origin, std_per_origin, strict=True)):
            mean_clean = float(mean) if np.isfinite(mean) else 0.0
            std_clean = float(std) if np.isfinite(std) and std >= 0 else 0.0
            samples[i] = rng.normal(loc=mean_clean, scale=std_clean, size=self.n_samples)

        self.reserves_posterior_ = xr.DataArray(
            samples,
            dims=["origin", "sample"],
            coords={"origin": origins, "sample": np.arange(self.n_samples)},
        )

        self._build_reserve_summaries()
        self._is_fitted = True
        return self

    # -- Override total-level methods to use calibrated total stderr --

    def sample_reserves(
        self,
        n_samples: int = 1000,
        random_seed: int | None = None,
    ) -> np.ndarray:
        self._check_is_fitted()
        rng = np.random.default_rng(random_seed)
        return rng.normal(
            loc=self.total_reserve_mean_,
            scale=self.total_reserve_stddev_,
            size=n_samples,
        )

    def total_summary(self) -> MethodSummary:
        self._check_is_fitted()
        mean = self.total_reserve_mean_
        stddev = self.total_reserve_stddev_
        q75, q90, q95 = stats.norm.ppf([0.75, 0.90, 0.95], loc=mean, scale=stddev)
        return MethodSummary(
            total_reserve_mean=float(mean),
            total_reserve_stddev=float(stddev),
            total_reserve_75th_percentile=float(q75),
            total_reserve_90th_percentile=float(q90),
            total_reserve_95th_percentile=float(q95),
        )

    def summary(self, include_totals: bool = True):
        # Use the base summary for per-origin rows, then replace the total
        # row with one calibrated to total_mack_std_err_.
        result = super().summary(include_totals=False)
        if not include_totals:
            return result

        total_paid = float(self.ultimate_["paid_to_date"].sum())
        mean = float(self.total_reserve_mean_)
        stddev = float(self.total_reserve_stddev_)
        q05, q25, median, q75, q95 = stats.norm.ppf(
            [0.05, 0.25, 0.50, 0.75, 0.95], loc=mean, scale=stddev
        )

        total_row = pd.DataFrame(
            {
                ("Ultimate", "paid_to_date"): [total_paid],
                ("Ultimate", "mean"): [total_paid + mean],
                ("Ultimate", "std"): [stddev],
                ("Ultimate", "median"): [total_paid + float(median)],
                ("IBNR", "mean"): [mean],
                ("IBNR", "std"): [stddev],
                ("IBNR", "median"): [float(median)],
            },
            index=["Total"],
        )
        return pd.concat([result, total_row])


class BootstrapODPChainLadder(BaseStochasticReserve):
    """ODP bootstrap chain ladder wrapped with the shared interface.

    Wraps ``chainladder.BootstrapODPSample`` (resampling) followed by
    ``chainladder.Chainladder`` (deterministic chain ladder applied to each
    resample). The resulting per-simulation IBNR distribution is stored in
    ``reserves_posterior_``.

    Parameters
    ----------
    n_sims : int, default 1000
        Number of bootstrap simulations.
    n_periods : int, default -1
        Forwarded to ``chainladder.BootstrapODPSample``. ``-1`` uses all origins.
    hat_adj : bool, default True
        Hat-matrix adjustment per Shapland.
    random_seed : int, optional
        Seed for the bootstrap resampler.
    """

    def __init__(
        self,
        n_sims: int = 1000,
        n_periods: int = -1,
        hat_adj: bool = True,
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.n_sims = n_sims
        self.n_periods = n_periods
        self.hat_adj = hat_adj
        self.random_seed = random_seed

    def fit(self, triangle):
        validate_triangle(triangle)
        self.triangle_ = triangle.copy()

        # chainladder's BootstrapODPSample chokes when key_labels length
        # doesn't match the resampled kdims shape (e.g., the triangle
        # originated from a multi-index). Force a single-key layout.
        prepared = triangle.copy()
        prepared.key_labels = ["triangle_id"]
        prepared.kdims = np.asarray([["resample"]], dtype=object)

        sampler = cl.BootstrapODPSample(
            n_sims=self.n_sims,
            n_periods=self.n_periods,
            hat_adj=self.hat_adj,
            random_state=self.random_seed,
        ).fit(prepared)
        resampled = sampler.transform(prepared)
        model = cl.Chainladder().fit(resampled)

        # ibnr_.values has shape (n_sims, 1, n_origin, n_dev). chainladder's
        # ibnr_ is already aggregated to per-origin IBNR (the n_dev axis is
        # effectively 1), so the dev-axis sum is a no-op for safety. We then
        # squeeze ONLY the singleton key axis (axis=1) to avoid collapsing
        # n_origin=1 or n_sims=1 cases.
        ibnr_vals = np.asarray(model.ibnr_.values)
        per_sim_per_origin = np.nansum(ibnr_vals, axis=-1)  # (n_sims, 1, n_origin)
        per_sim_per_origin = np.squeeze(per_sim_per_origin, axis=1)  # (n_sims, n_origin)
        per_origin_per_sim = per_sim_per_origin.T  # (n_origin, n_sims)

        origins = origin_labels(triangle)

        self.reserves_posterior_ = xr.DataArray(
            per_origin_per_sim,
            dims=["origin", "sample"],
            coords={
                "origin": origins,
                "sample": np.arange(per_origin_per_sim.shape[1]),
            },
        )

        self._build_reserve_summaries()
        self._is_fitted = True
        return self


def _build_exposure_triangle(triangle, exposure_triangle, n_sims: int | None = None):
    """Return an exposure triangle compatible with ``n_sims`` resampled rows.

    ``exposure_triangle`` must be a ``chainladder.Triangle`` with shape
    ``(1, 1, n_origin, 1)`` (one value per origin period). When the method
    is applied to all ``n_sims`` resamples at once, chainladder broadcasts a
    single-row exposure automatically — this helper just validates the input
    and returns it unchanged.
    """
    if not hasattr(exposure_triangle, "values"):
        raise TypeError(
            "exposure_triangle must be a chainladder Triangle with a .values attribute"
        )
    return exposure_triangle


def _extract_ibnr_from_bf_or_cc(model_fitted, triangle) -> xr.DataArray:
    """Extract per-origin-per-simulation IBNR into an :class:`xr.DataArray`.

    Works for any fitted chainladder method that exposes ``.ibnr_`` in shape
    ``(n_sims, 1, n_origin, 1)``.
    """
    ibnr_vals = np.asarray(model_fitted.ibnr_.values)
    # ibnr_.values has shape (n_sims, 1, n_origin, 1)
    per_sim_per_origin = np.nansum(ibnr_vals, axis=-1)  # (n_sims, 1, n_origin)
    per_sim_per_origin = np.squeeze(per_sim_per_origin, axis=1)  # (n_sims, n_origin)
    per_origin_per_sim = per_sim_per_origin.T  # (n_origin, n_sims)

    origins = origin_labels(triangle)
    return xr.DataArray(
        per_origin_per_sim,
        dims=["origin", "sample"],
        coords={
            "origin": origins,
            "sample": np.arange(per_origin_per_sim.shape[1]),
        },
    )


class BootstrapODPBornhuetterFerguson(BaseStochasticReserve):
    """ODP bootstrap Bornhuetter-Ferguson wrapped with the shared interface.

    Resamples triangle residuals via ``chainladder.BootstrapODPSample``, then
    fits ``chainladder.BornhuetterFerguson(apriori=...)`` to each resample.
    This separates *process risk* (from the ODP bootstrap) from the *parameter
    risk* correction introduced by the B-F a-priori, giving a distribution of
    B-F ultimates / IBNRs.

    Parameters
    ----------
    apriori : float or chainladder.Triangle
        Expected loss ratio (or per-origin Triangle of loss ratios). Passed
        directly to ``chainladder.BornhuetterFerguson``.
    apriori_sigma : float, default 0.0
        Uncertainty on the a-priori (see chainladder BF docs).
    n_sims : int, default 1000
        Number of bootstrap simulations.
    n_periods : int, default -1
        Forwarded to ``chainladder.BootstrapODPSample``. ``-1`` uses all.
    hat_adj : bool, default True
        Hat-matrix adjustment per Shapland.
    random_seed : int, optional
        Seed for the bootstrap resampler.

    Notes
    -----
    An ``exposure_triangle`` (premium) is **required** by B-F and must be
    supplied to :meth:`fit`. It is expected to be a
    ``chainladder.Triangle`` with one value per origin period (shape
    ``(1, 1, n_origin, 1)`` or the ``.latest_diagonal`` of a premium
    triangle).
    """

    def __init__(
        self,
        apriori: float | object = 0.65,
        apriori_sigma: float = 0.0,
        n_sims: int = 1000,
        n_periods: int = -1,
        hat_adj: bool = True,
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.apriori = apriori
        self.apriori_sigma = apriori_sigma
        self.n_sims = n_sims
        self.n_periods = n_periods
        self.hat_adj = hat_adj
        self.random_seed = random_seed

    def fit(self, triangle, exposure_triangle=None):
        """Fit the ODP bootstrap B-F model.

        Parameters
        ----------
        triangle : chainladder.Triangle
            Paid-loss triangle.
        exposure_triangle : chainladder.Triangle
            Premium (exposure) triangle, shape ``(1, 1, n_origin, 1)``.
            Required for Bornhuetter-Ferguson. Pass the ``.latest_diagonal``
            of a premium triangle or construct one directly.
        """
        if exposure_triangle is None:
            raise ValueError(
                "exposure_triangle is required for BootstrapODPBornhuetterFerguson. "
                "Pass the .latest_diagonal of a premium triangle."
            )
        validate_triangle(triangle)
        self.triangle_ = triangle.copy()
        exposure_triangle = _build_exposure_triangle(
            triangle, exposure_triangle, self.n_sims
        )

        prepared = triangle.copy()
        prepared.key_labels = ["triangle_id"]
        prepared.kdims = np.asarray([["resample"]], dtype=object)

        sampler = cl.BootstrapODPSample(
            n_sims=self.n_sims,
            n_periods=self.n_periods,
            hat_adj=self.hat_adj,
            random_state=self.random_seed,
        ).fit(prepared)
        resampled = sampler.transform(prepared)

        bf = cl.BornhuetterFerguson(
            apriori=self.apriori,
            apriori_sigma=self.apriori_sigma,
        ).fit(resampled, sample_weight=exposure_triangle)

        self.reserves_posterior_ = _extract_ibnr_from_bf_or_cc(bf, triangle)
        self._build_reserve_summaries()
        self._is_fitted = True
        return self


class BootstrapODPCapeCod(BaseStochasticReserve):
    """ODP bootstrap Cape Cod wrapped with the shared interface.

    Resamples triangle residuals via ``chainladder.BootstrapODPSample``, then
    fits ``chainladder.CapeCod(trend=..., decay=...)`` to each resample. Cape
    Cod estimates the a-priori expected loss ratio empirically from the data,
    making it more data-driven than a fixed B-F apriori.

    Parameters
    ----------
    trend : float, default 0.0
        Annual trend assumption for the Cape Cod method.
    decay : float, default 1.0
        Decay factor for the Cape Cod method.
    n_sims : int, default 1000
        Number of bootstrap simulations.
    n_periods : int, default -1
        Forwarded to ``chainladder.BootstrapODPSample``. ``-1`` uses all.
    hat_adj : bool, default True
        Hat-matrix adjustment per Shapland.
    random_seed : int, optional
        Seed for the bootstrap resampler.

    Notes
    -----
    An ``exposure_triangle`` (premium) is **required** and must be supplied to
    :meth:`fit`. It is expected to be a ``chainladder.Triangle`` with one value
    per origin period.
    """

    def __init__(
        self,
        trend: float = 0.0,
        decay: float = 1.0,
        n_sims: int = 1000,
        n_periods: int = -1,
        hat_adj: bool = True,
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.trend = trend
        self.decay = decay
        self.n_sims = n_sims
        self.n_periods = n_periods
        self.hat_adj = hat_adj
        self.random_seed = random_seed

    def fit(self, triangle, exposure_triangle=None):
        """Fit the ODP bootstrap Cape Cod model.

        Parameters
        ----------
        triangle : chainladder.Triangle
            Paid-loss triangle.
        exposure_triangle : chainladder.Triangle
            Premium (exposure) triangle, shape ``(1, 1, n_origin, 1)``.
            Required for Cape Cod. Pass the ``.latest_diagonal`` of a premium
            triangle or construct one directly.
        """
        if exposure_triangle is None:
            raise ValueError(
                "exposure_triangle is required for BootstrapODPCapeCod. "
                "Pass the .latest_diagonal of a premium triangle."
            )
        validate_triangle(triangle)
        self.triangle_ = triangle.copy()
        exposure_triangle = _build_exposure_triangle(
            triangle, exposure_triangle, self.n_sims
        )

        prepared = triangle.copy()
        prepared.key_labels = ["triangle_id"]
        prepared.kdims = np.asarray([["resample"]], dtype=object)

        sampler = cl.BootstrapODPSample(
            n_sims=self.n_sims,
            n_periods=self.n_periods,
            hat_adj=self.hat_adj,
            random_state=self.random_seed,
        ).fit(prepared)
        resampled = sampler.transform(prepared)

        cc = cl.CapeCod(trend=self.trend, decay=self.decay).fit(
            resampled, sample_weight=exposure_triangle
        )

        self.reserves_posterior_ = _extract_ibnr_from_bf_or_cc(cc, triangle)
        self._build_reserve_summaries()
        self._is_fitted = True
        return self


class CorrelatedBootstrapODPSample(DevelopmentBase):
    """Bootstrap sampler with calendar-year-correlated residuals.

    Implements the calendar-year correlation extension to the ODP bootstrap
    described in:

        Clark, D.R., Ding, H., and Zhou, L. (2022). "Making Bootstrap Reserve
        Ranges More Realistic." CAS E-Forum, Summer 2022.

    Cells on the same calendar-year diagonal have correlation ``rho``; the
    correlation decays multiplicatively for more distant diagonals
    (``rho``, ``rho^2``, ``rho^3``, ...). Correlation is induced via a
    Gaussian copula applied to either parametric (Normal/Lognormal) or
    nonparametric residual draws.

    Parameters
    ----------
    n_sims : int, default 1000
        Number of bootstrap simulations.
    n_periods : int, default -1
        Number of origin periods used in the LDF average; ``-1`` uses all.
    rho : float, default 0.0
        Same-diagonal correlation. ``0`` reproduces the standard independent
        bootstrap.
    parametric : bool, default True
        If True, parametric bootstrap (Normal or Lognormal multipliers on
        fitted values). If False, nonparametric (resamples residuals).
    parametric_dist : {"normal", "lognormal"}, default "normal"
        Parametric distribution choice when ``parametric=True``.
    hat_adj : bool, default True
        Apply Shapland's hat-matrix adjustment to standardised residuals.
    drop, drop_high, drop_low, drop_valuation
        Forwarded to ``chainladder.development.Development``.
    random_state : int or numpy.random.RandomState, optional
        Seed/state for reproducibility.
    min_fitted_value : float, default 1.0
        Floor on fitted incremental losses to keep residuals stable
        (``delta`` parameter, paper eq 2.1.4).

    Attributes
    ----------
    resampled_triangles_ : Triangle
        Bootstrap resamples (one per simulation).
    scale_ : float
        Dispersion (phi) for process risk.
    correlation_matrix_ : ndarray or None
        Calendar-year correlation matrix used by the Gaussian copula
        (``None`` when ``rho == 0``).
    cholesky_matrix_ : ndarray
        Cholesky factor of ``correlation_matrix_``.
    valid_indices_ : list[tuple[int, int]]
        ``(origin_idx, dev_idx)`` for each non-NaN cell.
    """

    def __init__(
        self,
        n_sims: int = 1000,
        n_periods: int = -1,
        rho: float = 0.0,
        parametric: bool = True,
        parametric_dist: str = "normal",
        hat_adj: bool = True,
        drop=None,
        drop_high=None,
        drop_low=None,
        drop_valuation=None,
        random_state=None,
        min_fitted_value: float = 1.0,
    ) -> None:
        if parametric_dist not in ("normal", "lognormal"):
            raise ValueError("parametric_dist must be 'normal' or 'lognormal'")
        if not (0.0 <= rho <= 1.0):
            raise ValueError(f"rho must be in [0, 1], got {rho}")
        self.n_sims = n_sims
        self.n_periods = n_periods
        self.rho = rho
        self.parametric = parametric
        self.parametric_dist = parametric_dist
        self.hat_adj = hat_adj
        self.drop = drop
        self.drop_high = drop_high
        self.drop_low = drop_low
        self.drop_valuation = drop_valuation
        self.random_state = random_state
        self.min_fitted_value = min_fitted_value

    # ----- correlation matrix construction -----

    def _build_full_correlation_matrix(self, n_origin, n_dev, nan_triangle, xp=np):
        valid_indices = []
        for i in range(n_origin):
            for j in range(n_dev):
                if not np.isnan(nan_triangle[i, j]):
                    valid_indices.append((i, j))

        n_cells = len(valid_indices)
        corr_matrix = xp.eye(n_cells)
        for idx1, (i1, j1) in enumerate(valid_indices):
            cy1 = i1 + j1
            for idx2, (i2, j2) in enumerate(valid_indices):
                if idx1 == idx2:
                    continue
                cy2 = i2 + j2
                cy_diff = abs(cy1 - cy2)
                if cy_diff == 0:
                    corr_matrix[idx1, idx2] = self.rho
                else:
                    corr_matrix[idx1, idx2] = self.rho ** (cy_diff + 1)
        return corr_matrix, valid_indices

    def _generate_correlated_uniforms(
        self, n_cells, n_sims, corr_matrix, random_state, xp=np
    ):
        try:
            L = cholesky(corr_matrix, lower=True)
        except np.linalg.LinAlgError:
            eps = 1e-6
            L = cholesky(corr_matrix + eps * np.eye(n_cells), lower=True)
        self.cholesky_matrix_ = L
        Z = random_state.standard_normal(size=(n_sims, n_cells))
        correlated_normals = Z @ L.T
        return stats.norm.cdf(correlated_normals)

    # ----- fit / transform -----

    def fit(self, X, y=None, sample_weight=None):
        if X.shape[1] > 1:
            from chainladder.utils.utility_functions import concat
            out = [
                CorrelatedBootstrapODPSample(**self.get_params()).fit(X.iloc[:, i])
                for i in range(X.shape[1])
            ]
            xp = X.get_array_module(out[0].design_matrix_)
            self.design_matrix_ = xp.concatenate(
                [i.design_matrix_[None] for i in out], axis=0
            )
            self.hat_ = xp.concatenate([i.hat_[None] for i in out], axis=0)
            self.resampled_triangles_ = concat(
                [i.resampled_triangles_ for i in out], axis=1
            )
            self.scale_ = xp.array([i.scale_ for i in out])
            self.w_ = out[0].w_
            self.correlation_matrix_ = out[0].correlation_matrix_
            return self

        backend = X.array_backend
        X = X.set_backend("numpy") if backend == "sparse" else X.copy()
        xp = X.get_array_module()

        if len(X) != 1:
            raise ValueError("Only single index triangles are supported")
        if not isinstance(X.ddims, np.ndarray):
            raise ValueError("Triangle must be expressed with development lags")

        obj = Development(
            n_periods=self.n_periods,
            drop=self.drop,
            drop_high=self.drop_high,
            drop_low=self.drop_low,
            drop_valuation=self.drop_valuation,
        ).fit_transform(X)
        self.w_ = obj.w_

        obj = Chainladder().fit(obj)
        exp_incr_triangle = obj.full_expectation_.cum_to_incr().values[
            0, 0, :, : X.shape[-1]
        ]
        exp_incr_triangle = xp.nan_to_num(exp_incr_triangle) * obj.X_.nan_triangle

        self.design_matrix_ = self._get_design_matrix(X)

        if self.hat_adj:
            try:
                self.hat_ = self._get_hat(X, exp_incr_triangle)
            except Exception:
                warn("Could not compute hat matrix. Setting hat_adj to False", stacklevel=2)
                self.hat_adj = False
                self.hat_ = None
        else:
            self.hat_ = None

        n_origin, n_dev = X.shape[2], X.shape[3]
        nan_triangle = obj.X_.nan_triangle

        if self.rho != 0:
            self.correlation_matrix_, self.valid_indices_ = (
                self._build_full_correlation_matrix(
                    n_origin, n_dev, nan_triangle, xp
                )
            )
        else:
            self.correlation_matrix_ = None
            self.valid_indices_ = None

        self.resampled_triangles_, self.scale_ = self._get_simulation(
            X, exp_incr_triangle, nan_triangle
        )
        return self

    def _get_simulation(self, X, exp_incr_triangle, nan_triangle):
        xp = X.get_array_module()
        fitted_for_resid = xp.maximum(xp.abs(exp_incr_triangle), self.min_fitted_value)
        unscaled_residuals = (
            (X.cum_to_incr().values - exp_incr_triangle) / xp.sqrt(fitted_for_resid)
        )[0, 0, ...]

        w_ = self.w_[0, 0]
        w_expanded = xp.ones_like(unscaled_residuals)
        w_expanded[:, 1:] = w_[:, :] * w_[:, :]
        unscaled_residuals = unscaled_residuals * w_expanded

        pearson_chi_sq = xp.nansum(unscaled_residuals ** 2)
        if self.hat_ is not None:
            standardized_residuals = self.hat_ * unscaled_residuals
        else:
            standardized_residuals = unscaled_residuals

        n_params = self.design_matrix_.shape[1]
        degree_freedom = xp.nansum(nan_triangle) - n_params
        scale_phi = pearson_chi_sq / degree_freedom

        resids_flat = standardized_residuals.flatten()
        adj_resid_dist = resids_flat[np.isfinite(resids_flat)]
        adj_resid_dist = adj_resid_dist[adj_resid_dist != 0]
        adj_resid_dist = adj_resid_dist - xp.mean(adj_resid_dist)

        if isinstance(self.random_state, np.random.RandomState):
            random_state = self.random_state
        else:
            random_state = np.random.RandomState(self.random_state)

        if self.rho != 0 and self.correlation_matrix_ is not None:
            resampled_triangles = self._generate_correlated_samples(
                X, exp_incr_triangle, nan_triangle, adj_resid_dist,
                scale_phi, random_state, xp,
            )
        else:
            resampled_triangles = self._generate_independent_samples(
                X, exp_incr_triangle, adj_resid_dist, random_state, xp,
            )

        obj = X.copy()
        obj.kdims = np.arange(self.n_sims)
        obj.values = resampled_triangles
        obj._set_slicers()
        return obj, scale_phi

    def _generate_independent_samples(
        self, X, exp_incr_triangle, adj_resid_dist, random_state, xp
    ):
        if self.parametric:
            return self._generate_parametric_samples(
                X, exp_incr_triangle, random_state, xp
            )

        resampled_residual = [
            (
                random_state.choice(
                    adj_resid_dist, size=exp_incr_triangle.shape, replace=True
                )
                * (exp_incr_triangle * 0 + 1)
            )[None, ...]
            for _ in range(self.n_sims)
        ]
        resampled_residual = xp.concatenate(tuple(resampled_residual), 0)
        b = xp.repeat(exp_incr_triangle[None, ...], self.n_sims, 0)
        resampled_incr = resampled_residual * xp.sqrt(xp.abs(b)) + b
        resampled_triangles = resampled_incr.cumsum(axis=2)
        return xp.swapaxes(resampled_triangles[None, ...], 0, 1)

    def _generate_parametric_samples(
        self, X, exp_incr_triangle, random_state, xp
    ):
        n_params = self.design_matrix_.shape[1]
        nan_triangle = X.nan_triangle
        degree_freedom = xp.nansum(nan_triangle) - n_params

        fitted_safe = xp.maximum(xp.abs(exp_incr_triangle), self.min_fitted_value)
        actual_incr = X.cum_to_incr().values[0, 0, ...]
        resid_sq = ((actual_incr - exp_incr_triangle) ** 2) / fitted_safe
        phi = xp.nansum(resid_sq) / degree_freedom

        std_dev = xp.sqrt(phi * fitted_safe)

        if self.parametric_dist == "normal":
            z = random_state.standard_normal(
                size=(self.n_sims,) + exp_incr_triangle.shape
            )
            resampled_incr = exp_incr_triangle + std_dev * z
        else:
            cv = std_dev / fitted_safe
            sigma_sq = xp.log(1 + cv ** 2)
            mu = -sigma_sq / 2
            sigma = xp.sqrt(sigma_sq)
            z = random_state.standard_normal(
                size=(self.n_sims,) + exp_incr_triangle.shape
            )
            multipliers = xp.exp(mu + sigma * z)
            resampled_incr = exp_incr_triangle * multipliers

        resampled_triangles = resampled_incr.cumsum(axis=2)
        return xp.swapaxes(resampled_triangles[None, ...], 0, 1)

    def _generate_correlated_samples(
        self, X, exp_incr_triangle, nan_triangle, adj_resid_dist, scale_phi,
        random_state, xp,
    ):
        n_cells = len(self.valid_indices_)
        correlated_uniforms = self._generate_correlated_uniforms(
            n_cells, self.n_sims, self.correlation_matrix_, random_state, xp
        )
        if self.parametric:
            return self._generate_correlated_parametric(
                X, exp_incr_triangle, nan_triangle, correlated_uniforms,
                scale_phi, random_state, xp,
            )
        return self._generate_correlated_nonparametric(
            X, exp_incr_triangle, nan_triangle, correlated_uniforms,
            adj_resid_dist, random_state, xp,
        )

    def _generate_correlated_parametric(
        self, X, exp_incr_triangle, nan_triangle, correlated_uniforms,
        scale_phi, random_state, xp,
    ):
        n_origin, n_dev = exp_incr_triangle.shape
        n_params = self.design_matrix_.shape[1]
        degree_freedom = xp.nansum(nan_triangle) - n_params
        fitted_safe = xp.maximum(xp.abs(exp_incr_triangle), self.min_fitted_value)
        actual_incr = X.cum_to_incr().values[0, 0, ...]
        resid_sq = ((actual_incr - exp_incr_triangle) ** 2) / fitted_safe
        phi = xp.nansum(resid_sq) / degree_freedom

        resampled_incr = xp.zeros((self.n_sims, n_origin, n_dev))
        for cell_idx, (i, j) in enumerate(self.valid_indices_):
            fitted_val = fitted_safe[i, j]
            std_dev = xp.sqrt(phi * fitted_val)
            z = stats.norm.ppf(correlated_uniforms[:, cell_idx])
            if self.parametric_dist == "normal":
                resampled_incr[:, i, j] = exp_incr_triangle[i, j] + std_dev * z
            else:
                cv = std_dev / fitted_val
                sigma_sq = xp.log(1 + cv ** 2)
                mu = -sigma_sq / 2
                sigma = xp.sqrt(sigma_sq)
                multiplier = xp.exp(mu + sigma * z)
                resampled_incr[:, i, j] = exp_incr_triangle[i, j] * multiplier

        for i in range(n_origin):
            for j in range(n_dev):
                if (i, j) not in self.valid_indices_:
                    resampled_incr[:, i, j] = xp.nan

        resampled_triangles = xp.cumsum(resampled_incr, axis=2)
        return xp.swapaxes(resampled_triangles[None, ...], 0, 1)

    def _generate_correlated_nonparametric(
        self, X, exp_incr_triangle, nan_triangle, correlated_uniforms,
        adj_resid_dist, random_state, xp,
    ):
        n_origin, n_dev = exp_incr_triangle.shape
        sorted_resids = xp.sort(adj_resid_dist)
        n_resids = len(sorted_resids)

        resampled_incr = xp.zeros((self.n_sims, n_origin, n_dev))
        for cell_idx, (i, j) in enumerate(self.valid_indices_):
            indices = (correlated_uniforms[:, cell_idx] * n_resids).astype(int)
            indices = xp.clip(indices, 0, n_resids - 1)
            selected_resids = sorted_resids[indices]
            fitted_val = exp_incr_triangle[i, j]
            resampled_incr[:, i, j] = (
                selected_resids * xp.sqrt(xp.abs(fitted_val)) + fitted_val
            )

        for i in range(n_origin):
            for j in range(n_dev):
                if (i, j) not in self.valid_indices_:
                    resampled_incr[:, i, j] = xp.nan

        resampled_triangles = xp.cumsum(resampled_incr, axis=2)
        return xp.swapaxes(resampled_triangles[None, ...], 0, 1)

    # ----- design / hat matrix -----

    def _get_design_matrix(self, X):
        xp = X.get_array_module()
        w = X.nan_triangle
        arr = xp.diag(w[:, 0])
        intra_beta = xp.zeros((w.shape[0], w.shape[1] - 1))
        arr = xp.concatenate((arr, intra_beta), axis=1)
        for i in range(w.shape[1] - 1):
            len_alpha = len(w[:, i + 1][~xp.isnan(w[:, i + 1])])
            intra_alpha = xp.diag(w[:, i + 1])[:len_alpha, :]
            intra_beta[:, i] = 1
            intra_beta = intra_beta[:len_alpha, :]
            intra_arr = xp.concatenate((intra_alpha, intra_beta), axis=1)
            arr = xp.concatenate((arr, intra_arr), axis=0)
        return arr

    def _get_hat(self, X, exp_incr_triangle):
        xp = X.get_array_module()
        weight_matrix = xp.diag(
            pd.DataFrame(exp_incr_triangle).unstack().dropna().values
        )
        design_matrix = self.design_matrix_
        hat = xp.matmul(
            xp.matmul(
                xp.matmul(
                    design_matrix,
                    xp.linalg.inv(
                        xp.matmul(
                            design_matrix.T,
                            xp.matmul(weight_matrix, design_matrix),
                        )
                    ),
                ),
                design_matrix.T,
            ),
            weight_matrix,
        )
        hat = xp.diagonal(
            xp.sqrt(
                xp.divide(
                    1,
                    abs(1 - hat),
                    where=(1 - hat) != 0,
                    out=xp.zeros_like(hat),
                )
            )
        )
        total_length = X.nan_triangle.shape[0]
        reshaped_hat = xp.reshape(hat[:total_length], (1, total_length))
        indices = xp.nansum(X.nan_triangle, axis=0).cumsum().astype(int)
        for num, _ in enumerate(indices[:-1]):
            col_length = int(indices[num + 1] - indices[num])
            col = xp.reshape(
                hat[int(indices[num]) : int(indices[num + 1])], (1, col_length)
            )
            nans = xp.repeat(
                xp.array([xp.nan])[None, :], total_length - col_length, axis=1
            )
            col = xp.concatenate((col, nans), axis=1)
            reshaped_hat = xp.concatenate((reshaped_hat, col), axis=0)
        return reshaped_hat.T

    def transform(self, X):
        X_new = self.resampled_triangles_.copy()
        n_keys = len(X.key_labels)
        if n_keys == 1:
            X_new.kdims = np.array([[str(i)] for i in range(self.n_sims)])
        else:
            original_kdims = (
                X.kdims[0] if len(X.kdims.shape) > 1 else X.kdims
            )
            X_new.kdims = np.array(
                [
                    [
                        f"{original_kdims[j]}_{i}" if j == 0 else original_kdims[j]
                        for j in range(n_keys)
                    ]
                    for i in range(self.n_sims)
                ]
            )
        X_new.key_labels = X.key_labels
        X_new.scale_ = self.scale_
        X_new.random_state = self.random_state
        X_new.rho_ = self.rho
        X_new._get_process_variance = types.MethodType(_get_process_variance, X_new)
        return X_new


class CorrelatedBootstrapChainLadder(BaseStochasticReserve):
    """Correlated ODP bootstrap chain ladder behind the shared interface.

    Wraps :class:`CorrelatedBootstrapODPSample` (calendar-year-correlated
    bootstrap, Clark/Ding/Zhou 2022) followed by
    ``chainladder.Chainladder``. Identical surface to
    :class:`BootstrapODPChainLadder`, plus correlation parameters.

    Parameters
    ----------
    n_sims : int, default 1000
    rho : float, default 0.0
        Same-diagonal correlation (``0`` reduces to independent ODP bootstrap).
    parametric : bool, default True
    parametric_dist : {"normal", "lognormal"}, default "normal"
    hat_adj : bool, default True
    n_periods : int, default -1
    random_seed : int, optional

    Notes
    -----
    With ``rho > 0`` the mean total reserve closely tracks the deterministic
    chain-ladder mean (within Monte Carlo noise on standard test triangles),
    while the standard deviation increases with ``rho`` as calendar-year
    correlation amplifies dispersion. This matches the intent of the
    Clark/Ding/Zhou (2022) correlated bootstrap: same point estimate as
    standard ODP, wider tails reflecting calendar-year shocks.
    """

    def __init__(
        self,
        n_sims: int = 1000,
        rho: float = 0.0,
        parametric: bool = True,
        parametric_dist: str = "normal",
        hat_adj: bool = True,
        n_periods: int = -1,
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.n_sims = n_sims
        self.rho = rho
        self.parametric = parametric
        self.parametric_dist = parametric_dist
        self.hat_adj = hat_adj
        self.n_periods = n_periods
        self.random_seed = random_seed

    def fit(self, triangle):
        validate_triangle(triangle)
        self.triangle_ = triangle.copy()

        prepared = triangle.copy()
        prepared.key_labels = ["triangle_id"]
        prepared.kdims = np.asarray([["resample"]], dtype=object)

        sampler = CorrelatedBootstrapODPSample(
            n_sims=self.n_sims,
            rho=self.rho,
            parametric=self.parametric,
            parametric_dist=self.parametric_dist,
            hat_adj=self.hat_adj,
            n_periods=self.n_periods,
            random_state=self.random_seed,
        ).fit(prepared)
        resampled = sampler.transform(prepared)
        model = Chainladder().fit(resampled)

        ibnr_vals = np.asarray(model.ibnr_.values)
        per_sim_per_origin = np.nansum(ibnr_vals, axis=-1)  # (n_sims, 1, n_origin)
        per_sim_per_origin = np.squeeze(per_sim_per_origin, axis=1)  # (n_sims, n_origin)
        per_origin_per_sim = per_sim_per_origin.T  # (n_origin, n_sims)

        origins = origin_labels(triangle)

        self.reserves_posterior_ = xr.DataArray(
            per_origin_per_sim,
            dims=["origin", "sample"],
            coords={
                "origin": origins,
                "sample": np.arange(per_origin_per_sim.shape[1]),
            },
        )

        self._build_reserve_summaries()
        self._is_fitted = True
        return self


class CorrelatedBootstrapODPBornhuetterFerguson(BaseStochasticReserve):
    """Correlated ODP bootstrap Bornhuetter-Ferguson behind the shared interface.

    Wraps :class:`CorrelatedBootstrapODPSample` (Clark/Ding/Zhou 2022) to
    generate calendar-year-correlated resamples, then fits
    ``chainladder.BornhuetterFerguson(apriori=...)`` to each resample.

    Parameters
    ----------
    apriori : float or chainladder.Triangle
        Expected loss ratio. Passed to ``chainladder.BornhuetterFerguson``.
    apriori_sigma : float, default 0.0
        Uncertainty on the a-priori.
    n_sims : int, default 1000
    rho : float, default 0.0
        Same-diagonal correlation. ``0`` reproduces the independent bootstrap.
    parametric : bool, default True
    parametric_dist : {"normal", "lognormal"}, default "normal"
    hat_adj : bool, default True
    n_periods : int, default -1
    random_seed : int, optional

    Notes
    -----
    An ``exposure_triangle`` (premium) is **required** by B-F and must be
    supplied to :meth:`fit`.
    """

    def __init__(
        self,
        apriori: float | object = 0.65,
        apriori_sigma: float = 0.0,
        n_sims: int = 1000,
        rho: float = 0.0,
        parametric: bool = True,
        parametric_dist: str = "normal",
        hat_adj: bool = True,
        n_periods: int = -1,
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.apriori = apriori
        self.apriori_sigma = apriori_sigma
        self.n_sims = n_sims
        self.rho = rho
        self.parametric = parametric
        self.parametric_dist = parametric_dist
        self.hat_adj = hat_adj
        self.n_periods = n_periods
        self.random_seed = random_seed

    def fit(self, triangle, exposure_triangle=None):
        """Fit the correlated ODP bootstrap B-F model.

        Parameters
        ----------
        triangle : chainladder.Triangle
            Paid-loss triangle.
        exposure_triangle : chainladder.Triangle
            Premium triangle per origin, shape ``(1, 1, n_origin, 1)``.
            Required for Bornhuetter-Ferguson.
        """
        if exposure_triangle is None:
            raise ValueError(
                "exposure_triangle is required for "
                "CorrelatedBootstrapODPBornhuetterFerguson. "
                "Pass the .latest_diagonal of a premium triangle."
            )
        validate_triangle(triangle)
        self.triangle_ = triangle.copy()
        exposure_triangle = _build_exposure_triangle(
            triangle, exposure_triangle, self.n_sims
        )

        prepared = triangle.copy()
        prepared.key_labels = ["triangle_id"]
        prepared.kdims = np.asarray([["resample"]], dtype=object)

        sampler = CorrelatedBootstrapODPSample(
            n_sims=self.n_sims,
            rho=self.rho,
            parametric=self.parametric,
            parametric_dist=self.parametric_dist,
            hat_adj=self.hat_adj,
            n_periods=self.n_periods,
            random_state=self.random_seed,
        ).fit(prepared)
        resampled = sampler.transform(prepared)

        bf = cl.BornhuetterFerguson(
            apriori=self.apriori,
            apriori_sigma=self.apriori_sigma,
        ).fit(resampled, sample_weight=exposure_triangle)

        self.reserves_posterior_ = _extract_ibnr_from_bf_or_cc(bf, triangle)
        self._build_reserve_summaries()
        self._is_fitted = True
        return self


class CorrelatedBootstrapODPCapeCod(BaseStochasticReserve):
    """Correlated ODP bootstrap Cape Cod behind the shared interface.

    Wraps :class:`CorrelatedBootstrapODPSample` (Clark/Ding/Zhou 2022) to
    generate calendar-year-correlated resamples, then fits
    ``chainladder.CapeCod(trend=..., decay=...)`` to each resample.

    Parameters
    ----------
    trend : float, default 0.0
        Annual trend assumption for the Cape Cod method.
    decay : float, default 1.0
        Decay factor for the Cape Cod method.
    n_sims : int, default 1000
    rho : float, default 0.0
        Same-diagonal correlation. ``0`` reproduces the independent bootstrap.
    parametric : bool, default True
    parametric_dist : {"normal", "lognormal"}, default "normal"
    hat_adj : bool, default True
    n_periods : int, default -1
    random_seed : int, optional

    Notes
    -----
    An ``exposure_triangle`` (premium) is **required** by Cape Cod and must be
    supplied to :meth:`fit`.
    """

    def __init__(
        self,
        trend: float = 0.0,
        decay: float = 1.0,
        n_sims: int = 1000,
        rho: float = 0.0,
        parametric: bool = True,
        parametric_dist: str = "normal",
        hat_adj: bool = True,
        n_periods: int = -1,
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.trend = trend
        self.decay = decay
        self.n_sims = n_sims
        self.rho = rho
        self.parametric = parametric
        self.parametric_dist = parametric_dist
        self.hat_adj = hat_adj
        self.n_periods = n_periods
        self.random_seed = random_seed

    def fit(self, triangle, exposure_triangle=None):
        """Fit the correlated ODP bootstrap Cape Cod model.

        Parameters
        ----------
        triangle : chainladder.Triangle
            Paid-loss triangle.
        exposure_triangle : chainladder.Triangle
            Premium triangle per origin, shape ``(1, 1, n_origin, 1)``.
            Required for Cape Cod.
        """
        if exposure_triangle is None:
            raise ValueError(
                "exposure_triangle is required for CorrelatedBootstrapODPCapeCod. "
                "Pass the .latest_diagonal of a premium triangle."
            )
        validate_triangle(triangle)
        self.triangle_ = triangle.copy()
        exposure_triangle = _build_exposure_triangle(
            triangle, exposure_triangle, self.n_sims
        )

        prepared = triangle.copy()
        prepared.key_labels = ["triangle_id"]
        prepared.kdims = np.asarray([["resample"]], dtype=object)

        sampler = CorrelatedBootstrapODPSample(
            n_sims=self.n_sims,
            rho=self.rho,
            parametric=self.parametric,
            parametric_dist=self.parametric_dist,
            hat_adj=self.hat_adj,
            n_periods=self.n_periods,
            random_state=self.random_seed,
        ).fit(prepared)
        resampled = sampler.transform(prepared)

        cc = cl.CapeCod(trend=self.trend, decay=self.decay).fit(
            resampled, sample_weight=exposure_triangle
        )

        self.reserves_posterior_ = _extract_ibnr_from_bf_or_cc(cc, triangle)
        self._build_reserve_summaries()
        self._is_fitted = True
        return self


def _get_process_variance(self, full_triangle):
    """Inject random gamma process noise into the lower-right (future) cells."""
    xp = full_triangle.get_array_module()
    lower_tri = full_triangle.cum_to_incr() - self.cum_to_incr()
    random_state = xp.random.RandomState(
        None if not self.random_state else self.random_state + 1
    )
    lower_tri.values = random_state.gamma(
        shape=abs(lower_tri.values) / self.scale_, scale=self.scale_
    ) * xp.sign(xp.nan_to_num(lower_tri.values))
    return (lower_tri + self.cum_to_incr()).incr_to_cum()
