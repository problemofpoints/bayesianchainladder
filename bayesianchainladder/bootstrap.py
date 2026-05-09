"""Stochastic reserve wrappers around chainladder bootstrap/Mack methods.

This module provides three wrapper estimators that share the
:class:`bayesianchainladder.base.BaseStochasticReserve` interface:

- :class:`MackChainLadder`: wraps ``chainladder.MackChainladder`` and exposes
  reserve uncertainty as a normal approximation calibrated to Mack's
  ``total_mack_std_err_``.
- :class:`BootstrapODPChainLadder`: wraps ``chainladder.BootstrapODPSample``
  + ``chainladder.Chainladder`` and exposes the bootstrap reserve samples
  directly.
- :class:`CorrelatedBootstrapChainLadder`: wraps
  :class:`CorrelatedBootstrapODPSample` (Clark/Ding/Zhou 2022) — the same
  bootstrap with calendar-year correlation between cells via a Gaussian
  copula.

The low-level :class:`CorrelatedBootstrapODPSample` lives here too so the
file is self-contained.
"""

from __future__ import annotations

import chainladder as cl
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats as st

from .base import BaseStochasticReserve, MethodSummary
from .utils import _extract_period_value, validate_triangle


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

        # Origins as integers, aligned with the per-origin arrays
        origins = [_extract_period_value(o) for o in ibnr_tri.origin]

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
        q75, q90, q95 = st.norm.ppf([0.75, 0.90, 0.95], loc=mean, scale=stddev)
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
        q05, q25, median, q75, q95 = st.norm.ppf(
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
