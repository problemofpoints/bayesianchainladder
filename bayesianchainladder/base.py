"""Shared abstract base class and dataclass for stochastic reserve estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    import chainladder as cl


@dataclass(frozen=True)
class MethodSummary:
    """Total-reserve summary returned by ``BaseStochasticReserve.total_summary``."""

    total_reserve_mean: float
    total_reserve_stddev: float
    total_reserve_75th_percentile: float
    total_reserve_90th_percentile: float
    total_reserve_95th_percentile: float

    @property
    def total_reserve_cv(self) -> float:
        if self.total_reserve_mean == 0 or not np.isfinite(self.total_reserve_mean):
            return float("nan")
        return self.total_reserve_stddev / abs(self.total_reserve_mean)


class BaseStochasticReserve(ABC):
    """Abstract base for stochastic reserve estimators.

    Subclass contract
    -----------------
    Subclasses implement ``fit(triangle, ...)``. During ``fit`` they MUST:
      1. Validate the input triangle.
      2. Set ``self.triangle_`` to a copy of the input.
      3. Populate ``self.reserves_posterior_`` as an ``xr.DataArray`` with an
         ``origin`` dimension and one or more sample dimension(s). The
         remaining dim(s) are flattened automatically when computing
         summaries.
      4. Call ``self._build_reserve_summaries()`` to populate ``ibnr_`` and
         ``ultimate_``.
      5. Set ``self._is_fitted = True``.

    Subclasses MAY override ``sample_reserves`` and ``total_summary`` if they
    need to substitute a calibrated total (e.g. Mack's ``total_mack_std_err_``).
    """

    triangle_: cl.Triangle | None
    ibnr_: pd.DataFrame | None
    ultimate_: pd.DataFrame | None
    reserves_posterior_: xr.DataArray | None
    _is_fitted: bool

    def __init__(self) -> None:
        self.triangle_ = None
        self.ibnr_ = None
        self.ultimate_ = None
        self.reserves_posterior_ = None
        self._is_fitted = False

    @abstractmethod
    def fit(self, triangle, **kwargs: Any) -> BaseStochasticReserve:
        """Fit the estimator to a triangle. Subclasses implement this."""

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise ValueError(
                "Model has not been fitted. Call fit() before using this method."
            )

    def _paid_to_date(self) -> pd.Series:
        """Per-origin paid-to-date totals from ``triangle_``."""
        from .utils import triangle_to_dataframe

        if self.triangle_ is None:
            raise ValueError("triangle_ is not set")
        df = triangle_to_dataframe(self.triangle_)
        return df.groupby("origin", observed=True)["incremental"].sum()

    def _build_reserve_summaries(self) -> None:
        """Populate ``ibnr_`` and ``ultimate_`` from ``reserves_posterior_``.

        Subclasses populate ``reserves_posterior_`` during ``fit`` then call
        this helper. They do not override it.
        """
        if self.reserves_posterior_ is None:
            raise ValueError("reserves_posterior_ must be populated first")

        paid = self._paid_to_date()
        origins = list(self.reserves_posterior_.coords["origin"].values)

        rows = []
        for origin in origins:
            samples = self.reserves_posterior_.sel(origin=origin).values.flatten()
            samples = samples[np.isfinite(samples)]
            paid_origin = float(paid.get(origin, 0.0))

            if samples.size == 0:
                ibnr_mean = ibnr_std = ibnr_median = 0.0
                ibnr_q05 = ibnr_q25 = ibnr_q75 = ibnr_q95 = 0.0
            else:
                ibnr_mean = float(np.mean(samples))
                ibnr_std = float(np.std(samples, ddof=1)) if samples.size > 1 else 0.0
                ibnr_median = float(np.median(samples))
                ibnr_q05 = float(np.percentile(samples, 5))
                ibnr_q25 = float(np.percentile(samples, 25))
                ibnr_q75 = float(np.percentile(samples, 75))
                ibnr_q95 = float(np.percentile(samples, 95))

            rows.append({
                "origin": origin,
                "paid_to_date": paid_origin,
                "ibnr_mean": ibnr_mean,
                "ibnr_std": ibnr_std,
                "ibnr_median": ibnr_median,
                "ibnr_5%": ibnr_q05,
                "ibnr_25%": ibnr_q25,
                "ibnr_75%": ibnr_q75,
                "ibnr_95%": ibnr_q95,
                "ultimate_mean": paid_origin + ibnr_mean,
                "ultimate_std": ibnr_std,
                "ultimate_median": paid_origin + ibnr_median,
                "ultimate_5%": paid_origin + ibnr_q05,
                "ultimate_25%": paid_origin + ibnr_q25,
                "ultimate_75%": paid_origin + ibnr_q75,
                "ultimate_95%": paid_origin + ibnr_q95,
            })

        df = pd.DataFrame(rows).set_index("origin")
        self.ibnr_ = df[
            ["ibnr_mean", "ibnr_std", "ibnr_median",
             "ibnr_5%", "ibnr_25%", "ibnr_75%", "ibnr_95%"]
        ].copy()
        self.ibnr_.columns = ["mean", "std", "median", "5%", "25%", "75%", "95%"]

        self.ultimate_ = df[
            ["paid_to_date", "ultimate_mean", "ultimate_std", "ultimate_median",
             "ultimate_5%", "ultimate_25%", "ultimate_75%", "ultimate_95%"]
        ].copy()
        self.ultimate_.columns = [
            "paid_to_date", "mean", "std", "median", "5%", "25%", "75%", "95%"
        ]

    def summary(self, include_totals: bool = True) -> pd.DataFrame:
        self._check_is_fitted()
        if self.ultimate_ is None or self.ibnr_ is None:
            raise ValueError("Reserve summaries are not available")

        result = pd.concat(
            [
                self.ultimate_[["paid_to_date", "mean", "std", "median"]],
                self.ibnr_[["mean", "std", "median"]],
            ],
            axis=1,
            keys=["Ultimate", "IBNR"],
        )

        if include_totals:
            total_paid = float(self.ultimate_["paid_to_date"].sum())
            total = self.reserves_posterior_.sum(dim="origin").values.flatten()
            total = total[np.isfinite(total)]
            total_ibnr_mean = float(np.mean(total)) if total.size else 0.0
            total_ibnr_std = float(np.std(total, ddof=1)) if total.size > 1 else 0.0
            total_ibnr_median = float(np.median(total)) if total.size else 0.0

            total_row = pd.DataFrame(
                {
                    ("Ultimate", "paid_to_date"): [total_paid],
                    ("Ultimate", "mean"): [total_paid + total_ibnr_mean],
                    ("Ultimate", "std"): [total_ibnr_std],
                    ("Ultimate", "median"): [total_paid + total_ibnr_median],
                    ("IBNR", "mean"): [total_ibnr_mean],
                    ("IBNR", "std"): [total_ibnr_std],
                    ("IBNR", "median"): [total_ibnr_median],
                },
                index=["Total"],
            )
            result = pd.concat([result, total_row])

        return result

    def sample_reserves(
        self,
        n_samples: int = 1000,
        random_seed: int | None = None,
    ) -> np.ndarray:
        self._check_is_fitted()
        if self.reserves_posterior_ is None:
            raise ValueError("No reserve posterior available")
        total = self.reserves_posterior_.sum(dim="origin").values.flatten()
        total = total[np.isfinite(total)]
        if total.size == 0:
            return np.full(n_samples, np.nan)
        rng = np.random.default_rng(random_seed)
        replace = n_samples > total.size
        idx = rng.choice(total.size, size=n_samples, replace=replace)
        return total[idx]

    def total_summary(self) -> MethodSummary:
        self._check_is_fitted()
        if self.reserves_posterior_ is None:
            raise ValueError("No reserve posterior available")
        total = self.reserves_posterior_.sum(dim="origin").values.flatten()
        total = total[np.isfinite(total)]
        if total.size == 0:
            return MethodSummary(
                total_reserve_mean=float("nan"),
                total_reserve_stddev=float("nan"),
                total_reserve_75th_percentile=float("nan"),
                total_reserve_90th_percentile=float("nan"),
                total_reserve_95th_percentile=float("nan"),
            )
        return MethodSummary(
            total_reserve_mean=float(np.mean(total)),
            total_reserve_stddev=float(np.std(total, ddof=1)) if total.size > 1 else 0.0,
            total_reserve_75th_percentile=float(np.quantile(total, 0.75)),
            total_reserve_90th_percentile=float(np.quantile(total, 0.90)),
            total_reserve_95th_percentile=float(np.quantile(total, 0.95)),
        )
