"""One-year and multi-year Claims Development Result (CDR).

Implements the "actuary-in-the-box" re-reserving of England, Verrall &
Wuthrich (2019) as coded in ``CDR_Full_Picture`` / ``CDR_Rev_Sum`` of Peter
England's StochasticReserving repository
(https://github.com/DrPeterEngland/StochasticReserving, MIT licence). Works on
any estimator exposing ``full_cumulative_posterior_``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from ._triangle_ops import (
    DropList,
    cumulative_array,
    drop_mask,
    link_ratio_mask,
    project_cumulative,
    volume_weighted_factors,
)
from .base import BaseStochasticReserve
from .riskmeasures import value_at_risk


@dataclass
class CDRResult:
    cdr: xr.DataArray
    total_cdr: xr.DataArray
    ultimates: xr.DataArray
    var_level: float

    def cumulative(self) -> xr.DataArray:
        return self.cdr.cumsum("future_period")

    def reverse_cumulative(self) -> xr.DataArray:
        """Sum of CDRs from each future period to run-off (England's CDR_Rev_Sum)."""
        vals = np.flip(np.cumsum(np.flip(self.cdr.values, axis=0), axis=0), axis=0)
        return self.cdr.copy(data=vals)

    def summary(self) -> pd.DataFrame:
        """Long-format table of mean, SD and VaR of the CDR per future period
        and origin (plus 'Total'). ``var`` follows England: mean minus the
        ``1 - var_level`` quantile of the CDR, i.e. the capital needed against
        an adverse one-year development at the chosen confidence."""
        rows = []
        periods = self.cdr.coords["future_period"].values
        origins = list(self.cdr.coords["origin"].values) + ["Total"]
        for t_idx, t in enumerate(periods):
            for origin in origins:
                if origin == "Total":
                    x = self.total_cdr.isel(future_period=t_idx).values
                else:
                    x = self.cdr.isel(future_period=t_idx).sel(origin=origin).values
                mean = float(np.mean(x))
                sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
                var = mean - value_at_risk(x, 1.0 - self.var_level)
                rows.append(
                    {
                        "future_period": int(t),
                        "origin": origin,
                        "mean": mean,
                        "sd": sd,
                        "var": var,
                    }
                )
        return pd.DataFrame(rows)


def claims_development_result(
    model: BaseStochasticReserve,
    future_periods: int | None = None,
    var_level: float = 0.995,
    drop: DropList = None,
) -> CDRResult:
    full = model._require_full_posterior()
    C = np.moveaxis(full.values, -1, 0)  # (S, n_o, n_d)
    cum, origins, devs = cumulative_array(model.triangle_)
    n_o, n_d = cum.shape
    if n_o != n_d:
        raise ValueError("claims_development_result requires a square triangle")
    if np.isnan(C).any():
        raise ValueError("full_cumulative_posterior_ contains NaN cells")

    excl = drop_mask(n_o, n_d, drop, origins, devs)
    max_periods = n_d - 1
    n_periods = (
        max_periods if future_periods is None else min(int(future_periods), max_periods)
    )
    if n_periods < 1:
        raise ValueError("future_periods must be >= 1")

    i_idx, j_idx = np.indices((n_o, n_d))
    n_sims = C.shape[0]

    # Period 0: deterministic chain ladder on the observed data.
    f0 = volume_weighted_factors(cum, link_ratio_mask(cum, drop, origins, devs))
    u0 = project_cumulative(cum, f0)[:, -1]
    ultimates = [np.broadcast_to(u0, (n_sims, n_o)).copy()]

    for t in range(1, n_periods + 1):
        known = (i_idx + j_idx) <= (n_d - 1) + t
        avail = (known[:, :-1] & known[:, 1:]).astype(float)
        mask_t = avail * excl
        c_known = np.where(known, C, np.nan)
        f_t = volume_weighted_factors(c_known, mask_t)
        ultimates.append(project_cumulative(c_known, f_t)[:, :, -1])

    U = np.stack(ultimates)  # (n_periods + 1, S, n_o)
    cdr = U[:-1] - U[1:]  # (n_periods, S, n_o)

    coords_common = {"origin": full.coords["origin"], "sample": full.coords["sample"]}
    cdr_da = xr.DataArray(
        np.moveaxis(cdr, 1, 2),
        dims=["future_period", "origin", "sample"],
        coords={"future_period": np.arange(1, n_periods + 1), **coords_common},
    )
    ult_da = xr.DataArray(
        np.moveaxis(U, 1, 2),
        dims=["future_period", "origin", "sample"],
        coords={"future_period": np.arange(0, n_periods + 1), **coords_common},
    )
    return CDRResult(
        cdr=cdr_da,
        total_cdr=cdr_da.sum("origin"),
        ultimates=ult_da,
        var_level=var_level,
    )
