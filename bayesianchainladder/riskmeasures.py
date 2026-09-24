"""Risk measures, discounting and cost-of-capital risk margins for reserve
distributions.

Ported from the ``VAR``, ``TVAR``, ``PHT``, ``Disc_Reserves``,
``Disc_Future_Reserves``, ``Capital_Profile`` and ``CoC_RM`` functions in
Peter England's StochasticReserving repository
(https://github.com/DrPeterEngland/StochasticReserving, MIT licence), which
reproduce England, Verrall & Wüthrich (2019). Quantiles use ``np.quantile``
rather than the original order-statistic index.

Timing convention: for a square triangle whose origin grain equals its
development grain, the cell ``(i, j)`` is paid ``k = i + j - (n - 1)`` periods
after the valuation date (``k <= 0`` is observed). A payment ``k`` periods
ahead is discounted by ``(1 + rate) ** -(k - 1 + offset)``; ``offset = 0.5``
is mid-period payment, ``offset = 1`` is payment in arrears.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.optimize import brentq

from .base import BaseStochasticReserve


def value_at_risk(samples, level: float) -> float:
    return float(np.quantile(np.asarray(samples, dtype=float), level))


def tail_value_at_risk(samples, level: float) -> float:
    x = np.asarray(samples, dtype=float)
    threshold = value_at_risk(x, level)
    return float(x[x >= threshold].mean())


def proportional_hazards_transform(samples, param: float) -> float:
    """Wang's proportional hazards transform E*[X] with S*(x) = S(x)^(1/param)."""
    if param < 1:
        raise ValueError("param must be >= 1 (param == 1 returns the mean)")
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)
    survival = (1.0 - np.arange(1, n + 1) / n) ** (1.0 / param)
    weights = np.concatenate([[1.0], survival[:-1]]) - survival
    return float(np.sum(weights * x))


def cash_flow_periods(n_origin: int, n_dev: int) -> np.ndarray:
    if n_origin != n_dev:
        raise ValueError(
            "cash-flow timing requires a square triangle (origin grain == development grain)"
        )
    i, j = np.indices((n_origin, n_dev))
    return i + j - (n_origin - 1)


def discount_factors(periods_ahead, rate: float, offset: float = 0.5) -> np.ndarray:
    k = np.asarray(periods_ahead, dtype=float)
    return (1.0 + rate) ** -(k - 1.0 + offset)


def _future_cash_flows(model: BaseStochasticReserve):
    incr = model.future_incremental_posterior()  # (origin, dev, sample), 0 on observed
    k = cash_flow_periods(incr.sizes["origin"], incr.sizes["dev"])
    return incr, k


def discounted_reserves(
    model: BaseStochasticReserve,
    rate: float,
    offset: float = 0.5,
    as_of_period: int = 0,
) -> xr.DataArray:
    """Per-origin discounted outstanding reserves as at ``as_of_period``
    (0 = valuation date), discounted back to that date only."""
    incr, k = _future_cash_flows(model)
    future = k > as_of_period
    factors = np.where(future, discount_factors(k - as_of_period, rate, offset), 0.0)
    vals = np.nansum(incr.values * factors[..., None], axis=1)
    return xr.DataArray(
        vals,
        dims=["origin", "sample"],
        coords={"origin": incr.coords["origin"], "sample": incr.coords["sample"]},
    )


def future_reserve_profile(
    model: BaseStochasticReserve, rate: float, offset: float = 0.5
) -> xr.DataArray:
    """Total discounted reserves remaining at the start of each future period
    ``t = 0 … n_dev-2`` (England's ``Disc_Future_Reserves``), per sample."""
    incr, _ = _future_cash_flows(model)
    n_periods = incr.sizes["dev"] - 1
    rows = [
        discounted_reserves(model, rate, offset, as_of_period=t).sum("origin").values
        for t in range(n_periods)
    ]
    return xr.DataArray(
        np.stack(rows),
        dims=["period", "sample"],
        coords={"period": np.arange(n_periods), "sample": incr.coords["sample"]},
    )


def capital_profile(basis) -> np.ndarray:
    b = np.asarray(basis, dtype=float)
    return b / b[0]


def cost_of_capital_risk_margin(
    opening_capital: float,
    profile,
    coc_rate: float,
    discount_rate: float,
    offset: float = 1.0,
) -> dict:
    """Risk margin = sum over future periods of capital × cost-of-capital rate,
    discounted with exponent ``t - 1 + min(offset, 1)`` (offset 1 = arrears)."""
    prof = np.asarray(profile, dtype=float)
    t = np.arange(1, len(prof) + 1)
    capital = opening_capital * prof
    cost = capital * coc_rate
    discounted = cost / (1.0 + discount_rate) ** (t - 1 + min(offset, 1.0))
    return {
        "capital": capital,
        "cost": cost,
        "discounted_cost": discounted,
        "risk_margin": float(discounted.sum()),
    }


def equivalent_risk_tolerance(
    samples, target_margin: float, measure: str = "var"
) -> float:
    """Solve for the confidence level (VaR/TVaR) or PHT parameter whose risk
    measure minus the mean equals ``target_margin``."""
    x = np.asarray(samples, dtype=float)
    mean = x.mean()
    if measure == "var":
        return float(
            brentq(lambda p: value_at_risk(x, p) - mean - target_margin, 0.01, 0.9999)
        )
    if measure == "tvar":
        return float(
            brentq(
                lambda p: tail_value_at_risk(x, p) - mean - target_margin, 0.01, 0.999
            )
        )
    if measure == "pht":
        return float(
            brentq(
                lambda q: proportional_hazards_transform(x, q) - mean - target_margin,
                1.0,
                1000.0,
            )
        )
    raise ValueError("measure must be 'var', 'tvar' or 'pht'")
