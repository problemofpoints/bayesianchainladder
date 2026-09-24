"""Private numpy chain-ladder primitives shared by link-ratio bootstraps, the
Claims Development Result, sensitivity analysis and analytic oracles.

Conventions follow the volume-weighted chain ladder in England & Verrall
(2002) and the reference implementation in Peter England's StochasticReserving
repository (https://github.com/DrPeterEngland/StochasticReserving, MIT).

All functions work on plain arrays shaped ``(..., n_origin, n_dev)``; a leading
simulation axis is allowed wherever documented.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .utils import _extract_period_value

DropList = Sequence[tuple[str | int, int]] | None


def cumulative_array(triangle) -> tuple[np.ndarray, list[int], list[int]]:
    """Return ``(cum, origins, devs)`` for a single-index chainladder Triangle."""
    vals = np.asarray(triangle.values, dtype=float)
    if vals.shape[0] != 1 or vals.shape[1] != 1:
        raise ValueError(
            "cumulative_array expects a single-index, single-column triangle"
        )
    origins = [_extract_period_value(o) for o in triangle.origin]
    devs = [int(d) for d in triangle.development]
    return vals[0, 0].copy(), origins, devs


def cumulative_to_incremental(cum: np.ndarray) -> np.ndarray:
    incr = np.array(cum, dtype=float, copy=True)
    incr[..., 1:] = cum[..., 1:] - cum[..., :-1]
    return incr


def latest_diagonal(cum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Latest observed value per origin and its column index (-1 if none)."""
    obs = ~np.isnan(cum)
    n_dev = cum.shape[-1]
    any_obs = obs.any(axis=-1)
    last_idx = np.where(any_obs, n_dev - 1 - np.argmax(obs[..., ::-1], axis=-1), -1)
    rows = np.arange(cum.shape[0])
    latest = np.where(any_obs, cum[rows, np.maximum(last_idx, 0)], np.nan)
    return latest, last_idx


def drop_mask(n_origin: int, n_dev: int, drop: DropList, origins, devs) -> np.ndarray:
    """Ones ``(n_origin, n_dev-1)`` with zeros at chainladder-style ``drop`` tuples."""
    mask = np.ones((n_origin, n_dev - 1))
    for origin_label, dev_months in drop or []:
        i = list(origins).index(int(origin_label))
        j = list(devs).index(int(dev_months))
        if j >= n_dev - 1:
            raise ValueError(f"dev {dev_months} has no link ratio to drop")
        mask[i, j] = 0.0
    return mask


def link_ratio_mask(cum: np.ndarray, drop: DropList, origins, devs) -> np.ndarray:
    """Availability mask (both cells observed) times ``drop_mask``."""
    n_o, n_d = cum.shape
    avail = (~np.isnan(cum[:, :-1]) & ~np.isnan(cum[:, 1:])).astype(float)
    return avail * drop_mask(n_o, n_d, drop, origins, devs)


def volume_weighted_factors(cum: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Volume-weighted development factors; ``cum`` may carry a leading sim axis."""
    c = np.nan_to_num(cum, nan=0.0)
    num = (c[..., :, 1:] * mask).sum(axis=-2)
    den = (c[..., :, :-1] * mask).sum(axis=-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(den > 0, num / den, 1.0)
    return f


def project_cumulative(cum: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """Fill every NaN cell forward from the previous column times the factor.

    ``factors`` is ``(n_dev-1,)`` for a 2-D ``cum`` or ``(S, n_dev-1)`` for a
    ``(S, n_origin, n_dev)`` ``cum``.
    """
    full = np.array(cum, dtype=float, copy=True)
    n_dev = full.shape[-1]
    for j in range(1, n_dev):
        need = np.isnan(full[..., :, j])
        fill = full[..., :, j - 1] * np.expand_dims(factors[..., j - 1], -1)
        full[..., :, j] = np.where(need, fill, full[..., :, j])
    return full


def link_ratio_sigma(
    cum: np.ndarray,
    mask: np.ndarray,
    factors: np.ndarray,
    variance_factor: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-development sigma for a link-ratio model (Mack when
    ``variance_factor`` is None, Negative Binomial when it is ``f*(f-1)``).

    Rules from England's ``Mack_Residuals``: sigma_j^2 = sum(w (F-f)^2 / v_j) /
    (n_j - 1); carry forward when n_j <= 1; last column = min of the previous
    two; zero where the cumulative factor is exactly 1.
    """
    vf = (
        np.ones_like(factors)
        if variance_factor is None
        else np.asarray(variance_factor, float)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = cum[:, 1:] / cum[:, :-1]
    w = cum[:, :-1]
    resid = (
        np.sqrt(np.abs(w)) * (ratios - factors) / np.sqrt(np.where(vf > 0, vf, np.nan))
    )
    resid = np.where(mask > 0, resid, np.nan)
    n_j = mask.sum(axis=0)
    ss = np.nansum(resid**2, axis=0)
    sigma = np.zeros(len(n_j))
    for j in range(len(n_j) - 1):
        if n_j[j] > 1:
            sigma[j] = np.sqrt(ss[j] / (n_j[j] - 1))
        else:
            sigma[j] = 0.0 if j == 0 else sigma[j - 1]
    if len(sigma) >= 3:
        sigma[-1] = min(sigma[-2], sigma[-3])
    elif len(sigma) == 2:
        sigma[-1] = sigma[-2]
    cum_factors = np.cumprod(factors[::-1])[::-1]
    sigma[np.isclose(cum_factors, 1.0)] = 0.0
    return sigma, resid
