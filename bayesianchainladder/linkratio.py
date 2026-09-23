"""Bootstrap (and, in a later task, Bayesian) estimators for link-ratio models.

Ports ``Main_Mack_Bstrap`` and ``Main_NegBin_Bstrap`` from Peter England's
StochasticReserving repository
(https://github.com/DrPeterEngland/StochasticReserving, MIT licence), which
implement England & Verrall (2002, 2006):

* Estimation error: resample scaled residuals (nonparametric) or draw
  pseudo link ratios from Gamma / Lognormal with the fitted mean and
  variance (parametric), then recompute volume-weighted factors.
* Process error: roll each origin forward from its latest cumulative using
  the pseudo factors, drawing each next cumulative from Gamma / Lognormal
  (or resampled residuals) with variance ``sigma_j^2 * v(f_j) * C_{i,j}``.

``v(f) = 1`` gives Mack's model; ``v(f) = f (f - 1)`` gives the over-dispersed
Negative Binomial model. When a mean is non-positive a Normal draw with the
same two moments is used, so results may contain negative increments.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np

from ._triangle_ops import (
    DropList,
    cumulative_array,
    link_ratio_mask,
    link_ratio_sigma,
    volume_weighted_factors,
)
from .base import BaseStochasticReserve
from .utils import validate_triangle

DISTRIBUTIONS = ("nonparametric", "gamma", "lognormal")
_TOL = 1e-12


def draw_with_moments(mean, sd, dist: str, rng: np.random.Generator, resid=None) -> np.ndarray:
    """Draw values with the given mean and sd from ``dist``.

    ``gamma`` / ``lognormal`` match the first two moments where ``mean > 0``
    and ``sd > 0``; cells with ``sd == 0`` return the mean; cells with a
    non-positive mean fall back to ``Normal(mean, sd)``. ``nonparametric``
    returns ``mean + resid * sd``.
    """
    mean = np.asarray(mean, dtype=float)
    sd = np.broadcast_to(np.asarray(sd, dtype=float), mean.shape)
    if dist == "nonparametric":
        if resid is None:
            raise ValueError("resid is required for nonparametric draws")
        return mean + np.asarray(resid, dtype=float) * sd
    out = np.array(mean, copy=True)
    pos = (mean > _TOL) & (sd > _TOL)
    if dist == "gamma":
        shape = mean[pos] ** 2 / sd[pos] ** 2
        out[pos] = rng.gamma(shape=shape, scale=sd[pos] ** 2 / mean[pos])
    elif dist == "lognormal":
        s2 = np.log1p((sd[pos] / mean[pos]) ** 2)
        out[pos] = rng.lognormal(mean=np.log(mean[pos]) - 0.5 * s2, sigma=np.sqrt(s2))
    else:
        raise ValueError(f"dist must be one of {DISTRIBUTIONS}, got {dist!r}")
    fallback = (mean <= _TOL) & (sd > _TOL)
    out[fallback] = rng.normal(mean[fallback], sd[fallback])
    return out


def sample_pseudo_factors(
    cum, mask, factors, sigma, variance_factor, residual_pool, dist, n_sims, rng
) -> np.ndarray:
    """Estimation-error stage: pseudo link ratios → volume-weighted factors, (S, n_dev-1)."""
    n_o, n_d = cum.shape
    w = np.nan_to_num(cum[:, :-1], nan=0.0)
    mean = np.broadcast_to(factors, (n_sims, n_o, n_d - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        sd_cell = sigma * np.sqrt(variance_factor) / np.sqrt(np.where(w > 0, w, np.nan))
    sd_cell = np.where(mask > 0, np.nan_to_num(sd_cell, nan=0.0), 0.0)
    sd = np.broadcast_to(sd_cell, mean.shape)
    resid = rng.choice(residual_pool, size=mean.shape) if dist == "nonparametric" else None
    pseudo_ratios = draw_with_moments(mean, sd, dist, rng, resid)
    weights = w * mask
    den = weights.sum(axis=0)
    num = (pseudo_ratios * weights).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, 1.0)


def forecast_link_ratio_paths(
    cum,
    factor_draws,
    sigma,
    variance_factor_fn: Callable[[np.ndarray], np.ndarray],
    dist: str,
    rng: np.random.Generator,
    residual_pool=None,
) -> np.ndarray:
    """Process-error stage: complete simulated cumulative triangles (S, n_o, n_d)."""
    factor_draws = np.asarray(factor_draws, dtype=float)
    n_sims = factor_draws.shape[0]
    n_o, n_d = cum.shape
    full = np.repeat(cum[None, ...], n_sims, axis=0)
    for j in range(1, n_d):
        need = np.isnan(cum[:, j])
        if not need.any():
            continue
        prev = full[:, :, j - 1]
        f = factor_draws[:, j - 1][:, None]
        mean = prev * f
        sd = sigma[j - 1] * np.sqrt(variance_factor_fn(f) * np.abs(prev))
        resid = rng.choice(residual_pool, size=mean.shape) if dist == "nonparametric" else None
        draw = draw_with_moments(mean, sd, dist, rng, resid)
        full[:, need, j] = draw[:, need]
    return full


class _LinkRatioBootstrap(BaseStochasticReserve):
    """Shared implementation; subclasses define the variance function."""

    def __init__(
        self,
        n_sims: int = 1000,
        bootstrap_dist: str = "gamma",
        forecast_dist: str = "gamma",
        drop: DropList = None,
        process_sigma=None,
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        if bootstrap_dist not in DISTRIBUTIONS:
            raise ValueError(f"bootstrap_dist must be one of {DISTRIBUTIONS}")
        if forecast_dist not in DISTRIBUTIONS:
            raise ValueError(f"forecast_dist must be one of {DISTRIBUTIONS}")
        self.n_sims = n_sims
        self.bootstrap_dist = bootstrap_dist
        self.forecast_dist = forecast_dist
        self.drop = drop
        self.process_sigma = None if process_sigma is None else np.asarray(process_sigma, float)
        self.random_seed = random_seed

    @staticmethod
    def variance_factor_fn(factors) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def fit(self, triangle):
        validate_triangle(triangle)
        self.triangle_ = triangle.copy()
        cum, origins, devs = cumulative_array(triangle)
        mask = link_ratio_mask(cum, self.drop, origins, devs)
        factors = volume_weighted_factors(cum, mask)
        vf = self.variance_factor_fn(factors)
        sigma, resid = link_ratio_sigma(cum, mask, factors, vf)

        n_j = mask.sum(axis=0)
        bias = np.where(n_j > 1, np.sqrt(n_j / np.maximum(n_j - 1, 1)), 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = resid / np.where(sigma > 0, sigma, np.nan) * bias
        pool = scaled[np.isfinite(scaled)]
        pool = pool - pool.mean()
        scaled = np.where(np.isfinite(scaled), scaled - np.nanmean(scaled[np.isfinite(scaled)]), np.nan)

        if self.process_sigma is not None and self.process_sigma.shape != sigma.shape:
            raise ValueError(
                f"process_sigma must have shape {sigma.shape}, got {self.process_sigma.shape}"
            )
        forecast_sigma = sigma if self.process_sigma is None else self.process_sigma

        rng = np.random.default_rng(self.random_seed)
        pseudo_factors = sample_pseudo_factors(
            cum, mask, factors, sigma, vf, pool, self.bootstrap_dist, self.n_sims, rng
        )
        full = forecast_link_ratio_paths(
            cum, pseudo_factors, forecast_sigma, self.variance_factor_fn,
            self.forecast_dist, rng, residual_pool=pool,
        )

        self.factors_ = factors
        self.sigma_ = sigma
        self.scaled_residuals_ = scaled
        self.pseudo_factors_ = pseudo_factors
        self._set_full_cumulative_posterior(np.moveaxis(full, 0, -1), origins, devs)
        # _is_fitted must be set before _reserves_from_full_posterior(), which
        # requires it via BaseStochasticReserve._require_full_posterior().
        self._is_fitted = True
        self.reserves_posterior_ = self._reserves_from_full_posterior()
        self._build_reserve_summaries()
        return self


class MackBootstrap(_LinkRatioBootstrap):
    """Bootstrap of Mack's model: Var(C_{i,j+1} | C_{i,j}) = sigma_j^2 C_{i,j}."""

    @staticmethod
    def variance_factor_fn(factors) -> np.ndarray:
        return np.ones_like(np.asarray(factors, dtype=float))


class NegativeBinomialBootstrap(_LinkRatioBootstrap):
    """Bootstrap of the over-dispersed Negative Binomial model:
    Var(C_{i,j+1} | C_{i,j}) = sigma_j^2 f_j (f_j - 1) C_{i,j}. Requires f_j > 1."""

    @staticmethod
    def variance_factor_fn(factors) -> np.ndarray:
        f = np.asarray(factors, dtype=float)
        return np.abs(f * (f - 1.0))

    def fit(self, triangle):
        cum, origins, devs = cumulative_array(triangle)
        f = volume_weighted_factors(cum, link_ratio_mask(cum, self.drop, origins, devs))
        if np.any(f <= 1.0):
            warnings.warn(
                "NegativeBinomialBootstrap: development factors <= 1 detected; the "
                "variance function f(f-1) is not valid there. Consider MackBootstrap.",
                UserWarning,
                stacklevel=2,
            )
        return super().fit(triangle)
