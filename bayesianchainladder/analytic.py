"""Closed-form prediction errors (RMSEP) for chain-ladder models.

``odp_analytic_rmsep`` ports ``ODP_ChainLadder`` from Peter England's
StochasticReserving repository
(https://github.com/DrPeterEngland/StochasticReserving, MIT licence): fit the
over-dispersed Poisson cross-classified GLM by IRLS, form the covariance of
the future fitted values from the parameter covariance, and add the process
variance on the diagonal (England & Verrall 2002, Section 7). The reserves
equal the volume-weighted chain ladder exactly when all cells are positive.

``mack_analytic_rmsep`` wraps ``chainladder.MackChainladder``. Both are used
as oracles in the test suite: a correctly implemented bootstrap should have a
standard deviation close to the analytic value.
"""

from __future__ import annotations

from dataclasses import dataclass

import chainladder as cl
import numpy as np
import pandas as pd

from ._triangle_ops import DropList, cumulative_array, cumulative_to_incremental


@dataclass
class AnalyticResult:
    origins: list[int]
    reserves: np.ndarray
    reserve_sd: np.ndarray
    total_reserve: float
    total_sd: float
    scale: np.ndarray | None = None
    coefficients: np.ndarray | None = None

    @property
    def total_cov(self) -> float:
        return (
            float(self.total_sd / self.total_reserve)
            if self.total_reserve
            else float("nan")
        )

    def to_frame(self) -> pd.DataFrame:
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = np.where(
                self.reserves != 0, self.reserve_sd / np.abs(self.reserves), np.nan
            )
        frame = pd.DataFrame(
            {"reserve": self.reserves, "sd": self.reserve_sd, "cov": cov},
            index=self.origins,
        )
        frame.loc["Total"] = [self.total_reserve, self.total_sd, self.total_cov]
        return frame


def poisson_irls(
    X: np.ndarray, y: np.ndarray, max_iter: int = 50, tol: float = 1e-10
) -> np.ndarray:
    """Poisson log-link GLM coefficients by iteratively reweighted least squares."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(y.mean(), 1e-8))
    for _ in range(max_iter):
        eta = X @ beta
        mu = np.exp(eta)
        z = eta + (y - mu) / mu
        xtw = X.T * mu
        new = np.linalg.solve(xtw @ X, xtw @ z)
        converged = np.max(np.abs(new - beta)) < tol
        beta = new
        if converged:
            break
    return beta


def _design_matrix(
    n_origin: int, n_dev: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i, j = np.indices((n_origin, n_dev))
    i, j = i.ravel(), j.ravel()
    X = np.zeros((n_origin * n_dev, 1 + (n_origin - 1) + (n_dev - 1)))
    X[:, 0] = 1.0
    rows = np.arange(len(i))
    X[rows[i > 0], i[i > 0]] = 1.0  # origin effects, columns 1..n_origin-1
    X[rows[j > 0], (n_origin - 1) + j[j > 0]] = 1.0  # dev effects
    return X, i, j


def _nonconstant_phi(pearson_sq, j_obs, n_dev, bias):
    n_j = np.bincount(j_obs, minlength=n_dev).astype(float)
    ss = np.bincount(j_obs, weights=pearson_sq, minlength=n_dev)
    phi = np.zeros(n_dev)
    for j in range(n_dev - 1):
        if n_j[j] > 1:
            phi[j] = bias * ss[j] / n_j[j]
        else:
            phi[j] = phi[j - 1] if j > 0 else 0.0
    phi[-1] = min(phi[-2], phi[-3]) if n_dev >= 3 else phi[-2]
    return phi


def odp_analytic_rmsep(triangle, scale: str = "nonconstant") -> AnalyticResult:
    if scale not in ("constant", "nonconstant"):
        raise ValueError("scale must be 'constant' or 'nonconstant'")
    cum, origins, _ = cumulative_array(triangle)
    incr = cumulative_to_incremental(cum)
    n_o, n_d = incr.shape
    X, i_all, j_all = _design_matrix(n_o, n_d)
    y_all = np.nan_to_num(incr, nan=0.0).ravel()
    obs = ~np.isnan(incr).ravel()
    if (np.bincount(j_all[obs], weights=y_all[obs], minlength=n_d) <= 0).any():
        raise ValueError(
            "ODP GLM needs a positive column sum of incrementals in every development period"
        )

    beta = poisson_irls(X[obs], y_all[obs])
    mu_all = np.exp(X @ beta)
    mu_obs, mu_fut = mu_all[obs], mu_all[~obs]
    n_obs, p = int(obs.sum()), X.shape[1]
    pearson_sq = (y_all[obs] - mu_obs) ** 2 / mu_obs
    if scale == "constant":
        phi = np.full(n_d, pearson_sq.sum() / (n_obs - p))
    else:
        phi = _nonconstant_phi(pearson_sq, j_all[obs], n_d, bias=n_obs / (n_obs - p))
    phi_safe = np.maximum(phi, 1e-12)

    w = mu_obs / phi_safe[j_all[obs]]
    sigma_beta = np.linalg.inv((X[obs].T * w) @ X[obs])
    X_fut = X[~obs]
    cov_mu = (X_fut @ sigma_beta @ X_fut.T) * np.outer(mu_fut, mu_fut)
    cov = cov_mu + np.diag(phi[j_all[~obs]] * mu_fut)

    origin_fut = i_all[~obs]
    reserves = np.bincount(origin_fut, weights=mu_fut, minlength=n_o)
    A = np.zeros((n_o, len(mu_fut)))
    A[origin_fut, np.arange(len(mu_fut))] = 1.0
    var_origin = np.einsum("ik,kl,il->i", A, cov, A)
    return AnalyticResult(
        origins=origins,
        reserves=reserves,
        reserve_sd=np.sqrt(np.maximum(var_origin, 0.0)),
        total_reserve=float(reserves.sum()),
        total_sd=float(np.sqrt(cov.sum())),
        scale=phi,
        coefficients=beta,
    )


def mack_analytic_rmsep(triangle, drop: DropList = None) -> AnalyticResult:
    cum, origins, _ = cumulative_array(triangle)
    dev = cl.Development(drop=list(drop) if drop else None).fit_transform(triangle)
    mack = cl.MackChainladder().fit(dev)
    reserves = np.nan_to_num(np.asarray(mack.ibnr_.values, dtype=float)[0, 0, :, 0])
    reserve_sd = np.nan_to_num(
        np.asarray(mack.mack_std_err_.latest_diagonal.values, dtype=float)[0, 0, :, 0]
    )
    sigma = np.asarray(dev.sigma_.values, dtype=float).flatten()
    return AnalyticResult(
        origins=origins,
        reserves=reserves,
        reserve_sd=reserve_sd,
        total_reserve=float(reserves.sum()),
        total_sd=float(np.asarray(mack.total_mack_std_err_).flatten()[0]),
        scale=np.concatenate([sigma, [np.nan]]),
        coefficients=np.log(np.asarray(dev.ldf_.values, dtype=float).flatten()),
    )
