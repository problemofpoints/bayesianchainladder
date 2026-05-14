"""
Stochastic Reserving Benchmark Script
======================================

Self-contained script that runs Mack Chain Ladder, ODP Bootstrap,
Correlated ODP Bootstrap, Bornhuetter-Ferguson Bootstrap, and Cape Cod
Bootstrap on a long-format triangle dataset.

Dependencies: chainladder, pandas, numpy, scipy. NO custom packages.

Usage: see argparse or `python run_stochastic_reserving.py --help`

Methods
-------
mack      : Mack Chain Ladder (normal approximation per Mack 1993)
odp       : ODP Bootstrap (chainladder.BootstrapODPSample + Chainladder)
              Non-parametric residual bootstrap — resamples empirical Pearson residuals.
              Can generate wild IBNR distributions when triangles have negative or near-zero
              incrementals (e.g. case reserve releases), because those cells produce huge
              Pearson residuals that inflate resampled triangles.
odp_param : ODP Bootstrap, parametric Normal (rho=0)
              Same Gaussian-copula machinery as odp_corr but with rho hard-coded to 0.
              Samples from Normal(fitted, sqrt(phi*fitted)) independently per cell —
              no residual resampling artifacts.  Use this to isolate the effect of
              non-parametric resampling vs. correlation.
odp_corr  : Correlated ODP Bootstrap (Clark/Ding/Zhou 2022 Gaussian copula)
              Parametric sampling with calendar-year correlation rho (default 0.1).
              Residual distribution is controlled by --residual-dist:
                normal  — Normal(0, 1) quantiles (default)
                t       — Student-t(df) quantiles; df derived from empirical excess kurtosis
                           via moment-matching: df = 6/ek + 4, clamped to [3, 15].
                skewt   — Hansen 1994 skew-t quantiles; df from kurtosis (clamped [3, 15])
                           and skew param from empirical skewness (clamped [-0.95, 0.95]).
odp_bf    : Parametric independent bootstrap (rho=0) + Bornhuetter-Ferguson (requires premium)
              Uses the same parametric machinery as odp_param (lognormal process variance by
              default) but applies cl.BornhuetterFerguson to each resampled triangle rather
              than cl.Chainladder.  More robust than the old non-parametric odp_bf because it
              avoids extreme-residual resampling artefacts.
odp_cc    : Parametric independent bootstrap (rho=0) + Cape Cod (requires premium)
              Same as odp_bf but applies cl.CapeCod.
odp_corr_bf : Parametric correlated bootstrap (rho>0) + Bornhuetter-Ferguson (requires premium)
              Adds calendar-year correlation (controlled by --rho) on top of odp_bf.
odp_corr_cc : Parametric correlated bootstrap (rho>0) + Cape Cod (requires premium)
              Adds calendar-year correlation (controlled by --rho) on top of odp_cc.

Residual distribution options (--residual-dist)
-----------------------------------------------
Only applies to odp_corr (and odp_param) when --process-variance odp.
mack/odp/odp_bf/odp_cc/odp_corr_bf/odp_corr_cc are unaffected when using
non-ODP process variance (lognormal/gamma/negbin use their own sampling).

  normal  : Standard Normal (backward-compatible default).
  t       : Student-t with empirically-derived df via moment-matching to excess kurtosis.
              df = 6 / excess_kurtosis + 4, clamped to [3, 15].
              df=3 → heaviest allowed tails; df=15 → near-Normal.
              For df>2 the t distribution has variance df/(df-2); we divide by
              sqrt(df/(df-2)) to normalise variance to 1 before scaling by sqrt(phi*mu).
  skewt   : Hansen (1994) skew-t.  Parameters: df (from kurtosis, as above) and
              lambda (skewness parameter in (-1, 1)) derived from empirical skewness.
              Implemented as a piecewise rescaling of the standard-t CDF / PPF.

Process variance options (--process-variance)
---------------------------------------------
Controls the variance-mean relationship when generating future incremental losses.
Applies to odp_corr, odp_param, odp_bf, odp_cc, odp_corr_bf, and odp_corr_cc.
mack and odp (non-parametric residual bootstrap) are unaffected.

  odp       : Var = phi * mu  (linear; standard ODP, backward-compatible default).
  gamma     : Var = mu^2 / alpha  (quadratic; alpha fit from chain-ladder residuals).
                Samples from Gamma(alpha, mu/alpha) via correlated uniform draws.
  lognormal : Var = mu^2 * (exp(sigma^2) - 1)  (multiplicative lognormal noise).
                sigma^2 fit from CV^2 of residuals: sigma^2 = log(1 + CV^2).
                Calibration on 200 Meyers (2015) triangles shows lognormal cuts the
                KS statistic from ~0.30 (ODP) to ~0.15 — the largest improvement
                observed of any variant tested.
  negbin    : Var = mu + mu^2 / k  (negative binomial; k fit from residuals).
                Provides super-Poisson variance but empirically performs similarly
                to gamma on the Meyers backtest (KS ~0.39).

Input CSV format
----------------
Required columns: origin, dev, and one or more loss columns
Optional columns: lob, group_id, premium

The loss column to model is selected via --loss-col (default: paid).
Common alternatives: case_incurred, incurred, reported.

Pass --loss-col both (a synonym for paid,case_incurred) or a
comma-separated list like --loss-col paid,case_incurred,reported
to run all methods on multiple loss columns in a single pass.

Output schema
-------------
lob, group_id, loss_type, method, accident_year, loss_to_date, paid_to_date,
mean_ultimate, mean_ibnr, cv_ibnr, ibnr_p5, ibnr_p50, ibnr_p75, ibnr_p95

IBNR convention
---------------
IBNR is always computed as ``ultimate − paid_to_date`` regardless of which
loss column was modelled.  The ``paid`` column must be present in the input
even when modelling ``case_incurred``.  ``loss_to_date`` shows the latest-
diagonal value of the *modelled* column (informational); ``paid_to_date``
always shows the paid latest-diagonal and is the offset used for IBNR.

Defaults
--------
--process-variance default is lognormal (previously odp).
  Lognormal captures the multiplicative noise structure of insurance losses
  (large losses scale proportionally to exposure, not additively), and
  back-testing on 200 Meyers (2015) triangles shows it cuts the KS
  statistic from ~0.30 (ODP) to ~0.15 — the largest single improvement
  of any variant tested.  The lognormal sigma is estimated once from the
  chain-ladder residuals as sigma^2 = log(1 + CV^2).
--rho default is 0.3 (previously 0.1).
  Clark/Ding/Zhou (2022) report empirical calendar-year correlations of
  0.2–0.4 across Schedule P lines; 0.3 is near the midpoint and produces
  well-calibrated reserve ranges in back-testing.
--n-sims default is 5000 to reduce Monte Carlo noise at the tail
  percentiles without materially increasing run time.
"""

from __future__ import annotations

import argparse
import logging
import sys
import types
import warnings
from typing import Any

import chainladder as cl
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cholesky

# Suppress known upstream warnings from chainladder
warnings.filterwarnings("ignore", category=UserWarning, module="chainladder")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="chainladder")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Optional tqdm support
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ---------------------------------------------------------------------------
# Residual distribution helpers for the correlated ODP bootstrap
# ---------------------------------------------------------------------------

def _t_df_from_kurtosis(residuals: np.ndarray, df_floor: float = 3.0, df_ceil: float = 15.0) -> float:
    """Derive Student-t degrees of freedom by moment-matching to excess kurtosis.

    For Student-t(df): excess kurtosis = 6 / (df - 4) when df > 4.
    Solving: df = 6 / ek + 4.

    Parameters
    ----------
    residuals : ndarray
        Finite standardised Pearson residuals.
    df_floor : float
        Minimum df (heaviest allowed tails).  Default 3.
    df_ceil : float
        Maximum df (near-Normal).  Default 15.

    Returns
    -------
    float
        Clamped df estimate in [df_floor, df_ceil].

    Notes
    -----
    Negative or zero excess kurtosis → df_ceil (platykurtic / Normal-like).
    The formula df = 6/ek + 4 requires ek > 0 and df > 4 to be strictly
    valid; we clamp the result to [df_floor, df_ceil] as a safety net.
    """
    r = residuals[np.isfinite(residuals)]
    if len(r) < 4:
        return df_ceil
    ek = float(stats.kurtosis(r, fisher=True))  # excess kurtosis (Normal=0)
    if ek <= 0:
        return df_ceil
    implied = 6.0 / ek + 4.0
    return float(np.clip(implied, df_floor, df_ceil))


def _hansen_skewt_ppf(u: np.ndarray, df: float, lam: float) -> np.ndarray:
    """Inverse CDF (PPF) of the Hansen (1994) skew-t distribution.

    Returns samples from the zero-mean, unit-variance Hansen (1994) skew-t.

    The CDF is a piecewise rescaling of the *standardised* Student-t CDF
    (i.e. with variance normalised to 1, not df/(df-2)):

        F(x) = (1-lam) * Ft_std( (bx + a) / (1-lam) )     x < -a/b
        F(x) = (1+lam) * Ft_std( (bx + a) / (1+lam) ) - lam   x >= -a/b

    where Ft_std is the standardised-t CDF (t.cdf(z*sqrt(df/(df-2)), df)),
    and the normalising constants are:

        c = Gamma((df+1)/2) / (sqrt(pi*(df-2)) * Gamma(df/2))
        a = 4 * lam * c * (df-2) / (df-1)   # ensures E[x] = 0
        b = sqrt(1 + 3*lam^2 - a^2)          # ensures Var(x) = 1

    The key subtlety is that stats.t.ppf draws from t with variance df/(df-2),
    so we divide each quantile by sqrt(df/(df-2)) to get the standardised-t
    quantile before applying the piecewise rescaling.

    Parameters
    ----------
    u : ndarray of floats in (0, 1)
        Uniform samples (from the Gaussian copula).
    df : float
        Degrees of freedom (> 2).
    lam : float
        Skewness parameter in (-1, 1).  lam>0 → right-skewed; lam<0 → left.

    Returns
    -------
    ndarray
        Samples from the standardised (mean=0, variance=1) Hansen skew-t.

    References
    ----------
    Hansen, B.E. (1994). Autoregressive conditional density estimation.
    Journal of Business & Economic Statistics 12(4): 705–712.
    """
    import math
    lam = float(np.clip(lam, -0.999, 0.999))
    df = max(df, 2.001)

    # Constants (Hansen 1994, equations 3–5)
    c = (
        math.gamma((df + 1) / 2)
        / (math.sqrt(math.pi * (df - 2)) * math.gamma(df / 2))
    )
    a = 4.0 * lam * c * (df - 2) / (df - 1)   # mean-zero constraint
    b = float(np.sqrt(max(1.0 + 3.0 * lam ** 2 - a ** 2, 1e-12)))  # unit-variance

    # Variance correction: stats.t.ppf gives t with Var = df/(df-2); divide by
    # sqrt(df/(df-2)) to obtain the standardised-t (Var = 1) quantile.
    t_scale = float(np.sqrt(df / (df - 2.0)))

    u = np.asarray(u, dtype=float)
    u_threshold = (1.0 - lam) / 2.0  # F(-a/b) = (1-lam)/2

    out = np.empty_like(u)
    left = u < u_threshold
    right = ~left

    # Left branch  (x < -a/b): F(x) = (1-lam)*Ft_std((bx+a)/(1-lam))
    #   => Ft_std((bx+a)/(1-lam)) = u/(1-lam)
    #   => (bx+a)/(1-lam) = stdT.ppf(u/(1-lam))
    #   => x = ((1-lam)*stdT.ppf(u/(1-lam)) - a) / b
    if np.any(left):
        ul = np.clip(u[left] / (1.0 - lam), 1e-12, 1.0 - 1e-12)
        std_t = stats.t.ppf(ul, df) / t_scale  # standardised-t quantile
        out[left] = ((1.0 - lam) * std_t - a) / b

    # Right branch (x >= -a/b): F(x) = (1+lam)*Ft_std((bx+a)/(1+lam)) - lam
    #   => (1+lam)*Ft_std((bx+a)/(1+lam)) = u + lam
    #   => Ft_std((bx+a)/(1+lam)) = (u+lam)/(1+lam)
    #   => x = ((1+lam)*stdT.ppf((u+lam)/(1+lam)) - a) / b
    if np.any(right):
        ur = np.clip((u[right] + lam) / (1.0 + lam), 1e-12, 1.0 - 1e-12)
        std_t = stats.t.ppf(ur, df) / t_scale  # standardised-t quantile
        out[right] = ((1.0 + lam) * std_t - a) / b

    return out


def _skewness_to_hansen_lambda(skewness: float) -> float:
    """Map empirical skewness to a Hansen-lambda parameter.

    Uses an empirical approximation: for moderate skewness |s| < 1,
    lambda ≈ skewness / 2 works well (exact for small lambda via Taylor
    expansion of the Hansen skewness formula).  We clamp to (-0.95, 0.95).
    """
    lam = skewness / 2.0
    return float(np.clip(lam, -0.95, 0.95))


def _get_residual_params(residuals: np.ndarray) -> dict:
    """Compute residual distribution parameters from observed standardised residuals.

    Returns a dict with keys: ``df`` (Student-t df, clamped [3, 15]) and
    ``lam`` (Hansen skew-t lambda, clamped [-0.95, 0.95]).
    """
    r = residuals[np.isfinite(residuals)]
    df = _t_df_from_kurtosis(r, df_floor=3.0, df_ceil=15.0)
    skewness = float(stats.skew(r)) if len(r) >= 3 else 0.0
    lam = _skewness_to_hansen_lambda(skewness)
    return {"df": df, "lam": lam}


# ---------------------------------------------------------------------------
# Correlated ODP Bootstrap (inlined from Clark/Ding/Zhou 2022)
# ---------------------------------------------------------------------------

def _build_full_correlation_matrix(n_origin, n_dev, nan_triangle, rho):
    """Build a cell-level correlation matrix for calendar-year correlated sampling.

    Implements the Clark/Ding/Zhou (2022) scheme: cells on the same calendar-year
    diagonal have correlation ``rho``; more distant diagonals decay as rho^(d+1).

    References
    ----------
    Clark, D.R., Ding, H., and Zhou, L. (2022). "Making Bootstrap Reserve
    Ranges More Realistic." CAS E-Forum, Summer 2022.
    """
    valid_indices = []
    for i in range(n_origin):
        for j in range(n_dev):
            if not np.isnan(nan_triangle[i, j]):
                valid_indices.append((i, j))

    n_cells = len(valid_indices)
    corr_matrix = np.eye(n_cells)
    for idx1, (i1, j1) in enumerate(valid_indices):
        cy1 = i1 + j1
        for idx2, (i2, j2) in enumerate(valid_indices):
            if idx1 == idx2:
                continue
            cy_diff = abs(cy1 - (i2 + j2))
            corr_matrix[idx1, idx2] = rho if cy_diff == 0 else rho ** (cy_diff + 1)

    return corr_matrix, valid_indices


def _generate_correlated_uniforms(n_cells, n_sims, corr_matrix, rng):
    """Convert correlated standard normals to [0,1] via the normal CDF."""
    try:
        L = cholesky(corr_matrix, lower=True)
    except np.linalg.LinAlgError:
        L = cholesky(corr_matrix + 1e-6 * np.eye(n_cells), lower=True)
    Z = rng.standard_normal(size=(n_sims, n_cells))
    correlated_normals = Z @ L.T
    return stats.norm.cdf(correlated_normals)


def _fit_process_variance_params(
    obs_incr: np.ndarray,
    fitted_incr: np.ndarray,
    phi: float,
    process_variance: str,
) -> dict:
    """Fit process-variance dispersion parameters from chain-ladder residuals.

    Parameters
    ----------
    obs_incr : ndarray (n_origin × n_dev)
        Observed incremental losses (NaN in future cells).
    fitted_incr : ndarray
        Fitted (expected) incremental losses from the chain-ladder model.
        Zero/negative values are floored at 1 before ratio calculations.
    phi : float
        Pearson dispersion (used as variance parameter for ODP).
    process_variance : {'odp', 'gamma', 'lognormal', 'negbin'}
        Target distribution family.

    Returns
    -------
    dict
        Keys depend on *process_variance*.  All paths return ``phi`` for
        compatibility with the ODP fallback.
    """
    min_fitted = 1.0
    fitted_safe = np.maximum(np.abs(fitted_incr), min_fitted)

    # Collect paired (fitted, obs) for observed cells
    mask = ~np.isnan(obs_incr)
    mu_vals = fitted_safe[mask]
    y_vals = obs_incr[mask]

    params: dict = {"phi": phi}

    if process_variance == "odp":
        return params

    # Squared residuals: (obs - fitted)^2 per cell — used by gamma and negbin
    sq_resid = (y_vals - mu_vals) ** 2

    if process_variance == "gamma":
        # Method of moments: Var = phi * mu  (ODP) → Var = mu^2 / alpha (Gamma)
        # Estimate alpha from: alpha = mu^2 / sample_variance_of_resid
        # Use a single pooled estimate: alpha = sum(mu^2) / sum(sq_resid) * (N-p)/N
        # (simple MoM pooled across cells, analogous to how phi is estimated)
        # Guard against near-zero sq_resid
        sum_sq = np.maximum(float(np.nansum(sq_resid)), 1e-6)
        sum_mu2 = float(np.nansum(mu_vals ** 2))
        alpha_raw = sum_mu2 / sum_sq
        # Clamp to a reasonable range: alpha in [0.1, 200]
        alpha = float(np.clip(alpha_raw, 0.1, 200.0))
        params["alpha"] = alpha

    elif process_variance == "lognormal":
        # Lognormal: Var = mu^2 * (exp(sigma^2) - 1)
        # => sigma^2 = log(1 + Var/mu^2) = log(1 + CV^2)
        # Estimate CV^2 from pooled (obs - fitted)^2 / fitted^2
        cv2_vals = sq_resid / (mu_vals ** 2)
        # Winsorise extreme CV2 values (cap at 10 = CV of ~316%)
        cv2_vals = np.clip(cv2_vals, 0.0, 10.0)
        cv2_mean = float(np.nanmean(cv2_vals))
        # Floor at a small positive value to avoid sigma≈0
        cv2_mean = max(cv2_mean, 1e-4)
        sigma2 = float(np.log(1.0 + cv2_mean))
        params["sigma2"] = sigma2

    elif process_variance == "negbin":
        # Negative Binomial: Var = mu + mu^2/k
        # => k = mu^2 / (Var - mu)
        # Estimate Var per cell = sq_resid, then pool: k = sum(mu^2) / sum(Var - mu)
        extra_var = sq_resid - mu_vals          # extra variance beyond Poisson
        extra_var = np.maximum(extra_var, 1e-6)  # ensure positive denominator
        sum_mu2 = float(np.nansum(mu_vals ** 2))
        sum_extra = float(np.nansum(extra_var))
        k_raw = sum_mu2 / sum_extra
        # Clamp: k in [0.01, 1000]
        k = float(np.clip(k_raw, 0.01, 1000.0))
        params["k"] = k

    return params


def _correlated_odp_bootstrap(
    triangle, n_sims, rho, hat_adj=True, random_state=None, residual_dist="normal",
    process_variance="odp",
):
    """Run the correlated ODP bootstrap (Clark/Ding/Zhou 2022).

    Parameters
    ----------
    triangle : chainladder.Triangle
        Single-entity loss triangle (shape (1, 1, n_origin, n_dev)).
    n_sims : int
        Number of bootstrap simulations.
    rho : float
        Same-diagonal correlation coefficient in [0, 1].
    hat_adj : bool
        Apply Shapland hat-matrix adjustment.
    random_state : int or None
        Seed for reproducibility.
    residual_dist : {'normal', 't', 'skewt'}
        Residual distribution to use for the quantile transform step.  Only
        applies when ``process_variance='odp'``.  Ignored for gamma / lognormal
        / negbin because those distributions are fully parameterised by their
        own fitted dispersion parameters.

        * ``'normal'``: standard Normal PPF (backward-compatible default).
        * ``'t'``: Student-t PPF with df estimated from empirical excess
          kurtosis via moment-matching (df = 6/ek + 4, clamped [3, 15]).
          The t-variate is then divided by sqrt(df/(df-2)) to normalise its
          variance back to 1 before multiplying by sqrt(phi * fitted).
        * ``'skewt'``: Hansen (1994) skew-t PPF.  df from kurtosis (clamped
          [3, 15]) and lambda from empirical skewness (clamped [-0.95, 0.95]).
          Already mean-0, variance-1 by construction.
    process_variance : {'odp', 'gamma', 'lognormal', 'negbin'}
        Variance–mean relationship to use when generating future incremental
        losses.

        * ``'odp'`` (default): ``Var = phi * mu`` — standard ODP, linear
          variance.  Quantile transform uses *residual_dist*.
        * ``'gamma'``: ``Var = mu^2 / alpha`` — quadratic variance.  Samples
          via ``stats.gamma.ppf`` applied to the correlated uniform draws.
        * ``'lognormal'``: ``Var = mu^2 * (exp(sigma^2) - 1)`` — log-normal
          multiplicative noise.  Samples via ``stats.lognorm.ppf``.
        * ``'negbin'``: ``Var = mu + mu^2 / k`` — negative binomial, heaviest
          tails.  Samples via ``stats.nbinom.ppf``.

        For all non-ODP options the dispersion parameter (alpha / sigma / k) is
        estimated ONCE from the observed chain-ladder residuals using
        ``_fit_process_variance_params`` before sampling begins.

    Returns
    -------
    per_origin_per_sim : ndarray, shape (n_origin, n_sims)
        IBNR per origin per simulation.
    """
    rng = np.random.RandomState(random_state)

    # Fit development and chain ladder to get expected incremental triangle
    # Development.fit_transform returns a Triangle (not a fitted estimator object)
    dev_tri = cl.Development(n_periods=-1).fit_transform(triangle)
    cl_model = cl.Chainladder().fit(dev_tri)
    exp_incr = cl_model.full_expectation_.cum_to_incr().values[0, 0, :, :triangle.shape[-1]]
    # nan_triangle from the development-transformed triangle
    nan_tri = dev_tri.nan_triangle
    exp_incr = np.nan_to_num(exp_incr) * nan_tri

    n_origin, n_dev = triangle.shape[2], triangle.shape[3]
    nan_triangle = nan_tri

    # Compute design matrix and hat adjustment
    design_matrix = _get_design_matrix(triangle)
    if hat_adj:
        try:
            hat_diag = _get_hat_diagonal(triangle, exp_incr, design_matrix)
        except Exception:
            hat_diag = None
    else:
        hat_diag = None

    # Residuals
    min_fitted = 1.0
    fitted_safe = np.maximum(np.abs(exp_incr), min_fitted)
    unscaled_resid = (
        (triangle.cum_to_incr().values[0, 0, :, :] - exp_incr) / np.sqrt(fitted_safe)
    )
    if hat_diag is not None:
        standardized_resid = hat_diag * unscaled_resid
    else:
        standardized_resid = unscaled_resid

    n_params = design_matrix.shape[1]
    degree_freedom = np.nansum(nan_triangle) - n_params
    pearson_chi_sq = np.nansum(standardized_resid ** 2)
    phi = pearson_chi_sq / degree_freedom

    # Derive residual distribution parameters from the observed residuals
    # (moment-matching on this triangle's empirical residuals)
    residual_params = _get_residual_params(standardized_resid)
    t_df = residual_params["df"]
    skewt_lam = residual_params["lam"]

    # Fit process-variance dispersion parameters (once, from this triangle's residuals)
    obs_incr = triangle.cum_to_incr().values[0, 0, :, :]
    pv_params = _fit_process_variance_params(obs_incr, exp_incr, phi, process_variance)

    def _apply_quantile_transform(u: np.ndarray) -> np.ndarray:
        """Map uniform samples u ∈ (0,1) → standardised residuals (ODP path only).

        The returned array has mean≈0 and variance≈1 regardless of which
        distribution is used, so downstream scaling by sqrt(phi * fitted)
        is consistent across residual_dist choices.
        """
        if residual_dist == "t":
            # Student-t(df) has variance df/(df-2) for df>2; normalise to 1.
            z = stats.t.ppf(u, df=t_df)
            if t_df > 2.0:
                z = z / float(np.sqrt(t_df / (t_df - 2.0)))
            return z
        elif residual_dist == "skewt":
            # Hansen skew-t is already mean-0, variance-1 by construction.
            return _hansen_skewt_ppf(u, df=t_df, lam=skewt_lam)
        else:
            # Normal (default, backward-compatible)
            return stats.norm.ppf(u)

    def _sample_cell(u: np.ndarray, mu: float) -> np.ndarray:
        """Sample n_sims incremental losses for a single cell using the chosen process-variance model.

        Parameters
        ----------
        u : ndarray, shape (n_sims,)
            Correlated uniform samples in (0, 1) from the Gaussian copula.
        mu : float
            Chain-ladder fitted incremental mean for this cell (already floored at 1).

        Returns
        -------
        ndarray, shape (n_sims,)
            Sampled incremental loss values (may be negative / non-integer; downstream
            cumsum handles these without further constraint).
        """
        if process_variance == "gamma":
            alpha = pv_params["alpha"]
            # Gamma(alpha, scale=mu/alpha): mean=mu, Var=mu^2/alpha
            scale = mu / alpha
            u_clipped = np.clip(u, 1e-9, 1.0 - 1e-9)
            return stats.gamma.ppf(u_clipped, a=alpha, scale=scale)

        elif process_variance == "lognormal":
            sigma2 = pv_params["sigma2"]
            sigma = float(np.sqrt(sigma2))
            # Lognormal with mean=mu: mu_log = log(mu) - sigma^2/2
            mu_log = float(np.log(max(mu, 1e-9))) - sigma2 / 2.0
            u_clipped = np.clip(u, 1e-9, 1.0 - 1e-9)
            return stats.lognorm.ppf(u_clipped, s=sigma, scale=float(np.exp(mu_log)))

        elif process_variance == "negbin":
            k = pv_params["k"]
            # NegBin: mean=mu, Var=mu+mu^2/k
            # scipy.stats.nbinom(n, p): mean=n*(1-p)/p, var=n*(1-p)/p^2
            # => n=k, p=k/(mu+k)
            n_param = k
            p_param = float(k / (mu + k))
            p_param = float(np.clip(p_param, 1e-9, 1.0 - 1e-9))
            u_clipped = np.clip(u, 1e-9, 1.0 - 1e-9)
            return stats.nbinom.ppf(u_clipped, n=n_param, p=p_param).astype(float)

        else:
            # ODP: Normal(mu, sqrt(phi * mu))
            std_dev = float(np.sqrt(phi * mu))
            z = _apply_quantile_transform(u)
            return mu + std_dev * z

    if rho != 0.0:
        corr_matrix, valid_indices = _build_full_correlation_matrix(
            n_origin, n_dev, nan_triangle, rho
        )
        n_cells = len(valid_indices)
        correlated_u = _generate_correlated_uniforms(n_cells, n_sims, corr_matrix, rng)

        # Parametric correlated sampling
        resampled_incr = np.zeros((n_sims, n_origin, n_dev))
        for cell_idx, (i, j) in enumerate(valid_indices):
            fitted_val = float(fitted_safe[i, j])
            resampled_incr[:, i, j] = _sample_cell(correlated_u[:, cell_idx], fitted_val)
        for i in range(n_origin):
            for j in range(n_dev):
                if (i, j) not in valid_indices:
                    resampled_incr[:, i, j] = np.nan

        resampled_triangles = np.cumsum(resampled_incr, axis=2)  # (n_sims, n_origin, n_dev)
    else:
        # Independent parametric sampling
        resampled_incr = np.zeros((n_sims, n_origin, n_dev))
        for i in range(n_origin):
            for j in range(n_dev):
                if np.isnan(nan_triangle[i, j]):
                    resampled_incr[:, i, j] = np.nan
                    continue
                raw_u = rng.uniform(0.0, 1.0, size=n_sims)
                resampled_incr[:, i, j] = _sample_cell(raw_u, float(fitted_safe[i, j]))
        resampled_triangles = np.cumsum(resampled_incr, axis=2)

    # Apply chain ladder to each simulation to get IBNR
    # We need to reconstruct chainladder Triangles and run predictions
    # Use the simpler approach: collect IBNR as final projected minus latest observed
    # Latest observed cumulative per origin
    latest_cum = np.array([
        _get_latest_value(triangle.values[0, 0, i, :]) for i in range(n_origin)
    ])
    # Development factors from the fitted CL model
    ldfs = np.asarray(cl_model.ldf_.values[0, 0, 0, :])  # (n_dev,) or similar

    # Ultimate per simulation = apply LDFs to each resampled triangle's latest diagonal
    per_origin_per_sim = np.zeros((n_origin, n_sims))
    for sim in range(n_sims):
        sim_triangle = resampled_triangles[sim]  # (n_origin, n_dev)
        for i in range(n_origin):
            # Find the last non-nan column for this origin in the sim
            row = sim_triangle[i, :]
            valid_js = np.where(~np.isnan(nan_triangle[i, :]))[0]
            if len(valid_js) == 0:
                continue
            last_j = valid_js[-1]
            # If fully developed, IBNR = 0
            if last_j >= n_dev - 1:
                per_origin_per_sim[i, sim] = 0.0
                continue
            cum_at_last = row[last_j]
            # Apply remaining LDFs
            ultimate = cum_at_last
            for j in range(last_j, n_dev - 1):
                if j < len(ldfs) and np.isfinite(ldfs[j]):
                    ultimate = ultimate * float(ldfs[j])
            per_origin_per_sim[i, sim] = ultimate - cum_at_last

    return per_origin_per_sim


def _get_latest_value(row):
    """Get last non-nan value from a row array."""
    valid = row[~np.isnan(row)]
    return float(valid[-1]) if len(valid) > 0 else 0.0


def _get_design_matrix(triangle):
    """Build the ODP design matrix (mirrors chainladder's BootstrapODPSample)."""
    w = triangle.nan_triangle
    arr = np.diag(w[:, 0])
    intra_beta = np.zeros((w.shape[0], w.shape[1] - 1))
    arr = np.concatenate((arr, intra_beta), axis=1)
    for i in range(w.shape[1] - 1):
        len_alpha = int(np.sum(~np.isnan(w[:, i + 1])))
        intra_alpha = np.diag(w[:, i + 1])[:len_alpha, :]
        intra_beta[:, i] = 1
        intra_beta = intra_beta[:len_alpha, :]
        intra_arr = np.concatenate((intra_alpha, intra_beta), axis=1)
        arr = np.concatenate((arr, intra_arr), axis=0)
    return arr


def _get_hat_diagonal(triangle, exp_incr_triangle, design_matrix):
    """Compute the Shapland hat-matrix diagonal adjustment."""
    weight_matrix = np.diag(
        pd.DataFrame(exp_incr_triangle).unstack().dropna().values
    )
    dtd = design_matrix.T @ weight_matrix @ design_matrix
    hat = design_matrix @ np.linalg.inv(dtd) @ design_matrix.T @ weight_matrix
    hat = np.diagonal(
        np.sqrt(np.where((1 - hat) != 0, 1.0 / np.abs(1 - hat), 0.0))
    )
    total_length = triangle.nan_triangle.shape[0]
    reshaped = hat[:total_length].reshape(1, total_length)
    indices = np.nansum(triangle.nan_triangle, axis=0).cumsum().astype(int)
    for num in range(len(indices) - 1):
        col_length = int(indices[num + 1] - indices[num])
        col = hat[int(indices[num]): int(indices[num + 1])].reshape(1, col_length)
        nans = np.full((1, total_length - col_length), np.nan)
        col = np.concatenate((col, nans), axis=1)
        reshaped = np.concatenate((reshaped, col), axis=0)
    return reshaped.T


# ---------------------------------------------------------------------------
# Triangle construction from long-format DataFrame
# ---------------------------------------------------------------------------

def df_to_triangle(df, value_col="paid", origin_col="origin", dev_col="dev"):
    """Convert a long-format DataFrame into a chainladder Triangle.

    Development is expected as elapsed months (12, 24, 36, …). It is
    converted to calendar end-of-year dates so that chainladder can infer
    the development lags correctly:

        eval_year = origin_year + dev_months // 12 - 1
        dev_date  = "{eval_year}-12-31"

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns ``origin`` (int year), ``dev`` (int months), and
        ``value_col`` (numeric cumulative losses).
    value_col : str
        Column containing the cumulative losses.
    origin_col, dev_col : str
        Names for origin and development columns.

    Returns
    -------
    chainladder.Triangle
    """
    work = df[[origin_col, dev_col, value_col]].copy()
    work.columns = ["origin", "dev", value_col]
    work["origin"] = work["origin"].astype(int)
    work["dev"] = work["dev"].astype(int)

    # Convert dev (elapsed months) to calendar end-of-year evaluation date.
    eval_year = work["origin"] + work["dev"] // 12 - 1
    work["dev_date"] = pd.to_datetime(
        eval_year.astype(str) + "-12-31", format="%Y-%m-%d"
    )

    tri = cl.Triangle(
        data=work,
        origin="origin",
        development="dev_date",
        columns=[value_col],
        cumulative=True,
        origin_format="%Y",
    )
    return tri


def _premium_as_exposure(loss_tri, prem_series):
    """Build a per-origin exposure triangle from a premium Series.

    Parameters
    ----------
    loss_tri : chainladder.Triangle
        The loss triangle (used as a structural template).
    prem_series : pd.Series
        Index = origin year (int), values = premium. Built from the
        ``premium`` column of the input DataFrame.

    Returns
    -------
    chainladder.Triangle with shape ``(1, 1, n_origin, 1)``
    """
    paid_origins = [int(str(o).split("-")[0]) for o in loss_tri.origin]
    prem_values = np.array(
        [prem_series.get(y, np.nan) for y in paid_origins], dtype=float
    )
    exposure = loss_tri.latest_diagonal.copy()
    exposure.values = prem_values[np.newaxis, np.newaxis, :, np.newaxis]
    return exposure


# ---------------------------------------------------------------------------
# Per-triangle method runners
# ---------------------------------------------------------------------------

def _run_mack(loss_tri, n_samples=5000, random_seed=None):
    """Run Mack Chain Ladder. Returns per-origin IBNR samples (n_origin, n_sims)."""
    dev = cl.Development(n_periods=-1).fit_transform(loss_tri)
    mack = cl.MackChainladder().fit(dev)

    ibnr_tri = mack.ibnr_.sum("development")
    ibnr_per_origin = np.asarray(ibnr_tri.values).flatten()
    std_per_origin = np.asarray(mack.mack_std_err_.latest_diagonal.values).flatten()

    rng = np.random.default_rng(random_seed)
    n_origin = len(ibnr_per_origin)
    samples = np.empty((n_origin, n_samples))
    for i in range(n_origin):
        mean = float(ibnr_per_origin[i]) if np.isfinite(ibnr_per_origin[i]) else 0.0
        std = float(std_per_origin[i]) if np.isfinite(std_per_origin[i]) and std_per_origin[i] >= 0 else 0.0
        samples[i] = rng.normal(loc=mean, scale=std, size=n_samples)

    return samples, ibnr_per_origin


def _run_odp_bootstrap(loss_tri, n_sims=1000, random_seed=None):
    """Run standard ODP bootstrap. Returns per-origin IBNR samples."""
    prepared = loss_tri.copy()
    prepared.key_labels = ["triangle_id"]
    prepared.kdims = np.asarray([["resample"]], dtype=object)

    sampler = cl.BootstrapODPSample(
        n_sims=n_sims, n_periods=-1, hat_adj=True, random_state=random_seed
    ).fit(prepared)
    resampled = sampler.transform(prepared)
    model = cl.Chainladder().fit(resampled)

    ibnr_vals = np.asarray(model.ibnr_.values)
    per_sim_per_origin = np.nansum(ibnr_vals, axis=-1)
    per_sim_per_origin = np.squeeze(per_sim_per_origin, axis=1)
    return per_sim_per_origin.T  # (n_origin, n_sims)


def _run_correlated_odp(loss_tri, n_sims=1000, rho=0.1, random_seed=None, residual_dist="normal",
                        process_variance="odp"):
    """Run correlated ODP bootstrap (Clark/Ding/Zhou 2022) via inline implementation."""
    return _correlated_odp_bootstrap(
        loss_tri, n_sims=n_sims, rho=rho, hat_adj=True, random_state=random_seed,
        residual_dist=residual_dist, process_variance=process_variance,
    )


def _run_odp_param(loss_tri, n_sims=1000, random_seed=None, residual_dist="normal",
                   process_variance="odp"):
    """Run parametric ODP bootstrap with independent (rho=0) Normal sampling.

    Uses the same Gaussian-copula machinery as odp_corr but with rho hard-coded
    to 0.  Unlike the non-parametric ``odp`` method, this draws from a fitted
    Normal(mu, sqrt(phi*mu)) distribution rather than resampling empirical
    Pearson residuals, so it is robust to triangles with negative incrementals.
    """
    return _correlated_odp_bootstrap(
        loss_tri, n_sims=n_sims, rho=0.0, hat_adj=True, random_state=random_seed,
        residual_dist=residual_dist, process_variance=process_variance,
    )


def _parametric_bootstrap_and_aggregate(
    loss_tri,
    exposure_tri,
    n_sims,
    rho,
    apriori,
    random_seed,
    aggregator,
    residual_dist="normal",
    process_variance="lognormal",
    apriori_sigma=0.15,
):
    """Parametric bootstrap (lognormal process variance by default) with BF or CC aggregation.

    Shared machinery for odp_bf / odp_corr_bf / odp_cc / odp_corr_cc.

    Parameters
    ----------
    loss_tri : chainladder.Triangle
    exposure_tri : chainladder.Triangle
        Premium exposure triangle.
    n_sims : int
    rho : float
        Calendar-year correlation. 0 → independent; >0 → correlated.
    apriori : float
        A-priori expected loss ratio (only used when aggregator='bf').
    apriori_sigma : float
        Standard deviation of the a-priori loss ratio.  When > 0, each
        bootstrap sample draws its own apriori from Normal(apriori,
        apriori_sigma) (BF) or Normal(cc_apriori, apriori_sigma) (CC),
        propagating apriori uncertainty into the reserve distribution.
        Default 0.15.  Set to 0 to recover the old deterministic-apriori
        behaviour (variance collapse).
    random_seed : int or None
    aggregator : {'chainladder', 'bf', 'cc'}
        Final aggregation method applied to each resampled triangle.
    residual_dist : str
        Passed to _correlated_odp_bootstrap.
    process_variance : str
        Passed to _correlated_odp_bootstrap.

    Returns
    -------
    per_origin_per_sim : ndarray, shape (n_origin, n_sims)
        IBNR per origin per simulation, as produced by the chosen aggregator.
    """
    # Step 1: generate n_sims resampled cumulative triangles via the parametric bootstrap.
    # We reuse _correlated_odp_bootstrap's triangle-generation logic but need to intercept
    # the per-sim triangles before the chain-ladder aggregation step.
    # Re-implement the triangle-generation piece inline.

    rng = np.random.RandomState(random_seed)

    dev_tri = cl.Development(n_periods=-1).fit_transform(loss_tri)
    cl_model = cl.Chainladder().fit(dev_tri)
    exp_incr = cl_model.full_expectation_.cum_to_incr().values[0, 0, :, :loss_tri.shape[-1]]
    nan_tri = dev_tri.nan_triangle
    exp_incr = np.nan_to_num(exp_incr) * nan_tri

    n_origin, n_dev = loss_tri.shape[2], loss_tri.shape[3]
    nan_triangle = nan_tri

    design_matrix = _get_design_matrix(loss_tri)
    try:
        hat_diag = _get_hat_diagonal(loss_tri, exp_incr, design_matrix)
    except Exception:
        hat_diag = None

    min_fitted = 1.0
    fitted_safe = np.maximum(np.abs(exp_incr), min_fitted)
    unscaled_resid = (
        (loss_tri.cum_to_incr().values[0, 0, :, :] - exp_incr) / np.sqrt(fitted_safe)
    )
    if hat_diag is not None:
        standardized_resid = hat_diag * unscaled_resid
    else:
        standardized_resid = unscaled_resid

    n_params = design_matrix.shape[1]
    degree_freedom = np.nansum(nan_triangle) - n_params
    pearson_chi_sq = np.nansum(standardized_resid ** 2)
    phi = pearson_chi_sq / degree_freedom

    pv_params = _fit_process_variance_params(
        loss_tri.cum_to_incr().values[0, 0, :, :], exp_incr, phi, process_variance
    )

    def _sample_cell_pv(u, mu):
        """Same as the inner _sample_cell in _correlated_odp_bootstrap."""
        if process_variance == "gamma":
            alpha = pv_params["alpha"]
            scale = mu / alpha
            u_c = np.clip(u, 1e-9, 1.0 - 1e-9)
            return stats.gamma.ppf(u_c, a=alpha, scale=scale)
        elif process_variance == "lognormal":
            sigma2 = pv_params["sigma2"]
            sigma = float(np.sqrt(sigma2))
            mu_log = float(np.log(max(mu, 1e-9))) - sigma2 / 2.0
            u_c = np.clip(u, 1e-9, 1.0 - 1e-9)
            return stats.lognorm.ppf(u_c, s=sigma, scale=float(np.exp(mu_log)))
        elif process_variance == "negbin":
            k = pv_params["k"]
            p_param = float(np.clip(k / (mu + k), 1e-9, 1.0 - 1e-9))
            u_c = np.clip(u, 1e-9, 1.0 - 1e-9)
            return stats.nbinom.ppf(u_c, n=k, p=p_param).astype(float)
        else:
            # ODP
            std_dev = float(np.sqrt(phi * mu))
            z = stats.norm.ppf(np.clip(u, 1e-9, 1.0 - 1e-9))
            return mu + std_dev * z

    # Generate resampled incremental arrays
    if rho != 0.0:
        corr_matrix, valid_indices = _build_full_correlation_matrix(
            n_origin, n_dev, nan_triangle, rho
        )
        n_cells = len(valid_indices)
        correlated_u = _generate_correlated_uniforms(n_cells, n_sims, corr_matrix, rng)
        resampled_incr = np.zeros((n_sims, n_origin, n_dev))
        for cell_idx, (i, j) in enumerate(valid_indices):
            fitted_val = float(fitted_safe[i, j])
            resampled_incr[:, i, j] = _sample_cell_pv(correlated_u[:, cell_idx], fitted_val)
        for i in range(n_origin):
            for j in range(n_dev):
                if (i, j) not in valid_indices:
                    resampled_incr[:, i, j] = np.nan
    else:
        resampled_incr = np.zeros((n_sims, n_origin, n_dev))
        for i in range(n_origin):
            for j in range(n_dev):
                if np.isnan(nan_triangle[i, j]):
                    resampled_incr[:, i, j] = np.nan
                    continue
                raw_u = rng.uniform(0.0, 1.0, size=n_sims)
                resampled_incr[:, i, j] = _sample_cell_pv(raw_u, float(fitted_safe[i, j]))

    resampled_cum = np.cumsum(resampled_incr, axis=2)  # (n_sims, n_origin, n_dev)

    # Step 2: apply BF or CC in batch by cloning the original triangle, stacking all
    # n_sims simulations along the key (index) dimension, and calling cl.Development +
    # cl.BornhuetterFerguson / cl.CapeCod once on the stacked triangle.
    import copy

    # Apply the observed mask: future cells become NaN
    # resampled_cum shape: (n_sims, n_origin, n_dev)
    masked_cum = np.where(
        nan_triangle[np.newaxis, :, :] == 1,  # observed cells → True
        resampled_cum,
        np.nan,
    )

    # Build a stacked chainladder Triangle:
    # values shape must be (n_sims, 1, n_origin, n_dev) to match (key, col, orig, dev).
    stacked_tri = copy.deepcopy(loss_tri)
    stacked_values = masked_cum[:, np.newaxis, :, :]  # (n_sims, 1, n_origin, n_dev)
    stacked_tri.values = stacked_values
    stacked_tri.key_labels = ["sim_id"]
    stacked_tri.kdims = np.array([[str(s)] for s in range(n_sims)], dtype=object)

    stacked_dev = cl.Development(n_periods=-1).fit_transform(stacked_tri)

    # Broadcast the exposure (premium) triangle to match n_sims keys.
    # exposure_tri has shape (1, 1, n_origin, 1).
    prem_values = exposure_tri.values  # (1, 1, n_origin, 1) or similar
    prem_broadcast = copy.deepcopy(exposure_tri)
    prem_broadcast.values = np.tile(prem_values, (n_sims, 1, 1, 1))
    prem_broadcast.key_labels = ["sim_id"]
    prem_broadcast.kdims = np.array([[str(s)] for s in range(n_sims)], dtype=object)

    if aggregator == "bf":
        model = cl.BornhuetterFerguson(
            apriori=apriori, apriori_sigma=apriori_sigma, random_state=random_seed
        ).fit(stacked_dev, sample_weight=prem_broadcast)
    elif aggregator == "cc":
        model = cl.CapeCod(
            apriori_sigma=apriori_sigma, random_state=random_seed
        ).fit(stacked_dev, sample_weight=prem_broadcast)
    else:
        model = cl.Chainladder().fit(stacked_dev)

    # ibnr_ shape: (n_sims, 1, n_origin, n_dev) — sum over dev, squeeze loss dim
    ibnr_arr = np.asarray(model.ibnr_.values)  # (n_sims, 1, n_origin, n_dev)
    ibnr_per_sim_origin = np.nansum(ibnr_arr[:, 0, :, :], axis=-1)  # (n_sims, n_origin)
    per_origin_per_sim = ibnr_per_sim_origin.T  # (n_origin, n_sims)

    return per_origin_per_sim


def _run_odp_bf(
    loss_tri, exposure_tri, apriori=0.65, n_sims=1000, random_seed=None,
    residual_dist="normal", process_variance="lognormal", apriori_sigma=0.15,
):
    """Run parametric independent bootstrap (rho=0) + Bornhuetter-Ferguson.

    Uses the parametric bootstrap framework (lognormal process variance by default)
    rather than chainladder's non-parametric residual bootstrap, making it robust
    to triangles with negative incrementals.  rho is hard-coded to 0 (independent).
    """
    return _parametric_bootstrap_and_aggregate(
        loss_tri, exposure_tri, n_sims=n_sims, rho=0.0, apriori=apriori,
        random_seed=random_seed, aggregator="bf",
        residual_dist=residual_dist, process_variance=process_variance,
        apriori_sigma=apriori_sigma,
    )


def _run_odp_cc(
    loss_tri, exposure_tri, n_sims=1000, random_seed=None,
    residual_dist="normal", process_variance="lognormal", apriori_sigma=0.15,
):
    """Run parametric independent bootstrap (rho=0) + Cape Cod.

    Uses the parametric bootstrap framework (lognormal process variance by default)
    rather than chainladder's non-parametric residual bootstrap.  rho is hard-coded
    to 0 (independent).
    """
    return _parametric_bootstrap_and_aggregate(
        loss_tri, exposure_tri, n_sims=n_sims, rho=0.0, apriori=0.65,
        random_seed=random_seed, aggregator="cc",
        residual_dist=residual_dist, process_variance=process_variance,
        apriori_sigma=apriori_sigma,
    )


def _run_odp_corr_bf(
    loss_tri, exposure_tri, apriori=0.65, n_sims=1000, rho=0.3, random_seed=None,
    residual_dist="normal", process_variance="lognormal", apriori_sigma=0.15,
):
    """Run parametric correlated bootstrap (rho>0) + Bornhuetter-Ferguson."""
    return _parametric_bootstrap_and_aggregate(
        loss_tri, exposure_tri, n_sims=n_sims, rho=rho, apriori=apriori,
        random_seed=random_seed, aggregator="bf",
        residual_dist=residual_dist, process_variance=process_variance,
        apriori_sigma=apriori_sigma,
    )


def _run_odp_corr_cc(
    loss_tri, exposure_tri, n_sims=1000, rho=0.3, random_seed=None,
    residual_dist="normal", process_variance="lognormal", apriori_sigma=0.15,
):
    """Run parametric correlated bootstrap (rho>0) + Cape Cod."""
    return _parametric_bootstrap_and_aggregate(
        loss_tri, exposure_tri, n_sims=n_sims, rho=rho, apriori=0.65,
        random_seed=random_seed, aggregator="cc",
        residual_dist=residual_dist, process_variance=process_variance,
        apriori_sigma=apriori_sigma,
    )


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def _loss_to_date_per_origin(loss_tri):
    """Latest diagonal values per origin as a Series."""
    diag = loss_tri.latest_diagonal
    vals = np.asarray(diag.values).flatten()
    origins = [str(o) for o in loss_tri.origin]
    return pd.Series(vals, index=origins)


def _samples_to_rows(
    per_origin_per_sim,
    loss_to_date,
    paid_per_origin,
    origins,
    lob,
    group_id,
    loss_type,
    method,
):
    """Convert (n_origin, n_sims) IBNR array to output rows.

    Parameters
    ----------
    per_origin_per_sim : ndarray, shape (n_origin, n_sims)
        IBNR samples from the reserving method (ultimate minus modelled
        latest diagonal, as produced by the method runners).
    loss_to_date : pd.Series
        Latest-diagonal values of the *modelled* loss column (informational).
    paid_per_origin : pd.Series
        Latest-diagonal values of the *paid* column.  Always used as the
        offset for IBNR: ``mean_ibnr = mean_ultimate - paid_to_date``.
    origins, lob, group_id, loss_type, method : forwarded to output rows

    Returns
    -------
    rows : list[dict]
        Per-origin + Total summary rows (output DataFrame schema).
    total_ibnr_samples : ndarray or None
        1-D array of total (summed-across-origins) IBNR samples after paid
        re-anchoring.  ``None`` if no finite samples were found.
    """
    rows = []
    total_loss_to_date = 0.0
    total_paid_to_date = 0.0
    total_mean_ibnr = 0.0
    all_total_ibnr = None

    for i, origin in enumerate(origins):
        samples = per_origin_per_sim[i, :]
        samples = samples[np.isfinite(samples)]
        loss = float(loss_to_date.iloc[i]) if i < len(loss_to_date) else 0.0
        paid = float(paid_per_origin.iloc[i]) if i < len(paid_per_origin) else 0.0
        total_loss_to_date += loss
        total_paid_to_date += paid

        # The method runners return IBNR = ultimate - modelled_latest_diagonal.
        # Re-anchor to paid: ibnr_vs_paid = (modelled_latest + ibnr_vs_modelled) - paid
        # i.e. ultimate - paid.  When loss_type == "paid", loss == paid so this
        # is a no-op; when loss_type == "case_incurred", loss > paid so samples
        # increase by (loss - paid).
        adjustment = loss - paid
        samples_vs_paid = samples + adjustment

        if samples_vs_paid.size == 0:
            mean_ibnr = std_ibnr = 0.0
            p5 = p50 = p75 = p95 = 0.0
            mean_ultimate = paid
        else:
            mean_ibnr = float(np.mean(samples_vs_paid))
            std_ibnr = float(np.std(samples_vs_paid, ddof=1)) if samples_vs_paid.size > 1 else 0.0
            p5, p50, p75, p95 = np.percentile(samples_vs_paid, [5, 50, 75, 95])
            mean_ultimate = paid + mean_ibnr

        cv_ibnr = abs(std_ibnr / mean_ibnr) if mean_ibnr != 0 else float("nan")
        total_mean_ibnr += mean_ibnr

        if all_total_ibnr is None:
            all_total_ibnr = samples_vs_paid.copy()
        else:
            min_len = min(len(all_total_ibnr), len(samples_vs_paid))
            all_total_ibnr = all_total_ibnr[:min_len] + samples_vs_paid[:min_len]

        rows.append({
            "lob": lob,
            "group_id": group_id,
            "loss_type": loss_type,
            "method": method,
            "accident_year": str(origin),
            "loss_to_date": loss,
            "paid_to_date": paid,
            "mean_ultimate": mean_ultimate,
            "mean_ibnr": mean_ibnr,
            "cv_ibnr": cv_ibnr,
            "ibnr_p5": float(p5),
            "ibnr_p50": float(p50),
            "ibnr_p75": float(p75),
            "ibnr_p95": float(p95),
        })

    # Total row
    if all_total_ibnr is not None and len(all_total_ibnr) > 0:
        t_mean = float(np.mean(all_total_ibnr))
        t_std = float(np.std(all_total_ibnr, ddof=1)) if len(all_total_ibnr) > 1 else 0.0
        t_cv = abs(t_std / t_mean) if t_mean != 0 else float("nan")
        t_p5, t_p50, t_p75, t_p95 = np.percentile(all_total_ibnr, [5, 50, 75, 95])
    else:
        t_mean = total_mean_ibnr
        t_std = t_cv = float("nan")
        t_p5 = t_p50 = t_p75 = t_p95 = float("nan")

    rows.append({
        "lob": lob,
        "group_id": group_id,
        "loss_type": loss_type,
        "method": method,
        "accident_year": "Total",
        "loss_to_date": total_loss_to_date,
        "paid_to_date": total_paid_to_date,
        "mean_ultimate": total_paid_to_date + t_mean,
        "mean_ibnr": t_mean,
        "cv_ibnr": t_cv,
        "ibnr_p5": float(t_p5),
        "ibnr_p50": float(t_p50),
        "ibnr_p75": float(t_p75),
        "ibnr_p95": float(t_p95),
    })

    return rows, all_total_ibnr


def run_methods_on_triangle(
    loss_tri,
    prem_series,
    methods,
    paid_per_origin,
    n_sims=5000,
    rho=0.1,
    apriori=0.65,
    apriori_sigma=0.15,
    random_seed=None,
    lob="unknown",
    group_id="unknown",
    loss_type="paid",
    collect_samples=False,
    residual_dist="normal",
    process_variance="odp",
):
    """Run all requested methods on a single (loss, premium) triangle pair.

    Parameters
    ----------
    loss_tri : chainladder.Triangle
    prem_series : pd.Series or None
        Premium per origin year (int index). Required for odp_bf and odp_cc.
    methods : list[str]
        Any subset of {"mack", "odp", "odp_param", "odp_corr",
        "odp_bf", "odp_cc", "odp_corr_bf", "odp_corr_cc"}.
    paid_per_origin : pd.Series
        Latest-diagonal paid values per origin.  Always used as the offset for
        IBNR computation (``mean_ibnr = mean_ultimate - paid_to_date``),
        regardless of which loss column was modelled.
    n_sims : int
    rho : float
    apriori : float
    apriori_sigma : float
        Standard deviation of the a-priori loss ratio for BF/CC methods.
        Default 0.15.  Set to 0 for deterministic apriori (variance collapse).
    random_seed : int or None
    lob, group_id : str
    loss_type : str
        Label for the loss column being modelled (e.g. "paid", "case_incurred").
        Passed through to output rows unchanged.
    collect_samples : bool
        If True, also return a list of dicts with total-IBNR sample arrays
        keyed by (lob, group_id, loss_type, method).
    residual_dist : {'normal', 't', 'skewt'}
        Residual distribution to use for odp_corr and odp_param.  See module
        docstring for details.
    process_variance : {'odp', 'gamma', 'lognormal', 'negbin'}
        Process variance model for odp_corr and odp_param.  See module
        docstring for details.

    Returns
    -------
    all_rows : list[dict]
        Rows for the output DataFrame.
    sample_chunks : list[dict]
        Only populated when ``collect_samples=True``.  Each dict has keys
        ``lob``, ``group_id``, ``loss_type``, ``method``, ``sample_idx``
        (0-based), and ``total_ibnr``.  Empty list when ``collect_samples=False``.
    """
    origins = [str(o) for o in loss_tri.origin]
    loss_per_origin = _loss_to_date_per_origin(loss_tri)
    exposure_tri = (
        _premium_as_exposure(loss_tri, prem_series)
        if prem_series is not None
        else None
    )

    all_rows = []
    sample_chunks = []

    for method in methods:
        try:
            if method == "mack":
                per_origin_sim, _ = _run_mack(loss_tri, n_samples=n_sims, random_seed=random_seed)
            elif method == "odp":
                per_origin_sim = _run_odp_bootstrap(loss_tri, n_sims=n_sims, random_seed=random_seed)
            elif method == "odp_param":
                per_origin_sim = _run_odp_param(
                    loss_tri, n_sims=n_sims, random_seed=random_seed, residual_dist=residual_dist,
                    process_variance=process_variance,
                )
            elif method == "odp_corr":
                per_origin_sim = _run_correlated_odp(
                    loss_tri, n_sims=n_sims, rho=rho, random_seed=random_seed,
                    residual_dist=residual_dist, process_variance=process_variance,
                )
            elif method == "odp_bf":
                if exposure_tri is None:
                    log.warning(
                        "lob=%s group_id=%s loss_type=%s: skipping odp_bf (no premium data)",
                        lob, group_id, loss_type,
                    )
                    continue
                per_origin_sim = _run_odp_bf(
                    loss_tri, exposure_tri, apriori=apriori, n_sims=n_sims,
                    random_seed=random_seed, residual_dist=residual_dist,
                    process_variance=process_variance, apriori_sigma=apriori_sigma,
                )
            elif method == "odp_cc":
                if exposure_tri is None:
                    log.warning(
                        "lob=%s group_id=%s loss_type=%s: skipping odp_cc (no premium data)",
                        lob, group_id, loss_type,
                    )
                    continue
                per_origin_sim = _run_odp_cc(
                    loss_tri, exposure_tri, n_sims=n_sims,
                    random_seed=random_seed, residual_dist=residual_dist,
                    process_variance=process_variance, apriori_sigma=apriori_sigma,
                )
            elif method == "odp_corr_bf":
                if exposure_tri is None:
                    log.warning(
                        "lob=%s group_id=%s loss_type=%s: skipping odp_corr_bf (no premium data)",
                        lob, group_id, loss_type,
                    )
                    continue
                per_origin_sim = _run_odp_corr_bf(
                    loss_tri, exposure_tri, apriori=apriori, n_sims=n_sims, rho=rho,
                    random_seed=random_seed, residual_dist=residual_dist,
                    process_variance=process_variance, apriori_sigma=apriori_sigma,
                )
            elif method == "odp_corr_cc":
                if exposure_tri is None:
                    log.warning(
                        "lob=%s group_id=%s loss_type=%s: skipping odp_corr_cc (no premium data)",
                        lob, group_id, loss_type,
                    )
                    continue
                per_origin_sim = _run_odp_corr_cc(
                    loss_tri, exposure_tri, n_sims=n_sims, rho=rho,
                    random_seed=random_seed, residual_dist=residual_dist,
                    process_variance=process_variance, apriori_sigma=apriori_sigma,
                )
            else:
                log.warning("Unknown method: %s — skipped", method)
                continue

            rows, total_ibnr_samples = _samples_to_rows(
                per_origin_sim, loss_per_origin, paid_per_origin,
                origins, lob, group_id, loss_type, method,
            )
            all_rows.extend(rows)

            if collect_samples and total_ibnr_samples is not None and len(total_ibnr_samples) > 0:
                for idx, val in enumerate(total_ibnr_samples):
                    sample_chunks.append({
                        "lob": lob,
                        "group_id": group_id,
                        "loss_type": loss_type,
                        "method": method,
                        "sample_idx": idx,
                        "total_ibnr": float(val),
                    })

        except Exception as exc:
            log.error(
                "lob=%s group_id=%s loss_type=%s method=%s failed: %s",
                lob, group_id, loss_type, method, exc,
                exc_info=True,
            )

    return all_rows, sample_chunks


def iterate_triangles(
    df,
    methods,
    loss_cols,
    n_sims=5000,
    rho=0.1,
    apriori=0.65,
    apriori_sigma=0.15,
    random_seed=None,
    collect_samples=False,
    residual_dist="normal",
    process_variance="odp",
):
    """Iterate over all (lob, group_id, loss_col) combinations and run all methods.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data with columns: origin, dev, plus the columns listed in
        ``loss_cols``, and optionally lob, group_id, premium.
        The ``paid`` column must be present — it is used as the offset for IBNR
        computation regardless of which loss column is modelled.
    methods : list[str]
    loss_cols : list[str]
        Loss columns to model (e.g. ["paid"] or ["paid", "case_incurred"]).
    n_sims, rho, apriori, apriori_sigma, random_seed : forwarded to run_methods_on_triangle
    collect_samples : bool
        If True, also accumulate full total-IBNR sample arrays.
    residual_dist : {'normal', 't', 'skewt'}
        Residual distribution for odp_corr / odp_param.
    process_variance : {'odp', 'gamma', 'lognormal', 'negbin'}
        Process variance model for odp_corr / odp_param.

    Returns
    -------
    results_df : pd.DataFrame with output schema
    samples_df : pd.DataFrame or None
        Only populated when ``collect_samples=True`` — columns are
        lob, group_id, loss_type, method, sample_idx, total_ibnr.
    """
    if "paid" not in df.columns:
        raise ValueError(
            "IBNR requires a `paid` column. Either include `paid` in the input "
            "or pass `--loss-col paid` (and only paid)."
        )

    # Normalize optional columns
    if "lob" not in df.columns:
        df = df.copy()
        df["lob"] = "all"
    if "group_id" not in df.columns:
        df = df.copy()
        df["group_id"] = "all"
    if "premium" not in df.columns:
        df = df.copy()
        df["premium"] = np.nan

    groups = df.groupby(["lob", "group_id"])
    group_list = list(groups)

    iterator = (
        tqdm(group_list, desc="Triangles", unit="tri")
        if _HAS_TQDM
        else group_list
    )

    all_rows = []
    all_sample_chunks = []
    for (lob, group_id), sub_df in iterator:
        # Build the paid latest-diagonal Series ONCE per (lob, group_id) so that
        # every loss_col shares the same paid_to_date offset for IBNR.
        try:
            paid_tri = df_to_triangle(sub_df, value_col="paid")
            paid_per_origin = _loss_to_date_per_origin(paid_tri)
        except Exception as exc:
            log.error(
                "lob=%s group_id=%s: failed to build paid triangle for IBNR offset: %s",
                lob, group_id, exc, exc_info=True,
            )
            continue

        for loss_col in loss_cols:
            if loss_col not in sub_df.columns:
                log.warning(
                    "lob=%s group_id=%s: loss column '%s' not found — skipped",
                    lob, group_id, loss_col,
                )
                continue
            if sub_df[loss_col].isna().all():
                log.warning(
                    "lob=%s group_id=%s: loss column '%s' is all-NaN — skipped",
                    lob, group_id, loss_col,
                )
                continue
            try:
                loss_tri = df_to_triangle(sub_df, value_col=loss_col)

                # Build per-origin premium Series if data available.
                prem_series = None
                if "premium" in sub_df.columns and sub_df["premium"].notna().any():
                    prem_series = (
                        sub_df.groupby("origin")["premium"]
                        .first()
                        .astype(float)
                    )

                rows, sample_chunks = run_methods_on_triangle(
                    loss_tri,
                    prem_series,
                    methods=methods,
                    paid_per_origin=paid_per_origin,
                    n_sims=n_sims,
                    rho=rho,
                    apriori=apriori,
                    apriori_sigma=apriori_sigma,
                    random_seed=random_seed,
                    lob=lob,
                    group_id=group_id,
                    loss_type=loss_col,
                    collect_samples=collect_samples,
                    residual_dist=residual_dist,
                    process_variance=process_variance,
                )
                all_rows.extend(rows)
                if collect_samples:
                    all_sample_chunks.extend(sample_chunks)

            except Exception as exc:
                log.error(
                    "lob=%s group_id=%s loss_col=%s: failed to process triangle: %s",
                    lob, group_id, loss_col, exc, exc_info=True,
                )

    _empty_cols = [
        "lob", "group_id", "loss_type", "method", "accident_year",
        "loss_to_date", "paid_to_date",
        "mean_ultimate", "mean_ibnr", "cv_ibnr",
        "ibnr_p5", "ibnr_p50", "ibnr_p75", "ibnr_p95",
    ]
    results_df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame(columns=_empty_cols)
    samples_df = pd.DataFrame(all_sample_chunks) if collect_samples else None
    return results_df, samples_df


# ---------------------------------------------------------------------------
# Parallel wrapper
# ---------------------------------------------------------------------------

def _run_single_group(args):
    """Worker function for multiprocessing pool."""
    (lob, group_id), sub_df, methods, loss_cols, n_sims, rho, apriori, apriori_sigma, random_seed, collect_samples, residual_dist, process_variance = args
    all_rows = []
    all_sample_chunks = []

    # Build paid latest-diagonal once for this group.
    if "paid" not in sub_df.columns:
        log.error(
            "lob=%s group_id=%s: `paid` column missing — cannot compute IBNR offset",
            lob, group_id,
        )
        return all_rows, all_sample_chunks
    try:
        paid_tri = df_to_triangle(sub_df, value_col="paid")
        paid_per_origin = _loss_to_date_per_origin(paid_tri)
    except Exception as exc:
        log.error(
            "lob=%s group_id=%s: failed to build paid triangle for IBNR offset: %s",
            lob, group_id, exc,
        )
        return all_rows, all_sample_chunks

    for loss_col in loss_cols:
        if loss_col not in sub_df.columns or sub_df[loss_col].isna().all():
            continue
        try:
            loss_tri = df_to_triangle(sub_df, value_col=loss_col)

            prem_series = None
            if "premium" in sub_df.columns and sub_df["premium"].notna().any():
                prem_series = sub_df.groupby("origin")["premium"].first().astype(float)

            rows, sample_chunks = run_methods_on_triangle(
                loss_tri, prem_series, methods=methods,
                paid_per_origin=paid_per_origin,
                n_sims=n_sims, rho=rho, apriori=apriori,
                apriori_sigma=apriori_sigma,
                random_seed=random_seed, lob=lob, group_id=group_id,
                loss_type=loss_col, collect_samples=collect_samples,
                residual_dist=residual_dist, process_variance=process_variance,
            )
            all_rows.extend(rows)
            if collect_samples:
                all_sample_chunks.extend(sample_chunks)
        except Exception as exc:
            log.error("lob=%s group_id=%s loss_col=%s: worker failed: %s", lob, group_id, loss_col, exc)
    return all_rows, all_sample_chunks


def iterate_triangles_parallel(
    df, methods, loss_cols, n_sims=5000, rho=0.1, apriori=0.65, apriori_sigma=0.15,
    random_seed=None, n_jobs=1,
    collect_samples=False, residual_dist="normal", process_variance="odp",
):
    """Parallel version of iterate_triangles using multiprocessing.Pool."""
    import multiprocessing

    if "paid" not in df.columns:
        raise ValueError(
            "IBNR requires a `paid` column. Either include `paid` in the input "
            "or pass `--loss-col paid` (and only paid)."
        )

    if "lob" not in df.columns:
        df = df.copy()
        df["lob"] = "all"
    if "group_id" not in df.columns:
        df = df.copy()
        df["group_id"] = "all"
    if "premium" not in df.columns:
        df = df.copy()
        df["premium"] = np.nan

    groups = list(df.groupby(["lob", "group_id"]))
    tasks = [
        ((lob, gid), sub, methods, loss_cols, n_sims, rho, apriori, apriori_sigma, random_seed, collect_samples, residual_dist, process_variance)
        for (lob, gid), sub in groups
    ]

    with multiprocessing.Pool(processes=n_jobs) as pool:
        results = pool.map(_run_single_group, tasks)

    all_rows = []
    all_sample_chunks = []
    for group_rows, group_samples in results:
        all_rows.extend(group_rows)
        if collect_samples:
            all_sample_chunks.extend(group_samples)

    _empty_cols = [
        "lob", "group_id", "loss_type", "method", "accident_year",
        "loss_to_date", "paid_to_date",
        "mean_ultimate", "mean_ibnr", "cv_ibnr",
        "ibnr_p5", "ibnr_p50", "ibnr_p75", "ibnr_p95",
    ]
    results_df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame(columns=_empty_cols)
    samples_df = pd.DataFrame(all_sample_chunks) if collect_samples else None
    return results_df, samples_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_loss_cols(value: str) -> list[str]:
    """Expand --loss-col value to a list of column names.

    "both" is a convenience alias for "paid,case_incurred".
    Comma-separated values are split and stripped.
    """
    if value.lower() == "both":
        return ["paid", "case_incurred"]
    return [c.strip() for c in value.split(",") if c.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="run_stochastic_reserving.py",
        description=(
            "Run Mack/ODP/Corr-ODP/BF/CC stochastic reserving on a "
            "long-format triangle CSV and write results."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", required=True, help="Input CSV path")
    p.add_argument("--output", "-o", default="results.csv", help="Output CSV path")
    p.add_argument(
        "--methods",
        nargs="+",
        default=["mack", "odp", "odp_corr", "odp_bf", "odp_cc", "odp_corr_bf", "odp_corr_cc"],
        choices=["mack", "odp", "odp_param", "odp_corr", "odp_bf", "odp_cc",
                 "odp_corr_bf", "odp_corr_cc"],
        metavar="METHOD",
        help=(
            "Methods to run. Choices: mack odp odp_param odp_corr odp_bf odp_cc "
            "odp_corr_bf odp_corr_cc. "
            "odp_param is parametric Normal with rho=0 (no residual-resampling artifacts). "
            "odp_bf and odp_cc are parametric independent bootstrap + BF/CC. "
            "odp_corr_bf and odp_corr_cc are parametric correlated bootstrap + BF/CC. "
            "All BF/CC variants require a 'premium' column."
        ),
    )
    p.add_argument(
        "--loss-col",
        default="paid",
        help=(
            "Loss column(s) to model. Use a single column name (e.g. 'paid', "
            "'case_incurred'), a comma-separated list (e.g. 'paid,case_incurred'), "
            "or the keyword 'both' (alias for 'paid,case_incurred'). "
            "When multiple columns are specified the script runs all methods on "
            "each column and adds a loss_type column to the output."
        ),
    )
    p.add_argument("--n-sims", type=int, default=5000, help="Bootstrap simulation count")
    p.add_argument(
        "--rho", type=float, default=0.3,
        help="Calendar-year correlation for odp_corr / odp_corr_bf / odp_corr_cc (0=independent)"
    )
    p.add_argument(
        "--apriori", type=float, default=0.65,
        help="A-priori expected loss ratio for odp_bf"
    )
    p.add_argument(
        "--apriori-sigma", type=float, default=0.15,
        dest="apriori_sigma",
        help=(
            "Standard deviation of the a-priori loss ratio for BF/CC methods.  "
            "Controls how much apriori uncertainty contributes to the reserve "
            "distribution.  Default 0.15 (empirical cross-triangle LR std across "
            "Meyers lines).  Set to 0 to use a deterministic apriori (WARNING: "
            "this causes near-zero BF/CC variance — the pre-fix behaviour)."
        ),
    )
    p.add_argument(
        "--n-jobs", type=int, default=1,
        help="Number of parallel workers (>1 uses multiprocessing.Pool)"
    )
    p.add_argument(
        "--random-seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    p.add_argument(
        "--save-samples",
        default=None,
        metavar="PATH",
        help=(
            "If set, write a parquet file with one row per "
            "(lob, group_id, loss_type, method, sample_idx) containing the "
            "simulated total IBNR (summed across all origin years).  "
            "Enables exact implied-percentile computation against actual ultimates. "
            "With 200 triangles × 2 loss types × 8 methods × 5000 sims = 16M rows, "
            "parquet keeps the file manageable. Omit this flag to keep output unchanged."
        ),
    )
    p.add_argument(
        "--residual-dist",
        default="normal",
        choices=["normal", "t", "skewt"],
        metavar="DIST",
        help=(
            "Residual distribution for odp_corr / odp_param.  "
            "Choices: normal (default), t, skewt.  "
            "normal: Normal(0,1) PPF (backward-compatible).  "
            "t: Student-t PPF; df derived from empirical excess kurtosis via "
            "moment-matching (df = 6/ek + 4, clamped [3, 15]).  "
            "skewt: Hansen (1994) skew-t PPF; df from kurtosis, skew from "
            "empirical skewness (lambda clamped [-0.95, 0.95])."
        ),
    )
    p.add_argument(
        "--process-variance",
        default="lognormal",
        choices=["odp", "gamma", "lognormal", "negbin"],
        metavar="PV",
        help=(
            "Process variance (mean-variance relationship) for odp_corr / odp_param.  "
            "Choices: odp (default), gamma, lognormal, negbin.  "
            "odp: Var = phi * mu (linear — standard ODP).  "
            "gamma: Var = mu^2 / alpha (quadratic; alpha fit from residuals).  "
            "lognormal: Var = mu^2*(exp(sigma^2)-1) (quadratic with heavier tails; "
            "sigma fit from residuals).  "
            "negbin: Var = mu + mu^2/k (super-Poisson; k fit from residuals, "
            "heaviest tails)."
        ),
    )
    p.add_argument(
        "--origin-col", default="origin", help="Name of the origin column"
    )
    p.add_argument(
        "--dev-col", default="dev", help="Name of the development column"
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    log.info("Reading input from %s", args.input)
    df = pd.read_csv(args.input)

    # Rename user-specified column names to canonical names
    rename = {}
    if args.origin_col != "origin":
        rename[args.origin_col] = "origin"
    if args.dev_col != "dev":
        rename[args.dev_col] = "dev"
    if rename:
        df = df.rename(columns=rename)

    for col in ["origin", "dev"]:
        if col not in df.columns:
            log.error("Required column '%s' not found. Available: %s", col, list(df.columns))
            sys.exit(1)

    df["origin"] = df["origin"].astype(int)
    df["dev"] = df["dev"].astype(int)

    # Resolve loss columns
    loss_cols = _parse_loss_cols(args.loss_col)
    log.info("Loss column(s): %s", loss_cols)

    # Coerce each requested loss column to numeric; warn if absent
    for lc in loss_cols:
        if lc not in df.columns:
            log.error(
                "Loss column '%s' not found in input. Available columns: %s",
                lc, list(df.columns),
            )
            sys.exit(1)
        df[lc] = pd.to_numeric(df[lc], errors="coerce")

    # The paid column must always be present for IBNR computation.
    if "paid" not in df.columns:
        log.error(
            "IBNR requires a `paid` column. Either include `paid` in the input "
            "or pass `--loss-col paid` (and only paid). "
            "Available columns: %s",
            list(df.columns),
        )
        sys.exit(1)
    df["paid"] = pd.to_numeric(df["paid"], errors="coerce")

    log.info(
        "Loaded %d rows — LOBs: %s",
        len(df),
        list(df["lob"].unique()) if "lob" in df.columns else ["all"],
    )
    log.info(
        "Methods: %s | n_sims=%d | rho=%.2f | apriori=%.3f | apriori_sigma=%.3f | residual_dist=%s | process_variance=%s",
        args.methods, args.n_sims, args.rho, args.apriori, args.apriori_sigma,
        args.residual_dist, args.process_variance,
    )

    collect_samples = args.save_samples is not None
    if collect_samples:
        log.info("Sample collection enabled — samples will be written to %s", args.save_samples)

    if args.n_jobs > 1:
        log.info("Running in parallel with %d workers", args.n_jobs)
        results, samples_df = iterate_triangles_parallel(
            df,
            methods=args.methods,
            loss_cols=loss_cols,
            n_sims=args.n_sims,
            rho=args.rho,
            apriori=args.apriori,
            apriori_sigma=args.apriori_sigma,
            random_seed=args.random_seed,
            n_jobs=args.n_jobs,
            collect_samples=collect_samples,
            residual_dist=args.residual_dist,
            process_variance=args.process_variance,
        )
    else:
        results, samples_df = iterate_triangles(
            df,
            methods=args.methods,
            loss_cols=loss_cols,
            n_sims=args.n_sims,
            rho=args.rho,
            apriori=args.apriori,
            apriori_sigma=args.apriori_sigma,
            random_seed=args.random_seed,
            collect_samples=collect_samples,
            residual_dist=args.residual_dist,
            process_variance=args.process_variance,
        )

    if results.empty:
        log.warning("No results produced. Check input data and method compatibility.")
    else:
        results.to_csv(args.output, index=False)
        log.info("Results written to %s (%d rows)", args.output, len(results))

        # Print a quick summary to stdout
        total_rows = results[results["accident_year"] == "Total"].copy()
        if not total_rows.empty:
            print("\n=== Total IBNR by loss_type / method ===")
            for _, row in total_rows.iterrows():
                print(
                    f"  LOB={row['lob']} | group={row['group_id']} | "
                    f"loss_type={row['loss_type']:15s} | "
                    f"method={row['method']:8s} | "
                    f"paid_to_date={row['paid_to_date']:>12,.0f} | "
                    f"mean_ibnr={row['mean_ibnr']:>12,.0f} | "
                    f"cv={row['cv_ibnr']:.3f} | "
                    f"p95={row['ibnr_p95']:>12,.0f}"
                )

    if collect_samples and samples_df is not None and not samples_df.empty:
        import pathlib
        samples_path = pathlib.Path(args.save_samples)
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        # Cast types for parquet efficiency
        samples_df["group_id"] = samples_df["group_id"].astype(str)
        samples_df["sample_idx"] = samples_df["sample_idx"].astype("int32")
        samples_df["total_ibnr"] = samples_df["total_ibnr"].astype("float32")
        samples_df.to_parquet(samples_path, index=False, compression="snappy")
        log.info(
            "Samples written to %s (%d rows, %d unique (lob,group_id,loss_type,method) combos)",
            args.save_samples,
            len(samples_df),
            samples_df.groupby(["lob", "group_id", "loss_type", "method"]).ngroups,
        )
    elif collect_samples:
        log.warning("--save-samples requested but no samples were collected.")

    return results


if __name__ == "__main__":
    main()
