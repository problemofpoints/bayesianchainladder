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
odp_corr  : Correlated ODP Bootstrap (Clark/Ding/Zhou 2022 Gaussian copula)
odp_bf    : ODP Bootstrap + Bornhuetter-Ferguson (requires premium)
odp_cc    : ODP Bootstrap + Cape Cod (requires premium)

Input CSV format
----------------
Required columns: origin, dev, paid
Optional columns: lob, group_id, premium

Output schema
-------------
lob, group_id, method, accident_year, paid_to_date,
mean_ultimate, mean_ibnr, cv_ibnr, ibnr_p5, ibnr_p50, ibnr_p75, ibnr_p95
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


def _correlated_odp_bootstrap(triangle, n_sims, rho, hat_adj=True, random_state=None):
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

    if rho != 0.0:
        corr_matrix, valid_indices = _build_full_correlation_matrix(
            n_origin, n_dev, nan_triangle, rho
        )
        n_cells = len(valid_indices)
        correlated_u = _generate_correlated_uniforms(n_cells, n_sims, corr_matrix, rng)

        # Parametric (Normal) correlated sampling
        resampled_incr = np.zeros((n_sims, n_origin, n_dev))
        for cell_idx, (i, j) in enumerate(valid_indices):
            fitted_val = fitted_safe[i, j]
            std_dev = np.sqrt(phi * fitted_val)
            z = stats.norm.ppf(correlated_u[:, cell_idx])
            resampled_incr[:, i, j] = exp_incr[i, j] + std_dev * z
        for i in range(n_origin):
            for j in range(n_dev):
                if (i, j) not in valid_indices:
                    resampled_incr[:, i, j] = np.nan

        resampled_triangles = np.cumsum(resampled_incr, axis=2)  # (n_sims, n_origin, n_dev)
    else:
        # Independent parametric (Normal) sampling
        std_dev = np.sqrt(phi * fitted_safe)
        z = rng.standard_normal(size=(n_sims,) + exp_incr.shape)
        resampled_incr = exp_incr + std_dev * z
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
        Column containing the cumulative paid losses.
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


def _premium_as_exposure(paid_tri, prem_series):
    """Build a per-origin exposure triangle from a premium Series.

    Parameters
    ----------
    paid_tri : chainladder.Triangle
        The paid-loss triangle (used as a structural template).
    prem_series : pd.Series
        Index = origin year (int), values = premium. Built from the
        ``premium`` column of the input DataFrame.

    Returns
    -------
    chainladder.Triangle with shape ``(1, 1, n_origin, 1)``
    """
    paid_origins = [int(str(o).split("-")[0]) for o in paid_tri.origin]
    prem_values = np.array(
        [prem_series.get(y, np.nan) for y in paid_origins], dtype=float
    )
    exposure = paid_tri.latest_diagonal.copy()
    exposure.values = prem_values[np.newaxis, np.newaxis, :, np.newaxis]
    return exposure


# ---------------------------------------------------------------------------
# Per-triangle method runners
# ---------------------------------------------------------------------------

def _run_mack(paid_tri, n_samples=5000, random_seed=None):
    """Run Mack Chain Ladder. Returns per-origin IBNR samples (n_origin, n_sims)."""
    dev = cl.Development(n_periods=-1).fit_transform(paid_tri)
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


def _run_odp_bootstrap(paid_tri, n_sims=1000, random_seed=None):
    """Run standard ODP bootstrap. Returns per-origin IBNR samples."""
    prepared = paid_tri.copy()
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


def _run_correlated_odp(paid_tri, n_sims=1000, rho=0.5, random_seed=None):
    """Run correlated ODP bootstrap (Clark/Ding/Zhou 2022) via inline implementation."""
    return _correlated_odp_bootstrap(
        paid_tri, n_sims=n_sims, rho=rho, hat_adj=True, random_state=random_seed
    )


def _run_odp_bf(paid_tri, exposure_tri, apriori=0.65, n_sims=1000, random_seed=None):
    """Run ODP bootstrap + Bornhuetter-Ferguson."""
    prepared = paid_tri.copy()
    prepared.key_labels = ["triangle_id"]
    prepared.kdims = np.asarray([["resample"]], dtype=object)

    sampler = cl.BootstrapODPSample(
        n_sims=n_sims, n_periods=-1, hat_adj=True, random_state=random_seed
    ).fit(prepared)
    resampled = sampler.transform(prepared)

    bf = cl.BornhuetterFerguson(apriori=apriori).fit(resampled, sample_weight=exposure_tri)

    ibnr_vals = np.asarray(bf.ibnr_.values)
    per_sim_per_origin = np.nansum(ibnr_vals, axis=-1)
    per_sim_per_origin = np.squeeze(per_sim_per_origin, axis=1)
    return per_sim_per_origin.T  # (n_origin, n_sims)


def _run_odp_cc(paid_tri, exposure_tri, trend=0.0, decay=1.0, n_sims=1000, random_seed=None):
    """Run ODP bootstrap + Cape Cod."""
    prepared = paid_tri.copy()
    prepared.key_labels = ["triangle_id"]
    prepared.kdims = np.asarray([["resample"]], dtype=object)

    sampler = cl.BootstrapODPSample(
        n_sims=n_sims, n_periods=-1, hat_adj=True, random_state=random_seed
    ).fit(prepared)
    resampled = sampler.transform(prepared)

    cc = cl.CapeCod(trend=trend, decay=decay).fit(resampled, sample_weight=exposure_tri)

    ibnr_vals = np.asarray(cc.ibnr_.values)
    per_sim_per_origin = np.nansum(ibnr_vals, axis=-1)
    per_sim_per_origin = np.squeeze(per_sim_per_origin, axis=1)
    return per_sim_per_origin.T  # (n_origin, n_sims)


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------

def _paid_to_date_per_origin(paid_tri):
    """Latest diagonal values per origin as a Series."""
    diag = paid_tri.latest_diagonal
    vals = np.asarray(diag.values).flatten()
    origins = [str(o) for o in paid_tri.origin]
    return pd.Series(vals, index=origins)


def _samples_to_rows(
    per_origin_per_sim,
    paid_to_date,
    origins,
    lob,
    group_id,
    method,
):
    """Convert (n_origin, n_sims) IBNR array to output rows."""
    rows = []
    total_paid = 0.0
    total_mean_ibnr = 0.0
    all_total_ibnr = None

    for i, origin in enumerate(origins):
        samples = per_origin_per_sim[i, :]
        samples = samples[np.isfinite(samples)]
        paid = float(paid_to_date.iloc[i]) if i < len(paid_to_date) else 0.0
        total_paid += paid

        if samples.size == 0:
            mean_ibnr = std_ibnr = 0.0
            p5 = p50 = p75 = p95 = 0.0
        else:
            mean_ibnr = float(np.mean(samples))
            std_ibnr = float(np.std(samples, ddof=1)) if samples.size > 1 else 0.0
            p5, p50, p75, p95 = np.percentile(samples, [5, 50, 75, 95])

        cv_ibnr = abs(std_ibnr / mean_ibnr) if mean_ibnr != 0 else float("nan")
        total_mean_ibnr += mean_ibnr

        if all_total_ibnr is None:
            all_total_ibnr = samples.copy()
        else:
            min_len = min(len(all_total_ibnr), len(samples))
            all_total_ibnr = all_total_ibnr[:min_len] + samples[:min_len]

        rows.append({
            "lob": lob,
            "group_id": group_id,
            "method": method,
            "accident_year": str(origin),
            "paid_to_date": paid,
            "mean_ultimate": paid + mean_ibnr,
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
        "method": method,
        "accident_year": "Total",
        "paid_to_date": total_paid,
        "mean_ultimate": total_paid + t_mean,
        "mean_ibnr": t_mean,
        "cv_ibnr": t_cv,
        "ibnr_p5": float(t_p5),
        "ibnr_p50": float(t_p50),
        "ibnr_p75": float(t_p75),
        "ibnr_p95": float(t_p95),
    })

    return rows


def run_methods_on_triangle(
    paid_tri,
    prem_series,
    methods,
    n_sims=1000,
    rho=0.5,
    apriori=0.65,
    random_seed=None,
    lob="unknown",
    group_id="unknown",
):
    """Run all requested methods on a single (paid, premium) triangle pair.

    Parameters
    ----------
    paid_tri : chainladder.Triangle
    prem_series : pd.Series or None
        Premium per origin year (int index). Required for odp_bf and odp_cc.
    methods : list[str]
        Any subset of {"mack", "odp", "odp_corr", "odp_bf", "odp_cc"}.
    n_sims : int
    rho : float
    apriori : float
    random_seed : int or None
    lob, group_id : str

    Returns
    -------
    list[dict] — rows for the output DataFrame
    """
    origins = [str(o) for o in paid_tri.origin]
    paid_per_origin = _paid_to_date_per_origin(paid_tri)
    exposure_tri = (
        _premium_as_exposure(paid_tri, prem_series)
        if prem_series is not None
        else None
    )

    all_rows = []

    for method in methods:
        try:
            if method == "mack":
                per_origin_sim, _ = _run_mack(paid_tri, n_samples=n_sims, random_seed=random_seed)
            elif method == "odp":
                per_origin_sim = _run_odp_bootstrap(paid_tri, n_sims=n_sims, random_seed=random_seed)
            elif method == "odp_corr":
                per_origin_sim = _run_correlated_odp(paid_tri, n_sims=n_sims, rho=rho, random_seed=random_seed)
            elif method == "odp_bf":
                if exposure_tri is None:
                    log.warning(
                        "lob=%s group_id=%s: skipping odp_bf (no premium data)", lob, group_id
                    )
                    continue
                per_origin_sim = _run_odp_bf(
                    paid_tri, exposure_tri, apriori=apriori, n_sims=n_sims, random_seed=random_seed
                )
            elif method == "odp_cc":
                if exposure_tri is None:
                    log.warning(
                        "lob=%s group_id=%s: skipping odp_cc (no premium data)", lob, group_id
                    )
                    continue
                per_origin_sim = _run_odp_cc(
                    paid_tri, exposure_tri, n_sims=n_sims, random_seed=random_seed
                )
            else:
                log.warning("Unknown method: %s — skipped", method)
                continue

            rows = _samples_to_rows(
                per_origin_sim, paid_per_origin, origins, lob, group_id, method
            )
            all_rows.extend(rows)

        except Exception as exc:
            log.error(
                "lob=%s group_id=%s method=%s failed: %s", lob, group_id, method, exc,
                exc_info=True,
            )

    return all_rows


def iterate_triangles(df, methods, n_sims=1000, rho=0.5, apriori=0.65, random_seed=None):
    """Iterate over all (lob, group_id) combinations and run all methods.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data with columns: origin, dev, paid, and optionally
        lob, group_id, premium.
    methods : list[str]
    n_sims, rho, apriori, random_seed : forwarded to run_methods_on_triangle

    Returns
    -------
    pd.DataFrame with output schema
    """
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
    for (lob, group_id), sub_df in iterator:
        try:
            paid_tri = df_to_triangle(sub_df, value_col="paid")

            # Build per-origin premium Series if data available.
            prem_series = None
            if "premium" in sub_df.columns and sub_df["premium"].notna().any():
                prem_series = (
                    sub_df.groupby("origin")["premium"]
                    .first()
                    .astype(float)
                )

            rows = run_methods_on_triangle(
                paid_tri,
                prem_series,
                methods=methods,
                n_sims=n_sims,
                rho=rho,
                apriori=apriori,
                random_seed=random_seed,
                lob=lob,
                group_id=group_id,
            )
            all_rows.extend(rows)

        except Exception as exc:
            log.error("lob=%s group_id=%s: failed to process triangle: %s", lob, group_id, exc, exc_info=True)

    if not all_rows:
        return pd.DataFrame(columns=[
            "lob", "group_id", "method", "accident_year", "paid_to_date",
            "mean_ultimate", "mean_ibnr", "cv_ibnr",
            "ibnr_p5", "ibnr_p50", "ibnr_p75", "ibnr_p95",
        ])

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Parallel wrapper
# ---------------------------------------------------------------------------

def _run_single_group(args):
    """Worker function for multiprocessing pool."""
    (lob, group_id), sub_df, methods, n_sims, rho, apriori, random_seed = args
    try:
        paid_tri = df_to_triangle(sub_df, value_col="paid")

        prem_series = None
        if "premium" in sub_df.columns and sub_df["premium"].notna().any():
            prem_series = sub_df.groupby("origin")["premium"].first().astype(float)

        return run_methods_on_triangle(
            paid_tri, prem_series, methods=methods,
            n_sims=n_sims, rho=rho, apriori=apriori,
            random_seed=random_seed, lob=lob, group_id=group_id,
        )
    except Exception as exc:
        log.error("lob=%s group_id=%s: worker failed: %s", lob, group_id, exc)
        return []


def iterate_triangles_parallel(
    df, methods, n_sims=1000, rho=0.5, apriori=0.65, random_seed=None, n_jobs=1
):
    """Parallel version of iterate_triangles using multiprocessing.Pool."""
    import multiprocessing

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
        ((lob, gid), sub, methods, n_sims, rho, apriori, random_seed)
        for (lob, gid), sub in groups
    ]

    with multiprocessing.Pool(processes=n_jobs) as pool:
        results = pool.map(_run_single_group, tasks)

    all_rows = [row for group_rows in results for row in group_rows]
    if not all_rows:
        return pd.DataFrame(columns=[
            "lob", "group_id", "method", "accident_year", "paid_to_date",
            "mean_ultimate", "mean_ibnr", "cv_ibnr",
            "ibnr_p5", "ibnr_p50", "ibnr_p75", "ibnr_p95",
        ])
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
        default=["mack", "odp", "odp_corr", "odp_bf", "odp_cc"],
        choices=["mack", "odp", "odp_corr", "odp_bf", "odp_cc"],
        metavar="METHOD",
        help=(
            "Methods to run. Choices: mack odp odp_corr odp_bf odp_cc. "
            "odp_bf and odp_cc require a 'premium' column."
        ),
    )
    p.add_argument("--n-sims", type=int, default=1000, help="Bootstrap simulation count")
    p.add_argument(
        "--rho", type=float, default=0.5,
        help="Calendar-year correlation for odp_corr (0=independent)"
    )
    p.add_argument(
        "--apriori", type=float, default=0.65,
        help="A-priori expected loss ratio for odp_bf"
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
        "--origin-col", default="origin", help="Name of the origin column"
    )
    p.add_argument(
        "--dev-col", default="dev", help="Name of the development column"
    )
    p.add_argument(
        "--paid-col", default="paid", help="Name of the cumulative paid column"
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
    if args.paid_col != "paid":
        rename[args.paid_col] = "paid"
    if rename:
        df = df.rename(columns=rename)

    for col in ["origin", "dev", "paid"]:
        if col not in df.columns:
            log.error("Required column '%s' not found. Available: %s", col, list(df.columns))
            sys.exit(1)

    df["origin"] = df["origin"].astype(int)
    df["dev"] = df["dev"].astype(int)
    df["paid"] = pd.to_numeric(df["paid"], errors="coerce")

    log.info(
        "Loaded %d rows — LOBs: %s",
        len(df),
        list(df["lob"].unique()) if "lob" in df.columns else ["all"],
    )
    log.info("Methods: %s | n_sims=%d | rho=%.2f | apriori=%.3f",
             args.methods, args.n_sims, args.rho, args.apriori)

    if args.n_jobs > 1:
        log.info("Running in parallel with %d workers", args.n_jobs)
        results = iterate_triangles_parallel(
            df,
            methods=args.methods,
            n_sims=args.n_sims,
            rho=args.rho,
            apriori=args.apriori,
            random_seed=args.random_seed,
            n_jobs=args.n_jobs,
        )
    else:
        results = iterate_triangles(
            df,
            methods=args.methods,
            n_sims=args.n_sims,
            rho=args.rho,
            apriori=args.apriori,
            random_seed=args.random_seed,
        )

    if results.empty:
        log.warning("No results produced. Check input data and method compatibility.")
    else:
        results.to_csv(args.output, index=False)
        log.info("Results written to %s (%d rows)", args.output, len(results))

        # Print a quick summary to stdout
        total_rows = results[results["accident_year"] == "Total"].copy()
        if not total_rows.empty:
            print("\n=== Total IBNR by method ===")
            for _, row in total_rows.iterrows():
                print(
                    f"  LOB={row['lob']} | group={row['group_id']} | "
                    f"method={row['method']:8s} | "
                    f"mean_ibnr={row['mean_ibnr']:>12,.0f} | "
                    f"cv={row['cv_ibnr']:.3f} | "
                    f"p95={row['ibnr_p95']:>12,.0f}"
                )

    return results


if __name__ == "__main__":
    main()
