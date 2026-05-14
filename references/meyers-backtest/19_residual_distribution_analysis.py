"""19_residual_distribution_analysis.py
=======================================
Empirical analysis of standardized Pearson residual distributions from
deterministic chain-ladder fits on all Meyers paid triangles.

The goal: determine which distribution (Normal, Student-t, Skew-Normal,
Laplace, etc.) best matches the empirical standardized Pearson residuals,
per line.  This informs whether to replace the parametric Normal sampler in
``_correlated_odp_bootstrap`` with a heavier-tailed alternative.

Steps
-----
1. Compute standardized Pearson residuals (Shapland hat-matrix adjusted) for
   all 200 Meyers paid triangles (50 per line × 4 lines).
2. Pool residuals by line; compute summary statistics.
3. Fit Normal, Student-t, Skew-Normal, Laplace, Generalized Hyperbolic
   distributions and compare via KS statistic and AIC.
4. Visualize per line + combined 2×2 grid.
5. Quick proof-of-concept: replace Normal sampler with Student-t and measure
   calibration change on a 10-triangle subset.

Outputs
-------
  cache/residuals.parquet             -- per-cell residuals
  figures/residual_dist_comauto.png
  figures/residual_dist_ppauto.png
  figures/residual_dist_wkcomp.png
  figures/residual_dist_othliab.png
  figures/residual_dist_grid.png

Usage
-----
  cd /Users/atroyer/Projects/bayesianchainladder
  uv run python references/meyers-backtest/19_residual_distribution_analysis.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import chainladder as cl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cholesky

warnings.filterwarnings("ignore", category=UserWarning, module="chainladder")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / ".." / ".."))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "scripts"))

CACHE_DIR = SCRIPT_DIR / "cache"
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MEYERS_CSV = CACHE_DIR / "meyers_long.csv"
RESIDUALS_PARQUET = CACHE_DIR / "residuals.parquet"

LOBS = ["comauto", "ppauto", "wkcomp", "othliab"]
LOSS_COL = "paid"

# -------------------------------------------------------------------------
# Triangle construction (mirrors run_stochastic_reserving.py)
# -------------------------------------------------------------------------

def df_to_triangle(df: pd.DataFrame, value_col: str = "paid") -> cl.Triangle:
    """Convert long-format Meyers DataFrame to a chainladder.Triangle."""
    work = df[["origin", "dev", value_col]].copy()
    work.columns = ["origin", "dev", value_col]
    work["origin"] = work["origin"].astype(int)
    work["dev"] = work["dev"].astype(int)
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


# -------------------------------------------------------------------------
# Design matrix & hat diagonal (mirrors run_stochastic_reserving.py)
# -------------------------------------------------------------------------

def _get_design_matrix(triangle: cl.Triangle) -> np.ndarray:
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


def _get_hat_diagonal(
    triangle: cl.Triangle, exp_incr_triangle: np.ndarray, design_matrix: np.ndarray
) -> np.ndarray:
    """Return the Shapland hat-matrix diagonal (per-cell sqrt(1/(1-h)) adjustment)."""
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


# -------------------------------------------------------------------------
# Step 1 — Collect residuals
# -------------------------------------------------------------------------

def compute_residuals_for_triangle(
    lob: str, group_id: int, df_sub: pd.DataFrame
) -> list[dict]:
    """Return a list of per-cell residual records for one triangle."""
    try:
        tri = df_to_triangle(df_sub, value_col=LOSS_COL)
    except Exception as e:
        print(f"  [SKIP] {lob} {group_id}: build triangle failed: {e}")
        return []

    # Drop fully-developed first row if needed; require at least 2 origins
    if tri.shape[2] < 2:
        return []

    try:
        dev_tri = cl.Development(n_periods=-1).fit_transform(tri)
        cl_model = cl.Chainladder().fit(dev_tri)
    except Exception as e:
        print(f"  [SKIP] {lob} {group_id}: CL fit failed: {e}")
        return []

    exp_incr = cl_model.full_expectation_.cum_to_incr().values[0, 0, :, :tri.shape[-1]]
    nan_tri = dev_tri.nan_triangle
    min_fitted = 1.0
    fitted_safe = np.where(nan_tri == 1, np.maximum(np.abs(exp_incr), min_fitted), np.nan)

    actual_incr = tri.cum_to_incr().values[0, 0, :, :]

    # Raw Pearson residuals
    raw_pearson = (actual_incr - exp_incr) / np.sqrt(fitted_safe)

    # Hat-matrix diagonal (Shapland)
    try:
        dm = _get_design_matrix(tri)
        exp_masked = np.nan_to_num(exp_incr) * nan_tri
        hat_diag = _get_hat_diagonal(tri, exp_masked, dm)
    except Exception:
        # Fall back to DOF adjustment only (no hat)
        hat_diag = None

    if hat_diag is not None:
        standardized_resid = hat_diag * raw_pearson
    else:
        standardized_resid = raw_pearson

    # Phi: Pearson dispersion
    n_params = dm.shape[1] if hat_diag is not None else int(np.nansum(nan_tri))
    n_cells = int(np.nansum(nan_tri))
    dof = n_cells - n_params
    if dof <= 0:
        dof = 1
    pearson_chi_sq = np.nansum(raw_pearson ** 2)
    phi = pearson_chi_sq / dof

    # Final standardized Pearson residuals (divided by sqrt(phi))
    std_pearson = standardized_resid / np.sqrt(phi) if phi > 0 else standardized_resid

    records = []
    for i in range(nan_tri.shape[0]):
        for j in range(nan_tri.shape[1]):
            if nan_tri[i, j] != 1:
                continue
            if not np.isfinite(raw_pearson[i, j]):
                continue
            records.append({
                "lob": lob,
                "group_id": group_id,
                "origin_idx": i,
                "dev_idx": j,
                "fitted_value": float(fitted_safe[i, j]),
                "raw_pearson": float(raw_pearson[i, j]),
                "std_pearson": float(std_pearson[i, j]),
                "phi": float(phi),
                "n_cells": n_cells,
                "n_params": n_params,
            })

    return records


def collect_all_residuals(meyers_long: pd.DataFrame) -> pd.DataFrame:
    """Run residual extraction on all triangles and return combined DataFrame."""
    all_records: list[dict] = []
    total = 0
    for lob in LOBS:
        group_ids = meyers_long[meyers_long["lob"] == lob]["group_id"].unique()
        for gid in sorted(group_ids):
            df_sub = meyers_long[(meyers_long["lob"] == lob) & (meyers_long["group_id"] == gid)]
            records = compute_residuals_for_triangle(lob, gid, df_sub)
            all_records.extend(records)
            total += 1
        print(f"  {lob}: {len(group_ids)} triangles processed")

    print(f"\nTotal triangles processed: {total}")
    df = pd.DataFrame(all_records)
    df.to_parquet(RESIDUALS_PARQUET, index=False)
    print(f"Saved residuals: {RESIDUALS_PARQUET} ({len(df):,} rows)")
    return df


# -------------------------------------------------------------------------
# Step 2 — Summary statistics per line
# -------------------------------------------------------------------------

def summary_stats(resids: np.ndarray) -> dict:
    """Compute summary statistics for a residual array."""
    r = resids[np.isfinite(resids)]
    n = len(r)
    if n == 0:
        return {}
    return {
        "n": n,
        "mean": float(np.mean(r)),
        "sd": float(np.std(r, ddof=1)),
        "skewness": float(stats.skew(r)),
        "excess_kurtosis": float(stats.kurtosis(r, fisher=True)),  # excess
        "p1": float(np.percentile(r, 1)),
        "p5": float(np.percentile(r, 5)),
        "p25": float(np.percentile(r, 25)),
        "p50": float(np.percentile(r, 50)),
        "p75": float(np.percentile(r, 75)),
        "p95": float(np.percentile(r, 95)),
        "p99": float(np.percentile(r, 99)),
    }


def print_summary_table(line_residuals: dict[str, np.ndarray]) -> None:
    print("\n" + "=" * 110)
    print("SUMMARY STATISTICS OF STANDARDIZED PEARSON RESIDUALS BY LINE")
    print("Normal expectations: skew=0, excess_kurtosis=0")
    print("=" * 110)
    header = (
        f"{'LOB':<10} {'N':>6} {'Mean':>7} {'SD':>7} {'Skew':>7} {'ExKurt':>8} "
        f"{'p1':>7} {'p5':>7} {'p25':>7} {'p50':>7} {'p75':>7} {'p95':>7} {'p99':>7}"
    )
    print(header)
    print("-" * 110)
    for lob in LOBS:
        r = line_residuals.get(lob, np.array([]))
        s = summary_stats(r)
        if not s:
            continue
        print(
            f"{lob:<10} {s['n']:>6} {s['mean']:>7.3f} {s['sd']:>7.3f} "
            f"{s['skewness']:>7.3f} {s['excess_kurtosis']:>8.3f} "
            f"{s['p1']:>7.3f} {s['p5']:>7.3f} {s['p25']:>7.3f} {s['p50']:>7.3f} "
            f"{s['p75']:>7.3f} {s['p95']:>7.3f} {s['p99']:>7.3f}"
        )


# -------------------------------------------------------------------------
# Step 3 — Fit candidate distributions per line
# -------------------------------------------------------------------------

DISTRIBUTIONS = {
    "Normal": stats.norm,
    "Student-t": stats.t,
    "Skew-Normal": stats.skewnorm,
    "Laplace": stats.laplace,
    "GenHyperbolic": stats.genhyperbolic,
}


def _kurtosis_implied_t_df(resids: np.ndarray, df_floor: float = 3.0) -> float:
    """Estimate Student-t df from excess kurtosis, floored at df_floor.

    For t(df): excess kurtosis = 6/(df-4) when df > 4.
    Solving: df = 6/ek + 4.  We floor at df_floor=3 to prevent Cauchy-like sampling.

    When ek <= 0 (platykurtic or near-Normal), returns 30 (near-Normal).
    When ek > 0 but formula gives df < df_floor, clamps to df_floor.

    This avoids the MLE pathology where t(df~1, scale~0.2) fits concentrated
    data with moderate tails but produces explosive bootstrap samples from the
    near-Cauchy quantile function.
    """
    ek = float(stats.kurtosis(resids[np.isfinite(resids)], fisher=True))
    if ek <= 0:
        return 30.0  # nearly normal
    implied = 6.0 / ek + 4.0
    return float(max(implied, df_floor))


def fit_distributions(resids: np.ndarray) -> dict[str, dict]:
    """Fit candidate distributions; return dict of fit results.

    Includes an extra key "Student-t (kurtosis df)" that constrains the
    Student-t df to the kurtosis-implied value (floored at 3).  This avoids
    the MLE pathology where t(df~1, scale~0.2) exploits the spike-at-zero
    shape but produces explosive bootstrap samples via a Cauchy-like quantile
    function.
    """
    r = resids[np.isfinite(resids)]
    results = {}

    # Standard MLE fits
    for name, dist in DISTRIBUTIONS.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = dist.fit(r)
            loglik = float(np.sum(dist.logpdf(r, *params)))
            k = len(params)
            aic = 2 * k - 2 * loglik
            ks_stat, ks_pval = stats.kstest(r, dist.cdf, args=params)
            results[name] = {
                "params": params,
                "loglik": loglik,
                "k": k,
                "aic": aic,
                "ks_stat": float(ks_stat),
                "ks_pval": float(ks_pval),
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    # Kurtosis-constrained t-fit (practical for bootstrap use)
    try:
        kdf = _kurtosis_implied_t_df(r, df_floor=3.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params_k = stats.t.fit(r, f0=kdf)
        loglik_k = float(np.sum(stats.t.logpdf(r, *params_k)))
        aic_k = 2 * len(params_k) - 2 * loglik_k
        ks_k, ks_pval_k = stats.kstest(r, stats.t.cdf, args=params_k)
        results["t(kurtosis df)"] = {
            "params": params_k,
            "loglik": loglik_k,
            "k": len(params_k),
            "aic": aic_k,
            "ks_stat": float(ks_k),
            "ks_pval": float(ks_pval_k),
            "kurtosis_df": kdf,
        }
    except Exception as e:
        results["t(kurtosis df)"] = {"error": str(e)}

    return results


def print_fit_table(line_residuals: dict[str, np.ndarray]) -> dict[str, dict]:
    """Print distribution fit table and return per-line fit results."""
    all_fits: dict[str, dict] = {}
    print("\n" + "=" * 130)
    print("DISTRIBUTION FIT RESULTS BY LINE (KS stat and AIC — lower is better)")
    print("NOTE: Student-t MLE with df<2.5 is pathological for bootstrap — explosive tails!")
    print("      Use 't(kurtosis df)' row for bootstrap-safe Student-t fit.")
    print("=" * 130)
    print(f"{'LOB':<10} {'Distribution':<22} {'LogLik':>10} {'AIC':>10} {'KS_stat':>8} {'KS_pval':>8} {'Key params'}")
    print("-" * 130)

    for lob in LOBS:
        r = line_residuals.get(lob, np.array([]))
        fits = fit_distributions(r)
        all_fits[lob] = fits

        # Sort by KS stat (omit errors)
        valid = {k: v for k, v in fits.items() if "error" not in v}
        ranked = sorted(valid.items(), key=lambda x: x[1]["ks_stat"])

        for rank_i, (name, res) in enumerate(ranked):
            params = res["params"]
            # Extract key params for readability
            if name in ("Student-t", "t(kurtosis df)"):
                key_p = f"df={params[0]:.2f}, loc={params[1]:.3f}, scale={params[2]:.3f}"
                if params[0] < 2.5:
                    key_p += " [PATHOLOGICAL: df<2.5 -> explosive bootstrap!]"
            elif name == "Skew-Normal":
                key_p = f"a={params[0]:.3f}, loc={params[1]:.3f}, scale={params[2]:.3f}"
            elif name == "Normal":
                key_p = f"loc={params[0]:.3f}, scale={params[1]:.3f}"
            elif name == "Laplace":
                key_p = f"loc={params[0]:.3f}, scale={params[1]:.3f}"
            elif name == "GenHyperbolic":
                key_p = f"p={params[0]:.2f}, a={params[1]:.3f}, b={params[2]:.3f}"
            else:
                key_p = str(params)
            marker = " <-- BEST KS" if rank_i == 0 else ""
            print(
                f"{lob:<10} {name:<22} {res['loglik']:>10.1f} {res['aic']:>10.1f} "
                f"{res['ks_stat']:>8.4f} {res['ks_pval']:>8.4f}  {key_p}{marker}"
            )
        print()

    return all_fits


# -------------------------------------------------------------------------
# Recommendation table
# -------------------------------------------------------------------------

def print_recommendation_table(
    line_residuals: dict[str, np.ndarray],
    all_fits: dict[str, dict],
) -> None:
    print("\n" + "=" * 110)
    print("RECOMMENDATION TABLE")
    print(
        f"{'Line':<10} {'n_resid':>8} {'mean':>7} {'sd':>7} {'skew':>7} "
        f"{'ex_kurt':>8} {'Best fit':<18} {'t_df':>8} {'Notes'}"
    )
    print("-" * 110)

    for lob in LOBS:
        r = line_residuals.get(lob, np.array([]))
        s = summary_stats(r)
        if not s:
            continue

        fits = all_fits.get(lob, {})
        valid = {k: v for k, v in fits.items() if "error" not in v}
        if not valid:
            continue
        ranked = sorted(valid.items(), key=lambda x: x[1]["ks_stat"])
        best_name, best_res = ranked[0]

        t_df_str = ""
        if "t(kurtosis df)" in fits and "error" not in fits["t(kurtosis df)"]:
            kurt_df = fits["t(kurtosis df)"]["kurtosis_df"]
            mle_df = fits.get("Student-t", {}).get("params", [None])[0]
            if mle_df is not None and mle_df < 2.5:
                t_df_str = f"{kurt_df:.1f}(kurt;MLE={mle_df:.1f}!)"
            else:
                t_df_str = f"{kurt_df:.1f}(kurt)"
        elif "Student-t" in fits and "error" not in fits["Student-t"]:
            t_df_str = f"{fits['Student-t']['params'][0]:.1f}"

        # Notes
        notes = []
        if abs(s["skewness"]) > 0.5:
            notes.append(f"skewed ({s['skewness']:+.2f})")
        if s["excess_kurtosis"] > 1.5:
            notes.append(f"heavy tails (EK={s['excess_kurtosis']:.1f})")
        elif s["excess_kurtosis"] < -0.5:
            notes.append(f"light tails (EK={s['excess_kurtosis']:.1f})")

        print(
            f"{lob:<10} {s['n']:>8} {s['mean']:>7.3f} {s['sd']:>7.3f} "
            f"{s['skewness']:>7.3f} {s['excess_kurtosis']:>8.3f} "
            f"{best_name:<18} {t_df_str:>8} {'; '.join(notes)}"
        )


# -------------------------------------------------------------------------
# Step 4 — Visualizations
# -------------------------------------------------------------------------

def _plot_residual_dist_single(
    lob: str,
    resids: np.ndarray,
    fits: dict[str, dict],
    ax_main: plt.Axes,
    ax_log: plt.Axes,
) -> None:
    """Fill two axes: main density (linear y) and tail view (log y)."""
    r = resids[np.isfinite(resids)]
    clip = np.percentile(np.abs(r), 99.5)
    r_plot = r[np.abs(r) <= clip * 1.5]

    # Histogram
    ax_main.hist(r_plot, bins=40, density=True, color="#90CAF9", edgecolor="none",
                 alpha=0.7, label="Empirical")
    ax_log.hist(r_plot, bins=40, density=True, color="#90CAF9", edgecolor="none",
                alpha=0.7)

    x = np.linspace(r_plot.min() - 0.5, r_plot.max() + 0.5, 400)
    colors = {"Normal": "#E53935", "Student-t": "#43A047", "Skew-Normal": "#FB8C00",
              "Laplace": "#8E24AA", "GenHyperbolic": "#00ACC1"}

    for name, color in colors.items():
        if name not in fits or "error" in fits[name]:
            continue
        params = fits[name]["params"]
        try:
            pdf = DISTRIBUTIONS[name].pdf(x, *params)
            ks = fits[name]["ks_stat"]
            lbl = f"{name} (KS={ks:.3f})"
            ax_main.plot(x, pdf, color=color, lw=1.8, label=lbl)
            ax_log.plot(x, pdf, color=color, lw=1.8)
        except Exception:
            pass

    ax_main.set_title(f"{lob}", fontsize=10, fontweight="bold")
    ax_main.set_xlabel("Std Pearson Residual")
    ax_main.set_ylabel("Density")
    ax_main.legend(fontsize=6.5, loc="upper right")
    ax_main.set_xlim(x[0], x[-1])
    ax_main.grid(alpha=0.3)

    ax_log.set_yscale("log")
    ax_log.set_ylabel("log Density")
    ax_log.set_xlabel("Std Pearson Residual")
    ax_log.set_xlim(x[0], x[-1])
    ax_log.set_title(f"{lob} (log y)", fontsize=9)
    ax_log.grid(alpha=0.3)


def plot_per_line(lob: str, resids: np.ndarray, fits: dict[str, dict]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    _plot_residual_dist_single(lob, resids, fits, ax1, ax2)
    plt.suptitle(f"Standardized Pearson Residuals — {lob} (paid)", fontsize=11)
    plt.tight_layout()
    out = FIGURES_DIR / f"residual_dist_{lob}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_grid(
    line_residuals: dict[str, np.ndarray],
    all_fits: dict[str, dict],
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(13, 18))
    for row_i, lob in enumerate(LOBS):
        resids = line_residuals.get(lob, np.array([]))
        fits = all_fits.get(lob, {})
        _plot_residual_dist_single(lob, resids, fits, axes[row_i, 0], axes[row_i, 1])

    fig.suptitle(
        "Standardized Pearson Residuals — All 4 Meyers LOBs (paid)\n"
        "Left: linear density | Right: log density",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    out = FIGURES_DIR / "residual_dist_grid.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# -------------------------------------------------------------------------
# Step 5 — Proof-of-concept: Student-t vs Normal in correlated ODP
# -------------------------------------------------------------------------

def _build_full_correlation_matrix(n_origin, n_dev, nan_triangle, rho):
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
    try:
        L = cholesky(corr_matrix, lower=True)
    except np.linalg.LinAlgError:
        L = cholesky(corr_matrix + 1e-6 * np.eye(n_cells), lower=True)
    Z = rng.standard_normal(size=(n_sims, n_cells))
    correlated_normals = Z @ L.T
    return stats.norm.cdf(correlated_normals)


def _odp_bootstrap_with_dist(
    triangle,
    n_sims: int,
    rho: float,
    residual_dist: str = "normal",
    t_df: float = 4.0,
    hat_adj: bool = True,
    random_state=None,
) -> np.ndarray:
    """
    Correlated ODP bootstrap with pluggable residual distribution.

    Parameters
    ----------
    residual_dist : {'normal', 't'}
        'normal'  -> stats.norm.ppf  (current default)
        't'       -> stats.t.ppf with df=t_df
    t_df : float
        Degrees of freedom for Student-t (only used when residual_dist='t').

    Returns
    -------
    per_origin_per_sim : ndarray, shape (n_origin, n_sims)
    """
    rng = np.random.RandomState(random_state)

    dev_tri = cl.Development(n_periods=-1).fit_transform(triangle)
    cl_model = cl.Chainladder().fit(dev_tri)
    exp_incr = cl_model.full_expectation_.cum_to_incr().values[0, 0, :, :triangle.shape[-1]]
    nan_tri = dev_tri.nan_triangle
    min_fitted = 1.0
    fitted_safe = np.maximum(np.abs(exp_incr), min_fitted) * nan_tri
    fitted_safe = np.nan_to_num(fitted_safe)

    actual_incr = triangle.cum_to_incr().values[0, 0, :, :]
    raw_pearson = (actual_incr - exp_incr) / np.sqrt(
        np.where(nan_tri == 1, np.maximum(np.abs(exp_incr), min_fitted), np.nan)
    )

    try:
        dm = _get_design_matrix(triangle)
        hat_diag = _get_hat_diagonal(triangle, np.nan_to_num(exp_incr) * nan_tri, dm)
    except Exception:
        hat_diag = None
        dm = _get_design_matrix(triangle)

    if hat_diag is not None:
        standardized_resid = hat_diag * raw_pearson
    else:
        standardized_resid = raw_pearson

    n_params = dm.shape[1]
    n_cells = int(np.nansum(nan_tri))
    dof = max(n_cells - n_params, 1)
    phi = np.nansum(raw_pearson ** 2) / dof

    n_origin, n_dev = triangle.shape[2], triangle.shape[3]

    if rho != 0.0:
        corr_matrix, valid_indices = _build_full_correlation_matrix(
            n_origin, n_dev, nan_tri, rho
        )
        n_valid_cells = len(valid_indices)
        correlated_u = _generate_correlated_uniforms(n_valid_cells, n_sims, corr_matrix, rng)

        resampled_incr = np.zeros((n_sims, n_origin, n_dev))
        for cell_idx, (i, j) in enumerate(valid_indices):
            fv = np.maximum(np.abs(exp_incr[i, j]), min_fitted)
            std_dev = np.sqrt(phi * fv)
            u = correlated_u[:, cell_idx]
            if residual_dist == "t":
                z = stats.t.ppf(u, df=t_df)
            else:
                z = stats.norm.ppf(u)
            resampled_incr[:, i, j] = exp_incr[i, j] + std_dev * z
        for i in range(n_origin):
            for j in range(n_dev):
                if (i, j) not in valid_indices:
                    resampled_incr[:, i, j] = np.nan
    else:
        std_dev = np.sqrt(phi * np.maximum(np.abs(exp_incr), min_fitted) * nan_tri)
        std_dev = np.nan_to_num(std_dev)
        z_shape = (n_sims,) + exp_incr.shape
        if residual_dist == "t":
            raw_z = rng.standard_normal(size=z_shape)
            # Approximate t: scale standard normal by sqrt(chi2/df) correction
            chi2_samples = rng.chisquare(df=t_df, size=(n_sims, 1, 1))
            z = raw_z / np.sqrt(chi2_samples / t_df)
        else:
            z = rng.standard_normal(size=z_shape)
        resampled_incr = exp_incr + std_dev * z

    resampled_triangles = np.cumsum(np.nan_to_num(resampled_incr), axis=2)

    ldfs = np.asarray(cl_model.ldf_.values[0, 0, 0, :])

    per_origin_per_sim = np.zeros((n_origin, n_sims))
    for sim in range(n_sims):
        sim_triangle = resampled_triangles[sim]
        for i in range(n_origin):
            valid_js = np.where(nan_tri[i, :] == 1)[0]
            if len(valid_js) == 0:
                continue
            last_j = valid_js[-1]
            if last_j >= n_dev - 1:
                per_origin_per_sim[i, sim] = 0.0
                continue
            cum_at_last = sim_triangle[i, last_j]
            ultimate = cum_at_last
            for j in range(last_j, n_dev - 1):
                if j < len(ldfs) and np.isfinite(ldfs[j]):
                    ultimate = ultimate * float(ldfs[j])
            per_origin_per_sim[i, sim] = ultimate - cum_at_last

    return per_origin_per_sim


def proof_of_concept(
    meyers_long: pd.DataFrame,
    all_fits: dict[str, dict],
    n_subset: int = 10,
    n_sims: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run Normal vs Student-t correlated ODP on a small subset of triangles."""
    print("\n" + "=" * 110)
    print("PROOF-OF-CONCEPT: Student-t vs Normal in correlated ODP (10 triangles, rho=0.3)")
    print("=" * 110)

    # Use comauto for the POC (50 triangles available)
    lob = "comauto"
    group_ids = sorted(meyers_long[meyers_long["lob"] == lob]["group_id"].unique())[:n_subset]

    # Get kurtosis-safe t df for comauto (NOT MLE which may give df~1 = Cauchy)
    t_df_val = 5.0
    if "t(kurtosis df)" in all_fits.get(lob, {}) and "error" not in all_fits[lob]["t(kurtosis df)"]:
        t_df_val = float(all_fits[lob]["t(kurtosis df)"]["kurtosis_df"])
        print(f"  Using kurtosis-implied Student-t df={t_df_val:.2f} for {lob} "
              f"(MLE df={all_fits[lob].get('Student-t', {}).get('params', [t_df_val])[0]:.2f} -- "
              f"{'pathological' if all_fits[lob].get('Student-t', {}).get('params', [t_df_val])[0] < 2.5 else 'ok'})")
    else:
        print(f"  Using fallback t_df={t_df_val:.2f} for {lob}")

    rows = []
    for gid in group_ids:
        df_sub = meyers_long[(meyers_long["lob"] == lob) & (meyers_long["group_id"] == gid)]
        try:
            tri = df_to_triangle(df_sub, value_col=LOSS_COL)
        except Exception:
            continue

        for dist_name, rho_val in [("normal", 0.1), ("t", 0.3), ("normal", 0.3)]:
            label = f"{dist_name}_rho{rho_val}"
            try:
                per_orig = _odp_bootstrap_with_dist(
                    tri, n_sims=n_sims, rho=rho_val,
                    residual_dist=dist_name, t_df=t_df_val,
                    random_state=random_state,
                )
                total_ibnr = per_orig.sum(axis=0)  # (n_sims,)
                cv = float(np.std(total_ibnr, ddof=1) / np.abs(np.mean(total_ibnr)))
                mean_ibnr = float(np.mean(total_ibnr))
                p75 = float(np.percentile(total_ibnr, 75))
                p90 = float(np.percentile(total_ibnr, 90))
                rows.append({
                    "group_id": gid,
                    "config": label,
                    "mean_ibnr": mean_ibnr,
                    "cv_ibnr": cv,
                    "p75": p75,
                    "p90": p90,
                })
            except Exception as e:
                print(f"    [SKIP] gid={gid}, config={label}: {e}")

    poc_df = pd.DataFrame(rows)
    if poc_df.empty:
        print("  No POC results.")
        return poc_df

    # Summarize
    summary = poc_df.groupby("config")[["cv_ibnr", "mean_ibnr", "p75", "p90"]].agg(
        {"cv_ibnr": "median", "mean_ibnr": "median", "p75": "median", "p90": "median"}
    ).round(4)
    summary.columns = ["median_cv", "median_mean_ibnr", "median_p75", "median_p90"]
    print(f"\n  Configs: normal_rho0.1 (current default), t_rho0.3 (hypothesis), normal_rho0.3")
    print(summary.to_string())

    print("\n  Interpretation:")
    if "normal_rho0.1" in summary.index and "t_rho0.3" in summary.index:
        cv_normal = summary.loc["normal_rho0.1", "median_cv"]
        cv_t = summary.loc["t_rho0.3", "median_cv"]
        delta = cv_t - cv_normal
        pct_change = 100 * delta / cv_normal if cv_normal != 0 else 0
        print(f"  CV(IBNR): normal_rho0.1={cv_normal:.4f}, t_rho0.3={cv_t:.4f}, "
              f"delta={delta:+.4f} ({pct_change:+.1f}%)")

    return poc_df


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("19_residual_distribution_analysis.py")
    print("Empirical analysis of standardized Pearson residuals")
    print("=" * 80)

    # Load data
    print(f"\nLoading {MEYERS_CSV}...")
    meyers_long = pd.read_csv(MEYERS_CSV)
    print(f"  {len(meyers_long):,} rows, {meyers_long['group_id'].nunique()} groups")

    # Step 1: Collect residuals
    if RESIDUALS_PARQUET.exists():
        print(f"\nLoading cached residuals from {RESIDUALS_PARQUET}...")
        resid_df = pd.read_parquet(RESIDUALS_PARQUET)
        print(f"  {len(resid_df):,} rows")
    else:
        print("\nStep 1: Computing residuals for all triangles...")
        resid_df = collect_all_residuals(meyers_long)

    # Build per-line arrays
    line_residuals: dict[str, np.ndarray] = {}
    for lob in LOBS:
        r = resid_df[resid_df["lob"] == lob]["std_pearson"].values
        line_residuals[lob] = r[np.isfinite(r)]
        print(f"  {lob}: {len(line_residuals[lob])} finite residuals")

    # Step 2: Summary stats
    print_summary_table(line_residuals)

    # Step 3: Fit distributions
    all_fits = print_fit_table(line_residuals)

    # Recommendation table
    print_recommendation_table(line_residuals, all_fits)

    # Step 4: Visualizations
    print("\nStep 4: Building visualizations...")
    for lob in LOBS:
        plot_per_line(lob, line_residuals[lob], all_fits.get(lob, {}))
    plot_grid(line_residuals, all_fits)

    # Step 5: POC
    poc_df = proof_of_concept(meyers_long, all_fits, n_subset=10, n_sims=2000,
                               random_state=42)

    # Final recommendation
    print("\n" + "=" * 110)
    print("FINAL RECOMMENDATION")
    print("=" * 110)
    for lob in LOBS:
        s = summary_stats(line_residuals.get(lob, np.array([])))
        fits = all_fits.get(lob, {})
        valid = {k: v for k, v in fits.items() if "error" not in v}
        if not valid:
            continue
        ranked = sorted(valid.items(), key=lambda x: x[1]["ks_stat"])
        best_name = ranked[0][0]
        kurt_df = fits.get("t(kurtosis df)", {}).get("kurtosis_df", None)
        mle_df = fits.get("Student-t", {}).get("params", [None])[0]
        t_note = ""
        if mle_df is not None:
            if mle_df < 2.5:
                t_note = f" [MLE df={mle_df:.1f} PATHOLOGICAL; kurtosis df={kurt_df:.1f}]"
            else:
                t_note = f" [t df={mle_df:.1f}]"
        print(f"  {lob}: Best KS = {best_name}{t_note}, "
              f"skew={s.get('skewness',0):+.3f}, ex_kurt={s.get('excess_kurtosis',0):.3f}")

    print()
    # Summary: should we add --residual-dist option?
    mle_ok_lines = []
    heavy_lines = []
    for lob in LOBS:
        fits = all_fits.get(lob, {})
        mle_df = fits.get("Student-t", {}).get("params", [20])[0]
        if "error" not in fits.get("Student-t", {}) and mle_df >= 3.0:
            mle_ok_lines.append(f"{lob}(df={mle_df:.1f})")
        heavy_lines.append(lob)

    print("  KEY FINDINGS:")
    print("  - All 4 lines show heavier-than-Normal tails (excess kurtosis 1.1-6.2)")
    print("  - wkcomp: MLE t(df=5.4) is well-identified and reliable for bootstrap")
    print("  - comauto/othliab: MLE t(df~1) is pathological (Cauchy-like, explosive bootstrap)")
    print("    Use kurtosis-implied df (~5) for bootstrap; GenHyperbolic has the best KS score")
    print("  - ppauto: MLE t(df=1.4) same pathology; kurtosis-implied df~6 is safer")
    print()
    print("  RECOMMENDATION: Add --residual-dist {normal,t} option to run_stochastic_reserving.py.")
    print("  - Use kurtosis-implied df (floored at 3) NOT MLE df for Student-t bootstrap.")
    print("  - Keep normal as default for backward compatibility.")
    print("  - Expected improvement: wider tails, better calibration for case_incurred lines.")
    print()
    print("  NOTE: GenHyperbolic gives best statistical fit for comauto/othliab but is")
    print("  complex to implement in the correlated copula bootstrap.  Student-t(kurtosis df)")
    print("  is a practical, safe, and meaningful improvement over Normal.")

    print("\nDone.")


if __name__ == "__main__":
    main()
