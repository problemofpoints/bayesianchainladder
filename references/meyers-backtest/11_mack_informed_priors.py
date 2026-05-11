"""11_mack_informed_priors.py — Test Mack-informed (CL-anchored) empirical Bayes priors.

The key advance over 10_data_informed_priors.py:
  - Prior means are derived from deterministic chain ladder (CL) ultimates,
    NOT from raw empirical means of observed incremental cells.
  - This eliminates the truncation-bias problem where sparse later origins
    only have early (large) dev periods observed, inflating their empirical mean.

Construction:
  - C(origin) priors: mu_k = log(ult_k_CL) - log(ult_ref_CL)
      -> log-ratio of CL ultimates (all on same basis, no truncation bias)
  - C(dev) priors: mu_j = log(incr_pct_j) - log(incr_pct_1)
      -> log-ratio of incremental fractions implied by LDFs
  - Intercept: Normal(log(ult_ref * incr_pct_1), 2.0)
      -> centers at CL-implied reference cell value
  - All with sigma=0.3 (tight enough to anchor, loose enough for data to update)

Previous results (commit bb07e94):
  - Default priors Normal(0, σ=1):        Celina ultimate 184,629 (+42% over Mack)
  - Naive empirical Bayes (raw means, σ=0.3): ~151,749 (+17% over Mack)
  - Failure mode: sparse origins have inflated empirical means → biased prior centers

Run with:
    cd references/meyers-backtest
    uv run python 11_mack_informed_priors.py
"""
from __future__ import annotations

import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

DIVIDER = "=" * 72


def section(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# ---------------------------------------------------------------------------
# Reference benchmarks (from prior analysis, commit d8ac08e / bb07e94)
# ---------------------------------------------------------------------------

MACK_CELINA = 129_779.0
ACTUAL_CELINA = 125_467.0
DEFAULT_GLM_CELINA = 184_629.0       # Normal(0, sigma=1) priors
NAIVE_EB_CELINA = 151_749.0          # naive empirical Bayes (sigma=0.3)
BOOTSTRAP_ODP_CELINA = 129_701.0

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import bambi as bmb
import chainladder as cl
import reservetestr as rt

from bayesianchainladder import BayesianChainLadderGLM


# ---------------------------------------------------------------------------
# Core helper: build Mack-informed priors from a chainladder Triangle
# ---------------------------------------------------------------------------

def build_mack_priors(triangle: "cl.Triangle", sigma: float = 0.3) -> dict:
    """Construct Bambi priors for C(origin) + C(dev) GLM using CL ultimates.

    The reference origin is the FIRST origin (row 0).  Bambi's treatment
    contrasts for C(origin) / C(dev) map the second through last levels
    relative to the first.

    Parameters
    ----------
    triangle : cl.Triangle
        Cumulative paid loss triangle (observed only).
    sigma : float
        Prior SD for all C(origin) and C(dev) contrasts.

    Returns
    -------
    dict with keys:
        priors        : dict of bambi.Prior objects (ready for BayesianChainLadderGLM)
        origin_mus    : np.ndarray of origin contrast prior means (n_origins - 1)
        dev_mus       : np.ndarray of dev contrast prior means (n_devs - 1)
        intercept_mu  : float
        origin_ults   : np.ndarray of CL per-origin ultimates (n_origins)
        incr_fracs    : np.ndarray of expected incremental fraction per dev (n_devs)
        origin_labels : list of str (years, second through last)
        dev_labels    : list of int (dev periods, second through last)
    """
    # --- Fit deterministic chain ladder ---
    fitted_cl = cl.Chainladder().fit(triangle)

    # Per-origin CL ultimates
    ult_arr = np.asarray(fitted_cl.ultimate_.to_frame().values, dtype=float).flatten()

    # CDF to ultimate: shape (1, n_cdf_cols)
    cdf_df = fitted_cl.cdf_.to_frame()
    cdf_arr = np.asarray(cdf_df.values, dtype=float).flatten()

    # Dev periods in the triangle
    devs = list(triangle.to_frame().columns)
    n_devs = len(devs)

    # Clip CDF to the triangle's dev periods (CDF may have extra tail columns)
    cdf_for_devs = cdf_arr[:n_devs]

    # pct_developed_at_j = 1 / CDF_j
    pct_dev = 1.0 / cdf_for_devs

    # Expected incremental fraction at each dev period
    incr_fracs = np.diff(np.concatenate([[0.0], pct_dev]))

    # Guard: if any incr_frac is <= 0 (can happen for tail stubs), floor at 1e-8
    incr_fracs = np.maximum(incr_fracs, 1e-8)

    # Reference values
    ref_ult = ult_arr[0]
    ref_frac = incr_fracs[0]

    # C(origin) contrasts: log(ult_k) - log(ult_ref) for k=1..n_origins-1
    origin_mus = np.log(ult_arr[1:]) - np.log(ref_ult)

    # C(dev) contrasts: log(incr_frac_j) - log(incr_frac_ref) for j=1..n_devs-1
    dev_mus = np.log(incr_fracs[1:]) - np.log(ref_frac)

    # Intercept: log of CL-implied expected value in reference cell
    intercept_mu = float(np.log(ref_ult * ref_frac))

    # Origin labels (years as strings)
    cum_df = triangle.to_frame()
    origin_labels = [str(o)[:4] for o in cum_df.index[1:]]

    priors = {
        "Intercept": bmb.Prior("Normal", mu=intercept_mu, sigma=2.0),
        "C(origin)": bmb.Prior(
            "Normal",
            mu=origin_mus,
            sigma=np.full(len(origin_mus), sigma, dtype=float),
        ),
        "C(dev)": bmb.Prior(
            "Normal",
            mu=dev_mus,
            sigma=np.full(len(dev_mus), sigma, dtype=float),
        ),
    }

    return {
        "priors": priors,
        "origin_mus": origin_mus,
        "dev_mus": dev_mus,
        "intercept_mu": intercept_mu,
        "origin_ults": ult_arr,
        "incr_fracs": incr_fracs,
        "origin_labels": origin_labels,
        "dev_labels": devs[1:],
    }


# ---------------------------------------------------------------------------
# Helper: fit GLM and extract ultimate posterior statistics
# ---------------------------------------------------------------------------

def fit_and_extract(
    triangle: "cl.Triangle",
    priors_dict: dict | None,
    label: str,
    actual_ultimate: float,
    latest_observed: float,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 2,
) -> dict:
    """Fit BayesianChainLadderGLM and return summary statistics."""
    print(f"\n  Fitting: {label} ...")

    # Detect family compatibility
    inc_vals = np.asarray(triangle.cum_to_incr().to_frame().values, dtype=float)
    obs_vals = inc_vals[~np.isnan(inc_vals)]
    family = "gamma" if (obs_vals > 0).all() else "gaussian"
    if family == "gaussian":
        print(f"    (using gaussian — triangle has non-positive incrementals)")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        glm = BayesianChainLadderGLM(
            formula="incremental ~ 1 + C(origin) + C(dev)",
            family=family,
            link="log",
            exposure=None,
            priors=priors_dict,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.95,
            random_seed=42,
        ).fit(triangle)

    reserves_post = glm.reserves_posterior_
    total_ibnr_samples = np.asarray(reserves_post.sum(dim="origin").values, dtype=float)
    total_ult_samples = total_ibnr_samples + latest_observed
    finite_ult = total_ult_samples[np.isfinite(total_ult_samples)]

    ult_median = float(np.median(finite_ult))
    ult_mean = float(np.mean(finite_ult))
    ult_sd = float(np.std(finite_ult, ddof=1))
    ult_p5 = float(np.percentile(finite_ult, 5))
    ult_p95 = float(np.percentile(finite_ult, 95))
    implied_pctl = float(np.mean(finite_ult <= actual_ultimate))

    return {
        "label": label,
        "ult_median": ult_median,
        "ult_mean": ult_mean,
        "ult_sd": ult_sd,
        "ult_p5": ult_p5,
        "ult_p95": ult_p95,
        "implied_pctl": implied_pctl,
        "finite_ult": finite_ult,
    }


# ---------------------------------------------------------------------------
# TASK 1 — Construct Mack-informed priors for Celina and print them
# ---------------------------------------------------------------------------
section("TASK 1 — Construct Mack-informed priors for Celina (ppauto/353)")

recs = rt.build_triangle_records()
celina_rec = [rec for rec in recs if rec.line == "ppauto" and rec.group_id == 353][0]
print(f"Company: {celina_rec.company}  |  line: {celina_rec.line}  |  group_id: {celina_rec.group_id}")

tri_celina = celina_rec.train_triangles["paid"]
actual_celina = float(celina_rec.actual_ultimates["paid"])

# Mack CL reference
mack_celina = cl.MackChainladder().fit(tri_celina)
mack_ult_arr = np.asarray(mack_celina.ultimate_.values, dtype=float).squeeze()
mack_total = float(np.nansum(mack_ult_arr))
latest_diag = np.asarray(mack_celina.latest_diagonal.values, dtype=float).squeeze()
latest_celina = float(np.nansum(latest_diag))

print(f"\nMack total ultimate: {mack_total:>12,.1f}")
print(f"Latest observed:     {latest_celina:>12,.1f}")
print(f"Actual ultimate:     {actual_celina:>12,.1f}")

# Build Mack-informed priors
mack_priors_info = build_mack_priors(tri_celina, sigma=0.3)

print(f"\nIntercept prior: Normal(mu={mack_priors_info['intercept_mu']:.4f}, sigma=2.0)")
print(f"  (log of CL-implied reference cell: exp({mack_priors_info['intercept_mu']:.4f}) = "
      f"{np.exp(mack_priors_info['intercept_mu']):,.1f})")

print(f"\nC(origin) prior means (log-ratio of CL ultimates vs 1988):")
for yr, mu, ult in zip(
    mack_priors_info["origin_labels"],
    mack_priors_info["origin_mus"],
    mack_priors_info["origin_ults"][1:],
):
    flag = " <-- SPARSE (late origin)" if int(yr) >= 1995 else ""
    print(f"  T.{yr}: Normal(mu={mu:+.4f}, sigma=0.30)  [CL ult={ult:,.1f}]{flag}")

print(f"\nC(dev) prior means (log-ratio of CL incremental fractions vs dev=12):")
devs_all = list(tri_celina.to_frame().columns)
for d, mu, frac in zip(
    mack_priors_info["dev_labels"],
    mack_priors_info["dev_mus"],
    mack_priors_info["incr_fracs"][1:],
):
    print(f"  T.{d}: Normal(mu={mu:+.4f}, sigma=0.30)  [incr_frac={frac:.6f}]")

print(f"\nReference origin (1988) CL ultimate: {mack_priors_info['origin_ults'][0]:,.1f}")
print(f"Reference dev (12) incremental fraction: {mack_priors_info['incr_fracs'][0]:.6f}")
print(f"  -> expected dev=12 incremental for 1988: "
      f"{mack_priors_info['origin_ults'][0] * mack_priors_info['incr_fracs'][0]:,.1f}")

# Compare to naive empirical Bayes (problem case: 1995)
print("\nKey comparison — naive vs Mack-informed for sparse origin 1995:")
inc_df_celina = tri_celina.cum_to_incr().to_frame()
vals_1995 = inc_df_celina.iloc[7].dropna().values
vals_1995 = vals_1995[vals_1995 > 0]
naive_mu_1995 = float(np.log(vals_1995.mean()) - np.log(inc_df_celina.iloc[0].dropna().values.mean()))
mack_mu_1995 = float(mack_priors_info["origin_mus"][6])  # index 6 = 1995 (1989=0,..,1995=6)
print(f"  1995 naive empirical mu:  {naive_mu_1995:+.4f}  "
      f"[biased by early-dev-only observation: {vals_1995.mean():,.1f}]")
print(f"  1995 Mack-informed mu:    {mack_mu_1995:+.4f}  "
      f"[CL ultimate properly accounts for unobserved tail]")
print(f"  Difference: {mack_mu_1995 - naive_mu_1995:+.4f} log-units "
      f"-> {100*(np.exp(mack_mu_1995 - naive_mu_1995) - 1):+.1f}% shift in prior center")


# ---------------------------------------------------------------------------
# TASK 2 — Fit Celina with Mack-informed priors vs Default vs Naive EB
# ---------------------------------------------------------------------------
section("TASK 2 — Fit Celina: Mack-informed vs Default vs Naive EB")

print("\nFitting 2 variants (default + Mack-informed). ~2-3 min each.\n")
print("Note: Default priors result was 184,629 (from prior run, commit bb07e94).")
print("Note: Naive empirical Bayes was ~151,749 (from prior run, commit bb07e94).")
print("We re-run Default for exact reproducibility, plus the new Mack-informed variant.")

# Default priors (priors=None — package builds adaptive data-driven priors)
res_default = fit_and_extract(
    tri_celina,
    priors_dict=None,
    label="Default priors (adaptive)",
    actual_ultimate=actual_celina,
    latest_observed=latest_celina,
    draws=2000, tune=1000, chains=2,
)

# Mack-informed priors, sigma=0.3
res_mack_03 = fit_and_extract(
    tri_celina,
    priors_dict=mack_priors_info["priors"],
    label="Mack-informed priors (sigma=0.3)",
    actual_ultimate=actual_celina,
    latest_observed=latest_celina,
    draws=2000, tune=1000, chains=2,
)

# Mack-informed priors, sigma=0.1 (tighter)
mack_priors_01 = build_mack_priors(tri_celina, sigma=0.1)
res_mack_01 = fit_and_extract(
    tri_celina,
    priors_dict=mack_priors_01["priors"],
    label="Mack-informed priors (sigma=0.1)",
    actual_ultimate=actual_celina,
    latest_observed=latest_celina,
    draws=2000, tune=1000, chains=2,
)

# Print results
print("\n")
section("TASK 2 — Results: Celina Mut Grp (ppauto/353)")

header = (
    f"{'Variant':<42s}  {'Ult (median)':>14s}  {'Ratio/Mack':>11s}"
    f"  {'Ratio/Actual':>13s}  {'Implied_pctl':>13s}"
)
sep = "-" * len(header)
print(header)
print(sep)

celina_results = [res_default, res_mack_03, res_mack_01]
for res in celina_results:
    ult = res["ult_median"]
    print(
        f"  {res['label']:<40s}  {ult:14,.1f}  "
        f"{ult/mack_total:11.4f}  {ult/actual_celina:13.4f}  "
        f"{res['implied_pctl']:13.4f}"
    )

# Historical reference rows
print(sep)
print(
    f"  {'Default priors (prior run bb07e94)':<40s}  {DEFAULT_GLM_CELINA:14,.1f}  "
    f"{DEFAULT_GLM_CELINA/mack_total:11.4f}  {DEFAULT_GLM_CELINA/actual_celina:13.4f}  {'~0.02':>13s}"
)
print(
    f"  {'Naive EB priors (prior run bb07e94)':<40s}  {NAIVE_EB_CELINA:14,.1f}  "
    f"{NAIVE_EB_CELINA/mack_total:11.4f}  {NAIVE_EB_CELINA/actual_celina:13.4f}  {'~0.09':>13s}"
)
print(
    f"  {'Bootstrap ODP':<40s}  {BOOTSTRAP_ODP_CELINA:14,.1f}  "
    f"{BOOTSTRAP_ODP_CELINA/mack_total:11.4f}  {BOOTSTRAP_ODP_CELINA/actual_celina:13.4f}  {'~0.29':>13s}"
)
print(
    f"  {'Mack (reference)':<40s}  {mack_total:14,.1f}  {'1.0000':>11s}  "
    f"{mack_total/actual_celina:13.4f}  {'N/A':>13s}"
)
print(
    f"  {'Actual (ground truth)':<40s}  {actual_celina:14,.1f}  "
    f"{actual_celina/mack_total:11.4f}  {'1.0000':>13s}  {'N/A':>13s}"
)

print("\nWithin 10% of Mack?")
for res in celina_results:
    pct = abs(res["ult_median"] - mack_total) / mack_total * 100
    status = "YES" if pct <= 10.0 else "NO"
    print(f"  {res['label']}: {pct:.2f}%  -> {status}")

best_celina = min(celina_results, key=lambda x: abs(x["ult_median"] - mack_total))
best_pct = abs(best_celina["ult_median"] - mack_total) / mack_total * 100
print(f"\nBest variant: '{best_celina['label']}' ({best_pct:.2f}% from Mack)")


# ---------------------------------------------------------------------------
# Task 2b — Per-origin diagnosis for best variant
# ---------------------------------------------------------------------------
section("TASK 2b — Per-origin IBNR diagnosis for Mack-informed priors (sigma=0.3)")

def per_origin_breakdown(res: dict, tri_ref: "cl.Triangle", label: str) -> None:
    glm = res.get("glm") if "glm" in res else None
    # reserves_posterior_ was not stored in fit_and_extract; we need the glm obj
    # We stored it via a quick re-run workaround — see glm_mack_03 below
    pass


# Refit to get the glm object for detailed per-origin analysis
print("(Per-origin breakdown uses the sigma=0.3 fit already completed above)")
print("Re-fitting to get GLM object for per-origin access ...")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    glm_mack_03 = BayesianChainLadderGLM(
        formula="incremental ~ 1 + C(origin) + C(dev)",
        family="gamma", link="log", exposure=None,
        priors=mack_priors_info["priors"],
        draws=2000, tune=1000, chains=2, target_accept=0.95, random_seed=42,
    ).fit(tri_celina)

reserves_post = glm_mack_03.reserves_posterior_
rp_origins = set(reserves_post.coords["origin"].values)

print(f"\n{'Origin':>8s}  {'Latest':>10s}  {'Mack Ult':>10s}  {'Mack IBNR':>10s}"
      f"  {'GLM IBNR p50':>13s}  {'GLM/Mack':>9s}")
print("  " + "-" * 68)

cum_df_celina = tri_celina.to_frame()
total_glm_ibnr = 0.0
total_mack_ibnr = 0.0
for i, o in enumerate(tri_celina.origin):
    yr_int = int(str(o)[:4])
    lat_o = float(latest_diag[i])
    m_ult_o = float(mack_ult_arr[i])
    m_ibnr_o = m_ult_o - lat_o
    total_mack_ibnr += m_ibnr_o
    if yr_int in rp_origins:
        orig_ibnr = np.asarray(
            reserves_post.sel(origin=yr_int).values, dtype=float
        )
        orig_ibnr = orig_ibnr[np.isfinite(orig_ibnr)]
        glm_p50 = float(np.median(orig_ibnr))
        total_glm_ibnr += glm_p50
        ratio = glm_p50 / m_ibnr_o if m_ibnr_o > 0 else float("nan")
        print(
            f"  {yr_int:>8d}  {lat_o:10.1f}  {m_ult_o:10.1f}  {m_ibnr_o:10.1f}"
            f"  {glm_p50:13.1f}  {ratio:9.4f}"
        )
    else:
        print(
            f"  {yr_int:>8d}  {lat_o:10.1f}  {m_ult_o:10.1f}  {m_ibnr_o:10.1f}"
            f"  {'(no future)':>13s}  {'N/A':>9s}"
        )

ratio_total = total_glm_ibnr / total_mack_ibnr if total_mack_ibnr > 0 else float("nan")
print(
    f"  {'TOTAL':>8s}  {latest_celina:10.1f}  {mack_total:10.1f}  {total_mack_ibnr:10.1f}"
    f"  {total_glm_ibnr:13.1f}  {ratio_total:9.4f}"
)


# ---------------------------------------------------------------------------
# TASK 3 — Smoke tests: Amerisafe (ppauto/6807) + NC Farm Bureau (othliab/3240)
# ---------------------------------------------------------------------------
section("TASK 3 — Smoke tests: Amerisafe + NC Farm Bureau")


def smoke_test(line: str, group_id: int, sigma: float = 0.3) -> dict:
    """Run default vs Mack-informed on one triangle. Return result dict."""
    all_recs = rt.build_triangle_records()
    matches = [r for r in all_recs if r.line == line and r.group_id == group_id]
    if not matches:
        print(f"  No record for line={line!r}, group_id={group_id}")
        return {}

    rec = matches[0]
    tri_ = rec.train_triangles["paid"]
    actual_ult_ = float(rec.actual_ultimates["paid"])

    mack_ = cl.MackChainladder().fit(tri_)
    mack_ult_arr_ = np.asarray(mack_.ultimate_.values, dtype=float).squeeze()
    mack_total_ = float(np.nansum(mack_ult_arr_))
    latest_ = float(np.nansum(
        np.asarray(mack_.latest_diagonal.values, dtype=float).squeeze()
    ))

    print(f"\n  Company: {rec.company}  ({line}/{group_id})")
    print(f"  Mack ultimate:   {mack_total_:>12,.1f}")
    print(f"  Actual ultimate: {actual_ult_:>12,.1f}")

    # Build Mack-informed priors
    try:
        mack_p_info = build_mack_priors(tri_, sigma=sigma)
        mack_p = mack_p_info["priors"]
    except Exception as e:
        print(f"  WARNING: could not build Mack priors ({e}), using defaults")
        mack_p = None

    # Fit default
    res_def_ = fit_and_extract(
        tri_, priors_dict=None,
        label="Default priors",
        actual_ultimate=actual_ult_, latest_observed=latest_,
        draws=2000, tune=1000, chains=2,
    )
    # Fit Mack-informed
    res_mack_ = fit_and_extract(
        tri_, priors_dict=mack_p,
        label=f"Mack-informed priors (sigma={sigma})",
        actual_ultimate=actual_ult_, latest_observed=latest_,
        draws=2000, tune=1000, chains=2,
    )

    print(f"\n  {'Variant':<42s}  {'Ult (median)':>14s}  {'Ratio/Mack':>11s}  {'Implied_pctl':>13s}")
    print("  " + "-" * 85)
    for res_, lbl in [(res_def_, "Default"), (res_mack_, f"Mack-informed (sigma={sigma})")]:
        ult_ = res_["ult_median"]
        print(
            f"  {res_['label']:<42s}  {ult_:14,.1f}  "
            f"{ult_/mack_total_:11.4f}  {res_['implied_pctl']:13.4f}"
        )
    print(f"  {'Mack (reference)':<42s}  {mack_total_:14,.1f}  {'1.0000':>11s}  {'N/A':>13s}")
    print(f"  {'Actual (ground truth)':<42s}  {actual_ult_:14,.1f}  {actual_ult_/mack_total_:11.4f}  {'N/A':>13s}")

    return {
        "line": line,
        "group_id": group_id,
        "company": rec.company,
        "mack_total": mack_total_,
        "actual_ult": actual_ult_,
        "default_ult_p50": res_def_["ult_median"],
        "default_pctl": res_def_["implied_pctl"],
        "mack_ult_p50": res_mack_["ult_median"],
        "mack_pctl": res_mack_["implied_pctl"],
        "mack_ratio": res_mack_["ult_median"] / mack_total_,
    }


print("\nRunning Amerisafe (ppauto/6807) ...")
res_amerisafe = smoke_test("ppauto", 6807)

print("\nRunning NC Farm Bureau (othliab/3240) ...")
res_ncfarm = smoke_test("othliab", 3240)


# ---------------------------------------------------------------------------
# TASK 4 — 10 random triangles (if 3-triangle median ratio < 10% from Mack)
# ---------------------------------------------------------------------------
section("TASK 4 — 10 random triangles smoke test")

# Check 3-triangle median ratio first
three_ratios = []
for r3 in [res_mack_03, res_amerisafe, res_ncfarm]:
    if isinstance(r3, dict) and "ult_median" in r3:
        three_ratios.append(r3["ult_median"] / MACK_CELINA)
    elif isinstance(r3, dict) and "mack_ratio" in r3:
        three_ratios.append(r3["mack_ratio"])

if len(three_ratios) > 0:
    median_3 = float(np.median(three_ratios))
    pct_from_mack_3 = abs(median_3 - 1.0) * 100
    print(f"\n3-triangle median ratio to Mack: {median_3:.4f} ({pct_from_mack_3:.1f}% from 1.0)")
else:
    pct_from_mack_3 = 999.0
    print("Could not compute 3-triangle ratios.")

# Always run the 10-triangle test (consistent with prior scripts)
print("\nRunning 10 random triangles regardless of 3-triangle threshold.")

# Fixed seed for reproducibility; pick 10 unique (line, group_id) combos
# excluding the 3 already tested
already_done = {("ppauto", 353), ("ppauto", 6807), ("othliab", 3240)}

# Load backtest cache to get all available combos
cache_df = pd.read_parquet(
    Path(__file__).resolve().parent / "cache" / "backtest_all.parquet"
)
combos_all = (
    cache_df[["line", "group_id", "company"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

# Filter out already-done
candidates = [
    (row.line, row.group_id, row.company)
    for _, row in combos_all.iterrows()
    if (row.line, row.group_id) not in already_done
]

# Stratify: 2-3 per line; random but reproducible
rng = random.Random(42)
by_line: dict[str, list] = {}
for ln, gid, comp in candidates:
    by_line.setdefault(ln, []).append((ln, gid, comp))

smoke_sample = []
lines_order = ["comauto", "ppauto", "wkcomp", "othliab"]
per_line = [3, 3, 2, 2]  # 10 total
for line_name, n in zip(lines_order, per_line):
    pool = by_line.get(line_name, [])
    rng.shuffle(pool)
    smoke_sample.extend(pool[:n])

print(f"\nSelected {len(smoke_sample)} triangles for 10-triangle sweep:")
for ln, gid, comp in smoke_sample:
    print(f"  {ln}/{gid:>6d}  {comp}")

smoke_10_results = []
for ln, gid, comp in smoke_sample:
    print(f"\n--- {comp} ({ln}/{gid}) ---")
    try:
        res_s = smoke_test(ln, gid, sigma=0.3)
        if res_s:
            smoke_10_results.append(res_s)
    except Exception as e:
        print(f"  ERROR: {e}")
        smoke_10_results.append({
            "line": ln, "group_id": gid, "company": comp,
            "mack_ratio": float("nan"), "mack_pctl": float("nan"),
        })


# ---------------------------------------------------------------------------
# Overall summary across 13 triangles
# ---------------------------------------------------------------------------
section("SUMMARY — All triangles (Celina + 2 smoke + 10 random = 13)")

all_13: list[dict] = []
# Celina
all_13.append({
    "line": "ppauto", "group_id": 353, "company": "Celina Mut Grp",
    "mack_ratio": res_mack_03["ult_median"] / MACK_CELINA,
    "mack_pctl": res_mack_03["implied_pctl"],
    "default_ratio": res_default["ult_median"] / MACK_CELINA,
})
# Amerisafe
if res_amerisafe:
    all_13.append(res_amerisafe)
# NC Farm Bureau
if res_ncfarm:
    all_13.append(res_ncfarm)
# 10 random
all_13.extend(smoke_10_results)

print(f"\n{'Company':<35s}  {'line':>8s}  {'Ratio/Mack':>11s}  {'Implied_pctl':>13s}")
print("-" * 72)
for r_ in all_13:
    ratio = r_.get("mack_ratio", float("nan"))
    pctl = r_.get("mack_pctl", float("nan"))
    print(
        f"  {r_.get('company','?'):<33s}  {r_.get('line','?'):>8s}"
        f"  {ratio:11.4f}  {pctl:13.4f}"
    )

ratios = [r_["mack_ratio"] for r_ in all_13 if np.isfinite(r_.get("mack_ratio", float("nan")))]
pctls = [r_["mack_pctl"] for r_ in all_13 if np.isfinite(r_.get("mack_pctl", float("nan")))]

print(f"\nAggregate statistics (Mack-informed priors, sigma=0.3):")
print(f"  N triangles: {len(ratios)}")
if ratios:
    print(f"  Median ratio to Mack:  {np.median(ratios):.4f}")
    print(f"  Mean ratio to Mack:    {np.mean(ratios):.4f}")
    print(f"  Pct within 5% of Mack: {np.mean(np.abs(np.array(ratios) - 1.0) <= 0.05)*100:.1f}%")
    print(f"  Pct within 10% of Mack: {np.mean(np.abs(np.array(ratios) - 1.0) <= 0.10)*100:.1f}%")
if pctls:
    print(f"  Median implied_pctl:   {np.median(pctls):.4f}")
    print(f"  Mean implied_pctl:     {np.mean(pctls):.4f}")


# ---------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------
section("VERDICT — Do Mack-informed priors close the gap?")

best_ratio = res_mack_03["ult_median"] / MACK_CELINA
best_pct = abs(best_ratio - 1.0) * 100

print(f"\nCelina Mut Grp (ppauto/353) — Progress summary:")
print(f"{'=' * 60}")
print(f"  Default priors (prior run):      {DEFAULT_GLM_CELINA:>12,.1f}  (+{(DEFAULT_GLM_CELINA/MACK_CELINA-1)*100:.1f}% over Mack)")
print(f"  Naive EB priors (prior run):     {NAIVE_EB_CELINA:>12,.1f}  (+{(NAIVE_EB_CELINA/MACK_CELINA-1)*100:.1f}% over Mack)")
print(f"  Default priors (this run):       {res_default['ult_median']:>12,.1f}  ({(res_default['ult_median']/MACK_CELINA-1)*100:+.1f}% vs Mack)")
print(f"  Mack-informed sigma=0.3 (this run): {res_mack_03['ult_median']:>12,.1f}  ({(res_mack_03['ult_median']/MACK_CELINA-1)*100:+.1f}% vs Mack)")
print(f"  Mack-informed sigma=0.1 (this run): {res_mack_01['ult_median']:>12,.1f}  ({(res_mack_01['ult_median']/MACK_CELINA-1)*100:+.1f}% vs Mack)")
print(f"  Mack reference:                  {MACK_CELINA:>12,.1f}")
print(f"  Actual:                          {ACTUAL_CELINA:>12,.1f}")

if ratios:
    median_ratio = float(np.median(ratios))
    pct_from_1 = abs(median_ratio - 1.0) * 100
    print(f"\n13-triangle median ratio: {median_ratio:.4f} ({pct_from_1:.1f}% from Mack)")

print(f"\nVerdict:")
if best_pct <= 5.0:
    print(f"  STRONG SUCCESS: Mack-informed priors bring Celina within {best_pct:.1f}% of Mack.")
    print(f"  The CL-anchored prior effectively eliminates the truncation-bias in sparse origins.")
    print(f"  RECOMMENDATION: Implement init_priors_from_chainladder=True as a package convenience.")
elif best_pct <= 10.0:
    print(f"  MODERATE SUCCESS: Best variant is {best_pct:.1f}% from Mack (within 10%).")
    print(f"  The CL-anchored prior substantially reduces the bias vs naive EB (+17%) and default (+42%).")
    print(f"  RECOMMENDATION: Implement init_priors_from_chainladder=True; sigma tuning may help further.")
else:
    print(f"  PARTIAL: Best variant is {best_pct:.1f}% from Mack.")
    print(f"  Improvement over naive EB but structural issues remain.")
    print(f"  Consider: random effects spec (1|origin), or tighter sigma.")

if ratios and float(np.median(ratios)) < 1.10 and float(np.median(ratios)) > 0.90:
    print(f"\n  13-triangle median ratio {float(np.median(ratios)):.3f} is within 10% of Mack.")
    print(f"  Mack-informed priors are a viable general approach.")

section("DONE")
