"""10_data_informed_priors.py — Test data-informed empirical priors on C(origin) and C(dev).

Investigates whether centering C(origin) and C(dev) priors at empirical means
(from the training triangle data) brings BayesianChainLadderGLM ultimate in line
with Mack on the Celina Mut Grp (ppauto/353) case.

Background:
  - Default Normal(0, σ=1) priors on C(origin) shrink sparse origins toward 0
    in log-space, causing exp(0)=1 to be the predicted multiplier when it should
    be smaller for late dev periods.
  - Data-informed priors center each contrast at the empirical mean log-ratio,
    providing better initial guidance especially for data-sparse origins/devs.

Three variants tested:
  1. Default priors (σ=1): baseline from commit d8ac08e
  2. Data-informed priors (σ=0.3)
  3. Tighter data-informed priors (σ=0.1)

Reference benchmarks:
  - Mack:         129,779
  - BootstrapODP: 129,701
  - Actual:       125,467
  - Default GLM:  184,629 (47% over actual, 42% over Mack) — from prior analysis

Run with:
    cd references/meyers-backtest
    uv run python 10_data_informed_priors.py
"""
from __future__ import annotations

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
# Reference benchmarks (from prior analysis, commit d8ac08e)
# ---------------------------------------------------------------------------

MACK_ULTIMATE = 129_779.0
BOOTSTRAP_ODP_ULTIMATE = 129_701.0
ACTUAL_ULTIMATE = 125_467.0
DEFAULT_GLM_ULTIMATE = 184_629.0  # Normal(0, sigma=1) priors

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import bambi as bmb
import chainladder as cl
import reservetestr as rt

from bayesianchainladder import BayesianChainLadderGLM

# ---------------------------------------------------------------------------
# TASK 1 — Verify Bambi accepts array-valued priors for categorical contrasts
# ---------------------------------------------------------------------------
section("TASK 1 — Verify array-valued priors in Bambi")

print("Testing bmb.Prior('Normal', mu=np.array([0.5, 1.0]), sigma=np.array([0.1, 0.2])) ...")

df_test = pd.DataFrame({
    "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    "g": pd.Categorical([1, 1, 1, 2, 2, 2, 3, 3, 3, 3]),
})

try:
    model_test = bmb.Model(
        "y ~ 1 + C(g)",
        data=df_test,
        family="gaussian",
        priors={
            "C(g)": bmb.Prior("Normal", mu=np.array([0.5, 1.0]), sigma=np.array([0.1, 0.2])),
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idata_test = model_test.fit(draws=200, tune=200, chains=1, random_seed=42)
    posterior_means = idata_test.posterior["C(g)"].mean(dim=["chain", "draw"]).values
    print(f"  Result: OK")
    print(f"  C(g) posterior means: {posterior_means}")
    print(f"  (Expected close to [0.5, 1.0] since tight sigma=[0.1, 0.2])")
    ARRAY_PRIORS_WORK = True
except Exception as e:
    print(f"  FAILED: {e}")
    ARRAY_PRIORS_WORK = False

print(f"\nVerdict: array-valued priors {'WORK' if ARRAY_PRIORS_WORK else 'DO NOT WORK'} in Bambi {bmb.__version__}")

if not ARRAY_PRIORS_WORK:
    print("Cannot proceed with data-informed priors — Bambi API incompatible.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# TASK 2 — Load Celina triangle and compute empirical priors
# ---------------------------------------------------------------------------
section("TASK 2 — Load Celina triangle and compute data-informed priors")

recs = rt.build_triangle_records()
r = [rec for rec in recs if rec.line == "ppauto" and rec.group_id == 353][0]
print(f"Company: {r.company}  |  line: {r.line}  |  group_id: {r.group_id}")

tri = r.train_triangles["paid"]
actual_ultimate = float(r.actual_ultimates["paid"])
print(f"Actual ultimate (from records):  {actual_ultimate:>12,.1f}")

# Mack reference
mack = cl.MackChainladder().fit(tri)
mack_ult = np.asarray(mack.ultimate_.values, dtype=float).squeeze()
mack_total = float(np.nansum(mack_ult))
latest_diag = np.asarray(mack.latest_diagonal.values, dtype=float).squeeze()
total_latest = float(np.nansum(latest_diag))
mack_ibnr = mack_total - total_latest
print(f"Mack ultimate (this run):        {mack_total:>12,.1f}")
print(f"Total latest observed:           {total_latest:>12,.1f}")

# Incremental triangle as DataFrame
tri_inc = tri.cum_to_incr()
inc_df = tri_inc.to_frame()
origins_idx = list(inc_df.index)  # pandas index (timestamps)
devs = list(inc_df.columns)       # [12, 24, 36, ..., 120]
n_origins = len(origins_idx)
n_devs = len(devs)

print(f"\nTriangle: {n_origins} origins x {n_devs} dev periods")
print("Incremental triangle:")
print(inc_df.to_string())


def _compute_celina_priors(sigma: float) -> dict:
    """Compute empirical Bayes priors for C(origin) and C(dev) contrasts.

    Parameters
    ----------
    sigma : float
        Common SD for all prior Normal distributions.

    Returns
    -------
    dict
        Dict with keys 'C(origin)', 'C(dev)', 'Intercept' as bambi.Prior objects,
        plus 'origin_means', 'dev_means', 'intercept_mu' for reporting.
    """
    # --- C(origin) priors ---
    # Reference = first origin (1988). Contrasts: T.1989, T.1990, ..., T.1997
    # Prior mean = log(mean_incr[origin]) - log(mean_incr[ref_origin])
    # Only use observed (non-NaN, positive) incrementals for each origin.

    ref_vals = inc_df.iloc[0].dropna().values
    ref_vals = ref_vals[ref_vals > 0]
    ref_log_mean = float(np.log(ref_vals.mean())) if len(ref_vals) > 0 else 0.0

    origin_means = []
    for i in range(1, n_origins):
        vals = inc_df.iloc[i].dropna().values
        vals = vals[vals > 0]
        if len(vals) > 0:
            log_mean = float(np.log(vals.mean()))
            effect = log_mean - ref_log_mean
        else:
            effect = 0.0
        origin_means.append(effect)

    origin_means_arr = np.array(origin_means, dtype=float)
    origin_sds_arr = np.full(len(origin_means), sigma, dtype=float)

    # --- C(dev) priors ---
    # Reference = first dev period (12). Contrasts: T.24, T.36, ..., T.120
    # Prior mean = average over origins of log(incr[o, dev_k] / incr[o, dev_1])
    # Only include origins with both observations positive.

    ref_dev = devs[0]  # 12

    dev_means = []
    for d in devs[1:]:
        log_ratios = []
        for i in range(n_origins):
            v_ref = inc_df.iloc[i][ref_dev]
            v_k = inc_df.iloc[i][d]
            if pd.notna(v_ref) and pd.notna(v_k) and v_ref > 0 and v_k > 0:
                log_ratios.append(float(np.log(v_k / v_ref)))
        if len(log_ratios) > 0:
            mu = float(np.mean(log_ratios))
        else:
            mu = 0.0
        dev_means.append(mu)

    dev_means_arr = np.array(dev_means, dtype=float)
    dev_sds_arr = np.full(len(dev_means), sigma, dtype=float)

    # --- Intercept prior ---
    # Center at log of reference cell (origin=1988, dev=12), i.e., the observed
    # value for the reference level of both C(origin) and C(dev).
    # Apply lognormal correction: mu = log(y_ref) - sigma^2/2 so E[exp(Int)] = y_ref.
    ref_cell_val = float(inc_df.iloc[0][ref_dev])
    intercept_sigma = 1.0  # keep intercept prior sigma generous
    intercept_mu = float(np.log(ref_cell_val)) - intercept_sigma**2 / 2

    priors = {
        "Intercept": bmb.Prior("Normal", mu=intercept_mu, sigma=intercept_sigma),
        "C(origin)": bmb.Prior("Normal", mu=origin_means_arr, sigma=origin_sds_arr),
        "C(dev)": bmb.Prior("Normal", mu=dev_means_arr, sigma=dev_sds_arr),
    }

    return {
        "priors": priors,
        "origin_means": origin_means_arr,
        "dev_means": dev_means_arr,
        "intercept_mu": intercept_mu,
        "intercept_sigma": intercept_sigma,
    }


# Print the constructed priors (sigma=0.3 version)
result_03 = _compute_celina_priors(sigma=0.3)
print(f"\n{'--- Constructed Priors (sigma=0.3) ---':}")
print(f"\nIntercept prior: Normal(mu={result_03['intercept_mu']:.4f}, sigma={result_03['intercept_sigma']:.1f})")
print(f"  (reference cell 1988,dev=12 = {inc_df.iloc[0][devs[0]]:.0f}, log={np.log(inc_df.iloc[0][devs[0]]):.4f})")
print()
print("C(origin) priors:")
origin_years = [str(o)[:4] for o in origins_idx[1:]]
for yr, mu in zip(origin_years, result_03["origin_means"]):
    print(f"  T.{yr}: Normal(mu={mu:+.4f}, sigma=0.30)")
print()
print("C(dev) priors:")
for d, mu in zip(devs[1:], result_03["dev_means"]):
    print(f"  T.{d}: Normal(mu={mu:+.4f}, sigma=0.30)")


# ---------------------------------------------------------------------------
# Helper: fit GLM and extract ultimates
# ---------------------------------------------------------------------------

def _fit_and_extract(
    triangle,
    priors_dict: dict | None,
    label: str,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 2,
) -> dict:
    """Fit BayesianChainLadderGLM and return summary dict."""
    print(f"\n  Fitting: {label} ...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        glm = BayesianChainLadderGLM(
            formula="incremental ~ 1 + C(origin) + C(dev)",
            family="gamma",
            link="log",
            exposure=None,
            priors=priors_dict,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.95,
            random_seed=42,
        ).fit(triangle)

    # Extract posterior of total ultimate
    reserves_post = glm.reserves_posterior_
    total_ibnr_samples = np.asarray(reserves_post.sum(dim="origin").values, dtype=float)
    total_ult_samples = total_ibnr_samples + total_latest
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
        "glm": glm,
        "finite_ult": finite_ult,
    }


# ---------------------------------------------------------------------------
# TASK 3 — Fit Celina with data-informed priors and compare
# ---------------------------------------------------------------------------
section("TASK 3 — Fit Celina (three variants)")

print("\nFitting 3 variants. Each fit: 2 chains x 2000 draws + 1000 tune (~2-3 min each).")

# Variant 1: Default priors (Normal(0, sigma=1) on C(origin) and C(dev))
# We let priors=None so the package builds its default adaptive priors
results_default = _fit_and_extract(tri, priors_dict=None, label="Default priors (σ=1)")

# Variant 2: Data-informed priors, sigma=0.3
result_03 = _compute_celina_priors(sigma=0.3)
results_03 = _fit_and_extract(tri, priors_dict=result_03["priors"], label="Data-informed priors (σ=0.3)")

# Variant 3: Tighter data-informed priors, sigma=0.1
result_01 = _compute_celina_priors(sigma=0.1)
results_01 = _fit_and_extract(tri, priors_dict=result_01["priors"], label="Tighter data-informed priors (σ=0.1)")

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
section("TASK 3 — Results: Celina Mut Grp (ppauto/353)")

header = f"{'Variant':<40s}  {'Ult (median)':>14s}  {'Ratio/Mack':>11s}  {'Ratio/Actual':>13s}  {'Implied_pctl':>13s}"
sep = "-" * len(header)
print(header)
print(sep)

all_results = [results_default, results_03, results_01]
for res in all_results:
    ult = res["ult_median"]
    print(
        f"  {res['label']:<38s}  {ult:14,.1f}  "
        f"{ult/mack_total:11.4f}  {ult/actual_ultimate:13.4f}  "
        f"{res['implied_pctl']:13.4f}"
    )

# Reference rows
print(sep)
print(f"  {'Mack (reference)':<38s}  {mack_total:14,.1f}  {'1.0000':>11s}  {mack_total/actual_ultimate:13.4f}  {'N/A':>13s}")
print(f"  {'BootstrapODP (reference)':<38s}  {BOOTSTRAP_ODP_ULTIMATE:14,.1f}  "
      f"{BOOTSTRAP_ODP_ULTIMATE/mack_total:11.4f}  {BOOTSTRAP_ODP_ULTIMATE/actual_ultimate:13.4f}  {'~0.29':>13s}")
print(f"  {'Actual (ground truth)':<38s}  {actual_ultimate:14,.1f}  "
      f"{actual_ultimate/mack_total:11.4f}  {'1.0000':>13s}  {'N/A':>13s}")

# Which variants are within 10% of Mack?
print("\nWithin 10% of Mack?")
for res in all_results:
    pct_diff = abs(res["ult_median"] - mack_total) / mack_total * 100
    status = "YES" if pct_diff <= 10.0 else "NO"
    print(f"  {res['label']}: {pct_diff:.2f}%  -> {status}")

# Detailed stats
print("\nDetailed statistics:")
print(f"{'Variant':<40s}  {'Median':>10s}  {'Mean':>10s}  {'SD':>10s}  {'p5':>10s}  {'p95':>10s}")
print("-" * 95)
for res in all_results:
    print(f"  {res['label']:<38s}  "
          f"{res['ult_median']:10,.1f}  {res['ult_mean']:10,.1f}  "
          f"{res['ult_sd']:10,.1f}  {res['ult_p5']:10,.1f}  {res['ult_p95']:10,.1f}")

# Identify best variant
best_result = min(all_results, key=lambda x: abs(x["ult_median"] - mack_total))
best_pct = abs(best_result["ult_median"] - mack_total) / mack_total * 100
print(f"\nBest variant: '{best_result['label']}' ({best_pct:.2f}% from Mack)")

# ---------------------------------------------------------------------------
# Diagnosis: if tight priors still don't fix it, look at where the bias is
# ---------------------------------------------------------------------------
section("DIAGNOSIS — Where is the bias for each variant?")

def _per_origin_breakdown(res, label, tri_ref, mack_ult_arr, latest_diag_arr):
    """Per-origin IBNR breakdown.

    reserves_posterior_ uses integer year coords; tri.origin uses Period objects.
    We match by integer year.
    """
    glm = res["glm"]
    reserves_post = glm.reserves_posterior_
    rp_origins = set(reserves_post.coords["origin"].values)

    print(f"\n{label}:")
    print(f"  {'Origin':>8s}  {'Latest':>10s}  {'Mack Ult':>10s}  {'Mack IBNR':>10s}  "
          f"{'GLM IBNR p50':>13s}  {'GLM/Mack':>9s}")
    print("  " + "-" * 68)

    total_glm_ibnr = 0.0
    total_mack_ibnr_sum = 0.0
    for i, o in enumerate(tri_ref.origin):
        yr_int = int(str(o)[:4])
        latest_o = float(latest_diag_arr[i])
        mack_ult_o = float(mack_ult_arr[i])
        mack_ibnr_o = mack_ult_o - latest_o
        total_mack_ibnr_sum += mack_ibnr_o
        if yr_int in rp_origins:
            orig_ibnr = np.asarray(reserves_post.sel(origin=yr_int).values, dtype=float)
            orig_ibnr = orig_ibnr[np.isfinite(orig_ibnr)]
            glm_ibnr_p50 = float(np.median(orig_ibnr))
            total_glm_ibnr += glm_ibnr_p50
            ratio = glm_ibnr_p50 / mack_ibnr_o if mack_ibnr_o > 0 else float("nan")
            print(f"  {str(yr_int):>8s}  {latest_o:10.1f}  {mack_ult_o:10.1f}  {mack_ibnr_o:10.1f}  "
                  f"{glm_ibnr_p50:13.1f}  {ratio:9.4f}")
        else:
            # Origin fully developed — no future cells
            print(f"  {str(yr_int):>8s}  {latest_o:10.1f}  {mack_ult_o:10.1f}  {mack_ibnr_o:10.1f}  "
                  f"{'(no future)':>13s}  {'N/A':>9s}")
    ratio_total = total_glm_ibnr / total_mack_ibnr_sum if total_mack_ibnr_sum > 0 else float("nan")
    print(f"  {'TOTAL':>8s}  {np.nansum(latest_diag_arr):10.1f}  {np.nansum(mack_ult_arr):10.1f}  "
          f"{total_mack_ibnr_sum:10.1f}  {total_glm_ibnr:13.1f}  {ratio_total:9.4f}")


for res in all_results:
    _per_origin_breakdown(res, res["label"], tri, mack_ult, latest_diag)

# ---------------------------------------------------------------------------
# TASK 4 — Smoke tests on Amerisafe and NC Farm Bureau (if best within 10% of Mack)
# ---------------------------------------------------------------------------
section("TASK 4 — Smoke tests: Amerisafe (ppauto/6807) and NC Farm Bureau (othliab/3240)")

# Choose the best priors for the smoke test
# Use sigma=0.3 data-informed priors
best_sigma_for_smoke = 0.3


def _compute_priors_for_company(tri_: "cl.Triangle", sigma: float) -> dict:
    """Compute empirical Bayes priors for a generic triangle."""
    tri_inc_ = tri_.cum_to_incr()
    inc_df_ = tri_inc_.to_frame()
    origins_idx_ = list(inc_df_.index)
    devs_ = list(inc_df_.columns)
    n_orig_ = len(origins_idx_)

    # Reference origin: first origin
    ref_vals = inc_df_.iloc[0].dropna().values
    ref_vals = ref_vals[ref_vals > 0]
    ref_log_mean = float(np.log(ref_vals.mean())) if len(ref_vals) > 0 else 0.0

    origin_means = []
    for i in range(1, n_orig_):
        vals = inc_df_.iloc[i].dropna().values
        vals = vals[vals > 0]
        if len(vals) > 0:
            effect = float(np.log(vals.mean())) - ref_log_mean
        else:
            effect = 0.0
        origin_means.append(effect)

    # Dev priors
    ref_dev = devs_[0]
    dev_means = []
    for d in devs_[1:]:
        log_ratios = []
        for i in range(n_orig_):
            v_ref = inc_df_.iloc[i][ref_dev]
            v_k = inc_df_.iloc[i][d]
            if pd.notna(v_ref) and pd.notna(v_k) and v_ref > 0 and v_k > 0:
                log_ratios.append(float(np.log(v_k / v_ref)))
        dev_means.append(float(np.mean(log_ratios)) if log_ratios else 0.0)

    # Intercept
    ref_cell_val = float(inc_df_.iloc[0][ref_dev])
    intercept_sigma = 1.0
    intercept_mu = float(np.log(ref_cell_val)) - intercept_sigma**2 / 2

    origin_arr = np.array(origin_means, dtype=float)
    dev_arr = np.array(dev_means, dtype=float)

    return {
        "Intercept": bmb.Prior("Normal", mu=intercept_mu, sigma=intercept_sigma),
        "C(origin)": bmb.Prior("Normal", mu=origin_arr, sigma=np.full(len(origin_arr), sigma, dtype=float)),
        "C(dev)": bmb.Prior("Normal", mu=dev_arr, sigma=np.full(len(dev_arr), sigma, dtype=float)),
    }


def run_smoke_test(line: str, group_id: int, company_label: str) -> None:
    """Run smoke test on a single triangle with data-informed priors."""
    print(f"\n--- {company_label} (line={line!r}, group_id={group_id}) ---")

    recs_all = rt.build_triangle_records()
    matches = [rec for rec in recs_all if rec.line == line and rec.group_id == group_id]
    if not matches:
        print(f"  No record found for line={line!r}, group_id={group_id}")
        return

    rec = matches[0]
    tri_ = rec.train_triangles["paid"]
    actual_ult_ = float(rec.actual_ultimates["paid"])

    # Mack
    mack_ = cl.MackChainladder().fit(tri_)
    mack_ult_arr = np.asarray(mack_.ultimate_.values, dtype=float).squeeze()
    mack_total_ = float(np.nansum(mack_ult_arr))
    latest_ = float(np.nansum(np.asarray(mack_.latest_diagonal.values, dtype=float).squeeze()))

    print(f"  Company: {rec.company}")
    print(f"  Actual ultimate: {actual_ult_:>12,.1f}")
    print(f"  Mack ultimate:   {mack_total_:>12,.1f}")
    print(f"  Latest obs:      {latest_:>12,.1f}")

    # Check whether gamma family is viable (need all positive incrementals)
    tri_inc_ = tri_.cum_to_incr()
    inc_vals = tri_inc_.to_frame().values
    obs_vals = inc_vals[~np.isnan(inc_vals)]
    if (obs_vals <= 0).any():
        print(f"  WARNING: triangle has non-positive incrementals; using gaussian instead of gamma")
        family_ = "gaussian"
    else:
        family_ = "gamma"

    # Compute data-informed priors
    try:
        priors_ = _compute_priors_for_company(tri_, sigma=best_sigma_for_smoke)
    except Exception as e:
        print(f"  WARNING: could not compute priors ({e}), using defaults")
        priors_ = None

    # Fit default
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        glm_default_ = BayesianChainLadderGLM(
            formula="incremental ~ 1 + C(origin) + C(dev)",
            family=family_, link="log", exposure=None, priors=None,
            draws=2000, tune=1000, chains=2, target_accept=0.95, random_seed=42,
        ).fit(tri_)

    # Fit data-informed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        glm_informed_ = BayesianChainLadderGLM(
            formula="incremental ~ 1 + C(origin) + C(dev)",
            family=family_, link="log", exposure=None, priors=priors_,
            draws=2000, tune=1000, chains=2, target_accept=0.95, random_seed=42,
        ).fit(tri_)

    def _ult(glm_model):
        res_post = glm_model.reserves_posterior_
        ibnr_s = np.asarray(res_post.sum(dim="origin").values, dtype=float)
        ult_s = ibnr_s + latest_
        finite_ = ult_s[np.isfinite(ult_s)]
        return {
            "median": float(np.median(finite_)),
            "mean": float(np.mean(finite_)),
            "pctl": float(np.mean(finite_ <= actual_ult_)),
        }

    stats_def = _ult(glm_default_)
    stats_inf = _ult(glm_informed_)

    print(f"\n  {'Variant':<35s}  {'Ult (median)':>14s}  {'Ratio/Mack':>11s}  {'Ratio/Actual':>13s}  {'Implied_pctl':>13s}")
    print("  " + "-" * 92)
    for label_, s_ in [("Default priors (σ=1)", stats_def), ("Data-informed priors (σ=0.3)", stats_inf)]:
        ult_ = s_["median"]
        print(f"  {label_:<35s}  {ult_:14,.1f}  {ult_/mack_total_:11.4f}  {ult_/actual_ult_:13.4f}  {s_['pctl']:13.4f}")
    print(f"  {'Mack (reference)':<35s}  {mack_total_:14,.1f}  {'1.0000':>11s}  {mack_total_/actual_ult_:13.4f}  {'N/A':>13s}")
    print(f"  {'Actual (ground truth)':<35s}  {actual_ult_:14,.1f}  {actual_ult_/mack_total_:11.4f}  {'1.0000':>13s}  {'N/A':>13s}")


# Only run smoke tests if best variant within 10% of Mack on Celina
best_pct_vs_mack = abs(best_result["ult_median"] - mack_total) / mack_total * 100

if best_pct_vs_mack <= 10.0:
    print(f"\nBest Celina variant is within 10% of Mack ({best_pct_vs_mack:.2f}%). Running smoke tests.")
    run_smoke_test("ppauto", 6807, "Amerisafe")
    run_smoke_test("othliab", 3240, "NC Farm Bureau")
else:
    print(f"\nBest Celina variant is {best_pct_vs_mack:.2f}% from Mack (>10%). Skipping smoke tests.")
    print("Running smoke tests anyway for diagnostic purposes.")
    run_smoke_test("ppauto", 6807, "Amerisafe")
    run_smoke_test("othliab", 3240, "NC Farm Bureau")

# ---------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------
section("VERDICT — Are data-informed priors the answer?")

print(f"\nCelina Mut Grp (ppauto/353) Results Summary:")
print(f"{'=' * 60}")
for res in all_results:
    ult = res["ult_median"]
    pct_mack = (ult - mack_total) / mack_total * 100
    pct_actual = (ult - actual_ultimate) / actual_ultimate * 100
    print(f"\n  {res['label']}:")
    print(f"    Ult (median):   {ult:>12,.1f}")
    print(f"    vs Mack:        {pct_mack:>+10.2f}%")
    print(f"    vs Actual:      {pct_actual:>+10.2f}%")
    print(f"    Implied pctl:   {res['implied_pctl']:>10.4f}")

print(f"\n  Mack reference:     {mack_total:>12,.1f}")
print(f"  Bootstrap ODP:      {BOOTSTRAP_ODP_ULTIMATE:>12,.1f}")
print(f"  Actual:             {actual_ultimate:>12,.1f}")

print(f"\nConclusion:")
if best_pct_vs_mack <= 5.0:
    print(f"  Data-informed priors SUCCEED: best variant '{best_result['label']}' "
          f"is {best_pct_vs_mack:.2f}% from Mack.")
    print(f"  Centering priors at empirical means effectively removes the shrinkage bias.")
elif best_pct_vs_mack <= 10.0:
    print(f"  Data-informed priors PARTIALLY succeed: best variant '{best_result['label']}' "
          f"is {best_pct_vs_mack:.2f}% from Mack (within 10%).")
    print(f"  Further improvements may require: per-cell empirical Bayes, random effects, "
          f"or a richer dev pattern prior.")
else:
    print(f"  Data-informed priors DO NOT fully fix Celina: best variant '{best_result['label']}' "
          f"is {best_pct_vs_mack:.2f}% from Mack.")
    print(f"  The bias is structural — empirical origin means are themselves biased because")
    print(f"  later origins have fewer observed dev periods (early dev is large), inflating")
    print(f"  the prior means for sparse origins. Tighter sigma forces stronger shrinkage to")
    print(f"  these inflated means rather than correcting the center.")
    print(f"\n  Recommended next step: use smoothed dev-adjusted priors, or switch to a")
    print(f"  random-effects spec (1|origin) with partial pooling across origins.")

section("DONE")
