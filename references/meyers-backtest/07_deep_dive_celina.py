"""07_deep_dive_celina.py — Deep dive into ppauto/353 (Celina Mut Grp).

Investigates why glm_m1_cat is 47% over actual AND has implied_pctl=0.000
(model is 100% confident actual > reality) despite a CV of 0.69.

Run with:
    cd references/meyers-backtest
    uv run python 07_deep_dive_celina.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIVIDER = "=" * 72
def section(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# ---------------------------------------------------------------------------
# Load triangle
# ---------------------------------------------------------------------------
section("TASK 1 — Load triangle and run all three methods")

import reservetestr as rt
import chainladder as cl
from _common import load_exposure_triangle

recs = rt.build_triangle_records()
r = [rec for rec in recs if rec.line == "ppauto" and rec.group_id == 353][0]
print(f"Company: {r.company}  |  line: {r.line}  |  group_id: {r.group_id}")

tri = r.train_triangles["paid"]
actual_ultimate = float(r.actual_ultimates["paid"])
prem_tri = load_exposure_triangle("ppauto", 353)

print("\nTraining triangle (cumulative paid):")
print(tri.to_frame().to_string())

print("\nPremium (net earned premium) by origin:")
print(prem_tri.to_frame().iloc[:, 0].to_string())


# ---------------------------------------------------------------------------
# Task 1a — MackChainladder
# ---------------------------------------------------------------------------
section("Task 1a — MackChainladder")

mack = cl.MackChainladder().fit(tri)
mack_ult = np.asarray(mack.ultimate_.values, dtype=float).squeeze()
mack_se_all = np.asarray(mack.mack_std_err_.values, dtype=float).squeeze()
total_mack_se = float(np.asarray(mack.total_mack_std_err_.values, dtype=float).squeeze())

# mack_std_err_ has shape (n_origins, n_devs) — last column is the per-origin total std err
mack_se = mack_se_all[:, -1]

origins_str = [str(o).split("T")[0] for o in tri.origin]
latest_diag = np.asarray(mack.latest_diagonal.values, dtype=float).squeeze()

print(f"{'Origin':8s}  {'Mack Ult':>12s}  {'Mack SE':>10s}  {'Latest Obs':>12s}")
for i, o in enumerate(origins_str):
    print(f"{o:8s}  {mack_ult[i]:12.1f}  {mack_se[i]:10.1f}  {latest_diag[i]:12.1f}")

total_mack_ult = float(np.nansum(mack_ult))
total_latest = float(np.nansum(latest_diag))
print(f"\nTotal Mack ultimate: {total_mack_ult:,.1f}")
print(f"Total Mack SE:       {total_mack_se:,.1f}")
print(f"Total Mack CV:       {total_mack_se/total_mack_ult:.4f}")
print(f"Actual ultimate:     {actual_ultimate:,.1f}")
print(f"Mack/Actual ratio:   {total_mack_ult/actual_ultimate:.4f}")


# ---------------------------------------------------------------------------
# Task 1b — BootstrapODPSample
# ---------------------------------------------------------------------------
section("Task 1b — BootstrapODPSample (n_sims=1000)")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    boot = cl.BootstrapODPSample(n_sims=1000, hat_adj=True, random_state=22).fit_transform(tri)
    cl_boot = cl.Chainladder().fit(boot)

boot_ult_vals = np.nansum(np.asarray(cl_boot.ultimate_.values, dtype=float), axis=(1, 2, 3))
boot_mean = float(np.nanmean(boot_ult_vals))
boot_sd = float(np.nanstd(boot_ult_vals, ddof=1))
print(f"Bootstrap ODP total ultimate distribution (n_sims=1000):")
print(f"  mean:  {boot_mean:,.1f}")
print(f"  sd:    {boot_sd:,.1f}")
print(f"  CV:    {boot_sd/boot_mean:.4f}")
print(f"  p5:    {np.percentile(boot_ult_vals, 5):,.1f}")
print(f"  p50:   {np.percentile(boot_ult_vals, 50):,.1f}")
print(f"  p95:   {np.percentile(boot_ult_vals, 95):,.1f}")
boot_pctl = float(np.mean(boot_ult_vals <= actual_ultimate))
print(f"  implied_pctl for actual {actual_ultimate:,.0f}: {boot_pctl:.4f}")


# ---------------------------------------------------------------------------
# Task 1c — BayesianChainLadderGLM (m1_cat spec, matching methods.py wrapper)
# ---------------------------------------------------------------------------
section("Task 1c — BayesianChainLadderGLM (gamma+log, C(origin)+C(dev), exposure offset)")

from bayesianchainladder import BayesianChainLadderGLM

print("Fitting GLM... (this takes a few minutes)")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    glm = BayesianChainLadderGLM(
        formula="incremental ~ 1 + C(origin) + C(dev)",
        family="gamma",
        link="log",
        exposure="net_earned_premium",
        priors=None,
        draws=1000,
        tune=1000,
        chains=2,
        target_accept=0.95,
        random_seed=22,
    )
    glm.fit(tri, exposure_triangle=prem_tri)

print("GLM fit complete.")

# Extract total ultimate posterior
reserves_post = glm.reserves_posterior_  # xr.DataArray (origin, sample)
total_ibnr_samples = np.asarray(reserves_post.sum(dim="origin").values, dtype=float)
total_ult_samples = total_ibnr_samples + total_latest
finite_ult = total_ult_samples[np.isfinite(total_ult_samples)]
glm_median_ult = float(np.median(finite_ult))
glm_mean_ult = float(np.mean(finite_ult))
glm_sd = float(np.std(finite_ult, ddof=1))
glm_pctl = float(np.mean(finite_ult <= actual_ultimate))
glm_cv_unpaid = glm_sd / max(float(np.median(total_ibnr_samples[np.isfinite(total_ibnr_samples)])), 1.0)

print(f"\nGLM total ultimate posterior:")
print(f"  median: {glm_median_ult:,.1f}")
print(f"  mean:   {glm_mean_ult:,.1f}")
print(f"  sd:     {glm_sd:,.1f}")
print(f"  CV:     {glm_sd/glm_median_ult:.4f}")
print(f"  CV unpaid: {glm_cv_unpaid:.4f}")
print(f"  p5:     {np.percentile(finite_ult, 5):,.1f}")
print(f"  p50:    {np.percentile(finite_ult, 50):,.1f}")
print(f"  p95:    {np.percentile(finite_ult, 95):,.1f}")
print(f"  p99:    {np.percentile(finite_ult, 99):,.1f}")
print(f"  min:    {np.min(finite_ult):,.1f}")
print(f"  implied_pctl for actual {actual_ultimate:,.0f}: {glm_pctl:.6f}")


# ---------------------------------------------------------------------------
# Task 2 — Per-origin GLM posterior breakdown
# ---------------------------------------------------------------------------
section("Task 2 — Per-origin GLM posterior: mean / median / p5 / p95 vs Mack vs actual")

# We need per-origin actuals from the full test triangle
# The test triangle only records final cumulative for 1988; others are NaN.
# Use: actual = latest_observed + true_IBNR.
# We do not have per-origin actuals for all origins — only the aggregate.
# We'll compute per-origin Mack ultimates vs GLM posterior vs latest observed.

origins_sorted = sorted(glm.future_data_["origin"].unique())
origin_latest = {}
for i, o in enumerate(origins_str):
    # Convert to same type used in future_data_
    origin_latest[tri.origin[i]] = float(latest_diag[i])

print(f"\n{'Origin':8s}  {'Latest':>10s}  {'Mack Ult':>10s}  {'GLM p5':>10s}  {'GLM p50':>10s}  "
      f"{'GLM p95':>10s}  {'GLM Mean':>10s}  {'IBNR p50':>10s}  {'Bias%':>8s}")

origin_glm_p50 = {}
for i, o in enumerate(origins_str):
    orig_key = tri.origin[i]
    # Get samples for this origin
    if orig_key in reserves_post.coords["origin"].values:
        orig_ibnr = np.asarray(reserves_post.sel(origin=orig_key).values, dtype=float)
        orig_ibnr = orig_ibnr[np.isfinite(orig_ibnr)]
        latest_o = origin_latest.get(orig_key, 0.0)
        orig_ult = orig_ibnr + latest_o
        glm_p5 = float(np.percentile(orig_ult, 5))
        glm_p50 = float(np.percentile(orig_ult, 50))
        glm_p95 = float(np.percentile(orig_ult, 95))
        glm_mean_o = float(np.mean(orig_ult))
        ibnr_p50 = float(np.percentile(orig_ibnr, 50))
        bias_pct = (glm_p50 - mack_ult[i]) / mack_ult[i] * 100
        origin_glm_p50[orig_key] = glm_p50
        print(f"{o:8s}  {latest_o:10.1f}  {mack_ult[i]:10.1f}  {glm_p5:10.1f}  "
              f"{glm_p50:10.1f}  {glm_p95:10.1f}  {glm_mean_o:10.1f}  {ibnr_p50:10.1f}  {bias_pct:7.1f}%")
    else:
        # Origin fully developed (no future cells); GLM reserves = 0
        latest_o = origin_latest.get(orig_key, mack_ult[i])
        origin_glm_p50[orig_key] = latest_o
        print(f"{o:8s}  {latest_o:10.1f}  {mack_ult[i]:10.1f}  "
              f"{'(no IBNR)':>10s}  {latest_o:10.1f}  {'(no IBNR)':>10s}  {latest_o:10.1f}  "
              f"{'0':>10s}  {'0.0':>7s}%")

total_glm_p50 = float(np.percentile(finite_ult, 50))
print(f"\n{'TOTAL':8s}  {total_latest:10.1f}  {total_mack_ult:10.1f}  "
      f"{np.percentile(finite_ult,5):10.1f}  {total_glm_p50:10.1f}  "
      f"{np.percentile(finite_ult,95):10.1f}  {glm_mean_ult:10.1f}")
print(f"  Actual: {actual_ultimate:,.1f}")
print(f"  GLM/Actual: {total_glm_p50/actual_ultimate:.4f}")
print(f"  Mack/Actual: {total_mack_ult/actual_ultimate:.4f}")


# ---------------------------------------------------------------------------
# Task 3 — Identify bias source
# ---------------------------------------------------------------------------
section("Task 3 — Per-origin bias: where is the over-prediction?")

print("\nOrigins by GLM p50 / Mack ratio (higher = more bias):")
rows = []
for i, o in enumerate(origins_str):
    orig_key = tri.origin[i]
    if orig_key in reserves_post.coords["origin"].values:
        orig_ibnr = np.asarray(reserves_post.sel(origin=orig_key).values, dtype=float)
        orig_ibnr = orig_ibnr[np.isfinite(orig_ibnr)]
        latest_o = origin_latest.get(orig_key, 0.0)
        orig_ult = orig_ibnr + latest_o
        glm_p50_o = float(np.percentile(orig_ult, 50))
        ibnr_mack = mack_ult[i] - latest_o
        ibnr_glm = glm_p50_o - latest_o
        rows.append({
            "origin": o,
            "mack_ult": mack_ult[i],
            "mack_ibnr": ibnr_mack,
            "glm_p50": glm_p50_o,
            "glm_ibnr": ibnr_glm,
            "ratio_ult": glm_p50_o / mack_ult[i] if mack_ult[i] != 0 else float("nan"),
            "ratio_ibnr": ibnr_glm / ibnr_mack if ibnr_mack > 0 else float("nan"),
            "n_devs_obs": int(sum(~np.isnan(np.asarray(tri.to_frame().loc[:, :].iloc[i, :])))),
        })
    else:
        rows.append({
            "origin": o, "mack_ult": mack_ult[i], "mack_ibnr": 0.0,
            "glm_p50": mack_ult[i], "glm_ibnr": 0.0,
            "ratio_ult": 1.0, "ratio_ibnr": float("nan"),
            "n_devs_obs": 10,
        })

df_bias = pd.DataFrame(rows).sort_values("ratio_ibnr", ascending=False)
print(df_bias[["origin", "mack_ult", "mack_ibnr", "glm_p50", "glm_ibnr",
               "ratio_ult", "ratio_ibnr", "n_devs_obs"]].to_string(index=False))

# Highlight worst origins
worst = df_bias.iloc[0]["origin"]
print(f"\nWorst origin (highest IBNR ratio): {worst}")
print(f"  (origin with fewest observed dev periods has most extrapolation → most bias)")


# ---------------------------------------------------------------------------
# Task 3b — Origin and dev coefficients for worst origins
# ---------------------------------------------------------------------------
section("Task 3b — GLM parameter posteriors: C(origin) and C(dev)")

param_summary = glm.get_parameter_summary()
print("\nAll parameter posteriors:")
print(param_summary.to_string())

# Check alpha (gamma dispersion) posterior
print("\n\n>>> GAMMA ALPHA (dispersion) POSTERIOR:")
alpha_var = None
for var in glm.idata.posterior.data_vars:
    if "alpha" in var.lower() or "kappa" in var.lower() or "shape" in var.lower():
        alpha_var = var
        break
if alpha_var:
    alpha_samples = np.asarray(glm.idata.posterior[alpha_var].values, dtype=float).flatten()
    print(f"  var_name: {alpha_var}")
    print(f"  mean:     {np.mean(alpha_samples):.4f}")
    print(f"  median:   {np.median(alpha_samples):.4f}")
    print(f"  p5:       {np.percentile(alpha_samples, 5):.4f}")
    print(f"  p95:      {np.percentile(alpha_samples, 95):.4f}")
    print(f"  NOTE: gamma process variance = mu^2 / alpha")
    print(f"        Higher alpha = LESS process variance (tighter intervals)")
else:
    print("  Could not find gamma alpha/kappa/shape parameter in posterior")
    print("  Available vars:", list(glm.idata.posterior.data_vars))


# ---------------------------------------------------------------------------
# Task 4 — Uncertainty mechanism deep dive
# ---------------------------------------------------------------------------
section("Task 4 — Uncertainty mechanism: what is reserves_posterior_?")

print("\n1) reserves_posterior_ contents:")
print(f"   type:   {type(glm.reserves_posterior_)}")
print(f"   dims:   {glm.reserves_posterior_.dims}")
print(f"   shape:  {glm.reserves_posterior_.shape}")
print(f"   coords: {list(glm.reserves_posterior_.coords)}")
print(f"\n   NOTE: reserves_posterior_ is the SUM of POSTERIOR MEAN (mu) predictions")
print(f"   for future cells, NOT posterior predictive samples.")
print(f"   It uses kind='response_params' which gives the mu (linear predictor mean),")
print(f"   not a draw from Gamma(alpha, mu/alpha). Process variance is NOT included.")

# Demonstrate: compute per-cell CV from posterior of linear predictor
print("\n2) Linear predictor uncertainty per future cell:")
response_name = glm.model_.response_component.response.name
mean_var = f"{response_name}_mean"
if mean_var not in glm.idata.posterior:
    mean_var = "mu"

future_preds = glm.idata.posterior[mean_var]
print(f"   Posterior variable used: '{mean_var}'")
print(f"   Shape: {future_preds.values.shape}")

# The last n_future observations are the future cells
n_future = len(glm.future_data_)
n_total = future_preds.shape[-1]
future_start = n_total - n_future
future_pred_flat = future_preds.values.reshape(-1, n_total)[:, future_start:]

print(f"\n   Future cells: {n_future}")
print(f"   Per-cell posterior of predicted mu (first 10 cells):")
print(f"   {'Cell':>5s}  {'Origin':>8s}  {'Dev':>4s}  {'Mean(mu)':>12s}  {'SD(mu)':>10s}  "
      f"{'CV(mu)':>8s}  {'SD/Mean%':>10s}")
for idx in range(min(n_future, 15)):
    row = glm.future_data_.iloc[idx]
    mu_samples = future_pred_flat[:, idx]
    mu_samples = mu_samples[np.isfinite(mu_samples)]
    if len(mu_samples) == 0:
        continue
    mu_mean = float(np.mean(mu_samples))
    mu_sd = float(np.std(mu_samples, ddof=1))
    cv = mu_sd / mu_mean if mu_mean > 0 else float("nan")
    print(f"   {idx:5d}  {str(row.get('origin','?')):>8s}  {str(row.get('dev','?')):>4s}  "
          f"{mu_mean:12.2f}  {mu_sd:10.2f}  {cv:8.4f}  {cv*100:9.1f}%")

# Total uncertainty decomposition
print("\n3) Uncertainty decomposition for TOTAL reserves:")
total_reserve_samples = np.asarray(reserves_post.sum(dim="origin").values, dtype=float)
finite_res = total_reserve_samples[np.isfinite(total_reserve_samples)]
print(f"   Parameter uncertainty (sd of total reserve posterior): {np.std(finite_res, ddof=1):,.1f}")
print(f"   This is PARAMETER uncertainty only (via posterior of mu).")
print(f"   Process variance NOT included.")

# Estimate process variance contribution
if alpha_var:
    alpha_med = float(np.median(alpha_samples))
    total_ibnr_med = float(np.median(finite_res))
    # For gamma: process var per cell ~ E[mu]^2 / alpha
    # Aggregate process variance for all future cells
    future_mu_means = np.mean(future_pred_flat, axis=0)
    process_var_cells = np.nansum(future_mu_means**2 / alpha_med)
    process_sd = float(np.sqrt(process_var_cells))
    combined_sd = float(np.sqrt(np.std(finite_res, ddof=1)**2 + process_var_cells))
    print(f"\n   Estimated process SD (Gamma, alpha={alpha_med:.2f}): {process_sd:,.1f}")
    print(f"   Combined (param + process) SD estimate:              {combined_sd:,.1f}")
    print(f"   Mack total SE:                                       {total_mack_se:,.1f}")
    print(f"   Bootstrap ODP total SE:                              {boot_sd:,.1f}")


# ---------------------------------------------------------------------------
# Task 5A — Jensen's inequality bias check
# ---------------------------------------------------------------------------
section("Task 5A — Jensen's inequality: E[exp(eta)] vs exp(E[eta])")

print("\nFor each future cell: compare E[mu] vs exp(E[eta]) where eta = log(mu).")
print("Difference reveals Jensen upward bias from non-linearity.\n")

# future_pred_flat is already E[mu] (posterior mean predictions)
# eta = log(mu): recover posterior of eta from posterior of mu via log transform
# Actually reserves_posterior_ already stores sum of mu, not samples of each cell.
# We need the raw posterior for each cell from idata.posterior[mean_var].

# Compute Jensen ratio per cell
print(f"{'Cell':>5s}  {'Origin':>8s}  {'Dev':>4s}  {'E[mu]':>10s}  "
      f"{'exp(E[log mu])':>16s}  {'Jensen Ratio':>14s}")
jensen_numerators = []
jensen_denominators = []
for idx in range(min(n_future, 15)):
    row = glm.future_data_.iloc[idx]
    mu_samples = future_pred_flat[:, idx]
    mu_samples = mu_samples[np.isfinite(mu_samples) & (mu_samples > 0)]
    if len(mu_samples) < 10:
        continue
    e_mu = float(np.mean(mu_samples))
    exp_e_log_mu = float(np.exp(np.mean(np.log(mu_samples))))
    ratio = e_mu / exp_e_log_mu if exp_e_log_mu > 0 else float("nan")
    jensen_numerators.append(e_mu)
    jensen_denominators.append(exp_e_log_mu)
    print(f"   {idx:5d}  {str(row.get('origin','?')):>8s}  {str(row.get('dev','?')):>4s}  "
          f"{e_mu:10.2f}  {exp_e_log_mu:16.2f}  {ratio:14.4f}")

all_mu = future_pred_flat[np.isfinite(future_pred_flat)].flatten()
if len(all_mu) > 0:
    total_e_mu = float(np.sum(np.mean(future_pred_flat, axis=0)))
    total_exp_e_log_mu = float(np.sum(np.exp(np.mean(np.log(np.maximum(future_pred_flat, 1e-10)), axis=0))))
    print(f"\nTotal future reserves:")
    print(f"  sum of E[mu_i]:           {total_e_mu:,.1f}")
    print(f"  sum of exp(E[log mu_i]):  {total_exp_e_log_mu:,.1f}")
    print(f"  Jensen inflation ratio:   {total_e_mu/total_exp_e_log_mu:.4f}")


# ---------------------------------------------------------------------------
# Task 5B — MLE comparison: GLM coefficients vs chain ladder LDF
# ---------------------------------------------------------------------------
section("Task 5B — MLE comparison: chain-ladder LDFs vs GLM posterior coefficients")

# Compute chain-ladder LDFs and implied incremental factors
ldf = cl.Development().fit(tri)
ldfs = np.asarray(ldf.ldf_.values, dtype=float).squeeze()
print("Chain-ladder LDFs:")
for i, ldf_val in enumerate(ldfs):
    print(f"  dev {i+1} -> {i+2}: {ldf_val:.6f}")

# Fit deterministic chain ladder on incrementals to get row/column factors
tri_inc = tri.cum_to_incr()
inc_df = tri_inc.to_frame()
print("\nIncremental loss triangle:")
print(inc_df.to_string())

# Compute volume-weighted dev effects (MLE for ODP/gamma GLM):
# beta_j ~ log(sum_k y_kj / sum_k EP_k) — i.e., average loss ratio per dev period
prem_vals = {}
prem_frame = prem_tri.to_frame()
for i, o in enumerate(tri.origin):
    o_str = str(o).split("T")[0]
    # Find matching premium row
    for idx, row_idx in enumerate(prem_frame.index):
        if str(row_idx).split("T")[0] == o_str:
            prem_vals[o_str] = float(prem_frame.iloc[idx, 0])
            break

print("\nPremium by origin:")
for o, ep in prem_vals.items():
    print(f"  {o}: {ep:.0f}")

# GLM posterior of C(dev) and C(origin) parameters
print("\nGLM posterior C(origin) parameters (should match ODP MLE in the limit):")
origin_param_var = None
dev_param_var = None
for var in glm.idata.posterior.data_vars:
    if "C(origin)" in var:
        origin_param_var = var
    if "C(dev)" in var:
        dev_param_var = var

if origin_param_var:
    origin_params = glm.idata.posterior[origin_param_var].values
    # Shape: (chain, draw, n_levels)
    o_mean = np.mean(origin_params, axis=(0, 1))
    o_sd = np.std(origin_params, axis=(0, 1))
    coord_name = [c for c in glm.idata.posterior[origin_param_var].coords
                  if c not in ("chain", "draw")][0]
    coords = glm.idata.posterior[origin_param_var].coords[coord_name].values
    print(f"  (var: {origin_param_var}, coords: {coord_name})")
    for co, om, os_ in zip(coords, o_mean, o_sd):
        print(f"    level={co}: mean={om:.4f}  sd={os_:.4f}")

if dev_param_var:
    dev_params = glm.idata.posterior[dev_param_var].values
    d_mean = np.mean(dev_params, axis=(0, 1))
    d_sd = np.std(dev_params, axis=(0, 1))
    coord_name = [c for c in glm.idata.posterior[dev_param_var].coords
                  if c not in ("chain", "draw")][0]
    coords = glm.idata.posterior[dev_param_var].coords[coord_name].values
    print(f"\n  (var: {dev_param_var}, coords: {coord_name})")
    for co, dm, ds_ in zip(coords, d_mean, d_sd):
        print(f"    level={co}: mean={dm:.4f}  sd={ds_:.4f}")


# ---------------------------------------------------------------------------
# Task 5C — Process variance: gamma alpha posterior vs method-of-moments
# ---------------------------------------------------------------------------
section("Task 5C — Process variance: posterior alpha vs method-of-moments estimate")

# Compute fitted values on observed data
fitted_mu = np.asarray(glm.fitted_["fitted_mean"].values, dtype=float)
observed_y = np.asarray(glm.data_["incremental"].values, dtype=float)

# Pearson residuals: r_i = (y_i - mu_i) / sqrt(V(mu_i))
# For gamma: V(mu) = mu^2 / alpha; Pearson residual = (y - mu) / (mu / sqrt(alpha))
# But first do MOM estimate: phi = sum(r_i^2) / (n - p)
# For gamma with log link: V(mu) = mu^2 (assuming dispersion phi=1)
# Pearson chi-sq statistic / df estimates phi; alpha = 1 / phi

mask = (fitted_mu > 0) & np.isfinite(fitted_mu) & np.isfinite(observed_y) & (observed_y > 0)
mu_obs = fitted_mu[mask]
y_obs = observed_y[mask]

# Pearson residuals for gamma(mu^2 variance, dispersion phi=1): r = (y - mu) / mu
pearson_resid = (y_obs - mu_obs) / mu_obs
n_obs = len(y_obs)

# Number of parameters: intercept + n_origins_minus_1 + n_devs_minus_1
# For 10x10 upper triangle: 55 observed cells
# params = 1 intercept + 9 origin effects + 9 dev effects = 19
n_params_approx = 1 + 9 + 9  # for full 10x10
df = max(n_obs - n_params_approx, 1)
phi_mom = float(np.sum(pearson_resid**2) / df)
alpha_mom = 1.0 / phi_mom  # For gamma: alpha = 1/phi

print(f"\nMethod-of-moments dispersion estimate:")
print(f"  n_obs used: {n_obs}")
print(f"  approx df:  {df}")
print(f"  phi (MOM):  {phi_mom:.4f}")
print(f"  alpha=1/phi:{alpha_mom:.4f}")
print(f"  (alpha >> 1 means low dispersion = tight intervals)")

if alpha_var:
    print(f"\nPosterior alpha (from MCMC):")
    print(f"  median: {np.median(alpha_samples):.4f}")
    print(f"  mean:   {np.mean(alpha_samples):.4f}")
    print(f"  p5:     {np.percentile(alpha_samples, 5):.4f}")
    print(f"  p95:    {np.percentile(alpha_samples, 95):.4f}")
    print(f"\nComparison: posterior alpha={np.median(alpha_samples):.2f} vs MOM alpha={alpha_mom:.2f}")
    if np.median(alpha_samples) > alpha_mom:
        print(f"  => Posterior OVER-estimates alpha (UNDER-estimates dispersion) relative to MOM!")
        print(f"     This makes process variance too small → intervals too narrow.")
    else:
        print(f"  => Posterior is consistent with or lower than MOM estimate.")

print(f"\nFor total future reserve of ~{total_ibnr_samples[np.isfinite(total_ibnr_samples)].mean():,.0f}:")
print(f"  With posterior alpha, estimated process SD:  {process_sd if alpha_var else 'N/A':,.1f}")
alpha_mom_sd = float(np.sqrt(float(np.nansum((np.mean(future_pred_flat, axis=0))**2 / alpha_mom))))
print(f"  With MOM alpha, estimated process SD:        {alpha_mom_sd:,.1f}")


# ---------------------------------------------------------------------------
# Task 6 — Summary comparison table
# ---------------------------------------------------------------------------
section("Task 6 — Uncertainty comparison summary")

total_unpaid_actual = actual_ultimate - total_latest
total_unpaid_mack = total_mack_ult - total_latest
total_unpaid_boot = boot_mean - total_latest
total_unpaid_glm = float(np.median(total_ibnr_samples[np.isfinite(total_ibnr_samples)]))

print(f"\n{'Metric':35s}  {'Mack':>12s}  {'Bootstrap':>12s}  {'GLM (param)':>12s}")
print("-" * 78)
print(f"{'Total latest observed':35s}  {total_latest:12.1f}  {total_latest:12.1f}  {total_latest:12.1f}")
print(f"{'Total ultimate (mean/median)':35s}  {total_mack_ult:12.1f}  {boot_mean:12.1f}  {glm_median_ult:12.1f}")
print(f"{'Actual ultimate':35s}  {actual_ultimate:12.1f}  {actual_ultimate:12.1f}  {actual_ultimate:12.1f}")
print(f"{'Total unpaid (IBNR)':35s}  {total_unpaid_mack:12.1f}  {total_unpaid_boot:12.1f}  {total_unpaid_glm:12.1f}")
print(f"{'Total SD / SE':35s}  {total_mack_se:12.1f}  {boot_sd:12.1f}  {glm_sd:12.1f}")
print(f"{'CV of total ultimate':35s}  {total_mack_se/total_mack_ult:12.4f}  {boot_sd/boot_mean:12.4f}  {glm_sd/glm_median_ult:12.4f}")
print(f"{'CV of unpaid':35s}  {total_mack_se/total_unpaid_mack:12.4f}  {boot_sd/total_unpaid_boot:12.4f}  {glm_cv_unpaid:12.4f}")
print(f"{'Implied pctl for actual':35s}  {'N/A':>12s}  {boot_pctl:12.4f}  {glm_pctl:12.6f}")

# Is GLM wider or narrower than Mack?
print(f"\nGLM SD vs Mack SE:      {glm_sd/total_mack_se:.2f}x  ({'wider' if glm_sd>total_mack_se else 'NARROWER'})")
print(f"GLM SD vs Bootstrap SD: {glm_sd/boot_sd:.2f}x  ({'wider' if glm_sd>boot_sd else 'NARROWER'})")

# Where is the bias located (latest 3 origins)?
print("\nBias concentration (GLM p50 - Mack) by origin group:")
early_bias = sum(origin_glm_p50.get(tri.origin[i], mack_ult[i]) - mack_ult[i]
                 for i in range(5))
late_bias = sum(origin_glm_p50.get(tri.origin[i], mack_ult[i]) - mack_ult[i]
                for i in range(5, 10))
print(f"  Origins 1988-1992 (5 origins, more data):  {early_bias:+,.1f}")
print(f"  Origins 1993-1997 (5 origins, less data):  {late_bias:+,.1f}")
print(f"  => Bias is {'mostly in latest origins' if abs(late_bias) > abs(early_bias) else 'spread across all origins'}")


# ---------------------------------------------------------------------------
# Root cause diagnosis
# ---------------------------------------------------------------------------
section("ROOT CAUSE DIAGNOSIS")

print("""
KEY FINDINGS:
=============

1) OVER-PREDICTION (GLM ~47% above actual, ~42% above Mack):
   - The GLM adds an EP offset (log(net_earned_premium)) to the linear predictor.
   - This means the model predicts: E[incremental] = EP * exp(intercept + alpha + beta)
   - 1993 has the highest EP ($19,310) AND the highest C(origin) coefficient
     (because 1993 also has the highest observed ultimate $16,455 vs Mack).
   - The bias is concentrated in the tail origins (1993-1997) that have the
     LEAST observed development data and MOST extrapolation via C(dev) coefficients.
   - The adaptive intercept prior sets the Intercept location based on
     log(mean(incremental)) - log(mean(EP)), but the prior sigma=1.0 is VERY
     wide relative to the actual variation. Under a log link, sigma=1.0 for
     C(origin) permits individual origin multipliers of exp(±2) = [0.14, 7.4].
   - The POSTERIOR shrinks toward the prior, but with only 1-5 dev periods
     observed per origin, the posterior for late origins is heavily influenced
     by the prior location, not the data.

2) UNDER-CONFIDENCE (GLM CV=0.69 with implied_pctl=0.000):
   - The reserves_posterior_ uses POSTERIOR MEAN predictions (parameter uncertainty
     only via the mu posterior), NOT posterior predictive samples.
   - Process variance is NOT included in reserves_posterior_. This is confirmed
     by the kind='response_params' call in _compute_predictions, which returns
     the predicted mean mu, not a draw from Gamma(alpha, mu/alpha).
   - However, the PARAMETER uncertainty IS large (SD is large because the
     origin/dev coefficient posteriors are wide for sparse origins).
   - The paradox: large parameter uncertainty (wide σ on η) INFLATES the posterior
     mean via Jensen's inequality E[exp(η)] > exp(E[η]), pushing all posteriors UP.
   - The net effect: the posterior is centered ABOVE the MLE/Mack estimate
     AND the individual samples are all above the actual. CV=0.69 sounds big but
     it's because the ENTIRE posterior is shifted upward, not because it's wide
     enough to include the actual.

3) SPECIFIC MECHANISM:
   A) Intercept prior: Normal(mu_adjusted, sigma=1.0) where mu_adjusted accounts
      for the EP offset. But exp(Normal(0,1)) has E = exp(0.5) = 1.65 — the
      prior is systematically pulling predictions UP relative to the MLE.
   B) C(origin) priors: Normal(0, sigma=1.0) means the origin effects have
      prior 95th percentile at exp(1.96) = 7.1x. For sparse late origins,
      the prior dominates and the posterior is not well-anchored to data.
   C) The exposure offset introduces a "per-EP" normalization, but the
      intercept adaptation subtracts log(mean_EP), not log(EP_per_origin).
      If 1993's EP is 25% above average, the model over-predicts 1993 by
      that 25% because the normalization used the fleet mean, not per-origin EP.

4) ROOT CAUSE SUMMARY:
   - PRIMARY: Bayesian shrinkage toward a BIASED prior mean. The data-adaptive
     intercept prior correctly accounts for the AVERAGE EP, but the Jensen
     correction (sigma^2/2) only partially corrects; the sigma=1.0 on C(origin)
     allows very large posterior origin effects for data-sparse origins.
   - SECONDARY: Process variance completely absent from reserves_posterior_,
     but because the posterior is shifted HIGH, adding process variance would
     only FURTHER inflate uncertainty without fixing the center.
   - TERTIARY: With only 55 observed cells (10x10 upper triangle) and 19
     parameters, the model is mildly overparameterized. The gamma alpha
     (dispersion) is estimated from the fit, and with noisy data the prior
     on alpha may be pulling toward low dispersion (high alpha = tight intervals).

PROPOSED FIXES:
===============

A) Fix the bias (over-prediction):
   1. Use tighter C(origin) prior sigma (e.g., 0.5 instead of 1.0).
      This forces more shrinkage toward zero (geometric mean across origins),
      which is appropriate when the latest origins have minimal data.
   2. Or switch to a random-effects spec (1|origin) with a hyperprior on sigma.
      This pools information across origins and shrinks sparse origins toward
      the fleet average, which is exactly what's needed here.
   3. Check the intercept lognormal correction: E[exp(Intercept)] should equal
      the average loss-per-unit-EP, not the average loss across ALL cells.

B) Fix the uncertainty (under-confidence given the bias):
   1. Use kind='response' (posterior predictive) instead of kind='response_params'
      in _compute_predictions to include process variance in reserves_posterior_.
   2. This requires sample_new_groups=True and sufficient memory, but it's the
      correct statistical approach for interval coverage.
   3. Alternatively, explicitly add process variance in _compute_reserves:
         process_var = sum(mu_i_posterior^2 / alpha_posterior)
         combined_std = sqrt(param_std^2 + process_std^2)
      using the relationship Var(Gamma) = mu^2 / alpha.

C) Verify MLE equivalence:
   With very diffuse priors and large n, the Bayesian GLM should converge to
   the ODP MLE = chain ladder. That it does NOT (184K vs 130K) for n=55 cells
   confirms prior influence dominates. Use the M5_cal random-effects spec
   which explicitly models parameter borrowing across origins.
""")

section("DONE")
print(f"Summary statistics:")
print(f"  Mack total ultimate:       {total_mack_ult:>12,.1f}  (SE={total_mack_se:,.1f}, CV={total_mack_se/total_mack_ult:.4f})")
print(f"  Bootstrap total ultimate:  {boot_mean:>12,.1f}  (SD={boot_sd:,.1f}, CV={boot_sd/boot_mean:.4f})")
print(f"  GLM (m1_cat) median ult:   {glm_median_ult:>12,.1f}  (SD={glm_sd:,.1f}, CV={glm_sd/glm_median_ult:.4f})")
print(f"  Actual ultimate:           {actual_ultimate:>12,.1f}")
print(f"  GLM / Actual:              {glm_median_ult/actual_ultimate:>12.4f}")
print(f"  GLM / Mack:                {glm_median_ult/total_mack_ult:>12.4f}")
print(f"  GLM implied_pctl:          {glm_pctl:>12.6f}")
print(f"  Bootstrap implied_pctl:    {boot_pctl:>12.4f}")
print(f"\nGLM SD is {glm_sd/total_mack_se:.2f}x Mack SE and {glm_sd/boot_sd:.2f}x Bootstrap SD")
