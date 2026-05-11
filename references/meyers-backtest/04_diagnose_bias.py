"""Diagnose the ~17% systematic over-prediction of BayesianChainLadderGLM M2.

Diagnostic triangle: wkcomp / group_id=21172 (Vanliner Ins Co)
Backtest ratio glm_m2 / mack = 1.143 (close to median 1.164).

Run:
    cd references/meyers-backtest
    uv run python 04_diagnose_bias.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import chainladder as cl
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths / imports
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))   # _common.py
sys.path.insert(0, str(REPO_ROOT))                         # bayesianchainladder package

from _common import load_exposure_triangle  # noqa: E402

import reservetestr  # noqa: E402
from bayesianchainladder import BayesianChainLadderGLM  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LINE = "wkcomp"
GROUP_ID = 21172
LOSS_TYPE = "paid"

DRAWS = 1000
TUNE = 1000
CHAINS = 2
TARGET_ACCEPT = 0.95
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Helper: empirical posterior median of total ultimate
# ---------------------------------------------------------------------------

def total_posterior_median(model: BayesianChainLadderGLM) -> float:
    """Posterior MEDIAN of total ultimate across all samples."""
    total_ibnr = model.reserves_posterior_.sum(dim="origin").values.flatten()
    total_ibnr = total_ibnr[np.isfinite(total_ibnr)]
    paid_total = float(model.ultimate_["paid_to_date"].sum())
    return float(np.median(total_ibnr)) + paid_total


def total_posterior_mean(model: BayesianChainLadderGLM) -> float:
    """Posterior MEAN of total ultimate across all samples."""
    total_ibnr = model.reserves_posterior_.sum(dim="origin").values.flatten()
    total_ibnr = total_ibnr[np.isfinite(total_ibnr)]
    paid_total = float(model.ultimate_["paid_to_date"].sum())
    return float(np.mean(total_ibnr)) + paid_total


# ---------------------------------------------------------------------------
# 1. Load triangle
# ---------------------------------------------------------------------------

print("=" * 72)
print(f"Diagnostic triangle: {LINE} / group_id={GROUP_ID}")
print("=" * 72)

records = reservetestr.build_triangle_records()
rec = next(r for r in records if r.line == LINE and r.group_id == GROUP_ID)
train_tri = rec.train_triangles[LOSS_TYPE]
prem_tri = load_exposure_triangle(LINE, GROUP_ID)

print(f"\nCompany: {rec.company}")
print(f"Actual ultimate (held-out): {rec.actual_ultimates.get(LOSS_TYPE, 'N/A'):,.0f}")
print("\nTrain triangle (cumulative paid):")
print(train_tri)

# ---------------------------------------------------------------------------
# 2. Mack Chain Ladder
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("2. Mack Chain Ladder")
print("=" * 72)

mack = cl.MackChainladder().fit(train_tri)
mack_ult_arr = np.asarray(mack.ultimate_.values, dtype=float).flatten()
mack_ult_arr = mack_ult_arr[np.isfinite(mack_ult_arr)]
mack_total = float(mack_ult_arr.sum())

mack_df = mack.ultimate_.to_frame()
mack_dict = {
    str(idx.year) if hasattr(idx, "year") else str(idx): float(v)
    for idx, v in zip(mack_df.index, mack_df.iloc[:, 0])
}
print("Mack ultimates per origin:")
for yr, v in sorted(mack_dict.items()):
    print(f"  {yr}: {v:>12,.1f}")
print(f"  Total: {mack_total:>11,.1f}")

# ---------------------------------------------------------------------------
# 3. Deterministic ODP (chain ladder MLE)
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("3. Deterministic Chain Ladder (ODP MLE)")
print("=" * 72)

det_cl = cl.Chainladder().fit(cl.Development().fit_transform(train_tri))
cl_ult_arr = np.asarray(det_cl.ultimate_.values, dtype=float).flatten()
cl_ult_arr = cl_ult_arr[np.isfinite(cl_ult_arr)]
cl_total = float(cl_ult_arr.sum())

cl_df = det_cl.ultimate_.to_frame()
cl_dict = {
    str(idx.year) if hasattr(idx, "year") else str(idx): float(v)
    for idx, v in zip(cl_df.index, cl_df.iloc[:, 0])
}
print("Det CL ultimates per origin:")
for yr, v in sorted(cl_dict.items()):
    print(f"  {yr}: {v:>12,.1f}")
print(f"  Total: {cl_total:>11,.1f}")

# ---------------------------------------------------------------------------
# 4. BayesianChainLadderGLM (formula = C(origin) + bs(dev_idx, df=4))
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("4. BayesianChainLadderGLM (M2 spec: gamma+log, C(origin)+bs(dev_idx,df=4))")
print("=" * 72)
print(f"   draws={DRAWS}, tune={TUNE}, chains={CHAINS}, target_accept={TARGET_ACCEPT}")

model = BayesianChainLadderGLM(
    formula="incremental ~ 1 + C(origin) + bs(dev_idx, df=4)",
    family="gamma",
    link="log",
    exposure="net_earned_premium",
    response_per_exposure=False,
    priors=None,    # adaptive priors
    draws=DRAWS,
    tune=TUNE,
    chains=CHAINS,
    target_accept=TARGET_ACCEPT,
    random_seed=RANDOM_SEED,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.fit(train_tri, exposure_triangle=prem_tri)

print("\nGLM ultimate_ (per origin):")
print(model.ultimate_[["paid_to_date", "mean", "median"]].to_string())

glm_total_mean = float(model.ultimate_["mean"].sum())
glm_total_median = float(model.ultimate_["median"].sum())

# Posterior MEDIAN and MEAN of TOTAL ultimate (aggregate over origins, then reduce)
rp = model.reserves_posterior_
total_ibnr_samples = rp.sum(dim="origin").values.flatten()
total_ibnr_samples = total_ibnr_samples[np.isfinite(total_ibnr_samples)]
paid_total = float(model.ultimate_["paid_to_date"].sum())

glm_total_ult_samples = total_ibnr_samples + paid_total
post_mean_total = float(np.mean(glm_total_ult_samples))
post_median_total = float(np.median(glm_total_ult_samples))
post_p75_total = float(np.percentile(glm_total_ult_samples, 75))

print(f"\nAggregate ultimate statistics:")
print(f"  Sum of per-origin posterior means:    {glm_total_mean:>12,.1f}")
print(f"  Sum of per-origin posterior medians:  {glm_total_median:>12,.1f}")
print(f"  Posterior MEAN of total:              {post_mean_total:>12,.1f}")
print(f"  Posterior MEDIAN of total:            {post_median_total:>12,.1f}")
print(f"  Posterior 75th pctile of total:       {post_p75_total:>12,.1f}")

# ---------------------------------------------------------------------------
# 5. Jensen's bias investigation
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("5. Jensen's Inequality Bias Investigation")
print("=" * 72)

# reserves_posterior_ contains mu[s,j] = E[Y_j | theta_s] = exp(eta[s,j])
# for gamma+log link. Bambi's kind='response_params' gives us E[Y|params] per draw.
# mean_ultimate_est = E_s[sum_origin(paid + sum_future(exp(eta[s,j])))]
#                  > sum_origin(paid + sum_future(exp(E_s[eta[s,j]])))
# by Jensen's inequality (exp is convex).

# Compute exp(E_s[eta_j]) per origin as a counterfactual
idata = model.idata
response_name = model.model_.response_component.response.name
mean_name = f"{response_name}_mean"
if mean_name not in idata.posterior:
    mean_name = "mu"

mu_da = idata.posterior[mean_name]  # shape (chain, draw, __obs__)
print(f"\nPosterior mu variable shape: {dict(mu_da.sizes)}")
print(f"Future cells: {len(model.future_data_)}")

# Per-cell posterior mean and std of log(mu)
# For gamma+log: mu = exp(eta), so log(mu) = eta (the linear predictor)
log_mu = np.log(mu_da.values)  # (chain, draw, obs)
log_mu_flat = log_mu.reshape(-1, log_mu.shape[-1])  # (samples, obs)

eta_post_mean = log_mu_flat.mean(axis=0)   # (obs,)
eta_post_std = log_mu_flat.std(axis=0)     # (obs,)

print(f"\nPosterior std of linear predictor (eta) per future cell:")
print(f"  Mean across cells: {eta_post_std.mean():.4f}")
print(f"  Min: {eta_post_std.min():.4f}, Max: {eta_post_std.max():.4f}")
print(f"  Median: {np.median(eta_post_std):.4f}")

# Jensen bias factor per cell: E[exp(eta)] / exp(E[eta]) = exp(sigma_eta^2 / 2)
# For a normal posterior this is exact; for any distribution it's a lower bound
jensen_bias_per_cell = mu_da.values.reshape(-1, mu_da.sizes["__obs__"]).mean(axis=0) / np.exp(eta_post_mean)
print(f"\nJensen bias per cell (E[exp(eta)] / exp(E[eta])):")
print(f"  Mean: {jensen_bias_per_cell.mean():.4f}")
print(f"  Min: {jensen_bias_per_cell.min():.4f}, Max: {jensen_bias_per_cell.max():.4f}")
print(f"  Median: {np.median(jensen_bias_per_cell):.4f}")

# Counterfactual: if we used exp(E[eta]) instead of E[exp(eta)]
# i.e., point-prediction at posterior mean parameters
mu_at_post_mean = np.exp(eta_post_mean)  # (obs,) for future cells only

# Map back to origins
future_data = model.future_data_.reset_index(drop=True)
origin_col = future_data["origin"]
origins_sorted = sorted(origin_col.unique())

print("\nPer-origin comparison:")
print(f"  {'Origin':<8} {'Mack':>12} {'Det CL':>12} {'GLM mean µ':>12} {'exp(mean η)':>12} {'Jensen ratio':>12}")
print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

counterfactual_totals = []
glm_mean_totals = []

paid = model._paid_to_date()

for origin in origins_sorted:
    yr = str(origin)
    mask = (origin_col == origin).values
    if mask.any():
        # GLM mean E[mu|theta] for this origin (what the model reports)
        glm_mean_origin = float(model.ultimate_["mean"].get(origin, np.nan))
        # exp(E[eta]): counterfactual using posterior mean of linear predictor
        mu_at_mean_origin = float(np.exp(eta_post_mean[mask]).sum()) + float(paid.get(origin, 0.0))
        counterfactual_totals.append(mu_at_mean_origin)
        glm_mean_totals.append(glm_mean_origin)
        jensen_ratio = glm_mean_origin / mu_at_mean_origin if mu_at_mean_origin > 0 else np.nan

        mack_v = mack_dict.get(yr, np.nan)
        cl_v = cl_dict.get(yr, np.nan)
        print(f"  {yr:<8} {mack_v:>12,.1f} {cl_v:>12,.1f} {glm_mean_origin:>12,.1f} {mu_at_mean_origin:>12,.1f} {jensen_ratio:>12.4f}")
    else:
        # Fully developed origin
        paid_origin = float(paid.get(origin, 0.0))
        counterfactual_totals.append(paid_origin)
        glm_mean_totals.append(paid_origin)
        mack_v = mack_dict.get(yr, np.nan)
        cl_v = cl_dict.get(yr, np.nan)
        print(f"  {yr:<8} {mack_v:>12,.1f} {cl_v:>12,.1f} {paid_origin:>12,.1f} {paid_origin:>12,.1f} {'1.0000':>12}")

cf_total = sum(counterfactual_totals)
print(f"\n  {'Total':<8} {mack_total:>12,.1f} {cl_total:>12,.1f} {post_mean_total:>12,.1f} {cf_total:>12,.1f}")
print(f"\nJensen bias (total posterior mean / counterfactual exp(E[eta]) total):")
print(f"  {post_mean_total / cf_total:.4f}")
print(f"\nBias vs Mack (total posterior mean / mack):")
print(f"  {post_mean_total / mack_total:.4f}")
print(f"\nBias vs Mack (counterfactual exp(E[eta]) total / mack):")
print(f"  {cf_total / mack_total:.4f}")

# ---------------------------------------------------------------------------
# 6. Side-by-side comparison table
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("6. Side-by-Side Comparison Table")
print("=" * 72)
print(f"\n{'Origin':<8} {'Mack':>12} {'Det CL':>12} {'GLM med':>12} {'GLM mean':>12}")
print(f"{'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

for origin in sorted(mack_dict.keys()):
    mack_v = mack_dict.get(origin, np.nan)
    cl_v = cl_dict.get(origin, np.nan)
    try:
        origin_int = int(origin)
    except ValueError:
        origin_int = None
    glm_med = float(model.ultimate_["median"].get(origin_int, np.nan)) if origin_int else np.nan
    glm_mn = float(model.ultimate_["mean"].get(origin_int, np.nan)) if origin_int else np.nan
    print(f"{origin:<8} {mack_v:>12,.1f} {cl_v:>12,.1f} {glm_med:>12,.1f} {glm_mn:>12,.1f}")

print(f"{'Total':<8} {mack_total:>12,.1f} {cl_total:>12,.1f} {glm_total_median:>12,.1f} {post_mean_total:>12,.1f}")

print(f"\nRatios vs Mack:")
print(f"  Det CL / Mack:           {cl_total / mack_total:.4f}")
print(f"  GLM median / Mack:       {glm_total_median / mack_total:.4f}")
print(f"  GLM posterior mean / Mack: {post_mean_total / mack_total:.4f}")
print(f"  GLM posterior median / Mack: {post_median_total / mack_total:.4f}")

# ---------------------------------------------------------------------------
# 7. Bias decomposition: C(dev) vs bs(dev_idx) vs no-exposure offset
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("7. Bias decomposition: alternate formula specs")
print("=" * 72)

# 7a. C(dev) instead of bs(dev_idx) — exact ODP-equivalent (no spline smoothing)
print("\n7a. Formula: C(origin) + C(dev) [exact ODP, no spline smoothing]")
model_cdev = BayesianChainLadderGLM(
    formula="incremental ~ 1 + C(origin) + C(dev)",
    family="gamma",
    link="log",
    exposure="net_earned_premium",
    response_per_exposure=False,
    priors=None,
    draws=DRAWS,
    tune=TUNE,
    chains=CHAINS,
    target_accept=TARGET_ACCEPT,
    random_seed=RANDOM_SEED,
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model_cdev.fit(train_tri, exposure_triangle=prem_tri)

cdev_ibnr = model_cdev.reserves_posterior_.sum(dim="origin").values.flatten()
cdev_ibnr = cdev_ibnr[np.isfinite(cdev_ibnr)]
cdev_paid = float(model_cdev.ultimate_["paid_to_date"].sum())
cdev_mean_total = float(np.mean(cdev_ibnr)) + cdev_paid
cdev_median_total = float(np.median(cdev_ibnr)) + cdev_paid
print(f"  Posterior mean total:   {cdev_mean_total:>12,.1f}  (ratio vs Mack: {cdev_mean_total/mack_total:.4f})")
print(f"  Posterior median total: {cdev_median_total:>12,.1f}  (ratio vs Mack: {cdev_median_total/mack_total:.4f})")

# 7b. bs(dev_idx) without exposure offset
print("\n7b. Formula: C(origin) + bs(dev_idx, df=4), NO exposure offset")
model_nox = BayesianChainLadderGLM(
    formula="incremental ~ 1 + C(origin) + bs(dev_idx, df=4)",
    family="gamma",
    link="log",
    exposure=None,           # no offset
    response_per_exposure=False,
    priors=None,
    draws=DRAWS,
    tune=TUNE,
    chains=CHAINS,
    target_accept=TARGET_ACCEPT,
    random_seed=RANDOM_SEED,
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model_nox.fit(train_tri)

nox_ibnr = model_nox.reserves_posterior_.sum(dim="origin").values.flatten()
nox_ibnr = nox_ibnr[np.isfinite(nox_ibnr)]
nox_paid = float(model_nox.ultimate_["paid_to_date"].sum())
nox_mean_total = float(np.mean(nox_ibnr)) + nox_paid
nox_median_total = float(np.median(nox_ibnr)) + nox_paid
print(f"  Posterior mean total:   {nox_mean_total:>12,.1f}  (ratio vs Mack: {nox_mean_total/mack_total:.4f})")
print(f"  Posterior median total: {nox_median_total:>12,.1f}  (ratio vs Mack: {nox_median_total/mack_total:.4f})")

print("\nBias decomposition summary:")
print(f"  Original M2 (bs+offset) posterior mean / Mack:  {post_mean_total/mack_total:.4f}")
print(f"  C(dev)+offset posterior mean / Mack:             {cdev_mean_total/mack_total:.4f}")
print(f"  bs+no-offset posterior mean / Mack:              {nox_mean_total/mack_total:.4f}")
print(f"  Spline vs C(dev) contribution:                   {post_mean_total/cdev_mean_total:.4f}")
print(f"  Offset contribution (bs+offset / bs+no-offset):  {post_mean_total/nox_mean_total:.4f}")

# ---------------------------------------------------------------------------
# 8. Exploded case: othliab / group_id=16373
# ---------------------------------------------------------------------------

print("\n" + "=" * 72)
print("8. Exploded case: othliab / group_id=16373")
print("=" * 72)

rec_bad = next((r for r in records if r.line == "othliab" and r.group_id == 16373), None)
if rec_bad is None:
    print("othliab/16373 not found in records")
else:
    bad_tri = rec_bad.train_triangles["paid"]
    bad_prem = load_exposure_triangle("othliab", 16373)

    print(f"Company: {rec_bad.company}")
    print(f"Triangle:")
    print(bad_tri)

    # Check for extreme cell values / near-zero incrementals
    inc_vals = bad_tri.cum_to_incr()
    flat = np.asarray(inc_vals.values, dtype=float).flatten()
    finite = flat[np.isfinite(flat)]
    print(f"\nIncremental stats: min={finite.min():.1f}, max={finite.max():.1f}, mean={finite.mean():.1f}")
    print(f"Near-zero cells (<1.0): {(np.abs(finite) < 1.0).sum()}")
    print(f"Zero cells: {(finite == 0.0).sum()}")

    # Try quick fit to see what explodes
    print("\nAttempting quick GLM fit (draws=200, tune=100, chains=1)...")
    model_bad = BayesianChainLadderGLM(
        formula="incremental ~ 1 + C(origin) + bs(dev_idx, df=4)",
        family="gamma",
        link="log",
        exposure="net_earned_premium",
        draws=200,
        tune=100,
        chains=1,
        target_accept=0.95,
        random_seed=42,
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_bad.fit(bad_tri, exposure_triangle=bad_prem)

        rp_bad = model_bad.reserves_posterior_
        total_ibnr_bad = rp_bad.sum(dim="origin").values.flatten()
        total_ibnr_bad = total_ibnr_bad[np.isfinite(total_ibnr_bad)]
        print(f"Bad model total ibnr samples (finite): {len(total_ibnr_bad)}")
        if len(total_ibnr_bad) > 0:
            print(f"  Max sample: {total_ibnr_bad.max():.4e}")
            print(f"  Mean sample: {total_ibnr_bad.mean():.4e}")
            print(f"  Median sample: {np.median(total_ibnr_bad):.4e}")
            print(f"  95th pctile: {np.percentile(total_ibnr_bad, 95):.4e}")

        # Rhat diagnostics
        import arviz as az
        rhat_vals = az.rhat(model_bad.idata).to_array()
        rhat_max = float(rhat_vals.max())
        print(f"  Max Rhat: {rhat_max:.4f}")
        print(f"  Divergences: {int(model_bad.idata.sample_stats.diverging.sum())}")

        paid_bad = float(model_bad.ultimate_["paid_to_date"].sum())
        print(f"\nBad model total posterior mean ultimate: {total_ibnr_bad.mean() + paid_bad:.4e}")
        print(f"Bad model total posterior median ultimate: {np.median(total_ibnr_bad) + paid_bad:.4e}")
    except Exception as e:
        print(f"Fit failed: {type(e).__name__}: {e}")

print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
print(f"""
Diagnostic triangle: {LINE}/{GROUP_ID} ({rec.company})
  Mack total ultimate:            {mack_total:>12,.1f}
  Det CL total ultimate:          {cl_total:>12,.1f}
  GLM posterior mean total:       {post_mean_total:>12,.1f}   (ratio: {post_mean_total/mack_total:.4f})
  GLM posterior median total:     {post_median_total:>12,.1f}   (ratio: {post_median_total/mack_total:.4f})
  Counterfactual exp(E[eta]):     {cf_total:>12,.1f}   (ratio: {cf_total/mack_total:.4f})

  Jensen bias factor (mean/cf):   {post_mean_total/cf_total:.4f}
  Posterior sigma of eta (mean):  {eta_post_std.mean():.4f}
  Expected Jensen bias (e^(s^2/2)): {np.exp(eta_post_std.mean()**2/2):.4f}

  >>> The backtest uses np.nanmean(total_ult_samples) which is the posterior
  >>> MEAN of total ultimate. For gamma+log, this systematically exceeds
  >>> the ODP MLE (Mack) by exp(sigma_eta^2/2) due to Jensen's inequality.
  >>> Switching to np.median would give ratio ~ {post_median_total/mack_total:.4f}.
""")
