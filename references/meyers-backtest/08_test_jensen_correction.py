"""08_test_jensen_correction.py — Jensen-corrected point estimate for GLM reserves.

Investigates whether a Jensen-corrected aggregate (sum of per-cell exp(E[log mu])
or sum of per-cell median) brings the BayesianChainLadderGLM ultimate in line
with Mack on the known-bad Celina Mut Grp triangle (ppauto/353).

Four aggregates compared:
  A — mean(sum of μ samples)          : Jensen-inflated posterior mean
  B — median(sum of μ samples)        : current wrapper output
  C — sum of per-cell median(μ)       : Jensen-corrected per cell, then sum
  D — sum of exp(per-cell mean(log μ)): MLE-equivalent (lognormal median per cell)

Run with:
    cd references/meyers-backtest
    uv run python 08_test_jensen_correction.py
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
# Constants
# ---------------------------------------------------------------------------

MACK_ULTIMATE = 129_779.0   # From deep-dive diagnostic (commit 8695596)
ACTUAL_ULTIMATE = 125_467.0

# ---------------------------------------------------------------------------
# Step 1: Load triangle and fit GLM (no offset, pure categorical spec)
# ---------------------------------------------------------------------------
section("STEP 1 — Load Celina Mut Grp triangle and fit GLM")

import reservetestr as rt
import chainladder as cl
from _common import load_exposure_triangle
from bayesianchainladder import BayesianChainLadderGLM

recs = rt.build_triangle_records()
r = [rec for rec in recs if rec.line == "ppauto" and rec.group_id == 353][0]
print(f"Company: {r.company}  |  line: {r.line}  |  group_id: {r.group_id}")

tri = r.train_triangles["paid"]
actual_ultimate = float(r.actual_ultimates["paid"])

print(f"Actual ultimate (from records):  {actual_ultimate:>12,.1f}")
print(f"Reference Mack ultimate:         {MACK_ULTIMATE:>12,.1f}")

# Mack reference from this run
mack = cl.MackChainladder().fit(tri)
mack_ult = np.asarray(mack.ultimate_.values, dtype=float).squeeze()
mack_total = float(np.nansum(mack_ult))
total_latest = float(np.nansum(np.asarray(mack.latest_diagonal.values, dtype=float).squeeze()))
mack_total_ibnr = mack_total - total_latest
print(f"Mack ultimate (this run):        {mack_total:>12,.1f}")
print(f"Total latest observed:           {total_latest:>12,.1f}")
print(f"Mack IBNR:                       {mack_total_ibnr:>12,.1f}")

# Fit GLM — NO offset (pure categorical, as specified in the task)
print("\nFitting BayesianChainLadderGLM (gamma+log, C(origin)+C(dev), NO offset)...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    glm = BayesianChainLadderGLM(
        formula="incremental ~ 1 + C(origin) + C(dev)",
        family="gamma",
        link="log",
        exposure=None,       # no offset — pure categorical per task spec
        priors=None,
        draws=2000,
        tune=1000,
        chains=2,
        target_accept=0.95,
        random_seed=22,
    )
    glm.fit(tri)

print("GLM fit complete.")

# ---------------------------------------------------------------------------
# Step 2: Extract posterior of linear predictor (log μ) per future cell
# ---------------------------------------------------------------------------
section("STEP 2 — Extract posterior of μ per future cell")

response_name = glm.model_.response_component.response.name
mean_var = f"{response_name}_mean"
if mean_var not in glm.idata.posterior:
    mean_var = "mu" if "mu" in glm.idata.posterior else response_name

print(f"Using posterior variable: '{mean_var}'")
future_predictions = glm.idata.posterior[mean_var]
print(f"Posterior shape: {future_predictions.values.shape}  (chain, draw, obs)")

n_future = len(glm.future_data_)
n_total_obs = future_predictions.shape[-1]
future_start = n_total_obs - n_future

# future_pred_flat: (n_samples, n_future_cells) where n_samples = n_chain * n_draw
future_pred_flat = future_predictions.values.reshape(-1, n_total_obs)[:, future_start:]
n_samples = future_pred_flat.shape[0]
print(f"Future cells: {n_future}, posterior samples: {n_samples}")

# Confirm shapes
assert future_pred_flat.shape == (n_samples, n_future), (
    f"Shape mismatch: {future_pred_flat.shape} vs ({n_samples}, {n_future})"
)

# ---------------------------------------------------------------------------
# Step 3: Compute FOUR alternative aggregates
# ---------------------------------------------------------------------------
section("STEP 3 — Four alternative IBNR aggregates")

# For each future cell, posterior of mu (shape: n_samples, n_future)
# Finite-only mask per cell
valid = np.isfinite(future_pred_flat) & (future_pred_flat > 0)

# A — mean(sum of μ samples)
total_ibnr_per_sample = np.nansum(future_pred_flat, axis=1)  # (n_samples,)
aggregate_A = float(np.nanmean(total_ibnr_per_sample))

# B — median(sum of μ samples)   [current wrapper behavior]
aggregate_B = float(np.nanmedian(total_ibnr_per_sample))

# C — sum of per-cell median(μ)
per_cell_median = np.array([
    float(np.nanmedian(future_pred_flat[:, j][valid[:, j]]))
    if valid[:, j].any() else 0.0
    for j in range(n_future)
])
aggregate_C = float(np.nansum(per_cell_median))

# D — sum of exp(per-cell mean(log μ))   [lognormal MLE-equivalent point estimate]
per_cell_exp_mean_log = np.array([
    float(np.exp(np.nanmean(np.log(future_pred_flat[:, j][valid[:, j]]))))
    if valid[:, j].any() else 0.0
    for j in range(n_future)
])
aggregate_D = float(np.nansum(per_cell_exp_mean_log))

# Ultimates (add latest observed)
ult_A = aggregate_A + total_latest
ult_B = aggregate_B + total_latest
ult_C = aggregate_C + total_latest
ult_D = aggregate_D + total_latest

print(f"\n{'Aggregate':45s}  {'IBNR':>12s}  {'Ultimate':>12s}  {'Ult/Mack':>10s}  {'Ult/Actual':>10s}")
print("-" * 95)
print(f"{'A — mean(sum of μ samples)  [Jensen-inflated]':45s}  {aggregate_A:12,.1f}  {ult_A:12,.1f}  {ult_A/mack_total:10.4f}  {ult_A/actual_ultimate:10.4f}")
print(f"{'B — median(sum of μ samples)  [current wrapper]':45s}  {aggregate_B:12,.1f}  {ult_B:12,.1f}  {ult_B/mack_total:10.4f}  {ult_B/actual_ultimate:10.4f}")
print(f"{'C — sum of per-cell median(μ)  [Jensen-corrected]':45s}  {aggregate_C:12,.1f}  {ult_C:12,.1f}  {ult_C/mack_total:10.4f}  {ult_C/actual_ultimate:10.4f}")
print(f"{'D — sum of exp(E[log μ])  [MLE-equivalent]':45s}  {aggregate_D:12,.1f}  {ult_D:12,.1f}  {ult_D/mack_total:10.4f}  {ult_D/actual_ultimate:10.4f}")
print(f"{'Mack  (reference)':45s}  {mack_total_ibnr:12,.1f}  {mack_total:12,.1f}  {'1.0000':>10s}  {mack_total/actual_ultimate:10.4f}")
print(f"{'Actual  (ground truth)':45s}  {actual_ultimate - total_latest:12,.1f}  {actual_ultimate:12,.1f}  {actual_ultimate/mack_total:10.4f}  {'1.0000':>10s}")

# Which aggregates are within 5% of Mack?
print(f"\nWithin 5% of Mack?")
for label, ult in [("A", ult_A), ("B", ult_B), ("C", ult_C), ("D", ult_D)]:
    pct_diff = abs(ult - mack_total) / mack_total * 100
    status = "YES" if pct_diff <= 5.0 else "NO"
    print(f"  {label}: |{ult:,.1f} - {mack_total:,.1f}| / {mack_total:,.1f} = {pct_diff:.2f}%  -> {status}")

# ---------------------------------------------------------------------------
# Step 4: Cell-level decomposition
# ---------------------------------------------------------------------------
section("STEP 4 — Per-cell decomposition")

origins = glm.future_data_["origin"].values
devs = glm.future_data_["dev"].values

# Per-origin Mack IBNR
mack_latest = np.asarray(mack.latest_diagonal.values, dtype=float).squeeze()
origins_str = [str(o).split("T")[0] for o in tri.origin]
mack_ibnr_per_origin = {
    str(tri.origin[i]).split("T")[0]: float(mack_ult[i]) - float(mack_latest[i])
    for i in range(len(mack_ult))
}

print(f"\n{'#':>4s}  {'Origin':>8s}  {'Dev':>4s}  {'E[mu]':>10s}  {'median(mu)':>12s}  "
      f"{'exp(E[lnμ])':>12s}  {'Jensen Ratio':>13s}  {'Corr Ratio':>11s}")
print("-" * 90)

for idx in range(n_future):
    mu_s = future_pred_flat[:, idx]
    mu_s = mu_s[np.isfinite(mu_s) & (mu_s > 0)]
    if len(mu_s) < 5:
        continue
    e_mu = float(np.mean(mu_s))
    med_mu = float(np.median(mu_s))
    exp_e_log_mu = float(np.exp(np.mean(np.log(mu_s))))
    jensen_ratio = e_mu / exp_e_log_mu if exp_e_log_mu > 0 else float("nan")
    corr_ratio = med_mu / exp_e_log_mu if exp_e_log_mu > 0 else float("nan")
    print(f"  {idx:4d}  {str(origins[idx]):>8s}  {str(devs[idx]):>4s}  "
          f"{e_mu:10.2f}  {med_mu:12.2f}  {exp_e_log_mu:12.2f}  "
          f"{jensen_ratio:13.4f}  {corr_ratio:11.4f}")

# Summary by origin
print(f"\n{'--- Per-origin IBNR decomposition ---':}")
print(f"\n{'Origin':>8s}  {'Mack IBNR':>12s}  {'A (E[sum])':>12s}  {'B (med(sum))':>14s}  "
      f"{'C (sum med)':>12s}  {'D (sum exp)':>12s}  {'D/Mack':>8s}")
print("-" * 90)

for orig in sorted(glm.future_data_["origin"].unique()):
    orig_str = str(orig).split("T")[0]
    mask = glm.future_data_["origin"].values == orig
    idx_arr = np.where(mask)[0]

    mu_mat = future_pred_flat[:, idx_arr]  # (n_samples, n_cells_for_origin)

    ibnr_A = float(np.nanmean(np.nansum(mu_mat, axis=1)))
    ibnr_B = float(np.nanmedian(np.nansum(mu_mat, axis=1)))
    ibnr_C = float(np.nansum([
        np.nanmedian(mu_mat[:, k][np.isfinite(mu_mat[:, k]) & (mu_mat[:, k] > 0)])
        if (np.isfinite(mu_mat[:, k]) & (mu_mat[:, k] > 0)).any() else 0.0
        for k in range(mu_mat.shape[1])
    ]))
    ibnr_D = float(np.nansum([
        np.exp(np.nanmean(np.log(mu_mat[:, k][np.isfinite(mu_mat[:, k]) & (mu_mat[:, k] > 0)])))
        if (np.isfinite(mu_mat[:, k]) & (mu_mat[:, k] > 0)).any() else 0.0
        for k in range(mu_mat.shape[1])
    ]))
    mack_ib = mack_ibnr_per_origin.get(orig_str, float("nan"))

    print(f"  {orig_str:>8s}  {mack_ib:12.1f}  {ibnr_A:12.1f}  {ibnr_B:14.1f}  "
          f"{ibnr_C:12.1f}  {ibnr_D:12.1f}  {ibnr_D/mack_ib if mack_ib>0 else float('nan'):8.4f}")

print(f"  {'TOTAL':>8s}  {mack_total_ibnr:12.1f}  {aggregate_A:12.1f}  {aggregate_B:14.1f}  "
      f"{aggregate_C:12.1f}  {aggregate_D:12.1f}  {aggregate_D/mack_total_ibnr:8.4f}")

# ---------------------------------------------------------------------------
# Step 5: Verdict
# ---------------------------------------------------------------------------
section("STEP 5 — Verdict: which aggregate matches Mack within 5%?")

verdict = {}
for label, ult in [("A", ult_A), ("B", ult_B), ("C", ult_C), ("D", ult_D)]:
    pct = (ult - mack_total) / mack_total * 100
    verdict[label] = pct

# Find best match
best = min(verdict, key=lambda k: abs(verdict[k]))
print(f"\nPct deviation from Mack:")
for label, pct in verdict.items():
    flag = " <-- BEST" if label == best else ""
    within5 = " [WITHIN 5%]" if abs(pct) <= 5.0 else ""
    print(f"  {label}: {pct:+.2f}%{within5}{flag}")

print(f"\nConclusion:")
within5_labels = [k for k, v in verdict.items() if abs(v) <= 5.0]
if within5_labels:
    print(f"  Aggregates {within5_labels} are within 5% of Mack.")
    print(f"  Best match: Aggregate {best} ({verdict[best]:+.2f}% from Mack).")
else:
    print(f"  No aggregate is within 5% of Mack on this triangle.")
    print(f"  Best available: Aggregate {best} ({verdict[best]:+.2f}% from Mack).")
    print(f"  The bias likely comes from prior influence, not Jensen correction alone.")

print(f"\n  Jensen inflation (A/D ratio):    {ult_A/ult_D:.4f}x")
print(f"  Median/MLE ratio (B/D):          {ult_B/ult_D:.4f}x")
print(f"  Per-cell correction (C/D):       {ult_C/ult_D:.4f}x")

# ---------------------------------------------------------------------------
# Step 6: Smoke tests on 2 more triangles (only if C or D within 5% of Mack)
# ---------------------------------------------------------------------------
section("STEP 6 — Smoke tests: Amerisafe (ppauto/6807) and NC Farm Bureau (othliab/3240)")


def _four_aggregates(glm_model, tri_for_mack):
    """Return (ult_A, ult_B, ult_C, ult_D, latest_obs, mack_ult, actual_ult) for a fitted model."""
    import chainladder as cl

    response_name = glm_model.model_.response_component.response.name
    m_var = f"{response_name}_mean"
    if m_var not in glm_model.idata.posterior:
        m_var = "mu" if "mu" in glm_model.idata.posterior else response_name

    fp = glm_model.idata.posterior[m_var]
    n_fut = len(glm_model.future_data_)
    n_tot = fp.shape[-1]
    fp_flat = fp.values.reshape(-1, n_tot)[:, (n_tot - n_fut):]

    per_sample = np.nansum(fp_flat, axis=1)
    agg_A = float(np.nanmean(per_sample))
    agg_B = float(np.nanmedian(per_sample))

    valid_mask = np.isfinite(fp_flat) & (fp_flat > 0)
    agg_C = float(np.nansum([
        float(np.nanmedian(fp_flat[:, j][valid_mask[:, j]]))
        if valid_mask[:, j].any() else 0.0
        for j in range(n_fut)
    ]))
    agg_D = float(np.nansum([
        float(np.exp(np.nanmean(np.log(fp_flat[:, j][valid_mask[:, j]]))))
        if valid_mask[:, j].any() else 0.0
        for j in range(n_fut)
    ]))

    mack_model = cl.MackChainladder().fit(tri_for_mack)
    mack_total = float(np.nansum(np.asarray(mack_model.ultimate_.values, dtype=float).squeeze()))
    latest_obs = float(np.nansum(np.asarray(mack_model.latest_diagonal.values, dtype=float).squeeze()))

    return (
        agg_A + latest_obs,
        agg_B + latest_obs,
        agg_C + latest_obs,
        agg_D + latest_obs,
        latest_obs,
        mack_total,
    )


def run_smoke_test(line, group_id, label):
    """Fit GLM and print 4-aggregate comparison for a single triangle."""
    print(f"\n--- {label} (line={line!r}, group_id={group_id}) ---")
    recs_all = rt.build_triangle_records()
    matches = [rec for rec in recs_all if rec.line == line and rec.group_id == group_id]
    if not matches:
        print(f"  No record found for line={line!r}, group_id={group_id}")
        return

    rec = matches[0]
    tri_ = rec.train_triangles["paid"]
    actual_ult_ = float(rec.actual_ultimates["paid"])

    print(f"  Company: {rec.company}")
    print(f"  Actual ultimate: {actual_ult_:,.1f}")

    print("  Fitting GLM (no offset, gamma+log, C(origin)+C(dev)) ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        glm_ = BayesianChainLadderGLM(
            formula="incremental ~ 1 + C(origin) + C(dev)",
            family="gamma",
            link="log",
            exposure=None,
            priors=None,
            draws=2000,
            tune=1000,
            chains=2,
            target_accept=0.95,
            random_seed=42,
        ).fit(tri_)

    a_, b_, c_, d_, latest_, mack_ = _four_aggregates(glm_, tri_)
    mack_ibnr_ = mack_ - latest_
    actual_ibnr_ = actual_ult_ - latest_

    print(f"\n  {'Aggregate':45s}  {'IBNR':>12s}  {'Ultimate':>12s}  {'Ult/Mack':>10s}  {'Ult/Actual':>10s}")
    print("  " + "-" * 93)
    for lbl_, ult_ in [
        ("A — mean(sum μ)  [Jensen-inflated]", a_),
        ("B — median(sum μ)  [current wrapper]", b_),
        ("C — sum of per-cell median(μ)", c_),
        ("D — sum of exp(E[log μ])  [MLE-equiv]", d_),
        ("Mack  (reference)", mack_),
        ("Actual  (ground truth)", actual_ult_),
    ]:
        ibnr_ = ult_ - latest_
        print(f"  {lbl_:45s}  {ibnr_:12,.1f}  {ult_:12,.1f}  {ult_/mack_:10.4f}  {ult_/actual_ult_:10.4f}")

    print(f"\n  Within 5% of Mack?")
    for lbl_, ult_ in [("A", a_), ("B", b_), ("C", c_), ("D", d_)]:
        pct_ = abs(ult_ - mack_) / mack_ * 100
        print(f"    {lbl_}: {pct_:.2f}%  -> {'YES' if pct_ <= 5.0 else 'NO'}")


run_smoke_test("ppauto", 6807, "Amerisafe")
run_smoke_test("othliab", 3240, "NC Farm Bureau")

section("DONE — all three triangles tested")
