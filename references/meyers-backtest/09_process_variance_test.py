"""09_process_variance_test.py — Test process-variance-inclusive GLM reserves.

Fits BayesianChainLadderGLM (M1_cat spec: gamma+log, C(origin)+C(dev)) on
three Meyers back-test triangles with include_process_variance=False and
include_process_variance=True, then compares implied_pctl and spread
against Mack and the actual ultimate.

Triangles tested:
  1. Celina Mut Grp   ppauto/353   (implied_pctl=0.000 with param-only)
  2. Amerisafe Grp    ppauto/6807  (implied_pctl=0.000 with param-only)
  3. NC Farm Bureau   othliab/3240 (implied_pctl=0.000 with param-only)

Run with:
    cd references/meyers-backtest
    uv run python 09_process_variance_test.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import chainladder as cl
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import load_exposure_triangle

DIVIDER = "=" * 72


def section(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# ---------------------------------------------------------------------------
# Triangle specs
# ---------------------------------------------------------------------------

TRIANGLES = [
    {"name": "Celina Mut Grp",   "line": "ppauto",   "group_id": 353},
    {"name": "Amerisafe Grp",    "line": "ppauto",   "group_id": 6807},
    {"name": "NC Farm Bureau",   "line": "othliab",  "group_id": 3240},
]

# GLM spec (M1_cat matching the back-test wrapper)
FORMULA  = "incremental ~ 1 + C(origin) + C(dev)"
FAMILY   = "gamma"
LINK     = "log"
EXPOSURE = "net_earned_premium"
DRAWS    = 1000
TUNE     = 1000
CHAINS   = 2
TARGET_ACCEPT = 0.95
RANDOM_SEED   = 22


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mack_total(tri: cl.Triangle) -> tuple[float, float]:
    """Return (total_mack_ultimate, total_mack_se)."""
    mack = cl.MackChainladder().fit(tri)
    ult = float(np.nansum(np.asarray(mack.ultimate_.values, dtype=float)))
    se  = float(np.asarray(mack.total_mack_std_err_.values, dtype=float).squeeze())
    return ult, se


def latest_obs(tri: cl.Triangle) -> float:
    """Sum of latest diagonal (paid to date)."""
    diag = np.asarray(tri.latest_diagonal.values, dtype=float).squeeze()
    return float(np.nansum(diag))


def empirical_pctl(samples: np.ndarray, actual: float) -> float:
    finite = samples[np.isfinite(samples)]
    return float(np.mean(finite <= actual)) if finite.size > 0 else float("nan")


def fit_glm(tri, prem_tri, include_pv: bool) -> tuple[np.ndarray, np.ndarray]:
    """Fit GLM and return (total_ibnr_samples, total_ult_samples)."""
    from bayesianchainladder import BayesianChainLadderGLM

    model = BayesianChainLadderGLM(
        formula=FORMULA,
        family=FAMILY,
        link=LINK,
        exposure=EXPOSURE,
        priors=None,
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        target_accept=TARGET_ACCEPT,
        random_seed=RANDOM_SEED,
        include_process_variance=include_pv,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(tri, exposure_triangle=prem_tri)

    reserves = model.reserves_posterior_
    ibnr_samples = np.asarray(reserves.sum(dim="origin").values, dtype=float)
    paid = latest_obs(tri)
    ult_samples = ibnr_samples + paid
    return ibnr_samples, ult_samples


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

import reservetestr as rt

recs_all = rt.build_triangle_records()
rec_map = {(r.line, r.group_id): r for r in recs_all}

all_rows = []

for spec in TRIANGLES:
    name     = spec["name"]
    line     = spec["line"]
    gid      = spec["group_id"]
    rec      = rec_map[(line, gid)]
    tri      = rec.train_triangles["paid"]
    actual   = float(rec.actual_ultimates["paid"])
    prem_tri = load_exposure_triangle(line, gid)
    paid     = latest_obs(tri)

    section(f"{name}  ({line}/{gid})")
    print(f"Actual ultimate:   {actual:>14,.1f}")
    print(f"Latest paid:       {paid:>14,.1f}")

    # ---- Mack ---------------------------------------------------------------
    mack_ult, mack_se = mack_total(tri)
    mack_cv = mack_se / mack_ult if mack_ult > 0 else float("nan")
    print(f"\nMack:")
    print(f"  ultimate:  {mack_ult:>14,.1f}  SE={mack_se:>12,.1f}  CV={mack_cv:.4f}")
    print(f"  Mack/Actual: {mack_ult/actual:.4f}")

    for include_pv in (False, True):
        label = "process_variance=ON " if include_pv else "process_variance=OFF"
        print(f"\nGLM ({label})  — fitting ...")
        try:
            ibnr_s, ult_s = fit_glm(tri, prem_tri, include_pv)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            all_rows.append({
                "name": name, "line": line, "group_id": gid,
                "include_pv": include_pv,
                "actual": actual, "mack_ult": mack_ult,
                "glm_median": float("nan"), "glm_mean": float("nan"),
                "glm_sd": float("nan"), "cv_unpaid": float("nan"),
                "implied_pctl": float("nan"), "n_samples": 0,
            })
            continue

        finite_ult  = ult_s[np.isfinite(ult_s)]
        finite_ibnr = ibnr_s[np.isfinite(ibnr_s)]

        glm_median  = float(np.median(finite_ult))
        glm_mean    = float(np.mean(finite_ult))
        glm_sd      = float(np.std(finite_ult, ddof=1)) if finite_ult.size > 1 else float("nan")
        ibnr_median = float(np.median(finite_ibnr))
        cv_unpaid   = glm_sd / ibnr_median if ibnr_median > 0 else float("nan")
        impl_pctl   = empirical_pctl(ult_s, actual)

        print(f"  n_samples:     {finite_ult.size}")
        print(f"  median ult:    {glm_median:>14,.1f}")
        print(f"  mean ult:      {glm_mean:>14,.1f}")
        print(f"  SD:            {glm_sd:>14,.1f}")
        print(f"  CV unpaid:     {cv_unpaid:.4f}")
        print(f"  p5 / p50 / p95: {np.percentile(finite_ult,5):>12,.1f} / "
              f"{np.percentile(finite_ult,50):>12,.1f} / "
              f"{np.percentile(finite_ult,95):>12,.1f}")
        print(f"  implied_pctl:  {impl_pctl:.6f}  "
              f"({'IMPROVEMENT' if include_pv else 'baseline'})")
        print(f"  GLM/Actual:    {glm_median/actual:.4f}")
        print(f"  GLM/Mack:      {glm_median/mack_ult:.4f}")

        all_rows.append({
            "name": name, "line": line, "group_id": gid,
            "include_pv": include_pv,
            "actual": actual, "mack_ult": mack_ult, "mack_se": mack_se,
            "glm_median": glm_median, "glm_mean": glm_mean,
            "glm_sd": glm_sd, "cv_unpaid": cv_unpaid,
            "implied_pctl": impl_pctl, "n_samples": int(finite_ult.size),
        })

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

section("SUMMARY TABLE — implied_pctl before / after process variance")

df = pd.DataFrame(all_rows)

print(f"\n{'Company':<20s}  {'Line':8s}  {'PV':5s}  {'Actual':>12s}  "
      f"{'Mack Ult':>12s}  {'GLM Med':>12s}  {'GLM SD':>10s}  {'CV Unp':>8s}  "
      f"{'pctl':>8s}")
print("-" * 105)
for _, row in df.iterrows():
    pv_str   = "ON " if row["include_pv"] else "OFF"
    glm_med  = row.get("glm_median", float("nan"))
    glm_sd   = row.get("glm_sd", float("nan"))
    cv_u     = row.get("cv_unpaid", float("nan"))
    impl_p   = row.get("implied_pctl", float("nan"))
    print(f"{row['name']:<20s}  {row['line']:8s}  {pv_str:5s}  "
          f"{row['actual']:>12,.0f}  {row['mack_ult']:>12,.0f}  "
          f"{glm_med:>12,.0f}  {glm_sd:>10,.0f}  {cv_u:>8.4f}  "
          f"{impl_p:>8.4f}")

print("\n")
section("DELTA — how much does process variance improve implied_pctl?")

for name_key in df["name"].unique():
    sub = df[df["name"] == name_key].set_index("include_pv")
    if False in sub.index and True in sub.index:
        before = sub.loc[False, "implied_pctl"]
        after  = sub.loc[True,  "implied_pctl"]
        delta  = after - before
        print(f"  {name_key:<22s}  pctl OFF={before:.4f}  ON={after:.4f}  "
              f"delta={delta:+.4f}  "
              f"({'IMPROVED' if delta > 0.02 else 'minimal' if delta >= 0 else 'WORSE'})")

section("VERDICT")
pctls_off = df[~df["include_pv"]]["implied_pctl"].dropna()
pctls_on  = df[ df["include_pv"]]["implied_pctl"].dropna()
mean_off  = float(pctls_off.mean()) if len(pctls_off) > 0 else float("nan")
mean_on   = float(pctls_on.mean())  if len(pctls_on)  > 0 else float("nan")
print(f"\n  Mean implied_pctl  OFF: {mean_off:.4f}")
print(f"  Mean implied_pctl  ON:  {mean_on:.4f}")
print(f"  Delta:                  {mean_on - mean_off:+.4f}")
if mean_on - mean_off > 0.05:
    verdict = "PROCESS VARIANCE HELPS — proceed with full back-test re-run."
elif mean_on - mean_off > 0.0:
    verdict = "Modest improvement — process variance narrows the calibration gap but does not fix it."
else:
    verdict = "Process variance does NOT help (or hurts). Root cause is bias (over-prediction), not missing variance."
print(f"\n  Verdict: {verdict}")
