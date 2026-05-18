"""12_structured_priors.py — Test structured (not strongly-informed) priors.

Key question from user after commit 0bfb5c3 (sigma=0.3 uniformly on C(dev)):
  "Do the C(dev) prior SDs vary? Later devs should have less absolute
   uncertainty since the projection period is shorter."

We test three prior configurations on the same CL-informed prior CENTERS:

  Approach A — Tapered SDs on C(dev):
      dev=24:   sigma=0.6   (high uncertainty about early decay rate)
      dev=36:   sigma=0.6
      dev=48:   sigma=0.5
      dev=60:   sigma=0.5
      dev=72:   sigma=0.4
      dev=84:   sigma=0.3
      dev=96:   sigma=0.25
      dev=108:  sigma=0.20
      dev=120:  sigma=0.15  (tightest — close to fully developed)
      C(origin): sigma=0.5 uniform (wider than σ=0.3 in commit 0bfb5c3)

  Approach B — Wider uniform SDs (σ=0.5):
      All C(dev) and C(origin) contrasts get sigma=0.5.
      Same CL centers.  Tests whether data dominates the prior.

  Approach C — Even wider uniform SDs (σ=1.0 with CL centers):
      All C(dev) and C(origin) contrasts get sigma=1.0.
      This tests whether the CENTERS are what's doing the work vs the tight SD.

The prior centers in all cases come from the chain-ladder (same as in
11_mack_informed_priors.py's build_mack_priors()).

Reference baselines (from commit 0bfb5c3 / prior analysis):
  Celina (ppauto/353):
    Mack = 129,779    Actual = 125,467
    Default priors (sigma=1 uninformed centers): 184,629
    Mack-informed sigma=0.3 (prior run):        ~129,700 (within 1% of Mack)

Run with:
    cd references/meyers-backtest
    uv run python 12_structured_priors.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import bambi as bmb
import chainladder as cl
import numpy as np
import pandas as pd
import reservetestr as rt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bayesianchainladder import BayesianChainLadderGLM

DIVIDER = "=" * 72


def section(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# ---------------------------------------------------------------------------
# Known baselines (from 11_mack_informed_priors.py / prior analysis)
# ---------------------------------------------------------------------------

BASELINES = {
    # (line, group_id): {"company", "mack", "actual", "default_sigma1", "mack_sigma03"}
    ("ppauto", 353): {
        "company": "Celina Mut Grp",
        "mack": 129_779.0,
        "actual": 125_467.0,
        "default_sigma1": 184_629.0,   # Normal(0, sigma=1) prior centers
        "mack_sigma03": None,           # filled in during run
    },
    ("ppauto", 6807): {
        "company": "Amerisafe Grp",
        "mack": 11_845.0,
        "actual": 11_771.0,
        "default_sigma1": None,
        "mack_sigma03": None,
    },
    ("othliab", 3240): {
        "company": "NC Farm Bureau Ins Grp",
        "mack": 13_606.0,
        "actual": 14_144.0,
        "default_sigma1": None,
        "mack_sigma03": None,
    },
}

# Tapered SDs for C(dev) — key = dev period, value = sigma
# These taper from 0.6 at early devs to 0.15 at the latest dev
TAPERED_SD_MAP = {
    24: 0.6,
    36: 0.6,
    48: 0.5,
    60: 0.5,
    72: 0.4,
    84: 0.3,
    96: 0.25,
    108: 0.20,
    120: 0.15,
}


# ---------------------------------------------------------------------------
# Build prior dicts for the three approaches
# ---------------------------------------------------------------------------

def build_mack_priors(triangle: "cl.Triangle", sigma: float = 0.3) -> dict:
    """Construct Bambi priors from CL chain-ladder (same helper as script 11)."""
    fitted_cl = cl.Chainladder().fit(triangle)
    ult_arr = np.asarray(fitted_cl.ultimate_.to_frame().values, dtype=float).flatten()
    cdf_arr = np.asarray(fitted_cl.cdf_.to_frame().values, dtype=float).flatten()
    devs = list(triangle.to_frame().columns)
    n_devs = len(devs)
    cdf_for_devs = cdf_arr[:n_devs]
    pct_dev = 1.0 / cdf_for_devs
    incr_fracs = np.diff(np.concatenate([[0.0], pct_dev]))
    incr_fracs = np.maximum(incr_fracs, 1e-8)
    ref_ult = ult_arr[0]
    ref_frac = incr_fracs[0]
    origin_mus = np.log(ult_arr[1:]) - np.log(ref_ult)
    dev_mus = np.log(incr_fracs[1:]) - np.log(ref_frac)
    intercept_mu = float(np.log(ref_ult * ref_frac))
    cum_df = triangle.to_frame()
    dev_labels = list(cum_df.columns)  # e.g., [12, 24, 36, ...]
    return {
        "origin_mus": origin_mus,
        "dev_mus": dev_mus,
        "intercept_mu": intercept_mu,
        "origin_ults": ult_arr,
        "incr_fracs": incr_fracs,
        "dev_labels": dev_labels,  # full dev list including ref dev=12
    }


def make_priors_approach_a(info: dict) -> dict:
    """Approach A: tapered SDs on C(dev), wider σ=0.5 on C(origin)."""
    dev_labels = info["dev_labels"]
    dev_contrasts = dev_labels[1:]  # dev=24..120 (relative to dev=12)
    dev_sigmas = np.array([
        TAPERED_SD_MAP.get(int(d), 0.5) for d in dev_contrasts
    ], dtype=float)
    n_origins = len(info["origin_mus"])
    return {
        "Intercept": bmb.Prior("Normal", mu=info["intercept_mu"], sigma=2.0),
        "C(origin)": bmb.Prior(
            "Normal",
            mu=info["origin_mus"],
            sigma=np.full(n_origins, 0.5, dtype=float),
        ),
        "C(dev)": bmb.Prior(
            "Normal",
            mu=info["dev_mus"],
            sigma=dev_sigmas,
        ),
    }


def make_priors_approach_b(info: dict, sigma: float = 0.5) -> dict:
    """Approach B: uniform σ=0.5 on all contrasts (wider than σ=0.3)."""
    n_origins = len(info["origin_mus"])
    n_devs = len(info["dev_mus"])
    return {
        "Intercept": bmb.Prior("Normal", mu=info["intercept_mu"], sigma=2.0),
        "C(origin)": bmb.Prior(
            "Normal",
            mu=info["origin_mus"],
            sigma=np.full(n_origins, sigma, dtype=float),
        ),
        "C(dev)": bmb.Prior(
            "Normal",
            mu=info["dev_mus"],
            sigma=np.full(n_devs, sigma, dtype=float),
        ),
    }


def make_priors_approach_c(info: dict) -> dict:
    """Approach C: uniform σ=1.0 on all contrasts (very wide — CL centers only)."""
    return make_priors_approach_b(info, sigma=1.0)


# ---------------------------------------------------------------------------
# Fit helper
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
    total_ibnr = np.asarray(reserves_post.sum(dim="origin").values, dtype=float)
    total_ult = total_ibnr + latest_observed
    finite_ult = total_ult[np.isfinite(total_ult)]

    return {
        "label": label,
        "ult_median": float(np.median(finite_ult)),
        "ult_mean": float(np.mean(finite_ult)),
        "ult_sd": float(np.std(finite_ult, ddof=1)),
        "implied_pctl": float(np.mean(finite_ult <= actual_ultimate)),
    }


# ---------------------------------------------------------------------------
# Main loop: 3 triangles × 4 approaches
# ---------------------------------------------------------------------------

section("12 — STRUCTURED PRIORS: 3 triangles × 4 approaches")

TARGETS = [("ppauto", 353), ("ppauto", 6807), ("othliab", 3240)]

all_recs = rt.build_triangle_records()

# Approach labels in display order
APPROACH_LABELS = [
    "Default (σ=1, uninformed)",
    "Mack-informed σ=0.3 (script 11)",
    "A: tapered dev SDs (0.15–0.6)",
    "B: uniform σ=0.5",
    "C: uniform σ=1.0 + CL centers",
]

# Results container: {(line, gid): {approach_label: result_dict}}
results: dict[tuple, dict[str, dict]] = {}

for line, gid in TARGETS:
    rec = [r for r in all_recs if r.line == line and r.group_id == gid][0]
    tri = rec.train_triangles["paid"]
    actual_ult = float(rec.actual_ultimates["paid"])
    mack = cl.MackChainladder().fit(tri)
    mack_ult_arr = np.asarray(mack.ultimate_.values, dtype=float).squeeze()
    mack_total = float(np.nansum(mack_ult_arr))
    latest = float(np.nansum(np.asarray(mack.latest_diagonal.values, dtype=float).squeeze()))

    BASELINES[(line, gid)]["mack"] = mack_total
    BASELINES[(line, gid)]["actual"] = actual_ult

    section(f"{rec.company} ({line}/{gid})")
    print(f"  Mack total ultimate: {mack_total:>12,.1f}")
    print(f"  Latest observed:     {latest:>12,.1f}")
    print(f"  Actual ultimate:     {actual_ult:>12,.1f}")

    # Build CL-informed prior info (centers + dev labels)
    cl_info = build_mack_priors(tri)

    # Build the four prior dicts
    priors_default = None  # package adaptive priors
    priors_mack_03 = make_priors_approach_b(cl_info, sigma=0.3)  # same as script 11
    priors_a = make_priors_approach_a(cl_info)
    priors_b = make_priors_approach_b(cl_info, sigma=0.5)
    priors_c = make_priors_approach_c(cl_info)

    tri_results: dict[str, dict] = {}

    # Default
    tri_results["Default (σ=1, uninformed)"] = fit_and_extract(
        tri, priors_default, "Default (σ=1, uninformed)",
        actual_ult, latest, draws=2000, tune=1000, chains=2,
    )

    # Mack-informed σ=0.3 (replicates script 11)
    tri_results["Mack-informed σ=0.3 (script 11)"] = fit_and_extract(
        tri, priors_mack_03, "Mack-informed σ=0.3 (script 11)",
        actual_ult, latest, draws=2000, tune=1000, chains=2,
    )

    # Approach A: tapered dev SDs
    tri_results["A: tapered dev SDs (0.15–0.6)"] = fit_and_extract(
        tri, priors_a, "A: tapered dev SDs (0.15–0.6)",
        actual_ult, latest, draws=2000, tune=1000, chains=2,
    )

    # Approach B: uniform σ=0.5
    tri_results["B: uniform σ=0.5"] = fit_and_extract(
        tri, priors_b, "B: uniform σ=0.5",
        actual_ult, latest, draws=2000, tune=1000, chains=2,
    )

    # Approach C: uniform σ=1.0 with CL centers
    tri_results["C: uniform σ=1.0 + CL centers"] = fit_and_extract(
        tri, priors_c, "C: uniform σ=1.0 + CL centers",
        actual_ult, latest, draws=2000, tune=1000, chains=2,
    )

    results[(line, gid)] = tri_results

    # Per-triangle summary
    print(f"\n  Results for {rec.company}:")
    hdr = f"  {'Approach':<42s}  {'Ult (p50)':>12s}  {'Ratio/Mack':>10s}  {'Ratio/Actual':>12s}  {'Implied%':>8s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for lbl in APPROACH_LABELS:
        if lbl not in tri_results:
            continue
        r = tri_results[lbl]
        u = r["ult_median"]
        print(
            f"  {lbl:<42s}  {u:12,.1f}  {u/mack_total:10.4f}  "
            f"{u/actual_ult:12.4f}  {r['implied_pctl']:8.3f}"
        )
    print(f"  {'Mack (reference)':<42s}  {mack_total:12,.1f}  {'1.0000':>10s}  "
          f"{mack_total/actual_ult:12.4f}  {'N/A':>8s}")
    print(f"  {'Actual (ground truth)':<42s}  {actual_ult:12,.1f}  "
          f"{actual_ult/mack_total:10.4f}  {'1.0000':>12s}  {'N/A':>8s}")


# ---------------------------------------------------------------------------
# CROSS-TRIANGLE COMPARISON TABLE
# ---------------------------------------------------------------------------

section("FULL COMPARISON TABLE — 3 Triangles × 5 Approaches")

companies = [(line, gid, BASELINES[(line, gid)]["company"]) for line, gid in TARGETS]

for lbl in APPROACH_LABELS:
    print(f"\n  Approach: {lbl}")
    print(f"  {'Company':<35s}  {'Mack':>10s}  {'Ult (p50)':>12s}  {'Ratio/Mack':>10s}  {'Implied%':>8s}  {'Within5%':>8s}")
    print("  " + "-" * 92)
    for line, gid, company in companies:
        mack = BASELINES[(line, gid)]["mack"]
        actual = BASELINES[(line, gid)]["actual"]
        tri_results = results.get((line, gid), {})
        if lbl not in tri_results:
            continue
        r = tri_results[lbl]
        u = r["ult_median"]
        w5 = "YES" if abs(u / mack - 1.0) <= 0.05 else "NO "
        print(
            f"  {company:<35s}  {mack:10,.1f}  {u:12,.1f}  {u/mack:10.4f}  "
            f"{r['implied_pctl']:8.3f}  {w5:>8s}"
        )


# ---------------------------------------------------------------------------
# SUMMARY: aggregate statistics across 3 triangles
# ---------------------------------------------------------------------------

section("AGGREGATE STATISTICS — Median across 3 triangles")

print(f"\n  {'Approach':<42s}  {'Median Ratio/Mack':>18s}  {'Median Implied%':>15s}  {'Pct within 5%':>13s}")
print("  " + "-" * 92)

for lbl in APPROACH_LABELS:
    ratios = []
    pctls = []
    for line, gid in TARGETS:
        mack = BASELINES[(line, gid)]["mack"]
        tri_results = results.get((line, gid), {})
        if lbl not in tri_results:
            continue
        r = tri_results[lbl]
        ratios.append(r["ult_median"] / mack)
        pctls.append(r["implied_pctl"])
    if not ratios:
        continue
    med_ratio = float(np.median(ratios))
    med_pctl = float(np.median(pctls))
    pct_w5 = float(np.mean(np.abs(np.array(ratios) - 1.0) <= 0.05)) * 100
    print(
        f"  {lbl:<42s}  {med_ratio:18.4f}  {med_pctl:15.3f}  {pct_w5:13.1f}%"
    )


# ---------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------

section("VERDICT — Structured vs Strongly-Informed Priors")

print("""
Key questions:
  1. Do tapered SDs (Approach A) match Mack within 5-10% while using
     WIDER uncertainty at early devs?
  2. Does Approach B (σ=0.5, wider) still track Mack closely?
  3. Does Approach C (σ=1.0 + CL centers) — revealing whether CENTERS
     alone drive the result, independent of SD.
""")

for line, gid in TARGETS:
    mack = BASELINES[(line, gid)]["mack"]
    actual = BASELINES[(line, gid)]["actual"]
    company = BASELINES[(line, gid)]["company"]
    tri_results = results.get((line, gid), {})

    print(f"  {company} ({line}/{gid}):")
    for lbl in APPROACH_LABELS:
        if lbl not in tri_results:
            continue
        r = tri_results[lbl]
        u = r["ult_median"]
        pct = (u / mack - 1.0) * 100
        w5 = "within 5%" if abs(pct) <= 5.0 else "outside 5%"
        print(f"    {lbl:<42s}: {u:>10,.1f}  ({pct:+.1f}% vs Mack)  {w5}")
    print()

# Assess: do CL centers dominate regardless of SD?
print("  SD sensitivity analysis (Celina — most informative triangle):")
celina = results.get(("ppauto", 353), {})
mack_celina = BASELINES[("ppauto", 353)]["mack"]
for lbl in ["Mack-informed σ=0.3 (script 11)", "A: tapered dev SDs (0.15–0.6)",
            "B: uniform σ=0.5", "C: uniform σ=1.0 + CL centers"]:
    if lbl in celina:
        u = celina[lbl]["ult_median"]
        pct = (u / mack_celina - 1.0) * 100
        print(f"    {lbl:<42s}: {pct:+.1f}%")

print()
print("  If Approaches A, B, C all yield similar ratios to Mack, this shows")
print("  the CENTERS are the binding constraint — SDs matter little in 0.3–1.0 range.")
print("  If A/B are closer to Mack than C, the SD is doing real work.")

section("DONE")
