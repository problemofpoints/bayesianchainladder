"""05_test_no_offset.py — Test whether no-offset + diffuse priors makes GLM match Mack.

Hypothesis: removing the exposure offset and using diffuse priors (Normal(0,10))
on C(origin)/C(dev) should drive the GLM/Mack ratio to ~1.000, matching ODP MLE.

Variants:
  A: M1_cat baseline     — C(origin)+C(dev), gamma+log, exposure="net_earned_premium", adaptive priors
  B: M1_cat no offset    — same formula,      gamma+log, exposure=None,                adaptive priors
  C: M1_cat no offset + diffuse priors — same formula, gamma+log, exposure=None,       Normal(0,10) priors

Reference methods:
  Mack            — MackChainladder
  Det CL          — deterministic Chainladder

Run:
    cd references/meyers-backtest
    uv run python 05_test_no_offset.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import bambi as bmb
import chainladder as cl
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))   # _common.py
sys.path.insert(0, str(REPO_ROOT))                         # bayesianchainladder package

from _common import load_exposure_triangle  # noqa: E402

import reservetestr  # noqa: E402
from bayesianchainladder import BayesianChainLadderGLM  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DRAWS = 1000
TUNE = 1000
CHAINS = 2
TARGET_ACCEPT = 0.95

FORMULA = "incremental ~ 1 + C(origin) + C(dev)"

DIFFUSE_PRIORS = {
    "Intercept": bmb.Prior("Normal", mu=0, sigma=10),
    "C(origin)": bmb.Prior("Normal", mu=0, sigma=10),
    "C(dev)": bmb.Prior("Normal", mu=0, sigma=10),
    "alpha": bmb.Prior("HalfNormal", sigma=10),
}

# (line, group_id) — one per line; wkcomp uses the diagnostic triangle 21172.
# All four triangles confirmed to have no negative incrementals (gamma-compatible).
TRIANGLES = [
    ("wkcomp",   21172),   # Vanliner Ins Co       — the diagnostic triangle
    ("comauto",  18767),   # Church Mut Ins Co      — comauto, no neg incrementals
    ("ppauto",   15199),   # Standard Mut Ins Co    — ppauto, no neg incrementals
    ("othliab",    620),   # Employers Mut Co       — othliab, no neg incrementals
]

# Deterministic seed per (variant_index, triangle_index) to avoid seed conflicts
# variant 0=A, 1=B, 2=C; triangle 0-3
def _seed(variant_idx: int, tri_idx: int) -> int:
    return 100 * (variant_idx + 1) + tri_idx


# ---------------------------------------------------------------------------
# Helper: posterior median total ultimate
# ---------------------------------------------------------------------------

def _posterior_median_total(model: BayesianChainLadderGLM) -> float:
    total_ibnr = model.reserves_posterior_.sum(dim="origin").values.flatten()
    total_ibnr = total_ibnr[np.isfinite(total_ibnr)]
    paid_total = float(model.ultimate_["paid_to_date"].sum())
    return float(np.median(total_ibnr)) + paid_total


# ---------------------------------------------------------------------------
# Load records
# ---------------------------------------------------------------------------
print("Loading triangle records…")
records = reservetestr.build_triangle_records()
rec_map = {(r.line, r.group_id): r for r in records}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
results = []  # list of dicts

for tri_idx, (line, group_id) in enumerate(TRIANGLES):
    rec = rec_map.get((line, group_id))
    if rec is None:
        print(f"\n[SKIP] {line}/{group_id}: not found in records")
        continue

    train_tri = rec.train_triangles["paid"]
    prem_tri = load_exposure_triangle(line, group_id)

    print(f"\n{'='*72}")
    print(f"Triangle {tri_idx+1}/4: {line} / group_id={group_id} ({rec.company})")
    print(f"{'='*72}")

    # -----------------------------------------------------------------------
    # Mack
    # -----------------------------------------------------------------------
    mack = cl.MackChainladder().fit(train_tri)
    mack_ult = np.asarray(mack.ultimate_.values, dtype=float).flatten()
    mack_total = float(np.nansum(mack_ult))

    # -----------------------------------------------------------------------
    # Det CL
    # -----------------------------------------------------------------------
    det_cl = cl.Chainladder().fit(cl.Development().fit_transform(train_tri))
    cl_ult = np.asarray(det_cl.ultimate_.values, dtype=float).flatten()
    cl_total = float(np.nansum(cl_ult))

    print(f"Mack total: {mack_total:>14,.1f}")
    print(f"Det CL total: {cl_total:>12,.1f}  ratio vs Mack: {cl_total/mack_total:.4f}")

    row = {
        "line": line,
        "group_id": group_id,
        "company": rec.company,
        "mack_total": mack_total,
        "cl_total": cl_total,
    }

    # -----------------------------------------------------------------------
    # Variant A: M1_cat baseline (exposure + adaptive priors)
    # -----------------------------------------------------------------------
    print("\nVariant A: M1_cat baseline (exposure=net_earned_premium, adaptive priors)…")
    try:
        model_a = BayesianChainLadderGLM(
            formula=FORMULA,
            family="gamma",
            link="log",
            exposure="net_earned_premium",
            response_per_exposure=False,
            priors=None,
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=_seed(0, tri_idx),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_a.fit(train_tri, exposure_triangle=prem_tri)

        a_total = _posterior_median_total(model_a)
        a_ratio = a_total / mack_total
        print(f"  Posterior median total: {a_total:>12,.1f}  ratio vs Mack: {a_ratio:.4f}")
        row["a_total"] = a_total
        row["a_ratio"] = a_ratio
        row["a_status"] = "ok"
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        row["a_total"] = float("nan")
        row["a_ratio"] = float("nan")
        row["a_status"] = f"error:{type(e).__name__}"

    # -----------------------------------------------------------------------
    # Variant B: no offset, adaptive priors
    # -----------------------------------------------------------------------
    print("\nVariant B: no exposure offset, adaptive priors…")
    try:
        model_b = BayesianChainLadderGLM(
            formula=FORMULA,
            family="gamma",
            link="log",
            exposure=None,
            response_per_exposure=False,
            priors=None,
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=_seed(1, tri_idx),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_b.fit(train_tri)

        b_total = _posterior_median_total(model_b)
        b_ratio = b_total / mack_total
        print(f"  Posterior median total: {b_total:>12,.1f}  ratio vs Mack: {b_ratio:.4f}")
        row["b_total"] = b_total
        row["b_ratio"] = b_ratio
        row["b_status"] = "ok"
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        row["b_total"] = float("nan")
        row["b_ratio"] = float("nan")
        row["b_status"] = f"error:{type(e).__name__}"

    # -----------------------------------------------------------------------
    # Variant C: no offset, diffuse priors Normal(0,10)
    # -----------------------------------------------------------------------
    print("\nVariant C: no exposure offset, diffuse priors Normal(0,10)…")
    try:
        model_c = BayesianChainLadderGLM(
            formula=FORMULA,
            family="gamma",
            link="log",
            exposure=None,
            response_per_exposure=False,
            priors=DIFFUSE_PRIORS,
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=_seed(2, tri_idx),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_c.fit(train_tri)

        c_total = _posterior_median_total(model_c)
        c_ratio = c_total / mack_total
        print(f"  Posterior median total: {c_total:>12,.1f}  ratio vs Mack: {c_ratio:.4f}")
        row["c_total"] = c_total
        row["c_ratio"] = c_ratio
        row["c_status"] = "ok"
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        row["c_total"] = float("nan")
        row["c_ratio"] = float("nan")
        row["c_status"] = f"error:{type(e).__name__}"

    results.append(row)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("SUMMARY TABLE  (all ultimates in $000s; ratios = GLM / Mack)")
print("=" * 90)

header = (
    f"{'Line':<10} {'Company':<28} "
    f"{'Mack':>12} {'Det CL':>12} "
    f"{'A (w/offset)':>13} {'A ratio':>8} "
    f"{'B (no off)':>12} {'B ratio':>8} "
    f"{'C (diffuse)':>12} {'C ratio':>8}"
)
print(header)
print("-" * 90)

for row in results:
    mack_v = row.get("mack_total", float("nan"))
    cl_v   = row.get("cl_total",   float("nan"))
    a_v    = row.get("a_total",    float("nan"))
    a_r    = row.get("a_ratio",    float("nan"))
    b_v    = row.get("b_total",    float("nan"))
    b_r    = row.get("b_ratio",    float("nan"))
    c_v    = row.get("c_total",    float("nan"))
    c_r    = row.get("c_ratio",    float("nan"))
    company_short = row.get("company", "")[:27]
    line   = row.get("line", "")

    def _fmt(v, r):
        if not np.isfinite(v):
            return f"{'ERR':>12}  {'ERR':>7}"
        return f"{v:>12,.0f}  {r:>7.4f}"

    print(
        f"{line:<10} {company_short:<28} "
        f"{mack_v:>12,.0f} {cl_v:>12,.0f} "
        + _fmt(a_v, a_r) + " "
        + _fmt(b_v, b_r) + " "
        + _fmt(c_v, c_r)
    )

# Averages
valid_a = [r["a_ratio"] for r in results if np.isfinite(r.get("a_ratio", float("nan")))]
valid_b = [r["b_ratio"] for r in results if np.isfinite(r.get("b_ratio", float("nan")))]
valid_c = [r["c_ratio"] for r in results if np.isfinite(r.get("c_ratio", float("nan")))]

print("-" * 90)
print(
    f"{'Average':<10} {'':<28} "
    f"{'':>12} {'':>12} "
    f"{'':>12}  {np.mean(valid_a):>7.4f}  "
    f"{'':>12}  {np.mean(valid_b):>7.4f}  "
    f"{'':>12}  {np.mean(valid_c):>7.4f}"
    if valid_a and valid_b and valid_c else "  (no valid results)"
)

print("""
Interpretation key
  A ratio ~ 1.07  → baseline (known bias from offset + adaptive priors)
  B ratio ~ 1.04  → offset removed, prior shrinkage remains
  C ratio ~ 1.00  → no offset + diffuse priors ≈ Mack MLE
""")

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
print("=" * 90)
print("VERDICT")
print("=" * 90)

if valid_c:
    mean_c = np.mean(valid_c)
    within_1pct = abs(mean_c - 1.0) <= 0.01
    within_2pct = abs(mean_c - 1.0) <= 0.02
    print(f"\nVariant C mean GLM/Mack ratio: {mean_c:.4f}")
    if within_1pct:
        print("  CONFIRMED: diffuse priors + no offset drives GLM/Mack to within ±1% of 1.000.")
        print("  Hypothesis is SUPPORTED.")
    elif within_2pct:
        print("  PARTIAL: ratio is within ±2% but not ±1%.")
        print("  Residual bias exists — investigate prior on 'alpha' (dispersion) or Bambi internals.")
    else:
        print(f"  REFUTED: ratio {mean_c:.4f} differs from 1.000 by >{abs(mean_c-1)*100:.1f}%.")
        print("  Further investigation needed (check for Jensen bias, alpha prior, etc.).")

    # Decompose offset vs prior contribution
    if valid_a and valid_b:
        offset_contrib = np.mean(valid_a) - np.mean(valid_b)
        prior_contrib  = np.mean(valid_b) - mean_c
        print(f"\nBias decomposition (in ratio units):")
        print(f"  Total bias (A-1):            {np.mean(valid_a) - 1:.4f}")
        print(f"  Offset contribution (A-B):   {offset_contrib:.4f}")
        print(f"  Prior contribution  (B-C):   {prior_contrib:.4f}")
        print(f"  Residual (C-1):              {mean_c - 1:.4f}")
else:
    print("\nNo valid Variant C results — cannot assess hypothesis.")
