"""14_calibration_diagnosis.py — Investigate calibration bias for BayesianCSR and MT5_cal.

Step 1: Diagnose the calibration bias from v9 backtest results.
Step 2: Refit 3 worst-calibrated triangles per method with tighter priors.
Step 3: Document findings.

Run:
    cd references/meyers-backtest
    uv run python 14_calibration_diagnosis.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths / imports
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))   # _common.py
sys.path.insert(0, str(REPO_ROOT))                         # bayesianchainladder

from _common import load_csr_priors_for_line, load_exposure_triangle  # noqa: E402
from methods import testr_bayesian_csr, testr_glm_mt5_cal  # noqa: E402

import reservetestr as rt  # noqa: E402
from bayesianchainladder import BayesianCSR, BayesianChainLadderGLM  # noqa: E402
from reservetestr.utils import latest_cumulative_sum  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent / "cache"

# ---------------------------------------------------------------------------
# STEP 1: Diagnose calibration from v9 backtest
# ---------------------------------------------------------------------------

def step1_diagnose():
    """Load v9 results and compute calibration statistics."""
    print("=" * 70)
    print("STEP 1: Calibration Diagnosis (v9 backtest results)")
    print("=" * 70)

    df = pd.read_parquet(CACHE_DIR / "backtest_all.parquet")

    for method in ["bayesian_csr", "glm_mt5_cal"]:
        sub = df[df["method"] == method].copy()
        pctl = sub["implied_pctl"].dropna()
        print(f"\n--- {method} (n={len(pctl)}) ---")
        print(f"  Mean implied_pctl:   {pctl.mean():.4f}  (ideal: 0.500)")
        print(f"  Median implied_pctl: {pctl.median():.4f}  (ideal: 0.500)")
        print(f"  Fraction in [0.00, 0.05): {(pctl < 0.05).mean():.2%}  (ideal: 5%)")
        print(f"  Fraction in [0.00, 0.10): {(pctl < 0.10).mean():.2%}  (ideal: 10%)")
        print(f"  Fraction in [0.05, 0.95]: {((pctl >= 0.05) & (pctl <= 0.95)).mean():.2%}  (ideal: 90%)")
        print(f"  Fraction in (0.95, 1.00]: {(pctl > 0.95).mean():.2%}  (ideal: 5%)")

        print(f"\n  Per-line median implied_pctl:")
        by_line = sub.groupby("line")["implied_pctl"].agg(["median", "mean", "count"])
        for line, row in by_line.iterrows():
            print(f"    {line:10s}: median={row['median']:.3f}  mean={row['mean']:.3f}  n={int(row['count'])}")

        print(f"\n  Worst-calibrated (pctl < 0.05):")
        worst = sub[sub["implied_pctl"] < 0.05].sort_values("implied_pctl")
        for _, row in worst.iterrows():
            print(f"    [{row['line']:8s} {int(row['group_id']):5d}] {row['company'][:40]:40s}  pctl={row['implied_pctl']:.4f}")

    # Histogram summary (5% bins)
    print("\n--- Histogram (5% bins) ---")
    bins = np.arange(0, 1.05, 0.05)
    for method in ["bayesian_csr", "glm_mt5_cal"]:
        sub = df[df["method"] == method]["implied_pctl"].dropna()
        counts, _ = np.histogram(sub, bins=bins)
        print(f"\n{method}:")
        print(f"  {'Bin':12s}  Count  Fraction")
        for i, cnt in enumerate(counts):
            lo, hi = bins[i], bins[i+1]
            print(f"  [{lo:.2f},{hi:.2f}):  {cnt:3d}    {cnt/len(sub):.2%}")

    return df


def step1_select_worst(df: pd.DataFrame, method: str, n: int = 3) -> list[tuple]:
    """Return the n worst-calibrated (lowest implied_pctl) records for a method."""
    sub = df[(df["method"] == method) & df["implied_pctl"].notna()].copy()
    worst = sub.nsmallest(n, "implied_pctl")
    return [(row["line"], int(row["group_id"]), row["company"], row["implied_pctl"])
            for _, row in worst.iterrows()]


# ---------------------------------------------------------------------------
# STEP 2: Refit worst triangles with tighter priors
# ---------------------------------------------------------------------------

def _get_record(line: str, group_id: int):
    """Load the train/test/actual data for a single triangle."""
    records = rt.build_triangle_records()
    for r in records:
        if r.line == line and r.group_id == group_id:
            return r
    raise ValueError(f"Record not found: {line}/{group_id}")


def _empirical_pctl(samples: np.ndarray, actual: float) -> float:
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan")
    return float(np.mean(s <= actual))


def refit_csr_with_tighter_gamma(record, baseline_pctl: float) -> dict:
    """Refit BayesianCSR with gamma_sigma=0.02 instead of default 0.05."""
    from bayesianchainladder import BayesianCSR

    line, group_id = record.line, record.group_id
    triangle = record.train_triangles.get("paid")
    prem_tri = load_exposure_triangle(line, group_id)
    actual_ultimate = record.actual_ultimates.get("paid", float("nan"))
    latest_observed = latest_cumulative_sum(triangle)

    # Load line-specific priors and override gamma sigma
    priors = load_csr_priors_for_line(line)
    priors_tight = dict(priors)
    priors_tight["gamma"] = {"mu": priors.get("gamma", {}).get("mu", 0.0), "sigma": 0.02}

    results = {}
    for label, gamma_sigma in [("default_gamma", 0.05), ("tight_gamma_002", 0.02), ("tight_gamma_001", 0.01)]:
        p = dict(priors)
        p["gamma"] = {"mu": priors.get("gamma", {}).get("mu", 0.0), "sigma": gamma_sigma}
        model = BayesianCSR(
            priors=p,
            draws=1000,
            tune=1000,
            chains=2,
            target_accept=0.95,
            random_seed=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(triangle, premium_triangle=prem_tri)

        reserves = model.reserves_posterior_
        total_ibnr_samples = np.asarray(reserves.sum(dim="origin").values, dtype=float).flatten()
        total_ult_samples = total_ibnr_samples + latest_observed
        pctl = _empirical_pctl(total_ult_samples, actual_ultimate)
        mean_ult = float(np.nanmean(total_ult_samples))
        std_ult = float(np.nanstd(total_ult_samples, ddof=1))
        results[label] = {"pctl": pctl, "mean_ult": mean_ult, "std_ult": std_ult, "cv": std_ult / abs(mean_ult - latest_observed) if mean_ult > latest_observed else float("nan")}

    return results


def refit_mt5cal_with_tighter_nu(record, baseline_pctl: float) -> dict:
    """Refit MT5_cal with tighter nu prior: Gamma(10, 0.5) vs default Gamma(2, 0.1)."""
    import bambi as bmb

    line, group_id = record.line, record.group_id
    triangle = record.train_triangles.get("paid")
    prem_tri = load_exposure_triangle(line, group_id)
    actual_ultimate = record.actual_ultimates.get("paid", float("nan"))
    latest_observed = latest_cumulative_sum(triangle)

    from _common import load_glm_priors_for_line

    results = {}
    nu_configs = [
        ("default_nu",     bmb.Prior("Gamma", alpha=2,  beta=0.1)),
        ("tighter_nu_10",  bmb.Prior("Gamma", alpha=10, beta=0.5)),
        ("tight_nu_20",    bmb.Prior("Gamma", alpha=20, beta=1.0)),
    ]

    base_priors = load_glm_priors_for_line(line, "MT5_cal")

    for label, nu_prior in nu_configs:
        p = dict(base_priors)
        p["nu"] = nu_prior

        model = BayesianChainLadderGLM(
            formula="incremental ~ 1 + (1|origin) + bs(dev_idx, df=4) + (1|calendar)",
            family="t",
            priors=p,
            draws=1000,
            tune=1000,
            chains=2,
            target_accept=0.95,
            random_seed=42,
            init_priors_from_chainladder=True,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(triangle)

            reserves = model.reserves_posterior_
            total_ibnr_samples = np.asarray(reserves.sum(dim="origin").values, dtype=float).flatten()
            total_ult_samples = total_ibnr_samples + latest_observed
            pctl = _empirical_pctl(total_ult_samples, actual_ultimate)
            mean_ult = float(np.nanmean(total_ult_samples))
            std_ult = float(np.nanstd(total_ult_samples, ddof=1))
            results[label] = {"pctl": pctl, "mean_ult": mean_ult, "std_ult": std_ult}
        except Exception as e:
            results[label] = {"pctl": float("nan"), "error": str(e)[:80]}

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Step 1: Diagnose
    df = step1_diagnose()

    print("\n" + "=" * 70)
    print("STEP 2: Test Fits with Tighter Priors")
    print("=" * 70)

    # Select worst CSR and MT5_cal triangles
    csr_worst = step1_select_worst(df, "bayesian_csr", n=3)
    mt5_worst = step1_select_worst(df, "glm_mt5_cal", n=3)

    print(f"\nCSR worst 3: {[(x[0], x[1], round(x[3], 4)) for x in csr_worst]}")
    print(f"MT5_cal worst 3: {[(x[0], x[1], round(x[3], 4)) for x in mt5_worst]}")

    # --- CSR: try tighter gamma ---
    print("\n--- BayesianCSR: tighter gamma_sigma ---")
    csr_results = {}
    for line, gid, company, baseline_pctl in csr_worst:
        print(f"\n  {line}/{gid} ({company}), baseline pctl={baseline_pctl:.4f}")
        try:
            record = _get_record(line, gid)
            res = refit_csr_with_tighter_gamma(record, baseline_pctl)
            csr_results[(line, gid)] = res
            for variant, stats in res.items():
                print(f"    {variant:22s}: pctl={stats['pctl']:.4f}  mean_ult={stats['mean_ult']:,.0f}  cv={stats.get('cv', float('nan')):.3f}")
        except Exception as e:
            print(f"    ERROR: {e}")

    # --- MT5_cal: try tighter nu ---
    print("\n--- MT5_cal: tighter nu prior ---")
    mt5_results = {}
    for line, gid, company, baseline_pctl in mt5_worst:
        print(f"\n  {line}/{gid} ({company}), baseline pctl={baseline_pctl:.4f}")
        try:
            record = _get_record(line, gid)
            res = refit_mt5cal_with_tighter_nu(record, baseline_pctl)
            mt5_results[(line, gid)] = res
            for variant, stats in res.items():
                if "error" in stats:
                    print(f"    {variant:22s}: ERROR: {stats['error']}")
                else:
                    print(f"    {variant:22s}: pctl={stats['pctl']:.4f}  mean_ult={stats['mean_ult']:,.0f}")
        except Exception as e:
            print(f"    ERROR: {e}")

    # --- Step 3: Conclusions ---
    print("\n" + "=" * 70)
    print("STEP 3: Conclusions")
    print("=" * 70)

    print("""
Summary of calibration findings:

BayesianCSR (mean implied_pctl = 0.424):
  - 9% of triangles have implied_pctl < 0.05 (ideal: 5%)
  - 19.5% have pctl < 0.10 (ideal: 10%)
  - Worst-affected lines: comauto (mean pctl=0.35) and ppauto (0.33)
  - Right-tail bias: posterior distributions systematically OVERESTIMATE
    (actual ultimates land below the median), suggesting the posterior is
    too heavy on the right tail.
  - Default gamma_sigma=0.05 is tested against 0.02 and 0.01 on 3 worst
    triangles.  See printed results above.

glm_mt5_cal (mean implied_pctl = 0.342):
  - 8.5% of triangles have implied_pctl < 0.05 (ideal: 5%)
  - 0% have pctl > 0.95 (ASYMMETRIC bias — only left-truncation issue)
  - Worst lines: ppauto (mean 0.26) and comauto (0.31)
  - Default nu prior Gamma(2, 0.1) allows heavy tails; tested against
    Gamma(10, 0.5) and Gamma(20, 1.0) on 3 worst triangles.
  - See printed results above.

Structural observation:
  Both methods show right-tail bias (actual < median posterior prediction).
  This is consistent with:
    (a) The Meyers data (1988-1997) having LOWER loss development than
        typical Schedule P data from prior-elicitation-2026 suggests
        (logelr priors may be too high)
    (b) The prior-elicitation-2026 data covering 2024 Schedule P which
        has different scale and time-period characteristics

If tighter priors DO NOT meaningfully improve calibration on the 6 test
triangles (pctl improvement < 0.05 absolute), the bias is structural:
  - logelr prior may need downward adjustment for 1988-1997 data vintage
  - Or the posterior aggregation method should be reviewed
""")


if __name__ == "__main__":
    main()
