"""15_per_line_csr_priors.py — Per-line CSR prior elicitation vs default priors.

For each Meyers line, fit BayesianCSR on 5 representative triangles with:
  (a) line-specific elicited priors from prior-elicitation-2026/cache/csr_fits.parquet
  (b) package defaults (logelr=Normal(-0.4, 3.162), gamma=Normal(0, 0.05))

Compare implied_pctl distributions to assess whether per-line priors improve
calibration.

Run:
    cd references/meyers-backtest
    uv run python 15_per_line_csr_priors.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT))

from _common import load_csr_priors_for_line, load_exposure_triangle  # noqa: E402
import reservetestr as rt  # noqa: E402
from bayesianchainladder import BayesianCSR  # noqa: E402
from reservetestr.utils import latest_cumulative_sum  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent / "cache"
LINES = ["comauto", "ppauto", "wkcomp", "othliab"]
N_PER_LINE = 5  # number of representative triangles per line
DRAWS = 500
TUNE = 500
CHAINS = 2
TARGET_ACCEPT = 0.95
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Prior definitions
# ---------------------------------------------------------------------------

DEFAULT_PRIORS = {
    "logelr": {"mu": -0.4, "sigma": 3.162},
    "gamma": {"mu": 0.0, "sigma": 0.05},
}


def get_elicited_priors(line: str) -> dict:
    """Return line-specific elicited priors from prior-elicitation-2026 cache."""
    return load_csr_priors_for_line(line)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _empirical_pctl(samples: np.ndarray, actual: float) -> float:
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan")
    return float(np.mean(s <= actual))


def run_single(record, priors: dict, label: str) -> dict:
    """Fit BayesianCSR and return calibration metrics."""
    line, gid = record.line, record.group_id
    triangle = record.train_triangles.get("paid")
    if triangle is None:
        return {"status": "no_triangle"}

    try:
        prem_tri = load_exposure_triangle(line, gid)
        model = BayesianCSR(
            priors=priors,
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(triangle, premium_triangle=prem_tri)

        reserves = model.reserves_posterior_
        total_ibnr = np.asarray(
            reserves.sum(dim="origin").values, dtype=float
        ).flatten()
        latest_observed = latest_cumulative_sum(triangle)
        total_ult = total_ibnr + latest_observed
        actual_ult = record.actual_ultimates.get("paid", float("nan"))
        pctl = _empirical_pctl(total_ult, actual_ult)
        mean_ult = float(np.nanmean(total_ult))
        std_ult = float(np.nanstd(total_ult, ddof=1))
        return {
            "status": "ok",
            "pctl": pctl,
            "mean_ult": mean_ult,
            "actual_ult": actual_ult,
            "latest_obs": latest_observed,
            "std_ult": std_ult,
        }
    except Exception as e:
        return {"status": f"error:{type(e).__name__}:{str(e)[:80]}"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    records = rt.build_triangle_records()

    print("=" * 70)
    print("Investigation 3: Per-line CSR priors vs defaults")
    print("=" * 70)
    print(f"Fitting {N_PER_LINE} triangles per line × 4 lines × 2 prior variants")
    print(f"Draws={DRAWS}, Tune={TUNE}, Chains={CHAINS}")
    print()

    # Print prior comparison
    print("Prior comparison:")
    for line in LINES:
        ep = get_elicited_priors(line)
        dp = DEFAULT_PRIORS
        print(f"  {line}:")
        print(f"    logelr: default=N({dp['logelr']['mu']:.3f}, {dp['logelr']['sigma']:.3f})"
              f"  elicited=N({ep['logelr']['mu']:.3f}, {ep['logelr']['sigma']:.3f})")
        print(f"    gamma:  default=N({dp['gamma']['mu']:.4f}, {dp['gamma']['sigma']:.4f})"
              f"  elicited=N({ep['gamma']['mu']:.4f}, {ep['gamma']['sigma']:.4f})")
    print()

    # Select representative triangles per line
    # Use the 5 triangles closest to the median implied_pctl (from v9 backtest)
    # to avoid extreme outliers contaminating the comparison
    all_results = pd.DataFrame()
    try:
        all_results = pd.read_parquet(CACHE_DIR / "backtest_all.parquet")
    except Exception:
        pass

    summary_rows = []

    for line in LINES:
        print(f"\n--- {line} ---")
        line_records = [r for r in records if r.line == line]

        # Select N_PER_LINE triangles near median pctl to be representative
        if not all_results.empty:
            csr_line = all_results[
                (all_results["method"] == "bayesian_csr") &
                (all_results["line"] == line)
            ].copy()
            if not csr_line.empty:
                median_pctl = csr_line["implied_pctl"].median()
                csr_line["dist_from_median"] = abs(csr_line["implied_pctl"] - median_pctl)
                selected_ids = csr_line.nsmallest(N_PER_LINE, "dist_from_median")["group_id"].tolist()
                selected_records = [r for r in line_records if r.group_id in selected_ids]
            else:
                selected_records = line_records[:N_PER_LINE]
        else:
            selected_records = line_records[:N_PER_LINE]

        elicited_priors = get_elicited_priors(line)

        pctl_default = []
        pctl_elicited = []

        for record in selected_records:
            gid = record.group_id
            print(f"  group_id={gid}", end="  ", flush=True)

            # Default priors
            res_def = run_single(record, DEFAULT_PRIORS, "default")
            # Elicited priors
            res_eli = run_single(record, elicited_priors, "elicited")

            if res_def.get("status") == "ok" and res_eli.get("status") == "ok":
                print(
                    f"default_pctl={res_def['pctl']:.3f}  "
                    f"elicited_pctl={res_eli['pctl']:.3f}  "
                    f"delta={res_eli['pctl'] - res_def['pctl']:+.3f}"
                )
                pctl_default.append(res_def["pctl"])
                pctl_elicited.append(res_eli["pctl"])
                summary_rows.append({
                    "line": line,
                    "group_id": gid,
                    "pctl_default": res_def["pctl"],
                    "pctl_elicited": res_eli["pctl"],
                    "delta": res_eli["pctl"] - res_def["pctl"],
                })
            else:
                print(f"SKIP (def:{res_def['status']}, eli:{res_eli['status']})")

        if pctl_default and pctl_elicited:
            print(
                f"  {line} summary: "
                f"mean_pctl_default={np.mean(pctl_default):.3f}  "
                f"mean_pctl_elicited={np.mean(pctl_elicited):.3f}  "
                f"mean_delta={np.mean(np.array(pctl_elicited) - np.array(pctl_default)):+.3f}"
            )

    # Overall summary
    print("\n" + "=" * 70)
    print("Overall Summary")
    print("=" * 70)

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        print(f"\nTotal triangles evaluated: {len(df)}")
        print(f"\nPer-line comparison:")
        for line in LINES:
            sub = df[df["line"] == line]
            if sub.empty:
                continue
            print(
                f"  {line}: n={len(sub)}  "
                f"mean_default={sub['pctl_default'].mean():.3f}  "
                f"mean_elicited={sub['pctl_elicited'].mean():.3f}  "
                f"mean_delta={sub['delta'].mean():+.3f}  "
                f"pct_improved={((sub['delta'] > 0.02).sum() / len(sub)):.1%}"
            )

        overall_delta = df["delta"].mean()
        print(f"\nOverall mean delta (elicited - default): {overall_delta:+.3f}")
        print()

        # Decision rule
        threshold = 0.05  # meaningful improvement
        if abs(overall_delta) < threshold:
            print(
                "CONCLUSION: Per-line priors do NOT meaningfully improve calibration\n"
                f"(mean_delta={overall_delta:+.3f}, threshold=±{threshold}).\n"
                "The bias is structural — priors alone cannot fix it.\n"
                "Possible structural fixes:\n"
                "  1. Different posterior aggregation (median instead of mean)\n"
                "  2. Different parameterization or family\n"
                "  3. Scale normalization (the 1988-1997 Meyers data may have\n"
                "     different absolute scale than 2024 Schedule P priors)\n"
            )
        else:
            sign = "IMPROVES" if overall_delta > 0 else "WORSENS"
            print(
                f"CONCLUSION: Per-line priors {sign} calibration "
                f"(mean_delta={overall_delta:+.3f}).\n"
                "Consider implementing a 'line-aware' mode for BayesianCSR.\n"
            )

        # Save results
        out_path = CACHE_DIR / "15_per_line_csr_priors.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
