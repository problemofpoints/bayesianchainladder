"""12_glm_prior_synthesis_t.py — Synthesize MT2 posteriors into recommended t-family GLM priors.

Reads cache/m1_t_posteriors.parquet (written by 11_glm_priors_t.py),
filters to converged fits (max_rhat < 1.1), and per line produces
recommended weakly-informative priors for the MT2 Student-t loss-ratio GLM.

Model: incremental ~ 1 + C(origin) + bs(dev_idx, df=4)
       family='t', link='identity', response_per_exposure=True

The response is on loss-ratio scale (incremental paid / net_earned_premium),
so the intercept represents an intercept loss ratio — typically in [0, 0.5].
The sigma parameter is the Student-t scale (NOT the variance); nu controls
tail weight (nu→∞ → Gaussian, nu≈2–5 → heavy tails).

Prior recommendations:
  Intercept:   Normal(mean_intercept, 1.5 * sd_intercept)  [loss-ratio scale]
  Sigma:       HalfNormal(1.5 * median_sigma_mean)
  Nu:          Gamma(alpha=2, beta=0.1)  [Bambi default — kept unless heavy evidence otherwise]
  Origin SD:   HalfNormal(1.5 * median_origin_effect_sd)
  Dev SD:      HalfNormal(1.5 * median_dev_effect_sd)

Output: cache/glm_t_priors_by_line.parquet
Run: uv run python references/prior-elicitation-2026/12_glm_prior_synthesis_t.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from _common import cache_path


def main() -> int:
    in_path = cache_path("m1_t_posteriors.parquet")
    if not in_path.exists():
        print(f"ERROR: {in_path} not found. Run 11_glm_priors_t.py first.")
        return 1

    df = pd.read_parquet(in_path)
    print(f"Loaded {len(df)} rows from {in_path}")

    # Filter to converged fits.
    converged = df[df["max_rhat"] < 1.1].copy()
    total = len(df)
    ok = len(converged)
    print(f"Converged fits: {ok}/{total} ({100*ok/total:.0f}%)")

    if ok == 0:
        print("ERROR: No converged fits — cannot synthesize priors.")
        return 1

    rows = []
    for line, grp in converged.groupby("line"):
        n = len(grp)

        # Intercept statistics (on loss-ratio scale).
        int_mean = float(grp["intercept_mean"].mean())
        int_sd = float(grp["intercept_sd"].mean())

        # Sigma (Student-t scale).
        sigma_median = float(grp["sigma_mean"].median())
        sigma_sd_avg = float(grp["sigma_sd"].mean())

        # Nu (degrees of freedom) — report median for awareness.
        nu_median = float(grp["nu_mean"].median())

        # Effect spread statistics.
        origin_sd_p50 = float(grp["origin_effect_sd_p50"].median())
        dev_sd_p50 = float(grp["dev_effect_sd_p50"].median())

        # Recommended priors (strings for documentation).
        intercept_prior = f"Normal({int_mean:.4f}, {1.5 * int_sd:.4f})"

        if np.isfinite(sigma_median) and sigma_median > 0:
            sigma_prior = f"HalfNormal({1.5 * sigma_median:.4f})"
        else:
            sigma_prior = "HalfNormal(0.05)  [fallback: insufficient data]"

        # Nu prior: Bambi's default for StudentT is Gamma(alpha=2, beta=0.1)
        # which puts median around 14 and allows both heavy and near-Gaussian tails.
        # We recommend keeping this default unless the posterior nu_median is
        # systematically very low (< 5), which would suggest even heavier tails.
        if np.isfinite(nu_median) and nu_median < 5:
            nu_prior = f"Gamma(alpha=2, beta=0.5)  [tighter: posterior nu_median={nu_median:.1f}]"
        else:
            nu_prior = f"Gamma(alpha=2, beta=0.1)  [Bambi default; posterior nu_median={nu_median:.1f}]"

        if np.isfinite(origin_sd_p50) and origin_sd_p50 > 0:
            origin_sigma_prior = f"HalfNormal({1.5 * origin_sd_p50:.4f})"
        else:
            origin_sigma_prior = "HalfNormal(0.1)  [fallback: insufficient data]"

        if np.isfinite(dev_sd_p50) and dev_sd_p50 > 0:
            dev_sigma_prior = f"HalfNormal({1.5 * dev_sd_p50:.4f})"
        else:
            dev_sigma_prior = "HalfNormal(0.1)  [fallback: insufficient data]"

        rows.append(
            {
                "line": line,
                "n_converged": n,
                # Raw statistics
                "intercept_mean_avg": int_mean,
                "intercept_sd_avg": int_sd,
                "sigma_median": sigma_median,
                "sigma_sd_avg": sigma_sd_avg,
                "nu_median": nu_median,
                "origin_effect_sd_p50": origin_sd_p50,
                "dev_effect_sd_p50": dev_sd_p50,
                # Recommended prior strings
                "t_intercept_prior": intercept_prior,
                "t_sigma_prior": sigma_prior,
                "t_nu_prior": nu_prior,
                "t_origin_sigma_prior": origin_sigma_prior,
                "t_dev_sigma_prior": dev_sigma_prior,
            }
        )

    out = pd.DataFrame(rows)
    out_path = cache_path("glm_t_priors_by_line.parquet")
    out.to_parquet(out_path, index=False)

    print(f"\nWrote {out_path}\n")
    print("=== Recommended GLM Priors (t family, identity link, loss-ratio scale) ===\n")
    display_cols = [
        "line",
        "n_converged",
        "t_intercept_prior",
        "t_sigma_prior",
        "t_nu_prior",
        "t_origin_sigma_prior",
        "t_dev_sigma_prior",
    ]
    print(out[display_cols].to_string(index=False))
    print()
    print("=== Underlying Statistics ===\n")
    stat_cols = [
        "line",
        "intercept_mean_avg",
        "intercept_sd_avg",
        "sigma_median",
        "sigma_sd_avg",
        "nu_median",
        "origin_effect_sd_p50",
        "dev_effect_sd_p50",
    ]
    print(out[stat_cols].to_string(index=False))
    print()
    print("NOTE: LOO for t-family (loss-ratio) fits is NOT directly comparable to")
    print("gamma+log (dollar-scale) fits. The response units differ, so the")
    print("log-likelihood densities have different reference scales. Compare")
    print("MT specs to each other and gamma specs to each other separately.")
    print("For cross-scale comparison, add log(EP_per_cell) to each MT")
    print("log-likelihood observation (Jacobian for the change of variables).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
