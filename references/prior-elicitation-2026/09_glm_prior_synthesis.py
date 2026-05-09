"""09_glm_prior_synthesis.py — Synthesize M1 posteriors into recommended GLM priors.

Reads cache/m1_posteriors.parquet (written by 08_glm_priors.py),
filters to converged fits (max_rhat < 1.1), and per line produces
recommended weakly-informative priors for the M1 gamma GLM.

Prior recommendations:
  Intercept:   Normal(mean_intercept, 1.5 * sd_intercept)
  Alpha shape: HalfNormal(1.5 * p90_alpha_mean)
  Origin SD:   HalfNormal(1.5 * median_origin_effect_sd)
  Dev SD:      HalfNormal(1.5 * median_dev_effect_sd)

Output: cache/glm_priors_by_line.parquet
Run: uv run python references/prior-elicitation-2026/09_glm_prior_synthesis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from _common import cache_path


def main() -> int:
    in_path = cache_path("m1_posteriors.parquet")
    if not in_path.exists():
        print(f"ERROR: {in_path} not found. Run 08_glm_priors.py first.")
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
        # Intercept statistics.
        int_mean = float(grp["intercept_mean"].mean())
        int_sd = float(grp["intercept_sd"].mean())
        # Alpha shape statistics.
        alpha_p90 = float(grp["alpha_mean"].quantile(0.90))
        alpha_mean_avg = float(grp["alpha_mean"].mean())
        alpha_sd_avg = float(grp["alpha_sd"].mean())
        # Effect spread statistics.
        origin_sd_p50 = float(grp["origin_effect_sd_p50"].median())
        dev_sd_p50 = float(grp["dev_effect_sd_p50"].median())

        # Recommended priors (strings for documentation).
        intercept_prior = f"Normal({int_mean:.3f}, {1.5 * int_sd:.3f})"
        # For HalfNormal, if alpha_p90 is NaN (no converged fits found alpha), fall back.
        if np.isfinite(alpha_p90) and alpha_p90 > 0:
            alpha_prior = f"HalfNormal({1.5 * alpha_p90:.3f})"
        else:
            alpha_prior = "HalfNormal(5.0)  [fallback: insufficient data]"
        if np.isfinite(origin_sd_p50) and origin_sd_p50 > 0:
            origin_sigma_prior = f"HalfNormal({1.5 * origin_sd_p50:.3f})"
        else:
            origin_sigma_prior = "HalfNormal(1.0)  [fallback: insufficient data]"
        if np.isfinite(dev_sd_p50) and dev_sd_p50 > 0:
            dev_sigma_prior = f"HalfNormal({1.5 * dev_sd_p50:.3f})"
        else:
            dev_sigma_prior = "HalfNormal(1.0)  [fallback: insufficient data]"

        rows.append(
            {
                "line": line,
                "n_converged": n,
                # Raw statistics
                "intercept_mean_avg": int_mean,
                "intercept_sd_avg": int_sd,
                "alpha_mean_avg": alpha_mean_avg,
                "alpha_sd_avg": alpha_sd_avg,
                "alpha_p90": alpha_p90,
                "origin_effect_sd_p50": origin_sd_p50,
                "dev_effect_sd_p50": dev_sd_p50,
                # Recommended prior strings
                "glm_intercept_prior": intercept_prior,
                "glm_alpha_prior": alpha_prior,
                "glm_origin_sigma_prior": origin_sigma_prior,
                "glm_dev_sigma_prior": dev_sigma_prior,
            }
        )

    out = pd.DataFrame(rows)
    out_path = cache_path("glm_priors_by_line.parquet")
    out.to_parquet(out_path, index=False)

    print(f"\nWrote {out_path}\n")
    print("=== Recommended GLM Priors by Line ===\n")
    display_cols = [
        "line",
        "n_converged",
        "glm_intercept_prior",
        "glm_alpha_prior",
        "glm_origin_sigma_prior",
        "glm_dev_sigma_prior",
    ]
    print(out[display_cols].to_string(index=False))
    print()
    print("=== Underlying Statistics ===\n")
    stat_cols = [
        "line",
        "intercept_mean_avg",
        "intercept_sd_avg",
        "alpha_mean_avg",
        "alpha_sd_avg",
        "alpha_p90",
        "origin_effect_sd_p50",
        "dev_effect_sd_p50",
    ]
    print(out[stat_cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
