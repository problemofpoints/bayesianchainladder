"""15_glm_prior_synthesis_v2.py — Synthesize M5_cal and MT5_cal posteriors into per-spec priors.

Reads:
  cache/m5cal_posteriors.parquet   (written by 13_glm_priors_m5cal.py)
  cache/mt5cal_posteriors.parquet  (written by 14_glm_priors_mt5cal.py)

Filters to converged fits (max_rhat < 1.1) and per line produces recommended
weakly-informative priors for the two winning specs.

M5_cal prior recommendations (gamma + log link, random-effect origin + calendar):
  Intercept:        Normal(mean_intercept, 1.5 * sd_intercept)
  Alpha (gamma):    HalfNormal(1.5 * p90_alpha_mean)
  1|origin sigma:   HalfNormal(1.5 * median_origin_sigma)
  1|calendar sigma: HalfNormal(1.5 * median_calendar_sigma)
  Spline SD:        HalfNormal(1.5 * median_spline_coef_sd)

MT5_cal prior recommendations (t + identity, loss-ratio, RE origin + calendar):
  Intercept:        Normal(mean_intercept, 1.5 * sd_intercept)  [LR scale]
  Sigma:            HalfNormal(1.5 * median_sigma_mean)
  Nu:               Gamma(alpha=2, beta=0.1)  [default unless nu_median < 5]
  1|origin sigma:   HalfNormal(1.5 * median_origin_sigma)
  1|calendar sigma: HalfNormal(1.5 * median_calendar_sigma)
  Spline SD:        HalfNormal(1.5 * median_spline_coef_sd)

Output: cache/glm_priors_per_spec.parquet
Run: uv run python references/prior-elicitation-2026/15_glm_prior_synthesis_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from _common import LINES, cache_path


def _synthesize_m5cal() -> pd.DataFrame:
    """Per-line M5_cal (gamma + log) prior recommendations."""
    in_path = cache_path("m5cal_posteriors.parquet")
    if not in_path.exists():
        print(f"WARNING: {in_path} not found. Run 13_glm_priors_m5cal.py first.")
        return pd.DataFrame()

    df = pd.read_parquet(in_path)
    print(f"M5_cal: Loaded {len(df)} rows from {in_path}")

    converged = df[df["max_rhat"] < 1.1].copy()
    total = len(df)
    ok = len(converged)
    print(f"  Converged: {ok}/{total} ({100*ok/total:.0f}%)")

    if ok == 0:
        print("  WARNING: No converged M5_cal fits — cannot synthesize priors.")
        return pd.DataFrame()

    rows = []
    for line, grp in converged.groupby("line"):
        n = len(grp)

        int_mean = float(grp["intercept_mean"].mean())
        int_sd = float(grp["intercept_sd"].mean())

        alpha_p90 = float(grp["alpha_mean"].quantile(0.90))
        alpha_mean_avg = float(grp["alpha_mean"].mean())

        origin_sigma_p50 = float(grp["origin_sigma_median"].median())
        calendar_sigma_p50 = float(grp["calendar_sigma_median"].median())
        spline_sd_p50 = float(grp["spline_coef_sd"].median())

        # --- Recommended prior strings ---
        intercept_prior = f"Normal({int_mean:.3f}, {1.5 * int_sd:.3f})"

        if np.isfinite(alpha_p90) and alpha_p90 > 1e-3:
            alpha_prior = f"HalfNormal({1.5 * alpha_p90:.3f})"
        elif np.isfinite(alpha_p90):
            alpha_prior = (
                f"HalfCauchy(1)  [alpha_p90={alpha_p90:.2e} near-zero; "
                f"use Bambi default or check link function]"
            )
        else:
            alpha_prior = "HalfNormal(5.0)  [fallback: insufficient data]"

        if np.isfinite(origin_sigma_p50) and origin_sigma_p50 > 0:
            origin_sigma_prior = f"HalfNormal({1.5 * origin_sigma_p50:.3f})"
        else:
            origin_sigma_prior = "HalfNormal(1.0)  [fallback]"

        if np.isfinite(calendar_sigma_p50) and calendar_sigma_p50 > 0:
            calendar_sigma_prior = f"HalfNormal({1.5 * calendar_sigma_p50:.3f})"
        else:
            calendar_sigma_prior = "HalfNormal(1.0)  [fallback]"

        if np.isfinite(spline_sd_p50) and spline_sd_p50 > 0:
            spline_prior = f"HalfNormal({1.5 * spline_sd_p50:.3f})"
        else:
            spline_prior = "HalfNormal(1.0)  [fallback]"

        rows.append({
            "line": line,
            "spec": "M5_cal",
            "n_converged": n,
            "intercept_mean_avg": int_mean,
            "intercept_sd_avg": int_sd,
            "alpha_mean_avg": alpha_mean_avg,
            "alpha_p90": alpha_p90,
            "origin_sigma_p50": origin_sigma_p50,
            "calendar_sigma_p50": calendar_sigma_p50,
            "spline_coef_sd_p50": spline_sd_p50,
            # Prior strings
            "intercept_prior": intercept_prior,
            "alpha_prior": alpha_prior,
            "origin_sigma_prior": origin_sigma_prior,
            "calendar_sigma_prior": calendar_sigma_prior,
            "spline_prior": spline_prior,
        })

    return pd.DataFrame(rows)


def _synthesize_mt5cal() -> pd.DataFrame:
    """Per-line MT5_cal (t + identity, loss-ratio) prior recommendations."""
    in_path = cache_path("mt5cal_posteriors.parquet")
    if not in_path.exists():
        print(f"WARNING: {in_path} not found. Run 14_glm_priors_mt5cal.py first.")
        return pd.DataFrame()

    df = pd.read_parquet(in_path)
    print(f"MT5_cal: Loaded {len(df)} rows from {in_path}")

    converged = df[df["max_rhat"] < 1.1].copy()
    total = len(df)
    ok = len(converged)
    print(f"  Converged: {ok}/{total} ({100*ok/total:.0f}%)")

    if ok == 0:
        print("  WARNING: No converged MT5_cal fits — cannot synthesize priors.")
        return pd.DataFrame()

    rows = []
    for line, grp in converged.groupby("line"):
        n = len(grp)

        int_mean = float(grp["intercept_mean"].mean())
        int_sd = float(grp["intercept_sd"].mean())

        sigma_median = float(grp["sigma_mean"].median())
        nu_median = float(grp["nu_mean"].median())

        origin_sigma_p50 = float(grp["origin_sigma_median"].median())
        calendar_sigma_p50 = float(grp["calendar_sigma_median"].median())
        spline_sd_p50 = float(grp["spline_coef_sd"].median())

        # --- Recommended prior strings ---
        intercept_prior = f"Normal({int_mean:.4f}, {1.5 * int_sd:.4f})"

        if np.isfinite(sigma_median) and sigma_median > 0:
            sigma_prior = f"HalfNormal({1.5 * sigma_median:.4f})"
        else:
            sigma_prior = "HalfNormal(0.05)  [fallback]"

        if np.isfinite(nu_median) and nu_median < 5:
            nu_prior = f"Gamma(alpha=2, beta=0.5)  [tighter; posterior nu_median={nu_median:.1f}]"
        else:
            nu_prior = f"Gamma(alpha=2, beta=0.1)  [Bambi default; nu_median={nu_median:.1f}]"

        if np.isfinite(origin_sigma_p50) and origin_sigma_p50 > 0:
            origin_sigma_prior = f"HalfNormal({1.5 * origin_sigma_p50:.4f})"
        else:
            origin_sigma_prior = "HalfNormal(0.1)  [fallback]"

        if np.isfinite(calendar_sigma_p50) and calendar_sigma_p50 > 0:
            calendar_sigma_prior = f"HalfNormal({1.5 * calendar_sigma_p50:.4f})"
        else:
            calendar_sigma_prior = "HalfNormal(0.1)  [fallback]"

        if np.isfinite(spline_sd_p50) and spline_sd_p50 > 0:
            spline_prior = f"HalfNormal({1.5 * spline_sd_p50:.4f})"
        else:
            spline_prior = "HalfNormal(0.1)  [fallback]"

        rows.append({
            "line": line,
            "spec": "MT5_cal",
            "n_converged": n,
            "intercept_mean_avg": int_mean,
            "intercept_sd_avg": int_sd,
            "sigma_median": sigma_median,
            "nu_median": nu_median,
            "origin_sigma_p50": origin_sigma_p50,
            "calendar_sigma_p50": calendar_sigma_p50,
            "spline_coef_sd_p50": spline_sd_p50,
            # Prior strings
            "intercept_prior": intercept_prior,
            "sigma_prior": sigma_prior,
            "nu_prior": nu_prior,
            "origin_sigma_prior": origin_sigma_prior,
            "calendar_sigma_prior": calendar_sigma_prior,
            "spline_prior": spline_prior,
        })

    return pd.DataFrame(rows)


def main() -> int:
    m5cal = _synthesize_m5cal()
    mt5cal = _synthesize_mt5cal()

    if m5cal.empty and mt5cal.empty:
        print("ERROR: Both M5_cal and MT5_cal posteriors are unavailable.")
        return 1

    # Combine into a single per-spec parquet.
    combined = pd.concat([m5cal, mt5cal], ignore_index=True)
    out_path = cache_path("glm_priors_per_spec.parquet")
    combined.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path}")

    # --- Pretty-print M5_cal priors ---
    if not m5cal.empty:
        print("\n=== M5_cal Prior Recommendations (gamma + log link, RE origin + calendar) ===\n")
        m5_cols = [
            "line", "n_converged",
            "intercept_prior", "alpha_prior",
            "origin_sigma_prior", "calendar_sigma_prior", "spline_prior",
        ]
        avail = [c for c in m5_cols if c in m5cal.columns]
        print(m5cal[avail].to_string(index=False))

        print("\n=== M5_cal Underlying Statistics ===\n")
        stat_cols = [
            "line", "intercept_mean_avg", "intercept_sd_avg",
            "alpha_mean_avg", "alpha_p90",
            "origin_sigma_p50", "calendar_sigma_p50", "spline_coef_sd_p50",
        ]
        avail_stat = [c for c in stat_cols if c in m5cal.columns]
        print(m5cal[avail_stat].to_string(index=False))

    # --- Pretty-print MT5_cal priors ---
    if not mt5cal.empty:
        print("\n=== MT5_cal Prior Recommendations (t + identity, loss-ratio, RE origin + calendar) ===\n")
        mt5_cols = [
            "line", "n_converged",
            "intercept_prior", "sigma_prior", "nu_prior",
            "origin_sigma_prior", "calendar_sigma_prior", "spline_prior",
        ]
        avail = [c for c in mt5_cols if c in mt5cal.columns]
        print(mt5cal[avail].to_string(index=False))

        print("\n=== MT5_cal Underlying Statistics ===\n")
        stat_cols = [
            "line", "intercept_mean_avg", "intercept_sd_avg",
            "sigma_median", "nu_median",
            "origin_sigma_p50", "calendar_sigma_p50", "spline_coef_sd_p50",
        ]
        avail_stat = [c for c in stat_cols if c in mt5cal.columns]
        print(mt5cal[avail_stat].to_string(index=False))

    print("\nNOTE: M5_cal priors are for gamma+log fits; MT5_cal priors are for")
    print("t+identity loss-ratio fits. LOO is NOT directly comparable across specs")
    print("without Jacobian correction (see sum_log_ep_obs in the v2 fit cache).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
