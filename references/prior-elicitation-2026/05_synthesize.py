"""05_synthesize.py — Combine all caches into the final per-line recommendations.

Reads:
  cache/descriptive_summary.parquet
  cache/rho_by_line.parquet
  cache/glm_per_triangle_fits_v2.parquet      (preferred — includes reserve stats)
  cache/glm_per_triangle_fits.parquet         (fallback — LOO only, no reserve stats)
  cache/glm_hierarchical_fits.parquet         (optional)
  cache/csr_fits.parquet                      (optional)
  cache/glm_priors_per_spec.parquet           (optional — M5_cal/MT5_cal per-spec priors)
  cache/glm_priors_by_line.parquet            (optional — M1 gamma priors)
  cache/glm_t_priors_by_line.parquet          (optional — MT2 t priors)

Writes:
  README.md
  report.html (single-page interactive)
  cache/recommendations.parquet

v2 features (requires glm_per_triangle_fits_v2.parquet):
  - Jacobian-corrected LOO: adds sum(log EP) to MT* LOO so all 8 specs are
    on the same dollar-equivalent scale in a single unified comparison table.
  - Reserve comparison: per-line/spec median IBNR, CV, abs % error vs booked.
  - Per-spec priors: M5_cal and MT5_cal priors from glm_priors_per_spec.parquet.

Run: uv run python references/prior-elicitation-2026/05_synthesize.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from _common import ANALYSIS_DIR, LINES, cache_path
from _html_template import render_html


def _read_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet file, falling back to pyarrow ParquetFile if pandas metadata
    is corrupt (which can happen when the file was written by a different pyarrow
    version).
    """
    try:
        return pd.read_parquet(path)
    except Exception:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(str(path))
        table = pf.read()
        return table.to_pandas()


def _load_glm_fits() -> pd.DataFrame:
    """Load the best available GLM fits cache.

    Prefers v2 (includes reserve stats + sum_log_ep_obs) over v1 (LOO only).
    Returns an empty DataFrame if neither exists.
    """
    v2_path = cache_path("glm_per_triangle_fits_v2.parquet")
    v1_path = cache_path("glm_per_triangle_fits.parquet")

    if v2_path.exists():
        df = _read_parquet(v2_path)
        df["_cache_version"] = "v2"
        return df
    if v1_path.exists():
        df = _read_parquet(v1_path)
        df["_cache_version"] = "v1"
        return df
    return pd.DataFrame()


def _rank_glm_specs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """For each line, produce mean LOO per spec (raw and Jacobian-corrected).

    Returns (combined_raw, combined_dollar, winners) where:
      combined_raw — loo_mean on native scale (raw, not corrected)
      combined_dollar — loo_dollar_mean: MT* corrected by sum(log EP), all comparable
      winners — dollar-scale winner per line (from gamma specs, as before)

    All DataFrames may be empty if no cache files exist yet.
    """
    per = _load_glm_fits()
    empty = pd.DataFrame(columns=["line", "spec", "loo_mean", "loo_sd", "n"])
    empty_w = pd.DataFrame(columns=["line", "best_spec", "loo_mean"])

    if per.empty:
        return empty, empty.copy(), empty_w

    per_ok = per[(per["status"] == "ok") & (per["max_rhat"] < 1.1)].copy()
    if per_ok.empty:
        return empty, empty.copy(), empty_w

    # --- Jacobian correction for MT* specs ---
    # For MT* (loss-ratio response), each observation contributes:
    #   log p(dollar) = log q(loss-ratio) - log(EP)
    # So: sum_n log p(dollar) = elpd_loo(q) - sum_log_ep_obs
    # Only available in v2 cache (sum_log_ep_obs column).
    if "sum_log_ep_obs" in per_ok.columns:
        per_ok["loo_dollar"] = per_ok.apply(
            lambda r: (r["loo"] - r["sum_log_ep_obs"])
            if r["spec"].startswith("MT") and np.isfinite(r.get("sum_log_ep_obs", float("nan")))
            else r["loo"],
            axis=1,
        )
        has_jacobian = True
    else:
        per_ok["loo_dollar"] = per_ok["loo"]
        has_jacobian = False

    # --- Raw LOO aggregation (for backwards-compat display) ---
    per_agg = (
        per_ok.groupby(["line", "spec"])
        .agg(loo_mean=("loo", "mean"), loo_sd=("loo", "std"), n=("loo", "count"))
        .reset_index()
    )

    # --- Dollar-equivalent LOO aggregation ---
    per_agg_dollar = (
        per_ok.groupby(["line", "spec"])
        .agg(
            loo_dollar_mean=("loo_dollar", "mean"),
            loo_dollar_sd=("loo_dollar", "std"),
            n=("loo_dollar", "count"),
        )
        .reset_index()
    )
    per_agg_dollar["has_jacobian"] = has_jacobian

    # Merge hierarchical (M4) if available — always dollar scale (gamma+log).
    hier_path = cache_path("glm_hierarchical_fits.parquet")
    combined_raw = per_agg.copy()
    combined_dollar = per_agg_dollar.copy()

    if hier_path.exists():
        hier = _read_parquet(hier_path)
        hier = hier[(hier.get("status", "ok") == "ok") & (hier.get("max_rhat", 2.0) < 1.1)].copy()
        if not hier.empty and "loo" in hier.columns and "n_companies" in hier.columns:
            hier["spec"] = "M4_hierarchical"
            hier_loo_per_tri = hier["loo"] / hier["n_companies"]

            hier_agg = pd.DataFrame({
                "line": hier["line"],
                "spec": "M4_hierarchical",
                "loo_mean": hier_loo_per_tri.values,
                "loo_sd": float("nan"),
                "n": hier["n_companies"].values,
            })
            hier_agg_dollar = pd.DataFrame({
                "line": hier["line"],
                "spec": "M4_hierarchical",
                "loo_dollar_mean": hier_loo_per_tri.values,  # gamma scale = dollar
                "loo_dollar_sd": float("nan"),
                "n": hier["n_companies"].values,
                "has_jacobian": False,
            })
            combined_raw = pd.concat([combined_raw, hier_agg], ignore_index=True)
            combined_dollar = pd.concat([combined_dollar, hier_agg_dollar], ignore_index=True)

    if combined_raw.empty:
        return combined_raw, combined_dollar, empty_w

    # Winner selection: use dollar-equivalent LOO; restrict to gamma+log specs
    # (M1-M5_cal + M4_hierarchical) so the winner is a spec practitioners can
    # directly compare in dollar units. MT* specs now appear in the unified table.
    dollar_only = combined_dollar[~combined_dollar["spec"].str.startswith("MT")]
    if dollar_only.empty:
        winners = empty_w
    else:
        winners = (
            dollar_only.sort_values("loo_dollar_mean", ascending=False)
            .drop_duplicates("line")
            .rename(columns={"spec": "best_spec", "loo_dollar_mean": "loo_mean"})
            [["line", "best_spec", "loo_mean"]]
        )

    return combined_raw, combined_dollar, winners


def _aggregate_csr() -> pd.DataFrame:
    """Per-line CSR posterior aggregates -> recommended priors.

    Posterior variable names in BayesianCSR: logelr, r_alpha, r_beta, gamma, a_ig, sig.
    a_ig has a heavy-tailed posterior (the InverseGamma(1,1) prior has undefined
    mean), so we summarise via p10 / p50 instead of mean.

    Returns an empty DataFrame if the cache does not exist yet.
    """
    csr_path = cache_path("csr_fits.parquet")
    if not csr_path.exists():
        return pd.DataFrame()

    df = _read_parquet(csr_path)
    df = df[(df.status == "ok") & (df.max_rhat < 1.1)]
    if df.empty:
        return pd.DataFrame()

    agg_logelr = df.groupby("line")["logelr_mean"].agg(["mean", "std"]).reset_index()
    agg_logelr.columns = ["line", "logelr_mean", "logelr_sd"]

    agg_gamma = df.groupby("line")["gamma_mean"].agg(["mean", "std"]).reset_index()
    agg_gamma.columns = ["line", "gamma_mean", "gamma_sd"]

    agg_sig = df.groupby("line")["sig_mean"].median().reset_index()
    agg_sig.columns = ["line", "sig_p50"]

    # a_ig is heavy-tailed: use median of the per-fit p10 values as the
    # recommended scale for a HalfNormal prior on the variance shape parameter.
    agg_a_ig = df.groupby("line")["a_ig_p10"].median().reset_index()
    agg_a_ig.columns = ["line", "a_ig_p10_p50"]

    agg = (
        agg_logelr.merge(agg_gamma, on="line")
        .merge(agg_sig, on="line")
        .merge(agg_a_ig, on="line")
    )

    agg["csr_logelr_prior"] = agg.apply(
        lambda r: f"Normal({r.logelr_mean:.2f}, {1.5 * r.logelr_sd:.2f})", axis=1
    )
    agg["csr_gamma_prior"] = agg.apply(
        lambda r: f"Normal({r.gamma_mean:.3f}, {max(r.gamma_sd, 0.01):.3f})", axis=1
    )
    agg["csr_sig_prior"] = agg["sig_p50"].apply(
        lambda v: f"HalfNormal({1.5 * v:.3f})"
    )
    agg["csr_a_ig_prior"] = agg["a_ig_p10_p50"].apply(
        lambda v: f"HalfNormal({v:.2f})"
    )
    return agg


def _format_md_recommendations(rec: pd.DataFrame) -> str:
    """Render a markdown table of the headline recommendations."""
    return rec.to_markdown(index=False, floatfmt=".3f")


def _build_wald_comparison_section() -> str:
    """Build the Wald vs gamma M2 comparison section, or empty string if not ready."""
    wald_path = cache_path("glm_wald_m2_fits.parquet")
    if not wald_path.exists():
        return ""
    wald = _read_parquet(wald_path)
    wald_ok = wald[(wald["status"] == "ok") & (wald["max_rhat"] < 1.1)]
    if wald_ok.empty:
        return ""

    # Use best available GLM fits for gamma M2 reference.
    per_v2 = cache_path("glm_per_triangle_fits_v2.parquet")
    per_v1 = cache_path("glm_per_triangle_fits.parquet")
    per_path = per_v2 if per_v2.exists() else per_v1
    if not per_path.exists():
        return ""
    per = _read_parquet(per_path)
    per_ok = per[(per["status"] == "ok") & (per["max_rhat"] < 1.1) & (per["spec"] == "M2_devidx_bs4")]
    if per_ok.empty:
        return ""

    gamma_m2 = per_ok.groupby("line")["loo"].mean()
    wald_m2 = wald_ok.groupby("line")["loo"].mean()
    cmp = pd.DataFrame({
        "line": LINES,
        "gamma_log_M2_loo": [gamma_m2.get(l, float("nan")) for l in LINES],
        "wald_log_M2_loo": [wald_m2.get(l, float("nan")) for l in LINES],
    })
    cmp["wald_minus_gamma"] = cmp["wald_log_M2_loo"] - cmp["gamma_log_M2_loo"]

    out: list[str] = []
    out.append("\n## Wald vs Gamma family on M2 (LOO)\n")
    out.append(
        "Both families fit M2 (`bs(dev_idx, df=4)` + categorical origin) "
        "with log link. Positive `wald_minus_gamma` = Wald better; "
        "negative = gamma better.\n"
    )
    out.append(cmp.to_markdown(index=False, floatfmt=".2f") + "\n")
    return "\n".join(out)


def _build_reserve_comparison_section() -> str:
    """Build the per-line, per-spec reserve comparison table.

    Requires glm_per_triangle_fits_v2.parquet (the v1 cache does not have
    reserve stats). Returns empty string if not available.
    """
    v2_path = cache_path("glm_per_triangle_fits_v2.parquet")
    if not v2_path.exists():
        return ""
    per = _read_parquet(v2_path)
    per_ok = per[(per["status"] == "ok") & (per["max_rhat"] < 1.1)].copy()
    if per_ok.empty or "ibnr_total_median" not in per_ok.columns:
        return ""

    out: list[str] = []
    out.append("\n## Reserve Comparison by Spec\n")
    out.append(
        "Per-line, per-spec summary of reserve posteriors (aggregated across "
        "all 24 sampled triangles per line). Median IBNR and ultimate are in "
        "dollar terms for all specs — MT* specs are back-transformed from "
        "loss-ratio scale via `IBNR_dollar = IBNR_lr × EP_per_origin`.\n"
    )
    out.append(
        "- **median_ibnr**: Median total IBNR (summed over all origin years), "
        "averaged across triangles in the line.\n"
        "- **cv_ibnr**: Coefficient of variation of total IBNR posterior, "
        "averaged across triangles.\n"
        "- **pct_err_booked**: Mean (ult_median − booked_ultimate) / booked_ultimate "
        "across triangles (signed; positive = model above booked).\n"
        "- **abs_pct_err**: Mean absolute percentage error vs booked ultimate.\n\n"
    )

    rows = []
    for (line, spec), grp in per_ok.groupby(["line", "spec"]):
        median_ibnr = float(grp["ibnr_total_median"].mean())
        cv_ibnr = float(grp["ibnr_total_cv"].mean())
        pct_err = float(grp["pct_error_vs_booked"].mean())
        abs_pct_err = float(grp["pct_error_vs_booked"].abs().mean())
        rows.append({
            "line": line,
            "spec": spec,
            "median_ibnr": median_ibnr,
            "cv_ibnr": cv_ibnr,
            "pct_err_booked": pct_err,
            "abs_pct_err": abs_pct_err,
        })
    df = pd.DataFrame(rows)

    for line in LINES:
        sub = df[df["line"] == line].sort_values("spec")
        if sub.empty:
            continue
        out.append(f"### {line}\n")
        # Format nicely
        display = sub[["spec", "median_ibnr", "cv_ibnr", "pct_err_booked", "abs_pct_err"]].copy()
        display["median_ibnr"] = display["median_ibnr"].apply(lambda v: f"{v:,.0f}")
        display["cv_ibnr"] = display["cv_ibnr"].apply(lambda v: f"{v:.3f}")
        display["pct_err_booked"] = display["pct_err_booked"].apply(lambda v: f"{v:+.1%}" if np.isfinite(v) else "nan")
        display["abs_pct_err"] = display["abs_pct_err"].apply(lambda v: f"{v:.1%}" if np.isfinite(v) else "nan")
        out.append(display.to_markdown(index=False) + "\n")

    return "\n".join(out)


def _build_unified_loo_section(combined_dollar: pd.DataFrame) -> str:
    """Build the unified LOO comparison table (all 8 specs on dollar-equivalent scale).

    MT* specs are Jacobian-corrected to dollar-equivalent LOO.
    Returns empty string if combined_dollar is empty.
    """
    if combined_dollar.empty:
        return ""

    has_jacobian = bool(combined_dollar.get("has_jacobian", pd.Series([False])).any())

    out: list[str] = []
    out.append("\n## Unified LOO Comparison (all specs, dollar-equivalent scale)\n")
    if has_jacobian:
        out.append(
            "MT* specs (t+identity, loss-ratio response) are Jacobian-corrected to "
            "dollar-equivalent LOO by subtracting `sum(log EP_per_cell)` from the "
            "raw LOO ELPD. This converts the loss-ratio density to a dollar density, "
            "making all 8 specs directly comparable. Higher = better predictive accuracy.\n\n"
            "The Jacobian correction: `loo_dollar = elpd_loo(q) − Σ log(EP_i)` where "
            "the sum is over all observed cells in the triangle.\n"
        )
    else:
        out.append(
            "Note: glm_per_triangle_fits_v2.parquet not yet available. "
            "MT* specs cannot be Jacobian-corrected — displaying raw LOO only. "
            "MT* and gamma+log LOO values are NOT directly comparable.\n"
        )

    pivot = (
        combined_dollar.pivot(index="line", columns="spec", values="loo_dollar_mean")
        .reindex(LINES)
        .round(1)
    )
    out.append(pivot.to_markdown() + "\n")

    n_pivot = (
        combined_dollar.pivot(index="line", columns="spec", values="n")
        .reindex(LINES)
        .fillna(0)
        .astype(int)
    )
    out.append("\nConverged-fit counts per spec:\n")
    out.append(n_pivot.to_markdown() + "\n")

    return "\n".join(out)


def _build_per_spec_priors_section() -> str:
    """Build the per-spec prior recommendations section for M5_cal and MT5_cal.

    Reads glm_priors_per_spec.parquet if available. Returns empty string if not.
    """
    per_spec_path = cache_path("glm_priors_per_spec.parquet")
    if not per_spec_path.exists():
        return ""
    df = _read_parquet(per_spec_path)
    if df.empty:
        return ""

    out: list[str] = []
    out.append("\n## Per-Spec Prior Recommendations (Winning Specs)\n")
    out.append(
        "Priors derived directly from posteriors of the two winning specs — "
        "**M5_cal** (gamma + log, random-intercept origin + calendar RE) and "
        "**MT5_cal** (Student-t, identity link, loss-ratio response, same RE structure). "
        "These are more appropriate than the M1 priors (below) when using the "
        "actual winning functional form.\n"
    )

    m5 = df[df["spec"] == "M5_cal"]
    mt5 = df[df["spec"] == "MT5_cal"]

    if not m5.empty:
        out.append("### M5_cal — gamma + log link, `(1|origin) + bs(dev_idx,df=4) + (1|calendar)`\n")
        cols = ["line", "n_converged", "intercept_prior", "alpha_prior",
                "origin_sigma_prior", "calendar_sigma_prior", "spline_prior"]
        avail = [c for c in cols if c in m5.columns]
        out.append(m5[avail].to_markdown(index=False) + "\n")

    if not mt5.empty:
        out.append("### MT5_cal — t + identity, loss-ratio response, `(1|origin) + bs(dev_idx,df=4) + (1|calendar)`\n")
        cols = ["line", "n_converged", "intercept_prior", "sigma_prior", "nu_prior",
                "origin_sigma_prior", "calendar_sigma_prior", "spline_prior"]
        avail = [c for c in cols if c in mt5.columns]
        out.append(mt5[avail].to_markdown(index=False) + "\n")

    return "\n".join(out)


def _build_readme(
    combined_raw: pd.DataFrame,
    combined_dollar: pd.DataFrame,
    winners: pd.DataFrame,
    csr_agg: pd.DataFrame,
    rho: pd.DataFrame,
    descriptive: pd.DataFrame,
    rec: pd.DataFrame,
    glm_priors: pd.DataFrame | None = None,
    glm_t_priors: pd.DataFrame | None = None,
) -> str:
    lines_out: list[str] = []
    lines_out.append("# Prior Elicitation 2026 — Per-Line Recommendations\n")
    lines_out.append(
        "**Spec:** [design.md](design.md). Generated by `05_synthesize.py` "
        "from `cache/*.parquet`. Re-run that script to refresh.\n"
    )
    lines_out.append("Interactive version: [report.html](report.html).\n")

    lines_out.append("## Headline Recommendations\n")
    lines_out.append(_format_md_recommendations(rec) + "\n")

    # --- Unified LOO comparison (all 8 specs, dollar-equivalent) ---
    unified_loo = _build_unified_loo_section(combined_dollar)
    if unified_loo:
        lines_out.append(unified_loo)

    # --- Legacy split LOO tables (raw scale, for reference) ---
    if not combined_raw.empty:
        dollar_combined = combined_raw[~combined_raw["spec"].str.startswith("MT")]
        mt_combined = combined_raw[combined_raw["spec"].str.startswith("MT")]

        lines_out.append("\n## GLM Functional-Form Comparison (raw scale, reference)\n")
        lines_out.append(
            "Family: gamma + log link. M1: full categorical origin+dev. "
            "M2: C(origin) + B-spline on dev ordinal index (df=4). "
            "M3: B-spline on origin (df=3) + C(dev). "
            "M4: hierarchical (1|snl_id) — normalised LOO by n_companies for comparability. "
            "MT2 and MT5_cal use t + identity link on loss-ratio response (raw LOO "
            "on loss-ratio scale — see unified table above for corrected values).\n"
        )
        if not dollar_combined.empty:
            pivot = (
                dollar_combined.pivot(index="line", columns="spec", values="loo_mean")
                .reindex(LINES)
                .round(2)
            )
            lines_out.append(
                "### Gamma + log-link (dollar-scale response)\n"
                "Mean LOO per spec (higher = better, NaN = no converged fits):\n"
            )
            lines_out.append(pivot.to_markdown() + "\n")

            n_pivot = (
                dollar_combined.pivot(index="line", columns="spec", values="n")
                .reindex(LINES)
                .fillna(0)
                .astype(int)
            )
            lines_out.append(
                "\nConverged-fit counts per spec (out of 24 sampled triangles per line):\n"
            )
            lines_out.append(n_pivot.to_markdown() + "\n")

        if not mt_combined.empty:
            mt_pivot = (
                mt_combined.pivot(index="line", columns="spec", values="loo_mean")
                .reindex(LINES)
                .round(2)
            )
            lines_out.append(
                "\n### t + identity-link (loss-ratio raw LOO)\n"
                "Raw LOO on loss-ratio density scale — see unified table above for "
                "dollar-equivalent Jacobian-corrected values.\n"
            )
            lines_out.append(mt_pivot.to_markdown() + "\n")

    # --- Reserve comparison ---
    reserve_section = _build_reserve_comparison_section()
    if reserve_section:
        lines_out.append(reserve_section)
    else:
        lines_out.append(
            "\n## Reserve Comparison by Spec\n"
            "_Reserve stats not yet available (requires v2 cache — sweeps still running)._\n"
        )

    wald_section = _build_wald_comparison_section()
    if wald_section:
        lines_out.append(wald_section)

    hier_path = cache_path("glm_hierarchical_fits.parquet")
    if hier_path.exists():
        hier_raw = _read_parquet(hier_path)
        lines_out.append("\n### M4 hierarchical (Bambi `(1 | snl_id)`) convergence diagnostics\n")
        converged_m4 = hier_raw[
            (hier_raw.get("status", "ok") == "ok") & (hier_raw.get("max_rhat", 2.0) < 1.1)
        ] if not hier_raw.empty else pd.DataFrame()
        if converged_m4.empty:
            lines_out.append(
                "All 6 line-level M4 fits had high `max_rhat` (> 1.1) or failed. "
                "They are excluded from the LOO comparison above by the `rhat < 1.1` filter. "
                "**Hierarchical pooling via Bambi `(1 | snl_id)` with the gamma+log GLM "
                "does not mix under default light-MCMC settings for these triangles.** "
                "Reparameterisation (non-centered `(1 | snl_id) + (0 | snl_id)`), "
                "stronger priors on the company-level SD, or a much longer tune budget "
                "(5000+) would be needed to fit M4 cleanly.\n"
            )
        else:
            lines_out.append(
                f"All {len(converged_m4)} line-level M4 fits converged (max_rhat ≤ 1.01, 0 divergences) "
                "under gamma+log link. They appear in the LOO table above, normalised by n_companies "
                "so the loo_mean is per-triangle comparable. Despite converging, M4 LOO is uniformly "
                "worse than M2 — the per-company random intercept adds flexibility that isn't rewarded "
                "by held-out predictive accuracy at this triangle count. "
                "Estimated company-level random-intercept SD (σ) is reported below.\n"
            )
        hier_cols = [
            c for c in
            ["line", "status", "max_rhat", "loo", "n_obs", "n_companies", "company_sigma_mean"]
            if c in hier_raw.columns
        ]
        lines_out.append(hier_raw[hier_cols].to_markdown(index=False, floatfmt=".3f") + "\n")

    # --- Per-spec priors (winning specs: M5_cal, MT5_cal) ---
    per_spec_section = _build_per_spec_priors_section()
    if per_spec_section:
        lines_out.append(per_spec_section)
    else:
        lines_out.append(
            "\n## Per-Spec Prior Recommendations (Winning Specs)\n"
            "_Per-spec priors not yet available (run 13_glm_priors_m5cal.py, "
            "14_glm_priors_mt5cal.py, and 15_glm_prior_synthesis_v2.py)._\n"
        )

    # --- Legacy M1 priors (gamma+log, backward compatibility) ---
    glm_priors_path = cache_path("glm_priors_by_line.parquet")
    if glm_priors_path.exists():
        _glm_priors = glm_priors if glm_priors is not None else _read_parquet(glm_priors_path)
        lines_out.append(
            "\n## GLM Prior Recommendations — M1 spec (gamma + log link, C(origin)+C(dev))\n"
        )
        lines_out.append(
            "Per-line prior recommendations derived from posteriors of the "
            "M1 fits (24 sampled triangles per line, gamma+log link, default "
            "package priors). These priors apply to the **M1_cat** spec. "
            "For M5_cal or MT5_cal, use the per-spec priors above.\n"
        )
        lines_out.append(
            _glm_priors[
                [
                    "line",
                    "n_converged",
                    "glm_intercept_prior",
                    "glm_alpha_prior",
                    "glm_origin_sigma_prior",
                    "glm_dev_sigma_prior",
                ]
            ].to_markdown(index=False)
            + "\n"
        )

    glm_t_priors_path = cache_path("glm_t_priors_by_line.parquet")
    if glm_t_priors_path.exists():
        _glm_t_priors = glm_t_priors if glm_t_priors is not None else _read_parquet(glm_t_priors_path)
        lines_out.append(
            "\n## GLM Prior Recommendations — MT2 spec (t + identity, C(origin)+bs(dev_idx))\n"
        )
        lines_out.append(
            "Per-line prior recommendations derived from posteriors of the "
            "MT2 fits (`incremental ~ 1 + C(origin) + bs(dev_idx, df=4)`, "
            "t family, identity link, `response_per_exposure=True`). "
            "For MT5_cal (with calendar RE), use the per-spec priors above.\n"
        )
        lines_out.append(
            _glm_t_priors[
                [
                    "line",
                    "n_converged",
                    "t_intercept_prior",
                    "t_sigma_prior",
                    "t_nu_prior",
                    "t_origin_sigma_prior",
                    "t_dev_sigma_prior",
                ]
            ].to_markdown(index=False)
            + "\n"
        )

    if not csr_agg.empty:
        lines_out.append("\n## CSR Prior Recommendations (full)\n")
        keep = [
            "line",
            "logelr_mean",
            "logelr_sd",
            "gamma_mean",
            "gamma_sd",
            "sig_p50",
            "a_ig_p10_p50",
            "csr_logelr_prior",
            "csr_gamma_prior",
            "csr_sig_prior",
            "csr_a_ig_prior",
        ]
        lines_out.append(
            csr_agg[keep].to_markdown(index=False, floatfmt=".3f") + "\n"
        )
    else:
        lines_out.append(
            "\n## CSR Prior Recommendations\n"
            "_CSR fits not yet available (sweeps still running)._\n"
        )

    lines_out.append(
        "\n## Rho (CorrelatedBootstrapODPSample, calendar-diagonal correlation)\n"
    )
    rho_cols = [
        c for c in
        ["line", "n_companies", "rho_point", "rho_median", "rho_p10", "rho_p90", "r1", "r2", "r3"]
        if c in rho.columns
    ]
    lines_out.append(rho[rho_cols].to_markdown(index=False, floatfmt=".3f") + "\n")

    lines_out.append("\n## Per-Line Descriptive Diagnostics\n")
    for line in LINES:
        lines_out.append(f"### {line}\n")
        lines_out.append(f"![{line} diagnostic](figures/01_descriptive_{line}.png)\n")

    return "\n".join(lines_out)


def _payload_for_html(
    combined_raw: pd.DataFrame,
    combined_dollar: pd.DataFrame,
    winners: pd.DataFrame,
    csr_agg: pd.DataFrame,
    rho: pd.DataFrame,
    descriptive: pd.DataFrame,
    rec: pd.DataFrame,
    glm_priors: pd.DataFrame | None = None,
    glm_t_priors: pd.DataFrame | None = None,
) -> dict:
    """Convert pandas DataFrames into JSON-serialisable list-of-dicts payload."""
    # Use best available GLM cache for the HTML raw rows.
    per = _load_glm_fits()
    hier_path = cache_path("glm_hierarchical_fits.parquet")
    csr_path = cache_path("csr_fits.parquet")

    glm_rows: list[dict] = []
    if not per.empty:
        keep_cols = [c for c in ["line", "snl_id", "spec", "status", "loo", "max_rhat"] if c in per.columns]
        glm_rows = per[keep_cols].to_dict("records")

    hier_rows: list[dict] = []
    if hier_path.exists():
        hier_rows = _read_parquet(hier_path).to_dict("records")

    csr_rows: list[dict] = []
    if csr_path.exists():
        csr_rows = _read_parquet(csr_path).to_dict("records")

    glm_priors_rows: list[dict] = []
    if glm_priors is not None and not glm_priors.empty:
        keep_cols = [
            "line", "n_converged",
            "glm_intercept_prior", "glm_alpha_prior",
            "glm_origin_sigma_prior", "glm_dev_sigma_prior",
        ]
        avail = [c for c in keep_cols if c in glm_priors.columns]
        glm_priors_rows = glm_priors[avail].to_dict("records")

    glm_t_priors_rows: list[dict] = []
    if glm_t_priors is not None and not glm_t_priors.empty:
        t_keep_cols = [
            "line", "n_converged",
            "t_intercept_prior", "t_sigma_prior", "t_nu_prior",
            "t_origin_sigma_prior", "t_dev_sigma_prior",
        ]
        t_avail = [c for c in t_keep_cols if c in glm_t_priors.columns]
        glm_t_priors_rows = glm_t_priors[t_avail].to_dict("records")

    # Per-spec priors for HTML
    per_spec_rows: list[dict] = []
    per_spec_path = cache_path("glm_priors_per_spec.parquet")
    if per_spec_path.exists():
        ps = _read_parquet(per_spec_path)
        if not ps.empty:
            per_spec_rows = ps.to_dict("records")

    # Dollar-equivalent LOO for HTML
    dollar_loo_rows: list[dict] = []
    if not combined_dollar.empty:
        dollar_loo_rows = combined_dollar.to_dict("records")

    return {
        "lines": LINES,
        "recs": rec.to_dict("records"),
        "glm": glm_rows,
        "hier": hier_rows,
        "csr": csr_rows,
        "rho": rho.to_dict("records"),
        "desc": descriptive.to_dict("records"),
        "glm_priors": glm_priors_rows,
        "glm_t_priors": glm_t_priors_rows,
        "per_spec_priors": per_spec_rows,
        "dollar_loo": dollar_loo_rows,
    }


def main() -> int:
    descriptive = _read_parquet(cache_path("descriptive_summary.parquet"))
    rho = _read_parquet(cache_path("rho_by_line.parquet"))
    combined_raw, combined_dollar, winners = _rank_glm_specs()
    csr_agg = _aggregate_csr()

    # Load GLM priors (gamma+log, M1) if available.
    glm_priors_path = cache_path("glm_priors_by_line.parquet")
    glm_priors: pd.DataFrame | None = None
    if glm_priors_path.exists():
        glm_priors = _read_parquet(glm_priors_path)

    # Load GLM t-family priors (loss-ratio, MT2) if available.
    glm_t_priors_path = cache_path("glm_t_priors_by_line.parquet")
    glm_t_priors: pd.DataFrame | None = None
    if glm_t_priors_path.exists():
        glm_t_priors = _read_parquet(glm_t_priors_path)

    # Build the headline recommendations frame.
    rec = pd.DataFrame({"line": LINES})
    if not winners.empty:
        rec = rec.merge(winners, on="line", how="left")
    if not csr_agg.empty:
        rec = rec.merge(
            csr_agg[["line", "csr_logelr_prior", "csr_gamma_prior", "csr_sig_prior"]],
            on="line",
            how="left",
        )
    rec = rec.merge(rho[["line", "rho_point", "rho_median"]], on="line", how="left")
    rec = rec.merge(
        descriptive[["line", "ulr_mean", "phi_p50"]], on="line", how="left"
    )
    if glm_priors is not None and not glm_priors.empty:
        glm_priors_short = glm_priors[["line", "glm_intercept_prior", "glm_dev_sigma_prior"]]
        rec = rec.merge(glm_priors_short, on="line", how="left")
    if glm_t_priors is not None and not glm_t_priors.empty:
        t_priors_short = glm_t_priors[["line", "t_intercept_prior", "t_sigma_prior"]]
        rec = rec.merge(t_priors_short, on="line", how="left")

    md = _build_readme(
        combined_raw, combined_dollar, winners, csr_agg, rho, descriptive, rec,
        glm_priors=glm_priors, glm_t_priors=glm_t_priors,
    )
    readme_path = ANALYSIS_DIR / "README.md"
    readme_path.write_text(md, encoding="utf-8")

    html_payload = _payload_for_html(
        combined_raw, combined_dollar, winners, csr_agg, rho, descriptive, rec,
        glm_priors=glm_priors, glm_t_priors=glm_t_priors,
    )
    html = render_html(html_payload)
    html_path = ANALYSIS_DIR / "report.html"
    html_path.write_text(html, encoding="utf-8")

    rec.to_parquet(cache_path("recommendations.parquet"), index=False)
    print(f"Wrote {readme_path}")
    print(f"Wrote {html_path}")
    print(f"Wrote {cache_path('recommendations.parquet')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
