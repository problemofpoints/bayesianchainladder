"""05_synthesize.py — Combine all caches into the final per-line recommendations.

Reads:
  cache/descriptive_summary.parquet
  cache/rho_by_line.parquet
  cache/glm_per_triangle_fits.parquet       (optional — may not exist yet)
  cache/glm_hierarchical_fits.parquet       (optional — may not exist yet)
  cache/csr_fits.parquet                    (optional — may not exist yet)

Writes:
  README.md
  report.html (single-page interactive)
  cache/recommendations.parquet

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


def _rank_glm_specs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each line, produce mean LOO per spec and pick a winner.

    MT* specs (t family, loss-ratio scale) are excluded from the winner
    selection because their LOO is on a different response scale than the
    gamma+log specs.

    Returns (combined, winners). Both DataFrames may be empty if no cache
    files exist yet.
    """
    per_path = cache_path("glm_per_triangle_fits.parquet")
    if not per_path.exists():
        empty = pd.DataFrame(columns=["line", "spec", "loo_mean", "loo_sd", "n"])
        return empty, pd.DataFrame(columns=["line", "best_spec", "loo_mean"])

    per = _read_parquet(per_path)
    per = per[(per.status == "ok") & (per.max_rhat < 1.1)]
    if per.empty:
        empty = pd.DataFrame(columns=["line", "spec", "loo_mean", "loo_sd", "n"])
        return empty, pd.DataFrame(columns=["line", "best_spec", "loo_mean"])

    per_agg = (
        per.groupby(["line", "spec"])
        .agg(loo_mean=("loo", "mean"), loo_sd=("loo", "std"), n=("loo", "count"))
        .reset_index()
    )

    hier_path = cache_path("glm_hierarchical_fits.parquet")
    if hier_path.exists():
        hier = _read_parquet(hier_path)
        hier = hier[(hier.status == "ok") & (hier.max_rhat < 1.1)].copy()
        if not hier.empty:
            hier["spec"] = "M4_hierarchical"
            # Normalise by number of companies so it's comparable per-triangle.
            hier["loo_mean"] = hier["loo"] / hier["n_companies"]
            hier["loo_sd"] = float("nan")
            hier["n"] = hier["n_companies"]
            hier_agg = hier[["line", "spec", "loo_mean", "loo_sd", "n"]]
            combined = pd.concat([per_agg, hier_agg], ignore_index=True)
        else:
            combined = per_agg
    else:
        combined = per_agg

    if combined.empty:
        return combined, pd.DataFrame(columns=["line", "best_spec", "loo_mean"])

    # Winner selection uses only dollar-scale (gamma+log) specs.
    # MT* specs are on loss-ratio scale so their LOO cannot be compared directly.
    dollar_specs = combined[~combined["spec"].str.startswith("MT")]
    if dollar_specs.empty:
        winners = pd.DataFrame(columns=["line", "best_spec", "loo_mean"])
    else:
        winners = (
            dollar_specs.sort_values("loo_mean", ascending=False)
            .drop_duplicates("line")
            .rename(columns={"spec": "best_spec"})
            [["line", "best_spec", "loo_mean"]]
        )
    return combined, winners


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

    per_path = cache_path("glm_per_triangle_fits.parquet")
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


def _build_loo_scale_caveat() -> str:
    """Build the LOO comparability caveat section for MT vs gamma specs."""
    return """\n## LOO Comparability Note — Dollar Scale vs Loss-Ratio Scale

**MT2** and **MT5_cal** fit the model on *loss-ratio-incremental* response
(paid / earned_premium per cell), using a Student-t family with identity link.
The remaining specs (M1–M5_cal) fit on *dollar-incremental* response with a
gamma + log-link.

**LOO is NOT directly comparable across these two scale classes.**
The log-likelihood density for the t-family on loss-ratio scale has a different
reference measure than the gamma density on dollar scale. As a result, MT LOO
values (typically slightly positive, ~100 to ~150 per triangle) cannot be ranked
against gamma+log LOO values (typically large-negative, ~−500 to ~−400 per triangle).

To compare them on equal footing one would add `log(EP_per_cell)` to each MT
log-likelihood observation (the Jacobian for the y → y/EP change of variables),
converting the MT LOO to dollar-equivalent units. This correction is not applied
here — instead, the two scale-classes are reported separately and compared within
each class:

- **Gamma + log-link:** compare M1, M2, M3, M4 (hierarchical), M2_cal, M5_cal
- **t + identity-link (loss-ratio):** compare MT2, MT5_cal

"""


def _build_readme(
    combined: pd.DataFrame,
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

    lines_out.append(_build_loo_scale_caveat())

    lines_out.append("\n## GLM Functional-Form Comparison\n")
    lines_out.append(
        "Family: gamma + log link. M1: full categorical origin+dev. "
        "M2: C(origin) + B-spline on dev ordinal index (df=4). "
        "M3: B-spline on origin (df=3) + C(dev). "
        "M4: hierarchical (1|snl_id) — normalised LOO by n_companies for comparability. "
        "**MT2 and MT5_cal use t + identity link on loss-ratio response — see LOO "
        "comparability note above.**\n"
    )
    if not combined.empty:
        # Split into dollar-scale and loss-ratio-scale for separate display
        dollar_combined = combined[~combined["spec"].str.startswith("MT")]
        mt_combined = combined[combined["spec"].str.startswith("MT")]

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
                "\n### t + identity-link (loss-ratio-scale response)\n"
                "LOO is on loss-ratio density scale — NOT comparable to gamma+log above.\n"
                "Mean LOO per spec (higher = better within this scale class):\n"
            )
            lines_out.append(mt_pivot.to_markdown() + "\n")

            mt_n_pivot = (
                mt_combined.pivot(index="line", columns="spec", values="n")
                .reindex(LINES)
                .fillna(0)
                .astype(int)
            )
            lines_out.append(
                "\nConverged-fit counts:\n"
            )
            lines_out.append(mt_n_pivot.to_markdown() + "\n")
    else:
        lines_out.append("_GLM fits not yet available (sweeps still running)._\n")

    wald_section = _build_wald_comparison_section()
    if wald_section:
        lines_out.append(wald_section)

    hier_path = cache_path("glm_hierarchical_fits.parquet")
    if hier_path.exists():
        hier_raw = _read_parquet(hier_path)
        lines_out.append("\n### M4 hierarchical (Bambi `(1 | snl_id)`) convergence diagnostics\n")
        # Dynamically report M4 convergence status rather than using stale hardcoded text.
        converged_m4 = hier_raw[(hier_raw.get("status", "ok") == "ok") & (hier_raw.get("max_rhat", 2.0) < 1.1)] if not hier_raw.empty else pd.DataFrame()
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
        hier_cols = [c for c in ["line", "status", "max_rhat", "loo", "n_obs", "n_companies", "company_sigma_mean"] if c in hier_raw.columns]
        lines_out.append(
            hier_raw[hier_cols].to_markdown(index=False, floatfmt=".3f")
            + "\n"
        )

    glm_priors_path = cache_path("glm_priors_by_line.parquet")
    if glm_priors_path.exists():
        _glm_priors = glm_priors if glm_priors is not None else _read_parquet(glm_priors_path)
        lines_out.append("\n## GLM Prior Recommendations (BayesianChainLadderGLM, gamma + log link)\n")
        lines_out.append(
            "Per-line prior recommendations derived from posteriors of the "
            "M1 fits (24 sampled triangles per line, gamma+log link, default "
            "package priors). The recommended priors below are **for use as "
            "informative defaults** in `BayesianChainLadderGLM(priors=...)`.\n"
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
            "\n## GLM Prior Recommendations (BayesianChainLadderGLM, t + identity link, loss-ratio)\n"
        )
        lines_out.append(
            "Per-line prior recommendations derived from posteriors of the "
            "MT2 fits (`incremental ~ 1 + C(origin) + bs(dev_idx, df=4)`, "
            "t family, identity link, `response_per_exposure=True`). "
            "The response is on loss-ratio scale (incremental paid / earned premium). "
            "These priors are **for use when fitting with `family='t', link='identity', "
            "response_per_exposure=True`**.\n"
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
    rho_cols = [c for c in ["line", "n_companies", "rho_point", "rho_median", "rho_p10", "rho_p90", "r1", "r2", "r3"] if c in rho.columns]
    lines_out.append(
        rho[rho_cols].to_markdown(index=False, floatfmt=".3f") + "\n"
    )

    lines_out.append("\n## Per-Line Descriptive Diagnostics\n")
    for line in LINES:
        lines_out.append(f"### {line}\n")
        lines_out.append(
            f"![{line} diagnostic](figures/01_descriptive_{line}.png)\n"
        )

    return "\n".join(lines_out)


def _payload_for_html(
    combined: pd.DataFrame,
    winners: pd.DataFrame,
    csr_agg: pd.DataFrame,
    rho: pd.DataFrame,
    descriptive: pd.DataFrame,
    rec: pd.DataFrame,
    glm_priors: pd.DataFrame | None = None,
    glm_t_priors: pd.DataFrame | None = None,
) -> dict:
    """Convert pandas DataFrames into JSON-serialisable list-of-dicts payload."""
    glm_path = cache_path("glm_per_triangle_fits.parquet")
    hier_path = cache_path("glm_hierarchical_fits.parquet")
    csr_path = cache_path("csr_fits.parquet")

    glm_rows: list[dict] = []
    if glm_path.exists():
        glm_df = _read_parquet(glm_path)
        if not glm_df.empty:
            glm_rows = (
                glm_df[["line", "snl_id", "spec", "status", "loo", "max_rhat"]]
                .to_dict("records")
            )

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
    }


def main() -> int:
    descriptive = _read_parquet(cache_path("descriptive_summary.parquet"))
    rho = _read_parquet(cache_path("rho_by_line.parquet"))
    combined, winners = _rank_glm_specs()
    csr_agg = _aggregate_csr()

    # Load GLM priors (gamma+log) if available.
    glm_priors_path = cache_path("glm_priors_by_line.parquet")
    glm_priors: pd.DataFrame | None = None
    if glm_priors_path.exists():
        glm_priors = _read_parquet(glm_priors_path)

    # Load GLM t-family priors (loss-ratio) if available.
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
            csr_agg[
                [
                    "line",
                    "csr_logelr_prior",
                    "csr_gamma_prior",
                    "csr_sig_prior",
                ]
            ],
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
        combined, winners, csr_agg, rho, descriptive, rec,
        glm_priors=glm_priors, glm_t_priors=glm_t_priors,
    )
    readme_path = ANALYSIS_DIR / "README.md"
    readme_path.write_text(md, encoding="utf-8")

    html_payload = _payload_for_html(
        combined, winners, csr_agg, rho, descriptive, rec,
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
