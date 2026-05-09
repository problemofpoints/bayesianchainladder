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

    winners = (
        combined.sort_values("loo_mean", ascending=False)
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


def _build_readme(
    combined: pd.DataFrame,
    winners: pd.DataFrame,
    csr_agg: pd.DataFrame,
    rho: pd.DataFrame,
    descriptive: pd.DataFrame,
    rec: pd.DataFrame,
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

    lines_out.append("\n## GLM Functional-Form Comparison\n")
    lines_out.append(
        "**Note:** M2 (`bs(dev, df=4)`) was excluded from the full sweep "
        "after smoke testing showed ~100% NUTS divergences regardless of MCMC "
        "budget. The comparison below is M1 (full categorical) vs M3 (origin "
        "spline) vs M4 (hierarchical pool).\n"
    )
    if not combined.empty:
        pivot = (
            combined.pivot(index="line", columns="spec", values="loo_mean")
            .reindex(LINES)
            .round(2)
        )
        lines_out.append(
            "Mean LOO per spec (higher = better, NaN = no converged fits):\n"
        )
        lines_out.append(pivot.to_markdown() + "\n")
    else:
        lines_out.append("_GLM fits not yet available (sweeps still running)._\n")

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

    return {
        "lines": LINES,
        "recs": rec.to_dict("records"),
        "glm": glm_rows,
        "hier": hier_rows,
        "csr": csr_rows,
        "rho": rho.to_dict("records"),
        "desc": descriptive.to_dict("records"),
    }


def main() -> int:
    descriptive = _read_parquet(cache_path("descriptive_summary.parquet"))
    rho = _read_parquet(cache_path("rho_by_line.parquet"))
    combined, winners = _rank_glm_specs()
    csr_agg = _aggregate_csr()

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

    md = _build_readme(combined, winners, csr_agg, rho, descriptive, rec)
    readme_path = ANALYSIS_DIR / "README.md"
    readme_path.write_text(md, encoding="utf-8")

    html_payload = _payload_for_html(combined, winners, csr_agg, rho, descriptive, rec)
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
