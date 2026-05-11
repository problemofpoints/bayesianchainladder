"""13_distribution_plots.py — Implied-percentile distribution visualizations.

Builds per-line implied-percentile histograms and calibration summary tables
from the completed back-test cache.  Produces:

  figures/implied_pctl_hist_<line>.png  — 7-method overlay histogram per line
  figures/implied_pctl_grid.png         — 4 lines × 7 methods grid
  figures/pp_<line>.png                 — PP plots (empirical CDF vs uniform)

Also prints a calibration summary table (pct_in_central_50, pct_in_central_80,
mean_implied_pctl, KS statistic vs uniform).

Run after 02_worker.py completes:
    cd references/meyers-backtest
    uv run python 13_distribution_plots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kstest

ANALYSIS_DIR = Path(__file__).resolve().parent
CACHE_DIR = ANALYSIS_DIR / "cache"
FIGURES_DIR = ANALYSIS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Method display config
# ---------------------------------------------------------------------------

METHOD_DISPLAY = {
    "mack": "Mack",
    "bootstrap_odp": "Bootstrap ODP",
    "corr_boot_odp": "Corr Bootstrap ODP",
    "bayesian_csr": "Bayesian CSR",
    "glm_m1_cat": "GLM M1 (categorical)",
    "glm_m2": "GLM M2 (spline dev)",
    "glm_m5_cal": "GLM M5_cal (RE+cal)",
    "glm_mt5_cal": "GLM MT5_cal (t-identity)",
}

METHOD_ORDER = list(METHOD_DISPLAY.keys())

LINE_DISPLAY = {
    "comauto": "Commercial Auto",
    "ppauto": "Private Passenger Auto",
    "wkcomp": "Workers Comp",
    "othliab": "Other Liability",
}

# Colors for up to 8 methods
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_all() -> pd.DataFrame:
    files = sorted(glob.glob(str(CACHE_DIR / "backtest_chunk_*.parquet")))
    if not files:
        raise FileNotFoundError(f"No chunk files found in {CACHE_DIR}")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# Calibration metrics helper
# ---------------------------------------------------------------------------

def calibration_metrics(pctls: np.ndarray) -> dict:
    """Compute calibration metrics for an array of implied percentiles."""
    finite = pctls[np.isfinite(pctls)]
    if finite.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "pct_central_50": float("nan"),
            "pct_central_80": float("nan"),
            "ks_stat": float("nan"),
            "ks_pvalue": float("nan"),
        }
    n = len(finite)
    mean_p = float(np.mean(finite))
    pct_50 = float(np.mean((finite >= 0.25) & (finite <= 0.75))) * 100
    pct_80 = float(np.mean((finite >= 0.10) & (finite <= 0.90))) * 100
    ks_stat, ks_pval = kstest(finite, "uniform")
    return {
        "n": n,
        "mean": mean_p,
        "pct_central_50": pct_50,
        "pct_central_80": pct_80,
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pval),
    }


# ---------------------------------------------------------------------------
# Per-line overlay histogram
# ---------------------------------------------------------------------------

def plot_hist_per_line(ok: pd.DataFrame) -> None:
    """One histogram per line; all methods overlaid with transparency."""
    lines = sorted(ok.line.unique())
    methods_present = [m for m in METHOD_ORDER if m in ok.method.unique()]
    colors = {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(methods_present)}

    for line in lines:
        sub = ok[ok.line == line]
        fig, ax = plt.subplots(figsize=(8, 5))
        for m in methods_present:
            ms_vals = sub[sub.method == m]["implied_pctl"].dropna().values
            if ms_vals.size == 0:
                continue
            label = METHOD_DISPLAY.get(m, m)
            ax.hist(
                ms_vals,
                bins=20,
                range=(0, 1),
                density=True,
                alpha=0.35,
                color=colors[m],
                edgecolor=colors[m],
                label=label,
            )
        # Ideal uniform density = 1.0
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Ideal (uniform)")
        ax.set_title(
            f"Implied-Percentile Distribution — {LINE_DISPLAY.get(line, line)}\n"
            f"({len(sub[sub.method == methods_present[0]].dropna())} triangles per method)",
            fontsize=11,
        )
        ax.set_xlabel("Implied Percentile", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        out = FIGURES_DIR / f"implied_pctl_hist_{line}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# 4-line × N-method grid
# ---------------------------------------------------------------------------

def plot_grid(ok: pd.DataFrame) -> None:
    """Grid: rows = lines, cols = methods; each cell is a mini histogram."""
    lines = sorted(ok.line.unique())
    methods_present = [m for m in METHOD_ORDER if m in ok.method.unique()]
    n_lines = len(lines)
    n_methods = len(methods_present)
    colors = {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(methods_present)}

    fig, axes = plt.subplots(
        n_lines, n_methods,
        figsize=(2.2 * n_methods, 2.0 * n_lines),
        sharey=False,
    )
    # Ensure 2D axes array
    if n_lines == 1:
        axes = [axes]
    if n_methods == 1:
        axes = [[ax] for ax in axes]

    for ri, line in enumerate(lines):
        sub = ok[ok.line == line]
        for ci, m in enumerate(methods_present):
            ax = axes[ri][ci]
            ms_vals = sub[sub.method == m]["implied_pctl"].dropna().values
            if ms_vals.size > 0:
                ax.hist(
                    ms_vals,
                    bins=10,
                    range=(0, 1),
                    density=True,
                    color=colors[m],
                    edgecolor="white",
                    linewidth=0.3,
                    alpha=0.85,
                )
                ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.5, 1])
            ax.tick_params(axis="both", labelsize=6)
            if ri == 0:
                ax.set_title(METHOD_DISPLAY.get(m, m), fontsize=7, pad=3)
            if ci == 0:
                ax.set_ylabel(LINE_DISPLAY.get(line, line), fontsize=7)

    fig.suptitle("Implied-Percentile Histograms: 4 Lines × 8 Methods", fontsize=10, y=1.01)
    fig.tight_layout()
    out = FIGURES_DIR / "implied_pctl_grid.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# PP plots (one per line, methods overlaid)
# ---------------------------------------------------------------------------

def plot_pp_per_line(ok: pd.DataFrame) -> None:
    """PP plots: empirical CDF of sorted implied percentiles vs the diagonal."""
    lines = sorted(ok.line.unique())
    methods_present = [m for m in METHOD_ORDER if m in ok.method.unique()]
    colors = {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(methods_present)}

    for line in lines:
        sub = ok[ok.line == line]
        fig, ax = plt.subplots(figsize=(6, 5))
        for m in methods_present:
            ms_vals = sub[sub.method == m]["implied_pctl"].dropna().sort_values().values
            if ms_vals.size == 0:
                continue
            x = np.linspace(0, 1, ms_vals.size)
            ax.plot(ms_vals, x, marker="o", markersize=3, alpha=0.7,
                    color=colors[m], label=METHOD_DISPLAY.get(m, m))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal (uniform)")
        ax.set_title(f"PP Plot — {LINE_DISPLAY.get(line, line)}", fontsize=11)
        ax.set_xlabel("Implied Percentile (sorted)", fontsize=10)
        ax.set_ylabel("Empirical CDF", fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        out = FIGURES_DIR / f"pp_{line}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"  Saved {out.name}")


# ---------------------------------------------------------------------------
# Calibration table
# ---------------------------------------------------------------------------

def build_calibration_table(ok: pd.DataFrame) -> pd.DataFrame:
    """Build a calibration summary table per method × line."""
    rows = []
    lines = sorted(ok.line.unique())
    methods_present = [m for m in METHOD_ORDER if m in ok.method.unique()]

    for line in lines:
        sub = ok[ok.line == line]
        for m in methods_present:
            pctls = sub[sub.method == m]["implied_pctl"].dropna().values
            metrics = calibration_metrics(pctls)
            rows.append({
                "line": LINE_DISPLAY.get(line, line),
                "method": METHOD_DISPLAY.get(m, m),
                **metrics,
            })

    # Also build overall (across lines) per method
    for m in methods_present:
        pctls = ok[ok.method == m]["implied_pctl"].dropna().values
        metrics = calibration_metrics(pctls)
        rows.append({
            "line": "ALL LINES",
            "method": METHOD_DISPLAY.get(m, m),
            **metrics,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# README writer
# ---------------------------------------------------------------------------

def write_readme(ok: pd.DataFrame, calib_df: pd.DataFrame) -> None:
    """Write README.md with calibration table and figure references."""
    lines = sorted(ok.line.unique())

    overall = calib_df[calib_df.line == "ALL LINES"].copy()
    # Format float columns
    overall = overall.round({"mean": 3, "pct_central_50": 1, "pct_central_80": 1,
                              "ks_stat": 4, "ks_pvalue": 4})

    md = [
        "# Meyers Back-Test Results\n",
        "Back-test of 7–8 reserve methods against the Meyers subset "
        "(200 triangles × 4 lines).\n",
        f"\nSuccessful fits: {len(ok)} / {len(ok.line.unique()) * ok.method.nunique() * 50}\n",
    ]

    md.append("\n## Calibration Summary (all lines combined)\n")
    md.append(
        "Ideal values: `mean_implied_pctl = 0.50`, "
        "`pct_central_50 = 50.0%`, `pct_central_80 = 80.0%`, `ks_stat ≈ 0`.\n"
    )
    md.append(overall[["method", "n", "mean", "pct_central_50", "pct_central_80",
                         "ks_stat", "ks_pvalue"]].to_markdown(index=False, floatfmt=".3f") + "\n")

    md.append("\n## Implied-Percentile Histograms\n")
    md.append("### All lines, all methods (grid)\n")
    md.append("![Histogram grid](figures/implied_pctl_grid.png)\n")
    for line in lines:
        md.append(f"### {LINE_DISPLAY.get(line, line)}\n")
        md.append(f"![{line} histogram](figures/implied_pctl_hist_{line}.png)\n")

    md.append("\n## PP Plots\n")
    for line in lines:
        md.append(f"### {LINE_DISPLAY.get(line, line)}\n")
        md.append(f"![{line} PP](figures/pp_{line}.png)\n")

    md.append("\n## Per-Line Calibration Table\n")
    per_line = calib_df[calib_df.line != "ALL LINES"].copy()
    per_line = per_line.round({"mean": 3, "pct_central_50": 1, "pct_central_80": 1,
                                "ks_stat": 4})
    md.append(per_line[["line", "method", "n", "mean", "pct_central_50",
                          "pct_central_80", "ks_stat"]].to_markdown(index=False, floatfmt=".3f") + "\n")

    (ANALYSIS_DIR / "README.md").write_text("\n".join(md), encoding="utf-8")
    print("  Wrote README.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("Loading data from cache ...")
    df = load_all()
    df.to_parquet(CACHE_DIR / "backtest_all.parquet", index=False)
    print(f"  Total rows: {len(df)}")

    ok = df[df.status == "ok"].copy()
    print(f"  Successful fits: {len(ok)}")

    if ok.empty:
        print("No successful fits found. Exiting.")
        return 1

    skip = df[df.status == "skipped:negative_incrementals"]
    print(f"  Skipped (negative incrementals): {len(skip)}")
    errors = df[~df.status.isin(["ok", "skipped:negative_incrementals"])]
    if len(errors) > 0:
        print(f"  Errors: {len(errors)}")
        print(errors.groupby(["method", "status"]).size().head(20).to_string())

    print("\nBuilding calibration table ...")
    calib_df = build_calibration_table(ok)
    calib_df.to_parquet(CACHE_DIR / "calibration.parquet", index=False)

    # Print overall calibration table to stdout
    overall = calib_df[calib_df.line == "ALL LINES"].copy()
    print("\n=== Calibration Summary (all lines) ===")
    print(f"{'Method':<30s}  {'N':>5s}  {'Mean%':>8s}  {'Central50':>10s}  {'Central80':>10s}  {'KS':>8s}")
    print("-" * 78)
    for _, row in overall.iterrows():
        print(
            f"  {row['method']:<28s}  {row['n']:>5.0f}  {row['mean']:>8.3f}  "
            f"{row['pct_central_50']:>10.1f}  {row['pct_central_80']:>10.1f}  "
            f"{row['ks_stat']:>8.4f}"
        )

    print("\nGenerating figures ...")
    plot_hist_per_line(ok)
    plot_grid(ok)
    plot_pp_per_line(ok)

    print("\nWriting README.md ...")
    write_readme(ok, calib_df)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
