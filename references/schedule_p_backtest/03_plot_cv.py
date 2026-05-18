"""03_plot_cv.py
================
Build CV(IBNR) bar charts from the aggregated Schedule P results.

Per-line chart: one file per line of business showing median CV by method × loss_type.
Overview chart: single figure showing all lines × methods (grid of subplots).

Reads: references/schedule_p_backtest/cache/schedp_cv_summary.csv
Saves: references/schedule_p_backtest/figures/cv_by_method_<line>.png
       references/schedule_p_backtest/figures/cv_overview.png
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

SUMMARY_PATH = Path(__file__).resolve().parent / "cache" / "schedp_cv_summary.csv"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

METHOD_ORDER = [
    "mack",
    "odp",
    "odp_param",
    "odp_corr",
    "odp_bf",
    "odp_cc",
    "odp_corr_bf",
    "odp_corr_cc",
]

METHOD_LABELS = {
    "mack": "Mack",
    "odp": "ODP",
    "odp_param": "ODP\n(param)",
    "odp_corr": "ODP\n(corr)",
    "odp_bf": "ODP\nBF",
    "odp_cc": "ODP\nCC",
    "odp_corr_bf": "ODP\ncorr+BF",
    "odp_corr_cc": "ODP\ncorr+CC",
}

LOSS_COLORS = {
    "paid": "#2c7bb6",
    "case_incurred": "#d7191c",
}

PRIORITY_LINES = ["OLO", "OLC", "CAL", "WC", "PPAL", "CMP", "ALL"]


def plot_line_cv(agg: pd.DataFrame, lob: str, ax: plt.Axes | None = None) -> plt.Figure | None:
    """Bar chart of median CV by (method, loss_type) for one line of business."""
    sub = agg[agg["lob"] == lob].copy()
    if sub.empty:
        return None

    methods = [m for m in METHOD_ORDER if m in sub["method"].unique()]
    loss_types = sorted(sub["loss_type"].unique())

    x = np.arange(len(methods))
    bar_width = 0.35
    n_types = len(loss_types)
    offsets = np.linspace(-(n_types - 1) * bar_width / 2, (n_types - 1) * bar_width / 2, n_types)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    for i, lt in enumerate(loss_types):
        lt_sub = sub[sub["loss_type"] == lt].set_index("method")
        vals = [lt_sub.loc[m, "median_cv"] if m in lt_sub.index else float("nan") for m in methods]
        n_companies = lt_sub["n_companies"].max() if not lt_sub.empty else 0
        label = f"{lt} (n={n_companies})"
        bars = ax.bar(
            x + offsets[i], vals, bar_width,
            label=label,
            color=LOSS_COLORS.get(lt, "#888888"),
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        # Annotate with value
        for bar, val in zip(bars, vals):
            if not math.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=6,
                    color="#333333",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=8)
    ax.set_ylabel("Median CV(IBNR)", fontsize=9)
    ax.set_title(f"LOB: {lob}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, None)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if standalone:
        fig.tight_layout()
        return fig
    return None


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading summary from {SUMMARY_PATH} ...")
    agg = pd.read_csv(SUMMARY_PATH)
    print(f"  {len(agg):,} rows, LOBs: {sorted(agg['lob'].unique())}")

    lobs = sorted(agg["lob"].unique())

    # -------------------------------------------------------------------
    # Per-line charts
    # -------------------------------------------------------------------
    for lob in lobs:
        fig = plot_line_cv(agg, lob)
        if fig is not None:
            out = FIGURES_DIR / f"cv_by_method_{lob}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {out}")

    # -------------------------------------------------------------------
    # Overview chart — all lines in a grid
    # -------------------------------------------------------------------
    n_lobs = len(lobs)
    ncols = 4
    nrows = math.ceil(n_lobs / ncols)

    fig_ov, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 4))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    for idx, lob in enumerate(lobs):
        ax = axes_flat[idx]
        plot_line_cv(agg, lob, ax=ax)

    # Hide unused axes
    for idx in range(n_lobs, nrows * ncols):
        axes_flat[idx].set_visible(False)

    fig_ov.suptitle(
        "Schedule P YE2024 — Median CV(IBNR) by Line × Method\n"
        "(lognormal process variance, rho=0.3, apriori_sigma=0.15, n_sims=5000)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig_ov.tight_layout()

    out_ov = FIGURES_DIR / "cv_overview.png"
    fig_ov.savefig(out_ov, dpi=150, bbox_inches="tight")
    plt.close(fig_ov)
    print(f"\nSaved overview chart: {out_ov}")


if __name__ == "__main__":
    main()
