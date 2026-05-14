"""18_standalone_calibration_v2.py
===================================
Calibration analysis for the v2 standalone stochastic reserving back-test that
adds ``odp_param`` (parametric Normal, rho=0) to isolate the effect of
non-parametric residual resampling from calendar-year correlation.

Inputs (from run_stochastic_reserving.py v2 run)
-------------------------------------------------
  references/meyers-backtest/cache/meyers_standalone_results_v2.csv
  references/meyers-backtest/cache/meyers_standalone_samples_v2.parquet

Outputs
-------
  figures/standalone_implied_pctl_grid_v2.png
  figures/standalone_pp_paid_v2.png
  figures/standalone_pp_case_incurred_v2.png
  references/meyers-backtest/cache/standalone_calibration_v2.csv
  Updated STANDALONE_BACKTEST_README.md section appended at end of stdout

Key comparisons
---------------
1. odp (non-param) vs odp_param (param, rho=0):
     Same rho=0 independence assumption; only bootstrap type differs.
     If case_incurred CV blow-up disappears, the artifact is non-param resampling.
2. odp_param (param, rho=0) vs odp_corr (param, rho=0.1):
     Same parametric Normal sampling; only correlation level differs.
     Pure effect of calendar-year correlation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ANALYSIS_DIR, CACHE_DIR, FIGURES_DIR

RESULTS_CSV = CACHE_DIR / "meyers_standalone_results_v2.csv"
SAMPLES_PARQUET = CACHE_DIR / "meyers_standalone_samples_v2.parquet"
CALIBRATION_CSV = CACHE_DIR / "standalone_calibration_v2.csv"

METHODS = ["mack", "odp", "odp_param", "odp_corr", "odp_bf", "odp_cc"]
LOSS_TYPES = ["paid", "case_incurred"]
LOBS = ["comauto", "ppauto", "wkcomp", "othliab"]

METHOD_LABELS = {
    "mack": "Mack",
    "odp": "ODP (non-param)",
    "odp_param": "ODP param (rho=0)",
    "odp_corr": "Corr-ODP (rho=0.1)",
    "odp_bf": "ODP+BF",
    "odp_cc": "ODP+CC",
}

LOSS_LABELS = {
    "paid": "Paid",
    "case_incurred": "Case Incurred",
}


# ---------------------------------------------------------------------------
# Load actual ultimates from reservetestr
# ---------------------------------------------------------------------------

def load_actual_ultimates() -> pd.DataFrame:
    """Return DataFrame with actual ultimate totals per (lob, group_id, loss_type)."""
    import reservetestr as rt

    records = rt.build_triangle_records()
    rows = []
    for rec in records:
        rows.append({
            "lob": rec.line,
            "group_id": rec.group_id,
            "loss_type": "paid",
            "actual_ultimate_total": rec.actual_ultimates.get("paid", np.nan),
        })
        rows.append({
            "lob": rec.line,
            "group_id": rec.group_id,
            "loss_type": "case_incurred",
            "actual_ultimate_total": rec.actual_ultimates.get("case", np.nan),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Implied percentile
# ---------------------------------------------------------------------------

def implied_pctl(actual_value: float, samples: np.ndarray) -> float:
    """Empirical CDF of actual_value under samples (fraction <= actual_value)."""
    if len(samples) == 0 or not np.isfinite(actual_value):
        return np.nan
    finite = samples[np.isfinite(samples)]
    if len(finite) == 0:
        return np.nan
    return float(np.mean(finite <= actual_value))


# ---------------------------------------------------------------------------
# Calibration metrics helper
# ---------------------------------------------------------------------------

def calibration_metrics(group: pd.DataFrame) -> pd.Series:
    p = group["implied_pctl"].dropna()
    cv_vals = group["cv_ibnr"].dropna() if "cv_ibnr" in group.columns else pd.Series([], dtype=float)
    sd_vals = group["sd_ibnr"].dropna() if "sd_ibnr" in group.columns else pd.Series([], dtype=float)
    if len(p) == 0:
        return pd.Series({
            "n": 0,
            "mean_pctl": np.nan,
            "pct_in_central50": np.nan,
            "pct_in_central80": np.nan,
            "ks_stat": np.nan,
            "ks_pval": np.nan,
            "median_abs_pct_err": np.nan,
            "median_cv_ibnr": np.nan,
            "median_sd_ibnr": np.nan,
        })
    ks_stat, ks_pval = stats.kstest(p, "uniform")
    return pd.Series({
        "n": len(p),
        "mean_pctl": p.mean(),
        "pct_in_central50": float(((p >= 0.25) & (p <= 0.75)).mean() * 100),
        "pct_in_central80": float(((p >= 0.10) & (p <= 0.90)).mean() * 100),
        "ks_stat": ks_stat,
        "ks_pval": ks_pval,
        "median_abs_pct_err": group["pct_err"].abs().median() * 100,
        "median_cv_ibnr": float(cv_vals.median()) if len(cv_vals) > 0 else np.nan,
        "median_sd_ibnr": float(sd_vals.median()) if len(sd_vals) > 0 else np.nan,
    })


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading results CSV...")
    results = pd.read_csv(RESULTS_CSV)
    print(f"  {len(results):,} rows")
    available_methods = results["method"].unique().tolist()
    print(f"  Methods in results: {sorted(available_methods)}")

    print("Loading samples parquet...")
    samples_df = pd.read_parquet(SAMPLES_PARQUET)
    samples_df["group_id"] = samples_df["group_id"].astype(int)
    n_combos = samples_df.groupby(["lob", "group_id", "loss_type", "method"]).ngroups
    print(f"  {len(samples_df):,} rows, {n_combos} combos")

    print("Loading actual ultimates from reservetestr...")
    actuals = load_actual_ultimates()
    print(f"  {len(actuals)} rows")

    # Get total-row paid_to_date from results
    total_results = results[results["accident_year"] == "Total"].copy()
    total_results["group_id"] = total_results["group_id"].astype(int)

    paid_to_date_lookup = (
        total_results[total_results["loss_type"] == "paid"]
        .groupby(["lob", "group_id"])[["paid_to_date"]]
        .first()
        .reset_index()
    )
    paid_to_date_lookup.rename(columns={"paid_to_date": "paid_to_date_total"}, inplace=True)

    actuals = actuals.merge(paid_to_date_lookup, on=["lob", "group_id"], how="left")
    actuals["actual_total_unpaid"] = actuals["actual_ultimate_total"] - actuals["paid_to_date_total"]

    # Extract per-group CV and SD from results total rows (for dispersion comparison)
    results_total = total_results.copy()
    results_total = results_total[["lob", "group_id", "loss_type", "method", "cv_ibnr", "mean_ibnr"]].copy()
    # Compute SD from cv and mean
    results_total["sd_ibnr"] = results_total["cv_ibnr"].abs() * results_total["mean_ibnr"].abs()

    print("Computing implied percentiles...")
    sample_groups = samples_df.groupby(["lob", "group_id", "loss_type", "method"])

    cal_rows = []
    for (lob, group_id, loss_type, method), grp in sample_groups:
        ibnr_samples = grp["total_ibnr"].values.astype(float)

        act_row = actuals[
            (actuals["lob"] == lob)
            & (actuals["group_id"] == group_id)
            & (actuals["loss_type"] == loss_type)
        ]
        if act_row.empty or not np.isfinite(act_row.iloc[0]["actual_total_unpaid"]):
            continue

        actual_unpaid = float(act_row.iloc[0]["actual_total_unpaid"])
        actual_ult = float(act_row.iloc[0]["actual_ultimate_total"])
        paid_td = float(act_row.iloc[0]["paid_to_date_total"])

        pctl = implied_pctl(actual_unpaid, ibnr_samples)

        mean_ibnr_samples = float(np.mean(ibnr_samples[np.isfinite(ibnr_samples)]))
        mean_ult_est = paid_td + mean_ibnr_samples
        pct_err = (mean_ult_est - actual_ult) / actual_ult if actual_ult != 0 else np.nan

        # CV from samples
        finite_samp = ibnr_samples[np.isfinite(ibnr_samples)]
        cv_from_samples = (
            float(np.std(finite_samp, ddof=1) / np.abs(np.mean(finite_samp)))
            if len(finite_samp) > 1 and np.mean(finite_samp) != 0
            else np.nan
        )
        sd_from_samples = float(np.std(finite_samp, ddof=1)) if len(finite_samp) > 1 else np.nan

        cal_rows.append({
            "lob": lob,
            "group_id": group_id,
            "loss_type": loss_type,
            "method": method,
            "actual_ultimate": actual_ult,
            "paid_to_date": paid_td,
            "actual_unpaid": actual_unpaid,
            "mean_ibnr_est": mean_ibnr_samples,
            "mean_ult_est": mean_ult_est,
            "pct_err": pct_err,
            "implied_pctl": pctl,
            "cv_ibnr": cv_from_samples,
            "sd_ibnr": sd_from_samples,
        })

    cal_df = pd.DataFrame(cal_rows)
    print(f"  Calibration table: {len(cal_df)} rows, "
          f"{cal_df.groupby(['loss_type','method']).ngroups} (loss_type,method) combos")

    # Save calibration detail CSV
    cal_df.to_csv(CALIBRATION_CSV, index=False)
    print(f"  Saved: {CALIBRATION_CSV}")

    # -----------------------------------------------------------------------
    # Calibration metrics per (loss_type, method)
    # -----------------------------------------------------------------------
    active_methods = [m for m in METHODS if m in available_methods]

    metrics = (
        cal_df.groupby(["loss_type", "method"])
        .apply(calibration_metrics, include_groups=False)
        .reset_index()
    )

    # -----------------------------------------------------------------------
    # Print main calibration table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("CALIBRATION TABLE: 6 methods × 2 loss types")
    print("=" * 120)
    print(f"{'Method':<22} {'LossType':<16} {'N':>5} {'MeanPctl':>10} {'%in50':>8} "
          f"{'%in80':>8} {'KS_stat':>8} {'KS_pval':>8} {'Med|%err|':>10} "
          f"{'MedCV':>8} {'MedSD':>12}")
    print("-" * 120)
    for _, row in metrics.sort_values(["loss_type", "method"]).iterrows():
        if row["method"] not in active_methods:
            continue
        print(
            f"{METHOD_LABELS.get(row['method'], row['method']):<22} "
            f"{LOSS_LABELS.get(row['loss_type'], row['loss_type']):<16} "
            f"{row['n']:>5.0f} "
            f"{row['mean_pctl']:>10.3f} "
            f"{row['pct_in_central50']:>8.1f} "
            f"{row['pct_in_central80']:>8.1f} "
            f"{row['ks_stat']:>8.4f} "
            f"{row['ks_pval']:>8.4f} "
            f"{row['median_abs_pct_err']:>10.2f} "
            f"{row['median_cv_ibnr']:>8.3f} "
            f"{row['median_sd_ibnr']:>12,.0f}"
        )

    # -----------------------------------------------------------------------
    # KEY COMPARISON 1: non-param vs param ODP (rho=0 vs rho=0)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("KEY COMPARISON 1: ODP non-param vs ODP param (rho=0 vs rho=0)")
    print("Same independence assumption; only bootstrap type differs.")
    print("=" * 120)
    comp1_methods = ["odp", "odp_param"]
    comp1_metrics = metrics[metrics["method"].isin(comp1_methods)]
    _print_comparison(comp1_metrics, "ks_stat", "KS stat", lower_better=True)
    _print_comparison(comp1_metrics, "pct_in_central80", "% in central 80%", lower_better=False)
    _print_comparison(comp1_metrics, "median_cv_ibnr", "Median CV(IBNR)", lower_better=True)

    # -----------------------------------------------------------------------
    # KEY COMPARISON 2: param rho=0 vs param rho=0.1 (pure correlation effect)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("KEY COMPARISON 2: ODP param rho=0 vs Corr-ODP rho=0.1 (pure correlation effect)")
    print("Same parametric Normal sampling; only correlation level differs.")
    print("=" * 120)
    comp2_methods = ["odp_param", "odp_corr"]
    comp2_metrics = metrics[metrics["method"].isin(comp2_methods)]
    _print_comparison(comp2_metrics, "ks_stat", "KS stat", lower_better=True)
    _print_comparison(comp2_metrics, "pct_in_central80", "% in central 80%", lower_better=False)
    _print_comparison(comp2_metrics, "median_cv_ibnr", "Median CV(IBNR)", lower_better=True)

    # -----------------------------------------------------------------------
    # Paid vs case_incurred comparison
    # -----------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("PAID vs CASE_INCURRED: KS stat (lower is better)")
    print("=" * 120)
    ks_compare = metrics[metrics["method"].isin(active_methods)].pivot_table(
        index="method", columns="loss_type", values="ks_stat"
    ).reindex(active_methods)
    ks_compare["delta"] = ks_compare.get("case_incurred", np.nan) - ks_compare.get("paid", np.nan)
    ks_compare.index = [METHOD_LABELS.get(m, m) for m in ks_compare.index]
    print(ks_compare.round(4).to_string())

    print("\nPAID vs CASE_INCURRED: Median CV(IBNR)")
    cv_compare = metrics[metrics["method"].isin(active_methods)].pivot_table(
        index="method", columns="loss_type", values="median_cv_ibnr"
    ).reindex(active_methods)
    cv_compare["delta"] = cv_compare.get("case_incurred", np.nan) - cv_compare.get("paid", np.nan)
    cv_compare.index = [METHOD_LABELS.get(m, m) for m in cv_compare.index]
    print(cv_compare.round(4).to_string())

    p80_compare = metrics[metrics["method"].isin(active_methods)].pivot_table(
        index="method", columns="loss_type", values="pct_in_central80"
    ).reindex(active_methods)
    p80_compare["delta"] = p80_compare.get("case_incurred", np.nan) - p80_compare.get("paid", np.nan)
    p80_compare.index = [METHOD_LABELS.get(m, m) for m in p80_compare.index]

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    _plot_implied_pctl_grid(cal_df, active_methods)
    _plot_pp_chart(cal_df, "paid", active_methods)
    _plot_pp_chart(cal_df, "case_incurred", active_methods)

    # -----------------------------------------------------------------------
    # Update STANDALONE_BACKTEST_README.md
    # -----------------------------------------------------------------------
    _update_readme(metrics, ks_compare, cv_compare, p80_compare, active_methods)

    print("\nDone.")
    return cal_df, metrics


def _print_comparison(metrics_subset: pd.DataFrame, col: str, label: str, lower_better: bool):
    """Print a 2-column paid/case_incurred comparison for a given metric."""
    pivot = metrics_subset.pivot_table(index="method", columns="loss_type", values=col)
    pivot.index = [METHOD_LABELS.get(m, m) for m in pivot.index]
    better = "lower" if lower_better else "higher"
    print(f"\n  {label} ({better} is better):")
    print("  " + pivot.round(4).to_string().replace("\n", "\n  "))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_implied_pctl_grid(cal_df: pd.DataFrame, active_methods: list[str]):
    """n_lobs × (n_methods × 2 loss_types) histograms of implied percentile."""
    n_lobs = len(LOBS)
    methods_in_data = [m for m in active_methods if m in cal_df["method"].unique()]
    n_combos = len(methods_in_data) * len(LOSS_TYPES)
    if n_combos == 0:
        return

    fig, axes = plt.subplots(
        n_lobs, n_combos, figsize=(n_combos * 1.5, n_lobs * 2.2), sharey=False
    )
    if n_lobs == 1:
        axes = axes[np.newaxis, :]
    if n_combos == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(
        f"Implied Percentile Histograms: 4 LOBs × ({len(methods_in_data)} Methods × 2 Loss Types)",
        fontsize=9, y=1.01,
    )

    for row_i, lob in enumerate(LOBS):
        col_i = 0
        for lt in LOSS_TYPES:
            for method in methods_in_data:
                ax = axes[row_i, col_i]
                subset = cal_df[
                    (cal_df["lob"] == lob)
                    & (cal_df["loss_type"] == lt)
                    & (cal_df["method"] == method)
                ]["implied_pctl"].dropna()
                color = "#2196F3" if lt == "paid" else "#FF9800"
                # Highlight odp_param in a different color to distinguish it
                if method == "odp_param":
                    color = "#4CAF50" if lt == "paid" else "#E91E63"
                ax.hist(subset, bins=10, range=(0, 1), color=color,
                        alpha=0.75, edgecolor="none")
                if len(subset) > 0:
                    ax.axhline(len(subset) / 10, color="red", lw=0.7, ls="--")
                ax.set_xlim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                if row_i == 0:
                    short_label = METHOD_LABELS.get(method, method).replace(" ", "\n")
                    ax.set_title(
                        f"{short_label}\n{LOSS_LABELS[lt]}",
                        fontsize=5.5, pad=2
                    )
                if col_i == 0:
                    ax.set_ylabel(lob.upper(), fontsize=7, labelpad=4)
                col_i += 1

    plt.tight_layout()
    out = FIGURES_DIR / "standalone_implied_pctl_grid_v2.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def _plot_pp_chart(cal_df: pd.DataFrame, loss_type: str, active_methods: list[str]):
    """PP chart for all methods on a given loss_type."""
    fig, ax = plt.subplots(figsize=(7, 7))

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]
    methods_to_plot = [m for m in active_methods if m in cal_df["method"].unique()]

    for method, color in zip(methods_to_plot, colors):
        p = cal_df[
            (cal_df["loss_type"] == loss_type) & (cal_df["method"] == method)
        ]["implied_pctl"].dropna().sort_values().values
        if len(p) == 0:
            continue
        emp = np.arange(1, len(p) + 1) / len(p)
        ax.plot(p, emp, label=METHOD_LABELS.get(method, method), color=color,
                lw=2.0 if method == "odp_param" else 1.5,
                ls="-" if method not in ("odp_param",) else "--")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
    ax.set_xlabel("Theoretical quantile (implied pctl)")
    ax.set_ylabel("Empirical CDF")
    lt_label = LOSS_LABELS.get(loss_type, loss_type)
    ax.set_title(f"PP Chart — {lt_label} — All 200 triangles (v2: +odp_param)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    out = FIGURES_DIR / f"standalone_pp_{loss_type}_v2.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# README update
# ---------------------------------------------------------------------------

def _update_readme(
    metrics: pd.DataFrame,
    ks_compare: pd.DataFrame,
    cv_compare: pd.DataFrame,
    p80_compare: pd.DataFrame,
    active_methods: list[str],
):
    """Append v2 parametric analysis section to STANDALONE_BACKTEST_README.md."""
    metrics_sorted = metrics[metrics["method"].isin(active_methods)].sort_values(
        ["loss_type", "ks_stat"]
    )

    lines = [
        "",
        "---",
        "",
        "## v2 Analysis: Parametric vs Non-Parametric Bootstrap",
        "",
        "Added `odp_param` (parametric Normal, rho=0) to isolate:",
        "1. Non-parametric residual resampling artifacts (odp vs odp_param)",
        "2. Pure calendar-year correlation effect (odp_param vs odp_corr)",
        "",
        "- **Data**: 200 Meyers triangles, 5,000 sims per method",
        "- **Script**: `scripts/run_stochastic_reserving.py` (v2 with odp_param)",
        "- **Outputs**: `cache/meyers_standalone_results_v2.csv`, `meyers_standalone_samples_v2.parquet`",
        "",
        "### Calibration table (6 methods × 2 loss types)",
        "",
        "| Method | Loss type | N | Mean pctl | % in 50% | % in 80% | KS stat | Med |%err| | Med CV(IBNR) |",
        "|--------|-----------|---|-----------|----------|----------|---------|------------|--------------|",
    ]

    for _, row in metrics_sorted.iterrows():
        cv_str = f"{row['median_cv_ibnr']:.3f}" if np.isfinite(row.get("median_cv_ibnr", np.nan)) else "n/a"
        lines.append(
            f"| {METHOD_LABELS.get(row['method'], row['method'])} "
            f"| {LOSS_LABELS.get(row['loss_type'], row['loss_type'])} "
            f"| {row['n']:.0f} "
            f"| {row['mean_pctl']:.3f} "
            f"| {row['pct_in_central50']:.1f}% "
            f"| {row['pct_in_central80']:.1f}% "
            f"| {row['ks_stat']:.4f} "
            f"| {row['median_abs_pct_err']:.2f}% "
            f"| {cv_str} |"
        )

    lines += [
        "",
        "### Paid vs Case-Incurred: KS stat",
        "",
        "Delta = case_incurred KS - paid KS (positive = case_incurred harder to calibrate).",
        "",
        "| Method | KS (paid) | KS (case) | Delta |",
        "|--------|-----------|-----------|-------|",
    ]
    for method_label, row in ks_compare.iterrows():
        paid_ks = row.get("paid", np.nan)
        case_ks = row.get("case_incurred", np.nan)
        delta = row.get("delta", np.nan)
        paid_str = f"{paid_ks:.4f}" if np.isfinite(paid_ks) else "n/a"
        case_str = f"{case_ks:.4f}" if np.isfinite(case_ks) else "n/a"
        delta_str = f"{delta:+.4f}" if np.isfinite(delta) else "n/a"
        lines.append(f"| {method_label} | {paid_str} | {case_str} | {delta_str} |")

    lines += [
        "",
        "### Paid vs Case-Incurred: Median CV(IBNR)",
        "",
        "| Method | CV(paid) | CV(case) | Delta |",
        "|--------|----------|----------|-------|",
    ]
    for method_label, row in cv_compare.iterrows():
        paid_cv = row.get("paid", np.nan)
        case_cv = row.get("case_incurred", np.nan)
        delta = row.get("delta", np.nan)
        paid_str = f"{paid_cv:.4f}" if np.isfinite(paid_cv) else "n/a"
        case_str = f"{case_cv:.4f}" if np.isfinite(case_cv) else "n/a"
        delta_str = f"{delta:+.4f}" if np.isfinite(delta) else "n/a"
        lines.append(f"| {method_label} | {paid_str} | {case_str} | {delta_str} |")

    lines += [
        "",
        "### Figures (v2)",
        "",
        "- `figures/standalone_implied_pctl_grid_v2.png` — 4×12 histogram grid (6 methods × 2 types)",
        "- `figures/standalone_pp_paid_v2.png` — PP chart for paid",
        "- `figures/standalone_pp_case_incurred_v2.png` — PP chart for case_incurred",
        "- `cache/standalone_calibration_v2.csv` — per-triangle calibration detail",
    ]

    readme_path = ANALYSIS_DIR / "STANDALONE_BACKTEST_README.md"
    existing = readme_path.read_text() if readme_path.exists() else ""
    readme_path.write_text(existing + "\n".join(lines) + "\n")
    print(f"  Updated: {readme_path}")


if __name__ == "__main__":
    cal_df, metrics = run_analysis()
