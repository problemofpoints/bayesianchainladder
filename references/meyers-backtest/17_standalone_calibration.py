"""17_standalone_calibration.py
================================
Calibration analysis for the standalone stochastic reserving back-test on
all 200 Meyers triangles.

Inputs (from run_stochastic_reserving.py)
-----------------------------------------
  references/meyers-backtest/cache/meyers_standalone_results.csv
  references/meyers-backtest/cache/meyers_standalone_samples.parquet

Outputs
-------
  figures/standalone_implied_pctl_grid.png
  figures/standalone_pp_paid.png
  figures/standalone_pp_case_incurred.png
  references/meyers-backtest/STANDALONE_BACKTEST_README.md
  (calibration tables printed to stdout)

Methodology
-----------
For each (lob, group_id, loss_type, method):
  1. Get actual_total_unpaid from reservetestr:
       actual_total_unpaid = actual_ultimate_total - paid_to_date_total
     where actual_ultimate_total comes from the reservetestr 'paid' or
     'case' actual ultimates (matching the loss_type).
  2. Compute implied_pctl = empirical CDF of actual_total_unpaid under
     the model's total IBNR samples (from the parquet).
  3. Aggregate calibration metrics per (loss_type, method):
       - mean implied pctl (ideal 0.5)
       - % in [0.25, 0.75] central 50%
       - % in [0.10, 0.90] central 80%
       - KS statistic vs uniform
       - median |% error| vs actual ultimate
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ANALYSIS_DIR, CACHE_DIR, FIGURES_DIR

RESULTS_CSV = CACHE_DIR / "meyers_standalone_results.csv"
SAMPLES_PARQUET = CACHE_DIR / "meyers_standalone_samples.parquet"

METHODS = ["mack", "odp", "odp_corr", "odp_bf", "odp_cc"]
LOSS_TYPES = ["paid", "case_incurred"]
LOBS = ["comauto", "ppauto", "wkcomp", "othliab"]

METHOD_LABELS = {
    "mack": "Mack",
    "odp": "ODP",
    "odp_corr": "Corr-ODP",
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
        # 'paid' actual ultimate is from reservetestr's paid dimension
        rows.append({
            "lob": rec.line,
            "group_id": rec.group_id,
            "loss_type": "paid",
            "actual_ultimate_total": rec.actual_ultimates.get("paid", np.nan),
        })
        # 'case_incurred' actual ultimate maps to the 'case' dimension in reservetestr
        rows.append({
            "lob": rec.line,
            "group_id": rec.group_id,
            "loss_type": "case_incurred",
            "actual_ultimate_total": rec.actual_ultimates.get("case", np.nan),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Compute implied percentile
# ---------------------------------------------------------------------------

def implied_pctl(actual_value: float, samples: np.ndarray) -> float:
    """Empirical CDF of actual_value under samples.

    Returns the fraction of samples that are <= actual_value.
    Uses mid-rank convention for ties.
    """
    if len(samples) == 0 or not np.isfinite(actual_value):
        return np.nan
    finite = samples[np.isfinite(samples)]
    if len(finite) == 0:
        return np.nan
    return float(np.mean(finite <= actual_value))


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading results CSV...")
    results = pd.read_csv(RESULTS_CSV)
    print(f"  {len(results):,} rows")

    print("Loading samples parquet...")
    samples_df = pd.read_parquet(SAMPLES_PARQUET)
    samples_df["group_id"] = samples_df["group_id"].astype(int)
    print(f"  {len(samples_df):,} rows, {samples_df.groupby(['lob','group_id','loss_type','method']).ngroups} combos")

    print("Loading actual ultimates from reservetestr...")
    actuals = load_actual_ultimates()
    print(f"  {len(actuals)} rows")

    # Get total-row paid_to_date from results (for IBNR computation check)
    total_results = results[results["accident_year"] == "Total"].copy()
    total_results["group_id"] = total_results["group_id"].astype(int)

    # Build per-group paid_to_date lookup (same for all methods/loss_types)
    paid_to_date_lookup = (
        total_results[total_results["loss_type"] == "paid"]
        .groupby(["lob", "group_id"])[["paid_to_date"]]
        .first()
        .reset_index()
    )
    paid_to_date_lookup.rename(columns={"paid_to_date": "paid_to_date_total"}, inplace=True)

    # Merge actuals with paid_to_date
    actuals = actuals.merge(paid_to_date_lookup, on=["lob", "group_id"], how="left")
    actuals["actual_total_unpaid"] = actuals["actual_ultimate_total"] - actuals["paid_to_date_total"]

    # Group samples by (lob, group_id, loss_type, method) → array of total_ibnr values
    print("Computing implied percentiles...")
    sample_groups = samples_df.groupby(["lob", "group_id", "loss_type", "method"])

    cal_rows = []
    for (lob, group_id, loss_type, method), grp in sample_groups:
        ibnr_samples = grp["total_ibnr"].values.astype(float)

        # Look up actual total unpaid
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

        # Mean IBNR from samples
        mean_ibnr_samples = float(np.mean(ibnr_samples[np.isfinite(ibnr_samples)]))
        mean_ult_est = paid_td + mean_ibnr_samples
        pct_err = (mean_ult_est - actual_ult) / actual_ult if actual_ult != 0 else np.nan

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
        })

    cal_df = pd.DataFrame(cal_rows)
    print(f"  Calibration table: {len(cal_df)} rows, "
          f"{cal_df.groupby(['loss_type','method']).ngroups} (loss_type,method) combos")

    # -----------------------------------------------------------------------
    # Calibration metrics per (loss_type, method)
    # -----------------------------------------------------------------------
    def calibration_metrics(group: pd.DataFrame) -> pd.Series:
        p = group["implied_pctl"].dropna()
        if len(p) == 0:
            return pd.Series({
                "n": 0, "mean_pctl": np.nan,
                "pct_in_central50": np.nan, "pct_in_central80": np.nan,
                "ks_stat": np.nan, "ks_pval": np.nan,
                "median_abs_pct_err": np.nan,
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
        })

    metrics = (
        cal_df.groupby(["loss_type", "method"])
        .apply(calibration_metrics, include_groups=False)
        .reset_index()
    )

    # -----------------------------------------------------------------------
    # Print calibration table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("CALIBRATION TABLE: 5 methods × 2 loss types")
    print("=" * 100)
    print(f"{'Method':<12} {'LossType':<16} {'N':>5} {'MeanPctl':>10} {'%in50':>8} "
          f"{'%in80':>8} {'KS_stat':>8} {'KS_pval':>8} {'Med|%err|':>10}")
    print("-" * 100)
    for _, row in metrics.sort_values(["loss_type", "method"]).iterrows():
        print(
            f"{METHOD_LABELS.get(row['method'], row['method']):<12} "
            f"{LOSS_LABELS.get(row['loss_type'], row['loss_type']):<16} "
            f"{row['n']:>5.0f} "
            f"{row['mean_pctl']:>10.3f} "
            f"{row['pct_in_central50']:>8.1f} "
            f"{row['pct_in_central80']:>8.1f} "
            f"{row['ks_stat']:>8.4f} "
            f"{row['ks_pval']:>8.4f} "
            f"{row['median_abs_pct_err']:>10.2f}"
        )

    # -----------------------------------------------------------------------
    # Per-LOB breakdown
    # -----------------------------------------------------------------------
    per_lob = (
        cal_df.groupby(["lob", "loss_type", "method"])
        .apply(calibration_metrics, include_groups=False)
        .reset_index()
    )

    print("\n" + "=" * 100)
    print("PER-LOB CALIBRATION: KS statistic")
    print("=" * 100)
    ks_pivot = per_lob.pivot_table(
        index=["lob", "loss_type"],
        columns="method",
        values="ks_stat",
    )
    print(ks_pivot.round(4).to_string())

    print("\n" + "=" * 100)
    print("PER-LOB CALIBRATION: % in central 80%")
    print("=" * 100)
    p80_pivot = per_lob.pivot_table(
        index=["lob", "loss_type"],
        columns="method",
        values="pct_in_central80",
    )
    print(p80_pivot.round(1).to_string())

    # -----------------------------------------------------------------------
    # Paid vs case_incurred comparison per method
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("PAID vs CASE_INCURRED COMPARISON (KS statistic, lower is better)")
    print("=" * 100)
    ks_compare = metrics.pivot_table(
        index="method", columns="loss_type", values="ks_stat"
    ).reindex(METHODS)
    ks_compare["delta"] = ks_compare["case_incurred"] - ks_compare["paid"]
    ks_compare.index = [METHOD_LABELS.get(m, m) for m in ks_compare.index]
    print(ks_compare.round(4).to_string())

    print("\nPAID vs CASE_INCURRED COMPARISON (% in central 80%, higher is better)")
    p80_compare = metrics.pivot_table(
        index="method", columns="loss_type", values="pct_in_central80"
    ).reindex(METHODS)
    p80_compare["delta"] = p80_compare["case_incurred"] - p80_compare["paid"]
    p80_compare.index = [METHOD_LABELS.get(m, m) for m in p80_compare.index]
    print(p80_compare.round(1).to_string())

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    _plot_implied_pctl_grid(cal_df)
    _plot_pp_chart(cal_df, "paid")
    _plot_pp_chart(cal_df, "case_incurred")

    # -----------------------------------------------------------------------
    # Write README
    # -----------------------------------------------------------------------
    _write_readme(metrics, ks_compare, p80_compare, per_lob, cal_df)

    print("\nDone. Figures and README written.")
    return cal_df, metrics


def _plot_implied_pctl_grid(cal_df: pd.DataFrame):
    """4 LOBs × (5 methods × 2 loss types) = 4×10 histograms of implied pctl."""
    n_lobs = len(LOBS)
    n_combos = len(METHODS) * len(LOSS_TYPES)

    fig, axes = plt.subplots(
        n_lobs, n_combos, figsize=(n_combos * 1.6, n_lobs * 2.4), sharey=False
    )
    fig.suptitle(
        "Implied Percentile Histograms: 4 LOBs × (5 Methods × 2 Loss Types)",
        fontsize=10, y=1.01,
    )

    for row_i, lob in enumerate(LOBS):
        col_i = 0
        for lt in LOSS_TYPES:
            for method in METHODS:
                ax = axes[row_i, col_i]
                subset = cal_df[
                    (cal_df["lob"] == lob)
                    & (cal_df["loss_type"] == lt)
                    & (cal_df["method"] == method)
                ]["implied_pctl"].dropna()
                ax.hist(subset, bins=10, range=(0, 1), color="#2196F3" if lt == "paid" else "#FF9800",
                        alpha=0.75, edgecolor="none")
                ax.axhline(len(subset) / 10, color="red", lw=0.7, ls="--")
                ax.set_xlim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                if row_i == 0:
                    ax.set_title(
                        f"{METHOD_LABELS[method]}\n{LOSS_LABELS[lt]}",
                        fontsize=6.5, pad=2
                    )
                if col_i == 0:
                    ax.set_ylabel(lob.upper(), fontsize=7, labelpad=4)
                col_i += 1

    plt.tight_layout()
    out = FIGURES_DIR / "standalone_implied_pctl_grid.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def _plot_pp_chart(cal_df: pd.DataFrame, loss_type: str):
    """PP (probability-probability) chart for all methods on a given loss_type."""
    fig, ax = plt.subplots(figsize=(6, 6))

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
    theoretical = np.linspace(0, 1, 101)

    for method, color in zip(METHODS, colors):
        p = cal_df[
            (cal_df["loss_type"] == loss_type) & (cal_df["method"] == method)
        ]["implied_pctl"].dropna().sort_values().values
        if len(p) == 0:
            continue
        # Empirical CDF of the implied percentiles
        emp = np.arange(1, len(p) + 1) / len(p)
        theo = p  # theoretical = the implied_pctl values themselves
        ax.plot(theo, emp, label=METHOD_LABELS[method], color=color, lw=1.8)

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
    ax.set_xlabel("Theoretical quantile (implied pctl)")
    ax.set_ylabel("Empirical CDF")
    lt_label = LOSS_LABELS[loss_type]
    ax.set_title(f"PP Chart — {lt_label} — All 200 triangles")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    out = FIGURES_DIR / f"standalone_pp_{loss_type}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def _write_readme(
    metrics: pd.DataFrame,
    ks_compare: pd.DataFrame,
    p80_compare: pd.DataFrame,
    per_lob: pd.DataFrame,
    cal_df: pd.DataFrame,
):
    """Write STANDALONE_BACKTEST_README.md with key findings."""

    # Find best method per loss_type (lowest KS stat)
    best_paid = metrics[metrics["loss_type"] == "paid"].sort_values("ks_stat").iloc[0]
    best_case = metrics[metrics["loss_type"] == "case_incurred"].sort_values("ks_stat").iloc[0]

    # Summary stats for writing
    metrics_sorted = metrics.sort_values(["loss_type", "ks_stat"])

    lines = [
        "# Standalone Stochastic Reserving Back-test Results",
        "",
        "**Date generated**: see git log.",
        "",
        "## Setup",
        "",
        "- **Data**: 200 Meyers (CAS Monograph 1) triangles — 50 each for",
        "  comauto, ppauto, wkcomp, othliab",
        "- **Methods**: Mack, ODP, Corr-ODP (rho=0.1), ODP+BF (apriori=0.65), ODP+CC",
        "- **Loss types**: paid and case_incurred",
        "- **Sims**: 5,000 per triangle/method",
        "- **Script**: `scripts/run_stochastic_reserving.py`",
        "",
        "## Calibration Summary",
        "",
        "Implied percentile = empirical CDF of actual total unpaid under each model's",
        "simulated total IBNR distribution. Perfect calibration: uniform on [0,1].",
        "KS statistic measures departure from uniform (lower = better).",
        "",
        "### Calibration table (all 200 triangles per combo)",
        "",
    ]

    # Header
    lines.append("| Method | Loss type | N | Mean pctl | % in 50% | % in 80% | KS stat | Med |%err| |")
    lines.append("|--------|-----------|---|-----------|----------|----------|---------|------------|")
    for _, row in metrics_sorted.iterrows():
        lines.append(
            f"| {METHOD_LABELS.get(row['method'], row['method'])} "
            f"| {LOSS_LABELS.get(row['loss_type'], row['loss_type'])} "
            f"| {row['n']:.0f} "
            f"| {row['mean_pctl']:.3f} "
            f"| {row['pct_in_central50']:.1f}% "
            f"| {row['pct_in_central80']:.1f}% "
            f"| {row['ks_stat']:.4f} "
            f"| {row['median_abs_pct_err']:.2f}% |"
        )

    lines += [
        "",
        "## Paid vs Case-Incurred",
        "",
        "Delta = case_incurred KS − paid KS (negative = case_incurred is better calibrated).",
        "",
        "| Method | KS (paid) | KS (case) | Delta |",
        "|--------|-----------|-----------|-------|",
    ]
    for method_label, row in ks_compare.iterrows():
        paid_ks = row.get("paid", np.nan)
        case_ks = row.get("case_incurred", np.nan)
        delta = row.get("delta", np.nan)
        lines.append(
            f"| {method_label} "
            f"| {paid_ks:.4f} "
            f"| {case_ks:.4f} "
            f"| {delta:+.4f} |"
        )

    lines += [
        "",
        "## Headline Findings",
        "",
        f"- **Best-calibrated method on paid data**: "
        f"{METHOD_LABELS.get(best_paid['method'], best_paid['method'])} "
        f"(KS={best_paid['ks_stat']:.4f}, {best_paid['pct_in_central80']:.1f}% in central 80%)",
        f"- **Best-calibrated method on case_incurred data**: "
        f"{METHOD_LABELS.get(best_case['method'], best_case['method'])} "
        f"(KS={best_case['ks_stat']:.4f}, {best_case['pct_in_central80']:.1f}% in central 80%)",
        "",
    ]

    # Paid vs case overall
    avg_ks_paid = metrics[metrics["loss_type"] == "paid"]["ks_stat"].mean()
    avg_ks_case = metrics[metrics["loss_type"] == "case_incurred"]["ks_stat"].mean()
    direction = "worse" if avg_ks_case > avg_ks_paid else "better"
    lines += [
        f"- **Paid vs case_incurred overall**: Average KS across methods: "
        f"paid={avg_ks_paid:.4f}, case_incurred={avg_ks_case:.4f}. "
        f"Case-incurred modelling is on average *{direction}*-calibrated vs paid.",
        "",
        "## Figures",
        "",
        "- `figures/standalone_implied_pctl_grid.png` — 4×10 histogram grid",
        "- `figures/standalone_pp_paid.png` — PP chart for paid (all methods)",
        "- `figures/standalone_pp_case_incurred.png` — PP chart for case_incurred",
    ]

    readme_path = ANALYSIS_DIR / "STANDALONE_BACKTEST_README.md"
    readme_path.write_text("\n".join(lines) + "\n")
    print(f"  Saved: {readme_path}")


if __name__ == "__main__":
    cal_df, metrics = run_analysis()
