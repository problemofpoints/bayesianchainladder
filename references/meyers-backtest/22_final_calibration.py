"""22_final_calibration.py
==========================
Final calibration analysis for the 9-method stochastic reserving comparison.

Loads cache/<prefix>.csv and cache/<prefix>_samples.parquet, joins with actual
ultimates, and computes full calibration metrics for all methods × 2 loss
types (method, loss_type) combinations.

Outputs
-------
  figures/<prefix>_calibration_grid.png  — histogram grid of implied percentiles
  figures/<prefix>_pp_chart.png          — PP chart, methods × 2 loss types overlaid
  (printed calibration table, per-line breakdown, and verdict)

Usage
-----
  cd /Users/atroyer/Projects/bayesianchainladder
  uv run python references/meyers-backtest/22_final_calibration.py --dataset meyers
  uv run python references/meyers-backtest/22_final_calibration.py --dataset clrd2025 --prefix clrd2025_final
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CACHE_DIR, FIGURES_DIR, ANALYSIS_DIR

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASETS = {
    "meyers": {
        "prefix": "meyers_final",
        "lobs": ["comauto", "ppauto", "wkcomp", "othliab"],
        "title": "Meyers (2015) 200 triangles, origins 1988-1997",
    },
    "clrd2025": {
        "prefix": "clrd2025_final",
        "lobs": ["comauto", "ppauto", "wkcomp", "othliab", "prodliab", "medmal"],
        "title": "clrd2025 Meyers-style window, origins 1998-2007",
    },
}

ALL_METHODS = [
    "mack",
    "odp",
    "odp_param",
    "odp_corr",
    "odp_bf",
    "odp_cc",
    "odp_corr_bf",
    "odp_corr_cc",
    "bz",
]
METHOD_LABELS = {
    "mack": "Mack",
    "odp": "ODP (non-param)",
    "odp_param": "ODP param (rho=0)",
    "odp_corr": "ODP corr (rho=0.3)",
    "odp_bf": "BF param (rho=0)",
    "odp_cc": "CC param (rho=0)",
    "odp_corr_bf": "BF corr (rho=0.3)",
    "odp_corr_cc": "CC corr (rho=0.3)",
    "bz": "Barnett-Zehnwirth PTF",
}
LOSS_TYPES = ["paid", "case_incurred"]
LOSS_LABELS = {"paid": "Paid", "case_incurred": "Case Incurred"}
LOB_LABELS = {
    "comauto": "Comm Auto",
    "ppauto": "PP Auto",
    "wkcomp": "Workers Comp",
    "othliab": "Other Liab",
    "prodliab": "Products Liab",
    "medmal": "Med Mal",
}


# ---------------------------------------------------------------------------
# Load actual ultimates
# ---------------------------------------------------------------------------

def load_actual_ultimates(dataset: str) -> pd.DataFrame:
    """DataFrame with lob, group_id, loss_type, actual_ultimate_total."""
    if dataset == "clrd2025":
        path = CACHE_DIR / "clrd2025_actuals.csv"
        if not path.exists():
            sys.exit(f"ERROR: {path} not found. Run 24_build_clrd2025_long.py first.")
        return pd.read_csv(path)

    try:
        import reservetestr as rt
    except ImportError:
        path = CACHE_DIR / "meyers_actuals_source.csv"
        if not path.exists():
            sys.exit(
                "ERROR: reservetestr is not installed and "
                f"{path} (a previous *_cal_detail.csv) is missing."
            )
        df = pd.read_csv(path)[["lob", "group_id", "loss_type", "actual_ultimate"]]
        return df.drop_duplicates().rename(columns={"actual_ultimate": "actual_ultimate_total"})

    records = rt.build_triangle_records()
    rows = []
    for rec in records:
        rows.append({"lob": rec.line, "group_id": rec.group_id, "loss_type": "paid",
                     "actual_ultimate_total": rec.actual_ultimates.get("paid", np.nan)})
        rows.append({"lob": rec.line, "group_id": rec.group_id, "loss_type": "case_incurred",
                     "actual_ultimate_total": rec.actual_ultimates.get("case", np.nan)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def implied_pctl(actual_value: float, samples: np.ndarray) -> float:
    """Fraction of samples <= actual_value."""
    if len(samples) == 0 or not np.isfinite(actual_value):
        return np.nan
    finite = samples[np.isfinite(samples)]
    if len(finite) == 0:
        return np.nan
    return float(np.mean(finite <= actual_value))


def calibration_metrics(
    p: np.ndarray,
    cv_vals: np.ndarray | None = None,
    pct_err_vals: np.ndarray | None = None,
) -> dict:
    """Compute calibration metrics for a vector of implied percentiles."""
    p = p[np.isfinite(p)]
    if len(p) == 0:
        return {
            "n": 0,
            "mean_pctl": np.nan,
            "pct_in_central50": np.nan,
            "pct_in_central80": np.nan,
            "ks_stat": np.nan,
            "ks_pval": np.nan,
            "median_cv_ibnr": np.nan,
            "median_abs_pct_err": np.nan,
        }
    ks_stat, ks_pval = stats.kstest(p, "uniform")
    med_cv = (
        float(np.median(cv_vals[np.isfinite(cv_vals)]))
        if cv_vals is not None and np.any(np.isfinite(cv_vals))
        else np.nan
    )
    med_abs_err = (
        float(np.median(np.abs(pct_err_vals[np.isfinite(pct_err_vals)])))
        if pct_err_vals is not None and np.any(np.isfinite(pct_err_vals))
        else np.nan
    )
    return {
        "n": len(p),
        "mean_pctl": float(p.mean()),
        "pct_in_central50": float(((p >= 0.25) & (p <= 0.75)).mean() * 100),
        "pct_in_central80": float(((p >= 0.10) & (p <= 0.90)).mean() * 100),
        "ks_stat": float(ks_stat),
        "ks_pval": float(ks_pval),
        "median_cv_ibnr": med_cv,
        "median_abs_pct_err": med_abs_err,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(dataset: str = "meyers", prefix: str | None = None):
    cfg = DATASETS[dataset]
    prefix = prefix or cfg["prefix"]
    lobs = cfg["lobs"]
    results_csv = CACHE_DIR / f"{prefix}.csv"
    samples_parquet = CACHE_DIR / f"{prefix}_samples.parquet"

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"Loading results from {results_csv}...")
    if not results_csv.exists():
        print(f"ERROR: {results_csv} not found. Run the Meyers sweep first.")
        sys.exit(1)
    results = pd.read_csv(results_csv)
    print(f"  results: {len(results):,} rows")

    print(f"Loading samples from {samples_parquet}...")
    if not samples_parquet.exists():
        print(f"ERROR: {samples_parquet} not found. Run the Meyers sweep first.")
        sys.exit(1)
    samples_df = pd.read_parquet(samples_parquet)
    samples_df["group_id"] = samples_df["group_id"].astype(int)
    n_combos = samples_df.groupby(["lob", "group_id", "loss_type", "method"]).ngroups
    print(f"  samples: {len(samples_df):,} rows, {n_combos} (lob,group,loss_type,method) combos")

    # ------------------------------------------------------------------
    # 2. Load actual ultimates
    # ------------------------------------------------------------------
    print("\nLoading actual ultimates...")
    actuals = load_actual_ultimates(dataset)
    actuals["group_id"] = actuals["group_id"].astype(int)
    print(f"  {len(actuals)} rows")

    # Build paid-to-date lookup from total rows
    total_results = results[results["accident_year"] == "Total"].copy()
    total_results["group_id"] = total_results["group_id"].astype(int)
    paid_lookup = (
        total_results[total_results["loss_type"] == "paid"]
        .groupby(["lob", "group_id"])[["paid_to_date"]]
        .first()
        .reset_index()
        .rename(columns={"paid_to_date": "paid_to_date_total"})
    )

    actuals_aug = actuals.merge(paid_lookup, on=["lob", "group_id"], how="left")
    actuals_aug["actual_total_unpaid"] = (
        actuals_aug["actual_ultimate_total"] - actuals_aug["paid_to_date_total"]
    )

    # ------------------------------------------------------------------
    # 3. Compute implied percentiles for every (lob, group, loss_type, method)
    # ------------------------------------------------------------------
    print("\nComputing implied percentiles...")
    all_cal_rows = []

    for (lob, group_id, loss_type, method), grp in samples_df.groupby(
        ["lob", "group_id", "loss_type", "method"]
    ):
        ibnr_samples = grp["total_ibnr"].values.astype(float)

        act_row = actuals_aug[
            (actuals_aug["lob"] == lob)
            & (actuals_aug["group_id"] == group_id)
            & (actuals_aug["loss_type"] == loss_type)
        ]
        if act_row.empty or not np.isfinite(act_row.iloc[0]["actual_total_unpaid"]):
            continue

        actual_unpaid = float(act_row.iloc[0]["actual_total_unpaid"])
        actual_ult = float(act_row.iloc[0]["actual_ultimate_total"])
        paid_td = float(act_row.iloc[0].get("paid_to_date_total", np.nan))

        pctl = implied_pctl(actual_unpaid, ibnr_samples)

        finite_samp = ibnr_samples[np.isfinite(ibnr_samples)]
        mean_ibnr = float(np.mean(finite_samp)) if len(finite_samp) > 0 else np.nan
        cv_val = (
            float(np.std(finite_samp, ddof=1) / np.abs(np.mean(finite_samp)))
            if len(finite_samp) > 1 and np.mean(finite_samp) != 0
            else np.nan
        )
        mean_ult_est = paid_td + mean_ibnr if np.isfinite(paid_td) else np.nan
        pct_err = (
            (mean_ult_est - actual_ult) / actual_ult
            if np.isfinite(mean_ult_est) and actual_ult != 0
            else np.nan
        )

        all_cal_rows.append({
            "lob": lob,
            "group_id": group_id,
            "loss_type": loss_type,
            "method": method,
            "actual_ultimate": actual_ult,
            "paid_to_date": paid_td,
            "actual_unpaid": actual_unpaid,
            "mean_ibnr_est": mean_ibnr,
            "pct_err": pct_err,
            "implied_pctl": pctl,
            "cv_ibnr": cv_val,
        })

    cal_df = pd.DataFrame(all_cal_rows)
    print(f"  {len(cal_df):,} calibration rows computed")

    # ------------------------------------------------------------------
    # 4. Summary calibration table (all methods × 2 loss types)
    # ------------------------------------------------------------------
    print("\n" + "=" * 130)
    print(f"CALIBRATION TABLE [{dataset}]: {len(ALL_METHODS)} methods × 2 loss types")
    print("Lognormal process variance, rho=0.3 for correlated methods, apriori=0.65 for BF")
    print("=" * 130)

    header = (
        f"{'Method':<18} {'LossType':<14} {'N':>5} "
        f"{'MeanPctl':>9} {'C50%':>7} {'C80%':>7} "
        f"{'KS':>6} {'KS_p':>7} "
        f"{'MedCV':>7} {'Med|%err|':>10}"
    )
    print(header)
    print("-" * 130)

    summary_rows = []
    for method in ALL_METHODS:
        for loss_type in LOSS_TYPES:
            sub = cal_df[(cal_df["method"] == method) & (cal_df["loss_type"] == loss_type)]
            if sub.empty:
                continue
            p = sub["implied_pctl"].values
            cv_v = sub["cv_ibnr"].values
            pe_v = sub["pct_err"].values
            m = calibration_metrics(p, cv_v, pe_v)
            print(
                f"{method:<18} {loss_type:<14} {m['n']:>5} "
                f"{m['mean_pctl']:>9.3f} {m['pct_in_central50']:>7.1f} {m['pct_in_central80']:>7.1f} "
                f"{m['ks_stat']:>6.3f} {m['ks_pval']:>7.3f} "
                f"{m['median_cv_ibnr']:>7.3f} {m['median_abs_pct_err']:>10.3f}"
            )
            summary_rows.append({
                "method": method,
                "loss_type": loss_type,
                **m,
            })

    summary_df = pd.DataFrame(summary_rows)

    # ------------------------------------------------------------------
    # 5. Per-line breakdown
    # ------------------------------------------------------------------
    print("\n" + "=" * 130)
    print("PER-LINE KS STATISTICS (all methods)")
    print("=" * 130)

    for lob in lobs:
        print(f"\n--- {LOB_LABELS.get(lob, lob)} ---")
        lob_header = f"{'Method':<18} {'LossType':<14} {'N':>4} {'KS':>6} {'MeanPctl':>9} {'C80%':>7}"
        print(lob_header)
        for method in ALL_METHODS:
            for loss_type in LOSS_TYPES:
                sub = cal_df[
                    (cal_df["lob"] == lob)
                    & (cal_df["method"] == method)
                    & (cal_df["loss_type"] == loss_type)
                ]
                if sub.empty:
                    continue
                p = sub["implied_pctl"].values
                m = calibration_metrics(p)
                print(
                    f"{method:<18} {loss_type:<14} {m['n']:>4} "
                    f"{m['ks_stat']:>6.3f} {m['mean_pctl']:>9.3f} {m['pct_in_central80']:>7.1f}"
                )

    # ------------------------------------------------------------------
    # 6. Histogram grid: all methods × 2 loss types
    # ------------------------------------------------------------------
    print("\nGenerating calibration histogram grid...")
    methods_to_plot = [m for m in ALL_METHODS if m in cal_df["method"].unique()]
    n_methods = len(methods_to_plot)
    n_cols = 2  # one column per loss_type
    n_rows = n_methods

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(12, 2.8 * n_rows),
        sharey=False,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, method in enumerate(methods_to_plot):
        for col_idx, loss_type in enumerate(LOSS_TYPES):
            ax = axes[row_idx, col_idx]
            sub = cal_df[(cal_df["method"] == method) & (cal_df["loss_type"] == loss_type)]
            p = sub["implied_pctl"].dropna().values

            if len(p) == 0:
                ax.set_visible(False)
                continue

            ks_stat, ks_pval = stats.kstest(p, "uniform")
            mean_p = float(p.mean())

            ax.hist(p, bins=20, range=(0, 1), density=False, color="#2196F3",
                    edgecolor="white", linewidth=0.5, alpha=0.85)
            ax.axhline(len(p) / 20, color="#FF5722", linestyle="--", linewidth=1.2,
                       label="Uniform")
            ax.set_xlim(0, 1)
            ax.set_title(
                f"{METHOD_LABELS.get(method, method)} — {LOSS_LABELS[loss_type]}\n"
                f"n={len(p)}, KS={ks_stat:.3f}, mean={mean_p:.3f}",
                fontsize=9,
            )
            ax.set_xlabel("Implied Percentile", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.tick_params(labelsize=7)
            if row_idx == 0:
                ax.legend(fontsize=7)

    fig.suptitle(
        f"{cfg['title']}\n"
        f"{len(methods_to_plot)} Methods × 2 Loss Types (lognormal PV, rho=0.3, n=5000 sims)",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    out_path = FIGURES_DIR / f"{prefix}_calibration_grid.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # ------------------------------------------------------------------
    # 7. PP chart
    # ------------------------------------------------------------------
    print("Generating PP chart...")
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(methods_to_plot)))
    for col_idx, loss_type in enumerate(LOSS_TYPES):
        ax = axes2[col_idx]
        for m_idx, method in enumerate(methods_to_plot):
            sub = cal_df[(cal_df["method"] == method) & (cal_df["loss_type"] == loss_type)]
            p = np.sort(sub["implied_pctl"].dropna().values)
            if len(p) == 0:
                continue
            ecdf = np.arange(1, len(p) + 1) / len(p)
            ks_stat, _ = stats.kstest(p, "uniform")
            ax.plot(
                p, ecdf,
                color=colors[m_idx],
                linewidth=1.5,
                label=f"{METHOD_LABELS.get(method, method)} (KS={ks_stat:.3f})",
                alpha=0.85,
            )
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.5, label="Ideal")
        ax.set_xlabel("Theoretical Quantile", fontsize=10)
        ax.set_ylabel("Empirical CDF", fontsize=10)
        ax.set_title(f"PP Chart — {LOSS_LABELS[loss_type]}", fontsize=11)
        ax.legend(fontsize=7, loc="upper left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

    fig2.suptitle(
        f"{cfg['title']}\n"
        "Lognormal PV, rho=0.3 for correlated methods",
        fontsize=11,
    )
    fig2.tight_layout()
    out_pp = FIGURES_DIR / f"{prefix}_pp_chart.png"
    fig2.savefig(out_pp, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {out_pp}")

    # ------------------------------------------------------------------
    # 8. Verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("VERDICT — KS statistics sorted (lower = better calibrated)")
    print("=" * 90)
    ks_summary = (
        summary_df[["method", "loss_type", "ks_stat", "mean_pctl", "pct_in_central80"]]
        .sort_values("ks_stat")
        .reset_index(drop=True)
    )
    print(ks_summary.to_string(index=False))

    # Best methods per loss type
    for lt in LOSS_TYPES:
        sub = summary_df[summary_df["loss_type"] == lt].sort_values("ks_stat")
        best = sub.iloc[0]
        print(
            f"\nBest for {LOSS_LABELS[lt]}: {best['method']} "
            f"(KS={best['ks_stat']:.3f}, mean_pctl={best['mean_pctl']:.3f}, "
            f"C80%={best['pct_in_central80']:.1f}%)"
        )

    # ------------------------------------------------------------------
    # 9. Save calibration table to CSV
    # ------------------------------------------------------------------
    cal_out = CACHE_DIR / f"{prefix}_calibration.csv"
    summary_df.to_csv(cal_out, index=False)
    print(f"\nCalibration summary saved to {cal_out}")

    cal_detail_out = CACHE_DIR / f"{prefix}_cal_detail.csv"
    cal_df.to_csv(cal_detail_out, index=False)
    print(f"Per-triangle detail saved to {cal_detail_out}")

    return summary_df, cal_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Calibration analysis for the standalone reserving back-test.")
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="meyers")
    ap.add_argument("--prefix", default=None,
                    help="Results file prefix in cache/ (default per dataset: meyers_final / clrd2025_final)")
    a = ap.parse_args()
    run_analysis(a.dataset, a.prefix)
