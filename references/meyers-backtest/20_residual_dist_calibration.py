"""20_residual_dist_calibration.py
==================================
Compare calibration across three correlated ODP bootstrap variants:

  1. odp_corr_normal  — Normal residuals,    rho=0.1  (current default)
  2. odp_corr_t       — Student-t residuals, rho=0.3  (kurt-implied df ∈ [3,15])
  3. odp_corr_skewt   — Hansen skew-t,       rho=0.3  (kurt+skew implied params)

Inputs
------
  cache/meyers_corr_normal.csv + meyers_corr_normal_samples.parquet
  cache/meyers_corr_t.csv      + meyers_corr_t_samples.parquet
  cache/meyers_corr_skewt.csv  + meyers_corr_skewt_samples.parquet

Outputs
-------
  figures/residual_dist_calibration_grid.png   — implied-pctl histogram grid
  figures/residual_dist_pp_chart.png           — PP chart comparing 3 configs × 2 loss types
  cache/residual_dist_calibration.csv          — per-triangle calibration table

Usage
-----
  cd /Users/atroyer/Projects/bayesianchainladder
  uv run python references/meyers-backtest/20_residual_dist_calibration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CACHE_DIR, FIGURES_DIR, ANALYSIS_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CONFIGS = {
    "normal (rho=0.1)": {
        "csv": CACHE_DIR / "meyers_corr_normal.csv",
        "parquet": CACHE_DIR / "meyers_corr_normal_samples.parquet",
        "rho": 0.1,
        "dist": "normal",
    },
    "t (rho=0.3)": {
        "csv": CACHE_DIR / "meyers_corr_t.csv",
        "parquet": CACHE_DIR / "meyers_corr_t_samples.parquet",
        "rho": 0.3,
        "dist": "t",
    },
    "skewt (rho=0.3)": {
        "csv": CACHE_DIR / "meyers_corr_skewt.csv",
        "parquet": CACHE_DIR / "meyers_corr_skewt_samples.parquet",
        "rho": 0.3,
        "dist": "skewt",
    },
}

LOSS_TYPES = ["paid", "case_incurred"]
LOBS = ["comauto", "ppauto", "wkcomp", "othliab"]

LOSS_LABELS = {"paid": "Paid", "case_incurred": "Case Incurred"}

# ---------------------------------------------------------------------------
# Load actual ultimates
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
# Implied percentile helpers
# ---------------------------------------------------------------------------

def implied_pctl(actual_value: float, samples: np.ndarray) -> float:
    """Fraction of samples <= actual_value."""
    if len(samples) == 0 or not np.isfinite(actual_value):
        return np.nan
    finite = samples[np.isfinite(samples)]
    if len(finite) == 0:
        return np.nan
    return float(np.mean(finite <= actual_value))


def calibration_metrics(p: np.ndarray, cv_vals: np.ndarray | None = None) -> dict:
    """Compute calibration metrics for a vector of implied percentiles."""
    p = p[np.isfinite(p)]
    if len(p) == 0:
        return {
            "n": 0, "mean_pctl": np.nan, "pct_in_central50": np.nan,
            "pct_in_central80": np.nan, "ks_stat": np.nan, "ks_pval": np.nan,
            "median_cv_ibnr": np.nan,
        }
    ks_stat, ks_pval = stats.kstest(p, "uniform")
    med_cv = float(np.median(cv_vals[np.isfinite(cv_vals)])) if cv_vals is not None and len(cv_vals) > 0 else np.nan
    return {
        "n": len(p),
        "mean_pctl": float(p.mean()),
        "pct_in_central50": float(((p >= 0.25) & (p <= 0.75)).mean() * 100),
        "pct_in_central80": float(((p >= 0.10) & (p <= 0.90)).mean() * 100),
        "ks_stat": float(ks_stat),
        "ks_pval": float(ks_pval),
        "median_cv_ibnr": med_cv,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading actual ultimates from reservetestr...")
    actuals = load_actual_ultimates()
    print(f"  {len(actuals)} rows")

    # We'll collect per-config, per-triangle calibration rows
    all_cal_rows = []

    for config_label, cfg in CONFIGS.items():
        if not cfg["csv"].exists() or not cfg["parquet"].exists():
            print(f"  [SKIP] {config_label}: missing files {cfg['csv']} or {cfg['parquet']}")
            continue

        print(f"\nLoading {config_label}...")
        results = pd.read_csv(cfg["csv"])
        samples_df = pd.read_parquet(cfg["parquet"])
        samples_df["group_id"] = samples_df["group_id"].astype(int)
        print(f"  results: {len(results):,} rows")
        print(f"  samples: {len(samples_df):,} rows, "
              f"{samples_df.groupby(['lob', 'group_id', 'loss_type', 'method']).ngroups} combos")

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

        # Compute implied percentiles per (lob, group_id, loss_type)
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
                "config": config_label,
                "residual_dist": cfg["dist"],
                "rho": cfg["rho"],
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
    cal_out = CACHE_DIR / "residual_dist_calibration.csv"
    cal_df.to_csv(cal_out, index=False)
    print(f"\nSaved calibration table: {cal_out} ({len(cal_df)} rows)")

    # ---------------------------------------------------------------------------
    # Calibration metrics table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 130)
    print("CALIBRATION TABLE: 3 residual-dist configs × 2 loss types")
    print("Ideal: mean_pctl=0.5, %in50=50, %in80=80, ks_stat~0 (uniform), small cv")
    print("=" * 130)
    print(
        f"{'Config':<22} {'Loss type':<16} {'N':>5} {'MeanPctl':>10} {'%in50':>7} "
        f"{'%in80':>7} {'KS_stat':>8} {'KS_pval':>8} {'MedCV':>8}"
    )
    print("-" * 130)

    summary_rows = []
    for config_label in CONFIGS:
        for lt in LOSS_TYPES:
            sub = cal_df[(cal_df["config"] == config_label) & (cal_df["loss_type"] == lt)]
            if sub.empty:
                continue
            p = sub["implied_pctl"].values
            cv_vals = sub["cv_ibnr"].values
            m = calibration_metrics(p, cv_vals)
            print(
                f"{config_label:<22} {LOSS_LABELS.get(lt, lt):<16} "
                f"{m['n']:>5} {m['mean_pctl']:>10.3f} {m['pct_in_central50']:>7.1f} "
                f"{m['pct_in_central80']:>7.1f} {m['ks_stat']:>8.4f} "
                f"{m['ks_pval']:>8.4f} {m['median_cv_ibnr']:>8.3f}"
            )
            summary_rows.append({
                "config": config_label,
                "loss_type": lt,
                **m,
            })

    summary_df = pd.DataFrame(summary_rows)

    # ---------------------------------------------------------------------------
    # Per-line calibration table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 130)
    print("PER-LINE CALIBRATION: KS statistic by (config, lob, loss_type)")
    print("-" * 130)
    print(f"{'Config':<22} {'LOB':<10} {'Paid KS':>10} {'Case KS':>10} {'Paid %80':>10} {'Case %80':>10} {'Paid MCV':>10} {'Case MCV':>10}")
    print("-" * 130)

    line_rows = []
    for config_label in CONFIGS:
        for lob in LOBS:
            row = {"config": config_label, "lob": lob}
            for lt in LOSS_TYPES:
                sub = cal_df[
                    (cal_df["config"] == config_label)
                    & (cal_df["lob"] == lob)
                    & (cal_df["loss_type"] == lt)
                ]
                p = sub["implied_pctl"].values
                cv_vals = sub["cv_ibnr"].values
                m = calibration_metrics(p, cv_vals)
                row[f"ks_{lt}"] = m["ks_stat"]
                row[f"pct80_{lt}"] = m["pct_in_central80"]
                row[f"mcv_{lt}"] = m["median_cv_ibnr"]
            line_rows.append(row)
            print(
                f"{config_label:<22} {lob:<10} "
                f"{row.get('ks_paid', np.nan):>10.4f} {row.get('ks_case_incurred', np.nan):>10.4f} "
                f"{row.get('pct80_paid', np.nan):>10.1f} {row.get('pct80_case_incurred', np.nan):>10.1f} "
                f"{row.get('mcv_paid', np.nan):>10.3f} {row.get('mcv_case_incurred', np.nan):>10.3f}"
            )

    # ---------------------------------------------------------------------------
    # Verdict
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 130)
    print("VERDICT")
    print("=" * 130)

    # Compare paid KS: normal vs t vs skewt
    for lt in LOSS_TYPES:
        lt_label = LOSS_LABELS[lt]
        cfgs = [(c, summary_df[(summary_df["config"]==c) & (summary_df["loss_type"]==lt)]) for c in CONFIGS]
        cfgs_ok = [(c, df.iloc[0]) for c, df in cfgs if not df.empty]
        if not cfgs_ok:
            continue
        best = min(cfgs_ok, key=lambda x: x[1]["ks_stat"])
        print(f"\n  {lt_label}:")
        for c, row in cfgs_ok:
            mark = " <-- BEST" if c == best[0] else ""
            print(f"    {c:<22}: KS={row['ks_stat']:.4f}  %in80={row['pct_in_central80']:.1f}%  "
                  f"MedCV={row['median_cv_ibnr']:.3f}  MeanPctl={row['mean_pctl']:.3f}{mark}")

    best_ks_paid = min(
        [(c, summary_df[(summary_df["config"]==c) & (summary_df["loss_type"]=="paid")].iloc[0]["ks_stat"])
         for c in CONFIGS
         if not summary_df[(summary_df["config"]==c) & (summary_df["loss_type"]=="paid")].empty]
        , key=lambda x: x[1]
    )
    best_ks_case = min(
        [(c, summary_df[(summary_df["config"]==c) & (summary_df["loss_type"]=="case_incurred")].iloc[0]["ks_stat"])
         for c in CONFIGS
         if not summary_df[(summary_df["config"]==c) & (summary_df["loss_type"]=="case_incurred")].empty]
        , key=lambda x: x[1]
    )

    # Get normal baseline KS values for comparison
    norm_paid = summary_df[(summary_df["config"].str.startswith("normal")) & (summary_df["loss_type"]=="paid")]
    norm_case = summary_df[(summary_df["config"].str.startswith("normal")) & (summary_df["loss_type"]=="case_incurred")]
    baseline_ks_paid = float(norm_paid.iloc[0]["ks_stat"]) if not norm_paid.empty else np.nan
    baseline_ks_case = float(norm_case.iloc[0]["ks_stat"]) if not norm_case.empty else np.nan

    print("\n  RECOMMENDATION:")
    improvement_threshold = 0.05  # At least 5 percentage points drop in KS to call it an improvement

    for lt, baseline_ks, best_label, best_ks in [
        ("paid", baseline_ks_paid, best_ks_paid[0], best_ks_paid[1]),
        ("case_incurred", baseline_ks_case, best_ks_case[0], best_ks_case[1]),
    ]:
        improvement = baseline_ks - best_ks
        if improvement >= improvement_threshold and not best_label.startswith("normal"):
            print(f"  [{lt}] Meaningful improvement: {best_label} reduces KS by {improvement:.4f} "
                  f"({baseline_ks:.4f} -> {best_ks:.4f}). Consider making this the default.")
        else:
            print(f"  [{lt}] No meaningful improvement from heavier tails (best KS delta = {improvement:.4f}). "
                  f"Residual distribution does not fix the parametric model's calibration issues.")

    # ---------------------------------------------------------------------------
    # Figures
    # ---------------------------------------------------------------------------
    _plot_calibration_grid(cal_df)
    _plot_pp_chart(cal_df, summary_df)

    print("\nDone.")
    return cal_df, summary_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_calibration_grid(cal_df: pd.DataFrame) -> None:
    """4 LOBs × 3 configs × 2 loss types histogram grid of implied percentile."""
    configs = list(CONFIGS.keys())
    n_lobs = len(LOBS)
    n_cols = len(configs) * len(LOSS_TYPES)

    fig, axes = plt.subplots(
        n_lobs, n_cols, figsize=(n_cols * 1.6, n_lobs * 2.0), sharey=False
    )
    if n_lobs == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    dist_colors = {
        "normal (rho=0.1)": {"paid": "#2196F3", "case_incurred": "#1565C0"},
        "t (rho=0.3)":       {"paid": "#43A047", "case_incurred": "#1B5E20"},
        "skewt (rho=0.3)":   {"paid": "#FB8C00", "case_incurred": "#E65100"},
    }

    fig.suptitle(
        "Implied Percentile Histograms: 4 LOBs × 3 Residual Configs × 2 Loss Types\n"
        "(Uniform = perfect calibration; red dashed = expected frequency)",
        fontsize=9, y=1.01,
    )

    for row_i, lob in enumerate(LOBS):
        col_i = 0
        for lt in LOSS_TYPES:
            for config_label in configs:
                ax = axes[row_i, col_i]
                sub = cal_df[
                    (cal_df["lob"] == lob)
                    & (cal_df["loss_type"] == lt)
                    & (cal_df["config"] == config_label)
                ]["implied_pctl"].dropna()

                color = dist_colors.get(config_label, {}).get(lt, "#90CAF9")
                ax.hist(sub, bins=10, range=(0, 1), color=color, alpha=0.75, edgecolor="none")
                if len(sub) > 0:
                    ax.axhline(len(sub) / 10, color="red", lw=0.7, ls="--")
                ax.set_xlim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                if row_i == 0:
                    short_cfg = config_label.replace(" (", "\n(")
                    ax.set_title(
                        f"{short_cfg}\n{LOSS_LABELS[lt]}",
                        fontsize=5.5, pad=2
                    )
                if col_i == 0:
                    ax.set_ylabel(lob.upper(), fontsize=7, labelpad=4)
                col_i += 1

    plt.tight_layout()
    out = FIGURES_DIR / "residual_dist_calibration_grid.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def _plot_pp_chart(cal_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """PP chart: 3 configs × 2 loss types, one panel per loss type."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    configs = list(CONFIGS.keys())
    line_styles = ["-", "--", ":"]
    colors = ["#2196F3", "#43A047", "#FB8C00"]

    for ax_i, lt in enumerate(LOSS_TYPES):
        ax = axes[ax_i]
        for ci, (config_label, ls, color) in enumerate(zip(configs, line_styles, colors)):
            p = cal_df[
                (cal_df["loss_type"] == lt) & (cal_df["config"] == config_label)
            ]["implied_pctl"].dropna().sort_values().values
            if len(p) == 0:
                continue
            emp = np.arange(1, len(p) + 1) / len(p)

            # Get KS stat for legend
            row = summary_df[(summary_df["config"] == config_label) & (summary_df["loss_type"] == lt)]
            ks_str = f"KS={row.iloc[0]['ks_stat']:.4f}" if not row.empty else ""
            mean_pctl_str = f"mean={row.iloc[0]['mean_pctl']:.3f}" if not row.empty else ""

            ax.plot(
                p, emp,
                label=f"{config_label} ({ks_str}, {mean_pctl_str})",
                color=color, lw=2.0, ls=ls,
            )

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration", alpha=0.5)
        ax.fill_between([0.1, 0.9], [0, 0], [1, 1], alpha=0.05, color="gray", label="Central 80%")
        ax.set_xlabel("Theoretical quantile (implied pctl)")
        ax.set_ylabel("Empirical CDF")
        ax.set_title(
            f"PP Chart — {LOSS_LABELS[lt]} — 200 Meyers triangles\n"
            "3 residual-dist configs (odp_corr only)"
        )
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    out = FIGURES_DIR / "residual_dist_pp_chart.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cal_df, summary_df = run_analysis()
