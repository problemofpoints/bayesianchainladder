"""21_process_variance_calibration.py
=====================================
Compare calibration across four process-variance models in the correlated ODP bootstrap:

  1. odp       — Var = phi * mu  (linear, standard ODP)         rho=0.3
  2. gamma     — Var = mu^2 / alpha  (quadratic)                rho=0.3
  3. lognormal — Var = mu^2*(exp(sigma^2)-1) (quadratic+heavier) rho=0.3
  4. negbin    — Var = mu + mu^2/k  (super-Poisson)             rho=0.3

All runs use --residual-dist normal --rho 0.3 to isolate the variance-model effect.

Inputs
------
  cache/meyers_pv_odp.csv + meyers_pv_odp_samples.parquet
  cache/meyers_pv_gamma.csv + meyers_pv_gamma_samples.parquet
  cache/meyers_pv_lognormal.csv + meyers_pv_lognormal_samples.parquet
  cache/meyers_pv_negbin.csv + meyers_pv_negbin_samples.parquet

Outputs
-------
  figures/process_variance_calibration_grid.png  — 4×2 histogram grid
  figures/process_variance_pp_chart.png          — PP chart, 4 configs × 2 loss types
  (printed calibration table + per-line breakdown + verdict)

Usage
-----
  cd /Users/atroyer/Projects/bayesianchainladder
  uv run python references/meyers-backtest/21_process_variance_calibration.py
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
# Configs
# ---------------------------------------------------------------------------

CONFIGS = {
    "odp": {
        "csv": CACHE_DIR / "meyers_pv_odp.csv",
        "parquet": CACHE_DIR / "meyers_pv_odp_samples.parquet",
        "rho": 0.3,
        "label": "ODP (linear, Var=phi*mu)",
    },
    "gamma": {
        "csv": CACHE_DIR / "meyers_pv_gamma.csv",
        "parquet": CACHE_DIR / "meyers_pv_gamma_samples.parquet",
        "rho": 0.3,
        "label": "Gamma (Var=mu^2/alpha)",
    },
    "lognormal": {
        "csv": CACHE_DIR / "meyers_pv_lognormal.csv",
        "parquet": CACHE_DIR / "meyers_pv_lognormal_samples.parquet",
        "rho": 0.3,
        "label": "Lognormal (Var~mu^2)",
    },
    "negbin": {
        "csv": CACHE_DIR / "meyers_pv_negbin.csv",
        "parquet": CACHE_DIR / "meyers_pv_negbin_samples.parquet",
        "rho": 0.3,
        "label": "NegBin (Var=mu+mu^2/k)",
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


def calibration_metrics(p: np.ndarray, cv_vals: np.ndarray | None = None,
                        pct_err_vals: np.ndarray | None = None) -> dict:
    """Compute calibration metrics for a vector of implied percentiles."""
    p = p[np.isfinite(p)]
    if len(p) == 0:
        return {
            "n": 0, "mean_pctl": np.nan, "pct_in_central50": np.nan,
            "pct_in_central80": np.nan, "ks_stat": np.nan, "ks_pval": np.nan,
            "median_cv_ibnr": np.nan, "median_abs_pct_err": np.nan,
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

def run_analysis():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading actual ultimates from reservetestr...")
    actuals = load_actual_ultimates()
    print(f"  {len(actuals)} rows")

    all_cal_rows = []

    for pv_key, cfg in CONFIGS.items():
        if not cfg["csv"].exists() or not cfg["parquet"].exists():
            print(f"  [SKIP] {pv_key}: missing files {cfg['csv']} or {cfg['parquet']}")
            continue

        print(f"\nLoading {pv_key} ({cfg['label']})...")
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
                "pv_key": pv_key,
                "label": cfg["label"],
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

    # ---------------------------------------------------------------------------
    # Summary calibration table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 140)
    print("CALIBRATION TABLE: 4 process-variance models × 2 loss types")
    print("All runs: odp_corr, rho=0.3, residual_dist=normal")
    print("Ideal: mean_pctl=0.5, %in50=50, %in80=80, ks_stat~0, small cv, small |%err|")
    print("=" * 140)
    print(
        f"{'Process Variance':<28} {'Loss type':<16} {'N':>5} {'MeanPctl':>9} "
        f"{'%in50':>7} {'%in80':>7} {'KS_stat':>8} {'KS_pval':>8} "
        f"{'MedCV':>7} {'Med|%err|':>10}"
    )
    print("-" * 140)

    summary_rows = []
    for pv_key, cfg in CONFIGS.items():
        for lt in LOSS_TYPES:
            sub = cal_df[(cal_df["pv_key"] == pv_key) & (cal_df["loss_type"] == lt)]
            if sub.empty:
                continue
            p = sub["implied_pctl"].values
            cv_vals = sub["cv_ibnr"].values
            pct_err_vals = sub["pct_err"].values
            m = calibration_metrics(p, cv_vals, pct_err_vals)
            print(
                f"{cfg['label']:<28} {LOSS_LABELS.get(lt, lt):<16} "
                f"{m['n']:>5} {m['mean_pctl']:>9.3f} "
                f"{m['pct_in_central50']:>7.1f} {m['pct_in_central80']:>7.1f} "
                f"{m['ks_stat']:>8.4f} {m['ks_pval']:>8.4f} "
                f"{m['median_cv_ibnr']:>7.3f} {m['median_abs_pct_err']:>10.4f}"
            )
            summary_rows.append({
                "pv_key": pv_key,
                "label": cfg["label"],
                "loss_type": lt,
                **m,
            })

    summary_df = pd.DataFrame(summary_rows)

    # ---------------------------------------------------------------------------
    # Per-line calibration table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 140)
    print("PER-LINE CALIBRATION: KS statistic by (process_variance, lob, loss_type)")
    print("-" * 140)
    print(
        f"{'Process Variance':<28} {'LOB':<10} "
        f"{'Paid KS':>10} {'Case KS':>10} "
        f"{'Paid %80':>10} {'Case %80':>10} "
        f"{'Paid MCV':>10} {'Case MCV':>10}"
    )
    print("-" * 140)

    for pv_key, cfg in CONFIGS.items():
        for lob in LOBS:
            row_vals: dict = {}
            for lt in LOSS_TYPES:
                sub = cal_df[
                    (cal_df["pv_key"] == pv_key)
                    & (cal_df["lob"] == lob)
                    & (cal_df["loss_type"] == lt)
                ]
                p = sub["implied_pctl"].values
                cv_vals = sub["cv_ibnr"].values
                m = calibration_metrics(p, cv_vals)
                row_vals[f"ks_{lt}"] = m["ks_stat"]
                row_vals[f"pct80_{lt}"] = m["pct_in_central80"]
                row_vals[f"mcv_{lt}"] = m["median_cv_ibnr"]
            print(
                f"{cfg['label']:<28} {lob:<10} "
                f"{row_vals.get('ks_paid', np.nan):>10.4f} "
                f"{row_vals.get('ks_case_incurred', np.nan):>10.4f} "
                f"{row_vals.get('pct80_paid', np.nan):>10.1f} "
                f"{row_vals.get('pct80_case_incurred', np.nan):>10.1f} "
                f"{row_vals.get('mcv_paid', np.nan):>10.3f} "
                f"{row_vals.get('mcv_case_incurred', np.nan):>10.3f}"
            )

    # ---------------------------------------------------------------------------
    # Verdict
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 140)
    print("VERDICT")
    print("=" * 140)

    odp_paid = summary_df[(summary_df["pv_key"] == "odp") & (summary_df["loss_type"] == "paid")]
    odp_case = summary_df[(summary_df["pv_key"] == "odp") & (summary_df["loss_type"] == "case_incurred")]
    baseline_ks_paid = float(odp_paid.iloc[0]["ks_stat"]) if not odp_paid.empty else np.nan
    baseline_ks_case = float(odp_case.iloc[0]["ks_stat"]) if not odp_case.empty else np.nan
    baseline_cv_paid = float(odp_paid.iloc[0]["median_cv_ibnr"]) if not odp_paid.empty else np.nan
    baseline_cv_case = float(odp_case.iloc[0]["median_cv_ibnr"]) if not odp_case.empty else np.nan

    improvement_threshold = 0.05  # ≥5pp drop in KS to be "meaningful"

    for lt, baseline_ks, baseline_cv in [
        ("paid", baseline_ks_paid, baseline_cv_paid),
        ("case_incurred", baseline_ks_case, baseline_cv_case),
    ]:
        lt_label = LOSS_LABELS[lt]
        print(f"\n  {lt_label}:")
        for pv_key, cfg in CONFIGS.items():
            row = summary_df[(summary_df["pv_key"] == pv_key) & (summary_df["loss_type"] == lt)]
            if row.empty:
                continue
            ks = float(row.iloc[0]["ks_stat"])
            cv = float(row.iloc[0]["median_cv_ibnr"])
            mp = float(row.iloc[0]["mean_pctl"])
            p80 = float(row.iloc[0]["pct_in_central80"])
            delta = baseline_ks - ks
            mark = ""
            if pv_key == "odp":
                mark = " <-- BASELINE"
            elif delta >= improvement_threshold:
                mark = " <-- IMPROVEMENT"
            print(
                f"    {cfg['label']:<30}: KS={ks:.4f}  delta={delta:+.4f}  "
                f"%in80={p80:.1f}%  MedCV={cv:.3f}  MeanPctl={mp:.3f}{mark}"
            )

    # Overall recommendation
    print("\n  RECOMMENDATION:")
    for lt, baseline_ks, baseline_pv in [
        ("paid", baseline_ks_paid, "odp"),
        ("case_incurred", baseline_ks_case, "odp"),
    ]:
        lt_label = LOSS_LABELS[lt]
        candidates = [
            (pv_key, float(summary_df[(summary_df["pv_key"] == pv_key) & (summary_df["loss_type"] == lt)].iloc[0]["ks_stat"]))
            for pv_key in CONFIGS
            if not summary_df[(summary_df["pv_key"] == pv_key) & (summary_df["loss_type"] == lt)].empty
        ]
        best_pv, best_ks = min(candidates, key=lambda x: x[1])
        improvement = baseline_ks - best_ks
        if improvement >= improvement_threshold and best_pv != "odp":
            print(
                f"  [{lt_label}] Meaningful improvement: {CONFIGS[best_pv]['label']} reduces "
                f"KS by {improvement:.4f} ({baseline_ks:.4f} -> {best_ks:.4f}). "
                f"Recommend making this the default."
            )
        else:
            print(
                f"  [{lt_label}] No meaningful improvement from alternative process-variance "
                f"models (best delta = {improvement:+.4f}, best = {CONFIGS[best_pv]['label']}). "
                f"Variance model does not fix the calibration issue."
            )

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
    """4 process-variance models × 2 loss types histogram grid per LOB."""
    pv_keys = list(CONFIGS.keys())
    n_rows = len(pv_keys)       # 4
    n_cols = 2 * len(LOBS)      # 8 (2 loss types × 4 LOBs)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 1.6, n_rows * 2.0), sharey=False
    )

    pv_colors = {
        "odp":      {"paid": "#2196F3", "case_incurred": "#1565C0"},
        "gamma":    {"paid": "#43A047", "case_incurred": "#1B5E20"},
        "lognormal":{"paid": "#FB8C00", "case_incurred": "#E65100"},
        "negbin":   {"paid": "#AB47BC", "case_incurred": "#6A1B9A"},
    }

    fig.suptitle(
        "Implied Percentile Histograms: 4 Process-Variance Models × 4 LOBs × 2 Loss Types\n"
        "odp_corr rho=0.3, residual_dist=normal  |  Uniform = perfect calibration",
        fontsize=8, y=1.01,
    )

    for row_i, pv_key in enumerate(pv_keys):
        col_i = 0
        for lob in LOBS:
            for lt in LOSS_TYPES:
                ax = axes[row_i, col_i]
                sub = cal_df[
                    (cal_df["pv_key"] == pv_key)
                    & (cal_df["lob"] == lob)
                    & (cal_df["loss_type"] == lt)
                ]["implied_pctl"].dropna()

                color = pv_colors.get(pv_key, {}).get(lt, "#90CAF9")
                ax.hist(sub, bins=10, range=(0, 1), color=color, alpha=0.75, edgecolor="none")
                if len(sub) > 0:
                    ax.axhline(len(sub) / 10, color="red", lw=0.7, ls="--")

                ks_val = np.nan
                if len(sub) > 0:
                    ks_val, _ = stats.kstest(sub.values, "uniform")

                ax.set_xlim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                if row_i == 0:
                    ax.set_title(
                        f"{lob.upper()}\n{LOSS_LABELS[lt]}",
                        fontsize=6, pad=2
                    )
                if col_i == 0:
                    short_label = CONFIGS[pv_key]["label"].split("(")[0].strip()
                    ax.set_ylabel(short_label, fontsize=6, labelpad=4)
                if np.isfinite(ks_val):
                    ax.text(
                        0.98, 0.97, f"KS={ks_val:.3f}",
                        transform=ax.transAxes, fontsize=5,
                        ha="right", va="top", color="black"
                    )
                col_i += 1

    plt.tight_layout()
    out = FIGURES_DIR / "process_variance_calibration_grid.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def _plot_pp_chart(cal_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    """PP chart: 4 process-variance configs × 2 loss types."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    pv_keys = list(CONFIGS.keys())
    line_styles = ["-", "--", "-.", ":"]
    colors = ["#2196F3", "#43A047", "#FB8C00", "#AB47BC"]

    for ax_i, lt in enumerate(LOSS_TYPES):
        ax = axes[ax_i]
        for ci, (pv_key, ls, color) in enumerate(zip(pv_keys, line_styles, colors)):
            p = cal_df[
                (cal_df["loss_type"] == lt) & (cal_df["pv_key"] == pv_key)
            ]["implied_pctl"].dropna().sort_values().values
            if len(p) == 0:
                continue
            emp = np.arange(1, len(p) + 1) / len(p)

            row = summary_df[(summary_df["pv_key"] == pv_key) & (summary_df["loss_type"] == lt)]
            ks_str = f"KS={row.iloc[0]['ks_stat']:.4f}" if not row.empty else ""
            mean_pctl_str = f"mean={row.iloc[0]['mean_pctl']:.3f}" if not row.empty else ""
            cv_str = f"MedCV={row.iloc[0]['median_cv_ibnr']:.3f}" if not row.empty else ""
            label = f"{CONFIGS[pv_key]['label']} ({ks_str}, {mean_pctl_str}, {cv_str})"

            ax.plot(p, emp, label=label, color=color, lw=2.0, ls=ls)

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration", alpha=0.5)
        ax.fill_between([0.1, 0.9], [0, 0], [1, 1], alpha=0.05, color="gray", label="Central 80%")
        ax.set_xlabel("Theoretical quantile (implied pctl)")
        ax.set_ylabel("Empirical CDF")
        ax.set_title(
            f"PP Chart — {LOSS_LABELS[lt]} — 200 Meyers triangles\n"
            "4 process-variance models, odp_corr rho=0.3"
        )
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    out = FIGURES_DIR / "process_variance_pp_chart.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cal_df, summary_df = run_analysis()
