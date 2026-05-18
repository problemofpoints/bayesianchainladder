"""02_aggregate_cv.py
====================
Aggregate CV(IBNR) by (lob, loss_type, method) from the stochastic reserving results.

Loads: references/schedule_p_backtest/cache/schedp_results.csv
Saves: references/schedule_p_backtest/cache/schedp_cv_summary.csv

Output tables
-------------
1. By line × method × loss_type — full breakdown (median CV, n_companies)
2. By method × loss_type — overall median across all lines (and n_companies)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

RESULTS_PATH = Path(__file__).resolve().parent / "cache" / "schedp_results.csv"
SUMMARY_PATH = Path(__file__).resolve().parent / "cache" / "schedp_cv_summary.csv"

PRIORITY_LINES = ["OLO", "OLC", "CAL", "WC", "PPAL", "CMP", "ALL"]

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


def main():
    print(f"Loading results from {RESULTS_PATH} ...")
    df = pd.read_csv(RESULTS_PATH)

    print(f"  Total rows: {len(df):,}")
    print(f"  Columns:    {df.columns.tolist()}")
    print()

    # Filter to Total rows (accident_year == 'Total' aggregates all origins)
    total_df = df[df["accident_year"] == "Total"].copy()
    print(f"  Total rows (accident_year=='Total'): {len(total_df):,}")

    # Drop rows with null cv_ibnr
    n_before = len(total_df)
    total_df = total_df.dropna(subset=["cv_ibnr"])
    print(f"  After dropping null cv_ibnr: {len(total_df):,} (dropped {n_before - len(total_df):,})")
    print()

    # -------------------------------------------------------------------
    # Table 1: By (lob, loss_type, method) — median CV, n_companies
    # -------------------------------------------------------------------
    agg1 = (
        total_df.groupby(["lob", "loss_type", "method"])
        .agg(
            median_cv=("cv_ibnr", "median"),
            mean_cv=("cv_ibnr", "mean"),
            n_companies=("group_id", "nunique"),
        )
        .reset_index()
    )
    agg1["median_cv"] = agg1["median_cv"].round(4)
    agg1["mean_cv"] = agg1["mean_cv"].round(4)

    # Pivot to wide format: rows = (lob, loss_type), columns = methods
    pivot1 = agg1.pivot_table(
        index=["lob", "loss_type"],
        columns="method",
        values="median_cv",
        aggfunc="first",
    ).reset_index()
    # Reorder method columns
    method_cols = [m for m in METHOD_ORDER if m in pivot1.columns]
    pivot1 = pivot1[["lob", "loss_type"] + method_cols]
    pivot1.columns.name = None

    print("=" * 90)
    print("TABLE 1: Median CV(IBNR) by (LOB, loss_type, method)")
    print("=" * 90)
    # Show priority lines first
    priority_pivot = pivot1[pivot1["lob"].isin(PRIORITY_LINES)].copy()
    other_pivot = pivot1[~pivot1["lob"].isin(PRIORITY_LINES)].copy()

    print("\n--- Priority lines ---")
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(priority_pivot.to_string(index=False))
    print("\n--- Other lines ---")
    print(other_pivot.to_string(index=False))
    print()

    # -------------------------------------------------------------------
    # Table 2: By (method, loss_type) — overall median across all lines
    # -------------------------------------------------------------------
    agg2 = (
        total_df.groupby(["loss_type", "method"])
        .agg(
            median_cv=("cv_ibnr", "median"),
            mean_cv=("cv_ibnr", "mean"),
            n_companies=("group_id", "nunique"),
        )
        .reset_index()
    )
    agg2["median_cv"] = agg2["median_cv"].round(4)
    agg2["mean_cv"] = agg2["mean_cv"].round(4)

    pivot2 = agg2.pivot_table(
        index="method",
        columns="loss_type",
        values=["median_cv", "mean_cv", "n_companies"],
        aggfunc="first",
    )
    pivot2.columns = [f"{val}_{lt}" for val, lt in pivot2.columns]
    pivot2 = pivot2.reset_index()
    # Reorder by method
    pivot2["method"] = pd.Categorical(pivot2["method"], categories=METHOD_ORDER, ordered=True)
    pivot2 = pivot2.sort_values("method").reset_index(drop=True)

    print("=" * 90)
    print("TABLE 2: Overall median CV(IBNR) by (method, loss_type) — all lines combined")
    print("=" * 90)
    print(pivot2.to_string(index=False))
    print()

    # -------------------------------------------------------------------
    # Save full detail to CSV
    # -------------------------------------------------------------------
    # Save the full detail table (not pivoted) so it's easy to filter/plot
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    agg1.to_csv(SUMMARY_PATH, index=False)
    print(f"Saved summary to {SUMMARY_PATH}")

    # Also save pivot tables to separate sheets-compatible CSVs
    pivot1_path = SUMMARY_PATH.parent / "schedp_cv_by_lob_method.csv"
    pivot2_path = SUMMARY_PATH.parent / "schedp_cv_by_method.csv"
    pivot1.to_csv(pivot1_path, index=False)
    pivot2.to_csv(pivot2_path, index=False)
    print(f"Saved pivot by (lob, method)     : {pivot1_path}")
    print(f"Saved pivot by method (overall)   : {pivot2_path}")


if __name__ == "__main__":
    main()
