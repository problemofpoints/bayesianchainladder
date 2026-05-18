"""01_build_long.py
==================
Build a long-format CSV from the Schedule P YE2024 chainladder Triangle JSON
for use with ``scripts/run_stochastic_reserving.py``.

Columns produced
----------------
lob          : line_of_business from the triangle index
group_id     : snl_id from the triangle index
origin       : accident year (int, 2015–2024)
dev          : development period in months (12, 24, …, 120)
paid         : cumulative paid loss
case_incurred: cumulative case-incurred loss
premium      : net earned premium for that origin year (constant per origin)

Filters applied
---------------
- At least 8 origin years observed (paid_loss non-null, positive)
- All cumulative paid_loss values > 0 (no zero or negative cumulatives)
- Positive net_earned_premium for all observed origins
- Both paid_loss AND case_incurred_loss must be available

Output
------
references/schedule_p_backtest/cache/schedp_long.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import chainladder as cl

JSON_PATH = Path("/Users/atroyer/Projects/reserve-risk-benchmarking/results/schedule_p_triangle.json")
OUTPUT_PATH = Path(__file__).resolve().parent / "cache" / "schedp_long.csv"


def build_schedule_p_long() -> pd.DataFrame:
    """Load the Schedule P triangle JSON and return a filtered long-format DataFrame."""
    print(f"Loading triangle from {JSON_PATH} ...")
    with open(JSON_PATH) as f:
        raw = f.read()
    tri = cl.read_json(raw)

    # Extract the three columns we need
    cols = ["paid_loss", "case_incurred_loss", "net_earned_premium"]
    t_sub = tri[cols]

    # Convert to long format with index levels retained
    df = t_sub.to_frame(keepdims=True).reset_index()
    # Columns: snl_id, line_of_business, origin, development, paid_loss, case_incurred_loss, net_earned_premium

    # Rename for clarity
    df = df.rename(columns={
        "snl_id": "group_id",
        "line_of_business": "lob",
        "development": "dev",
        "paid_loss": "paid",
        "case_incurred_loss": "case_incurred",
        "net_earned_premium": "premium",
    })

    # Extract integer year from origin (datetime64 / Period)
    df["origin"] = pd.to_datetime(df["origin"]).dt.year
    df["dev"] = df["dev"].astype(int)

    # For premium, it should be constant per (group_id, lob, origin);
    # take the first non-null value per group
    prem_df = (
        df.groupby(["group_id", "lob", "origin"])["premium"]
        .first()
        .reset_index()
    )

    # Work without premium column in main df; we'll merge it back
    df = df.drop(columns=["premium"])

    print(f"Raw rows (before filter): {len(df):,}")
    print(f"Raw (group_id, lob) combos: {df.groupby(['group_id','lob']).ngroups:,}")

    # -----------------------------------------------------------------------
    # Filtering: per (group_id, lob), apply validity checks
    # -----------------------------------------------------------------------
    valid_keys = []
    reasons = {}

    for (gid, lob), sub in df.groupby(["group_id", "lob"]):
        key = (gid, lob)

        # 1) Both paid and case_incurred must be available (not all-null)
        if sub["paid"].isna().all():
            reasons[key] = "paid all-null"
            continue
        if sub["case_incurred"].isna().all():
            reasons[key] = "case_incurred all-null"
            continue

        # 2) Consider only rows where paid is not null
        paid_rows = sub.dropna(subset=["paid"])

        # 3) At least 8 distinct origin years with paid data
        n_origins = paid_rows["origin"].nunique()
        if n_origins < 8:
            reasons[key] = f"only {n_origins} origins"
            continue

        # 4) All paid values must be positive (cumulative paid can't be 0 or negative)
        if (paid_rows["paid"] <= 0).any():
            reasons[key] = "paid has zero/negative values"
            continue

        # 5) Premium must be positive for all origins
        prem_sub = prem_df[(prem_df["group_id"] == gid) & (prem_df["lob"] == lob)]
        if prem_sub["premium"].isna().any() or (prem_sub["premium"] <= 0).any():
            reasons[key] = "premium missing or non-positive"
            continue

        # 6) Case incurred must be available for at least 8 origins too
        case_rows = sub.dropna(subset=["case_incurred"])
        if case_rows["origin"].nunique() < 8:
            reasons[key] = f"case_incurred only {case_rows['origin'].nunique()} origins"
            continue

        valid_keys.append(key)

    print(f"\nValid (group_id, lob) combos after filtering: {len(valid_keys):,}")
    reason_counts = pd.Series(list(reasons.values())).value_counts()
    print("Filtered-out reasons:")
    print(reason_counts.to_string())

    # Keep only valid keys
    valid_df = pd.DataFrame(valid_keys, columns=["group_id", "lob"])
    df_filtered = df.merge(valid_df, on=["group_id", "lob"], how="inner")

    # Merge premium back
    df_filtered = df_filtered.merge(prem_df, on=["group_id", "lob", "origin"], how="left")

    # Drop rows where paid is null (upper triangle only)
    df_filtered = df_filtered.dropna(subset=["paid"])

    # Reorder columns
    df_filtered = df_filtered[["lob", "group_id", "origin", "dev", "paid", "case_incurred", "premium"]]
    df_filtered = df_filtered.sort_values(["lob", "group_id", "origin", "dev"]).reset_index(drop=True)

    return df_filtered


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = build_schedule_p_long()

    print(f"\nLong-format summary:")
    print(f"  Total rows       : {len(df):,}")
    print(f"  Unique triangles : {df.groupby(['lob','group_id']).ngroups:,}")
    print(f"  LOBs             : {sorted(df['lob'].unique())}")
    print(f"  Origin range     : {df['origin'].min()} – {df['origin'].max()}")
    print(f"  Dev range        : {df['dev'].min()} – {df['dev'].max()}")
    print(f"  Missing paid     : {df['paid'].isna().sum()}")
    print(f"  Missing case_inc : {df['case_incurred'].isna().sum()}")
    print(f"  Missing premium  : {df['premium'].isna().sum()}")
    print()
    print("Breakdown by LOB:")
    lob_summary = (
        df.groupby("lob")["group_id"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"group_id": "n_companies"})
    )
    print(lob_summary.to_string(index=False))
    print()
    print(df.head(15).to_string(index=False))

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
