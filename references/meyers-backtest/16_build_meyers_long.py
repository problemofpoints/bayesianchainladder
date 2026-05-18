"""16_build_meyers_long.py
========================
Build a long-format CSV from all 200 Meyers triangles for use with
``scripts/run_stochastic_reserving.py``.

Columns produced
----------------
lob         : Meyers line of business (comauto, ppauto, wkcomp, othliab)
group_id    : Meyers group_id integer
origin      : accident year (1988–1997)
dev         : development period in months (12, 24, …, 120)
paid        : cumulative paid loss (upper triangle only)
case_incurred: cumulative case-incurred loss (upper triangle only)
premium     : net earned premium for that origin year

Only TRAINING triangles (upper triangle, evaluation at 1997) are written;
test-triangle diagonals are excluded. The standalone reserving script does not
need actual ultimates — those are joined back in during the calibration step.

Output
------
references/meyers-backtest/cache/meyers_long.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import reservetestr as rt

# Ensure _common is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUTPUT_PATH = Path(__file__).resolve().parent / "cache" / "meyers_long.csv"


def triangle_to_long(tri, value_col: str) -> pd.DataFrame:
    """Convert a chainladder Triangle to long-format with origin (int year) and dev (months).

    Parameters
    ----------
    tri : chainladder.Triangle
        Training triangle (upper triangle only).
    value_col : str
        Name to give the values column.

    Returns
    -------
    pd.DataFrame with columns [origin, dev, value_col]
    """
    df = tri.to_frame(keepdims=True).reset_index(drop=True)
    # columns are: origin, development, values
    # origin is a datetime-like period; development is already in months
    df = df.rename(columns={"values": value_col})
    # Extract integer year from origin (datetime64)
    df["origin"] = pd.to_datetime(df["origin"]).dt.year
    df["dev"] = df["development"].astype(int)
    df = df.drop(columns=["development"])
    # Drop NaN values (lower triangle excluded by reservetestr already, but be safe)
    df = df.dropna(subset=[value_col])
    return df[["origin", "dev", value_col]]


def build_meyers_long() -> pd.DataFrame:
    """Load all 200 Meyers training triangles and return a single long-format DataFrame."""
    # Also load the raw CLRD data for premium (net_ep is constant per accident_year)
    clrd_df = rt.load_meyers_subset(rt.load_clrd_dataframe())
    # Premium: net_ep is constant within (line, group_id, accident_year) across development
    premium_df = (
        clrd_df.groupby(["line", "group_id", "accident_year"])["net_ep"]
        .first()
        .reset_index()
        .rename(columns={"line": "lob", "accident_year": "origin", "net_ep": "premium"})
    )
    premium_df["group_id"] = premium_df["group_id"].astype(int)
    premium_df["origin"] = premium_df["origin"].astype(int)

    records = rt.build_triangle_records()
    print(f"Building long CSV from {len(records)} Meyers records ...")

    all_rows = []
    for r in records:
        lob = r.line
        gid = int(r.group_id)

        # Convert paid training triangle
        paid_long = triangle_to_long(r.train_triangles["paid"], "paid")
        # Convert case-incurred training triangle
        case_long = triangle_to_long(r.train_triangles["case"], "case_incurred")

        # Merge paid and case on (origin, dev)
        merged = pd.merge(paid_long, case_long, on=["origin", "dev"], how="outer")
        merged["lob"] = lob
        merged["group_id"] = gid

        # Attach premium
        prem_sub = premium_df[(premium_df["lob"] == lob) & (premium_df["group_id"] == gid)][
            ["origin", "premium"]
        ]
        merged = merged.merge(prem_sub, on="origin", how="left")

        all_rows.append(merged)

    df = pd.concat(all_rows, ignore_index=True)

    # Reorder columns
    df = df[["lob", "group_id", "origin", "dev", "paid", "case_incurred", "premium"]]
    df = df.sort_values(["lob", "group_id", "origin", "dev"]).reset_index(drop=True)

    return df


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = build_meyers_long()

    print(f"\nLong-format summary:")
    print(f"  Total rows       : {len(df):,}")
    print(f"  Unique triangles : {df.groupby(['lob','group_id']).ngroups}")
    print(f"  LOBs             : {sorted(df['lob'].unique())}")
    print(f"  Origin range     : {df['origin'].min()} – {df['origin'].max()}")
    print(f"  Dev range        : {df['dev'].min()} – {df['dev'].max()}")
    print(f"  Missing paid     : {df['paid'].isna().sum()}")
    print(f"  Missing case_inc : {df['case_incurred'].isna().sum()}")
    print(f"  Missing premium  : {df['premium'].isna().sum()}")
    print()
    print(df.head(15).to_string(index=False))

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
