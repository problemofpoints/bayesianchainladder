"""24_build_clrd2025_long.py
==========================
Build Meyers-style back-test inputs from chainladder's ``clrd2025`` sample.

Window
------
Origins 1998-2007, development 12..120 months.  Training data is the upper
triangle valued at 2007-12-31 (origin_year + dev/12 - 1 <= 2007).  Actual
ultimates are the dev-120 values, fully revealed by the 2016 valuation.  This
mirrors Meyers (2015), who trained on 1988-1997 valued at 1997 and tested on
the square revealed by 2006.

Eligibility (per GRNAME x LOB)
------------------------------
1. Complete 10x10 square for CumPaidLoss and IncurredLosses.
2. EarnedPremNet > 0 in every origin year.
3. Every paid and case-incurred cell > 0 (case incurred = IncurredLosses -
   BulkLoss, with missing BulkLoss treated as 0).
4. max/min EarnedPremNet across the 10 origins <= --max-premium-ratio
   (default 5; pass 0 to disable).  Excludes books with large structural
   changes, in the spirit of Meyers' stable-book screen.

Outputs (references/meyers-backtest/cache/)
-------------------------------------------
clrd2025_long.csv     lob, group_id, origin, dev, paid, case_incurred, premium
clrd2025_actuals.csv  lob, group_id, loss_type, actual_ultimate_total
clrd2025_groups.csv   group_id, grname, lob, premium_ratio

Usage
-----
  uv run python references/meyers-backtest/24_build_clrd2025_long.py [--max-premium-ratio 5]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import chainladder as cl
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CACHE_DIR  # noqa: E402

FIRST_ORIGIN = 1998
LAST_ORIGIN = 2007
N_DEV = 10
LONG_PATH = CACHE_DIR / "clrd2025_long.csv"
ACTUALS_PATH = CACHE_DIR / "clrd2025_actuals.csv"
GROUPS_PATH = CACHE_DIR / "clrd2025_groups.csv"


def load_window() -> cl.Triangle:
    """clrd2025 restricted to origins <= 2007 and development <= 120 months."""
    tri = cl.load_sample("clrd2025")
    tri = tri[tri.origin <= str(LAST_ORIGIN)][tri.development <= 12 * N_DEV]
    assert tri.shape[2:] == (N_DEV, N_DEV), tri.shape
    return tri


def extract_arrays(tri: cl.Triangle) -> dict[str, np.ndarray]:
    """Dense arrays per group: paid/case (n, 10, 10), premium (n, 10)."""
    paid = np.asarray(tri["CumPaidLoss"].values, dtype=float)[:, 0]
    incurred = np.asarray(tri["IncurredLosses"].values, dtype=float)[:, 0]
    bulk = np.nan_to_num(np.asarray(tri["BulkLoss"].values, dtype=float)[:, 0])
    premium = np.asarray(tri["EarnedPremNet"].values, dtype=float)[:, 0, :, 0]
    return {"paid": paid, "incurred": incurred, "case": incurred - bulk, "premium": premium}


def eligibility(arrays: dict[str, np.ndarray], max_premium_ratio: float) -> pd.DataFrame:
    """Per-group boolean filters, cumulative in the listed order."""
    paid, incurred, case, premium = (
        arrays["paid"], arrays["incurred"], arrays["case"], arrays["premium"]
    )
    complete = np.isfinite(paid).all(axis=(1, 2)) & np.isfinite(incurred).all(axis=(1, 2))
    prem_ok = complete & (np.nan_to_num(premium) > 0).all(axis=1)
    positive = (
        prem_ok
        & (np.nan_to_num(paid) > 0).all(axis=(1, 2))
        & (np.nan_to_num(case) > 0).all(axis=(1, 2))
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.nanmax(premium, axis=1) / np.nanmin(premium, axis=1)
    stable = positive & ((ratio <= max_premium_ratio) if max_premium_ratio > 0 else True)
    return pd.DataFrame(
        {"complete": complete, "premium_positive": prem_ok, "losses_positive": positive,
         "stable": stable, "premium_ratio": ratio}
    )


def build(max_premium_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tri = load_window()
    arrays = extract_arrays(tri)
    flags = eligibility(arrays, max_premium_ratio)
    index = tri.index.reset_index(drop=True)

    print("Eligibility funnel (GRNAME x LOB):")
    for col in ["complete", "premium_positive", "losses_positive", "stable"]:
        print(f"  {col:<18} {int(flags[col].sum()):>4} of {len(flags)}")

    keep = np.where(flags["stable"].to_numpy())[0]
    groups = index.iloc[keep].copy()
    groups["_row"] = keep
    groups["premium_ratio"] = flags["premium_ratio"].to_numpy()[keep]
    groups = groups.rename(columns={"GRNAME": "grname", "LOB": "lob"})
    groups = groups.sort_values(["lob", "grname"]).reset_index(drop=True)
    groups.insert(0, "group_id", np.arange(1, len(groups) + 1))

    origins = np.arange(FIRST_ORIGIN, LAST_ORIGIN + 1)
    devs = 12 * np.arange(1, N_DEV + 1)
    k, j = np.meshgrid(np.arange(N_DEV), np.arange(N_DEV), indexing="ij")
    train_mask = (k + j) <= (N_DEV - 1)  # origin_year + dev_years - 1 <= 2007

    long_rows, actual_rows = [], []
    for _, g in groups.iterrows():
        r = int(g["_row"])
        paid, case, prem = arrays["paid"][r], arrays["case"][r], arrays["premium"][r]
        long_rows.append(pd.DataFrame({
            "lob": g["lob"],
            "group_id": int(g["group_id"]),
            "origin": origins[k[train_mask]],
            "dev": devs[j[train_mask]],
            "paid": paid[train_mask],
            "case_incurred": case[train_mask],
            "premium": prem[k[train_mask]],
        }))
        actual_rows.append({"lob": g["lob"], "group_id": int(g["group_id"]),
                            "loss_type": "paid", "actual_ultimate_total": float(paid[:, -1].sum())})
        actual_rows.append({"lob": g["lob"], "group_id": int(g["group_id"]),
                            "loss_type": "case_incurred", "actual_ultimate_total": float(case[:, -1].sum())})

    long_df = pd.concat(long_rows, ignore_index=True).sort_values(
        ["lob", "group_id", "origin", "dev"]).reset_index(drop=True)
    actuals_df = pd.DataFrame(actual_rows)
    groups_df = groups.drop(columns=["_row"])
    return long_df, actuals_df, groups_df


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-premium-ratio", type=float, default=5.0,
                   help="Exclude groups whose max/min net earned premium exceeds this (0 disables).")
    args = p.parse_args(argv)

    warnings.filterwarnings("ignore")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    long_df, actuals_df, groups_df = build(args.max_premium_ratio)

    n_groups = groups_df["group_id"].nunique()
    rows_per_group = long_df.groupby("group_id").size()
    assert (rows_per_group == N_DEV * (N_DEV + 1) // 2).all(), rows_per_group.describe()
    assert long_df[["paid", "case_incurred", "premium"]].gt(0).all().all()
    assert len(actuals_df) == 2 * n_groups

    print(f"\nEligible groups: {n_groups}")
    print(groups_df["lob"].value_counts().to_string())
    print(f"\nLong rows: {len(long_df):,}  (55 per group)")
    print(f"Origins {long_df['origin'].min()}-{long_df['origin'].max()}, dev {long_df['dev'].min()}-{long_df['dev'].max()}")
    print(f"Premium ratio: median {groups_df['premium_ratio'].median():.2f}, max {groups_df['premium_ratio'].max():.2f}")

    long_df.to_csv(LONG_PATH, index=False)
    actuals_df.to_csv(ACTUALS_PATH, index=False)
    groups_df.to_csv(GROUPS_PATH, index=False)
    print(f"\nWrote {LONG_PATH}\n      {ACTUALS_PATH}\n      {GROUPS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
