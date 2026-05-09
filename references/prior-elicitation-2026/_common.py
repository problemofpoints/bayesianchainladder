"""Shared utilities for the prior-elicitation-2026 analysis.

This module is imported by every numbered analysis script (01–05) and the
test_common.py suite. Keep it small and side-effect-free at import time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import chainladder as cl
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LINES = ["OLO", "OLC", "CAL", "WC", "PPAL", "CMP"]

TRIANGLE_JSON = (
    Path.home()
    / "Projects/reserve-risk-benchmarking/results/schedule_p_triangle.json"
)

ANALYSIS_DIR = Path(__file__).resolve().parent
CACHE_DIR = ANALYSIS_DIR / "cache"
FIGURES_DIR = ANALYSIS_DIR / "figures"

MIN_ORIGIN_YEARS = 8
SAMPLE_SEED = 20260508
SAMPLE_PER_LINE = 24  # 8 per tercile

# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def is_eligible_triangle(tri: cl.Triangle) -> bool:
    """Return True if a single (snl_id, lob) Triangle is eligible for the study.

    Rules (must all hold):
      1. >= MIN_ORIGIN_YEARS origin years have at least one observed cell.
      2. All observed cumulative `paid_loss` values are strictly positive.
      3. Net earned premium is positive for every origin with paid data.
      4. At least one origin shows positive cumulative growth beyond dev=12
         (i.e., paid_loss increases somewhere past the first development period).

    Parameters
    ----------
    tri : cl.Triangle
        A Triangle whose vdims include `paid_loss` and `net_earned_premium`,
        and whose index identifies a single (snl_id, line_of_business) pair.
    """
    vdims = list(tri.vdims)
    if "paid_loss" not in vdims:
        return False
    paid = tri["paid_loss"].values[0, 0]  # shape: (n_origin, n_dev)
    if "net_earned_premium" in vdims:
        prem = tri["net_earned_premium"].values[0, 0]
    else:
        prem = None

    # Rule 1: >= MIN_ORIGIN_YEARS origin years with any observation.
    origin_has_data = np.any(~np.isnan(paid), axis=1)
    if origin_has_data.sum() < MIN_ORIGIN_YEARS:
        return False

    # Rule 2: all observed cumulative paid > 0.
    observed = paid[~np.isnan(paid)]
    if observed.size == 0 or np.any(observed <= 0):
        return False

    # Rule 3: positive premium for every origin with paid data.
    if prem is not None:
        prem_for_active = prem[origin_has_data]
        prem_observed = prem_for_active[~np.isnan(prem_for_active)]
        if prem_observed.size == 0 or np.any(prem_observed <= 0):
            return False

    # Rule 4: positive incremental somewhere at dev > 12.
    # np.diff propagates NaN, so a row with an interior NaN gap
    # (observed, NaN, observed) would yield NaN diffs even if real growth
    # occurred. Acceptable for Schedule P upper triangles which have no
    # interior gaps; revisit if data shape changes.
    if paid.shape[1] < 2:
        return False
    incremental = np.diff(paid, axis=1)  # shape: (n_origin, n_dev - 1)
    if not np.any(incremental[~np.isnan(incremental)] > 0):
        return False

    return True


# ---------------------------------------------------------------------------
# Triangle loading
# ---------------------------------------------------------------------------


def load_full_triangle() -> cl.Triangle:
    """Read the Schedule P JSON file and return a multi-index chainladder Triangle."""
    with open(TRIANGLE_JSON, "r", encoding="utf-8") as f:
        json_str = f.read()
    return cl.read_json(json_str)


def iter_eligible_triangles(
    full_tri: cl.Triangle, line: str
) -> Iterator[tuple[str, cl.Triangle]]:
    """Yield (snl_id, single-row Triangle) pairs for triangles in `line` that pass eligibility."""
    sub = full_tri[full_tri["line_of_business"] == line]
    n = len(sub.index)
    for i in range(n):
        row = sub.iloc[i]
        if not is_eligible_triangle(row):
            continue
        snl_id = row.index["snl_id"].iloc[0]
        yield snl_id, row


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------


def booked_reserve(tri: cl.Triangle) -> float:
    """Return the latest-diagonal booked reserve for a single-row Triangle.

    booked_reserve = booked_ultimate_loss − cumulative paid_loss, summed across
    accident years.
    """
    if "booked_reserve" in list(tri.vdims):
        latest = tri["booked_reserve"].latest_diagonal.values
        return float(np.nansum(latest))
    ult = tri["booked_ultimate_loss"].latest_diagonal.values
    paid = tri["paid_loss"].latest_diagonal.values
    return float(np.nansum(ult - paid))


def select_sample(full_tri: cl.Triangle, line: str) -> list[str]:
    """Return SAMPLE_PER_LINE snl_ids for `line` stratified into terciles by booked reserve.

    8 ids per tercile, sampled uniformly within tercile, deterministic on SAMPLE_SEED + line.
    """
    eligible = list(iter_eligible_triangles(full_tri, line))
    if len(eligible) < SAMPLE_PER_LINE:
        # Edge case: not enough eligible triangles. Return all of them.
        return [snl for snl, _ in eligible]

    df = pd.DataFrame(
        {
            "snl_id": [snl for snl, _ in eligible],
            "br": [booked_reserve(t) for _, t in eligible],
        }
    )
    df["tercile"] = pd.qcut(df["br"], 3, labels=["small", "mid", "large"])

    # Seed combines the global seed with a line hash so each line gets its own draw.
    seed = SAMPLE_SEED + abs(hash(line)) % 10_000
    rng = np.random.default_rng(seed)
    per_tercile = SAMPLE_PER_LINE // 3
    picked: list[str] = []
    for label in ["small", "mid", "large"]:
        pool = df.loc[df["tercile"] == label, "snl_id"].tolist()
        n = min(per_tercile, len(pool))
        picked.extend(rng.choice(pool, size=n, replace=False).tolist())
    # Tie-breaker if 24 not divisible by 3 in some future change.
    while len(picked) < SAMPLE_PER_LINE and len(picked) < len(df):
        leftover = [s for s in df["snl_id"] if s not in picked]
        picked.append(rng.choice(leftover))
    return sorted(picked)
