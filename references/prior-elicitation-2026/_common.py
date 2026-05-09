"""Shared utilities for the prior-elicitation-2026 analysis.

This module is imported by every numbered analysis script (01–05) and the
test_common.py suite. Keep it small and side-effect-free at import time.
"""
from __future__ import annotations

from pathlib import Path

import chainladder as cl
import numpy as np

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
      4. There is at least one observed cell at dev > 12 months
         (excludes triangles with only a single dev period).

    Parameters
    ----------
    tri : cl.Triangle
        A Triangle whose vdims include `paid_loss` and `net_earned_premium`,
        and whose index identifies a single (snl_id, line_of_business) pair.
    """
    if "paid_loss" not in list(tri.vdims):
        return False
    paid = tri["paid_loss"].values[0, 0]  # shape: (n_origin, n_dev)
    if "net_earned_premium" in list(tri.vdims):
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

    # Rule 4: at least one observed cell at dev > 12 (i.e. j > 0) with
    # positive incremental paid loss (cumulative increases beyond first period).
    if paid.shape[1] < 2:
        return False
    incremental = np.diff(paid, axis=1)  # shape: (n_origin, n_dev - 1)
    if not np.any(incremental[~np.isnan(incremental)] > 0):
        return False

    return True
