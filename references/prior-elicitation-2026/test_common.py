"""Unit tests for _common.py."""
import os

import numpy as np
import pandas as pd
import pytest

from _common import (
    CACHE_DIR,
    LINES,
    SAMPLE_PER_LINE,
    booked_reserve,
    is_eligible_triangle,
    iter_eligible_triangles,
    load_full_triangle,
    select_sample,
)


def _make_triangle(values: np.ndarray):
    """Build a 1x1x10x10 chainladder Triangle from a numpy array of shape (10, 10).

    NaN cells in `values` represent unobserved cells (lower triangle and
    missing data).
    """
    import chainladder as cl

    rows = []
    origins = list(range(2015, 2025))
    for i, oy in enumerate(origins):
        for j in range(10):
            v = values[i, j]
            if np.isnan(v):
                continue
            rows.append(
                {
                    "snl_id": "TEST",
                    "line_of_business": "TEST",
                    "acc_year": oy,
                    "dev_year": oy + j,
                    "paid_loss": v,
                    "net_earned_premium": 1000.0,
                }
            )
    df = pd.DataFrame(rows)
    return cl.Triangle(
        df,
        origin="acc_year",
        development="dev_year",
        columns=["paid_loss", "net_earned_premium"],
        index=["snl_id", "line_of_business"],
        cumulative=True,
        array_backend="numpy",
    )


def test_eligible_triangle_passes():
    """A triangle with 9 origin years, all positive paid, positive premium passes."""
    vals = np.full((10, 10), np.nan)
    # Build a typical upper triangle with positive cumulative paid loss.
    for i in range(9):  # 9 origin years
        for j in range(10 - i):
            vals[i, j] = 100.0 * (i + 1) * (j + 1)
    tri = _make_triangle(vals)
    assert is_eligible_triangle(tri) is True


def test_too_few_origin_years_fails():
    """Triangle with only 7 origin years of data fails the >= 8 rule."""
    vals = np.full((10, 10), np.nan)
    for i in range(7):
        for j in range(10 - i):
            vals[i, j] = 100.0 * (i + 1) * (j + 1)
    tri = _make_triangle(vals)
    assert is_eligible_triangle(tri) is False


def test_negative_paid_fails():
    """Triangle with a negative cumulative paid value fails."""
    vals = np.full((10, 10), np.nan)
    for i in range(9):
        for j in range(10 - i):
            vals[i, j] = 100.0 * (i + 1) * (j + 1)
    vals[3, 2] = -50.0  # one negative cell
    tri = _make_triangle(vals)
    assert is_eligible_triangle(tri) is False


def test_no_late_dev_activity_fails():
    """Triangle with flat cumulative paid (no incremental growth) fails Rule 4."""
    vals = np.full((10, 10), np.nan)
    for i in range(9):
        vals[i, 0] = 1.0
        for j in range(1, 10 - i):
            vals[i, j] = 1.0  # cumulative stays at 1 — no incremental at dev>12
    tri = _make_triangle(vals)
    # No incremental paid > 0 at dev > 12.
    assert is_eligible_triangle(tri) is False


# ---------------------------------------------------------------------------
# Triangle loading and sample selection tests
# ---------------------------------------------------------------------------

_TRIANGLE_JSON = os.path.expanduser(
    "~/Projects/reserve-risk-benchmarking/results/schedule_p_triangle.json"
)
_SKIPIF = pytest.mark.skipif(
    not os.path.exists(_TRIANGLE_JSON),
    reason="Schedule P triangle JSON not present on this machine",
)


@pytest.fixture(scope="module")
def full_triangle():
    """Load the real Schedule P triangle once for tests that need it."""
    return load_full_triangle()


@_SKIPIF
def test_load_full_triangle(full_triangle):
    """Loaded triangle has 'paid_loss' and 'net_earned_premium' vdims."""
    vdims = set(full_triangle.vdims)
    assert "paid_loss" in vdims
    assert "net_earned_premium" in vdims
    assert len(full_triangle.index) > 100


@_SKIPIF
def test_iter_eligible_triangles_olo(full_triangle):
    """OLO has at least 50 eligible triangles after filtering."""
    eligible = list(iter_eligible_triangles(full_triangle, "OLO"))
    assert len(eligible) >= 50
    snl_id, sub_tri = eligible[0]
    assert isinstance(snl_id, str)
    assert sub_tri.shape[0] == 1


@_SKIPIF
def test_select_sample_is_deterministic(full_triangle):
    """Calling select_sample twice with the same seed returns identical IDs."""
    s1 = select_sample(full_triangle, "OLO")
    s2 = select_sample(full_triangle, "OLO")
    assert s1 == s2
    # 24 sampled, plus checks that they're all real eligible IDs.
    assert len(s1) == SAMPLE_PER_LINE
