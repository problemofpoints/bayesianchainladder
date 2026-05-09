"""Unit tests for _common.py."""
import numpy as np
import pandas as pd
import pytest

from _common import is_eligible_triangle


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
    """Triangle where all paid > 0 only appears at dev=12 fails the dev>12 rule."""
    vals = np.full((10, 10), np.nan)
    for i in range(9):
        vals[i, 0] = 1.0
        for j in range(1, 10 - i):
            vals[i, j] = 1.0  # cumulative stays at 1 — no incremental at dev>12
    tri = _make_triangle(vals)
    # No incremental paid > 0 at dev > 12.
    assert is_eligible_triangle(tri) is False
