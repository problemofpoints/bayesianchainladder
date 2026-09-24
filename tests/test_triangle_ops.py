"""Tests for the private numpy chain-ladder primitives."""

import chainladder as cl
import numpy as np
import pandas as pd
import pytest

from bayesianchainladder._triangle_ops import (
    cumulative_array,
    cumulative_to_incremental,
    drop_mask,
    latest_diagonal,
    link_ratio_mask,
    link_ratio_sigma,
    project_cumulative,
    volume_weighted_factors,
)


@pytest.fixture
def genins():
    return cl.load_sample("genins")


def test_cumulative_array_shape_and_coords(genins):
    cum, origins, devs = cumulative_array(genins)
    assert cum.shape == (10, 10)
    assert origins == list(range(2001, 2011))
    assert devs == [12 * k for k in range(1, 11)]
    assert np.isnan(cum[1, 9]) and not np.isnan(cum[1, 8])


def test_cumulative_to_incremental_roundtrip(genins):
    cum, _, _ = cumulative_array(genins)
    incr = cumulative_to_incremental(cum)
    expected = np.asarray(genins.cum_to_incr().values, dtype=float)[0, 0]
    np.testing.assert_allclose(incr, expected, equal_nan=True)


def test_latest_diagonal(genins):
    cum, _, _ = cumulative_array(genins)
    latest, idx = latest_diagonal(cum)
    np.testing.assert_allclose(
        latest, np.asarray(genins.latest_diagonal.values)[0, 0, :, 0]
    )
    assert list(idx) == list(range(9, -1, -1))


def test_masks(genins):
    cum, origins, devs = cumulative_array(genins)
    dm = drop_mask(10, 10, [("2003", 72)], origins, devs)
    assert dm.shape == (10, 9)
    assert dm[2, 5] == 0 and dm.sum() == 89
    m = link_ratio_mask(cum, None, origins, devs)
    assert m.sum() == 45  # 9+8+...+1 available ratios
    m2 = link_ratio_mask(cum, [("2003", 72)], origins, devs)
    assert m2.sum() == 44


def test_volume_weighted_factors_match_chainladder(genins):
    cum, origins, devs = cumulative_array(genins)
    f = volume_weighted_factors(cum, link_ratio_mask(cum, None, origins, devs))
    expected = np.asarray(cl.Development().fit_transform(genins).ldf_.values).flatten()
    np.testing.assert_allclose(f, expected, rtol=1e-10)


def test_volume_weighted_factors_broadcast_over_sims(genins):
    cum, origins, devs = cumulative_array(genins)
    mask = link_ratio_mask(cum, None, origins, devs)
    stacked = np.repeat(cum[None, ...], 3, axis=0)
    f = volume_weighted_factors(stacked, mask)
    assert f.shape == (3, 9)
    np.testing.assert_allclose(f[0], volume_weighted_factors(cum, mask))


def test_project_cumulative_matches_chainladder(genins):
    cum, origins, devs = cumulative_array(genins)
    f = volume_weighted_factors(cum, link_ratio_mask(cum, None, origins, devs))
    full = project_cumulative(cum, f)
    expected = np.asarray(cl.Chainladder().fit(genins).full_triangle_.values)[
        0, 0, :, :10
    ]
    np.testing.assert_allclose(full, expected, rtol=1e-10)
    assert not np.isnan(full).any()


def test_project_cumulative_with_sim_dim(genins):
    cum, origins, devs = cumulative_array(genins)
    mask = link_ratio_mask(cum, None, origins, devs)
    stacked = np.repeat(cum[None, ...], 2, axis=0)
    f = volume_weighted_factors(stacked, mask)
    full = project_cumulative(stacked, f)
    assert full.shape == (2, 10, 10)
    np.testing.assert_allclose(full[1], project_cumulative(cum, f[1]))


def test_link_ratio_sigma_matches_mack_except_last(genins):
    cum, origins, devs = cumulative_array(genins)
    mask = link_ratio_mask(cum, None, origins, devs)
    f = volume_weighted_factors(cum, mask)
    sigma, resid = link_ratio_sigma(cum, mask, f)
    expected = np.asarray(
        cl.Development().fit_transform(genins).sigma_.values
    ).flatten()
    # chainladder extrapolates the last sigma log-linearly; England uses min of previous two
    np.testing.assert_allclose(sigma[:-1], expected[:-1], rtol=1e-8)
    assert sigma[-1] == pytest.approx(min(sigma[-2], sigma[-3]))
    assert resid.shape == (10, 9)
    assert np.isnan(resid[9, 0]) and np.isfinite(resid[0, 0])


def test_cumulative_array_rejects_subannual_origin_grain():
    # cl.load_sample("quarterly") has an annual origin grain (only its
    # development grain is quarterly), so build a triangle with quarterly
    # origins directly: _extract_period_value collapses each origin to its
    # calendar year, producing duplicate integer labels.
    rows = []
    origins = pd.period_range("2020Q1", periods=4, freq="Q")
    for oi, origin in enumerate(origins):
        for d in range(4 - oi):
            eval_period = origin + d
            rows.append(
                {
                    "origin": origin.to_timestamp(),
                    "dev": (d + 1) * 3,
                    "value": 100.0 + oi * 10 + d * 5,
                    "dev_date": eval_period.to_timestamp(how="end"),
                }
            )
    df = pd.DataFrame(rows)
    tri = cl.Triangle(
        df,
        origin="origin",
        development="dev_date",
        columns=["value"],
        cumulative=True,
    )
    assert tri.origin_grain == "Q"
    with pytest.raises(ValueError, match="annual origin grain"):
        cumulative_array(tri)
