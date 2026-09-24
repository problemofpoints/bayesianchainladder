"""Tests for the one-year Claims Development Result (England, Verrall & Wuthrich 2019)."""

import chainladder as cl
import numpy as np
import pytest

from bayesianchainladder._triangle_ops import (
    cumulative_array,
    link_ratio_mask,
    project_cumulative,
    volume_weighted_factors,
)
from bayesianchainladder.bootstrap import BootstrapODPChainLadder, MackChainLadder
from bayesianchainladder.cdr import CDRResult, claims_development_result


@pytest.fixture(scope="module")
def odp():
    return BootstrapODPChainLadder(n_sims=400, random_seed=11).fit(
        cl.load_sample("genins")
    )


def test_cdr_shapes_and_coords(odp):
    res = claims_development_result(odp)
    assert isinstance(res, CDRResult)
    assert res.cdr.dims == ("future_period", "origin", "sample")
    assert res.cdr.sizes == {"future_period": 9, "origin": 10, "sample": 400}
    assert list(res.cdr.coords["future_period"].values) == list(range(1, 10))
    assert res.total_cdr.dims == ("future_period", "sample")
    assert res.ultimates.sizes["future_period"] == 10  # periods 0..9


def test_cdr_sums_to_lifetime_deviation(odp):
    res = claims_development_result(odp)
    cum, origins, devs = cumulative_array(odp.triangle_)
    f0 = volume_weighted_factors(cum, link_ratio_mask(cum, None, origins, devs))
    cl_ultimate = project_cumulative(cum, f0)[:, -1]
    sim_ultimate = odp.full_cumulative_posterior_.isel(
        dev=-1
    ).values  # (origin, sample)
    lifetime = cl_ultimate[:, None] - sim_ultimate
    np.testing.assert_allclose(
        res.cdr.sum("future_period").transpose("origin", "sample").values,
        lifetime,
        rtol=1e-8,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        res.cumulative().isel(future_period=-1).transpose("origin", "sample").values,
        lifetime,
        rtol=1e-8,
        atol=1e-6,
    )


def test_one_year_sd_below_lifetime_sd(odp):
    res = claims_development_result(odp, future_periods=1)
    assert res.cdr.sizes["future_period"] == 1
    one_year_sd = float(res.total_cdr.isel(future_period=0).std(ddof=1))
    lifetime_sd = float(odp.reserves_posterior_.sum("origin").std(ddof=1))
    assert 0.3 * lifetime_sd < one_year_sd < lifetime_sd


def test_fully_developed_origin_has_zero_cdr(odp):
    res = claims_development_result(odp)
    first = res.cdr.sel(origin=2001).values
    np.testing.assert_allclose(first, 0.0, atol=1e-6)


def test_summary_and_reverse_cumulative(odp):
    res = claims_development_result(odp, var_level=0.99)
    table = res.summary()
    assert set(table.columns) == {"future_period", "origin", "mean", "sd", "var"}
    assert "Total" in set(table["origin"].astype(str))
    total_row = table[
        (table["origin"].astype(str) == "Total") & (table["future_period"] == 1)
    ]
    assert total_row["var"].iloc[0] >= total_row["mean"].iloc[0]
    rev = res.reverse_cumulative()
    np.testing.assert_allclose(
        rev.isel(future_period=0).values, res.cdr.sum("future_period").values
    )


def test_drop_changes_period_zero_ultimate(odp):
    base = claims_development_result(odp)
    dropped = claims_development_result(odp, drop=[("2003", 72)])
    assert not np.allclose(
        base.ultimates.isel(future_period=0).values,
        dropped.ultimates.isel(future_period=0).values,
    )


def test_drop_propagates_to_future_period_masks(odp):
    """``drop`` must also affect the t >= 1 volume-weighted refits, not just
    the deterministic period-0 ultimate."""
    base = claims_development_result(odp)
    dropped = claims_development_result(odp, drop=[("2003", 72)])
    assert not np.allclose(
        base.ultimates.isel(future_period=1).values,
        dropped.ultimates.isel(future_period=1).values,
    )


def test_requires_full_posterior():
    mack = MackChainLadder().fit(cl.load_sample("raa"))
    with pytest.raises(ValueError, match="per-cell"):
        claims_development_result(mack)
