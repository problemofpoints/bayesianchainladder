"""Tests for risk measures, discounting and cost-of-capital helpers."""

import chainladder as cl
import numpy as np
import pytest

from bayesianchainladder.bootstrap import BootstrapODPChainLadder
from bayesianchainladder.riskmeasures import (
    capital_profile,
    cash_flow_periods,
    cost_of_capital_risk_margin,
    discount_factors,
    discounted_reserves,
    equivalent_risk_tolerance,
    future_reserve_profile,
    proportional_hazards_transform,
    tail_value_at_risk,
    value_at_risk,
)


@pytest.fixture(scope="module")
def odp():
    return BootstrapODPChainLadder(n_sims=300, random_seed=7).fit(
        cl.load_sample("genins")
    )


def test_var_tvar_pht_on_simple_samples():
    x = np.arange(1.0, 101.0)
    assert value_at_risk(x, 0.5) == pytest.approx(50.5)
    assert tail_value_at_risk(x, 0.9) == pytest.approx(
        np.mean(x[x >= np.quantile(x, 0.9)])
    )
    assert proportional_hazards_transform(x, 1.0) == pytest.approx(x.mean())
    assert proportional_hazards_transform(x, 2.0) > x.mean()
    with pytest.raises(ValueError):
        proportional_hazards_transform(x, 0.5)


def test_cash_flow_periods_and_discount_factors():
    k = cash_flow_periods(3, 3)
    np.testing.assert_array_equal(k, [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
    with pytest.raises(ValueError, match="square"):
        cash_flow_periods(3, 4)
    np.testing.assert_allclose(
        discount_factors([1, 2], 0.05, offset=0.5), [1.05**-0.5, 1.05**-1.5]
    )
    np.testing.assert_allclose(
        discount_factors([1, 2, 3], 0.05, offset=1.0), 1.05 ** -np.arange(1, 4)
    )


def test_discounted_reserves_zero_rate_equals_reserves(odp):
    disc = discounted_reserves(odp, rate=0.0)
    np.testing.assert_allclose(
        disc.transpose("origin", "sample").values,
        odp.reserves_posterior_.transpose("origin", "sample").values,
        rtol=1e-6,
        atol=1e-6,
    )
    disc3 = discounted_reserves(odp, rate=0.03)
    assert float(disc3.sum("origin").mean()) < float(disc.sum("origin").mean())


def test_future_reserve_profile_shape_and_monotone(odp):
    prof = future_reserve_profile(odp, rate=0.03)
    assert prof.dims == ("period", "sample")
    assert prof.sizes["period"] == 9
    means = prof.mean("sample").values
    assert np.all(np.diff(means) < 0)  # reserves run off over time
    # period 0 with zero discount equals total undiscounted reserve
    prof0 = future_reserve_profile(odp, rate=0.0)
    np.testing.assert_allclose(
        prof0.isel(period=0).values,
        odp.reserves_posterior_.sum("origin").values,
        rtol=1e-6,
    )


def test_cost_of_capital_hand_calculation():
    out = cost_of_capital_risk_margin(
        100.0, np.array([1.0, 0.5]), coc_rate=0.06, discount_rate=0.0
    )
    np.testing.assert_allclose(out["capital"], [100.0, 50.0])
    np.testing.assert_allclose(out["cost"], [6.0, 3.0])
    assert out["risk_margin"] == pytest.approx(9.0)
    out2 = cost_of_capital_risk_margin(
        100.0, np.array([1.0, 0.5]), 0.06, 0.10, offset=1.0
    )
    assert out2["risk_margin"] == pytest.approx(6.0 / 1.1 + 3.0 / 1.1**2)
    np.testing.assert_allclose(
        capital_profile(np.array([200.0, 100.0, 50.0])), [1.0, 0.5, 0.25]
    )


def test_equivalent_risk_tolerance_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.lognormal(10, 0.3, 20000)
    for measure in ("var", "tvar", "pht"):
        p0 = 0.8 if measure != "pht" else 2.0
        fn = {
            "var": value_at_risk,
            "tvar": tail_value_at_risk,
            "pht": proportional_hazards_transform,
        }[measure]
        target = fn(x, p0) - x.mean()
        p = equivalent_risk_tolerance(x, target, measure=measure)
        assert p == pytest.approx(p0, rel=1e-3)
