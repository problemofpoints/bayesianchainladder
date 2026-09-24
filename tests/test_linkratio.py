"""Tests for the Mack and Negative Binomial link-ratio bootstraps (England & Verrall 2006)."""

import chainladder as cl
import numpy as np
import pytest

from bayesianchainladder.base import BaseStochasticReserve
from bayesianchainladder.linkratio import (
    MackBootstrap,
    NegativeBinomialBootstrap,
    draw_with_moments,
)

CL_RESERVE = 18_680_856.0
MACK_SE = 2_441_364.0  # chainladder Mack total SE on genins (log-linear last sigma)


@pytest.fixture(scope="module")
def genins():
    return cl.load_sample("genins")


def test_draw_with_moments_matches_targets():
    rng = np.random.default_rng(0)
    mean = np.full(200_000, 50.0)
    sd = np.full(200_000, 10.0)
    for dist in ("gamma", "lognormal"):
        x = draw_with_moments(mean, sd, dist, rng)
        assert x.mean() == pytest.approx(50.0, rel=0.01)
        assert x.std() == pytest.approx(10.0, rel=0.02)
        assert (x > 0).all()
    neg = draw_with_moments(np.full(1000, -5.0), np.full(1000, 1.0), "gamma", rng)
    assert neg.mean() == pytest.approx(-5.0, abs=0.2)  # Normal fallback
    zero_sd = draw_with_moments(np.array([3.0]), np.array([0.0]), "gamma", rng)
    assert zero_sd[0] == 3.0
    np_draw = draw_with_moments(np.array([1.0, 2.0]), np.array([0.5, 0.5]), "nonparametric", rng, resid=np.array([2.0, -2.0]))
    np.testing.assert_allclose(np_draw, [2.0, 1.0])


def test_mack_bootstrap_matches_analytic(genins):
    model = MackBootstrap(n_sims=4000, random_seed=42).fit(genins)
    assert isinstance(model, BaseStochasticReserve)
    s = model.total_summary()
    assert s.total_reserve_mean == pytest.approx(CL_RESERVE, rel=0.02)
    assert s.total_reserve_stddev == pytest.approx(MACK_SE, rel=0.12)
    assert model.full_cumulative_posterior_.shape == (10, 10, 4000)
    np.testing.assert_allclose(
        model._reserves_from_full_posterior().values, model.reserves_posterior_.values
    )
    cum = np.asarray(genins.values)[0, 0]
    obs = ~np.isnan(cum)
    np.testing.assert_allclose(
        model.full_cumulative_posterior_.values[obs],
        np.repeat(cum[obs][:, None], 4000, axis=1),
    )
    assert model.factors_.shape == (9,) and model.sigma_.shape == (9,)
    assert model.scaled_residuals_.shape == (10, 9)
    assert abs(np.nanmean(model.scaled_residuals_)) < 1e-9  # zero-centred


@pytest.mark.parametrize("bootstrap_dist", ["nonparametric", "gamma", "lognormal"])
@pytest.mark.parametrize("forecast_dist", ["nonparametric", "gamma", "lognormal"])
def test_all_distribution_combinations_run(genins, bootstrap_dist, forecast_dist):
    model = MackBootstrap(
        n_sims=300, bootstrap_dist=bootstrap_dist, forecast_dist=forecast_dist, random_seed=1
    ).fit(genins)
    assert model.total_summary().total_reserve_mean == pytest.approx(CL_RESERVE, rel=0.08)


def test_invalid_distribution_raises():
    with pytest.raises(ValueError, match="bootstrap_dist"):
        MackBootstrap(bootstrap_dist="normal")


def test_drop_and_process_sigma(genins):
    base = MackBootstrap(n_sims=1500, random_seed=3).fit(genins)
    dropped = MackBootstrap(n_sims=1500, random_seed=3, drop=[("2003", 72)]).fit(genins)
    assert dropped.factors_[5] != pytest.approx(base.factors_[5])
    no_process = MackBootstrap(
        n_sims=1500, random_seed=3, process_sigma=np.zeros(9)
    ).fit(genins)
    assert no_process.total_summary().total_reserve_stddev < base.total_summary().total_reserve_stddev
    assert no_process.total_summary().total_reserve_mean == pytest.approx(
        base.total_summary().total_reserve_mean, rel=0.03
    )
    with pytest.raises(ValueError, match="process_sigma"):
        MackBootstrap(process_sigma=np.zeros(3)).fit(genins)


def test_negbin_bootstrap_close_to_mack(genins):
    nb = NegativeBinomialBootstrap(n_sims=3000, random_seed=42).fit(genins)
    mk = MackBootstrap(n_sims=3000, random_seed=42).fit(genins)
    assert nb.total_summary().total_reserve_mean == pytest.approx(CL_RESERVE, rel=0.02)
    assert nb.total_summary().total_reserve_stddev == pytest.approx(
        mk.total_summary().total_reserve_stddev, rel=0.25
    )


def test_mack_bootstrap_handles_negative_incrementals():
    raa = cl.load_sample("raa")  # has negative incrementals
    model = MackBootstrap(n_sims=500, random_seed=0).fit(raa)
    assert np.isfinite(model.total_summary().total_reserve_mean)
    assert np.isfinite(model.reserves_posterior_.values).all()


@pytest.mark.slow
def test_bayesian_mack_recovers_chain_ladder_factors(genins):
    from bayesianchainladder.linkratio import BayesianMackChainLadder

    model = BayesianMackChainLadder(draws=300, tune=300, chains=1, random_seed=42).fit(genins)
    assert model.idata is not None
    post_mean = model.factor_draws_.mean(axis=0)
    np.testing.assert_allclose(post_mean, model.factors_, rtol=0.02)
    assert model.total_summary().total_reserve_mean == pytest.approx(CL_RESERVE, rel=0.05)
    assert model.full_cumulative_posterior_.shape[:2] == (10, 10)
    assert model.full_cumulative_posterior_.shape[2] == model.factor_draws_.shape[0]
