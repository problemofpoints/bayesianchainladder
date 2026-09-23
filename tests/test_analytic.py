"""Analytic (closed-form) prediction errors used as oracles for the bootstraps."""

import chainladder as cl
import numpy as np
import pytest

from bayesianchainladder.analytic import (
    AnalyticResult,
    mack_analytic_rmsep,
    odp_analytic_rmsep,
    poisson_irls,
)
from bayesianchainladder.bootstrap import BootstrapODPChainLadder


@pytest.fixture(scope="module")
def genins():
    return cl.load_sample("genins")


def test_poisson_irls_recovers_known_coefficients():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(5000), rng.normal(size=5000)])
    beta_true = np.array([1.0, 0.5])
    y = rng.poisson(np.exp(X @ beta_true))
    beta = poisson_irls(X, y.astype(float))
    np.testing.assert_allclose(beta, beta_true, atol=0.05)


def test_odp_reserves_equal_chain_ladder(genins):
    res = odp_analytic_rmsep(genins, scale="constant")
    assert isinstance(res, AnalyticResult)
    cl_ibnr = np.asarray(cl.Chainladder().fit(genins).ibnr_.values)[0, 0, :, 0]
    # chainladder's ibnr_ reports NaN (not 0) for a fully developed origin;
    # nan_to_num maps that to 0, matching our (correct) reserve of 0 there.
    np.testing.assert_allclose(res.reserves, np.nan_to_num(cl_ibnr), rtol=1e-6)
    assert res.total_reserve == pytest.approx(18_680_856, rel=1e-6)
    assert res.reserve_sd[0] == 0.0  # fully developed origin


def test_odp_constant_scale_matches_chainladder_phi(genins):
    res = odp_analytic_rmsep(genins, scale="constant")
    prepared = genins.copy()
    prepared.key_labels = ["triangle_id"]
    prepared.kdims = np.asarray([["resample"]], dtype=object)
    sampler = cl.BootstrapODPSample(n_sims=5, hat_adj=False, random_state=1).fit(prepared)
    assert res.scale.shape == (10,)
    np.testing.assert_allclose(res.scale, float(np.asarray(sampler.scale_).flatten()[0]), rtol=1e-6)


def test_odp_analytic_sd_close_to_bootstrap(genins):
    res = odp_analytic_rmsep(genins, scale="constant")
    boot = BootstrapODPChainLadder(n_sims=4000, random_seed=9).fit(genins)
    assert res.total_sd == pytest.approx(boot.total_summary().total_reserve_stddev, rel=0.12)
    assert 0.10 < res.total_cov < 0.25


def test_odp_nonconstant_scale_varies(genins):
    res = odp_analytic_rmsep(genins, scale="nonconstant")
    assert res.scale.shape == (10,)
    assert not np.allclose(res.scale, res.scale[0])
    assert res.scale[-1] == pytest.approx(min(res.scale[-2], res.scale[-3]))
    assert res.total_sd > 0
    with pytest.raises(ValueError):
        odp_analytic_rmsep(genins, scale="odd")


def test_mack_analytic_matches_chainladder(genins):
    res = mack_analytic_rmsep(genins)
    assert res.total_reserve == pytest.approx(18_680_856, rel=1e-6)
    assert res.total_sd == pytest.approx(2_441_364, rel=1e-4)
    dropped = mack_analytic_rmsep(genins, drop=[("2003", 72)])
    assert dropped.total_reserve != pytest.approx(res.total_reserve, rel=1e-6)
    frame = res.to_frame()
    assert list(frame.columns) == ["reserve", "sd", "cov"]
    assert frame.index[-1] == "Total"
