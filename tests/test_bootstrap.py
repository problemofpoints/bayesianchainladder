"""Tests for stochastic reserve wrappers in bayesianchainladder.bootstrap."""

import chainladder as cl
import numpy as np
import pytest

from bayesianchainladder.base import BaseStochasticReserve


@pytest.fixture
def raa_triangle():
    return cl.load_sample("raa")


class TestMackChainLadder:
    def test_inherits_from_base(self):
        from bayesianchainladder.bootstrap import MackChainLadder

        model = MackChainLadder()
        assert isinstance(model, BaseStochasticReserve)

    def test_fit_returns_self(self, raa_triangle):
        from bayesianchainladder.bootstrap import MackChainLadder

        model = MackChainLadder()
        result = model.fit(raa_triangle)
        assert result is model
        assert model._is_fitted is True

    def test_fit_populates_attributes(self, raa_triangle):
        from bayesianchainladder.bootstrap import MackChainLadder

        model = MackChainLadder().fit(raa_triangle)
        assert model.triangle_ is not None
        assert model.reserves_posterior_ is not None
        assert model.ibnr_ is not None
        assert model.ultimate_ is not None

    def test_ibnr_columns(self, raa_triangle):
        from bayesianchainladder.bootstrap import MackChainLadder

        model = MackChainLadder().fit(raa_triangle)
        assert list(model.ibnr_.columns) == [
            "mean", "std", "median", "5%", "25%", "75%", "95%"
        ]

    def test_summary_has_total_row(self, raa_triangle):
        from bayesianchainladder.bootstrap import MackChainLadder

        model = MackChainLadder().fit(raa_triangle)
        summary = model.summary()
        assert "Total" in summary.index

    def test_total_summary_uses_calibrated_total_stderr(self, raa_triangle):
        """MackChainLadder's total_summary().total_reserve_stddev should
        match chainladder's total_mack_std_err_, NOT the std of summed
        per-origin samples (which would be lower because per-origin draws
        are independent)."""
        from bayesianchainladder.bootstrap import MackChainLadder

        mack_native = cl.MackChainladder().fit(raa_triangle)
        expected_total_std = float(
            np.asarray(mack_native.total_mack_std_err_).flatten()[0]
        )

        model = MackChainLadder().fit(raa_triangle)
        result = model.total_summary()
        assert result.total_reserve_stddev == pytest.approx(
            expected_total_std, rel=1e-6
        )

    def test_total_summary_mean_matches_native_mack(self, raa_triangle):
        from bayesianchainladder.bootstrap import MackChainLadder

        mack_native = cl.MackChainladder().fit(raa_triangle)
        expected_mean = float(np.nansum(np.asarray(mack_native.ibnr_.values)))

        model = MackChainLadder().fit(raa_triangle)
        result = model.total_summary()
        assert result.total_reserve_mean == pytest.approx(expected_mean, rel=1e-6)

    def test_sample_reserves_uses_calibrated_total(self, raa_triangle):
        """sample_reserves() should draw from Normal(total_mean,
        total_mack_std_err_), so the empirical std of a large sample should
        be close to total_mack_std_err_."""
        from bayesianchainladder.bootstrap import MackChainLadder

        mack_native = cl.MackChainladder().fit(raa_triangle)
        expected_total_std = float(
            np.asarray(mack_native.total_mack_std_err_).flatten()[0]
        )

        model = MackChainLadder(random_seed=42).fit(raa_triangle)
        samples = model.sample_reserves(n_samples=20000, random_seed=42)
        assert np.std(samples) == pytest.approx(expected_total_std, rel=0.05)

    def test_sample_reserves_is_finite(self, raa_triangle):
        from bayesianchainladder.bootstrap import MackChainLadder

        model = MackChainLadder().fit(raa_triangle)
        samples = model.sample_reserves(n_samples=500, random_seed=0)
        assert np.all(np.isfinite(samples))

    def test_unfit_summary_raises(self):
        from bayesianchainladder.bootstrap import MackChainLadder

        model = MackChainLadder()
        with pytest.raises(ValueError, match="has not been fitted"):
            model.summary()


@pytest.fixture
def genins_triangle():
    """GenIns triangle has all positive incrementals — works with bootstrap."""
    return cl.load_sample("genins")


class TestBootstrapODPChainLadder:
    def test_inherits_from_base(self):
        from bayesianchainladder.bootstrap import BootstrapODPChainLadder

        model = BootstrapODPChainLadder()
        assert isinstance(model, BaseStochasticReserve)

    def test_fit_returns_self(self, genins_triangle):
        from bayesianchainladder.bootstrap import BootstrapODPChainLadder

        model = BootstrapODPChainLadder(n_sims=100, random_seed=42)
        result = model.fit(genins_triangle)
        assert result is model
        assert model._is_fitted is True

    def test_fit_populates_reserves_posterior(self, genins_triangle):
        from bayesianchainladder.bootstrap import BootstrapODPChainLadder

        model = BootstrapODPChainLadder(n_sims=100, random_seed=42).fit(
            genins_triangle
        )
        assert model.reserves_posterior_ is not None
        # Expect dims (origin, sample) with sample size = n_sims
        assert "origin" in model.reserves_posterior_.dims
        sample_dims = [d for d in model.reserves_posterior_.dims if d != "origin"]
        sample_size = int(np.prod([model.reserves_posterior_.sizes[d]
                                   for d in sample_dims]))
        assert sample_size == 100

    def test_summary_has_total_row(self, genins_triangle):
        from bayesianchainladder.bootstrap import BootstrapODPChainLadder

        model = BootstrapODPChainLadder(n_sims=100, random_seed=42).fit(
            genins_triangle
        )
        summary = model.summary()
        assert "Total" in summary.index

    def test_total_summary_returns_finite(self, genins_triangle):
        from bayesianchainladder.bootstrap import BootstrapODPChainLadder

        model = BootstrapODPChainLadder(n_sims=200, random_seed=42).fit(
            genins_triangle
        )
        result = model.total_summary()
        assert np.isfinite(result.total_reserve_mean)
        assert np.isfinite(result.total_reserve_stddev)
        assert result.total_reserve_stddev > 0

    def test_total_summary_mean_close_to_native_chainladder(self, genins_triangle):
        """Bootstrap mean reserve should be close to the native chainladder
        IBNR (within ~5% with n_sims=500)."""
        from bayesianchainladder.bootstrap import BootstrapODPChainLadder

        cl_model = cl.Chainladder().fit(genins_triangle)
        cl_total = float(np.nansum(np.asarray(cl_model.ibnr_.values)))

        model = BootstrapODPChainLadder(n_sims=500, random_seed=42).fit(
            genins_triangle
        )
        boot_total = model.total_summary().total_reserve_mean

        assert boot_total == pytest.approx(cl_total, rel=0.05)

    def test_random_seed_makes_run_deterministic(self, genins_triangle):
        from bayesianchainladder.bootstrap import BootstrapODPChainLadder

        model_a = BootstrapODPChainLadder(n_sims=100, random_seed=7).fit(
            genins_triangle
        )
        model_b = BootstrapODPChainLadder(n_sims=100, random_seed=7).fit(
            genins_triangle
        )
        assert model_a.total_summary().total_reserve_mean == pytest.approx(
            model_b.total_summary().total_reserve_mean
        )
