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


class TestCorrelatedBootstrapODPSample:
    """Smoke tests for the low-level transformer."""

    def test_fit_with_rho_zero(self, genins_triangle):
        from bayesianchainladder.bootstrap import CorrelatedBootstrapODPSample

        sampler = CorrelatedBootstrapODPSample(
            n_sims=50, rho=0.0, random_state=42
        )
        sampler.fit(genins_triangle)
        # rho=0 path doesn't build a correlation matrix
        assert sampler.correlation_matrix_ is None
        assert sampler.scale_ is not None

    def test_fit_with_rho_positive_builds_correlation_matrix(self, genins_triangle):
        from bayesianchainladder.bootstrap import CorrelatedBootstrapODPSample

        sampler = CorrelatedBootstrapODPSample(
            n_sims=50, rho=0.5, random_state=42
        )
        sampler.fit(genins_triangle)
        assert sampler.correlation_matrix_ is not None
        # Diagonal should be 1
        diag = np.diag(sampler.correlation_matrix_)
        assert np.allclose(diag, 1.0)
        # Same-calendar-year off-diagonals should equal rho
        # (cell (0,1) and (1,0) are both calendar year 1)
        idx_a = sampler.valid_indices_.index((0, 1))
        idx_b = sampler.valid_indices_.index((1, 0))
        assert sampler.correlation_matrix_[idx_a, idx_b] == pytest.approx(0.5)

    def test_invalid_parametric_dist_raises(self):
        from bayesianchainladder.bootstrap import CorrelatedBootstrapODPSample

        with pytest.raises(ValueError, match="parametric_dist"):
            CorrelatedBootstrapODPSample(parametric_dist="weibull")

    def test_invalid_rho_raises(self):
        from bayesianchainladder.bootstrap import CorrelatedBootstrapODPSample

        with pytest.raises(ValueError, match="rho must be in"):
            CorrelatedBootstrapODPSample(rho=1.5)
        with pytest.raises(ValueError, match="rho must be in"):
            CorrelatedBootstrapODPSample(rho=-0.1)

    def test_transform_produces_n_sims_resamples(self, genins_triangle):
        from bayesianchainladder.bootstrap import CorrelatedBootstrapODPSample

        sampler = CorrelatedBootstrapODPSample(
            n_sims=20, rho=0.3, random_state=42
        )
        sampler.fit(genins_triangle)
        resampled = sampler.transform(genins_triangle)
        # The resampled triangle's first dim should be n_sims
        assert resampled.values.shape[0] == 20


class TestCorrelatedBootstrapChainLadder:
    def test_inherits_from_base(self):
        from bayesianchainladder.bootstrap import CorrelatedBootstrapChainLadder

        model = CorrelatedBootstrapChainLadder()
        assert isinstance(model, BaseStochasticReserve)

    def test_fit_returns_self(self, genins_triangle):
        from bayesianchainladder.bootstrap import CorrelatedBootstrapChainLadder

        model = CorrelatedBootstrapChainLadder(
            n_sims=100, rho=0.3, random_seed=42
        )
        result = model.fit(genins_triangle)
        assert result is model
        assert model._is_fitted is True

    def test_summary_has_total_row(self, genins_triangle):
        from bayesianchainladder.bootstrap import CorrelatedBootstrapChainLadder

        model = CorrelatedBootstrapChainLadder(
            n_sims=200, rho=0.3, random_seed=42
        ).fit(genins_triangle)
        summary = model.summary()
        assert "Total" in summary.index

    def test_rho_zero_matches_independent_bootstrap(self, genins_triangle):
        """With rho=0, the correlated wrapper should produce the same total
        std as the independent BootstrapODPChainLadder when seeded
        identically."""
        from bayesianchainladder.bootstrap import (
            BootstrapODPChainLadder,
            CorrelatedBootstrapChainLadder,
        )

        # Note: not exact match because the underlying samplers differ in
        # implementation, but the totals should be in the same ballpark.
        indep = BootstrapODPChainLadder(n_sims=500, random_seed=42).fit(
            genins_triangle
        )
        corr = CorrelatedBootstrapChainLadder(
            n_sims=500, rho=0.0, random_seed=42
        ).fit(genins_triangle)

        indep_std = indep.total_summary().total_reserve_stddev
        corr_std = corr.total_summary().total_reserve_stddev
        assert corr_std == pytest.approx(indep_std, rel=0.20)

    def test_rho_positive_increases_total_std(self, genins_triangle):
        """Higher rho should produce a wider total reserve distribution."""
        from bayesianchainladder.bootstrap import CorrelatedBootstrapChainLadder

        low = CorrelatedBootstrapChainLadder(
            n_sims=500, rho=0.0, random_seed=42
        ).fit(genins_triangle)
        high = CorrelatedBootstrapChainLadder(
            n_sims=500, rho=0.5, random_seed=42
        ).fit(genins_triangle)

        assert (
            high.total_summary().total_reserve_stddev
            > low.total_summary().total_reserve_stddev
        )

    def test_lognormal_distribution_runs(self, genins_triangle):
        from bayesianchainladder.bootstrap import CorrelatedBootstrapChainLadder

        model = CorrelatedBootstrapChainLadder(
            n_sims=200,
            rho=0.3,
            parametric_dist="lognormal",
            random_seed=42,
        ).fit(genins_triangle)
        result = model.total_summary()
        assert np.isfinite(result.total_reserve_mean)
        assert result.total_reserve_stddev > 0

    def test_mean_unbiased_for_positive_rho(self, genins_triangle):
        """Regression guard: mean total reserve must track deterministic
        chain ladder for rho > 0 (within Monte Carlo noise). Previously
        biased ~26% downward due to nancumsum in correlated paths."""
        from bayesianchainladder.bootstrap import CorrelatedBootstrapChainLadder

        cl_model = cl.Chainladder().fit(genins_triangle)
        cl_total = float(np.nansum(np.asarray(cl_model.ibnr_.values)))

        for rho in [0.1, 0.3, 0.5]:
            model = CorrelatedBootstrapChainLadder(
                n_sims=1000, rho=rho, random_seed=42
            ).fit(genins_triangle)
            mean = model.total_summary().total_reserve_mean
            # Allow 10% tolerance for Monte Carlo noise at n_sims=1000
            assert abs(mean / cl_total - 1) < 0.10, (
                f"rho={rho}: mean={mean:,.0f} drifted >{10}% from CL={cl_total:,.0f}"
            )
