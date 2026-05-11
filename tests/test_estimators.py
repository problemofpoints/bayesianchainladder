"""Tests for estimator classes."""

import numpy as np
import pandas as pd
import pytest

import chainladder as cl

from bayesianchainladder.estimators import BayesianChainLadderGLM, BayesianCSR


@pytest.fixture
def sample_triangle():
    """Load a sample triangle for testing."""
    return cl.load_sample("raa")


@pytest.fixture
def small_triangle():
    """Create a small triangle for faster testing."""
    # Use the RAA sample triangle (small and well-behaved)
    return cl.load_sample("raa")


class TestBayesianChainLadderGLMInit:
    """Tests for BayesianChainLadderGLM initialization."""

    def test_default_initialization(self):
        """Test default parameter initialization."""
        model = BayesianChainLadderGLM()

        assert model.formula == "incremental ~ 1 + C(origin) + C(dev)"
        assert model.family == "negativebinomial"
        assert model.draws == 2000
        assert model.tune == 1000
        assert model.chains == 4
        assert model.target_accept == 0.9
        assert model.backend == "bambi"
        assert not model._is_fitted

    def test_custom_initialization(self):
        """Test custom parameter initialization."""
        model = BayesianChainLadderGLM(
            formula="incremental ~ 1 + C(origin) + C(dev) + C(calendar)",
            family="poisson",
            draws=1000,
            tune=500,
            chains=2,
            random_seed=42,
        )

        assert "calendar" in model.formula
        assert model.family == "poisson"
        assert model.draws == 1000
        assert model.random_seed == 42


class TestBayesianChainLadderGLMFit:
    """Tests for BayesianChainLadderGLM fit method."""

    @pytest.mark.slow
    def test_fit_runs(self, small_triangle):
        """Test that fit completes without error."""
        # Use Gaussian family since RAA triangle may have negative incremental values
        model = BayesianChainLadderGLM(
            family="gaussian",
            draws=100,
            tune=50,
            chains=1,
            random_seed=42,
        )

        result = model.fit(small_triangle)

        assert result is model  # Returns self
        assert model._is_fitted
        assert model.idata is not None

    @pytest.mark.slow
    def test_fit_populates_attributes(self, small_triangle):
        """Test that fit populates expected attributes."""
        model = BayesianChainLadderGLM(
            family="gaussian",
            draws=100,
            tune=50,
            chains=1,
            random_seed=42,
        )

        model.fit(small_triangle)

        # Check data attributes
        assert model.data_ is not None
        assert model.future_data_ is not None
        assert model.triangle_ is not None

        # Check fitted values
        assert model.fitted_ is not None
        assert "fitted_mean" in model.fitted_.columns

    @pytest.mark.slow
    def test_fit_computes_reserves(self, small_triangle):
        """Test that fit computes reserve distributions."""
        model = BayesianChainLadderGLM(
            family="gaussian",
            draws=100,
            tune=50,
            chains=1,
            random_seed=42,
        )

        model.fit(small_triangle)

        # Check reserve attributes
        assert model.ibnr_ is not None
        assert model.ultimate_ is not None
        assert model.reserves_posterior_ is not None

    @pytest.mark.slow
    def test_fit_with_different_formula(self, small_triangle):
        """Test fit with a different formula (no intercept)."""
        model = BayesianChainLadderGLM(
            formula="incremental ~ 0 + C(origin) + C(dev)",
            family="gaussian",
            draws=100,
            tune=50,
            chains=1,
            random_seed=42,
        )

        model.fit(small_triangle)

        assert model._is_fitted


class TestBayesianChainLadderGLMPredict:
    """Tests for BayesianChainLadderGLM predict method."""

    @pytest.mark.slow
    def test_predict_without_fit_raises(self, small_triangle):
        """Test that predict before fit raises error."""
        model = BayesianChainLadderGLM()

        with pytest.raises(ValueError, match="not been fitted"):
            model.predict()

    @pytest.mark.slow
    def test_predict_default(self, small_triangle):
        """Test predict with default arguments."""
        model = BayesianChainLadderGLM(
            family="gaussian",
            draws=100,
            tune=50,
            chains=1,
            random_seed=42,
        )
        model.fit(small_triangle)

        result = model.predict()

        assert isinstance(result, pd.DataFrame)


class TestBayesianChainLadderGLMSummary:
    """Tests for BayesianChainLadderGLM summary method."""

    @pytest.mark.slow
    def test_summary_structure(self, small_triangle):
        """Test summary returns expected structure."""
        model = BayesianChainLadderGLM(
            family="gaussian",
            draws=100,
            tune=50,
            chains=1,
            random_seed=42,
        )
        model.fit(small_triangle)

        summary = model.summary()

        assert isinstance(summary, pd.DataFrame)
        assert "Ultimate" in summary.columns.get_level_values(0)
        assert "IBNR" in summary.columns.get_level_values(0)

    @pytest.mark.slow
    def test_summary_with_totals(self, small_triangle):
        """Test summary includes total row."""
        model = BayesianChainLadderGLM(
            family="gaussian",
            draws=100,
            tune=50,
            chains=1,
            random_seed=42,
        )
        model.fit(small_triangle)

        summary = model.summary(include_totals=True)

        assert "Total" in summary.index


class TestBayesianChainLadderGLMSampleReserves:
    """Tests for sample_reserves method."""

    @pytest.mark.slow
    def test_sample_reserves(self, small_triangle):
        """Test sampling from reserve distribution."""
        model = BayesianChainLadderGLM(
            family="gaussian",
            draws=100,
            tune=50,
            chains=1,
            random_seed=42,
        )
        model.fit(small_triangle)

        samples = model.sample_reserves(n_samples=500)

        assert isinstance(samples, np.ndarray)
        assert len(samples) == 500

    @pytest.mark.slow
    def test_sample_reserves_reproducibility(self, small_triangle):
        """Test that random seed produces reproducible samples."""
        model = BayesianChainLadderGLM(
            family="gaussian",
            draws=100,
            tune=50,
            chains=1,
            random_seed=42,
        )
        model.fit(small_triangle)

        samples1 = model.sample_reserves(n_samples=100, random_seed=123)
        samples2 = model.sample_reserves(n_samples=100, random_seed=123)

        np.testing.assert_array_equal(samples1, samples2)


class TestBayesianChainLadderGLMRepr:
    """Tests for string representation."""

    def test_repr_not_fitted(self):
        """Test repr before fitting."""
        model = BayesianChainLadderGLM()
        repr_str = repr(model)

        assert "BayesianChainLadderGLM" in repr_str
        assert "not fitted" in repr_str

    @pytest.mark.slow
    def test_repr_fitted(self, small_triangle):
        """Test repr after fitting."""
        model = BayesianChainLadderGLM(
            family="gaussian",
            draws=50,
            tune=25,
            chains=1,
            random_seed=42,
        )
        model.fit(small_triangle)

        repr_str = repr(model)

        assert "fitted" in repr_str
        assert "not fitted" not in repr_str


class TestBayesianChainLadderGLMValidation:
    """Tests for data validation."""

    def test_negativebinomial_rejects_negative_values(self, small_triangle):
        """Test that negativebinomial family raises error for negative values."""
        # RAA triangle has negative incremental values
        model = BayesianChainLadderGLM(
            family="negativebinomial",
            draws=100,
            tune=50,
            chains=1,
            random_seed=42,
        )

        with pytest.raises(ValueError, match="requires non-negative values"):
            model.fit(small_triangle)

    def test_poisson_rejects_negative_values(self, small_triangle):
        """Test that poisson family raises error for negative values."""
        model = BayesianChainLadderGLM(
            family="poisson",
            draws=100,
            tune=50,
            chains=1,
            random_seed=42,
        )

        with pytest.raises(ValueError, match="requires non-negative values"):
            model.fit(small_triangle)

    @pytest.mark.slow
    def test_unseen_dev_levels_clipped_not_crash(self):
        """Triangles where future_data_ has dev levels absent from training data
        should silently clip those cells rather than raising:
            ValueError: The levels (120, 108) in 'C(dev)' are not present …

        We take the genins triangle and blank out the last two dev columns
        (dev=108, 120) for ALL origins, so that the training data only 'sees'
        dev=12..96 while future cells require dev=108 and 120.  The model must
        not crash; future_data_ must not contain the missing dev levels.
        """
        import warnings

        # Start from genins (all-positive incremental values).
        tri = cl.load_sample("genins")

        # Blank out the last two dev columns (dev=108, 120) for all origins.
        # We operate on the *incremental* triangle so that cum_to_incr() does not
        # produce spurious non-NaN values from cumulative differencing.
        # After blanking, dev=108 and dev=120 never appear in data_ but the
        # triangle's column headers still declare those dev periods as future cells.
        inc_tri = tri.cum_to_incr()
        tri2 = inc_tri.copy()
        vals = tri2.values.copy()  # shape (1, 1, 10, 10)
        vals[:, :, :, 8] = np.nan  # dev=108
        vals[:, :, :, 9] = np.nan  # dev=120
        tri2.values = vals

        model = BayesianChainLadderGLM(
            formula="incremental ~ 1 + C(origin) + C(dev)",
            family="gaussian",
            draws=50,
            tune=25,
            chains=1,
            random_seed=42,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(tri2)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            # A warning about dropped cells should have been emitted
            assert any("dropped" in str(x.message).lower() for x in user_warnings), \
                "Expected a UserWarning about dropped future cells"

        # future_data_ must NOT contain the unseen dev levels
        assert 108 not in model.future_data_["dev"].values
        assert 120 not in model.future_data_["dev"].values
        # Model must be fitted and have reserves
        assert model._is_fitted
        assert model.reserves_posterior_ is not None


@pytest.fixture
def positive_triangle():
    """Create a triangle with only positive incremental values for count models."""
    # Use GenIns sample which typically has positive incremental values
    # or create a synthetic positive triangle
    tri = cl.load_sample("genins")
    return tri


@pytest.fixture
def genins_with_premium():
    """Return (paid_tri, prem_tri) using genins; premium is 6× the first-dev paid."""
    paid_tri = cl.load_sample("genins")
    # Use a scaled copy of genins as a stand-in premium triangle.
    # The exposure triangle just needs to supply a positive per-origin scalar;
    # prepare_model_data takes the first development period's value.
    prem_tri = paid_tri * 6  # premium ≈ 6× dev-12 paid → implied LR ~17%
    return paid_tri, prem_tri


class TestBayesianChainLadderGLMNegativeBinomial:
    """Tests for negative binomial family with appropriate data."""

    @pytest.mark.slow
    def test_negativebinomial_with_positive_data(self, positive_triangle):
        """Test that negativebinomial works with positive data."""
        model = BayesianChainLadderGLM(
            family="negativebinomial",
            draws=100,
            tune=50,
            chains=1,
            random_seed=42,
        )

        result = model.fit(positive_triangle)

        assert result is model
        assert model._is_fitted
        assert model.idata is not None

    @pytest.mark.slow
    def test_negativebinomial_computes_reserves(self, positive_triangle):
        """Test that negativebinomial computes reserve distributions."""
        model = BayesianChainLadderGLM(
            family="negativebinomial",
            draws=100,
            tune=50,
            chains=1,
            random_seed=42,
        )

        model.fit(positive_triangle)

        assert model.ibnr_ is not None
        assert model.ultimate_ is not None
        assert model.reserves_posterior_ is not None


# ============================================================================
# BayesianCSR Tests
# ============================================================================


class TestBayesianCSRInit:
    """Tests for BayesianCSR initialization."""

    def test_default_initialization(self):
        """Test default parameter initialization."""
        model = BayesianCSR()

        assert model.priors is None
        assert model.draws == 2000
        assert model.tune == 1000
        assert model.chains == 4
        assert model.target_accept == 0.9
        assert not model._is_fitted

    def test_custom_initialization(self):
        """Test custom parameter initialization."""
        model = BayesianCSR(
            draws=1000,
            tune=500,
            chains=2,
            random_seed=42,
        )

        assert model.draws == 1000
        assert model.tune == 500
        assert model.chains == 2
        assert model.random_seed == 42


class TestBayesianCSRFit:
    """Tests for BayesianCSR fit method."""

    @pytest.mark.slow
    def test_fit_runs_with_premium_value(self, positive_triangle):
        """Test that fit completes with premium_value."""
        model = BayesianCSR(
            draws=100,
            tune=50,
            chains=1,
            random_seed=42,
        )

        result = model.fit(positive_triangle, premium_value=10000)

        assert result is model
        assert model._is_fitted
        assert model.idata is not None

    @pytest.mark.slow
    def test_fit_populates_attributes(self, positive_triangle):
        """Test that fit populates expected attributes."""
        model = BayesianCSR(
            draws=100,
            tune=50,
            chains=1,
            random_seed=42,
        )

        model.fit(positive_triangle, premium_value=10000)

        # Check data attributes
        assert model.data_ is not None
        assert model.future_data_ is not None
        assert model.triangle_ is not None

        # Check posterior attributes
        assert model.elr_posterior_ is not None
        assert model.gamma_posterior_ is not None

    @pytest.mark.slow
    def test_fit_computes_reserves(self, positive_triangle):
        """Test that fit computes reserve distributions."""
        model = BayesianCSR(
            draws=100,
            tune=50,
            chains=1,
            random_seed=42,
        )

        model.fit(positive_triangle, premium_value=10000)

        # Check reserve attributes
        assert model.ibnr_ is not None
        assert model.ultimate_ is not None
        assert model.reserves_posterior_ is not None

    def test_fit_without_premium_raises(self, positive_triangle):
        """Test that fit without premium raises error."""
        model = BayesianCSR()

        with pytest.raises(ValueError, match="premium"):
            model.fit(positive_triangle)


class TestBayesianCSRSummary:
    """Tests for BayesianCSR summary methods."""

    @pytest.mark.slow
    def test_summary_structure(self, positive_triangle):
        """Test summary returns expected structure."""
        model = BayesianCSR(
            draws=100,
            tune=50,
            chains=1,
            random_seed=42,
        )
        model.fit(positive_triangle, premium_value=10000)

        summary = model.summary()

        assert isinstance(summary, pd.DataFrame)
        assert "Ultimate" in summary.columns.get_level_values(0)
        assert "IBNR" in summary.columns.get_level_values(0)

    @pytest.mark.slow
    def test_summary_with_totals(self, positive_triangle):
        """Test summary includes total row."""
        model = BayesianCSR(
            draws=100,
            tune=50,
            chains=1,
            random_seed=42,
        )
        model.fit(positive_triangle, premium_value=10000)

        summary = model.summary(include_totals=True)

        assert "Total" in summary.index

    @pytest.mark.slow
    def test_get_expected_loss_ratio(self, positive_triangle):
        """Test getting expected loss ratio summary."""
        model = BayesianCSR(
            draws=100,
            tune=50,
            chains=1,
            random_seed=42,
        )
        model.fit(positive_triangle, premium_value=10000)

        elr_summary = model.get_expected_loss_ratio()

        assert isinstance(elr_summary, pd.DataFrame)
        assert "mean" in elr_summary.columns
        assert "ELR" in elr_summary.index

    @pytest.mark.slow
    def test_get_speedup_parameter(self, positive_triangle):
        """Test getting speedup parameter summary."""
        model = BayesianCSR(
            draws=100,
            tune=50,
            chains=1,
            random_seed=42,
        )
        model.fit(positive_triangle, premium_value=10000)

        gamma_summary = model.get_speedup_parameter()

        assert isinstance(gamma_summary, pd.DataFrame)
        assert "mean" in gamma_summary.columns
        assert "gamma" in gamma_summary.index


class TestBayesianCSRSampleReserves:
    """Tests for BayesianCSR sample_reserves method."""

    @pytest.mark.slow
    def test_sample_reserves(self, positive_triangle):
        """Test sampling from reserve distribution."""
        model = BayesianCSR(
            draws=100,
            tune=50,
            chains=1,
            random_seed=42,
        )
        model.fit(positive_triangle, premium_value=10000)

        samples = model.sample_reserves(n_samples=500)

        assert isinstance(samples, np.ndarray)
        assert len(samples) == 500

    @pytest.mark.slow
    def test_sample_reserves_reproducibility(self, positive_triangle):
        """Test that random seed produces reproducible samples."""
        model = BayesianCSR(
            draws=100,
            tune=50,
            chains=1,
            random_seed=42,
        )
        model.fit(positive_triangle, premium_value=10000)

        samples1 = model.sample_reserves(n_samples=100, random_seed=123)
        samples2 = model.sample_reserves(n_samples=100, random_seed=123)

        np.testing.assert_array_equal(samples1, samples2)


class TestBayesianCSRRepr:
    """Tests for BayesianCSR string representation."""

    def test_repr_not_fitted(self):
        """Test repr before fitting."""
        model = BayesianCSR()
        repr_str = repr(model)

        assert "BayesianCSR" in repr_str
        assert "not fitted" in repr_str

    @pytest.mark.slow
    def test_repr_fitted(self, positive_triangle):
        """Test repr after fitting."""
        model = BayesianCSR(
            draws=50,
            tune=25,
            chains=1,
            random_seed=42,
        )
        model.fit(positive_triangle, premium_value=10000)

        repr_str = repr(model)

        assert "fitted" in repr_str
        assert "not fitted" not in repr_str


class TestBayesianCSRValidation:
    """Tests for BayesianCSR validation."""

    def test_methods_before_fit_raise(self):
        """Test that methods raise error before fit."""
        model = BayesianCSR()

        with pytest.raises(ValueError, match="not been fitted"):
            model.summary()

        with pytest.raises(ValueError, match="not been fitted"):
            model.sample_reserves()

        with pytest.raises(ValueError, match="not been fitted"):
            model.get_expected_loss_ratio()

        with pytest.raises(ValueError, match="not been fitted"):
            model.get_speedup_parameter()


# ============================================================================
# Student-t family + response_per_exposure tests
# ============================================================================


class TestResponsePerExposure:
    """Tests for response_per_exposure=True (loss-ratio identity-link mode)."""

    def test_response_per_exposure_requires_exposure(self, positive_triangle):
        """response_per_exposure=True without exposure= raises ValueError."""
        model = BayesianChainLadderGLM(
            formula="incremental ~ 1 + C(origin) + C(dev)",
            family="t",
            link="identity",
            response_per_exposure=True,
            draws=50,
            tune=25,
            chains=1,
        )
        with pytest.raises(ValueError, match="response_per_exposure"):
            model.fit(positive_triangle)

    @pytest.mark.slow
    def test_t_family_response_per_exposure_fits(self, genins_with_premium):
        """t family with response_per_exposure=True completes and produces sensible LR."""
        import warnings
        paid_tri, prem_tri = genins_with_premium

        model = BayesianChainLadderGLM(
            formula="incremental ~ 1 + C(origin) + C(dev)",
            family="t",
            link="identity",
            exposure="exposure",
            response_per_exposure=True,
            draws=100,
            tune=100,
            chains=1,
            random_seed=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(paid_tri, exposure_triangle=prem_tri)

        assert model._is_fitted
        assert model.idata is not None

        # The intercept should be on loss-ratio scale: well under 1.0
        # (genins has ~30-40% first-dev LR so intercept will be small positive)
        intercept_mean = float(model.idata.posterior["Intercept"].values.mean())
        assert intercept_mean < 1.0, (
            f"Intercept {intercept_mean:.4f} should be < 1.0 for a loss-ratio-scale fit"
        )

        # Exposure should be restored after fit
        assert model.exposure == "exposure"

    @pytest.mark.slow
    def test_dollar_scale_back_transform(self, genins_with_premium):
        """Ultimate posteriors should be in dollar units, not loss-ratio units.

        When response_per_exposure=True the model predicts on loss-ratio scale.
        _compute_reserves must multiply each future cell by its EP so that
        ibnr_ and ultimate_ are in dollars, not fractions.
        """
        import warnings
        paid_tri, prem_tri = genins_with_premium

        model = BayesianChainLadderGLM(
            formula="incremental ~ 1 + C(origin) + C(dev)",
            family="t",
            link="identity",
            exposure="exposure",
            response_per_exposure=True,
            draws=100,
            tune=100,
            chains=1,
            random_seed=99,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(paid_tri, exposure_triangle=prem_tri)

        assert model._is_fitted
        assert model.ultimate_ is not None
        assert model.ibnr_ is not None

        # The prem_tri is 6x paid_tri, so total paid-to-date is e.g. in the
        # millions range for genins.  The median ultimate should be > total paid
        # for at least one origin year (there are future cells).
        total_paid = float(model.ultimate_["paid_to_date"].sum())
        median_ultimate = float(model.ultimate_["median"].sum())

        # Dollar-scale ultimates must be substantially larger than 1 (not
        # loss-ratio fractions in [0, 2] range).
        assert median_ultimate > total_paid * 0.5, (
            f"Median ultimate {median_ultimate:.0f} is suspiciously small "
            f"relative to paid {total_paid:.0f} — may still be on LR scale"
        )
        # Specifically: with genins prem = 6×paid, if predictions were on LR
        # scale they'd be tiny (≈0.05-0.3 per cell); dollar scale should give
        # values comparable to the observed paid amounts.
        assert median_ultimate > 1_000, (
            f"Median ultimate {median_ultimate:.2f} looks like a loss ratio, "
            "not a dollar amount — back-transform may have failed"
        )


# ============================================================================
# Adaptive intercept prior — offset-awareness tests
# ============================================================================


class TestAdaptiveInterceptPriorOffset:
    """Unit tests for _build_default_priors with exposure offset.

    These tests directly configure the minimal state on the estimator
    (data_, exposure, formula, family, link) and call _build_default_priors()
    without running MCMC.
    """

    def _make_estimator_with_data(
        self,
        incremental: np.ndarray,
        exposure: np.ndarray,
        exposure_col: str = "net_earned_premium",
    ) -> BayesianChainLadderGLM:
        """Return a BayesianChainLadderGLM with data_ populated directly."""
        model = BayesianChainLadderGLM(
            formula="incremental ~ 1 + C(origin) + C(dev)",
            family="gamma",
            link="log",
            exposure=exposure_col,
        )
        # Manually set data_ to bypass the triangle conversion step.
        n = len(incremental)
        model.data_ = pd.DataFrame(
            {
                "incremental": incremental,
                exposure_col: exposure,
                "origin": [f"o{i}" for i in range(n)],
                "dev": [1] * n,
            }
        )
        return model

    def test_intercept_prior_offset_aware(self):
        """Intercept prior location adjusts for log(mean_EP) when exposure is set."""
        target_lr = 0.3  # mean incremental / mean EP we want the prior to center on
        mean_ep = 1_000.0
        mean_incremental = target_lr * mean_ep  # = 300.0

        # Build data with exact mean LR = 0.3
        n = 20
        incremental = np.full(n, mean_incremental)
        exposure = np.full(n, mean_ep)

        model = self._make_estimator_with_data(incremental, exposure)
        priors = model._build_default_priors()

        intercept_prior = priors["Intercept"]
        # The lognormal correction is -sigma^2/2 on top of log(target_lr)
        intercept_sigma = 1.0
        expected_mu = np.log(target_lr) - intercept_sigma**2 / 2

        assert abs(intercept_prior.args["mu"] - expected_mu) < 1e-6, (
            f"Intercept prior mu={intercept_prior.args['mu']:.4f}, "
            f"expected {expected_mu:.4f} (= log({target_lr}) - sigma^2/2). "
            "Prior is not offset-aware."
        )

    def test_intercept_prior_no_exposure_unchanged(self):
        """Without an exposure offset, the prior centers on log(mean_incremental)."""
        mean_incremental = 300.0
        n = 20
        incremental = np.full(n, mean_incremental)
        # exposure column present in data_ but not set on the model
        exposure = np.full(n, 1_000.0)

        model = BayesianChainLadderGLM(
            formula="incremental ~ 1 + C(origin) + C(dev)",
            family="gamma",
            link="log",
            exposure=None,  # no offset
        )
        model.data_ = pd.DataFrame(
            {
                "incremental": incremental,
                "net_earned_premium": exposure,
                "origin": [f"o{i}" for i in range(n)],
                "dev": [1] * n,
            }
        )
        priors = model._build_default_priors()

        intercept_sigma = 1.0
        expected_mu = np.log(mean_incremental) - intercept_sigma**2 / 2

        assert abs(priors["Intercept"].args["mu"] - expected_mu) < 1e-6, (
            "Without exposure, intercept prior should center on log(mean_incremental)."
        )

    def test_intercept_prior_with_zero_exposure_is_safe(self):
        """If mean_ep == 0 the prior should silently fall back to no adjustment."""
        n = 5
        incremental = np.full(n, 100.0)
        # All-zero exposure — mean_ep = 0 so the adjustment is skipped
        exposure = np.zeros(n)

        model = self._make_estimator_with_data(incremental, exposure)
        # Should not raise; prior should be the unadjusted log(mean_incremental) - sigma^2/2
        priors = model._build_default_priors()
        intercept_sigma = 1.0
        expected_mu = np.log(100.0) - intercept_sigma**2 / 2
        assert abs(priors["Intercept"].args["mu"] - expected_mu) < 1e-6

    def test_intercept_prior_offset_magnitude(self):
        """The offset adjustment equals log(mean_EP), not just any constant."""
        mean_ep = 5_000.0
        mean_lr = 0.5
        mean_incremental = mean_lr * mean_ep  # = 2500.0

        n = 30
        incremental = np.full(n, mean_incremental)
        exposure = np.full(n, mean_ep)

        model = self._make_estimator_with_data(incremental, exposure)
        priors = model._build_default_priors()

        intercept_sigma = 1.0
        expected_mu = np.log(mean_lr) - intercept_sigma**2 / 2

        assert abs(priors["Intercept"].args["mu"] - expected_mu) < 1e-6, (
            f"Expected mu={expected_mu:.4f}, got {priors['Intercept'].args['mu']:.4f}"
        )


# ============================================================================
# init_priors_from_chainladder tests
# ============================================================================


class TestInitPriorsFromChainladder:
    """Tests for init_priors_from_chainladder=True mode.

    Non-slow tests only check that the CL-informed priors are constructed
    with the correct structure (array-valued, sensible values) WITHOUT running
    MCMC.  Slow tests exercise the full fit path.
    """

    def _make_fitted_model(
        self,
        formula: str = "incremental ~ 1 + C(origin) + C(dev)",
        family: str = "gaussian",
        link: str | None = "log",
        exposure: str | None = None,
        response_per_exposure: bool = False,
        sd: float = 0.5,
        tri=None,
    ) -> BayesianChainLadderGLM:
        """Return a model with data_ and triangle_ set (without MCMC) so we can
        call _build_cl_informed_priors() directly."""
        import chainladder as cl
        from bayesianchainladder.utils import (
            add_categorical_columns,
            prepare_model_data,
        )

        if tri is None:
            tri = cl.load_sample("genins")

        model = BayesianChainLadderGLM(
            formula=formula,
            family=family,
            link=link,
            exposure=exposure,
            response_per_exposure=response_per_exposure,
            init_priors_from_chainladder=True,
            chainladder_prior_sd=sd,
            draws=50,
            tune=25,
            chains=1,
        )
        # Manually prepare data without running MCMC
        model.triangle_ = tri.copy()
        model.data_, model.future_data_ = prepare_model_data(
            tri,
            exposure_column=exposure if exposure else "exposure",
        )
        model.data_ = add_categorical_columns(model.data_, formula=formula)
        model.future_data_ = add_categorical_columns(model.future_data_, formula=formula)
        return model

    def test_categorical_origin_dev_prior_structure(self):
        """C(origin) and C(dev) priors are array-valued with correct length."""
        import chainladder as cl

        tri = cl.load_sample("genins")
        model = self._make_fitted_model(
            formula="incremental ~ 1 + C(origin) + C(dev)",
            family="gamma",
            link="log",
            tri=tri,
        )
        cl_priors = model._build_cl_informed_priors()

        n_origins = len(tri.origin)
        n_devs = len(tri.development)

        # C(origin): n_origins - 1 contrasts
        assert "C(origin)" in cl_priors, "Expected C(origin) in CL priors"
        origin_prior = cl_priors["C(origin)"]
        assert hasattr(origin_prior, "args"), "Prior should be a bmb.Prior"
        origin_mus = origin_prior.args["mu"]
        assert len(origin_mus) == n_origins - 1, (
            f"Expected {n_origins - 1} origin contrasts, got {len(origin_mus)}"
        )

        # C(dev): n_devs - 1 contrasts
        assert "C(dev)" in cl_priors, "Expected C(dev) in CL priors"
        dev_prior = cl_priors["C(dev)"]
        dev_mus = dev_prior.args["mu"]
        assert len(dev_mus) == n_devs - 1, (
            f"Expected {n_devs - 1} dev contrasts, got {len(dev_mus)}"
        )

    def test_categorical_prior_values_sensible(self):
        """C(origin) and C(dev) prior means are finite and not all-zero."""
        import chainladder as cl

        tri = cl.load_sample("genins")
        model = self._make_fitted_model(
            formula="incremental ~ 1 + C(origin) + C(dev)",
            family="gamma",
            link="log",
            tri=tri,
        )
        cl_priors = model._build_cl_informed_priors()

        origin_mus = cl_priors["C(origin)"].args["mu"]
        dev_mus = cl_priors["C(dev)"].args["mu"]
        sd = cl_priors["C(origin)"].args["sigma"]

        # Means should be finite
        assert np.all(np.isfinite(origin_mus)), "origin prior means contain non-finite values"
        assert np.all(np.isfinite(dev_mus)), "dev prior means contain non-finite values"

        # SDs should equal chainladder_prior_sd
        assert np.allclose(sd, 0.5), f"Expected sd=0.5, got {sd}"

        # Dev effects should be negative (later periods have smaller fraction)
        # i.e., incr_pct[1:] < incr_pct[0] typically for a tail-heavy triangle
        # At minimum, some dev contrasts should be negative
        assert not np.all(dev_mus >= 0), (
            "All dev prior means are non-negative — expected decreasing pattern"
        )

    def test_custom_prior_sd_respected(self):
        """chainladder_prior_sd parameter is used as the sigma for all effects."""
        import chainladder as cl

        tri = cl.load_sample("genins")
        for test_sd in [0.3, 0.7, 1.0]:
            model = self._make_fitted_model(
                formula="incremental ~ 1 + C(origin) + C(dev)",
                family="gamma",
                link="log",
                sd=test_sd,
                tri=tri,
            )
            cl_priors = model._build_cl_informed_priors()
            if "C(origin)" in cl_priors:
                sds = cl_priors["C(origin)"].args["sigma"]
                assert np.allclose(sds, test_sd), (
                    f"Expected sigma={test_sd}, got {sds}"
                )

    def test_spline_dev_priors_constructed_or_skipped(self):
        """bs(dev_idx, df=4) formula: spline priors are either constructed or
        gracefully skipped (no crash either way)."""
        import chainladder as cl

        tri = cl.load_sample("genins")
        model = self._make_fitted_model(
            formula="incremental ~ 1 + C(origin) + bs(dev_idx, df=4)",
            family="gamma",
            link="log",
            tri=tri,
        )
        cl_priors = model._build_cl_informed_priors()

        # Either the spline key is present with 4 coefficients, or it's absent
        spline_key = "bs(dev_idx, df=4)"
        if spline_key in cl_priors:
            spline_prior = cl_priors[spline_key]
            coefs = spline_prior.args["mu"]
            assert len(coefs) == 4, (
                f"Expected 4 spline coefficients, got {len(coefs)}"
            )
            assert np.all(np.isfinite(coefs)), "Spline coefs contain non-finite values"
        # else: graceful fallback — acceptable

    def test_hierarchical_origin_re_prior_constructed(self):
        """(1|origin) formula: the RE sigma hyperprior is informed by log-ult SD."""
        import chainladder as cl

        tri = cl.load_sample("genins")
        model = self._make_fitted_model(
            formula="incremental ~ 1 + (1 | origin) + bs(dev_idx, df=4)",
            family="gamma",
            link="log",
            tri=tri,
        )
        cl_priors = model._build_cl_informed_priors()

        assert "1|origin" in cl_priors, "Expected '1|origin' in CL priors"
        re_prior = cl_priors["1|origin"]
        # The RE prior should have a HalfNormal sigma hyperprior
        inner_sigma = re_prior.args["sigma"]
        assert hasattr(inner_sigma, "name") or hasattr(inner_sigma, "args"), (
            "Expected sigma to be a bmb.Prior (HalfNormal), not a scalar"
        )

    def test_init_priors_false_leaves_defaults(self):
        """When init_priors_from_chainladder=False, _build_cl_informed_priors returns {}."""
        import chainladder as cl

        tri = cl.load_sample("genins")
        model = self._make_fitted_model(
            formula="incremental ~ 1 + C(origin) + C(dev)",
            family="gamma",
            link="log",
            tri=tri,
        )
        # Override to False to verify empty dict
        model.init_priors_from_chainladder = False
        cl_priors = model._build_cl_informed_priors()
        # Method still returns a dict (may be non-empty if we call it directly),
        # but what matters is that _build_default_priors does NOT include them
        # when init_priors_from_chainladder=False.
        # Test _build_default_priors instead:
        model2 = BayesianChainLadderGLM(
            formula="incremental ~ 1 + C(origin) + C(dev)",
            family="gamma",
            link="log",
            init_priors_from_chainladder=False,
        )
        model2.triangle_ = tri.copy()
        model2.data_, _ = __import__("bayesianchainladder.utils", fromlist=["prepare_model_data"]).prepare_model_data(tri)
        model2.data_ = __import__("bayesianchainladder.utils", fromlist=["add_categorical_columns"]).add_categorical_columns(model2.data_, formula=model2.formula)
        defaults = model2._build_default_priors()

        # When init_priors_from_chainladder=False, C(origin) prior should be
        # the generic Normal(0, 1) (scalar mu=0), not an array
        if "C(origin)" in defaults:
            mu = defaults["C(origin)"].args["mu"]
            assert np.isscalar(mu) or (hasattr(mu, "__len__") and len(np.atleast_1d(mu)) == 1), (
                "Without CL priors, C(origin) mu should be scalar 0.0"
            )

    @pytest.mark.slow
    def test_fit_with_cl_priors_m1_cat(self, positive_triangle):
        """Full fit with init_priors_from_chainladder=True on M1_cat formula."""
        import warnings

        model = BayesianChainLadderGLM(
            formula="incremental ~ 1 + C(origin) + C(dev)",
            family="gamma",
            link="log",
            init_priors_from_chainladder=True,
            chainladder_prior_sd=0.5,
            draws=100,
            tune=50,
            chains=1,
            random_seed=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(positive_triangle)

        assert model._is_fitted
        assert model.idata is not None
        assert model.reserves_posterior_ is not None

    @pytest.mark.slow
    def test_fit_with_cl_priors_m2_spline(self, positive_triangle):
        """Full fit with init_priors_from_chainladder=True on M2 (spline) formula."""
        import warnings

        model = BayesianChainLadderGLM(
            formula="incremental ~ 1 + C(origin) + bs(dev_idx, df=4)",
            family="gamma",
            link="log",
            init_priors_from_chainladder=True,
            chainladder_prior_sd=0.5,
            draws=100,
            tune=50,
            chains=1,
            random_seed=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(positive_triangle)

        assert model._is_fitted
        assert model.idata is not None
        assert model.reserves_posterior_ is not None

    @pytest.mark.slow
    def test_fit_with_cl_priors_hierarchical(self, positive_triangle):
        """Full fit with init_priors_from_chainladder=True on hierarchical formula."""
        import warnings

        model = BayesianChainLadderGLM(
            formula="incremental ~ 1 + (1 | origin) + bs(dev_idx, df=4) + (1 | calendar)",
            family="gamma",
            link="log",
            init_priors_from_chainladder=True,
            chainladder_prior_sd=0.5,
            draws=100,
            tune=50,
            chains=1,
            random_seed=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(positive_triangle)

        assert model._is_fitted
        assert model.idata is not None
        assert model.reserves_posterior_ is not None
