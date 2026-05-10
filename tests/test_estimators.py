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
