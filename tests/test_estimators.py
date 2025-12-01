"""Tests for estimator classes."""

import numpy as np
import pandas as pd
import pytest

import chainladder as cl

from bayesianchainladder.estimators import BayesianChainLadderGLM


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
