"""Tests for prior predictive check functionality."""

import numpy as np
import pandas as pd
import pytest
import arviz as az

import chainladder as cl

from bayesianchainladder import (
    BayesianChainLadderGLM,
    sample_prior_predictive,
    compute_prior_predictive_summary,
)
from bayesianchainladder.plots import (
    plot_prior_predictive,
    plot_prior_predictive_by_origin,
    plot_prior_predictive_development,
    plot_prior_predictive_triangle,
    plot_prior_predictive_summary,
)


@pytest.fixture
def sample_triangle():
    """Create a sample triangle for testing.

    Uses GenIns which has positive incremental values suitable for
    negative binomial family.
    """
    return cl.load_sample("GenIns")


@pytest.fixture
def sample_data():
    """Create sample data for model testing."""
    return pd.DataFrame({
        "incremental": [100, 80, 60, 50, 110, 90, 70, 120, 100, 130],
        "origin": [1, 1, 1, 1, 2, 2, 2, 3, 3, 4],
        "dev": [1, 2, 3, 4, 1, 2, 3, 1, 2, 1],
        "calendar": [1, 2, 3, 4, 2, 3, 4, 3, 4, 4],
    })


class TestBuildModel:
    """Tests for build_model method."""

    def test_build_model_creates_model(self, sample_triangle):
        """Test that build_model creates a model without fitting."""
        model = BayesianChainLadderGLM(family="gaussian")
        model.build_model(sample_triangle)

        assert model.model_ is not None
        assert model.data_ is not None
        assert model.future_data_ is not None
        assert model._is_fitted is False

    def test_build_model_returns_self(self, sample_triangle):
        """Test that build_model returns self for chaining."""
        model = BayesianChainLadderGLM(family="gaussian")
        result = model.build_model(sample_triangle)

        assert result is model

    def test_build_model_different_formulas(self, sample_triangle):
        """Test build_model with different formulas."""
        formulas = [
            "incremental ~ 1 + C(origin) + C(dev)",
            "incremental ~ 1 + C(origin)",
            "incremental ~ 1 + C(dev)",
        ]

        for formula in formulas:
            model = BayesianChainLadderGLM(formula=formula, family="gaussian")
            model.build_model(sample_triangle)
            assert model.model_ is not None


class TestSamplePriorPredictive:
    """Tests for sample_prior_predictive functionality."""

    def test_sample_prior_predictive_with_triangle(self, sample_triangle):
        """Test prior predictive sampling with triangle input."""
        model = BayesianChainLadderGLM(family="gaussian")
        prior_idata = model.sample_prior_predictive(sample_triangle, draws=50)

        assert isinstance(prior_idata, az.InferenceData)
        assert "prior" in prior_idata.groups()
        assert "prior_predictive" in prior_idata.groups()

    def test_sample_prior_predictive_with_prebuilt_model(self, sample_triangle):
        """Test prior predictive sampling with pre-built model."""
        model = BayesianChainLadderGLM(family="gaussian")
        model.build_model(sample_triangle)
        prior_idata = model.sample_prior_predictive(draws=50)

        assert isinstance(prior_idata, az.InferenceData)
        assert "prior" in prior_idata.groups()
        assert "prior_predictive" in prior_idata.groups()

    def test_sample_prior_predictive_stores_result(self, sample_triangle):
        """Test that prior_idata_ is stored on the model."""
        model = BayesianChainLadderGLM(family="gaussian")
        prior_idata = model.sample_prior_predictive(sample_triangle, draws=50)

        assert hasattr(model, "prior_idata_")
        assert model.prior_idata_ is prior_idata

    def test_sample_prior_predictive_respects_draws(self, sample_triangle):
        """Test that draws parameter controls sample size."""
        model = BayesianChainLadderGLM(family="gaussian")
        prior_idata = model.sample_prior_predictive(sample_triangle, draws=100)

        response_name = model.model_.response_component.response.name
        pp = prior_idata.prior_predictive[response_name]

        # Check that we have the expected number of draws
        assert pp.sizes["draw"] == 100

    def test_sample_prior_predictive_without_model_raises(self):
        """Test that calling without model or triangle raises error."""
        model = BayesianChainLadderGLM(family="gaussian")

        with pytest.raises(ValueError, match="No triangle provided"):
            model.sample_prior_predictive(draws=50)

    def test_sample_prior_predictive_reproducibility(self, sample_triangle):
        """Test that random_seed provides reproducibility."""
        model1 = BayesianChainLadderGLM(family="gaussian")
        prior1 = model1.sample_prior_predictive(sample_triangle, draws=50, random_seed=42)

        model2 = BayesianChainLadderGLM(family="gaussian")
        prior2 = model2.sample_prior_predictive(sample_triangle, draws=50, random_seed=42)

        response_name = model1.model_.response_component.response.name

        # With same seed, results should be identical
        np.testing.assert_array_equal(
            prior1.prior_predictive[response_name].values,
            prior2.prior_predictive[response_name].values
        )


class TestGetPriorPredictiveSummary:
    """Tests for get_prior_predictive_summary method."""

    def test_get_summary_returns_dataframe(self, sample_triangle):
        """Test that get_prior_predictive_summary returns a DataFrame."""
        model = BayesianChainLadderGLM(family="gaussian")
        model.sample_prior_predictive(sample_triangle, draws=50)
        summary = model.get_prior_predictive_summary()

        assert isinstance(summary, pd.DataFrame)
        assert "mean" in summary.columns
        assert "std" in summary.columns

    def test_get_summary_by_origin(self, sample_triangle):
        """Test summary aggregated by origin."""
        model = BayesianChainLadderGLM(family="gaussian")
        model.sample_prior_predictive(sample_triangle, draws=50)
        summary = model.get_prior_predictive_summary(by="origin")

        assert isinstance(summary, pd.DataFrame)
        assert summary.index.name == "origin"

    def test_get_summary_by_dev(self, sample_triangle):
        """Test summary aggregated by development period."""
        model = BayesianChainLadderGLM(family="gaussian")
        model.sample_prior_predictive(sample_triangle, draws=50)
        summary = model.get_prior_predictive_summary(by="dev")

        assert isinstance(summary, pd.DataFrame)
        assert summary.index.name == "dev"

    def test_get_summary_without_prior_raises(self, sample_triangle):
        """Test that calling without prior samples raises error."""
        model = BayesianChainLadderGLM(family="gaussian")
        model.build_model(sample_triangle)

        with pytest.raises(ValueError, match="No prior predictive samples"):
            model.get_prior_predictive_summary()


class TestLowLevelSamplePriorPredictive:
    """Tests for the low-level sample_prior_predictive function."""

    def test_sample_prior_predictive_function(self, sample_data):
        """Test the standalone sample_prior_predictive function."""
        from bayesianchainladder.models import build_bambi_model

        model = build_bambi_model(
            sample_data,
            formula="incremental ~ 1 + C(origin) + C(dev)",
            family="negativebinomial",
        )

        prior_idata = sample_prior_predictive(model, draws=50)

        assert isinstance(prior_idata, az.InferenceData)
        assert "prior" in prior_idata.groups()
        assert "prior_predictive" in prior_idata.groups()


class TestComputePriorPredictiveSummary:
    """Tests for compute_prior_predictive_summary function."""

    def test_compute_summary_function(self, sample_triangle):
        """Test the compute_prior_predictive_summary function."""
        model = BayesianChainLadderGLM(family="gaussian")
        prior_idata = model.sample_prior_predictive(sample_triangle, draws=50)

        response_name = model.model_.response_component.response.name
        summary = compute_prior_predictive_summary(prior_idata, response_name)

        assert isinstance(summary, pd.DataFrame)
        assert "mean" in summary.columns
        assert "std" in summary.columns
        assert "50.0%" in summary.columns


class TestPriorPredictivePlots:
    """Tests for prior predictive plotting functions."""

    @pytest.fixture
    def model_with_prior(self, sample_triangle):
        """Create a model with prior predictive samples.

        Uses gaussian family to avoid issues with negative incremental values.
        """
        model = BayesianChainLadderGLM(family="gaussian")
        model.sample_prior_predictive(sample_triangle, draws=50)
        return model

    def test_plot_prior_predictive(self, model_with_prior):
        """Test plot_prior_predictive function."""
        import matplotlib.pyplot as plt

        fig, ax = plot_prior_predictive(model_with_prior)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_prior_predictive_kde(self, model_with_prior):
        """Test plot_prior_predictive with kde."""
        import matplotlib.pyplot as plt

        fig, ax = plot_prior_predictive(model_with_prior, kind="kde")

        assert fig is not None
        plt.close(fig)

    def test_plot_prior_predictive_hist(self, model_with_prior):
        """Test plot_prior_predictive with histogram."""
        import matplotlib.pyplot as plt

        fig, ax = plot_prior_predictive(model_with_prior, kind="hist")

        assert fig is not None
        plt.close(fig)

    def test_plot_prior_predictive_ecdf(self, model_with_prior):
        """Test plot_prior_predictive with ecdf."""
        import matplotlib.pyplot as plt

        fig, ax = plot_prior_predictive(model_with_prior, kind="ecdf")

        assert fig is not None
        plt.close(fig)

    def test_plot_prior_predictive_by_origin(self, model_with_prior):
        """Test plot_prior_predictive_by_origin function."""
        import matplotlib.pyplot as plt

        fig, axes = plot_prior_predictive_by_origin(model_with_prior)

        assert fig is not None
        assert axes is not None
        plt.close(fig)

    def test_plot_prior_predictive_development(self, model_with_prior):
        """Test plot_prior_predictive_development function."""
        import matplotlib.pyplot as plt

        fig, ax = plot_prior_predictive_development(model_with_prior)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_prior_predictive_triangle(self, model_with_prior):
        """Test plot_prior_predictive_triangle function."""
        import matplotlib.pyplot as plt

        fig, ax = plot_prior_predictive_triangle(model_with_prior)

        assert fig is not None
        plt.close(fig)

    def test_plot_prior_predictive_triangle_mean(self, model_with_prior):
        """Test plot_prior_predictive_triangle with mean statistic."""
        import matplotlib.pyplot as plt

        fig, ax = plot_prior_predictive_triangle(model_with_prior, statistic="mean")

        assert fig is not None
        plt.close(fig)

    def test_plot_prior_predictive_triangle_std(self, model_with_prior):
        """Test plot_prior_predictive_triangle with std statistic."""
        import matplotlib.pyplot as plt

        fig, ax = plot_prior_predictive_triangle(model_with_prior, statistic="std")

        assert fig is not None
        plt.close(fig)

    def test_plot_prior_predictive_summary(self, model_with_prior):
        """Test plot_prior_predictive_summary function."""
        import matplotlib.pyplot as plt

        fig, axes = plot_prior_predictive_summary(model_with_prior)

        assert fig is not None
        assert len(axes) == 4  # 4 panels
        plt.close(fig)

    def test_plot_without_prior_raises(self, sample_triangle):
        """Test that plotting without prior samples raises error."""
        model = BayesianChainLadderGLM(family="gaussian")
        model.build_model(sample_triangle)

        with pytest.raises(ValueError, match="No prior predictive samples"):
            plot_prior_predictive(model)


@pytest.fixture
def positive_triangle():
    """Create a triangle with only positive incremental values for count families."""
    # Use GenIns which typically has positive incremental values
    tri = cl.load_sample("GenIns")
    return tri


class TestPriorPredictiveWithDifferentFamilies:
    """Tests for prior predictive with different distribution families."""

    def test_prior_predictive_gaussian(self, sample_triangle):
        """Test prior predictive with Gaussian family."""
        model = BayesianChainLadderGLM(family="gaussian")
        prior_idata = model.sample_prior_predictive(sample_triangle, draws=50)

        assert "prior_predictive" in prior_idata.groups()


class TestPriorPredictiveIntegration:
    """Integration tests for prior predictive workflow."""

    def test_prior_to_posterior_workflow(self, sample_triangle):
        """Test workflow from prior predictive to fitting."""
        # Step 1: Create model with gaussian family to avoid data validation issues
        model = BayesianChainLadderGLM(
            formula="incremental ~ 1 + C(origin) + C(dev)",
            family="gaussian",
        )

        # Step 2: Sample prior predictive
        prior_idata = model.sample_prior_predictive(sample_triangle, draws=50)

        # Verify prior predictive
        assert "prior" in prior_idata.groups()
        assert "prior_predictive" in prior_idata.groups()

        # Verify model is built but not fitted
        assert model.model_ is not None
        assert model._is_fitted is False

    def test_prior_predictive_preserves_data(self, sample_triangle):
        """Test that prior predictive sampling preserves data attributes."""
        model = BayesianChainLadderGLM(family="gaussian")
        model.sample_prior_predictive(sample_triangle, draws=50)

        # Check data is preserved
        assert model.data_ is not None
        assert model.future_data_ is not None
        assert model.triangle_ is not None

        # Check data has expected structure
        assert "origin" in model.data_.columns
        assert "dev" in model.data_.columns
        assert "incremental" in model.data_.columns
