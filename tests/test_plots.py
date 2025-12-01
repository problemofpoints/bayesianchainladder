"""Tests for plotting functions."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Use non-interactive backend for testing
matplotlib.use("Agg")

import chainladder as cl

from bayesianchainladder.estimators import BayesianChainLadderGLM
from bayesianchainladder.plots import (
    create_summary_table,
    plot_actual_vs_fitted,
    plot_development_pattern,
    plot_energy,
    plot_forest,
    plot_heatmap_residuals,
    plot_posterior,
    plot_ppc,
    plot_rank,
    plot_reserve_distribution,
    plot_residuals,
    plot_trace,
)


@pytest.fixture
def small_triangle():
    """Create a small triangle for faster testing."""
    # Use the RAA sample triangle (small and well-behaved)
    return cl.load_sample("raa")


@pytest.fixture
def fitted_model(small_triangle):
    """Create a fitted model for testing plots."""
    model = BayesianChainLadderGLM(
        family="gaussian",  # Use Gaussian for RAA triangle which may have negative values
        draws=50,
        tune=25,
        chains=1,
        random_seed=42,
    )
    model.fit(small_triangle)
    return model


class TestPlotTrace:
    """Tests for plot_trace function."""

    def test_unfitted_model_raises(self, small_triangle):
        """Test that unfitted model raises error."""
        model = BayesianChainLadderGLM()

        with pytest.raises(ValueError, match="fitted"):
            plot_trace(model)

    @pytest.mark.slow
    def test_returns_figure_axes(self, fitted_model):
        """Test that function returns figure and axes."""
        fig, ax = plot_trace(fitted_model)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotPosterior:
    """Tests for plot_posterior function."""

    @pytest.mark.slow
    def test_returns_figure_axes(self, fitted_model):
        """Test that function returns figure and axes."""
        fig, ax = plot_posterior(fitted_model)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotPPC:
    """Tests for plot_ppc function."""

    @pytest.mark.slow
    def test_returns_figure_axes(self, fitted_model):
        """Test that function returns figure and axes."""
        fig, ax = plot_ppc(fitted_model)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotEnergy:
    """Tests for plot_energy function."""

    @pytest.mark.slow
    def test_returns_figure_axes(self, fitted_model):
        """Test that function returns figure and axes."""
        fig, ax = plot_energy(fitted_model)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotRank:
    """Tests for plot_rank function."""

    @pytest.mark.slow
    def test_returns_figure_axes(self, fitted_model):
        """Test that function returns figure and axes."""
        fig, ax = plot_rank(fitted_model)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotForest:
    """Tests for plot_forest function."""

    @pytest.mark.slow
    def test_returns_figure_axes(self, fitted_model):
        """Test that function returns figure and axes."""
        fig, ax = plot_forest(fitted_model)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotReserveDistribution:
    """Tests for plot_reserve_distribution function."""

    @pytest.mark.slow
    def test_by_origin(self, fitted_model):
        """Test plot by origin."""
        fig, ax = plot_reserve_distribution(fitted_model, by="origin")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    @pytest.mark.slow
    def test_by_total(self, fitted_model):
        """Test plot by total."""
        fig, ax = plot_reserve_distribution(fitted_model, by="total")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    @pytest.mark.slow
    def test_kde_kind(self, fitted_model):
        """Test KDE plot kind."""
        fig, ax = plot_reserve_distribution(fitted_model, kind="kde")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    @pytest.mark.slow
    def test_hist_kind(self, fitted_model):
        """Test histogram plot kind."""
        fig, ax = plot_reserve_distribution(fitted_model, kind="hist")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotResiduals:
    """Tests for plot_residuals function."""

    @pytest.mark.slow
    def test_by_origin(self, fitted_model):
        """Test residual plot by origin."""
        fig, ax = plot_residuals(fitted_model, by="origin")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    @pytest.mark.slow
    def test_by_dev(self, fitted_model):
        """Test residual plot by development."""
        fig, ax = plot_residuals(fitted_model, by="dev")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    @pytest.mark.slow
    def test_by_calendar(self, fitted_model):
        """Test residual plot by calendar."""
        fig, ax = plot_residuals(fitted_model, by="calendar")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    @pytest.mark.slow
    def test_by_fitted(self, fitted_model):
        """Test residual plot by fitted values."""
        fig, ax = plot_residuals(fitted_model, by="fitted")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotActualVsFitted:
    """Tests for plot_actual_vs_fitted function."""

    @pytest.mark.slow
    def test_returns_figure_axes(self, fitted_model):
        """Test that function returns figure and axes."""
        fig, ax = plot_actual_vs_fitted(fitted_model)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    @pytest.mark.slow
    def test_log_scale(self, fitted_model):
        """Test log scale option."""
        fig, ax = plot_actual_vs_fitted(fitted_model, log_scale=True)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotHeatmapResiduals:
    """Tests for plot_heatmap_residuals function."""

    @pytest.mark.slow
    def test_returns_figure_axes(self, fitted_model):
        """Test that function returns figure and axes."""
        fig, ax = plot_heatmap_residuals(fitted_model)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestCreateSummaryTable:
    """Tests for create_summary_table function."""

    @pytest.mark.slow
    def test_dataframe_format(self, fitted_model):
        """Test DataFrame output format."""
        result = create_summary_table(fitted_model, format="dataframe")

        assert isinstance(result, pd.DataFrame)

    @pytest.mark.slow
    def test_latex_format(self, fitted_model):
        """Test LaTeX output format."""
        result = create_summary_table(fitted_model, format="latex")

        assert isinstance(result, str)
        assert "tabular" in result or "\\\\" in result

    @pytest.mark.slow
    def test_html_format(self, fitted_model):
        """Test HTML output format."""
        result = create_summary_table(fitted_model, format="html")

        assert isinstance(result, str)
        assert "<table" in result

    @pytest.mark.slow
    def test_invalid_format_raises(self, fitted_model):
        """Test that invalid format raises error."""
        with pytest.raises(ValueError, match="Unknown format"):
            create_summary_table(fitted_model, format="invalid")
