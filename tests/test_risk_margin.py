"""Tests for risk margin calculations."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from bayesianchainladder.risk_margin import (
    RiskMarginCalculator,
    RiskMarginResult,
    compute_tvar,
    compute_var,
)


class TestComputeTvar:
    """Tests for the compute_tvar function."""

    def test_tvar_basic(self):
        """Test basic TVaR calculation."""
        # Create a known distribution
        samples = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # 80% TVaR should be the mean of the top 20% = mean of [9, 10] = 9.5
        tvar = compute_tvar(samples, percentile=0.80)
        assert tvar == 9.5

    def test_tvar_uniform(self):
        """Test TVaR on uniform distribution."""
        np.random.seed(42)
        samples = np.random.uniform(0, 100, 10000)

        # 95% TVaR for uniform(0,100) is approximately (95+100)/2 = 97.5
        tvar = compute_tvar(samples, percentile=0.95)
        assert 96 < tvar < 99

    def test_tvar_normal(self):
        """Test TVaR on normal distribution."""
        np.random.seed(42)
        samples = np.random.normal(100, 20, 100000)

        # 97% TVaR for N(100, 20) should be around 100 + 2.4 * 20 = 148
        tvar = compute_tvar(samples, percentile=0.97)
        assert 140 < tvar < 160

    def test_tvar_default_percentile(self):
        """Test TVaR uses default percentile of 0.97."""
        samples = np.arange(1, 101)
        tvar = compute_tvar(samples)

        # Should use 97% percentile by default
        tvar_97 = compute_tvar(samples, percentile=0.97)
        assert tvar == tvar_97


class TestComputeVar:
    """Tests for the compute_var function."""

    def test_var_basic(self):
        """Test basic VaR calculation."""
        samples = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # 90% VaR should be at the 90th percentile
        var = compute_var(samples, percentile=0.90)
        assert var == 9.1  # numpy percentile interpolates

    def test_var_vs_percentile(self):
        """Test that VaR equals numpy percentile."""
        np.random.seed(42)
        samples = np.random.lognormal(10, 1, 10000)

        var = compute_var(samples, percentile=0.95)
        expected = np.percentile(samples, 95)

        assert var == expected


class TestRiskMarginResult:
    """Tests for RiskMarginResult dataclass."""

    def test_result_creation(self):
        """Test creating a RiskMarginResult."""
        result = RiskMarginResult(
            num_samples=1000,
            expected_ultimate=500000.0,
            best_estimate=450000.0,
            pred_mean=np.zeros((1000, 10)),
            pred_assets=np.zeros((1000, 10)),
            pred_capital=np.zeros((1000, 10)),
            capital_release=np.zeros((1000, 10)),
            risk_margin=np.random.normal(10000, 2000, 1000),
            risk_margin_pct=np.random.normal(10, 2, 1000),
            fixed_rate=0.04,
            risky_rate=0.10,
            tvar_percentile=0.97,
            time_horizon="ultimate",
        )

        assert result.num_samples == 1000
        assert result.expected_ultimate == 500000.0
        assert result.best_estimate == 450000.0
        assert result.fixed_rate == 0.04
        assert result.risky_rate == 0.10
        assert result.tvar_percentile == 0.97
        assert result.time_horizon == "ultimate"

    def test_result_summary(self):
        """Test RiskMarginResult.summary() method."""
        np.random.seed(42)
        risk_margin = np.random.normal(10000, 2000, 1000)

        result = RiskMarginResult(
            num_samples=1000,
            expected_ultimate=500000.0,
            best_estimate=450000.0,
            pred_mean=np.full((1000, 10), 500000),
            pred_assets=np.full((1000, 10), 520000),
            pred_capital=np.full((1000, 10), 20000),
            capital_release=np.zeros((1000, 10)),
            risk_margin=risk_margin,
            risk_margin_pct=100 * risk_margin / 20000,
            fixed_rate=0.04,
            risky_rate=0.10,
            tvar_percentile=0.97,
            time_horizon="ultimate",
        )

        summary = result.summary()

        assert isinstance(summary, pd.DataFrame)
        assert "Expected Ultimate" in summary.index
        assert "Best Estimate" in summary.index
        assert "Risk Margin (Mean)" in summary.index
        assert "Risk Margin (Std)" in summary.index
        assert "Risk Margin % (Mean)" in summary.index


class TestRiskMarginCalculatorInit:
    """Tests for RiskMarginCalculator initialization."""

    def test_init_requires_fitted_model(self, mocker):
        """Test that initialization fails for unfitted model."""

        # Create a mock unfitted model
        class MockUnfittedModel:
            def _check_is_fitted(self):
                raise ValueError("Model has not been fitted.")

        mock_model = MockUnfittedModel()

        with pytest.raises(ValueError, match="Model has not been fitted"):
            RiskMarginCalculator(mock_model)


class TestRiskMarginCalculatorWithMockModel:
    """Tests for RiskMarginCalculator with mocked model."""

    @pytest.fixture
    def mock_csr_model(self, mocker):
        """Create a mock BayesianCSR model."""

        # Create mock posterior data
        n_chains, n_draws = 2, 500
        n_origins, n_dev = 10, 10

        # Create mock posterior arrays
        posterior_data = {
            "alpha": (["chain", "draw", "origin"],
                     np.random.normal(0, 0.1, (n_chains, n_draws, n_origins))),
            "beta": (["chain", "draw", "dev"],
                    np.linspace(0, -2, n_dev)[np.newaxis, np.newaxis, :].repeat(n_chains, axis=0).repeat(n_draws, axis=1)),
            "speedup": (["chain", "draw", "origin"],
                       np.ones((n_chains, n_draws, n_origins))),
            "logelr": (["chain", "draw"],
                      np.random.normal(-0.4, 0.1, (n_chains, n_draws))),
            "sig": (["chain", "draw", "dev"],
                   np.linspace(0.5, 0.1, n_dev)[np.newaxis, np.newaxis, :].repeat(n_chains, axis=0).repeat(n_draws, axis=1)),
            "gamma": (["chain", "draw"],
                     np.random.normal(0, 0.02, (n_chains, n_draws))),
        }

        posterior = xr.Dataset(
            posterior_data,
            coords={
                "chain": np.arange(n_chains),
                "draw": np.arange(n_draws),
                "origin": np.arange(1, n_origins + 1),
                "dev": np.arange(1, n_dev + 1),
            }
        )

        # Create mock InferenceData
        mock_idata = mocker.Mock()
        mock_idata.posterior = posterior

        # Create mock data
        data_rows = []
        premium = np.array([10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000])
        for w in range(1, n_origins + 1):
            for d in range(1, min(n_dev + 1, n_origins - w + 2)):
                cum_loss = premium[w-1] * 0.6 * (1 - np.exp(-0.5 * d))  # Simulated development
                data_rows.append({
                    "origin": w,
                    "dev": d,
                    "cumulative": cum_loss,
                    "logloss": np.log(max(cum_loss, 1)),
                    "premium": premium[w-1],
                    "logprem": np.log(premium[w-1]),
                })

        data_df = pd.DataFrame(data_rows)

        # Create mock model
        mock_model = mocker.Mock()
        mock_model.idata = mock_idata
        mock_model.data_ = data_df
        mock_model.future_data_ = pd.DataFrame()  # Empty for simplicity
        mock_model.reserves_posterior_ = None
        mock_model._check_is_fitted = mocker.Mock()

        # Add model coords
        mock_model.model_ = mocker.Mock()
        mock_model.model_.coords = {
            "origin": list(range(1, n_origins + 1)),
            "dev": list(range(1, n_dev + 1)),
        }

        return mock_model

    def test_setup_model_info(self, mock_csr_model):
        """Test that model info is correctly extracted."""
        calc = RiskMarginCalculator(mock_csr_model)

        assert calc.n_origins == 10
        assert calc.n_dev == 10
        assert calc.num_samples == 1000  # 2 chains * 500 draws

    def test_extract_csr_parameters(self, mock_csr_model):
        """Test parameter extraction."""
        calc = RiskMarginCalculator(mock_csr_model)
        params = calc._extract_csr_parameters()

        assert "alpha" in params
        assert "beta" in params
        assert "speedup" in params
        assert "logelr" in params
        assert "sig" in params
        assert "gamma" in params

        # Check shapes
        assert params["alpha"].shape == (1000, 10)
        assert params["beta"].shape == (1000, 10)
        assert params["logelr"].shape == (1000,)

    def test_get_paid_triangle(self, mock_csr_model):
        """Test paid triangle extraction."""
        calc = RiskMarginCalculator(mock_csr_model)
        trpaid = calc._get_paid_triangle()

        assert trpaid.shape == (10, 10)
        # First origin should have all 10 dev periods
        # Last origin should only have first dev period

    def test_compute_tvar_method(self, mock_csr_model):
        """Test the _compute_tvar method."""
        calc = RiskMarginCalculator(mock_csr_model, tvar_percentile=0.97)

        samples = np.arange(1, 101)
        tvar = calc._compute_tvar(samples)

        # Should be mean of top 3%
        expected = np.mean([98, 99, 100])
        assert tvar == expected


class TestRiskMarginPlots:
    """Tests for risk margin plotting functions."""

    @pytest.fixture
    def mock_result(self):
        """Create a mock RiskMarginResult for plotting tests."""
        np.random.seed(42)
        n_samples = 100
        n_periods = 10

        return RiskMarginResult(
            num_samples=n_samples,
            expected_ultimate=500000.0,
            best_estimate=450000.0,
            pred_mean=np.random.normal(500000, 20000, (n_samples, n_periods)),
            pred_assets=np.random.normal(520000, 25000, (n_samples, n_periods)),
            pred_capital=np.random.normal(20000, 5000, (n_samples, n_periods)),
            capital_release=np.random.normal(2000, 1000, (n_samples, n_periods)),
            risk_margin=np.random.normal(10000, 2000, n_samples),
            risk_margin_pct=np.random.normal(50, 10, n_samples),
            fixed_rate=0.04,
            risky_rate=0.10,
            tvar_percentile=0.97,
            time_horizon="ultimate",
        )

    def test_plot_ultimate_loss_paths_import(self):
        """Test that plot functions can be imported."""
        from bayesianchainladder.plots import (
            plot_ultimate_loss_paths,
            plot_required_capital_paths,
            plot_capital_release_paths,
            plot_risk_margin_distribution,
            plot_risk_margin_pct_distribution,
            plot_risk_margin_summary,
            plot_capital_vs_estimate,
        )

        # All imports should succeed
        assert plot_ultimate_loss_paths is not None
        assert plot_required_capital_paths is not None
        assert plot_capital_release_paths is not None
        assert plot_risk_margin_distribution is not None
        assert plot_risk_margin_pct_distribution is not None
        assert plot_risk_margin_summary is not None
        assert plot_capital_vs_estimate is not None


class TestComputeTvarVsVar:
    """Tests comparing TVaR and VaR."""

    def test_tvar_greater_than_var(self):
        """TVaR should always be >= VaR at the same percentile."""
        np.random.seed(42)
        samples = np.random.lognormal(10, 1, 10000)

        for percentile in [0.90, 0.95, 0.97, 0.99]:
            tvar = compute_tvar(samples, percentile)
            var = compute_var(samples, percentile)
            assert tvar >= var, f"TVaR {tvar} should be >= VaR {var} at {percentile}"

    def test_tvar_equals_var_for_uniform_tail(self):
        """For a constant tail, TVaR should equal the constant."""
        # All values in tail are 100
        samples = np.concatenate([np.random.uniform(0, 50, 9000), np.full(1000, 100)])

        # 90% TVaR should be exactly 100 (mean of the constant tail)
        tvar = compute_tvar(samples, percentile=0.90)
        assert abs(tvar - 100) < 0.1
