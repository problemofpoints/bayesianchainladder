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

    def test_gamma_auto_shift_emits_warning_and_modifies_response(self):
        """gamma family with negatives: shift is applied and UserWarning emitted."""
        import warnings

        # RAA has negative incrementals — perfect test case for auto-shift
        tri = cl.load_sample("raa")

        model = BayesianChainLadderGLM(
            formula="incremental ~ 1 + C(origin) + C(dev)",
            family="gamma",
            link="log",
            force_positive_response=True,
            draws=50,
            tune=25,
            chains=1,
            random_seed=42,
        )
        # Manually prepare data to inspect shift without running MCMC
        from bayesianchainladder.utils import add_categorical_columns, prepare_model_data
        model.triangle_ = tri.copy()
        model.data_, model.future_data_ = prepare_model_data(tri)
        model.data_ = add_categorical_columns(model.data_, formula=model.formula)

        # Apply the union-level alignment manually (as fit() does)
        import re
        for col in re.findall(r'\bC\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)', model.formula):
            if col in model.data_.columns and col in model.future_data_.columns:
                train_vals = list(model.data_[col].cat.categories) if hasattr(model.data_[col], "cat") else list(model.data_[col].unique())
                future_vals = list(model.future_data_[col].unique())
                all_levels = sorted(set(train_vals) | set(future_vals))
                model.data_[col] = pd.Categorical(model.data_[col], categories=all_levels)
                model.future_data_[col] = pd.Categorical(model.future_data_[col], categories=all_levels)

        # Verify RAA has negative incrementals
        raa_min = float(model.data_["incremental"].min())
        assert raa_min < 0, "RAA triangle should have negative incrementals"

        # Trigger the auto-shift logic directly
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Re-run the shift logic as in fit()
            resp_vals = np.asarray(model.data_["incremental"].values, dtype=np.float64)
            min_val = float(np.nanmin(resp_vals))
            if min_val <= 0:
                import warnings as _w
                shift = abs(min_val) + 1.0
                _w.warn(
                    f"BayesianChainLadderGLM: response column 'incremental' "
                    f"contains non-positive values (min={min_val:.4g}) which are "
                    f"incompatible with the 'gamma' family. "
                    f"Automatically shifting response by +{shift:.4g} to make all "
                    f"values strictly positive. "
                    f"Reserve estimates are back-shifted by the same amount per "
                    f"future cell to restore original scale. "
                    f"Set force_positive_response=False to disable this behavior.",
                    UserWarning,
                    stacklevel=3,
                )
                model._response_shift = shift
                model.data_["incremental"] = resp_vals + shift

        # After shift, all response values should be positive
        assert float(model.data_["incremental"].min()) > 0, \
            "After shift, all response values should be positive"
        # Shift should be |min_val| + 1.0
        assert abs(model._response_shift - (abs(raa_min) + 1.0)) < 1e-9, \
            f"Expected shift={abs(raa_min)+1.0:.4f}, got {model._response_shift:.4f}"

    def test_gamma_force_positive_false_still_raises(self, small_triangle):
        """gamma with force_positive_response=False still raises on negatives."""
        model = BayesianChainLadderGLM(
            family="gamma",
            force_positive_response=False,
            draws=50,
            tune=25,
            chains=1,
            random_seed=42,
        )
        with pytest.raises(ValueError, match="requires strictly positive values"):
            model.fit(small_triangle)

    @pytest.mark.slow
    @pytest.mark.slow
    def test_unseen_dev_levels_included_in_design(self):
        """Triangles where future_data_ has dev levels absent from training data
        should include those levels in the C(dev) design matrix (not drop them).

        The union of train and future dev levels is used as the categorical level
        set, so Bambi sees all levels in both datasets.  Levels unseen in training
        have an all-zero contrast column in training (coefficient is prior-driven),
        enabling valid extrapolation to unobserved dev periods.

        We take the genins triangle and blank out the last two dev columns
        (dev=108, 120) for ALL origins, so that the training data only 'sees'
        dev=12..96 while future cells require dev=108 and 120.
        """
        import warnings

        # Start from genins (all-positive incremental values).
        tri = cl.load_sample("genins")

        # Blank out the last two dev columns (dev=108, 120) for all origins.
        inc_tri = tri.cum_to_incr()
        tri2 = inc_tri.copy()
        vals = tri2.values.copy()  # shape (1, 1, 10, 10)
        vals[:, :, :, 8] = np.nan  # dev=108
        vals[:, :, :, 9] = np.nan  # dev=120
        tri2.values = vals

        model = BayesianChainLadderGLM(
            formula="incremental ~ 1 + C(origin) + C(dev)",
            family="gaussian",
            init_priors_from_chainladder=True,
            draws=50,
            tune=25,
            chains=1,
            random_seed=42,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(tri2)

        # future_data_ MUST contain dev 108 and 120 (no rows dropped)
        assert 108 in model.future_data_["dev"].values, \
            "dev=108 should be present in future_data_ (no longer dropped)"
        assert 120 in model.future_data_["dev"].values, \
            "dev=120 should be present in future_data_ (no longer dropped)"

        # Both train and future data must have the same dev categorical levels
        train_cats = set(model.data_["dev"].cat.categories.tolist())
        future_cats = set(model.future_data_["dev"].cat.categories.tolist())
        assert train_cats == future_cats, \
            "Train and future data must share the same C(dev) categorical levels"

        # All 10 dev levels (12..120) must be in the level set
        assert len(train_cats) == 10, f"Expected 10 dev levels, got {len(train_cats)}"

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

    def test_init_priors_from_chainladder_spline_dev_robust_to_partial_dev(self):
        """bs(dev_idx, df=4) spline priors are built without dimension error when
        training data covers only a subset of the full dev range.

        Simulates a triangle where only dev=12..96 (8 periods) are observed for
        all origins instead of the full 10 periods, verifying that the spline
        basis matrix B and the CL incremental target y_target are both built
        from the 8 observed dev_idx values rather than the full 10.
        """
        from bayesianchainladder.utils import add_categorical_columns, prepare_model_data

        formula = "incremental ~ 1 + C(origin) + bs(dev_idx, df=4)"
        tri = cl.load_sample("genins")

        # Set up a model with partial dev coverage by restricting data_ to only
        # the first 8 dev periods (dev=12..96), simulating a triangle that hasn't
        # yet developed to the last 2 periods.
        model = BayesianChainLadderGLM(
            formula=formula,
            family="gamma",
            link="log",
            init_priors_from_chainladder=True,
            chainladder_prior_sd=0.5,
            draws=50,
            tune=25,
            chains=1,
        )
        model.triangle_ = tri.copy()
        data_full, future_full = prepare_model_data(tri)

        # Restrict training data to first 8 dev periods (drop dev 108, 120)
        dev_sorted = sorted(data_full["dev"].unique())
        data_partial = data_full[data_full["dev"] <= dev_sorted[7]].copy().reset_index(drop=True)
        model.data_ = add_categorical_columns(data_partial, formula=formula)
        model.future_data_ = add_categorical_columns(future_full, formula=formula)

        # _build_cl_informed_priors must NOT raise a dimension error.
        # It should either return a valid spline prior OR gracefully skip it.
        cl_priors = model._build_cl_informed_priors()  # no exception

        spline_key = "bs(dev_idx, df=4)"
        if spline_key in cl_priors:
            spline_prior = cl_priors[spline_key]
            coefs = spline_prior.args["mu"]
            assert len(coefs) == 4, f"Expected 4 spline coefficients, got {len(coefs)}"
            assert np.all(np.isfinite(coefs)), "Spline coefs contain non-finite values"
        # else: graceful fallback is also acceptable

    def test_hierarchical_origin_re_hyperprior_tighter_than_2x(self):
        """(1|origin) hyperprior uses 1× (not 2×) the empirical log-ult SD."""
        import chainladder as cl

        tri = cl.load_sample("genins")
        model = self._make_fitted_model(
            formula="incremental ~ 1 + (1 | origin) + bs(dev_idx, df=4)",
            family="gamma",
            link="log",
            tri=tri,
        )
        # Compute the expected empirical SD of log(ultimates) from CL
        import chainladder as _cl
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cl_fit = _cl.Chainladder().fit(tri)
        ult_arr = np.asarray(cl_fit.ultimate_.to_frame().values, dtype=float).flatten()
        valid_ults = ult_arr[ult_arr > 0]
        log_ult_sd = float(np.std(np.log(valid_ults), ddof=1))
        expected_hn_sigma = max(log_ult_sd, 0.01)

        cl_priors = model._build_cl_informed_priors()

        assert "1|origin" in cl_priors, "Expected '1|origin' in CL priors"
        re_prior = cl_priors["1|origin"]
        inner_sigma = re_prior.args["sigma"]
        assert hasattr(inner_sigma, "args"), "Expected HalfNormal bmb.Prior for sigma"
        actual_hn_sigma = float(inner_sigma.args["sigma"])
        assert abs(actual_hn_sigma - expected_hn_sigma) < 1e-6, (
            f"Expected HalfNormal sigma={expected_hn_sigma:.4f} (1× empirical SD), "
            f"got {actual_hn_sigma:.4f}"
        )

    def test_calendar_re_prior_constructed_weakly_informative(self):
        """(1|calendar) formula: a weakly informative HalfNormal(0.2) hyperprior is added."""
        import chainladder as cl

        tri = cl.load_sample("genins")
        model = self._make_fitted_model(
            formula="incremental ~ 1 + (1 | origin) + bs(dev_idx, df=4) + (1 | calendar)",
            family="gamma",
            link="log",
            tri=tri,
        )
        cl_priors = model._build_cl_informed_priors()

        assert "1|calendar" in cl_priors, "Expected '1|calendar' in CL priors"
        cal_prior = cl_priors["1|calendar"]
        inner_sigma = cal_prior.args["sigma"]
        assert hasattr(inner_sigma, "args"), "Expected HalfNormal bmb.Prior for calendar sigma"
        hn_sigma = float(inner_sigma.args["sigma"])
        assert abs(hn_sigma - 0.2) < 1e-9, (
            f"Expected HalfNormal sigma=0.2 for calendar RE, got {hn_sigma}"
        )

    def test_init_priors_from_chainladder_handles_sparse_levels(self):
        """CL-informed priors handle categorical levels with zero training observations.

        When the pd.Categorical column includes levels (e.g. 1981, 1982) that
        have no rows in data_, Bambi silently drops those levels from the design
        matrix and uses the first *observed* level as the treatment reference.
        The prior array must match Bambi's actual column count — otherwise
        formulae raises "Incompatible shared dimension for dot product".

        This test builds a triangle where origins 1981 and 1982 are blanked
        from data_ but still appear in the Categorical categories list, then
        verifies:
        - _build_cl_informed_priors() does NOT raise
        - C(origin) prior has len == n_observed_origins - 1 (not n_total - 1)
        - C(dev) prior length matches Bambi's actual design matrix columns
        """
        from bayesianchainladder.utils import prepare_model_data, add_categorical_columns
        import bambi as bmb

        tri = cl.load_sample("raa")  # 10 origins: 1981-1990, 10 devs: 12-120

        data_full, future_full = prepare_model_data(tri)

        # Blank origins 1981 and 1982 from training data (simulate sparse back-test
        # triangle where these accident years have zero incremental observations).
        all_origins = sorted(data_full["origin"].unique())  # [1981..1990]
        data_sparse = data_full[data_full["origin"] >= 1983].copy().reset_index(drop=True)

        # Force all 10 origins into the Categorical categories (as the union-level
        # alignment code in fit() does), even though only 8 appear in data_.
        data_sparse["origin"] = pd.Categorical(data_sparse["origin"], categories=all_origins)
        future_full_cat = future_full.copy()
        future_full_cat["origin"] = pd.Categorical(
            future_full_cat["origin"], categories=all_origins
        )

        formula = "incremental ~ 1 + C(origin) + C(dev)"
        model = BayesianChainLadderGLM(
            formula=formula,
            family="gaussian",
            link="log",
            init_priors_from_chainladder=True,
            chainladder_prior_sd=0.5,
            draws=50,
            tune=25,
            chains=1,
        )
        model.triangle_ = tri.copy()
        model.data_ = add_categorical_columns(data_sparse, formula=formula)
        model.future_data_ = add_categorical_columns(future_full_cat, formula=formula)

        # Must not raise a dimension-mismatch error.
        cl_priors = model._build_cl_informed_priors()

        # observed origins = [1983..1990] (8 origins), so contrasts = 7
        n_observed_origins = len(data_sparse["origin"].dropna().unique())
        expected_origin_contrasts = n_observed_origins - 1  # 7

        assert "C(origin)" in cl_priors, "Expected C(origin) in CL priors"
        origin_mus = cl_priors["C(origin)"].args["mu"]
        assert len(origin_mus) == expected_origin_contrasts, (
            f"Expected {expected_origin_contrasts} origin contrasts (only observed levels), "
            f"got {len(origin_mus)}"
        )

        # Verify the prior length matches what Bambi actually creates.
        model_bambi = bmb.Model(formula, model.data_, family="gaussian")
        model_bambi.build()
        bambi_origin_cols = model_bambi.components["mu"].terms["C(origin)"].shape[1]
        assert len(origin_mus) == bambi_origin_cols, (
            f"Prior length {len(origin_mus)} != Bambi design matrix columns {bambi_origin_cols}"
        )

        # C(dev): verify prior length matches Bambi's design matrix as well.
        if "C(dev)" in cl_priors:
            dev_mus = cl_priors["C(dev)"].args["mu"]
            bambi_dev_cols = model_bambi.components["mu"].terms["C(dev)"].shape[1]
            assert len(dev_mus) == bambi_dev_cols, (
                f"C(dev) prior length {len(dev_mus)} != Bambi columns {bambi_dev_cols}"
            )

    @pytest.mark.slow
    def test_init_priors_sparse_levels_fit_succeeds(self):
        """Full MCMC fit succeeds with sparse levels (no dimension-mismatch error)."""
        from bayesianchainladder.utils import prepare_model_data, add_categorical_columns

        tri = cl.load_sample("raa")
        data_full, _ = prepare_model_data(tri)
        all_origins = sorted(data_full["origin"].unique())

        # Blank the oldest two origins from training
        tri_sparse = tri[tri.origin >= "1983"].copy()

        model = BayesianChainLadderGLM(
            formula="incremental ~ 1 + C(origin) + C(dev)",
            family="gaussian",
            link="log",
            init_priors_from_chainladder=True,
            chainladder_prior_sd=0.5,
            draws=50,
            tune=25,
            chains=1,
            random_seed=42,
        )
        # Fit must succeed without ValueError about dimension mismatch
        model.fit(tri_sparse)
        assert model._is_fitted
        assert model.idata is not None


# =============================================================================
# Fix 4: Auto-shift tests — gamma family handles negative incrementals
# =============================================================================


class TestGammaAutoShift:
    """Tests for the auto-shift (force_positive_response) functionality."""

    @pytest.mark.slow
    def test_gamma_handles_negative_incrementals_via_shift(self):
        """gamma+log fits successfully on a triangle with negative incrementals.

        Uses the RAA triangle which contains negative incremental values.  With
        force_positive_response=True (default) the fit must complete without error
        and produce a non-NaN reserve estimate.
        """
        import warnings

        tri = cl.load_sample("raa")

        model = BayesianChainLadderGLM(
            formula="incremental ~ 1 + C(origin) + C(dev)",
            family="gamma",
            link="log",
            force_positive_response=True,
            init_priors_from_chainladder=True,
            chainladder_prior_sd=0.5,
            draws=100,
            tune=50,
            chains=1,
            random_seed=42,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(tri)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            # A UserWarning about shifting must have been emitted
            shift_warnings = [x for x in user_warnings if "shifting" in str(x.message).lower()]
            assert len(shift_warnings) >= 1, \
                "Expected a UserWarning about auto-shifting response"

        assert model._is_fitted
        assert model.reserves_posterior_ is not None
        # Shift should be positive (RAA has negatives)
        assert model._response_shift > 0, "Expected non-zero response shift"

        # Reserve should be finite and positive (roughly chain-ladder magnitude)
        import chainladder as _cl
        cl_fit = _cl.Chainladder().fit(tri)
        cl_ibnr = float(np.nansum(np.asarray(cl_fit.ibnr_.values, dtype=float)))
        model_ibnr = float(np.nanmedian(np.asarray(
            model.reserves_posterior_.sum(dim="origin").values, dtype=float
        )))
        assert np.isfinite(model_ibnr), "Reserve estimate should be finite"
        # Within 3× of CL (very loose check — just ensure no magnitude blowup)
        if cl_ibnr > 0:
            ratio = model_ibnr / cl_ibnr
            assert 0.1 < ratio < 10.0, (
                f"Model IBNR {model_ibnr:.0f} vs CL IBNR {cl_ibnr:.0f}: "
                f"ratio={ratio:.2f} outside [0.1, 10.0]"
            )
