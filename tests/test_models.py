"""Tests for model building functions."""

import numpy as np
import pandas as pd
import pytest

import bambi as bmb

import pymc as pm

from bayesianchainladder.models import (
    build_bambi_model,
    build_csr_model,
    extract_parameter_summary,
)


@pytest.fixture
def sample_data():
    """Create sample data for model testing."""
    return pd.DataFrame({
        "incremental": [100, 80, 60, 50, 110, 90, 70, 120, 100, 130],
        "origin": [1, 1, 1, 1, 2, 2, 2, 3, 3, 4],
        "dev": [1, 2, 3, 4, 1, 2, 3, 1, 2, 1],
        "calendar": [1, 2, 3, 4, 2, 3, 4, 3, 4, 4],
    })


class TestBuildBambiModel:
    """Tests for build_bambi_model function."""

    def test_basic_model_creation(self, sample_data):
        """Test basic model creation."""
        model = build_bambi_model(
            sample_data,
            formula="incremental ~ 1 + C(origin) + C(dev)",
            family="negativebinomial",
        )

        assert isinstance(model, bmb.Model)

    def test_poisson_family(self, sample_data):
        """Test model with Poisson family."""
        model = build_bambi_model(
            sample_data,
            formula="incremental ~ 1 + C(origin) + C(dev)",
            family="poisson",
        )

        assert isinstance(model, bmb.Model)

    def test_gamma_family(self, sample_data):
        """Test model with Gamma family."""
        model = build_bambi_model(
            sample_data,
            formula="incremental ~ 1 + C(origin) + C(dev)",
            family="gamma",
        )

        assert isinstance(model, bmb.Model)

    def test_offset_from_column(self, sample_data):
        """Test offset from data column."""
        sample_data["exposure"] = [1000] * len(sample_data)

        model = build_bambi_model(
            sample_data,
            formula="incremental ~ 1 + C(origin) + C(dev)",
            family="negativebinomial",
            offset="exposure",
        )

        assert isinstance(model, bmb.Model)

    def test_offset_from_array(self, sample_data):
        """Test offset from numpy array."""
        offset = np.log(np.ones(len(sample_data)) * 1000)

        model = build_bambi_model(
            sample_data,
            formula="incremental ~ 1 + C(origin) + C(dev)",
            family="negativebinomial",
            offset=offset,
        )

        assert isinstance(model, bmb.Model)

    def test_invalid_family_raises(self, sample_data):
        """Test that invalid family raises error."""
        with pytest.raises(ValueError, match="Unknown family"):
            build_bambi_model(
                sample_data,
                formula="incremental ~ 1",
                family="invalid_family",
            )

    def test_missing_offset_column_raises(self, sample_data):
        """Test that missing offset column raises error."""
        with pytest.raises(ValueError, match="not found"):
            build_bambi_model(
                sample_data,
                formula="incremental ~ 1",
                family="negativebinomial",
                offset="nonexistent_column",
            )

    def test_wrong_offset_length_raises(self, sample_data):
        """Test that wrong offset length raises error."""
        offset = np.array([1, 2, 3])  # Wrong length

        with pytest.raises(ValueError, match="length"):
            build_bambi_model(
                sample_data,
                formula="incremental ~ 1",
                family="negativebinomial",
                offset=offset,
            )

    def test_custom_priors(self, sample_data):
        """Test model with custom priors."""
        priors = {
            "Intercept": bmb.Prior("Normal", mu=5, sigma=2),
        }

        model = build_bambi_model(
            sample_data,
            formula="incremental ~ 1",
            family="negativebinomial",
            priors=priors,
        )

        assert isinstance(model, bmb.Model)

    def test_calendar_effects(self, sample_data):
        """Test model with calendar effects."""
        model = build_bambi_model(
            sample_data,
            formula="incremental ~ 1 + C(origin) + C(dev) + C(calendar)",
            family="negativebinomial",
        )

        assert isinstance(model, bmb.Model)


class TestFamilyMapping:
    """Tests for family name mapping."""

    def test_negativebinomial_variants(self, sample_data):
        """Test different spellings of negative binomial."""
        for family in ["negativebinomial", "negative_binomial", "negbinom"]:
            model = build_bambi_model(
                sample_data,
                formula="incremental ~ 1",
                family=family,
            )
            assert isinstance(model, bmb.Model)

    def test_gaussian_variants(self, sample_data):
        """Test different spellings of Gaussian."""
        for family in ["gaussian", "normal"]:
            model = build_bambi_model(
                sample_data,
                formula="incremental ~ 1",
                family=family,
            )
            assert isinstance(model, bmb.Model)


@pytest.fixture
def csr_sample_data():
    """Create sample data for CSR model testing."""
    return pd.DataFrame({
        "logprem": [10.0, 10.0, 10.0, 10.0, 10.1, 10.1, 10.1, 10.2, 10.2, 10.3],
        "logloss": [8.0, 8.5, 8.8, 8.9, 8.1, 8.6, 8.85, 8.2, 8.65, 8.3],
        "origin": [1, 1, 1, 1, 2, 2, 2, 3, 3, 4],
        "dev": [1, 2, 3, 4, 1, 2, 3, 1, 2, 1],
    })


class TestBuildCSRModel:
    """Tests for build_csr_model function."""

    def test_basic_model_creation(self, csr_sample_data):
        """Test basic CSR model creation."""
        model = build_csr_model(csr_sample_data)

        assert isinstance(model, pm.Model)

    def test_model_has_expected_variables(self, csr_sample_data):
        """Test that model contains expected random variables."""
        model = build_csr_model(csr_sample_data)

        # Check free RVs
        free_rv_names = [rv.name for rv in model.free_RVs]
        assert "r_alpha" in free_rv_names
        assert "r_beta" in free_rv_names
        assert "logelr" in free_rv_names
        assert "gamma" in free_rv_names
        assert "a_ig" in free_rv_names

    def test_model_has_expected_deterministics(self, csr_sample_data):
        """Test that model contains expected deterministic variables."""
        model = build_csr_model(csr_sample_data)

        deterministic_names = [var.name for var in model.deterministics]
        assert "alpha" in deterministic_names
        assert "beta" in deterministic_names
        assert "speedup" in deterministic_names
        assert "sig" in deterministic_names
        assert "mu" in deterministic_names

    def test_model_has_observed(self, csr_sample_data):
        """Test that model has observed variable."""
        model = build_csr_model(csr_sample_data)

        observed_names = [rv.name for rv in model.observed_RVs]
        assert "logloss" in observed_names

    def test_model_coords(self, csr_sample_data):
        """Test that model has correct coordinates."""
        model = build_csr_model(csr_sample_data)

        assert "origin" in model.coords
        assert "dev" in model.coords
        assert "obs" in model.coords
        assert "origin_raw" in model.coords
        assert "dev_raw" in model.coords

        # Check coordinate sizes
        assert len(model.coords["origin"]) == 4
        assert len(model.coords["dev"]) == 4
        assert len(model.coords["obs"]) == 10
        assert len(model.coords["origin_raw"]) == 3  # n_origin - 1
        assert len(model.coords["dev_raw"]) == 3  # n_dev - 1

    def test_custom_priors(self, csr_sample_data):
        """Test model with custom prior specifications."""
        priors = {
            "alpha": {"sigma": 2.0},
            "beta": {"sigma": 2.0},
            "logelr": {"mu": 0.0, "sigma": 1.0},
            "gamma": {"mu": 0.0, "sigma": 0.1},
            "a_ig": {"alpha": 2.0, "beta": 2.0},
        }

        model = build_csr_model(csr_sample_data, priors=priors)

        assert isinstance(model, pm.Model)

    def test_custom_column_names(self):
        """Test model with custom column names."""
        data = pd.DataFrame({
            "log_premium": [10.0, 10.0, 10.1, 10.2],
            "log_cumulative": [8.0, 8.5, 8.1, 8.2],
            "accident_year": [1, 1, 2, 3],
            "development": [1, 2, 1, 1],
        })

        model = build_csr_model(
            data,
            logprem_col="log_premium",
            logloss_col="log_cumulative",
            origin_col="accident_year",
            dev_col="development",
        )

        assert isinstance(model, pm.Model)
