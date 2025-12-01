"""Tests for model building functions."""

import numpy as np
import pandas as pd
import pytest

import bambi as bmb

from bayesianchainladder.models import (
    build_bambi_model,
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
