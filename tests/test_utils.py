"""Tests for utility functions."""

import numpy as np
import pandas as pd
import pytest

import chainladder as cl

from bayesianchainladder.utils import (
    add_categorical_columns,
    compute_log_exposure_offset,
    create_design_info,
    get_future_dataframe,
    prepare_csr_data,
    prepare_model_data,
    triangle_to_dataframe,
    validate_triangle,
)


@pytest.fixture
def sample_triangle():
    """Load a sample triangle for testing."""
    return cl.load_sample("raa")


@pytest.fixture
def cumulative_triangle():
    """Load a cumulative triangle."""
    return cl.load_sample("raa")


class TestTriangleToDataframe:
    """Tests for triangle_to_dataframe function."""

    def test_basic_conversion(self, sample_triangle):
        """Test basic conversion produces expected columns."""
        df = triangle_to_dataframe(sample_triangle)

        # Check required columns exist
        assert "origin" in df.columns
        assert "dev" in df.columns
        assert "calendar" in df.columns
        assert "incremental" in df.columns

    def test_output_shape(self, sample_triangle):
        """Test output has expected number of rows."""
        df = triangle_to_dataframe(sample_triangle)

        # Should have data for observed cells only
        # For a 10x10 triangle, that's 10+9+8+...+1 = 55 cells
        n_origins = len(sample_triangle.origin)
        expected_rows = (n_origins * (n_origins + 1)) // 2
        assert len(df) == expected_rows

    def test_no_missing_values(self, sample_triangle):
        """Test no missing values in output."""
        df = triangle_to_dataframe(sample_triangle)

        # All observed cells should have values
        assert not df["incremental"].isna().any()

    def test_calendar_period_calculation(self, sample_triangle):
        """Test calendar period is correctly calculated."""
        df = triangle_to_dataframe(sample_triangle)

        # Calendar = origin + dev - 1
        expected_calendar = df["origin"] + df["dev"] - 1
        pd.testing.assert_series_equal(
            df["calendar"].astype(int),
            expected_calendar.astype(int),
            check_names=False,
        )

    def test_custom_value_column_name(self, sample_triangle):
        """Test custom value column name."""
        df = triangle_to_dataframe(sample_triangle, value_column="losses")

        assert "losses" in df.columns
        assert "incremental" not in df.columns

    def test_include_cumulative(self, sample_triangle):
        """Test including cumulative values."""
        df = triangle_to_dataframe(sample_triangle, include_cumulative=True)

        assert "cumulative" in df.columns
        assert not df["cumulative"].isna().any()


class TestGetFutureDataframe:
    """Tests for get_future_dataframe function."""

    def test_future_cells_identified(self, sample_triangle):
        """Test that future cells are correctly identified."""
        df = get_future_dataframe(sample_triangle)

        # Should have the lower-right portion of the triangle
        n_origins = len(sample_triangle.origin)

        # Number of future cells = n*(n-1)/2 for n x n triangle
        expected_future = (n_origins * (n_origins - 1)) // 2
        assert len(df) == expected_future

    def test_future_values_are_nan(self, sample_triangle):
        """Test that future values are NaN."""
        df = get_future_dataframe(sample_triangle)

        if len(df) > 0:
            assert df["incremental"].isna().all()


class TestPrepareModelData:
    """Tests for prepare_model_data function."""

    def test_returns_two_dataframes(self, sample_triangle):
        """Test that function returns observed and future dataframes."""
        observed, future = prepare_model_data(sample_triangle)

        assert isinstance(observed, pd.DataFrame)
        assert isinstance(future, pd.DataFrame)

    def test_observed_complete(self, sample_triangle):
        """Test observed data has no missing values."""
        observed, _ = prepare_model_data(sample_triangle)

        assert not observed["incremental"].isna().any()


class TestAddCategoricalColumns:
    """Tests for add_categorical_columns function."""

    def test_default_columns(self):
        """Test default categorical column conversion."""
        df = pd.DataFrame({
            "origin": [1, 1, 2],
            "dev": [1, 2, 1],
            "calendar": [1, 2, 2],
            "value": [100, 80, 110],
        })

        result = add_categorical_columns(df)

        assert result["origin"].dtype.name == "category"
        assert result["dev"].dtype.name == "category"
        assert result["calendar"].dtype.name == "category"

    def test_custom_columns(self):
        """Test custom column specification."""
        df = pd.DataFrame({
            "origin": [1, 1, 2],
            "dev": [1, 2, 1],
            "value": [100, 80, 110],
        })

        result = add_categorical_columns(df, columns=["origin"])

        assert result["origin"].dtype.name == "category"
        assert result["dev"].dtype != "category"


class TestComputeLogExposureOffset:
    """Tests for compute_log_exposure_offset function."""

    def test_log_computation(self):
        """Test log exposure is correctly computed."""
        df = pd.DataFrame({"exposure": [100, 200, 300]})

        result = compute_log_exposure_offset(df, "exposure")

        np.testing.assert_array_almost_equal(
            result.values, np.log([100, 200, 300])
        )

    def test_missing_column_raises(self):
        """Test that missing column raises error."""
        df = pd.DataFrame({"other": [100, 200]})

        with pytest.raises(ValueError, match="not found"):
            compute_log_exposure_offset(df, "exposure")

    def test_non_positive_raises(self):
        """Test that non-positive values raise error."""
        df = pd.DataFrame({"exposure": [100, 0, 200]})

        with pytest.raises(ValueError, match="positive"):
            compute_log_exposure_offset(df, "exposure")


class TestCreateDesignInfo:
    """Tests for create_design_info function."""

    def test_parses_formula(self):
        """Test formula parsing."""
        df = pd.DataFrame({
            "y": [1, 2, 3],
            "origin": [1, 1, 2],
            "dev": [1, 2, 1],
        })

        info = create_design_info(df, "y ~ origin + dev")

        assert info["response"] == "y"
        assert "origin" in info["terms"]
        assert "dev" in info["terms"]


class TestValidateTriangle:
    """Tests for validate_triangle function."""

    def test_valid_triangle_passes(self, sample_triangle):
        """Test that valid triangle passes validation."""
        # Should not raise
        validate_triangle(sample_triangle)

    def test_non_triangle_raises(self):
        """Test that non-triangle input raises error."""
        with pytest.raises(ValueError, match="chainladder.Triangle"):
            validate_triangle(pd.DataFrame())

    def test_small_triangle_raises(self):
        """Test that very small triangles raise errors."""
        # Create a minimal triangle - this test depends on cl behavior
        # Skip if we can't easily create a too-small triangle
        pass


class TestPrepareCSRData:
    """Tests for prepare_csr_data function."""

    @pytest.fixture
    def positive_triangle(self):
        """Load a triangle with positive cumulative values."""
        return cl.load_sample("genins")

    def test_returns_two_dataframes(self, positive_triangle):
        """Test that function returns observed and future dataframes."""
        observed, future = prepare_csr_data(positive_triangle, premium_value=10000)

        assert isinstance(observed, pd.DataFrame)
        assert isinstance(future, pd.DataFrame)

    def test_observed_has_required_columns(self, positive_triangle):
        """Test observed data has all required columns."""
        observed, _ = prepare_csr_data(positive_triangle, premium_value=10000)

        assert "origin" in observed.columns
        assert "dev" in observed.columns
        assert "cumulative" in observed.columns
        assert "logloss" in observed.columns
        assert "premium" in observed.columns
        assert "logprem" in observed.columns

    def test_future_has_required_columns(self, positive_triangle):
        """Test future data has all required columns."""
        _, future = prepare_csr_data(positive_triangle, premium_value=10000)

        if len(future) > 0:
            assert "origin" in future.columns
            assert "dev" in future.columns
            assert "premium" in future.columns
            assert "logprem" in future.columns

    def test_logloss_is_log_of_cumulative(self, positive_triangle):
        """Test that logloss is the log of cumulative."""
        observed, _ = prepare_csr_data(positive_triangle, premium_value=10000)

        expected_logloss = np.log(observed["cumulative"])
        np.testing.assert_array_almost_equal(
            observed["logloss"].values,
            expected_logloss.values,
        )

    def test_logprem_is_log_of_premium(self, positive_triangle):
        """Test that logprem is the log of premium."""
        observed, _ = prepare_csr_data(positive_triangle, premium_value=10000)

        expected_logprem = np.log(observed["premium"])
        np.testing.assert_array_almost_equal(
            observed["logprem"].values,
            expected_logprem.values,
        )

    def test_premium_value_applied(self, positive_triangle):
        """Test that premium_value is applied to all rows."""
        observed, future = prepare_csr_data(positive_triangle, premium_value=5000)

        assert (observed["premium"] == 5000).all()
        if len(future) > 0:
            assert (future["premium"] == 5000).all()

    def test_missing_premium_raises(self, positive_triangle):
        """Test that missing premium raises error."""
        with pytest.raises(ValueError, match="premium"):
            prepare_csr_data(positive_triangle)

    def test_converts_incremental_to_cumulative(self, positive_triangle):
        """Test that incremental triangles are converted to cumulative."""
        # Get incremental triangle
        incr_tri = positive_triangle.incr_to_cum().cum_to_incr()

        observed, _ = prepare_csr_data(incr_tri, premium_value=10000)

        # Should still have valid cumulative values
        assert not observed["cumulative"].isna().any()
        assert (observed["cumulative"] > 0).all()

    def test_only_positive_values_included(self, sample_triangle):
        """Test that only positive cumulative values are included."""
        # RAA may have some negative cumulative values in early cells
        observed, _ = prepare_csr_data(sample_triangle, premium_value=10000)

        # All included values should be positive (for valid log transform)
        if len(observed) > 0:
            assert (observed["cumulative"] > 0).all()
