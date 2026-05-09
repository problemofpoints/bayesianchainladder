"""Tests for BaseStochasticReserve ABC and MethodSummary dataclass."""

import math
from dataclasses import FrozenInstanceError

import chainladder as cl
import numpy as np
import pytest
import xarray as xr

from bayesianchainladder.base import BaseStochasticReserve, MethodSummary


class TestMethodSummary:
    def test_basic_construction(self):
        summary = MethodSummary(
            total_reserve_mean=100.0,
            total_reserve_stddev=10.0,
            total_reserve_75th_percentile=107.0,
            total_reserve_90th_percentile=113.0,
            total_reserve_95th_percentile=117.0,
        )
        assert summary.total_reserve_mean == 100.0
        assert summary.total_reserve_stddev == 10.0
        assert summary.total_reserve_75th_percentile == 107.0
        assert summary.total_reserve_90th_percentile == 113.0
        assert summary.total_reserve_95th_percentile == 117.0

    def test_cv_property(self):
        summary = MethodSummary(
            total_reserve_mean=100.0,
            total_reserve_stddev=20.0,
            total_reserve_75th_percentile=0.0,
            total_reserve_90th_percentile=0.0,
            total_reserve_95th_percentile=0.0,
        )
        assert summary.total_reserve_cv == pytest.approx(0.2)

    def test_cv_with_zero_mean_returns_nan(self):
        summary = MethodSummary(
            total_reserve_mean=0.0,
            total_reserve_stddev=10.0,
            total_reserve_75th_percentile=0.0,
            total_reserve_90th_percentile=0.0,
            total_reserve_95th_percentile=0.0,
        )
        assert math.isnan(summary.total_reserve_cv)

    def test_cv_with_nan_mean_returns_nan(self):
        summary = MethodSummary(
            total_reserve_mean=float("nan"),
            total_reserve_stddev=10.0,
            total_reserve_75th_percentile=0.0,
            total_reserve_90th_percentile=0.0,
            total_reserve_95th_percentile=0.0,
        )
        assert math.isnan(summary.total_reserve_cv)

    def test_cv_with_negative_mean(self):
        # CV uses abs(mean) so negative means produce positive CVs
        summary = MethodSummary(
            total_reserve_mean=-100.0,
            total_reserve_stddev=20.0,
            total_reserve_75th_percentile=0.0,
            total_reserve_90th_percentile=0.0,
            total_reserve_95th_percentile=0.0,
        )
        assert summary.total_reserve_cv == pytest.approx(0.2)

    def test_dataclass_is_frozen(self):
        summary = MethodSummary(
            total_reserve_mean=100.0,
            total_reserve_stddev=10.0,
            total_reserve_75th_percentile=0.0,
            total_reserve_90th_percentile=0.0,
            total_reserve_95th_percentile=0.0,
        )
        with pytest.raises(FrozenInstanceError):
            summary.total_reserve_mean = 200.0  # type: ignore[misc]


class _StubReserve(BaseStochasticReserve):
    """Minimal subclass for exercising the base helpers without a real fit."""

    def fit(self, triangle, samples=None, random_seed=None):
        from bayesianchainladder.utils import _extract_period_value, validate_triangle

        validate_triangle(triangle)
        self.triangle_ = triangle.copy()

        if samples is None:
            # Default: 100 samples per origin, drawn from a fixed normal so
            # the tests are deterministic.
            origins = sorted({_extract_period_value(o) for o in triangle.origin})
            rng = np.random.default_rng(random_seed if random_seed is not None else 42)
            arr = rng.normal(loc=1000.0, scale=100.0, size=(len(origins), 100))
            self.reserves_posterior_ = xr.DataArray(
                arr,
                dims=["origin", "sample"],
                coords={"origin": origins, "sample": np.arange(100)},
            )
        else:
            self.reserves_posterior_ = samples

        self._build_reserve_summaries()
        self._is_fitted = True
        return self


@pytest.fixture
def stub_fitted():
    triangle = cl.load_sample("raa")
    return _StubReserve().fit(triangle, random_seed=42)


class TestBaseStochasticReserve:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            BaseStochasticReserve()  # type: ignore[abstract]

    def test_check_is_fitted_raises_before_fit(self):
        stub = _StubReserve()
        with pytest.raises(ValueError, match="has not been fitted"):
            stub.summary()

    def test_fit_populates_attributes(self, stub_fitted):
        assert stub_fitted._is_fitted is True
        assert stub_fitted.triangle_ is not None
        assert stub_fitted.reserves_posterior_ is not None
        assert stub_fitted.ibnr_ is not None
        assert stub_fitted.ultimate_ is not None

    def test_ibnr_columns(self, stub_fitted):
        assert list(stub_fitted.ibnr_.columns) == [
            "mean", "std", "median", "5%", "25%", "75%", "95%"
        ]

    def test_ultimate_columns(self, stub_fitted):
        assert list(stub_fitted.ultimate_.columns) == [
            "paid_to_date", "mean", "std", "median", "5%", "25%", "75%", "95%"
        ]

    def test_ultimate_mean_equals_paid_plus_ibnr(self, stub_fitted):
        diff = (
            stub_fitted.ultimate_["mean"]
            - stub_fitted.ultimate_["paid_to_date"]
            - stub_fitted.ibnr_["mean"]
        )
        assert (diff.abs() < 1e-9).all()

    def test_summary_includes_total_row_by_default(self, stub_fitted):
        summary = stub_fitted.summary()
        assert "Total" in summary.index

    def test_summary_can_exclude_total_row(self, stub_fitted):
        summary = stub_fitted.summary(include_totals=False)
        assert "Total" not in summary.index

    def test_summary_has_multiindex_columns(self, stub_fitted):
        summary = stub_fitted.summary()
        assert summary.columns.nlevels == 2
        assert set(summary.columns.get_level_values(0)) == {"Ultimate", "IBNR"}

    def test_sample_reserves_returns_array_of_correct_shape(self, stub_fitted):
        samples = stub_fitted.sample_reserves(n_samples=500, random_seed=0)
        assert isinstance(samples, np.ndarray)
        assert samples.shape == (500,)

    def test_sample_reserves_is_finite(self, stub_fitted):
        samples = stub_fitted.sample_reserves(n_samples=200, random_seed=0)
        assert np.all(np.isfinite(samples))

    def test_total_summary_returns_method_summary(self, stub_fitted):
        from bayesianchainladder.base import MethodSummary

        result = stub_fitted.total_summary()
        assert isinstance(result, MethodSummary)
        assert np.isfinite(result.total_reserve_mean)
        assert result.total_reserve_stddev > 0
        assert result.total_reserve_95th_percentile > result.total_reserve_75th_percentile

    def test_total_summary_cv_is_positive(self, stub_fitted):
        result = stub_fitted.total_summary()
        assert result.total_reserve_cv > 0
