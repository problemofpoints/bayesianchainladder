"""Tests for BaseStochasticReserve ABC and MethodSummary dataclass."""

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from bayesianchainladder.base import MethodSummary


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
