"""Shared abstract base class and dataclass for stochastic reserve estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MethodSummary:
    """Total-reserve summary returned by ``BaseStochasticReserve.total_summary``.

    Mirrors the shape consumed by ``01_run_stochastic_methods.py`` so that any
    estimator can be plugged into that script with a single call.
    """

    total_reserve_mean: float
    total_reserve_stddev: float
    total_reserve_75th_percentile: float
    total_reserve_90th_percentile: float
    total_reserve_95th_percentile: float

    @property
    def total_reserve_cv(self) -> float:
        if self.total_reserve_mean == 0 or not np.isfinite(self.total_reserve_mean):
            return float("nan")
        return self.total_reserve_stddev / abs(self.total_reserve_mean)
