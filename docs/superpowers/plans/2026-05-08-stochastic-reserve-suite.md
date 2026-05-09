# Stochastic Reserve Suite — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `BaseStochasticReserve` ABC plus three new wrapper estimators (Mack, ODP Bootstrap, Correlated ODP Bootstrap) that share a common interface; retrofit the existing `BayesianChainLadderGLM` and `BayesianCSR` to inherit from the same base.

**Architecture:** New `base.py` module owns the ABC and a `MethodSummary` dataclass. New `bootstrap.py` module contains the three new wrappers and the low-level `CorrelatedBootstrapODPSample` (a `chainladder.DevelopmentBase` subclass implementing Clark/Ding/Zhou 2022 calendar-year-correlated bootstrap). The base class stores per-origin samples in an `xr.DataArray` with dims `(origin, sample)` and computes `ibnr_`, `ultimate_`, `summary()`, `sample_reserves()`, and `total_summary()` once for all subclasses. Subclasses only need to populate `triangle_` and `reserves_posterior_` during `fit()`.

**Tech Stack:** Python 3.11+, chainladder ≥0.8, numpy, pandas, scipy, xarray; existing tooling (uv, pytest, ruff, mypy).

**Spec:** [`docs/superpowers/specs/2026-05-08-stochastic-reserve-suite-design.md`](../specs/2026-05-08-stochastic-reserve-suite-design.md)

---

## Task 1: Add `MethodSummary` dataclass and base module skeleton

**Files:**
- Create: `bayesianchainladder/base.py`
- Create: `tests/test_base.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_base.py` with the following content:

```python
"""Tests for BaseStochasticReserve ABC and MethodSummary dataclass."""

import math

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
        with pytest.raises(Exception):
            summary.total_reserve_mean = 200.0  # type: ignore[misc]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_base.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'bayesianchainladder.base'`.

- [ ] **Step 3: Create `bayesianchainladder/base.py` with the dataclass**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_base.py -v
```

Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add bayesianchainladder/base.py tests/test_base.py
git commit -m "$(cat <<'EOF'
Add MethodSummary dataclass to bayesianchainladder.base

First piece of the stochastic reserve suite (Phase 1 design).
MethodSummary holds total-reserve stats in the shape consumed by
01_run_stochastic_methods.py and exposes a derived total_reserve_cv.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add `BaseStochasticReserve` ABC with concrete helpers

**Files:**
- Modify: `bayesianchainladder/base.py`
- Modify: `tests/test_base.py`

- [ ] **Step 1: Write the failing tests for the ABC**

Append the following to `tests/test_base.py` (after the `TestMethodSummary` class):

```python
import chainladder as cl
import pandas as pd
import xarray as xr

from bayesianchainladder.base import BaseStochasticReserve


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_base.py -v
```

Expected: tests in `TestBaseStochasticReserve` FAIL (`ImportError: cannot import name 'BaseStochasticReserve'`); tests in `TestMethodSummary` continue to pass.

- [ ] **Step 3: Add the ABC to `bayesianchainladder/base.py`**

Replace the contents of `bayesianchainladder/base.py` with:

```python
"""Shared abstract base class and dataclass for stochastic reserve estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    import chainladder as cl


@dataclass(frozen=True)
class MethodSummary:
    """Total-reserve summary returned by ``BaseStochasticReserve.total_summary``."""

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


class BaseStochasticReserve(ABC):
    """Abstract base for stochastic reserve estimators.

    Subclass contract
    -----------------
    Subclasses implement ``fit(triangle, ...)``. During ``fit`` they MUST:
      1. Validate the input triangle.
      2. Set ``self.triangle_`` to a copy of the input.
      3. Populate ``self.reserves_posterior_`` as an ``xr.DataArray`` with an
         ``origin`` dimension and one or more sample dimension(s). The
         remaining dim(s) are flattened automatically when computing
         summaries.
      4. Call ``self._build_reserve_summaries()`` to populate ``ibnr_`` and
         ``ultimate_``.
      5. Set ``self._is_fitted = True``.

    Subclasses MAY override ``sample_reserves`` and ``total_summary`` if they
    need to substitute a calibrated total (e.g. Mack's ``total_mack_std_err_``).
    """

    triangle_: "cl.Triangle | None"
    ibnr_: "pd.DataFrame | None"
    ultimate_: "pd.DataFrame | None"
    reserves_posterior_: "xr.DataArray | None"
    _is_fitted: bool

    def __init__(self) -> None:
        self.triangle_ = None
        self.ibnr_ = None
        self.ultimate_ = None
        self.reserves_posterior_ = None
        self._is_fitted = False

    @abstractmethod
    def fit(self, triangle, **kwargs: Any) -> "BaseStochasticReserve":
        """Fit the estimator to a triangle. Subclasses implement this."""

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise ValueError(
                "Model has not been fitted. Call fit() before using this method."
            )

    def _paid_to_date(self) -> pd.Series:
        """Per-origin paid-to-date totals from ``triangle_``."""
        from .utils import triangle_to_dataframe

        if self.triangle_ is None:
            raise ValueError("triangle_ is not set")
        df = triangle_to_dataframe(self.triangle_)
        return df.groupby("origin", observed=True)["incremental"].sum()

    def _build_reserve_summaries(self) -> None:
        """Populate ``ibnr_`` and ``ultimate_`` from ``reserves_posterior_``.

        Subclasses populate ``reserves_posterior_`` during ``fit`` then call
        this helper. They do not override it.
        """
        if self.reserves_posterior_ is None:
            raise ValueError("reserves_posterior_ must be populated first")

        paid = self._paid_to_date()
        origins = list(self.reserves_posterior_.coords["origin"].values)

        rows = []
        for origin in origins:
            samples = self.reserves_posterior_.sel(origin=origin).values.flatten()
            samples = samples[np.isfinite(samples)]
            paid_origin = float(paid.get(origin, 0.0))

            if samples.size == 0:
                ibnr_mean = ibnr_std = ibnr_median = 0.0
                ibnr_q05 = ibnr_q25 = ibnr_q75 = ibnr_q95 = 0.0
            else:
                ibnr_mean = float(np.mean(samples))
                ibnr_std = float(np.std(samples)) if samples.size > 1 else 0.0
                ibnr_median = float(np.median(samples))
                ibnr_q05 = float(np.percentile(samples, 5))
                ibnr_q25 = float(np.percentile(samples, 25))
                ibnr_q75 = float(np.percentile(samples, 75))
                ibnr_q95 = float(np.percentile(samples, 95))

            rows.append({
                "origin": origin,
                "paid_to_date": paid_origin,
                "ibnr_mean": ibnr_mean,
                "ibnr_std": ibnr_std,
                "ibnr_median": ibnr_median,
                "ibnr_5%": ibnr_q05,
                "ibnr_25%": ibnr_q25,
                "ibnr_75%": ibnr_q75,
                "ibnr_95%": ibnr_q95,
                "ultimate_mean": paid_origin + ibnr_mean,
                "ultimate_std": ibnr_std,
                "ultimate_median": paid_origin + ibnr_median,
                "ultimate_5%": paid_origin + ibnr_q05,
                "ultimate_25%": paid_origin + ibnr_q25,
                "ultimate_75%": paid_origin + ibnr_q75,
                "ultimate_95%": paid_origin + ibnr_q95,
            })

        df = pd.DataFrame(rows).set_index("origin")
        self.ibnr_ = df[
            ["ibnr_mean", "ibnr_std", "ibnr_median",
             "ibnr_5%", "ibnr_25%", "ibnr_75%", "ibnr_95%"]
        ].copy()
        self.ibnr_.columns = ["mean", "std", "median", "5%", "25%", "75%", "95%"]

        self.ultimate_ = df[
            ["paid_to_date", "ultimate_mean", "ultimate_std", "ultimate_median",
             "ultimate_5%", "ultimate_25%", "ultimate_75%", "ultimate_95%"]
        ].copy()
        self.ultimate_.columns = [
            "paid_to_date", "mean", "std", "median", "5%", "25%", "75%", "95%"
        ]

    def summary(self, include_totals: bool = True) -> pd.DataFrame:
        self._check_is_fitted()
        if self.ultimate_ is None or self.ibnr_ is None:
            raise ValueError("Reserve summaries are not available")

        result = pd.concat(
            [
                self.ultimate_[["paid_to_date", "mean", "std", "median"]],
                self.ibnr_[["mean", "std", "median"]],
            ],
            axis=1,
            keys=["Ultimate", "IBNR"],
        )

        if include_totals:
            total_paid = float(self.ultimate_["paid_to_date"].sum())
            total = self.reserves_posterior_.sum(dim="origin").values.flatten()
            total = total[np.isfinite(total)]
            total_ibnr_mean = float(np.mean(total)) if total.size else 0.0
            total_ibnr_std = float(np.std(total)) if total.size > 1 else 0.0
            total_ibnr_median = float(np.median(total)) if total.size else 0.0

            total_row = pd.DataFrame(
                {
                    ("Ultimate", "paid_to_date"): [total_paid],
                    ("Ultimate", "mean"): [total_paid + total_ibnr_mean],
                    ("Ultimate", "std"): [total_ibnr_std],
                    ("Ultimate", "median"): [total_paid + total_ibnr_median],
                    ("IBNR", "mean"): [total_ibnr_mean],
                    ("IBNR", "std"): [total_ibnr_std],
                    ("IBNR", "median"): [total_ibnr_median],
                },
                index=["Total"],
            )
            result = pd.concat([result, total_row])

        return result

    def sample_reserves(
        self,
        n_samples: int = 1000,
        random_seed: int | None = None,
    ) -> np.ndarray:
        self._check_is_fitted()
        if self.reserves_posterior_ is None:
            raise ValueError("No reserve posterior available")
        total = self.reserves_posterior_.sum(dim="origin").values.flatten()
        total = total[np.isfinite(total)]
        if total.size == 0:
            return np.full(n_samples, np.nan)
        rng = np.random.default_rng(random_seed)
        replace = n_samples > total.size
        idx = rng.choice(total.size, size=n_samples, replace=replace)
        return total[idx]

    def total_summary(self) -> MethodSummary:
        self._check_is_fitted()
        if self.reserves_posterior_ is None:
            raise ValueError("No reserve posterior available")
        total = self.reserves_posterior_.sum(dim="origin").values.flatten()
        total = total[np.isfinite(total)]
        if total.size == 0:
            return MethodSummary(
                total_reserve_mean=float("nan"),
                total_reserve_stddev=float("nan"),
                total_reserve_75th_percentile=float("nan"),
                total_reserve_90th_percentile=float("nan"),
                total_reserve_95th_percentile=float("nan"),
            )
        return MethodSummary(
            total_reserve_mean=float(np.mean(total)),
            total_reserve_stddev=float(np.std(total, ddof=1)) if total.size > 1 else 0.0,
            total_reserve_75th_percentile=float(np.quantile(total, 0.75)),
            total_reserve_90th_percentile=float(np.quantile(total, 0.90)),
            total_reserve_95th_percentile=float(np.quantile(total, 0.95)),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_base.py -v
```

Expected: all tests in `TestMethodSummary` and `TestBaseStochasticReserve` PASS (~18 tests).

- [ ] **Step 5: Commit**

```bash
git add bayesianchainladder/base.py tests/test_base.py
git commit -m "$(cat <<'EOF'
Add BaseStochasticReserve ABC with shared summary helpers

ABC defines the contract every stochastic reserve estimator follows:
fit(), ibnr_, ultimate_, reserves_posterior_, summary(), sample_reserves(),
total_summary(). All summary/sampling logic lives in the base; subclasses
just populate triangle_ and reserves_posterior_ during fit.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add `MackChainLadder` wrapper

**Files:**
- Create: `bayesianchainladder/bootstrap.py`
- Create: `tests/test_bootstrap.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_bootstrap.py`:

```python
"""Tests for stochastic reserve wrappers in bayesianchainladder.bootstrap."""

import chainladder as cl
import numpy as np
import pytest

from bayesianchainladder.base import BaseStochasticReserve, MethodSummary


@pytest.fixture
def raa_triangle():
    return cl.load_sample("raa")


class TestMackChainLadder:
    def test_inherits_from_base(self):
        from bayesianchainladder.bootstrap import MackChainLadder

        model = MackChainLadder()
        assert isinstance(model, BaseStochasticReserve)

    def test_fit_returns_self(self, raa_triangle):
        from bayesianchainladder.bootstrap import MackChainLadder

        model = MackChainLadder()
        result = model.fit(raa_triangle)
        assert result is model
        assert model._is_fitted is True

    def test_fit_populates_attributes(self, raa_triangle):
        from bayesianchainladder.bootstrap import MackChainLadder

        model = MackChainLadder().fit(raa_triangle)
        assert model.triangle_ is not None
        assert model.reserves_posterior_ is not None
        assert model.ibnr_ is not None
        assert model.ultimate_ is not None

    def test_ibnr_columns(self, raa_triangle):
        from bayesianchainladder.bootstrap import MackChainLadder

        model = MackChainLadder().fit(raa_triangle)
        assert list(model.ibnr_.columns) == [
            "mean", "std", "median", "5%", "25%", "75%", "95%"
        ]

    def test_summary_has_total_row(self, raa_triangle):
        from bayesianchainladder.bootstrap import MackChainLadder

        model = MackChainLadder().fit(raa_triangle)
        summary = model.summary()
        assert "Total" in summary.index

    def test_total_summary_uses_calibrated_total_stderr(self, raa_triangle):
        """MackChainLadder's total_summary().total_reserve_stddev should
        match chainladder's total_mack_std_err_, NOT the std of summed
        per-origin samples (which would be lower because per-origin draws
        are independent)."""
        from bayesianchainladder.bootstrap import MackChainLadder

        mack_native = cl.MackChainladder().fit(raa_triangle)
        expected_total_std = float(
            np.asarray(mack_native.total_mack_std_err_).flatten()[0]
        )

        model = MackChainLadder().fit(raa_triangle)
        result = model.total_summary()
        assert result.total_reserve_stddev == pytest.approx(
            expected_total_std, rel=1e-6
        )

    def test_total_summary_mean_matches_native_mack(self, raa_triangle):
        from bayesianchainladder.bootstrap import MackChainLadder

        mack_native = cl.MackChainladder().fit(raa_triangle)
        expected_mean = float(np.nansum(np.asarray(mack_native.ibnr_.values)))

        model = MackChainLadder().fit(raa_triangle)
        result = model.total_summary()
        assert result.total_reserve_mean == pytest.approx(expected_mean, rel=1e-6)

    def test_sample_reserves_uses_calibrated_total(self, raa_triangle):
        """sample_reserves() should draw from Normal(total_mean,
        total_mack_std_err_), so the empirical std of a large sample should
        be close to total_mack_std_err_."""
        from bayesianchainladder.bootstrap import MackChainLadder

        mack_native = cl.MackChainladder().fit(raa_triangle)
        expected_total_std = float(
            np.asarray(mack_native.total_mack_std_err_).flatten()[0]
        )

        model = MackChainLadder(random_seed=42).fit(raa_triangle)
        samples = model.sample_reserves(n_samples=20000, random_seed=42)
        assert np.std(samples) == pytest.approx(expected_total_std, rel=0.05)

    def test_sample_reserves_is_finite(self, raa_triangle):
        from bayesianchainladder.bootstrap import MackChainLadder

        model = MackChainLadder().fit(raa_triangle)
        samples = model.sample_reserves(n_samples=500, random_seed=0)
        assert np.all(np.isfinite(samples))

    def test_unfit_summary_raises(self):
        from bayesianchainladder.bootstrap import MackChainLadder

        model = MackChainLadder()
        with pytest.raises(ValueError, match="has not been fitted"):
            model.summary()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_bootstrap.py -v
```

Expected: FAIL with `ImportError: cannot import name 'MackChainLadder'`.

- [ ] **Step 3: Create `bayesianchainladder/bootstrap.py` with `MackChainLadder`**

```python
"""Stochastic reserve wrappers around chainladder bootstrap/Mack methods.

This module provides three wrapper estimators that share the
:class:`bayesianchainladder.base.BaseStochasticReserve` interface:

- :class:`MackChainLadder`: wraps ``chainladder.MackChainladder`` and exposes
  reserve uncertainty as a normal approximation calibrated to Mack's
  ``total_mack_std_err_``.
- :class:`BootstrapODPChainLadder`: wraps ``chainladder.BootstrapODPSample``
  + ``chainladder.Chainladder`` and exposes the bootstrap reserve samples
  directly.
- :class:`CorrelatedBootstrapChainLadder`: wraps
  :class:`CorrelatedBootstrapODPSample` (Clark/Ding/Zhou 2022) — the same
  bootstrap with calendar-year correlation between cells via a Gaussian
  copula.

The low-level :class:`CorrelatedBootstrapODPSample` lives here too so the
file is self-contained.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import chainladder as cl
import numpy as np
import xarray as xr

from .base import BaseStochasticReserve, MethodSummary
from .utils import _extract_period_value, validate_triangle

if TYPE_CHECKING:
    pass


class MackChainLadder(BaseStochasticReserve):
    """Mack chain ladder wrapped with the shared stochastic reserve interface.

    Mack only produces per-origin mean + standard error; there is no native
    sample distribution. We populate ``reserves_posterior_`` by drawing
    independent ``Normal(mean_i, stderr_i)`` samples per origin, which gives
    correct *per-origin* marginals. The total of these per-origin draws,
    however, will understate the true total uncertainty because Mack's
    cross-origin covariance is not exposed by chainladder's public API.

    To compensate, ``sample_reserves()`` and ``total_summary()`` are
    overridden to draw from ``Normal(total_mean, total_mack_std_err_)``,
    where ``total_mack_std_err_`` is the calibrated total stderr that
    accounts for cross-origin correlation. The total row of ``summary()``
    is similarly recomputed using the calibrated total.

    Parameters
    ----------
    n_periods : int, default -1
        Forwarded to ``chainladder.MackChainladder``. ``-1`` uses all origins.
    random_seed : int, optional
        Seed used when drawing the per-origin Normal samples that populate
        ``reserves_posterior_``. ``sample_reserves()`` accepts its own seed.
    n_samples : int, default 5000
        Number of per-origin samples drawn into ``reserves_posterior_``.
    """

    def __init__(
        self,
        n_periods: int = -1,
        random_seed: int | None = None,
        n_samples: int = 5000,
    ) -> None:
        super().__init__()
        self.n_periods = n_periods
        self.random_seed = random_seed
        self.n_samples = n_samples
        self.total_reserve_mean_: float | None = None
        self.total_reserve_stddev_: float | None = None

    def fit(self, triangle):  # type: ignore[override]
        validate_triangle(triangle)
        self.triangle_ = triangle.copy()

        mack = cl.MackChainladder().fit(triangle)

        # Per-origin IBNR (sum across development) as a Triangle, then to ndarray
        ibnr_tri = mack.ibnr_.sum("development")
        ibnr_per_origin = np.asarray(ibnr_tri.values).flatten()

        # Per-origin Mack std error
        std_per_origin = np.asarray(mack.mack_std_err_.values).flatten()

        # Calibrated totals (used to override sample_reserves / total_summary)
        self.total_reserve_mean_ = float(np.nansum(ibnr_per_origin))
        self.total_reserve_stddev_ = float(
            np.asarray(mack.total_mack_std_err_).flatten()[0]
        )

        # Origins as integers, aligned with the per-origin arrays
        origins = [_extract_period_value(o) for o in ibnr_tri.origin]

        # Draw independent Normal samples per origin to populate
        # reserves_posterior_. These give correct per-origin marginals.
        rng = np.random.default_rng(self.random_seed)
        samples = np.empty((len(origins), self.n_samples))
        for i, (mean, std) in enumerate(zip(ibnr_per_origin, std_per_origin)):
            mean_clean = float(mean) if np.isfinite(mean) else 0.0
            std_clean = float(std) if np.isfinite(std) and std >= 0 else 0.0
            samples[i] = rng.normal(loc=mean_clean, scale=std_clean, size=self.n_samples)

        self.reserves_posterior_ = xr.DataArray(
            samples,
            dims=["origin", "sample"],
            coords={"origin": origins, "sample": np.arange(self.n_samples)},
        )

        self._build_reserve_summaries()
        self._is_fitted = True
        return self

    # -- Override total-level methods to use calibrated total stderr --

    def sample_reserves(  # type: ignore[override]
        self,
        n_samples: int = 1000,
        random_seed: int | None = None,
    ) -> np.ndarray:
        self._check_is_fitted()
        rng = np.random.default_rng(random_seed)
        return rng.normal(
            loc=self.total_reserve_mean_,
            scale=self.total_reserve_stddev_,
            size=n_samples,
        )

    def total_summary(self) -> MethodSummary:  # type: ignore[override]
        from scipy import stats as st

        self._check_is_fitted()
        mean = self.total_reserve_mean_
        stddev = self.total_reserve_stddev_
        q75, q90, q95 = st.norm.ppf([0.75, 0.90, 0.95], loc=mean, scale=stddev)
        return MethodSummary(
            total_reserve_mean=float(mean),
            total_reserve_stddev=float(stddev),
            total_reserve_75th_percentile=float(q75),
            total_reserve_90th_percentile=float(q90),
            total_reserve_95th_percentile=float(q95),
        )

    def summary(self, include_totals: bool = True):  # type: ignore[override]
        # Use the base summary for per-origin rows, then replace the total
        # row with one calibrated to total_mack_std_err_.
        result = super().summary(include_totals=False)
        if not include_totals:
            return result

        from scipy import stats as st
        import pandas as pd

        total_paid = float(self.ultimate_["paid_to_date"].sum())
        mean = float(self.total_reserve_mean_)
        stddev = float(self.total_reserve_stddev_)
        q05, q25, median, q75, q95 = st.norm.ppf(
            [0.05, 0.25, 0.50, 0.75, 0.95], loc=mean, scale=stddev
        )

        total_row = pd.DataFrame(
            {
                ("Ultimate", "paid_to_date"): [total_paid],
                ("Ultimate", "mean"): [total_paid + mean],
                ("Ultimate", "std"): [stddev],
                ("Ultimate", "median"): [total_paid + float(median)],
                ("IBNR", "mean"): [mean],
                ("IBNR", "std"): [stddev],
                ("IBNR", "median"): [float(median)],
            },
            index=["Total"],
        )
        return pd.concat([result, total_row])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_bootstrap.py -v
```

Expected: all tests in `TestMackChainLadder` PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add bayesianchainladder/bootstrap.py tests/test_bootstrap.py
git commit -m "$(cat <<'EOF'
Add MackChainLadder wrapper with calibrated total stderr

Wraps chainladder.MackChainladder behind BaseStochasticReserve.
Per-origin samples are drawn from independent Normal(mean, stderr).
Total methods (sample_reserves, total_summary, summary total row)
are overridden to use Mack's total_mack_std_err_ which accounts for
cross-origin correlation.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add `BootstrapODPChainLadder` wrapper

**Files:**
- Modify: `bayesianchainladder/bootstrap.py`
- Modify: `tests/test_bootstrap.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bootstrap.py`:

```python
@pytest.fixture
def genins_triangle():
    """GenIns triangle has all positive incrementals — works with bootstrap."""
    return cl.load_sample("genins")


class TestBootstrapODPChainLadder:
    def test_inherits_from_base(self):
        from bayesianchainladder.bootstrap import BootstrapODPChainLadder

        model = BootstrapODPChainLadder()
        assert isinstance(model, BaseStochasticReserve)

    def test_fit_returns_self(self, genins_triangle):
        from bayesianchainladder.bootstrap import BootstrapODPChainLadder

        model = BootstrapODPChainLadder(n_sims=100, random_seed=42)
        result = model.fit(genins_triangle)
        assert result is model
        assert model._is_fitted is True

    def test_fit_populates_reserves_posterior(self, genins_triangle):
        from bayesianchainladder.bootstrap import BootstrapODPChainLadder

        model = BootstrapODPChainLadder(n_sims=100, random_seed=42).fit(
            genins_triangle
        )
        assert model.reserves_posterior_ is not None
        # Expect dims (origin, sample) with sample size = n_sims
        assert "origin" in model.reserves_posterior_.dims
        sample_dims = [d for d in model.reserves_posterior_.dims if d != "origin"]
        sample_size = int(np.prod([model.reserves_posterior_.sizes[d]
                                   for d in sample_dims]))
        assert sample_size == 100

    def test_summary_has_total_row(self, genins_triangle):
        from bayesianchainladder.bootstrap import BootstrapODPChainLadder

        model = BootstrapODPChainLadder(n_sims=100, random_seed=42).fit(
            genins_triangle
        )
        summary = model.summary()
        assert "Total" in summary.index

    def test_total_summary_returns_finite(self, genins_triangle):
        from bayesianchainladder.bootstrap import BootstrapODPChainLadder

        model = BootstrapODPChainLadder(n_sims=200, random_seed=42).fit(
            genins_triangle
        )
        result = model.total_summary()
        assert np.isfinite(result.total_reserve_mean)
        assert np.isfinite(result.total_reserve_stddev)
        assert result.total_reserve_stddev > 0

    def test_total_summary_mean_close_to_native_chainladder(self, genins_triangle):
        """Bootstrap mean reserve should be close to the native chainladder
        IBNR (within ~5% with n_sims=500)."""
        from bayesianchainladder.bootstrap import BootstrapODPChainLadder

        cl_model = cl.Chainladder().fit(genins_triangle)
        cl_total = float(np.nansum(np.asarray(cl_model.ibnr_.values)))

        model = BootstrapODPChainLadder(n_sims=500, random_seed=42).fit(
            genins_triangle
        )
        boot_total = model.total_summary().total_reserve_mean

        assert boot_total == pytest.approx(cl_total, rel=0.05)

    def test_random_seed_makes_run_deterministic(self, genins_triangle):
        from bayesianchainladder.bootstrap import BootstrapODPChainLadder

        model_a = BootstrapODPChainLadder(n_sims=100, random_seed=7).fit(
            genins_triangle
        )
        model_b = BootstrapODPChainLadder(n_sims=100, random_seed=7).fit(
            genins_triangle
        )
        assert model_a.total_summary().total_reserve_mean == pytest.approx(
            model_b.total_summary().total_reserve_mean
        )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_bootstrap.py::TestBootstrapODPChainLadder -v
```

Expected: FAIL with `ImportError: cannot import name 'BootstrapODPChainLadder'`.

- [ ] **Step 3: Add `BootstrapODPChainLadder` to `bayesianchainladder/bootstrap.py`**

Append to the bottom of `bayesianchainladder/bootstrap.py` (after `MackChainLadder`):

```python
class BootstrapODPChainLadder(BaseStochasticReserve):
    """ODP bootstrap chain ladder wrapped with the shared interface.

    Wraps ``chainladder.BootstrapODPSample`` (resampling) followed by
    ``chainladder.Chainladder`` (deterministic chain ladder applied to each
    resample). The resulting per-simulation IBNR distribution is stored in
    ``reserves_posterior_``.

    Parameters
    ----------
    n_sims : int, default 1000
        Number of bootstrap simulations.
    n_periods : int, default -1
        Forwarded to ``chainladder.BootstrapODPSample``. ``-1`` uses all origins.
    hat_adj : bool, default True
        Hat-matrix adjustment per Shapland.
    random_seed : int, optional
        Seed for the bootstrap resampler.
    """

    def __init__(
        self,
        n_sims: int = 1000,
        n_periods: int = -1,
        hat_adj: bool = True,
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.n_sims = n_sims
        self.n_periods = n_periods
        self.hat_adj = hat_adj
        self.random_seed = random_seed

    def fit(self, triangle):  # type: ignore[override]
        validate_triangle(triangle)
        self.triangle_ = triangle.copy()

        # chainladder's BootstrapODPSample chokes when key_labels length
        # doesn't match the resampled kdims shape (e.g., the triangle
        # originated from a multi-index). Force a single-key layout.
        prepared = triangle.copy()
        prepared.key_labels = ["triangle_id"]
        prepared.kdims = np.asarray([["resample"]], dtype=object)

        sampler = cl.BootstrapODPSample(
            n_sims=self.n_sims,
            n_periods=self.n_periods,
            hat_adj=self.hat_adj,
            random_state=self.random_seed,
        ).fit(prepared)
        resampled = sampler.transform(prepared)
        model = cl.Chainladder().fit(resampled)

        # ibnr_.values has shape (n_sims, 1, n_origin, n_dev)
        ibnr_vals = np.asarray(model.ibnr_.values)
        # Sum across dev axis to get per-origin per-sim IBNR; may produce
        # shape (n_sims, 1, n_origin) → squeeze the singleton.
        per_sim_per_origin = np.nansum(ibnr_vals, axis=-1)
        per_sim_per_origin = np.squeeze(per_sim_per_origin)
        # Now expect shape (n_sims, n_origin); transpose to (origin, sample).
        if per_sim_per_origin.ndim == 1:
            per_sim_per_origin = per_sim_per_origin[None, :]
        per_origin_per_sim = per_sim_per_origin.T  # (n_origin, n_sims)

        origins = [_extract_period_value(o) for o in triangle.origin]

        self.reserves_posterior_ = xr.DataArray(
            per_origin_per_sim,
            dims=["origin", "sample"],
            coords={
                "origin": origins,
                "sample": np.arange(per_origin_per_sim.shape[1]),
            },
        )

        self._build_reserve_summaries()
        self._is_fitted = True
        return self
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_bootstrap.py::TestBootstrapODPChainLadder -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add bayesianchainladder/bootstrap.py tests/test_bootstrap.py
git commit -m "$(cat <<'EOF'
Add BootstrapODPChainLadder wrapper

Wraps chainladder.BootstrapODPSample + chainladder.Chainladder behind
BaseStochasticReserve. Includes the triangle_id/kdims workaround needed
for triangles built from multi-index sources.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Port `CorrelatedBootstrapODPSample` low-level class

**Files:**
- Modify: `bayesianchainladder/bootstrap.py`
- Modify: `tests/test_bootstrap.py`

- [ ] **Step 1: Write the failing smoke test**

Append to `tests/test_bootstrap.py`:

```python
class TestCorrelatedBootstrapODPSample:
    """Smoke tests for the low-level transformer."""

    def test_fit_with_rho_zero(self, genins_triangle):
        from bayesianchainladder.bootstrap import CorrelatedBootstrapODPSample

        sampler = CorrelatedBootstrapODPSample(
            n_sims=50, rho=0.0, random_state=42
        )
        sampler.fit(genins_triangle)
        # rho=0 path doesn't build a correlation matrix
        assert sampler.correlation_matrix_ is None
        assert sampler.scale_ is not None

    def test_fit_with_rho_positive_builds_correlation_matrix(self, genins_triangle):
        from bayesianchainladder.bootstrap import CorrelatedBootstrapODPSample

        sampler = CorrelatedBootstrapODPSample(
            n_sims=50, rho=0.5, random_state=42
        )
        sampler.fit(genins_triangle)
        assert sampler.correlation_matrix_ is not None
        # Diagonal should be 1
        diag = np.diag(sampler.correlation_matrix_)
        assert np.allclose(diag, 1.0)
        # Same-calendar-year off-diagonals should equal rho
        # (cell (0,1) and (1,0) are both calendar year 1)
        idx_a = sampler.valid_indices_.index((0, 1))
        idx_b = sampler.valid_indices_.index((1, 0))
        assert sampler.correlation_matrix_[idx_a, idx_b] == pytest.approx(0.5)

    def test_invalid_parametric_dist_raises(self):
        from bayesianchainladder.bootstrap import CorrelatedBootstrapODPSample

        with pytest.raises(ValueError, match="parametric_dist"):
            CorrelatedBootstrapODPSample(parametric_dist="weibull")

    def test_transform_produces_n_sims_resamples(self, genins_triangle):
        from bayesianchainladder.bootstrap import CorrelatedBootstrapODPSample

        sampler = CorrelatedBootstrapODPSample(
            n_sims=20, rho=0.3, random_state=42
        )
        sampler.fit(genins_triangle)
        resampled = sampler.transform(genins_triangle)
        # The resampled triangle's first dim should be n_sims
        assert resampled.values.shape[0] == 20
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_bootstrap.py::TestCorrelatedBootstrapODPSample -v
```

Expected: FAIL with `ImportError: cannot import name 'CorrelatedBootstrapODPSample'`.

- [ ] **Step 3: Append the low-level class to `bayesianchainladder/bootstrap.py`**

Add the following imports near the top of the file (next to the existing imports):

```python
import types
from warnings import warn

import pandas as pd
from scipy import stats
from scipy.linalg import cholesky

from chainladder.development import Development, DevelopmentBase
from chainladder.methods.chainladder import Chainladder
```

Then append at the bottom of `bayesianchainladder/bootstrap.py`:

```python
class CorrelatedBootstrapODPSample(DevelopmentBase):
    """Bootstrap sampler with calendar-year-correlated residuals.

    Implements the calendar-year correlation extension to the ODP bootstrap
    described in:

        Clark, D.R., Ding, H., and Zhou, L. (2022). "Making Bootstrap Reserve
        Ranges More Realistic." CAS E-Forum, Summer 2022.

    Cells on the same calendar-year diagonal have correlation ``rho``; the
    correlation decays multiplicatively for more distant diagonals
    (``rho``, ``rho^2``, ``rho^3``, ...). Correlation is induced via a
    Gaussian copula applied to either parametric (Normal/Lognormal) or
    nonparametric residual draws.

    Parameters
    ----------
    n_sims : int, default 1000
        Number of bootstrap simulations.
    n_periods : int, default -1
        Number of origin periods used in the LDF average; ``-1`` uses all.
    rho : float, default 0.0
        Same-diagonal correlation. ``0`` reproduces the standard independent
        bootstrap.
    parametric : bool, default True
        If True, parametric bootstrap (Normal or Lognormal multipliers on
        fitted values). If False, nonparametric (resamples residuals).
    parametric_dist : {"normal", "lognormal"}, default "normal"
        Parametric distribution choice when ``parametric=True``.
    hat_adj : bool, default True
        Apply Shapland's hat-matrix adjustment to standardised residuals.
    drop, drop_high, drop_low, drop_valuation
        Forwarded to ``chainladder.development.Development``.
    random_state : int or numpy.random.RandomState, optional
        Seed/state for reproducibility.
    min_fitted_value : float, default 1.0
        Floor on fitted incremental losses to keep residuals stable
        (``delta`` parameter, paper eq 2.1.4).

    Attributes
    ----------
    resampled_triangles_ : Triangle
        Bootstrap resamples (one per simulation).
    scale_ : float
        Dispersion (phi) for process risk.
    correlation_matrix_ : ndarray or None
        Calendar-year correlation matrix used by the Gaussian copula
        (``None`` when ``rho == 0``).
    cholesky_matrix_ : ndarray
        Cholesky factor of ``correlation_matrix_``.
    valid_indices_ : list[tuple[int, int]]
        ``(origin_idx, dev_idx)`` for each non-NaN cell.
    """

    def __init__(
        self,
        n_sims: int = 1000,
        n_periods: int = -1,
        rho: float = 0.0,
        parametric: bool = True,
        parametric_dist: str = "normal",
        hat_adj: bool = True,
        drop=None,
        drop_high=None,
        drop_low=None,
        drop_valuation=None,
        random_state=None,
        min_fitted_value: float = 1.0,
    ) -> None:
        if parametric_dist not in ("normal", "lognormal"):
            raise ValueError("parametric_dist must be 'normal' or 'lognormal'")
        self.n_sims = n_sims
        self.n_periods = n_periods
        self.rho = rho
        self.parametric = parametric
        self.parametric_dist = parametric_dist
        self.hat_adj = hat_adj
        self.drop = drop
        self.drop_high = drop_high
        self.drop_low = drop_low
        self.drop_valuation = drop_valuation
        self.random_state = random_state
        self.min_fitted_value = min_fitted_value

    # ----- correlation matrix construction -----

    def _build_full_correlation_matrix(self, n_origin, n_dev, nan_triangle, xp=np):
        valid_indices = []
        for i in range(n_origin):
            for j in range(n_dev):
                if not np.isnan(nan_triangle[i, j]):
                    valid_indices.append((i, j))

        n_cells = len(valid_indices)
        corr_matrix = xp.eye(n_cells)
        for idx1, (i1, j1) in enumerate(valid_indices):
            cy1 = i1 + j1
            for idx2, (i2, j2) in enumerate(valid_indices):
                if idx1 == idx2:
                    continue
                cy2 = i2 + j2
                cy_diff = abs(cy1 - cy2)
                if cy_diff == 0:
                    corr_matrix[idx1, idx2] = self.rho
                else:
                    corr_matrix[idx1, idx2] = self.rho ** (cy_diff + 1)
        return corr_matrix, valid_indices

    def _generate_correlated_uniforms(
        self, n_cells, n_sims, corr_matrix, random_state, xp=np
    ):
        try:
            L = cholesky(corr_matrix, lower=True)
        except np.linalg.LinAlgError:
            eps = 1e-6
            L = cholesky(corr_matrix + eps * np.eye(n_cells), lower=True)
        self.cholesky_matrix_ = L
        Z = random_state.standard_normal(size=(n_sims, n_cells))
        correlated_normals = Z @ L.T
        return stats.norm.cdf(correlated_normals)

    # ----- fit / transform -----

    def fit(self, X, y=None, sample_weight=None):  # noqa: D401
        if X.shape[1] > 1:
            from chainladder.utils.utility_functions import concat
            out = [
                CorrelatedBootstrapODPSample(**self.get_params()).fit(X.iloc[:, i])
                for i in range(X.shape[1])
            ]
            xp = X.get_array_module(out[0].design_matrix_)
            self.design_matrix_ = xp.concatenate(
                [i.design_matrix_[None] for i in out], axis=0
            )
            self.hat_ = xp.concatenate([i.hat_[None] for i in out], axis=0)
            self.resampled_triangles_ = concat(
                [i.resampled_triangles_ for i in out], axis=1
            )
            self.scale_ = xp.array([i.scale_ for i in out])
            self.w_ = out[0].w_
            self.correlation_matrix_ = out[0].correlation_matrix_
            return self

        backend = X.array_backend
        X = X.set_backend("numpy") if backend == "sparse" else X.copy()
        xp = X.get_array_module()

        if len(X) != 1:
            raise ValueError("Only single index triangles are supported")
        if not isinstance(X.ddims, np.ndarray):
            raise ValueError("Triangle must be expressed with development lags")

        obj = Development(
            n_periods=self.n_periods,
            drop=self.drop,
            drop_high=self.drop_high,
            drop_low=self.drop_low,
            drop_valuation=self.drop_valuation,
        ).fit_transform(X)
        self.w_ = obj.w_

        obj = Chainladder().fit(obj)
        exp_incr_triangle = obj.full_expectation_.cum_to_incr().values[
            0, 0, :, : X.shape[-1]
        ]
        exp_incr_triangle = xp.nan_to_num(exp_incr_triangle) * obj.X_.nan_triangle

        self.design_matrix_ = self._get_design_matrix(X)

        if self.hat_adj:
            try:
                self.hat_ = self._get_hat(X, exp_incr_triangle)
            except Exception:
                warn("Could not compute hat matrix. Setting hat_adj to False")
                self.hat_adj = False
                self.hat_ = None
        else:
            self.hat_ = None

        n_origin, n_dev = X.shape[2], X.shape[3]
        nan_triangle = obj.X_.nan_triangle

        if self.rho != 0:
            self.correlation_matrix_, self.valid_indices_ = (
                self._build_full_correlation_matrix(
                    n_origin, n_dev, nan_triangle, xp
                )
            )
        else:
            self.correlation_matrix_ = None
            self.valid_indices_ = None

        self.resampled_triangles_, self.scale_ = self._get_simulation(
            X, exp_incr_triangle, nan_triangle
        )
        return self

    def _get_simulation(self, X, exp_incr_triangle, nan_triangle):
        xp = X.get_array_module()
        fitted_for_resid = xp.maximum(xp.abs(exp_incr_triangle), self.min_fitted_value)
        unscaled_residuals = (
            (X.cum_to_incr().values - exp_incr_triangle) / xp.sqrt(fitted_for_resid)
        )[0, 0, ...]

        w_ = self.w_[0, 0]
        w_expanded = xp.ones_like(unscaled_residuals)
        w_expanded[:, 1:] = w_[:, :] * w_[:, :]
        unscaled_residuals = unscaled_residuals * w_expanded

        pearson_chi_sq = xp.nansum(unscaled_residuals ** 2)
        if self.hat_ is not None:
            standardized_residuals = self.hat_ * unscaled_residuals
        else:
            standardized_residuals = unscaled_residuals

        n_params = self.design_matrix_.shape[1]
        degree_freedom = xp.nansum(nan_triangle) - n_params
        scale_phi = pearson_chi_sq / degree_freedom

        resids_flat = standardized_residuals.flatten()
        adj_resid_dist = resids_flat[np.isfinite(resids_flat)]
        adj_resid_dist = adj_resid_dist[adj_resid_dist != 0]
        adj_resid_dist = adj_resid_dist - xp.mean(adj_resid_dist)

        if isinstance(self.random_state, np.random.RandomState):
            random_state = self.random_state
        else:
            random_state = np.random.RandomState(self.random_state)

        if self.rho != 0 and self.correlation_matrix_ is not None:
            resampled_triangles = self._generate_correlated_samples(
                X, exp_incr_triangle, nan_triangle, adj_resid_dist,
                scale_phi, random_state, xp,
            )
        else:
            resampled_triangles = self._generate_independent_samples(
                X, exp_incr_triangle, adj_resid_dist, random_state, xp,
            )

        obj = X.copy()
        obj.kdims = np.arange(self.n_sims)
        obj.values = resampled_triangles
        obj._set_slicers()
        return obj, scale_phi

    def _generate_independent_samples(
        self, X, exp_incr_triangle, adj_resid_dist, random_state, xp
    ):
        if self.parametric:
            return self._generate_parametric_samples(
                X, exp_incr_triangle, random_state, xp
            )

        resampled_residual = [
            (
                random_state.choice(
                    adj_resid_dist, size=exp_incr_triangle.shape, replace=True
                )
                * (exp_incr_triangle * 0 + 1)
            )[None, ...]
            for _ in range(self.n_sims)
        ]
        resampled_residual = xp.concatenate(tuple(resampled_residual), 0)
        b = xp.repeat(exp_incr_triangle[None, ...], self.n_sims, 0)
        resampled_incr = resampled_residual * xp.sqrt(xp.abs(b)) + b
        resampled_triangles = resampled_incr.cumsum(axis=2)
        return xp.swapaxes(resampled_triangles[None, ...], 0, 1)

    def _generate_parametric_samples(
        self, X, exp_incr_triangle, random_state, xp
    ):
        n_params = self.design_matrix_.shape[1]
        nan_triangle = X.nan_triangle
        degree_freedom = xp.nansum(nan_triangle) - n_params

        fitted_safe = xp.maximum(xp.abs(exp_incr_triangle), self.min_fitted_value)
        actual_incr = X.cum_to_incr().values[0, 0, ...]
        resid_sq = ((actual_incr - exp_incr_triangle) ** 2) / fitted_safe
        phi = xp.nansum(resid_sq) / degree_freedom

        std_dev = xp.sqrt(phi * fitted_safe)

        if self.parametric_dist == "normal":
            z = random_state.standard_normal(
                size=(self.n_sims,) + exp_incr_triangle.shape
            )
            resampled_incr = exp_incr_triangle + std_dev * z
        else:
            cv = std_dev / fitted_safe
            sigma_sq = xp.log(1 + cv ** 2)
            mu = -sigma_sq / 2
            sigma = xp.sqrt(sigma_sq)
            z = random_state.standard_normal(
                size=(self.n_sims,) + exp_incr_triangle.shape
            )
            multipliers = xp.exp(mu + sigma * z)
            resampled_incr = exp_incr_triangle * multipliers

        resampled_triangles = resampled_incr.cumsum(axis=2)
        return xp.swapaxes(resampled_triangles[None, ...], 0, 1)

    def _generate_correlated_samples(
        self, X, exp_incr_triangle, nan_triangle, adj_resid_dist, scale_phi,
        random_state, xp,
    ):
        n_cells = len(self.valid_indices_)
        correlated_uniforms = self._generate_correlated_uniforms(
            n_cells, self.n_sims, self.correlation_matrix_, random_state, xp
        )
        if self.parametric:
            return self._generate_correlated_parametric(
                X, exp_incr_triangle, nan_triangle, correlated_uniforms,
                scale_phi, random_state, xp,
            )
        return self._generate_correlated_nonparametric(
            X, exp_incr_triangle, nan_triangle, correlated_uniforms,
            adj_resid_dist, random_state, xp,
        )

    def _generate_correlated_parametric(
        self, X, exp_incr_triangle, nan_triangle, correlated_uniforms,
        scale_phi, random_state, xp,
    ):
        n_origin, n_dev = exp_incr_triangle.shape
        n_params = self.design_matrix_.shape[1]
        degree_freedom = xp.nansum(nan_triangle) - n_params
        fitted_safe = xp.maximum(xp.abs(exp_incr_triangle), self.min_fitted_value)
        actual_incr = X.cum_to_incr().values[0, 0, ...]
        resid_sq = ((actual_incr - exp_incr_triangle) ** 2) / fitted_safe
        phi = xp.nansum(resid_sq) / degree_freedom

        resampled_incr = xp.zeros((self.n_sims, n_origin, n_dev))
        for cell_idx, (i, j) in enumerate(self.valid_indices_):
            fitted_val = fitted_safe[i, j]
            std_dev = xp.sqrt(phi * fitted_val)
            z = stats.norm.ppf(correlated_uniforms[:, cell_idx])
            if self.parametric_dist == "normal":
                resampled_incr[:, i, j] = exp_incr_triangle[i, j] + std_dev * z
            else:
                cv = std_dev / fitted_val
                sigma_sq = xp.log(1 + cv ** 2)
                mu = -sigma_sq / 2
                sigma = xp.sqrt(sigma_sq)
                multiplier = xp.exp(mu + sigma * z)
                resampled_incr[:, i, j] = exp_incr_triangle[i, j] * multiplier

        for i in range(n_origin):
            for j in range(n_dev):
                if (i, j) not in self.valid_indices_:
                    resampled_incr[:, i, j] = xp.nan

        resampled_triangles = xp.nancumsum(resampled_incr, axis=2)
        return xp.swapaxes(resampled_triangles[None, ...], 0, 1)

    def _generate_correlated_nonparametric(
        self, X, exp_incr_triangle, nan_triangle, correlated_uniforms,
        adj_resid_dist, random_state, xp,
    ):
        n_origin, n_dev = exp_incr_triangle.shape
        sorted_resids = xp.sort(adj_resid_dist)
        n_resids = len(sorted_resids)

        resampled_incr = xp.zeros((self.n_sims, n_origin, n_dev))
        for cell_idx, (i, j) in enumerate(self.valid_indices_):
            indices = (correlated_uniforms[:, cell_idx] * n_resids).astype(int)
            indices = xp.clip(indices, 0, n_resids - 1)
            selected_resids = sorted_resids[indices]
            fitted_val = exp_incr_triangle[i, j]
            resampled_incr[:, i, j] = (
                selected_resids * xp.sqrt(xp.abs(fitted_val)) + fitted_val
            )

        for i in range(n_origin):
            for j in range(n_dev):
                if (i, j) not in self.valid_indices_:
                    resampled_incr[:, i, j] = xp.nan

        resampled_triangles = xp.nancumsum(resampled_incr, axis=2)
        return xp.swapaxes(resampled_triangles[None, ...], 0, 1)

    # ----- design / hat matrix -----

    def _get_design_matrix(self, X):
        xp = X.get_array_module()
        w = X.nan_triangle
        arr = xp.diag(w[:, 0])
        intra_beta = xp.zeros((w.shape[0], w.shape[1] - 1))
        arr = xp.concatenate((arr, intra_beta), axis=1)
        for i in range(w.shape[1] - 1):
            len_alpha = len(w[:, i + 1][~xp.isnan(w[:, i + 1])])
            intra_alpha = xp.diag(w[:, i + 1])[:len_alpha, :]
            intra_beta[:, i] = 1
            intra_beta = intra_beta[:len_alpha, :]
            intra_arr = xp.concatenate((intra_alpha, intra_beta), axis=1)
            arr = xp.concatenate((arr, intra_arr), axis=0)
        return arr

    def _get_hat(self, X, exp_incr_triangle):
        xp = X.get_array_module()
        weight_matrix = xp.diag(
            pd.DataFrame(exp_incr_triangle).unstack().dropna().values
        )
        design_matrix = self.design_matrix_
        hat = xp.matmul(
            xp.matmul(
                xp.matmul(
                    design_matrix,
                    xp.linalg.inv(
                        xp.matmul(
                            design_matrix.T,
                            xp.matmul(weight_matrix, design_matrix),
                        )
                    ),
                ),
                design_matrix.T,
            ),
            weight_matrix,
        )
        hat = xp.diagonal(
            xp.sqrt(xp.divide(1, abs(1 - hat), where=(1 - hat) != 0))
        )
        total_length = X.nan_triangle.shape[0]
        reshaped_hat = xp.reshape(hat[:total_length], (1, total_length))
        indices = xp.nansum(X.nan_triangle, axis=0).cumsum().astype(int)
        for num, _ in enumerate(indices[:-1]):
            col_length = int(indices[num + 1] - indices[num])
            col = xp.reshape(
                hat[int(indices[num]) : int(indices[num + 1])], (1, col_length)
            )
            nans = xp.repeat(
                xp.array([xp.nan])[None, :], total_length - col_length, axis=1
            )
            col = xp.concatenate((col, nans), axis=1)
            reshaped_hat = xp.concatenate((reshaped_hat, col), axis=0)
        return reshaped_hat.T

    def transform(self, X):
        X_new = self.resampled_triangles_.copy()
        n_keys = len(X.key_labels)
        if n_keys == 1:
            X_new.kdims = np.array([[str(i)] for i in range(self.n_sims)])
        else:
            original_kdims = (
                X.kdims[0] if len(X.kdims.shape) > 1 else X.kdims
            )
            X_new.kdims = np.array(
                [
                    [
                        f"{original_kdims[j]}_{i}" if j == 0 else original_kdims[j]
                        for j in range(n_keys)
                    ]
                    for i in range(self.n_sims)
                ]
            )
        X_new.key_labels = X.key_labels
        X_new.scale_ = self.scale_
        X_new.random_state = self.random_state
        X_new.rho_ = self.rho
        X_new._get_process_variance = types.MethodType(_get_process_variance, X_new)
        return X_new


def _get_process_variance(self, full_triangle):
    """Inject random gamma process noise into the lower-right (future) cells."""
    xp = full_triangle.get_array_module()
    lower_tri = full_triangle.cum_to_incr() - self.cum_to_incr()
    random_state = xp.random.RandomState(
        None if not self.random_state else self.random_state + 1
    )
    lower_tri.values = random_state.gamma(
        shape=abs(lower_tri.values) / self.scale_, scale=self.scale_
    ) * xp.sign(xp.nan_to_num(lower_tri.values))
    return (lower_tri + self.cum_to_incr()).incr_to_cum()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_bootstrap.py::TestCorrelatedBootstrapODPSample -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add bayesianchainladder/bootstrap.py tests/test_bootstrap.py
git commit -m "$(cat <<'EOF'
Port CorrelatedBootstrapODPSample low-level transformer

Calendar-year-correlated ODP bootstrap (Clark/Ding/Zhou 2022) ported
into the package. Builds correlation via a Gaussian copula over a
block-AR(1) calendar-year structure and supports parametric (Normal/
Lognormal) or nonparametric residual draws. Dropped the unused
_build_correlation_matrix dead code from the source file.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Add `CorrelatedBootstrapChainLadder` wrapper

**Files:**
- Modify: `bayesianchainladder/bootstrap.py`
- Modify: `tests/test_bootstrap.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_bootstrap.py`:

```python
class TestCorrelatedBootstrapChainLadder:
    def test_inherits_from_base(self):
        from bayesianchainladder.bootstrap import CorrelatedBootstrapChainLadder

        model = CorrelatedBootstrapChainLadder()
        assert isinstance(model, BaseStochasticReserve)

    def test_fit_returns_self(self, genins_triangle):
        from bayesianchainladder.bootstrap import CorrelatedBootstrapChainLadder

        model = CorrelatedBootstrapChainLadder(
            n_sims=100, rho=0.3, random_seed=42
        )
        result = model.fit(genins_triangle)
        assert result is model
        assert model._is_fitted is True

    def test_summary_has_total_row(self, genins_triangle):
        from bayesianchainladder.bootstrap import CorrelatedBootstrapChainLadder

        model = CorrelatedBootstrapChainLadder(
            n_sims=200, rho=0.3, random_seed=42
        ).fit(genins_triangle)
        summary = model.summary()
        assert "Total" in summary.index

    def test_rho_zero_matches_independent_bootstrap(self, genins_triangle):
        """With rho=0, the correlated wrapper should produce the same total
        std as the independent BootstrapODPChainLadder when seeded
        identically."""
        from bayesianchainladder.bootstrap import (
            BootstrapODPChainLadder,
            CorrelatedBootstrapChainLadder,
        )

        # Note: not exact match because the underlying samplers differ in
        # implementation, but the totals should be in the same ballpark.
        indep = BootstrapODPChainLadder(n_sims=500, random_seed=42).fit(
            genins_triangle
        )
        corr = CorrelatedBootstrapChainLadder(
            n_sims=500, rho=0.0, random_seed=42
        ).fit(genins_triangle)

        indep_std = indep.total_summary().total_reserve_stddev
        corr_std = corr.total_summary().total_reserve_stddev
        assert corr_std == pytest.approx(indep_std, rel=0.20)

    def test_rho_positive_increases_total_std(self, genins_triangle):
        """Higher rho should produce a wider total reserve distribution."""
        from bayesianchainladder.bootstrap import CorrelatedBootstrapChainLadder

        low = CorrelatedBootstrapChainLadder(
            n_sims=500, rho=0.0, random_seed=42
        ).fit(genins_triangle)
        high = CorrelatedBootstrapChainLadder(
            n_sims=500, rho=0.5, random_seed=42
        ).fit(genins_triangle)

        assert (
            high.total_summary().total_reserve_stddev
            > low.total_summary().total_reserve_stddev
        )

    def test_lognormal_distribution_runs(self, genins_triangle):
        from bayesianchainladder.bootstrap import CorrelatedBootstrapChainLadder

        model = CorrelatedBootstrapChainLadder(
            n_sims=200,
            rho=0.3,
            parametric_dist="lognormal",
            random_seed=42,
        ).fit(genins_triangle)
        result = model.total_summary()
        assert np.isfinite(result.total_reserve_mean)
        assert result.total_reserve_stddev > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_bootstrap.py::TestCorrelatedBootstrapChainLadder -v
```

Expected: FAIL with `ImportError: cannot import name 'CorrelatedBootstrapChainLadder'`.

- [ ] **Step 3: Add `CorrelatedBootstrapChainLadder` to `bayesianchainladder/bootstrap.py`**

Append to the bottom of `bayesianchainladder/bootstrap.py`:

```python
class CorrelatedBootstrapChainLadder(BaseStochasticReserve):
    """Correlated ODP bootstrap chain ladder behind the shared interface.

    Wraps :class:`CorrelatedBootstrapODPSample` (calendar-year-correlated
    bootstrap, Clark/Ding/Zhou 2022) followed by
    ``chainladder.Chainladder``. Identical surface to
    :class:`BootstrapODPChainLadder`, plus correlation parameters.

    Parameters
    ----------
    n_sims : int, default 1000
    rho : float, default 0.0
        Same-diagonal correlation (``0`` reduces to independent ODP bootstrap).
    parametric : bool, default True
    parametric_dist : {"normal", "lognormal"}, default "normal"
    hat_adj : bool, default True
    n_periods : int, default -1
    random_seed : int, optional
    """

    def __init__(
        self,
        n_sims: int = 1000,
        rho: float = 0.0,
        parametric: bool = True,
        parametric_dist: str = "normal",
        hat_adj: bool = True,
        n_periods: int = -1,
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.n_sims = n_sims
        self.rho = rho
        self.parametric = parametric
        self.parametric_dist = parametric_dist
        self.hat_adj = hat_adj
        self.n_periods = n_periods
        self.random_seed = random_seed

    def fit(self, triangle):  # type: ignore[override]
        validate_triangle(triangle)
        self.triangle_ = triangle.copy()

        prepared = triangle.copy()
        prepared.key_labels = ["triangle_id"]
        prepared.kdims = np.asarray([["resample"]], dtype=object)

        sampler = CorrelatedBootstrapODPSample(
            n_sims=self.n_sims,
            rho=self.rho,
            parametric=self.parametric,
            parametric_dist=self.parametric_dist,
            hat_adj=self.hat_adj,
            n_periods=self.n_periods,
            random_state=self.random_seed,
        ).fit(prepared)
        resampled = sampler.transform(prepared)
        model = Chainladder().fit(resampled)

        ibnr_vals = np.asarray(model.ibnr_.values)
        per_sim_per_origin = np.nansum(ibnr_vals, axis=-1)
        per_sim_per_origin = np.squeeze(per_sim_per_origin)
        if per_sim_per_origin.ndim == 1:
            per_sim_per_origin = per_sim_per_origin[None, :]
        per_origin_per_sim = per_sim_per_origin.T

        origins = [_extract_period_value(o) for o in triangle.origin]

        self.reserves_posterior_ = xr.DataArray(
            per_origin_per_sim,
            dims=["origin", "sample"],
            coords={
                "origin": origins,
                "sample": np.arange(per_origin_per_sim.shape[1]),
            },
        )

        self._build_reserve_summaries()
        self._is_fitted = True
        return self
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_bootstrap.py::TestCorrelatedBootstrapChainLadder -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add bayesianchainladder/bootstrap.py tests/test_bootstrap.py
git commit -m "$(cat <<'EOF'
Add CorrelatedBootstrapChainLadder wrapper

High-level wrapper that pairs CorrelatedBootstrapODPSample with
chainladder.Chainladder behind BaseStochasticReserve. Total stddev
increases with rho as expected (verified by test).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Retrofit `BayesianChainLadderGLM` to inherit from base

**Files:**
- Modify: `bayesianchainladder/estimators.py:33-1019` (the `BayesianChainLadderGLM` class)

**Goal:** Make `BayesianChainLadderGLM` inherit from `BaseStochasticReserve` and remove the duplicated `summary()`, `sample_reserves()`, `_compute_reserve_summaries()` logic. Existing tests must continue to pass.

**Key implementation points:**
- Standardize `reserves_posterior_` dim layout to `(origin, sample)` by stacking `chain × draw`. The base class expects an `origin` dim plus arbitrary remaining dims, so this also keeps existing posterior-access code working.
- Delete the inline `_compute_reserve_summaries` (~90 lines around `estimators.py:383-474`); the base method does the same work.
- Delete the inline `summary()` (~70 lines around `estimators.py:533-598`); use the inherited one.
- Delete the inline `sample_reserves()` (~40 lines around `estimators.py:660-698`); use the inherited one.
- Keep `_check_is_fitted` only if it differs from the base; otherwise remove.

- [ ] **Step 1: Run existing GLM tests to set a baseline**

```bash
uv run pytest tests/test_estimators.py -v -k "BayesianChainLadderGLM"
```

Note which tests pass under the current implementation. The retrofit must not regress any of these.

- [ ] **Step 2: Modify the class signature and remove duplicated methods**

In `bayesianchainladder/estimators.py`, make these changes:

1. Add the import near the top:
   ```python
   from .base import BaseStochasticReserve
   ```

2. Change the class declaration:
   ```python
   class BayesianChainLadderGLM(BaseStochasticReserve):
   ```

3. In `__init__`, replace the manual attribute initialization with a `super().__init__()` call followed by the param-storing assignments. The fitted attributes (`triangle_`, `ibnr_`, `ultimate_`, `reserves_posterior_`, `_is_fitted`) come from the base; remove their assignments here. Keep `model_`, `idata`, `data_`, `future_data_`, `fitted_` since those are GLM-specific. Final `__init__`:

   ```python
   def __init__(
       self,
       formula: str = "incremental ~ 1 + C(origin) + C(dev)",
       family: str = "negativebinomial",
       link: str | None = None,
       exposure: str | None = None,
       priors: dict[str, Any] | None = None,
       draws: int = 2000,
       tune: int = 1000,
       chains: int = 4,
       target_accept: float = 0.9,
       random_seed: int | None = None,
       backend: str = "bambi",
   ):
       super().__init__()
       self.formula = formula
       self.family = family
       self.link = link
       self.exposure = exposure
       self.priors = priors
       self.draws = draws
       self.tune = tune
       self.chains = chains
       self.target_accept = target_accept
       self.random_seed = random_seed
       self.backend = backend

       # GLM-specific fitted attributes (not in base)
       self.model_: bmb.Model | None = None
       self.idata: az.InferenceData | None = None
       self.data_: pd.DataFrame | None = None
       self.future_data_: pd.DataFrame | None = None
       self.fitted_: pd.DataFrame | None = None
   ```

4. In `_compute_reserves` (around `estimators.py:328-381`), after `self.reserves_posterior_ = xr.concat(...)`, stack the chain/draw dims into a single `sample` dim:

   Find:
   ```python
   self.reserves_posterior_ = xr.concat(
       reserves_list, dim=pd.Index(origins, name="origin")
   )

   # Compute ultimate and IBNR summaries
   self._compute_reserve_summaries()
   ```

   Replace with:
   ```python
   reserves_posterior = xr.concat(
       reserves_list, dim=pd.Index(origins, name="origin")
   )
   # Standardize to (origin, sample) for the base class helper
   self.reserves_posterior_ = (
       reserves_posterior
       .stack(sample=["chain", "draw"])
       .reset_index("sample", drop=True)
   )

   # Compute ultimate and IBNR summaries (helper now lives in the base)
   self._build_reserve_summaries()
   ```

5. Delete `_compute_reserve_summaries` (the entire method body, around `estimators.py:383-474`).

6. Delete `summary` (around `estimators.py:533-598`).

7. Delete `sample_reserves` (around `estimators.py:660-698`).

8. Delete the local `_check_is_fitted` (around `estimators.py:956-961`) — it duplicates the base.

- [ ] **Step 3: Run GLM tests to verify everything still passes**

```bash
uv run pytest tests/test_estimators.py -v -k "BayesianChainLadderGLM"
```

Expected: same set of tests pass as the baseline from Step 1. If a test fails specifically because it was inspecting `chain` or `draw` dims of `reserves_posterior_`, update the test to use `sample` instead — the dim layout standardization is intentional.

- [ ] **Step 4: Run the full test suite to confirm no other regressions**

```bash
uv run pytest -v
```

Expected: same overall pass count as before (87 fast tests pass, 44 slow skipped).

- [ ] **Step 5: Commit**

```bash
git add bayesianchainladder/estimators.py tests/
git commit -m "$(cat <<'EOF'
Retrofit BayesianChainLadderGLM onto BaseStochasticReserve

GLM now inherits summary(), sample_reserves(), and the per-origin
ibnr_/ultimate_ assembly from the base. reserves_posterior_ is
standardized to dims (origin, sample) by stacking chain x draw.
~150 lines of duplicated DataFrame plumbing removed.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Retrofit `BayesianCSR` to inherit from base

**Files:**
- Modify: `bayesianchainladder/estimators.py:1022-1655` (the `BayesianCSR` class)

**Goal:** Same retrofit as Task 7 for `BayesianCSR`. Existing CSR tests must keep passing.

**Key points:**
- `BayesianCSR.reserves_posterior_` already uses dims `(origin, sample)` (see `estimators.py:1442-1454`). No layout change needed.
- Remove duplicated `_compute_reserve_summaries` (~90 lines around `estimators.py:1348-1440`).
- Remove duplicated `summary` (~60 lines around `estimators.py:1456-1517`).
- Remove duplicated `sample_reserves` (~35 lines around `estimators.py:1599-1636`).
- Remove duplicated `_check_is_fitted`.

- [ ] **Step 1: Run existing CSR tests to set a baseline**

```bash
uv run pytest tests/test_estimators.py -v -k "BayesianCSR"
```

Note which tests pass.

- [ ] **Step 2: Modify the class**

1. Change the class declaration:
   ```python
   class BayesianCSR(BaseStochasticReserve):
   ```

2. In `__init__`, replace the inline attribute init with `super().__init__()`. Keep CSR-specific attributes (`model_`, `idata`, `data_`, `future_data_`, `elr_posterior_`, `gamma_posterior_`). Final `__init__`:

   ```python
   def __init__(
       self,
       priors: dict[str, Any] | None = None,
       draws: int = 2000,
       tune: int = 1000,
       chains: int = 4,
       target_accept: float = 0.9,
       random_seed: int | None = None,
       include_process_variance: bool = True,
   ):
       super().__init__()
       self.priors = priors
       self.draws = draws
       self.tune = tune
       self.chains = chains
       self.target_accept = target_accept
       self.random_seed = random_seed
       self.include_process_variance = include_process_variance

       # CSR-specific fitted attributes (not in base)
       self.model_: pm.Model | None = None
       self.idata: az.InferenceData | None = None
       self.data_: pd.DataFrame | None = None
       self.future_data_: pd.DataFrame | None = None
       self.elr_posterior_: xr.DataArray | None = None
       self.gamma_posterior_: xr.DataArray | None = None
   ```

3. In `_compute_reserve_summaries` (the existing CSR-specific implementation), the dictionary-keyed signature differs from the base. Rename the existing method to `_build_per_origin_arrays` (or inline its logic into `_compute_predictions`), then call `self._build_reserve_summaries()` from the base after `reserves_posterior_` is set. The simpler path:

   In `_compute_predictions`, after the dict `all_predictions` is built, replace the call:
   ```python
   if all_predictions:
       self._compute_reserve_summaries(all_predictions)
   ```
   with code that:
   - Stacks the IBNR samples into the `(origin, sample)` `xr.DataArray` (this is already done at the bottom of the existing `_compute_reserve_summaries`).
   - Calls `self._build_reserve_summaries()` from the base.

   Concretely, replace the existing `_compute_reserve_summaries` method body with a smaller helper that only sets `reserves_posterior_` and then delegates:

   ```python
   def _compute_reserve_summaries(
       self, future_predictions: dict[Any, dict[str, np.ndarray]]
   ) -> None:
       """Build reserves_posterior_ from per-origin samples and delegate
       to the base class for ibnr_/ultimate_ assembly."""
       origins = sorted(future_predictions.keys())
       ibnr_samples_list = [
           future_predictions[origin]["ibnr_samples"].flatten() for origin in origins
       ]
       ibnr_array = np.stack(ibnr_samples_list, axis=0)  # (n_origin, n_samples)

       self.reserves_posterior_ = xr.DataArray(
           ibnr_array,
           dims=["origin", "sample"],
           coords={
               "origin": origins,
               "sample": np.arange(ibnr_array.shape[1]),
           },
       )

       # Base class builds ibnr_ / ultimate_ tables
       self._build_reserve_summaries()
   ```

4. Delete the inline `summary` method (~60 lines around `estimators.py:1456-1517`).

5. Delete the inline `sample_reserves` method (~35 lines around `estimators.py:1599-1636`).

6. Delete the inline `_check_is_fitted` (it duplicates the base).

- [ ] **Step 3: Run CSR tests**

```bash
uv run pytest tests/test_estimators.py -v -k "BayesianCSR"
```

Expected: same set of CSR tests pass as before.

- [ ] **Step 4: Run the full test suite**

```bash
uv run pytest -v
```

Expected: full test count unchanged (existing pass/skip counts preserved).

- [ ] **Step 5: Commit**

```bash
git add bayesianchainladder/estimators.py
git commit -m "$(cat <<'EOF'
Retrofit BayesianCSR onto BaseStochasticReserve

CSR now inherits summary(), sample_reserves(), and per-origin table
assembly from the base. _compute_reserve_summaries shrinks to a thin
adapter that builds reserves_posterior_ then delegates to the base.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Update `__init__.py` exports

**Files:**
- Modify: `bayesianchainladder/__init__.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_base.py`:

```python
class TestPackageExports:
    def test_base_class_is_exported(self):
        import bayesianchainladder as bcl

        assert hasattr(bcl, "BaseStochasticReserve")
        assert hasattr(bcl, "MethodSummary")

    def test_new_estimators_are_exported(self):
        import bayesianchainladder as bcl

        assert hasattr(bcl, "MackChainLadder")
        assert hasattr(bcl, "BootstrapODPChainLadder")
        assert hasattr(bcl, "CorrelatedBootstrapChainLadder")
        assert hasattr(bcl, "CorrelatedBootstrapODPSample")

    def test_existing_estimators_still_exported(self):
        import bayesianchainladder as bcl

        assert hasattr(bcl, "BayesianChainLadderGLM")
        assert hasattr(bcl, "BayesianCSR")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_base.py::TestPackageExports -v
```

Expected: 2 of 3 tests FAIL (the new symbols are not yet exported); `test_existing_estimators_still_exported` should pass.

- [ ] **Step 3: Update `bayesianchainladder/__init__.py`**

In `bayesianchainladder/__init__.py`, after the existing import block, add:

```python
# Base contract
from .base import BaseStochasticReserve, MethodSummary

# Frequentist estimators
from .bootstrap import (
    BootstrapODPChainLadder,
    CorrelatedBootstrapChainLadder,
    CorrelatedBootstrapODPSample,
    MackChainLadder,
)
```

In the `__all__` list, after the existing `"BayesianChainLadderGLM",` and `"BayesianCSR",` entries, add:

```python
    # Base contract
    "BaseStochasticReserve",
    "MethodSummary",
    # Frequentist estimators
    "MackChainLadder",
    "BootstrapODPChainLadder",
    "CorrelatedBootstrapChainLadder",
    "CorrelatedBootstrapODPSample",
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_base.py::TestPackageExports -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Final full-suite check**

```bash
uv run pytest -v
```

Expected: all fast tests pass; slow tests skipped as usual. Total tests should be ~115+ (87 prior + ~30 new tests in `test_base.py` and `test_bootstrap.py`).

- [ ] **Step 6: Commit**

```bash
git add bayesianchainladder/__init__.py tests/test_base.py
git commit -m "$(cat <<'EOF'
Export new stochastic reserve API in bayesianchainladder.__init__

BaseStochasticReserve, MethodSummary, MackChainLadder,
BootstrapODPChainLadder, CorrelatedBootstrapChainLadder, and
CorrelatedBootstrapODPSample are now top-level imports.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Verification — End-to-end smoke check

After all tasks land, run a final cross-method smoke check from the package root to confirm the consistent interface works as intended:

```bash
uv run python -c "
import warnings; warnings.filterwarnings('ignore')
import chainladder as cl
from bayesianchainladder import (
    MackChainLadder,
    BootstrapODPChainLadder,
    CorrelatedBootstrapChainLadder,
)

triangle = cl.load_sample('genins')
for cls in (MackChainLadder, BootstrapODPChainLadder, CorrelatedBootstrapChainLadder):
    kwargs = dict(random_seed=42)
    if cls is BootstrapODPChainLadder or cls is CorrelatedBootstrapChainLadder:
        kwargs['n_sims'] = 200
    if cls is CorrelatedBootstrapChainLadder:
        kwargs['rho'] = 0.3
    model = cls(**kwargs).fit(triangle)
    summary = model.total_summary()
    print(f'{cls.__name__:<35} mean={summary.total_reserve_mean:>15,.0f}  '
          f'std={summary.total_reserve_stddev:>13,.0f}  '
          f'cv={summary.total_reserve_cv:.3f}')
"
```

Expected: three lines printed, each with finite mean / std / cv. The Mack and Bootstrap means should be in the same ballpark; Correlated should have a slightly larger std than the other two with `rho=0.3`.
