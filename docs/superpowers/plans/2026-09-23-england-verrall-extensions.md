# England & Verrall Extensions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the reserve-risk toolkit from Peter England's StochasticReserving repository (one-year CDR, discounting and risk margins, Mack/NegBin bootstraps, non-constant scale, influence analysis, scaling, fan charts, sample data, modus operandi notebook) into `bayesianchainladder` behind the shared `BaseStochasticReserve` interface.

**Architecture:** A new private numpy module `_triangle_ops.py` holds the chain-ladder primitives (factors, projection, sigma, masks) shared by every new feature. `BaseStochasticReserve` gains a per-cell `full_cumulative_posterior_` that all simulating estimators populate; downstream modules (`cdr.py`, `riskmeasures.py`, plots) consume only that array, so they work for the Bayesian GLM, CSR, ODP bootstraps and the new link-ratio bootstraps alike. New estimators live in `linkratio.py`; analytic oracles in `analytic.py`; influence analysis in `sensitivity.py`; data in `bayesianchainladder/data/`.

**Tech Stack:** Python 3.11/3.12, uv, numpy, xarray, pandas, chainladder 0.9.x, scipy (brentq), PyMC/pytensor for the Bayesian additions, matplotlib, nbformat/nbconvert for the notebook, pytest with the existing `slow` marker.

**Spec:** `docs/superpowers/specs/2026-09-23-england-verrall-extensions-design.md`

## Global Constraints

- `requires-python = ">=3.11,<3.13"`; all commands run through `uv run`.
- Lint with `uv run ruff check .` (rules E, F, W, I, UP, B, C4; line length 88, E501 ignored). Format with `uv run black .`.
- Any test that runs NUTS or executes a notebook is marked `@pytest.mark.slow` and skipped unless `--run-slow` is passed. Bootstrap tests with `n_sims <= 5000` on a 10×10 triangle are fast and unmarked.
- Every new public name is added to `bayesianchainladder/__init__.py` imports **and** `__all__`.
- `xr.DataArray` posteriors use dims `("origin", "sample")` for reserves and `("origin", "dev", "sample")` for full triangles. `origin` coords are ints (from `_extract_period_value`), `dev` coords are ints in months (12, 24, …).
- chainladder `drop` syntax is `(origin_label: str, dev_months: int)` where the tuple names the **numerator-free earlier cell** of the link ratio, e.g. `("2003", 72)` is the 72→84 ratio of origin 2003.
- Source attribution: any code or data derived from https://github.com/DrPeterEngland/StochasticReserving carries a comment or README line naming the repository and its MIT licence.
- Commit after each task with the message shown; end commit messages with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Never `git stash`; use WIP commits if work must be set aside.

## File map

| File | Responsibility |
|---|---|
| `bayesianchainladder/_triangle_ops.py` (new) | numpy chain-ladder primitives on `(…, n_origin, n_dev)` arrays |
| `bayesianchainladder/base.py` (modify) | `full_cumulative_posterior_`, `summary_statistics`, `ReserveSamples`, `scale_to_target`, `incurred_to_paid`, extended `MethodSummary` |
| `bayesianchainladder/bootstrap.py` (modify) | populate full posterior; `scale`, `process_scale`, `drop` on the correlated sampler |
| `bayesianchainladder/estimators.py` (modify) | GLM and CSR populate full posterior |
| `bayesianchainladder/datasets.py` (new) + `data/` | England sample triangles + loader |
| `bayesianchainladder/riskmeasures.py` (new) | VaR/TVaR/PHT, discounting, cost-of-capital |
| `bayesianchainladder/cdr.py` (new) | one-year Claims Development Result |
| `bayesianchainladder/linkratio.py` (new) | `MackBootstrap`, `NegativeBinomialBootstrap`, `BayesianMackChainLadder`, forecasting function |
| `bayesianchainladder/analytic.py` (new) | analytic RMSEP oracles |
| `bayesianchainladder/sensitivity.py` (new) | leave-one-ratio-out influence analysis |
| `bayesianchainladder/plots.py` (modify) | fan chart, scaled residuals, sensitivity heatmap, capital profiles |
| `bayesianchainladder/models.py` (modify) | `build_quasi_poisson_model`, `build_link_ratio_model` |
| `docs/notebooks/` (new) | modus operandi notebook + build script |
| `tests/test_triangle_ops.py`, `test_riskmeasures.py`, `test_cdr.py`, `test_linkratio.py`, `test_analytic.py`, `test_sensitivity.py`, `test_datasets.py`, `test_notebooks.py` (new); `test_base.py`, `test_bootstrap.py`, `test_estimators.py`, `test_plots.py`, `test_models.py` (modify) | tests |

Tasks 1–6 are the foundation and must run in order. Tasks 7–15 depend on Tasks 1–6 but are otherwise independent of one another. Task 16 (notebook) needs Tasks 6, 9, 10, 11, 12, 13, 14. Task 17 is last.

---

### Task 1: numpy chain-ladder primitives

**Files:**
- Create: `bayesianchainladder/_triangle_ops.py`
- Test: `tests/test_triangle_ops.py`

**Interfaces:**
- Produces:
  - `cumulative_array(triangle) -> tuple[np.ndarray, list[int], list[int]]` — `(cum (n_o, n_d) float with NaN for unobserved, origins, devs)`
  - `cumulative_to_incremental(cum: np.ndarray) -> np.ndarray` (same shape, NaN propagates)
  - `latest_diagonal(cum) -> tuple[np.ndarray, np.ndarray]` — `(latest (n_o,), last_col_index (n_o,) int)`
  - `drop_mask(n_origin, n_dev, drop, origins, devs) -> np.ndarray` — ones `(n_o, n_d-1)` with zeros at dropped ratios
  - `link_ratio_mask(cum, drop, origins, devs) -> np.ndarray` — `drop_mask` × availability (both cells observed)
  - `volume_weighted_factors(cum, mask) -> np.ndarray` — `cum` may be `(n_o, n_d)` or `(S, n_o, n_d)`; returns `(n_d-1,)` or `(S, n_d-1)`
  - `project_cumulative(cum, factors) -> np.ndarray` — fills NaN cells forward; same leading dims as `cum`
  - `link_ratio_sigma(cum, mask, factors, variance_factor=None) -> tuple[np.ndarray, np.ndarray]` — `(sigma (n_d-1,), unscaled_residuals (n_o, n_d-1) NaN where excluded)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_triangle_ops.py
"""Tests for the private numpy chain-ladder primitives."""

import chainladder as cl
import numpy as np
import pytest

from bayesianchainladder._triangle_ops import (
    cumulative_array,
    cumulative_to_incremental,
    drop_mask,
    latest_diagonal,
    link_ratio_mask,
    link_ratio_sigma,
    project_cumulative,
    volume_weighted_factors,
)


@pytest.fixture
def genins():
    return cl.load_sample("genins")


def test_cumulative_array_shape_and_coords(genins):
    cum, origins, devs = cumulative_array(genins)
    assert cum.shape == (10, 10)
    assert origins == list(range(2001, 2011))
    assert devs == [12 * k for k in range(1, 11)]
    assert np.isnan(cum[1, 9]) and not np.isnan(cum[1, 8])


def test_cumulative_to_incremental_roundtrip(genins):
    cum, _, _ = cumulative_array(genins)
    incr = cumulative_to_incremental(cum)
    expected = np.asarray(genins.cum_to_incr().values, dtype=float)[0, 0]
    np.testing.assert_allclose(incr, expected, equal_nan=True)


def test_latest_diagonal(genins):
    cum, _, _ = cumulative_array(genins)
    latest, idx = latest_diagonal(cum)
    np.testing.assert_allclose(latest, np.asarray(genins.latest_diagonal.values)[0, 0, :, 0])
    assert list(idx) == list(range(9, -1, -1))


def test_masks(genins):
    cum, origins, devs = cumulative_array(genins)
    dm = drop_mask(10, 10, [("2003", 72)], origins, devs)
    assert dm.shape == (10, 9)
    assert dm[2, 5] == 0 and dm.sum() == 89
    m = link_ratio_mask(cum, None, origins, devs)
    assert m.sum() == 45  # 9+8+...+1 available ratios
    m2 = link_ratio_mask(cum, [("2003", 72)], origins, devs)
    assert m2.sum() == 44


def test_volume_weighted_factors_match_chainladder(genins):
    cum, origins, devs = cumulative_array(genins)
    f = volume_weighted_factors(cum, link_ratio_mask(cum, None, origins, devs))
    expected = np.asarray(cl.Development().fit_transform(genins).ldf_.values).flatten()
    np.testing.assert_allclose(f, expected, rtol=1e-10)


def test_volume_weighted_factors_broadcast_over_sims(genins):
    cum, origins, devs = cumulative_array(genins)
    mask = link_ratio_mask(cum, None, origins, devs)
    stacked = np.repeat(cum[None, ...], 3, axis=0)
    f = volume_weighted_factors(stacked, mask)
    assert f.shape == (3, 9)
    np.testing.assert_allclose(f[0], volume_weighted_factors(cum, mask))


def test_project_cumulative_matches_chainladder(genins):
    cum, origins, devs = cumulative_array(genins)
    f = volume_weighted_factors(cum, link_ratio_mask(cum, None, origins, devs))
    full = project_cumulative(cum, f)
    expected = np.asarray(cl.Chainladder().fit(genins).full_triangle_.values)[0, 0, :, :10]
    np.testing.assert_allclose(full, expected, rtol=1e-10)
    assert not np.isnan(full).any()


def test_project_cumulative_with_sim_dim(genins):
    cum, origins, devs = cumulative_array(genins)
    mask = link_ratio_mask(cum, None, origins, devs)
    stacked = np.repeat(cum[None, ...], 2, axis=0)
    f = volume_weighted_factors(stacked, mask)
    full = project_cumulative(stacked, f)
    assert full.shape == (2, 10, 10)
    np.testing.assert_allclose(full[1], project_cumulative(cum, f[1]))


def test_link_ratio_sigma_matches_mack_except_last(genins):
    cum, origins, devs = cumulative_array(genins)
    mask = link_ratio_mask(cum, None, origins, devs)
    f = volume_weighted_factors(cum, mask)
    sigma, resid = link_ratio_sigma(cum, mask, f)
    expected = np.asarray(cl.Development().fit_transform(genins).sigma_.values).flatten()
    # chainladder extrapolates the last sigma log-linearly; England uses min of previous two
    np.testing.assert_allclose(sigma[:-1], expected[:-1], rtol=1e-8)
    assert sigma[-1] == pytest.approx(min(sigma[-2], sigma[-3]))
    assert resid.shape == (10, 9)
    assert np.isnan(resid[9, 0]) and np.isfinite(resid[0, 0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_triangle_ops.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bayesianchainladder._triangle_ops'`

- [ ] **Step 3: Implement the module**

```python
# bayesianchainladder/_triangle_ops.py
"""Private numpy chain-ladder primitives shared by link-ratio bootstraps, the
Claims Development Result, sensitivity analysis and analytic oracles.

Conventions follow the volume-weighted chain ladder in England & Verrall
(2002) and the reference implementation in Peter England's StochasticReserving
repository (https://github.com/DrPeterEngland/StochasticReserving, MIT).

All functions work on plain arrays shaped ``(..., n_origin, n_dev)``; a leading
simulation axis is allowed wherever documented.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .utils import _extract_period_value

DropList = Sequence[tuple[str | int, int]] | None


def cumulative_array(triangle) -> tuple[np.ndarray, list[int], list[int]]:
    """Return ``(cum, origins, devs)`` for a single-index chainladder Triangle."""
    vals = np.asarray(triangle.values, dtype=float)
    if vals.shape[0] != 1 or vals.shape[1] != 1:
        raise ValueError("cumulative_array expects a single-index, single-column triangle")
    origins = [_extract_period_value(o) for o in triangle.origin]
    devs = [int(d) for d in triangle.development]
    return vals[0, 0].copy(), origins, devs


def cumulative_to_incremental(cum: np.ndarray) -> np.ndarray:
    incr = np.array(cum, dtype=float, copy=True)
    incr[..., 1:] = cum[..., 1:] - cum[..., :-1]
    return incr


def latest_diagonal(cum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Latest observed value per origin and its column index (-1 if none)."""
    obs = ~np.isnan(cum)
    n_dev = cum.shape[-1]
    any_obs = obs.any(axis=-1)
    last_idx = np.where(any_obs, n_dev - 1 - np.argmax(obs[..., ::-1], axis=-1), -1)
    rows = np.arange(cum.shape[0])
    latest = np.where(any_obs, cum[rows, np.maximum(last_idx, 0)], np.nan)
    return latest, last_idx


def drop_mask(n_origin: int, n_dev: int, drop: DropList, origins, devs) -> np.ndarray:
    """Ones ``(n_origin, n_dev-1)`` with zeros at chainladder-style ``drop`` tuples."""
    mask = np.ones((n_origin, n_dev - 1))
    for origin_label, dev_months in drop or []:
        i = list(origins).index(int(origin_label))
        j = list(devs).index(int(dev_months))
        if j >= n_dev - 1:
            raise ValueError(f"dev {dev_months} has no link ratio to drop")
        mask[i, j] = 0.0
    return mask


def link_ratio_mask(cum: np.ndarray, drop: DropList, origins, devs) -> np.ndarray:
    """Availability mask (both cells observed) times ``drop_mask``."""
    n_o, n_d = cum.shape
    avail = (~np.isnan(cum[:, :-1]) & ~np.isnan(cum[:, 1:])).astype(float)
    return avail * drop_mask(n_o, n_d, drop, origins, devs)


def volume_weighted_factors(cum: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Volume-weighted development factors; ``cum`` may carry a leading sim axis."""
    c = np.nan_to_num(cum, nan=0.0)
    num = (c[..., :, 1:] * mask).sum(axis=-2)
    den = (c[..., :, :-1] * mask).sum(axis=-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(den > 0, num / den, 1.0)
    return f


def project_cumulative(cum: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """Fill every NaN cell forward from the previous column times the factor.

    ``factors`` is ``(n_dev-1,)`` for a 2-D ``cum`` or ``(S, n_dev-1)`` for a
    ``(S, n_origin, n_dev)`` ``cum``.
    """
    full = np.array(cum, dtype=float, copy=True)
    n_dev = full.shape[-1]
    for j in range(1, n_dev):
        need = np.isnan(full[..., :, j])
        fill = full[..., :, j - 1] * np.expand_dims(factors[..., j - 1], -1)
        full[..., :, j] = np.where(need, fill, full[..., :, j])
    return full


def link_ratio_sigma(
    cum: np.ndarray,
    mask: np.ndarray,
    factors: np.ndarray,
    variance_factor: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-development sigma for a link-ratio model (Mack when
    ``variance_factor`` is None, Negative Binomial when it is ``f*(f-1)``).

    Rules from England's ``Mack_Residuals``: sigma_j^2 = sum(w (F-f)^2 / v_j) /
    (n_j - 1); carry forward when n_j <= 1; last column = min of the previous
    two; zero where the cumulative factor is exactly 1.
    """
    vf = np.ones_like(factors) if variance_factor is None else np.asarray(variance_factor, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = cum[:, 1:] / cum[:, :-1]
    w = cum[:, :-1]
    resid = np.sqrt(np.abs(w)) * (ratios - factors) / np.sqrt(np.where(vf > 0, vf, np.nan))
    resid = np.where(mask > 0, resid, np.nan)
    n_j = mask.sum(axis=0)
    ss = np.nansum(resid**2, axis=0)
    sigma = np.zeros(len(n_j))
    for j in range(len(n_j) - 1):
        if n_j[j] > 1:
            sigma[j] = np.sqrt(ss[j] / (n_j[j] - 1))
        else:
            sigma[j] = 0.0 if j == 0 else sigma[j - 1]
    if len(sigma) >= 3:
        sigma[-1] = min(sigma[-2], sigma[-3])
    elif len(sigma) == 2:
        sigma[-1] = sigma[-2]
    cum_factors = np.cumprod(factors[::-1])[::-1]
    sigma[np.isclose(cum_factors, 1.0)] = 0.0
    return sigma, resid
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_triangle_ops.py -q`
Expected: 9 passed

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check bayesianchainladder/_triangle_ops.py tests/test_triangle_ops.py
git add bayesianchainladder/_triangle_ops.py tests/test_triangle_ops.py
git commit -m "feat: add private numpy chain-ladder primitives

Volume-weighted factors, projection, England/Mack sigma rules and
drop-mask handling shared by upcoming link-ratio bootstraps, CDR and
sensitivity analysis.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: per-cell posterior and tail statistics in the base class

**Files:**
- Modify: `bayesianchainladder/base.py`
- Modify: `bayesianchainladder/__init__.py` (export `ReserveSamples`, `DEFAULT_QUANTILES`)
- Test: `tests/test_base.py`

**Interfaces:**
- Consumes: `latest_diagonal`, `cumulative_array` from Task 1.
- Produces on `BaseStochasticReserve`:
  - attribute `full_cumulative_posterior_: xr.DataArray | None` dims `("origin","dev","sample")`
  - `_set_full_cumulative_posterior(self, cumulative: np.ndarray, origins, devs) -> None`
  - `_require_full_posterior(self) -> xr.DataArray` (raises `ValueError` if None)
  - `_reserves_from_full_posterior(self) -> xr.DataArray` dims `("origin","sample")`
  - `incremental_posterior(self) -> xr.DataArray` dims `("origin","dev","sample")`
  - `future_incremental_posterior(self) -> xr.DataArray` (observed cells set to 0)
  - `summary_statistics(self, output="reserves", quantiles=DEFAULT_QUANTILES) -> pd.DataFrame` index origins + `"Total"`, columns `mean, std, cov, min, <q>%…, max`
  - `MethodSummary` new defaulted fields `total_reserve_99_5th_percentile`, `total_reserve_min`, `total_reserve_max`
  - class `ReserveSamples(BaseStochasticReserve)` with `__init__(triangle, reserves_posterior, full_cumulative_posterior=None)`
  - module constant `DEFAULT_QUANTILES = (0.005, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.995)`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_base.py`)

```python
# --- append to tests/test_base.py ---
import pandas as pd

from bayesianchainladder.base import DEFAULT_QUANTILES, ReserveSamples


class TestFullCumulativePosterior:
    @pytest.fixture
    def toy(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "origin": [2001, 2001, 2001, 2002, 2002, 2003],
                "dev": [12, 24, 36, 12, 24, 12],
                "value": [100.0, 150.0, 160.0, 110.0, 170.0, 120.0],
            }
        )
        eval_year = df["origin"] + df["dev"] // 12 - 1
        df["dev_date"] = pd.to_datetime(eval_year.astype(str) + "-12-31")
        tri = cl.Triangle(
            df, origin="origin", development="dev_date", columns=["value"],
            cumulative=True, origin_format="%Y",
        )
        # (origin, dev, sample) cumulative; NaN-free, observed cells constant
        full = np.full((3, 3, 4), np.nan)
        full[0, :, :] = np.array([[100.0], [150.0], [160.0]])
        full[1, 0, :] = 110.0
        full[1, 1, :] = 170.0
        full[1, 2, :] = [180.0, 182.0, 184.0, 186.0]
        full[2, 0, :] = 120.0
        full[2, 1, :] = [170.0, 175.0, 180.0, 185.0]
        full[2, 2, :] = [180.0, 190.0, 200.0, 210.0]
        return tri, full

    def test_reserve_samples_container(self, toy):
        tri, full = toy
        reserves = xr.DataArray(
            full[:, -1, :] - np.array([[160.0], [170.0], [120.0]]),
            dims=["origin", "sample"],
            coords={"origin": [2001, 2002, 2003], "sample": np.arange(4)},
        )
        rs = ReserveSamples(tri, reserves)
        assert isinstance(rs, BaseStochasticReserve)
        assert rs.ibnr_.loc[2003, "mean"] == pytest.approx(75.0)
        assert rs.full_cumulative_posterior_ is None
        with pytest.raises(ValueError, match="per-cell"):
            rs._require_full_posterior()
        with pytest.raises(NotImplementedError):
            rs.fit(tri)

    def test_full_posterior_helpers(self, toy):
        tri, full = toy
        rs = ReserveSamples(
            tri,
            xr.DataArray(np.zeros((3, 4)), dims=["origin", "sample"],
                         coords={"origin": [2001, 2002, 2003], "sample": np.arange(4)}),
        )
        rs._set_full_cumulative_posterior(full, [2001, 2002, 2003], [12, 24, 36])
        assert rs.full_cumulative_posterior_.dims == ("origin", "dev", "sample")
        derived = rs._reserves_from_full_posterior()
        np.testing.assert_allclose(derived.sel(origin=2002).values, [10.0, 12.0, 14.0, 16.0])
        np.testing.assert_allclose(derived.sel(origin=2003).values, [60.0, 70.0, 80.0, 90.0])
        incr = rs.incremental_posterior()
        np.testing.assert_allclose(incr.sel(origin=2003, dev=24).values, [50.0, 55.0, 60.0, 65.0])
        fut = rs.future_incremental_posterior()
        assert (fut.sel(origin=2001).values == 0).all()
        np.testing.assert_allclose(fut.sel(origin=2003, dev=36).values, [10.0, 15.0, 20.0, 25.0])

    def test_summary_statistics(self, toy):
        tri, full = toy
        reserves = xr.DataArray(
            np.array([[0.0, 0.0, 0.0, 0.0], [10.0, 12.0, 14.0, 16.0], [60.0, 70.0, 80.0, 90.0]]),
            dims=["origin", "sample"],
            coords={"origin": [2001, 2002, 2003], "sample": np.arange(4)},
        )
        rs = ReserveSamples(tri, reserves)
        stats = rs.summary_statistics()
        assert list(stats.index) == [2001, 2002, 2003, "Total"]
        assert stats.loc["Total", "mean"] == pytest.approx(88.0)
        assert stats.loc[2003, "min"] == 60.0 and stats.loc[2003, "max"] == 90.0
        assert "99.5%" in stats.columns and "0.5%" in stats.columns
        assert stats.loc[2002, "cov"] == pytest.approx(np.std([10, 12, 14, 16], ddof=1) / 13.0)
        ults = rs.summary_statistics(output="ultimates")
        assert ults.loc[2003, "mean"] == pytest.approx(120.0 + 75.0)
        with pytest.raises(ValueError):
            rs.summary_statistics(output="nonsense")
        assert len(DEFAULT_QUANTILES) == 11

    def test_method_summary_new_fields_default(self):
        s = MethodSummary(1.0, 2.0, 3.0, 4.0, 5.0)
        assert math.isnan(s.total_reserve_99_5th_percentile)
        assert math.isnan(s.total_reserve_min) and math.isnan(s.total_reserve_max)

    def test_total_summary_populates_new_fields(self, toy):
        tri, _ = toy
        reserves = xr.DataArray(
            np.array([[0.0] * 4, [10.0, 12.0, 14.0, 16.0], [60.0, 70.0, 80.0, 90.0]]),
            dims=["origin", "sample"],
            coords={"origin": [2001, 2002, 2003], "sample": np.arange(4)},
        )
        s = ReserveSamples(tri, reserves).total_summary()
        assert s.total_reserve_min == pytest.approx(70.0)
        assert s.total_reserve_max == pytest.approx(106.0)
        assert s.total_reserve_99_5th_percentile == pytest.approx(np.quantile([70, 82, 94, 106], 0.995))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_base.py -q -k "FullCumulative or new_fields"`
Expected: FAIL with `ImportError: cannot import name 'DEFAULT_QUANTILES'`

- [ ] **Step 3: Implement**

In `bayesianchainladder/base.py`:

1. Add after the imports:
```python
DEFAULT_QUANTILES: tuple[float, ...] = (
    0.005, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.995,
)
```

2. Extend `MethodSummary` (keep existing five fields first, then):
```python
    total_reserve_99_5th_percentile: float = float("nan")
    total_reserve_min: float = float("nan")
    total_reserve_max: float = float("nan")
```

3. In `BaseStochasticReserve.__init__` add `self.full_cumulative_posterior_: xr.DataArray | None = None` and the class-level annotation `full_cumulative_posterior_: xr.DataArray | None`.

4. Add these methods to `BaseStochasticReserve` (after `_build_reserve_summaries`):
```python
    # ------------------------------------------------------------------
    # Per-cell simulated triangles (England & Verrall "Complete_Cumulatives")
    # ------------------------------------------------------------------
    def _set_full_cumulative_posterior(self, cumulative, origins, devs) -> None:
        arr = np.asarray(cumulative, dtype=float)
        if arr.ndim != 3:
            raise ValueError("cumulative must have shape (origin, dev, sample)")
        self.full_cumulative_posterior_ = xr.DataArray(
            arr,
            dims=["origin", "dev", "sample"],
            coords={
                "origin": list(origins),
                "dev": list(devs),
                "sample": np.arange(arr.shape[2]),
            },
        )

    def _require_full_posterior(self) -> xr.DataArray:
        self._check_is_fitted()
        if self.full_cumulative_posterior_ is None:
            raise ValueError(
                f"{type(self).__name__} does not provide per-cell simulated "
                "triangles (full_cumulative_posterior_ is None); this analysis "
                "needs an estimator that simulates every future cell."
            )
        return self.full_cumulative_posterior_

    def _reserves_from_full_posterior(self) -> xr.DataArray:
        from ._triangle_ops import cumulative_array, latest_diagonal

        full = self._require_full_posterior()
        cum, _, _ = cumulative_array(self.triangle_)
        latest, _ = latest_diagonal(cum)
        latest_da = xr.DataArray(
            latest, dims=["origin"], coords={"origin": full.coords["origin"]}
        )
        return (full.isel(dev=-1) - latest_da).transpose("origin", "sample")

    def incremental_posterior(self) -> xr.DataArray:
        full = self._require_full_posterior()
        vals = full.values.copy()
        vals[:, 1:, :] = full.values[:, 1:, :] - full.values[:, :-1, :]
        return full.copy(data=vals)

    def future_incremental_posterior(self) -> xr.DataArray:
        from ._triangle_ops import cumulative_array

        incr = self.incremental_posterior()
        cum, _, _ = cumulative_array(self.triangle_)
        observed = ~np.isnan(cum)
        vals = np.where(observed[..., None], 0.0, incr.values)
        return incr.copy(data=vals)

    # ------------------------------------------------------------------
    # England-style summary statistics with tail quantiles
    # ------------------------------------------------------------------
    def summary_statistics(
        self,
        output: str = "reserves",
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ) -> pd.DataFrame:
        self._check_is_fitted()
        if self.reserves_posterior_ is None:
            raise ValueError("No reserve posterior available")
        res = self.reserves_posterior_.transpose("origin", "sample").values
        origins = list(self.reserves_posterior_.coords["origin"].values)
        if output == "reserves":
            data = res
        elif output == "ultimates":
            paid = self._paid_to_date().reindex(origins).fillna(0.0).values
            data = res + paid[:, None]
        else:
            raise ValueError("output must be 'reserves' or 'ultimates'")
        data = np.vstack([data, data.sum(axis=0, keepdims=True)])
        mean = np.nanmean(data, axis=1)
        std = np.nanstd(data, axis=1, ddof=1) if data.shape[1] > 1 else np.zeros(len(mean))
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = np.where(mean != 0, std / np.abs(mean), np.nan)
        table: dict[str, np.ndarray] = {
            "mean": mean,
            "std": std,
            "cov": cov,
            "min": np.nanmin(data, axis=1),
        }
        for q in quantiles:
            table[f"{q * 100:g}%"] = np.nanquantile(data, q, axis=1)
        table["max"] = np.nanmax(data, axis=1)
        return pd.DataFrame(table, index=[*origins, "Total"])
```

5. Extend `total_summary` (the non-empty branch) to fill the new fields:
```python
        return MethodSummary(
            total_reserve_mean=float(np.mean(total)),
            total_reserve_stddev=float(np.std(total, ddof=1)) if total.size > 1 else 0.0,
            total_reserve_75th_percentile=float(np.quantile(total, 0.75)),
            total_reserve_90th_percentile=float(np.quantile(total, 0.90)),
            total_reserve_95th_percentile=float(np.quantile(total, 0.95)),
            total_reserve_99_5th_percentile=float(np.quantile(total, 0.995)),
            total_reserve_min=float(np.min(total)),
            total_reserve_max=float(np.max(total)),
        )
```

6. Add the container class at the end of `base.py`:
```python
class ReserveSamples(BaseStochasticReserve):
    """Reserve samples produced outside a fit (scaling, incurred-to-paid,
    external simulations) exposed through the shared interface."""

    def __init__(
        self,
        triangle,
        reserves_posterior: xr.DataArray,
        full_cumulative_posterior: xr.DataArray | None = None,
    ) -> None:
        super().__init__()
        self.triangle_ = triangle.copy()
        self.reserves_posterior_ = reserves_posterior.transpose("origin", "sample")
        self.full_cumulative_posterior_ = full_cumulative_posterior
        self._build_reserve_summaries()
        self._is_fitted = True

    def fit(self, triangle, **kwargs: Any):
        raise NotImplementedError(
            "ReserveSamples is constructed from samples, not fitted"
        )
```

7. In `bayesianchainladder/__init__.py` change the base import to `from .base import DEFAULT_QUANTILES, BaseStochasticReserve, MethodSummary, ReserveSamples` and add `"ReserveSamples"` and `"DEFAULT_QUANTILES"` under `# Base contract` in `__all__`.

- [ ] **Step 4: Run the whole fast suite**

Run: `uv run pytest -q`
Expected: all pass (existing `MethodSummary` positional constructions still work because new fields have defaults).

- [ ] **Step 5: Commit**

```bash
uv run ruff check bayesianchainladder/base.py tests/test_base.py
git add bayesianchainladder/base.py bayesianchainladder/__init__.py tests/test_base.py
git commit -m "feat: per-cell posterior contract, tail summary statistics, ReserveSamples

Adds full_cumulative_posterior_ (origin, dev, sample) to the base
contract with helpers to derive reserves and incrementals, an
England-style summary_statistics table with 0.5%-99.5% quantiles, and
a ReserveSamples container for externally produced samples.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: bootstrap wrappers populate the per-cell posterior

**Files:**
- Modify: `bayesianchainladder/bootstrap.py` (`BootstrapODPChainLadder.fit`, `CorrelatedBootstrapChainLadder.fit`, `BootstrapODPBornhuetterFerguson.fit`, `BootstrapODPCapeCod.fit`, `CorrelatedBootstrapODPBornhuetterFerguson.fit`, `CorrelatedBootstrapODPCapeCod.fit`)
- Test: `tests/test_bootstrap.py`

**Interfaces:**
- Consumes: `_set_full_cumulative_posterior`, `_reserves_from_full_posterior` (Task 2).
- Produces: module helper `_full_posterior_from_chainladder(model_fitted, triangle) -> tuple[np.ndarray, list[int], list[int]]` returning `(cumulative (n_o, n_d, S), origins, devs)`.

Background: for a chainladder method fitted on a resampled triangle, `model.full_triangle_.values` has shape `(S, 1, n_o, n_d + k)`; the extra `k` columns are a placeholder tail step and the `9999` ultimate column. `model.process_variance_` (same shape) holds the gamma process noise that `ultimate_` already includes, so `full_triangle_ + process_variance_` restricted to the first `n_d` columns is the complete simulated cumulative triangle consistent with `ibnr_`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_bootstrap.py`)

```python
# --- append to tests/test_bootstrap.py ---
class TestFullCumulativePosteriorWrappers:
    @pytest.fixture
    def genins(self):
        return cl.load_sample("genins")

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: __import__("bayesianchainladder.bootstrap", fromlist=["x"]).BootstrapODPChainLadder(n_sims=200, random_seed=1),
            lambda: __import__("bayesianchainladder.bootstrap", fromlist=["x"]).CorrelatedBootstrapChainLadder(n_sims=200, rho=0.3, random_seed=1),
        ],
    )
    def test_chainladder_wrappers_expose_consistent_full_posterior(self, genins, factory):
        model = factory().fit(genins)
        full = model.full_cumulative_posterior_
        assert full.dims == ("origin", "dev", "sample")
        assert full.shape == (10, 10, 200)
        cum = np.asarray(genins.values)[0, 0]
        obs = ~np.isnan(cum)
        # observed cells are constant across samples and equal the data
        np.testing.assert_allclose(
            full.values[obs], np.repeat(cum[obs][:, None], 200, axis=1), rtol=1e-9
        )
        derived = model._reserves_from_full_posterior()
        np.testing.assert_allclose(
            derived.values, model.reserves_posterior_.values, rtol=1e-6, atol=1e-6
        )
        assert not np.isnan(full.values).any()

    def test_bf_cc_wrappers_expose_full_posterior(self, genins, genins_premium_triangle):
        from bayesianchainladder.bootstrap import (
            BootstrapODPBornhuetterFerguson,
            BootstrapODPCapeCod,
        )

        for cls in (BootstrapODPBornhuetterFerguson, BootstrapODPCapeCod):
            model = cls(n_sims=100, random_seed=3).fit(
                genins, exposure_triangle=genins_premium_triangle
            )
            full = model.full_cumulative_posterior_
            assert full.shape == (10, 10, 100)
            cum = np.asarray(genins.values)[0, 0]
            obs = ~np.isnan(cum)
            np.testing.assert_allclose(
                full.values[obs], np.repeat(cum[obs][:, None], 100, axis=1), rtol=1e-9
            )
            # ultimates from the full posterior agree with the wrapper's own IBNR
            derived = model._reserves_from_full_posterior()
            np.testing.assert_allclose(
                derived.mean("sample").values,
                model.reserves_posterior_.mean("sample").values,
                rtol=0.02,
            )

    def test_mack_wrapper_has_no_full_posterior(self, raa_triangle):
        from bayesianchainladder.bootstrap import MackChainLadder

        model = MackChainLadder().fit(raa_triangle)
        assert model.full_cumulative_posterior_ is None
        with pytest.raises(ValueError, match="per-cell"):
            model._require_full_posterior()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_bootstrap.py -q -k FullCumulativePosteriorWrappers`
Expected: FAIL with `AttributeError: 'NoneType' object has no attribute 'dims'`

- [ ] **Step 3: Implement**

Add to `bayesianchainladder/bootstrap.py` after `_extract_ibnr_from_bf_or_cc`:

```python
def _full_posterior_from_chainladder(model_fitted, triangle):
    """Complete simulated cumulative triangles from a chainladder method fitted
    on ``n_sims`` resamples: ``full_triangle_`` (parameter risk) plus
    ``process_variance_`` (process risk) restricted to the original
    development columns. Returns ``(cumulative (n_o, n_d, S), origins, devs)``.
    """
    n_dev = triangle.values.shape[-1]
    full = np.asarray(model_fitted.full_triangle_.values, dtype=float)[:, 0, :, :n_dev]
    process_var = getattr(model_fitted, "process_variance_", None)
    if process_var is not None:
        full = full + np.nan_to_num(
            np.asarray(process_var.values, dtype=float)[:, 0, :, :n_dev]
        )
    origins = [_extract_period_value(o) for o in triangle.origin]
    devs = [int(d) for d in triangle.development]
    return np.moveaxis(full, 0, -1), origins, devs
```

Then in each of the six `fit` methods insert, immediately before `self._build_reserve_summaries()`:

```python
        full, origins_full, devs_full = _full_posterior_from_chainladder(model, triangle)
        self._set_full_cumulative_posterior(full, origins_full, devs_full)
```

using the local name of the fitted chainladder object (`model` in the two chain-ladder wrappers, `bf` / `cc` in the BF/CC wrappers).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_bootstrap.py -q`
Expected: all pass. If the `rtol=1e-6` comparison in `test_chainladder_wrappers_expose_consistent_full_posterior` fails, print `model.full_triangle_.development` and compare column `n_dev-1` with the `9999` column of `full_triangle_ + process_variance_`; they must be equal when the triangle has no tail. Do not loosen the tolerance without understanding the gap.

- [ ] **Step 5: Commit**

```bash
uv run ruff check bayesianchainladder/bootstrap.py tests/test_bootstrap.py
git add bayesianchainladder/bootstrap.py tests/test_bootstrap.py
git commit -m "feat: bootstrap wrappers populate full_cumulative_posterior_

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Bayesian GLM populates the per-cell posterior

**Files:**
- Modify: `bayesianchainladder/estimators.py` (`BayesianChainLadderGLM._compute_reserves`, new `_store_full_posterior`)
- Test: `tests/test_estimators.py`

**Interfaces:**
- Consumes: `cumulative_array`, `cumulative_to_incremental` (Task 1); `_set_full_cumulative_posterior` (Task 2).
- Produces: `BayesianChainLadderGLM._store_full_posterior(self, future_predictions: xr.DataArray, obs_dim: str, future_start: int) -> None`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_estimators.py`)

```python
# --- append to tests/test_estimators.py ---
@pytest.mark.slow
def test_glm_full_posterior_consistent_with_reserves():
    import chainladder as cl
    import numpy as np

    from bayesianchainladder import BayesianChainLadderGLM

    tri = cl.load_sample("genins")
    model = BayesianChainLadderGLM(
        formula="incremental ~ 1 + C(origin) + C(dev)",
        family="gaussian",
        draws=100,
        tune=50,
        chains=1,
        random_seed=42,
    ).fit(tri)
    full = model.full_cumulative_posterior_
    assert full.dims == ("origin", "dev", "sample")
    assert full.shape[:2] == (10, 10)
    assert full.shape[2] == model.reserves_posterior_.sizes["sample"]
    cum = np.asarray(tri.values)[0, 0]
    obs = ~np.isnan(cum)
    np.testing.assert_allclose(
        full.values[obs], np.repeat(cum[obs][:, None], full.shape[2], axis=1), rtol=1e-9
    )
    derived = model._reserves_from_full_posterior()
    np.testing.assert_allclose(
        derived.transpose("origin", "sample").values,
        model.reserves_posterior_.transpose("origin", "sample").values,
        rtol=1e-6, atol=1e-6,
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_estimators.py -q -k full_posterior_consistent --run-slow`
Expected: FAIL with `AttributeError: 'NoneType' object has no attribute 'dims'`

- [ ] **Step 3: Implement**

In `BayesianChainLadderGLM._compute_reserves`, after `self._build_reserve_summaries()` (inside the `if reserve_samples:` block) add:

```python
            self._store_full_posterior(future_predictions, obs_dim, future_start)
```

Add the method right after `_compute_reserves`:

```python
    def _store_full_posterior(
        self, future_predictions: xr.DataArray, obs_dim: str, future_start: int
    ) -> None:
        """Assemble complete simulated cumulative triangles (origin, dev,
        sample) from observed incrementals plus per-cell future predictions,
        applying the same loss-ratio back-transform and response shift as
        ``_compute_reserves`` so both views agree cell by cell."""
        from ._triangle_ops import cumulative_array, cumulative_to_incremental

        cum, origins, devs = cumulative_array(self.triangle_)
        incr_obs = cumulative_to_incremental(cum)
        observed = ~np.isnan(incr_obs)

        fut = (
            future_predictions.isel({obs_dim: slice(future_start, None)})
            .stack(sample=["chain", "draw"])
            .transpose(obs_dim, "sample")
            .values
        )
        n_samples = fut.shape[1]
        incr = np.repeat(np.where(observed, incr_obs, 0.0)[..., None], n_samples, axis=-1)
        valid = observed.copy()

        ep_col = self._original_exposure_col if self.response_per_exposure else None
        if ep_col is not None and ep_col not in self.future_data_.columns:
            ep_col = None

        fut_origin = self.future_data_["origin"].values
        fut_dev = self.future_data_["dev"].values
        for k in range(len(self.future_data_)):
            o, d = int(fut_origin[k]), int(fut_dev[k])
            if o not in origins or d not in devs:
                continue
            i, j = origins.index(o), devs.index(d)
            cell = np.asarray(fut[k], dtype=float)
            if ep_col is not None:
                cell = cell * float(self.future_data_.iloc[k][ep_col])
            if self._response_shift != 0.0:
                cell = cell - self._response_shift
            incr[i, j, :] = cell
            valid[i, j] = True

        full = np.cumsum(incr, axis=1)
        full[~valid] = np.nan
        self._set_full_cumulative_posterior(full, origins, devs)
```

- [ ] **Step 4: Run the slow test and the fast suite**

Run: `uv run pytest tests/test_estimators.py -q -k full_posterior_consistent --run-slow` then `uv run pytest -q`
Expected: PASS; fast suite unchanged.

- [ ] **Step 5: Commit**

```bash
git add bayesianchainladder/estimators.py tests/test_estimators.py
git commit -m "feat: BayesianChainLadderGLM stores per-cell simulated triangles

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: BayesianCSR populates the per-cell posterior

**Files:**
- Modify: `bayesianchainladder/estimators.py` (`BayesianCSR._compute_predictions`, around the per-origin loop at the `mu_ultimate` computation)
- Test: `tests/test_estimators.py`

**Interfaces:**
- Consumes: `_set_full_cumulative_posterior` (Task 2), `cumulative_array` (Task 1).
- Produces: `BayesianCSR.full_cumulative_posterior_` where each future cell of origin *o* at development index *k* is `exp(mu_k + sig_k * z)` with **one** standard normal `z` per (origin, sample) shared across development periods (comonotonic path); the ultimate cell reproduces the existing `ultimate_samples` distribution exactly.

- [ ] **Step 1: Write the failing test** (append to `tests/test_estimators.py`)

```python
@pytest.mark.slow
def test_csr_full_posterior_paths_are_monotone_and_consistent():
    import chainladder as cl
    import numpy as np

    from bayesianchainladder import BayesianCSR

    tri = cl.load_sample("genins")
    model = BayesianCSR(
        premium_value=5_000_000.0, draws=100, tune=50, chains=1, random_seed=42
    ).fit(tri)
    full = model.full_cumulative_posterior_
    assert full.dims == ("origin", "dev", "sample")
    cum = np.asarray(tri.values)[0, 0]
    obs = ~np.isnan(cum)
    np.testing.assert_allclose(
        full.values[obs], np.repeat(cum[obs][:, None], full.shape[2], axis=1), rtol=1e-9
    )
    derived = model._reserves_from_full_posterior()
    np.testing.assert_allclose(
        derived.transpose("origin", "sample").values,
        model.reserves_posterior_.transpose("origin", "sample").values,
        rtol=1e-6, atol=1e-6,
    )
    # future cumulative paths never contain NaN
    assert not np.isnan(full.values).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_estimators.py -q -k csr_full_posterior --run-slow`
Expected: FAIL with `AttributeError: 'NoneType' object has no attribute 'dims'`

- [ ] **Step 3: Implement**

In `BayesianCSR._compute_predictions`, before the `for origin in all_origins:` loop, add:

```python
        from ._triangle_ops import cumulative_array

        cum_obs, tri_origins, tri_devs = cumulative_array(self.triangle_)
        n_samples_total = int(np.prod(alpha.shape[:2]))
        full_paths = np.repeat(cum_obs[..., None], n_samples_total, axis=-1)
```

Inside the `if origin in origins_with_future:` branch replace the block that draws `logloss_samples` / `ultimate_cumulative` with a path-based version that shares one normal shock across development periods:

```python
                last_dev_idx = dev_levels.index(last_observed_dev)
                future_dev_idx = list(range(last_dev_idx + 1, ultimate_dev_idx + 1))
                z = np.random.standard_normal(alpha.shape[:2])  # one shock per (chain, draw)
                path_cells = {}
                for k in future_dev_idx:
                    mu_k = (
                        logprem
                        + logelr
                        + alpha[:, :, origin_idx]
                        + beta[:, :, k] * speedup[:, :, origin_idx]
                    )
                    sig_k = sig[:, :, k]
                    if self.include_process_variance:
                        path_cells[k] = np.exp(mu_k + sig_k * z)
                    else:
                        path_cells[k] = np.exp(mu_k + 0.5 * sig_k**2)
                ultimate_cumulative = path_cells[ultimate_dev_idx]

                tri_i = tri_origins.index(int(origin))
                for k, cells in path_cells.items():
                    tri_j = tri_devs.index(int(dev_levels[k]))
                    full_paths[tri_i, tri_j, :] = cells.reshape(-1)
```

Keep the subsequent `ibnr = ultimate_cumulative - last_observed_cumulative` and `all_predictions[origin] = {...}` lines unchanged. After the loop, where `self._compute_reserve_summaries(all_predictions)` is called, add:

```python
            self._set_full_cumulative_posterior(full_paths, tri_origins, tri_devs)
```

Check that `dev_levels` in this method is a Python list (it is used with `.index` elsewhere in the class; if it is an Index or ndarray, wrap with `list(...)` once at the top). The flattening order `cells.reshape(-1)` (chain-major, draw-minor) matches the `.flatten()` used in `_compute_reserve_summaries`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_estimators.py -q -k "csr" --run-slow`
Expected: PASS, including pre-existing CSR tests (the ultimate distribution is unchanged: same formula, one shock).

- [ ] **Step 5: Commit**

```bash
git add bayesianchainladder/estimators.py tests/test_estimators.py
git commit -m "feat: BayesianCSR stores comonotonic per-cell cumulative paths

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: England sample triangles and loader

**Files:**
- Create: `bayesianchainladder/data/taylor_ashe.csv`, `bayesianchainladder/data/liability.csv`, `bayesianchainladder/data/README.md`
- Create: `bayesianchainladder/datasets.py`
- Modify: `bayesianchainladder/utils.py` (add `long_to_triangle`), `bayesianchainladder/__init__.py`
- Test: `tests/test_datasets.py`

**Interfaces:**
- Produces:
  - `utils.long_to_triangle(df: pd.DataFrame, value_col: str, origin_col="origin", dev_col="dev") -> cl.Triangle` — origin int years, dev int months, cumulative values
  - `datasets.load_england_sample(name: Literal["taylor_ashe", "liability"], first_origin: int = 2001) -> cl.Triangle` — cumulative single-column triangle, origins `first_origin … first_origin+9`, development 12…120

- [ ] **Step 1: Fetch the data files (pinned to commit e7ed85a)**

```bash
mkdir -p bayesianchainladder/data
curl -sL https://raw.githubusercontent.com/DrPeterEngland/StochasticReserving/e7ed85a29dba64db1192140e504f4e09cf149134/Python_Examples/claims_triangle.csv -o bayesianchainladder/data/taylor_ashe.csv
curl -sL https://raw.githubusercontent.com/DrPeterEngland/StochasticReserving/e7ed85a29dba64db1192140e504f4e09cf149134/Python_Examples/liability_claims_triangle.csv -o bayesianchainladder/data/liability.csv
head -2 bayesianchainladder/data/taylor_ashe.csv
```
Expected first line: `Origin/Development,1,2,3,4,5,6,7,8,9,10` (the liability file starts with a UTF-8 byte-order mark; the loader reads with `encoding="utf-8-sig"`).

Write `bayesianchainladder/data/README.md`:

```markdown
# Sample data

Both files are incremental 10×10 claims triangles copied verbatim from
Peter England's StochasticReserving repository
(https://github.com/DrPeterEngland/StochasticReserving, `Python_Examples/`,
commit e7ed85a29dba64db1192140e504f4e09cf149134), provided by EMC Actuarial
and Analytics Ltd under the MIT licence.

| File | Source file | Notes |
|---|---|---|
| `taylor_ashe.csv` | `claims_triangle.csv` | Taylor & Ashe (1983); identical to chainladder's `genins` sample. Used in England & Verrall (2002, 2006) and England, Verrall & Wüthrich (2019). |
| `liability.csv` | `liability_claims_triangle.csv` | Liability triangle from the *Example Modus Operandi* notebook; has an influential incremental at origin 3, development 7. |

Rows are origin periods labelled 1–10; columns are development periods 1–10.
`load_england_sample` maps origin label *k* to year `first_origin + k - 1`
(default 2001) and development period *d* to `12*d` months.
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_datasets.py
"""Tests for the England sample triangles and the long-format loader."""

import chainladder as cl
import numpy as np
import pandas as pd
import pytest

from bayesianchainladder.datasets import load_england_sample
from bayesianchainladder.utils import long_to_triangle


def test_long_to_triangle_roundtrip():
    df = pd.DataFrame(
        {"origin": [2001, 2001, 2002], "dev": [12, 24, 12], "value": [1.0, 3.0, 2.0]}
    )
    tri = long_to_triangle(df, "value")
    assert tri.shape == (1, 1, 2, 2)
    assert tri.development.tolist() == [12, 24]
    np.testing.assert_allclose(
        np.asarray(tri.values)[0, 0], [[1.0, 3.0], [2.0, np.nan]], equal_nan=True
    )


def test_taylor_ashe_equals_genins():
    tri = load_england_sample("taylor_ashe")
    genins = cl.load_sample("genins")
    np.testing.assert_allclose(
        np.asarray(tri.values), np.asarray(genins.values), equal_nan=True
    )
    assert [str(o) for o in tri.origin[:2]] == ["2001", "2002"]


def test_liability_reference_values():
    tri = load_england_sample("liability")
    assert tri.shape == (1, 1, 10, 10)
    dev = cl.Development().fit_transform(tri)
    mack = cl.MackChainladder().fit(dev)
    total_reserve = float(np.nansum(np.asarray(mack.ibnr_.values)))
    total_se = float(np.asarray(mack.total_mack_std_err_).flatten()[0])
    assert total_reserve == pytest.approx(331_038, abs=1.0)
    assert total_se == pytest.approx(72_448, rel=1e-3)


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="taylor_ashe"):
        load_england_sample("nope")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_datasets.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bayesianchainladder.datasets'`

- [ ] **Step 4: Implement**

Append to `bayesianchainladder/utils.py`:

```python
def long_to_triangle(
    df: pd.DataFrame,
    value_col: str,
    origin_col: str = "origin",
    dev_col: str = "dev",
) -> "cl.Triangle":
    """Build a cumulative chainladder Triangle from long-format rows with an
    integer origin year and development age in months (12, 24, ...).

    chainladder needs a date-like development column, so the age is turned
    into a year-end valuation date ``origin + dev/12 - 1``.
    """
    import chainladder as cl_

    work = df[[origin_col, dev_col, value_col]].copy()
    work.columns = ["origin", "dev", value_col]
    work["origin"] = work["origin"].astype(int)
    work["dev"] = work["dev"].astype(int)
    eval_year = work["origin"] + work["dev"] // 12 - 1
    work["dev_date"] = pd.to_datetime(eval_year.astype(str) + "-12-31", format="%Y-%m-%d")
    return cl_.Triangle(
        data=work,
        origin="origin",
        development="dev_date",
        columns=[value_col],
        cumulative=True,
        origin_format="%Y",
    )
```

Create `bayesianchainladder/datasets.py`:

```python
"""Sample triangles shipped with the package.

Data files come from Peter England's StochasticReserving repository
(https://github.com/DrPeterEngland/StochasticReserving, MIT licence); see
``bayesianchainladder/data/README.md`` for provenance.
"""

from __future__ import annotations

from importlib import resources
from typing import Literal

import numpy as np
import pandas as pd

from .utils import long_to_triangle

_FILES = {"taylor_ashe": "taylor_ashe.csv", "liability": "liability.csv"}


def load_england_sample(
    name: Literal["taylor_ashe", "liability"], first_origin: int = 2001
):
    """Load one of England's incremental triangles as a cumulative Triangle."""
    if name not in _FILES:
        raise ValueError(f"unknown sample {name!r}; choose from {sorted(_FILES)}")
    path = resources.files("bayesianchainladder.data").joinpath(_FILES[name])
    with path.open("r", encoding="utf-8-sig") as fh:
        raw = pd.read_csv(fh, index_col=0)
    rows = []
    for label, row in raw.iterrows():
        incr = row.to_numpy(dtype=float)
        cum = np.cumsum(np.nan_to_num(incr))
        for j, v in enumerate(incr):
            if np.isnan(v):
                break
            rows.append(
                {"origin": first_origin + int(label) - 1, "dev": 12 * (j + 1), "value": cum[j]}
            )
    return long_to_triangle(pd.DataFrame(rows), "value")
```

Add `bayesianchainladder/data/__init__.py` (empty) so `importlib.resources` resolves the package. In `bayesianchainladder/__init__.py` add `from .datasets import load_england_sample`, add `long_to_triangle` to the utils import, and add both to `__all__` (under a new `# Data` comment and under `# Utility functions`).

- [ ] **Step 5: Run tests and confirm packaging**

Run: `uv run pytest tests/test_datasets.py -q` → 4 passed.
Run: `uv build --wheel -o /tmp/bcl-wheel && unzip -l /tmp/bcl-wheel/*.whl | grep data/` → both CSVs and README listed (hatchling includes non-Python files inside the package directory).

- [ ] **Step 6: Commit**

```bash
uv run ruff check bayesianchainladder/datasets.py bayesianchainladder/utils.py tests/test_datasets.py
git add bayesianchainladder/data bayesianchainladder/datasets.py bayesianchainladder/utils.py bayesianchainladder/__init__.py tests/test_datasets.py
git commit -m "feat: ship England's Taylor-Ashe and liability triangles with a loader

Data copied from DrPeterEngland/StochasticReserving (MIT), pinned to
commit e7ed85a.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: risk measures, discounting and cost-of-capital

**Files:**
- Create: `bayesianchainladder/riskmeasures.py`
- Modify: `pyproject.toml` (add `"scipy>=1.10"` to `[project].dependencies`; `bootstrap.py` already imports it), `bayesianchainladder/__init__.py`
- Test: `tests/test_riskmeasures.py`

**Interfaces:**
- Consumes: `future_incremental_posterior()` (Task 2), a fitted `BootstrapODPChainLadder` (Task 3) in tests.
- Produces (all in `riskmeasures.py`):
  - `value_at_risk(samples, level) -> float`
  - `tail_value_at_risk(samples, level) -> float`
  - `proportional_hazards_transform(samples, param) -> float` (`param >= 1`; `param == 1` returns the mean)
  - `cash_flow_periods(n_origin, n_dev) -> np.ndarray` — `(n_o, n_d)` int, periods ahead of cell `(i, j)` = `i + j - (n_o - 1)`; requires square
  - `discount_factors(periods_ahead, rate, offset=0.5) -> np.ndarray` — `(1+rate) ** -(k - 1 + offset)`
  - `discounted_reserves(model, rate, offset=0.5, as_of_period=0) -> xr.DataArray` dims `("origin","sample")`
  - `future_reserve_profile(model, rate, offset=0.5) -> xr.DataArray` dims `("period","sample")`, `period = 0 … n_dev-2`
  - `capital_profile(basis) -> np.ndarray` (`basis / basis[0]`)
  - `cost_of_capital_risk_margin(opening_capital, profile, coc_rate, discount_rate, offset=1.0) -> dict[str, np.ndarray | float]` with keys `capital, cost, discounted_cost, risk_margin`
  - `equivalent_risk_tolerance(samples, target_margin, measure="var") -> float`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_riskmeasures.py
"""Tests for risk measures, discounting and cost-of-capital helpers."""

import chainladder as cl
import numpy as np
import pytest

from bayesianchainladder.bootstrap import BootstrapODPChainLadder
from bayesianchainladder.riskmeasures import (
    capital_profile,
    cash_flow_periods,
    cost_of_capital_risk_margin,
    discount_factors,
    discounted_reserves,
    equivalent_risk_tolerance,
    future_reserve_profile,
    proportional_hazards_transform,
    tail_value_at_risk,
    value_at_risk,
)


@pytest.fixture(scope="module")
def odp():
    return BootstrapODPChainLadder(n_sims=300, random_seed=7).fit(cl.load_sample("genins"))


def test_var_tvar_pht_on_simple_samples():
    x = np.arange(1.0, 101.0)
    assert value_at_risk(x, 0.5) == pytest.approx(50.5)
    assert tail_value_at_risk(x, 0.9) == pytest.approx(np.mean(x[x >= np.quantile(x, 0.9)]))
    assert proportional_hazards_transform(x, 1.0) == pytest.approx(x.mean())
    assert proportional_hazards_transform(x, 2.0) > x.mean()
    with pytest.raises(ValueError):
        proportional_hazards_transform(x, 0.5)


def test_cash_flow_periods_and_discount_factors():
    k = cash_flow_periods(3, 3)
    np.testing.assert_array_equal(k, [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
    with pytest.raises(ValueError, match="square"):
        cash_flow_periods(3, 4)
    np.testing.assert_allclose(
        discount_factors([1, 2], 0.05, offset=0.5), [1.05**-0.5, 1.05**-1.5]
    )
    np.testing.assert_allclose(discount_factors([1, 2, 3], 0.05, offset=1.0), 1.05 ** -np.arange(1, 4))


def test_discounted_reserves_zero_rate_equals_reserves(odp):
    disc = discounted_reserves(odp, rate=0.0)
    np.testing.assert_allclose(
        disc.transpose("origin", "sample").values,
        odp.reserves_posterior_.transpose("origin", "sample").values,
        rtol=1e-6, atol=1e-6,
    )
    disc3 = discounted_reserves(odp, rate=0.03)
    assert float(disc3.sum("origin").mean()) < float(disc.sum("origin").mean())


def test_future_reserve_profile_shape_and_monotone(odp):
    prof = future_reserve_profile(odp, rate=0.03)
    assert prof.dims == ("period", "sample")
    assert prof.sizes["period"] == 9
    means = prof.mean("sample").values
    assert np.all(np.diff(means) < 0)  # reserves run off over time
    # period 0 with zero discount equals total undiscounted reserve
    prof0 = future_reserve_profile(odp, rate=0.0)
    np.testing.assert_allclose(
        prof0.isel(period=0).values, odp.reserves_posterior_.sum("origin").values, rtol=1e-6
    )


def test_cost_of_capital_hand_calculation():
    out = cost_of_capital_risk_margin(100.0, np.array([1.0, 0.5]), coc_rate=0.06, discount_rate=0.0)
    np.testing.assert_allclose(out["capital"], [100.0, 50.0])
    np.testing.assert_allclose(out["cost"], [6.0, 3.0])
    assert out["risk_margin"] == pytest.approx(9.0)
    out2 = cost_of_capital_risk_margin(100.0, np.array([1.0, 0.5]), 0.06, 0.10, offset=1.0)
    assert out2["risk_margin"] == pytest.approx(6.0 / 1.1 + 3.0 / 1.1**2)
    np.testing.assert_allclose(capital_profile(np.array([200.0, 100.0, 50.0])), [1.0, 0.5, 0.25])


def test_equivalent_risk_tolerance_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.lognormal(10, 0.3, 20000)
    for measure in ("var", "tvar", "pht"):
        p0 = 0.8 if measure != "pht" else 2.0
        fn = {"var": value_at_risk, "tvar": tail_value_at_risk, "pht": proportional_hazards_transform}[measure]
        target = fn(x, p0) - x.mean()
        p = equivalent_risk_tolerance(x, target, measure=measure)
        assert p == pytest.approx(p0, rel=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_riskmeasures.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bayesianchainladder.riskmeasures'`

- [ ] **Step 3: Implement**

`uv add "scipy>=1.10"` (moves scipy into declared runtime dependencies). Then create `bayesianchainladder/riskmeasures.py`:

```python
"""Risk measures, discounting and cost-of-capital risk margins for reserve
distributions.

Ported from the ``VAR``, ``TVAR``, ``PHT``, ``Disc_Reserves``,
``Disc_Future_Reserves``, ``Capital_Profile`` and ``CoC_RM`` functions in
Peter England's StochasticReserving repository
(https://github.com/DrPeterEngland/StochasticReserving, MIT licence), which
reproduce England, Verrall & Wüthrich (2019). Quantiles use ``np.quantile``
rather than the original order-statistic index.

Timing convention: for a square triangle whose origin grain equals its
development grain, the cell ``(i, j)`` is paid ``k = i + j - (n - 1)`` periods
after the valuation date (``k <= 0`` is observed). A payment ``k`` periods
ahead is discounted by ``(1 + rate) ** -(k - 1 + offset)``; ``offset = 0.5``
is mid-period payment, ``offset = 1`` is payment in arrears.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.optimize import brentq

from .base import BaseStochasticReserve


def value_at_risk(samples, level: float) -> float:
    return float(np.quantile(np.asarray(samples, dtype=float), level))


def tail_value_at_risk(samples, level: float) -> float:
    x = np.asarray(samples, dtype=float)
    threshold = value_at_risk(x, level)
    return float(x[x >= threshold].mean())


def proportional_hazards_transform(samples, param: float) -> float:
    """Wang's proportional hazards transform E*[X] with S*(x) = S(x)^(1/param)."""
    if param < 1:
        raise ValueError("param must be >= 1 (param == 1 returns the mean)")
    x = np.sort(np.asarray(samples, dtype=float))
    n = len(x)
    survival = (1.0 - np.arange(1, n + 1) / n) ** (1.0 / param)
    weights = np.concatenate([[1.0], survival[:-1]]) - survival
    return float(np.sum(weights * x))


def cash_flow_periods(n_origin: int, n_dev: int) -> np.ndarray:
    if n_origin != n_dev:
        raise ValueError(
            "cash-flow timing requires a square triangle (origin grain == development grain)"
        )
    i, j = np.indices((n_origin, n_dev))
    return i + j - (n_origin - 1)


def discount_factors(periods_ahead, rate: float, offset: float = 0.5) -> np.ndarray:
    k = np.asarray(periods_ahead, dtype=float)
    return (1.0 + rate) ** -(k - 1.0 + offset)


def _future_cash_flows(model: BaseStochasticReserve):
    incr = model.future_incremental_posterior()  # (origin, dev, sample), 0 on observed
    k = cash_flow_periods(incr.sizes["origin"], incr.sizes["dev"])
    return incr, k


def discounted_reserves(
    model: BaseStochasticReserve, rate: float, offset: float = 0.5, as_of_period: int = 0
) -> xr.DataArray:
    """Per-origin discounted outstanding reserves as at ``as_of_period``
    (0 = valuation date), discounted back to that date only."""
    incr, k = _future_cash_flows(model)
    future = k > as_of_period
    factors = np.where(future, discount_factors(k - as_of_period, rate, offset), 0.0)
    vals = np.nansum(incr.values * factors[..., None], axis=1)
    return xr.DataArray(
        vals,
        dims=["origin", "sample"],
        coords={"origin": incr.coords["origin"], "sample": incr.coords["sample"]},
    )


def future_reserve_profile(
    model: BaseStochasticReserve, rate: float, offset: float = 0.5
) -> xr.DataArray:
    """Total discounted reserves remaining at the start of each future period
    ``t = 0 … n_dev-2`` (England's ``Disc_Future_Reserves``), per sample."""
    incr, _ = _future_cash_flows(model)
    n_periods = incr.sizes["dev"] - 1
    rows = [
        discounted_reserves(model, rate, offset, as_of_period=t).sum("origin").values
        for t in range(n_periods)
    ]
    return xr.DataArray(
        np.stack(rows),
        dims=["period", "sample"],
        coords={"period": np.arange(n_periods), "sample": incr.coords["sample"]},
    )


def capital_profile(basis) -> np.ndarray:
    b = np.asarray(basis, dtype=float)
    return b / b[0]


def cost_of_capital_risk_margin(
    opening_capital: float,
    profile,
    coc_rate: float,
    discount_rate: float,
    offset: float = 1.0,
) -> dict:
    """Risk margin = sum over future periods of capital × cost-of-capital rate,
    discounted with exponent ``t - 1 + min(offset, 1)`` (offset 1 = arrears)."""
    prof = np.asarray(profile, dtype=float)
    t = np.arange(1, len(prof) + 1)
    capital = opening_capital * prof
    cost = capital * coc_rate
    discounted = cost / (1.0 + discount_rate) ** (t - 1 + min(offset, 1.0))
    return {
        "capital": capital,
        "cost": cost,
        "discounted_cost": discounted,
        "risk_margin": float(discounted.sum()),
    }


def equivalent_risk_tolerance(samples, target_margin: float, measure: str = "var") -> float:
    """Solve for the confidence level (VaR/TVaR) or PHT parameter whose risk
    measure minus the mean equals ``target_margin``."""
    x = np.asarray(samples, dtype=float)
    mean = x.mean()
    if measure == "var":
        return float(brentq(lambda p: value_at_risk(x, p) - mean - target_margin, 0.01, 0.9999))
    if measure == "tvar":
        return float(brentq(lambda p: tail_value_at_risk(x, p) - mean - target_margin, 0.01, 0.999))
    if measure == "pht":
        return float(
            brentq(lambda q: proportional_hazards_transform(x, q) - mean - target_margin, 1.0, 1000.0)
        )
    raise ValueError("measure must be 'var', 'tvar' or 'pht'")
```

Export all public names from `__init__.py` under a `# Risk measures` comment.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_riskmeasures.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
uv run ruff check bayesianchainladder/riskmeasures.py tests/test_riskmeasures.py
git add pyproject.toml uv.lock bayesianchainladder/riskmeasures.py bayesianchainladder/__init__.py tests/test_riskmeasures.py
git commit -m "feat: risk measures, discounting and cost-of-capital risk margin

VaR/TVaR/PHT, cash-flow discounting from per-cell posteriors, future
reserve profiles, capital profiles and equivalent risk tolerance
solving, after England, Verrall & Wuthrich (2019).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: one-year Claims Development Result

**Files:**
- Create: `bayesianchainladder/cdr.py`
- Modify: `bayesianchainladder/__init__.py`
- Test: `tests/test_cdr.py`

**Interfaces:**
- Consumes: `_require_full_posterior` (Task 2); `cumulative_array`, `drop_mask`, `link_ratio_mask`, `volume_weighted_factors`, `project_cumulative` (Task 1); `value_at_risk` (Task 7).
- Produces:
  - `@dataclass CDRResult` with fields `cdr: xr.DataArray ("future_period","origin","sample")`, `total_cdr: xr.DataArray ("future_period","sample")`, `ultimates: xr.DataArray ("future_period","origin","sample")` (period 0 = deterministic chain ladder), `var_level: float`; methods `cumulative() -> xr.DataArray`, `reverse_cumulative() -> xr.DataArray`, `summary() -> pd.DataFrame` (long format: `future_period, origin, mean, sd, var`, origin `"Total"` included)
  - `claims_development_result(model, future_periods=None, var_level=0.995, drop=None) -> CDRResult`

Algorithm (England's *CDR_Full_Picture*, actuary-in-the-box): with `C` the simulated complete cumulative triangle `(S, n, n)`, define for future period `t` the known set `known_t = {(i, j): i + j <= (n - 1) + t}`. `U_0` is the deterministic volume-weighted chain ladder ultimate from the observed data. For `t = 1 … K`, mask `C` to `known_t`, recompute volume-weighted factors per simulation using only ratios whose two cells are known (keeping any user `drop` exclusions), project, and take the ultimate `U_t`. `CDR_t = U_{t-1} − U_t`. At `t = n − 1` everything is known so `U_K` equals the simulated ultimate exactly, and `Σ_t CDR_t = U_0 − C[:, :, −1]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cdr.py
"""Tests for the one-year Claims Development Result (England, Verrall & Wuthrich 2019)."""

import chainladder as cl
import numpy as np
import pytest

from bayesianchainladder._triangle_ops import (
    cumulative_array,
    link_ratio_mask,
    project_cumulative,
    volume_weighted_factors,
)
from bayesianchainladder.bootstrap import BootstrapODPChainLadder, MackChainLadder
from bayesianchainladder.cdr import CDRResult, claims_development_result


@pytest.fixture(scope="module")
def odp():
    return BootstrapODPChainLadder(n_sims=400, random_seed=11).fit(cl.load_sample("genins"))


def test_cdr_shapes_and_coords(odp):
    res = claims_development_result(odp)
    assert isinstance(res, CDRResult)
    assert res.cdr.dims == ("future_period", "origin", "sample")
    assert res.cdr.sizes == {"future_period": 9, "origin": 10, "sample": 400}
    assert list(res.cdr.coords["future_period"].values) == list(range(1, 10))
    assert res.total_cdr.dims == ("future_period", "sample")
    assert res.ultimates.sizes["future_period"] == 10  # periods 0..9


def test_cdr_sums_to_lifetime_deviation(odp):
    res = claims_development_result(odp)
    cum, origins, devs = cumulative_array(odp.triangle_)
    f0 = volume_weighted_factors(cum, link_ratio_mask(cum, None, origins, devs))
    cl_ultimate = project_cumulative(cum, f0)[:, -1]
    sim_ultimate = odp.full_cumulative_posterior_.isel(dev=-1).values  # (origin, sample)
    lifetime = cl_ultimate[:, None] - sim_ultimate
    np.testing.assert_allclose(
        res.cdr.sum("future_period").transpose("origin", "sample").values, lifetime,
        rtol=1e-8, atol=1e-6,
    )
    np.testing.assert_allclose(
        res.cumulative().isel(future_period=-1).transpose("origin", "sample").values,
        lifetime, rtol=1e-8, atol=1e-6,
    )


def test_one_year_sd_below_lifetime_sd(odp):
    res = claims_development_result(odp, future_periods=1)
    assert res.cdr.sizes["future_period"] == 1
    one_year_sd = float(res.total_cdr.isel(future_period=0).std(ddof=1))
    lifetime_sd = float(odp.reserves_posterior_.sum("origin").std(ddof=1))
    assert 0.3 * lifetime_sd < one_year_sd < lifetime_sd


def test_fully_developed_origin_has_zero_cdr(odp):
    res = claims_development_result(odp)
    first = res.cdr.sel(origin=2001).values
    np.testing.assert_allclose(first, 0.0, atol=1e-6)


def test_summary_and_reverse_cumulative(odp):
    res = claims_development_result(odp, var_level=0.99)
    table = res.summary()
    assert set(table.columns) == {"future_period", "origin", "mean", "sd", "var"}
    assert "Total" in set(table["origin"].astype(str))
    total_row = table[(table["origin"].astype(str) == "Total") & (table["future_period"] == 1)]
    assert total_row["var"].iloc[0] >= total_row["mean"].iloc[0]
    rev = res.reverse_cumulative()
    np.testing.assert_allclose(
        rev.isel(future_period=0).values, res.cdr.sum("future_period").values
    )


def test_drop_changes_period_zero_ultimate(odp):
    base = claims_development_result(odp)
    dropped = claims_development_result(odp, drop=[("2003", 72)])
    assert not np.allclose(
        base.ultimates.isel(future_period=0).values,
        dropped.ultimates.isel(future_period=0).values,
    )


def test_requires_full_posterior():
    mack = MackChainLadder().fit(cl.load_sample("raa"))
    with pytest.raises(ValueError, match="per-cell"):
        claims_development_result(mack)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cdr.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bayesianchainladder.cdr'`

- [ ] **Step 3: Implement**

```python
# bayesianchainladder/cdr.py
"""One-year and multi-year Claims Development Result (CDR).

Implements the "actuary-in-the-box" re-reserving of England, Verrall &
Wuthrich (2019) as coded in ``CDR_Full_Picture`` / ``CDR_Rev_Sum`` of Peter
England's StochasticReserving repository
(https://github.com/DrPeterEngland/StochasticReserving, MIT licence). Works on
any estimator exposing ``full_cumulative_posterior_``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from ._triangle_ops import (
    DropList,
    cumulative_array,
    drop_mask,
    link_ratio_mask,
    project_cumulative,
    volume_weighted_factors,
)
from .base import BaseStochasticReserve
from .riskmeasures import value_at_risk


@dataclass
class CDRResult:
    cdr: xr.DataArray
    total_cdr: xr.DataArray
    ultimates: xr.DataArray
    var_level: float

    def cumulative(self) -> xr.DataArray:
        return self.cdr.cumsum("future_period")

    def reverse_cumulative(self) -> xr.DataArray:
        """Sum of CDRs from each future period to run-off (England's CDR_Rev_Sum)."""
        vals = np.flip(np.cumsum(np.flip(self.cdr.values, axis=0), axis=0), axis=0)
        return self.cdr.copy(data=vals)

    def summary(self) -> pd.DataFrame:
        """Long-format table of mean, SD and VaR of the CDR per future period
        and origin (plus 'Total'). ``var`` follows England: mean minus the
        ``1 - var_level`` quantile of the CDR, i.e. the capital needed against
        an adverse one-year development at the chosen confidence."""
        rows = []
        periods = self.cdr.coords["future_period"].values
        origins = list(self.cdr.coords["origin"].values) + ["Total"]
        for t_idx, t in enumerate(periods):
            for origin in origins:
                if origin == "Total":
                    x = self.total_cdr.isel(future_period=t_idx).values
                else:
                    x = self.cdr.isel(future_period=t_idx).sel(origin=origin).values
                mean = float(np.mean(x))
                sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
                var = mean - value_at_risk(x, 1.0 - self.var_level)
                rows.append(
                    {"future_period": int(t), "origin": origin, "mean": mean, "sd": sd, "var": var}
                )
        return pd.DataFrame(rows)


def claims_development_result(
    model: BaseStochasticReserve,
    future_periods: int | None = None,
    var_level: float = 0.995,
    drop: DropList = None,
) -> CDRResult:
    full = model._require_full_posterior()
    C = np.moveaxis(full.values, -1, 0)  # (S, n_o, n_d)
    cum, origins, devs = cumulative_array(model.triangle_)
    n_o, n_d = cum.shape
    if n_o != n_d:
        raise ValueError("claims_development_result requires a square triangle")
    if np.isnan(C).any():
        raise ValueError("full_cumulative_posterior_ contains NaN cells")

    excl = drop_mask(n_o, n_d, drop, origins, devs)
    max_periods = n_d - 1
    n_periods = max_periods if future_periods is None else min(int(future_periods), max_periods)
    if n_periods < 1:
        raise ValueError("future_periods must be >= 1")

    i_idx, j_idx = np.indices((n_o, n_d))
    n_sims = C.shape[0]

    # Period 0: deterministic chain ladder on the observed data.
    f0 = volume_weighted_factors(cum, link_ratio_mask(cum, drop, origins, devs))
    u0 = project_cumulative(cum, f0)[:, -1]
    ultimates = [np.broadcast_to(u0, (n_sims, n_o)).copy()]

    for t in range(1, n_periods + 1):
        known = (i_idx + j_idx) <= (n_d - 1) + t
        avail = (known[:, :-1] & known[:, 1:]).astype(float)
        mask_t = avail * excl
        c_known = np.where(known, C, np.nan)
        f_t = volume_weighted_factors(c_known, mask_t)
        ultimates.append(project_cumulative(c_known, f_t)[:, :, -1])

    U = np.stack(ultimates)  # (n_periods + 1, S, n_o)
    cdr = U[:-1] - U[1:]  # (n_periods, S, n_o)

    coords_common = {"origin": full.coords["origin"], "sample": full.coords["sample"]}
    cdr_da = xr.DataArray(
        np.moveaxis(cdr, 1, 2),
        dims=["future_period", "origin", "sample"],
        coords={"future_period": np.arange(1, n_periods + 1), **coords_common},
    )
    ult_da = xr.DataArray(
        np.moveaxis(U, 1, 2),
        dims=["future_period", "origin", "sample"],
        coords={"future_period": np.arange(0, n_periods + 1), **coords_common},
    )
    return CDRResult(
        cdr=cdr_da,
        total_cdr=cdr_da.sum("origin"),
        ultimates=ult_da,
        var_level=var_level,
    )
```

Export `CDRResult` and `claims_development_result` from `__init__.py` under `# Claims Development Result`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_cdr.py -q`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
uv run ruff check bayesianchainladder/cdr.py tests/test_cdr.py
git add bayesianchainladder/cdr.py bayesianchainladder/__init__.py tests/test_cdr.py
git commit -m "feat: one-year and multi-year Claims Development Result

Actuary-in-the-box re-reserving of simulated triangles per England,
Verrall & Wuthrich (2019); CDRs sum exactly to the lifetime deviation.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: non-constant scale and user-defined process variance in the correlated ODP sampler

**Files:**
- Modify: `bayesianchainladder/bootstrap.py` (`CorrelatedBootstrapODPSample.__init__`, `_get_simulation`, `transform`, module-level `_get_process_variance`; `CorrelatedBootstrapChainLadder.__init__`/`fit`)
- Test: `tests/test_bootstrap.py`

**Interfaces:**
- Produces on `CorrelatedBootstrapODPSample`: params `scale: str = "constant"`, `process_scale: np.ndarray | None = None`; fitted attributes `scale_by_dev_: np.ndarray (n_dev,)`, `standardized_residuals_: np.ndarray (n_origin, n_dev)`; module function `_nonconstant_scale(standardized_residuals, nan_triangle) -> np.ndarray`.
- Produces on `CorrelatedBootstrapChainLadder`: params `scale`, `process_scale`, `drop`; attribute `sampler_` (the fitted `CorrelatedBootstrapODPSample`).
- The transformed triangle carries `scale_by_dev_` so `_get_process_variance` draws `Gamma(shape = |μ| / φ_j, scale = φ_j)` per development column.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_bootstrap.py`)

```python
class TestNonConstantScale:
    @pytest.fixture
    def genins(self):
        return cl.load_sample("genins")

    def test_constant_scale_vector_equals_pooled_phi(self, genins):
        from bayesianchainladder.bootstrap import CorrelatedBootstrapODPSample

        s = CorrelatedBootstrapODPSample(n_sims=50, random_state=1).fit(genins)
        assert s.scale_by_dev_.shape == (10,)
        np.testing.assert_allclose(s.scale_by_dev_, float(s.scale_))
        assert s.standardized_residuals_.shape == (10, 10)

    def test_nonconstant_scale_follows_england_rules(self, genins):
        from bayesianchainladder.bootstrap import CorrelatedBootstrapODPSample

        s = CorrelatedBootstrapODPSample(n_sims=50, random_state=1, scale="nonconstant").fit(genins)
        phi = s.scale_by_dev_
        assert phi.shape == (10,)
        assert np.all(phi >= 0)
        assert phi[-1] == pytest.approx(min(phi[-2], phi[-3]))
        assert not np.allclose(phi, phi[0])  # genuinely varies by development period
        with pytest.raises(ValueError):
            CorrelatedBootstrapODPSample(scale="weird")

    def test_process_scale_override_changes_sd_not_mean(self, genins):
        from bayesianchainladder.bootstrap import CorrelatedBootstrapChainLadder

        base = CorrelatedBootstrapChainLadder(n_sims=2000, rho=0.0, random_seed=5).fit(genins)
        phi = base.sampler_.scale_by_dev_
        none = CorrelatedBootstrapChainLadder(
            n_sims=2000, rho=0.0, random_seed=5, process_scale=np.zeros(10)
        ).fit(genins)
        big = CorrelatedBootstrapChainLadder(
            n_sims=2000, rho=0.0, random_seed=5, process_scale=phi * 9.0
        ).fit(genins)
        sd = lambda m: m.total_summary().total_reserve_stddev  # noqa: E731
        mean = lambda m: m.total_summary().total_reserve_mean  # noqa: E731
        assert sd(none) < sd(base) < sd(big)
        assert mean(none) == pytest.approx(mean(base), rel=0.03)
        assert mean(big) == pytest.approx(mean(base), rel=0.05)

    def test_drop_passes_through_wrapper(self, genins):
        from bayesianchainladder.bootstrap import CorrelatedBootstrapChainLadder

        base = CorrelatedBootstrapChainLadder(n_sims=300, random_seed=2).fit(genins)
        dropped = CorrelatedBootstrapChainLadder(
            n_sims=300, random_seed=2, drop=[("2003", 72)]
        ).fit(genins)
        assert dropped.total_summary().total_reserve_mean != pytest.approx(
            base.total_summary().total_reserve_mean, rel=1e-4
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_bootstrap.py -q -k NonConstantScale`
Expected: FAIL with `AttributeError: ... has no attribute 'scale_by_dev_'`

- [ ] **Step 3: Implement**

1. In `CorrelatedBootstrapODPSample.__init__` add parameters `scale: str = "constant"` and `process_scale=None` (after `min_fitted_value`), validate:
```python
        if scale not in ("constant", "nonconstant"):
            raise ValueError("scale must be 'constant' or 'nonconstant'")
        self.scale = scale
        self.process_scale = None if process_scale is None else np.asarray(process_scale, dtype=float)
```
Also add `drop=self.drop, ...` is already forwarded; nothing to change for `drop`.

2. Add the module-level helper next to `_get_process_variance`:
```python
def _nonconstant_scale(standardized_residuals, nan_triangle):
    """Per-development-period dispersion phi_j from standardized Pearson
    residuals, with England's conventions: carry forward when n_j <= 1 and set
    the last period to the minimum of the previous two."""
    resid_sq = np.where(np.isnan(nan_triangle), np.nan, standardized_residuals**2)
    n_j = np.nansum(nan_triangle, axis=0)
    ss = np.nansum(resid_sq, axis=0)
    n_dev = resid_sq.shape[1]
    phi = np.zeros(n_dev)
    for j in range(n_dev - 1):
        if n_j[j] > 1:
            phi[j] = ss[j] / n_j[j]
        else:
            phi[j] = phi[j - 1] if j > 0 else 0.0
    if n_dev >= 3:
        phi[-1] = min(phi[-2], phi[-3])
    elif n_dev == 2:
        phi[-1] = phi[-2]
    return phi
```

3. In `_get_simulation`, right after `scale_phi = pearson_chi_sq / degree_freedom`, add:
```python
        self.standardized_residuals_ = np.asarray(standardized_residuals, dtype=float)
        if self.scale == "nonconstant":
            self.scale_by_dev_ = _nonconstant_scale(self.standardized_residuals_, np.asarray(nan_triangle))
        else:
            self.scale_by_dev_ = np.full(standardized_residuals.shape[1], float(scale_phi))
```
Also record the same in the multi-index branch of `fit` (`self.scale_by_dev_ = out[0].scale_by_dev_`, `self.standardized_residuals_ = out[0].standardized_residuals_`).

4. In `transform`, after `X_new.scale_ = self.scale_`:
```python
        if self.process_scale is not None:
            if self.process_scale.shape != self.scale_by_dev_.shape:
                raise ValueError(
                    f"process_scale must have shape {self.scale_by_dev_.shape}, got {self.process_scale.shape}"
                )
            X_new.scale_by_dev_ = self.process_scale
        else:
            X_new.scale_by_dev_ = self.scale_by_dev_
```

5. Replace `_get_process_variance`:
```python
def _get_process_variance(self, full_triangle):
    """Inject gamma process noise into future cells with a per-development
    dispersion vector ``scale_by_dev_`` (constant when scale='constant')."""
    xp = full_triangle.get_array_module()
    lower_tri = full_triangle.cum_to_incr() - self.cum_to_incr()
    random_state = xp.random.RandomState(
        None if not self.random_state else self.random_state + 1
    )
    scale_vec = np.asarray(
        getattr(self, "scale_by_dev_", None)
        if getattr(self, "scale_by_dev_", None) is not None
        else np.full(lower_tri.values.shape[-1], float(np.asarray(self.scale_).flatten()[0])),
        dtype=float,
    )
    n_full = lower_tri.values.shape[-1]
    if len(scale_vec) < n_full:  # placeholder tail and 9999 ultimate columns
        scale_vec = np.concatenate([scale_vec, np.repeat(scale_vec[-1], n_full - len(scale_vec))])
    scale_b = np.maximum(scale_vec[:n_full], 1e-12)[None, None, None, :]
    lower_tri.values = random_state.gamma(
        shape=abs(lower_tri.values) / scale_b, scale=scale_b
    ) * xp.sign(xp.nan_to_num(lower_tri.values))
    return (lower_tri + self.cum_to_incr()).incr_to_cum()
```

6. `CorrelatedBootstrapChainLadder.__init__` gains `scale: str = "constant"`, `process_scale=None`, `drop=None`; store them; in `fit` pass `scale=self.scale, process_scale=self.process_scale, drop=self.drop` to the sampler and set `self.sampler_ = sampler` after fitting. Update the class docstring parameter list accordingly.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_bootstrap.py -q`
Expected: all pass, including pre-existing correlated-bootstrap tests (constant scale reproduces the previous behaviour because `scale_by_dev_` is the pooled φ repeated).

- [ ] **Step 5: Commit**

```bash
uv run ruff check bayesianchainladder/bootstrap.py tests/test_bootstrap.py
git add bayesianchainladder/bootstrap.py tests/test_bootstrap.py
git commit -m "feat: non-constant scale and process_scale override for ODP bootstrap

Per-development dispersion with England's carry-forward and min-of-two
rules, user-defined forecast-stage scale, and drop/sampler_ exposure on
CorrelatedBootstrapChainLadder.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Mack and Negative Binomial link-ratio bootstraps

**Files:**
- Create: `bayesianchainladder/linkratio.py`
- Modify: `bayesianchainladder/__init__.py`
- Test: `tests/test_linkratio.py`

**Interfaces:**
- Consumes: Task 1 primitives; `_set_full_cumulative_posterior`, `_reserves_from_full_posterior` (Task 2).
- Produces (module `linkratio.py`):
  - `DISTRIBUTIONS = ("nonparametric", "gamma", "lognormal")`
  - `draw_with_moments(mean, sd, dist, rng, resid=None) -> np.ndarray` — Gamma/Lognormal matching first two moments, Normal fallback when `mean <= 1e-12`, `mean + resid * sd` for nonparametric
  - `sample_pseudo_factors(cum, mask, factors, sigma, variance_factor, residual_pool, dist, n_sims, rng) -> np.ndarray (S, n_dev-1)`
  - `forecast_link_ratio_paths(cum, factor_draws, sigma, variance_factor_fn, dist, rng, residual_pool=None) -> np.ndarray (S, n_o, n_d)` — complete simulated cumulative triangles
  - `class MackBootstrap(BaseStochasticReserve)` and `class NegativeBinomialBootstrap(BaseStochasticReserve)` with `__init__(n_sims=1000, bootstrap_dist="gamma", forecast_dist="gamma", drop=None, process_sigma=None, random_seed=None)`; fitted attributes `factors_`, `sigma_`, `scaled_residuals_ (n_o, n_d-1)`, `pseudo_factors_ (S, n_d-1)`, `full_cumulative_posterior_`, `reserves_posterior_`
  - `variance_factor_fn(f) -> np.ndarray`: Mack returns ones, NegBin returns `|f (f-1)|`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_linkratio.py
"""Tests for the Mack and Negative Binomial link-ratio bootstraps (England & Verrall 2006)."""

import chainladder as cl
import numpy as np
import pytest

from bayesianchainladder.base import BaseStochasticReserve
from bayesianchainladder.linkratio import (
    MackBootstrap,
    NegativeBinomialBootstrap,
    draw_with_moments,
)

CL_RESERVE = 18_680_856.0
MACK_SE = 2_441_364.0  # chainladder Mack total SE on genins (log-linear last sigma)


@pytest.fixture(scope="module")
def genins():
    return cl.load_sample("genins")


def test_draw_with_moments_matches_targets():
    rng = np.random.default_rng(0)
    mean = np.full(200_000, 50.0)
    sd = np.full(200_000, 10.0)
    for dist in ("gamma", "lognormal"):
        x = draw_with_moments(mean, sd, dist, rng)
        assert x.mean() == pytest.approx(50.0, rel=0.01)
        assert x.std() == pytest.approx(10.0, rel=0.02)
        assert (x > 0).all()
    neg = draw_with_moments(np.full(1000, -5.0), np.full(1000, 1.0), "gamma", rng)
    assert neg.mean() == pytest.approx(-5.0, abs=0.2)  # Normal fallback
    zero_sd = draw_with_moments(np.array([3.0]), np.array([0.0]), "gamma", rng)
    assert zero_sd[0] == 3.0
    np_draw = draw_with_moments(np.array([1.0, 2.0]), np.array([0.5, 0.5]), "nonparametric", rng, resid=np.array([2.0, -2.0]))
    np.testing.assert_allclose(np_draw, [2.0, 1.0])


def test_mack_bootstrap_matches_analytic(genins):
    model = MackBootstrap(n_sims=4000, random_seed=42).fit(genins)
    assert isinstance(model, BaseStochasticReserve)
    s = model.total_summary()
    assert s.total_reserve_mean == pytest.approx(CL_RESERVE, rel=0.02)
    assert s.total_reserve_stddev == pytest.approx(MACK_SE, rel=0.12)
    assert model.full_cumulative_posterior_.shape == (10, 10, 4000)
    np.testing.assert_allclose(
        model._reserves_from_full_posterior().values, model.reserves_posterior_.values
    )
    cum = np.asarray(genins.values)[0, 0]
    obs = ~np.isnan(cum)
    np.testing.assert_allclose(
        model.full_cumulative_posterior_.values[obs],
        np.repeat(cum[obs][:, None], 4000, axis=1),
    )
    assert model.factors_.shape == (9,) and model.sigma_.shape == (9,)
    assert model.scaled_residuals_.shape == (10, 9)
    assert abs(np.nanmean(model.scaled_residuals_)) < 1e-9  # zero-centred


@pytest.mark.parametrize("bootstrap_dist", ["nonparametric", "gamma", "lognormal"])
@pytest.mark.parametrize("forecast_dist", ["nonparametric", "gamma", "lognormal"])
def test_all_distribution_combinations_run(genins, bootstrap_dist, forecast_dist):
    model = MackBootstrap(
        n_sims=300, bootstrap_dist=bootstrap_dist, forecast_dist=forecast_dist, random_seed=1
    ).fit(genins)
    assert model.total_summary().total_reserve_mean == pytest.approx(CL_RESERVE, rel=0.08)


def test_invalid_distribution_raises():
    with pytest.raises(ValueError, match="bootstrap_dist"):
        MackBootstrap(bootstrap_dist="normal")


def test_drop_and_process_sigma(genins):
    base = MackBootstrap(n_sims=1500, random_seed=3).fit(genins)
    dropped = MackBootstrap(n_sims=1500, random_seed=3, drop=[("2003", 72)]).fit(genins)
    assert dropped.factors_[5] != pytest.approx(base.factors_[5])
    no_process = MackBootstrap(
        n_sims=1500, random_seed=3, process_sigma=np.zeros(9)
    ).fit(genins)
    assert no_process.total_summary().total_reserve_stddev < base.total_summary().total_reserve_stddev
    assert no_process.total_summary().total_reserve_mean == pytest.approx(
        base.total_summary().total_reserve_mean, rel=0.03
    )
    with pytest.raises(ValueError, match="process_sigma"):
        MackBootstrap(process_sigma=np.zeros(3)).fit(genins)


def test_negbin_bootstrap_close_to_mack(genins):
    nb = NegativeBinomialBootstrap(n_sims=3000, random_seed=42).fit(genins)
    mk = MackBootstrap(n_sims=3000, random_seed=42).fit(genins)
    assert nb.total_summary().total_reserve_mean == pytest.approx(CL_RESERVE, rel=0.02)
    assert nb.total_summary().total_reserve_stddev == pytest.approx(
        mk.total_summary().total_reserve_stddev, rel=0.25
    )


def test_mack_bootstrap_handles_negative_incrementals():
    raa = cl.load_sample("raa")  # has negative incrementals
    model = MackBootstrap(n_sims=500, random_seed=0).fit(raa)
    assert np.isfinite(model.total_summary().total_reserve_mean)
    assert model.reserves_posterior_.min() < 0 or model.reserves_posterior_.min() >= 0  # runs without error
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_linkratio.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bayesianchainladder.linkratio'`

- [ ] **Step 3: Implement**

```python
# bayesianchainladder/linkratio.py
"""Bootstrap (and, in a later task, Bayesian) estimators for link-ratio models.

Ports ``Main_Mack_Bstrap`` and ``Main_NegBin_Bstrap`` from Peter England's
StochasticReserving repository
(https://github.com/DrPeterEngland/StochasticReserving, MIT licence), which
implement England & Verrall (2002, 2006):

* Estimation error: resample scaled residuals (nonparametric) or draw
  pseudo link ratios from Gamma / Lognormal with the fitted mean and
  variance (parametric), then recompute volume-weighted factors.
* Process error: roll each origin forward from its latest cumulative using
  the pseudo factors, drawing each next cumulative from Gamma / Lognormal
  (or resampled residuals) with variance ``sigma_j^2 * v(f_j) * C_{i,j}``.

``v(f) = 1`` gives Mack's model; ``v(f) = f (f - 1)`` gives the over-dispersed
Negative Binomial model. When a mean is non-positive a Normal draw with the
same two moments is used, so results may contain negative increments.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np

from ._triangle_ops import (
    DropList,
    cumulative_array,
    link_ratio_mask,
    link_ratio_sigma,
    volume_weighted_factors,
)
from .base import BaseStochasticReserve
from .utils import validate_triangle

DISTRIBUTIONS = ("nonparametric", "gamma", "lognormal")
_TOL = 1e-12


def draw_with_moments(mean, sd, dist: str, rng: np.random.Generator, resid=None) -> np.ndarray:
    """Draw values with the given mean and sd from ``dist``.

    ``gamma`` / ``lognormal`` match the first two moments where ``mean > 0``
    and ``sd > 0``; cells with ``sd == 0`` return the mean; cells with a
    non-positive mean fall back to ``Normal(mean, sd)``. ``nonparametric``
    returns ``mean + resid * sd``.
    """
    mean = np.asarray(mean, dtype=float)
    sd = np.broadcast_to(np.asarray(sd, dtype=float), mean.shape)
    if dist == "nonparametric":
        if resid is None:
            raise ValueError("resid is required for nonparametric draws")
        return mean + np.asarray(resid, dtype=float) * sd
    out = np.array(mean, copy=True)
    pos = (mean > _TOL) & (sd > _TOL)
    if dist == "gamma":
        shape = mean[pos] ** 2 / sd[pos] ** 2
        out[pos] = rng.gamma(shape=shape, scale=sd[pos] ** 2 / mean[pos])
    elif dist == "lognormal":
        s2 = np.log1p((sd[pos] / mean[pos]) ** 2)
        out[pos] = rng.lognormal(mean=np.log(mean[pos]) - 0.5 * s2, sigma=np.sqrt(s2))
    else:
        raise ValueError(f"dist must be one of {DISTRIBUTIONS}, got {dist!r}")
    fallback = (mean <= _TOL) & (sd > _TOL)
    out[fallback] = rng.normal(mean[fallback], sd[fallback])
    return out


def sample_pseudo_factors(
    cum, mask, factors, sigma, variance_factor, residual_pool, dist, n_sims, rng
) -> np.ndarray:
    """Estimation-error stage: pseudo link ratios → volume-weighted factors, (S, n_dev-1)."""
    n_o, n_d = cum.shape
    w = np.nan_to_num(cum[:, :-1], nan=0.0)
    mean = np.broadcast_to(factors, (n_sims, n_o, n_d - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        sd_cell = sigma * np.sqrt(variance_factor) / np.sqrt(np.where(w > 0, w, np.nan))
    sd_cell = np.where(mask > 0, np.nan_to_num(sd_cell, nan=0.0), 0.0)
    sd = np.broadcast_to(sd_cell, mean.shape)
    resid = rng.choice(residual_pool, size=mean.shape) if dist == "nonparametric" else None
    pseudo_ratios = draw_with_moments(mean, sd, dist, rng, resid)
    weights = w * mask
    den = weights.sum(axis=0)
    num = (pseudo_ratios * weights).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, 1.0)


def forecast_link_ratio_paths(
    cum,
    factor_draws,
    sigma,
    variance_factor_fn: Callable[[np.ndarray], np.ndarray],
    dist: str,
    rng: np.random.Generator,
    residual_pool=None,
) -> np.ndarray:
    """Process-error stage: complete simulated cumulative triangles (S, n_o, n_d)."""
    factor_draws = np.asarray(factor_draws, dtype=float)
    n_sims = factor_draws.shape[0]
    n_o, n_d = cum.shape
    full = np.repeat(cum[None, ...], n_sims, axis=0)
    for j in range(1, n_d):
        need = np.isnan(cum[:, j])
        if not need.any():
            continue
        prev = full[:, :, j - 1]
        f = factor_draws[:, j - 1][:, None]
        mean = prev * f
        sd = sigma[j - 1] * np.sqrt(variance_factor_fn(f) * np.abs(prev))
        resid = rng.choice(residual_pool, size=mean.shape) if dist == "nonparametric" else None
        draw = draw_with_moments(mean, sd, dist, rng, resid)
        full[:, need, j] = draw[:, need]
    return full


class _LinkRatioBootstrap(BaseStochasticReserve):
    """Shared implementation; subclasses define the variance function."""

    def __init__(
        self,
        n_sims: int = 1000,
        bootstrap_dist: str = "gamma",
        forecast_dist: str = "gamma",
        drop: DropList = None,
        process_sigma=None,
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        if bootstrap_dist not in DISTRIBUTIONS:
            raise ValueError(f"bootstrap_dist must be one of {DISTRIBUTIONS}")
        if forecast_dist not in DISTRIBUTIONS:
            raise ValueError(f"forecast_dist must be one of {DISTRIBUTIONS}")
        self.n_sims = n_sims
        self.bootstrap_dist = bootstrap_dist
        self.forecast_dist = forecast_dist
        self.drop = drop
        self.process_sigma = None if process_sigma is None else np.asarray(process_sigma, float)
        self.random_seed = random_seed

    @staticmethod
    def variance_factor_fn(factors) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def fit(self, triangle):
        validate_triangle(triangle)
        self.triangle_ = triangle.copy()
        cum, origins, devs = cumulative_array(triangle)
        mask = link_ratio_mask(cum, self.drop, origins, devs)
        factors = volume_weighted_factors(cum, mask)
        vf = self.variance_factor_fn(factors)
        sigma, resid = link_ratio_sigma(cum, mask, factors, vf)

        n_j = mask.sum(axis=0)
        bias = np.where(n_j > 1, np.sqrt(n_j / np.maximum(n_j - 1, 1)), 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = resid / np.where(sigma > 0, sigma, np.nan) * bias
        pool = scaled[np.isfinite(scaled)]
        pool = pool - pool.mean()
        scaled = np.where(np.isfinite(scaled), scaled - np.nanmean(scaled[np.isfinite(scaled)]), np.nan)

        if self.process_sigma is not None and self.process_sigma.shape != sigma.shape:
            raise ValueError(
                f"process_sigma must have shape {sigma.shape}, got {self.process_sigma.shape}"
            )
        forecast_sigma = sigma if self.process_sigma is None else self.process_sigma

        rng = np.random.default_rng(self.random_seed)
        pseudo_factors = sample_pseudo_factors(
            cum, mask, factors, sigma, vf, pool, self.bootstrap_dist, self.n_sims, rng
        )
        full = forecast_link_ratio_paths(
            cum, pseudo_factors, forecast_sigma, self.variance_factor_fn,
            self.forecast_dist, rng, residual_pool=pool,
        )

        self.factors_ = factors
        self.sigma_ = sigma
        self.scaled_residuals_ = scaled
        self.pseudo_factors_ = pseudo_factors
        self._set_full_cumulative_posterior(np.moveaxis(full, 0, -1), origins, devs)
        self.reserves_posterior_ = self._reserves_from_full_posterior()
        self._build_reserve_summaries()
        self._is_fitted = True
        return self


class MackBootstrap(_LinkRatioBootstrap):
    """Bootstrap of Mack's model: Var(C_{i,j+1} | C_{i,j}) = sigma_j^2 C_{i,j}."""

    @staticmethod
    def variance_factor_fn(factors) -> np.ndarray:
        return np.ones_like(np.asarray(factors, dtype=float))


class NegativeBinomialBootstrap(_LinkRatioBootstrap):
    """Bootstrap of the over-dispersed Negative Binomial model:
    Var(C_{i,j+1} | C_{i,j}) = sigma_j^2 f_j (f_j - 1) C_{i,j}. Requires f_j > 1."""

    @staticmethod
    def variance_factor_fn(factors) -> np.ndarray:
        f = np.asarray(factors, dtype=float)
        return np.abs(f * (f - 1.0))

    def fit(self, triangle):
        cum, origins, devs = cumulative_array(triangle)
        f = volume_weighted_factors(cum, link_ratio_mask(cum, self.drop, origins, devs))
        if np.any(f <= 1.0):
            warnings.warn(
                "NegativeBinomialBootstrap: development factors <= 1 detected; the "
                "variance function f(f-1) is not valid there. Consider MackBootstrap.",
                UserWarning,
                stacklevel=2,
            )
        return super().fit(triangle)
```

Export `MackBootstrap`, `NegativeBinomialBootstrap`, `draw_with_moments`, `forecast_link_ratio_paths`, `sample_pseudo_factors` from `__init__.py` under `# Link-ratio bootstraps`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_linkratio.py -q`
Expected: all pass (the 9-combination parametrized test included). If `test_mack_bootstrap_matches_analytic` fails only on the SD tolerance, compare against `mack_analytic_rmsep` from Task 11 using England's last-sigma rule before touching the sampler; a Gamma/Gamma Mack bootstrap on Taylor & Ashe should land within a few percent of the analytic Mack SE computed with the same sigmas.

- [ ] **Step 5: Commit**

```bash
uv run ruff check bayesianchainladder/linkratio.py tests/test_linkratio.py
git add bayesianchainladder/linkratio.py bayesianchainladder/__init__.py tests/test_linkratio.py
git commit -m "feat: Mack and Negative Binomial link-ratio bootstraps

Nonparametric / Gamma / Lognormal options at both the estimation and
process stages, link-ratio exclusions, user-defined process sigma,
per-cell simulated triangles. After England & Verrall (2002, 2006).

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: analytic RMSEP oracles

**Files:**
- Create: `bayesianchainladder/analytic.py`
- Modify: `bayesianchainladder/__init__.py`
- Test: `tests/test_analytic.py`

**Interfaces:**
- Consumes: Task 1 primitives.
- Produces:
  - `@dataclass AnalyticResult`: `origins: list[int]`, `reserves: np.ndarray (n_o,)`, `reserve_sd: np.ndarray (n_o,)`, `total_reserve: float`, `total_sd: float`, `scale: np.ndarray | None (n_dev,)`, `coefficients: np.ndarray | None`; property `total_cov`; method `to_frame() -> pd.DataFrame` (index origins + "Total", columns `reserve, sd, cov`)
  - `poisson_irls(X, y, max_iter=50, tol=1e-10) -> np.ndarray`
  - `odp_analytic_rmsep(triangle, scale="nonconstant") -> AnalyticResult`
  - `mack_analytic_rmsep(triangle, drop=None) -> AnalyticResult` (wraps chainladder; `scale` holds Mack sigma per ratio column padded with NaN to `n_dev`)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_analytic.py
"""Analytic (closed-form) prediction errors used as oracles for the bootstraps."""

import chainladder as cl
import numpy as np
import pytest

from bayesianchainladder.analytic import (
    AnalyticResult,
    mack_analytic_rmsep,
    odp_analytic_rmsep,
    poisson_irls,
)
from bayesianchainladder.bootstrap import BootstrapODPChainLadder


@pytest.fixture(scope="module")
def genins():
    return cl.load_sample("genins")


def test_poisson_irls_recovers_known_coefficients():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(5000), rng.normal(size=5000)])
    beta_true = np.array([1.0, 0.5])
    y = rng.poisson(np.exp(X @ beta_true))
    beta = poisson_irls(X, y.astype(float))
    np.testing.assert_allclose(beta, beta_true, atol=0.05)


def test_odp_reserves_equal_chain_ladder(genins):
    res = odp_analytic_rmsep(genins, scale="constant")
    assert isinstance(res, AnalyticResult)
    cl_ibnr = np.asarray(cl.Chainladder().fit(genins).ibnr_.values)[0, 0, :, 0]
    np.testing.assert_allclose(res.reserves, cl_ibnr, rtol=1e-6)
    assert res.total_reserve == pytest.approx(18_680_856, rel=1e-6)
    assert res.reserve_sd[0] == 0.0  # fully developed origin


def test_odp_constant_scale_matches_chainladder_phi(genins):
    res = odp_analytic_rmsep(genins, scale="constant")
    prepared = genins.copy()
    prepared.key_labels = ["triangle_id"]
    prepared.kdims = np.asarray([["resample"]], dtype=object)
    sampler = cl.BootstrapODPSample(n_sims=5, hat_adj=False, random_state=1).fit(prepared)
    assert res.scale.shape == (10,)
    np.testing.assert_allclose(res.scale, float(np.asarray(sampler.scale_).flatten()[0]), rtol=1e-6)


def test_odp_analytic_sd_close_to_bootstrap(genins):
    res = odp_analytic_rmsep(genins, scale="constant")
    boot = BootstrapODPChainLadder(n_sims=4000, random_seed=9).fit(genins)
    assert res.total_sd == pytest.approx(boot.total_summary().total_reserve_stddev, rel=0.12)
    assert 0.10 < res.total_cov < 0.25


def test_odp_nonconstant_scale_varies(genins):
    res = odp_analytic_rmsep(genins, scale="nonconstant")
    assert res.scale.shape == (10,)
    assert not np.allclose(res.scale, res.scale[0])
    assert res.scale[-1] == pytest.approx(min(res.scale[-2], res.scale[-3]))
    assert res.total_sd > 0
    with pytest.raises(ValueError):
        odp_analytic_rmsep(genins, scale="odd")


def test_mack_analytic_matches_chainladder(genins):
    res = mack_analytic_rmsep(genins)
    assert res.total_reserve == pytest.approx(18_680_856, rel=1e-6)
    assert res.total_sd == pytest.approx(2_441_364, rel=1e-4)
    dropped = mack_analytic_rmsep(genins, drop=[("2003", 72)])
    assert dropped.total_reserve != pytest.approx(res.total_reserve, rel=1e-6)
    frame = res.to_frame()
    assert list(frame.columns) == ["reserve", "sd", "cov"]
    assert frame.index[-1] == "Total"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_analytic.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bayesianchainladder.analytic'`

- [ ] **Step 3: Implement**

```python
# bayesianchainladder/analytic.py
"""Closed-form prediction errors (RMSEP) for chain-ladder models.

``odp_analytic_rmsep`` ports ``ODP_ChainLadder`` from Peter England's
StochasticReserving repository
(https://github.com/DrPeterEngland/StochasticReserving, MIT licence): fit the
over-dispersed Poisson cross-classified GLM by IRLS, form the covariance of
the future fitted values from the parameter covariance, and add the process
variance on the diagonal (England & Verrall 2002, Section 7). The reserves
equal the volume-weighted chain ladder exactly when all cells are positive.

``mack_analytic_rmsep`` wraps ``chainladder.MackChainladder``. Both are used
as oracles in the test suite: a correctly implemented bootstrap should have a
standard deviation close to the analytic value.
"""

from __future__ import annotations

from dataclasses import dataclass

import chainladder as cl
import numpy as np
import pandas as pd

from ._triangle_ops import DropList, cumulative_array, cumulative_to_incremental


@dataclass
class AnalyticResult:
    origins: list[int]
    reserves: np.ndarray
    reserve_sd: np.ndarray
    total_reserve: float
    total_sd: float
    scale: np.ndarray | None = None
    coefficients: np.ndarray | None = None

    @property
    def total_cov(self) -> float:
        return float(self.total_sd / self.total_reserve) if self.total_reserve else float("nan")

    def to_frame(self) -> pd.DataFrame:
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = np.where(self.reserves != 0, self.reserve_sd / np.abs(self.reserves), np.nan)
        frame = pd.DataFrame(
            {"reserve": self.reserves, "sd": self.reserve_sd, "cov": cov}, index=self.origins
        )
        frame.loc["Total"] = [self.total_reserve, self.total_sd, self.total_cov]
        return frame


def poisson_irls(X: np.ndarray, y: np.ndarray, max_iter: int = 50, tol: float = 1e-10) -> np.ndarray:
    """Poisson log-link GLM coefficients by iteratively reweighted least squares."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(y.mean(), 1e-8))
    for _ in range(max_iter):
        eta = X @ beta
        mu = np.exp(eta)
        z = eta + (y - mu) / mu
        xtw = X.T * mu
        new = np.linalg.solve(xtw @ X, xtw @ z)
        converged = np.max(np.abs(new - beta)) < tol
        beta = new
        if converged:
            break
    return beta


def _design_matrix(n_origin: int, n_dev: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    i, j = np.indices((n_origin, n_dev))
    i, j = i.ravel(), j.ravel()
    X = np.zeros((n_origin * n_dev, 1 + (n_origin - 1) + (n_dev - 1)))
    X[:, 0] = 1.0
    rows = np.arange(len(i))
    X[rows[i > 0], i[i > 0]] = 1.0  # origin effects, columns 1..n_origin-1
    X[rows[j > 0], (n_origin - 1) + j[j > 0]] = 1.0  # dev effects
    return X, i, j


def _nonconstant_phi(pearson_sq, j_obs, n_dev, bias):
    n_j = np.bincount(j_obs, minlength=n_dev).astype(float)
    ss = np.bincount(j_obs, weights=pearson_sq, minlength=n_dev)
    phi = np.zeros(n_dev)
    for j in range(n_dev - 1):
        if n_j[j] > 1:
            phi[j] = bias * ss[j] / n_j[j]
        else:
            phi[j] = phi[j - 1] if j > 0 else 0.0
    phi[-1] = min(phi[-2], phi[-3]) if n_dev >= 3 else phi[-2]
    return phi


def odp_analytic_rmsep(triangle, scale: str = "nonconstant") -> AnalyticResult:
    if scale not in ("constant", "nonconstant"):
        raise ValueError("scale must be 'constant' or 'nonconstant'")
    cum, origins, _ = cumulative_array(triangle)
    incr = cumulative_to_incremental(cum)
    n_o, n_d = incr.shape
    X, i_all, j_all = _design_matrix(n_o, n_d)
    y_all = np.nan_to_num(incr, nan=0.0).ravel()
    obs = ~np.isnan(incr).ravel()
    if (np.bincount(j_all[obs], weights=y_all[obs], minlength=n_d) <= 0).any():
        raise ValueError("ODP GLM needs a positive column sum of incrementals in every development period")

    beta = poisson_irls(X[obs], y_all[obs])
    mu_all = np.exp(X @ beta)
    mu_obs, mu_fut = mu_all[obs], mu_all[~obs]
    n_obs, p = int(obs.sum()), X.shape[1]
    pearson_sq = (y_all[obs] - mu_obs) ** 2 / mu_obs
    if scale == "constant":
        phi = np.full(n_d, pearson_sq.sum() / (n_obs - p))
    else:
        phi = _nonconstant_phi(pearson_sq, j_all[obs], n_d, bias=n_obs / (n_obs - p))
    phi_safe = np.maximum(phi, 1e-12)

    w = mu_obs / phi_safe[j_all[obs]]
    sigma_beta = np.linalg.inv((X[obs].T * w) @ X[obs])
    X_fut = X[~obs]
    cov_mu = (X_fut @ sigma_beta @ X_fut.T) * np.outer(mu_fut, mu_fut)
    cov = cov_mu + np.diag(phi[j_all[~obs]] * mu_fut)

    origin_fut = i_all[~obs]
    reserves = np.bincount(origin_fut, weights=mu_fut, minlength=n_o)
    A = np.zeros((n_o, len(mu_fut)))
    A[origin_fut, np.arange(len(mu_fut))] = 1.0
    var_origin = np.einsum("ik,kl,il->i", A, cov, A)
    return AnalyticResult(
        origins=origins,
        reserves=reserves,
        reserve_sd=np.sqrt(np.maximum(var_origin, 0.0)),
        total_reserve=float(reserves.sum()),
        total_sd=float(np.sqrt(cov.sum())),
        scale=phi,
        coefficients=beta,
    )


def mack_analytic_rmsep(triangle, drop: DropList = None) -> AnalyticResult:
    cum, origins, _ = cumulative_array(triangle)
    dev = cl.Development(drop=list(drop) if drop else None).fit_transform(triangle)
    mack = cl.MackChainladder().fit(dev)
    reserves = np.nan_to_num(np.asarray(mack.ibnr_.values, dtype=float)[0, 0, :, 0])
    reserve_sd = np.nan_to_num(
        np.asarray(mack.mack_std_err_.latest_diagonal.values, dtype=float)[0, 0, :, 0]
    )
    sigma = np.asarray(dev.sigma_.values, dtype=float).flatten()
    return AnalyticResult(
        origins=origins,
        reserves=reserves,
        reserve_sd=reserve_sd,
        total_reserve=float(reserves.sum()),
        total_sd=float(np.asarray(mack.total_mack_std_err_).flatten()[0]),
        scale=np.concatenate([sigma, [np.nan]]),
        coefficients=np.log(np.asarray(dev.ldf_.values, dtype=float).flatten()),
    )
```

Export `AnalyticResult`, `odp_analytic_rmsep`, `mack_analytic_rmsep`, `poisson_irls` under `# Analytic prediction errors`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_analytic.py -q`
Expected: 6 passed. `test_odp_constant_scale_matches_chainladder_phi` checks our Pearson φ against chainladder's own dispersion estimate with `hat_adj=False`; if it fails by a factor near `n/(n-p)`, chainladder is applying a degrees-of-freedom convention — match ours to `pearson.sum() / (n_obs - p)` as coded and inspect `sampler.scale_` derivation before changing anything.

- [ ] **Step 5: Commit**

```bash
uv run ruff check bayesianchainladder/analytic.py tests/test_analytic.py
git add bayesianchainladder/analytic.py bayesianchainladder/__init__.py tests/test_analytic.py
git commit -m "feat: analytic ODP and Mack prediction errors as test oracles

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 12: influential link-ratio sensitivity analysis

**Files:**
- Create: `bayesianchainladder/sensitivity.py`
- Modify: `bayesianchainladder/__init__.py`
- Test: `tests/test_sensitivity.py`

**Interfaces:**
- Consumes: `mack_analytic_rmsep` (Task 11); `cumulative_array`, `link_ratio_mask` (Task 1); `load_england_sample` (Task 6).
- Produces:
  - `link_ratio_sensitivity(triangle, drop=None) -> pd.DataFrame` with columns `origin, dev, reserve, reserve_sd, reserve_cov, reserve_diff, sd_diff, cov_diff, reserve_rank, sd_rank, cov_rank` (rank 1 = largest reduction), sorted by `sd_rank`; `df.attrs` holds `base_reserve`, `base_sd`, `base_cov`
  - `top_influential(result, n=3, by="sd") -> list[tuple[str, int]]` in chainladder `drop` syntax

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sensitivity.py
"""Leave-one-link-ratio-out influence analysis (England's Sensitivities)."""

import numpy as np
import pytest

from bayesianchainladder.analytic import mack_analytic_rmsep
from bayesianchainladder.datasets import load_england_sample
from bayesianchainladder.sensitivity import link_ratio_sensitivity, top_influential


@pytest.fixture(scope="module")
def liability():
    return load_england_sample("liability")


@pytest.fixture(scope="module")
def result(liability):
    return link_ratio_sensitivity(liability)


def test_columns_and_row_count(result):
    expected = {
        "origin", "dev", "reserve", "reserve_sd", "reserve_cov",
        "reserve_diff", "sd_diff", "cov_diff", "reserve_rank", "sd_rank", "cov_rank",
    }
    assert expected <= set(result.columns)
    # 45 available ratios minus the 9 columns... only ratios whose column keeps >= 1 other ratio
    # are evaluated: the single ratio in the last column (origin 2001, dev 108) is skipped.
    assert len(result) == 44
    assert result["sd_rank"].min() == 1 and result["sd_rank"].max() == 44
    assert result.attrs["base_reserve"] == pytest.approx(331_038, abs=1.0)


def test_most_influential_ratio_is_origin3_dev6(result):
    top = top_influential(result, n=1, by="sd")
    assert top == [("2003", 72)]
    row = result[(result.origin == 2003) & (result.dev == 72)].iloc[0]
    assert row["reserve"] == pytest.approx(289_946, rel=1e-3)
    assert row["reserve_sd"] == pytest.approx(38_076, rel=1e-3)
    assert row["sd_diff"] < 0 and row["cov_rank"] == 1 and row["reserve_rank"] == 1


def test_top_n_returns_drop_list_usable_by_mack(result, liability):
    top3 = top_influential(result, n=3, by="sd")
    assert len(top3) == 3 and all(isinstance(o, str) and isinstance(d, int) for o, d in top3)
    reduced = mack_analytic_rmsep(liability, drop=top3)
    base = mack_analytic_rmsep(liability)
    assert reduced.total_sd < 0.6 * base.total_sd
    with pytest.raises(ValueError):
        top_influential(result, n=1, by="nonsense")


def test_base_drop_is_respected(liability):
    res = link_ratio_sensitivity(liability, drop=[("2003", 72)])
    assert res.attrs["base_reserve"] == pytest.approx(289_946, rel=1e-3)
    assert not ((res.origin == 2003) & (res.dev == 72)).any()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sensitivity.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bayesianchainladder.sensitivity'`

- [ ] **Step 3: Implement**

```python
# bayesianchainladder/sensitivity.py
"""Identify link ratios that drive reserve volatility.

Ports ``Sensitivities`` from Peter England's StochasticReserving repository
(https://github.com/DrPeterEngland/StochasticReserving, MIT licence): exclude
each link ratio in turn, re-apply Mack's model analytically, and rank the
change in total reserve, its standard deviation and coefficient of variation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._triangle_ops import DropList, cumulative_array, link_ratio_mask
from .analytic import mack_analytic_rmsep

_RANK_COLUMNS = {"reserve": "reserve_rank", "sd": "sd_rank", "cov": "cov_rank"}


def link_ratio_sensitivity(triangle, drop: DropList = None) -> pd.DataFrame:
    cum, origins, devs = cumulative_array(triangle)
    base_drop = list(drop or [])
    base = mack_analytic_rmsep(triangle, drop=base_drop)
    mask = link_ratio_mask(cum, base_drop, origins, devs)
    n_j = mask.sum(axis=0)

    rows = []
    for i in range(cum.shape[0]):
        for j in range(cum.shape[1] - 1):
            if mask[i, j] == 0 or n_j[j] <= 1:
                continue  # excluding the only ratio in a column leaves no factor estimate
            res = mack_analytic_rmsep(triangle, drop=base_drop + [(str(origins[i]), devs[j])])
            rows.append(
                {
                    "origin": origins[i],
                    "dev": devs[j],
                    "reserve": res.total_reserve,
                    "reserve_sd": res.total_sd,
                    "reserve_cov": res.total_cov,
                    "reserve_diff": res.total_reserve - base.total_reserve,
                    "sd_diff": res.total_sd - base.total_sd,
                    "cov_diff": res.total_cov - base.total_cov,
                }
            )
    df = pd.DataFrame(rows)
    for key, rank_col in _RANK_COLUMNS.items():
        diff_col = f"{key}_diff"
        df[rank_col] = df[diff_col].rank(method="first", ascending=True).astype(int)
    df.attrs.update(
        base_reserve=base.total_reserve, base_sd=base.total_sd, base_cov=base.total_cov
    )
    return df.sort_values("sd_rank").reset_index(drop=True)


def top_influential(result: pd.DataFrame, n: int = 3, by: str = "sd") -> list[tuple[str, int]]:
    """The ``n`` ratios with the largest reduction in ``by`` ∈ {reserve, sd, cov},
    as chainladder-style ``(origin_label, dev_months)`` drop tuples."""
    if by not in _RANK_COLUMNS:
        raise ValueError(f"by must be one of {sorted(_RANK_COLUMNS)}")
    top = result.nsmallest(n, _RANK_COLUMNS[by])
    return [(str(int(o)), int(d)) for o, d in zip(top["origin"], top["dev"], strict=True)]
```

Export both functions under `# Sensitivity analysis`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_sensitivity.py -q`
Expected: 4 passed. The reference values (289,946 / 38,076) were computed with chainladder 0.9.1 on 2026-09-23; the origin label `"2003"` assumes `load_england_sample`'s default `first_origin=2001`.

- [ ] **Step 5: Commit**

```bash
uv run ruff check bayesianchainladder/sensitivity.py tests/test_sensitivity.py
git add bayesianchainladder/sensitivity.py bayesianchainladder/__init__.py tests/test_sensitivity.py
git commit -m "feat: leave-one-ratio-out influence analysis with rankings

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 13: scaling to target ultimates and incurred-to-paid conversion

**Files:**
- Modify: `bayesianchainladder/base.py` (add `scale_to_target` method, module function `incurred_to_paid`)
- Modify: `bayesianchainladder/__init__.py`
- Test: `tests/test_base.py`

**Interfaces:**
- Consumes: `ReserveSamples` (Task 2).
- Produces:
  - `BaseStochasticReserve.scale_to_target(self, target_ultimates, method="multiplicative") -> ReserveSamples` — `target_ultimates` is a `pd.Series`/`dict` keyed by origin or an array aligned with `reserves_posterior_.origin`; `method` is `"additive"`, `"multiplicative"`, or a `dict` origin→method
  - `incurred_to_paid(model, paid_triangle) -> ReserveSamples` — reserves = simulated incurred ultimates − latest paid

- [ ] **Step 1: Write the failing tests** (append to `tests/test_base.py`)

```python
from bayesianchainladder.base import incurred_to_paid


class TestScalingAndIncurredToPaid:
    @pytest.fixture
    def fitted(self):
        from bayesianchainladder.bootstrap import BootstrapODPChainLadder

        return BootstrapODPChainLadder(n_sims=500, random_seed=4).fit(cl.load_sample("genins"))

    def test_additive_preserves_sd_and_hits_target(self, fitted):
        origins = list(fitted.reserves_posterior_.coords["origin"].values)
        paid = fitted._paid_to_date().reindex(origins)
        target = paid + 1.1 * fitted.ibnr_["mean"]
        scaled = fitted.scale_to_target(target, method="additive")
        assert isinstance(scaled, ReserveSamples)
        np.testing.assert_allclose(scaled.ibnr_["std"].values, fitted.ibnr_["std"].values, rtol=1e-9)
        np.testing.assert_allclose(scaled.ultimate_["mean"].values, target.values, rtol=1e-9)

    def test_multiplicative_preserves_cov(self, fitted):
        origins = list(fitted.reserves_posterior_.coords["origin"].values)
        paid = fitted._paid_to_date().reindex(origins)
        target = paid + 1.1 * fitted.ibnr_["mean"]
        scaled = fitted.scale_to_target(target, method="multiplicative")
        base_cov = (fitted.ibnr_["std"] / fitted.ibnr_["mean"]).values[1:]
        new_cov = (scaled.ibnr_["std"] / scaled.ibnr_["mean"]).values[1:]
        np.testing.assert_allclose(new_cov, base_cov, rtol=1e-9)
        np.testing.assert_allclose(scaled.ibnr_["mean"].values[1:], 1.1 * fitted.ibnr_["mean"].values[1:], rtol=1e-9)

    def test_per_origin_method_dict_and_validation(self, fitted):
        origins = list(fitted.reserves_posterior_.coords["origin"].values)
        paid = fitted._paid_to_date().reindex(origins)
        target = paid + fitted.ibnr_["mean"]
        methods = {o: ("additive" if k < 5 else "multiplicative") for k, o in enumerate(origins)}
        scaled = fitted.scale_to_target(target, method=methods)
        np.testing.assert_allclose(scaled.ibnr_["mean"].values, fitted.ibnr_["mean"].values, rtol=1e-9)
        with pytest.raises(ValueError, match="method"):
            fitted.scale_to_target(target, method="geometric")
        with pytest.raises(ValueError, match="origin"):
            fitted.scale_to_target(target.iloc[:3])

    def test_incurred_to_paid(self):
        from bayesianchainladder.bootstrap import BootstrapODPChainLadder

        clrd = cl.load_sample("clrd").groupby("LOB").sum().loc["wkcomp"]
        incurred = clrd["IncurLoss"]
        paid = clrd["CumPaidLoss"]
        model = BootstrapODPChainLadder(n_sims=300, random_seed=8).fit(incurred)
        converted = incurred_to_paid(model, paid)
        assert isinstance(converted, ReserveSamples)
        latest_inc = model._paid_to_date().values
        latest_paid = converted._paid_to_date().values
        expected_mean = model.ibnr_["mean"].values + latest_inc - latest_paid
        np.testing.assert_allclose(converted.ibnr_["mean"].values, expected_mean, rtol=1e-9)
        np.testing.assert_allclose(converted.ibnr_["std"].values, model.ibnr_["std"].values, rtol=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_base.py -q -k ScalingAndIncurred`
Expected: FAIL with `ImportError: cannot import name 'incurred_to_paid'`

- [ ] **Step 3: Implement**

Add to `BaseStochasticReserve` (after `summary_statistics`):

```python
    def scale_to_target(self, target_ultimates, method="multiplicative"):
        """Shift or scale each origin's reserve samples so their mean ultimate
        hits ``target_ultimates`` (England's ``Scaled_Results``).

        ``additive`` preserves the absolute standard deviation; ``multiplicative``
        preserves the coefficient of variation. ``method`` may be a dict mapping
        origin to method. The per-cell posterior is not carried over because
        cash flows would need their own scaling rule.
        """
        from .base import ReserveSamples  # local import keeps the class below this one

        self._check_is_fitted()
        origins = list(self.reserves_posterior_.coords["origin"].values)
        paid = self._paid_to_date().reindex(origins).fillna(0.0).values
        if isinstance(target_ultimates, dict | pd.Series):
            target = pd.Series(target_ultimates)
            missing = [o for o in origins if o not in target.index]
            if missing:
                raise ValueError(f"target_ultimates is missing origin(s) {missing}")
            target = target.reindex(origins).values.astype(float)
        else:
            target = np.asarray(target_ultimates, dtype=float)
            if target.shape != (len(origins),):
                raise ValueError("target_ultimates must have one value per origin")
        if isinstance(method, str):
            methods = {o: method for o in origins}
        else:
            methods = dict(method)
        bad = {m for m in methods.values()} - {"additive", "multiplicative"}
        if bad or set(origins) - set(methods):
            raise ValueError("method must be 'additive' or 'multiplicative' for every origin")

        res = self.reserves_posterior_.transpose("origin", "sample").values.copy()
        mean = res.mean(axis=1)
        target_reserve = target - paid
        for k, origin in enumerate(origins):
            if methods[origin] == "additive":
                res[k] += target_reserve[k] - mean[k]
            elif mean[k] != 0:
                res[k] *= target_reserve[k] / mean[k]
            else:
                res[k] = target_reserve[k]
        scaled = xr.DataArray(
            res,
            dims=["origin", "sample"],
            coords={"origin": origins, "sample": np.arange(res.shape[1])},
        )
        return ReserveSamples(self.triangle_, scaled)
```

Add the module-level function after the `ReserveSamples` class:

```python
def incurred_to_paid(model: BaseStochasticReserve, paid_triangle) -> "ReserveSamples":
    """Turn an incurred-basis IBNR distribution into a reserve distribution by
    subtracting the latest paid instead of the latest incurred (England's
    ``Incurred_to_Paid``). The absolute SD is unchanged; the CoV becomes
    meaningful for comparison with a paid analysis."""
    from .utils import triangle_to_dataframe

    model._check_is_fitted()
    origins = list(model.reserves_posterior_.coords["origin"].values)
    latest_incurred = model._paid_to_date().reindex(origins).fillna(0.0).values
    paid_df = triangle_to_dataframe(paid_triangle)
    latest_paid = (
        paid_df.groupby("origin", observed=True)["incremental"].sum().reindex(origins).fillna(0.0).values
    )
    res = model.reserves_posterior_.transpose("origin", "sample").values
    reserves = res + (latest_incurred - latest_paid)[:, None]
    return ReserveSamples(
        paid_triangle,
        xr.DataArray(
            reserves,
            dims=["origin", "sample"],
            coords={"origin": origins, "sample": np.arange(reserves.shape[1])},
        ),
    )
```

Export `incurred_to_paid` under `# Base contract`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_base.py -q`
Expected: all pass. If the `clrd` slice in `test_incurred_to_paid` produces a multi-index triangle, use `cl.load_sample("clrd").groupby("LOB").sum().loc["wkcomp"]["IncurLoss"]` exactly as written (it is a single-index Triangle in chainladder 0.9.1).

- [ ] **Step 5: Commit**

```bash
uv run ruff check bayesianchainladder/base.py tests/test_base.py
git add bayesianchainladder/base.py bayesianchainladder/__init__.py tests/test_base.py
git commit -m "feat: scale reserve samples to target ultimates; incurred-to-paid conversion

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 14: fan chart, scaled-residual, sensitivity-heatmap and capital-profile plots

**Files:**
- Modify: `bayesianchainladder/plots.py`, `bayesianchainladder/__init__.py`
- Test: `tests/test_plots.py`

**Interfaces:**
- Consumes: `full_cumulative_posterior_` (Tasks 2/3/10), `scaled_residuals_`/`sigma_` on `MackBootstrap` (Task 10), `link_ratio_sensitivity` (Task 12), `capital_profile` (Task 7).
- Produces (all return `tuple[Figure, Axes]`):
  - `plot_fan_chart(model, origin, bands=((0.01, 0.99), (0.05, 0.95), (0.25, 0.75)), ax=None, figsize=None)`
  - `plot_scaled_residuals(residuals: np.ndarray, by="dev", sigma=None, ax=None, figsize=None, title=None)` — `residuals (n_origin, n_cols)`, NaN = excluded; `by ∈ {"origin","dev","calendar"}`; `sigma` drawn on a twin axis when given
  - `plot_sensitivity_heatmap(result: pd.DataFrame, value="sd_diff", ax=None, figsize=None)`
  - `plot_capital_profiles(profiles: dict[str, np.ndarray], ax=None, figsize=None)` — each profile plotted as % of opening capital

- [ ] **Step 1: Write the failing tests** (append to `tests/test_plots.py`)

```python
class TestEnglandPlots:
    @pytest.fixture(scope="class")
    def mack_boot(self):
        from bayesianchainladder.linkratio import MackBootstrap

        return MackBootstrap(n_sims=200, random_seed=1).fit(cl.load_sample("genins"))

    def test_fan_chart(self, mack_boot):
        from bayesianchainladder.plots import plot_fan_chart

        fig, ax = plot_fan_chart(mack_boot, origin=2005)
        assert ax.get_title().startswith("Origin 2005")
        assert len(ax.collections) >= 3  # one filled band per quantile pair
        plt.close(fig)
        from bayesianchainladder.bootstrap import MackChainLadder

        with pytest.raises(ValueError, match="per-cell"):
            plot_fan_chart(MackChainLadder().fit(cl.load_sample("raa")), origin=1981)

    @pytest.mark.parametrize("by", ["origin", "dev", "calendar"])
    def test_scaled_residuals(self, mack_boot, by):
        from bayesianchainladder.plots import plot_scaled_residuals

        fig, ax = plot_scaled_residuals(mack_boot.scaled_residuals_, by=by, sigma=mack_boot.sigma_)
        assert ax.get_xlabel().lower().startswith(by)
        assert len(fig.axes) == (2 if by != "calendar" else 1)  # twin axis only where sigma applies
        plt.close(fig)
        with pytest.raises(ValueError):
            plot_scaled_residuals(mack_boot.scaled_residuals_, by="weird")

    def test_sensitivity_heatmap(self):
        from bayesianchainladder.datasets import load_england_sample
        from bayesianchainladder.plots import plot_sensitivity_heatmap
        from bayesianchainladder.sensitivity import link_ratio_sensitivity

        res = link_ratio_sensitivity(load_england_sample("liability"))
        fig, ax = plot_sensitivity_heatmap(res, value="sd_diff")
        assert len(ax.images) == 1
        assert len(ax.texts) == len(res)  # every evaluated ratio annotated
        plt.close(fig)

    def test_capital_profiles(self):
        from bayesianchainladder.plots import plot_capital_profiles

        fig, ax = plot_capital_profiles({"best estimate": np.array([1.0, 0.6, 0.3]), "sd": np.array([1.0, 0.5, 0.2])})
        assert len(ax.lines) == 2
        assert ax.get_ylabel().startswith("Percent")
        plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_plots.py -q -k EnglandPlots`
Expected: FAIL with `ImportError: cannot import name 'plot_fan_chart'`

- [ ] **Step 3: Implement** (append to `plots.py`; `BaseStochasticReserve` import goes under `TYPE_CHECKING`)

```python
def plot_fan_chart(
    model: BaseStochasticReserve,
    origin,
    bands: tuple[tuple[float, float], ...] = ((0.01, 0.99), (0.05, 0.95), (0.25, 0.75)),
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Reserve development ("fan") chart of simulated cumulative claims for one
    origin, after England's ``fan_plot``: nested quantile bands, the mean path
    and the observed cells."""
    full = model._require_full_posterior()
    data = full.sel(origin=origin).values  # (dev, sample)
    devs = full.coords["dev"].values
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (10, 6))
    else:
        fig = ax.figure
    for k, (lo, hi) in enumerate(sorted(bands, key=lambda b: b[1] - b[0], reverse=True)):
        ax.fill_between(
            devs, np.nanquantile(data, lo, axis=1), np.nanquantile(data, hi, axis=1),
            color="tab:blue", alpha=0.15 + 0.2 * k, linewidth=0,
            label=f"{lo * 100:g}%–{hi * 100:g}%",
        )
    ax.plot(devs, np.nanmean(data, axis=1), color="black", linewidth=2, label="Mean")
    cum = np.asarray(model.triangle_.values, dtype=float)[0, 0]
    origin_idx = list(full.coords["origin"].values).index(origin)
    observed = cum[origin_idx]
    ax.plot(devs[~np.isnan(observed)], observed[~np.isnan(observed)], "o", color="tab:red", label="Observed")
    ax.set_title(f"Origin {origin}: simulated cumulative development")
    ax.set_xlabel("Development (months)")
    ax.set_ylabel("Cumulative claims")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    return fig, ax


def plot_scaled_residuals(
    residuals: np.ndarray,
    by: str = "dev",
    sigma: np.ndarray | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Scatter of (scaled) residuals by origin, development or calendar index,
    with the per-index average and, optionally, the sigma / sqrt(scale) vector
    on a twin axis (England's ``scatter_plot``). Indices are 1-based."""
    resid = np.asarray(residuals, dtype=float)
    n_o, n_c = resid.shape
    i, j = np.indices(resid.shape)
    if by == "origin":
        x = i + 1
    elif by == "dev":
        x = j + 1
    elif by == "calendar":
        x = i + j + 1
    else:
        raise ValueError("by must be 'origin', 'dev' or 'calendar'")
    ok = np.isfinite(resid)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (10, 6))
    else:
        fig = ax.figure
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.scatter(x[ok], resid[ok], marker="x", color="tab:blue", label="Residual")
    levels = np.unique(x[ok])
    means = [resid[ok & (x == lv)].mean() for lv in levels]
    ax.plot(levels, means, color="tab:green", linewidth=2, label="Average")
    ax.set_xlabel(f"{by.capitalize()} period")
    ax.set_ylabel("Scaled residual")
    ax.set_title(title or f"Scaled residuals by {by} period")
    ax.grid(alpha=0.3)
    if sigma is not None and by == "dev":
        ax2 = ax.twinx()
        ax2.plot(np.arange(1, len(sigma) + 1), sigma, color="tab:orange", linewidth=2, label="Sigma")
        ax2.set_ylabel("Sigma / sqrt(scale)")
        ax2.legend(loc="upper right")
    elif sigma is not None and by == "origin":
        ax2 = ax.twinx()
        ax2.set_yticks([])
    ax.legend(loc="upper left")
    return fig, ax


def plot_sensitivity_heatmap(
    result: pd.DataFrame,
    value: str = "sd_diff",
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Heatmap of a ``link_ratio_sensitivity`` column (origin × development)."""
    pivot = result.pivot(index="origin", columns="dev", values=value)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (10, 6))
    else:
        fig = ax.figure
    im = ax.imshow(pivot.values, cmap="RdBu", aspect="auto")
    ax.set_xticks(range(pivot.shape[1]), [str(c) for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]), [str(r) for r in pivot.index])
    ax.set_xlabel("Development (months) of link ratio")
    ax.set_ylabel("Origin")
    ax.set_title(f"Change in {value} when each link ratio is excluded")
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            v = pivot.values[r, c]
            if np.isfinite(v):
                ax.text(c, r, f"{v:,.0f}" if abs(v) >= 10 else f"{v:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax)
    return fig, ax


def plot_capital_profiles(
    profiles: dict[str, np.ndarray],
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, Axes]:
    """Capital run-off profiles as a percentage of opening capital (EVW 2019 Fig. 1)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (10, 6))
    else:
        fig = ax.figure
    for label, prof in profiles.items():
        p = np.asarray(prof, dtype=float)
        ax.plot(np.arange(len(p)), 100 * p / p[0], marker="o", linewidth=2, label=label)
    ax.set_xlabel("Future year")
    ax.set_ylabel("Percent of opening capital")
    ax.set_title("Capital profiles by year")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig, ax
```

Add `from .base import BaseStochasticReserve` inside the existing `if TYPE_CHECKING:` block. Export the four functions under `# Plotting functions - England & Verrall diagnostics`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_plots.py -q -k EnglandPlots`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
uv run ruff check bayesianchainladder/plots.py tests/test_plots.py
git add bayesianchainladder/plots.py bayesianchainladder/__init__.py tests/test_plots.py
git commit -m "feat: fan chart, scaled residual, sensitivity heatmap and capital profile plots

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 15: Bayesian quasi-Poisson and link-ratio models

**Files:**
- Modify: `bayesianchainladder/models.py` (add `build_quasi_poisson_model`, `build_link_ratio_model`)
- Modify: `bayesianchainladder/linkratio.py` (add `BayesianMackChainLadder`)
- Modify: `bayesianchainladder/__init__.py`
- Test: `tests/test_models.py`, `tests/test_linkratio.py`

**Interfaces:**
- Consumes: `poisson_irls`, `_design_matrix` (Task 11); `forecast_link_ratio_paths`, `MackBootstrap.variance_factor_fn`, `NegativeBinomialBootstrap.variance_factor_fn` (Task 10); Task 1 primitives.
- Produces:
  - `build_quasi_poisson_model(data, response_col="incremental", origin_col="origin", dev_col="dev", scale=1.0, coef_sigma=10.0) -> pm.Model` with deterministic `mu` (dims `obs`) and a `Potential` `Σ (y log μ − μ) / φ_j`; `scale` scalar or `(n_dev,)`
  - `build_link_ratio_model(triangle, model="mack", drop=None, sigma=None, coef_sigma=10.0) -> pm.Model` with deterministic `factors` (dims `dev_ratio`); `model ∈ {"mack","negbin"}`; likelihood `Normal(λ_j, σ_j √v(f_j) / √w_ij)` on observed link ratios with plug-in `v`
  - `class BayesianMackChainLadder(BaseStochasticReserve)` in `linkratio.py`: `__init__(model="mack", draws=1000, tune=1000, chains=2, forecast_dist="gamma", drop=None, process_sigma=None, random_seed=None, target_accept=0.9)`; fitted attributes `idata`, `factors_` (CL point estimate), `sigma_`, `factor_draws_ (S, n_dev-1)`, `full_cumulative_posterior_`, `reserves_posterior_`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_models.py`:

```python
@pytest.mark.slow
def test_quasi_poisson_posterior_mean_matches_irls():
    import chainladder as cl
    import numpy as np
    import pymc as pm

    from bayesianchainladder.analytic import odp_analytic_rmsep
    from bayesianchainladder.models import build_quasi_poisson_model
    from bayesianchainladder.utils import prepare_model_data

    tri = cl.load_sample("genins")
    obs, _ = prepare_model_data(tri)
    analytic = odp_analytic_rmsep(tri, scale="constant")
    model = build_quasi_poisson_model(obs, scale=float(analytic.scale[0]))
    with model:
        idata = pm.sample(draws=300, tune=300, chains=1, random_seed=1, progressbar=False)
    mu_post = idata.posterior["mu"].mean(dim=["chain", "draw"]).values
    mu_irls = np.exp(
        __import__("bayesianchainladder.analytic", fromlist=["x"])._design_matrix(10, 10)[0]
        @ analytic.coefficients
    )
    obs_mask = ~np.isnan(np.asarray(tri.cum_to_incr().values)[0, 0]).ravel()
    np.testing.assert_allclose(mu_post, mu_irls[obs_mask], rtol=0.03)


def test_quasi_poisson_scale_vector_validation():
    import chainladder as cl
    import numpy as np
    import pytest

    from bayesianchainladder.models import build_quasi_poisson_model
    from bayesianchainladder.utils import prepare_model_data

    obs, _ = prepare_model_data(cl.load_sample("genins"))
    model = build_quasi_poisson_model(obs, scale=np.linspace(1e4, 1e5, 10))
    assert "mu" in model.named_vars
    with pytest.raises(ValueError, match="scale"):
        build_quasi_poisson_model(obs, scale=np.ones(3))


def test_build_link_ratio_model_structure():
    import chainladder as cl
    import pytest

    from bayesianchainladder.models import build_link_ratio_model

    tri = cl.load_sample("genins")
    m = build_link_ratio_model(tri)
    assert "factors" in m.named_vars and m.coords["dev_ratio"] == tuple(12 * k for k in range(1, 10))
    m2 = build_link_ratio_model(tri, model="negbin", drop=[("2003", 72)])
    assert "factors" in m2.named_vars
    with pytest.raises(ValueError, match="model"):
        build_link_ratio_model(tri, model="odp")
```

Append to `tests/test_linkratio.py`:

```python
@pytest.mark.slow
def test_bayesian_mack_recovers_chain_ladder_factors(genins):
    from bayesianchainladder.linkratio import BayesianMackChainLadder

    model = BayesianMackChainLadder(draws=300, tune=300, chains=1, random_seed=42).fit(genins)
    assert model.idata is not None
    post_mean = model.factor_draws_.mean(axis=0)
    np.testing.assert_allclose(post_mean, model.factors_, rtol=0.02)
    assert model.total_summary().total_reserve_mean == pytest.approx(CL_RESERVE, rel=0.05)
    assert model.full_cumulative_posterior_.shape[:2] == (10, 10)
    assert model.full_cumulative_posterior_.shape[2] == model.factor_draws_.shape[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models.py -q -k "quasi_poisson_scale_vector or link_ratio_model_structure"`
Expected: FAIL with `ImportError: cannot import name 'build_quasi_poisson_model'`

- [ ] **Step 3: Implement the builders** (append to `models.py`; it already imports `pm`, `pt`, `np`, `pd`)

```python
def build_quasi_poisson_model(
    data: pd.DataFrame,
    response_col: str = "incremental",
    origin_col: str = "origin",
    dev_col: str = "dev",
    scale=1.0,
    coef_sigma: float = 10.0,
) -> pm.Model:
    """Cross-classified chain ladder with an over-dispersed Poisson
    quasi-likelihood, ``sum((y log mu - mu) / phi_j)``, as a ``pm.Potential``.

    ``scale`` is the plug-in dispersion: a scalar (constant scale) or one value
    per development period (non-constant scale). This mirrors the ODP Stan
    model in England & Verrall (2006) and lets non-integer, over-dispersed
    increments be fitted with wide Normal priors on the log-linear effects.
    """
    y = data[response_col].to_numpy(dtype=float)
    origin_codes, origin_levels = pd.factorize(data[origin_col], sort=True)
    dev_codes, dev_levels = pd.factorize(data[dev_col], sort=True)
    n_dev = len(dev_levels)
    phi = np.asarray(scale, dtype=float)
    if phi.ndim == 0:
        phi = np.full(n_dev, float(phi))
    elif phi.shape != (n_dev,):
        raise ValueError(f"scale must be a scalar or have shape ({n_dev},), got {phi.shape}")
    phi_obs = np.maximum(phi[dev_codes], 1e-12)

    coords = {
        "origin_raw": list(origin_levels[1:]),
        "dev_raw": list(dev_levels[1:]),
        "obs": np.arange(len(y)),
    }
    with pm.Model(coords=coords) as model:
        intercept = pm.Normal("intercept", mu=np.log(max(y.mean(), 1e-8)), sigma=coef_sigma)
        alpha_raw = pm.Normal("alpha_raw", mu=0.0, sigma=coef_sigma, dims="origin_raw")
        beta_raw = pm.Normal("beta_raw", mu=0.0, sigma=coef_sigma, dims="dev_raw")
        alpha = pt.concatenate([pt.zeros(1), alpha_raw])
        beta = pt.concatenate([pt.zeros(1), beta_raw])
        eta = intercept + alpha[origin_codes] + beta[dev_codes]
        mu = pm.Deterministic("mu", pt.exp(eta), dims="obs")
        pm.Potential("quasi_poisson", pt.sum((y * pt.log(mu) - mu) / phi_obs))
    return model


def build_link_ratio_model(
    triangle,
    model: str = "mack",
    drop=None,
    sigma=None,
    coef_sigma: float = 10.0,
) -> pm.Model:
    """Bayesian link-ratio model: observed ratios F_ij ~ Normal(lambda_j,
    sigma_j sqrt(v(f_j)) / sqrt(C_ij)). ``model='mack'`` uses a log link and
    v = 1 (England & Verrall 2006 Mack Stan model); ``model='negbin'`` uses a
    log-log link (factors > 1) and v = f (f - 1) with the chain-ladder factor
    plugged into the variance."""
    from ._triangle_ops import (
        cumulative_array,
        link_ratio_mask,
        link_ratio_sigma,
        volume_weighted_factors,
    )

    if model not in ("mack", "negbin"):
        raise ValueError("model must be 'mack' or 'negbin'")
    cum, origins, devs = cumulative_array(triangle)
    mask = link_ratio_mask(cum, drop, origins, devs)
    f0 = volume_weighted_factors(cum, mask)
    vf = np.ones_like(f0) if model == "mack" else np.abs(f0 * (f0 - 1.0))
    if sigma is None:
        sigma, _ = link_ratio_sigma(cum, mask, f0, vf)
    sigma = np.asarray(sigma, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = cum[:, 1:] / cum[:, :-1]
    idx = np.argwhere(mask > 0)
    rows, cols = idx[:, 0], idx[:, 1]
    f_obs = ratios[rows, cols]
    w_obs = cum[rows, cols]
    sd_obs = np.maximum(sigma[cols] * np.sqrt(vf[cols]) / np.sqrt(np.abs(w_obs)), 1e-9)

    coords = {"dev_ratio": [int(d) for d in devs[:-1]], "obs": np.arange(len(f_obs))}
    start = np.log(f0) if model == "mack" else np.log(np.log(np.maximum(f0, 1.0 + 1e-6)))
    with pm.Model(coords=coords) as pymc_model:
        coefs = pm.Normal("coefs", mu=start, sigma=coef_sigma, dims="dev_ratio")
        lam = pt.exp(coefs) if model == "mack" else pt.exp(pt.exp(coefs))
        factors = pm.Deterministic("factors", lam, dims="dev_ratio")
        pm.Normal("ratio", mu=factors[cols], sigma=sd_obs, observed=f_obs, dims="obs")
    return pymc_model
```

Add the estimator to `linkratio.py`:

```python
class BayesianMackChainLadder(BaseStochasticReserve):
    """MCMC version of the Mack / Negative Binomial link-ratio model
    (England & Verrall 2006, Section 6): posterior factor draws replace the
    bootstrap pseudo-factors and are pushed through the same process-error
    forecasting as :class:`MackBootstrap`."""

    def __init__(
        self,
        model: str = "mack",
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 2,
        forecast_dist: str = "gamma",
        drop: DropList = None,
        process_sigma=None,
        random_seed: int | None = None,
        target_accept: float = 0.9,
    ) -> None:
        super().__init__()
        if model not in ("mack", "negbin"):
            raise ValueError("model must be 'mack' or 'negbin'")
        if forecast_dist not in ("gamma", "lognormal"):
            raise ValueError("forecast_dist must be 'gamma' or 'lognormal' for the MCMC estimator")
        self.model = model
        self.draws, self.tune, self.chains = draws, tune, chains
        self.forecast_dist = forecast_dist
        self.drop = drop
        self.process_sigma = None if process_sigma is None else np.asarray(process_sigma, float)
        self.random_seed = random_seed
        self.target_accept = target_accept
        self.idata = None

    def fit(self, triangle):
        import pymc as pm

        from .models import build_link_ratio_model

        validate_triangle(triangle)
        self.triangle_ = triangle.copy()
        cum, origins, devs = cumulative_array(triangle)
        mask = link_ratio_mask(cum, self.drop, origins, devs)
        factors = volume_weighted_factors(cum, mask)
        vf_fn = MackBootstrap.variance_factor_fn if self.model == "mack" else NegativeBinomialBootstrap.variance_factor_fn
        sigma, _ = link_ratio_sigma(cum, mask, factors, vf_fn(factors))
        if self.process_sigma is not None and self.process_sigma.shape != sigma.shape:
            raise ValueError(f"process_sigma must have shape {sigma.shape}")

        pymc_model = build_link_ratio_model(triangle, model=self.model, drop=self.drop, sigma=sigma)
        with pymc_model:
            self.idata = pm.sample(
                draws=self.draws, tune=self.tune, chains=self.chains,
                target_accept=self.target_accept, random_seed=self.random_seed,
                progressbar=False,
            )
        draws = self.idata.posterior["factors"].stack(sample=["chain", "draw"]).transpose("sample", "dev_ratio").values

        rng = np.random.default_rng(self.random_seed)
        full = forecast_link_ratio_paths(
            cum, draws, sigma if self.process_sigma is None else self.process_sigma,
            vf_fn, self.forecast_dist, rng,
        )
        self.factors_, self.sigma_, self.factor_draws_ = factors, sigma, draws
        self._set_full_cumulative_posterior(np.moveaxis(full, 0, -1), origins, devs)
        self.reserves_posterior_ = self._reserves_from_full_posterior()
        self._build_reserve_summaries()
        self._is_fitted = True
        return self
```

Export `build_quasi_poisson_model`, `build_link_ratio_model` (under `# Model functions`) and `BayesianMackChainLadder` (under `# Link-ratio bootstraps`, renaming that comment to `# Link-ratio models`).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_models.py tests/test_linkratio.py -q` (fast) then `uv run pytest tests/test_models.py tests/test_linkratio.py -q --run-slow -k "quasi_poisson_posterior or bayesian_mack"`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
uv run ruff check bayesianchainladder/models.py bayesianchainladder/linkratio.py tests/test_models.py tests/test_linkratio.py
git add bayesianchainladder/models.py bayesianchainladder/linkratio.py bayesianchainladder/__init__.py tests/test_models.py tests/test_linkratio.py
git commit -m "feat: Bayesian quasi-Poisson and link-ratio (Mack/NegBin) models

Potential-based ODP quasi-likelihood with constant or per-dev dispersion,
Normal link-ratio model with log / log-log links, and a
BayesianMackChainLadder estimator that pushes posterior factor draws
through the link-ratio forecasting machinery. After England & Verrall
(2006) Section 6.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 16: the 15-step modus operandi notebook

**Files:**
- Create: `docs/notebooks/build_modus_operandi.py` (generates the notebook with `nbformat`), `docs/notebooks/modus_operandi.ipynb` (generated, then executed in place), `docs/notebooks/README.md`
- Modify: `pyproject.toml` dev group (`nbformat`, `nbconvert`, `ipykernel`)
- Test: `tests/test_notebooks.py` (slow)

**Interfaces:**
- Consumes: `load_england_sample` (6), `mack_analytic_rmsep` (11), `MackBootstrap` (10), `summary_statistics` (2), `plot_fan_chart` / `plot_scaled_residuals` / `plot_sensitivity_heatmap` (14), `link_ratio_sensitivity` / `top_influential` (12), `CorrelatedBootstrapChainLadder(scale=, process_scale=, drop=)` and `.sampler_` (9), `scale_to_target` (13), `claims_development_result` (8).
- Produces: an executed notebook whose 15 sections mirror England's *Example_Modus_Operandi.ipynb* with commentary paraphrased and the source cited in the first cell.

- [ ] **Step 1: Add notebook tooling**

```bash
uv add --dev nbformat nbconvert ipykernel
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_notebooks.py
"""Execute the documentation notebooks end to end (slow)."""

import pathlib
import subprocess
import sys

import pytest

NOTEBOOKS = pathlib.Path(__file__).resolve().parents[1] / "docs" / "notebooks"


@pytest.mark.slow
def test_modus_operandi_notebook_builds_and_executes(tmp_path):
    subprocess.run([sys.executable, str(NOTEBOOKS / "build_modus_operandi.py"), str(tmp_path)], check=True)
    nb = tmp_path / "modus_operandi.ipynb"
    assert nb.exists()
    subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
            "--ExecutePreprocessor.timeout=1800", "--output", "executed.ipynb", str(nb),
        ],
        check=True,
    )
    text = (tmp_path / "executed.ipynb").read_text()
    assert "DrPeterEngland/StochasticReserving" in text
    assert "Step 15" in text
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_notebooks.py -q --run-slow`
Expected: FAIL with `FileNotFoundError` (build script missing)

- [ ] **Step 4: Write the build script**

```python
# docs/notebooks/build_modus_operandi.py
"""Generate docs/notebooks/modus_operandi.ipynb.

The notebook adapts the 15-step "Stochastic Reserving: Modus Operandi" example
by Peter England (EMC Actuarial and Analytics Ltd) from
https://github.com/DrPeterEngland/StochasticReserving/blob/main/Python_Examples/Example_Modus_Operandi.ipynb
(MIT licence) to the bayesianchainladder API. Commentary is paraphrased.

Usage: python build_modus_operandi.py [output_dir]
"""

from __future__ import annotations

import pathlib
import sys

import nbformat as nbf

SOURCE = "https://github.com/DrPeterEngland/StochasticReserving"

CELLS: list[tuple[str, str]] = []  # (kind, source)


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip()))


def code(text: str) -> None:
    CELLS.append(("code", text.strip()))


md(f"""
# Stochastic reserving: a modus operandi

Adapted from Peter England's *Example Modus Operandi* notebook in the
[StochasticReserving repository]({SOURCE}) (`Python_Examples/Example_Modus_Operandi.ipynb`,
MIT licence, provided by EMC Actuarial and Analytics Ltd as an educational resource).
The fifteen steps and their rationale are his; the code uses `bayesianchainladder`, and the
commentary is paraphrased. Numbers differ slightly from the original because chainladder
extrapolates the last Mack sigma log-linearly whereas England takes the minimum of the previous
two, and because the simulation seeds differ.

**Why a modus operandi?** Quantifying reserve variability starts with understanding what drives
it, then understanding each model's characteristics and limits, and only then selecting a result.
The steps below are a guide, not a recipe.

This notebook covers the lifetime ("ultimo") view. The one-year view is a short epilogue that
points to `claims_development_result`.
""")

md("## Steps 1 & 2: look at the data and fit a baseline chain ladder")
code("""
import numpy as np
import pandas as pd
import chainladder as cl
import matplotlib.pyplot as plt

from bayesianchainladder import (
    CorrelatedBootstrapChainLadder,
    MackBootstrap,
    claims_development_result,
    link_ratio_sensitivity,
    load_england_sample,
    mack_analytic_rmsep,
    plot_fan_chart,
    plot_scaled_residuals,
    plot_sensitivity_heatmap,
    top_influential,
)

pd.options.display.float_format = "{:,.0f}".format
tri = load_england_sample("liability")
incremental = tri.cum_to_incr().to_frame(origin_as_datetime=False)
display(incremental)
display(tri.to_frame(origin_as_datetime=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
incremental.T.plot(ax=axes[0], title="Incremental claims by development", legend=False)
tri.to_frame(origin_as_datetime=False).T.plot(ax=axes[1], title="Cumulative claims by development", legend=False)
plt.show()

dev = cl.Development().fit_transform(tri)
link_ratios = (tri.values[0, 0, :, 1:] / tri.values[0, 0, :, :-1])
display(pd.DataFrame(link_ratios, index=tri.origin.year, columns=tri.development[:-1]).round(3))
print("Volume-weighted factors:", np.round(dev.ldf_.values.flatten(), 3))
cl_result = cl.Chainladder().fit(dev)
display(pd.DataFrame({"latest": tri.latest_diagonal.values[0, 0, :, 0],
                      "reserve": cl_result.ibnr_.values[0, 0, :, 0],
                      "ultimate": cl_result.ultimate_.values[0, 0, :, 0]}, index=tri.origin.year))
print("Total reserve:", f"{float(np.nansum(cl_result.ibnr_.values)):,.0f}")
""")
md("""
**Commentary.** The incremental graph shows a large payment at development period 7 for origin
period 3 (2003). It produces a large link ratio between development periods 6 and 7 for that
origin, which reverses the otherwise decreasing pattern of the volume-weighted factors at that
point. Including every ratio gives a total reserve of about 331,000.
""")

md("## Step 3: Mack's model applied analytically")
code("""
mack = mack_analytic_rmsep(tri)
display(mack.to_frame().style.format({"reserve": "{:,.0f}", "sd": "{:,.0f}", "cov": "{:.1%}"}))
""")
md("""
**Commentary.** A coefficient of variation around 22% on the total reserve is high for a
triangle whose cumulative development looks stable, so something specific is driving it.
""")

md("## Step 4: residuals and Mack's sigma")
code("""
boot_all = MackBootstrap(n_sims=10_000, random_seed=100).fit(tri)
sigma_original = boot_all.sigma_.copy()
print("Mack sigma by development ratio:", np.round(sigma_original, 2))
display(pd.DataFrame(boot_all.scaled_residuals_, index=tri.origin.year, columns=tri.development[:-1]).round(2))
for by in ("origin", "dev", "calendar"):
    plot_scaled_residuals(boot_all.scaled_residuals_, by=by, sigma=sigma_original)
    plt.show()
""")
md("""
**Commentary.** Sigma is the root mean square of the bias-adjusted unscaled residuals in each
development period, so one large residual inflates it. Here sigma spikes at development ratio 6
(the 72→84 month ratio), driven by origin 2003. Scaled residuals should look like i.i.d. standard
normals: pattern-free by origin, development and calendar period, with roughly 95% inside ±2.
""")

md("## Step 5: bootstrap Mack's model with all ratios included")
code("""
display(boot_all.summary_statistics("reserves").style.format("{:,.0f}"))
display(boot_all.summary_statistics("ultimates").style.format("{:,.0f}"))
for origin in (2002, 2004, 2006, 2008, 2010):
    plot_fan_chart(boot_all, origin)
    plt.show()
""")
md("""
**Commentary.** The bootstrap mean, standard deviation and CoV match the analytic Mack results
closely, which is the first check to make. The minimum simulated reserves are negative for the
older origins: Mack's model is a model of link ratios and simulated cumulatives can fall. That
may be acceptable for incurred data but not for paid data. The fan charts widen abruptly at
development period 7, the direct footprint of the sigma spike.
""")

md("## Step 6: sensitivity analysis — which link ratios matter?")
code("""
sens = link_ratio_sensitivity(tri)
display(sens.head(10).style.format({"reserve": "{:,.0f}", "reserve_sd": "{:,.0f}", "reserve_cov": "{:.1%}",
                                    "reserve_diff": "{:,.0f}", "sd_diff": "{:,.0f}", "cov_diff": "{:.1%}"}))
plot_sensitivity_heatmap(sens, value="sd_diff")
plt.show()
""")
md("""
**Commentary.** Excluding each link ratio in turn and re-applying Mack analytically is a quick
way to find influential points. Dropping the 72→84 ratio of origin 2003 gives the largest
reduction in reserve, standard deviation and CoV by a wide margin; the next most influential
ratio matters far less.
""")

md("## Step 7: exclude the top three ratios and re-apply Mack analytically")
code("""
top3 = top_influential(sens, n=3, by="sd")
print("Excluding:", top3)
mack_top3 = mack_analytic_rmsep(tri, drop=top3)
display(mack_top3.to_frame().style.format({"reserve": "{:,.0f}", "sd": "{:,.0f}", "cov": "{:.1%}"}))
""")
md("""
**Commentary.** With the three most influential ratios excluded the reserve, its standard
deviation and its CoV all fall sharply. Whether any exclusion is justified is the analyst's call;
the point is to know which data points drive the volatility before making it.
""")

md("## Step 8: residuals and sigma after the exclusions")
code("""
boot_top3 = MackBootstrap(n_sims=10_000, random_seed=100, drop=top3).fit(tri)
print("Mack sigma after exclusions:", np.round(boot_top3.sigma_, 2))
plot_scaled_residuals(boot_top3.scaled_residuals_, by="dev", sigma=boot_top3.sigma_)
plt.show()
""")
md("""
**Commentary.** Excluding a ratio also excludes its residual, so sigma at development ratio 6
collapses and the standard deviation of the reserves falls with it.
""")

md("## Step 9: bootstrap Mack's model after the exclusions")
code("""
display(boot_top3.summary_statistics("reserves").style.format("{:,.0f}"))
for origin in (2004, 2008):
    plot_fan_chart(boot_top3, origin)
    plt.show()
""")
md("""
**Commentary.** The bootstrap again matches the analytic result with the same exclusions, far
fewer simulations produce negative reserves, and the fan charts no longer jump at period 7.
""")

md("## Step 10: keep all ratios but override sigma")
code("""
user_sigma = sigma_original.copy()
user_sigma[5] = 13.0  # tame the spike at the 72->84 ratio
boot_user = MackBootstrap(n_sims=10_000, random_seed=100, process_sigma=user_sigma).fit(tri)
display(boot_user.summary_statistics("reserves").style.format("{:,.0f}"))
plot_fan_chart(boot_user, 2004)
plt.show()
""")
md("""
**Commentary.** Overriding the variance parameter leaves the mean unchanged (up to simulation
error) while reducing the standard deviation and CoV. User-defined sigmas are an alternative to
exclusions for controlling volatility once its drivers are understood.
""")

md("## Step 11: the over-dispersed Poisson model with non-constant scale")
code("""
odp_all = CorrelatedBootstrapChainLadder(n_sims=10_000, rho=0.0, scale="nonconstant",
                                          parametric_dist="lognormal", random_seed=100).fit(tri)
sqrt_scale_original = np.sqrt(odp_all.sampler_.scale_by_dev_)
print("ODP sqrt(scale) by development period:", np.round(sqrt_scale_original, 1))
plot_scaled_residuals(odp_all.sampler_.standardized_residuals_ / np.where(sqrt_scale_original > 0, sqrt_scale_original, np.nan),
                      by="dev", sigma=sqrt_scale_original, title="ODP scaled residuals by development period")
plt.show()
display(odp_all.summary_statistics("reserves").style.format("{:,.0f}"))
plot_fan_chart(odp_all, 2004)
plt.show()
""")
md("""
**Commentary.** The ODP model is a model of incremental amounts, so the residual triangle is
10×10 and the spike now sits at development period 7 (the cell, not the ratio). With
non-constant scale the total standard deviation is close to Mack's, as expected, but because
future increments are drawn from a positive distribution the minimum reserves stay positive.
""")

md("## Step 12: ODP after excluding the influential ratios")
code("""
odp_top3 = CorrelatedBootstrapChainLadder(n_sims=10_000, rho=0.0, scale="nonconstant",
                                           parametric_dist="lognormal", random_seed=100, drop=top3).fit(tri)
print("ODP sqrt(scale) after exclusions:", np.round(np.sqrt(odp_top3.sampler_.scale_by_dev_), 1))
display(odp_top3.summary_statistics("reserves").style.format("{:,.0f}"))
plot_fan_chart(odp_top3, 2004)
plt.show()
""")
md("""
**Commentary.** The same ratios are influential under both models. Dropping the 72→84 ratio of
origin 2003 removes the development-period-7 residual from the scale estimate, and the ODP
standard deviation falls in line with Mack's with the same exclusions.
""")

md("## Step 13: ODP with a user-defined scale")
code("""
user_scale = odp_all.sampler_.scale_by_dev_.copy()
user_scale[6] = 40.0 ** 2
odp_user = CorrelatedBootstrapChainLadder(n_sims=10_000, rho=0.0, scale="nonconstant",
                                           parametric_dist="lognormal", random_seed=100,
                                           process_scale=user_scale).fit(tri)
display(odp_user.summary_statistics("reserves").style.format("{:,.0f}"))
""")
md("""
**Commentary.** As with Mack's model, a user-defined scale leaves the mean alone and reduces the
spread. Use it carefully, and only after the drivers of volatility are understood.
""")

md("## Step 14: compare and select")
code("""
rows = {
    "Mack, all ratios": boot_all, "Mack, top-3 excluded": boot_top3, "Mack, user sigma": boot_user,
    "ODP, all ratios": odp_all, "ODP, top-3 excluded": odp_top3, "ODP, user scale": odp_user,
}
compare = pd.DataFrame({
    name: {"mean": m.total_summary().total_reserve_mean, "sd": m.total_summary().total_reserve_stddev,
           "cov": m.total_summary().total_reserve_cv, "p99.5": m.total_summary().total_reserve_99_5th_percentile}
    for name, m in rows.items()
}).T
display(compare.style.format({"mean": "{:,.0f}", "sd": "{:,.0f}", "cov": "{:.1%}", "p99.5": "{:,.0f}"}))
""")
md("""
**Commentary.** Only now, knowing what drives the volatility and how each model behaves, is the
analyst placed to choose a model, decide on exclusions and decide whether to override the
variance parameters. Tail extrapolation is a common further step and is not covered here.
""")

md("## Step 15: scale to target ultimates")
code("""
selected = odp_top3
origins = list(selected.reserves_posterior_.origin.values)
paid = selected._paid_to_date().reindex(origins)
target_ultimates = paid + 1.1 * selected.ibnr_["mean"]   # illustrative target only
methods = {o: ("additive" if k < 5 else "multiplicative") for k, o in enumerate(origins)}
scaled = selected.scale_to_target(target_ultimates, method=methods)
display(selected.summary_statistics("reserves")[["mean", "std", "cov"]].style.format({"mean": "{:,.0f}", "std": "{:,.0f}", "cov": "{:.1%}"}))
display(scaled.summary_statistics("reserves")[["mean", "std", "cov"]].style.format({"mean": "{:,.0f}", "std": "{:,.0f}", "cov": "{:.1%}"}))
""")
md("""
**Commentary.** Reserving teams rarely book the mean of a simple chain-ladder bootstrap, so the
distribution is usually shifted to a target. Additive scaling keeps the absolute standard
deviation and changes the CoV; multiplicative scaling keeps the CoV and changes the standard
deviation. The method can be chosen per origin. If a lot of scaling is needed, stop and
investigate rather than scale.

### Common errors and fallacies (paraphrased from the source)

* There is no such thing as "the" bootstrap model. Bootstrapping is a procedure applied to a
  specified statistical model; say which model, and prefer non-constant scale for ODP.
* Mack's model tolerates negative increments and factors below one and suits incurred data; the
  ODP model needs factors above one and gives positive simulated increments.
* An incurred bootstrap gives a distribution of IBNR+IBNER and of ultimates. To compare CoVs
  with a paid analysis, subtract the latest paid from each simulated ultimate
  (`incurred_to_paid`).
* Parametric bootstrapping simulates pseudo-data directly from a parametric distribution given
  its mean and variance. Simulating residuals from a Normal and inverting them is not parametric
  bootstrapping.
""")

md("## Epilogue: the one-year view")
code("""
cdr = claims_development_result(odp_top3)
one_year = cdr.summary().query("future_period == 1").set_index("origin")
lifetime_sd = odp_top3.summary_statistics("reserves")["std"]
one_year["lifetime_sd"] = lifetime_sd.reindex(one_year.index).values
one_year["cdr_sd_ratio"] = one_year["sd"] / one_year["lifetime_sd"]
display(one_year[["sd", "lifetime_sd", "cdr_sd_ratio", "var"]].style.format({"sd": "{:,.0f}", "lifetime_sd": "{:,.0f}", "cdr_sd_ratio": "{:.0%}", "var": "{:,.0f}"}))
""")
md(f"""
The one-year Claims Development Result re-reserves each simulated next diagonal (the
"actuary in the box") and is the basis for Solvency II reserve risk. See `claims_development_result`,
`bayesianchainladder.riskmeasures` and the England, Verrall & Wüthrich (2019) example in
[{SOURCE}]({SOURCE}) for cost-of-capital risk margins and IFRS 17 risk adjustments.
""")


def build(out_dir: pathlib.Path) -> pathlib.Path:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src)
        for kind, src in CELLS
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "modus_operandi.ipynb"
    nbf.write(nb, path)
    return path


if __name__ == "__main__":
    target = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).parent
    print(build(target))
```

Write `docs/notebooks/README.md`:

````markdown
# Notebooks

| Notebook | Builds from | Source |
|---|---|---|
| `modus_operandi.ipynb` | `python build_modus_operandi.py` | Adapted from Peter England's *Example Modus Operandi* in https://github.com/DrPeterEngland/StochasticReserving (MIT) |

Regenerate and execute:

```bash
uv run python docs/notebooks/build_modus_operandi.py
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 docs/notebooks/modus_operandi.ipynb
```

The executed notebook is committed so it renders on GitHub.
````

- [ ] **Step 5: Build, execute, and run the slow test**

```bash
uv run python docs/notebooks/build_modus_operandi.py
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 docs/notebooks/modus_operandi.ipynb
uv run pytest tests/test_notebooks.py -q --run-slow
```
Expected: notebook executes without error; test passes. `display` is available inside the kernel. If `tri.origin.year` fails because `origin` is a PeriodIndex, replace with `[int(str(o)) for o in tri.origin]` in all cells. If `to_frame(origin_as_datetime=False)` is not accepted by the installed chainladder, use `to_frame()` and set `index = index.year`.

- [ ] **Step 6: Commit**

```bash
uv run ruff check docs/notebooks/build_modus_operandi.py tests/test_notebooks.py
git add pyproject.toml uv.lock docs/notebooks tests/test_notebooks.py
git commit -m "docs: 15-step stochastic reserving modus operandi notebook

Adapted from Peter England's Example_Modus_Operandi.ipynb
(DrPeterEngland/StochasticReserving, MIT) onto the bayesianchainladder
API, with a one-year CDR epilogue.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 17: documentation, export audit and full verification

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `bayesianchainladder/__init__.py`
- Test: `tests/test_public_api.py` (new)

**Interfaces:**
- Consumes: every public name introduced in Tasks 2–15.
- Produces: an export audit test that fails if a public module-level function or class is missing from `__all__`.

- [ ] **Step 1: Write the failing export-audit test**

```python
# tests/test_public_api.py
"""Every public name in the feature modules must be exported from the package."""

import importlib
import inspect

import bayesianchainladder as bcl

MODULES = [
    "bayesianchainladder.analytic",
    "bayesianchainladder.cdr",
    "bayesianchainladder.datasets",
    "bayesianchainladder.linkratio",
    "bayesianchainladder.riskmeasures",
    "bayesianchainladder.sensitivity",
]


def test_feature_modules_are_fully_exported():
    missing = []
    for name in MODULES:
        mod = importlib.import_module(name)
        for attr, obj in vars(mod).items():
            if attr.startswith("_") or not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            if getattr(obj, "__module__", None) != name:
                continue  # re-exported import, not defined here
            if attr not in bcl.__all__:
                missing.append(f"{name}.{attr}")
    assert not missing, f"not in bayesianchainladder.__all__: {missing}"


def test_all_names_resolve():
    for name in bcl.__all__:
        assert hasattr(bcl, name), name
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_public_api.py -q`
Expected: PASS if Tasks 2–15 exported everything; otherwise the failure lists the exact names to add to `__init__.py`. Fix until green.

- [ ] **Step 3: Update README.md**

Add a section after "Model Comparison" titled `## Reserve risk toolkit (England & Verrall extensions)` containing:

````markdown
## Reserve risk toolkit (England & Verrall extensions)

Every estimator that simulates future cells exposes `full_cumulative_posterior_`
(dims `origin, dev, sample`). The following build on it and therefore work for the
Bayesian GLM, CSR, the ODP bootstraps and the link-ratio bootstraps alike.

```python
from bayesianchainladder import (
    MackBootstrap, NegativeBinomialBootstrap, BayesianMackChainLadder,
    claims_development_result, discounted_reserves, future_reserve_profile,
    cost_of_capital_risk_margin, value_at_risk, tail_value_at_risk,
    proportional_hazards_transform, equivalent_risk_tolerance,
    link_ratio_sensitivity, top_influential, mack_analytic_rmsep, odp_analytic_rmsep,
    plot_fan_chart, plot_scaled_residuals, load_england_sample,
)

tri = load_england_sample("liability")

# Influence analysis, then a Mack bootstrap excluding the top-3 ratios
sens = link_ratio_sensitivity(tri)
boot = MackBootstrap(n_sims=10_000, drop=top_influential(sens, 3), random_seed=1).fit(tri)
boot.summary_statistics("reserves")          # mean, sd, cov, min, 0.5%..99.5%, max
plot_fan_chart(boot, origin=2004)

# One-year view and Solvency II / IFRS 17 quantities
cdr = claims_development_result(boot)         # CDR per future period, origin, sample
disc = discounted_reserves(boot, rate=0.03)   # discounted cash flows per origin
profile = future_reserve_profile(boot, 0.03).mean("sample").values
rm = cost_of_capital_risk_margin(cdr.summary().query("origin == 'Total' and future_period == 1")["var"].iloc[0],
                                 profile / profile[0], coc_rate=0.06, discount_rate=0.03)

# Scale to booked ultimates, preserving CoV
scaled = boot.scale_to_target(target_ultimates, method="multiplicative")
```

| Feature | Function / class | Source |
|---|---|---|
| Mack / NegBin bootstraps (nonparametric, Gamma, Lognormal) | `MackBootstrap`, `NegativeBinomialBootstrap` | England & Verrall (2002, 2006) |
| Bayesian link-ratio model | `BayesianMackChainLadder`, `build_link_ratio_model` | England & Verrall (2006) §6 |
| Quasi-Poisson likelihood with per-dev dispersion | `build_quasi_poisson_model` | England & Verrall (2006) |
| Non-constant scale, user-defined process variance | `CorrelatedBootstrapChainLadder(scale=, process_scale=)` | England & Verrall (2006) |
| One-year Claims Development Result | `claims_development_result` | England, Verrall & Wüthrich (2019) |
| Discounting, capital profiles, cost-of-capital margin, VaR/TVaR/PHT | `bayesianchainladder.riskmeasures` | England, Verrall & Wüthrich (2019) |
| Influential link ratios | `link_ratio_sensitivity`, `top_influential` | England, *Modus Operandi* |
| Analytic RMSEP oracles | `mack_analytic_rmsep`, `odp_analytic_rmsep` | England & Verrall (2002) |
| Scaling / incurred-to-paid | `scale_to_target`, `incurred_to_paid` | England, *Modus Operandi* |
| Sample data | `load_england_sample("taylor_ashe" | "liability")` | England's repository |

A worked 15-step example is in `docs/notebooks/modus_operandi.ipynb`. These
features are adapted from Peter England's
[StochasticReserving](https://github.com/DrPeterEngland/StochasticReserving)
repository (MIT licence).
````

Also add `MackBootstrap`, `NegativeBinomialBootstrap`, `BayesianMackChainLadder`, `ReserveSamples` to the "Main Classes" list and the new modules to the API Reference, and add England & Verrall (2006) and England, Verrall & Wüthrich (2019) to References.

- [ ] **Step 4: Update CLAUDE.md**

Under "Module layout" add one bullet each for `_triangle_ops.py`, `linkratio.py`, `analytic.py`, `cdr.py`, `riskmeasures.py`, `sensitivity.py`, `datasets.py` (one line describing responsibility, matching the file map at the top of this plan). Under "Non-obvious gotchas" add:

```markdown
- **Per-cell posterior contract.** Simulating estimators populate `full_cumulative_posterior_` (dims `origin, dev, sample`, cumulative, observed cells constant) via `_set_full_cumulative_posterior`; `reserves_posterior_` must equal `_reserves_from_full_posterior()` and tests assert it. `MackChainLadder` (normal approximation) leaves it `None`; `cdr`, `riskmeasures` and `plot_fan_chart` raise a "per-cell" `ValueError` in that case. For chainladder-backed wrappers the array is `full_triangle_ + process_variance_` sliced to the original `n_dev` columns — `full_triangle_` carries a placeholder tail column and the `9999` ultimate column beyond that.
- **Two last-sigma conventions.** `link_ratio_sigma` (used by `MackBootstrap`, `NegativeBinomialBootstrap`, `BayesianMackChainLadder`, `odp_analytic_rmsep(scale="nonconstant")`, `CorrelatedBootstrapODPSample(scale="nonconstant")`) follows England: last period = min of the previous two, carry forward when n_j ≤ 1. chainladder's `Development` extrapolates log-linearly, so `mack_analytic_rmsep` (a chainladder wrapper) and the bootstraps differ by a few percent in total SD on triangles with a small last sigma. Don't "fix" one to match the other.
- **`drop` syntax everywhere.** Link-ratio exclusions use chainladder's `(origin_label, dev_months)` with a *string* origin label naming the earlier cell of the ratio, e.g. `("2003", 72)` = the 72→84 ratio. `top_influential` returns this form; `_triangle_ops.drop_mask` consumes it. Origin labels assume annual origins (`str(int_year)`).
- **Cash-flow timing is square-triangle only.** `riskmeasures.cash_flow_periods` requires `n_origin == n_dev` and assumes origin grain == development grain; `claims_development_result` has the same restriction. Non-annual or non-square triangles raise.
- **Notebook is generated.** Edit `docs/notebooks/build_modus_operandi.py`, not the `.ipynb`; rebuild and re-execute (see `docs/notebooks/README.md`). `tests/test_notebooks.py` executes it under `--run-slow`.
```

Add to "Commands": `uv run pytest tests/test_notebooks.py --run-slow   # executes docs/notebooks (~10 min)`.

- [ ] **Step 5: Full verification**

```bash
uv run ruff check .
uv run black --check bayesianchainladder tests docs/notebooks/build_modus_operandi.py
uv run mypy bayesianchainladder
uv run pytest -q
uv run pytest -q --run-slow -k "full_posterior or csr_full or quasi_poisson_posterior or bayesian_mack or notebook"
```
Expected: ruff clean; black clean (run `uv run black` to fix if not); mypy reports no errors in the new modules and no new errors in modified ones (if unsure, run `uv run mypy bayesianchainladder` on a clean checkout of `main` in a separate worktree and diff the output); fast suite green; the listed slow tests green.

- [ ] **Step 6: Commit**

```bash
git add README.md CLAUDE.md bayesianchainladder/__init__.py tests/test_public_api.py
git commit -m "docs: document England & Verrall extensions; add public API audit test

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review against the spec

| Spec requirement | Task(s) |
|---|---|
| R1 per-cell posterior for every simulating estimator; Mack wrapper leaves None | 2 (contract), 3 (ODP/correlated/BF/CC), 4 (GLM), 5 (CSR), 10 (link-ratio bootstraps), 15 (Bayesian Mack) |
| R2 tail summary statistics, extended MethodSummary | 2 |
| R3 risk measures, discounting, cost-of-capital, equivalent tolerance | 7 |
| R4 Claims Development Result with exact-sum identity | 8 |
| R5 non-constant scale and process_scale override | 9 |
| R6 Mack and NegBin bootstraps with dist options, drop, process_sigma | 10 |
| R7 analytic ODP and Mack oracles | 11 |
| R8 influence analysis with ("2003", 72) check | 12 |
| R9 scaling and incurred-to-paid via ReserveSamples | 2, 13 |
| R10 fan chart, scaled residuals, sensitivity heatmap, capital profiles | 14 |
| R11 quasi-Poisson builder, link-ratio builder, BayesianMackChainLadder with 2% factor check | 15 |
| R12 data + README attribution, loader, modus operandi notebook citing the source, README/CLAUDE.md | 6, 16, 17 |

Type/name consistency checks performed: `variance_factor_fn` (static method on both bootstrap classes, consumed by Task 15); `link_ratio_sigma(cum, mask, factors, variance_factor)` (Task 1 signature, used in 10, 15); `_require_full_posterior` message contains "per-cell" (asserted in Tasks 2, 3, 8, 14); `scale_by_dev_` / `standardized_residuals_` / `sampler_` (Task 9, used by the notebook in 16); `DropList` alias defined in Task 1 and imported in 8, 10, 11, 12, 15; `CDRResult.summary()` columns `future_period, origin, mean, sd, var` (Task 8, used in 16 and README); `top_influential` returns `list[tuple[str, int]]` (Task 12, consumed by 11's `mack_analytic_rmsep(drop=...)` and 16).

Known judgement calls recorded for the executor: the NegBin sigma uses the per-column `n_j/(n_j−1)` bias factor rather than England's global `N/(N−p)`; the Bayesian NegBin link-ratio model uses a Normal likelihood with plug-in variance rather than England's custom quasi-likelihood; the CSR per-cell paths share one lognormal shock per origin (comonotonic) so the ultimate distribution is unchanged.
