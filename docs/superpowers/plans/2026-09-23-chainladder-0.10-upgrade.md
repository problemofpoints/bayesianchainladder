# chainladder 0.10 Upgrade, Calendar Fix, and clrd2025 Back-test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the package to chainladder 0.10.1, fix the calendar-period labelling bug in the Triangle-to-DataFrame layer, adopt the new chainladder features in the standalone reserving script (integer-month triangles, Mack sigma interpolation, lognormal BF/CC apriori, a Barnett-Zehnwirth method), and extend the Meyers-style calibration back-test to the `clrd2025` sample dataset.

**Architecture:** The package change is confined to `bayesianchainladder/utils.py`: one new helper `_triangle_cells` built on chainladder's per-cell `valuation` grid replaces three hand-rolled loops, so `origin`/`dev`/`calendar` labels are derived from chainladder rather than recomputed. The script changes add two options and one method to `scripts/run_stochastic_reserving.py` and thread them through the existing serial and parallel dispatch. The back-test extension adds a data builder that emits the long CSV schema the script already consumes, and parametrizes the existing calibration analysis by dataset. No new package modules.

**Tech Stack:** Python 3.11/3.12, uv, chainladder 0.10.1, pandas 2.3, numpy 2, patsy (via chainladder), scikit-learn (via chainladder), pytest.

**Spec:** Design approved in chat on 2026-09-23; the "Design summary" section below is the authoritative record of it.

## Global Constraints

- `chainladder>=0.10.1` and `numpy>=2.0` are the new floors in `pyproject.toml`; `requires-python` stays `>=3.11,<3.13`.
- Fast test suite (`uv run pytest`) must stay at 100% pass; baseline is 189 passed, 55 skipped. New tests must not run MCMC and must not be marked slow.
- `dev` column semantics are unchanged: development age in months (12, 24, ...).
- `origin` encoding: the year for annual origin grain; otherwise the integer `YYYYMM` of the origin period end month.
- `calendar` encoding: the valuation year when both origin and development grains are annual; otherwise the integer `YYYYMM` of the cell's valuation date.
- Formatting: run `uv run black` and `uv run ruff check` only on `bayesianchainladder/` and `tests/` plus any file you edit. Never run `black .` over `references/` or `scripts/`.
- Commit after every task. End each commit message with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Work from the worktree root `/Users/atroyer/projects/bayesianchainladder/.claude/worktrees/chainladder-package-review-c9655c`. Files under `references/meyers-backtest/cache/` and `*.png` are gitignored; the Meyers cache from the previous run lives in the main checkout at `/Users/atroyer/projects/bayesianchainladder/references/meyers-backtest/cache/` and is copied, never modified.
- Scratch files go in `/private/tmp/claude-501/-Users-atroyer-Projects-bayesianchainladder--claude-worktrees-chainladder-package-review-c9655c/9ae7b296-9eab-45a3-b638-1396430879a9/scratchpad` (referred to below as `$SCRATCH`).

## Design summary (approved)

1. Bump chainladder to 0.10.1 and numpy to 2.0.
2. Calendar bug: `triangle_to_dataframe`, `get_future_dataframe`, `prepare_csr_data` currently compute `calendar = origin + dev - 1` with `dev` in months, so every cell gets a unique label and `C(calendar)` is saturated. Replace with a shared helper that reads chainladder's `Triangle.valuation` grid.
3. Script: `df_to_triangle` passes integer months to the Triangle constructor (chainladder 0.10.1 feature). New `--mack-sigma-interpolation {mack,log-linear}` (default `mack`). Document that BF/CC apriori draws are lognormal in chainladder 0.10.1. New method `bz` (Barnett-Zehnwirth probabilistic trend family with parameter and process uncertainty), with `--bz-formula`.
4. clrd2025 back-test: builder script producing `clrd2025_long.csv`, `clrd2025_actuals.csv`, `clrd2025_groups.csv` for the Meyers-style window (origins 1998-2007, valued 2007, actuals from the square revealed by 2016). Eligibility: complete 10x10 paid and incurred, positive net earned premium each year, all paid and case-incurred cells positive, max/min premium ratio at most 5 (flag, 0 disables). Calibration script parametrized by dataset.
5. Re-run the Meyers 200-triangle sweep on 0.10.1 and refresh README tables.

## Reference facts for implementers

- `tri.valuation` is a `DatetimeIndex` of length `n_origin * n_dev` covering the full grid including future cells. It is stored column-major, so reshape with `np.asarray(tri.valuation).reshape((n_origin, n_dev), order="F")`. Verified on `raa`, `quarterly`, `prism_oqdm`, `prism_osds`.
- `tri.origin` is a `PeriodIndex`; `p.year` and `p.month` give the year and the period's end month (3 for `1995Q1`, 12 for annual).
- `tri.development` is an integer Series of months.
- `tri.origin_grain` and `tri.development_grain` are one of `"Y"`, `"S"`, `"Q"`, `"M"`.
- `cl.load_sample("quarterly")` has shape `(1, 2, 12, 45)`, annual origins 1995-2006, quarterly development 3..135 months; use `["paid"]` to select a single column. `cl.load_sample("genins")` has strictly positive incrementals; `cl.load_sample("raa")` has one non-positive incremental.
- `cl.BarnettZehnwirth(formula=...)` fits `LinearRegression(fit_intercept=False)` on a patsy design matrix whose `Intercept` column is included. After fit: `model.model_.estimator_ml.named_steps["design_matrix"]` is a `PatsyFormula` with `.transform(df)`; `named_steps["model"].coef_` is the coefficient vector; `model.mse_resid_` is the residual variance (float-like); `model.model_._prep_X_ml(tri.cum_to_incr().log())` returns the observed-cell frame with columns `origin` (float code), `development` (months), `valuation` (float code), and the value column; `model.model_.origin_encoder_` maps origin start Timestamp to float code 0..n-1 in sorted order. Fitting a triangle with a non-positive incremental raises `ValueError: sample_weight.shape == (54,), expected (55,)`.
- `cl.load_sample("clrd2025")` has shape `(768, 6, 19, 19)`, index labels `["GRNAME", "LOB"]`, columns `IncurredLosses, CumPaidLoss, BulkLoss, EarnedPremDIR, EarnedPremCeded, EarnedPremNet`, origins 1998-2016, development 12..228 months. Boolean slicing `tri[tri.origin <= "2007"][tri.development <= 120]` gives `(768, 6, 10, 10)`. LOB values: othliab, comauto, ppauto, wkcomp, prodliab, medmal. Measured eligibility: 376 complete, 328 after positivity filters, 240 with premium ratio at most 5.
- One triangle through 7 methods at 5000 sims takes about 9 seconds including interpreter start-up.

---

### Task 1: Dependency bump

**Files:**
- Modify: `pyproject.toml:20-30`
- Modify: `uv.lock` (regenerated)

**Interfaces:**
- Produces: environment with `chainladder==0.10.1` for all later tasks.

- [ ] **Step 1: Edit the dependency floors**

In `pyproject.toml`, change these two lines in the `dependencies` list:

```toml
    "numpy>=2.0",
    "chainladder>=0.10.1",
```

(replacing `"numpy>=1.24.0"` and `"chainladder>=0.8.0"`).

- [ ] **Step 2: Regenerate the lock and sync**

```bash
uv lock --upgrade-package chainladder && uv sync
```

Expected: output includes `Update chainladder v0.9.1 -> v0.10.1`. Run `uv run python -c "import chainladder, numpy; print(chainladder.__version__, numpy.__version__)"` and confirm `0.10.1` and a `2.x` numpy.

- [ ] **Step 3: Run the fast suite**

```bash
uv run pytest -q -p no:cacheprovider 2>&1 | tail -1
```

Expected: `189 passed, 55 skipped`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: require chainladder>=0.10.1 and numpy>=2.0

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Fix calendar labelling with a chainladder-backed cell helper

**Files:**
- Modify: `bayesianchainladder/utils.py:19-234` (`triangle_to_dataframe`, `_extract_period_value`, `_find_matching_period`, `get_future_dataframe`) and `bayesianchainladder/utils.py:466-598` (`prepare_csr_data`)
- Modify: `tests/test_utils.py:63-73` (replace `test_calendar_period_calculation`) and add a new test class
- Modify: `CLAUDE.md` (gotchas section)

**Interfaces:**
- Produces: `_triangle_cells(triangle) -> pd.DataFrame` with columns `origin` (int), `dev` (int months), `calendar` (int), `value` (float, NaN when unobserved), `observed` (bool); one row per origin x development cell in row-major order. Public function signatures are unchanged.

- [ ] **Step 1: Replace the wrong calendar test and add encoding tests**

In `tests/test_utils.py`, delete `test_calendar_period_calculation` (lines 63-73) and add this method in its place inside `TestTriangleToDataframe`:

```python
    def test_calendar_is_valuation_year_for_annual_triangle(self, sample_triangle):
        """Cells on one diagonal share a calendar label; RAA has 10 diagonals."""
        df = triangle_to_dataframe(sample_triangle)

        assert sorted(df["calendar"].unique()) == list(range(1981, 1991))
        # Annual origin and annual development: calendar = origin + dev/12 - 1
        expected = df["origin"] + df["dev"] // 12 - 1
        pd.testing.assert_series_equal(
            df["calendar"].astype(int), expected.astype(int), check_names=False
        )
        # Same diagonal, different cells
        c1 = df.loc[(df["origin"] == 1981) & (df["dev"] == 24), "calendar"].iloc[0]
        c2 = df.loc[(df["origin"] == 1982) & (df["dev"] == 12), "calendar"].iloc[0]
        assert c1 == c2 == 1982
```

Then append this new class at the end of the file:

```python
class TestPeriodEncoding:
    """Origin and calendar encodings across grains."""

    def test_future_calendar_labels_follow_observed_ones(self):
        tri = cl.load_sample("raa")
        observed, future = prepare_model_data(tri)

        assert not future["calendar"].isin(observed["calendar"]).any()
        assert future["calendar"].min() == observed["calendar"].max() + 1
        assert len(observed) + len(future) == 100

    def test_annual_origin_quarterly_development(self):
        tri = cl.load_sample("quarterly")["paid"]
        df = triangle_to_dataframe(tri)

        assert df["dev"].min() == 3
        assert df["origin"].min() == 1995  # annual origins stay as years
        assert df["calendar"].min() == 199503  # sub-annual valuation -> YYYYMM
        c1 = df.loc[(df["origin"] == 1995) & (df["dev"] == 15), "calendar"].iloc[0]
        c2 = df.loc[(df["origin"] == 1996) & (df["dev"] == 3), "calendar"].iloc[0]
        assert c1 == c2 == 199603
        vd = tri.valuation_date
        assert df["calendar"].max() == vd.year * 100 + vd.month

    def test_calendar_matches_chainladder_valuation(self):
        tri = cl.load_sample("quarterly")["paid"]
        df = triangle_to_dataframe(tri)
        frame = tri.to_frame(
            keepdims=True, implicit_axis=True, origin_as_datetime=False
        ).reset_index(drop=True)
        val = pd.to_datetime(frame["valuation"])
        frame = pd.DataFrame(
            {
                "origin": frame["origin"].dt.year.astype(int),
                "dev": frame["development"].astype(int),
                "expected": (val.dt.year * 100 + val.dt.month).astype(int),
            }
        )
        merged = df.merge(frame, on=["origin", "dev"], how="inner")
        assert len(merged) == len(df) == len(frame)
        assert (merged["calendar"] == merged["expected"]).all()

    def test_quarterly_origin_grain_gets_unique_origin_labels(self):
        data = pd.DataFrame(
            {
                "origin": ["2020-01-01", "2020-01-01", "2020-04-01", "2020-04-01", "2020-07-01"],
                "valuation": ["2020-03-31", "2020-06-30", "2020-06-30", "2020-09-30", "2020-09-30"],
                "paid": [10.0, 15.0, 12.0, 18.0, 11.0],
            }
        )
        tri = cl.Triangle(
            data, origin="origin", development="valuation", columns="paid", cumulative=True
        )
        assert tri.origin_grain == "Q"
        df = triangle_to_dataframe(tri).sort_values(["origin", "dev"]).reset_index(drop=True)

        assert df["origin"].tolist() == [202003, 202003, 202006, 202006, 202009]
        assert df["dev"].tolist() == [3, 6, 3, 6, 3]
        assert df["calendar"].tolist() == [202003, 202006, 202006, 202009, 202009]

    def test_csr_data_shares_encoding(self):
        tri = cl.load_sample("genins")
        observed, future = prepare_csr_data(tri, premium_value=1.0)
        glm_observed, glm_future = prepare_model_data(tri)

        pd.testing.assert_frame_equal(
            observed[["origin", "dev"]].reset_index(drop=True),
            glm_observed[["origin", "dev"]].reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            future[["origin", "dev"]].reset_index(drop=True),
            glm_future[["origin", "dev"]].reset_index(drop=True),
        )
```

- [ ] **Step 2: Run the new tests to verify they fail**

```bash
uv run pytest tests/test_utils.py -q -p no:cacheprovider -k "calendar or Encoding" 2>&1 | tail -15
```

Expected: `test_calendar_is_valuation_year_for_annual_triangle`, `test_future_calendar_labels_follow_observed_ones`, `test_annual_origin_quarterly_development`, `test_calendar_matches_chainladder_valuation`, `test_quarterly_origin_grain_gets_unique_origin_labels` FAIL (calendar values like 1992). `test_csr_data_shares_encoding` may pass already.

- [ ] **Step 3: Add the helper and rewrite the three converters**

In `bayesianchainladder/utils.py`, replace everything from the start of `def triangle_to_dataframe(` (line 19) through the end of `get_future_dataframe` (line 234) with:

```python
def _encode_period_end(ts: pd.Timestamp, annual: bool) -> int:
    """Integer label for a period ending at ``ts``: the year, or ``YYYYMM``."""
    if annual:
        return int(ts.year)
    return int(ts.year) * 100 + int(ts.month)


def _triangle_cells(triangle: cl.Triangle) -> pd.DataFrame:
    """
    Long-format view of every origin x development cell of a single triangle.

    Labels come from chainladder itself rather than being recomputed:

    - ``origin``: the origin period's year when ``origin_grain == "Y"``,
      otherwise ``YYYYMM`` of the origin period's end month.
    - ``dev``: development age in months (``triangle.development``).
    - ``calendar``: the cell's valuation date from ``Triangle.valuation``,
      encoded as the year when both grains are annual, otherwise ``YYYYMM``.
      Cells on one diagonal share a label; future cells get later labels.
    - ``value``: the cell value (NaN for unobserved / future cells).
    - ``observed``: ``True`` where ``value`` is finite.

    Rows are in row-major (origin, development) order.

    Raises
    ------
    ValueError
        If the triangle has more than one index or column.
    """
    tri = triangle.copy()
    if tri.shape[0] != 1 or tri.shape[1] != 1:
        raise ValueError(
            "Triangle must have a single index and a single column "
            f"(got shape {tri.shape}); slice it first, e.g. tri['paid'] or tri.sum()."
        )

    n_origin, n_dev = len(tri.origin), len(tri.development)
    origin_annual = tri.origin_grain == "Y"
    calendar_annual = origin_annual and tri.development_grain == "Y"

    # Triangle.valuation covers the full grid (future cells included) and is
    # stored column-major, hence order="F".
    valuation = np.asarray(tri.valuation).reshape((n_origin, n_dev), order="F")
    values = np.asarray(tri.values, dtype=float)[0, 0]

    origin_codes = np.array(
        [
            int(p.year) if origin_annual else int(p.year) * 100 + int(p.month)
            for p in tri.origin
        ],
        dtype=int,
    )
    dev_ages = np.asarray(tri.development, dtype=int)

    df = pd.DataFrame(
        {
            "origin": np.repeat(origin_codes, n_dev),
            "dev": np.tile(dev_ages, n_origin),
            "calendar": [
                _encode_period_end(pd.Timestamp(v), calendar_annual)
                for v in valuation.ravel()
            ],
            "value": values.ravel(),
        }
    )
    df["observed"] = df["value"].notna()
    return df


def triangle_to_dataframe(
    triangle: cl.Triangle,
    value_column: str = "incremental",
    include_cumulative: bool = False,
) -> pd.DataFrame:
    """
    Convert a chainladder Triangle to a long-format DataFrame of observed cells.

    Parameters
    ----------
    triangle : chainladder.Triangle
        A single-index, single-column Triangle. Can be cumulative or incremental.
    value_column : str, optional
        Name for the incremental value column. Default is "incremental".
    include_cumulative : bool, optional
        If True, include a "cumulative" column as well. Default is False.

    Returns
    -------
    pd.DataFrame
        One row per observed cell with columns:

        - origin: origin period (year for annual grain, else ``YYYYMM``)
        - dev: development age in months
        - calendar: valuation period of the cell (year when both grains are
          annual, else ``YYYYMM``); cells on one diagonal share a label
        - incremental (or ``value_column``): the incremental value
        - cumulative (optional)

    Examples
    --------
    >>> import chainladder as cl
    >>> from bayesianchainladder.utils import triangle_to_dataframe
    >>> tri = cl.load_sample("raa")
    >>> df = triangle_to_dataframe(tri)
    >>> df.head()
    """
    tri = triangle.copy()
    incremental = tri.cum_to_incr() if tri.is_cumulative else tri
    cells = _triangle_cells(incremental)
    observed = cells["observed"].to_numpy()

    df = (
        cells.loc[observed, ["origin", "dev", "calendar", "value"]]
        .rename(columns={"value": value_column})
        .reset_index(drop=True)
    )

    if include_cumulative:
        cumulative = tri if tri.is_cumulative else tri.incr_to_cum()
        df["cumulative"] = _triangle_cells(cumulative).loc[observed, "value"].to_numpy()

    return df


def get_future_dataframe(
    triangle: cl.Triangle,
    value_column: str = "incremental",
) -> pd.DataFrame:
    """
    Create a DataFrame of the future (unobserved) cells of a triangle.

    Parameters
    ----------
    triangle : chainladder.Triangle
        A single-index, single-column Triangle.
    value_column : str, optional
        Name for the value column (NaN for every row). Default is "incremental".

    Returns
    -------
    pd.DataFrame
        Columns ``origin``, ``dev``, ``calendar`` (same encoding as
        :func:`triangle_to_dataframe`) and ``value_column`` (all NaN).
    """
    cells = _triangle_cells(triangle)
    df = cells.loc[~cells["observed"], ["origin", "dev", "calendar"]].reset_index(drop=True)
    df[value_column] = np.nan
    return df
```

Then in `prepare_csr_data`, replace the block from `tri = triangle.copy()` (line 517) through the line `future_df = pd.DataFrame(future_rows)` (line 564) with:

```python
    tri = triangle.copy()

    # Ensure cumulative
    if not tri.is_cumulative:
        tri = tri.incr_to_cum()

    cells = _triangle_cells(tri)

    # Observed cells: only positive values can be log-transformed
    observed_mask = cells["observed"] & (cells["value"] > 0)
    observed_df = (
        cells.loc[observed_mask, ["origin", "dev", "value"]]
        .rename(columns={"value": "cumulative"})
        .reset_index(drop=True)
    )
    observed_df["logloss"] = np.log(observed_df["cumulative"])

    future_df = cells.loc[~cells["observed"], ["origin", "dev"]].reset_index(drop=True)
    future_df["cumulative"] = np.nan
    future_df["logloss"] = np.nan
```

Keep the premium and dtype code that follows unchanged. Finally confirm the removed helpers have no other callers:

```bash
grep -rn "_extract_period_value\|_find_matching_period" bayesianchainladder tests scripts references --include=*.py
```

Expected: no output.

- [ ] **Step 4: Run the utils tests**

```bash
uv run pytest tests/test_utils.py -q -p no:cacheprovider 2>&1 | tail -3
```

Expected: all pass (the previous 28 plus 6 new).

- [ ] **Step 5: Run the full fast suite, lint, format**

```bash
uv run pytest -q -p no:cacheprovider 2>&1 | tail -1
uv run ruff check bayesianchainladder tests && uv run black bayesianchainladder/utils.py tests/test_utils.py
```

Expected: `195 passed, 55 skipped`; ruff clean.

- [ ] **Step 6: Document the encoding in CLAUDE.md**

Add this bullet to the "Non-obvious gotchas" list in `CLAUDE.md`, directly after the "Formula-aware categorical encoding" bullet:

```markdown
- **Period encoding comes from chainladder, not arithmetic** ([utils.py:_triangle_cells](bayesianchainladder/utils.py)). `origin` is the year for annual origin grain and `YYYYMM` otherwise; `dev` is the development age in months; `calendar` is the cell's valuation date from `Triangle.valuation` (year when both grains are annual, else `YYYYMM`). Cells on one diagonal share a `calendar` label and future cells get later labels — this is what makes `C(calendar)` and `(1 | calendar)` identifiable. Do not reintroduce `origin + dev - 1`: with `dev` in months that gave every cell a unique label.
```

- [ ] **Step 7: Commit**

```bash
git add bayesianchainladder/utils.py tests/test_utils.py CLAUDE.md
git commit -m "fix: derive origin/calendar labels from chainladder valuation grid

calendar was origin + dev - 1 with dev in months, so every cell got a
unique label and C(calendar) was saturated. Encode origin/calendar from
Triangle.valuation via a shared _triangle_cells helper; sub-annual grains
use YYYYMM labels.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Script — integer-month triangles, Mack sigma interpolation, lognormal apriori docs

**Files:**
- Modify: `scripts/run_stochastic_reserving.py` (`df_to_triangle` ~721-765, `_run_mack` ~840-858, `_parametric_bootstrap_and_aggregate` docstring ~926-932, `run_methods_on_triangle`, `iterate_triangles`, `_run_single_group`, `iterate_triangles_parallel`, `parse_args`, `main`, module docstring)
- Modify: `scripts/README.md` (methods table, defaults table, apriori note)
- Create: `tests/test_reserving_script.py`

**Interfaces:**
- Produces: `_run_mack(loss_tri, n_samples=5000, random_seed=None, sigma_interpolation="mack")`; keyword `mack_sigma_interpolation: str = "mack"` on `run_methods_on_triangle`, `iterate_triangles`, `iterate_triangles_parallel`; `_run_single_group` tuple gains `mack_sigma_interpolation` as its 13th element; CLI flag `--mack-sigma-interpolation`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reserving_script.py`:

```python
"""Fast tests for the standalone reserving script (no MCMC)."""

from __future__ import annotations

import importlib.util
import pathlib

import chainladder as cl
import numpy as np
import pandas as pd
import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_stochastic_reserving.py"
EXAMPLE = SCRIPT.parent / "example_input.csv"


@pytest.fixture(scope="module")
def rs():
    spec = importlib.util.spec_from_file_location("run_stochastic_reserving", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example_df():
    return pd.read_csv(EXAMPLE)


class TestDfToTriangle:
    def test_matches_date_based_construction(self, rs, example_df):
        tri = rs.df_to_triangle(example_df, value_col="paid")

        work = example_df[["origin", "dev", "paid"]].copy()
        eval_year = work["origin"].astype(int) + work["dev"].astype(int) // 12 - 1
        work["dev_date"] = pd.to_datetime(eval_year.astype(str) + "-12-31")
        expected = cl.Triangle(
            data=work, origin="origin", development="dev_date",
            columns=["paid"], cumulative=True, origin_format="%Y",
        )

        assert tri.shape == expected.shape == (1, 1, 10, 10)
        assert tri.development.tolist() == list(range(12, 121, 12))
        assert tri.valuation_date.year == expected.valuation_date.year == 2010
        np.testing.assert_allclose(np.nan_to_num(tri.values), np.nan_to_num(expected.values))
        assert (np.isnan(tri.values) == np.isnan(expected.values)).all()


class TestMack:
    @pytest.mark.parametrize("interp", ["mack", "log-linear"])
    def test_sigma_interpolation_options_run(self, rs, example_df, interp):
        tri = rs.df_to_triangle(example_df, value_col="paid")
        samples, ibnr = rs._run_mack(
            tri, n_samples=200, random_seed=1, sigma_interpolation=interp
        )
        assert samples.shape == (10, 200)
        assert np.isfinite(samples).all()
        assert ibnr.shape == (10,)

    def test_cli_default_is_mack(self, rs):
        args = rs.parse_args(["--input", "x.csv"])
        assert args.mack_sigma_interpolation == "mack"
```

- [ ] **Step 2: Run them to verify the failures**

```bash
uv run pytest tests/test_reserving_script.py -q -p no:cacheprovider 2>&1 | tail -8
```

Expected: `TestDfToTriangle` passes already (both constructions are equivalent), `test_sigma_interpolation_options_run` FAILS with `TypeError: _run_mack() got an unexpected keyword argument 'sigma_interpolation'`, `test_cli_default_is_mack` FAILS with `AttributeError`.

- [ ] **Step 3: Simplify `df_to_triangle`**

Replace the body of `df_to_triangle` (keep the signature) with:

```python
    """Convert a long-format DataFrame into a chainladder Triangle.

    Development is expected as elapsed months (12, 24, 36, ...). chainladder
    (>= 0.10.1) accepts an integer development column as an age in months
    measured from the start of each origin period, so no date conversion is
    needed.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns ``origin`` (int year), ``dev`` (int months), and
        ``value_col`` (numeric cumulative losses).
    value_col : str
        Column containing the cumulative losses.
    origin_col, dev_col : str
        Names for origin and development columns.

    Returns
    -------
    chainladder.Triangle
    """
    work = df[[origin_col, dev_col, value_col]].copy()
    work.columns = ["origin", "dev", value_col]
    work["origin"] = work["origin"].astype(int)
    work["dev"] = work["dev"].astype(int)

    tri = cl.Triangle(
        data=work,
        origin="origin",
        development="dev",
        columns=[value_col],
        cumulative=True,
        origin_format="%Y",
    )
    return tri
```

- [ ] **Step 4: Add the Mack option**

Change `_run_mack` to:

```python
def _run_mack(loss_tri, n_samples=5000, random_seed=None, sigma_interpolation="mack"):
    """Run Mack Chain Ladder. Returns per-origin IBNR samples (n_origin, n_sims).

    ``sigma_interpolation="mack"`` reproduces the tail-sigma rule of Mack (1994);
    ``"log-linear"`` is chainladder's historical default.
    """
    dev = cl.Development(
        n_periods=-1, sigma_interpolation=sigma_interpolation
    ).fit_transform(loss_tri)
    mack = cl.MackChainladder().fit(dev)
```

(the rest of the function is unchanged). Then thread the option:

1. `run_methods_on_triangle(...)`: add parameter `mack_sigma_interpolation="mack",` after `process_variance="odp",` and pass it: `per_origin_sim, _ = _run_mack(loss_tri, n_samples=n_sims, random_seed=random_seed, sigma_interpolation=mack_sigma_interpolation)`. Add to the docstring: `mack_sigma_interpolation : {'mack', 'log-linear'}  Tail sigma rule for Mack (see _run_mack).`
2. `iterate_triangles(...)`: add parameter `mack_sigma_interpolation="mack",` and forward `mack_sigma_interpolation=mack_sigma_interpolation,` in its `run_methods_on_triangle(...)` call.
3. `_run_single_group(args)`: change the unpack line to
   ```python
   (lob, group_id), sub_df, methods, loss_cols, n_sims, rho, apriori, apriori_sigma, random_seed, collect_samples, residual_dist, process_variance, mack_sigma_interpolation = args
   ```
   and forward `mack_sigma_interpolation=mack_sigma_interpolation,` in its `run_methods_on_triangle(...)` call.
4. `iterate_triangles_parallel(...)`: add parameter `mack_sigma_interpolation="mack",` and append `mack_sigma_interpolation` as the last element of the task tuple that currently ends with `process_variance)`.
5. `parse_args`: add after the `--process-variance` argument:
   ```python
    p.add_argument(
        "--mack-sigma-interpolation",
        default="mack",
        choices=["mack", "log-linear"],
        dest="mack_sigma_interpolation",
        help=(
            "Tail sigma extrapolation for the mack method. 'mack' (default) uses the "
            "Mack (1994) rule via chainladder's Development(sigma_interpolation='mack'); "
            "'log-linear' is chainladder's historical default and the pre-0.10 behaviour."
        ),
    )
   ```
6. `main`: pass `mack_sigma_interpolation=args.mack_sigma_interpolation,` to both `iterate_triangles_parallel` and `iterate_triangles` calls, and add `| mack_sigma_interpolation=%s` with `args.mack_sigma_interpolation` to the "Methods:" log line.

- [ ] **Step 5: Update the apriori docs in the script**

In `_parametric_bootstrap_and_aggregate`'s docstring replace the `apriori_sigma` paragraph with:

```
    apriori_sigma : float
        Standard deviation of the a-priori loss ratio.  When > 0, each
        bootstrap sample draws its own apriori.  chainladder >= 0.10.1 draws
        from a lognormal matched by method of moments to mean ``apriori``
        (BF) or the Cape Cod estimate (CC) and sd ``apriori_sigma``, so draws
        are strictly positive; earlier versions used a Normal, which could
        produce negative expected ultimates.  Default 0.15.  Set to 0 to
        recover the deterministic-apriori behaviour (variance collapse).
```

In the module docstring's `mack` entry change the line to:

```
mack      : Mack Chain Ladder (normal approximation per Mack 1993). Tail sigma via
              --mack-sigma-interpolation (default 'mack' = Mack 1994 rule).
```

- [ ] **Step 6: Run the script tests and the example input**

```bash
uv run pytest tests/test_reserving_script.py -q -p no:cacheprovider 2>&1 | tail -3
uv run python scripts/run_stochastic_reserving.py --input scripts/example_input.csv --output $SCRATCH/example_task3.csv --loss-col both --n-sims 500 2>&1 | grep -v Warning | tail -20
```

Expected: tests pass; the run prints 7 methods x 2 loss types of totals and no `failed` lines.

- [ ] **Step 7: Update `scripts/README.md`**

1. In the methods table, change the `mack` row description to: `Mack Chain Ladder (Mack 1993) — normal approximation; tail sigma per Mack (1994) by default (\`--mack-sigma-interpolation\`)`.
2. In the Defaults table add a row: `| \`--mack-sigma-interpolation\` | \`mack\` | Mack (1994) tail-sigma rule; \`log-linear\` restores the pre-0.10 chainladder default |`.
3. Replace the sentence in the `--apriori-sigma` note that reads `so each simulation samples its own apriori from Normal(apriori, 0.15) (BF) or Normal(cc_apriori, 0.15) (CC)` with `so each simulation samples its own apriori from a lognormal with mean apriori (BF) or the Cape Cod estimate (CC) and standard deviation 0.15 (chainladder >= 0.10.1; earlier versions drew from a Normal and could produce negative aprioris)`.
4. Under Input Format, change the `dev` row description to `Development age in months (12, 24, 36, …); passed to chainladder directly as an age from the origin period start`.

- [ ] **Step 8: Commit**

```bash
git add scripts/run_stochastic_reserving.py scripts/README.md tests/test_reserving_script.py
git commit -m "feat(script): integer-month triangles, --mack-sigma-interpolation, lognormal apriori docs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Script — Barnett-Zehnwirth `bz` method

**Files:**
- Modify: `scripts/run_stochastic_reserving.py` (new `_run_bz` after `_run_mack`; dispatch in `run_methods_on_triangle`; `bz_formula` plumbing mirroring Task 3; `parse_args` choices/default/`--bz-formula`; module docstring)
- Modify: `scripts/README.md` (methods table, "when to use", usage example)
- Modify: `tests/test_reserving_script.py`

**Interfaces:**
- Produces: `_run_bz(loss_tri, n_sims=5000, random_seed=None, formula="C(origin)+C(development)") -> np.ndarray` of shape `(n_origin, n_sims)`; keyword `bz_formula: str = "C(origin)+C(development)"` on `run_methods_on_triangle`, `iterate_triangles`, `iterate_triangles_parallel`; `_run_single_group` tuple gains `bz_formula` as its 14th element; method name `"bz"` accepted by `--methods` and included in the default method list; CLI flag `--bz-formula`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_reserving_script.py`:

```python
class TestBarnettZehnwirth:
    def test_returns_positive_ibnr_samples_close_to_chain_ladder(self, rs):
        tri = cl.load_sample("genins")
        samples = rs._run_bz(tri, n_sims=400, random_seed=7)

        assert samples.shape == (10, 400)
        assert np.isfinite(samples).all()
        assert (samples >= 0).all()
        assert (samples[0] == 0).all()  # first origin is fully developed
        assert (samples[1:] > 0).all()

        cl_ult = cl.Chainladder().fit(cl.Development().fit_transform(tri)).ultimate_
        cl_ibnr = float(np.nansum(cl_ult.values)) - float(np.nansum(tri.latest_diagonal.values))
        bz_ibnr = float(samples.sum(axis=0).mean())
        assert abs(bz_ibnr - cl_ibnr) / cl_ibnr < 0.25

    def test_rejects_non_positive_incrementals(self, rs):
        with pytest.raises(ValueError, match="positive incremental"):
            rs._run_bz(cl.load_sample("raa"), n_sims=10, random_seed=0)

    def test_cli_accepts_bz(self, rs):
        args = rs.parse_args(["--input", "x.csv", "--methods", "bz"])
        assert args.methods == ["bz"]
        assert args.bz_formula == "C(origin)+C(development)"
        assert "bz" in rs.parse_args(["--input", "x.csv"]).methods
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/test_reserving_script.py -q -p no:cacheprovider -k Barnett 2>&1 | tail -6
```

Expected: 3 failures (`AttributeError: module has no attribute '_run_bz'`, argparse `invalid choice: 'bz'`).

- [ ] **Step 3: Implement `_run_bz`**

Insert after `_run_mack`:

```python
def _run_bz(loss_tri, n_sims=5000, random_seed=None, formula="C(origin)+C(development)"):
    """Barnett-Zehnwirth probabilistic trend family forecast.

    Fits ``cl.BarnettZehnwirth`` — ordinary least squares on log incremental
    losses with a patsy ``formula`` over ``origin`` (period code) and
    ``development`` (age in months) — then simulates IBNR by drawing the
    regression coefficients from their large-sample normal approximation
    ``N(beta, sigma^2 (X'X)^-1)`` and adding lognormal process noise with the
    residual variance ``sigma^2``.  This is the frequentist analogue of the
    cross-classified log-link GLM.

    Returns per-origin IBNR samples, shape (n_origin, n_sims).  Origins with no
    future cells get zeros.

    Raises
    ------
    ValueError
        If any observed incremental is non-positive (the log-linear model is
        undefined there; chainladder itself fails on such triangles).
    """
    incr = np.asarray(loss_tri.cum_to_incr().values, dtype=float)[0, 0]
    observed = np.isfinite(incr)
    if (incr[observed] <= 0).any():
        raise ValueError(
            "bz requires strictly positive incremental losses (log-linear model); "
            f"found {int((incr[observed] <= 0).sum())} non-positive cell(s)"
        )

    model = cl.BarnettZehnwirth(formula=formula).fit(loss_tri)
    pipeline = model.model_.estimator_ml
    design = pipeline.named_steps["design_matrix"]
    ols = pipeline.named_steps["model"]
    beta = np.asarray(ols.coef_, dtype=float)
    sigma2 = float(np.asarray(model.mse_resid_, dtype=float).ravel()[0])

    # Observed-cell design matrix -> coefficient covariance sigma^2 (X'X)^-1
    x_obs = model.model_._prep_X_ml(loss_tri.cum_to_incr().log())
    d_obs = np.asarray(design.transform(x_obs), dtype=float)
    cov = sigma2 * np.linalg.pinv(d_obs.T @ d_obs)

    # Future-cell design matrix using the same origin codes as the fit
    # (origin_encoder_ assigns 0..n-1 to the sorted origin start dates).
    origin_codes = np.array(
        [code for _, code in sorted(model.model_.origin_encoder_.items())], dtype=float
    )
    dev_ages = np.asarray(loss_tri.development, dtype=int)
    n_origin = incr.shape[0]
    oi, di = np.where(~observed)
    if len(oi) == 0:
        return np.zeros((n_origin, n_sims))
    x_fut = pd.DataFrame({"origin": origin_codes[oi], "development": dev_ages[di]})
    d_fut = np.asarray(design.transform(x_fut), dtype=float)

    rng = np.random.default_rng(random_seed)
    beta_draws = rng.multivariate_normal(beta, cov, size=n_sims, method="svd")  # (n_sims, p)
    log_mu = d_fut @ beta_draws.T  # (n_future, n_sims)
    log_mu = log_mu + rng.normal(0.0, np.sqrt(sigma2), size=log_mu.shape)
    future_incr = np.exp(log_mu)

    per_origin = np.zeros((n_origin, n_sims))
    np.add.at(per_origin, oi, future_incr)
    return per_origin
```

- [ ] **Step 4: Wire the method and its formula option**

1. `run_methods_on_triangle`: add parameter `bz_formula="C(origin)+C(development)",` after `mack_sigma_interpolation="mack",`; add a dispatch branch before the `else: log.warning("Unknown method...` branch:
   ```python
            elif method == "bz":
                per_origin_sim = _run_bz(
                    loss_tri, n_sims=n_sims, random_seed=random_seed, formula=bz_formula
                )
   ```
   Update the docstring `methods` entry to include `"bz"` and add `bz_formula : str  patsy formula for the bz method over origin/development.`
2. `iterate_triangles`: add `bz_formula="C(origin)+C(development)",` and forward it.
3. `_run_single_group`: unpack a 14th element `bz_formula` and forward it.
4. `iterate_triangles_parallel`: add the parameter and append `bz_formula` to the task tuple after `mack_sigma_interpolation`.
5. `parse_args`: add `"bz"` to the `--methods` `default` list (at the end) and to `choices`; extend the help text with `bz is the Barnett-Zehnwirth log-linear trend model (parameter + lognormal process uncertainty; skips triangles with non-positive incrementals).` Add:
   ```python
    p.add_argument(
        "--bz-formula",
        default="C(origin)+C(development)",
        dest="bz_formula",
        help=(
            "patsy formula for the bz method. Columns available: origin (integer "
            "period code) and development (age in months). Default is the "
            "cross-classified model C(origin)+C(development)."
        ),
    )
   ```
6. `main`: forward `bz_formula=args.bz_formula,` to both iterate calls.
7. Module docstring: add under Methods:
   ```
   bz        : Barnett-Zehnwirth probabilistic trend family (cl.BarnettZehnwirth).
                 OLS on log incrementals with --bz-formula (default C(origin)+C(development));
                 IBNR simulated from the coefficient normal approximation plus lognormal
                 process noise. Requires strictly positive incrementals — triangles with
                 negative development are skipped with an error log line.
   ```

- [ ] **Step 5: Run tests and a script smoke run**

```bash
uv run pytest tests/test_reserving_script.py -q -p no:cacheprovider 2>&1 | tail -3
uv run python scripts/run_stochastic_reserving.py --input scripts/example_input.csv --output $SCRATCH/example_task4.csv --loss-col both --methods mack bz --n-sims 500 2>&1 | grep -v Warning | tail -8
```

Expected: all tests pass; the run prints `method=bz` totals for `paid` (and either a total or a single logged `bz ... failed: bz requires strictly positive incremental losses` for `case_incurred` if the example has a negative case incremental — both are acceptable).

- [ ] **Step 6: Update `scripts/README.md`**

Add to the methods table: `| \`bz\` | Barnett-Zehnwirth probabilistic trend family (\`cl.BarnettZehnwirth\`) — OLS on log incrementals, coefficient-normal + lognormal process simulation; needs strictly positive incrementals | |`. Add to "When to use": `| \`bz\` | Frequentist analogue of the Bayesian log-link GLM; positive-incremental paid triangles; want a regression-based benchmark with explicit origin/development structure (\`--bz-formula\`) |`. Add `bz` to the `--methods` list in the Usage example.

- [ ] **Step 7: Commit**

```bash
git add scripts/run_stochastic_reserving.py scripts/README.md tests/test_reserving_script.py
git commit -m "feat(script): add Barnett-Zehnwirth (bz) stochastic method

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: clrd2025 back-test data builder

**Files:**
- Create: `references/meyers-backtest/24_build_clrd2025_long.py`

**Interfaces:**
- Produces (in `references/meyers-backtest/cache/`): `clrd2025_long.csv` with columns `lob, group_id, origin, dev, paid, case_incurred, premium` (training upper triangle only, 55 rows per group); `clrd2025_actuals.csv` with columns `lob, group_id, loss_type, actual_ultimate_total`; `clrd2025_groups.csv` with columns `group_id, grname, lob, premium_ratio`. `group_id` is a 1-based integer.

- [ ] **Step 1: Write the builder**

Create `references/meyers-backtest/24_build_clrd2025_long.py`:

```python
"""24_build_clrd2025_long.py
==========================
Build Meyers-style back-test inputs from chainladder's ``clrd2025`` sample.

Window
------
Origins 1998-2007, development 12..120 months.  Training data is the upper
triangle valued at 2007-12-31 (origin_year + dev/12 - 1 <= 2007).  Actual
ultimates are the dev-120 values, fully revealed by the 2016 valuation.  This
mirrors Meyers (2015), who trained on 1988-1997 valued at 1997 and tested on
the square revealed by 2006.

Eligibility (per GRNAME x LOB)
------------------------------
1. Complete 10x10 square for CumPaidLoss and IncurredLosses.
2. EarnedPremNet > 0 in every origin year.
3. Every paid and case-incurred cell > 0 (case incurred = IncurredLosses -
   BulkLoss, with missing BulkLoss treated as 0).
4. max/min EarnedPremNet across the 10 origins <= --max-premium-ratio
   (default 5; pass 0 to disable).  Excludes books with large structural
   changes, in the spirit of Meyers' stable-book screen.

Outputs (references/meyers-backtest/cache/)
-------------------------------------------
clrd2025_long.csv     lob, group_id, origin, dev, paid, case_incurred, premium
clrd2025_actuals.csv  lob, group_id, loss_type, actual_ultimate_total
clrd2025_groups.csv   group_id, grname, lob, premium_ratio

Usage
-----
  uv run python references/meyers-backtest/24_build_clrd2025_long.py [--max-premium-ratio 5]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import chainladder as cl
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CACHE_DIR  # noqa: E402

FIRST_ORIGIN = 1998
LAST_ORIGIN = 2007
N_DEV = 10
LONG_PATH = CACHE_DIR / "clrd2025_long.csv"
ACTUALS_PATH = CACHE_DIR / "clrd2025_actuals.csv"
GROUPS_PATH = CACHE_DIR / "clrd2025_groups.csv"


def load_window() -> cl.Triangle:
    """clrd2025 restricted to origins <= 2007 and development <= 120 months."""
    tri = cl.load_sample("clrd2025")
    tri = tri[tri.origin <= str(LAST_ORIGIN)][tri.development <= 12 * N_DEV]
    assert tri.shape[2:] == (N_DEV, N_DEV), tri.shape
    return tri


def extract_arrays(tri: cl.Triangle) -> dict[str, np.ndarray]:
    """Dense arrays per group: paid/case (n, 10, 10), premium (n, 10)."""
    paid = np.asarray(tri["CumPaidLoss"].values, dtype=float)[:, 0]
    incurred = np.asarray(tri["IncurredLosses"].values, dtype=float)[:, 0]
    bulk = np.nan_to_num(np.asarray(tri["BulkLoss"].values, dtype=float)[:, 0])
    premium = np.asarray(tri["EarnedPremNet"].values, dtype=float)[:, 0, :, 0]
    return {"paid": paid, "incurred": incurred, "case": incurred - bulk, "premium": premium}


def eligibility(arrays: dict[str, np.ndarray], max_premium_ratio: float) -> pd.DataFrame:
    """Per-group boolean filters, cumulative in the listed order."""
    paid, incurred, case, premium = (
        arrays["paid"], arrays["incurred"], arrays["case"], arrays["premium"]
    )
    complete = np.isfinite(paid).all(axis=(1, 2)) & np.isfinite(incurred).all(axis=(1, 2))
    prem_ok = complete & (np.nan_to_num(premium) > 0).all(axis=1)
    positive = (
        prem_ok
        & (np.nan_to_num(paid) > 0).all(axis=(1, 2))
        & (np.nan_to_num(case) > 0).all(axis=(1, 2))
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.nanmax(premium, axis=1) / np.nanmin(premium, axis=1)
    stable = positive & ((ratio <= max_premium_ratio) if max_premium_ratio > 0 else True)
    return pd.DataFrame(
        {"complete": complete, "premium_positive": prem_ok, "losses_positive": positive,
         "stable": stable, "premium_ratio": ratio}
    )


def build(max_premium_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tri = load_window()
    arrays = extract_arrays(tri)
    flags = eligibility(arrays, max_premium_ratio)
    index = tri.index.reset_index(drop=True)

    print("Eligibility funnel (GRNAME x LOB):")
    for col in ["complete", "premium_positive", "losses_positive", "stable"]:
        print(f"  {col:<18} {int(flags[col].sum()):>4} of {len(flags)}")

    keep = np.where(flags["stable"].to_numpy())[0]
    groups = index.iloc[keep].copy()
    groups["_row"] = keep
    groups["premium_ratio"] = flags["premium_ratio"].to_numpy()[keep]
    groups = groups.rename(columns={"GRNAME": "grname", "LOB": "lob"})
    groups = groups.sort_values(["lob", "grname"]).reset_index(drop=True)
    groups.insert(0, "group_id", np.arange(1, len(groups) + 1))

    origins = np.arange(FIRST_ORIGIN, LAST_ORIGIN + 1)
    devs = 12 * np.arange(1, N_DEV + 1)
    k, j = np.meshgrid(np.arange(N_DEV), np.arange(N_DEV), indexing="ij")
    train_mask = (k + j) <= (N_DEV - 1)  # origin_year + dev_years - 1 <= 2007

    long_rows, actual_rows = [], []
    for _, g in groups.iterrows():
        r = int(g["_row"])
        paid, case, prem = arrays["paid"][r], arrays["case"][r], arrays["premium"][r]
        long_rows.append(pd.DataFrame({
            "lob": g["lob"],
            "group_id": int(g["group_id"]),
            "origin": origins[k[train_mask]],
            "dev": devs[j[train_mask]],
            "paid": paid[train_mask],
            "case_incurred": case[train_mask],
            "premium": prem[k[train_mask]],
        }))
        actual_rows.append({"lob": g["lob"], "group_id": int(g["group_id"]),
                            "loss_type": "paid", "actual_ultimate_total": float(paid[:, -1].sum())})
        actual_rows.append({"lob": g["lob"], "group_id": int(g["group_id"]),
                            "loss_type": "case_incurred", "actual_ultimate_total": float(case[:, -1].sum())})

    long_df = pd.concat(long_rows, ignore_index=True).sort_values(
        ["lob", "group_id", "origin", "dev"]).reset_index(drop=True)
    actuals_df = pd.DataFrame(actual_rows)
    groups_df = groups.drop(columns=["_row"])
    return long_df, actuals_df, groups_df


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-premium-ratio", type=float, default=5.0,
                   help="Exclude groups whose max/min net earned premium exceeds this (0 disables).")
    args = p.parse_args(argv)

    warnings.filterwarnings("ignore")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    long_df, actuals_df, groups_df = build(args.max_premium_ratio)

    n_groups = groups_df["group_id"].nunique()
    rows_per_group = long_df.groupby("group_id").size()
    assert (rows_per_group == N_DEV * (N_DEV + 1) // 2).all(), rows_per_group.describe()
    assert long_df[["paid", "case_incurred", "premium"]].gt(0).all().all()
    assert len(actuals_df) == 2 * n_groups

    print(f"\nEligible groups: {n_groups}")
    print(groups_df["lob"].value_counts().to_string())
    print(f"\nLong rows: {len(long_df):,}  (55 per group)")
    print(f"Origins {long_df['origin'].min()}-{long_df['origin'].max()}, dev {long_df['dev'].min()}-{long_df['dev'].max()}")
    print(f"Premium ratio: median {groups_df['premium_ratio'].median():.2f}, max {groups_df['premium_ratio'].max():.2f}")

    long_df.to_csv(LONG_PATH, index=False)
    actuals_df.to_csv(ACTUALS_PATH, index=False)
    groups_df.to_csv(GROUPS_PATH, index=False)
    print(f"\nWrote {LONG_PATH}\n      {ACTUALS_PATH}\n      {GROUPS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it with the default and with the filter disabled**

```bash
uv run python references/meyers-backtest/24_build_clrd2025_long.py
uv run python references/meyers-backtest/24_build_clrd2025_long.py --max-premium-ratio 0 | grep -E "Eligible|stable|losses_positive"
uv run python references/meyers-backtest/24_build_clrd2025_long.py > /dev/null
```

Expected funnel: complete 376, premium_positive 351, losses_positive 328, stable 240 (default) / 328 (ratio 0). Long rows 13,200 for the default. The third command restores the default outputs.

- [ ] **Step 3: Sanity-check the outputs against the reserving script**

```bash
head -3 references/meyers-backtest/cache/clrd2025_long.csv
uv run python - <<'EOF'
import pandas as pd
d = pd.read_csv("references/meyers-backtest/cache/clrd2025_long.csv")
a = pd.read_csv("references/meyers-backtest/cache/clrd2025_actuals.csv")
one = d[d.group_id == 1]
last_diag = one[one.origin + one.dev // 12 - 1 == 2007]
print("group 1 paid-to-date", last_diag.paid.sum(), "actual paid ultimate", a[(a.group_id == 1) & (a.loss_type == "paid")].actual_ultimate_total.iloc[0])
print("groups", d.group_id.nunique(), "lobs", sorted(d.lob.unique()))
EOF
uv run python scripts/run_stochastic_reserving.py --input references/meyers-backtest/cache/clrd2025_long.csv --output $SCRATCH/clrd_smoke.csv --loss-col both --methods mack bz --n-sims 200 --n-jobs 4 2>&1 | grep -c "method="
```

Expected: 10 training diagonals with paid-to-date below the actual ultimate; six LOBs; the smoke run prints a large count of `method=` total lines (240 groups x 2 loss types x up to 2 methods) and completes in about a minute.

- [ ] **Step 4: Commit**

```bash
git add references/meyers-backtest/24_build_clrd2025_long.py
git commit -m "feat(backtest): build Meyers-style inputs from chainladder clrd2025 sample

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Parametrize the final calibration analysis by dataset

**Files:**
- Modify: `references/meyers-backtest/22_final_calibration.py`

**Interfaces:**
- Consumes: `cache/<prefix>.csv` and `cache/<prefix>_samples.parquet` from the reserving script; `cache/clrd2025_actuals.csv` (Task 5) or `cache/meyers_actuals_source.csv` (fallback for Meyers when `reservetestr` is absent).
- Produces: CLI `--dataset {meyers,clrd2025}`, `--prefix PREFIX` (default `meyers_final` / `clrd2025_final`); outputs `cache/<prefix>_calibration.csv`, `cache/<prefix>_cal_detail.csv`, `figures/<prefix>_calibration_grid.png`, `figures/<prefix>_pp_chart.png`; `run_analysis(dataset, prefix)` returns `(summary_df, cal_df)`.

- [ ] **Step 1: Copy the previous Meyers results into the worktree cache (regression baseline)**

```bash
mkdir -p references/meyers-backtest/cache
cp /Users/atroyer/projects/bayesianchainladder/references/meyers-backtest/cache/meyers_final.csv /Users/atroyer/projects/bayesianchainladder/references/meyers-backtest/cache/meyers_final_samples.parquet /Users/atroyer/projects/bayesianchainladder/references/meyers-backtest/cache/meyers_final_cal_detail.csv /Users/atroyer/projects/bayesianchainladder/references/meyers-backtest/cache/meyers_long.csv references/meyers-backtest/cache/
cp references/meyers-backtest/cache/meyers_final_cal_detail.csv references/meyers-backtest/cache/meyers_actuals_source.csv
```

(The last copy preserves the actuals source under a name the re-run in Task 8 will not overwrite.)

- [ ] **Step 2: Refactor the config and loaders**

In `22_final_calibration.py`:

1. Replace the `RESULTS_CSV` / `SAMPLES_PARQUET` constants with:
   ```python
   DATASETS = {
       "meyers": {
           "prefix": "meyers_final",
           "lobs": ["comauto", "ppauto", "wkcomp", "othliab"],
           "title": "Meyers (2015) 200 triangles, origins 1988-1997",
       },
       "clrd2025": {
           "prefix": "clrd2025_final",
           "lobs": ["comauto", "ppauto", "wkcomp", "othliab", "prodliab", "medmal"],
           "title": "clrd2025 Meyers-style window, origins 1998-2007",
       },
   }
   ```
2. Add `"bz"` to the end of `ALL_METHODS` and `"bz": "Barnett-Zehnwirth PTF"` to `METHOD_LABELS`. Add `"prodliab": "Products Liab", "medmal": "Med Mal"` to `LOB_LABELS`. Delete the module-level `LOBS` constant (it moves into `DATASETS`).
3. Replace `load_actual_ultimates()` with:
   ```python
   def load_actual_ultimates(dataset: str) -> pd.DataFrame:
       """DataFrame with lob, group_id, loss_type, actual_ultimate_total."""
       if dataset == "clrd2025":
           path = CACHE_DIR / "clrd2025_actuals.csv"
           if not path.exists():
               sys.exit(f"ERROR: {path} not found. Run 24_build_clrd2025_long.py first.")
           return pd.read_csv(path)

       try:
           import reservetestr as rt
       except ImportError:
           path = CACHE_DIR / "meyers_actuals_source.csv"
           if not path.exists():
               sys.exit(
                   "ERROR: reservetestr is not installed and "
                   f"{path} (a previous *_cal_detail.csv) is missing."
               )
           df = pd.read_csv(path)[["lob", "group_id", "loss_type", "actual_ultimate"]]
           return df.drop_duplicates().rename(columns={"actual_ultimate": "actual_ultimate_total"})

       records = rt.build_triangle_records()
       rows = []
       for rec in records:
           rows.append({"lob": rec.line, "group_id": rec.group_id, "loss_type": "paid",
                        "actual_ultimate_total": rec.actual_ultimates.get("paid", np.nan)})
           rows.append({"lob": rec.line, "group_id": rec.group_id, "loss_type": "case_incurred",
                        "actual_ultimate_total": rec.actual_ultimates.get("case", np.nan)})
       return pd.DataFrame(rows)
   ```
4. Change `def run_analysis():` to `def run_analysis(dataset: str = "meyers", prefix: str | None = None):` and at its top add:
   ```python
       cfg = DATASETS[dataset]
       prefix = prefix or cfg["prefix"]
       lobs = cfg["lobs"]
       results_csv = CACHE_DIR / f"{prefix}.csv"
       samples_parquet = CACHE_DIR / f"{prefix}_samples.parquet"
   ```
   Then replace every use of `RESULTS_CSV` with `results_csv`, `SAMPLES_PARQUET` with `samples_parquet`, `LOBS` with `lobs`, `load_actual_ultimates()` with `load_actual_ultimates(dataset)`, and the four output paths with `FIGURES_DIR / f"{prefix}_calibration_grid.png"`, `FIGURES_DIR / f"{prefix}_pp_chart.png"`, `CACHE_DIR / f"{prefix}_calibration.csv"`, `CACHE_DIR / f"{prefix}_cal_detail.csv"`. Add `actuals["group_id"] = actuals["group_id"].astype(int)` right after loading actuals so both datasets merge on integers. Replace the hard-coded title strings `"Final Back-Test: ..."` with `f"{cfg['title']}\n..."` keeping the second line of each title unchanged, and make the `print("FINAL CALIBRATION TABLE: 8 methods × 2 loss types")` line say `f"CALIBRATION TABLE [{dataset}]: {len(ALL_METHODS)} methods × 2 loss types"`.
5. Replace the `__main__` block with:
   ```python
   if __name__ == "__main__":
       ap = argparse.ArgumentParser(description="Calibration analysis for the standalone reserving back-test.")
       ap.add_argument("--dataset", choices=sorted(DATASETS), default="meyers")
       ap.add_argument("--prefix", default=None,
                       help="Results file prefix in cache/ (default per dataset: meyers_final / clrd2025_final)")
       a = ap.parse_args()
       run_analysis(a.dataset, a.prefix)
   ```
   and add `import argparse` to the imports. Update the module docstring's Usage to show `--dataset` and `--prefix`.

- [ ] **Step 3: Reproduce the documented Meyers table from the cached results**

```bash
uv run python references/meyers-backtest/22_final_calibration.py --dataset meyers 2>&1 | grep -E "^odp_corr +paid|^odp +case_incurred|^mack +paid|Best for"
```

Expected: `odp_corr paid ... KS 0.151`, `odp case_incurred ... KS 0.066`, `mack paid ... KS 0.266` (matching the v4 table in the README), and `Best for Paid: odp_corr`. Figures land in `references/meyers-backtest/figures/meyers_final_*.png`.

- [ ] **Step 4: Commit**

```bash
git add references/meyers-backtest/22_final_calibration.py
git commit -m "refactor(backtest): parametrize calibration analysis by dataset and prefix

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Run the clrd2025 sweep and record results

**Files:**
- Modify: `references/meyers-backtest/STANDALONE_BACKTEST_README.md` (new section)
- Outputs (gitignored): `cache/clrd2025_final.csv`, `cache/clrd2025_final_samples.parquet`, `cache/clrd2025_final.log`, `cache/clrd2025_final_calibration.csv`, `cache/clrd2025_final_cal_detail.csv`, `figures/clrd2025_final_*.png`

- [ ] **Step 1: Run the sweep (about 10-15 minutes with 8 workers)**

```bash
uv run python scripts/run_stochastic_reserving.py \
  --input references/meyers-backtest/cache/clrd2025_long.csv \
  --output references/meyers-backtest/cache/clrd2025_final.csv \
  --methods mack odp odp_param odp_corr odp_bf odp_cc odp_corr_bf odp_corr_cc bz \
  --loss-col both --n-sims 5000 --rho 0.3 --apriori 0.65 --apriori-sigma 0.15 \
  --n-jobs 8 --random-seed 42 \
  --save-samples references/meyers-backtest/cache/clrd2025_final_samples.parquet \
  2>&1 | grep -v Warning > references/meyers-backtest/cache/clrd2025_final.log
tail -3 references/meyers-backtest/cache/clrd2025_final.log
grep -c "failed" references/meyers-backtest/cache/clrd2025_final.log
```

Expected: `Results written ... rows` and `Samples written ...`. Failures should be limited to `bz` on triangles with negative case-incurred development; inspect with `grep failed ... | cut -d: -f4 | sort | uniq -c`.

- [ ] **Step 2: Run the calibration**

```bash
uv run python references/meyers-backtest/22_final_calibration.py --dataset clrd2025 2>&1 | grep -v Warning | tee references/meyers-backtest/cache/clrd2025_final_analysis.txt | head -60
```

Expected: the calibration table for up to 9 methods x 2 loss types, per-line KS for six lines, verdict.

- [ ] **Step 3: Append a results section to the README**

Append to `references/meyers-backtest/STANDALONE_BACKTEST_README.md` a section `## v5 Analysis: clrd2025 extension (origins 1998-2007)` containing:

1. Setup: data source (`cl.load_sample("clrd2025")`, chainladder 0.10.1), window, eligibility funnel numbers printed by Task 5 (376 / 351 / 328 / 240), LOB counts, methods (9 incl. `bz`), defaults, script commands from Steps 1-2, output paths.
2. The full calibration table copied from `cache/clrd2025_final_calibration.csv`, formatted like the v4 table (columns Method, Loss type, N, Mean pctl, C50%, C80%, KS stat, Med CV, Med |%err|), sorted by KS.
3. The per-line KS table for the best two methods per loss type.
4. A "Comparison with Meyers 1988-1997" paragraph with a small table of KS(paid) and KS(case) per method for v4 Meyers vs clrd2025, and 3-5 bullet findings written from the numbers actually produced (which methods stay best, whether BF/CC over-reserving persists with apriori 0.65 for the 1998-2007 era, how `bz` compares with `odp_corr`, the effect of the small prodliab/medmal groups). Do not invent numbers; every figure quoted must appear in the CSV.
5. Figure list.

- [ ] **Step 4: Commit**

```bash
git add references/meyers-backtest/STANDALONE_BACKTEST_README.md
git commit -m "docs(backtest): clrd2025 Meyers-style calibration results

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Refresh the Meyers 200-triangle results on chainladder 0.10.1

**Files:**
- Modify: `references/meyers-backtest/STANDALONE_BACKTEST_README.md` (new section)
- Modify: `scripts/README.md` (calibration table)
- Outputs (gitignored): `cache/meyers_v5.csv`, `cache/meyers_v5_samples.parquet`, `cache/meyers_v5.log`, `cache/meyers_v5_calibration.csv`, `cache/meyers_v5_cal_detail.csv`, `figures/meyers_v5_*.png`

- [ ] **Step 1: Run the sweep on the cached Meyers long CSV (copied in Task 6 Step 1)**

```bash
uv run python scripts/run_stochastic_reserving.py \
  --input references/meyers-backtest/cache/meyers_long.csv \
  --output references/meyers-backtest/cache/meyers_v5.csv \
  --methods mack odp odp_param odp_corr odp_bf odp_cc odp_corr_bf odp_corr_cc bz \
  --loss-col both --n-sims 5000 --rho 0.3 --apriori 0.65 --apriori-sigma 0.15 \
  --n-jobs 8 --random-seed 42 \
  --save-samples references/meyers-backtest/cache/meyers_v5_samples.parquet \
  2>&1 | grep -v Warning > references/meyers-backtest/cache/meyers_v5.log
tail -2 references/meyers-backtest/cache/meyers_v5.log
uv run python references/meyers-backtest/22_final_calibration.py --dataset meyers --prefix meyers_v5 2>&1 | grep -v Warning | tee references/meyers-backtest/cache/meyers_v5_analysis.txt | head -40
```

Expected: 200 triangles processed; the calibration table prints for 9 methods.

- [ ] **Step 2: Compare against v4**

```bash
uv run python - <<'EOF'
import pandas as pd
v4 = pd.read_csv("references/meyers-backtest/cache/meyers_final_calibration.csv")[["method","loss_type","ks_stat","mean_pctl"]]
v5 = pd.read_csv("references/meyers-backtest/cache/meyers_v5_calibration.csv")[["method","loss_type","ks_stat","mean_pctl"]]
m = v4.merge(v5, on=["method","loss_type"], how="outer", suffixes=("_v4","_v5"))
m["delta_ks"] = m["ks_stat_v5"] - m["ks_stat_v4"]
print(m.sort_values(["loss_type","method"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
EOF
```

Expected: `odp`, `odp_param`, `odp_corr` unchanged to three decimals (they do not touch the changed code paths); `mack` changes (sigma interpolation); `odp_bf`, `odp_cc`, `odp_corr_bf`, `odp_corr_cc` change (lognormal apriori); `bz` is new. If `odp_corr` changes by more than 0.005, stop and investigate before documenting.

- [ ] **Step 3: Document**

1. Append to `STANDALONE_BACKTEST_README.md` a section `## v5 Analysis: chainladder 0.10.1 refresh (Meyers 200 triangles)` with: what changed (chainladder 0.10.1, `--mack-sigma-interpolation mack`, lognormal BF/CC apriori draws, new `bz`), the v4-vs-v5 comparison table from Step 2, the full v5 calibration table from `cache/meyers_v5_calibration.csv`, and 3-4 bullet findings drawn from those numbers only.
2. In `scripts/README.md`, replace the "Calibration results" table with a v5 table: columns Method, Paid KS, Case KS, Notes, one row per method including `bz`, values from `cache/meyers_v5_calibration.csv`. Update the intro sentence to `(Meyers 2015, 200 triangles, chainladder 0.10.1, lognormal PV, rho=0.3, n=5000, apriori_sigma=0.15)` and keep a one-line pointer to the README in `references/meyers-backtest/` for the history.

- [ ] **Step 4: Commit**

```bash
git add references/meyers-backtest/STANDALONE_BACKTEST_README.md scripts/README.md
git commit -m "docs(backtest): refresh Meyers calibration on chainladder 0.10.1

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Final documentation and verification

**Files:**
- Modify: `CLAUDE.md` (commands/architecture notes)
- Modify: `scripts/test_non_annual.py` (only if the 18-month combo now passes)

- [ ] **Step 1: Update CLAUDE.md**

1. In the "Commands" intro sentence add: `chainladder>=0.10.1 is required (integer-month development ages, Mack sigma interpolation, lognormal BF/CC aprioris, BarnettZehnwirth).`
2. In "Architecture" add a short paragraph after the two model families:
   ```markdown
   `scripts/run_stochastic_reserving.py` is a standalone frequentist benchmark (no `bayesianchainladder` import) with nine methods including `bz` (Barnett-Zehnwirth). Its calibration back-tests live in `references/meyers-backtest/`: `16_build_meyers_long.py` (needs `reservetestr`, not a declared dependency) and `24_build_clrd2025_long.py` (needs only chainladder) produce the long CSVs; `22_final_calibration.py --dataset {meyers,clrd2025}` computes implied-percentile calibration. Results and figures are gitignored; the tables are copied into `STANDALONE_BACKTEST_README.md`.
   ```

- [ ] **Step 2: Run the non-annual grain script**

```bash
timeout 1200 uv run python scripts/test_non_annual.py 2>&1 | grep -v Warning | grep -E "COMBO|STATUS|OVERALL"
```

Expected: `OVERALL: ALL COMBOS PASSED` with the 18-month combo reported as `EXPECTED FAILURE`. If the 18-month combo now passes under chainladder 0.10.1, change its `expected_failure` flag in the `builders` list to `False` and reduce its docstring's "KNOWN LIMITATION" paragraph to a one-line note that it passes from chainladder 0.10.1 onward.

- [ ] **Step 3: Full verification**

```bash
uv run ruff check bayesianchainladder tests
uv run black --check bayesianchainladder/utils.py tests/test_utils.py tests/test_reserving_script.py
uv run mypy bayesianchainladder 2>&1 | tail -1
uv run pytest -q -p no:cacheprovider 2>&1 | tail -1
git status --short
```

Expected: ruff clean; black clean; mypy reports no errors in `utils.py` (compare any output against `git diff main -- bayesianchainladder/utils.py`); pytest `201 passed, 55 skipped` (195 + 6 script tests); `git status` shows only files you intend to commit (no cache or png files).

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md scripts/test_non_annual.py
git commit -m "docs: record chainladder 0.10.1 features and back-test layout

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review notes

- Coverage: design items 1-5 map to Tasks 1, 2, 3+4, 5+6+7, 8; docs in Tasks 2, 3, 4, 7, 8, 9.
- Names used across tasks: `_triangle_cells`, `_run_mack(..., sigma_interpolation)`, `mack_sigma_interpolation`, `_run_bz(..., formula)`, `bz_formula`, `run_analysis(dataset, prefix)`, `load_actual_ultimates(dataset)`, cache prefixes `meyers_final`, `meyers_v5`, `clrd2025_final`, actuals columns `lob, group_id, loss_type, actual_ultimate_total`.
- Test counts: Task 2 adds 6 tests (28 -> 34 in test_utils, suite 189 -> 195); Tasks 3-4 add 6 script tests (suite -> 201).
