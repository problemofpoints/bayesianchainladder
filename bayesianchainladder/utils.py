"""
Utility functions for Bayesian chain ladder modeling.

This module provides helper functions for converting chainladder Triangle objects
to long-format DataFrames suitable for Bambi/PyMC modeling, and other utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import chainladder as cl


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

    Rows are in row-major (origin, development) order. The caller decides
    which triangle (cumulative or incremental) to pass in, and that choice
    is what defines the observation mask returned here.

    Pass cumulative triangles where possible; an incremental input cannot
    represent an observed zero increment (chainladder stores it as NaN), so
    such cells are treated as unobserved.

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

    Pass cumulative triangles where possible; an incremental input cannot
    represent an observed zero increment (chainladder stores it as NaN), so
    such cells are treated as unobserved.

    Raises
    ------
    ValueError
        If a cumulative triangle has an interior gap — an unobserved cell
        followed by an observed one at a later development period for the
        same origin. ``cum_to_incr()`` treats an interior NaN as zero, so the
        derived incrementals would not reconcile to the latest cumulative;
        such triangles must be observed contiguously from the first
        development period.

    Examples
    --------
    >>> import chainladder as cl
    >>> from bayesianchainladder.utils import triangle_to_dataframe
    >>> tri = cl.load_sample("raa")
    >>> df = triangle_to_dataframe(tri)
    >>> df.head()
    """
    tri = triangle.copy()
    # The observation mask always comes from the triangle as supplied: for a
    # cumulative triangle that's its own NaN pattern, not cum_to_incr()'s.
    cells = _triangle_cells(tri)
    observed = cells["observed"].to_numpy()

    if tri.is_cumulative:
        # Interior gaps (an unobserved cell followed by an observed one at a
        # later development period, for the same origin) are not supported:
        # cum_to_incr() treats an interior NaN as zero, so the incrementals
        # derived below would not reconcile to the latest cumulative value.
        n_origin, n_dev = len(tri.origin), len(tri.development)
        observed_grid = observed.reshape(n_origin, n_dev)
        gap_mask = observed_grid[:, 1:] & ~observed_grid[:, :-1]
        if gap_mask.any():
            origin_labels = cells["origin"].to_numpy().reshape(n_origin, n_dev)[:, 0]
            dev_labels = cells["dev"].to_numpy().reshape(n_origin, n_dev)[0, :]
            gap_rows, gap_cols = np.where(gap_mask)
            offenders = "; ".join(
                f"origin={origin_labels[i]} unobserved at dev={dev_labels[j]} "
                f"but observed at dev={dev_labels[j + 1]}"
                for i, j in zip(gap_rows, gap_cols, strict=True)
            )
            raise ValueError(
                "Cumulative triangle has an interior gap and cannot be "
                f"converted to incrementals: {offenders}. Cumulative "
                "triangles must be observed contiguously from the first "
                "development period, because chainladder's cum_to_incr() "
                "treats an interior NaN as zero and the derived incrementals "
                "would not reconcile to the latest cumulative value."
            )

        # chainladder's cum_to_incr() stores a zero increment as NaN, and for
        # some triangle shapes it can emit a non-NaN value for a cell that is
        # NaN (unobserved) in the cumulative triangle above — so its output
        # is a value payload, not a mask. Treat NaN inside the observed
        # region as a zero increment, and drop anything outside it.
        incr = _triangle_cells(tri.cum_to_incr())["value"].to_numpy()
        incremental = np.where(observed, np.nan_to_num(incr, nan=0.0), np.nan)
        cumulative = cells["value"].to_numpy()
    else:
        incremental = cells["value"].to_numpy()
        cumulative = (
            _triangle_cells(tri.incr_to_cum())["value"].to_numpy()
            if include_cumulative
            else None
        )

    df = (
        cells.loc[observed, ["origin", "dev", "calendar"]]
        .assign(**{value_column: incremental[observed]})
        .reset_index(drop=True)
    )

    if include_cumulative:
        df["cumulative"] = cumulative[observed]

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
    df = cells.loc[~cells["observed"], ["origin", "dev", "calendar"]].reset_index(
        drop=True
    )
    df[value_column] = np.nan
    return df


def _extract_period_value(period) -> int:
    """Extract integer value from a period (Timestamp, int, etc.).

    Kept as an internal helper for :mod:`bayesianchainladder.bootstrap`, which
    uses it independently of the ``_triangle_cells``-based converters above.
    """
    if hasattr(period, "year"):
        return period.year
    elif hasattr(period, "days"):
        # Development period as timedelta - convert to months/years
        days = period.days
        # Assume annual periods
        return max(1, round(days / 365))
    else:
        return int(period)


def prepare_model_data(
    triangle: cl.Triangle,
    exposure_triangle: cl.Triangle | None = None,
    exposure_column: str = "exposure",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for Bayesian chain ladder modeling.

    This function prepares both the observed data and the future prediction
    data from a chainladder Triangle.

    Parameters
    ----------
    triangle : chainladder.Triangle
        The claims triangle (cumulative or incremental).
    exposure_triangle : chainladder.Triangle, optional
        Optional exposure triangle (e.g., earned premium).
    exposure_column : str, optional
        Name for the exposure column. Default is "exposure".

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple of (observed_df, future_df) DataFrames.
    """
    observed_df = triangle_to_dataframe(triangle, include_cumulative=True)
    future_df = get_future_dataframe(triangle)

    if exposure_triangle is not None:
        # Add exposure to observed data
        exp_df = triangle_to_dataframe(exposure_triangle, value_column=exposure_column)
        # Merge on origin (exposure typically only varies by origin)
        if "dev" in exp_df.columns:
            # Take first development period's exposure
            exp_first = exp_df[exp_df["dev"] == exp_df["dev"].min()][
                ["origin", exposure_column]
            ]
            observed_df = observed_df.merge(exp_first, on="origin", how="left")
            future_df = future_df.merge(exp_first, on="origin", how="left")
        else:
            observed_df = observed_df.merge(
                exp_df[["origin", exposure_column]], on="origin", how="left"
            )
            future_df = future_df.merge(
                exp_df[["origin", exposure_column]], on="origin", how="left"
            )

    return observed_df, future_df


def add_categorical_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    formula: str | None = None,
) -> pd.DataFrame:
    """
    Convert specified columns to categorical type for Bambi.

    Columns used in spline terms (bs(), cr()) are kept as numeric.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list[str], optional
        Columns to convert to categorical. If None, converts
        origin, dev, and calendar columns.
    formula : str, optional
        Model formula. If provided, columns used in spline terms like
        bs() or cr() will be kept as numeric instead of categorical.

    Returns
    -------
    pd.DataFrame
        DataFrame with appropriate column types.
    """
    df = df.copy()

    if columns is None:
        columns = ["origin", "dev", "calendar"]

    # Detect columns that should stay numeric and/or need indexed versions
    numeric_columns: set[str] = set()
    indexed_columns: set[str] = set()

    if formula is not None:
        import re

        # Match bs(...) or cr(...) - spline terms
        spline_pattern = r"\b(?:bs|cr)\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)"
        numeric_columns.update(re.findall(spline_pattern, formula))

        # Match column**N or pow(column, N) - polynomial terms
        power_pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\*\*\s*\d"
        numeric_columns.update(re.findall(power_pattern, formula))
        pow_pattern = r"\bpow\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)"
        numeric_columns.update(re.findall(pow_pattern, formula))

        # Match np.log(), np.sqrt(), np.maximum(), etc. - numpy transforms
        # Handles both np.func(col) and np.func(val, col) patterns
        np_pattern = r"\bnp\.\w+\s*\([^)]*\b([a-zA-Z_][a-zA-Z0-9_]*_idx)\b"
        numeric_columns.update(re.findall(np_pattern, formula))
        np_pattern_simple = r"\bnp\.\w+\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*[,)]"
        numeric_columns.update(re.findall(np_pattern_simple, formula))

        # Match {expr} syntax with column names inside (e.g., {origin**2})
        brace_pattern = r"\{[^}]*\b([a-zA-Z_][a-zA-Z0-9_]*)\b[^}]*\}"
        numeric_columns.update(re.findall(brace_pattern, formula))

        # Match bare column name (not wrapped in C()) used directly in formula
        # This catches "origin + ..." but not "C(origin) + ..."
        # Split by common operators and check each term
        terms = re.split(r"[~+\-*/(),\s]+", formula)
        for term in terms:
            # If a column appears as a bare term (not empty, not a number, not a function)
            if term and term in columns and not re.match(r"^\d+\.?\d*$", term):
                # Check if this column is NOT wrapped in C() in the formula
                c_wrapped = re.search(rf"\bC\s*\(\s*{re.escape(term)}\s*\)", formula)
                if not c_wrapped:
                    numeric_columns.add(term)

        # Check for _idx suffix usage - these need indexed versions
        # Match origin_idx, dev_idx, calendar_idx anywhere in formula
        idx_pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)_idx\b"
        indexed_columns.update(re.findall(idx_pattern, formula))

        # Also add _idx columns to numeric_columns so they stay numeric
        for col in re.findall(idx_pattern, formula):
            numeric_columns.add(f"{col}_idx")

    for col in columns:
        if col in df.columns:
            # Create indexed version (1, 2, 3, ...) if needed for _idx references
            if col in indexed_columns:
                # Sort unique values and create mapping to 1-based index
                unique_vals = sorted(df[col].unique())
                val_to_idx = {v: i + 1 for i, v in enumerate(unique_vals)}
                df[f"{col}_idx"] = df[col].map(val_to_idx)

            if col in numeric_columns:
                # Keep as numeric for spline/polynomial/transform terms
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                # Convert to categorical
                df[col] = df[col].astype("category")

    return df


def compute_log_exposure_offset(
    df: pd.DataFrame,
    exposure_column: str = "exposure",
) -> pd.Series:
    """
    Compute log-exposure offset for use in GLM.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with exposure column.
    exposure_column : str, optional
        Name of the exposure column. Default is "exposure".

    Returns
    -------
    pd.Series
        Log of exposure values (for use as offset in GLM).
    """
    if exposure_column not in df.columns:
        raise ValueError(f"Exposure column '{exposure_column}' not found in DataFrame")

    exposure = df[exposure_column]
    if (exposure <= 0).any():
        raise ValueError("Exposure values must be positive for log transformation")

    return np.log(exposure)


def create_design_info(
    df: pd.DataFrame,
    formula: str,
) -> dict:
    """
    Extract design matrix information from a formula and DataFrame.

    This is useful for understanding what terms will be in the model.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    formula : str
        Bambi/Patsy-style formula.

    Returns
    -------
    dict
        Dictionary with information about model terms.
    """
    # Parse formula to extract terms
    terms = []
    response = None

    if "~" in formula:
        parts = formula.split("~")
        response = parts[0].strip()
        rhs = parts[1].strip()
    else:
        rhs = formula

    # Split by + and extract term names
    for term in rhs.split("+"):
        term = term.strip()
        if term:
            terms.append(term)

    return {
        "response": response,
        "terms": terms,
        "n_observations": len(df),
        "origin_levels": (
            sorted(df["origin"].unique()) if "origin" in df.columns else []
        ),
        "dev_levels": sorted(df["dev"].unique()) if "dev" in df.columns else [],
        "calendar_levels": (
            sorted(df["calendar"].unique()) if "calendar" in df.columns else []
        ),
    }


def prepare_csr_data(
    triangle: cl.Triangle,
    premium_triangle: cl.Triangle | None = None,
    premium_value: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for the Changing Settlement Rate (CSR) model.

    The CSR model works on log cumulative paid loss with log premium as an offset.
    This function converts triangles to the appropriate format.

    Parameters
    ----------
    triangle : chainladder.Triangle
        The claims triangle (cumulative paid loss). If incremental, will be
        converted to cumulative.
    premium_triangle : chainladder.Triangle, optional
        Premium triangle (earned premium by origin). If provided, premium values
        are extracted and matched to each origin year.
    premium_value : float, optional
        Single premium value to use for all origin years (if premium_triangle
        is not provided). Required if premium_triangle is None.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple of (observed_df, future_df) DataFrames with columns:
        - origin: Origin period
        - dev: Development period
        - cumulative: Cumulative paid loss
        - logloss: Log of cumulative paid loss
        - premium: Premium amount
        - logprem: Log of premium

    Raises
    ------
    ValueError
        If neither premium_triangle nor premium_value is provided.

    Examples
    --------
    >>> import chainladder as cl
    >>> from bayesianchainladder.utils import prepare_csr_data
    >>> tri = cl.load_sample("raa")
    >>> observed, future = prepare_csr_data(tri, premium_value=10000)
    """
    if premium_triangle is None and premium_value is None:
        raise ValueError(
            "Either premium_triangle or premium_value must be provided for CSR model"
        )

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

    # Add premium
    if premium_triangle is not None:
        # Extract premium values by origin
        prem_df = triangle_to_dataframe(premium_triangle, value_column="premium")
        # Take first dev period's premium value for each origin
        if "dev" in prem_df.columns:
            prem_first = prem_df[prem_df["dev"] == prem_df["dev"].min()][
                ["origin", "premium"]
            ]
        else:
            prem_first = prem_df[["origin", "premium"]]

        observed_df = observed_df.merge(prem_first, on="origin", how="left")
        if len(future_df) > 0:
            future_df = future_df.merge(prem_first, on="origin", how="left")
    else:
        # Use constant premium value
        observed_df["premium"] = premium_value
        if len(future_df) > 0:
            future_df["premium"] = premium_value

    # Add log premium
    observed_df["logprem"] = np.log(observed_df["premium"])
    if len(future_df) > 0:
        future_df["logprem"] = np.log(future_df["premium"])

    # Ensure proper dtypes
    for df in [observed_df, future_df]:
        if len(df) > 0:
            df["origin"] = df["origin"].astype(int)
            df["dev"] = df["dev"].astype(int)

    return observed_df, future_df


def validate_triangle(triangle: cl.Triangle) -> None:
    """
    Validate that a triangle is suitable for Bayesian chain ladder modeling.

    Parameters
    ----------
    triangle : chainladder.Triangle
        Triangle to validate.

    Raises
    ------
    ValueError
        If the triangle is not suitable for modeling.
    """
    import chainladder as cl

    if not isinstance(triangle, cl.Triangle):
        raise ValueError("Input must be a chainladder.Triangle object")

    # Check for minimum size
    if len(triangle.origin) < 2:
        raise ValueError("Triangle must have at least 2 origin periods")

    if len(triangle.development) < 2:
        raise ValueError("Triangle must have at least 2 development periods")

    # Check for negative values
    values = triangle.values
    while values.ndim > 2:
        values = values[0]

    finite_values = values[np.isfinite(values)]
    if (finite_values < 0).any():
        raise ValueError(
            "Triangle contains negative values. "
            "Consider using a family that supports negative values."
        )
