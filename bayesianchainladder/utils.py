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


def triangle_to_dataframe(
    triangle: cl.Triangle,
    value_column: str = "incremental",
    include_cumulative: bool = False,
) -> pd.DataFrame:
    """
    Convert a chainladder Triangle to a long-format DataFrame.

    This function converts a chainladder.Triangle object into a long-format
    pandas DataFrame suitable for use with Bambi/PyMC GLM models.

    Parameters
    ----------
    triangle : chainladder.Triangle
        A chainladder Triangle object. Can be cumulative or incremental.
    value_column : str, optional
        Name for the value column in the output DataFrame.
        Default is "incremental".
    include_cumulative : bool, optional
        If True, include a "cumulative" column in addition to incremental.
        Default is False.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - origin: Origin period (accident year)
        - dev: Development period
        - calendar: Calendar period (origin + dev - 1)
        - incremental (or value_column): The cell values
        - cumulative (optional): Cumulative values if include_cumulative=True

    Examples
    --------
    >>> import chainladder as cl
    >>> from bayesianchainladder.utils import triangle_to_dataframe
    >>> tri = cl.load_sample("raa")
    >>> df = triangle_to_dataframe(tri)
    >>> df.head()
    """
    # Ensure we have a single triangle (squeeze any singleton dimensions)
    tri = triangle.copy()

    # Get the triangle as a pandas DataFrame in long format
    # First, convert to incremental if cumulative
    if tri.is_cumulative:
        tri_incr = tri.incr_to_cum().cum_to_incr()  # Ensure incremental
    else:
        tri_incr = tri

    # Get origin and development indices
    origins = tri_incr.origin
    developments = tri_incr.development

    # Build the long-format DataFrame
    rows = []

    # Get the values - handle multi-index and single triangle cases
    values = tri_incr.values

    # Squeeze singleton dimensions
    while values.ndim > 2:
        if values.shape[0] == 1:
            values = values[0]
        else:
            break

    for i, origin in enumerate(origins):
        for j, dev in enumerate(developments):
            val = values[i, j]

            # Skip NaN values (future/unobserved cells)
            if np.isnan(val):
                continue

            # Extract origin year as integer
            origin_val = _extract_period_value(origin)
            dev_val = _extract_period_value(dev)

            # Calendar period = origin + dev - 1 (for annual data)
            calendar = origin_val + dev_val - 1

            row = {
                "origin": origin_val,
                "dev": dev_val,
                "calendar": calendar,
                value_column: val,
            }

            rows.append(row)

    df = pd.DataFrame(rows)

    # Add cumulative if requested
    if include_cumulative and len(df) > 0:
        # Get cumulative triangle
        if tri.is_cumulative:
            tri_cum = tri
        else:
            tri_cum = tri.incr_to_cum()

        cum_values = tri_cum.values
        while cum_values.ndim > 2:
            if cum_values.shape[0] == 1:
                cum_values = cum_values[0]
            else:
                break

        # Match cumulative values to the DataFrame
        cumulative = []
        for _, row in df.iterrows():
            origin_idx = list(origins).index(
                _find_matching_period(origins, row["origin"])
            )
            dev_idx = list(developments).index(
                _find_matching_period(developments, row["dev"])
            )
            cumulative.append(cum_values[origin_idx, dev_idx])

        df["cumulative"] = cumulative

    # Ensure proper dtypes
    df["origin"] = df["origin"].astype(int)
    df["dev"] = df["dev"].astype(int)
    df["calendar"] = df["calendar"].astype(int)

    return df


def _extract_period_value(period) -> int:
    """Extract integer value from a period (Timestamp, int, etc.)."""
    if hasattr(period, "year"):
        return period.year
    elif hasattr(period, "days"):
        # Development period as timedelta - convert to months/years
        days = period.days
        # Assume annual periods
        return max(1, round(days / 365))
    else:
        return int(period)


def _find_matching_period(periods, value):
    """Find the matching period in a list of periods."""
    for p in periods:
        if _extract_period_value(p) == value:
            return p
    return None


def get_future_dataframe(
    triangle: cl.Triangle,
    value_column: str = "incremental",
) -> pd.DataFrame:
    """
    Create a DataFrame for future (unobserved) cells in a triangle.

    This function creates a DataFrame containing the cells that need to be
    predicted (the lower-right portion of the triangle that is unobserved).

    Parameters
    ----------
    triangle : chainladder.Triangle
        A chainladder Triangle object.
    value_column : str, optional
        Name for the value column (will be NaN for future cells).
        Default is "incremental".

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns: origin, dev, calendar, and value_column (NaN).
    """
    tri = triangle.copy()

    origins = tri.origin
    developments = tri.development

    values = tri.values
    while values.ndim > 2:
        if values.shape[0] == 1:
            values = values[0]
        else:
            break

    rows = []

    for i, origin in enumerate(origins):
        for j, dev in enumerate(developments):
            val = values[i, j]

            # Only include NaN values (future/unobserved cells)
            if not np.isnan(val):
                continue

            origin_val = _extract_period_value(origin)
            dev_val = _extract_period_value(dev)
            calendar = origin_val + dev_val - 1

            row = {
                "origin": origin_val,
                "dev": dev_val,
                "calendar": calendar,
                value_column: np.nan,
            }

            rows.append(row)

    df = pd.DataFrame(rows)

    if len(df) > 0:
        df["origin"] = df["origin"].astype(int)
        df["dev"] = df["dev"].astype(int)
        df["calendar"] = df["calendar"].astype(int)

    return df


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
        exp_df = triangle_to_dataframe(
            exposure_triangle, value_column=exposure_column
        )
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

    # Detect columns used in spline terms
    spline_columns: set[str] = set()
    if formula is not None:
        import re
        # Match bs(...) or cr(...) and extract the first argument (column name)
        spline_pattern = r'\b(?:bs|cr)\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(spline_pattern, formula)
        spline_columns = set(matches)

    for col in columns:
        if col in df.columns:
            if col in spline_columns:
                # Keep as numeric for spline terms
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
        "origin_levels": sorted(df["origin"].unique()) if "origin" in df.columns else [],
        "dev_levels": sorted(df["dev"].unique()) if "dev" in df.columns else [],
        "calendar_levels": (
            sorted(df["calendar"].unique()) if "calendar" in df.columns else []
        ),
    }


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
