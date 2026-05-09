"""Shared utilities for the prior-elicitation-2026 analysis.

This module is imported by every numbered analysis script (01–05) and the
test_common.py suite. Keep it small and side-effect-free at import time.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

import chainladder as cl
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LINES = ["OLO", "OLC", "CAL", "WC", "PPAL", "CMP"]

TRIANGLE_JSON = (
    Path.home()
    / "Projects/reserve-risk-benchmarking/results/schedule_p_triangle.json"
)

ANALYSIS_DIR = Path(__file__).resolve().parent
CACHE_DIR = ANALYSIS_DIR / "cache"
FIGURES_DIR = ANALYSIS_DIR / "figures"

MIN_ORIGIN_YEARS = 8
SAMPLE_SEED = 20260508
SAMPLE_PER_LINE = 24  # 8 per tercile

# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def is_eligible_triangle(tri: cl.Triangle) -> bool:
    """Return True if a single (snl_id, lob) Triangle is eligible for the study.

    Rules (must all hold):
      1. >= MIN_ORIGIN_YEARS origin years have at least one observed cell.
      2. All observed cumulative `paid_loss` values are strictly positive.
      3. Net earned premium is positive for every origin with paid data.
      4. At least one positive incremental at dev > 12 (this is implied by Rule 5
         plus origin coverage but kept explicit for clarity).
      5. All observed incremental paid_loss values are strictly positive
         (cumulative paid never decreases between observed dev periods).

    Parameters
    ----------
    tri : cl.Triangle
        A Triangle whose vdims include `paid_loss` and `net_earned_premium`,
        and whose index identifies a single (snl_id, line_of_business) pair.
    """
    vdims = list(tri.vdims)
    if "paid_loss" not in vdims:
        return False
    paid = tri["paid_loss"].values[0, 0]  # shape: (n_origin, n_dev)
    if "net_earned_premium" in vdims:
        prem = tri["net_earned_premium"].values[0, 0]
    else:
        prem = None

    # Rule 1: >= MIN_ORIGIN_YEARS origin years with any observation.
    origin_has_data = np.any(~np.isnan(paid), axis=1)
    if origin_has_data.sum() < MIN_ORIGIN_YEARS:
        return False

    # Rule 2: all observed cumulative paid > 0.
    observed = paid[~np.isnan(paid)]
    if observed.size == 0 or np.any(observed <= 0):
        return False

    # Rule 3: positive premium for every origin with paid data.
    if prem is not None:
        prem_for_active = prem[origin_has_data]
        prem_observed = prem_for_active[~np.isnan(prem_for_active)]
        if prem_observed.size == 0 or np.any(prem_observed <= 0):
            return False

    # Rule 5: every observed incremental must be > 0.
    inc = np.full_like(paid, np.nan, dtype=float)
    inc[:, 0] = paid[:, 0]
    inc[:, 1:] = paid[:, 1:] - paid[:, :-1]
    obs_inc = inc[~np.isnan(inc)]
    if obs_inc.size == 0 or np.any(obs_inc <= 0):
        return False

    # Rule 4: positive incremental somewhere at dev > 12.
    # (Implied by Rule 5 for any triangle with >1 dev column, kept explicit
    # for clarity.  inc was computed above; reuse it here.)
    if paid.shape[1] < 2:
        return False
    if not np.any(inc[~np.isnan(inc)] > 0):
        return False

    return True


# ---------------------------------------------------------------------------
# Triangle loading
# ---------------------------------------------------------------------------


def load_full_triangle() -> cl.Triangle:
    """Read the Schedule P JSON file and return a multi-index chainladder Triangle."""
    with open(TRIANGLE_JSON, "r", encoding="utf-8") as f:
        json_str = f.read()
    return cl.read_json(json_str)


def iter_eligible_triangles(
    full_tri: cl.Triangle, line: str
) -> Iterator[tuple[str, cl.Triangle]]:
    """Yield (snl_id, single-row Triangle) pairs for triangles in `line` that pass eligibility."""
    sub = full_tri[full_tri["line_of_business"] == line]
    n = len(sub.index)
    for i in range(n):
        row = sub.iloc[i]
        if not is_eligible_triangle(row):
            continue
        snl_id = row.index["snl_id"].iloc[0]
        yield snl_id, row


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------


def booked_reserve(tri: cl.Triangle) -> float:
    """Return the latest-diagonal booked reserve for a single-row Triangle.

    booked_reserve = booked_ultimate_loss − cumulative paid_loss, summed across
    accident years.
    """
    vdims = list(tri.vdims)
    if "booked_reserve" in vdims:
        latest = tri["booked_reserve"].latest_diagonal.values
        return float(np.nansum(latest))
    if "booked_ultimate_loss" in vdims and "paid_loss" in vdims:
        ult = tri["booked_ultimate_loss"].latest_diagonal.values
        paid = tri["paid_loss"].latest_diagonal.values
        return float(np.nansum(ult - paid))
    raise ValueError(
        f"Triangle missing both 'booked_reserve' and 'booked_ultimate_loss' "
        f"vdims; available: {vdims}"
    )


def select_sample(full_tri: cl.Triangle, line: str) -> list[str]:
    """Return SAMPLE_PER_LINE snl_ids for `line` stratified into terciles by booked reserve.

    8 ids per tercile, sampled uniformly within tercile, deterministic on SAMPLE_SEED + line.
    """
    eligible = list(iter_eligible_triangles(full_tri, line))
    if len(eligible) < SAMPLE_PER_LINE:
        # Edge case: not enough eligible triangles. Return all of them.
        return [snl for snl, _ in eligible]

    df = pd.DataFrame(
        {
            "snl_id": [snl for snl, _ in eligible],
            "br": [booked_reserve(t) for _, t in eligible],
        }
    )
    df["tercile"] = pd.qcut(
        df["br"].rank(method="first"), 3, labels=["small", "mid", "large"]
    )

    # Seed combines the global seed with a stable line hash so each line gets its own draw.
    line_hash = int(hashlib.md5(line.encode()).hexdigest(), 16) % 10_000
    seed = SAMPLE_SEED + line_hash
    rng = np.random.default_rng(seed)
    per_tercile = SAMPLE_PER_LINE // 3
    picked: list[str] = []
    for label in ["small", "mid", "large"]:
        pool = df.loc[df["tercile"] == label, "snl_id"].tolist()
        n = min(per_tercile, len(pool))
        picked.extend(rng.choice(pool, size=n, replace=False).tolist())
    # Tie-breaker if 24 not divisible by 3 in some future change.
    while len(picked) < SAMPLE_PER_LINE and len(picked) < len(df):
        leftover = [s for s in df["snl_id"] if s not in picked]
        picked.append(rng.choice(leftover))
    return sorted(picked)


# ---------------------------------------------------------------------------
# Deterministic chain ladder fit and Pearson residuals
# ---------------------------------------------------------------------------


def _to_incremental(cum: np.ndarray) -> np.ndarray:
    """Cumulative → incremental along the dev axis. NaN-safe."""
    inc = np.full_like(cum, np.nan, dtype=float)
    inc[:, 0] = cum[:, 0]
    inc[:, 1:] = cum[:, 1:] - cum[:, :-1]
    return inc


def _chain_ladder_fitted(cum: np.ndarray) -> np.ndarray:
    """Fit deterministic chain ladder, return fitted *incremental* values for the upper triangle.

    Uses volume-weighted age-to-age factors. Returns an array of shape
    cum.shape with NaN where cum is NaN.

    Notes
    -----
    Anchored at the *latest* observed cell of each origin. For triangles with
    interior NaN gaps (i.e., a row that is observed → NaN → observed) the
    back-cast direction can lose otherwise-recoverable fitted values.
    Schedule P upper triangles are contiguous from j=0, so this limitation
    does not affect the present project but should be revisited if the data
    shape changes.
    """
    n_origin, n_dev = cum.shape
    # Volume-weighted age-to-age factors f[j] = sum cum[:, j+1] / sum cum[:, j],
    # with the sums taken over origins where both cells are observed.
    f = np.full(n_dev - 1, np.nan)
    for j in range(n_dev - 1):
        mask = ~np.isnan(cum[:, j]) & ~np.isnan(cum[:, j + 1])
        denom = np.sum(cum[mask, j])
        numer = np.sum(cum[mask, j + 1])
        if denom > 0:
            f[j] = numer / denom

    # Fitted cumulative: anchor at the latest observed cell of each origin, then
    # back-fill earlier dev periods by dividing by f, and forward-fill later dev
    # periods by multiplying by f. For a standard contiguous upper triangle this
    # is numerically equivalent to forward-casting from the first cell.
    fit_cum = np.full_like(cum, np.nan)
    for i in range(n_origin):
        # Use the last observed cumulative as the anchor and back-cast / forward-cast.
        observed_j = np.where(~np.isnan(cum[i]))[0]
        if observed_j.size == 0:
            continue
        # Anchor at the latest observation, then back-fill earlier dev's by dividing.
        last_j = observed_j[-1]
        fit_cum[i, last_j] = cum[i, last_j]
        for j in range(last_j - 1, -1, -1):
            if np.isnan(f[j]) or f[j] == 0:
                fit_cum[i, j] = np.nan
            else:
                fit_cum[i, j] = fit_cum[i, j + 1] / f[j]
        for j in range(last_j + 1, n_dev):
            if np.isnan(f[j - 1]):
                fit_cum[i, j] = np.nan
            else:
                fit_cum[i, j] = fit_cum[i, j - 1] * f[j - 1]

    fit_inc = _to_incremental(fit_cum)
    # Restrict fitted to upper triangle (where cum was observed).
    fit_inc[np.isnan(cum)] = np.nan
    return fit_inc


def pearson_residuals(tri: cl.Triangle) -> pd.DataFrame:
    """Compute Shapland's standardised Pearson residuals for an upper-triangle paid_loss.

    Returns long-format with columns:
      - origin_idx (0-based)
      - dev_idx   (0-based)
      - cy_idx    (origin_idx + dev_idx, the calendar diagonal)
      - actual    (observed incremental)
      - fitted    (fitted incremental)
      - residual  (standardised Pearson, hat-matrix adjusted)

    Cells with non-positive fitted incrementals or where the actual
    incremental is undefined are dropped.

    The hat-matrix adjustment follows the standard ODP bootstrap formulation:
        r_std = (a − f) / sqrt(phi * f) * sqrt(n / (n - p))
    where n is the number of observed cells and p is the number of free
    parameters (origin + dev factors), p = n_origin + (n_dev − 1).
    """
    paid = tri["paid_loss"].values[0, 0]
    inc_actual = _to_incremental(paid)
    inc_fitted = _chain_ladder_fitted(paid)

    rows = []
    for i in range(paid.shape[0]):
        for j in range(paid.shape[1]):
            a = inc_actual[i, j]
            f = inc_fitted[i, j]
            if np.isnan(a) or np.isnan(f) or f <= 0:
                continue
            rows.append(
                {
                    "origin_idx": i,
                    "dev_idx": j,
                    "cy_idx": i + j,
                    "actual": float(a),
                    "fitted": float(f),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["origin_idx", "dev_idx", "cy_idx", "actual", "fitted", "residual"]
        )
    df = pd.DataFrame(rows)
    n = len(df)
    n_origin, n_dev = paid.shape
    p = n_origin + (n_dev - 1)
    raw_pearson = (df["actual"] - df["fitted"]) / np.sqrt(df["fitted"])
    # Pearson dispersion phi (Shapland eq. 2.1.4).
    if n - p > 0:
        phi = float(np.sum(raw_pearson**2) / (n - p))
    else:
        phi = float(np.var(raw_pearson, ddof=0))
    # Hat-matrix adjustment factor.
    if n - p > 0:
        adj = np.sqrt(n / (n - p))
    else:
        adj = 1.0
    # Guard: if all raw residuals are essentially zero (perfect fit), return zeros
    # directly to avoid dividing near-zero by clamped near-zero phi.
    if np.allclose(raw_pearson, 0, atol=1e-9):
        df["residual"] = 0.0
        return df
    df["residual"] = raw_pearson / np.sqrt(phi) * adj
    return df


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------


def cache_path(name: str) -> Path:
    """Return the absolute path of a parquet cache file by short name."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / name


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def fisher_z_mean(rs: np.ndarray) -> float:
    """Average a vector of Pearson correlations via Fisher z-transform.

    Clips values to (-1+eps, 1-eps) to keep arctanh finite.
    """
    rs = np.asarray(rs, dtype=float)
    rs = rs[np.isfinite(rs)]
    if rs.size == 0:
        return float("nan")
    eps = 1e-9
    rs = np.clip(rs, -1.0 + eps, 1.0 - eps)
    z = np.arctanh(rs)
    return float(np.tanh(np.mean(z)))


# ---------------------------------------------------------------------------
# Cross-company calendar-diagonal correlation
# ---------------------------------------------------------------------------


def rho_from_residual_panel(
    panel: np.ndarray,
    min_pairs: int = 20,
    max_diag_distance: int = 4,
) -> dict:
    """Compute calendar-diagonal residual correlation by pooling across companies.

    Parameters
    ----------
    panel : ndarray, shape (n_companies, n_origin, n_dev)
        Standardised Pearson residuals; NaN for unobserved cells.
    min_pairs : int
        Minimum number of jointly-observed companies required for a cell-pair
        correlation to be included in the Fisher-z aggregation.
    max_diag_distance : int
        Largest cy_diff to report (we report 0 .. max_diag_distance).

    Returns
    -------
    dict with keys:
        rho           : float — recommended same-diagonal correlation (r at d=0)
        r_by_d        : dict[int, float] — Fisher-z mean correlation by cy_diff
        n_pairs_by_d  : dict[int, int]   — number of cell-pairs aggregated per d
    """
    n_companies, n_origin, n_dev = panel.shape
    # Flatten the (origin, dev) cells into a single index 0..n_cells-1.
    cell_index = [(i, j) for i in range(n_origin) for j in range(n_dev)]
    n_cells = len(cell_index)
    flat = panel.reshape(n_companies, n_cells)  # (n_companies, n_cells)

    # Pre-compute calendar diagonal for each cell.
    cy = np.array([i + j for (i, j) in cell_index])

    pair_corrs_by_d: dict[int, list[float]] = {d: [] for d in range(max_diag_distance + 1)}

    for a in range(n_cells):
        col_a = flat[:, a]
        for b in range(a + 1, n_cells):
            d = abs(int(cy[a] - cy[b]))
            if d > max_diag_distance:
                continue
            col_b = flat[:, b]
            mask = ~np.isnan(col_a) & ~np.isnan(col_b)
            n_pairs = int(mask.sum())
            if n_pairs < min_pairs:
                continue
            xa = col_a[mask]
            xb = col_b[mask]
            sd_a = xa.std(ddof=1)
            sd_b = xb.std(ddof=1)
            if sd_a == 0 or sd_b == 0:
                continue
            r = float(np.corrcoef(xa, xb)[0, 1])
            pair_corrs_by_d[d].append(r)

    r_by_d = {d: fisher_z_mean(np.array(v)) for d, v in pair_corrs_by_d.items() if v}
    n_pairs_by_d = {d: len(v) for d, v in pair_corrs_by_d.items()}
    rho = r_by_d.get(0, float("nan"))
    return {"rho": rho, "r_by_d": r_by_d, "n_pairs_by_d": n_pairs_by_d}
