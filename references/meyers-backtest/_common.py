"""Shared utilities for the meyers-backtest analysis.

Imports line mappings and functions to load prior-elicitation-2026 cache data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Line mappings
# ---------------------------------------------------------------------------

MEYERS_TO_PRIOR_LINE = {
    "comauto": "CAL",
    "ppauto": "PPAL",
    "wkcomp": "WC",
    "othliab": "OLO",
}
PRIOR_TO_MEYERS_LINE = {v: k for k, v in MEYERS_TO_PRIOR_LINE.items()}
MEYERS_LINES = list(MEYERS_TO_PRIOR_LINE.keys())

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ANALYSIS_DIR = Path(__file__).resolve().parent
CACHE_DIR = ANALYSIS_DIR / "cache"
FIGURES_DIR = ANALYSIS_DIR / "figures"

# Prior-elicitation-2026 cache directory (sibling to meyers-backtest)
PRIOR_CACHE_DIR = ANALYSIS_DIR.parent / "prior-elicitation-2026" / "cache"

# Meyers exposure file (bundled with reservetestr-python)
_MEYERS_EXPOSURE_FILE: Path | None = None


def _get_meyers_exposure_path() -> Path:
    global _MEYERS_EXPOSURE_FILE
    if _MEYERS_EXPOSURE_FILE is not None:
        return _MEYERS_EXPOSURE_FILE
    try:
        from importlib import resources
        p = resources.files("reservetestr.data").joinpath("meyers_exposure.csv")
        _MEYERS_EXPOSURE_FILE = Path(str(p))
        return _MEYERS_EXPOSURE_FILE
    except Exception:
        # Fallback: resolve relative to reservetestr package location
        import reservetestr
        base = Path(reservetestr.__file__).parent
        _MEYERS_EXPOSURE_FILE = base / "data" / "meyers_exposure.csv"
        return _MEYERS_EXPOSURE_FILE


# ---------------------------------------------------------------------------
# CSR priors loader
# ---------------------------------------------------------------------------


def load_csr_priors_for_line(line: str) -> dict:
    """Return BayesianCSR priors dict for a Meyers line.

    Reads prior-elicitation-2026/cache/csr_fits.parquet, maps the Meyers line
    to the prior-elicitation line code, filters to converged fits (max_rhat < 1.1),
    and synthesises priors following the same logic as 05_synthesize._aggregate_csr().

    Returns a dict suitable for BayesianCSR(priors=...).
    """
    prior_line = MEYERS_TO_PRIOR_LINE.get(line)
    if prior_line is None:
        raise ValueError(f"Unknown Meyers line: {line!r}. Known: {MEYERS_LINES}")

    csr_path = PRIOR_CACHE_DIR / "csr_fits.parquet"
    if not csr_path.exists():
        raise FileNotFoundError(f"CSR fits cache not found: {csr_path}")

    df = pd.read_parquet(csr_path)
    df = df[(df["line"] == prior_line) & (df["status"] == "ok") & (df["max_rhat"] < 1.1)]

    if df.empty:
        raise ValueError(
            f"No converged CSR fits for prior line {prior_line!r} "
            f"(Meyers line {line!r})"
        )

    # logelr: Normal(mean_of_posterior_means, 1.5 * sd_of_posterior_means)
    logelr_mu = float(df["logelr_mean"].mean())
    logelr_sd = float(df["logelr_mean"].std(ddof=1)) if len(df) > 1 else float(df["logelr_sd"].mean())
    logelr_sigma = 1.5 * logelr_sd

    # gamma: Normal(mean, 1.5 * sd), floor sigma at 0.01
    gamma_mu = float(df["gamma_mean"].mean())
    gamma_sd = float(df["gamma_mean"].std(ddof=1)) if len(df) > 1 else float(df["gamma_sd"].mean())
    gamma_sigma = max(1.5 * gamma_sd, 0.01)

    # No specific a_ig / sig priors built into the return dict here;
    # BayesianCSR falls back to its own defaults for those.
    # We only inform the parameters where the prior elicitation was explicit.

    return {
        "logelr": {"mu": logelr_mu, "sigma": logelr_sigma},
        "gamma": {"mu": gamma_mu, "sigma": gamma_sigma},
    }


# ---------------------------------------------------------------------------
# GLM priors loader
# ---------------------------------------------------------------------------

_SPEC_TO_PARQUET = {
    "M2": "m1_posteriors.parquet",           # M2 (C(origin)+bs(dev_idx)) uses M1 posterior as proxy
    "M5_cal": "m5cal_posteriors.parquet",
    "MT5_cal": "mt5cal_posteriors.parquet",
}


def load_glm_priors_for_line(line: str, spec: str) -> dict:
    """Return a Bambi priors dict for BayesianChainLadderGLM for a Meyers line.

    Parameters
    ----------
    line : str
        Meyers line (e.g. "comauto").
    spec : str
        One of "M2", "M5_cal", "MT5_cal".

    Returns
    -------
    dict
        Keys are Bambi parameter names; values are bambi.Prior objects.
    """
    import bambi as bmb

    prior_line = MEYERS_TO_PRIOR_LINE.get(line)
    if prior_line is None:
        raise ValueError(f"Unknown Meyers line: {line!r}. Known: {MEYERS_LINES}")

    parquet_name = _SPEC_TO_PARQUET.get(spec)
    if parquet_name is None:
        raise ValueError(f"Unknown spec {spec!r}. Known: {list(_SPEC_TO_PARQUET)}")

    cache_path = PRIOR_CACHE_DIR / parquet_name
    if not cache_path.exists():
        raise FileNotFoundError(f"Posteriors cache not found: {cache_path}")

    df = pd.read_parquet(cache_path)
    df = df[(df["line"] == prior_line) & (df["status"] == "ok") & (df["max_rhat"] < 1.1)]

    if df.empty:
        raise ValueError(
            f"No converged fits for spec={spec!r}, prior_line={prior_line!r}"
        )

    priors: dict = {}

    if spec == "M2":
        # m1_posteriors: intercept_mean, intercept_sd, alpha_mean, alpha_sd,
        # origin_effect_sd_p50, dev_effect_sd_p50
        int_mean = float(df["intercept_mean"].mean())
        int_sd = float(df["intercept_sd"].mean())
        priors["Intercept"] = bmb.Prior("Normal", mu=int_mean, sigma=1.5 * int_sd)
        # C(origin) effects
        origin_sd = float(df["origin_effect_sd_p50"].median())
        if np.isfinite(origin_sd) and origin_sd > 0:
            priors["C(origin)"] = bmb.Prior("Normal", mu=0.0, sigma=1.5 * origin_sd)
        # dev spline coefficients: use dev_effect_sd_p50 as scale
        dev_sd = float(df["dev_effect_sd_p50"].median())
        if np.isfinite(dev_sd) and dev_sd > 0:
            priors["bs(dev_idx, df=4)"] = bmb.Prior("Normal", mu=0.0, sigma=1.5 * dev_sd)

    elif spec == "M5_cal":
        # m5cal_posteriors: intercept_mean, intercept_sd, alpha_mean,
        # origin_sigma_median, calendar_sigma_median, spline_coef_sd
        int_mean = float(df["intercept_mean"].mean())
        int_sd = float(df["intercept_sd"].mean())
        priors["Intercept"] = bmb.Prior("Normal", mu=int_mean, sigma=1.5 * int_sd)

        origin_sigma = float(df["origin_sigma_median"].median())
        if np.isfinite(origin_sigma) and origin_sigma > 0:
            priors["1|origin"] = bmb.Prior(
                "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1.5 * origin_sigma)
            )

        cal_sigma = float(df["calendar_sigma_median"].median())
        if np.isfinite(cal_sigma) and cal_sigma > 0:
            priors["1|calendar"] = bmb.Prior(
                "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1.5 * cal_sigma)
            )

        spline_sd = float(df["spline_coef_sd"].median())
        if np.isfinite(spline_sd) and spline_sd > 0:
            priors["bs(dev_idx, df=4)"] = bmb.Prior("Normal", mu=0.0, sigma=1.5 * spline_sd)

    elif spec == "MT5_cal":
        # mt5cal_posteriors: intercept_mean, intercept_sd, sigma_mean, nu_mean,
        # origin_sigma_median, calendar_sigma_median, spline_coef_sd
        int_mean = float(df["intercept_mean"].mean())
        int_sd = float(df["intercept_sd"].mean())
        priors["Intercept"] = bmb.Prior("Normal", mu=int_mean, sigma=1.5 * int_sd)

        sigma_med = float(df["sigma_mean"].median())
        if np.isfinite(sigma_med) and sigma_med > 0:
            priors["sigma"] = bmb.Prior("HalfNormal", sigma=1.5 * sigma_med)

        nu_med = float(df["nu_mean"].median())
        if np.isfinite(nu_med) and nu_med < 5:
            priors["nu"] = bmb.Prior("Gamma", alpha=2, beta=0.5)
        else:
            priors["nu"] = bmb.Prior("Gamma", alpha=2, beta=0.1)

        origin_sigma = float(df["origin_sigma_median"].median())
        if np.isfinite(origin_sigma) and origin_sigma > 0:
            priors["1|origin"] = bmb.Prior(
                "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1.5 * origin_sigma)
            )

        cal_sigma = float(df["calendar_sigma_median"].median())
        if np.isfinite(cal_sigma) and cal_sigma > 0:
            priors["1|calendar"] = bmb.Prior(
                "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1.5 * cal_sigma)
            )

        spline_sd = float(df["spline_coef_sd"].median())
        if np.isfinite(spline_sd) and spline_sd > 0:
            priors["bs(dev_idx, df=4)"] = bmb.Prior("Normal", mu=0.0, sigma=1.5 * spline_sd)

    return priors


# ---------------------------------------------------------------------------
# Rho loader
# ---------------------------------------------------------------------------


def load_rho_for_line(line: str) -> float:
    """Return the bootstrap-median rho for a Meyers line.

    Reads prior-elicitation-2026/cache/rho_by_line.parquet and returns
    the ``rho_median`` column for the mapped prior-elicitation line.
    """
    prior_line = MEYERS_TO_PRIOR_LINE.get(line)
    if prior_line is None:
        raise ValueError(f"Unknown Meyers line: {line!r}. Known: {MEYERS_LINES}")

    rho_path = PRIOR_CACHE_DIR / "rho_by_line.parquet"
    if not rho_path.exists():
        raise FileNotFoundError(f"Rho cache not found: {rho_path}")

    df = pd.read_parquet(rho_path)
    row = df[df["line"] == prior_line]
    if row.empty:
        raise ValueError(f"No rho entry for prior line {prior_line!r}")

    return float(row.iloc[0]["rho_median"])


# ---------------------------------------------------------------------------
# Meyers exposure loader
# ---------------------------------------------------------------------------


def load_exposure_triangle(line: str, group_id: int) -> "cl.Triangle":
    """Build a chainladder Triangle of net earned premium for a Meyers company.

    Returns a single-column Triangle with vdim ``net_earned_premium`` and
    10 origins (1988–1997), each with a single dev cell at dev=12 (the first
    development period only — CSR/GLM code extracts the first diagonal for
    the premium offset).

    Any origin with zero exposure is replaced with 1 to avoid log(0) errors
    (these are flagged in the source data as genuinely zero, not missing).
    """
    import chainladder as cl

    exp_path = _get_meyers_exposure_path()
    df = pd.read_csv(exp_path)

    sub = df[(df["line"] == line) & (df["group_id"] == group_id)].copy()
    if sub.empty:
        raise ValueError(
            f"No exposure data for line={line!r}, group_id={group_id}"
        )

    # Replace zero/negative exposure with 1 to avoid log errors
    sub["exposure"] = sub["exposure"].clip(lower=1.0)

    sub["origin"] = pd.PeriodIndex(sub["accident_year"].astype(int), freq="Y")
    # Use the same period as development so it occupies the first dev column
    sub["development"] = sub["origin"].copy()
    sub = sub.rename(columns={"exposure": "net_earned_premium"})

    tri = cl.Triangle(
        data=sub[["origin", "development", "net_earned_premium"]],
        origin="origin",
        development="development",
        columns=["net_earned_premium"],
        cumulative=True,
    )
    return tri["net_earned_premium"]
