"""Method wrappers for the Meyers backtest.

Each wrapper mirrors the signature of reservetestr.testr_mack_chainladder:

    fn(train_triangles, test_triangles, loss_type, actual_ultimates, **kwargs)
        -> dict | None

The returned dict always has the schema:
    {
        actual_ultimate, actual_unpaid,
        mean_ultimate_est, mean_unpaid_est,
        stddev_est, cv_unpaid_est,
        implied_pctl,
        status,          # "ok", "skipped:negative_incrementals", "error:<Type>:<msg>", etc.
    }
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional

import chainladder as cl
import numpy as np

from reservetestr.utils import latest_cumulative_sum, safe_divide

LossTypeMapping = Dict[str, Optional[cl.Triangle]]

_NAN_RESULT = {
    "actual_ultimate": float("nan"),
    "actual_unpaid": float("nan"),
    "mean_ultimate_est": float("nan"),
    "mean_unpaid_est": float("nan"),
    "stddev_est": float("nan"),
    "cv_unpaid_est": float("nan"),
    "implied_pctl": float("nan"),
}


def _negative_incrementals_skip(actual_ultimates: Optional[dict], loss_type: str, train_triangles: LossTypeMapping) -> dict:
    """Return a skip result for negative-incrementals failure."""
    actual_ultimate = _resolve_actual(actual_ultimates, loss_type)
    if actual_ultimate is None:
        actual_ultimate = float("nan")
    tri = train_triangles.get(loss_type)
    latest_observed = latest_cumulative_sum(tri) if tri is not None else float("nan")
    actual_unpaid = actual_ultimate - latest_observed if (
        np.isfinite(actual_ultimate) and np.isfinite(latest_observed)
    ) else float("nan")
    return {
        "actual_ultimate": actual_ultimate,
        "actual_unpaid": actual_unpaid,
        "mean_ultimate_est": float("nan"),
        "mean_unpaid_est": float("nan"),
        "stddev_est": float("nan"),
        "cv_unpaid_est": float("nan"),
        "implied_pctl": float("nan"),
        "status": "skipped:negative_incrementals",
    }


def _has_negative_incrementals(triangle: cl.Triangle) -> bool:
    """Return True if the triangle has any negative incremental values."""
    try:
        if triangle.is_cumulative:
            inc = triangle.cum_to_incr()
        else:
            inc = triangle
        vals = np.asarray(inc.values, dtype=float)
        finite = vals[np.isfinite(vals)]
        return bool((finite < 0).any())
    except Exception:
        return False


def _resolve_actual(actual_ultimates: Optional[dict], loss_type: str) -> Optional[float]:
    if not actual_ultimates:
        return None
    val = actual_ultimates.get(loss_type)
    if val is None or np.isnan(val):
        return None
    return float(val)


def _get_triangle(
    triangles: LossTypeMapping, loss_type: str
) -> Optional[cl.Triangle]:
    if loss_type not in triangles:
        raise ValueError(f"Unknown loss_type {loss_type!r}")
    return triangles[loss_type]


def _empirical_pctl(samples: np.ndarray, actual: float) -> float:
    """Empirical CDF P(X <= actual) from a sample array."""
    samples = np.asarray(samples, dtype=float)
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite <= actual))


# ---------------------------------------------------------------------------
# CorrelatedBootstrapODP wrapper
# ---------------------------------------------------------------------------


def testr_correlated_bootstrap_odp(
    train_triangles: LossTypeMapping,
    test_triangles: LossTypeMapping,
    loss_type: str = "paid",
    actual_ultimates: Optional[dict] = None,
    line: str = "",
    n_sims: int = 1000,
    hat_adj: bool = True,
    random_state: int = 22,
    **kwargs,
) -> Optional[dict]:
    """Back-test wrapper for CorrelatedBootstrapODPSample.

    Parameters
    ----------
    line : str
        Meyers line name (used to look up rho from the prior cache).
    """
    try:
        from bayesianchainladder import CorrelatedBootstrapODPSample
        from _common import load_rho_for_line

        triangle = _get_triangle(train_triangles, loss_type)
        if triangle is None:
            return None

        rho = load_rho_for_line(line) if line else 0.0

        bootstrap = CorrelatedBootstrapODPSample(
            n_sims=n_sims,
            rho=rho,
            hat_adj=hat_adj,
            random_state=random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bootstrap.fit(triangle)
            resampled = bootstrap.transform(triangle)

        # Fit chain ladder to each simulated triangle and extract total ultimates
        model = cl.Chainladder().fit(resampled)
        ult_vals = np.asarray(model.ultimate_.values, dtype=float)
        # shape: (n_sims, 1, n_origin, n_dev) — sum over origins
        samples = np.nansum(ult_vals, axis=(1, 2, 3))  # (n_sims,)

        if samples.size == 0:
            return None

        mean_ultimate = float(np.nanmean(samples))
        stddev_est = float(np.nanstd(samples, ddof=1)) if samples.size > 1 else float("nan")
        latest_observed = latest_cumulative_sum(triangle)
        actual_ultimate = _resolve_actual(actual_ultimates, loss_type)
        if actual_ultimate is None:
            test_tri = _get_triangle(test_triangles, loss_type)
            if test_tri is None:
                return None
            actual_ultimate = latest_cumulative_sum(test_tri)

        actual_unpaid = actual_ultimate - latest_observed
        mean_unpaid_est = mean_ultimate - latest_observed
        cv_unpaid_est = safe_divide(stddev_est, mean_unpaid_est)
        implied_pctl = _empirical_pctl(samples, actual_ultimate)

        return {
            "actual_ultimate": actual_ultimate,
            "actual_unpaid": actual_unpaid,
            "mean_ultimate_est": mean_ultimate,
            "mean_unpaid_est": mean_unpaid_est,
            "stddev_est": stddev_est,
            "cv_unpaid_est": cv_unpaid_est,
            "implied_pctl": implied_pctl,
            "status": "ok",
        }
    except Exception as e:
        return {**_NAN_RESULT, "status": f"error:{type(e).__name__}:{str(e)[:100]}"}


# ---------------------------------------------------------------------------
# BayesianCSR wrapper
# ---------------------------------------------------------------------------


def testr_bayesian_csr(
    train_triangles: LossTypeMapping,
    test_triangles: LossTypeMapping,
    loss_type: str = "paid",
    actual_ultimates: Optional[dict] = None,
    line: str = "",
    group_id: int = 0,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    target_accept: float = 0.95,
    random_seed: int = 22,
    **kwargs,
) -> Optional[dict]:
    """Back-test wrapper for BayesianCSR.

    Parameters
    ----------
    line : str
        Meyers line name (used to look up priors and exposure).
    group_id : int
        Meyers group_id (used to load the net earned premium exposure triangle).
    """
    try:
        from bayesianchainladder import BayesianCSR
        from _common import load_csr_priors_for_line, load_exposure_triangle

        triangle = _get_triangle(train_triangles, loss_type)
        if triangle is None:
            return None

        # Build premium triangle from Meyers exposure data
        prem_tri = load_exposure_triangle(line, group_id)

        # Load line-specific priors
        priors = load_csr_priors_for_line(line) if line else None

        model = BayesianCSR(
            priors=priors,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(triangle, premium_triangle=prem_tri)

        # reserves_posterior_ has dims (origin, sample), values = IBNR per origin
        reserves = model.reserves_posterior_  # xr.DataArray (origin, sample)
        total_ibnr_samples = np.asarray(reserves.sum(dim="origin").values, dtype=float)

        # latest observed across all origins
        latest_observed = latest_cumulative_sum(triangle)
        actual_ultimate = _resolve_actual(actual_ultimates, loss_type)
        if actual_ultimate is None:
            test_tri = _get_triangle(test_triangles, loss_type)
            if test_tri is None:
                return None
            actual_ultimate = latest_cumulative_sum(test_tri)

        total_ult_samples = total_ibnr_samples + latest_observed
        mean_ultimate = float(np.nanmean(total_ult_samples))
        stddev_est = float(np.nanstd(total_ult_samples, ddof=1))
        actual_unpaid = actual_ultimate - latest_observed
        mean_unpaid_est = mean_ultimate - latest_observed
        cv_unpaid_est = safe_divide(stddev_est, mean_unpaid_est)
        implied_pctl = _empirical_pctl(total_ult_samples, actual_ultimate)

        return {
            "actual_ultimate": actual_ultimate,
            "actual_unpaid": actual_unpaid,
            "mean_ultimate_est": mean_ultimate,
            "mean_unpaid_est": mean_unpaid_est,
            "stddev_est": stddev_est,
            "cv_unpaid_est": cv_unpaid_est,
            "implied_pctl": implied_pctl,
            "status": "ok",
        }
    except Exception as e:
        return {**_NAN_RESULT, "status": f"error:{type(e).__name__}:{str(e)[:100]}"}


# ---------------------------------------------------------------------------
# Generic BayesianChainLadderGLM wrapper
# ---------------------------------------------------------------------------


def _testr_bayesian_glm(
    train_triangles: LossTypeMapping,
    test_triangles: LossTypeMapping,
    loss_type: str,
    actual_ultimates: Optional[dict],
    line: str,
    group_id: int,
    spec: str,
    formula: str,
    family: str,
    link: Optional[str],
    response_per_exposure: bool = False,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    target_accept: float = 0.95,
    random_seed: int = 22,
    **kwargs,
) -> Optional[dict]:
    """Generic BayesianChainLadderGLM back-test wrapper (internal)."""
    try:
        from bayesianchainladder import BayesianChainLadderGLM
        from _common import load_glm_priors_for_line, load_exposure_triangle

        triangle = _get_triangle(train_triangles, loss_type)
        if triangle is None:
            return None

        # Gamma family cannot handle negative incrementals — skip early.
        if family == "gamma" and _has_negative_incrementals(triangle):
            return _negative_incrementals_skip(actual_ultimates, loss_type, train_triangles)

        # Load premium/exposure triangle
        prem_tri = load_exposure_triangle(line, group_id)

        # Load priors (best effort — fall back to defaults if they fail)
        priors = None
        if line:
            try:
                priors = load_glm_priors_for_line(line, spec)
            except Exception:
                priors = None  # fall back to adaptive defaults

        model = BayesianChainLadderGLM(
            formula=formula,
            family=family,
            link=link,
            exposure="net_earned_premium",
            response_per_exposure=response_per_exposure,
            priors=priors,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(triangle, exposure_triangle=prem_tri)

        # reserves_posterior_: (origin, sample), values = IBNR
        reserves = model.reserves_posterior_
        total_ibnr_samples = np.asarray(reserves.sum(dim="origin").values, dtype=float)

        latest_observed = latest_cumulative_sum(triangle)
        actual_ultimate = _resolve_actual(actual_ultimates, loss_type)
        if actual_ultimate is None:
            test_tri = _get_triangle(test_triangles, loss_type)
            if test_tri is None:
                return None
            actual_ultimate = latest_cumulative_sum(test_tri)

        total_ult_samples = total_ibnr_samples + latest_observed
        mean_ultimate = float(np.nanmean(total_ult_samples))
        stddev_est = float(np.nanstd(total_ult_samples, ddof=1))
        actual_unpaid = actual_ultimate - latest_observed
        mean_unpaid_est = mean_ultimate - latest_observed
        cv_unpaid_est = safe_divide(stddev_est, mean_unpaid_est)
        implied_pctl = _empirical_pctl(total_ult_samples, actual_ultimate)

        return {
            "actual_ultimate": actual_ultimate,
            "actual_unpaid": actual_unpaid,
            "mean_ultimate_est": mean_ultimate,
            "mean_unpaid_est": mean_unpaid_est,
            "stddev_est": stddev_est,
            "cv_unpaid_est": cv_unpaid_est,
            "implied_pctl": implied_pctl,
            "status": "ok",
        }
    except Exception as e:
        return {**_NAN_RESULT, "status": f"error:{type(e).__name__}:{str(e)[:100]}"}


# ---------------------------------------------------------------------------
# Public GLM wrappers
# ---------------------------------------------------------------------------

# M2: gamma + log, fixed categorical origin + B-spline on dev_idx
_FORMULA_M2 = "incremental ~ 1 + C(origin) + bs(dev_idx, df=4)"

# M5_cal: gamma + log, random-intercept origin + B-spline dev + random calendar
_FORMULA_M5_CAL = "incremental ~ 1 + (1 | origin) + bs(dev_idx, df=4) + (1 | calendar)"

# MT5_cal: same formula, but t + identity on loss-ratio scale
_FORMULA_MT5_CAL = "incremental ~ 1 + (1 | origin) + bs(dev_idx, df=4) + (1 | calendar)"


def testr_glm_m2(
    train_triangles: LossTypeMapping,
    test_triangles: LossTypeMapping,
    loss_type: str = "paid",
    actual_ultimates: Optional[dict] = None,
    line: str = "",
    group_id: int = 0,
    **kwargs,
) -> Optional[dict]:
    """BCL_GLM_M2: gamma + log, C(origin) + bs(dev_idx, df=4), exposure offset."""
    try:
        return _testr_bayesian_glm(
            train_triangles=train_triangles,
            test_triangles=test_triangles,
            loss_type=loss_type,
            actual_ultimates=actual_ultimates,
            line=line,
            group_id=group_id,
            spec="M2",
            formula=_FORMULA_M2,
            family="gamma",
            link="log",
            response_per_exposure=False,
            **kwargs,
        )
    except Exception as e:
        return {**_NAN_RESULT, "status": f"error:{type(e).__name__}:{str(e)[:100]}"}


def testr_glm_m5_cal(
    train_triangles: LossTypeMapping,
    test_triangles: LossTypeMapping,
    loss_type: str = "paid",
    actual_ultimates: Optional[dict] = None,
    line: str = "",
    group_id: int = 0,
    **kwargs,
) -> Optional[dict]:
    """BCL_GLM_M5_cal: gamma + log, (1|origin) + bs(dev_idx,4) + (1|calendar), exposure offset."""
    try:
        return _testr_bayesian_glm(
            train_triangles=train_triangles,
            test_triangles=test_triangles,
            loss_type=loss_type,
            actual_ultimates=actual_ultimates,
            line=line,
            group_id=group_id,
            spec="M5_cal",
            formula=_FORMULA_M5_CAL,
            family="gamma",
            link="log",
            response_per_exposure=False,
            **kwargs,
        )
    except Exception as e:
        return {**_NAN_RESULT, "status": f"error:{type(e).__name__}:{str(e)[:100]}"}


def testr_glm_mt5_cal(
    train_triangles: LossTypeMapping,
    test_triangles: LossTypeMapping,
    loss_type: str = "paid",
    actual_ultimates: Optional[dict] = None,
    line: str = "",
    group_id: int = 0,
    **kwargs,
) -> Optional[dict]:
    """BCL_GLM_MT5_cal: t + identity, loss-ratio, (1|origin) + bs(dev_idx,4) + (1|calendar)."""
    try:
        return _testr_bayesian_glm(
            train_triangles=train_triangles,
            test_triangles=test_triangles,
            loss_type=loss_type,
            actual_ultimates=actual_ultimates,
            line=line,
            group_id=group_id,
            spec="MT5_cal",
            formula=_FORMULA_MT5_CAL,
            family="t",
            link="identity",
            response_per_exposure=True,
            **kwargs,
        )
    except Exception as e:
        return {**_NAN_RESULT, "status": f"error:{type(e).__name__}:{str(e)[:100]}"}
