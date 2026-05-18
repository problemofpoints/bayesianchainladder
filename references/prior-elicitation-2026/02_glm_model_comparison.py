"""02_glm_model_comparison.py — WAIC/LOO comparison of GLM functional forms.

Specs:
  M1: incremental ~ 1 + C(origin) + C(dev)                           (full categorical)
  M2: incremental ~ 1 + C(origin) + bs(dev_idx, df=4)                (spline on dev ordinal index)
  M3: incremental ~ 1 + bs(origin, df=3) + C(dev)                    (restricted origin)
  M4: lives in 02b — fits one Bambi model per line with (1 | snl_id)
  M5: incremental ~ 1 + (1 | origin) + bs(dev_idx, df=4)             (random-effect origin)
  M2_cal: incremental ~ 1 + C(origin) + bs(dev_idx, df=4) + (1 | calendar)   (M2 + calendar RE)
  M5_cal: incremental ~ 1 + (1 | origin) + bs(dev_idx, df=4) + (1 | calendar) (M5 + calendar RE)

M2 uses dev_idx (1-based integer ordinal, not raw dev-months) to avoid the
pathological posterior geometry that the raw dev-months B-spline produced under
inverse-link gamma. Under gamma+log link M2_devidx_bs4 mixes cleanly (all
max_rhat < 1.01, 0 divergences across 3 PPAL triangles in 06_diagnose_m2.py).

For M1/M2/M3 we fit one model per (line, snl_id) over the 24-triangle stratified
sample. WAIC and LOO are extracted from the fitted idata.

Family: gamma, link: log (explicit; Bambi's default gamma link is inverse).
Exposure: net_earned_premium. Light fits: 1000 draws / 1000 tune /
2 chains / target_accept=0.95. Random seed deterministic per (line, snl_id, spec).

Output:
  cache/glm_per_triangle_fits_v2.parquet — one row per (line, snl_id, spec)
  (v2 adds reserve stats: ibnr_total_median, ibnr_total_cv, ibnr_p10/p90,
   ult_total_median, ult_total_cv, pct_error_vs_booked, sum_log_ep_obs, n_obs)

Run: uv run python references/prior-elicitation-2026/02_glm_model_comparison.py
"""
from __future__ import annotations

import hashlib
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from bayesianchainladder import BayesianChainLadderGLM, compute_loo, compute_waic
from _common import (
    LINES,
    cache_path,
    load_full_triangle,
    select_sample,
)

SPECS: dict[str, str] = {
    "M1_cat": "incremental ~ 1 + C(origin) + C(dev)",
    # M2: spline on the 1-based ordinal dev index (not raw dev-months).
    # dev_idx is auto-materialized by add_categorical_columns when the _idx
    # suffix is detected. Under gamma+log link this mixes cleanly (confirmed
    # in 06_diagnose_m2.py: all 9 smoke fits had max_rhat < 1.01, 0 divergences).
    "M2_devidx_bs4": "incremental ~ 1 + C(origin) + bs(dev_idx, df=4)",
    # df=2 is below the minimum of 3 for a cubic B-spline without intercept;
    # df=3 is the smallest valid value.
    "M3_restorigin": "incremental ~ 1 + bs(origin, df=3) + C(dev)",
    # M5: random intercept by origin + spline on dev ordinal index.
    # With only 10 origin levels mixing may be imperfect — rhat flagged if > 1.5.
    "M5_origin_re": "incremental ~ 1 + (1 | origin) + bs(dev_idx, df=4)",
    # M2_cal: M2 + random calendar-period intercept (diagonal effect).
    "M2_cal": "incremental ~ 1 + C(origin) + bs(dev_idx, df=4) + (1 | calendar)",
    # M5_cal: M5 + random calendar-period intercept.
    "M5_cal": "incremental ~ 1 + (1 | origin) + bs(dev_idx, df=4) + (1 | calendar)",
    # MT2: t family, identity link, loss-ratio response (response_per_exposure=True).
    # Same functional form as M2_devidx_bs4 but on loss-ratio scale — allows
    # negative residuals and thick tails. LOO is on loss-ratio scale and NOT
    # directly comparable to the gamma+log specs above.
    "MT2": "incremental ~ 1 + C(origin) + bs(dev_idx, df=4)",
    # MT5_cal: t family, identity link, loss-ratio, with random calendar effect.
    "MT5_cal": "incremental ~ 1 + (1 | origin) + bs(dev_idx, df=4) + (1 | calendar)",
}


def _seed_for(line: str, snl_id: str, spec: str) -> int:
    """Deterministic per-(line, snl_id, spec) seed using a stable hash."""
    h = hashlib.md5(f"{line}|{snl_id}|{spec}".encode()).hexdigest()
    return int(h, 16) % (2**31)


def _spec_kwargs(spec_name: str) -> dict:
    """Per-spec family/link/response_per_exposure kwargs."""
    if spec_name.startswith("MT"):
        return {
            "family": "t",
            "link": "identity",
            "exposure": "net_earned_premium",
            "response_per_exposure": True,
        }
    return {
        "family": "gamma",
        "link": "log",
        "exposure": "net_earned_premium",
        "response_per_exposure": False,
    }


def _fit_one(triangle, formula: str, spec_name: str, seed: int) -> dict:
    """Fit a single BayesianChainLadderGLM and return diagnostic + reserve stats.

    The input triangle is multi-vdim (paid_loss, net_earned_premium, …).
    We split it into a single-vdim paid_loss triangle and a separate
    exposure triangle so that triangle_to_dataframe sees (1, 1, n_orig, n_dev).

    MT* specs use family='t', link='identity', response_per_exposure=True
    (loss-ratio scale). Their LOO is NOT directly comparable to gamma+log specs
    (different response units / reference densities) without a Jacobian correction
    — see sum_log_ep_obs in the returned dict.
    """
    import arviz as az

    paid_tri = triangle["paid_loss"]
    prem_tri = triangle["net_earned_premium"]

    kwargs = _spec_kwargs(spec_name)
    model = BayesianChainLadderGLM(
        formula=formula,
        draws=1000,
        tune=1000,
        chains=2,
        target_accept=0.95,
        random_seed=seed,
        **kwargs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(paid_tri, exposure_triangle=prem_tri)

    waic = compute_waic(model.idata)
    loo = compute_loo(model.idata)
    # Convergence diagnostic: max r-hat across all named posterior variables.
    summ = az.summary(model.idata, kind="diagnostics")
    max_rhat = float(summ["r_hat"].max()) if "r_hat" in summ.columns else float("nan")

    # -------------------------------------------------------------------------
    # Reserve stats — total IBNR/Ultimate posterior summaries.
    # -------------------------------------------------------------------------
    reserves = model.reserves_posterior_  # xr.DataArray (origin, sample)
    total_ibnr_samples = reserves.sum(dim="origin").values.flatten()
    total_ibnr_samples = total_ibnr_samples[np.isfinite(total_ibnr_samples)]

    # Latest diagonal cumulative paid (dollar scale, already in paid_tri).
    latest_cum_total = float(
        np.nansum(paid_tri.latest_diagonal.values)
    )

    # booked ultimate = booked_ultimate_loss latest diagonal (if present).
    vdims = list(triangle.vdims)
    if "booked_ultimate_loss" in vdims:
        booked_ult_total = float(
            np.nansum(triangle["booked_ultimate_loss"].latest_diagonal.values)
        )
    else:
        booked_ult_total = float("nan")

    ibnr_median = float(np.median(total_ibnr_samples)) if total_ibnr_samples.size > 0 else float("nan")
    ibnr_cv = float(
        np.std(total_ibnr_samples, ddof=1) / max(abs(ibnr_median), 1.0)
    ) if total_ibnr_samples.size > 1 else float("nan")
    ibnr_p10 = float(np.percentile(total_ibnr_samples, 10)) if total_ibnr_samples.size > 0 else float("nan")
    ibnr_p90 = float(np.percentile(total_ibnr_samples, 90)) if total_ibnr_samples.size > 0 else float("nan")

    total_ult_samples = total_ibnr_samples + latest_cum_total
    ult_median = float(np.median(total_ult_samples)) if total_ult_samples.size > 0 else float("nan")
    ult_cv = float(
        np.std(total_ult_samples, ddof=1) / max(abs(ult_median), 1.0)
    ) if total_ult_samples.size > 1 else float("nan")

    pct_error_vs_booked = (
        float((ult_median - booked_ult_total) / booked_ult_total)
        if np.isfinite(booked_ult_total) and booked_ult_total > 0
        else float("nan")
    )

    # Sum of log(EP) over OBSERVED cells, for the Jacobian correction that
    # converts MT* LOO from loss-ratio scale to dollar-equivalent scale:
    #   log p(dollar) = log q(loss-ratio) - log(EP)
    #   sum_n log p = sum_n log q - sum_log_ep_obs
    data = model.data_  # long-format observed data
    # net_earned_premium is the exposure column used for both gamma and MT specs.
    ep_col_obs = "net_earned_premium"
    if ep_col_obs in data.columns:
        ep_obs_arr = np.asarray(data[ep_col_obs].values, dtype=float)
        pos_ep = ep_obs_arr[ep_obs_arr > 0]
        sum_log_ep_obs = float(np.sum(np.log(pos_ep))) if pos_ep.size > 0 else float("nan")
    else:
        sum_log_ep_obs = float("nan")

    return {
        "waic": float(waic.elpd_waic),
        "p_waic": float(waic.p_waic),
        "loo": float(loo.elpd_loo),
        "p_loo": float(loo.p_loo),
        "max_rhat": max_rhat,
        "ibnr_total_median": ibnr_median,
        "ibnr_total_cv": ibnr_cv,
        "ibnr_total_p10": ibnr_p10,
        "ibnr_total_p90": ibnr_p90,
        "ult_total_median": ult_median,
        "ult_total_cv": ult_cv,
        "booked_ult_total": booked_ult_total,
        "pct_error_vs_booked": pct_error_vs_booked,
        "sum_log_ep_obs": sum_log_ep_obs,
        "n_obs": len(data),
    }


def _result_path() -> Path:
    return cache_path("glm_per_triangle_fits_v2.parquet")


def _existing_keys() -> set[tuple[str, str, str]]:
    p = _result_path()
    if not p.exists():
        return set()
    df = pd.read_parquet(p)
    return set(zip(df["line"], df["snl_id"], df["spec"]))


def _append_row(row: dict) -> None:
    p = _result_path()
    new = pd.DataFrame([row])
    if p.exists():
        existing = pd.read_parquet(p)
        out = pd.concat([existing, new], ignore_index=True)
    else:
        out = new
    out.to_parquet(p, index=False)


def main() -> int:
    print("Loading triangle JSON…", flush=True)
    full = load_full_triangle()
    done = _existing_keys()
    print(f"  resuming from {len(done)} cached fits", flush=True)

    for line in LINES:
        snl_ids = select_sample(full, line)
        sub_full = full[full["line_of_business"] == line]
        for snl_id in snl_ids:
            sub_tri = sub_full[sub_full["snl_id"] == snl_id]
            for spec_name, formula in SPECS.items():
                key = (line, snl_id, spec_name)
                if key in done:
                    continue
                seed = _seed_for(line, snl_id, spec_name)
                print(f"  {line:5s} {snl_id:14s} {spec_name:14s} seed={seed}", flush=True)
                try:
                    res = _fit_one(sub_tri, formula, spec_name, seed)
                    status = "ok"
                except Exception as e:  # noqa: BLE001
                    res = {
                        "waic": float("nan"),
                        "p_waic": float("nan"),
                        "loo": float("nan"),
                        "p_loo": float("nan"),
                        "max_rhat": float("nan"),
                        "ibnr_total_median": float("nan"),
                        "ibnr_total_cv": float("nan"),
                        "ibnr_total_p10": float("nan"),
                        "ibnr_total_p90": float("nan"),
                        "ult_total_median": float("nan"),
                        "ult_total_cv": float("nan"),
                        "booked_ult_total": float("nan"),
                        "pct_error_vs_booked": float("nan"),
                        "sum_log_ep_obs": float("nan"),
                        "n_obs": 0,
                    }
                    status = f"error: {type(e).__name__}: {e}"
                _append_row(
                    {
                        "line": line,
                        "snl_id": snl_id,
                        "spec": spec_name,
                        "status": status,
                        **res,
                    }
                )
                done.add(key)

    print(f"Done. {len(done)} fits cached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
