"""13_glm_priors_m5cal.py — Mine M5_cal posteriors for per-spec GLM prior elicitation.

Fits incremental ~ 1 + (1 | origin) + bs(dev_idx, df=4) + (1 | calendar)
with family='gamma', link='log', exposure='net_earned_premium',
on the same 24 sampled triangles per line used in 02_glm_model_comparison.py.

Per fit, extracts:
  - Intercept posterior summary (mean, sd, p10, p90)
  - Gamma shape (alpha) posterior summary
  - 1|origin_sigma (random-intercept SD on origin RE)
  - 1|calendar_sigma (calendar RE SD)
  - Spline coefficient SD from bs(dev_idx, df=4) terms

Output: cache/m5cal_posteriors.parquet
Run: uv run python references/prior-elicitation-2026/13_glm_priors_m5cal.py
"""
from __future__ import annotations

import hashlib
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arviz as az
import numpy as np
import pandas as pd

from bayesianchainladder import BayesianChainLadderGLM
from _common import (
    LINES,
    cache_path,
    load_full_triangle,
    select_sample,
)

FORMULA = "incremental ~ 1 + (1 | origin) + bs(dev_idx, df=4) + (1 | calendar)"
SPEC = "M5_cal"


def _seed_for(line: str, snl_id: str) -> int:
    """Same seed logic as 02_glm_model_comparison for reproducibility."""
    h = hashlib.md5(f"{line}|{snl_id}|{SPEC}".encode()).hexdigest()
    return int(h, 16) % (2**31)


def _fit_and_extract(triangle, seed: int) -> dict:
    """Fit M5_cal (gamma, log, RE origin + calendar) and extract posterior summaries."""
    paid_tri = triangle["paid_loss"]
    prem_tri = triangle["net_earned_premium"]

    model = BayesianChainLadderGLM(
        formula=FORMULA,
        family="gamma",
        link="log",
        exposure="net_earned_premium",
        draws=1000,
        tune=1000,
        chains=2,
        target_accept=0.95,
        random_seed=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(paid_tri, exposure_triangle=prem_tri)

    idata = model.idata

    # --- Convergence diagnostic ---
    summ = az.summary(idata, kind="diagnostics")
    max_rhat = float(summ["r_hat"].max()) if "r_hat" in summ.columns else float("nan")

    # --- Extract posteriors from idata.posterior ---
    post = idata.posterior

    def _flat(var: str) -> np.ndarray | None:
        if var in post:
            return post[var].values.flatten()
        return None

    # Intercept.
    intercept_arr = _flat("Intercept")
    if intercept_arr is None:
        intercept_mean = intercept_sd = intercept_p10 = intercept_p90 = float("nan")
    else:
        intercept_mean = float(np.mean(intercept_arr))
        intercept_sd = float(np.std(intercept_arr))
        intercept_p10 = float(np.percentile(intercept_arr, 10))
        intercept_p90 = float(np.percentile(intercept_arr, 90))

    # Gamma shape (alpha) — scalar posterior parameter.
    alpha_arr = None
    for var in post.data_vars:
        vname = str(var)
        if "alpha" in vname.lower():
            arr = post[var].values
            if arr.ndim == 2:
                alpha_arr = arr.flatten()
                break
    if alpha_arr is None:
        alpha_mean = alpha_sd = alpha_p10 = alpha_p90 = float("nan")
    else:
        alpha_mean = float(np.mean(alpha_arr))
        alpha_sd = float(np.std(alpha_arr))
        alpha_p10 = float(np.percentile(alpha_arr, 10))
        alpha_p90 = float(np.percentile(alpha_arr, 90))

    # 1|origin sigma (random-intercept SD on origin).
    origin_sigma = float("nan")
    for var in post.data_vars:
        vname = str(var)
        if "origin" in vname.lower() and "sigma" in vname.lower():
            arr = post[var].values.flatten()
            origin_sigma = float(np.median(arr))
            break
    # Fallback: look for origin sigma via Bambi's standard naming "1|origin_sigma"
    if not np.isfinite(origin_sigma):
        for var in post.data_vars:
            vname = str(var)
            if "1|origin" in vname or "origin_sigma" in vname.lower():
                arr = post[var].values.flatten()
                origin_sigma = float(np.median(arr))
                break

    # 1|calendar sigma (calendar RE SD).
    calendar_sigma = float("nan")
    for var in post.data_vars:
        vname = str(var)
        if "calendar" in vname.lower() and "sigma" in vname.lower():
            arr = post[var].values.flatten()
            calendar_sigma = float(np.median(arr))
            break
    if not np.isfinite(calendar_sigma):
        for var in post.data_vars:
            vname = str(var)
            if "1|calendar" in vname or "calendar_sigma" in vname.lower():
                arr = post[var].values.flatten()
                calendar_sigma = float(np.median(arr))
                break

    # Spline coefficient SD — bs(dev_idx, df=4) terms.
    # Bambi names these like "bs(dev_idx, df = 4)[...]" with multiple levels.
    spline_coef_sd = float("nan")
    for var in post.data_vars:
        vname = str(var)
        if ("dev" in vname.lower() or "bs(" in vname.lower()) and "sigma" not in vname.lower() and "origin" not in vname.lower() and "calendar" not in vname.lower():
            arr = post[var].values
            if arr.ndim == 3 and arr.shape[2] > 1:
                level_means = arr.reshape(-1, arr.shape[2]).mean(axis=0)
                spline_coef_sd = float(np.std(level_means))
                break
            elif arr.ndim == 2 and arr.shape[1] > 1:
                level_means = arr.mean(axis=0)
                spline_coef_sd = float(np.std(level_means))
                break

    return {
        "status": "ok" if max_rhat < 1.1 else "not_converged",
        "max_rhat": max_rhat,
        "intercept_mean": intercept_mean,
        "intercept_sd": intercept_sd,
        "intercept_p10": intercept_p10,
        "intercept_p90": intercept_p90,
        "alpha_mean": alpha_mean,
        "alpha_sd": alpha_sd,
        "alpha_p10": alpha_p10,
        "alpha_p90": alpha_p90,
        "origin_sigma_median": origin_sigma,
        "calendar_sigma_median": calendar_sigma,
        "spline_coef_sd": spline_coef_sd,
    }


def _result_path() -> Path:
    return cache_path("m5cal_posteriors.parquet")


def _existing_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    df = pd.read_parquet(path)
    return set(zip(df["line"], df["snl_id"]))


def _append_row(row: dict, path: Path) -> None:
    new = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_parquet(path)
        out = pd.concat([existing, new], ignore_index=True)
    else:
        out = new
    out.to_parquet(path, index=False)


def main() -> int:
    print("Loading triangle JSON…", flush=True)
    full = load_full_triangle()

    out_path = _result_path()
    done = _existing_keys(out_path)
    print(f"  resuming from {len(done)} cached fits", flush=True)

    rows_total = len(done)
    for line in LINES:
        snl_ids = select_sample(full, line)
        sub_full = full[full["line_of_business"] == line]
        for snl_id in snl_ids:
            key = (line, snl_id)
            if key in done:
                continue
            seed = _seed_for(line, snl_id)
            print(f"  {line:5s} {snl_id:14s} seed={seed}", flush=True)
            t0 = time.time()
            try:
                res = _fit_and_extract(
                    sub_full[sub_full["snl_id"] == snl_id], seed
                )
                elapsed = time.time() - t0
                print(
                    f"    → max_rhat={res['max_rhat']:.3f}  "
                    f"intercept_mean={res['intercept_mean']:.3f}  "
                    f"origin_sigma={res['origin_sigma_median']:.3f}  "
                    f"calendar_sigma={res['calendar_sigma_median']:.3f}  "
                    f"({elapsed:.0f}s)",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                elapsed = time.time() - t0
                res = {
                    "status": f"error: {type(e).__name__}: {e}",
                    "max_rhat": float("nan"),
                    "intercept_mean": float("nan"),
                    "intercept_sd": float("nan"),
                    "intercept_p10": float("nan"),
                    "intercept_p90": float("nan"),
                    "alpha_mean": float("nan"),
                    "alpha_sd": float("nan"),
                    "alpha_p10": float("nan"),
                    "alpha_p90": float("nan"),
                    "origin_sigma_median": float("nan"),
                    "calendar_sigma_median": float("nan"),
                    "spline_coef_sd": float("nan"),
                }
                print(f"    → FAILED: {e} ({elapsed:.0f}s)", flush=True)
            _append_row(
                {"line": line, "snl_id": snl_id, **res},
                out_path,
            )
            done.add(key)
            rows_total += 1

    print(f"\nDone. {rows_total} fits cached to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
