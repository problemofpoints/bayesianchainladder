"""11_glm_priors_t.py — Mine MT2 posteriors for t-family GLM prior elicitation.

Fits incremental ~ 1 + C(origin) + bs(dev_idx, df=4) with family='t',
link='identity', response_per_exposure=True (loss-ratio scale) and
exposure='net_earned_premium', on the same 24 sampled triangles per line
used in 02_glm_model_comparison.py and 08_glm_priors.py.

Per fit, extracts:
  - Intercept posterior summary (mean, sd, p10, p90)  — on loss-ratio scale
  - Sigma posterior summary (Student-t scale parameter)
  - Nu posterior summary (Student-t degrees of freedom)
  - Spread of origin effect levels (origin_effect_sd_p50)
  - Spread of dev spline effect values (dev_effect_sd_p50)

Output: cache/m1_t_posteriors.parquet
Run: uv run python references/prior-elicitation-2026/11_glm_priors_t.py
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

FORMULA = "incremental ~ 1 + C(origin) + bs(dev_idx, df=4)"
SPEC = "MT2"


def _seed_for(line: str, snl_id: str) -> int:
    """Same seed logic as 02_glm_model_comparison for reproducibility."""
    h = hashlib.md5(f"{line}|{snl_id}|{SPEC}".encode()).hexdigest()
    return int(h, 16) % (2**31)


def _fit_and_extract(triangle, seed: int) -> dict:
    """Fit MT2 (t, identity, loss-ratio) and extract posterior summaries."""
    paid_tri = triangle["paid_loss"]
    prem_tri = triangle["net_earned_premium"]

    model = BayesianChainLadderGLM(
        formula=FORMULA,
        family="t",
        link="identity",
        exposure="net_earned_premium",
        response_per_exposure=True,
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

    # Intercept (on loss-ratio scale for t + identity + response_per_exposure).
    intercept_arr = _flat("Intercept")
    if intercept_arr is None:
        intercept_mean = intercept_sd = intercept_p10 = intercept_p90 = float("nan")
    else:
        intercept_mean = float(np.mean(intercept_arr))
        intercept_sd = float(np.std(intercept_arr))
        intercept_p10 = float(np.percentile(intercept_arr, 10))
        intercept_p90 = float(np.percentile(intercept_arr, 90))

    # Student-t sigma (scale parameter).
    sigma_arr = _flat("incremental_sigma")
    if sigma_arr is None:
        # Bambi may name it just "sigma"
        sigma_arr = _flat("sigma")
    if sigma_arr is None:
        sigma_mean = sigma_sd = sigma_p10 = sigma_p90 = float("nan")
    else:
        sigma_mean = float(np.mean(sigma_arr))
        sigma_sd = float(np.std(sigma_arr))
        sigma_p10 = float(np.percentile(sigma_arr, 10))
        sigma_p90 = float(np.percentile(sigma_arr, 90))

    # Student-t nu (degrees of freedom).
    nu_arr = _flat("incremental_nu")
    if nu_arr is None:
        nu_arr = _flat("nu")
    if nu_arr is None:
        nu_mean = nu_sd = nu_p10 = nu_p90 = float("nan")
    else:
        nu_mean = float(np.mean(nu_arr))
        nu_sd = float(np.std(nu_arr))
        nu_p10 = float(np.percentile(nu_arr, 10))
        nu_p90 = float(np.percentile(nu_arr, 90))

    # Origin effect spread.
    origin_effect_sd = float("nan")
    for var in post.data_vars:
        vname = str(var)
        if "origin" in vname.lower() and "sigma" not in vname.lower():
            arr = post[var].values
            if arr.ndim == 3 and arr.shape[2] > 1:
                level_means = arr.reshape(-1, arr.shape[2]).mean(axis=0)
                origin_effect_sd = float(np.std(level_means))
                break
            elif arr.ndim == 2 and arr.shape[1] > 1:
                level_means = arr.mean(axis=0)
                origin_effect_sd = float(np.std(level_means))
                break

    # Dev spline effect spread: look for bs(dev_idx) coefficients.
    dev_effect_sd = float("nan")
    for var in post.data_vars:
        vname = str(var)
        if ("dev" in vname.lower() or "bs(" in vname.lower()) and "sigma" not in vname.lower() and "origin" not in vname.lower():
            arr = post[var].values
            if arr.ndim == 3 and arr.shape[2] > 1:
                level_means = arr.reshape(-1, arr.shape[2]).mean(axis=0)
                dev_effect_sd = float(np.std(level_means))
                break
            elif arr.ndim == 2 and arr.shape[1] > 1:
                level_means = arr.mean(axis=0)
                dev_effect_sd = float(np.std(level_means))
                break

    return {
        "status": "ok" if max_rhat < 1.1 else "not_converged",
        "max_rhat": max_rhat,
        "intercept_mean": intercept_mean,
        "intercept_sd": intercept_sd,
        "intercept_p10": intercept_p10,
        "intercept_p90": intercept_p90,
        "sigma_mean": sigma_mean,
        "sigma_sd": sigma_sd,
        "sigma_p10": sigma_p10,
        "sigma_p90": sigma_p90,
        "nu_mean": nu_mean,
        "nu_sd": nu_sd,
        "nu_p10": nu_p10,
        "nu_p90": nu_p90,
        "origin_effect_sd_p50": origin_effect_sd,
        "dev_effect_sd_p50": dev_effect_sd,
    }


def _result_path() -> Path:
    return cache_path("m1_t_posteriors.parquet")


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
                    f"intercept_mean={res['intercept_mean']:.4f}  "
                    f"sigma_mean={res['sigma_mean']:.4f}  "
                    f"nu_mean={res['nu_mean']:.2f}  ({elapsed:.0f}s)",
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
                    "sigma_mean": float("nan"),
                    "sigma_sd": float("nan"),
                    "sigma_p10": float("nan"),
                    "sigma_p90": float("nan"),
                    "nu_mean": float("nan"),
                    "nu_sd": float("nan"),
                    "nu_p10": float("nan"),
                    "nu_p90": float("nan"),
                    "origin_effect_sd_p50": float("nan"),
                    "dev_effect_sd_p50": float("nan"),
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
