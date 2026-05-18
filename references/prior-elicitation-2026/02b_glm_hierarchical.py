"""02b_glm_hierarchical.py — Hierarchical GLM (M4) per line via Bambi (1 | snl_id).

For each line, builds a long-format dataframe stacking the 24 sampled triangles
(observed cells only), fits one Bambi model with formula:

    incremental ~ 1 + C(origin) + C(dev) + (1 | snl_id) + offset(logoffset)

with gamma family and explicit log link (Bambi's default gamma link is inverse;
using log gives the log-linear chain-ladder parameterisation). Computes WAIC/LOO.
Also extracts the posterior summary for the company-level random-intercept SD so
we can inform M4's hyper-prior.

Output: cache/glm_hierarchical_fits.parquet  (one row per line)
Run:    uv run python references/prior-elicitation-2026/02b_glm_hierarchical.py
"""
from __future__ import annotations

import hashlib
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

from _common import (
    LINES,
    cache_path,
    load_full_triangle,
    select_sample,
)


def _seed_for(line: str) -> int:
    """Stable per-line seed via md5 (PYTHONHASHSEED-safe)."""
    h = hashlib.md5(f"M4|{line}".encode()).hexdigest()
    return int(h, 16) % (2**31)


def _build_long_df(full_tri, line: str, snl_ids: list[str]) -> pd.DataFrame:
    """Stack observed (origin_idx, dev_idx) cells for `snl_ids` into a long-format frame."""
    rows = []
    sub_full = full_tri[full_tri["line_of_business"] == line]
    for snl_id in snl_ids:
        sub = sub_full[sub_full["snl_id"] == snl_id]
        paid = sub["paid_loss"].values[0, 0]  # cumulative
        prem = sub["net_earned_premium"].values[0, 0]
        n_origin, n_dev = paid.shape
        # Convert cumulative to incremental.
        inc = np.full_like(paid, np.nan, dtype=float)
        inc[:, 0] = paid[:, 0]
        inc[:, 1:] = paid[:, 1:] - paid[:, :-1]
        for i in range(n_origin):
            origin_year = 2015 + i
            for j in range(n_dev):
                v = inc[i, j]
                if np.isnan(v) or v <= 0:
                    continue
                exp = prem[i, 0] if not np.isnan(prem[i, 0]) else np.nan
                if np.isnan(exp) or exp <= 0:
                    continue
                rows.append(
                    {
                        "snl_id": snl_id,
                        "origin": origin_year,
                        "dev": (j + 1) * 12,
                        "incremental": float(v),
                        "net_earned_premium": float(exp),
                    }
                )
    df = pd.DataFrame(rows)
    df["origin"] = pd.Categorical(df["origin"])
    df["dev"] = pd.Categorical(df["dev"])
    df["snl_id"] = pd.Categorical(df["snl_id"])
    return df


def _fit_hierarchical(df: pd.DataFrame, seed: int) -> dict:
    """Fit a Bambi gamma GLM with (1 | snl_id) random intercept and return WAIC/LOO.

    Follows the same offset convention as the package's `build_bambi_model`:
    pre-compute a `logoffset` column and append `offset(logoffset)` to the
    formula (Bambi 0.13+ does not accept an `offset=` kwarg).
    """
    df = df.copy()
    df["logoffset"] = np.log(df["net_earned_premium"].astype(float).values)
    formula = (
        "incremental ~ 1 + C(origin) + C(dev) + (1 | snl_id) + offset(logoffset)"
    )
    # Explicitly request log link — Bambi's default gamma link is inverse, which
    # produces a different (and less standard for chain-ladder) model geometry.
    from bayesianchainladder.models import _get_family
    gamma_log = _get_family("gamma", "log")
    model = bmb.Model(formula=formula, data=df, family=gamma_log)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idata = model.fit(
            draws=1000,
            tune=1000,
            chains=2,
            target_accept=0.95,
            random_seed=seed,
            idata_kwargs={"log_likelihood": True},
        )
    waic = az.waic(idata)
    loo = az.loo(idata)
    summ = az.summary(idata, kind="diagnostics")
    max_rhat = float(summ["r_hat"].max()) if "r_hat" in summ.columns else float("nan")
    # Posterior summary for the company-level random effect SD.
    # Bambi names it "1|snl_id_sigma" or similar — find any var that contains
    # both "snl_id" and "sigma".
    sigma_var = None
    for v in idata.posterior.data_vars:
        if "snl_id" in str(v) and "sigma" in str(v):
            sigma_var = v
            break
    if sigma_var is not None:
        s_arr = idata.posterior[sigma_var].values.flatten()
        sigma_mean = float(np.mean(s_arr))
        sigma_p90 = float(np.percentile(s_arr, 90))
    else:
        sigma_mean = float("nan")
        sigma_p90 = float("nan")
    return {
        "waic": float(waic.elpd_waic),
        "p_waic": float(waic.p_waic),
        "loo": float(loo.elpd_loo),
        "p_loo": float(loo.p_loo),
        "max_rhat": max_rhat,
        "company_sigma_mean": sigma_mean,
        "company_sigma_p90": sigma_p90,
        "n_obs": int(len(df)),
        "n_companies": int(df["snl_id"].nunique()),
    }


def main() -> int:
    print("Loading triangle JSON…", flush=True)
    full = load_full_triangle()
    out_path = cache_path("glm_hierarchical_fits.parquet")
    done_lines: set[str] = set()
    if out_path.exists():
        done_lines = set(pd.read_parquet(out_path)["line"].unique())
        print(f"  resuming; {len(done_lines)} lines already cached", flush=True)

    rows: list[dict] = []
    if out_path.exists():
        rows = pd.read_parquet(out_path).to_dict("records")

    for line in LINES:
        if line in done_lines:
            continue
        snl_ids = select_sample(full, line)
        df = _build_long_df(full, line, snl_ids)
        print(
            f"  {line}: n_obs={len(df)} n_companies={df['snl_id'].nunique()}",
            flush=True,
        )
        seed = _seed_for(line)
        try:
            res = _fit_hierarchical(df, seed)
            status = "ok"
        except Exception as e:  # noqa: BLE001
            res = {
                "waic": float("nan"),
                "p_waic": float("nan"),
                "loo": float("nan"),
                "p_loo": float("nan"),
                "max_rhat": float("nan"),
                "company_sigma_mean": float("nan"),
                "company_sigma_p90": float("nan"),
                "n_obs": int(len(df)),
                "n_companies": int(df["snl_id"].nunique()),
            }
            status = f"error: {type(e).__name__}: {e}"
        rows.append({"line": line, "spec": "M4_hierarchical", "status": status, **res})
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        print(f"    {line} done; status={status}", flush=True)

    print(f"Wrote {out_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
