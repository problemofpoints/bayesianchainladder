"""07_diagnose_m4.py — Reparameterization experiments for M4 (hierarchical GLM).

Tries 4 variants of the line-level hierarchical GLM on PPAL (all 24 companies).
Reuses _build_long_df from 02b_glm_hierarchical.py.

Variants:
  M4a: baseline (same as 02b but with longer tune=2000, target_accept=0.99)
  M4b: tighter SD prior on company random effect
  M4c: gaussian on log-incremental (lognormal reparameterization)
  M4d: deferred — non-centered reparam requires raw PyMC, not expressible in Bambi

Output: cache/m4_diagnostic.parquet
Run: uv run python references/prior-elicitation-2026/07_diagnose_m4.py
"""
from __future__ import annotations

import hashlib
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

from _common import (
    cache_path,
    load_full_triangle,
    select_sample,
)
from importlib import import_module

# Import _build_long_df from 02b_glm_hierarchical
_02b = import_module("02b_glm_hierarchical")
_build_long_df = _02b._build_long_df

TARGET_LINE = "PPAL"


def _seed_for(spec: str) -> int:
    h = hashlib.md5(f"M4diag|{TARGET_LINE}|{spec}".encode()).hexdigest()
    return int(h, 16) % (2**31)


def _fit_m4a(df: pd.DataFrame, seed: int) -> dict:
    """M4a: baseline gamma GLM, longer tune, higher target_accept."""
    df = df.copy()
    df["logoffset"] = np.log(df["net_earned_premium"].astype(float).values)
    formula = (
        "incremental ~ 1 + C(origin) + C(dev) + (1 | snl_id) + offset(logoffset)"
    )
    model = bmb.Model(formula=formula, data=df, family="gamma")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idata = model.fit(
            draws=1000,
            tune=2000,
            chains=2,
            target_accept=0.99,
            random_seed=seed,
            idata_kwargs={"log_likelihood": True},
        )
    return _extract_diagnostics(idata, df)


def _fit_m4b(df: pd.DataFrame, seed: int) -> dict:
    """M4b: tighter HalfNormal(0.5) prior on company random-effect SD."""
    df = df.copy()
    df["logoffset"] = np.log(df["net_earned_premium"].astype(float).values)
    formula = (
        "incremental ~ 1 + C(origin) + C(dev) + (1 | snl_id) + offset(logoffset)"
    )
    # Bambi names the SD of the random effect as "1|snl_id_sigma".
    priors = {"1|snl_id": bmb.Prior("HalfNormal", sigma=0.5)}
    model = bmb.Model(formula=formula, data=df, family="gamma", priors=priors)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idata = model.fit(
            draws=1000,
            tune=2000,
            chains=2,
            target_accept=0.99,
            random_seed=seed,
            idata_kwargs={"log_likelihood": True},
        )
    return _extract_diagnostics(idata, df)


def _fit_m4c(df: pd.DataFrame, seed: int) -> dict:
    """M4c: gaussian on log-incremental (lognormal reparameterization).

    log(incremental) ~ offset(log_premium) effectively removes the premium
    scale, giving a dimensionless loss-ratio in log space.
    The offset shifts the predictor: log_inc - log_prem = log(loss_ratio).
    """
    df = df.copy()
    df["log_incremental"] = np.log(df["incremental"].astype(float).values)
    # Use log-premium as offset (gaussian has identity link, so offset is
    # additive on the linear predictor: mu = X*beta + log_prem).
    df["log_prem"] = np.log(df["net_earned_premium"].astype(float).values)
    formula = (
        "log_incremental ~ 1 + C(origin) + C(dev) + (1 | snl_id) + offset(log_prem)"
    )
    model = bmb.Model(formula=formula, data=df, family="gaussian")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idata = model.fit(
            draws=1000,
            tune=2000,
            chains=2,
            target_accept=0.99,
            random_seed=seed,
            idata_kwargs={"log_likelihood": True},
        )
    return _extract_diagnostics(idata, df)


def _extract_diagnostics(idata, df: pd.DataFrame) -> dict:
    """Extract max_rhat, divergences, LOO, and n_obs from a fitted idata."""
    summ = az.summary(idata, kind="diagnostics")
    max_rhat = float(summ["r_hat"].max()) if "r_hat" in summ.columns else float("nan")

    n_divergences = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        n_divergences = int(idata.sample_stats["diverging"].values.sum())

    try:
        loo_result = az.loo(idata)
        loo_val = float(loo_result.elpd_loo)
        p_loo_val = float(loo_result.p_loo)
    except Exception:  # noqa: BLE001
        loo_val = float("nan")
        p_loo_val = float("nan")

    return {
        "max_rhat": max_rhat,
        "n_divergences": n_divergences,
        "loo": loo_val,
        "p_loo": p_loo_val,
        "n_obs": int(len(df)),
        "n_companies": int(df["snl_id"].nunique()),
        "status": "ok" if max_rhat < 1.1 else "not_converged",
    }


VARIANTS: dict[str, object] = {
    "M4a": _fit_m4a,
    "M4b_tighter_sd": _fit_m4b,
    "M4c_lognormal": _fit_m4c,
}

M4D_NOTE = (
    "M4d (non-centered reparam) would require raw PyMC to express the "
    "non-centered parameterization explicitly. Bambi's formula syntax does not "
    "support manual non-centering. Deferred."
)


def _result_path() -> Path:
    return cache_path("m4_diagnostic.parquet")


def _existing_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    df = pd.read_parquet(path)
    return set(zip(df["line"], df["spec"]))


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

    snl_ids = select_sample(full, TARGET_LINE)
    df_long = _build_long_df(full, TARGET_LINE, snl_ids)
    print(
        f"PPAL long df: n_obs={len(df_long)}, n_companies={df_long['snl_id'].nunique()}",
        flush=True,
    )

    out_path = _result_path()
    done = _existing_keys(out_path)
    print(f"  resuming from {len(done)} cached fits", flush=True)

    for spec_name, fit_fn in VARIANTS.items():
        key = (TARGET_LINE, spec_name)
        if key in done:
            print(f"  {spec_name}: already cached, skipping", flush=True)
            continue
        seed = _seed_for(spec_name)
        print(f"  {TARGET_LINE} {spec_name}  seed={seed}", flush=True)
        t0 = time.time()
        try:
            res = fit_fn(df_long, seed)
            elapsed = time.time() - t0
            print(
                f"    → max_rhat={res['max_rhat']:.3f}  "
                f"divs={res['n_divergences']}  "
                f"loo={res['loo']:.1f}  "
                f"status={res['status']}  ({elapsed:.0f}s)",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            elapsed = time.time() - t0
            res = {
                "max_rhat": float("nan"),
                "n_divergences": -1,
                "loo": float("nan"),
                "p_loo": float("nan"),
                "n_obs": int(len(df_long)),
                "n_companies": int(df_long["snl_id"].nunique()),
                "status": f"error: {type(e).__name__}: {e}",
            }
            print(f"    → FAILED: {e} ({elapsed:.0f}s)", flush=True)
        _append_row(
            {
                "line": TARGET_LINE,
                "spec": spec_name,
                **res,
            },
            out_path,
        )
        done.add(key)

    # Add M4d deferred row if not present.
    if (TARGET_LINE, "M4d_deferred") not in done:
        _append_row(
            {
                "line": TARGET_LINE,
                "spec": "M4d_deferred",
                "max_rhat": float("nan"),
                "n_divergences": -1,
                "loo": float("nan"),
                "p_loo": float("nan"),
                "n_obs": int(len(df_long)),
                "n_companies": int(df_long["snl_id"].nunique()),
                "status": M4D_NOTE,
            },
            out_path,
        )

    # Print summary table.
    print("\n=== M4 Diagnostic Summary ===\n")
    df = pd.read_parquet(out_path)
    print(df[["spec", "max_rhat", "n_divergences", "loo", "n_obs", "n_companies", "status"]].to_string(index=False))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
