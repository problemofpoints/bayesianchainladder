"""03_csr_priors.py — Fit BayesianCSR on the 24-triangle sample per line and
extract posteriors for `logelr`, `r_alpha`, `r_beta`, `gamma`, `a_ig`, `sig`.

Cache: cache/csr_fits.parquet — one row per (line, snl_id).

Run: uv run python references/prior-elicitation-2026/03_csr_priors.py
"""
from __future__ import annotations

import hashlib
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import arviz as az
import numpy as np
import pandas as pd

from bayesianchainladder import BayesianCSR
from _common import (
    LINES,
    cache_path,
    load_full_triangle,
    select_sample,
)


def _seed_for(line: str, snl_id: str) -> int:
    """Stable per-(line, snl_id) seed via md5 (PYTHONHASHSEED-safe)."""
    h = hashlib.md5(f"CSR|{line}|{snl_id}".encode()).hexdigest()
    return int(h, 16) % (2**31)


def _summarise(idata, var_name: str) -> dict[str, float]:
    """Return mean / sd / p10 / p90 for a posterior variable; NaN if missing."""
    if var_name not in idata.posterior.data_vars:
        return {
            f"{var_name}_mean": float("nan"),
            f"{var_name}_sd": float("nan"),
            f"{var_name}_p10": float("nan"),
            f"{var_name}_p90": float("nan"),
        }
    arr = idata.posterior[var_name].values.flatten()
    return {
        f"{var_name}_mean": float(np.mean(arr)),
        f"{var_name}_sd": float(np.std(arr, ddof=1)),
        f"{var_name}_p10": float(np.percentile(arr, 10)),
        f"{var_name}_p90": float(np.percentile(arr, 90)),
    }


def _fit_one(triangle, seed: int) -> dict:
    model = BayesianCSR(
        draws=1000,
        tune=500,
        chains=2,
        target_accept=0.9,
        random_seed=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prem_tri = triangle["net_earned_premium"]
        paid_tri = triangle["paid_loss"]
        model.fit(paid_tri, premium_triangle=prem_tri)
    idata = model.idata
    summ = az.summary(idata, kind="diagnostics")
    max_rhat = float(summ["r_hat"].max()) if "r_hat" in summ.columns else float("nan")
    out = {"max_rhat": max_rhat}
    # Verified posterior variable names (from idata.posterior.data_vars):
    #   logelr, gamma, a_ig, r_alpha, r_beta, sig, alpha, beta, speedup, mu
    # Note: alpha_sigma and beta_sigma are fixed prior hyperparameters (not
    # sampled), so they do not appear in the posterior. r_alpha / r_beta are
    # the non-centred raw draws for origin/dev effects; sig is observation
    # noise. If a future refactor renames them, _summarise records NaN and
    # the synthesis step can still aggregate available names.
    for var in ("logelr", "r_alpha", "r_beta", "gamma", "a_ig", "sig"):
        out.update(_summarise(idata, var))
    return out


def _result_path() -> Path:
    return cache_path("csr_fits.parquet")


def _existing_keys() -> set[tuple[str, str]]:
    p = _result_path()
    if not p.exists():
        return set()
    df = pd.read_parquet(p)
    return set(zip(df["line"], df["snl_id"]))


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
    print(f"  resuming with {len(done)} rows cached", flush=True)

    for line in LINES:
        snl_ids = select_sample(full, line)
        sub_full = full[full["line_of_business"] == line]
        for snl_id in snl_ids:
            if (line, snl_id) in done:
                continue
            sub_tri = sub_full[sub_full["snl_id"] == snl_id]
            seed = _seed_for(line, snl_id)
            print(f"  {line:5s} {snl_id:14s} seed={seed}", flush=True)
            try:
                res = _fit_one(sub_tri, seed)
                status = "ok"
            except Exception as e:  # noqa: BLE001
                res = {"max_rhat": float("nan")}
                status = f"error: {type(e).__name__}: {e}"
            _append_row(
                {"line": line, "snl_id": snl_id, "status": status, **res}
            )
            done.add((line, snl_id))

    print(f"Done. {len(done)} fits cached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
