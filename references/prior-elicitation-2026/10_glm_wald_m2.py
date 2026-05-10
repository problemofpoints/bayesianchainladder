"""10_glm_wald_m2.py — Refit M2 (`incremental ~ 1 + C(origin) + bs(dev_idx, df=4)`)
under Wald (inverse-Gaussian) family with log link, for direct comparison vs gamma+log.

Wald is sometimes preferred for heavy-tailed loss data: variance scales as μ³ rather
than μ² for gamma, giving thicker right tails.

Output: cache/glm_wald_m2_fits.parquet — one row per (line, snl_id).

Run: uv run python references/prior-elicitation-2026/10_glm_wald_m2.py
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

from bayesianchainladder import BayesianChainLadderGLM, compute_loo, compute_waic
from _common import LINES, cache_path, load_full_triangle, select_sample


FORMULA = "incremental ~ 1 + C(origin) + bs(dev_idx, df=4)"


def _seed_for(line: str, snl_id: str) -> int:
    h = hashlib.md5(f"WALD_M2|{line}|{snl_id}".encode()).hexdigest()
    return int(h, 16) % (2**31)


def _fit_one(triangle, seed: int) -> dict:
    paid_tri = triangle["paid_loss"]
    prem_tri = triangle["net_earned_premium"]

    model = BayesianChainLadderGLM(
        formula=FORMULA,
        family="wald",
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
    waic = compute_waic(model.idata)
    loo = compute_loo(model.idata)
    summ = az.summary(model.idata, kind="diagnostics")
    max_rhat = float(summ["r_hat"].max()) if "r_hat" in summ.columns else float("nan")
    return {
        "waic": float(waic.elpd_waic),
        "p_waic": float(waic.p_waic),
        "loo": float(loo.elpd_loo),
        "p_loo": float(loo.p_loo),
        "max_rhat": max_rhat,
    }


def _result_path() -> Path:
    return cache_path("glm_wald_m2_fits.parquet")


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
        out = pd.concat([pd.read_parquet(p), new], ignore_index=True)
    else:
        out = new
    out.to_parquet(p, index=False)


def main() -> int:
    print("Loading triangle JSON…", flush=True)
    full = load_full_triangle()
    done = _existing_keys()
    print(f"  resuming with {len(done)} cached fits", flush=True)
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
                res = {k: float("nan") for k in ("waic", "p_waic", "loo", "p_loo", "max_rhat")}
                status = f"error: {type(e).__name__}: {e}"
            _append_row({"line": line, "snl_id": snl_id, "status": status, **res})
            done.add((line, snl_id))
    print(f"Done. {len(done)} fits cached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
