"""06_diagnose_m2.py — Diagnose why M2 (dev spline) failed and test alternatives.

Hypothesis: the original M2 used bs(dev, df=4) where dev is in months (12, 24, ...,
120). Large numeric values in B-spline bases make the posterior geometry extreme.

Fix: use dev_idx (1-based ordinal index, auto-materialized by add_categorical_columns
when _idx suffix is detected in the formula).

Three alternative M2 specs tested on the first 3 PPAL triangles from
glm_per_triangle_fits.parquet:
  M2_devidx_bs4:   incremental ~ 1 + C(origin) + bs(dev_idx, df=4)
  M2_devidx_bs3:   incremental ~ 1 + C(origin) + bs(dev_idx, df=3)
  M2_devidx_poly3: incremental ~ 1 + C(origin) + I(dev_idx) + I(dev_idx**2) + I(dev_idx**3)

Output: cache/m2_diagnostic.parquet
Run: uv run python references/prior-elicitation-2026/06_diagnose_m2.py
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
    cache_path,
    load_full_triangle,
    select_sample,
)

# The 3 PPAL snl_ids to test (first 3 from the existing M1 fits, in sorted order).
TARGET_LINE = "PPAL"
N_TRIANGLES = 3

SPECS: dict[str, str] = {
    "M2_devidx_bs4": "incremental ~ 1 + C(origin) + bs(dev_idx, df=4)",
    "M2_devidx_bs3": "incremental ~ 1 + C(origin) + bs(dev_idx, df=3)",
    "M2_devidx_poly3": (
        "incremental ~ 1 + C(origin) + I(dev_idx) + I(dev_idx**2) + I(dev_idx**3)"
    ),
}


def _seed_for(line: str, snl_id: str, spec: str) -> int:
    h = hashlib.md5(f"{line}|{snl_id}|{spec}".encode()).hexdigest()
    return int(h, 16) % (2**31)


def _fit_one(triangle, formula: str, seed: int) -> dict:
    """Fit BayesianChainLadderGLM and return diagnostics."""
    paid_tri = triangle["paid_loss"]
    prem_tri = triangle["net_earned_premium"]

    model = BayesianChainLadderGLM(
        formula=formula,
        family="gamma",
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
    summ = az.summary(idata, kind="diagnostics")
    max_rhat = float(summ["r_hat"].max()) if "r_hat" in summ.columns else float("nan")

    # Count divergences if available.
    n_divergences = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        n_divergences = int(idata.sample_stats["diverging"].values.sum())

    return {
        "max_rhat": max_rhat,
        "n_divergences": n_divergences,
        "status": "ok" if max_rhat < 1.1 else "not_converged",
    }


def _result_path() -> Path:
    return cache_path("m2_diagnostic.parquet")


def _existing_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    df = pd.read_parquet(path)
    return set(zip(df["line"], df["snl_id"], df["spec"]))


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

    # Determine the 3 PPAL snl_ids.
    existing_fits = cache_path("glm_per_triangle_fits.parquet")
    if existing_fits.exists():
        all_fits = pd.read_parquet(existing_fits)
        ppal_m1 = all_fits[
            (all_fits["line"] == TARGET_LINE) & (all_fits["spec"] == "M1_cat")
        ].sort_values("snl_id")
        snl_ids = ppal_m1["snl_id"].head(N_TRIANGLES).tolist()
        print(f"Using first {N_TRIANGLES} PPAL snl_ids from M1 cache: {snl_ids}", flush=True)
    else:
        # Fallback: select fresh sample and take first N_TRIANGLES.
        all_ids = select_sample(full, TARGET_LINE)
        snl_ids = all_ids[:N_TRIANGLES]
        print(f"No M1 cache; using first {N_TRIANGLES} from fresh sample: {snl_ids}", flush=True)

    out_path = _result_path()
    done = _existing_keys(out_path)
    print(f"  resuming from {len(done)} cached fits", flush=True)

    sub_full = full[full["line_of_business"] == TARGET_LINE]

    rows_total = 0
    for snl_id in snl_ids:
        sub_tri = sub_full[sub_full["snl_id"] == snl_id]
        for spec_name, formula in SPECS.items():
            key = (TARGET_LINE, snl_id, spec_name)
            if key in done:
                rows_total += 1
                continue
            seed = _seed_for(TARGET_LINE, snl_id, spec_name)
            print(
                f"  {TARGET_LINE} {snl_id} {spec_name}  seed={seed}",
                flush=True,
            )
            t0 = time.time()
            try:
                res = _fit_one(sub_tri, formula, seed)
                elapsed = time.time() - t0
                print(
                    f"    → max_rhat={res['max_rhat']:.3f}  "
                    f"divs={res['n_divergences']}  "
                    f"status={res['status']}  ({elapsed:.0f}s)",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                elapsed = time.time() - t0
                res = {
                    "max_rhat": float("nan"),
                    "n_divergences": -1,
                    "status": f"error: {type(e).__name__}: {e}",
                }
                print(f"    → FAILED: {e} ({elapsed:.0f}s)", flush=True)
            _append_row(
                {
                    "line": TARGET_LINE,
                    "snl_id": snl_id,
                    "spec": spec_name,
                    "formula": formula,
                    **res,
                },
                out_path,
            )
            done.add(key)
            rows_total += 1

    # Print summary table.
    print("\n=== M2 Diagnostic Summary ===\n")
    df = pd.read_parquet(out_path)
    pivot = df.pivot_table(
        index="spec",
        columns="snl_id",
        values=["max_rhat", "n_divergences", "status"],
        aggfunc="first",
    )
    print(df[["snl_id", "spec", "max_rhat", "n_divergences", "status"]].to_string(index=False))
    print()
    # Summary: per spec, any ok?
    summary = (
        df.groupby("spec")
        .agg(
            max_rhat_max=("max_rhat", "max"),
            max_rhat_mean=("max_rhat", "mean"),
            total_divs=("n_divergences", "sum"),
            n_ok=("status", lambda x: (x == "ok").sum()),
            n_total=("status", "count"),
        )
        .reset_index()
    )
    print("Per-spec summary:")
    print(summary.to_string(index=False))
    print(f"\nWrote {out_path} ({rows_total} fits total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
