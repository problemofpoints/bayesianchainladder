"""02_worker.py — Process a chunk of Meyers records, running all 7 methods.

Usage: uv run python 02_worker.py --chunk-id N --n-chunks M [--out PATH]

Each chunk processes records[chunk_id::n_chunks] (stride partition for balance),
writes to cache/backtest_chunk_{chunk_id}.parquet, and supports resume.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import reservetestr as rt

from _common import MEYERS_TO_PRIOR_LINE
from methods import (
    testr_correlated_bootstrap_odp,
    testr_bayesian_csr,
    testr_glm_m2,
    testr_glm_m5_cal,
    testr_glm_mt5_cal,
)

METHODS = [
    ("mack",            rt.testr_mack_chainladder, {}),
    ("bootstrap_odp",   rt.testr_bootstrap_odp,    {"n_sims": 1000, "hat_adj": True, "random_state": 22}),
    ("corr_boot_odp",   testr_correlated_bootstrap_odp, {"n_sims": 1000, "random_state": 22}),
    ("bayesian_csr",    testr_bayesian_csr,        {}),
    ("glm_m2",          testr_glm_m2,              {}),
    ("glm_m5_cal",      testr_glm_m5_cal,          {}),
    ("glm_mt5_cal",     testr_glm_mt5_cal,         {}),
]


def _existing_keys(out_path: Path) -> set[tuple[str, int, str]]:
    if not out_path.exists():
        return set()
    df = pd.read_parquet(out_path)
    return set(zip(df["line"], df["group_id"], df["method"]))


def _append_row(out_path: Path, row: dict):
    new = pd.DataFrame([row])
    if out_path.exists():
        out = pd.concat([pd.read_parquet(out_path), new], ignore_index=True)
    else:
        out = new
    out.to_parquet(out_path, index=False)


def main(chunk_id: int, n_chunks: int, out_path: Path) -> int:
    print(f"[chunk {chunk_id}/{n_chunks}] starting", flush=True)
    records = rt.build_triangle_records()
    # Stride-partition for load balancing — alternating lines per worker.
    my_records = records[chunk_id::n_chunks]
    print(f"[chunk {chunk_id}] {len(my_records)} records assigned", flush=True)
    done = _existing_keys(out_path)
    print(f"[chunk {chunk_id}] resuming from {len(done)} cached fits", flush=True)

    for r in my_records:
        for method_label, fn, kwargs in METHODS:
            key = (r.line, r.group_id, method_label)
            if key in done:
                continue
            kw = dict(kwargs)
            if method_label not in ("mack", "bootstrap_odp"):
                kw["line"] = r.line
                kw["group_id"] = r.group_id
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = fn(r.train_triangles, r.test_triangles,
                             loss_type="paid",
                             actual_ultimates=r.actual_ultimates, **kw)
                if res is None:
                    res = {"status": "none_returned"}
                # If wrapper didn't set status, default to "ok".
                if "status" not in res:
                    res["status"] = "ok"
            except Exception as e:
                res = {
                    "actual_ultimate": float("nan"),
                    "actual_unpaid": float("nan"),
                    "mean_ultimate_est": float("nan"),
                    "mean_unpaid_est": float("nan"),
                    "stddev_est": float("nan"),
                    "cv_unpaid_est": float("nan"),
                    "implied_pctl": float("nan"),
                    "status": f"error:{type(e).__name__}:{str(e)[:100]}",
                }
            print(f"[chunk {chunk_id}] {r.line:8s} {r.group_id:>5d} {method_label:15s} status={res['status'][:30]}", flush=True)
            _append_row(out_path, {
                "line": r.line, "group_id": r.group_id, "company": r.company,
                "method": method_label, **res,
            })
            done.add(key)
    print(f"[chunk {chunk_id}] done; {len(done)} fits total in cache", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--n-chunks", type=int, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or (Path(__file__).parent / "cache" / f"backtest_chunk_{args.chunk_id}.parquet")
    raise SystemExit(main(args.chunk_id, args.n_chunks, out))
