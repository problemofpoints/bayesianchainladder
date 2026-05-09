"""04_rho_estimation.py — Empirical rho per line for CorrelatedBootstrapODPSample.

Pools standardised Pearson residuals across all eligible companies in a line
into a single (n_companies, n_origin, n_dev) panel, then aggregates
calendar-diagonal correlations via Fisher z-mean. Bootstrap CI on rho is a
company-level resample.

Output: cache/rho_by_line.parquet
Run:    uv run python references/prior-elicitation-2026/04_rho_estimation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from _common import (
    LINES,
    cache_path,
    iter_eligible_triangles,
    load_full_triangle,
    pearson_residuals,
    rho_from_residual_panel,
)


def _build_panel(full_tri, line: str) -> tuple[np.ndarray, list[str]]:
    """Stack residuals for `line` into a (n_companies, n_origin, n_dev) panel."""
    panels = []
    snl_ids = []
    for snl_id, sub_tri in iter_eligible_triangles(full_tri, line):
        res_df = pearson_residuals(sub_tri)
        if len(res_df) < 30:
            continue
        # 10x10 grid of NaN, fill with residuals at (origin_idx, dev_idx).
        grid = np.full((10, 10), np.nan)
        for _, r in res_df.iterrows():
            grid[int(r.origin_idx), int(r.dev_idx)] = r.residual
        panels.append(grid)
        snl_ids.append(snl_id)
    if not panels:
        return np.empty((0, 10, 10)), []
    return np.stack(panels, axis=0), snl_ids


def _bootstrap_ci(panel: np.ndarray, n_boot: int = 1000, seed: int = 20260508):
    """Resample companies (axis 0) with replacement; recompute rho each iteration."""
    rng = np.random.default_rng(seed)
    n = panel.shape[0]
    rhos = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = panel[idx]
        rhos[b] = rho_from_residual_panel(sample)["rho"]
    rhos = rhos[np.isfinite(rhos)]
    return {
        "rho_median": float(np.median(rhos)),
        "rho_p10": float(np.percentile(rhos, 10)),
        "rho_p90": float(np.percentile(rhos, 90)),
    }


def main() -> int:
    print("Loading triangle JSON…", flush=True)
    full = load_full_triangle()

    rows = []
    for line in LINES:
        print(f"  • {line}…", flush=True)
        panel, snl_ids = _build_panel(full, line)
        if panel.shape[0] == 0:
            print(f"    no eligible triangles for {line}; skipping")
            continue
        out = rho_from_residual_panel(panel)
        ci = _bootstrap_ci(panel)
        rows.append(
            {
                "line": line,
                "n_companies": int(panel.shape[0]),
                "rho_point": out["rho"],
                "rho_median": ci["rho_median"],
                "rho_p10": ci["rho_p10"],
                "rho_p90": ci["rho_p90"],
                "r1": out["r_by_d"].get(1, float("nan")),
                "r2": out["r_by_d"].get(2, float("nan")),
                "r3": out["r_by_d"].get(3, float("nan")),
                "n_pairs_d0": out["n_pairs_by_d"].get(0, 0),
            }
        )
        print(
            f"    rho={out['rho']:.3f}  CI=({ci['rho_p10']:.3f}, {ci['rho_p90']:.3f})  "
            f"n_companies={panel.shape[0]}  n_pairs_d0={out['n_pairs_by_d'].get(0, 0)}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    out_path = cache_path("rho_by_line.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
