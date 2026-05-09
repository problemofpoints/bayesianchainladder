"""01_descriptive.py — Empirical, no-MCMC summary statistics per line.

For each line, computes (and caches):
  - Age-to-age factor distribution by dev period
  - Loss-ratio (ultimate / premium) distribution by accident year
  - Pearson dispersion phi (one per triangle)
  - Variance-vs-mean log-log slope of incremental paid loss
  - Between-company variance of log loss-ratio (informs M4 hyper-prior)

Produces:
  - cache/descriptive_summary.parquet  (per-line rows; one row per line)
  - figures/01_descriptive_<line>.png (3-panel diagnostic figure)

Run: uv run python references/prior-elicitation-2026/01_descriptive.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _common import (
    FIGURES_DIR,
    LINES,
    cache_path,
    iter_eligible_triangles,
    load_full_triangle,
    pearson_residuals,
)


def _atau_factors(paid: np.ndarray) -> dict[int, list[float]]:
    """Per-origin age-to-age factors for a single 10x10 paid triangle."""
    out: dict[int, list[float]] = {}
    n_origin, n_dev = paid.shape
    for j in range(n_dev - 1):
        for i in range(n_origin):
            a, b = paid[i, j], paid[i, j + 1]
            if np.isnan(a) or np.isnan(b) or a <= 0:
                continue
            out.setdefault(j + 1, []).append(float(b / a))
    return out


def _ultimate_to_premium(tri) -> dict[int, float]:
    """Loss-ratio per origin: booked_ultimate / net_earned_premium."""
    if "booked_ultimate_loss" not in list(tri.vdims):
        return {}
    ult = tri["booked_ultimate_loss"].latest_diagonal.values.flatten()
    prem = tri["net_earned_premium"].latest_diagonal.values.flatten()
    out = {}
    for k, (u, p) in enumerate(zip(ult, prem)):
        if np.isnan(u) or np.isnan(p) or p <= 0:
            continue
        out[k] = float(u / p)
    return out


def _phi_for_triangle(tri) -> float | None:
    res = pearson_residuals(tri)
    if len(res) == 0:
        return None
    # The residuals are already standardised; phi was used in standardisation,
    # so re-derive phi from raw Pearson via (actual − fitted) / sqrt(fitted).
    raw = (res["actual"] - res["fitted"]) / np.sqrt(res["fitted"])
    n = len(res)
    paid_shape = tri["paid_loss"].values[0, 0].shape
    p = paid_shape[0] + (paid_shape[1] - 1)
    if n - p <= 0:
        return None
    return float(np.sum(raw**2) / (n - p))


def _var_mean_slope(tri) -> float | None:
    """Slope of log(var(inc)) vs log(mean(inc)) across dev periods.

    Slope ≈ 1 → Poisson-like, ≈ 2 → gamma-like.
    """
    paid = tri["paid_loss"].values[0, 0]
    n_origin, n_dev = paid.shape
    inc = np.full_like(paid, np.nan)
    inc[:, 0] = paid[:, 0]
    inc[:, 1:] = paid[:, 1:] - paid[:, :-1]
    means, vars_ = [], []
    for j in range(n_dev):
        col = inc[:, j]
        col = col[~np.isnan(col)]
        if len(col) >= 3 and col.mean() > 0 and col.var(ddof=1) > 0:
            means.append(col.mean())
            vars_.append(col.var(ddof=1))
    if len(means) < 3:
        return None
    lm = np.log(means)
    lv = np.log(vars_)
    slope, _ = np.polyfit(lm, lv, 1)
    return float(slope)


def _figure(line: str, atau: dict, lr: list, vm_slopes: list, phi: list) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # Age-to-age boxplot.
    devs = sorted(atau.keys())
    axes[0].boxplot([atau[d] for d in devs], tick_labels=[f"{d*12}" for d in devs], showfliers=False)
    axes[0].set_yscale("log")
    axes[0].set_title(f"{line} — age-to-age by dev (mo)")
    axes[0].axhline(1.0, color="gray", linestyle=":")
    # Loss ratio histogram.
    axes[1].hist(lr, bins=30, color="C0", edgecolor="white")
    axes[1].set_title(f"{line} — booked ULR (latest diagonal)")
    axes[1].set_xlabel("ultimate / premium")
    # Var-mean slope and phi distributions.
    axes[2].hist([s for s in vm_slopes if s is not None], bins=20, alpha=0.6, label="var-mean slope")
    axes[2].hist([p for p in phi if p is not None], bins=20, alpha=0.6, label="phi")
    axes[2].axvline(1.0, color="C0", linestyle=":")
    axes[2].axvline(2.0, color="C0", linestyle="--")
    axes[2].set_title(f"{line} — variance diagnostics")
    axes[2].legend()
    fig.tight_layout()
    out = FIGURES_DIR / f"01_descriptive_{line}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> int:
    print("Loading triangle JSON…", flush=True)
    full = load_full_triangle()

    summary_rows = []
    for line in LINES:
        print(f"  • {line}", flush=True)
        atau_pool: dict[int, list[float]] = {}
        lr_pool, phi_pool, vm_pool = [], [], []
        log_ulr_pool = []
        for snl_id, sub in iter_eligible_triangles(full, line):
            paid = sub["paid_loss"].values[0, 0]
            for d, fs in _atau_factors(paid).items():
                atau_pool.setdefault(d, []).extend(fs)
            lrs = _ultimate_to_premium(sub)
            lr_pool.extend(lrs.values())
            if lrs:
                log_ulr_pool.append(np.log(np.mean(list(lrs.values()))))
            phi_pool.append(_phi_for_triangle(sub))
            vm_pool.append(_var_mean_slope(sub))

        # Per-line summary.
        atau_summary = {
            f"atau_d{d*12}_p50": float(np.median(v)) for d, v in atau_pool.items()
        }
        atau_summary.update(
            {
                f"atau_d{d*12}_p90": float(np.percentile(v, 90))
                for d, v in atau_pool.items()
            }
        )
        row = {
            "line": line,
            "n_eligible": len(phi_pool),
            "ulr_mean": float(np.nanmean(lr_pool)),
            "ulr_sd": float(np.nanstd(lr_pool, ddof=1)),
            "log_ulr_between_company_sd": float(np.nanstd(log_ulr_pool, ddof=1))
            if len(log_ulr_pool) > 1
            else float("nan"),
            "phi_p50": float(np.nanmedian([p for p in phi_pool if p is not None])),
            "phi_p90": float(np.nanpercentile([p for p in phi_pool if p is not None], 90)),
            "var_mean_slope_p50": float(
                np.nanmedian([s for s in vm_pool if s is not None])
            ),
            **atau_summary,
        }
        summary_rows.append(row)
        _figure(line, atau_pool, lr_pool, vm_pool, phi_pool)

    df = pd.DataFrame(summary_rows)
    out_path = cache_path("descriptive_summary.parquet")
    df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
