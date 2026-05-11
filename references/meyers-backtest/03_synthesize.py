"""03_synthesize.py — Combine chunk results and produce summary."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ANALYSIS_DIR = Path(__file__).resolve().parent
CACHE_DIR = ANALYSIS_DIR / "cache"

def load_all():
    files = sorted(glob.glob(str(CACHE_DIR / "backtest_chunk_*.parquet")))
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def main() -> int:
    df = load_all()
    df.to_parquet(CACHE_DIR / "backtest_all.parquet", index=False)
    print(f"Total rows: {len(df)}")
    ok = df[df.status == "ok"].copy()
    print(f"Successful fits: {len(ok)}")
    print(f"Failures by method:")
    print(df[df.status != "ok"].groupby(["method","status"]).size().head(20).to_string())

    # Per-line, per-method summary.
    ok["pct_err"] = (ok.mean_ultimate_est - ok.actual_ultimate) / ok.actual_ultimate
    ok["abs_pct_err"] = ok["pct_err"].abs()
    summary = ok.groupby(["line", "method"]).agg(
        n=("group_id", "count"),
        mean_implied_pctl=("implied_pctl", "mean"),
        median_implied_pctl=("implied_pctl", "median"),
        mean_pct_err=("pct_err", "mean"),
        median_abs_pct_err=("abs_pct_err", "median"),
        mean_cv=("cv_unpaid_est", "mean"),
    ).reset_index()
    print("\n=== Per-line, per-method summary ===")
    print(summary.to_string(index=False))

    summary.to_parquet(CACHE_DIR / "summary.parquet", index=False)

    # PP plots per line (one chart per line, 7 method curves overlaid).
    figures_dir = ANALYSIS_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)
    lines = sorted(ok.line.unique())
    methods = sorted(ok.method.unique())
    for line in lines:
        fig, ax = plt.subplots(figsize=(6, 5))
        sub = ok[ok.line == line]
        for m in methods:
            ms = sub[sub.method == m]["implied_pctl"].dropna().sort_values().values
            if ms.size == 0:
                continue
            x = np.linspace(0, 1, ms.size)
            ax.plot(ms, x, marker='o', markersize=3, alpha=0.7, label=m)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal (uniform)")
        ax.set_title(f"PP plot — {line}")
        ax.set_xlabel("implied percentile (sorted)")
        ax.set_ylabel("empirical CDF")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / f"pp_{line}.png", dpi=120)
        plt.close(fig)

    # Build README.
    md = ["# Meyers Back-Test Results\n"]
    md.append("Back-test of 7 reserve methods against the Meyers subset (200 triangles × 4 lines).\n")
    md.append(f"\n## Coverage\n\nSuccessful fits: {len(ok)} / {len(df)}\n")
    md.append("\nFailures by method:\n")
    fail_tab = df[df.status != "ok"].groupby(["method","status"]).size().reset_index(name="count")
    md.append(fail_tab.to_markdown(index=False) + "\n")
    md.append("\n## Per-Line, Per-Method Summary\n")
    md.append(summary.to_markdown(index=False, floatfmt=".3f") + "\n")
    md.append("\n## PP Plots\n")
    for line in lines:
        md.append(f"### {line}\n![{line} PP](figures/pp_{line}.png)\n")
    (ANALYSIS_DIR / "README.md").write_text("\n".join(md), encoding="utf-8")
    print("Wrote README.md and PP figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
