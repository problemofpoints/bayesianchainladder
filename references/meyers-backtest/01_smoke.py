"""01_smoke.py — Smoke test: 4 triangles × 7 methods = 28 fits."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import reservetestr as rt
from _common import MEYERS_LINES
from methods import (
    testr_correlated_bootstrap_odp,
    testr_bayesian_csr,
    testr_glm_m2,
    testr_glm_m5_cal,
    testr_glm_mt5_cal,
)


def main():
    records = rt.build_triangle_records()
    # 1 record per line.
    smoke = {}
    for r in records:
        if r.line not in smoke:
            smoke[r.line] = r
        if len(smoke) == 4:
            break

    rows = []
    methods = [
        ("mack",            rt.testr_mack_chainladder, {}),
        ("bootstrap_odp",   rt.testr_bootstrap_odp,    {"n_sims": 1000, "hat_adj": True, "random_state": 22}),
        ("corr_boot_odp",   testr_correlated_bootstrap_odp, {"n_sims": 1000}),
        ("bayesian_csr",    testr_bayesian_csr, {}),
        ("glm_m2",          testr_glm_m2, {}),
        ("glm_m5_cal",      testr_glm_m5_cal, {}),
        ("glm_mt5_cal",     testr_glm_mt5_cal, {}),
    ]
    for line, r in smoke.items():
        print(f"\n=== {line} — {r.company} ({r.group_id}) ===")
        for method_label, fn, kwargs in methods:
            try:
                kw = dict(kwargs)
                # All our custom methods need `line` and `group_id` kwargs.
                if method_label not in ("mack", "bootstrap_odp"):
                    kw["line"] = line
                    kw["group_id"] = r.group_id
                res = fn(r.train_triangles, r.test_triangles,
                         loss_type="paid", actual_ultimates=r.actual_ultimates, **kw)
                if res:
                    print(f"  {method_label:18s} mean_ult={res['mean_ultimate_est']:.0f}  "
                          f"cv={res['cv_unpaid_est']:.3f}  pctl={res['implied_pctl']:.2f}")
                    rows.append({"line": line, "group_id": r.group_id, "method": method_label, **res})
                else:
                    print(f"  {method_label:18s} (None)")
            except Exception as e:
                print(f"  {method_label:18s} ERROR: {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()
    pd.DataFrame(rows).to_parquet(Path(__file__).parent / "cache" / "smoke_results.parquet")
    print(f"\nSaved {len(rows)} rows to cache/smoke_results.parquet")


if __name__ == "__main__":
    main()
