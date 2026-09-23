# bayesianchainladder/sensitivity.py
"""Identify link ratios that drive reserve volatility.

Ports ``Sensitivities`` from Peter England's StochasticReserving repository
(https://github.com/DrPeterEngland/StochasticReserving, MIT licence): exclude
each link ratio in turn, re-apply Mack's model analytically, and rank the
change in total reserve, its standard deviation and coefficient of variation.
"""

from __future__ import annotations

import pandas as pd

from ._triangle_ops import DropList, cumulative_array, link_ratio_mask
from .analytic import mack_analytic_rmsep

_RANK_COLUMNS = {"reserve": "reserve_rank", "sd": "sd_rank", "cov": "cov_rank"}


def link_ratio_sensitivity(triangle, drop: DropList = None) -> pd.DataFrame:
    cum, origins, devs = cumulative_array(triangle)
    base_drop = list(drop or [])
    base = mack_analytic_rmsep(triangle, drop=base_drop)
    mask = link_ratio_mask(cum, base_drop, origins, devs)
    n_j = mask.sum(axis=0)

    rows = []
    for i in range(cum.shape[0]):
        for j in range(cum.shape[1] - 1):
            if mask[i, j] == 0 or n_j[j] <= 1:
                continue  # excluding the only ratio in a column leaves no factor estimate
            res = mack_analytic_rmsep(triangle, drop=base_drop + [(str(origins[i]), devs[j])])
            rows.append(
                {
                    "origin": origins[i],
                    "dev": devs[j],
                    "reserve": res.total_reserve,
                    "reserve_sd": res.total_sd,
                    "reserve_cov": res.total_cov,
                    "reserve_diff": res.total_reserve - base.total_reserve,
                    "sd_diff": res.total_sd - base.total_sd,
                    "cov_diff": res.total_cov - base.total_cov,
                }
            )
    df = pd.DataFrame(rows)
    for key, rank_col in _RANK_COLUMNS.items():
        diff_col = f"{key}_diff"
        df[rank_col] = df[diff_col].rank(method="first", ascending=True).astype(int)
    df.attrs.update(
        base_reserve=base.total_reserve, base_sd=base.total_sd, base_cov=base.total_cov
    )
    return df.sort_values("sd_rank").reset_index(drop=True)


def top_influential(result: pd.DataFrame, n: int = 3, by: str = "sd") -> list[tuple[str, int]]:
    """The ``n`` ratios with the largest reduction in ``by`` ∈ {reserve, sd, cov},
    as chainladder-style ``(origin_label, dev_months)`` drop tuples."""
    if by not in _RANK_COLUMNS:
        raise ValueError(f"by must be one of {sorted(_RANK_COLUMNS)}")
    top = result.nsmallest(n, _RANK_COLUMNS[by])
    return [(str(int(o)), int(d)) for o, d in zip(top["origin"], top["dev"], strict=True)]
