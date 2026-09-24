"""Sample triangles shipped with the package.

Data files come from Peter England's StochasticReserving repository
(https://github.com/DrPeterEngland/StochasticReserving, MIT licence); see
``bayesianchainladder/data/README.md`` for provenance.
"""

from __future__ import annotations

from importlib import resources
from typing import Literal

import numpy as np
import pandas as pd

from .utils import long_to_triangle

_FILES = {"taylor_ashe": "taylor_ashe.csv", "liability": "liability.csv"}


def load_england_sample(
    name: Literal["taylor_ashe", "liability"], first_origin: int = 2001
):
    """Load one of England's incremental triangles as a cumulative Triangle."""
    if name not in _FILES:
        raise ValueError(f"unknown sample {name!r}; choose from {sorted(_FILES)}")
    path = resources.files("bayesianchainladder.data").joinpath(_FILES[name])
    with path.open("r", encoding="utf-8-sig") as fh:
        raw = pd.read_csv(fh, index_col=0)
    rows = []
    for label, row in raw.iterrows():
        incr = row.to_numpy(dtype=float)
        cum = np.cumsum(np.nan_to_num(incr))
        for j, v in enumerate(incr):
            if np.isnan(v):
                break
            rows.append(
                {
                    "origin": first_origin + int(label) - 1,
                    "dev": 12 * (j + 1),
                    "value": cum[j],
                }
            )
    return long_to_triangle(pd.DataFrame(rows), "value")
