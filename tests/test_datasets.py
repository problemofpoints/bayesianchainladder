"""Tests for the England sample triangles and the long-format loader."""

import chainladder as cl
import numpy as np
import pandas as pd
import pytest

from bayesianchainladder.datasets import load_england_sample
from bayesianchainladder.utils import long_to_triangle


def test_long_to_triangle_roundtrip():
    df = pd.DataFrame(
        {"origin": [2001, 2001, 2002], "dev": [12, 24, 12], "value": [1.0, 3.0, 2.0]}
    )
    tri = long_to_triangle(df, "value")
    assert tri.shape == (1, 1, 2, 2)
    assert tri.development.tolist() == [12, 24]
    np.testing.assert_allclose(
        np.asarray(tri.values)[0, 0], [[1.0, 3.0], [2.0, np.nan]], equal_nan=True
    )


def test_taylor_ashe_equals_genins():
    tri = load_england_sample("taylor_ashe")
    genins = cl.load_sample("genins")
    np.testing.assert_allclose(
        np.asarray(tri.values), np.asarray(genins.values), equal_nan=True
    )
    assert [str(o) for o in tri.origin[:2]] == ["2001", "2002"]


def test_liability_reference_values():
    tri = load_england_sample("liability")
    assert tri.shape == (1, 1, 10, 10)
    dev = cl.Development().fit_transform(tri)
    mack = cl.MackChainladder().fit(dev)
    total_reserve = float(np.nansum(np.asarray(mack.ibnr_.values)))
    total_se = float(np.asarray(mack.total_mack_std_err_).flatten()[0])
    assert total_reserve == pytest.approx(331_038, abs=1.0)
    assert total_se == pytest.approx(72_448, rel=1e-3)


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="taylor_ashe"):
        load_england_sample("nope")
