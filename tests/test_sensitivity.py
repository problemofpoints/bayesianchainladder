# tests/test_sensitivity.py
"""Leave-one-link-ratio-out influence analysis (England's Sensitivities)."""

import pytest

from bayesianchainladder.analytic import mack_analytic_rmsep
from bayesianchainladder.datasets import load_england_sample
from bayesianchainladder.sensitivity import link_ratio_sensitivity, top_influential


@pytest.fixture(scope="module")
def liability():
    return load_england_sample("liability")


@pytest.fixture(scope="module")
def result(liability):
    return link_ratio_sensitivity(liability)


def test_columns_and_row_count(result):
    expected = {
        "origin", "dev", "reserve", "reserve_sd", "reserve_cov",
        "reserve_diff", "sd_diff", "cov_diff", "reserve_rank", "sd_rank", "cov_rank",
    }
    assert expected <= set(result.columns)
    # 45 available ratios minus the 9 columns... only ratios whose column keeps >= 1 other ratio
    # are evaluated: the single ratio in the last column (origin 2001, dev 108) is skipped.
    assert len(result) == 44
    assert result["sd_rank"].min() == 1 and result["sd_rank"].max() == 44
    assert result.attrs["base_reserve"] == pytest.approx(331_038, abs=1.0)


def test_most_influential_ratio_is_origin3_dev6(result):
    top = top_influential(result, n=1, by="sd")
    assert top == [("2003", 72)]
    row = result[(result.origin == 2003) & (result.dev == 72)].iloc[0]
    assert row["reserve"] == pytest.approx(289_946, rel=1e-3)
    assert row["reserve_sd"] == pytest.approx(38_076, rel=1e-3)
    assert row["sd_diff"] < 0 and row["cov_rank"] == 1 and row["reserve_rank"] == 1


def test_top_n_returns_drop_list_usable_by_mack(result, liability):
    top3 = top_influential(result, n=3, by="sd")
    assert len(top3) == 3 and all(isinstance(o, str) and isinstance(d, int) for o, d in top3)
    reduced = mack_analytic_rmsep(liability, drop=top3)
    base = mack_analytic_rmsep(liability)
    assert reduced.total_sd < 0.6 * base.total_sd
    with pytest.raises(ValueError):
        top_influential(result, n=1, by="nonsense")


def test_base_drop_is_respected(liability):
    res = link_ratio_sensitivity(liability, drop=[("2003", 72)])
    assert res.attrs["base_reserve"] == pytest.approx(289_946, rel=1e-3)
    assert not ((res.origin == 2003) & (res.dev == 72)).any()
