"""Fast checks against England's published reference values.

Loads ``bayesianchainladder/data/england_reference_values.json`` (transcribed
from the executed EVW_2019 / EV_2006_PredictiveDistributions notebooks in
https://github.com/DrPeterEngland/StochasticReserving, MIT licence) and
compares it to the analytic, bootstrap and discounted outputs of this
package's Taylor & Ashe triangle. This validates the JSON transcription
against the existing library; it does not itself build or execute the
documentation notebooks (see ``tests/test_notebooks.py``).
"""

import json
from importlib import resources

import numpy as np
import pytest

from bayesianchainladder import (
    MackBootstrap,
    discounted_reserves,
    load_england_sample,
    mack_analytic_rmsep,
    odp_analytic_rmsep,
)

with (
    resources.files("bayesianchainladder.data")
    .joinpath("england_reference_values.json")
    .open("r", encoding="utf-8") as fh
):
    REF = json.load(fh)


@pytest.fixture(scope="module")
def tri():
    return load_england_sample("taylor_ashe")


def test_provenance_commit():
    assert REF["_provenance"]["commit"] == "e7ed85a29dba64db1192140e504f4e09cf149134"


def test_odp_analytic_matches_ml_reference(tri):
    ref = REF["ev_2006"]["ml_analytic_odp_constant"]
    result = odp_analytic_rmsep(tri, scale="constant")
    assert result.total_sd == pytest.approx(ref["total_sd"], abs=1.0)
    np.testing.assert_allclose(result.reserve_sd, ref["reserve_sd"], atol=1.0)
    assert np.sqrt(result.scale[0]) == pytest.approx(ref["sqrt_scale"], abs=0.01)
    params = REF["ev_2006"]["ml_parameters"]
    np.testing.assert_allclose(result.coefficients, params["estimate"], atol=5e-4)


def test_mack_analytic_matches_evw_reference(tri):
    ref = REF["evw_2019"]["table2_analytic_mack"]
    result = mack_analytic_rmsep(tri)
    assert result.total_reserve == pytest.approx(ref["total_reserve"], abs=1)
    assert result.total_sd == pytest.approx(
        ref["total_sd"], rel=0.003
    )  # within 0.3%; see JSON note on the last-sigma rule


def test_mack_bootstrap_matches_evw_table4(tri):
    ref = REF["evw_2019"]["table4_bootstrap_and_one_year_cdr"]
    boot = MackBootstrap(n_sims=10_000, random_seed=101).fit(tri)
    total = boot.total_summary()
    assert total.total_reserve_mean == pytest.approx(ref["total_avg_reserve"], rel=0.01)
    assert total.total_reserve_stddev == pytest.approx(
        ref["total_bootstrap_sd"], rel=0.03
    )


def test_one_year_cdr_sd_matches_evw_table4(tri):
    from bayesianchainladder import claims_development_result

    ref = REF["evw_2019"]["table4_bootstrap_and_one_year_cdr"]
    boot = MackBootstrap(n_sims=10_000, random_seed=101).fit(tri)
    cdr = claims_development_result(boot)
    one_year_total_sd = (
        cdr.summary().query("future_period == 1 and origin == 'Total'")["sd"].iloc[0]
    )
    assert one_year_total_sd == pytest.approx(ref["total_cdr_sd"], rel=0.03)


def test_discounted_reserves_match_evw_table5(tri):
    ref = REF["evw_2019"]["table5_discounted_3pct"]
    boot = MackBootstrap(n_sims=10_000, random_seed=101).fit(tri)
    disc = discounted_reserves(boot, 0.03, 0.5)
    total_mean = float(disc.sum("origin").mean("sample"))
    assert total_mean == pytest.approx(ref["total_avg"], rel=0.01)
