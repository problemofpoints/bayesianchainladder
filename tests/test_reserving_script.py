"""Fast tests for the standalone reserving script (no MCMC)."""

from __future__ import annotations

import importlib.util
import pathlib

import chainladder as cl
import numpy as np
import pandas as pd
import pytest

SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_stochastic_reserving.py"
)
EXAMPLE = SCRIPT.parent / "example_input.csv"


@pytest.fixture(scope="module")
def rs():
    spec = importlib.util.spec_from_file_location("run_stochastic_reserving", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example_df():
    return pd.read_csv(EXAMPLE)


class TestDfToTriangle:
    def test_matches_date_based_construction(self, rs, example_df):
        tri = rs.df_to_triangle(example_df, value_col="paid")

        work = example_df[["origin", "dev", "paid"]].copy()
        eval_year = work["origin"].astype(int) + work["dev"].astype(int) // 12 - 1
        work["dev_date"] = pd.to_datetime(eval_year.astype(str) + "-12-31")
        expected = cl.Triangle(
            data=work,
            origin="origin",
            development="dev_date",
            columns=["paid"],
            cumulative=True,
            origin_format="%Y",
        )

        assert tri.shape == expected.shape == (1, 1, 10, 10)
        assert tri.development.tolist() == list(range(12, 121, 12))
        assert tri.valuation_date.year == expected.valuation_date.year == 2010
        np.testing.assert_allclose(
            np.nan_to_num(tri.values), np.nan_to_num(expected.values)
        )
        assert (np.isnan(tri.values) == np.isnan(expected.values)).all()


class TestMack:
    @pytest.mark.parametrize("interp", ["mack", "log-linear"])
    def test_sigma_interpolation_options_run(self, rs, example_df, interp):
        tri = rs.df_to_triangle(example_df, value_col="paid")
        samples, ibnr = rs._run_mack(
            tri, n_samples=200, random_seed=1, sigma_interpolation=interp
        )
        assert samples.shape == (10, 200)
        assert np.isfinite(samples).all()
        assert ibnr.shape == (10,)

    def test_cli_default_is_mack(self, rs):
        args = rs.parse_args(["--input", "x.csv"])
        assert args.mack_sigma_interpolation == "mack"


class TestBarnettZehnwirth:
    def test_returns_positive_ibnr_samples_close_to_chain_ladder(self, rs):
        tri = cl.load_sample("genins")
        samples = rs._run_bz(tri, n_sims=400, random_seed=7)

        assert samples.shape == (10, 400)
        assert np.isfinite(samples).all()
        assert (samples >= 0).all()
        assert (samples[0] == 0).all()  # first origin is fully developed
        assert (samples[1:] > 0).all()

        cl_ult = cl.Chainladder().fit(cl.Development().fit_transform(tri)).ultimate_
        cl_ibnr = float(np.nansum(cl_ult.values)) - float(
            np.nansum(tri.latest_diagonal.values)
        )
        bz_ibnr = float(samples.sum(axis=0).mean())
        assert abs(bz_ibnr - cl_ibnr) / cl_ibnr < 0.25

    def test_rejects_non_positive_incrementals(self, rs):
        with pytest.raises(ValueError, match="positive incremental"):
            rs._run_bz(cl.load_sample("raa"), n_sims=10, random_seed=0)

    def test_cli_accepts_bz(self, rs):
        args = rs.parse_args(["--input", "x.csv", "--methods", "bz"])
        assert args.methods == ["bz"]
        assert args.bz_formula == "C(origin)+C(development)"
        assert "bz" in rs.parse_args(["--input", "x.csv"]).methods
