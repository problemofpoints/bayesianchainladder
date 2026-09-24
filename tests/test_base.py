"""Tests for BaseStochasticReserve ABC and MethodSummary dataclass."""

import math
from dataclasses import FrozenInstanceError

import chainladder as cl
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from bayesianchainladder.base import (
    DEFAULT_QUANTILES,
    BaseStochasticReserve,
    MethodSummary,
    ReserveSamples,
    incurred_to_paid,
)


class TestMethodSummary:
    def test_basic_construction(self):
        summary = MethodSummary(
            total_reserve_mean=100.0,
            total_reserve_stddev=10.0,
            total_reserve_75th_percentile=107.0,
            total_reserve_90th_percentile=113.0,
            total_reserve_95th_percentile=117.0,
        )
        assert summary.total_reserve_mean == 100.0
        assert summary.total_reserve_stddev == 10.0
        assert summary.total_reserve_75th_percentile == 107.0
        assert summary.total_reserve_90th_percentile == 113.0
        assert summary.total_reserve_95th_percentile == 117.0

    def test_cv_property(self):
        summary = MethodSummary(
            total_reserve_mean=100.0,
            total_reserve_stddev=20.0,
            total_reserve_75th_percentile=0.0,
            total_reserve_90th_percentile=0.0,
            total_reserve_95th_percentile=0.0,
        )
        assert summary.total_reserve_cv == pytest.approx(0.2)

    def test_cv_with_zero_mean_returns_nan(self):
        summary = MethodSummary(
            total_reserve_mean=0.0,
            total_reserve_stddev=10.0,
            total_reserve_75th_percentile=0.0,
            total_reserve_90th_percentile=0.0,
            total_reserve_95th_percentile=0.0,
        )
        assert math.isnan(summary.total_reserve_cv)

    def test_cv_with_nan_mean_returns_nan(self):
        summary = MethodSummary(
            total_reserve_mean=float("nan"),
            total_reserve_stddev=10.0,
            total_reserve_75th_percentile=0.0,
            total_reserve_90th_percentile=0.0,
            total_reserve_95th_percentile=0.0,
        )
        assert math.isnan(summary.total_reserve_cv)

    def test_cv_with_negative_mean(self):
        # CV uses abs(mean) so negative means produce positive CVs
        summary = MethodSummary(
            total_reserve_mean=-100.0,
            total_reserve_stddev=20.0,
            total_reserve_75th_percentile=0.0,
            total_reserve_90th_percentile=0.0,
            total_reserve_95th_percentile=0.0,
        )
        assert summary.total_reserve_cv == pytest.approx(0.2)

    def test_dataclass_is_frozen(self):
        summary = MethodSummary(
            total_reserve_mean=100.0,
            total_reserve_stddev=10.0,
            total_reserve_75th_percentile=0.0,
            total_reserve_90th_percentile=0.0,
            total_reserve_95th_percentile=0.0,
        )
        with pytest.raises(FrozenInstanceError):
            summary.total_reserve_mean = 200.0  # type: ignore[misc]


class _StubReserve(BaseStochasticReserve):
    """Minimal subclass for exercising the base helpers without a real fit."""

    def fit(self, triangle, samples=None, random_seed=None):
        from bayesianchainladder.utils import _extract_period_value, validate_triangle

        validate_triangle(triangle)
        self.triangle_ = triangle.copy()

        if samples is None:
            # Default: 100 samples per origin, drawn from a fixed normal so
            # the tests are deterministic.
            origins = sorted({_extract_period_value(o) for o in triangle.origin})
            rng = np.random.default_rng(random_seed if random_seed is not None else 42)
            arr = rng.normal(loc=1000.0, scale=100.0, size=(len(origins), 100))
            self.reserves_posterior_ = xr.DataArray(
                arr,
                dims=["origin", "sample"],
                coords={"origin": origins, "sample": np.arange(100)},
            )
        else:
            self.reserves_posterior_ = samples

        self._build_reserve_summaries()
        self._is_fitted = True
        return self


@pytest.fixture
def stub_fitted():
    triangle = cl.load_sample("raa")
    return _StubReserve().fit(triangle, random_seed=42)


class TestBaseStochasticReserve:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            BaseStochasticReserve()  # type: ignore[abstract]

    def test_check_is_fitted_raises_before_fit(self):
        stub = _StubReserve()
        with pytest.raises(ValueError, match="has not been fitted"):
            stub.summary()

    def test_fit_populates_attributes(self, stub_fitted):
        assert stub_fitted._is_fitted is True
        assert stub_fitted.triangle_ is not None
        assert stub_fitted.reserves_posterior_ is not None
        assert stub_fitted.ibnr_ is not None
        assert stub_fitted.ultimate_ is not None

    def test_ibnr_columns(self, stub_fitted):
        assert list(stub_fitted.ibnr_.columns) == [
            "mean",
            "std",
            "median",
            "5%",
            "25%",
            "75%",
            "95%",
        ]

    def test_ultimate_columns(self, stub_fitted):
        assert list(stub_fitted.ultimate_.columns) == [
            "paid_to_date",
            "mean",
            "std",
            "median",
            "5%",
            "25%",
            "75%",
            "95%",
        ]

    def test_ultimate_mean_equals_paid_plus_ibnr(self, stub_fitted):
        diff = (
            stub_fitted.ultimate_["mean"]
            - stub_fitted.ultimate_["paid_to_date"]
            - stub_fitted.ibnr_["mean"]
        )
        assert (diff.abs() < 1e-9).all()

    def test_summary_includes_total_row_by_default(self, stub_fitted):
        summary = stub_fitted.summary()
        assert "Total" in summary.index

    def test_summary_can_exclude_total_row(self, stub_fitted):
        summary = stub_fitted.summary(include_totals=False)
        assert "Total" not in summary.index

    def test_summary_has_multiindex_columns(self, stub_fitted):
        summary = stub_fitted.summary()
        assert summary.columns.nlevels == 2
        assert set(summary.columns.get_level_values(0)) == {"Ultimate", "IBNR"}

    def test_sample_reserves_returns_array_of_correct_shape(self, stub_fitted):
        samples = stub_fitted.sample_reserves(n_samples=500, random_seed=0)
        assert isinstance(samples, np.ndarray)
        assert samples.shape == (500,)

    def test_sample_reserves_is_finite(self, stub_fitted):
        samples = stub_fitted.sample_reserves(n_samples=200, random_seed=0)
        assert np.all(np.isfinite(samples))

    def test_total_summary_returns_method_summary(self, stub_fitted):
        from bayesianchainladder.base import MethodSummary

        result = stub_fitted.total_summary()
        assert isinstance(result, MethodSummary)
        assert np.isfinite(result.total_reserve_mean)
        assert result.total_reserve_stddev > 0
        assert (
            result.total_reserve_95th_percentile > result.total_reserve_75th_percentile
        )

    def test_total_summary_cv_is_positive(self, stub_fitted):
        result = stub_fitted.total_summary()
        assert result.total_reserve_cv > 0


class TestPackageExports:
    def test_base_class_is_exported(self):
        import bayesianchainladder as bcl

        assert hasattr(bcl, "BaseStochasticReserve")
        assert hasattr(bcl, "MethodSummary")

    def test_new_estimators_are_exported(self):
        import bayesianchainladder as bcl

        assert hasattr(bcl, "MackChainLadder")
        assert hasattr(bcl, "BootstrapODPChainLadder")
        assert hasattr(bcl, "CorrelatedBootstrapChainLadder")
        assert hasattr(bcl, "CorrelatedBootstrapODPSample")

    def test_existing_estimators_still_exported(self):
        import bayesianchainladder as bcl

        assert hasattr(bcl, "BayesianChainLadderGLM")
        assert hasattr(bcl, "BayesianCSR")


class TestFullCumulativePosterior:
    @pytest.fixture
    def toy(self):
        df = pd.DataFrame(
            {
                "origin": [2001, 2001, 2001, 2002, 2002, 2003],
                "dev": [12, 24, 36, 12, 24, 12],
                "value": [100.0, 150.0, 160.0, 110.0, 170.0, 120.0],
            }
        )
        eval_year = df["origin"] + df["dev"] // 12 - 1
        df["dev_date"] = pd.to_datetime(eval_year.astype(str) + "-12-31")
        tri = cl.Triangle(
            df,
            origin="origin",
            development="dev_date",
            columns=["value"],
            cumulative=True,
            origin_format="%Y",
        )
        # (origin, dev, sample) cumulative; NaN-free, observed cells constant
        full = np.full((3, 3, 4), np.nan)
        full[0, :, :] = np.array([[100.0], [150.0], [160.0]])
        full[1, 0, :] = 110.0
        full[1, 1, :] = 170.0
        full[1, 2, :] = [180.0, 182.0, 184.0, 186.0]
        full[2, 0, :] = 120.0
        full[2, 1, :] = [170.0, 175.0, 180.0, 185.0]
        full[2, 2, :] = [180.0, 190.0, 200.0, 210.0]
        return tri, full

    def test_reserve_samples_container(self, toy):
        tri, full = toy
        reserves = xr.DataArray(
            full[:, -1, :] - np.array([[160.0], [170.0], [120.0]]),
            dims=["origin", "sample"],
            coords={"origin": [2001, 2002, 2003], "sample": np.arange(4)},
        )
        rs = ReserveSamples(tri, reserves)
        assert isinstance(rs, BaseStochasticReserve)
        assert rs.ibnr_.loc[2003, "mean"] == pytest.approx(75.0)
        assert rs.full_cumulative_posterior_ is None
        with pytest.raises(ValueError, match="per-cell"):
            rs._require_full_posterior()
        with pytest.raises(NotImplementedError):
            rs.fit(tri)

    def test_full_posterior_helpers(self, toy):
        tri, full = toy
        rs = ReserveSamples(
            tri,
            xr.DataArray(
                np.zeros((3, 4)),
                dims=["origin", "sample"],
                coords={"origin": [2001, 2002, 2003], "sample": np.arange(4)},
            ),
        )
        rs._set_full_cumulative_posterior(full, [2001, 2002, 2003], [12, 24, 36])
        assert rs.full_cumulative_posterior_.dims == ("origin", "dev", "sample")
        derived = rs._reserves_from_full_posterior()
        np.testing.assert_allclose(
            derived.sel(origin=2002).values, [10.0, 12.0, 14.0, 16.0]
        )
        np.testing.assert_allclose(
            derived.sel(origin=2003).values, [60.0, 70.0, 80.0, 90.0]
        )
        incr = rs.incremental_posterior()
        np.testing.assert_allclose(
            incr.sel(origin=2003, dev=24).values, [50.0, 55.0, 60.0, 65.0]
        )
        fut = rs.future_incremental_posterior()
        assert (fut.sel(origin=2001).values == 0).all()
        np.testing.assert_allclose(
            fut.sel(origin=2003, dev=36).values, [10.0, 15.0, 20.0, 25.0]
        )

    def test_full_cumulative_posterior_dim_order_normalized(self, toy):
        tri, full = toy
        # full is (origin, dev, sample); hand ReserveSamples a differently
        # ordered but equally valid DataArray and confirm it gets normalized.
        reordered = xr.DataArray(
            np.transpose(full, (2, 0, 1)),
            dims=["sample", "origin", "dev"],
            coords={
                "origin": [2001, 2002, 2003],
                "dev": [12, 24, 36],
                "sample": np.arange(4),
            },
        )
        rs = ReserveSamples(
            tri,
            xr.DataArray(
                np.zeros((3, 4)),
                dims=["origin", "sample"],
                coords={"origin": [2001, 2002, 2003], "sample": np.arange(4)},
            ),
            full_cumulative_posterior=reordered,
        )
        assert rs.full_cumulative_posterior_.dims == ("origin", "dev", "sample")
        np.testing.assert_allclose(
            rs.incremental_posterior().sel(origin=2003, dev=24).values,
            [50.0, 55.0, 60.0, 65.0],
        )

    def test_summary_statistics(self, toy):
        tri, full = toy
        reserves = xr.DataArray(
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [10.0, 12.0, 14.0, 16.0],
                    [60.0, 70.0, 80.0, 90.0],
                ]
            ),
            dims=["origin", "sample"],
            coords={"origin": [2001, 2002, 2003], "sample": np.arange(4)},
        )
        rs = ReserveSamples(tri, reserves)
        stats = rs.summary_statistics()
        assert list(stats.index) == [2001, 2002, 2003, "Total"]
        assert stats.loc["Total", "mean"] == pytest.approx(88.0)
        assert stats.loc[2003, "min"] == 60.0 and stats.loc[2003, "max"] == 90.0
        assert "99.5%" in stats.columns and "0.5%" in stats.columns
        assert stats.loc[2002, "cov"] == pytest.approx(
            np.std([10, 12, 14, 16], ddof=1) / 13.0
        )
        ults = rs.summary_statistics(output="ultimates")
        assert ults.loc[2003, "mean"] == pytest.approx(120.0 + 75.0)
        with pytest.raises(ValueError):
            rs.summary_statistics(output="nonsense")
        assert len(DEFAULT_QUANTILES) == 11

    def test_method_summary_new_fields_default(self):
        s = MethodSummary(1.0, 2.0, 3.0, 4.0, 5.0)
        assert math.isnan(s.total_reserve_99_5th_percentile)
        assert math.isnan(s.total_reserve_min) and math.isnan(s.total_reserve_max)

    def test_total_summary_populates_new_fields(self, toy):
        tri, _ = toy
        reserves = xr.DataArray(
            np.array([[0.0] * 4, [10.0, 12.0, 14.0, 16.0], [60.0, 70.0, 80.0, 90.0]]),
            dims=["origin", "sample"],
            coords={"origin": [2001, 2002, 2003], "sample": np.arange(4)},
        )
        s = ReserveSamples(tri, reserves).total_summary()
        assert s.total_reserve_min == pytest.approx(70.0)
        assert s.total_reserve_max == pytest.approx(106.0)
        assert s.total_reserve_99_5th_percentile == pytest.approx(
            np.quantile([70, 82, 94, 106], 0.995)
        )


class TestScalingAndIncurredToPaid:
    @pytest.fixture
    def fitted(self):
        from bayesianchainladder.bootstrap import BootstrapODPChainLadder

        return BootstrapODPChainLadder(n_sims=500, random_seed=4).fit(
            cl.load_sample("genins")
        )

    def test_additive_preserves_sd_and_hits_target(self, fitted):
        origins = list(fitted.reserves_posterior_.coords["origin"].values)
        paid = fitted._paid_to_date().reindex(origins)
        target = paid + 1.1 * fitted.ibnr_["mean"]
        scaled = fitted.scale_to_target(target, method="additive")
        assert isinstance(scaled, ReserveSamples)
        np.testing.assert_allclose(
            scaled.ibnr_["std"].values, fitted.ibnr_["std"].values, rtol=1e-9
        )
        np.testing.assert_allclose(
            scaled.ultimate_["mean"].values, target.values, rtol=1e-9
        )

    def test_multiplicative_preserves_cov(self, fitted):
        origins = list(fitted.reserves_posterior_.coords["origin"].values)
        paid = fitted._paid_to_date().reindex(origins)
        target = paid + 1.1 * fitted.ibnr_["mean"]
        scaled = fitted.scale_to_target(target, method="multiplicative")
        base_cov = (fitted.ibnr_["std"] / fitted.ibnr_["mean"]).values[1:]
        new_cov = (scaled.ibnr_["std"] / scaled.ibnr_["mean"]).values[1:]
        np.testing.assert_allclose(new_cov, base_cov, rtol=1e-9)
        np.testing.assert_allclose(
            scaled.ibnr_["mean"].values[1:],
            1.1 * fitted.ibnr_["mean"].values[1:],
            rtol=1e-9,
        )

    def test_per_origin_method_dict_and_validation(self, fitted):
        origins = list(fitted.reserves_posterior_.coords["origin"].values)
        paid = fitted._paid_to_date().reindex(origins)
        target = paid + fitted.ibnr_["mean"]
        methods = {
            o: ("additive" if k < 5 else "multiplicative")
            for k, o in enumerate(origins)
        }
        scaled = fitted.scale_to_target(target, method=methods)
        np.testing.assert_allclose(
            scaled.ibnr_["mean"].values, fitted.ibnr_["mean"].values, rtol=1e-9
        )
        with pytest.raises(ValueError, match="method"):
            fitted.scale_to_target(target, method="geometric")
        with pytest.raises(ValueError, match="origin"):
            fitted.scale_to_target(target.iloc[:3])

    def test_incurred_to_paid(self):
        from bayesianchainladder.bootstrap import BootstrapODPChainLadder

        clrd = cl.load_sample("clrd").groupby("LOB").sum().loc["wkcomp"]
        incurred = clrd["IncurLoss"]
        paid = clrd["CumPaidLoss"]
        model = BootstrapODPChainLadder(n_sims=300, random_seed=8).fit(incurred)
        converted = incurred_to_paid(model, paid)
        assert isinstance(converted, ReserveSamples)
        latest_inc = model._paid_to_date().values
        latest_paid = converted._paid_to_date().values
        expected_mean = model.ibnr_["mean"].values + latest_inc - latest_paid
        np.testing.assert_allclose(
            converted.ibnr_["mean"].values, expected_mean, rtol=1e-9
        )
        np.testing.assert_allclose(
            converted.ibnr_["std"].values, model.ibnr_["std"].values, rtol=1e-9
        )
