"""Generate docs/notebooks/modus_operandi.ipynb.

The notebook adapts the 15-step "Stochastic Reserving: Modus Operandi" example
by Peter England (EMC Actuarial and Analytics Ltd) from
https://github.com/DrPeterEngland/StochasticReserving/blob/main/Python_Examples/Example_Modus_Operandi.ipynb
(MIT licence) to the bayesianchainladder API. Commentary is paraphrased.

Usage: python build_modus_operandi.py [output_dir]
"""

from __future__ import annotations

import pathlib
import sys

import nbformat as nbf

SOURCE = "https://github.com/DrPeterEngland/StochasticReserving"

CELLS: list[tuple[str, str]] = []  # (kind, source)


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip()))


def code(text: str) -> None:
    CELLS.append(("code", text.strip()))


md(f"""
# Stochastic reserving: a modus operandi

Adapted from Peter England's *Example Modus Operandi* notebook in the
[StochasticReserving repository]({SOURCE}) (`Python_Examples/Example_Modus_Operandi.ipynb`,
MIT licence, provided by EMC Actuarial and Analytics Ltd as an educational resource).
The fifteen steps and their rationale are his; the code uses `bayesianchainladder`, and the
commentary is paraphrased. Numbers differ slightly from the original because chainladder
extrapolates the last Mack sigma log-linearly whereas England takes the minimum of the previous
two, and because the simulation seeds differ.

**Why a modus operandi?** Quantifying reserve variability starts with understanding what drives
it, then understanding each model's characteristics and limits, and only then selecting a result.
The steps below are a guide, not a recipe.

This notebook covers the lifetime ("ultimo") view. The one-year view is a short epilogue that
points to `claims_development_result`.
""")

md("## Steps 1 & 2: look at the data and fit a baseline chain ladder")
code("""
import numpy as np
import pandas as pd
import chainladder as cl
import matplotlib.pyplot as plt

from bayesianchainladder import (
    CorrelatedBootstrapChainLadder,
    MackBootstrap,
    claims_development_result,
    link_ratio_sensitivity,
    load_england_sample,
    mack_analytic_rmsep,
    plot_fan_chart,
    plot_scaled_residuals,
    plot_sensitivity_heatmap,
    top_influential,
)

pd.options.display.float_format = "{:,.0f}".format
tri = load_england_sample("liability")
incremental = tri.cum_to_incr().to_frame(origin_as_datetime=False)
display(incremental)
display(tri.to_frame(origin_as_datetime=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
incremental.T.plot(ax=axes[0], title="Incremental claims by development", legend=False)
tri.to_frame(origin_as_datetime=False).T.plot(ax=axes[1], title="Cumulative claims by development", legend=False)
plt.show()

dev = cl.Development().fit_transform(tri)
link_ratios = (tri.values[0, 0, :, 1:] / tri.values[0, 0, :, :-1])
display(pd.DataFrame(link_ratios, index=tri.origin.year, columns=tri.development[:-1]).round(3))
print("Volume-weighted factors:", np.round(dev.ldf_.values.flatten(), 3))
cl_result = cl.Chainladder().fit(dev)
display(pd.DataFrame({"latest": tri.latest_diagonal.values[0, 0, :, 0],
                      "reserve": cl_result.ibnr_.values[0, 0, :, 0],
                      "ultimate": cl_result.ultimate_.values[0, 0, :, 0]}, index=tri.origin.year))
print("Total reserve:", f"{float(np.nansum(cl_result.ibnr_.values)):,.0f}")
""")
md("""
**Commentary.** The incremental graph shows a large payment at development period 7 for origin
period 3 (2003). It produces a large link ratio between development periods 6 and 7 for that
origin, which reverses the otherwise decreasing pattern of the volume-weighted factors at that
point. Including every ratio gives a total reserve of about 331,000.
""")

md("## Step 3: Mack's model applied analytically")
code("""
mack = mack_analytic_rmsep(tri)
display(mack.to_frame().style.format({"reserve": "{:,.0f}", "sd": "{:,.0f}", "cov": "{:.1%}"}))
""")
md("""
**Commentary.** A coefficient of variation around 22% on the total reserve is high for a
triangle whose cumulative development looks stable, so something specific is driving it.
""")

md("## Step 4: residuals and Mack's sigma")
code("""
boot_all = MackBootstrap(n_sims=10_000, random_seed=100).fit(tri)
sigma_original = boot_all.sigma_.copy()
print("Mack sigma by development ratio:", np.round(sigma_original, 2))
display(pd.DataFrame(boot_all.scaled_residuals_, index=tri.origin.year, columns=tri.development[:-1]).round(2))
for by in ("origin", "dev", "calendar"):
    plot_scaled_residuals(boot_all.scaled_residuals_, by=by, sigma=sigma_original)
    plt.show()
""")
md("""
**Commentary.** Sigma is the root mean square of the bias-adjusted unscaled residuals in each
development period, so one large residual inflates it. Here sigma spikes at development ratio 6
(the 72->84 month ratio), driven by origin 2003. Scaled residuals should look like i.i.d. standard
normals: pattern-free by origin, development and calendar period, with roughly 95% inside ±2.
""")

md("## Step 5: bootstrap Mack's model with all ratios included")
code("""
display(boot_all.summary_statistics("reserves").style.format("{:,.0f}"))
display(boot_all.summary_statistics("ultimates").style.format("{:,.0f}"))
for origin in (2002, 2004, 2006, 2008, 2010):
    plot_fan_chart(boot_all, origin)
    plt.show()
""")
md("""
**Commentary.** The bootstrap mean, standard deviation and CoV match the analytic Mack results
closely, which is the first check to make. The minimum simulated reserves are negative for the
older origins: Mack's model is a model of link ratios and simulated cumulatives can fall. That
may be acceptable for incurred data but not for paid data. The fan charts widen abruptly at
development period 7, the direct footprint of the sigma spike.
""")

md("## Step 6: sensitivity analysis — which link ratios matter?")
code("""
sens = link_ratio_sensitivity(tri)
display(sens.head(10).style.format({"reserve": "{:,.0f}", "reserve_sd": "{:,.0f}", "reserve_cov": "{:.1%}",
                                    "reserve_diff": "{:,.0f}", "sd_diff": "{:,.0f}", "cov_diff": "{:.1%}"}))
plot_sensitivity_heatmap(sens, value="sd_diff")
plt.show()
""")
md("""
**Commentary.** Excluding each link ratio in turn and re-applying Mack analytically is a quick
way to find influential points. Dropping the 72->84 ratio of origin 2003 gives the largest
reduction in reserve, standard deviation and CoV by a wide margin; the next most influential
ratio matters far less.
""")

md("## Step 7: exclude the top three ratios and re-apply Mack analytically")
code("""
top3 = top_influential(sens, n=3, by="sd")
print("Excluding:", top3)
mack_top3 = mack_analytic_rmsep(tri, drop=top3)
display(mack_top3.to_frame().style.format({"reserve": "{:,.0f}", "sd": "{:,.0f}", "cov": "{:.1%}"}))
""")
md("""
**Commentary.** With the three most influential ratios excluded the reserve, its standard
deviation and its CoV all fall sharply. Whether any exclusion is justified is the analyst's call;
the point is to know which data points drive the volatility before making it.
""")

md("## Step 8: residuals and sigma after the exclusions")
code("""
boot_top3 = MackBootstrap(n_sims=10_000, random_seed=100, drop=top3).fit(tri)
print("Mack sigma after exclusions:", np.round(boot_top3.sigma_, 2))
plot_scaled_residuals(boot_top3.scaled_residuals_, by="dev", sigma=boot_top3.sigma_)
plt.show()
""")
md("""
**Commentary.** Excluding a ratio also excludes its residual, so sigma at development ratio 6
collapses and the standard deviation of the reserves falls with it.
""")

md("## Step 9: bootstrap Mack's model after the exclusions")
code("""
display(boot_top3.summary_statistics("reserves").style.format("{:,.0f}"))
for origin in (2004, 2008):
    plot_fan_chart(boot_top3, origin)
    plt.show()
""")
md("""
**Commentary.** The bootstrap again matches the analytic result with the same exclusions, far
fewer simulations produce negative reserves, and the fan charts no longer jump at period 7.
""")

md("## Step 10: keep all ratios but override sigma")
code("""
user_sigma = sigma_original.copy()
user_sigma[5] = 13.0  # tame the spike at the 72->84 ratio
boot_user = MackBootstrap(n_sims=10_000, random_seed=100, process_sigma=user_sigma).fit(tri)
display(boot_user.summary_statistics("reserves").style.format("{:,.0f}"))
plot_fan_chart(boot_user, 2004)
plt.show()
""")
md("""
**Commentary.** Overriding the variance parameter leaves the mean unchanged (up to simulation
error) while reducing the standard deviation and CoV. User-defined sigmas are an alternative to
exclusions for controlling volatility once its drivers are understood.
""")

md("## Step 11: the over-dispersed Poisson model with non-constant scale")
code("""
odp_all = CorrelatedBootstrapChainLadder(n_sims=10_000, rho=0.0, scale="nonconstant",
                                          parametric_dist="lognormal", random_seed=100).fit(tri)
sqrt_scale_original = np.sqrt(odp_all.sampler_.scale_by_dev_)
print("ODP sqrt(scale) by development period:", np.round(sqrt_scale_original, 1))
plot_scaled_residuals(odp_all.sampler_.standardized_residuals_ / np.where(sqrt_scale_original > 0, sqrt_scale_original, np.nan),
                      by="dev", sigma=sqrt_scale_original, title="ODP scaled residuals by development period")
plt.show()
display(odp_all.summary_statistics("reserves").style.format("{:,.0f}"))
plot_fan_chart(odp_all, 2004)
plt.show()
""")
md("""
**Commentary.** The ODP model is a model of incremental amounts, so the residual triangle is
10×10 and the spike now sits at development period 7 (the cell, not the ratio). With
non-constant scale the total standard deviation is close to Mack's, as expected, but because
future increments are drawn from a positive distribution the minimum reserves stay positive.
""")

md("## Step 12: ODP after excluding the influential ratios")
code("""
odp_top3 = CorrelatedBootstrapChainLadder(n_sims=10_000, rho=0.0, scale="nonconstant",
                                           parametric_dist="lognormal", random_seed=100, drop=top3).fit(tri)
print("ODP sqrt(scale) after exclusions:", np.round(np.sqrt(odp_top3.sampler_.scale_by_dev_), 1))
display(odp_top3.summary_statistics("reserves").style.format("{:,.0f}"))
plot_fan_chart(odp_top3, 2004)
plt.show()
""")
md("""
**Commentary.** The same ratios are influential under both models. Dropping the 72→84 ratio of
origin 2003 removes the development-period-7 residual from the scale estimate, and the ODP
standard deviation falls in line with Mack's with the same exclusions.
""")

md("## Step 13: ODP with a user-defined scale")
code("""
user_scale = odp_all.sampler_.scale_by_dev_.copy()
user_scale[6] = 40.0 ** 2
odp_user = CorrelatedBootstrapChainLadder(n_sims=10_000, rho=0.0, scale="nonconstant",
                                           parametric_dist="lognormal", random_seed=100,
                                           process_scale=user_scale).fit(tri)
display(odp_user.summary_statistics("reserves").style.format("{:,.0f}"))
""")
md("""
**Commentary.** As with Mack's model, a user-defined scale leaves the mean alone and reduces the
spread. Use it carefully, and only after the drivers of volatility are understood.
""")

md("## Step 14: compare and select")
code("""
rows = {
    "Mack, all ratios": boot_all, "Mack, top-3 excluded": boot_top3, "Mack, user sigma": boot_user,
    "ODP, all ratios": odp_all, "ODP, top-3 excluded": odp_top3, "ODP, user scale": odp_user,
}
compare = pd.DataFrame({
    name: {"mean": m.total_summary().total_reserve_mean, "sd": m.total_summary().total_reserve_stddev,
           "cov": m.total_summary().total_reserve_cv, "p99.5": m.total_summary().total_reserve_99_5th_percentile}
    for name, m in rows.items()
}).T
display(compare.style.format({"mean": "{:,.0f}", "sd": "{:,.0f}", "cov": "{:.1%}", "p99.5": "{:,.0f}"}))
""")
md("""
**Commentary.** Only now, knowing what drives the volatility and how each model behaves, is the
analyst placed to choose a model, decide on exclusions and decide whether to override the
variance parameters. Tail extrapolation is a common further step and is not covered here.
""")

md("## Step 15: scale to target ultimates")
code("""
selected = odp_top3
origins = list(selected.reserves_posterior_.origin.values)
paid = selected._paid_to_date().reindex(origins)
target_ultimates = paid + 1.1 * selected.ibnr_["mean"]   # illustrative target only
methods = {o: ("additive" if k < 5 else "multiplicative") for k, o in enumerate(origins)}
scaled = selected.scale_to_target(target_ultimates, method=methods)
display(selected.summary_statistics("reserves")[["mean", "std", "cov"]].style.format({"mean": "{:,.0f}", "std": "{:,.0f}", "cov": "{:.1%}"}))
display(scaled.summary_statistics("reserves")[["mean", "std", "cov"]].style.format({"mean": "{:,.0f}", "std": "{:,.0f}", "cov": "{:.1%}"}))
""")
md("""
**Commentary.** Reserving teams rarely book the mean of a simple chain-ladder bootstrap, so the
distribution is usually shifted to a target. Additive scaling keeps the absolute standard
deviation and changes the CoV; multiplicative scaling keeps the CoV and changes the standard
deviation. The method can be chosen per origin. If a lot of scaling is needed, stop and
investigate rather than scale.

### Common errors and fallacies (paraphrased from the source)

* There is no such thing as "the" bootstrap model. Bootstrapping is a procedure applied to a
  specified statistical model; say which model, and prefer non-constant scale for ODP.
* Mack's model tolerates negative increments and factors below one and suits incurred data; the
  ODP model needs factors above one and gives positive simulated increments.
* An incurred bootstrap gives a distribution of IBNR+IBNER and of ultimates. To compare CoVs
  with a paid analysis, subtract the latest paid from each simulated ultimate
  (`incurred_to_paid`).
* Parametric bootstrapping simulates pseudo-data directly from a parametric distribution given
  its mean and variance. Simulating residuals from a Normal and inverting them is not parametric
  bootstrapping.
""")

md("## Epilogue: the one-year view")
code("""
cdr = claims_development_result(odp_top3)
one_year = cdr.summary().query("future_period == 1").set_index("origin")
lifetime_sd = odp_top3.summary_statistics("reserves")["std"]
one_year["lifetime_sd"] = lifetime_sd.reindex(one_year.index).values
one_year["cdr_sd_ratio"] = one_year["sd"] / one_year["lifetime_sd"]
display(one_year[["sd", "lifetime_sd", "cdr_sd_ratio", "var"]].style.format({"sd": "{:,.0f}", "lifetime_sd": "{:,.0f}", "cdr_sd_ratio": "{:.0%}", "var": "{:,.0f}"}))
""")
md(f"""
The one-year Claims Development Result re-reserves each simulated next diagonal (the
"actuary in the box") and is the basis for Solvency II reserve risk. See `claims_development_result`,
`bayesianchainladder.riskmeasures` and the England, Verrall & Wüthrich (2019) example in
[{SOURCE}]({SOURCE}) for cost-of-capital risk margins and IFRS 17 risk adjustments.
""")


def build(out_dir: pathlib.Path) -> pathlib.Path:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src)
        for kind, src in CELLS
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "modus_operandi.ipynb"
    nbf.write(nb, path)
    return path


if __name__ == "__main__":
    target = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).parent
    print(build(target))
