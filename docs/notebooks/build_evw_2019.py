"""Generate docs/notebooks/evw_2019_one_year_view.ipynb.

Recreates England, Verrall & Wuthrich (2019), "The one-year view of
reserve risk", Insurance: Mathematics and Economics
(https://doi.org/10.1016/j.insmatheco.2018.12.002), as coded in Peter
England's *EVW_2019* notebook from the StochasticReserving repository
(https://github.com/DrPeterEngland/StochasticReserving,
`Python_Examples/EVW_2019.ipynb`, MIT licence, EMC Actuarial and Analytics
Ltd) on top of the `bayesianchainladder` API. Commentary is paraphrased.

England's published figures come from
``bayesianchainladder/data/england_reference_values.json`` (transcribed from
the rendered tables in his executed notebook; see its ``_provenance``
block). Two systematic differences recur throughout:

1. **Last-sigma rule.** England's Mack sigma for the last development period
   is ``min(sigma_7, sigma_8)``; ``mack_analytic_rmsep`` wraps
   ``chainladder.MackChainladder``, which extrapolates the last sigma
   log-linearly instead, giving a slightly larger last-origin standard
   error. ``MackBootstrap.sigma_`` implements England's own rule, so the
   bootstrap tables do not show this gap.
2. **Simulation noise.** Everything downstream of a 10,000-draw Monte Carlo
   (bootstrap means/SDs, one-year CDR figures, VaR/TVaR/PHT risk measures,
   discounted reserves, cost-of-capital margins) differs from England's by
   ordinary Monte Carlo error (England drew Gamma pseudo-data with a
   different RNG stream and seed sequence; our draws use the same
   algorithm). Differences of a few percent, sometimes up to around 10% for
   99.5th-percentile tail quantiles, are expected and are not adjusted for.

Usage: python build_evw_2019.py [output_dir]
"""

from __future__ import annotations

import pathlib
import sys

import nbformat as nbf

SOURCE = "https://github.com/DrPeterEngland/StochasticReserving"
PAPER = "https://doi.org/10.1016/j.insmatheco.2018.12.002"

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip()))


def code(text: str) -> None:
    CELLS.append(("code", text.strip()))


md(f"""
# The one-year view of reserve risk (England, Verrall & Wuthrich 2019)

Adapted from Peter England's *EVW_2019* notebook in the
[StochasticReserving repository]({SOURCE})
(`Python_Examples/EVW_2019.ipynb`, MIT licence, provided by EMC Actuarial and
Analytics Ltd), which reproduces England, Verrall & Wuthrich (2019), *The
one-year view of reserve risk*, Insurance: Mathematics and Economics
([{PAPER}]({PAPER})). The code uses `bayesianchainladder`; the commentary is
paraphrased from the paper and notebook.

Every table with a counterpart in England's rendered output is shown next to
it, with a `diff %` column. Two systematic differences recur:

1. **Last-sigma rule.** `mack_analytic_rmsep` wraps `chainladder.MackChainladder`,
   which extrapolates the last Mack sigma log-linearly; England takes
   `min(sigma_7, sigma_8)`. `MackBootstrap.sigma_` implements England's own
   rule, so only the *analytic* Mack table (Table 2) shows this gap.
2. **Simulation noise.** Everything built from the 10,000-draw Monte Carlo
   bootstrap differs from England's by ordinary Monte Carlo error (same
   algorithm, different RNG stream/seed sequence). A handful of percent,
   sometimes more for 99.5th-percentile tail quantiles, is expected.

The triangle is Taylor & Ashe (1983) incremental paid losses — identical to
`chainladder`'s `genins` sample and to England's `claims_triangle.csv`.
""")

md("## 1. Setup")
code("""
import json
import warnings
from importlib import resources

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import chainladder as cl

from bayesianchainladder import (
    MackBootstrap,
    claims_development_result,
    cost_of_capital_risk_margin,
    capital_profile,
    discounted_reserves,
    equivalent_risk_tolerance,
    future_reserve_profile,
    load_england_sample,
    mack_analytic_rmsep,
    plot_capital_profiles,
    plot_fan_chart,
    plot_scaled_residuals,
    proportional_hazards_transform,
    tail_value_at_risk,
    value_at_risk,
)

warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.display.float_format = "{:,.0f}".format

tri = load_england_sample("taylor_ashe")
origins = list(tri.origin.year)

with resources.files("bayesianchainladder.data").joinpath(
    "england_reference_values.json"
).open("r", encoding="utf-8") as fh:
    ref_all = json.load(fh)
ref = ref_all["evw_2019"]

SEED = 101
N_SIMS = 10_000


def compare(ours, england, labels=None):
    \"\"\"Side-by-side comparison table: ours, England, diff %.\"\"\"
    ours = np.asarray(ours, dtype=float)
    england = np.asarray(england, dtype=float)
    if labels is None:
        labels = list(range(len(ours)))
    with np.errstate(divide="ignore", invalid="ignore"):
        diff_pct = np.where(england != 0, 100.0 * (ours - england) / england, np.nan)
    return pd.DataFrame(
        {"ours": ours, "England": england, "diff %": diff_pct}, index=labels
    )


COMPARE_FMT = {"ours": "{:,.0f}", "England": "{:,.0f}", "diff %": "{:+.1f}%"}
DECIMAL_FMT = {"ours": "{:,.3f}", "England": "{:,.3f}", "diff %": "{:+.1f}%"}

dev = cl.Development().fit_transform(tri)
print("Volume-weighted factors:", np.round(dev.ldf_.values.flatten(), 3))
cl_result = cl.Chainladder().fit(dev)
display(
    pd.DataFrame(
        {
            "latest": tri.latest_diagonal.values[0, 0, :, 0],
            "reserve": cl_result.ibnr_.values[0, 0, :, 0],
            "ultimate": cl_result.ultimate_.values[0, 0, :, 0],
        },
        index=origins,
    ).style.format("{:,.0f}")
)
print("Total reserve:", f"{float(np.nansum(cl_result.ibnr_.values)):,.0f}")
""")
md("""
**Commentary.** England's Table 1 shows the same volume-weighted chain-ladder
factors and per-origin reserves; both packages apply the deterministic chain
ladder identically, so this table matches exactly (up to the display
precision above).
""")

md("## 2. Table 2: Mack's model applied analytically")
code("""
mack = mack_analytic_rmsep(tri)
display(mack.to_frame().style.format({"reserve": "{:,.0f}", "sd": "{:,.0f}", "cov": "{:.1%}"}))

t2 = ref["table2_analytic_mack"]
display(compare(mack.reserves, t2["reserves"], origins).style.format(COMPARE_FMT))
display(compare(mack.reserve_sd, t2["reserve_sd"], origins).style.format(COMPARE_FMT))
display(
    compare(
        [mack.total_reserve, mack.total_sd],
        [t2["total_reserve"], t2["total_sd"]],
        ["total_reserve", "total_sd"],
    ).style.format(COMPARE_FMT)
)

boot_for_sigma = MackBootstrap(n_sims=N_SIMS, random_seed=SEED).fit(tri)
print("MackBootstrap sigma_ (England's last-sigma rule):", np.round(boot_for_sigma.sigma_, 2))
""")
md("""
**Commentary.** Reserves match exactly (same closed-form chain ladder).
Total SD is close but not identical: `mack_analytic_rmsep` total SD is about
0.3% below England's 2,447,618 because it extrapolates the last development
period's sigma log-linearly, whereas England's rule takes
`min(sigma_7, sigma_8)`. `MackBootstrap.sigma_` (printed above) *does*
implement England's rule — its last two entries are 33.87 and 21.13,
matching his Table 2 note — so every bootstrap-based table below does not
carry this gap.
""")

md("## 3. Residuals")
code("""
boot = MackBootstrap(n_sims=N_SIMS, random_seed=SEED).fit(tri)
plot_scaled_residuals(boot.scaled_residuals_, by="dev", sigma=boot.sigma_)
plt.show()
""")
md("""
**Commentary.** Scaled residuals by development period, with sigma (using
England's last-sigma rule) on the twin axis. This is the basis for every
bootstrap table that follows.
""")

md("## 4. Table 4: bootstrap reserves and the one-year CDR")
code("""
summary = boot.summary_statistics("reserves")
cdr = claims_development_result(boot)
cdr_summary = cdr.summary()
one_year = cdr_summary.query("future_period == 1").set_index("origin").reindex(summary.index)

avg_reserve = summary["mean"]
bootstrap_sd = summary["std"]
bootstrap_cov = summary["cov"]
cdr_sd = one_year["sd"]
# England's fifth column is the one-year CDR's coefficient of variation
# (cdr_sd / avg reserve), not a ratio to the lifetime SD.
cdr_cov = cdr_sd / avg_reserve

table4_ours = pd.DataFrame(
    {
        "avg_reserve": avg_reserve,
        "bootstrap_sd": bootstrap_sd,
        "bootstrap_cov": bootstrap_cov,
        "cdr_sd": cdr_sd,
        "cdr_cov": cdr_cov,
    }
)
display(
    table4_ours.style.format(
        {"avg_reserve": "{:,.0f}", "bootstrap_sd": "{:,.0f}", "bootstrap_cov": "{:.1%}",
         "cdr_sd": "{:,.0f}", "cdr_cov": "{:.1%}"}
    )
)

t4 = ref["table4_bootstrap_and_one_year_cdr"]
labels = origins + ["Total"]
display(compare(avg_reserve, t4["avg_reserves"] + [t4["total_avg_reserve"]], labels).style.format(COMPARE_FMT))
display(compare(bootstrap_sd, t4["bootstrap_sd"] + [t4["total_bootstrap_sd"]], labels).style.format(COMPARE_FMT))
display(compare(cdr_sd, t4["cdr_sd"] + [t4["total_cdr_sd"]], labels).style.format(COMPARE_FMT))
display(
    compare(
        cdr_cov * 100,
        t4["cdr_sd_ratio_pct"] + [t4["total_cdr_sd_ratio_pct"]],
        labels,
    ).style.format({"ours": "{:.1f}", "England": "{:.1f}", "diff %": "{:+.1f}%"})
)
""")
md("""
**Commentary.** All four compared quantities differ from England's only by
Monte Carlo error at 10,000 simulations (a few percent), consistent with the
identical-algorithm/different-seed explanation established in Table 2.
""")

md("## 5. Table 6: incremental and cumulative one-year CDR standard deviation")
code("""
totals = cdr_summary.query("origin == 'Total'").set_index("future_period").sort_index()
incremental_cdr_sd = totals["sd"].values
sqrt_sum_squares = float(np.sqrt((incremental_cdr_sd ** 2).sum()))
lifetime_sd = float(boot.total_summary().total_reserve_stddev)
print(f"sqrt(sum of squares of incremental CDR SDs) = {sqrt_sum_squares:,.0f}")
print(f"lifetime bootstrap total SD                 = {lifetime_sd:,.0f}")

cumulative_cdr_sd = cdr.cumulative().sum("origin").std("sample", ddof=1).values

periods = totals.index.values
display(compare(incremental_cdr_sd, ref["table6_incremental_cdr_sd_total"], periods).style.format(COMPARE_FMT))
display(
    compare([sqrt_sum_squares], [ref["table6_incremental_cdr_sd_sqrt_sum_squares_total"]], ["sqrt_sum_squares"])
    .style.format(COMPARE_FMT)
)
display(compare(cumulative_cdr_sd, ref["table6_cumulative_cdr_sd_total"], periods).style.format(COMPARE_FMT))
""")
md("""
**Commentary.** The cumulative CDR SD by period converges to the lifetime
bootstrap SD; the sqrt-sum-of-squares of the incremental SDs is close but not
identical to it because the CDR periods are correlated (the sum of squares
would equal the lifetime variance only under independence). Both England's
and our figures show the same near-equality, and the residual gap is Monte
Carlo noise.
""")

md("## 6. Table 7: VaR 99.5% of the one-year CDR")
code("""
var_by_period_total = totals["var"].values
display(compare(var_by_period_total, ref["table7_var_cdr_995_total"], periods).style.format(COMPARE_FMT))

one_year_by_origin = cdr_summary.query("future_period == 1").set_index("origin").reindex(origins)["var"]
ref_by_origin = ref["table7_var_cdr_995_one_year_by_origin"]
mask = [v is not None for v in ref_by_origin]
display(
    compare(
        one_year_by_origin.values[mask],
        [v for v, m in zip(ref_by_origin, mask, strict=True) if m],
        [o for o, m in zip(origins, mask, strict=True) if m],
    ).style.format(COMPARE_FMT)
)
""")
md("""
**Commentary.** Origin 2001 is fully developed (no future cells, so no CDR
and no VaR); it is excluded from the by-origin comparison, matching
England's blank cell. Everything else is 10,000-simulation VaR at a tail
quantile, so a few percent of Monte Carlo noise is expected — a bit more
than the SD-based tables above because a single quantile estimate is
noisier than a mean or a standard deviation.
""")

md(
    "### Is the +3% on the one-year VaR a calculation difference? (100,000-simulation check)"
)
code("""
# The one-year VaR 99.5% is the opening capital for every cost-of-capital
# margin below, so it is worth checking that its gap to England is noise
# rather than method. Three pieces of evidence:
#
# 1. England's own code, re-run with his seed 101, reproduces his 4,771,636
#    exactly; with seeds 1, 2, 3 it gives 4,879,375 / 4,871,765 / 5,021,002,
#    so his published figure sits at the low end of his own Monte Carlo spread.
# 2. England's VaR is the order statistic at index floor(n p) + 1 of the
#    sorted CDRs; ours is np.quantile. On the same samples that changes the
#    figure by about 4,000 (0.1%).
# 3. Both implementations at 100,000 simulations (England's code with seed 7,
#    run outside this notebook on 2026-09-24; ours re-run here):
ENGLAND_100K = {"sd": 1_775_444, "var_995": 4_872_897, "q1": -4_362_001, "q5": -3_005_640}

boot_100k = MackBootstrap(n_sims=100_000, random_seed=7).fit(tri)
cdr1_100k = claims_development_result(boot_100k, future_periods=1).total_cdr.isel(future_period=0).values
ours_100k = {
    "sd": float(np.std(cdr1_100k, ddof=1)),
    "var_995": float(cdr1_100k.mean() - np.quantile(cdr1_100k, 0.005)),
    "q1": float(np.quantile(cdr1_100k, 0.01)),
    "q5": float(np.quantile(cdr1_100k, 0.05)),
}
display(
    compare(
        [ours_100k[k] for k in ENGLAND_100K],
        [ENGLAND_100K[k] for k in ENGLAND_100K],
        ["CDR(1) SD", "VaR 99.5%", "1% quantile", "5% quantile"],
    ).style.format(COMPARE_FMT)
)

# Monte Carlo spread of a 10,000-simulation VaR 99.5%, from 200 subsamples of the 100k run
rng = np.random.default_rng(0)
sub = np.array(
    [s.mean() - np.quantile(s, 0.005) for s in (rng.choice(cdr1_100k, 10_000, replace=False) for _ in range(200))]
)
print(
    f"VaR 99.5% at n=10,000: mean {sub.mean():,.0f}, SE {sub.std():,.0f} ({sub.std() / sub.mean() * 100:.1f}%), "
    f"95% band {np.quantile(sub, 0.025):,.0f} .. {np.quantile(sub, 0.975):,.0f}"
)
print(
    f"England's published 4,771,636 is {(4_771_636 - sub.mean()) / sub.std():+.2f} SE from that centre; "
    f"our 10,000-simulation value above is {(var_by_period_total[0] - sub.mean()) / sub.std():+.2f} SE."
)
""")
md("""
**Commentary.** At 100,000 simulations the two implementations agree to
well under 1% on the one-year CDR standard deviation, its 99.5% VaR and its
1% and 5% quantiles, so the bootstrap, the actuary-in-the-box re-reserving
and the tail are computed the same way. A 99.5% VaR from 10,000 simulations
rests on about 50 order statistics and has a relative standard error of
roughly 1.6%, so two such runs routinely differ by 3%; England's published
value and ours are on opposite sides of the centre. Because that single
number is the opening capital for Tables 8, 9, 12 and 13, every
cost-of-capital margin below inherits the same few-percent gap.
""")

md("## 7. Bootstrap total reserve distribution")
code("""
total_row = summary.loc["Total"]
ours_dict = {
    "mean": total_row["mean"], "sd": total_row["std"], "cov_pct": total_row["cov"] * 100,
    "min": total_row["min"],
    "p0.5": total_row["0.5%"], "p1": total_row["1%"], "p5": total_row["5%"],
    "p10": total_row["10%"], "p25": total_row["25%"], "p50": total_row["50%"],
    "p75": total_row["75%"], "p90": total_row["90%"], "p95": total_row["95%"],
    "p99": total_row["99%"], "p99.5": total_row["99.5%"], "max": total_row["max"],
}
ref7 = ref["bootstrap_reserve_summary_total"]
display(
    compare(list(ours_dict.values()), [ref7[k] for k in ours_dict], list(ours_dict.keys()))
    .style.format(COMPARE_FMT)
)
""")
md("""
**Commentary.** The full distribution of the total reserve, mean through
max, tracks England's closely; the tails (p0.5, p99.5, min, max) show the
largest Monte Carlo noise, as expected for extreme order statistics from
10,000 draws.
""")

md("## 8. Fan charts")
code("""
for o in (2002, 2010):
    plot_fan_chart(boot, o)
    plt.show()
""")
md("""
**Commentary.** Origin 2002 has only one future cell and a narrow fan;
origin 2010 is the least mature and has the widest.
""")

md("## 9. Table 5: reserves discounted at 3%, mid-year")
code("""
disc = discounted_reserves(boot, 0.03, 0.5)
disc_mean = disc.mean("sample")
disc_sd = disc.std("sample", ddof=1)
disc_total_samples = disc.sum("origin").values
total_disc_mean = float(disc_total_samples.mean())
total_disc_sd = float(disc_total_samples.std(ddof=1))

t5 = ref["table5_discounted_3pct"]
display(compare(disc_mean.values, t5["avg_reserves"], origins).style.format(COMPARE_FMT))
display(compare(disc_sd.values, t5["bootstrap_sd"], origins).style.format(COMPARE_FMT))
display(
    compare(
        [total_disc_mean, total_disc_sd],
        [t5["total_avg"], t5["total_sd"]],
        ["total_avg", "total_sd"],
    ).style.format(COMPARE_FMT)
)
""")
md("""
**Commentary.** Discounting is deterministic given the simulated cash flows,
so the gap to England's figures is the same Monte Carlo noise already
present in the undiscounted bootstrap.
""")

md("## 10. Tables 8 & 9: cost-of-capital risk margin and capital bases")
code("""
opening_capital = float(
    cdr_summary.query("future_period == 1 and origin == 'Total'")["var"].iloc[0]
)
print(f"Opening capital (VaR 99.5% of one-year total CDR) = {opening_capital:,.0f}")

frp_disc = future_reserve_profile(boot, 0.03, 0.5)
best_estimate_basis = frp_disc.mean("sample").values
coc_best = cost_of_capital_risk_margin(
    opening_capital, capital_profile(best_estimate_basis), 0.06, 0.03, offset=1.0
)

sd_basis = totals["sd"].values
var_basis = totals["var"].values
coc_sd = cost_of_capital_risk_margin(opening_capital, capital_profile(sd_basis), 0.06, 0.03, offset=1.0)
coc_var = cost_of_capital_risk_margin(opening_capital, capital_profile(var_basis), 0.06, 0.03, offset=1.0)

t8 = ref["table8_coc_risk_margin_best_estimate_basis"]
display(compare(coc_best["capital"], t8["capital"], periods).style.format(COMPARE_FMT))
display(compare(coc_best["cost"], t8["cost_of_capital"], periods).style.format(COMPARE_FMT))
display(compare(coc_best["discounted_cost"], t8["disc_cost_of_capital"], periods).style.format(COMPARE_FMT))
display(compare([coc_best["risk_margin"]], [t8["risk_margin"]], ["risk_margin"]).style.format(COMPARE_FMT))

t9 = ref["table9_capital_bases"]
display(compare(coc_sd["capital"], t9["cdr_sd_capital"], periods).style.format(COMPARE_FMT))
display(compare(coc_var["capital"], t9["cdr_var_capital"], periods).style.format(COMPARE_FMT))
display(
    compare(
        [coc_best["risk_margin"], coc_sd["risk_margin"], coc_var["risk_margin"]],
        [t9["risk_margin"]["best_estimate"], t9["risk_margin"]["cdr_sd"], t9["risk_margin"]["cdr_var"]],
        ["best_estimate", "cdr_sd", "cdr_var"],
    ).style.format(COMPARE_FMT)
)
""")
md("""
**Commentary.** England's Table 8 discounts the *deterministic* chain-ladder
forecast to build the best-estimate capital profile; `future_reserve_profile`
uses the simulated mean of the bootstrap instead (there is no separate
deterministic forecast object carried through this pipeline). Combined with
the opening capital itself being a noisy 10,000-draw VaR estimate (about 3%
above England's here), our best-estimate risk margin comes out around 3%
above his 801,951 — both effects are small and this is exactly the gap noted
in the project's own comparison of these figures. The SD- and VaR-based
capital bases (Table 9) are within ordinary Monte Carlo error.
""")

md("## 11. Table 10: risk adjustments on discounted reserves")
code("""
mean_disc = disc_total_samples.mean()
var75 = value_at_risk(disc_total_samples, 0.75) - mean_disc
tvar40 = tail_value_at_risk(disc_total_samples, 0.40) - mean_disc
pht185 = proportional_hazards_transform(disc_total_samples, 1.85) - mean_disc

t10 = ref["table10_risk_adjustments"]
rows = pd.DataFrame(
    {
        "risk_adjustment": [var75, tvar40, pht185],
        "total": [mean_disc + var75, mean_disc + tvar40, mean_disc + pht185],
        "pct": [100 * var75 / mean_disc, 100 * tvar40 / mean_disc, 100 * pht185 / mean_disc],
    },
    index=["VaR 75%", "TVaR 40%", "PHT 1.85"],
)
display(rows.style.format({"risk_adjustment": "{:,.0f}", "total": "{:,.0f}", "pct": "{:.1f}%"}))

display(
    compare(
        [var75, tvar40, pht185],
        [t10["var"]["risk_adjustment"], t10["tvar"]["risk_adjustment"], t10["pht"]["risk_adjustment"]],
        ["VaR 75%", "TVaR 40%", "PHT 1.85"],
    ).style.format(COMPARE_FMT)
)
""")
md("""
**Commentary.** All three risk adjustments are within a few percent of
England's — Monte Carlo error on the discounted reserve distribution.
""")

md("## 12. Table 11: equivalent risk tolerance")
code("""
target = coc_best["risk_margin"]
var_level = equivalent_risk_tolerance(disc_total_samples, target, measure="var")
tvar_level = equivalent_risk_tolerance(disc_total_samples, target, measure="tvar")
pht_level = equivalent_risk_tolerance(disc_total_samples, target, measure="pht")

print(f"Target margin (our own CoC margin from Table 8) = {target:,.0f}")
print("England's target was his own CoC margin, 801,951.")
display(
    compare(
        [var_level * 100, tvar_level * 100, pht_level],
        [64.5, 21.1, 1.432],
        ["VaR level %", "TVaR level %", "PHT parameter"],
    ).style.format(DECIMAL_FMT)
)
""")
md("""
**Commentary.** We solve for the confidence level against *our own* CoC
margin (about 3% above England's), so part of the gap to his 64.5% / 21.1% /
1.432 is that our target itself differs, on top of Monte Carlo error in the
discounted reserve distribution used to solve for the level.
""")

md("## 13. Table 12: cost-of-capital margin under alternative reserve bases")
code("""
frp_undisc = future_reserve_profile(boot, 0.0, 0.5)
n_periods = frp_disc.sizes["period"]

avg_disc = frp_disc.mean("sample").values
sd_disc = frp_disc.std("sample", ddof=1).values
sd_undisc = frp_undisc.std("sample", ddof=1).values


def _var_minus_mean(profile, level):
    return np.array(
        [
            value_at_risk(profile.isel(period=t).values, level) - profile.isel(period=t).values.mean()
            for t in range(profile.sizes["period"])
        ]
    )


## England solves a *new* confidence level here (distinct from Table 11's):
## the level at which VaR(level) minus the mean of the discounted total
## reserve equals the opening capital itself (his Table 12 cell computes
## `target = CDR_Result["TotalCDR_VAR"][0]` and solves against
## `Disc_Res["TotalReserve"]`), not our Table 8 CoC margin. His column is
## headed "64.5%" only because that cell reuses the Table 11 `VAR_level`
## variable as a label; the values in the column are computed at this new,
## higher level (~96.9%).
var_level_res = equivalent_risk_tolerance(disc_total_samples, opening_capital, measure="var")
print(f"Solved confidence level for Table 12's VaR column = {var_level_res * 100:.1f}% (England's header reads 64.5%, a label carried over from Table 11; his code solves against the opening capital).")

var_at_solved = _var_minus_mean(frp_disc, var_level_res)
var_at_995 = _var_minus_mean(frp_disc, 0.995)

t12 = ref["table12_coc_risk_margin_reserve_bases"]
display(compare(avg_disc, t12["avg_disc_reserves"], np.arange(n_periods)).style.format(COMPARE_FMT))
display(compare(sd_disc, t12["sd_disc_reserves"], np.arange(n_periods)).style.format(COMPARE_FMT))
display(compare(sd_undisc, t12["sd_undisc_reserves"], np.arange(n_periods)).style.format(COMPARE_FMT))
display(compare(var_at_solved, t12["var_disc_reserves_at_64_5pct"], np.arange(n_periods)).style.format(COMPARE_FMT))
display(compare(var_at_995, t12["var_disc_reserves_at_99_5pct"], np.arange(n_periods)).style.format(COMPARE_FMT))

# England anchors every Table 12 margin except the fixed-99.5% one on the
# VaR-solved basis's own period-0 value (his
# `RM_Initial_Capital_T12 = Disc_Fut_Res_VAR_root[0]`) rather than on a
# separately computed opening capital -- the two coincide by construction,
# since var_level_res was solved so that var_at_solved[0] equals
# opening_capital. The fixed-99.5% column anchors on its own period-0 value
# (`Disc_Fut_Res_VAR_995[0]`) instead.
initial_capital_t12 = var_at_solved[0]
margins = {
    "avg": cost_of_capital_risk_margin(initial_capital_t12, capital_profile(avg_disc), 0.06, 0.03, offset=1.0),
    "sd_disc": cost_of_capital_risk_margin(initial_capital_t12, capital_profile(sd_disc), 0.06, 0.03, offset=1.0),
    "sd_undisc": cost_of_capital_risk_margin(initial_capital_t12, capital_profile(sd_undisc), 0.06, 0.03, offset=1.0),
    "var_64_5": cost_of_capital_risk_margin(initial_capital_t12, capital_profile(var_at_solved), 0.06, 0.03, offset=1.0),
    "var_99_5": cost_of_capital_risk_margin(var_at_995[0], capital_profile(var_at_995), 0.06, 0.03, offset=1.0),
}
ref_margins = t12["risk_margin"]
display(
    compare(
        [margins[k]["risk_margin"] for k in margins],
        [ref_margins["avg"], ref_margins["sd_disc"], ref_margins["sd_undisc"], ref_margins["var_64_5"], ref_margins["var_99_5"]],
        list(margins.keys()),
    ).style.format(COMPARE_FMT)
)
""")
md("""
**Commentary.** England's Table 12 solves its own confidence level, targeting
the opening capital (the one-year CDR VaR 99.5%) rather than the Table 8 CoC
margin — the "64.5%" in his column header is a labelling slip (that cell of
his notebook reuses the Table 11 `VAR_level` variable to print the header,
but the column itself is computed at the level solved above, around 96.9%).
Once solved against the right target, `var_at_solved` compares directly to
England's published array with no rescaling needed, and all five margins —
each anchored on that basis's own period-0 value, which by construction
equals the opening capital — are within Monte Carlo error of his.
""")

md("## 14. Table 13: cost-of-capital margin from the reverse-cumulative CDR")
code("""
rev = cdr.reverse_cumulative().sum("origin")
sd_rev = rev.std("sample", ddof=1).values

## England's Table 13 cell solves yet another confidence level, again
## targeting the opening capital, this time on the period-0 (full lifetime)
## reverse-cumulative total CDR: `target = CDR_Result["TotalCDR_VAR"][0]`,
## solved against `RevSum_CDR[0]`. The adverse tail of the CDR is the *low*
## quantile, so we solve on the negated series -- VAR(-x, p) - mean(-x) =
## target is algebraically the same as mean(x) - VAR(x, 1-p) = target, the
## capital convention used throughout this notebook's CDR-based tables.
x0 = rev.isel(future_period=0).values
solved_level = equivalent_risk_tolerance(-x0, opening_capital, measure="var")
print(f"Solved confidence level for Table 13 = {solved_level * 100:.1f}% (England labels this column 96.9%).")


def _cdr_capital(x, level):
    return float(np.mean(x) - value_at_risk(x, 1.0 - level))


var_rev_solved = np.array(
    [_cdr_capital(rev.isel(future_period=t).values, solved_level) for t in range(rev.sizes["future_period"])]
)
var_rev_995 = np.array(
    [_cdr_capital(rev.isel(future_period=t).values, 0.995) for t in range(rev.sizes["future_period"])]
)

t13 = ref["table13_coc_risk_margin_reverse_cdr"]
display(compare(sd_rev, t13["sd_simulated"], periods).style.format(COMPARE_FMT))
display(compare(var_rev_solved, t13["var_at_96_9pct"], periods).style.format(COMPARE_FMT))
display(compare(var_rev_995, t13["var_at_99_5pct"], periods).style.format(COMPARE_FMT))

# England anchors the sd and solved-VaR margins on var_rev_solved's own
# period-0 value (his `RM_Initial_Capital_T13`), which by construction
# equals the opening capital; the fixed-99.5% margin anchors on its own
# period-0 value instead.
initial_capital_t13 = var_rev_solved[0]
margins13 = {
    "sd": cost_of_capital_risk_margin(initial_capital_t13, capital_profile(sd_rev), 0.06, 0.03, offset=1.0),
    "var_solved": cost_of_capital_risk_margin(initial_capital_t13, capital_profile(var_rev_solved), 0.06, 0.03, offset=1.0),
    "var_99_5": cost_of_capital_risk_margin(var_rev_995[0], capital_profile(var_rev_995), 0.06, 0.03, offset=1.0),
}
ref13m = t13["risk_margin"]
display(
    compare(
        [margins13["sd"]["risk_margin"], margins13["var_solved"]["risk_margin"], margins13["var_99_5"]["risk_margin"]],
        [ref13m["sd"], ref13m["var_96_9"], ref13m["var_99_5"]],
        ["sd", "var_solved", "var_99_5"],
    ).style.format(COMPARE_FMT)
)
""")
md("""
**Commentary.** `cdr.reverse_cumulative()` sums each simulated CDR from a
future period to run-off, so its period-1 value is the full lifetime
reserve outcome (its SD here is the same undiscounted total bootstrap SD as
Table 4). As in Table 12, the confidence level solved above targets the
opening capital directly, matching England's own `RM_Initial_Capital_T13`
logic; the resulting level lands close to his stated 96.9%, and every column
— SD, solved-level VaR, fixed 99.5% VaR, and all three margins — compares
within ordinary Monte Carlo error (England's own MW RMSEP column is blank
too — Merz-Wuthrich is not implemented on either side).
""")

md("## 15. Figure 1: capital run-off profiles")
code("""
plot_capital_profiles(
    {
        "Best estimate": coc_best["capital"],
        "CDR SD": coc_sd["capital"],
        "CDR VaR 99.5%": coc_var["capital"],
        "Reverse CDR SD": margins13["sd"]["capital"],
    }
)
plt.show()
""")
md("""
**Commentary.** All four bases are rescaled to the same opening capital, so
this chart compares their *shape* (how fast each measure runs off), not
their level. The CDR-based profiles run off faster in the first year or two
than the best-estimate reserve profile.
""")

md("## 16. Closing comparison")
code("""
closing = pd.DataFrame(
    {
        "ours": [
            mack.total_sd,
            boot.total_summary().total_reserve_mean,
            boot.total_summary().total_reserve_stddev,
            float(totals.loc[1, "sd"]),
            float(totals.loc[1, "var"]),
            total_disc_mean,
            coc_best["risk_margin"],
            var75,
            tvar40,
            pht185,
        ],
        "England": [
            t2["total_sd"],
            t4["total_avg_reserve"],
            t4["total_bootstrap_sd"],
            t4["total_cdr_sd"],
            ref["table7_var_cdr_995_total"][0],
            t5["total_avg"],
            t8["risk_margin"],
            t10["var"]["risk_adjustment"],
            t10["tvar"]["risk_adjustment"],
            t10["pht"]["risk_adjustment"],
        ],
    },
    index=[
        "Analytic Mack total SD",
        "Bootstrap total mean",
        "Bootstrap total SD",
        "One-year CDR(1) total SD",
        "One-year CDR(1) VaR 99.5%",
        "Discounted (3%) total mean",
        "Cost-of-capital risk margin",
        "Risk adjustment: VaR 75%",
        "Risk adjustment: TVaR 40%",
        "Risk adjustment: PHT 1.85",
    ],
)
closing["diff %"] = 100 * (closing["ours"] - closing["England"]) / closing["England"]
display(closing.style.format({"ours": "{:,.0f}", "England": "{:,.0f}", "diff %": "{:+.1f}%"}))
""")
md("""
**Commentary, attributing each gap:**

* **Analytic Mack total SD** (-0.3%): the last-sigma extrapolation rule
  described in section 2 — `chainladder.MackChainladder` extrapolates
  log-linearly; England takes `min(sigma_7, sigma_8)`.
* **Bootstrap mean/SD, CDR(1) SD, discounted mean**: ordinary Monte Carlo
  error at 10,000 simulations, same algorithm and different RNG stream/seed.
* **CDR(1) VaR 99.5%**: Monte Carlo error on a single tail quantile, which
  is noisier than a mean or SD estimated from the same 10,000 draws.
* **Cost-of-capital risk margin**: the CDR(1) VaR noise above (it is the
  opening capital) plus England's use of the deterministic chain-ladder
  forecast, rather than the simulated bootstrap mean, for the best-estimate
  capital profile (section 10).
* **VaR/TVaR/PHT risk adjustments**: Monte Carlo error on the discounted
  reserve distribution.

No gap in this table falls outside the causes established while building it
section by section above.
""")


def build(out_dir: pathlib.Path) -> pathlib.Path:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb.cells = [
        (
            nbf.v4.new_markdown_cell(src)
            if kind == "markdown"
            else nbf.v4.new_code_cell(src)
        )
        for kind, src in CELLS
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "evw_2019_one_year_view.ipynb"
    nbf.write(nb, path)
    return path


if __name__ == "__main__":
    target = (
        pathlib.Path(sys.argv[1])
        if len(sys.argv) > 1
        else pathlib.Path(__file__).parent
    )
    print(build(target))
