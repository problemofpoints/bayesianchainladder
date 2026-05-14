# Standalone Stochastic Reserving Back-test Results

**Date generated**: see git log.

## Setup

- **Data**: 200 Meyers (CAS Monograph 1) triangles — 50 each for
  comauto, ppauto, wkcomp, othliab
- **Methods**: Mack, ODP, Corr-ODP (rho=0.1), ODP+BF (apriori=0.65), ODP+CC
- **Loss types**: paid and case_incurred
- **Sims**: 5,000 per triangle/method
- **Script**: `scripts/run_stochastic_reserving.py`

## Calibration Summary

Implied percentile = empirical CDF of actual total unpaid under each model's
simulated total IBNR distribution. Perfect calibration: uniform on [0,1].
KS statistic measures departure from uniform (lower = better).

### Calibration table (all 200 triangles per combo)

| Method | Loss type | N | Mean pctl | % in 50% | % in 80% | KS stat | Med |%err| |
|--------|-----------|---|-----------|----------|----------|---------|------------|
| ODP | Case Incurred | 200 | 0.523 | 50.0% | 74.5% | 0.0658 | 2.96% |
| ODP+CC | Case Incurred | 200 | 0.498 | 47.0% | 74.0% | 0.0678 | 2.95% |
| ODP+BF | Case Incurred | 200 | 0.515 | 46.0% | 71.0% | 0.0852 | 3.08% |
| Mack | Case Incurred | 200 | 0.527 | 31.0% | 53.5% | 0.1750 | 2.79% |
| Corr-ODP | Case Incurred | 200 | 0.526 | 16.0% | 27.5% | 0.3226 | 2.75% |
| ODP | Paid | 200 | 0.382 | 29.5% | 53.0% | 0.2608 | 3.94% |
| Mack | Paid | 200 | 0.372 | 30.0% | 50.0% | 0.2656 | 3.85% |
| Corr-ODP | Paid | 200 | 0.371 | 20.5% | 38.5% | 0.3320 | 3.83% |
| ODP+BF | Paid | 200 | 0.407 | 21.5% | 34.5% | 0.3550 | 4.58% |
| ODP+CC | Paid | 200 | 0.328 | 21.0% | 36.5% | 0.3730 | 4.05% |

## Paid vs Case-Incurred

Delta = case_incurred KS − paid KS (negative = case_incurred is better calibrated).

| Method | KS (paid) | KS (case) | Delta |
|--------|-----------|-----------|-------|
| Mack | 0.2656 | 0.1750 | -0.0906 |
| ODP | 0.2608 | 0.0658 | -0.1950 |
| Corr-ODP | 0.3320 | 0.3226 | -0.0094 |
| ODP+BF | 0.3550 | 0.0852 | -0.2698 |
| ODP+CC | 0.3730 | 0.0678 | -0.3052 |

## Headline Findings

- **Best-calibrated method on paid data**: ODP (KS=0.2608, 53.0% in central 80%)
- **Best-calibrated method on case_incurred data**: ODP (KS=0.0658, 74.5% in central 80%)

- **Paid vs case_incurred overall**: Average KS across methods: paid=0.3173, case_incurred=0.1433. Case-incurred modelling is on average *better*-calibrated vs paid.

## Figures

- `figures/standalone_implied_pctl_grid.png` — 4×10 histogram grid
- `figures/standalone_pp_paid.png` — PP chart for paid (all methods)
- `figures/standalone_pp_case_incurred.png` — PP chart for case_incurred

---

## v2 Analysis: Parametric vs Non-Parametric Bootstrap

Added `odp_param` (parametric Normal, rho=0) to isolate:
1. Non-parametric residual resampling artifacts (odp vs odp_param)
2. Pure calendar-year correlation effect (odp_param vs odp_corr)

- **Data**: 200 Meyers triangles, 5,000 sims per method
- **Script**: `scripts/run_stochastic_reserving.py` (v2 with odp_param)
- **Outputs**: `cache/meyers_standalone_results_v2.csv`, `meyers_standalone_samples_v2.parquet`

### Calibration table (6 methods × 2 loss types)

| Method | Loss type | N | Mean pctl | % in 50% | % in 80% | KS stat | Med |%err| | Med CV(IBNR) |
|--------|-----------|---|-----------|----------|----------|---------|------------|--------------|
| ODP (non-param) | Case Incurred | 200 | 0.523 | 50.0% | 74.5% | 0.0658 | 2.96% | 0.442 |
| ODP+CC | Case Incurred | 200 | 0.498 | 47.0% | 74.0% | 0.0678 | 2.95% | 0.413 |
| ODP+BF | Case Incurred | 200 | 0.515 | 46.0% | 71.0% | 0.0852 | 3.08% | 0.376 |
| Mack | Case Incurred | 200 | 0.527 | 31.0% | 53.5% | 0.1750 | 2.79% | 0.135 |
| ODP param (rho=0) | Case Incurred | 200 | 0.525 | 15.5% | 27.5% | 0.3222 | 2.75% | 0.070 |
| Corr-ODP (rho=0.1) | Case Incurred | 200 | 0.526 | 16.0% | 27.5% | 0.3226 | 2.75% | 0.072 |
| ODP (non-param) | Paid | 200 | 0.382 | 29.5% | 53.0% | 0.2608 | 3.94% | 0.190 |
| Mack | Paid | 200 | 0.372 | 30.0% | 50.0% | 0.2656 | 3.85% | 0.159 |
| Corr-ODP (rho=0.1) | Paid | 200 | 0.371 | 20.5% | 38.5% | 0.3320 | 3.83% | 0.133 |
| ODP param (rho=0) | Paid | 200 | 0.369 | 20.0% | 36.0% | 0.3430 | 3.83% | 0.127 |
| ODP+BF | Paid | 200 | 0.407 | 21.5% | 34.5% | 0.3550 | 4.58% | 0.110 |
| ODP+CC | Paid | 200 | 0.328 | 21.0% | 36.5% | 0.3730 | 4.05% | 0.126 |

### Paid vs Case-Incurred: KS stat

Delta = case_incurred KS - paid KS (positive = case_incurred harder to calibrate).

| Method | KS (paid) | KS (case) | Delta |
|--------|-----------|-----------|-------|
| Mack | 0.2656 | 0.1750 | -0.0906 |
| ODP (non-param) | 0.2608 | 0.0658 | -0.1950 |
| ODP param (rho=0) | 0.3430 | 0.3222 | -0.0208 |
| Corr-ODP (rho=0.1) | 0.3320 | 0.3226 | -0.0094 |
| ODP+BF | 0.3550 | 0.0852 | -0.2698 |
| ODP+CC | 0.3730 | 0.0678 | -0.3052 |

### Paid vs Case-Incurred: Median CV(IBNR)

| Method | CV(paid) | CV(case) | Delta |
|--------|----------|----------|-------|
| Mack | 0.1595 | 0.1352 | -0.0243 |
| ODP (non-param) | 0.1897 | 0.4415 | +0.2519 |
| ODP param (rho=0) | 0.1274 | 0.0698 | -0.0576 |
| Corr-ODP (rho=0.1) | 0.1334 | 0.0717 | -0.0617 |
| ODP+BF | 0.1102 | 0.3761 | +0.2660 |
| ODP+CC | 0.1263 | 0.4126 | +0.2862 |

### Figures (v2)

- `figures/standalone_implied_pctl_grid_v2.png` — 4×12 histogram grid (6 methods × 2 types)
- `figures/standalone_pp_paid_v2.png` — PP chart for paid
- `figures/standalone_pp_case_incurred_v2.png` — PP chart for case_incurred
- `cache/standalone_calibration_v2.csv` — per-triangle calibration detail

---

## v3 Analysis: Process Variance Calibration (lognormal default)

Previous experiments (script `21_process_variance_calibration.py`) showed that
**lognormal process variance** cuts the KS statistic dramatically:

| Process variance | KS (paid) | KS (case_incurred) |
|------------------|:---------:|:-----------------:|
| ODP (Var=phi*mu) | ~0.30 | ~0.30 |
| Gamma | ~0.39 | ~0.39 |
| NegBin | ~0.39 | ~0.39 |
| **Lognormal** | **~0.15** | **~0.21** |

This motivated changing the script default from `--process-variance odp` to
`--process-variance lognormal`, and `--rho` from 0.1 to 0.3 (matching
Clark/Ding/Zhou 2022 empirical midpoint).

---

## v4 Analysis: Final 8-Method Comparison

**Date**: see git log (commit on `prior-elicitation-2026` branch).

### Setup

- **Data**: 200 Meyers triangles (50 each for comauto, ppauto, wkcomp, othliab)
- **Methods**: mack, odp, odp_param, odp_corr, odp_bf, odp_cc, odp_corr_bf, odp_corr_cc
- **Defaults**: `--process-variance lognormal --rho 0.3 --n-sims 5000 --apriori 0.65`
- **Script**: `scripts/run_stochastic_reserving.py`
- **Analysis**: `references/meyers-backtest/22_final_calibration.py`
- **Outputs**: `cache/meyers_final.csv`, `cache/meyers_final_samples.parquet`

### Key changes in BF/CC vs v1/v2

The old `odp_bf` and `odp_cc` used `cl.BootstrapODPSample` (non-parametric residual
bootstrap) — the same approach as plain `odp` — then applied BF/CC on each resample.
These had poor calibration on paid data (KS ~0.35–0.37) because residual resampling
artifacts inflated IBNR uncertainty.

The new parametric approach:
1. Generates n_sims resampled triangles using lognormal process variance
   (sigma² estimated from CV² of chain-ladder residuals)
2. Stacks all simulations into a single batched chainladder Triangle
3. Applies `cl.BornhuetterFerguson` or `cl.CapeCod` once in batch

This is robust to negative incrementals and benefits from the same lognormal
calibration improvement as `odp_corr`.

### Final calibration table (lognormal PV, rho=0.3, n_sims=5000)

KS statistic vs uniform (lower = better); C80% = % of actuals in central 80% interval.

| Method | Loss type | N | Mean pctl | C50% | C80% | KS stat | Med CV | Med |%err| |
|--------|-----------|:-:|:---------:|:----:|:----:|:-------:|:------:|:---------:|
| odp | case_incurred | 200 | 0.523 | 50.0% | 74.5% | **0.066** | 0.442 | 3.0% |
| odp_corr | paid | 200 | 0.442 | 43.5% | 69.5% | **0.151** | 0.318 | 3.8% |
| mack | case_incurred | 200 | 0.527 | 31.0% | 53.5% | 0.175 | 0.135 | 2.8% |
| odp_param | paid | 200 | 0.429 | 38.5% | 66.0% | 0.176 | 0.267 | 3.8% |
| odp_param | case_incurred | 200 | 0.567 | 29.0% | 50.5% | 0.200 | 0.250 | 2.8% |
| odp_corr | case_incurred | 200 | 0.572 | 29.0% | 54.5% | 0.208 | 0.273 | 2.8% |
| odp | paid | 200 | 0.382 | 29.5% | 53.0% | 0.261 | 0.190 | 3.9% |
| mack | paid | 200 | 0.372 | 30.0% | 50.0% | 0.266 | 0.159 | 3.9% |
| odp_corr_cc | paid | 200 | 0.310 | 26.0% | 49.0% | 0.341 | 0.201 | 4.4% |
| odp_corr_bf | paid | 200 | 0.348 | 16.5% | 27.5% | 0.438 | 0.096 | 4.7% |
| odp_corr_cc | case_incurred | 200 | 0.245 | 23.5% | 41.0% | 0.453 | 0.259 | 5.2% |
| odp_bf | paid | 200 | 0.344 | 14.5% | 25.0% | 0.467 | 0.080 | 4.7% |
| odp_cc | paid | 200 | 0.273 | 15.0% | 27.0% | 0.476 | 0.119 | 4.4% |
| odp_corr_bf | case_incurred | 200 | 0.243 | 16.0% | 36.5% | 0.490 | 0.144 | 4.7% |
| odp_bf | case_incurred | 200 | 0.234 | 14.5% | 32.0% | 0.522 | 0.127 | 4.6% |
| odp_cc | case_incurred | 200 | 0.221 | 16.0% | 28.0% | 0.545 | 0.157 | 4.8% |

**Winners**: `odp_corr` (paid, KS=0.151) and `odp` (case_incurred, KS=0.066).

### Key findings

1. **Best overall**: `odp_corr` with lognormal process variance achieves the best
   calibration on paid data (KS=0.151, C80%=69.5%). This is the recommended method
   for paid loss triangles.

2. **BF/CC with fixed apriori=0.65 is systematically over-reserved**: Mean percentiles
   of 0.22–0.35 (vs ideal 0.50) indicate actuals consistently fall in the upper tail of
   the BF/CC distributions. The Meyers triangles tend to develop less than expected,
   making a fixed ELR prior of 0.65 too low. If using BF/CC, set `--apriori` to a value
   calibrated for the specific book; don't use the default 0.65 uncritically.

3. **Lognormal vs ODP process variance**: `odp_corr` with lognormal (KS=0.151)
   improves substantially over ODP process variance (KS~0.30 from v2 results).

4. **odp_param vs odp_corr**: Adding rho=0.3 calendar-year correlation further
   improves calibration on paid data (odp_param KS=0.176 → odp_corr KS=0.151).

5. **Case-incurred**: The non-parametric `odp` bootstrap achieves near-perfect
   calibration on case_incurred (KS=0.066), similar to v1/v2 results. Parametric
   methods are slightly over-dispersed on case_incurred.

See `cache/meyers_final_calibration.csv` for the full table.

### Figures (v4)

- `figures/final_calibration_grid.png` — 8×2 histogram grid (8 methods × 2 loss types)
- `figures/final_pp_chart.png` — PP chart (all 8 methods × 2 loss types overlaid)
