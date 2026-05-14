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

### Final calibration table

*(Results will be populated once `22_final_calibration.py` completes.)*

See `cache/meyers_final_calibration.csv` for the full table after the sweep.

### Figures (v4)

- `figures/final_calibration_grid.png` — 8×2 histogram grid (8 methods × 2 loss types)
- `figures/final_pp_chart.png` — PP chart (all 8 methods × 2 loss types overlaid)
