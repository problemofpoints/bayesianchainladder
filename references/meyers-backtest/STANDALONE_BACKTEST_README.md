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
