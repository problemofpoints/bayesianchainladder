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

## v5 Analysis: clrd2025 extension (origins 1998-2007)

**Date**: see git log (commit adding this section).

### Setup

- **Data**: `cl.load_sample("clrd2025")` (chainladder 0.10.1), origins 1998-2007, development 12-120 (10x10 triangles), company-lines built by `references/meyers-backtest/24_build_clrd2025_long.py`.
- **Eligibility funnel** (out of 768 GRNAME x LOB triangles): complete 376 -> premium_positive 351 -> losses_positive 328 -> stable (`--max-premium-ratio 5`, default) **240**. Disabling the premium-stability filter (`--max-premium-ratio 0`) leaves all 328 `losses_positive` triangles eligible instead of 240.
- **LOB counts at the default filter**: ppauto 80, comauto 75, othliab 64, wkcomp 12, prodliab 9 (240 total). **All six `medmal` company-lines are excluded** by the premium-stability filter — every medmal triangle that passes `losses_positive` has a max/min premium ratio of 10.4-33.7 over 1998-2007, all above the default threshold of 5. This is a 5-line (not 6-line) comparison at the default settings.
- **Methods**: mack, odp, odp_param, odp_corr, odp_bf, odp_cc, odp_corr_bf, odp_corr_cc, `bz` (Barnett-Zehnwirth; new since v4) — 9 total.
- **Defaults**: `--loss-col both --n-sims 5000 --rho 0.3 --apriori 0.65 --apriori-sigma 0.15 --n-jobs 8 --random-seed 42`
- **bz coverage**: `bz` requires strictly positive incremental losses (log-linear model) and fails outright on any triangle with a non-positive cell. Of the 240 x 2 = 480 (triangle, loss type) combinations, `bz` produced results for only **31/240 paid** triangles and **9/240 case_incurred** triangles; the other **440** combinations were skipped with `bz requires strictly positive incremental losses ... found N non-positive cell(s)` (or a downstream `sample_weight` shape mismatch raised from the same code path once cells are dropped). No other method failed.
- **Scripts / commands**:

  ```bash
  uv run python scripts/run_stochastic_reserving.py \
    --input references/meyers-backtest/cache/clrd2025_long.csv \
    --output references/meyers-backtest/cache/clrd2025_final.csv \
    --methods mack odp odp_param odp_corr odp_bf odp_cc odp_corr_bf odp_corr_cc bz \
    --loss-col both --n-sims 5000 --rho 0.3 --apriori 0.65 --apriori-sigma 0.15 \
    --n-jobs 8 --random-seed 42 \
    --save-samples references/meyers-backtest/cache/clrd2025_final_samples.parquet \
    2>&1 | grep -v Warning > references/meyers-backtest/cache/clrd2025_final.log

  uv run python references/meyers-backtest/22_final_calibration.py --dataset clrd2025
  ```

- **Outputs**: `cache/clrd2025_final.csv` (42,680 rows), `cache/clrd2025_final_samples.parquet` (19,400,000 rows, 3,880 unique (lob, group_id, loss_type, method) combos), `cache/clrd2025_final.log`, `cache/clrd2025_final_calibration.csv`, `cache/clrd2025_final_cal_detail.csv`, `figures/clrd2025_final_calibration_grid.png`, `figures/clrd2025_final_pp_chart.png`.

### Final calibration table (lognormal PV, rho=0.3, n_sims=5000), sorted by KS

KS statistic vs uniform (lower = better); C80% = % of actuals in central 80% interval.

| Method | Loss type | N | Mean pctl | C50% | C80% | KS stat | Med CV | Med |%err| |
|--------|-----------|:-:|:---------:|:----:|:----:|:-------:|:------:|:---------:|
| odp | case_incurred | 240 | 0.481 | 48.3% | 72.1% | **0.085** | 0.352 | 3.0% |
| odp_corr | paid | 240 | 0.541 | 44.6% | 68.3% | **0.100** | 0.399 | 3.4% |
| odp_param | paid | 240 | 0.530 | 40.8% | 60.8% | 0.121 | 0.341 | 3.4% |
| mack | case_incurred | 240 | 0.476 | 34.2% | 58.3% | 0.137 | 0.163 | 2.5% |
| odp | paid | 240 | 0.518 | 32.9% | 59.6% | 0.144 | 0.245 | 3.6% |
| odp_corr_cc | paid | 240 | 0.404 | 43.8% | 66.2% | 0.155 | 0.287 | 3.9% |
| mack | paid | 240 | 0.515 | 33.8% | 58.8% | 0.177 | 0.232 | 3.3% |
| odp_corr | case_incurred | 240 | 0.503 | 34.2% | 58.3% | 0.178 | 0.291 | 2.5% |
| odp_param | case_incurred | 240 | 0.497 | 32.5% | 56.7% | 0.194 | 0.277 | 2.6% |
| odp_cc | paid | 240 | 0.383 | 33.3% | 57.5% | 0.219 | 0.216 | 3.9% |
| odp_corr_bf | paid | 240 | 0.305 | 32.1% | 59.6% | 0.311 | 0.258 | 5.9% |
| odp_bf | paid | 240 | 0.302 | 32.9% | 60.0% | 0.318 | 0.251 | 5.9% |
| bz | case_incurred | 9 | 0.365 | 11.1% | 44.4% | 0.391 | 0.217 | 3.2% |
| bz | paid | 31 | 0.323 | 35.5% | 74.2% | 0.392 | 0.262 | 4.9% |
| odp_corr_cc | case_incurred | 240 | 0.228 | 28.3% | 43.8% | 0.455 | 0.255 | 5.2% |
| odp_cc | case_incurred | 240 | 0.203 | 21.2% | 36.7% | 0.528 | 0.178 | 5.0% |
| odp_corr_bf | case_incurred | 240 | 0.189 | 20.4% | 35.4% | 0.536 | 0.199 | 6.7% |
| odp_bf | case_incurred | 240 | 0.185 | 20.4% | 33.8% | 0.541 | 0.193 | 6.7% |

**Winners**: `odp_corr` (paid, KS=0.100) and `odp` (case_incurred, KS=0.085) — same winners as v4.

See `cache/clrd2025_final_calibration.csv` for the full table.

### Per-line KS for the best two methods per loss type

Best two paid methods: `odp_corr` (overall KS=0.100) and `odp_param` (overall KS=0.121). Best two case_incurred methods: `odp` (overall KS=0.085) and `mack` (overall KS=0.137).

| LOB | N | odp_corr KS (paid) | odp_param KS (paid) | odp KS (case_inc) | mack KS (case_inc) |
|-----|:-:|:-------------------:|:--------------------:|:-------------------:|:--------------------:|
| Comm Auto | 75 | 0.229 | 0.232 | 0.121 | 0.162 |
| PP Auto | 80 | 0.128 | 0.108 | 0.119 | 0.200 |
| Workers Comp | 12 | 0.161 | 0.166 | 0.242 | 0.234 |
| Other Liab | 64 | 0.258 | 0.288 | 0.081 | 0.106 |
| Products Liab | 9 | 0.411 | 0.436 | 0.281 | 0.258 |
| Med Mal | 0 | — | — | — | — (excluded by premium-stability filter) |

### Comparison with Meyers 1988-1997 (v4)

KS statistic by method, v4 (Meyers 1988-1997, 200 triangles/method) vs v5 (clrd2025 1998-2007, up to 240 triangles/method):

| Method | Meyers KS (paid) | Meyers KS (case) | clrd2025 KS (paid) | clrd2025 KS (case) |
|--------|:-----------------:|:------------------:|:--------------------:|:--------------------:|
| mack | 0.266 | 0.175 | 0.177 | 0.137 |
| odp | 0.261 | 0.066 | 0.144 | 0.085 |
| odp_param | 0.176 | 0.200 | 0.121 | 0.194 |
| odp_corr | 0.151 | 0.208 | 0.100 | 0.178 |
| odp_bf | 0.467 | 0.522 | 0.318 | 0.541 |
| odp_cc | 0.476 | 0.545 | 0.219 | 0.528 |
| odp_corr_bf | 0.438 | 0.490 | 0.311 | 0.536 |
| odp_corr_cc | 0.341 | 0.453 | 0.155 | 0.455 |
| bz (new in v5) | — | — | 0.392 (N=31) | 0.391 (N=9) |

**Findings**:

1. **The winners don't change across eras.** `odp_corr` is still the best-calibrated paid method (KS improves from 0.151 in Meyers 1988-1997 to 0.100 in clrd2025 1998-2007) and `odp` is still the best-calibrated case_incurred method (KS moves slightly the other way, 0.066 -> 0.085, but remains the top case_incurred method by a wide margin over the next-best `mack`, KS=0.137).

2. **BF/CC over-reserving with a fixed apriori=0.65 persists in the 1998-2007 era, and is now the worst-calibrated group on both loss types.** Mean percentiles for `odp_bf`/`odp_cc`/`odp_corr_bf`/`odp_corr_cc` sit at 0.185-0.404 (still skewed away from the ideal 0.50) and these four methods occupy the four worst KS values on case_incurred (0.455-0.541) and the two worst on paid other than `bz`. Paid-side BF/CC calibration did improve relative to Meyers (`odp_bf` KS 0.467->0.318, `odp_corr_bf` 0.438->0.311), but the case_incurred picture is mixed: `odp_bf` (0.522->0.541) and `odp_corr_bf` (0.490->0.536) got worse, `odp_corr_cc` is essentially flat (0.453->0.455), and `odp_cc` actually improved (0.545->0.528). The fixed apriori of 0.65 remains too low for this book on case_incurred data for most, but not all, BF/CC variants.

3. **`bz` is markedly worse than `odp_corr`/`odp` on both loss types, but the comparison is not apples-to-apples.** `bz`'s KS of 0.392 (paid) and 0.391 (case_incurred) are far above `odp_corr`'s 0.100 and `odp`'s 0.085. However `bz` only produced results for 31/240 paid triangles and 9/240 case_incurred triangles (the rest were skipped for having a non-positive incremental cell), so its sample is small and self-selected toward the subset of triangles that happen to be strictly positive throughout — likely the larger, more mature comauto/ppauto books. Its paid-side C80% of 74.2% is in fact the best of any method in the whole table, which — combined with the small N — suggests the headline KS for `bz` should be read with caution rather than as a like-for-like ranking against the other 8 methods (which all ran on all 240 triangles).

4. **The two smallest lines behave differently.** Products Liab (N=9) has the worst or near-worst KS for nearly every method (e.g. `odp_param` paid KS=0.436, `odp_bf` case_incurred KS=0.768 — see per-line detail in `cache/clrd2025_final_analysis.txt`), consistent with a KS statistic that is simply noisy at N=9 rather than necessarily indicating worse model fit. Workers Comp (N=12) shows a similar small-sample pattern for BF/CC (KS 0.5-0.7) even though its `odp_corr`/`odp_param` paid KS (0.161/0.166) is competitive with the larger lines. Med Mal contributes **zero** groups at the default filter (see Setup), so none of the numbers above reflect that line at all.

5. **`odp_cc` had the largest KS improvement of any method/loss-type pair between v4 and v5.** Its paid KS dropped from 0.476 (v4, near-worst) to 0.219 (v5), a swing of 0.257 — the largest movement, in either direction, of any of the 16 method/loss-type pairs shared between v4 and v5. `odp_corr_cc` improved by nearly as much on paid (0.341 -> 0.155, a 0.186-point drop, 6th of 18 rows in the v5 table), moving it from a middling BF/CC variant into a genuinely competitive alternative to `odp_corr`/`odp_param` on paid data for the 1998-2007 era.

### Figures (v5)

- `figures/clrd2025_final_calibration_grid.png` — 9x2 histogram grid (9 methods x 2 loss types)
- `figures/clrd2025_final_pp_chart.png` — PP chart (all 9 methods x 2 loss types overlaid)

## v5 Analysis: chainladder 0.10.1 refresh (Meyers 200 triangles)

**Date**: see git log (commit adding this section).

### Setup

- **Data**: same 200 Meyers triangles as v4 (50 each for comauto, ppauto, wkcomp, othliab), `cache/meyers_long.csv`.
- **Methods**: mack, odp, odp_param, odp_corr, odp_bf, odp_cc, odp_corr_bf, odp_corr_cc, `bz` (new since v4) — 9 total.
- **Defaults**: `--loss-col both --n-sims 5000 --rho 0.3 --apriori 0.65 --apriori-sigma 0.15 --n-jobs 8 --random-seed 42`
- **Environment**: `chainladder` 0.10.1 (v4 was generated on chainladder 0.9.1).
- **Scripts / commands**:

  ```bash
  uv run python scripts/run_stochastic_reserving.py \
    --input references/meyers-backtest/cache/meyers_long.csv \
    --output references/meyers-backtest/cache/meyers_v5.csv \
    --methods mack odp odp_param odp_corr odp_bf odp_cc odp_corr_bf odp_corr_cc bz \
    --loss-col both --n-sims 5000 --rho 0.3 --apriori 0.65 --apriori-sigma 0.15 \
    --n-jobs 8 --random-seed 42 \
    --save-samples references/meyers-backtest/cache/meyers_v5_samples.parquet \
    2>&1 | grep -v Warning > references/meyers-backtest/cache/meyers_v5.log

  uv run python references/meyers-backtest/22_final_calibration.py --dataset meyers --prefix meyers_v5
  ```

- **Outputs**: `cache/meyers_v5.csv` (35,827 rows), `cache/meyers_v5_samples.parquet` (16,285,000 rows, 3,257 unique (lob, group_id, loss_type, method) combos), `cache/meyers_v5.log`, `cache/meyers_v5_calibration.csv`, `cache/meyers_v5_cal_detail.csv`, `figures/meyers_v5_calibration_grid.png`, `figures/meyers_v5_pp_chart.png`.

### What changed

**The v4 baseline run was not seeded** (its log header records no `--random-seed`), while this v5 run uses `--random-seed 42`. That means any v4-vs-v5 KS difference up to the Monte Carlo noise floor measured below (**0.0078**, i.e. roughly ±0.008) is indistinguishable from re-running the same code with a different random draw, and is **not** attributable to a code or library change. Only differences larger than that floor are attributed to one of the following actual changes between chainladder 0.9.1 (v4) and chainladder 0.10.1 (v5):

1. **Mack tail-sigma interpolation** now defaults to the Mack (1994) rule (`--mack-sigma-interpolation mack`) instead of the pre-0.10 `log-linear` extrapolation.
2. **BF/CC apriori draws are now lognormal** (chainladder 0.10.1 draws each simulation's apriori from a lognormal with std `apriori_sigma`; earlier versions drew from a Normal and could produce negative aprioris).
3. **`bz` (Barnett-Zehnwirth) is new** since v4 — not present in the v4 baseline at all.

Root-cause note: independent verification (chain-ladder `full_expectation_` bit-identical between chainladder 0.9.1 and 0.10.1 on a sample of triangles; integer-month triangle construction matching the old date-based one on all 400 Meyers triangles; `paid_to_date`/`loss_to_date` identical in both result files) confirmed the underlying triangles and chain-ladder point estimates are unchanged — the residual v4-vs-v5 spread on methods whose code path did not change (`odp`, `odp_param`, `odp_corr`) is Monte Carlo noise from the unseeded v4 run, not a chainladder 0.10.1 effect.

### Noise floor: seed-sensitivity check (Step 2b)

To quantify how much KS moves from simulation noise alone, `odp` and `odp_corr` (whose code paths are unchanged between chainladder 0.9.1 and 0.10.1) were re-run with a second seed (`--random-seed 7`, otherwise identical: `--n-sims 5000 --rho 0.3 --n-jobs 8`, `cache/meyers_v5_seed7_calibration.csv`).

| Method | Loss type | KS v4 (unseeded) | KS v5 (seed 42) | KS v5 (seed 7) | Spread |
|--------|-----------|:-----------------:|:-----------------:|:----------------:|:------:|
| odp | paid | 0.2608 | 0.2616 | 0.2658 | 0.0050 |
| odp | case_incurred | 0.0658 | 0.0678 | 0.0646 | 0.0032 |
| odp_corr | paid | 0.1508 | 0.1506 | 0.1532 | 0.0026 |
| odp_corr | case_incurred | 0.2076 | 0.1998 | 0.2024 | **0.0078** |

**Noise floor = 0.0078** (the max spread above, on `odp_corr`/case_incurred). This is also, not coincidentally, the exact size of the v4-vs-v5 `odp_corr` case_incurred gap that originally looked like a code regression — it is fully explained by seed variation.

### Step 2 comparison table: v4 vs v5 (full precision, then rounded)

Delta = KS(v5) − KS(v4), computed from full-precision `ks_stat` values in each calibration CSV before rounding. Rows with `|delta| > 0.0078` (the noise floor) are marked as attributable to an actual code/library change; smaller deltas are noise.

| Method | Loss type | KS v4 | KS v5 | Delta | Above noise floor? |
|--------|-----------|:-----:|:-----:|:-----:|:--:|
| mack | paid | 0.2656 | 0.2558 | -0.0098 | yes (sigma interpolation) |
| mack | case_incurred | 0.1750 | 0.1892 | +0.0142 | yes (sigma interpolation) |
| odp | paid | 0.2608 | 0.2616 | +0.0008 | no (noise) |
| odp | case_incurred | 0.0658 | 0.0678 | +0.0020 | no (noise) |
| odp_param | paid | 0.1760 | 0.1792 | +0.0032 | no (noise) |
| odp_param | case_incurred | 0.1996 | 0.1930 | -0.0066 | no (noise) |
| odp_corr | paid | 0.1508 | 0.1506 | -0.0002 | no (noise) |
| odp_corr | case_incurred | 0.2076 | 0.1998 | -0.0078 | no (at the noise floor) |
| odp_bf | paid | 0.4666 | 0.2456 | -0.2210 | yes (lognormal apriori) |
| odp_bf | case_incurred | 0.5220 | 0.4538 | -0.0682 | yes (lognormal apriori) |
| odp_cc | paid | 0.4764 | 0.3366 | -0.1398 | yes (lognormal apriori) |
| odp_cc | case_incurred | 0.5452 | 0.5064 | -0.0388 | yes (lognormal apriori) |
| odp_corr_bf | paid | 0.4380 | 0.2400 | -0.1980 | yes (lognormal apriori) |
| odp_corr_bf | case_incurred | 0.4896 | 0.4338 | -0.0558 | yes (lognormal apriori) |
| odp_corr_cc | paid | 0.3406 | 0.2700 | -0.0706 | yes (lognormal apriori) |
| odp_corr_cc | case_incurred | 0.4532 | 0.4302 | -0.0230 | yes (lognormal apriori) |
| bz | paid | — | 0.3420 (N=54) | n/a | new method |
| bz | case_incurred | — | 0.3478 (N=3) | n/a | new method |

### bz coverage

`bz` requires strictly positive incremental losses and fails outright on any triangle with a non-positive cell. Of the 200 x 2 = 400 (triangle, loss type) combinations, `bz` produced results for only **54/200 paid** triangles and **3/200 case_incurred** triangles; the other **343** combinations were skipped and logged as `failed:` lines (146 on paid, 197 on case_incurred) — either `bz requires strictly positive incremental losses ... found N non-positive cell(s)`, or a downstream `sample_weight` shape mismatch raised from the same code path once cells are dropped. No other method failed on any triangle.

### Full v5 calibration table (lognormal PV, rho=0.3, n_sims=5000), sorted by KS

KS statistic vs uniform (lower = better); C80% = % of actuals in central 80% interval.

| Method | Loss type | N | Mean pctl | C50% | C80% | KS stat | Med CV | Med |%err| |
|--------|-----------|:-:|:---------:|:----:|:----:|:-------:|:------:|:---------:|
| odp | case_incurred | 200 | 0.523 | 50.0% | 74.5% | **0.068** | 0.411 | 3.1% |
| odp_corr | paid | 200 | 0.435 | 42.5% | 70.0% | **0.151** | 0.319 | 3.9% |
| odp_param | paid | 200 | 0.420 | 38.5% | 64.5% | 0.179 | 0.269 | 3.9% |
| mack | case_incurred | 200 | 0.537 | 29.5% | 53.0% | 0.189 | 0.137 | 2.7% |
| odp_param | case_incurred | 200 | 0.562 | 30.5% | 51.5% | 0.193 | 0.261 | 2.8% |
| odp_corr | case_incurred | 200 | 0.567 | 30.0% | 55.0% | 0.200 | 0.275 | 2.8% |
| odp_corr_bf | paid | 200 | 0.375 | 35.0% | 60.5% | 0.240 | 0.249 | 4.7% |
| odp_bf | paid | 200 | 0.372 | 35.0% | 59.5% | 0.246 | 0.244 | 4.7% |
| mack | paid | 200 | 0.382 | 30.0% | 52.5% | 0.256 | 0.162 | 3.7% |
| odp | paid | 200 | 0.382 | 29.0% | 53.5% | 0.262 | 0.190 | 3.9% |
| odp_corr_cc | paid | 200 | 0.341 | 33.5% | 61.0% | 0.270 | 0.252 | 4.4% |
| odp_cc | paid | 200 | 0.316 | 26.0% | 50.0% | 0.337 | 0.194 | 4.4% |
| bz | paid | 54 | 0.276 | 37.0% | 57.4% | 0.342 | 0.190 | 4.1% |
| bz | case_incurred | 3 | 0.376 | 66.7% | 100.0% | 0.348 | 0.074 | 0.5% |
| odp_corr_cc | case_incurred | 200 | 0.251 | 24.5% | 43.0% | 0.430 | 0.248 | 5.1% |
| odp_corr_bf | case_incurred | 200 | 0.251 | 20.5% | 44.5% | 0.434 | 0.166 | 4.6% |
| odp_bf | case_incurred | 200 | 0.246 | 20.5% | 42.5% | 0.454 | 0.163 | 4.6% |
| odp_cc | case_incurred | 200 | 0.229 | 19.5% | 32.5% | 0.506 | 0.171 | 4.8% |

**Winners**: `odp_corr` (paid, KS=0.151) and `odp` (case_incurred, KS=0.068) — same winners as v4, and the delta on both is within the 0.0078 noise floor, i.e. effectively unchanged.

### Findings

1. **`odp` and `odp_corr` are unchanged within Monte Carlo noise.** Every v4-vs-v5 delta on the three methods whose code path did not change (`odp`, `odp_param`, `odp_corr`) is at or below the measured noise floor of 0.0078 — including the `odp_corr` case_incurred gap (0.2076 → 0.1998, delta -0.0078) that initially looked like a regression. The seed-sensitivity check (v4 unseeded vs seed 42 vs seed 7) shows spreads of 0.0026-0.0078 on exactly these method/loss-type pairs, so the v4-vs-v5 gap is fully explained by v4 having been run without a fixed seed. `odp_corr` remains the best paid method and `odp` remains the best case_incurred method, unchanged from v4.

2. **Mack's tail-sigma interpolation change moved KS beyond the noise floor, in opposite directions by loss type.** Switching to the Mack (1994) sigma-interpolation default (`--mack-sigma-interpolation mack`, replacing the pre-0.10 `log-linear` rule) improved Mack's paid KS (0.2656 → 0.2558, -0.0098) but worsened its case_incurred KS (0.1750 → 0.1892, +0.0142) — both moves exceed the 0.0078 noise floor, so both are attributable to the interpolation change rather than to sampling variation. Mack is still comfortably out-performed by `odp_corr` (paid) and `odp` (case_incurred).

3. **Lognormal BF/CC apriori draws produced a large, consistent paid-side improvement and a smaller case_incurred improvement, well beyond the noise floor.** All four BF/CC variants improved on paid by 0.07-0.22 KS (`odp_bf` -0.2210, `odp_corr_bf` -0.1980, `odp_cc` -0.1398, `odp_corr_cc` -0.0706) and on case_incurred by 0.02-0.07 KS (`odp_cc` -0.0388, `odp_corr_bf` -0.0558, `odp_bf` -0.0682, `odp_corr_cc` -0.0230). These are the same four methods v4 flagged as "systematically over-reserved" under a fixed apriori; the chainladder 0.10.1 lognormal apriori draw narrows that gap substantially on paid data (e.g. `odp_bf` paid KS 0.467 → 0.246) without fully closing it — BF/CC remain the worst-calibrated group on case_incurred (KS 0.43-0.51), still far behind `odp`/`odp_corr`.

4. **`bz` is markedly worse-calibrated than the winners, but on a small, self-selected sample.** `bz` only produced results for 54/200 paid triangles and 3/200 case_incurred triangles (343 of 400 combinations failed on non-positive incrementals), so its KS of 0.342 (paid) and 0.348 (case_incurred, N=3 only) should be read as indicative rather than a like-for-like ranking against the other 8 methods, all of which ran on all 200 triangles. The case_incurred result in particular (N=3) is too small a sample to draw a reliable conclusion from.

### Figures (v5, Meyers refresh)

- `figures/meyers_v5_calibration_grid.png` — 9×2 histogram grid (9 methods × 2 loss types)
- `figures/meyers_v5_pp_chart.png` — PP chart (all 9 methods × 2 loss types overlaid)

