# Stochastic Reserving Benchmark Script

Self-contained Python script that runs eight stochastic reserving methods on a
long-format triangle dataset. Requires only `chainladder`, `pandas`, `numpy`,
and `scipy` — no `bayesianchainladder` package needed.

## Methods

| Key | Description | Requires premium |
|-----|-------------|:---:|
| `mack` | Mack Chain Ladder (Mack 1993) — normal approximation | |
| `odp` | ODP Bootstrap + Chain Ladder (Shapland; non-parametric residual bootstrap via `cl.BootstrapODPSample`) | |
| `odp_param` | Parametric ODP, rho=0 — Normal(mu, sqrt(phi\*mu)) sampling, no residual-resampling artefacts | |
| `odp_corr` | Correlated ODP Bootstrap (Clark/Ding/Zhou 2022) — Gaussian copula, rho=0.3 by default | |
| `odp_bf` | Parametric independent bootstrap (rho=0) + Bornhuetter-Ferguson — lognormal process variance | yes |
| `odp_cc` | Parametric independent bootstrap (rho=0) + Cape Cod — lognormal process variance | yes |
| `odp_corr_bf` | Parametric correlated bootstrap (rho=0.3) + Bornhuetter-Ferguson — lognormal process variance | yes |
| `odp_corr_cc` | Parametric correlated bootstrap (rho=0.3) + Cape Cod — lognormal process variance | yes |

### When to use each method

| Method | Use when… |
|--------|-----------|
| `mack` | Quick sanity check; small triangles; approximate Gaussian reserve range |
| `odp` | Standard non-parametric residual bootstrap; triangle has no negative incrementals |
| `odp_param` | Triangle has negative or near-zero incrementals (e.g. case reserve releases); want parametric noise model without calendar-year correlation |
| `odp_corr` | Want calendar-year correlation to widen reserve ranges appropriately; best general-purpose parametric method |
| `odp_bf` | Prior belief about loss ratio is informative and you want BF credibility weighting; moderate development maturity |
| `odp_cc` | Premium is reliable; want Cape Cod ELR from the data itself; well-developed triangles |
| `odp_corr_bf` | BF credibility plus calendar-year correlation (recommended when both apply) |
| `odp_corr_cc` | Cape Cod plus calendar-year correlation (recommended when both apply) |

### Calibration results (Meyers 2015, 200 triangles, lognormal PV, rho=0.3, n=5000)

KS statistic against uniform — lower is better calibrated (ideal = 0, uniform CDF):

| Method | Paid KS | Case-Incurred KS |
|--------|:-------:|:----------------:|
| `mack` | ~0.22 | ~0.30 |
| `odp` | ~0.28 | ~0.30 |
| `odp_param` | ~0.18 | ~0.25 |
| `odp_corr` | ~0.15 | ~0.21 |
| `odp_bf` | *TBD* | *TBD* |
| `odp_cc` | *TBD* | *TBD* |
| `odp_corr_bf` | *TBD* | *TBD* |
| `odp_corr_cc` | *TBD* | *TBD* |

*BF/CC calibration results to be added after the final Meyers back-test completes.*

## Defaults

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `--process-variance` | `lognormal` | Multiplicative noise model; back-test shows KS 0.30 → 0.15 vs ODP |
| `--rho` | `0.3` | Mid-point of Clark/Ding/Zhou (2022) empirical range 0.2–0.4 |
| `--n-sims` | `5000` | Reduces Monte Carlo noise at tail percentiles (p95) |
| `--apriori` | `0.65` | Expected loss ratio for BF; override with your own estimate |

## Process variance options (`--process-variance`)

Controls the variance-mean relationship when sampling future incremental losses.
Applies to `odp_corr`, `odp_param`, `odp_bf`, `odp_cc`, `odp_corr_bf`, `odp_corr_cc`.
`mack` and `odp` (non-parametric) are unaffected.

| Option | Variance formula | Notes |
|--------|-----------------|-------|
| `lognormal` | Var = mu²·(exp(σ²)−1) | **Default.** Multiplicative noise; σ² fit from CV² of residuals |
| `odp` | Var = phi·mu | Linear; standard ODP (backward-compatible) |
| `gamma` | Var = mu²/alpha | Quadratic; alpha fit from residuals |
| `negbin` | Var = mu + mu²/k | Super-Poisson; heaviest tails |

## Input Format

CSV file with one row per (origin, dev) observation:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `origin` | int | yes | Accident year (e.g. 2001) |
| `dev` | int | yes | Development age in months (12, 24, 36, …) |
| `paid` | numeric | **yes** | Cumulative paid losses — always required, even when modelling `case_incurred` |
| `case_incurred` | numeric | no* | Cumulative case-incurred losses |
| `lob` | string | no | Line of business (default: `all`) |
| `group_id` | string | no | Company / entity identifier (default: `all`) |
| `premium` | numeric | no | Earned premium — required for `odp_bf`, `odp_cc`, `odp_corr_bf`, `odp_corr_cc` |

\* `paid` is always required. `--loss-col` selects which column(s) to *model*; IBNR is always
computed as `ultimate − paid_to_date` regardless of the modelled column.

The script runs one triangle per unique `(lob, group_id)` combination.

## Output Schema

One row per `(lob, group_id, loss_type, method, accident_year)` plus a `"Total"` row:

| Column | Description |
|--------|-------------|
| `lob` | Line of business |
| `group_id` | Entity identifier |
| `loss_type` | Loss column that was modelled (e.g. `paid`, `case_incurred`) |
| `method` | `mack / odp / odp_param / odp_corr / odp_bf / odp_cc / odp_corr_bf / odp_corr_cc` |
| `accident_year` | Origin year or `"Total"` |
| `loss_to_date` | Latest-diagonal value of the *modelled* loss column — informational |
| `paid_to_date` | Latest-diagonal value of `paid` — the offset used for IBNR |
| `mean_ultimate` | Mean ultimate = `paid_to_date + mean_ibnr` |
| `mean_ibnr` | Mean IBNR = `mean_ultimate − paid_to_date` (always paid-based) |
| `cv_ibnr` | Coefficient of variation of the IBNR distribution |
| `ibnr_p5` | 5th percentile of IBNR |
| `ibnr_p50` | Median IBNR |
| `ibnr_p75` | 75th percentile |
| `ibnr_p95` | 95th percentile |

### IBNR convention

**IBNR is always computed as `ultimate − paid_to_date` for ALL loss types.**

- The `paid` column is required in the input even when modelling `case_incurred`.
- `loss_to_date` shows the modelled column's latest-diagonal value.
- `paid_to_date` always shows paid; IBNR uses this as the offset.
- When `loss_type = paid`, `loss_to_date == paid_to_date`.
- When `loss_type = case_incurred`, IBNR = case-incurred ultimate − paid_to_date,
  which measures the true incurred-but-not-paid reserve.

## Usage

```bash
python run_stochastic_reserving.py \
  --input data.csv \
  --output results.csv \
  --methods mack odp odp_param odp_corr odp_bf odp_cc odp_corr_bf odp_corr_cc \
  --loss-col both \
  --n-sims 5000 \
  --rho 0.3 \
  --apriori 0.65 \
  --process-variance lognormal \
  --n-jobs 4
```

### All options

```
--input FILE          Input CSV path (required)
--output FILE         Output CSV path (default: results.csv)
--methods ...         Space-separated list of methods to run.
                        Default: mack odp odp_corr odp_bf odp_cc odp_corr_bf odp_corr_cc
                        All 8: mack odp odp_param odp_corr odp_bf odp_cc odp_corr_bf odp_corr_cc
--loss-col VALUE      Loss column(s) to model (default: paid).
                        Single column:    --loss-col paid
                                          --loss-col case_incurred
                        Multiple columns: --loss-col paid,case_incurred,reported
                        Convenience alias: --loss-col both  (= paid,case_incurred)
--n-sims INT          Bootstrap simulation count (default: 5000)
--rho FLOAT           Calendar-year correlation for odp_corr / odp_corr_bf / odp_corr_cc
                        (default: 0.3; set 0 for independent)
--apriori FLOAT       Expected loss ratio for odp_bf / odp_corr_bf (default: 0.65)
--process-variance PV Process variance model for parametric methods (default: lognormal)
                        Choices: lognormal (default), odp, gamma, negbin
--residual-dist DIST  Residual distribution for ODP path only (default: normal)
                        Choices: normal (default), t, skewt
--n-jobs INT          Parallel workers; >1 uses multiprocessing.Pool (default: 1)
--random-seed INT     Random seed for reproducibility
--save-samples PATH   Write parquet of total-IBNR samples per simulation (for
                        back-testing implied percentiles against actual ultimates)
--origin-col NAME     Column name for origin year (default: origin)
--dev-col NAME        Column name for development period (default: dev)
```

## Examples

### Quick smoke test — all 8 methods, both loss types

```bash
python run_stochastic_reserving.py \
  --input example_input.csv \
  --output smoke_results.csv \
  --loss-col both \
  --methods mack odp odp_param odp_corr odp_bf odp_cc odp_corr_bf odp_corr_cc \
  --n-sims 500 \
  --random-seed 42
```

### Full back-test with sample collection

```bash
python run_stochastic_reserving.py \
  --input meyers_long.csv \
  --output meyers_results.csv \
  --save-samples meyers_samples.parquet \
  --loss-col both \
  --methods mack odp odp_param odp_corr odp_bf odp_cc odp_corr_bf odp_corr_cc \
  --n-sims 5000 --rho 0.3 --apriori 0.65 \
  --process-variance lognormal \
  --n-jobs 4 --random-seed 22
```

### Example output (n_sims=1000, lognormal PV, rho=0.3, example_input.csv)

```
loss_type  method       mean_ibnr    cv    p95
paid       mack         18,671,000  0.105  21,883,000
paid       odp          18,806,000  0.162  24,583,000
paid       odp_param    18,716,000  0.094  21,852,000
paid       odp_corr     18,685,000  0.123  22,634,000
paid       odp_bf       14,049,000  0.029  14,723,000
paid       odp_cc       19,887,000  0.055  21,653,000
paid       odp_corr_bf  14,039,000  0.043  15,049,000
paid       odp_corr_cc  19,903,000  0.104  23,459,000
```

## Design Notes

- **Self-contained**: the correlated ODP bootstrap (Clark/Ding/Zhou 2022) is
  inlined. No `bayesianchainladder` package required.
- **Parametric BF/CC**: `odp_bf`, `odp_cc`, `odp_corr_bf`, `odp_corr_cc` use
  a vectorised stacked-Triangle approach — n_sims simulated triangles are
  assembled into a single chainladder Triangle along the key dimension, and
  `cl.BornhuetterFerguson` / `cl.CapeCod` is called once in batch. This avoids
  Python loops over simulations and is robust to negative incrementals (no
  residual resampling).
- **Fault-tolerant**: errors in individual triangles are logged and skipped.
- **IBNR always paid-based**: `paid` must be present in the input.
- **Premium optional**: if `premium` is absent or all-null, BF/CC methods are
  automatically skipped with a warning.
- **Development convention**: `dev` values are elapsed months since accident
  year start (12 = end of first year, 24 = end of second year, etc.).
- **Lognormal default**: captures the multiplicative noise structure of
  insurance losses (sigma² = log(1 + CV²), estimated once from residuals).
  Back-testing shows KS drops from ~0.30 (ODP) to ~0.15 (lognormal).
- **rho=0.3 default**: near the midpoint of the Clark/Ding/Zhou (2022)
  empirical range of 0.2–0.4 across Schedule P lines.
