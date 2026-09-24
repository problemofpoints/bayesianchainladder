# Stochastic Reserving Benchmark Script

Self-contained Python script that runs nine stochastic reserving methods on a
long-format triangle dataset. Requires only `chainladder`, `pandas`, `numpy`,
and `scipy` — no `bayesianchainladder` package needed.

## Methods

| Key | Description | Requires premium |
|-----|-------------|:---:|
| `mack` | Mack Chain Ladder (Mack 1993) — normal approximation; tail sigma per Mack (1994) by default (`--mack-sigma-interpolation`) | |
| `odp` | ODP Bootstrap + Chain Ladder (Shapland; non-parametric residual bootstrap via `cl.BootstrapODPSample`) | |
| `odp_param` | Parametric ODP, rho=0 — Normal(mu, sqrt(phi\*mu)) sampling, no residual-resampling artefacts | |
| `odp_corr` | Correlated ODP Bootstrap (Clark/Ding/Zhou 2022) — Gaussian copula, rho=0.3 by default | |
| `odp_bf` | Parametric independent bootstrap (rho=0) + Bornhuetter-Ferguson — lognormal process variance | yes |
| `odp_cc` | Parametric independent bootstrap (rho=0) + Cape Cod — lognormal process variance | yes |
| `odp_corr_bf` | Parametric correlated bootstrap (rho=0.3) + Bornhuetter-Ferguson — lognormal process variance | yes |
| `odp_corr_cc` | Parametric correlated bootstrap (rho=0.3) + Cape Cod — lognormal process variance | yes |
| `bz` | Barnett-Zehnwirth probabilistic trend family (`cl.BarnettZehnwirth`) — OLS on log incrementals, coefficient-normal + lognormal process simulation; needs strictly positive incrementals | |

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
| `bz` | Frequentist analogue of the Bayesian log-link GLM; positive-incremental paid triangles; want a regression-based benchmark with explicit origin/development structure (`--bz-formula`) |

### Calibration results (Meyers 2015, 200 triangles, chainladder 0.10.1, lognormal PV, rho=0.3, n=5000, apriori_sigma=0.15)

KS statistic against uniform — lower is better calibrated (ideal = 0, uniform CDF).
Values below are the v5 refresh on chainladder 0.10.1 (`cache/meyers_v5_calibration.csv`);
see `references/meyers-backtest/STANDALONE_BACKTEST_README.md` for the full v1-v5 history,
including the v4-vs-v5 comparison and the seed-sensitivity check used to separate real
code/library changes from Monte Carlo noise.

| Method | Paid KS | Case KS | Notes |
|--------|:-------:|:-------:|-------|
| `mack` | 0.256 | 0.189 | Tail-sigma now uses Mack (1994) interpolation (chainladder 0.10.1 default) |
| `odp` | 0.262 | **0.068** | Best for case_incurred; unchanged vs v4 within Monte Carlo noise |
| `odp_param` | 0.179 | 0.193 | Unchanged vs v4 within Monte Carlo noise |
| `odp_corr` | **0.151** | 0.200 | Best for paid; unchanged vs v4 within Monte Carlo noise |
| `odp_bf` | 0.246 | 0.454 | Large paid improvement from lognormal apriori draws |
| `odp_cc` | 0.337 | 0.506 | Paid improvement from lognormal apriori draws |
| `odp_corr_bf` | 0.240 | 0.434 | Best BF/CC paid; lognormal apriori draws |
| `odp_corr_cc` | 0.270 | 0.430 | Paid improvement from lognormal apriori draws |
| `bz` | 0.342 (N=54/200) | 0.348 (N=3/200) | New method; requires strictly positive incrementals — fails on 343/400 (triangle, loss type) combos |

KS values above carry roughly ±0.005 to ±0.01 Monte Carlo uncertainty at 5,000 sims (bounded
by a seed-sensitivity check on `odp`/`odp_corr` — see `references/meyers-backtest/STANDALONE_BACKTEST_README.md`
for the full derivation); treat differences within that range as noise, not a confirmed
calibration change.

**Note on `--apriori-sigma`**: Prior to the fix, `cl.BornhuetterFerguson` was called with
`apriori_sigma=0` (the chainladder default), treating the a-priori as deterministic.
This causes near-zero BF/CC variance because IBNR = (1 − 1/CDF) × apriori × premium is
essentially deterministic when apriori is fixed — only the tiny bootstrap noise on the
latest diagonal varies across simulations.  The fix passes `apriori_sigma=0.15` so each
simulation samples its own apriori from a lognormal with mean apriori (BF) or the
Cape Cod estimate (CC) and standard deviation 0.15 (chainladder >= 0.10.1; earlier
versions drew from a Normal and could produce negative aprioris), propagating apriori
uncertainty into the reserve distribution.  Empirical cross-triangle loss-ratio std
across Meyers lines is ~0.15, making this a reasonable default.  Set `--apriori-sigma 0`
to recover the old behaviour.

## Defaults

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `--process-variance` | `lognormal` | Multiplicative noise model; back-test shows KS 0.30 → 0.15 vs ODP |
| `--rho` | `0.3` | Mid-point of Clark/Ding/Zhou (2022) empirical range 0.2–0.4 |
| `--n-sims` | `5000` | Reduces Monte Carlo noise at tail percentiles (p95) |
| `--apriori` | `0.65` | Expected loss ratio for BF; override with your own estimate |
| `--apriori-sigma` | `0.15` | Std dev of the a-priori LR for BF/CC; prevents variance collapse (see note above) |
| `--mack-sigma-interpolation` | `mack` | Mack (1994) tail-sigma rule; `log-linear` restores the pre-0.10 chainladder default |

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
| `dev` | int | yes | Development age in months (12, 24, 36, …); passed to chainladder directly as an age from the origin period start |
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
| `method` | `mack / odp / odp_param / odp_corr / odp_bf / odp_cc / odp_corr_bf / odp_corr_cc / bz` |
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
  --methods mack odp odp_param odp_corr odp_bf odp_cc odp_corr_bf odp_corr_cc bz \
  --loss-col both \
  --n-sims 5000 \
  --rho 0.3 \
  --apriori 0.65 \
  --apriori-sigma 0.15 \
  --process-variance lognormal \
  --n-jobs 4
```

### All options

```
--input FILE          Input CSV path (required)
--output FILE         Output CSV path (default: results.csv)
--methods ...         Space-separated list of methods to run.
                        Default: mack odp odp_corr odp_bf odp_cc odp_corr_bf odp_corr_cc bz
                        All 9: mack odp odp_param odp_corr odp_bf odp_cc odp_corr_bf odp_corr_cc bz
--loss-col VALUE      Loss column(s) to model (default: paid).
                        Single column:    --loss-col paid
                                          --loss-col case_incurred
                        Multiple columns: --loss-col paid,case_incurred,reported
                        Convenience alias: --loss-col both  (= paid,case_incurred)
--n-sims INT          Bootstrap simulation count (default: 5000)
--rho FLOAT           Calendar-year correlation for odp_corr / odp_corr_bf / odp_corr_cc
                        (default: 0.3; set 0 for independent)
--apriori FLOAT       Expected loss ratio for odp_bf / odp_corr_bf (default: 0.65)
--apriori-sigma FLOAT Std dev of a-priori LR for BF/CC methods (default: 0.15).
                        Controls how much apriori uncertainty widens the BF/CC reserve
                        distribution. Set 0 for deterministic apriori (variance collapse).
--process-variance PV Process variance model for parametric methods (default: lognormal)
                        Choices: lognormal (default), odp, gamma, negbin
--residual-dist DIST  Residual distribution for ODP path only (default: normal)
                        Choices: normal (default), t, skewt
--bz-formula FORMULA  Patsy formula for bz over origin/development
                        (default: C(origin)+C(development))
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
  --n-sims 5000 --rho 0.3 --apriori 0.65 --apriori-sigma 0.15 \
  --process-variance lognormal \
  --n-jobs 4 --random-seed 22
```

### Example output (n_sims=1000, lognormal PV, rho=0.3, apriori_sigma=0.15, example_input.csv)

```
loss_type  method       mean_ibnr    cv    p95
paid       odp_param    18,716,000  0.094  21,852,000
paid       odp_bf       14,132,000  0.243  19,824,000
paid       odp_corr_bf  14,121,000  0.245  19,968,000
paid       odp_cc       19,965,000  0.167  25,758,000
paid       odp_corr_cc  19,972,000  0.187  26,453,000
```

The BF/CC CV values of 0.17–0.25 are now comparable to `odp_param` (0.09) and
`odp_corr` (0.12), reflecting the apriori uncertainty contribution. Before the fix
(apriori_sigma=0) the BF/CC CV was 0.03–0.05 — near-zero variance collapse.

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
