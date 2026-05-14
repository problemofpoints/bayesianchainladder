# Stochastic Reserving Benchmark Script

Self-contained Python script that runs five stochastic reserving methods on a
long-format triangle dataset. Requires only `chainladder`, `pandas`, `numpy`,
and `scipy` — no `bayesianchainladder` package needed.

## Methods

| Key | Description |
|-----|-------------|
| `mack` | Mack Chain Ladder (Mack 1993) — normal approximation |
| `odp` | ODP Bootstrap + Chain Ladder (Shapland; via `cl.BootstrapODPSample`) |
| `odp_corr` | Correlated ODP Bootstrap (Clark/Ding/Zhou 2022) — Gaussian copula calendar-year correlation |
| `odp_bf` | ODP Bootstrap + Bornhuetter-Ferguson — requires premium |
| `odp_cc` | ODP Bootstrap + Cape Cod — requires premium |

## Input Format

CSV file with one row per (origin, dev) observation:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `origin` | int | yes | Accident year (e.g. 2001) |
| `dev` | int | yes | Development age in months (12, 24, 36, …) |
| `paid` | numeric | yes | Cumulative paid losses at this development age |
| `lob` | string | no | Line of business (default: `all`) |
| `group_id` | string | no | Company / entity identifier (default: `all`) |
| `premium` | numeric | no | Earned premium for the origin year — required for `odp_bf` and `odp_cc` |

The script runs one triangle per unique `(lob, group_id)` combination.

## Output Schema

One row per `(lob, group_id, method, accident_year)` plus a `"Total"` row:

| Column | Description |
|--------|-------------|
| `lob` | Line of business |
| `group_id` | Entity identifier |
| `method` | `mack / odp / odp_corr / odp_bf / odp_cc` |
| `accident_year` | Origin year or `"Total"` |
| `paid_to_date` | Cumulative paid at the latest diagonal |
| `mean_ultimate` | Mean ultimate (paid + mean IBNR) |
| `mean_ibnr` | Mean IBNR from the distribution |
| `cv_ibnr` | Coefficient of variation of the IBNR distribution |
| `ibnr_p5` | 5th percentile of IBNR |
| `ibnr_p50` | Median IBNR |
| `ibnr_p75` | 75th percentile |
| `ibnr_p95` | 95th percentile |

## Usage

```bash
python run_stochastic_reserving.py \
  --input data.csv \
  --output results.csv \
  --methods mack odp odp_corr odp_bf odp_cc \
  --n-sims 1000 \
  --rho 0.5 \
  --apriori 0.65 \
  --n-jobs 4
```

### All options

```
--input FILE       Input CSV path (required)
--output FILE      Output CSV path (default: results.csv)
--methods ...      Space-separated list of methods to run
--n-sims INT       Bootstrap simulation count (default: 1000)
--rho FLOAT        Calendar-year correlation for odp_corr, 0=independent (default: 0.5)
--apriori FLOAT    Expected loss ratio for odp_bf (default: 0.65)
--n-jobs INT       Parallel workers; >1 uses multiprocessing.Pool (default: 1)
--random-seed INT  Random seed for reproducibility
--origin-col NAME  Column name for origin year (default: origin)
--dev-col NAME     Column name for development period (default: dev)
--paid-col NAME    Column name for cumulative paid (default: paid)
```

## Example

```bash
# Run on the bundled example (genins-based triangle):
python run_stochastic_reserving.py \
  --input example_input.csv \
  --output example_output.csv \
  --n-sims 500 \
  --random-seed 42
```

Expected total IBNR summary (approximate, n_sims=500):

```
method    mean_ibnr     cv
mack      18,200,000   0.11
odp       19,000,000   0.14
odp_corr  18,800,000   0.17
odp_bf    14,000,000   0.07
odp_cc    20,000,000   0.09
```

## Design Notes

- **Self-contained**: the correlated ODP bootstrap (Clark/Ding/Zhou 2022) is
  inlined rather than imported from `bayesianchainladder`.
- **Fault-tolerant**: errors in individual triangles are logged and skipped;
  the script continues with remaining triangles.
- **Premium optional**: if the `premium` column is absent or all-null, `odp_bf`
  and `odp_cc` are automatically skipped with a warning.
- **Development convention**: `dev` values are elapsed months since accident
  year start (12 = end of first year, 24 = end of second year, etc.). The
  script converts these to calendar dates internally.
