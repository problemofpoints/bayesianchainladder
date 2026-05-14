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
| `paid` | numeric | **yes** | Cumulative paid losses at this development age — always required, even when modelling `case_incurred` |
| `case_incurred` | numeric | no* | Cumulative case-incurred losses (required for `--loss-col case_incurred` or `--loss-col both`) |
| `lob` | string | no | Line of business (default: `all`) |
| `group_id` | string | no | Company / entity identifier (default: `all`) |
| `premium` | numeric | no | Earned premium for the origin year — required for `odp_bf` and `odp_cc` |

\* `paid` is always required. `--loss-col` selects which column(s) to *model*; IBNR is always computed as `ultimate − paid_to_date` regardless of the modelled column.

The script runs one triangle per unique `(lob, group_id)` combination.

## Output Schema

One row per `(lob, group_id, loss_type, method, accident_year)` plus a `"Total"` row:

| Column | Description |
|--------|-------------|
| `lob` | Line of business |
| `group_id` | Entity identifier |
| `loss_type` | Loss column that was modelled (e.g. `paid`, `case_incurred`) |
| `method` | `mack / odp / odp_corr / odp_bf / odp_cc` |
| `accident_year` | Origin year or `"Total"` |
| `loss_to_date` | Latest-diagonal value of the *modelled* loss column (`paid` or `case_incurred`) — informational only |
| `paid_to_date` | Latest-diagonal value of the `paid` column — the offset used for IBNR |
| `mean_ultimate` | Mean ultimate = `paid_to_date + mean_ibnr` |
| `mean_ibnr` | Mean IBNR = `mean_ultimate − paid_to_date` (always paid-based) |
| `cv_ibnr` | Coefficient of variation of the IBNR distribution |
| `ibnr_p5` | 5th percentile of IBNR |
| `ibnr_p50` | Median IBNR |
| `ibnr_p75` | 75th percentile |
| `ibnr_p95` | 95th percentile |

### IBNR convention

**IBNR is always computed as `ultimate − paid_to_date` for ALL loss types.**

- The `paid` column is required in the input even when modelling `case_incurred`
  (it is used as the offset for IBNR).
- `loss_to_date` shows the modelled column's latest-diagonal value (`paid` or
  `case_incurred`); `paid_to_date` always shows paid.
- When `loss_type = paid`, `loss_to_date == paid_to_date`.
- When `loss_type = case_incurred`, `loss_to_date` shows case-incurred
  (typically higher than paid), while `paid_to_date` shows paid; IBNR uses
  `paid_to_date` so it represents the true incurred-but-not-paid amount.

## Usage

```bash
python run_stochastic_reserving.py \
  --input data.csv \
  --output results.csv \
  --methods mack odp odp_corr odp_bf odp_cc \
  --loss-col paid \
  --n-sims 5000 \
  --rho 0.1 \
  --apriori 0.65 \
  --n-jobs 4
```

### All options

```
--input FILE          Input CSV path (required)
--output FILE         Output CSV path (default: results.csv)
--methods ...         Space-separated list of methods to run
--loss-col VALUE      Loss column(s) to model (default: paid).
                        Single column:   --loss-col paid
                                         --loss-col case_incurred
                        Multiple columns: --loss-col paid,case_incurred,reported
                        Convenience alias: --loss-col both
                          (expands to paid,case_incurred)
--n-sims INT          Bootstrap simulation count (default: 5000)
--rho FLOAT           Calendar-year correlation for odp_corr, 0=independent
                        (default: 0.1; empirical rho is typically 0.05–0.15
                        based on Schedule P / Meyers 2015 CAS Monograph 1 data)
--apriori FLOAT       Expected loss ratio for odp_bf (default: 0.65)
--n-jobs INT          Parallel workers; >1 uses multiprocessing.Pool (default: 1)
--random-seed INT     Random seed for reproducibility
--origin-col NAME     Column name for origin year (default: origin)
--dev-col NAME        Column name for development period (default: dev)
```

## Examples

### Single loss column (paid only)

```bash
python run_stochastic_reserving.py \
  --input example_input.csv \
  --output example_output_paid.csv \
  --loss-col paid \
  --n-sims 500 \
  --random-seed 42
```

### Both paid and case_incurred in one pass

```bash
python run_stochastic_reserving.py \
  --input example_input.csv \
  --output example_output_both.csv \
  --loss-col both \
  --n-sims 500 \
  --random-seed 42
```

The output will contain rows with `loss_type=paid` and `loss_type=case_incurred`.

### Three loss columns

```bash
python run_stochastic_reserving.py \
  --input data.csv \
  --output results.csv \
  --loss-col paid,case_incurred,reported \
  --n-sims 5000 \
  --random-seed 42
```

Expected total IBNR summary for the bundled example (approximate, n_sims=500, loss_col=paid):

```
loss_type  method    mean_ibnr     cv
paid       mack      18,200,000   0.11
paid       odp       19,000,000   0.14
paid       odp_corr  18,800,000   0.15
paid       odp_bf    14,000,000   0.07
paid       odp_cc    20,000,000   0.09
```

## Design Notes

- **Self-contained**: the correlated ODP bootstrap (Clark/Ding/Zhou 2022) is
  inlined rather than imported from `bayesianchainladder`.
- **Fault-tolerant**: errors in individual triangles are logged and skipped;
  the script continues with remaining triangles.
- **IBNR always paid-based**: `paid` must be present in the input. IBNR is
  computed as `ultimate − paid_to_date` for every loss type. When modelling
  `case_incurred`, the model produces a case-incurred ultimate; IBNR is then
  `case_incurred_ultimate − paid_to_date`, which correctly measures the
  incurred-but-not-paid reserve. `loss_to_date` records the modelled column's
  latest diagonal for reference; `paid_to_date` is the IBNR offset.
- **Premium optional**: if the `premium` column is absent or all-null, `odp_bf`
  and `odp_cc` are automatically skipped with a warning.
- **Development convention**: `dev` values are elapsed months since accident
  year start (12 = end of first year, 24 = end of second year, etc.). The
  script converts these to calendar dates internally.
- **Default rho=0.1**: empirical calendar-year correlations in Schedule P data
  (Meyers 2015, CAS Monograph 1) are typically 0.05–0.15. The previous default
  of 0.5 overstated correlation and inflated reserve ranges; 0.1 is more
  representative of typical portfolios.
- **Default n_sims=5000**: increased from 1000 to reduce Monte Carlo noise at
  tail percentiles (p95) without materially increasing run time.
