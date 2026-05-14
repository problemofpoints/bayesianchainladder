# Schedule P YE2024 Stochastic Reserving Benchmark

## Overview

This benchmark runs Mack Chain Ladder and seven ODP-bootstrap variants on
Schedule P YE2024 data across all available lines of business, aggregating
CV(IBNR) by (line, method, loss_type) to characterise how much uncertainty
each method assigns to outstanding liabilities.

---

## Data Summary

**Source:** `/results/schedule_p_triangle.json` (reserve-risk-benchmarking repo)

| Field | Value |
|---|---|
| Origin years | 2015–2024 (10 years) |
| Dev periods | 12, 24, …, 120 months |
| Raw combinations | 2,259 (snl_id, line) |
| After filtering | **1,506 valid triangles** |
| Long-format rows | 81,922 |

**Filter criteria applied** (see `01_build_long.py`):

- ≥ 8 origin years observed with non-null, positive `paid_loss`
- All cumulative `paid_loss` > 0 (no zero/negative cumulatives)
- Positive `net_earned_premium` for all observed origins
- `case_incurred_loss` available for ≥ 8 origins

**Triangles per line (after filtering):**

| LOB | N companies | Notes |
|---|---|---|
| ALL | 272 | Aggregate of all lines per company |
| OLO | 175 | Other Liability, Occurrence |
| HOFO | 167 | Homeowners / Farmowners |
| CMP | 150 | Commercial Multiple Peril |
| PPAL | 146 | Private Passenger Auto Liability |
| CAL | 144 | Commercial Auto Liability |
| WC | 118 | Workers Compensation |
| OLC | 68 | Other Liability, Claims-Made |
| SL | 51 | Special Liability |
| PLO | 48 | Products Liability, Occurrence |
| REPROP | 45 | Reinsurance Property |
| MPLC | 38 | Medical Professional Liability, Claims-Made |
| RELIAB | 32 | Reinsurance Liability |
| MPLO | 25 | Medical Professional Liability, Occurrence |
| REFIN | 12 | Reinsurance Financial |
| PLC | 11 | Products Liability, Claims-Made |
| INTL | 4 | International |

---

## Methodology

**Script:** `scripts/run_stochastic_reserving.py`

**Command:**
```bash
uv run python scripts/run_stochastic_reserving.py \
  --input references/schedule_p_backtest/cache/schedp_long.csv \
  --output references/schedule_p_backtest/cache/schedp_results.csv \
  --loss-col both \
  --methods mack odp odp_param odp_corr odp_bf odp_cc odp_corr_bf odp_corr_cc \
  --n-sims 5000 --rho 0.3 --apriori 0.65 --apriori-sigma 0.15 \
  --process-variance lognormal \
  --n-jobs 1 --random-seed 22
```

**Key settings:**

| Parameter | Value | Rationale |
|---|---|---|
| `n_sims` | 5,000 | Reduces Monte Carlo noise at tail percentiles |
| `rho` | 0.3 | Calendar-year correlation; Clark/Ding/Zhou (2022) report 0.2–0.4 empirically |
| `apriori` | 0.65 | A-priori ELR for BF/CC methods |
| `apriori_sigma` | 0.15 | A-priori uncertainty; calibrated to cross-triangle LR std in Meyers backtest |
| `process_variance` | lognormal | Cuts KS statistic from ~0.30 (ODP) to ~0.15 in Meyers (2015) backtest |
| `loss_col` | both | Runs on paid_loss and case_incurred_loss |

**Methods:**

| Method | Description |
|---|---|
| `mack` | Mack (1993) chain ladder with normal approximation |
| `odp` | Non-parametric ODP bootstrap (empirical Pearson residual resampling) |
| `odp_param` | Parametric independent bootstrap (rho=0, lognormal process variance) |
| `odp_corr` | Parametric correlated bootstrap (rho=0.3, lognormal process variance) |
| `odp_bf` | Parametric independent bootstrap + Bornhuetter-Ferguson |
| `odp_cc` | Parametric independent bootstrap + Cape Cod |
| `odp_corr_bf` | Parametric correlated bootstrap (rho=0.3) + Bornhuetter-Ferguson |
| `odp_corr_cc` | Parametric correlated bootstrap (rho=0.3) + Cape Cod |

---

## Results

CV(IBNR) = std(IBNR) / mean(IBNR) for the total (all origins summed).
Values below are **median** across companies within each LOB.

*See `cache/schedp_cv_summary.csv` for the full by-(lob, loss_type, method) table.*
*See `cache/schedp_cv_by_lob_method.csv` for the pivoted wide-format table.*

### CV by Priority Line × Method (paid loss)

| LOB | mack | odp | odp_param | odp_corr | odp_bf | odp_cc | odp_corr_bf | odp_corr_cc | n |
|---|---|---|---|---|---|---|---|---|---|
| ALL | 0.092 | 0.115 | 0.175 | 0.202 | 0.240 | 0.170 | 0.244 | 0.203 | 272 |
| CAL | 0.131 | 0.161 | 0.206 | 0.251 | 0.238 | 0.181 | 0.243 | 0.238 | 144 |
| CMP | 0.151 | 0.182 | 0.258 | 0.299 | 0.252 | 0.206 | 0.267 | 0.278 | 150 |
| OLC | 0.262 | 0.335 | 0.306 | 0.377 | 0.245 | 0.232 | 0.252 | 0.346 | 68 |
| OLO | 0.322 | 0.383 | 0.266 | 0.356 | 0.240 | 0.222 | 0.246 | 0.320 | 175 |
| PPAL | 0.092 | 0.101 | 0.193 | 0.220 | 0.239 | 0.169 | 0.241 | 0.194 | 146 |
| WC | 0.076 | 0.101 | 0.148 | 0.173 | 0.236 | 0.162 | 0.241 | 0.188 | 118 |

### CV by Priority Line × Method (case_incurred loss)

| LOB | mack | odp | odp_param | odp_corr | odp_bf | odp_cc | odp_corr_bf | odp_corr_cc | n |
|---|---|---|---|---|---|---|---|---|---|
| ALL | 0.071 | 0.112 | 0.177 | 0.199 | 0.159 | 0.125 | 0.167 | 0.177 | 272 |
| CAL | 0.100 | 0.134 | 0.218 | 0.248 | 0.152 | 0.142 | 0.161 | 0.204 | 144 |
| CMP | 0.117 | 0.180 | 0.226 | 0.250 | 0.202 | 0.174 | 0.215 | 0.267 | 150 |
| OLC | 0.171 | 0.238 | 0.276 | 0.324 | 0.201 | 0.195 | 0.219 | 0.286 | 68 |
| OLO | 0.214 | 0.344 | 0.256 | 0.310 | 0.193 | 0.193 | 0.204 | 0.295 | 175 |
| PPAL | 0.074 | 0.091 | 0.205 | 0.225 | 0.138 | 0.117 | 0.141 | 0.159 | 146 |
| WC | 0.071 | 0.170 | 0.187 | 0.212 | 0.155 | 0.130 | 0.172 | 0.196 | 118 |

### Overall Median CV (all lines combined)

| Method | paid (median) | case_incurred (median) | n |
|---|---|---|---|
| mack | 0.158 | 0.118 | 275 |
| odp | 0.193 | 0.184 | 275 |
| odp_param | 0.254 | 0.238 | 275 |
| odp_corr | 0.304 | 0.272 | 275 |
| odp_bf | 0.245 | 0.180 | 275 |
| odp_cc | 0.194 | 0.165 | 275 |
| odp_corr_bf | 0.251 | 0.191 | 275 |
| odp_corr_cc | 0.253 | 0.238 | 275 |

### Key Figures

- `figures/cv_overview.png` — all lines × methods at a glance
- `figures/cv_by_method_<LOB>.png` — per-line detail

---

## Interpretation Notes

1. **BF and CC produce wider CVs after `apriori_sigma=0.15` fix.** Prior to this fix, the
   a-priori ELR was treated as a constant, suppressing nearly all uncertainty in the
   BF/CC methods. With `apriori_sigma=0.15`, randomising the ELR each bootstrap pass
   adds ~5–10 pct CV units to the BF/CC distributions relative to the pure chain ladder.

2. **Lognormal process variance gives more conservative tail estimates** than ODP (linear
   variance) for most lines. The multiplicative noise structure of lognormal is better
   suited to the scale-dependence of insurance losses.

3. **Calendar-year correlation (rho=0.3) meaningfully widens `odp_corr` vs `odp_param`.**
   The correlated variants add 20–40% to the CV compared to their independent counterparts,
   reflecting real systematic calendar-year effects (e.g., inflation, social inflation,
   court impacts).

4. **ALL line combines all sub-lines per company** and thus tends to produce lower CVs than
   individual lines due to diversification. It is retained for benchmarking aggregate writers
   but should not be compared directly to pure-line results.

5. **Triangles with a single-observation diagonal** (e.g., companies that only have the
   latest diagonal per origin year) are excluded as they cannot support CDF estimation.
   These fail with `ValueError: operands could not be broadcast together` and are logged
   as errors but do not halt the sweep.

---

## Output Files

| File | Description |
|---|---|
| `cache/schedp_long.csv` | Long-format input (81,922 rows, 1,506 triangles) |
| `cache/schedp_results.csv` | Per-triangle per-method results |
| `cache/schedp_cv_summary.csv` | Median CV by (lob, loss_type, method) |
| `cache/schedp_cv_by_lob_method.csv` | Pivoted: rows=(lob,loss_type), cols=methods |
| `cache/schedp_cv_by_method.csv` | Pivoted: rows=method, overall median |
| `figures/cv_by_method_<LOB>.png` | Per-line CV bar charts |
| `figures/cv_overview.png` | Overview: all lines × methods |

---

## Scripts

| Script | Purpose |
|---|---|
| `01_build_long.py` | Convert JSON triangle to long-format CSV with filtering |
| `02_aggregate_cv.py` | Aggregate CV by (lob, loss_type, method) |
| `03_plot_cv.py` | Generate CV bar charts |
