# Prior Elicitation & Functional-Form Study — Design

**Date:** 2026-05-08
**Owner:** AT
**Scope lines:** OLO, OLC, CAL, WC, PPAL, CMP

## Goal

Use the Schedule P YE2024 paid-loss triangles (~129 P&C groups × 15 lines, 2015–2024 origin years) at [reserve-risk-benchmarking/results/schedule_p_triangle.json](../../../reserve-risk-benchmarking/results/schedule_p_triangle.json) to produce empirically-grounded recommendations for three things, by line of business:

1. **Default prior parameter distributions** for `BayesianChainLadderGLM` and `BayesianCSR`.
2. **Most appropriate functional form** for `BayesianChainLadderGLM` (full categorical vs. dev spline vs. restricted origin vs. hierarchical pooling).
3. **A single empirical rho** per line for `CorrelatedBootstrapODPSample`.

The deliverable is an analysis report — not (yet) a code change to the package. If results suggest specific changes to defaults or new model variants, those will be filed as follow-up work.

## Output location

All artifacts live under `bayesianchainladder/references/prior-elicitation-2026/`:

```
references/prior-elicitation-2026/
├── design.md                      # this file
├── README.md                      # final writeup with per-line recommendations
├── 01_descriptive.py              # empirical / no-MCMC layer
├── 02_glm_model_comparison.py     # 4 GLM specs × 24 triangles × 6 lines
├── 03_csr_priors.py               # BayesianCSR posteriors → CSR prior recommendations
├── 04_rho_estimation.py           # empirical rho for CorrelatedBootstrapODP
├── 05_synthesize.py               # combines cached results into the README table
├── cache/                         # parquet/pickle of fits — keeps the report regenerable
└── figures/                       # png figures referenced by README.md
```

Each `0X_*.py` script is independently runnable, idempotent on its cache dir, and resumable per (line, snl_id, spec). The driver order is 01 → 02 / 03 / 04 (parallel-safe) → 05.

## Sample selection

For each line in {OLO, OLC, CAL, WC, PPAL, CMP}:

1. Load the chainladder JSON, restrict to (`line_of_business == line`), keep the `paid_loss` triangle.
2. **Eligibility filter**:
   - ≥ 8 origin years of data (out of the 10 available 2015–2024)
   - All observed cumulative paid-loss values strictly positive (lets us fit gamma without filtering cells)
   - Positive net earned premium for every origin
   - At least one cell with paid-loss > 0 in dev > 12 (excludes triangles with no real development pattern)
3. **Stratify** the surviving triangles by **booked reserve magnitude** into terciles (small / mid / large).
4. **Sample** 8 triangles per tercile uniformly at random → **24 triangles per line, 144 total**. Random seed fixed (`np.random.default_rng(20260508)`).

Sample membership is recomputed deterministically each script run via
`select_sample(line)` (md5-based seed); no separate parquet is needed.

## 01 — Descriptive layer (no MCMC)

Per line, across **all eligible triangles** (not just the 24-triangle MCMC sample), produce summary stats and figures:

| Diagnostic | Used to inform |
|---|---|
| Distribution of empirical age-to-age factors by dev | β (dev) prior scale |
| Distribution of ultimate / premium by accident year | intercept (GLM) and logelr (CSR) prior location & scale |
| Pearson dispersion φ from deterministic CL fit | σ / dispersion prior; whether overdispersion supports negbinom |
| Variance-vs-mean log-log slope of incrementals | family choice diagnostic (slope ≈ 1 → Poisson, ≈ 2 → gamma) |
| Within-triangle same-diagonal residual correlation, pooled by line | preview of the rho estimate computed formally in 04 |
| Between-company variance of log-AY-ultimate / premium | informs hyper-prior on company-level random intercept in M4 |

Figures (one per line, faceted into `figures/01_descriptive_<line>.png`):
- Age-to-age factor box plots by dev
- Loss-ratio distribution by accident year
- Variance-vs-mean log-log scatter

## 02 — GLM functional-form comparison

For each of the 24 triangles per line, fit four specs and compute WAIC + LOO:

| Spec | Formula | Notes |
|---|---|---|
| **M1** | `paid_loss ~ C(origin) + C(dev)` | Full categorical baseline (current default) |
| **M2** | `paid_loss ~ C(origin) + cr(dev, df=4)` | Dev as natural cubic spline |
| **M3** | `paid_loss ~ bs(origin, df=2) + C(dev)` | Restricted origin (linear-ish trend) |
| **M4** | `paid_loss ~ C(origin) + C(dev) + (1 \| snl_id)` | Bambi mixed-effects: random intercept per company; pools all 24 sampled companies for the line into one fit |

**Note (added during execution, 2026-05-09):** M2 (`bs(dev, df=4)`) was
attempted in smoke testing but excluded from the full sweep after every
fit produced ~100% NUTS divergences (max R-hat 2.4–3.0) across multiple
triangles and MCMC budgets. The recommendation that emerges from this
finding is that the package should not expose B-spline-on-dev as a default
spec for chain-ladder GLMs under gamma+log; categorical dev factors are
the safe choice. M3 (origin spline) remains in the comparison.

**Held fixed:**
- Family = `gamma` (positive incrementals after eligibility filter; package recommendation for non-integer counts)
- Log-exposure offset = `log(net_earned_premium)`
- Sampler = NUTS, 1000 draws, 500 tune, 2 chains, target_accept=0.9
- Default priors as currently emitted by the package (we're testing functional form, not priors here)

**Per fit, cache:**
- WAIC, LOO, p_waic, p_loo
- Posterior summary (mean / sd / r_hat / ess) for headline parameters
- Convergence flags (any r_hat > 1.05 → mark as failed; exclude from rank stats)

**Aggregation:** for each line, compute LOO ranks across the 24 triangles per spec; use Friedman test as a sanity check, and a model-rank histogram. Pick the most-frequent winner; tie-break by mean LOO delta vs M1 baseline.

**M4 design detail:** for each line, build one long-format dataframe stacking observed cells from all 24 sampled companies (columns: `snl_id`, `origin`, `dev`, `paid_loss`, `net_earned_premium`). Fit one Bambi model per line:

```python
bmb.Model(
    "paid_loss ~ C(origin) + C(dev) + (1 | snl_id)",
    data=long_df,
    family="gamma",
    link="log",
).fit(draws=1000, tune=500, chains=2)
```

This piggybacks on the package's existing `build_bambi_model` machinery — Bambi expands `(1 | snl_id)` into a partial-pooling random intercept with a hyper-prior on the company-level standard deviation. No raw-PyMC code needed.

WAIC/LOO is computed pointwise across all (c, k, j) cells. For fair head-to-head vs M1–M3, we also compute per-company aggregated log-likelihoods so each company contributes one pointwise score, matching the resolution of the per-triangle fits.

The implication if M4 wins: hierarchical pooling is the right default for sparse triangles, and the package should grow a new public API (`HierarchicalBayesianChainLadderGLM` or a `group=` argument on the existing class) — filed as follow-up, not in scope here.

## 03 — BayesianCSR prior elicitation

Fit `BayesianCSR` (default priors) on the same 24-triangle sample per line. Extract posteriors for:

- `logelr` → recommend `Normal(empirical_mean, 1.5 × empirical_sd)`
- `alpha_sig` → recommend HalfNormal scale = 1.5 × posterior 90th percentile
- `beta_sig` → same recipe as alpha_sig
- `gamma` (settlement-rate drift) → recommend `Normal(posterior_mean, posterior_sd)` — explicitly note this is the most line-sensitive parameter
- `a_ig` (variance path) → recommend HalfNormal scale matching posterior median

**Note (added during execution, 2026-05-09):** `alpha_sig` and `beta_sig` are
prior hyperparameters in the package's `BayesianCSR`, not posterior variables —
they don't appear in the fitted `idata`. The actual posterior contains
`r_alpha` (raw origin effects), `r_beta` (raw dev effects), and `sig`
(observation noise). The synthesis script summarises those instead and
recommends `csr_sig_prior` based on the median `sig` posterior across the line.

Cache one row per (line, snl_id) with posterior mean/sd/q10/q90 for each parameter. The synthesis script aggregates these to per-line recommended priors.

## 04 — Empirical rho for CorrelatedBootstrapODP

`rho` in `CorrelatedBootstrapODPSample` ([bootstrap.py:263-371](../../bayesianchainladder/bootstrap.py#L263-L371)) is the **within-triangle, calendar-year-diagonal residual correlation** following Clark/Ding/Zhou (2022). The induced correlation matrix on standardised Pearson residuals is:

- corr(cell, cell) = 1
- corr(cell_a, cell_b) = `rho` when both cells lie on the same calendar diagonal (cy_diff = 0, distinct cells)
- corr(cell_a, cell_b) = `rho^(cy_diff + 1)` when cells are on different diagonals

So the empirical estimator targets `rho` directly: the average correlation of standardised Pearson residuals between distinct cells **on the same calendar diagonal within a triangle**.

**Estimator** (cross-company pooling — exploits the fact that under the Clark/Ding/Zhou model the same `rho` governs every triangle for a given line, so we get ~129 paired observations per cell-pair instead of a few cells per diagonal in any single triangle):

For each line:
1. For each eligible triangle, fit deterministic chain-ladder and compute standardised Pearson residuals at every observed (origin, dev) cell. Use the package's residual definition (`(actual - fitted) / sqrt(phi * fitted)` with Shapland's hat-matrix adjustment) so the empirical estimate matches what the bootstrap consumes.
2. Stack residuals into a matrix `R` of shape `(n_companies, n_cells)` where `n_cells` is the number of valid upper-triangle cells (e.g., 55 for a 10×10). Cells not observed for a given company are NaN.
3. Compute the `n_cells × n_cells` pairwise sample correlation matrix `C_line` across companies, using only companies where both cells in a pair are observed (NaN-aware Pearson). Each off-diagonal entry is a correlation across ~120+ companies — a *much* tighter estimate than a single 10×10 triangle can provide.
4. Group cell-pairs by calendar-diagonal distance `cy_diff = |(i_a + j_a) - (i_b + j_b)|`. For each `d ∈ {0, 1, 2, 3, ...}`, aggregate the entries of `C_line` at that distance using a **Fisher z-transform mean** (more honest for averaging correlations than arithmetic mean): `r_d = tanh(mean(arctanh(C_line[pairs at cy_diff = d])))`.
5. **Recommended rho per line = r_0**. We also tabulate r_1, r_2, r_3 and the implied geometric ratios `r_d / rho^(d+1)` to check whether the package's geometric-decay structure fits the line; deviations are reported as a diagnostic.
6. **Bootstrap CI**: 1,000 company-level resamples (resample companies with replacement, recompute steps 2–4) → median + 80% CI on `rho`.

Implementation notes:
- Drop companies with too-sparse residuals (e.g., < 30 valid cells) before stacking.
- Cell-pairs with fewer than ~20 jointly-observed companies are excluded from the Fisher-z aggregation.
- The rho preview (correlation-by-cy_diff plot per line) is produced by `04_rho_estimation.py`, not in the descriptive layer (01).

## 05 — Synthesis

Reads all caches and produces the master recommendations table:

| Line | Family | Best GLM form | Intercept prior | β (dev) prior | σ prior | CSR logelr | CSR γ | ρ |
|---|---|---|---|---|---|---|---|---|

…plus per-line subsections in README.md showing the supporting evidence (figures + summary stats).

## Compute budget

- M1 / M2 / M3 (per-triangle): 24 triangles × 3 specs × 6 lines = **432 GLM fits**
- M4 (hierarchical, one fit per line): **6 fits**, each on a stacked dataset of ~24 × ~55 = ~1,300 rows
- CSR: 24 triangles × 6 lines = **144 fits**

At ~30 sec per per-triangle GLM, ~3–5 min per M4 hierarchical fit, and ~60 sec per CSR fit: roughly **4–6 hours wall-clock**. Scripts cache per (line, snl_id, spec) so partial runs are resumable; the 6 M4 fits cache per (line, spec) instead.

## Reproducibility

- All randomness seeded (sample selection, NUTS init).
- `cache/` written as parquet (so partial state is inspectable).
- README is regenerated from cache by `05_synthesize.py` — no hand-edits.
- The `requirements` are already in the repo's `pyproject.toml` (chainladder, bambi, pymc, arviz). No new deps expected.

## Out of scope

- Calendar-year (γ) effect priors for `BayesianChainLadderGLM` — the cross-classified default in the package has γ off by default; we'll note an empirical observation if calendar effects are strong but won't elicit a γ prior in this round.
- Family selection sweep (e.g., negbinom vs gamma vs gaussian) — descriptive diagnostics will flag if gamma is implausible for any line, but the formal sweep is held to keep the functional-form comparison clean.
- Lines outside {OLO, OLC, CAL, WC, PPAL, CMP}.
- Code changes to `bayesianchainladder/` — this study informs design, doesn't ship it.
