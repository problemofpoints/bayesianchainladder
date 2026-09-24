# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

uv is required (`requires-python = ">=3.11,<3.13"`). Project metadata uses PEP 621 (`[project]`) with PEP 735 dev deps (`[dependency-groups]`); the build backend is `hatchling`. `chainladder>=0.10.1` is required (integer-month development ages, Mack sigma interpolation, lognormal BF/CC aprioris, BarnettZehnwirth).

```bash
uv sync                                       # install package + dev deps (creates .venv, regenerates uv.lock if needed)
uv run pytest                                 # fast tests only — MCMC tests are skipped by default
uv run pytest --run-slow                      # full suite including MCMC fits (slow)
uv run pytest tests/test_models.py -k test_name --run-slow   # single test
uv run ruff check .                           # lint (rules: E, F, W, I, UP, B, C4)
uv run black .                                # format
uv run mypy bayesianchainladder               # type check
uv add <pkg>                                  # add a runtime dependency
uv add --dev <pkg>                            # add a dev dependency (goes into [dependency-groups] dev)
```

The slow-test gating lives in [tests/conftest.py](tests/conftest.py): any test marked `@pytest.mark.slow` is skipped unless `--run-slow` is passed. Most fit/predict tests are slow because they actually run NUTS — keep this discipline when adding tests. Quick smoke fits typically use `family="gaussian"`, `draws=100`, `tune=50`, `chains=1`, `random_seed=42` (the RAA sample has negative incrementals, so it can't be fit with negbinom/poisson — see "Family compatibility" below).

CI ([.github/workflows/tests.yml](.github/workflows/tests.yml)) uses `astral-sh/setup-uv@v5` with cache + `uv sync --frozen` and runs `uv run pytest tests/ -v --tb=short` (slow tests excluded) on Python 3.11 and 3.12.

## Architecture

This package layers a scikit-learn-style estimator API on top of Bambi/PyMC for actuarial loss reserving. There are **two independent model families** sharing utility code:

1. **`BayesianChainLadderGLM`** — Bambi-based cross-classified chain ladder GLM. Formula-driven (Patsy/Bambi), supports `negativebinomial` / `poisson` / `gamma` / `gaussian` / `wald` (inverse-Gaussian) families, optional `C(calendar)` effects, and an optional log-exposure offset. The standard model is `log(μ_kj) = intercept + α_k + β_j [+ γ_{k+j-1}] [+ log(exposure)]`.
2. **`BayesianCSR`** — Glenn Meyers' (CAS Monograph 1, 2015) Changing Settlement Rate model written directly in PyMC. Lognormal on **cumulative paid loss** with log-premium offset, plus a geometric `speedup[origin] = (1-gamma)^i` factor allowing settlement-rate drift across accident years. Premium must be supplied (`premium_triangle=` or `premium_value=`).

`scripts/run_stochastic_reserving.py` is a standalone frequentist benchmark (no `bayesianchainladder` import) with nine methods including `bz` (Barnett-Zehnwirth). Its calibration back-tests live in `references/meyers-backtest/`: `16_build_meyers_long.py` (needs `reservetestr`, not a declared dependency) and `24_build_clrd2025_long.py` (needs only chainladder) produce the long CSVs; `22_final_calibration.py --dataset {meyers,clrd2025}` computes implied-percentile calibration. Results and figures are gitignored; the tables are copied into `STANDALONE_BACKTEST_README.md`.

Both estimators expose the same fitted surface: `.idata`, `.ibnr_`, `.ultimate_`, `.reserves_posterior_`, `.summary()`, `.sample_reserves()`. The GLM additionally supports `.build_model()` + `.sample_prior_predictive()` for prior predictive checks before committing to a full fit (see [bayesianchainladder/estimators.py:700-876](bayesianchainladder/estimators.py#L700-L876)).

### Module layout

- [bayesianchainladder/utils.py](bayesianchainladder/utils.py) — Triangle ↔ DataFrame conversion. `prepare_model_data` splits a `chainladder.Triangle` into observed and future long-format DataFrames (with `origin`, `dev`, `calendar` columns); `prepare_csr_data` emits `cumulative` / `logloss` / `premium` / `logprem` for the CSR path. `validate_triangle` is the input gate.
- [bayesianchainladder/models.py](bayesianchainladder/models.py) — Low-level builders (`build_bambi_model`, `build_pymc_model`, `build_csr_model`), MCMC driver (`fit_model`), prediction helpers (`predict_posterior`, `_predict_pymc`, `posterior_predictive_check`), prior predictive sampling (`sample_prior_predictive`), and information criteria (`compute_waic`, `compute_loo`).
- [bayesianchainladder/estimators.py](bayesianchainladder/estimators.py) — `BayesianChainLadderGLM` and `BayesianCSR`. These are the user-facing classes; the rest of the package supports them.
- [bayesianchainladder/plots.py](bayesianchainladder/plots.py) — Thin wrappers over ArviZ for standard MCMC diagnostics, plus reserve-specific plots (`plot_reserve_distribution`, `plot_heatmap_residuals`, `plot_actual_vs_fitted`) and an extensive prior predictive plotting suite (`plot_prior_predictive*`).

### Data flow (GLM path)

`chainladder.Triangle` → `prepare_model_data` → `data_` (observed) + `future_data_` (NaN cells) → `add_categorical_columns(..., formula=...)` → `build_bambi_model` → `fit_model` (NUTS, with `idata_kwargs={"log_likelihood": True}` so WAIC/LOO work) → `model.predict` on `data_` (gives `fitted_`) and `future_data_` (gives `reserves_posterior_` aggregated per origin → `ibnr_` / `ultimate_`).

### Non-obvious gotchas

- **Formula-aware categorical encoding** ([utils.py:289-384](bayesianchainladder/utils.py#L289-L384)). `add_categorical_columns` *parses the formula* to decide which of `origin`/`dev`/`calendar` should remain numeric vs. become `pd.Categorical`. It special-cases `bs(...)` / `cr(...)` splines, `**N` / `pow()` polynomials, `np.<fn>(col)` transforms, `{...}` Patsy expressions, bare unwrapped column names (`origin` vs `C(origin)`), and `_idx`-suffixed columns (which it materializes as 1-based integer indices). When adding new formula features, exercise this regex carefully — incorrect numeric/categorical inference will silently produce a wrong model.
- **Period encoding comes from chainladder, not arithmetic** ([utils.py:_triangle_cells](bayesianchainladder/utils.py)). `origin` is the year for annual origin grain and `YYYYMM` otherwise; `dev` is the development age in months; `calendar` is the cell's valuation date from `Triangle.valuation` (year when both grains are annual, else `YYYYMM`). Cells on one diagonal share a `calendar` label and future cells get later labels — this is what makes `C(calendar)` and `(1 | calendar)` identifiable. Do not reintroduce `origin + dev - 1`: with `dev` in months that gave every cell a unique label. `origin_labels(triangle)` is the single origin encoder: `_triangle_cells`, the bootstrap estimators' `reserves_posterior_` coordinates and `init_priors_from_chainladder`'s origin lookup all use it, so the three paths agree for every origin grain. Do not add a new `.year`-based encoding anywhere.
- **Bambi prediction-kind compatibility**. `_compute_predictions` ([estimators.py:275-326](bayesianchainladder/estimators.py#L275-L326)) tries `kind="response_params"` (newer Bambi) and falls back to `kind="mean"` (older). The mean variable is then discovered as `<response>_mean` → `mu` → `posterior_predictive[response]` in that order. Preserve this fallback chain when touching prediction code.
- **Family compatibility validation** ([estimators.py:963-1007](bayesianchainladder/estimators.py#L963-L1007)). Count families reject negative values; `gamma` rejects non-positive values; non-integer counts emit a warning. The default intercept prior is data-adaptive and link-aware: log-link families use `Normal(log(positive_mean), 2.0)`; gaussian uses original-scale mean/std. Override via the `priors=` constructor arg if needed.
- **Non-default link functions** ([models.py:_get_family](bayesianchainladder/models.py)). `_get_family(family, link)` fully honours `link=` by constructing a `bmb.Family` object when the requested link differs from the family default. The critical case is `family="gamma", link="log"`: Bambi's default gamma link is `inverse`; passing `link="log"` gives the log-linear model the docstring describes. Implementation uses `bambi.defaults.utils.generate_family` so that auxiliary parameters (`alpha`, `sigma`) retain their canonical log links and Bambi's automatic prior machinery remains intact.
- **CSR model conventions** ([models.py:649-830](bayesianchainladder/models.py#L649-L830)). Mirrors the original Stan formulation: `alpha[0]` is constrained to 0, `beta[-1]` is constrained to 0 (raw parameters live on `origin_raw` / `dev_raw` coords), and variance is built from a reverse-cumsum of `1 - exp(-1/a_ig)` so it decreases with development age. Fully developed origins (no future cells) get `Ultimate = Paid` with zero uncertainty, no sampling.
- **Process variance toggle for CSR**. `BayesianCSR(include_process_variance=True)` (default) samples `Normal(mu, sigma)` then exponentiates → full lognormal posterior predictive. `False` uses the lognormal mean correction `exp(mu + σ²/2)` (parameter uncertainty only). This is **not** equivalent to "drop sigma" — make sure to keep the correction when changing.
- **CSR reserve indexing**. `_compute_predictions` evaluates `mu` at the *ultimate* development period (max of `dev_levels`), not at the next future cell — reserves are `exp(mu_ultimate) - last_observed_cumulative`. Don't try to sum incremental future cells; that's the GLM path.
- **Public API surface**. Everything in [bayesianchainladder/__init__.py](bayesianchainladder/__init__.py)'s `__all__` is exported — when adding new functions, register them there or downstream users can't import them.

### References for understanding the methodology

- [references/Stochastic Loss Reserving Using Generalized Linear Models.txt](references/Stochastic%20Loss%20Reserving%20Using%20Generalized%20Linear%20Models.txt) — Taylor & McGuire (2016), the GLM cross-classified model basis.
- [references/test-notebook.ipynb](references/test-notebook.ipynb) — runnable end-to-end usage notebook.
- Meyers (2015), CAS Monograph 1 — the CSR model.
