# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

uv is required (`requires-python = ">=3.11,<3.13"`). Project metadata uses PEP 621 (`[project]`) with PEP 735 dev deps (`[dependency-groups]`); the build backend is `hatchling`.

```bash
uv sync                                       # install package + dev deps (creates .venv, regenerates uv.lock if needed)
uv run pytest                                 # fast tests only — MCMC tests are skipped by default
uv run pytest --run-slow                      # full suite including MCMC fits (slow)
uv run pytest tests/test_models.py -k test_name --run-slow   # single test
uv run pytest tests/test_notebooks.py --run-slow   # executes docs/notebooks (~10 min)
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

Both estimators expose the same fitted surface: `.idata`, `.ibnr_`, `.ultimate_`, `.reserves_posterior_`, `.summary()`, `.sample_reserves()`. The GLM additionally supports `.build_model()` + `.sample_prior_predictive()` for prior predictive checks before committing to a full fit (see [bayesianchainladder/estimators.py:700-876](bayesianchainladder/estimators.py#L700-L876)).

### Module layout

- [bayesianchainladder/utils.py](bayesianchainladder/utils.py) — Triangle ↔ DataFrame conversion. `prepare_model_data` splits a `chainladder.Triangle` into observed and future long-format DataFrames (with `origin`, `dev`, `calendar` columns); `prepare_csr_data` emits `cumulative` / `logloss` / `premium` / `logprem` for the CSR path. `validate_triangle` is the input gate.
- [bayesianchainladder/models.py](bayesianchainladder/models.py) — Low-level builders (`build_bambi_model`, `build_pymc_model`, `build_csr_model`), MCMC driver (`fit_model`), prediction helpers (`predict_posterior`, `_predict_pymc`, `posterior_predictive_check`), prior predictive sampling (`sample_prior_predictive`), and information criteria (`compute_waic`, `compute_loo`).
- [bayesianchainladder/estimators.py](bayesianchainladder/estimators.py) — `BayesianChainLadderGLM` and `BayesianCSR`. These are the user-facing classes; the rest of the package supports them.
- [bayesianchainladder/plots.py](bayesianchainladder/plots.py) — Thin wrappers over ArviZ for standard MCMC diagnostics, plus reserve-specific plots (`plot_reserve_distribution`, `plot_heatmap_residuals`, `plot_actual_vs_fitted`) and an extensive prior predictive plotting suite (`plot_prior_predictive*`); also the England & Verrall diagnostics (`plot_fan_chart`, `plot_scaled_residuals`, `plot_sensitivity_heatmap`, `plot_capital_profiles`).
- [bayesianchainladder/_triangle_ops.py](bayesianchainladder/_triangle_ops.py) — Private numpy chain-ladder primitives operating on `(..., n_origin, n_dev)` arrays: `cumulative_array`, `cumulative_to_incremental`, `latest_diagonal`, `drop_mask`, `link_ratio_mask`, `volume_weighted_factors`, `project_cumulative`, `link_ratio_sigma`. Shared by every England & Verrall feature below.
- [bayesianchainladder/linkratio.py](bayesianchainladder/linkratio.py) — `MackBootstrap`, `NegativeBinomialBootstrap`, and the Bayesian `BayesianMackChainLadder`, plus the shared `forecast_link_ratio_paths` forecasting function.
- [bayesianchainladder/analytic.py](bayesianchainladder/analytic.py) — Analytic RMSEP oracles (`mack_analytic_rmsep`, `odp_analytic_rmsep`) used as ground truth in tests and for comparison against simulation-based reserve variability.
- [bayesianchainladder/cdr.py](bayesianchainladder/cdr.py) — One-year Claims Development Result (`claims_development_result`, `CDRResult`).
- [bayesianchainladder/riskmeasures.py](bayesianchainladder/riskmeasures.py) — VaR/TVaR/proportional-hazards transform, discounting, and cost-of-capital risk margin functions.
- [bayesianchainladder/sensitivity.py](bayesianchainladder/sensitivity.py) — Leave-one-ratio-out influence analysis (`link_ratio_sensitivity`, `top_influential`).
- [bayesianchainladder/datasets.py](bayesianchainladder/datasets.py) (+ [bayesianchainladder/data/](bayesianchainladder/data/)) — England's Taylor-Ashe and liability sample triangles plus the `load_england_sample` loader.

### Data flow (GLM path)

`chainladder.Triangle` → `prepare_model_data` → `data_` (observed) + `future_data_` (NaN cells) → `add_categorical_columns(..., formula=...)` → `build_bambi_model` → `fit_model` (NUTS, with `idata_kwargs={"log_likelihood": True}` so WAIC/LOO work) → `model.predict` on `data_` (gives `fitted_`) and `future_data_` (gives `reserves_posterior_` aggregated per origin → `ibnr_` / `ultimate_`).

### Non-obvious gotchas

- **Formula-aware categorical encoding** ([utils.py:289-384](bayesianchainladder/utils.py#L289-L384)). `add_categorical_columns` *parses the formula* to decide which of `origin`/`dev`/`calendar` should remain numeric vs. become `pd.Categorical`. It special-cases `bs(...)` / `cr(...)` splines, `**N` / `pow()` polynomials, `np.<fn>(col)` transforms, `{...}` Patsy expressions, bare unwrapped column names (`origin` vs `C(origin)`), and `_idx`-suffixed columns (which it materializes as 1-based integer indices). When adding new formula features, exercise this regex carefully — incorrect numeric/categorical inference will silently produce a wrong model.
- **Bambi prediction-kind compatibility**. `_compute_predictions` ([estimators.py:275-326](bayesianchainladder/estimators.py#L275-L326)) tries `kind="response_params"` (newer Bambi) and falls back to `kind="mean"` (older). The mean variable is then discovered as `<response>_mean` → `mu` → `posterior_predictive[response]` in that order. Preserve this fallback chain when touching prediction code.
- **Family compatibility validation** ([estimators.py:963-1007](bayesianchainladder/estimators.py#L963-L1007)). Count families reject negative values; `gamma` rejects non-positive values; non-integer counts emit a warning. The default intercept prior is data-adaptive and link-aware: log-link families use `Normal(log(positive_mean), 2.0)`; gaussian uses original-scale mean/std. Override via the `priors=` constructor arg if needed.
- **Non-default link functions** ([models.py:_get_family](bayesianchainladder/models.py)). `_get_family(family, link)` fully honours `link=` by constructing a `bmb.Family` object when the requested link differs from the family default. The critical case is `family="gamma", link="log"`: Bambi's default gamma link is `inverse`; passing `link="log"` gives the log-linear model the docstring describes. Implementation uses `bambi.defaults.utils.generate_family` so that auxiliary parameters (`alpha`, `sigma`) retain their canonical log links and Bambi's automatic prior machinery remains intact.
- **CSR model conventions** ([models.py:649-830](bayesianchainladder/models.py#L649-L830)). Mirrors the original Stan formulation: `alpha[0]` is constrained to 0, `beta[-1]` is constrained to 0 (raw parameters live on `origin_raw` / `dev_raw` coords), and variance is built from a reverse-cumsum of `1 - exp(-1/a_ig)` so it decreases with development age. Fully developed origins (no future cells) get `Ultimate = Paid` with zero uncertainty, no sampling.
- **Process variance toggle for CSR**. `BayesianCSR(include_process_variance=True)` (default) samples `Normal(mu, sigma)` then exponentiates → full lognormal posterior predictive. `False` uses the lognormal mean correction `exp(mu + σ²/2)` (parameter uncertainty only). This is **not** equivalent to "drop sigma" — make sure to keep the correction when changing.
- **CSR reserve indexing**. `_compute_predictions` evaluates `mu` at the *ultimate* development period (max of `dev_levels`), not at the next future cell — reserves are `exp(mu_ultimate) - last_observed_cumulative`. Don't try to sum incremental future cells; that's the GLM path.
- **Public API surface**. Everything in [bayesianchainladder/__init__.py](bayesianchainladder/__init__.py)'s `__all__` is exported — when adding new functions, register them there or downstream users can't import them.
- **Per-cell posterior contract.** Simulating estimators populate `full_cumulative_posterior_` (dims `origin, dev, sample`, cumulative, observed cells constant) via `_set_full_cumulative_posterior`; `reserves_posterior_` must equal `_reserves_from_full_posterior()` and tests assert it. `MackChainLadder` (normal approximation) leaves it `None`; `cdr`, `riskmeasures` and `plot_fan_chart` raise a "per-cell" `ValueError` in that case. For chainladder-backed wrappers, naively using `full_triangle_ + process_variance_` double-counts noise: `_include_process_variance` mutates `ultimate_` in place during `fit()`, and `full_triangle_` is a live property recomputed from that already-noisy `ultimate_`, so adding `process_variance_` on top applies the same noise twice. `_full_posterior_from_chainladder` (`bootstrap.py`) instead undoes the mutation (`ultimate_pre = ultimate_ - process_variance_[..., -1:]`), rebuilds the pre-noise smooth triangle via `_get_full_triangle(model.X_, ultimate_pre)`, and adds `process_variance_` back to recover the true per-cell noisy path, sliced to the original `n_dev` columns. Because each simulation's resample also perturbs the observed cells, that path is then rebased onto the real latest diagonal (`real_latest + (obj - resampled_latest)`) so observed cells stay exact while each simulation keeps its own noisy future — `full_triangle_` carries a placeholder tail column and the `9999` ultimate column beyond that. `BayesianChainLadderGLM.reserves_posterior_` omits fully developed origins while `full_cumulative_posterior_` includes every origin, so `summary_statistics()` and `scale_to_target()` see n−1 origins for a GLM and n for a bootstrap; the omitted origins carry zero reserve.
- **Two last-sigma conventions.** `link_ratio_sigma` (used by `MackBootstrap`, `NegativeBinomialBootstrap`, `BayesianMackChainLadder`, `odp_analytic_rmsep(scale="nonconstant")`, `CorrelatedBootstrapODPSample(scale="nonconstant")`) follows England: last period = min of the previous two, carry forward when n_j ≤ 1. chainladder's `Development` extrapolates log-linearly, so `mack_analytic_rmsep` (a chainladder wrapper) and the bootstraps differ by a few percent in total SD on triangles with a small last sigma. Don't "fix" one to match the other. `scale='nonconstant'` in the correlated ODP sampler changes the process-variance stage only; England also rescales the residual pool column-wise, which we do not.
- **`drop` syntax everywhere.** Link-ratio exclusions use chainladder's `(origin_label, dev_months)` with a *string* origin label naming the earlier cell of the ratio, e.g. `("2003", 72)` = the 72→84 ratio. `top_influential` returns this form; `_triangle_ops.drop_mask` consumes it. Origin labels assume annual origins (`str(int_year)`). In `CorrelatedBootstrapODPSample`/`CorrelatedBootstrapChainLadder`, `drop` is excluded from the fitted expectation used for residuals and scale estimation; the per-resample development factors are unrestricted (unlike `MackBootstrap`, `claims_development_result`, `mack_analytic_rmsep` and `link_ratio_sensitivity`, where `drop` changes the projection factors).
- **Cash-flow timing is square-triangle only.** `riskmeasures.cash_flow_periods` requires `n_origin == n_dev` and assumes origin grain == development grain; `claims_development_result` has the same restriction. Non-square triangles raise; sub-annual origin grains are rejected upstream by `_triangle_ops.cumulative_array` (duplicate integer origin labels).
- **Notebook is generated.** Edit `docs/notebooks/build_modus_operandi.py`, not the `.ipynb`; rebuild and re-execute (see `docs/notebooks/README.md`). `tests/test_notebooks.py` executes it under `--run-slow`.

### References for understanding the methodology

- [references/Stochastic Loss Reserving Using Generalized Linear Models.txt](references/Stochastic%20Loss%20Reserving%20Using%20Generalized%20Linear%20Models.txt) — Taylor & McGuire (2016), the GLM cross-classified model basis.
- [references/test-notebook.ipynb](references/test-notebook.ipynb) — runnable end-to-end usage notebook.
- Meyers (2015), CAS Monograph 1 — the CSR model.
