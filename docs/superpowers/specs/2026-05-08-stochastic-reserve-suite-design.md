# Stochastic Reserve Suite — Phase 1 Design

**Status**: Approved (2026-05-08)
**Phase**: 1 of N
**Author**: brainstormed with Claude

## Goal

Add a consistent estimator interface across all stochastic loss-reserving methods in the `bayesianchainladder` package, then implement three new method wrappers (Mack, ODP Bootstrap, Correlated ODP Bootstrap) that conform to it. The two existing Bayesian estimators are retrofitted to share the same base.

## Phase 1 Scope

In scope:
- `BaseStochasticReserve` ABC defining the shared contract
- `MackChainLadder` wrapper around `chainladder.MackChainladder`
- `BootstrapODPChainLadder` wrapper around `chainladder.BootstrapODPSample + chainladder.Chainladder`
- `CorrelatedBootstrapChainLadder` wrapper around a new low-level `CorrelatedBootstrapODPSample` (ported from the user's existing `correlated_bootstrap.py`)
- Refactor `BayesianChainLadderGLM` and `BayesianCSR` to inherit from `BaseStochasticReserve`, moving duplicated `summary()` and `sample_reserves()` logic into the base
- A `MethodSummary` dataclass and `.total_summary()` helper returning the same shape consumed by `01_run_stochastic_methods.py`

Out of scope (Phase 2+):
- Shared `DevelopmentPattern` config so multiple estimators can be forced to use the same age-to-age factors
- Bornhuetter–Ferguson and Cape Cod methods
- Cross-method model selection (covered separately in the model-selection design)

## Package Structure

```
bayesianchainladder/
├── base.py          [NEW] BaseStochasticReserve ABC + MethodSummary dataclass
├── bootstrap.py     [NEW] CorrelatedBootstrapODPSample low-level class +
│                          MackChainLadder, BootstrapODPChainLadder,
│                          CorrelatedBootstrapChainLadder wrappers
├── estimators.py    [MODIFIED] BayesianChainLadderGLM/BayesianCSR now
│                                inherit from BaseStochasticReserve
├── models.py        unchanged
├── utils.py         unchanged
├── plots.py         unchanged
└── __init__.py      [MODIFIED] export new classes + MethodSummary
```

The low-level `CorrelatedBootstrapODPSample` is a `chainladder.DevelopmentBase` subclass (sklearn-style transformer producing resampled triangles). It lives next to its wrapper in `bootstrap.py`. Source: the user's attached `correlated_bootstrap.py` module — ported unchanged except for any minor cleanup discovered during integration.

## Interface Contract

```python
# bayesianchainladder/base.py

@dataclass(frozen=True)
class MethodSummary:
    """Total-reserve summary in the shape consumed by the
    01_run_stochastic_methods.py script."""
    total_reserve_mean: float
    total_reserve_stddev: float
    total_reserve_75th_percentile: float
    total_reserve_90th_percentile: float
    total_reserve_95th_percentile: float

    @property
    def total_reserve_cv(self) -> float: ...


class BaseStochasticReserve(ABC):
    """Shared contract for all stochastic reserve estimators."""

    # Public attributes set by fit()
    triangle_: cl.Triangle | None
    ibnr_: pd.DataFrame | None             # per-origin: mean, std, median, 5/25/75/95%
    ultimate_: pd.DataFrame | None         # per-origin: paid_to_date + same stats
    reserves_posterior_: xr.DataArray | None  # dims: (origin, sample)
    _is_fitted: bool

    @abstractmethod
    def fit(self, triangle: cl.Triangle, **kwargs) -> Self: ...

    # Concrete in base — subclasses do not override:
    def summary(self, include_totals: bool = True) -> pd.DataFrame: ...
    def sample_reserves(self, n_samples: int = 1000,
                        random_seed: int | None = None) -> np.ndarray: ...
    def total_summary(self) -> MethodSummary: ...

    # Concrete helper in the base — subclasses call it from fit():
    def _build_reserve_summaries(self) -> None:
        """Compute ibnr_, ultimate_ from self.reserves_posterior_ and
        self.triangle_. Subclasses populate reserves_posterior_ during
        their fit() and then call this — they do not override it."""
```

The internal contract a subclass must satisfy after `fit()`: populate
`triangle_`, `reserves_posterior_` (an xr.DataArray of per-origin samples
with dims `(origin, sample)`), and call `self._build_reserve_summaries()`.
Everything else (`ibnr_`, `ultimate_`, `summary()`, `sample_reserves()`,
`total_summary()`) is computed once in the base.

## New Estimator Classes

### `MackChainLadder`

```python
MackChainLadder(n_periods: int = -1, ...).fit(triangle)
```

Wraps `chainladder.MackChainladder`. Mack only produces per-origin mean +
stderr; there is no native sample distribution. We populate
`reserves_posterior_` by drawing from per-origin
`Normal(mean_origin, stderr_origin)` (independent across origins). Total
samples are calibrated to `model.total_mack_std_err_` (which accounts for
cross-origin correlation) — i.e. the per-origin samples may not sum to
the calibrated total stddev, and `sample_reserves()` of the total uses
the calibrated total directly rather than summing per-origin draws.
Docstring documents this as a normal approximation.

### `BootstrapODPChainLadder`

```python
BootstrapODPChainLadder(n_sims: int = 1000, hat_adj: bool = True,
                        random_state: int | None = None,
                        n_periods: int = -1, ...).fit(triangle)
```

Wraps `chainladder.BootstrapODPSample + chainladder.Chainladder`.
`reserves_posterior_` is populated with real (non-parametric) bootstrap
samples by extracting `model.ibnr_.values`. Includes the `triangle_id` /
`kdims` workaround the user's existing script applies for multi-index
triangles.

### `CorrelatedBootstrapChainLadder`

```python
CorrelatedBootstrapChainLadder(n_sims: int = 1000, rho: float = 0.0,
                                parametric: bool = True,
                                parametric_dist: str = "normal",
                                hat_adj: bool = True,
                                random_state: int | None = None,
                                ...).fit(triangle)
```

Wraps the new low-level `CorrelatedBootstrapODPSample`. Same surface as
`BootstrapODPChainLadder` plus the correlation parameter `rho` and the
parametric distribution choice (`"normal"` or `"lognormal"`). Implements
the Clark/Ding/Zhou (2022) calendar-year-correlated bootstrap.

## Mack Sampling Treatment

Documented behavior: `MackChainLadder.sample_reserves(n_samples)` returns
`n_samples` draws from `Normal(total_mean, total_mack_std_err_)`. This is
a normal approximation, not a true posterior. Calling code that wants
non-parametric uncertainty should use `BootstrapODPChainLadder` or
`CorrelatedBootstrapChainLadder` instead. The class docstring will state
this explicitly.

Per-origin sample distributions in `reserves_posterior_` are *also*
normal approximations and are independent across origins (Mack's
covariance structure isn't computed by chainladder's public API). This
is acceptable because consumers wanting per-origin distributions almost
always want a method that produces real samples.

## Bayesian Retrofit

`BayesianChainLadderGLM` and `BayesianCSR` already match the surface.
The retrofit:

1. Inherit from `BaseStochasticReserve`.
2. Move shared logic (`summary()` total-row computation,
   `sample_reserves()` total resampling) into the base. Each Bayesian
   class loses ~80 lines of duplicated DataFrame assembly.
3. Their `fit()`, prior derivation, and prediction logic are unchanged.
4. Existing tests pass without modification.

## Public API

`__init__.py` adds to `__all__`:

```python
# Base
"BaseStochasticReserve",
"MethodSummary",

# New estimators
"MackChainLadder",
"BootstrapODPChainLadder",
"CorrelatedBootstrapChainLadder",

# Low-level (advanced users)
"CorrelatedBootstrapODPSample",
```

## Testing

Each new class needs:
- `fit_runs` smoke test on the RAA sample triangle (fast — the bootstrap
  methods are quick at `n_sims=100`).
- `summary_has_expected_columns` and `summary_total_row_present`.
- `sample_reserves_returns_correct_shape`.
- `ibnr_ultimate_consistency` (paid + ibnr_mean = ultimate_mean per
  origin).
- `total_summary_returns_method_summary` smoke test.

Bootstrap-specific:
- `correlated_bootstrap_increases_total_stddev_with_rho` (sanity check
  that rho=0.5 produces a wider distribution than rho=0).

The bootstrap tests are fast enough not to need `@pytest.mark.slow`.
Mack is fast. Only the existing Bayesian fit tests stay slow.

CI matrix unchanged.

## Migration Notes for Consumers

The user's existing `01_run_stochastic_methods.py` script can be migrated
to use the new wrappers — but this is post-Phase-1 work. The script will
keep using `cl.MackChainladder` etc. directly until then. The new
wrappers are purely additive in Phase 1.

## Risks / Open Questions

- The `CorrelatedBootstrapODPSample` low-level class in the user's
  attached file has some commented-out code blocks suggesting iterative
  development of the correlation matrix construction. We will port it
  as-is and clean up only what's clearly dead code. Any algorithmic
  changes are out of scope for Phase 1.
- The `MackChainLadder` per-origin sample distribution is a normal
  approximation. If a future consumer needs Mack with proper
  cross-origin correlation in `reserves_posterior_`, that's a Phase 2
  enhancement (probably implementing the Mack covariance manually since
  chainladder doesn't expose it).
