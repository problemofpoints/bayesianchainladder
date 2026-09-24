# Notebooks

| Notebook | Builds from | Source |
|---|---|---|
| `modus_operandi.ipynb` | `python build_modus_operandi.py` | Adapted from Peter England's *Example Modus Operandi* in https://github.com/DrPeterEngland/StochasticReserving (MIT) |
| `evw_2019_one_year_view.ipynb` | `python build_evw_2019.py` | Adapted from Peter England's *EVW_2019* notebook in https://github.com/DrPeterEngland/StochasticReserving (MIT), reproducing England, Verrall & Wüthrich (2019), *The one-year view of reserve risk*, Insurance: Mathematics and Economics, https://doi.org/10.1016/j.insmatheco.2018.12.002 |
| `ev_2006_predictive_distributions.ipynb` | `python build_ev_2006.py` | Adapted from Peter England's *EV_2006_PredictiveDistributions* notebook in https://github.com/DrPeterEngland/StochasticReserving (MIT), reproducing England & Verrall (2006), *Predictive distributions of outstanding liabilities in general insurance*, Annals of Actuarial Science 1(II), https://doi.org/10.1017/S1748499500000142 |

Regenerate and execute:

```bash
uv run python docs/notebooks/build_modus_operandi.py
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 docs/notebooks/modus_operandi.ipynb

uv run python docs/notebooks/build_evw_2019.py
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 docs/notebooks/evw_2019_one_year_view.ipynb

uv run python docs/notebooks/build_ev_2006.py
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 docs/notebooks/ev_2006_predictive_distributions.ipynb
```

The executed notebooks are committed so they render on GitHub. Never hand-edit
a `.ipynb` — change its `build_*.py` script and rebuild.

## Reference-value comparisons

`evw_2019_one_year_view.ipynb` and `ev_2006_predictive_distributions.ipynb`
compare this package's output, cell by cell, against Peter England's own
published figures, transcribed into
[`bayesianchainladder/data/england_reference_values.json`](../../bayesianchainladder/data/README.md)
from the rendered tables in his executed notebooks (see the JSON's
`_provenance` block for the exact source commit). Every comparable table is
shown as `ours` / `England` / `diff %`.

Two systematic differences recur and are called out inline wherever they
apply:

1. **Last-sigma rule.** `mack_analytic_rmsep` wraps
   `chainladder.MackChainladder`, which extrapolates the last Mack sigma
   log-linearly; England's own rule takes `min(sigma_7, sigma_8)`. This only
   affects the *analytic* Mack standard error (EVW 2019 Table 2) —
   `MackBootstrap.sigma_` implements England's rule directly, so every
   bootstrap-based table is unaffected.
2. **Simulation noise.** Everything downstream of a 10,000-draw Monte Carlo
   bootstrap or an MCMC fit differs from England's by ordinary sampling
   noise: same algorithm, different RNG stream and seed sequence (plus, for
   the MCMC comparisons in `ev_2006_predictive_distributions.ipynb`, this
   package's weakly informative priors versus England's flat priors, and a
   different variance function where the closest available estimator is not
   an exact replica of his model). Gaps of a few percent are typical, more
   for 99.5th-percentile tail quantiles derived from a single Monte Carlo
   sample.
