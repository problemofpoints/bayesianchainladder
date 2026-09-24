# Sample data

The two CSV files are incremental 10×10 claims triangles copied verbatim from
Peter England's StochasticReserving repository
(https://github.com/DrPeterEngland/StochasticReserving, `Python_Examples/`,
commit e7ed85a29dba64db1192140e504f4e09cf149134), provided by EMC Actuarial
and Analytics Ltd under the MIT licence.

| File | Source file | Notes |
|---|---|---|
| `taylor_ashe.csv` | `claims_triangle.csv` | Taylor & Ashe (1983); identical to chainladder's `genins` sample. Used in England & Verrall (2002, 2006) and England, Verrall & Wüthrich (2019). |
| `liability.csv` | `liability_claims_triangle.csv` | Liability triangle from the *Example Modus Operandi* notebook; has an influential incremental at origin 3, development 7. |
| `england_reference_values.json` | rendered output tables of `EVW_2019.ipynb` and `EV_2006_PredictiveDistributions.ipynb` | Not source data — England's *published results* (reserves, standard errors, bootstrap/MCMC summaries, risk measures) transcribed from the rendered tables in his executed notebooks, used for the side-by-side comparisons in `docs/notebooks/evw_2019_one_year_view.ipynb` and `docs/notebooks/ev_2006_predictive_distributions.ipynb`, and checked against this package's own output in `tests/test_reference_values.py`. See its `_provenance` block for the exact source commit and transcription date. |

Rows are origin periods labelled 1–10; columns are development periods 1–10.
`load_england_sample` maps origin label *k* to year `first_origin + k - 1`
(default 2001) and development period *d* to `12*d` months.
