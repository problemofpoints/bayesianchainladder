# Notebooks

| Notebook | Builds from | Source |
|---|---|---|
| `modus_operandi.ipynb` | `python build_modus_operandi.py` | Adapted from Peter England's *Example Modus Operandi* in https://github.com/DrPeterEngland/StochasticReserving (MIT) |

Regenerate and execute:

```bash
uv run python docs/notebooks/build_modus_operandi.py
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 docs/notebooks/modus_operandi.ipynb
```

The executed notebook is committed so it renders on GitHub.
