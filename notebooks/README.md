# Notebooks

| notebook | what it covers |
|---|---|
| `01_metalplanet_with_anvil.ipynb` | Forward modelling (orbits, limb darkening, finite exposures, batching) and fitting real Kepler/TESS-style photometry with anvil's gradient-based HMC. |

Run them in the environment that has `metalplanet` and `anvil` installed.
`matplotlib` is optional — every cell prints its numbers, and figures appear
only if it is importable.

The notebooks are **generated** from a sibling `build_*.py` so the prose and
code live in reviewable source rather than in JSON:

```
python notebooks/build_01.py notebooks/01_metalplanet_with_anvil.ipynb
```

Every code cell is executed before the notebook is committed. The committed
copy is stored without outputs, so opening it gives a clean run rather than
someone else's stale numbers.

A GP tutorial (transit mean model + correlated stellar noise sampled jointly)
is not included yet.
