# Notebooks

| notebook | what it covers |
|---|---|
| `01_metalplanet_with_anvil.ipynb` | Forward modelling (orbits, limb darkening, finite exposures, batching) and fitting real Kepler/TESS-style photometry with anvil's gradient-based HMC. |
| `02_joint_transit_and_gp.ipynb` | Fitting a transit **and** correlated stellar variability together: MetalPlanet as the mean model, anvil-gp's Gaussian process over the residual, all twelve parameters sampled at once. Needs `anvilgp`. |

Run them in the environment that has `metalplanet` and `anvil` installed;
notebook 02 also needs `anvilgp`.
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

Measured on an M2 Max, both notebooks produce healthy fits at the settings
committed: notebook 01 reaches 0 divergences and R-hat 1.002 in ~25 s;
notebook 02's twelve-parameter joint fit reaches 0 divergences, R-hat 1.001 and
every parameter within 1.2 sigma in ~2 minutes.

The validation-grade version of notebook 02's fit, including a second sampler
and a cross-sampler agreement check, is anvil-gp's
`examples/hotjupiter_gp_joint.py`.
