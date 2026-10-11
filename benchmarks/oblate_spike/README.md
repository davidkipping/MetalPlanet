# Oblate-planet Stage 0 spike (2026-10-10)

Throwaway measurement code behind the "Stage 0 results" section of
`docs/oblate-plan.md`. Not part of the package and not maintained; kept so
the plan's numbers can be reproduced. Paths to SquishierPlanet and the
scratch output directory are hard-coded; JAX scripts run in SquishierPlanet's
venv, MLX/Metal ones in MetalPlanet's.

| script | question |
|---|---|
| `jax_side.py`, `mp_side.py` | reference cost; fp64 f -> 0 behaviour vs the spherical path |
| `fp32_viability.py`, `fp32_smallf.py` | the reference algorithm in fp32 (stress classes; transit chords, f from 0 to 0.4) |
| `solvers.py`, `aberth_e2e.py` | quartic solvers in fp32; Aberth end to end |
| `spike_kernel.py`, `spike_time.py` | a forward fp32 Metal kernel: accuracy, ns/point, cost breakdown, housing both paths |
| `contacts.py` | per-chain contact solve; contact-rule convergence with oblate vs spherical splits |
| `savings.py`, `symmetry.py` | cost review: the inside-regime closed forms (exact) and the inner-root symmetry (false) |

`spike_kernel.build(..., inside_fast=True, fast=True, warm=True)` carries the
cost review's accepted kernel changes; `spike_time.py regime_{inside,partial}_{general,fast,warm}`
times them per regime.
