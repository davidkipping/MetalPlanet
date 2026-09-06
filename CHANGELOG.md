# Changelog

All notable changes to MetalPlanet. Versioning: semantic-ish
(MAJOR.MINOR.PATCH); every release is tagged `vX.Y.Z` in git.

## [0.2.0] — 2026-09-06

### Added
- **Fused Metal kernels** (`metalplanet/metal.py`): the fp32 GPU path
  runs the whole model in registers. v1: photometric core
  (z, r, u1, u2) → flux + analytic-VJP kernel; v2: model-level kernel
  consuming the anvil (v, x) contract with the orbit folded in.
  Measured (M2 Max, 1024×65,536): forward 17.5 ms, value+gradient
  52.5 ms (519× reverse-mode autodiff); ~55,000 light curves/s in
  population batches. `core="metal"` is the default for the anvil
  target and the fp32 frontend, with silent graph fallback for
  fp64/CPU.
- **Markley Kepler solver** (`kepler`, `kepler_E_sincos`):
  non-iterative starter + one fifth-order refinement, a single sincos
  per point, implicit-function-theorem custom VJPs at both the true-
  anomaly and eccentric-anomaly level.
- Benchmark suite (`benchmarks/`): precision vs a 30-digit mpmath
  oracle, subprocess-isolated speed sweeps vs batman / PyTransit /
  exoplanet-core / jaxoplanet, batch scaling, and
  `verify_doc_claims.py` making every documented number reproducible.
- Docs: `docs/sampler-integration.md` (the batching rule, sampler
  recipes, verified against a spy test), `docs/eccentric-kernel-notes.md`
  (verified backward-pass formulas and kernel design findings).
- Engine-side (anvil repo): `run(progress=N)` flushed progress ticks
  with rate/acceptance/divergences/ETA.

### Changed
- **Eccentric separation is now Cartesian-from-E** — the true anomaly
  is never computed. Exactly equivalent to the previous conic tail
  (proven symbolically and to 4.5e-25 at 40 digits) and strictly
  better conditioned: removes a ~1e-6 fp32 flux-error floor near every
  transit and a ≥1e-3-flux failure mode at near-apastron transits of
  high-e orbits; fp32-trustworthy at the 1e-6 flux level to e ≤ 0.999.
- batman-style frontend evaluates through `mx.compile`d graphs with
  parameters as traced scalars (no retrace on parameter updates).

### Fixed
- Three latent dtype bugs where MLX ops on pure-Python scalar operands
  minted float32 constants inside fp64 paths (`beta`, `fac`, `ome2`).
- Circular orbit no longer fabricates a mirror transit at the far
  conjunction (front-side masking).
- 15 code-review findings in the sampler guide, including an incorrect
  emcee batching claim (half-ensembles), a priors-free example that
  sampled an improper posterior, and unreproducible measured numbers.

## [0.1.0] — 2026-09-05

Initial release: ALFM19 quadratic limb-darkened transits in MLX.
Batched fixed-iteration Bulirsch `cel`; deviation-form solution vector
with masked regimes and Taylor stability switches; Green's-basis flux
assembly; analytic custom VJP; Kipping (2013) (q1,q2) limb darkening;
epoch-centered circular orbit; batman-parity frontend
(TransitParams/TransitModel incl. eccentric orbits, supersampling,
secondary eclipses); anvil/applemcmc integration target; accurate fp64
sincos workaround for MLX's fp32-accurate fp64 trig; validation against
batman, a Limbdark.jl port, and direct mpmath integration.
