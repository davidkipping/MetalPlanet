# Changelog

All notable changes to MetalPlanet. Versioning: semantic-ish
(MAJOR.MINOR.PATCH); every release is tagged `vX.Y.Z` in git.

## [0.3.0] — 2026-09-27

The eccentric release: eccentric orbits now run on the fused Metal
kernel with analytic gradients, and have their own anvil sampling
target. Plan, gates and review log: `docs/v3eccentrickernel_plan.md`.

### Added
- **Transit-anchored eccentric orbit** (`metalplanet/anchored.py`).
  Solves for δ = E − E₀ about inferior conjunction rather than for E
  about periastron. Algebraically identical to the direct formulation
  (verified to 2.5e-15 in flux over an (e, w) grid including
  near-apastron and near-periastron transits at e = 0.999), but it never
  forms ω as an intermediate — only e·cosω and e·sinω, taken directly
  from the (√e cosω, √e sinω) sampling pair. Since ∂ω/∂k = −h/e
  diverges, the direct form computes a finite gradient as the difference
  of two O(a/e) terms: measured fp32 gradient error 5e-2 at e = 1e-5 and
  1e+1 at e = 1e-8, versus ~1e-7 *flat* for the anchored form
  (`benchmarks/v3_kh_grad_conditioning.py`).
- **v3 eccentric Metal kernel** (`make_ecc_core_metal`): anchored solve,
  Cartesian tail and ALFM19 photometry in one register-resident pass,
  plus its analytic VJP (14 per-point partials). At 1024 × 65,536:
  forward 29.0 ms, value+grad 76.0 ms — 1.41x / 1.38x the circular
  kernel, and 20x / 902x the same model as a compiled MLX graph.
  Parity 3.7e-7 vs the graph over e ∈ [0, 0.999].
- **In-kernel gradient reduction** for the circular v2 VJP
  (`make_model_core_metal(reduce="simd"|"grid")`, simd default).
  `metal::simd_sum` reduces over the *active* lanes, so early-returned
  lanes drop out by themselves — no predication, threadgroup memory,
  barrier or grid padding. Backward 35.8 → 30.8 ms (1.16x), peak memory
  4.56 → 2.74 GB, transients 1.88 GB → 58.7 MB. `"grid"` is retained as
  the parity oracle.
- **Eccentric anvil target**: `make_ecc_transit_flux` / `make_ecc_target`
  (10 parameters), `ecc_constraint_penalty` and `PenalizedLogLike` for
  the two joint constraints a box of ParamSpecs cannot express
  (periastron clearance, |cos i| ≤ 1) — without them an unphysical
  geometry returns a finite log-likelihood and the chain silently
  samples an improper posterior.
- `examples/chees_ecc.py`: eccentric ChEES-HMC injection-recovery at a
  deliberately low truth (e = 0.02), where the parameterization matters.
- Benchmarks: `bench_ecc_kernel.py`, `v3_reduction_spike.py`,
  `v3_kh_grad_conditioning.py`, `v3_ecc_graph_baseline.py`; the VJP
  profiler now A/Bs both reduction strategies.

### Changed
- Markley starter cbrt: `metal::precise::powr` (~220x a multiply, ~20%
  of orbit time) replaced by the inverse-cbrt bit trick with the
  principled (4/3)·0x3f800000 seed and three division-free Newton steps;
  1.7e-6 max relative error in-kernel against a ~1e-4 requirement.
- The batman-style frontend uses the anchored formulation for eccentric
  orbits, so φ is measured straight from t0 and the
  mean-anomaly-at-transit offset no longer appears in the compiled graph.
- Kernel sources share one `_PHOT_PARTIALS` fragment and expand through
  `_subst`, which asserts no marker survives — an unexpanded marker is a
  Metal compile failure, which aborts the process rather than raising.

### Fixed
- **NaN gradients at δ ≈ π** (ordinary apastron-side geometry):
  `_one_minus_cos` floored its denominator instead of setting it to 1 on
  the inactive branch, so the VJP hit 0 × inf for every parameter routed
  through the solve.
- **NaN gradients at exactly e = 0** — an interior point of the sampling
  disc — from `sqrt(max(e, 0))`, whose derivative is infinite there, and
  from `arctan2(0, 0)`, whose gradient is undefined.
- `test_loglike_at_truth_is_sane` now accounts for upstream anvil's
  recentring policy (the unnormalized logL is `value + log_offset_const`).

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
