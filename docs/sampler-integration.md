# Driving MetalPlanet from a sampler: the batching rule

Every measured number in this guide is reproducible by
`benchmarks/verify_doc_claims.py` (results cached in
`benchmarks/doc_claims.json`); performance figures quoted from the main
benchmark suite cite their scripts inline.

## The one rule

**Never evaluate parameter sets one at a time. Hand the model as many
of them as your sampler allows in a single call.**

A GPU dispatch costs ~0.2–0.7 ms regardless of size, and the M2 Max
needs ~10^5+ threads in flight before its cores are busy. The model's
cost per likelihood evaluation is approximately

    t  ≈  n_dispatches × t_floor  +  (total points) / R

with R ≈ 3.5–5.5 Gpt/s for the fused kernel (the upper end in large
compiled forward-only batches; `benchmarks/batch_scaling.json`), and
"total points" = (parameter sets per dispatch) × (points per curve).
The two axes are *roughly* interchangeable — but not exactly: measured
throughput varies ~35% across splits (4.1 Gpt/s at 64×100k vs 5.5 at
4096×100k), and the one-curve 10^7-point case through the batman-style
frontend measures 9.3 ms (`benchmarks/speed.json`), not the ~2 ms the
naive formula suggests. Use the formula for orders of magnitude and the
benchmark JSONs for budgets.

The rule itself is three orders of magnitude, measured
(`verify_doc_claims.py`, 10,000 parameter sets × 1,000-point curve):

| strategy | wall time | per curve |
|---|---:|---:|
| 10,000 separate calls (Python loop) | ~2,900 ms | ~290 µs |
| ONE batched call | 2.9 ms | 290 ns |

A per-walker loop turns MetalPlanet into the slowest code in the
benchmark; a batched call makes it the fastest. If a dispatch carries
fewer than ~3×10^4 points, a CPU code genuinely is faster — batch
harder or use a CPU path.

## How each kind of sampler should call the model

### anvil (native — nothing to do)

`metalplanet.anvil.make_quad_transit_flux(period_ref)` returns the
engine-contract `model_fn(v, x)`: v = (n_chains, 8) parameters,
x = (2, m) epoch-centered times. The engine evaluates every chain per
likelihood call. Note the engine's `ChunkedGaussianLogLike` slices the
*data* axis into 65,536-point chunks — so a 100,000-point likelihood is
two kernel dispatches, not one; at m = 10^6 it is sixteen. That
chunking exists for **float32 accumulation control** (a two-level
summation tree, error O(eps·√n_chunks) instead of O(eps·n) — see
anvil's precision.py), and you should keep it even when memory is not a
concern.

Two models are available: `make_quad_transit_flux` (8 parameters,
circular) and **`make_ecc_transit_flux`** (10 parameters, eccentric:
`secosw`, `sesinw` replace nothing and `b` is reinterpreted through
cos i = b (1 + e sin w) / (a (1 - e^2))). Both are fused Metal kernels.

**The eccentric model needs a prior term that the engine cannot give
you.** Periastron clearance (a(1-e) > 1+r) and a real inclination
(|cos i| <= 1) couple parameters, so no box of ParamSpecs expresses
them, and MetalPlanet clamps every numerical hazard — an unphysical
geometry returns an ordinary *finite* log-likelihood. Add
`ecc_constraint_penalty` (a smooth quadratic barrier; `-inf` walls make
HMC diverge) or wrap your likelihood in `PenalizedLogLike`, as
`make_ecc_target` does. This is the same failure mode as the missing
bounds in the emcee recipe below, and it is just as silent.

### The stretch move without CPU emcee: anvil's emcee facade

If you want an emcee-style workflow but have no hard dependency on
emcee itself, prefer `anvil.EnsembleSampler` (anvil/emcee_api.py): the
same `(nwalkers, ndim, log_prob_fn)` + `run_mcmc` + `get_chain` surface,
but the StretchMove runs entirely on-GPU — no per-move numpy upload,
no device-to-host sync inside the loop. Its `log_prob_fn` must be
batched MLX ((nwalkers, ndim) → (nwalkers,)); pair it with
`ChunkedGaussianLogLike` exactly as `metalplanet.anvil.make_target`
does.

### CPU emcee (when you need emcee itself)

emcee can batch through `vectorize=True`, **but know what it batches**:
the default `StretchMove` is a red–blue move that updates the ensemble
in halves, so your log-prob receives `(n_walkers/2, ndim)` — *two*
dispatches per move, each carrying `n_walkers/2 × m` points. Size the
ensemble so a **half**-ensemble dispatch clears the ~3×10^4-point
floor.

A correct, fast recipe (priors included — see the warning after it):

```python
import numpy as np, mlx.core as mx, emcee
from anvil.precision import ChunkedGaussianLogLike
from metalplanet.anvil import make_quad_transit_flux
from metalplanet.orbit import epoch_center_times

model = make_quad_transit_flux(period_ref)        # fused Metal kernel inside
x64 = epoch_center_times(t, t0_ref, period_ref)   # float64 host preprocessing
loglike = ChunkedGaussianLogLike(model, x64, y - 1.0, yerr)  # fp32-safe sums
loglike_c = mx.compile(lambda v: loglike(v))      # ~2x: fuses the reduction

lo = np.array([-0.5, -0.05, 0.01, 0.0, 2.0, 0.0, 0.0, -0.01])
hi = np.array([ 0.5,  0.05, 0.50, 0.9, 50., 1.0, 1.0,  0.01])

def log_prob_batch(theta):                        # (n_walkers/2, 8) from emcee
    lp = np.array(loglike_c(mx.array(theta.astype(np.float32))),
                  dtype=np.float64)
    bad = np.any((theta < lo) | (theta > hi), axis=1)
    lp[bad] = -np.inf                             # emcee treats -inf as reject
    return lp

sampler = emcee.EnsembleSampler(n_walkers, 8, log_prob_batch, vectorize=True)
```

Three things this recipe gets right that a minimal one silently gets
wrong:

1. **Priors are mandatory.** MetalPlanet clamps every numerical hazard,
   so unphysical proposals (negative r, q1 outside [0,1], |b| > a)
   return ordinary *finite* log-likelihoods — without the bounds term
   the chain wanders into flat unbounded directions and mirror modes
   (the flux depends on r²-like combinations) and samples an improper
   posterior with no error message. anvil's ParamSpec transforms impose
   these bounds for you; raw emcee does not.
2. **`mx.compile` the likelihood.** The advertised throughputs come
   from compiled graphs; the same likelihood measured eager runs ~2×
   slower at 1024×65k (12.97 vs 6.32 ms) because each eager elementwise
   op streams a full-size temporary that compile fuses away.
3. **Use `ChunkedGaussianLogLike`, not a bare `mx.sum`.** A single
   fp32 sum over ≫65k points accumulates rounding toward the ~1-unit
   Metropolis decision scale; the chunked tree is the repo's
   conditioning answer and accepts this exact `model_fn(v, x)`
   contract. (Casting the finished fp32 sum to float64 recovers
   nothing.)

### Sampling the eccentric model: what to expect

Two findings from the reference run (`examples/chees_ecc.py`, measured
by `benchmarks/v3_ecc_sampling_geometry.py`), both about the *posterior*
rather than the model:

1. **Give it a long warmup.** ChEES produced ~1% divergences at 200
   warmup iterations and exactly zero at 400-600. They are a
   step-size-adaptation artefact. They are *not* the constraint barrier:
   removing it leaves the divergence count bit-identical, and it is
   active in 0.000% of the samples drawn.
2. **Budget for slow mixing.** Transit photometry constrains a
   *combination* of (a, b, e, w) through the transit duration, so the
   posterior is a curved, strongly correlated ridge that a diagonal mass
   matrix crawls along. At 256 chains x (400 warmup + 400 samples) the
   circular 8-parameter problem reaches R-hat 1.46; the 10-parameter
   eccentric one reaches 3.53, improving from 5.54 at 100 samples — slow
   mixing, not a trap. Truth is recovered within 1 sigma on all ten
   parameters either way.

### Custom Metropolis / anything else

Same principle: propose for all chains, stack into (n_chains, ndim),
one likelihood call, vectorized accept/reject. Adding chains is close
to free **only while the GPU is unsaturated and within memory** — below
~10^6 total points per dispatch, extra chains ride along at the
dispatch floor; beyond saturation, wall time grows linearly with
chains like anywhere else.

## Gradients: free when unused, cheap when used

There is no gradient overhead to switch off. MLX autodiff is
transformation-based: the backward computation exists only when a
sampler applies `mx.grad`/`mx.vjp`, and MetalPlanet's VJPs recompute
forward-style rather than storing a tape, so plain forward calls carry
nothing extra. Forward-only samplers (Metropolis, stretch, emcee) get
the pure forward kernel — 17.5 ms at 1024×65k (`examples/bench_vjp.py`).
Gradient samplers (ChEES-HMC) pay 52.5 ms for value+gradient — ~3× a
forward, 519× cheaper than reverse-mode autodiff.

## Practical limits and conditioning

- **Memory, forward**: the fused kernel keeps intermediates in
  registers; a forward batch materializes only inputs and the output
  (~12 B per point).
- **Memory, gradients**: the analytic VJP kernel writes seven full
  (n_chains × m) per-point gradient grids plus reads the cotangent
  (~32 B per point transiently) before reducing. Budget gradient
  batches accordingly: at n×m = 10^9 that is ~28 GB of transients —
  chunk the data axis long before that (anvil's 65,536-point chunking
  handles this automatically, though its primary purpose is the fp32
  summation control described above).
- **Long baselines in fp32**: raw absolute times lose the phase wrap.
  Measured through the frontend (`verify_doc_claims.py`, P = 10 d,
  r = 0.1 circular): max flux error 2×10⁻⁷ after 1 orbit, 1.1×10⁻⁵
  after 100, 1.7×10⁻⁴ after 1,000. Use `epoch_center_times` (float64
  host preprocessing → per-orbit residual + orbit number) as the anvil
  path does; the circular fp32 path is then good to ~10⁻⁷-flux, and
  the frontend's *eccentric* path to the ~10⁻⁶-flux level for
  e ≤ 0.999.
- **Eccentricity near zero**: sample `(sqrt(e) cos w, sqrt(e) sin w)`,
  not `(e, w)` — and note that e = 0 is then an *interior* point the
  sampler genuinely visits. MetalPlanet's eccentric path is
  transit-anchored (`metalplanet/anchored.py`) precisely so that fp32
  gradients survive there: measured relative gradient error is ~1e-7
  flat from e = 0.1 down to e = 1e-8, where the textbook formulation is
  already 100% wrong (`benchmarks/v3_kh_grad_conditioning.py`).
- **Frontend note**: the batman-style `TransitModel.light_curve(params)`
  is deliberately a one-parameter-set API (batman parity) — fine for
  plotting and single evaluations, wrong as a sampler's inner loop.
  Samplers should use `make_quad_transit_flux` (times + parameters) or
  the array-level `flux_dev`/`flux_dev_metal` (separations +
  parameters).

## Why GPU-sampler + GPU-model is the strategic pairing

Of the four possible CPU/GPU splits between sampler and model, running
both on the GPU (anvil + MetalPlanet) is the only one with no
per-iteration device crossing, no Python in the hot loop, and access to
the 52.5 ms analytic gradients — while the CPU concurrently owns the
float64 work (preprocessing, re-anchoring, diagnostics) that Metal
cannot do. The measured outcome on the reference problem: ~13 ESS/s
for GPU-HMC vs ~2 ESS/s for the conventional CPU-sampler + CPU-model
stack.
