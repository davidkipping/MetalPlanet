# Driving MetalPlanet from a sampler: the batching rule

## The one rule

**Never evaluate parameter sets one at a time. Hand the model ALL of
them in a single call.**

A GPU dispatch costs ~0.2–0.7 ms regardless of size, and the M2 Max
needs ~10^5+ threads in flight before its cores are busy. The model's
cost is therefore

    t  ≈  t_floor  +  (total points) / (~3–5 Gpt/s)

and "total points" = (parameter sets) × (points per light curve). The
GPU does not care how the product splits — 1 curve × 10^7 points and
10^4 curves × 10^3 points cost the same. Measured on the exact
looped-vs-batched comparison (10,000 parameter sets × 1,000-point
light curve):

| strategy | wall time | per curve |
|---|---:|---:|
| 10,000 separate calls (Python loop) | 1,886 ms | 189 µs |
| ONE batched call | 2.9 ms | 290 ns |

**650×.** A per-walker loop silently turns MetalPlanet into the slowest
code in the benchmark; a batched call makes it the fastest. If your
total per call is below ~3×10^4 points, a CPU code genuinely is faster
— batch harder (more walkers per call) or use the CPU path.

## How each kind of sampler should call the model

### anvil (native — nothing to do)

`metalplanet.anvil.make_quad_transit_flux(period_ref)` returns the
engine-contract function `model_fn(v, x)` with v = (n_chains, 8)
parameters and x = (2, m) epoch-centered times. The engine evaluates
every chain per likelihood call by construction; the fused Metal kernel
receives the whole (n_chains × m) grid as one 2D dispatch. This is the
production path (topology 2: GPU sampler + GPU model).

### emcee / any ensemble sampler with a vectorized log-prob

emcee accepts `vectorize=True`: the log-prob receives the whole
(n_walkers, ndim) array at once. Build the likelihood so ALL walkers go
into one model call:

```python
import numpy as np, mlx.core as mx, emcee
from metalplanet.anvil import make_quad_transit_flux
from metalplanet.orbit import epoch_center_times

model = make_quad_transit_flux(period_ref)           # fused Metal kernel inside
x = mx.array(epoch_center_times(t, t0_ref, period_ref).astype(np.float32))
y_dev = mx.array((y - 1.0).astype(np.float32))
w = mx.array((1.0 / yerr).astype(np.float32))

def log_prob_all(theta):                              # theta: (n_walkers, 8)
    v = mx.array(theta.astype(np.float32))
    r = (y_dev - model(v, x)) * w                     # (n_walkers, m)
    return np.array(mx.sum(-0.5 * r * r, axis=-1), dtype=np.float64)

sampler = emcee.EnsembleSampler(n_walkers, 8, log_prob_all, vectorize=True)
```

One GPU dispatch per ensemble move. Without `vectorize=True`, emcee
calls the model per walker and you pay the 650× penalty.

### Custom Metropolis / anything else

Same principle: propose for all chains, stack into (n_chains, ndim),
one model call, vectorized accept/reject. If your sampler framework
cannot batch, run more chains until it can — parallel chains are free
throughput on the GPU.

## Gradients: free when unused, cheap when used

There is no "gradient overhead" to switch off. MLX is lazy: the
backward pass only exists when a sampler calls `mx.grad`/`mx.vjp`, and
the forward keeps no tape (the VJP recomputes). Every forward-only
sampler (Metropolis, stretch, emcee) automatically gets the pure
forward kernel — 17.5 ms at 1024×65k. Gradient samplers (ChEES-HMC)
pay 52.5 ms for value+gradient, ~3× a forward and 519× cheaper than
reverse-mode autodiff.

## Practical limits and conditioning

- **Memory**: a batch materializes a few (n_chains × m) fp32 arrays.
  The anvil engine chunks the data axis at 65,536 points; do the same
  if n_chains × m approaches ~10^9 (the kernel itself holds everything
  in registers, so pressure comes only from inputs/outputs).
- **Long baselines in fp32**: raw absolute times lose the phase wrap
  after ~1–2 orbits (measured: 1.2e-3 flux error at 1,000 orbits). Use
  `epoch_center_times` (float64 host preprocessing → per-orbit residual
  + orbit number) as the anvil path does; then fp32 is good to the
  ~1e-6 flux level (e ≤ 0.999 with the Cartesian separation).
- **Frontend note**: the batman-style `TransitModel.light_curve(params)`
  is deliberately a one-parameter-set API (batman parity) — fine for
  plotting and single evaluations, wrong for a sampler loop. Samplers
  should use `make_quad_transit_flux` (times + parameters) or the
  array-level `flux_dev`/`flux_dev_metal` (separations + parameters).
