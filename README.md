# MetalPlanet

Analytic, differentiable exoplanet transit light curves on Apple Silicon,
built on [MLX](https://github.com/ml-explore/mlx) — the
[Agol, Luger & Foreman-Mackey (2020)](https://arxiv.org/abs/1908.03222)
formulation with a [batman](https://github.com/lkreidberg/batman)-style
frontend.

If you know batman, you know MetalPlanet:

```python
import numpy as np
import metalplanet

params = metalplanet.TransitParams()
params.t0 = 0.0                    # time of inferior conjunction
params.per = 1.0                   # orbital period
params.rp = 0.1                    # planet radius [stellar radii]
params.a = 15.0                    # semi-major axis [stellar radii]
params.inc = 87.0                  # inclination [degrees]
params.ecc = 0.0                   # eccentricity
params.w = 90.0                    # longitude of periastron [degrees]
params.u = [0.1, 0.3]              # limb-darkening coefficients
params.limb_dark = "quadratic"     # or "linear", "uniform"

t = np.linspace(-0.05, 0.05, 1000)
m = metalplanet.TransitModel(params, t)
flux = m.light_curve(params)       # numpy array, batman-style
```

Supported batman-isms: parameter updates between `light_curve` calls,
`supersample_factor`/`exp_time` (same endpoint-inclusive convention),
eccentric orbits (non-iterative Markley solver, differentiable),
`transittype="secondary"` with `params.fp`. Tested against batman itself
across orbital configurations — agreement is at batman's own ~2e-8
accuracy floor (its quadratic model uses Hastings polynomial E/K; the
ALFM19 closed forms here are float64-exact to ~1e-13, verified against a
Limbdark.jl port and direct mpmath integration).

Not (yet) supported: nonlinear/power-2/exponential limb darkening,
`max_err`/`fac` error-tolerance machinery (nothing to tune — the model
is closed-form).

## Why another transit code

1. **Differentiable end to end.** The whole model — Kepler solve, orbit,
   photometric core — is an MLX graph. `mx.grad` works everywhere (no
   NaNs at regime boundaries, by construction), and the photometric core
   ships an analytic custom VJP (the paper's closed-form ∂F/∂r, ∂F/∂z,
   ∂F/∂u) that avoids reverse-mode's memory-bound backward pass.
2. **GPU-batched.** Thousands of parameter vectors × 10⁵ data points in
   float32 on Apple-Silicon GPUs, with a float64 CPU verification path
   from the same code.
3. **Sampler-ready.** `metalplanet.anvil` provides the batched,
   float32-conditioned model contract for the
   [anvil](../anvil) (formerly applemcmc) MCMC engine: epoch-centered
   times, offset parameters, Kipping (2013) (q₁,q₂) limb darkening, and
   a synthetic injection-recovery target. ChEES-HMC runs with zero
   divergences on this model.

## Layout

| module        | role |
|---------------|------|
| `api.py`      | batman-style `TransitParams` / `TransitModel` |
| `ellip.py`    | Bulirsch `cel`, batched, fixed-iteration (the numerical heart) |
| `solution.py` | ALFM19 solution vector s₀, s₁, s₂ — masked regimes, stability switches |
| `flux.py`     | flux assembly (Green's basis, quadratic LD) |
| `vjp.py`      | analytic custom VJP for the photometric core (MLX graph) |
| `metal.py`    | hand-fused Metal kernels (forward + analytic VJP) — the fp32 GPU fast path |
| `kepler.py`   | Markley non-iterative Kepler solver; Cartesian-from-E separation with E-level implicit VJP |
| `orbit.py`    | epoch-centered circular orbit (float32-safe sampling path) |
| `trig.py`     | accurate fp64 sin/cos (MLX's are float32-accurate) |
| `greens.py`   | limb-darkening → Green's-basis transform (any order, host-side) |
| `ld.py`       | Kipping (2013) (q₁,q₂) ↔ (u₁,u₂) |
| `anvil.py`    | anvil/applemcmc integration (lazy import; core stays standalone) |

## Install / test

```bash
pip install -e .                       # needs mlx >= 0.30, numpy
pip install -e ".[test]"               # scipy, batman, mpmath, pytest
python -m pytest tests -m "not slow"   # fast suite (~5 s)
python examples/injection_recovery.py  # full GPU injection-recovery
```

The engine integration (`metalplanet.anvil`) needs `anvil`/`applemcmc`
installed; everything else imports standalone. If you rename or move
either repo, re-run the editable installs (`pip install -e ...`) — the
venv stores absolute paths.

## The batching rule (read this before writing a sampler)

A GPU dispatch has a fixed ~0.2–0.7 ms floor; throughput comes from
total points per call, and (parameter sets) x (points per curve) counts
equally on both axes. Evaluate ALL walkers/chains in ONE call: 10,000
parameter sets x 1,000 points costs 2.9 ms batched vs 1.9 s looped
(650x). anvil does this natively; emcee needs `vectorize=True`; the
batman-style `TransitModel.light_curve` is a one-curve API and must not
be the inner loop of a sampler. Full guidance with code:
[docs/sampler-integration.md](docs/sampler-integration.md).

## Performance (M2 Max, fp32, 1024 chains x 65536 points, compiled)

Median of 7, each configuration in an isolated process
(`examples/bench_vjp.py`):

| path | forward | value+grad |
|---|---:|---:|
| reverse-mode autodiff (MLX graph) | 495 ms | 27252 ms |
| analytic VJP (MLX graph) | 498 ms | 618 ms |
| **fused Metal model kernel** (orbit + photometry) | **17.5 ms** | **52.5 ms** |

The MLX graph is memory-bandwidth-bound (~KB of intermediate traffic per
point); the hand-fused kernels (`metal.py`) keep the whole model — the
epoch-centered orbit AND the ALFM19 photometric core — in registers
(~12 B/pt of traffic): 28x the forward, 519x autodiff's value+grad,
~5.7 G pts/s in population batches (55,000 curves/s at npv x 100k for
any npv 64-4096; no unified-memory cliff). End-to-end: the full-scale
1024-chain ChEES-HMC injection-recovery runs in 143 s at 13.1 ESS/s —
1.8x the stretch move on the same core, zero divergences. Kernel parity
vs the graph core is <= 5e-7 with fp64-oracle adjudication; the whole
oracle battery + kernel failure-mode matrix runs in CI (97 tests).
`core="metal"` is the default in the anvil target (v2 model-level
kernel) and the fp32 frontend (photometric kernel); fp64/CPU paths
silently use the graph core.

## Numerical notes worth knowing

* Regime selection is `mx.where` masks with *both-branch sanitization*:
  beyond clamping sqrt/acos arguments, every masked division's
  denominator is replaced by 1 where inactive — a clamped-tiny
  denominator yields a finite forward value but an overflowing −x/y²
  VJP, and 0 × inf = NaN in the backward pass.
* Two razor-thin Taylor switches (|z−r| < 10 eps, |z+r−1| < √eps) guard
  the closed forms' only degenerate lines; the analytic VJP needs no
  switches at all (the partials contain no Π-integral and are continuous
  there).
* MLX 0.32's fp64 `sin`/`cos` (and `exp`) are only float32-accurate;
  `trig.sincos` provides Cody–Waite-reduced fp64 versions for the
  verification paths. `sqrt`, `arccos`, `arctan2` are true fp64.
* The eccentric separation never computes the true anomaly: z comes
  from the orbital-plane Cartesians X = a(cosE−e), Y = a√(1−e²) sinE
  (rotated, projected). Exactly equivalent to the conic form and
  strictly better conditioned — no near-transit or apastron
  cancellations; fp32-trustworthy at the 1e-6 flux level to e ≤ 0.999.
  See docs/eccentric-kernel-notes.md for the verified backward-pass
  formulas and fused-kernel design findings.
* MLX ops on pure-Python operands mint float32 scalars: when a
  parameter may be a Python float, scalar branches must use `math.*`
  or an fp64 path silently carries fp32-accurate constants.
* Fixed iteration counts everywhere (cel: 10/12 for fp32/fp64;
  Markley Kepler: starter + one fifth-order refinement) — no
  data-dependent control flow, so everything is `mx.compile`-safe and
  batches cleanly.
