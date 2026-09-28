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

**Scope**: MetalPlanet is strictly a *forward model* — (parameters,
times) → flux, plus that map's analytic derivatives. It contains no
sampler, likelihood, or fitting machinery and depends only on mlx and
numpy; samplers (anvil, emcee, …) sit on the other side of a one-way
dependency arrow. See [CHANGELOG.md](CHANGELOG.md) for version history
(releases are git-tagged).

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

## Evaluating many parameter sets

`TransitModel.light_curve(params)` is one parameter set at a time, for
batman parity. `TransitModel.light_curves(sets)` is the batched form —
same times, many parameter sets, one dispatch:

```python
m = metalplanet.TransitModel(params, t, dtype=mx.float32)
flux = m.light_curves([p1, p2, p3])        # -> (3, len(t))
# or one TransitParams whose attributes are arrays:
p.rp = np.array([0.09, 0.10, 0.11])
flux = m.light_curves(p)                   # -> (3, len(t))
```

Measured at **112x** a Python loop over `light_curve` (2,000 sets x 301
points, float32 GPU) — a GPU dispatch costs ~0.2-0.7 ms whatever its
size, so looping pays that floor 2,000 times. Mixed circular and
eccentric sets are fine in one batch, and every exposure and
limb-darkening mode works. For fitting with thousands of chains prefer
`metalplanet.anvil`, which owns the likelihood and the float32
conditioning too.

## The batching rule (read this before writing a sampler)

Batch every walker/chain into one model call — a Python loop over
parameter sets pays the GPU dispatch floor per iteration and is three
orders of magnitude slower than the identical work batched. The full
rule, working sampler recipes (anvil, emcee, custom), the priors and
`mx.compile` pitfalls, and reproducible measurements live in
[docs/sampler-integration.md](docs/sampler-integration.md).

## Performance (M2 Max, fp32, 1024 chains x 65536 points, compiled)

Median of 7, each configuration in an isolated process
(`examples/bench_vjp.py`):

| path | forward | value+grad |
|---|---:|---:|
| reverse-mode autodiff (MLX graph) | 495 ms | 27252 ms |
| analytic VJP (MLX graph) | 498 ms | 618 ms |
| **fused Metal model kernel** — circular orbit (e = 0) | **~22 ms** | **~70 ms** |
| **fused Metal model kernel** — eccentric orbit (e = 0.3) | **29.0 ms** | **76.0 ms** |

Since v0.6.0 there is **one** model kernel. A circular orbit is e = 0 on
the transit-anchored eccentric kernel — exact, not approximate (the
anchored form degenerates to the circular one to 7e-16) — and a
per-chain branch skips the Kepler solve there. The chain index is uniform
across a threadgroup, so the branch cannot diverge and costs genuinely
eccentric chains ~1%. The circular figures above are the retired
dedicated kernel's clean measurements (20.6 / 54.9 ms) scaled by the
unified kernel's measured ratios against it, **1.07x forward and 1.27x
value+grad**; re-measure with `benchmarks/bench_ecc_kernel.py` on a quiet
GPU. That 27% on circular gradients is the price of one kernel instead
of two — and of mixed circular/eccentric batches for free. Against the
same model as a compiled MLX graph (567 ms / 68.6 s) the eccentric kernel
is 20x the forward and 900x the gradient.

The MLX graph is memory-bandwidth-bound (~KB of intermediate traffic per
point); the hand-fused kernels (`metal.py`) keep the whole model — the
epoch-centered orbit AND the ALFM19 photometric core — in registers
(~12 B/pt of traffic): 28x the forward, 519x autodiff's value+grad,
~5.5 G pts/s in population batches (55,000 curves/s at npv x 100k for
any npv 64-4096; no unified-memory cliff). End-to-end: the full-scale
1024-chain ChEES-HMC injection-recovery runs in 143 s at 13.1 ESS/s —
1.8x the stretch move on the same core, zero divergences. Kernel parity
vs the graph core is <= 5e-7 with fp64-oracle adjudication; the whole
oracle battery + kernel failure-mode matrix runs in CI (97 tests).
`core="metal"` is the default in the anvil targets and the fp32
frontend, circular and eccentric alike; fp64/CPU paths silently use the
graph core. The eccentric model has its own anvil target
(`make_ecc_transit_flux`, 10 parameters) — see the sampler guide for the
joint-constraint barrier it requires.

## Limb darkening and finite exposures

`limb_dark="polynomial"` takes `u = [u_1 ... u_N]` at **any** order for
I(mu)/I0 = 1 - sum u_n (1-mu)^n, via ALFM19's M_n recursion
(`metalplanet/poly.py`), validated against 40-digit mpmath direct
integration: 1e-15 at N = 8, 2e-13 at N = 16. The coefficients stay
traced, so they can change between calls on a built model and the limb
darkening itself is differentiable. (batman's *non*-polynomial laws —
nonlinear, squareroot, logarithmic — are outside this formulation.)

For finite exposures, `integration="contact"` replaces uniform
supersampling with Gauss-Legendre quadrature on windows **split at the
contact times**, which is ALFM19's recipe. The light curve's derivative
jumps where the planet's limb crosses the star's, so uniform sampling
converges only as O(1/N); splitting removes the kinks and each smooth
piece then converges geometrically (`benchmarks/bench_exposure.py`, a
29-minute exposure on a 0.126 d transit):

| method | evaluations per exposure | max abs error |
|---|---:|---:|
| supersample N=101 | 101 | 1.7e-5 |
| supersample N=10,001 | 10,001 | 1.7e-7 |
| **contact GL n=5** | **25** | **8.9e-8** |
| contact GL n=11 | 55 | 2.0e-9 |

Matching the 25-evaluation result with uniform supersampling would take
N ~ 19,500 — about 780x the model evaluations. `supersample_factor`
remains the default for batman parity.

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
* The eccentric orbit is additionally **transit-anchored**
  (`anchored.py`): solving for δ = E − E₀ about inferior conjunction
  instead of for E about periastron. Algebraically identical (2.5e-15
  in flux), but it never forms ω as an intermediate — only e·cosω and
  e·sinω, taken straight from the (√e cosω, √e sinω) sampling pair as
  k√e and h√e. That matters because ∂ω/∂k = −h/e diverges: in the
  textbook form two O(a/e) terms cancel, and fp32 *gradients* w.r.t.
  the sampling coordinates are 5% wrong at e = 1e-5 and 100% wrong at
  e = 1e-8. Anchored, the same measurement is ~1e-7 flat
  (`benchmarks/v3_kh_grad_conditioning.py`). e = 0 is an interior point
  of that disc, so a sampler really does go there.
* MLX ops on pure-Python operands mint float32 scalars: when a
  parameter may be a Python float, scalar branches must use `math.*`
  or an fp64 path silently carries fp32-accurate constants.
* Fixed iteration counts everywhere (cel: 10/12 for fp32/fp64;
  Markley Kepler: starter + one fifth-order refinement) — no
  data-dependent control flow, so everything is `mx.compile`-safe and
  batches cleanly.
