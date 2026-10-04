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

Measured at **~160x** a Python loop over `light_curve` (2,000 sets x 301
points, float32 GPU; 112x when first measured under load) — a GPU dispatch costs ~0.2-0.7 ms whatever its
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
| **fused Metal model kernel** — circular orbit (e = 0) | **21.8 ms** | **60.2 ms** |
| **fused Metal model kernel** — eccentric orbit (e = 0.3) | **29.4 ms** | **76.2 ms** |

Since v0.6.0 there is **one** model kernel. A circular orbit is e = 0 on
the transit-anchored eccentric kernel — exact, not approximate (the
anchored form degenerates to the circular one to 7e-16) — and a
per-chain branch skips the Kepler solve there. The chain index is uniform
across a threadgroup, so the branch cannot diverge and costs genuinely
eccentric chains 1%. Against the retired dedicated circular kernel
(20.6 / 54.9 ms) that is **1.06x forward and 1.10x value+grad**, measured
in one run on a quiet machine (`benchmarks/ab_retired_kernel.py`) — the
price of one kernel instead of two, and of mixed circular/eccentric
batches for free. "value+grad" here means forward and VJP evaluated
together, as a sampler does; evaluating only the gradients skips the
forward entirely and reads ~1.2x, which is a different (and less
relevant) quantity. Against the
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

## Sampling per-transit times: `flux_dev_from_tau`

The fused model kernel (`make_quad_transit_flux`) derives each point's
phase from a linear ephemeris, so *times are data* there and its VJP
returns zero for them. A sampler fitting **per-transit mid-times** — TTVs,
30-80 of them sampled jointly with the shape parameters — cannot express
its model that way. `flux_dev_from_tau` is the entry point for that case:

```python
from metalplanet import flux_dev_from_tau

# tau: (n_chains, m) time since each point's OWN mid-transit, built
# however the parameterisation requires -- MLX chains the rest
dev = flux_dev_from_tau(tau, period, a, b, r, u1, u2,
                        exp_time=29.4 / 60 / 24,   # Kepler long cadence
                        integration="contact", n_gl=5)
flux = 1.0 + dev
```

Gradients flow in `tau` **and** in all six parameters. The exposure rule
runs per output point in registers, so the sub-exposure axis never becomes
an MLX array — which is the whole point. Expanding it instead multiplies
every forward intermediate and every gradient grid by the number of
sub-exposures. At 512 chains x 5,000 points x 15 sub-exposures
(`benchmarks/bench_tau_kernel.py`, M2 Max), against
`separation_circular` + `flux_dev_metal` + averaging outside the kernel:

| | forward | value+grad | peak memory (value+grad) |
|---|---:|---:|---:|
| expanded sub-exposure axis | 41.0 ms | 99.7 ms | 2776 MB |
| same rule, in-kernel (n_sub=15) | 14.5 ms (**2.8x**) | 28.0 ms (**3.6x**) | 51 MB (**55x**) |
| contact rule, in-kernel (n_gl=5) | 24.4 ms (**1.7x**) | 43.9 ms (**2.3x**) | 51 MB (**55x**) |

The third row is both faster than the route it replaces and ~1,200x more
accurate than it (1.3e-7 against 1.6e-4, vs an fp64 reference). Reaching
1e-6 by supersampling needs n_sub ~ 2,271, which at 512 chains would ask
for ~368 GB; measured where both fit, the contact kernel is 30-57x faster
on 160-517x less memory.

`integration="none"` is `separation_circular` + `flux_dev_metal` with
tau -> z folded in, agreeing with it to the standing 5e-7 kernel-vs-graph
tolerance. fp64, the CPU stream, and machines without a usable Metal
device take an MLX graph path that computes the *same* function —
including freezing the quadrature's split points, so a gradient certified
in fp64 is the gradient that runs in fp32.

### Limb darkening as a linear block: `ld_basis=True`

A quadratic-limb-darkened light curve is linear in the intensity
coefficients. Write I(mu) = c0 + c1 mu + c2 mu^2, so that
c = (1 - u1 - u2, u1 + 2 u2, -u2). Then

    F - 1 = (B @ c) / (N @ c),     N = (pi, 2 pi / 3, pi / 2)

Here `B[..., j]` is the (negative, unnormalised) deficit of a star whose
intensity is mu^j. It depends on the geometry alone. A sampler that
marginalises or Gibbs-samples the limb darkening at each geometry needs
`B` itself, and `ld_basis=True` returns it from **one** kernel launch:

```python
B = flux_dev_from_tau(tau, period, a, b, r, exp_time=29.4 / 60 / 24,
                      integration="contact", n_gl=5, ld_basis=True)
# (n, m, 3), or (m, 3) under the usual squeeze rule; u1/u2 not needed
c = mx.array([1 - u1 - u2, u1 + 2 * u2, -u2])
dev = (B @ c) / (math.pi * (1 - u1 / 3 - u2 / 6))   # == the scalar call
```

The kernel already holds the three Green's-basis deficits before it
collapses them with (u1, u2), and exposure integration is linear, so it
commutes with the basis. The VJP takes an (n, m, 3) cotangent and returns
gradients in `tau`, `period`, `a`, `b` and `r`. Off by default: the scalar
kernels are not edited, and with `ld_basis=False` every output and
gradient is bit-identical to 0.6.1. The fp64 graph path takes the same
keyword and meets the identity to ~1e-17. `flux_dev_metal(z, r,
ld_basis=True)` does the same for the z-input kernel.

Cost at 512 chains x 5,000 points (`benchmarks/bench_ld_basis.py`; each
configuration runs in its own process, quiet machine):

| contact rule, n_gl=5 | scalar | `ld_basis=True` | 3 scalar calls |
|---|---:|---:|---:|
| forward | 14.0 ms | 13.6 ms (**0.97x**) | 41.2 ms |
| value+grad | 29.5 ms | 28.3 ms (**0.96x**) | 87.7 ms |

That is the cost of one scalar call, to within noise, and 3x faster than
forming three vertex laws one call at a time. For the instantaneous rule,
where there is almost no arithmetic per output, the larger store shows:
1.12x forward and 0.99x value+grad.

### Eccentric orbits: `secosw`, `sesinw`

Pass (sqrt(e) cos w, sqrt(e) sin w) and the orbit becomes the
transit-anchored eccentric one (`anchored.py`), the same Kepler solve the
model kernel runs. Omit both, and the call stays circular:

```python
dev = flux_dev_from_tau(tau, period, a, b, r, u1, u2,
                        secosw=k, sesinw=h,           # (n,) or scalars
                        exp_time=29.4 / 60 / 24, integration="contact")
```

- `tau` is the time since inferior conjunction.
- `b` is the impact parameter there, a cos i (1 - e^2) / (1 + e sin w).
  That is the same algebra as anvil's eccentric targets, and it reduces to
  the circular `b` at e = 0.
- Gradients flow in all nine inputs. (k, h) stays well conditioned as
  e -> 0, which is what the anchored form exists for: at exactly e = 0,
  d/dk = d/dh = 0 and every other gradient equals the circular path's.
- `ld_basis=True` works on eccentric orbits too.
- A batch can mix circular (k = h = 0) and eccentric chains.

The contact rule needs the contact times. For an eccentric orbit the
usual linearised ones can be off by minutes (2.3e-3 d on a grazing
e = 0.5 orbit). A split that misses its kink costs accuracy (20x there at
n_gl = 5). It also costs the frozen-split gradient its exactness, because
dF/dtheta then jumps inside a Gauss-Legendre piece. So the linearised
contacts are refined with Newton steps through the anchored solve
(`exposure.contact_offsets_anchored`), which agree with bisected roots to
1e-11 d. `TransitModel(integration="contact")` uses the same exact
contacts. With them, n_gl = 5 is accurate to <= 6.8e-7 against the exact
integral on every orbit tested, the same accuracy the circular rule has.

| 512 x 5,000, contact, n_gl=5 | circular | eccentric, e = 0 | eccentric, e = 0.3 |
|---|---:|---:|---:|
| forward | 14.0 ms | 23.6 ms (1.68x) | 31.8 ms (2.27x) |
| value+grad | 29.5 ms | 48.6 ms (1.65x) | 62.6 ms (2.12x) |

(`benchmarks/bench_tau_ecc.py`.) The eccentric cost is a Kepler solve at
each of the 25 quadrature nodes. Chains with e = 0 skip the solve, but
they still run in the larger kernel, so a purely circular fit should
leave `secosw`/`sesinw` unset.

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
