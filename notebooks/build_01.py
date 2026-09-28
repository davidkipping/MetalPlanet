"""Regenerate notebooks/01_metalplanet_with_anvil.ipynb.

    python notebooks/build_01.py notebooks/01_metalplanet_with_anvil.ipynb

The notebook is generated rather than hand-edited so its prose and code stay
reviewable as ordinary source in a diff. Every code cell is executed before
the notebook is committed; the measured outcome of the sampling cell at the
settings below is 0 divergences, R-hat 1.002 and ~1500 effective samples/s in
about 25 s on an M2 Max.
"""
import json, sys

CELLS = []
def md(text):   CELLS.append(("markdown", text.strip("\n")))
def code(text): CELLS.append(("code", text.strip("\n")))

md(r"""
# MetalPlanet + anvil: modelling and fitting transits on Apple Silicon

**MetalPlanet** computes transit light curves: `(parameters, times) -> flux`,
plus that map's analytic derivatives. It is *only* a forward model — no
sampler, no likelihood, no fitting machinery. **anvil** is the MCMC engine.
The dependency arrow points one way: anvil imports MetalPlanet, never the
reverse.

This notebook covers:

1. **Forward modelling** — the batman-style frontend, eccentric orbits, limb
   darkening, finite exposures, and evaluating many parameter sets at once.
2. **Fitting real photometry** — building an anvil target from Kepler/TESS
   data and sampling it with gradient-based HMC.
3. **The three things that bite people**, with the reasons.

Everything here runs in well under a minute on an M2-class machine. Cells are
independent of any plotting library: if `matplotlib` is present you get
figures, otherwise the numbers still print.

> Correlated stellar noise (spot modulation, granulation) wants a Gaussian
> process *around* this mean model. That pairing is a separate tutorial;
> nothing below assumes it.
""")

md("""
## 0. Setup

Two dtype/device regimes matter, and MetalPlanet picks between them for you:

| | where it runs | when to use it |
|---|---|---|
| `float64` (default) | CPU | plotting, single curves, reference checks |
| `float32` | GPU, fused Metal kernel | sampling — thousands of parameter sets |

The float64 path is the accurate one (~1e-13 against an mpmath oracle). The
float32 path is the fast one and is what a sampler uses.
""")

code("""
import math
import time

import numpy as np
import mlx.core as mx

import metalplanet

try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except ImportError:                      # the notebook works either way
    HAVE_PLT = False
    print("matplotlib not installed - figures are skipped, numbers still print")

print("MetalPlanet", metalplanet.__version__)
print("Metal kernel available:", metalplanet.metal_available())
""")

md(r"""
## 1. A first light curve

The frontend mirrors `batman`, so if you know that package this is familiar.
Build a `TransitParams`, hand it and a time array to `TransitModel`, call
`light_curve`.

The parameters: `t0` time of inferior conjunction, `per` period, `rp` planet
radius in stellar radii, `a` semi-major axis in stellar radii, `inc`
inclination in degrees, `ecc`/`w` eccentricity and longitude of periastron,
`u` limb-darkening coefficients, `limb_dark` which law they belong to.
""")

code("""
params = metalplanet.TransitParams()
params.t0 = 0.0             # time of inferior conjunction [d]
params.per = 3.456          # orbital period [d]
params.rp = 0.1             # Rp / R*
params.a = 8.8              # a / R*
params.inc = 87.07          # inclination [deg]
params.ecc = 0.0            # circular for now
params.w = 90.0             # longitude of periastron [deg]
params.u = [0.4, 0.25]      # quadratic limb darkening
params.limb_dark = "quadratic"

t = np.linspace(-0.15, 0.15, 1000)          # days from mid-transit
model = metalplanet.TransitModel(params, t)
flux = model.light_curve(params)

print(f"depth = {1 - flux.min():.6f}   out-of-transit flux = {flux.max():.1f}")
if HAVE_PLT:
    plt.figure(figsize=(7, 3.2))
    plt.plot(t, flux, lw=1.6)
    plt.xlabel("time from mid-transit [d]"); plt.ylabel("relative flux")
    plt.title("Quadratic limb-darkened transit"); plt.tight_layout(); plt.show()
""")

md("""
`TransitModel` precomputes everything that depends on the *time grid*, so you
build it once and vary parameters freely between calls — the batman workflow.
Only the limb-darkening **law** (and, for polynomial laws, its order) is fixed
per model.
""")

code("""
if HAVE_PLT:
    plt.figure(figsize=(7, 3.2))
for rp in (0.06, 0.09, 0.12):
    params.rp = rp                       # same model object, new parameters
    f = model.light_curve(params)
    if HAVE_PLT:
        plt.plot(t, f, lw=1.4, label=f"Rp/R* = {rp}")
    print(f"Rp/R* = {rp}: depth {1 - f.min():.5f}")
params.rp = 0.1                          # restore
if HAVE_PLT:
    plt.legend(); plt.xlabel("time from mid-transit [d]")
    plt.ylabel("relative flux"); plt.tight_layout(); plt.show()
""")

md(r"""
## 2. Eccentric orbits

Set `ecc` and `w` and you are done. Two things worth knowing:

* Eccentricity is handled by a **transit-anchored** solve — Kepler's equation
  is solved about inferior conjunction rather than about periastron. It is
  exactly equivalent, and it is what keeps float32 *gradients* accurate as
  `e -> 0`, which matters when you fit.
* `e = 0` is **exact**, not a limiting case. You never need to nudge it to
  `1e-4` to avoid a singularity — and you should not: that is a real model
  error of ~3e-6 in flux, above `batman`'s own accuracy floor.

Eccentricity changes the transit *duration* (the planet moves faster or slower
at conjunction) and, at fixed `inc`, the impact parameter.
""")

code("""
if HAVE_PLT:
    plt.figure(figsize=(7, 3.2))
for ecc, w in ((0.0, 90.0), (0.4, 90.0), (0.4, 270.0)):
    params.ecc, params.w = ecc, w
    f = model.light_curve(params)
    in_transit = f < 1.0 - 1e-9
    dur = (t[in_transit].max() - t[in_transit].min()) if in_transit.any() else 0.0
    label = f"e = {ecc}, w = {w:.0f} deg"
    print(f"{label:24s} duration = {dur*24:.2f} h")
    if HAVE_PLT:
        plt.plot(t, f, lw=1.4, label=label)
params.ecc, params.w = 0.0, 90.0
if HAVE_PLT:
    plt.legend(); plt.xlabel("time from mid-transit [d]")
    plt.ylabel("relative flux"); plt.tight_layout(); plt.show()
""")

md(r"""
## 3. Limb darkening

Three classic laws plus an arbitrary-order polynomial:

| `limb_dark` | `u` | intensity profile |
|---|---|---|
| `"uniform"` | `[]` | 1 |
| `"linear"` | `[u1]` | 1 - u1 (1 - mu) |
| `"quadratic"` | `[u1, u2]` | 1 - u1 (1-mu) - u2 (1-mu)^2 |
| `"polynomial"` | `[u1, ..., uN]` | 1 - sum_n u_n (1-mu)^n, **any N** |

The polynomial law uses ALFM19's M_n recursion and is accurate to ~1e-15 at
N = 8 against direct numerical integration. Note that `batman`'s
*non*-polynomial laws (`nonlinear`, `squareroot`, `logarithmic`) are outside
this formulation and are not supported — you get a clear error, not a silent
approximation.
""")

code("""
if HAVE_PLT:
    plt.figure(figsize=(7, 3.2))
for law, u in (("uniform", []), ("linear", [0.5]), ("quadratic", [0.4, 0.25]),
               ("polynomial", [0.35, 0.2, 0.1, 0.05])):
    p2 = metalplanet.TransitParams()
    for k in ("t0", "per", "rp", "a", "inc", "ecc", "w"):
        setattr(p2, k, getattr(params, k))
    p2.u, p2.limb_dark = u, law
    # the LAW is fixed per model, so each law needs its own TransitModel
    f = metalplanet.TransitModel(p2, t).light_curve(p2)
    print(f"{law:12s} (N={len(u)}): depth {1 - f.min():.5f}")
    if HAVE_PLT:
        plt.plot(t, f, lw=1.4, label=f"{law} (N={len(u)})")
if HAVE_PLT:
    plt.legend(); plt.xlabel("time from mid-transit [d]")
    plt.ylabel("relative flux"); plt.tight_layout(); plt.show()
""")

md(r"""
## 4. Finite exposures

Real photometry integrates over an exposure. Kepler long cadence is 29.4 min;
a transit ingress is minutes. Ignoring that smears and shallows your model.

Two ways to handle it:

* `supersample_factor=N, exp_time=...` — average N uniform samples. This is
  batman's approach and the default here.
* `integration="contact", exp_time=...` — Gauss-Legendre quadrature on the
  exposure window **split at the contact times**.

The second converges dramatically faster, and the reason is worth
understanding: the light curve's derivative *jumps* where the planet's limb
crosses the star's, so uniform sampling converges only as O(1/N) — ten times
the work buys one digit. Splitting at the contacts makes each piece smooth,
and Gauss-Legendre on a smooth piece converges geometrically.
""")

code("""
EXP = 0.0204                      # Kepler long cadence [d] = 29.4 min
p3 = metalplanet.TransitParams()
for k in ("t0", "per", "rp", "a", "inc", "ecc", "w", "u", "limb_dark"):
    setattr(p3, k, getattr(params, k))
t_lc = np.linspace(-0.15, 0.15, 400)

# a trustworthy reference: a very high-order contact rule
ref = metalplanet.TransitModel(p3, t_lc, exp_time=EXP,
                               integration="contact", n_gl=40).light_curve(p3)

print(f"{'method':<28}{'model evals/exposure':>22}{'max error':>13}")
for n in (7, 101, 1001):
    m = metalplanet.TransitModel(p3, t_lc, exp_time=EXP, supersample_factor=n)
    err = np.abs(m.light_curve(p3) - ref).max()
    print(f"{'supersample N=' + str(n):<28}{n:>22}{err:>13.2e}")
for n_gl in (3, 5, 7):
    m = metalplanet.TransitModel(p3, t_lc, exp_time=EXP,
                                 integration="contact", n_gl=n_gl)
    err = np.abs(m.light_curve(p3) - ref).max()
    # five sub-intervals per window, n_gl nodes each
    print(f"{'contact GL n=' + str(n_gl):<28}{5*n_gl:>22}{err:>13.2e}")
""")

code("""
# what the smearing actually does to the curve
sharp = metalplanet.TransitModel(p3, t_lc).light_curve(p3)
if HAVE_PLT:
    plt.figure(figsize=(7, 3.2))
    plt.plot(t_lc, sharp, lw=1.4, label="instantaneous")
    plt.plot(t_lc, ref, lw=1.8, ls="--", label=f"averaged over {EXP*24*60:.0f} min")
    plt.legend(); plt.xlabel("time from mid-transit [d]"); plt.ylabel("relative flux")
    plt.title("Finite exposure smears ingress and egress")
    plt.tight_layout(); plt.show()
print(f"peak difference: {np.abs(sharp - ref).max():.5f} in flux")
""")

md(r"""
## 5. Many parameter sets at once — the single most important habit

A GPU dispatch costs roughly 0.2-0.7 ms **regardless of its size**. So
evaluating 2,000 parameter sets in a Python loop pays that floor 2,000 times,
while one batched call pays it once.

`light_curve` is deliberately one-set-at-a-time (batman parity).
`light_curves` is the batched form: same time grid, many parameter sets, one
dispatch. It accepts either a list of `TransitParams` or a single one whose
attributes are arrays.

If you are writing your own sampler, **this is the call to use**. Mixing
circular and eccentric sets in one batch is fine.
""")

code("""
rng = np.random.default_rng(0)
n_sets = 2000
sets = []
for _ in range(n_sets):
    q = metalplanet.TransitParams()
    q.t0, q.per, q.a = 0.0, 3.456, 8.8
    q.u, q.limb_dark = [0.4, 0.25], "quadratic"
    q.rp = 0.1 + 0.02 * rng.standard_normal()
    q.inc = 87.0 + 0.5 * rng.standard_normal()
    q.ecc = abs(0.2 * rng.standard_normal())     # a mix of circular and not
    q.w = rng.uniform(0, 360)
    sets.append(q)

# float32 on the GPU is the right regime for bulk evaluation
m32 = metalplanet.TransitModel(sets[0], np.linspace(-0.15, 0.15, 301),
                               dtype=mx.float32)
m32.light_curves(sets[:8])                       # warm up the kernel

t0 = time.perf_counter()
batched = m32.light_curves(sets)
t_batched = time.perf_counter() - t0

m32.light_curve(sets[0])
t0 = time.perf_counter()
for q in sets[:100]:
    m32.light_curve(q)
t_loop = (time.perf_counter() - t0) / 100 * n_sets       # extrapolated

print(f"shape returned     : {batched.shape}")
print(f"one batched call   : {t_batched*1e3:8.1f} ms")
print(f"a loop (estimated) : {t_loop*1e3:8.0f} ms   ->  {t_loop/t_batched:.0f}x slower")
""")

md(r"""
## 6. Fitting real photometry with anvil

Now the point of all this. `metalplanet.anvil.make_transit_target` turns
`(t, y, yerr)` into a ready-to-sample anvil target, and it owns the
conditioning that the float32 sampling path depends on:

* absolute mission times are reduced by a float64 reference and turned into
  (per-orbit residual, orbit number). **This matters:** a raw BJD of 2.457e6
  has a float32 spacing of 0.25 d, and even a TESS BTJD of ~2500 d has a 21 s
  spacing — enough to corrupt a transit model.
* flux is fit as `y - 1`, so the graph carries small numbers.
* your `t0`/`period` guesses become the references the sampler perturbs
  around, and reported values come back on your original time system.

Below we synthesise one TESS-like sector so the notebook is self-contained.
Replace that block with a real light curve and everything after it stands.
""")

code("""
from metalplanet.anvil import import_engine, make_transit_target
from metalplanet.ld import u_to_q_np

engine, _ = import_engine()          # anvil, imported lazily

# ---- stand-in for real data: one TESS sector, BTJD time stamps ------------
T0_TRUE, PER_TRUE, RP_TRUE, A_TRUE, INC_TRUE = 2500.123456, 4.2345, 0.085, 11.3, 88.6
U1_TRUE, U2_TRUE, YERR = 0.42, 0.21, 3e-4
rng = np.random.default_rng(7)
t_obs = 2500.0 + np.sort(rng.uniform(0.0, 27.0, 6_000))
_p = metalplanet.TransitParams()
_p.t0, _p.per, _p.rp, _p.a, _p.inc = T0_TRUE, PER_TRUE, RP_TRUE, A_TRUE, INC_TRUE
_p.ecc, _p.w, _p.u, _p.limb_dark = 0.0, 90.0, [U1_TRUE, U2_TRUE], "quadratic"
truth_curve = metalplanet.TransitModel(_p, t_obs).light_curve(_p)
y_obs = truth_curve + YERR * rng.standard_normal(t_obs.size)
yerr_obs = np.full(t_obs.size, YERR)
# --------------------------------------------------------------------------

# 6,000 points over 27 days ~ TESS at a few-minute cadence, trimmed so the
# notebook samples quickly; real sectors run to tens of thousands
print(f"{t_obs.size:,} points over BTJD {t_obs.min():.2f}-{t_obs.max():.2f}")
print(f"depth {1 - truth_curve.min():.5f}, per-point SNR "
      f"{(1 - truth_curve.min())/YERR:.1f}")
if HAVE_PLT:
    ph = (t_obs - T0_TRUE + 0.5*PER_TRUE) % PER_TRUE - 0.5*PER_TRUE
    plt.figure(figsize=(7, 3.2))
    plt.plot(ph*24, y_obs, ".", ms=1.5, alpha=0.3)
    plt.xlim(-5, 5); plt.xlabel("hours from mid-transit")
    plt.ylabel("relative flux"); plt.title("Phase-folded input data")
    plt.tight_layout(); plt.show()
""")

md("""
### Build the target

Your `t0_guess` and `period_guess` only need to be good enough to land inside
the sampler's boxes — 0.5 d and 0.05 d by default. Here we deliberately pass
slightly wrong values, as a real ephemeris would be.
""")

code("""
target = make_transit_target(
    t_obs, y_obs, yerr_obs,
    t0_guess=T0_TRUE + 0.004,        # a slightly stale ephemeris
    period_guess=PER_TRUE - 0.0008,
)

print("parameters sampled:", target.param_names)
print(f"time reference    : t_ref = {target.t_ref}  (subtracted in float64)")
print(f"epoch-centred x   : shape {target.x64.shape}, "
      f"|residual| <= {np.abs(target.x64[0]).max():.3f} d, "
      f"orbits 0-{int(target.x64[1].max())}")
""")

md(r"""
The eight sampled parameters are:

| name | meaning |
|---|---|
| `t0_off`, `p_off` | offsets from your guesses (so the graph sees small numbers) |
| `r` | Rp / R* |
| `b` | impact parameter |
| `a` | a / R* |
| `q1`, `q2` | limb darkening in Kipping (2013) triangular coordinates — sampling these on the unit square is exactly the physically allowed (u1, u2) region |
| `df0` | baseline flux offset |

`target.model_params(...)` builds the sampler's vector from *physical* values
so you never have to remember which reference is subtracted from what.
""")

code("""
q1_true, q2_true = u_to_q_np(U1_TRUE, U2_TRUE)
v_true = target.model_params(
    t0=T0_TRUE, period=PER_TRUE, r=RP_TRUE,
    b=A_TRUE * math.cos(math.radians(INC_TRUE)),
    a=A_TRUE, q1=float(q1_true), q2=float(q2_true),
)

# sanity: chi^2/n at the truth should be ~1 for correctly-scaled errors
with mx.stream(mx.cpu):
    logL = float(np.array(target.loglike.hi(
        mx.array(v_true[None, :], dtype=mx.float64)))[0])
chi2 = -2.0 * (logL + target.loglike.log_offset_const)
print(f"chi^2 / n at the truth = {chi2 / target.loglike.n_data:.4f}   (expect ~1)")
""")

md(r"""
### Trust check before sampling

The sampler evaluates the likelihood in float32. `validate_precision`
compares it against float64 on states like the ones you are about to sample,
and tells you whether the float32 error is small compared with the ~1-unit
scale of a Metropolis accept/reject decision. **Run this before every
production fit**; it is cheap and it catches conditioning problems that would
otherwise show up as mysterious sampling pathology.
""")

code("""
u_true = target.transform.from_model_np(v_true)      # to unconstrained space
u_probe = mx.array((u_true[None, :]
                    + 1e-3 * rng.standard_normal((32, len(v_true)))
                    ).astype(np.float32))
print(engine.validate_precision(target.target, u_probe))
""")

md(r"""
### Sample

MetalPlanet supplies analytic gradients, so gradient-based HMC is the natural
choice — that is the whole reason for the fused kernel. anvil's `ChEESHMC`
adapts the trajectory length across an *ensemble* of chains running
simultaneously on the GPU.

Settings here are sized so the notebook finishes in about half a minute while
still producing a *healthy* fit — the diagnostics below are the real check, not
decoration. For production, use more chains (1024+) and a longer run.

`max_leapfrog=384` was measured best on a well-conditioned circular posterior
like this one; an *eccentric* fit wants something far smaller, around 24 (see
the pitfalls section).

**Warmup and trajectory length are coupled**, which is easy to get wrong: long
trajectories need enough warmup to adapt a step size that can support them.
Building this notebook, 200 warmup iterations at `max_leapfrog=384` produced
**3,120 divergences and R-hat 1.22**; 300 warmup on the same problem produced
**zero divergences and R-hat 1.002**. If you see many divergences, lengthening
warmup is the first thing to try — before you touch the model.
""")

code("""
N_CHAINS = 256                # production: 1024+
u0 = mx.array((u_true[None, :]
               + 1e-3 * rng.standard_normal((N_CHAINS, len(v_true)))
               ).astype(np.float32))

kernel = engine.ChEESHMC(target.target, max_leapfrog=384)
t0 = time.perf_counter()
res = engine.run(kernel, target.target, u0,
                 n_warmup=300, n_samples=200,
                 seed=1, reanchor_every=100, progress=False)
wall = time.perf_counter() - t0

chain = res.get_chain()
ess = engine.diagnostics.ess_bulk(chain)
rhat = engine.diagnostics.split_rhat(chain)
n_div = res.extras.get("n_divergent", 0)
print(f"wall time    : {wall:.1f} s")
print(f"divergences  : {n_div:<8}  (want 0 - a nonzero count means the "
      f"integrator broke, not that the fit is merely imprecise)")
print(f"max R-hat    : {rhat.max():.3f}    (want < 1.01)")
print(f"min ESS      : {ess.min():.0f}  ->  {ess.min()/wall:.0f} effective samples/s")
print()
print("healthy" if (n_div == 0 and rhat.max() < 1.01)
      else "NOT converged - lengthen warmup before trusting the numbers below")
""")

md("""
### Results

`to_physical(model_np(...))` maps the chain back to physical units, with the
time references added in, so `t0` comes out in the BTJD you supplied.
""")

code("""
flat = res.get_chain(flat=True).astype(np.float64)
posterior = target.transform.to_physical(target.transform.model_np(flat))
truth_physical = target.transform.to_physical(v_true)

print(f"{'parameter':>9}  {'posterior':>26}  {'truth':>14}  pull")
for i, name in enumerate(target.param_names):
    mean, sd = posterior[:, i].mean(), posterior[:, i].std()
    pull = (mean - truth_physical[i]) / max(sd, 1e-300)
    print(f"{name:>9}  {mean:>14.8g} +/- {sd:<9.3g}  "
          f"{truth_physical[i]:>14.8g}  {pull:+.2f}")
""")

code("""
if HAVE_PLT:
    # the fitted model over the folded data
    best = posterior.mean(axis=0)
    pf = metalplanet.TransitParams()
    pf.t0, pf.per, pf.rp = best[0], best[1], best[2]
    pf.a = best[4]
    pf.inc = math.degrees(math.acos(best[3] / best[4]))   # b = a cos i
    u1_fit = 2*math.sqrt(best[5])*best[6]
    u2_fit = math.sqrt(best[5])*(1 - 2*best[6])
    pf.ecc, pf.w, pf.u, pf.limb_dark = 0.0, 90.0, [u1_fit, u2_fit], "quadratic"
    t_fine = np.linspace(-0.2, 0.2, 800) + best[0]
    model_fit = metalplanet.TransitModel(pf, t_fine).light_curve(pf)

    ph_d = (t_obs - best[0] + 0.5*best[1]) % best[1] - 0.5*best[1]
    nb = 120
    edges = np.linspace(-0.2, 0.2, nb + 1)
    idx = np.digitize(ph_d, edges) - 1
    keep = (idx >= 0) & (idx < nb)
    binned = np.array([y_obs[keep][idx[keep] == j].mean()
                       if np.any(idx[keep] == j) else np.nan
                       for j in range(nb)])
    centres = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(ph_d*24, y_obs, ".", ms=1.2, alpha=0.2, color="0.6")
    ax[0].plot(centres*24, binned, "o", ms=3.5, label="binned data")
    ax[0].plot((t_fine - best[0])*24, model_fit, lw=1.8, label="posterior mean")
    ax[0].set_ylabel("relative flux"); ax[0].legend(); ax[0].set_xlim(-5, 5)
    ax[0].set_title("Fitted transit")
    resid = np.interp(centres, t_fine - best[0], model_fit)
    ax[1].axhline(0, color="0.7", lw=0.8)
    ax[1].plot(centres*24, (binned - resid)*1e6, "o", ms=3)
    ax[1].set_ylabel("resid [ppm]"); ax[1].set_xlabel("hours from mid-transit")
    plt.tight_layout(); plt.show()
""")

md(r"""
## 7. Fitting an eccentric orbit

Pass `eccentric=True` and you get a ten-parameter model: `secosw` and
`sesinw` — that is `sqrt(e) cos w` and `sqrt(e) sin w` — replace nothing and
are added. Sampling in those coordinates rather than in `(e, w)` is standard
practice: it has no coordinate singularity at `e = 0` and a uniform prior on
the disc gives a uniform prior on `e`.

**One thing you must not skip.** MetalPlanet clamps every numerical hazard, so
a geometry that is *unphysical* — a planet whose periastron is inside the star,
or an impact parameter implying `|cos i| > 1` — still returns a perfectly
ordinary finite log-likelihood. Those constraints couple several parameters,
so no box of per-parameter bounds can express them. `make_transit_target`
attaches a smooth barrier for you when `eccentric=True`. If you assemble a
target by hand and forget it, your chain will happily sample an improper
posterior and nothing will complain.
""")

code("""
from metalplanet.anvil import ecc_constraint_penalty

ecc_target = make_transit_target(t_obs, y_obs, yerr_obs,
                                 t0_guess=T0_TRUE, period_guess=PER_TRUE,
                                 eccentric=True)
print("parameters:", ecc_target.param_names)
print("barrier attached:", type(ecc_target.loglike).__name__)

# what the barrier does: 0 when the geometry is physical, negative when not
v_ok = ecc_target.model_params(
    t0=T0_TRUE, period=PER_TRUE, r=RP_TRUE,
    b=A_TRUE*math.cos(math.radians(INC_TRUE)), a=A_TRUE,
    q1=float(q1_true), q2=float(q2_true), secosw=0.1, sesinw=0.2)
v_bad = v_ok.copy()
v_bad[ecc_target.param_names.index("a")] = 1.05   # periastron inside the star

pen = np.array(ecc_constraint_penalty(
    mx.array(np.stack([v_ok, v_bad]).astype(np.float32))))
print(f"barrier at a physical geometry  : {pen[0]:.1f}")
print(f"barrier with periastron inside  : {pen[1]:.1f}  <- rejected")
""")

md(r"""
## 8. Three things that bite people

**1. Never loop where you can batch.** A GPU dispatch costs the same whether
it carries one curve or four thousand. Section 5 measured a ~100x penalty for
looping. If you write your own sampler, hand the model every parameter set you
have in one call.

**2. Use `float32` only for sampling, and let the library condition your
times.** `make_transit_target` reduces absolute times against a float64
reference. If you build a float32 `TransitModel` directly it does the same
internally — but if you assemble your own graph from raw BJD, float32 cannot
represent those numbers finely enough (0.25 d spacing at 2.457e6) and your
model will be quietly wrong.

**3. Sampler settings do not transfer between problems.** `max_leapfrog=384`
is excellent on a well-conditioned circular posterior and *actively harmful*
on the eccentric one, where transit photometry constrains a combination of
`(a, b, e, w)` through the duration and long trajectories buy nothing while
eventually breaking the integrator. Measure on your problem; it is two lines
of configuration.

## Where to go next

* `docs/sampler-integration.md` — batching, conditioning and the constraint
  barrier in more depth, with reproducible measurements.
* `examples/fit_mission_data.py` — this fit as a standalone script.
* `benchmarks/RESULTS.md` — accuracy and speed against batman, PyTransit,
  exoplanet-core and jaxoplanet.
* Correlated stellar noise: pair this mean model with a GP likelihood and
  sample the hyperparameters jointly. That is its own tutorial.
""")

nb = {"cells": [], "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
for kind, text in CELLS:
    lines = text.split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    cell = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    nb["cells"].append(cell)
with open(sys.argv[1], "w") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")
print(f"wrote {sys.argv[1]}: {len(CELLS)} cells "
      f"({sum(1 for k,_ in CELLS if k=='code')} code)")
