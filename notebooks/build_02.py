"""Regenerate notebooks/02_joint_transit_and_gp.ipynb.

    python notebooks/build_02.py notebooks/02_joint_transit_and_gp.ipynb

Generated rather than hand-edited so the prose and code stay reviewable in a
diff. Every code cell is executed before the notebook is committed. At the
settings below the joint fit measured 0 divergences, R-hat 1.002 and every one
of the twelve parameters within 1.2 sigma, in about two minutes on an M2 Max.

The authoritative non-tutorial version of this fit, including the
cross-sampler agreement check, is anvil-gp's
examples/hotjupiter_gp_joint.py.
"""
import json, sys

CELLS = []
def md(text):   CELLS.append(("markdown", text.strip("\n")))
def code(text): CELLS.append(("code", text.strip("\n")))

md(r"""
# Joint fitting: a transit *and* the star's variability

Notebook 01 fitted a transit against white noise. Real stars are not that
tidy: spots, granulation and pulsations leave **correlated** signals in the
photometry, often on the same timescale as the transit itself.

This notebook fits both at once — the transit and the noise — using three
packages that compose without any glue code:

| package | role |
|---|---|
| **MetalPlanet** | the transit, as the *mean model* |
| **anvil-gp** | a Gaussian-process likelihood for the correlated residual |
| **anvil** | the sampler that moves all the parameters together |

**Why jointly, and not in two steps?** Detrending first with a filter that
does not know where the transits are will absorb part of the transit and bias
the depth. Fitting the transit afterwards while pretending the noise is white
underestimates every error bar. Sampling them together is the only way the
depth's uncertainty can honestly include *"some of this dip might be
starspots"*.

Expect this notebook to take about two to three minutes, most of it the fit.

> The validation version of this same fit — with a second sampler run and a
> cross-sampler agreement check — is `examples/hotjupiter_gp_joint.py` in the
> anvil-gp repository. This notebook is the annotated walkthrough.
""")

md("""
## 0. Setup

`anvilgp` must be installed alongside `metalplanet` and `anvil`.
""")

code("""
import time

import numpy as np
import mlx.core as mx

import anvil
from anvil.diagnostics import split_rhat
import anvilgp as ag
from anvilgp import GPPolicy
from anvilgp.anvil import make_target
from anvilgp.oracle import sample_ssm
from anvilgp.priors import SpecHint

import metalplanet
from metalplanet.anvil import (DEFAULT_BOUNDS, PARAM_NAMES,
                               make_quad_transit_flux)
from metalplanet.ld import u_to_q_np
from metalplanet.orbit import epoch_center_times

try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False
    print("matplotlib not installed - figures skipped, numbers still print")

print("MetalPlanet", metalplanet.__version__)
print("GP terms available:", [n for n in dir(ag) if n.endswith("Term")
                              and n != "Term"])
""")

md(r"""
## 1. The key idea: the contract already matches

MetalPlanet's engine-facing model is

```python
model_fn(v, x) -> (n_chains, n_data)
```

where `v` is a batch of parameter vectors and `x` is the `(2, n_data)`
epoch-centred time array. anvil-gp's mean-model hook wants

```python
mean_fn(params, x_mean) -> (n_chains, n_data)
```

These are the same signature. So MetalPlanet's model **is** anvil-gp's mean
function — you pass it straight in, with no adapter and no wrapper. The mean
model's parameters take the leading columns of the sampled vector and the GP's
own parameters follow.

That is the whole integration. The rest of this notebook is data and
diagnostics.
""")

md(r"""
## 2. Synthesise a hot Jupiter on an active star

A 3-day hot Jupiter with a ~12,000 ppm transit, on a star with **1000 ppm of
variability correlated on a 2-day timescale** — about 8% of the transit depth,
on a timescale comparable to the transit duration. That is the regime where
two-step detrending goes wrong.

The variability is drawn from a *stochastically-driven harmonic oscillator*
(`SHOTerm`), a standard celerite kernel for stellar signals with a
characteristic timescale. `sample_ssm` draws from that kernel's own
state-space model in O(n) — a dense Cholesky at these sizes would be
prohibitive.

Note the time system: absolute BJD near 2.457e6, which **cannot** be
represented finely enough in float32. MetalPlanet's `epoch_center_times`
reduces it on the host in float64 to (per-orbit residual, orbit number) before
anything reaches the GPU.
""")

code("""
T_REF = 2_457_000.0                       # BJD zero point
BASELINE, N_DATA, YERR = 30.0, 16_000, 500e-6
P_TRUE, T0_TRUE = 3.0, 1.2345             # hot Jupiter
R_TRUE, B_TRUE, A_TRUE = 0.103, 0.5, 8.75
Q1_TRUE, Q2_TRUE = (float(q) for q in u_to_q_np(0.40, 0.25))
SIGMA_TRUE, RHO_TRUE, Q_TRUE = 1000e-6, 2.0, 5.0   # SHO variability
JITTER_TRUE = 200e-6                      # excess white noise

# regular cadence, as a mission light curve has
dt = BASELINE / N_DATA
t_model = np.arange(N_DATA) * dt + 0.5 * dt

# references from an imperfect "preliminary fit", as in a real workflow
t0_ref, period_ref = T0_TRUE - 0.009, P_TRUE + 0.0005
x64 = epoch_center_times(t_model, t0_ref=t0_ref, period_ref=period_ref)

truth_mean = np.array([T0_TRUE - t0_ref, P_TRUE - period_ref, R_TRUE,
                       B_TRUE, A_TRUE, Q1_TRUE, Q2_TRUE, 0.0])
model_fn = make_quad_transit_flux(period_ref=period_ref)
with mx.stream(mx.cpu):                   # float64 reference evaluation
    transit = np.array(
        model_fn(mx.array(truth_mean[None, :], dtype=mx.float64),
                 mx.array(x64, dtype=mx.float64))[0], dtype=np.float64)

rng = np.random.default_rng(7)
variability = sample_ssm(ag.SHOTerm(), [SIGMA_TRUE, RHO_TRUE, Q_TRUE],
                         dt, N_DATA, rng)
white = rng.standard_normal(N_DATA) * np.sqrt(YERR**2 + JITTER_TRUE**2)
y_fit = transit + variability + white     # flux - 1, MetalPlanet's convention

print(f"{N_DATA:,} points over {BASELINE:g} d, cadence {dt*24*60:.2f} min")
print(f"transit depth      {-transit.min()*1e6:6.0f} ppm "
      f"({int(BASELINE/P_TRUE)} transits)")
print(f"stellar variability{variability.std()*1e6:6.0f} ppm  = "
      f"{variability.std()/-transit.min()*100:.0f}% of the depth")
print(f"white noise        {white.std()*1e6:6.0f} ppm")
""")

code("""
if HAVE_PLT:
    fig, ax = plt.subplots(2, 1, figsize=(8, 4.4), sharex=True)
    ax[0].plot(t_model, y_fit + 1.0, lw=0.6, color="0.45")
    ax[0].set_ylabel("flux"); ax[0].set_title(
        "What you actually observe: transits buried in stellar variability")
    ax[1].plot(t_model, variability*1e6, lw=0.8, label="variability (truth)")
    ax[1].plot(t_model, transit*1e6, lw=1.0, label="transit (truth)")
    ax[1].set_xlabel("time [d]"); ax[1].set_ylabel("ppm"); ax[1].legend()
    ax[1].set_xlim(0, 10)
    plt.tight_layout(); plt.show()
print("the variability wanders by more than the transit is deep -- which is "
      "why\\na filter that does not know about the transits will eat part of it")
""")

md(r"""
## 3. Build the joint target

Three arguments do the coupling:

* `mean_fn=model_fn` — MetalPlanet's model, unmodified;
* `x_mean=x64` — the epoch-centred times it expects;
* `mean_specs=[...]` — one `SpecHint` per transit parameter, giving its name
  and bounds. MetalPlanet ships `PARAM_NAMES` and `DEFAULT_BOUNDS` so you do
  not have to invent them.

`make_target` then adds the GP's own parameters, picks sensible priors and
float64 references for them, and returns a target anvil can sample.
`GPPolicy(metal=True)` turns on anvil-gp's fused Metal kernel.
""")

code("""
specs = [SpecHint(name, lo=DEFAULT_BOUNDS[name][0],
                  hi=DEFAULT_BOUNDS[name][1]) for name in PARAM_NAMES]

gt = make_target(
    ag.SHOTerm(),                  # the GP kernel
    t_model, y_fit, np.full(N_DATA, YERR),
    mean_fn=model_fn,              # <- MetalPlanet, straight in
    x_mean=x64,
    mean_specs=specs,
    policy=GPPolicy(metal=True),
)

print(f"{gt.transform.dim} parameters sampled jointly:")
print("  transit (MetalPlanet):", PARAM_NAMES)
print("  GP + noise (anvil-gp):", [n for n in gt.names
                                   if n not in PARAM_NAMES])
ok, why = gt.loglike.metal_eligible()
print(f"\\nanvil-gp fused Metal kernel usable here: {ok}"
      + (f"  ({why})" if why else ""))
""")

md("""
### The truth vector

The transit parameters are in model units already. The GP's are sampled as
logs relative to the float64 references `make_target` chose, so we convert the
true values the same way to compare against them later.
""")

code("""
gp_truth = np.array([SIGMA_TRUE, RHO_TRUE, Q_TRUE, JITTER_TRUE])
truth_model = np.concatenate([
    truth_mean,
    np.log(gp_truth / np.array([h.ref for h in gt.hypers])),
])
u_truth = gt.transform.from_model_np(truth_model)
print("model-space truth vector:", np.round(truth_model, 4))
""")

md(r"""
## 4. Trust check, then sample

The same float32 pre-flight check as notebook 01 — and it matters more here,
because the GP likelihood adds a linear-algebra chain on top of the transit
model.
""")

code("""
init_rng = np.random.default_rng(3)


def ball(n, spread=1e-3):
    \"\"\"Chains in a tight ball around a preliminary fit -- what ChEES wants.\"\"\"
    return mx.array((u_truth + spread * init_rng.standard_normal(
        (n, gt.transform.dim))).astype(np.float32))


print(anvil.validate_precision(gt.target, ball(32)))
""")

md(r"""
Two settings differ from notebook 01, and both are worth understanding:

* **`max_leapfrog=128`, not 384.** The circular transit-only posterior was
  well conditioned and rewarded very long trajectories. Adding GP
  hyperparameters changes the geometry, and 128 is what this problem wants.
  Sampler settings do not transfer between problems — measure on yours.
* **More data helps convergence here.** Shrinking the light curve to make the
  notebook faster *hurt* R-hat, because the GP hyperparameters become poorly
  constrained and their posterior broadens. 16,000 points is the smallest size
  that still samples cleanly.
""")

code("""
t0 = time.perf_counter()
with mx.stream(mx.gpu):
    res = anvil.run(anvil.ChEESHMC(gt.target, max_leapfrog=128), gt.target,
                    ball(192), n_warmup=300, n_samples=300, seed=2,
                    progress=False)
wall = time.perf_counter() - t0

n_div = int(res.extras.get("n_divergent", 0))
rhat = split_rhat(np.asarray(res.get_chain(), dtype=np.float64))
print(f"wall time   : {wall:.0f} s")
print(f"divergences : {n_div}        (want 0)")
print(f"max R-hat   : {rhat.max():.3f}    (want < 1.01)")
print(f"acceptance  : {np.mean(res.accept_fraction):.2f}")
print()
print("healthy" if (n_div == 0 and rhat.max() < 1.01)
      else "NOT converged - see the warmup verdict below")
""")

md(r"""
### Was warmup long enough?

Warmup produces no samples, so it is pure overhead — but cutting it too short
biases everything downstream. Rather than guess, anvil records a cheap trace
during warmup and `warmup_report` reads it back with a verdict: `OK`,
`TOO SHORT`, `LONGER THAN NEEDED`, or `INCONCLUSIVE`.

This is the diagnostic to reach for first when you see divergences or a high
R-hat.
""")

code("""
print(anvil.warmup_report(res))
""")

md("""
## 5. Results

The transit parameters come back in model units; the GP's come back in
physical units through `gt.natural()`, which undoes the log-and-reference
parameterisation for you.
""")

code("""
flat = gt.transform.model_np(res.get_chain(flat=True).astype(np.float64))
natural = gt.natural(flat)

print(f"{'parameter':>12s} {'median':>13s} {'sd':>11s} {'truth':>13s} {'z':>7s}")
print("  -- transit (MetalPlanet) " + "-"*44)
for i, name in enumerate(PARAM_NAMES):
    v = flat[:, i]
    z = (np.median(v) - truth_model[i]) / v.std()
    print(f"{name:>12s} {np.median(v):13.6f} {v.std():11.6f} "
          f"{truth_model[i]:13.6f} {z:+7.2f}")
print("  -- GP and noise (anvil-gp), physical units " + "-"*26)
for name, truth in zip(("gp_sigma", "gp_rho", "gp_Q", "jitter"), gp_truth):
    v = natural[name]
    z = (np.median(v) - truth) / v.std()
    print(f"{name:>12s} {np.median(v):13.6f} {v.std():11.6f} "
          f"{truth:13.6f} {z:+7.2f}")
""")

md(r"""
Two things to notice.

**The GP recovered the variability it was meant to.** `gp_sigma` lands near
1000 ppm and `gp_rho` near 2 days — the injected values — while the transit
parameters stay unbiased. The GP absorbed the correlated signal *without*
eating the transit, which is exactly the point of fitting them together.

**The depth uncertainty is honest.** Because the GP amplitude was free, the
error on `r` includes the possibility that some of the dip was stellar. A
white-noise fit would have quoted a smaller uncertainty and been wrong about
it.
""")

code("""
if HAVE_PLT:
    import math
    best = np.median(flat, axis=0)
    with mx.stream(mx.cpu):
        fit_transit = np.array(model_fn(
            mx.array(best[None, :8], dtype=mx.float64),
            mx.array(x64, dtype=mx.float64))[0], dtype=np.float64)

    # phase-fold on the FITTED ephemeris
    per_fit = period_ref + best[1]
    t0_fit = t0_ref + best[0]
    ph = (t_model - t0_fit + 0.5*per_fit) % per_fit - 0.5*per_fit
    order = np.argsort(ph)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    # left: folded data vs fitted transit -- the variability is the scatter
    ax[0].plot(ph[order]*24, y_fit[order]*1e6, ".", ms=1.0, alpha=0.25,
               color="0.6", label="data (transit + variability)")
    ax[0].plot(ph[order]*24, fit_transit[order]*1e6, lw=2.0,
               label="fitted transit")
    ax[0].set_xlim(-6, 6); ax[0].set_xlabel("hours from mid-transit")
    ax[0].set_ylabel("flux - 1 [ppm]"); ax[0].legend(fontsize=8)
    ax[0].set_title("Folded data and the fitted transit")
    # right: what is left after removing the transit = what the GP models
    resid = y_fit - fit_transit
    ax[1].plot(t_model, resid*1e6, lw=0.7, color="0.45",
               label=f"residual, rms {resid.std()*1e6:.0f} ppm")
    ax[1].plot(t_model, variability*1e6, lw=1.2,
               label="injected variability")
    ax[1].set_xlim(0, 10); ax[1].set_xlabel("time [d]")
    ax[1].set_ylabel("ppm"); ax[1].legend(fontsize=8)
    ax[1].set_title("Residual after the transit: the GP's job")
    plt.tight_layout(); plt.show()

print("The residual is visibly correlated, not white - that structure is "
      "what the\\nGP absorbs, and what a white-noise fit would have "
      "mis-attributed to the planet.")
""")

md(r"""
## 6. Cross-checking a joint fit

A joint fit has more ways to be quietly wrong than a transit-only one, so two
checks are worth the time:

1. **Run a second, gradient-free sampler** and compare. A wrong gradient
   anywhere in the chain — transit model, GP likelihood, or the transform —
   moves the ChEES posterior but not the ensemble's. anvil-gp's
   `examples/hotjupiter_gp_joint.py` does exactly this and reports agreement
   on every median to 0.04 sigma and every width to 2%. It is not repeated
   here only because the ensemble needs thousands of iterations to
   decorrelate, which would dominate this notebook's runtime.
2. **Check the GP did not eat the transit.** If `gp_sigma` drifts up toward
   the transit depth while `r` drifts down, the two are trading. Comparing the
   recovered depth against a white-noise-only fit of the same data is a quick
   sanity check — the depth should agree, with the joint fit's error bar
   larger.

## 7. Pitfalls specific to joint fitting

**Sampler settings do not carry over.** `max_leapfrog=384` was measured best
for the transit-only circular posterior in notebook 01; here 128 is
appropriate. Re-measure per problem — it is two lines of configuration.

**Do not shrink the data to save time.** Counterintuitively, fewer points made
convergence *worse* here: the GP hyperparameters lose their constraint, the
posterior broadens and mixing slows. If a joint fit will not converge, more
data can help more than more iterations.

**A GP is not a licence to skip conditioning.** The absolute BJD times in this
notebook are reduced in float64 by `epoch_center_times` before anything
touches the GPU. Hand raw BJD to a float32 graph and the transit model is
wrong before the GP ever sees it.

**Keep the kernel honest about what it can serve.**
`gt.loglike.metal_eligible()` tells you whether anvil-gp's fused path can take
your term and jitter configuration; a request it cannot honour warns rather
than silently falling back.

## Where to go next

* `notebooks/01_metalplanet_with_anvil.ipynb` — the transit-only fit, and the
  forward-model tour.
* anvil-gp's `examples/hotjupiter_gp_joint.py` — this fit with the
  cross-sampler agreement check, at full size.
* `docs/sampler-integration.md` — batching, conditioning and the eccentric
  constraint barrier in depth.
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
