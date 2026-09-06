"""End-to-end injection-recovery: synthetic Kepler-like light curve
(absolute BJD times ~2.457e6 d), fit with ChEES-HMC and the stretch move.

Acceptance criteria (the ones the trapezoid could not meet):
  * ChEES-HMC runs with ZERO divergences on this smooth model,
  * truth recovered within uncertainties by both samplers,
  * ESS/s comparison — on a smooth model HMC should win per-draw
    efficiency decisively.

Run:  .venv/bin/python metalplanet/examples/injection_recovery.py
"""

import time

import mlx.core as mx
import numpy as np

from metalplanet.anvil import import_engine
applemcmc, _ = import_engine()
diagnostics = applemcmc.diagnostics
ess_bulk, split_rhat, summary = (diagnostics.ess_bulk, diagnostics.split_rhat, diagnostics.summary)
from metalplanet.anvil import PARAM_NAMES, make_target

N_CHAINS = 1024

tt = make_target(n_data=100_000, seed=42)
u_truth = tt.transform.from_model_np(tt.truth_model)
rng = np.random.default_rng(0)
u0 = mx.array(
    (u_truth + 1e-3 * rng.standard_normal((N_CHAINS, 8))).astype(np.float32))

print(applemcmc.validate_precision(tt.target, u0[:32]), "\n", flush=True)

results = {}
for label, kernel, n_warmup, n_samples, thin in (
    ("stretch", applemcmc.EnsembleKernel(tt.target, seed=0), 1500, 400, 2),
    ("ChEES-HMC", applemcmc.ChEESHMC(tt.target, max_leapfrog=128), 300, 150, 1),
):
    t0 = time.perf_counter()
    res = applemcmc.run(kernel, tt.target, u0, n_warmup=n_warmup,
                        n_samples=n_samples, thin=thin, seed=1,
                        reanchor_every=100)
    wall = time.perf_counter() - t0
    chain = res.get_chain()
    ess = ess_bulk(chain)
    ndiv = res.extras.get("n_divergent", 0)
    print(f"== {label}: {wall:.1f}s, min ESS {ess.min():.0f} "
          f"({ess.min() / wall:.1f} ESS/s), max R-hat "
          f"{split_rhat(chain).max():.4f}, divergences {ndiv}", flush=True)
    results[label] = dict(wall=wall, min_ess=float(ess.min()),
                          ess_s=float(ess.min() / wall), ndiv=ndiv)

    flat = res.get_chain(flat=True).astype(np.float64)
    model_draws = tt.transform.model_np(flat)
    phys = tt.transform.to_physical(model_draws)
    truth_phys = tt.transform.to_physical(tt.truth_model)
    mean, sd = phys.mean(axis=0), phys.std(axis=0)
    pulls = (mean - truth_phys) / np.maximum(sd, 1e-300)
    for i, n in enumerate(PARAM_NAMES):
        print(f"  {n:>7s} = {mean[i]:.10g} +/- {sd[i]:.2g}   "
              f"(truth {truth_phys[i]:.10g}, pull {pulls[i]:+.2f})")
    print(flush=True)

hmc, st = results["ChEES-HMC"], results["stretch"]
print(f"ChEES-HMC divergences: {hmc['ndiv']} (must be 0)")
print(f"ESS/s — ChEES-HMC: {hmc['ess_s']:.1f}, stretch: {st['ess_s']:.1f}, "
      f"ratio {hmc['ess_s'] / st['ess_s']:.2f}x")
