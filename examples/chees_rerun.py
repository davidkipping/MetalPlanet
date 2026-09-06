"""Bounded, instrumented ChEES-HMC injection-recovery (the stretch-move
half already ran; see injection_recovery.py for the full two-sampler
script).

Bounds vs the first attempt: max_leapfrog capped at 24 (the unbounded
adaptation drove trajectories toward 128 x ~1.2 s gradients), schedule
200 warmup + 100 samples, progress ticks every 10 iterations (rate, ETA,
acceptance, divergences) so the run is never blind.
"""

import time

import mlx.core as mx
import numpy as np

from metalplanet.anvil import PARAM_NAMES, import_engine, make_target

engine, _ = import_engine()

N_CHAINS = 1024

tt = make_target(n_data=100_000, seed=42)
u_truth = tt.transform.from_model_np(tt.truth_model)
rng = np.random.default_rng(0)
u0 = mx.array(
    (u_truth + 1e-3 * rng.standard_normal((N_CHAINS, 8))).astype(np.float32))

print(engine.validate_precision(tt.target, u0[:32]), "\n", flush=True)

kernel = engine.ChEESHMC(tt.target, max_leapfrog=24)
t0 = time.perf_counter()
res = engine.run(kernel, tt.target, u0, n_warmup=200, n_samples=100,
                 seed=1, reanchor_every=100, progress=10)
wall = time.perf_counter() - t0

chain = res.get_chain()
ess = engine.diagnostics.ess_bulk(chain)
rhat = engine.diagnostics.split_rhat(chain)
ndiv = res.extras.get("n_divergent", 0)
print(f"\n== ChEES-HMC: {wall:.1f}s, min ESS {ess.min():.0f} "
      f"({ess.min() / wall:.1f} ESS/s), max R-hat {rhat.max():.4f}, "
      f"divergences {ndiv}", flush=True)

flat = res.get_chain(flat=True).astype(np.float64)
phys = tt.transform.to_physical(tt.transform.model_np(flat))
truth_phys = tt.transform.to_physical(tt.truth_model)
mean, sd = phys.mean(axis=0), phys.std(axis=0)
for i, n in enumerate(PARAM_NAMES):
    pull = (mean[i] - truth_phys[i]) / max(sd[i], 1e-300)
    print(f"  {n:>7s} = {mean[i]:.10g} +/- {sd[i]:.2g}   "
          f"(truth {truth_phys[i]:.10g}, pull {pull:+.2f})")

# reference: the stretch move measured 1.0 ESS/s on this problem
print(f"\nESS/s vs stretch (1.0): {ess.min() / wall:.2f}x", flush=True)
