"""Bounded, instrumented ChEES-HMC injection-recovery (the stretch-move
half already ran; see injection_recovery.py for the full two-sampler
script).

The cap of 24 that this script used to carry dated from when gradients
cost ~1.2 s; they now cost 55-76 ms, and re-measuring
(benchmarks/bench_leapfrog_cap.py) showed the cap was the binding
constraint on this target:

    max_leapfrog   ESS/s   max R-hat   divergences
              24    8.17        1.83             0
              96   10.89        1.13             0
             192   21.18        1.03             0
             384   71.24        1.00             0

i.e. 8.7x the throughput AND convergence from R-hat 1.83 to 1.00, with
no divergences anywhere. 384 is used here; beyond it ESS approaches its
ceiling of n_chains x n_samples, so ESS/s must fall however well the
trajectories decorrelate. NOTE this is target-specific: on the
*eccentric* target longer trajectories are actively harmful (see
examples/chees_ecc.py).

Progress ticks every 10 iterations (rate, ETA, acceptance, divergences)
so the run is never blind.
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

kernel = engine.ChEESHMC(tt.target, max_leapfrog=384)
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
