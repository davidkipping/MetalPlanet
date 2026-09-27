"""E5: end-to-end eccentric ChEES-HMC injection-recovery.

The truth is deliberately LOW eccentricity (e = 0.02). That is where the
(sqrt(e) cos w, sqrt(e) sin w) parameterization earns its keep — and
where the direct orbit formulation's float32 gradients are already
degrading, so it is the honest test of the transit-anchored solve the
v3 kernel is built on (see metalplanet/anchored.py and
benchmarks/v3_kh_grad_conditioning.py).

Success = zero divergences, truth recovered within a few sigma, and a
respectable ESS/s on the 10-parameter problem.
"""

import math
import time

import mlx.core as mx
import numpy as np

from metalplanet.anvil import PARAM_NAMES_ECC, import_engine, make_ecc_target

engine, _ = import_engine()

N_CHAINS = 1024
ECC, OMEGA_DEG = 0.02, 63.0

tt = make_ecc_target(n_data=100_000, seed=42, ecc=ECC, omega_deg=OMEGA_DEG)
u_truth = tt.transform.from_model_np(tt.truth_model)
rng = np.random.default_rng(0)
u0 = mx.array(
    (u_truth + 1e-3 * rng.standard_normal((N_CHAINS, 10))).astype(np.float32))

print(f"eccentric ChEES: e = {ECC}, w = {OMEGA_DEG} deg, "
      f"{N_CHAINS} chains, 10 parameters", flush=True)
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
print(f"\n== eccentric ChEES-HMC: {wall:.1f}s, min ESS {ess.min():.0f} "
      f"({ess.min() / wall:.1f} ESS/s), max R-hat {rhat.max():.4f}, "
      f"divergences {ndiv}", flush=True)

flat = res.get_chain(flat=True).astype(np.float64)
phys = tt.transform.to_physical(tt.transform.model_np(flat))
truth_phys = tt.transform.to_physical(tt.truth_model)
mean, sd = phys.mean(axis=0), phys.std(axis=0)
for i, n in enumerate(PARAM_NAMES_ECC):
    pull = (mean[i] - truth_phys[i]) / max(sd[i], 1e-300)
    print(f"  {n:>7s} = {mean[i]:.10g} +/- {sd[i]:.2g}   "
          f"(truth {truth_phys[i]:.10g}, pull {pull:+.2f})")

# the derived eccentricity posterior, which is what the parameterization
# exists to sample cleanly through e = 0
ks = flat[:, 7]
hs = flat[:, 8]
mk = tt.transform.model_np(flat)
e_post = mk[:, 7] ** 2 + mk[:, 8] ** 2
w_post = np.degrees(np.arctan2(mk[:, 8], mk[:, 7])) % 360.0
print(f"\n  e = {e_post.mean():.4f} +/- {e_post.std():.4f} "
      f"(truth {ECC}), 95% upper {np.percentile(e_post, 95):.4f}")
print(f"  w = {w_post.mean():.1f} +/- {w_post.std():.1f} deg "
      f"(truth {OMEGA_DEG})", flush=True)
