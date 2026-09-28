"""Fit real mission photometry (Kepler/TESS-shaped) with anvil.

The point of this script is that there is almost nothing in it:
``make_transit_target`` owns the conditioning that the float32 sampling
path depends on -- the float64 time reduction, the epoch centering, the
flux offset, the ParamSpec boxes, the report offsets, and (for eccentric
fits) the joint-constraint barrier without which an unphysical geometry
returns a perfectly ordinary finite log-likelihood.

Swap the synthetic block for a real light curve and the rest stands.
"""

import math
import time

import numpy as np
import mlx.core as mx

import metalplanet
from metalplanet.anvil import import_engine, make_transit_target
from metalplanet.ld import u_to_q_np

engine, _ = import_engine()

# ---- stand-in for real photometry: one TESS sector, BTJD time stamps ------
T0, PER, RP, A, INC = 2500.123456, 4.2345, 0.085, 11.3, 88.6
U1, U2, YERR = 0.42, 0.21, 3e-4
rng = np.random.default_rng(7)
t = 2500.0 + np.sort(rng.uniform(0.0, 27.0, 40_000))
_p = metalplanet.TransitParams()
_p.t0, _p.per, _p.rp, _p.a, _p.inc = T0, PER, RP, A, INC
_p.ecc, _p.w, _p.u, _p.limb_dark = 0.0, 90.0, [U1, U2], "quadratic"
truth_flux = metalplanet.TransitModel(_p, t).light_curve(_p)
y = truth_flux + YERR * rng.standard_normal(t.size)
yerr = np.full(t.size, YERR)
# --------------------------------------------------------------------------

# t0/period guesses need only land inside the t0_off / p_off boxes
tt = make_transit_target(t, y, yerr, t0_guess=T0 + 0.004,
                         period_guess=PER - 0.0008)
print(f"{t.size:,} points, BTJD {t.min():.2f}-{t.max():.2f}, "
      f"depth {1 - truth_flux.min():.5f}", flush=True)
print(f"conditioning: t_ref={tt.t_ref}  t0_ref={tt.t0_ref:.6f}  "
      f"period_ref={tt.period_ref}", flush=True)

q1, q2 = u_to_q_np(U1, U2)
v_true = tt.model_params(t0=T0, period=PER, r=RP,
                         b=A * math.cos(math.radians(INC)), a=A,
                         q1=float(q1), q2=float(q2))
u_true = tt.transform.from_model_np(v_true)
print(engine.validate_precision(
    tt.target, mx.array((u_true[None, :]
                         + 1e-3 * rng.standard_normal((32, len(v_true)))
                         ).astype(np.float32))), "\n", flush=True)

N_CHAINS = 1024
u0 = mx.array((u_true[None, :]
               + 1e-3 * rng.standard_normal((N_CHAINS, len(v_true)))
               ).astype(np.float32))
# max_leapfrog: 384 measured best on a well-conditioned circular posterior
# (benchmarks/bench_leapfrog_cap.py); an eccentric fit wants ~24 instead.
kernel = engine.ChEESHMC(tt.target, max_leapfrog=384)
t0 = time.perf_counter()
res = engine.run(kernel, tt.target, u0, n_warmup=300, n_samples=200,
                 seed=1, reanchor_every=100, progress=10)
wall = time.perf_counter() - t0

chain = res.get_chain()
ess = engine.diagnostics.ess_bulk(chain)
print(f"\n== {wall:.1f}s, min ESS {ess.min():.0f} ({ess.min()/wall:.1f} "
      f"ESS/s), max R-hat "
      f"{engine.diagnostics.split_rhat(chain).max():.3f}, divergences "
      f"{res.extras.get('n_divergent', 0)}", flush=True)
phys = tt.transform.to_physical(
    tt.transform.model_np(res.get_chain(flat=True).astype(np.float64)))
truth_phys = tt.transform.to_physical(v_true)
for i, name in enumerate(tt.param_names):
    m, sd = phys[:, i].mean(), phys[:, i].std()
    pull = (m - truth_phys[i]) / max(sd, 1e-300)
    print(f"  {name:>7s} = {m:.10g} +/- {sd:.3g}   "
          f"(truth {truth_phys[i]:.10g}, pull {pull:+.2f})")
