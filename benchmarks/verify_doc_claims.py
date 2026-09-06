"""Reproduce the measured numbers quoted in docs/sampler-integration.md.

Writes doc_claims.json; every prose number in the guide traces here.
Run on an otherwise-idle GPU.
"""

import json
import math
import os
import time

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}

# ---- claim 1: looped vs batched (the batching rule) ----------------------
from metalplanet.metal import flux_dev_metal  # noqa: E402

rng = np.random.default_rng(0)
npv, npt = 10_000, 1_000
z_row = np.abs(rng.uniform(0, 1.3, npt)).astype(np.float32)
r_all = (0.1 * (1 + 0.1 * rng.standard_normal(npv))).astype(np.float32)
u1 = np.full(npv, 0.4, np.float32)
u2 = np.full(npv, 0.25, np.float32)

z1 = mx.array(z_row[None, :])
for _ in range(3):
    mx.eval(flux_dev_metal(z1, mx.array(r_all[:1]), mx.array(u1[:1]),
                           mx.array(u2[:1])))
t0 = time.perf_counter()
for i in range(200):
    mx.eval(flux_dev_metal(z1, mx.array(r_all[i:i + 1]), mx.array(u1[:1]),
                           mx.array(u2[:1])))
t_loop = (time.perf_counter() - t0) / 200 * npv

zb = mx.array(np.broadcast_to(z_row, (npv, npt)).copy())
rb, u1b, u2b = mx.array(r_all), mx.array(u1), mx.array(u2)
for _ in range(3):
    mx.eval(flux_dev_metal(zb, rb, u1b, u2b))
ts = []
for _ in range(7):
    t0 = time.perf_counter()
    mx.eval(flux_dev_metal(zb, rb, u1b, u2b))
    ts.append(time.perf_counter() - t0)
t_batch = float(np.median(ts))
out["batch_vs_loop"] = {
    "npv": npv, "npt": npt,
    "looped_ms": round(t_loop * 1e3, 1),
    "batched_ms": round(t_batch * 1e3, 2),
    "speedup": round(t_loop / t_batch),
}
print("batch_vs_loop:", out["batch_vs_loop"], flush=True)

# ---- claim 2: fp32 raw-time phase-wrap error vs orbit count --------------
# frontend-style raw absolute times through the fp32 circular path,
# compared to the fp64 path, in-transit flux error via |dF/dz| ~ 0.05
from metalplanet import TransitModel, TransitParams  # noqa: E402

per = 10.0
wrap = {}
for n_orbits in (1, 100, 1000):
    p = TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = n_orbits * per, per, 0.1, 12.0, 88.0
    p.ecc, p.w, p.u, p.limb_dark = 0.0, 90.0, [0.4, 0.25], "quadratic"
    t = n_orbits * per + np.linspace(-0.15, 0.15, 4001)
    f32 = TransitModel(p, t, dtype=mx.float32).light_curve(p)
    f64 = TransitModel(p, t).light_curve(p)
    wrap[str(n_orbits)] = float(np.abs(f32 - f64).max())
out["fp32_wrap_flux_error"] = {k: f"{v:.1e}" for k, v in wrap.items()}
print("fp32_wrap_flux_error:", out["fp32_wrap_flux_error"], flush=True)

with open(os.path.join(HERE, "doc_claims.json"), "w") as f:
    json.dump(out, f, indent=1)
print("wrote doc_claims.json")
