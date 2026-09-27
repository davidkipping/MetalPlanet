"""v3 eccentric kernel vs the v2 circular kernel and the graph path.

Gates (docs/v3eccentrickernel_plan.md): forward <= 1.5x the circular
kernel, value+grad <= 2x. The compiled graph eccentric path (567 ms /
68.6 s at this size, benchmarks/v3_ecc_graph_baseline.py) is far too
slow to make a useful denominator, so the gates are absolute.
"""
import math
import time

import numpy as np
import mlx.core as mx

from metalplanet.metal import (_get_model_kernels, _get_ecc_kernels,
                               make_ecc_core_metal, make_model_core_metal)
from metalplanet.anchored import pack_orbit_constants

N, M, PREF, REPS = 1024, 65536, 3.456, 12
E, W = 0.3, 1.1

rng = np.random.default_rng(0)
dt = rng.uniform(-0.2, 0.2, M).astype(np.float32)
x2d = mx.array(np.vstack([dt, np.zeros(M, np.float32)]))
o = np.ones(N, np.float32)
t0 = mx.array(0.0 * o); pp = mx.array(0.0 * o)
r = mx.array(0.1 * o); b = mx.array(0.3 * o); a = mx.array(8.8 * o)
u1 = mx.array(0.4225 * o); u2 = mx.array(0.3077 * o)
kk = mx.array(np.float32(math.sqrt(E) * math.cos(W)) * o)
hh = mx.array(np.float32(math.sqrt(E) * math.sin(W)) * o)
ci = mx.array(np.float32(0.3 / 8.8) * o)
orb = pack_orbit_constants(kk, hh, ci)
mx.eval(x2d, t0, pp, r, b, a, u1, u2, orb)

kc = _get_model_kernels()["model_fwd"]
ke = _get_ecc_kernels()["ecc_fwd"]


def timeit(fn, label):
    fn(); fn()
    ts = []
    for _ in range(REPS):
        mx.synchronize(); t = time.perf_counter(); fn(); mx.synchronize()
        ts.append(time.perf_counter() - t)
    med = sorted(ts)[len(ts) // 2]
    print(f"{label:40s} {med*1e3:8.2f} ms")
    return med


def circ_fwd():
    mx.eval(kc(inputs=[x2d, t0, pp, r, b, a, u1, u2, PREF, M],
               output_shapes=[(N, M)], output_dtypes=[mx.float32],
               grid=(M, N, 1), threadgroup=(256, 1, 1))[0])


def ecc_fwd():
    mx.eval(ke(inputs=[x2d, t0, pp, r, a, orb, u1, u2, PREF, M],
               output_shapes=[(N, M)], output_dtypes=[mx.float32],
               grid=(M, N, 1), threadgroup=(256, 1, 1))[0])


print(f"=== forward, {N} x {M:,} ===")
tc = timeit(circ_fwd, "v2 circular kernel")
te = timeit(ecc_fwd, "v3 eccentric kernel")
print(f"\neccentric / circular = {te/tc:.2f}x   (gate: <= 1.50x)")
print(f"vs compiled graph eccentric (566.7 ms): {566.7/(te*1e3):.0f}x faster")
print(f"throughput: {N*M/te/1e9:.2f} Gpt/s")


# ---- value + gradient ------------------------------------------------------
from metalplanet.metal import make_model_core_metal  # noqa: E402
from metalplanet.anchored import pack_orbit_constants as _pack  # noqa: E402

ecc_core = make_ecc_core_metal(PREF)
circ_core = make_model_core_metal(PREF, reduce="simd")


def ecc_vg():
    def loss(t0_, pp_, r_, a_, k_, h_, ci_, u1_, u2_):
        return mx.sum(ecc_core(x2d, t0_, pp_, r_, a_,
                               _pack(k_, h_, ci_), u1_, u2_))
    v, g = mx.value_and_grad(loss, argnums=tuple(range(9)))(
        t0, pp, r, a, kk, hh, ci, u1, u2)
    mx.eval(v, *g)


def circ_vg():
    def loss(*p):
        return mx.sum(circ_core(x2d, *p))
    v, g = mx.value_and_grad(loss, argnums=tuple(range(7)))(
        t0, pp, r, b, a, u1, u2)
    mx.eval(v, *g)


print(f"\n=== value + gradient, {N} x {M:,} ===")
tcg = timeit(circ_vg, "v2 circular value+grad")
teg = timeit(ecc_vg, "v3 eccentric value+grad")
print(f"\neccentric / circular = {teg/tcg:.2f}x   (gate: <= 2.00x)")
print(f"vs compiled graph eccentric (68.55 s): {68.55/teg:,.0f}x faster")
mx.clear_cache(); mx.reset_peak_memory(); ecc_vg()
print(f"peak memory, eccentric value+grad: {mx.get_peak_memory()/1e9:.2f} GB")


# ---- batch scaling: does the eccentric kernel keep the flat profile? -------
print(f"\n=== batch scaling (npv x 100,000 points), eccentric forward ===")
M_B = 100_000
dtb = rng.uniform(-0.2, 0.2, M_B).astype(np.float32)
x2db = mx.array(np.vstack([dtb, np.zeros(M_B, np.float32)]))
for npv in (64, 256, 1024, 4096):
    ob = np.ones(npv, np.float32)
    args_b = [mx.array(0.0 * ob), mx.array(0.0 * ob), mx.array(0.1 * ob),
              mx.array(8.8 * ob), mx.array(0.4225 * ob), mx.array(0.3077 * ob)]
    orbb = _pack(mx.array(np.float32(math.sqrt(E) * math.cos(W)) * ob),
                 mx.array(np.float32(math.sqrt(E) * math.sin(W)) * ob),
                 mx.array(np.float32(0.3 / 8.8) * ob))
    mx.eval(orbb, *args_b)

    def one(_a=args_b, _o=orbb, _n=npv):
        mx.eval(ke(inputs=[x2db, _a[0], _a[1], _a[2], _a[3], _o, _a[4],
                           _a[5], PREF, M_B],
                   output_shapes=[(_n, M_B)], output_dtypes=[mx.float32],
                   grid=(M_B, _n, 1), threadgroup=(256, 1, 1))[0])
    t = timeit(one, f"  npv = {npv:5d}")
    print(f"        -> {npv/t:,.0f} curves/s, {npv*M_B/t/1e9:.2f} Gpt/s")
