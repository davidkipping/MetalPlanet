"""The fused model kernel on circular vs eccentric orbits, and vs the graph.

One kernel serves both: a circular orbit is e = 0, where the transit-
anchored formulation is exact and a per-chain branch skips the Kepler
solve. The compiled graph eccentric path (567 ms / 68.6 s at this size,
benchmarks/v3_ecc_graph_baseline.py) is far too slow to be a useful
denominator, so the numbers here are absolute.
"""
import math
import time

import numpy as np
import mlx.core as mx

from metalplanet.metal import make_model_core_metal
from metalplanet.anchored import pack_orbit_constants

N, M, PREF, REPS = 1024, 65536, 3.456, 12

rng = np.random.default_rng(0)
dt = rng.uniform(-0.2, 0.2, M).astype(np.float32)
x2d = mx.array(np.vstack([dt, np.zeros(M, np.float32)]))
o = np.ones(N, np.float32)
t0 = mx.array(0.0 * o); pp = mx.array(0.0 * o)
r = mx.array(0.1 * o); a = mx.array(8.8 * o)
u1 = mx.array(0.4225 * o); u2 = mx.array(0.3077 * o)
ci = mx.array(np.float32(0.3 / 8.8) * o)


def orbit(e, w=1.1):
    return pack_orbit_constants(
        mx.array(np.float32(math.sqrt(e) * math.cos(w)) * o),
        mx.array(np.float32(math.sqrt(e) * math.sin(w)) * o), ci)


orb_circ, orb_ecc = orbit(0.0), orbit(0.3)
mx.eval(x2d, t0, pp, r, a, u1, u2, orb_circ, orb_ecc)
core = make_model_core_metal(PREF)


def timeit(fn, label):
    fn(); fn()
    ts = []
    for _ in range(REPS):
        mx.synchronize(); t = time.perf_counter(); fn(); mx.synchronize()
        ts.append(time.perf_counter() - t)
    med = sorted(ts)[len(ts) // 2]
    print(f"{label:44s} {med*1e3:8.2f} ms")
    return med


def fwd(orb):
    return lambda: mx.eval(core(x2d, t0, pp, r, a, orb, u1, u2))


def vg(orb):
    def loss(t0_, pp_, r_, a_, orb_, u1_, u2_):
        return mx.sum(core(x2d, t0_, pp_, r_, a_, orb_, u1_, u2_))
    f = mx.value_and_grad(loss, argnums=tuple(range(7)))
    return lambda: mx.eval(*f(t0, pp, r, a, orb, u1, u2)[1])


print(f"=== fused model kernel, {N} x {M:,} ===")
fc = timeit(fwd(orb_circ), "forward, circular (e = 0, fast path)")
fe = timeit(fwd(orb_ecc), "forward, eccentric (e = 0.3)")
gc = timeit(vg(orb_circ), "value+grad, circular")
ge = timeit(vg(orb_ecc), "value+grad, eccentric")
print(f"\neccentric / circular: forward {fe/fc:.2f}x   value+grad {ge/gc:.2f}x")
print(f"vs compiled graph eccentric (566.7 ms / 68.55 s): "
      f"{566.7/(fe*1e3):.0f}x / {68.55/ge:,.0f}x")
print(f"throughput: circular {N*M/fc/1e9:.2f} Gpt/s, eccentric {N*M/fe/1e9:.2f} Gpt/s")
mx.clear_cache(); mx.reset_peak_memory(); vg(orb_ecc)()
print(f"peak memory, eccentric value+grad: {mx.get_peak_memory()/1e9:.2f} GB")

print(f"\n=== batch scaling (npv x 100,000 points), eccentric forward ===")
M_B = 100_000
x2db = mx.array(np.vstack([rng.uniform(-0.2, 0.2, M_B).astype(np.float32),
                           np.zeros(M_B, np.float32)]))
for npv in (64, 256, 1024, 4096):
    ob = np.ones(npv, np.float32)
    args = [mx.array(0.0 * ob), mx.array(0.0 * ob), mx.array(0.1 * ob),
            mx.array(8.8 * ob)]
    orbb = pack_orbit_constants(
        mx.array(np.float32(math.sqrt(0.3) * math.cos(1.1)) * ob),
        mx.array(np.float32(math.sqrt(0.3) * math.sin(1.1)) * ob),
        mx.array(np.float32(0.3 / 8.8) * ob))
    lds = [mx.array(0.4225 * ob), mx.array(0.3077 * ob)]
    mx.eval(orbb, *args, *lds)
    t = timeit(lambda: mx.eval(core(x2db, *args, orbb, *lds)), f"  npv = {npv:5d}")
    print(f"        -> {npv/t:,.0f} curves/s, {npv*M_B/t/1e9:.2f} Gpt/s")
