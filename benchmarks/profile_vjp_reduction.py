"""Decompose the fused kernel's value+grad: forward kernel, VJP kernel
(which reduces its per-chain gradients in-kernel via simd_sum), and the
tiny mx.sum over the (n, 14, ceil(m/32)) partials -- on a circular and an
eccentric chain set.

History this file measured, kept because the docs cite it. The original
v2 circular kernel wrote seven full (n, m) partial arrays and reduced
them with mx.sum. An external estimate put that at ~45% of the VJP and
2-2.4x available; measured, the reductions were ~15% (5.4 of 35.9 ms at
1024 x 65,536; sums at DRAM-peak 345 GB/s -- the kernel is
compute-bound). Moving the reduction in-kernel gave backward 35.8 -> 30.8
ms (1.16x), value+grad 61.1 -> 55.7 ms, and peak memory 4.56 -> 2.74 GB,
transients 1.88 GB -> 58.7 MB. That in-kernel reduction is now the only
path, so the grid-vs-simd A/B this script once ran no longer exists.
"""
import math
import time

import numpy as np
import mlx.core as mx

from metalplanet.metal import _get_model_kernels, make_model_core_metal, NGRAD
from metalplanet.anchored import pack_orbit_constants

N, M, PREF, REPS = 1024, 65536, 3.456, 12

rng = np.random.default_rng(0)
x2d = mx.array(np.vstack([rng.uniform(-0.2, 0.2, M).astype(np.float32),
                          np.zeros(M, np.float32)]))
o = np.ones(N, np.float32)
t0 = mx.array(0.0 * o); pp = mx.array(0.0 * o)
r = mx.array(0.1 * o); a = mx.array(8.8 * o)
u1 = mx.array(0.4225 * o); u2 = mx.array(0.3077 * o)
ci = mx.array(np.float32(0.3 / 8.8) * o)
ct = mx.ones((N, M), dtype=mx.float32)
kf = _get_model_kernels()["model_fwd"]
kv = _get_model_kernels()["model_vjp"]
COLS = (M + 31) // 32


def timeit(fn, label):
    fn(); fn()
    ts = []
    for _ in range(REPS):
        mx.synchronize(); t = time.perf_counter(); fn(); mx.synchronize()
        ts.append(time.perf_counter() - t)
    med = sorted(ts)[len(ts) // 2]
    print(f"  {label:40s} {med*1e3:8.2f} ms")
    return med


for label, e in (("circular (e = 0)", 0.0), ("eccentric (e = 0.3)", 0.3)):
    orb = pack_orbit_constants(
        mx.array(np.float32(math.sqrt(e) * math.cos(1.1)) * o),
        mx.array(np.float32(math.sqrt(e) * math.sin(1.1)) * o), ci)
    mx.eval(orb)
    args = [x2d, t0, pp, r, a, orb, u1, u2]
    print(f"=== {label}, {N} x {M:,} ===")
    tf = timeit(lambda: mx.eval(kf(inputs=args + [PREF, M], output_shapes=[(N, M)],
                                   output_dtypes=[mx.float32], grid=(M, N, 1),
                                   threadgroup=(256, 1, 1))[0]), "forward kernel")
    part = kv(inputs=args + [ct, PREF, M], output_shapes=[(N, NGRAD, COLS)],
              output_dtypes=[mx.float32], init_value=0.0, grid=(M, N, 1),
              threadgroup=(256, 1, 1))[0]
    mx.eval(part)
    tk = timeit(lambda: mx.eval(kv(inputs=args + [ct, PREF, M],
                                   output_shapes=[(N, NGRAD, COLS)],
                                   output_dtypes=[mx.float32], init_value=0.0,
                                   grid=(M, N, 1), threadgroup=(256, 1, 1))[0]),
                "VJP kernel (in-kernel simd reduction)")
    ts = timeit(lambda: mx.eval(mx.sum(part, axis=2)), "mx.sum over the partials")
    core = make_model_core_metal(PREF)
    f = mx.value_and_grad(lambda *p: mx.sum(core(x2d, *p)), argnums=tuple(range(7)))
    def _vg():
        v, g = f(t0, pp, r, a, orb, u1, u2)
        mx.eval(v, *g)          # value AND grads, or the forward is skipped
    tv = timeit(_vg, "full value_and_grad (forward + VJP)")
    print(f"  partials: {N*NGRAD*COLS*4/1e6:.1f} MB; reduction share of VJP "
          f"{ts/(tk+ts)*100:.1f}%; VJP kernel / forward {tk/tf:.2f}x\n")
