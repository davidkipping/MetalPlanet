"""Baseline the compiled GRAPH eccentric path at 1024 x 65,536 — the
denominator of the v3 plan's '>= 5x the graph path' gate."""
import math, time, numpy as np, mlx.core as mx
from metalplanet.kepler import separation_keplerian
from metalplanet.flux import flux_dev
from metalplanet.trig import sincos

N, M, REPS = 1024, 65536, 10
rng = np.random.default_rng(0)
phi = mx.array(rng.uniform(-0.2, 0.2, M).astype(np.float32))
def col(v): return mx.array(np.full((N, 1), v, np.float32))
e, w, a, inc = col(0.3), col(1.1), col(8.8), col(math.acos(0.3/8.8))
r, u1, u2 = col(0.1), col(0.4225), col(0.3077)
mx.eval(phi, e, w, a, inc, r, u1, u2)

def model(e, w, a, inc, r, u1, u2):
    f0 = 0.5 * math.pi - w
    s2, c2 = sincos(0.5 * f0)
    E0 = 2.0 * mx.arctan2(mx.sqrt(1.0 - e) * s2, mx.sqrt(1.0 + e) * c2)
    M_tra = E0 - e * sincos(E0)[0]
    z, front = separation_keplerian(phi[None, :] + M_tra, e, a, inc, w)
    return mx.where(front, flux_dev(z, r, u1, u2), 0.0)

fwd = mx.compile(model)
def loss(*p): return mx.sum(model(*p))
vg = mx.compile(mx.value_and_grad(loss, argnums=tuple(range(7))))

def timeit(fn, label):
    for _ in range(2): mx.eval(fn())
    ts = []
    for _ in range(REPS):
        mx.synchronize(); t0 = time.perf_counter(); mx.eval(fn())
        mx.synchronize(); ts.append(time.perf_counter() - t0)
    med = sorted(ts)[len(ts)//2] * 1e3
    print(f"{label:40s} {med:8.2f} ms"); return med

args = (e, w, a, inc, r, u1, u2)
tf = timeit(lambda: fwd(*args), "graph eccentric forward (compiled)")
tg = timeit(lambda: vg(*args), "graph eccentric value+grad (compiled)")
print(f"\nv2 circular kernel for scale: fwd 20.8 ms, value+grad ~52-60 ms")
print(f"5x gate implies eccentric kernel fwd <= {tf/5:.1f} ms")
