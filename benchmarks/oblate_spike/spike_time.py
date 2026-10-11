"""usage: spike_time.py ROUTE LAW  -> prints ns/point (median of 15)."""
import sys, time, json, numpy as np, mlx.core as mx
sys.path.insert(0, sys.path[0])
from spike_kernel import build
from metalplanet.hybrid import LAWS
from metalplanet.metal_hybrid import flux_dev_metal_hybrid
route, law = sys.argv[1], sys.argv[2]
L = LAWS[law]
n_max = 2 if L.uses_mu4 else 1
rng = np.random.default_rng(0)
n = 1 << 20
r, f = 0.1, 0.1
A = r / np.sqrt(1 - f); B = A * (1 - f)
X = rng.uniform(-(1 + r) * 1.15, (1 + r) * 1.15, n)          # ~half the points in transit
b = rng.uniform(0, 0.8, n); th = rng.uniform(0, np.pi, n)
x0 = X * np.cos(th) + b * np.sin(th); y0 = -X * np.sin(th) + b * np.cos(th)
f32 = lambda v: mx.array(np.asarray(v, dtype=np.float32))
xs, ys, As, Bs = f32(x0), f32(y0), f32(np.full(n, A)), f32(np.full(n, B))
z = f32(np.hypot(x0, y0))
if route.startswith("regime_"):
    # points of one regime only: inside (|c| + A < 1) or partial
    _, reg, var = route.split("_")
    rr = np.hypot(x0, y0)
    sel = (rr + A < 1 - 1e-6) if reg == "inside" else ((rr + A > 1 + 1e-6) & (rr - A < 1 - 1e-6))
    idx = rng.choice(np.where(sel)[0], n, replace=True)
    xs, ys = f32(x0[idx]), f32(y0[idx])
    run = build(L.eps, n_max, inside_fast=(var != "general"), warm=(var == "warm")); fn = lambda: run(xs, ys, As, Bs)
elif route == "spherical":
    fn = lambda: flux_dev_metal_hybrid(z, r, law, None, basis=True)
elif route == "oblate":
    run = build(L.eps, n_max); fn = lambda: run(xs, ys, As, Bs)
elif route == "oblate_fast":
    run = build(L.eps, n_max, fast=True); fn = lambda: run(xs, ys, As, Bs)
elif route.startswith("oblate_tg"):
    run = build(L.eps, n_max, tg=int(route[9:])); fn = lambda: run(xs, ys, As, Bs)
elif route == "oblate_nopoles":
    run = build((), n_max); fn = lambda: run(xs, ys, As, Bs)
elif route == "oblate_it6":
    run = build(L.eps, n_max, maxit=6); fn = lambda: run(xs, ys, As, Bs)
elif route == "oblate_onepole":
    run = build(L.eps[:1], n_max); fn = lambda: run(xs, ys, As, Bs)
elif route.startswith("both"):
    run = build(L.eps, n_max, law_for_sph=law)
    sph = mx.array(np.full(n, 1 if route == "both_sph" else 0, dtype=np.int32))
    fn = lambda: run(xs, ys, As, Bs, sph)
for _ in range(3): mx.eval(fn())
mx.synchronize(); ts = []
for _ in range(15):
    s = time.perf_counter(); mx.eval(fn()); mx.synchronize(); ts.append(time.perf_counter() - s)
print(json.dumps({"ns": sorted(ts)[7] / n * 1e9}))
