"""fp32 gradient conditioning of the (k,h) = (sqrt(e) cos w, sqrt(e) sin w)
parameterization near e -> 0, through the production graph orbit path.
Reference: same graph in fp64 on the CPU stream (Cody-Waite trig)."""
import math, numpy as np, mlx.core as mx
from metalplanet.kepler import separation_keplerian
from metalplanet.trig import sincos

A, INC = 8.8, math.acos(0.3 / 8.8)
phi = np.linspace(-0.12, 0.12, 4001)          # transit vicinity, rad

def loss(k, h, phi_arr):
    e = k * k + h * h
    se = mx.sqrt(mx.maximum(e, 1e-30))
    cw, sw = k / se, h / se                     # 0/0 guard irrelevant at e>0
    w = mx.arctan2(h, k)
    f0 = 0.5 * math.pi - w
    s2, c2 = sincos(0.5 * f0)
    E0 = 2.0 * mx.arctan2(mx.sqrt(1.0 - e) * s2, mx.sqrt(1.0 + e) * c2)
    sE0, _ = sincos(E0)
    M_tra = E0 - e * sE0
    z, _ = separation_keplerian(phi_arr + M_tra, e, A, INC, w)
    return mx.sum(z)                            # stands in for sum(ct*dz)

g = mx.grad(loss, argnums=(0, 1))
print(f"{'e':>8} {'|grad| fp64':>12} {'fp32 rel err':>13}")
for e in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    w = 1.1
    k0, h0 = math.sqrt(e) * math.cos(w), math.sqrt(e) * math.sin(w)
    with mx.stream(mx.cpu):
        g64 = g(mx.array(k0, mx.float64), mx.array(h0, mx.float64),
                mx.array(phi, mx.float64))
        g64 = np.array([float(v) for v in g64])
    with mx.stream(mx.gpu):
        g32 = g(mx.array(k0, mx.float32), mx.array(h0, mx.float32),
                mx.array(phi.astype(np.float32)))
        g32 = np.array([float(v) for v in g32])
    rel = np.abs(g32 - g64).max() / max(np.abs(g64).max(), 1e-30)
    print(f"{e:8.0e} {np.abs(g64).max():12.4e} {rel:13.2e}")
