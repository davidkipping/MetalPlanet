"""SquishierPlanet's JAX oblate path: cost (XLA cost analysis + timing) and
light curves at small f, for comparison with MetalPlanet's spherical path."""
import sys, time, json, numpy as np
sys.path.insert(0, "/Users/dkipping/Storage1/Work/Documents/Transit_Work/CODES/SquishierPlanet")
import jax, jax.numpy as jnp
from squishierplanet import laws as L, jaxlc
W = {"hybrid2": [0.3, 0.2], "hybrid4": [0.2, 0.2, 0.1, 0.1], "hybrid5": [0.2, 0.2, 0.1, 0.1, 0.1]}
P, A, R = 3.45, 8.8, 0.1
inc = float(np.arccos(0.3 / A))
out = {}
# --- cost: XLA's own count, per point, for each law (one law per call, as a kernel would)
for law in W:
    laws = [L.hybrid(law, W[law])]
    n = 4096
    t = jnp.linspace(-0.1, 0.1, n)
    fn = jax.jit(lambda t, f, th: jaxlc.generator_lightcurves(t, laws, t0=0.0, period=P, a_rs=A, inc=inc, r_eff=R, f=f, theta=th))
    ca = fn.lower(t, 0.1, 0.5).compile().cost_analysis()
    ca = ca[0] if isinstance(ca, (list, tuple)) else ca
    out[f"cost_{law}"] = {k: float(ca.get(k, 0)) / n for k in ("flops", "transcendentals", "bytes accessed")}
    # timing, f = 0.1 and f = 0 (same code: branch-free)
    for f in (0.1, 0.0):
        y = fn(t, f, 0.5); y.block_until_ready()
        ts = []
        for _ in range(5):
            s = time.perf_counter(); fn(t, f, 0.5).block_until_ready(); ts.append(time.perf_counter() - s)
        out[f"time_{law}_f{f}"] = sorted(ts)[2] / n * 1e9      # ns per point
# --- light curves at small f (hybrid5 and hybrid2), dense grid through ingress/egress
t = np.linspace(-0.11, 0.11, 4001)
X, Y, Z = (np.asarray(v) for v in jaxlc.sky_circular(jnp.asarray(t), 0.0, P, A, inc))
out["z"] = np.sqrt(X * X + Y * Y).tolist()
for law in ("hybrid2", "hybrid5"):
    laws = [L.hybrid(law, W[law])]
    for f in (0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
        F = jaxlc.generator_lightcurves(jnp.asarray(t), laws, t0=0.0, period=P, a_rs=A, inc=inc, r_eff=R, f=f, theta=0.5)
        out[f"F_{law}_{f}"] = np.asarray(F)[:, 0].tolist()
json.dump(out, open(sys.argv[1], "w"))
print({k: v for k, v in out.items() if k.startswith(("cost", "time"))})
