"""fp32 vs fp64 moments along realistic transit chords, f from 0 to 0.4.
usage: fp32_smallf.py {fp64|fp32} out.npz"""
import sys, numpy as np
SP = "/Users/dkipping/Storage1/Work/Documents/Transit_Work/CODES/SquishierPlanet"
sys.path.insert(0, SP)
mode, out = sys.argv[1], sys.argv[2]
import jax
from squishierplanet import jaxlc
from squishierplanet.laws import HYBRID2_EPS, ladder
if mode == "fp32":
    jax.config.update("jax_enable_x64", False)
    jaxlc._G_TOL, jaxlc._Z_TOL, jaxlc._PAIR_TOL = 1e-5, 3e-3, 1e-3
import jax.numpy as jnp
EPS = tuple(sorted({HYBRID2_EPS, *ladder(2), *ladder(3)}))
rng = np.random.default_rng(7)
rows = []
for r in (0.01, 0.05, 0.1, 0.3):
    for f in (0.0, 1e-7, 1e-5, 1e-3, 1e-2, 0.1, 0.4):
        for _ in range(3):
            th, b = rng.uniform(0, np.pi), rng.uniform(0, 0.9)
            A = r / np.sqrt(1 - f); Bx = A * (1 - f)
            for X in np.concatenate([np.linspace(0, 1 + r + 0.01, 40),
                                     np.sqrt(np.maximum((1 + r) ** 2 - b * b, 0)) + np.array([-1e-4, 1e-5, 0.0]),
                                     np.sqrt(np.maximum((1 - r) ** 2 - b * b, 0)) + np.array([-1e-4, 1e-5, 0.0])]):
                c, s = np.cos(th), np.sin(th)
                rows.append((X * c + b * s, -X * s + b * c, A, Bx, r, f))
C = np.array(rows).T
dt = jnp.float64 if mode == "fp64" else jnp.float32
B, P = jaxlc.moments(*[jnp.asarray(v, dtype=dt) for v in C[:4]], 2, EPS)
np.savez(out, B=np.asarray(B, np.float64), P=np.asarray(P, np.float64), C=C, eps=np.array(EPS))
print(mode, C.shape[1], "finite:", bool(np.isfinite(np.asarray(B)).all() and np.isfinite(np.asarray(P)).all()))
