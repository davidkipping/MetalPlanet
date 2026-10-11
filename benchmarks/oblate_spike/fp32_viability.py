"""SquishierPlanet's oblate moments (jaxlc) in fp32 vs fp64, on its own
stress configurations. usage: fp32_viability.py {fp64|fp32} out.npz"""
import sys, numpy as np
SP = "/Users/dkipping/Storage1/Work/Documents/Transit_Work/CODES/SquishierPlanet"
sys.path.insert(0, SP); sys.path.insert(0, SP + "/tests")
mode, out = sys.argv[1], sys.argv[2]
import jax
from squishierplanet import jaxlc                 # enables x64 on import
from squishierplanet.laws import HYBRID2_EPS, ladder
from conftest import CLASSES, make_configs, tangent_config
if mode == "fp32":
    jax.config.update("jax_enable_x64", False)
    # validity tolerances scaled from fp64 rounding to fp32 rounding
    jaxlc._G_TOL = 1e-5          # |g| residual after Newton, relative (fp64: 1e-12)
    jaxlc._Z_TOL = 3e-3          # |z| ~ 1 candidate test (fp64: 1e-4)
    jaxlc._PAIR_TOL = 1e-3       # duplicate-angle merge, rad (fp64: 1e-6)
import jax.numpy as jnp
EPS = tuple(sorted({HYBRID2_EPS, *ladder(2), *ladder(3)}))
cfg, tag = [], []
for k, cls in enumerate(CLASSES):
    for c in make_configs(900 + k, cls, 60):
        cfg.append(c); tag.append(cls)
for gap in (0.0, 1e-9, 1e-7, 1e-5, 1e-3, -1e-7, -1e-5):              # limb tangencies
    for kind in ("external", "internal"):
        for a, b, ph in ((0.12, 0.08, 1.1), (0.05, 0.035, 2.0), (0.3, 0.2, 4.0)):
            cfg.append(tangent_config(a, b, ph, kind, gap)); tag.append("limb_tangent")
for e in EPS:                                                          # pole-circle tangencies
    rho = np.sqrt(1.0 + e)
    for gap in (0.0, 1e-9, 1e-7, 1e-5, 1e-3, -1e-7, -1e-5):
        for a, b, ph in ((0.12, 0.08, 1.1), (0.3, 0.2, 2.5), (0.05, 0.05, 0.3), (0.02, 0.014, 4.0)):
            x0, y0, A, B = tangent_config(a / rho, b / rho, ph, "internal", gap)
            cfg.append((x0 * rho, y0 * rho, A * rho, B * rho)); tag.append(f"pole_tangent_{e:.4g}")
C = np.array(cfg).T
dt = jnp.float64 if mode == "fp64" else jnp.float32
args = [jnp.asarray(v, dtype=dt) for v in C]
B, P = jaxlc.moments(*args, 2, EPS)
np.savez(out, B=np.asarray(B, dtype=np.float64), P=np.asarray(P, dtype=np.float64),
         C=C, tag=np.array(tag), eps=np.array(EPS))
print(mode, "configs", C.shape[1], "B", np.asarray(B).dtype, "finite", np.isfinite(np.asarray(B)).all(), np.isfinite(np.asarray(P)).all())
