"""jaxlc moments in fp32 with the eigenvalue solve replaced by Aberth
(circle-limit seeds, 25 iterations, 2 Newton polish) -- the GPU solver."""
import sys, numpy as np
SP = "/Users/dkipping/Storage1/Work/Documents/Transit_Work/CODES/SquishierPlanet"
sys.path.insert(0, SP); sys.path.insert(0, SP + "/tests")
src, out = sys.argv[1], sys.argv[2]
import jax
from squishierplanet import jaxlc
jax.config.update("jax_enable_x64", False)
jaxlc._G_TOL, jaxlc._Z_TOL, jaxlc._PAIR_TOL = 1e-5, 3e-3, 1e-3
import jax.numpy as jnp
_orig = jaxlc._roots

def aberth_roots(S0, S1, S2, level, scale):
    z_eig, ok = _orig(S0, S1, S2, level, scale)          # keeps the circle (S2 ~ 0) branch & mask
    big = jnp.abs(S2) > 1e-14 * scale
    k = [S2 + 0j, S1, (S0 - level) + 0j, jnp.conj(S1), S2 + 0j]
    D = lambda z: (((k[0][:, None] * z + k[1][:, None]) * z + k[2][:, None]) * z + k[3][:, None]) * z + k[4][:, None]
    dD = lambda z: ((4 * k[0][:, None] * z + 3 * k[1][:, None]) * z + 2 * k[2][:, None]) * z + k[3][:, None]
    a2, a1, a0 = k[1], k[2], k[3]
    disc = jnp.sqrt(a1 * a1 - 4 * a2 * a0)
    q = -0.5 * (a1 + jnp.where(jnp.real(jnp.conj(a1) * disc) >= 0, disc, -disc))
    small = -k[0] / jnp.where(jnp.abs(k[3]) > 0, k[3], 1)
    z = jnp.stack([q / a2, a0 / q, small, 1 / jnp.conj(small)], 1)
    for _ in range(25):
        r = D(z) / dD(z)
        s = jnp.stack([sum(1 / (z[:, i] - z[:, j]) for j in range(4) if j != i) for i in range(4)], 1)
        z = z - r / (1 - r * s)
    for _ in range(2):
        z = z - D(z) / dD(z)
    return jnp.where(big[:, None], z, z_eig), ok

jaxlc._roots = aberth_roots
d = np.load(src); C = d["C"]
B, P = jaxlc.moments(*[jnp.asarray(v, dtype=jnp.float32) for v in C[:4]], 2, tuple(d["eps"]))
np.savez(out, B=np.asarray(B, np.float64), P=np.asarray(P, np.float64))
print("finite:", bool(np.isfinite(np.asarray(B)).all() and np.isfinite(np.asarray(P)).all()))
