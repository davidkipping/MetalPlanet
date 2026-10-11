"""Two candidate savings, checked against SquishierPlanet's fp64 reference:
(A) fully-inside regime: even moments = 2 pi q_n0, poles = Re(2 pi i sum of
    INNER residues)/(2p) -- no arcs, no logs;
(B) partial regime: self-inversive symmetry, Re(sum over all 4 roots) ==
    2 Re(sum over the 2 inner roots)  (so only 2 roots need dlog/residues)."""
import sys, numpy as np
SP = "/Users/dkipping/Storage1/Work/Documents/Transit_Work/CODES/SquishierPlanet"
sys.path.insert(0, SP); sys.path.insert(0, SP + "/tests")
import jax.numpy as jnp
from squishierplanet import jaxlc
from squishierplanet.laws import ladder, HYBRID2_EPS
from conftest import make_configs
EPS = tuple(sorted({HYBRID2_EPS, *ladder(2), *ladder(3)}))

def coeffs(x0, y0, a, b):
    S0 = x0*x0 + y0*y0 + 0.5*(a*a + b*b); S1 = a*x0 - 1j*b*y0; S2 = 0.25*(a*a - b*b)
    K0 = a*b; K1 = 0.5*(b*x0 - 1j*a*y0)
    return S0, S1, S2, K0, K1

def pole_inside(x0, y0, a, b, e):
    """full-ellipse arc: 2 pi i * sum of residues at roots inside |z| < 1"""
    S0, S1, S2, K0, K1 = coeffs(x0, y0, a, b); p = 1 + e
    out = np.empty(len(x0))
    for i in range(len(x0)):
        k = [S2[i], S1[i], S0[i] - p, np.conj(S1[i]), S2[i]]
        z = np.roots(k)
        D1 = np.polyval(np.polyder(k), z)
        N = np.conj(K1[i]) + K0[i]*z + K1[i]*z*z
        R = -N / (1j * D1)
        inner = np.abs(z) < 1
        out[i] = np.real(2j*np.pi * R[inner].sum()) / (2*p)
    return out

def even_inside(x0, y0, a, b):
    """DC Fourier coefficient of h_n(s) K on the full ellipse, n = 0,1,2, via
    the reference's own q coefficients (the m = 0 column only)."""
    q = np.asarray(jaxlc._q_coeffs(jnp.asarray(x0), jnp.asarray(y0), jnp.asarray(a), jnp.asarray(b), 2))
    return 2*np.pi*np.real(q[:, :, 0])

rng = np.random.default_rng(5)
cfg = np.array(make_configs(11, "inside", 400)).T
# keep planets: both semi-axes < 1, ellipse strictly inside the unit disc
x0, y0, a, b = cfg
ok = np.hypot(x0, y0) + np.maximum(a, b) < 1 - 1e-9
x0, y0, a, b = x0[ok], y0[ok], a[ok], b[ok]
Bref, Pref = (np.asarray(v) for v in jaxlc.moments(jnp.asarray(x0), jnp.asarray(y0), jnp.asarray(a), jnp.asarray(b), 2, EPS))
Be = even_inside(x0, y0, a, b)
nB = np.array([np.pi, np.pi/2, np.pi/3])
print(f"(A) inside regime, {len(x0)} configs:")
print(f"    even moments via 2 pi q_n0 vs reference: worst {np.abs(Be - Bref).max() / nB.min():.1e} (of unocculted flux)")
for k, e in enumerate(EPS):
    Pi = pole_inside(x0, y0, a, b, e)
    print(f"    pole eps={e:<8.4g} via inner residues vs reference: worst {np.abs(Pi - Pref[:, k]).max() / (np.pi/(e*(1+e))):.1e}")

# (B) symmetry on partial configs, non-pair cases, reconstructing the reference's residue sum
cfg = np.array(make_configs(12, "two_int", 300)).T
x0, y0, a, b = cfg
lo, hi, keep, lo_s, hi_s, keep_s = (np.asarray(v) for v in jaxlc._arcs(jnp.asarray(x0), jnp.asarray(y0), jnp.asarray(a), jnp.asarray(b)))
def dlog(z, lo, hi):
    w1, w2 = np.exp(1j*lo), np.exp(1j*hi)
    dm = np.log(abs(w2 - z)) - np.log(abs(w1 - z))
    if abs(z) < 1: di = (hi - lo) + np.angle(1 - z/w2) - np.angle(1 - z/w1)
    else: di = np.angle(1 - w2/z) - np.angle(1 - w1/z)
    return dm + 1j*di
S0, S1, S2, K0, K1 = coeffs(x0, y0, a, b)
worst = 0.0; n_used = 0
for e in EPS:
    p = 1 + e
    for i in range(len(x0)):
        k = [S2[i], S1[i], S0[i] - p, np.conj(S1[i]), S2[i]]
        z = np.roots(k)
        # skip near-double pairs (the reference pairs them: different formula)
        dmin = min(abs(z[j] - z[l]) for j in range(4) for l in range(j+1, 4))
        if dmin < 0.05: continue
        D1 = np.polyval(np.polyder(k), z); N = np.conj(K1[i]) + K0[i]*z + K1[i]*z*z
        R = -N / (1j*D1)
        contrib = np.array([sum(R[r]*dlog(z[r], lo[i, s], hi[i, s]) for s in range(5) if keep[i, s]) for r in range(4)])
        full = np.real(contrib.sum()); inner = 2*np.real(contrib[np.abs(z) < 1].sum())
        worst = max(worst, abs(full - inner) / max(1.0, abs(full))); n_used += 1
print(f"(B) partial regime, {n_used} (config, pole) cases away from double roots:")
print(f"    Re(sum over 4 roots) vs 2 Re(sum over the 2 inner roots): worst rel diff {worst:.1e}")
