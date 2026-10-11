"""Corrected symmetry test: roots of the self-inversive D come as mirror
pairs (z, 1/conj z) OR lie on |z| = 1. Claim: a mirror pair's two
contributions have equal real parts, so only the inner root of each pair
needs dlog; unit-modulus roots are counted individually. Also: how often
each root type occurs on realistic chords."""
import sys, numpy as np
SP = "/Users/dkipping/Storage1/Work/Documents/Transit_Work/CODES/SquishierPlanet"
sys.path.insert(0, SP); sys.path.insert(0, SP + "/tests")
import jax.numpy as jnp
from squishierplanet import jaxlc
from squishierplanet.laws import ladder, HYBRID2_EPS
from conftest import make_configs
EPS = tuple(sorted({HYBRID2_EPS, *ladder(2), *ladder(3)}))
def coeffs(x0, y0, a, b):
    return (x0*x0 + y0*y0 + 0.5*(a*a + b*b), a*x0 - 1j*b*y0, 0.25*(a*a - b*b), a*b, 0.5*(b*x0 - 1j*a*y0))
def dlog(z, lo, hi):
    w1, w2 = np.exp(1j*lo), np.exp(1j*hi)
    dm = np.log(abs(w2 - z)) - np.log(abs(w1 - z))
    di = ((hi - lo) + np.angle(1 - z/w2) - np.angle(1 - z/w1)) if abs(z) < 1 else (np.angle(1 - w2/z) - np.angle(1 - w1/z))
    return dm + 1j*di
cfg = np.array(make_configs(12, "two_int", 300)).T
x0, y0, a, b = cfg
lo, hi, keep, *_ = (np.asarray(v) for v in jaxlc._arcs(jnp.asarray(x0), jnp.asarray(y0), jnp.asarray(a), jnp.asarray(b)))
S0, S1, S2, K0, K1 = coeffs(x0, y0, a, b)
worst = 0.0; n = 0; counts = {"on-circle": 0, "mirror-pair": 0}
for e in EPS:
    p = 1 + e
    for i in range(len(x0)):
        k = [S2[i], S1[i], S0[i] - p, np.conj(S1[i]), S2[i]]
        z = np.roots(k)
        if min(abs(z[j] - z[l]) for j in range(4) for l in range(j+1, 4)) < 0.05: continue
        D1 = np.polyval(np.polyder(k), z); N = np.conj(K1[i]) + K0[i]*z + K1[i]*z*z
        R = -N / (1j*D1)
        c = np.array([sum(R[r]*dlog(z[r], lo[i, s], hi[i, s]) for s in range(5) if keep[i, s]) for r in range(4)])
        onc = np.abs(np.abs(z) - 1) < 1e-9
        inner = (np.abs(z) < 1) & ~onc
        counts["on-circle"] += int(onc.sum()); counts["mirror-pair"] += int(inner.sum())
        full = np.real(c.sum()); red = 2*np.real(c[inner].sum()) + np.real(c[onc].sum())
        worst = max(worst, abs(full - red) / max(1.0, abs(full))); n += 1
print(f"{n} cases: worst rel diff of the reduced sum {worst:.1e}; root types per quartic: on-circle {counts['on-circle']/n:.2f}, mirror pairs {counts['mirror-pair']/n:.2f}")
# on realistic chords (f >= 1e-5), how many pole quartics have unit-modulus roots, by eps
d = np.load(sys.argv[1]); C = d["C"]; m = C[5] >= 1e-5
x0, y0, a, b = (C[j][m] for j in range(4))
intr = np.hypot(x0, y0) < 1 + np.maximum(a, b)
part = intr & (np.hypot(x0, y0) + np.maximum(a, b) > 1)
S0, S1, S2, K0, K1 = coeffs(x0[part], y0[part], a[part], b[part])
print(f"realistic chords: in-transit points {intr.sum()}, of which partial {part.sum()} ({part.sum()/intr.sum():.0%}), inside {(intr & ~part).sum()} ({(intr & ~part).sum()/intr.sum():.0%})")
for e in EPS:
    onc = 0
    for i in range(len(S0)):
        z = np.roots([S2[i], S1[i], S0[i] - 1 - e, np.conj(S1[i]), S2[i]])
        onc += int((np.abs(np.abs(z) - 1) < 1e-6).sum() >= 2)
    print(f"   eps={e:<8.4g}: partial-regime quartics with unit-modulus roots {onc/len(S0):.0%}")
