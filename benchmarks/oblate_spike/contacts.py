"""Oblate contact times: (1) a fixed-iteration solve bracketed by the
circumscribed (A) and inscribed (B) circles' contacts, vs a dense reference;
(2) what the 5-node contact rule loses with spherical (r_eff) splits."""
import sys, numpy as np
SP = "/Users/dkipping/Storage1/Work/Documents/Transit_Work/CODES/SquishierPlanet"
sys.path.insert(0, SP)
import jax, jax.numpy as jnp
from squishierplanet import jaxlc, laws as L
P, AR = 3.45, 8.8

def centre(phi, b, th):
    X, Y = AR * np.sin(phi), b * np.cos(phi)           # X along the orbit (MetalPlanet's sign)
    return X * np.cos(th) + Y * np.sin(th), -X * np.sin(th) + Y * np.cos(th)

PSI = np.linspace(0, 2 * np.pi, 64, endpoint=False)
def extrema(phi, b, th, A, B, newton=3):
    """min and max over the ellipse boundary of |P|^2: 64 samples + Newton."""
    x0, y0 = centre(phi, b, th)
    A = np.asarray(A)[..., None]; B = np.asarray(B)[..., None]
    s = lambda p: (x0[..., None] + A * np.cos(p)) ** 2 + (y0[..., None] + B * np.sin(p)) ** 2
    def refine(p):
        for _ in range(newton):
            c, sn = np.cos(p), np.sin(p)
            xx, yy = x0[..., None] + A * c, y0[..., None] + B * sn
            d1 = 2 * (-xx * A * sn + yy * B * c)
            d2 = 2 * (A * A * sn * sn - xx * A * c + B * B * c * c - yy * B * sn)
            p = p - np.where(np.abs(d2) > 0, d1 / np.where(np.abs(d2) > 0, d2, 1), 0)
        return p
    v = s(PSI)
    pmin = refine(PSI[np.argmin(v, -1)][..., None]); pmax = refine(PSI[np.argmax(v, -1)][..., None])
    return s(pmin)[..., 0], s(pmax)[..., 0]

def circle_contact(Z, b):
    s2 = (Z * Z - b * b) / (AR * AR - b * b)
    return np.arcsin(np.sqrt(np.clip(s2, 0, 1)))

def contacts(b, th, r, f, n_bisect=40):
    """phases of the egress-side outer and inner contacts (fixed iterations)."""
    A = r / np.sqrt(1 - f); B = A * (1 - f)
    out = []
    for which, lo_r, hi_r in (("outer", 1 + B, 1 + A), ("inner", 1 - A, 1 - B)):
        lo = circle_contact(lo_r, b); hi = circle_contact(hi_r, b)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        g = lambda ph: (extrema(ph, b, th, A, B)[0] - 1) if which == "outer" else (extrema(ph, b, th, A, B)[1] - 1)
        glo = g(lo)
        exists = np.sign(glo) != np.sign(g(hi))
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi); gm = g(mid)
            same = np.sign(gm) == np.sign(glo)
            lo = np.where(same, mid, lo); glo = np.where(same, gm, glo); hi = np.where(same, hi, mid)
        out.append(np.where(exists, 0.5 * (lo + hi), np.nan))
    return out

rng = np.random.default_rng(3)
n = 4000
r = rng.uniform(0.02, 0.3, n); f = rng.choice([1e-4, 0.01, 0.05, 0.1, 0.3, 0.5], n)
th = rng.uniform(0, np.pi, n); b = rng.uniform(0, 1.3, n)
co, ci = contacts(b, th, r, f)
# dense reference: scan the phase finely and polish by 60 bisections (fp64)
def dense(which):
    A = r / np.sqrt(1 - f); B = A * (1 - f)
    grid = np.linspace(0, 0.2, 8001)
    vals = np.stack([(extrema(np.full(n, ph), b, th, A, B, newton=6)[0 if which == "outer" else 1] - 1) for ph in grid], 1)
    k = np.argmax(np.diff(np.sign(vals), axis=1) != 0, axis=1)
    has = np.any(np.diff(np.sign(vals), axis=1) != 0, axis=1)
    lo, hi = grid[k], grid[k + 1]
    sgn = lambda ph: np.sign(extrema(ph, b, th, A, B, newton=6)[0 if which == "outer" else 1] - 1)
    slo = sgn(lo)
    for _ in range(60):
        mid = 0.5 * (lo + hi); same = sgn(mid) == slo
        lo = np.where(same, mid, lo); hi = np.where(same, hi, mid)
    return np.where(has, 0.5 * (lo + hi), np.nan)
ro, ri = dense("outer"), dense("inner")
tscale = P / (2 * np.pi) * 86400
for name, est, ref in (("outer (1st/4th)", co, ro), ("inner (2nd/3rd)", ci, ri)):
    both = np.isfinite(est) & np.isfinite(ref)
    mism = np.isfinite(est) != np.isfinite(ref)
    print(f"  {name}: found {np.isfinite(est).sum()}/{np.isfinite(ref).sum()} (existence mismatches {mism.sum()}),"
          f" worst |error| {np.nanmax(np.abs(est - ref)[both]) * tscale:.2e} s")
# (2) quadrature: 5-node contact rule split at spherical r_eff contacts vs true oblate contacts
EXP = 1800.0 / 86400 * 2 * np.pi / P                   # 30-min exposure in phase
xg5, wg5 = np.polynomial.legendre.leggauss(5); xg64, wg64 = np.polynomial.legendre.leggauss(64)
law = [L.hybrid("hybrid5", [0.2, 0.2, 0.1, 0.1, 0.1])]
def flux(phi, r, f, th, b):
    t = phi * P / (2 * np.pi)
    inc = float(np.arccos(b / AR))
    F = jaxlc.generator_lightcurves(jnp.asarray(t), law, t0=0.0, period=P, a_rs=AR, inc=inc, r_eff=r, f=f, theta=-th)
    return np.asarray(F)[:, 0]
def avg(c0, splits, xg, wg, r, f, th, b):
    lo, hi = c0 - EXP / 2, c0 + EXP / 2
    cuts = [lo] + sorted(s for s in splits if lo < s < hi) + [hi]
    nodes = np.concatenate([0.5 * (q - p) * xg + 0.5 * (q + p) for p, q in zip(cuts[:-1], cuts[1:])])
    w = np.concatenate([0.5 * (q - p) * wg for p, q in zip(cuts[:-1], cuts[1:])])
    return (flux(nodes, r, f, th, b) * w).sum() / EXP
print("  contact-rule error (5 nodes, 30-min exposures) vs exact piecewise; hybrid5, r=0.1:")
for f_, th_, b_ in ((0.1, 0.4, 0.3), (0.3, 1.0, 0.5), (0.5, 0.2, 0.0)):
    rr = np.array([0.1]); ff = np.array([f_]); tt = np.array([th_]); bb = np.array([b_])
    global_r, global_f = r, f
    r, f = rr, ff
    o, i_ = contacts(bb, tt, rr, ff); ro_, ri_ = o[0], i_[0]
    r, f = global_r, global_f
    true = [s for c in (ro_, ri_) if np.isfinite(c) for s in (c, -c)]
    sph = [s for c in (circle_contact(1.1, b_), circle_contact(0.9, b_)) for s in (c, -c)]
    cs = np.linspace(-1.3 * ro_, 1.3 * ro_, 61)
    def fine(c0):            # contact-independent reference: 256 pieces x 16 nodes
        lo, hi = c0 - EXP / 2, c0 + EXP / 2
        edges = np.linspace(lo, hi, 257)
        x16, w16 = np.polynomial.legendre.leggauss(16)
        nodes = np.concatenate([0.5 * (q - p) * x16 + 0.5 * (q + p) for p, q in zip(edges[:-1], edges[1:])])
        w = np.concatenate([0.5 * (q - p) * w16 for p, q in zip(edges[:-1], edges[1:])])
        return (flux(nodes, 0.1, f_, th_, b_) * w).sum() / EXP
    ex = np.array([fine(c) for c in cs])
    e_true = max(abs(avg(c, true, xg5, wg5, 0.1, f_, th_, b_) - x) for c, x in zip(cs, ex))
    e_sph = max(abs(avg(c, sph, xg5, wg5, 0.1, f_, th_, b_) - x) for c, x in zip(cs, ex))
    e_none = max(abs(avg(c, [], xg5, wg5, 0.1, f_, th_, b_) - x) for c, x in zip(cs, ex))
    print(f"    f={f_} theta={th_} b={b_}: split at oblate contacts {e_true:.1e}   at spherical r_eff contacts {e_sph:.1e}   no splits {e_none:.1e}")
