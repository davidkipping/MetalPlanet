"""Oblate (elliptical) planets for the hybrid limb-darkening laws -- the
reference MLX graph, fp64 and fp32 (Stage 1 of docs/oblate-plan.md).

A planet of projected area-equivalent radius r and flattening f is an
ellipse with semi-axes A = r / sqrt(1 - f) (long) and B = A (1 - f), so
A B = r^2. Its centre, in the ellipse's principal frame, is (x0, y0)
(``principal_frame`` maps a sky position to it). The occulted flux of a
radial intensity I(rho^2) follows from Green's theorem with the potential
h(s) = (1/2s) int_0^s I, as a contour integral over the ellipse arc inside
the star plus h(1) times the star's limb arc inside the ellipse
(SquishierPlanet, docs/notes; reference ``squishierplanet.jaxlc``).

Generator columns, as for the spherical path (``hybrid.py``), in deviation
form (minus the occulted flux; exactly 0 out of transit):

* even: mu^0, mu^2, mu^4 -- the arc integral of a trigonometric polynomial,
  closed-form Fourier sums;
* each pole (mu^2 + eps)^-2 -- partial fractions over the roots of the
  self-inversive quartic D(z) = S2 z^4 + S1 z^3 + (S0 - p) z^2 + conj(S1) z
  + S2, p = 1 + eps, with logs continued along the arc; a near-double pair
  of roots is integrated as one quadratic factor (sigma, pi).

Three regimes, as masks (docs/oblate-plan.md, "Cost review"):

* outside -- zero;
* inside (the ellipse wholly on the disc) -- the contour is the whole
  ellipse: even moments are 2 pi times a zero-frequency Fourier
  coefficient, each pole is -(pi/p) Re sum of N/D' over the roots inside
  |z| < 1. No intersections, no arcs, no logs;
* partial -- exactly two limb crossings (guaranteed for planets: the
  domain r <= (1 - f)^(3/2) makes A^2/B <= 1), one ellipse arc and one
  star arc.

Below a flattening f_sw (1e-10 fp64, 1e-5 fp32) the spherical closed forms
are used: the quartic's leading coefficient S2 ~ r^2 f / 2 vanishes there
and two roots run to 0 and infinity, and the flattening's effect, ~1e-3 f,
is below rounding.

Gradients by autodiff, structured as the reference's: roots are found
under stop_gradient and given exact implicit derivatives by one live
Newton (isolated root) or Bairstow (pair) step; arc endpoints are frozen
and the endpoint motion's net effect is a gradient-only corner term
h(1) [C(P_lo) - C(P_hi)]. Every branch is evaluated on sanitized inputs in
the lanes it does not own, so no masked lane can poison a gradient.

MLX has no complex128, so complex numbers are pairs of real arrays (``_C``),
and MLX's fp64 sin/cos/exp are float32-accurate, so every e^{i t} goes
through ``trig.sincos``.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from .dtypes import fp64_on_cpu
from .hybrid import _combine, combine_cols, get_law, shape_cols as _sph_cols
from .trig import sincos

__all__ = ["axes", "principal_frame", "shape_cols_oblate", "flux_dev_oblate",
           "generator_cols_oblate", "max_r_eff"]

_PI = math.pi
_TWO_PI = 2.0 * math.pi
_sg = mx.stop_gradient

# per-precision tolerances (fp64 as the reference; fp32 from Stage 0)
_F_SW = {mx.float64: 1e-10, mx.float32: 1e-5}      # spherical below this f
_Z_TOL = {mx.float64: 1e-4, mx.float32: 3e-3}      # |z| ~ 1 crossing candidate
_G_TOL = {mx.float64: 1e-12, mx.float32: 1e-5}     # crossing residual, relative
_PAIR_TOL = {mx.float64: 1e-6, mx.float32: 1e-3}   # tangent crossings, rad
_PAIR_RATIO = 0.1                                  # near-double pole roots
_N_ABERTH = 8
_N_SERIES = 18
_N_MAX = 2                                         # mu^0, mu^2, mu^4
_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


# --------------------------------------------------------------------------
# complex numbers as pairs of real arrays
# --------------------------------------------------------------------------

class _C:
    __slots__ = ("re", "im")

    def __init__(self, re, im=0.0):
        self.re, self.im = re, im

    @staticmethod
    def of(x):
        return x if isinstance(x, _C) else _C(x, 0.0)

    def __add__(self, o):
        o = _C.of(o)
        return _C(self.re + o.re, self.im + o.im)

    __radd__ = __add__

    def __sub__(self, o):
        o = _C.of(o)
        return _C(self.re - o.re, self.im - o.im)

    def __rsub__(self, o):
        return _C.of(o) - self

    def __neg__(self):
        return _C(-self.re, -self.im)

    def __mul__(self, o):
        if not isinstance(o, _C):
            return _C(self.re * o, self.im * o)
        return _C(self.re * o.re - self.im * o.im,
                  self.re * o.im + self.im * o.re)

    __rmul__ = __mul__

    def __truediv__(self, o):
        if not isinstance(o, _C):
            return _C(self.re / o, self.im / o)
        d = o.re * o.re + o.im * o.im
        return _C((self.re * o.re + self.im * o.im) / d,
                  (self.im * o.re - self.re * o.im) / d)

    def __rtruediv__(self, o):
        return _C.of(o) / self

    def conj(self):
        return _C(self.re, -self.im)

    def abs2(self):
        return self.re * self.re + self.im * self.im

    def abs(self):
        return mx.sqrt(self.abs2())

    def arg(self):
        return mx.arctan2(self.im, self.re)

    def sg(self):
        return _C(_sg(self.re), _sg(self.im))


def _cwhere(m, a, b):
    a, b = _C.of(a), _C.of(b)
    return _C(mx.where(m, a.re, b.re), mx.where(m, a.im, b.im))


def _expi(t):
    s, c = sincos(t)
    return _C(c, s)


def _mod2pi(t):
    return t - _TWO_PI * mx.floor(t / _TWO_PI)


def _stack(zs):
    return (mx.stack([z.re for z in zs], -1), mx.stack([z.im for z in zs], -1))


def _take(stacked, idx):
    re, im = stacked
    i = mx.stop_gradient(idx)[..., None]
    return _C(mx.take_along_axis(re, i, axis=-1)[..., 0],
              mx.take_along_axis(im, i, axis=-1)[..., 0])


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def axes(r, f):
    """(A, B): semi-axes of the area-equivalent ellipse, A B = r^2."""
    A = r / mx.sqrt(1.0 - f) if isinstance(f, mx.array) else r / math.sqrt(1.0 - f)
    return A, A * (1.0 - f)


def max_r_eff(f):
    """The largest r_eff for which the planet crosses the limb at most
    twice (A^2 / B <= 1): (1 - f)^(3/2)."""
    return (1.0 - f) ** 1.5


def principal_frame(X, Y, theta):
    """Sky position (X along the orbit, Y across) -> the planet centre in
    the ellipse's principal frame, the long axis at sky angle theta from
    the along-orbit direction: x0 = X cos theta + Y sin theta,
    y0 = -X sin theta + Y cos theta."""
    if isinstance(theta, mx.array):
        s, c = sincos(theta)
    else:
        s, c = np.sin(theta), np.cos(theta)
    return X * c + Y * s, -X * s + Y * c


def _coeffs(x0, y0, a, b):
    """s(phi) = |P(phi)|^2 = S0 + 2 Re(S1 e^{i phi}) + 2 S2 cos(2 phi)
    (S0, S2 real; S1 complex) and the arc-element factor K0 + 2 Re(K1 e^{i phi})."""
    S0 = x0 * x0 + y0 * y0 + 0.5 * (a * a + b * b)
    S1 = _C(a * x0, -(b * y0))
    S2 = 0.25 * (a * a - b * b)
    K0 = a * b
    K1 = _C(0.5 * b * x0, -0.5 * a * y0)
    return S0, S1, S2, K0, K1


def _D(z, S0, S1, S2, p):
    return (((z * S2 + S1) * z + (S0 - p)) * z + S1.conj()) * z + S2


def _dD(z, S0, S1, S2, p):
    return ((z * (4.0 * S2) + S1 * 3.0) * z + 2.0 * (S0 - p)) * z + S1.conj()


# --------------------------------------------------------------------------
# the quartic: Aberth, from the better of two seed sets (stop-gradient)
# --------------------------------------------------------------------------

def _aberth(z, S0, S1, S2, p, iters):
    for _ in range(iters):
        r = [_D(zi, S0, S1, S2, p) / _dD(zi, S0, S1, S2, p) for zi in z]
        new = []
        for i in range(4):
            s = _C(0.0 * S0, 0.0 * S0)
            for j in range(4):
                if j != i:
                    s = s + 1.0 / (z[i] - z[j])
            new.append(z[i] - r[i] / (1.0 - r[i] * s))
        z = new
    for _ in range(2):
        z = [zi - _D(zi, S0, S1, S2, p) / _dD(zi, S0, S1, S2, p) for zi in z]
    return z


def _roots(S0, S1, S2, p):
    """The four roots of D at level p, as _C; all inputs stop-gradient.

    Seeds from our own companion-matrix eigensolver (``_qr_seeds``), then
    a short Aberth iteration and two Newton polishes in the working
    precision. The QR seeds cannot stall the way analytic seeds do at
    large f (two roots near the unit circle), so a few Aberth iterations
    reach rounding. A near-double pair converges only linearly, to
    ~sqrt(eps): the pair formula integrates it through (sigma, pi), which a
    Bairstow step makes exact, never through the two roots separately."""
    return _aberth(_qr_seeds(S0, S1, S2, p), S0, S1, S2, p, _N_ABERTH)


# --------------------------------------------------------------------------
# a 4x4 complex eigensolver: balanced, Wilkinson-shifted Hessenberg QR
# --------------------------------------------------------------------------
#
# Written out in elementwise operations on (re, im) pairs, so that it runs
# per lane on any stream and in either precision, and so that the Stage 3
# Metal kernels can transcribe it line for line (Metal has no eigensolver).
# The companion matrix is already upper Hessenberg. Branch-free: a fixed
# number of sweeps per deflation stage, then the trailing eigenvalue is
# read off and the active block shrinks; the last 2x2 is solved directly.
# Measured on the Stage 1 stress sets, (5, 4) sweeps already match
# (12, 8); the counts below keep a margin. A sweep that has converged is
# harmless to repeat (the shift then equals the eigenvalue).

_QR_SWEEPS = (6, 5)        # sweeps before deflating the 4th, then the 3rd
_QR_BALANCE = 4            # balancing passes


def _csqrt(a):
    """Principal square root, cancellation-free."""
    m = a.abs()
    t = mx.sqrt(mx.maximum(0.5 * (m + mx.abs(a.re)), 0.0))
    ok = t > 0
    q = mx.where(ok, a.im / (2.0 * mx.where(ok, t, 1.0)), 0.0)
    pos = a.re >= 0
    return _C(mx.where(pos, t, mx.abs(q)),
              mx.where(pos, q, mx.where(a.im >= 0, t, -t)))


def _givens(x, y):
    """(c, s), c real, with [[c, s], [-conj(s), c]] @ [x, y] = [r, 0]."""
    xa = x.abs()
    rho = mx.sqrt(xa * xa + y.abs2())
    okx, okr = xa > 0, rho > 0
    xs = mx.where(okx, xa, 1.0)
    ph = _cwhere(okx, _C(x.re / xs, x.im / xs), 1.0)
    rs = mx.where(okr, rho, 1.0)
    c = mx.where(okr, xa / rs, 1.0)
    s = _cwhere(okr, ph * y.conj() * (1.0 / rs), 0.0)
    return c, s


def _wilkinson(a, b, c, d):
    """The eigenvalue of [[a, b], [c, d]] nearer d."""
    h = (a - d) * 0.5
    disc = _csqrt(h * h + b * c)
    den1, den2 = h + disc, h - disc
    den = _cwhere(den1.abs2() >= den2.abs2(), den1, den2)
    ok = den.abs2() > 0
    return d - _cwhere(ok, (b * c) / _cwhere(ok, den, 1.0), 0.0)


def _qr_sweep(H, m):
    """One Wilkinson-shifted QR step on the leading m x m block of H."""
    sig = _wilkinson(H[m - 2][m - 2], H[m - 2][m - 1], H[m - 1][m - 2],
                     H[m - 1][m - 1])
    for i in range(m):
        H[i][i] = H[i][i] - sig
    rots = []
    for k in range(m - 1):
        c, s = _givens(H[k][k], H[k + 1][k])
        rots.append((c, s))
        for j in range(k, m):
            x, y = H[k][j], H[k + 1][j]
            H[k][j] = x * c + s * y
            H[k + 1][j] = y * c - s.conj() * x
    for k, (c, s) in enumerate(rots):
        for i in range(min(k + 2, m - 1) + 1):
            x, y = H[i][k], H[i][k + 1]
            H[i][k] = x * c + y * s.conj()
            H[i][k + 1] = y * c - x * s
    for i in range(m):
        H[i][i] = H[i][i] + sig


def _balance(H):
    """Diagonal similarity equalising off-diagonal row and column norms
    (LAPACK gebal's idea, without the powers of two: Aberth polishes
    against the polynomial itself, so the rounding is immaterial)."""
    n = len(H)
    for _ in range(_QR_BALANCE):
        for i in range(n):
            cn = sum((H[j][i].abs() for j in range(n) if j != i), 0.0)
            rn = sum((H[i][j].abs() for j in range(n) if j != i), 0.0)
            ok = (cn > 0) & (rn > 0)
            f = mx.where(ok, mx.sqrt(mx.where(ok, rn, 1.0)
                                     / mx.where(ok, cn, 1.0)), 1.0)
            for j in range(n):
                if j != i:
                    H[j][i] = H[j][i] * f
                    H[i][j] = H[i][j] * (1.0 / f)


def _qr_seeds(S0, S1, S2, p):
    """The four eigenvalues of D's companion matrix, as _C."""
    zero = 0.0 * S0
    S2s = mx.where(mx.abs(S2) > 0, S2, 1.0)
    inv = 1.0 / S2s
    Z, one = _C(zero, zero), _C(1.0 + zero, zero)
    H = [[-(S1 * inv), _C(-(S0 - p) * inv, zero), -(S1.conj() * inv), -one],
         [one, Z, Z, Z], [Z, one, Z, Z], [Z, Z, one, Z]]
    _balance(H)
    for _ in range(_QR_SWEEPS[0]):
        _qr_sweep(H, 4)
    for _ in range(_QR_SWEEPS[1]):
        _qr_sweep(H, 3)
    a, b, c, d = H[0][0], H[0][1], H[1][0], H[1][1]
    h = (a - d) * 0.5
    disc = _csqrt(h * h + b * c)
    mid = (a + d) * 0.5
    return [mid + disc, mid - disc, H[2][2], H[3][3]]


# --------------------------------------------------------------------------
# regimes and the partial-regime arcs (all stop-gradient)
# --------------------------------------------------------------------------

def _g_dg(phi, x0, y0, a, b):
    s, c = sincos(phi)
    x, y = x0 + a * c, y0 + b * s
    return x * x + y * y - 1.0, 2.0 * (-x * a * s + y * b * c)


def _crossings(x0, y0, a, b, dtype):
    """Limb crossings of the ellipse: (count, phi_a <= phi_b) on the
    ellipse parameter. Candidates are roots of D at level 1 with |z| ~ 1,
    polished by guarded Newton on g(phi) = s(phi) - 1 and accepted on the
    residue; crossings closer than _PAIR_TOL are a tangency and dropped
    together."""
    S0, S1, S2, _, _ = _coeffs(x0, y0, a, b)
    z = _roots(S0, S1, S2, 1.0)
    sc = 1.0 + (mx.abs(x0) + a) ** 2 + (mx.abs(y0) + b) ** 2
    phis, valid = [], []
    for zi in z:
        cand = mx.abs(zi.abs() - 1.0) < _Z_TOL[dtype]
        phi = mx.where(cand, zi.arg(), 0.0)
        g, dg = _g_dg(phi, x0, y0, a, b)
        for _ in range(3):
            ok = mx.abs(dg) > 0
            trial = phi - mx.where(ok, g / mx.where(ok, dg, 1.0), 0.0)
            gt, dgt = _g_dg(trial, x0, y0, a, b)
            better = mx.abs(gt) < mx.abs(g)
            phi = mx.where(better, trial, phi)
            g = mx.where(better, gt, g)
            dg = mx.where(better, dgt, dg)
        phis.append(_mod2pi(phi))
        valid.append(cand & (mx.abs(g) <= _G_TOL[dtype] * sc))
    drop = [mx.zeros_like(v) for v in valid]
    for i in range(4):
        for j in range(i + 1, 4):
            d = mx.abs(_mod2pi(phis[i] - phis[j] + _PI) - _PI)
            close = valid[i] & valid[j] & (d < _PAIR_TOL[dtype])
            drop[i] = drop[i] | close
            drop[j] = drop[j] | close
    valid = [v & ~d for v, d in zip(valid, drop)]
    cnt = sum(v.astype(mx.int32) for v in valid)
    key = mx.sort(mx.stack([mx.where(v, p, mx.inf) for v, p in zip(valid, phis)], -1), axis=-1)
    return cnt, key[..., 0], key[..., 1]


def _arcs(x0, y0, a, b, phi_a, phi_b):
    """(lo, hi) of the ellipse arc inside the star and the length dth of
    the star's limb arc inside the ellipse, for two crossings.

    No decision rests on a point near a crossing, where a graze's sliver
    makes an inside/outside test a coin toss in rounding: the ellipse arc
    is chosen by the midpoint of the LONGER of its two arcs (at least
    pi/2 from both crossings), and the star arc inside the ellipse is the
    shorter one -- always, since the planet lies within its circumscribed
    circle of radius A < 1, whose limb arc is under pi."""
    dphi = phi_b - phi_a                      # in (0, 2 pi)
    long_first = dphi >= _PI                  # [phi_a, phi_b] is the long arc
    mid = mx.where(long_first, 0.5 * (phi_a + phi_b),
                   0.5 * (phi_b + phi_a + _TWO_PI))
    s, c = sincos(mid)
    xm, ym = x0 + a * c, y0 + b * s
    mid_in = xm * xm + ym * ym < 1.0
    first = mx.where(long_first, mid_in, ~mid_in)   # [phi_a, phi_b] inside
    lo = mx.where(first, phi_a, phi_b)
    hi = mx.where(first, phi_b, phi_a + _TWO_PI)
    sa, ca = sincos(phi_a)
    sb, cb = sincos(phi_b)
    ta = _mod2pi(mx.arctan2(y0 + b * sa, x0 + a * ca))
    tb = _mod2pi(mx.arctan2(y0 + b * sb, x0 + a * cb))
    dt = mx.abs(ta - tb)
    dth = mx.minimum(dt, _TWO_PI - dt)
    return lo, hi, dth


def _corner(x0, y0, a, b, lo, hi):
    """Gradient-only corner term C(P_lo) - C(P_hi), value 0: the net effect
    of the frozen endpoints' motion (squishierplanet.jaxlc._corner)."""
    def C(phi):
        s, c = sincos(phi)
        xp, yp = _sg(x0) + _sg(a) * c, _sg(y0) + _sg(b) * s
        x, y = x0 + a * c, y0 + b * s
        return xp * y - yp * x
    v = C(lo) - C(hi)
    return v - _sg(v)


# --------------------------------------------------------------------------
# even moments: Fourier coefficients of h_n(s) K along the ellipse
# --------------------------------------------------------------------------

def _alpha(n_max):
    """h_n(s) = sum_p alpha[n, p] s^p, with 2 d(s h_n)/ds = (1 - s)^n."""
    al = np.zeros((n_max + 1, n_max + 1))
    for n in range(n_max + 1):
        for p in range(n + 1):
            al[n, p] = (-1) ** p * math.comb(n, p) / (2.0 * (p + 1))
    return al


def _q_coeffs(S0, S1, S2, K0, K1, n_max):
    """q[n][m], m = 0..2 n_max + 1: h_n(s) K = Re sum_m q_m e^{i m phi}
    (positive frequencies; squishierplanet.jaxlc._q_coeffs)."""
    H = 2 * n_max + 1
    W = 2 * H + 1
    zero = _C(0.0 * S0, 0.0 * S0)
    T = [zero] * W
    T[H - 1], T[H], T[H + 1] = K1.conj(), _C(K0 + 0.0 * S0, 0.0 * S0), K1
    Sk = [_C(S2 + 0.0 * S0, 0.0 * S0), S1.conj(), _C(S0, 0.0 * S0), S1,
          _C(S2 + 0.0 * S0, 0.0 * S0)]
    al = _alpha(n_max)
    M = 2 * n_max + 2
    q = [[zero] * M for _ in range(n_max + 1)]
    for p in range(n_max + 1):
        if p > 0:
            Tn = [zero] * W
            for i in range(W):
                acc = zero
                for jj, j in enumerate(range(-2, 3)):
                    src = i - j
                    if 0 <= src < W:
                        acc = acc + Sk[jj] * T[src]
                Tn[i] = acc
            T = Tn
        for nn in range(p, n_max + 1):
            for m in range(M):
                q[nn][m] = q[nn][m] + T[H + m] * float(al[nn, p])
    return q


def _even_arc(q, lo, hi):
    """Psi_n(hi) - Psi_n(lo) along one ellipse arc, n = 0..n_max."""
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    out = []
    for qn in q:
        acc = qn[0].re * (hi - lo)
        for m in range(1, len(qn)):
            em = _expi(m * mid)
            sm, _ = sincos(m * half)
            acc = acc + (4.0 / m) * (qn[m] * em).re * sm
        out.append(acc)
    return out


# --------------------------------------------------------------------------
# poles: partial fractions over the roots of D
# --------------------------------------------------------------------------

def _N(z, K0, K1):
    return K1.conj() + z * K0 + K1 * (z * z)


def _live_roots(z0, S0, S1, S2, p):
    """One Newton step from the frozen roots with live coefficients: the
    value is unchanged to rounding, the derivative is the implicit one."""
    out = []
    for zi in z0:
        d = _dD(zi, S0, S1, S2, p)
        ds = _cwhere(d.abs() > 0, d, 1.0)
        out.append(zi - _D(zi, S0, S1, S2, p) / ds)
    return out


def _pair_sum(S0, S1, S2, K0, K1, p, sig0, pi0):
    """sum of N/D' over the two roots of the quadratic factor
    z^2 - sigma z + pi of D: with N = aN z + bN and Q = D/q = aQ z + bQ
    (mod q), it is (aN bQ - bN aQ) / (aQ^2 pi + aQ bQ sigma + bQ^2) --
    no division by the roots' separation, so neither cancellation when
    they are close or tiny, nor a derivative through each root alone.
    (sigma, pi) from one Bairstow step off the frozen (sig0, pi0)."""
    dc = (_C(S2 + 0.0 * S0, 0.0 * S0), S1.conj(), _C(S0 - p, 0.0 * S0), S1,
          _C(S2 + 0.0 * S0, 0.0 * S0))
    sig, pi_ = _bairstow(dc, sig0, pi0)
    Q2 = dc[0]
    Q1 = S1 + sig * Q2
    Q0 = sig * Q1 + (S0 - p) - pi_ * Q2
    aN, bN = K1 * sig + K0, K1.conj() - K1 * pi_
    aQ, bQ = Q2 * sig + Q1, Q0 - Q2 * pi_
    den = aQ * aQ * pi_ + aQ * bQ * sig + bQ * bQ
    den = _cwhere(den.abs() > 0, den, 1.0)
    return (aN * bQ - bN * aQ) / den


def _pole_inside(S0, S1, S2, K0, K1, p, z0):
    """-(pi/p) Re sum_{|z| < 1} N(z) / D'(z): the whole-ellipse contour.
    With the ellipse inside the star no root lies on the unit circle and
    they come in pairs (z, 1/conj z), so exactly two are inside: summed as
    one quadratic factor (_pair_sum). Summing the two residues separately
    loses fp32 derivatives to cancellation when the roots are tiny (a
    nearly centred, nearly round planet)."""
    one = _C(1.0 + 0.0 * S0, 0.0 * S0)
    sig0, pi0 = _C(0.0 * S0, 0.0 * S0), one
    for zf in z0:
        inner = zf.abs() < 1.0
        sig0 = sig0 + _cwhere(inner, zf, 0.0)
        pi0 = pi0 * _cwhere(inner, zf, one)
    acc = _pair_sum(S0, S1, S2, K0, K1, p, sig0.sg(), pi0.sg()).re
    return -_PI * acc / p


def _dlog(z, lo, hi, inner, w1, w2):
    """Continuous change of log(e^{i phi} - z) over [lo, hi]."""
    dm = mx.log((w2 - z).abs()) - mx.log((w1 - z).abs())
    zi = _cwhere(inner, z, 0.5)
    zo = _cwhere(inner, 2.0, z)
    d_in = (hi - lo) + (1.0 - zi / w2).arg() - (1.0 - zi / w1).arg()
    d_out = (1.0 - w2 / zo).arg() - (1.0 - w1 / zo).arg()
    return _C(dm, mx.where(inner, d_in, d_out))


def _J(u, Delta):
    x = Delta / (u * u)
    acc = _C(0.0 * u.re, 0.0 * u.re)
    for k in range(_N_SERIES - 1, -1, -1):
        acc = acc * x + 1.0 / (2 * k + 1)
    return -(acc / u)


def _bairstow(dc, sig0, pi0):
    """One Newton step on the remainder of D mod (z^2 - sigma z + pi);
    (sig0, pi0) frozen, dc = (d0..d4) live (squishierplanet.jaxlc._bairstow)."""
    zero = _C(0.0 * sig0.re, 0.0 * sig0.re)
    al, be = zero, zero + 1.0
    das = dbs = dap = dbp = zero
    r1, r0 = dc[0] * al, dc[0] * be
    J11 = J12 = J21 = J22 = zero
    for k in range(1, 5):
        al, be, das, dbs, dap, dbp = (al * sig0 + be, -(al * pi0),
                                      das * sig0 + al + dbs, -(das * pi0),
                                      dap * sig0 + dbp, -(dap * pi0) - al)
        r1 = r1 + dc[k] * al
        r0 = r0 + dc[k] * be
        J11, J12 = J11 + dc[k] * das, J12 + dc[k] * dap
        J21, J22 = J21 + dc[k] * dbs, J22 + dc[k] * dbp
    J11, J12, J21, J22 = J11.sg(), J12.sg(), J21.sg(), J22.sg()
    det = J11 * J22 - J12 * J21
    det = _cwhere(det.abs() > 0, det, 1.0)
    return (sig0 - (J22 * r1 - J12 * r0) / det,
            pi0 - (-(J21 * r1) + J11 * r0) / det)


def _pole_arc(S0, S1, S2, K0, K1, p, z0, lo, hi):
    """(1/2p) Re of the ellipse-arc contour integral over [lo, hi]
    (squishierplanet.jaxlc._pole_ellipse, one arc)."""
    w1, w2 = _expi(lo), _expi(hi)
    zst = _stack(z0)
    inner0 = [zf.abs() < 1.0 for zf in z0]
    # near-double pairs: up to two disjoint, by |Delta| / (distance to arc)^2
    used = [mx.zeros(S0.shape, dtype=mx.bool_) for _ in range(4)]
    pairs = []
    PJ = mx.array([j for j, _ in _PAIRS])
    PL = mx.array([l for _, l in _PAIRS])
    for _ in range(2):
        scores = []
        for j, l in _PAIRS:
            okp = ~used[j] & ~used[l]
            Dl = (z0[j] - z0[l]) * (z0[j] - z0[l]) * 0.25
            c = (z0[j] + z0[l]) * 0.5
            ang = _mod2pi(c.arg())
            on = ((ang >= lo) & (ang <= hi)) | ((ang + _TWO_PI >= lo) & (ang + _TWO_PI <= hi))
            d_end = mx.minimum((w1 - c).abs(), (w2 - c).abs())
            d = mx.where(on, mx.abs(c.abs() - 1.0), d_end)
            sc = Dl.abs() / mx.maximum(d * d, 1e-300 if S0.dtype == mx.float64 else 1e-30)
            scores.append(mx.where(okp, sc, mx.inf))
        S = mx.stack(scores, -1)
        best = mx.stop_gradient(mx.argmin(mx.stop_gradient(S), axis=-1))
        good = mx.take_along_axis(S, best[..., None], axis=-1)[..., 0] <= _PAIR_RATIO
        jj, ll = mx.take(PJ, best), mx.take(PL, best)
        for k in range(4):
            used[k] = used[k] | (((jj == k) | (ll == k)) & good)
        pairs.append((jj, ll, good))
    # isolated roots: live Newton step, residue, continued log
    z = _live_roots(z0, S0, S1, S2, p)
    ell = _C(0.0 * S0, 0.0 * S0)
    for k in range(4):
        dD = _dD(z[k], S0, S1, S2, p)
        live = ~used[k] & (dD.abs() > 0)
        R = -(_N(z[k], K0, K1) / (_C(0.0, 1.0) * _cwhere(live, dD, 1.0)))
        dl = _dlog(z[k], lo, hi, inner0[k], w1, w2)
        ell = ell + _cwhere(live, R * dl, 0.0)
    # pairs as one quadratic factor (sigma, pi)
    dc = (_C(S2 + 0.0 * S0, 0.0 * S0), S1.conj(), _C(S0 - p, 0.0 * S0), S1,
          _C(S2 + 0.0 * S0, 0.0 * S0))
    for jj, ll, good in pairs:
        z1, z2 = _take(zst, jj), _take(zst, ll)
        sig0 = _cwhere(good, z1 + z2, 7.0).sg()
        pi0 = _cwhere(good, z1 * z2, 12.0).sg()
        sig, pi_ = _bairstow(dc, sig0, pi0)
        Q2 = dc[0]
        Q1 = S1 + sig * Q2
        Q0 = sig * Q1 + (S0 - p) - pi_ * Q2
        aN, bN = K1 * sig + K0, K1.conj() - K1 * pi_
        aQ, bQ = Q2 * sig + Q1, Q0 - Q2 * pi_
        den = aQ * aQ * pi_ + aQ * bQ * sig + bQ * bQ
        den = _cwhere(good & (den.abs() > 0), den, 1.0)
        Acoef = _C(0.0, 1.0) * (aN * bQ - bN * aQ) / den
        Ccoef = _C(0.0, 0.5) * (aN * aQ * pi_ * 2.0 + (aN * bQ + bN * aQ) * sig
                                + bN * bQ * 2.0) / den
        qh = w2 * w2 - sig * w2 + pi_
        ql = w1 * w1 - sig * w1 + pi_
        darg = (qh / ql).arg()
        in1 = z1.abs() < 1.0
        in2 = z2.abs() < 1.0
        rough = (_dlog(z1, lo, hi, in1, w1, w2).im + _dlog(z2, lo, hi, in2, w1, w2).im)
        wind = _sg(mx.round((_sg(rough) - _sg(darg)) / _TWO_PI))
        LQ = _C(mx.log(qh.abs()) - mx.log(ql.abs()), darg + _TWO_PI * wind)
        Delta = sig * sig * 0.25 - pi_
        c = sig * 0.5
        Jd = _J(w2 - c, Delta) - _J(w1 - c, Delta)
        contrib = Acoef * LQ * 0.5 + Ccoef * Jd
        ell = ell + _cwhere(good, contrib, 0.0)
    return ell.re / (2.0 * p)


# --------------------------------------------------------------------------
# the columns
# --------------------------------------------------------------------------

def _check_domain(r, f):
    """r_eff <= (1 - f)^(3/2) for host values (Python or numpy), raising.
    An mx.array is never read back here -- under mx.compile even an
    attempted read poisons the trace -- so array arguments are checked in
    the graph instead (``_in_domain``: NaN outside)."""
    if isinstance(r, mx.array) or isinstance(f, mx.array):
        return
    rv = np.asarray(r, dtype=np.float64)
    fv = np.asarray(f, dtype=np.float64)
    if np.any(fv < 0.0) or np.any(fv >= 1.0):
        raise ValueError(f"flattening f must be in [0, 1); got {fv}")
    if np.any(rv > max_r_eff(fv) * (1.0 + 1e-12)):
        raise ValueError(
            "r_eff > (1 - f)^(3/2): the planet could cross the stellar limb "
            "more than twice, outside the oblate path's domain")


def _in_domain(r, f):
    """The domain as a mask, for array arguments (see _check_domain)."""
    return ((f >= 0.0) & (f < 1.0)
            & (r <= (1.0 - mx.minimum(mx.maximum(f, 0.0), 1.0)) ** 1.5
               * (1.0 + 1e-12)))


def _safe(m, live, safe):
    return mx.where(m, live, safe)


# Sanitized stand-in for the lanes a branch does not own: a small ellipse
# straddling the limb (two clean crossings, no near-double pole roots), and
# its crossing angles, solved once here in float64.
_SAFE_PART = (1.0, 0.02, 0.1, 0.08)
_SAFE_IN = (0.05, 0.03, 0.1, 0.08)


def _safe_crossings():
    x0, y0, a, b = _SAFE_PART
    g = lambda t: (x0 + a * np.cos(t)) ** 2 + (y0 + b * np.sin(t)) ** 2 - 1.0
    t = np.linspace(0.0, 2 * np.pi, 4001)
    v = g(t)
    out = []
    for k in np.nonzero(np.sign(v[:-1]) != np.sign(v[1:]))[0]:
        lo, hi = t[k], t[k + 1]
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if np.sign(g(mid)) == np.sign(g(lo)) else (lo, mid)
        out.append(0.5 * (lo + hi))
    assert len(out) == 2
    return tuple(sorted(out))


_SAFE_PHI = _safe_crossings()
_SAFE_ROOTS = {}


def _safe_roots(geom, p):
    """The four roots of D at level p for a constant safe geometry, solved
    once on the host (float64) and cached: a branch's non-owned lanes take
    these, so the shared root solve need not run on them."""
    key = (geom, float(p))
    if key not in _SAFE_ROOTS:
        x0, y0, a, b = geom
        S1 = a * x0 - 1j * b * y0
        coef = [0.25 * (a * a - b * b), S1, x0 * x0 + y0 * y0 + 0.5 * (a * a + b * b) - p,
                np.conj(S1), 0.25 * (a * a - b * b)]
        z = np.roots(coef)
        for _ in range(3):
            z = z - np.polyval(coef, z) / np.polyval(np.polyder(coef), z)
        _SAFE_ROOTS[key] = tuple(complex(v) for v in z)
    return _SAFE_ROOTS[key]


def _with_safe_roots(m, z, geom, p):
    """Lanes in m keep z; the rest take the safe geometry's roots."""
    return [_cwhere(m, zi, _C(c.real, c.imag))
            for zi, c in zip(z, _safe_roots(geom, p))]


def generator_cols_oblate(x0, y0, r, f, law):
    """Generator deviation columns [E0, E1, E2, P_k...] (minus the occulted
    flux of mu^0, mu^2, mu^4 and each pole of ``law``), elementwise over
    broadcast (x0, y0, r, f). Lanes with f below f_sw are not filled
    (shape_cols_oblate takes the spherical forms there)."""
    law = get_law(law)
    dt = x0.dtype
    zero = x0 * 0.0 + y0 * 0.0 + r * 0.0 + f * 0.0
    x0, y0, r, f = x0 + zero, y0 + zero, r + zero, f + zero
    small = f < _F_SW[dt]
    fs = mx.where(small, 0.1, f)                       # sanitized for the regimes
    A, B = axes(r, fs)
    # -- regimes (frozen) ----------------------------------------------------
    gx0, gy0, gA, gB = _sg(x0), _sg(y0), _sg(A), _sg(B)
    d = mx.sqrt(gx0 * gx0 + gy0 * gy0)
    fast_in = d + gA < 1.0
    fast_out = d - gA >= 1.0
    general = ~fast_in & ~fast_out
    sx, sy, sa_, sb_ = _SAFE_PART
    cnt, phi_a, phi_b = _crossings(_safe(general, gx0, sx), _safe(general, gy0, sy),
                                   _safe(general, gA, sa_), _safe(general, gB, sb_), dt)
    partial = general & (cnt == 2) & ~small
    inside = (fast_in | (general & (cnt != 2) & (d < 1.0))) & ~small
    # -- inside: the whole-ellipse closed forms ---------------------------------
    ix0, iy0 = _safe(inside, x0, _SAFE_IN[0]), _safe(inside, y0, _SAFE_IN[1])
    iA, iB = _safe(inside, A, _SAFE_IN[2]), _safe(inside, B, _SAFE_IN[3])
    S0, S1, S2, K0, K1 = _coeffs(ix0, iy0, iA, iB)
    q = _q_coeffs(S0, S1, S2, K0, K1, _N_MAX)
    even_in = [_TWO_PI * qn[0].re for qn in q]
    # one root solve per pole, shared by both branches: live geometry where
    # either owns the lane, the partial branch's safe geometry elsewhere
    own = inside | partial
    gx, gy = _safe(own, _sg(x0), sx), _safe(own, _sg(y0), sy)
    gA_, gB_ = _safe(own, _sg(A), sa_), _safe(own, _sg(B), sb_)
    hS0, hS1, hS2, _, _ = _coeffs(gx, gy, gA_, gB_)
    shared = {e: [z.sg() for z in _roots(hS0, hS1, hS2, 1.0 + e)] for e in law.eps}
    pole_in = [_pole_inside(S0, S1, S2, K0, K1, 1.0 + e,
                            _with_safe_roots(inside, shared[e], _SAFE_IN, 1.0 + e))
               for e in law.eps]
    # -- partial: one ellipse arc, one star arc ---------------------------------
    px0, py0 = _safe(partial, x0, sx), _safe(partial, y0, sy)
    pA, pB = _safe(partial, A, sa_), _safe(partial, B, sb_)
    gpx0, gpy0, gpA, gpB = _sg(px0), _sg(py0), _sg(pA), _sg(pB)
    pa = _sg(mx.where(partial, phi_a, _SAFE_PHI[0]))
    pb = _sg(mx.where(partial, phi_b, _SAFE_PHI[1]))
    lo, hi, dth = (_sg(v) for v in _arcs(gpx0, gpy0, gpA, gpB, pa, pb))
    corner = _corner(px0, py0, pA, pB, lo, hi)
    S0, S1, S2, K0, K1 = _coeffs(px0, py0, pA, pB)
    q = _q_coeffs(S0, S1, S2, K0, K1, _N_MAX)
    star = dth + corner
    even_pt = [v + star / (2.0 * (n + 1.0)) for n, v in enumerate(_even_arc(q, lo, hi))]
    pole_pt = []
    for e in law.eps:
        p = 1.0 + e
        z0 = _with_safe_roots(partial, shared[e], _SAFE_PART, p)
        pole_pt.append(_pole_arc(S0, S1, S2, K0, K1, p, z0, lo, hi)
                       + star / (2.0 * p * e))
    # -- assemble: deviation = minus the occulted flux ----------------------------
    occ = [mx.where(inside, ei, mx.where(partial, ep, 0.0))
           for ei, ep in zip(even_in + pole_in, even_pt + pole_pt)]
    return [-o for o in occ]


def _to_dtype(v, dt):
    """An argument in the data's dtype (numpy float64 is not demoted
    through float32 on the way, as mx.array() alone would)."""
    if isinstance(v, mx.array):
        return v if v.dtype == dt else v.astype(dt)
    return mx.array(np.asarray(v, dtype=np.float64), dtype=dt)


@fp64_on_cpu(any_arg=True)
def shape_cols_oblate(x0, y0, r, f, law) -> mx.array:
    """Shape-basis columns [E0, T_1..T_n] of an oblate planet, stacked on a
    trailing axis -- the spherical path's ``hybrid.shape_cols`` layout and
    ``ld_basis`` contract (flux - 1 == (B @ c) / (N @ c), c = (1, -w),
    N = hybrid_norms(law)). (x0, y0): centre in the principal frame
    (``principal_frame``); r: area-equivalent radius; f: flattening, with
    r <= (1 - f)^(3/2). y0, r and f arrays are cast to x0's dtype."""
    law = get_law(law)
    dt = x0.dtype
    _check_domain(r, f)
    y0, r, f = (_to_dtype(v, dt) for v in (y0, r, f))
    gen = generator_cols_oblate(x0, y0, r, f, law)
    S = law.generator_matrix()
    obl = [gen[0]] + [_combine(gen, S[j]) for j in range(law.n_w)]
    small = (f + 0.0 * x0) < _F_SW[dt]
    rr = r + 0.0 * x0
    # floored like the orbit's separations: sqrt's VJP is infinite at a
    # centred planet, and the spherical branch is evaluated (then masked)
    # on every lane -- 0 * inf would be NaN in d/dx0, d/dy0
    tiny = (10.0 * (2.220446049250313e-16 if dt == mx.float64 else 1.1920929e-07)) ** 2
    z = mx.sqrt(mx.maximum(x0 * x0 + y0 * y0, tiny))
    sph = _sph_cols(z, rr, law)
    ok = _in_domain(rr, f + 0.0 * x0)
    cols = [mx.where(ok, mx.where(small, sph[..., k], obl[k]), mx.nan)
            for k in range(law.n_col)]
    return mx.stack(cols, axis=-1)


@fp64_on_cpu(any_arg=True)
def flux_dev_oblate(x0, y0, r, f, w, law) -> mx.array:
    """F - 1 for an oblate planet and a hybrid law with shape weights w
    (a host sequence, (n_w,) or (..., n_w)); arguments as
    ``shape_cols_oblate``."""
    B = shape_cols_oblate(x0, y0, r, f, law)
    cols = [B[..., k] for k in range(B.shape[-1])]
    return combine_cols(cols, w, law)
