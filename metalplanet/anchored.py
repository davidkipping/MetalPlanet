"""Transit-anchored eccentric orbit: the fp32-safe low-eccentricity path.

The direct formulation (solve Kepler for E from M = phi + M_tra, then
rotate the orbital-plane Cartesians by w) is exact but badly conditioned
in float32 as e -> 0, and the damage lands on the *gradients*, not the
values. Sampling in (k, h) = (sqrt(e) cos w, sqrt(e) sin w) makes
dw/dk = -h/e, which diverges; the divergence cancels against a matching
term routed through M_tra, so the finite answer is a difference of two
O(a/e) quantities. Measured through the graph path, that costs 6e-4
relative gradient error at e = 1e-3, 5% at 1e-5 and 40% at 1e-6
(benchmarks/v3_kh_grad_conditioning.py) — i.e. HMC would be sampling
noise exactly where the (k, h) parameterization is supposed to shine.

This module removes the cancellation structurally by anchoring the
solve at inferior conjunction. Writing E = E0 + delta with E0 the
eccentric anomaly at transit, Kepler's equation becomes

    phi = delta + es (1 - cos delta) - ec sin delta,
    es = e sin E0,   ec = e cos E0,   phi = 2 pi (t - t0) / P,

and the sky-plane Cartesians become

    u / a = e cos w (cos delta - 1) - B1 sin delta
    v / a = A2 cos delta - B2 sin delta - e sin w

with (writing Delta = f0 - E0 = O(e), omb = 1 - sqrt(1 - e^2) = O(e^2))

    B1 = cos Delta - omb cos E0 sin w
    A2 = cos Delta - omb sin E0 cos w
    B2 = -sin Delta + omb cos E0 cos w.

The identity that kills the cancellation is A1 = cos E0 cos w -
beta sin E0 sin w = e cos w exactly, which follows from the conjunction
condition (cos E0 - e) cos w = beta sin E0 sin w. Every w-dependence
that survives is multiplied by e or e^2, and the two quantities that
carry w at O(1) — e cos w and e sin w — are computed directly from
(k, h) as k sqrt(e) and h sqrt(e), never through w = atan2(h, k). So no
1/e amplification is ever formed. At e = 0 the equations degenerate
exactly to the circular orbit (Delta = omb = 0, B1 = A2 = 1, B2 = 0,
delta = phi).

The solve keeps the cost of the direct path: the Markley starter (whose
~1e-4 accuracy makes its own cancellation irrelevant, and whose
gradients are identically zero by the implicit function theorem) plus
ONE sincos and one fifth-order refinement carried out entirely in the
anchored, O(e)-conditioned coefficients.
"""

from __future__ import annotations

import math

import mlx.core as mx

from .kepler import _markley_starter, _unbroadcast
from .trig import sincos

__all__ = ["anchor_constants", "anchor_constants_ew", "solve_sincos_delta",
           "separation_anchored"]

_TWO_PI = 2.0 * math.pi
_PI = math.pi


def _sqrt_any(x):
    if isinstance(x, mx.array):
        return mx.sqrt(mx.maximum(x, 0.0))
    return math.sqrt(max(x, 0.0))


def _one_minus_cos(s, c):
    """1 - cos, from (sin, cos), without cancellation at either end.

    The inactive branch's denominator is set to 1, not floored: a floored
    1 + c is forward-safe, but its VJP -s^2/(1+c)^2 blows up as c -> -1
    and the mask's zero cotangent then yields 0 * inf = NaN. delta ~ pi
    is an ordinary apastron-side geometry, so this is reached in normal
    sampling, not just at a measure-zero point.
    """
    pos = c > 0.0
    den = mx.where(pos, 1.0 + c, mx.ones_like(c))
    return mx.where(pos, s * s / den, 1.0 - c)


def anchor_constants(k, h):
    """(k, h) = (sqrt(e) cos w, sqrt(e) sin w) -> anchored orbit constants.

    Returns (e, ecw, esw, es, ec, B1, A2, B2). Every output is a smooth
    function of (k, h) with a bounded derivative, including at e = 0 —
    that is the whole point of the module. All of them are per-chain
    scalars, so this runs in the MLX graph and its Jacobian rides
    ordinary autodiff; only the per-point solve goes into a kernel.
    """
    e = k * k + h * h
    sq = mx.sqrt(mx.maximum(e, 1e-30))
    ecw = k * sq                       # e cos w, no division by sqrt(e)
    esw = h * sq                       # e sin w
    # cos w / sin w are needed only inside O(e^2) corrections below, so
    # their O(1) noise at e -> 0 is harmless. The denominator is set to
    # 1 (not merely floored) where the branch is inactive: a floored
    # denominator is forward-safe but its VJP -k/safe^2 overflows and
    # the mask's zero cotangent then gives 0 * inf = NaN.
    pos = e > 0.0
    denom = mx.where(pos, mx.maximum(sq, 1e-30), mx.ones_like(sq))
    cw = mx.where(pos, k / denom, mx.ones_like(k))
    sw = mx.where(pos, h / denom, mx.zeros_like(h))
    # w from the GUARDED (sin w, cos w), never from (h, k) directly:
    # arctan2(0, 0) has an undefined gradient, and e = 0 is an interior
    # point of the (k, h) disc that a sampler genuinely visits.
    w = mx.arctan2(sw, cw)
    return _anchor_core(e, ecw, esw, cw, sw, w)


def anchor_constants_ew(e, w):
    """(e, w) -> anchored orbit constants, w in radians: the same tuple as
    ``anchor_constants``, for callers holding orbital elements.

    (k, h) is the right pair for a *sampler* -- it keeps e -> 0 an interior
    point. But it is singular as a map from (e, w): dk/de = cos w / (2
    sqrt e) is infinite at e = 0, so a gradient in e taken through (k, h)
    is inf * 0 = NaN exactly there, although F is perfectly differentiable
    in e (one-sided) at e = 0 -- eccentricity changes the duration at
    first order. With w given, nothing here needs sqrt(e) or a division:
    every constant is a smooth function of (e, w) on [0, 1) x R, and every
    w-dependence is multiplied by e, so d/dw is exactly 0 at e = 0.
    """
    sw, cw = sincos(w)
    return _anchor_core(e, e * cw, e * sw, cw, sw, w)


def _anchor_core(e, ecw, esw, cw, sw, w):
    beta = mx.sqrt(mx.maximum(1.0 - e * e, 0.0))
    omb = e * e / (1.0 + beta)         # 1 - beta, without cancellation

    # Delta = f0 - E0 with f0 = pi/2 - w, from the half-angle relation
    # tan(E0/2) = kk tan(f0/2). Numerator carries the explicit O(e)
    # factor 1 - kk; the denominator is bounded away from zero, and
    # f0/2 in [-pi/4, 3pi/4) never reaches the tan pole.
    s2, c2 = sincos(0.5 * (0.5 * _PI - w))
    kk = mx.sqrt(mx.maximum((1.0 - e) / (1.0 + e), 0.0))
    one_m_kk = 2.0 * e / ((1.0 + e) * (1.0 + kk))
    delta_a = 2.0 * mx.arctan2(one_m_kk * s2 * c2, c2 * c2 + kk * s2 * s2)
    sD, cD = sincos(delta_a)

    # E0 = pi/2 - w - Delta  =>  sin E0 = cos(w + Delta), cos E0 = sin(w + Delta)
    sE0 = cw * cD - sw * sD
    cE0 = sw * cD + cw * sD
    es = ecw * cD - esw * sD           # e sin E0
    ec = esw * cD + ecw * sD           # e cos E0

    B1 = cD - omb * cE0 * sw
    A2 = cD - omb * sE0 * cw
    B2 = -sD + omb * cE0 * cw
    return e, ecw, esw, es, ec, B1, A2, B2


@mx.custom_function
def solve_sincos_delta(phi: mx.array, es: mx.array, ec: mx.array):
    """(sin delta, cos delta) solving phi = delta + es (1-cos delta)
    - ec sin delta. One sincos; gradients by implicit differentiation."""
    zero = phi * 0.0 + es * 0.0 + ec * 0.0
    phi = phi + zero
    es = es + zero
    ec = ec + zero

    e = mx.sqrt(mx.maximum(es * es + ec * ec, 0.0))
    E0 = mx.arctan2(es, ec)
    M_tra = E0 - es

    # wrap the *standard* mean anomaly, and carry phi along with it so
    # the anchored residual below stays consistent with the fold.
    M = phi + M_tra
    nw = mx.round(M / _TWO_PI)
    M_w = M - _TWO_PI * nw
    phi_w = phi - _TWO_PI * nw

    # starter: Markley on the standard equation, differenced against E0.
    # ~1e-4 accurate; the difference cancels about one digit, which the
    # refinement below discards entirely.
    sign = mx.where(M_w >= 0.0, 1.0 + zero, -1.0 + zero)
    d0 = _markley_starter(mx.abs(M_w), e) * sign - E0

    sd, cd = sincos(d0)                # the ONLY trig call
    omc = _one_minus_cos(sd, cd)
    f_0 = d0 + es * omc - ec * sd - phi_w
    f_1 = 1.0 + es * sd - ec * cd      # = 1 - e cos E > 0
    f_2 = es * cd + ec * sd            # = e sin E
    f_3 = -es * sd + ec * cd           # = e cos E
    d_3 = -f_0 / (f_1 - 0.5 * f_0 * f_2 / f_1)
    d_4 = -f_0 / (f_1 + 0.5 * d_3 * f_2 + d_3 * d_3 * f_3 / 6.0)
    dd = -f_0 / (f_1 + 0.5 * d_4 * f_2 + d_4 * d_4 * f_3 / 6.0
                 - d_4 * d_4 * d_4 * f_2 / 24.0)

    dd2 = dd * dd
    sdd = dd * (1.0 - dd2 / 6.0 * (1.0 - dd2 / 20.0))
    cdd = 1.0 - dd2 * 0.5 * (1.0 - dd2 / 12.0)
    return sd * cdd + cd * sdd, cd * cdd - sd * sdd


@solve_sincos_delta.vjp
def _solve_delta_vjp(primals, cotangents, outputs):
    phi, es, ec = primals
    ct_s, ct_c = cotangents
    sind, cosd = outputs
    # g(delta) = delta + es (1-cos d) - ec sin d = phi
    # D = dg/ddelta = 1 + es sin d - ec cos d = 1 - e cos E
    # ddelta = [dphi - (1-cos d) des + sin d dec] / D
    g = ct_s * cosd - ct_c * sind
    omc = _one_minus_cos(sind, cosd)
    # D = 1 - e cos E >= 1 - e > 0 over the supported range
    D = mx.maximum(1.0 + es * sind - ec * cosd, 1e-12)
    gD = g / D
    shapes = [p.shape if isinstance(p, mx.array) else () for p in primals]
    return (_unbroadcast(gD, shapes[0]),
            _unbroadcast(-gD * omc, shapes[1]),
            _unbroadcast(gD * sind, shapes[2]))


def separation_anchored(phi, k, h, a, ci, eps: float = 1.1920929e-07,
                        consts=None):
    """(z, front) for an eccentric orbit, anchored at inferior conjunction.

    phi = 2 pi (t - t0) / P (t0 = transit time), (k, h) the
    (sqrt(e) cos w, sqrt(e) sin w) pair, a in stellar radii, ci = cos i.
    Equivalent to ``kepler.separation_keplerian`` to round-off, and
    conditioned so that fp32 *gradients* survive e -> 0. ``consts``, if
    given, replaces (k, h) with an ``anchor_constants[_ew]`` tuple.
    """
    if consts is None:
        consts = anchor_constants(k, h)
    e, ecw, esw, es, ec, B1, A2, B2 = consts
    sind, cosd = solve_sincos_delta(phi, es, ec)
    omc = _one_minus_cos(sind, cosd)
    u = a * (-ecw * omc - B1 * sind)
    v = a * (A2 * cosd - B2 * sind - esw)
    z = mx.sqrt(mx.maximum(u * u + (v * ci) ** 2, (10.0 * eps) ** 2))
    return z, v > 0.0


#: column order of the packed per-chain orbit constants consumed by the
#: v3 Metal kernel; kept in lockstep with ``metal._ORB_COLS``.
ORB_COLS = ("ecw", "esw", "es", "ec", "b1", "a2", "b2",
            "ecc", "e0", "mtra", "ci")


def pack_orbit_constants(k, h, ci, consts=None):
    """(k, h, ci) -> (n, 11) per-chain constants for the fused model kernel.

    The last three of the first eight columns feed the Cartesian tail and
    the solve; ``ecc``/``e0``/``mtra`` seed the Markley starter and the
    2-pi fold *only*, so they are detached: the implicit function theorem
    makes the converged root independent of its starter, and rint is
    locally constant, so their true gradient contribution is zero.
    Detaching also keeps E0 = atan2(es, ec) — undefined at e = 0 — out
    of the autodiff graph entirely. ``consts``, if given, replaces (k, h)
    with an ``anchor_constants[_ew]`` tuple.
    """
    if consts is None:
        consts = anchor_constants(k, h)
    e, ecw, esw, es, ec, B1, A2, B2 = consts
    E0 = mx.stop_gradient(mx.arctan2(es, ec))
    cols = [ecw, esw, es, ec, B1, A2, B2,
            mx.stop_gradient(e), E0, mx.stop_gradient(E0 - es),
            ci * mx.ones_like(ecw)]
    return mx.stack([mx.reshape(c, (-1,)) for c in cols], axis=1)
