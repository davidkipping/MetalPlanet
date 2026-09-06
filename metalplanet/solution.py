"""ALFM19 solution vector (s0, s1, s2) — batched, masked, autodiff-clean.

Everything is returned in *deviation* form:

    s0d = s0 - pi        (0 out of transit; -pi r^2 in full transit)
    s1d = s1 - 2 pi / 3  (0 out of transit)
    s2d = s2             (already 0 out of transit)

so the assembled flux deviation is exactly zero out of transit and is
built from O(depth) quantities — the float32 conditioning discipline of
the sampling engine (never let the fp32 graph subtract O(1) constants
from O(1) results to get a small number).

Geometry (z = sky-projected separation, r = radius ratio, both in stellar
radii; the ALFM19 paper calls z "b"):

    onembmr2 = 1-(z-r)^2 = (r+1-z)(1-r+z)     [product form: no cancellation]
    onembpr2 = 1-(z+r)^2 = (1-z-r)(1+z+r)
    k^2  = onembmr2 / (4 z r),  k^2 - 1 = onembpr2 / (4 z r)
    complete transit  (z <= 1-r)  <=>  k^2 >= 1
    partial overlap   (1-r < z < 1+r)  <=>  k^2 < 1

Regime selection is pure mx.where on masks. Every branch's inputs are
sanitized *before* the branch function is applied (the double-where
trick): mx.where evaluates both branches everywhere, and a NaN/inf in an
inactive branch poisons the *gradient* through NaN * 0 = NaN even though
the forward value is fine.

Two razor-thin Taylor switches guard the only numerically degenerate
lines of the closed forms (ALFM19's special cases, with their first-
derivative corrections as in Limbdark.jl):

* |z - r|   < 10 eps      : Pi-integral becomes 0/0 (Cases 5/6/7)
* |z + r - 1| < sqrt(eps) : k^2 = 1 contact, kc -> 0 (Case 4)

Known autodiff caveat ON the |z - r| < 10 eps sliver only: the Taylor
form there reuses the shared cel modulus 4zr/(1-(z-r)^2) instead of
Limbdark's pure-r 4r^2, which keeps the *value* correct to O(eps) but
lets autodiff redistribute O(1) gradient between dz and dr (their sum —
the derivative along the z = r line — stays exact; both stay finite).
The analytic VJP (vjp.py) uses the paper's continuous partial formulas
and is exact there; production sampling uses that path. The contact
switch has no such artifact (its Taylor form depends on z only through
(z + r - 1) * r, so autodiff partials are faithful).

Outside those slivers the generic cel-based formulas are well-conditioned
(b1 ~ (z-r) and sqrt(p) ~ |z-r| shrink together, so their ratio is exact
to rounding).

Assumes 0 < r < 1 (no total occultation branch) and z >= 0.
"""

from __future__ import annotations

import math

import mlx.core as mx

from .ellip import cel3, dtype_eps

__all__ = ["sn_dev", "sn_dev_with_aux"]

_PI = math.pi
_TWO_PI = 2.0 * math.pi


def _kite_sqarea(z, r):
    """Sixteen x (area of the triangle with sides 1, z, r)^2, Kahan-stable.

    Positive iff the three circles' geometry has a real intersection
    (partial overlap); negative outside — callers clamp.
    """
    # sort (1, z, r) descending: a >= b >= c
    a = mx.maximum(z, r)
    c = mx.minimum(z, r)
    b_ = mx.minimum(a, 1.0 + 0.0 * a)
    a = mx.maximum(a, 1.0 + 0.0 * a)
    b = mx.maximum(b_, c)
    c = mx.minimum(b_, c)
    return (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))


def sn_dev(z: mx.array, r) -> tuple[mx.array, mx.array, mx.array]:
    """(s0 - pi, s1 - 2 pi/3, s2) broadcast over z and r."""
    s0d, s1d, s2d, _ = sn_dev_with_aux(z, r)
    return s0d, s1d, s2d


def sn_dev_with_aux(z: mx.array, r):
    """Solution-vector deviations plus the auxiliary quantities the
    analytic VJP reuses (masks, kappas, kite area, elliptic integrals).
    """
    dtype = z.dtype
    eps = dtype_eps(dtype)
    tiny = 1e-30 if dtype == mx.float32 else 1e-280
    d_req = 10.0 * eps          # |z - r| Taylor switch half-width
    d_con = math.sqrt(eps)      # |z + r - 1| Taylor switch half-width

    z = mx.abs(z)
    zero = z * 0.0 + r * 0.0
    z = z + zero
    r = r + zero
    r2 = r * r
    z2 = z * z

    # ---- masks -----------------------------------------------------------
    m_none = z >= 1.0 + r                    # no overlap
    m_comp = mx.logical_and(z <= 1.0 - r, mx.logical_not(m_none))
    m_part = mx.logical_not(mx.logical_or(m_none, m_comp))
    m_occ = mx.logical_not(m_none)           # any overlap at all
    m_ps = z + r > 1.0                       # "partial side": k^2 < 1
    m_req = mx.logical_and(mx.abs(z - r) < d_req, m_occ)
    m_con = mx.logical_and(mx.abs(z + r - 1.0) < d_con, m_occ)

    # ---- shared geometry (sanitized once, used by every branch) ----------
    onembmr2 = mx.maximum((r + 1.0 - z) * (1.0 - r + z), tiny)   # 1-(z-r)^2
    onembpr2 = (1.0 - z - r) * (1.0 + z + r)                     # 1-(z+r)^2
    fourzr = mx.maximum(4.0 * z * r, tiny)
    sqbr = mx.sqrt(mx.maximum(z * r, tiny))
    zmr = z - r
    zpr = mx.maximum(z + r, tiny)

    # ---- uniform disk: kappas and kite area ------------------------------
    sqarea = _kite_sqarea(z, r)
    kite_floor = (10.0 * eps) ** 2
    kite_safe = mx.sqrt(mx.where(m_part, mx.maximum(sqarea, kite_floor), 1.0))
    kite = mx.where(m_part, kite_safe, 0.0)
    kap0 = mx.arctan2(kite, r2 + z2 - 1.0)        # angle at planet center
    kap1 = mx.arctan2(kite, 1.0 + z2 - r2)        # angle at star center

    s0d_part = -(kap1 + r2 * kap0 - 0.5 * kite)   # = -A_lens
    s0d_comp = -_PI * r2
    s0d = mx.where(m_part, s0d_part, mx.where(m_comp, s0d_comp, 0.0))

    # ---- quadratic term s2 ----------------------------------------------
    eta2 = r2 * (r2 + 2.0 * z2)
    s2_comp = _TWO_PI * r2 * (r2 + 2.0 * z2 - 1.0)     # 2 pi (eta2 - r^2)
    s2_part = 2.0 * s0d_part + 2.0 * (
        kap1 + eta2 * kap0 - 0.25 * kite * (1.0 + 5.0 * r2 + z2)
    )
    s2d = mx.where(m_part, s2_part, mx.where(m_comp, s2_comp, 0.0))

    # ---- linear term s1: one fused cel3 with per-regime arguments --------
    # partial side (k^2 < 1):  kc^2 = -onembpr2/(4 z r)
    # complete side (k^2 > 1): kc^2 = onembpr2/onembmr2  (cel runs at 1/k^2)
    #
    # Backward-pass discipline: a division's VJP is -x/y^2, which overflows
    # for a tiny *clamped* denominator even though the forward value is
    # masked away — and 0 * inf = NaN. So every masked division gets its
    # denominator replaced by 1 where the branch is inactive (the forward
    # value there is irrelevant; the gradient becomes a clean 0).
    fourzr_ps = mx.where(m_ps, fourzr, 1.0)
    on_mr_cs = mx.where(m_ps, 1.0, onembmr2)
    # true range of kc^2 is [0, 1] in-branch; the clip keeps the inactive
    # branch from feeding cel out-of-range kc
    kc2_ps = mx.clip(mx.where(m_ps, -onembpr2, 0.0) / fourzr_ps, 0.0, 1.0)
    kc2_cs = mx.clip(mx.where(m_ps, 0.0, onembpr2) / on_mr_cs, 0.0, 1.0)
    kc2 = mx.where(m_ps, kc2_ps, kc2_cs)
    kc = mx.sqrt(mx.maximum(kc2, tiny))

    bmr_dpr = zmr / zpr
    mu = 3.0 * bmr_dpr / on_mr_cs
    p_cs = bmr_dpr * bmr_dpr * mx.maximum(onembpr2, 0.0) / on_mr_cs
    p_ps = zmr * zmr * kc2_ps

    p_cel = mx.where(m_ps, p_ps, p_cs)
    a1 = mx.where(m_ps, 0.0 * zero, 1.0 + mu)
    b1 = mx.where(m_ps, 3.0 * kc2_ps * zmr * zpr, p_cs + mu)
    # b2 = kc2 gives E(k); b3 = 0 gives (E - (1-m)K)/m — in both regimes.
    Piofk, Eofk, Em1mKdm = cel3(kc, p_cel, a1, b1, kc2, 0.0 * zero)

    sqbr_ps = mx.where(m_ps, sqbr, 1.0)
    lam_ps = onembmr2 * (
        Piofk + (-3.0 + 6.0 * r2 + 2.0 * z * r) * Em1mKdm - fourzr * Eofk
    ) / (3.0 * sqbr_ps)
    sqonembmr2 = mx.sqrt(onembmr2)
    lam_cs = (2.0 / 3.0) * sqonembmr2 * (
        onembpr2 * Piofk - (4.0 - 7.0 * r2 - z2) * Eofk
    )
    lam = mx.where(m_ps, lam_ps, lam_cs)
    # s1 = (2 pi [z >= r] - Lambda1)/3  ->  s1d = -(Lambda1 + 2 pi [z < r])/3
    s1d_gen = -(lam + _TWO_PI * mx.where(z < r, 1.0 + 0.0 * zero, 0.0 * zero)) / 3.0

    # -- Taylor switch at z ~ r (ALFM19 Cases 5/6/7, with first-derivative
    #    corrections; reuses E and Em1mKdm at the branch modulus, which
    #    matches the case modulus to O(|z-r|) — inside the 10 eps window
    #    that error is at rounding level).
    on_mr_req = mx.where(m_req, onembmr2, 1.0)   # ~1 where active (z ~ r)
    m_req_mod = mx.where(m_req, fourzr, 0.0) / on_mr_req
    lam5 = _PI + (2.0 / 3.0) * (
        (2.0 * m_req_mod - 3.0) * Eofk - m_req_mod * Em1mKdm
    ) + zmr * 4.0 * r * (Eofk - 2.0 * Em1mKdm)
    # Case 7 applies only on the partial side of the z ~ r switch, where
    # z ~ r and z + r > 1 force r > ~0.5 — the denominator is never small
    r7 = mx.where(mx.logical_and(m_req, m_ps), r, 1.0)
    lam7 = _PI + (1.0 / (3.0 * r7)) * (
        -m_req_mod * Eofk + (2.0 * m_req_mod - 3.0) * Em1mKdm
    ) - zmr * 2.0 * (2.0 * Eofk - Em1mKdm)
    lam_req = mx.where(m_ps, lam7, lam5)
    s1d_req = -lam_req / 3.0

    # -- Taylor switch at contact z + r ~ 1 (Case 4). Smooth on both sides.
    acos_arg = mx.clip(1.0 - 2.0 * r, -1.0 + eps, 1.0 - eps)
    sqr1mr = mx.sqrt(mx.maximum(r * (1.0 - r), tiny))
    s1_con = (
        _TWO_PI
        - 2.0 * mx.arccos(acos_arg)
        + ((4.0 / 3.0) * (3.0 + 2.0 * r - 8.0 * r2) + 8.0 * (z + r - 1.0) * r)
        * sqr1mr
    ) / 3.0
    s1d_con = s1_con - _TWO_PI / 3.0

    s1d = mx.where(m_req, s1d_req, s1d_gen)
    s1d = mx.where(m_con, s1d_con, s1d)
    s1d = mx.where(m_none, 0.0 * zero, s1d)

    aux = {
        "m_none": m_none, "m_comp": m_comp, "m_part": m_part,
        "m_ps": m_ps, "m_req": m_req, "m_con": m_con,
        "kap0": kap0, "kap1": kap1, "kite": kite,
        "Eofk": Eofk, "Em1mKdm": Em1mKdm,
        "onembmr2": onembmr2, "sqonembmr2": sqonembmr2, "sqbr": sqbr,
        "sqr1mr": sqr1mr,
    }
    return s0d, s1d, s2d, aux
