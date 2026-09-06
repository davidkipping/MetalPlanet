"""Scalar float64 oracle: direct numpy port of Limbdark.jl (Agol's own
reference implementation of ALFM19), with adaptive-iteration cel.

Used only by tests. Exercises the *exact* special-case structure of the
Julia source (b==0, b==r, b+r==1 handled as exact branches), so it is an
independent check on the masked/Taylor-switch MLX implementation.

Validated anchors:
  * cel matches scipy ellipk/ellipe and mpmath quadrature (see study)
  * s1(0.1, 0.5) = 2.067294367278038 (Limbdark.jl docstring)
"""

from __future__ import annotations

import numpy as np

EPS = np.finfo(np.float64).eps


def cel(kc, p, a, b, itmax=60):
    ca = np.sqrt(EPS)
    if kc == 0.0:
        kc = EPS
    kc = abs(kc)
    ee = kc
    m = 1.0
    if p > 0.0:
        p = np.sqrt(p)
        pinv = 1.0 / p
        b = b * pinv
    else:
        q = 1.0 - kc * kc  # k^2
        g = 1.0 - p
        f = g - (1.0 - kc * kc)
        q *= b - a * p
        ginv = 1.0 / g
        p = np.sqrt(f * ginv)
        a = (a - b) * ginv
        pinv = 1.0 / p
        b = -q * ginv * ginv * pinv + a * p
    f = a
    a = a + b * pinv
    g = ee * pinv
    b = 2.0 * (b + f * g)
    p = p + g
    g = m
    m = m + kc
    it = 0
    while abs(g - kc) > g * ca and it < itmax:
        kc = 2.0 * np.sqrt(ee)
        ee = kc * m
        f = a
        pinv = 1.0 / p
        a = a + b * pinv
        g = ee * pinv
        b = 2.0 * (b + f * g)
        p = p + g
        g = m
        m = m + kc
        it += 1
    return 0.5 * np.pi * (a * m + b) / (m * (m + p))


def s1_ref(r, b):
    """Linear limb-darkening term s1 (Limbdark's s2_ell, scalar fp64)."""
    third = 1.0 / 3.0
    Lambda1 = 0.0
    if b >= 1.0 + r or r == 0.0:
        Lambda1 = 0.0
    elif b <= r - 1.0:
        Lambda1 = 0.0
    else:
        if b == 0.0:
            Lambda1 = -2.0 * np.pi * np.sqrt(1.0 - r * r) ** 3
        elif b == r:
            if r == 0.5:
                Lambda1 = np.pi - 4.0 * third
            elif r < 0.5:
                m = 4.0 * r * r
                Eofk = cel(np.sqrt(1.0 - m), 1.0, 1.0, 1.0 - m)
                Em1mKdm = cel(np.sqrt(1.0 - m), 1.0, 1.0, 0.0)
                Lambda1 = np.pi + 2.0 * third * ((2.0 * m - 3.0) * Eofk - m * Em1mKdm)
            else:
                m = 4.0 * r * r
                minv = 1.0 / m
                Eofk = cel(np.sqrt(1.0 - minv), 1.0, 1.0, 1.0 - minv)
                Em1mKdm = cel(np.sqrt(1.0 - minv), 1.0, 1.0, 0.0)
                Lambda1 = np.pi + third / r * (-m * Eofk + (2.0 * m - 3.0) * Em1mKdm)
        else:
            onembpr2 = (1.0 - r - b) * (1.0 + r + b)
            onembmr2 = (r + 1.0 - b) * (1.0 - r + b)
            fourbr = 4.0 * b * r
            if b + r > 1.0:  # k^2 < 1
                kc2 = -onembpr2 / fourbr
                kc = np.sqrt(kc2)
                Piofk = cel(kc, (b - r) ** 2 * kc2, 0.0, 3.0 * kc2 * (b - r) * (b + r))
                Eofk = cel(kc, 1.0, 1.0, kc2)
                Em1mKdm = cel(kc, 1.0, 1.0, 0.0)
                Lambda1 = onembmr2 * (
                    Piofk + (-3.0 + 6.0 * r * r + 2.0 * b * r) * Em1mKdm
                    - fourbr * Eofk
                ) * third / np.sqrt(b * r)
            elif b + r < 1.0:  # k^2 > 1
                kc2 = onembpr2 / onembmr2
                kc = np.sqrt(kc2)
                bmrdbpr = (b - r) / (b + r)
                mu = 3.0 * bmrdbpr / onembmr2
                p = bmrdbpr * bmrdbpr * onembpr2 / onembmr2
                Piofk = cel(kc, p, 1.0 + mu, p + mu)
                Eofk = cel(kc, 1.0, 1.0, kc2)
                Lambda1 = 2.0 * np.sqrt(onembmr2) * (
                    onembpr2 * Piofk - (4.0 - 7.0 * r * r - b * b) * Eofk
                ) * third
            else:  # b + r == 1
                Lambda1 = (
                    2.0 * np.arccos(1.0 - 2.0 * r)
                    - 4.0 * third * (3.0 + 2.0 * r - 8.0 * r * r)
                    * np.sqrt(r * (1.0 - r))
                    - 2.0 * np.pi * float(r > 0.5)
                )
    return ((1.0 - float(r > b)) * 2.0 * np.pi - Lambda1) * third


def s0_ref(r, b):
    """Uniform-disk term s0 (pi minus lens area)."""
    if b >= 1.0 + r or r == 0.0:
        return np.pi
    if b <= 1.0 - r:
        return np.pi * (1.0 - r * r)
    # partial overlap
    # sixteen x triangle(1, r, b) area^2, Kahan-sorted
    a_, b_, c_ = sorted((1.0, r, b), reverse=True)
    sqarea = (a_ + (b_ + c_)) * (c_ - (a_ - b_)) * (c_ + (a_ - b_)) * (a_ + (b_ - c_))
    kite = np.sqrt(max(sqarea, 0.0))
    kap0 = np.arctan2(kite, (r - 1.0) * (r + 1.0) + b * b)
    pimkap1 = np.arctan2(kite, (r - 1.0) * (r + 1.0) - b * b)
    return pimkap1 - r * r * kap0 + 0.5 * kite


def s2_ref(r, b):
    """Quadratic term s2 = 2 s0 + 4 pi eta-type correction."""
    if b >= 1.0 + r or r == 0.0:
        return 0.0
    r2 = r * r
    b2 = b * b
    eta2 = r2 * (r2 + 2.0 * b2)
    s0 = s0_ref(r, b)
    if b <= 1.0 - r:
        four_pi_eta = 2.0 * np.pi * (eta2 - 1.0)
    else:
        a_, b_, c_ = sorted((1.0, r, b), reverse=True)
        sqarea = (a_ + (b_ + c_)) * (c_ - (a_ - b_)) * (c_ + (a_ - b_)) * (a_ + (b_ - c_))
        kite = np.sqrt(max(sqarea, 0.0))
        kap0 = np.arctan2(kite, (r - 1.0) * (r + 1.0) + b2)
        pimkap1 = np.arctan2(kite, (r - 1.0) * (r + 1.0) - b2)
        four_pi_eta = 2.0 * (
            -pimkap1 + eta2 * kap0 - 0.25 * kite * (1.0 + 5.0 * r2 + b2)
        )
    return 2.0 * s0 + four_pi_eta


def flux_quad_ref(r, b, u1, u2):
    """Quadratic limb-darkened flux (absolute, 1 out of transit)."""
    g0 = 1.0 - u1 - 1.5 * u2
    g1 = u1 + 2.0 * u2
    g2 = -0.25 * u2
    num = g0 * s0_ref(r, b) + g1 * s1_ref(r, b) + g2 * s2_ref(r, b)
    return num / (np.pi * (g0 + 2.0 * g1 / 3.0))
