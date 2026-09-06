"""Bulirsch's complete elliptic integral ``cel`` — batched, fixed-iteration,
dtype-polymorphic MLX.

cel(kc, p, a, b) = \\int_0^{\\pi/2}
    (a cos^2 t + b sin^2 t) /
    ((cos^2 t + p sin^2 t) sqrt(cos^2 t + kc^2 sin^2 t)) dt

This is the single numerical workhorse of the ALFM19/Limbdark transit model:
E(k) = cel(kc, 1, 1, kc^2), K(k) = cel(kc, 1, 1, 1),
(E(m) - (1-m)K(m))/m = cel(kc, 1, 1, 0), and the Pi-like integral of the
linear limb-darkening term. Only p > 0 is required by the transit path, so
the negative-p reduction of Bulirsch (1969) is omitted; ``k^2`` never enters
the recursion at all — everything runs on kc.

The iteration count is *fixed* (data-dependent stopping breaks batching and
mx.compile). An adaptive scan over kc in [1e-40, 1], p in [1e-20, 1e3]
showed the Limbdark stopping rule |g - kc| <= g*sqrt(eps) is met within 8
iterations (fp64) / 6 (fp32) everywhere; N=10/12 reproduces scipy's
ellipk/ellipe and mpmath quadrature of the general integrand to ~1e-15
relative in fp64. Extra iterations past convergence are stable (m doubles
per step; no overflow for the counts used here).
"""

from __future__ import annotations

import mlx.core as mx

__all__ = ["cel", "cel3", "dtype_eps", "n_iter_for"]

_HALF_PI = 1.5707963267948966


def dtype_eps(dtype) -> float:
    """Machine epsilon for an MLX float dtype, as a Python float."""
    if dtype == mx.float32:
        return 1.1920929e-07
    if dtype == mx.float64:
        return 2.220446049250313e-16
    raise ValueError(f"unsupported dtype for cel: {dtype}")


def n_iter_for(dtype) -> int:
    """Fixed iteration count with margin over the measured worst case
    (6 iterations fp32 / 8 fp64 across the full argument range)."""
    return 10 if dtype == mx.float32 else 12


def _p_floor(dtype) -> float:
    """Division-by-zero guard for the p-chain. Deliberately far below any
    physically reachable p: near the z=r / contact corner the Pi-integral's
    p ~ (z-r)^2 kc^2 gets tiny while b1 ~ (z-r) kc^2 shrinks with it, so
    the ratio stays conditioned — clamping at a *large* floor (e.g. eps^2)
    would distort that regime. sqrt(floor)^-2 must not overflow."""
    return 1e-30 if dtype == mx.float32 else 1e-280


def cel(kc: mx.array, p, a, b, n_iter: int | None = None) -> mx.array:
    """Batched cel(kc, p, a, b) for p > 0.

    ``kc`` must be an mx.array (its dtype drives the constants); p, a, b
    may be arrays or Python scalars broadcastable against it. ``kc`` is
    clamped to machine epsilon (the kc -> 0 limit of K diverges only
    logarithmically, so the clamped value is accurate to ~eps*|log eps|)
    and ``p`` to a tiny floor, making the masked-out branches of callers
    NaN-free in both the forward and backward passes.
    """
    dtype = kc.dtype
    eps = dtype_eps(dtype)
    n = n_iter_for(dtype) if n_iter is None else n_iter

    kc = mx.maximum(mx.abs(kc), eps)
    p = mx.maximum(p, _p_floor(dtype))

    # broadcast everything to the common shape via a zero of that shape
    zero = kc * 0.0 + p * 0.0 + a * 0.0 + b * 0.0
    kc = kc + zero
    a = a + zero
    b = b + zero

    ee = kc
    m = 1.0 + zero
    sp = mx.sqrt(p + zero)
    pinv = 1.0 / sp
    b = b * pinv
    f = a
    a = a + b * pinv
    g = ee * pinv
    b = 2.0 * (b + f * g)
    pp = sp + g
    m_next = m + kc
    m = m_next

    for _ in range(n):
        kc_n = 2.0 * mx.sqrt(ee)
        ee = kc_n * m
        f = a
        pinv = 1.0 / pp
        a = a + b * pinv
        g = ee * pinv
        b = 2.0 * (b + f * g)
        pp = pp + g
        m = m + kc_n

    return _HALF_PI * (a * m + b) / (m * (m + pp))


def cel3(kc: mx.array, p, a1, b1, b2, b3, n_iter: int | None = None):
    """Three cel integrals sharing one kc-recursion (Limbdark's vector cel).

    Returns (cel(kc, p, a1, b1), cel(kc, 1, 1, b2), cel(kc, 1, 1, b3)).
    The transit model uses this as (Pi-like, E(k), (E-(1-m)K)/m) — one
    fixed-iteration loop instead of three. p > 0 required (clamped).
    """
    dtype = kc.dtype
    eps = dtype_eps(dtype)
    n = n_iter_for(dtype) if n_iter is None else n_iter

    kc = mx.maximum(mx.abs(kc), eps)
    p = mx.maximum(p, _p_floor(dtype))

    zero = kc * 0.0 + p * 0.0 + a1 * 0.0 + b1 * 0.0 + b2 * 0.0 + b3 * 0.0
    kc = kc + zero
    a1 = a1 + zero
    b1 = b1 + zero
    a2 = 1.0 + zero
    b2 = b2 + zero
    a3 = 1.0 + zero
    b3 = b3 + zero

    ee = kc
    m = 1.0 + zero

    # chain 1 (with p); chains 2 and 3 share the p = 1 chain
    sp = mx.sqrt(p + zero)
    pinv = 1.0 / sp
    b1 = b1 * pinv
    f1 = a1
    a1 = a1 + b1 * pinv
    g = ee * pinv
    b1 = 2.0 * (b1 + f1 * g)
    pp = sp + g

    g1 = ee
    f2 = a2
    a2 = a2 + b2
    b2 = 2.0 * (b2 + f2 * g1)
    f3 = a3
    a3 = a3 + b3
    b3 = 2.0 * (b3 + f3 * g1)
    p1 = 1.0 + g1

    m = m + kc

    for _ in range(n):
        kc_n = 2.0 * mx.sqrt(ee)
        ee = kc_n * m
        f1 = a1
        f2 = a2
        f3 = a3
        pinv = 1.0 / pp
        pinv1 = 1.0 / p1
        a1 = a1 + b1 * pinv
        a2 = a2 + b2 * pinv1
        a3 = a3 + b3 * pinv1
        g = ee * pinv
        g1 = ee * pinv1
        b1 = 2.0 * (b1 + f1 * g)
        b2 = 2.0 * (b2 + f2 * g1)
        b3 = 2.0 * (b3 + f3 * g1)
        pp = pp + g
        p1 = p1 + g1
        m = m + kc_n

    out1 = _HALF_PI * (a1 * m + b1) / (m * (m + pp))
    out2 = _HALF_PI * (a2 * m + b2) / (m * (m + p1))
    out3 = _HALF_PI * (a3 * m + b3) / (m * (m + p1))
    return out1, out2, out3
