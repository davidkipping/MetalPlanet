"""Arbitrary-order polynomial limb darkening (ALFM19 M_n recursion).

The quadratic path needs only s_0, s_1, s_2 (solution.py). For a general
polynomial law

    I(mu)/I0 = 1 - sum_{n=1}^{N} u_n (1 - mu)^n

the flux needs the whole solution vector s_0..s_N. ALFM19 supplies it
through the integrals

    M_n(r, z) = (4 z r)^{n/2} int_{-kap/2}^{kap/2} (k^2 - sin^2 x)^{n/2} dx

with k^2 = (1 - (z-r)^2) / (4 z r), which satisfy the three-term
recursion

    M_n = [2 (n-1) (1 - r^2 - z^2) M_{n-2} + (n-2) * sqarea * M_{n-4}] / n

seeded by closed forms for M_0..M_3, and then

    s_n = -(2 r^2 M_n - n/(n+2) [(1 - r^2 - z^2) M_n + sqarea M_{n-2}])

for n >= 3, where ``sqarea`` is 16 x (area of the triangle with sides
1, z, r) -- Heron's form, which is *negative* for a complete transit
where the triangle inequality fails. That sign is load-bearing: clamping
it to zero (as the partial-overlap kite area is clamped) silently breaks
every term above n = 2.

Recursion direction. Limbdark.jl switches to a downward, series-seeded
recursion when k^2 < 0.5, because the upward one loses precision. We use
the upward recursion only, having measured the loss against a 40-digit
mpmath direct-integration oracle across r in [0.01, 0.8] and the full b
range: worst |dF| is 1e-15 at N = 8, 2e-13 at N = 16, 2e-12 at N = 20 and
3e-9 at N = 30 -- i.e. still far below batman's own 2e-8 floor at orders
no physical law uses. Anyone needing N > 24 at full float64 precision
wants the downward recursion; see docs for the reference.

``n_max`` is a *Python int*, so the recursion unrolls at graph-build
time: no data-dependent control flow, and everything stays
mx.compile-safe and batched.
"""

from __future__ import annotations

import math

import mlx.core as mx

from .greens import greens_affine, greens_transform_np
from .solution import _kite_sqarea, sn_dev_with_aux
from .dtypes import fp64_on_cpu

__all__ = ["sn_dev_poly", "flux_dev_poly", "poly_norm"]

_PI = math.pi


@fp64_on_cpu(any_arg=True)
def sn_dev_poly(z: mx.array, r, n_max: int) -> list:
    """[s_0 - pi, s_1 - 2 pi/3, s_2, s_3, ..., s_n_max], broadcast over
    z and r. Every element is zero out of transit."""
    if n_max < 0:
        raise ValueError("n_max must be >= 0")
    s0d, s1d, s2d, aux = sn_dev_with_aux(z, r)
    out = [s0d, s1d, s2d][:max(n_max + 1, 1)]
    if n_max < 3:
        return out

    zero = s0d * 0.0
    zz = mx.abs(z) + zero
    rr = r + zero
    r2 = rr * rr
    onemr2mb2 = 1.0 - r2 - zz * zz
    onembmr2 = aux["onembmr2"]
    fourzr = mx.maximum(4.0 * zz * rr, 1e-300 if z.dtype == mx.float64
                        else 1e-30)
    # signed Heron area^2: negative for a complete transit, and needed
    # with that sign by both the recursion and the s_n assembly
    sqarea = _kite_sqarea(zz, rr)

    m_comp, m_part = aux["m_comp"], aux["m_part"]
    kap0, kite = aux["kap0"], aux["kite"]
    Eofk, Em1mKdm = aux["Eofk"], aux["Em1mKdm"]
    sqbr, sqonembmr2 = aux["sqbr"], aux["sqonembmr2"]
    # Per-branch denominator sanitisation. mx.where evaluates BOTH sides,
    # so k^2 = onembmr2/(4zr) must stay finite on complete-transit lanes
    # (where 4zr -> 0 at z = 0) and 1/k^2 must stay finite on partial and
    # out-of-transit lanes (where onembmr2 is floored to `tiny`). A merely
    # floored denominator gives a finite value but an overflowing VJP, and
    # the mask's zero cotangent then yields 0 * inf = NaN.
    one = zero + 1.0
    # Total occultation (r > 1) is a partial-mask lane: kite = kap0 = 0
    # there (sn_dev_with_aux), so M_0 = M_2 = 0 and, with k^2 at its floor,
    # M_1, M_3 and every s_n (n >= 3) come out ~1e-270 -- the full-disk 0.
    # Its 4zr vanishes at z = 0, hence the mask here (0 * inf in the VJP).
    fourzr_p = mx.where(mx.logical_and(m_part, mx.logical_not(aux["m_tot"])),
                        fourzr, one)
    onembmr2_c = mx.where(m_comp, onembmr2, one)
    k2 = onembmr2 / fourzr_p            # only read on partial lanes
    k2inv = fourzr / onembmr2_c         # only read on complete lanes
    twothird = 2.0 / 3.0

    # M_0..M_3, complete (k^2 >= 1) and partial (k^2 < 1) closed forms
    Mc = [_PI + zero,
          2.0 * sqonembmr2 * Eofk,
          _PI * onemr2mb2,
          sqonembmr2 ** 3 * twothird
          * ((3.0 - 2.0 * k2inv) * Eofk + k2inv * Em1mKdm)]
    two_sqbr = 2.0 * sqbr
    Mp = [kap0,
          two_sqbr * 2.0 * k2 * Em1mKdm,
          kap0 * onemr2mb2 + kite,
          two_sqbr ** 3 * twothird * k2
          * (Eofk + (3.0 * k2 - 2.0) * Em1mKdm)]
    M = [mx.where(m_comp, c, mx.where(m_part, p, zero))
         for c, p in zip(Mc, Mp)]

    # upward recursion; n_max is a Python int so this unrolls
    for n in range(4, n_max + 1):
        M.append((2.0 * (n - 1) * onemr2mb2 * M[n - 2]
                  + (n - 2) * sqarea * M[n - 4]) / float(n))

    for n in range(3, n_max + 1):
        s_n = -(2.0 * r2 * M[n]
                - (n / (n + 2.0)) * (onemr2mb2 * M[n] + sqarea * M[n - 2]))
        out.append(mx.where(aux["m_none"], zero, s_n))
    return out


def poly_norm(u) -> float:
    """pi (g_0 + 2 g_1 / 3): the unocculted flux for a polynomial law."""
    g = greens_transform_np(u)
    return float(_PI * (g[0] + 2.0 * g[1] / 3.0))


@fp64_on_cpu(any_arg=True)
def flux_dev_poly(z: mx.array, r, u, n_max: int | None = None) -> mx.array:
    """F - 1 for an arbitrary-order polynomial law.

    ``u`` may be a host sequence (u_1..u_N) — then the u -> g transform
    is folded into float64 constants — or an mx.array, in which case g
    is built in-graph through the affine form of the same transform, so
    the coefficients stay traced (batman-style updates, and gradients
    with respect to the limb darkening).
    """
    if isinstance(u, mx.array):
        n = u.shape[-1] if u.ndim else 1
        n_max = int(n) if n_max is None else n_max
        A, c = greens_affine(int(n))
        Am = mx.array(A, dtype=u.dtype)
        cm = mx.array(c, dtype=u.dtype)
        if u.ndim >= 2:
            # batched coefficients: (n_sets, N) -> (n_sets, N+1), kept as
            # a column so it broadcasts against the (n_sets, m) solution
            g = mx.matmul(u, Am.T) + cm
            gs = [g[..., i:i + 1] for i in range(g.shape[-1])]
        else:
            g = mx.matmul(Am, mx.reshape(u, (int(n),))) + cm
            gs = [g[i] for i in range(g.shape[0])]
        s = sn_dev_poly(z, r, n_max)
        norm = _PI * (gs[0] + 2.0 * gs[1] / 3.0)
        acc = None
        for gn, sn in zip(gs, s):
            term = (gn / norm) * sn
            acc = term if acc is None else acc + term
        return z * 0.0 if acc is None else acc

    g = greens_transform_np(u)
    n_max = g.size - 1
    s = sn_dev_poly(z, r, n_max)
    norm = _PI * (g[0] + 2.0 * g[1] / 3.0)
    acc = None
    for gn, sn in zip(g.tolist(), s):
        if gn == 0.0:
            continue
        term = float(gn / norm) * sn
        acc = term if acc is None else acc + term
    if acc is None:
        return z * 0.0
    return acc
