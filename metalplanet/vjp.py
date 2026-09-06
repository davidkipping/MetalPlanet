"""Stage 2: analytic custom VJP for the photometric core (ALFM19 closed-
form partials).

Reverse-mode autodiff through the elementwise cel3 graph is memory-
bandwidth-bound: the backward pass re-reads every stored intermediate of
10-12 unrolled iterations. The ALFM19 partials

    ds0/dr = -2 r kappa0                 ds0/dz = kite / z
    ds1/dr = -2 r sqrt(1-(z-r)^2) * {Em1mKdm / sqrt(zr)   (k^2 < 1)
                                     2 E                   (k^2 > 1)}
    ds1/dz = (2 r / 3) sqrt(1-(z-r)^2) * {(2E - Em1mKdm)/sqrt(zr)  (k^2<1)
                                          -2 (E - 2 Em1mKdm)       (k^2>1)}
    ds2/dr = 2 ds0/dr + 8 r ((r^2+z^2) kappa0 - kite)      (partial)
    ds2/dz = 2 ds0/dz + (2/z)(4 z^2 r^2 kappa0 - (1+r^2+z^2) kite)

(with the complete-transit specializations kappa0 -> pi, kite -> 0)
need only quantities the forward pass already computes — E, Em1mKdm,
kappa0, kite — so the VJP is a *forward-style* recomputation plus an
elementwise contraction: no tape, no stored iteration intermediates.

Unlike the value formulas, these derivative forms contain no Pi-integral
and are continuous across z = r, z + r = 1 and z -> 0 (verified: the
one-sided limits agree with Limbdark's Cases 4/5/7/10 gradients), so no
Taylor switches are needed here — just the partial/complete branch and
the out-of-transit zero.
"""

from __future__ import annotations

import math

import mlx.core as mx

from .solution import sn_dev_with_aux

__all__ = ["flux_dev_analytic", "sn_partials"]

_PI = math.pi


def _unbroadcast(grad: mx.array, shape) -> mx.array:
    """Sum a full-shape gradient down to a (possibly broadcast) primal
    shape — mx.custom_function requires grads with the primal shapes."""
    if grad.shape == tuple(shape):
        return grad
    # sum leading extra axes
    while grad.ndim > len(shape):
        grad = mx.sum(grad, axis=0)
    for ax, n in enumerate(shape):
        if n == 1 and grad.shape[ax] != 1:
            grad = mx.sum(grad, axis=ax, keepdims=True)
    return grad


def sn_partials(z: mx.array, r, aux):
    """(ds0/dz, ds0/dr, ds1/dz, ds1/dr, ds2/dz, ds2/dr) from forward aux."""
    m_part = aux["m_part"]
    m_comp = aux["m_comp"]
    m_occ = mx.logical_or(m_part, m_comp)
    m_ps = aux["m_ps"]
    kap0 = aux["kap0"]
    kite = aux["kite"]
    E = aux["Eofk"]
    Em = aux["Em1mKdm"]
    onembmr2 = aux["onembmr2"]
    sqonembmr2 = aux["sqonembmr2"]
    sqbr = aux["sqbr"]

    zero = z * 0.0 + r * 0.0
    z = z + zero
    r = r + zero
    r2 = r * r
    z2 = z * z

    z_part = mx.where(m_part, z, 1.0)      # guarded divisions
    sqbr_ps = mx.where(m_ps, sqbr, 1.0)

    # ---- s0 --------------------------------------------------------------
    ds0dr = mx.where(m_part, -2.0 * r * kap0,
                     mx.where(m_comp, -2.0 * _PI * r, 0.0))
    ds0dz = mx.where(m_part, kite / z_part, 0.0)

    # ---- s1 (generic forms valid across all interior special lines) -----
    ds1dr_ps = -2.0 * r * onembmr2 * Em / sqbr_ps
    ds1dz_ps = (2.0 / 3.0) * r * onembmr2 * (2.0 * E - Em) / sqbr_ps
    ds1dr_cs = -4.0 * r * sqonembmr2 * E
    ds1dz_cs = -(4.0 / 3.0) * r * sqonembmr2 * (E - 2.0 * Em)
    ds1dr = mx.where(m_occ, mx.where(m_ps, ds1dr_ps, ds1dr_cs), 0.0)
    ds1dz = mx.where(m_occ, mx.where(m_ps, ds1dz_ps, ds1dz_cs), 0.0)

    # ---- s2 --------------------------------------------------------------
    ds2dr_part = -4.0 * r * kap0 + 8.0 * r * ((r2 + z2) * kap0 - kite)
    ds2dz_part = 2.0 * kite / z_part + (2.0 / z_part) * (
        4.0 * z2 * r2 * kap0 - (1.0 + r2 + z2) * kite
    )
    ds2dr_comp = -4.0 * _PI * r + 8.0 * _PI * r * (r2 + z2)
    ds2dz_comp = 8.0 * _PI * z * r2
    ds2dr = mx.where(m_part, ds2dr_part,
                     mx.where(m_comp, ds2dr_comp, 0.0))
    ds2dz = mx.where(m_part, ds2dz_part,
                     mx.where(m_comp, ds2dz_comp, 0.0))

    return ds0dz, ds0dr, ds1dz, ds1dr, ds2dz, ds2dr


@mx.custom_function
def flux_dev_analytic(z: mx.array, r: mx.array, u1: mx.array,
                      u2: mx.array) -> mx.array:
    """flux_dev with an analytic backward pass. Same contract as
    flux.flux_dev; inputs must be mx.arrays (broadcastable)."""
    s0d, s1d, s2d, _ = sn_dev_with_aux(z, r)
    g0 = 1.0 - u1 - 1.5 * u2
    g1 = u1 + 2.0 * u2
    g2 = -0.25 * u2
    norm = _PI * (1.0 - u1 / 3.0 - u2 / 6.0)
    return (g0 * s0d + g1 * s1d + g2 * s2d) / norm


@flux_dev_analytic.vjp
def _flux_dev_vjp(primals, cotangent, output):
    z, r, u1, u2 = primals
    ct = cotangent if isinstance(cotangent, mx.array) else cotangent[0]

    # forward-style recomputation (no tape)
    s0d, s1d, s2d, aux = sn_dev_with_aux(z, r)
    g0 = 1.0 - u1 - 1.5 * u2
    g1 = u1 + 2.0 * u2
    g2 = -0.25 * u2
    inv_norm = 1.0 / (_PI * (1.0 - u1 / 3.0 - u2 / 6.0))
    fdev = (g0 * s0d + g1 * s1d + g2 * s2d) * inv_norm

    ds0dz, ds0dr, ds1dz, ds1dr, ds2dz, ds2dr = sn_partials(z, r, aux)

    dfdz = (g0 * ds0dz + g1 * ds1dz + g2 * ds2dz) * inv_norm
    dfdr = (g0 * ds0dr + g1 * ds1dr + g2 * ds2dr) * inv_norm
    # d/du_n: dg/du contracted with s, plus the normalization term
    dfdu1 = (s1d - s0d) * inv_norm + fdev * (_PI / 3.0) * inv_norm
    dfdu2 = (-1.5 * s0d + 2.0 * s1d - 0.25 * s2d) * inv_norm \
        + fdev * (_PI / 6.0) * inv_norm

    return (
        _unbroadcast(ct * dfdz, z.shape),
        _unbroadcast(ct * dfdr, r.shape),
        _unbroadcast(ct * dfdu1, u1.shape),
        _unbroadcast(ct * dfdu2, u2.shape),
    )
