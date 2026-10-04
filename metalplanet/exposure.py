"""Exposure-time averaging by contact-split Gauss-Legendre quadrature.

A finite exposure measures the flux averaged over [t - dt/2, t + dt/2].
The usual fix (batman's, and ``TransitModel(supersample_factor=N)``) is
to average N uniformly spaced samples. That converges only as O(dt^2/N^2)
because the light curve is *not* smooth inside the window: its
derivative jumps at each contact, where the planet's limb crosses the
stellar limb.

ALFM19's recipe -- and Limbdark.jl's ``integrate_lightcurve`` -- is to
split the integration at the contact times and integrate each smooth
piece separately. There the pieces are handled by adaptive Simpson; an
adaptive depth is data-dependent control flow, which would break
batching and mx.compile, so here each piece gets a *fixed-order*
Gauss-Legendre rule instead. On a smooth piece an n-point rule is exact
for polynomials of degree 2n-1 and converges geometrically, so a handful
of nodes beats hundreds of uniform samples.

The split is done branchlessly: the window always has five
sub-intervals, with edges

    t1 <= clamp(tc1) <= clamp(tc2) <= clamp(tc3) <= clamp(tc4) <= t2

after clamping the four contact times into the window. Contacts outside
it collapse to a zero-width sub-interval that contributes nothing, so
grazing transits (two contacts), full transits (four) and windows that
straddle no contact at all are all the same code path -- and the node
positions stay differentiable in the parameters.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

__all__ = ["gauss_legendre", "contact_offsets", "contact_geometry",
           "contact_offsets_anchored", "exposure_nodes"]


def gauss_legendre(n: int):
    """(nodes, weights) on [-1, 1], float64 host-side."""
    x, w = np.polynomial.legendre.leggauss(int(n))
    return x.astype(np.float64), w.astype(np.float64)


def contact_offsets(r, a_sky, b):
    """Contact times relative to transit centre, as phase offsets
    phi = 2 pi (t - t0) / P.

    For a circular orbit z^2 = a^2 sin^2(phi) + b^2 cos^2(phi) with
    b = a cos i, so z = Z is solved exactly by

        sin^2(phi) = (Z^2 - b^2) / (a^2 - b^2).

    For an eccentric orbit the same expression is used with ``a_sky``
    set to the sky-speed-equivalent semi-major axis
    a (1 + e sin w) / sqrt(1 - e^2) and ``b`` the true impact parameter
    at conjunction. That linearises the contact, which is all the split
    needs: an error dtau in a split point leaves a residual kink of
    order dtau^3 in the quadrature, against the O(dtau) error of not
    splitting at all.

    Returns (phi_1, phi_2, phi_3, phi_4), ordered. A grazing transit
    (b > 1 - r) collapses the inner pair onto the transit centre, which
    the branchless five-interval split absorbs.
    """
    a2mb2 = mx.maximum(a_sky * a_sky - b * b, 1e-30)
    out = []
    for sign in (1.0, -1.0):          # Z = 1 + r (outer), 1 - r (inner)
        Z = 1.0 + sign * r
        s2 = (Z * Z - b * b) / a2mb2
        s = mx.sqrt(mx.clip(s2, 0.0, 1.0))
        out.append(mx.arcsin(s))
    phi_out, phi_in = out
    return -phi_out, -phi_in, phi_in, phi_out


def contact_geometry(a, ecc, esw, ci, sqrt=None, maximum=None):
    """(a_sky, b_conj) for ``contact_offsets``, from the orbital elements.

    ``a_sky`` is the sky-velocity-equivalent semi-major axis
    a (1 + e sin w) / sqrt(1 - e^2) and ``b_conj`` the impact parameter at
    inferior conjunction, a (1 - e^2) / (1 + e sin w) cos i. At e = 0 both
    reduce to a and a cos i.

    Written once and called from all three contact sites (the eager
    frontend, the compiled frontend branch and the batched path), which
    previously carried three copies that already differed in how they
    floored ``esw`` and ``beta``. Pass ``e sin w`` rather than w so the
    caller can build it however it likes -- h * sqrt(e) in the anchored
    parameterisation, e * sin(w) from elements -- and pass the sqrt /
    maximum for the flavour of array in play (math, numpy or mlx).
    """
    if sqrt is None:
        sqrt, maximum = mx.sqrt, mx.maximum
    beta = sqrt(maximum(1.0 - ecc * ecc, 1e-30))
    one_p = 1.0 + esw
    return a * one_p / beta, a * (1.0 - ecc * ecc) / one_p * ci


def exposure_nodes(t, t0, period, exp_time, contacts, n_gl: int,
                   dtype=mx.float64):
    """Quadrature nodes and weights for each exposure window.

    ``t`` are exposure mid-times, ``contacts`` the four phase offsets
    from ``contact_offsets``. Returns (times, weights), both with one
    extra trailing axis of length 5 * n_gl, the weights already
    normalised so that summing weight * flux gives the average. A leading
    batch axis is allowed throughout (many parameter sets at once), in
    which case t and the scalars broadcast against each other.
    """
    xg, wg = gauss_legendre(n_gl)
    xg = mx.array(xg, dtype=dtype)
    wg = mx.array(wg, dtype=dtype)

    half = 0.5 * exp_time
    t1 = t - half
    t2 = t + half
    # transit centre of the epoch each exposure belongs to
    n_ep = mx.round((t - t0) / period)
    tc_mid = t0 + n_ep * period
    scale = period / (2.0 * math.pi)

    edges = [t1]
    for phi in contacts:
        tc = tc_mid + phi * scale
        edges.append(mx.minimum(mx.maximum(tc, t1), t2))
    edges.append(t2)
    # enforce monotonicity after clamping (contacts are already ordered,
    # but clamping two of them to the same bound must not invert them)
    for i in range(1, len(edges)):
        edges[i] = mx.maximum(edges[i], edges[i - 1])

    times, weights = [], []
    inv_dt = 1.0 / exp_time
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mid = 0.5 * (lo + hi)
        halfw = 0.5 * (hi - lo)
        times.append(mid[..., None] + halfw[..., None] * xg)
        weights.append(halfw[..., None] * wg * inv_dt)
    axis = times[0].ndim - 1
    W = mx.concatenate(weights, axis=axis)
    # The sub-interval widths sum to the exposure only up to rounding in
    # the clamped edges, which left the out-of-transit flux at 1 + 2e-14
    # instead of exactly 1. Renormalising restores that contract (the
    # weights are a partition of unity by construction, so this only
    # removes round-off).
    W = W / mx.sum(W, axis=axis, keepdims=True)
    return mx.concatenate(times, axis=axis), W


def contact_offsets_anchored(r, a, b, k, h, ci, n_iter: int = 4):
    """Exact contact phases of the transit-anchored eccentric orbit.

    ``contact_offsets`` with ``contact_geometry``'s sky-equivalent a is a
    linearisation: measured, it misplaces a contact by up to 2.3e-3 d on a
    grazing e = 0.5 orbit. A split that misses the kink costs the
    quadrature accuracy (20x there at n_gl = 5) and -- worse for a sampler
    -- breaks the frozen-split gradient, whose integrand dF/dtheta then
    jumps *inside* a Gauss-Legendre piece (~1% on d/dperiod at e = 0.7).

    So: start from the linearisation and take ``n_iter`` Newton steps on
    z(phi)^2 = (1 +- r)^2 through the anchored solve itself, with

        d(z^2)/dphi = 2 (u du/ddelta + ci^2 v dv/ddelta) / D,
        D = 1 + es sin delta - ec cos delta = 1 - e cos E > 0.

    The linear start is within ~1e-3 in phase, so four steps reach
    round-off. A contact with no root -- the inner pair of a grazing
    transit, everything when b >= 1 + r -- keeps its collapsed linearised
    value, and a step may neither change a contact's sign nor exceed half
    the linearised outer phase. (k, h) = (secosw, sesinw), ci = cos i.
    Returns (phi_1, phi_2, phi_3, phi_4), ordered. Callers detach them.
    """
    from .anchored import _one_minus_cos, anchor_constants, solve_sincos_delta
    e, ecw, esw, es, ec, B1, A2, B2 = anchor_constants(k, h)
    a_sky, _ = contact_geometry(a, e, esw, ci)
    lin = contact_offsets(r, a_sky, b)
    lim = 0.5 * mx.abs(lin[3]) + 1e-12
    out = []
    for phi0, Z, sgn in zip(lin, (1.0 + r, 1.0 - r, 1.0 - r, 1.0 + r),
                            (-1.0, -1.0, 1.0, 1.0)):
        exists = b < Z
        phi = phi0
        for _ in range(int(n_iter)):
            sd, cd = solve_sincos_delta(phi, es, ec)
            omc = _one_minus_cos(sd, cd)
            u = a * (-ecw * omc - B1 * sd)
            v = a * (A2 * cd - B2 * sd - esw)
            dud = a * (-ecw * sd - B1 * cd)
            dvd = a * (-A2 * sd - B2 * cd)
            D = 1.0 + es * sd - ec * cd
            g = u * u + (v * ci) ** 2 - Z * Z
            dg = 2.0 * (u * dud + ci * ci * v * dvd) / D
            ok = mx.logical_and(exists, mx.abs(dg) > 1e-30)
            step = mx.where(ok, g / mx.where(ok, dg, mx.ones_like(dg)), 0.0)
            phi = phi - mx.clip(step, -lim, lim)
            # a root on the other side of conjunction is the wrong contact
            phi = (mx.minimum(phi, 0.0) if sgn < 0
                   else mx.maximum(phi, 0.0))
        out.append(mx.where(exists, phi, phi0))
    # Newton keeps each contact on its own side, but near-grazing inner
    # roots can still cross; the edge clamp downstream needs them ordered
    for i in range(1, 4):
        out[i] = mx.maximum(out[i], out[i - 1])
    return tuple(out)
