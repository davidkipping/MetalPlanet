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

__all__ = ["gauss_legendre", "contact_offsets", "exposure_nodes"]


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


def exposure_nodes(t, t0, period, exp_time, contacts, n_gl: int,
                   dtype=mx.float64):
    """Quadrature nodes and weights for each exposure window.

    ``t`` are exposure mid-times (m,), ``contacts`` the four phase
    offsets from ``contact_offsets``. Returns (times, weights), both
    (m, 5 * n_gl), with the weights already normalised by the exposure
    time so that summing weight * flux gives the average.
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
        times.append(mid[:, None] + halfw[:, None] * xg[None, :])
        weights.append(halfw[:, None] * wg[None, :] * inv_dt)
    return mx.concatenate(times, axis=1), mx.concatenate(weights, axis=1)
