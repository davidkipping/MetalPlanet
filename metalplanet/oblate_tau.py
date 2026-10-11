"""Oblate planets on an orbit: sky positions, contacts, exposures.

``metal.flux_dev_from_tau(..., f=, theta=)`` delegates here. The light
curve is ``oblate.shape_cols_oblate`` at the planet's sky position, put in
its principal frame, and integrated over each exposure as the spherical
path does; this module adds only what the orbit and the exposure need.

**Sky frame.** X along the orbit (the direction of motion at transit), Y
across, both in stellar radii, with SquishierPlanet's convention (its
``jaxlc.sky_circular`` and ``orbit.sky_position`` with Omega = pi, pinned
by a test):

    circular:   X = a sin phi,   Y = -b cos phi,   in front iff cos phi > 0
    eccentric:  X = -u,          Y = -v cos i,     in front iff v > 0

with (u, v) the transit-anchored orbit's sky-plane Cartesians
(``anchored.py``; at e = 0 they reduce to the circular ones). theta is the
sky angle of the planet's long axis from X, and
``oblate.principal_frame(X, Y, theta)`` gives the centre in the planet's
own frame. The far side is replaced by a fixed point well clear of the
star, so it is exactly 0 with zero gradient, as the spherical path's
pushed separation is.

**Contacts, per side.** A tilted ellipse's ingress and egress differ, and a
grazing one can have both outer contacts on one side of conjunction, so
nothing here is mirrored. Along the transit chord (a straight line to
within (duration / period)^2) two functions of phase are convex:

    q(phi) = (distance from the star's centre to the planet's disc)^2
    M(phi) = (largest distance from it to the planet's boundary)^2

The outer contacts are q = 1 and the inner ones M = 1. Newton's method on
a convex function, started on the outside of a root, approaches it
monotonically and never overshoots; so the outer contacts start from the
circumscribed (radius A) circle's contacts, where q >= 1, and the inner
ones from the outer contacts, where M >= 1. The derivatives come from the
envelope theorem: dq/dphi = 2 P* . dc/dphi at the extremal boundary point
P*. Contacts are quadrature split points only -- interior points of a
continuous integrand, detached as the spherical ones are -- so a solve
that has not converged (a near-tangent graze, whose dip is tiny) costs
quadrature accuracy there, never a wrong model. A contact that does not
exist collapses onto a point inside the transit.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from . import metal as M
from .dtypes import fp64_on_cpu
from .hybrid import combine_cols, get_law
from .oblate import _check_domain, principal_frame, shape_cols_oblate
from .trig import sincos

__all__ = ["sky_circular", "sky_anchored", "contact_offsets_oblate",
           "flux_dev_from_tau_oblate"]

_TWO_PI = 2.0 * math.pi
_N_PSI = 16              # boundary samples per extremum search
_N_PSI_NEWTON = 3        # Newton steps polishing the sampled extremum
_N_CONTACT = {mx.float64: 14, mx.float32: 10}     # Newton steps in phase
_FAR = 4.0               # far-side stand-in, |(X, Y)| > 1 + A for A < 1


# --------------------------------------------------------------------------
# sky positions
# --------------------------------------------------------------------------

def sky_circular(phi, a, b):
    """(X, Y, dX/dphi, dY/dphi, front) on a circular orbit."""
    s, c = sincos(phi)
    return a * s, -(b * c), a * c, b * s, c > 0.0


def sky_anchored(phi, a, ci, consts):
    """(X, Y, dX/dphi, dY/dphi, front) on the transit-anchored eccentric
    orbit; ``consts`` = ``anchored.anchor_constants(k, h)``. The lines are
    ``anchored.separation_anchored``'s, kept separate so that its compiled
    graphs are untouched."""
    from .anchored import _one_minus_cos, solve_sincos_delta
    e, ecw, esw, es, ec, B1, A2, B2 = consts
    sd, cd = solve_sincos_delta(phi, es, ec)
    omc = _one_minus_cos(sd, cd)
    u = a * (-ecw * omc - B1 * sd)
    v = a * (A2 * cd - B2 * sd - esw)
    D = 1.0 + es * sd - ec * cd            # d delta / d phi = 1 / D
    du = a * (-ecw * sd - B1 * cd) / D
    dv = a * (-A2 * sd - B2 * cd) / D
    return -u, -(v * ci), -du, -(dv * ci), v > 0.0


def _sky(phi, a, b, ecc):
    """``ecc`` is None (circular) or (ci, anchor constants)."""
    if ecc is None:
        return sky_circular(phi, a, b)
    ci, consts = ecc
    return sky_anchored(phi, a, ci, consts)


# --------------------------------------------------------------------------
# contacts
# --------------------------------------------------------------------------

def _boundary_ext(x0, y0, A, B, kind):
    """(s*, psi*) with s(psi) = (x0 + A cos psi)^2 + (y0 + B sin psi)^2 at
    its minimum (kind = -1) or maximum (+1) over the boundary: the best of
    _N_PSI samples, polished by Newton on ds/dpsi."""
    dt = x0.dtype
    grid = mx.array(np.linspace(0.0, _TWO_PI, _N_PSI, endpoint=False), dtype=dt)
    xs, ys = x0[..., None], y0[..., None]
    sg, cg = sincos(grid)
    sv = (xs + A[..., None] * cg) ** 2 + (ys + B[..., None] * sg) ** 2
    k = mx.argmax(sv, axis=-1) if kind > 0 else mx.argmin(sv, axis=-1)
    psi = mx.take(grid, k)
    for _ in range(_N_PSI_NEWTON):
        sp, cp = sincos(psi)
        xx, yy = x0 + A * cp, y0 + B * sp
        d1 = 2.0 * (-xx * A * sp + yy * B * cp)
        d2 = 2.0 * (A * A * sp * sp - xx * A * cp + B * B * cp * cp - yy * B * sp)
        # only a step toward the extremum sought: d2 must have its sign
        ok = (d2 * kind) < 0.0
        psi = psi - mx.where(ok, d1 / mx.where(ok, d2, 1.0), 0.0)
    sp, cp = sincos(psi)
    return (x0 + A * cp) ** 2 + (y0 + B * sp) ** 2, psi


def _q_and_slope(phi, a, b, ecc, A, B, cth, sth, kind):
    """q (kind = -1: squared distance to the disc, 0 inside it) or M
    (kind = +1: squared largest boundary distance) at phase phi, and its
    phase derivative."""
    X, Y, dX, dY, _ = _sky(phi, a, b, ecc)
    x0, y0 = X * cth + Y * sth, -X * sth + Y * cth
    dx0, dy0 = dX * cth + dY * sth, -dX * sth + dY * cth
    s, psi = _boundary_ext(x0, y0, A, B, kind)
    sp, cp = sincos(psi)
    slope = 2.0 * ((x0 + A * cp) * dx0 + (y0 + B * sp) * dy0)
    if kind < 0:
        inside = (x0 / A) ** 2 + (y0 / B) ** 2 < 1.0
        s = mx.where(inside, 0.0, s)
        slope = mx.where(inside, 0.0, slope)
    return s, slope


def _newton_contact(phi, a, b, ecc, A, B, cth, sth, kind, side, lim, n):
    """Newton on g(phi) - 1 = 0 from phi, on the outside of the root
    (side -1: ingress, the root lies at larger phi; +1: egress)."""
    for _ in range(n):
        g, dg = _q_and_slope(phi, a, b, ecc, A, B, cth, sth, kind)
        # outside the root the slope points away from it: dg * side > 0
        ok = (dg * side) > 0.0
        step = mx.where(ok, (g - 1.0) / mx.where(ok, dg, 1.0), 0.0)
        phi = phi - mx.clip(step, -lim, lim)
    g, _ = _q_and_slope(phi, a, b, ecc, A, B, cth, sth, kind)
    return phi, g


def contact_offsets_oblate(a, b, r, f, theta, k=None, h=None, ci=None):
    """The four contact phases (ingress outer, ingress inner, egress inner,
    egress outer) of an oblate planet, as offsets from mid-transit
    (phi = 2 pi (t - t0) / P), per chain and detached; ordered.

    a, b, r, f, theta: (n,) arrays of one dtype. Give (k, h, ci) for the
    transit-anchored eccentric orbit. A missing contact collapses onto a
    point inside the transit (the inner pair of a graze), or all four onto
    one point (no transit)."""
    from .exposure import contact_offsets, contact_offsets_anchored
    a, b, r, f, theta = (mx.stop_gradient(v) for v in (a, b, r, f, theta))
    dt = a.dtype
    A = r / mx.sqrt(1.0 - f)
    B = A * (1.0 - f)
    if k is None:
        ecc = None
        cA = contact_offsets(A, a, b)
    else:
        from .anchored import anchor_constants
        k, h, ci = (mx.stop_gradient(v) for v in (k, h, ci))
        consts = anchor_constants(k, h)
        ecc = (ci, consts)
        cA = contact_offsets_anchored(A, a, b, k, h, ci, consts=consts)
    sth, cth = sincos(theta)
    lo, hi = cA[0], cA[3]
    lim = mx.maximum(hi - lo, 1e-12)
    n = _N_CONTACT[dt]
    tol = 1e-8 if dt == mx.float64 else 1e-4

    c1, g1 = _newton_contact(lo, a, b, ecc, A, B, cth, sth, -1, -1.0, lim, n)
    c4, g4 = _newton_contact(hi, a, b, ecc, A, B, cth, sth, -1, 1.0, lim, n)
    outer = (mx.abs(g1 - 1.0) < tol) & (mx.abs(g4 - 1.0) < tol) & (c1 < c4)
    mid = 0.5 * (lo + hi)
    c1, c4 = mx.where(outer, c1, mid), mx.where(outer, c4, mid)

    lim_i = mx.maximum(c4 - c1, 1e-12)
    c2, g2 = _newton_contact(c1, a, b, ecc, A, B, cth, sth, 1, -1.0, lim_i, n)
    c3, g3 = _newton_contact(c4, a, b, ecc, A, B, cth, sth, 1, 1.0, lim_i, n)
    inner = (outer & (mx.abs(g2 - 1.0) < tol) & (mx.abs(g3 - 1.0) < tol)
             & (c1 <= c2) & (c2 < c3) & (c3 <= c4))
    mid = 0.5 * (c1 + c4)
    c2, c3 = mx.where(inner, c2, mid), mx.where(inner, c3, mid)
    return tuple(mx.stop_gradient(c) for c in (c1, c2, c3, c4))


# --------------------------------------------------------------------------
# the graph
# --------------------------------------------------------------------------

def _tau_graph(tau, period, a, b, r, f, theta, w2d, exp_time, mode, n_gl,
               n_sub, law, basis=False, k=None, h=None):
    """The oblate counterpart of ``metal_hybrid._tau_graph``: the same
    exposure rules, contacts detached and the period detached in the node
    map exactly as there."""
    from .exposure import exposure_nodes
    law = get_law(law)
    ecc = k is not None
    pars = (period, a, b, r, f, theta) + ((k, h) if ecc else ())

    def col(nk):
        return [mx.reshape(p, (-1,) + (1,) * nk) for p in pars]

    def orbit(cols):
        per, av, bv = cols[:3]
        if not ecc:
            return per, av, bv, None
        from .anchored import anchor_constants
        kv, hv = cols[-2:]
        ci = M._ecc_shape(av, bv, kv, hv)[2]
        return per, av, bv, (ci, anchor_constants(kv, hv))

    def inst(tt, nk):
        cols = col(nk)
        per, av, bv, eo = orbit(cols)
        rv, fv, thv = cols[3:6]
        X, Y, _, _, front = _sky((_TWO_PI / per) * tt, av, bv, eo)
        X = mx.where(front, X, _FAR)
        Y = mx.where(front, Y, 0.0)
        sth, cth = sincos(thv)
        x0, y0 = X * cth + Y * sth, -X * sth + Y * cth
        return shape_cols_oblate(x0, y0, rv, fv, law)

    if mode == M._INT_NONE or exp_time == 0.0:
        Bc = inst(tau, 1)
    elif mode == M._INT_SUPER:
        half = 0.5 * exp_time
        off = (np.linspace(-half, half, int(n_sub)) if n_sub > 1
               else np.zeros(1))
        nodes = tau[..., None] + mx.array(off, dtype=tau.dtype)
        Bc = mx.mean(inst(nodes, 2), axis=-2)
    else:
        per1 = col(1)[0]
        if ecc:
            ci = M._ecc_shape(a, b, k, h)[2]
            cs = contact_offsets_oblate(a, b, r, f, theta, k, h, ci)
        else:
            cs = contact_offsets_oblate(a, b, r, f, theta)
        cs = tuple(mx.reshape(c, (-1, 1)) for c in cs)
        T, W = exposure_nodes(tau, tau * 0.0, mx.stop_gradient(per1),
                              exp_time, cs, int(n_gl), dtype=tau.dtype)
        Bc = mx.sum(inst(T, 2) * W[..., None], axis=-2)
    if basis:
        return Bc
    return combine_cols(Bc, w2d, law)


@fp64_on_cpu
def flux_dev_from_tau_oblate(tau, period, a, b, r, f, theta, law, u,
                             exp_time, mode, n_gl, n_sub, basis=False,
                             k=None, h=None):
    """``metal.flux_dev_from_tau`` with ``f=`` (that function validates the
    keywords and delegates here). fp32 data on the GPU stream takes the
    oblate Metal kernels (``metal_oblate``); anything else runs this
    graph, fp64 on the CPU stream."""
    from .metal_hybrid import _canon_w
    law = get_law(law)
    if tau.ndim not in (1, 2):
        raise ValueError(f"tau must be (m,) or (n, m); got {tau.shape}")
    _check_domain(r, f)
    squeeze = tau.ndim == 1
    tau2d = tau[None, :] if squeeze else tau
    ecc = k is not None
    params = (period, a, b, r, f, theta) + ((k, h) if ecc else ())
    n_param = max((p.shape[0] if isinstance(p, mx.array) and p.ndim >= 1
                   else 1) for p in params)
    if not basis and isinstance(u, mx.array) and u.ndim == 2:
        n_param = max(n_param, u.shape[0])
    n = max(tau2d.shape[0], n_param)
    pc = [M._canon_param(p, n, tau2d.dtype) for p in params]
    if tau2d.shape[0] != n:
        tau2d = mx.broadcast_to(tau2d, (n, tau2d.shape[1]))
    kh = pc[-2:] if ecc else [None, None]
    w2d = (None if basis else
           _canon_w(u, n, law.n_w, tau2d.dtype, law.name))
    if (tau2d.dtype == mx.float32 and M._gpu_stream_active()
            and M.metal_available()):
        from .metal_oblate import flux_dev_from_tau_oblate_kernel
        from .oblate import _in_domain
        out = flux_dev_from_tau_oblate_kernel(
            tau2d, *pc[:6], w2d, exp_time, mode, n_gl, n_sub, law,
            basis=basis, k=kh[0], h=kh[1])
        ok = _in_domain(pc[3], pc[4])[:, None]
        out = mx.where(ok[..., None] if basis else ok, out, mx.nan)
    else:
        out = _tau_graph(tau2d, *pc[:6], w2d, exp_time, mode, n_gl, n_sub,
                         law, basis=basis, k=kh[0], h=kh[1])
    return out[0] if squeeze and n == 1 else out
