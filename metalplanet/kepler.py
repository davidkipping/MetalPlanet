"""Kepler solvers and eccentric-orbit separations.

Production path: ``kepler(M, e) -> (sin f, cos f)`` — the Markley (1995)
non-iterative scheme as used by jaxoplanet (cubic-equation starter
accurate to ~1e-4, then ONE Nijenhuis/Markley fifth-order refinement),
with two MetalPlanet-specific improvements:

* a single sincos evaluation total: sin/cos of the refined anomaly are
  reconstructed from sin/cos of the *starter* anomaly via a Taylor
  rotation by the (small) final correction dE, instead of a second trig
  call — and the eccentric-anomaly -> true-anomaly conversion is a
  rational half-angle form with no tan() and no overflow at E ~ pi:
      fac^2 = (1+e)/(1-e), A = fac^2 (1-cosE)/2, B = (1+cosE)/2,
      sinf = fac sinE / (A+B),  cosf = (B-A)/(A+B),  A+B > 0 always;
* gradients via the implicit function theorem (`mx.custom_function`,
  transposing jaxoplanet's JVP) — nothing differentiates through the
  solve.

``kepler_E`` (Danby starter + fixed Halley iterations, returning E) is
kept for verification and back-compat; it is no longer the production
path.

Geometry convention (matches batman): the primary transit (inferior
conjunction) is at true anomaly f = pi/2 - w; ``t0`` is the transit time,
converted to periastron passage via ``mean_anomaly_offset_at_transit``.
"""

from __future__ import annotations

import math

import mlx.core as mx

from .ellip import dtype_eps
from .trig import sincos

__all__ = ["kepler", "kepler_E", "separation_keplerian",
           "mean_anomaly_offset_at_transit"]

_TWO_PI = 2.0 * math.pi
_PI = math.pi
# Markley starter constants: 3 pi / (pi - 6/pi) and 1.6 / (pi - 6/pi)
_MK_A = 3.0 * math.pi / (math.pi - 6.0 / math.pi)
_MK_B = 1.6 / (math.pi - 6.0 / math.pi)


def _unbroadcast(grad: mx.array, shape) -> mx.array:
    if grad.shape == tuple(shape):
        return grad
    while grad.ndim > len(shape):
        grad = mx.sum(grad, axis=0)
    for ax, n in enumerate(shape):
        if n == 1 and grad.shape[ax] != 1:
            grad = mx.sum(grad, axis=ax, keepdims=True)
    return grad


@mx.custom_function
def kepler(M: mx.array, e) -> tuple[mx.array, mx.array]:
    """(sin f, cos f) for mean anomaly M and eccentricity 0 <= e < 1.

    Markley starter + one fifth-order refinement; exactly one sincos
    evaluation per point; residual at working-precision roundoff for
    e <= 0.95 (verified in tests). Gradients come from the implicit
    function theorem, not from differentiating the solve.
    """
    zero = M * 0.0 + e * 0.0
    M = M + zero
    e = e + zero

    # wrap to [-pi, pi], solve on |M| in [0, pi] (E is odd in M)
    M = M - _TWO_PI * mx.round(M / _TWO_PI)
    sign = mx.where(M >= 0.0, 1.0 + zero, -1.0 + zero)
    Ma = mx.abs(M)

    # ---- Markley (1995) cubic starter (pure algebra; ~1e-4 accurate) ----
    ome = 1.0 - e
    M2 = Ma * Ma
    alpha = _MK_A + _MK_B * (_PI - Ma) / (1.0 + e)
    d = 3.0 * ome + alpha * e
    alphad = alpha * d
    rr = (3.0 * alphad * (d - ome) + M2) * Ma
    q = 2.0 * alphad * ome - M2
    q2 = q * q
    # w = cbrt(|r| + sqrt(q^3 + r^2))^2, computed as exp(2/3 log(.));
    # the argument is >= q^3 > 0 for e < 1, and the starter only needs
    # ~1e-4 so MLX's transcendental accuracy is irrelevant here
    c = mx.abs(rr) + mx.sqrt(mx.maximum(q2 * q + rr * rr, 1e-300))
    w = mx.exp((2.0 / 3.0) * mx.log(c))
    E = (2.0 * rr * w / (w * w + w * q + q2) + Ma) / d

    # ---- one fifth-order (Nijenhuis/Markley) correction -----------------
    sE_raw, cE_raw = sincos(E)          # the ONLY trig call
    sE = E - sE_raw                     # E - sin E  (conditioned small-E)
    cE = 1.0 - cE_raw                   # 1 - cos E
    f_0 = e * sE + E * ome - Ma
    f_1 = e * cE + ome                  # 1 - e cos E  >= 1 - e > 0
    f_2 = e * (E - sE)                  # e sin E
    f_3 = 1.0 - f_1                     # e cos E
    d_3 = -f_0 / (f_1 - 0.5 * f_0 * f_2 / f_1)
    d_4 = -f_0 / (f_1 + 0.5 * d_3 * f_2 + d_3 * d_3 * f_3 / 6.0)
    dE = -f_0 / (f_1 + 0.5 * d_4 * f_2 + d_4 * d_4 * f_3 / 6.0
                 - d_4 * d_4 * d_4 * f_2 / 24.0)

    # sin/cos of E + dE via a Taylor rotation (|dE| <~ 1e-3: the dE^5
    # truncation is ~1e-17, below either working precision)
    dE2 = dE * dE
    sdE = dE * (1.0 - dE2 / 6.0 * (1.0 - dE2 / 20.0))
    cdE = 1.0 - dE2 * 0.5 * (1.0 - dE2 / 12.0)
    sinE = (sE_raw * cdE + cE_raw * sdE) * sign
    cosE = cE_raw * cdE - sE_raw * sdE

    # ---- true anomaly, tan-free rational half-angle form ----------------
    fac = mx.sqrt((1.0 + e) / mx.maximum(1.0 - e, 1e-12))
    A = fac * fac * (1.0 - cosE) * 0.5
    B = (1.0 + cosE) * 0.5
    Dinv = 1.0 / (A + B)                # A + B >= (1 - |cosE|)/2 ... > 0
    sinf = fac * sinE * Dinv
    cosf = (B - A) * Dinv
    return sinf, cosf


@kepler.vjp
def _kepler_vjp(primals, cotangents, outputs):
    M, e = primals
    ct_s, ct_c = cotangents
    sinf, cosf = outputs
    # implicit function theorem (transpose of jaxoplanet's JVP):
    # df = dM (1+e cosf)^2/(1-e^2)^{3/2} + de (2+e cosf) sinf/(1-e^2)
    g = ct_s * cosf - ct_c * sinf
    ecosf = e * cosf
    ome2 = mx.maximum(1.0 - e * e, 1e-12)
    dM = g * (1.0 + ecosf) ** 2 / (ome2 * mx.sqrt(ome2))
    de = g * (2.0 + ecosf) * sinf / ome2
    M_shape = M.shape if isinstance(M, mx.array) else ()
    e_shape = e.shape if isinstance(e, mx.array) else ()
    return _unbroadcast(dM, M_shape), _unbroadcast(de, e_shape)


def kepler_E(M: mx.array, e, n_iter: int = 5) -> mx.array:
    """Eccentric anomaly E(M, e), Danby starter + fixed Halley steps.

    Verification/back-compat path (the production solver is ``kepler``).
    """
    M = M - _TWO_PI * mx.round(M / _TWO_PI)
    sM, _ = sincos(M)
    sign = mx.where(sM >= 0.0, 1.0 + 0.0 * sM, -1.0 + 0.0 * sM)
    E = M + 0.85 * e * sign
    for _ in range(n_iter):
        sE, cE = sincos(E)
        f0 = E - e * sE - M
        f1 = 1.0 - e * cE          # >= 1 - e > 0
        f2 = e * sE
        denom = 2.0 * f1 * f1 - f0 * f2
        E = E - 2.0 * f0 * f1 / denom
    return E


def true_anomaly(E: mx.array, e) -> mx.array:
    """f(E, e) via the half-angle identity (verification path)."""
    sE2, cE2 = sincos(0.5 * E)
    fac = mx.sqrt(mx.maximum((1.0 + e) / mx.maximum(1.0 - e, 1e-12), 0.0))
    return 2.0 * mx.arctan2(fac * sE2, cE2)


def mean_anomaly_offset_at_transit(e, w):
    """M at inferior conjunction (f = pi/2 - w), host-side floats.

    Used to convert t0 (transit time) to periastron passage:
    M(t) = 2 pi (t - t0)/P + M_transit.
    """
    f0 = 0.5 * math.pi - w
    E0 = 2.0 * math.atan2(
        math.sqrt(max(1.0 - e, 0.0)) * math.sin(0.5 * f0),
        math.sqrt(1.0 + e) * math.cos(0.5 * f0),
    )
    return E0 - e * math.sin(E0)


def _sincos_any(x):
    """sincos for a python float or mx scalar/array."""
    if isinstance(x, mx.array):
        return sincos(x)
    return math.sin(x), math.cos(x)


def separation_keplerian(M: mx.array, e, a, inc, w, n_iter: int = 5):
    """(z, front) for a Keplerian orbit.

    M: mean anomaly from periastron (see mean_anomaly_offset_at_transit).
    e, a (stellar radii), inc, w in radians — arrays or floats.

    z = r_orb sqrt(1 - sin^2(w+f) sin^2 i), with the conic form
    r_orb = a (1-e^2)/(1+e cos f); ``front`` is True where a primary
    transit can occur (sin(w+f) > 0). ``n_iter`` is accepted for
    back-compat and unused (the Markley solve is non-iterative).
    """
    eps = dtype_eps(M.dtype)
    sinf, cosf = kepler(M, e)
    sw, cw = _sincos_any(w)
    swf = sw * cosf + cw * sinf
    si = _sincos_any(inc)[0]
    r_orb = a * (1.0 - e * e) / (1.0 + e * cosf)
    arg = 1.0 - (swf * si) ** 2
    z = r_orb * mx.sqrt(mx.maximum(arg, (10.0 * eps) ** 2))
    return z, swf > 0.0
