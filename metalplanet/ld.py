"""Limb-darkening reparameterizations.

Kipping (2013) triangular sampling for quadratic limb darkening: the unit
box (q1, q2) in [0,1]^2 maps exactly onto the physically-allowed
(u1, u2) region (I(mu) everywhere positive, monotonically darkening):

    u1 = 2 sqrt(q1) q2
    u2 = sqrt(q1) (1 - 2 q2)

Sample q1, q2 with flat bounded ParamSpecs and apply q_to_u *inside* the
model wrapper (MLX ops, differentiable); the box-to-box Jacobian is
handled by the ParamSpec sigmoid transform as usual.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

__all__ = ["q_to_u", "q_to_u_np", "u_to_q_np",
           "hybrid2_vertices", "hybrid2_from_q", "hybrid2_from_q_np",
           "hybrid2_to_q_np", "simplex_from_q", "simplex_from_q_np",
           "q_from_simplex_np"]


def q_to_u(q1: mx.array, q2: mx.array):
    """(q1, q2) in [0,1]^2 -> (u1, u2), MLX ops (differentiable).

    q1 is clamped away from 0 so sqrt has a finite gradient everywhere
    (q1 -> 0 is the no-limb-darkening corner; the clamp is far below any
    posterior mass in practice).
    """
    sq1 = mx.sqrt(mx.maximum(q1, 1e-12))
    u1 = 2.0 * sq1 * q2
    u2 = sq1 * (1.0 - 2.0 * q2)
    return u1, u2


def q_to_u_np(q1, q2):
    """Float64 host-side replica of q_to_u."""
    sq1 = np.sqrt(np.maximum(np.asarray(q1, dtype=np.float64), 1e-12))
    q2 = np.asarray(q2, dtype=np.float64)
    return 2.0 * sq1 * q2, sq1 * (1.0 - 2.0 * q2)


def u_to_q_np(u1, u2):
    """Inverse map (host-side, for building initial states from u guesses).

    q1 = (u1 + u2)^2, q2 = u1 / (2 (u1 + u2)).
    """
    u1 = np.asarray(u1, dtype=np.float64)
    u2 = np.asarray(u2, dtype=np.float64)
    s = u1 + u2
    return s * s, 0.5 * u1 / s


# ---------------------------------------------------------------------------
# hybrid laws (metalplanet/hybrid.py): uniform priors on their physical
# regions, after SquishierPlanet's laws.py
# ---------------------------------------------------------------------------

def hybrid2_vertices(eps=None):
    """(V_c, V_l) of hybrid2's exact physical triangle in (w1, w2); the
    third vertex is the origin (uniform disc). With g0, g1 the pole shape's
    slopes in xi = mu^2 at the limb and the centre, V_c has zero limb
    intensity and zero central slope, V_l zero limb intensity and zero
    limb slope. Float64 numpy."""
    from .hybrid import HYBRID2_EPS
    e = HYBRID2_EPS if eps is None else float(eps)
    N = e ** -2 - (1.0 + e) ** -2
    g0 = 2.0 / (N * e ** 3)
    g1 = 2.0 / (N * (1.0 + e) ** 3)
    Vc = np.array([-g1 / (1.0 - g1), 1.0 / (1.0 - g1)])
    Vl = np.array([g0 / (g0 - 1.0), -1.0 / (g0 - 1.0)])
    return Vc, Vl


def hybrid2_from_q(q1: mx.array, q2: mx.array):
    """(q1, q2) in [0,1]^2 -> (w1, w2) uniform on hybrid2's exact triangle,
    MLX ops (differentiable): w = sqrt(q1) [(1 - q2) V_c + q2 V_l]. q1 is
    clamped away from 0 as ``q_to_u`` does."""
    Vc, Vl = hybrid2_vertices()
    s = mx.sqrt(mx.maximum(q1, 1e-12))
    w1 = s * ((1.0 - q2) * float(Vc[0]) + q2 * float(Vl[0]))
    w2 = s * ((1.0 - q2) * float(Vc[1]) + q2 * float(Vl[1]))
    return w1, w2


def hybrid2_from_q_np(q1, q2):
    """Float64 host-side replica of hybrid2_from_q."""
    Vc, Vl = hybrid2_vertices()
    s = np.sqrt(np.maximum(np.asarray(q1, dtype=np.float64), 1e-12))
    q2 = np.asarray(q2, dtype=np.float64)
    return (s * ((1.0 - q2) * Vc[0] + q2 * Vl[0]),
            s * ((1.0 - q2) * Vc[1] + q2 * Vl[1]))


def hybrid2_to_q_np(w1, w2):
    """Inverse of hybrid2_from_q_np: solve w = s V_c + (s q2)(V_l - V_c)."""
    Vc, Vl = hybrid2_vertices()
    M = np.stack([Vc, Vl - Vc], axis=1)
    w = np.stack([np.asarray(w1, dtype=np.float64),
                  np.asarray(w2, dtype=np.float64)], axis=-1)
    sol = np.linalg.solve(M, w[..., None])[..., 0]
    s, sq2 = sol[..., 0], sol[..., 1]
    return s * s, sq2 / s


def simplex_from_q(q: mx.array) -> mx.array:
    """q (..., n) in [0,1]^n -> w (..., n) uniform on the simplex
    {w_j >= 0, sum w_j <= 1} (hybrid4/5's physical region), by stick-
    breaking a flat Dirichlet: w_j = (1 - q_j^(1/(n-j))) prod_{i<j}
    q_i^(1/(n-i)). MLX ops; q clamped away from 0 for a finite gradient."""
    n = q.shape[-1]
    q = mx.maximum(q, 1e-12)
    stick = 1.0 + 0.0 * q[..., 0]
    ws = []
    for j in range(n):
        f = mx.power(q[..., j], 1.0 / (n - j))
        ws.append((1.0 - f) * stick)
        stick = stick * f
    return mx.stack(ws, axis=-1)


def simplex_from_q_np(q):
    """Float64 host-side replica of simplex_from_q."""
    q = np.maximum(np.asarray(q, dtype=np.float64), 1e-12)
    n = q.shape[-1]
    w = np.empty_like(q)
    stick = np.ones(q.shape[:-1])
    for j in range(n):
        f = q[..., j] ** (1.0 / (n - j))
        w[..., j] = (1.0 - f) * stick
        stick = stick * f
    return w


def q_from_simplex_np(w):
    """Inverse of simplex_from_q_np (host-side, for initial states)."""
    w = np.asarray(w, dtype=np.float64)
    n = w.shape[-1]
    q = np.empty_like(w)
    stick = np.ones(w.shape[:-1])
    for j in range(n):
        f = 1.0 - w[..., j] / stick
        q[..., j] = f ** (n - j)
        stick = stick * f
    return q
