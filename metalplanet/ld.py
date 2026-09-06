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

__all__ = ["q_to_u", "q_to_u_np", "u_to_q_np"]


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
