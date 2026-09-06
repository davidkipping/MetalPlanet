"""Orbit module: epoch-centered times -> sky-projected separation z(t).

v1 is a circular orbit parameterized by (t0_off, p_off, b, a) with
a = a/R*, the scaled semi-major axis. Why a/R* rather than a duration
variant: it is the native geometric quantity in z(t) — no extra nonlinear
map (and its conditioning) inside the float32 graph — and duration- or
density-based parameterizations are exact host-side wrappers at the
ParamSpec level, not model changes.

Float32 conditioning: times enter as per-orbit residuals dt and integer
orbit numbers k, produced ONCE by float64 CPU preprocessing
(epoch_center_times); the graph only ever combines O(1) quantities:
tau = dt - (t0_off + k p_off).
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from .ellip import dtype_eps
from .trig import sincos

__all__ = ["epoch_center_times", "separation_circular", "tau_from_epochs"]

_TWO_PI = 2.0 * math.pi


def epoch_center_times(
    t_model: np.ndarray, t0_ref: float, period_ref: float
) -> np.ndarray:
    """Float64 CPU preprocessing: times -> (2, m) [dt, orbit number].

    dt = t - t0_ref - k * period_ref with k = round((t - t0_ref)/period_ref):
    each datum's time relative to its own transit, plus which orbit it
    belongs to (exact small integers). Same contract as the applemcmc
    trapezoid example.
    """
    t_model = np.asarray(t_model, dtype=np.float64)
    k = np.round((t_model - t0_ref) / period_ref)
    dt = t_model - t0_ref - k * period_ref
    return np.stack([dt, k])


def tau_from_epochs(dt: mx.array, k: mx.array, t0_off, p_off, period):
    """Time from nearest mid-transit, re-wrapped in case a large offset
    pushed a datum into the neighboring orbit. All O(1) quantities."""
    tau = dt - (t0_off + k * p_off)
    return tau - period * mx.round(tau / period)


def separation_circular(tau: mx.array, period, b, a) -> mx.array:
    """z(tau) for a circular orbit: z^2 = a^2 sin^2 phi + b^2 cos^2 phi,
    phi = 2 pi tau / P (phi = 0 at mid-transit; cos i = b/a).

    Far-side masking: the raw formula is symmetric under phi -> phi + pi,
    so without a mask it would fabricate a mirror transit at the *far*
    conjunction (z = b at phase 0.5). Points with cos phi <= 0 (planet
    behind the star) are pushed to z > 1 + r for any r < 1, where the
    flux and its gradients are identically zero/flat — so the where is
    smooth in effect (both branches give flux 1 near |phi| = pi/2 for
    any sane a > 1 + r).

    The sqrt argument is floored so the backward pass stays finite at the
    exact point z = 0 (b = 0 at mid-transit); the floor changes z by
    ~10 eps where the flux is quadratically flat in z, i.e. invisibly.
    """
    eps = dtype_eps(tau.dtype)
    phi = (_TWO_PI / period) * tau
    sphi, cphi = sincos(phi)
    z2 = (a * sphi) ** 2 + (b * cphi) ** 2
    z = mx.sqrt(mx.maximum(z2, (10.0 * eps) ** 2))
    return mx.where(cphi > 0.0, z, 2.0 + z)
