"""Quadratic limb-darkened transit flux from the ALFM19 solution vector.

The photometric core: (z, r, u1, u2) -> flux. Batched, dtype-polymorphic,
compile-safe, autodiff-clean. The primary output is the flux *deviation*
F - 1 (exactly 0 out of transit), matching baseline-subtracted data and
the engine's float32 conditioning discipline; ``light_curve`` adds the
baseline back for standalone use.
"""

from __future__ import annotations

import math

import mlx.core as mx

from .greens import quad_g_coeffs
from .solution import sn_dev

__all__ = ["flux_dev", "light_curve"]

_PI = math.pi
_TWO_PI_3 = 2.0 * math.pi / 3.0


def flux_dev(z: mx.array, r, u1, u2) -> mx.array:
    """F - 1 for a quadratic limb-darkened transit at separation(s) z.

    F = sum_n g_n s_n / (pi (g0 + 2 g1/3)); writing s_n as deviations from
    their unocculted values makes the numerator O(depth):

        F - 1 = (g0 s0d + g1 s1d + g2 s2d) / (pi (1 - u1/3 - u2/6))

    All inputs broadcast; z drives the dtype.
    """
    g0, g1, g2 = quad_g_coeffs(u1, u2)
    s0d, s1d, s2d = sn_dev(z, r)
    norm = _PI * (1.0 - u1 / 3.0 - u2 / 6.0)
    return (g0 * s0d + g1 * s1d + g2 * s2d) / norm


def light_curve(z: mx.array, r, u1, u2) -> mx.array:
    """Absolute flux (1 out of transit) — standalone convenience."""
    return 1.0 + flux_dev(z, r, u1, u2)
