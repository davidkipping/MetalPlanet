"""Accurate sin/cos for the float64 path.

MLX 0.32's fp64 ``mx.sin``/``mx.cos`` are only float32-accurate
(~1.5e-7 max abs error, measured on the CPU stream — presumably a
single-precision SIMD path), while add/mul/div/sqrt and
arccos/arctan2 are true fp64. That poisons any fp64 verification path
involving orbital phases at the ~1e-7 level.

``sincos`` dispatches: native ops for float32 (native accuracy ~ eps32
is exactly right there), and a Cody–Waite argument reduction plus
Taylor evaluation in exact fp64 arithmetic for float64 — max abs error
~1e-16 for |x| up to ~1e6 (verified against numpy in tests).
"""

from __future__ import annotations

import mlx.core as mx

__all__ = ["sincos"]

# fdlibm-style three-part pi/2: HI has 33 trailing zero bits, so n * HI
# is exact for |n| < 2^33
_PIO2_HI = 1.57079632673412561417e+00
_PIO2_MID = 6.07710050650619224932e-11
_PIO2_LO = 2.02226624879595063154e-21
_TWO_OVER_PI = 0.6366197723675814

# Taylor coefficients: sin y = y * P(y^2), cos y = Q(y^2), |y| <= pi/4
_SIN_C = [
    1.0,
    -1.0 / 6.0,
    1.0 / 120.0,
    -1.0 / 5040.0,
    1.0 / 362880.0,
    -1.0 / 39916800.0,
    1.0 / 6227020800.0,
    -1.0 / 1307674368000.0,
    1.0 / 355687428096000.0,
]
_COS_C = [
    1.0,
    -0.5,
    1.0 / 24.0,
    -1.0 / 720.0,
    1.0 / 40320.0,
    -1.0 / 3628800.0,
    1.0 / 479001600.0,
    -1.0 / 87178291200.0,
    1.0 / 20922789888000.0,
]


def _horner(z, coeffs):
    acc = coeffs[-1] + 0.0 * z
    for c in reversed(coeffs[:-1]):
        acc = acc * z + c
    return acc


def _sincos64(x: mx.array):
    n = mx.round(x * _TWO_OVER_PI)
    y = ((x - n * _PIO2_HI) - n * _PIO2_MID) - n * _PIO2_LO
    z2 = y * y
    s = y * _horner(z2, _SIN_C)
    c = _horner(z2, _COS_C)
    q = n - 4.0 * mx.floor(n * 0.25)  # quadrant 0..3
    sin = mx.where(q == 0.0, s,
                   mx.where(q == 1.0, c,
                            mx.where(q == 2.0, -s, -c)))
    cos = mx.where(q == 0.0, c,
                   mx.where(q == 1.0, -s,
                            mx.where(q == 2.0, -c, s)))
    return sin, cos


def sincos(x: mx.array):
    """(sin x, cos x), full working-precision in both dtypes."""
    if x.dtype == mx.float64:
        return _sincos64(x)
    return mx.sin(x), mx.cos(x)
