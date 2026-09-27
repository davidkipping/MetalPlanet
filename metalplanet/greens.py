"""Limb-darkening polynomial -> Green's basis coefficients (ALFM19).

I(mu)/I0 = 1 - sum_{n>=1} u_n (1-mu)^n  is re-expressed so the occulted
flux becomes F = sum_n g_n s_n / (pi (g_0 + 2 g_1 / 3)), with the s_n
solution vector doing all the geometric work. The u -> g map is *affine
with constant coefficients*, so it is precomputed on the host in float64;
the in-graph version is a tiny matmul-free closed form for the quadratic
case (all constants plain Python floats — never numpy scalars).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "greens_affine",
    "greens_transform_np",
    "quad_g_coeffs",
    "quad_norm",
]


def greens_transform_np(u: np.ndarray) -> np.ndarray:
    """u_1..u_N (any N >= 0) -> g_0..g_N, float64 host-side.

    Port of the ALFM19 transform (as in jaxoplanet): prepend -1, binomial
    change of basis (1-mu)^n -> mu^n with alternating signs, then the
    downward g-recursion.
    """
    u = np.asarray(u, dtype=np.float64)
    ut = np.concatenate([[-1.0], u])
    n = ut.size
    j = np.arange(n)
    # comb[a, b] = C(b, a);  p_a = (-1)^(a+1) * sum_b C(b, a) ut_b
    comb = np.array([[math.comb(int(bb), int(aa)) for bb in j] for aa in j],
                    dtype=np.float64)
    p = ((-1.0) ** (j + 1)) * (comb @ ut)
    g = np.zeros(n + 2)
    for k in range(n - 1, 1, -1):
        g[k] = p[k] / (k + 2) + g[k + 2]
    g[1] = p[1] + 3 * g[3]
    g[0] = p[0] + 2 * g[2]
    return g[:n]


def quad_g_coeffs(u1, u2):
    """Quadratic-case g coefficients as MLX-safe expressions.

    g0 = 1 - u1 - 1.5 u2,  g1 = u1 + 2 u2,  g2 = -0.25 u2
    (verified against greens_transform_np). Works on mx.arrays or floats.
    """
    g0 = 1.0 - u1 - 1.5 * u2
    g1 = u1 + 2.0 * u2
    g2 = -0.25 * u2
    return g0, g1, g2


def quad_norm(u1, u2):
    """pi (g0 + 2 g1 / 3) = pi (1 - u1/3 - u2/6): the unocculted flux."""
    return math.pi * (1.0 - u1 / 3.0 - u2 / 6.0)


def greens_affine(n: int):
    """(A, c) with g = A @ u + c — the u -> g map written as the affine
    map it is, so the g_n can be built from *traced* coefficients.

    ``greens_transform_np`` bakes u in as float64 host constants, which
    is right for a fixed law but breaks the batman workflow of varying
    u between calls on one compiled model. The transform is linear in u
    (the -1 prepended to the coefficient vector supplies the constant
    part), so the columns are recovered exactly by evaluating it on the
    unit vectors. Returns float64 arrays of shape (n+1, n) and (n+1,).
    """
    n = int(n)
    zero_u = np.zeros(n, dtype=np.float64)
    c = greens_transform_np(zero_u)
    A = np.empty((c.size, n), dtype=np.float64)
    for j in range(n):
        e = np.zeros(n, dtype=np.float64)
        e[j] = 1.0
        A[:, j] = greens_transform_np(e) - c
    return A, c
