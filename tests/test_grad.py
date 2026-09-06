"""M3: autodiff cleanliness — no NaN gradients anywhere, and agreement
with float64 central finite differences across all regimes including
points at (and straddling) the regime boundaries.
"""

import math

import mlx.core as mx
import numpy as np

from metalplanet import flux_dev, q_to_u

RNG = np.random.default_rng(31)


def _boundary_heavy_samples(n, dtype):
    """(z, r, u1, u2) with a large fraction sitting on/near the degenerate
    lines: z = 1-r, z = 1+r, z = r, z = 0, plus the ulp-neighborhoods."""
    r = 10.0 ** RNG.uniform(-3, math.log10(0.9), n)
    kind = RNG.integers(0, 8, n)
    off = np.where(RNG.random(n) < 0.5, 0.0,
                   10.0 ** RNG.uniform(-9, -1, n) * RNG.choice([-1, 1], n))
    z = np.select(
        [kind == 0, kind == 1, kind == 2, kind == 3],
        [1 - r + off, 1 + r + off, r + off, np.zeros(n)],
        default=RNG.uniform(0, 1.5, n) * (1 + r),
    )
    z = np.abs(z)
    q1 = RNG.uniform(0.01, 0.99, n)
    q2 = RNG.uniform(0.01, 0.99, n)
    u1 = 2 * np.sqrt(q1) * q2
    u2 = np.sqrt(q1) * (1 - 2 * q2)
    return (z.astype(dtype), r.astype(dtype),
            u1.astype(dtype), u2.astype(dtype))


def _grad_all(z, r, u1, u2):
    """Gradients of sum(flux_dev) w.r.t. all four inputs."""

    def f(z_, r_, u1_, u2_):
        return mx.sum(flux_dev(z_, r_, u1_, u2_))

    return mx.grad(f, argnums=(0, 1, 2, 3))(z, r, u1, u2)


class TestNoNaNs:
    def test_fp32_100k_boundary_heavy(self):
        z, r, u1, u2 = _boundary_heavy_samples(100_000, np.float32)
        grads = _grad_all(mx.array(z), mx.array(r), mx.array(u1),
                          mx.array(u2))
        for g, tag in zip(grads, ["z", "r", "u1", "u2"]):
            arr = np.array(g, dtype=np.float64)
            assert np.all(np.isfinite(arr)), f"non-finite d/d{tag}"

    def test_fp64_100k_boundary_heavy(self):
        z, r, u1, u2 = _boundary_heavy_samples(100_000, np.float64)
        with mx.stream(mx.cpu):
            grads = _grad_all(
                mx.array(z, dtype=mx.float64), mx.array(r, dtype=mx.float64),
                mx.array(u1, dtype=mx.float64), mx.array(u2, dtype=mx.float64))
            for g, tag in zip(grads, ["z", "r", "u1", "u2"]):
                arr = np.array(g, dtype=np.float64)
                assert np.all(np.isfinite(arr)), f"non-finite d/d{tag}"

    def test_grad_through_q_parameterization(self):
        z = mx.array(RNG.uniform(0, 1.3, 1000).astype(np.float32))

        def f(q1, q2, r):
            u1, u2 = q_to_u(q1, q2)
            return mx.sum(flux_dev(z, r, u1, u2))

        for q1v, q2v in [(0.25, 0.5), (1e-9, 0.5), (0.999, 0.001)]:
            g = mx.grad(f, argnums=(0, 1, 2))(
                mx.array(q1v), mx.array(q2v), mx.array(0.1))
            assert all(bool(mx.isfinite(gi)) for gi in g)


class TestFiniteDifferences:
    def _fd_check(self, z, r, u1, u2, h, rtol, atol):
        """Central FD in fp64 on the CPU stream vs autodiff, elementwise."""
        n = z.size

        def fd(idx, args):
            plus = [a.copy() for a in args]
            minus = [a.copy() for a in args]
            plus[idx] += h
            minus[idx] -= h

            def ev(a):
                with mx.stream(mx.cpu):
                    return np.array(flux_dev(
                        mx.array(a[0], dtype=mx.float64),
                        mx.array(a[1], dtype=mx.float64),
                        mx.array(a[2], dtype=mx.float64),
                        mx.array(a[3], dtype=mx.float64)), dtype=np.float64)

            return (ev(plus) - ev(minus)) / (2 * h)

        with mx.stream(mx.cpu):
            grads = _grad_all(
                mx.array(z, dtype=mx.float64), mx.array(r, dtype=mx.float64),
                mx.array(u1, dtype=mx.float64), mx.array(u2, dtype=mx.float64))
        args = [z, r, u1, u2]
        for idx, tag in enumerate(["z", "r", "u1", "u2"]):
            ad = np.array(grads[idx], dtype=np.float64)
            fdg = fd(idx, args)
            np.testing.assert_allclose(
                ad, fdg, rtol=rtol, atol=atol,
                err_msg=f"autodiff vs FD mismatch in d/d{tag}")

    def test_interior_1k(self):
        """Random points kept > 1e-3 away from every degenerate line."""
        n, rows = 0, []
        while n < 1000:
            r = 10.0 ** RNG.uniform(-2, math.log10(0.6))
            z = RNG.uniform(0, 1.3) * (1 + r)
            dists = [abs(z - (1 - r)), abs(z - (1 + r)), abs(z - r), z]
            if min(dists) > 1e-3:
                rows.append((z, r))
                n += 1
        z = np.array([p[0] for p in rows])
        r = np.array([p[1] for p in rows])
        u1 = RNG.uniform(0.0, 1.0, 1000)
        u2 = RNG.uniform(-0.3, 0.5, 1000)
        self._fd_check(z, r, u1, u2, h=1e-7, rtol=2e-6, atol=1e-8)

    def test_near_boundaries(self):
        """1e-4 .. 1e-2 away from each boundary line; second derivatives
        blow up like dist^{-1/2} there, so tolerances are looser."""
        rows = []
        for r in (0.01, 0.1, 0.3, 0.5, 0.7):
            for base in (1 - r, 1 + r, r):
                for d in 10.0 ** np.arange(-4, -1.5, 0.25):
                    for s in (-1, 1):
                        zz = base + s * d
                        if zz > 0:
                            rows.append((zz, r))
        z = np.array([p[0] for p in rows])
        r = np.array([p[1] for p in rows])
        u1 = np.full_like(z, 0.4)
        u2 = np.full_like(z, 0.2)
        self._fd_check(z, r, u1, u2, h=1e-9, rtol=1e-3, atol=1e-6)

    def test_symmetry_grad_z_zero(self):
        """dF/dz = 0 at z = 0 by symmetry (abs kink has zero subgradient)."""
        with mx.stream(mx.cpu):
            g = _grad_all(
                mx.array(np.zeros(8), dtype=mx.float64),
                mx.array(np.full(8, 0.1), dtype=mx.float64),
                mx.array(np.full(8, 0.4), dtype=mx.float64),
                mx.array(np.full(8, 0.2), dtype=mx.float64))
            assert np.allclose(np.array(g[0]), 0.0, atol=1e-12)
