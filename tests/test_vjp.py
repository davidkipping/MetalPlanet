"""M5: analytic VJP — forward identical to Stage 1, gradients match the
autodiff oracle, NaN-free at boundaries."""

import math

import mlx.core as mx
import numpy as np

from metalplanet import flux_dev
from metalplanet.vjp import flux_dev_analytic

RNG = np.random.default_rng(41)


def _samples(n, dtype, boundary_frac=0.5):
    r = 10.0 ** RNG.uniform(-2, math.log10(0.5), n)
    kind = RNG.integers(0, 6, n)
    off = np.where(RNG.random(n) < 0.5, 0.0,
                   10.0 ** RNG.uniform(-9, -1, n) * RNG.choice([-1, 1], n))
    z = np.select(
        [kind == 0, kind == 1, kind == 2, kind == 3],
        [1 - r + off, 1 + r + off, r + off, np.zeros(n)],
        default=RNG.uniform(0, 1.4, n) * (1 + r),
    )
    z = np.abs(z)
    u1 = RNG.uniform(0.0, 1.0, n)
    u2 = RNG.uniform(-0.3, 0.5, n)
    return (z.astype(dtype), r.astype(dtype),
            u1.astype(dtype), u2.astype(dtype))


class TestForwardIdentical:
    def test_matches_stage1_fp32(self):
        z, r, u1, u2 = _samples(20_000, np.float32)
        a = np.array(flux_dev_analytic(mx.array(z), mx.array(r),
                                       mx.array(u1), mx.array(u2)))
        b = np.array(flux_dev(mx.array(z), mx.array(r),
                              mx.array(u1), mx.array(u2)))
        assert np.array_equal(a, b)


class TestGradientsMatchAutodiff:
    def _compare(self, z, r, u1, u2, rtol, atol, interior_only):
        if interior_only:
            keep = (np.abs(z - r) > 1e-4) & (np.abs(z + r - 1.0) > 1e-4) \
                & (np.abs(z - 1.0 - r) > 1e-4) & (z > 1e-4)
            z, r, u1, u2 = z[keep], r[keep], u1[keep], u2[keep]
        args64 = [mx.array(v, dtype=mx.float64) for v in (z, r, u1, u2)]

        def loss_ad(*a):
            return mx.sum(flux_dev(*a))

        def loss_an(*a):
            return mx.sum(flux_dev_analytic(*a))

        with mx.stream(mx.cpu):
            g_ad = mx.grad(loss_ad, argnums=(0, 1, 2, 3))(*args64)
            g_an = mx.grad(loss_an, argnums=(0, 1, 2, 3))(*args64)
            for ga, gn, tag in zip(g_ad, g_an, ["z", "r", "u1", "u2"]):
                np.testing.assert_allclose(
                    np.array(gn), np.array(ga), rtol=rtol, atol=atol,
                    err_msg=f"analytic vs autodiff d/d{tag}")

    def test_interior_fp64(self):
        z, r, u1, u2 = _samples(5000, np.float64)
        self._compare(z, r, u1, u2, rtol=1e-9, atol=1e-11,
                      interior_only=True)

    def test_including_boundaries_fp64(self):
        """Everywhere except the z ~ r sliver: tight agreement. ON the
        sliver the autodiff of the Taylor switch redistributes gradient
        between dz and dr (see solution.py docstring) — there the
        *analytic* partials are the correct ones, and the invariant that
        must hold is the directional derivative along the z = r line."""
        z, r, u1, u2 = _samples(5000, np.float64)
        on_sliver = np.abs(z - r) < 25 * 2.22e-16
        # within ~1e-7 of the degenerate lines the *autodiff* path loses
        # precision (eps/|z-r| amplification through the Pi cancellation)
        # while the analytic partials stay conditioned — compare loosely
        # in absolute terms there, tightly everywhere else
        near_deg = ((np.abs(z - r) < 1e-6) | (np.abs(z + r - 1) < 1e-6)
                    | (np.abs(z - 1 - r) < 1e-6)) & ~on_sliver
        far = ~on_sliver & ~near_deg
        self._compare(z[far], r[far], u1[far], u2[far],
                      rtol=1e-6, atol=1e-9, interior_only=False)
        self._compare(z[near_deg], r[near_deg], u1[near_deg], u2[near_deg],
                      rtol=1e-2, atol=1e-5, interior_only=False)

        zs, rs = z[on_sliver], r[on_sliver]
        assert zs.size > 50  # the sampler must actually hit the sliver
        args64 = [mx.array(v, dtype=mx.float64)
                  for v in (zs, rs, u1[on_sliver], u2[on_sliver])]
        with mx.stream(mx.cpu):
            g_ad = mx.grad(lambda *a: mx.sum(flux_dev(*a)),
                           argnums=(0, 1))(*args64)
            g_an = mx.grad(lambda *a: mx.sum(flux_dev_analytic(*a)),
                           argnums=(0, 1))(*args64)
            np.testing.assert_allclose(
                np.array(g_ad[0]) + np.array(g_ad[1]),
                np.array(g_an[0]) + np.array(g_an[1]),
                rtol=1e-9, atol=1e-11,
                err_msg="directional derivative along z=r must agree")

    def test_batched_broadcast_shapes(self):
        """(n_chains, m) z against (n_chains, 1) params — the engine's
        actual layout — with per-chain reduction of param grads. z is
        kept off the z ~ r sliver (see test above)."""
        n, m = 8, 512
        z_np = RNG.uniform(0, 1.3, (n, m)).astype(np.float32)
        # keep clear of the degenerate lines (z=r, contact, outer contact)
        # where fp32 autodiff noise exceeds the comparison tolerance
        for line in (0.1, 0.9, 1.1):
            z_np[np.abs(z_np - line) < 0.01] += 0.02
        z = mx.array(z_np)
        r = mx.array(np.full((n, 1), 0.1, np.float32))
        u1 = mx.array(np.full((n, 1), 0.4, np.float32))
        u2 = mx.array(np.full((n, 1), 0.2, np.float32))

        def loss_ad(*a):
            return mx.sum(flux_dev(*a))

        def loss_an(*a):
            return mx.sum(flux_dev_analytic(*a))

        g_ad = mx.grad(loss_ad, argnums=(0, 1, 2, 3))(z, r, u1, u2)
        g_an = mx.grad(loss_an, argnums=(0, 1, 2, 3))(z, r, u1, u2)
        for ga, gn in zip(g_ad, g_an):
            assert ga.shape == gn.shape
            np.testing.assert_allclose(np.array(gn), np.array(ga),
                                       rtol=3e-4, atol=3e-5)


class TestNoNaNs:
    def test_fp32_100k(self):
        z, r, u1, u2 = _samples(100_000, np.float32)

        def loss(*a):
            return mx.sum(flux_dev_analytic(*a))

        g = mx.grad(loss, argnums=(0, 1, 2, 3))(
            mx.array(z), mx.array(r), mx.array(u1), mx.array(u2))
        for gi, tag in zip(g, ["z", "r", "u1", "u2"]):
            assert bool(mx.all(mx.isfinite(gi))), f"non-finite d/d{tag}"
