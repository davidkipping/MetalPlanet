"""Fixed-iteration Kepler solver: residuals at machine precision,
gradients finite, geometry against a converged scalar reference."""

import math

import mlx.core as mx
import numpy as np

from metalplanet.kepler import (
    kepler_E,
    mean_anomaly_offset_at_transit,
    separation_keplerian,
)

RNG = np.random.default_rng(53)


def _kepler_ref(M, e, tol=1e-14):
    """Scalar Newton to convergence (float64 oracle)."""
    M = np.mod(M + np.pi, 2 * np.pi) - np.pi
    E = M + 0.85 * e * np.sign(np.sin(M))
    for _ in range(60):
        d = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= d
        if abs(d) < tol:
            break
    return E


class TestKeplerE:
    def test_residual_fp64(self):
        e = np.concatenate([RNG.uniform(0, 0.95, 3000), [0.0, 0.9, 0.95]])
        M = RNG.uniform(-4 * np.pi, 4 * np.pi, e.size)
        with mx.stream(mx.cpu):
            E = np.array(kepler_E(mx.array(M, dtype=mx.float64),
                                  mx.array(e, dtype=mx.float64)),
                         dtype=np.float64)
        resid = E - e * np.sin(E) - np.mod(M + np.pi, 2 * np.pi) + np.pi
        # E solves for the wrapped M
        Mw = np.mod(M + np.pi, 2 * np.pi) - np.pi
        resid = E - e * np.sin(E) - Mw
        assert np.max(np.abs(resid)) < 5e-15

    def test_residual_fp32(self):
        e = RNG.uniform(0, 0.9, 2000).astype(np.float32)
        M = RNG.uniform(-np.pi, np.pi, 2000).astype(np.float32)
        E = np.array(kepler_E(mx.array(M), mx.array(e)), dtype=np.float64)
        e64 = e.astype(np.float64)
        M64 = M.astype(np.float64)
        resid = E - e64 * np.sin(E) - M64
        assert np.max(np.abs(resid)) < 5e-6

    def test_matches_scalar_reference(self):
        e = RNG.uniform(0, 0.95, 500)
        M = RNG.uniform(-np.pi, np.pi, 500)
        with mx.stream(mx.cpu):
            E = np.array(kepler_E(mx.array(M, dtype=mx.float64),
                                  mx.array(e, dtype=mx.float64)),
                         dtype=np.float64)
        ref = np.array([_kepler_ref(m, ee) for m, ee in zip(M, e)])
        np.testing.assert_allclose(E, ref, rtol=0, atol=5e-13)

    def test_gradients_finite(self):
        M = mx.array(RNG.uniform(-np.pi, np.pi, 500).astype(np.float32))
        e = mx.array(RNG.uniform(0, 0.9, 500).astype(np.float32))

        def f(M_, e_):
            return mx.sum(kepler_E(M_, e_))

        g = mx.grad(f, argnums=(0, 1))(M, e)
        assert all(bool(mx.all(mx.isfinite(gi))) for gi in g)

    def test_implicit_gradient_correct(self):
        """dE/dM = 1/(1 - e cos E) — the fixed-iteration solve should
        reproduce the implicit-function derivative to high accuracy."""
        M = mx.array(RNG.uniform(-3, 3, 200), dtype=mx.float64)
        e = 0.4
        with mx.stream(mx.cpu):
            g = mx.grad(lambda m: mx.sum(kepler_E(m, e)))(M)
            E = np.array(kepler_E(M, e), dtype=np.float64)
        want = 1.0 / (1.0 - e * np.cos(E))
        np.testing.assert_allclose(np.array(g), want, rtol=1e-10, atol=0)


class TestGeometry:
    def test_circular_limit_matches_circular_orbit(self):
        """e -> 0: z from separation_keplerian equals the circular form."""
        per, a, inc = 3.0, 12.0, math.radians(88.0)
        t = RNG.uniform(-1.5, 1.5, 2000)
        M = 2 * np.pi * t / per + mean_anomaly_offset_at_transit(
            0.0, math.radians(90.0))
        with mx.stream(mx.cpu):
            z, front = separation_keplerian(
                mx.array(M, dtype=mx.float64), 0.0, a, inc,
                math.radians(90.0))
            z = np.array(z, dtype=np.float64)
            front = np.array(front)
        phi = 2 * np.pi * t / per
        b = a * math.cos(inc)
        z_circ = np.sqrt((a * np.sin(phi)) ** 2 + (b * np.cos(phi)) ** 2)
        np.testing.assert_allclose(z, z_circ, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(front, np.cos(phi) > 0)

    def test_transit_at_t0(self):
        """At M = M_transit the separation equals the eccentric-orbit
        impact parameter b_tra = a cos i (1-e^2)/(1+e sin w)."""
        for e, w_deg in [(0.0, 90.0), (0.3, 45.0), (0.6, 130.0), (0.2, 271.0)]:
            w = math.radians(w_deg)
            a, inc = 15.0, math.radians(87.5)
            m0 = mean_anomaly_offset_at_transit(e, w)
            with mx.stream(mx.cpu):
                z, front = separation_keplerian(
                    mx.array([m0], dtype=mx.float64), e, a, inc, w)
            b_tra = a * math.cos(inc) * (1 - e * e) / (1 + e * math.sin(w))
            assert abs(float(np.array(z)[0]) - abs(b_tra)) < 1e-10
            assert bool(np.array(front)[0])


class TestMarkleyKepler:
    """Production solver: kepler(M, e) -> (sinf, cosf), Markley starter +
    one 5th-order refinement, implicit-function-theorem VJP."""

    @staticmethod
    def _ref_sincos_f(M, e):
        E = _kepler_ref(M, e)
        fac = np.sqrt((1 + e) / (1 - e))
        A = fac * fac * (1 - np.cos(E)) / 2
        B = (1 + np.cos(E)) / 2
        return fac * np.sin(E) / (A + B), (B - A) / (A + B)

    def test_fp64_accuracy(self):
        from metalplanet.kepler import kepler
        e = np.concatenate([RNG.uniform(0, 0.95, 3000),
                            [0.0, 0.5, 0.9, 0.95]])
        M = np.concatenate([RNG.uniform(-4 * np.pi, 4 * np.pi, 3000),
                            [0.0, np.pi, -np.pi, 1e-8]])
        with mx.stream(mx.cpu):
            sf, cf = kepler(mx.array(M, dtype=mx.float64),
                            mx.array(e, dtype=mx.float64))
            sf, cf = np.array(sf, dtype=np.float64), np.array(cf, dtype=np.float64)
        Mw = np.mod(M + np.pi, 2 * np.pi) - np.pi
        ref = np.array([self._ref_sincos_f(m, ee) for m, ee in zip(Mw, e)])
        np.testing.assert_allclose(sf, ref[:, 0], rtol=0, atol=3e-14)
        np.testing.assert_allclose(cf, ref[:, 1], rtol=0, atol=3e-14)
        # exactly on the unit circle
        np.testing.assert_allclose(sf * sf + cf * cf, 1.0, atol=5e-14)

    def test_fp32_accuracy(self):
        from metalplanet.kepler import kepler
        e = RNG.uniform(0, 0.9, 2000).astype(np.float32)
        M = RNG.uniform(-np.pi, np.pi, 2000).astype(np.float32)
        sf, cf = kepler(mx.array(M), mx.array(e))
        e64 = e.astype(np.float64)
        M64 = M.astype(np.float64)
        ref = np.array([self._ref_sincos_f(m, ee) for m, ee in zip(M64, e64)])
        np.testing.assert_allclose(np.array(sf, dtype=np.float64), ref[:, 0],
                                   rtol=0, atol=5e-6)
        np.testing.assert_allclose(np.array(cf, dtype=np.float64), ref[:, 1],
                                   rtol=0, atol=5e-6)

    def test_implicit_vjp_vs_finite_differences(self):
        from metalplanet.kepler import kepler
        M0 = RNG.uniform(-3, 3, 200)
        e0 = RNG.uniform(0.01, 0.9, 200)

        def loss(M_, e_):
            sf, cf = kepler(M_, e_)
            return mx.sum(2.0 * sf + 3.0 * cf)

        with mx.stream(mx.cpu):
            g = mx.grad(loss, argnums=(0, 1))(
                mx.array(M0, dtype=mx.float64), mx.array(e0, dtype=mx.float64))
        h = 1e-7
        for idx, x0, other in ((0, M0, e0), (1, e0, M0)):
            def ev(x):
                args = [None, None]
                args[idx] = mx.array(x, dtype=mx.float64)
                args[1 - idx] = mx.array(other, dtype=mx.float64)
                with mx.stream(mx.cpu):
                    sf, cf = kepler(args[0], args[1])
                    return np.array(2.0 * sf + 3.0 * cf, dtype=np.float64)
            fd = (ev(x0 + h) - ev(x0 - h)) / (2 * h)
            np.testing.assert_allclose(np.array(g[idx]), fd, rtol=5e-6,
                                       atol=1e-7)

    def test_odd_in_M_and_circular_limit(self):
        from metalplanet.kepler import kepler
        M = RNG.uniform(0.01, np.pi - 0.01, 500)
        with mx.stream(mx.cpu):
            sp, cp = kepler(mx.array(M, dtype=mx.float64), 0.3)
            sn, cn = kepler(mx.array(-M, dtype=mx.float64), 0.3)
            np.testing.assert_allclose(np.array(sp), -np.array(sn), atol=1e-15)
            np.testing.assert_allclose(np.array(cp), np.array(cn), atol=1e-15)
            # e = 0: f = M exactly
            s0, c0 = kepler(mx.array(M, dtype=mx.float64), 0.0)
            np.testing.assert_allclose(np.array(s0), np.sin(M), atol=5e-15)
            np.testing.assert_allclose(np.array(c0), np.cos(M), atol=5e-15)


class TestCartesianTail:
    """Cartesian-from-E separation (graph-path swap): equivalence to the
    conic tail where the conic is well-conditioned, superiority where it
    is not, and the E-level implicit VJP."""

    def test_matches_fp64_cartesian_reference(self):
        import math
        rng = np.random.default_rng(71)
        n = 20000
        e = rng.uniform(0, 0.95, n)
        M = rng.uniform(-4 * np.pi, 4 * np.pi, n)
        a = rng.uniform(2, 100, n)
        inc = rng.uniform(0.05, np.pi / 2, n)
        w = rng.uniform(0, 2 * np.pi, n)
        with mx.stream(mx.cpu):
            z, front = separation_keplerian(
                mx.array(M, dtype=mx.float64), mx.array(e, dtype=mx.float64),
                mx.array(a, dtype=mx.float64),
                mx.array(inc, dtype=mx.float64),
                mx.array(w, dtype=mx.float64))
            z = np.array(z, dtype=np.float64)
            front = np.array(front)
        E = np.array([_kepler_ref(m, ee) for m, ee in zip(M, e)])
        X = a * (np.cos(E) - e)
        Y = a * np.sqrt(1 - e * e) * np.sin(E)
        u = X * np.cos(w) - Y * np.sin(w)
        v = X * np.sin(w) + Y * np.cos(w)
        z_ref = np.sqrt(u * u + (v * np.cos(inc)) ** 2)
        np.testing.assert_allclose(z / a, z_ref / a, rtol=0, atol=5e-13)
        np.testing.assert_array_equal(front, v > 0)

    def test_apastron_transit_fp32_accuracy(self):
        """The regime the old conic tail failed (6e-3 flux-level error at
        e~0.99, a=300): the Cartesian tail must stay ~1e-4 R* accurate."""
        import math
        e, a, w = 0.99, 300.0, math.radians(271.0)
        inc = math.radians(89.9)
        m0 = mean_anomaly_offset_at_transit(e, w)
        M = m0 + np.linspace(-2e-4, 2e-4, 4001)
        # plain python floats only: numpy scalars silently poison mx graphs
        z32 = np.array(separation_keplerian(
            mx.array(M.astype(np.float32)), float(e), float(a),
            float(inc), float(w))[0], dtype=np.float64)
        with mx.stream(mx.cpu):
            z64 = np.array(separation_keplerian(
                mx.array(M, dtype=mx.float64), e, a, inc, w)[0],
                dtype=np.float64)
        near = z64 < 2.0
        assert near.sum() > 10
        assert np.abs(z32 - z64)[near].max() < 2e-4

    def test_E_level_implicit_vjp(self):
        from metalplanet.kepler import kepler_E_sincos
        rng = np.random.default_rng(73)
        M0 = rng.uniform(-3, 3, 200)
        e0 = rng.uniform(0.01, 0.9, 200)

        def loss(M_, e_):
            sE, cE = kepler_E_sincos(M_, e_)
            return mx.sum(1.7 * sE - 0.6 * cE)

        with mx.stream(mx.cpu):
            g = mx.grad(loss, argnums=(0, 1))(
                mx.array(M0, dtype=mx.float64), mx.array(e0, dtype=mx.float64))
        h = 1e-7
        for idx, x0, other in ((0, M0, e0), (1, e0, M0)):
            def ev(x):
                args = [None, None]
                args[idx] = mx.array(x, dtype=mx.float64)
                args[1 - idx] = mx.array(other, dtype=mx.float64)
                with mx.stream(mx.cpu):
                    sE, cE = kepler_E_sincos(args[0], args[1])
                    return np.array(1.7 * sE - 0.6 * cE, dtype=np.float64)
            fd = (ev(x0 + h) - ev(x0 - h)) / (2 * h)
            np.testing.assert_allclose(np.array(g[idx]), fd, rtol=5e-6,
                                       atol=1e-7)

    def test_separation_gradients_finite_and_match_fd(self):
        rng = np.random.default_rng(79)
        M0 = rng.uniform(-3, 3, 100)

        def zs(M_, e_, a_, i_, w_):
            z, _ = separation_keplerian(M_, e_, a_, i_, w_)
            return mx.sum(z)

        args = [mx.array(M0, dtype=mx.float64),
                mx.array(0.4, dtype=mx.float64),
                mx.array(15.0, dtype=mx.float64),
                mx.array(1.53, dtype=mx.float64),
                mx.array(0.8, dtype=mx.float64)]
        with mx.stream(mx.cpu):
            g = mx.grad(zs, argnums=(0, 1, 2, 3, 4))(*args)
            for k in range(5):
                h = 1e-7
                ap = [x for x in args]
                am = [x for x in args]
                ap[k] = args[k] + h
                am[k] = args[k] - h
                fd = (float(np.array(zs(*ap))) - float(np.array(zs(*am)))) / (2 * h)
                gk = float(np.array(mx.sum(g[k])))
                assert abs(gk - fd) < 5e-5 * max(abs(fd), 1.0), (k, gk, fd)
