"""M1: cel unit tests vs scipy and the adaptive scalar reference."""

import mlx.core as mx
import numpy as np
import pytest
from scipy import special

from metalplanet.ellip import cel, cel3
from reference_limbdark import cel as cel_ref

RNG = np.random.default_rng(7)


def _mx64(x):
    return mx.array(np.asarray(x, dtype=np.float64), dtype=mx.float64)


def _mx32(x):
    return mx.array(np.asarray(x, dtype=np.float32))


class TestAgainstScipy:
    def test_E_and_K_fp64(self):
        m = np.concatenate([
            RNG.uniform(0, 1, 3000),
            1.0 - 10.0 ** RNG.uniform(-15, -1, 3000),
            [0.0, 1e-18, 1e-12],
        ])
        kc = np.sqrt(1.0 - m)
        with mx.stream(mx.cpu):
            E = np.array(cel(_mx64(kc), 1.0, 1.0, _mx64(kc * kc)))
            K = np.array(cel(_mx64(kc), 1.0, 1.0, 1.0))
        np.testing.assert_allclose(E, special.ellipe(m), rtol=5e-15, atol=5e-15)
        np.testing.assert_allclose(K, special.ellipk(m), rtol=5e-15, atol=0)

    def test_E_and_K_fp32(self):
        m = np.concatenate([
            RNG.uniform(0, 1, 2000).astype(np.float32),
            (1.0 - 10.0 ** RNG.uniform(-6, -1, 2000)).astype(np.float32),
        ]).astype(np.float64)
        kc = np.sqrt(1.0 - m)
        E = np.array(cel(_mx32(kc), 1.0, 1.0, _mx32(kc * kc)), dtype=np.float64)
        K = np.array(cel(_mx32(kc), 1.0, 1.0, 1.0), dtype=np.float64)
        # fp32: a few ULP of the fp32 result
        np.testing.assert_allclose(E, special.ellipe(m), rtol=3e-6, atol=3e-6)
        np.testing.assert_allclose(K, special.ellipk(m), rtol=3e-6, atol=0)

    def test_Em1mKdm_identity_fp64(self):
        """cel(kc,1,1,0) = (E(m) - (1-m)K(m))/m."""
        m = np.concatenate([RNG.uniform(1e-3, 1 - 1e-3, 2000), [0.5]])
        kc = np.sqrt(1.0 - m)
        with mx.stream(mx.cpu):
            v = np.array(cel(_mx64(kc), 1.0, 1.0, 0.0))
        ref = (special.ellipe(m) - (1.0 - m) * special.ellipk(m)) / m
        np.testing.assert_allclose(v, ref, rtol=2e-13, atol=1e-14)


class TestAgainstAdaptiveReference:
    def _random_args(self, n):
        kc = 10.0 ** RNG.uniform(-12, 0, n)
        p = 10.0 ** RNG.uniform(-15, 3, n)
        a = RNG.uniform(-2, 2, n)
        b = RNG.uniform(-2, 2, n)
        return kc, p, a, b

    def test_general_args_fp64(self):
        kc, p, a, b = self._random_args(500)
        with mx.stream(mx.cpu):
            v = np.array(cel(_mx64(kc), _mx64(p), _mx64(a), _mx64(b)))
        ref = np.array([cel_ref(*args) for args in zip(kc, p, a, b)])
        np.testing.assert_allclose(v, ref, rtol=1e-12, atol=1e-12)

    def test_general_args_fp32(self):
        kc, p, a, b = self._random_args(500)
        # restrict to fp32-representable regime
        kc = np.maximum(kc, 2e-7)
        p = np.maximum(p, 1e-10)
        v = np.array(cel(_mx32(kc), _mx32(p), _mx32(a), _mx32(b)),
                     dtype=np.float64)
        ref = np.array([cel_ref(*args) for args in
                        zip(kc.astype(np.float32).astype(np.float64),
                            p.astype(np.float32).astype(np.float64),
                            a.astype(np.float32).astype(np.float64),
                            b.astype(np.float32).astype(np.float64))])
        scale = np.maximum(np.abs(ref), 1.0)
        assert np.max(np.abs(v - ref) / scale) < 5e-5

    def test_cel3_matches_three_cels_fp64(self):
        kc, p, _, b1 = self._random_args(300)
        a1 = RNG.uniform(-2, 2, 300)
        b2 = RNG.uniform(-2, 2, 300)
        b3 = RNG.uniform(-2, 2, 300)
        with mx.stream(mx.cpu):
            o1, o2, o3 = cel3(_mx64(kc), _mx64(p), _mx64(a1), _mx64(b1),
                              _mx64(b2), _mx64(b3))
            r1 = cel(_mx64(kc), _mx64(p), _mx64(a1), _mx64(b1))
            r2 = cel(_mx64(kc), 1.0, 1.0, _mx64(b2))
            r3 = cel(_mx64(kc), 1.0, 1.0, _mx64(b3))
            for o, r in [(o1, r1), (o2, r2), (o3, r3)]:
                np.testing.assert_allclose(np.array(o), np.array(r),
                                           rtol=1e-13, atol=1e-13)


class TestRobustness:
    def test_degenerate_inputs_finite(self):
        """kc = 0 exactly, p = 0 exactly: clamped, finite, no NaN."""
        kc = _mx32(np.array([0.0, 1e-30, 1.0]))
        v = cel(kc, _mx32([0.0, 1e-25, 1.0]), 1.0, 1.0)
        assert bool(mx.all(mx.isfinite(v)))

    def test_gradient_finite(self):
        """d cel/d kc finite across the range incl. clamped kc."""
        def f(kc):
            return mx.sum(cel(kc, 1.0, 1.0, kc * kc))

        kc = _mx32(np.concatenate([[0.0, 1e-8], RNG.uniform(1e-6, 1, 100)]))
        g = mx.grad(f)(kc)
        assert bool(mx.all(mx.isfinite(g)))

    def test_broadcast_and_scalars(self):
        kc = _mx32(RNG.uniform(0.1, 1, (4, 5)))
        v = cel(kc, 1.0, 1.0, 0.5)
        assert v.shape == (4, 5)
