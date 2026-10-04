"""Arbitrary-order polynomial limb darkening (ALFM19 M_n recursion).

Ground truth is a 40-digit mpmath direct integration of

    F(b) = 1 - [int I(x) alpha(x, b, r) x dx] / [2 pi int I(x) x dx]

which shares no code with the recursion under test.
"""

import math

import numpy as np
import mlx.core as mx
import pytest

import metalplanet
from metalplanet.greens import greens_affine, greens_transform_np
from metalplanet.poly import flux_dev_poly, sn_dev_poly

mp = pytest.importorskip("mpmath")
mp.mp.dps = 40

R = 0.1


def _oracle(b, r, u):
    b, r = mp.mpf(b), mp.mpf(r)

    def I(x):
        mu = mp.sqrt(1 - x * x)
        s = mp.mpf(1)
        for n, un in enumerate(u, start=1):
            s -= mp.mpf(un) * (1 - mu) ** n
        return s

    def alpha(x):
        if x + b <= r:
            return 2 * mp.pi
        if x >= b + r or x + r <= b:
            return mp.mpf(0)
        c = (x * x + b * b - r * r) / (2 * x * b)
        return 2 * mp.acos(mp.mpf(min(1, max(-1, c))))

    total = 2 * mp.pi * mp.quad(lambda x: I(x) * x, [0, 1])
    if b >= 1 + r:
        return 1.0
    hi = min(mp.mpf(1), b + r)
    pts = sorted({mp.mpf(0), min(abs(b - r), hi), hi})
    return float(1 - mp.quad(lambda x: I(x) * alpha(x) * x, pts) / total)


def _flux64(b, r, u):
    with mx.stream(mx.cpu):
        return 1.0 + np.array(
            flux_dev_poly(mx.array(np.asarray(b), dtype=mx.float64), r, u),
            dtype=np.float64)


class TestAgainstOracle:
    BS = np.array([0.0, 0.02, 0.3, 0.6, 0.9, 1.0, 1.05, 1.09])

    @pytest.mark.parametrize("u,tol", [
        ([0.4, 0.25], 1e-15),
        ([0.3, 0.2, 0.1], 1e-15),
        ([0.4, 0.25, 0.1, 0.05], 1e-15),
        ([0.05] * 8, 1e-14),
        ([0.05] * 12, 1e-13),
    ])
    def test_flux_matches_direct_integration(self, u, tol):
        got = _flux64(self.BS, R, u)
        ref = np.array([_oracle(b, R, u) for b in self.BS])
        assert np.abs(got - ref).max() < tol

    @pytest.mark.parametrize("r", [0.01, 0.3, 0.8])
    def test_radius_ratios(self, r):
        u = [0.2, 0.1, 0.05, 0.02]
        bs = np.array([0.0, 0.5 * r, 1.0 - r, 1.0, 1.0 + r - 1e-6])
        got = _flux64(bs, r, u)
        ref = np.array([_oracle(b, r, u) for b in bs])
        assert np.abs(got - ref).max() < 1e-14


class TestConsistency:
    def test_n2_reproduces_the_quadratic_core(self):
        """The polynomial path at N=2 must equal the dedicated quadratic
        core bit-for-bit-ish — they share s_0..s_2 and differ only in
        assembly."""
        bs = np.linspace(0.0, 1.15, 57)
        with mx.stream(mx.cpu):
            z = mx.array(bs, dtype=mx.float64)
            a = np.array(flux_dev_poly(z, R, [0.4, 0.25]))
            b = np.array(metalplanet.flux_dev(z, R, 0.4, 0.25))
        assert np.abs(a - b).max() < 1e-15

    def test_zero_outside_transit(self):
        with mx.stream(mx.cpu):
            z = mx.array(np.array([1.0 + R, 1.5, 3.0]), dtype=mx.float64)
            out = np.array(flux_dev_poly(z, R, [0.3, 0.2, 0.1, 0.05]))
        assert np.all(out == 0.0)

    def test_sqarea_sign_matters(self):
        """Regression: Heron's 16A^2 is NEGATIVE for a complete transit,
        and clamping it to zero (as the kite area is clamped) silently
        corrupts every s_n above n=2."""
        u = [0.3, 0.2, 0.1, 0.05]
        b = 0.3                      # complete transit: b < 1 - r
        got = _flux64(np.array([b]), R, u)[0]
        assert abs(got - _oracle(b, R, u)) < 1e-15

    def test_affine_transform_is_exact(self):
        rng = np.random.default_rng(0)
        for n in (1, 2, 4, 8):
            u = rng.uniform(-0.3, 0.3, n)
            A, c = greens_affine(n)
            assert np.abs(A @ u + c - greens_transform_np(u)).max() < 1e-14

    def test_traced_coefficients_match_host_constants(self):
        u = [0.3, 0.2, 0.1, 0.05]
        bs = np.array([0.05, 0.3, 0.6, 0.95, 1.05])
        with mx.stream(mx.cpu):
            z = mx.array(bs, dtype=mx.float64)
            host = np.array(flux_dev_poly(z, R, u))
            traced = np.array(flux_dev_poly(
                z, R, mx.array(np.array(u), dtype=mx.float64)))
        assert np.abs(host - traced).max() < 1e-15


class TestGradients:
    BOUND = np.array([0.0, 1e-12, R - 1e-9, R, R + 1e-9, 0.5,
                      1 - R - 1e-9, 1 - R, 1 - R + 1e-9, 1.0,
                      1 + R - 1e-9, 1 + R, 1 + R + 1e-9, 2.0])

    @pytest.mark.parametrize("n", [3, 4, 8, 12])
    @pytest.mark.parametrize("dtype", [mx.float64, mx.float32])
    def test_no_nan_gradients_on_the_boundaries(self, n, dtype):
        """mx.where evaluates both branches, so k^2 = onembmr2/(4zr) must
        stay finite where 4zr -> 0 and 1/k^2 where onembmr2 is floored."""
        u = [0.05] * n
        stream = mx.cpu if dtype == mx.float64 else mx.gpu
        with mx.stream(stream):
            z = mx.array(self.BOUND.astype(
                np.float64 if dtype == mx.float64 else np.float32),
                dtype=dtype)
            gz = np.array(mx.grad(
                lambda zz: mx.sum(flux_dev_poly(zz, R, u)))(z))
            gr = np.array(mx.grad(
                lambda rr: mx.sum(flux_dev_poly(z, rr, u)))(
                    mx.array(R, dtype=dtype)))
        assert np.isfinite(gz).all()
        assert np.isfinite(gr).all()

    def test_gradients_match_finite_differences(self):
        u = [0.3, 0.2, 0.1, 0.05]
        bs = np.array([0.05, 0.3, 0.6, 0.88, 0.95, 1.05])
        h = 1e-7
        with mx.stream(mx.cpu):
            z = mx.array(bs, dtype=mx.float64)
            gz = np.array(mx.grad(
                lambda zz: mx.sum(flux_dev_poly(zz, R, u)))(z))
            for i in range(bs.size):
                e = np.zeros_like(bs)
                e[i] = h
                fp = float(mx.sum(flux_dev_poly(
                    mx.array(bs + e, dtype=mx.float64), R, u)))
                fm = float(mx.sum(flux_dev_poly(
                    mx.array(bs - e, dtype=mx.float64), R, u)))
                assert abs(gz[i] - (fp - fm) / (2 * h)) < 1e-6

    def test_gradient_with_respect_to_coefficients(self):
        """Traced u makes the limb darkening itself differentiable."""
        u = np.array([0.3, 0.2, 0.1, 0.05])
        bs = np.array([0.05, 0.3, 0.6, 0.95])
        with mx.stream(mx.cpu):
            z = mx.array(bs, dtype=mx.float64)
            g = np.array(mx.grad(lambda uu: mx.sum(
                flux_dev_poly(z, R, uu)))(mx.array(u, dtype=mx.float64)))
            h = 1e-7
            for j in range(u.size):
                e = np.zeros_like(u)
                e[j] = h
                fp = float(mx.sum(flux_dev_poly(
                    z, R, mx.array(u + e, dtype=mx.float64))))
                fm = float(mx.sum(flux_dev_poly(
                    z, R, mx.array(u - e, dtype=mx.float64))))
                assert abs(g[j] - (fp - fm) / (2 * h)) < 1e-7


class TestFrontend:
    T = np.linspace(-0.12, 0.12, 301)

    def _model(self, u, law="polynomial", **kw):
        p = metalplanet.TransitParams()
        p.t0, p.per, p.rp, p.a, p.inc = 0.0, 3.456, R, 8.8, 87.07
        p.ecc, p.w, p.u, p.limb_dark = 0.0, 90.0, u, law
        for k, v in kw.items():
            setattr(p, k, v)
        return metalplanet.TransitModel(p, self.T), p

    def test_polynomial_n2_equals_quadratic(self):
        mq, pq = self._model([0.4, 0.25], law="quadratic")
        mp_, pp = self._model([0.4, 0.25])
        assert np.abs(mq.light_curve(pq) - mp_.light_curve(pp)).max() < 1e-14

    def test_coefficients_update_without_rebuild(self):
        m, p = self._model([0.3, 0.2, 0.1])
        f1 = m.light_curve(p)
        p.u = [0.1, 0.1, 0.1]
        f2 = m.light_curve(p)
        assert not np.allclose(f1, f2)
        m2, p2 = self._model([0.1, 0.1, 0.1])
        assert np.abs(f2 - m2.light_curve(p2)).max() < 1e-15

    def test_eccentric_polynomial_runs_and_skips_the_kernel(self):
        m, p = self._model([0.3, 0.2, 0.1], ecc=0.3, w=63.0)
        f = m.light_curve(p)
        assert f.min() < 0.99 and np.isfinite(f).all()
        assert list(m._compiled) == [(False, False)]   # kernel is quadratic-only

    def test_empty_coefficients_rejected(self):
        with pytest.raises(ValueError, match="at least|>= 1|needs"):
            self._model([])

    def test_non_polynomial_law_still_rejected(self):
        with pytest.raises(ValueError, match="not supported"):
            self._model([0.1, 0.2, 0.3, 0.4], law="nonlinear")
