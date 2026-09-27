"""Transit-anchored eccentric orbit (E1 formulation decision).

Two things must hold: it is the same orbit as the direct formulation to
round-off in float64, and — the reason it exists — its float32
*gradients* w.r.t. the (sqrt(e) cos w, sqrt(e) sin w) sampling pair stay
accurate as e -> 0, where the direct path degrades without bound.
"""

import math

import numpy as np
import mlx.core as mx
import pytest

from metalplanet.anchored import (anchor_constants, separation_anchored,
                                  solve_sincos_delta, _one_minus_cos)
from metalplanet.kepler import (separation_keplerian,
                                mean_anomaly_offset_at_transit)
from metalplanet.flux import flux_dev
from metalplanet.trig import sincos

A, CI, R, U1, U2 = 8.8, 0.3 / 8.8, 0.1, 0.4225, 0.3077
ECCS = [0.0, 1e-8, 1e-3, 0.05, 0.3, 0.7, 0.95, 0.99]
WDEGS = [0.0, 47.0, 90.0, 180.0, 270.0, 333.0]


def _kh(e, w):
    return math.sqrt(e) * math.cos(w), math.sqrt(e) * math.sin(w)


def _f64(x):
    return mx.array(x, dtype=mx.float64)


class TestEquivalence:
    """Same orbit as kepler.separation_keplerian, to float64 round-off."""

    @pytest.mark.parametrize("e", ECCS)
    @pytest.mark.parametrize("wdeg", WDEGS)
    def test_flux_matches_direct_formulation(self, e, wdeg):
        w = math.radians(wdeg)
        k, h = _kh(e, w)
        phi = _f64(np.linspace(-math.pi, math.pi, 4001))
        with mx.stream(mx.cpu):
            za, fa = separation_anchored(phi, _f64(k), _f64(h),
                                         _f64(A), _f64(CI))
            zd, fd = separation_keplerian(
                phi + mean_anomaly_offset_at_transit(e, w),
                _f64(e), _f64(A), _f64(math.acos(CI)), _f64(w))
            Fa = np.array(mx.where(fa, flux_dev(za, R, U1, U2), 0.0))
            Fd = np.array(mx.where(fd, flux_dev(zd, R, U1, U2), 0.0))
        assert np.abs(Fa - Fd).max() < 1e-12

    def test_circular_limit_is_exact(self):
        """At e = 0 the anchored equations must degenerate to the
        circular orbit: delta = phi, u = -a sin phi, v = a cos phi."""
        phi = _f64(np.linspace(-math.pi, math.pi, 1001))
        with mx.stream(mx.cpu):
            z, front = separation_anchored(phi, _f64(0.0), _f64(0.0),
                                           _f64(A), _f64(CI))
            sp, cp = sincos(phi)
            z_circ = mx.sqrt((A * sp) ** 2 + (A * CI * cp) ** 2)
            assert float(mx.max(mx.abs(z - z_circ))) < 1e-12
            assert bool(mx.all(front == (cp > 0.0)))

    @pytest.mark.parametrize("e,wdeg", [(0.99, 270.0), (0.99, 90.0),
                                        (0.999, 270.0), (0.999, 90.0)])
    def test_extreme_eccentric_transits(self, e, wdeg):
        """Near-apastron (w = 270) and near-periastron (w = 90) transits
        at e -> 1: the geometries that broke the old conic tail."""
        w = math.radians(wdeg)
        k, h = _kh(e, w)
        phi = _f64(np.linspace(-0.3, 0.3, 2001))
        with mx.stream(mx.cpu):
            za, _ = separation_anchored(phi, _f64(k), _f64(h),
                                        _f64(A), _f64(CI))
            zd, _ = separation_keplerian(
                phi + mean_anomaly_offset_at_transit(e, w),
                _f64(e), _f64(A), _f64(math.acos(CI)), _f64(w))
            assert float(mx.max(mx.abs(za - zd))) < 1e-12


class TestSolver:
    @pytest.mark.parametrize("e", [0.0, 1e-6, 0.3, 0.9, 0.99, 0.999])
    def test_residual_of_its_own_equation(self, e):
        """phi = delta + es (1 - cos delta) - ec sin delta, mod 2 pi."""
        w = 1.1
        k, h = _kh(e, w)
        with mx.stream(mx.cpu):
            _, _, _, es, ec, _, _, _ = anchor_constants(_f64(k), _f64(h))
            phi = _f64(np.linspace(-math.pi, math.pi, 20001))
            sd, cd = solve_sincos_delta(phi, es, ec)
            d = mx.arctan2(sd, cd)
            res = d + es * _one_minus_cos(sd, cd) - ec * sd - phi
            res = res - 2 * math.pi * mx.round(res / (2 * math.pi))
            assert float(mx.max(mx.abs(res))) < 1e-13
            assert float(mx.max(mx.abs(sd * sd + cd * cd - 1.0))) < 1e-14

    def test_vjp_matches_finite_differences(self):
        """Implicit-function gradients of (sin delta, cos delta)."""
        w, e = 0.7, 0.4
        k, h = _kh(e, w)
        with mx.stream(mx.cpu):
            _, _, _, es0, ec0, _, _, _ = anchor_constants(_f64(k), _f64(h))
            phi = _f64(np.array([-2.0, -0.3, 0.0, 0.11, 1.9]))
            ct_s = _f64(np.array([0.3, -1.1, 0.7, 2.0, -0.4]))
            ct_c = _f64(np.array([-0.9, 0.2, 1.3, -0.6, 0.8]))

            def scalarized(p, es, ec):
                s, c = solve_sincos_delta(p, es, ec)
                return mx.sum(ct_s * s + ct_c * c)

            g = mx.grad(scalarized, argnums=(0, 1, 2))(phi, es0, ec0)
            hstep = 1e-7
            for idx, (arg, base) in enumerate(
                    [(0, phi), (1, es0), (2, ec0)]):
                args = [phi, es0, ec0]
                args[arg] = base + hstep
                fp = float(scalarized(*args))
                args[arg] = base - hstep
                fm = float(scalarized(*args))
                fd = (fp - fm) / (2 * hstep)
                got = float(mx.sum(g[idx]))
                assert abs(got - fd) <= 1e-5 * max(abs(fd), 1.0), idx


class TestConditioning:
    """The reason this module exists: float32 gradients w.r.t. (k, h)
    must not degrade as e -> 0 (benchmarks/v3_kh_grad_conditioning.py
    measures the direct path at 5% relative error at e = 1e-5)."""

    @staticmethod
    def _loss(k, h, phi):
        z, _ = separation_anchored(phi, k, h, A, CI)
        return mx.sum(z)

    @pytest.mark.parametrize("e", [1e-1, 1e-3, 1e-5, 1e-7])
    def test_fp32_gradient_accuracy_flat_in_e(self, e):
        w = 1.1
        k0, h0 = _kh(e, w)
        phi = np.linspace(-0.12, 0.12, 2001)
        g = mx.grad(self._loss, argnums=(0, 1))
        with mx.stream(mx.cpu):
            g64 = np.array([float(v) for v in g(
                _f64(k0), _f64(h0), _f64(phi))])
        g32 = np.array([float(v) for v in g(
            mx.array(np.float32(k0)), mx.array(np.float32(h0)),
            mx.array(phi.astype(np.float32)))])
        rel = np.abs(g32 - g64).max() / max(np.abs(g64).max(), 1e-30)
        assert rel < 1e-5, f"e={e}: relative gradient error {rel:.2e}"

    def test_gradients_are_finite_at_exactly_zero_eccentricity(self):
        """e = 0 is an interior point of the (k, h) disc, so the sampler
        visits it: no NaN from the masked cos w / sin w division."""
        phi = mx.array(np.linspace(-0.2, 0.2, 501).astype(np.float32))
        g = mx.grad(self._loss, argnums=(0, 1))(
            mx.array(np.float32(0.0)), mx.array(np.float32(0.0)), phi)
        assert all(np.isfinite(np.array(v)).all() for v in g)
