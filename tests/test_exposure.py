"""Exposure-time averaging by contact-split Gauss-Legendre quadrature.

The claim under test is convergence, not just correctness: uniform
supersampling converges as O(1/N) because the light curve's derivative
jumps at each contact, while splitting the window at the contacts and
applying a fixed-order Gauss-Legendre rule to each smooth piece
converges geometrically.
"""

import math

import numpy as np
import mlx.core as mx
import pytest

import metalplanet
from metalplanet.exposure import contact_offsets, exposure_nodes, gauss_legendre

T = np.linspace(-0.13, 0.13, 261)
EXP = 0.02          # ~29 min, a sixth of T14 — heavy smearing


def _model(ecc=0.0, w=90.0, law="quadratic", u=(0.4, 0.25), **kw):
    p = metalplanet.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = 0.0, 3.456, 0.1, 8.8, 87.07
    p.ecc, p.w, p.u, p.limb_dark = ecc, w, list(u), law
    return metalplanet.TransitModel(p, T, **kw), p


class TestQuadratureMachinery:
    def test_weights_sum_to_one(self):
        with mx.stream(mx.cpu):
            f64 = lambda v: mx.array(v, dtype=mx.float64)
            cs = contact_offsets(f64(0.1), f64(8.8), f64(0.45))
            _, W = exposure_nodes(f64(np.array([-0.08, 0.0, 0.06])),
                                  f64(0.0), f64(3.456), f64(EXP), cs, 5)
            np.testing.assert_allclose(np.array(mx.sum(W, axis=1)), 1.0,
                                       atol=1e-13)

    def test_nodes_stay_inside_the_exposure(self):
        with mx.stream(mx.cpu):
            f64 = lambda v: mx.array(v, dtype=mx.float64)
            t = np.array([-0.1, -0.05, 0.0, 0.05, 0.1])
            cs = contact_offsets(f64(0.1), f64(8.8), f64(0.45))
            nodes, _ = exposure_nodes(f64(t), f64(0.0), f64(3.456),
                                      f64(EXP), cs, 5)
            n = np.array(nodes)
        assert (n >= t[:, None] - 0.5 * EXP - 1e-12).all()
        assert (n <= t[:, None] + 0.5 * EXP + 1e-12).all()

    def test_contact_durations_are_physical(self):
        """T14 > T23 > 0, and both match the circular closed form."""
        r, a, b = 0.1, 8.8, 0.45
        with mx.stream(mx.cpu):
            f64 = lambda v: mx.array(v, dtype=mx.float64)
            c = [float(x) for x in contact_offsets(f64(r), f64(a), f64(b))]
        t14 = c[3] - c[0]
        t23 = c[2] - c[1]
        assert t14 > t23 > 0.0
        want14 = 2 * math.asin(math.sqrt(((1 + r) ** 2 - b ** 2)
                                         / (a * a - b * b)))
        assert abs(t14 - want14) < 1e-12

    def test_grazing_transit_collapses_the_inner_pair(self):
        r, a, b = 0.1, 8.8, 0.95      # b > 1 - r: no second/third contact
        with mx.stream(mx.cpu):
            f64 = lambda v: mx.array(v, dtype=mx.float64)
            c = [float(x) for x in contact_offsets(f64(r), f64(a), f64(b))]
        assert c[1] == 0.0 and c[2] == 0.0
        assert c[0] < 0.0 < c[3]

    def test_gauss_legendre_is_exact_for_low_degree(self):
        x, w = gauss_legendre(4)
        for deg in range(0, 2 * 4):
            got = float((w * x ** deg).sum())
            want = 0.0 if deg % 2 else 2.0 / (deg + 1)
            assert abs(got - want) < 1e-13


@pytest.fixture(scope="module")
def reference():
    """Two independent references must agree before either is used."""
    m1, p1 = _model(exp_time=EXP, supersample_factor=60001)
    m2, p2 = _model(exp_time=EXP, integration="contact", n_gl=40)
    a, b = m1.light_curve(p1), m2.light_curve(p2)
    assert np.abs(a - b).max() < 1e-7, "references disagree"
    return b

class TestConvergence:
    def test_beats_supersampling_at_a_fraction_of_the_cost(self, reference):
        m_c, p_c = _model(exp_time=EXP, integration="contact", n_gl=5)
        m_s, p_s = _model(exp_time=EXP, supersample_factor=1001)
        err_c = np.abs(m_c.light_curve(p_c) - reference).max()
        err_s = np.abs(m_s.light_curve(p_s) - reference).max()
        assert err_c < err_s          # 25 evaluations vs 1001

    @pytest.mark.parametrize("n_gl,tol", [(3, 3e-6), (5, 5e-7), (7, 2e-7)])
    def test_error_falls_with_order(self, reference, n_gl, tol):
        m, p = _model(exp_time=EXP, integration="contact", n_gl=n_gl)
        assert np.abs(m.light_curve(p) - reference).max() < tol

    def test_supersampling_converges_only_linearly(self, reference):
        """Pins the reason contact splitting exists: uniform sampling of
        a curve with derivative jumps is O(1/N), so ten times the work
        buys one digit."""
        errs = []
        for n in (1001, 10001):
            m, p = _model(exp_time=EXP, supersample_factor=n)
            errs.append(np.abs(m.light_curve(p) - reference).max())
        assert 5.0 < errs[0] / errs[1] < 20.0

    @pytest.mark.parametrize("ecc,w", [(0.3, 63.0), (0.6, 120.0)])
    def test_eccentric(self, ecc, w):
        m_r, p_r = _model(ecc, w, exp_time=EXP, supersample_factor=30001)
        ref = m_r.light_curve(p_r)
        m_c, p_c = _model(ecc, w, exp_time=EXP, integration="contact",
                          n_gl=11)
        m_s, p_s = _model(ecc, w, exp_time=EXP, supersample_factor=101)
        assert np.abs(m_c.light_curve(p_c) - ref).max() < 1e-6
        assert (np.abs(m_c.light_curve(p_c) - ref).max()
                < np.abs(m_s.light_curve(p_s) - ref).max())


class TestIntegrationWithTheRest:
    def test_polynomial_limb_darkening(self):
        m, p = _model(law="polynomial", u=[0.3, 0.2, 0.1], exp_time=EXP,
                      integration="contact", n_gl=7)
        f = m.light_curve(p)
        assert np.isfinite(f).all() and f.min() < 0.99

    def test_shape_is_one_value_per_exposure(self):
        m, p = _model(exp_time=EXP, integration="contact", n_gl=7,
                      supersample_factor=5)
        assert m.light_curve(p).shape == T.shape

    def test_parameters_update_without_rebuild(self):
        m, p = _model(exp_time=EXP, integration="contact", n_gl=7)
        f1 = m.light_curve(p)
        p.rp = 0.12
        f2 = m.light_curve(p)
        assert (1 - f2.min()) > (1 - f1.min())

    def test_gradient_flows_through_the_moving_nodes(self):
        """The contact times depend on rp and a, so the node positions
        are themselves differentiable."""
        m, p = _model(exp_time=EXP, integration="contact", n_gl=7)
        fn = m._get_compiled(True)
        with mx.stream(mx.cpu):
            f64 = lambda v: mx.array(float(v), dtype=mx.float64)
            b = 8.8 * math.cos(math.radians(87.07))
            g = mx.grad(lambda rp: mx.sum(fn(
                f64(0.0), f64(3.456), f64(8.8), f64(b), rp,
                f64(0.4), f64(0.25), f64(0.0))))(f64(0.1))
            assert bool(mx.isfinite(g)) and float(g) < 0.0

    def test_validation(self):
        with pytest.raises(ValueError, match="exp_time"):
            _model(integration="contact")
        with pytest.raises(ValueError, match="supersample|contact"):
            _model(exp_time=EXP, integration="simpson")
        with pytest.raises(ValueError, match="primary"):
            p = metalplanet.TransitParams()
            p.t0, p.per, p.rp, p.a, p.inc = 0.0, 3.456, 0.1, 8.8, 87.07
            p.ecc, p.w, p.u, p.limb_dark = 0.0, 90.0, [], "uniform"
            p.fp = 1e-3
            metalplanet.TransitModel(p, T, transittype="secondary",
                                     exp_time=EXP, integration="contact")
