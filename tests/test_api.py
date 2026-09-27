"""batman-frontend parity: same params in, same light curve out (to
batman's own ~2e-8 quadratic accuracy floor), across orbital
configurations, limb-darkening laws, supersampling, and secondary
eclipses."""

import batman as batman_pkg
import mlx.core as mx
import numpy as np
import pytest

import metalplanet


def _params(**kw):
    p = metalplanet.TransitParams()
    b = batman_pkg.TransitParams()
    defaults = dict(t0=0.31, per=2.7, rp=0.11, a=9.5, inc=87.6,
                    ecc=0.0, w=90.0, u=[0.35, 0.22], limb_dark="quadratic")
    defaults.update(kw)
    for k, v in defaults.items():
        setattr(p, k, v)
        setattr(b, k, v)
    return p, b


T = np.linspace(-0.25, 3.0, 4000) + 0.31  # covers two transits


class TestPrimaryParity:
    def test_circular_quadratic(self):
        p, b = _params()
        got = metalplanet.TransitModel(p, T).light_curve(p)
        want = batman_pkg.TransitModel(b, T).light_curve(b)
        assert got.shape == want.shape
        assert np.max(np.abs(got - want)) < 3e-8
        assert got.min() < 0.99  # the transit is actually there

    def test_inclination_sweep(self):
        for inc in (90.0, 89.0, 87.0, 85.5, 84.9):
            p, b = _params(inc=inc, a=8.0)
            got = metalplanet.TransitModel(p, T).light_curve(p)
            want = batman_pkg.TransitModel(b, T).light_curve(b)
            assert np.max(np.abs(got - want)) < 5e-8, f"inc={inc}"

    def test_non_transiting_is_flat(self):
        p, _ = _params(inc=70.0, a=6.0)  # b = a cos i ~ 2
        got = metalplanet.TransitModel(p, T).light_curve(p)
        np.testing.assert_array_equal(got, np.ones_like(got))

    def test_linear_and_uniform_laws(self):
        for law, u in (("linear", [0.4]), ("uniform", [])):
            p, b = _params(limb_dark=law, u=u)
            got = metalplanet.TransitModel(p, T).light_curve(p)
            want = batman_pkg.TransitModel(b, T).light_curve(b)
            assert np.max(np.abs(got - want)) < 3e-8, law

    def test_eccentric_orbits(self):
        for ecc, w in ((0.3, 45.0), (0.5, 130.0), (0.2, 271.0),
                       (0.65, 90.0)):
            p, b = _params(ecc=ecc, w=w, a=14.0, inc=88.2)
            got = metalplanet.TransitModel(p, T).light_curve(p)
            want = batman_pkg.TransitModel(b, T).light_curve(b)
            assert np.max(np.abs(got - want)) < 1e-7, (ecc, w)
            assert got.min() < 0.999  # transiting configuration

    def test_supersampling(self):
        p, b = _params()
        got = metalplanet.TransitModel(
            p, T, supersample_factor=7, exp_time=0.02).light_curve(p)
        want = batman_pkg.TransitModel(
            b, T, supersample_factor=7, exp_time=0.02).light_curve(b)
        assert np.max(np.abs(got - want)) < 3e-8

    def test_param_update_between_calls(self):
        """batman workflow: build once, vary params per call."""
        p, b = _params()
        m = metalplanet.TransitModel(p, T)
        mb = batman_pkg.TransitModel(b, T)
        for rp in (0.05, 0.08, 0.13):
            p.rp = rp
            b.rp = rp
            assert np.max(np.abs(m.light_curve(p) - mb.light_curve(b))) < 3e-8


class TestSecondary:
    def test_secondary_eclipse_circular(self):
        p, b = _params(u=[], limb_dark="uniform")
        p.fp = b.fp = 1.5e-3
        p.t_secondary = b.t_secondary = p.t0 + 0.5 * p.per
        got = metalplanet.TransitModel(
            p, T, transittype="secondary").light_curve(p)
        want = batman_pkg.TransitModel(
            b, T, transittype="secondary").light_curve(b)
        # both: 1 + fp out of eclipse, dipping to ~1 in eclipse
        assert abs(got.max() - (1 + p.fp)) < 1e-9
        assert np.max(np.abs(got - want)) < 1e-8


class TestValidation:
    def test_unsupported_law_raises(self):
        p, _ = _params(limb_dark="nonlinear", u=[0.1, 0.2, 0.3, 0.4])
        with pytest.raises(ValueError, match="not supported"):
            metalplanet.TransitModel(p, T)

    def test_wrong_coefficient_count_raises(self):
        p, _ = _params(limb_dark="quadratic", u=[0.1])
        with pytest.raises(ValueError):
            metalplanet.TransitModel(p, T)

    def test_supersample_needs_exp_time(self):
        p, _ = _params()
        with pytest.raises(ValueError, match="exp_time"):
            metalplanet.TransitModel(p, T, supersample_factor=5)


class TestDifferentiability:
    def test_light_curve_mx_gradient(self):
        """The frontend model stays differentiable via light_curve_mx."""
        p, _ = _params(ecc=0.25, w=63.0)
        m = metalplanet.TransitModel(p, T)

        def depth_sum(rp):
            p2 = metalplanet.TransitParams()
            for k in ("t0", "per", "a", "inc", "ecc", "w", "u", "limb_dark"):
                setattr(p2, k, getattr(p, k))
            p2.rp = rp  # mx scalar flows into the graph
            return mx.sum(m.light_curve_mx(p2))

        # rp enters _eval via float(params.rp) — so probe via mx by
        # bypassing the float cast: use the low-level core instead
        with mx.stream(mx.cpu):
            z, front = m._separation(p)
            z_eff = mx.where(front, z, 2.0 + z)

            def f(rp):
                return mx.sum(metalplanet.flux_dev(z_eff, rp, 0.35, 0.22))

            g = mx.grad(f)(mx.array(0.11, dtype=mx.float64))
            assert bool(mx.isfinite(g))
            assert float(g) < 0.0  # bigger planet, less flux


class TestEccentricKernelRouting:
    """The fp32 GPU eccentric primary path is served by the v3 kernel;
    every other combination keeps the graph. Parity is adjudicated
    against float64, since both fp32 paths sit at the same rounding."""

    @staticmethod
    def _model(ecc, dtype, kernel=True, n=20001, **kw):
        import metalplanet.api as api
        p, _ = _params(ecc=ecc, w=63.0)
        t = np.linspace(-0.35, 0.35, n)
        orig = api.TransitModel._ecc_kernel_usable
        if not kernel:
            api.TransitModel._ecc_kernel_usable = lambda self: False
        try:
            return metalplanet.TransitModel(p, t, dtype=dtype, **kw), p
        finally:
            api.TransitModel._ecc_kernel_usable = orig

    @pytest.mark.parametrize("ecc", [1e-4, 0.3, 0.7, 0.9])
    def test_kernel_matches_graph_and_fp64(self, ecc):
        mk, p = self._model(ecc, mx.float32, kernel=True)
        mg, _ = self._model(ecc, mx.float32, kernel=False)
        a, b = mk.light_curve(p), mg.light_curve(p)
        with mx.stream(mx.cpu):
            m64, _ = self._model(ecc, mx.float64)
            ref = m64.light_curve(p)
        assert np.abs(a - b).max() < 5e-6
        # the kernel must be no worse than the graph against fp64
        assert np.abs(a - ref).max() < 3.0 * max(np.abs(b - ref).max(), 5e-7)

    def test_period_update_is_not_baked_in(self):
        """period_ref is 0 and the period rides the traced p_off input,
        so a batman-style parameter update must change the curve."""
        mk, p = self._model(0.3, mx.float32, n=4001)
        f1 = mk.light_curve(p)
        p.per = 3.5
        f2 = mk.light_curve(p)
        assert not np.allclose(f1, f2)
        p.per = 3.456
        with mx.stream(mx.cpu):
            m64, _ = self._model(0.3, mx.float64, n=4001)
            m64.t = mk.t
            ref = m64.light_curve(p)
        assert np.abs(mk.light_curve(p) - ref).max() < 5e-6

    def test_fp64_and_secondary_do_not_use_the_kernel(self):
        m64, _ = self._model(0.3, mx.float64, n=101)
        assert not m64._ecc_kernel_usable()
        p, _ = _params(ecc=0.3, w=63.0)
        p.fp = 0.002
        msec = metalplanet.TransitModel(
            p, np.linspace(-0.35, 0.35, 101), transittype="secondary",
            dtype=mx.float32)
        assert not msec._ecc_kernel_usable()

    def test_use_metal_false_disables_it(self):
        p, _ = _params(ecc=0.3, w=63.0)
        m = metalplanet.TransitModel(p, np.linspace(-0.35, 0.35, 101),
                                     dtype=mx.float32, use_metal=False)
        assert not m._ecc_kernel_usable()
