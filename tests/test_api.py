"""batman-frontend parity: same params in, same light curve out (to
batman's own ~2e-8 quadratic accuracy floor), across orbital
configurations, limb-darkening laws, supersampling, and secondary
eclipses."""

import batman as batman_pkg
import mlx.core as mx
import numpy as np
import pytest

import metalplanet
from metalplanet.metal import _gpu_stream_active, metal_available


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


class TestLightCurveMx:
    """light_curve_mx used to build its graph on the default device, so a
    default (fp64) TransitModel raised 'float64 is not supported on the
    GPU' unless the caller had already switched to the CPU stream."""

    @pytest.mark.parametrize("dtype", [None, mx.float32],
                             ids=["default_fp64", "fp32"])
    @pytest.mark.parametrize("kw", [{}, dict(exp_time=0.02,
                                             integration="contact"),
                                    dict(exp_time=0.02,
                                         supersample_factor=5)],
                             ids=["plain", "contact", "supersample"])
    @pytest.mark.parametrize("ecc,w", [(0.0, 90.0), (0.3, 63.0)],
                             ids=["circular", "eccentric"])
    def test_works_on_the_default_stream(self, dtype, kw, ecc, w):
        p, _ = _params(ecc=ecc, w=w)
        m = metalplanet.TransitModel(p, T, dtype=dtype, **kw)
        out = m.light_curve_mx(p)            # no stream context here
        mx.eval(out)                         # nor here
        assert out.dtype == m.dtype
        got = np.asarray(out, dtype=np.float64)
        if kw.get("supersample_factor", 1) > 1:   # raw grid: average it
            got = got.reshape(T.size, -1).mean(axis=1)
        tol = 1e-14 if m.dtype == mx.float64 else 5e-7
        assert np.abs(got - m.light_curve(p)).max() < tol


class TestEccentricKernelRouting:
    """The fp32 GPU primary path (circular and eccentric) is served by the
    fused model kernel;
    every other combination keeps the graph. Parity is adjudicated
    against float64, since both fp32 paths sit at the same rounding."""

    @staticmethod
    def _model(ecc, dtype, kernel=True, n=20001, **kw):
        """kernel=False is use_metal=False: the graph model. (Until
        0.10.4 this patched _kernel_usable around *construction* only,
        but the check runs when the graph is first compiled, so the
        'graph' model ran the kernel and the parity test compared the
        kernel with itself.)"""
        p, _ = _params(ecc=ecc, w=63.0)
        t = np.linspace(-0.35, 0.35, n)
        if not kernel:
            kw["use_metal"] = False
        return metalplanet.TransitModel(p, t, dtype=dtype, **kw), p

    @pytest.mark.skipif(not (metal_available() and _gpu_stream_active()),
                        reason="the kernel needs Metal and the GPU stream")
    @pytest.mark.parametrize("ecc", [1e-4, 0.3, 0.7, 0.9])
    def test_kernel_matches_graph_and_fp64(self, ecc):
        mk, p = self._model(ecc, mx.float32, kernel=True)
        mg, _ = self._model(ecc, mx.float32, kernel=False)
        a, b = mk.light_curve(p), mg.light_curve(p)
        assert list(mk._compiled) == [(False, True)]       # the kernel
        assert list(mg._compiled) == [(False, False)]      # the graph
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
        """The kernel decision is recorded in the compiled-graph cache
        key, (circular, kernel): fp64 cannot run it, and a secondary
        eclipse is a graph the kernel does not serve."""
        m64, p64 = self._model(0.3, mx.float64, n=101)
        assert not m64._kernel_usable()
        m64.light_curve(p64)
        assert list(m64._compiled) == [(False, False)]
        p, _ = _params(ecc=0.3, w=63.0)
        p.fp = 0.002
        msec = metalplanet.TransitModel(
            p, np.linspace(-0.35, 0.35, 101), transittype="secondary",
            dtype=mx.float32)
        msec.light_curve(p)
        assert list(msec._compiled) == [(False, False)]

    def test_use_metal_false_disables_it(self):
        p, _ = _params(ecc=0.3, w=63.0)
        m = metalplanet.TransitModel(p, np.linspace(-0.35, 0.35, 101),
                                     dtype=mx.float32, use_metal=False)
        assert not m._kernel_usable()
