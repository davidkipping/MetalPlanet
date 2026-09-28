"""make_transit_target: fitting REAL photometry, and the review fixes.

This is the surface a Kepler/TESS fit uses, so the tests here are about
conditioning and about failing loudly: mission timestamps must survive
float32, unphysical geometry must not return a finite log-likelihood, and
the target must be copyable/picklable for checkpointing.
"""

import copy
import math
import pickle

import numpy as np
import mlx.core as mx
import pytest

import metalplanet
from metalplanet.anvil import (DEFAULT_BOUNDS, PenalizedLogLike,
                               ecc_constraint_penalty, import_engine,
                               make_transit_target)
from metalplanet.ld import u_to_q_np

applemcmc, _ = import_engine()
T0, PER, RP, A, INC = 2500.123456, 4.2345, 0.085, 11.3, 88.6


def _photometry(n=8000, yerr=3e-4, seed=7, ecc=0.0, w=90.0):
    """TESS-like BTJD timestamps, synthesised through the frontend."""
    rng = np.random.default_rng(seed)
    t = 2500.0 + np.sort(rng.uniform(0.0, 27.0, n))
    p = metalplanet.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = T0, PER, RP, A, INC
    p.ecc, p.w, p.u, p.limb_dark = ecc, w, [0.42, 0.21], "quadratic"
    flux = metalplanet.TransitModel(p, t).light_curve(p)
    return t, flux + yerr * rng.standard_normal(n), yerr, flux


def _truth_vector(tt, ecc=0.0, w=90.0):
    q1, q2 = u_to_q_np(0.42, 0.21)
    b = A * math.cos(math.radians(INC))
    kw = dict(r=RP, a=A, q1=float(q1), q2=float(q2))
    if tt.eccentric:
        # the model defines cos i = b (1 + e sin w) / (a (1 - e^2)), so the
        # b that reproduces the frontend's inclination is the INVERSE of
        # that factor times a cos i
        kw["b"] = b * (1.0 - ecc ** 2) / (1.0 + ecc * math.sin(math.radians(w)))
        kw["secosw"] = math.sqrt(ecc) * math.cos(math.radians(w))
        kw["sesinw"] = math.sqrt(ecc) * math.sin(math.radians(w))
    else:
        kw["b"] = b
    return tt.model_params(t0=T0, period=PER, **kw)


class TestRealDataTarget:
    def test_chi2_at_truth_is_one(self):
        t, y, yerr, _ = _photometry()
        tt = make_transit_target(t, y, yerr, T0 + 0.004, PER - 0.0008)
        v = _truth_vector(tt)
        with mx.stream(mx.cpu):
            lp = float(np.array(tt.loglike.hi(
                mx.array(v[None, :], dtype=mx.float64)))[0])
        chi2 = -2.0 * (lp + tt.loglike.log_offset_const)
        assert abs(chi2 / tt.loglike.n_data - 1.0) < 0.05

    def test_conditioning_is_owned_by_the_builder(self):
        """Mission times never reach the graph raw: t_ref is subtracted in
        float64 and the result is (per-orbit residual, orbit number)."""
        t, y, yerr, _ = _photometry()
        tt = make_transit_target(t, y, yerr, T0, PER)
        assert tt.t_ref == 2500.0
        assert tt.x64.shape == (2, t.size)
        assert np.abs(tt.x64[0]).max() <= 0.5 * PER + 1e-9   # residuals
        assert tt.x64[1].min() >= 0 and tt.x64[1].max() <= 27.0 / PER + 1
        assert np.abs(tt.y_fit).max() < 0.05                 # flux - 1

    def test_report_offsets_put_values_back_on_the_input_system(self):
        t, y, yerr, _ = _photometry()
        tt = make_transit_target(t, y, yerr, T0 + 0.004, PER - 0.0008)
        v = _truth_vector(tt)
        phys = tt.transform.to_physical(v)
        assert abs(phys[0] - T0) < 1e-6        # t0 back in BTJD
        assert abs(phys[1] - PER) < 1e-9
        assert abs(phys[-1] - 1.0) < 1e-9      # df0 reported around unity

    def test_model_params_roundtrip(self):
        t, y, yerr, _ = _photometry()
        tt = make_transit_target(t, y, yerr, T0 + 0.01, PER + 0.002)
        v = _truth_vector(tt)
        u = tt.transform.from_model_np(v)
        np.testing.assert_allclose(tt.transform.model_np(u), v,
                                   rtol=1e-9, atol=1e-12)

    def test_missing_parameter_is_named(self):
        t, y, yerr, _ = _photometry()
        tt = make_transit_target(t, y, yerr, T0, PER)
        with pytest.raises(ValueError, match="missing parameters"):
            tt.model_params(t0=T0, period=PER, r=RP)      # no b/a/q1/q2

    def test_gradient_is_finite_and_nonzero(self):
        t, y, yerr, _ = _photometry()
        tt = make_transit_target(t, y, yerr, T0, PER)
        u = mx.array((tt.transform.from_model_np(_truth_vector(tt))[None, :]
                      + 1e-3 * np.random.default_rng(0).standard_normal((8, 8))
                      ).astype(np.float32))
        val, g = mx.value_and_grad(
            lambda uu: mx.sum(tt.target.log_prob(uu)))(u)
        assert np.isfinite(np.array(val)).all()
        assert np.isfinite(np.array(g)).all()
        assert np.abs(np.array(g)).max() > 0.0

    @pytest.mark.parametrize("bad,match", [
        (dict(y=None), "match in shape"),
        (dict(nan=True), "finite"),
        (dict(yerr=-1e-4), "positive"),
        (dict(empty=True), "no data"),
    ])
    def test_rejects_bad_photometry(self, bad, match):
        t, y, yerr, _ = _photometry(n=200)
        if bad.get("y") is None and "y" in bad:
            y = y[:-1]
        if bad.get("nan"):
            y = y.copy(); y[5] = np.nan
        if "yerr" in bad:
            yerr = bad["yerr"]
        if bad.get("empty"):
            t, y, yerr = t[:0], y[:0], np.zeros(0)
        with pytest.raises(ValueError, match=match):
            make_transit_target(t, y, yerr, T0, PER)

    def test_unsorted_times_are_sorted(self):
        t, y, yerr, _ = _photometry(n=500)
        idx = np.random.default_rng(1).permutation(t.size)
        tt = make_transit_target(t[idx], y[idx], yerr, T0, PER)
        ref = make_transit_target(t, y, yerr, T0, PER)
        np.testing.assert_allclose(tt.x64, ref.x64)
        np.testing.assert_allclose(tt.y_fit, ref.y_fit)

    def test_custom_bounds_and_unknown_name(self):
        t, y, yerr, _ = _photometry(n=500)
        tt = make_transit_target(t, y, yerr, T0, PER,
                                 bounds={"r": (0.05, 0.12)})
        v = _truth_vector(tt)
        assert np.isfinite(tt.transform.from_model_np(v)).all()
        with pytest.raises(ValueError, match="unknown parameter"):
            make_transit_target(t, y, yerr, T0, PER, bounds={"nope": (0, 1)})

    def test_default_b_box_admits_grazing(self):
        """The synthetic target's b < 0.9 excluded grazing transits; a real
        fit must be allowed to explore them."""
        assert DEFAULT_BOUNDS["b"][1] > 1.0


class TestEccentricRealTarget:
    def test_barrier_is_attached_automatically(self):
        t, y, yerr, _ = _photometry(ecc=0.3, w=63.0)
        tt = make_transit_target(t, y, yerr, T0, PER, eccentric=True)
        assert isinstance(tt.loglike, PenalizedLogLike)
        assert len(tt.param_names) == 10
        good = mx.array(_truth_vector(tt, 0.3, 63.0)[None, :].astype(np.float32))
        bad = np.array(_truth_vector(tt, 0.3, 63.0))[None, :].copy()
        bad[0, tt.param_names.index("a")] = 1.05     # periastron inside star
        lp_good = float(np.array(tt.loglike(good))[0])
        lp_bad = float(np.array(tt.loglike(
            mx.array(bad.astype(np.float32))))[0])
        assert lp_bad < lp_good - 1e3

    def test_chi2_at_truth_eccentric(self):
        t, y, yerr, _ = _photometry(ecc=0.3, w=63.0)
        tt = make_transit_target(t, y, yerr, T0, PER, eccentric=True)
        with mx.stream(mx.cpu):
            lp = float(np.array(tt.loglike.hi(mx.array(
                _truth_vector(tt, 0.3, 63.0)[None, :], dtype=mx.float64)))[0])
        chi2 = -2.0 * (lp + tt.loglike.log_offset_const)
        assert abs(chi2 / tt.loglike.n_data - 1.0) < 0.05


class TestReviewFixes:
    def test_penalized_loglike_is_copyable(self):
        """__getattr__ delegated unconditionally, so copy/deepcopy/pickle
        recursed on `base` before __init__ had set it -> RecursionError
        where AttributeError was required. Checkpointing needs this."""
        t, y, yerr, _ = _photometry(n=300, ecc=0.2)
        tt = make_transit_target(t, y, yerr, T0, PER, eccentric=True)
        assert isinstance(copy.copy(tt.loglike), PenalizedLogLike)
        assert isinstance(copy.deepcopy(tt.loglike), PenalizedLogLike)
        with pytest.raises(AttributeError):
            tt.loglike.definitely_not_an_attribute

    def test_barrier_pushes_inward_outside_the_eccentricity_disc(self):
        """The model projects (secosw, sesinw) onto the e_max disc, so
        beyond it the likelihood is flat in the radial direction. Without a
        term on the UNPROJECTED e the force there is purely tangential and
        a chain thrown into the corners cannot find its way back."""
        base = [0.004, -0.0007, 0.1, 0.01, 8.8, 0.4225, 0.3077, 0.0, 0.0, 0.0]
        for k, h in ((0.95, 0.95), (0.9, -0.9), (-0.8, 0.8)):
            v = np.array(base, dtype=np.float32)[None, :].repeat(2, 0)
            v[:, 7], v[:, 8] = k, h
            vm = mx.array(v)
            pen = float(np.array(ecc_constraint_penalty(vm))[0])
            g = np.array(mx.grad(
                lambda vv: mx.sum(ecc_constraint_penalty(vv)))(vm))[0]
            radial = (g[7] * k + g[8] * h) / math.hypot(k, h)
            assert pen < 0.0, (k, h)
            assert radial < 0.0, (k, h, radial)      # inward

    def test_barrier_force_is_bounded(self):
        """A pure quadratic on |cos i| - 1 reached ~1e12 with ~1e11
        gradients once the disc projection drove 1 - e^2 to 0.002, which
        would collapse HMC's step size. The Huber form keeps it sane."""
        v = np.array([0.004, -0.0007, 0.1, 0.3, 8.8, 0.4225, 0.3077,
                      0.95, 0.95, 0.0], dtype=np.float32)[None, :]
        vm = mx.array(v.repeat(2, 0))
        g = np.array(mx.grad(
            lambda vv: mx.sum(ecc_constraint_penalty(vv)))(vm))
        assert np.abs(g).max() < 1e8
        assert np.isfinite(g).all()

    def test_feasible_states_are_exactly_unpenalized(self):
        v = np.array([0.004, -0.0007, 0.1, 0.3, 8.8, 0.4225, 0.3077,
                      0.3, 0.2, 0.0], dtype=np.float32)[None, :].repeat(3, 0)
        assert np.all(np.array(ecc_constraint_penalty(mx.array(v))) == 0.0)
