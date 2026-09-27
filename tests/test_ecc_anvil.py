"""E4: the 10-parameter eccentric anvil target.

Beyond ordinary forward/gradient correctness this pins the thing the
plan's review flagged: MetalPlanet clamps every numerical hazard, so an
unphysical geometry returns a perfectly ordinary finite log-likelihood.
Without the joint-constraint barrier the chain samples an improper
posterior and nothing complains.
"""

import math

import numpy as np
import mlx.core as mx
import pytest

from metalplanet.anvil import (PARAM_NAMES_ECC, ecc_constraint_penalty,
                               import_engine, make_ecc_target,
                               make_ecc_transit_flux)
from metalplanet.orbit import epoch_center_times

applemcmc, _ = import_engine()
PREF = 3.456
RNG = np.random.default_rng(4)


def _v(n=5, e=0.3, wdeg=63.0, **over):
    w = math.radians(wdeg)
    base = [0.004, -0.0007, 0.10, 0.30, 8.80, 0.4225, 0.3077,
            math.sqrt(e) * math.cos(w), math.sqrt(e) * math.sin(w), 0.0]
    v = np.tile(base, (n, 1))
    for key, val in over.items():
        v[:, PARAM_NAMES_ECC.index(key)] = val
    return v


def _x(m=4096):
    t = np.sort(RNG.uniform(0.0, 20.0, m))
    return epoch_center_times(t, t0_ref=0.0, period_ref=PREF)


class TestModel:
    @pytest.mark.parametrize("e", [0.0, 1e-5, 0.1, 0.5, 0.85])
    def test_metal_matches_graph_fp64(self, e):
        v, x = _v(e=e), _x()
        mk = make_ecc_transit_flux(PREF, core="metal")
        gr = make_ecc_transit_flux(PREF, core="graph")
        a32 = np.array(mk(mx.array(v.astype(np.float32)),
                          mx.array(x.astype(np.float32))), dtype=np.float64)
        with mx.stream(mx.cpu):
            a64 = np.array(gr(mx.array(v, dtype=mx.float64),
                              mx.array(x, dtype=mx.float64)),
                           dtype=np.float64)
        assert np.abs(a32 - a64).max() < 1e-6

    def test_fp64_cpu_falls_back_to_graph(self):
        mk = make_ecc_transit_flux(PREF, core="metal")
        v, x = _v(n=3), _x(1024)
        with mx.stream(mx.cpu):
            out = mk(mx.array(v, dtype=mx.float64),
                     mx.array(x, dtype=mx.float64))
            assert out.dtype == mx.float64
            assert np.isfinite(np.array(out)).all()

    def test_eccentricity_is_shrunk_not_nan(self):
        """A proposal outside the unit disc must stay finite: the solve
        never sees e > E_MAX_NUMERICAL."""
        v = _v(n=2, secosw=0.9, sesinw=0.9)          # e_raw = 1.62
        mk = make_ecc_transit_flux(PREF, core="metal")
        out = np.array(mk(mx.array(v.astype(np.float32)),
                          mx.array(_x(512).astype(np.float32))))
        assert np.isfinite(out).all()

    def test_gradients_finite_including_zero_eccentricity(self):
        mk = make_ecc_transit_flux(PREF, core="metal")
        x = mx.array(_x(2048).astype(np.float32))
        for e in (0.0, 1e-6, 0.4):
            v = mx.array(_v(n=4, e=e).astype(np.float32))
            g = mx.grad(lambda vv: mx.sum(mk(vv, x) ** 2))(v)
            assert np.isfinite(np.array(g)).all(), e


class TestConstraintBarrier:
    """No unphysical state may receive a finite, penalty-free log-prob."""

    def test_feasible_states_are_unpenalized(self):
        for e in (0.0, 0.1, 0.5, 0.8):
            p = np.array(ecc_constraint_penalty(
                mx.array(_v(n=3, e=e).astype(np.float32))))
            assert np.all(p == 0.0), e

    @pytest.mark.parametrize("over,why", [
        (dict(a=1.5), "periastron inside the star"),
        (dict(a=2.0, secosw=0.8, sesinw=0.3), "high e, small a"),
        (dict(b=8.0), "cos i > 1"),
        (dict(secosw=0.95, sesinw=0.95), "e > 1 before the shrink"),
    ])
    def test_unphysical_states_are_penalized(self, over, why):
        p = np.array(ecc_constraint_penalty(
            mx.array(_v(n=3, **over).astype(np.float32))))
        assert np.all(p < 0.0), why

    def test_penalty_is_smooth_and_zero_at_the_boundary(self):
        """Quadratic barrier, not a -inf wall: HMC needs a finite force."""
        a_crit = (1.0 + 0.10) / (1.0 - 0.30)          # a (1-e) = 1 + r
        vals = []
        for da in (-0.02, -0.005, 0.0, 0.005, 0.02):
            p = float(np.array(ecc_constraint_penalty(
                mx.array(_v(n=1, a=a_crit - da).astype(np.float32))))[0])
            vals.append(p)
        assert all(abs(v) < 1e-6 for v in vals[:3])   # feasible side: flat
        assert vals[3] < -1e-6 and vals[4] < vals[3]  # grows past it
        assert abs(vals[3]) < abs(vals[4])            # smoothly, not a wall

    def test_penalty_gradient_is_finite(self):
        v = mx.array(_v(n=3, a=1.5).astype(np.float32))
        g = mx.grad(lambda vv: mx.sum(ecc_constraint_penalty(vv)))(v)
        assert np.isfinite(np.array(g)).all()

    def test_target_loglike_rejects_unphysical(self, ):
        tt = make_ecc_target(n_data=4000, seed=3)
        good = mx.array(tt.truth_model[None, :].astype(np.float32))
        bad = tt.truth_model[None, :].copy()
        bad[0, PARAM_NAMES_ECC.index("a")] = 1.5
        lp_good = float(np.array(tt.loglike(good))[0])
        lp_bad = float(np.array(tt.loglike(mx.array(bad.astype(np.float32))))[0])
        assert lp_bad < lp_good - 1e3


@pytest.fixture(scope="module")
def tt():
    return make_ecc_target(n_data=20_000, seed=7)


class TestTarget:

    def test_chi2_at_truth(self, tt):
        with mx.stream(mx.cpu):
            lp = float(np.array(tt.loglike.hi(
                mx.array(tt.truth_model[None, :], dtype=mx.float64)))[0])
        chi2 = -2.0 * (lp + tt.loglike.log_offset_const)
        assert abs(chi2 / tt.loglike.n_data - 1.0) < 0.05

    def test_transform_roundtrip(self, tt):
        u = tt.transform.from_model_np(tt.truth_model)
        back = tt.transform.model_np(u)
        np.testing.assert_allclose(back, tt.truth_model, rtol=1e-9,
                                   atol=1e-12)

    def test_target_value_and_grad(self, tt):
        u = mx.array((tt.transform.from_model_np(tt.truth_model)[None, :]
                      + 1e-3 * RNG.standard_normal((8, 10))
                      ).astype(np.float32))
        val, grad = mx.value_and_grad(
            lambda uu: mx.sum(tt.target.log_prob(uu)))(u)
        assert np.isfinite(np.array(val)).all()
        assert np.isfinite(np.array(grad)).all()
        assert np.abs(np.array(grad)).max() > 0.0
