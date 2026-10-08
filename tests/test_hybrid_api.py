"""Hybrid limb-darkening laws through the batman-style TransitModel.

The laws ride the existing vector-law plumbing (the polynomial law's):
one coefficient vector per model, the same compiled graphs, contact
integration, batched light_curves and the differentiable light_curve_mx.
These tests pin that plumbing for the three laws; the closed forms
themselves are tested in test_hybrid.py.
"""

import math

import numpy as np
import mlx.core as mx
import pytest

import metalplanet
from metalplanet import ld
from metalplanet.hybrid import LAWS, flux_dev_hybrid
from metalplanet.metal import metal_available

T = np.linspace(-0.15, 0.15, 601)
P, A, INC, RP = 3.45, 8.8, 87.0, 0.1
EXP = 0.02
WEIGHTS = {"hybrid2": [0.3, 0.2], "hybrid4": [0.2, 0.2, 0.1, 0.1],
           "hybrid5": [0.2, 0.2, 0.1, 0.1, 0.1]}
NAMES = list(WEIGHTS)


def params(law, u=None, **kw):
    p = metalplanet.TransitParams()
    d = dict(t0=0.0, per=P, rp=RP, a=A, inc=INC, ecc=0.0, w=90.0,
             limb_dark=law, u=list(WEIGHTS[law]) if u is None else list(u))
    d.update(kw)
    for k, v in d.items():
        setattr(p, k, v)
    return p


def circular_z(t, t0=0.0, per=P, a=A, inc=INC):
    phi = 2 * math.pi * (t - t0) / per
    b = a * math.cos(math.radians(inc))
    z = np.sqrt((a * np.sin(phi)) ** 2 + (b * np.cos(phi)) ** 2)
    return z, np.cos(phi) > 0


@pytest.mark.parametrize("law", NAMES)
def test_light_curve_is_flux_dev_hybrid_on_the_orbit(law):
    m = metalplanet.TransitModel(params(law), T)
    got = m.light_curve(params(law))
    z, front = circular_z(T)
    with mx.stream(mx.cpu):
        f = np.asarray(flux_dev_hybrid(
            mx.array(np.where(front, z, 2 + z), dtype=mx.float64),
            mx.array(RP, dtype=mx.float64), WEIGHTS[law], law))
    assert np.abs(got - (1 + f)).max() < 1e-13
    assert got.min() < 0.99 and np.isfinite(got).all()


@pytest.mark.parametrize("law", NAMES)
def test_weights_update_without_rebuild(law):
    m = metalplanet.TransitModel(params(law), T)
    f1 = m.light_curve(params(law))
    u = list(WEIGHTS[law])
    u[0] += 0.3
    f2 = m.light_curve(params(law, u=u))
    assert not np.allclose(f1, f2)
    assert np.array_equal(f2, metalplanet.TransitModel(params(law, u=u), T)
                          .light_curve(params(law, u=u)))


@pytest.mark.parametrize("law", NAMES)
def test_wrong_or_changed_count_raises(law):
    n = LAWS[law].n_w
    with pytest.raises(ValueError, match=f"{law} takes {n} weights"):
        metalplanet.TransitModel(params(law, u=[0.1] * (n + 1)), T)
    m = metalplanet.TransitModel(params(law), T)
    with pytest.raises(ValueError, match="coefficient count changed"):
        m.light_curve(params(law, u=[0.1] * (n - 1)))
    with pytest.raises(ValueError, match="law changed"):
        m.light_curve(params("quadratic", u=[0.4, 0.25]))


@pytest.mark.parametrize("law", NAMES)
def test_eccentric_runs_on_the_graph(law):
    """Vector laws never take the fused kernel; the cache key says so."""
    p = params(law, ecc=0.3, w=63.0)
    m = metalplanet.TransitModel(p, T, dtype=mx.float32)
    f = m.light_curve(p)
    assert f.min() < 0.99 and np.isfinite(f).all()
    assert list(m._compiled) == [(False, False)]


@pytest.mark.parametrize("law", NAMES)
def test_contact_integration(law):
    """The contact rule needs nothing law-specific: n_gl = 7 agrees with
    n_gl = 40 (which is converged) to its own truncation, ~2e-8 on this
    heavily smeared exposure (a sixth of T14; the README's quadratic
    figure is 9e-8 at n_gl = 5), and with a heavy supersampling to the
    latter's own O(1/N) accuracy."""
    def lc(**kw):
        return metalplanet.TransitModel(params(law), T, exp_time=EXP,
                                        **kw).light_curve(params(law))
    c7 = lc(integration="contact", n_gl=7)
    c40 = lc(integration="contact", n_gl=40)
    s = lc(integration="supersample", supersample_factor=1001)
    assert np.abs(c7 - c40).max() < 1e-7
    assert np.abs(c7 - s).max() < 5e-6
    assert np.abs(c7 - lc(integration="supersample")).max() > 1e-5


@pytest.mark.parametrize("law", NAMES)
@pytest.mark.parametrize("form", ["sequence", "array"])
def test_light_curves_per_set_weights(law, form):
    n = 4
    rng = np.random.default_rng(1)
    W = (ld.simplex_from_q_np(rng.random((n, LAWS[law].n_w)))
         if law != "hybrid2"
         else np.stack(ld.hybrid2_from_q_np(rng.random(n), rng.random(n)), -1))
    rp = np.linspace(0.08, 0.12, n)
    m = metalplanet.TransitModel(params(law), T)
    if form == "sequence":
        got = m.light_curves([params(law, u=W[j], rp=float(rp[j]))
                              for j in range(n)])
    else:
        pa = params(law)
        pa.rp, pa.u = rp, W
        got = m.light_curves(pa)
    assert got.shape == (n, T.size)
    for j in range(n):
        ref = m.light_curve(params(law, u=W[j], rp=float(rp[j])))
        assert np.abs(got[j] - ref).max() < 1e-15, j


@pytest.mark.parametrize("law", NAMES)
def test_light_curves_contact_mode(law):
    m = metalplanet.TransitModel(params(law), T, exp_time=EXP,
                                 integration="contact", n_gl=7)
    ps = [params(law, rp=r) for r in (0.08, 0.1, 0.12)]
    got = m.light_curves(ps)
    for j, p in enumerate(ps):
        assert np.abs(got[j] - m.light_curve(p)).max() < 1e-15


@pytest.mark.parametrize("law", NAMES)
def test_fp32_model_matches_fp64(law):
    if not metal_available():
        pytest.skip("Metal unavailable")
    p = params(law)
    f32 = metalplanet.TransitModel(p, T, dtype=mx.float32).light_curve(p)
    f64 = metalplanet.TransitModel(p, T).light_curve(p)
    assert np.abs(f32 - f64).max() < 2e-6


@pytest.mark.parametrize("law", NAMES)
def test_light_curve_mx_gradient_in_every_weight(law):
    """light_curve_mx differentiates in each w_j through the affine
    shape map; FD in fp64 on the CPU."""
    m = metalplanet.TransitModel(params(law), T)
    n = LAWS[law].n_w
    w0 = np.array(WEIGHTS[law])
    ct = np.random.default_rng(2).normal(size=T.size)
    with mx.stream(mx.cpu):
        c = mx.array(ct)
        g = mx.grad(lambda w: mx.sum(c * m.light_curve_mx(params(law, u=w))))(
            mx.array(w0, dtype=mx.float64))
        g = np.asarray(g)
        # h = 1e-4: the flux is smooth in w (a ratio of two affine forms),
        # and the innermost pole's gradient is ~1e-6, which a 1e-6 step
        # cannot resolve against O(10) sums
        for j in range(n):
            h = 1e-4
            wp, wm = w0.copy(), w0.copy()
            wp[j] += h
            wm[j] -= h
            fd = np.sum(ct * (np.asarray(m.light_curve_mx(params(law, u=wp)))
                              - np.asarray(m.light_curve_mx(params(law, u=wm))))) / (2 * h)
            scale = max(abs(fd), 1e-3 * np.abs(g).max())
            assert abs(g[j] - fd) / scale < 1e-6, j


@pytest.mark.parametrize("law", NAMES)
def test_secondary_eclipse_ignores_limb_darkening(law):
    p = params(law, fp=1e-3, t_secondary=P / 2)
    m = metalplanet.TransitModel(p, T + P / 2, transittype="secondary")
    q = params("quadratic", u=[0.4, 0.25], fp=1e-3, t_secondary=P / 2)
    mq = metalplanet.TransitModel(q, T + P / 2, transittype="secondary")
    assert np.array_equal(m.light_curve(p), mq.light_curve(q))


def test_priors_feed_the_model():
    """The README recipe: weights from the uniform prior maps."""
    rng = np.random.default_rng(3)
    w5 = ld.simplex_from_q_np(rng.random(5))
    w2 = ld.hybrid2_from_q_np(rng.random(), rng.random())
    for law, w in (("hybrid5", w5), ("hybrid2", np.asarray(w2))):
        p = params(law, u=w)
        f = metalplanet.TransitModel(p, T).light_curve(p)
        assert np.isfinite(f).all() and f.min() < 0.99 and f.max() <= 1.0
