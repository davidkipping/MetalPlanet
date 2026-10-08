"""Hybrid limb-darkening laws through the batman-style TransitModel.

The laws ride the existing vector-law plumbing (the polynomial law's):
one coefficient vector per model, the same compiled graphs, contact
integration, batched light_curves and the differentiable light_curve_mx.
These tests pin that plumbing for the three laws; the closed forms
themselves are tested in test_hybrid.py.
"""

import contextlib
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
def test_eccentric_runs(law):
    p = params(law, ecc=0.3, w=63.0)
    m = metalplanet.TransitModel(p, T)
    f = m.light_curve(p)
    assert f.min() < 0.99 and np.isfinite(f).all()
    assert list(m._compiled) == [(False, False)]          # fp64: the graph


# ---------------------------------------------------------------------------
# fp32 GPU: the fused z-input kernel behind _photom
# ---------------------------------------------------------------------------

MODES = {"plain": {}, "contact": dict(exp_time=EXP, integration="contact"),
         "supersample": dict(exp_time=EXP, supersample_factor=5)}


@contextlib.contextmanager
def kernel_spy():
    """Records, per trace of the hybrid z-kernel entry point, whether the
    GPU stream was active -- i.e. whether that trace took the kernel."""
    import metalplanet.api as api
    from metalplanet import metal as M
    orig, calls = api.flux_dev_metal_hybrid, []

    def spy(z, r, law, u, basis=False):
        calls.append(bool(M._gpu_stream_active()))
        return orig(z, r, law, u, basis=basis)

    api.flux_dev_metal_hybrid = spy
    try:
        yield calls
    finally:
        api.flux_dev_metal_hybrid = orig


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
@pytest.mark.parametrize("law", NAMES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("ecc", [0.0, 0.3], ids=["circ", "e0.3"])
def test_fp32_gpu_model_runs_the_fused_kernel(law, mode, ecc):
    """The photometry runs in metal_hybrid's z-kernel -- the spy sees it
    traced on the GPU stream -- and agrees with the fp32 graph
    (use_metal=False) at fp32 noise and with the fp64 model within 2e-6.
    The cache keys are the quadratic law's: the z-kernel decision lives in
    the trace, not the key."""
    p = params(law, ecc=ecc, w=63.0)
    m = metalplanet.TransitModel(p, T, dtype=mx.float32, **MODES[mode])
    with kernel_spy() as calls:
        got = m.light_curve(p)
    assert calls == [True]
    assert list(m._compiled) == [(ecc == 0.0, False)]
    graph = metalplanet.TransitModel(p, T, dtype=mx.float32, use_metal=False,
                                     **MODES[mode]).light_curve(p)
    f64 = metalplanet.TransitModel(p, T, **MODES[mode]).light_curve(p)
    assert np.abs(got - graph).max() < 1e-6
    assert np.abs(got - f64).max() < 2e-6


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
@pytest.mark.parametrize("law", NAMES)
@pytest.mark.parametrize("mode", ["plain", "contact"])
def test_fp32_light_curves_match_the_loop(law, mode):
    n = 4
    rng = np.random.default_rng(5)
    W = (ld.simplex_from_q_np(rng.random((n, LAWS[law].n_w)))
         if law != "hybrid2"
         else np.stack(ld.hybrid2_from_q_np(rng.random(n), rng.random(n)), -1))
    rp = np.linspace(0.08, 0.12, n)
    m = metalplanet.TransitModel(params(law), T, dtype=mx.float32,
                                 **MODES[mode])
    pa = params(law)
    pa.rp, pa.u = rp, W
    got = m.light_curves(pa)
    # The same per-point kernel on both sides, but light_curves forms z on
    # the anchored orbit (k = h = 0) and light_curve on the circular graph:
    # two fp32 expressions, apart by ~a*eps, times |dF/dz| <~ 0.1, plus the
    # rounding of 1 + dev -- about one fp32 ulp of 1 (measured 0.5 plain,
    # 1.0 contact, over seeds). Gate at four.
    ulp = float(np.finfo(np.float32).eps)
    for j in range(n):
        ref = m.light_curve(params(law, u=W[j], rp=float(rp[j])))
        assert np.abs(got[j] - ref).max() <= 4 * ulp, j


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
@pytest.mark.parametrize("mode", ["plain", "contact"])
def test_a_cpu_stream_first_call_does_not_pin_the_graph(mode):
    """The v0.9.2 trap does not apply to a z-kernel: its decision is made
    inside the trace, and mx.compile keeps one trace per stream, so the
    graph first traced on the CPU stream (graph photometry) is retraced
    on the GPU with the kernel. One cache key, two traces, two outcomes;
    the spy sees the stream each trace saw."""
    law = "hybrid5"
    m = metalplanet.TransitModel(params(law), T, dtype=mx.float32,
                                 **MODES[mode])
    with kernel_spy() as calls:
        with mx.stream(mx.cpu):
            a = m.light_curve(params(law))
        b = m.light_curve(params(law))
        m.light_curve(params(law))
    assert calls == [False, True]
    assert list(m._compiled) == [(True, False)]
    assert np.abs(a - b).max() < 1e-6


def test_float_radius_with_per_set_weights_takes_the_kernel_path():
    """_hybrid_dev's batched branch must not reshape a Python float."""
    m = metalplanet.TransitModel(params("hybrid5"), T, dtype=mx.float32)
    z = mx.array(np.linspace(0.5, 1.2, 8, dtype=np.float32)).reshape(2, 4)
    w = mx.array(np.array([WEIGHTS["hybrid5"]] * 2, np.float32))
    out = m._hybrid_dev(z, 0.1, w)
    ref = flux_dev_hybrid(z, 0.1, w, "hybrid5")
    assert out.shape == (2, 4)
    assert np.abs(np.asarray(out - ref, np.float64)).max() < 1e-6


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
@pytest.mark.parametrize("law", NAMES)
def test_fp32_light_curve_mx_gradients_through_the_kernel(law):
    """Gradients in rp, a, inc and every weight through the kernel's VJP
    against the fp64 model's, condition-scaled."""
    n_w = LAWS[law].n_w
    w0 = np.array(WEIGHTS[law])
    ct = np.random.default_rng(6).normal(size=T.size)

    def grads(dtype):
        m = metalplanet.TransitModel(params(law), T, dtype=dtype)
        ctx = mx.stream(mx.cpu) if dtype == mx.float64 else mx.stream(mx.gpu)
        with ctx:
            c = mx.array(ct, dtype=dtype)

            def loss(rp, a, inc, w):
                return mx.sum(c * m.light_curve_mx(
                    params(law, u=w, rp=rp, a=a, inc=inc)))

            g = mx.grad(loss, argnums=(0, 1, 2, 3))(
                *[mx.array(v, dtype=dtype) for v in (RP, A, INC)],
                mx.array(w0, dtype=dtype))
            mx.eval(g)
        return np.concatenate([np.atleast_1d(np.asarray(x, np.float64))
                               for x in g])

    g32, g64 = grads(mx.float32), grads(mx.float64)
    assert g32.shape == (3 + n_w,)
    scale = np.abs(g64).max()
    assert np.abs(g32 - g64).max() / scale < 2e-3


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
def test_array_eccentricity_route_uses_the_kernel_too():
    law = "hybrid4"
    m = metalplanet.TransitModel(params(law, ecc=0.3, w=63.0), T,
                                 dtype=mx.float32)
    with kernel_spy() as calls:
        out = np.asarray(m.light_curve_mx(params(law, ecc=mx.array(0.3),
                                                 w=63.0)), dtype=np.float64)
    assert calls == [True] and list(m._compiled) == ["ew"]
    assert np.abs(out - m.light_curve(params(law, ecc=0.3, w=63.0))).max() < 1e-6


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
