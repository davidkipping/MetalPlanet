"""TransitModel.light_curve_mx: differentiable in every TransitParams field.

Up to v0.8.2 it read every field through float(), so a gradient taken
through it was silently zero. It now runs the compiled model graphs with
array-valued fields kept in the graph; eccentricity enters as (e, w)
directly (anchored.anchor_constants_ew), so d/d(ecc) is right at e = 0.

Gradients are checked against fp64 finite differences -- central, or
one-sided at e = 0, where e >= 0 makes the derivative one-sided -- for
every field, across integration modes, limb-darkening laws, transit types
and both precisions.
"""

import math

import numpy as np
import mlx.core as mx
import pytest

import metalplanet
from metalplanet.metal import metal_available

T = np.linspace(-0.15, 0.15, 601)
EXP = 0.02
LAWS = {"quadratic": [0.4, 0.25], "linear": [0.5], "uniform": [],
        "polynomial": [0.4, 0.25, 0.05]}
MODES = {"plain": {}, "contact": dict(exp_time=EXP, integration="contact"),
         "supersample": dict(exp_time=EXP, supersample_factor=5)}


def params(**kw):
    p = metalplanet.TransitParams()
    d = dict(t0=0.0, per=3.45, rp=0.1, a=8.8, inc=87.0, ecc=0.3, w=63.0,
             u=[0.4, 0.25], limb_dark="quadratic")
    d.update(kw)
    for k, v in d.items():
        setattr(p, k, v)
    return p


def fields(p):
    names = ["t0", "per", "rp", "a", "inc", "ecc", "w"]
    names += [f"u{i}" for i in range(len(p.u))]
    if p.fp is not None:
        names.append("fp")
    return names


def get(p, k):
    return p.u[int(k[1:])] if k.startswith("u") else getattr(p, k)


def put(p, k, v):
    if k.startswith("u"):
        u = list(p.u)
        u[int(k[1:])] = v
        p.u = u
    else:
        setattr(p, k, v)


def check_gradients(m, base_kw, tol, ct_seed=0):
    """Every field's gradient through light_curve_mx against fp64 FD."""
    p0 = params(**base_kw)
    names = fields(p0)
    n_out = np.asarray(m.light_curve(p0)).size * m.supersample_factor \
        if m.integration == "supersample" else T.size
    ct = np.random.default_rng(ct_seed).normal(size=n_out)

    def build(vals):
        q = params(**base_kw)
        for k, v in zip(names, vals):
            put(q, k, v)
        return q

    with mx.stream(mx.cpu):
        c = mx.array(ct)
        base = [get(p0, k) for k in names]
        g = mx.grad(lambda *v: mx.sum(c * m.light_curve_mx(build(v))),
                    argnums=tuple(range(len(names))))(
            *[mx.array(float(x), dtype=mx.float64) for x in base])
        mx.eval(g)

        def val(j, x):
            v = list(base)
            v[j] = x
            return float(np.sum(ct * np.asarray(m.light_curve_mx(build(v)))))

        for j, k in enumerate(names):
            got = float(g[j])
            assert math.isfinite(got), k
            h = 1e-6 * max(abs(base[j]), 1.0)
            if k == "ecc" and base[j] == 0.0:
                # one-sided, second order: e >= 0
                fd = (4 * val(j, h) - val(j, 2 * h) - 3 * val(j, 0.0)) / (2 * h)
            else:
                fd = (val(j, base[j] + h) - val(j, base[j] - h)) / (2 * h)
            if k == "w" and base_kw.get("ecc", 0.3) == 0.0:
                assert got == 0.0          # w is meaningless at e = 0
                continue
            assert abs(got - fd) <= tol * max(abs(fd), 1e-6), (k, got, fd)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("ecc", [0.3, 0.0, 0.7], ids=["e0.3", "e0", "e0.7"])
def test_gradients_fp64(mode, ecc):
    p = params(ecc=ecc)
    m = metalplanet.TransitModel(p, T, **MODES[mode])
    # one-sided FD at e = 0 under the contact rule is the loosest
    check_gradients(m, dict(ecc=ecc), tol=1e-4 if ecc == 0.0 else 2e-5)


@pytest.mark.parametrize("law", LAWS)
def test_gradients_every_law(law):
    kw = dict(limb_dark=law, u=LAWS[law])
    m = metalplanet.TransitModel(params(**kw), T,
                                 exp_time=EXP, integration="contact")
    check_gradients(m, kw, tol=2e-5)


def test_gradients_secondary_eclipse():
    kw = dict(fp=1e-3, t_secondary=3.45 / 2)
    p = params(**kw)
    m = metalplanet.TransitModel(p, T + 3.45 / 2, transittype="secondary")
    check_gradients(m, kw, tol=2e-5)


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
@pytest.mark.parametrize("mode", ["plain", "contact"])
@pytest.mark.parametrize("ecc", [0.3, 0.0], ids=["e0.3", "e0"])
def test_fp32_gradients_match_fp64(mode, ecc):
    """fp32 on the GPU against the fp64 model, relative to each gradient's
    condition scale: the sum of |ct * dF/dtheta| over points."""
    ct = np.random.default_rng(3).normal(size=T.size)
    names = fields(params(ecc=ecc))
    base = [get(params(ecc=ecc), k) for k in names]

    def grads(dtype):
        m = metalplanet.TransitModel(params(ecc=ecc), T, dtype=dtype,
                                     **MODES[mode])
        ctx = mx.stream(mx.cpu) if dtype == mx.float64 else mx.stream(mx.gpu)
        with ctx:
            c = mx.array(ct, dtype=dtype)

            def loss(*v):
                q = params(ecc=ecc)
                for k, x in zip(names, v):
                    put(q, k, x)
                return mx.sum(c * m.light_curve_mx(q))

            # each model differentiated in its own dtype: under mx.grad,
            # MLX needs the CPU stream for ANY float64 input (t0 = 0 here,
            # so fp32 holds it exactly)
            args = [mx.array(float(x), dtype=dtype) for x in base]
            g = mx.grad(loss, argnums=tuple(range(len(names))))(*args)
            mx.eval(g)
        return [float(x) for x in g]

    g32, g64 = grads(mx.float32), grads(mx.float64)
    for k, a, b in zip(names, g32, g64):
        assert math.isfinite(a), k
        assert abs(a - b) <= 2e-3 * max(abs(b), 1e-3 * max(map(abs, g64))), k


# ---------------------------------------------------------------------------
# values, routing, precision
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [mx.float64, mx.float32],
                         ids=["fp64", "fp32"])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("ecc", [0.0, 0.3], ids=["circular", "e0.3"])
def test_python_parameters_reproduce_light_curve_bitwise(dtype, mode, ecc):
    """All-Python fields take light_curve's own compiled graph."""
    if dtype == mx.float32 and not metal_available():
        pytest.skip("Metal unavailable")
    p = params(ecc=ecc)
    m = metalplanet.TransitModel(p, T, dtype=dtype, **MODES[mode])
    got = np.asarray(m.light_curve_mx(p), dtype=np.float64)
    if mode == "supersample":
        got = got.reshape(T.size, -1).mean(axis=1)
    assert np.array_equal(got, m.light_curve(p))


@pytest.mark.parametrize("dtype", [mx.float64, mx.float32],
                         ids=["fp64", "fp32"])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("ecc", [0.0, 0.3, 0.8], ids=["circ", "e0.3", "e0.8"])
def test_array_route_matches_light_curve(dtype, mode, ecc):
    """The (e, w) eccentric graph is the same function as light_curve's
    circular and (k, h) graphs."""
    if dtype == mx.float32 and not metal_available():
        pytest.skip("Metal unavailable")
    p = params(ecc=ecc)
    m = metalplanet.TransitModel(p, T, dtype=dtype, **MODES[mode])
    q = params(ecc=ecc)
    q.rp = mx.array(0.1, dtype=dtype)          # any array field will do
    with (mx.stream(mx.cpu) if dtype == mx.float64 else mx.stream(mx.gpu)):
        got = np.asarray(m.light_curve_mx(q), dtype=np.float64)
    if mode == "supersample":
        got = got.reshape(T.size, -1).mean(axis=1)
    tol = 1e-14 if dtype == mx.float64 else 2e-6
    assert np.abs(got - m.light_curve(p)).max() < tol


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
def test_absolute_t0_in_fp64_survives_an_fp32_model():
    """The reference-time subtraction happens in fp64 before the cast. The
    gradient in an fp64 input has to be taken on the CPU stream -- MLX's
    rule for any float64 input under mx.grad, not ours."""
    t0 = 2455000.3125
    p = params(t0=t0)
    m = metalplanet.TransitModel(p, T + t0, dtype=mx.float32)
    q = params(t0=mx.array(t0, dtype=mx.float64))
    got = np.asarray(m.light_curve_mx(q), dtype=np.float64)
    assert np.abs(got - m.light_curve(p)).max() < 2e-6
    with mx.stream(mx.cpu):
        g = mx.grad(lambda x: mx.sum(m.light_curve_mx(params(t0=x))))(
            mx.array(t0, dtype=mx.float64))
        mx.eval(g)
    assert math.isfinite(float(g)) and float(g) != 0.0


def test_callers_compile_and_grad():
    """No concrete-value branching on array fields, so a caller can wrap
    light_curve_mx in mx.compile and differentiate the result."""
    m = metalplanet.TransitModel(params(), T)
    with mx.stream(mx.cpu):
        def loss(rp, ecc):
            return mx.sum(m.light_curve_mx(params(rp=rp, ecc=ecc)))

        f = mx.compile(mx.value_and_grad(loss, argnums=(0, 1)))
        args = (mx.array(0.1, dtype=mx.float64),
                mx.array(0.3, dtype=mx.float64))
        v, (gr, ge) = f(*args)
        v2, (gr2, ge2) = mx.value_and_grad(loss, argnums=(0, 1))(*args)
        mx.eval(v, gr, ge, v2, gr2, ge2)
    assert abs(float(v) - float(v2)) < 1e-9
    assert abs(float(gr) - float(gr2)) < 1e-9 * abs(float(gr2))
    assert abs(float(ge) - float(ge2)) < 1e-9 * max(abs(float(ge2)), 1.0)


def test_eccentricity_is_validated():
    m = metalplanet.TransitModel(params(), T)
    for bad in (-0.1, 1.0):
        with pytest.raises(ValueError, match="eccentricity"):
            m.light_curve_mx(params(ecc=mx.array(bad, dtype=mx.float64)))


def test_works_on_the_default_stream():
    """The default model is fp64; no stream context anywhere."""
    m = metalplanet.TransitModel(params(), T)
    out = m.light_curve_mx(params(rp=mx.array(0.1, dtype=mx.float64)))
    mx.eval(out)
    assert out.dtype == mx.float64
