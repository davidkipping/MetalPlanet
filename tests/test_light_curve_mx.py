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


def check_gradients(m, base_kw, tol, ct_seed=0, fixed=()):
    """Every field's gradient through light_curve_mx against fp64 FD.
    ``fixed`` fields stay Python numbers (which selects the route)."""
    p0 = params(**base_kw)
    names = [k for k in fields(p0) if k not in fixed]
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
@pytest.mark.parametrize("mode", ["plain", "contact", "supersample"])
@pytest.mark.parametrize("ecc", [0.3, 0.0], ids=["e0.3", "e0"])
@pytest.mark.parametrize("ecc_as", ["array", "python"])
def test_fp32_gradients_match_fp64(mode, ecc, ecc_as):
    """fp32 on the GPU against the fp64 model, every route: an array ecc
    takes the (e, w) graph; a Python ecc keeps light_curve's graphs --
    the fused kernel in plain mode, which is differentiated here through
    its custom VJP, and the contact / supersample graphs otherwise."""
    ct = np.random.default_rng(3).normal(size=T.size)
    names = [k for k in fields(params(ecc=ecc))
             if not (ecc_as == "python" and k == "ecc")]
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
                f = m.light_curve_mx(q)
                if mode == "supersample":
                    f = mx.mean(mx.reshape(f, (T.size, -1)), axis=1)
                return mx.sum(c * f)

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


def test_scalar_value_helper():
    """Pins the one MLX dependency behind readable-vs-traced: a scalar
    reads eagerly and under mx.grad, is None under compile and vmap, and
    a vector is a caller error rather than 'traced'."""
    from metalplanet.api import _scalar_value
    assert _scalar_value(mx.array(0.3), "e") == pytest.approx(0.3)
    seen = {}
    mx.grad(lambda e: (seen.__setitem__("g", _scalar_value(e, "e")),
                       mx.sum(e))[1])(mx.array(0.3))
    assert seen["g"] == pytest.approx(0.3)
    mx.compile(lambda e: (seen.__setitem__("c", _scalar_value(e, "e")),
                          e)[1])(mx.array(0.3))
    mx.vmap(lambda e: (seen.__setitem__("v", _scalar_value(e, "e")),
                       e)[1])(mx.array([0.3]))
    assert seen["c"] is None and seen["v"] is None
    with pytest.raises(ValueError, match="must be a scalar"):
        _scalar_value(mx.array([0.3, 0.4]), "e")


def test_vector_eccentricity_is_rejected():
    """A (601,) ecc used to run, one eccentricity per time sample."""
    m = metalplanet.TransitModel(params(), T)
    with mx.stream(mx.cpu):
        with pytest.raises(ValueError, match="ecc must be a scalar"):
            m.light_curve_mx(params(ecc=mx.full((T.size,), 0.3,
                                                dtype=mx.float64)))


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
def test_fp64_ecc_that_rounds_to_one_in_fp32_is_rejected():
    """The check sees the value the graph sees: 1 - 1e-9 passes in fp64
    but is exactly 1.0 in fp32, which used to run to a flat curve."""
    m = metalplanet.TransitModel(params(), T, dtype=mx.float32)
    with pytest.raises(ValueError, match="eccentricity"):
        m.light_curve_mx(params(ecc=mx.array(1 - 1e-9, dtype=mx.float64)))


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
def test_ew_route_is_as_accurate_as_the_kh_route_in_fp32():
    """The (e, w) and (k, h) graphs are the same function (4e-16 apart in
    fp64) computed along different fp32 paths: each sits ~1.8e-7 from
    fp64 truth and they differ from each other by ~2.4e-7. That is path
    rounding, not a fold error -- a Python w (folded on the host) and an
    array w (in-graph) give the identical (e, w) result."""
    m32 = metalplanet.TransitModel(params(), T, dtype=mx.float32)
    truth = metalplanet.TransitModel(params(), T).light_curve(params())
    kh = m32.light_curve(params())
    ew = np.asarray(m32.light_curve_mx(params(ecc=mx.array(0.3))),
                    dtype=np.float64)
    ew_w = np.asarray(m32.light_curve_mx(params(ecc=mx.array(0.3),
                                                w=mx.array(63.0))),
                      dtype=np.float64)
    assert np.abs(ew - truth).max() <= 1.5 * np.abs(kh - truth).max()
    assert np.array_equal(ew, ew_w)


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
def test_contact_mode_compiles_once_across_streams():
    """The kernel never serves the contact rule, so its cache key must not
    split on the stream (0.9.2 compiled the same graph twice)."""
    m = metalplanet.TransitModel(params(), T, dtype=mx.float32,
                                 exp_time=EXP, integration="contact")
    assert not m._kernel_usable()
    m.light_curve(params())
    with mx.stream(mx.cpu):
        mx.eval(m.light_curve_mx(params(rp=mx.array(0.1))))
    assert list(m._compiled) == [(False, False)]


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
@pytest.mark.parametrize("ecc", [0.0, 0.3], ids=["circular", "e0.3"])
def test_fp32_python_ecc_runs_the_fused_kernel_and_is_bitwise(ecc):
    """With a Python ecc the differentiable call is light_curve's graph
    -- kernel included, which the cache key records -- and the output is
    light_curve's bit for bit, whichever other fields are arrays."""
    m = metalplanet.TransitModel(params(ecc=ecc), T, dtype=mx.float32)
    assert m._kernel_usable()
    q = params(ecc=ecc, rp=mx.array(0.1), w=mx.array(63.0),
               a=mx.array(8.8), inc=mx.array(87.0))
    got = np.asarray(m.light_curve_mx(q), dtype=np.float64)
    assert (ecc == 0.0, True) in m._compiled and "ew" not in m._compiled
    # a, inc, w as arrays go in-graph in fp32 (1-ulp from the host fold);
    # with those three Python, the fold is light_curve's and bitwise
    assert np.abs(got - m.light_curve(params(ecc=ecc))).max() < 1e-6
    q = params(ecc=ecc, rp=mx.array(0.1), per=mx.array(3.45))
    got = np.asarray(m.light_curve_mx(q), dtype=np.float64)
    assert np.array_equal(got, m.light_curve(params(ecc=ecc)))


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
def test_a_cpu_stream_call_does_not_poison_the_gpu_cache():
    """0.9.1 regression: the compiled-graph cache was keyed on circular
    only, and the kernel decision is made at build time from the active
    stream. The README recipe (an fp64 t0 gradient under the CPU stream)
    as a model's FIRST call then cached a kernel-less graph that every
    later GPU light_curve reused (3.4 ms vs 1.4 ms at 2e6 points)."""
    m = metalplanet.TransitModel(params(), T, dtype=mx.float32)
    with mx.stream(mx.cpu):
        g = mx.grad(lambda x: mx.sum(m.light_curve_mx(params(t0=x))))(
            mx.array(0.0, dtype=mx.float64))
        mx.eval(g)
    assert (False, False) in m._compiled          # built without the kernel
    got = m.light_curve(params())
    assert (False, True) in m._compiled           # and now WITH it
    fresh = metalplanet.TransitModel(params(), T, dtype=mx.float32)
    assert np.array_equal(got, fresh.light_curve(params()))


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
    q = params(ecc=mx.array(ecc, dtype=dtype))   # the (e, w) route
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


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.2])
def test_eccentricity_out_of_range_raises_when_readable(bad):
    """Python ecc on both entry points (light_curve used to return a flat
    curve for e >= 1; light_curves always raised), and an array ecc
    wherever its value can be read: eagerly, and under mx.grad."""
    m = metalplanet.TransitModel(params(), T)
    for f in (m.light_curve, m.light_curve_mx):
        with pytest.raises(ValueError, match="eccentricity"):
            f(params(ecc=bad))
    with mx.stream(mx.cpu):
        f64 = lambda v: mx.array(v, dtype=mx.float64)
        with pytest.raises(ValueError, match="eccentricity"):
            m.light_curve_mx(params(ecc=bad, rp=f64(0.1)))
        with pytest.raises(ValueError, match="eccentricity"):
            m.light_curve_mx(params(ecc=f64(bad)))
        with pytest.raises(ValueError, match="eccentricity"):
            mx.grad(lambda e: mx.sum(m.light_curve_mx(params(ecc=e))))(
                f64(bad))


def test_traced_eccentricity_out_of_range_is_nan_in_value_and_gradient():
    """Under vmap or a caller's compile the value cannot be read, so it
    cannot raise: out of range, the output AND every gradient are NaN --
    never a plausible curve (e = 1.5 used to give a flat one) and never
    a zero gradient beside a NaN loss (0.9.1 masked it to zero)."""
    m = metalplanet.TransitModel(params(), T)
    with mx.stream(mx.cpu):
        f64 = lambda v: mx.array(v, dtype=mx.float64)
        rows = mx.vmap(lambda e: m.light_curve_mx(params(ecc=e)))(
            f64([0.1, 1.5, -0.1]))
        comp = mx.compile(lambda e: m.light_curve_mx(params(ecc=e)))(
            f64(-0.1))
        mx.eval(rows, comp)
        rows = np.asarray(rows)
        assert np.abs(rows[0] - m.light_curve(params(ecc=0.1))).max() < 1e-14
        assert np.isnan(rows[1:]).all() and np.isnan(np.asarray(comp)).all()
        # gradients of a loss over ALL rows: the invalid row's is NaN, the
        # valid row's is still finite and right
        loss = lambda e: mx.sum(mx.vmap(
            lambda x: m.light_curve_mx(params(ecc=x)))(e))
        g = np.asarray(mx.grad(loss)(f64([0.1, 1.5])))
        g_ok = np.asarray(mx.grad(loss)(f64([0.1, 0.2])))
        gc = mx.compile(mx.grad(lambda e: mx.sum(m.light_curve_mx(
            params(ecc=e)))))(f64(1.5))
        mx.eval(gc)
    assert np.isnan(g[1]) and np.isnan(float(gc))
    assert np.isfinite(g[0]) and g[0] == g_ok[0]


@pytest.mark.parametrize("law,u", [("quadratic", [0.4, 0.25]),
                                   ("polynomial", [0.4, 0.25, 0.05])])
@pytest.mark.parametrize("kind", ["numpy", "mx_vector"])
def test_vector_limb_darkening(law, u, kind):
    """u as a numpy array (batman style) or an mx vector (to differentiate
    it whole); both used to crash light_curve_mx."""
    m = metalplanet.TransitModel(params(limb_dark=law, u=u), T)
    ref = m.light_curve(params(limb_dark=law, u=u))
    with mx.stream(mx.cpu):
        uu = (np.array(u) if kind == "numpy"
              else mx.array(u, dtype=mx.float64))
        got = np.asarray(m.light_curve_mx(params(limb_dark=law, u=uu)))
        assert np.abs(got - ref).max() < 1e-14
        if kind == "mx_vector":
            g = np.asarray(mx.grad(lambda v: mx.sum(m.light_curve_mx(
                params(limb_dark=law, u=v))))(mx.array(u, dtype=mx.float64)))
            for j in range(len(u)):
                h = 1e-6
                up, um = list(u), list(u)
                up[j] += h
                um[j] -= h
                fd = (m.light_curve(params(limb_dark=law, u=up)).sum()
                      - m.light_curve(params(limb_dark=law, u=um)).sum()
                      ) / (2 * h)
                assert abs(g[j] - fd) < 1e-6 * max(abs(fd), 1.0), j


# ---------------------------------------------------------------------------
# routing: only an array-valued ecc takes the (e, w) graph
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ecc,key", [(0.0, True), (0.3, False)],
                         ids=["circular", "e0.3"])
@pytest.mark.parametrize("mode", MODES)
def test_fixed_eccentricity_keeps_light_curves_graph(ecc, key, mode):
    """With a Python ecc, light_curve_mx differentiating every other field
    runs light_curve's own compiled graph (and its fused kernel on fp32),
    and its gradients are right there too."""
    m = metalplanet.TransitModel(params(ecc=ecc), T, **MODES[mode])
    check_gradients(m, dict(ecc=ecc), tol=2e-5, fixed=("ecc",))
    assert "ew" not in m._compiled and (key, False) in m._compiled


def test_works_on_the_default_stream():
    """The default model is fp64; no stream context anywhere."""
    m = metalplanet.TransitModel(params(), T)
    out = m.light_curve_mx(params(rp=mx.array(0.1, dtype=mx.float64)))
    mx.eval(out)
    assert out.dtype == mx.float64
