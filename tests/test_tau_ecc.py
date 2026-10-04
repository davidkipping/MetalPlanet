"""flux_dev_from_tau on an eccentric orbit (secosw / sesinw).

The orbit is the transit-anchored one (anchored.py), so tau is time since
inferior conjunction and b the impact parameter there. What is pinned:

* the fp64 graph path against TransitModel -- an independent route through
  the same physics, from orbital elements -- to round-off, all three rules;
* the contact rule against the exact integral, with its split points
  checked against bisected contact roots (exposure.contact_offsets_anchored);
* the fp32 kernel against that graph, values and gradients;
* e = 0 through the eccentric path against the circular path;
* ld_basis on an eccentric orbit: the same identity as test_ld_basis;
* gradients in all nine inputs, including (secosw, sesinw) at and near
  e = 0, where the anchored form exists to keep them finite and accurate;
* batches mixing circular and eccentric chains.
"""

import contextlib
import math

import numpy as np
import mlx.core as mx
import pytest

from metalplanet import api
from metalplanet.metal import flux_dev_from_tau, metal_available

P = 3.4525
EXP = 29.4 / 60.0 / 24.0
T0 = 0.35
U = (0.4225, 0.3077)
N_VEC = np.array([math.pi, 2.0 * math.pi / 3.0, math.pi / 2.0])
needs_metal = pytest.mark.skipif(not metal_available(),
                                 reason="Metal kernels unavailable")

# (name, a, inc_deg, r, e, w_deg)
ORBITS = [
    ("e0.3", 12.0, 88.0, 0.10, 0.3, 63.0),
    ("e0.5_grazing", 10.0, 86.0, 0.20, 0.5, -40.0),
    ("e0.7", 15.0, 88.5, 0.08, 0.7, 150.0),
    ("e0.1_big_rp", 9.0, 86.0, 0.30, 0.1, 10.0),
    ("e1e-4", 12.0, 88.0, 0.10, 1e-4, 200.0),
]
MODES = [("none", {}), ("contact", dict(exp_time=EXP, n_gl=5)),
         ("supersample", dict(exp_time=EXP, n_sub=11))]
TAU = np.linspace(-0.3, 0.3, 2001)


def geom(a, inc, r, e, w):
    """Elements -> this entry point's inputs: (a, b, r, secosw, sesinw)."""
    wr = math.radians(w)
    b = a * math.cos(math.radians(inc)) * (1 - e * e) / (1 + e * math.sin(wr))
    return dict(a=a, b=b, r=r, k=math.sqrt(e) * math.cos(wr),
                h=math.sqrt(e) * math.sin(wr))


def _ctx(dtype):
    return (mx.stream(mx.cpu) if dtype == mx.float64
            else contextlib.nullcontext())


def run(tau, g, dtype, basis=False, u=U, **kw):
    with _ctx(dtype):
        args = [mx.array(v, dtype=dtype) for v in (P, g["a"], g["b"], g["r"])]
        lds = [] if basis else [mx.array(x, dtype=dtype) for x in u]
        out = flux_dev_from_tau(mx.array(tau, dtype=dtype), *args, *lds,
                                secosw=mx.array(g["k"], dtype=dtype),
                                sesinw=mx.array(g["h"], dtype=dtype),
                                ld_basis=basis, **kw)
        mx.eval(out)
    return np.asarray(out, dtype=np.float64)


def api_dev(name, integ, n_gl=5, ssf=1):
    _, a, inc, r, e, w = next(o for o in ORBITS if o[0] == name)
    p = api.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = T0, P, r, a, inc
    p.ecc, p.w = e, w
    p.u, p.limb_dark = list(U), "quadratic"
    m = api.TransitModel(p, TAU + T0,
                         exp_time=0.0 if integ == "none" else EXP,
                         integration=("supersample" if integ == "none"
                                      else integ),
                         n_gl=n_gl, supersample_factor=ssf,
                         dtype=mx.float64)
    return np.asarray(m.light_curve(p), dtype=np.float64) - 1.0


IDS = [o[0] for o in ORBITS]
GEOMS = [geom(*o[1:]) for o in ORBITS]


# ---------------------------------------------------------------------------
# values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,g", list(zip(IDS, GEOMS)), ids=IDS)
@pytest.mark.parametrize("integ,kw", MODES, ids=[m[0] for m in MODES])
def test_fp64_graph_matches_transit_model(name, g, integ, kw):
    ref = api_dev(name, integ, n_gl=kw.get("n_gl", 5),
                  ssf=kw.get("n_sub", 1))
    got = run(TAU, g, mx.float64, integration=integ, **kw)
    assert np.abs(ref).max() > 1e-3
    assert np.abs(got - ref).max() < 1e-12


@pytest.mark.parametrize("name,g", list(zip(IDS, GEOMS)), ids=IDS)
def test_contact_rule_against_the_exact_integral(name, g):
    """n_gl = 40 converges to ~2e-11 however the window is split (a
    misplaced split only slows it), so it stands in for the exact
    integral. n_gl = 5 then has to be as good as the circular rule makes
    it: measured worst 6.8e-7, on e = 0.7's fast ingress. (Linearised
    contacts, as TransitModel used before 0.8.1, reach 2.9e-6 on the
    e = 0.5 orbit.)"""
    def at(n):
        return run(TAU, g, mx.float64, integration="contact",
                   exp_time=EXP, n_gl=n)
    ref = at(40)
    assert np.abs(ref - at(30)).max() < 1e-10
    assert np.abs(at(5) - ref).max() < 1e-6


def _bisect_contacts(g):
    """Every front-side z = 1 +- r crossing, by sweep + bisection in fp64."""
    from metalplanet.anchored import separation_anchored
    e = g["k"] ** 2 + g["h"] ** 2
    esw = g["h"] * math.sqrt(e)
    ci = g["b"] * (1 + esw) / (g["a"] * (1 - e * e))
    c = [mx.array(v, dtype=mx.float64) for v in (g["k"], g["h"], g["a"], ci)]

    def zf(t):
        with mx.stream(mx.cpu):
            z, front = separation_anchored(
                mx.array(2 * math.pi * np.asarray(t) / P, dtype=mx.float64),
                *c)
            mx.eval(z, front)
        return np.where(np.asarray(front), np.asarray(z), 1e9)

    tt = np.linspace(-0.3, 0.3, 60001)
    z = zf(tt)
    out = []
    for Z in (1 - g["r"], 1 + g["r"]):
        for i in np.nonzero(np.diff(np.sign(z - Z)))[0]:
            lo, hi = tt[i], tt[i + 1]
            slo = np.sign(z[i] - Z)
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if np.sign(zf([mid])[0] - Z) == slo:
                    lo = mid
                else:
                    hi = mid
            out.append(0.5 * (lo + hi))
    return np.sort(out)


@pytest.mark.parametrize("dtype", [mx.float64, mx.float32],
                         ids=["fp64", "fp32"])
def test_contacts_are_exact(dtype):
    from metalplanet.exposure import contact_offsets_anchored
    from metalplanet.metal import _ecc_shape
    cases = list(GEOMS) + [
        dict(a=10.0, b=0.90, r=0.15, k=0.45, h=0.40),    # grazing
        dict(a=10.0, b=0.79, r=0.20, k=-0.3, h=0.55)]    # nearly grazing
    for g in cases:
        with mx.stream(mx.cpu):
            v = [mx.array(g[q], dtype=dtype) for q in ("r", "a", "b", "k", "h")]
            ci = _ecc_shape(v[1], v[2], v[3], v[4])[2]
            got = np.array([float(np.asarray(x)) * P / (2 * math.pi)
                            for x in contact_offsets_anchored(*v, ci)])
        ref = _bisect_contacts(g)
        assert np.all(np.diff(got) >= 0.0)
        tol = 1e-11 if dtype == mx.float64 else 2e-7
        if g["b"] < 1.0 - g["r"]:
            assert len(ref) == 4
            assert np.abs(got - ref).max() < tol, g
        else:
            # grazing: the outer pair is exact; the inner pair has no root
            # and collapses (harmlessly) between them
            assert len(ref) == 2
            assert np.abs(got[[0, 3]] - ref).max() < tol, g
            assert ref[0] <= got[1] <= got[2] <= ref[1]


@needs_metal
@pytest.mark.parametrize("name,g", list(zip(IDS, GEOMS)), ids=IDS)
@pytest.mark.parametrize("integ,kw", MODES, ids=[m[0] for m in MODES])
def test_fp32_kernel_matches_fp64_graph(name, g, integ, kw):
    k32 = run(TAU, g, mx.float32, integration=integ, **kw)
    g64 = run(TAU, g, mx.float64, integration=integ, **kw)
    assert np.abs(k32 - g64).max() <= 5e-7


@pytest.mark.parametrize("dtype", [mx.float64, mx.float32],
                         ids=["fp64", "fp32"])
@pytest.mark.parametrize("integ,kw", MODES, ids=[m[0] for m in MODES])
def test_e0_matches_the_circular_path(dtype, integ, kw):
    """The anchored orbit degenerates exactly to the circular one."""
    if dtype == mx.float32 and not metal_available():
        pytest.skip("Metal kernels unavailable")
    g = dict(a=12.0, b=0.4, r=0.1, k=0.0, h=0.0)
    with _ctx(dtype):
        args = [mx.array(v, dtype=dtype) for v in (P, 12.0, 0.4, 0.1, *U)]
        c = flux_dev_from_tau(mx.array(TAU, dtype=dtype), *args,
                              integration=integ, **kw)
        mx.eval(c)
    e = run(TAU, g, dtype, integration=integ, **kw)
    tol = 1e-15 if dtype == mx.float64 else 2e-7
    assert np.abs(np.asarray(c, np.float64) - e).max() <= tol


@needs_metal
def test_far_side_and_out_of_transit_are_exactly_zero():
    """The secondary-eclipse side (front = v > 0 is false there) and the
    wings return exactly 0, on an orbit whose far side passes behind
    the star at small separation."""
    g = geom(6.0, 89.5, 0.1, 0.4, 100.0)
    tau = np.concatenate([np.linspace(-0.5 * P, -0.4, 300),
                          np.linspace(0.4, 0.5 * P, 300)])
    for integ, kw in MODES:
        assert (run(tau, g, mx.float32, integration=integ, **kw)
                == 0.0).all(), integ
        assert (run(tau, g, mx.float64, integration=integ, **kw)
                == 0.0).all(), integ


# ---------------------------------------------------------------------------
# ld_basis on an eccentric orbit
# ---------------------------------------------------------------------------

def _laws():
    q = np.random.default_rng(0).random((6, 2))
    s = np.sqrt(q[:, 0])
    return [(0.0, 0.0), (2.0, -1.0), (0.0, 1.0)] + list(
        zip(2.0 * s * q[:, 1], s * (1.0 - 2.0 * q[:, 1])))


@pytest.mark.parametrize("dtype", [mx.float64, mx.float32],
                         ids=["fp64", "fp32"])
@pytest.mark.parametrize("integ,kw", MODES, ids=[m[0] for m in MODES])
def test_ld_basis_identity(dtype, integ, kw):
    if dtype == mx.float32 and not metal_available():
        pytest.skip("Metal kernels unavailable")
    for g in GEOMS[:3]:
        B = run(TAU, g, dtype, basis=True, integration=integ, **kw)
        assert B.shape == TAU.shape + (3,)
        for u in _laws():
            c = np.array([1 - u[0] - u[1], u[0] + 2 * u[1], -u[1]])
            F = run(TAU, g, dtype, u=u, integration=integ, **kw)
            tol = 1e-14 if dtype == mx.float64 else 5e-8
            assert np.abs(B @ c / (N_VEC @ c) - F).max() < tol, u


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------

NAMES = ("tau", "per", "a", "b", "r", "u1", "u2", "secosw", "sesinw")


def _args(tau, g, dtype, basis):
    vals = [tau, P, g["a"], g["b"], g["r"]]
    vals += [] if basis else list(U)
    vals += [g["k"], g["h"]]
    return [mx.array(v, dtype=dtype) for v in vals]


def _call(v, basis, **kw):
    lds = [] if basis else v[5:7]
    return flux_dev_from_tau(*v[:5], *lds, secosw=v[-2], sesinw=v[-1],
                             ld_basis=basis, **kw)


def _grad(tau, g, ct, dtype, basis=False, **kw):
    with _ctx(dtype):
        c = mx.array(ct, dtype=dtype)
        args = _args(tau, g, dtype, basis)
        gr = mx.grad(lambda *v: mx.sum(c * _call(v, basis, **kw)),
                     argnums=tuple(range(len(args))))(*args)
        mx.eval(gr)
    return [np.asarray(x, dtype=np.float64) for x in gr]


def _cond(tau, g, ct, basis=False, **kw):
    """sum |ct * d out / d theta| per scalar parameter, fp64 forward mode:
    the scale an fp32 sum of those terms is accurate relative to."""
    with mx.stream(mx.cpu):
        base = _args(tau, g, mx.float64, basis)
        out = []
        for j in range(1, len(base)):
            t = [mx.zeros_like(x) for x in base]
            t[j] = mx.ones_like(base[j])
            _, (d,) = mx.jvp(lambda *v: _call(v, basis, **kw), base, t)
            out.append(float(np.sum(np.abs(ct * np.asarray(d)))))
    return out


@pytest.mark.parametrize("name,g", list(zip(IDS, GEOMS))[:3], ids=IDS[:3])
@pytest.mark.parametrize("integ,kw", MODES, ids=[m[0] for m in MODES])
def test_graph_gradients_vs_finite_differences(name, g, integ, kw):
    """fp64 central FD, contacts excluded from the tau comparison (the
    curve kinks there; see test_tau_kernel).

    Contact rule at n_gl = 10, not 5. The gradient freezes the split
    points (exact for the integral) while a finite difference also sees
    the *quadrature's* dependence on where they sit. That gap converges
    at third order in n_gl -- the ingress flux goes as dt^(3/2) -- and is
    the same function of sky speed for both orbits: measured equal to the
    circular path's at the sky-equivalent a, and to it exactly at
    e = 1e-4. At n_gl = 5 it reaches 3e-3 on d/dperiod for these faster-
    or slower-than-circular transits; at 10, 4e-4."""
    if integ == "contact":
        kw = dict(kw, n_gl=10)
    tau = TAU[::4]
    ct = np.random.default_rng(7).normal(size=tau.shape)
    kw = dict(kw, integration=integ)
    gr = _grad(tau, g, ct, mx.float64, **kw)
    base = [tau, P, g["a"], g["b"], g["r"], *U, g["k"], g["h"]]

    def val_any(v):
        with mx.stream(mx.cpu):
            vv = [mx.array(x, dtype=mx.float64) for x in v]
            o = flux_dev_from_tau(*vv[:7], secosw=vv[7], sesinw=vv[8], **kw)
            mx.eval(o)
        return np.asarray(o, dtype=np.float64)

    h0 = 1e-7
    vp, vm = list(base), list(base)
    vp[0], vm[0] = tau + h0, tau - h0
    fd_tau = ct * (val_any(vp) - val_any(vm)) / (2 * h0)
    cs = _bisect_contacts(g)
    if integ == "contact":
        edge = np.concatenate([cs - 0.5 * EXP, cs + 0.5 * EXP])
    elif integ == "supersample":
        off = np.linspace(-0.5 * EXP, 0.5 * EXP, kw["n_sub"])
        edge = (cs[:, None] - off[None, :]).ravel()
    else:
        edge = cs
    far = np.min(np.abs(tau[:, None] - edge[None, :]), axis=1) > 1e-5
    assert np.abs(gr[0] - fd_tau)[far].max() / np.abs(gr[0]).max() < 1e-5
    for j in range(1, 9):
        hj = max(abs(base[j]), 1.0) * 1e-6
        vp, vm = list(base), list(base)
        vp[j], vm[j] = base[j] + hj, base[j] - hj
        fd = float(np.sum(ct * (val_any(vp) - val_any(vm)) / (2 * hj)))
        scale = max(abs(fd), 1e-3 * abs(float(gr[j])) + 1e-12)
        assert abs(float(gr[j]) - fd) / scale < 2e-3, NAMES[j]


@needs_metal
@pytest.mark.parametrize("basis", [False, True], ids=["scalar", "basis"])
@pytest.mark.parametrize("name,g", list(zip(IDS, GEOMS)), ids=IDS)
@pytest.mark.parametrize("integ,kw", MODES, ids=[m[0] for m in MODES])
def test_kernel_gradients_match_fp64_graph(basis, name, g, integ, kw):
    shape = TAU.shape + ((3,) if basis else ())
    ct = np.random.default_rng(7).normal(size=shape)
    kw = dict(kw, integration=integ)
    g64 = _grad(TAU, g, ct, mx.float64, basis, **kw)
    g32 = _grad(TAU, g, ct, mx.float32, basis, **kw)
    assert all(np.isfinite(x).all() for x in g32)
    s = np.abs(g64[0]).max()
    assert np.abs(g32[0] - g64[0]).max() / s < 1e-3
    for j, c in enumerate(_cond(TAU, g, ct, basis, **kw), start=1):
        if c == 0.0:
            assert abs(float(g32[j])) < 1e-6
            continue
        assert abs(float(g32[j]) - float(g64[j])) / c < 5e-4, j


@pytest.mark.parametrize("dtype", [mx.float64, mx.float32],
                         ids=["fp64", "fp32"])
def test_gradients_at_exactly_e0_are_finite_and_zero_in_kh(dtype):
    """e = 0 is an interior point of the (secosw, sesinw) disc. Every
    orbit constant is second order in (k, h) there, so d/dk = d/dh = 0
    (to the 1e-30 floor under sqrt(e)); the other gradients must equal
    the circular path's."""
    if dtype == mx.float32 and not metal_available():
        pytest.skip("Metal kernels unavailable")
    g = dict(a=12.0, b=0.4, r=0.1, k=0.0, h=0.0)
    ct = np.random.default_rng(3).normal(size=TAU.shape)
    kw = dict(exp_time=EXP, integration="contact")
    ge = _grad(TAU, g, ct, dtype, **kw)
    assert all(np.isfinite(x).all() for x in ge)
    s = max(abs(float(x)) for x in ge[1:7])
    assert abs(float(ge[7])) < 1e-12 * s and abs(float(ge[8])) < 1e-12 * s
    with _ctx(dtype):
        c = mx.array(ct, dtype=dtype)
        args = [mx.array(v, dtype=dtype)
                for v in (TAU, P, 12.0, 0.4, 0.1, *U)]
        gc = mx.grad(lambda *v: mx.sum(c * flux_dev_from_tau(*v, **kw)),
                     argnums=tuple(range(7)))(*args)
        mx.eval(gc)
    tol = 1e-12 if dtype == mx.float64 else 1e-4
    for x, y in zip(ge[:7], gc):
        y = np.asarray(y, np.float64)
        assert np.abs(x - y).max() / (np.abs(y).max() + 1e-12) < tol


@needs_metal
def test_no_nan_gradients_on_a_sweep():
    rng = np.random.default_rng(11)
    n, m = 64, 400
    e = np.concatenate([[0.0, 1e-6, 1e-3], rng.uniform(0, 0.9, n - 3)])
    w = rng.uniform(-math.pi, math.pi, n)
    k, h = np.sqrt(e) * np.cos(w), np.sqrt(e) * np.sin(w)
    a = rng.uniform(6, 20, n)
    r = rng.uniform(0.02, 0.3, n)
    b = rng.uniform(0, 1.0 + r)
    tau = np.sort(rng.uniform(-0.2, 0.2, (n, m)), axis=1)
    ct = rng.normal(size=(n, m))
    for integ, kw in MODES:
        args = [mx.array(x, dtype=mx.float32)
                for x in (tau, np.full(n, P), a, b, r, np.full(n, U[0]),
                          np.full(n, U[1]), k, h)]
        c = mx.array(ct, dtype=mx.float32)
        gr = mx.grad(lambda *v: mx.sum(c * flux_dev_from_tau(
            *v[:7], secosw=v[7], sesinw=v[8], integration=integ, **kw)),
            argnums=tuple(range(9)))(*args)
        mx.eval(gr)
        for j, x in enumerate(gr):
            assert np.isfinite(np.asarray(x)).all(), (integ, NAMES[j])


# ---------------------------------------------------------------------------
# batching, shapes, arguments
# ---------------------------------------------------------------------------

def _mixed(n):
    """Distinct chains, a circular one (k = h = 0) among them."""
    e = np.linspace(0.0, 0.6, n)
    w = np.linspace(-2.0, 2.5, n)
    return dict(per=np.linspace(3.0, 4.2, n), a=np.linspace(8.0, 14.0, n),
                b=np.linspace(0.0, 0.8, n), r=np.linspace(0.05, 0.2, n),
                u1=np.linspace(0.2, 0.5, n), u2=np.linspace(0.1, 0.3, n),
                k=np.sqrt(e) * np.cos(w), h=np.sqrt(e) * np.sin(w))


@pytest.mark.parametrize("dtype", [mx.float64, mx.float32],
                         ids=["fp64", "fp32"])
@pytest.mark.parametrize("basis", [False, True], ids=["scalar", "basis"])
@pytest.mark.parametrize("integ,kw", MODES, ids=[m[0] for m in MODES])
def test_each_chain_is_its_own(dtype, basis, integ, kw):
    if dtype == mx.float32 and not metal_available():
        pytest.skip("Metal kernels unavailable")
    n, m = 6, 128
    p = _mixed(n)
    tau = np.stack([np.linspace(-0.12, 0.12, m) + 1e-3 * j
                    for j in range(n)])
    keys = ("per", "a", "b", "r") + (() if basis else ("u1", "u2"))

    def go(sel):
        with _ctx(dtype):
            v = [mx.array(np.atleast_1d(p[q][sel]), dtype=dtype)
                 for q in keys + ("k", "h")]
            o = flux_dev_from_tau(
                mx.array(np.atleast_2d(tau[sel]), dtype=dtype), *v[:-2],
                secosw=v[-2], sesinw=v[-1], ld_basis=basis,
                integration=integ, **kw)
            mx.eval(o)
        return np.asarray(o, dtype=np.float64)

    batch = go(slice(None))
    assert batch.shape == (n, m) + ((3,) if basis else ())
    tol = 1e-15 if dtype == mx.float64 else 0.0
    for j in range(n):
        assert np.abs(batch[j] - go(slice(j, j + 1))[0]).max() <= tol, j
    assert min(np.abs(batch[i] - batch[j]).max()
               for i in range(n) for j in range(i + 1, n)) > 1e-4


@pytest.mark.parametrize("dtype", [mx.float64, mx.float32],
                         ids=["fp64", "fp32"])
def test_shapes_and_smaller_batch_after_compile(dtype):
    if dtype == mx.float32 and not metal_available():
        pytest.skip("Metal kernels unavailable")
    n, m = 64, 200
    p = _mixed(n)
    tau = np.stack([np.linspace(-0.12, 0.12, m)] * n)
    ct = np.random.default_rng(1).normal(size=(n, m))
    keys = ("per", "a", "b", "r", "u1", "u2", "k", "h")
    with _ctx(dtype):
        one = flux_dev_from_tau(mx.array(tau[0], dtype=dtype), P, 10.0, 0.3,
                                0.1, *U, secosw=0.3, sesinw=-0.2,
                                exp_time=EXP)
        assert one.shape == (m,)

        @mx.compile
        def vg(t, *v):
            c = v[-1]
            return mx.value_and_grad(lambda *q: mx.sum(c * flux_dev_from_tau(
                *q[:7], secosw=q[7], sesinw=q[8], exp_time=EXP)),
                argnums=tuple(range(9)))(t, *v[:-1])

        def go(k):
            v, gr = vg(mx.array(tau[:k], dtype=dtype),
                       *[mx.array(p[q][:k], dtype=dtype) for q in keys],
                       mx.array(ct[:k], dtype=dtype))
            mx.eval(v, gr)
            return [np.asarray(x, np.float64) for x in gr]

        big = go(n)
        for k in (32, 1):
            small = go(k)
            tol = 1e-12 if dtype == mx.float64 else 1e-5
            for x, y in zip(small, big):
                s = np.abs(y[:k]).max() + 1e-12
                assert np.abs(x - y[:k]).max() / s < tol


def test_secosw_and_sesinw_come_as_a_pair():
    with pytest.raises(ValueError, match="secosw and sesinw"):
        flux_dev_from_tau(mx.zeros((4,)), P, 8.8, 0.3, 0.1, *U, secosw=0.1)
