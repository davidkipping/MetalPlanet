"""Oblate planets, Stage 3: the fp32 Metal kernels (metal_oblate.py).

The truth is the fp64 graph (oblate.py / oblate_tau.py), itself pinned to
SquishierPlanet by Stages 1 and 2. Here: the device functions point by
point on SquishierPlanet's stress sets (columns and the dual-number
gradient); the tau kernels against the fp64 graph under every rule, orbit
and law, scalar and basis; the hand-assembled VJP against fp64 autodiff in
every input; the documented d/df = 0 below the fp32 switch; per-chain
independence; a no-NaN sweep; and the kernel routing.
"""

import math
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx
import pytest

import metalplanet as mp
from metalplanet import metal as M
from metalplanet import metal_oblate as MO
from metalplanet import oblate as O
from metalplanet.hybrid import LAWS, hybrid_norms

pytestmark = pytest.mark.skipif(
    not (M.metal_available() and M._gpu_stream_active()),
    reason="the oblate kernels need Metal and the GPU stream")

_SP = Path(__file__).resolve().parents[2] / "SquishierPlanet"
try:
    import importlib.util
    _spec = importlib.util.spec_from_file_location("sp_conftest", _SP / "tests" / "conftest.py")
    spc = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(spc)
except Exception:
    spc = None
needs_sp = pytest.mark.skipif(spc is None, reason="squishierplanet not importable")

P, A_RS = 3.45, 8.8
EXP = 1800.0 / 86400.0
W = {"hybrid2": [0.3, 0.2], "hybrid4": [0.2, 0.2, 0.1, 0.1],
     "hybrid5": [0.2, 0.2, 0.1, 0.1, 0.1]}
TAU = np.linspace(-0.13, 0.13, 257)
F_SW = MO._F_SW32


def canon(c):
    x0, y0, a, b = c
    return (x0, y0, a, b) if a >= b else (y0, -x0, b, a)


def stress(law):
    """SquishierPlanet's classes, limb and pole tangencies, plus random
    f in [f_sw, 0.5]; in the planetary domain. (x0, y0, A, B) rows."""
    cfg = []
    for k, cls in enumerate(("two_int", "inside", "disjoint", "origin_in_partial")):
        cfg += [canon(c) for c in spc.make_configs(700 + k, cls, 100)]
    for gap in (0.0, 1e-9, 1e-6, 1e-3, -1e-9, -1e-6, -1e-3):
        for kind in ("external", "internal"):
            for a, b, ph in ((0.12, 0.08, 1.1), (0.3, 0.2, 4.0), (0.2, 0.19, 0.5)):
                cfg.append(canon(spc.tangent_config(a, b, ph, kind, gap)))
    for e in LAWS[law].eps:
        rho = math.sqrt(1.0 + e)
        for gap in (0.0, 1e-12, 1e-9, 1e-6, -1e-9):
            for a, b, ph in ((0.12, 0.08, 1.1), (0.3, 0.2, 2.5), (0.02, 0.014, 4.0)):
                x0, y0, A, B = spc.tangent_config(a / rho, b / rho, ph, "internal", gap)
                cfg.append(canon((x0 * rho, y0 * rho, A * rho, B * rho)))
    rng = np.random.default_rng(3)
    for _ in range(200):
        f = 10 ** rng.uniform(math.log10(2 * F_SW), math.log10(0.5))
        r = rng.uniform(0.01, min(0.3, 0.95 * (1 - f) ** 1.5))
        A = r / math.sqrt(1 - f)
        d, t = rng.uniform(0, 1 + A), rng.uniform(0, 2 * np.pi)
        cfg.append((d * math.cos(t), d * math.sin(t), A, A * (1 - f)))
    x0, y0, A, B = np.array(cfg).T
    m = (A < 1.0) & (A * A / B <= 1.0) & (B / A >= 0.5) & (B / A < 1.0 - 2 * F_SW)
    return x0[m], y0[m], A[m], B[m]


def graph_cols(X, Y, r, f, th, law, wct=None):
    """fp64 graph columns at sky (X, Y, theta), and the gradient of wct.B."""
    with mx.stream(mx.cpu):
        f64 = lambda v: mx.array(np.asarray(v, np.float64), dtype=mx.float64)

        def fn(X_, Y_, r_, f_, t_):
            xx, yy = O.principal_frame(X_, Y_, t_)
            return O.shape_cols_oblate(xx, yy, r_, f_, law)
        B = np.asarray(fn(f64(X), f64(Y), f64(r), f64(f), f64(th)))
        if wct is None:
            return B, None
        G = mx.grad(lambda *a: mx.sum(f64(wct) * fn(*a)), argnums=(0, 1, 2, 3, 4))(
            f64(X), f64(Y), f64(r), f64(f), f64(th))
        return B, np.stack([np.asarray(g) for g in G], 1)


# ---------------------------------------------------------------------------
# the device functions, point by point
# ---------------------------------------------------------------------------

@needs_sp
@pytest.mark.parametrize("law", ["hybrid2", "hybrid4", "hybrid5"])
def test_device_columns_match_the_fp64_graph(law):
    x0, y0, A, B = stress(law)
    rng = np.random.default_rng(0)
    th = rng.uniform(0, np.pi, len(x0))
    X = x0 * np.cos(th) - y0 * np.sin(th)
    Y = x0 * np.sin(th) + y0 * np.cos(th)
    r, f = np.sqrt(A * B), 1 - B / A
    Bk, _ = MO._point_cols(X, Y, r, f, th, law)
    Bg, _ = graph_cols(X, Y, r, f, th, law)
    assert np.isfinite(Bk).all()
    assert (np.abs(Bk - Bg) / hybrid_norms(law)).max() < 1.5e-6


@needs_sp
@pytest.mark.parametrize("law", ["hybrid2", "hybrid5"])
def test_device_gradient_matches_fp64_autodiff(law):
    """The dual-number Jacobian, contracted with a random cotangent, in
    (X, Y, r, f, theta). Within 1e-6 of a limb contact the fp32 kernel
    cannot resolve the sliver (crossings closer than its pair tolerance
    are a tangency), so those geometries are compared on value only."""
    x0, y0, A, B = stress(law)
    d = np.hypot(x0, y0)
    far = (np.abs(d - 1 - A) > 1e-5) & (np.abs(d - 1 + A) > 1e-5) \
        & (np.abs(d - 1 - B) > 1e-5) & (np.abs(d - 1 + B) > 1e-5)
    x0, y0, A, B = x0[far], y0[far], A[far], B[far]
    rng = np.random.default_rng(1)
    th = rng.uniform(0, np.pi, len(x0))
    X = x0 * np.cos(th) - y0 * np.sin(th)
    Y = x0 * np.sin(th) + y0 * np.cos(th)
    r, f = np.sqrt(A * B), 1 - B / A
    wct = rng.normal(size=(len(x0), LAWS[law].n_col))
    _, gk = MO._point_cols(X, Y, r, f, th, law, wct, grad=True)
    _, gg = graph_cols(X, Y, r, f, th, law, wct)
    assert np.isfinite(gk).all()
    gmax = np.abs(gg).max(1, keepdims=True)
    # typically fp32's level; at worst 1e-3 of the gradient, plus 5e-4
    # absolute on grazes whose sliver of overlap has a gradient of 1e-4
    # (its crossings sit at the edge of what fp32 resolves)
    assert np.median(np.abs(gk - gg) / np.maximum(gmax, 1e-3)) < 1e-5
    assert (np.abs(gk - gg) <= 1e-3 * gmax + 5e-4).all()


# ---------------------------------------------------------------------------
# the tau kernels against the fp64 graph
# ---------------------------------------------------------------------------

CASES = [(0.3, 0.1, 0.3, 0.6, None), (1.02, 0.12, 0.4, 0.8, None),
         (0.4, 0.1, 0.2, 2.0, (0.4, 0.3)), (0.0, 0.1, 0.1, 0.3, None),
         (0.5, 0.1, 3e-6, 0.4, None), (0.2, 0.15, 0.5, 1.5, (0.2, -0.5))]


def _flux(tau, dtype, law, b, r, f, th, ecc, basis=False, **kw):
    kw = dict(kw, f=f, theta=th, limb_dark=law)
    if ecc:
        kw.update(secosw=ecc[0], sesinw=ecc[1])
    if basis:
        kw["ld_basis"] = True
    else:
        kw["u"] = W[law]
    stream = mx.cpu if dtype == mx.float64 else mx.gpu
    with mx.stream(stream):
        out = mp.flux_dev_from_tau(mx.array(tau, dtype=dtype), P, A_RS, b, r, **kw)
        mx.eval(out)
        return np.asarray(out.astype(mx.float32) if dtype == mx.float32 else out,
                          dtype=np.float64)


@pytest.mark.parametrize("law", ["hybrid2", "hybrid4", "hybrid5"])
@pytest.mark.parametrize("basis", [False, True], ids=["flux", "basis"])
@pytest.mark.parametrize("integration,extra", [
    ("none", {}), ("contact", {"exp_time": EXP}),
    ("supersample", {"exp_time": EXP, "n_sub": 7})])
def test_tau_kernel_matches_the_fp64_graph(law, basis, integration, extra):
    for case in CASES:
        k64 = _flux(TAU, mx.float64, law, *case, basis, integration=integration, **extra)
        k32 = _flux(TAU, mx.float32, law, *case, basis, integration=integration, **extra)
        assert np.isfinite(k32).all()
        assert np.abs(k32 - k64).max() < (5e-7 if basis else 2e-7), case


# ---------------------------------------------------------------------------
# the VJP against fp64 autodiff
# ---------------------------------------------------------------------------

_NAMES = ["tau", "period", "a", "b", "r", "f", "theta", "w", "secosw", "sesinw"]


def _grads(dtype, integration, extra, b, r, f, th, ecc, law, basis):
    k, h = ecc if ecc else (0.0, 0.0)
    ncol = LAWS[law].n_col
    ct = np.random.default_rng(1).normal(size=TAU.shape + ((ncol,) if basis else ()))
    stream = mx.cpu if dtype == mx.float64 else mx.gpu
    with mx.stream(stream):
        A = lambda v: mx.array(np.asarray(v, np.float64), dtype=dtype)

        def loss(t_, per, a, b_, r_, f_, th_, w_, k_, h_):
            kw = dict(limb_dark=law, f=f_, theta=th_, secosw=k_, sesinw=h_,
                      integration=integration, **extra)
            if basis:
                kw["ld_basis"] = True
            else:
                kw["u"] = w_
            return mx.sum(A(ct) * mp.flux_dev_from_tau(t_, per, a, b_, r_, **kw))
        g = mx.grad(loss, argnums=tuple(range(10)))(
            *(A(v) for v in (TAU, P, A_RS, b, r, f, th, W[law], k, h)))
        mx.eval(g)
        return [np.asarray(x.astype(mx.float32) if dtype == mx.float32 else x,
                           dtype=np.float64) for x in g]


@pytest.mark.parametrize("law,basis", [("hybrid5", False), ("hybrid2", False),
                                       ("hybrid4", True)])
@pytest.mark.parametrize("integration,extra", [
    ("none", {}), ("contact", {"exp_time": EXP}),
    ("supersample", {"exp_time": EXP, "n_sub": 5})])
def test_vjp_matches_fp64_autodiff(law, basis, integration, extra):
    for case in CASES:
        g64 = _grads(mx.float64, integration, extra, *case, law, basis)
        g32 = _grads(mx.float32, integration, extra, *case, law, basis)
        for name, a64, a32 in zip(_NAMES, g64, g32):
            if basis and name == "w":
                continue
            if name == "f" and case[2] < F_SW:
                continue                      # documented: zero below f_sw
            assert np.isfinite(a32).all(), (name, case)
            scale = max(np.abs(a64).max(), 1e-3 * np.abs(g64[4]).max())
            assert np.abs(a32 - a64).max() <= 1e-3 * scale, (name, case)


def test_d_df_is_zero_below_the_switch():
    """Below f_sw = 1e-5 a chain takes the spherical columns, so d/df is 0
    there (the flattening moves the flux by < 4e-8, fp32's noise)."""
    g = _grads(mx.float32, "none", {}, 0.4, 0.1, 0.5 * F_SW, 0.4, None,
               "hybrid4", False)
    assert float(np.abs(g[5]).max()) == 0.0
    g = _grads(mx.float32, "none", {}, 0.4, 0.1, 0.1, 0.4, None, "hybrid4", False)
    assert float(np.abs(g[5]).max()) > 0.0


# ---------------------------------------------------------------------------
# chains, sweeps, routing
# ---------------------------------------------------------------------------

def test_each_chain_is_its_own():
    """(n, m) with per-chain parameters equals the per-chain calls, and a
    smaller batch after a larger one is unaffected."""
    n = 5
    rng = np.random.default_rng(2)
    pars = dict(b=rng.uniform(0, 0.8, n), r=rng.uniform(0.05, 0.15, n),
                f=rng.uniform(0.05, 0.5, n), theta=rng.uniform(0, np.pi, n))
    tau = np.stack([TAU] * n).astype(np.float32)
    w = np.array(W["hybrid4"], np.float32)
    rows = mp.flux_dev_from_tau(tau, P, A_RS, mx.array(pars["b"], dtype=mx.float32),
                                mx.array(pars["r"], dtype=mx.float32), limb_dark="hybrid4",
                                u=w, f=mx.array(pars["f"], dtype=mx.float32),
                                theta=mx.array(pars["theta"], dtype=mx.float32),
                                exp_time=EXP)
    rows = np.asarray(rows)
    for j in range(n):
        one = np.asarray(mp.flux_dev_from_tau(
            TAU.astype(np.float32), P, A_RS, float(pars["b"][j]), float(pars["r"][j]),
            limb_dark="hybrid4", u=w, f=float(pars["f"][j]),
            theta=float(pars["theta"][j]), exp_time=EXP))
        assert np.abs(rows[j] - one).max() < 1e-7, j


@pytest.mark.parametrize("integration,extra", [("none", {}), ("contact", {"exp_time": EXP})])
def test_no_nan_on_a_sweep(integration, extra):
    """Centre crossings, f = 0 and either side of f_sw, theta at 0 and
    pi/2, grazes and near-tangent grazes: values and gradients finite."""
    tau = np.linspace(-0.13, 0.13, 513).astype(np.float32)
    cases = [(0.0, 0.1, 0.3, 0.0), (0.0, 0.1, 0.0, 0.5), (0.4, 0.1, 0.3, math.pi / 2),
             (1.05, 0.15, 0.5, 0.6), (1.167483, 0.15, 0.5, math.pi / 4),
             (0.9, 0.12, 0.2, 1.3), (0.3, 0.1, 0.99 * F_SW, 0.2),
             (0.3, 0.1, 1.01 * F_SW, 0.2), (0.0, 0.3, 0.5, 0.0)]
    for b, r, f, th in cases:
        def loss(r_, f_, th_, b_):
            return mx.sum(mp.flux_dev_from_tau(mx.array(tau), P, A_RS, b_, r_,
                                               limb_dark="hybrid5", u=W["hybrid5"],
                                               f=f_, theta=th_, integration=integration,
                                               **extra))
        v, g = mx.value_and_grad(loss, argnums=(0, 1, 2, 3))(
            *(mx.array(x, dtype=mx.float32) for x in (r, f, th, b)))
        mx.eval(v, g)
        assert np.isfinite(float(v)), (b, f, th)
        for gi in g:
            assert np.isfinite(float(gi)), (b, f, th)


def test_out_of_transit_and_far_side_exactly_zero():
    tau = np.concatenate([np.linspace(-1.7, -0.2, 64),
                          np.linspace(0.2, 1.7, 64)]).astype(np.float32)
    got = np.asarray(mp.flux_dev_from_tau(tau, P, A_RS, 0.1, 0.1, limb_dark="hybrid5",
                                          u=W["hybrid5"], f=0.3, theta=0.4))
    assert np.array_equal(got, np.zeros_like(got))


def test_kernels_cached_per_definition_orbit_and_basis():
    from metalplanet.hybrid import HybridLaw
    before = {k for k in M._kernels if k[0] == "obl"}
    custom = HybridLaw("my oblate law", (0.2,), LAWS["hybrid2"].shapes)
    mp.flux_dev_from_tau(TAU.astype(np.float32), P, A_RS, 0.3, 0.1,
                         limb_dark=custom, u=[0.3, 0.2], f=0.2, theta=0.4)
    new = {k for k in M._kernels if k[0] == "obl"} - before
    assert any(k[1] == custom.definition for k in new) or \
        ("obl", custom.definition, "circ", False) in M._kernels
    again = len(M._kernels)
    mp.flux_dev_from_tau(TAU.astype(np.float32), P, A_RS, 0.3, 0.1,
                         limb_dark=custom, u=[0.3, 0.2], f=0.2, theta=0.4)
    assert len(M._kernels) == again


def test_out_of_domain_arrays_give_nan_on_the_kernel_route():
    got = np.asarray(mp.flux_dev_from_tau(
        TAU.astype(np.float32), P, A_RS, 0.3, mx.array(0.6, dtype=mx.float32),
        limb_dark="hybrid4", u=W["hybrid4"], f=mx.array(0.5, dtype=mx.float32)))
    assert np.isnan(got[len(TAU) // 2])


@pytest.mark.parametrize("orbit", ["circ", "ecc"])
def test_contact_kernel_matches_the_fp64_graph(orbit):
    """One thread per chain, the graph's scheme: to fp32's level, including
    grazes (a third of the chains) and tilted planets."""
    from metalplanet.metal import _ecc_shape
    from metalplanet.oblate_tau import contact_offsets_oblate
    rng = np.random.default_rng(4)
    n = 200
    f = rng.choice([0.01, 0.1, 0.3, 0.5], n)
    r = np.minimum(rng.uniform(0.02, 0.3, n), 0.95 * (1 - f) ** 1.5)
    A = r / np.sqrt(1 - f)
    a = rng.uniform(5, 20, n)
    b = np.where(np.arange(n) % 3 == 0, rng.uniform(1 - A, 1 + A), rng.uniform(0, 1 + A))
    th = rng.uniform(0, np.pi, n)
    e, w = rng.uniform(0.05, 0.6, n), rng.uniform(-np.pi, np.pi, n)
    k, h = np.sqrt(e) * np.cos(w), np.sqrt(e) * np.sin(w)
    out = {}
    for dt, stream, fn in ((mx.float64, mx.cpu, contact_offsets_oblate),
                           (mx.float32, mx.gpu, MO.contact_offsets_oblate_kernel)):
        with mx.stream(stream):
            A_ = lambda v: mx.array(v, dtype=dt)
            args = [A_(a), A_(b), A_(r), A_(f), A_(th)]
            if orbit == "ecc":
                args += [A_(k), A_(h), _ecc_shape(A_(a), A_(b), A_(k), A_(h))[2]]
            c = fn(*args)
            c = mx.stack(c, 1) if isinstance(c, tuple) else c
            mx.eval(c)
            out[dt] = np.asarray(c.astype(mx.float32) if dt == mx.float32 else c,
                                 dtype=np.float64)
    assert np.abs(out[mx.float32] - out[mx.float64]).max() < 3e-6
