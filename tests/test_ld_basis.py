"""ld_basis=True: the limb-darkening basis instead of one curve.

A quadratic law is linear in its intensity coefficients, so

    flux_dev_from_tau(..., u1, u2) == (B @ c) / (N @ c),
    c = (1 - u1 - u2, u1 + 2 u2, -u2),   N = (pi, 2 pi / 3, pi / 2)

with B = flux_dev_from_tau(..., ld_basis=True) depending on geometry
alone. That identity is the contract, and these tests hold both paths to
it: the fp32 kernel at the scalar kernel's own noise, the fp64 graph path
at ~1e-14. Requested by SquishierPlanet, whose collapsed-LD target builds
every vertex law from one call instead of three.

ld_basis=False must be untouched. The scalar kernels' source is not
edited at all (the basis kernels are separate), and the rest of the suite
pins their outputs; here we only check the keyword's own plumbing.
"""

import contextlib
import math

import numpy as np
import mlx.core as mx
import pytest

from metalplanet.metal import flux_dev_from_tau, flux_dev_metal, metal_available
from test_tau_kernel import EXP, P, boundary_taus

needs_metal = pytest.mark.skipif(not metal_available(),
                                 reason="Metal kernels unavailable")

N_VEC = np.array([math.pi, 2.0 * math.pi / 3.0, math.pi / 2.0])
VERTICES = [(0.0, 0.0), (2.0, -1.0), (0.0, 1.0)]   # corners of the q-box

GEOMS = [
    ("nominal", dict(a=8.84, b=0.30, r=0.1153)),
    ("central", dict(a=8.84, b=0.0, r=0.1153)),
    ("grazing", dict(a=10.0, b=0.85, r=0.20)),      # b > 1 - r
    ("big_rp", dict(a=12.0, b=0.45, r=0.30)),
    ("deep_graze", dict(a=12.0, b=1.10, r=0.30)),   # never fully inside
]
MODES = [("none", {}), ("contact", dict(exp_time=EXP, n_gl=5)),
         ("supersample", dict(exp_time=EXP, n_sub=11))]


def cvec(u1, u2):
    return np.array([1.0 - u1 - u2, u1 + 2.0 * u2, -u2])


def laws(k=12, seed=0):
    """The three vertices plus random laws drawn uniformly in Kipping
    (2013) (q1, q2) -- the box the collapsed target integrates over."""
    q = np.random.default_rng(seed).random((k, 2))
    s = np.sqrt(q[:, 0])
    return VERTICES + list(zip(2.0 * s * q[:, 1], s * (1.0 - 2.0 * q[:, 1])))


def _ctx(dtype):
    return (mx.stream(mx.cpu) if dtype == mx.float64
            else contextlib.nullcontext())


def run(tau, g, dtype, u=None, **kw):
    """flux_dev_from_tau at geometry g: the basis if u is None, else the
    scalar curve for law u. fp64 goes on the CPU stream (graph path)."""
    with _ctx(dtype):
        args = [mx.array(v, dtype=dtype) for v in (P, g["a"], g["b"], g["r"])]
        if u is None:
            out = flux_dev_from_tau(mx.array(tau, dtype=dtype), *args,
                                    ld_basis=True, **kw)
        else:
            out = flux_dev_from_tau(mx.array(tau, dtype=dtype), *args,
                                    *[mx.array(x, dtype=dtype) for x in u],
                                    **kw)
        mx.eval(out)
    return np.asarray(out, dtype=np.float64)


def taus(g):
    return boundary_taus(g["a"], g["b"], g["r"])


# ---------------------------------------------------------------------------
# Acceptance 1: the defining identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,g", GEOMS, ids=[x[0] for x in GEOMS])
@pytest.mark.parametrize("integ,kw", MODES, ids=[m[0] for m in MODES])
def test_identity_fp64_graph(name, g, integ, kw):
    tau = taus(g)
    kw = dict(kw, integration=integ)
    B = run(tau, g, mx.float64, **kw)
    assert B.shape == tau.shape + (3,)
    assert np.abs(B).max() > 1e-3                    # actually in transit
    for u in laws():
        c = cvec(*u)
        F = run(tau, g, mx.float64, u=u, **kw)
        assert np.abs(B @ c / (N_VEC @ c) - F).max() < 1e-14, u


@needs_metal
@pytest.mark.parametrize("name,g", GEOMS, ids=[x[0] for x in GEOMS])
@pytest.mark.parametrize("integ,kw", MODES, ids=[m[0] for m in MODES])
def test_identity_fp32_kernel(name, g, integ, kw):
    """Against the scalar KERNEL, at the scalar kernel's own fp32 noise:
    the gate is the scalar kernel's distance from fp64, so the basis may
    not add more error than the path it replaces already has."""
    tau = taus(g)
    kw = dict(kw, integration=integ)
    B = run(tau, g, mx.float32, **kw)
    assert B.shape == tau.shape + (3,)
    for u in laws():
        c = cvec(*u)
        F32 = run(tau, g, mx.float32, u=u, **kw)
        F64 = run(tau, g, mx.float64, u=u, **kw)
        noise = max(np.abs(F32 - F64).max(), 1e-8)
        got = B @ c / (N_VEC @ c)
        assert np.abs(got - F32).max() <= 2.0 * noise, u
        assert np.abs(got - F64).max() <= 2.0 * noise, u


@needs_metal
@pytest.mark.parametrize("integ,kw", MODES, ids=[m[0] for m in MODES])
def test_kernel_basis_matches_graph_basis(integ, kw):
    for _, g in GEOMS:
        tau = taus(g)
        k32 = run(tau, g, mx.float32, integration=integ, **kw)
        g64 = run(tau, g, mx.float64, integration=integ, **kw)
        assert np.abs(k32 - g64).max() <= 5e-7


@needs_metal
def test_out_of_transit_is_exactly_zero():
    g = GEOMS[0][1]
    tau = np.concatenate([np.linspace(-1.5, -0.3, 50),
                          np.linspace(0.3, 1.5, 50)])
    for integ, kw in MODES:
        B = run(tau, g, mx.float32, integration=integ, **kw)
        assert (B == 0.0).all(), integ


# ---------------------------------------------------------------------------
# Acceptance 2: gradients
#
# No finite differences across contacts (see test_tau_kernel). Two
# FD-free checks instead: the fp64 graph basis VJP against the scalar
# graph VJP through the identity -- with ct_B = ct c / (N . c) the two
# are the same number -- and the fp32 kernel VJP against fp64 graph
# autodiff for a random (n, m, 3) cotangent.
# ---------------------------------------------------------------------------

def _grad(tau, g, ct, dtype, u=None, **kw):
    """d(sum ct * out)/d(tau, per, a, b, r) for the basis (u None) or the
    scalar curve for law u."""
    base = [tau, P, g["a"], g["b"], g["r"]]
    with _ctx(dtype):
        ctv = mx.array(ct, dtype=dtype)
        extra = ([] if u is None
                 else [mx.array(x, dtype=dtype) for x in u])

        def f(*v):
            out = flux_dev_from_tau(*v, *extra, ld_basis=u is None, **kw)
            return mx.sum(ctv * out)

        gr = mx.grad(f, argnums=tuple(range(5)))(
            *[mx.array(v, dtype=dtype) for v in base])
        mx.eval(gr)
    return [np.asarray(x, dtype=np.float64) for x in gr]


@pytest.mark.parametrize("name,g", GEOMS, ids=[x[0] for x in GEOMS])
@pytest.mark.parametrize("integ,kw", MODES, ids=[m[0] for m in MODES])
def test_graph_basis_vjp_equals_scalar_vjp_through_identity(name, g, integ,
                                                            kw):
    tau = taus(g)
    ct = np.random.default_rng(5).normal(size=tau.shape)
    kw = dict(kw, integration=integ)
    for u in (VERTICES[1], laws(1, seed=9)[-1]):
        c = cvec(*u)
        gs = _grad(tau, g, ct, mx.float64, u=u, **kw)
        gb = _grad(tau, g, ct[:, None] * c / (N_VEC @ c), mx.float64, **kw)
        for k, (x, y) in enumerate(zip(gb, gs)):
            s = np.abs(y).max() + 1e-300
            assert np.abs(x - y).max() / s < 1e-12, (u, k)


def _cond(tau, g, ct, **kw):
    """sum |ct * dB/dtheta| per parameter, from fp64 forward mode: the
    scale an fp32 sum of those terms is accurate relative to. A random
    (n, m, 3) cotangent cancels harder than a one-component one, so
    normalising by the result's own size would flag cancellation, not
    error."""
    with mx.stream(mx.cpu):
        base = [mx.array(v, dtype=mx.float64)
                for v in (tau, P, g["a"], g["b"], g["r"])]
        out = []
        for k in range(1, 5):
            t = [mx.zeros_like(x) for x in base]
            t[k] = mx.ones_like(base[k])
            _, (d,) = mx.jvp(lambda *v: flux_dev_from_tau(
                *v, ld_basis=True, **kw), base, t)
            out.append(float(np.sum(np.abs(ct * np.asarray(d)))))
    return out


@needs_metal
@pytest.mark.parametrize("name,g", GEOMS, ids=[x[0] for x in GEOMS])
@pytest.mark.parametrize("integ,kw", MODES, ids=[m[0] for m in MODES])
def test_kernel_vjp_matches_fp64_graph_autodiff(name, g, integ, kw):
    tau = taus(g)
    ct = np.random.default_rng(7).normal(size=tau.shape + (3,))
    kw = dict(kw, integration=integ)
    g64 = _grad(tau, g, ct, mx.float64, **kw)
    g32 = _grad(tau, g, ct, mx.float32, **kw)
    assert all(np.isfinite(x).all() for x in g32)
    # tau: per point. Twice the scalar kernel's 5e-4 gate -- the worst
    # points sit 3e-4 from a contact, where fp32 dF/dz is cancellation-
    # prone, and three independent cotangent components carry ~2x the
    # noise of one (measured: 4.8e-4 here vs 2.5e-4 scalar, big_rp/none).
    s = np.abs(g64[0]).max()
    assert np.abs(g32[0] - g64[0]).max() / s < 1e-3
    # per-chain parameters: relative to the sum's condition scale
    # (measured worst 1.0e-4, instantaneous rule on big_rp)
    for k, c in zip(range(1, 5), _cond(tau, g, ct, **kw)):
        if c == 0.0:                     # e.g. d/db at b = 0
            assert abs(float(g32[k])) < 1e-6
            continue
        assert abs(float(g32[k]) - float(g64[k])) / c < 5e-4, k


@needs_metal
@pytest.mark.parametrize("integ,kw", MODES, ids=[m[0] for m in MODES])
def test_kernel_vjp_is_the_scalar_kernel_vjp(integ, kw):
    """Through the identity, with ct_B = ct c / (N . c), the basis VJP and
    the scalar kernel's VJP are the same number -- so in fp32 they should
    differ by rounding, far below either one's distance from fp64."""
    kw = dict(kw, integration=integ)
    for _, g in GEOMS:
        tau = taus(g)
        ct = np.random.default_rng(5).normal(size=tau.shape)
        for u in laws(2, seed=3):
            c = cvec(*u)
            cb = ct[:, None] * c / (N_VEC @ c)
            gs = _grad(tau, g, ct, mx.float32, u=u, **kw)
            gb = _grad(tau, g, cb, mx.float32, **kw)
            assert (np.abs(gb[0] - gs[0]).max()
                    / np.abs(gs[0]).max()) < 1e-5, u
            for k, sc in zip(range(1, 5), _cond(tau, g, cb, **kw)):
                assert abs(float(gb[k]) - float(gs[k])) <= 1e-5 * sc + 1e-9


@needs_metal
def test_no_nan_gradients_on_boundary_sweep():
    rng = np.random.default_rng(3)
    n = 20000
    g = dict(a=9.0, b=0.4, r=float(rng.uniform(0.02, 0.4)))
    tau = np.concatenate([taus(g)] * 40)
    tau = tau + np.where(rng.random(tau.size) < 0.5, 0.0,
                         10.0 ** rng.uniform(-7, -2, tau.size))
    ct = rng.normal(size=tau.shape + (3,))
    for integ, kw in MODES:
        for x in _grad(tau[:n], g, ct[:n], mx.float32,
                       integration=integ, **kw):
            assert np.isfinite(x).all(), integ


# ---------------------------------------------------------------------------
# Acceptance 3: the default is untouched
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [mx.float32, mx.float64],
                         ids=["fp32", "fp64"])
def test_ld_basis_false_is_the_scalar_call(dtype):
    g = GEOMS[0][1]
    tau = taus(g)
    u = (0.4225, 0.3077)
    with _ctx(dtype):
        args = [mx.array(v, dtype=dtype)
                for v in (P, g["a"], g["b"], g["r"], *u)]
        t = mx.array(tau, dtype=dtype)
        a = flux_dev_from_tau(t, *args, exp_time=EXP)
        b = flux_dev_from_tau(t, *args, exp_time=EXP, ld_basis=False)
        mx.eval(a, b)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_u_required_without_basis():
    with pytest.raises(ValueError, match="u1 and u2"):
        flux_dev_from_tau(mx.zeros((4,)), P, 8.8, 0.3, 0.1)
    with pytest.raises(ValueError, match="u1 and u2"):
        flux_dev_metal(mx.zeros((4,)), 0.1)


# ---------------------------------------------------------------------------
# Acceptance 4: batching
# ---------------------------------------------------------------------------

def _chains(n):
    return dict(per=np.linspace(3.0, 4.2, n), a=np.linspace(7.5, 11.0, n),
                b=np.linspace(0.0, 0.85, n), r=np.linspace(0.05, 0.22, n))


@pytest.mark.parametrize("dtype", [mx.float32, mx.float64],
                         ids=["fp32", "fp64"])
@pytest.mark.parametrize("integ,kw", MODES, ids=[m[0] for m in MODES])
def test_each_chain_is_its_own(dtype, integ, kw):
    """Row j of a batch is chain j alone, on deliberately distinct chains."""
    if dtype == mx.float32 and not metal_available():
        pytest.skip("Metal kernels unavailable")
    n, m = 6, 128
    p = _chains(n)
    tau = np.stack([np.linspace(-0.12, 0.12, m) + 1e-3 * j
                    for j in range(n)])
    kw = dict(kw, integration=integ)

    def go(sel):
        with _ctx(dtype):
            o = flux_dev_from_tau(
                mx.array(np.atleast_2d(tau[sel]), dtype=dtype),
                *[mx.array(np.atleast_1d(p[k][sel]), dtype=dtype)
                  for k in ("per", "a", "b", "r")], ld_basis=True, **kw)
            mx.eval(o)
        return np.asarray(o, dtype=np.float64)

    batch = go(slice(None))
    assert batch.shape == (n, m, 3)
    tol = 1e-15 if dtype == mx.float64 else 0.0
    for j in range(n):
        assert np.abs(batch[j] - go(slice(j, j + 1))[0]).max() <= tol, j
    assert min(np.abs(batch[i] - batch[j]).max()
               for i in range(n) for j in range(i + 1, n)) > 1e-4


@pytest.mark.parametrize("dtype", [mx.float32, mx.float64],
                         ids=["fp32", "fp64"])
def test_shapes(dtype):
    if dtype == mx.float32 and not metal_available():
        pytest.skip("Metal kernels unavailable")
    m, n = 97, 5
    tau1 = np.linspace(-0.1, 0.1, m)
    with _ctx(dtype):
        def f(t, *p, **kw):
            o = flux_dev_from_tau(mx.array(t, dtype=dtype),
                                  *[mx.array(v, dtype=dtype) if
                                    isinstance(v, np.ndarray) else v
                                    for v in p], ld_basis=True,
                                  exp_time=EXP, **kw)
            mx.eval(o)
            return o
        assert f(tau1, P, 8.8, 0.3, 0.1).shape == (m, 3)
        # per-chain parameters broadcast a 1-D tau
        assert f(tau1, P, 8.8, np.full(n, 0.3), 0.1).shape == (n, m, 3)
        tau2 = np.broadcast_to(tau1, (n, m)).copy()
        assert f(tau2, P, 8.8, 0.3, 0.1).shape == (n, m, 3)
        assert f(tau2[:1], P, 8.8, 0.3, 0.1).shape == (1, m, 3)
        # u1/u2 are ignored outright -- their chain count included
        assert f(tau1, P, 8.8, 0.3, 0.1, np.full(n, 0.4),
                 np.full(n, 0.2)).shape == (m, 3)


@pytest.mark.parametrize("dtype", [mx.float32, mx.float64],
                         ids=["fp32", "fp64"])
def test_smaller_batch_after_compile(dtype):
    """turin's precision probe calls with the first 32 of 512 chains,
    through the same compiled function."""
    if dtype == mx.float32 and not metal_available():
        pytest.skip("Metal kernels unavailable")
    n, m = 64, 200
    p = _chains(n)
    tau = np.stack([np.linspace(-0.12, 0.12, m)] * n)
    ct = np.random.default_rng(1).normal(size=(n, m, 3))
    with _ctx(dtype):
        @mx.compile
        def vg(t, per, a, b, r, c):
            def f(*v):
                return mx.sum(c * flux_dev_from_tau(
                    *v, exp_time=EXP, ld_basis=True))
            return mx.value_and_grad(f, argnums=(0, 1, 2, 3, 4))(
                t, per, a, b, r)

        def go(k):
            args = [mx.array(tau[:k], dtype=dtype)] + [
                mx.array(p[j][:k], dtype=dtype)
                for j in ("per", "a", "b", "r")]
            v, gr = vg(*args, mx.array(ct[:k], dtype=dtype))
            mx.eval(v, gr)
            return [np.asarray(x, np.float64) for x in gr]

        big = go(n)
        for k in (32, 1):
            small = go(k)
            assert small[0].shape == (k, m)
            tol = 1e-12 if dtype == mx.float64 else 1e-5
            for x, y in zip(small, big):
                s = np.abs(y[:k]).max() + 1e-12
                assert np.abs(x - y[:k]).max() / s < tol


# ---------------------------------------------------------------------------
# flux_dev_metal(..., ld_basis=True): the z-input kernel, for symmetry
# ---------------------------------------------------------------------------

Z = np.concatenate([np.linspace(0.0, 1.45, 400),
                    [0.1153, 1 - 0.1153, 1 + 0.1153, 0.0]])


@pytest.mark.parametrize("dtype", [mx.float32, mx.float64],
                         ids=["fp32", "fp64"])
def test_z_basis_identity(dtype):
    if dtype == mx.float32 and not metal_available():
        pytest.skip("Metal kernels unavailable")
    for r in (0.05, 0.1153, 0.3):
        with _ctx(dtype):
            z = mx.array(Z, dtype=dtype)
            B = flux_dev_metal(z, mx.array(r, dtype=dtype), ld_basis=True)
            Fs = [flux_dev_metal(z, *[mx.array(x, dtype=dtype)
                                      for x in (r, *u)]) for u in laws(4)]
            mx.eval(B, *Fs)
        B = np.asarray(B, np.float64)
        assert B.shape == Z.shape + (3,)
        tol = 1e-14 if dtype == mx.float64 else 5e-8
        for u, F in zip(laws(4), Fs):
            c = cvec(*u)
            assert np.abs(B @ c / (N_VEC @ c)
                          - np.asarray(F, np.float64)).max() < tol, (r, u)


@needs_metal
def test_z_basis_vjp_matches_fp64():
    n = 3
    z = np.stack([Z] * n)
    r = np.array([0.05, 0.1153, 0.3])
    ct = np.random.default_rng(2).normal(size=z.shape + (3,))

    def grad(dtype):
        with _ctx(dtype):
            c = mx.array(ct, dtype=dtype)
            gr = mx.grad(lambda zz, rr: mx.sum(
                c * flux_dev_metal(zz, rr, ld_basis=True)), argnums=(0, 1))(
                mx.array(z, dtype=dtype), mx.array(r, dtype=dtype))
            mx.eval(gr)
        return [np.asarray(x, np.float64) for x in gr]

    g32, g64 = grad(mx.float32), grad(mx.float64)
    assert g32[0].shape == z.shape and g32[1].shape == (n,)
    for x, y in zip(g32, g64):
        assert np.abs(x - y).max() / np.abs(y).max() < 5e-4
