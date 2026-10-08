"""Fused fp32 kernels for the hybrid laws (metalplanet/metal_hybrid.py).

What is pinned:

* the fp64 graph path of flux_dev_from_tau against TransitModel -- an
  independent route from absolute times -- for every rule and orbit;
* the fp32 kernel against that fp64 graph, values and gradients (every
  input: tau, period, a, b, r, the weights, and secosw/sesinw);
* the ld_basis contract for the hybrid columns, both paths, and the basis
  VJP against the scalar VJP through it;
* the z-input kernels (flux_dev_metal);
* batching, shapes, argument checks; and that the quadratic kernels' own
  sources are untouched (their outputs are pinned by the rest of the
  suite and the release corpus).
"""

import contextlib
import math

import numpy as np
import mlx.core as mx
import pytest

import metalplanet
from metalplanet import ld
from metalplanet.hybrid import LAWS, hybrid_norms
from metalplanet.metal import flux_dev_from_tau, flux_dev_metal, metal_available
from test_tau_kernel import EXP, P, boundary_taus

needs_metal = pytest.mark.skipif(not metal_available(),
                                 reason="Metal kernels unavailable")

NAMES = ["hybrid2", "hybrid4", "hybrid5"]
MODES = [("none", {}), ("contact", dict(exp_time=EXP, n_gl=5)),
         ("supersample", dict(exp_time=EXP, n_sub=11))]
GEOMS = [
    ("nominal", dict(a=8.84, b=0.30, r=0.1153)),
    ("grazing", dict(a=10.0, b=0.85, r=0.20)),
    ("big_rp", dict(a=12.0, b=0.45, r=0.30)),
]
ORBITS = [("circ", None), ("ecc", (0.35, -0.25))]


def weights(law, seed=0):
    rng = np.random.default_rng(seed)
    if law == "hybrid2":
        return np.array(ld.hybrid2_from_q_np(rng.random(), rng.random()))
    return ld.simplex_from_q_np(rng.random(LAWS[law].n_w))


def _ctx(dtype):
    return (mx.stream(mx.cpu) if dtype == mx.float64
            else contextlib.nullcontext())


def taus(g):
    return boundary_taus(g["a"], g["b"], g["r"])


def run(tau, g, dtype, law, w=None, orbit=None, basis=False, **kw):
    with _ctx(dtype):
        args = [mx.array(v, dtype=dtype) for v in (P, g["a"], g["b"], g["r"])]
        ekw = ({} if orbit is None else
               dict(secosw=mx.array(orbit[0], dtype=dtype),
                    sesinw=mx.array(orbit[1], dtype=dtype)))
        u = None if basis else mx.array(w, dtype=dtype)
        out = flux_dev_from_tau(mx.array(tau, dtype=dtype), *args,
                                limb_dark=law, u=u, ld_basis=basis,
                                **ekw, **kw)
        mx.eval(out)
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# the fp64 graph path is the frontend's function
# ---------------------------------------------------------------------------

class TestGraphPathVsTransitModel:
    T0 = 0.35
    T = np.linspace(T0 - 0.25, T0 + 0.25, 801)

    def _api(self, law, w, integ, orbit, n_gl=5, ssf=1):
        g = GEOMS[0][1]
        p = metalplanet.TransitParams()
        p.t0, p.per, p.rp, p.a = self.T0, P, g["r"], g["a"]
        if orbit is None:
            e, wdeg, ci = 0.0, 90.0, g["b"] / g["a"]
        else:
            k, h = orbit
            e = k * k + h * h
            wdeg = math.degrees(math.atan2(h, k))
            esw = h * math.sqrt(e)
            ci = g["b"] * (1 + esw) / (g["a"] * (1 - e * e))
        p.inc, p.ecc, p.w = math.degrees(math.acos(ci)), e, wdeg
        p.limb_dark, p.u = law, list(w)
        m = metalplanet.TransitModel(
            p, self.T, exp_time=0.0 if integ == "none" else EXP,
            integration="supersample" if integ == "none" else integ,
            n_gl=n_gl, supersample_factor=ssf, dtype=mx.float64)
        return m.light_curve(p) - 1.0

    @pytest.mark.parametrize("law", NAMES)
    @pytest.mark.parametrize("integ,kw", MODES, ids=[m[0] for m in MODES])
    @pytest.mark.parametrize("orbit", [o[1] for o in ORBITS],
                             ids=[o[0] for o in ORBITS])
    def test_fp64_graph_matches_transit_model(self, law, integ, kw, orbit):
        w = weights(law, 1)
        ref = self._api(law, w, integ, orbit, n_gl=kw.get("n_gl", 5),
                        ssf=kw.get("n_sub", 1))
        got = run(self.T - self.T0, GEOMS[0][1], mx.float64, law, w,
                  orbit=orbit, integration=integ, **kw)
        assert np.abs(ref).max() > 1e-3
        assert np.abs(got - ref).max() < 1e-12


# ---------------------------------------------------------------------------
# kernel values and the ld_basis contract
# ---------------------------------------------------------------------------

@needs_metal
@pytest.mark.parametrize("law", NAMES)
@pytest.mark.parametrize("name,g", GEOMS, ids=[x[0] for x in GEOMS])
@pytest.mark.parametrize("integ,kw", MODES, ids=[m[0] for m in MODES])
@pytest.mark.parametrize("orbit", [o[1] for o in ORBITS],
                         ids=[o[0] for o in ORBITS])
def test_kernel_matches_fp64_graph(law, name, g, integ, kw, orbit):
    """Gate 5e-7 (hybrid5: 2e-6, its innermost pole's measured 0.33 ppm
    near the contacts), on a point set packed 3e-4 from the contacts."""
    w = weights(law, 2)
    tau = taus(g)
    k32 = run(tau, g, mx.float32, law, w, orbit=orbit, integration=integ, **kw)
    g64 = run(tau, g, mx.float64, law, w, orbit=orbit, integration=integ, **kw)
    gate = 2e-6 if law == "hybrid5" else 5e-7
    assert np.abs(k32 - g64).max() <= gate
    assert np.isfinite(k32).all()


@pytest.mark.parametrize("dtype", [mx.float64, mx.float32],
                         ids=["fp64", "fp32"])
@pytest.mark.parametrize("law", NAMES)
@pytest.mark.parametrize("integ,kw", MODES, ids=[m[0] for m in MODES])
@pytest.mark.parametrize("orbit", [o[1] for o in ORBITS],
                         ids=[o[0] for o in ORBITS])
def test_ld_basis_identity(dtype, law, integ, kw, orbit):
    """flux - 1 == (B @ c)/(N @ c), c = (1, -w): fp64 to 1e-14, fp32 at
    the scalar kernel's own noise."""
    if dtype == mx.float32 and not metal_available():
        pytest.skip("Metal unavailable")
    g = GEOMS[0][1]
    tau = taus(g)
    B = run(tau, g, dtype, law, orbit=orbit, basis=True, integration=integ,
            **kw)
    assert B.shape == tau.shape + (LAWS[law].n_col,)
    N = hybrid_norms(law)
    for seed in (3, 4, 5):
        w = weights(law, seed)
        c = np.concatenate([[1.0], -w])
        F = run(tau, g, dtype, law, w, orbit=orbit, integration=integ, **kw)
        tol = 1e-14 if dtype == mx.float64 else 5e-8
        assert np.abs(B @ c / (N @ c) - F).max() < tol, seed


@needs_metal
@pytest.mark.parametrize("law", NAMES)
def test_out_of_transit_and_far_side_are_exactly_zero(law):
    g = GEOMS[0][1]
    tau = np.concatenate([np.linspace(-0.5 * P, -0.3, 200),
                          np.linspace(0.3, 0.5 * P, 200)])
    for integ, kw in MODES:
        for orbit in (None, (0.35, -0.25)):
            assert (run(tau, g, mx.float32, law, weights(law), orbit=orbit,
                        integration=integ, **kw) == 0.0).all()
            assert (run(tau, g, mx.float32, law, orbit=orbit, basis=True,
                        integration=integ, **kw) == 0.0).all()


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------

def _inputs(tau, g, w, orbit, basis, dtype):
    vals = [tau, P, g["a"], g["b"], g["r"]]
    if not basis:
        vals.append(w)
    if orbit is not None:
        vals += list(orbit)
    return [mx.array(np.asarray(v, dtype=np.float64), dtype=dtype)
            for v in vals]


def _call(v, law, orbit, basis, **kw):
    u = None if basis else v[5]
    ekw = {} if orbit is None else dict(secosw=v[-2], sesinw=v[-1])
    return flux_dev_from_tau(*v[:5], limb_dark=law, u=u, ld_basis=basis,
                             **ekw, **kw)


def _grad(tau, g, w, ct, dtype, law, orbit, basis=False, **kw):
    with _ctx(dtype):
        c = mx.array(ct, dtype=dtype)
        args = _inputs(tau, g, w, orbit, basis, dtype)
        gr = mx.grad(lambda *v: mx.sum(c * _call(v, law, orbit, basis, **kw)),
                     argnums=tuple(range(len(args))))(*args)
        mx.eval(gr)
    return [np.asarray(x, dtype=np.float64) for x in gr]


def _cond(tau, g, w, ct, law, orbit, basis=False, **kw):
    """sum |ct * d out / d theta| per scalar input (each weight
    separately), fp64 forward mode: the scale an fp32 sum is accurate to."""
    with mx.stream(mx.cpu):
        base = _inputs(tau, g, w, orbit, basis, mx.float64)
        out = []
        for j in range(1, len(base)):
            comps = range(base[j].size) if base[j].ndim else [None]
            row = []
            for q in comps:
                t = [mx.zeros_like(x) for x in base]
                if q is None:
                    t[j] = mx.ones_like(base[j])
                else:
                    e = np.zeros(base[j].size)
                    e[q] = 1.0
                    t[j] = mx.array(e, dtype=mx.float64)
                _, (d,) = mx.jvp(lambda *v: _call(v, law, orbit, basis, **kw),
                                 base, t)
                row.append(float(np.sum(np.abs(ct * np.asarray(d)))))
            out.append(row)
    return out


@needs_metal
@pytest.mark.parametrize("basis", [False, True], ids=["scalar", "basis"])
@pytest.mark.parametrize("law", NAMES)
@pytest.mark.parametrize("integ,kw", MODES, ids=[m[0] for m in MODES])
@pytest.mark.parametrize("orbit", [o[1] for o in ORBITS],
                         ids=[o[0] for o in ORBITS])
def test_kernel_vjp_matches_fp64_graph(basis, law, integ, kw, orbit):
    """Every input, gated as the quadratic kernel's (test_ld_basis.py):
    tau per point 1e-3 of max|g|, each per-chain scalar 5e-4 of its
    condition scale."""
    g = GEOMS[2][1]                                  # r = 0.3, the hardest
    tau = taus(g)
    w = weights(law, 6)
    shape = tau.shape + ((LAWS[law].n_col,) if basis else ())
    ct = np.random.default_rng(7).normal(size=shape)
    kw = dict(kw, integration=integ)
    g64 = _grad(tau, g, w, ct, mx.float64, law, orbit, basis, **kw)
    g32 = _grad(tau, g, w, ct, mx.float32, law, orbit, basis, **kw)
    assert all(np.isfinite(x).all() for x in g32)
    assert np.abs(g32[0] - g64[0]).max() / np.abs(g64[0]).max() < 1e-3
    for j, scales in enumerate(_cond(tau, g, w, ct, law, orbit, basis, **kw),
                               start=1):
        a32, a64 = np.atleast_1d(g32[j]), np.atleast_1d(g64[j])
        for q, c in enumerate(scales):
            if c == 0.0:
                assert abs(a32[q]) < 1e-6
                continue
            assert abs(a32[q] - a64[q]) / c < 5e-4, (j, q)


@needs_metal
@pytest.mark.parametrize("law", NAMES)
@pytest.mark.parametrize("integ,kw", MODES, ids=[m[0] for m in MODES])
def test_basis_vjp_is_the_scalar_vjp_through_the_identity(law, integ, kw):
    """With ct_B = ct c / (N . c) the two VJPs are the same number, so in
    fp32 they differ by rounding only (in tau, period, a, b, r)."""
    g = GEOMS[0][1]
    tau = taus(g)
    kw = dict(kw, integration=integ)
    N = hybrid_norms(law)
    w = weights(law, 8)
    c = np.concatenate([[1.0], -w])
    ct = np.random.default_rng(5).normal(size=tau.shape)
    gs = _grad(tau, g, w, ct, mx.float32, law, None, **kw)
    gb = _grad(tau, g, w, ct[:, None] * c / (N @ c), mx.float32, law, None,
               basis=True, **kw)
    assert np.abs(gb[0] - gs[0]).max() / np.abs(gs[0]).max() < 1e-5
    scales = _cond(tau, g, w, ct, law, None, **kw)
    for j in range(1, 5):
        assert abs(float(gb[j]) - float(gs[j])) <= 1e-5 * scales[j - 1][0] + 1e-9


@needs_metal
@pytest.mark.parametrize("law", NAMES)
def test_no_nan_gradients_on_a_sweep(law):
    """Contacts, z = r, and every pole's Q = 0 line, in fp32."""
    rng = np.random.default_rng(11)
    n, m = 32, 400
    r = rng.uniform(0.02, 0.3, n)
    a = rng.uniform(6, 20, n)
    b = rng.uniform(0, 1.0, n) * (1 + r)
    tau = np.sort(rng.uniform(-0.2, 0.2, (n, m)), axis=1)
    W = np.stack([weights(law, s) for s in range(n)])
    ct = rng.normal(size=(n, m))
    for integ, kw in MODES:
        args = [mx.array(x, dtype=mx.float32)
                for x in (tau, np.full(n, P), a, b, r, W)]
        c = mx.array(ct, dtype=mx.float32)
        gr = mx.grad(lambda *v: mx.sum(c * flux_dev_from_tau(
            *v[:5], limb_dark=law, u=v[5], integration=integ, **kw)),
            argnums=tuple(range(6)))(*args)
        mx.eval(gr)
        for x in gr:
            assert np.isfinite(np.asarray(x)).all(), integ


# ---------------------------------------------------------------------------
# z-input kernels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [mx.float64, mx.float32],
                         ids=["fp64", "fp32"])
@pytest.mark.parametrize("law", NAMES)
def test_flux_dev_metal_hybrid(dtype, law):
    if dtype == mx.float32 and not metal_available():
        pytest.skip("Metal unavailable")
    from metalplanet.hybrid import flux_dev_hybrid
    z = np.concatenate([np.linspace(0.0, 1.45, 400),
                        [0.1153, 1 - 0.1153, 1 + 0.1153]])
    w = weights(law, 9)
    with _ctx(dtype):
        F = flux_dev_metal(mx.array(z, dtype=dtype), mx.array(0.1153, dtype=dtype),
                           limb_dark=law, u=w)
        B = flux_dev_metal(mx.array(z, dtype=dtype), mx.array(0.1153, dtype=dtype),
                           limb_dark=law, ld_basis=True)
        mx.eval(F, B)
    with mx.stream(mx.cpu):
        ref = np.asarray(flux_dev_hybrid(mx.array(z, dtype=mx.float64),
                                         mx.array(0.1153, dtype=mx.float64),
                                         w, law))
    tol = 1e-14 if dtype == mx.float64 else 5e-7
    assert np.abs(np.asarray(F, np.float64) - ref).max() < tol
    c = np.concatenate([[1.0], -w])
    N = hybrid_norms(law)
    assert np.abs(np.asarray(B, np.float64) @ c / (N @ c) - ref).max() < tol


@needs_metal
@pytest.mark.parametrize("law", NAMES)
def test_flux_dev_metal_hybrid_vjp(law):
    n = 3
    z = np.stack([np.linspace(0.0, 1.4, 300)] * n)
    r = np.array([0.05, 0.1153, 0.3])
    W = np.stack([weights(law, s) for s in range(n)])
    ct = np.random.default_rng(2).normal(size=z.shape)

    def grad(dtype):
        with _ctx(dtype):
            c = mx.array(ct, dtype=dtype)
            gr = mx.grad(lambda zz, rr, ww: mx.sum(c * flux_dev_metal(
                zz, rr, limb_dark=law, u=ww)), argnums=(0, 1, 2))(
                mx.array(z, dtype=dtype), mx.array(r, dtype=dtype),
                mx.array(W, dtype=dtype))
            mx.eval(gr)
        return [np.asarray(x, np.float64) for x in gr]

    g32, g64 = grad(mx.float32), grad(mx.float64)
    assert g32[0].shape == z.shape and g32[1].shape == (n,)
    assert g32[2].shape == (n, LAWS[law].n_w)
    for x, y in zip(g32, g64):
        assert np.abs(x - y).max() / np.abs(y).max() < 5e-4


# ---------------------------------------------------------------------------
# batching, shapes, arguments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [mx.float64, mx.float32],
                         ids=["fp64", "fp32"])
@pytest.mark.parametrize("law", NAMES)
@pytest.mark.parametrize("basis", [False, True], ids=["scalar", "basis"])
def test_each_chain_is_its_own(dtype, law, basis):
    if dtype == mx.float32 and not metal_available():
        pytest.skip("Metal unavailable")
    n, m = 5, 128
    rng = np.random.default_rng(3)
    p = dict(per=np.linspace(3.0, 4.2, n), a=np.linspace(8.0, 14.0, n),
             b=np.linspace(0.0, 0.8, n), r=np.linspace(0.05, 0.2, n),
             k=np.linspace(0.0, 0.4, n), h=np.linspace(-0.3, 0.2, n))
    W = np.stack([weights(law, s) for s in range(n)])
    tau = np.stack([np.linspace(-0.12, 0.12, m) + 1e-3 * j for j in range(n)])

    def go(sel):
        with _ctx(dtype):
            v = [mx.array(np.atleast_1d(p[q][sel]), dtype=dtype)
                 for q in ("per", "a", "b", "r", "k", "h")]
            u = None if basis else mx.array(np.atleast_2d(W[sel]), dtype=dtype)
            o = flux_dev_from_tau(
                mx.array(np.atleast_2d(tau[sel]), dtype=dtype), *v[:4],
                secosw=v[4], sesinw=v[5], limb_dark=law, u=u, ld_basis=basis,
                exp_time=EXP)
            mx.eval(o)
        return np.asarray(o, dtype=np.float64)

    batch = go(slice(None))
    assert batch.shape == (n, m) + ((LAWS[law].n_col,) if basis else ())
    tol = 1e-15 if dtype == mx.float64 else 0.0
    for j in range(n):
        assert np.abs(batch[j] - go(slice(j, j + 1))[0]).max() <= tol, j


@pytest.mark.parametrize("dtype", [mx.float64, mx.float32],
                         ids=["fp64", "fp32"])
def test_shapes_and_smaller_batch_after_compile(dtype):
    if dtype == mx.float32 and not metal_available():
        pytest.skip("Metal unavailable")
    law, n, m = "hybrid5", 64, 200
    W = np.stack([weights(law, s) for s in range(n)])
    tau = np.stack([np.linspace(-0.12, 0.12, m)] * n)
    ct = np.random.default_rng(1).normal(size=(n, m))
    with _ctx(dtype):
        one = flux_dev_from_tau(mx.array(tau[0], dtype=dtype), P, 10.0, 0.3,
                                0.1, limb_dark=law, u=W[0], exp_time=EXP)
        assert one.shape == (m,)
        shared = flux_dev_from_tau(mx.array(tau, dtype=dtype), P, 10.0, 0.3,
                                   0.1, limb_dark=law, u=W[0], exp_time=EXP)
        assert shared.shape == (n, m)

        @mx.compile
        def vg(t, ww, c):
            return mx.value_and_grad(lambda tt, w_: mx.sum(c * flux_dev_from_tau(
                tt, P, 10.0, 0.3, 0.1, limb_dark=law, u=w_, exp_time=EXP)),
                argnums=(0, 1))(t, ww)

        def go(k):
            v, gr = vg(mx.array(tau[:k], dtype=dtype),
                       mx.array(W[:k], dtype=dtype),
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


def test_argument_checks():
    t = mx.zeros((4,))
    with pytest.raises(ValueError, match="pass the hybrid5 weights as u="):
        flux_dev_from_tau(t, P, 8.8, 0.3, 0.1, 0.4, 0.2, limb_dark="hybrid5")
    with pytest.raises(ValueError, match="needs its weights as u="):
        flux_dev_from_tau(t, P, 8.8, 0.3, 0.1, limb_dark="hybrid4")
    with pytest.raises(ValueError, match="u= is for the hybrid laws"):
        flux_dev_from_tau(t, P, 8.8, 0.3, 0.1, 0.4, 0.2, u=[0.1, 0.2])
    with pytest.raises(ValueError, match="takes 2 weights"):
        flux_dev_from_tau(t, P, 8.8, 0.3, 0.1, limb_dark="hybrid2",
                          u=[0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="unknown hybrid law"):
        flux_dev_from_tau(t, P, 8.8, 0.3, 0.1, limb_dark="hybrid3", u=[0.1])
    with pytest.raises(ValueError, match="pass the hybrid2 weights as u="):
        flux_dev_metal(t, 0.1, 0.4, 0.2, limb_dark="hybrid2")


# ---------------------------------------------------------------------------
# the entry point's own fallback is hybrid.py's graph, bitwise (0.10.4)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("law", ["hybrid2", "hybrid4", "hybrid5"])
@pytest.mark.parametrize("dtype", [mx.float64, mx.float32],
                         ids=["fp64", "fp32-cpu-stream"])
def test_z_entry_fallback_is_flux_dev_hybrid_bitwise(law, dtype):
    """flux_dev_metal_hybrid off the kernel (fp64, or fp32 on the CPU
    stream) is flux_dev_hybrid's expression exactly -- not a dot-product
    contraction of the columns, which differed by 2.4e-7 in fp32."""
    from metalplanet.hybrid import LAWS, flux_dev_hybrid
    from metalplanet.metal_hybrid import flux_dev_metal_hybrid
    n_w = LAWS[law].n_w
    rng = np.random.default_rng(3)
    z = mx.array(np.sort(rng.uniform(0.0, 1.3, (3, 400)), axis=1), dtype=dtype)
    r = mx.array([0.05, 0.1, 0.3], dtype=dtype)
    w = mx.array(rng.dirichlet(np.ones(n_w + 1), 3)[:, :n_w], dtype=dtype)
    with mx.stream(mx.cpu):
        a = flux_dev_metal_hybrid(z, r, law, w)
        b = flux_dev_hybrid(z, r[:, None], w, law)
        mx.eval(a, b)
        assert np.array_equal(np.asarray(a), np.asarray(b))
