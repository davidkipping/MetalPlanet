"""flux_dev_from_tau: the tau-input kernel with in-kernel exposure
integration.

The three things this entry point promises that the z-input kernel does
not -- a gradient in tau, an exposure rule that never materialises the
sub-exposure axis, and a graph path that computes the *same function* so
fp64 certification is meaningful -- are each pinned here.

The kernel tests skip on machines without a usable Metal device; the
graph-path tests do not, because that path is the fp64 contract.
"""

import math

import numpy as np
import mlx.core as mx
import pytest

from metalplanet import api
from metalplanet.metal import flux_dev_from_tau, metal_available

P = 3.4525
EXP = 29.4 / 60.0 / 24.0          # Kepler long cadence, in days
NOM = dict(a=8.84, b=0.30, r=0.1153, u1=0.4225, u2=0.3077)
needs_metal = pytest.mark.skipif(not metal_available(),
                                 reason="Metal kernels unavailable")


def call(tau, per=P, dtype=mx.float32, **kw):
    """flux_dev_from_tau at the nominal parameters, everything one dtype."""
    p = dict(NOM)
    p.update({k: kw.pop(k) for k in list(kw) if k in NOM})
    args = [mx.array(v, dtype=dtype)
            for v in (per, p["a"], p["b"], p["r"], p["u1"], p["u2"])]
    out = flux_dev_from_tau(mx.array(tau, dtype=dtype), *args, **kw)
    mx.eval(out)
    return np.asarray(out, dtype=np.float64)


def api_dev(t, t0, integration, n_gl=5, ssf=1, **kw):
    p = dict(NOM)
    p.update(kw)
    pars = api.TransitParams()
    pars.t0, pars.per, pars.rp, pars.a = t0, P, p["r"], p["a"]
    pars.inc = math.degrees(math.acos(p["b"] / p["a"]))
    pars.ecc, pars.w = 0.0, 90.0
    pars.u, pars.limb_dark = [p["u1"], p["u2"]], "quadratic"
    m = api.TransitModel(pars, t,
                         exp_time=0.0 if integration == "none" else EXP,
                         integration=("supersample" if integration == "none"
                                      else integration),
                         n_gl=n_gl, supersample_factor=ssf, dtype=mx.float64)
    return np.asarray(m.light_curve(pars), dtype=np.float64) - 1.0


def tau_at_z(zt, a, b, per=P):
    """tau on the front side with z(tau) = zt, or None if unreachable."""
    s2 = (zt * zt - b * b) / (a * a - b * b)
    if not 0.0 <= s2 <= 1.0:
        return None
    return math.asin(math.sqrt(s2)) * per / (2.0 * math.pi)


def boundary_taus(a, b, r, per=P):
    """z ~ r, 1-r, 1+r, z = b (mid-transit) and a scatter around them."""
    out = [0.0]
    for zt in (r, 1.0 - r, 1.0 + r, 1.0, 0.5 * (1.0 - r)):
        for eps in (-3e-4, 0.0, 3e-4):
            v = tau_at_z(zt + eps, a, b, per)
            if v is not None:
                out += [v, -v]
    out += list(np.linspace(-0.12, 0.12, 21))
    return np.array(sorted(set(np.round(out, 12))))


# ---------------------------------------------------------------------------
# Acceptance 1: parity with the frontend at matched settings
# ---------------------------------------------------------------------------

class TestParityWithTransitModel:
    T0 = 0.35
    T = np.linspace(T0 - 0.25, T0 + 0.25, 1001)

    @pytest.mark.parametrize("n_gl", [5, 7])
    def test_contact_fp64_graph(self, n_gl):
        ref = api_dev(self.T, self.T0, "contact", n_gl=n_gl)
        got = call(self.T - self.T0, dtype=mx.float64, exp_time=EXP,
                   integration="contact", n_gl=n_gl)
        assert np.abs(got - ref).max() < 1e-12

    @pytest.mark.parametrize("ssf", [11, 15])
    def test_supersample_fp64_graph(self, ssf):
        ref = api_dev(self.T, self.T0, "supersample", ssf=ssf)
        got = call(self.T - self.T0, dtype=mx.float64, exp_time=EXP,
                   integration="supersample", n_sub=ssf)
        assert np.abs(got - ref).max() < 1e-12

    def test_instantaneous_fp64_graph(self):
        ref = api_dev(self.T, self.T0, "none")
        got = call(self.T - self.T0, dtype=mx.float64)
        assert np.abs(got - ref).max() < 1e-12

    @needs_metal
    @pytest.mark.parametrize("integration,kw", [
        ("none", {}),
        ("contact", dict(exp_time=EXP, n_gl=5)),
        ("contact", dict(exp_time=EXP, n_gl=7)),
        ("supersample", dict(exp_time=EXP, n_sub=15)),
    ])
    def test_fp32_kernel(self, integration, kw):
        ref = api_dev(self.T, self.T0, integration,
                      n_gl=kw.get("n_gl", 5), ssf=kw.get("n_sub", 1))
        got = call(self.T - self.T0, integration=integration, **kw)
        assert np.abs(got - ref).max() <= 5e-7

    @needs_metal
    def test_kernel_matches_its_own_graph_path(self):
        """The fallback is a supported path, so it must compute the same
        function -- turin certifies in fp64 and runs in fp32."""
        tau = boundary_taus(NOM["a"], NOM["b"], NOM["r"])
        for integ, kw in (("none", {}), ("contact", dict(exp_time=EXP)),
                          ("supersample", dict(exp_time=EXP, n_sub=15))):
            k = call(tau, integration=integ, **kw)
            g = call(tau, dtype=mx.float64, integration=integ, **kw)
            assert np.abs(k - g).max() <= 5e-7, integ


# ---------------------------------------------------------------------------
# Acceptance 2: gradients against fp64 central finite differences
# ---------------------------------------------------------------------------

CASES = [
    ("nominal", dict(NOM)),
    ("central", dict(NOM, b=0.0)),
    ("grazing", dict(NOM, b=1.0 - NOM["r"])),
    ("big_rp", dict(a=12.0, b=0.45, r=0.30, u1=0.35, u2=0.22)),
]
MODES = [("none", {}), ("contact", dict(exp_time=EXP, n_gl=5)),
         ("supersample", dict(exp_time=EXP, n_sub=11))]


def _grad(tau, p, ct, dtype, **kw):
    """d(sum ct * F)/d(tau, per, a, b, r, u1, u2).

    flux_dev_from_tau puts itself on the CPU stream for fp64, but the
    surrounding graph (the cotangent product, the sum) is the caller's --
    so an fp64 caller still has to pick the stream, exactly as
    TransitModel does for its own fp64 graph.
    """
    base = [tau, P, p["a"], p["b"], p["r"], p["u1"], p["u2"]]
    import contextlib
    ctx = (mx.stream(mx.cpu) if dtype == mx.float64
           else contextlib.nullcontext())
    with ctx:
        ctv = mx.array(ct, dtype=dtype)

        def f(*v):
            return mx.sum(ctv * flux_dev_from_tau(*v, **kw))

        g = mx.grad(f, argnums=tuple(range(7)))(
            *[mx.array(v, dtype=dtype) for v in base])
        mx.eval(g)
    return [np.asarray(x, dtype=np.float64) for x in g]


def _fd(tau, p, ct, **kw):
    base = [tau, P, p["a"], p["b"], p["r"], p["u1"], p["u2"]]

    def val(v):
        with mx.stream(mx.cpu):
            o = flux_dev_from_tau(*[mx.array(x, dtype=mx.float64) for x in v],
                                  **kw)
            mx.eval(o)
        return np.asarray(o, dtype=np.float64)

    h0 = 1e-7
    vp, vm = list(base), list(base)
    vp[0], vm[0] = tau + h0, tau - h0
    out = [ct * (val(vp) - val(vm)) / (2 * h0)]
    for k in range(1, 7):
        h = max(abs(base[k]), 1.0) * 1e-6
        vp, vm = list(base), list(base)
        vp[k], vm[k] = base[k] + h, base[k] - h
        out.append(float(np.sum(ct * (val(vp) - val(vm)) / (2 * h))))
    return out


@pytest.mark.parametrize("name,p", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("integ,kw", MODES, ids=[m[0] for m in MODES])
def test_gradients_vs_finite_differences(name, p, integ, kw):
    """Central FD in fp64, with the four contact times excluded from the
    tau comparison: the light curve's tau-derivative genuinely jumps
    there, so a central difference straddling a contact reports the
    average of two different one-sided slopes no matter how small h is.
    Everywhere else the agreement is the O(h^2) of the difference itself.
    """
    tau = boundary_taus(p["a"], p["b"], p["r"])
    ct = np.random.default_rng(7).normal(size=tau.shape)
    kw = dict(kw, integration=integ)
    g = _grad(tau, p, ct, mx.float64, **kw)
    fd = _fd(tau, p, ct, **kw)

    cs = [tau_at_z(z, p["a"], p["b"]) for z in (1 - p["r"], 1 + p["r"])]
    cs = np.array([s * c for c in cs if c is not None for s in (-1.0, 1.0)])
    # Where the curve kinks depends on the rule. Instantaneous flux kinks
    # at a contact. The contact-split average kinks where a contact enters
    # or leaves the window, half an exposure either side. A supersampled
    # average kinks once per *node*, wherever any fixed node crosses a
    # contact -- n_sub kinks per contact, which is exactly the reason the
    # contact rule converges and this one does not.
    if integ == "contact":
        edge = np.concatenate([cs - 0.5 * EXP, cs + 0.5 * EXP])
    elif integ == "supersample":
        nsub = kw.get("n_sub", 1)
        off = (np.linspace(-0.5 * EXP, 0.5 * EXP, nsub) if nsub > 1
               else np.zeros(1))
        edge = (cs[:, None] - off[None, :]).ravel()
    else:
        edge = cs
    far = np.min(np.abs(tau[:, None] - edge[None, :]), axis=1) > 1e-5

    s = np.abs(g[0]).max()
    assert np.abs(g[0] - fd[0])[far].max() / s < 1e-5
    for k in range(1, 7):
        scale = max(abs(fd[k]), 1e-3 * abs(g[k]).max() + 1e-12)
        assert abs(float(g[k]) - fd[k]) / scale < 2e-3, k
    assert all(np.isfinite(x).all() for x in g)


@needs_metal
@pytest.mark.parametrize("name,p", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("integ,kw", MODES, ids=[m[0] for m in MODES])
def test_kernel_gradients_match_graph_gradients(name, p, integ, kw):
    tau = boundary_taus(p["a"], p["b"], p["r"])
    ct = np.random.default_rng(7).normal(size=tau.shape)
    kw = dict(kw, integration=integ)
    g64 = _grad(tau, p, ct, mx.float64, **kw)
    g32 = _grad(tau, p, ct, mx.float32, **kw)
    # 5e-4 relative: the same gate the model kernel's gradient parity
    # uses, on a point set deliberately packed 3e-4 from the contacts,
    # where dF/dz is at its most cancellation-prone in fp32.
    for k, (x, y) in enumerate(zip(g32, g64)):
        s = np.abs(y).max() + 1e-12
        assert np.abs(x - y).max() / s < 5e-4, k
        assert np.isfinite(x).all(), k


@needs_metal
def test_no_nan_gradients_on_boundary_sweep():
    """Both-branch sanitization: every point sits on a boundary, and no
    gradient of any output w.r.t. any input may be NaN or inf."""
    rng = np.random.default_rng(3)
    n = 20000
    r = float(rng.uniform(0.02, 0.4))
    a, b = 9.0, 0.4
    kind = rng.integers(0, 5, n)
    base = np.select([kind == 0, kind == 1, kind == 2, kind == 3],
                     [tau_at_z(1 - r, a, b), tau_at_z(1 + r, a, b),
                      tau_at_z(r, a, b) or 0.0, 0.0],
                     default=rng.uniform(-0.15, 0.15, n))
    off = np.where(rng.random(n) < 0.5, 0.0,
                   10.0 ** rng.uniform(-7, -2, n) * rng.choice([-1, 1], n))
    tau = (base + off) * rng.choice([-1.0, 1.0], n)
    p = dict(NOM, a=a, b=b, r=r)
    for integ, kw in MODES:
        g = _grad(tau, p, np.ones(n), mx.float32,
                  **dict(kw, integration=integ))
        for k, x in enumerate(g):
            assert np.isfinite(x).all(), (integ, k)


# ---------------------------------------------------------------------------
# Contract: shapes, dispatch, argument validation
# ---------------------------------------------------------------------------

@needs_metal
class TestContract:
    def test_shapes_and_broadcasting(self):
        m, n = 97, 5
        tau1 = np.linspace(-0.1, 0.1, m)
        assert call(tau1).shape == (m,)
        tau2 = np.broadcast_to(tau1, (n, m)).copy()
        got = flux_dev_from_tau(
            mx.array(tau2, dtype=mx.float32),
            *[mx.array(np.full(n, v), dtype=mx.float32)
              for v in (P, NOM["a"], NOM["b"], NOM["r"], NOM["u1"],
                        NOM["u2"])])
        mx.eval(got)
        assert got.shape == (n, m)
        # identical chains must give identical rows
        g = np.asarray(got)
        assert np.abs(g - g[0]).max() == 0.0
        # (m,) tau with (n,) parameters broadcasts to (n, m)
        got2 = flux_dev_from_tau(
            mx.array(tau1, dtype=mx.float32), P, NOM["a"],
            mx.array(np.linspace(0.0, 0.8, n), dtype=mx.float32),
            NOM["r"], NOM["u1"], NOM["u2"])
        mx.eval(got2)
        assert got2.shape == (n, m)

    @pytest.mark.parametrize("m", [1, 31, 33, 257, 1000])
    def test_m_not_multiple_of_threadgroup(self, m):
        tau = np.linspace(-0.1, 0.1, m)
        k = call(tau, exp_time=EXP, integration="contact")
        g = call(tau, dtype=mx.float64, exp_time=EXP, integration="contact")
        assert np.abs(k - g).max() <= 5e-7

    @pytest.mark.parametrize("n", [1, 2, 7, 8, 9, 64])
    def test_small_chain_counts(self, n):
        m = 64
        tau = np.broadcast_to(np.linspace(-0.1, 0.1, m), (n, m)).copy()
        pars = [mx.array(np.full(n, v), dtype=mx.float32)
                for v in (P, NOM["a"], NOM["b"], NOM["r"], NOM["u1"],
                          NOM["u2"])]
        out = flux_dev_from_tau(mx.array(tau, dtype=mx.float32), *pars,
                                exp_time=EXP)
        mx.eval(out)
        assert out.shape == (n, m)
        assert np.isfinite(np.asarray(out)).all()

    def test_out_of_transit_is_exactly_zero(self):
        """anvil adds this to a baseline, so 'no transit' must be 0, not
        a rounding of it."""
        tau = np.concatenate([np.linspace(0.2, P / 2, 50),
                              np.linspace(-P / 2, -0.2, 50)])
        for integ, kw in MODES:
            got = call(tau, integration=integ, **kw)
            assert (got == 0.0).all(), integ

    def test_none_matches_the_route_it_replaces(self):
        """integration="none" is today's separation_circular +
        flux_dev_metal with tau -> z folded in, to the standing
        kernel-vs-graph tolerance (the two compute z in a different
        order, so they are not bit-identical)."""
        from metalplanet.metal import flux_dev_metal
        from metalplanet.orbit import separation_circular
        n, m = 8, 2048
        tau = np.broadcast_to(np.linspace(-0.2, 0.2, m), (n, m)).copy()
        t = mx.array(tau, dtype=mx.float32)
        pa = [mx.array(np.full(n, v), dtype=mx.float32)
              for v in (P, NOM["a"], NOM["b"], NOM["r"], NOM["u1"],
                        NOM["u2"])]
        col = [p[:, None] for p in pa]

        def old(tt):
            return flux_dev_metal(
                separation_circular(tt, col[0], col[2], col[1]),
                col[3], col[4], col[5])

        new = flux_dev_from_tau(t, *pa, integration="none")
        mx.eval(new, old(t))
        assert np.abs(np.asarray(new, dtype=np.float64)
                      - np.asarray(old(t), dtype=np.float64)).max() <= 5e-7
        ga = mx.grad(lambda tt: mx.sum(
            flux_dev_from_tau(tt, *pa, integration="none")))(t)
        gb = mx.grad(lambda tt: mx.sum(old(tt)))(t)
        mx.eval(ga, gb)
        ga, gb = np.asarray(ga, np.float64), np.asarray(gb, np.float64)
        assert np.abs(ga - gb).max() / np.abs(gb).max() < 5e-5

    def test_far_side_is_not_a_transit(self):
        """z is symmetric under phi -> phi + pi; without the cos phi cut
        the model would fabricate a transit at phase 0.5."""
        assert (call(np.linspace(0.45 * P, 0.55 * P, 101)) == 0.0).all()

    def test_exp_time_zero_forces_instantaneous(self):
        tau = np.linspace(-0.1, 0.1, 64)
        assert np.abs(call(tau, exp_time=0.0, integration="contact")
                      - call(tau, integration="none")).max() == 0.0

    def test_gradient_flows_through_a_tau_built_from_parameters(self):
        """The point of the entry point: turin's tau is a function of
        sampled offsets, so d/d(offset) must be non-zero and correct."""
        t = mx.array(np.linspace(-0.1, 0.1, 128), dtype=mx.float32)

        def f(off):
            return mx.sum(flux_dev_from_tau(
                t - off, P, NOM["a"], NOM["b"], NOM["r"], NOM["u1"],
                NOM["u2"], exp_time=EXP))

        g = mx.grad(f)(mx.array(0.001, dtype=mx.float32))
        mx.eval(g)
        assert np.isfinite(float(g)) and abs(float(g)) > 0.0

    def test_compile_grad_composition(self):
        """The shape an engine step actually has."""
        tau = mx.array(np.linspace(-0.1, 0.1, 256), dtype=mx.float32)

        def f(t, r):
            return mx.sum(flux_dev_from_tau(t, P, NOM["a"], NOM["b"], r,
                                            NOM["u1"], NOM["u2"],
                                            exp_time=EXP))

        eager = mx.grad(f, argnums=(0, 1))(tau, mx.array(NOM["r"],
                                                         dtype=mx.float32))
        comp = mx.compile(mx.grad(f, argnums=(0, 1)))(
            tau, mx.array(NOM["r"], dtype=mx.float32))
        mx.eval(eager, comp)
        for e, c in zip(eager, comp):
            assert np.abs(np.asarray(e) - np.asarray(c)).max() == 0.0

    def test_fp64_and_cpu_route_to_the_graph(self):
        tau = np.linspace(-0.1, 0.1, 64)
        ref = call(tau, dtype=mx.float64, exp_time=EXP)
        with mx.stream(mx.cpu):
            got = call(tau, exp_time=EXP)
        assert np.abs(got - ref).max() <= 5e-7

    @pytest.mark.parametrize("kw,msg", [
        (dict(integration="trapezoid"), "integration must be"),
        (dict(n_gl=0), "n_gl and n_sub"),
        (dict(n_sub=0), "n_gl and n_sub"),
        (dict(exp_time=-1.0), "exp_time must be"),
    ])
    def test_bad_arguments_raise(self, kw, msg):
        with pytest.raises(ValueError, match=msg):
            call(np.linspace(-0.1, 0.1, 8), **kw)

    def test_bad_tau_rank_raises(self):
        with pytest.raises(ValueError, match="tau must be"):
            call(np.zeros((2, 3, 4)))
