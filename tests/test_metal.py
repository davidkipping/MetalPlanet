"""Fused Metal kernel: parity with the graph core, the kernel-specific
failure-mode matrix from the reviewed plan, and fp64-oracle adjudication.
Skips wholesale on machines without a usable Metal device."""

import math

import numpy as np
import mlx.core as mx
import pytest

from metalplanet import flux_dev, flux_dev_analytic
from metalplanet.metal import flux_dev_metal, metal_available
from reference_limbdark import flux_quad_ref

pytestmark = pytest.mark.skipif(not metal_available(),
                                reason="Metal kernels unavailable")

RNG = np.random.default_rng(17)
U1, U2 = 0.4, 0.25


def _boundary_heavy(n, seed=17):
    rng = np.random.default_rng(seed)
    r = (10.0 ** rng.uniform(-2, np.log10(0.5), n)).astype(np.float32)
    kind = rng.integers(0, 6, n)
    off = np.where(rng.random(n) < 0.5, 0.0,
                   10.0 ** rng.uniform(-7, -1, n) * rng.choice([-1, 1], n))
    z = np.abs(np.select(
        [kind == 0, kind == 1, kind == 2, kind == 3],
        [1 - r + off, 1 + r + off, r + off, np.zeros(n)],
        default=rng.uniform(0, 1.4, n) * (1 + r))).astype(np.float32)
    return z, r


def _kernel_1d(z, r, u1=U1, u2=U2):
    """Evaluate the kernel with per-point params via (n, 1) layout."""
    n = z.size
    out = flux_dev_metal(
        mx.array(z.reshape(n, 1)), mx.array(r.astype(np.float32)),
        mx.array(np.full(n, u1, np.float32)),
        mx.array(np.full(n, u2, np.float32)))
    return np.array(out, dtype=np.float64)[:, 0]


class TestParity:
    def test_vs_graph_with_oracle_adjudication(self):
        """<= 5e-7 vs graph-fp32; exceedances allowed only if the kernel
        is closer to the fp64 oracle than the graph is."""
        z, r = _boundary_heavy(20_000)
        graph = np.array(flux_dev(mx.array(z), mx.array(r), U1, U2),
                         dtype=np.float64)
        kern = _kernel_1d(z, r)
        d = np.abs(graph - kern)
        bad = np.where(d > 5e-7)[0]
        for i in bad:
            ref = flux_quad_ref(float(r[i]), float(z[i]), U1, U2) - 1.0
            assert abs(kern[i] - ref) <= abs(graph[i] - ref), (
                f"kernel further from oracle at z={z[i]}, r={r[i]}")

    def test_vs_fp64_oracle(self):
        """Same 5e-6 bar as the graph fp32 path."""
        z, r = _boundary_heavy(3000)
        kern = _kernel_1d(z, r)
        ref = np.array([flux_quad_ref(float(rr), float(zz), U1, U2) - 1.0
                        for zz, rr in zip(z, r)])
        assert np.max(np.abs(kern - ref)) < 5e-6

    def test_double_sliver_precedence(self):
        """r = 0.5, z inside BOTH Taylor windows: contact must win, as in
        the graph's where-stack."""
        eps = np.float32(1.19209e-7)
        zs = np.array([0.5, 0.5 + 5 * eps, 0.5 - 5 * eps,
                       0.5 + 1e-4, 0.5 - 1e-4], dtype=np.float32)
        rs = np.full_like(zs, 0.5)
        graph = np.array(flux_dev(mx.array(zs), mx.array(rs), U1, U2),
                         dtype=np.float64)
        kern = _kernel_1d(zs, rs)
        np.testing.assert_allclose(kern, graph, rtol=0, atol=5e-7)


class TestKernelSpecific:
    def test_determinism_and_empty_transit(self):
        z = np.linspace(1.2, 3.0, 1000, dtype=np.float32)[None, :]
        args = (mx.array(z), mx.array([0.1], dtype=mx.float32),
                mx.array([0.4], dtype=mx.float32),
                mx.array([0.25], dtype=mx.float32))
        o1 = np.array(flux_dev_metal(*args))
        o2 = np.array(flux_dev_metal(*args))
        assert np.array_equal(o1, o2)
        assert (o1 == 0.0).all()  # all out of transit: exact zeros
        # all IN transit
        z2 = np.full((1, 257), 0.3, np.float32)
        oin = np.array(flux_dev_metal(mx.array(z2), *args[1:]))
        assert np.isfinite(oin).all() and (oin < 0).all()

    @pytest.mark.parametrize("n", [1, 2, 7, 8, 9, 64])
    def test_small_chain_counts_constant_address_space(self, n):
        z = RNG.uniform(0, 1.3, (n, 33)).astype(np.float32)
        r = np.full((n, 1), 0.1, np.float32)
        got = np.array(flux_dev_metal(
            mx.array(z), mx.array(r), mx.array(r * 4), mx.array(r * 2.5)))
        want = np.array(flux_dev(
            mx.array(z), mx.array(r), mx.array(r * 4), mx.array(r * 2.5)))
        np.testing.assert_allclose(got, want, rtol=0, atol=5e-7)

    def test_interleaved_small_large_kernel_cache(self):
        for n in (4, 512, 2, 512, 7):
            z = RNG.uniform(0, 1.3, (n, 65)).astype(np.float32)
            r = np.full((n, 1), 0.11, np.float32)
            got = np.array(flux_dev_metal(mx.array(z), mx.array(r),
                                          mx.array(r * 4), mx.array(r * 2)))
            want = np.array(flux_dev(mx.array(z), mx.array(r),
                                     mx.array(r * 4), mx.array(r * 2)))
            np.testing.assert_allclose(got, want, rtol=0, atol=5e-7)

    @pytest.mark.parametrize("m", [1, 31, 33, 257, 1000])
    def test_m_not_multiple_of_threadgroup(self, m):
        z = RNG.uniform(0, 1.3, (3, m)).astype(np.float32)
        r = np.full((3, 1), 0.1, np.float32)
        got = np.array(flux_dev_metal(mx.array(z), mx.array(r),
                                      mx.array(r * 4), mx.array(r * 2.5)))
        assert got.shape == (3, m) and np.isfinite(got).all()

    def test_broadcast_and_noncontiguous_layouts(self):
        base = RNG.uniform(0, 1.3, (4, 128)).astype(np.float32)
        cases = [
            (mx.array(base)[:, ::2], mx.array(np.full((4, 1), 0.1, np.float32))),
            (mx.array(base[:1]), mx.array(np.full((4, 1), 0.1, np.float32))),
            (mx.array(base[0]), mx.array(0.1, dtype=mx.float32)),
            (mx.array(base.T.copy().T), mx.array(np.full(4, 0.1, np.float32))),
        ]
        for z, r in cases:
            got = np.array(flux_dev_metal(z, r, 0.4, 0.25), dtype=np.float64)
            # graph needs strict numpy broadcasting: lift (n,) to (n, 1)
            r_g = r[:, None] if (isinstance(r, mx.array) and r.ndim == 1
                                 and z.ndim == 2) else r
            want = np.array(flux_dev(z, r_g, 0.4, 0.25), dtype=np.float64)
            assert got.shape == want.shape
            np.testing.assert_allclose(got, want, rtol=0, atol=5e-7)

    def test_nan_inf_contract(self):
        z = mx.array(np.array([[np.inf, np.nan, 0.5, 2.0]], dtype=np.float32))
        out = np.array(flux_dev_metal(
            z, mx.array([0.1], dtype=mx.float32),
            mx.array([0.4], dtype=mx.float32),
            mx.array([0.25], dtype=mx.float32)))[0]
        assert out[0] == 0.0          # +inf: past the exit test, exact 0
        assert np.isnan(out[1])       # NaN propagates, never garbage
        assert np.isfinite(out[2]) and out[3] == 0.0

    def test_dispatch_fallback(self):
        """fp64 and CPU-stream calls route to the graph core silently."""
        z64 = mx.array(np.linspace(0, 1.3, 64), dtype=mx.float64)
        with mx.stream(mx.cpu):
            a = np.array(flux_dev_metal(z64, 0.1, 0.4, 0.25))
            b = np.array(flux_dev_analytic(z64, 0.1, 0.4, 0.25))
        np.testing.assert_array_equal(a, b)
        z32 = mx.array(np.linspace(0, 1.3, 64).astype(np.float32))
        with mx.stream(mx.cpu):
            c = flux_dev_metal(z32, 0.1, 0.4, 0.25)
            assert np.isfinite(np.array(c)).all()


class TestGradients:
    def test_vjp_matches_graph(self):
        n, m = 32, 256
        z = mx.array(RNG.uniform(0, 1.3, (n, m)).astype(np.float32))
        r = mx.array(np.full((n, 1), 0.1, np.float32))
        u1 = mx.array(np.full((n, 1), 0.4, np.float32))
        u2 = mx.array(np.full((n, 1), 0.25, np.float32))
        gm = mx.grad(lambda *a: mx.sum(flux_dev_metal(*a)),
                     argnums=(0, 1, 2, 3))(z, r, u1, u2)
        gg = mx.grad(lambda *a: mx.sum(flux_dev_analytic(*a)),
                     argnums=(0, 1, 2, 3))(z, r, u1, u2)
        for a, b in zip(gm, gg):
            assert a.shape == b.shape
            aa, bb = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
            assert np.abs(aa - bb).max() <= 5e-5 * max(np.abs(bb).max(), 1e-6)

    def test_compile_grad_composition(self):
        """The engine's actual step shape: mx.compile(mx.grad(...))."""
        z = mx.array(RNG.uniform(0, 1.3, (8, 64)).astype(np.float32))
        r = mx.array(np.full((8, 1), 0.1, np.float32))
        u1 = mx.array(np.full((8, 1), 0.4, np.float32))
        u2 = mx.array(np.full((8, 1), 0.25, np.float32))

        def loss(*a):
            return mx.sum(flux_dev_metal(*a) ** 2)

        eager = mx.grad(loss, argnums=(0, 1, 2, 3))(z, r, u1, u2)
        comp = mx.compile(mx.grad(loss, argnums=(0, 1, 2, 3)))(z, r, u1, u2)
        for a, b in zip(eager, comp):
            assert bool(mx.allclose(a, b, atol=1e-7))

    def test_nan_free_boundary_sweep(self):
        z, r = _boundary_heavy(100_000, seed=23)
        g = mx.grad(lambda *a: mx.sum(flux_dev_metal(*a)),
                    argnums=(0, 1, 2, 3))(
            mx.array(z[:, None]), mx.array(r),
            mx.array(np.full_like(r, 0.4)), mx.array(np.full_like(r, 0.25)))
        assert all(bool(mx.all(mx.isfinite(gi))) for gi in g)


class TestModelKernel:
    """v2: orbit folded into the kernel (the anvil (v, x) contract)."""

    PERIOD_REF = 3.4565

    def _models(self):
        from metalplanet.anvil import make_quad_transit_flux
        return (make_quad_transit_flux(self.PERIOD_REF, core="metal"),
                make_quad_transit_flux(self.PERIOD_REF, core="analytic"))

    def _vx(self, n=32, m=4096, seed=3):
        rng = np.random.default_rng(seed)
        v = np.tile([0.009, -0.0005, 0.10, 0.30, 8.80, 0.4225, 0.3077,
                     1e-4], (n, 1)).astype(np.float32)
        v += 1e-3 * rng.standard_normal(v.shape).astype(np.float32)
        dt = rng.uniform(-1.7, 1.7, m)
        k = rng.integers(0, 26, m).astype(float)
        return mx.array(v), mx.array(np.stack([dt, k]).astype(np.float32))

    def test_forward_parity(self):
        m2, mg = self._models()
        v, x = self._vx()
        a = np.array(m2(v, x), dtype=np.float64)
        b = np.array(mg(v, x), dtype=np.float64)
        assert np.abs(a - b).max() < 1e-6

    def test_gradient_parity_all_eight(self):
        m2, mg = self._models()
        v, x = self._vx(n=16, m=2048)

        def loss(fn):
            return lambda vv: mx.sum(fn(vv, x) ** 2)

        g2 = np.array(mx.grad(loss(m2))(v), dtype=np.float64)
        gg = np.array(mx.grad(loss(mg))(v), dtype=np.float64)
        for j in range(8):
            sc = max(np.abs(gg[:, j]).max(), 1e-10)
            assert np.abs(g2[:, j] - gg[:, j]).max() / sc < 2e-4, j

    def test_wrap_tie_rounding(self):
        """dt exactly at the half-period wrap: rint must match mx.round
        (half-to-even) or points flip orbits between cores."""
        m2, mg = self._models()
        n = 4
        v = np.tile([0.0, 0.0, 0.10, 0.30, 8.80, 0.4225, 0.3077, 0.0],
                    (n, 1)).astype(np.float32)
        half = np.float32(self.PERIOD_REF / 2)
        dt = np.array([half, -half, half * 3, -half * 3], dtype=np.float32)
        x = mx.array(np.stack([dt, np.zeros(4, np.float32)]))
        a = np.array(m2(mx.array(v), x))
        b = np.array(mg(mx.array(v), x))
        np.testing.assert_array_equal(a, b)

    def test_compile_grad_composition(self):
        m2, _ = self._models()
        v, x = self._vx(n=8, m=512)

        def loss(vv):
            return mx.sum(m2(vv, x) ** 2)

        eager = mx.grad(loss)(v)
        comp = mx.compile(mx.grad(loss))(v)
        assert bool(mx.allclose(eager, comp, atol=1e-7))

    def test_fp64_cpu_fallback(self):
        """make_target's data-gen path: fp64 on the CPU stream must give
        the graph result exactly."""
        m2, mg = self._models()
        v, x = self._vx(n=4, m=256)
        with mx.stream(mx.cpu):
            v64 = v.astype(mx.float64)
            x64 = x.astype(mx.float64)
            a = np.array(m2(v64, x64))
            b = np.array(mg(v64, x64))
        np.testing.assert_array_equal(a, b)

    def test_out_of_transit_zero_dev(self):
        """Far-side and no-overlap points: deviation exactly df0."""
        m2, _ = self._models()
        n = 3
        v = np.tile([0.0, 0.0, 0.10, 0.30, 8.80, 0.4225, 0.3077, 2e-4],
                    (n, 1)).astype(np.float32)
        dt = np.linspace(0.9, 1.6, 64)  # far from any transit
        x = mx.array(np.stack([dt, np.zeros_like(dt)]).astype(np.float32))
        out = np.array(m2(mx.array(v), x), dtype=np.float64)
        np.testing.assert_allclose(out, 2e-4, rtol=0, atol=1e-9)


class TestSimdReduction:
    """The VJP reduces per-chain gradients inside the kernel via
    metal::simd_sum over ACTIVE lanes; early-exited lanes drop out by
    themselves and init_value covers fully-exited simdgroups."""

    PREF = 3.456

    @staticmethod
    def _args(n, m, dt_value=None, span=0.5):
        from metalplanet.anchored import pack_orbit_constants
        dt = (np.full(m, dt_value, np.float32) if dt_value is not None
              else np.linspace(-span, span, m).astype(np.float32))
        x2d = mx.array(np.vstack([dt, np.zeros(m, np.float32)]))
        o = np.ones(n, np.float32)
        zero = mx.array(0.0 * o)
        orb = pack_orbit_constants(zero, zero, mx.array(np.float32(0.3 / 8.8) * o))
        return (x2d, mx.array(0.0 * o), mx.array(0.0 * o), mx.array(0.1 * o),
                mx.array(8.8 * o), orb, mx.array(0.4225 * o), mx.array(0.3077 * o))

    def test_all_out_of_transit_is_exactly_zero(self):
        """Every lane early-returns, so no simdgroup writes at all: the
        kernel's init_value is what makes the partials read as zero."""
        from metalplanet.metal import make_model_core_metal
        args = self._args(4, 512, dt_value=0.5)
        core = make_model_core_metal(self.PREF)

        def f(*p):
            return mx.sum(core(args[0], *p))

        g = mx.grad(f, argnums=tuple(range(7)))(*args[1:])
        assert all(np.all(np.array(v) == 0.0) for v in g)

    def test_simd_sum_ignores_returned_lanes(self):
        """Pin the MSL semantics the design rests on: simd_sum reduces
        over ACTIVE lanes, so lanes that hit `return` drop out by
        themselves and simd_is_first() picks the lowest surviving lane.
        Guards against a compiler/runtime change invalidating it."""
        src = """
            uint x = thread_position_in_grid.x;
            if (x >= (uint)npts) return;
            float v = xin[x];
            if (v <= 0.0f) return;
            float s = metal::simd_sum(v);
            if (metal::simd_is_first()) part[x / 32u] = s;
        """
        k = mx.fast.metal_kernel(name="mp_test_simd_active",
                                 input_names=["xin", "npts"],
                                 output_names=["part"], source=src)
        m = 1000
        v = np.random.default_rng(5).uniform(-1, 1, m).astype(np.float32)
        out = k(inputs=[mx.array(v), m], output_shapes=[((m + 31) // 32,)],
                output_dtypes=[mx.float32], init_value=0.0,
                grid=(m, 1, 1), threadgroup=(256, 1, 1))[0]
        got = float(mx.sum(out))
        want = float(v[v > 0].sum())
        assert abs(got - want) <= 1e-5 * abs(want)

    def test_mixed_circular_and_eccentric_chains_in_one_batch(self):
        """The e == 0 fast path is a per-chain branch. A batch mixing
        circular and eccentric chains must reproduce each chain exactly
        as it computes alone -- values and gradients."""
        from metalplanet.anchored import pack_orbit_constants
        from metalplanet.metal import make_model_core_metal
        core = make_model_core_metal(self.PREF)
        m = 2048
        dt = np.linspace(-0.5, 0.5, m).astype(np.float32)
        x2d = mx.array(np.vstack([dt, np.zeros(m, np.float32)]))
        es = np.array([0.0, 0.3, 0.0, 0.7, 0.0, 0.05], np.float32)
        ws = np.array([0.0, 1.1, 0.0, 2.0, 0.0, 4.0], np.float32)
        n = es.size
        o = np.ones(n, np.float32)

        def pack(e, w):
            return pack_orbit_constants(mx.array(np.sqrt(e) * np.cos(w)),
                                        mx.array(np.sqrt(e) * np.sin(w)),
                                        mx.array(np.float32(0.3 / 8.8) * np.ones_like(e)))

        args = [mx.array(0.004 * o), mx.array(-0.0007 * o), mx.array(0.1 * o),
                mx.array(8.8 * o), pack(es, ws), mx.array(0.4225 * o),
                mx.array(0.3077 * o)]
        ct = mx.array(np.random.default_rng(3).standard_normal((n, m)).astype(np.float32))
        mixed = np.array(core(x2d, *args), dtype=np.float64)
        gm = mx.grad(lambda *p: mx.sum(core(x2d, *p) * ct), argnums=tuple(range(7)))(*args)
        gm = [np.array(v, dtype=np.float64) for v in gm]
        for j in range(n):                       # each chain alone
            a1 = [mx.array(np.array(v)[j:j + 1]) for v in args]
            alone = np.array(core(x2d, *a1), dtype=np.float64)[0]
            np.testing.assert_array_equal(mixed[j], alone)
            ga = mx.grad(lambda *p: mx.sum(core(x2d, *p) * ct[j:j + 1]),
                         argnums=tuple(range(7)))(*a1)
            for gmv, gav in zip(gm, ga):
                np.testing.assert_array_equal(np.array(gmv)[j:j + 1], np.array(gav))


class TestEccentricKernel:
    """v3: the transit-anchored eccentric orbit fused with the
    photometric core, plus its analytic VJP.

    The graph path (metalplanet.anchored) is the reference; it is itself
    pinned against kepler.separation_keplerian and finite differences in
    tests/test_anchored.py, so kernel -> graph -> FD closes the chain.
    """

    PREF, A, RP, U1c, U2c = 3.456, 8.8, 0.1, 0.4225, 0.3077
    CI = 0.3 / 8.8

    @staticmethod
    def _core():
        from metalplanet.metal import make_model_core_metal
        return make_model_core_metal(TestEccentricKernel.PREF)

    def _inputs(self, e, wdeg, n=4, m=2048, seed=2, f64=False):
        rng = np.random.default_rng(seed)
        w = math.radians(wdeg)
        dt = np.sort(rng.uniform(-1.8, 1.8, m))
        kk = rng.integers(0, 40, m).astype(float)
        npd = np.float64 if f64 else np.float32
        x2d = mx.array(np.stack([dt, kk]).astype(npd))
        o = np.ones(n)

        def arr(v):
            return mx.array((v * o).astype(npd))

        return x2d, [arr(0.004), arr(-0.0007), arr(self.RP), arr(self.A),
                     arr(math.sqrt(e) * math.cos(w)),
                     arr(math.sqrt(e) * math.sin(w)),
                     arr(self.CI), arr(self.U1c), arr(self.U2c)]

    def _graph(self, x2d, t0, pof, r, a, k, h, ci, u1, u2):
        from metalplanet.anchored import separation_anchored
        dt = x2d[0][None, :]
        kk = x2d[1][None, :]
        P = (self.PREF + pof)[:, None]
        tau = dt - (t0[:, None] + kk * pof[:, None])
        tau = tau - P * mx.round(tau / P)
        phi = (2.0 * math.pi) * tau / P
        z, front = separation_anchored(phi, k[:, None], h[:, None],
                                       a[:, None], ci[:, None])
        f = flux_dev(z, r[:, None], u1[:, None], u2[:, None])
        return mx.where(front & (z < 1.0 + r[:, None]), f, 0.0)

    def _kernel(self, x2d, t0, pof, r, a, k, h, ci, u1, u2):
        from metalplanet.anchored import pack_orbit_constants
        return self._core()(x2d, t0, pof, r, a,
                            pack_orbit_constants(k, h, ci), u1, u2)

    @pytest.mark.parametrize("e", [0.0, 1e-6, 1e-3, 0.05, 0.3, 0.7, 0.9,
                                   0.99, 0.999])
    @pytest.mark.parametrize("wdeg", [0.0, 90.0, 180.0, 270.0])
    def test_forward_parity(self, e, wdeg):
        """5e-7 for e <= 0.9; above that dE/dM = 1/(1 - e cos E) reaches
        1/(1-e), so two correct fp32 solvers legitimately part company
        and the fp64 oracle adjudicates instead."""
        x2d, args = self._inputs(e, wdeg)
        kn = np.array(self._kernel(x2d, *args), dtype=np.float64)
        gr = np.array(self._graph(x2d, *args), dtype=np.float64)
        x64, a64 = self._inputs(e, wdeg, f64=True)
        with mx.stream(mx.cpu):
            ref = np.array(self._graph(x64, *a64), dtype=np.float64)
        d_kg = np.abs(kn - gr).max()
        if e <= 0.9:
            assert d_kg < 5e-7
        else:
            assert np.abs(kn - ref).max() < 3.0 * max(
                np.abs(gr - ref).max(), 5e-7)

    def test_exact_circular_limit(self):
        """e = 0 takes the kernel's circular fast path and must reproduce
        the circular closed form -- z^2 = (a sin phi)^2 + (b cos phi)^2
        with b = a cos i -- evaluated independently in float64."""
        x2d, args = self._inputs(0.0, 0.0, n=4, m=2048)
        t0, pof, r, a, k, h, ci, u1, u2 = args
        ecc = np.array(self._kernel(x2d, *args), dtype=np.float64)
        dt = np.array(x2d[0], dtype=np.float64)
        kk = np.array(x2d[1], dtype=np.float64)
        P = self.PREF + float(pof[0])
        tau = dt - (float(t0[0]) + kk * float(pof[0]))
        tau -= P * np.round(tau / P)
        phi = 2.0 * math.pi * tau / P
        b = self.A * self.CI
        z = np.sqrt((self.A * np.sin(phi)) ** 2 + (b * np.cos(phi)) ** 2)
        with mx.stream(mx.cpu):
            f = np.array(flux_dev(mx.array(z, dtype=mx.float64), self.RP,
                                  self.U1c, self.U2c), dtype=np.float64)
        ref = np.where((np.cos(phi) > 0) & (z < 1.0 + self.RP), f, 0.0)
        assert np.abs(ecc[0] - ref).max() < 5e-7

    @pytest.mark.parametrize("e,wdeg", [(0.0, 0.0), (1e-4, 90.0),
                                        (0.05, 270.0), (0.3, 90.0),
                                        (0.7, 180.0)])
    def test_gradient_parity_all_nine(self, e, wdeg):
        """Every chain rule, against the float64 graph. A wrong rule is
        an O(1) error; the tolerance only has to exclude that, since
        both fp32 paths share the solve's ~1e-7 floor in delta, which
        dz/dci amplifies geometrically."""
        x2d, args = self._inputs(e, wdeg, n=3, m=2048)
        x64, a64 = self._inputs(e, wdeg, n=3, m=2048, f64=True)
        ct = mx.ones((3, 2048))
        idx = tuple(range(1, 10))

        def kloss(x, *p):
            return mx.sum(self._kernel(x, *p) * ct)

        def gloss(x, *p):
            return mx.sum(self._graph(x, *p)
                          * mx.array(np.ones((3, 2048), np.float64)))

        gk = mx.grad(kloss, argnums=idx)(x2d, *args)
        with mx.stream(mx.cpu):
            gr = mx.grad(gloss, argnums=idx)(x64, *a64)
        for j, (kv, rv) in enumerate(zip(gk, gr)):
            k_ = np.array(kv, dtype=np.float64)
            r_ = np.array(rv, dtype=np.float64)
            sc = max(np.abs(r_).max(), 1e-12)
            assert np.abs(k_ - r_).max() / sc < 2e-3, (j, e, wdeg)

    def test_starter_columns_get_zero_gradient(self):
        """ecc / e0 / mtra seed only the Markley starter and the 2-pi
        fold; their exact gradient is zero and pack_orbit_constants
        detaches them, so the kernel must return zeros there."""
        from metalplanet.anchored import pack_orbit_constants
        from metalplanet.metal import _ORB_COLS
        x2d, args = self._inputs(0.3, 47.0, n=2, m=1024)
        t0, pof, r, a, k, h, ci, u1, u2 = args
        orb = pack_orbit_constants(k, h, ci)

        def loss(ob):
            return mx.sum(self._core()(x2d, t0, pof, r, a, ob, u1, u2) ** 2)

        g = np.array(mx.grad(loss)(orb))
        for name in ("ecc", "e0", "mtra"):
            assert np.all(g[:, _ORB_COLS.index(name)] == 0.0), name
        assert np.abs(g[:, _ORB_COLS.index("ecw")]).max() > 0.0

    @pytest.mark.parametrize("m", [1, 31, 32, 33, 255, 256, 257, 1000])
    def test_partial_simdgroups(self, m):
        x2d, args = self._inputs(0.3, 47.0, n=3, m=m)
        out = np.array(self._kernel(x2d, *args), dtype=np.float64)
        ref = np.array(self._graph(x2d, *args), dtype=np.float64)
        assert np.abs(out - ref).max() < 5e-7
        ct = mx.ones((3, m))

        def kloss(*p):
            return mx.sum(self._kernel(x2d, *p) * ct)

        g = mx.grad(kloss, argnums=tuple(range(9)))(*args)
        assert all(np.isfinite(np.array(v)).all() for v in g)

    def test_all_out_of_transit_zero_flux_and_gradient(self):
        n, m = 3, 512
        o = np.ones(n, np.float32)
        x2d = mx.array(np.vstack([np.full(m, 1.2, np.float32),
                                  np.zeros(m, np.float32)]))
        args = [mx.array(0.0 * o), mx.array(0.0 * o), mx.array(0.1 * o),
                mx.array(self.A * o), mx.array(np.float32(0.3) * o),
                mx.array(np.float32(0.4) * o), mx.array(np.float32(self.CI) * o),
                mx.array(0.4225 * o), mx.array(0.3077 * o)]
        out = np.array(self._kernel(x2d, *args))
        assert np.all(out == 0.0)

        def kloss(*p):
            return mx.sum(self._kernel(x2d, *p))

        g = mx.grad(kloss, argnums=tuple(range(9)))(*args)
        assert all(np.all(np.array(v) == 0.0) for v in g)

    def test_compile_grad_composition(self):
        x2d, args = self._inputs(0.3, 47.0, n=2, m=512)

        def loss(*p):
            return mx.sum(self._kernel(x2d, *p) ** 2)

        f = mx.compile(mx.grad(loss, argnums=tuple(range(9))))
        g = f(*args)
        assert all(np.isfinite(np.array(v)).all() for v in g)

    def test_deterministic(self):
        x2d, args = self._inputs(0.3, 47.0, n=3, m=1024)
        a = np.array(self._kernel(x2d, *args))
        b = np.array(self._kernel(x2d, *args))
        np.testing.assert_array_equal(a, b)
