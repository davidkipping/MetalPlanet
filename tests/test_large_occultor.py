"""Occultors larger than the star: Rp/R* > 1 (0.11.0; white-dwarf hosts).

Four regimes for r > 1: none (z >= 1 + r), partial, and total occultation
(z <= r - 1), where the disc is fully covered and every term takes its
full-disk value with zero gradients. The reference is an mpmath integral
of the occulted intensity over circles about the star's centre -- the arc
inside the occultor, or the whole circle when x + z <= r -- i.e.
test_poly._oracle generalised to any radial law.

See docs/large-occultor-plan.md for the measured baseline this pins.
"""

import functools
import math

import numpy as np
import mlx.core as mx
import pytest

import metalplanet as mp
from metalplanet.hybrid import LAWS, w_to_c
from metalplanet.metal import flux_dev_from_tau, flux_dev_metal, metal_available
from metalplanet.metal_hybrid import flux_dev_metal_hybrid

mpm = pytest.importorskip("mpmath")
mpm.mp.dps = 20

KERNEL = metal_available()
U = (0.4, 0.25)
UP = (0.3, 0.2, -0.1, 0.05)
W = {"hybrid2": (0.3, 0.2), "hybrid4": (0.2, 0.2, 0.1, 0.1),
     "hybrid5": (0.2, 0.2, 0.1, 0.1, 0.1)}
RS = (1.2, 3.0, 7.28, 20.0, 50.0)


def f64(v):
    return mx.array(np.asarray(v, dtype=np.float64), dtype=mx.float64)


def f32(v):
    return mx.array(np.asarray(v, dtype=np.float32), dtype=mx.float32)


# ---------------------------------------------------------------------------
# the oracle
# ---------------------------------------------------------------------------

def _I(law):
    if law == "quadratic":
        return lambda x: (1 - U[0] * (1 - mpm.sqrt(1 - x * x))
                          - U[1] * (1 - mpm.sqrt(1 - x * x)) ** 2)
    if law == "poly4":
        return lambda x: 1 - sum(u * (1 - mpm.sqrt(1 - x * x)) ** (n + 1)
                                 for n, u in enumerate(UP))
    c, d = w_to_c(law, W[law])
    eps = LAWS[law].eps

    def I(x):
        m2 = 1 - x * x
        return (c[0] + c[1] * m2 + c[2] * m2 ** 2
                + sum(dk / (m2 + e) ** 2 for dk, e in zip(d, eps)))
    return I


@functools.lru_cache(maxsize=None)
def oracle(z, r, law):
    """F - 1 by direct integration of the occulted intensity."""
    I = _I(law)
    z, r = mpm.mpf(z), mpm.mpf(r)

    def alpha(x):
        if x + z <= r:
            return 2 * mpm.pi
        if x >= z + r or x + r <= z:
            return mpm.mpf(0)
        c = (x * x + z * z - r * r) / (2 * x * z)
        return 2 * mpm.acos(max(-1, min(1, c)))

    total = 2 * mpm.pi * mpm.quad(lambda x: I(x) * x, [0, 1])
    if z >= 1 + r:
        return 0.0
    hi = min(mpm.mpf(1), z + r)
    pts = sorted({mpm.mpf(0), min(abs(z - r), hi), hi})
    return float(-mpm.quad(lambda x: I(x) * alpha(x) * x, pts) / total)


def zgrid(r):
    """total (centre, middle, just inside), inner contact +, z ~ r,
    partial, outer contact -, none."""
    return [0.0, 0.5 * (r - 1), r - 1 - 1e-4, r - 1 + 1e-4, r - 0.3, r,
            r + 0.4, r + 1 - 1e-3, r + 1.1]


def graph64(z, r, law):
    with mx.stream(mx.cpu):
        z = f64(z)
        if law == "quadratic":
            out = mp.flux_dev(z, r, *U)
        elif law == "poly4":
            out = mp.flux_dev_poly(z, r, list(UP))
        else:
            out = mp.flux_dev_hybrid(z, r, list(W[law]), law)
        mx.eval(out)
        return np.asarray(out)


def kernel32(z, r, law):
    z = f32(z)
    if law == "quadratic":
        out = flux_dev_metal(z, r, *U)
    else:
        out = flux_dev_metal_hybrid(z, r, law, list(W[law]))
    mx.eval(out)
    return np.asarray(out, dtype=np.float64)


# fp64 gates, from the measured baseline with headroom (the polynomial
# recursion loses digits fastest with r)
TOL64 = {1.2: 1e-14, 3.0: 1e-14, 7.28: 1e-13, 20.0: 1e-12, 50.0: 1e-11}
TOL64_POLY = {1.2: 1e-14, 3.0: 1e-13, 7.28: 1e-12, 20.0: 1e-10,
              50.0: 1e-8}
# fp32 kernels: the partial-overlap error grows ~ r^2 (terms of size ~r
# cancel to O(1)); about 2x the measured worst case
TOL32 = {1.2: 5e-7, 3.0: 1e-6, 7.28: 2e-5, 20.0: 3e-4, 50.0: 5e-3}


@pytest.mark.parametrize("r", RS)
@pytest.mark.parametrize("law", ["quadratic", "poly4", "hybrid2", "hybrid4",
                                 "hybrid5"])
def test_fp64_graph_matches_the_oracle(r, law):
    zs = zgrid(r)
    ref = np.array([oracle(z, r, law) for z in zs])
    got = graph64(zs, r, law)
    tol = (TOL64_POLY if law == "poly4" else TOL64)[r]
    assert np.abs(got - ref).max() < tol


@pytest.mark.skipif(not KERNEL, reason="Metal unavailable")
@pytest.mark.parametrize("r", RS)
@pytest.mark.parametrize("law", ["quadratic", "hybrid2", "hybrid5"])
def test_fp32_kernels_match_the_oracle(r, law):
    zs = zgrid(r)
    ref = np.array([oracle(z, r, law) for z in zs])
    got = kernel32(zs, r, law)
    assert np.isfinite(got).all()
    assert np.abs(got - ref).max() < TOL32[r]


@pytest.mark.parametrize("r", RS)
def test_ld_basis_matches_its_contract(r):
    """B @ c / N @ c reproduces the scalar law on every regime, fp64."""
    zs = np.array(zgrid(r))
    with mx.stream(mx.cpu):
        B = np.asarray(flux_dev_metal(f64(zs), r, ld_basis=True))
    c = np.array([1 - U[0] - U[1], U[0] + 2 * U[1], -U[1]])
    N = np.array([math.pi, 2 * math.pi / 3, math.pi / 2])
    assert np.abs(B @ c / (N @ c) - graph64(zs, r, "quadratic")).max() < 1e-14


# ---------------------------------------------------------------------------
# total occultation: exact, flat
# ---------------------------------------------------------------------------

def _total_z(r):
    return np.array([0.0, 1e-9, 0.37 * (r - 1), r - 1 - 1e-9, r - 1])


@pytest.mark.parametrize("r", (1.2, 7.28, 50.0))
def test_total_occultation_is_exact_on_the_graph(r):
    """F - 1 = -1 and the basis is minus the full-disk fluxes; gradients in
    z, r and the limb darkening vanish (to ~1e-140 at z = 0: floors)."""
    z = _total_z(r)
    for law in ("quadratic", "poly4"):
        assert np.array_equal(graph64(z, r, law), -np.ones_like(z)), law
    for law in W:
        assert np.abs(graph64(z, r, law) + 1.0).max() <= 4.5e-16, law
        with mx.stream(mx.cpu):
            B = np.asarray(mp.shape_cols(f64(z), r, law))
        assert np.abs(B + LAWS[law].norms()).max() <= 4.5e-16, law
    with mx.stream(mx.cpu):
        s = mp.sn_dev_poly(f64(z), r, 8)
        mx.eval(*s)
        assert np.array_equal(np.asarray(s[0]), np.full(len(z), -math.pi))
        assert np.array_equal(np.asarray(s[1]),
                              np.full(len(z), -2 * math.pi / 3))
        assert max(np.abs(np.asarray(x)).max() for x in s[2:]) < 1e-200
        zz, rr = f64(z), f64(r)
        for law in ("quadratic", "hybrid5"):
            if law == "quadratic":
                fn = lambda z, r, u: mx.sum(mp.flux_dev(z, r, u[0], u[1]))
                u = f64(U)
            else:
                fn = lambda z, r, u: mx.sum(mp.flux_dev_hybrid(z, r, u, law))
                u = f64(W[law])
            g = mx.grad(fn, argnums=(0, 1, 2))(zz, rr, u)
            mx.eval(*g)
            gz, gr, gu = (np.asarray(x) for x in g)
            assert np.isfinite(gz).all() and np.isfinite(gr).all()
            assert np.abs(gz).max() < 1e-100 and abs(float(gr)) < 1e-100
            assert np.abs(gu).max() < 1e-14


@pytest.mark.skipif(not KERNEL, reason="Metal unavailable")
@pytest.mark.parametrize("r", (1.001, 1.2, 7.28, 50.0))
def test_total_occultation_is_exact_in_the_kernels(r):
    """Every fp32 kernel family: -1 exactly (NaN at z = 0 and errors up to
    17 before 0.11.0), the basis at minus the full-disk fluxes, and zero
    gradients from the analytic VJPs."""
    z = f32(_total_z(r)[:-1])                  # strictly inside, fp32
    assert np.array_equal(kernel32(z, r, "quadratic"), -np.ones(z.size))
    for law in W:
        assert np.abs(kernel32(z, r, law) + 1.0).max() <= 1.2e-7, law
        B = np.asarray(flux_dev_metal_hybrid(z, r, law, None, basis=True))
        assert np.abs(B + LAWS[law].norms()).max() <= 1e-6 * np.abs(
            LAWS[law].norms()).max(), law
    B = np.asarray(flux_dev_metal(z, r, ld_basis=True))
    assert np.array_equal(B, np.tile(np.float32(
        [-math.pi, -2 * math.pi / 3, -math.pi / 2]), (z.size, 1)))
    rr, uu = f32([r]), f32([U[0]])
    g = mx.grad(lambda z, r, u: mx.sum(flux_dev_metal(z[None], r, u, 0.25)),
                argnums=(0, 1, 2))(z, rr, uu)
    mx.eval(*g)
    gz, gr, gu = (np.asarray(x) for x in g)
    # z and r partials are assigned 0; d/du is (pi/3 - pi/3) / norm, an
    # fp32 cancellation: rounding, not signal
    assert np.array_equal(gz, np.zeros_like(gz))
    assert np.array_equal(gr, np.zeros_like(gr))
    assert np.abs(gu).max() < 1e-6


# ---------------------------------------------------------------------------
# the far side
# ---------------------------------------------------------------------------

P_, A_ = 1.4079, 336.0


def _params(r, b, ecc=0.0, law="quadratic"):
    p = mp.TransitParams()
    p.t0, p.per, p.rp, p.a = 0.0, P_, r, A_
    p.inc = math.degrees(math.acos(b / A_))
    p.ecc, p.w = ecc, 90.0 if ecc == 0.0 else 63.0
    p.limb_dark = law
    p.u = list(U) if law == "quadratic" else list(W[law])
    return p


@pytest.mark.parametrize("b", (7.79, 3.0, 0.3))
@pytest.mark.parametrize("ecc", (0.0, 0.2))
@pytest.mark.parametrize("dtype", (None, mx.float32), ids=("fp64", "fp32"))
@pytest.mark.parametrize("law", ("quadratic", "hybrid4"))
def test_no_eclipse_at_the_far_conjunction(b, ecc, dtype, law):
    """Away from transit the flux is exactly 1. Before 0.11.0 the far-side
    push 2 + z did not clear an occultor with r > 1: b < r - 1 got a full
    eclipse at the far conjunction on the fp64 graph."""
    if dtype is mx.float32 and not KERNEL:
        pytest.skip("Metal unavailable")
    p = _params(7.28, b, ecc, law)
    t = np.concatenate([np.linspace(0.05, 0.95, 901) * P_])
    f = mp.TransitModel(p, t, dtype=dtype).light_curve(p)
    assert np.array_equal(f, np.ones_like(f))


def test_secondary_eclipse_has_no_dip_at_primary_conjunction():
    """The secondary branch pushes the near side the same way."""
    p = _params(7.28, 3.0)
    p.fp, p.t_secondary = 1e-3, 0.5 * P_
    t = np.linspace(-0.02, 0.02, 401)
    f = mp.TransitModel(p, t, transittype="secondary").light_curve(p)
    assert np.array_equal(f, np.full_like(f, 1.0 + 1e-3))


def test_tau_entry_far_side_is_flat():
    tau = np.linspace(0.3, 0.5, 201) * P_
    with mx.stream(mx.cpu):
        g = np.asarray(flux_dev_from_tau(f64(tau), P_, A_, 3.0, 7.28, *U))
    assert np.array_equal(g, np.zeros_like(g))
    if KERNEL:
        k = np.asarray(flux_dev_from_tau(f32(tau), P_, A_, 3.0, 7.28, *U))
        assert np.array_equal(k, np.zeros_like(k))


# ---------------------------------------------------------------------------
# the exposure contact rule
# ---------------------------------------------------------------------------

EXP = 2.0 / 1440.0


def _z_of_t(t, b):
    ph = 2 * np.pi * t / P_
    return np.sqrt((A_ * np.sin(ph)) ** 2 + (b * np.cos(ph)) ** 2)


def _t_at(Z, b):
    if Z <= b:
        return None
    lo, hi = 0.0, 0.25 * P_
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _z_of_t(mid, b) < Z else (lo, mid)
    return 0.5 * (lo + hi)


def _exact_exposures(tc, b, r):
    """Exposure averages with 64-point Gauss-Legendre on every piece
    between the true contacts (z = 1 + r and z = |1 - r|)."""
    xg, wg = np.polynomial.legendre.leggauss(64)
    cps = [s * c for c in (_t_at(1 + r, b), _t_at(abs(r - 1), b))
           if c is not None for s in (1, -1)]
    nodes, wts, owner = [], [], []
    for i, t0 in enumerate(tc):
        lo, hi = t0 - EXP / 2, t0 + EXP / 2
        cuts = [lo] + sorted(c for c in cps if lo < c < hi) + [hi]
        for a, c in zip(cuts[:-1], cuts[1:]):
            nodes.append(0.5 * (c - a) * xg + 0.5 * (c + a))
            wts.append(0.5 * (c - a) * wg)
            owner.append(np.full(64, i))
    nodes, wts, owner = map(np.concatenate, (nodes, wts, owner))
    with mx.stream(mx.cpu):
        f = np.asarray(mp.flux_dev(f64(_z_of_t(nodes, b)), r, *U))
    return np.bincount(owner, weights=f * wts) / EXP


@pytest.mark.parametrize("r,b", [(7.28, 7.79), (7.28, 3.0), (1.5, 0.2)])
def test_contact_rule_against_the_exact_piecewise_integral(r, b):
    """The contact rule converges as for r < 1: n_gl = 9 is within 2e-6 of
    a 100% eclipse (n_gl = 5, 2e-5 -- about 1e-5 of the depth, finer than
    a 1% transit's 2.4e-6 at the same settings)."""
    tc = np.linspace(-0.012, 0.012, 241)
    ref = _exact_exposures(tc, b, r)
    p = _params(r, b)
    for n_gl, tol in ((5, 3e-5), (9, 2e-6)):
        with mx.stream(mx.cpu):
            tau = np.asarray(flux_dev_from_tau(
                f64(tc), P_, A_, b, r, *U, exp_time=EXP,
                integration="contact", n_gl=n_gl))
        tm = mp.TransitModel(p, tc, exp_time=EXP, integration="contact",
                             n_gl=n_gl).light_curve(p) - 1.0
        assert np.abs(tau - ref).max() < tol, n_gl
        assert np.abs(tm - ref).max() < tol, n_gl


def test_inner_contacts_sit_at_r_minus_one():
    from metalplanet.exposure import contact_offsets
    r, b = 7.28, 3.0
    with mx.stream(mx.cpu):               # an internal helper: fp64 on CPU
        phis = [float(np.asarray(x)) for x in contact_offsets(
            f64(r), f64(A_), f64(b))]
    ts = np.array(phis) * P_ / (2 * np.pi)
    z = _z_of_t(ts, b)
    assert np.allclose(z, [1 + r, r - 1, r - 1, 1 + r], rtol=1e-12)


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r", (1.5, 7.28, 20.0))
@pytest.mark.parametrize("law", ("quadratic", "hybrid5"))
def test_graph_gradients_match_finite_differences(r, law):
    """Partial overlap, off the z == r tie (MLX's autodiff splits the
    gradient exactly there, for any r)."""
    zs = np.array([r - 0.7, r - 0.2, r + 0.3, r + 0.8])

    def F(z, rr):
        if law == "quadratic":
            return mp.flux_dev(z, rr, *U)
        return mp.flux_dev_hybrid(z, rr, list(W[law]), law)

    with mx.stream(mx.cpu):
        gz, gr = mx.grad(lambda z, rr: mx.sum(F(z, rr)), argnums=(0, 1))(
            f64(zs), f64([r]))
        mx.eval(gz, gr)
        h = 1e-6
        fdz = (np.asarray(F(f64(zs + h), r)) - np.asarray(F(f64(zs - h), r))) / (2 * h)
        fdr = (np.asarray(F(f64(zs), r + h)) - np.asarray(F(f64(zs), r - h))).sum() / (2 * h)
    assert np.allclose(np.asarray(gz), fdz, rtol=1e-6, atol=1e-9)
    assert abs(float(np.asarray(gr)[0]) - fdr) <= 1e-6 * abs(fdr) + 1e-9


@pytest.mark.skipif(not KERNEL, reason="Metal unavailable")
@pytest.mark.parametrize("b", (7.79, 3.0))
def test_kernel_vjp_matches_fp64_autodiff(b):
    tau = np.linspace(-0.006, 0.006, 2001)

    def loss(dt):
        return lambda r, u: mx.sum(flux_dev_from_tau(
            mx.array(tau, dtype=dt), P_, A_, b, r, u, 0.25))

    g32 = mx.grad(loss(mx.float32), argnums=(0, 1))(f32(7.28), f32(0.4))
    mx.eval(*g32)
    with mx.stream(mx.cpu):
        g64 = mx.grad(loss(mx.float64), argnums=(0, 1))(f64(7.28), f64(0.4))
        mx.eval(*g64)
    for a, b_ in zip(g32, g64):
        a, b_ = float(np.asarray(a)), float(np.asarray(b_))
        assert abs(a - b_) <= 2e-3 * abs(b_)


# ---------------------------------------------------------------------------
# r near 1, and the hybrid internal-contact NaN
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r", (1 - 1e-7, 1.0, 1 + 1e-7, 1 + 1e-3))
def test_r_near_one_is_finite_and_accurate(r):
    """r == 1 is degenerate (the contacts meet at z = 0): 5e-9 in fp64,
    1e-4 in fp32, as before 0.11.0; finite everywhere."""
    zs = [0.0, 1e-8, 1e-4, 1e-3, 0.3, 1.0, 1.9]
    ref = np.array([oracle(z, r, "quadratic") for z in zs])
    got = graph64(zs, r, "quadratic")
    assert np.isfinite(got).all() and np.abs(got - ref).max() < 1e-8
    if KERNEL:
        k = kernel32(zs, r, "quadratic")
        assert np.isfinite(k).all() and np.abs(k - ref).max() < 2e-4


@pytest.mark.parametrize("law", list(W))
def test_hybrid_internal_contact_is_finite_in_fp32(law):
    """Where z + r rounds to exactly 1 in fp32 the lens depth -Bp was -0.0:
    the pole's ratio flipped to -inf and the lane took sqrt(-Q) -> NaN, in
    every column (generator mixing) -- about 1 point in 10 within two ulps
    of the internal contact, graph and kernel alike (v0.10.x)."""
    rs = np.linspace(0.05, 0.95, 91).astype(np.float32)
    zs, rr = [], []
    for r in rs:
        zc = np.float32(1) - r
        for k in range(-2, 3):
            z = zc
            for _ in range(abs(k)):
                z = np.nextafter(z, np.float32(2 if k > 0 else 0),
                                 dtype=np.float32)
            zs.append(z)
            rr.append(r)
    z = f32(zs)[None, :]
    r = f32(rr)[None, :]
    w = list(W[law])
    g = np.asarray(mp.flux_dev_hybrid(z, r, w, law))
    gb = np.asarray(mp.shape_cols(z, r, law))
    assert np.isfinite(g).all() and np.isfinite(gb).all()
    if KERNEL:
        k = np.asarray(flux_dev_metal_hybrid(f32(zs), f32(rr), law, w))
        assert np.isfinite(k).all()


# ---------------------------------------------------------------------------
# end to end: a WD 1856+534 b-like system
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("b", (7.79, 3.0), ids=("grazing", "total"))
def test_white_dwarf_end_to_end(b):
    """r = 7.28, a = 336: mid-transit against the oracle; fp32 within the
    r^2 budget; the contact rule finite."""
    p = _params(7.28, b)
    t = np.linspace(-0.01, 0.01, 801)
    f64_ = mp.TransitModel(p, t).light_curve(p)
    mid = f64_[400] - 1.0
    assert abs(mid - oracle(b, 7.28, "quadratic")) < 1e-12
    fc = mp.TransitModel(p, t, exp_time=EXP, integration="contact",
                         n_gl=9).light_curve(p)
    assert np.isfinite(fc).all() and fc.min() >= -1e-12
    if KERNEL:
        f32_ = mp.TransitModel(p, t, dtype=mx.float32).light_curve(p)
        assert np.abs(f32_ - f64_).max() < TOL32[7.28]


@pytest.mark.parametrize("b", (7.79, 3.0), ids=("grazing", "total"))
def test_anvil_target_samples_large_occultors(b):
    """make_transit_target with widened bounds (README): finite density and
    gradient at r > 1, and the truth scores highest."""
    A = pytest.importorskip("metalplanet.anvil")
    p = _params(7.28, b)
    t = np.linspace(-0.01, 0.01, 1001)
    y = mp.TransitModel(p, t).light_curve(p)
    y = y + np.random.default_rng(0).normal(0.0, 1e-3, t.size)
    tg = A.make_transit_target(t, y, 1e-3, 0.0, P_, bounds={
        "r": (1.0, 12.0), "b": (0.0, 13.0), "a": (1.5, 1000.0)})
    names = list(tg.transform.names)
    truth = {"t0_off": 0.0, "p_off": 0.0, "r": 7.28, "b": b, "a": A_,
             "q1": (U[0] + U[1]) ** 2, "q2": U[0] / (2 * (U[0] + U[1])),
             "df0": 0.0}
    th = np.array([[truth[n] for n in names]] * 3)
    th[1, names.index("r")] = 6.9
    th[2, names.index("b")] = b - 0.4
    u = tg.transform.from_model_np(th)
    lp, g = tg.target.log_prob_and_grad(mx.array(np.asarray(u),
                                                 dtype=mx.float32))
    mx.eval(lp, g)
    lp, g = np.asarray(lp), np.asarray(g)
    assert np.isfinite(lp).all() and np.isfinite(g).all()
    assert lp.argmax() == 0
