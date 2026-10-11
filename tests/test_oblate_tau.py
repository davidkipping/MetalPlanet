"""Oblate planets, Stage 2: flux_dev_from_tau(..., f=, theta=).

Sky frame and theta pinned to SquishierPlanet (importorskip); per-side
contacts against a dense, independent reference; the exposure rules; f = 0
against the spherical path; fp32 against fp64; gradients against finite
differences in every input; the keyword checks.
"""

import math
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx
import pytest

import metalplanet as mp
from metalplanet import oblate_tau as OT
from metalplanet.hybrid import hybrid_norms
from metalplanet.metal import _ecc_shape

_SP = Path(__file__).resolve().parents[2] / "SquishierPlanet"
if _SP.is_dir() and str(_SP) not in sys.path:
    sys.path.insert(0, str(_SP))
try:
    from squishierplanet import laws as spl
    from squishierplanet.model import light_curve as sp_light_curve
    from squishierplanet.orbit import sky_position
except Exception:
    spl = None

needs_sp = pytest.mark.skipif(spl is None, reason="squishierplanet not importable")

P, A_RS = 3.45, 8.8
W = {"hybrid2": [0.3, 0.2], "hybrid4": [0.2, 0.2, 0.1, 0.1],
     "hybrid5": [0.2, 0.2, 0.1, 0.1, 0.1]}
TAU = np.linspace(-0.13, 0.13, 321)
EXP = 1800.0 / 86400.0


def flux(tau, b, r, law="hybrid5", **kw):
    out = mp.flux_dev_from_tau(tau, P, A_RS, b, r, limb_dark=law,
                               u=kw.pop("u", W[law]), **kw)
    mx.eval(out)
    return np.asarray(out.astype(mx.float32) if out.dtype == mx.float32 else out,
                      dtype=np.float64)


def kh(e, w):
    return math.sqrt(e) * math.cos(w), math.sqrt(e) * math.sin(w)


def cos_i(b, k, h):
    with mx.stream(mx.cpu):
        v = [mx.array(x, dtype=mx.float64) for x in (A_RS, b, k, h)]
        return float(_ecc_shape(*v)[2])


# ---------------------------------------------------------------------------
# sky frame
# ---------------------------------------------------------------------------

@needs_sp
@pytest.mark.parametrize("e,w", [(0.0, math.pi / 2), (0.3, 1.0), (0.6, -2.0)])
def test_sky_frame_is_squishierplanets(e, w):
    k, h = kh(e, w)
    b = 0.3
    ci = cos_i(b, k, h)
    t = np.linspace(-0.2, 0.2, 41)
    phi = 2 * np.pi * t / P
    with mx.stream(mx.cpu):
        f64 = lambda v: mx.array(v, dtype=mx.float64)
        if e == 0.0:
            X, Y, _, _, front = OT.sky_circular(f64(phi), A_RS, b)
        else:
            from metalplanet.anchored import anchor_constants
            X, Y, _, _, front = OT.sky_anchored(f64(phi), A_RS, ci,
                                                anchor_constants(f64(k), f64(h)))
        X, Y, front = np.asarray(X), np.asarray(Y), np.asarray(front)
    Xr, Yr, Zr = sky_position(t, period=P, a=A_RS, i=math.acos(ci), t0=0.0,
                              e=e, omega=w)
    assert np.abs(X - Xr).max() < 1e-13 and np.abs(Y - Yr).max() < 1e-13
    assert np.array_equal(front, Zr > 0)


def test_sky_derivatives():
    from metalplanet.anchored import anchor_constants
    k, h = kh(0.4, 0.7)
    with mx.stream(mx.cpu):
        f64 = lambda v: mx.array(v, dtype=mx.float64)
        phi = f64(np.linspace(-0.3, 0.3, 13))
        consts = anchor_constants(f64(k), f64(h))
        for fn in (lambda p: OT.sky_circular(p, A_RS, 0.4),
                   lambda p: OT.sky_anchored(p, A_RS, 0.05, consts)):
            X, Y, dX, dY, _ = fn(phi)
            hstep = 1e-6
            Xp, Yp = fn(phi + hstep)[:2]
            Xm, Ym = fn(phi - hstep)[:2]
            assert np.abs(np.asarray(dX) - np.asarray((Xp - Xm) / (2 * hstep))).max() < 1e-7
            assert np.abs(np.asarray(dY) - np.asarray((Yp - Ym) / (2 * hstep))).max() < 1e-7


# ---------------------------------------------------------------------------
# light curves against SquishierPlanet
# ---------------------------------------------------------------------------

@needs_sp
@pytest.mark.parametrize("law", ["hybrid2", "hybrid4", "hybrid5"])
@pytest.mark.parametrize("r,f,theta,b,e,w", [
    (0.1, 0.1, 0.4, 0.3, 0.0, 0.0),
    (0.15, 0.3, 1.0, 0.6, 0.0, 0.0),
    (0.05, 0.5, 2.5, 0.0, 0.0, 0.0),
    (0.12, 0.4, 0.8, 1.02, 0.0, 0.0),            # grazing
    (0.1, 0.3, 2.0, 0.5, 0.3, 1.0),
    (0.08, 0.2, 0.3, 0.2, 0.6, -2.0),
])
def test_light_curve_matches_squishierplanet(law, r, f, theta, b, e, w):
    if law == "hybrid2" and float(spl.HYBRID2_EPS) != mp.HYBRID2_EPS:
        pytest.skip("this SquishierPlanet checkout's hybrid2 pole is not MetalPlanet's")
    kw = dict(f=f, theta=theta, integration="none")
    inc = math.acos(b / A_RS)
    if e > 0:
        k, h = kh(e, w)
        kw.update(secosw=k, sesinw=h)
        inc = math.acos(cos_i(b, k, h))
    got = flux(TAU, b, r, law, **kw)
    ref = sp_light_curve(TAU, spl.hybrid(law, W[law]), r_eff=r, f=f,
                         theta=theta, period=P, a=A_RS, i=inc, t0=0.0, e=e,
                         omega=w if e > 0 else np.pi / 2) - 1.0
    assert ref.min() < -1e-4                       # a transit is tested
    assert np.abs(got - ref).max() < 1e-12


# ---------------------------------------------------------------------------
# contacts
# ---------------------------------------------------------------------------

_PSI = np.linspace(0, 2 * np.pi, 1024, endpoint=False)


def _ref_q_M(phi, a, b, e, w, A, B, th):
    """Independent q and M (numpy, SquishierPlanet's orbit), far side inf."""
    k, h = kh(e, w) if e > 0 else (0.0, 0.0)
    inc = math.acos(cos_i(b, k, h)) if e > 0 else math.acos(b / a)
    X, Y, Z = sky_position(np.atleast_1d(phi) * P / (2 * np.pi), period=P,
                           a=a, i=inc, t0=0.0, e=e,
                           omega=w if e > 0 else np.pi / 2)
    c, s = math.cos(th), math.sin(th)
    x0, y0 = X * c + Y * s, -X * s + Y * c
    sv = (x0[:, None] + A * np.cos(_PSI)) ** 2 + (y0[:, None] + B * np.sin(_PSI)) ** 2

    def polish(p):
        for _ in range(6):
            xx, yy = x0 + A * np.cos(p), y0 + B * np.sin(p)
            d1 = 2 * (-xx * A * np.sin(p) + yy * B * np.cos(p))
            d2 = 2 * (A * A * np.sin(p) ** 2 - xx * A * np.cos(p)
                      + B * B * np.cos(p) ** 2 - yy * B * np.sin(p))
            p = p - np.where(d2 != 0, d1 / np.where(d2 != 0, d2, 1), 0)
        return (x0 + A * np.cos(p)) ** 2 + (y0 + B * np.sin(p)) ** 2

    q = np.where((x0 / A) ** 2 + (y0 / B) ** 2 < 1, 0.0, polish(_PSI[sv.argmin(1)]))
    Mx = polish(_PSI[sv.argmax(1)])
    return np.where(Z > 0, q, np.inf), np.where(Z > 0, Mx, np.inf)


def _ref_contacts(a, b, e, w, A, B, th):
    grid = np.linspace(-0.6, 0.6, 12001)
    q, Mx = _ref_q_M(grid, a, b, e, w, A, B, th)
    out = []
    for vals, which in ((q, 0), (Mx, 1)):
        sg = np.sign(vals - 1)
        roots = []
        for j in np.nonzero(np.diff(sg) != 0)[0]:
            lo, hi, slo = grid[j], grid[j + 1], sg[j]
            for _ in range(45):
                mid = 0.5 * (lo + hi)
                if np.sign(_ref_q_M(mid, a, b, e, w, A, B, th)[which][0] - 1) == slo:
                    lo = mid
                else:
                    hi = mid
            roots.append(0.5 * (lo + hi))
        out.append(roots)
    return out


def _contact_cases():
    rng = np.random.default_rng(4)
    rows = []
    for n in range(40):
        r = rng.uniform(0.02, 0.3)
        f = rng.choice([0.01, 0.1, 0.3, 0.5])
        r = min(r, 0.95 * (1 - f) ** 1.5)
        A = r / math.sqrt(1 - f)
        e = 0.0 if n % 2 == 0 else rng.uniform(0.05, 0.6)
        w = rng.uniform(-np.pi, np.pi)
        b = rng.uniform(1 - A, 1 + A) if n % 3 == 0 else rng.uniform(0, 1 + A)
        rows.append((b, r, f, rng.uniform(0, np.pi), e, w))
    # a tilted graze 3e-4 inside tangency, whose outer contacts are both
    # before conjunction (-0.0146, -0.0087), and a planet that misses
    rows += [(1.167483, 0.15, 0.5, math.pi / 4, 0.0, 0.0),
             (1.4, 0.1, 0.3, 0.5, 0.0, 0.0)]
    return np.array(rows).T


@needs_sp
@pytest.mark.parametrize("dtype", [mx.float64, mx.float32], ids=["fp64", "fp32"])
def test_contacts_match_a_dense_reference(dtype):
    b, r, f, th, e, w = _contact_cases()
    k, h = np.sqrt(e) * np.cos(w), np.sqrt(e) * np.sin(w)
    stream = mx.cpu if dtype == mx.float64 else mx.gpu
    with mx.stream(stream):
        arr = lambda v: mx.array(np.broadcast_to(v, b.shape).copy(), dtype=dtype)
        ci = _ecc_shape(arr(A_RS), arr(b), arr(k), arr(h))[2]
        circ = OT.contact_offsets_oblate(arr(A_RS), arr(b), arr(r), arr(f), arr(th))
        ecc = OT.contact_offsets_oblate(arr(A_RS), arr(b), arr(r), arr(f), arr(th),
                                        arr(k), arr(h), ci)
        mx.eval(*circ, *ecc)
        cv = lambda c: np.asarray(c.astype(mx.float32) if dtype == mx.float32 else c,
                                  dtype=np.float64)
        circ = np.stack([cv(c) for c in circ], 1)
        ecc = np.stack([cv(c) for c in ecc], 1)
    tol = 1e-12 if dtype == mx.float64 else 2e-6
    A = r / np.sqrt(1 - f)
    B = A * (1 - f)
    seen_same_side = seen_none = seen_graze = False
    for j in range(len(b)):
        got = ecc[j] if e[j] > 0 else circ[j]
        assert np.all(np.diff(got) >= 0), j
        outer, inner = _ref_contacts(A_RS, b[j], e[j], w[j], A[j], B[j], th[j])
        if not outer:
            assert got[0] == got[3], j
            seen_none = True
            continue
        assert len(outer) == 2, j
        assert abs(got[0] - outer[0]) < tol and abs(got[3] - outer[1]) < tol, j
        seen_same_side |= outer[0] > 0 or outer[1] < 0
        if len(inner) == 2:
            assert abs(got[1] - inner[0]) < tol and abs(got[2] - inner[1]) < tol, j
        else:
            assert got[1] == got[2] and got[0] <= got[1] <= got[3], j
            seen_graze = True
    assert seen_same_side and seen_none and seen_graze


# ---------------------------------------------------------------------------
# exposures, f -> 0, precision
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r,f,theta,b,ecc", [
    (0.1, 0.3, 1.0, 0.5, None), (0.1, 0.5, 0.2, 0.0, None),
    (0.12, 0.4, 0.8, 1.02, None), (0.1, 0.3, 2.0, 0.6, (0.4, 0.3))])
def test_contact_rule_converges(r, f, theta, b, ecc):
    """30-minute exposures, 5 Gauss-Legendre nodes per piece, split at the
    oblate contacts: the rule's own floor (~1e-7), against 40 nodes."""
    kw = dict(f=f, theta=theta, exp_time=EXP)
    if ecc:
        kw.update(secosw=ecc[0], sesinw=ecc[1])
    ref = flux(TAU, b, r, n_gl=40, **kw)
    got = flux(TAU, b, r, n_gl=5, **kw)
    assert np.abs(got - ref).max() < 3e-7


@pytest.mark.parametrize("theta", [0.0, 0.7])
@pytest.mark.parametrize("integration,extra", [
    ("none", {}), ("contact", {"exp_time": EXP}),
    ("supersample", {"exp_time": EXP, "n_sub": 7})])
def test_f_zero_is_the_spherical_path(theta, integration, extra):
    obl = flux(TAU, 0.3, 0.1, f=0.0, theta=theta, integration=integration, **extra)
    sph = flux(TAU, 0.3, 0.1, integration=integration, **extra)
    assert np.abs(obl - sph).max() < 1e-16


@pytest.mark.parametrize("integration,extra", [("none", {}), ("contact", {"exp_time": EXP})])
@pytest.mark.parametrize("f", [2e-5, 0.1, 0.4])
def test_fp32_matches_fp64(integration, extra, f):
    kw = dict(f=f, theta=0.9, integration=integration, **extra)
    d64 = flux(TAU, 0.4, 0.1, **kw)
    d32 = flux(TAU.astype(np.float32), 0.4, 0.1, **kw)
    assert np.abs(d32 - d64).max() < 3e-7


def test_out_of_transit_and_far_side_are_exactly_zero():
    tau = np.concatenate([np.linspace(-1.7, -0.2, 50), np.linspace(0.2, 1.7, 50)])
    got = flux(tau, 0.1, 0.1, f=0.3, theta=0.4)
    assert np.array_equal(got, np.zeros_like(got))


def test_ld_basis_identity():
    for law in ("hybrid2", "hybrid4", "hybrid5"):
        kw = dict(f=0.25, theta=0.6, exp_time=EXP)
        B = np.asarray(mp.flux_dev_from_tau(TAU, P, A_RS, 0.3, 0.1, limb_dark=law,
                                            ld_basis=True, **kw))
        c = np.concatenate([[1.0], -np.asarray(W[law])])
        got = flux(TAU, 0.3, 0.1, law, **kw)
        assert np.abs(got - (B @ c) / (hybrid_norms(law) @ c)).max() < 1e-16


def test_per_chain_f_and_theta():
    fs, ths = [0.1, 0.3, 0.5], [0.0, 1.0, 2.5]
    rows = flux(np.stack([TAU] * 3), 0.3, 0.1, f=mx.array(fs, dtype=mx.float64),
                theta=mx.array(ths, dtype=mx.float64), exp_time=EXP)
    for j in range(3):
        one = flux(TAU, 0.3, 0.1, f=fs[j], theta=ths[j], exp_time=EXP)
        assert np.abs(rows[j] - one).max() < 1e-16, j


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------

_NAMES = ["tau", "period", "a", "b", "r", "f", "theta", "w", "secosw", "sesinw"]


def _loss_fn(integration, extra):
    ct = np.random.default_rng(1).normal(size=TAU.shape)

    def loss(tau, period, a, b, r, f, theta, w, k, h):
        out = mp.flux_dev_from_tau(tau, period, a, b, r, limb_dark="hybrid4",
                                   u=w, f=f, theta=theta, secosw=k, sesinw=h,
                                   integration=integration, **extra)
        return mx.sum(mx.array(ct, dtype=mx.float64) * out)
    return loss


@pytest.mark.parametrize("integration,extra,tol", [
    ("none", {}, 1e-6),
    ("supersample", {"exp_time": EXP, "n_sub": 5}, 1e-6),
    # contacts are detached split points: FD moves them, autodiff does not,
    # so the two agree to the rule's own error -- measured 1e-5 at 12 nodes,
    # 3e-7 at 40 (both shrink together, parameter by parameter)
    pytest.param("contact", {"exp_time": EXP, "n_gl": 40}, 2e-6,
                 marks=pytest.mark.slow)])
def test_gradients_match_finite_differences(integration, extra, tol):
    loss = _loss_fn(integration, extra)
    x = [TAU, P, A_RS, 0.35, 0.1, 0.3, 0.8, [0.2, 0.2, 0.1, 0.1], 0.3, 0.2]
    with mx.stream(mx.cpu):
        args = [mx.array(np.asarray(v, dtype=np.float64), dtype=mx.float64) for v in x]
        grads = mx.grad(loss, argnums=tuple(range(10)))(*args)
        mx.eval(grads)
        h = 1e-6
        for i, name in enumerate(_NAMES):
            g = np.asarray(grads[i])
            if name == "tau":                       # spot-check a few points
                idx = [60, 120, 160, 200, 260]
            elif name == "w":
                idx = list(range(4))
            else:
                idx = [None]
            for j in idx:
                up = [mx.array(a) for a in args]
                dn = [mx.array(a) for a in args]
                base = np.asarray(args[i]).copy()
                if j is None:
                    up[i] = mx.array(base + h, dtype=mx.float64)
                    dn[i] = mx.array(base - h, dtype=mx.float64)
                    gi = float(g)
                else:
                    bu, bd = base.copy(), base.copy()
                    bu[j] += h
                    bd[j] -= h
                    up[i] = mx.array(bu, dtype=mx.float64)
                    dn[i] = mx.array(bd, dtype=mx.float64)
                    gi = float(g[j])
                fd = (float(loss(*up)) - float(loss(*dn))) / (2 * h)
                assert abs(gi - fd) <= tol * max(abs(fd), 1.0), (name, j, gi, fd)


@pytest.mark.parametrize("dtype", [mx.float64, mx.float32], ids=["fp64", "fp32"])
def test_gradients_finite_on_a_sweep(dtype):
    """Centre crossings (b = 0), f = 0, theta at 0 and pi/2, grazes, an
    exposure across every contact: every gradient finite."""
    stream = mx.cpu if dtype == mx.float64 else mx.gpu
    tau = np.linspace(-0.13, 0.13, 257)
    cases = [(0.0, 0.1, 0.3, 0.0), (0.0, 0.1, 0.0, 0.5), (0.4, 0.1, 0.3, math.pi / 2),
             (1.05, 0.15, 0.5, 0.6), (0.9, 0.12, 0.2, 1.3), (0.3, 0.1, 1e-5, 0.2)]
    with mx.stream(stream):
        for b, r, f, th in cases:
            for extra in ({}, {"exp_time": EXP}):
                def loss(r_, f_, th_, b_):
                    return mx.sum(mp.flux_dev_from_tau(
                        mx.array(tau, dtype=dtype), P, A_RS, b_, r_,
                        limb_dark="hybrid5", u=W["hybrid5"], f=f_, theta=th_, **extra))
                g = mx.grad(loss, argnums=(0, 1, 2, 3))(
                    *(mx.array(v, dtype=dtype) for v in (r, f, th, b)))
                mx.eval(g)
                for gi in g:
                    assert np.isfinite(np.asarray(gi.astype(mx.float32))).all(), (b, f, th, extra)


# ---------------------------------------------------------------------------
# keywords and the domain
# ---------------------------------------------------------------------------

def test_keyword_errors():
    with pytest.raises(ValueError, match="hybrid law"):
        mp.flux_dev_from_tau(TAU, P, A_RS, 0.3, 0.1, 0.4, 0.2, f=0.2)
    with pytest.raises(ValueError, match="pass f= too"):
        mp.flux_dev_from_tau(TAU, P, A_RS, 0.3, 0.1, limb_dark="hybrid4",
                             u=W["hybrid4"], theta=0.3)
    with pytest.raises(ValueError, match="u1/u2"):
        mp.flux_dev_from_tau(TAU, P, A_RS, 0.3, 0.1, 0.4, 0.2,
                             limb_dark="hybrid4", f=0.2)
    with pytest.raises(ValueError, match="weights as u="):
        mp.flux_dev_from_tau(TAU, P, A_RS, 0.3, 0.1, limb_dark="hybrid4", f=0.2)


def test_domain():
    with pytest.raises(ValueError, match="r_eff"):
        flux(TAU, 0.3, 0.6, f=0.5)
    with pytest.raises(ValueError, match="flattening"):
        flux(TAU, 0.3, 0.1, f=1.0)
    # array arguments are never read back (mx.compile); NaN outside instead
    got = flux(TAU, 0.3, mx.array(0.6, dtype=mx.float64),
               f=mx.array(0.5, dtype=mx.float64))
    assert np.isnan(got[len(TAU) // 2])


def test_theta_defaults_to_zero():
    a = flux(TAU, 0.3, 0.1, f=0.3)
    b = flux(TAU, 0.3, 0.1, f=0.3, theta=0.0)
    assert np.array_equal(a, b)
