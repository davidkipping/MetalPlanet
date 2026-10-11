"""Oblate planets, Stage 1: the reference MLX graph (metalplanet/oblate.py).

The truth is SquishierPlanet's analytic reference (numpy; importorskip) and
its independent quadrature oracle; the contract is its brief's
(docs/upstream/metalplanet_oblate_prompt.md there): <= 1e-12 of each
column's unocculted flux in fp64 over its stress sets; f -> 0 continuous
with the spherical path; the ld_basis identity; gradients against finite
differences, finite everywhere.
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx
import pytest

import metalplanet as mp
from metalplanet import oblate as O
from metalplanet.hybrid import HYBRID2, HybridLaw, LAWS, hybrid_norms, shape_cols

_SP = Path(__file__).resolve().parents[2] / "SquishierPlanet"
if _SP.is_dir() and str(_SP) not in sys.path:
    sys.path.insert(0, str(_SP))
try:
    import squishierplanet as sp
    from squishierplanet import laws as spl
    from squishierplanet.assembly import basis_moments
    from squishierplanet.oracle import occulted_flux
    from squishierplanet.poles import pole_moments
    _spec = importlib.util.spec_from_file_location("sp_conftest", _SP / "tests" / "conftest.py")
    spc = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(spc)
except Exception:                                           # optional cross-check
    sp = None

needs_sp = pytest.mark.skipif(sp is None, reason="squishierplanet not importable")

EPS = tuple(sorted({e for name in ("hybrid2", "hybrid4", "hybrid5") for e in LAWS[name].eps}))
ALLPOLES = HybridLaw("allpoles", EPS, ((1.0, -1.0, 0.0),))
NORM = np.concatenate([[math.pi, math.pi / 2, math.pi / 3],
                       math.pi / (np.array(EPS) * (1.0 + np.array(EPS)))])
W = {"hybrid2": [0.3, 0.2], "hybrid4": [0.2, 0.2, 0.1, 0.1],
     "hybrid5": [0.2, 0.2, 0.1, 0.1, 0.1]}


def f64(v):
    return mx.array(np.asarray(v, dtype=np.float64), dtype=mx.float64)


def cpu_np(fn):
    """Run fn on the CPU stream and return numpy float64."""
    with mx.stream(mx.cpu):
        out = fn()
        mx.eval(out)
        return np.asarray(out, dtype=np.float64)


def generators(x0, y0, r, f, dtype=mx.float64):
    stream = mx.cpu if dtype == mx.float64 else mx.gpu
    with mx.stream(stream):
        g = O.generator_cols_oblate(*(mx.array(np.asarray(v, np.float64), dtype=dtype)
                                      for v in (x0, y0, r, f)), ALLPOLES)
        mx.eval(*g)
        return np.stack([np.asarray(v.astype(mx.float32) if dtype == mx.float32 else v,
                                    dtype=np.float64) for v in g], -1)


def reference(x0, y0, A, B):
    return -np.concatenate([basis_moments(x0, y0, A, B, 2),
                            pole_moments(x0, y0, A, B, np.array(EPS))], -1)


def canon(c):
    """A configuration with its long axis along x (ours) from one with
    either axis order (SquishierPlanet's generators)."""
    x0, y0, a, b = c
    return (x0, y0, a, b) if a >= b else (y0, -x0, b, a)


def planets(C):
    """Keep the planetary domain: A < 1, A^2/B <= 1, f in (0, 0.5]."""
    x0, y0, A, B = C
    m = (A < 1.0) & (A * A / B <= 1.0) & (B / A >= 0.5) & (B / A < 1.0 - 1e-9)
    return x0[m], y0[m], A[m], B[m]


def stress_sets():
    sets = {}
    for k, cls in enumerate(("two_int", "inside", "disjoint", "origin_in_partial")):
        sets[cls] = [canon(c) for c in spc.make_configs(700 + k, cls, 150)]
    lt = []
    for gap in (0.0, 1e-12, 1e-9, 1e-6, 1e-3, -1e-9, -1e-6, -1e-3):
        for kind in ("external", "internal"):
            for a, b, ph in ((0.12, 0.08, 1.1), (0.05, 0.035, 2.0), (0.3, 0.2, 4.0)):
                lt.append(canon(spc.tangent_config(a, b, ph, kind, gap)))
    sets["limb_tangent"] = lt
    pt = []
    for e in EPS:
        rho = math.sqrt(1.0 + e)
        for gap in (0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, -1e-12, -1e-9):
            for a, b, ph in ((0.12, 0.08, 1.1), (0.3, 0.2, 2.5), (0.02, 0.014, 4.0)):
                x0, y0, A, B = spc.tangent_config(a / rho, b / rho, ph, "internal", gap)
                pt.append(canon((x0 * rho, y0 * rho, A * rho, B * rho)))
    sets["pole_tangent"] = pt
    return sets


# ---------------------------------------------------------------------------
# geometry and domain
# ---------------------------------------------------------------------------

def test_axes_and_domain():
    A, B = O.axes(0.1, 0.3)
    assert abs(A * B - 0.01) < 1e-17 and abs(B / A - 0.7) < 1e-15
    assert O.max_r_eff(0.5) == pytest.approx(0.5 ** 1.5)
    with pytest.raises(ValueError, match="r_eff"):
        mp.flux_dev_oblate(f64([0.5]), f64([0.0]), 0.6, 0.5, W["hybrid4"], "hybrid4")
    for bad_f in (-0.1, 1.0):
        with pytest.raises(ValueError, match="flattening"):
            mp.flux_dev_oblate(f64([0.5]), f64([0.0]), 0.1, bad_f, W["hybrid4"], "hybrid4")


@needs_sp
def test_principal_frame_is_squishierplanets():
    from squishierplanet.model import to_principal
    rng = np.random.default_rng(0)
    X, Y, th = rng.normal(size=50), rng.normal(size=50), rng.uniform(0, np.pi, 50)
    ours = O.principal_frame(X, Y, th)
    ref = to_principal(X, Y, th)
    assert np.allclose(ours, ref, atol=1e-15, rtol=0)


# ---------------------------------------------------------------------------
# against SquishierPlanet's analytic reference
# ---------------------------------------------------------------------------

@needs_sp
@pytest.mark.parametrize("name", ["two_int", "inside", "disjoint", "origin_in_partial",
                                  "limb_tangent", "pole_tangent"])
@pytest.mark.parametrize("dtype", [mx.float64, mx.float32], ids=["fp64", "fp32"])
def test_generators_match_squishierplanet(name, dtype):
    """Every generator column, every topology class and tangency set
    (planetary domain), to 1e-12 (fp64) / 1e-6 (fp32) of its unocculted
    flux."""
    x0, y0, A, B = planets(np.array(stress_sets()[name]).T)
    r, f = np.sqrt(A * B), 1.0 - B / A
    got = generators(x0, y0, r, f, dtype)
    err = np.abs(got - reference(x0, y0, A, B)) / NORM
    assert np.isfinite(got).all()
    assert err.max() < (1e-12 if dtype == mx.float64 else 1e-6)


@needs_sp
@pytest.mark.parametrize("law", ["hybrid2", "hybrid4", "hybrid5"])
def test_law_light_curve_matches_squishierplanet(law):
    """flux_dev_oblate along tilted transit chords vs squishierplanet.flux,
    sky frame and theta convention included."""
    if law == "hybrid2" and float(spl.HYBRID2_EPS) != LAWS["hybrid2"].eps[0]:
        pytest.skip("this SquishierPlanet checkout's hybrid2 pole is not MetalPlanet's")
    L = spl.hybrid(law, W[law])
    for r, f, th, b in ((0.1, 0.1, 0.4, 0.3), (0.2, 0.3, 1.0, 0.6), (0.05, 0.5, 2.5, 0.0)):
        A, B = sp.axes_from_reff(r, f)
        X = np.linspace(-(1 + A) * 1.02, (1 + A) * 1.02, 301)
        Y = np.full_like(X, b)
        x0, y0 = O.principal_frame(X, Y, th)
        got = cpu_np(lambda: mp.flux_dev_oblate(f64(x0), f64(y0), r, f, W[law], law))
        ref = sp.flux(L, X, Y, A, B, th) - 1.0
        assert np.abs(got - ref).max() < 1e-12, (r, f, th, b)


@needs_sp
def test_generators_match_the_quadrature_oracle():
    """An independent check: SquishierPlanet's polar-ray quadrature."""
    prims = [lambda rr, n=n: -(1.0 - rr * rr) ** (n + 1) / (2.0 * (n + 1)) for n in range(3)]
    prims += [lambda rr, e=e: 1.0 / (2.0 * (1.0 - rr * rr + e)) for e in EPS]
    cfg = [(0.95, 0.1, 0.12, 0.09), (-0.4, 0.75, 0.2, 0.12), (0.3, 0.2, 0.1, 0.08),
           (0.0, 0.99, 0.15, 0.1), (0.7, -0.6, 0.05, 0.04)]
    x0, y0, A, B = np.array(cfg).T
    got = generators(x0, y0, np.sqrt(A * B), 1.0 - B / A)
    for i, c in enumerate(cfg):
        for k, prim in enumerate(prims):
            ref = -occulted_flux(*c, prim)
            assert abs(got[i, k] - ref) / NORM[k] < 1e-11, (c, k)


@needs_sp
def test_generators_match_mpmath_quadrature():
    """Spot checks at 30 digits (mpmath.quad through the same oracle):
    a partial and an inside configuration, the even column and the
    innermost and outermost poles."""
    pytest.importorskip("mpmath")
    cols = [(0, lambda rr: -(1.0 - rr * rr) / 2.0)]
    for k in (0, len(EPS) - 1):
        cols.append((3 + k, lambda rr, e=EPS[k]: 1.0 / (2.0 * (1.0 - rr * rr + e))))
    cfg = [(0.95, 0.1, 0.12, 0.09), (0.3, 0.2, 0.1, 0.08)]
    x0, y0, A, B = np.array(cfg).T
    got = generators(x0, y0, np.sqrt(A * B), 1.0 - B / A)
    for i, c in enumerate(cfg):
        for k, prim in cols:
            ref = -occulted_flux(*c, prim, use_mpmath=True)
            assert abs(got[i, k] - ref) / NORM[k] < 1e-12, (c, k)


@needs_sp
@pytest.mark.parametrize("gap", [1e-3, 1e-6, 1e-9, -1e-9, -1e-6, -1e-3])
def test_inside_and_partial_meet_at_the_limb(gap):
    """Where |centre| + A crosses 1 the inside closed forms hand over to the
    partial-regime arcs: both sides agree with the reference."""
    rng = np.random.default_rng(int(abs(gap) * 1e12) % 1000)
    cfg = []
    for _ in range(20):
        r, f = rng.uniform(0.02, 0.3), rng.uniform(0.01, 0.5)
        f = min(f, 1 - r ** (2 / 3) - 1e-3)
        A, B = r / math.sqrt(1 - f), r * math.sqrt(1 - f)
        t = rng.uniform(0, 2 * np.pi)
        d = 1.0 - A - gap                        # |centre| + A = 1 - gap
        cfg.append((d * math.cos(t), d * math.sin(t), A, B))
    x0, y0, A, B = np.array(cfg).T
    got = generators(x0, y0, np.sqrt(A * B), 1.0 - B / A)
    assert (np.abs(got - reference(x0, y0, A, B)) / NORM).max() < 1e-12


# ---------------------------------------------------------------------------
# f -> 0, the ld_basis contract, zero outside
# ---------------------------------------------------------------------------

def test_small_f_is_the_spherical_path():
    """Below f_sw the spherical closed forms (exactly the spherical path's
    values); above it the oblate path, continuous across the switch."""
    X = np.linspace(0.0, 1.12, 300)
    x0, y0 = O.principal_frame(X, 0.3, 0.7)
    sph = cpu_np(lambda: shape_cols(f64(np.hypot(x0, y0)), 0.1, "hybrid5"))
    for f in (0.0, 1e-12, 0.99 * O._F_SW[mx.float64]):
        got = cpu_np(lambda: O.shape_cols_oblate(f64(x0), f64(y0), 0.1, f, "hybrid5"))
        assert np.abs(got - sph).max() < 1e-15
    above = cpu_np(lambda: O.shape_cols_oblate(f64(x0), f64(y0), 0.1,
                                               1.01 * O._F_SW[mx.float64], "hybrid5"))
    assert np.abs(above - sph).max() < 1e-11          # the flattening's own ~4e-3 f


def test_ld_basis_identity():
    rng = np.random.default_rng(3)
    X = np.linspace(-1.2, 1.2, 200)
    x0, y0 = O.principal_frame(X, 0.2, 1.1)
    for law in ("hybrid2", "hybrid4", "hybrid5"):
        Bc = cpu_np(lambda: O.shape_cols_oblate(f64(x0), f64(y0), 0.12, 0.25, law))
        w = rng.dirichlet(np.ones(LAWS[law].n_w + 1))[:-1]
        c = np.concatenate([[1.0], -w])
        got = cpu_np(lambda: mp.flux_dev_oblate(f64(x0), f64(y0), 0.12, 0.25, list(w), law))
        assert np.abs(got - (Bc @ c) / (hybrid_norms(law) @ c)).max() < 1e-15


def test_exactly_zero_out_of_transit():
    X = np.linspace(1.2, 3.0, 50)
    x0, y0 = O.principal_frame(X, 0.1, 0.4)
    got = cpu_np(lambda: O.shape_cols_oblate(f64(x0), f64(y0), 0.1, 0.3, "hybrid5"))
    assert np.array_equal(got, np.zeros_like(got))


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------

POINTS = [(0.3, 0.2, 0.1, 0.2), (0.95, 0.1, 0.1, 0.3), (-0.6, 0.75, 0.15, 0.25),
          (0.02, 0.01, 0.12, 0.1)]


@pytest.mark.parametrize("pt", POINTS, ids=["inside", "partial-a", "partial-b", "near-centre"])
def test_gradients_match_finite_differences(pt):
    law, w = "hybrid5", W["hybrid5"]

    def fn(x, y, r, f, ww):
        return mx.sum(mp.flux_dev_oblate(x, y, r, f, ww, law))

    with mx.stream(mx.cpu):
        g = mx.grad(fn, argnums=(0, 1, 2, 3, 4))(*map(f64, pt), f64(w))
        mx.eval(g)
        h = 1e-6
        for i in range(4):
            up, dn = list(pt), list(pt)
            up[i] += h
            dn[i] -= h
            fd = (float(fn(*map(f64, up), f64(w))) - float(fn(*map(f64, dn), f64(w)))) / (2 * h)
            assert abs(float(g[i]) - fd) <= 1e-6 * max(abs(fd), 1e-3), (i, float(g[i]), fd)
        for j in range(5):
            wu, wd = list(w), list(w)
            wu[j] += h
            wd[j] -= h
            fd = (float(fn(*map(f64, pt), f64(wu))) - float(fn(*map(f64, pt), f64(wd)))) / (2 * h)
            assert abs(float(np.asarray(g[4])[j]) - fd) <= 1e-6 * max(abs(fd), 1e-3), j


@needs_sp
@pytest.mark.parametrize("dtype", [mx.float64,
                                   pytest.param(mx.float32, marks=pytest.mark.slow)],
                         ids=["fp64", "fp32"])
def test_gradients_finite_on_a_boundary_sweep(dtype):
    """Exact tangencies, a centred planet, f = 0 and either side of f_sw,
    the inside/partial seam, out of transit: every gradient finite."""
    cfgs = []
    for gap in (0.0, 1e-9, -1e-9, 1e-6):
        for kind in ("external", "internal"):
            x0, y0, a, b = canon(spc.tangent_config(0.12, 0.08, 1.1, kind, gap))
            cfgs.append((x0, y0, math.sqrt(a * b), 1 - b / a))
    sw = O._F_SW[dtype]
    cfgs += [(0.0, 0.0, 0.1, 0.2), (0.3, 0.0, 0.1, 0.0), (0.3, 0.0, 0.1, 0.99 * sw),
             (0.3, 0.0, 0.1, 1.01 * sw), (1.2, 0.1, 0.1, 0.2),
             (1.0 - 0.1 / math.sqrt(0.8), 0.0, 0.1, 0.2)]
    stream = mx.cpu if dtype == mx.float64 else mx.gpu
    with mx.stream(stream):
        for c in cfgs:
            def fn(x, y, r, f):
                return mx.sum(mp.flux_dev_oblate(x, y, r, f, W["hybrid5"], "hybrid5"))
            g = mx.grad(fn, argnums=(0, 1, 2, 3))(*(mx.array(v, dtype=dtype) for v in c))
            mx.eval(g)
            for gi in g:
                assert np.isfinite(np.asarray(gi.astype(mx.float32))).all(), c


# ---------------------------------------------------------------------------
# calling forms
# ---------------------------------------------------------------------------

def test_weight_forms_agree():
    X = np.linspace(-1.1, 1.1, 64)
    x0, y0 = O.principal_frame(X, 0.3, 0.5)
    w = W["hybrid4"]
    host = cpu_np(lambda: mp.flux_dev_oblate(f64(x0), f64(y0), 0.1, 0.2, w, "hybrid4"))
    traced = cpu_np(lambda: mp.flux_dev_oblate(f64(x0), f64(y0), 0.1, 0.2, f64(w), "hybrid4"))
    rows = cpu_np(lambda: mp.flux_dev_oblate(f64(np.stack([x0, x0])), f64(np.stack([y0, y0])),
                                             0.1, 0.2, f64([w, w]), "hybrid4"))
    assert np.abs(host - traced).max() < 1e-16
    assert np.abs(rows - host[None]).max() < 1e-16


def test_data_contract_and_keyword_calls():
    """numpy float64 data is fp64 (on the CPU stream, put there for us);
    float32 stays float32; the data may be passed by name."""
    X = np.linspace(-1.1, 1.1, 32)
    x0, y0 = O.principal_frame(X, 0.3, 0.5)
    a = mp.flux_dev_oblate(x0, y0, 0.1, 0.2, W["hybrid2"], "hybrid2")
    assert a.dtype == mx.float64
    b = mp.flux_dev_oblate(x0=x0, y0=y0, r=0.1, f=0.2, w=W["hybrid2"], law="hybrid2")
    with mx.stream(mx.cpu):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    c = mp.flux_dev_oblate(x0.astype(np.float32), y0.astype(np.float32), 0.1, 0.2,
                           W["hybrid2"], "hybrid2")
    assert c.dtype == mx.float32


def test_a_custom_law_object():
    law = HybridLaw("custom2", (0.15,), HYBRID2.shapes)
    X = np.linspace(-1.1, 1.1, 32)
    x0, y0 = O.principal_frame(X, 0.3, 0.5)
    B = cpu_np(lambda: O.shape_cols_oblate(f64(x0), f64(y0), 0.1, 0.2, law))
    assert B.shape == (32, 3) and np.isfinite(B).all() and B.min() < 0


def test_fp64_graph_compiles():
    X = np.linspace(-1.1, 1.1, 64)
    x0, y0 = O.principal_frame(X, 0.3, 0.5)
    with mx.stream(mx.cpu):
        xa, ya = f64(x0), f64(y0)
        eager = mp.flux_dev_oblate(xa, ya, 0.1, 0.2, W["hybrid5"], "hybrid5")
        comp = mx.compile(lambda x, y: mp.flux_dev_oblate(x, y, 0.1, 0.2, W["hybrid5"], "hybrid5"))
        got = comp(xa, ya)
        assert np.abs(np.asarray(got) - np.asarray(eager)).max() < 1e-15


@pytest.mark.parametrize("dtype", [mx.float64, mx.float32], ids=["fp64", "fp32"])
def test_theta_gradient_through_the_sky_frame(dtype):
    """theta enters through principal_frame: autodiff vs central FD (fp64),
    and finite at theta = 0 and pi/2 along a whole chord (both dtypes)."""
    X = np.linspace(-1.15, 1.15, 81)
    stream = mx.cpu if dtype == mx.float64 else mx.gpu

    def fn(th, b):
        x0, y0 = O.principal_frame(mx.array(X, dtype=dtype), b, th)
        return mx.sum(mp.flux_dev_oblate(x0, y0, 0.1, 0.25, W["hybrid4"], "hybrid4"))

    with mx.stream(stream):
        for th in (0.0, 0.4, math.pi / 2):
            g = mx.grad(fn)(mx.array(th, dtype=dtype), mx.array(0.3, dtype=dtype))
            mx.eval(g)
            assert np.isfinite(float(g.astype(mx.float32)))
            if dtype == mx.float64 and th == 0.4:
                h = 1e-6
                fd = (float(fn(mx.array(th + h, dtype=dtype), mx.array(0.3, dtype=dtype)))
                      - float(fn(mx.array(th - h, dtype=dtype), mx.array(0.3, dtype=dtype)))) / (2 * h)
                assert abs(float(g) - fd) <= 1e-6 * max(abs(fd), 1e-3)
