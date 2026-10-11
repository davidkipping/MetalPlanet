"""Oblate planets, Stage 4: TransitParams.f / .theta through TransitModel.

The model forms time since each point's own mid-transit and calls
flux_dev_from_tau(f=, theta=) (the oblate kernels on an fp32 GPU model,
the exact graph otherwise), with its own exposure rule. Pinned against
SquishierPlanet's model.light_curve in absolute times, against the
spherical model at f = 0, across precisions and surfaces, and in its
gradients.
"""

import math
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx
import pytest

import metalplanet as mp
from metalplanet import metal as M

_SP = Path(__file__).resolve().parents[2] / "SquishierPlanet"
if _SP.is_dir() and str(_SP) not in sys.path:
    sys.path.insert(0, str(_SP))
try:
    from squishierplanet import laws as spl
    from squishierplanet.model import light_curve as sp_light_curve
except Exception:
    spl = None
needs_sp = pytest.mark.skipif(spl is None, reason="squishierplanet not importable")
KERNEL = M.metal_available() and M._gpu_stream_active()

W5 = [0.2, 0.2, 0.1, 0.1, 0.1]
T0, PER = 0.3, 3.45
T = T0 + 2 * PER + np.linspace(-0.15, 0.15, 241)
EXP = 0.02


def P(**kw):
    p = mp.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = T0, PER, 0.1, 8.8, 88.0, 0.0, 90.0
    p.limb_dark, p.u = "hybrid5", list(W5)
    p.f, p.theta = 0.3, 35.0
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def spherical(**kw):
    p = P(**kw)
    p.f = p.theta = None
    return p


@needs_sp
@pytest.mark.parametrize("ecc,w", [(0.0, 90.0), (0.3, 40.0), (0.6, -70.0)])
def test_matches_squishierplanet(ecc, w):
    p = P(ecc=ecc, w=w)
    got = mp.TransitModel(p, T).light_curve(p)
    ref = sp_light_curve(T, spl.hybrid("hybrid5", W5), r_eff=0.1, f=0.3,
                         theta=math.radians(35.0), period=PER, a=8.8,
                         i=math.radians(88.0), t0=T0, e=ecc, omega=math.radians(w))
    assert 1.0 - ref.min() > 1e-3
    assert np.abs(got - ref).max() < 1e-13


def test_theta_is_in_degrees_and_defaults_to_zero():
    p = P(theta=35.0)
    got = mp.TransitModel(p, T).light_curve(p)
    tau = (T - T0) - PER * np.round((T - T0) / PER)
    ref = 1.0 + np.asarray(mp.flux_dev_from_tau(
        tau, PER, 8.8, 8.8 * math.cos(math.radians(88.0)), 0.1, limb_dark="hybrid5",
        u=W5, f=0.3, theta=math.radians(35.0)))
    assert np.abs(got - ref).max() < 1e-15
    a, b = P(theta=None), P(theta=0.0)
    assert np.array_equal(mp.TransitModel(a, T).light_curve(a),
                          mp.TransitModel(b, T).light_curve(b))


@pytest.mark.parametrize("kw", [{}, {"supersample_factor": 7, "exp_time": EXP},
                                {"integration": "contact", "exp_time": EXP}],
                         ids=["none", "supersample", "contact"])
def test_f_zero_is_the_spherical_model(kw):
    o, s = P(f=0.0), spherical()
    got = mp.TransitModel(o, T, **kw).light_curve(o)
    ref = mp.TransitModel(s, T, **kw).light_curve(s)
    assert np.abs(got - ref).max() < 1e-14


@pytest.mark.parametrize("kw", [{}, {"supersample_factor": 7, "exp_time": EXP},
                                {"integration": "contact", "exp_time": EXP}],
                         ids=["none", "supersample", "contact"])
@pytest.mark.parametrize("ecc", [0.0, 0.3])
def test_fp32_model_matches_fp64(kw, ecc):
    p = P(ecc=ecc, w=40.0)
    d64 = mp.TransitModel(p, T, **kw).light_curve(p)
    d32 = mp.TransitModel(p, T, dtype=mx.float32, **kw).light_curve(p)
    assert np.abs(d32 - d64).max() < 3e-7


@pytest.mark.skipif(not KERNEL, reason="needs Metal and the GPU stream")
def test_fp32_model_takes_the_oblate_kernel():
    from metalplanet import metal_oblate as MO
    seen = []
    orig = MO._make_core

    def spy(*a, **k):
        seen.append(1)
        return orig(*a, **k)
    MO._make_core = spy
    try:
        p = P(rp=0.1013)               # a fresh value, not a cached trace
        mp.TransitModel(p, T, dtype=mx.float32).light_curve(p)
    finally:
        MO._make_core = orig
    assert seen


@pytest.mark.parametrize("kw", [{}, {"integration": "contact", "exp_time": EXP}],
                         ids=["none", "contact"])
def test_light_curves_equal_the_loop(kw):
    m = mp.TransitModel(P(), T, **kw)
    sets = [P(f=0.1, theta=0.0), P(f=0.4, theta=80.0, ecc=0.2, w=10.0),
            P(rp=0.12, f=0.2, theta=150.0, u=[0.3, 0.1, 0.1, 0.05, 0.05])]
    loop = np.stack([m.light_curve(q) for q in sets])
    assert np.abs(m.light_curves(sets) - loop).max() < 1e-15
    pa = P()
    pa.f, pa.theta = np.array([0.1, 0.4, 0.2]), np.array([0.0, 80.0, 150.0])
    pa.rp = np.array([0.1, 0.1, 0.12])
    arr = m.light_curves(pa)
    ref = np.stack([m.light_curve(P(f=f, theta=t, rp=r))
                    for f, t, r in zip(pa.f, pa.theta, pa.rp)])
    assert np.abs(arr - ref).max() < 1e-15


def test_light_curve_mx_equals_light_curve_and_differentiates():
    p = P(ecc=0.2, w=30.0)
    ct = np.random.default_rng(0).normal(size=T.shape)
    with mx.stream(mx.cpu):
        m = mp.TransitModel(p, T)
        assert np.abs(np.asarray(m.light_curve_mx(p)) - m.light_curve(p)).max() < 1e-15
        names = ("f", "theta", "rp", "t0", "inc", "ecc", "w")
        x = [0.3, 35.0, 0.1, T0, 88.0, 0.2, 30.0]

        def loss(*v):
            q = P()
            for nm, val in zip(names, v):
                setattr(q, nm, val)
            q.u = [v[-1]] + W5[1:]
            return mx.sum(mx.array(ct) * m.light_curve_mx(q))
        args = [mx.array(v, dtype=mx.float64) for v in x + [W5[0]]]
        g = mx.grad(loss, argnums=tuple(range(8)))(*args)
        for i in range(8):
            def fd(h):
                up = [mx.array(a) for a in args]
                dn = [mx.array(a) for a in args]
                up[i] = args[i] + h
                dn[i] = args[i] - h
                return (float(loss(*up)) - float(loss(*dn))) / (2 * h)
            h = 1e-4 * max(abs(float(args[i])), 0.01)
            ref = (4 * fd(h / 2) - fd(h)) / 3
            assert abs(float(g[i]) - ref) <= 2e-4 * max(abs(ref), 1e-6), i


def test_errors():
    with pytest.raises(ValueError, match="hybrid law"):
        mp.TransitModel(P(limb_dark="quadratic", u=[0.4, 0.2]), T)
    with pytest.raises(ValueError, match="set params.f too"):
        q = spherical()
        q.theta = 30.0
        mp.TransitModel(q, T)
    with pytest.raises(ValueError, match="primary-transit only"):
        mp.TransitModel(P(fp=1e-4), T, transittype="secondary")
    m = mp.TransitModel(P(), T)
    with pytest.raises(ValueError, match="build a new TransitModel"):
        m.light_curve(spherical())
    ms = mp.TransitModel(spherical(), T)
    with pytest.raises(ValueError, match="build a new TransitModel"):
        ms.light_curve(P())
    with pytest.raises(ValueError, match="r_eff"):
        m.light_curve(P(rp=0.6, f=0.5))


def test_out_of_domain_array_is_nan():
    with mx.stream(mx.cpu):
        m = mp.TransitModel(P(), T)
        out = np.asarray(m.light_curve_mx(P(rp=mx.array(0.6, dtype=mx.float64),
                                            f=mx.array(0.5, dtype=mx.float64))))
    assert np.isnan(out[len(T) // 2])


def test_spherical_models_unchanged_by_the_new_fields():
    """A TransitParams built before 0.13 (no f / theta attributes) still
    works, and a spherical model's batch keys are what they were."""
    p = spherical()
    del p.f, p.theta
    m = mp.TransitModel(p, T)
    assert m._batch_keys == mp.TransitModel._BATCH_KEYS
    ref = mp.TransitModel(spherical(), T).light_curve(spherical())
    assert np.array_equal(m.light_curve(p), ref)
