"""TransitModel.light_curves: many parameter sets, one dispatch.

``light_curve`` is one-set-at-a-time for batman parity, and looping it is
the failure mode docs/sampler-integration.md warns about. These tests pin
that the batched form agrees with the loop to round-off in every mode the
frontend supports, and that mixed circular/eccentric batches work (the
transit-anchored orbit degenerates exactly at e = 0).
"""

import numpy as np
import mlx.core as mx
import pytest

import metalplanet

T = np.linspace(-0.13, 0.13, 201)
RNG = np.random.default_rng(4)


def _p(**kw):
    p = metalplanet.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = 0.0, 3.456, 0.1, 8.8, 87.07
    p.ecc, p.w, p.u, p.limb_dark = 0.0, 90.0, [0.4, 0.25], "quadratic"
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def _sets(n, **base):
    return [_p(rp=0.1 + 0.02 * RNG.standard_normal(),
               inc=87.0 + 0.5 * RNG.standard_normal(),
               ecc=abs(0.2 * RNG.standard_normal()),
               w=RNG.uniform(0, 360), **base) for _ in range(n)]


MODES = [
    ("plain", {}, {}),
    ("supersample", dict(exp_time=0.02, supersample_factor=7), {}),
    ("contact", dict(exp_time=0.02, integration="contact", n_gl=7), {}),
    ("polynomial", {}, dict(u=[0.3, 0.2, 0.1], limb_dark="polynomial")),
    ("poly+contact", dict(exp_time=0.02, integration="contact", n_gl=5),
     dict(u=[0.3, 0.2, 0.1], limb_dark="polynomial")),
    ("linear", {}, dict(u=[0.4], limb_dark="linear")),
    ("uniform", {}, dict(u=[], limb_dark="uniform")),
]


@pytest.mark.parametrize("label,kw,base", MODES, ids=[m[0] for m in MODES])
def test_matches_the_looped_api(label, kw, base):
    m = metalplanet.TransitModel(_p(**base), T, **kw)
    sets = _sets(5, **base)
    batched = m.light_curves(sets)
    looped = np.stack([m.light_curve(p) for p in sets])
    assert batched.shape == (5, T.size)
    assert np.abs(batched - looped).max() < 1e-14


def test_mixed_circular_and_eccentric():
    """e = 0 and e > 0 in one batch: the anchored orbit covers both."""
    m = metalplanet.TransitModel(_p(), T)
    sets = [_p(ecc=0.0), _p(ecc=0.3, w=63.0), _p(ecc=0.0, rp=0.12),
            _p(ecc=0.7, w=200.0)]
    batched = m.light_curves(sets)
    looped = np.stack([m.light_curve(p) for p in sets])
    assert np.abs(batched - looped).max() < 1e-14


def test_array_valued_transitparams():
    m = metalplanet.TransitModel(_p(), T)
    pa = _p()
    pa.rp = np.array([0.08, 0.1, 0.12])
    pa.inc = np.array([86.5, 87.07, 88.0])
    pa.ecc = np.array([0.0, 0.3, 0.5])
    pa.w = np.array([90.0, 63.0, 120.0])
    batched = m.light_curves(pa)
    looped = np.stack([m.light_curve(
        _p(rp=r, inc=i, ecc=e, w=w))
        for r, i, e, w in zip(pa.rp, pa.inc, pa.ecc, pa.w)])
    assert np.abs(batched - looped).max() < 1e-14


def test_single_set_still_works():
    m = metalplanet.TransitModel(_p(), T)
    assert m.light_curves([_p()]).shape == (1, T.size)


def test_fp32_gpu_path():
    m = metalplanet.TransitModel(_p(), T, dtype=mx.float32)
    sets = _sets(16)
    batched = m.light_curves(sets)
    looped = np.stack([m.light_curve(p) for p in sets])
    assert np.abs(batched - looped).max() < 5e-6


def test_rejects_empty_and_mismatched_law():
    m = metalplanet.TransitModel(_p(), T)
    with pytest.raises(ValueError, match="no parameter sets"):
        m.light_curves([])
    with pytest.raises(ValueError, match="limb-darkening"):
        m.light_curves([_p(u=[0.4], limb_dark="linear")])


def test_batching_is_much_faster_than_looping():
    """The whole point. Not a tight timing assertion — just that the
    batched call is not merely a loop in disguise."""
    import time
    m = metalplanet.TransitModel(_p(), T, dtype=mx.float32)
    sets = _sets(400)
    m.light_curves(sets)                      # warm
    t0 = time.perf_counter()
    m.light_curves(sets)
    t_batch = time.perf_counter() - t0
    m.light_curve(sets[0])
    t0 = time.perf_counter()
    for p in sets[:40]:
        m.light_curve(p)
    t_loop = (time.perf_counter() - t0) / 40 * len(sets)
    assert t_batch < 0.25 * t_loop


def test_array_form_infers_the_batch_size_from_any_attribute():
    """Regression: the size was taken from t0 alone, so an array rp with
    a scalar t0 raised a broadcast error."""
    m = metalplanet.TransitModel(_p(), T)
    pa = _p()
    pa.rp = np.array([0.09, 0.1, 0.11])        # t0/per/a/inc stay scalars
    assert m.light_curves(pa).shape == (3, T.size)
    with pytest.raises(ValueError, match="inconsistent"):
        bad = _p()
        bad.rp = np.array([0.09, 0.1, 0.11])
        bad.inc = np.array([87.0, 88.0])
        m.light_curves(bad)
