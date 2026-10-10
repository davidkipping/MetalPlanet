"""Every public entry point that takes z or tau, under every calling form
(0.10.7).

Six review rounds on the 0.10 frontend kept finding holes one calling
form at a time -- a keyword data argument, a numpy float64 array, an fp64
parameter, a Python-float coefficient under mx.grad. This file holds the
whole matrix, so a change to a shared convention is checked against every
entry point and every form at once:

* call forms: positional data, data by keyword, all keywords;
* data containers: mx fp32/fp64/fp16/int32, numpy fp64/fp32/int, list;
* parameter dtypes: fp64 parameters with fp32 data and the reverse, on
  the default (GPU) stream and the CPU stream;
* layouts: (m,) and (n, m) data;
* transforms: mx.grad (fp32 on the default stream; fp64 with the grad
  call on the CPU stream, the MLX rule), mx.compile.

The contract (metalplanet/dtypes.py): float32 data stays float32 and
takes the kernels on the GPU; any other data is float64 and takes the
exact graph on the CPU stream. Kernel entry points cast parameters to the
data's dtype; graph functions keep MLX broadcasting and promotion.
"""

import contextlib

import numpy as np
import mlx.core as mx
import pytest

import metalplanet as mp
from metalplanet import metal as M
from metalplanet import metal_hybrid as MH
from metalplanet.metal import metal_available
from metalplanet.metal_hybrid import flux_dev_metal_hybrid

KERNEL = metal_available() and M._gpu_stream_active()

Z = np.linspace(0.0, 1.25, 64)
TAU = np.linspace(-0.13, 0.13, 64)
W4 = [0.2, 0.2, 0.1, 0.1]
GEO = [("period", 3.45), ("a", 8.8), ("b", 0.3), ("r", 0.1)]
QUAD = [("u1", 0.4), ("u2", 0.25)]
HYB = {"limb_dark": "hybrid4", "u": W4}

# name -> (fn, data name, base data, positional params, keywords, family)
# family: the kernel dispatch point the fp32 GPU call must reach, or None
# for a graph function.
KERNEL_ENTRIES = {
    "z/quad": (mp.flux_dev_metal, "z", Z, [("r", 0.1)] + QUAD, {}, "zq"),
    "z/ld_basis": (mp.flux_dev_metal, "z", Z, [("r", 0.1)],
                   {"ld_basis": True}, "zb"),
    "z/hybrid": (mp.flux_dev_metal, "z", Z, [("r", 0.1)], HYB, "zh"),
    "z/hybrid_basis": (mp.flux_dev_metal, "z", Z, [("r", 0.1)],
                       {"limb_dark": "hybrid4", "ld_basis": True}, "zh"),
    "z-hybrid-entry": (flux_dev_metal_hybrid, "z", Z,
                       [("r", 0.1), ("law", "hybrid4"), ("u", W4)], {}, "zh"),
    "tau/quad": (mp.flux_dev_from_tau, "tau", TAU, GEO + QUAD, {}, "tq"),
    "tau/quad_contact": (mp.flux_dev_from_tau, "tau", TAU, GEO + QUAD,
                         {"exp_time": 0.02}, "tq"),
    "tau/ld_basis": (mp.flux_dev_from_tau, "tau", TAU, GEO,
                     {"ld_basis": True}, "tq"),
    "tau/quad_ecc": (mp.flux_dev_from_tau, "tau", TAU, GEO + QUAD,
                     {"secosw": 0.3, "sesinw": 0.2}, "tq"),
    "tau/hybrid": (mp.flux_dev_from_tau, "tau", TAU, GEO, HYB, "th"),
    "tau/hybrid_contact_ecc": (mp.flux_dev_from_tau, "tau", TAU, GEO,
                               dict(HYB, exp_time=0.02, secosw=0.3,
                                    sesinw=0.2), "th"),
    "tau/hybrid_basis": (mp.flux_dev_from_tau, "tau", TAU, GEO,
                         {"limb_dark": "hybrid4", "ld_basis": True}, "th"),
}
# Occultors larger than the star (0.11.0): a total occultation through
# every kernel family, so each calling form also runs the r > 1 branches.
ZW = np.linspace(0.0, 9.0, 64)                      # total, partial, none
GEO_WD = [("period", 1.4079), ("a", 336.0), ("b", 3.0), ("r", 7.28)]
TAU_WD = np.linspace(-0.012, 0.012, 64)
KERNEL_ENTRIES.update({
    "z/quad r>1": (mp.flux_dev_metal, "z", ZW, [("r", 7.28)] + QUAD, {},
                   "zq"),
    "z/ld_basis r>1": (mp.flux_dev_metal, "z", ZW, [("r", 7.28)],
                       {"ld_basis": True}, "zb"),
    "z-hybrid-entry r>1": (flux_dev_metal_hybrid, "z", ZW,
                           [("r", 7.28), ("law", "hybrid4"), ("u", W4)], {},
                           "zh"),
    "tau/quad_contact r>1": (mp.flux_dev_from_tau, "tau", TAU_WD,
                             GEO_WD + QUAD, {"exp_time": 0.0014}, "tq"),
    "tau/hybrid_ecc r>1": (mp.flux_dev_from_tau, "tau", TAU_WD, GEO_WD,
                           dict(HYB, secosw=0.3, sesinw=0.2), "th"),
})
GRAPH_ENTRIES = {
    "flux_dev": (mp.flux_dev, "z", Z, [("r", 0.1)] + QUAD, {}, None),
    "light_curve": (mp.light_curve, "z", Z, [("r", 0.1)] + QUAD, {}, None),
    "flux_dev_poly": (mp.flux_dev_poly, "z", Z,
                      [("r", 0.1), ("u", [0.4, 0.25])], {}, None),
    "flux_dev_hybrid": (mp.flux_dev_hybrid, "z", Z,
                        [("r", 0.1), ("w", W4), ("law", "hybrid4")], {}, None),
    "shape_cols": (mp.shape_cols, "z", Z, [("r", 0.1), ("law", "hybrid4")],
                   {}, None),
    "sn_dev": (mp.sn_dev, "z", Z, [("r", 0.1)], {}, None),
    "sn_dev_with_aux": (mp.sn_dev_with_aux, "z", Z, [("r", 0.1)], {}, None),
    "flux_dev_analytic": (mp.flux_dev_analytic, "z", Z,
                          [("r", 0.1)] + QUAD, {}, None),
}
GRAPH_ENTRIES.update({
    "flux_dev r>1": (mp.flux_dev, "z", ZW, [("r", 7.28)] + QUAD, {}, None),
    "flux_dev_hybrid r>1": (mp.flux_dev_hybrid, "z", ZW,
                            [("r", 7.28), ("w", W4), ("law", "hybrid4")], {},
                            None),
})
ALL = {**KERNEL_ENTRIES, **GRAPH_ENTRIES}


def _r0(entry):
    """The entry's own radius: tests that vary r scale it from here, so an
    r > 1 row is exercised at r > 1."""
    return float(dict(ALL[entry][3])["r"])


def _fp32_tol(entry):
    """Route-to-route fp32 agreement: 2.5e-7 for r < 1. An occultor larger
    than the star cancels terms of size ~r (the fp32 error grows ~ r^2), and
    on its 100%-deep, steep eclipse an ulp moved in a contact time shows:
    the documented fp32 budget there (test_large_occultor.TOL32)."""
    return 2.5e-7 if _r0(entry) < 1.0 else 2e-5


def _first(out):
    """The leading array of an entry's output (sn_dev returns a tuple)."""
    return out[0] if isinstance(out, (tuple, list)) else out


def _np(out):
    o = _first(out)
    mx.eval(o)
    return np.asarray(o)


def _call(entry, data, params=None, form="positional"):
    fn, dn, _, pos, kw, _ = ALL[entry]
    params = dict(pos) if params is None else params
    if form == "positional":
        return fn(data, *params.values(), **kw)
    if form == "data-keyword":
        return fn(**{dn: data}, **params, **kw)
    if form == "all-keyword":
        return fn(**{dn: data, **params, **kw})
    raise ValueError(form)


def _with(entry, **over):
    """The entry's positional params with some replaced."""
    p = dict(ALL[entry][3])
    p.update(over)
    return p


@contextlib.contextmanager
def kernel_spy():
    """Counts calls to each kernel family's dispatch point."""
    seen = []
    targets = [(M, "_flux_dev_metal_core", "zq"),
               (M, "_ld_basis_metal_core", "zb"),
               (M, "_make_tau_core_g", "tq"),
               (MH, "_make_core", "th"),
               (MH, "_z_core", "zh")]
    saved = []
    for mod, name, tag in targets:
        orig = getattr(mod, name)
        saved.append((mod, name, orig))

        def spy(*a, _orig=orig, _tag=tag, **k):
            seen.append(_tag)
            return _orig(*a, **k)
        setattr(mod, name, spy)
    try:
        yield seen
    finally:
        for mod, name, orig in saved:
            setattr(mod, name, orig)


# ---------------------------------------------------------------------------
# call forms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("entry", list(ALL))
@pytest.mark.parametrize("form", ["data-keyword", "all-keyword"])
def test_call_forms_agree_bitwise(entry, form):
    """Data by keyword and all-keyword calls equal the positional call
    (0.10.6's decorator took the data positionally only: z= raised)."""
    base = ALL[entry][2]
    d = mx.array(base, dtype=mx.float32)
    ref = _np(_call(entry, d))
    got = _np(_call(entry, d, form=form))
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("entry", list(ALL))
def test_missing_data_is_a_clear_type_error(entry):
    fn, dn, _, pos, kw, _ = ALL[entry]
    with pytest.raises(TypeError):
        fn(**dict(pos), **kw)


# ---------------------------------------------------------------------------
# data containers: the dtype rule
# ---------------------------------------------------------------------------

CONTAINERS = {
    "mx-f32": (lambda b: mx.array(b, dtype=mx.float32), np.float32),
    "np-f32": (lambda b: b.astype(np.float32), np.float32),
    "mx-f64": (lambda b: mx.array(b, dtype=mx.float64), np.float64),
    "np-f64": (lambda b: b, np.float64),
    "list": (lambda b: b.tolist(), np.float64),
    "mx-f16": (lambda b: mx.array(b, dtype=mx.float16), np.float64),
    "mx-i32": (lambda b: mx.array(np.round(b * 100).astype(np.int32)),
               np.float64),
    "np-i64": (lambda b: np.round(b * 100).astype(np.int64), np.float64),
}


def _values(x):
    """The float64 values a container holds (fp16 / int exactly)."""
    if isinstance(x, mx.array):
        with mx.stream(mx.cpu):
            return np.asarray(x.astype(mx.float64))
    return np.asarray(x, dtype=np.float64)


@pytest.mark.parametrize("entry", list(ALL))
@pytest.mark.parametrize("container", list(CONTAINERS))
def test_data_dtype_is_the_computation_dtype(entry, container):
    """float32 in any container is float32, bitwise the explicit fp32
    mx.array call; anything else is float64, bitwise the explicit fp64
    mx.array call on the same values. Never an exception, never a silent
    demotion (mx.array() alone takes float64 numpy to float32)."""
    make, want = CONTAINERS[container]
    raw = make(ALL[entry][2])
    got = _np(_call(entry, raw))
    assert got.dtype == want
    vals = _values(raw)
    ref_data = mx.array(vals, dtype=mx.float32 if want == np.float32
                        else mx.float64)
    ref = _np(_call(entry, ref_data))
    assert np.array_equal(got, ref, equal_nan=True)


# ---------------------------------------------------------------------------
# parameter dtypes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("entry", list(KERNEL_ENTRIES))
@pytest.mark.parametrize("stream", ["default", "cpu"])
def test_kernel_entries_cast_fp64_parameters_to_fp32_data(entry, stream):
    """An fp64 radius with fp32 data: fp32 out, on either stream, equal to
    the all-fp32 call to an ulp (the cast rounds 0.1 once, as fp32 does)."""
    d = mx.array(ALL[entry][2], dtype=mx.float32)
    ctx = mx.stream(mx.cpu) if stream == "cpu" else contextlib.nullcontext()
    with ctx:
        got = _np(_call(entry, d, _with(entry, r=mx.array(_r0(entry), mx.float64))))
        ref = _np(_call(entry, d, _with(entry, r=mx.array(_r0(entry), mx.float32))))
    assert got.dtype == np.float32
    assert np.array_equal(got, ref)


@pytest.mark.skipif(not KERNEL, reason="the kernels need Metal and the GPU "
                    "stream")
@pytest.mark.parametrize("entry", list(KERNEL_ENTRIES))
def test_fp32_data_with_fp64_parameters_takes_the_kernel(entry):
    """The cast keeps the call on the kernel: the family's dispatch point
    is reached on the default stream, and not on the CPU stream."""
    family = ALL[entry][5]
    d = mx.array(ALL[entry][2], dtype=mx.float32)
    p = _with(entry, r=mx.array(_r0(entry), mx.float64))
    with kernel_spy() as seen:
        _np(_call(entry, d, p))
    assert family in seen
    with kernel_spy() as seen, mx.stream(mx.cpu):
        _np(_call(entry, d, p))
    assert family not in seen


@pytest.mark.parametrize("entry", list(KERNEL_ENTRIES))
def test_kernel_entries_cast_fp32_parameters_to_fp64_data(entry):
    """The reverse: fp64 data with an fp32 radius is fp64 throughout, the
    radius widened exactly (bitwise the fp64 call with that fp32 value)."""
    d = mx.array(ALL[entry][2], dtype=mx.float64)
    r32 = mx.array(_r0(entry), dtype=mx.float32)
    got = _np(_call(entry, d, _with(entry, r=r32)))
    ref = _np(_call(entry, d, _with(entry, r=float(np.float32(_r0(entry))))))
    assert got.dtype == np.float64
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("entry", list(GRAPH_ENTRIES))
@pytest.mark.parametrize("stream", ["default", "cpu"])
def test_graph_entries_promote_as_mlx_does(entry, stream):
    """Graph functions keep MLX promotion: fp32 data with an fp64 radius
    is an fp64 result, on either stream (the call goes to the CPU stream
    itself; it raised on the GPU before 0.10.7)."""
    d = mx.array(ALL[entry][2], dtype=mx.float32)
    ctx = mx.stream(mx.cpu) if stream == "cpu" else contextlib.nullcontext()
    with ctx:
        got = _np(_call(entry, d, _with(entry, r=mx.array(_r0(entry), mx.float64))))
    assert got.dtype == np.float64 and np.isfinite(got).all()


# ---------------------------------------------------------------------------
# layouts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("entry", list(KERNEL_ENTRIES))
def test_kernel_entries_take_rows_with_per_row_parameters(entry):
    """(n, m) data with (n,) parameters: row j equals the (m,) call with
    the j-th parameter value."""
    base = ALL[entry][2]
    d2 = mx.array(np.stack([base, base]), dtype=mx.float32)
    rs = [_r0(entry), 1.2 * _r0(entry)]
    got = _np(_call(entry, d2, _with(entry, r=mx.array(rs, mx.float32))))
    for j, rj in enumerate(rs):
        row = _np(_call(entry, mx.array(base, dtype=mx.float32),
                        _with(entry, r=rj)))
        assert np.abs(got[j].astype(np.float64)
                      - row.astype(np.float64)).max() <= _fp32_tol(entry), j


@pytest.mark.parametrize("entry", list(GRAPH_ENTRIES))
def test_graph_entries_broadcast(entry):
    """Graph functions broadcast: (n, m) data takes r as (n, 1) -- an (n,)
    r broadcasts against the point axis instead and raises, as MLX does.
    That is their contract, distinct from the kernels' per-row one."""
    base = ALL[entry][2]
    d2 = mx.array(np.stack([base, base]), dtype=mx.float32)
    got = _np(_call(entry, d2, _with(
        entry, r=mx.array([[_r0(entry)], [1.2 * _r0(entry)]],
                           dtype=mx.float32))))
    for j, rj in enumerate([_r0(entry), 1.2 * _r0(entry)]):
        row = _np(_call(entry, mx.array(base, dtype=mx.float32),
                        _with(entry, r=mx.array(rj, dtype=mx.float32))))
        assert np.array_equal(got[j], row), j


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------

def _loss(entry, data):
    def f(r):
        return mx.sum(_first(_call(entry, data, _with(entry, r=r))))
    return f


@pytest.mark.parametrize("entry", list(ALL))
def test_grad_fp32_on_the_default_stream_and_fp64_on_the_cpu_stream(entry):
    """d/dr in fp32 on the default stream, and in fp64 with the mx.grad
    call itself on the CPU stream (MLX's rule for a float64 input; see
    metalplanet.dtypes). Both finite and in agreement; the fp64 route
    with Python-float coefficients crashed flux_dev_analytic's backward
    pass before 0.10.7."""
    base = ALL[entry][2]
    g32 = mx.grad(_loss(entry, mx.array(base, dtype=mx.float32)))(
        mx.array(_r0(entry), dtype=mx.float32))
    # evaluated before the CPU block: MLX cannot evaluate a pending GPU
    # fp32 graph and a CPU fp64 graph in one mx.eval made there
    mx.eval(g32)
    with mx.stream(mx.cpu):
        g64 = mx.grad(_loss(entry, mx.array(base, dtype=mx.float64)))(
            mx.array(_r0(entry), dtype=mx.float64))
        mx.eval(g64)
    a, b = float(g32.item()), float(g64.item())
    assert np.isfinite(a) and np.isfinite(b)
    assert abs(a - b) <= 2e-3 * max(abs(b), 1.0)


@pytest.mark.parametrize("entry", list(KERNEL_ENTRIES))
def test_grad_in_an_fp64_parameter_with_fp32_data(entry):
    """The supported form of d/dr for an fp64 r with fp32 data: the
    mx.grad call on the CPU stream (from the GPU default MLX's own
    transform machinery raises, whatever the function does -- see
    metalplanet.dtypes). The gradient comes back fp64, equal to the
    all-fp32 one on the CPU stream: the cast is the identity map's."""
    d = mx.array(ALL[entry][2], dtype=mx.float32)
    with mx.stream(mx.cpu):
        g = mx.grad(_loss(entry, d))(mx.array(_r0(entry), dtype=mx.float64))
        g32 = mx.grad(_loss(entry, d))(mx.array(_r0(entry), dtype=mx.float32))
        mx.eval(g)
        mx.eval(g32)
    assert g.dtype == mx.float64
    assert float(g.item()) == float(g32.item())


@pytest.mark.parametrize("entry", list(ALL))
def test_compile_equals_eager(entry):
    d = mx.array(ALL[entry][2], dtype=mx.float32)
    eager = _np(_call(entry, d))
    comp = mx.compile(lambda r: _first(_call(entry, d, _with(entry, r=r))))
    got = _np(comp(mx.array(_r0(entry), dtype=mx.float32)))
    assert np.abs(got.astype(np.float64)
                  - eager.astype(np.float64)).max() <= _fp32_tol(entry)


# ---------------------------------------------------------------------------
# TransitModel: its dtype argument decides, whatever container t is
# ---------------------------------------------------------------------------

def _params():
    p = mp.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w = 0.0, 3.45, 0.1, 8.8, 87.0, 0.0, 90.0
    p.limb_dark, p.u = "quadratic", [0.4, 0.25]
    return p


T = np.linspace(-0.15, 0.15, 201)
T_CONTAINERS = {
    "np-f64": T, "np-f32": T.astype(np.float32), "list": T.tolist(),
    "mx-f32": mx.array(T, dtype=mx.float32),
    "mx-f64": mx.array(T, dtype=mx.float64),
    "mx-f16": mx.array(T, dtype=mx.float16),
}


@pytest.mark.parametrize("container", list(T_CONTAINERS))
@pytest.mark.parametrize("dtype", [None, mx.float32, mx.float64],
                         ids=["default", "fp32", "fp64"])
def test_transit_model_dtype_argument_decides(container, dtype):
    """TransitModel's contract is batman's: an explicit ``dtype`` (fp64 by
    default) whatever container ``t`` comes in; the times themselves are
    the container's values. Checked against an fp64 model on the same
    (possibly rounded) times."""
    t = T_CONTAINERS[container]
    tv = _values(t)
    ref = mp.TransitModel(_params(), tv).light_curve(_params())
    m = mp.TransitModel(_params(), t, dtype=dtype)
    want = mx.float64 if dtype is None else dtype
    assert m.dtype == want
    got = m.light_curve(_params())
    tol = 2e-6 if want == mx.float32 else 1e-12
    assert np.abs(got - ref).max() < tol


def test_transit_model_keyword_calls():
    p = _params()
    m = mp.TransitModel(params=p, t=T)
    ref = m.light_curve(p)
    assert np.array_equal(m.light_curve(params=p), ref)
    assert np.array_equal(np.asarray(m.light_curve_mx(params=p)), ref)
    # the batched path is its own arithmetic route (test_batched_frontend)
    assert np.abs(m.light_curves(params_seq=[p])[0] - ref).max() < 1e-14
