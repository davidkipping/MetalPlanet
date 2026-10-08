"""The data/dtype contract shared by every public function that takes z or
tau (0.10.6-0.10.7).

* The data's dtype is the computation's. float32 stays float32 (the fused
  kernels on the GPU); anything else -- float64, float16, bfloat16,
  integers, Python floats and sequences, numpy arrays of any of these --
  is float64 (the exact graph). ``as_data`` applies the rule.
* MLX has no float64 on Metal at all, so a float64 computation runs on
  the CPU stream; ``fp64_on_cpu`` puts it there.
* The kernel entry points cast parameters to the data's dtype
  (``_as_dtype``); the graph functions keep MLX's broadcasting and type
  promotion.
* ``mx.grad`` with respect to a float64 *input* must itself be called
  under ``mx.stream(mx.cpu)``: MLX's transform machinery runs float64 ops
  on the default stream, which no function body can redirect.
"""

from __future__ import annotations

import functools
import inspect

import mlx.core as mx
import numpy as np

__all__ = ["as_data", "fp64_on_cpu"]


def _gpu_stream_active() -> bool:
    try:
        return mx.default_device() == mx.Device(mx.DeviceType.gpu)
    except Exception:
        return False


def as_data(x) -> mx.array:
    """A data array (z or tau) in one of the two computation dtypes. The
    rule, the same for every container: float32 stays float32 (the
    kernels); anything else -- float64, float16, bfloat16, integers,
    Python floats and sequences -- is float64 (the exact graph). An
    mx.array of another dtype is cast on the CPU stream, where float64
    may live. mx.array() alone would silently take a float64 numpy array
    to float32."""
    if isinstance(x, mx.array):
        if x.dtype in (mx.float32, mx.float64):
            return x
        with mx.stream(mx.cpu):
            return x.astype(mx.float64)
    a = np.asarray(x)
    return mx.array(a, dtype=mx.float32 if a.dtype == np.float32
                    else mx.float64)


def _as_dtype(p, dtype):
    """A parameter in the data's dtype: an mx.array of another dtype is
    cast -- on the CPU stream when float64 is on either side (MLX has no
    float64 on Metal, so even the cast cannot run there), on the active
    stream otherwise; Python numbers stay as they are (weakly typed, they
    take the data's dtype by themselves)."""
    if not isinstance(p, mx.array) or p.dtype == dtype:
        return p
    if mx.float64 in (p.dtype, dtype):
        with mx.stream(mx.cpu):
            return p.astype(dtype)
    return p.astype(dtype)


def fp64_on_cpu(fn=None, *, any_arg: bool = False):
    """Entry-point decorator: the first parameter of ``fn`` is the data
    array (z or tau), passed positionally or by its own name. It is
    converted by ``as_data``, and when it is float64 on the GPU stream the
    call runs on the CPU stream.

    Two families use it. The kernel entry points (metal.py,
    metal_hybrid.py) cast every parameter to the data's dtype, so only the
    data decides (``any_arg=False``). The graph functions (flux_dev,
    flux_dev_hybrid, ...) keep MLX's broadcasting and type promotion -- an
    fp64 parameter makes an fp64 result -- so any float64 mx.array among
    the top-level arguments sends the call to the CPU stream
    (``any_arg=True``). Either way the decorator only changes calls that
    would otherwise raise (float64 on the GPU stream; non-MLX data):
    every call that already worked runs exactly as before. MLX has no float64 on Metal *at all* --
    even a slice raises -- so fp64 is not a dispatch choice but a device
    one, and this is what makes "fp64 falls back to the graph" true rather
    than an exception the caller has to pre-empt; TransitModel does the
    same (api.py: _stream). One mechanism for every kernel entry point.

    It cannot help ``mx.grad`` with respect to a float64 *input*: MLX's
    transform machinery itself runs float64 ops on the default stream
    (verified: it raises even when the differentiated body runs wholly on
    the CPU stream), so that call must be made under
    ``mx.stream(mx.cpu)`` by the caller, as the README says."""
    if fn is None:
        return lambda f: fp64_on_cpu(f, any_arg=any_arg)
    name = next(iter(inspect.signature(fn).parameters))

    def _fp64(x):
        return isinstance(x, mx.array) and x.dtype == mx.float64

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if args:
            data, args = args[0], args[1:]
        elif name in kwargs:
            data = kwargs.pop(name)
        else:
            raise TypeError(f"{fn.__name__}() missing its data argument "
                            f"'{name}'")
        data = as_data(data)
        fp64 = data.dtype == mx.float64 or (
            any_arg and (any(map(_fp64, args))
                         or any(map(_fp64, kwargs.values()))))
        if fp64 and _gpu_stream_active():
            with mx.stream(mx.cpu):
                return fn(data, *args, **kwargs)
        return fn(data, *args, **kwargs)
    return wrapped
