"""batman-style frontend: TransitParams + TransitModel.

Mirrors the API and conventions of Kreidberg's ``batman`` package for
drop-in familiarity:

    import metalplanet
    params = metalplanet.TransitParams()
    params.t0 = 0.0            # time of inferior conjunction
    params.per = 1.0           # orbital period
    params.rp = 0.1            # planet radius [stellar radii]
    params.a = 15.0            # semi-major axis [stellar radii]
    params.inc = 87.0          # inclination [deg]
    params.ecc = 0.0           # eccentricity
    params.w = 90.0            # longitude of periastron [deg]
    params.u = [0.1, 0.3]      # limb-darkening coefficients
    params.limb_dark = "quadratic"

    m = metalplanet.TransitModel(params, t)
    flux = m.light_curve(params)          # numpy array

Differences from batman, by design:

* the quadratic/linear/uniform models are evaluated with the analytic
  ALFM19 formulation (float64 accuracy ~1e-13; batman's quadratic path
  carries a ~2e-8 floor from its Hastings E/K approximations), and the
  whole computation is an MLX graph — differentiable and GPU-capable;
* supported limb_dark: "uniform", "linear", "quadratic" (arbitrary-order
  polynomial and nonlinear laws are roadmap);
* no error-tolerance machinery (`max_err`, `fac`, `nthreads`): the model
  is closed-form, there is no integration error to budget.

For float32 GPU *sampling* with thousands of chains, use
``metalplanet.anvil`` instead — absolute times of order 2.45e6 days
cannot survive float32, so this frontend computes in float64 on the CPU
by default (`dtype=mx.float32` opts into the GPU at your own risk for
well-conditioned times).
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from .flux import flux_dev
from .metal import flux_dev_metal
from .anchored import separation_anchored
from .solution import sn_dev
from .trig import sincos

__all__ = ["TransitParams", "TransitModel"]

_SUPPORTED_LD = ("uniform", "linear", "quadratic")


class TransitParams:
    """Object to store the physical parameters of the transit
    (attribute-compatible with batman.TransitParams)."""

    def __init__(self):
        self.t0 = None            # time of inferior conjunction
        self.per = None           # orbital period
        self.rp = None            # planet radius [stellar radii]
        self.a = None             # semi-major axis [stellar radii]
        self.inc = None           # orbital inclination [degrees]
        self.ecc = 0.0            # eccentricity
        self.w = 90.0             # longitude of periastron [degrees]
        self.u = []               # limb-darkening coefficients
        self.limb_dark = None     # "uniform" | "linear" | "quadratic"
        self.fp = None            # planet/star flux ratio (secondary)
        self.t_secondary = None   # unused; secondary timing is computed


def _ld_coeffs(params) -> tuple[float, float]:
    law = params.limb_dark
    u = list(params.u) if params.u is not None else []
    if law == "uniform":
        if len(u) != 0:
            raise ValueError("uniform limb darkening takes no coefficients")
        return 0.0, 0.0
    if law == "linear":
        if len(u) != 1:
            raise ValueError("linear limb darkening takes 1 coefficient")
        return float(u[0]), 0.0
    if law == "quadratic":
        if len(u) != 2:
            raise ValueError("quadratic limb darkening takes 2 coefficients")
        return float(u[0]), float(u[1])
    raise ValueError(
        f"limb_dark {law!r} not supported; choose from {_SUPPORTED_LD} "
        "(arbitrary-order polynomial laws are on the roadmap)")


class TransitModel:
    """Precomputes the (super)sampled time grid; ``light_curve(params)``
    evaluates the model for (possibly updated) parameters, batman-style.

    Args:
        params: TransitParams (validated here; values may change between
            light_curve calls, but the limb-darkening *law* and the
            transit type are fixed per model).
        t: times, same units as params.t0/per (absolute values are fine —
            computation is float64 on the CPU by default).
        transittype: "primary" or "secondary".
        supersample_factor: average this many samples over exp_time.
        exp_time: exposure time (same units as t).
        dtype: mx.float64 (default; CPU) or mx.float32 (default stream —
            only sensible for well-conditioned times).
    """

    def __init__(self, params, t, transittype: str = "primary",
                 supersample_factor: int = 1, exp_time: float = 0.0,
                 dtype=None, use_metal: bool = True):
        _ld_coeffs(params)  # validate law/coefficients early
        if transittype not in ("primary", "secondary"):
            raise ValueError("transittype must be 'primary' or 'secondary'")
        if transittype == "secondary" and params.fp is None:
            raise ValueError("secondary eclipse needs params.fp")
        if supersample_factor > 1 and exp_time <= 0.0:
            raise ValueError("supersampling needs exp_time > 0")
        self.transittype = transittype
        self.limb_dark = params.limb_dark
        self.dtype = mx.float64 if dtype is None else dtype
        self._stream = mx.cpu if self.dtype == mx.float64 else None
        # fused Metal kernel for fp32 GPU evaluation (falls back on its
        # own for fp64/CPU, so leaving this True is always safe)
        self.use_metal = bool(use_metal)

        t = np.asarray(t, dtype=np.float64)
        self.t = t
        self.supersample_factor = int(supersample_factor)
        self.exp_time = float(exp_time)
        if self.supersample_factor > 1:
            n = self.supersample_factor
            # batman convention: endpoint-inclusive uniform samples
            offs = np.linspace(-0.5 * self.exp_time, 0.5 * self.exp_time, n)
            t_super = (t[:, None] + offs[None, :]).ravel()
        else:
            t_super = t
        self._t_super = t_super
        self._t_mx = mx.array(t_super.astype(np.float64), dtype=self.dtype)
        self._compiled = {}

    # -- internals ---------------------------------------------------------
    #
    # light_curve runs through mx.compile'd graphs (kernel fusion is
    # worth ~2x+ over eager MLX here). Parameters enter as *traced 0-d
    # arrays*, so batman-style parameter updates between calls reuse the
    # same compiled kernels — only the circular/eccentric branch (a
    # Python-level dispatch on ecc == 0) selects between two graphs,
    # built lazily and cached per model.

    def _separation(self, params):
        """(z, front) over the supersampled grid, in the model dtype
        (eager; used by light_curve_mx and tests)."""
        per = float(params.per)
        t0 = float(params.t0)
        ecc = float(params.ecc)
        inc = math.radians(float(params.inc))
        t = self._t_mx
        if ecc == 0.0:
            phase = (2.0 * math.pi / per) * (t - t0)
            sphi, cphi = sincos(phase)
            a = float(params.a)
            b = a * math.cos(inc)
            z2 = (a * sphi) ** 2 + (b * cphi) ** 2
            z = mx.sqrt(mx.maximum(z2, 1e-24))
            return z, cphi > 0.0
        w = math.radians(float(params.w))
        phi = (2.0 * math.pi / per) * (t - t0)
        k = math.sqrt(ecc) * math.cos(w)
        h = math.sqrt(ecc) * math.sin(w)
        return separation_anchored(phi, mx.array(k, dtype=self.dtype),
                                   mx.array(h, dtype=self.dtype),
                                   mx.array(float(params.a), dtype=self.dtype),
                                   mx.array(math.cos(inc), dtype=self.dtype))

    def _photom(self, z, front, rp, u1, u2, fp):
        if self.transittype == "primary":
            z_eff = mx.where(front, z, 2.0 + z)
            core = flux_dev_metal if self.use_metal else flux_dev
            return 1.0 + core(z_eff, rp, u1, u2)
        z_eff = mx.where(front, 2.0 + z, z)
        s0d, _, _ = sn_dev(z_eff, rp)
        # visible fraction of the (uniform) planet disk
        return 1.0 + fp * (1.0 + s0d / (math.pi * rp * rp))

    def _eval(self, params) -> mx.array:
        u1, u2 = _ld_coeffs(params)
        z, front = self._separation(params)
        fp = 0.0 if params.fp is None else float(params.fp)
        return self._photom(z, front, float(params.rp), u1, u2, fp)

    def _get_compiled(self, circular: bool):
        fn = self._compiled.get(circular)
        if fn is not None:
            return fn
        t = self._t_mx

        if circular:
            def raw(t0, per, a, b, rp, u1, u2, fp):
                phase = (2.0 * math.pi) * (t - t0) / per
                sphi, cphi = sincos(phase)
                z = mx.sqrt(mx.maximum((a * sphi) ** 2 + (b * cphi) ** 2,
                                       1e-24))
                return self._photom(z, cphi > 0.0, rp, u1, u2, fp)
        else:
            # transit-anchored: phi is measured straight from t0, so no
            # mean-anomaly-at-transit offset is needed, and float32 stays
            # accurate all the way to e = 0 (see anchored.py).
            def raw(t0, per, a, k, h, ci, rp, u1, u2, fp):
                phi = (2.0 * math.pi) * (t - t0) / per
                z, front = separation_anchored(phi, k, h, a, ci)
                return self._photom(z, front, rp, u1, u2, fp)

        fn = mx.compile(raw)
        self._compiled[circular] = fn
        return fn

    def _eval_compiled(self, params) -> mx.array:
        u1, u2 = _ld_coeffs(params)
        fp = 0.0 if params.fp is None else float(params.fp)
        ecc = float(params.ecc)
        inc = math.radians(float(params.inc))

        def s(x):
            return mx.array(float(x), dtype=self.dtype)

        if ecc == 0.0:
            a = float(params.a)
            return self._get_compiled(True)(
                s(params.t0), s(params.per), s(a), s(a * math.cos(inc)),
                s(params.rp), s(u1), s(u2), s(fp))
        w = math.radians(float(params.w))
        return self._get_compiled(False)(
            s(params.t0), s(params.per), s(params.a),
            s(math.sqrt(ecc) * math.cos(w)), s(math.sqrt(ecc) * math.sin(w)),
            s(math.cos(inc)), s(params.rp), s(u1), s(u2), s(fp))

    # -- batman-compatible surface ----------------------------------------

    def light_curve(self, params) -> np.ndarray:
        """Model flux at the times given at construction (numpy array)."""
        if params.limb_dark != self.limb_dark:
            raise ValueError(
                "limb-darkening law changed since model construction; "
                "build a new TransitModel")
        if self._stream is not None:
            with mx.stream(self._stream):
                f = self._eval_compiled(params)
                out = np.array(f, dtype=np.float64)
        else:
            out = np.array(self._eval_compiled(params), dtype=np.float64)
        if self.supersample_factor > 1:
            out = out.reshape(self.t.size, self.supersample_factor).mean(axis=1)
        return out

    def light_curve_mx(self, params) -> mx.array:
        """Supersampled-grid flux as an MLX array (stays in the graph;
        no averaging applied) — for building differentiable pipelines."""
        return self._eval(params)
