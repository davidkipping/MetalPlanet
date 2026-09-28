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
* supported limb_dark: "uniform", "linear", "quadratic", and
  "polynomial" for I(mu)/I0 = 1 - sum_n u_n (1-mu)^n at ANY order
  (ALFM19's M_n recursion, metalplanet/poly.py); batman's
  non-polynomial laws ("nonlinear", "squareroot", ...) are not
  covered by this formulation;
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
from .poly import flux_dev_poly
from .exposure import (contact_geometry, contact_offsets,
                       exposure_nodes)
from .solution import sn_dev
from .trig import sincos

__all__ = ["TransitParams", "TransitModel"]

_SUPPORTED_LD = ("uniform", "linear", "quadratic", "polynomial")


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
    if law == "polynomial":
        if len(u) == 0:
            raise ValueError("polynomial limb darkening needs >= 1 "
                             "coefficient (use 'uniform' for none)")
        return None, None          # handled by the polynomial core
    raise ValueError(
        f"limb_dark {law!r} not supported; choose from {_SUPPORTED_LD}. "
        "Non-polynomial laws (nonlinear, squareroot, logarithmic) are "
        "outside the ALFM19 formulation.")


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _check_ecc(ecc):
    """Reject eccentricities the model cannot represent. The single-set
    path gets this for free from math.sqrt raising on a negative; the
    batched path would instead hand np.sqrt a negative, emit a warning
    and return an all-NaN row that a sampler turns into -inf with nothing
    pointing at the parameter."""
    bad = ~((ecc >= 0.0) & (ecc < 1.0))
    if bool(np.any(bad)):
        raise ValueError(
            f"eccentricity must be in [0, 1); got "
            f"{np.asarray(ecc)[bad][:4]} (and possibly more)")


def _ld_coeffs_batch(law, u):
    """(u1, u2) columns for the legacy laws, from a (n_sets, N) array."""
    if law == "uniform":
        z = np.zeros(u.shape[0])
        return z, z
    if law == "linear":
        return u[:, 0], np.zeros(u.shape[0])
    if law == "quadratic":
        return u[:, 0], u[:, 1]
    raise ValueError(f"unexpected law {law!r} in the batched path")


def _unpack_ld(poly, tail):
    """(u1, u2, uvec, fp) from the trailing limb-darkening + fp args."""
    if poly:
        uv, fp = tail
        return None, None, uv, fp
    u1, u2, fp = tail
    return u1, u2, None, fp


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
                 dtype=None, use_metal: bool = True,
                 integration: str = "supersample", n_gl: int = 7):
        _ld_coeffs(params)  # validate law/coefficients early
        if transittype not in ("primary", "secondary"):
            raise ValueError("transittype must be 'primary' or 'secondary'")
        if transittype == "secondary" and params.fp is None:
            raise ValueError("secondary eclipse needs params.fp")
        if supersample_factor > 1 and exp_time <= 0.0:
            raise ValueError("supersampling needs exp_time > 0")
        if integration not in ("supersample", "contact"):
            raise ValueError('integration must be "supersample" or "contact"')
        if integration == "contact":
            if exp_time <= 0.0:
                raise ValueError("contact integration needs exp_time > 0")
            if transittype != "primary":
                raise ValueError("contact integration is primary-transit only")
        self.integration = integration
        self.n_gl = int(n_gl)
        self.transittype = transittype
        self.limb_dark = params.limb_dark
        self._n_poly = (len(list(params.u))
                        if params.limb_dark == "polynomial" else 0)
        self.dtype = mx.float64 if dtype is None else dtype
        self._stream = mx.cpu if self.dtype == mx.float64 else None
        # fused Metal kernel for fp32 GPU evaluation (falls back on its
        # own for fp64/CPU, so leaving this True is always safe)
        self.use_metal = bool(use_metal)

        t = np.asarray(t, dtype=np.float64)
        self.t = t
        self.supersample_factor = int(supersample_factor)
        self.exp_time = float(exp_time)
        if self.supersample_factor > 1 and integration == "supersample":
            n = self.supersample_factor
            # batman convention: endpoint-inclusive uniform samples
            offs = np.linspace(-0.5 * self.exp_time, 0.5 * self.exp_time, n)
            t_super = (t[:, None] + offs[None, :]).ravel()
        else:
            t_super = t
        self._t_super = t_super
        # (6) Absolute mission time stamps cannot survive float32: a TESS
        # BTJD of ~2500 d has an fp32 ulp of 2.4e-4 d (21 s), Kepler's
        # BJD-2454833 is similar and raw BJD (2.457e6 d) has a 0.25 d ulp.
        # So for any non-float64 dtype the grid is re-centred on a float64
        # reference HERE, on the host, and t0 is shifted by the same
        # amount at every call site -- exactly what epoch_center_times does
        # for the anvil path. The fp32 graph then only ever sees O(baseline)
        # numbers. float64 keeps a zero offset so its heavily-tested
        # round-off behaviour is untouched.
        self._t_ref = (0.0 if self.dtype == mx.float64
                       else float(0.5 * (t_super.min() + t_super.max())))
        self._t_mx = mx.array(t_super - self._t_ref, dtype=self.dtype)
        self._compiled = {}

    # -- internals ---------------------------------------------------------
    #
    # light_curve runs through mx.compile'd graphs (kernel fusion is
    # worth ~2x+ over eager MLX here). Parameters enter as *traced 0-d
    # arrays*, so batman-style parameter updates between calls reuse the
    # same compiled kernels — only the circular/eccentric branch (a
    # Python-level dispatch on ecc == 0) selects between two graphs,
    # built lazily and cached per model.

    def _contact_nodes(self, params):
        """(times, weights) for the contact-split exposure rule, eager."""
        a = float(params.a)
        ci = math.cos(math.radians(float(params.inc)))
        ecc = float(params.ecc)
        esw = ecc * math.sin(math.radians(float(params.w)))
        a_sky, b = contact_geometry(a, ecc, esw, ci,
                                    sqrt=math.sqrt,
                                    maximum=lambda x, y: max(x, y))

        def s(x):
            return mx.array(float(x), dtype=self.dtype)

        cs = contact_offsets(s(params.rp), s(a_sky), s(b))
        return exposure_nodes(self._t_mx, s(params.t0 - self._t_ref),
                              s(params.per),
                              s(self.exp_time), cs, self.n_gl,
                              dtype=self.dtype)

    def _separation(self, params, t=None):
        """(z, front) over the supersampled grid (or an explicit time
        array), in the model dtype (eager; used by light_curve_mx and
        tests)."""
        per = float(params.per)
        t0 = float(params.t0) - self._t_ref
        ecc = float(params.ecc)
        inc = math.radians(float(params.inc))
        t = self._t_mx if t is None else t
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

    def _photom(self, z, front, rp, u1, u2, fp, uvec=None):
        if self.transittype == "primary":
            z_eff = mx.where(front, z, 2.0 + z)
            if uvec is not None:          # arbitrary-order polynomial law
                return 1.0 + flux_dev_poly(z_eff, rp, uvec,
                                           n_max=self._n_poly)
            core = flux_dev_metal if self.use_metal else flux_dev
            return 1.0 + core(z_eff, rp, u1, u2)
        z_eff = mx.where(front, 2.0 + z, z)
        s0d, _, _ = sn_dev(z_eff, rp)
        # visible fraction of the (uniform) planet disk
        return 1.0 + fp * (1.0 + s0d / (math.pi * rp * rp))

    def _uvec(self, params):
        """Traced coefficient vector for the polynomial law (else None)."""
        if not self._n_poly:
            return None
        return mx.array(np.asarray(list(params.u), dtype=np.float64),
                        dtype=self.dtype)

    def _eval(self, params) -> mx.array:
        u1, u2 = _ld_coeffs(params)
        fp = 0.0 if params.fp is None else float(params.fp)
        uvec = self._uvec(params)
        if self.integration == "contact":
            # the eager path must average too, or light_curve_mx would
            # quietly return the INSTANTANEOUS flux at the exposure
            # mid-times while light_curve returns the averaged one
            T, W = self._contact_nodes(params)
            z, front = self._separation(params, T)
            f = self._photom(z, front, float(params.rp), u1, u2, fp,
                             uvec=uvec)
            return mx.sum(f * W, axis=1)
        z, front = self._separation(params)
        return self._photom(z, front, float(params.rp), u1, u2, fp,
                            uvec=uvec)

    def _kernel_usable(self) -> bool:
        """The fused kernel serves the fp32 GPU *primary*-transit path for
        both circular and eccentric orbits; fp64, CPU streams, secondary
        eclipses and polynomial limb darkening keep the graph."""
        if not self.use_metal or self.transittype != "primary":
            return False
        if self._n_poly:            # the kernel is quadratic-only
            return False
        if self.dtype != mx.float32:
            return False
        from .metal import _gpu_stream_active, metal_available
        return metal_available() and _gpu_stream_active()

    def _get_compiled(self, circular: bool):
        fn = self._compiled.get(circular)
        if fn is not None:
            return fn
        t = self._t_mx

        poly = bool(self._n_poly)

        if self.integration == "contact":
            # Node times depend on the parameters (the contacts move), so
            # the grid is rebuilt inside the compiled graph each call and
            # `t` stays the exposure mid-times.
            n_gl = self.n_gl
            ex = self.exp_time

            def _avg(z, front, rp, u1, u2, fp, uv, w):
                f = self._photom(z, front, rp, u1, u2, fp, uvec=uv)
                return mx.sum(f * w, axis=1)

            if circular:
                def raw(t0, per, a, b, rp, *ld_fp):
                    u1, u2, uv, fp = _unpack_ld(poly, ld_fp)
                    ci = b / a
                    cs = contact_offsets(rp, a, b)
                    T, W = exposure_nodes(t, t0, per, ex, cs, n_gl,
                                          dtype=self.dtype)
                    phase = (2.0 * math.pi) * (T - t0) / per
                    sphi, cphi = sincos(phase)
                    z = mx.sqrt(mx.maximum((a * sphi) ** 2 + (b * cphi) ** 2,
                                           1e-24))
                    return _avg(z, cphi > 0.0, rp, u1, u2, fp, uv, W)
            else:
                def raw(t0, per, a, k, h, ci, rp, *ld_fp):
                    u1, u2, uv, fp = _unpack_ld(poly, ld_fp)
                    e = k * k + h * h
                    esw = h * mx.sqrt(mx.maximum(e, 1e-30))   # e sin w
                    a_sky, b_conj = contact_geometry(a, e, esw, ci)
                    cs = contact_offsets(rp, a_sky, b_conj)
                    T, W = exposure_nodes(t, t0, per, ex, cs, n_gl,
                                          dtype=self.dtype)
                    phi = (2.0 * math.pi) * (T - t0) / per
                    z, front = separation_anchored(phi, k, h, a, ci)
                    return _avg(z, front, rp, u1, u2, fp, uv, W)

        elif circular and poly:
            def raw(t0, per, a, b, rp, uv, fp):
                phase = (2.0 * math.pi) * (t - t0) / per
                sphi, cphi = sincos(phase)
                z = mx.sqrt(mx.maximum((a * sphi) ** 2 + (b * cphi) ** 2,
                                       1e-24))
                return self._photom(z, cphi > 0.0, rp, None, None, fp,
                                    uvec=uv)
        elif poly:
            def raw(t0, per, a, k, h, ci, rp, uv, fp):
                phi = (2.0 * math.pi) * (t - t0) / per
                z, front = separation_anchored(phi, k, h, a, ci)
                return self._photom(z, front, rp, None, None, fp, uvec=uv)
        elif circular and self._kernel_usable():
            # Circular orbit on the SAME fused kernel as the eccentric one:
            # k = h = 0 and cos i = b / a. Exact (the anchored orbit
            # degenerates to the circular one) and the kernel skips the
            # Kepler solve on e == 0 chains. See the eccentric branch below
            # for why period_ref is 0 and the period rides p_off.
            from .anchored import pack_orbit_constants
            from .metal import make_model_core_metal
            core = make_model_core_metal(0.0)
            m = t.shape[0]
            xdat = mx.stack([t, mx.zeros_like(t)])

            def raw(t0, per, a, b, rp, u1, u2, fp):
                def col(v):
                    return mx.reshape(v, (1,))
                zero = mx.zeros((1,), dtype=self.dtype)
                orb = pack_orbit_constants(zero, zero, col(b / a))
                dev = core(xdat, col(t0), col(per), col(rp), col(a), orb,
                           col(u1), col(u2))
                return 1.0 + mx.reshape(dev, (m,))
        elif circular:
            def raw(t0, per, a, b, rp, u1, u2, fp):
                phase = (2.0 * math.pi) * (t - t0) / per
                sphi, cphi = sincos(phase)
                z = mx.sqrt(mx.maximum((a * sphi) ** 2 + (b * cphi) ** 2,
                                       1e-24))
                return self._photom(z, cphi > 0.0, rp, u1, u2, fp)
        elif self._kernel_usable():
            # Whole eccentric model in one kernel. The anchored Kepler
            # solve as graph ops streams a lot of intermediates: measured
            # 0.23 Gpt/s against the kernel's 2.3, so this is ~10x at
            # large N.
            #
            # period_ref is baked into the kernel factory as a constant,
            # which would defeat batman-style parameter updates — so it
            # is set to zero and the period is carried by the *traced*
            # p_off input instead. With the epoch column k = 0 that is
            # exactly equivalent, gradients included: the kernel's
            # dphi/dp_off = 2 pi ((-k - n_w) P - tau_w) / P^2 reduces to
            # dphi/dP for k = 0.
            from .anchored import pack_orbit_constants
            from .metal import make_model_core_metal
            core = make_model_core_metal(0.0)
            m = t.shape[0]
            xdat = mx.stack([t, mx.zeros_like(t)])

            def raw(t0, per, a, k, h, ci, rp, u1, u2, fp):
                def col(v):
                    return mx.reshape(v, (1,))
                orb = pack_orbit_constants(col(k), col(h), col(ci))
                dev = core(xdat, col(t0), col(per), col(rp), col(a), orb,
                           col(u1), col(u2))
                return 1.0 + mx.reshape(dev, (m,))
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

        ld = ((self._uvec(params),) if self._n_poly
              else (s(u1), s(u2)))
        t0 = params.t0 - self._t_ref
        if ecc == 0.0:
            a = float(params.a)
            return self._get_compiled(True)(
                s(t0), s(params.per), s(a), s(a * math.cos(inc)),
                s(params.rp), *ld, s(fp))
        w = math.radians(float(params.w))
        return self._get_compiled(False)(
            s(t0), s(params.per), s(params.a),
            s(math.sqrt(ecc) * math.cos(w)), s(math.sqrt(ecc) * math.sin(w)),
            s(math.cos(inc)), s(params.rp), *ld, s(fp))

    # -- batman-compatible surface ----------------------------------------

    def _check_law(self, params):
        """The limb-darkening law AND its order are fixed per model: the
        order is baked into the compiled graph (and into the g_n affine
        map), so a changed count would be silently truncated by the zip in
        flux_dev_poly rather than raising."""
        if params.limb_dark != self.limb_dark:
            raise ValueError(
                "limb-darkening law changed since model construction; "
                "build a new TransitModel")
        if self._n_poly and len(list(params.u)) != self._n_poly:
            raise ValueError(
                f"polynomial limb-darkening order changed since model "
                f"construction ({self._n_poly} -> "
                f"{len(list(params.u))} coefficients); build a new "
                f"TransitModel")

    def light_curve(self, params) -> np.ndarray:
        """Model flux at the times given at construction (numpy array)."""
        self._check_law(params)
        if self._stream is not None:
            with mx.stream(self._stream):
                f = self._eval_compiled(params)
                out = np.array(f, dtype=np.float64)
        else:
            out = np.array(self._eval_compiled(params), dtype=np.float64)
        if (self.supersample_factor > 1
                and self.integration == "supersample"):
            out = out.reshape(self.t.size,
                              self.supersample_factor).mean(axis=1)
        return out

    # -- batched surface (the sampler-friendly one) ------------------------

    _BATCH_KEYS = ("t0", "per", "rp", "a", "inc", "ecc", "w", "fp")

    def _stack_params(self, params_seq):
        """(n_sets, ) arrays for each scalar parameter, plus (n_sets, N)
        limb-darkening coefficients. Accepts a sequence of TransitParams
        or one TransitParams whose attributes are already arrays."""
        if isinstance(params_seq, (list, tuple)):
            seq = list(params_seq)
            if not seq:
                raise ValueError("no parameter sets given")
            cols = {}
            for k in self._BATCH_KEYS:
                vals = [getattr(p, k) for p in seq]
                cols[k] = np.array([0.0 if v is None else float(v)
                                    for v in vals], dtype=np.float64)
            for p in seq:
                # the same validation the single-set path performs: an
                # unchecked set returns a plausible but wrong curve, or a
                # silent NaN row that a sampler reads as -inf
                self._check_law(p)
                _ld_coeffs(p)
            u = np.array([list(p.u) for p in seq], dtype=np.float64)
            _check_ecc(cols["ecc"])
            return cols, u
        p = params_seq                       # array-valued TransitParams
        self._check_law(p)
        # any subset of the attributes may be arrays; the batch size is
        # the longest of them (scalars broadcast against it)
        sizes = set()
        for k in self._BATCH_KEYS:
            v = getattr(p, k)
            if v is None:
                continue
            sizes.add(np.atleast_1d(np.asarray(v, dtype=np.float64)).size)
        sizes.discard(1)
        if len(sizes) > 1:
            raise ValueError(f"inconsistent parameter-array lengths: "
                             f"{sorted(sizes)}")
        n = sizes.pop() if sizes else 1
        cols = {}
        for k in self._BATCH_KEYS:
            v = getattr(p, k)
            v = 0.0 if v is None else v
            cols[k] = np.broadcast_to(
                np.atleast_1d(np.asarray(v, dtype=np.float64)),
                (n,)).astype(np.float64)
        u = np.asarray(p.u, dtype=np.float64)
        u = np.broadcast_to(np.atleast_2d(u), (n, u.shape[-1])) \
            if u.size else np.zeros((n, 0))
        _check_ecc(cols["ecc"])
        return cols, np.ascontiguousarray(u)

    def light_curves(self, params_seq) -> np.ndarray:
        """(n_sets, n_times) for MANY parameter sets in ONE batched call.

        ``light_curve`` is deliberately one-parameter-set-at-a-time, for
        batman parity — and calling it in a loop is the single worst thing
        a sampler can do here (a GPU dispatch costs ~0.2-0.7 ms whatever
        its size, so looping is two to three orders of magnitude slower
        than batching; see docs/sampler-integration.md). This is the
        batched form: the same times, many parameter sets, one dispatch.

        ``params_seq`` is either a sequence of TransitParams or a single
        TransitParams whose scalar attributes are arrays of equal length.
        Mixed circular and eccentric sets are fine: the transit-anchored
        orbit degenerates exactly to the circular one at e = 0, so one
        code path serves both.

        For fitting with thousands of chains prefer ``metalplanet.anvil``,
        which owns the likelihood and the float32 conditioning as well.
        """
        cols, u_np = self._stack_params(params_seq)
        n = cols["t0"].size
        dt = self.dtype

        def col(a):
            return mx.array(np.asarray(a, np.float64).reshape(n, 1), dtype=dt)

        stream = self._stream
        ctx = mx.stream(stream) if stream is not None else _nullcontext()
        with ctx:
            t0, per = col(cols["t0"] - self._t_ref), col(cols["per"])
            rp, a = col(cols["rp"]), col(cols["a"])
            ecc = cols["ecc"]
            inc = np.radians(cols["inc"])
            ci = col(np.cos(inc))
            w = np.radians(cols["w"])
            sq = np.sqrt(ecc)
            k, h = col(sq * np.cos(w)), col(sq * np.sin(w))
            fp = col(cols["fp"])
            if self._n_poly:
                uvec = mx.array(u_np, dtype=dt)
                u1 = u2 = None
            else:
                uvec = None
                u1, u2 = _ld_coeffs_batch(self.limb_dark, u_np)
                u1, u2 = col(u1), col(u2)

            if self.integration == "contact":
                a_sky_np, b_np = contact_geometry(
                    cols["a"], ecc, ecc * np.sin(w), np.cos(inc),
                    sqrt=np.sqrt, maximum=np.maximum)
                cs = contact_offsets(rp, col(a_sky_np), col(b_np))
                # (7) self._t_mx already holds the (re-centred) grid on
                # the device -- re-uploading it per call is a pure waste
                # on the surface the docs tell samplers to use. In contact
                # mode it equals the exposure mid-times.
                T, W = exposure_nodes(self._t_mx[None, :], t0, per,
                                      self.exp_time, cs, self.n_gl,
                                      dtype=dt)
                phi = (2.0 * math.pi) * (T - t0[..., None]) / per[..., None]
                z, front = separation_anchored(phi, k[..., None],
                                               h[..., None], a[..., None],
                                               ci[..., None])
                # every per-set column needs the trailing node axis
                nd = lambda c: None if c is None else c[..., None]
                f = self._photom(z, front, rp[..., None], nd(u1), nd(u2),
                                 nd(fp),
                                 uvec=None if uvec is None
                                 else uvec[:, None, :])
                out = mx.sum(f * W, axis=-1)
            else:
                tt = self._t_mx[None, :]
                phi = (2.0 * math.pi) * (tt - t0) / per
                z, front = separation_anchored(phi, k, h, a, ci)
                out = self._photom(z, front, rp, u1, u2, fp, uvec=uvec)
            res = np.array(out, dtype=np.float64)
        if (self.supersample_factor > 1
                and self.integration == "supersample"):
            res = res.reshape(n, self.t.size,
                              self.supersample_factor).mean(axis=2)
        return res

    def light_curve_mx(self, params) -> mx.array:
        """Supersampled-grid flux as an MLX array (stays in the graph;
        no averaging applied) — for building differentiable pipelines.

        Note this is the *eager* path: unlike ``light_curve`` it does not
        go through the mx.compile'd graph or the fused kernels, because
        it must accept MLX scalars for parameters rather than the Python
        floats those paths trace. Expect roughly an order of magnitude
        less throughput at large N; for bulk evaluation use
        ``light_curve``, and for a sampler use ``metalplanet.anvil``
        (see docs/sampler-integration.md).

        With ``integration="contact"`` the exposure average *is* applied
        here, so the result matches ``light_curve`` one-for-one. With
        ``supersample_factor`` it is not: that mode returns the raw
        supersampled grid, as the summary line says.
        """
        return self._eval(params)
