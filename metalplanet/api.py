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
* supported limb_dark: "uniform", "linear", "quadratic", "polynomial"
  for I(mu)/I0 = 1 - sum_n u_n (1-mu)^n at ANY order (ALFM19's M_n
  recursion, metalplanet/poly.py), and the hybrid laws "hybrid2",
  "hybrid4", "hybrid5" (even powers of mu plus double poles, elementary
  and more accurate than quadratic; metalplanet/hybrid.py), whose ``u``
  holds the shape-basis weights w; batman's non-polynomial laws
  ("nonlinear", "squareroot", ...) are not covered by this formulation;
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
from .metal import _gpu_stream_active, flux_dev_metal, metal_available
from .metal_hybrid import flux_dev_metal_hybrid
from .anchored import anchor_constants_ew, separation_anchored
from .hybrid import LAWS as _HYBRID_LAWS, flux_dev_hybrid
from .poly import flux_dev_poly
from .exposure import (contact_geometry, contact_offsets,
                       contact_offsets_anchored, exposure_nodes)
from .solution import sn_dev
from .trig import sincos

__all__ = ["TransitParams", "TransitModel"]

_SUPPORTED_LD = ("uniform", "linear", "quadratic", "polynomial",
                 "hybrid2", "hybrid4", "hybrid5")
#: laws whose coefficients enter the graph as one vector (uvec) rather
#: than as (u1, u2): the polynomial law at any order, and the hybrid laws
#: at their fixed count of shape weights
_VECTOR_LAWS = {"polynomial": None,
                **{n: law.n_w for n, law in _HYBRID_LAWS.items()}}


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


def _u_vector(u, sets=False):
    """``params.u`` as ONE 1-D sequence of coefficients, whatever it
    arrived as -- or, with ``sets``, as a (n_sets, N) float64 array.

    A list or tuple of numbers, a numpy array and an mx.array all come
    through here, as does a list holding mx.array scalars (the way to
    differentiate one coefficient). The result is a float64 numpy vector
    for host input, the mx.array itself for a whole-array u, and for the
    mixed list the list of its validated scalars -- each converted to the
    model dtype later, on its own, so no entry borrows a sibling's dtype
    (an int mx scalar used to truncate its Python neighbours to 0).
    Anything that is not exactly one axis raises: a (n, 1) column -- a
    loadtxt slice, or a nested list -- used to pass the coefficient
    *count* and then run flux_dev_poly's batched branch as a wrong-order
    model, and a 0-d mx.array died in list() with an opaque IndexError.
    One container rule in one place is what makes "every container" true.

    ``sets`` is the array-valued ``light_curves`` form, whose u may be
    (N,) for all sets or (n_sets, N) per set; it is returned 2-D.
    """
    if u is None:
        v = np.zeros((1, 0) if sets else 0)
    elif isinstance(u, mx.array):
        v = u
    elif (isinstance(u, (list, tuple))
          and any(isinstance(x, mx.array) for x in u)):
        for x in u:
            if isinstance(x, mx.array):
                _need_scalar(x, "u")
            elif np.ndim(x) != 0:
                raise ValueError("u must be a 1-D vector; got a nested "
                                 f"entry of shape {np.shape(x)}")
        v = list(u)
    else:
        v = np.asarray(u, dtype=np.float64)
    if sets:
        v = np.asarray(v, dtype=np.float64)
        if v.ndim == 1:
            v = v[None, :]
        if v.ndim != 2:
            raise ValueError("u must be (N,) or (n_sets, N) here; got shape "
                             f"{tuple(v.shape)}")
        return v
    if np.ndim(v) != 1:
        raise ValueError(f"u must be a 1-D vector; got shape {tuple(np.shape(v))}")
    return v


def _ld_coeffs(params, conv=float, u=None) -> tuple[float, float]:
    """(u1, u2) for the quadratic core, validated. ``conv`` maps each
    value: float for the traced-scalar paths, a graph-preserving cast for
    light_curve_mx's differentiable one. ``u`` is the already-normalised
    vector when the caller has it."""
    law = params.limb_dark
    if u is None:
        u = _u_vector(params.u)
    if law == "uniform":
        if len(u) != 0:
            raise ValueError("uniform limb darkening takes no coefficients")
        return conv(0.0), conv(0.0)
    if law == "linear":
        if len(u) != 1:
            raise ValueError("linear limb darkening takes 1 coefficient")
        return conv(u[0]), conv(0.0)
    if law == "quadratic":
        if len(u) != 2:
            raise ValueError("quadratic limb darkening takes 2 coefficients")
        return conv(u[0]), conv(u[1])
    if law == "polynomial":
        if len(u) == 0:
            raise ValueError("polynomial limb darkening needs >= 1 "
                             "coefficient (use 'uniform' for none)")
        return None, None          # handled by the polynomial core
    if law in _HYBRID_LAWS:
        n_w = _HYBRID_LAWS[law].n_w
        if len(u) != n_w:
            raise ValueError(f"{law} takes {n_w} weights; got {len(u)}")
        return None, None          # handled by the hybrid core
    raise ValueError(
        f"limb_dark {law!r} not supported; choose from {_SUPPORTED_LD}. "
        "Non-polynomial laws (nonlinear, squareroot, logarithmic) are "
        "outside the ALFM19 formulation.")


def _need_scalar(x, name):
    """Every per-set field is one number. A vector would run -- each time
    sample at its own value -- or die deep in a graph with an opaque
    reshape error; a (1,) or (1, 1) would change the output's shape."""
    if x.ndim != 0:
        raise ValueError(f"{name} must be a scalar; got shape {x.shape}")


def _scalar_value(x):
    """The Python float of a 0-d mx.array, or None if it is traced.

    Under mx.compile / mx.vmap an array cannot be read -- MLX raises on
    the attempt -- while eagerly, and under mx.grad (whose inputs are plain
    arrays), it can; the read is free for a leaf and otherwise evaluates
    the upstream graph. With the shape already checked, that refusal is
    the only ValueError float() can raise here. The MLX dependency is this
    one place (pinned by test_light_curve_mx.test_scalar_value_helper).
    """
    try:
        return float(x)
    except ValueError:
        return None


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


_ECC_MSG = "eccentricity must be in [0, 1); got {}"


def _check_ecc(ecc):
    """Reject eccentricities the model cannot represent, before they reach
    a graph that would return a plausible flat curve (e >= 1) or an
    all-NaN row (e < 0) with nothing pointing at the parameter. A Python
    number is checked as a scalar; the batched path passes an array."""
    if not isinstance(ecc, np.ndarray):
        if not 0.0 <= ecc < 1.0:
            raise ValueError(_ECC_MSG.format(ecc))
        return
    bad = ~((ecc >= 0.0) & (ecc < 1.0))
    if bool(np.any(bad)):
        raise ValueError(_ECC_MSG.format(
            f"{ecc[bad][:4]} (and possibly more)"))


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


def _unpack_ld(vec, tail):
    """(u1, u2, uvec, fp) from the trailing limb-darkening + fp args."""
    if vec:
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
        # vector laws (polynomial, hybrid): the coefficient count is fixed
        # per model, baked into the compiled graph
        self._n_vec = (len(_u_vector(params.u))
                       if params.limb_dark in _VECTOR_LAWS else 0)
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

    def _photom(self, z, front, rp, u1, u2, fp, uvec=None):
        if self.transittype == "primary":
            z_eff = mx.where(front, z, 2.0 + z)
            if uvec is not None:          # a vector law: polynomial or hybrid
                if self.limb_dark == "polynomial":
                    return 1.0 + flux_dev_poly(z_eff, rp, uvec,
                                               n_max=self._n_vec)
                return 1.0 + self._hybrid_dev(z_eff, rp, uvec)
            core = flux_dev_metal if self.use_metal else flux_dev
            return 1.0 + core(z_eff, rp, u1, u2)
        z_eff = mx.where(front, 2.0 + z, z)
        s0d, _, _ = sn_dev(z_eff, rp)
        # visible fraction of the (uniform) planet disk
        return 1.0 + fp * (1.0 + s0d / (math.pi * rp * rp))

    def _hybrid_dev(self, z, rp, uvec):
        """F - 1 for a hybrid law on separations ``z``: the fused z-input
        kernel (metal_hybrid) wherever it can run -- the role flux_dev_metal
        plays for the quadratic law -- and hybrid.py's graph otherwise.

        The decision is _kernel_usable's, taken here, inside the trace,
        from the active stream (mx.compile keeps one trace per stream, so
        a graph first traced on the CPU stream is retraced, kernel and
        all, on the GPU; no cache key is needed). An fp32 model on the CPU
        stream, a Metal-less machine and use_metal=False therefore run one
        and the same graph, bitwise.

        The kernel takes (n, m) points with r and the weights per row, so
        the shapes are flattened onto that: one row for a single parameter
        set (any node axes folded into m), n rows for light_curves' sets.
        r goes through as is: the entry point's canonicaliser takes a
        scalar or anything of size n (light_curves' contact path hands it
        (n, 1, 1)). One radius with one set of weights -- (n_w,) or a
        (1, n_w) row -- is one row of the grid, never a row per point.
        """
        law = self.limb_dark
        if not self._kernel_usable():
            return flux_dev_hybrid(z, rp, uvec, law)
        n_w = _HYBRID_LAWS[law].n_w
        one_set = ((not isinstance(rp, mx.array) or rp.size == 1)
                   and uvec.size == n_w)
        if one_set:                        # (n_w,) or a (1, n_w) row
            out = flux_dev_metal_hybrid(mx.reshape(z, (-1,)), rp, law,
                                        mx.reshape(uvec, (-1,)))
        else:
            n = z.shape[0]
            w = (mx.reshape(uvec, (uvec.shape[0], -1)) if uvec.ndim >= 2
                 else uvec)
            out = flux_dev_metal_hybrid(mx.reshape(z, (n, -1)), rp, law, w)
        return mx.reshape(out, z.shape)

    def _kernel_usable(self) -> bool:
        """Can a fused kernel run *here*: fp32, a usable Metal device, the
        GPU stream active, and not switched off. Asked before tracing by
        _get_compiled for the fused *model* kernel (whose reach -- primary
        transits, quadratic limb darkening, no contact rule -- is stated
        beside the branches it governs, and whose answer is in the cache
        key), and inside the trace by _hybrid_dev for the hybrid z-kernel.
        """
        if not self.use_metal:
            return False
        if self.dtype != mx.float32:
            return False
        return metal_available() and _gpu_stream_active()

    def _get_compiled(self, circular: bool, ew: bool = False):
        """The compiled model graph. ``ew`` selects, for the eccentric
        graphs, (e, w [rad]) inputs in place of (k, h) -- light_curve_mx's
        differentiable route, exact at e = 0 (anchored.anchor_constants_ew).
        With ew=False every graph is exactly what it always was."""
        vec = bool(self._n_vec)
        # The kernel decision depends on the *active* stream, so it is part
        # of the key: a graph first built under the CPU stream (an fp64
        # gradient, say) must not be the one every later GPU call reuses.
        # It enters the key only where a kernel branch below is reachable,
        # so a graph the kernel never serves is compiled once, not once
        # per stream. This is THE statement of what the *model* kernel
        # serves: primary transits with quadratic limb darkening, on the
        # (k, h) or circular graph, without the contact rule. Every other
        # fp32 graph reaches a z-input kernel (quadratic or hybrid) through
        # _photom, decided inside the trace, which mx.compile keeps per
        # stream: no key needed.
        kernel_branch = (not ew and not vec and self.transittype == "primary"
                         and self.integration != "contact")
        kern = kernel_branch and self._kernel_usable()
        key = "ew" if ew else (circular, kern)
        fn = self._compiled.get(key)
        if fn is not None:
            return fn
        t = self._t_mx
        if ew:
            circular = False

        def consts(k, h):
            """Keyword for the anchored helpers: nothing in (k, h) mode,
            so those graphs are unchanged; the (e, w) constants else."""
            return dict(consts=anchor_constants_ew(k, h)) if ew else {}

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
                    u1, u2, uv, fp = _unpack_ld(vec, ld_fp)
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
                    u1, u2, uv, fp = _unpack_ld(vec, ld_fp)
                    kw = consts(k, h)
                    if ew:
                        e, esw = kw["consts"][0], kw["consts"][2]
                    else:
                        e = k * k + h * h
                        esw = h * mx.sqrt(mx.maximum(e, 1e-30))  # e sin w
                    b_conj = contact_geometry(a, e, esw, ci)[1]
                    cs = contact_offsets_anchored(rp, a, b_conj, k, h, ci,
                                                  **kw)
                    T, W = exposure_nodes(t, t0, per, ex, cs, n_gl,
                                          dtype=self.dtype)
                    phi = (2.0 * math.pi) * (T - t0) / per
                    z, front = separation_anchored(phi, k, h, a, ci, **kw)
                    return _avg(z, front, rp, u1, u2, fp, uv, W)

        elif circular and vec:
            def raw(t0, per, a, b, rp, uv, fp):
                phase = (2.0 * math.pi) * (t - t0) / per
                sphi, cphi = sincos(phase)
                z = mx.sqrt(mx.maximum((a * sphi) ** 2 + (b * cphi) ** 2,
                                       1e-24))
                return self._photom(z, cphi > 0.0, rp, None, None, fp,
                                    uvec=uv)
        elif vec:
            def raw(t0, per, a, k, h, ci, rp, uv, fp):
                phi = (2.0 * math.pi) * (t - t0) / per
                z, front = separation_anchored(phi, k, h, a, ci,
                                               **consts(k, h))
                return self._photom(z, front, rp, None, None, fp, uvec=uv)
        elif circular and kern:
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
        elif kern and not ew:
            # Whole eccentric model in one kernel. (Not on the (e, w) route:
            # the kernel skips its seven eccentric-only gradient slots on
            # e == 0 chains, which is exact for (k, h) -- whose Jacobian
            # vanishes there -- but not for (e, w), where d(e cos w)/de =
            # cos w. The graph below carries every slot.) The anchored Kepler
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
                z, front = separation_anchored(phi, k, h, a, ci,
                                               **consts(k, h))
                return self._photom(z, front, rp, u1, u2, fp)

        fn = mx.compile(raw)
        self._compiled[key] = fn
        return fn

    def _model_eval(self, params, uv) -> mx.array:
        """One parameter set -> flux, through the compiled graphs. The one
        path behind light_curve and light_curve_mx.

        A field holding an mx.array stays in the graph (gradients flow); a
        Python number is folded on the host in fp64 and cast once, so with
        all-Python fields the arguments -- b = a cos i, (k, h) =
        sqrt(e) (cos w, sin w) -- are formed exactly as they always were.

        Routing follows ``ecc``: a Python 0 takes the circular graph, a
        Python e > 0 the (k, h) graph (smooth in w for fixed e), an
        mx.array the (e, w) graph, which alone is exact in d/de at e = 0
        (anchored.anchor_constants_ew) and never uses the fused *model*
        kernel (its photometry still takes the z-input kernel on an fp32
        GPU model).
        """
        dt = self.dtype

        def is_arr(x):
            return isinstance(x, mx.array)

        # Shapes first, once, before any routing: every per-set field is
        # one number (``uv`` is u already normalised by _check_law). A
        # route that never reads a field (w on a circular orbit) still
        # rejects a bad one.
        for name in self._BATCH_KEYS:
            x = getattr(params, name)
            if is_arr(x):
                _need_scalar(x, name)

        def cast(x):
            if is_arr(x):
                if x.dtype == dt:
                    return x
                with mx.stream(mx.cpu):   # an fp64 value may not touch Metal
                    return x.astype(dt)
            if isinstance(x, np.ndarray):          # a host u vector
                return mx.array(x, dtype=dt)
            if isinstance(x, list):                # mixed u: entry by entry
                return mx.stack([cast(y) for y in x])
            return mx.array(float(x), dtype=dt)

        def rad(x):                       # degrees -> radians
            if is_arr(x):
                return cast(x) * (math.pi / 180.0)
            return cast(math.radians(float(x)))   # on the host, in fp64

        t0 = params.t0
        if is_arr(t0):
            # the reference-time subtraction in fp64, then the model dtype
            with mx.stream(mx.cpu):
                t0_off = (t0.astype(mx.float64) - self._t_ref).astype(dt)
        else:
            t0_off = cast(t0 - self._t_ref)

        inc = params.inc
        if is_arr(inc):
            ci = sincos(rad(inc))[1]      # fp64-accurate (MLX's cos is not)
        else:
            ci_py = math.cos(math.radians(float(inc)))
            ci = cast(ci_py)
        fp = cast(0.0 if params.fp is None else params.fp)
        per, a, rp = cast(params.per), cast(params.a), cast(params.rp)

        if self._n_vec:
            ld = (cast(uv),)
        else:
            ld = _ld_coeffs(params, conv=cast, u=uv)

        ecc = params.ecc
        if is_arr(ecc):
            # Validate the value the GRAPH will see: an fp64 e just below
            # 1 can round to exactly 1 in fp32. Readable (eagerly, under
            # mx.grad) it raises like a number; traced (mx.compile,
            # mx.vmap) it cannot, and an out-of-range e instead makes the
            # output NaN -- and, through the factor, every gradient.
            e = cast(ecc)
            ev = _scalar_value(e)
            if ev is not None:
                _check_ecc(ev)
            f = self._get_compiled(False, ew=True)(
                t0_off, per, a, e, rad(params.w), ci, rp, *ld, fp)
            if ev is None:
                ok = mx.logical_and(e >= 0.0, e < 1.0)
                f = f * mx.where(ok, 1.0, float("nan")).astype(dt)
            return f
        ev = float(ecc)
        _check_ecc(ev)
        if ev == 0.0:                     # circular: w is irrelevant
            b = (a * ci if is_arr(params.a) or is_arr(inc)
                 else cast(float(params.a) * ci_py))
            return self._get_compiled(True)(t0_off, per, a, b, rp, *ld, fp)
        sq = math.sqrt(ev)
        if is_arr(params.w):
            sw, cw = sincos(rad(params.w))
            k, h = sq * cw, sq * sw
        else:
            w = math.radians(float(params.w))
            k, h = cast(sq * math.cos(w)), cast(sq * math.sin(w))
        return self._get_compiled(False)(t0_off, per, a, k, h, ci, rp,
                                         *ld, fp)

    # -- batman-compatible surface ----------------------------------------

    def _check_law(self, params, sets=False):
        """The limb-darkening law AND its order are fixed per model: the
        order is baked into the compiled graph (and into the g_n affine
        map), so a changed count would be silently truncated by the zip in
        flux_dev_poly rather than raising. Returns the normalised u
        (_u_vector: shape judged before the count), so each entry point
        normalises once and threads it through."""
        if params.limb_dark != self.limb_dark:
            raise ValueError(
                "limb-darkening law changed since model construction; "
                "build a new TransitModel")
        u = _u_vector(params.u, sets=sets)
        n_u = u.shape[-1] if sets else len(u)
        if self._n_vec and n_u != self._n_vec:
            raise ValueError(
                f"limb-darkening coefficient count changed since model "
                f"construction ({self._n_vec} -> {n_u}); build a new "
                f"TransitModel")
        return u

    def light_curve(self, params) -> np.ndarray:
        """Model flux at the times given at construction (numpy array)."""
        uv = self._check_law(params)
        if self._stream is not None:
            with mx.stream(self._stream):
                f = self._model_eval(params, uv)
                out = np.array(f, dtype=np.float64)
        else:
            out = np.array(self._model_eval(params, uv), dtype=np.float64)
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
            rows = []
            for p in seq:
                # the same validation the single-set path performs: an
                # unchecked set returns a plausible but wrong curve, or a
                # silent NaN row that a sampler reads as -inf
                uv = self._check_law(p)
                _ld_coeffs(p, u=uv)
                rows.append([float(x) for x in uv])
            u = np.array(rows, dtype=np.float64).reshape(len(seq), -1)
            _check_ecc(cols["ecc"])
            return cols, u
        p = params_seq                       # array-valued TransitParams
        u = self._check_law(p, sets=True)    # (1 or n_sets, N)
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
        if u.shape[0] not in (1, n):
            raise ValueError(f"inconsistent parameter-array lengths: u has "
                             f"{u.shape[0]} sets, the other fields {n}")
        u = np.broadcast_to(u, (n, u.shape[1]))
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
            if self._n_vec:
                uvec = mx.array(u_np, dtype=dt)
                u1 = u2 = None
            else:
                uvec = None
                u1, u2 = _ld_coeffs_batch(self.limb_dark, u_np)
                u1, u2 = col(u1), col(u2)

            if self.integration == "contact":
                b_np = contact_geometry(
                    cols["a"], ecc, ecc * np.sin(w), np.cos(inc),
                    sqrt=np.sqrt, maximum=np.maximum)[1]
                # exact contacts (linearised ones returned as is at e = 0)
                cs = contact_offsets_anchored(rp, a, col(b_np), k, h, ci)
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
        """Model flux as an MLX array that stays in the graph, for building
        on in MLX -- differentiable in the parameters.

        Any ``TransitParams`` field may be an ``mx.array`` scalar (t0, per,
        rp, a, inc, ecc, w, fp, and ``u`` or its entries), and gradients
        flow to every such field. Python numbers stay constants, folded on
        the host exactly as ``light_curve`` folds them. With ``ecc`` a
        Python number the result is ``light_curve``'s bit for bit when the
        arrays are among t0, per, rp, fp, u and -- on an eccentric orbit
        -- a, which enter the graph as they are. An array ``inc`` or
        ``w``, or ``a`` on a circular orbit, is combined in-graph in the
        model dtype (b = a cos i, (k, h) = sqrt(e) (cos w, sin w)), which
        on an fp32 model is within ~1 ulp of the host fold (see
        ``_model_eval``). Every field must be a scalar and ``u`` a flat
        vector -- list, tuple, numpy or mx.array, or a list holding
        mx.array scalars; a wrong shape raises before anything is built.

        Returns the flux at the times given at construction, exposure-
        averaged for ``integration="contact"``. With ``supersample_factor``
        it is the raw supersampled grid -- reshape to (n_times, factor) and
        average, as ``light_curve`` does.

        Differentiable details:

        * an array-valued ``ecc`` enters as (e, w) directly
          (``anchor_constants_ew``), so d/d(ecc) is finite and correct at
          e = 0 -- the one-sided derivative, since e >= 0 -- and d/dw is
          exactly 0 there. Out of [0, 1) it raises ValueError whenever the
          value can be read (eagerly, or under ``mx.grad``); when it is
          traced -- under ``mx.vmap`` or a caller's ``mx.compile`` -- it
          cannot raise, and instead the output and every gradient are NaN.
        * a Python ``ecc`` keeps ``light_curve``'s own graph -- circular at
          e = 0, (k, h) above it -- so a fit that differentiates, say, only
          rp and t0 runs exactly as fast as ``light_curve``, fused kernel
          included. Only an array-valued ``ecc`` takes the (e, w) graph.
          For the quadratic law that forgoes the fused *model* kernel (see
          _get_compiled): measured 3-4x slower on an fp32 GPU model at
          2e6 points (3.0 vs 0.75-0.91 ms). The hybrid laws' photometry
          runs in their z-input kernel on every graph, this one included.
        * ``u`` may be a list, a numpy array, or an mx.array vector.
        * the contact-rule nodes move with the parameters, so a gradient
          is that of the quadrature actually evaluated.
        * pass t0 as fp64 (a Python float or an fp64 array): an absolute
          BJD stored in fp32 has already lost its precision (0.25 d at
          2.45e6), before the model subtracts its reference time in fp64.
        * under ``mx.grad`` MLX needs the CPU stream for *any* float64
          input. With an fp32 (GPU) model, differentiate float32 fields on
          the GPU; for an fp64 field such as an absolute t0, take the
          gradient inside ``with mx.stream(mx.cpu):`` (or use an fp64
          model, the default, which lives there anyway).

        Precision: the graph is built on the model's own stream -- for the
        default ``dtype=mx.float64`` that is the CPU, as MLX has no float64
        on Metal. The result is a lazy fp64 array, so whatever you build on
        it must stay on the CPU too: wrap your own ops (and ``mx.grad``) in
        ``with mx.stream(mx.cpu):``.
        """
        uv = self._check_law(params)
        ctx = (mx.stream(self._stream) if self._stream is not None
               else _nullcontext())
        with ctx:
            return self._model_eval(params, uv)

