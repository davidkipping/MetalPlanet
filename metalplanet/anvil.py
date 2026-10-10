"""anvil (formerly applemcmc) integration: the batched transit model and
a synthetic transit-fitting target.

``make_quad_transit_flux`` returns the engine-contract model function
    model_fn(v: (n_chains, 8), x: (2, m)) -> (n_chains, m) flux deviation
and is pure metalplanet (mlx + numpy only). ``make_target`` assembles the
full sampling problem (chunked likelihood + transform + synthetic data);
the engine is imported lazily — as ``anvil`` if available, falling back
to its former name ``applemcmc`` — so the core package stays standalone.

Model-space parameters v (all O(1), float32-safe offsets):
    0: t0_off  mid-transit offset from t0_ref [days]
    1: p_off   period offset from period_ref [days]
    2: r       radius ratio Rp/R*
    3: b       impact parameter
    4: a       scaled semi-major axis a/R*
    5: q1      Kipping (2013) triangular LD
    6: q2      Kipping (2013) triangular LD
    7: df0     baseline flux deviation from 1

``x`` rows: dt (per-orbit residual times, float64-preprocessed) and k
(orbit numbers). Predicts the flux *deviation* df0 + (F - 1), matching
data y - 1.
"""

from __future__ import annotations

import math

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .flux import flux_dev
from .ld import q_to_u, q_to_u_np, u_to_q_np
from .orbit import epoch_center_times, separation_circular, tau_from_epochs

__all__ = ["make_quad_transit_flux", "make_target", "QuadTransitTarget",
           "import_engine", "make_ecc_transit_flux", "make_ecc_target",
           "EccTransitTarget", "ecc_constraint_penalty",
           "PenalizedLogLike", "PARAM_NAMES", "PARAM_NAMES_ECC",
           "make_transit_target", "TransitTarget", "DEFAULT_BOUNDS"]

PARAM_NAMES = ["t0_off", "p_off", "r", "b", "a", "q1", "q2", "df0"]


def import_engine():
    """Import the sampling engine under either of its names.

    Returns (engine_module, ChunkedGaussianLogLike). Tries ``anvil``
    first (the engine's new name), then ``applemcmc``.
    """
    try:
        import anvil as engine
        from anvil.precision import ChunkedGaussianLogLike
    except ImportError:
        import applemcmc as engine
        from applemcmc.precision import ChunkedGaussianLogLike
    return engine, ChunkedGaussianLogLike


def make_quad_transit_flux(period_ref: float, analytic_vjp: bool = True,
                           core: str = "metal"):
    """Engine-contract batched model. period_ref is baked in as a plain
    Python float (a numpy scalar here would silently force-evaluate the
    graph mid-compile).

    core selects the photometric implementation (identical forward math):
      "metal"    — fused Metal kernels on the fp32 GPU stream, silently
                   falling back to "analytic" for fp64/CPU/unsupported
                   machines (the default; anvil's fp64 verification path
                   and CPU data generation ride the fallback);
      "analytic" — MLX graph forward + ALFM19 closed-form backward;
      "autodiff" — pure reverse-mode autodiff (Stage 1 oracle).
    analytic_vjp=False is a back-compat alias for core="autodiff"."""
    period_ref = float(period_ref)
    if not analytic_vjp:
        core = "autodiff"
    if core == "metal":
        from .anchored import pack_orbit_constants
        from .metal import (_gpu_stream_active, flux_dev_metal,
                            make_model_core_metal, metal_available)
        graph_fn = _build_graph_model(period_ref, flux_dev_metal)
        if not metal_available():
            return graph_fn
        model_core = make_model_core_metal(period_ref)

        def flux_dev_fn(v: mx.array, x: mx.array) -> mx.array:
            # Fused path: whole orbit + photometry in one kernel. The
            # circular MODE is kept (8 parameters, b sampled directly) but
            # runs on the same kernel as the eccentric model, with
            # k = h = 0 and cos i = b / a — exact, since the anchored orbit
            # degenerates to the circular one, and the kernel skips the
            # Kepler solve on e == 0 chains. Anything the kernel can't
            # serve (fp64, CPU stream — anvil's data generation and
            # verification paths) falls back to the graph model.
            if (v.dtype == mx.float32 and x.dtype == mx.float32
                    and _gpu_stream_active()):
                u1, u2 = q_to_u(v[:, 5], v[:, 6])
                zero = mx.zeros_like(v[:, 3])
                orb = pack_orbit_constants(zero, zero, v[:, 3] / v[:, 4])
                dev = model_core(x, v[:, 0], v[:, 1], v[:, 2], v[:, 4],
                                 orb, u1, u2)
                return v[:, 7:8] + dev
            return graph_fn(v, x)

        return flux_dev_fn
    elif core == "analytic":
        from .vjp import flux_dev_analytic as core_fn
    elif core == "autodiff":
        core_fn = flux_dev
    else:
        raise ValueError(f"unknown core {core!r}")
    return _build_graph_model(period_ref, core_fn)


def _build_graph_model(period_ref: float, core):
    def flux_dev_fn(v: mx.array, x: mx.array) -> mx.array:
        dt, k = x[0], x[1]
        t0_off, p_off = v[:, 0:1], v[:, 1:2]
        r, b, a = v[:, 2:3], v[:, 3:4], v[:, 4:5]
        q1, q2 = v[:, 5:6], v[:, 6:7]
        df0 = v[:, 7:8]
        u1, u2 = q_to_u(q1, q2)
        period = period_ref + p_off
        tau = tau_from_epochs(dt[None, :], k[None, :], t0_off, p_off, period)
        z = separation_circular(tau, period, b, a)
        return df0 + core(z, r, u1, u2)

    return flux_dev_fn


@dataclass
class QuadTransitTarget:
    """Synthetic quadratic-LD transit-fitting problem (applemcmc-ready)."""

    target: object                # applemcmc.TransformedLogDensity
    transform: object             # applemcmc.Transform
    loglike: object               # applemcmc.ChunkedGaussianLogLike
    truth_model: np.ndarray       # (8,) true params, model units
    t_ref: float
    t_model: np.ndarray
    y: np.ndarray


def make_target(
    n_data: int = 100_000,
    yerr: float = 5e-4,
    seed: int = 0,
    policy=None,
    baseline_days: float = 90.0,
    analytic_vjp: bool = True,
    core: str = "metal",
):
    """Build the synthetic problem in well-conditioned units.

    Absolute BJD-like times (~2.457e6 days) are generated and reduced in
    float64 on the CPU; only per-orbit residuals, orbit numbers, and
    baseline-subtracted fluxes ever reach float32.
    """
    applemcmc, ChunkedGaussianLogLike = import_engine()

    rng = np.random.default_rng(seed)

    t_ref = 2_457_000.0
    t_abs = t_ref + np.sort(rng.uniform(0.0, baseline_days, size=n_data))
    t_model = t_abs - t_ref

    # truth: warm-Jupiter-ish, quadratic LD via Kipping q
    u1_true, u2_true = 0.40, 0.25
    q1_true, q2_true = u_to_q_np(u1_true, u2_true)
    truth = np.array([
        1.2345,       # t0 [days since t_ref]
        3.456,        # period [days]
        0.10,         # r
        0.30,         # b
        8.80,         # a/R*
        float(q1_true),
        float(q2_true),
        1.0,          # baseline flux
    ])

    # generate data through the model itself, in float64 on the CPU
    t0_ref = float(truth[0]) - 0.009
    period_ref = float(truth[1]) + 0.0005
    truth_model = np.array([
        truth[0] - t0_ref, truth[1] - period_ref,
        truth[2], truth[3], truth[4], truth[5], truth[6],
        truth[7] - 1.0,
    ])
    x64 = epoch_center_times(t_model, t0_ref=t0_ref, period_ref=period_ref)
    model_fn = make_quad_transit_flux(period_ref=period_ref,
                                      analytic_vjp=analytic_vjp, core=core)
    with mx.stream(mx.cpu):
        dev_true = np.array(
            model_fn(mx.array(truth_model[None, :], dtype=mx.float64),
                     mx.array(x64, dtype=mx.float64))[0],
            dtype=np.float64,
        )
    y = 1.0 + dev_true + yerr * rng.standard_normal(n_data)
    y_fit = y - 1.0
    yerr_arr = np.full(n_data, yerr)

    transform = applemcmc.Transform([
        applemcmc.ParamSpec("t0_off", lo=-0.5, hi=0.5,
                            report_offset=t_ref + t0_ref),
        applemcmc.ParamSpec("p_off", lo=-0.05, hi=0.05,
                            report_offset=period_ref),
        applemcmc.ParamSpec("r", lo=0.01, hi=0.5),
        applemcmc.ParamSpec("b", lo=0.0, hi=0.9),
        applemcmc.ParamSpec("a", lo=2.0, hi=50.0),
        applemcmc.ParamSpec("q1", lo=0.0, hi=1.0),
        applemcmc.ParamSpec("q2", lo=0.0, hi=1.0),
        applemcmc.ParamSpec("df0", lo=-0.01, hi=0.01, report_offset=1.0),
    ])
    loglike = ChunkedGaussianLogLike(model_fn, x64, y_fit, yerr_arr, policy)
    target = applemcmc.TransformedLogDensity(
        loglike, transform, model_log_prob_hi=loglike.hi)
    return QuadTransitTarget(
        target=target, transform=transform, loglike=loglike,
        truth_model=truth_model, t_ref=t_ref, t_model=t_model, y=y,
    )



# ---------------------------------------------------------------------------
# real-data entry point
# ---------------------------------------------------------------------------

#: default ParamSpec boxes, in model units. ``b`` reaches past 1 so
#: grazing geometries are inside the box; widen or narrow per target.
DEFAULT_BOUNDS = {
    "t0_off": (-0.5, 0.5), "p_off": (-0.05, 0.05),
    "r": (0.005, 0.5), "b": (0.0, 1.2), "a": (1.5, 200.0),
    "q1": (0.0, 1.0), "q2": (0.0, 1.0), "df0": (-0.01, 0.01),
    "secosw": (-0.95, 0.95), "sesinw": (-0.95, 0.95),
}


@dataclass
class TransitTarget:
    """A fitting problem over REAL data, ready for anvil.

    ``target`` is the TransformedLogDensity to sample; ``x64`` and
    ``y_fit`` are the conditioned inputs (epoch-centered times, flux minus
    one) should a different likelihood -- an anvil-gp GP over the same
    mean function, say -- want to wrap the same model.
    """

    target: object                # applemcmc.TransformedLogDensity
    transform: object             # applemcmc.Transform
    loglike: object               # ChunkedGaussianLogLike (maybe penalized)
    model_fn: object              # the batched (v, x) -> flux deviation
    x64: np.ndarray               # (2, n) epoch-centered times
    y_fit: np.ndarray             # y - 1
    yerr: np.ndarray
    t_ref: float                  # float64 time origin subtracted first
    t0_ref: float                 # t0 reference (model units, from t_ref)
    period_ref: float
    param_names: list
    eccentric: bool

    def model_params(self, t0, period, **kw):
        """Physical values -> the model-space vector the sampler uses.

        Handles the two offset parameters so callers never have to
        remember which reference is subtracted from what.
        """
        vals = dict(kw)
        vals["t0_off"] = float(t0) - self.t_ref - self.t0_ref
        vals["p_off"] = float(period) - self.period_ref
        vals.setdefault("df0", 0.0)
        missing = [n for n in self.param_names if n not in vals]
        if missing:
            raise ValueError(f"missing parameters: {missing}")
        return np.array([float(vals[n]) for n in self.param_names])


def make_transit_target(t, y, yerr, t0_guess, period_guess,
                        eccentric: bool = False, bounds=None, policy=None,
                        core: str = "metal", strength: float = 1e6):
    """Build an anvil fitting problem from REAL photometry.

    This is the entry point for mission data (Kepler, TESS, ...). It owns
    the conditioning that the float32 sampling path depends on, so a
    caller never has to rediscover it:

    * absolute times are reduced by a float64 ``t_ref`` and then turned
      into (per-orbit residual, orbit number) by ``epoch_center_times``
      -- raw BJD cannot survive float32 (0.25 d ulp at 2.457e6);
    * flux is fit as ``y - 1`` so the graph carries O(depth) numbers;
    * ``t0_guess``/``period_guess`` become the references the sampler
      perturbs around, and ``report_offset`` puts the reported values back
      on the original time system;
    * for ``eccentric=True`` the joint-constraint barrier is attached.
      MetalPlanet clamps every numerical hazard, so an unphysical geometry
      otherwise returns an ordinary finite log-likelihood and the chain
      samples an improper posterior in silence -- omitting the barrier is
      the single easiest way to get a wrong eccentric fit.

    Args:
        t, y, yerr: photometry, any consistent time system; y normalized
            to ~1 out of transit.
        t0_guess, period_guess: references in the SAME system as ``t``.
            They need only be good to the width of the ``t0_off``/``p_off``
            boxes (0.5 d and 0.05 d by default).
        eccentric: 10-parameter model with (sqrt(e) cos w, sqrt(e) sin w)
            instead of the 8-parameter circular one.
        bounds: per-parameter (lo, hi) overrides on DEFAULT_BOUNDS. The
            defaults cap r at 0.5, b at 1.2 and a at 200; an occultor
            larger than the star (a white-dwarf host) needs them widened,
            e.g. ``{"r": (1.0, 12.0), "b": (0.0, 13.0), "a": (1.5, 1000.0)}``.
            The model takes any r > 0 (fp32 error grows ~ r^2; README,
            "rp > 1").
        strength: barrier weight for the eccentric constraints.
    """
    applemcmc, ChunkedGaussianLogLike = import_engine()
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    yerr = (np.full(t.size, float(yerr), dtype=np.float64)
            if np.ndim(yerr) == 0 else np.asarray(yerr, dtype=np.float64))
    if not (t.shape == y.shape == yerr.shape):
        raise ValueError(f"t, y, yerr must match in shape; got {t.shape}, "
                         f"{y.shape}, {yerr.shape}")
    if t.size == 0:
        raise ValueError("no data points")
    if not np.all(np.isfinite(t) & np.isfinite(y) & np.isfinite(yerr)):
        raise ValueError("t, y and yerr must all be finite "
                         "(mask your data before fitting)")
    if np.any(yerr <= 0.0):
        raise ValueError("yerr must be positive")
    if np.any(np.diff(t) < 0.0):
        order = np.argsort(t)
        t, y, yerr = t[order], y[order], yerr[order]

    # float64 host conditioning, in the order that matters
    t_ref = float(np.floor(t.min()))
    t_model = t - t_ref
    t0_ref = float(t0_guess) - t_ref
    period_ref = float(period_guess)
    x64 = epoch_center_times(t_model, t0_ref=t0_ref, period_ref=period_ref)
    y_fit = y - 1.0

    names = PARAM_NAMES_ECC if eccentric else PARAM_NAMES
    lims = dict(DEFAULT_BOUNDS)
    for k, v in (bounds or {}).items():
        if k not in lims:
            raise ValueError(f"unknown parameter {k!r}; expected "
                             f"{sorted(lims)}")
        lims[k] = v
    report = {"t0_off": t_ref + t0_ref, "p_off": period_ref, "df0": 1.0}
    transform = applemcmc.Transform([
        applemcmc.ParamSpec(n, lo=lims[n][0], hi=lims[n][1],
                            **({"report_offset": report[n]}
                               if n in report else {}))
        for n in names])

    model_fn = (make_ecc_transit_flux(period_ref=period_ref, core=core)
                if eccentric
                else make_quad_transit_flux(period_ref=period_ref,
                                            core=core))
    loglike = ChunkedGaussianLogLike(model_fn, x64, y_fit, yerr, policy)
    if eccentric:
        loglike = PenalizedLogLike(
            loglike, lambda v: ecc_constraint_penalty(v, strength=strength))
    target = applemcmc.TransformedLogDensity(
        loglike, transform, model_log_prob_hi=loglike.hi)
    return TransitTarget(
        target=target, transform=transform, loglike=loglike,
        model_fn=model_fn, x64=x64, y_fit=y_fit, yerr=yerr, t_ref=t_ref,
        t0_ref=t0_ref, period_ref=period_ref, param_names=list(names),
        eccentric=eccentric)


# ---------------------------------------------------------------------------
# synthetic targets (injection-recovery)
# ---------------------------------------------------------------------------

PARAM_NAMES_ECC = ["t0_off", "p_off", "r", "b", "a", "q1", "q2",
                   "secosw", "sesinw", "df0"]

#: numerical validity ceiling of the solve (docs/eccentric-kernel-notes.md:
#: the Markley + one-refinement scheme holds |dE| <= 4.1e-7 at e = 0.999).
E_MAX_NUMERICAL = 0.999


def _ecc_orbit_from_v(v, e_max: float = E_MAX_NUMERICAL):
    """Sampler coordinates -> (k, h, ci, e) for the anchored orbit.

    (secosw, sesinw) is shrunk radially onto the disc of radius
    sqrt(e_max) so the solve never sees an unsupported eccentricity; the
    *physical* limits (periastron clearance, cos i <= 1) are a prior
    concern and live in ``ecc_constraint_penalty``, not here. The shrink
    denominator is set to 1 where inactive rather than floored, per the
    masked-division rule.
    """
    k_raw, h_raw = v[:, 7], v[:, 8]
    e_raw = k_raw * k_raw + h_raw * h_raw
    big = e_raw > e_max
    den = mx.where(big, e_raw, mx.ones_like(e_raw))
    sc = mx.where(big, mx.sqrt(e_max / den), mx.ones_like(e_raw))
    k, h = k_raw * sc, h_raw * sc
    e = k * k + h * h
    # floor before the sqrt: d(sqrt)/de is infinite at e == 0, and
    # h * inf = NaN there. e = 0 is an interior point of the disc.
    sq = mx.sqrt(mx.maximum(e, 1e-30))
    esw = h * sq                                   # e sin w, division-free
    b, a = v[:, 3], v[:, 4]
    # b = (a cos i) (1 - e^2) / (1 + e sin w)  =>  cos i = b (1 + e sin w) / (a (1 - e^2))
    ci = b * (1.0 + esw) / (a * mx.maximum(1.0 - e * e, 1.0 - e_max * e_max))
    return k, h, ci, e


def _huber(c, delta: float = 1.0):
    """One-sided Huber violation: 0 for c <= 0, c^2 near the boundary,
    linear beyond delta.

    A pure quadratic is the textbook HMC barrier, but here a far-outside
    proposal can drive a violation to O(1000) -- the projection onto the
    e_max disc sends 1 - e^2 to 0.002, which makes |cos i| explode -- and
    c^2 then contributes ~1e12 with ~1e11 gradients, collapsing the step
    size. Going linear past delta keeps the force bounded (2 delta) while
    never letting it vanish, so a chain thrown into the failure region is
    pushed back instead of either exploding or stalling.
    """
    x = mx.maximum(c, 0.0)
    return mx.where(x <= delta, x * x, delta * (2.0 * x - delta))


def ecc_constraint_penalty(v, strength: float = 1e6,
                           e_max: float = E_MAX_NUMERICAL) -> mx.array:
    """Smooth barrier for the JOINT physical constraints that a box of
    ParamSpecs cannot express (returns <= 0, exactly 0 when feasible).

      * periastron clearance   a (1 - e) > 1 + r
      * a real inclination     |cos i| <= 1
      * the eccentricity disc  secosw^2 + sesinw^2 <= e_max

    MetalPlanet clamps every numerical hazard, so an unphysical proposal
    otherwise returns an ordinary *finite* log-likelihood and the chain
    samples an improper posterior silently. A quadratic barrier is used
    rather than a -inf wall because hard walls make HMC diverge.

    The third term is what makes the barrier *escapable*. The model
    projects (secosw, sesinw) radially onto the e_max disc, so outside it
    the likelihood is exactly constant along the radial direction: a
    chain thrown into the corners of the +/-0.95 box (where e_raw reaches
    1.8) would sit on a plateau with no radial force pushing it back,
    only the ParamSpec sigmoid Jacobian. Penalising the UNPROJECTED
    e_raw restores a gradient that points inward.
    """
    k, h, ci, e = _ecc_orbit_from_v(v, e_max)
    r, a, b = v[:, 2], v[:, 4], v[:, 3]
    e_raw = v[:, 7] ** 2 + v[:, 8] ** 2
    sq = mx.sqrt(mx.maximum(e, 1e-30))
    esw = mx.where(e_raw > e_max,
                   v[:, 8] * mx.sqrt(mx.maximum(e_max / mx.maximum(
                       e_raw, 1e-30), 0.0)) * sq, v[:, 8] * sq)
    # each residual is DIMENSIONLESS and O(1): |cos i| - 1 would instead
    # explode, because projecting onto the e_max disc drives 1 - e^2 to
    # 0.002 and cos i with it, contributing ~1e12 with ~1e11 gradients.
    # Same zero set, same sign, bounded scale.
    c_peri = ((1.0 + r) - a * (1.0 - e)) / (1.0 + r + a)
    c_inc = (b * (1.0 + esw) - a * (1.0 - e * e)) / (1.0 + b + a)
    c_disc = (e_raw - e_max) / (1.0 + e_max)
    return -strength * (_huber(c_peri) + _huber(c_inc) + _huber(c_disc))


def make_ecc_transit_flux(period_ref: float, core: str = "metal"):
    """Engine-contract batched eccentric model, v = (n_chains, 10) with
    PARAM_NAMES_ECC and x = (2, m) epoch-centered times.

    Eccentricity is sampled as (sqrt(e) cos w, sqrt(e) sin w), whose
    e = 0 interior point a sampler genuinely visits; the transit-anchored
    orbit keeps float32 gradients accurate there (see anchored.py). Pair
    it with ``ecc_constraint_penalty`` — this function deliberately
    returns finite flux for unphysical geometries, exactly as the
    circular model does.
    """
    period_ref = float(period_ref)
    graph_fn = _build_graph_ecc_model(period_ref)
    if core == "graph":
        return graph_fn
    if core != "metal":
        raise ValueError(f"unknown core {core!r}")
    from .metal import _gpu_stream_active, make_model_core_metal, metal_available
    if not metal_available():
        return graph_fn
    from .anchored import pack_orbit_constants
    model_core = make_model_core_metal(period_ref)

    def flux_dev_fn(v: mx.array, x: mx.array) -> mx.array:
        if (v.dtype == mx.float32 and x.dtype == mx.float32
                and _gpu_stream_active()):
            u1, u2 = q_to_u(v[:, 5], v[:, 6])
            k, h, ci, _ = _ecc_orbit_from_v(v)
            orb = pack_orbit_constants(k, h, ci)
            dev = model_core(x, v[:, 0], v[:, 1], v[:, 2], v[:, 4],
                             orb, u1, u2)
            return v[:, 9:10] + dev
        return graph_fn(v, x)

    return flux_dev_fn


def _build_graph_ecc_model(period_ref: float):
    """Same math in pure MLX graph ops: the fp64 / CPU verification path."""
    from .anchored import separation_anchored

    def flux_dev_fn(v: mx.array, x: mx.array) -> mx.array:
        dt, kk = x[0], x[1]
        t0_off, p_off = v[:, 0:1], v[:, 1:2]
        r, a = v[:, 2:3], v[:, 4:5]
        u1, u2 = q_to_u(v[:, 5:6], v[:, 6:7])
        df0 = v[:, 9:10]
        k, h, ci, _ = _ecc_orbit_from_v(v)
        period = period_ref + p_off
        tau = tau_from_epochs(dt[None, :], kk[None, :], t0_off, p_off, period)
        phi = (2.0 * np.pi) * tau / period
        z, front = separation_anchored(phi, k[:, None], h[:, None], a,
                                       ci[:, None])
        f = flux_dev(z, r, u1, u2)
        return df0 + mx.where(front & (z < 1.0 + r), f, 0.0)

    return flux_dev_fn


class PenalizedLogLike:
    """``ChunkedGaussianLogLike`` plus a joint-constraint barrier.

    The engine's ParamSpecs are per-parameter boxes, so constraints that
    couple parameters have nowhere else to live; without this the chain
    samples an improper posterior with no error message.
    """

    def __init__(self, base, penalty):
        self.base = base
        self.penalty = penalty

    def __call__(self, v: mx.array) -> mx.array:
        return self.base(v) + self.penalty(v)

    def hi(self, v: mx.array) -> mx.array:
        return self.base.hi(v) + self.penalty(v.astype(mx.float64))

    def __getattr__(self, name):
        # __getattr__ runs for ANY attribute missing from the instance,
        # including lookups that happen before __init__ has assigned
        # `base` -- copy/deepcopy/unpickle all probe __deepcopy__,
        # __getstate__, __setstate__ on a bare object first. Delegating
        # those blindly recurses on `base` itself and raises
        # RecursionError where AttributeError is required.
        if name.startswith("__") or "base" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.__dict__["base"], name)


@dataclass
class EccTransitTarget:
    """Synthetic eccentric transit-fitting problem (anvil-ready)."""

    target: object
    transform: object
    loglike: object
    truth_model: np.ndarray       # (10,) true params, model units
    t_ref: float
    t_model: np.ndarray
    y: np.ndarray


def make_ecc_target(n_data: int = 100_000, baseline_days: float = 90.0,
                    yerr: float = 5e-4, seed: int = 11,
                    core: str = "metal", ecc: float = 0.3,
                    omega_deg: float = 63.0) -> EccTransitTarget:
    """Injection-recovery problem for the 10-parameter eccentric model."""
    applemcmc, ChunkedGaussianLogLike = import_engine()
    rng = np.random.default_rng(seed)

    t_ref = 2_457_000.0
    t_abs = t_ref + np.sort(rng.uniform(0.0, baseline_days, size=n_data))
    t_model = t_abs - t_ref

    u1_true, u2_true = 0.40, 0.25
    q1_true, q2_true = u_to_q_np(u1_true, u2_true)
    w = math.radians(omega_deg)
    truth = np.array([
        1.2345, 3.456, 0.10, 0.30, 8.80,
        float(q1_true), float(q2_true),
        math.sqrt(ecc) * math.cos(w), math.sqrt(ecc) * math.sin(w), 1.0,
    ])
    t0_ref = float(truth[0]) - 0.009
    period_ref = float(truth[1]) + 0.0005
    truth_model = np.array([
        truth[0] - t0_ref, truth[1] - period_ref, truth[2], truth[3],
        truth[4], truth[5], truth[6], truth[7], truth[8], truth[9] - 1.0,
    ])
    x64 = epoch_center_times(t_model, t0_ref=t0_ref, period_ref=period_ref)
    model_fn = make_ecc_transit_flux(period_ref=period_ref, core=core)
    with mx.stream(mx.cpu):
        dev_true = np.array(
            model_fn(mx.array(truth_model[None, :], dtype=mx.float64),
                     mx.array(x64, dtype=mx.float64))[0], dtype=np.float64)
    y = 1.0 + dev_true + yerr * rng.standard_normal(n_data)
    y_fit = y - 1.0

    policy = applemcmc.PrecisionPolicy()
    transform = applemcmc.Transform([
        applemcmc.ParamSpec("t0_off", lo=-0.5, hi=0.5,
                            report_offset=t_ref + t0_ref),
        applemcmc.ParamSpec("p_off", lo=-0.05, hi=0.05,
                            report_offset=period_ref),
        applemcmc.ParamSpec("r", lo=0.01, hi=0.5),
        applemcmc.ParamSpec("b", lo=0.0, hi=1.2),
        applemcmc.ParamSpec("a", lo=2.0, hi=50.0),
        applemcmc.ParamSpec("q1", lo=0.0, hi=1.0),
        applemcmc.ParamSpec("q2", lo=0.0, hi=1.0),
        # the (sqrt(e) cos w, sqrt(e) sin w) disc inscribed in its box;
        # e <= e_max(a, r) and |cos i| <= 1 are joint, so they are the
        # barrier's job, not the box's.
        applemcmc.ParamSpec("secosw", lo=-0.95, hi=0.95),
        applemcmc.ParamSpec("sesinw", lo=-0.95, hi=0.95),
        applemcmc.ParamSpec("df0", lo=-0.01, hi=0.01, report_offset=1.0),
    ])
    loglike = PenalizedLogLike(
        ChunkedGaussianLogLike(model_fn, x64, y_fit,
                               np.full(n_data, yerr), policy),
        ecc_constraint_penalty)
    target = applemcmc.TransformedLogDensity(
        loglike, transform, model_log_prob_hi=loglike.hi)
    return EccTransitTarget(
        target=target, transform=transform, loglike=loglike,
        truth_model=truth_model, t_ref=t_ref, t_model=t_model, y=y)
