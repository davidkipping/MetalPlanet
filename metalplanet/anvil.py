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

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .flux import flux_dev
from .ld import q_to_u, q_to_u_np, u_to_q_np
from .orbit import epoch_center_times, separation_circular, tau_from_epochs

__all__ = ["make_quad_transit_flux", "make_target", "QuadTransitTarget",
           "import_engine"]

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
        from .metal import (_gpu_stream_active, flux_dev_metal,
                            make_model_core_metal, metal_available)
        graph_fn = _build_graph_model(period_ref, flux_dev_metal)
        if not metal_available():
            return graph_fn
        model_core = make_model_core_metal(period_ref)

        def flux_dev_fn(v: mx.array, x: mx.array) -> mx.array:
            # v2 fused path: whole orbit + photometry in one kernel.
            # Anything the kernel can't serve (fp64, CPU stream — e.g.
            # anvil's data generation and verification paths) falls back
            # to the graph model, whose own core falls back below fp32.
            if (v.dtype == mx.float32 and x.dtype == mx.float32
                    and _gpu_stream_active()):
                u1, u2 = q_to_u(v[:, 5], v[:, 6])
                dev = model_core(x, v[:, 0], v[:, 1], v[:, 2], v[:, 3],
                                 v[:, 4], u1, u2)
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
