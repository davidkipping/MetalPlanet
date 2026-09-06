"""Uniform per-code adapters for the cross-code benchmark.

Each adapter provides:
    prepare(n)        -> state (built once per process; jit/warmup here)
    run(state)        -> flux ndarray for the standard scenario, n points
    prepare_batch()/run_batch(state) (optional) -> (npv, npt) fluxes

Thread control happens via environment variables set BEFORE the worker
process imports the code (numba/OpenMP read them at import/first-call).

Capability notes (measured on this machine, macOS/arm64):
    batman        : nthreads > 1 needs OpenMP-built extension (probe)
    pytransit     : numba threading (NUMBA_NUM_THREADS)
    exoplanet-core: single-threaded C++ ops loop
    jaxoplanet    : XLA CPU thread pool (not user-partitionable cleanly)
    metalplanet   : MLX — one GPU (all cores; not partitionable) or CPU
    ellc          : needs a Fortran toolchain; unavailable without
                    gfortran (`brew install gcc` would enable it)
"""

import math

import numpy as np

from scenario import (A_RS, B_IMPACT, INC_DEG, PER, RP, T0, U1, U2,
                      batch_params, time_grid, z_of_t)


# ---------------------------------------------------------------------------
# MetalPlanet
# ---------------------------------------------------------------------------

def _mp_params():
    import metalplanet
    p = metalplanet.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = T0, PER, RP, A_RS, INC_DEG
    p.ecc, p.w, p.u, p.limb_dark = 0.0, 90.0, [U1, U2], "quadratic"
    return p


def metalplanet_fp64_prepare(n):
    import metalplanet
    p = _mp_params()
    m = metalplanet.TransitModel(p, time_grid(n))
    m.light_curve(p)  # warm
    return (m, p)


def metalplanet_fp64_run(state):
    m, p = state
    return m.light_curve(p)


def metalplanet_fp32_prepare(n):
    import mlx.core as mx
    import metalplanet
    p = _mp_params()
    m = metalplanet.TransitModel(p, time_grid(n), dtype=mx.float32)
    m.light_curve(p)
    return (m, p)


def metalplanet_fp32_run(state):
    m, p = state
    return m.light_curve(p)


def metalplanet_gpu_batch_prepare(_n=None):
    """Engine-contract batched model: (npv, 8) params x (2, npt) times,
    fp32 on the GPU — MetalPlanet's design center."""
    import mlx.core as mx
    from metalplanet.anvil import make_quad_transit_flux
    from metalplanet.ld import u_to_q_np
    from scenario import BATCH_NPT, BATCH_NPV

    bp = batch_params()
    q1, q2 = u_to_q_np(bp["u1"], bp["u2"])
    v = np.stack([
        bp["t0"], bp["per"] - PER, bp["rp"], bp["b"], bp["a"], q1, q2,
        np.zeros(BATCH_NPV),
    ], axis=1).astype(np.float32)
    t = time_grid(BATCH_NPT)
    x = np.stack([t, np.zeros_like(t)]).astype(np.float32)
    model = mx.compile(make_quad_transit_flux(PER))
    v_mx, x_mx = mx.array(v), mx.array(x)
    out = model(v_mx, x_mx)
    mx.eval(out)  # warm/compile
    return (model, v_mx, x_mx, mx)


def metalplanet_gpu_batch_run(state):
    model, v, x, mx = state
    out = model(v, x)
    mx.eval(out)
    return out


# ---------------------------------------------------------------------------
# batman
# ---------------------------------------------------------------------------

def batman_prepare(n, nthreads=1):
    import batman
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = T0, PER, RP, A_RS, INC_DEG
    p.ecc, p.w, p.u, p.limb_dark = 0.0, 90.0, [U1, U2], "quadratic"
    m = batman.TransitModel(p, time_grid(n), nthreads=nthreads)
    m.light_curve(p)
    return (m, p)


def batman_run(state):
    m, p = state
    return m.light_curve(p)


def batman_batch_prepare(_n=None):
    import batman
    from scenario import BATCH_NPT
    bp = batch_params()
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = T0, PER, RP, A_RS, INC_DEG
    p.ecc, p.w, p.u, p.limb_dark = 0.0, 90.0, [U1, U2], "quadratic"
    m = batman.TransitModel(p, time_grid(BATCH_NPT))
    m.light_curve(p)
    return (m, p, bp)


def batman_batch_run(state):
    m, p, bp = state
    out = []
    for i in range(bp["rp"].size):
        p.t0 = bp["t0"][i]
        p.per = bp["per"][i]
        p.rp = bp["rp"][i]
        p.a = bp["a"][i]
        p.inc = math.degrees(math.acos(bp["b"][i] / bp["a"][i]))
        p.u = [bp["u1"][i], bp["u2"][i]]
        out.append(m.light_curve(p))
    return np.stack(out)


# ---------------------------------------------------------------------------
# PyTransit
# ---------------------------------------------------------------------------

def pytransit_prepare(n, interpolate=False):
    from pytransit import QuadraticModel
    tm = QuadraticModel(interpolate=interpolate)
    tm.set_data(time_grid(n))
    kw = dict(k=RP, ldc=[U1, U2], t0=T0, p=PER, a=A_RS,
              i=math.radians(INC_DEG))
    tm.evaluate(**kw)  # trigger numba compile
    return (tm, kw)


def pytransit_run(state):
    tm, kw = state
    return np.asarray(tm.evaluate(**kw))


def pytransit_batch_prepare(_n=None):
    from pytransit import QuadraticModel
    from scenario import BATCH_NPT
    bp = batch_params()
    tm = QuadraticModel(interpolate=False)
    tm.set_data(time_grid(BATCH_NPT))
    inc = np.arccos(bp["b"] / bp["a"])
    ldc = np.stack([bp["u1"], bp["u2"]], axis=1)
    kw = dict(k=bp["rp"], ldc=ldc, t0=bp["t0"], p=bp["per"], a=bp["a"],
              i=inc)
    tm.evaluate(**kw)
    return (tm, kw)


def pytransit_batch_run(state):
    tm, kw = state
    return np.asarray(tm.evaluate(**kw))


# ---------------------------------------------------------------------------
# exoplanet-core (the compiled kernels inside `exoplanet`, FM+21)
# ---------------------------------------------------------------------------

def _xo_flux(b, r, u1, u2):
    from exoplanet_core.numpy import ops
    b = np.ascontiguousarray(b, dtype=np.float64)
    s = ops.quad_solution_vector(b, np.full_like(b, r))
    g = np.array([1.0 - u1 - 1.5 * u2, u1 + 2.0 * u2, -0.25 * u2])
    norm = np.pi * (g[0] + 2.0 * g[1] / 3.0)
    return s @ (g / norm)


def exoplanet_prepare(n):
    t = time_grid(n)
    _xo_flux(z_of_t(t[:16]), RP, U1, U2)
    return t


def exoplanet_run(t):
    # orbit -> z inside the timed region: every code is measured from
    # orbital elements, not a precomputed separation
    return _xo_flux(z_of_t(t), RP, U1, U2)


def exoplanet_batch_prepare(_n=None):
    from scenario import BATCH_NPT
    bp = batch_params()
    t = time_grid(BATCH_NPT)
    return (t, bp)


def exoplanet_batch_run(state):
    t, bp = state
    out = []
    for i in range(bp["rp"].size):
        phi = 2 * np.pi * (t - bp["t0"][i]) / bp["per"][i]
        cosi = bp["b"][i] / bp["a"][i]
        z = bp["a"][i] * np.sqrt(np.sin(phi) ** 2
                                 + cosi ** 2 * np.cos(phi) ** 2)
        z = np.where(np.cos(phi) > 0, z, 2.0 + z)
        out.append(_xo_flux(z, bp["rp"][i], bp["u1"][i], bp["u2"][i]))
    return np.stack(out)


# ---------------------------------------------------------------------------
# jaxoplanet
# ---------------------------------------------------------------------------

def jaxoplanet_prepare(n, order=10):
    import jax
    import jax.numpy as jnp
    from jaxoplanet.core.limb_dark import light_curve
    jax.config.update("jax_enable_x64", True)
    u = jnp.array([U1, U2])
    t = jnp.asarray(time_grid(n))
    cosi = B_IMPACT / A_RS

    @jax.jit
    def f(t_):
        # orbit inside the jit: measured from orbital elements like
        # every other code (XLA fuses it, so the penalty is small)
        phi = 2.0 * jnp.pi * (t_ - T0) / PER
        z = A_RS * jnp.sqrt(jnp.sin(phi) ** 2
                            + cosi ** 2 * jnp.cos(phi) ** 2)
        z = jnp.where(jnp.cos(phi) > 0, z, 2.0 + z)
        return 1.0 + light_curve(u, z, RP, order=order)

    f(t).block_until_ready()
    return (f, t)


def jaxoplanet_run(state):
    f, t = state
    return np.asarray(f(t).block_until_ready())


def jaxoplanet_batch_prepare(_n=None):
    import jax
    import jax.numpy as jnp
    from jaxoplanet.core.limb_dark import light_curve
    from scenario import BATCH_NPT
    jax.config.update("jax_enable_x64", True)
    bp = batch_params()
    t = jnp.asarray(time_grid(BATCH_NPT))

    def one(rp, t0, per, a, b_imp, u1, u2):
        phi = 2 * jnp.pi * (t - t0) / per
        cosi = b_imp / a
        z = a * jnp.sqrt(jnp.sin(phi) ** 2 + cosi ** 2 * jnp.cos(phi) ** 2)
        z = jnp.where(jnp.cos(phi) > 0, z, 2.0 + z)
        return 1.0 + light_curve(jnp.stack([u1, u2]), z, rp, order=10)

    f = jax.jit(jax.vmap(one))
    args = tuple(jnp.asarray(bp[k]) for k in
                 ("rp", "t0", "per", "a", "b", "u1", "u2"))
    f(*args).block_until_ready()
    return (f, args)


def jaxoplanet_batch_run(state):
    f, args = state
    return np.asarray(f(*args).block_until_ready())
