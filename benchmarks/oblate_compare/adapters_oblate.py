"""Per-code adapters for the oblate cross-code benchmark.

Each code computes the scenario's light curve (scenario_oblate.py) in its
own parameterisation; the mappings were checked against each other before
timing (README.md here):

    metalplanet    TransitParams.f / .theta [deg], hybrid2 with w = (0.5, 0)
    squishyplanet  projected_effective_r / projected_f / projected_theta
    JoJo           oblate_lc: rp_me = r_eff, obliquity = -theta (its Y axis
                   points the other way), a/R* through the stellar density
    GreenLantern   ellipsoid semi-axes (B, A, A), third angle = theta,
                   beta = -(pi/2 - i); quadratic LD as Kipping (q1, q2)

prepare(n) -> state; run(state) -> flux (numpy). Optional: *_batch_* for
many parameter sets, *_grad_* for value + gradient of a scalar loss.
"""

import math

import numpy as np

from scenario_oblate import (A_AX, A_RS, B_AX, B_IMPACT, BATCH_NPT, F, INC,
                             PER, RP, T0, THETA, U1, U2, W_LD, batch_params,
                             time_grid)

Q1 = (U1 + U2) ** 2
Q2 = U1 / (2.0 * (U1 + U2))
RHO = (A_RS / 3.753) ** 3 / PER ** 2      # JoJo's a/R* = 3.753 (P^2 rho)^(1/3)


# ---------------------------------------------------------------------------
# MetalPlanet
# ---------------------------------------------------------------------------

def _mp_params(**over):
    import metalplanet
    p = metalplanet.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc = T0, PER, RP, A_RS, math.degrees(INC)
    p.ecc, p.w = 0.0, 90.0
    p.limb_dark, p.u = "hybrid2", [W_LD, 0.0]
    p.f, p.theta = F, math.degrees(THETA)
    for k, v in over.items():
        setattr(p, k, v)
    return p


def _mp_prepare(n, dtype):
    import metalplanet
    p = _mp_params()
    m = metalplanet.TransitModel(p, time_grid(n), dtype=dtype)
    m.light_curve(p)
    return m, p


def metalplanet_fp64_prepare(n):
    import mlx.core as mx
    return _mp_prepare(n, mx.float64)


def metalplanet_fp32_prepare(n):
    import mlx.core as mx
    return _mp_prepare(n, mx.float32)


def metalplanet_fp64_run(state):
    m, p = state
    return m.light_curve(p)


metalplanet_fp32_run = metalplanet_fp64_run


def metalplanet_fp32_batch_prepare():
    import mlx.core as mx
    import metalplanet
    bp = batch_params()
    p = _mp_params()
    p.rp, p.f, p.theta = bp["rp"], bp["f"], np.degrees(bp["theta"])
    p.t0, p.per, p.a = bp["t0"], bp["per"], bp["a"]
    p.inc = np.degrees(np.arccos(bp["b"] / bp["a"]))
    m = metalplanet.TransitModel(_mp_params(), time_grid(BATCH_NPT),
                                 dtype=mx.float32)
    m.light_curves(p)
    return m, p


def metalplanet_fp32_batch_run(state):
    m, p = state
    return m.light_curves(p)


def _mp_grad_prepare(n, dtype):
    """value + gradient of sum(ct * flux) in rp, f, theta, inc, t0, per, a
    and the limb-darkening weight, through light_curve_mx."""
    import mlx.core as mx
    import metalplanet
    p = _mp_params()
    stream = mx.cpu if dtype == mx.float64 else mx.gpu
    m = metalplanet.TransitModel(p, time_grid(n), dtype=dtype)
    ct = mx.array(np.random.default_rng(0).normal(size=n), dtype=dtype)
    names = ("rp", "f", "theta", "inc", "t0", "per", "a")
    x0 = [mx.array(float(getattr(p, k)), dtype=dtype) for k in names] + \
         [mx.array(W_LD, dtype=dtype)]

    def loss(*v):
        q = _mp_params()
        for k, val in zip(names, v):
            setattr(q, k, val)
        q.u = [v[-1], 0.0]
        return mx.sum(ct * m.light_curve_mx(q))
    vg = mx.value_and_grad(loss, argnums=tuple(range(len(x0))))
    state = (vg, x0, stream)
    metalplanet_grad_run(state)
    return state


def metalplanet_grad_run(state):
    import mlx.core as mx
    vg, x0, stream = state
    with mx.stream(stream):
        val, g = vg(*x0)
        mx.eval(val, *g)
    return np.array([float(x) for x in g])


def metalplanet_fp64_grad_prepare(n):
    import mlx.core as mx
    return _mp_grad_prepare(n, mx.float64)


def metalplanet_fp32_grad_prepare(n):
    import mlx.core as mx
    return _mp_grad_prepare(n, mx.float32)


metalplanet_fp64_grad_run = metalplanet_grad_run
metalplanet_fp32_grad_run = metalplanet_grad_run


# ---------------------------------------------------------------------------
# squishyplanet (JAX, float64 on the CPU)
# ---------------------------------------------------------------------------

def _sq_system(t):
    import jax
    jax.config.update("jax_enable_x64", True)
    from squishyplanet import OblateSystem
    return OblateSystem(times=t, t0=T0, period=PER, a=A_RS, i=INC, r=RP,
                        ld_u_coeffs=np.array([U1, U2]),
                        projected_effective_r=RP, projected_f=F,
                        projected_theta=THETA,
                        parameterize_with_projected_ellipse=True,
                        tidally_locked=False)


def squishyplanet_prepare(n):
    s = _sq_system(time_grid(n))
    np.asarray(s.lightcurve())
    return s


def squishyplanet_run(s):
    return np.asarray(s.lightcurve())


def squishyplanet_batch_prepare():
    import jax
    import jax.numpy as jnp
    s = _sq_system(time_grid(BATCH_NPT))
    bp = batch_params()
    params = {"projected_effective_r": jnp.asarray(bp["rp"]),
              "projected_f": jnp.asarray(bp["f"]),
              "projected_theta": jnp.asarray(bp["theta"]),
              "t0": jnp.asarray(bp["t0"]), "period": jnp.asarray(bp["per"]),
              "a": jnp.asarray(bp["a"]),
              "i": jnp.asarray(np.arccos(bp["b"] / bp["a"]))}
    fn = jax.jit(jax.vmap(s.lightcurve))
    np.asarray(fn(params))
    return fn, params


def squishyplanet_batch_run(state):
    fn, params = state
    return np.asarray(fn(params))


def squishyplanet_grad_prepare(n):
    import jax
    import jax.numpy as jnp
    s = _sq_system(time_grid(n))
    ct = jnp.asarray(np.random.default_rng(0).normal(size=n))
    params = {"projected_effective_r": jnp.asarray(RP),
              "projected_f": jnp.asarray(F),
              "projected_theta": jnp.asarray(THETA), "i": jnp.asarray(INC),
              "t0": jnp.asarray(T0), "period": jnp.asarray(PER),
              "a": jnp.asarray(A_RS), "ld_u_coeffs": jnp.asarray([U1, U2])}
    vg = jax.jit(jax.value_and_grad(lambda prm: jnp.sum(ct * s.lightcurve(prm))))
    state = (vg, params)
    squishyplanet_grad_run(state)
    return state


def squishyplanet_grad_run(state):
    import jax
    vg, params = state
    val, g = vg(params)
    jax.block_until_ready(g)
    return g


# ---------------------------------------------------------------------------
# JoJo (numpy, float64, numerical line integral with n_step = 100)
# ---------------------------------------------------------------------------

def _jojo_args(tc=T0, b=B_IMPACT, per=PER, rp=RP, f=F, theta=THETA, a=A_RS):
    rho = (a / 3.753) ** 3 / per ** 2
    return [tc, b, per, rp, f, -theta, 0.0, math.pi / 2, U1, U2,
            math.log10(rho)]


def jojo_prepare(n):
    from JoJo.jojo_oblate import oblate_lc
    t = time_grid(n)
    oblate_lc(_jojo_args(), t, exp_time=0.0)
    return oblate_lc, t


def jojo_run(state):
    oblate_lc, t = state
    return np.asarray(oblate_lc(_jojo_args(), t, exp_time=0.0)[0])


def jojo_batch_prepare():
    from JoJo.jojo_oblate import oblate_lc
    return oblate_lc, time_grid(BATCH_NPT), batch_params()


def jojo_batch_run(state):
    oblate_lc, t, bp = state
    out = np.empty((bp["rp"].size, t.size))
    for j in range(bp["rp"].size):              # no native batch
        out[j] = oblate_lc(_jojo_args(bp["t0"][j], bp["b"][j], bp["per"][j],
                                      bp["rp"][j], bp["f"][j], bp["theta"][j],
                                      bp["a"][j]), t, exp_time=0.0)[0]
    return out


# ---------------------------------------------------------------------------
# GreenLantern (OpenCL, float32, Simpson's rule with 2048 boundary samples)
# ---------------------------------------------------------------------------

def _gl_row(rp=RP, f=F, theta=THETA, a=A_RS, tc=T0, b=B_IMPACT, per=PER):
    A = rp / math.sqrt(1.0 - f)
    B = A * (1.0 - f)
    inc = math.acos(b / a)
    return [B, A, A, a, tc, -(math.pi / 2 - inc), 0.0, 0.0, theta, Q1, Q2, per]


def _gl_ctx():
    import pocky
    import greenlantern
    ctx = pocky.Context.default()
    return pocky, ctx, greenlantern.Context(ctx)


def greenlantern_prepare(n):
    pocky, ctx, g = _gl_ctx()
    t = pocky.BufferPair(ctx, time_grid(n).astype(np.float32))
    t.copy_to_device()
    t.dirty = False
    params = pocky.BufferPair(ctx, np.array([_gl_row()], dtype=np.float32))
    flux = pocky.BufferPair(ctx, np.empty((1, n), dtype=np.float32))
    state = (g, t, params, flux)
    greenlantern_run(state)
    return state


def greenlantern_run(state):
    g, t, params, flux = state
    g.ellipsoid_transit_flux(t, params, flux=flux, eccentric=False)
    return flux.host[0].astype(np.float64)


def greenlantern_batch_prepare():
    pocky, ctx, g = _gl_ctx()
    bp = batch_params()
    rows = [_gl_row(bp["rp"][j], bp["f"][j], bp["theta"][j], bp["a"][j],
                    bp["t0"][j], bp["b"][j], bp["per"][j])
            for j in range(bp["rp"].size)]
    t = pocky.BufferPair(ctx, time_grid(BATCH_NPT).astype(np.float32))
    t.copy_to_device()
    t.dirty = False
    params = pocky.BufferPair(ctx, np.array(rows, dtype=np.float32))
    flux = pocky.BufferPair(ctx, np.empty((len(rows), BATCH_NPT), dtype=np.float32))
    state = (g, t, params, flux)
    greenlantern_batch_run(state)
    return state


def greenlantern_batch_run(state):
    g, t, params, flux = state
    g.ellipsoid_transit_flux(t, params, flux=flux, eccentric=False)
    return flux.host


def greenlantern_grad_prepare(n):
    """Its forward-mode dual kernel: flux and d flux / d (12 parameters)
    per point; the loss gradient is that Jacobian times the cotangent."""
    pocky, ctx, g = _gl_ctx()
    t = pocky.BufferPair(ctx, time_grid(n).astype(np.float32))
    t.copy_to_device()
    t.dirty = False
    params = pocky.BufferPair(ctx, np.array([_gl_row()], dtype=np.float32))
    flux = pocky.BufferPair(ctx, np.empty((n,), dtype=np.float32))
    dflux = pocky.BufferPair(ctx, np.empty((12, n), dtype=np.float32))
    ct = np.random.default_rng(0).normal(size=n).astype(np.float32)
    state = (g, t, params, flux, dflux, ct)
    greenlantern_grad_run(state)
    return state


def greenlantern_grad_run(state):
    g, t, params, flux, dflux, ct = state
    g.ellipsoid_transit_flux_dual(t, params, flux=flux, dflux=dflux)
    return dflux.host @ ct


CODES = ["metalplanet_fp64", "metalplanet_fp32", "squishyplanet", "jojo",
         "greenlantern"]
BATCH_CODES = ["metalplanet_fp32", "squishyplanet", "jojo", "greenlantern"]
GRAD_CODES = ["metalplanet_fp64", "metalplanet_fp32", "squishyplanet",
              "greenlantern"]
