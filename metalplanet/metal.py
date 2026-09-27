"""Hand-fused Metal kernels for the fp32 photometric core.

The MLX elementwise graph of the ALFM19 core is memory-bandwidth-bound
(~KB of intermediate traffic per point). These kernels keep the whole
computation — geometry, kappas, the merged 10-iteration cel3 recursion,
Taylor switches, assembly — in registers: one read of z (+ per-chain
parameters) and one write of flux per point, with a real early exit for
out-of-transit points.

Design decisions (from the reviewed plan):

* 2D grid (m, n_chains): `.y` is the chain, so per-chain r/u1/u2 loads
  are simdgroup-uniform and there is no emulated integer divide; `.x`
  needs an explicit bounds check (grid is dispatchThreads).
* Early exit STORES 0 then returns — outputs are uninitialized memory
  without it, and the exact-0 out-of-transit contract is load-bearing.
  The exit test `z >= 1 + r` pairs with `(r + 1 - z)` in onembmr2 so the
  in-branch quantities are strictly positive.
* Branch precedence m_con over m_req (both slivers can hold at
  z ~ r ~ 0.5; the graph's where-stack resolves contact first).
* Clamp triage: the graph's masked-lane sanitizers (denominator wheres,
  tiny floors) are identities inside real branches and are dropped; the
  kite floor, the Kahan sorted-product sqarea grouping, the kc^2 clips,
  and the acos-argument clip change in-branch values and are kept
  verbatim.
* `metal::precise::` transcendentals (measured at fp32 rounding accuracy
  on this machine); math_mode stays "safe".

The VJP kernel recomputes forward-style and emits ct*dF/dz full-shape
plus per-point ct*dF/d{r,u1,u2}, reduced afterwards by mx.sum (the
~24 B/pt of traffic costs ~10-15 ms at 1024x65536 — far under budget;
simdgroup reductions are a deliberate non-goal for v1).

Dispatch: fp32 on the GPU stream only; anything else silently falls back
to the analytic-VJP graph core (the fp64 CPU verification path and
anvil's CPU data generation rely on this).
"""

from __future__ import annotations

import mlx.core as mx

from .vjp import _unbroadcast, flux_dev_analytic

__all__ = ["flux_dev_metal", "metal_available"]

_EPS = "1.1920929e-07f"
_D_CON = "3.4526698e-04f"          # sqrt(fp32 eps)
_KITE_FLOOR = "1.4210855e-12f"     # (10 eps)^2

_HEADER = """
constant float MP_PI = 3.14159265358979f;
constant float MP_TWO_PI = 6.28318530717959f;
constant float MP_HALF_PI = 1.57079632679490f;
"""

# ---------------------------------------------------------------------------
# shared source fragments
# ---------------------------------------------------------------------------

# geometry + masks + s0d/s2d + merged cel3 -> defines:
#   z, r, u1, u2, r2, z2, m_comp, m_ps, m_req, m_con, dzr, zpr,
#   onembmr2, onembpr2, fourzr, kap0, kite, s0d, s2d, Pi_, E_, Em
_CORE = """
    uint x = thread_position_in_grid.x;
    uint y = thread_position_in_grid.y;
    if (x >= (uint)npts) return;
    uint i = y * (uint)npts + x;

    float z = fabs(zin[i]);
    float r = rin[y];
    EARLY_EXIT

    float u1 = u1in[y];
    float u2 = u2in[y];
    float r2 = r * r;
    float z2 = z * z;
    float dzr = z - r;
    float zpr = z + r;

    bool m_comp = z <= 1.0f - r;
    bool m_ps = zpr > 1.0f;
    bool m_req = fabs(dzr) < 10.0f * EPS;
    bool m_con = fabs(zpr - 1.0f) < D_CON;

    float onembmr2 = (r + 1.0f - z) * (1.0f - r + z);
    float onembpr2 = (1.0f - z - r) * (1.0f + z + r);
    float fourzr = 4.0f * z * r;

    float s0d, s2d;
    float kap0 = 0.0f, kite = 0.0f;
    float eta2 = r2 * (r2 + 2.0f * z2);
    if (m_comp) {
        s0d = -MP_PI * r2;
        s2d = MP_TWO_PI * r2 * (r2 + 2.0f * z2 - 1.0f);
    } else {
        // Kahan sorted-product sqarea: grouping kept verbatim
        float sa = max(z, r);
        float sc = min(z, r);
        float sb2 = min(sa, 1.0f);
        sa = max(sa, 1.0f);
        float sb = max(sb2, sc);
        sc = min(sb2, sc);
        float sqarea = (sa + (sb + sc)) * (sc - (sa - sb))
                     * (sc + (sa - sb)) * (sa + (sb - sc));
        kite = metal::precise::sqrt(max(sqarea, KITE_FLOOR));
        kap0 = metal::precise::atan2(kite, r2 + z2 - 1.0f);
        float kap1 = metal::precise::atan2(kite, 1.0f + z2 - r2);
        s0d = -(kap1 + r2 * kap0 - 0.5f * kite);
        s2d = 2.0f * s0d
            + 2.0f * (kap1 + eta2 * kap0
                      - 0.25f * kite * (1.0f + 5.0f * r2 + z2));
    }

    // merged cel3 argument setup (kc^2 clips kept: rounding gives -eps
    // near k^2 = 1 and sqrt(negative) would poison the lane)
    float kc2, p_cel, a1, b1;
    if (m_ps) {
        kc2 = clamp(-onembpr2 / fourzr, 0.0f, 1.0f);
        p_cel = dzr * dzr * kc2;
        a1 = 0.0f;
        b1 = 3.0f * kc2 * dzr * zpr;
    } else {
        kc2 = clamp(onembpr2 / onembmr2, 0.0f, 1.0f);
        float bmr_dpr = dzr / zpr;
        float mu = 3.0f * bmr_dpr / onembmr2;
        p_cel = bmr_dpr * bmr_dpr * max(onembpr2, 0.0f) / onembmr2;
        a1 = 1.0f + mu;
        b1 = p_cel + mu;
    }
    float kc = max(metal::precise::sqrt(kc2), EPS);
    p_cel = max(p_cel, 1e-30f);

    // one 10-iteration Bulirsch recursion, three chains (Pi-like, E, Em)
    float ee = kc;
    float mm = 1.0f;
    float sp = metal::precise::sqrt(p_cel);
    float pinv = 1.0f / sp;
    b1 *= pinv;
    float f1 = a1;
    a1 += b1 * pinv;
    float g = ee * pinv;
    b1 = 2.0f * (b1 + f1 * g);
    float pp = sp + g;
    float a2 = 1.0f, b2 = kc2, a3 = 1.0f, b3 = 0.0f;
    float g1 = ee;
    float f2 = a2;
    a2 += b2;
    b2 = 2.0f * (b2 + f2 * g1);
    float f3 = a3;
    a3 += b3;
    b3 = 2.0f * (b3 + f3 * g1);
    float p1 = 1.0f + g1;
    mm += kc;
    for (int it = 0; it < 10; ++it) {
        float kcn = 2.0f * metal::precise::sqrt(ee);
        ee = kcn * mm;
        f1 = a1; f2 = a2; f3 = a3;
        pinv = 1.0f / pp;
        float pinv1 = 1.0f / p1;
        a1 += b1 * pinv;
        a2 += b2 * pinv1;
        a3 += b3 * pinv1;
        g = ee * pinv;
        g1 = ee * pinv1;
        b1 = 2.0f * (b1 + f1 * g);
        b2 = 2.0f * (b2 + f2 * g1);
        b3 = 2.0f * (b3 + f3 * g1);
        pp += g;
        p1 += g1;
        mm += kcn;
    }
    float Pi_ = MP_HALF_PI * (a1 * mm + b1) / (mm * (mm + pp));
    float E_  = MP_HALF_PI * (a2 * mm + b2) / (mm * (mm + p1));
    float Em  = MP_HALF_PI * (a3 * mm + b3) / (mm * (mm + p1));

    // s1 deviation; precedence: contact sliver > z~r sliver > generic
    float s1d;
    if (m_con) {
        float ac = metal::precise::acos(
            clamp(1.0f - 2.0f * r, -1.0f + EPS, 1.0f - EPS));
        float sq = metal::precise::sqrt(max(r * (1.0f - r), 1e-30f));
        s1d = (MP_TWO_PI - 2.0f * ac
               + ((4.0f / 3.0f) * (3.0f + 2.0f * r - 8.0f * r2)
                  + 8.0f * (zpr - 1.0f) * r) * sq) / 3.0f
              - MP_TWO_PI / 3.0f;
    } else if (m_req) {
        float mreq = fourzr / onembmr2;
        float lam;
        if (m_ps) {
            lam = MP_PI + (1.0f / (3.0f * r))
                  * (-mreq * E_ + (2.0f * mreq - 3.0f) * Em)
                  - dzr * 2.0f * (2.0f * E_ - Em);
        } else {
            lam = MP_PI + (2.0f / 3.0f)
                  * ((2.0f * mreq - 3.0f) * E_ - mreq * Em)
                  + dzr * 4.0f * r * (E_ - 2.0f * Em);
        }
        s1d = -lam / 3.0f;
    } else {
        float lam;
        if (m_ps) {
            float sqbr = metal::precise::sqrt(z * r);
            lam = onembmr2
                  * (Pi_ + (-3.0f + 6.0f * r2 + 2.0f * z * r) * Em
                     - fourzr * E_) / (3.0f * sqbr);
        } else {
            float sq1 = metal::precise::sqrt(onembmr2);
            lam = (2.0f / 3.0f) * sq1
                  * (onembpr2 * Pi_ - (4.0f - 7.0f * r2 - z2) * E_);
        }
        s1d = -(lam + (z < r ? MP_TWO_PI : 0.0f)) / 3.0f;
    }

    float gc0 = 1.0f - u1 - 1.5f * u2;
    float gc1 = u1 + 2.0f * u2;
    float gc2 = -0.25f * u2;
    float inv_norm = 1.0f / (MP_PI * (1.0f - u1 / 3.0f - u2 / 6.0f));
    float fdev = (gc0 * s0d + gc1 * s1d + gc2 * s2d) * inv_norm;
"""


def _src(early_exit: str, tail: str) -> str:
    return (_CORE.replace("EARLY_EXIT", early_exit)
            .replace("EPS", _EPS)
            .replace("D_CON", _D_CON)
            .replace("KITE_FLOOR", _KITE_FLOOR) + tail)


_FWD_SRC = _src(
    "if (z >= 1.0f + r) { out[i] = 0.0f; return; }",
    "    out[i] = fdev;\n",
)

# analytic partials (continuous generic forms, as in vjp.sn_partials)
_VJP_TAIL = """
    float ds0dz, ds0dr, ds2dz, ds2dr;
    if (m_comp) {
        ds0dr = -2.0f * MP_PI * r;
        ds0dz = 0.0f;
        ds2dr = -4.0f * MP_PI * r + 8.0f * MP_PI * r * (r2 + z2);
        ds2dz = 8.0f * MP_PI * z * r2;
    } else {
        ds0dr = -2.0f * r * kap0;
        ds0dz = kite / z;
        ds2dr = -4.0f * r * kap0 + 8.0f * r * ((r2 + z2) * kap0 - kite);
        ds2dz = 2.0f * kite / z
              + (2.0f / z) * (4.0f * z2 * r2 * kap0
                              - (1.0f + r2 + z2) * kite);
    }
    float ds1dz, ds1dr;
    if (m_ps) {
        float sqbr = metal::precise::sqrt(z * r);
        ds1dr = -2.0f * r * onembmr2 * Em / sqbr;
        ds1dz = (2.0f / 3.0f) * r * onembmr2 * (2.0f * E_ - Em) / sqbr;
    } else {
        float sq1 = metal::precise::sqrt(onembmr2);
        ds1dr = -4.0f * r * sq1 * E_;
        ds1dz = -(4.0f / 3.0f) * r * sq1 * (E_ - 2.0f * Em);
    }
    float ctv = ct[i];
    gz[i] = ctv * (gc0 * ds0dz + gc1 * ds1dz + gc2 * ds2dz) * inv_norm;
    gr[i] = ctv * (gc0 * ds0dr + gc1 * ds1dr + gc2 * ds2dr) * inv_norm;
    gu1[i] = ctv * ((s1d - s0d) * inv_norm
                    + fdev * (MP_PI / 3.0f) * inv_norm);
    gu2[i] = ctv * ((-1.5f * s0d + 2.0f * s1d - 0.25f * s2d) * inv_norm
                    + fdev * (MP_PI / 6.0f) * inv_norm);
"""

_VJP_SRC = _src(
    "if (z >= 1.0f + r) { gz[i] = 0.0f; gr[i] = 0.0f; "
    "gu1[i] = 0.0f; gu2[i] = 0.0f; return; }",
    _VJP_TAIL,
)

# ---------------------------------------------------------------------------
# v2: model-level kernels — orbit folded in (consume the anvil (v, x)
# contract directly; ~12 B/pt total traffic)
# ---------------------------------------------------------------------------

# the photometric section of _CORE, reused verbatim after the orbit
_PHOT = "    float r2 = r * r;" + _CORE.split("float r2 = r * r;", 1)[1]

# epoch-centered circular orbit, in-register. Matches orbit.py exactly:
# tau wrap uses rint (mx.round is half-to-even on both streams, verified),
# z2 keeps the graph's (a*s)^2 + (b*c)^2 rounding order, z is floored at
# (10 eps)^2 before the far-side test. Exits must STORE (see v1 notes).
_ORBIT = """
    uint x = thread_position_in_grid.x;
    uint y = thread_position_in_grid.y;
    if (x >= (uint)npts) return;
    uint i = y * (uint)npts + x;

    float dt = xdat[x];
    float kk = xdat[(uint)npts + x];
    float pof = poff[y];
    float P = pref + pof;
    float tau = dt - (t0off[y] + kk * pof);
    float n_w = metal::rint(tau / P);
    tau -= P * n_w;
    float phi = MP_TWO_PI * tau / P;
    float sphi = metal::precise::sin(phi);
    float cphi = metal::precise::cos(phi);
    float r = rin[y];
    float av = ain[y];
    float bv = bin[y];
    float as_ = av * sphi;
    float bc_ = bv * cphi;
    float z2o = as_ * as_ + bc_ * bc_;
    float z = metal::precise::sqrt(max(z2o, KITE_FLOOR));
    if (cphi <= 0.0f || z >= 1.0f + r) { ORBIT_EXIT }

    float u1 = u1in[y];
    float u2 = u2in[y];
"""


def _model_src(exit_stores: str, tail: str) -> str:
    return (_HEADER_UNUSED_GUARD + _ORBIT.replace("ORBIT_EXIT", exit_stores)
            + _PHOT + tail).replace("EPS", _EPS).replace(
                "D_CON", _D_CON).replace("KITE_FLOOR", _KITE_FLOOR)


_HEADER_UNUSED_GUARD = ""  # placeholder to keep _model_src symmetrical

_MODEL_FWD_SRC = _model_src("out[i] = 0.0f; return;",
                            "    out[i] = fdev;\n")

# photometric partials + hand-derived orbit chain rules:
#   phi = 2 pi tau_w / P, tau_w = dt - t0off - kk*poff - n_w*P,
#   P = pref + poff, n_w treated locally constant (a.e. derivative):
#     dphi/dt0off = -2 pi / P
#     dphi/dpoff  = 2 pi ((-kk - n_w) P - tau_w) / P^2
#   z = sqrt(max((a s)^2 + (b c)^2, floor)):
#     dz/dphi = (a^2 - b^2) s c / z,  dz/da = a s^2 / z,  dz/db = b c^2 / z
#     (all zero when the floor clamps, matching the graph's max() grad)
_MODEL_VJP_TAIL = """
    float ds0dz, ds0dr, ds2dz, ds2dr;
    if (m_comp) {
        ds0dr = -2.0f * MP_PI * r;
        ds0dz = 0.0f;
        ds2dr = -4.0f * MP_PI * r + 8.0f * MP_PI * r * (r2 + z2);
        ds2dz = 8.0f * MP_PI * z * r2;
    } else {
        ds0dr = -2.0f * r * kap0;
        ds0dz = kite / z;
        ds2dr = -4.0f * r * kap0 + 8.0f * r * ((r2 + z2) * kap0 - kite);
        ds2dz = 2.0f * kite / z
              + (2.0f / z) * (4.0f * z2 * r2 * kap0
                              - (1.0f + r2 + z2) * kite);
    }
    float ds1dz, ds1dr;
    if (m_ps) {
        float sqbr = metal::precise::sqrt(z * r);
        ds1dr = -2.0f * r * onembmr2 * Em / sqbr;
        ds1dz = (2.0f / 3.0f) * r * onembmr2 * (2.0f * E_ - Em) / sqbr;
    } else {
        float sq1 = metal::precise::sqrt(onembmr2);
        ds1dr = -4.0f * r * sq1 * E_;
        ds1dz = -(4.0f / 3.0f) * r * sq1 * (E_ - 2.0f * Em);
    }
    float ctv = ct[i];
    float dFdz = (gc0 * ds0dz + gc1 * ds1dz + gc2 * ds2dz) * inv_norm;
    float dzdphi, dzda, dzdb;
    if (z2o <= KITE_FLOOR) {
        dzdphi = 0.0f; dzda = 0.0f; dzdb = 0.0f;
    } else {
        dzdphi = (av * av - bv * bv) * sphi * cphi / z;
        dzda = av * sphi * sphi / z;
        dzdb = bv * cphi * cphi / z;
    }
    float ctz = ctv * dFdz;
    float p_gt0 = ctz * dzdphi * (-MP_TWO_PI / P);
    float p_gp  = ctz * dzdphi
                  * (MP_TWO_PI * ((-kk - n_w) * P - tau) / (P * P));
    float p_ga  = ctz * dzda;
    float p_gb  = ctz * dzdb;
    float p_gr  = ctv * (gc0 * ds0dr + gc1 * ds1dr + gc2 * ds2dr) * inv_norm;
    float p_gu1 = ctv * ((s1d - s0d) * inv_norm
                         + fdev * (MP_PI / 3.0f) * inv_norm);
    float p_gu2 = ctv * ((-1.5f * s0d + 2.0f * s1d - 0.25f * s2d) * inv_norm
                         + fdev * (MP_PI / 6.0f) * inv_norm);
VJP_STORE
"""

# Two reduction strategies for the seven per-chain gradients.
#
# "grid" (v2 original): write full (n, m) partial arrays, reduce with
# mx.sum. 28 B/pt of transient DRAM traffic.
#
# "simd" (default): reduce inside the kernel. metal::simd_sum operates
# over the *active* lanes of the SIMD-group, which is exactly what is
# wanted here — lanes that took the early `return` are inactive and
# contribute nothing, so no predication, threadgroup memory, barrier or
# grid padding is needed (verified: benchmarks/v3_reduction_spike.py;
# MSL specifies these reductions "across all active threads"). The first
# still-active lane stores one value per simdgroup, so the outputs are
# (n, ceil(m/32)) instead of (n, m) — 1/32 the traffic. A simdgroup all
# of whose lanes exited writes nothing at all, which is why the call
# MUST pass init_value=0.0.
_VJP_STORE_GRID = """
    gt0[i] = p_gt0;
    gp[i]  = p_gp;
    ga[i]  = p_ga;
    gb[i]  = p_gb;
    gr[i]  = p_gr;
    gu1[i] = p_gu1;
    gu2[i] = p_gu2;
"""

_VJP_STORE_SIMD = """
    p_gt0 = metal::simd_sum(p_gt0);
    p_gp  = metal::simd_sum(p_gp);
    p_ga  = metal::simd_sum(p_ga);
    p_gb  = metal::simd_sum(p_gb);
    p_gr  = metal::simd_sum(p_gr);
    p_gu1 = metal::simd_sum(p_gu1);
    p_gu2 = metal::simd_sum(p_gu2);
    if (metal::simd_is_first()) {
        uint ngrp = ((uint)npts + 31u) / 32u;
        uint o = y * ngrp + x / 32u;
        gt0[o] = p_gt0;
        gp[o]  = p_gp;
        ga[o]  = p_ga;
        gb[o]  = p_gb;
        gr[o]  = p_gr;
        gu1[o] = p_gu1;
        gu2[o] = p_gu2;
    }
"""

_MODEL_VJP_SRC_GRID = _model_src(
    "gt0[i] = 0.0f; gp[i] = 0.0f; gr[i] = 0.0f; gb[i] = 0.0f; "
    "ga[i] = 0.0f; gu1[i] = 0.0f; gu2[i] = 0.0f; return;",
    _MODEL_VJP_TAIL.replace("VJP_STORE", _VJP_STORE_GRID),
)

# the simd path needs no exit stores at all: an exited lane is inactive
# and drops out of simd_sum by itself.
_MODEL_VJP_SRC_SIMD = _model_src(
    "return;",
    _MODEL_VJP_TAIL.replace("VJP_STORE", _VJP_STORE_SIMD),
)

_kernels: dict = {}
_metal_ok: bool | None = None


def _get_kernels():
    if "fwd" not in _kernels:
        _kernels["fwd"] = mx.fast.metal_kernel(
            name="mp_flux_fwd",
            input_names=["zin", "rin", "u1in", "u2in", "npts"],
            output_names=["out"],
            header=_HEADER,
            source=_FWD_SRC,
        )
        _kernels["vjp"] = mx.fast.metal_kernel(
            name="mp_flux_vjp",
            input_names=["zin", "rin", "u1in", "u2in", "ct", "npts"],
            output_names=["gz", "gr", "gu1", "gu2"],
            header=_HEADER,
            source=_VJP_SRC,
        )
    return _kernels


def _get_model_kernels():
    if "model_fwd" not in _kernels:
        _kernels["model_fwd"] = mx.fast.metal_kernel(
            name="mp_model_fwd",
            input_names=["xdat", "t0off", "poff", "rin", "bin", "ain",
                         "u1in", "u2in", "pref", "npts"],
            output_names=["out"],
            header=_HEADER,
            source=_MODEL_FWD_SRC,
        )
        for tag, src in (("grid", _MODEL_VJP_SRC_GRID),
                         ("simd", _MODEL_VJP_SRC_SIMD)):
            _kernels["model_vjp_" + tag] = mx.fast.metal_kernel(
                name="mp_model_vjp_" + tag,
                input_names=["xdat", "t0off", "poff", "rin", "bin", "ain",
                             "u1in", "u2in", "ct", "pref", "npts"],
                output_names=["gt0", "gp", "gr", "gb", "ga", "gu1", "gu2"],
                header=_HEADER,
                source=src,
            )
    return _kernels


def make_model_core_metal(period_ref: float, reduce: str = "simd"):
    """v2 model-level kernel: (x, t0_off, p_off, r, b, a, u1, u2) ->
    flux deviation (n, m), the whole orbit + photometric chain in one
    kernel (~12 B/pt of traffic). q -> u and the df0 offset stay in the
    MLX graph so their gradients ride ordinary autodiff. period_ref is a
    runtime kernel input, never baked into source.

    ``reduce`` selects how the VJP sums its per-point gradients over the
    data axis: "simd" (default) reduces inside the kernel, "grid" writes
    full (n, m) partial arrays and reduces with mx.sum. They compute
    identical per-point partials and differ only in summation order, so
    "grid" doubles as a parity oracle for "simd" in the tests.
    """
    if reduce not in ("simd", "grid"):
        raise ValueError('reduce must be "simd" or "grid"')
    pref = float(period_ref)

    @mx.custom_function
    def core(x2d, t0_off, p_off, r, b, a, u1, u2):
        n = t0_off.shape[0]
        m = x2d.shape[1]
        k = _get_model_kernels()["model_fwd"]
        return k(inputs=[x2d, t0_off, p_off, r, b, a, u1, u2, pref, int(m)],
                 output_shapes=[(n, m)], output_dtypes=[mx.float32],
                 grid=(m, n, 1), threadgroup=(256, 1, 1))[0]

    @core.vjp
    def core_vjp(primals, cotangent, output):
        x2d, t0_off, p_off, r, b, a, u1, u2 = primals
        ct = cotangent if isinstance(cotangent, mx.array) else cotangent[0]
        n = t0_off.shape[0]
        m = x2d.shape[1]
        k = _get_model_kernels()["model_vjp_" + reduce]
        inputs = [x2d, t0_off, p_off, r, b, a, u1, u2, ct, pref, int(m)]
        if reduce == "simd":
            # one value per 32-lane simdgroup; init_value is load-bearing
            # (a fully-exited simdgroup never writes).
            cols = (m + 31) // 32
            outs = k(inputs=inputs, output_shapes=[(n, cols)] * 7,
                     output_dtypes=[mx.float32] * 7, init_value=0.0,
                     grid=(m, n, 1), threadgroup=(256, 1, 1))
        else:
            outs = k(inputs=inputs, output_shapes=[(n, m)] * 7,
                     output_dtypes=[mx.float32] * 7,
                     grid=(m, n, 1), threadgroup=(256, 1, 1))
        sums = [mx.sum(o, axis=1) for o in outs]
        return (mx.zeros_like(x2d), *sums)

    return core


def metal_available() -> bool:
    """Probe: can the fused kernel actually run on this machine?"""
    global _metal_ok
    if _metal_ok is None:
        try:
            k = _get_kernels()["fwd"]
            out = k(inputs=[mx.zeros((1, 4)), mx.full((1,), 0.1),
                            mx.full((1,), 0.1), mx.full((1,), 0.1), 4],
                    output_shapes=[(1, 4)], output_dtypes=[mx.float32],
                    grid=(4, 1, 1), threadgroup=(64, 1, 1))[0]
            mx.eval(out)
            _metal_ok = True
        except Exception:
            _metal_ok = False
    return _metal_ok


def _gpu_stream_active() -> bool:
    try:
        return mx.default_device() == mx.Device(mx.DeviceType.gpu)
    except Exception:
        return False


def _canon_param(p, n, dtype):
    """Parameter -> (n,) fp32 array (accepts scalar, 0-d, (n,), (n,1), (1,1))."""
    if not isinstance(p, mx.array):
        return mx.full((n,), float(p), dtype=dtype)
    if p.ndim == 0:
        return mx.broadcast_to(mx.reshape(p, (1,)), (n,))
    q = mx.reshape(p, (-1,))
    if q.shape[0] == n:
        return q
    if q.shape[0] == 1:
        return mx.broadcast_to(q, (n,))
    raise ValueError("parameter shape incompatible with z")


@mx.custom_function
def _flux_dev_metal_core(z2d: mx.array, r: mx.array, u1: mx.array,
                         u2: mx.array) -> mx.array:
    n, m = z2d.shape
    k = _get_kernels()["fwd"]
    return k(inputs=[z2d, r, u1, u2, int(m)],
             output_shapes=[(n, m)], output_dtypes=[mx.float32],
             grid=(m, n, 1), threadgroup=(256, 1, 1))[0]


@_flux_dev_metal_core.vjp
def _flux_dev_metal_vjp(primals, cotangent, output):
    z2d, r, u1, u2 = primals
    ct = cotangent if isinstance(cotangent, mx.array) else cotangent[0]
    n, m = z2d.shape
    k = _get_kernels()["vjp"]
    gz, gr, gu1, gu2 = k(
        inputs=[z2d, r, u1, u2, ct, int(m)],
        output_shapes=[(n, m)] * 4, output_dtypes=[mx.float32] * 4,
        grid=(m, n, 1), threadgroup=(256, 1, 1))
    return gz, mx.sum(gr, axis=1), mx.sum(gu1, axis=1), mx.sum(gu2, axis=1)


def flux_dev_metal(z: mx.array, r, u1, u2) -> mx.array:
    """F - 1 via the fused Metal kernels (fp32, GPU stream); silently
    falls back to flux_dev_analytic for fp64, CPU streams, unsupported
    layouts, or machines where the kernel probe fails."""
    if (not isinstance(z, mx.array) or z.dtype != mx.float32
            or z.ndim not in (1, 2) or not _gpu_stream_active()
            or not metal_available()):
        return flux_dev_analytic(z, r, u1, u2)

    squeeze = z.ndim == 1
    z2d = z[None, :] if squeeze else z
    n_param = max(
        (p.shape[0] if isinstance(p, mx.array) and p.ndim >= 1 else 1)
        for p in (r, u1, u2))
    n = max(z2d.shape[0], n_param)
    try:
        rc = _canon_param(r, n, z.dtype)
        u1c = _canon_param(u1, n, z.dtype)
        u2c = _canon_param(u2, n, z.dtype)
    except ValueError:
        return flux_dev_analytic(z, r, u1, u2)
    if z2d.shape[0] != n:
        z2d = mx.broadcast_to(z2d, (n, z2d.shape[1]))

    out = _flux_dev_metal_core(z2d, rc, u1c, u2c)
    return out[0] if squeeze and n == 1 else out
