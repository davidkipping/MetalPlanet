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

import math

import mlx.core as mx
import numpy as np

from .vjp import _unbroadcast, flux_dev_analytic

__all__ = ["flux_dev_metal", "flux_dev_from_tau", "metal_available",
           "make_model_core_metal", "make_ecc_core_metal"]

_EPS = "1.1920929e-07f"
_D_CON = "3.4526698e-04f"          # sqrt(fp32 eps)
_KITE_FLOOR = "1.4210855e-12f"     # (10 eps)^2

_HEADER = """
constant float MP_PI = 3.14159265358979f;
constant float MP_TWO_PI = 6.28318530717959f;
constant float MP_HALF_PI = 1.57079632679490f;
constant float MP_MK_A = 7.651638290f;      // 3 pi / (pi - 6/pi)
constant float MP_MK_B = 1.298982460f;      // 1.6 / (pi - 6/pi)

// cbrt(c)^2 for the Markley starter. metal::precise::powr costs ~220x
// a multiply and was ~20% of the orbit's time; this is the inverse-cbrt
// bit trick plus three division-free Newton steps
// (r <- r (4 - c r^3) / 3, converging to r = c^-1/3), then cbrt = c r^2.
// The seed constant is the exact (4/3) * 0x3f800000 of the standard
// i_y = (1 - p) B + p i_x construction with p = -1/3, not a tuned one.
// Measured max relative error 1.7e-6 in-kernel over c in
// [1e-15, 1e15] (fp32 Newton steps; 1.2e-6 in exact arithmetic); the
// starter itself only needs ~1e-4, and the fifth-order refinement
// downstream squashes what is left. c is always >= 1e-15 here (a sqrt
// of a floored argument), so denormals never reach the trick.
inline float mp_cbrt2(float c) {
    int ic = as_type<int>(c);
    float r = as_type<float>(0x54AAAAAB - ic / 3);
    r = r * (1.33333333f - 0.33333333f * c * r * r * r);
    r = r * (1.33333333f - 0.33333333f * c * r * r * r);
    r = r * (1.33333333f - 0.33333333f * c * r * r * r);
    float y = c * r * r;
    return y * y;
}
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


# ds_n/dz and ds_n/dr — the continuous generic forms of
# vjp.sn_partials, shared verbatim by every VJP tail below.
_PHOT_PARTIALS = '''
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
'''

def _subst(src: str) -> str:
    """Expand the shared fragments and numeric constants. Applied to the
    WHOLE kernel body, tail included — a marker left unexpanded is not a
    Python error but a Metal compile failure, which aborts the process."""
    out = (src.replace("PHOT_PARTIALS", _PHOT_PARTIALS)
              .replace("EPS", _EPS)
              .replace("D_CON", _D_CON)
              .replace("KITE_FLOOR", _KITE_FLOOR))
    for marker in ("PHOT_PARTIALS", "EARLY_EXIT", "ORBIT_EXIT", "VJP_STORE"):
        assert marker not in out, f"unexpanded {marker} in kernel source"
    return out


def _src(early_exit: str, tail: str) -> str:
    return _subst(_CORE.replace("EARLY_EXIT", early_exit) + tail)


_FWD_SRC = _src(
    "if (z >= 1.0f + r) { out[i] = 0.0f; return; }",
    "    out[i] = fdev;\n",
)

# analytic partials (continuous generic forms, as in vjp.sn_partials)
_VJP_TAIL = """
PHOT_PARTIALS
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

# ---------------------------------------------------------------------------
# THE model kernel: the transit-anchored orbit (anchored.py) folded in
# beside the photometric core. One kernel serves circular and eccentric
# orbits — the anchored formulation is exact at e = 0, and a per-chain
# branch on e == 0 skips the Kepler solve there. The chain index is
# uniform across a threadgroup, so that branch cannot diverge within a
# simdgroup; measured cost to genuinely eccentric chains ~1%.
# ---------------------------------------------------------------------------

# Per-chain orbit constants, packed into one (n, NORB) array so the
# kernel stays well inside Metal's buffer budget. Order is fixed by
# _ORB_COLS and shared with anchored.pack_orbit_constants.
_ORB_COLS = ("ecw", "esw", "es", "ec", "b1", "a2", "b2",
             "ecc", "e0", "mtra", "ci")
NORB = len(_ORB_COLS)

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

    uint ob = y * NORB_C;
    float o_ecw = orb[ob + 0u];
    float o_esw = orb[ob + 1u];
    float o_es  = orb[ob + 2u];
    float o_ec  = orb[ob + 3u];
    float o_b1  = orb[ob + 4u];
    float o_a2  = orb[ob + 5u];
    float o_b2  = orb[ob + 6u];
    float o_e   = orb[ob + 7u];
    float o_E0  = orb[ob + 8u];
    float o_Mt  = orb[ob + 9u];
    float o_ci  = orb[ob + 10u];

    // (sin delta, cos delta) and 1 - cos delta. For a CIRCULAR chain
    // delta == phi exactly (the anchored form degenerates), so the Markley
    // starter and the refinement below are dead work: skip them. o_e is
    // per-chain and the chain index is uniform over the threadgroup, so
    // this branch is simdgroup-uniform and costs eccentric chains ~1%.
    float sind, cosd, omcf;
    if (o_e == 0.0f) {
        sind = metal::precise::sin(phi);
        cosd = metal::precise::cos(phi);
        omcf = (cosd > 0.0f) ? (sind * sind / (1.0f + cosd)) : (1.0f - cosd);
    } else {
        // fold the STANDARD mean anomaly; carry phi along with the fold so
        // the anchored residual below stays consistent with it.
        float Mm = phi + o_Mt;
        float n_m = metal::rint(Mm / MP_TWO_PI);
        float M_w = Mm - MP_TWO_PI * n_m;
        float phi_w = phi - MP_TWO_PI * n_m;

        // Markley starter on |M| (~1e-4): its own cancellation against E0 is
        // irrelevant at that accuracy, and the refinement below runs wholly
        // in the anchored, O(e)-conditioned coefficients.
        float sgn = (M_w >= 0.0f) ? 1.0f : -1.0f;
        float Ma = fabs(M_w);
        float ome = 1.0f - o_e;
        float M2 = Ma * Ma;
        float alph = MP_MK_A + MP_MK_B * (MP_PI - Ma) / (1.0f + o_e);
        float dstn = 3.0f * ome + alph * o_e;
        float alphad = alph * dstn;
        float rstn = (3.0f * alphad * (dstn - ome) + M2) * Ma;
        float qstn = 2.0f * alphad * ome - M2;
        float q2stn = qstn * qstn;
        float cstn = fabs(rstn)
                   + metal::precise::sqrt(max(q2stn * qstn + rstn * rstn, 1e-30f));
        float wstn = MP_CBRT2(cstn);
        float d0 = (2.0f * rstn * wstn / (wstn * wstn + wstn * qstn + q2stn)
                    + Ma) / dstn * sgn - o_E0;

        float sd = metal::precise::sin(d0);      // the ONLY trig call
        float cd = metal::precise::cos(d0);
        float omc = (cd > 0.0f) ? (sd * sd / (1.0f + cd)) : (1.0f - cd);
        float fa0 = d0 + o_es * omc - o_ec * sd - phi_w;
        float fa1 = 1.0f + o_es * sd - o_ec * cd;    // = 1 - e cos E > 0
        float fa2 = o_es * cd + o_ec * sd;           // = e sin E
        float fa3 = -o_es * sd + o_ec * cd;          // = e cos E
        float c3 = -fa0 / (fa1 - 0.5f * fa0 * fa2 / fa1);
        float c4 = -fa0 / (fa1 + 0.5f * c3 * fa2 + c3 * c3 * fa3 / 6.0f);
        float dcorr = -fa0 / (fa1 + 0.5f * c4 * fa2 + c4 * c4 * fa3 / 6.0f
                              - c4 * c4 * c4 * fa2 / 24.0f);
        float dc2 = dcorr * dcorr;
        float sdd = dcorr * (1.0f - dc2 / 6.0f * (1.0f - dc2 / 20.0f));
        float cdd = 1.0f - dc2 * 0.5f * (1.0f - dc2 / 12.0f);
        sind = sd * cdd + cd * sdd;
        cosd = cd * cdd - sd * sdd;
        omcf = (cosd > 0.0f) ? (sind * sind / (1.0f + cosd))
                                   : (1.0f - cosd);
    }

    float r = rin[y];
    float av = ain[y];
    float uu = av * (-o_ecw * omcf - o_b1 * sind);
    float vv = av * (o_a2 * cosd - o_b2 * sind - o_esw);
    float vc = vv * o_ci;
    float z2o = uu * uu + vc * vc;
    float z = metal::precise::sqrt(max(z2o, KITE_FLOOR));
    if (vv <= 0.0f || z >= 1.0f + r) { ORBIT_EXIT }

    float u1 = u1in[y];
    float u2 = u2in[y];
"""


def _model_src(exit_stores: str, tail: str) -> str:
    return _subst(_ORBIT.replace("ORBIT_EXIT", exit_stores) + _PHOT
                  + tail).replace("NORB_C", f"{NORB}u").replace(
                      "MP_CBRT2", _CBRT2)

# cbrt(c)^2 for the Markley starter. E2 replaces precise::powr (~220x a
# multiply, ~20% of orbit time) with a bit-trick + Newton cbrt.
_CBRT2 = "mp_cbrt2"

_MODEL_FWD_SRC = _model_src("out[i] = 0.0f; return;",
                            "    out[i] = fdev;\n")

#: per-point gradient slots emitted by the VJP, in output order. The
#: first seven are shared by every chain; the last seven are the
#: eccentric-only coefficients and are skipped on circular chains.
_GRAD_SLOTS = ("t0", "p", "r", "a", "u1", "u2", "ci",
               "ecw", "esw", "es", "ec", "b1", "a2", "b2")
NGRAD = len(_GRAD_SLOTS)

# Transit-anchored chain rules. With gu = u/z, gv = v ci^2 / z:
#   du/ddelta = a (-ecw sin d - b1 cos d)      [d(1-cos d)/dd = sin d]
#   dv/ddelta = a (-a2 sin d - b2 cos d)
#   dz/ddelta = gu du/ddelta + gv dv/ddelta
#   D = dg/ddelta = 1 + es sin d - ec cos d = 1 - e cos E > 0, so the
#   implicit rule gives ddelta/dphi = 1/D, ddelta/des = -(1-cos d)/D,
#   ddelta/dec = sin d / D.
# The tail is linear in each anchored coefficient, so
#   dz/decw = -gu a (1-cos d),  dz/desw = -gv a,
#   dz/db1  = -gu a sin d,      dz/da2  =  gv a cos d,
#   dz/db2  = -gv a sin d,      dz/dci  =  v^2 ci / z,
#   dz/da   =  z / a            (exact: z is homogeneous of degree 1 in a)
# and the phi wrap chains are the original circular kernel's, verbatim.
# ecc / e0 / mtra seed only the starter and the 2-pi fold, whose exact
# gradient contribution is zero (implicit function theorem; rint locally
# constant) — they are detached in anchored.pack_orbit_constants.
_MODEL_VJP_TAIL = """
PHOT_PARTIALS
    float ctv = ct[i];
    float dFdz = (gc0 * ds0dz + gc1 * ds1dz + gc2 * ds2dz) * inv_norm;
    float ctz = ctv * dFdz;

    float p_r  = ctv * (gc0 * ds0dr + gc1 * ds1dr + gc2 * ds2dr) * inv_norm;
    float p_u1 = ctv * ((s1d - s0d) * inv_norm
                        + fdev * (MP_PI / 3.0f) * inv_norm);
    float p_u2 = ctv * ((-1.5f * s0d + 2.0f * s1d - 0.25f * s2d) * inv_norm
                        + fdev * (MP_PI / 6.0f) * inv_norm);

    float p_t0 = 0.0f, p_p = 0.0f, p_a = 0.0f, p_ci = 0.0f;
    float p_ecw = 0.0f, p_esw = 0.0f, p_es = 0.0f, p_ec = 0.0f;
    float p_b1 = 0.0f, p_a2 = 0.0f, p_b2 = 0.0f;
    bool ecc_chain = (o_e != 0.0f);           // simdgroup-uniform
    if (z2o > KITE_FLOOR) {
        float gu = uu / z;
        float gv = vc * o_ci / z;
        float dud = av * (-o_ecw * sind - o_b1 * cosd);
        float dvd = av * (-o_a2 * sind - o_b2 * cosd);
        float dzdd = gu * dud + gv * dvd;
        float Dk = 1.0f + o_es * sind - o_ec * cosd;   // == 1 when circular
        float dzdphi = dzdd / Dk;
        p_t0  = ctz * dzdphi * (-MP_TWO_PI / P);
        p_p   = ctz * dzdphi
                * (MP_TWO_PI * ((-kk - n_w) * P - tau) / (P * P));
        p_a   = ctz * (z / av);
        p_ci  = ctz * (vv * vv * o_ci / z);
        if (ecc_chain) {
            p_ecw = ctz * gu * (-av * omcf);
            p_esw = ctz * gv * (-av);
            p_b1  = ctz * gu * (-av * sind);
            p_a2  = ctz * gv * (av * cosd);
            p_b2  = ctz * gv * (-av * sind);
            p_es  = ctz * dzdd * (-omcf / Dk);
            p_ec  = ctz * dzdd * (sind / Dk);
        }
    }

    // reductions over the ACTIVE lanes (exited lanes drop out by
    // themselves; init_value=0 covers fully-exited simdgroups). The seven
    // eccentric-only slots are neither reduced nor stored on a circular
    // chain: they read as the init_value zero.
    p_t0  = metal::simd_sum(p_t0);
    p_p   = metal::simd_sum(p_p);
    p_r   = metal::simd_sum(p_r);
    p_a   = metal::simd_sum(p_a);
    p_u1  = metal::simd_sum(p_u1);
    p_u2  = metal::simd_sum(p_u2);
    p_ci  = metal::simd_sum(p_ci);
    if (ecc_chain) {
        p_ecw = metal::simd_sum(p_ecw);
        p_esw = metal::simd_sum(p_esw);
        p_es  = metal::simd_sum(p_es);
        p_ec  = metal::simd_sum(p_ec);
        p_b1  = metal::simd_sum(p_b1);
        p_a2  = metal::simd_sum(p_a2);
        p_b2  = metal::simd_sum(p_b2);
    }
    if (metal::simd_is_first()) {
        uint ngrp = ((uint)npts + 31u) / 32u;
        uint o = y * NGRAD_C * ngrp + x / 32u;
        gpart[o +  0u * ngrp] = p_t0;
        gpart[o +  1u * ngrp] = p_p;
        gpart[o +  2u * ngrp] = p_r;
        gpart[o +  3u * ngrp] = p_a;
        gpart[o +  4u * ngrp] = p_u1;
        gpart[o +  5u * ngrp] = p_u2;
        gpart[o +  6u * ngrp] = p_ci;
        if (ecc_chain) {
            gpart[o +  7u * ngrp] = p_ecw;
            gpart[o +  8u * ngrp] = p_esw;
            gpart[o +  9u * ngrp] = p_es;
            gpart[o + 10u * ngrp] = p_ec;
            gpart[o + 11u * ngrp] = p_b1;
            gpart[o + 12u * ngrp] = p_a2;
            gpart[o + 13u * ngrp] = p_b2;
        }
    }
"""

_MODEL_VJP_SRC = _model_src("return;", _MODEL_VJP_TAIL).replace(
    "NGRAD_C", f"{NGRAD}u")

# ---------------------------------------------------------------------------
# tau-input kernel with in-kernel exposure integration (flux_dev_from_tau)
#
# The existing kernels evaluate ONE point per thread. Integrating a finite
# exposure means evaluating many sub-exposure nodes per output point, and the
# whole reason to do it in-kernel is that those nodes never become an MLX
# array: a caller that expands them pays n_sub x the memory in every forward
# intermediate and every gradient grid.
#
# That requires the photometric core as a callable device FUNCTION rather than
# an inlined straight-line fragment. _PHOT is already pure in (z, r, u1, u2)
# with no early return, so it wraps verbatim -- the numerics below are the
# same text the other kernels compile, not a reimplementation.
# ---------------------------------------------------------------------------

_PHOT_FN = """
inline float mp_phot(float z, float r, float u1, float u2) {
    // the photometric core assumes an overlap: out of transit is exactly
    // zero and must be short-circuited here, because the early exit that
    // does this in the point kernels lives in their ORBIT fragment.
    if (z >= 1.0f + r) return 0.0f;
PHOT_BODY
    return fdev;
}

// fdev plus the four analytic partials, for the backward pass
inline float mp_phot_d(float z, float r, float u1, float u2,
                       thread float *dFdz, thread float *dFdr,
                       thread float *dFdu1, thread float *dFdu2) {
    if (z >= 1.0f + r) {
        *dFdz = 0.0f; *dFdr = 0.0f; *dFdu1 = 0.0f; *dFdu2 = 0.0f;
        return 0.0f;
    }
PHOT_BODY
PHOT_PARTIALS
    *dFdz  = (gc0 * ds0dz + gc1 * ds1dz + gc2 * ds2dz) * inv_norm;
    *dFdr  = (gc0 * ds0dr + gc1 * ds1dr + gc2 * ds2dr) * inv_norm;
    *dFdu1 = (s1d - s0d) * inv_norm + fdev * (MP_PI / 3.0f) * inv_norm;
    *dFdu2 = (-1.5f * s0d + 2.0f * s1d - 0.25f * s2d) * inv_norm
             + fdev * (MP_PI / 6.0f) * inv_norm;
    return fdev;
}

// z(tau) for a circular orbit, matching orbit.separation_circular: the
// far side is pushed beyond contact rather than mirrored.
inline float mp_z_of_tau(float tau, float per, float a, float b,
                         thread float *sphi, thread float *cphi) {
    float phi = MP_TWO_PI * tau / per;
    *sphi = metal::precise::sin(phi);
    *cphi = metal::precise::cos(phi);
    float as_ = a * (*sphi);
    float bc_ = b * (*cphi);
    float z2 = as_ * as_ + bc_ * bc_;
    return metal::precise::sqrt(max(z2, KITE_FLOOR));
}
"""


def _phot_header() -> str:
    return _subst(_PHOT_FN.replace("PHOT_BODY", _PHOT))


# Integration modes, shared with the Python wrapper.
_INT_NONE, _INT_CONTACT, _INT_SUPER = 0, 1, 2

# Shared preamble: indices, per-chain parameters, and the exposure window.
_TAU_HEAD = """
    uint x = thread_position_in_grid.x;
    uint y = thread_position_in_grid.y;
    if (x >= (uint)npts) return;
    uint i = y * (uint)npts + x;

    float tau0 = taui[i];
    float per = perin[y];
    float av  = ain[y];
    float bv  = bin[y];
    float r   = rin[y];
    float u1  = u1in[y];
    float u2  = u2in[y];
    float hw_exp = 0.5f * expt;
"""

# The contact-split window. Contacts arrive per chain in TAU units and are
# ordered; clamping into [t1, t2] preserves that order, so an interval whose
# contact lies outside the window simply has zero width and contributes
# nothing -- the branchless five-interval split, same as exposure.py.
#
# d(edge)/d(tau0) is 0 or 1 and nothing else: the window ends move with the
# point, an interior contact does not. That is what makes the exact
# derivative of the QUADRATURE (not merely of the integral it approximates)
# cheap enough to do in-kernel.
_TAU_EDGES = """
    float t1 = tau0 - hw_exp, t2 = tau0 + hw_exp;
    float edge[6]; float dedge[6];
    edge[0] = t1;  dedge[0] = 1.0f;
    edge[5] = t2;  dedge[5] = 1.0f;
    for (int c = 0; c < 4; ++c) {
        float ct_ = cs[y * 4u + (uint)c];
        float e = ct_ < t1 ? t1 : (ct_ > t2 ? t2 : ct_);
        edge[c + 1] = e;
        dedge[c + 1] = (ct_ < t1 || ct_ > t2) ? 1.0f : 0.0f;
    }
"""

_TAU_FWD_SRC = _TAU_HEAD + """
    float sphi, cphi, z, acc;
    if (mode == MODE_NONE) {
        z = mp_z_of_tau(tau0, per, av, bv, &sphi, &cphi);
        out[i] = (cphi <= 0.0f) ? 0.0f : mp_phot(z, r, u1, u2);
        return;
    }
    if (mode == MODE_SUPER) {
        acc = 0.0f;
        for (int j = 0; j < nsub; ++j) {
            float frac = (nsub == 1) ? 0.5f
                                     : (float)j / (float)(nsub - 1);
            float tt = t1_of(tau0, hw_exp) + 2.0f * hw_exp * frac;
            z = mp_z_of_tau(tt, per, av, bv, &sphi, &cphi);
            acc += (cphi <= 0.0f) ? 0.0f : mp_phot(z, r, u1, u2);
        }
        out[i] = acc / (float)nsub;
        return;
    }
""" + _TAU_EDGES + """
    float A = 0.0f, S = 0.0f;
    for (int iv = 0; iv < 5; ++iv) {
        float lo = edge[iv], hi = edge[iv + 1];
        float mid = 0.5f * (lo + hi), hw = 0.5f * (hi - lo);
        for (int j = 0; j < ngl; ++j) {
            float w = hw * wg[j];
            float tt = mid + hw * xg[j];
            z = mp_z_of_tau(tt, per, av, bv, &sphi, &cphi);
            float f = (cphi <= 0.0f) ? 0.0f : mp_phot(z, r, u1, u2);
            A += w * f;
            S += w;
        }
    }
    out[i] = (S > 0.0f) ? (A / S) : 0.0f;
"""

# Backward. Per-point gradient in tau (full shape, like flux_dev_metal's gz);
# per-chain gradients in (per, a, b, r, u1, u2), reduced in-kernel by
# simd_sum over the ACTIVE lanes exactly as the model VJP does.
#
# dz/dtau  = [(a^2 - b^2) sin phi cos phi / z] (2 pi / per)
# df/dper  = -(df/dtau) * tt / per       (the same chain, reused)
# dz/da    = a sin^2 phi / z ,  dz/db = b cos^2 phi / z
_TAU_PAR = ("per", "a", "b", "r", "u1", "u2")
NTAUPAR = len(_TAU_PAR)

_TAU_VJP_SRC = _TAU_HEAD + """
    float ctv = ct[i];
    float sphi, cphi, z, dFdz, dFdr, dFdu1, dFdu2;
    float p_per = 0.0f, p_a = 0.0f, p_b = 0.0f;
    float p_r = 0.0f, p_u1 = 0.0f, p_u2 = 0.0f;
    float g_tau = 0.0f;

    if (mode == MODE_NONE || mode == MODE_SUPER) {
        int n_node = (mode == MODE_NONE) ? 1 : nsub;
        float scale = 1.0f / (float)n_node;
        for (int j = 0; j < n_node; ++j) {
            float tt;
            if (mode == MODE_NONE) {
                tt = tau0;
            } else {
                float frac = (nsub == 1) ? 0.5f
                                         : (float)j / (float)(nsub - 1);
                tt = t1_of(tau0, hw_exp) + 2.0f * hw_exp * frac;
            }
            z = mp_z_of_tau(tt, per, av, bv, &sphi, &cphi);
            if (cphi <= 0.0f) continue;
            mp_phot_d(z, r, u1, u2, &dFdz, &dFdr, &dFdu1, &dFdu2);
            float zi = (z > 0.0f) ? (1.0f / z) : 0.0f;
            float dzdphi = (av * av - bv * bv) * sphi * cphi * zi;
            float dfdtau = dFdz * dzdphi * (MP_TWO_PI / per);
            g_tau += ctv * scale * dfdtau;
            p_per += ctv * scale * (-dfdtau * tt / per);
            p_a   += ctv * scale * dFdz * (av * sphi * sphi * zi);
            p_b   += ctv * scale * dFdz * (bv * cphi * cphi * zi);
            p_r   += ctv * scale * dFdr;
            p_u1  += ctv * scale * dFdu1;
            p_u2  += ctv * scale * dFdu2;
        }
    } else {
""" + _TAU_EDGES + """
        // accumulate the quadrature AND its exact derivative in tau0
        float A = 0.0f, S = 0.0f, dA = 0.0f, dS = 0.0f;
        float a_per = 0.0f, a_a = 0.0f, a_b = 0.0f;
        float a_r = 0.0f, a_u1 = 0.0f, a_u2 = 0.0f;
        for (int iv = 0; iv < 5; ++iv) {
            float lo = edge[iv], hi = edge[iv + 1];
            float dlo = dedge[iv], dhi = dedge[iv + 1];
            float mid = 0.5f * (lo + hi), hw = 0.5f * (hi - lo);
            float dmid = 0.5f * (dlo + dhi), dhw = 0.5f * (dhi - dlo);
            for (int j = 0; j < ngl; ++j) {
                float w = hw * wg[j];
                float dw = dhw * wg[j];
                float tt = mid + hw * xg[j];
                float dtt = dmid + dhw * xg[j];
                z = mp_z_of_tau(tt, per, av, bv, &sphi, &cphi);
                float f = 0.0f, dfdtt = 0.0f;
                if (cphi > 0.0f) {
                    f = mp_phot_d(z, r, u1, u2, &dFdz, &dFdr, &dFdu1, &dFdu2);
                    float zi = (z > 0.0f) ? (1.0f / z) : 0.0f;
                    float dzdphi = (av * av - bv * bv) * sphi * cphi * zi;
                    dfdtt = dFdz * dzdphi * (MP_TWO_PI / per);
                    // parameter partials integrate with the SAME rule; the
                    // true integral does not depend on where it is split, so
                    // the contacts' own parameter dependence is not chased.
                    a_per += w * (-dfdtt * tt / per);
                    a_a   += w * dFdz * (av * sphi * sphi * zi);
                    a_b   += w * dFdz * (bv * cphi * cphi * zi);
                    a_r   += w * dFdr;
                    a_u1  += w * dFdu1;
                    a_u2  += w * dFdu2;
                }
                A += w * f;   S += w;
                dA += dw * f + w * dfdtt * dtt;
                dS += dw;
            }
        }
        if (S > 0.0f) {
            float inv = 1.0f / S;
            g_tau = ctv * (dA * S - A * dS) * inv * inv;
            p_per = ctv * a_per * inv;  p_a  = ctv * a_a  * inv;
            p_b   = ctv * a_b   * inv;  p_r  = ctv * a_r  * inv;
            p_u1  = ctv * a_u1  * inv;  p_u2 = ctv * a_u2 * inv;
        }
    }

    gtau[i] = g_tau;
    p_per = metal::simd_sum(p_per);
    p_a   = metal::simd_sum(p_a);
    p_b   = metal::simd_sum(p_b);
    p_r   = metal::simd_sum(p_r);
    p_u1  = metal::simd_sum(p_u1);
    p_u2  = metal::simd_sum(p_u2);
    if (metal::simd_is_first()) {
        uint ngrp = ((uint)npts + 31u) / 32u;
        uint o = y * NTAUPAR_C * ngrp + x / 32u;
        gpar[o + 0u * ngrp] = p_per;
        gpar[o + 1u * ngrp] = p_a;
        gpar[o + 2u * ngrp] = p_b;
        gpar[o + 3u * ngrp] = p_r;
        gpar[o + 4u * ngrp] = p_u1;
        gpar[o + 5u * ngrp] = p_u2;
    }
"""


def _tau_src(body: str) -> str:
    return _subst(body).replace("MODE_NONE", str(_INT_NONE)).replace(
        "MODE_SUPER", str(_INT_SUPER)).replace(
        "NTAUPAR_C", f"{NTAUPAR}u").replace(
        "t1_of(tau0, hw_exp)", "(tau0 - hw_exp)")


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
            input_names=["xdat", "t0off", "poff", "rin", "ain", "orb",
                         "u1in", "u2in", "pref", "npts"],
            output_names=["out"],
            header=_HEADER,
            source=_MODEL_FWD_SRC,
        )
        _kernels["model_vjp"] = mx.fast.metal_kernel(
            name="mp_model_vjp",
            input_names=["xdat", "t0off", "poff", "rin", "ain", "orb",
                         "u1in", "u2in", "ct", "pref", "npts"],
            output_names=["gpart"],
            header=_HEADER,
            source=_MODEL_VJP_SRC,
        )
    return _kernels


def make_model_core_metal(period_ref: float):
    """The fused model kernel: (x, t0_off, p_off, r, a, orb, u1, u2) ->
    flux deviation (n, m), orbit and photometry in one register-resident
    pass, with an analytic VJP.

    ``orb`` is the (n, 11) packed transit-anchored orbit constants from
    ``anchored.pack_orbit_constants``. A circular orbit is k = h = 0 with
    ci = b / a — exact, not approximate: the anchored form degenerates to
    the circular one, and the kernel skips the Kepler solve on such
    chains. Everything that maps sampler coordinates onto the constants
    stays in the MLX graph, so its Jacobian rides ordinary autodiff and
    only the per-point solve and photometry are fused here. period_ref is
    a runtime kernel input, never baked into source.
    """
    pref = float(period_ref)

    @mx.custom_function
    def core(x2d, t0_off, p_off, r, a, orb, u1, u2):
        n = t0_off.shape[0]
        m = x2d.shape[1]
        k = _get_model_kernels()["model_fwd"]
        return k(inputs=[x2d, t0_off, p_off, r, a, orb, u1, u2, pref,
                         int(m)],
                 output_shapes=[(n, m)], output_dtypes=[mx.float32],
                 grid=(m, n, 1), threadgroup=(256, 1, 1))[0]

    @core.vjp
    def core_vjp(primals, cotangent, output):
        x2d, t0_off, p_off, r, a, orb, u1, u2 = primals
        ct = cotangent if isinstance(cotangent, mx.array) else cotangent[0]
        n = t0_off.shape[0]
        m = x2d.shape[1]
        cols = (m + 31) // 32
        k = _get_model_kernels()["model_vjp"]
        part = k(inputs=[x2d, t0_off, p_off, r, a, orb, u1, u2, ct, pref,
                         int(m)],
                 output_shapes=[(n, NGRAD, cols)],
                 output_dtypes=[mx.float32], init_value=0.0,
                 grid=(m, n, 1), threadgroup=(256, 1, 1))[0]
        g = mx.sum(part, axis=2)                      # (n, NGRAD)
        # scatter the orbit slots back into (n, NORB); the three starter
        # columns (ecc, e0, mtra) are exactly zero by construction, and
        # so are the seven eccentric-only slots on circular chains.
        g_orb = mx.zeros((n, NORB), dtype=g.dtype)
        idx = mx.array([_ORB_COLS.index(c) for c in _GRAD_SLOTS[6:]])
        g_orb[:, idx] = g[:, 6:]
        return (mx.zeros_like(x2d), g[:, 0], g[:, 1], g[:, 2], g[:, 3],
                g_orb, g[:, 4], g[:, 5])

    return core


#: retired name — the eccentric kernel IS the model kernel now
make_ecc_core_metal = make_model_core_metal


def _get_tau_kernels():
    if "tau_fwd" not in _kernels:
        ins = ["taui", "perin", "ain", "bin", "rin", "u1in", "u2in", "cs",
               "xg", "wg", "expt", "mode", "ngl", "nsub", "npts"]
        _kernels["tau_fwd"] = mx.fast.metal_kernel(
            name="mp_tau_fwd", input_names=ins, output_names=["out"],
            header=_HEADER + _phot_header(),
            source=_tau_src(_TAU_FWD_SRC))
        _kernels["tau_vjp"] = mx.fast.metal_kernel(
            name="mp_tau_vjp", input_names=ins[:-1] + ["ct", "npts"],
            output_names=["gtau", "gpar"],
            header=_HEADER + _phot_header(),
            source=_tau_src(_TAU_VJP_SRC))
    return _kernels


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
    raise ValueError(f"parameter shape {p.shape} incompatible "
                     f"with a leading axis of {n}")


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


# ---------------------------------------------------------------------------
# flux_dev_from_tau: the tau-input entry point
# ---------------------------------------------------------------------------

_INT_MODES = {"none": _INT_NONE, "contact": _INT_CONTACT,
              "supersample": _INT_SUPER}


def _contact_taus(r, a, b, period, n):
    """The four contact times as offsets from mid-transit, per chain.

    ``contact_offsets`` works in orbital phase; the kernel wants tau, so the
    conversion is period / 2 pi. Done here rather than in the kernel because
    it is per-chain work, and detached because the contacts are *interior*
    split points of a continuous integrand: moving one adds +f(c) dc and
    -f(c) dc to the two intervals it separates, which cancel exactly. So
    the exact integral does not depend on where it is split, and neither
    parameter gradients nor the tau gradient need to chase the contacts.
    (The window *ends* are a different matter -- those are genuine Leibniz
    boundary terms, and the kernel carries them in ``dedge``.)
    """
    from .exposure import contact_offsets
    cs = contact_offsets(r, a, b)
    scale = period / (2.0 * math.pi)
    cols = [mx.reshape(mx.broadcast_to(c * scale, (n,)), (n, 1)) for c in cs]
    return mx.stop_gradient(mx.concatenate(cols, axis=1))


def _tau_graph(tau, period, a, b, r, u1, u2, exp_time, mode, n_gl, n_sub):
    """The MLX-graph equivalent of the kernel, used for fp64, the CPU
    stream, and any machine where the kernel probe fails.

    turin needs this for anvil's ``validate_precision`` and ``certify``, so
    it is a supported path and not a fallback of last resort: it computes
    the *same function*, including detaching the contacts, so that the two
    paths' gradients agree and not merely their values.

    ``tau`` is (n, m); the parameters arrive as (n,), one per chain.
    """
    from .exposure import contact_offsets, exposure_nodes
    from .orbit import separation_circular
    from .vjp import flux_dev_analytic

    pars = (period, a, b, r, u1, u2)

    def col(k):
        """Per-chain parameters with ``k`` trailing axes.

        How many they need is a property of the *rule*, not of the caller:
        each exposure rule appends a node axis to the times, so (n, 1) is
        right for the instantaneous path and (n, 1, 1) for the other two.
        This is invisible at n = 1, where (1, 1) broadcasts against
        anything -- which is exactly how it shipped wrong.
        """
        return [mx.reshape(p, (-1,) + (1,) * k) for p in pars]

    def inst(tt, k):
        per, av, bv, rv, u1v, u2v = col(k)
        return flux_dev_analytic(separation_circular(tt, per, bv, av),
                                 rv, u1v, u2v)

    if mode == _INT_NONE or exp_time == 0.0:
        return inst(tau, 1)
    half = 0.5 * exp_time
    if mode == _INT_SUPER:
        off = (np.linspace(-half, half, int(n_sub)) if n_sub > 1
               else np.zeros(1))
        nodes = tau[..., None] + mx.array(off, dtype=tau.dtype)
        return mx.mean(inst(nodes, 2), axis=-1)
    # contact: exposure_nodes owns the branchless five-interval split.
    # tau is measured from each point's own mid-transit, so t0 = 0.
    #
    # The period is detached HERE and only here: exposure_nodes converts
    # the contact *phases* to times with period / 2 pi, which would make
    # the split points move with the period while the kernel's are frozen.
    # Both are valid gradients of the same model -- a split point is
    # interior, so moving it adds +f(c) dc and -f(c) dc -- but they are not
    # the same function, and turin certifies in fp64 what it runs in fp32.
    # The period keeps its gradient where it belongs: in the phase.
    per1, a1, b1, r1 = col(1)[:4]
    cs = tuple(mx.stop_gradient(c) for c in contact_offsets(r1, a1, b1))
    T, W = exposure_nodes(tau, tau * 0.0, mx.stop_gradient(per1),
                          exp_time, cs, int(n_gl), dtype=tau.dtype)
    return mx.sum(inst(T, 2) * W, axis=-1)


def flux_dev_from_tau(tau: mx.array, period, a, b, r, u1, u2, *,
                      exp_time: float = 0.0, integration: str = "contact",
                      n_gl: int = 5, n_sub: int = 1) -> mx.array:
    """F - 1 from time-since-mid-transit, with the exposure integrated
    *inside* the kernel.

    ``tau`` is (n_chains, m) or (m,): each point's time relative to **its
    own** mid-transit. That is the difference from ``make_quad_transit_flux``,
    whose kernel derives the time from a linear ephemeris and therefore
    returns a zero gradient for it -- a caller sampling per-epoch mid-times
    (TTVs) cannot express its model that way. Here ``tau`` is an ordinary
    differentiable input: build it however the parameterisation requires and
    MLX chains the rest, which keeps this package out of the business of
    knowing about TTVs or anyone's parameter vector.

    The exposure rule runs per output point *in registers*, so the
    sub-exposure axis never becomes an MLX array. A caller that expands it
    instead pays ``n_sub`` times the memory in every forward intermediate
    and in every gradient grid.

    Args:
        tau: (n, m) or (m,) times since each point's own mid-transit, in the
            same units as ``period``. Expected to lie within half a period
            of it -- that is what "its own" means.
        period, a, b, r, u1, u2: scalars or (n,), broadcast per chain as
            ``flux_dev_metal`` does. ``a`` is a/R*, ``b`` the impact
            parameter, ``r`` = Rp/R*, ``u1``/``u2`` quadratic limb darkening.
        exp_time: exposure duration in tau's units; 0 means instantaneous
            (and forces ``integration="none"``).
        integration: ``"contact"`` -- Gauss-Legendre on the exposure window
            split at the four contact times, which MetalPlanet's own numbers
            make strictly better per evaluation than supersampling (25 evals
            for 8.9e-8 against 101 for 1.7e-5 on a 29-minute exposure);
            ``"supersample"`` -- ``n_sub`` uniform nodes, batman's rule;
            ``"none"`` -- instantaneous.
        n_gl: Gauss-Legendre nodes per sub-interval. The window always has
            five sub-intervals, so the cost is 5 * n_gl evaluations.
        n_sub: number of nodes for ``"supersample"``.

    Returns:
        (n, m) -- or (m,) for 1-D ``tau`` and no per-chain parameters -- of
        F - 1, exposure-averaged. Exactly 0 out of transit, as
        ``flux_dev_metal`` is.

    Gradients flow in ``tau`` and in all six parameters. fp64, the CPU
    stream, and machines without a usable Metal device take the MLX graph
    path, which computes the same function.
    """
    mode = _INT_MODES.get(integration)
    if mode is None:
        raise ValueError(f"integration must be one of {sorted(_INT_MODES)}; "
                         f"got {integration!r}")
    if int(n_gl) < 1 or int(n_sub) < 1:
        raise ValueError("n_gl and n_sub must be >= 1")
    if float(exp_time) < 0.0:
        raise ValueError("exp_time must be >= 0")
    if float(exp_time) == 0.0:
        mode = _INT_NONE
    return _flux_dev_from_tau_impl(tau, period, a, b, r, u1, u2,
                                   float(exp_time), mode, int(n_gl),
                                   int(n_sub))


def _flux_dev_from_tau_impl(tau, period, a, b, r, u1, u2, exp_time, mode,
                            n_gl, n_sub):
    if not isinstance(tau, mx.array):
        tau = mx.array(tau)
    if tau.ndim not in (1, 2):
        raise ValueError(f"tau must be (m,) or (n, m); got {tau.shape}")
    if tau.dtype == mx.float64 and _gpu_stream_active():
        # MLX has no float64 on Metal *at all* -- even a slice raises -- so
        # fp64 is not a dispatch choice here but a device one. Putting it on
        # the CPU stream ourselves is what makes "fp64 falls back to the
        # graph path" true rather than an exception the caller has to know
        # to pre-empt; TransitModel does the same (api.py: _stream).
        with mx.stream(mx.cpu):
            return _flux_dev_from_tau_impl(tau, period, a, b, r, u1, u2,
                                           exp_time, mode, n_gl, n_sub)
    squeeze = tau.ndim == 1
    tau2d = tau[None, :] if squeeze else tau
    params = (period, a, b, r, u1, u2)
    n_param = max((p.shape[0] if isinstance(p, mx.array) and p.ndim >= 1
                   else 1) for p in params)
    n = max(tau2d.shape[0], n_param)
    # a parameter carrying a different float width than tau would reach
    # the kernel as a mismatched input; tau sets the precision
    pc = [_canon_param(p, n, tau2d.dtype).astype(tau2d.dtype)
          for p in params]
    if tau2d.shape[0] != n:
        tau2d = mx.broadcast_to(tau2d, (n, tau2d.shape[1]))

    if (tau2d.dtype == mx.float32 and _gpu_stream_active()
            and metal_available()):
        out = _make_tau_core(exp_time, mode, n_gl, n_sub)(tau2d, *pc)
    else:
        # the graph path takes the same (n,) parameters as the kernel and
        # adds the trailing axes its own node grid needs
        out = _tau_graph(tau2d, *pc, exp_time, mode, n_gl, n_sub)
    return out[0] if squeeze and n == 1 else out


_tau_cores: dict = {}


def _make_tau_core(exp_time, mode, n_gl, n_sub):
    """One custom_function per (exposure, mode, order) so the VJP closes
    over them; cached because building it per call would defeat compile."""
    key = (float(exp_time), int(mode), int(n_gl), int(n_sub))
    if key in _tau_cores:
        return _tau_cores[key]
    from .exposure import gauss_legendre
    xg_np, wg_np = gauss_legendre(n_gl)

    def _static(dtype):
        return mx.array(xg_np, dtype=dtype), mx.array(wg_np, dtype=dtype)

    @mx.custom_function
    def core(tau2d, period, a, b, r, u1, u2):
        n, m = tau2d.shape
        xg, wg = _static(tau2d.dtype)
        cs = _contact_taus(r, a, b, period, n)
        k = _get_tau_kernels()["tau_fwd"]
        return k(inputs=[tau2d, period, a, b, r, u1, u2, cs, xg, wg,
                         exp_time, mode, n_gl, n_sub, int(m)],
                 output_shapes=[(n, m)], output_dtypes=[mx.float32],
                 grid=(m, n, 1), threadgroup=(256, 1, 1))[0]

    @core.vjp
    def core_vjp(primals, cotangent, output):
        tau2d, period, a, b, r, u1, u2 = primals
        ct = cotangent if isinstance(cotangent, mx.array) else cotangent[0]
        n, m = tau2d.shape
        cols = (m + 31) // 32
        xg, wg = _static(tau2d.dtype)
        cs = _contact_taus(r, a, b, period, n)
        k = _get_tau_kernels()["tau_vjp"]
        gtau, gpar = k(
            inputs=[tau2d, period, a, b, r, u1, u2, cs, xg, wg, exp_time,
                    mode, n_gl, n_sub, ct, int(m)],
            output_shapes=[(n, m), (n, NTAUPAR, cols)],
            output_dtypes=[mx.float32] * 2, init_value=0.0,
            grid=(m, n, 1), threadgroup=(256, 1, 1))
        g = mx.sum(gpar, axis=2)
        return (gtau, g[:, 0], g[:, 1], g[:, 2], g[:, 3], g[:, 4], g[:, 5])

    _tau_cores[key] = core
    return core
