"""Fused fp32 Metal kernels for the hybrid limb-darkening laws.

The closed forms are hybrid.py's (read its docstring first); this module
puts them in registers. Nothing here edits a quadratic kernel: the hybrid
device functions are their own source, and the tau kernels are *derived*
from metal.py's orbit-generic templates (_TAU_FWD_G, _TAU_VJP_G) by
asserted substitutions -- the quadrature, its exact tau derivative, the
contact split and the per-chain simd_sum reduction stay one copy, shared
with the quadratic law. Only the photometry call and the limb-darkening
gradient slots change:

    quadratic   u1, u2 per chain     mp_phot / mp_phot_d       gpar (per, theta..., r, u1, u2)
    hybrid      w[NW] per chain      mp_hyb_phot / _phot_d     gpar (per, theta..., r, w_0..w_{NW-1})
    basis       none                 mp_hyb_cols / _cols_bd    gpar (per, theta..., r, 0, 0)

Each law is a compile-time constant set -- its poles, the shape matrix S
mapping the generator columns [mu^0, mu^2, mu^4, P_eps_k...] onto the
shape columns, and the norms -- substituted into one template, so the
three laws are three instantiations of the same source.

No column needs an elliptic integral: per point the kernel evaluates the
lens geometry (two atan2, one sqrt, as the quadratic core), the even
recursion M_0 -> M_2 -> M_4 (rational), and per pole one atan2 or log and
two sqrt. The quadratic core's 10-iteration cel3 recursion is gone.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from . import metal as M
from .hybrid import combine_cols, flux_dev_hybrid, get_law, shape_cols

__all__ = ["flux_dev_from_tau_hybrid", "flux_dev_metal_hybrid"]

#: series half-width across the Q = 0 line in fp32 (hybrid.Y_SW[float32])
_YSW32 = 0.05


def _f(x: float) -> str:
    """A float as an fp32 Metal literal, to full fp64 repr precision."""
    return f"{float(x)!r}f"


# ---------------------------------------------------------------------------
# device functions (per law: constants substituted into one template)
# ---------------------------------------------------------------------------

_HYB_FN = r"""
#define HYB_K CONST_K
#define HYB_NW CONST_NW
#define HYB_NCOL CONST_NCOL
#define HYB_NGEN CONST_NGEN
constant float HYB_POLE[HYB_K] = { CONST_POLES };
constant float HYB_SMAT[HYB_NW * HYB_NGEN] = { CONST_SMAT };
constant float HYB_NORM[HYB_NCOL] = { CONST_NORM };

// Generator columns G = [mu^0, mu^2, mu^4, P_k...] in deviation form
// (minus the occulted flux; exactly 0 out of transit) and their z, r
// partials, from the boundary rule d occ/dr = r int I dpsi,
// d occ/dz = -r int I cos psi dpsi over the planet arc inside the star.
inline void mp_hyb_gen(float z, float r, thread float *G,
                       thread float *Gz, thread float *Gr) {
    for (int c = 0; c < HYB_NGEN; ++c) {
        G[c] = 0.0f; Gz[c] = 0.0f; Gr[c] = 0.0f;
    }
    z = fabs(z);
    if (z >= 1.0f + r) return;
    float r2 = r * r, z2 = z * z;
    bool comp = z <= 1.0f - r;
    float A = (r + 1.0f - z) * (1.0f - r + z);      // 1 - (z-r)^2
    float Bp = (1.0f - z - r) * (1.0f + z + r);     // 1 - (z+r)^2
    // signed Heron area^2, Kahan-sorted exactly as the quadratic core
    float sa = max(z, r), sc = min(z, r);
    float sb2 = min(sa, 1.0f);
    sa = max(sa, 1.0f);
    float sb = max(sb2, sc);
    sc = min(sb2, sc);
    float sqarea = (sa + (sb + sc)) * (sc - (sa - sb))
                 * (sc + (sa - sb)) * (sa + (sb - sc));
    float kite = 0.0f, kap0 = MP_PI, kap1 = 0.0f, sink = 0.0f, cosk = -1.0f;
    if (!comp) {
        kite = metal::precise::sqrt(max(sqarea, KITE_FLOOR));
        kap0 = metal::precise::atan2(kite, r2 + z2 - 1.0f);
        kap1 = metal::precise::atan2(kite, 1.0f + z2 - r2);
        float tzr = 2.0f * z * r;
        sink = kite / tzr;
        cosk = (r2 + z2 - 1.0f) / tzr;
    }
    float alpha = 1.0f - r2 - z2, beta = 2.0f * z * r;

    // even columns: s0, s2 and the even recursion M0 -> M2 -> M4
    float s0d, s2d, M0, M2;
    if (comp) {
        s0d = -MP_PI * r2;
        s2d = MP_TWO_PI * r2 * (r2 + 2.0f * z2 - 1.0f);
        M0 = MP_PI;
        M2 = MP_PI * alpha;
    } else {
        s0d = -(kap1 + r2 * kap0 - 0.5f * kite);
        float eta2 = r2 * (r2 + 2.0f * z2);
        s2d = 2.0f * s0d + 2.0f * (kap1 + eta2 * kap0
                                   - 0.25f * kite * (1.0f + 5.0f * r2 + z2));
        M0 = kap0;
        M2 = kap0 * alpha + kite;
    }
    float M4 = (6.0f * alpha * M2 + 2.0f * sqarea * M0) / 4.0f;
    float s4 = -(2.0f * r2 * M4 - (2.0f / 3.0f) * (alpha * M4 + sqarea * M2));
    G[0] = s0d;
    G[1] = 0.5f * s0d + 0.25f * s2d;
    G[2] = s0d / 3.0f + s2d / 6.0f + s4 / 6.0f;
    float i0 = 2.0f * kap0, i1 = 2.0f * sink;
    float i2 = kap0 + sink * cosk;
    float i3 = 2.0f * (sink - sink * sink * sink / 3.0f);
    float a2 = alpha * alpha, ab = alpha * beta, b2 = beta * beta;
    Gr[0] = -r * i0;                          Gz[0] = r * i1;
    Gr[1] = -r * (alpha * i0 + beta * i1);    Gz[1] = r * (alpha * i1 + beta * i2);
    Gr[2] = -r * (a2 * i0 + 2.0f * ab * i1 + b2 * i2);
    Gz[2] = r * (a2 * i1 + 2.0f * ab * i2 + b2 * i3);

    // pole columns (hybrid.py's _pole_terms)
    for (int q = 0; q < HYB_K; ++q) {
        float e = HYB_POLE[q];
        float p = 1.0f + e;
        float a_ = p - z2 - r2, K = p + r2 - z2;
        float Q = (e + A) * (e + Bp);
        float occ, dz, dr;
        if (comp) {
            float sq = metal::precise::sqrt(Q);
            occ = MP_TWO_PI * r2 / (sq * (K + sq));
            float Q32 = Q * sq;
            dz = 4.0f * MP_PI * z * r2 / Q32;
            dr = MP_TWO_PI * r * a_ / Q32;
        } else {
            float nBp = -Bp, apb = e + A;
            float U = apb * nBp;
            float ratio = (e + Bp) / nBp;           // V = ratio * kite^2
            float y = ratio * kite * kite / U;
            float J, J2, J2c;
            if (y > YSW) {
                J = 4.0f / metal::precise::sqrt(Q)
                    * metal::precise::atan2(kite * metal::precise::sqrt(ratio),
                                            metal::precise::sqrt(U));
            } else if (y < -YSW) {
                J = 4.0f / metal::precise::sqrt(-Q)
                    * metal::precise::log(
                        (metal::precise::sqrt(U)
                         + kite * metal::precise::sqrt(-ratio))
                        / (2.0f * metal::precise::sqrt(z * r * e)));
            }
            if (fabs(y) > YSW) {
                J2 = (a_ * J - 2.0f * kite / e) / Q;
                J2c = (a_ * kite / (z * r * e) - beta * J) / Q;
            } else {
                float T = kite / nBp;               // tan(kap0 / 2)
                float y2 = y * y;
                float Gs = 1.0f - y / 3.0f + y2 / 5.0f - y * y2 / 7.0f
                         + y2 * y2 / 9.0f - y * y2 * y2 / 11.0f;
                float Gp = -1.0f / 3.0f + 2.0f * y / 5.0f - 3.0f * y2 / 7.0f
                         + 4.0f * y * y2 / 9.0f - 5.0f * y2 * y2 / 11.0f;
                float T2 = T * T, ia = 1.0f / apb;
                J = 4.0f * T * Gs * ia;
                J2 = 4.0f * T * ia * ia * (Gs - 2.0f * beta * T2 * Gp * ia);
                J2c = 4.0f * T * ia * ia * (Gs + 2.0f * a_ * T2 * Gp * ia);
            }
            occ = kap1 / (p * e) + (K * J - 2.0f * kap0) / (4.0f * p);
            dz = -r * J2c;
            dr = r * J2;
        }
        G[3 + q] = -occ;
        Gz[3 + q] = -dz;
        Gr[3 + q] = -dr;
    }
}

// Shape columns B = [E0, T_1..T_NW] = [G_0, S G] and partials.
inline void mp_hyb_cols_d(float z, float r, thread float *B,
                          thread float *Bz, thread float *Br) {
    float G[HYB_NGEN], Gz[HYB_NGEN], Gr[HYB_NGEN];
    mp_hyb_gen(z, r, G, Gz, Gr);
    B[0] = G[0]; Bz[0] = Gz[0]; Br[0] = Gr[0];
    for (int j = 0; j < HYB_NW; ++j) {
        float b = 0.0f, bz = 0.0f, br = 0.0f;
        for (int c = 0; c < HYB_NGEN; ++c) {
            float s = HYB_SMAT[j * HYB_NGEN + c];
            b += s * G[c]; bz += s * Gz[c]; br += s * Gr[c];
        }
        B[j + 1] = b; Bz[j + 1] = bz; Br[j + 1] = br;
    }
}

// forward only: the partials are dead and inlining drops them
inline void mp_hyb_cols(float z, float r, thread float *B) {
    float Bz[HYB_NCOL], Br[HYB_NCOL];
    mp_hyb_cols_d(z, r, B, Bz, Br);
}

// F - 1 = (E0 - sum w_j T_j) * inv_norm, inv_norm = 1/(pi - sum w_j N_j)
inline float mp_hyb_phot(float z, float r, thread const float *wv,
                         float inv_norm) {
    float B[HYB_NCOL];
    mp_hyb_cols(z, r, B);
    float num = B[0];
    for (int j = 0; j < HYB_NW; ++j) num -= wv[j] * B[j + 1];
    return num * inv_norm;
}

inline float mp_hyb_phot_d(float z, float r, thread const float *wv,
                           float inv_norm, thread float *dFdz,
                           thread float *dFdr, thread float *dFdw) {
    float B[HYB_NCOL], Bz[HYB_NCOL], Br[HYB_NCOL];
    mp_hyb_cols_d(z, r, B, Bz, Br);
    float num = B[0], nz = Bz[0], nr = Br[0];
    for (int j = 0; j < HYB_NW; ++j) {
        num -= wv[j] * B[j + 1];
        nz -= wv[j] * Bz[j + 1];
        nr -= wv[j] * Br[j + 1];
    }
    float F = num * inv_norm;
    *dFdz = nz * inv_norm;
    *dFdr = nr * inv_norm;
    for (int j = 0; j < HYB_NW; ++j)
        dFdw[j] = (-B[j + 1] + F * HYB_NORM[j + 1]) * inv_norm;
    return F;
}

// ct . B and its z, r partials for a cotangent on the basis columns; the
// signature of the quadratic mp_phot_bd, so the derived VJP body is shared
// (the two limb-darkening slots stay zero).
inline float mp_hyb_cols_bd(float z, float r, thread const float *wct,
                            thread float *dFdz, thread float *dFdr,
                            thread float *dFdu1, thread float *dFdu2) {
    *dFdu1 = 0.0f; *dFdu2 = 0.0f;
    float B[HYB_NCOL], Bz[HYB_NCOL], Br[HYB_NCOL];
    mp_hyb_cols_d(z, r, B, Bz, Br);
    float f = 0.0f, fz = 0.0f, fr = 0.0f;
    for (int c = 0; c < HYB_NCOL; ++c) {
        f += wct[c] * B[c]; fz += wct[c] * Bz[c]; fr += wct[c] * Br[c];
    }
    *dFdz = fz;
    *dFdr = fr;
    return f;
}
"""

#: the markers _HYB_FN must not contain once instantiated
_FN_MARKERS = ("CONST_", "YSW", "KITE_FLOOR")


def _hyb_header(law) -> str:
    law = get_law(law)
    S = law.generator_matrix()
    src = (_HYB_FN.replace("CONST_K", str(len(law.eps)))
                  .replace("CONST_NW", str(law.n_w))
                  .replace("CONST_NCOL", str(law.n_col))
                  .replace("CONST_NGEN", str(S.shape[1]))
                  .replace("CONST_POLES", ", ".join(_f(e) for e in law.eps))
                  .replace("CONST_SMAT", ", ".join(_f(v) for v in S.ravel()))
                  .replace("CONST_NORM", ", ".join(_f(v) for v in law.norms()))
                  .replace("YSW", _f(_YSW32))
                  .replace("KITE_FLOOR", M._KITE_FLOOR))
    for marker in _FN_MARKERS:
        assert marker not in src, f"unexpanded {marker} in hybrid header"
    return src


# ---------------------------------------------------------------------------
# tau kernels, derived from metal.py's generic templates
# ---------------------------------------------------------------------------

_U_LOADS = "    float u1  = u1in[y];\n    float u2  = u2in[y];\n"
_W_LOADS = """    float wv[HYB_NW];
    float inv_norm = MP_PI;
    for (int q = 0; q < HYB_NW; ++q) {
        wv[q] = win[y * (uint)HYB_NW + (uint)q];
        inv_norm -= wv[q] * HYB_NORM[q + 1];
    }
    inv_norm = 1.0f / inv_norm;
"""


def _fwd_body() -> str:
    """Scalar forward: the generic body with the hybrid photometry."""
    s = M._swap(M._TAU_FWD_G, _U_LOADS, _W_LOADS)
    return M._swap(s, "mp_phot(z, r, u1, u2)", "mp_hyb_phot(z, r, wv, inv_norm)")


def _vjp_body(nslot: int) -> str:
    """Scalar VJP: the generic body with NW weight slots for (u1, u2)."""
    s = M._TAU_VJP_G
    for old, new in [
        (_U_LOADS, _W_LOADS),
        ("    float z, dFdz, dFdr, dFdu1, dFdu2, dzdphi;\n",
         "    float z, dFdz, dFdr, dzdphi;\n    float dFdw[HYB_NW];\n"),
        ("    float p_per = 0.0f, p_r = 0.0f, p_u1 = 0.0f, p_u2 = 0.0f;\n",
         "    float p_per = 0.0f, p_r = 0.0f;\n    float p_w[HYB_NW];\n"
         "    for (int q = 0; q < HYB_NW; ++q) p_w[q] = 0.0f;\n"),
        ("mp_phot_d(z, r, u1, u2, &dFdz, &dFdr, &dFdu1, &dFdu2)",
         "mp_hyb_phot_d(z, r, wv, inv_norm, &dFdz, &dFdr, dFdw)"),
        ("            p_u1  += ctv * scale * dFdu1;\n"
         "            p_u2  += ctv * scale * dFdu2;\n",
         "            for (int q = 0; q < HYB_NW; ++q)\n"
         "                p_w[q] += ctv * scale * dFdw[q];\n"),
        ("        float a_per = 0.0f, a_r = 0.0f, a_u1 = 0.0f, a_u2 = 0.0f;\n",
         "        float a_per = 0.0f, a_r = 0.0f;\n        float a_w[HYB_NW];\n"
         "        for (int q = 0; q < HYB_NW; ++q) a_w[q] = 0.0f;\n"),
        ("                    a_u1  += w * dFdu1;\n"
         "                    a_u2  += w * dFdu2;\n",
         "                    for (int q = 0; q < HYB_NW; ++q)\n"
         "                        a_w[q] += w * dFdw[q];\n"),
        ("            p_u1 = ctv * a_u1 * inv;\n"
         "            p_u2 = ctv * a_u2 * inv;\n",
         "            for (int q = 0; q < HYB_NW; ++q)\n"
         "                p_w[q] = ctv * a_w[q] * inv;\n"),
        ("    p_u1  = metal::simd_sum(p_u1);\n"
         "    p_u2  = metal::simd_sum(p_u2);\n",
         "    for (int q = 0; q < HYB_NW; ++q)\n"
         "        p_w[q] = metal::simd_sum(p_w[q]);\n"),
        ("        gpar[ob + (2u + NTH_U) * ngrp] = p_u1;\n"
         "        gpar[ob + (3u + NTH_U) * ngrp] = p_u2;\n",
         "        for (int q = 0; q < HYB_NW; ++q)\n"
         "            gpar[ob + (2u + NTH_U + (uint)q) * ngrp] = p_w[q];\n"),
        ("NSLOT_C", f"{nslot}u"),
    ]:
        s = M._swap(s, old, new)
    assert "u1" not in s and "u2" not in s, "a (u1, u2) leftover in the VJP"
    return s


_FWD_B_BODY = M._swap(M._TAU_HEAD_G, _U_LOADS, "") + """
    float z;
    bool front;
ORB_DECL
    float acc[HYB_NCOL], Bc[HYB_NCOL];
    for (int q = 0; q < HYB_NCOL; ++q) acc[q] = 0.0f;
    if (mode == MODE_NONE) {
        ORB_Z(tau0)
        if (front) {
            mp_hyb_cols(z, r, Bc);
            for (int q = 0; q < HYB_NCOL; ++q) acc[q] = Bc[q];
        }
    } else if (mode == MODE_SUPER) {
        for (int j = 0; j < nsub; ++j) {
            float frac = (nsub == 1) ? 0.5f
                                     : (float)j / (float)(nsub - 1);
            float tt = t1_of(tau0, hw_exp) + 2.0f * hw_exp * frac;
            ORB_Z(tt)
            if (front) {
                mp_hyb_cols(z, r, Bc);
                for (int q = 0; q < HYB_NCOL; ++q) acc[q] += Bc[q];
            }
        }
        for (int q = 0; q < HYB_NCOL; ++q) acc[q] = acc[q] / (float)nsub;
    } else {
""" + M._TAU_EDGES + """
        float S = 0.0f;
        for (int iv = 0; iv < 5; ++iv) {
            float lo = edge[iv], hi = edge[iv + 1];
            float mid = 0.5f * (lo + hi), hw = 0.5f * (hi - lo);
            for (int j = 0; j < ngl; ++j) {
                float w = hw * wg[j];
                float tt = mid + hw * xg[j];
                ORB_Z(tt)
                if (front) {
                    mp_hyb_cols(z, r, Bc);
                    for (int q = 0; q < HYB_NCOL; ++q) acc[q] += w * Bc[q];
                }
                S += w;
            }
        }
        float inv = (S > 0.0f) ? (1.0f / S) : 0.0f;
        for (int q = 0; q < HYB_NCOL; ++q) acc[q] = acc[q] * inv;
    }
    for (int q = 0; q < HYB_NCOL; ++q)
        out[(uint)HYB_NCOL * i + (uint)q] = acc[q];
"""


def _vjp_b_body() -> str:
    """Basis VJP: the generic body with the cotangent contracted into the
    core and ctv = 1, exactly as metal._tau_vjp_b_g does for quadratic."""
    s = M._swap(M._TAU_VJP_G, _U_LOADS, "")
    s = M._swap(s, "    float ctv = ct[i];\n",
                "    float wct[HYB_NCOL];\n"
                "    for (int q = 0; q < HYB_NCOL; ++q)\n"
                "        wct[q] = ct[(uint)HYB_NCOL * i + (uint)q];\n"
                "    float ctv = 1.0f;\n")
    return M._swap(s, "mp_phot_d(z, r, u1, u2, ", "mp_hyb_cols_bd(z, r, wct, ")


def _nth(orbit: str) -> int:
    return len(M._TAU_ORBITS[orbit]["theta"])


def _nslot(law, orbit: str, basis: bool) -> int:
    """gpar slots: per, theta..., r, then n_w weights (or the two zero
    slots of the derived basis VJP)."""
    return 2 + _nth(orbit) + (2 if basis else get_law(law).n_w)


def _get_kernels(law, orbit: str, basis: bool):
    law = get_law(law)
    key = ("hyb", law.name, orbit, basis)
    if key not in M._kernels:
        hdr = M._HEADER + M._phot_header()
        if orbit == "ecc":
            hdr += M._ecc_header()
        hdr += _hyb_header(law)
        ins = (["taui", "perin", "ain"] + M._TAU_ORBITS[orbit]["inputs"]
               + ["rin"] + ([] if basis else ["win"])
               + ["cs", "xg", "wg", "expt", "mode", "ngl", "nsub", "npts"])
        if basis:
            fwd, vjp = _FWD_B_BODY, _vjp_b_body()
        else:
            fwd = _fwd_body()
            vjp = _vjp_body(_nslot(law, orbit, False))
        tag = f"{law.name}_{orbit}{'_b' if basis else ''}"
        M._kernels[key] = (
            mx.fast.metal_kernel(
                name=f"mp_htau_fwd_{tag}", input_names=ins,
                output_names=["out"], header=hdr,
                source=M._tau_g_src(fwd, orbit)),
            mx.fast.metal_kernel(
                name=f"mp_htau_vjp_{tag}", input_names=ins[:-1] + ["ct", "npts"],
                output_names=["gtau", "gpar"], header=hdr,
                source=M._tau_g_src(vjp, orbit)))
    return M._kernels[key]


def _make_core(law, exp_time, mode, n_gl, n_sub, orbit, basis):
    """custom_function over the hybrid tau kernels. Primals:
    (tau2d, period, a, shape, r, [w2d,] cs) -- as metal._make_tau_core_g,
    with the (n, n_w) weights in place of u1, u2."""
    law = get_law(law)
    key = ("hyb", law.name, orbit, bool(basis), float(exp_time), int(mode),
           int(n_gl), int(n_sub))
    if key in M._tau_cores:
        return M._tau_cores[key]
    from .exposure import gauss_legendre
    xg_np, wg_np = gauss_legendre(n_gl)
    nth = _nth(orbit)
    nslot = _nslot(law, orbit, basis)
    ncol = law.n_col

    def _static(dtype):
        return mx.array(xg_np, dtype=dtype), mx.array(wg_np, dtype=dtype)

    def _fwd(*primals):
        tau2d = primals[0]
        n, m = tau2d.shape
        xg, wg = _static(tau2d.dtype)
        k = _get_kernels(law, orbit, basis)[0]
        return k(inputs=[*primals, xg, wg, exp_time, mode, n_gl, n_sub,
                         int(m)],
                 output_shapes=[(n, m, ncol) if basis else (n, m)],
                 output_dtypes=[mx.float32],
                 grid=(m, n, 1), threadgroup=(256, 1, 1))[0]

    def _vjp(primals, cotangent, output):
        ct = cotangent if isinstance(cotangent, mx.array) else cotangent[0]
        tau2d = primals[0]
        n, m = tau2d.shape
        cols = (m + 31) // 32
        xg, wg = _static(tau2d.dtype)
        k = _get_kernels(law, orbit, basis)[1]
        gtau, gpar = k(
            inputs=[*primals, xg, wg, exp_time, mode, n_gl, n_sub, ct,
                    int(m)],
            output_shapes=[(n, m), (n, nslot, cols)],
            output_dtypes=[mx.float32] * 2, init_value=0.0,
            grid=(m, n, 1), threadgroup=(256, 1, 1))
        g = mx.sum(gpar, axis=2)
        g_per, g_a = g[:, 0], g[:, 1]
        if orbit == "circ":
            g_shape = g[:, 2]
        else:
            g_shape = mx.zeros((n, M.NORB), dtype=g.dtype)
            g_shape[:, mx.array(M._ECC_TH_COLS)] = g[:, 2:1 + nth]
        g_r = g[:, 1 + nth]
        ws = () if basis else (g[:, 2 + nth:2 + nth + law.n_w],)
        return (gtau, g_per, g_a, g_shape, g_r, *ws,
                mx.zeros_like(primals[-1]))

    if basis:
        @mx.custom_function
        def core(tau2d, period, a, shape, r, cs):
            return _fwd(tau2d, period, a, shape, r, cs)
    else:
        @mx.custom_function
        def core(tau2d, period, a, shape, r, w2d, cs):
            return _fwd(tau2d, period, a, shape, r, w2d, cs)
    core.vjp(_vjp)
    M._tau_cores[key] = core
    return core


# ---------------------------------------------------------------------------
# the graph path (fp64, CPU stream, no Metal): the same function
# ---------------------------------------------------------------------------

def _tau_graph(tau, period, a, b, r, w2d, exp_time, mode, n_gl, n_sub,
               law, basis=False, k=None, h=None):
    """metal._tau_graph for a hybrid law: the MLX-graph equivalent of the
    kernels, contacts detached and the period detached in the node map
    exactly as there, so the two paths compute the same function."""
    from .anchored import separation_anchored
    from .exposure import (contact_offsets, contact_offsets_anchored,
                           exposure_nodes)
    from .orbit import separation_circular

    law = get_law(law)
    ecc = k is not None
    pars = (period, a, b, r) + ((k, h) if ecc else ())

    def col(nk):
        return [mx.reshape(p, (-1,) + (1,) * nk) for p in pars]

    def inst(tt, nk):
        cols = col(nk)
        per, av, bv, rv = cols[:4]
        if ecc:
            kv, hv = cols[-2:]
            ci = M._ecc_shape(av, bv, kv, hv)[2]
            z, front = separation_anchored((2.0 * math.pi) * tt / per,
                                           kv, hv, av, ci)
            B = shape_cols(z, rv, law)
            return mx.where(front[..., None], B, 0.0)
        return shape_cols(separation_circular(tt, per, bv, av), rv, law)

    if mode == M._INT_NONE or exp_time == 0.0:
        B = inst(tau, 1)
    else:
        half = 0.5 * exp_time
        if mode == M._INT_SUPER:
            off = (np.linspace(-half, half, int(n_sub)) if n_sub > 1
                   else np.zeros(1))
            nodes = tau[..., None] + mx.array(off, dtype=tau.dtype)
            B = mx.mean(inst(nodes, 2), axis=-2)
        else:
            per1, a1, b1, r1 = col(1)[:4]
            if ecc:
                k1, h1 = col(1)[-2:]
                ci1 = M._ecc_shape(a1, b1, k1, h1)[2]
                cs = contact_offsets_anchored(r1, a1, b1, k1, h1, ci1)
            else:
                cs = contact_offsets(r1, a1, b1)
            cs = tuple(mx.stop_gradient(c) for c in cs)
            T, W = exposure_nodes(tau, tau * 0.0, mx.stop_gradient(per1),
                                  exp_time, cs, int(n_gl), dtype=tau.dtype)
            B = mx.sum(inst(T, 2) * W[..., None], axis=-2)
    if basis:
        return B
    # the one off-kernel expression (hybrid.combine_cols): w2d (n, n_w)
    # rows broadcast against B's (n, m) as flux_dev_hybrid's batched form
    return combine_cols(B, w2d, law)


# ---------------------------------------------------------------------------
# entry points (called from metal.flux_dev_from_tau / flux_dev_metal)
# ---------------------------------------------------------------------------

def _canon_w(u, n, n_w, dtype, name):
    """Weights -> (n, n_w): a host sequence or (n_w,) shared by every
    chain, or (n, n_w) / (1, n_w) per chain."""
    # a host sequence goes straight to the target dtype: mx.array() of a
    # float64 numpy array without a dtype rounds it through fp32 first
    w = u if isinstance(u, mx.array) else mx.array(
        np.asarray(u, dtype=np.float64), dtype=dtype)
    if w.dtype != dtype:
        # before any GPU op on it: an fp64 array cannot even be broadcast
        # there (0.10.5 cast last, after the broadcast, and so raised)
        with mx.stream(mx.cpu):
            w = w.astype(dtype)
    if w.ndim == 1:
        if w.shape[0] != n_w:
            raise ValueError(f"{name} takes {n_w} weights; got {w.shape[0]}")
        w = mx.broadcast_to(w[None, :], (n, n_w))
    elif w.ndim == 2 and w.shape[1] == n_w and w.shape[0] in (1, n):
        w = mx.broadcast_to(w, (n, n_w))
    else:
        raise ValueError(f"{name} weights must be ({n_w},) or (n, {n_w}); "
                         f"got shape {w.shape}")
    return w


@M.fp64_on_cpu
def flux_dev_from_tau_hybrid(tau, period, a, b, r, law, u, exp_time, mode,
                             n_gl, n_sub, basis=False, k=None, h=None):
    """``metal.flux_dev_from_tau`` for a hybrid law (that function validates
    the keywords and delegates here)."""
    law = get_law(law)
    if tau.ndim not in (1, 2):
        raise ValueError(f"tau must be (m,) or (n, m); got {tau.shape}")
    squeeze = tau.ndim == 1
    tau2d = tau[None, :] if squeeze else tau
    ecc = k is not None
    params = (period, a, b, r) + ((k, h) if ecc else ())
    n_param = max((p.shape[0] if isinstance(p, mx.array) and p.ndim >= 1
                   else 1) for p in params)
    if not basis and isinstance(u, mx.array) and u.ndim == 2:
        n_param = max(n_param, u.shape[0])
    n = max(tau2d.shape[0], n_param)
    pc = [M._canon_param(p, n, tau2d.dtype).astype(tau2d.dtype)
          for p in params]
    if tau2d.shape[0] != n:
        tau2d = mx.broadcast_to(tau2d, (n, tau2d.shape[1]))
    per_c, a_c, b_c, r_c = pc[:4]
    kh = pc[-2:] if ecc else [None, None]
    w2d = (None if basis else
           _canon_w(u, n, law.n_w, tau2d.dtype, law.name))

    if (tau2d.dtype == mx.float32 and M._gpu_stream_active()
            and M.metal_available()):
        if ecc:
            shape, cs = M._ecc_kernel_inputs(per_c, a_c, b_c, r_c, *kh, n)
        else:
            shape, cs = b_c, M._contact_taus(r_c, a_c, b_c, per_c, n)
        core = _make_core(law, exp_time, mode, n_gl, n_sub,
                          "ecc" if ecc else "circ", basis)
        ws = () if basis else (w2d,)
        out = core(tau2d, per_c, a_c, shape, r_c, *ws, cs)
    else:
        out = _tau_graph(tau2d, per_c, a_c, b_c, r_c, w2d, exp_time, mode,
                         n_gl, n_sub, law, basis=basis, k=kh[0], h=kh[1])
    return out[0] if squeeze and n == 1 else out


# z-input kernels: one point per thread, as metal.flux_dev_metal

_ZH_HEAD = """
    uint x = thread_position_in_grid.x;
    uint y = thread_position_in_grid.y;
    if (x >= (uint)npts) return;
    uint i = y * (uint)npts + x;
    float z = fabs(zin[i]);
    float r = rin[y];
"""

_ZH_FWD = _ZH_HEAD + _W_LOADS + """
    out[i] = mp_hyb_phot(z, r, wv, inv_norm);
"""

_ZH_VJP = _ZH_HEAD + _W_LOADS + """
    float dz, dr, dw[HYB_NW];
    mp_hyb_phot_d(z, r, wv, inv_norm, &dz, &dr, dw);
    float ctv = ct[i];
    gz[i] = ctv * dz;
    gr[i] = ctv * dr;
    for (int q = 0; q < HYB_NW; ++q)
        gw[(uint)HYB_NW * i + (uint)q] = ctv * dw[q];
"""

_ZH_FWD_B = _ZH_HEAD + """
    float B[HYB_NCOL];
    mp_hyb_cols(z, r, B);
    for (int q = 0; q < HYB_NCOL; ++q)
        out[(uint)HYB_NCOL * i + (uint)q] = B[q];
"""

_ZH_VJP_B = _ZH_HEAD + """
    float wct[HYB_NCOL];
    for (int q = 0; q < HYB_NCOL; ++q)
        wct[q] = ct[(uint)HYB_NCOL * i + (uint)q];
    float dz, dr, d1, d2;
    mp_hyb_cols_bd(z, r, wct, &dz, &dr, &d1, &d2);
    gz[i] = dz;
    gr[i] = dr;
"""


def _get_z_kernels(law, basis: bool):
    law = get_law(law)
    key = ("hybz", law.name, basis)
    if key not in M._kernels:
        hdr = M._HEADER + _hyb_header(law)
        tag = f"{law.name}{'_b' if basis else ''}"
        if basis:
            fwd = mx.fast.metal_kernel(
                name=f"mp_hz_fwd_{tag}", input_names=["zin", "rin", "npts"],
                output_names=["out"], header=hdr, source=M._subst(_ZH_FWD_B))
            vjp = mx.fast.metal_kernel(
                name=f"mp_hz_vjp_{tag}",
                input_names=["zin", "rin", "ct", "npts"],
                output_names=["gz", "gr"], header=hdr,
                source=M._subst(_ZH_VJP_B))
        else:
            fwd = mx.fast.metal_kernel(
                name=f"mp_hz_fwd_{tag}",
                input_names=["zin", "rin", "win", "npts"],
                output_names=["out"], header=hdr, source=M._subst(_ZH_FWD))
            vjp = mx.fast.metal_kernel(
                name=f"mp_hz_vjp_{tag}",
                input_names=["zin", "rin", "win", "ct", "npts"],
                output_names=["gz", "gr", "gw"], header=hdr,
                source=M._subst(_ZH_VJP))
        M._kernels[key] = (fwd, vjp)
    return M._kernels[key]


def _z_core(law, basis: bool):
    law = get_law(law)
    key = ("hybz", law.name, bool(basis))
    if key in M._tau_cores:
        return M._tau_cores[key]
    ncol, n_w = law.n_col, law.n_w

    if basis:
        @mx.custom_function
        def core(z2d, r):
            n, m = z2d.shape
            k = _get_z_kernels(law, True)[0]
            return k(inputs=[z2d, r, int(m)], output_shapes=[(n, m, ncol)],
                     output_dtypes=[mx.float32], grid=(m, n, 1),
                     threadgroup=(256, 1, 1))[0]

        @core.vjp
        def _vjp(primals, cotangent, output):
            z2d, r = primals
            ct = cotangent if isinstance(cotangent, mx.array) else cotangent[0]
            n, m = z2d.shape
            gz, gr = _get_z_kernels(law, True)[1](
                inputs=[z2d, r, ct, int(m)], output_shapes=[(n, m)] * 2,
                output_dtypes=[mx.float32] * 2, grid=(m, n, 1),
                threadgroup=(256, 1, 1))
            return gz, mx.sum(gr, axis=1)
    else:
        @mx.custom_function
        def core(z2d, r, w2d):
            n, m = z2d.shape
            k = _get_z_kernels(law, False)[0]
            return k(inputs=[z2d, r, w2d, int(m)], output_shapes=[(n, m)],
                     output_dtypes=[mx.float32], grid=(m, n, 1),
                     threadgroup=(256, 1, 1))[0]

        @core.vjp
        def _vjp(primals, cotangent, output):
            z2d, r, w2d = primals
            ct = cotangent if isinstance(cotangent, mx.array) else cotangent[0]
            n, m = z2d.shape
            gz, gr, gw = _get_z_kernels(law, False)[1](
                inputs=[z2d, r, w2d, ct, int(m)],
                output_shapes=[(n, m), (n, m), (n, m, n_w)],
                output_dtypes=[mx.float32] * 3, grid=(m, n, 1),
                threadgroup=(256, 1, 1))
            return gz, mx.sum(gr, axis=1), mx.sum(gw, axis=1)
    M._tau_cores[key] = core
    return core


@M.fp64_on_cpu
def flux_dev_metal_hybrid(z, r, law, u, basis=False):
    """``metal.flux_dev_metal`` for a hybrid law: (n, m) or (m,) points,
    r per chain; F - 1, or with ``basis`` the z.shape + (1 + n_w,)
    shape-basis columns. z's dtype is the computation's; fp64 takes
    hybrid.py's graph on the CPU stream (metal.fp64_on_cpu)."""
    law = get_law(law)
    if z.ndim not in (1, 2):
        raise ValueError(f"z must be (m,) or (n, m); got {z.shape}")
    squeeze = z.ndim == 1
    z2d = z[None, :] if squeeze else z
    n_param = r.shape[0] if isinstance(r, mx.array) and r.ndim >= 1 else 1
    if not basis and isinstance(u, mx.array) and u.ndim == 2:
        n_param = max(n_param, u.shape[0])
    n = max(z2d.shape[0], n_param)
    rc = M._canon_param(r, n, z.dtype).astype(z.dtype)
    if z2d.shape[0] != n:
        z2d = mx.broadcast_to(z2d, (n, z2d.shape[1]))
    w2d = None if basis else _canon_w(u, n, law.n_w, z.dtype, law.name)
    if z.dtype == mx.float32 and M._gpu_stream_active() and M.metal_available():
        core = _z_core(law, basis)
        out = core(z2d, rc) if basis else core(z2d, rc, w2d)
    elif basis:
        out = shape_cols(z2d, rc[:, None], law)
    else:
        # hybrid.py's graph itself, so every route to this function --
        # fp64, the CPU stream, a Metal-less machine -- is one expression,
        # bitwise (0.10.3's dot-product contraction was 2.4e-7 apart)
        out = flux_dev_hybrid(z2d, rc[:, None], w2d, law)
    return out[0] if squeeze and n == 1 else out
