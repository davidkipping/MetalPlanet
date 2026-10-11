"""Fused fp32 Metal kernels for oblate planets (hybrid laws).

The mathematics is ``oblate.py``'s (read its docstring first); this module
transcribes it per point, in registers, and adds the orbit and the exposure
rule as ``metal_hybrid`` does for spherical planets. Nothing here edits an
existing kernel: the oblate device functions and tau bodies are their own
source, and the shared fragments (the hybrid law constants, the spherical
hybrid columns used below f_sw, the exposure edges, the eccentric Kepler
solve) are included, not changed.

Per point the work splits in two, exactly as the reference graph's
stop-gradients split it:

* **the detached solve** (plain float, ``ob_solve``): the regime, the limb
  crossings and arcs, each level's four quartic roots -- seeded by the same
  balanced, Wilkinson-shifted complex QR on the companion matrix as
  ``oblate._qr_seeds`` (Metal has no eigensolver), then Aberth and Newton --
  and the near-double pair selection;
* **the live evaluation** (``ob_eval<T>``): coefficients, Fourier sums,
  one live Newton step per root, residues, continued logarithms, the pair
  formula with its Bairstow step, and the corner term.

``ob_eval`` is a template. Instantiated on ``float`` it is the forward
kernel; on ``dv`` -- a value with four tangents, d/d(x0, y0, A, B) -- it is
the VJP's per-point Jacobian, the derivative of exactly the function the
forward computes, with the same frozen pieces as the graph's autodiff. The
orbit, the principal frame and the axes then chain by hand.

Below f_sw = 1e-5 a chain takes the spherical hybrid columns (the reference
graph's switch in fp32), with d/df = 0 there: the flattening's own effect,
~4e-3 f of a column, is under 4e-8 below the switch -- fp32 noise -- so a
sampler loses nothing it could resolve.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from . import metal as M
from .hybrid import get_law
from .metal_hybrid import _f, _hyb_header, _law_tag

__all__ = ["flux_dev_from_tau_oblate_kernel"]

_F_SW32 = 1e-5


# ---------------------------------------------------------------------------
# device code
# ---------------------------------------------------------------------------

_OB_FN = r"""
// ============================ dual numbers =============================
// a value and its derivatives d/d(x0, y0, A, B)
struct dv {
    float v; float4 d;
    dv() {}
    dv(float a) : v(a), d(float4(0.0f)) {}
    dv(float a, float4 b) : v(a), d(b) {}
};
inline dv operator+(dv a, dv b) { return dv(a.v + b.v, a.d + b.d); }
inline dv operator+(dv a, float b) { return dv(a.v + b, a.d); }
inline dv operator+(float a, dv b) { return dv(a + b.v, b.d); }
inline dv operator-(dv a, dv b) { return dv(a.v - b.v, a.d - b.d); }
inline dv operator-(dv a, float b) { return dv(a.v - b, a.d); }
inline dv operator-(float a, dv b) { return dv(a - b.v, -b.d); }
inline dv operator-(dv a) { return dv(-a.v, -a.d); }
inline dv operator*(dv a, dv b) { return dv(a.v * b.v, a.v * b.d + b.v * a.d); }
inline dv operator*(dv a, float b) { return dv(a.v * b, a.d * b); }
inline dv operator*(float a, dv b) { return dv(a * b.v, a * b.d); }
inline dv operator/(dv a, dv b) {
    float q = a.v / b.v;
    return dv(q, (a.d - q * b.d) / b.v);
}
inline dv operator/(dv a, float b) { return dv(a.v / b, a.d / b); }
inline dv operator/(float a, dv b) {
    float q = a / b.v;
    return dv(q, (-q / b.v) * b.d);
}
inline float t_val(float a) { return a; }
inline float t_val(dv a) { return a.v; }
inline float t_log(float a) { return metal::precise::log(a); }
inline dv t_log(dv a) { return dv(metal::precise::log(a.v), a.d / a.v); }
inline float t_atan2(float y, float x) { return metal::precise::atan2(y, x); }
inline dv t_atan2(dv y, dv x) {
    float r2 = x.v * x.v + y.v * y.v;
    return dv(metal::precise::atan2(y.v, x.v), (x.v * y.d - y.v * x.d) / r2);
}
// strip the tangent (a stop-gradient): v - t_nograd(v) is a pure tangent
inline float t_nograd(float a) { return a; }
inline dv t_nograd(dv a) { return dv(a.v); }

// ====================== complex numbers over T ==========================
template <typename T> struct cx {
    T re; T im;
    cx() {}
    cx(T a, T b) : re(a), im(b) {}
};
template <typename T> inline cx<T> operator+(cx<T> a, cx<T> b) {
    return cx<T>(a.re + b.re, a.im + b.im);
}
template <typename T> inline cx<T> operator-(cx<T> a, cx<T> b) {
    return cx<T>(a.re - b.re, a.im - b.im);
}
template <typename T> inline cx<T> operator-(cx<T> a) {
    return cx<T>(-a.re, -a.im);
}
template <typename T> inline cx<T> operator*(cx<T> a, cx<T> b) {
    return cx<T>(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
template <typename T> inline cx<T> operator/(cx<T> a, cx<T> b) {
    T d = b.re * b.re + b.im * b.im;
    return cx<T>((a.re * b.re + a.im * b.im) / d,
                 (a.im * b.re - a.re * b.im) / d);
}
template <typename T> inline cx<T> c_sc(cx<T> a, T s) {
    return cx<T>(a.re * s, a.im * s);
}
template <typename T> inline cx<T> c_scf(cx<T> a, float s) {
    return cx<T>(a.re * s, a.im * s);
}
template <typename T> inline cx<T> c_conj(cx<T> a) { return cx<T>(a.re, -a.im); }
template <typename T> inline cx<T> c_const(float2 a) {
    return cx<T>(T(a.x), T(a.y));
}
template <typename T> inline cx<T> c_real(T a) { return cx<T>(a, T(0.0f)); }
template <typename T> inline cx<T> c_i(cx<T> a) { return cx<T>(-a.im, a.re); }
template <typename T> inline T c_arg(cx<T> a) { return t_atan2(a.im, a.re); }
template <typename T> inline T c_logabs(cx<T> a) {
    return 0.5f * t_log(a.re * a.re + a.im * a.im);
}
template <typename T> inline float2 c_val(cx<T> a) {
    return float2(t_val(a.re), t_val(a.im));
}

// ===================== complex float2 (the solve) ========================
inline float2 cm(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
inline float2 cdiv(float2 a, float2 b) {
    float d = b.x * b.x + b.y * b.y;
    return float2((a.x * b.x + a.y * b.y) / d, (a.y * b.x - a.x * b.y) / d);
}
inline float2 cj(float2 a) { return float2(a.x, -a.y); }
inline float ca2(float2 a) { return a.x * a.x + a.y * a.y; }
inline float cab(float2 a) { return metal::precise::sqrt(ca2(a)); }
inline float carg(float2 a) { return metal::precise::atan2(a.y, a.x); }
inline float2 cexpi(float t) {
    return float2(metal::precise::cos(t), metal::precise::sin(t));
}
inline float ob_mod2pi(float t) {
    return t - MP_TWO_PI * metal::floor(t / MP_TWO_PI);
}
// principal square root, cancellation-free (oblate._csqrt)
inline float2 csq(float2 a) {
    float m = cab(a);
    float t = metal::precise::sqrt(max(0.5f * (m + fabs(a.x)), 0.0f));
    float q = (t > 0.0f) ? a.y / (2.0f * t) : 0.0f;
    return (a.x >= 0.0f) ? float2(t, q)
                         : float2(fabs(q), (a.y >= 0.0f) ? t : -t);
}
inline float2 ob_D(float2 z, float S0, float2 S1, float S2, float p) {
    float2 t = z * S2 + S1;
    t = cm(t, z) + float2(S0 - p, 0.0f);
    t = cm(t, z) + cj(S1);
    return cm(t, z) + float2(S2, 0.0f);
}
inline float2 ob_dD(float2 z, float S0, float2 S1, float S2, float p) {
    float2 t = z * (4.0f * S2) + S1 * 3.0f;
    t = cm(t, z) + float2(2.0f * (S0 - p), 0.0f);
    return cm(t, z) + cj(S1);
}

// ------------- the 4x4 eigensolver (oblate._qr_seeds, verbatim) ----------
inline void ob_givens(float2 x, float2 y, thread float *c, thread float2 *s) {
    float xa = cab(x);
    float rho = metal::precise::sqrt(xa * xa + ca2(y));
    float2 ph = (xa > 0.0f) ? x / xa : float2(1.0f, 0.0f);
    if (rho > 0.0f) {
        *c = xa / rho;
        *s = cm(ph, cj(y)) * (1.0f / rho);
    } else {
        *c = 1.0f;
        *s = float2(0.0f);
    }
}
inline float2 ob_wilkinson(float2 a, float2 b, float2 c, float2 d) {
    float2 h = (a - d) * 0.5f;
    float2 bc = cm(b, c);
    float2 disc = csq(cm(h, h) + bc);
    float2 den1 = h + disc, den2 = h - disc;
    float2 den = (ca2(den1) >= ca2(den2)) ? den1 : den2;
    return (ca2(den) > 0.0f) ? d - cdiv(bc, den) : d;
}
inline void ob_qr_sweep(thread float2 (&H)[4][4], int m) {
    float2 sig = ob_wilkinson(H[m - 2][m - 2], H[m - 2][m - 1],
                              H[m - 1][m - 2], H[m - 1][m - 1]);
    for (int i = 0; i < m; ++i) H[i][i] -= sig;
    float cs[3]; float2 sn[3];
    for (int k = 0; k < m - 1; ++k) {
        ob_givens(H[k][k], H[k + 1][k], &cs[k], &sn[k]);
        for (int j = k; j < m; ++j) {
            float2 x = H[k][j], y = H[k + 1][j];
            H[k][j] = x * cs[k] + cm(sn[k], y);
            H[k + 1][j] = y * cs[k] - cm(cj(sn[k]), x);
        }
    }
    for (int k = 0; k < m - 1; ++k) {
        int top = min(k + 2, m - 1);
        for (int i = 0; i <= top; ++i) {
            float2 x = H[i][k], y = H[i][k + 1];
            H[i][k] = x * cs[k] + cm(y, cj(sn[k]));
            H[i][k + 1] = y * cs[k] - cm(x, sn[k]);
        }
    }
    for (int i = 0; i < m; ++i) H[i][i] += sig;
}
inline void ob_qr_seeds(float S0, float2 S1, float S2, float p,
                        thread float2 *zs) {
    float inv = 1.0f / ((fabs(S2) > 0.0f) ? S2 : 1.0f);
    float2 H[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) H[i][j] = float2(0.0f);
    H[0][0] = -(S1 * inv);
    H[0][1] = float2(-(S0 - p) * inv, 0.0f);
    H[0][2] = -(cj(S1) * inv);
    H[0][3] = float2(-1.0f, 0.0f);
    H[1][0] = float2(1.0f, 0.0f);
    H[2][1] = float2(1.0f, 0.0f);
    H[3][2] = float2(1.0f, 0.0f);
    for (int pass = 0; pass < OB_BALANCE; ++pass) {
        for (int i = 0; i < 4; ++i) {
            float cn = 0.0f, rn = 0.0f;
            for (int j = 0; j < 4; ++j) {
                if (j == i) continue;
                cn += cab(H[j][i]);
                rn += cab(H[i][j]);
            }
            float f = (cn > 0.0f && rn > 0.0f)
                      ? metal::precise::sqrt(rn / cn) : 1.0f;
            float finv = 1.0f / f;
            for (int j = 0; j < 4; ++j) {
                if (j == i) continue;
                H[j][i] *= f;
                H[i][j] *= finv;
            }
        }
    }
    // early deflation: the trailing subdiagonal at rounding relative to
    // its neighbours (LAPACK's test); the sweep counts are the caps
    for (int s = 0; s < OB_SWEEP4; ++s) {
        ob_qr_sweep(H, 4);
        if (cab(H[3][2]) <= OB_DEFL * (cab(H[3][3]) + cab(H[2][2]))) break;
    }
    for (int s = 0; s < OB_SWEEP3; ++s) {
        ob_qr_sweep(H, 3);
        if (cab(H[2][1]) <= OB_DEFL * (cab(H[2][2]) + cab(H[1][1]))) break;
    }
    float2 h = (H[0][0] - H[1][1]) * 0.5f;
    float2 disc = csq(cm(h, h) + cm(H[0][1], H[1][0]));
    float2 mid = (H[0][0] + H[1][1]) * 0.5f;
    zs[0] = mid + disc;
    zs[1] = mid - disc;
    zs[2] = H[2][2];
    zs[3] = H[3][3];
}

// Aberth (at most OB_ABERTH iterations, stopping once every correction is
// at rounding) then two Newton polishes (oblate._aberth)
inline void ob_aberth(float S0, float2 S1, float S2, float p, thread float2 *z) {
    for (int it = 0; it < OB_ABERTH; ++it) {
        float2 rr[4];
        for (int i = 0; i < 4; ++i)
            rr[i] = cdiv(ob_D(z[i], S0, S1, S2, p), ob_dD(z[i], S0, S1, S2, p));
        float2 nz[4];
        bool done = true;
        for (int i = 0; i < 4; ++i) {
            float2 s = float2(0.0f);
            for (int j = 0; j < 4; ++j)
                if (j != i) s += cdiv(float2(1.0f, 0.0f), z[i] - z[j]);
            float2 corr = cdiv(rr[i], float2(1.0f, 0.0f) - cm(rr[i], s));
            nz[i] = z[i] - corr;
            done = done && (cab(corr) <= OB_ATOL * (1.0f + cab(z[i])));
        }
        for (int i = 0; i < 4; ++i) z[i] = nz[i];
        if (done) break;
    }
    for (int it = 0; it < 2; ++it)
        for (int i = 0; i < 4; ++i)
            z[i] = z[i] - cdiv(ob_D(z[i], S0, S1, S2, p),
                               ob_dD(z[i], S0, S1, S2, p));
}
// all four roots and only them: small residuals, and Vieta's sum
// (-S1/S2) and product (1) -- a duplicated root standing in for a missing
// one (Stage 1's failure) fails the sum or the product
inline bool ob_good(thread const float2 *z, float S0, float2 S1, float S2,
                    float p) {
    float2 sum = float2(0.0f), prod = float2(1.0f, 0.0f);
    float asum = 0.0f;
    for (int i = 0; i < 4; ++i) {
        float a = cab(z[i]), a2 = a * a;
        float sc = fabs(S2) * (a2 * a2 + 1.0f) + cab(S1) * a * (a2 + 1.0f)
                 + fabs(S0 - p) * a2;
        if (!(cab(ob_D(z[i], S0, S1, S2, p)) <= OB_RTOL * sc)) return false;
        sum += z[i];
        prod = cm(prod, z[i]);
        asum += a;
    }
    return (cab(sum * S2 + S1) <= OB_VTOL * (asum * fabs(S2) + cab(S1)))
           && (cab(prod - float2(1.0f, 0.0f)) <= OB_VTOL * 10.0f);
}
inline void ob_roots(float S0, float2 S1, float S2, float p, thread float2 *z) {
    ob_qr_seeds(S0, S1, S2, p, z);
    ob_aberth(S0, S1, S2, p, z);
}
// z holds a neighbouring level's roots (the quartics differ only in their
// z^2 coefficient): Aberth from there, QR if that fails the check
inline void ob_roots_warm(float S0, float2 S1, float S2, float p,
                          thread float2 *z) {
    float2 w[4] = {z[0], z[1], z[2], z[3]};
    ob_aberth(S0, S1, S2, p, w);
    if (ob_good(w, S0, S1, S2, p)) {
        for (int i = 0; i < 4; ++i) z[i] = w[i];
    } else {
        ob_roots(S0, S1, S2, p, z);
    }
}

// ---------------- crossings and arcs (oblate._crossings, _arcs) ----------
inline void ob_g_dg(float phi, float x0, float y0, float a, float b,
                    thread float *g, thread float *dg) {
    float s = metal::precise::sin(phi), c = metal::precise::cos(phi);
    float x = x0 + a * c, y = y0 + b * s;
    *g = x * x + y * y - 1.0f;
    *dg = 2.0f * (-x * a * s + y * b * c);
}
inline int ob_crossings(float x0, float y0, float a, float b,
                        thread float *pa, thread float *pb, thread float2 *z) {
    float S0 = x0 * x0 + y0 * y0 + 0.5f * (a * a + b * b);
    float2 S1 = float2(a * x0, -(b * y0));
    float S2 = 0.25f * (a * a - b * b);
    ob_roots(S0, S1, S2, 1.0f, z);
    float sc = 1.0f + (fabs(x0) + a) * (fabs(x0) + a)
                    + (fabs(y0) + b) * (fabs(y0) + b);
    float phis[4]; bool valid[4];
    for (int i = 0; i < 4; ++i) {
        bool cand = fabs(cab(z[i]) - 1.0f) < OB_ZTOL;
        float phi = cand ? carg(z[i]) : 0.0f;
        float g, dg;
        ob_g_dg(phi, x0, y0, a, b, &g, &dg);
        for (int it = 0; it < 3; ++it) {
            float trial = phi - ((fabs(dg) > 0.0f) ? g / dg : 0.0f);
            float gt, dgt;
            ob_g_dg(trial, x0, y0, a, b, &gt, &dgt);
            if (fabs(gt) < fabs(g)) { phi = trial; g = gt; dg = dgt; }
        }
        phis[i] = ob_mod2pi(phi);
        valid[i] = cand && (fabs(g) <= OB_GTOL * sc);
    }
    bool drop[4] = {false, false, false, false};
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j) {
            float d = fabs(ob_mod2pi(phis[i] - phis[j] + MP_PI) - MP_PI);
            if (valid[i] && valid[j] && d < OB_PAIRTOL) {
                drop[i] = true; drop[j] = true;
            }
        }
    int cnt = 0;
    float k0 = INFINITY, k1 = INFINITY;
    for (int i = 0; i < 4; ++i) {
        if (!valid[i] || drop[i]) continue;
        ++cnt;
        float v = phis[i];
        if (v < k0) { k1 = k0; k0 = v; } else if (v < k1) { k1 = v; }
    }
    *pa = k0; *pb = k1;
    return cnt;
}
inline void ob_arcs(float x0, float y0, float a, float b, float pa, float pb,
                    thread float *lo, thread float *hi, thread float *dth) {
    // oblate._arcs: no decision rests on a point near a crossing
    bool long_first = (pb - pa) >= MP_PI;
    float mid = long_first ? 0.5f * (pa + pb) : 0.5f * (pb + pa + MP_TWO_PI);
    float xm = x0 + a * metal::precise::cos(mid);
    float ym = y0 + b * metal::precise::sin(mid);
    bool mid_in = xm * xm + ym * ym < 1.0f;
    bool first = long_first ? mid_in : !mid_in;
    *lo = first ? pa : pb;
    *hi = first ? pb : pa + MP_TWO_PI;
    float ta = ob_mod2pi(metal::precise::atan2(y0 + b * metal::precise::sin(pa),
                                               x0 + a * metal::precise::cos(pa)));
    float tb = ob_mod2pi(metal::precise::atan2(y0 + b * metal::precise::sin(pb),
                                               x0 + a * metal::precise::cos(pb)));
    float dt = fabs(ta - tb);
    *dth = min(dt, MP_TWO_PI - dt);
}

// the frozen part of one point (everything the graph stop-gradients)
struct ob_frozen {
    int regime;                      // 0 none, 1 inside, 2 partial
    float lo, hi, dth;
    float2 z[HYB_K * 4];             // each pole level's roots
    int pj[HYB_K * 2], pl[HYB_K * 2];
    bool pg[HYB_K * 2];              // near-double pairs, two per pole
};

constant int OB_PJ[6] = {0, 0, 0, 1, 1, 2};
constant int OB_PL[6] = {1, 2, 3, 2, 3, 3};

inline void ob_pairs(thread const float2 *z0, float lo, float hi,
                     thread int *pj, thread int *pl, thread bool *pg) {
    float2 w1 = cexpi(lo), w2 = cexpi(hi);
    bool used[4] = {false, false, false, false};
    for (int round = 0; round < 2; ++round) {
        int best = 0;
        float bsc = INFINITY;
        for (int q = 0; q < 6; ++q) {
            int j = OB_PJ[q], l = OB_PL[q];
            float score = INFINITY;
            if (!used[j] && !used[l]) {
                float2 dz = z0[j] - z0[l];
                float Dl = cab(cm(dz, dz) * 0.25f);
                float2 c = (z0[j] + z0[l]) * 0.5f;
                float ang = ob_mod2pi(carg(c));
                bool on = (ang >= lo && ang <= hi)
                          || (ang + MP_TWO_PI >= lo && ang + MP_TWO_PI <= hi);
                float d_end = min(cab(w1 - c), cab(w2 - c));
                float d = on ? fabs(cab(c) - 1.0f) : d_end;
                score = Dl / max(d * d, 1e-30f);
            }
            if (score < bsc) { bsc = score; best = q; }
        }
        bool good = bsc <= OB_PAIRRATIO;
        pj[round] = OB_PJ[best];
        pl[round] = OB_PL[best];
        pg[round] = good;
        if (good) { used[OB_PJ[best]] = true; used[OB_PL[best]] = true; }
    }
}

// the regime, crossings, arcs, roots and pairs (no f < f_sw here)
inline void ob_solve(float x0, float y0, float A, float B,
                     thread ob_frozen &fz) {
    fz.regime = 0;
    fz.lo = 0.0f; fz.hi = 0.0f; fz.dth = 0.0f;
    float d = metal::precise::sqrt(x0 * x0 + y0 * y0);
    if (d - A >= 1.0f) return;
    float2 zprev[4];
    bool have = false;                 // zprev holds the level-1 roots
    if (d + A < 1.0f) {
        fz.regime = 1;
    } else {
        float pa, pb;
        int cnt = ob_crossings(x0, y0, A, B, &pa, &pb, zprev);
        have = true;
        if (cnt == 2) {
            fz.regime = 2;
            ob_arcs(x0, y0, A, B, pa, pb, &fz.lo, &fz.hi, &fz.dth);
        } else if (d < 1.0f) {
            fz.regime = 1;
        } else {
            return;
        }
    }
    float S0 = x0 * x0 + y0 * y0 + 0.5f * (A * A + B * B);
    float2 S1 = float2(A * x0, -(B * y0));
    float S2 = 0.25f * (A * A - B * B);
    // one QR solve per point; every other level warm-starts from the last
    for (int k = 0; k < HYB_K; ++k) {
        thread float2 *zk = &fz.z[4 * k];
        if (k == 0 && !have) {
            ob_roots(S0, S1, S2, 1.0f + HYB_POLE[k], zk);
        } else {
            thread const float2 *src = (k == 0) ? zprev : &fz.z[4 * (k - 1)];
            for (int i = 0; i < 4; ++i) zk[i] = src[i];
            ob_roots_warm(S0, S1, S2, 1.0f + HYB_POLE[k], zk);
        }
        if (fz.regime == 2)
            ob_pairs(&fz.z[4 * k], fz.lo, fz.hi, &fz.pj[2 * k], &fz.pl[2 * k],
                     &fz.pg[2 * k]);
    }
}

// ============================ live evaluation ============================
// e^{i t} for the live evaluation: always precise (the solve above may be
// built with fast transcendentals; nothing it computes feeds a derivative
// except through the polished, live-stepped quantities below)
inline float2 cexpi_live(float t) {
    return float2(metal::precise::cos(t), metal::precise::sin(t));
}

template <typename T>
inline cx<T> ob_Dt(cx<T> z, T S0, cx<T> S1, T S2, float p) {
    cx<T> t = c_sc(z, S2) + S1;
    t = t * z + c_real(S0 - p);
    t = t * z + c_conj(S1);
    return t * z + c_real(S2);
}
template <typename T>
inline cx<T> ob_dDt(cx<T> z, T S0, cx<T> S1, T S2, float p) {
    cx<T> t = c_sc(z, 4.0f * S2) + c_scf(S1, 3.0f);
    t = t * z + c_real(2.0f * (S0 - p));
    return t * z + c_conj(S1);
}
template <typename T>
inline cx<T> ob_Nt(cx<T> z, T K0, cx<T> K1) {
    return c_conj(K1) + c_sc(z, K0) + K1 * (z * z);
}
// one Newton step from frozen z0 with live coefficients (implicit derivative)
template <typename T>
inline cx<T> ob_live(float2 z0, T S0, cx<T> S1, T S2, float p) {
    cx<T> z = c_const<T>(z0);
    cx<T> d = ob_dDt(z, S0, S1, S2, p);
    if (ca2(c_val(d)) == 0.0f) d = c_const<T>(float2(1.0f, 0.0f));
    return z - ob_Dt(z, S0, S1, S2, p) / d;
}
// continuous change of log(e^{i phi} - z) over [lo, hi] (oblate._dlog)
template <typename T>
inline cx<T> ob_dlog(cx<T> z, float lo, float hi, bool inner, float2 w1, float2 w2) {
    cx<T> W1 = c_const<T>(w1), W2 = c_const<T>(w2);
    T dm = c_logabs(W2 - z) - c_logabs(W1 - z);
    T di;
    cx<T> one = c_const<T>(float2(1.0f, 0.0f));
    if (inner) {
        di = (hi - lo) + c_arg(one - z / W2) - c_arg(one - z / W1);
    } else {
        di = c_arg(one - W2 / z) - c_arg(one - W1 / z);
    }
    return cx<T>(dm, di);
}
template <typename T>
inline cx<T> ob_J(cx<T> u, cx<T> Delta) {
    cx<T> x = Delta / (u * u);
    cx<T> acc = c_const<T>(float2(0.0f));
    for (int k = OB_NSERIES - 1; k >= 0; --k)
        acc = acc * x + c_const<T>(float2(1.0f / (float)(2 * k + 1), 0.0f));
    return -(acc / u);
}
// one Newton step on the remainder of D mod (z^2 - sigma z + pi), the
// Jacobian frozen (oblate._bairstow)
template <typename T>
inline void ob_bairstow(thread const cx<T> *dc, float2 sig0, float2 pi0,
                        thread cx<T> *sig, thread cx<T> *pi_) {
    float2 al = float2(0.0f), be = float2(1.0f, 0.0f);
    float2 das = float2(0.0f), dbs = float2(0.0f);
    float2 dap = float2(0.0f), dbp = float2(0.0f);
    cx<T> r1 = dc[0] * c_const<T>(al), r0 = dc[0] * c_const<T>(be);
    float2 J11 = float2(0.0f), J12 = float2(0.0f);
    float2 J21 = float2(0.0f), J22 = float2(0.0f);
    for (int k = 1; k < 5; ++k) {
        float2 nal = cm(al, sig0) + be, nbe = -cm(al, pi0);
        float2 ndas = cm(das, sig0) + al + dbs, ndbs = -cm(das, pi0);
        float2 ndap = cm(dap, sig0) + dbp, ndbp = -cm(dap, pi0) - al;
        al = nal; be = nbe; das = ndas; dbs = ndbs; dap = ndap; dbp = ndbp;
        r1 = r1 + dc[k] * c_const<T>(al);
        r0 = r0 + dc[k] * c_const<T>(be);
        float2 dk = c_val(dc[k]);
        J11 += cm(dk, das); J12 += cm(dk, dap);
        J21 += cm(dk, dbs); J22 += cm(dk, dbp);
    }
    float2 det = cm(J11, J22) - cm(J12, J21);
    if (ca2(det) == 0.0f) det = float2(1.0f, 0.0f);
    cx<T> Det = c_const<T>(det);
    *sig = c_const<T>(sig0)
           - (c_const<T>(J22) * r1 - c_const<T>(J12) * r0) / Det;
    *pi_ = c_const<T>(pi0)
           - (-(c_const<T>(J21) * r1) + c_const<T>(J11) * r0) / Det;
}

// the frozen (sigma, pi) of a near-double pair, polished by detached
// Bairstow steps: the live step below is then taken at convergence, so its
// derivative is the implicit one however roughly the pair was located
// (warm-started roots converge only linearly onto a near-double pair)
template <typename T>
inline void ob_pair_polish(thread const cx<T> *dc, thread float2 *sig0,
                           thread float2 *pi0) {
    cx<float> d[5];
    for (int k = 0; k < 5; ++k) {
        float2 v = c_val(dc[k]);
        d[k] = cx<float>(v.x, v.y);
    }
    for (int it = 0; it < OB_PAIRPOLISH; ++it) {
        cx<float> s, q;
        ob_bairstow(d, *sig0, *pi0, &s, &q);
        *sig0 = float2(s.re, s.im);
        *pi0 = float2(q.re, q.im);
    }
}

// sum of N/D' over the roots of the quadratic factor (sigma, pi) of D
// (oblate._pair_sum)
template <typename T>
inline cx<T> ob_pair_sum(T S0, cx<T> S1, T S2, T K0, cx<T> K1, float p,
                         float2 sig0, float2 pi0) {
    cx<T> dc[5] = {c_real(S2), c_conj(S1), c_real(S0 - p), S1, c_real(S2)};
    ob_pair_polish(dc, &sig0, &pi0);
    cx<T> sig, pi_;
    ob_bairstow(dc, sig0, pi0, &sig, &pi_);
    cx<T> Q2 = dc[0];
    cx<T> Q1 = S1 + sig * Q2;
    cx<T> Q0 = sig * Q1 + c_real(S0 - p) - pi_ * Q2;
    cx<T> aN = K1 * sig + c_real(K0);
    cx<T> bN = c_conj(K1) - K1 * pi_;
    cx<T> aQ = Q2 * sig + Q1;
    cx<T> bQ = Q0 - Q2 * pi_;
    cx<T> den = aQ * aQ * pi_ + aQ * bQ * sig + bQ * bQ;
    if (ca2(c_val(den)) == 0.0f) den = c_const<T>(float2(1.0f, 0.0f));
    return (aN * bQ - bN * aQ) / den;
}

// the occulted flux of each generator, [mu^0, mu^2, mu^4, P_k...]
template <typename T>
inline void ob_eval(T x0, T y0, T A, T B, thread const ob_frozen &fz,
                    float x0f, float y0f, float Af, float Bf,
                    thread T *occ) {
    for (int c = 0; c < HYB_NGEN; ++c) occ[c] = T(0.0f);
    if (fz.regime == 0) return;
    T S0 = x0 * x0 + y0 * y0 + 0.5f * (A * A + B * B);
    cx<T> S1 = cx<T>(A * x0, -(B * y0));
    T S2 = 0.25f * (A * A - B * B);
    T K0 = A * B;
    cx<T> K1 = cx<T>(0.5f * (B * x0), -0.5f * (A * y0));

    // Fourier coefficients of s^p K (p = 0, 1, 2), non-negative m:
    // c^(p+1)_m = sum_j sigma_j c^(p)_{m-j}, sigma = (S2, S1*, S0, S1, S2)
    cx<T> sg[5] = {c_real(S2), c_conj(S1), c_real(S0), S1, c_real(S2)};
    cx<T> c0[2] = {c_real(K0), K1};
    cx<T> c1[4], c2[6];
    for (int m = 0; m < 4; ++m) {
        cx<T> acc = c_const<T>(float2(0.0f));
        for (int j = -2; j <= 2; ++j) {
            int k = m - j;
            if (k < -1 || k > 1) continue;
            cx<T> v = (k >= 0) ? c0[k] : c_conj(c0[-k]);
            acc = acc + sg[j + 2] * v;
        }
        c1[m] = acc;
    }
    for (int m = 0; m < 6; ++m) {
        cx<T> acc = c_const<T>(float2(0.0f));
        for (int j = -2; j <= 2; ++j) {
            int k = m - j;
            if (k < -3 || k > 3) continue;
            cx<T> v = (k >= 0) ? c1[k] : c_conj(c1[-k]);
            acc = acc + sg[j + 2] * v;
        }
        c2[m] = acc;
    }
    // h_n(s) = sum_p alpha[n][p] s^p
    const float al[3][3] = {{0.5f, 0.0f, 0.0f}, {0.5f, -0.25f, 0.0f},
                            {0.5f, -0.5f, 1.0f / 6.0f}};

    if (fz.regime == 1) {
        // the whole ellipse: 2 pi Re q_n0; each pole -(pi/p) Re sum_in N/D'
        T I[3] = {c0[0].re, c1[0].re, c2[0].re};
        for (int n = 0; n < 3; ++n) {
            T acc = T(0.0f);
            for (int p = 0; p <= n; ++p) acc = acc + al[n][p] * I[p];
            occ[n] = MP_TWO_PI * acc;
        }
        for (int k = 0; k < HYB_K; ++k) {
            float p = 1.0f + HYB_POLE[k];
            // exactly two roots inside the unit disc: one quadratic factor
            float2 sig0 = float2(0.0f), pi0 = float2(1.0f, 0.0f);
            for (int q = 0; q < 4; ++q) {
                float2 zf = fz.z[4 * k + q];
                if (cab(zf) < 1.0f) { sig0 += zf; pi0 = cm(pi0, zf); }
            }
            occ[3 + k] = (-MP_PI / p)
                         * ob_pair_sum(S0, S1, S2, K0, K1, p, sig0, pi0).re;
        }
        return;
    }

    // ---- partial: one ellipse arc [lo, hi], one star arc dth ----
    float lo = fz.lo, hi = fz.hi;
    float mid = 0.5f * (lo + hi), hlf = 0.5f * (hi - lo);
    // corner term: zero value, the frozen endpoints' motion as derivative
    T corner;
    {
        float cl = metal::precise::cos(lo), sl = metal::precise::sin(lo);
        float ch = metal::precise::cos(hi), sh = metal::precise::sin(hi);
        T vlo = (x0f + Af * cl) * (y0 + B * sl) - (y0f + Bf * sl) * (x0 + A * cl);
        T vhi = (x0f + Af * ch) * (y0 + B * sh) - (y0f + Bf * sh) * (x0 + A * ch);
        T v = vlo - vhi;
        corner = v - t_nograd(v);
    }
    T star = fz.dth + corner;
    // E_p = Re c_0 (hi - lo) + sum_m (4/m) Re(c_m e^{i m mid}) sin(m half)
    T E[3];
    for (int p = 0; p < 3; ++p) {
        int mmax = 2 * p + 1;
        T acc = ((p == 0) ? c0[0].re : ((p == 1) ? c1[0].re : c2[0].re)) * (hi - lo);
        for (int m = 1; m <= mmax; ++m) {
            cx<T> cm_ = (p == 0) ? c0[m] : ((p == 1) ? c1[m] : c2[m]);
            float2 em = cexpi_live((float)m * mid);
            float sm = metal::precise::sin((float)m * hlf);
            acc = acc + (4.0f / (float)m) * sm * (cm_.re * em.x - cm_.im * em.y);
        }
        E[p] = acc;
    }
    for (int n = 0; n < 3; ++n) {
        T acc = T(0.0f);
        for (int p = 0; p <= n; ++p) acc = acc + al[n][p] * E[p];
        occ[n] = acc + star / (2.0f * ((float)n + 1.0f));
    }

    float2 w1 = cexpi_live(lo), w2 = cexpi_live(hi);
    for (int k = 0; k < HYB_K; ++k) {
        float ep = HYB_POLE[k];
        float p = 1.0f + ep;
        thread const float2 *z0 = &fz.z[4 * k];
        bool used[4] = {false, false, false, false};
        for (int r = 0; r < 2; ++r)
            if (fz.pg[2 * k + r]) {
                used[fz.pj[2 * k + r]] = true;
                used[fz.pl[2 * k + r]] = true;
            }
        cx<T> ell = c_const<T>(float2(0.0f));
        // isolated roots: live Newton step, residue, continued log
        for (int q = 0; q < 4; ++q) {
            if (used[q]) continue;
            cx<T> z = ob_live(z0[q], S0, S1, S2, p);
            cx<T> dD = ob_dDt(z, S0, S1, S2, p);
            if (ca2(c_val(dD)) == 0.0f) continue;
            cx<T> R = -(ob_Nt(z, K0, K1) / c_i(dD));
            ell = ell + R * ob_dlog(z, lo, hi, cab(z0[q]) < 1.0f, w1, w2);
        }
        // near-double pairs as one quadratic factor (sigma, pi)
        cx<T> dc[5] = {c_real(S2), c_conj(S1), c_real(S0 - p), S1, c_real(S2)};
        for (int r = 0; r < 2; ++r) {
            if (!fz.pg[2 * k + r]) continue;
            float2 z1 = z0[fz.pj[2 * k + r]], z2 = z0[fz.pl[2 * k + r]];
            float2 sig0 = z1 + z2, pi0 = cm(z1, z2);
            ob_pair_polish(dc, &sig0, &pi0);
            cx<T> sig, pi_;
            ob_bairstow(dc, sig0, pi0, &sig, &pi_);
            cx<T> Q2 = dc[0];
            cx<T> Q1 = S1 + sig * Q2;
            cx<T> Q0 = sig * Q1 + c_real(S0 - p) - pi_ * Q2;
            cx<T> aN = K1 * sig + c_real(K0);
            cx<T> bN = c_conj(K1) - K1 * pi_;
            cx<T> aQ = Q2 * sig + Q1;
            cx<T> bQ = Q0 - Q2 * pi_;
            cx<T> den = aQ * aQ * pi_ + aQ * bQ * sig + bQ * bQ;
            if (ca2(c_val(den)) == 0.0f) den = c_const<T>(float2(1.0f, 0.0f));
            cx<T> Acoef = c_i((aN * bQ - bN * aQ) / den);
            cx<T> Ccoef = c_scf(c_i((c_scf(aN * aQ * pi_, 2.0f)
                                     + (aN * bQ + bN * aQ) * sig
                                     + c_scf(bN * bQ, 2.0f)) / den), 0.5f);
            cx<T> W1 = c_const<T>(w1), W2 = c_const<T>(w2);
            cx<T> qh = W2 * W2 - sig * W2 + pi_;
            cx<T> ql = W1 * W1 - sig * W1 + pi_;
            T darg = c_arg(qh / ql);
            // the branch of the pair's log: frozen, from the roots' own
            // continued logs (oblate._pole_arc)
            float rough = t_val(ob_dlog(c_const<float>(z1), lo, hi,
                                        cab(z1) < 1.0f, w1, w2).im)
                        + t_val(ob_dlog(c_const<float>(z2), lo, hi,
                                        cab(z2) < 1.0f, w1, w2).im);
            float wind = metal::rint((rough - t_val(darg)) / MP_TWO_PI);
            cx<T> LQ = cx<T>(c_logabs(qh) - c_logabs(ql), darg + MP_TWO_PI * wind);
            cx<T> Delta = c_scf(sig * sig, 0.25f) - pi_;
            cx<T> c = c_scf(sig, 0.5f);
            cx<T> Jd = ob_J(W2 - c, Delta) - ob_J(W1 - c, Delta);
            ell = ell + c_scf(Acoef * LQ, 0.5f) + Ccoef * Jd;
        }
        occ[3 + k] = ell.re / (2.0f * p) + star / (2.0f * p * ep);
    }
}

// ======================= shape columns and partials =======================
// B = [E0, T_1..T_NW] (deviation form, as hybrid.shape_cols) of an oblate
// planet at sky (X, Y), from the occulted generators.
inline void ob_shape(thread const float *gen, thread float *Bc) {
    Bc[0] = -gen[0];
    for (int j = 0; j < HYB_NW; ++j) {
        float b = 0.0f;
        for (int c = 0; c < HYB_NGEN; ++c) b += HYB_SMAT[j * HYB_NGEN + c] * gen[c];
        Bc[j + 1] = -b;
    }
}

inline void ob_cols(float X, float Y, float r, float fl, float cth, float sth,
                    float sqf, thread float *Bc) {
    if (fl < OB_FSW) {
        float z = metal::precise::sqrt(max(X * X + Y * Y, KITE_FLOOR));
        mp_hyb_cols(z, r, Bc);
        return;
    }
    float x0 = X * cth + Y * sth, y0 = -X * sth + Y * cth;
    float A = r / sqf, B = r * sqf;
    ob_frozen fz;
    ob_solve(x0, y0, A, B, fz);
    float gen[HYB_NGEN];
    ob_eval<float>(x0, y0, A, B, fz, x0, y0, A, B, gen);
    ob_shape(gen, Bc);
}

// wct . B and its gradient g = d/d(X, Y, r, f, theta); B returned too
inline void ob_cols_bd(float X, float Y, float r, float fl, float cth,
                       float sth, float sqf, thread const float *wct,
                       thread float *Bc, thread float *g) {
    for (int c = 0; c < 5; ++c) g[c] = 0.0f;
    if (fl < OB_FSW) {
        float z2 = X * X + Y * Y;
        float z = metal::precise::sqrt(max(z2, KITE_FLOOR));
        float Bz[HYB_NCOL], Br[HYB_NCOL];
        mp_hyb_cols_d(z, r, Bc, Bz, Br);
        float gz = 0.0f, fr = 0.0f;
        for (int c = 0; c < HYB_NCOL; ++c) { gz += wct[c] * Bz[c]; fr += wct[c] * Br[c]; }
        if (z2 > KITE_FLOOR) { g[0] = gz * X / z; g[1] = gz * Y / z; }
        g[2] = fr;
        return;
    }
    float x0 = X * cth + Y * sth, y0 = -X * sth + Y * cth;
    float A = r / sqf, B = r * sqf;
    ob_frozen fz;
    ob_solve(x0, y0, A, B, fz);
    dv gen[HYB_NGEN];
    ob_eval<dv>(dv(x0, float4(1.0f, 0.0f, 0.0f, 0.0f)),
                dv(y0, float4(0.0f, 1.0f, 0.0f, 0.0f)),
                dv(A, float4(0.0f, 0.0f, 1.0f, 0.0f)),
                dv(B, float4(0.0f, 0.0f, 0.0f, 1.0f)),
                fz, x0, y0, A, B, gen);
    float gv[HYB_NGEN];
    for (int c = 0; c < HYB_NGEN; ++c) gv[c] = gen[c].v;
    ob_shape(gv, Bc);
    // d(wct . B)/d(x0, y0, A, B): B = -[G_0, S G]
    float4 t = -wct[0] * gen[0].d;
    for (int j = 0; j < HYB_NW; ++j)
        for (int c = 0; c < HYB_NGEN; ++c)
            t -= (wct[j + 1] * HYB_SMAT[j * HYB_NGEN + c]) * gen[c].d;
    // chain: x0 = X c + Y s, y0 = -X s + Y c; A = r/sqf, B = r sqf
    g[0] = t.x * cth - t.y * sth;
    g[1] = t.x * sth + t.y * cth;
    g[2] = t.z / sqf + t.w * sqf;
    float omf = sqf * sqf;
    g[3] = (t.z * A - t.w * B) / (2.0f * omf);
    g[4] = t.x * y0 - t.y * x0;
}
"""

_OB_CONST = {
    "OB_BALANCE": "4", "OB_SWEEP4": "6", "OB_SWEEP3": "5", "OB_ABERTH": "8",
    "OB_NSERIES": "18", "OB_ZTOL": "3e-3f", "OB_GTOL": "1e-5f",
    "OB_PAIRTOL": "1e-3f", "OB_PAIRRATIO": "0.1f", "OB_FSW": _f(_F_SW32),
    "OB_DEFL": "1.2e-7f", "OB_ATOL": "2.4e-7f", "OB_RTOL": "3e-5f",
    "OB_VTOL": "1e-4f", "OB_PAIRPOLISH": "2",
}


#: the solve section (everything before the live evaluation) is built with
#: fast transcendentals: it only makes detached decisions and seeds, and
#: its outputs are polished (Newton, Bairstow) before anything is
#: differentiated. 20-35% of the kernel; the live part stays precise --
#: fast log/atan2 there cost 4% in gradients at pole tangencies.
_LIVE_SPLIT = "// ============================ live evaluation ============================"


def _ob_header(law) -> str:
    head, tail = _OB_FN.split(_LIVE_SPLIT)
    src = head.replace("metal::precise::", "metal::fast::") + _LIVE_SPLIT + tail
    for k, v in _OB_CONST.items():
        src = src.replace(k, v)
    assert "OB_" not in src.replace("ob_", "").replace("OB_PJ", "").replace(
        "OB_PL", ""), "unexpanded oblate constant"
    return _hyb_header(law) + M._subst(src)


# ---------------------------------------------------------------------------
# the orbit: sky position (X, Y) and its derivatives (oblate_tau's frame)
# ---------------------------------------------------------------------------

_OB_ORBIT_FN = r"""
// circular: X = a sin phi, Y = -b cos phi; theta = (a, b)
inline void ob_xy_circ_d(float tau, float per, float av, float bv,
                         thread float *X, thread float *Y, thread bool *front,
                         thread float *dXp, thread float *dYp,
                         thread float *dXt, thread float *dYt) {
    float phi = MP_TWO_PI * tau / per;
    float s = metal::precise::sin(phi), c = metal::precise::cos(phi);
    *X = av * s;
    *Y = -(bv * c);
    *front = c > 0.0f;
    *dXp = av * c;
    *dYp = bv * s;
    dXt[0] = s;    dYt[0] = 0.0f;
    dXt[1] = 0.0f; dYt[1] = -c;
}
inline void ob_xy_circ(float tau, float per, float av, float bv,
                       thread float *X, thread float *Y, thread bool *front) {
    float a, b, ct[2], dt[2];
    ob_xy_circ_d(tau, per, av, bv, X, Y, front, &a, &b, ct, dt);
}
"""

_OB_ECC_FN = r"""
// the transit-anchored eccentric orbit: X = -u, Y = -v ci, the same solve
// as mp_z_ecc_d; theta = (a, ci, ecw, esw, es, ec, b1, a2, b2)
inline void ob_xy_ecc_d(float tau, float per, float av,
                        thread const float *o, thread float *X,
                        thread float *Y, thread bool *front,
                        thread float *dXp, thread float *dYp,
                        thread float *dXt, thread float *dYt) {
    float o_ecw = o[0], o_esw = o[1], o_es = o[2], o_ec = o[3];
    float o_b1 = o[4], o_a2 = o[5], o_b2 = o[6], o_e = o[7];
    float o_E0 = o[8], o_Mt = o[9], o_ci = o[10];
    float phi = MP_TWO_PI * tau / per;
SOLVE
    float uu = av * (-o_ecw * omcf - o_b1 * sind);
    float vv = av * (o_a2 * cosd - o_b2 * sind - o_esw);
    *X = -uu;
    *Y = -(vv * o_ci);
    *front = vv > 0.0f;
    float dud = av * (-o_ecw * sind - o_b1 * cosd);
    float dvd = av * (-o_a2 * sind - o_b2 * cosd);
    float Dk = 1.0f + o_es * sind - o_ec * cosd;   // d delta / d phi = 1 / Dk
    *dXp = -dud / Dk;
    *dYp = -(o_ci * dvd) / Dk;
    for (int k = 0; k < 9; ++k) { dXt[k] = 0.0f; dYt[k] = 0.0f; }
    dXt[0] = -uu / av;
    dYt[0] = -(vv * o_ci) / av;
    dYt[1] = -vv;
    if (o_e != 0.0f) {
        dXt[2] = av * omcf;                          // ecw
        dYt[3] = av * o_ci;                          // esw
        dXt[4] = dud * omcf / Dk;                    // es (via delta)
        dYt[4] = o_ci * dvd * omcf / Dk;
        dXt[5] = -dud * sind / Dk;                   // ec (via delta)
        dYt[5] = -(o_ci * dvd * sind) / Dk;
        dXt[6] = av * sind;                          // b1
        dYt[7] = -(o_ci * av * cosd);                // a2
        dYt[8] = o_ci * av * sind;                   // b2
    }
}
inline void ob_xy_ecc(float tau, float per, float av, thread const float *o,
                      thread float *X, thread float *Y, thread bool *front) {
    float a, b, ct[9], dt[9];
    ob_xy_ecc_d(tau, per, av, o, X, Y, front, &a, &b, ct, dt);
}
"""

_OB_PHOT_FN = r"""
// F - 1 = (E0 - sum w_j T_j) * inv_norm
inline float ob_phot(float X, float Y, float r, float fl, float cth, float sth,
                     float sqf, thread const float *wv, float inv_norm) {
    float Bc[HYB_NCOL];
    ob_cols(X, Y, r, fl, cth, sth, sqf, Bc);
    float num = Bc[0];
    for (int j = 0; j < HYB_NW; ++j) num -= wv[j] * Bc[j + 1];
    return num * inv_norm;
}
// F and its gradient g = d/d(X, Y, r, f, theta), dFdw
inline float ob_phot_d(float X, float Y, float r, float fl, float cth,
                       float sth, float sqf, thread const float *wv,
                       float inv_norm, thread float *g, thread float *dFdw) {
    float wct[HYB_NCOL], Bc[HYB_NCOL];
    wct[0] = inv_norm;
    for (int j = 0; j < HYB_NW; ++j) wct[j + 1] = -wv[j] * inv_norm;
    ob_cols_bd(X, Y, r, fl, cth, sth, sqf, wct, Bc, g);
    float num = Bc[0];
    for (int j = 0; j < HYB_NW; ++j) num -= wv[j] * Bc[j + 1];
    float F = num * inv_norm;
    for (int j = 0; j < HYB_NW; ++j)
        dFdw[j] = (-Bc[j + 1] + F * HYB_NORM[j + 1]) * inv_norm;
    return F;
}
"""

_OB_ORBITS = {
    "circ": dict(
        inputs=["bin"], load="    float bv  = bin[y];\n",
        xy="ob_xy_circ(TT, per, av, bv, &X, &Y, &front);",
        xyd="ob_xy_circ_d(TT, per, av, bv, &X, &Y, &front, &dXp, &dYp, dXt, dYt);",
        nth=2, live="2"),
    "ecc": dict(
        inputs=["orb"],
        load="    float o[NORB_I];\n"
             "    for (int c = 0; c < NORB_I; ++c) "
             "o[c] = orb[y * NORB_C + (uint)c];\n",
        xy="ob_xy_ecc(TT, per, av, o, &X, &Y, &front);",
        xyd="ob_xy_ecc_d(TT, per, av, o, &X, &Y, &front, &dXp, &dYp, dXt, dYt);",
        nth=9, live="((o[7] != 0.0f) ? 9 : 2)"),
}

_OB_HEAD = """
    uint x = thread_position_in_grid.x;
    uint y = thread_position_in_grid.y;
    if (x >= (uint)npts) return;
    uint i = y * (uint)npts + x;

    float tau0 = taui[i];
    float per = perin[y];
    float av  = ain[y];
ORB_LOAD
    float r   = rin[y];
    float fl  = fin[y];
    float th  = thin[y];
    float cth = metal::precise::cos(th), sth = metal::precise::sin(th);
    float sqf = metal::precise::sqrt(1.0f - fl);
    float hw_exp = 0.5f * expt;
"""

_W_LOADS = """    float wv[HYB_NW];
    float inv_norm = MP_PI;
    for (int q = 0; q < HYB_NW; ++q) {
        wv[q] = win[y * (uint)HYB_NW + (uint)q];
        inv_norm -= wv[q] * HYB_NORM[q + 1];
    }
    inv_norm = 1.0f / inv_norm;
"""

# forward: F (scalar) or the NCOL basis columns, by one exposure rule
_OB_FWD = _OB_HEAD + "W_LOADS" + """
    float X, Y;
    bool front;
    float acc[NACC];
    for (int q = 0; q < NACC; ++q) acc[q] = 0.0f;
    float val[NACC];
    float S = 0.0f;
    int n_node = (mode == MODE_NONE) ? 1 : ((mode == MODE_SUPER) ? nsub : 5 * ngl);
""" + M._TAU_EDGES + """
    for (int node = 0; node < n_node; ++node) {
        float tt, w;
        if (mode == MODE_NONE) {
            tt = tau0; w = 1.0f;
        } else if (mode == MODE_SUPER) {
            float frac = (nsub == 1) ? 0.5f : (float)node / (float)(nsub - 1);
            tt = (tau0 - hw_exp) + 2.0f * hw_exp * frac; w = 1.0f;
        } else {
            int iv = node / ngl, j = node - iv * ngl;
            float lo = edge[iv], hi = edge[iv + 1];
            float hw = 0.5f * (hi - lo);
            tt = 0.5f * (lo + hi) + hw * xg[j];
            w = hw * wg[j];
        }
        ORB_XY(tt)
        if (front) {
            EVAL
            for (int q = 0; q < NACC; ++q) acc[q] += w * val[q];
        }
        S += w;
    }
    float inv = (S > 0.0f) ? (1.0f / S) : 0.0f;
    for (int q = 0; q < NACC; ++q) out[(uint)NACC * i + (uint)q] = acc[q] * inv;
"""

_EVAL_F = "val[0] = ob_phot(X, Y, r, fl, cth, sth, sqf, wv, inv_norm);"
_EVAL_B = "ob_cols(X, Y, r, fl, cth, sth, sqf, val);"

# Backward. Per-point gradient in tau; per-chain gradients in gpar slots
# (per, theta..., r, f, theta_sky, then the weights), reduced by simd_sum
# over the active lanes as the generic tau VJP does. The quadrature's exact
# tau derivative (the Leibniz edge terms and the A/S quotient) is the
# generic body's, written here for the (X, Y) chain.
_OB_VJP = _OB_HEAD + "W_LOADS" + "CT_LOAD" + """
    float X, Y, dXp, dYp;
    float dXt[NTH], dYt[NTH];
    bool front;
    float g[5];
    float dFdw[NWS];
    float A = 0.0f, S = 0.0f, dA = 0.0f, dS = 0.0f;
    float a_per = 0.0f, a_r = 0.0f, a_f = 0.0f, a_ts = 0.0f;
    float a_th[NTH], a_w[NWS];
    for (int k = 0; k < NTH; ++k) a_th[k] = 0.0f;
    for (int q = 0; q < NWS; ++q) a_w[q] = 0.0f;
    int n_node = (mode == MODE_NONE) ? 1 : ((mode == MODE_SUPER) ? nsub : 5 * ngl);
""" + M._TAU_EDGES + """
    for (int node = 0; node < n_node; ++node) {
        float tt, w, dtt, dw;
        if (mode == MODE_NONE) {
            tt = tau0; w = 1.0f; dtt = 1.0f; dw = 0.0f;
        } else if (mode == MODE_SUPER) {
            float frac = (nsub == 1) ? 0.5f : (float)node / (float)(nsub - 1);
            tt = (tau0 - hw_exp) + 2.0f * hw_exp * frac;
            w = 1.0f; dtt = 1.0f; dw = 0.0f;
        } else {
            int iv = node / ngl, j = node - iv * ngl;
            float lo = edge[iv], hi = edge[iv + 1];
            float dlo = dedge[iv], dhi = dedge[iv + 1];
            float hw = 0.5f * (hi - lo), dhw = 0.5f * (dhi - dlo);
            tt = 0.5f * (lo + hi) + hw * xg[j];
            dtt = 0.5f * (dlo + dhi) + dhw * xg[j];
            w = hw * wg[j];
            dw = dhw * wg[j];
        }
        ORB_XYD(tt)
        float f = 0.0f, dfdtt = 0.0f;
        if (front) {
            EVAL_D
            // parameter partials integrate with the SAME rule (the contacts'
            // own dependence is not chased: interior split points)
            float dFdphi = g[0] * dXp + g[1] * dYp;
            dfdtt = dFdphi * (MP_TWO_PI / per);
            a_per += w * (-dfdtt * tt / per);
            for (int k = 0; k < NTH; ++k) a_th[k] += w * (g[0] * dXt[k] + g[1] * dYt[k]);
            a_r  += w * g[2];
            a_f  += w * g[3];
            a_ts += w * g[4];
            for (int q = 0; q < NWS; ++q) a_w[q] += w * dFdw[q];
        }
        A += w * f;   S += w;
        dA += dw * f + w * dfdtt * dtt;
        dS += dw;
    }
    float g_tau = 0.0f;
    float p_per = 0.0f, p_r = 0.0f, p_f = 0.0f, p_ts = 0.0f;
    float p_th[NTH], p_w[NWS];
    for (int k = 0; k < NTH; ++k) p_th[k] = 0.0f;
    for (int q = 0; q < NWS; ++q) p_w[q] = 0.0f;
    if (S > 0.0f) {
        float inv = 1.0f / S;
        g_tau = ctv * (dA * S - A * dS) * inv * inv;
        p_per = ctv * a_per * inv;
        for (int k = 0; k < NTH; ++k) p_th[k] = ctv * a_th[k] * inv;
        p_r  = ctv * a_r * inv;
        p_f  = ctv * a_f * inv;
        p_ts = ctv * a_ts * inv;
        for (int q = 0; q < NWS; ++q) p_w[q] = ctv * a_w[q] * inv;
    }
    gtau[i] = g_tau;
    int n_live = TH_LIVE;                     // uniform over the simdgroup
    p_per = metal::simd_sum(p_per);
    for (int k = 0; k < NTH; ++k)
        if (k < n_live) p_th[k] = metal::simd_sum(p_th[k]);
    p_r  = metal::simd_sum(p_r);
    p_f  = metal::simd_sum(p_f);
    p_ts = metal::simd_sum(p_ts);
    for (int q = 0; q < NWS; ++q) p_w[q] = metal::simd_sum(p_w[q]);
    if (metal::simd_is_first()) {
        uint ngrp = ((uint)npts + 31u) / 32u;
        uint ob = y * NSLOT_C * ngrp + x / 32u;
        gpar[ob] = p_per;
        for (int k = 0; k < NTH; ++k)
            if (k < n_live) gpar[ob + (1u + (uint)k) * ngrp] = p_th[k];
        gpar[ob + (1u + NTH_U) * ngrp] = p_r;
        gpar[ob + (2u + NTH_U) * ngrp] = p_f;
        gpar[ob + (3u + NTH_U) * ngrp] = p_ts;
        for (int q = 0; q < NWS; ++q)
            gpar[ob + (4u + NTH_U + (uint)q) * ngrp] = p_w[q];
    }
"""

# scalar: F with the law's weights; basis: the cotangent contracted into
# the columns, ctv = 1 (as metal._tau_vjp_b_g) and no weight slots
_EVAL_D = "f = ob_phot_d(X, Y, r, fl, cth, sth, sqf, wv, inv_norm, g, dFdw);"
_EVAL_BD = ("{ float Bc[HYB_NCOL];\n"
            "              ob_cols_bd(X, Y, r, fl, cth, sth, sqf, wct, Bc, g);\n"
            "              f = 0.0f;\n"
            "              for (int q = 0; q < HYB_NCOL; ++q) f += wct[q] * Bc[q]; }")
_CT_SCALAR = "    float ctv = ct[i];\n"
_CT_BASIS = ("    float wct[HYB_NCOL];\n"
             "    for (int q = 0; q < HYB_NCOL; ++q)\n"
             "        wct[q] = ct[(uint)HYB_NCOL * i + (uint)q];\n"
             "    float ctv = 1.0f;\n")


def _nslot(law, orbit, basis):
    """gpar slots: per, theta..., r, f, theta_sky, then n_w weights."""
    return 4 + _OB_ORBITS[orbit]["nth"] + (0 if basis else get_law(law).n_w)


def _src(body: str, orbit: str, law, basis: bool, vjp: bool) -> str:
    import re
    plug = _OB_ORBITS[orbit]
    nth = plug["nth"]
    nw = 0 if basis else get_law(law).n_w
    s = (body.replace("ORB_LOAD\n", plug["load"])
             .replace("W_LOADS", "" if basis else _W_LOADS)
             .replace("CT_LOAD", _CT_BASIS if basis else _CT_SCALAR))
    s = re.sub(r"ORB_XYD\((\w+)\)", lambda m: plug["xyd"].replace("TT", m.group(1)), s)
    s = re.sub(r"ORB_XY\((\w+)\)", lambda m: plug["xy"].replace("TT", m.group(1)), s)
    if vjp:
        s = s.replace("EVAL_D", _EVAL_BD if basis else _EVAL_D)
        # a zero-length weight array is not C++: keep one dead slot
        s = s.replace("NWS", str(max(nw, 1)))
        if basis:
            s = (s.replace("for (int q = 0; q < 1; ++q) a_w[q] += w * dFdw[q];", "")
                  .replace("    for (int q = 0; q < 1; ++q)\n"
                           "            gpar[ob + (4u + NTH_U + (uint)q) * ngrp] = p_w[q];\n", ""))
    else:
        s = s.replace("EVAL", _EVAL_B if basis else _EVAL_F)
        s = s.replace("NACC", "HYB_NCOL" if basis else "1")
    s = (s.replace("TH_LIVE", plug["live"])
          .replace("NSLOT_C", f"{_nslot(law, orbit, basis)}u")
          .replace("NTH_U", f"{nth}u")
          .replace("NTH", str(nth))
          .replace("NORB_I", str(M.NORB))
          .replace("NORB_C", f"{M.NORB}u"))
    for marker in ("ORB_", "TH_LIVE", "NTH", "NSLOT", "EVAL", "NACC", "NWS",
                   "W_LOADS", "CT_LOAD"):
        assert marker not in s, f"unexpanded {marker} in oblate tau kernel"
    return M._tau_src(s)


def _header(law, orbit):
    hdr = M._HEADER + M._phot_header()
    if orbit == "ecc":
        hdr += M._ecc_header()
        hdr += M._subst(_OB_ECC_FN.replace("SOLVE", M._SOLVE))
    hdr += M._subst(_OB_ORBIT_FN)
    return hdr + _ob_header(law) + M._subst(_OB_PHOT_FN)


def _get_kernels(law, orbit: str, basis: bool):
    law = get_law(law)
    key = ("obl", law.definition, orbit, basis)
    if key not in M._kernels:
        hdr = _header(law, orbit)
        ins = (["taui", "perin", "ain"] + _OB_ORBITS[orbit]["inputs"]
               + ["rin", "fin", "thin"] + ([] if basis else ["win"])
               + ["cs", "xg", "wg", "expt", "mode", "ngl", "nsub", "npts"])
        tag = f"{_law_tag(law)}_{orbit}{'_b' if basis else ''}"
        M._kernels[key] = (
            mx.fast.metal_kernel(
                name=f"mp_otau_fwd_{tag}", input_names=ins,
                output_names=["out"], header=hdr,
                source=_src(_OB_FWD, orbit, law, basis, False)),
            mx.fast.metal_kernel(
                name=f"mp_otau_vjp_{tag}", input_names=ins[:-1] + ["ct", "npts"],
                output_names=["gtau", "gpar"], header=hdr,
                source=_src(_OB_VJP, orbit, law, basis, True)))
    return M._kernels[key]


def _make_core(law, exp_time, mode, n_gl, n_sub, orbit, basis):
    """custom_function over the oblate tau kernels. Primals:
    (tau2d, period, a, shape, r, f, theta, [w2d,] cs) -- metal_hybrid's,
    with the per-chain flattening and sky angle after r."""
    law = get_law(law)
    key = ("obl", law.definition, orbit, bool(basis), float(exp_time),
           int(mode), int(n_gl), int(n_sub))
    if key in M._tau_cores:
        return M._tau_cores[key]
    from .exposure import gauss_legendre
    xg_np, wg_np = gauss_legendre(n_gl)
    nth = _OB_ORBITS[orbit]["nth"]
    nslot = _nslot(law, orbit, basis)
    ncol = law.n_col

    def _static(dtype):
        return mx.array(xg_np, dtype=dtype), mx.array(wg_np, dtype=dtype)

    def _fwd(*primals):
        tau2d = primals[0]
        n, m = tau2d.shape
        xg, wg = _static(tau2d.dtype)
        k = _get_kernels(law, orbit, basis)[0]
        out = k(inputs=[*primals, xg, wg, exp_time, mode, n_gl, n_sub, int(m)],
                output_shapes=[(n, m, ncol) if basis else (n, m)],
                output_dtypes=[mx.float32],
                grid=(m, n, 1), threadgroup=(256, 1, 1))[0]
        return out

    def _vjp(primals, cotangent, output):
        ct = cotangent if isinstance(cotangent, mx.array) else cotangent[0]
        tau2d = primals[0]
        n, m = tau2d.shape
        cols = (m + 31) // 32
        xg, wg = _static(tau2d.dtype)
        k = _get_kernels(law, orbit, basis)[1]
        gtau, gpar = k(
            inputs=[*primals, xg, wg, exp_time, mode, n_gl, n_sub, ct, int(m)],
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
        g_r, g_f, g_th = g[:, 1 + nth], g[:, 2 + nth], g[:, 3 + nth]
        ws = () if basis else (g[:, 4 + nth:4 + nth + law.n_w],)
        return (gtau, g_per, g_a, g_shape, g_r, g_f, g_th, *ws,
                mx.zeros_like(primals[-1]))

    if basis:
        @mx.custom_function
        def core(tau2d, period, a, shape, r, f, theta, cs):
            return _fwd(tau2d, period, a, shape, r, f, theta, cs)
    else:
        @mx.custom_function
        def core(tau2d, period, a, shape, r, f, theta, w2d, cs):
            return _fwd(tau2d, period, a, shape, r, f, theta, w2d, cs)
    core.vjp(_vjp)
    M._tau_cores[key] = core
    return core


def flux_dev_from_tau_oblate_kernel(tau2d, per_c, a_c, b_c, r_c, f_c, th_c,
                                    w2d, exp_time, mode, n_gl, n_sub, law,
                                    basis=False, k=None, h=None):
    """The fp32 kernel route of ``oblate_tau.flux_dev_from_tau_oblate``:
    canonical (n,) parameters in, (n, m) or (n, m, ncol) out. Contacts are
    the oblate per-side ones (detached), as tau offsets."""
    ecc = k is not None
    if ecc:
        ci = M._ecc_shape(a_c, b_c, k, h)[2]
        from .anchored import pack_orbit_constants
        shape = pack_orbit_constants(k, h, ci)
        cs = contact_offsets_oblate_kernel(a_c, b_c, r_c, f_c, th_c, k, h, ci)
    else:
        shape = b_c
        cs = contact_offsets_oblate_kernel(a_c, b_c, r_c, f_c, th_c)
    cs = mx.stop_gradient(cs * (per_c / (2.0 * math.pi))[:, None])
    core = _make_core(law, exp_time, mode, n_gl, n_sub,
                      "ecc" if ecc else "circ", basis)
    ws = () if basis else (w2d,)
    return core(tau2d, per_c, a_c, shape, r_c, f_c, th_c, *ws, cs)


# ---------------------------------------------------------------------------
# per-point access to the device functions (tests and diagnostics)
# ---------------------------------------------------------------------------

_POINT_SRC = """
    uint i = thread_position_in_grid.x;
    if (i >= (uint)npts) return;
    float X = Xi[i], Y = Yi[i], r = ri[i], fl = fi[i], th = ti[i];
    float cth = metal::precise::cos(th), sth = metal::precise::sin(th);
    float sqf = metal::precise::sqrt(1.0f - fl);
    float wct[HYB_NCOL];
    for (int c = 0; c < HYB_NCOL; ++c) wct[c] = wi[(uint)HYB_NCOL * i + (uint)c];
    float Bc[HYB_NCOL], g[5];
    if (grad == 0) {
        ob_cols(X, Y, r, fl, cth, sth, sqf, Bc);
        for (int c = 0; c < 5; ++c) g[c] = 0.0f;
    } else {
        ob_cols_bd(X, Y, r, fl, cth, sth, sqf, wct, Bc, g);
    }
    uint o = (uint)(HYB_NCOL + 5) * i;
    for (int c = 0; c < HYB_NCOL; ++c) out[o + (uint)c] = Bc[c];
    for (int c = 0; c < 5; ++c) out[o + (uint)HYB_NCOL + (uint)c] = g[c];
"""


def _point_cols(X, Y, r, f, theta, law, wct=None, grad=False):
    """The device functions at single points (no orbit, no exposure):
    the shape columns B (n, ncol) and, with ``grad``, the gradient of
    wct . B in (X, Y, r, f, theta) (n, 5). Host arrays in, numpy out."""
    law = get_law(law)
    key = ("obl-point", law.definition)
    if key not in M._kernels:
        M._kernels[key] = mx.fast.metal_kernel(
            name=f"mp_opoint_{_law_tag(law)}",
            input_names=["Xi", "Yi", "ri", "fi", "ti", "wi", "grad", "npts"],
            output_names=["out"],
            header=M._HEADER + M._phot_header() + _ob_header(law),
            source=_POINT_SRC)
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    f32 = lambda v: mx.array(np.broadcast_to(np.asarray(v, np.float32), (n,)).copy())
    nc = law.n_col
    w = np.zeros((n, nc)) if wct is None else np.asarray(wct)
    with mx.stream(mx.gpu):
        out = M._kernels[key](
            inputs=[f32(X), f32(Y), f32(r), f32(f), f32(theta),
                    mx.array(w.astype(np.float32).reshape(-1)), int(grad), n],
            output_shapes=[(n * (nc + 5),)], output_dtypes=[mx.float32],
            grid=(n, 1, 1), threadgroup=(256, 1, 1))[0]
        mx.eval(out)
    out = np.asarray(out, dtype=np.float64).reshape(n, nc + 5)
    return out[:, :nc], out[:, nc:]


# ---------------------------------------------------------------------------
# contacts: one thread per chain (oblate_tau.contact_offsets_oblate)
# ---------------------------------------------------------------------------
#
# The graph version is thousands of tiny MLX operations -- 20-40 ms a call
# whatever the batch -- so the fp32 kernel route solves its contacts here,
# with the same scheme: monotone Newton on the convex q (outer) and M
# (inner) from the outside of each root, the slope by the envelope
# theorem. The circumscribed circle's contacts (the outer starts) come in.

_OB_CONTACT_FN = r"""
// q (kind -1: squared distance to the disc, 0 inside it) or M (kind +1:
// squared largest boundary distance) at phase phi, and d/dphi
inline void ob_q_slope(float phi, ORB_ARGS, float A, float B, float cth,
                       float sth, int kind, thread float *q, thread float *dq) {
    float X, Y, dXp, dYp, dXt[NTH], dYt[NTH];
    bool front;
    ORB_XYD
    float x0 = X * cth + Y * sth, y0 = -X * sth + Y * cth;
    float dx0 = dXp * cth + dYp * sth, dy0 = -dXp * sth + dYp * cth;
    float best = (kind > 0) ? -INFINITY : INFINITY, psi = 0.0f;
    for (int k = 0; k < 16; ++k) {
        float p = (float)k * (MP_TWO_PI / 16.0f);
        float xs = x0 + A * metal::precise::cos(p), ys = y0 + B * metal::precise::sin(p);
        float sv = xs * xs + ys * ys;
        if ((kind > 0) ? (sv > best) : (sv < best)) { best = sv; psi = p; }
    }
    for (int k = 0; k < 3; ++k) {
        float sp = metal::precise::sin(psi), cp = metal::precise::cos(psi);
        float xx = x0 + A * cp, yy = y0 + B * sp;
        float d1 = 2.0f * (-xx * A * sp + yy * B * cp);
        float d2 = 2.0f * (A * A * sp * sp - xx * A * cp + B * B * cp * cp - yy * B * sp);
        if (d2 * (float)kind < 0.0f) psi -= d1 / d2;
    }
    float sp = metal::precise::sin(psi), cp = metal::precise::cos(psi);
    float xx = x0 + A * cp, yy = y0 + B * sp;
    *q = xx * xx + yy * yy;
    *dq = 2.0f * (xx * dx0 + yy * dy0);
    if (kind < 0 && (x0 / A) * (x0 / A) + (y0 / B) * (y0 / B) < 1.0f) {
        *q = 0.0f; *dq = 0.0f;
    }
}
// Newton from the outside of the root (side -1 ingress, +1 egress)
inline float ob_newton_contact(float phi, ORB_ARGS, float A, float B,
                               float cth, float sth, int kind, float side,
                               float lim, thread float *g) {
    float q, dq;
    for (int it = 0; it < NCONTACT; ++it) {
        ob_q_slope(phi, ORB_PASS, A, B, cth, sth, kind, &q, &dq);
        float step = (dq * side > 0.0f) ? (q - 1.0f) / dq : 0.0f;
        phi -= clamp(step, -lim, lim);
    }
    ob_q_slope(phi, ORB_PASS, A, B, cth, sth, kind, &q, &dq);
    *g = q;
    return phi;
}
"""

_OB_CONTACT_SRC = """
    uint y = thread_position_in_grid.x;
    if (y >= (uint)nchain) return;
    float av = ain[y];
ORB_LOAD
    float r = rin[y], fl = fin[y], th = thin[y];
    float sth = metal::precise::sin(th), cth = metal::precise::cos(th);
    float sqf = metal::precise::sqrt(1.0f - fl);
    float A = r / sqf, B = r * sqf;
    float lo = st[4u * y], hi = st[4u * y + 3u];
    float lim = max(hi - lo, 1e-12f);
    float g1, g2, g3, g4;
    float c1 = ob_newton_contact(lo, ORB_PASS, A, B, cth, sth, -1, -1.0f, lim, &g1);
    float c4 = ob_newton_contact(hi, ORB_PASS, A, B, cth, sth, -1, 1.0f, lim, &g4);
    bool outer = fabs(g1 - 1.0f) < CTOL && fabs(g4 - 1.0f) < CTOL && c1 < c4;
    if (!outer) { c1 = 0.5f * (lo + hi); c4 = c1; }
    float lim_i = max(c4 - c1, 1e-12f);
    float c2 = ob_newton_contact(c1, ORB_PASS, A, B, cth, sth, 1, -1.0f, lim_i, &g2);
    float c3 = ob_newton_contact(c4, ORB_PASS, A, B, cth, sth, 1, 1.0f, lim_i, &g3);
    bool inner = outer && fabs(g2 - 1.0f) < CTOL && fabs(g3 - 1.0f) < CTOL
                 && c1 <= c2 && c2 < c3 && c3 <= c4;
    if (!inner) { c2 = 0.5f * (c1 + c4); c3 = c2; }
    out[4u * y] = c1; out[4u * y + 1u] = c2;
    out[4u * y + 2u] = c3; out[4u * y + 3u] = c4;
"""

_OB_CONTACT_ORBITS = {
    "circ": dict(args="float av, float bv", pass_="av, bv",
                 xyd="ob_xy_circ_d(phi, MP_TWO_PI, av, bv, &X, &Y, &front, &dXp, &dYp, dXt, dYt);",
                 load="    float bv = bin[y];\n", inputs=["bin"], nth=2),
    "ecc": dict(args="float av, thread const float *o", pass_="av, o",
                xyd="ob_xy_ecc_d(phi, MP_TWO_PI, av, o, &X, &Y, &front, &dXp, &dYp, dXt, dYt);",
                load="    float o[NORB_I];\n"
                     "    for (int c = 0; c < NORB_I; ++c) o[c] = orb[y * NORB_C + (uint)c];\n",
                inputs=["orb"], nth=9),
}


def _contact_kernel(orbit):
    key = ("obl-contacts", orbit)
    if key not in M._kernels:
        plug = _OB_CONTACT_ORBITS[orbit]

        def fill(src):
            src = (src.replace("ORB_ARGS", plug["args"])
                      .replace("ORB_PASS", plug["pass_"])
                      .replace("ORB_XYD", plug["xyd"])
                      .replace("ORB_LOAD\n", plug["load"])
                      .replace("NCONTACT", "10").replace("CTOL", "1e-4f")
                      .replace("NTH", str(plug["nth"]))
                      .replace("NORB_I", str(M.NORB))
                      .replace("NORB_C", f"{M.NORB}u"))
            for marker in ("ORB_", "NCONTACT", "CTOL", "NTH", "NORB"):
                assert marker not in src, f"unexpanded {marker} in contact kernel"
            return src
        hdr = M._HEADER + M._phot_header()
        if orbit == "ecc":
            hdr += M._ecc_header() + M._subst(_OB_ECC_FN.replace("SOLVE", M._SOLVE))
        hdr += M._subst(_OB_ORBIT_FN) + fill(_OB_CONTACT_FN)
        M._kernels[key] = mx.fast.metal_kernel(
            name=f"mp_ocontacts_{orbit}",
            input_names=["ain"] + plug["inputs"] + ["rin", "fin", "thin", "st", "nchain"],
            output_names=["out"], header=hdr, source=fill(_OB_CONTACT_SRC))
    return M._kernels[key]


def contact_offsets_oblate_kernel(a, b, r, f, theta, k=None, h=None, ci=None):
    """``oblate_tau.contact_offsets_oblate`` in one fp32 kernel launch:
    (n, 4) contact phases, detached. Same arguments."""
    from .exposure import contact_offsets, contact_offsets_anchored
    a, b, r, f, theta = (mx.stop_gradient(v) for v in (a, b, r, f, theta))
    A = r / mx.sqrt(1.0 - f)
    n = a.shape[0]
    if k is None:
        orbit, shape = "circ", b
        cA = contact_offsets(A, a, b)
    else:
        from .anchored import anchor_constants, pack_orbit_constants
        k, h, ci = (mx.stop_gradient(v) for v in (k, h, ci))
        consts = anchor_constants(k, h)
        orbit, shape = "ecc", pack_orbit_constants(k, h, ci, consts=consts)
        cA = contact_offsets_anchored(A, a, b, k, h, ci, consts=consts)
    st = mx.stack([mx.broadcast_to(c, (n,)) for c in cA], axis=1)
    out = _contact_kernel(orbit)(
        inputs=[a, mx.stop_gradient(shape), r, f, theta, st, n],
        output_shapes=[(n, 4)], output_dtypes=[mx.float32],
        grid=(n, 1, 1), threadgroup=(min(n, 64), 1, 1))[0]
    return mx.stop_gradient(out)
