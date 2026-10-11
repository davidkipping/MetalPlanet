"""Stage 0 spike: a forward-only fp32 Metal kernel for the oblate (elliptical)
planet, ported from SquishierPlanet's jaxlc.moments (no gradients).

Per point: (x0, y0, a, b) in the ellipse principal frame -> blocked fluxes
[B_0..B_nmax (even basis), P_1..P_K (poles)], comparable with
jaxlc.moments. With SPHERICAL_BRANCH, chains flagged spherical instead run
MetalPlanet's spherical column function (to measure the cost of housing
both paths in one kernel).

Throwaway code: it exists to measure cost and fp32 behaviour.
"""
import sys
import numpy as np
import mlx.core as mx

SP = "/Users/dkipping/Storage1/Work/Documents/Transit_Work/CODES/SquishierPlanet"
sys.path.insert(0, SP)
from squishierplanet.basis import alpha            # noqa: E402  (pure numpy)

from metalplanet import metal as M
from metalplanet import metal_hybrid as MH


def _f(v):
    return f"{float(v):.9e}f"


COMPLEX = r"""
inline float2 cm(float2 a, float2 b) { return float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
inline float2 cj(float2 a) { return float2(a.x, -a.y); }
inline float2 cdv(float2 a, float2 b) {
    float s = max(fabs(b.x), fabs(b.y));
    float2 bs = b / s; float2 as_ = a / s;
    float d = bs.x*bs.x + bs.y*bs.y;
    return float2(as_.x*bs.x + as_.y*bs.y, as_.y*bs.x - as_.x*bs.y) / d;
}
inline float cab(float2 a) {
    float s = max(fabs(a.x), fabs(a.y));
    if (s == 0.0f) return 0.0f;
    float2 t = a / s;
    return s * metal::precise::sqrt(t.x*t.x + t.y*t.y);
}
inline float carg(float2 a) { return metal::precise::atan2(a.y, a.x); }
inline float2 cei(float t) { return float2(metal::precise::cos(t), metal::precise::sin(t)); }
inline float2 csq(float2 a) {
    float r = cab(a);
    float re = metal::precise::sqrt(max(0.5f * (r + a.x), 0.0f));
    float im = metal::precise::sqrt(max(0.5f * (r - a.x), 0.0f));
    return float2(re, a.y < 0.0f ? -im : im);
}
inline float2 cD(float2 z, float2 k0, float2 k1, float2 k2, float2 k3, float2 k4) {
    return cm(cm(cm(cm(k0, z) + k1, z) + k2, z) + k3, z) + k4;
}
inline float2 cdD(float2 z, float2 k0, float2 k1, float2 k2, float2 k3) {
    return cm(cm(cm(4.0f * k0, z) + 3.0f * k1, z) + 2.0f * k2, z) + k3;
}
// Aberth on S2 z^4 + S1 z^3 + (S0 - lev) z^2 + conj(S1) z + S2, seeded from
// the circle limit; early exit when every correction is tiny.
inline void roots4w(float S0, float2 S1, float S2, float lev, thread float2 *z, bool warm) {
    float2 k0 = float2(S2, 0.0f), k1 = S1, k2 = float2(S0 - lev, 0.0f), k3 = cj(S1), k4 = k0;
    if (!warm) {
        float2 disc = csq(cm(k2, k2) - 4.0f * cm(k1, k3));
        float2 qq = (k2.x*disc.x + k2.y*disc.y) >= 0.0f ? (k2 + disc) : (k2 - disc);
        qq = -0.5f * qq;
        float2 small = -cdv(k0, k3);
        z[0] = cdv(qq, k1); z[1] = cdv(k3, qq); z[2] = small; z[3] = cdv(float2(1.0f, 0.0f), cj(small));
    }
    for (int it = 0; it < MAXIT; ++it) {
        float worst = 0.0f;
        float2 nz[4];
        for (int i = 0; i < 4; ++i) {
            float2 r = cdv(cD(z[i], k0, k1, k2, k3, k4), cdD(z[i], k0, k1, k2, k3));
            float2 s = float2(0.0f);
            for (int j = 0; j < 4; ++j) if (j != i) s += cdv(float2(1.0f, 0.0f), z[i] - z[j]);
            float2 step = cdv(r, float2(1.0f, 0.0f) - cm(r, s));
            nz[i] = z[i] - step;
            worst = max(worst, cab(step) / max(1.0f, cab(z[i])));
        }
        for (int i = 0; i < 4; ++i) z[i] = nz[i];
        if (worst < 3e-7f) break;
    }
    for (int i = 0; i < 4; ++i)
        z[i] = z[i] - cdv(cD(z[i], k0, k1, k2, k3, k4), cdD(z[i], k0, k1, k2, k3));
}
inline void roots4(float S0, float2 S1, float S2, float lev, thread float2 *z) { roots4w(S0, S1, S2, lev, z, false); }
inline float2 dlogw(float2 z, float lo, float hi, float2 w1, float2 w2) {
    float dm = metal::precise::log(cab(w2 - z)) - metal::precise::log(cab(w1 - z));
    float di;
    if (cab(z) < 1.0f)
        di = (hi - lo) + carg(float2(1.0f, 0.0f) - cdv(z, w2)) - carg(float2(1.0f, 0.0f) - cdv(z, w1));
    else
        di = carg(float2(1.0f, 0.0f) - cdv(w2, z)) - carg(float2(1.0f, 0.0f) - cdv(w1, z));
    return float2(dm, di);
}
inline float2 Jser(float2 u, float2 Delta) {
    float2 x = cdv(Delta, cm(u, u));
    float2 acc = float2(0.0f);
    for (int k = NSER - 1; k >= 0; --k) acc = cm(acc, x) + float2(1.0f / (2.0f * k + 1.0f), 0.0f);
    return -cdv(acc, u);
}
"""

BODY = r"""
    uint i = thread_position_in_grid.x;
    if (i >= (uint)npts) return;
    float x0 = xin[i], y0 = yin[i], a = ain[i], b = bin_[i];
    SPHERICAL_GUARD
    const float TP = MP_TWO_PI;
    float S0 = x0*x0 + y0*y0 + 0.5f*(a*a + b*b);
    float2 S1 = float2(a*x0, -b*y0);
    float S2 = 0.25f*(a*a - b*b);
    float K0 = a*b;  float2 K1 = float2(0.5f*b*x0, -0.5f*a*y0);
    float sc = 1.0f + (fabs(x0) + a)*(fabs(x0) + a) + (fabs(y0) + b)*(fabs(y0) + b);
    INSIDE_FAST

    // ---- limb intersections ------------------------------------------
    float2 z[4];
    roots4(S0, S1, S2, 1.0f, z);
    float ph[4]; bool vd[4];
    for (int r = 0; r < 4; ++r) {
        bool cand = fabs(cab(z[r]) - 1.0f) < ZTOL;
        float phi = cand ? carg(z[r]) : 0.0f;
        float c = metal::precise::cos(phi), s = metal::precise::sin(phi);
        float xx = x0 + a*c, yy = y0 + b*s;
        float g = xx*xx + yy*yy - 1.0f, dg = 2.0f*(-xx*a*s + yy*b*c);
        for (int it = 0; it < 3; ++it) {
            float tr = phi - (dg != 0.0f ? g / dg : 0.0f);
            float ct = metal::precise::cos(tr), st = metal::precise::sin(tr);
            float xt = x0 + a*ct, yt = y0 + b*st;
            float gt = xt*xt + yt*yt - 1.0f;
            if (fabs(gt) < fabs(g)) { phi = tr; g = gt; dg = 2.0f*(-xt*a*st + yt*b*ct); }
        }
        vd[r] = cand && fabs(g) <= GTOL * sc;
        ph[r] = fmod(fmod(phi, TP) + TP, TP);
    }
    bool drop[4] = {false, false, false, false};
    for (int p = 0; p < 4; ++p) for (int q = p + 1; q < 4; ++q) {
        float d = fabs(carg(cei(ph[p] - ph[q])));
        if (vd[p] && vd[q] && d < PTOL) { drop[p] = true; drop[q] = true; }
    }
    int cnt = 0;
    for (int r = 0; r < 4; ++r) { vd[r] = vd[r] && !drop[r]; cnt += vd[r] ? 1 : 0; }
    // sort valid angles ascending (invalid -> 2 pi)
    float phs[4];
    for (int r = 0; r < 4; ++r) phs[r] = vd[r] ? ph[r] : TP;
    for (int p = 0; p < 3; ++p) for (int q = 0; q < 3 - p; ++q)
        if (phs[q] > phs[q+1]) { float t = phs[q]; phs[q] = phs[q+1]; phs[q+1] = t; }
    float ths[4]; bool ene[4], ens[4];
    for (int r = 0; r < 4; ++r) {
        bool v = r < cnt;
        float c = metal::precise::cos(phs[r]), s = metal::precise::sin(phs[r]);
        float xi = x0 + a*c, yi = y0 + b*s;
        float th = metal::precise::atan2(yi, xi);
        ths[r] = v ? fmod(th + TP, TP) : TP;
        ene[r] = (-xi*a*s + yi*b*c) < 0.0f;
        float ct = metal::precise::cos(ths[r]), st = metal::precise::sin(ths[r]);
        ens[r] = (-(ct - x0)/(a*a)*st + (st - y0)/(b*b)*ct) < 0.0f;
    }
    float lo[5], hi[5]; bool kp[5];
    lo[0] = 0.0f; for (int r = 0; r < 4; ++r) { lo[r+1] = phs[r]; hi[r] = phs[r]; } hi[4] = TP;
    int last = clamp(cnt - 1, 0, 3);
    kp[0] = ene[last] && hi[0] > lo[0];
    for (int r = 0; r < 4; ++r) kp[r+1] = ene[r] && hi[r+1] > lo[r+1];
    // stellar arcs, sorted by angle with their flags
    float tht[4]; bool en2[4];
    for (int r = 0; r < 4; ++r) { tht[r] = ths[r]; en2[r] = ens[r]; }
    for (int p = 0; p < 3; ++p) for (int q = 0; q < 3 - p; ++q)
        if (tht[q] > tht[q+1]) { float t = tht[q]; tht[q] = tht[q+1]; tht[q+1] = t;
                                 bool e = en2[q]; en2[q] = en2[q+1]; en2[q+1] = e; }
    float los[5], his[5]; bool kps[5];
    los[0] = 0.0f; for (int r = 0; r < 4; ++r) { los[r+1] = tht[r]; his[r] = tht[r]; } his[4] = TP;
    kps[0] = en2[last] && his[0] > los[0];
    for (int r = 0; r < 4; ++r) kps[r+1] = en2[r] && his[r+1] > los[r+1];
    if (cnt % 2 == 1) {
        for (int s = 0; s < 5; ++s) {
            float m = 0.5f*(lo[s] + hi[s]);
            float xm = x0 + a*metal::precise::cos(m), ym = y0 + b*metal::precise::sin(m);
            kp[s] = (xm*xm + ym*ym < 1.0f) && hi[s] > lo[s];
            float ms = 0.5f*(los[s] + his[s]);
            float u = (metal::precise::cos(ms) - x0)/a, v = (metal::precise::sin(ms) - y0)/b;
            kps[s] = (u*u + v*v < 1.0f) && his[s] > los[s];
        }
    }
    if (cnt == 0) {
        bool star_in = ((x0/a)*(x0/a) + (y0/b)*(y0/b) < 1.0f) && (a*b >= 1.0f);
        bool ell_in = !star_in && (x0*x0 + y0*y0 < 1.0f);
        for (int s = 0; s < 5; ++s) { lo[s] = 0.0f; hi[s] = 0.0f; kp[s] = false;
                                      los[s] = 0.0f; his[s] = 0.0f; kps[s] = false; }
        if (ell_in) { hi[0] = TP; kp[0] = true; }
        if (star_in) { his[0] = TP; kps[0] = true; }
    }
    float2 WL[5], WH[5];
    for (int s = 0; s < 5; ++s) { WL[s] = cei(lo[s]); WH[s] = cei(hi[s]); }
    float dth = 0.0f, span = 0.0f;
    for (int s = 0; s < 5; ++s) { dth += kps[s] ? his[s] - los[s] : 0.0f;
                                  span += kp[s] ? hi[s] - lo[s] : 0.0f; }

    // ---- even moments: Fourier coefficients and arc sums ----------------
    float2 T[WW], Tn[WW];
    for (int m = 0; m < WW; ++m) T[m] = float2(0.0f);
    T[HH-1] = cj(K1); T[HH] = float2(K0, 0.0f); T[HH+1] = K1;
    float2 Sk[5] = {float2(S2, 0.0f), cj(S1), float2(S0, 0.0f), S1, float2(S2, 0.0f)};
    float2 q[NB][MM];
    for (int n = 0; n < NB; ++n) for (int m = 0; m < MM; ++m) q[n][m] = float2(0.0f);
    for (int p = 0; p < NB; ++p) {
        if (p > 0) {
            for (int m = 0; m < WW; ++m) Tn[m] = float2(0.0f);
            for (int j = -2; j <= 2; ++j)
                for (int m = 0; m < WW; ++m) { int src = m - j;
                    if (src >= 0 && src < WW) Tn[m] += cm(Sk[j+2], T[src]); }
            for (int m = 0; m < WW; ++m) T[m] = Tn[m];
        }
        for (int n = p; n < NB; ++n) for (int m = 0; m < MM; ++m) q[n][m] += ALPHA[n*NB + p] * T[HH + m];
    }
    for (int n = 0; n < NB; ++n) {
        float be = 0.0f;
        for (int s = 0; s < 5; ++s) {
            if (!kp[s]) continue;
            float mid = 0.5f*(lo[s] + hi[s]), hw = 0.5f*(hi[s] - lo[s]);
            float acc = q[n][0].x * (hi[s] - lo[s]);
            float2 e1 = cei(mid), em = e1; float2 h1 = cei(hw), hm = h1;
            for (int m = 1; m < MM; ++m) {
                acc += (4.0f/m) * cm(q[n][m], em).x * hm.y;
                em = cm(em, e1); hm = cm(hm, h1);
            }
            be += acc;
        }
        out[i*NG + n] = be + dth / (2.0f*(n + 1.0f));
    }

    // ---- poles: partial fractions, near-double pairs as one factor -------
    for (int k = 0; k < KP; ++k) {
        float e = POLES[k], pl = 1.0f + e;
        float2 zr[4];
        roots4(S0, S1, S2, pl, zr);
        // pair selection: up to two disjoint near-double pairs
        bool used[4] = {false, false, false, false};
        int pj[2] = {0, 0}, pl2[2] = {0, 0}; bool pg[2] = {false, false};
        for (int rd = 0; rd < 2; ++rd) {
            float best = 3e38f; int bj = 0, bl = 1;
            for (int j = 0; j < 4; ++j) for (int l = j + 1; l < 4; ++l) {
                if (used[j] || used[l]) continue;
                float2 dz = zr[j] - zr[l]; float2 Dl = 0.25f*cm(dz, dz);
                float2 c = 0.5f*(zr[j] + zr[l]);
                float ang = fmod(carg(c) + TP, TP), dmin = 3e38f;
                for (int s = 0; s < 5; ++s) { if (!kp[s]) continue;
                    bool on = (ang >= lo[s] && ang <= hi[s]) || (ang + TP >= lo[s] && ang + TP <= hi[s]);
                    float dd = on ? fabs(cab(c) - 1.0f)
                                  : min(cab(WL[s] - c), cab(WH[s] - c));
                    dmin = min(dmin, dd); }
                float score = cab(Dl) / max(dmin*dmin, 1e-30f);
                if (score < best) { best = score; bj = j; bl = l; }
            }
            pg[rd] = best <= 0.1f; pj[rd] = bj; pl2[rd] = bl;
            if (pg[rd]) { used[bj] = true; used[bl] = true; }
        }
        float2 ell = float2(0.0f);
        float2 k0 = float2(S2, 0.0f), k1 = S1, k2 = float2(S0 - pl, 0.0f), k3 = cj(S1);
        for (int r = 0; r < 4; ++r) {
            if (used[r]) continue;
            float2 Nz = cj(K1) + K0*zr[r] + cm(K1, cm(zr[r], zr[r]));
            float2 dDz = cdD(zr[r], k0, k1, k2, k3);
            float2 R = -cdv(Nz, cm(float2(0.0f, 1.0f), dDz));
            float2 dl = float2(0.0f);
            for (int s = 0; s < 5; ++s) if (kp[s]) dl += dlogw(zr[r], lo[s], hi[s], WL[s], WH[s]);
            ell += cm(R, dl);
        }
        for (int pr = 0; pr < 2; ++pr) {
            if (!pg[pr]) continue;
            float2 z1 = zr[pj[pr]], z2 = zr[pl2[pr]];
            float2 sg = z1 + z2, pi_ = cm(z1, z2);
            {   // one Bairstow step on D mod (z^2 - sg z + pi_), as the reference
                float2 dc[5] = {float2(S2, 0.0f), cj(S1), float2(S0 - pl, 0.0f), S1, float2(S2, 0.0f)};
                float2 al = float2(0.0f), be = float2(1.0f, 0.0f), das = float2(0.0f), dbs = float2(0.0f),
                       dap = float2(0.0f), dbp = float2(0.0f);
                float2 r1 = cm(dc[0], al), r0 = cm(dc[0], be);
                float2 J11 = float2(0.0f), J12 = float2(0.0f), J21 = float2(0.0f), J22 = float2(0.0f);
                for (int kk = 1; kk < 5; ++kk) {
                    float2 nal = cm(al, sg) + be, nbe = -cm(al, pi_);
                    float2 ndas = cm(das, sg) + al + dbs, ndbs = -cm(das, pi_);
                    float2 ndap = cm(dap, sg) + dbp, ndbp = -cm(dap, pi_) - al;
                    al = nal; be = nbe; das = ndas; dbs = ndbs; dap = ndap; dbp = ndbp;
                    r1 += cm(dc[kk], al); r0 += cm(dc[kk], be);
                    J11 += cm(dc[kk], das); J12 += cm(dc[kk], dap);
                    J21 += cm(dc[kk], dbs); J22 += cm(dc[kk], dbp);
                }
                float2 det = cm(J11, J22) - cm(J12, J21);
                if (cab(det) > 0.0f) {
                    sg = sg - cdv(cm(J22, r1) - cm(J12, r0), det);
                    pi_ = pi_ - cdv(-cm(J21, r1) + cm(J11, r0), det);
                }
            }
            float2 Q2 = float2(S2, 0.0f), Q1 = S1 + cm(sg, Q2);
            float2 Q0 = float2(S0 - pl, 0.0f) + cm(sg, Q1) - cm(pi_, Q2);
            float2 aN = cm(K1, sg) + float2(K0, 0.0f), bN = cj(K1) - cm(K1, pi_);
            float2 aQ = cm(Q2, sg) + Q1, bQ = Q0 - cm(Q2, pi_);
            float2 den = cm(cm(aQ, aQ), pi_) + cm(cm(aQ, bQ), sg) + cm(bQ, bQ);
            float2 Ac = cm(float2(0.0f, 1.0f), cdv(cm(aN, bQ) - cm(bN, aQ), den));
            float2 Cc = cm(float2(0.0f, 0.5f), cdv(2.0f*cm(cm(aN, aQ), pi_)
                          + cm(cm(aN, bQ) + cm(bN, aQ), sg) + 2.0f*cm(bN, bQ), den));
            float2 Dl = 0.25f*cm(sg, sg) - pi_, c = 0.5f*sg;
            float2 LQs = float2(0.0f), Js = float2(0.0f);
            for (int s = 0; s < 5; ++s) { if (!kp[s]) continue;
                float2 wh = WH[s], wl = WL[s];
                float2 qh = cm(wh, wh) - cm(sg, wh) + pi_, ql = cm(wl, wl) - cm(sg, wl) + pi_;
                float darg = carg(cdv(qh, ql));
                float rough = dlogw(z1, lo[s], hi[s], wl, wh).y + dlogw(z2, lo[s], hi[s], wl, wh).y;
                float wind = metal::precise::round((rough - darg) / TP);
                LQs += float2(metal::precise::log(cab(qh)) - metal::precise::log(cab(ql)), darg + TP*wind);
                Js += Jser(wh - c, Dl) - Jser(wl - c, Dl);
            }
            ell += 0.5f*cm(Ac, LQs) + cm(Cc, Js);
        }
        float val = ell.x / (2.0f*pl);
        bool conc = fabs(S2) <= 1e-14f*sc && cab(S1) <= 1e-14f*sc;
        if (conc) val = K0 / (pl - S0) * span / (2.0f*pl);
        out[i*NG + NB + k] = val + dth / (2.0f*pl*e);
    }
"""

INSIDE = r"""
    {   // ellipse fully inside the star: the arc is the whole ellipse. Even
        // moments are 2 pi times the DC Fourier coefficient of h_n(s) K;
        // each pole is -pi/p times Re(sum over roots INSIDE |z| < 1 of N/D').
        // No limb roots, no arc layout, no logarithms.
        float cc = metal::precise::sqrt(x0*x0 + y0*y0);
        if (cc + max(a, b) < 1.0f - 1e-6f) {
            float2 s1c = cj(S1);
            float2 s2_1 = 2.0f*S0*S1 + 2.0f*S2*s1c;             // (s^2)_1
            float s2_0 = S0*S0 + 2.0f*(S1.x*S1.x + S1.y*S1.y) + 2.0f*S2*S2;
            float dc[3];
            dc[0] = K0;
            dc[1] = S0*K0 + 2.0f*cm(S1, cj(K1)).x;
            dc[2] = s2_0*K0 + 2.0f*cm(s2_1, cj(K1)).x;
            for (int n = 0; n < NB; ++n) {
                float q0 = 0.0f;
                for (int pp = 0; pp <= n; ++pp) q0 += ALPHA[n*NB + pp] * dc[pp];
                out[i*NG + n] = MP_TWO_PI * q0;
            }
            float2 zr[4];
            for (int k = 0; k < KP; ++k) {
                float pl = 1.0f + POLES[k];
                roots4w(S0, S1, S2, pl, zr, WARM && k > 0);
                float2 k0 = float2(S2, 0.0f), k1 = S1, k2 = float2(S0 - pl, 0.0f), k3 = cj(S1);
                float acc = 0.0f;
                for (int r = 0; r < 4; ++r) {
                    if (cab(zr[r]) >= 1.0f) continue;
                    float2 Nz = cj(K1) + K0*zr[r] + cm(K1, cm(zr[r], zr[r]));
                    acc += cdv(Nz, cdD(zr[r], k0, k1, k2, k3)).x;
                }
                out[i*NG + NB + k] = -MP_PI * acc / pl;
            }
            return;
        }
    }
"""

SPH = r"""
    if (sph[i] != 0) {
        float zz = metal::precise::sqrt(x0*x0 + y0*y0);
        float Bc[HYB_NCOL];
        mp_hyb_cols(zz, metal::precise::sqrt(a*b), Bc);
        for (int c = 0; c < NG; ++c) out[i*NG + c] = (c < HYB_NCOL) ? Bc[c] : 0.0f;
        return;
    }
"""


def build(eps, n_max, law_for_sph=None, maxit=25, fast=False, tg=256, inside_fast=False, warm=False):
    NB, K = n_max + 1, len(eps)
    HH = 2 * n_max + 1
    al = alpha(n_max)
    consts = (f"#define MAXIT {maxit}\n#define WARM {int(warm)}\n#define NSER 18\n#define ZTOL 3e-3f\n"
              f"#define GTOL 1e-5f\n#define PTOL 1e-3f\n#define NB {NB}\n#define KP {K}\n"
              f"#define NG {NB + K}\n#define HH {HH}\n#define WW {2 * HH + 1}\n#define MM {2 * n_max + 2}\n"
              f"constant float POLES[{max(K, 1)}] = {{ {', '.join(_f(e) for e in eps) or '0.0f'} }};\n"
              f"constant float ALPHA[{NB * NB}] = {{ {', '.join(_f(v) for v in al.ravel())} }};\n")
    cx, bd = COMPLEX, BODY
    if fast:
        cx = cx.replace("metal::precise::", "metal::fast::"); bd = bd.replace("metal::precise::", "metal::fast::")
    header = M._HEADER + consts + cx
    bd = bd.replace("INSIDE_FAST", INSIDE if inside_fast else "")
    body = bd.replace("SPHERICAL_GUARD", "")
    ins = ["xin", "yin", "ain", "bin_", "npts"]
    name = f"oblate_spike_{n_max}_{K}_{abs(hash(tuple(eps))) % 10**8}_{int(fast)}_{maxit}_{int(inside_fast)}_{int(warm)}"
    if law_for_sph is not None:
        header += MH._hyb_header(law_for_sph)
        body = BODY.replace("SPHERICAL_GUARD", SPH)
        ins = ["xin", "yin", "ain", "bin_", "sph", "npts"]
        name += "_sph"
    k = mx.fast.metal_kernel(name=name, input_names=ins, output_names=["out"],
                             header=header, source=M._subst(body))
    NG = NB + K

    def run(x0, y0, a, b, sph=None):
        n = x0.shape[0]
        args = [x0, y0, a, b] + ([sph] if sph is not None else []) + [n]
        return k(inputs=args, grid=(n, 1, 1), threadgroup=(tg, 1, 1),
                 output_shapes=[(n * NG,)], output_dtypes=[mx.float32])[0].reshape(n, NG)
    return run
