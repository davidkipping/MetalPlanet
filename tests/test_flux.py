"""M2: quadratic flux vs batman + mpmath direct integration (independent cross-code) and the oracle.

batman's _quadratic_ld is Kreidberg's C implementation of the exact
Mandel & Agol (2002) analytic quadratic model — a fully independent
lineage from Limbdark/ALFM19. Its E/K use Hastings polynomial
approximations with ~2e-8 absolute accuracy, which bounds any agreement
with it (adjudicated: at every point where |ours - batman| > 1e-9, the
fp64 oracle sits on our value to ~1e-16). Cross-code tolerance is
therefore 3e-8 vs batman, with true 1e-9+ verification delegated to
(a) the Limbdark.jl-port oracle (2e-13, all regimes + boundary scans in
test_solution) and (b) direct mpmath numerical integration of the
limb-darkened overlap here — an oracle sharing no code or algorithm
lineage with ALFM19.
"""

import math

import mlx.core as mx
import numpy as np
from batman import _quadratic_ld

from metalplanet import flux_dev, greens_transform_np, light_curve, quad_g_coeffs
from reference_limbdark import flux_quad_ref

RNG = np.random.default_rng(23)


def _lc64(z, r, u1, u2):
    with mx.stream(mx.cpu):
        v = light_curve(
            mx.array(np.asarray(z, np.float64), dtype=mx.float64),
            mx.array(np.asarray(r, np.float64), dtype=mx.float64),
            mx.array(np.asarray(u1, np.float64), dtype=mx.float64),
            mx.array(np.asarray(u2, np.float64), dtype=mx.float64),
        )
        return np.array(v, dtype=np.float64)


def _sample_params(n):
    r = 10.0 ** RNG.uniform(-2, math.log10(0.5), n)
    z = np.where(RNG.random(n) < 0.75,
                 RNG.uniform(0, 1.2, n) * (1 + r),
                 np.abs(1 - r + RNG.uniform(-1, 1, n) * 2 * r))
    q1 = RNG.uniform(0.01, 0.99, n)
    q2 = RNG.uniform(0.01, 0.99, n)
    u1 = 2 * np.sqrt(q1) * q2
    u2 = np.sqrt(q1) * (1 - 2 * q2)
    return z, r, u1, u2


class TestVsBatman:
    def test_random_10k(self):
        z, r, u1, u2 = _sample_params(10_000)
        got = _lc64(z, r, u1, u2)
        want = np.concatenate([
            _quadratic_ld._quadratic_ld(
                np.ascontiguousarray(z[i:i + 1]), r[i], u1[i], u2[i], 1)
            for i in range(z.size)
        ])
        np.testing.assert_allclose(got, want, rtol=0, atol=3e-8)

    def test_lightcurve_grid(self):
        """Dense z-grid at several (r, u1, u2): max abs deviation < 3e-8.

        This test prints "Convergence failure in ellpic_bulirsch" twice.
        That comes from *batman's* C elliptic-integral routine, on grid
        points just outside contact (z + r - 1 ~ 1e-4), and it writes to
        the C-level stdout so pytest cannot capture it. Adjudicated
        against the Limbdark.jl-port oracle at exactly those points, our
        error is <= 3e-16 and batman's is 1e-9 to 1.8e-8 — which is why
        the tolerance here is batman's floor, not ours.
        """
        for r, u1, u2 in [(0.1, 0.4, 0.25), (0.01, 0.1, 0.5),
                          (0.3, 0.6, -0.2), (0.05, 0.9, 0.05)]:
            z = np.linspace(0.0, 1.0 + 2 * r, 4001)
            got = _lc64(z, np.full_like(z, r), np.full_like(z, u1),
                        np.full_like(z, u2))
            want = _quadratic_ld._quadratic_ld(z, r, u1, u2, 1)
            assert np.max(np.abs(got - want)) < 3e-8


class TestVsOracle:
    def test_random_vs_reference(self):
        z, r, u1, u2 = _sample_params(3000)
        got = _lc64(z, r, u1, u2)
        want = np.array([flux_quad_ref(r[i], z[i], u1[i], u2[i])
                         for i in range(z.size)])
        np.testing.assert_allclose(got, want, rtol=2e-13, atol=2e-13)

    def test_fp32_vs_fp64(self):
        """fp32 path within a few 1e-6 of its own fp64 path."""
        z, r, u1, u2 = _sample_params(5000)
        f64 = _lc64(z, r, u1, u2)
        f32 = np.array(light_curve(
            mx.array(z.astype(np.float32)), mx.array(r.astype(np.float32)),
            mx.array(u1.astype(np.float32)), mx.array(u2.astype(np.float32)),
        ), dtype=np.float64)
        assert np.max(np.abs(f32 - f64)) < 5e-6


class TestIdentitiesAndAPI:
    def test_uniform_ld_matches_lens_area(self):
        """u1 = u2 = 0: flux = 1 - A_lens/pi (classic uniform source)."""
        z, r, _, _ = _sample_params(2000)
        got = _lc64(z, r, np.zeros_like(z), np.zeros_like(z))
        # closed-form uniform overlap
        want = np.ones_like(z)
        inside = z <= 1 - r
        part = (z > 1 - r) & (z < 1 + r)
        want[inside] = 1 - r[inside] ** 2
        zz, rr = z[part], r[part]
        k0 = np.arccos(np.clip((rr**2 + zz**2 - 1) / (2 * rr * zz), -1, 1))
        k1 = np.arccos(np.clip((1 + zz**2 - rr**2) / (2 * zz), -1, 1))
        area = np.sqrt(np.maximum(
            4 * zz**2 - (1 + zz**2 - rr**2) ** 2, 0.0))
        want[part] = 1 - (rr**2 * k0 + k1 - 0.5 * area) / np.pi
        np.testing.assert_allclose(got, want, rtol=0, atol=5e-13)

    def test_flux_dev_exactly_zero_out_of_transit(self):
        z = np.array([1.2, 1.5, 3.0], dtype=np.float32)
        d = flux_dev(mx.array(z), 0.1, 0.4, 0.25)
        assert np.all(np.array(d) == 0.0)

    def test_greens_transform_quadratic_closed_form(self):
        u1, u2 = 0.37, 0.21
        g = greens_transform_np(np.array([u1, u2]))
        g0, g1, g2 = quad_g_coeffs(u1, u2)
        np.testing.assert_allclose(g, [g0, g1, g2], rtol=1e-15)

    def test_symmetry_in_z_sign(self):
        z = np.linspace(-1.5, 1.5, 301)
        got = _lc64(np.abs(z), np.full_like(z, 0.1),
                    np.full_like(z, 0.4), np.full_like(z, 0.2))
        got_signed = _lc64(z, np.full_like(z, 0.1),
                           np.full_like(z, 0.4), np.full_like(z, 0.2))
        np.testing.assert_allclose(got, got_signed, rtol=0, atol=0)


class TestVsDirectIntegration:
    """Independent oracle: F = 1 - (occulted intensity)/(total intensity)
    by direct mpmath quadrature in polar coordinates about the star center
    — no elliptic integrals, no Green's basis, no shared lineage.
    """

    @staticmethod
    def _flux_direct(r, z, u1, u2, dps=30):
        import mpmath as mp
        with mp.workdps(dps):
            r_, z_, u1_, u2_ = map(mp.mpf, (r, z, u1, u2))

            def intensity(rho):
                mu = mp.sqrt(1 - rho * rho)
                return 1 - u1_ * (1 - mu) - u2_ * (1 - mu) ** 2

            if z_ >= 1 + r_:
                return 1.0
            lo = max(z_ - r_, mp.mpf(0))
            hi = min(z_ + r_, mp.mpf(1))

            def occ_ring(rho):
                if rho == 0:
                    return mp.mpf(0)
                c = (z_ * z_ + rho * rho - r_ * r_) / (2 * z_ * rho) \
                    if z_ > 0 else mp.mpf(-1)
                if c <= -1:
                    phi = mp.pi
                elif c >= 1:
                    phi = mp.mpf(0)
                else:
                    phi = mp.acos(c)
                return 2 * phi * rho * intensity(rho)

            # split at the phi(rho) kink rho = |z - r| (inner contact)
            pieces = [lo, hi]
            kink = abs(z_ - r_)
            if lo < kink < hi:
                pieces = [lo, kink, hi]
            occ = mp.quad(occ_ring, pieces)
            if z_ < r_ and lo > 0:
                # rho < r - z ring fully occulted
                occ += mp.quad(lambda rho: 2 * mp.pi * rho * intensity(rho),
                               [0, lo])
            elif z_ == 0:
                occ = mp.quad(lambda rho: 2 * mp.pi * rho * intensity(rho),
                              [0, hi])
            total = mp.pi * (1 - u1_ / 3 - u2_ / 6)
            return float(1 - occ / total)

    def test_direct_integration_60pts(self):
        rng = np.random.default_rng(5)
        pts = []
        for _ in range(40):
            r = 10.0 ** rng.uniform(-2, np.log10(0.5))
            z = rng.uniform(0, 1 + r)
            pts.append((r, z))
        # deliberate soft spots
        for r in (0.1, 0.3):
            for z in (r, 1 - r, 1 - r + 1e-6, r + 1e-7, 0.0):
                pts.append((r, abs(z)))
        u1, u2 = 0.42, 0.18
        for r, z in pts:
            got = _lc64(np.array([z]), np.array([r]),
                        np.array([u1]), np.array([u2]))[0]
            want = self._flux_direct(r, z, u1, u2)
            assert abs(got - want) < 2e-10, (r, z, got, want, got - want)
