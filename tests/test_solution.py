"""M1/M2: solution vector s0, s1, s2 vs the scalar Limbdark oracle."""

import math

import mlx.core as mx
import numpy as np

from metalplanet.solution import sn_dev
from reference_limbdark import s0_ref, s1_ref, s2_ref

RNG = np.random.default_rng(11)
PI = math.pi
TWO_PI_3 = 2.0 * math.pi / 3.0


def _eval64(z, r):
    with mx.stream(mx.cpu):
        s0d, s1d, s2d = sn_dev(
            mx.array(np.asarray(z, dtype=np.float64), dtype=mx.float64),
            mx.array(np.asarray(r, dtype=np.float64), dtype=mx.float64),
        )
        return (np.array(s0d, dtype=np.float64),
                np.array(s1d, dtype=np.float64),
                np.array(s2d, dtype=np.float64))


def _eval32(z, r):
    s0d, s1d, s2d = sn_dev(
        mx.array(np.asarray(z, dtype=np.float32)),
        mx.array(np.asarray(r, dtype=np.float32)),
    )
    return (np.array(s0d, dtype=np.float64),
            np.array(s1d, dtype=np.float64),
            np.array(s2d, dtype=np.float64))


def _ref(z, r):
    s0 = np.array([s0_ref(rr, zz) for zz, rr in zip(z, r)])
    s1 = np.array([s1_ref(rr, zz) for zz, rr in zip(z, r)])
    s2 = np.array([s2_ref(rr, zz) for zz, rr in zip(z, r)])
    return s0 - PI, s1 - TWO_PI_3, s2


def _random_zr(n):
    r = 10.0 ** RNG.uniform(-3, math.log10(0.99), n)
    kind = RNG.integers(0, 4, n)
    z = np.where(kind == 0, RNG.uniform(0, 1, n) * (1 - r),          # complete
        np.where(kind == 1, 1 - r + RNG.uniform(0, 1, n) * 2 * r,    # partial
        np.where(kind == 2, (1 + r) * (1 + RNG.uniform(0, 1, n)),    # none
                 RNG.uniform(0, 2, n))))                             # anything
    return z, r


class TestVsOracleFp64:
    def test_random_all_regimes(self):
        z, r = _random_zr(4000)
        got = _eval64(z, r)
        want = _ref(z, r)
        for g, w, tag in zip(got, want, "012"):
            np.testing.assert_allclose(g, w, rtol=2e-13, atol=2e-13,
                                       err_msg=f"s{tag}d mismatch")

    def test_docstring_anchor(self):
        # Limbdark.jl: s2_ell(0.1, 0.5) = 2.067294367278038 (their s2 = s1)
        _, s1d, _ = _eval64([0.5], [0.1])
        assert abs((s1d[0] + TWO_PI_3) - 2.067294367278038) < 1e-13

    def test_boundary_scans(self):
        """z at 1-r, 1+r, r, 0 and offsets 10^-k around them."""
        rows_z, rows_r = [], []
        offs = np.concatenate([[0.0], 10.0 ** np.arange(-15, -2, 1.0),
                               -10.0 ** np.arange(-15, -2, 1.0)])
        for r in (0.01, 0.1, 0.3, 0.5, 0.7, 0.9):
            for base in (1 - r, 1 + r, r, 0.0, 0.5):
                for o in offs:
                    z = base + o
                    if z < 0 or z > 2.5:
                        continue
                    rows_z.append(z)
                    rows_r.append(r)
        z = np.array(rows_z)
        r = np.array(rows_r)
        got = _eval64(z, r)
        want = _ref(z, r)
        for g, w, tag in zip(got, want, "012"):
            np.testing.assert_allclose(
                g, w, rtol=5e-8, atol=5e-11,
                err_msg=f"s{tag}d boundary-scan mismatch")

    def test_limits(self):
        # r -> 0: all deviations vanish
        z = RNG.uniform(0, 2, 200)
        r = np.full(200, 1e-8)
        for g in _eval64(z, r):
            assert np.max(np.abs(g)) < 1e-14
        # no overlap: exactly zero
        z, r = _random_zr(500)
        z = 1 + r + np.abs(z) + 1e-12
        for g in _eval64(z, r):
            assert np.max(np.abs(g)) == 0.0
        # z = 0 annular: s1d = 2 pi (1-r^2)^{3/2}/3 - ... (Case 10)
        r = np.linspace(0.01, 0.95, 50)
        z = np.zeros(50)
        _, s1d, _ = _eval64(z, r)
        # s1(b=0) = (2pi [r <= b] - Lambda1)/3 with Lambda1 = -2pi (1-r^2)^{3/2}
        # and r > b = 0, so s1 = 2pi (1-r^2)^{3/2} / 3
        want = 2 * PI * (1 - r * r) ** 1.5 / 3.0 - TWO_PI_3
        np.testing.assert_allclose(s1d, want, rtol=1e-12, atol=1e-13)


class TestFp32:
    def test_random_all_regimes_fp32(self):
        z, r = _random_zr(4000)
        got = _eval32(z, r)
        want = _ref(z.astype(np.float32).astype(np.float64),
                    r.astype(np.float32).astype(np.float64))
        for g, w, tag in zip(got, want, "012"):
            np.testing.assert_allclose(
                g, w, rtol=0, atol=4e-6,
                err_msg=f"s{tag}d fp32 mismatch")

    def test_boundary_scans_fp32(self):
        rows_z, rows_r = [], []
        offs = np.concatenate([[0.0], 10.0 ** np.arange(-7, -2, 0.5),
                               -10.0 ** np.arange(-7, -2, 0.5)])
        for r in (0.01, 0.1, 0.5, 0.9):
            for base in (1 - r, 1 + r, r):
                for o in offs:
                    z = base + o
                    if z < 0:
                        continue
                    rows_z.append(z)
                    rows_r.append(r)
        z = np.array(rows_z, dtype=np.float32).astype(np.float64)
        r = np.array(rows_r, dtype=np.float32).astype(np.float64)
        got = _eval32(z, r)
        want = _ref(z, r)
        for g, w, tag in zip(got, want, "012"):
            np.testing.assert_allclose(
                g, w, rtol=0, atol=2e-5,
                err_msg=f"s{tag}d fp32 boundary mismatch")
