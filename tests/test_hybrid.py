"""Hybrid limb-darkening laws (metalplanet/hybrid.py): closed forms,
gradients, priors.

Ground truth is a 30-digit mpmath direct integration

    F - 1 = -[int I(x) alpha(x, z, r) x dx] / [2 pi int I(x) x dx]

(alpha the angle of the disc annulus at radius x inside the planet), which
shares no code with the closed forms. SquishierPlanet's ellipse code at
a = b, an independent route (Green/Fourier/quartic), is a second oracle
when it is importable from the sibling checkout.
"""

import math
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx
import pytest

import metalplanet
from metalplanet import hybrid as H
from metalplanet import ld
from metalplanet.hybrid import (HYBRID2, HYBRID4, HYBRID5, LAWS, HybridLaw, flux_dev_hybrid,
                                hybrid_norms, ladder, lens_geometry, pole_col,
                                shape_cols, shape_partials, w_to_c)
from metalplanet.poly import sn_dev_poly
from metalplanet.vjp import ld_basis_analytic

mp = pytest.importorskip("mpmath")
mp.mp.dps = 30

_SP = Path(__file__).resolve().parents[2] / "SquishierPlanet"
if _SP.is_dir() and str(_SP) not in sys.path:
    sys.path.insert(0, str(_SP))
try:
    import squishierplanet as sp
    from squishierplanet.laws import Law as SPLaw
except Exception:                                     # optional cross-check
    sp = None

from hybrid_weights import phys_w as _phys_w, vertex_weight  # noqa: E402

NAMES = ["hybrid2", "hybrid4", "hybrid5"]
RS = [0.01, 0.1, 0.3, 0.8]


def f64(x):
    return mx.array(np.asarray(x, dtype=np.float64), dtype=mx.float64)


def intensity_np(law, w, mu):
    c, d = w_to_c(law, w)
    m2 = np.asarray(mu) ** 2
    out = c[0] + c[1] * m2 + c[2] * m2 ** 2
    for dk, ek in zip(d, law.eps):
        out = out + dk / (m2 + ek) ** 2
    return out


def phys_w(law, seed):
    """A weight vector inside the physical region (triangle / simplex),
    chosen by the law's structure (hybrid_weights)."""
    return _phys_w(law, np.random.default_rng(seed))


def sp_same_hybrid2():
    """Does the SquishierPlanet checkout carry MetalPlanet's hybrid2 pole?
    (Its hybrid2 is then the same law; a checkout with another pole cannot
    be compared on hybrid2.)"""
    from squishierplanet import laws as spl
    return float(spl.HYBRID2_EPS) == H.HYBRID2_EPS


def z_grid(r, law=None):
    """Every regime and switch: contacts, z ~ r, the Q = 0 lines."""
    pts = [0.0, 0.5 * r, r, 0.5 * (1.0 - r), 1.0 - r - 1e-7, 1.0 - r,
           1.0 - r + 1e-7, 1.0, 1.0 + r - 1e-7, 1.0 + r, 1.0 + r + 0.1, 1.5]
    pts += list(np.linspace(0.02, 1.0 + r - 0.02, 7))
    if law is not None:
        for e in law.eps:
            zq = math.sqrt(1.0 + e) - r
            pts += [zq + d for d in (-1e-4, -1e-7, 0.0, 1e-7, 1e-4)]
    return np.array(sorted({float(p) for p in pts if p >= 0.0}))


# ---------------------------------------------------------------------------
# oracle
# ---------------------------------------------------------------------------

def _alpha(x, z, r):
    if x + z <= r:
        return 2 * mp.pi
    if x >= z + r or x + r <= z:
        return mp.mpf(0)
    c = (x * x + z * z - r * r) / (2 * x * z)
    return 2 * mp.acos(max(mp.mpf(-1), min(mp.mpf(1), c)))


def _occulted(I, z, r):
    """int I(x) alpha x dx over the disc, split at every kink."""
    z, r = mp.mpf(z), mp.mpf(r)
    # kinks of alpha: |z - r| (the annulus that first clips the planet's
    # edge, on either side of z = r) and z + r
    pts = sorted({mp.mpf(0), abs(z - r), min(z + r, mp.mpf(1)), mp.mpf(1)})
    if z >= 1 + r:
        return mp.mpf(0)
    return mp.quad(lambda x: I(x) * _alpha(x, z, r) * x, pts)


def oracle_flux_dev(law, w, z, r):
    c, d = w_to_c(law, w)

    def I(x):
        m2 = 1 - x * x
        out = mp.mpf(c[0]) + mp.mpf(c[1]) * m2 + mp.mpf(c[2]) * m2 ** 2
        for dk, ek in zip(d, law.eps):
            out += mp.mpf(dk) / (m2 + mp.mpf(ek)) ** 2
        return out
    total = 2 * mp.pi * mp.quad(lambda x: I(x) * x, [0, 1])
    return float(-_occulted(I, z, r) / total)


def oracle_column(I, z, r):
    """Deviation column: minus the occulted flux of intensity I."""
    return float(-_occulted(I, z, r))


# ---------------------------------------------------------------------------
# constants and law definitions
# ---------------------------------------------------------------------------

class TestDefinitions:
    def test_ladders(self):
        assert np.allclose(ladder(2), [0.01176, 0.3041], rtol=1e-3)
        assert np.allclose(ladder(3), [0.001597, 0.04128, 0.3874], rtol=1e-3)
        assert HYBRID2.eps == (0.36,)
        assert (HYBRID2.n_w, HYBRID4.n_w, HYBRID5.n_w) == (2, 4, 5)

    @pytest.mark.skipif(sp is None, reason="squishierplanet not importable")
    def test_matches_squishierplanet_definitions(self):
        from squishierplanet import laws as spl
        assert np.allclose(ladder(2), spl.ladder(2), rtol=1e-15)
        assert np.allclose(ladder(3), spl.ladder(3), rtol=1e-15)
        for name in NAMES:
            if name == "hybrid2" and not sp_same_hybrid2():
                continue
            law = LAWS[name]
            w = phys_w(law, 1)
            ref = spl.hybrid(name, w)
            c, d = w_to_c(law, w)
            ref_c = np.zeros(3)
            ref_c[:ref.c.size] = ref.c
            assert np.allclose(c, ref_c, atol=1e-14)
            order = np.argsort(ref.eps)
            assert np.allclose(ref.eps[order], law.eps, rtol=1e-15)
            assert np.allclose(ref.d[order], d, rtol=1e-12)

    @pytest.mark.parametrize("name", NAMES)
    def test_shape_form_equals_expanded_form(self, name):
        """I/I(1) = 1 - sum w_j T_j(mu), each T_j from 1 at the limb to 0
        at the centre; and the expanded c-form agrees on a mu grid."""
        law = LAWS[name]
        w = phys_w(law, 2)
        mu = np.linspace(0.0, 1.0, 101)
        m2 = mu ** 2
        T = []
        for a in law.shapes:
            T.append(a[0] + a[1] * m2 + a[2] * m2 ** 2)
        for e in law.eps:
            N = e ** -2 - (1 + e) ** -2
            T.append(((m2 + e) ** -2 - (1 + e) ** -2) / N)
        for t in T:
            assert abs(t[0] - 1.0) < 1e-12 and abs(t[-1]) < 1e-12
        shape = 1.0 - sum(wj * tj for wj, tj in zip(w, T))
        assert np.abs(intensity_np(law, w, mu) - shape).max() < 1e-12
        assert abs(intensity_np(law, w, 0.0) - (1.0 - w.sum())) < 1e-12

    @pytest.mark.parametrize("name", NAMES)
    def test_norms_are_full_disc_fluxes(self, name):
        law = LAWS[name]
        N = hybrid_norms(law)
        assert abs(N[0] - math.pi) < 1e-15
        m2 = lambda x: 1 - x * x
        shapes = [lambda x, a=a: a[0] + a[1] * m2(x) + a[2] * m2(x) ** 2
                  for a in law.shapes]
        for e in law.eps:
            Ne = e ** -2 - (1 + e) ** -2
            shapes.append(lambda x, e=e, Ne=Ne: ((m2(x) + e) ** -2
                                                 - (1 + e) ** -2) / Ne)
        for j, Tj in enumerate(shapes):
            full = float(2 * mp.pi * mp.quad(lambda x: Tj(x) * x, [0, 1]))
            assert abs(N[j + 1] - full) < 1e-13, j

    def test_unknown_law_and_wrong_count(self):
        with pytest.raises(ValueError, match="unknown hybrid law"):
            H.get_law("hybrid3")
        with pytest.raises(ValueError, match="takes 5 weights"):
            w_to_c(HYBRID5, [0.1, 0.2])
        z = f64([0.5])
        with mx.stream(mx.cpu):
            with pytest.raises(ValueError, match="takes 4 weights"):
                flux_dev_hybrid(z, f64(0.1), [0.1, 0.2], "hybrid4")
            with pytest.raises(ValueError, match="takes 2 weights"):
                flux_dev_hybrid(z, f64(0.1), f64([0.1, 0.2, 0.3]), "hybrid2")


# ---------------------------------------------------------------------------
# columns
# ---------------------------------------------------------------------------

class TestColumns:
    @pytest.mark.parametrize("r", RS)
    def test_even_columns_match_the_elliptic_recursion(self, r):
        """E0, E1, E2 from the even recursion equal the ALFM19 s_n route
        (which runs cel3) recombined; E0, E1 are ld_basis's B_0, B_2."""
        z = f64(z_grid(r))
        with mx.stream(mx.cpu):
            E0, E1, E2 = H.even_cols(z, f64(r))
            s = sn_dev_poly(z, f64(r), 4)
            B = ld_basis_analytic(z, f64(r))
            mx.eval(E0, E1, E2, *s, B)
        s0d, s1d, s2d, s3, s4 = [np.asarray(x) for x in s]
        assert np.abs(np.asarray(E0) - s0d).max() < 1e-15
        assert np.abs(np.asarray(E1) - (0.5 * s0d + 0.25 * s2d)).max() < 1e-15
        assert np.abs(np.asarray(E2) - (s0d / 3 + s2d / 6 + s4 / 6)).max() < 1e-14
        assert np.array_equal(np.asarray(E0), np.asarray(B)[:, 0])
        assert np.array_equal(np.asarray(E1), np.asarray(B)[:, 2])

    @pytest.mark.parametrize("r", RS)
    @pytest.mark.parametrize("eps", sorted(set(ladder(3) + ladder(2)
                                               + (H.HYBRID2_EPS,))))
    def test_pole_column_vs_oracle(self, r, eps):
        law = H.HybridLaw("one", (eps,), ())
        zs = z_grid(r, law)
        with mx.stream(mx.cpu):
            got = np.asarray(pole_col(f64(zs), f64(r), eps))
        I = lambda x, e=mp.mpf(eps): 1 / (1 - x * x + e) ** 2
        ref = np.array([oracle_column(I, z, r) for z in zs])
        scale = math.pi / (eps * (1 + eps))               # the column's norm
        err = np.abs(got - ref) / scale
        # within 1e-6 of a contact the lens depth d is itself resolved to
        # ~1e-16/d, and kite (the core's own geometry) carries that: 2e-13
        # measured at d = 1e-7 past the internal contact; 1e-13 elsewhere
        near = (np.abs(zs - (1 - r)) < 1e-6) | (np.abs(zs - (1 + r)) < 1e-6)
        assert err[~near].max() < 1e-13
        assert err[near].max() < 1e-12
        assert (got[zs >= 1 + r] == 0.0).all()
        # z = 0, planet fully inside: pi r^2 / (p (p - r^2))
        if r < 1:
            p = 1 + eps
            assert abs(got[zs == 0.0][0] + math.pi * r * r / (p * (p - r * r))) < 1e-14 * scale

    @pytest.mark.skipif(sp is None, reason="squishierplanet not importable")
    @pytest.mark.parametrize("r", RS)
    def test_pole_column_vs_squishierplanet(self, r):
        for eps in ladder(3) + (H.HYBRID2_EPS,):
            law = H.HybridLaw("one", (eps,), ())
            zs = z_grid(r, law)
            with mx.stream(mx.cpu):
                got = np.asarray(pole_col(f64(zs), f64(r), eps))
            ref = -np.array([SPLaw([0.0], [1.0], [eps]).blocked(z, 0.0, r, r)
                             for z in zs])
            scale = math.pi / (eps * (1 + eps))
            assert np.abs(got - ref).max() / scale < 1e-12, eps

    @pytest.mark.parametrize("name", NAMES)
    def test_shape_columns_contract(self, name):
        """flux_dev == (B @ c)/(N @ c) with c = (1, -w): the ld_basis
        contract, and the columns vanish out of transit."""
        law = LAWS[name]
        r = 0.1
        zs = z_grid(r, law)
        w = phys_w(law, 3)
        with mx.stream(mx.cpu):
            B = np.asarray(shape_cols(f64(zs), f64(r), law))
            F = np.asarray(flux_dev_hybrid(f64(zs), f64(r), w, law))
        assert B.shape == (zs.size, law.n_col)
        c = np.concatenate([[1.0], -w])
        N = hybrid_norms(law)
        assert np.abs(B @ c / (N @ c) - F).max() < 1e-15
        assert (B[zs >= 1 + r] == 0.0).all()


# ---------------------------------------------------------------------------
# the laws against the oracle
# ---------------------------------------------------------------------------

class TestFlux:
    @pytest.mark.parametrize("name", NAMES)
    @pytest.mark.parametrize("r", [0.1, 0.3])
    def test_vs_direct_integration(self, name, r):
        law = LAWS[name]
        zs = z_grid(r, law)
        for seed in (4, 5):
            w = phys_w(law, seed)
            with mx.stream(mx.cpu):
                got = np.asarray(flux_dev_hybrid(f64(zs), f64(r), w, law))
            ref = np.array([oracle_flux_dev(law, w, z, r) for z in zs])
            assert np.abs(got - ref).max() < 1e-13, (name, r, seed)

    @pytest.mark.skipif(sp is None, reason="squishierplanet not importable")
    @pytest.mark.parametrize("name", NAMES)
    @pytest.mark.parametrize("r", RS)
    def test_vs_squishierplanet(self, name, r):
        from squishierplanet import laws as spl
        if name == "hybrid2" and not sp_same_hybrid2():
            pytest.skip("this SquishierPlanet checkout's hybrid2 pole is not "
                        "MetalPlanet's")
        law = LAWS[name]
        zs = z_grid(r, law)
        for seed in (6, 7, 8):
            w = phys_w(law, seed)
            with mx.stream(mx.cpu):
                got = np.asarray(flux_dev_hybrid(f64(zs), f64(r), w, law))
            L = spl.hybrid(name, w)
            ref = np.array([sp.flux(L, z, 0.0, r, r) - 1.0 for z in zs])
            assert np.abs(got - ref).max() < 1e-12, (name, r, seed)

    @pytest.mark.parametrize("name", NAMES)
    def test_uniform_disc_at_zero_weights(self, name):
        law = LAWS[name]
        zs = z_grid(0.2)
        with mx.stream(mx.cpu):
            got = np.asarray(flux_dev_hybrid(f64(zs), f64(0.2),
                                             np.zeros(law.n_w), law))
            E0 = np.asarray(H.even_cols(f64(zs), f64(0.2))[0])
        assert np.array_equal(got, E0 / math.pi)

    @pytest.mark.parametrize("name", NAMES)
    def test_host_traced_and_batched_weights_agree(self, name):
        law = LAWS[name]
        r = 0.15
        zs = z_grid(r, law)
        W = np.stack([phys_w(law, s) for s in (9, 10, 11)])
        with mx.stream(mx.cpu):
            host = [np.asarray(flux_dev_hybrid(f64(zs), f64(r), w, law))
                    for w in W]
            traced = [np.asarray(flux_dev_hybrid(f64(zs), f64(r), f64(w), law))
                      for w in W]
            z2 = f64(np.broadcast_to(zs, (3, zs.size)))
            batch = np.asarray(flux_dev_hybrid(z2, f64(r), f64(W), law))
        for j in range(3):
            assert np.abs(host[j] - traced[j]).max() < 1e-15
            assert np.abs(host[j] - batch[j]).max() < 1e-15

    @pytest.mark.parametrize("name", NAMES)
    def test_fp32_vs_fp64(self, name):
        """At the simplex vertices (each weight alone, and all zero) the
        fp32 graph stays within 1e-6 of fp64 everywhere -- the measured
        worst is hybrid5's innermost pole at 0.33 ppm near the contacts."""
        law = LAWS[name]
        worst = 0.0
        for r in (0.05, 0.1, 0.2):
            zs = np.concatenate([np.linspace(0.0, 1 + r, 4001),
                                 z_grid(r, law)])
            for j in range(law.n_w + 1):
                w = np.zeros(law.n_w)
                if j:
                    w[j - 1] = vertex_weight(law)
                with mx.stream(mx.cpu):
                    a = np.asarray(flux_dev_hybrid(f64(zs), f64(r), w, law))
                    b = np.asarray(flux_dev_hybrid(
                        mx.array(zs, dtype=mx.float32),
                        mx.array(r, dtype=mx.float32), w, law),
                        dtype=np.float64)
                assert np.isfinite(b).all()
                worst = max(worst, np.abs(a - b).max())
        assert worst < 1e-6, worst


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------

def _switch_free(zs, r, law, h):
    """Points whose FD stencil of half-width h does not straddle a regime
    switch (contacts, z = r, the Q = 0 lines), and not z == 1 exactly:
    there the Kahan sort in _kite_sqarea ties (max(z, 1), min(z, 1)) and
    MLX's autodiff splits the gradient at the tie -- a measure-zero
    artefact of autodiff, not of the function (the analytic partials and
    both one-sided slopes agree there to 1e-8)."""
    edges = [1 - r, 1 + r, r]
    for e in law.eps:
        edges.append(math.sqrt(1 + e) - r)
    edges = np.array(edges)
    far = np.min(np.abs(zs[:, None] - edges[None, :]), axis=1) > 3 * h
    return np.logical_and(far, zs != 1.0)


class TestGradients:
    @pytest.mark.parametrize("name", NAMES)
    def test_autodiff_vs_finite_differences(self, name):
        law = LAWS[name]
        r = 0.12
        zs = z_grid(r, law)
        w = phys_w(law, 12)
        h = 1e-6
        # the cotangent is zeroed on the tie / switch points (see
        # _switch_free): perturbing r moves the switches, so the FD sums
        # for r and w would straddle them there just as the z stencil does
        ct = np.random.default_rng(0).normal(size=zs.shape)
        ct = ct * _switch_free(zs, r, law, h)
        with mx.stream(mx.cpu):
            c = f64(ct)

            def loss(z, rr, ww):
                return mx.sum(c * flux_dev_hybrid(z, rr, ww, law))

            gz, gr, gw = mx.grad(loss, argnums=(0, 1, 2))(f64(zs), f64(r), f64(w))
            mx.eval(gz, gr, gw)
            gz, gr, gw = np.asarray(gz), float(gr), np.asarray(gw)

            def val(z, rr, ww):
                return np.asarray(flux_dev_hybrid(f64(z), f64(rr), ww, law))

            fz = ct * (val(zs + h, r, w) - val(zs - h, r, w)) / (2 * h)
            fr = float(np.sum(ct * (val(zs, r + h, w) - val(zs, r - h, w)) / (2 * h)))
            fw = []
            for j in range(law.n_w):
                wp, wm = w.copy(), w.copy()
                wp[j] += h
                wm[j] -= h
                fw.append(float(np.sum(ct * (val(zs, r, wp) - val(zs, r, wm)) / (2 * h))))
        ok = _switch_free(zs, r, law, h)
        assert np.abs(gz - fz)[ok].max() / np.abs(fz).max() < 1e-7
        assert abs(gr - fr) / abs(fr) < 1e-6
        for j in range(law.n_w):
            assert abs(gw[j] - fw[j]) / max(abs(fw[j]), 1e-6) < 1e-7, j

    @pytest.mark.parametrize("name", NAMES)
    def test_analytic_partials_match_autodiff(self, name):
        """shape_partials is the reference for the kernel VJP: it must equal
        autodiff of shape_cols everywhere, the regime switches included.
        Everywhere except the two max/min ties z == r and z == 1 of the
        Kahan sort in _kite_sqarea, where MLX's autodiff splits the
        gradient at the tie (measure zero; the analytic partials there
        are the ones that match finite differences)."""
        law = LAWS[name]
        for r in (0.05, 0.3):
            zs = z_grid(r, law)
            zs = zs[(zs != r) & (zs != 1.0)]
            with mx.stream(mx.cpu):
                dz, dr = shape_partials(f64(zs), f64(r), law)
                mx.eval(dz, dr)
                for j in range(law.n_col):
                    ct = np.zeros((zs.size, law.n_col))
                    ct[:, j] = 1.0
                    c = f64(ct)
                    gz, gr_tot = mx.grad(
                        lambda z, rr: mx.sum(c * shape_cols(z, rr, law)),
                        argnums=(0, 1))(f64(zs), f64(r))
                    mx.eval(gz, gr_tot)
                    s = np.abs(np.asarray(dz)[:, j]).max() + 1e-12
                    assert np.abs(np.asarray(gz) - np.asarray(dz)[:, j]).max() / s < 1e-11, j
                    s = np.abs(np.asarray(dr)[:, j]).sum() + 1e-12
                    assert abs(float(gr_tot) - float(np.asarray(dr)[:, j].sum())) / s < 1e-11, j

    @pytest.mark.parametrize("name", NAMES)
    def test_partials_continuous_across_switches(self, name):
        """The Q = 0 lines are removable singularities of the pole forms:
        one-sided values of every partial agree to rounding. At the
        internal contact z = 1 - r the partials are continuous but with
        the square-root cusp every occulted-flux derivative has there
        (d occ/dz ~ kite ~ sqrt(depth)), so the one-sided values at a
        distance d agree only to O(sqrt d)."""
        law = LAWS[name]
        r = 0.2
        with mx.stream(mx.cpu):
            for e in law.eps:
                z0 = math.sqrt(1 + e) - r
                # the partials curve (B'' ~ 10 for the innermost pole), so
                # the one-sided gap at +-delta is ~2 delta B'' even when
                # continuous; extrapolate the gap to zero separation
                d1, _ = shape_partials(f64([z0 - 1e-9, z0 + 1e-9]), f64(r), law)
                d2, _ = shape_partials(f64([z0 - 2e-9, z0 + 2e-9]), f64(r), law)
                d1, d2 = np.asarray(d1), np.asarray(d2)
                gap1 = np.abs(d1[1] - d1[0])
                gap2 = np.abs(d2[1] - d2[0])
                jump = np.abs(2.0 * gap1 - gap2)
                scale = np.abs(d1).max(axis=0) + 1e-9
                assert (jump / scale < 1e-7).all(), (e, jump / scale)
            for d in (1e-6, 1e-8):
                z0 = 1 - r
                dz, dr = shape_partials(f64([z0 - d, z0 + d]), f64(r), law)
                dz, dr = np.asarray(dz), np.asarray(dr)
                gap = max(np.abs(dz[0] - dz[1]).max(), np.abs(dr[0] - dr[1]).max())
                assert gap < 5.0 * math.sqrt(d), (d, gap)

    @pytest.mark.parametrize("dtype", [mx.float64, mx.float32],
                             ids=["fp64", "fp32"])
    @pytest.mark.parametrize("name", NAMES)
    def test_no_nan_gradients_on_boundary_sweep(self, dtype, name):
        law = LAWS[name]
        rng = np.random.default_rng(3)
        r = 0.17
        base = np.array([0.0, r, 1 - r, 1.0, 1 + r]
                        + [math.sqrt(1 + e) - r for e in law.eps])
        n = 5000
        off = np.where(rng.random(n) < 0.3, 0.0,
                       10.0 ** rng.uniform(-9, -2, n) * rng.choice([-1, 1], n))
        zs = np.clip(rng.choice(base, n) + off, 0.0, None)
        w = phys_w(law, 13)
        with mx.stream(mx.cpu):
            c = mx.array(np.ones(n), dtype=dtype)

            def loss(z, rr, ww):
                return mx.sum(c * flux_dev_hybrid(z, rr, ww, law))

            g = mx.grad(loss, argnums=(0, 1, 2))(
                mx.array(zs, dtype=dtype), mx.array(r, dtype=dtype),
                mx.array(w, dtype=dtype))
            mx.eval(g)
        for x in g:
            assert np.isfinite(np.asarray(x)).all()


# ---------------------------------------------------------------------------
# priors
# ---------------------------------------------------------------------------

class TestPriors:
    def test_hybrid2_vertices(self):
        Vc, Vl = ld.hybrid2_vertices()
        assert np.allclose(Vc, [-0.1246, 1.1246], atol=2e-4)
        assert np.allclose(Vl, [1.2010, -0.2010], atol=2e-4)
        assert all(np.array_equal(a, b) for a, b in zip(
            ld.hybrid2_vertices(law="hybrid2"), (Vc, Vl)))

    @pytest.mark.parametrize("which", ["hybrid2", "custom-pole"])
    def test_hybrid2_triangle(self, which):
        """Uniform samples of the exact triangle satisfy its (A)(B)(C) and
        give a non-negative, centre-brightening profile; the map inverts;
        MLX agrees with numpy -- for hybrid2, and for a custom hybrid2-type
        law whose pole reaches the prior only through ``law=``."""
        law = (HYBRID2 if which == "hybrid2" else
               HybridLaw("custom2", (0.15,), HYBRID2.shapes))
        kw = {} if which == "hybrid2" else {"law": law}
        e = law.eps[0]
        N = e ** -2 - (1 + e) ** -2
        g0, g1 = 2 / (N * e ** 3), 2 / (N * (1 + e) ** 3)
        rng = np.random.default_rng(0)
        q1, q2 = rng.random(5000), rng.random(5000)
        w1, w2 = ld.hybrid2_from_q_np(q1, q2, **kw)
        assert (w1 + w2 <= 1 + 1e-12).all()
        assert (w1 + g0 * w2 >= -1e-12).all()
        assert (w1 + g1 * w2 >= -1e-12).all()
        mu = np.linspace(0, 1, 201)
        for k in range(0, 5000, 250):
            I = intensity_np(law, [w1[k], w2[k]], mu)
            assert (I >= -1e-12).all() and (np.diff(I) >= -1e-12).all()
        b1, b2 = ld.hybrid2_to_q_np(w1, w2, **kw)
        assert np.allclose(b1, q1, atol=1e-12) and np.allclose(b2, q2, atol=1e-12)
        with mx.stream(mx.cpu):
            m1, m2 = ld.hybrid2_from_q(f64(q1[:50]), f64(q2[:50]), **kw)
        assert np.allclose(np.asarray(m1), w1[:50], atol=1e-15)
        assert np.allclose(np.asarray(m2), w2[:50], atol=1e-15)

    def test_prior_pole_arguments(self):
        """law= ties the triangle to a law; it refuses a non-hybrid2-type
        law and a conflicting eps=."""
        custom = HybridLaw("custom2", (0.15,), HYBRID2.shapes)
        assert all(np.array_equal(a, b) for a, b in zip(
            ld.hybrid2_vertices(law=custom), ld.hybrid2_vertices(eps=0.15)))
        with pytest.raises(ValueError, match="not a hybrid2-type law"):
            ld.hybrid2_vertices(law="hybrid4")
        with pytest.raises(ValueError, match="not both"):
            ld.hybrid2_from_q_np(0.5, 0.5, eps=0.2, law=custom)

    @pytest.mark.skipif(sp is None, reason="squishierplanet not importable")
    def test_priors_match_squishierplanet(self):
        from squishierplanet import laws as spl
        rng = np.random.default_rng(1)
        q1, q2 = rng.random(100), rng.random(100)
        ref = spl.hybrid2_from_q(q1, q2)
        w1, w2 = ld.hybrid2_from_q_np(q1, q2, eps=spl.HYBRID2_EPS)
        assert np.allclose(np.stack([w1, w2], -1), ref, atol=1e-15)
        q = rng.random((100, 5))
        assert np.allclose(ld.simplex_from_q_np(q), spl.simplex_from_q(q),
                           atol=1e-15)

    @pytest.mark.parametrize("n", [4, 5])
    def test_simplex(self, n):
        rng = np.random.default_rng(2)
        q = rng.random((20000, n))
        w = ld.simplex_from_q_np(q)
        assert (w >= -1e-12).all() and (w.sum(1) <= 1 + 1e-12).all()
        # flat Dirichlet(1,...,1) on n+1 parts: each mean 1/(n+1)
        assert np.allclose(w.mean(0), 1 / (n + 1), atol=0.01)
        assert np.allclose(ld.q_from_simplex_np(w), q, atol=1e-10)
        with mx.stream(mx.cpu):
            wm = np.asarray(ld.simplex_from_q(f64(q[:50])))
        assert np.allclose(wm, w[:50], atol=1e-15)
        law = HYBRID4 if n == 4 else HYBRID5
        mu = np.linspace(0, 1, 201)
        for k in range(0, 20000, 1000):
            I = intensity_np(law, w[k], mu)
            assert (I >= -1e-12).all() and (np.diff(I) >= -1e-12).all()
