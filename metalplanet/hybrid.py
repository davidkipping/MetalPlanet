"""Hybrid limb-darkening laws: even powers of mu plus double poles.

SquishierPlanet (docs/notes, "The resulting law") defines a family of laws
built from a few even powers of mu and "double-pole" terms

    P_eps(mu) = (mu^2 + eps)^-2,        eps > 0,

which are infinite only at a projected radius sqrt(1 + eps) *outside* the
star and so rise steeply at the limb, the way a real profile does and a
polynomial in mu^2 cannot. Three members are named, each written in a
*shape basis* in which every term falls from 1 at the limb to 0 at the
centre, so each weight is simply the share of the centre-to-limb drop its
term carries:

    I(mu) / I(1) = 1 - sum_j w_j T_j(mu),          I(0) / I(1) = 1 - sum_j w_j

    hybrid2 :  T = {1 - mu^2,  Pi_0.208}                           2 weights
    hybrid4 :  T = {1 - mu^4,  (1 - mu^2)^2,  Pi_eps0, Pi_eps1}    4 weights
    hybrid5 :  T = {1 - mu^4,  (1 - mu^2)^2,  Pi_eps0..2}          5 weights

    Pi_eps(mu) = [P_eps(mu) - (1 + eps)^-2] / N_eps,   N_eps = eps^-2 - (1 + eps)^-2,

with the poles of hybrid4 and hybrid5 on the analytic ladder
eps_k = exp[2 pi (sqrt(k + 1/2) - sqrt K)] (K = 2, 3) and hybrid2's single
pole tuned on J-band stars. On simulated M-G dwarfs hybrid4 is 15-20x more
accurate than the quadratic law and hybrid5 ties the Claret four-parameter
law. The physical region is an exact triangle for hybrid2 and, for the
others, the simplex w_j >= 0, sum w_j <= 1 (``ld.hybrid2_from_q``,
``ld.simplex_from_q`` sample them uniformly).

Why they are cheap here. The laws were designed so an *oblate* planet's
light curve is elementary; for the spherical planet this package handles,
everything collapses onto quantities the ALFM19 core already computes
(``solution.sn_dev_with_aux``: the kappas, the kite area, the signed Heron
area), and -- unlike the quadratic law, whose mu^1 term needs elliptic
integrals -- **no column needs an elliptic integral**:

* even powers mu^0, mu^2, mu^4 come from s_0, s_2 and the ALFM19 *even*
  recursion M_0 -> M_2 -> M_4 (``poly.py``'s recursion restricted to even
  n, which never touches the elliptic M_1, M_3);
* a pole term's occulted flux follows from Green's theorem with the
  potential h(s) = 1/(2p(p - s)), p = 1 + eps, s = r_sky^2: the stellar
  limb arc gives kap1 / (p eps), and the planet arc reduces to
  J = int_{-kap0}^{kap0} dpsi / (a_ + b_ cos psi) with a_ = p - z^2 - r^2,
  b_ = 2 z r, an atan/log of rational functions of (z, r).

Every column is computed in *deviation* form (visible minus full-disk,
i.e. minus the occulted flux: negative, exactly 0 out of transit), the
package's float32 discipline, and the law is assembled in the shape form

    F - 1 = (E0 - sum_j w_j T_j) / (pi - sum_j w_j N_j)

whose columns are all O(depth) -- the expanded coefficient form carries the
pole's eps^-2 scale (3.9e5 for hybrid5's innermost pole) and is kept only
for cross-checks (``w_to_c``). The ``ld_basis`` contract of the quadratic
law carries over unchanged: with B = [E0, T_1..T_n], c = (1, -w) and
N = ``hybrid_norms``, flux - 1 == (B @ c) / (N @ c).

Numerics of the pole term (partial-overlap regime), with
A = 1 - (z - r)^2, Bp = 1 - (z + r)^2 (product forms, as the core),
Q = a_^2 - b_^2 = (eps + A)(eps + Bp), U = (eps + A)(-Bp), V = (eps + Bp) A,
y = V / U:

    Q > 0  (z + r < sqrt p):  J = 4 / sqrt(Q) * atan2(sqrt V, sqrt U)
    Q < 0:                    J = 4 / sqrt(-Q) * ln[(sqrt U + sqrt(-V)) / (2 sqrt(z r eps))]
    |y| small:                J = 4 T G(y) / (a_ + b_),  G = sum_k (-y)^k/(2k+1),  T = sqrt(A / -Bp)

The sign of Q flips on the line z + r = sqrt(1 + eps), which lies inside
the partial regime of every transit (twice per transit per pole): a
removable singularity of J, bridged by the series. The atan2 form carries
no tan(kap0 / 2), so kap0 -> pi is harmless, and the log form's argument is
written without the cancelling difference sqrt U - sqrt(-V) = 4 z r eps /
(sqrt U + sqrt(-V)). The complete-transit value is the cancellation-free
2 pi r^2 / (sqrt Q (K + sqrt Q)), K = p + r^2 - z^2.

Partials for the analytic VJP come from the boundary rule for any radial
intensity I(rho^2), rho^2 = z^2 + r^2 - 2 z r cos psi: d occ/dr = r int I dpsi
and d occ/dz = -r int I cos psi dpsi over the planet arc inside the star
(it reproduces ds0/dr = -2 r kap0 and ds0/dz = kite / z). For the pole they
are J2 = int (a_ + b_ cos)^-2 and its cosine moment, elementary in J; for
the even columns, sin and sin 2 of kap0. All of it is checked against a
40-digit direct integration and against finite differences in the tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .ellip import dtype_eps
from .solution import _kite_sqarea

__all__ = ["HybridLaw", "HYBRID2", "HYBRID4", "HYBRID5", "LAWS",
           "HYBRID2_EPS", "ladder", "get_law", "hybrid_norms", "w_to_c",
           "shape_cols", "flux_dev_hybrid", "combine_cols", "shape_partials",
           "even_cols", "pole_col", "lens_geometry"]

_PI = math.pi
_TWO_PI = 2.0 * math.pi

#: hybrid2's single pole, tuned on the J-band training stars
HYBRID2_EPS = 0.208

#: half-width in y = V/U of the series bridge across the Q = 0 line, by
#: precision. Outside it the closed forms are used, and their partials
#: J2, J2c divide a numerator that vanishes like y by Q, losing ~log10(1/y)
#: digits: 1e-3 costs 3 of fp64's 16, 0.05 costs 1.3 of fp32's 7. Inside
#: it the six-term series of G(y) = sum (-y)^k / (2k + 1) truncates at
#: y^6/13 (1e-19 in fp64 at 1e-3; 1.2e-9 in fp32 at 0.05, and its
#: derivative at 6 y^5/13 = 1.4e-7). The Metal kernels use the fp32 rule.
Y_SW = {mx.float64: 1e-3, mx.float32: 0.05}


def ladder(K: int, alpha_eff: float = 1.0) -> tuple:
    """The analytic pole ladder eps_k = exp[(2 pi / sqrt alpha)
    (sqrt(k + 1/2) - sqrt K)], k = 0..K-1 (alpha_eff = 1 for light curves),
    as plain Python floats."""
    return tuple(math.exp(2.0 * math.pi / math.sqrt(alpha_eff)
                          * (math.sqrt(k + 0.5) - math.sqrt(K)))
                 for k in range(int(K)))


@dataclass(frozen=True)
class HybridLaw:
    """A named law: its poles and its even shape terms.

    ``shapes`` holds the even shape terms as (a0, a1, a2) coefficients over
    the monomial columns (mu^0, mu^2, mu^4): (1, -1, 0) is 1 - mu^2,
    (1, 0, -1) is 1 - mu^4, (1, -2, 1) is (1 - mu^2)^2. The pole shapes
    Pi_eps follow, one per entry of ``eps``, in that order. Weights are
    ordered (even shapes..., pole shapes...), as in the notes.
    """
    name: str
    eps: tuple
    shapes: tuple

    @property
    def n_w(self) -> int:
        return len(self.shapes) + len(self.eps)

    @property
    def n_col(self) -> int:
        """Columns of the shape basis: E0 plus one per weight."""
        return 1 + self.n_w

    @property
    def uses_mu4(self) -> bool:
        return any(a[2] != 0.0 for a in self.shapes)

    def generator_matrix(self) -> np.ndarray:
        """S (n_w, 3 + K): each shape term T_j as a combination of the
        generator columns [mu^0, mu^2, mu^4, P_eps_0, ...]."""
        K = len(self.eps)
        S = np.zeros((self.n_w, 3 + K))
        for j, a in enumerate(self.shapes):
            S[j, :3] = a
        for k, e in enumerate(self.eps):
            p = 1.0 + e
            N_eps = e ** -2 - p ** -2
            j = len(self.shapes) + k
            S[j, 0] = -(p ** -2) / N_eps          # Pi_eps = (P_eps - p^-2) / N_eps
            S[j, 3 + k] = 1.0 / N_eps
        return S

    def generator_norms(self) -> np.ndarray:
        """Full-disk flux of each generator: pi/(n+1) for mu^{2n};
        pi (1/eps - 1/(1+eps)) = pi / (eps (1+eps)) for a pole."""
        return np.array([_PI, _PI / 2.0, _PI / 3.0]
                        + [_PI / (e * (1.0 + e)) for e in self.eps])

    def norms(self) -> np.ndarray:
        """[pi, N_1..N_n]: the full-disk flux of E0 and of each shape term
        (e.g. pi/2 for 1 - mu^2, pi eps/(1 + 2 eps) for Pi_eps)."""
        return np.concatenate([[_PI], self.generator_matrix()
                               @ self.generator_norms()])


HYBRID2 = HybridLaw("hybrid2", (HYBRID2_EPS,), ((1.0, -1.0, 0.0),))
HYBRID4 = HybridLaw("hybrid4", ladder(2), ((1.0, 0.0, -1.0), (1.0, -2.0, 1.0)))
HYBRID5 = HybridLaw("hybrid5", ladder(3), ((1.0, 0.0, -1.0), (1.0, -2.0, 1.0)))
LAWS = {law.name: law for law in (HYBRID2, HYBRID4, HYBRID5)}


def get_law(law) -> HybridLaw:
    if isinstance(law, HybridLaw):
        return law
    try:
        return LAWS[law]
    except (KeyError, TypeError):
        raise ValueError(f"unknown hybrid law {law!r}; expected one of "
                         f"{sorted(LAWS)}") from None


def hybrid_norms(law) -> np.ndarray:
    """N = [pi, N_1..N_n] for the ``ld_basis`` contract
    flux - 1 == (B @ c) / (N @ c), c = (1, -w). Float64, host-side."""
    return get_law(law).norms()


def w_to_c(law, w):
    """The expanded form I = c0 + c1 mu^2 + c2 mu^4 + sum_k d_k P_eps_k
    for weights w: (c (3,), d (K,)). For cross-checks only -- the d_k carry
    the eps^-2 scale, which is why nothing else here uses this form."""
    law = get_law(law)
    w = np.asarray(w, dtype=np.float64)
    if w.shape != (law.n_w,):
        raise ValueError(f"{law.name} takes {law.n_w} weights; got shape "
                         f"{w.shape}")
    coef = -(w @ law.generator_matrix())
    coef[0] += 1.0
    return coef[:3], coef[3:]


# ---------------------------------------------------------------------------
# geometry shared by every column (the first half of solution.sn_dev_with_aux,
# without the elliptic block)
# ---------------------------------------------------------------------------

def lens_geometry(z: mx.array, r):
    """Masks and the lens geometry of a circular planet on the disc, with
    every masked denominator already sanitised for autodiff."""
    dtype = z.dtype
    eps = dtype_eps(dtype)
    tiny = 1e-30 if dtype == mx.float32 else 1e-280

    z = mx.abs(z)
    zero = z * 0.0 + r * 0.0
    z = z + zero
    r = r + zero
    r2 = r * r
    z2 = z * z

    m_none = z >= 1.0 + r
    m_comp = mx.logical_and(z <= 1.0 - r, mx.logical_not(m_none))
    m_part = mx.logical_not(mx.logical_or(m_none, m_comp))
    one = zero + 1.0

    A = mx.maximum((r + 1.0 - z) * (1.0 - r + z), tiny)     # 1 - (z-r)^2
    Bp = (1.0 - z - r) * (1.0 + z + r)                       # 1 - (z+r)^2
    sqarea = _kite_sqarea(z, r)                              # signed Heron
    kite_floor = (10.0 * eps) ** 2
    kite_safe = mx.sqrt(mx.where(m_part, mx.maximum(sqarea, kite_floor), 1.0))
    kite = mx.where(m_part, kite_safe, 0.0)
    kap0 = mx.arctan2(kite, r2 + z2 - 1.0)      # pi complete, 0 outside
    kap1 = mx.arctan2(kite, 1.0 + z2 - r2)

    # sin and cos of kap0 from the geometry, not from trig
    twozr = mx.where(m_part, mx.maximum(2.0 * z * r, tiny), one)
    sink = mx.where(m_part, kite / twozr, 0.0)
    cosk = mx.where(m_part, (r2 + z2 - 1.0) / twozr, -one)
    return dict(z=z, r=r, r2=r2, z2=z2, zero=zero, one=one, tiny=tiny,
                m_none=m_none, m_comp=m_comp, m_part=m_part,
                A=A, Bp=Bp, sqarea=sqarea, kite=kite, kap0=kap0, kap1=kap1,
                sink=sink, cosk=cosk)


# ---------------------------------------------------------------------------
# the columns and their partials
# ---------------------------------------------------------------------------

def _even_terms(g):
    """Deviation columns (E0, E1, E2) of mu^0, mu^2, mu^4 and their
    (d/dz, d/dr) partials."""
    r, r2, z2, zero = g["r"], g["r2"], g["z2"], g["zero"]
    m_comp, m_part, m_none = g["m_comp"], g["m_part"], g["m_none"]
    kap0, kap1, kite, sqarea = g["kap0"], g["kap1"], g["kite"], g["sqarea"]
    sink, cosk = g["sink"], g["cosk"]
    alpha = 1.0 - r2 - z2
    beta = 2.0 * g["z"] * r

    # s0, s2 as in solution.py; s4 from the even recursion M0, M2 -> M4
    s0d_part = -(kap1 + r2 * kap0 - 0.5 * kite)
    s0d = mx.where(m_part, s0d_part, mx.where(m_comp, -_PI * r2, 0.0))
    eta2 = r2 * (r2 + 2.0 * z2)
    s2_part = 2.0 * s0d_part + 2.0 * (
        kap1 + eta2 * kap0 - 0.25 * kite * (1.0 + 5.0 * r2 + z2))
    s2d = mx.where(m_part, s2_part,
                   mx.where(m_comp, _TWO_PI * r2 * (r2 + 2.0 * z2 - 1.0), 0.0))
    M0 = mx.where(m_comp, _PI + zero, kap0)
    M2 = mx.where(m_comp, _PI * alpha, kap0 * alpha + kite)
    M4 = (6.0 * alpha * M2 + 2.0 * sqarea * M0) / 4.0
    s4 = -(2.0 * r2 * M4 - (2.0 / 3.0) * (alpha * M4 + sqarea * M2))
    s4 = mx.where(m_none, zero, s4)
    E0 = s0d
    E1 = 0.5 * s0d + 0.25 * s2d
    E2 = s0d / 3.0 + s2d / 6.0 + s4 / 6.0

    # boundary rule, I = (alpha + beta cos psi)^n on the planet arc
    sin2k = 2.0 * sink * cosk
    occ = mx.logical_not(m_none)
    k = mx.where(occ, kap0, 0.0)
    i0 = 2.0 * k                                   # int 1
    i1 = 2.0 * sink                                # int cos
    i2 = k + 0.5 * sin2k                           # int cos^2
    i3 = 2.0 * (sink - sink ** 3 / 3.0)            # int cos^3
    a2, ab, b2 = alpha * alpha, alpha * beta, beta * beta
    # d occ/dr = r int I,  d occ/dz = -r int I cos;  columns are -occ
    dE0_dr, dE0_dz = -r * i0, r * i1
    dE1_dr, dE1_dz = -r * (alpha * i0 + beta * i1), r * (alpha * i1 + beta * i2)
    dE2_dr = -r * (a2 * i0 + 2.0 * ab * i1 + b2 * i2)
    dE2_dz = r * (a2 * i1 + 2.0 * ab * i2 + b2 * i3)
    return (E0, E1, E2), ((dE0_dz, dE0_dr), (dE1_dz, dE1_dr), (dE2_dz, dE2_dr))


def _pole_terms(g, eps: float):
    """Deviation column of P_eps = (mu^2 + eps)^-2 and its partials."""
    z, r, r2, z2 = g["z"], g["r"], g["r2"], g["z2"]
    zero, one = g["zero"], g["one"]
    m_comp, m_part = g["m_comp"], g["m_part"]
    A, Bp, kite, kap0, kap1 = g["A"], g["Bp"], g["kite"], g["kap0"], g["kap1"]
    eps = float(eps)
    p = 1.0 + eps
    a_ = p - z2 - r2
    b_ = 2.0 * z * r
    K = p + r2 - z2
    Q = (eps + A) * (eps + Bp)

    # complete transit: Q > 0 there (z + r <= 1 < sqrt p)
    sqQ = mx.sqrt(mx.where(m_comp, Q, one))
    occ_c = _TWO_PI * r2 / (sqQ * (K + sqQ))
    Q32c = mx.where(m_comp, Q, one) * sqQ
    docc_c_dz = 4.0 * _PI * z * r2 / Q32c
    docc_c_dr = _TWO_PI * r * a_ / Q32c

    # partial overlap: J by regime (-Bp > 0 here). Every quantity that
    # vanishes like sqrt(lens depth) at the external contact -- sqrt(V),
    # T -- is routed through kite (kite^2 = A (-Bp) exactly), the same
    # source kap0 and kap1 come from: near the contact the depth sits at
    # fp32 resolution, and the three large terms of occ_p only cancel
    # if they see the same rounded lens. A itself enters only through
    # eps + A, which is insensitive there.
    nBp = mx.where(m_part, -Bp, one)
    apb = eps + A                                   # a_ + b_, > 0 always
    U = mx.where(m_part, apb * nBp, one)
    ratio = (eps + Bp) / nBp                        # V = ratio * kite^2
    y = ratio * kite * kite / U
    ysw = Y_SW.get(z.dtype, 1e-3)
    m_atan = mx.logical_and(m_part, y > ysw)
    m_log = mx.logical_and(m_part, y < -ysw)
    m_ser = mx.logical_and(m_part, mx.logical_not(mx.logical_or(m_atan, m_log)))

    sqU = mx.sqrt(U)
    sqQp = mx.sqrt(mx.where(m_atan, Q, one))
    sqV = kite * mx.sqrt(mx.where(m_atan, ratio, one))
    J_atan = 4.0 / sqQp * mx.arctan2(sqV, sqU)
    sqQn = mx.sqrt(mx.where(m_log, -Q, one))
    sqVn = kite * mx.sqrt(mx.where(m_log, -ratio, one))
    zre = mx.sqrt(mx.where(m_log, mx.maximum(z * r * eps, g["tiny"]), one))
    J_log = 4.0 / sqQn * mx.log((sqU + sqVn) / (2.0 * zre))
    T = kite / nBp                                  # tan(kap0 / 2)
    ys = mx.where(m_ser, y, zero)
    y2 = ys * ys
    G = 1.0 - ys / 3.0 + y2 / 5.0 - ys * y2 / 7.0 + y2 * y2 / 9.0 \
        - ys * y2 * y2 / 11.0
    Gp = (-1.0 / 3.0 + 2.0 * ys / 5.0 - 3.0 * y2 / 7.0 + 4.0 * ys * y2 / 9.0
          - 5.0 * y2 * y2 / 11.0)                         # dG/dy
    J_ser = 4.0 * T * G / apb
    J = mx.where(m_atan, J_atan, mx.where(m_log, J_log, J_ser))
    occ_p = kap1 / (p * eps) + (K * J - 2.0 * kap0) / (4.0 * p)

    # J2 = int (a_ + b_ cos)^-2, J2c = its cos moment; closed in J away
    # from Q = 0, through the series (bounded T) on the bridge
    Qs = mx.where(m_ser, one, mx.where(m_part, Q, one))
    J2_cl = (a_ * J - 2.0 * kite / eps) / Qs
    zre2 = mx.where(m_part, mx.maximum(z * r * eps, g["tiny"]), one)
    J2c_cl = (a_ * kite / zre2 - b_ * J) / Qs
    T2 = T * T
    J2_ser = 4.0 * T / (apb * apb) * (G - 2.0 * b_ * T2 * Gp / apb)
    J2c_ser = 4.0 * T / (apb * apb) * (G + 2.0 * a_ * T2 * Gp / apb)
    J2 = mx.where(m_ser, J2_ser, J2_cl)
    J2c = mx.where(m_ser, J2c_ser, J2c_cl)
    docc_p_dz = -r * J2c
    docc_p_dr = r * J2

    occ = mx.where(m_part, occ_p, mx.where(m_comp, occ_c, 0.0))
    docc_dz = mx.where(m_part, docc_p_dz, mx.where(m_comp, docc_c_dz, 0.0))
    docc_dr = mx.where(m_part, docc_p_dr, mx.where(m_comp, docc_c_dr, 0.0))
    return -occ, (-docc_dz, -docc_dr)


def even_cols(z: mx.array, r):
    """Deviation columns (E0, E1, E2) of mu^0, mu^2, mu^4: elementary, no
    elliptic integrals (E0, E1 coincide with ld_basis's B_0, B_2)."""
    return _even_terms(lens_geometry(z, r))[0]


def pole_col(z: mx.array, r, eps: float):
    """Deviation column of the pole term (mu^2 + eps)^-2."""
    return _pole_terms(lens_geometry(z, r), eps)[0]


def _generator_terms(g, law: HybridLaw):
    """Generator columns [E0, E1, E2, P_k...] and partials, one geometry."""
    (E0, E1, E2), dE = _even_terms(g)
    cols, parts = [E0, E1, E2], list(dE)
    for e in law.eps:
        c, d = _pole_terms(g, e)
        cols.append(c)
        parts.append(d)
    return cols, parts


def _combine(gen, S_row):
    acc = None
    for coef, col in zip(S_row, gen):
        if coef == 0.0:
            continue
        term = float(coef) * col
        acc = term if acc is None else acc + term
    return acc


def _shape_terms(z, r, law):
    g = lens_geometry(z, r)
    gen, parts = _generator_terms(g, law)
    S = law.generator_matrix()
    cols = [gen[0]] + [_combine(gen, S[j]) for j in range(law.n_w)]
    dz = [parts[0][0]] + [_combine([p[0] for p in parts], S[j])
                          for j in range(law.n_w)]
    dr = [parts[0][1]] + [_combine([p[1] for p in parts], S[j])
                          for j in range(law.n_w)]
    return cols, dz, dr


def shape_cols(z: mx.array, r, law) -> mx.array:
    """The shape-basis columns B = [E0, T_1..T_n], stacked on a trailing
    axis: shape z.shape + (1 + n_w,). Deviation form; 0 out of transit."""
    cols, _, _ = _shape_terms(z, r, get_law(law))
    return mx.stack(cols, axis=-1)


def shape_partials(z: mx.array, r, law):
    """(dB/dz, dB/dr), each z.shape + (1 + n_w,): the analytic partials of
    ``shape_cols`` (the reference for the kernel VJP)."""
    _, dz, dr = _shape_terms(z, r, get_law(law))
    return mx.stack(dz, axis=-1), mx.stack(dr, axis=-1)


def flux_dev_hybrid(z: mx.array, r, w, law) -> mx.array:
    """F - 1 for a hybrid law with shape weights ``w``.

    ``w`` may be a host sequence (folded into float64 constants), an
    mx.array of shape (n_w,) -- traced, so gradients flow to the weights --
    or (n_sets, n_w) batched, in which case each weight is a column that
    broadcasts against z's leading (n_sets, ...) axes, exactly as
    ``flux_dev_poly`` handles its coefficients.
    """
    law = get_law(law)
    cols, _, _ = _shape_terms(z, r, law)
    return combine_cols(cols, w, law)


def combine_cols(cols, w, law) -> mx.array:
    """F - 1 from the shape columns: (E0 - sum_j w_j T_j) / (pi - sum_j
    w_j N_j), as one sequential expression. ``cols`` is the list
    [E0, T_1..T_n] (each any shape), ``w`` as flux_dev_hybrid takes it.
    THE off-kernel expression: flux_dev_hybrid, the z-kernel entry point's
    fallback and the fp64 tau graph all call it, so every route off the
    kernels agrees bitwise."""
    law = get_law(law)
    N = law.norms()
    if isinstance(w, mx.array):
        n = w.shape[-1] if w.ndim else 1
        if n != law.n_w:
            raise ValueError(f"{law.name} takes {law.n_w} weights; got {n}")
        if w.ndim >= 2:
            ws = [w[..., j:j + 1] for j in range(law.n_w)]
        else:
            ws = [w[j] for j in range(law.n_w)]
        num = cols[0]
        norm = _PI + 0.0 * ws[0]
        for j, wj in enumerate(ws):
            num = num - wj * cols[j + 1]
            norm = norm - wj * float(N[j + 1])
        return num / norm
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if w.size != law.n_w:
        raise ValueError(f"{law.name} takes {law.n_w} weights; got {w.size}")
    norm = float(_PI - float(w @ N[1:]))
    num = cols[0]
    for j, wj in enumerate(w.tolist()):
        if wj != 0.0:
            num = num - wj * cols[j + 1]
    return num / norm
