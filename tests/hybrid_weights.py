"""Physical weight vectors for any hybrid law -- one helper for every
hybrid test file. The region depends on the law's *structure*, not its
name: a hybrid2-type law (shapes {1 - mu^2, Pi_eps}, one pole) has an exact
triangle that reaches negative weights; every other law the simplex."""

import numpy as np

from metalplanet import ld
from metalplanet.hybrid import HYBRID2, get_law


def is_hybrid2_type(law) -> bool:
    L = get_law(law)
    return len(L.eps) == 1 and tuple(map(tuple, L.shapes)) == HYBRID2.shapes


def phys_w(law, rng, n=None):
    """One weight vector (n None) or an (n, n_w) batch, uniform on the
    law's physical region."""
    L = get_law(law)
    if is_hybrid2_type(L):
        size = None if n is None else n
        w = ld.hybrid2_from_q_np(rng.random(size), rng.random(size), law=L)
        return np.stack(w, axis=-1)
    return ld.simplex_from_q_np(rng.random((L.n_w,) if n is None
                                           else (n, L.n_w)))


def vertex_weight(law) -> float:
    """The unit-weight corner on the physical region's boundary: 1 for the
    hybrid2 triangle (its V_c, V_l bracket it), just inside for the simplex
    in the tests that put one weight at a time."""
    return 1.0 if is_hybrid2_type(law) else 0.9
