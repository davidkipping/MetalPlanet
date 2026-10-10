"""Physical weight vectors for any hybrid law -- one helper for every
hybrid test file. The region follows the law's structure
(HybridLaw.is_hybrid2_type, the rule the priors use): an exact triangle,
reaching negative weights, for a hybrid2-type law; the simplex otherwise."""

import numpy as np

from metalplanet import ld
from metalplanet.hybrid import get_law


def phys_w(law, rng, n=None):
    """One weight vector (n None) or an (n, n_w) batch, uniform on the
    law's physical region."""
    L = get_law(law)
    if L.is_hybrid2_type:
        w = ld.hybrid2_from_q_np(rng.random(n), rng.random(n), law=L)
        return np.stack(w, axis=-1)
    return ld.simplex_from_q_np(rng.random((L.n_w,) if n is None
                                           else (n, L.n_w)))


def vertex_weight(law) -> float:
    """The unit-weight corner: 1 on the hybrid2 triangle (its V_c, V_l
    bracket it), just inside for the simplex, in the tests that put one
    weight at a time."""
    return 1.0 if get_law(law).is_hybrid2_type else 0.9
