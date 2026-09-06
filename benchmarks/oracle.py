"""High-precision reference flux via direct mpmath integration (30
significant digits): polar-coordinate integral of the limb-darkened
intensity over the occulted region. No elliptic integrals, no shared
lineage with any of the benchmarked codes.
"""

import json
import os

import numpy as np

_CACHE = os.path.join(os.path.dirname(__file__), "oracle_cache.json")


def flux_direct(r, z, u1, u2, dps=30):
    import mpmath as mp
    with mp.workdps(dps):
        r_, z_, u1_, u2_ = map(mp.mpf, (float(r), float(z), float(u1),
                                        float(u2)))

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

        pieces = [lo, hi]
        kink = abs(z_ - r_)
        if lo < kink < hi:
            pieces = [lo, kink, hi]
        occ = mp.quad(occ_ring, pieces)
        if z_ == 0:
            occ = mp.quad(lambda rho: 2 * mp.pi * rho * intensity(rho),
                          [0, hi])
        total = mp.pi * (1 - u1_ / 3 - u2_ / 6)
        return float(1 - occ / total)


def oracle_lightcurve(t, r, u1, u2, z_of_t, tag):
    """Cached oracle flux at times t (keyed by tag)."""
    t = np.asarray(t, dtype=np.float64)
    cache = {}
    if os.path.exists(_CACHE):
        with open(_CACHE) as f:
            cache = json.load(f)
    if tag in cache and len(cache[tag]["flux"]) == t.size:
        return np.array(cache[tag]["flux"])
    z = z_of_t(t)
    flux = np.array([flux_direct(r, zz, u1, u2) for zz in z])
    cache[tag] = {"t": t.tolist(), "flux": flux.tolist()}
    with open(_CACHE, "w") as f:
        json.dump(cache, f)
    return flux
