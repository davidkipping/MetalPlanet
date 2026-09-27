"""Shared benchmark scenario: one standard quadratic-LD transit.

Every code computes the same physical light curve so precision and speed
are comparable. Circular orbit (all codes support it), parameters chosen
to exercise ingress/egress and limb darkening.
"""

import math

import numpy as np

# physical scenario
T0 = 0.0
PER = 3.456          # d
A_RS = 8.8           # a/R*
B_IMPACT = 0.45
INC_DEG = math.degrees(math.acos(B_IMPACT / A_RS))
RP = 0.1
U1, U2 = 0.40, 0.25

# time window: full transit (T14 ~ 0.127 d) plus out-of-transit shoulders
T_HALF = 0.12


def time_grid(n: int) -> np.ndarray:
    return np.linspace(-T_HALF, T_HALF, int(n))


def z_of_t(t: np.ndarray) -> np.ndarray:
    """Sky-projected separation for the circular scenario, float64."""
    phi = 2.0 * np.pi * (np.asarray(t, dtype=np.float64) - T0) / PER
    cosi = B_IMPACT / A_RS
    z = A_RS * np.sqrt(np.sin(phi) ** 2 + cosi ** 2 * np.cos(phi) ** 2)
    return np.where(np.cos(phi) > 0, z, 2.0 + z)


# speed sweep
N_SWEEP = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
THREADS = [1, 4, 8, 12]

# native-batch benchmark (npv overridable for the batch-scaling sweep)
import os as _os
BATCH_NPV = int(_os.environ.get("BENCH_NPV", "512"))
BATCH_NPT = 100_000


def batch_params(npv: int = BATCH_NPV, seed: int = 0):
    """npv parameter draws around the scenario (for batch benchmarks)."""
    rng = np.random.default_rng(seed)
    return {
        "rp": RP * (1 + 0.1 * rng.standard_normal(npv)),
        "t0": 0.002 * rng.standard_normal(npv),
        "per": PER + 1e-4 * rng.standard_normal(npv),
        "a": A_RS * (1 + 0.02 * rng.standard_normal(npv)),
        "b": np.clip(B_IMPACT + 0.05 * rng.standard_normal(npv), 0, 0.9),
        "u1": np.clip(U1 + 0.05 * rng.standard_normal(npv), 0, 1),
        "u2": np.clip(U2 + 0.05 * rng.standard_normal(npv), -0.2, 0.6),
    }


# ---------------------------------------------------------------------------
# eccentric scenario (same star/planet, e != 0)
#
# Every comparison code solves Kepler's equation itself, so the eccentric
# sweep measures the orbit solver as well as the photometry. The impact
# parameter is held at B_IMPACT *at inferior conjunction*, which for an
# eccentric orbit means cos i = b (1 + e sin w) / (a (1 - e^2)) — matching
# b to the circular case keeps the transit shape comparable.
# ---------------------------------------------------------------------------

ECC = 0.3
OMEGA_DEG = 63.0
_W = math.radians(OMEGA_DEG)
COSI_ECC = B_IMPACT * (1.0 + ECC * math.sin(_W)) / (A_RS * (1.0 - ECC ** 2))
INC_ECC_DEG = math.degrees(math.acos(COSI_ECC))

# the eccentric transit is shorter here (the planet is near periastron),
# so the window is scaled by the instantaneous sky speed ratio
_SPEED = (1.0 + ECC * math.sin(_W)) / math.sqrt(1.0 - ECC ** 2)
T_HALF_ECC = T_HALF / _SPEED


def time_grid_ecc(n: int) -> np.ndarray:
    return np.linspace(-T_HALF_ECC, T_HALF_ECC, int(n))


def mean_anomaly_at_transit(e: float = ECC, w: float = _W) -> float:
    """M at inferior conjunction (true anomaly f = pi/2 - w)."""
    f0 = 0.5 * math.pi - w
    E0 = 2.0 * math.atan2(math.sqrt(1.0 - e) * math.sin(0.5 * f0),
                          math.sqrt(1.0 + e) * math.cos(0.5 * f0))
    return E0 - e * math.sin(E0)


def _solve_E(M: np.ndarray, e: float) -> np.ndarray:
    """Kepler by Newton to float64 convergence — an INDEPENDENT solver,
    so the oracle never leans on the code under test."""
    M = np.asarray(M, dtype=np.float64)
    Mw = np.mod(M + math.pi, 2.0 * math.pi) - math.pi
    E = Mw + e * np.sin(Mw)
    for _ in range(60):
        f = E - e * np.sin(E) - Mw
        E = E - f / (1.0 - e * np.cos(E))
        if np.max(np.abs(f)) < 1e-15:
            break
    return E


def z_of_t_ecc(t: np.ndarray) -> np.ndarray:
    """Sky-projected separation for the eccentric scenario, float64.

    Returns the same 'far side pushed beyond contact' convention as
    z_of_t so a photometric core can consume it directly.
    """
    M = 2.0 * math.pi * (np.asarray(t, dtype=np.float64) - T0) / PER \
        + mean_anomaly_at_transit()
    E = _solve_E(M, ECC)
    f = 2.0 * np.arctan2(math.sqrt(1.0 + ECC) * np.sin(0.5 * E),
                         math.sqrt(1.0 - ECC) * np.cos(0.5 * E))
    r_orb = A_RS * (1.0 - ECC ** 2) / (1.0 + ECC * np.cos(f))
    swf = np.sin(_W + f)
    z = r_orb * np.sqrt(np.maximum(
        1.0 - swf ** 2 * (1.0 - COSI_ECC ** 2), 0.0))
    return np.where(swf > 0.0, z, 2.0 + z)
