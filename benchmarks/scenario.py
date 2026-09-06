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
