"""The oblate cross-code scenario: the spherical benchmark's transit
(benchmarks/scenario.py) with a flattened, tilted planet.

Limb darkening is the one profile every code here can express exactly:
I(mu) = 1 - w (1 - mu^2), w = 0.5. That is MetalPlanet's hybrid2 law with
its pole weight 0 (the oblate path takes hybrid laws only), and the
quadratic law with (u1, u2) = (2w, -w) = (1.0, -0.5) for squishyplanet,
JoJo and GreenLantern (JoJo is quadratic-only). Circular orbit,
instantaneous fluxes (no exposure integration), so every code computes the
same function.
"""

import math
import os

import numpy as np

# BENCH_SCENARIO=graze: a precision stress case -- a grazing transit of a
# strongly flattened, tilted planet (b = 1.02, f = 0.4, theta = 60 deg)
GRAZE = os.environ.get("BENCH_SCENARIO", "central") == "graze"

T0 = 0.0
PER = 3.456                       # d
A_RS = 8.8
B_IMPACT = 1.02 if GRAZE else 0.45
INC = math.acos(B_IMPACT / A_RS)  # rad
INC_DEG = math.degrees(INC)
RP = 0.1                          # area-equivalent radius sqrt(A B)
F = 0.4 if GRAZE else 0.2         # projected flattening 1 - B/A
THETA_DEG = 60.0 if GRAZE else 30.0   # sky angle of the long axis from the motion
THETA = math.radians(THETA_DEG)
W_LD = 0.5
U1, U2 = 2.0 * W_LD, -W_LD        # quadratic equivalent

A_AX = RP / math.sqrt(1.0 - F)    # semi-major axis of the projected ellipse
B_AX = A_AX * (1.0 - F)

T_HALF = 0.12                     # as the spherical benchmark (T14 ~ 0.13 d)
N_PREC = 241
N_SWEEP = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
BATCH_NPV = int(os.environ.get("BENCH_NPV", "512"))
BATCH_NPT = 10_000
GRAD_N = [10_000, 100_000]


def time_grid(n):
    return np.linspace(-T_HALF, T_HALF, int(n))


def sky(t):
    """(X, Y) in stellar radii, X along the motion, Y = -b cos(phi)
    (SquishierPlanet's / squishyplanet's frame); all front-side here."""
    phi = 2.0 * np.pi * (np.asarray(t, np.float64) - T0) / PER
    return A_RS * np.sin(phi), -B_IMPACT * np.cos(phi)


def principal(t):
    X, Y = sky(t)
    c, s = math.cos(THETA), math.sin(THETA)
    return X * c + Y * s, -X * s + Y * c


def batch_params(npv=BATCH_NPV, seed=0):
    """npv parameter draws around the scenario."""
    rng = np.random.default_rng(seed)
    return {
        "rp": RP * (1 + 0.1 * rng.standard_normal(npv)),
        "f": np.clip(F + 0.05 * rng.standard_normal(npv), 0.05, 0.4),
        "theta": THETA + 0.2 * rng.standard_normal(npv),
        "t0": 0.002 * rng.standard_normal(npv),
        "per": PER + 1e-4 * rng.standard_normal(npv),
        "a": A_RS * (1 + 0.02 * rng.standard_normal(npv)),
        "b": np.clip(B_IMPACT + 0.05 * rng.standard_normal(npv), 0.0, 0.8),
    }
