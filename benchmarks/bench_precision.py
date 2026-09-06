"""Precision benchmark: every code vs the mpmath direct-integration
oracle on the standard scenario (241 points across the transit,
including ingress/egress). Prints a table and writes precision.json.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import adapters
from oracle import oracle_lightcurve
from scenario import RP, U1, U2, time_grid, z_of_t

N = 241
t = time_grid(N)
print(f"computing/loading oracle at {N} points (mpmath, 30 digits)...",
      flush=True)
ref = oracle_lightcurve(t, RP, U1, U2, z_of_t, tag=f"std-{N}")

rows = []


def check(name, flux):
    err = np.abs(np.asarray(flux, dtype=np.float64) - ref)
    rows.append({"code": name, "max_err": float(err.max()),
                 "median_err": float(np.median(err))})
    print(f"{name:>22s}: max {err.max():.3e}  median {np.median(err):.3e}",
          flush=True)


check("metalplanet fp64", adapters.metalplanet_fp64_run(
    adapters.metalplanet_fp64_prepare(N)))
check("metalplanet fp32(GPU)", adapters.metalplanet_fp32_run(
    adapters.metalplanet_fp32_prepare(N)))
check("batman", adapters.batman_run(adapters.batman_prepare(N)))
check("pytransit (exact)", adapters.pytransit_run(
    adapters.pytransit_prepare(N, interpolate=False)))
check("pytransit (interp)", adapters.pytransit_run(
    adapters.pytransit_prepare(N, interpolate=True)))
check("exoplanet-core", adapters.exoplanet_run(
    adapters.exoplanet_prepare(N)))
check("jaxoplanet (order=10)", adapters.jaxoplanet_run(
    adapters.jaxoplanet_prepare(N)))
check("jaxoplanet (order=50)", adapters.jaxoplanet_run(
    adapters.jaxoplanet_prepare(N, order=50)))

out = os.path.join(os.path.dirname(__file__), "precision.json")
with open(out, "w") as f:
    json.dump(rows, f, indent=1)
print(f"wrote {out}")
