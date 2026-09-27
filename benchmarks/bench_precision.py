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
from scenario import (ECC, OMEGA_DEG, RP, U1, U2, time_grid,
                      time_grid_ecc, z_of_t, z_of_t_ecc)

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

# ---------------------------------------------------------------------------
# eccentric scenario: the same comparison with every code solving Kepler's
# equation itself. The oracle's separation comes from an INDEPENDENT
# float64 Newton solver in scenario.py, so no code under test defines the
# geometry it is being judged against.
# ---------------------------------------------------------------------------

t_ecc = time_grid_ecc(N)
print(f"\ncomputing/loading eccentric oracle (e={ECC}, w={OMEGA_DEG} deg)...",
      flush=True)
ref = oracle_lightcurve(t_ecc, RP, U1, U2, z_of_t_ecc, tag=f"ecc-{N}")
ecc_rows = []


def check_ecc(name, flux):
    err = np.abs(np.asarray(flux, dtype=np.float64) - ref)
    ecc_rows.append({"code": name, "max_err": float(err.max()),
                     "median_err": float(np.median(err))})
    print(f"{name:>22s}: max {err.max():.3e}  median {np.median(err):.3e}",
          flush=True)


check_ecc("metalplanet fp64", adapters.metalplanet_fp64_ecc_run(
    adapters.metalplanet_fp64_ecc_prepare(N)))
check_ecc("metalplanet fp32(GPU)", adapters.metalplanet_fp32_ecc_run(
    adapters.metalplanet_fp32_ecc_prepare(N)))
check_ecc("batman", adapters.batman_ecc_run(
    adapters.batman_ecc_prepare(N)))
check_ecc("pytransit (exact)", adapters.pytransit_ecc_run(
    adapters.pytransit_ecc_prepare(N, interpolate=False)))
check_ecc("exoplanet-core", adapters.exoplanet_ecc_run(
    adapters.exoplanet_ecc_prepare(N)))
check_ecc("jaxoplanet (order=10)", adapters.jaxoplanet_ecc_run(
    adapters.jaxoplanet_ecc_prepare(N)))

out = os.path.join(os.path.dirname(__file__), "precision.json")
with open(out, "w") as f:
    json.dump(rows, f, indent=1)
out_e = os.path.join(os.path.dirname(__file__), "precision_ecc.json")
with open(out_e, "w") as f:
    json.dump(ecc_rows, f, indent=1)
print(f"wrote {out} and {out_e}")
