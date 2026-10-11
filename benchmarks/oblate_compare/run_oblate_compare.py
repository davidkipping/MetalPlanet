"""Oblate-planet cross-code benchmark: precision and speed.

    benchmarks/oblate_compare/.venv/bin/python run_oblate_compare.py [--quick]

Precision: every code on the scenario's 241-point grid against a 30-digit
mpmath polar-ray integral (SquishierPlanet's oracle). Speed: each
(code, mode, N) in its own subprocess (worker_oblate.py). Writes
results_oblate.json; make_report_oblate.py renders RESULTS.md.
Run on a quiet machine.
"""
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "SquishierPlanet"))

import numpy as np  # noqa: E402

import adapters_oblate as A  # noqa: E402
from scenario_oblate import (A_AX, B_AX, BATCH_NPT, BATCH_NPV, GRAD_N,  # noqa: E402
                             N_PREC, N_SWEEP, W_LD, principal, time_grid)


def oracle_flux():
    from squishierplanet.oracle import occulted_flux
    w = W_LD

    def prim(rr):
        return (1 - w) * (-(1 - rr * rr) / 2) + w * (-(1 - rr * rr) ** 2 / 4)
    total = np.pi * (1 - w / 2)
    x0, y0 = principal(time_grid(N_PREC))
    occ = np.array([occulted_flux(x, y, A_AX, B_AX, prim, use_mpmath=True, dps=30)
                    for x, y in zip(x0, y0)])
    return 1.0 - occ / total


def precision():
    ref = oracle_flux()
    out = {}
    for code in A.CODES:
        f = getattr(A, f"{code}_run")(getattr(A, f"{code}_prepare")(N_PREC))
        err = np.abs(np.asarray(f, np.float64) - ref)
        out[code] = {"max": float(err.max()), "median": float(np.median(err))}
        print(f"  precision {code:18s} max {err.max():.2e} median {np.median(err):.2e}")
    return out


def measure(code, mode, n):
    r = subprocess.run([sys.executable, os.path.join(HERE, "worker_oblate.py"),
                        code, mode, str(n)], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
        return None
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    if "--graze" in sys.argv:
        # precision only, the stress scenario (scenario_oblate.GRAZE)
        res = {"precision_graze": precision()}
        path = os.path.join(HERE, "results_oblate.json")
        old = json.load(open(path)) if os.path.exists(path) else {}
        old.update(res)
        with open(path, "w") as f:
            json.dump(old, f, indent=1)
        return
    quick = "--quick" in sys.argv
    res = {"machine": platform.platform(), "python": platform.python_version(),
           "started": time.strftime("%Y-%m-%d %H:%M"),
           "load": os.getloadavg()}
    print("precision vs the 30-digit oracle:")
    res["precision"] = precision()
    sweep = N_SWEEP[:3] if quick else N_SWEEP
    res["single"] = {}
    for code in A.CODES:
        res["single"][code] = {}
        for n in sweep:
            m = measure(code, "single", n)
            if m is None:
                break
            res["single"][code][str(n)] = m["median_s"]
            print(f"  single {code:18s} N={n:>10,d}: {m['median_s'] * 1e3:10.2f} ms")
            if m["median_s"] > 8.0:          # the next decade would take minutes
                break
    res["batch"] = {}
    for code in A.BATCH_CODES:
        m = measure(code, "batch", 0)
        if m is not None:
            res["batch"][code] = m["median_s"]
            print(f"  batch {code:18s} {BATCH_NPV} x {BATCH_NPT}: {m['median_s']:.3f} s")
    res["grad"] = {}
    for code in A.GRAD_CODES:
        res["grad"][code] = {}
        for n in GRAD_N:
            m = measure(code, "grad", n)
            if m is not None:
                res["grad"][code][str(n)] = m["median_s"]
                print(f"  value+grad {code:18s} N={n:>8,d}: {m['median_s'] * 1e3:.2f} ms")
    res["load_after"] = os.getloadavg()
    with open(os.path.join(HERE, "results_oblate.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
