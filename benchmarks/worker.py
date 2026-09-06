"""One benchmark measurement in an isolated process.

Usage: python worker.py <code> <mode> <n> [reps]
Emits one JSON line: {"code", "mode", "n", "median_s", "reps", ...}.

Thread environment (NUMBA_NUM_THREADS / OMP_NUM_THREADS / batman
nthreads) must be set by the parent BEFORE spawning; numba and OpenMP
read them at import/first-call time.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import adapters  # noqa: E402


def main():
    code, mode, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    max_reps = int(sys.argv[4]) if len(sys.argv) > 4 else 15
    nthreads = int(os.environ.get("BENCH_NTHREADS", "1"))

    if mode == "single":
        if code == "batman":
            state = adapters.batman_prepare(n, nthreads=nthreads)
            run = adapters.batman_run
        else:
            state = getattr(adapters, f"{code}_prepare")(n)
            run = getattr(adapters, f"{code}_run")
    elif mode == "batch":
        state = getattr(adapters, f"{code}_batch_prepare")()
        run = getattr(adapters, f"{code}_batch_run")
    else:
        raise SystemExit(f"unknown mode {mode}")

    # adaptive repetitions: at least 3, stop when total > 1.5 s or max_reps
    ts = []
    total = 0.0
    while len(ts) < max_reps and (len(ts) < 3 or total < 1.5):
        t0 = time.perf_counter()
        run(state)
        dt = time.perf_counter() - t0
        ts.append(dt)
        total += dt
        if dt > 30.0:
            break
    print(json.dumps({
        "code": code, "mode": mode, "n": n, "threads": nthreads,
        "median_s": float(np.median(ts)), "min_s": float(np.min(ts)),
        "reps": len(ts),
    }))


if __name__ == "__main__":
    main()
