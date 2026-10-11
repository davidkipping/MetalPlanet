"""One oblate-benchmark measurement in an isolated process.

Usage: python worker_oblate.py <code> <mode> <n>   (mode: single | batch | grad)
Emits one JSON line: {"code", "mode", "n", "median_s", "min_s", "reps"}.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import adapters_oblate as A  # noqa: E402


def main():
    code, mode, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    if mode == "single":
        state, run = getattr(A, f"{code}_prepare")(n), getattr(A, f"{code}_run")
    elif mode == "batch":
        state, run = getattr(A, f"{code}_batch_prepare")(), getattr(A, f"{code}_batch_run")
    elif mode == "grad":
        state, run = getattr(A, f"{code}_grad_prepare")(n), getattr(A, f"{code}_grad_run")
    else:
        raise SystemExit(f"unknown mode {mode}")
    ts, total = [], 0.0
    # at least 3 repetitions, stop past 2 s or 15 reps (or one slow call)
    while len(ts) < 15 and (len(ts) < 3 or total < 2.0):
        t0 = time.perf_counter()
        run(state)
        dt = time.perf_counter() - t0
        ts.append(dt)
        total += dt
        if dt > 30.0:
            break
    print(json.dumps({"code": code, "mode": mode, "n": n,
                      "median_s": float(np.median(ts)),
                      "min_s": float(np.min(ts)), "reps": len(ts)}))


if __name__ == "__main__":
    main()
