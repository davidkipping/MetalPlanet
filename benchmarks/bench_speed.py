"""Speed benchmark runner: one subprocess per (code, mode, n, threads)
so thermal state, JIT caches, and thread pools never leak between
measurements. Writes speed.json incrementally (safe to re-run; completed
measurements are skipped).

Axes on this machine (M2 Max):
  * CPU cores: thread counts 1/4/8/12 for codes that support threading
    (batman via OpenMP if its extension was built with it — probed;
    pytransit via numba). exoplanet-core is single-threaded by design;
    jaxoplanet's XLA pool and MLX's CPU stream are all-cores and not
    cleanly partitionable (reported as threads=0 meaning "all").
  * GPU cores: not partitionable on Apple Silicon — instead the N-sweep
    shows the single-GPU saturation curve (small N under-fills the 38
    cores; large N saturates them). Cross-chip scaling (30-core Max vs
    76-core Ultra) is the only true GPU-core axis.
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "speed.json")

from scenario import N_SWEEP, THREADS  # noqa: E402

# (code, thread-counts). 0 = "all cores / not partitionable".
SINGLE = [
    ("metalplanet_fp32", [0]),          # GPU
    ("metalplanet_fp64", [0]),          # MLX CPU stream
    ("batman", [1]),                    # probed: this build lacks OpenMP
    ("pytransit", THREADS),             # numba
    ("exoplanet", [1]),                 # single-threaded C++
    ("jaxoplanet", [0]),                # XLA CPU pool (all cores)
]
BATCH = ["metalplanet_gpu", "pytransit", "jaxoplanet", "batman",
         "exoplanet"]
# eccentric sweep: same codes, each solving Kepler's equation itself, so
# the numbers include the orbit as well as the photometry.
ECC_CODES = [
    ("metalplanet_fp32", [0]),
    ("metalplanet_fp64", [0]),
    ("batman", [1]),
    ("pytransit", THREADS),
    ("exoplanet", [1]),
    ("jaxoplanet", [0]),
]

SLOW_SKIP_S = 20.0  # skip larger N once a code's median exceeds this


def load():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return []


def save(rows):
    with open(OUT, "w") as f:
        json.dump(rows, f, indent=1)


def have(rows, **kw):
    return any(all(r.get(k) == v for k, v in kw.items()) for r in rows)


def run_one(code, mode, n, threads):
    env = dict(os.environ)
    t = threads if threads > 0 else 12
    env.update(BENCH_NTHREADS=str(threads if threads > 0 else 1),
               NUMBA_NUM_THREADS=str(t), OMP_NUM_THREADS=str(t),
               VECLIB_MAXIMUM_THREADS=str(t))
    r = subprocess.run([sys.executable, os.path.join(HERE, "worker.py"),
                        code, mode, str(n)],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        return {"code": code, "mode": mode, "n": n, "threads": threads,
                "error": (r.stderr or "")[-400:]}
    rec = json.loads(r.stdout.strip().splitlines()[-1])
    rec["threads"] = threads
    return rec


def main():
    rows = load()
    for code, thread_list in SINGLE:
        for threads in thread_list:
            slow = False
            for n in N_SWEEP:
                if have(rows, code=code, mode="single", n=n,
                        threads=threads):
                    continue
                if slow:
                    break
                rec = run_one(code, "single", n, threads)
                rows.append(rec)
                save(rows)
                med = rec.get("median_s")
                print(f"{code:>18s} single n={n:>9d} threads={threads}: "
                      f"{med if med is None else f'{med*1e3:10.2f} ms'} "
                      f"{rec.get('error', '')[:60]}", flush=True)
                if rec.get("error") or (med and med > SLOW_SKIP_S):
                    slow = True

    for code, thread_list in ECC_CODES:
        for threads in thread_list:
            slow = False
            for n in N_SWEEP:
                if have(rows, code=code, mode="ecc", n=n, threads=threads):
                    continue
                if slow:
                    break
                rec = run_one(code, "ecc", n, threads)
                rows.append(rec)
                save(rows)
                med = rec.get("median_s")
                print(f"{code:>18s}    ecc n={n:>9d} threads={threads}: "
                      f"{med if med is None else f'{med*1e3:10.2f} ms'} "
                      f"{rec.get('error', '')[:60]}", flush=True)
                if rec.get("error") or (med and med > SLOW_SKIP_S):
                    slow = True

    for code in BATCH:
        if have(rows, code=code, mode="batch"):
            continue
        rec = run_one(code, "batch", 0, 0)
        rows.append(rec)
        save(rows)
        med = rec.get("median_s")
        print(f"{code:>18s} batch 512x1e5: "
              f"{med if med is None else f'{med:10.3f} s'} "
              f"{rec.get('error', '')[:60]}", flush=True)

    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
