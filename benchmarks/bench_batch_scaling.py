"""Batch-size scaling: MetalPlanet GPU vs PyTransit (its strongest CPU
rival) at npv x 100k points — the GPU's core-utilization axis."""
import json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for npv in (64, 256, 512, 1024, 2048, 4096):
    for code in ("metalplanet_gpu", "pytransit"):
        env = dict(os.environ, BENCH_NPV=str(npv), BENCH_NTHREADS="1",
                   NUMBA_NUM_THREADS="12", OMP_NUM_THREADS="12")
        r = subprocess.run([sys.executable, os.path.join(HERE, "worker.py"),
                            code, "batch", "0"], env=env,
                           capture_output=True, text=True)
        if r.returncode != 0:
            rows.append({"code": code, "npv": npv,
                         "error": (r.stderr or "")[-200:]})
            print(f"{code} npv={npv}: ERROR", flush=True)
            continue
        rec = json.loads(r.stdout.strip().splitlines()[-1])
        rec["npv"] = npv
        rows.append(rec)
        print(f"{code:>16s} npv={npv:>5d}: {rec['median_s']:8.3f} s "
              f"({npv/rec['median_s']:8.0f} curves/s)", flush=True)
with open(os.path.join(HERE, "batch_scaling.json"), "w") as f:
    json.dump(rows, f, indent=1)
