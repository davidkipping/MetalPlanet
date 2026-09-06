"""M5 benchmark: analytic VJP vs reverse-mode autodiff for the gradient
of the chunked log-likelihood — the operation ChEES-HMC pays for at
every leapfrog step.

Each (path x measurement) runs in a FRESH SUBPROCESS: sustained GPU work
perturbs later measurements through thermal state and the MLX buffer
cache (a first, naive in-process version mis-read a 10x forward
"slowdown" for the custom-function path that was pure ordering artifact
— in isolation the custom forward fuses identically).

Setup mirrors production: 1024 chains x 65536-point chunk, fp32, GPU,
compiled. Reports medians of 7 in-process reps.
"""

import json
import os
import subprocess
import sys

CFGS = [
    ("autodiff", "forward"),
    ("analytic", "forward"),
    ("metal", "forward"),
    ("autodiff", "value_grad"),
    ("analytic", "value_grad"),
    ("metal", "value_grad"),
]

WORKER = r"""
import json, os, time
import mlx.core as mx
import numpy as np
from metalplanet.anvil import make_quad_transit_flux

mode = os.environ["BENCH_MODE"]
meas = os.environ["BENCH_MEAS"]
N_CHAINS, CHUNK, PERIOD_REF = 1024, 65536, 3.4565

rng = np.random.default_rng(0)
v0 = np.tile([0.009, -0.0005, 0.10, 0.30, 8.80, 0.4225, 0.3077, 0.0],
             (N_CHAINS, 1)).astype(np.float32)
v0 += 1e-3 * rng.standard_normal(v0.shape).astype(np.float32)
v = mx.array(v0)
dt = rng.uniform(-PERIOD_REF / 2, PERIOD_REF / 2, CHUNK)
k = rng.integers(0, 26, CHUNK).astype(np.float64)
x = mx.array(np.stack([dt, k]).astype(np.float32))
y = mx.array(rng.normal(0.0, 5e-4, CHUNK).astype(np.float32))
w = mx.array(np.full(CHUNK, 1.0 / 5e-4, np.float32))

model = make_quad_transit_flux(
    PERIOD_REF, core={"autodiff": "autodiff", "analytic": "analytic",
                      "metal": "metal"}[mode])

def loglike(v_):
    r_ = (y - model(v_, x)) * w
    return mx.sum(-0.5 * r_ * r_, axis=-1)

if meas == "forward":
    fn = mx.compile(loglike)
else:
    def value_grad(v_):
        val, vjps = mx.vjp(loglike, [v_], [mx.ones((N_CHAINS,))])
        return val[0], vjps[0]
    fn = mx.compile(value_grad)

for _ in range(3):
    mx.eval(fn(v))
ts = []
for _ in range(7):
    t0 = time.perf_counter()
    mx.eval(fn(v))
    ts.append(time.perf_counter() - t0)
print(json.dumps({"mode": mode, "meas": meas,
                  "median_ms": float(np.median(ts) * 1e3)}))
"""

results = {}
for mode, meas in CFGS:
    env = dict(os.environ, BENCH_MODE=mode, BENCH_MEAS=meas)
    out = subprocess.run([sys.executable, "-c", WORKER], env=env,
                         capture_output=True, text=True, check=True)
    rec = json.loads(out.stdout.strip().splitlines()[-1])
    results[(mode, meas)] = rec["median_ms"]
    print(f"{mode:>9s} {meas:>10s}: {rec['median_ms']:9.2f} ms", flush=True)

fa = results[("autodiff", "forward")]
fv = results[("analytic", "forward")]
fm = results[("metal", "forward")]
ga = results[("autodiff", "value_grad")]
gv = results[("analytic", "value_grad")]
gm = results[("metal", "value_grad")]
print(f"\nforward: autodiff {fa:.1f} / analytic {fv:.1f} / metal {fm:.1f} ms "
      f"(metal speedup {fv / fm:.2f}x over graph)")
print(f"value+grad: autodiff {ga:.0f} / analytic {gv:.0f} / metal {gm:.1f} ms "
      f"(metal {gv / gm:.2f}x over analytic, {ga / gm:.1f}x over autodiff)")
