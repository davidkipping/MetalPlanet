"""flux_dev_from_tau with the hybrid laws against the quadratic law.

The hybrid laws need no elliptic integral (metalplanet/hybrid.py); the
quadratic law's mu term needs the 10-iteration cel3 recursion that
dominates its kernel. Does that show in the fused fp32 kernels? Routes,
at the production shape (512 chains x 5,000 phase-ordered points, contact
rule, n_gl = 5, Kepler long cadence):

  quadratic         the existing kernel, u1/u2
  hybrid2/4/5       the hybrid kernels, shape weights per chain
  quadratic basis   ld_basis=True, 3 columns
  hybrid5 basis     ld_basis=True, 6 columns

Each configuration runs in its own subprocess, interleaved over rounds;
reported is the median over rounds of each round's median. value+grad
evaluates value AND gradients, in (tau, period, a, b, r) and the limb
darkening where the route has it.

    python benchmarks/bench_hybrid.py [--chains 512] [--points 5000]
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

P, A, B, R = 3.4525, 8.84, 0.30, 0.1153
EXP = 29.4 / 60.0 / 24.0
ROUTES = ("quadratic", "hybrid2", "hybrid4", "hybrid5",
          "quadratic_basis", "hybrid5_basis")
WEIGHTS = {"hybrid2": [0.3, 0.2], "hybrid4": [0.2, 0.2, 0.1, 0.1],
           "hybrid5": [0.2, 0.2, 0.1, 0.1, 0.1]}


def worker(route, grad, n, m, integration, reps):
    import numpy as np
    import mlx.core as mx
    from metalplanet.metal import flux_dev_from_tau

    rng = np.random.default_rng(11)
    tau = np.linspace(-0.14, 0.14, m)[None, :] + np.zeros((n, 1))
    tau += rng.normal(0.0, 1e-4, (n, m))
    tau = mx.array(np.sort(tau, axis=1).astype(np.float32))

    def full(v):
        return mx.array(np.full(n, v, np.float32))

    geo = [tau] + [full(v) for v in (P, A, B, R)]
    basis = route.endswith("_basis")
    law = route.replace("_basis", "")
    kw = dict(exp_time=EXP, integration=integration, n_gl=5, ld_basis=basis)
    if law == "quadratic":
        ld = [] if basis else [full(0.4225), full(0.3077)]
        model = lambda *v: flux_dev_from_tau(*v, **kw)
    else:
        ld = [] if basis else [mx.array(np.tile(WEIGHTS[law], (n, 1))
                                        .astype(np.float32))]
        model = lambda *v: flux_dev_from_tau(*v[:5], limb_dark=law,
                                             u=v[5] if len(v) > 5 else None,
                                             **kw)
    args = geo + ld
    ncol = {"quadratic": 3, "hybrid5": 6}.get(law, 1)
    ct = mx.array(rng.normal(size=(n, m) + ((ncol,) if basis else ()))
                  .astype(np.float32))
    vg = mx.value_and_grad(lambda *v: mx.sum(ct * model(*v)),
                           argnums=tuple(range(len(args))))

    def once():
        if grad:
            val, gs = vg(*args)
            mx.eval(val, *gs)
        else:
            mx.eval(model(*args))

    for _ in range(3):
        once()
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        once()
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=int, default=512)
    ap.add_argument("--points", type=int, default=5000)
    ap.add_argument("--integration", default="contact")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--_worker", nargs=2, help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a._worker:
        dt = worker(a._worker[0], a._worker[1] == "1", a.chains, a.points,
                    a.integration, a.reps)
        print(json.dumps({"s": dt}))
        return

    here = os.path.abspath(__file__)
    res = {(r, g): [] for r in ROUTES for g in (False, True)}
    for _ in range(a.rounds):
        for g in (False, True):
            for r in ROUTES:
                out = subprocess.run(
                    [sys.executable, here, "--chains", str(a.chains),
                     "--points", str(a.points), "--integration",
                     a.integration, "--reps", str(a.reps),
                     "--_worker", r, "1" if g else "0"],
                    check=True, capture_output=True, text=True).stdout
                res[(r, g)].append(json.loads(out.splitlines()[-1])["s"])

    print(f"\n{a.chains} chains x {a.points} points, integration="
          f"{a.integration!r}, n_gl=5 (median of {a.rounds} isolated rounds"
          f" x {a.reps} reps)\n")
    print(f"  {'route':<18s}{'forward':>12s}{'vs quad':>9s}"
          f"{'value+grad':>14s}{'vs quad':>9s}")
    base = {g: statistics.median(res[("quadratic", g)]) for g in (False, True)}
    for r in ROUTES:
        f = statistics.median(res[(r, False)])
        v = statistics.median(res[(r, True)])
        print(f"  {r:<18s}{f * 1e3:10.2f}ms{f / base[False]:8.2f}x"
              f"{v * 1e3:12.2f}ms{v / base[True]:8.2f}x")
    print()


if __name__ == "__main__":
    main()
