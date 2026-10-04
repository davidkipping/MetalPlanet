"""flux_dev_from_tau on an eccentric orbit against the circular one.

What does eccentricity cost on the tau kernel? Four routes, all at the
production shape (512 chains x 5,000 phase-ordered points, contact rule,
n_gl = 5, Kepler long cadence):

  circular      secosw/sesinw omitted -- the circular plug-in
  ecc, e = 0    the eccentric plug-in fed k = h = 0: its per-chain
                e == 0 fast path skips the Kepler solve
  ecc, e = 0.3  the eccentric plug-in on a genuinely eccentric orbit
  ecc basis     the same, with ld_basis=True

Each configuration runs in its own subprocess, interleaved over rounds;
reported is the median over rounds of each round's median. value+grad
evaluates value AND gradients, in every differentiable input.

    python benchmarks/bench_tau_ecc.py [--chains 512] [--points 5000]
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time

P, A, B, R, U1, U2 = 3.4525, 8.84, 0.30, 0.1153, 0.4225, 0.3077
EXP = 29.4 / 60.0 / 24.0
ROUTES = ("circular", "ecc_e0", "ecc_e0.3", "ecc_basis")


def worker(route, grad, n, m, reps):
    import numpy as np
    import mlx.core as mx
    from metalplanet.metal import flux_dev_from_tau

    rng = np.random.default_rng(11)
    tau = np.linspace(-0.14, 0.14, m)[None, :] + np.zeros((n, 1))
    tau += rng.normal(0.0, 1e-4, (n, m))
    tau = mx.array(np.sort(tau, axis=1).astype(np.float32))

    def full(v):
        return mx.array(np.full(n, v, np.float32))

    e = 0.0 if route in ("circular", "ecc_e0") else 0.3
    w = math.radians(63.0)
    args = [tau] + [full(v) for v in (P, A, B, R)]
    basis = route == "ecc_basis"
    if not basis:
        args += [full(U1), full(U2)]
    ecc = route != "circular"
    if ecc:
        args += [full(math.sqrt(e) * math.cos(w)),
                 full(math.sqrt(e) * math.sin(w))]
    kw = dict(exp_time=EXP, integration="contact", n_gl=5, ld_basis=basis)
    ct = mx.array(rng.normal(size=(n, m) + ((3,) if basis else ()))
                  .astype(np.float32))

    def model(*v):
        if ecc:
            return flux_dev_from_tau(*v[:-2], secosw=v[-2], sesinw=v[-1],
                                     **kw)
        return flux_dev_from_tau(*v, **kw)

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
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--_worker", nargs=2, help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a._worker:
        dt = worker(a._worker[0], a._worker[1] == "1", a.chains, a.points,
                    a.reps)
        print(json.dumps({"s": dt}))
        return

    here = os.path.abspath(__file__)
    res = {(r, g): [] for r in ROUTES for g in (False, True)}
    for _ in range(a.rounds):
        for g in (False, True):
            for r in ROUTES:
                out = subprocess.run(
                    [sys.executable, here, "--chains", str(a.chains),
                     "--points", str(a.points), "--reps", str(a.reps),
                     "--_worker", r, "1" if g else "0"],
                    check=True, capture_output=True, text=True).stdout
                res[(r, g)].append(json.loads(out.splitlines()[-1])["s"])

    print(f"\n{a.chains} chains x {a.points} points, contact rule, n_gl=5 "
          f"(median of {a.rounds} isolated rounds x {a.reps} reps)\n")
    print(f"  {'':<12s}" + "".join(f"{r:>16s}" for r in ROUTES))
    for g in (False, True):
        t = {r: statistics.median(res[(r, g)]) * 1e3 for r in ROUTES}
        base = t["circular"]
        print(f"  {'value+grad' if g else 'forward':<12s}" + "".join(
            f"{t[r]:8.2f}ms {t[r] / base:5.2f}x" for r in ROUTES))
    print()


if __name__ == "__main__":
    main()
