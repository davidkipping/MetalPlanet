"""flux_dev_from_tau(..., ld_basis=True) against the scalar call.

The question SquishierPlanet asked: what does the limb-darkening basis
cost relative to one scalar call -- t(ld_basis=True) / t(scalar) -- for
the forward and for value+grad? Their working assumption was <= 1.3x.
Alongside it, the route the basis replaces: three scalar calls, one per
vertex law u = (0,0), (2,-1), (0,1).

Shape: 512 chains x 5,000 phase-ordered points, contact rule, n_gl = 5,
Kepler long cadence -- the collapsed target's production configuration.

Every configuration runs in its own subprocess (a memory-heavy run
poisons later timings through thermal and buffer-cache state), and the
configurations are interleaved over several rounds; the reported time is
the median over rounds of each round's median.

value+grad evaluates the value AND the gradients (the custom VJPs
recompute from the primals, so evaluating only the gradients would skip
the forward). Gradients are taken in (tau, period, a, b, r) for every
route, so each does the same parameter work.

    python benchmarks/bench_ld_basis.py [--chains 512] [--points 5000]
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
VERTICES = ((0.0, 0.0), (2.0, -1.0), (0.0, 1.0))
ROUTES = ("scalar", "basis", "three_scalar")


def worker(route, grad, n, m, integration, reps):
    import numpy as np
    import mlx.core as mx
    from metalplanet.metal import flux_dev_from_tau

    rng = np.random.default_rng(11)
    # phase-ordered: each chain's points ascend in tau, as turin feeds them
    tau = np.linspace(-0.14, 0.14, m)[None, :] + np.zeros((n, 1))
    tau += rng.normal(0.0, 1e-4, (n, m))
    tau = mx.array(np.sort(tau, axis=1).astype(np.float32))
    geo = [mx.array(np.full(n, v, np.float32)) for v in (P, A, B, R)]
    kw = dict(exp_time=EXP, integration=integration, n_gl=5)
    lds = [(mx.array(np.full(n, u1, np.float32)),
            mx.array(np.full(n, u2, np.float32))) for u1, u2 in VERTICES]
    ct3 = mx.array(rng.normal(size=(n, m, 3)).astype(np.float32))

    def model(*g):
        if route == "basis":
            return [flux_dev_from_tau(*g, ld_basis=True, **kw)]
        laws = lds[:1] if route == "scalar" else lds
        return [flux_dev_from_tau(*g, u1, u2, **kw) for u1, u2 in laws]

    def loss(*g):
        outs = model(*g)
        if route == "basis":
            return mx.sum(ct3 * outs[0])
        return sum(mx.sum(ct3[..., j] * o) for j, o in enumerate(outs))

    vg = mx.value_and_grad(loss, argnums=(0, 1, 2, 3, 4))

    def once():
        if grad:
            v, gs = vg(tau, *geo)
            mx.eval(v, *gs)
        else:
            mx.eval(*model(tau, *geo))

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
        route, grad = a._worker[0], a._worker[1] == "1"
        dt = worker(route, grad, a.chains, a.points, a.integration, a.reps)
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

    print(f"\n{a.chains} chains x {a.points} points, "
          f"integration={a.integration!r}, n_gl=5 "
          f"(median of {a.rounds} isolated rounds x {a.reps} reps)\n")
    print(f"  {'':<16s} {'scalar':>10s} {'ld_basis':>10s} {'3 x scalar':>11s}"
          f" {'basis/scalar':>13s} {'3x/basis':>9s}")
    for g in (False, True):
        t = {r: statistics.median(res[(r, g)]) * 1e3 for r in ROUTES}
        print(f"  {'value+grad' if g else 'forward':<16s} "
              f"{t['scalar']:8.2f}ms {t['basis']:8.2f}ms "
              f"{t['three_scalar']:9.2f}ms "
              f"{t['basis'] / t['scalar']:12.2f}x "
              f"{t['three_scalar'] / t['basis']:8.2f}x")
    print()


if __name__ == "__main__":
    main()
