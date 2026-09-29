"""flux_dev_from_tau against the route it replaces.

The route turin uses today builds the sub-exposure axis as an MLX array:

    tau  (n, m * n_sub)  ->  separation_circular  ->  flux_dev_metal
                         ->  reshape + mean over the sub-exposure axis

Every forward intermediate and every gradient grid is then n_sub times
larger than the light curve. flux_dev_from_tau runs the same rule in
registers, so only the (n, m) result and, in the backward pass, the (n, m)
tau gradient ever exist.

Reports wall time and MLX peak memory for the forward and for value+grad.
value+grad evaluates BOTH -- MLX is lazy and the custom VJP recomputes
from the primals, so evaluating only the gradients silently skips the
forward and flatters the measurement.

    python benchmarks/bench_tau_kernel.py [--chains 512] [--points 5000]
"""

import argparse
import time

import numpy as np
import mlx.core as mx

from metalplanet.metal import flux_dev_from_tau, flux_dev_metal
from metalplanet.orbit import separation_circular

P, A, B, R, U1, U2 = 3.4525, 8.84, 0.30, 0.1153, 0.4225, 0.3077
EXP = 29.4 / 60.0 / 24.0          # Kepler long cadence


def make_inputs(n, m):
    rng = np.random.default_rng(11)
    tau = np.linspace(-0.14, 0.14, m)[None, :] + np.zeros((n, 1))
    tau += rng.normal(0.0, 1e-4, (n, m))
    pars = [mx.array(np.full(n, v, np.float32)) for v in (P, A, B, R, U1, U2)]
    return mx.array(tau.astype(np.float32)), pars


def route_expanded(tau, pars, n_sub):
    """What turin does today: expand the sub-exposure axis into MLX."""
    per, a, b, r, u1, u2 = [p[:, None] for p in pars]
    off = mx.array(np.linspace(-0.5 * EXP, 0.5 * EXP, n_sub, dtype=np.float32))
    n, m = tau.shape
    tt = mx.reshape(tau[:, :, None] + off, (n, m * n_sub))
    dev = flux_dev_metal(separation_circular(tt, per, b, a), r, u1, u2)
    return mx.mean(mx.reshape(dev, (n, m, n_sub)), axis=-1)


def route_kernel(tau, pars, n_sub, integration, n_gl=5):
    return flux_dev_from_tau(tau, *pars, exp_time=EXP,
                             integration=integration, n_gl=n_gl,
                             n_sub=n_sub)


def timeit(fn, reps, grad):
    """(seconds per call, peak MLX memory in MB)."""
    if grad:
        def once():
            v, g = mx.value_and_grad(
                lambda *a: mx.sum(fn(*a) ** 2), argnums=(0, 1))(*fn.args)
            mx.eval(v, *g)          # both: the VJP recomputes the forward
    else:
        def once():
            mx.eval(fn(*fn.args))

    once()
    mx.synchronize()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    for _ in range(reps):
        once()
    mx.synchronize()
    dt = (time.perf_counter() - t0) / reps
    return dt, mx.get_peak_memory() / 2 ** 20


class Bound:
    def __init__(self, fn, tau, pars, **kw):
        self.fn, self.kw = fn, kw
        self.args = (tau, pars)

    def __call__(self, tau, pars):
        return self.fn(tau, pars, **self.kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=int, default=512)
    ap.add_argument("--points", type=int, default=5000)
    ap.add_argument("--nsub", type=int, default=15)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    n, m, ns = args.chains, args.points, args.nsub

    tau, pars = make_inputs(n, m)
    rows = [
        ("expanded axis (today)",
         Bound(route_expanded, tau, pars, n_sub=ns)),
        (f"kernel, supersample n={ns}",
         Bound(route_kernel, tau, pars, n_sub=ns, integration="supersample")),
        ("kernel, contact n_gl=5",
         Bound(route_kernel, tau, pars, n_sub=1, integration="contact",
               n_gl=5)),
    ]

    print(f"\n{n} chains x {m} points x {ns} sub-exposures "
          f"= {n * m * ns / 1e6:.1f} M model evaluations\n")
    base = {}
    for grad in (False, True):
        what = "value+grad" if grad else "forward"
        print(f"  {what:<28s} {'ms':>9s} {'Mpts/s':>10s} "
              f"{'peak MB':>9s} {'speedup':>9s} {'memory':>9s}")
        for name, fn in rows:
            dt, mb = timeit(fn, args.reps, grad)
            if name.startswith("expanded"):
                base[what] = (dt, mb)
            s = base[what][0] / dt
            mem = base[what][1] / mb
            print(f"  {name:<28s} {dt * 1e3:9.2f} "
                  f"{n * m * ns / dt / 1e6:10.1f} {mb:9.1f} "
                  f"{s:8.2f}x {mem:8.2f}x")
        print()

    # Speed is only comparable at equal accuracy, and the reference has to
    # be neither of the rules being compared -- scoring supersampling
    # against a supersampled reference flatters it as n approaches the
    # reference's own n. So: the fp64 graph path, contact rule, n_gl = 12.
    one_tau, one_par = tau[:1], [p[:1] for p in pars]

    with mx.stream(mx.cpu):
        ref_arr = flux_dev_from_tau(one_tau.astype(mx.float64),
                                    *[p.astype(mx.float64) for p in one_par],
                                    exp_time=EXP, integration="contact",
                                    n_gl=12)
        mx.eval(ref_arr)
    ref = np.asarray(ref_arr, dtype=np.float64)

    def err_of(**kw):
        got = route_kernel(one_tau, one_par, **kw)
        mx.eval(got)
        return np.abs(np.asarray(got, dtype=np.float64) - ref).max()

    e_super = err_of(n_sub=ns, integration="supersample")
    e_cont = err_of(n_sub=1, integration="contact", n_gl=5)
    print("  accuracy vs an fp64 contact-rule reference (n_gl = 12):")
    print(f"    supersample n={ns:<4d}({ns:4d} evals)  {e_super:.2e}")
    print(f"    contact n_gl=5    (  25 evals)  {e_cont:.2e}")

    # Match a stated target rather than the contact rule's own error: at
    # 25 evaluations in fp32 that rule is already at the arithmetic's
    # noise floor, so "match it" would be asking supersampling to reach
    # fp32 rounding. 1e-6 in relative flux is an order of magnitude below
    # the best per-point Kepler precision and two below a typical one.
    target = 1e-6
    n_match, cap = ns, 4000
    while n_match < cap and err_of(n_sub=n_match,
                                   integration="supersample") > target:
        n_match = int(n_match * 1.25) + 1
    print(f"\n  supersampling reaches {target:.0e} at n_sub = {n_match} "
          f"(it converges as 1/N on a kinked curve, not 1/N^2).")

    # At 512 chains that route wants tens of GB and starts paging, which
    # would report a speedup measuring the pager rather than the kernel.
    # Size the matched comparison so BOTH fit, and state the footprint the
    # full-width version would have asked for.
    budget = 6e9
    n_small = max(1, min(n, int(budget / (m * n_match * 4 * 17))))
    proj = n * m * n_match * 4 * 17 / 2 ** 30
    print(f"  measured at {n_small} chains so both routes stay resident "
          f"(at {n} it would ask for ~{proj:.0f} GB).\n")
    tau_s, pars_s = tau[:n_small], [p[:n_small] for p in pars]
    print(f"  {'matched accuracy':<28s} {'ms':>9s} {'peak MB':>9s} "
          f"{'speedup':>9s} {'memory':>9s}")
    for grad in (False, True):
        what = "value+grad" if grad else "forward"
        dt_e, mb_e = timeit(Bound(route_expanded, tau_s, pars_s,
                                  n_sub=n_match), args.reps, grad)
        dt_k, mb_k = timeit(Bound(route_kernel, tau_s, pars_s, n_sub=1,
                                  integration="contact", n_gl=5),
                            args.reps, grad)
        print(f"  {what + ', expanded':<28s} {dt_e * 1e3:9.2f} {mb_e:9.1f} "
              f"{1.0:8.2f}x {1.0:8.2f}x")
        print(f"  {what + ', kernel contact':<28s} {dt_k * 1e3:9.2f} "
              f"{mb_k:9.1f} {dt_e / dt_k:8.2f}x {mb_e / mb_k:8.2f}x")


if __name__ == "__main__":
    main()
