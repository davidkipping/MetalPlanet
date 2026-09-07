"""Decompose the v2 VJP cost: kernel dispatch vs the 7 mx.sum reductions.

Claim under test (from anvil session): the per-point partial grids +
mx.sum re-read are ~45% of the VJP; in-kernel reduction ~2-2.4x on grads.
"""
import time
import numpy as np
import mlx.core as mx

from metalplanet.metal import _get_model_kernels, make_model_core_metal

N, M = 1024, 65536
PREF = 3.456
REPS = 12

rng = np.random.default_rng(0)
dt = rng.uniform(-0.2, 0.2, M).astype(np.float32)
kk = np.zeros(M, dtype=np.float32)
x2d = mx.array(np.vstack([dt, kk]))
ones = np.ones(N, dtype=np.float32)
t0 = mx.array(0.0 * ones); pp = mx.array(0.0 * ones)
r = mx.array(0.1 * ones); b = mx.array(0.3 * ones)
a = mx.array(8.8 * ones)
u1 = mx.array(0.4225 * ones); u2 = mx.array(0.3077 * ones)
ct = mx.ones((N, M), dtype=mx.float32)
mx.eval(x2d, t0, pp, r, b, a, u1, u2, ct)

kf = _get_model_kernels()["model_fwd"]
kv = _get_model_kernels()["model_vjp"]

def timeit(fn, label):
    fn()  # warmup (JIT + buffers)
    fn()
    ts = []
    for _ in range(REPS):
        mx.synchronize()
        t0_ = time.perf_counter()
        fn()
        mx.synchronize()
        ts.append(time.perf_counter() - t0_)
    med = sorted(ts)[len(ts) // 2]
    print(f"{label:34s} {med*1e3:8.2f} ms")
    return med

def run_fwd():
    out = kf(inputs=[x2d, t0, pp, r, b, a, u1, u2, PREF, M],
             output_shapes=[(N, M)], output_dtypes=[mx.float32],
             grid=(M, N, 1), threadgroup=(256, 1, 1))[0]
    mx.eval(out)

def run_vjp_kernel_only():
    outs = kv(inputs=[x2d, t0, pp, r, b, a, u1, u2, ct, PREF, M],
              output_shapes=[(N, M)] * 7, output_dtypes=[mx.float32] * 7,
              grid=(M, N, 1), threadgroup=(256, 1, 1))
    mx.eval(*outs)

def run_vjp_with_sums():
    outs = kv(inputs=[x2d, t0, pp, r, b, a, u1, u2, ct, PREF, M],
              output_shapes=[(N, M)] * 7, output_dtypes=[mx.float32] * 7,
              grid=(M, N, 1), threadgroup=(256, 1, 1))
    sums = [mx.sum(o, axis=1) for o in outs]
    mx.eval(*sums)

grids = kv(inputs=[x2d, t0, pp, r, b, a, u1, u2, ct, PREF, M],
           output_shapes=[(N, M)] * 7, output_dtypes=[mx.float32] * 7,
           grid=(M, N, 1), threadgroup=(256, 1, 1))
mx.eval(*grids)

def run_sums_alone():
    sums = [mx.sum(o, axis=1) for o in grids]
    mx.eval(*sums)

core = make_model_core_metal(PREF)
def loss(t0_, pp_, r_, b_, a_, u1_, u2_):
    return mx.sum(core(x2d, t0_, pp_, r_, b_, a_, u1_, u2_))
vg = mx.value_and_grad(loss, argnums=tuple(range(7)))
def run_value_grad():
    v, g = vg(t0, pp, r, b, a, u1, u2)
    mx.eval(v, *g)

t_fwd = timeit(run_fwd, "forward kernel")
t_k = timeit(run_vjp_kernel_only, "VJP kernel only (7 grids, no sum)")
t_ks = timeit(run_vjp_with_sums, "VJP kernel + 7 sums (as shipped)")
t_s = timeit(run_sums_alone, "7 sums alone (grids pre-built)")
t_vg = timeit(run_value_grad, "full value_and_grad")

gb = N * M * 7 * 4 / 1e9
print(f"\ngrid traffic: {gb:.2f} GB written + {gb:.2f} GB re-read")
print(f"sum read bandwidth: {gb / t_s:.0f} GB/s")
print(f"reduction share of VJP: {(t_ks - t_k) / t_ks * 100:.0f}% "
      f"(in-situ) / {t_s / t_ks * 100:.0f}% (sums alone)")
print(f"write-side saving upper bound: {t_s*1e3:.1f} ms "
      f"(store GB = read GB; only if stores were bandwidth-serialized)")
lo = t_k + 0.3e-3           # keep all kernel time, tiny partial-sum cost
hi = max(t_k - gb / (gb / t_s), t_fwd) + 0.3e-3  # also save the stores
print(f"projected VJP after in-kernel reduction: {lo*1e3:.1f}-"
      f"{hi*1e3:.1f} ms -> backward speedup {t_ks/lo:.2f}-{t_ks/hi:.2f}x")
print(f"projected value+grad: {(t_fwd+lo)*1e3:.1f}-{(t_fwd+hi)*1e3:.1f} ms "
      f"vs {t_vg*1e3:.1f} ms now")
