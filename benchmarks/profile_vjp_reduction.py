"""Decompose the v2 VJP cost and A/B the two reduction strategies.

"grid": kernel writes 7 full (n, m) partial arrays, mx.sum reduces them.
"simd": metal::simd_sum reduces inside the kernel; outputs are
        (n, ceil(m/32)) — 1/32 the transient traffic.

Claim tested originally (from an anvil session): the per-point grids +
mx.sum re-read were ~45% of the VJP and in-kernel reduction would give
2-2.4x. Measured: the reductions are ~15% and the kernel is
compute-bound, so the realistic payoff is ~1.2x — the decisive win is
the transient memory, not the time.
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
x2d = mx.array(np.vstack([dt, np.zeros(M, dtype=np.float32)]))
ones = np.ones(N, dtype=np.float32)
t0 = mx.array(0.0 * ones); pp = mx.array(0.0 * ones)
r = mx.array(0.1 * ones); b = mx.array(0.3 * ones)
a = mx.array(8.8 * ones)
u1 = mx.array(0.4225 * ones); u2 = mx.array(0.3077 * ones)
ct = mx.ones((N, M), dtype=mx.float32)
mx.eval(x2d, t0, pp, r, b, a, u1, u2, ct)

kf = _get_model_kernels()["model_fwd"]
kg = _get_model_kernels()["model_vjp_grid"]
ks = _get_model_kernels()["model_vjp_simd"]
COLS = (M + 31) // 32
ARGS = [x2d, t0, pp, r, b, a, u1, u2, ct, PREF, M]


def timeit(fn, label):
    fn(); fn()                      # warmup (JIT + buffers)
    ts = []
    for _ in range(REPS):
        mx.synchronize()
        t0_ = time.perf_counter()
        fn()
        mx.synchronize()
        ts.append(time.perf_counter() - t0_)
    med = sorted(ts)[len(ts) // 2]
    print(f"{label:38s} {med*1e3:8.2f} ms")
    return med


def run_fwd():
    mx.eval(kf(inputs=[x2d, t0, pp, r, b, a, u1, u2, PREF, M],
               output_shapes=[(N, M)], output_dtypes=[mx.float32],
               grid=(M, N, 1), threadgroup=(256, 1, 1))[0])


def grid_kernel_only():
    mx.eval(*kg(inputs=ARGS, output_shapes=[(N, M)] * 7,
                output_dtypes=[mx.float32] * 7,
                grid=(M, N, 1), threadgroup=(256, 1, 1)))


def grid_full():
    outs = kg(inputs=ARGS, output_shapes=[(N, M)] * 7,
              output_dtypes=[mx.float32] * 7,
              grid=(M, N, 1), threadgroup=(256, 1, 1))
    mx.eval(*[mx.sum(o, axis=1) for o in outs])


def simd_full():
    outs = ks(inputs=ARGS, output_shapes=[(N, COLS)] * 7,
              output_dtypes=[mx.float32] * 7, init_value=0.0,
              grid=(M, N, 1), threadgroup=(256, 1, 1))
    mx.eval(*[mx.sum(o, axis=1) for o in outs])


grids = kg(inputs=ARGS, output_shapes=[(N, M)] * 7,
           output_dtypes=[mx.float32] * 7,
           grid=(M, N, 1), threadgroup=(256, 1, 1))
mx.eval(*grids)


def sums_alone():
    mx.eval(*[mx.sum(o, axis=1) for o in grids])


def value_grad(reduce):
    core = make_model_core_metal(PREF, reduce=reduce)

    def loss(*p):
        return mx.sum(core(x2d, *p))
    vg = mx.value_and_grad(loss, argnums=tuple(range(7)))

    def run():
        v, g = vg(t0, pp, r, b, a, u1, u2)
        mx.eval(v, *g)
    return run


print(f"=== v2 model kernel, {N} x {M:,} ===")
t_fwd = timeit(run_fwd, "forward kernel")
t_gk = timeit(grid_kernel_only, "grid: VJP kernel only (7 grids)")
t_g = timeit(grid_full, "grid: kernel + 7 mx.sum")
t_s = timeit(sums_alone, "grid: the 7 sums alone")
t_simd = timeit(simd_full, "simd: kernel + reduction (in-kernel)")
t_vgg = timeit(value_grad("grid"), "value_and_grad (grid)")
t_vgs = timeit(value_grad("simd"), "value_and_grad (simd)")

gb = N * M * 7 * 4 / 1e9
gb_s = N * COLS * 7 * 4 / 1e9
print(f"\ntransients: grid {gb:.2f} GB written + re-read"
      f"  |  simd {gb_s*1e3:.1f} MB  ({gb/gb_s:.0f}x less)")
print(f"reduction share of grid VJP: {t_s / t_g * 100:.0f}%")
print(f"backward speedup (simd vs grid):   {t_g / t_simd:.2f}x")
print(f"value+grad speedup (simd vs grid): {t_vgg / t_vgs:.2f}x")

for tag, run in (("grid", value_grad("grid")), ("simd", value_grad("simd"))):
    mx.clear_cache()
    mx.reset_peak_memory()
    run()
    print(f"peak memory, value_and_grad ({tag}): "
          f"{mx.get_peak_memory() / 1e9:.2f} GB")
