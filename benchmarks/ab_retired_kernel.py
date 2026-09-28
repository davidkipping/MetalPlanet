"""Unified kernel vs the retired dedicated circular kernel, same run.

The pre-0.6.0 metal.py is extracted from git, its relative imports made
absolute, and its kernels RENAMED -- MLX caches JIT kernels by name, so
two different sources under one name would collide silently. Both
"value+grad" columns evaluate the value AND the gradients: our custom VJP
recomputes from the primals and ignores the forward output, so evaluating
only the gradients would skip the forward kernel entirely.

Quiet-machine result (M2 Max, 1024 x 65,536): circular fits on the
unified kernel cost 1.06x forward / 1.10x value+grad relative to the
retired kernel; eccentric chains 1.01x; flux parity 4e-7 at e = 0
(fp32 operation order), bit-identical at e = 0.3.
"""
import importlib.util
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import mlx.core as mx

import metalplanet.metal as NEW
from metalplanet.anchored import pack_orbit_constants

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RETIRE_COMMIT = "6da12b4"        # the retirement; its parent has the kernel


def load_retired():
    src = subprocess.check_output(
        ["git", "-C", ROOT, "show", f"{RETIRE_COMMIT}^:metalplanet/metal.py"],
        text=True)
    src = src.replace("from .vjp import", "from metalplanet.vjp import")
    for tag in ("model_", "ecc_", "flux_"):
        src = src.replace(f'name="mp_{tag}', f'name="mp_retired_{tag}')
    path = os.path.join(tempfile.mkdtemp(), "metal_retired.py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("metal_retired", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OLD = load_retired()
N, MM, PREF = 1024, 65536, 3.456
rng = np.random.default_rng(0)
x2d = mx.array(np.vstack([rng.uniform(-0.2, 0.2, MM).astype(np.float32),
                          np.zeros(MM, np.float32)]))
o = np.ones(N, np.float32)
t0, pp, r, a, b = (mx.array(0 * o), mx.array(0 * o), mx.array(0.1 * o),
                   mx.array(8.8 * o), mx.array(0.3 * o))
u1, u2, ci = mx.array(0.4225 * o), mx.array(0.3077 * o), mx.array(np.float32(0.3 / 8.8) * o)


def orbit(e, w=1.1):
    return pack_orbit_constants(
        mx.array(np.float32(math.sqrt(e) * math.cos(w)) * o),
        mx.array(np.float32(math.sqrt(e) * math.sin(w)) * o), ci)


orb0, orbE = orbit(0.0), orbit(0.3)
mx.eval(x2d, orb0, orbE)
old_v2 = OLD.make_model_core_metal(PREF, reduce="simd")
old_v3 = OLD.make_ecc_core_metal(PREF)
new = NEW.make_model_core_metal(PREF)


def timeit(f):
    f(); f()
    ts = []
    for _ in range(12):
        mx.synchronize(); s = time.perf_counter(); f(); mx.synchronize()
        ts.append(time.perf_counter() - s)
    return sorted(ts)[6] * 1e3


def fwd(core, ob):
    return lambda: mx.eval(core(x2d, t0, pp, r, a, ob, u1, u2))


def vg(core, ob):
    f = mx.value_and_grad(lambda *p: mx.sum(core(x2d, *p)), argnums=tuple(range(7)))

    def run():
        v, g = f(t0, pp, r, a, ob, u1, u2)
        mx.eval(v, *g)
    return run


def fwd_v2():
    mx.eval(old_v2(x2d, t0, pp, r, b, a, u1, u2))


_f2 = mx.value_and_grad(lambda *p: mx.sum(old_v2(x2d, *p)), argnums=tuple(range(7)))


def vg_v2():
    v, g = _f2(t0, pp, r, b, a, u1, u2)
    mx.eval(v, *g)


rows = {
    "retired circular kernel        (e=0)": (timeit(fwd_v2), timeit(vg_v2)),
    "retired eccentric kernel, fed  (e=0)": (timeit(fwd(old_v3, orb0)), timeit(vg(old_v3, orb0))),
    "unified kernel                 (e=0)": (timeit(fwd(new, orb0)), timeit(vg(new, orb0))),
    "retired eccentric kernel     (e=0.3)": (timeit(fwd(old_v3, orbE)), timeit(vg(old_v3, orbE))),
    "unified kernel               (e=0.3)": (timeit(fwd(new, orbE)), timeit(vg(new, orbE))),
}
print(f"{'kernel':<40}{'forward':>10}{'value+grad':>12}")
for k, (f, g) in rows.items():
    print(f"{k:<40}{f:10.2f}{g:12.2f}   ms")
(f2, g2), (f3, g3), (fn, gn) = (rows[k] for k in list(rows)[:3])
(fe3, ge3), (fen, gen) = (rows[k] for k in list(rows)[3:])
print(f"\ncircular fits, unified vs retired kernel:  forward {fn/f2:.2f}x  value+grad {gn/g2:.2f}x")
print(f"  without the fast paths it would be:      forward {f3/f2:.2f}x  value+grad {g3/g2:.2f}x")
print(f"eccentric fits, unified vs retired kernel: forward {fen/fe3:.3f}x  value+grad {gen/ge3:.3f}x")
fl2 = np.array(old_v2(x2d, t0, pp, r, b, a, u1, u2)[0], dtype=np.float64)
fln = np.array(new(x2d, t0, pp, r, a, orb0, u1, u2)[0], dtype=np.float64)
fle = np.array(old_v3(x2d, t0, pp, r, a, orbE, u1, u2)[0], dtype=np.float64)
flu = np.array(new(x2d, t0, pp, r, a, orbE, u1, u2)[0], dtype=np.float64)
print(f"flux parity: e=0 vs retired circular {np.abs(fln-fl2).max():.2e}; "
      f"e=0.3 vs retired eccentric {np.abs(flu-fle).max():.2e}")
