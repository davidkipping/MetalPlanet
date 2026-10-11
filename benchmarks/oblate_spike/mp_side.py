import sys, json, time, numpy as np, mlx.core as mx, metalplanet as mp
from metalplanet.metal_hybrid import flux_dev_metal_hybrid
d = json.load(open(sys.argv[1]))
W = {"hybrid2": [0.3, 0.2], "hybrid4": [0.2, 0.2, 0.1, 0.1], "hybrid5": [0.2, 0.2, 0.1, 0.1, 0.1]}
z = np.array(d["z"])
print("== f -> 0: SquishierPlanet's oblate path vs MetalPlanet's spherical closed forms (fp64), same geometry ==")
for law in ("hybrid2", "hybrid5"):
    with mx.stream(mx.cpu):
        Fs = 1.0 + np.asarray(mp.flux_dev_hybrid(mx.array(z, dtype=mx.float64), 0.1, W[law], law))
    intr = Fs < 1.0
    for f in (0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
        Fe = np.array(d[f"F_{law}_{f}"])
        dmax = np.abs(Fe - Fs).max()
        print(f"  {law} f={f:<7g} max|F_oblate - F_sphere| = {dmax:.2e}" + (f"   / f = {dmax / f:.3e}" if f > 0 else ""))
print("== MetalPlanet spherical cost, ns per point ==")
for law in W:
    for n in (4096, 262144):
        zz = mx.array(np.tile(z, n // z.size + 1)[:n], dtype=mx.float64)
        with mx.stream(mx.cpu):
            fn = mx.compile(lambda zz: mp.flux_dev_hybrid(zz, 0.1, W[law], law))
            mx.eval(fn(zz)); ts = []
            for _ in range(7):
                s = time.perf_counter(); mx.eval(fn(zz)); ts.append(time.perf_counter() - s)
        z32 = mx.array(np.asarray(zz), dtype=mx.float32)
        mx.eval(flux_dev_metal_hybrid(z32, 0.1, law, W[law])); tk = []
        for _ in range(7):
            s = time.perf_counter(); mx.eval(flux_dev_metal_hybrid(z32, 0.1, law, W[law])); tk.append(time.perf_counter() - s)
        print(f"  {law} n={n:>7}: fp64 CPU graph {sorted(ts)[3]/n*1e9:7.1f} ns/pt   fp32 GPU kernel {sorted(tk)[3]/n*1e9:7.2f} ns/pt")
