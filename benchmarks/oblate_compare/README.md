# Oblate planets: cross-code benchmark

MetalPlanet's oblate-planet model against the other public oblate transit
codes: [squishyplanet](https://github.com/ben-cassese/squishyplanet)
(Cassese et al. 2024), [JoJo](https://github.com/Flippedx/JoJo) and
[GreenLantern](https://github.com/emprice/greenlantern) (Price). Results:
[RESULTS.md](RESULTS.md) (generated).

## Files

| file | role |
|---|---|
| `scenario_oblate.py` | the transit; `BENCH_SCENARIO=graze` gives the precision stress case |
| `adapters_oblate.py` | each code's parameter mapping; single curve, batch, value + gradient |
| `worker_oblate.py` | one timed measurement per subprocess |
| `run_oblate_compare.py` | precision against a 30-digit mpmath integral, then the timings |
| `make_report_oblate.py` | `results_oblate.json` -> `RESULTS.md` |
| `build_greenlantern_macos.py` | the macOS OpenCL 1.2 build of pocky and GreenLantern |

```bash
cd benchmarks/oblate_compare
.venv/bin/python run_oblate_compare.py                         # ~5 min, quiet machine
BENCH_SCENARIO=graze .venv/bin/python run_oblate_compare.py --graze
.venv/bin/python make_report_oblate.py
```

## Parameter mappings (checked against each other before timing)

| code | mapping | agreement with MetalPlanet |
|---|---|---|
| squishyplanet | `projected_effective_r`, `projected_f`, `projected_theta` = theta | 1e-15 |
| JoJo | `rp_me` = r_eff, obliquity = -theta (its Y axis is ours reversed), a/R* via the stellar density | 1e-14 |
| GreenLantern | semi-axes (B, A, A), third angle = theta, beta = -(pi/2 - i), (q1, q2) | 3-5e-6 (fp32) |

Limb darkening is I(mu) = 1 - w (1 - mu^2): MetalPlanet's hybrid2 with its
pole weight at 0, and the quadratic law (2w, -w) for the others.

## Environment (local only; `.venv/` and `vendor/` are git-ignored)

A separate venv, so squishyplanet's JAX does not touch anvil's:

```bash
cd benchmarks/oblate_compare
python3.11 -m venv .venv
.venv/bin/pip install numpy scipy mpmath matplotlib setuptools wheel squishyplanet
.venv/bin/pip install -e ../..                  # MetalPlanet
mkdir vendor && cd vendor
git clone https://github.com/Flippedx/JoJo && ../.venv/bin/pip install -e JoJo
git clone https://github.com/emprice/pocky
git clone https://github.com/emprice/greenlantern
```

**pocky and GreenLantern on macOS.** Both target OpenCL 3.0 on Linux;
Apple ships OpenCL 1.2 as a framework. `build_greenlantern_macos.py`
applies compatibility shims (idempotent; no algorithmic change):

- pocky: `<OpenCL/opencl.h>` on Apple; `clCreateCommandQueue` (1.2) for
  `clCreateCommandQueueWithProperties` (2.0); `-framework OpenCL` instead
  of `-lOpenCL`.
- GreenLantern: the same link flag; its embedded-kernel header, which
  upstream generates only in `sdist`, generated with `orbit.cl` first,
  `work_group_barrier` mapped to `barrier`, and `work_group_reduce_add`
  (OpenCL 2.0) replaced by a local-memory tree reduction over the
  (flattened 2-D) work group -- the same sums in a different order.

```bash
.venv/bin/python build_greenlantern_macos.py
.venv/bin/pip install --no-build-isolation -e vendor/pocky
.venv/bin/pip install --no-build-isolation -e vendor/greenlantern
```

After the shims GreenLantern agrees with MetalPlanet's spherical model to
1.2e-6 and with its oblate model to 3-5e-6, its fp32 Simpson-rule level.

## Notes on fairness

- Each (code, size) runs in a fresh process; JIT compilation (JAX, MLX,
  OpenCL) is excluded, the median of >= 3 repetitions is reported.
- squishyplanet runs XLA on all CPU cores; JoJo is single-threaded numpy;
  GreenLantern and MetalPlanet fp32 use the GPU.
- GreenLantern's timed call includes its device-to-host copy, as
  MetalPlanet's returns numpy; its value + gradient is its forward-mode
  dual kernel (all 12 parameter derivatives per point) contracted with the
  cotangent on the host.
- MetalPlanet's fp32 route includes its per-call contact solve.
