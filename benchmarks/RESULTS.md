# MetalPlanet cross-code benchmark

Machine: Apple M2 Max (12 CPU cores: 8P+4E; one 30-core GPU), macOS, MLX GPU fp32 / CPU fp64.

Scenario: quadratic limb-darkened primary transit (P=3.456 d, a/R*=8.8, b=0.45, Rp/R*=0.1, u=[0.40, 0.25]), 241-point precision grid, N-point speed sweeps.


## Precision vs mpmath direct-integration oracle (30 digits)

| code | max |err| | median |err| |
|---|---:|---:|
| metalplanet fp64 | 2.22e-16 | 0.00e+00 |
| exoplanet-core | 3.33e-16 | 0.00e+00 |
| jaxoplanet (order=50) | 1.75e-12 | 0.00e+00 |
| jaxoplanet (order=10) | 4.58e-09 | 0.00e+00 |
| batman | 6.04e-09 | 2.10e-10 |
| pytransit (exact) | 4.04e-08 | 7.96e-10 |
| metalplanet fp32(GPU) | 1.75e-07 | 5.24e-09 |
| pytransit (interp) | 6.26e-06 | 1.17e-07 |

`ellc` excluded: its PyPI wheel ships an x86_64-only binary (incompatible with arm64) and source builds need gfortran.


## Single light curve: wall time vs N (median)

| code (threads) | N=1,000 | N=10,000 | N=100,000 | N=1,000,000 | N=10,000,000 |
|---|---:|---:|---:|---:|---:|
| batman (1) | 0.02 ms | 0.20 ms | 1.92 ms | 18.91 ms | 193.80 ms |
| exoplanet (1) | 0.04 ms | 0.29 ms | 2.82 ms | 29.79 ms | 318.36 ms |
| jaxoplanet (all) | 0.13 ms | 0.56 ms | 2.10 ms | 20.19 ms | 182.95 ms |
| metalplanet_fp32 (all) | 0.24 ms | 0.70 ms | 0.74 ms | 1.43 ms | 9.27 ms |
| metalplanet_fp64 (all) | 0.48 ms | 1.55 ms | 14.64 ms | 132.65 ms | 1,317.12 ms |
| pytransit (1) | 0.04 ms | 0.36 ms | 3.55 ms | 35.72 ms | 360.43 ms |
| pytransit (4) | 0.04 ms | 0.36 ms | 3.55 ms | 35.71 ms | 359.28 ms |
| pytransit (8) | 0.04 ms | 0.36 ms | 3.59 ms | 35.76 ms | 358.83 ms |
| pytransit (12) | 0.04 ms | 0.36 ms | 3.54 ms | 35.81 ms | 359.70 ms |

## Native batch: 512 parameter sets x 100,000 points

| code | wall | curves/s | notes |
|---|---:|---:|---|
| metalplanet_gpu | 0.010 s | 53,590 | fp32 GPU, one broadcast graph |
| pytransit | 0.236 s | 2,174 | native parameter arrays (numba) |
| jaxoplanet | 0.903 s | 567 | jax.vmap, CPU x64 |
| batman | 1.787 s | 287 | python loop (no native batch) |
| exoplanet | 1.498 s | 342 | python loop (no native batch) |

## Batch-size scaling (npv x 100,000 points): GPU vs its strongest CPU rival

| npv | MetalPlanet GPU [curves/s] | PyTransit 12-core [curves/s] |
|---:|---:|---:|
| 64 | 40,638 | 2,037 |
| 256 | 52,037 | 2,168 |
| 512 | 53,592 | 2,149 |
| 1024 | 54,757 | 1,986 |
| 2048 | 55,126 | 2,004 |
| 4096 | 55,268 | 2,024 |

With the fused model-level Metal kernel (orbit + photometry in one register-resident pass, ~12 B/pt of memory traffic) the GPU streams ~55,000 curves/s flat to npv = 4096 with no memory cliff. PyTransit's numba batch streams ~1,950 curves/s at every size on 12 cores.


## What the speed tables do not show

Only MetalPlanet and jaxoplanet are differentiable, and only MetalPlanet
ships analytic gradients, fused into a Metal backward kernel: value+gradient
costs 52.5 ms vs 27.3 s for reverse-mode autodiff at 1024 x 65,536 — a
519x spread that is the number that matters for HMC sampling. batman,
PyTransit, exoplanet-core (numpy layer), and ellc provide no gradients.

## Reading the scaling axes

* **CPU cores**: thread counts are honest core scaling for batman
  (OpenMP, if available) and PyTransit (numba). exoplanet-core is
  single-threaded by design; jaxoplanet (XLA) and MLX's CPU stream
  manage their own all-core pools and cannot be cleanly partitioned.
* **GPU cores**: Apple Silicon exposes one GPU whose cores cannot be
  partitioned from user space, so the N-sweep doubles as the GPU
  *saturation* curve — at small N most of the GPU's cores idle and
  latency dominates; throughput (points/s) rises with N until all
  cores saturate. True GPU-core scaling requires comparing chips
  (e.g. 30-core M2 Max vs 76-core M2 Ultra).
