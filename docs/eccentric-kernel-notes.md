# Eccentric fused-kernel design notes

> Implementation plan (v3 kernel + in-kernel VJP reduction, milestones
> E0-E5): [v3eccentrickernel_plan.md](v3eccentrickernel_plan.md).
> **Executed in v0.3.0; since v0.6.0 this is THE model kernel** -- the
> dedicated circular kernel was retired, a circular orbit being e = 0 on
> the same kernel with a per-chain fast path. These notes describe the
> *direct* (E, e, w)
> formulation that informed the design. The shipped kernel uses the
> **transit-anchored** formulation instead (`metalplanet/anchored.py`),
> which is algebraically equivalent — verified to 2.5e-15 in flux — but
> keeps float32 gradients accurate as e -> 0. The backward formulas
> below remain correct for the direct form and were the starting point
> for the anchored chain rules in `metal._ECC_VJP_TAIL`; where the two
> differ, the code is authoritative.

Findings from a three-agent verification pass (2026-09-06: symbolic/
numerical math check, Metal micro-benchmarks, fp32 stability mapping)
that inform the future eccentric model-level kernel. The graph path
already uses the Cartesian tail (kepler.py); this file preserves the
verified pieces the kernel will need.

## Separation from eccentric anomaly (implemented in the graph path)

With E from the Markley solve, beta = sqrt(1-e^2):

    X = a (cosE - e)          # orbital-plane Cartesians = (r cos f, r sin f)
    Y = a beta sinE
    u = X cos w - Y sin w     # = r cos(w+f)
    v = X sin w + Y cos w     # = r sin(w+f); FRONT TEST IS v > 0 (unprojected)
    z = sqrt(u^2 + (v cos i)^2)

Exactly equivalent to the conic form (proven symbolically; 4.5e-25
residual at 40 digits) and strictly better conditioned: no
1 - sin^2(w+f) sin^2 i cancellation near transit (was a ~1e-6 fp32 flux
floor even at e = 0), no (1 + e cos f) collapse at near-apastron
transits (was up to 6e-3 flux error at e ~ 0.99, a = 300; Cartesian
stays ~6e-7). fp32-trustworthy at the 1e-6 flux level for e <= 0.999.
In a kernel, all six rotation coefficients hoist per chain:
x = c1 cosE + c2 sinE + c0, y' = c3 cosE + c4 sinE + c5 (cos i folded
into c3..c5; sign of y'/ci preserved for the front test).

## Verified backward-pass formulas (checked to ~1e-30 vs FD)

Save sinE, cosE, beta, X, Y, u, v, z from the forward; ct = dL/dz;
zinv = 1/max(z, tiny) (the forward needs no clamp; only this divide
does). D = 1 - e cosE >= 1 - e.

    g_u  = ct * u * zinv
    g_v  = ct * v * ci^2 * zinv
    g_X  =  g_u cw + g_v sw
    g_Y  = -g_u sw + g_v cw
    dzdE = a (-g_X sinE + g_Y beta cosE)
    dz/dM = dzdE / D
    dz/de = dzdE sinE / D - a g_X - (a e sinE / beta) g_Y
    dz/da = ct * z / a                    # exact
    dz/dw = -ct * u v si^2 * zinv
    dz/di = -ct * v^2 ci si * zinv

E-level implicit rule (kepler_E_sincos.vjp, already implemented):
dE = (dM + sinE de)/D.

## Measured kernel economics (M2 Max, toy kernels mirroring metal.py)

- Cartesian vs true-anomaly tail: +13.3-13.7% model throughput at a
  realistic 5% in-transit mix; +8-10% at 50%; +15.8% orbit-only.
  The delta is mix-independent (every thread pays the tail).
- Divide throughput is ~44x a multiply under safe math (fast::divide
  ~9x); precise::sin/cos ~84x; precise::powr ~220x.
- THE hotspot is the Markley starter's cbrt via precise::powr (~20% of
  orbit time): a cheap cbrt (bit-trick + Newton) buys as much as the
  tail switch itself. fast::divide is a second lever if accuracy
  permits.
- The Markley+one-refinement solve itself is fp32-robust: |dE| <=
  4.1e-7 rad at e = 0.999 down to M = 1e-7 (the e->1, M->0 corner
  lives in the tail choice, not the solve).
- Long baselines: fp32 raw-time phase wrap costs 1.75e-7 rad/orbit —
  crosses 1e-6 flux at ~1-2 orbits. The eccentric kernel MUST take
  epoch-centered (dt, k) inputs like the circular v2 kernel does.
- Kernel VJP: match the circular kernel's pattern (per-point partial
  arrays + mx.sum); the wrap term in dphi/dp_off analog is the
  (-k - n) chain as in metal.py's model VJP.

## Measured VJP decomposition: the mx.sum reductions are NOT the lever

An external suggestion claimed the seven per-point partial grids + the
mx.sum re-read were ~45% of the VJP and that reducing in-kernel would
give 2-2.4x on gradients. Measured on the shipped v2 kernel at
1024 x 65,536, uncontended GPU (`benchmarks/profile_vjp_reduction.py`):

| phase | median |
|---|---:|
| forward kernel | 20.8 ms |
| VJP kernel only (writes 7 grids) | 30.6 ms |
| VJP kernel + 7 sums (as shipped) | 35.9 ms |
| 7 sums alone | 5.4 ms (345 GB/s — DRAM peak) |
| full value_and_grad | 60.6 ms |

The traffic arithmetic behind the claim is right (1.88 GB written +
re-read here; 2.87 GB at m = 100k) but the VJP kernel is
**compute-bound** — the forward recompute plus eight-parameter chain
rules dominate — so the reductions are **~15% of the VJP**, and the
re-read already streams at DRAM peak. A two-stage in-kernel reduction
(simd_sum per threadgroup -> (n, 7, ceil(m/256)) partials -> tiny
mx.sum; no atomics needed; requires the predicated early exit noted
above) projects to **1.16-1.41x on backward, ~1.2x on value+grad** —
worth bundling into this eccentric kernel's VJP when it is written,
not shipping as a standalone change. The stronger argument for it is
**memory, not speed**: it removes the ~28 B/pt transient grids, which
is what currently caps gradient batch sizes (see
docs/sampler-integration.md "Memory, gradients").

Agent scratch/verification scripts (session-local, not in repo) were
validated against tests/test_kepler.py::TestCartesianTail, which pins
the equivalence, the apastron fp32 bound, and both implicit VJPs.
