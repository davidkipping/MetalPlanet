# Plan (not implemented): route the frontend's circular path through the v2 kernel

**Status: SUPERSEDED (2026-09-28).** An adversarial review of this plan
asked a question it had not: does a separate circular *kernel* need to
exist at all, given the transit-anchored eccentric orbit is exact at
e = 0? Measured, the eccentric kernel fed e = 0 cost 1.46x/1.41x
(forward/value+grad) the dedicated circular one -- almost all of it the
Kepler starter and refinement run as dead work -- and a per-chain
`if (e == 0)` branch (simdgroup-uniform, so it cannot diverge) recovered
most of that at ~1% cost to eccentric chains. The decision taken was to
**keep circular as a first-class mode and retire the circular kernel**:
one fused kernel now serves both, the frontend's circular path routes
through it automatically (which is what this plan wanted, without a
second route), and mixed circular/eccentric batches are free. See
CHANGELOG 0.6.0 for the measured cost of the unification. The text below
is kept as the record of the alternative that was *not* chosen.

## What it would do

`TransitModel.light_curve` on the fp32 GPU path currently computes the
*circular* orbit as MLX graph ops and calls only the **photometric**
kernel (`flux_dev_metal`). The eccentric path was moved onto the full
fused model kernel in v0.3.0; this would do the same for circular, using
`make_model_core_metal` instead of `make_ecc_core_metal`.

## Why it is not obviously worth it

| path | throughput at N = 10^7, fp32 GPU |
|---|---:|
| frontend, circular (graph orbit + photometric kernel) | 2.31 Gpt/s |
| fused v2 model kernel, measured directly | 3.26 Gpt/s |

So the ceiling is **~1.4x**, against **2.1x** measured for the eccentric
switch — because a circular orbit is two trig calls and a square root,
which the graph already does cheaply, whereas the eccentric orbit is a
Kepler solve whose intermediates dominated memory traffic.

Against that: `light_curve` is the most-used entry point in the package
and the only one with batman-parity tests. The eccentric switch was
low-risk because that path was new; this one is not.

## Design, if it is done

Mirror `_ecc_kernel_usable` / the contact branch in `api.py`:

1. Add `_circ_kernel_usable()`: `use_metal`, `transittype == "primary"`,
   `dtype == mx.float32`, GPU stream, `metal_available()`, **and not**
   `self._n_poly` (the v2 kernel is quadratic-only, exactly as the v3
   kernel is).
2. In `_get_compiled(circular=True)`, add a branch building
   `make_model_core_metal(0.0)` and calling it with
   `xdat = stack([t, zeros_like(t)])`.
3. **Carry the period in the traced `p_off` input with `period_ref = 0`**,
   as the eccentric routing does. `period_ref` is baked into the kernel
   factory as a Python float, so baking the real period would break
   batman-style parameter updates. With the epoch column `k = 0` this is
   exactly equivalent, gradients included: the kernel's
   `dphi/dp_off = 2 pi ((-k - n_w) P - tau_w) / P^2` reduces to `dphi/dP`.
4. The kernel takes `b` and `a` directly, which is what the circular
   `raw` already has — no reparameterization needed (this is *simpler*
   than the eccentric case, which had to pack 11 orbit constants).
5. `light_curve` returns `1 + dev`; the kernel's out-of-transit output is
   exactly 0, so the existing contract holds.

## The traps, all already paid for once

- **Do not** bake `period_ref`; see step 3. This was the subtle part of
  the eccentric routing and the reason a regression test exists for it
  (`test_period_update_is_not_baked_in`).
- The kernel is **quadratic-only**. Polynomial and secondary-eclipse
  models must keep the graph path.
- Benchmark `light_curve`, **not** `light_curve_mx`: the latter is the
  eager path and cannot use the compiled graph or the kernels, which
  produced a wrong 10x estimate for the eccentric case before it was
  caught. And pass `dtype=mx.float32` — the default is fp64 on the CPU.

## Gates

- Parity vs the current graph path < 5e-6 on `light_curve`, with the
  fp64 oracle adjudicating any exceedance (the kernel must be no worse
  against fp64 than the graph is).
- All existing batman-parity tests unchanged, especially
  `TestPrimaryParity::test_circular_quadratic` at 3e-8.
- `test_param_update_between_calls` still passes, plus a new
  period-update test mirroring the eccentric one.
- Measured >= 1.3x on `light_curve` at N = 10^7 fp32. **If it comes in
  below that, revert** — the risk is only justified by the speed.

## Effort and verdict

Half a day including tests; the eccentric routing is a working template
and this case is strictly simpler. The verdict stands: do it only if
single-curve fp32 GPU throughput becomes a goal in itself. For sampling
workloads it is irrelevant — `metalplanet.anvil` already uses the fused
kernel, and `TransitModel.light_curves` (v0.5.0) batches the frontend,
which is worth ~112x where this is worth ~1.4x.
