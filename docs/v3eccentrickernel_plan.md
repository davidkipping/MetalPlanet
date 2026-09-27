# v3 eccentric model kernel + in-kernel VJP reduction — combined plan

**Revision 2 (2026-09-27)** — amended after an adversarial review that
tested the plan's assumptions against Metal and the production code.
The review log at the end records what changed and why; every new
number below reproduces from a script in `benchmarks/`.

> **STATUS: executed (2026-09-27), shipped in v0.3.0.** E0-E5 all
> landed and every gate was met. Measured at 1024 x 65,536 on an M2 Max:
>
> | milestone | gate | measured |
> |---|---|---|
> | E0 simd reduction | backward >= 1.1x, transients gone | **1.16x**, 1.88 GB -> 58.7 MB, peak 4.56 -> 2.74 GB |
> | E1 eccentric forward | parity <= 5e-7, <= 1.5x circular | **3.7e-7**, **1.41x** (with E2) |
> | E2 cheap cbrt | E1 gates held | 1.54x -> **1.41x**, cbrt 1.7e-6 |
> | E3 eccentric VJP | all chains vs fp64, <= 2x circular | **1.38x**, 902x the graph path |
> | E4 anvil target | no unphysical state finite-and-unpenalized | `ecc_constraint_penalty` + gate tests |
> | E5 eccentric ChEES | zero divergences, truth recovered | see CHANGELOG 0.3.0 |
>
> Two real bugs were found *by building the gates*, both the same class
> and both in code this plan introduced: a floored denominator in
> `_one_minus_cos` (NaN gradients at delta ~ pi, ordinary apastron
> geometry) and `sqrt(max(e, 0))` in the anvil transform (infinite
> derivative at e = 0, an interior point of the sampling disc). Recorded
> here because the plan's risk list did *not* anticipate them: it worried
> about register pressure and the sign fold, neither of which bit.

Two pieces of work: the v3 fused kernel for eccentric orbits (forward +
analytic VJP), and a two-stage in-kernel gradient reduction replacing
the per-point partial grids + `mx.sum` pattern. Revision 1 coupled them
because the reduction was believed to require predicated early exits
that the v3 VJP would have to be written around. **That premise was
false** (review finding 1): the reduction works with the existing
store-and-return exits, so E0 is now a small, independent change that
ships first. The two remain in one plan only because the reduction
matters most for v3 — its VJP has 11 per-point partials, so the naive
grid pattern would write ~44 B/pt (2.95 GB at 1024 x 65,536) versus
v2's 28 B/pt.

## Grounding measurements (M2 Max, 1024 x 65,536, uncontended)

- v2 circular kernel (`benchmarks/profile_vjp_reduction.py`): forward
  20.8 ms; VJP kernel 30.6 ms; + 7 sums 35.9 ms; sums alone 5.4 ms at
  345 GB/s (DRAM peak). The VJP kernel is compute-bound; reductions are
  ~15% of it. Reduction payoff 1.16-1.41x backward, ~1.2x value+grad;
  the larger win is removing the transient grids (the gradient
  batch-size cap in docs/sampler-integration.md).
- **Compiled graph eccentric path** (`benchmarks/v3_ecc_graph_baseline.py`):
  forward **566.7 ms**, value+grad **68.6 s** (photometric autodiff).
  A kernel near 25-30 ms forward is ~20x; gates below are therefore
  absolute (relative to the circular kernel), not relative to the graph.
- **Reduction spike** (`benchmarks/v3_reduction_spike.py`): `simd_sum`
  over the *active* lanes after early-`return`ed lanes gives the correct
  masked sum (1.8e-7 rel, fp32 rounding); `simd_is_first()` selects the
  lowest active lane; threadgroup memory + `threadgroup_barrier` also
  work in `mx.fast.metal_kernel`, even on an exact (unpadded) grid;
  `init_value` works; attribute names `thread_index_in_simdgroup`,
  `simdgroup_index_in_threadgroup`, `threadgroup_position_in_grid` are
  all supported in MLX 0.32.2. MSL specifies SIMD-group reductions
  "across all active threads"; a returned thread is inactive.
- **(k, h) gradient conditioning** (`benchmarks/v3_kh_grad_conditioning.py`):
  fp32 gradients w.r.t. (sqrt(e) cos w, sqrt(e) sin w) through the
  production graph path vs fp64 — relative error 4e-6 at e = 0.1,
  6e-4 at 1e-3, **5% at 1e-5, 40% at 1e-6**. See "Parameterization".
- Kernel economics (docs/eccentric-kernel-notes.md): Cartesian tail
  +13-16% over the true-anomaly tail; the Markley starter's cbrt via
  `precise::powr` is ~220x a multiply and ~20% of orbit time.
- Solver: Markley + one 5th-order refinement holds |dE| <= 4.1e-7 rad at
  e = 0.999 for an *exact* M. Input conditioning is separate: dE/dM =
  1/(1 - e cos E) reaches 1/(1-e) = 1000 at e = 0.999 near periastron,
  so fp32 rounding of M (~2e-7) becomes ~2e-4 in E there. Contract:
  e <= 0.999 (clamped in-graph); parity gates are scoped accordingly.
- Backward formulas verified to ~1e-30 vs finite differences
  (docs/eccentric-kernel-notes.md).

## Design

### Parameterization and the in-graph / in-kernel split

Follow the v2 precedent: the custom_function boundary sits at the
kernel inputs; sampler-facing transforms stay in the MLX graph so their
Jacobians ride autodiff.

Sampler-facing vector for the new anvil target (10 parameters):

    v = (t0_off, p_off, r, b, a, q1, q2, secosw, sesinw, df0)

**The fp32 conditioning hazard (review finding 2).** The flux's
dependence on w cancels between M_tra and the frame rotation to O(e).
Computed naively it is a difference of O(1) fp32 terms (absolute error
~1e-6), then amplified by dw/dk = h/e ~ e^(-1/2) against a signal that
shrinks like sqrt(e): relative gradient error ~1e-6/e, matching the
measurements above. HMC on a typical low-e planet would sit in that
noise. Two mitigations, decided in **E1** when the orbit fragment is
written (cheap then, a rewrite later):

- **Preferred — transit-anchored formulation.** Solve for delta = E - E0
  (E0 = eccentric anomaly at inferior conjunction) via the shifted
  Kepler equation phi = delta - e [sin E0 (cos delta - 1) + cos E0 sin
  delta], and carry psi = w + E0 = pi/2 - Delta(e, w) with Delta = O(e)
  computed stably (Delta = f0 - E0 with f0 = pi/2 - w, from the
  half-angle form 2 atan[(1 - sqrt((1-e)/(1+e))) tan(f0/2) / (1 +
  sqrt((1-e)/(1+e)) tan^2(f0/2))], whose leading factor is O(e)). Every
  w-dependence then enters through e-scaled terms and the gradient
  error is flat in e. Cost: a modified starter/refinement in delta;
  the Markley starter still supplies E for delta's initial value only
  if its absolute error is acceptable — evaluate in E1.
- **Fallback — floor.** ParamSpec e >= 1e-3 (relative gradient error
  <= ~6e-4) plus a gradient-accuracy gate at the floor. Astrophysically
  harmless; the e-effects on a light curve at 1e-3 are below any
  photometric constraint.

In-graph, per chain (where-guarded denominators per the masked-division
lesson; isinstance/math.* branches for float inputs):

- (q1, q2) -> (u1, u2) as v2.
- e = secosw^2 + sesinw^2, clamped <= 0.999; (cw, sw) = (secosw,
  sesinw)/sqrt(e) with the denominator where-guarded at e = 0.
- ci = b (1 + e sw) / (a (1 - e^2)) — impact parameter at inferior
  conjunction to cos(inc); guard (1 - e^2).
- M_tra: port `kepler.mean_anomaly_offset_at_transit` to mx ops
  (currently host-side math.*): f0 = pi/2 - w, E0 = 2 atan2(sqrt(1-e)
  sin(f0/2), sqrt(1+e) cos(f0/2)), M_tra = E0 - e sin E0. Smooth;
  autodiff handles it.

**Joint physical constraints (review finding 3).** Periastron clearance
a(1-e) > 1 + r and cos i <= 1 (b <= a(1-e^2)/(1+e sw)) are joint
constraints; anvil offers only box ParamSpecs (its transforms.py says
extra priors belong in the model log-prob). The where-guards would make
violations return *finite* values — a silently improper posterior, the
failure the sampler-guide review caught. E4 must implement, and gate:

- Reparameterize where a box then suffices: sample q_e in [0, 1) with
  e = q_e * e_max(a, r), e_max = min(0.999, 1 - (1+r)/a) — in-graph,
  differentiable.
- For cos i <= 1, a smooth barrier penalty in a thin wrapper around
  `ChunkedGaussianLogLike` (HMC-friendly; -inf walls cause
  divergences), plus a test asserting no unphysical state receives a
  finite penalty-free log-prob.

Kernel inputs: x2d (2, m) epoch-centered (dt, k) rows as v2, per-chain
arrays t0_off, p_off, r, ci, a, e, sw, cw, u1, u2, M_tra (11), plus
`period_ref` and m as scalars (runtime args, never baked into source).
beta = sqrt(1 - e^2) is recomputed per thread so e remains the only
eccentricity input and the verified dz/de formula applies unchanged.
(If E1 adopts the transit-anchored formulation, the per-chain inputs
become (Delta or psi, E0-derived quantities) instead of (sw, cw, M_tra);
the partial count stays 11.)

### Forward pipeline (`_ORBIT_ECC` fragment)

Reuse the v2 wrap verbatim (metal.py `_ORBIT`): tau, n_w =
`metal::rint`(tau/P) (NOT round — half-to-even), phi. Then:

1. M = phi + M_tra[y]. M is linear in t with phi's slope, so the v2
   wrap conditioning and chain rules carry over verbatim.
2. Fold: wrap M to [-pi, pi], solve on |M| (E odd in M; keep the sign).
3. Markley starter — port kepler.py `_solve_sincos_E` lines 71-85,
   cbrt via `metal::precise::powr(c, 2/3)` initially (E2 replaces it).
4. One 5th-order refinement + Taylor rotation — port lines 87-107
   exactly, including the sE = E - sin E conditioning. This is the
   kernel's ONE sincos: it *replaces* v2's sincos(phi), it does not add
   to it. The eccentric delta over circular is the starter algebra,
   the cbrt, and the refinement (~40 flops + powr) — nothing else.
5. X = a (cos E - e); Y = a beta sin E; u = X cw - Y sw; v = X sw + Y cw.
6. Far-side exit v <= 0 (the cphi <= 0 analog) and out-of-transit exit
   z >= 1 + r, both **store-and-return exactly as v2** (finding 1).
   z2 = u^2 + (v ci)^2 needs no forward clamp.
7. Photometric core: `_CORE`/`_PHOT` fragments unchanged.

### VJP kernel: verified chains + simd reduction

Forward recompute (as v2), then per point, with ctz = ct * dF/dz from
the existing photometric partial fragments (which also yield gr, gu1,
gu2):

    g_u = u/z;  g_v = v ci^2 / z;         dz/dci = v^2 ci / z
    g_X = g_u cw + g_v sw;  g_Y = -g_u sw + g_v cw
    dz/dsw = -Y g_u + X g_v;  dz/dcw = X g_u + Y g_v
    dz/da  = z / a                          (exact)
    dzdE = a (-g_X sinE + g_Y beta cosE);  D = 1 - e cosE
    dz/dM = dzdE / D;   dz/dM_tra = dz/dM
    dz/de = dzdE sinE / D - a g_X - (a e sinE / beta) g_Y
    dz/dt0_off = dz/dM * (-2 pi / P)
    dz/dp_off  = dz/dM * 2 pi ((-kk - n_w) P - tau_w) / P^2

The last two are the v2 wrap chains verbatim (n_w locally constant).
The backward 1/z is the only place needing the z floor. The M sign-fold
needs **no** sign term: differentiate implicitly at the final signed
(sinE, cosE) exactly as `kepler._kepler_E_vjp` does — dE/dM = 1/D is
even in M. (Revision 1 over-weighted this; the conditioning hazard
above is the real backward risk.)

**Reduction — Design B (finding 1).** No per-point grids, no atomics,
no threadgroup memory, no barrier, no padded grid:

- Exact grid (m, n, 1), threadgroup (256, 1, 1), early exits unchanged.
- After the 11 partials are in registers: `metal::simd_sum` each; the
  first active lane (`metal::simd_is_first()`) stores the 11 values to
  `part[y, p, x/32]`.
- Output (n, 11, ceil(m/32)) with **`init_value=0.0`** — a simdgroup
  whose lanes all exited writes nothing and must read as zero. (Do NOT
  put init_value on the forward (n, m) output.) 92 MB at 1024 x 65,536
  versus 2.95 GB of naive grids; the fill is trivial.
- In-graph: `mx.sum(part, axis=2)`, then autodiff chains the per-chain
  grads back through the in-graph transforms. x2d gets `mx.zeros_like`.

`simd_sum` re-associates: a two-level tree of the same shape as
ChunkedGaussianLogLike's fp32 accumulation argument. Parity tolerances
account for it.

## Milestones and gates

**E0 — simd reduction on the circular v2 VJP.** *Independent of v3;
ship immediately.* Replace the 7 grid stores in `_MODEL_VJP_TAIL` with
Design B behind the same custom_function, keeping the grid path behind
a `reduce="grid"|"simd"` switch (A/B measurement + a parity oracle:
identical per-point partials, two reductions). Gates: (a) grads match
the grid path to re-association tolerance, adjudicated vs fp64 graph
grads; (b) `mx.get_peak_memory` around a value+grad shows the 7 (n, m)
transients gone; (c) `profile_vjp_reduction.py` on a rested machine:
backward >= 1.1x the grid path (expect 1.16-1.41x); (d) a permanent
test pinning `simd_sum`-after-`return` semantics (the spike, as a
regression guard against compiler changes). All existing tests green.

**E1 — eccentric forward kernel + formulation decision.** Decide
transit-anchored vs floor (measure the anchored variant's gradient
conditioning with `v3_kh_grad_conditioning.py` adapted to it). Build
`_ORBIT_ECC`; dispatch: frontend fp32-GPU eccentric -> v3, e == 0 ->
v2, fp64/CPU -> graph. Gates: max|dflux| <= 5e-7 vs the graph path for
**e <= 0.9** over the sweep below (exceedances adjudicated vs fp64);
e in {0.99, 0.999} as fp64-adjudicated stability cases with tolerance
scaled by the condition number 1/(1 - e cos E); batman eccentric parity
through the frontend at batman's 2e-8-limited tolerance; **forward
<= 1.5x the circular kernel** (<= ~31 ms at 1024 x 65,536; the graph
path's 567 ms makes any relative-to-graph gate meaningless).

**E2 — cbrt + divide triage.** Bit-trick + two-Newton cbrt (starter
needs ~1e-4); `fast::divide` where parity allows. Gates: E1 gates
unchanged; |dE| <= 5e-7 at e = 0.999 held; orbit-time gain recorded.

**E3 — eccentric VJP kernel** on E0's reduction with the chains above.
Gates: all 11 grads match graph autodiff / FD on a boundary-heavy sweep
(contacts, slivers, conjunction, apastron transit); gradient accuracy
sweep in e down to the floor (or, anchored, down to 1e-8) with relative
error <= 1e-3; zero NaN gradients on the 100k boundary sweep;
value+grad under `mx.compile(mx.grad(...))`; determinism (bitwise);
**value+grad <= 2x the circular kernel**.

**E4 — integration + release.** `make_ecc_transit_flux(period_ref,
core=...)` anvil target (10 params, in-graph transforms, q_e
reparameterization, barrier wrapper, ParamSpec bounds incl. the
secosw/sesinw unit disk); constraint gate (no unphysical state gets a
finite penalty-free log-prob); frontend `use_metal` eccentric routing;
benchmarks (bench_vjp eccentric rows, batch scaling), docs (sampler
guide's eccentric note from "planned" to real), CHANGELOG + 0.3.0 tag.

**E5 (optional) — end-to-end eccentric ChEES** injection-recovery at a
low-e truth (e ~ 0.02, where finding 2 bites): zero divergences, truth
recovery, ESS/s vs stretch at matched cores.

## Test matrix (beyond re-running the parameterized suites)

- e in {0, 1e-6, 1e-3, 0.3, 0.7, 0.9, 0.99, 0.999} x w in {0, pi/2,
  pi, 3pi/2}, including w = 3pi/2 with e ~ 0.99 (near-apastron transit,
  the old conic-collapse case) and e = 0 exact parity vs the v2 kernel.
- Gradient conditioning sweep in e (finding 2) — the gate above.
- M near 0, +/-pi, 2pi and wrap ties (rint half-to-even).
- Bounds: m in {1, 31, 32, 33, 255, 256, 257, 1000, 65536} (partial
  simdgroups and partial threadgroups); n_chains in {1, 2, 7, 8, 9,
  1024} (constant-address-space path).
- Reduction: all-out-of-transit (partials exactly zero via
  init_value), all-far-side, mixed; `simd_sum`-after-`return` pin.
- Constraints: unphysical (a, e, r) and (b, e, w, a) states are
  rejected/penalized, never finite-and-wrong.
- NaN/Inf contract; dispatch routing (fp64/CPU -> graph, e == 0 -> v2);
  double-sliver precedence unchanged.

## Risks

- **Register pressure** — an *estimate*, not a measurement: the v2 VJP
  was reasoned to run ~50-70 live registers (MLX exposes no occupancy
  counters). The Kepler solve + 11 partials may approach the ~100
  step. Mitigate by staging partials and recomputing sinE/cosE-derived
  terms; detect via throughput vs the forward kernel; last resort a
  two-kernel split sharing the recompute (costs traffic — avoid).
- **Conditioning at low e** (finding 2) — the dominant backward risk;
  handled by the E1 decision and the E3 gate.
- **Constraints** (finding 3) — a correctness risk for E4, not the
  kernel; handled by the reparameterization + barrier + gate.
- **e -> 1 corner** — contract e <= 0.999; sqrt(1-e) gradients blow up
  only beyond the clamp; parity scoped to e <= 0.9.
- **Python-scalar minting** (the 4.6e-9 b_tra lesson) — every in-graph
  transform branches isinstance / uses math.* for float inputs.
- **JIT latency** — the larger eccentric VJP source lengthens the
  first-call Metal compile (seconds, cached thereafter); note in docs.

## Performance expectations

At 1024 x 65,536 versus circular v2 (20.8 ms fwd / 35.9 ms VJP):
because the eccentric kernel *replaces* v2's sincos(phi) with
sincos(E), the extra cost is only the starter, cbrt and refinement —
expect forward ~1.2-1.4x circular before E2 and closer to 1.2x after;
value+grad <= 2x circular. Against the compiled graph path (567 ms /
68.6 s) that is ~20x forward and >1000x value+grad. Transients stay
at ~92 MB instead of 2.95 GB, which is what lifts the gradient
batch-size ceiling; E0 alone delivers 1.16-1.41x on circular backward.
Batch scaling should stay flat (register-resident, no memory cliff).

## Adversarial review log (2026-09-27)

Tested rather than argued; scripts in `benchmarks/v3_*.py`.

1. **Predicated exits / threadgroup memory / barrier / padded grid were
   unnecessary** — `simd_sum` over active lanes after `return` is
   correct and specified. Design B adopted; E0 decoupled from v3.
2. **(sqrt(e) cos w, sqrt(e) sin w) has an fp32 gradient noise floor**
   (5% at e = 1e-5, 40% at 1e-6); revision 1 guarded only the value.
   Transit-anchored formulation preferred, floor as fallback, gate added.
3. **Joint constraints had nowhere to live** (anvil: box ParamSpecs
   only); where-guards would yield finite-but-wrong values. q_e
   reparameterization + barrier wrapper + gate added to E4.
4. **5e-7 parity gate unattainable at high e** for non-bug reasons
   (dE/dM up to 1000); scoped to e <= 0.9 with condition-scaled
   adjudication above.
5. **">= 5x the graph path" gate was unmeasured and turned out lax by
   ~4x** (graph: 567 ms); gates re-pinned to the circular kernel.
6. Sign-fold risk downgraded (implicit rule at the final signed E has
   no fold term). 7. "Adds one sincos" corrected (it replaces one);
   forward expectation revised 1.5-2x -> 1.2-1.4x. 8. Register count
   labeled an estimate. 9. E0 keeps a `reduce=` switch for A/B and
   parity.
