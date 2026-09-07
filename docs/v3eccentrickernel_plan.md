# v3 eccentric model kernel + in-kernel VJP reduction — combined plan

Two pieces of work, planned together because they share their hardest
constraint: the v3 fused kernel for eccentric orbits (forward + analytic
VJP), and the two-stage in-kernel gradient reduction that replaces the
per-point partial grids + `mx.sum` pattern. They are combined
deliberately — the reduction requires predicated (non-returning) early
exits, which the v3 VJP must be written around from the start, and the
reduction matters *more* for v3: the eccentric VJP has 11 per-point
partials, so the naive grid pattern would write ~44 B/pt (2.95 GB at
1024 x 65,536) versus v2's 28 B/pt.

## Grounding measurements (do not re-derive; scripts cited)

- v2 circular kernel, 1024 x 65,536, uncontended
  (`benchmarks/profile_vjp_reduction.py`): forward 20.8 ms; VJP kernel
  30.6 ms; + 7 sums 35.9 ms; sums alone 5.4 ms at 345 GB/s (DRAM peak).
  The VJP kernel is compute-bound; reductions are ~15% of the VJP, not
  the externally claimed 45%. Reduction payoff: 1.16-1.41x backward,
  ~1.2x value+grad — the stronger win is removing the transient grids
  (the gradient batch-size cap in docs/sampler-integration.md).
- Kernel economics (docs/eccentric-kernel-notes.md): Cartesian tail
  +13-16% over the true-anomaly tail; the Markley starter's cbrt via
  `precise::powr` is ~220x a multiply and ~20% of orbit time (THE
  hotspot); divides ~44x a multiply under safe math (`fast::divide`
  ~9x); `precise::sin/cos` ~84x.
- Solver robustness: Markley + one 5th-order refinement holds
  |dE| <= 4.1e-7 rad at e = 0.999 down to M = 1e-7. Contract: e <= 0.999
  (clamped in-graph).
- Backward formulas verified to ~1e-30 vs finite differences
  (docs/eccentric-kernel-notes.md "Verified backward-pass formulas").

## Design

### Parameterization and the in-graph / in-kernel split

Follow the v2 precedent (q_to_u and df0 stay in the MLX graph so their
Jacobians ride autodiff; the custom_function boundary sits at the kernel
inputs).

Sampler-facing vector for the new anvil target (10 parameters):

    v = (t0_off, p_off, r, b, a, q1, q2, secosw, sesinw, df0)

In-graph, per chain (all differentiable; where-guarded denominators per
the masked-division lesson):

- (q1, q2) -> (u1, u2) — as v2.
- e = secosw^2 + sesinw^2; (cw, sw) = (secosw, sesinw) / sqrt(e) with
  the denominator where-guarded at e = 0 (0/0 there; substitute
  (1, 0)). Clamp e <= 0.999.
- ci = b (1 + e sw) / (a (1 - e^2)) — impact parameter at inferior
  conjunction to cos(inc); guard (1 - e^2).
- M_tra: mean anomaly at inferior conjunction. Port
  `kepler.mean_anomaly_offset_at_transit` to mx ops (it is
  currently host-side math.*): f0 = pi/2 - w, E0 = 2 atan2(sqrt(1-e)
  sin(f0/2), sqrt(1+e) cos(f0/2)), M_tra = E0 - e sin E0. Smooth in
  (e, w); autodiff handles it — no custom VJP. (atan2/sqrt are
  fp32-exact ops; per-chain cost is negligible.)

Kernel inputs: x2d (2, m) epoch-centered (dt, k) rows exactly as v2,
plus per-chain arrays t0_off, p_off, r, ci, a, e, sw, cw, u1, u2, M_tra
(11), plus `period_ref` and m as scalars (runtime args, never baked into
source). beta = sqrt(1 - e^2) is recomputed per thread (one sqrt) so
that e remains the only eccentricity input and the verified dz/de
formula (which folds dbeta/de) applies unchanged.

### Forward pipeline (`_ORBIT_ECC` fragment)

Reuse the v2 wrap verbatim (metal.py `_ORBIT`, lines ~298-301): tau,
n_w = rint(tau/P) (`metal::rint`, NOT round — half-to-even), phi. Then:

1. M = phi + M_tra[y]. Because M is linear in t with the same slope as
   phi, the v2 wrap conditioning and its chain rules carry over
   verbatim; fp32 long-baseline guarantees are inherited.
2. Fold: wrap M to [-pi, pi], solve on |M| (E odd in M; keep the sign).
3. Markley starter — port kepler.py `_solve_sincos_E` lines 71-85
   exactly, cbrt via `metal::precise::powr(c, 2.0f/3.0f)` initially
   (E2 replaces it).
4. One 5th-order refinement + Taylor rotation — port lines 87-107
   exactly, including the sE = E - sin E conditioning. ONE sincos.
5. X = a (cosE - e); Y = a beta sinE; u = X cw - Y sw; v = X sw + Y cw.
6. Far-side exit: v <= 0 (the cphi <= 0 analog). z2 = u^2 + (v ci)^2 —
   no clamp needed forward (sum of squares); out-of-transit exit
   z >= 1 + r with the v2 operand-order convention.
7. Photometric core: `_CORE`/`_PHOT` fragments unchanged.

Early exits in v3 are **predicated, not returning** (store the result,
set an inactive flag, fall through to the reduction barrier) — see
below.

### VJP kernel: verified chains + two-stage reduction

Forward recompute (as v2), then per point, with ctz = ct * dF/dz from
the existing photometric partial fragments (which also yield the pure
photometric gr, gu1, gu2):

    g_u = u/z;  g_v = v ci^2 / z;         dz/dci = v^2 ci / z
    g_X = g_u cw + g_v sw;  g_Y = -g_u sw + g_v cw
    dz/dsw = -Y g_u + X g_v;  dz/dcw = X g_u + Y g_v
    dz/da  = z / a                          (exact)
    dzdE = a (-g_X sinE + g_Y beta cosE);  D = 1 - e cosE
    dz/dM = dzdE / D;   dz/dM_tra = dz/dM
    dz/de = dzdE sinE / D - a g_X - (a e sinE / beta) g_Y
    dz/dt0_off = dz/dM * (-2 pi / P)
    dz/dp_off  = dz/dM * 2 pi ((-kk - n_w) P - tau_w) / P^2

The last two are the v2 wrap chains verbatim (n_w from rint treated
locally constant). The backward 1/z is the only place needing the z
floor (max(z, 10 eps) as in the graph path). The M sign-fold must be
chained consistently (E odd in M: the solve returns sign-folded
sinE; dz/dM at the folded point picks up the sign — pin with tests at
M ~ 0 and +/-pi).

**Reduction (the fix).** No per-point grids, no atomics:

- Dispatch a **padded grid**: ceil(m/256)*256 threads in x (a change
  from v2's exact-m grid) so every threadgroup is full; predicate
  x < m. Out-of-bounds and inactive (early-exit) lanes contribute 0.0
  to every partial and still reach the barrier.
- Stage 1: `simd_sum` each of the 11 partials across the 32-lane
  simdgroup; lane 0 writes to threadgroup memory float[8][11] (352 B).
- `threadgroup_barrier(mem_flags::mem_threadgroup)`.
- Stage 2: simdgroup 0 reduces the 8 rows; one store per group per
  partial. Output shape (n, 11, ngroups): 11.5 MB at 1024 x 65,536
  versus 2.95 GB of naive grids.
- In-graph: `mx.sum(partials, axis=2)` (trivial), then autodiff chains
  the per-chain grads back through the in-graph transforms
  (sesinw/secosw, b -> ci, M_tra, q -> u) automatically. x2d gets
  `mx.zeros_like` (times non-differentiable, as v2).

simd_sum re-associates the sum — this is a two-level tree, the same
shape as ChunkedGaussianLogLike's fp32 accumulation argument, so
gradient accuracy should improve, not degrade; parity tolerances below
account for re-association either way.

## Milestones and gates

**E0 — reduction retrofit on the circular v2 VJP** (de-risks the fix in
isolation; ships value on its own). Swap `_MODEL_VJP_TAIL`'s 7 grid
stores for the padded-grid + predication + two-stage reduction pattern
behind the same custom_function. Gates: (a) gradient parity with the
shipped path <= few-ulp-scale re-association tolerance, adjudicated vs
fp64 graph grads; (b) `mx.get_peak_memory` around a value+grad shows the
7 (n, m) transients gone; (c) `profile_vjp_reduction.py` measures
backward >= 1.1x (expect 1.16-1.41x). All 103 existing tests green.

**E1 — eccentric forward kernel.** `_ORBIT_ECC` + dispatch: frontend
fp32-GPU eccentric routes to v3; e == 0 keeps routing to v2 (no Kepler
solve, strictly faster); fp64/CPU falls back to the graph path as
today. Gates: max|dflux| <= ~5e-7 vs the graph eccentric path over the
e/w sweep below, exceedances adjudicated against fp64 (the batman-test
pattern); batman eccentric parity through the frontend at batman's
2e-8-limited tolerance; forward >= 5x the compiled graph eccentric path
at 1024 x 65,536.

**E2 — cbrt + divide triage.** Replace `precise::powr` cbrt with a
bit-trick + two-Newton cbrt (starter needs only ~1e-4); audit divides
for `fast::divide` where the parity gate allows. Gates: E1 parity gates
unchanged; |dE| <= 5e-7 at e = 0.999 held; measured orbit-time gain
recorded (expect up to ~20% of orbit time back).

**E3 — eccentric VJP kernel** with the E0 reduction infrastructure and
the verified chains above. Gates: all 11 grads match graph autodiff /
FD on a boundary-heavy sweep (contacts, slivers, conjunction, apastron
transit); zero NaN gradients on the 100k boundary sweep; value+grad
under `mx.compile(mx.grad(...))` (engine step shape); determinism
(same input twice, bitwise).

**E4 — integration + release.** `make_ecc_transit_flux(period_ref,
core=...)` anvil target (10 params, in-graph transforms, ParamSpec
bounds incl. e <= 0.999 and the secosw/sesinw unit disk); frontend
`use_metal` eccentric routing; benchmarks (bench_vjp eccentric rows,
batch scaling), docs (sampler guide's eccentric note updated from
"planned" to real), CHANGELOG + version 0.3.0 + tag, per the release
convention.

**E5 (optional) — end-to-end eccentric ChEES** injection-recovery
(10-param, engine progress ticks, bounded max_leapfrog): zero
divergences, truth recovery, ESS/s vs stretch at matched cores.

## Test matrix (beyond re-running the parameterized suites)

- e in {0, 1e-6, 0.3, 0.7, 0.9, 0.99, 0.999} x w in {0, pi/2, pi,
  3pi/2} — including w = 3pi/2 with e ~ 0.99 (near-apastron transit,
  the old conic collapse case) and e = 0 exact parity vs the v2
  circular kernel.
- M near 0, +/-pi, 2pi and wrap ties (rint half-to-even).
- Padded-grid bounds: m in {1, 255, 256, 257, 1000, 65536};
  n_chains in {1, 2, 7, 8, 9, 1024} (constant-address-space path).
- Early-exit predication: all-out-of-transit, all-far-side, mixed —
  gradients exactly 0 for inactive contributions, no barrier hangs.
- NaN/Inf contract: z-producing inputs NaN -> NaN, +inf -> flux 0.
- Dispatch: fp64 and CPU-stream route to graph; e == 0 routes to v2.
- Double-sliver precedence unchanged (photometric fragments untouched).

## Risks

- **Register pressure**: v2 VJP runs ~50-70 live registers; adding the
  Kepler solve + 11 partials may approach the ~100-register occupancy
  step. Mitigate by staging (compute-and-accumulate partials in phases,
  recompute sinE/cosE-derived terms rather than holding them); detect
  empirically via throughput vs the forward kernel; last resort is a
  two-kernel split sharing the recompute (costs traffic — avoid).
- **Barrier + divergence**: predicated exits keep out-of-transit lanes
  alive to the barrier; their cost is bounded by the reduction share
  (~15% worst case, measured). If the all-out-of-transit fast path
  regresses badly, add a per-group early-out AFTER a first barrier
  (all-inactive groups store zeros and exit as a group — safe).
- **Sign-fold chain** (E odd in M) is the most error-prone backward
  detail; it is pinned by dedicated M ~ 0 / +/-pi gradient tests before
  anything else builds on E3.
- **e -> 1 corner**: contract stays e <= 0.999 (in-graph clamp);
  sqrt(1-e) gradients blow up only beyond the clamp.
- **Python-scalar minting** (the 4.6e-9 b_tra lesson): every in-graph
  transform must branch isinstance / use math.* for float inputs.

## Performance expectations (honest)

At 1024 x 65,536 versus circular v2 (20.8 ms fwd / ~36 ms VJP measured
by the profiler): the orbit solve adds one sincos, the starter algebra,
and (until E2) the powr cbrt — expect eccentric forward within
~1.5-2x of circular initially, ~1.3x after E2. Value+grad target:
<= 2x circular (i.e. <= ~105 ms), and >= 5x the compiled graph
eccentric path. The reduction keeps VJP transients at ~11.5 MB instead
of 2.95 GB, which — not the ~1.2x speed — is what lifts the gradient
batch-size ceiling; E0 alone delivers 1.16-1.41x on circular backward.
Batch scaling should stay flat (register-resident, no memory cliff),
matching v2's ~55k curves/s pattern at a lower absolute rate.
