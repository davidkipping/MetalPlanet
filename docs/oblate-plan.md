# Plan: oblate planets (hybrid limb darkening)

**Status: Stage 0 complete and reviewed for cost (2026-10-10); Stages 1-4
not started.** Target: v0.13.0 for the fp64 reference and
`flux_dev_from_tau`. Stage 0 says go for fp32 kernels (Stage 3); the cost
review found a verified 2-2.5x saving and shapes Stages 1 and 3 around three
regimes (see "Cost review").

## Goal

The transit of an oblate planet, seen in projection as an ellipse, for the
hybrid laws (`hybrid2`, `hybrid4`, `hybrid5`, and any `HybridLaw`).
Gradients come in every parameter, and the `ld_basis` columns come with the
same contract as the spherical path. MetalPlanet stays a forward model:
sampling, priors and the f/θ degeneracies belong to the caller (turin,
anvil, SquishierPlanet).

There are two new parameters. **f** is the flattening, f = 1 − B/A.
**theta** is the sky angle of the long axis, measured from the along-orbit
direction. **r** becomes the area-equivalent radius r_eff, so the semi-axes
are A = r/√(1 − f) and B = A(1 − f), with A·B = r². This is SquishierPlanet's
parameterisation (`docs/upstream/metalplanet_oblate_prompt.md`, a draft
brief, and `squishierplanet/jaxlc.py`).

## The design decision: one API, two kernel families

Decided 2026-10-10 after measurement:

- **`f=None` (the default) runs today's code**: spherical kernels and graphs
  for every law, untouched and bitwise identical.
- **`f` given runs a separate oblate path**, for the hybrid laws only.
  `limb_dark="quadratic"` (or polynomial) with `f` given raises.
- **Both paths share everything that isn't photometry**: the entry points,
  the orbit templates (the oblate kernels are generated from the same tau
  templates by asserted substitution), exposure handling, the `ld_basis`
  contract and `HybridLaw` definitions. Spherical and oblate therefore
  cannot drift apart on a law.

Why not a single always-oblate mode, where spherical is just f = 0? Measured
against SquishierPlanet's fp64 reference (scratchpad `oblate/jax_side.py`,
`mp_side.py`):

| | oblate reference (JAX) | MetalPlanet spherical |
|---|---|---|
| flops per point (XLA count), hybrid2 / hybrid5 | 17k / 52k | ~a few hundred |
| transcendentals per point, hybrid2 / hybrid5 | 1.0k / 2.4k | ~10-20 |
| fp64 CPU time per point | 11-24 µs | 58-121 ns |
| cost at f = 0 | unchanged (branch-free) | -- |

**Accuracy is not the obstacle.** At f = 0 the reference matches the
spherical closed forms to 2e-16. As f falls the difference shrinks smoothly,
about 1e-3·f, with no noise floor down to f = 1e-12.

**Cost is the obstacle.** The JAX reference is a generic, unoptimised
transliteration: complex eigenvalue solves and every branch evaluated. An
optimised kernel still has to do, per point, one quartic solve per pole plus
the limb intersections (four for hybrid5), and tens of complex logarithms.
Estimated cost is 10-30x the spherical kernel; Stage 0 measures it.
Always-oblate would charge every spherical call that cost.

**The quadratic law is not elementary for an ellipse** (its μ term needs
elliptic or Carlson integrals), so a spherical path is needed regardless.

## Invariants

- **`f=None` is bitwise today's MetalPlanet.** That covers the 296-array
  release corpus, the kernel snapshot (146 arrays: outputs and VJPs for
  every spherical kernel family) and the hybrid graph snapshot (120
  arrays). No spherical kernel source changes; the oblate kernels are new
  sources.
- **One law, one definition.** The oblate path takes its poles and shapes
  from `HybridLaw`. Its columns are `[E0, T_1..T_n]` in deviation form, and
  `flux - 1 == (B @ c) / (N @ c)` with `N = hybrid_norms(law)`, as the
  spherical path.
- **f → 0 continuity.** At f = 1e-6 the oblate path agrees with the
  spherical one to ~1e-15 in fp64. The fp32 target is set in Stage 0, since
  the brief's 1e-8 is a request, not yet a measured possibility.

## Stage 0 -- spike: results (done, 2026-10-10)

The measurement code is in `benchmarks/oblate_spike/` (throwaway; its README
maps scripts to questions). The truth is SquishierPlanet's fp64 reference
(`jaxlc`). Errors are relative to each column's unocculted (full-disk) flux,
which is how a light curve sees them. The machine's CPU was loaded
throughout, so GPU timings use 2^20-point launches in isolated, interleaved
processes; read the ratios as +/-30%.

**1. fp32 is viable -- go.** These numbers are from SquishierPlanet's
algorithm run in fp32, and from the spike kernel.

| set | configurations | fp32 worst error |
|---|---|---|
| transit chords, f >= 1e-5 (r 0.01-0.3, f to 0.4) | 2,760 | 3.1e-7 |
| SquishierPlanet stress classes, planet-like | 450 | 3.1e-7 |

The stress classes include 168 pole-circle tangencies (near-double roots),
the limb tangencies, and hybrid5's innermost pole, eps = 0.0016. That is the
spherical fp32 kernels' level. Two things are required for it:

- **The near-double pair's Bairstow step changes the value, not only the
  gradient.** Without it, the pole-tangency cases reached 2.8e-5.
- **Small f must take the spherical forms.** As f -> 0 the quartic's leading
  coefficient S2 ~ r^2 f / 2 vanishes and two roots run to 0 and infinity.
  In fp32, f = 1e-7 gave errors to 6e-5 near the contacts, and f = 0 at an
  exact tangency failed outright. So the fp32 kernel routes **f < 1e-5 to
  the spherical closed forms**, where ignoring f costs <~ 1e-3 f <= 1e-8 in
  flux. fp64 needs no switch: there the reference matches the spherical
  forms to 2e-16 at f = 0 and stays smooth (~1e-3 f) down to f = 1e-12.

**2. Quartic solver: Aberth.** The quartic is solved by Aberth iteration
seeded from the circle limit: the two finite roots of the f = 0 quadratic,
plus -S2/conj(S1) and its inverse conjugate. It runs at most 25 iterations
with an early exit, then 2 Newton polishes. In fp32 it matches LAPACK's
fp32 eigenvalues: p99 <= 3e-5 per root, worst 8e-4 on near-tangent limb
roots, and no failures on 2,760 x 7 quartics. With 6 or 12 iterations it
fails on a small tail at f = 0.1-0.4. Rejected:

- **Ferrari closed form:** ill-conditioned on these polynomials, wrong even
  in fp64.
- **A two-root self-inversive Newton:** unreliable.

Root-finding is not the cost: capping Aberth at 6 iterations saves only 4%.

**3. Cost: about 100x the spherical kernel per in-transit point.** These are
from the spike forward kernel (no VJP), after hoisting the arc-endpoint
exponentials:

| law | spherical kernel | oblate spike | ratio |
|---|---|---|---|
| hybrid2 | 0.7-1.0 ns/pt | 74 ns/pt | ~75-110x |
| hybrid4 | 0.9-1.4 | 127 | ~90-145x |
| hybrid5 | 1.3 | 168 | ~130x |

- **Breakdown.** The limb roots, arcs and even moments cost 21-34 ns; each
  pole costs ~50-64 ns (partial fractions, logs, pair handling).
- **Bound by registers and complex arithmetic.** Metal caps the spike at
  384 threads per threadgroup against 1,024 for the spherical kernels, so
  it holds far more registers. Threadgroup size makes no difference; fast
  math saves 20%.
- **Expect 2-3x from a production kernel**, not 10x: smaller register
  footprint, fewer complex divisions, kept-arcs-only loops. That leaves
  roughly 40-150x spherical per in-transit point.
- **Out-of-transit points exit early** (centre beyond 1 + A) and are nearly
  free.
- **Against SquishierPlanet's CPU reference** (~24 us/pt, fp64 JAX), the
  unoptimised GPU spike is already ~140x faster.

**4. Housing both paths: keep them separate.** A combined kernel ran
spherical chains 1.0-1.6x slower than the plain spherical kernel, which
confirms separate kernel families for f=None. In the other direction, a
spherical branch inside the oblate kernel cost oblate chains nothing
(0.99-1.00x). So the small-f spherical route in (1) is free.

**5. Exposure contacts: solve per chain, per side.** The solve brackets
each contact between those of the circumscribed (radius A) and inscribed
(radius B) circles. It then runs 40 bisections on the boundary's min/max
distance from the star's centre, from 64 samples plus 3 Newton steps. The
iteration count is fixed, so it is graph-friendly. Over 4,000 random
geometries it found every outer (3,584) and inner (2,624) contact, with no
existence mismatches and timing error <= 1.4e-9 s.

- **Ingress and egress need separate solves:** a tilted ellipse transits
  asymmetrically, by 164 s for f = 0.3, theta = 1 rad.
- **True contacts are needed.** With them, the 5-node contact rule
  converges spectrally: 9e-8 at n_gl = 5, 6e-9 at 9, 5e-10 at 15 (30-min
  exposures). With spherical r_eff splits it stalls at 3e-7 to 4.5e-6;
  with no splits it is 2-3e-5.

**6. MLX has no complex128.** Its only complex type is complex64, so the
fp64 graph carries complex numbers as pairs of fp64 real arrays with
hand-written complex helpers.

## Cost review (2026-10-10): savings without loss of precision

Prompted by the ~100x in-transit cost. Each candidate was tested against
SquishierPlanet's fp64 reference and, where it is a kernel change, timed in
the spike (`benchmarks/oblate_spike/savings.py`, `symmetry.py`,
`spike_kernel.py`). Accepted items change Stages 1 and 3 below.

**Accepted**

1. **A closed-form fast path for the fully-inside regime** (the ellipse
   entirely on the disc: |centre| + max(A, B) < 1). The contour is the
   whole ellipse, so the even moments are 2 pi times the zero-frequency
   Fourier coefficient of h_n(s) K -- three short expressions, no arc sums
   -- and each pole is -(pi/p) Re sum of N/D' over the roots inside the
   unit disc: a root solve and two residues, no arc layout, no logarithms,
   no pair handling. Verified exact against the reference on 400 inside
   configurations (even 5e-17, poles <= 1.5e-15 of the unocculted flux);
   the fp32 kernel version is at 1.6e-7, the general path's level. Timed on
   inside points:

   | law | general path | fast path | spherical kernel |
   |---|---|---|---|
   | hybrid2 | 42 ns/pt | 10 ns/pt | 1.1 |
   | hybrid5 | 106 ns/pt | 29 ns/pt | 2.4 |

   Inside points are 60-80% of a transit's in-transit points (T23/T14,
   60% on the Stage 0 chords; ~80% for a central r = 0.1 transit), so this
   alone is 2-2.5x off the in-transit average. The near-double-root
   conditioning when the ellipse approaches the limb from inside (within
   ~sqrt(1 + eps) - 1 of it) is the same as the general path's at that
   geometry; Stage 1 adds a stress test for that band.
2. **Restrict to at most two limb crossings.** A planet whose radius of
   curvature at the ends of its short axis is below the star's,
   A^2/B <= 1, i.e. r_eff <= (1 - f)^(3/2) (0.35 at f = 0.5, 0.59 at
   f = 0.3), crosses the limb at most twice. The topology is then
   exactly the spherical one -- outside, partial, inside -- with one
   ellipse arc and one star arc, so the five-slot arc machinery, its
   sorts and its per-slot loops go, and with them registers. Outside that
   domain the entry points raise. No planet is excluded.
3. **Early exit outside** (|centre| - max(A, B) >= 1), before any root
   solve; free. The spike has none, so its "general" timings include
   out-of-transit points at full cost.
4. **Fast-math transcendentals** keep the precision budget: 2.9e-7 /
   3.4e-7 against 3.1e-7 with precise math, on the chords and stress sets.
   Worth 20%.
5. **Warm-start each pole's root solve from the previous pole's roots**
   (the K + 1 quartics differ only in their z^2 coefficient): 20% off the
   hybrid5 fast path (28.7 -> 22.7 ns/pt) at 3.8e-7. Optional; take it in
   Stage 3 if the budget holds.

**Tested and rejected**

- **Using only the two roots inside the unit disc by self-inversive
  symmetry.** The roots do come as mirror pairs or on |z| = 1, but a
  mirror pair's two arc contributions do not share a real part (worst
  relative difference 35 on 1,779 cases), and in the partial regime most
  pole quartics have unit-modulus roots anyway (81% for eps = 0.0016,
  40% for the large poles). No saving there.
- **Perturbation in f around the spherical forms** for moderate f: the
  first-order error is ~1e-3 f^2, 1e-5 at f = 0.1. Loses precision.
- **Numerical quadrature along the arcs** in place of the partial
  fractions: cheap away from the pole circles, but the near-tangency cases
  that the pair formula handles exactly are where quadrature fails. Loses
  precision.
- Fewer Aberth iterations, a looser exit, or dropping the pair's Bairstow
  step: all cost accuracy (Stage 0).

**Revised budget.** In-transit average for hybrid5, from the spike: 0.7 x
29 + 0.3 x 170 ~ 70 ns/pt against 170 today (2.4x), i.e. ~30x the spherical
kernel; with the production-kernel work (register footprint, the
two-crossing simplification, fast math) an estimated 15-25x. Out-of-transit
points are free. The remaining cost is the partial regime's partial
fractions and logs, which have no cheaper exact form that we found.

**Where the rest of a fit's cost lives.** Per evaluation the oblate model is
~20x spherical on in-transit points; the number of evaluations is the
caller's: collapsed limb darkening (SquishierPlanet's path B) removes 4-5
sampled dimensions and was measured there at 10-100x in ESS/s, which
outweighs the kernel ratio.

## Stage 1 -- fp64 reference in MetalPlanet (medium-large)

New module `metalplanet/oblate.py`, an MLX graph in fp64 and fp32 that
mirrors `hybrid.py`.

- **Geometry.** (x0, y0) is the planet centre in the planet's principal
  frame, from the sky position (X along orbit, Y across) and theta.
  Semi-axes A and B come from (r, f); r_eff <= (1 - f)^(3/2) is enforced.
- **Three regimes, as masks, mirroring the kernel** (cost review): outside
  (zero), inside (the closed forms: 2 pi q_n0 and inner residues) and
  partial (one ellipse arc, one star arc). The kernel is then validated
  regime by regime against this graph.
- **Intersections** (partial regime only). The ellipse ∩ unit circle
  quartic, solved with Stage 0's solver, plus Newton polish; exactly two
  crossings, so the arc layout is one ellipse arc and one star arc.
- **Complex arithmetic** on (re, im) pairs of fp64 arrays (finding 6); the
  quartic solver is Stage 0's Aberth, shared in spirit with the kernel.
- **Even moments** (μ⁰, μ², μ⁴) as closed-form Fourier sums along the kept
  ellipse arcs, plus h(1) times the stellar-limb arc angles.
- **Pole moments**, by partial fractions over the roots of D(z), with logs
  tracked continuously along the arcs. A near-double pair is integrated as
  one quadratic factor (the (σ, π) form and the J(u) series), and is never
  polished.
- **Columns `[E0, T_1..T_n]`** through the law's generator matrix, exactly
  as `hybrid.shape_cols`. `flux_dev_oblate(x0, y0, r, f, w, law)` and
  `shape_cols_oblate(...)` are the public graph functions, under the
  0.10.7 data contract (`fp64_on_cpu`, `as_data`).
- **Gradients by autodiff.** Arc endpoints are frozen (stop_gradient) and
  carry jaxlc's corner term; root derivatives are implicit (one in-graph
  Newton or Bairstow step). This is the same structure the Stage 3 VJP will
  hand-code.

Tests (`tests/test_oblate.py`):
- Against SquishierPlanet when importable, at ≤ 1e-12: its planet-like
  configuration classes, limb tangencies, the pole-tangency stress set,
  r ∈ [0.01, 0.3], f ∈ [0, 0.5], theta ∈ [0, π); plus the inside/limb
  boundary band (|centre| + A within 1e-3 of 1), where the inside closed
  forms meet the partial arcs.
- Against an mpmath area integral at a handful of points.
- f → 0 against `hybrid.shape_cols`.
- The `ld_basis` identity.
- Autodiff against central finite differences away from contacts, in
  x0, y0, r, f, theta and every weight.
- No NaN in values or gradients over a boundary sweep, including f = 0,
  theta at 0 and π/2, and exact tangencies.

### Stage 1 results (done, 2026-10-10)

`metalplanet/oblate.py` and `tests/test_oblate.py` (40 tests, all green;
full fast suite 1856). The spherical paths are bitwise unchanged: the
light-curve, kernel and hybrid-graph snapshots (296, 146 and 120 arrays) and
the 18 kernel sources all equal v0.12.2's.

| check | fp64 | fp32 |
|---|---|---|
| SquishierPlanet reference, 6 stress sets (630 configs, all columns) | 2.8e-15 | 4.0e-7 |
| inside/partial seam, ±1e-9 .. ±1e-3 | < 1e-12 | |
| quadrature oracle (scipy) / mpmath at 30 digits | < 1e-11 / < 1e-12 | |
| autodiff vs central FD (x0, y0, r, f, θ, w) | ≤ 8e-8 rel | |
| f just below the switch vs spherical | 1e-16 | |
| f just above the switch vs spherical | 4e-13 (the flattening's own ~4e-3·f) | |

Findings:

1. **The analytic Aberth seeds stall at large f.** With two roots near the
   unit circle, both of Stage 0's seed sets (circle-limit, rotated
   biquadratic) could converge onto the wrong pair. A crossing pair was
   missed (error 2e-2), or a pole level kept a spurious root (5e-5). The
   graph now seeds Aberth with companion-matrix eigenvalues (MLX `eigvals`,
   complex64, CPU stream only) and runs 8 iterations. It then reaches
   rounding everywhere the reference does. A near-double pair stays at
   ~sqrt(eps), which is harmless because the pair formula integrates it
   through (σ, π) after a Bairstow step.
   **Consequence for Stage 3:** a Metal kernel has no `eigvals`.
   *Resolved in Stage 2* (decision: write our own): the graph now uses its
   own balanced, Wilkinson-shifted complex QR on the 4×4 companion, in
   elementwise operations, which Stage 3 transcribes. The stress sets
   above, with large-f near-limb configurations in particular, are the
   gate. Stage 0's fp32 viability numbers stand,
   because they were measured where the seeds converged.
2. **Shared root solves.** Each pole level is solved once per lane and
   shared by the inside and partial branches. Lanes neither branch owns
   take host-precomputed roots of a safe geometry.
3. **A centred planet gave NaN in d/dx0** through the spherical branch's
   sqrt(x0² + y0²), which is evaluated on every lane. The separation is now
   floored, as the orbit floors it.
4. **Pair selection** gathers by an argmin over scores. Its indices must be
   stop-gradient, or MLX refuses the VJP.
5. **Cost** (reference only): ~22 µs/pt compiled in fp64; fp32 runs eager
   at ~25 µs/pt. The fp32 graph cannot be `mx.compile`d, because it exceeds
   Metal's argument-buffer limit for a fused primitive.

## Stage 2 -- `flux_dev_from_tau` and the orbit, fp64 graph (medium)

- **API.** `flux_dev_from_tau(..., limb_dark=law, u=w, f=f, theta=theta)`
  and the `ld_basis=True` form, with f and theta as scalars or per-chain
  `(n,)` arrays, like the orbit parameters. Eccentric orbits use `secosw`
  and `sesinw` as now.
- **Sky position.** Circular orbits give X = a sinφ and Y = −b cosφ;
  eccentric (anchored) orbits give the (u, v·cos i) the orbit code already
  computes. The theta convention and Y's sign are pinned against
  SquishierPlanet's `sky_circular` and `generator_lightcurves` by a test.
  The far side is masked as now.
- **Exposures.** Supersampling, and the contact rule with Stage 0's oblate
  contacts, solved per chain and per side (finding 5) and detached, as
  today's spherical contacts.
- **The data contract and calling-form matrix** gain oblate rows
  (`tests/test_entry_contract.py`).

### Stage 2 results (done, 2026-10-10)

`metalplanet/oblate_tau.py`, the `f=`/`theta=` keywords on
`flux_dev_from_tau`, and `tests/test_oblate_tau.py` (51 tests) plus three
oblate rows in `tests/test_entry_contract.py`. Fast suite: 1961 passed,
3 skipped (the oblate rows' kernel-dispatch checks: no kernel yet). The
spherical paths are bitwise v0.12.2's (all four gates).

1. **Own eigensolver.** Balanced (4 passes), Wilkinson-shifted complex
   Hessenberg QR on the companion matrix: 6 sweeps, deflate, 5 sweeps,
   deflate, then the 2×2 in closed form; Aberth (8) and Newton (2) polish
   as before. (5, 4) sweeps already equal (12, 8) on the stress sets.
   Generator columns unchanged: 2.4e-15 fp64, 4.4e-7 fp32 on the stress
   sets; fp32 against fp64 at f from 1.01e-5 to 1e-2 at most 4.3e-7 (the
   companion is worst conditioned there). Raw QR seeds are rounding-level
   for f >= 2e-5 except near-double pairs (~sqrt(eps), as expected).
2. **Sky frame.** Pinned to SquishierPlanet: X = a sin φ, Y = −b cos φ
   (circular); X = −u, Y = −v cos i (anchored eccentric). The plan's
   earlier "Y = b cos φ" had the sign of Stage 0's spike, which passed −θ
   to SquishierPlanet to compensate. Light curves equal
   `squishierplanet.model.light_curve` to 5e-16 (circular and e = 0.3,
   0.6, tilted).
3. **Contacts per side.** q (squared distance to the disc) and M (squared
   largest boundary distance) are convex along the chord; Newton from the
   outside of each root is monotone. Outer contacts start at the
   circumscribed circle's, inner ones at the outer ones; the slope comes
   from the envelope theorem. Against a dense independent reference (240
   configurations, 130 in the grazing band, half eccentric): 2.3e-15 rad
   fp64, 4.4e-7 rad fp32, no missed or spurious contacts, including a
   tilted graze with both outer contacts before conjunction.
4. **Exposures.** 5-node contact rule, 30-minute exposures: ~1e-7, the
   rule's own floor (vs 40 nodes); splitting at the spherical r_eff
   contacts instead costs 4e-7 to 5e-6; 25-point supersampling 2e-5 to
   8e-5. f = 0 equals the spherical path to 2e-17 under all three rules.
   fp32 graph within 1.1e-7 of fp64.
5. **Gradients** in tau, period, a, b, r, f, theta, w, secosw and sesinw
   match central differences (1e-6 instantaneous and supersampled; 2e-6
   for the contact rule at 40 nodes, where finite differences move the
   frozen split points and the two derivatives agree to the rule's error:
   1e-5 at 12 nodes). Finite on a sweep through centre crossings, f = 0,
   theta at 0 and pi/2, and grazes, in both precisions.
6. **Domain check under mx.compile.** Reading an mx.array back inside a
   trace poisons it even when the error is caught. Host values still
   raise; array arguments give NaN outside the domain instead.
7. **fp32 GPU compile of the graph** still exceeds Metal's argument
   buffers (an MLX limit); the contract checks its compile on the CPU
   stream until the kernels take fp32 GPU calls.

## Stage 3 -- fp32 Metal kernels (large; Stage 0 says go)

Stage 0 and the cost review fix the core: three regimes (early exit
outside; the inside closed forms; the partial-regime arcs with two
crossings), Aberth roots with the pair's Bairstow step in the value path,
f < 1e-5 through the spherical closed forms in-kernel, fast-math
transcendentals, and optionally warm-started pole roots.
`benchmarks/oblate_spike/spike_kernel.py` (with `inside_fast=True`) is a
validated forward starting point at 3.1e-7. Optimisation targets are
register footprint, the complex-division count and the two-arc layout.
Budget: ~15-30x the spherical kernel per in-transit point, averaged over a
transit.


- **An oblate photometry device function in `metal_oblate.py`**, the
  `_HYB_FN` analogue: the per-law constants come from `HybridLaw`, and
  complex arithmetic is in `float2`.
- **Tau kernels generated from the orbit-generic templates** by `_swap`.
  The orbit plug-ins emit (X, Y) as well as z, and the gradient slots gain
  f and theta.
- **A hand-written VJP**, mirroring Stage 1's structure: frozen endpoints
  plus a corner term, and implicit root derivatives. It must cover tau,
  period, a, b, r, f, theta and every weight. In the inside regime it is
  closed-form too: the residues' derivatives through the implicit root
  derivatives, no corner term.
- **A basis variant** for `ld_basis=True`, with cotangent pre-contraction
  as `mp_hyb_cols_bd`.
- **Kernel caching per (law definition, orbit, basis, oblate).**
- **f < 1e-5 chains** take the spherical forms (free for oblate chains;
  needed for fp32 accuracy). Below the switch, d/df is zero rather than the
  true ~1e-3. The candidate fix is a first-order term in f from the
  spherical arc moments. Decide in this stage whether samplers need it.

Tests: kernel against the fp64 graph per column (gates from Stage 0's fp32
report); VJP against fp64 autodiff; continuity sweeps through tangencies and
near-double roots; per-chain independence; and every spherical kernel
snapshot bitwise.

### Stage 3 results (done, 2026-10-10)

`metalplanet/metal_oblate.py`, `tests/test_oblate_metal.py` (41 tests); the
three oblate rows of `tests/test_entry_contract.py` now reach their own
kernel family (dispatch spy, GPU compile). Fast suite 2003 passed. All four
bitwise gates equal v0.12.2.

1. **The VJP is a template, not hand-derived.** The post-solve evaluation
   is written once as `ob_eval<T>`: T = float is the forward; T = `dv`
   (a value with four tangents, d/d(x0, y0, A, B)) is the VJP's per-point
   Jacobian -- the derivative of exactly the forward's function, with the
   graph's frozen pieces (roots, regimes, arcs, pairs) frozen, one live
   Newton or Bairstow step, and the corner term. The orbit, the principal
   frame and (r, f) -> (A, B) chain by hand. MSL templates and operator
   overloading work in `mx.fast.metal_kernel`.
2. **Accuracy.** Device columns vs the fp64 graph on SquishierPlanet's
   stress sets and random f in [2e-5, 0.5]: <= 7e-7 of each column's
   full-disk flux, all laws. Tau kernels vs the fp64 graph: <= 7e-8 in
   flux (all rules, orbits, laws; basis 5e-7). VJP vs fp64 autodiff in
   tau, period, a, b, r, f, theta, w, secosw, sesinw: <= 1e-3 of each
   gradient's scale, typically 1e-5; pole tangencies 3e-4. Two fp32
   limits: within ~1e-6 of a limb contact a sliver's crossings fall inside
   the pair tolerance (gradient ~1e-4 lost there, flux ~1e-13); d/df near
   f = 1e-5 is ~1% (a small difference of d/dA and d/dB), falling with f.
3. **f < 1e-5: decided, no first-order term.** Spherical forms, d/df = 0:
   the flattening moves the flux by < 4e-8 there and the band is 2e-5 of
   a uniform prior on [0, 0.5]. Pinned by a test.
4. **Fixes found by the kernel, applied to the graph too.** (a) Arc
   choice: the ellipse arc by the midpoint of its LONGER arc, the star arc
   as the shorter one (always, since A < 1); the old midpoint-of-short-arc
   test flipped on rounding at an internal graze (error = the whole star).
   (b) Inside poles: the two inner residues summed as one quadratic factor
   (`_pair_sum`); separately they cancel ~1e3 for a nearly centred, nearly
   round planet and lose fp32 derivatives (0.1 -> 2e-6).
5. **Cost** (loaded machine, interleaved; 128 chains x 4000, half in
   transit). Root solving was ~90% of the kernel. Applied: one QR solve per
   point, other levels warm-started (Aberth with early exit; residual +
   Vieta sum and product check, QR fallback); early QR deflation; fast
   transcendentals in the detached solve only (fast log/atan2 in the live
   part cost 4% in pole-tangency gradients); a detached Bairstow polish of
   each pair before the live step (warm-started roots converge linearly on
   near-double pairs; without it pole-tangency gradients were 4e-2).
   Contacts: one thread per chain (`contact_offsets_oblate_kernel`), 1.1 ms
   a call for 128 chains vs 18-42 ms as a graph.

   | hybrid5 | spherical | oblate | ratio |
   |---|---|---|---|
   | one evaluation per point, forward | 0.55 ns/pt | 12.6 | 23x |
   | one evaluation per point, value+grad | 0.90 | 22.9 | 26x |
   | contact rule (25 nodes), forward | 4.9 | 346 | 71x |
   | contact rule, value+grad | 9.1 | 621 | 69x |

   hybrid2: 13-14x / 67-75x. The contact-rule ratio is large because the
   spherical hybrid kernel costs only ~0.2 ns per node; the oblate one
   costs ~14 ns per node (~28 per in-transit evaluation, under the cost
   review's ~70 estimate). Further cuts would come from the QR solve
   (still the largest single cost) -- e.g. seeding the first level from a
   neighbouring node's roots, which the kernels do not share today.

## Stage 4 -- frontend, docs, release (small-medium)

- **`TransitParams.f` and `.theta`**, both optional and defaulting to None,
  i.e. spherical. `TransitModel` routes through the oblate graph or kernels
  when f is given. `light_curve`, `light_curves` and `light_curve_mx` all
  apply.
- **A README section**: parameterisation, laws, accuracy, cost relative to
  spherical, and the fp32 report.
- **`benchmarks/bench_oblate.py`**, subprocess-isolated and interleaved, run
  on a quiet machine.
- **A reply to SquishierPlanet** against its brief. The brief asks for
  bitwise agreement with MetalPlanet 0.10.7 at f=None; 0.12.x is the
  baseline now, since hybrid2's pole has changed.

### Stage 4 results (done, 2026-10-10; release pending)

- `TransitParams.f`, `.theta` (degrees); `TransitModel` routes an oblate
  model through `flux_dev_from_tau(f=, theta=)` -- one compiled function
  per orbit type, tau = time since each point's own mid-transit; the
  spherical graphs and a spherical model's batch keys are untouched.
  `tests/test_oblate_api.py` (20 tests): SquishierPlanet's
  `model.light_curve` to 1e-13 (measured 8e-16, circular and e = 0.3,
  0.6, times near zero; at absolute BJD the time itself carries ~5e-10 d,
  which showed as 9e-11), f = 0 vs spherical 2e-15, fp32 vs fp64 3e-7
  under every exposure setting, batched = looped, light_curve_mx = 
  light_curve, gradients vs Richardson differences in f, theta, rp, t0,
  inc, ecc, w and a weight (<= 6e-5; the contact rule's gradients agree
  to its quadrature error, as in Stage 2), the error paths.
- README section "Oblate planets"; layout rows; `benchmarks/bench_oblate.py`,
  re-run on a quiet machine: 16-37x the spherical hybrid kernels with one
  evaluation per point, 77-85x under the contact rule.
- Cross-code benchmark `benchmarks/oblate_compare/` against squishyplanet,
  JoJo and GreenLantern (built for macOS OpenCL 1.2 with compatibility
  shims), mirroring the spherical one: precision vs a 30-digit integral
  (central transit and a graze), single-curve times to 10^7 points, a
  512-set batch, value + gradient. RESULTS.md there.
- The spherical hybrid pole fix (inner contact, large r), on request.
- Reply to SquishierPlanet: `../SquishierPlanet/docs/upstream/
  metalplanet_oblate_reply.md` (not committed there). It corrects our
  earlier hybrid note: SquishierPlanet's spherical pole moments are right
  near the inner contact at r = 0.8, ours are off by up to 1e-10 of the
  column norm (1/sqrt(distance) growth; every r <= 0.5 at rounding).
  Fixed in this release on request (CHANGELOG, "Fixed").
- Not done here: the version bump, commit, push and tag (on request).

## Risks

- **fp32 near-double roots:** resolved by Stage 0 (Bairstow in the value
  path; 3.1e-7 on the pole-tangency stress set).
- **Register pressure:** measured at 384 threads per threadgroup, against
  1,024 spherical; it is part of the ~100x. Its effect on the VJP kernel,
  with more live state, is unmeasured.
- **Cost for samplers.** An oblate fit costs ~15-30x a spherical one per
  in-transit point on the same GPU (after the cost review's savings).
  Collapsed limb darkening (SquishierPlanet's path B) is the lever on the
  number of evaluations.
- **The VJP is the largest hand-derivation in the codebase.** Stage 1's
  autodiff graph is its oracle at every step.
- **Contact times** for an ellipse need a per-chain solve. A wrong split
  costs quadrature accuracy and breaks the frozen-split gradient, as on the
  eccentric path before v0.8.1.
- **Conventions.** Theta's zero, Y's sign and the r_eff definition must
  match SquishierPlanet exactly; a test pins all three.

## Out of scope

- The quadratic law and odd polynomial terms for an oblate planet (not
  elementary).
- Elliptical stars, moons, ring systems.
- Second derivatives.
- Samplers, priors, the collapsed log-density, and the f/θ degeneracy
  handling (caller's).
- Oblate anvil sampler targets: a separate decision, as for the hybrid laws.

## Effort

| Stage | Size | Notes |
|---|---|---|
| 0 | small | prototypes and measurements; decides 3's shape |
| 1 | medium-large | the algorithm, fp64, with autodiff gradients |
| 2 | medium | orbit, exposures, entry point |
| 3 | large | kernels and a hand VJP; conditional on Stage 0 |
| 4 | small-medium | frontend, docs, benchmark, reply |

Stages 0-2 make a useful release on their own: an exact, differentiable
fp64 oblate forward model at CPU speed. Stage 0 is done.
