# Changelog

All notable changes to MetalPlanet. Versioning: semantic-ish
(MAJOR.MINOR.PATCH); every release is tagged `vX.Y.Z` in git.

## [Unreleased]

## [0.10.0] — 2026-10-08

### Added
- **Hybrid limb-darkening laws, stage 1 of 3: closed forms and the fp64
  reference** (`metalplanet/hybrid.py`). SquishierPlanet's hybrid2,
  hybrid4 and hybrid5 -- even powers of mu plus double-pole terms
  1/(mu^2 + eps)^2, written in a shape basis whose weights w are each the
  share of the centre-to-limb drop a term carries -- as
  `flux_dev_hybrid(z, r, w, law)`, the shape-basis columns `shape_cols`
  (the `ld_basis` contract `flux - 1 == (B @ c)/(N @ c)`, `c = (1, -w)`,
  `N = hybrid_norms`), their analytic partials `shape_partials`, and the
  uniform priors on the physical regions (`ld.hybrid2_from_q` on the
  exact triangle, `ld.simplex_from_q` by stick-breaking, with inverses).

  For a spherical planet the mathematics simplifies: every column is
  elementary. The even powers come from s_0, s_2 and the ALFM19 even
  recursion M_0 -> M_2 -> M_4 (never the elliptic M_1, M_3); a pole term's
  occulted flux is kap1/(p eps) plus an atan or log of rational functions
  of (z, r) (Green's theorem with h = 1/(2p(p - s))), with a series
  bridging the removable singularity on z + r = sqrt(1 + eps). Unlike the
  quadratic law's mu^1 term, **no column needs an elliptic integral**.
  Every sqrt(depth) dependence is routed through the core's `kite`, so
  the three cancelling terms of the pole form see one rounded lens near
  the contacts (without that, fp32 lost 1e-6 there).

  Measured: against a 30-digit direct integration, the pole columns agree
  to 1e-13 of their norm (1e-12 within 1e-6 of a contact, the geometry's
  own conditioning), the laws to 1e-13; against SquishierPlanet's ellipse
  code at a = b, 1e-12; the even columns match the cel3 route to 1e-14.
  Autodiff matches finite differences to 1e-7 in z, r and every weight;
  the analytic partials match autodiff to 1e-11 (except at MLX's max/min
  ties z == r and z == 1, where autodiff splits the gradient and the
  analytic partials are the ones that match FD). fp32 stays within 1e-6
  of fp64 at every simplex vertex. 90 tests (`tests/test_hybrid.py`).
  Existing paths bitwise unchanged (296 arrays).
- **Hybrid limb-darkening laws, stage 2 of 3: `TransitModel`.**
  `limb_dark="hybrid2" | "hybrid4" | "hybrid5"` with `params.u` the shape
  weights (2, 4 or 5 of them), on every frontend path: `light_curve`,
  `light_curves` (sequence and array-valued, per-set weights), the
  contact and supersample rules, eccentric orbits, and `light_curve_mx`
  differentiable in each weight (FD to 1e-6; the every-field gradient
  test now runs over the three laws). Secondary eclipses ignore limb
  darkening as before. The laws use the polynomial law's vector-law
  plumbing: `_n_poly` is `_n_vec`, the graph branches read `vec`, and the
  one law-specific line is the dispatch in `_photom`. They run on the
  MLX graph (fp32 within 2e-6 of fp64), never the fused kernel, as the
  polynomial law does. 34 tests (`tests/test_hybrid_api.py`); existing
  paths bitwise unchanged (296 arrays).
- **Hybrid limb-darkening laws, stage 3 of 3: fused fp32 kernels**
  (`metalplanet/metal_hybrid.py`). `flux_dev_from_tau(..., limb_dark=,
  u=)` and `flux_dev_metal(..., limb_dark=, u=)` for all three laws:
  circular and eccentric orbits, every exposure rule, analytic VJPs in
  tau, the period, a, b, r, every weight (shared `(n_w,)` or per chain
  `(n, n_w)`) and secosw/sesinw; `ld_basis=True` returns the
  `(n, m, 1 + n_w)` shape-basis columns with the contract
  `flux - 1 == (B @ c)/(N @ c)`, `c = (1, -w)`. The tau kernels are
  derived from metal.py's orbit-generic templates by asserted
  substitution (the quadrature, its tau derivative, the contact split and
  the simd_sum reduction stay one copy); the device functions put
  hybrid.py's closed forms in registers, per law by substituting its
  poles, shape matrix and norms into one template. The fp64 graph path
  computes the same function.

  Measured: fp32 kernel within 1.2e-8 of the fp64 graph on every law,
  rule and orbit (gates 5e-7, 2e-6 for hybrid5); the fp64 graph path
  equals `TransitModel` to 1e-12; kernel VJPs match fp64 autodiff at the
  quadratic kernel's gates; the basis VJP equals the scalar VJP through
  the identity to 1e-5. At 512 x 5,000 with the contact rule: hybrid2 is
  0.50x quadratic's forward time and 0.60x its value+grad, hybrid4
  0.62x / 0.82x, hybrid5 0.81x / 1.02x (`benchmarks/bench_hybrid.py`) --
  no elliptic integral. 183 tests (`tests/test_hybrid_metal.py`).

  hybrid.py's series bridge across the Q = 0 line now has six terms and,
  in fp32, a 0.05 half-width (1e-3 in fp64), so the closed-form partials'
  near-cancellation there never costs fp32 more than ~1.3 digits; the
  kernels use the fp32 rule. Quadratic kernels: all 18 sources
  byte-identical, outputs bitwise unchanged (296 arrays); 1119 green.

## [0.9.7] — 2026-10-04

Fixes from a seventh code review (of 0.9.6).

### Fixed
- **0.9.6 broke the array-valued `light_curves` form with a per-set `u`
  (regression).** `light_curves(params)` with array attributes takes a
  per-set `u` of shape (n_sets, N) (README), and 0.9.6 routed it through
  the single-set normaliser, which demands 1-D: it raised
  `u must be a 1-D vector; got shape (3, 2)`. On 0.9.5 it ran and matched
  the loop to 4e-16. The bitwise corpus never held that case, which is
  why it passed. That branch now normalises `u` to (n_sets, N) itself
  (`_u_vector(..., sets=True)`) and compares N -- not the number of sets,
  which 0.9.5 had counted, wrongly, for a polynomial law -- against the
  model order; a mismatched set count or a 3-D `u` is a clear error.
- **A mixed `u` list no longer borrows a sibling's dtype.** 0.9.6
  converted the Python entries of a list holding an `mx.array` scalar to
  that scalar's dtype: `[mx.array(1), 0.4]` turned 0.4 into int 0 (flux
  2e-3 off, no error), and an fp32 entry rounded its neighbours before
  the model cast. Each entry is now cast on its own, to the model dtype,
  as 0.9.5 did. A nested non-mx entry beside an mx one gets the shape
  error rather than an opaque `TypeError`.
- **`u` is normalised once per entry point.** `_check_law` returns the
  normalised vector and `light_curve`, `light_curve_mx` and the
  sequence form of `light_curves` thread it through; 0.9.6 normalised two
  or three times per call on the sampler-facing path.
- `TransitModel.__init__` derives the polynomial order through the same
  normaliser, so a wrong-shaped `u` is rejected at construction too.
- A test compared `f is m.light_curves`, which is always False (a bound
  method is a fresh object per access), so the `light_curves` branch it
  meant to cover was never exercised. It compares by name.

4 new tests, including the array-valued per-set `u` case for quadratic,
polynomial and linear laws against the looped result; 809 green.
`light_curve`, `light_curves` and `flux_dev_from_tau` remain bitwise
identical to 0.8.2 (296 arrays).

## [0.9.6] — 2026-10-04

Fixes from a sixth code review (of 0.9.5).

### Fixed
- **0.9.5's 2-D `u` fix covered only an `mx.array`.** A numpy (3, 1)
  column (a `loadtxt` slice, say) or a nested list `[[.4], [.25], [.05]]`
  reproduced the identical bug on 0.9.5: a (3, 601) curve of a wrong-order
  model, 8.8e-4 off, with no error, on a release whose notes said the
  case now raises. Every container -- list, tuple, numpy, `mx.array`, a
  list holding `mx.array` scalars -- now goes through one normaliser,
  `_u_vector`, which returns one 1-D vector or raises
  `u must be a 1-D vector`. That was the root cause: each container type
  had been validated by its own code path.
- **The shape is judged before the count.** `_check_law` runs first in
  both entry points and in `light_curves`, and it now normalises `u`
  before counting coefficients. So a 0-d `mx.array` `u`, which died in
  `list()` with `IndexError: SmallVector out of range`, and a (1, n) `u`,
  which got a misleading "order changed" message (a message 0.9.5's test
  had codified), both get the shape error.
- A quadratic `u` with size-1 numpy entries (a (2, 1) array) ran through
  numpy's deprecated array-to-float conversion, which a future numpy
  turns into an opaque `TypeError`; it is now rejected with the same
  shape error.
- The up-front scalar loop uses `_BATCH_KEYS` rather than restating it.
- The scalar-field test matched the first letter of a field name, so it
  could not tell `per`, `rp` and `fp` apart; it anchors the full name.

The `u` shape test now runs over all three containers, on
`light_curve_mx`, `light_curve` and `light_curves`. 6 new tests;
805 green. `light_curve`, `light_curves` and `flux_dev_from_tau` remain
bitwise identical to 0.8.2 (296 arrays).

## [0.9.5] — 2026-10-04

Fixes from a fifth code review (of 0.9.4).

### Fixed
- **A 2-D polynomial `u` ran a wrong-order model.** A whole-array `u` was
  cast without a shape check (0.9.4's guard was per entry), so a (3, 1)
  `u` passed `_check_law`'s count of three, took `flux_dev_poly`'s
  batched branch and returned a (3, 601) curve of a one-coefficient
  model, 8.8e-4 off, with no error. A whole-array `u` must now be a flat
  vector.
- **Shape validation is one pass, before any routing.** 0.9.4's notes
  said "one chokepoint"; it was three sites (`cast`, a manual check for
  t0, another for w), and `fp` and the whole-array `u` had none. All
  nine fields and `u` are now checked in one loop at the top of
  `_model_eval`, so a route that never reads a field still rejects a bad
  one and the `w`-on-a-circular-orbit special case is gone.
- **What the kernel serves is stated once.** `_kernel_usable()` now
  answers only the device question (fp32, Metal, GPU stream, switched
  on); which graphs the kernel serves (primary transits, quadratic limb
  darkening, no contact rule) is decided in `_get_compiled`, beside the
  branches it governs, and nowhere else. The three tests that pinned the
  old answer pin the cache key instead.
- A test docstring still claimed Python and array `w` give the identical
  result, which 0.9.4 retracted; it now states the 1-ulp invariant its
  body checks.

### Not changed
- `_scalar_value` treats any `ValueError` from reading a 0-d array as
  "traced". The review notes an upstream evaluation failure under
  `mx.grad` would then skip the range check and surface at the caller's
  later `mx.eval` with a less specific traceback. That error still
  surfaces; the previous message-matching guard (removed in 0.9.4 on the
  prior review's advice) traded that traceback for a dependency on MLX's
  exact wording. Left as is.

The scalar-field test now runs on fp32 as well as fp64 and includes
`fp`; 33 new tests; 799 green. `light_curve`, `light_curves` and
`flux_dev_from_tau` remain bitwise identical to 0.8.2 (296 arrays).

## [0.9.4] — 2026-10-04

Fixes from a fourth code review (of 0.9.3).

### Fixed
- **Every field of `light_curve_mx` must be a scalar, not just `ecc`.**
  0.9.3 guarded `ecc` alone: a vector `rp` (or a, inc, w, t0, per, fp, an
  entry of u) still ran, one value per time sample, and on the fp32
  fused-kernel route died with an opaque reshape error. The check is now
  at the one chokepoint every field passes through (`cast`), including
  `w` on a circular orbit, where it is otherwise unread.
- **"Scalar" means shape (), not size 1.** A (1, 1) `ecc` passed 0.9.3's
  check and changed the output's shape to (1, 601).
- **The readable-vs-traced helper no longer matches MLX's error text.**
  With the shape checked first, the traced-eval refusal is the only
  `ValueError` `float()` can raise, so a plain `except` is both simpler
  and robust to MLX rewording its message.
- **The cache key follows `_get_compiled`'s branch structure.** The
  kernel decision enters the key only where a kernel branch is reachable
  (not (e, w), not polynomial, not the contact rule), computed beside
  those branches rather than restated in `_kernel_usable`. A polynomial
  fp32 model, which 0.9.3 still compiled once per stream, now compiles
  once.
- **Docstring: an array `a` on an eccentric orbit is bitwise with
  `light_curve`** (it enters the (k, h) graph as it is); only `inc`,
  `w`, and `a` on a circular orbit combine in-graph.
- **A claim in 0.9.3's notes was too strong.** A Python `w` and an array
  `w` do not give the identical (e, w) result in general: fp32(radians(w))
  differs from fp32(w) * fp32(pi/180) for ~9% of w values (13.5, 27, ...),
  so the fixture's w = 63 agreed by luck. They agree to 1 fp32 ulp, and
  the test now asserts that over w = 63, 13.5 and 27.

24 new tests (every field x three bad shapes x both routes); 766 green.
`light_curve`, `light_curves` and `flux_dev_from_tau` remain bitwise
identical to 0.8.2 (296 arrays).

## [0.9.3] — 2026-10-04

Fixes from a third code review (of 0.9.2).

### Fixed
- **A vector-valued `ecc` ran unvalidated.** The readable-vs-traced test
  caught `ValueError`, and MLX raises that same type for "cannot convert
  a multi-element array", so `ecc = mx.full((601,), 0.3)` returned a
  (601,) curve with each time sample at its own eccentricity. `ecc` must
  now be a scalar (a clear `ValueError` otherwise). The readable-vs-traced
  decision lives in one named helper, `_scalar_value`, which checks the
  shape first and matches MLX's traced-eval error explicitly; a test pins
  that dependency.
- **An fp64 `ecc` just below 1 reached an fp32 graph as exactly 1.**
  The range check read the fp64 host value; the graph received the fp32
  cast, so `ecc = 1 - 1e-9` passed and ran to a flat curve. The check now
  sees the value the graph sees.
- **The kernel decision is no longer in the cache key where it cannot
  matter.** `_kernel_usable()` is False for `integration="contact"` (the
  kernel never served that path), so an fp32 contact model called on
  both streams compiles its graph once, not twice.
- **Docstring overclaim.** `light_curve_mx` matches `light_curve` bit for
  bit when the array fields are among t0, per, rp, fp and u, which enter
  the graph as they are; an array a, inc or w is combined in-graph in
  the model dtype, ~1 ulp from the host fold on fp32. The docstring now
  says exactly that.
- A Python `w` on the (e, w) route is folded to radians on the host in
  fp64 again (0.9.2 did it in-graph), matching the other Python fields.
  Note the review that flagged it attributed the (e, w) route's 2.4e-7
  fp32 difference from `light_curve` to this; it is not. The two routes
  are the same function (4e-16 apart in fp64) computed along different
  fp32 paths, each ~1.8e-7 from fp64 truth, and a Python and an array
  `w` give the identical (e, w) result. A test now pins that.

Not changed: an array `ecc` is read (and so evaluated) under `mx.grad`
to validate it, as 0.9.2 documents. 5 new tests; 741 green.
`light_curve`, `light_curves` and `flux_dev_from_tau` remain bitwise
identical to 0.8.2 (296 arrays).

## [0.9.2] — 2026-10-04

Fixes from two code reviews of 0.9.1.

### Fixed
- **A CPU-stream call could permanently slow `light_curve` on an fp32
  model (0.9.1 regression).** The compiled-graph cache was keyed on
  `circular` alone, while the fused-kernel decision is made at build
  time from the *active* stream. With 0.9.1's shared cache, the README's
  own recipe (an fp64 t0 gradient through `light_curve_mx` under the CPU
  stream) as a model's first call cached a kernel-less graph that every
  later GPU `light_curve` reused: 3.4 ms instead of 1.4 ms at 2e6 points.
  The kernel decision is now part of the cache key.
- **An out-of-range array `ecc` gave NaN with a zero gradient.** 0.9.1
  masked both the input and the output, so a fitter holding e as an
  array saw a NaN loss and a zero gradient, with nothing pointing at the
  parameter, and an eager call with a bad array `ecc` returned NaN where
  0.9.0 raised. Now: wherever the value can be read (eagerly, or under
  `mx.grad`, whose inputs are plain arrays) it is validated and raises
  `ValueError`. Only when traced, under `mx.vmap` or a caller's
  `mx.compile`, is raising impossible, and there the output and every
  gradient are NaN, through a NaN factor rather than a mask. A valid
  point's gradient beside an invalid one is unchanged, which the test
  now checks over a loss summed across all rows.
- **One array field changed the result on an fp32 model.** 0.9.1's
  fixed-e routes formed (k, h) and b in-graph in the model dtype, while
  `light_curve` forms them on the host in fp64 and casts once -- a 1-ulp
  difference, so adding an array `rp` moved the output at 1e-7. Python
  fields are now folded on the host exactly as `light_curve` folds them,
  and the output is `light_curve`'s bit for bit whichever other fields
  are arrays (only `ecc` as an array changes the route).

### Changed
- **One evaluation path.** `_eval_compiled`, `_eval_graph`, `_has_arrays`
  and `_uvec` are replaced by a single `_model_eval` behind both
  `light_curve` and `light_curve_mx`: a Python field is folded on the
  host, an array field stays in the graph. The routing and the
  eccentricity validation (`_check_ecc`, now scalar-aware, with one
  message) live in one place. An `mx.array` vector `u` is cast whole
  rather than sliced and restacked.

### Tests
- fp32 gradients through the fused kernel's custom VJP (Python `ecc`,
  plain mode) and through the fp32 contact / supersample graphs, against
  the fp64 model -- the routes 0.9.1 enabled but never differentiated.
- The kernel route is asserted from the cache key, not inferred from the
  absence of the (e, w) key; the cache-poisoning scenario is a regression
  test. 9 new tests; 736 green.

`light_curve`, `light_curves` and `flux_dev_from_tau` remain bitwise
identical to 0.8.2 (296 arrays).

## [0.9.1] — 2026-10-04

Fixes from a code review of 0.9.0.

### Fixed
- **`light_curve_mx` crashed on vector limb darkening.** It failed with a
  numpy-array `u` (the usual batman style; "truth value of an array is
  ambiguous") and with an `mx.array` vector `u`, the natural way to
  differentiate it whole. `light_curve` and 0.8.2 accepted both. The
  cause was `params.u or []`. Both forms now work, and an `mx.array`
  vector `u` is differentiable.
- **An out-of-range eccentricity could return a plausible curve.**
  - 0.9.0 checked an array-valued `ecc` with `float()` and skipped the
    check whenever that failed, so under `mx.vmap` or a caller's
    `mx.compile`, e = 1.5 returned a flat curve. Such an `ecc` is now
    checked *in the graph*: out of [0, 1) the output is NaN (it cannot
    raise when traced), and the graph sees a safe e, so a valid point's
    gradient stays clean beside an invalid one. This also removes a
    forced evaluation on every call.
  - Separately, and since long before 0.9.0, single-set
    `light_curve(params)` with a Python e >= 1 silently returned a flat
    curve, while `light_curves` raised. Both now raise `ValueError`.
- **One array field moved a fit off `light_curve`'s graph.** In 0.9.0,
  any array-valued field sent `light_curve_mx` through the (e, w)
  eccentric graph, which never uses the fused kernel. Making only `rp`
  an array on a circular fp32 model then lost the kernel. Routing now
  follows `ecc`:
  - a Python 0 takes the circular graph;
  - a Python e > 0 takes the (k, h) graph, with w in-graph (smooth
    there, since sqrt(e) is a constant);
  - only an array-valued `ecc` takes the (e, w) graph.

  At 2e6 points (fp32) the fixed-e routes run at `light_curve`'s speed
  (0.71 / 0.91 ms against 0.75 / 0.91), against 3.0 ms on the (e, w)
  route.
- Dead code: an unreachable `return self._eval(params)` calling a
  deleted method, and a no-op `consts=` in the fused-kernel branch,
  which wrongly suggested that branch serves the (e, w) route.
- `docs/frontend-circular-kernel-plan.md` still called `light_curve_mx`
  the eager path.

`light_curve`, `light_curves` and `flux_dev_from_tau` are bitwise
unchanged for every valid input (296 arrays). 15 new tests; 727 green.

## [0.9.0] — 2026-10-04

### Added
- **`TransitModel.light_curve_mx` is differentiable in every
  `TransitParams` field.** Up to 0.8.2 it read each field through
  `float()`, so a gradient taken through it was silently zero; 0.8.2
  documented that. It now runs the *compiled* model graphs that
  `light_curve` uses, with array-valued fields (t0, per, rp, a, inc, ecc,
  w, fp, `u`) kept in the graph. Python numbers stay constants, and with
  all-Python parameters the output is `light_curve`'s, bit for bit.
  Details:
  - Eccentricity enters as (e, w) through the new
    `anchored.anchor_constants_ew`. (k, h) is the right pair for a
    sampler, but it is singular as a map from (e, w): dk/de is infinite
    at e = 0, so d/d(ecc) taken through it is inf * 0 = NaN, although F
    is differentiable in e there (one-sided). With w given, the anchored
    constants need no sqrt(e) and no division. d/d(ecc) at e = 0 now
    matches a one-sided finite difference, and d/dw is exactly 0 there.
  - `anchor_constants` is now a front end on a shared core;
    `separation_anchored`, `contact_offsets_anchored` and
    `pack_orbit_constants` accept precomputed `consts=`. With the default
    (k, h) inputs every graph is unchanged: `light_curve`, `light_curves`
    and the eccentric `flux_dev_from_tau` are bitwise identical to 0.8.2
    (296 arrays).
  - The (e, w) route uses the compiled graph rather than the fused model
    kernel. The kernel skips its eccentric-only gradient slots on e == 0
    chains, which is exact for (k, h), whose Jacobian vanishes there, but
    not for (e, w).
  - Under `mx.grad`, MLX needs the CPU stream for any float64 input. That
    is MLX's rule, now documented: an fp32 GPU model differentiates fp32
    fields on the GPU, and an absolute t0, which must be fp64, is
    differentiated under `mx.stream(mx.cpu)`.

  Measured: every gradient matches fp64 finite differences to ~1e-7
  (2e-5 gate; 1e-4 for the one-sided e = 0 case under the contact rule),
  across plain, contact and supersampled modes, all four laws, primary
  and secondary eclipses, circular, e = 0.3 and e = 0.7. fp32 gradients
  are within 2e-3 of fp64. A caller can wrap `light_curve_mx` in
  `mx.compile`. Speed is 1.4-3.3x faster than the old eager path at
  100,000 points. The eager internals (`_eval`, `_contact_nodes`,
  `_separation`) are removed, and with them the misleading
  `test_light_curve_mx_gradient`, which differentiated a hand-built graph
  rather than the method. 52 tests in `tests/test_light_curve_mx.py`;
  712 tests green.

## [0.8.2] — 2026-10-04

### Fixed
- **`TransitModel.light_curve_mx` failed on a default model.**
  `TransitModel` defaults to `dtype=mx.float64`, and MLX has no float64 on
  Metal. `light_curve` and `light_curves` build on the model's CPU stream;
  `light_curve_mx` built on the default (GPU) device and raised
  `float64 is not supported on the GPU` unless the caller had already
  switched streams. It now builds on the model's stream too. The result
  is a lazy fp64 array, so the caller's own downstream ops still have to
  run on `mx.cpu`; the docstring now says so. 12 tests cover the default
  fp64 and fp32 models, circular and eccentric, plain, contact and
  supersampled, with no stream context.
- **`light_curve_mx`'s docstring claimed differentiability it does not
  have.** It reads the `TransitParams` fields as Python floats, so a
  gradient taken through it in a parameter is silently **zero**. The
  docstring now says so and points to `flux_dev_from_tau` and
  `metalplanet.anvil`, and a test pins the behaviour so that a future fix
  must update the docs. (`TestDifferentiability.test_light_curve_mx_gradient`
  never exercised it: it differentiates a hand-built graph instead.)

## [0.8.1] — 2026-10-04

### Fixed
- **`TransitModel(integration="contact")` on eccentric orbits now splits
  each exposure at the exact contact times.** It used the linearised
  contacts (`contact_geometry` + `contact_offsets`). Those can miss a kink
  by minutes, which degrades the quadrature: at the default n_gl = 7 the
  worst orbit tested (e = 0.5, grazing-adjacent) was at 1.8e-6 against
  the exact integral, and is now at 3.1e-8. All three contact sites (the
  eager path, the compiled eccentric graph and the batched
  `light_curves`) now call `exposure.contact_offsets_anchored`, as
  `flux_dev_from_tau` does, and the two now agree to 1e-12 in contact
  mode as well. Eccentric outputs change by up to 3.1e-6. Circular output
  is bit-identical: `contact_offsets_anchored` returns the linearisation
  unchanged at e = 0, where it is exact (40 arrays: all three sites, fp32
  and fp64, quadratic and polynomial limb darkening).
- **Grazing transits gave NaN gradients through the contact times**,
  circular included. Where the clip collapses the inner pair, `sqrt` and
  `arcsin` sit at an infinite derivative, and the clip's zero cotangent
  makes 0 * inf = NaN. Only the differentiable frontend graph saw this
  (`flux_dev_from_tau` detaches its contacts, and the public `TransitModel`
  methods return plain arrays), so it never reached a user. The active
  branch now sees a sanitised argument and the collapsed value is
  detached. Values are bit-identical, and the gradients match finite
  differences.

## [0.8.0] — 2026-10-04

### Added
- **Eccentric orbits on `flux_dev_from_tau`** (`secosw=`, `sesinw=`). Both
  are omitted by default, and then nothing changes. Given, the orbit is
  the transit-anchored one (`anchored.py`) and the Kepler solve is the
  model kernel's own, lifted into a device function. So there is one copy
  of the eccentric numerics, and it is exact at e = 0 with fp32-safe
  gradients as e -> 0. `tau` is the time since inferior conjunction and
  `b` the impact parameter there (anvil's algebra), so e = 0 reduces to
  the circular call. Gradients flow in all nine inputs. `ld_basis=True`
  works too, and a batch can mix circular and eccentric chains.

  Measured:
  - The fp64 graph path agrees with `TransitModel` to <= 6e-16 for the
    instantaneous and supersampled rules, e = 0 to 0.7.
  - The fp32 kernel is within 2.5e-7 of that graph.
  - e = 0 through the eccentric path equals the circular path (1.6e-16
    fp64).
  - Cost at 512 x 5,000, contact, n_gl = 5: 2.27x forward and 2.12x
    value+grad relative to circular for e = 0.3. e = 0 chains in the
    eccentric kernel cost 1.68x / 1.65x.
  - 102 tests (`tests/test_tau_ecc.py`); benchmark
    `benchmarks/bench_tau_ecc.py`.

- **`exposure.contact_offsets_anchored`**: exact contact times for the
  anchored orbit, refined from the linearised ones by Newton steps
  through the anchored solve. They agree with bisected roots to 1e-11 d
  and handle grazing geometry. The linearisation `TransitModel` uses can
  misplace a contact by 2.3e-3 d on a grazing e = 0.5 orbit. That makes
  the contact rule 20x less accurate there at n_gl = 5 (2.9e-6 vs
  1.5e-7). It also puts the jump of dF/dtheta *inside* a quadrature
  piece, which costs the frozen-split gradient ~1% on d/dperiod. The
  eccentric tau path uses the exact contacts. `TransitModel` still uses
  the linearised ones (unchanged).

### Changed
- **The tau kernels are one orbit-generic template.** The exposure
  quadrature, its exact tau derivative and the per-chain simd_sum
  reduction are now written once, with the orbit (circular or anchored
  eccentric) as a plug-in that supplies z, dz/dphi and dz/dtheta. The
  circular instantiation is **bitwise identical** to v0.7.0: outputs and
  every gradient, scalar and `ld_basis`, all three rules, fp32 and fp64
  (94 arrays). Its speed is unchanged too (0.998x to 1.001x in an
  interleaved A/B). The four hand-written circular sources it replaces
  are gone.

### Fixed
- **README `ld_basis` timings** were measured on a contended machine.
  Quiet: 14.0 / 13.6 ms forward, 29.5 / 28.3 ms value+grad (scalar /
  basis). The ratios stand (0.97x / 0.96x). The instantaneous-rule
  overhead is 1.12x, not 1.30x.

## [0.7.0] — 2026-10-03

### Fixed
- **`flux_dev_from_tau`'s graph path could not batch chains.** Reported by
  turin, who hit it on the fp64 reference path and worked around it by
  looping chain-by-chain (that workaround can go). With `n_chains > 1` and
  `integration="contact"` or `"supersample"`, any parameter spelling --
  including scalars -- raised `[broadcast_shapes] Shapes (n,1) and
  (n,m,k) cannot be broadcast`. Each exposure rule appends a node axis to
  the times, but the parameters were shaped `(n, 1)` regardless; that
  broadcasts against anything when `n == 1` and against nothing when it is
  not. How many trailing axes the parameters need is a property of the
  rule, so `_tau_graph` now derives it there.

  The fp32 kernel was unaffected -- it indexes parameters by chain rather
  than broadcasting -- which is why it never showed. The test matrix had
  the same blind spot: every fp64 test used a 1-D `tau`, so `n` was always
  1, and the `n > 1` cases were all behind the Metal skip. 60 tests added
  that run the graph at `n` in {1, 2, 4, 33} across all three rules, both
  precisions and both parameter spellings, and check that row `j` is what
  chain `j`'s parameters produce alone -- on chains deliberately made
  distinct, since shapes that broadcast are not automatically shapes that
  broadcast correctly. 455 tests green.

### Added
- **`ld_basis=True` on `flux_dev_from_tau` (and `flux_dev_metal`)**: off by
  default. Requested by SquishierPlanet
  (`../SquishierPlanet/docs/upstream/metalplanet_ldbasis_prompt.md`),
  whose collapsed-limb-darkening target treats (u1, u2) as a linear block.
  It returns the (n, m, 3) basis `B`, where `B[..., j]` is the
  exposure-integrated, unnormalised deficit for intensity mu^j. For any
  quadratic law, with c = (1 - u1 - u2, u1 + 2 u2, -u2) and
  N = (pi, 2 pi/3, pi/2), the scalar call equals `(B @ c) / (N @ c)`.
  The kernel already held the Green's-basis deficits s0d, s1d, s2d, so
  the basis is B = (s0d, s1d, s0d/2 + s2d/4) stored where one collapsed
  number used to be. The VJP takes an (n, m, 3) cotangent and returns
  gradients in tau, period, a, b and r. It contracts the cotangent into
  the core first (ct . B is a scalar function of z, r), so the
  quadrature-derivative logic is shared with the scalar VJP rather than
  copied. The fp64 graph path (`vjp.ld_basis_analytic`, which has an
  analytic VJP) takes the same keyword.

  Measured: identity to <= 1.2e-8 (fp32 kernel) and ~1e-17 (fp64 graph),
  including grazing geometry and r = 0.3, over the whole q-box. Cost at
  512 x 5,000, contact, n_gl = 5: 0.95x forward and 0.97x value+grad
  relative to one scalar call, against 3x for the three calls it
  replaces. `ld_basis=False` is bit-identical to 0.6.1. The scalar
  kernel sources are byte-for-byte unchanged (the basis kernels are
  separate), and outputs and every gradient were compared bitwise against
  the previous commit on both paths. `u1`/`u2` are now optional keywords,
  required only when `ld_basis=False`. 84 tests
  (`tests/test_ld_basis.py`), benchmark `benchmarks/bench_ld_basis.py`.
  538 tests green.

- **`flux_dev_from_tau`** — a `tau`-input entry point with the exposure
  integration *inside* the kernel, requested by
  [turin](../turin/docs/upstream/metalplanet_prompt.md) (Kepler/TESS
  fitting with per-transit mid-times). Two gaps it closes. First, the
  fused model kernel treats times as data and returns a zero gradient for
  them, so a sampler fitting TTVs — a mid-time per epoch, sampled jointly
  with the shape parameters — could not use it; here `tau` is an ordinary
  differentiable input and the VJP returns `d(F)/d(tau)` alongside all six
  parameter gradients. Second, exposure integration existed only on the
  batman-style frontend, never on the path a sampler takes.

  The rule runs per output point in registers, so the sub-exposure axis
  never becomes an MLX array. At 512 chains x 5,000 points x 15
  sub-exposures: 2.8x forward and 3.6x value+grad on the *same*
  arithmetic, with peak memory 2776 MB -> 51 MB (55x), and the contact
  rule (25 evaluations) beats the 15-node expanded route on speed *and* is
  ~1,200x more accurate. Matched to 1e-6 the ratio is 30-57x, because
  supersampling converges as O(1/N) and needs n_sub ~ 2,271 — ~368 GB at
  512 chains. `benchmarks/bench_tau_kernel.py`, `tests/test_tau_kernel.py`.

  The fp64 / CPU / no-Metal graph path computes the same *function*, not
  merely the same value: both freeze the quadrature's split points, so a
  gradient certified in fp64 is the gradient that runs in fp32. (Freezing
  is exact, not an approximation — a split point is interior to a
  continuous integrand, so moving it adds +f(c)dc and -f(c)dc, which
  cancel. The window *ends* are genuine Leibniz boundary terms and the
  kernel carries them.) Getting there needed one fix: `exposure_nodes`
  converts contact *phases* to times with `period / 2 pi`, which leaked a
  period dependence into the split points that the kernel did not have —
  worth 9e-4 relative on `d/d(period)`, 2,000x the fp32 noise floor.
- **`notebooks/02_joint_transit_and_gp.ipynb`** — fitting a transit and
  correlated stellar variability *together*: MetalPlanet as anvil-gp's mean
  model, a SHO Gaussian process over the residual, and anvil sampling all
  twelve parameters at once. The two packages compose with no adapter, because
  `make_quad_transit_flux` already returns exactly the `(v, x) -> (n_chains, m)`
  contract anvil-gp's `mean_fn` wants — which is what exposing
  `x64`/`y_fit`/`model_fn` on `TransitTarget` anticipated. Measured at the
  committed settings: 0 divergences, R-hat 1.001, every parameter within
  1.11 sigma (including the GP amplitude and timescale), ~2 minutes. It also
  records two findings from building it: `max_leapfrog=384`, best for the
  transit-only posterior, is wrong here (128 is right), and *shrinking* the
  light curve to save time made convergence worse, because the GP
  hyperparameters lose their constraint.
- **`notebooks/01_metalplanet_with_anvil.ipynb`** — a tutorial covering
  forward modelling (orbits, limb-darkening laws, finite exposures, batched
  evaluation) and fitting real Kepler/TESS-style photometry end to end with
  anvil. Generated from `notebooks/build_01.py` so the prose and code stay
  reviewable in a diff, and every code cell is executed before commit; the
  committed copy carries no outputs. The sampling cell is sized to finish in
  ~25 s while producing a genuinely healthy fit (0 divergences, R-hat 1.002,
  ~1500 effective samples/s), and it teaches the coupling that produced the
  first draft's failure: 200 warmup iterations at `max_leapfrog=384` gave
  3,120 divergences and R-hat 1.22, where 300 gave zero and 1.002.
- **`make_transit_target(t, y, yerr, t0_guess, period_guess, ...)`** — the
  entry point for real Kepler/TESS photometry, and the piece that was
  missing for production use. It owns the float64 conditioning the
  float32 sampler depends on (time reduction, epoch centering, flux
  offset), the ParamSpec boxes (with `b` reaching past 1 so grazing
  geometries are inside the box, unlike the synthetic targets' 0.9),
  `report_offset` so results come back on the input time system, and the
  eccentric barrier. `TransitTarget.model_params()` builds the
  model-space vector from physical values; `x64`/`y_fit`/`model_fn` are
  exposed for anvil-gp to wrap. `examples/fit_mission_data.py` fits a
  TESS-shaped sector end to end: 40,000 points, zero divergences,
  R-hat 1.002, all eight parameters within 1.7 sigma.

### Fixed
- **float32 models silently corrupted mission time stamps.** The frontend
  uploaded absolute times raw, so a TESS BTJD (~2500 d, fp32 ulp 21 s)
  gave 1.65e-4 flux error — larger than many planet depths — Kepler BKJD
  5.1e-5, and raw BJD 1.2e-2 (unusable). The grid is now re-centred on a
  float64 reference at construction with the offset carried into t0
  everywhere, so the error is **9.9e-7 regardless of the time system**.
  float64 models keep a zero offset and are bit-unchanged.
- **`PenalizedLogLike` raised `RecursionError` under `copy`, `deepcopy`
  and unpickling**: `__getattr__` delegated unconditionally, so lookups
  that precede `__init__` recursed on `base` itself. Checkpointing or
  forking an eccentric target crashed.
- **The eccentric barrier had a plateau outside the e_max disc.** The
  model projects (secosw, sesinw) radially onto the disc, so beyond it the
  likelihood is exactly constant in the radial direction and the previous
  penalty's gradient there was purely tangential — a chain thrown into the
  box corners (e_raw up to 1.8) had no force pushing it back. The barrier
  now also penalises the *unprojected* eccentricity, and every residual is
  dimensionless: `|cos i| - 1` used to explode to ~1e12 with ~1e11
  gradients once the projection drove 1 - e^2 to 0.002, which would
  collapse HMC's step size. A one-sided Huber shape keeps the force
  bounded but never zero.
- **Changing the polynomial limb-darkening order between calls returned
  wrong flux silently** (2.6e-4 off; the extra terms were dropped by a
  `zip`). The order is now validated alongside the law.
- **The batched `light_curves` skipped the single path's validation**:
  `ecc < 0` produced an all-NaN row with only a numpy warning, and a
  quadratic set carrying three coefficients silently used the first two.
  Both now raise, as they already did for `light_curve`.

### Changed
- `light_curves` reuses the device-resident time grid instead of
  re-uploading it per call, the contact geometry
  (`a_sky`, `b_conj`) is computed by one `contact_geometry` helper rather
  than three copies that had already drifted apart, and `_unbroadcast` is
  shared from `vjp.py` instead of duplicated in `kepler.py`.

- `benchmarks/RESULTS.md` MetalPlanet single-curve rows re-measured on a
  quiet machine after the 0.6.0 unification (the circular frontend now
  runs on the fused kernel): fp32 0.74 → 0.42 ms at N = 1e5 and
  1.43 → 1.13 ms at 1e6; unchanged at 1e7 (host-copy-bound); 0.24 →
  0.34 ms at N ≤ 1e4, where the kernel's dispatch floor is slightly above
  the graph path's — a 0.1 ms regression at sizes no GPU should be used
  for. The crossover against batman (between 1e4 and 1e5) is unchanged.
  fp64 rows, an unchanged path, moved < 3%.

## [0.6.1] — 2026-09-28

Quiet-machine re-measurement of 0.6.0, and a correction to how its
headline number was defined.

### Fixed
- **The cost of retiring the circular kernel is 1.06x forward / 1.10x
  value+grad**, not the 1.07x / 1.27x stated in 0.6.0. Two things were
  wrong with the earlier figure: it was measured under GPU contention,
  and its "value+grad" evaluated only the gradients. Our custom VJP
  recomputes from the primals and ignores the forward output, so when
  only gradients are evaluated MLX's lazy graph never runs the forward
  kernel at all and the number silently becomes VJP-only (1.21x on a
  quiet machine). A sampler needs energy *and* force — forward plus VJP —
  which is what every earlier table meant. `bench_ecc_kernel.py`,
  `profile_vjp_reduction.py` and the new `ab_retired_kernel.py` now
  evaluate both. Without the fast paths the same measurement reads
  1.43x / 1.40x; eccentric chains pay 1.01x.
- README kernel figures are now measured, not derived: circular
  21.8 / 60.2 ms, eccentric 29.4 / 76.2 ms at 1024 x 65,536 (3.08 and
  2.28 Gpt/s); eccentric value+grad peak memory 0.39 GB.
- The batched frontend measures **163x** a loop on a quiet machine
  (2,000 sets x 301 points, fp32); the 112x in 0.5.0 was taken under
  load. The batching claim in the sampler guide re-verified at 782x,
  inside its documented range.

### Added
- `benchmarks/ab_retired_kernel.py`: reproduces the unified-vs-retired
  comparison by extracting the pre-0.6.0 `metal.py` from git, fixing its
  imports and renaming its kernels (MLX caches JIT kernels by name, so
  two sources with one name would collide silently).

### Noted
- The frontend's fp32 `light_curve()` gains **nothing** from the kernel
  route at N >= 1e6 (0.99x circular, 1.01x eccentric at 1e7) because it
  is dominated by the float64 host copy — 80 MB at 1e7 — and 2.06x
  (circular) at 1e5 where dispatch overhead matters. The superseded
  `frontend-circular-kernel-plan.md` had a "revert below 1.3x at 1e7"
  gate for a *separate* route; this route is the same kernel and costs
  nothing to keep, so it stays, and the earlier claim of 2.1x for the
  eccentric frontend routing (measured under load) should be read as
  "2x at 1e5, host-copy-bound above".

## [0.6.0] — 2026-09-28

One model kernel. The dedicated circular kernel is retired; circular
stays a first-class *mode* (the 8-parameter anvil target, `ecc = 0.0` in
the frontend) and runs as e = 0 on the transit-anchored eccentric kernel,
which is exact there (7e-16 against the circular closed form).

### Changed
- **Circular orbits run on the eccentric kernel, with per-chain fast
  paths.** A simdgroup-uniform `if (e == 0)` skips the Markley starter
  and refinement in the forward pass, and skips computing, reducing and
  storing the seven eccentric-only gradient slots in the VJP. Measured
  against the retired kernel at 1024 x 65,536, same run: forward
  **1.07x**, value+grad **1.27x** (without the fast paths it would have
  been 1.46x / 1.41x); genuinely eccentric chains pay 0.98x — nothing.
  *Superseded by 0.6.1: measured under contention and with a VJP-only
  definition of value+grad; the corrected figure is 1.06x / 1.10x.*
  Flux parity 4e-7 at e = 0 (fp32 operation order) and bit-identical at
  e = 0.3. That cost (10% on value+grad once measured properly, see
  0.6.1) is the deliberate price of one kernel: half the surface for
  every future photometric fix, and mixed circular/eccentric batches
  (`TransitModel.light_curves`) for free.
- `make_model_core_metal(period_ref)` is now the unified factory taking
  the packed orbit constants; `make_ecc_core_metal` remains as an alias.
  The `reduce="grid"|"simd"` switch is gone with the kernel it belonged
  to — the in-kernel `simd_sum` reduction is the only path (its A/B
  numbers are preserved in `benchmarks/profile_vjp_reduction.py`'s
  docstring).
- The frontend's fp32 GPU *circular* path now routes through the kernel
  too, which is what `docs/frontend-circular-kernel-plan.md` proposed —
  achieved by unification rather than by a second route. That plan is
  marked superseded and kept as the record of the road not taken.
- A "set e = 1e-4" fudge is neither needed nor harmless: measured, it is
  a real model error scaling linearly with e — 3e-6 at e = 1e-4, above
  batman's floor. Use exactly zero.

### Removed
- `_ORBIT` (circular), `_MODEL_VJP_TAIL` (7-slot) and the v2 kernels. The
  retired kernel is recoverable from git history (the commit before this
  release) if a fixed-circular workload ever needs the last 27% back.

## [0.5.0] — 2026-09-27

### Added
- **`TransitModel.light_curves(params_seq)`** — the batched frontend.
  Same time grid, many parameter sets, one dispatch: **112x** a Python
  loop over `light_curve` (2,000 sets x 301 points, float32 GPU). Accepts
  a sequence of `TransitParams` or one whose scalar attributes are
  arrays. Works in every mode (uniform/linear/quadratic/polynomial,
  supersampled or contact-integrated) and handles **mixed circular and
  eccentric sets in one batch**, since the transit-anchored orbit
  degenerates exactly to the circular one at e = 0. This closes the gap
  between what `docs/sampler-integration.md` prescribes — never loop,
  always batch — and what the batman-style API made easy.
- `docs/frontend-circular-kernel-plan.md`: a full plan for routing the
  frontend's *circular* path through the v2 kernel, deliberately **not
  implemented**. Measured reward is only ~1.4x (2.31 vs 3.26 Gpt/s)
  against the highest risk on the backlog, so the design is recorded for
  whoever revisits it rather than executed.

### Changed
- `exposure_nodes` and `flux_dev_poly` accept a leading batch axis, so
  contact integration and polynomial limb darkening both work under the
  batched frontend.
- **ChEES `max_leapfrog` re-measured** (`benchmarks/bench_leapfrog_cap.py`).
  The examples' cap of 24 dated from when gradients cost ~1.2 s; at
  55–76 ms it was the binding constraint on the circular target, where
  384 gives **8.7× the ESS/s (8.2 → 71.2)** and takes R̂ from 1.83 to
  1.00 with no divergences. On the *eccentric* target the same change is
  harmful — ESS/s falls monotonically and 96 steps produces 172
  divergences — so `examples/chees_ecc.py` keeps 24 and says why. The
  guidance, and the caution that ESS is ceiling-limited by
  `n_chains × n_samples`, is in `docs/sampler-integration.md`.

## [0.4.1] — 2026-09-27

Two defects in 0.4.0's new exposure code, found by a closing audit.

### Fixed
- `light_curve_mx` returned the *instantaneous* flux at the exposure
  mid-times under `integration="contact"` while `light_curve` returned
  the averaged one — a silent ~1e-3 discrepancy on the entry point
  documented for differentiable pipelines. The eager path now performs
  the same contact-split average (they agree to 2e-16).
  `supersample_factor` still returns the raw supersampled grid, which is
  its documented behaviour.
- Contact-split quadrature weights are renormalised. The clamped
  sub-interval widths summed to the exposure only to ~1e-14, so
  out-of-transit flux came back as 1 + 2e-14 rather than 1 to round-off,
  breaking the exact-unity contract kept everywhere else. The weights are
  a partition of unity by construction, so this removes round-off and
  nothing else: now 1 ulp, convergence figures unchanged.

## [0.4.0] — 2026-09-27

Closes the two remaining optional items from the original M6 list
(arbitrary order, exposure-time integration) and gives the cross-code
benchmark an eccentric half.

### Added
- **Arbitrary-order polynomial limb darkening**, `limb_dark="polynomial"`
  with `u = [u_1 ... u_N]` for any N (`metalplanet/poly.py`). ALFM19's
  three-term M_n recursion seeded by closed forms for M_0..M_3, then
  s_n = -(2 r² M_n - n/(n+2)[(1-r²-z²) M_n + sqarea M_{n-2}]). Validated
  against a 40-digit mpmath direct integration sharing no code with it:
  1e-15 at N = 8, 2e-13 at N = 16, 3e-9 even at N = 30 (the upward
  recursion loses ~1 digit per 2 orders; Limbdark.jl's downward
  series-seeded variant is documented but unnecessary at any order a
  physical law uses). Non-polynomial laws remain unsupported — they are
  outside the ALFM19 formulation, and the error now says so.
- `greens_affine()` writes the u → g map as the affine map it is, so the
  coefficients stay **traced**: they can change between calls on a built
  model (the batman workflow) and the limb darkening is differentiable.
- **Contact-split exposure integration**, `integration="contact"`
  (`metalplanet/exposure.py`). The light curve's derivative jumps at each
  contact, so uniform supersampling converges only as O(1/N). Splitting
  the exposure window at the contacts and applying a fixed-order
  Gauss-Legendre rule to each smooth piece converges geometrically:
  25 evaluations per exposure reach 8.9e-8 where supersampling needs
  N ~ 19,500 — ~780x fewer model evaluations
  (`benchmarks/bench_exposure.py`). Fixed order rather than Limbdark's
  adaptive Simpson, because adaptive depth is data-dependent control
  flow and would break batching and `mx.compile`; the five-interval split
  is branchless, so grazing and full transits share one code path and the
  node positions stay differentiable.
- **Eccentric cross-code benchmarks**: `RESULTS.md` now compares all five
  codes at e = 0.3 on precision and orbit-inclusive speed, each solving
  Kepler's equation itself. MetalPlanet fp64 is the most accurate
  (4.4e-16, tied with exoplanet-core) and its fp32 GPU path the fastest
  past N ~ 1e5 (15.3 ms at 10^7 points vs 207 ms for the next code).

## [0.3.0] — 2026-09-27

The eccentric release: eccentric orbits now run on the fused Metal
kernel with analytic gradients, and have their own anvil sampling
target. Plan, gates and review log: `docs/v3eccentrickernel_plan.md`.

### Added
- **Transit-anchored eccentric orbit** (`metalplanet/anchored.py`).
  Solves for δ = E − E₀ about inferior conjunction rather than for E
  about periastron. Algebraically identical to the direct formulation
  (verified to 2.5e-15 in flux over an (e, w) grid including
  near-apastron and near-periastron transits at e = 0.999), but it never
  forms ω as an intermediate — only e·cosω and e·sinω, taken directly
  from the (√e cosω, √e sinω) sampling pair. Since ∂ω/∂k = −h/e
  diverges, the direct form computes a finite gradient as the difference
  of two O(a/e) terms: measured fp32 gradient error 5e-2 at e = 1e-5 and
  1e+1 at e = 1e-8, versus ~1e-7 *flat* for the anchored form
  (`benchmarks/v3_kh_grad_conditioning.py`).
- **v3 eccentric Metal kernel** (`make_ecc_core_metal`): anchored solve,
  Cartesian tail and ALFM19 photometry in one register-resident pass,
  plus its analytic VJP (14 per-point partials). At 1024 × 65,536:
  forward 29.0 ms, value+grad 76.0 ms — 1.41x / 1.38x the circular
  kernel, and 20x / 902x the same model as a compiled MLX graph.
  Parity 3.7e-7 vs the graph over e ∈ [0, 0.999].
- **In-kernel gradient reduction** for the circular v2 VJP
  (`make_model_core_metal(reduce="simd"|"grid")`, simd default).
  `metal::simd_sum` reduces over the *active* lanes, so early-returned
  lanes drop out by themselves — no predication, threadgroup memory,
  barrier or grid padding. Backward 35.8 → 30.8 ms (1.16x), peak memory
  4.56 → 2.74 GB, transients 1.88 GB → 58.7 MB. `"grid"` is retained as
  the parity oracle.
- **Eccentric anvil target**: `make_ecc_transit_flux` / `make_ecc_target`
  (10 parameters), `ecc_constraint_penalty` and `PenalizedLogLike` for
  the two joint constraints a box of ParamSpecs cannot express
  (periastron clearance, |cos i| ≤ 1) — without them an unphysical
  geometry returns a finite log-likelihood and the chain silently
  samples an improper posterior.
- `examples/chees_ecc.py`: eccentric ChEES-HMC injection-recovery at a
  deliberately low truth (e = 0.02), where the parameterization matters.
  1024 chains, 600 warmup + 200 samples: **zero divergences**, truth
  recovered within 1σ on all ten parameters. The 10-parameter posterior
  mixes far more slowly than the circular one (R-hat 3.5 vs 1.5 for the
  circular control at matched 256 chains × 400+400) — the well-known
  (a, b, e, ω) transit-duration degeneracy, not a model defect.
  `benchmarks/v3_ecc_sampling_geometry.py` reproduces the diagnosis,
  including the A/B that eliminates the constraint barrier as the cause
  of the divergences seen at shorter warmups.
- The batman-style frontend routes its fp32 GPU *eccentric* primary
  transits through the v3 kernel: 2.1x at N = 1e7, 1.5x at 1e6, 1.2x at
  1e5 on `light_curve()`. The period is carried by the kernel's traced
  `p_off` input (with `period_ref = 0`) so batman-style parameter
  updates never see a baked-in constant.
- Benchmarks: `bench_ecc_kernel.py`, `v3_reduction_spike.py`,
  `v3_kh_grad_conditioning.py`, `v3_ecc_graph_baseline.py`; the VJP
  profiler now A/Bs both reduction strategies.

### Changed
- Markley starter cbrt: `metal::precise::powr` (~220x a multiply, ~20%
  of orbit time) replaced by the inverse-cbrt bit trick with the
  principled (4/3)·0x3f800000 seed and three division-free Newton steps;
  1.7e-6 max relative error in-kernel against a ~1e-4 requirement.
- The batman-style frontend uses the anchored formulation for eccentric
  orbits, so φ is measured straight from t0 and the
  mean-anomaly-at-transit offset no longer appears in the compiled graph.
- Kernel sources share one `_PHOT_PARTIALS` fragment and expand through
  `_subst`, which asserts no marker survives — an unexpanded marker is a
  Metal compile failure, which aborts the process rather than raising.

### Fixed
- **NaN gradients at δ ≈ π** (ordinary apastron-side geometry):
  `_one_minus_cos` floored its denominator instead of setting it to 1 on
  the inactive branch, so the VJP hit 0 × inf for every parameter routed
  through the solve.
- **NaN gradients at exactly e = 0** — an interior point of the sampling
  disc — from `sqrt(max(e, 0))`, whose derivative is infinite there, and
  from `arctan2(0, 0)`, whose gradient is undefined.
- `test_loglike_at_truth_is_sane` now accounts for upstream anvil's
  recentring policy (the unnormalized logL is `value + log_offset_const`).

## [0.2.0] — 2026-09-06

### Added
- **Fused Metal kernels** (`metalplanet/metal.py`): the fp32 GPU path
  runs the whole model in registers. v1: photometric core
  (z, r, u1, u2) → flux + analytic-VJP kernel; v2: model-level kernel
  consuming the anvil (v, x) contract with the orbit folded in.
  Measured (M2 Max, 1024×65,536): forward 17.5 ms, value+gradient
  52.5 ms (519× reverse-mode autodiff); ~55,000 light curves/s in
  population batches. `core="metal"` is the default for the anvil
  target and the fp32 frontend, with silent graph fallback for
  fp64/CPU.
- **Markley Kepler solver** (`kepler`, `kepler_E_sincos`):
  non-iterative starter + one fifth-order refinement, a single sincos
  per point, implicit-function-theorem custom VJPs at both the true-
  anomaly and eccentric-anomaly level.
- Benchmark suite (`benchmarks/`): precision vs a 30-digit mpmath
  oracle, subprocess-isolated speed sweeps vs batman / PyTransit /
  exoplanet-core / jaxoplanet, batch scaling, and
  `verify_doc_claims.py` making every documented number reproducible.
- Docs: `docs/sampler-integration.md` (the batching rule, sampler
  recipes, verified against a spy test), `docs/eccentric-kernel-notes.md`
  (verified backward-pass formulas and kernel design findings).
- Engine-side (anvil repo): `run(progress=N)` flushed progress ticks
  with rate/acceptance/divergences/ETA.

### Changed
- **Eccentric separation is now Cartesian-from-E** — the true anomaly
  is never computed. Exactly equivalent to the previous conic tail
  (proven symbolically and to 4.5e-25 at 40 digits) and strictly
  better conditioned: removes a ~1e-6 fp32 flux-error floor near every
  transit and a ≥1e-3-flux failure mode at near-apastron transits of
  high-e orbits; fp32-trustworthy at the 1e-6 flux level to e ≤ 0.999.
- batman-style frontend evaluates through `mx.compile`d graphs with
  parameters as traced scalars (no retrace on parameter updates).

### Fixed
- Three latent dtype bugs where MLX ops on pure-Python scalar operands
  minted float32 constants inside fp64 paths (`beta`, `fac`, `ome2`).
- Circular orbit no longer fabricates a mirror transit at the far
  conjunction (front-side masking).
- 15 code-review findings in the sampler guide, including an incorrect
  emcee batching claim (half-ensembles), a priors-free example that
  sampled an improper posterior, and unreproducible measured numbers.

## [0.1.0] — 2026-09-05

Initial release: ALFM19 quadratic limb-darkened transits in MLX.
Batched fixed-iteration Bulirsch `cel`; deviation-form solution vector
with masked regimes and Taylor stability switches; Green's-basis flux
assembly; analytic custom VJP; Kipping (2013) (q1,q2) limb darkening;
epoch-centered circular orbit; batman-parity frontend
(TransitParams/TransitModel incl. eccentric orbits, supersampling,
secondary eclipses); anvil/applemcmc integration target; accurate fp64
sincos workaround for MLX's fp32-accurate fp64 trig; validation against
batman, a Limbdark.jl port, and direct mpmath integration.
