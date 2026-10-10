# Plan: occultors larger than the star (Rp/R* > 1)

**Status: implemented (2026-10-10), unreleased.** Target release: v0.11.0.
See "What changed from the plan" at the end.

## Why

White-dwarf hosts put the companion's radius well above the star's:
WD 1856+534 b has Rp/R* = 7.28 (a/R* = 336, b = 7.79, a grazing transit
of about 8 minutes). Eclipsing WD + brown-dwarf and WD + M-dwarf systems
reach total occultation (b < r - 1), where the white dwarf vanishes
entirely for part of the event.

MetalPlanet never claimed r > 1. `solution.py` states "Assumes 0 < r < 1
(no total occultation branch)", but nothing rejects a larger r.

## Baseline (measured 2026-10-10)

The reference is an mpmath integral of the occulted intensity over
circles of radius x about the star's centre: the arc inside the occultor,
or the full circle when x + z <= r. That is the repo's `test_poly._oracle`
generalised to any radial law. Scratchpad probes: `rbig_probe.py`,
`rbig_e2e.py` and `rbig_more.py` (session d78f4d29).

**Already correct: the fp64 graph, every law.** That covers `flux_dev`,
`flux_dev_poly`, `flux_dev_hybrid`, `shape_cols`, `ld_basis_analytic`,
and `TransitModel`'s fp64 paths during transit, including total
occultation. It gets there by accident: the kite floor and the cel
modulus clip happen to produce the full-disk values. Gradients match
finite differences, except exactly at z == r, which is MLX's known tie
artefact, as for r < 1.

| r | quadratic | 4th-order polynomial | hybrid5 |
|---|---|---|---|
| 7.28 | 6e-15 | 7e-14 | 1e-14 |
| 20 | 8e-14 | 2e-11 | 1e-13 |
| 50 | 3e-12 | 1e-9 | 4e-12 |

**Broken:**

1. **The far-side push.** Points with the planet behind the star are
   moved to separation `2 + z`, which clears the star only if 1 + r <= 2.
   With b < r - 1, the fp64 model shows a full eclipse at secondary
   conjunction. The Metal kernels test `vv <= 0` explicitly and are fine.
   Sites: `orbit.separation_circular`, which `anvil.py`, `metal.py`'s tau
   graph and `metal_hybrid.py`'s tau graph call, and `api._photom` for
   both the primary and secondary branches.
2. **No total-occultation branch in the fp32 quadratic kernel core.**
   `metal._CORE` returns NaN at z = 0, errors up to 17 for 0 < z < r - 1,
   and inf at r = 1.001. It reaches every quadratic kernel: the z kernel,
   the tau kernels (through `_PHOT_FN` and `_PHOT_B_FN`), the model
   kernel (through `_PHOT`) and the `ld_basis` kernels.
3. **The contact rule never places the inner contacts.** They sit at
   z = |1 - r|, but `exposure.contact_offsets` and
   `contact_offsets_anchored` solve for 1 - r. For r > 1 that is negative,
   so the inner pair collapses as if the transit were grazing. Error is
   1.1-1.9e-5 for total geometries, against 2.4e-6 for an r = 0.1 transit
   on the same exposure; grazing r > 1 is unaffected. Measured against an
   exact reference that places 64-point Gauss-Legendre rules between the
   true contacts.
4. **fp32 precision falls roughly as r squared in partial overlap.** Terms
   of size ~r cancel to order 1:

   | r | quadratic kernel | hybrid5 kernel |
   |---|---|---|
   | 1.5 | 7e-8 | 8e-8 |
   | 7.28 | 3-9e-6 | 6e-6 |
   | 20 | 7e-5 | 1e-4 |
   | 50 | 1e-3 | 2e-3 |

5. **r = 1 exactly is degenerate.** Contacts meet at z = 0. fp64 is off by
   5e-9 and fp32 by 1e-4. The contact Taylor switch (|z + r - 1| <
   sqrt(eps)) evaluates `sqrt(r (1 - r))` and is reachable for r just
   above 1.
6. **The anvil sampler bounds cap the geometry.** `DEFAULT_BOUNDS` has
   r <= 0.5 and b <= 1.2. Callers can override them with `bounds=`.

## Invariants

- **r < 1 is untouched, bitwise.** Every new branch is unreachable for
  r < 1: z <= r - 1 < 0 never happens, and |1 - r| == 1 - r exactly. The
  far-side push only moves points that already have zero flux.
  - Gate 1: the 296-array release corpus (`lc_snap.py` vs
    `lc_before.npz`) must stay identical.
  - Gate 2: `tests/test_entry_contract.py` must pass.
- **Kernel sources change on purpose in Stage 2.** The 18-source
  byte-identity gate is retired for that stage. In its place, kernel
  outputs and VJPs for r < 1 must be bitwise identical to v0.10.7 over
  the corpus's kernel arrays and a fixed (z, r < 1) sweep, recorded
  before the edit.
- **No API change.** `rp` already passes through every entry point.

## Stage 1: graph paths (fp64 and fp32 graph)

1. **Far-side push.**
   - `separation_circular` returns `z + 2a` on the far side instead of
     `2 + z`. a > (1 + r)/2 holds for any non-contact orbit, so the point
     clears the star. The docstring drops the "r < 1" premise.
   - `api._photom` pushes to `z + 2 (1 + rp)` in both branches.
   - Check that `anvil.py:519` (`front & (z < 1 + r)`) is already right.
2. **Inner contacts at |1 - r|.**
   - In `exposure.contact_offsets`, the inner Z becomes `mx.abs(1.0 - r)`;
     in `contact_offsets_anchored`, the Z tuple does the same.
   - The "grazing" docstring condition becomes b > |1 - r|.
   - Every route gets the contacts from these two functions; the kernels
     receive them as inputs (`_contact_taus`, `_TAU_EDGES`).
3. **Explicit total-occultation branch.** Add `m_tot = z <= r - 1` in
   `solution.sn_dev_with_aux`. On those lanes set (s0d, s1d, s2d) =
   (-pi, -2 pi/3, 0), mask its partials to zero in `vjp.sn_partials`, and
   sanitise the partial-branch inputs there (the `mx.where(mask, x, 1)`
   discipline). Do the same in `hybrid.lens_geometry` and its column
   functions: E0 = -pi and T_j = -N_j, zero partials. `poly.sn_dev_poly`'s
   higher terms take their full-disk values. The current values on those
   lanes are right by accident; this makes them right by construction,
   gradients included.
4. **r near 1.** Gate the contact Taylor switch to r < 1, so
   `sqrt(r (1 - r))` is never read for r >= 1. Measure r in {1 - 1e-7, 1,
   1 + 1e-7, 1 + 1e-3} after the change. If r == 1 stays at 5e-9 in fp64,
   document it as a degenerate point rather than chase it.
5. Docstrings in `solution.py`, `hybrid.py` and `orbit.py` describe the
   three regimes: none, partial, and complete or total.

Exit: fp64 oracle agreement at the levels in the baseline table across
the Stage 4 grid. No far-side dip. Contact-rule error for total geometries
at or below the r < 1 level against the exact piecewise reference.

## Stage 2: fp32 kernels

1. **`metal._CORE`.** After `float r2 = r * r;`, so that `_PHOT` and
   every derived kernel inherit it, add `bool m_tot = z <= r - 1.0f;`. On
   those lanes, s0d, s1d, s2d take (-pi, -2 pi/3, 0) and the z/r partials
   take 0, selected before the cel3 call so it never sees a total lane's
   arguments. The scalar, VJP and basis tails then give F - 1 = -1 with
   zero gradients, and B = -(pi, 2 pi/3, pi/2).
2. **`metal_hybrid._HYB_FN`.** An explicit total branch in
   `mp_hyb_cols_d`: E0 = -pi, T_j = -HYB_NORM[j], zero partials. Today it
   is 1e-6 by accident.
3. **Regenerate and check the derived sources.** `_swap` asserts its
   anchors, so a template change that breaks a substitution fails loudly.
   Re-run the leftover-marker asserts.
4. **r < 1 bitwise gate,** as set out under Invariants.

Exit: fp32 kernels agree with fp64 to the baseline partial-region levels.
Exact -1 with zero gradients under total occultation, for every kernel
family. No NaN or inf anywhere on the Stage 4 grid.

## Stage 3: precision policy, sampler bounds, docs

1. **Precision policy.** The README gets a "large occultors" section with
   the r-squared table: use fp64 above r ~ 10, or accept the table's
   error. Construction-time warnings are out of scope unless asked for;
   they would add a host read on the hot path.
2. **anvil.** Document overriding `bounds=` for r and b, with a WD
   1856+534 b-like `make_transit_target` example. `DEFAULT_BOUNDS` stays
   unchanged, so existing targets don't move.
3. **README.** A white-dwarf section covering the grazing and total
   regimes and an fp64/fp32 example.

## Stage 4: tests (`tests/test_large_occultor.py`)

- **Oracle grid.**
  - r in {1.2, 3, 7.28, 20, 50}.
  - z spans total (0, mid, r - 1 - 1e-4), inner contact, r - 1 + 1e-4,
    z ~ r, partial, outer contact and none.
  - Every law: quadratic, polynomial (order 4), hybrid2/4/5, and both
    `ld_basis` forms.
  - fp64 graph gates from the baseline table. fp32 kernel gates per r,
    at 2x the baseline table's measured error (e.g. 2e-5 at r = 7.28,
    2.5e-4 at 20, 4e-3 at 50): a uniform 2e-7 r^2 would already fail at
    r = 20.
- **Total occultation.** F - 1 == -1 exactly; gradients in z, r and every
  limb-darkening coefficient exactly 0; fp64 and fp32, graph and kernel.
- **Far side.** Full-orbit light curves for b < r - 1 and b > r - 1,
  circular and eccentric, fp64 and fp32: flux is 1 everywhere away from
  primary conjunction. The secondary-eclipse variant should show no dip
  at primary conjunction.
- **Contact rule.** Against the exact piecewise reference for (r, b) in
  {(7.28, 7.79), (7.28, 3), (7.28, 0.3), (1.5, 0.2)}: error at or below
  the r = 0.1 baseline (3e-6 at a 2-minute exposure), on `TransitModel`
  and `flux_dev_from_tau`, circular and eccentric.
- **Gradients.** FD agreement off the z == r tie at r in {1.5, 7.28, 20},
  fp64; kernel VJP against fp64 autodiff within 2e-3 relative at r =
  7.28.
- **r near 1.** {1 - 1e-7, 1, 1 + 1e-7, 1 + 1e-3}: finite everywhere,
  gates per Stage 1 step 4.
- **WD 1856+534 b-like end to end.** `TransitModel` and
  `flux_dev_from_tau`, fp64 and fp32, contact rule.
- **Entry contract.** Add an r = 7.28 total-occultation row to
  `test_entry_contract.py`'s kernel entries, so every calling form is
  exercised at r > 1.

## Risks

- **Stage 2 edits shared kernel text.** The model kernel inlines `_PHOT`
  verbatim, so one edit reaches the z, tau, model and basis kernels. The
  r < 1 bitwise output gate is the guard. It must cover VJP outputs, not
  just forward.
- **An explicit total branch could change values the accidental path got
  right,** for example s2d's ~1e-14 residue becoming an exact 0. That is
  only for r > 1, where nothing was promised.
- **The far-side push in `separation_circular` changes a returned
  value.** Flux there is identically 0 for every r < 1 (m_none), and
  gradients are masked. The corpus confirms.
- **fp32 at large r** stays a documented limitation. A reformulation that
  removes the r-squared cancellation is a research item, not part of this
  plan.

## Out of scope

- An fp32 reformulation for r >> 10.
- A non-spherical white dwarf.
- Gravitational self-lensing, which can matter in some white-dwarf
  binaries.
- Changing `DEFAULT_BOUNDS`.

## Effort

| Stage | Size | Notes |
|---|---|---|
| 1 | small | about 10 edits in 5 files |
| 2 | medium | 2 kernel templates plus regeneration and gates |
| 3 | small | docs and an example |
| 4 | medium | the oracle grid is the bulk |

Roughly one to two working sessions, released as v0.11.0. That is a minor
bump because it adds a supported regime, with no API change.

## What changed from the plan (2026-10-10)

- **The contact-rule diagnosis was wrong.** The solver uses Z^2, and
  (1 - r)^2 = (r - 1)^2, so the inner contacts were always placed
  correctly. The 1e-5 deep-eclipse error is quadrature resolution: it
  falls with n_gl like an r < 1 transit's (n_gl = 9: < 1e-6) and is
  *smaller* relative to the depth. The |1 - r| edit stays as
  documentation; the README tells deep-eclipse users to raise n_gl.
- **Output-level selects broke the r < 1 bitwise gate.** Selecting
  full-disk constants at sn_dev_with_aux's outputs moved compiled
  polynomial-frontend results by 1-2 ulp (MLX fused the graph
  differently), though no r < 1 value was selected. The total branch is
  implemented upstream instead: kite = 0 on total lanes makes the
  existing formulas exact. Same on the hybrid graph.
- **A pre-existing fp32 NaN** at the internal contact of every hybrid law
  (z + r rounding to exactly 1) surfaced in the r < 1 kernel gate and is
  fixed in this release.
- **Gradient NaNs at z = 0** on total lanes (masked divisions whose
  floors square to 0 in the VJP) were fixed in the solver, the polynomial
  recursion and the hybrid poles.
- **The contact Taylor switch** is guarded to r <= 1, not r < 1: at
  r == 1, z = 0 is both the contact and the onset of total occultation,
  and the generic forms divide by zero there.
- **An early return in the hybrid kernel** cost its forward pass
  1.35-1.65x; the kernel zeroes the lens instead, branch-free. (GPU
  timings late in the session were taken on a loaded machine, so only
  the early return's cost was established firmly.)
- **The anvil target** needs `a` widened too for WD 1856+534 b
  (a/R* = 336 > 200); documented and tested.

