"""Is ChEES's max_leapfrog cap of 24 still right?

The cap was set when an unbounded adaptation drove trajectories to ~128
steps at ~1.2 s per gradient. Gradients now cost 55-76 ms, so the cap is
re-measured here on ESS/s (and R-hat, and divergences) rather than
assumed. The answer is target-dependent -- see docs/sampler-integration.md.

Note on interpretation: ESS cannot exceed n_chains x n_samples, so at
high caps the circular run approaches that ceiling and ESS/s must
eventually fall no matter how well the trajectories decorrelate. The
robust conclusion is the direction and size of the effect, not a single
optimal integer.
"""
import json
import os
import time

import numpy as np
import mlx.core as mx

from metalplanet.anvil import import_engine, make_ecc_target, make_target

engine, _ = import_engine()
HERE = os.path.dirname(os.path.abspath(__file__))
NC, NW, NS = 256, 300, 200


def run(tt, ndim, mlf):
    u_t = tt.transform.from_model_np(tt.truth_model)
    rng = np.random.default_rng(0)
    u0 = mx.array((u_t + 1e-3 * rng.standard_normal((NC, ndim))
                   ).astype(np.float32))
    k = engine.ChEESHMC(tt.target, max_leapfrog=mlf)
    t0 = time.perf_counter()
    r = engine.run(k, tt.target, u0, n_warmup=NW, n_samples=NS, seed=1,
                   reanchor_every=100, progress=False)
    wall = time.perf_counter() - t0
    ch = r.get_chain()
    ess = float(engine.diagnostics.ess_bulk(ch).min())
    rec = {"max_leapfrog": mlf, "wall_s": wall, "min_ess": ess,
           "ess_per_s": ess / wall,
           "max_rhat": float(engine.diagnostics.split_rhat(ch).max()),
           "divergences": int(r.extras.get("n_divergent", 0))}
    print(f"  max_leapfrog={mlf:4d}: {wall:6.1f}s  minESS {ess:8.0f}  "
          f"{rec['ess_per_s']:6.2f} ESS/s  Rhat {rec['max_rhat']:5.2f}  "
          f"div {rec['divergences']}", flush=True)
    return rec


def main():
    out = {"n_chains": NC, "n_warmup": NW, "n_samples": NS,
           "ess_ceiling": NC * NS}
    print(f"=== circular 8-parameter target (ESS ceiling "
          f"{NC * NS:,}) ===", flush=True)
    tt = make_target(n_data=20_000, seed=42)
    out["circular"] = [run(tt, 8, m) for m in (16, 24, 48, 96, 192, 384)]
    print("=== eccentric 10-parameter target ===", flush=True)
    tte = make_ecc_target(n_data=20_000, seed=42, ecc=0.30, omega_deg=63.0)
    out["eccentric"] = [run(tte, 10, m) for m in (16, 24, 48, 96)]
    with open(os.path.join(HERE, "leapfrog_cap.json"), "w") as f:
        json.dump(out, f, indent=1)
    for name in ("circular", "eccentric"):
        rows = out[name]
        best = max(rows, key=lambda r: r["ess_per_s"])
        at24 = next(r for r in rows if r["max_leapfrog"] == 24)
        print(f"{name}: best {best['ess_per_s']:.2f} ESS/s at "
              f"max_leapfrog={best['max_leapfrog']} vs "
              f"{at24['ess_per_s']:.2f} at the old cap of 24 "
              f"({best['ess_per_s'] / at24['ess_per_s']:.2f}x)")


if __name__ == "__main__":
    main()
