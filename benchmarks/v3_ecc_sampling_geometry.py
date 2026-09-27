"""Where do the eccentric target's ChEES divergences come from?

Reproduces the E5 diagnostic. Two candidate explanations were tested and
one was eliminated outright:

  * the joint-constraint barrier — ELIMINATED. Divergence counts are
    bit-identical with and without it, and it is active in 0.000% of the
    samples drawn (the chains never approach e_max(a, r)).
  * warmup length — divergences fall to ZERO with a long enough warmup,
    so they are a step-size-adaptation artefact, not a broken gradient.

What does NOT go away is the between-chain spread (R-hat), which is a
property of the 10-parameter posterior: transit photometry constrains a
combination of (a, b, e, w) through the duration, leaving a curved,
strongly correlated ridge that a diagonal mass matrix mixes slowly. The
circular 8-parameter control run through the same machinery is included
so the comparison is like-for-like.
"""
import numpy as np
import mlx.core as mx

from metalplanet.anvil import (PARAM_NAMES_ECC, ecc_constraint_penalty,
                               import_engine, make_ecc_target, make_target)

engine, _ = import_engine()
NC = 256


def run(tt, ndim, label, nw, ns, target=None):
    u_t = tt.transform.from_model_np(tt.truth_model)
    rng = np.random.default_rng(0)
    u0 = mx.array((u_t + 1e-3 * rng.standard_normal((NC, ndim))
                   ).astype(np.float32))
    k = engine.ChEESHMC(target or tt.target, max_leapfrog=24)
    r = engine.run(k, target or tt.target, u0, n_warmup=nw, n_samples=ns,
                   seed=1, reanchor_every=100, progress=False)
    ch = r.get_chain()
    print(f"{label:48s} div={r.extras.get('n_divergent', 0):6d} "
          f"Rhat={engine.diagnostics.split_rhat(ch).max():6.2f} "
          f"ESS={engine.diagnostics.ess_bulk(ch).min():6.0f}")
    return r


class _Unpenalized:
    """The same likelihood with the barrier stripped off."""

    def __init__(self, pen_loglike):
        self.base = pen_loglike.base

    def __call__(self, v):
        return self.base(v)

    def hi(self, v):
        return self.base.hi(v)

    def __getattr__(self, n):
        return getattr(self.base, n)


if __name__ == "__main__":
    applemcmc, _ = import_engine()
    tt = make_ecc_target(n_data=20_000, seed=42, ecc=0.30, omega_deg=63.0)

    print("--- is it the barrier? ---")
    r = run(tt, 10, "eccentric, with barrier", 150, 50)
    bare = _Unpenalized(tt.loglike)
    run(tt, 10, "eccentric, barrier removed", 150, 50,
        target=applemcmc.TransformedLogDensity(
            bare, tt.transform, model_log_prob_hi=bare.hi))
    v = tt.transform.model_np(r.get_chain(flat=True).astype(np.float64))
    pen = np.array(ecc_constraint_penalty(mx.array(v.astype(np.float32))))
    e_s = v[:, 7] ** 2 + v[:, 8] ** 2
    print(f"    barrier active in {100 * np.mean(pen < 0):.3f}% of samples; "
          f"e sampled {e_s.min():.3f}-{e_s.max():.3f} vs "
          f"e_max(a, r) ~ {1 - 1.1 / 8.8:.3f}")

    print("\n--- is it warmup length? ---")
    run(tt, 10, "eccentric, warmup 150", 150, 50)
    run(tt, 10, "eccentric, warmup 600", 600, 50)

    print("\n--- is the geometry harder than the circular problem? ---")
    run(make_target(n_data=20_000, seed=42), 8,
        "circular 8-param control, 400 + 400", 400, 400)
    run(tt, 10, "eccentric 10-param, 400 + 400", 400, 400)
