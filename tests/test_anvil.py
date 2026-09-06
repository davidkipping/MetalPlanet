"""M4: applemcmc integration — precision harness and injection-recovery.

The fast tests here gate the milestone: the fp32 GPU likelihood must sit
well under the ~1-unit Metropolis decision scale against the fp64 CPU
path (the trapezoid target achieves ~0.2; this model is smoother and
must beat it). The full HMC/ensemble injection-recovery lives in
run_injection_recovery.py (GPU-heavy, run separately) with a smaller
smoke version marked slow here.
"""

import mlx.core as mx
import numpy as np
import pytest

from metalplanet.anvil import import_engine
from metalplanet.anvil import make_quad_transit_flux, make_target
from metalplanet.orbit import epoch_center_times

applemcmc, _ = import_engine()

RNG = np.random.default_rng(3)


@pytest.fixture(scope="module")
def tt():
    return make_target(n_data=100_000, seed=42)


class TestModelContract:
    def test_shapes_and_dtype_polymorphism(self):
        model = make_quad_transit_flux(3.456)
        v32 = mx.array(RNG.uniform(0.1, 0.3, (7, 8)).astype(np.float32))
        x32 = mx.array(RNG.uniform(-1, 1, (2, 100)).astype(np.float32))
        out32 = model(v32, x32)
        assert out32.shape == (7, 100) and out32.dtype == mx.float32
        with mx.stream(mx.cpu):
            out64 = model(v32.astype(mx.float64), x32.astype(mx.float64))
            assert out64.dtype == mx.float64

    def test_compile_safe(self):
        model = make_quad_transit_flux(3.456)
        compiled = mx.compile(model)
        v = mx.array(RNG.uniform(0.1, 0.3, (4, 8)).astype(np.float32))
        x = mx.array(RNG.uniform(-1, 1, (2, 64)).astype(np.float32))
        np.testing.assert_allclose(
            np.array(compiled(v, x)), np.array(model(v, x)),
            rtol=2e-6, atol=2e-7)

    def test_grad_finite_through_full_model(self):
        model = make_quad_transit_flux(3.456)
        x = mx.array(np.stack([
            RNG.uniform(-1.7, 1.7, 4096), RNG.integers(0, 26, 4096)
        ]).astype(np.float32))
        v = mx.array(np.tile(
            [0.01, 1e-4, 0.1, 0.3, 8.8, 0.42, 0.31, 1e-4], (16, 1)
        ).astype(np.float32))
        v = v + mx.array(1e-3 * RNG.standard_normal(v.shape).astype(np.float32))

        def loss(v_):
            return mx.sum(model(v_, x) ** 2)

        g = mx.grad(loss)(v)
        assert bool(mx.all(mx.isfinite(g)))


class TestPrecisionHarness:
    """fp32 logL error scales with the |logL| magnitude of the test
    states (per-term fp32 representation of chi^2 terms dominates far
    from truth; decomposition: per-point model error is at the fp32
    floor, rms ~1.4e-8 flux with ~1e-9 bias, and neither chunk size nor
    fp64_anchor moves the total). So the comparison against the
    trapezoid's ~0.2 is made at *matched* |logL| magnitude ~5e4 — where
    accept/reject actually operates — and in relative terms at the
    engine-convention displaced states.
    """

    def test_relative_precision_beats_trapezoid_wide_states(self, tt):
        """Engine-recipe states (u +- 0.05): relative error must beat the
        trapezoid's measured 0.18/5.2e4 = 3.5e-6."""
        u_truth = tt.transform.from_model_np(tt.truth_model)
        u = mx.array(
            (u_truth + 0.05 * RNG.standard_normal((32, 8))).astype(np.float32))
        report = applemcmc.validate_precision(tt.target, u)
        print("\n", report, sep="")
        rel = report.median_abs_err / report.median_logl_magnitude
        assert rel < 2.5e-6, f"relative fp32 error {rel:.2e}"

    def test_absolute_precision_at_matched_magnitude(self, tt):
        """States with |logL| ~ 5e4 (the trapezoid test's magnitude):
        median well under its 0.18, max well under 1."""
        u_truth = tt.transform.from_model_np(tt.truth_model)
        u = mx.array(
            (u_truth + 0.005 * RNG.standard_normal((32, 8))).astype(np.float32))
        report = applemcmc.validate_precision(tt.target, u)
        print("\n", report, sep="")
        assert report.median_logl_magnitude < 1.2e5  # sanity: comparable
        assert report.median_abs_err < 0.15, "did not beat trapezoid's 0.18"
        assert report.max_abs_err < 1.0, "fp32 error at Metropolis scale"

    def test_near_posterior_states(self, tt):
        """Warmed-up-sampler displacements: tight absolute bounds."""
        u_truth = tt.transform.from_model_np(tt.truth_model)
        u = mx.array(
            (u_truth + 0.002 * RNG.standard_normal((32, 8))).astype(np.float32))
        report = applemcmc.validate_precision(tt.target, u)
        assert report.median_abs_err < 0.1
        assert report.max_abs_err < 0.6

    def test_loglike_at_truth_is_sane(self, tt):
        """At truth, chi^2/n ~ 1."""
        with mx.stream(mx.cpu):
            lp = np.array(tt.loglike.hi(
                mx.array(tt.truth_model[None, :], dtype=mx.float64)),
                dtype=np.float64)[0]
        n = tt.y.size
        assert abs(-2.0 * lp / n - 1.0) < 0.02


@pytest.mark.slow
class TestInjectionRecoverySmoke:
    def test_hmc_small(self, tt):
        """Short ChEES-HMC run: zero divergences and truth recovery to
        within 5 sigma on every parameter (full run in the script)."""
        n_chains = 256
        u_truth = tt.transform.from_model_np(tt.truth_model)
        u0 = mx.array(
            (u_truth + 0.01 * RNG.standard_normal((n_chains, 8))
             ).astype(np.float32))
        kernel = applemcmc.ChEESHMC(tt.target, max_leapfrog=64)
        res = applemcmc.run(kernel, tt.target, u0, n_warmup=150,
                            n_samples=100, seed=1, reanchor_every=100)
        assert res.extras.get("n_divergent", 0) == 0
        flat = res.get_chain(flat=True).astype(np.float64)
        model_draws = tt.transform.model_np(flat)
        mean = model_draws.mean(axis=0)
        sd = model_draws.std(axis=0) + 1e-12
        pull = np.abs(mean - tt.truth_model) / sd
        assert np.all(pull < 5.0), f"pulls {pull}"
