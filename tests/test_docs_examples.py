"""The sampler-integration guide's emcee recipe must actually work:
half-ensemble batching, priors rejecting unphysical proposals, compiled
chunked likelihood, finite chains. Small-scale smoke of the exact
pattern documented in docs/sampler-integration.md."""

import numpy as np
import mlx.core as mx
import pytest

emcee = pytest.importorskip("emcee")

from metalplanet.anvil import import_engine, make_quad_transit_flux
from metalplanet.orbit import epoch_center_times

_, ChunkedGaussianLogLike = import_engine()

LO = np.array([-0.5, -0.05, 0.01, 0.0, 2.0, 0.0, 0.0, -0.01])
HI = np.array([0.5, 0.05, 0.50, 0.9, 50., 1.0, 1.0, 0.01])
TRUTH = np.array([0.0, 0.0, 0.1, 0.3, 8.8, 0.4225, 0.3077, 0.0])


@pytest.fixture(scope="module")
def recipe():
    rng = np.random.default_rng(5)
    period_ref = 3.456
    t = np.sort(rng.uniform(0, 10.0, 4000))
    model = make_quad_transit_flux(period_ref)
    x64 = epoch_center_times(t, t0_ref=0.0, period_ref=period_ref)
    with mx.stream(mx.cpu):
        y_dev = np.array(
            model(mx.array(TRUTH[None, :], dtype=mx.float64),
                  mx.array(x64, dtype=mx.float64))[0], dtype=np.float64)
    yerr = 5e-4
    y_dev = y_dev + yerr * rng.standard_normal(t.size)
    loglike = ChunkedGaussianLogLike(model, x64, y_dev,
                                     np.full(t.size, yerr))
    loglike_c = mx.compile(lambda v: loglike(v))

    def log_prob_batch(theta):
        lp = np.array(loglike_c(mx.array(theta.astype(np.float32))),
                      dtype=np.float64)
        bad = np.any((theta < LO) | (theta > HI), axis=1)
        lp[bad] = -np.inf
        return lp

    return log_prob_batch


class TestEmceeRecipe:
    def test_priors_reject_unphysical(self, recipe):
        theta = np.tile(TRUTH, (4, 1))
        theta[1, 2] = -0.2       # negative r
        theta[2, 5] = 1.7        # q1 out of box
        theta[3, 3] = 5.0        # b out of box
        lp = recipe(theta)
        assert np.isfinite(lp[0])
        assert np.all(np.isinf(lp[1:])) and np.all(lp[1:] < 0)

    def test_half_ensemble_batching_and_finite_chain(self, recipe):
        seen_shapes = []

        def spy(theta):
            seen_shapes.append(theta.shape)
            return recipe(theta)

        n_walkers = 32
        rng = np.random.default_rng(9)
        p0 = TRUTH + 1e-3 * rng.standard_normal((n_walkers, 8))
        sampler = emcee.EnsembleSampler(n_walkers, 8, spy, vectorize=True)
        sampler.run_mcmc(p0, 20, progress=False)
        chain = sampler.get_chain()
        assert np.isfinite(chain).all()
        # documented behavior: default StretchMove batches HALF-ensembles
        assert all(s[0] == n_walkers // 2 for s in seen_shapes[1:])
