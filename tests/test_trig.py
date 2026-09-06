"""Accurate fp64 sincos (works around MLX's float32-accurate fp64
sin/cos) and its use in the orbit paths."""

import mlx.core as mx
import numpy as np

from metalplanet.trig import sincos

RNG = np.random.default_rng(61)


class TestSincos64:
    def test_accuracy_broad_range(self):
        x = np.concatenate([
            RNG.uniform(-10, 10, 200_000),
            RNG.uniform(-1e4, 1e4, 100_000),
            [0.0, np.pi / 2, np.pi, -np.pi / 2, 1e-300, 3 * np.pi / 4],
        ])
        with mx.stream(mx.cpu):
            s, c = sincos(mx.array(x, dtype=mx.float64))
            s = np.array(s, dtype=np.float64)
            c = np.array(c, dtype=np.float64)
        assert np.max(np.abs(s - np.sin(x))) < 5e-16
        assert np.max(np.abs(c - np.cos(x))) < 5e-16

    def test_native_mlx_fp64_sin_is_inaccurate(self):
        """Documents the MLX limitation this module exists for; if this
        starts failing, MLX fixed fp64 sin and trig.py can be retired."""
        x = RNG.uniform(-np.pi, np.pi, 100_000)
        with mx.stream(mx.cpu):
            err = np.max(np.abs(
                np.array(mx.sin(mx.array(x, dtype=mx.float64)),
                         dtype=np.float64) - np.sin(x)))
        assert err > 1e-9, "MLX fp64 sin became accurate — simplify trig.py"

    def test_fp32_passthrough(self):
        x = RNG.uniform(-10, 10, 10_000).astype(np.float32)
        s, c = sincos(mx.array(x))
        assert s.dtype == mx.float32
        np.testing.assert_allclose(np.array(s, dtype=np.float64),
                                   np.sin(x.astype(np.float64)), atol=5e-7)
        np.testing.assert_allclose(np.array(c, dtype=np.float64),
                                   np.cos(x.astype(np.float64)), atol=5e-7)

    def test_gradients(self):
        x = mx.array(RNG.uniform(-10, 10, 1000), dtype=mx.float64)
        with mx.stream(mx.cpu):
            g_s = mx.grad(lambda v: mx.sum(sincos(v)[0]))(x)
            g_c = mx.grad(lambda v: mx.sum(sincos(v)[1]))(x)
            x_np = np.array(x, dtype=np.float64)
            np.testing.assert_allclose(np.array(g_s), np.cos(x_np),
                                       atol=1e-13)
            np.testing.assert_allclose(np.array(g_c), -np.sin(x_np),
                                       atol=1e-13)
