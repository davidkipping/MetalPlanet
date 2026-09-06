"""MetalPlanet quickstart — the batman tutorial, plus what batman can't
do: gradients.

Run:  ../anvil/.venv/bin/python examples/quickstart.py
"""

import mlx.core as mx
import numpy as np

import metalplanet

# -- 1. the batman workflow ------------------------------------------------
params = metalplanet.TransitParams()
params.t0 = 0.0
params.per = 1.0
params.rp = 0.1
params.a = 15.0
params.inc = 87.0
params.ecc = 0.0
params.w = 90.0
params.u = [0.1, 0.3]
params.limb_dark = "quadratic"

t = np.linspace(-0.05, 0.05, 1000)
m = metalplanet.TransitModel(params, t)
flux = m.light_curve(params)
print(f"transit depth ~ {1 - flux.min():.6f} "
      f"(rp^2 = {params.rp**2:.6f} + limb darkening)")

# vary parameters between calls, batman-style
for rp in (0.05, 0.1, 0.15):
    params.rp = rp
    print(f"  rp={rp:.2f}: min flux {m.light_curve(params).min():.6f}")
params.rp = 0.1

# eccentric orbit + supersampling
params.ecc = 0.3
params.w = 45.0
m_ecc = metalplanet.TransitModel(params, t, supersample_factor=7,
                                 exp_time=0.001)
print(f"eccentric, supersampled: min flux "
      f"{m_ecc.light_curve(params).min():.6f}")

# -- 2. gradients (the point of the MLX backend) ---------------------------
# d(flux)/d(rp, u1, u2) at every time sample, via the analytic VJP core
z = mx.array(np.abs(np.linspace(-1.3, 1.3, 9)), dtype=mx.float64)

def total_dev(rp, u1, u2):
    return mx.sum(metalplanet.flux_dev_analytic(z, rp, u1, u2))

with mx.stream(mx.cpu):
    g = mx.grad(total_dev, argnums=(0, 1, 2))(
        mx.array(0.1, dtype=mx.float64),
        mx.array(0.1, dtype=mx.float64),
        mx.array(0.3, dtype=mx.float64))
    print("d sum(flux_dev) / d(rp, u1, u2) =",
          [f"{float(x):+.5f}" for x in g])

print("\nFor GPU sampling with thousands of MCMC chains, see "
      "metalplanet.anvil and examples/injection_recovery.py")
