"""MetalPlanet — analytic, differentiable transit light curves on Apple
Silicon (MLX).

Quadratic limb-darkened transits per Agol, Luger & Foreman-Mackey (2020),
with a batman-style frontend for everyday use:

    import metalplanet
    params = metalplanet.TransitParams()
    ...
    m = metalplanet.TransitModel(params, t)
    flux = m.light_curve(params)

and a float32-conditioned batched backend (``metalplanet.anvil``) for
GPU sampling with the anvil (formerly applemcmc) MCMC engine — which is
an optional integration dependency; the core needs only mlx and numpy.
"""

from .api import TransitModel, TransitParams
from .ellip import cel, cel3
from .flux import flux_dev, light_curve
from .greens import greens_transform_np, quad_g_coeffs, quad_norm
from .kepler import (kepler, kepler_E, kepler_E_sincos,
                     separation_keplerian)
from .anchored import anchor_constants, separation_anchored
from .poly import flux_dev_poly, sn_dev_poly
from .ld import q_to_u, q_to_u_np, u_to_q_np
from .metal import flux_dev_from_tau, flux_dev_metal, metal_available
from .solution import sn_dev, sn_dev_with_aux
from .vjp import flux_dev_analytic

__all__ = [
    "TransitModel",
    "TransitParams",
    "cel",
    "cel3",
    "flux_dev",
    "flux_dev_analytic",
    "flux_dev_poly",
    "sn_dev_poly",
    "light_curve",
    "flux_dev_from_tau",
    "flux_dev_metal",
    "metal_available",
    "greens_transform_np",
    "quad_g_coeffs",
    "quad_norm",
    "kepler",
    "kepler_E",
    "kepler_E_sincos",
    "separation_keplerian",
    "separation_anchored",
    "anchor_constants",
    "q_to_u",
    "q_to_u_np",
    "u_to_q_np",
    "sn_dev",
    "sn_dev_with_aux",
]

__version__ = "0.8.2"
