"""
Initialize Grassmann Gaussians (3-plane projector parameterization) from a
set of triangulated 3D points.

Strategy: ``spatial_slice`` -- n = e_0 for every Gaussian; the disk lies in
the spatial slice {t=0}, i.e. a static-3DGS-like rank-3 blob. v7-doc Sec. 7.2.
Pairs with the ``soft`` clamp in `condition_on_time` (v7-doc Prop 5.3) so the
n = e_0 -> tilted-disk transition is smooth.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .projection import Camera
from .gaussian import GaussianParams


DTYPE = torch.float64


def init_gaussian_from_point(
    X_world: Tensor,
    t: float,
    cameras: list[Camera],
    *,
    color: Optional[Tensor] = None,
    sigma_init_sq: float = 0.02,
    opacity: float = 0.5,
    sigma_k_pixel: float = 1.0,
    sigma_k_temporal: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> GaussianParams:
    """Build a single 3-plane Grassmann Gaussian from a 3D point X_world at time t.

    `cameras` is accepted for signature compatibility with the batched
    `init_gaussians_from_points` and is otherwise unused.
    """
    if color is None:
        color = torch.full((3,), 0.5, dtype=DTYPE)

    n = torch.zeros(4, dtype=DTYPE)
    n[0] = 1.0                                                       # n = e_0
    sigma_L = (sigma_init_sq / 3.0) ** 0.5
    L_raw = torch.randn(4, 3, dtype=DTYPE, generator=generator) * sigma_L

    mu = torch.cat(
        [torch.tensor([float(t)], dtype=DTYPE), X_world.to(dtype=DTYPE)],
        dim=0,
    )                                                                    # (4,)

    return GaussianParams(
        n=n.unsqueeze(0),
        L_raw=L_raw.unsqueeze(0),
        mu=mu.unsqueeze(0),
        opacity=torch.tensor([opacity], dtype=DTYPE),
        color=color.unsqueeze(0),
        sigma_k_pixel=sigma_k_pixel,
        sigma_k_temporal=sigma_k_temporal,
    )


def init_gaussians_from_points(
    points: Tensor,                                # (N, 3)
    times: Tensor,                                 # (N,)
    cameras: list[Camera],
    *,
    colors: Optional[Tensor] = None,               # (N, 3)
    observability: Optional[list[list[int]]] = None,   # accepted but unused
    sigma_init_sq: float = 0.02,
    opacity: float = 0.5,
    sigma_k_pixel: float = 1.0,
    sigma_k_temporal: float = 0.0,
    seed: Optional[int] = None,
) -> GaussianParams:
    """Initialize a batch of Gaussians from (points[i], times[i]) pairs.

    Each row becomes one 3-plane Gaussian in the {t=0} spatial slice.
    `cameras` and `observability` are accepted for monocular-dataset
    signature compatibility and are otherwise unused.
    """
    N = points.shape[0]
    if colors is None:
        colors = torch.full((N, 3), 0.5, dtype=DTYPE)
    if observability is not None and len(observability) != N:
        raise ValueError(
            f"observability has length {len(observability)} but expected N={N}"
        )

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    sigma_L = (float(sigma_init_sq) / 3.0) ** 0.5
    # v7-doc Sec. 7.2: n = e_0 for every Gaussian. E_n is the spatial slice
    # {x_0 = 0}; the Gaussian is rank-3 (a static-3DGS blob) at init and tilts
    # into a dynamic disk via the bridge of Prop 5.3 only when the optimizer
    # pushes n away from e_0. Pairs with the soft clamp in condition_on_time
    # to keep Sigma_tt_pure differentiable through 0.
    n_all = torch.zeros(N, 4, dtype=DTYPE)
    n_all[:, 0] = 1.0                                                    # n = e_0

    L_raw_all = torch.randn(N, 4, 3, dtype=DTYPE, generator=generator) * sigma_L
    mu_all = torch.cat(
        [times.to(dtype=DTYPE).unsqueeze(-1), points.to(dtype=DTYPE)],
        dim=-1,
    )                                                                         # (N, 4)

    return GaussianParams(
        n=n_all,
        L_raw=L_raw_all,
        mu=mu_all,
        opacity=torch.full((N,), float(opacity), dtype=DTYPE),
        color=colors.to(dtype=DTYPE),
        sigma_k_pixel=sigma_k_pixel,
        sigma_k_temporal=sigma_k_temporal,
    )
