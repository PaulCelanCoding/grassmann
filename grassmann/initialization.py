"""
Initialize Grassmann Gaussians (3-plane projector parameterization) from a
set of triangulated 3D points.

Two strategies:

  * `random` -- sample n ~ Uniform(S^3), L_raw ~ small isotropic Gaussian, and
    set mu = (t_i, X_world_i). Lets training discover orientations without
    biasing the rank-2 disk toward any particular ray.

  * `spatial_slice` -- n = e_0 for every Gaussian; the disk lies in the
    spatial slice {t=0}, i.e. a static-3DGS-like rank-3 blob. v7-doc Sec. 7.2.
    Requires `clamp_mode='soft'` so the bridge of Prop 5.3 makes the
    n = e_0 → tilted-disk transition smooth.
"""
from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor

from .projection import Camera
from .gaussian import GaussianParams


DTYPE = torch.float64


InitStrategy = Literal["random", "spatial_slice"]


def _random_n_and_L(
    sigma_init_sq: float,
    *,
    generator: Optional[torch.Generator] = None,
    dtype: torch.dtype = DTYPE,
) -> tuple[Tensor, Tensor]:
    """Sample one random plane normal n in S^3 and one L_raw factor 4x3
    targeting an in-plane covariance of approximately sigma_init_sq * (I - nn^T).

    L_raw entries are i.i.d. N(0, sigma_L^2) with sigma_L = sqrt(sigma_init_sq / 3),
    so E[L_raw L_raw^T] = sigma_init_sq * I_4 (rank-3 in expectation; one
    direction is killed after the projector).
    """
    n = torch.randn(4, dtype=dtype, generator=generator)
    n = n / n.norm().clamp_min(1e-12)
    sigma_L = (sigma_init_sq / 3.0) ** 0.5
    L_raw = torch.randn(4, 3, dtype=dtype, generator=generator) * sigma_L
    return n, L_raw


def init_gaussian_from_point(
    X_world: Tensor,
    t: float,
    cameras: list[Camera],
    *,
    color: Optional[Tensor] = None,
    strategy: InitStrategy = "random",
    sigma_init_sq: float = 0.02,
    opacity: float = 0.5,
    sigma_k_pixel: float = 1.0,
    sigma_k_temporal: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> GaussianParams:
    """Build a single 3-plane Grassmann Gaussian from a 3D point X_world at time t.

    `cameras` is accepted for signature compatibility with the batched
    `init_gaussians_from_points` and is otherwise unused (no ref-camera
    dependence in either active strategy).
    """
    if strategy not in ("random", "spatial_slice"):
        raise ValueError(
            f"Unknown init strategy {strategy!r}. Use 'random' or 'spatial_slice'."
        )

    if color is None:
        color = torch.full((3,), 0.5, dtype=DTYPE)

    if strategy == "spatial_slice":
        n = torch.zeros(4, dtype=DTYPE)
        n[0] = 1.0                                                       # n = e_0
        sigma_L = (sigma_init_sq / 3.0) ** 0.5
        L_raw = torch.randn(4, 3, dtype=DTYPE, generator=generator) * sigma_L
    else:
        n, L_raw = _random_n_and_L(sigma_init_sq, generator=generator, dtype=DTYPE)

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
    strategy: InitStrategy = "random",
    observability: Optional[list[list[int]]] = None,   # accepted but unused
    sigma_init_sq: float = 0.02,
    opacity: float = 0.5,
    sigma_k_pixel: float = 1.0,
    sigma_k_temporal: float = 0.0,
    seed: Optional[int] = None,
) -> GaussianParams:
    """Initialize a batch of Gaussians from (points[i], times[i]) pairs.

    Each row becomes one 3-plane Gaussian via the requested strategy.
    `cameras` and `observability` are accepted for monocular-dataset
    signature compatibility and are otherwise unused.
    """
    if strategy not in ("random", "spatial_slice"):
        raise ValueError(
            f"Unknown init strategy {strategy!r}. Use 'random' or 'spatial_slice'."
        )

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
    if strategy == "spatial_slice":
        # v7-doc Sec. 7.2: n = e_0 for every Gaussian. The plane E_n is the
        # spatial slice {x_0 = 0}; the Gaussian is rank-3 (a static-3DGS blob)
        # at init and tilts into a dynamic disk via the bridge of Prop 5.3
        # only when the optimizer pushes n away from e_0. Requires
        # `clamp_mode='soft'` to keep Sigma_tt_pure differentiable through 0.
        n_all = torch.zeros(N, 4, dtype=DTYPE)
        n_all[:, 0] = 1.0                                                    # n = e_0
    else:
        n_all = torch.randn(N, 4, dtype=DTYPE, generator=generator)
        n_all = n_all / n_all.norm(dim=-1, keepdim=True).clamp_min(1e-12)    # (N, 4)

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


def sample_color_from_image(image: Tensor, uv: Tensor) -> Tensor:
    """Bilinearly sample RGB color from an image at pixel coord (u, v).

    image: (H, W, 3)
    uv: (2,) or (N, 2)
    Returns: (3,) or (N, 3).
    """
    H, W, _ = image.shape
    if uv.dim() == 1:
        u, v = uv[0].item(), uv[1].item()
        u = max(0, min(u, W - 1))
        v = max(0, min(v, H - 1))
        ui, vi = int(round(u)), int(round(v))
        return image[vi, ui]
    else:
        us = uv[:, 0].clamp(0, W - 1).round().long()
        vs = uv[:, 1].clamp(0, H - 1).round().long()
        return image[vs, us]
