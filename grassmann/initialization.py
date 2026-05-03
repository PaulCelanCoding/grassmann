"""
Initialize Grassmann Gaussians (3-plane projector parameterization) from a
set of triangulated 3D points.

Phase A intentionally exposes only the `random` strategy:

  * `random` -- sample n ~ Uniform(S^3), L_raw ~ small isotropic Gaussian, and
    set mu = (t_i, X_world_i). Lets training discover orientations without
    biasing the rank-2 disk toward any particular ray. Matches the empirical
    finding that random init was the only competitive strategy under the
    legacy 2-plane parameterization (see
    docs/issues/monocular_streak_and_density_control.md).

The legacy ray-aware strategies (`lookat`, `birth`, `median`, `orthogonal`,
`tripod`) targeted the rank-1 pathology and are no longer applicable under
the 3-plane parameterization. They will be re-introduced in Phase B with
semantically corrected geometry (e.g. `frontal`: n_hat aligned with the
view ray, so the disk faces the init camera). For now, calling them raises
NotImplementedError so misuse fails fast.
"""
from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor

from .projection import Camera
from .gaussian import GaussianParams


DTYPE = torch.float64


InitStrategy = Literal["lookat", "birth", "median", "random", "orthogonal", "tripod"]


def _phase_b_only(name: str) -> None:
    raise NotImplementedError(
        f"Init strategy {name!r} targets the legacy 2-plane parameterization and is not "
        f"applicable to the 3-plane (G(3,4)) projector form. Use 'random' for Phase A; "
        f"see docs/issues/monocular_streak_and_density_control.md and the plan in "
        f"~/.claude/plans/grassmann-splatting-on-imperative-rocket.md for the Phase B "
        f"re-introduction (`frontal` etc.)."
    )


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
    ref_cam_idx: Optional[int] = None,         # unused under 3-plane random
    strategy: InitStrategy = "random",
    observability_idx: Optional[list[int]] = None,  # unused under 3-plane random
    sigma_init_sq: float = 0.02,
    opacity: float = 0.5,
    sigma_k_pixel: float = 1.0,
    sigma_k_temporal: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> GaussianParams:
    """Build a single 3-plane Grassmann Gaussian from a 3D point P at time t.

    `cameras`, `ref_cam_idx`, `observability_idx` are accepted for signature
    compatibility with the monocular dataset wiring but are unused for the
    `random` strategy (no ref camera dependence).
    """
    if strategy != "random":
        _phase_b_only(strategy)

    if color is None:
        color = torch.full((3,), 0.5, dtype=DTYPE)

    n, L_raw = _random_n_and_L(sigma_init_sq, generator=generator, dtype=DTYPE)
    mu = torch.cat(
        [torch.tensor([float(t)], dtype=DTYPE), X_world.to(dtype=DTYPE)],
        dim=0,
    )                                                   # (4,)

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
    observability: Optional[list[list[int]]] = None,
    sigma_init_sq: float = 0.02,
    opacity: float = 0.5,
    sigma_k_pixel: float = 1.0,
    sigma_k_temporal: float = 0.0,
    seed: Optional[int] = None,
) -> GaussianParams:
    """Initialize a batch of Gaussians from a set of (point, time) pairs.

    Each row (points[i], times[i]) becomes one 3-plane Gaussian via the
    `random` strategy.
    """
    if strategy != "random":
        _phase_b_only(strategy)

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

    sigma_L = (sigma_init_sq / 3.0) ** 0.5
    n_all = torch.randn(N, 4, dtype=DTYPE, generator=generator)
    n_all = n_all / n_all.norm(dim=-1, keepdim=True).clamp_min(1e-12)        # (N, 4)
    L_raw_all = torch.randn(N, 4, 3, dtype=DTYPE, generator=generator) * sigma_L  # (N, 4, 3)
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
