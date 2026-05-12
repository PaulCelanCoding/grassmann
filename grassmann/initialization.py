"""
Initialize Grassmann Gaussians (3-plane projector parameterization) from a
set of triangulated 3D points.

Available strategies:

  * `random` -- sample n ~ Uniform(S^3), L_raw ~ small isotropic Gaussian, and
    set mu = (t_i, X_world_i). Lets training discover orientations without
    biasing the rank-2 disk toward any particular ray.

  * `spatial_slice` -- n = e_0 for every Gaussian; the disk lies in the
    spatial slice {t=0}, i.e. a static-3DGS-like rank-3 blob. v7-doc §7.2.

  * `frontal` -- n_hat is the view ray from the init camera through the
    point, lifted to 4D as n = (0, d_hat). The 4-plane E_n contains the
    time axis and the two in-image-plane spatial directions; after
    condition_on_time the spatial-pure cov is rank-2 with kernel along
    d_hat -- i.e. the splat's flat face is parallel to the image plane of
    its init camera. Phase B re-introduction (see plan in
    ~/.claude/plans/grassmann-splatting-on-imperative-rocket.md).

The other legacy ray-aware strategies (`lookat`, `birth`, `median`,
`orthogonal`, `tripod`) targeted the rank-1 pathology and are not
applicable to the 3-plane parameterization; calling them raises
NotImplementedError so misuse fails fast.
"""
from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor

from .projection import Camera
from .gaussian import GaussianParams


DTYPE = torch.float64


InitStrategy = Literal["lookat", "birth", "median", "random", "orthogonal", "tripod",
                       "spatial_slice", "frontal"]


_PHASE_A_OK = ("random", "spatial_slice", "frontal")


def _phase_b_only(name: str) -> None:
    raise NotImplementedError(
        f"Init strategy {name!r} targets the legacy 2-plane parameterization and is not "
        f"applicable to the 3-plane (G(3,4)) projector form. Use 'random' for Phase A; "
        f"see results/rca/monocular_streak_and_density_control.md and the plan in "
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


def _frontal_n(X_world: Tensor, cam_center: Tensor, *, dtype: torch.dtype) -> Tensor:
    """4D plane normal for the `frontal` strategy: n = (0, d_hat) where d_hat
    is the unit view ray from `cam_center` to `X_world`.

    The 3-plane E_n then contains the time axis and the two in-image-plane
    spatial directions; after condition_on_time the spatial-pure cov is
    rank-2 with kernel along d_hat (splat face parallel to image plane).
    """
    d = X_world.to(dtype) - cam_center.to(dtype)
    d = d / d.norm().clamp_min(1e-12)                       # (3,)
    n = torch.zeros(4, dtype=dtype)
    n[1:] = d
    return n


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

    `cameras`, `ref_cam_idx`, `observability_idx` are unused for `random`
    and `spatial_slice` (no ref camera dependence). For `frontal` the init
    camera is `cameras[observability_idx[len/2]]` if observability is given,
    else `cameras[ref_cam_idx]`, else `cameras[len(cameras)//2]`.
    """
    if strategy not in _PHASE_A_OK:
        _phase_b_only(strategy)

    if color is None:
        color = torch.full((3,), 0.5, dtype=DTYPE)

    if strategy == "spatial_slice":
        n = torch.zeros(4, dtype=DTYPE); n[0] = 1.0                       # n = e_0
        sigma_L = (sigma_init_sq / 3.0) ** 0.5
        L_raw = torch.randn(4, 3, dtype=DTYPE, generator=generator) * sigma_L
    elif strategy == "frontal":
        if not cameras:
            raise ValueError("frontal strategy requires `cameras`")
        if observability_idx:
            cam = cameras[observability_idx[len(observability_idx) // 2]]
        elif ref_cam_idx is not None:
            cam = cameras[ref_cam_idx]
        else:
            cam = cameras[len(cameras) // 2]
        n = _frontal_n(X_world, cam.c, dtype=DTYPE)
        sigma_L = (sigma_init_sq / 3.0) ** 0.5
        L_raw = torch.randn(4, 3, dtype=DTYPE, generator=generator) * sigma_L
    else:
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


def compute_knn_sigma_init_sq(
    points: Tensor,                                # (N, 3)
    times: Tensor,                                 # (N,)
    *,
    k: int = 3,
    alpha_t: float = 0.0,
    eps: float = 1e-6,
) -> Tensor:
    """#3.1: per-point σ²_init from k-NN distance in 4D.

    Returns a (N,) tensor where σ²_init[i] = mean of the k smallest 4D squared
    distances from point i to other points (in [Δx, sqrt(alpha_t)·Δt] space).
    The standard 3DGS heuristic uses k=3 and floors the result.

    For N <= k+1, falls back to a global default of 0.02 to avoid degenerate
    cases on tiny clouds.
    """
    N = points.shape[0]
    if N <= k + 1:
        return torch.full((N,), 0.02, dtype=points.dtype, device=points.device)
    # Build (N, 4) feature in [x, y, z, sqrt(alpha_t)*t] units.
    feat = torch.cat([points, (alpha_t ** 0.5) * times.to(points.dtype).unsqueeze(-1)], dim=-1)
    # Pairwise squared distances. For N up to ~1e5 this fits in L4 memory.
    d2 = torch.cdist(feat, feat, p=2.0).pow(2)                              # (N, N)
    d2.fill_diagonal_(float("inf"))
    knn = d2.topk(k, dim=-1, largest=False).values                          # (N, k)
    sigma_sq = knn.mean(dim=-1).clamp_min(eps)                              # (N,)
    return sigma_sq


def init_gaussians_from_points(
    points: Tensor,                                # (N, 3)
    times: Tensor,                                 # (N,)
    cameras: list[Camera],
    *,
    colors: Optional[Tensor] = None,               # (N, 3)
    strategy: InitStrategy = "random",
    observability: Optional[list[list[int]]] = None,
    sigma_init_sq: float | Tensor = 0.02,
    opacity: float = 0.5,
    sigma_k_pixel: float = 1.0,
    sigma_k_temporal: float = 0.0,
    seed: Optional[int] = None,
) -> GaussianParams:
    """Initialize a batch of Gaussians from a set of (point, time) pairs.

    Each row (points[i], times[i]) becomes one 3-plane Gaussian. Strategy
    is one of `random`, `spatial_slice`, `frontal`. For `frontal` each
    point i picks its init camera from
    `cameras[observability[i][len(observability[i])//2]]` if observability
    is non-empty for that point, else `cameras[len(cameras)//2]`.
    """
    if strategy not in _PHASE_A_OK:
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

    # σ_init_sq may be a scalar (legacy) or per-point Tensor (#3.1 k-NN).
    if isinstance(sigma_init_sq, Tensor):
        if sigma_init_sq.shape != (N,):
            raise ValueError(
                f"per-point sigma_init_sq must have shape ({N},); got {tuple(sigma_init_sq.shape)}"
            )
        sigma_L_per = (sigma_init_sq.to(dtype=DTYPE) / 3.0).clamp_min(1e-12).sqrt()  # (N,)
    else:
        sigma_L_per = torch.full((N,), float(sigma_init_sq), dtype=DTYPE)
        sigma_L_per = (sigma_L_per / 3.0).sqrt()
    sigma_L = (float(sigma_init_sq) / 3.0) ** 0.5 if not isinstance(sigma_init_sq, Tensor) else None
    if strategy == "spatial_slice":
        # v7-doc default §7.2: n = e_0 for every Gaussian. The plane E_n is
        # the spatial slice {x_0 = 0}; the Gaussian is rank-3 (a static-3DGS
        # blob) at init and tilts into a dynamic disk via the bridge of
        # Prop 5.3 only when the optimizer pushes n away from e_0.
        # Requires the soft-clamp to avoid NaN at Sigma_tt_pure = 0; see
        # `clamp_mode='soft'` in compute_derived / condition_on_time.
        n_all = torch.zeros(N, 4, dtype=DTYPE)
        n_all[:, 0] = 1.0                                                    # n = e_0
    elif strategy == "frontal":
        if not cameras:
            raise ValueError("frontal strategy requires `cameras`")
        T_cams = len(cameras)
        cam_centers = torch.stack(
            [c.c.to(dtype=DTYPE) for c in cameras], dim=0
        )                                                                    # (T, 3)
        # Per-point init-camera index: median observed frame, fallback to mid.
        if observability is not None:
            cam_idx = torch.tensor(
                [
                    obs[len(obs) // 2] if obs else T_cams // 2
                    for obs in observability
                ],
                dtype=torch.long,
            )                                                                # (N,)
        else:
            cam_idx = torch.full((N,), T_cams // 2, dtype=torch.long)
        C_per_pt = cam_centers[cam_idx]                                      # (N, 3)
        d = points.to(dtype=DTYPE) - C_per_pt                                # (N, 3)
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-12)                # (N, 3) unit ray
        n_all = torch.zeros(N, 4, dtype=DTYPE)
        n_all[:, 1:] = d                                                     # n = (0, d_hat)
    else:
        n_all = torch.randn(N, 4, dtype=DTYPE, generator=generator)
        n_all = n_all / n_all.norm(dim=-1, keepdim=True).clamp_min(1e-12)    # (N, 4)
    # Per-point σ_L scaling: L_raw[i] ~ N(0, σ_L_per[i]² I).
    L_raw_all = torch.randn(N, 4, 3, dtype=DTYPE, generator=generator)
    L_raw_all = L_raw_all * sigma_L_per.view(N, 1, 1)                       # (N, 4, 3)
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
