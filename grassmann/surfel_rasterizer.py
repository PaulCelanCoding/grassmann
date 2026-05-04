"""
Surfel-rasterizer adapter for the Grassmann pipeline (2DGS A/B probe).

Background: our `fast_rasterizer.fast_rasterize` packs the rank-2 Σ_3D(t_0) into a
6-element cov3D_precomp for the Inria 3DGS rasterizer, after lifting rank-2 → rank-3
with an isotropic σ_lift² I floor (~1cm² in scene units). The lift is a numerical
accommodation of the EWA kernel's invertibility requirement, not a modelling choice.

This module routes the same Σ_3D(t_0) through `diff_surfel_rasterization` (Huang
et al., SIGGRAPH 2024), which uses perspective-correct ray-plane intersection
and accepts an explicit (scale_u, scale_v, R_disk) parameterization of a 2D disk.
No rank lift is needed.

Adapter logic per Gaussian:
    1. eigvals, eigvecs = eigh(Σ_3D(t_0))  with eigvals ascending.
    2. Drop eigvals[..., 0] (the rank-2 kernel direction).
    3. scales = sqrt(clamp(eigvals[..., 1:], min=1e-6))   shape (N, 2).
    4. Build R = [v_max | v_mid | v_min] as a proper rotation (det = +1).
    5. quaternion = matrix_to_quat(R).
    6. Pass to surfel rasterizer with scales(N,2), rotations(N,4).

The surfel rasterizer returns (image, radii, allmap) where allmap has 7 channels
(see RENDER_PKG_KEYS).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .gaussian import GaussianParams, compute_derived, condition_on_time
from .projection import Camera

# Reuse helpers from the gaussian-rasterizer adapter so view/proj matrices stay
# bit-identical between the two paths.
from .fast_rasterizer import (
    camera_to_view_matrix, camera_to_proj_matrix, compute_tanfov,
)


# ---- Lazy import (matches fast_rasterizer.py pattern) ------------------------

_DSR_AVAILABLE: Optional[bool] = None
_SurfelSettings = None
_SurfelRasterizer = None


def _try_import() -> bool:
    global _DSR_AVAILABLE, _SurfelSettings, _SurfelRasterizer
    if _DSR_AVAILABLE is not None:
        return _DSR_AVAILABLE
    try:
        from diff_surfel_rasterization import (  # type: ignore
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )
        _SurfelSettings = GaussianRasterizationSettings
        _SurfelRasterizer = GaussianRasterizer
        _DSR_AVAILABLE = True
    except Exception:
        _DSR_AVAILABLE = False
    return _DSR_AVAILABLE


def is_available() -> bool:
    return _try_import()


# ---- Eigendecomp adapter -----------------------------------------------------

def sigma3d_to_disk(
    sigma_3d_t: Tensor,
    eigval_floor: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Eigendecompose rank-2 Σ_3D(t_0) → (scales[N,2], quaternion[N,4]).

    `sigma_3d_t`: (N, 3, 3) symmetric PSD, generically rank 2.
    `eigval_floor`: tiny positive floor applied to all eigvals before sqrt;
                    keeps the backward pass of eigh away from exact zero.

    Returns:
        scales: (N, 2), the two larger sqrt-eigvals (in-plane disk extents).
        quat:   (N, 4) real-first (w, x, y, z), unit quaternion such that
                R = quat_to_matrix(q) has columns [v_max, v_mid, v_min] with
                det(R) = +1. Local x = largest in-plane axis, local y = mid,
                local z = disk normal.
    """
    # Symmetrize (eigh requires exact symmetry; numerical noise from outer-product
    # construction can break this).
    sym = 0.5 * (sigma_3d_t + sigma_3d_t.transpose(-1, -2))

    eigvals, eigvecs = torch.linalg.eigh(sym)               # ascending
    eigvals = eigvals.clamp_min(eigval_floor)               # (N, 3)

    # Reorder: largest first along the last axis. eigh returns ascending, so
    # reverse to descending: [λ_max, λ_mid, λ_min].
    eigvals_desc = eigvals.flip(-1)                         # (N, 3)
    eigvecs_desc = eigvecs.flip(-1)                         # (N, 3, 3) columns reversed

    # In-plane scales (drop λ_min; that's the disk normal).
    scales = torch.sqrt(eigvals_desc[..., :2])              # (N, 2)

    # R has columns [v_max, v_mid, v_min].
    R = eigvecs_desc                                        # (N, 3, 3)

    # Sign convention. eigh returns eigvecs with arbitrary per-iter sign, which
    # leaks into rotation matrices / quaternions and produces inconsistent
    # gradient directions in L_raw space across iters (Adam's momentum then
    # builds zero net signal on rotation-affecting params). We enforce a
    # canonical sign by making the largest-absolute-component of each eigvec
    # column positive, then ensuring det(R)=+1 (flip v_min if needed).
    abs_R = R.abs()
    argmax_per_col = abs_R.argmax(dim=-2)                   # (N, 3); which row has max |v_ij|
    # Gather the value at that max-row for each column.
    rng = torch.arange(R.shape[-1], device=R.device)
    sign_per_col = torch.where(
        R.gather(-2, argmax_per_col.unsqueeze(-2)).squeeze(-2) < 0,
        torch.full_like(argmax_per_col, -1, dtype=R.dtype),
        torch.full_like(argmax_per_col,  1, dtype=R.dtype),
    )                                                        # (N, 3)
    R = R * sign_per_col.unsqueeze(-2)                       # broadcast over rows

    # After per-column sign, det may be -1; restore det(R)=+1 by flipping v_min.
    det = torch.linalg.det(R)                                # (N,)
    flip_z = torch.where(det < 0,
                         torch.full_like(det, -1.0),
                         torch.full_like(det,  1.0))         # (N,)
    R = R.clone()
    R[..., :, 2] = R[..., :, 2] * flip_z.unsqueeze(-1)

    quat = _matrix_to_quat(R)                                # (N, 4) wxyz
    return scales, quat


def _matrix_to_quat(R: Tensor) -> Tensor:
    """Convert batched 3x3 rotation matrices to unit quaternions, wxyz convention.

    Uses the stable branch-based formula (Shepperd/Shoemake): pick the largest
    of (1+R00+R11+R22, 1+R00-R11-R22, ...) to avoid divide-by-near-zero.
    """
    # R has shape (..., 3, 3). Compute the four trace-like quantities.
    r00, r01, r02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    r10, r11, r12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    r20, r21, r22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    # 4 candidates (proportional to 4*qw², 4*qx², 4*qy², 4*qz² up to sign).
    cand = torch.stack([
        1.0 + r00 + r11 + r22,
        1.0 + r00 - r11 - r22,
        1.0 - r00 + r11 - r22,
        1.0 - r00 - r11 + r22,
    ], dim=-1)                                              # (..., 4)
    cand = cand.clamp_min(0.0)                              # numerical safety
    idx = cand.argmax(dim=-1)                               # (...,)

    # Compute the four cases, then gather. We can't avoid computing all four
    # because torch.where forks gradients; this is fine for a (N, 3, 3) batch.
    sqrt_safe = lambda v: torch.sqrt(v.clamp_min(1e-20))

    # Case 0: qw is largest.
    s0 = sqrt_safe(cand[..., 0]) * 2.0
    qw0 = 0.25 * s0
    qx0 = (r21 - r12) / s0
    qy0 = (r02 - r20) / s0
    qz0 = (r10 - r01) / s0

    # Case 1: qx is largest.
    s1 = sqrt_safe(cand[..., 1]) * 2.0
    qw1 = (r21 - r12) / s1
    qx1 = 0.25 * s1
    qy1 = (r01 + r10) / s1
    qz1 = (r02 + r20) / s1

    # Case 2: qy is largest.
    s2 = sqrt_safe(cand[..., 2]) * 2.0
    qw2 = (r02 - r20) / s2
    qx2 = (r01 + r10) / s2
    qy2 = 0.25 * s2
    qz2 = (r12 + r21) / s2

    # Case 3: qz is largest.
    s3 = sqrt_safe(cand[..., 3]) * 2.0
    qw3 = (r10 - r01) / s3
    qx3 = (r02 + r20) / s3
    qy3 = (r12 + r21) / s3
    qz3 = 0.25 * s3

    qw = torch.where(idx == 0, qw0,
         torch.where(idx == 1, qw1,
         torch.where(idx == 2, qw2, qw3)))
    qx = torch.where(idx == 0, qx0,
         torch.where(idx == 1, qx1,
         torch.where(idx == 2, qx2, qx3)))
    qy = torch.where(idx == 0, qy0,
         torch.where(idx == 1, qy1,
         torch.where(idx == 2, qy2, qy3)))
    qz = torch.where(idx == 0, qz0,
         torch.where(idx == 1, qz1,
         torch.where(idx == 2, qz2, qz3)))

    quat = torch.stack([qw, qx, qy, qz], dim=-1)             # wxyz
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    # Canonicalize the q vs -q ambiguity (both represent the same rotation):
    # enforce qw ≥ 0 by flipping sign of any quaternion with negative qw.
    sign = torch.where(quat[..., 0:1] < 0,
                       torch.full_like(quat[..., 0:1], -1.0),
                       torch.full_like(quat[..., 0:1],  1.0))
    quat = quat * sign
    return quat


# ---- Render entry point ------------------------------------------------------

# allmap channel layout (from 2DGS gaussian_renderer/__init__.py).
RENDER_PKG_KEYS = {
    "rend_alpha": (1, 2),       # allmap[1:2]
    "rend_normal": (2, 5),      # allmap[2:5]
    "rend_dist":  (6, 7),       # allmap[6:7]
    "expected_depth": (0, 1),   # allmap[0:1]
    "median_depth": (5, 6),     # allmap[5:6]
}


@dataclass
class SurfelRasterConfig:
    znear: float = 0.01
    zfar: float = 100.0
    scale_modifier: float = 1.0
    sh_degree: int = 0
    prefiltered: bool = False
    debug: bool = False
    eigval_floor: float = 1e-6
    sigma_3d_blur: float = 0.0
    # Anisotropic jitter added to Σ_3D(t_0) before eigh to break degeneracy.
    # eigh's backward has 1/Δλ singularities when in-plane eigvals are close
    # (mathematically: random rotations in the ambiguous eigenbasis). 13.6%
    # of A1's converged Gaussians have (λ_max-λ_mid)/λ_max < 0.05. The jitter
    # is constructed as ε * (J + J^T) with J random and small.
    # 0 disables; 1e-5 to 1e-3 are reasonable.
    eigh_jitter: float = 0.0


def surfel_rasterize(
    params: GaussianParams,
    t_0: float,
    cam: Camera,
    H: int, W: int,
    *,
    background: Optional[Tensor] = None,
    config: Optional[SurfelRasterConfig] = None,
    means2d_capture: Optional[list] = None,
    static_baseline: bool = False,
    return_aux: bool = False,
):
    """Render via diff_surfel_rasterization.

    Returns:
        if return_aux is False:  rendered_image (H, W, 3)
        if return_aux is True:   (rendered_image (H, W, 3), aux dict with keys
                                  matching RENDER_PKG_KEYS, each (C, H, W) tensor)

    Raises:
        RuntimeError if diff_surfel_rasterization is not importable or no CUDA.
    """
    if config is None:
        config = SurfelRasterConfig()
    if background is None:
        background = torch.zeros(3, dtype=params.n.dtype, device=params.n.device)

    if not (_try_import() and params.n.is_cuda):
        raise RuntimeError(
            "surfel_rasterize requires CUDA + diff_surfel_rasterization installed; "
            "fall back to the gaussian rasterizer or check Modal image."
        )

    # Compute time-conditioned spatial Gaussian (rank-2 by Proposition rank2).
    derived = compute_derived(params)
    tc = condition_on_time(params, derived, t_0, static=static_baseline)

    dtype = params.n.dtype
    device = params.n.device

    cam_dev = Camera(
        R=cam.R.to(dtype=dtype, device=device),
        c=cam.c.to(dtype=dtype, device=device),
        fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy,
    )

    tanfovx, tanfovy = compute_tanfov(cam_dev, H, W)
    view_mat = camera_to_view_matrix(cam_dev)
    proj_mat = camera_to_proj_matrix(cam_dev, H, W, config.znear, config.zfar)
    campos = cam_dev.c.to(dtype=dtype, device=device)

    raster_settings = _SurfelSettings(
        image_height=int(H),
        image_width=int(W),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=background.to(dtype=dtype, device=device),
        scale_modifier=config.scale_modifier,
        viewmatrix=view_mat,
        projmatrix=proj_mat,
        sh_degree=config.sh_degree,
        campos=campos,
        prefiltered=config.prefiltered,
        debug=config.debug,
    )
    rasterizer = _SurfelRasterizer(raster_settings=raster_settings)

    # Optional pre-eigh lift (isotropic) + jitter (anisotropic).
    sigma_3d_t = tc.Sigma_3D_t
    if config.sigma_3d_blur > 0.0:
        eye = torch.eye(3, dtype=sigma_3d_t.dtype, device=sigma_3d_t.device)
        sigma_3d_t = sigma_3d_t + (config.sigma_3d_blur ** 2) * eye
    if config.eigh_jitter > 0.0:
        # Anisotropic jitter: ε * (J + J^T) with J random per-Gaussian per-call.
        # Breaks eigh degeneracy in expectation; symmetric so Σ stays symmetric.
        N = sigma_3d_t.shape[0]
        J = torch.randn(N, 3, 3, dtype=sigma_3d_t.dtype, device=sigma_3d_t.device)
        sigma_3d_t = sigma_3d_t + config.eigh_jitter * (J + J.transpose(-1, -2))

    # Eigendecompose Σ_3D(t_0) → (scales, quat).
    scales, rotations = sigma3d_to_disk(sigma_3d_t, eigval_floor=config.eigval_floor)

    means3D = tc.V_3D_t                                          # (N, 3)
    means2D = torch.zeros_like(means3D, requires_grad=True)
    if means2d_capture is not None:
        means2d_capture.append(means2D)
    opacities = tc.alpha_eff.unsqueeze(-1)                       # (N, 1)

    use_sh = params.sh is not None and config.sh_degree > 0
    if use_sh:
        shs_in = params.sh
        colors_precomp_in = None
    else:
        shs_in = None
        colors_precomp_in = params.color

    # The surfel rasterizer returns (image, radii, allmap).
    rendered_image, radii, allmap = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs_in,
        colors_precomp=colors_precomp_in,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    image_hwc = rendered_image.permute(1, 2, 0).contiguous()     # (H, W, 3)

    if not return_aux:
        return image_hwc

    aux = {}
    for key, (a, b) in RENDER_PKG_KEYS.items():
        aux[key] = allmap[a:b]
    return image_hwc, aux
