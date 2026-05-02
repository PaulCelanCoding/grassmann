"""
Phase 7: Fast rasterization via diff-gaussian-rasterization (GPU/CUDA).

Adapter between our Grassmann model and the original Inria 3DGS rasterizer.
The Jacobian paper §9.1 and §10 specifically designed the 3D-lifted approach
so that we can feed `(V_3D(t_0), Σ_3D(t_0), α_eff, color)` directly into an
UNMODIFIED 3D Gaussian Splatting rasterizer. This module does exactly that.

Requirements:
  - NVIDIA CUDA GPU.
  - `pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git`
    (compiles a CUDA extension; requires CUDA toolkit + nvcc installed).

If the package isn't available, `fast_rasterize()` falls back to our toy
rasterizer automatically, so all tests and CPU runs still work.

The API we wrap:

    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings, GaussianRasterizer,
    )

    raster_settings = GaussianRasterizationSettings(
        image_height, image_width,
        tanfovx, tanfovy,
        bg, scale_modifier, viewmatrix, projmatrix,
        sh_degree, campos, prefiltered, debug,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rendered_image, radii = rasterizer(
        means3D, means2D,
        shs=None, colors_precomp=colors,
        opacities=opacities,
        scales=None, rotations=None,
        cov3D_precomp=cov3D,    # (N, 6) upper-triangular
    )

The key conversion is:
  Sigma_3D(t_0)  (N, 3, 3)  -->  cov3D_precomp  (N, 6)
with cov3D row i = [Σ[0,0], Σ[0,1], Σ[0,2], Σ[1,1], Σ[1,2], Σ[2,2]].
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .gaussian import GaussianParams, compute_derived, condition_on_time
from .projection import Camera
from .rasterizer import project_to_screen, rasterize as toy_rasterize


# ---- Availability check ---------------------------------------------------

_DGR_AVAILABLE: Optional[bool] = None
_GaussianRasterizationSettings = None
_GaussianRasterizer = None


def is_available() -> bool:
    """True if diff-gaussian-rasterization is importable AND a CUDA device exists.

    Result is cached to avoid repeated import attempts.
    """
    global _DGR_AVAILABLE, _GaussianRasterizationSettings, _GaussianRasterizer
    if _DGR_AVAILABLE is not None:
        return _DGR_AVAILABLE
    try:
        from diff_gaussian_rasterization import (  # type: ignore
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )
        _GaussianRasterizationSettings = GaussianRasterizationSettings
        _GaussianRasterizer = GaussianRasterizer
        _DGR_AVAILABLE = torch.cuda.is_available()
    except ImportError:
        _DGR_AVAILABLE = False
    return _DGR_AVAILABLE


# ---- Geometry helpers -----------------------------------------------------

def sigma3d_to_cov6(Sigma_3D: Tensor) -> Tensor:
    """Pack 3x3 symmetric covariance into 6-element upper-triangular form.

    Input:  (N, 3, 3) symmetric
    Output: (N, 6)  with columns [xx, xy, xz, yy, yz, zz]
    """
    return torch.stack([
        Sigma_3D[..., 0, 0],
        Sigma_3D[..., 0, 1],
        Sigma_3D[..., 0, 2],
        Sigma_3D[..., 1, 1],
        Sigma_3D[..., 1, 2],
        Sigma_3D[..., 2, 2],
    ], dim=-1)


def camera_to_view_matrix(cam: Camera) -> Tensor:
    """Build the 4x4 world-to-camera view matrix.

    The 3DGS rasterizer uses the convention:
        X_cam = X_world @ viewmatrix  (note: left-multiply the ROW vector)
    So viewmatrix is the TRANSPOSE of [R | -Rc; 0 1]:
        [[ R_00  R_10  R_20  0 ],
         [ R_01  R_11  R_21  0 ],
         [ R_02  R_12  R_22  0 ],
         [ -(Rc)_0  -(Rc)_1  -(Rc)_2  1 ]]

    That is the standard glm view matrix convention used by the 3DGS repo.
    """
    dtype = cam.R.dtype
    device = cam.R.device
    R = cam.R                         # (3, 3)
    c = cam.c                         # (3,)
    Rc = R @ c                        # (3,)
    V = torch.zeros(4, 4, dtype=dtype, device=device)
    V[:3, :3] = R.T                   # transpose for row-vector convention
    V[3, :3] = -Rc
    V[3, 3] = 1.0
    return V


def camera_to_proj_matrix(cam: Camera, H: int, W: int,
                           znear: float = 0.01, zfar: float = 100.0) -> Tensor:
    """Build the 4x4 world-to-clip projection matrix expected by diff-gaussian-rasterization.

    The rasterizer expects `projmatrix = viewmatrix @ K_projective` where
    K_projective is the glm-style perspective projection that maps the view
    frustum to NDC [-1, 1]^3.

    For our pinhole camera with (fx, fy, cx, cy):
        tanfovx = W / (2 * fx)
        tanfovy = H / (2 * fy)
    and the projection matrix (in row-vector convention) is:
        [[ 1/tanfovx                0                    0              0 ],
         [     0               1/tanfovy                 0              0 ],
         [     0                    0            zfar/(zfar-znear)     1 ],
         [     0                    0         -zfar*znear/(zfar-znear) 0 ]]

    Actually -- the 3DGS rasterizer is known to require that the FULL
    projmatrix = viewmatrix @ proj. We pre-multiply here.
    """
    dtype = cam.R.dtype
    device = cam.R.device
    tanfovx = W / (2.0 * cam.fx)
    tanfovy = H / (2.0 * cam.fy)

    P = torch.zeros(4, 4, dtype=dtype, device=device)
    P[0, 0] = 1.0 / tanfovx
    P[1, 1] = 1.0 / tanfovy
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = 1.0
    P[3, 2] = -(zfar * znear) / (zfar - znear)

    V = camera_to_view_matrix(cam)
    return V @ P   # full world-to-clip


def compute_tanfov(cam: Camera, H: int, W: int) -> tuple[float, float]:
    """tan(fov/2) for x and y; required by the rasterizer."""
    tanfovx = W / (2.0 * cam.fx)
    tanfovy = H / (2.0 * cam.fy)
    return float(tanfovx), float(tanfovy)


# ---- High-level fast rasterizer -------------------------------------------

@dataclass
class FastRasterConfig:
    """Configuration knobs that don't fit naturally in the Camera."""
    znear: float = 0.01
    zfar: float = 100.0
    scale_modifier: float = 1.0       # passed through to raster_settings
    sh_degree: int = 0                # we use colors_precomp
    prefiltered: bool = False
    debug: bool = False


def fast_rasterize(
    params: GaussianParams,
    t_0: float,
    cam: Camera,
    H: int, W: int,
    *,
    background: Optional[Tensor] = None,
    config: Optional[FastRasterConfig] = None,
    force_fallback: bool = False,
) -> Tensor:
    """Render the current model using the CUDA 3DGS rasterizer if available,
    otherwise fall back to our toy rasterizer.

    params:     current Grassmann Gaussians.
    t_0:        time instant to render.
    cam:        camera (static).
    H, W:       image dimensions.
    background: (3,) RGB background in [0, 1]. Defaults to black.
    config:     optional rasterizer settings.
    force_fallback: if True, always use the toy rasterizer (useful for testing
                    that the toy + fast paths agree).

    Returns: (H, W, 3) rendered image.
    """
    if config is None:
        config = FastRasterConfig()
    if background is None:
        background = torch.zeros(3, dtype=params.p_im.dtype, device=params.p_im.device)

    # Always compute the derived + time-conditioned quantities.
    derived = compute_derived(params)
    tc = condition_on_time(params, derived, t_0)

    use_fast = (not force_fallback) and is_available() and params.p_im.is_cuda

    if not use_fast:
        sg = project_to_screen(params, tc, cam)
        bg = background.to(dtype=params.p_im.dtype, device=params.p_im.device)
        return toy_rasterize(sg, H=H, W=W, background=bg)

    # ---- Fast CUDA path ----
    assert _GaussianRasterizationSettings is not None
    assert _GaussianRasterizer is not None

    dtype = params.p_im.dtype
    device = params.p_im.device

    # Move camera tensors to the same device/dtype.
    cam_dev = Camera(
        R=cam.R.to(dtype=dtype, device=device),
        c=cam.c.to(dtype=dtype, device=device),
        fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy,
    )

    tanfovx, tanfovy = compute_tanfov(cam_dev, H, W)
    view_mat = camera_to_view_matrix(cam_dev)
    proj_mat = camera_to_proj_matrix(cam_dev, H, W, config.znear, config.zfar)
    campos = cam_dev.c.to(dtype=dtype, device=device)

    raster_settings = _GaussianRasterizationSettings(
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
    rasterizer = _GaussianRasterizer(raster_settings=raster_settings)

    # Inputs to the rasterizer.
    N = tc.V_3D_t.shape[0]
    means3D = tc.V_3D_t                                          # (N, 3)
    # means2D: a dummy tensor for 2D-gradient tracking (used by adaptive densification
    # in standard 3DGS; we don't need it for forward, but the API requires it).
    means2D = torch.zeros_like(means3D, requires_grad=True)
    colors_precomp = params.color                                # (N, 3) in [0, 1]
    opacities = tc.alpha_eff.unsqueeze(-1)                       # (N, 1)
    cov3D_precomp = sigma3d_to_cov6(tc.Sigma_3D_t)               # (N, 6)

    # Call the CUDA kernel.
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacities,
        scales=None,
        rotations=None,
        cov3D_precomp=cov3D_precomp,
    )
    # Output shape is (3, H, W); transpose to (H, W, 3) to match our convention.
    return rendered_image.permute(1, 2, 0).contiguous()
