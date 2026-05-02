"""
Toy rasterizer: slow, correct, differentiable.

Given time-conditioned 3D Gaussians (from gaussian.condition_on_time), we:
  1. Transform means to camera coords and project to pixel coords (exact nonlinear pi).
  2. Compute the 2D pixel-space covariance via EWA:
         Sigma_2D = J_pi @ R @ Sigma_3D_t @ R^T @ J_pi^T + sigma_k^2 I_2
     where J_pi is the 2x3 perspective Jacobian at the camera-space mean.
     (This is the STANDARD 3DGS step -- see Zwicker et al. 2001 and also the
      Jacobian paper's eq. (18) with the camera-motion terms dropped, i.e. Case A.)
  3. Evaluate each Gaussian at every pixel and alpha-composite front-to-back
     (sorted by camera-space depth).

We deliberately DON'T optimize -- no tile binning, no early-out, no CUDA.
This is the reference implementation used to develop/debug the model.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .gaussian import GaussianParams, DerivedQuantities, TimeConditioned
from .projection import Camera, world_to_camera, perspective, perspective_jacobian


@dataclass
class ScreenGaussians:
    """2D splats ready for compositing."""
    uv: Tensor          # (N, 2)  pixel-space means
    cov2d: Tensor       # (N, 2, 2)  pixel-space covariance (incl. +sigma_k^2 I)
    alpha: Tensor       # (N,)     effective opacity
    color: Tensor       # (N, 3)
    depth: Tensor       # (N,)     camera-space Z (for sorting)
    valid: Tensor       # (N,)     bool mask: False if behind camera etc.


def project_to_screen(
    params: GaussianParams,
    tc: TimeConditioned,
    cam: Camera,
    min_depth: float = 0.01,
) -> ScreenGaussians:
    """Project time-conditioned 3D Gaussians onto the image plane.

    If the camera tensors have a different dtype than the Gaussian means, we
    cast the camera to match the Gaussians. This lets us train in float32
    against cameras defined in float64 (standard test setup) without a manual
    cast at every call site.
    """
    target_dtype = tc.V_3D_t.dtype
    target_device = tc.V_3D_t.device
    if cam.R.dtype != target_dtype or cam.R.device != target_device:
        cam = Camera(
            R=cam.R.to(dtype=target_dtype, device=target_device),
            c=cam.c.to(dtype=target_dtype, device=target_device),
            fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy,
        )

    # Transform mean to camera coords.
    X_cam = world_to_camera(tc.V_3D_t, cam)                     # (N, 3)
    depth = X_cam[..., 2]                                        # (N,)
    valid = depth > min_depth

    # Project to pixel space (exact pi, no linearization).
    uv = perspective(X_cam, cam)                                 # (N, 2)

    # J_pi at X_cam: (N, 2, 3).
    J_pi = perspective_jacobian(X_cam, cam)                      # (N, 2, 3)

    # Rotate Sigma_3D_t into camera coords: Sigma_cam = R Sigma_3D_t R^T.
    R = cam.R                                                    # (3, 3)
    Sigma_cam = R @ tc.Sigma_3D_t @ R.T                          # (N, 3, 3)

    # EWA: Sigma_2D = J_pi @ Sigma_cam @ J_pi^T + sigma_k^2 I_2.
    cov2d = J_pi @ Sigma_cam @ J_pi.transpose(-1, -2)            # (N, 2, 2)
    # Regularize with isotropic pixel blur.
    eye2 = torch.eye(2, dtype=cov2d.dtype, device=cov2d.device)
    cov2d = cov2d + params.sigma_k * eye2

    return ScreenGaussians(
        uv=uv, cov2d=cov2d, alpha=tc.alpha_eff, color=params.color,
        depth=depth, valid=valid,
    )


def eval_2d_gaussian(uv: Tensor, mu: Tensor, cov: Tensor) -> Tensor:
    """Evaluate an UNNORMALIZED 2D Gaussian at pixel grid uv.

    Following standard 3DGS convention: we use the value in [0, 1] with peak 1
    at the mean, NOT the probability density. (The peak normalization constant
    is absorbed into the learned opacity -- see Remark 18 of the Jacobian paper.)

    uv:  (H, W, 2)   pixel coordinates
    mu:  (2,)        Gaussian mean
    cov: (2, 2)      Gaussian covariance

    Returns: (H, W), values in [0, 1] (approximately; clamped below by Inf-guard).
    """
    diff = uv - mu                                          # (H, W, 2)
    # Solve cov @ x = diff  =>  x = cov^{-1} @ diff
    # Mahalanobis: diff^T cov^{-1} diff
    # For 2x2 we can invert analytically.
    a, b, c, d = cov[0, 0], cov[0, 1], cov[1, 0], cov[1, 1]
    det = a * d - b * c
    det = det.clamp_min(1e-12)   # avoid singular covariance

    inv_a = d / det
    inv_b = -b / det
    inv_c = -c / det
    inv_d = a / det

    dx, dy = diff[..., 0], diff[..., 1]
    mah = inv_a * dx * dx + (inv_b + inv_c) * dx * dy + inv_d * dy * dy
    # Clip to avoid overflow; values below exp(-20) ~ 2e-9 round to 0 anyway.
    mah = mah.clamp(min=0.0, max=40.0)
    return torch.exp(-0.5 * mah)


def rasterize(
    sg: ScreenGaussians,
    H: int,
    W: int,
    background: Tensor | None = None,
    device=None,
    dtype=None,
) -> Tensor:
    """Alpha-composite splats onto an HxW image (front-to-back).

    Implements eq. (38) of the Jacobian paper:
        C(y | t_0) = sum_k  c_k * alpha_eff_k * p_k(y | t_0) * prod_{j<k} (1 - alpha_eff_j * p_j(y | t_0))

    Returns: (H, W, 3) image.
    """
    if device is None:
        device = sg.uv.device
    if dtype is None:
        dtype = sg.uv.dtype
    if background is None:
        background = torch.zeros(3, dtype=dtype, device=device)

    # Build pixel grid (H, W, 2). u -> columns, v -> rows.
    u_coords = torch.arange(W, dtype=dtype, device=device)
    v_coords = torch.arange(H, dtype=dtype, device=device)
    vv, uu = torch.meshgrid(v_coords, u_coords, indexing="ij")
    grid = torch.stack([uu, vv], dim=-1)                        # (H, W, 2)

    # Filter valid Gaussians and sort by depth (front-to-back).
    valid_idx = torch.nonzero(sg.valid).squeeze(-1)
    if valid_idx.numel() == 0:
        img = background.expand(H, W, 3).clone()
        return img

    depths = sg.depth[valid_idx]
    sort = torch.argsort(depths)
    order = valid_idx[sort]

    # Initialize image and transmittance.
    image = torch.zeros(H, W, 3, dtype=dtype, device=device)
    T = torch.ones(H, W, dtype=dtype, device=device)            # running transmittance

    for k in order.tolist():
        g = eval_2d_gaussian(grid, sg.uv[k], sg.cov2d[k])       # (H, W)
        alpha = (sg.alpha[k] * g).clamp(max=0.999)              # (H, W); avoid T=0 exactly
        contrib = T * alpha                                      # (H, W)
        image = image + contrib.unsqueeze(-1) * sg.color[k]
        T = T * (1.0 - alpha)

    # Composite background.
    image = image + T.unsqueeze(-1) * background

    return image
