"""
Initialize Grassmann Gaussians from a set of triangulated 3D points.

For each (3D point P_i, time t_i), we:
  1. Pick a reference camera c_ref that observes P_i.
  2. Build a line from c_ref through P_i (a RAY).
  3. Construct a GaussianParams for that ray.
  4. Set alpha_0 so the mean sits at depth |P_i - c_ref| along the ray.
  5. Set beta_0 so the temporal center v_0 equals t_i.
  6. Initialize covariance and opacity to small/moderate defaults.
  7. Sample color from the reference image if available.

Why rays from a specific camera? As we saw in Phase 3, a line through the
CAMERA ORIGIN has c = 1, s = 0 -- e2_hat is purely temporal. No spatial drift.
For static cameras, this means the Gaussian sits exactly on the ray it was
initialized on, and any spatial motion in the scene must be learned via
sigma_ab + appropriate Gaussian density.

We use each point's CLOSEST camera (smallest angle to the ray) as the reference,
which gives the ray best aligned with the point's natural line of sight and is
most robust to small errors in the triangulated 3D position.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor

from . import quaternion as Q
from . import grassmann as G
from .projection import Camera
from .gaussian import GaussianParams


DTYPE = torch.float64


def pick_reference_camera(X_world: Tensor, cameras: list[Camera]) -> int:
    """Return the index of the camera that most directly faces X_world.

    Criterion: maximize dot(forward_k, (X - c_k).normalized()). This is the
    camera whose optical axis is most aligned with the line-of-sight to the
    point.
    """
    best_idx = 0
    best_score = -float("inf")
    for k, cam in enumerate(cameras):
        # Forward direction in world coords = R^T @ (0, 0, 1) = R[2, :] (third row of R).
        forward_world = cam.R[2]
        dir_to_pt = X_world - cam.c
        dir_to_pt = dir_to_pt / dir_to_pt.norm().clamp_min(1e-12)
        score = (forward_world * dir_to_pt).sum().item()
        if score > best_score:
            best_score = score
            best_idx = k
    return best_idx


def init_gaussian_from_point(
    X_world: Tensor,
    t: float,
    cameras: list[Camera],
    *,
    color: Optional[Tensor] = None,
    ref_cam_idx: Optional[int] = None,
    sigma_aa: float = 0.02,
    sigma_bb: float = 0.05,
    sigma_ab: float = 0.0,
    opacity: float = 0.5,
    sigma_k: float = 1.0,
) -> GaussianParams:
    """Build a single Grassmann Gaussian from a 3D point P at time t.

    X_world: (3,)
    t:       scalar
    cameras: list of K cameras
    color:   (3,) in [0, 1]; defaults to mid-gray.
    ref_cam_idx: which camera to build the ray from. Defaults to the
                 most-directly-facing camera.

    sigma_aa: variance along alpha (radial, along the ray). Controls how
              thick the Gaussian is in depth.
    sigma_bb: variance along beta (temporal). Controls how many frames the
              Gaussian persists.
    sigma_ab: coupling between alpha and beta. 0 means the Gaussian stays
              put; nonzero couples depth-change with time-change (useful if
              you know a priori that the point is moving in depth).
    opacity:  initial opacity (before sigmoid). 0.5 is a moderate default.
    sigma_k:  pixel-space blur variance.

    Returns a GaussianParams object containing a single Gaussian (N=1).
    """
    if color is None:
        color = torch.full((3,), 0.5, dtype=DTYPE)
    if ref_cam_idx is None:
        ref_cam_idx = pick_reference_camera(X_world, cameras)

    ref_cam = cameras[ref_cam_idx]

    # Direction from camera center TOWARD the point, in WORLD coordinates.
    dir_world = X_world - ref_cam.c                                        # (3,)
    dist = dir_world.norm().clamp_min(1e-8)
    u_hat_world = dir_world / dist                                          # (3,)

    # Build a line IN WORLD COORDINATES passing through ref_cam.c in direction u_hat_world.
    # To make (t, X_world) lie EXACTLY in the resulting plane E_{p,q}, we scale
    # the line's foot-of-perpendicular: use x_line = ref_cam.c / t instead of ref_cam.c.
    # This is equivalent to saying the Grassmann plane phi_t(L) = span{(t, t*y_std), (0, u_hat)},
    # which we achieve by calling line_to_pq on the scaled x_line (since that produces
    # the plane span{(1, y_std/t), (0, u_hat)} = span{(t, y_std), (0, u_hat)} = phi_t(L)).
    #
    # Edge case: if t == 0 we cannot scale. In practice t > 0 in all reasonable use cases.
    t_float = float(t)
    if abs(t_float) < 1e-8:
        # Fallback: use unscaled (gives the plane for t=1; small residual in V_k at t=0).
        x_line = ref_cam.c.unsqueeze(0)
    else:
        x_line = (ref_cam.c / t_float).unsqueeze(0)                         # (1, 3)
    u_line = u_hat_world.unsqueeze(0)                                       # (1, 3)
    p_quat, q_quat = G.line_to_pq(x_line, u_line)                           # (1, 4), (1, 4)

    # Place the Gaussian mean at X_world. In the canonical basis:
    #   v = alpha_0 * e1_hat + beta_0 * e2_hat  must have spatial part = X_world
    #                                                and time part = t.
    # We solve this as an R^4 linear system in (alpha_0, beta_0).
    e1_hat, e2_hat = G.orthonormal_basis(p_quat, q_quat)                    # (1, 4)
    # Target v in R^4: (t, X_world)
    target = torch.cat([torch.tensor([[float(t)]], dtype=DTYPE), X_world.unsqueeze(0)], dim=-1)  # (1, 4)

    # Solve: [e1_hat, e2_hat]^T @ [alpha; beta] = target
    # In practice, target may not lie exactly in span{e1_hat, e2_hat} (the plane
    # E_{p,q}) due to the way line_to_pq works. But (1, y) IS in E_{p,q} where
    # y is the standard-form point of THIS line. By construction X_world lies on
    # our line, so (1, X_world) lies in the plane spanned by (1, y) and (0, u_hat).
    # However (t, X_world) only lies in the plane if t happens to equal a specific
    # value tied to |X_world - y|. In general, the time coordinate of the canonical
    # embedding is NOT freely adjustable.
    #
    # Solution: project (t, X_world) onto the plane span{e1_hat, e2_hat} using
    # least squares. The projection gives the best-fit (alpha_0, beta_0); any
    # residual is absorbed into the loss, and training will adjust p, q to reduce it.
    # For initialization this small residual is harmless.
    #
    # Inner products (both basis vectors are orthonormal).
    alpha_val = (target * e1_hat).sum(dim=-1)                               # (1,)
    beta_val = (target * e2_hat).sum(dim=-1)                                # (1,)

    # Cholesky of Sigma_k.
    sa = np.sqrt(sigma_aa)
    L10 = sigma_ab / sa
    inner = sigma_bb - sigma_ab * sigma_ab / sigma_aa
    if inner <= 0:
        raise ValueError(f"Invalid Sigma_k (not PD): inner = {inner}")
    L11 = np.sqrt(inner)
    L = torch.tensor([[[sa, 0.0], [L10, L11]]], dtype=DTYPE)

    return GaussianParams(
        p_im=Q.imag(p_quat),
        q_im=Q.imag(q_quat),
        alpha_0=alpha_val,
        beta_0=beta_val,
        L=L,
        opacity=torch.tensor([opacity], dtype=DTYPE),
        color=color.unsqueeze(0),
        sigma_k=sigma_k,
    )


def init_gaussians_from_points(
    points: Tensor,          # (N, 3)
    times: Tensor,           # (N,)
    cameras: list[Camera],
    *,
    colors: Optional[Tensor] = None,   # (N, 3)
    sigma_aa: float = 0.02,
    sigma_bb: float = 0.05,
    sigma_ab: float = 0.0,
    opacity: float = 0.5,
    sigma_k: float = 1.0,
) -> GaussianParams:
    """Initialize a batch of Gaussians from a set of (point, time) pairs.

    Each row (points[i], times[i]) becomes one Gaussian.
    Returns a single GaussianParams containing all N Gaussians.
    """
    N = points.shape[0]
    if colors is None:
        colors = torch.full((N, 3), 0.5, dtype=DTYPE)

    per_gaussian = [
        init_gaussian_from_point(
            points[i], float(times[i].item()), cameras,
            color=colors[i],
            sigma_aa=sigma_aa, sigma_bb=sigma_bb, sigma_ab=sigma_ab,
            opacity=opacity, sigma_k=sigma_k,
        )
        for i in range(N)
    ]

    return GaussianParams(
        p_im=torch.cat([g.p_im for g in per_gaussian]),
        q_im=torch.cat([g.q_im for g in per_gaussian]),
        alpha_0=torch.cat([g.alpha_0 for g in per_gaussian]),
        beta_0=torch.cat([g.beta_0 for g in per_gaussian]),
        L=torch.cat([g.L for g in per_gaussian]),
        opacity=torch.cat([g.opacity for g in per_gaussian]),
        color=torch.cat([g.color for g in per_gaussian]),
        sigma_k=sigma_k,
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
        # Batched
        us = uv[:, 0].clamp(0, W - 1).round().long()
        vs = uv[:, 1].clamp(0, H - 1).round().long()
        return image[vs, us]
