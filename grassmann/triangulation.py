"""
Multi-view triangulation: given K >= 2 cameras and corresponding 2D pixel
observations of the same 3D point, recover the 3D point.

We implement the standard DLT (Direct Linear Transform) method, which is
closed-form, differentiable, and works for any K >= 2. For K = 2 it's
equivalent to classical stereo triangulation.

The intuition: a camera ray through pixel (u, v) with camera (R, c, K_intr) is
the set of 3D points X such that the perspective projection of X in that
camera gives (u, v). Each observation in each camera gives us 2 linear
equations in X (one from u, one from v). With K cameras, we have 2K equations
for 3 unknowns (x, y, z of X) -- over-determined. Solve by least squares via SVD.

Reference:
  Hartley & Zisserman, "Multiple View Geometry", Chapter 12.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .projection import Camera, project_static


# ---- Projection matrix in 3x4 homogeneous form ------------------------------

def projection_matrix(cam: Camera) -> Tensor:
    """Build the 3x4 projection matrix P such that
       [u, v, 1]^T ~ P @ [X, Y, Z, 1]^T  (up to a scale factor = depth).

    For a pinhole camera with intrinsics K, rotation R, center c:
       X_cam = R (X_world - c)  =>   X_cam = R X_world - R c
       x_pix = K X_cam
    In 3x4 form: P = K [R | -R c].
    """
    K_intr = torch.tensor([
        [cam.fx, 0.0,     cam.cx],
        [0.0,    cam.fy,  cam.cy],
        [0.0,    0.0,     1.0],
    ], dtype=cam.R.dtype, device=cam.R.device)
    Rc = cam.R @ cam.c.unsqueeze(-1)                                 # (3, 1)
    Rt = torch.cat([cam.R, -Rc], dim=1)                              # (3, 4)
    return K_intr @ Rt                                                # (3, 4)


# ---- DLT triangulation -------------------------------------------------------

def triangulate_point_dlt(
    cameras: list[Camera],
    pixel_observations: Tensor,   # (K, 2)
) -> Tensor:
    """Triangulate a single 3D point from K observations using DLT.

    For each camera k with projection P_k = [p_k^1; p_k^2; p_k^3] (rows),
    and observation (u_k, v_k), the projection equation gives:
        u_k = (p_k^1 X) / (p_k^3 X)   =>  (u_k * p_k^3 - p_k^1) X = 0
        v_k = (p_k^2 X) / (p_k^3 X)   =>  (v_k * p_k^3 - p_k^2) X = 0
    where X = (X, Y, Z, 1) is the homogeneous 3D point. Stacking across K
    cameras gives a 2K x 4 linear system A X = 0. The least-squares solution
    (up to scale) is the right-singular-vector of A corresponding to the
    smallest singular value.

    cameras: list of K Camera objects.
    pixel_observations: (K, 2) tensor of (u, v) observations.
    Returns: (3,) tensor giving the triangulated 3D point in world coordinates.
    """
    K_cams = len(cameras)
    assert pixel_observations.shape == (K_cams, 2), \
        f"pixel_observations must be ({K_cams}, 2), got {pixel_observations.shape}"

    rows = []
    for k, cam in enumerate(cameras):
        P = projection_matrix(cam)                           # (3, 4)
        u = pixel_observations[k, 0]
        v = pixel_observations[k, 1]
        # (u * P[2] - P[0])^T X = 0
        # (v * P[2] - P[1])^T X = 0
        row_u = u * P[2] - P[0]
        row_v = v * P[2] - P[1]
        rows.append(row_u)
        rows.append(row_v)

    A = torch.stack(rows, dim=0)                             # (2K, 4)

    # Solve via SVD. The homogeneous solution is the right singular vector
    # corresponding to the smallest singular value.
    _, _, Vh = torch.linalg.svd(A, full_matrices=False)      # Vh is (4, 4)
    X_hom = Vh[-1]                                           # (4,) last row of V^T
    # Normalize: divide first 3 components by the 4th. The sign of the 4th
    # component is arbitrary (null vector defined up to scale), but we must
    # only guard against zero, NOT force positivity. Use signed clamping.
    w = X_hom[3]
    eps = torch.tensor(1e-12, dtype=w.dtype, device=w.device)
    # Preserve sign: if |w| < eps, set it to eps * sign(w). If w==0 exactly,
    # the triangulation is degenerate; return a sensible fallback.
    sign = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
    w_safe = torch.where(w.abs() < eps, sign * eps, w)
    X = X_hom[:3] / w_safe
    return X


def triangulate_points_batch(
    cameras: list[Camera],
    pixel_observations: Tensor,    # (N, K, 2)
) -> Tensor:
    """Batched triangulation of N points across K shared cameras.

    pixel_observations[i, k] = (u, v) for point i in camera k.
    Returns: (N, 3).
    """
    N = pixel_observations.shape[0]
    points = []
    for i in range(N):
        X = triangulate_point_dlt(cameras, pixel_observations[i])
        points.append(X)
    return torch.stack(points, dim=0)


# ---- Reprojection error (for validation) ------------------------------------

def reprojection_error(
    cameras: list[Camera],
    X_world: Tensor,               # (3,) or (N, 3)
    pixel_observations: Tensor,    # (K, 2) or (N, K, 2)
) -> Tensor:
    """Mean L2 reprojection error per camera per point.

    Returns scalar if X is (3,), or (N,) if X is (N, 3).
    """
    if X_world.dim() == 1:
        # Single point case
        errors = []
        for k, cam in enumerate(cameras):
            uv_pred = project_static(X_world.unsqueeze(0), cam).squeeze(0)
            uv_obs = pixel_observations[k]
            errors.append((uv_pred - uv_obs).norm())
        return torch.stack(errors).mean()
    else:
        N = X_world.shape[0]
        all_errs = torch.zeros(N, dtype=X_world.dtype, device=X_world.device)
        for i in range(N):
            all_errs[i] = reprojection_error(cameras, X_world[i], pixel_observations[i])
        return all_errs


# ---- Feature matching (simple, for synthetic data) ---------------------------
# For real data you'd use ORB/SIFT/LightGlue. For synthetic data where we know
# ground-truth correspondences, matching is trivial -- we already know which
# 2D observation in each camera corresponds to which 3D point.


def observe_scene_point(
    scene_point_traj_callable,
    t: float,
    cameras: list[Camera],
    add_noise_std: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Compute ground-truth pixel observations of a scene point in all cameras.

    Returns:
        pixel_obs: (K, 2) tensor.
        depth_per_cam: (K,) tensor, the camera-frame Z of the point (for
                       filtering out cameras that don't see it).
    """
    X_world = scene_point_traj_callable(t)                    # (3,)
    K = len(cameras)
    uvs = []
    depths = []
    for cam in cameras:
        X_cam = cam.R @ (X_world - cam.c)
        depths.append(X_cam[2])
        uv = project_static(X_world.unsqueeze(0), cam).squeeze(0)
        uvs.append(uv)
    uvs = torch.stack(uvs)                                    # (K, 2)
    depths = torch.stack(depths)                              # (K,)

    if add_noise_std > 0:
        uvs = uvs + add_noise_std * torch.randn_like(uvs)

    return uvs, depths
