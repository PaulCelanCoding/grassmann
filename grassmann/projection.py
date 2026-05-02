"""
Perspective projection and static camera model.

We implement the two building blocks of the projection P: E_{p,q} -> R^2 x R:

  1. A static camera (R_0, c_0): world-space point X -> camera-space Rc(X - c).
  2. Perspective projection: (X_cam, Y_cam, Z_cam) -> (fx X/Z + cx, fy Y/Z + cy).

The time component passes through unchanged.

In Case A of the Jacobian paper (static camera), the full projection of a point
z = (z_0, z_spatial) in E_{p,q} is:

    P(z) = ( pi(R_0 (z_spatial - c_0)),  z_0 )  in R^2 x R,

where pi is perspective projection. See Jacobian paper Section 3.2, eq. (5),
and Section 5.2, eq. (12).

All operations are batched and autograd-compatible.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


# ---- Camera --------------------------------------------------------------

@dataclass
class Camera:
    """Pinhole camera with static extrinsics.

    Intrinsics (pixel units):
      fx, fy: focal lengths
      cx, cy: principal point

    Extrinsics:
      R: rotation matrix, shape (3, 3) -- world -> camera
      c: camera center in world, shape (3,)
    so that X_cam = R @ (X_world - c).
    """
    R: Tensor       # (3, 3)
    c: Tensor       # (3,)
    fx: float
    fy: float
    cx: float
    cy: float

    @staticmethod
    def at_origin(fx: float = 1.0, fy: float = 1.0,
                  cx: float = 0.0, cy: float = 0.0,
                  dtype=torch.float64, device="cpu") -> "Camera":
        """Canonical camera: at origin, looking down +z axis."""
        R = torch.eye(3, dtype=dtype, device=device)
        c = torch.zeros(3, dtype=dtype, device=device)
        return Camera(R=R, c=c, fx=fx, fy=fy, cx=cx, cy=cy)


def world_to_camera(X_world: Tensor, cam: Camera) -> Tensor:
    """Apply extrinsics: X_cam = R (X_world - c).

    X_world: shape (..., 3).
    Returns shape (..., 3).
    """
    # Broadcasting: (..., 3) - (3,) -> (..., 3), then @ R^T along last dim.
    X_centered = X_world - cam.c
    return X_centered @ cam.R.T


def perspective(X_cam: Tensor, cam: Camera) -> Tensor:
    """Perspective projection of camera-space points to pixel (u, v).

    X_cam: shape (..., 3) with components (X, Y, Z).
    Returns shape (..., 2) with (u, v).

    Division by Z is the source of the nonlinearity.
    """
    X, Y, Z = X_cam[..., 0], X_cam[..., 1], X_cam[..., 2]
    u = cam.fx * X / Z + cam.cx
    v = cam.fy * Y / Z + cam.cy
    return torch.stack([u, v], dim=-1)


def project_static(X_world: Tensor, cam: Camera) -> Tensor:
    """Full spatial projection: world -> camera -> pixel.

    X_world: shape (..., 3).
    Returns shape (..., 2).
    """
    return perspective(world_to_camera(X_world, cam), cam)


# ---- Analytic Jacobian of perspective ---------------------------------------

def perspective_jacobian(X_cam: Tensor, cam: Camera) -> Tensor:
    """Analytic Jacobian d pi / d X_cam of the perspective projection.

    For pi(X, Y, Z) = (fx X/Z + cx, fy Y/Z + cy):

        J_pi = (1/Z) * [[ fx,  0, -fx X/Z ],
                        [  0, fy, -fy Y/Z ]]

    X_cam: shape (..., 3).
    Returns shape (..., 2, 3).

    This is equation (7) in the Jacobian paper.
    """
    X, Y, Z = X_cam[..., 0], X_cam[..., 1], X_cam[..., 2]
    inv_Z = 1.0 / Z

    # Build the 2x3 matrix for each batch entry.
    zero = torch.zeros_like(X)
    fx = torch.full_like(X, cam.fx)
    fy = torch.full_like(X, cam.fy)

    row0 = torch.stack([fx,              zero, -fx * X * inv_Z], dim=-1)  # (..., 3)
    row1 = torch.stack([zero,            fy,   -fy * Y * inv_Z], dim=-1)  # (..., 3)

    J = torch.stack([row0, row1], dim=-2)                                  # (..., 2, 3)
    return J * inv_Z.unsqueeze(-1).unsqueeze(-1)
