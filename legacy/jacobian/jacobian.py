"""
The 3x2 Jacobian J_full of the projection P: E_{p,q} -> R^2 x R,
evaluated at a Gaussian mean.

This implements Proposition 5 (Jacobian paper, eq. 6-10):

    J_full = [  J_persp @ J_embed   ]     in R^{2x2}   (spatial rows)
             [  J_time              ]     in R^{1x2}   (time row)

The local coordinates on E_{p,q} are (alpha, beta), referring to the orthonormal
basis (e1_hat, e2_hat) of the canonical plane. A point in the plane is

    z(alpha, beta) = v + alpha * e1_hat + beta * e2_hat.

For a STATIC CAMERA (Case A, Section 5.2), the full Jacobian uses the
perspective Jacobian evaluated at X_cam = R_0 (V_k - c_0), and an extra
rotation factor R_0 applied to J_embed (Proposition 6):

    J_static = [  J_pi @ R_0 @ J_embed  ]
               [  J_time                ]

We implement J_embed, J_time, and the assembly.
Dynamic camera (Case B) is deferred.
"""
from __future__ import annotations

import torch
from torch import Tensor

from . import quaternion as Q
from . import grassmann as G
from .projection import Camera, world_to_camera, perspective_jacobian


# ---- J_embed: spatial embedding (eq. 8) -------------------------------------

def jacobian_embed(p: Tensor, q: Tensor) -> Tensor:
    """J_embed: the 3x2 matrix whose columns are the SPATIAL parts of (e1_hat, e2_hat).

    From Definition 2 of the Jacobian paper:
      e1_hat = r * (0, d)               -> spatial part = r * d
      e2_hat = r * (1 + c, -s)          -> spatial part = -r * s
    where d = p + q, s = p x q, r = 1 / sqrt(2(1 + c)).

    So J_embed = r * [ d  | -s ] (stacked as 3x2 columns).

    p, q: shape (..., 4) unit imaginary quaternions.
    Returns: shape (..., 3, 2).

    This is equation (8) in the Jacobian paper.
    """
    f = G.canonical_frame(p, q)
    # r has shape (...,) -> (..., 1, 1) for broadcasting.
    r = f.r[..., None, None]
    d = f.d[..., None]          # (..., 3, 1)
    minus_s = -f.s[..., None]   # (..., 3, 1)
    # Concatenate along the columns (last dim).
    J = torch.cat([d, minus_s], dim=-1)  # (..., 3, 2)
    return r * J


# ---- J_time: time row (eq. 9) -----------------------------------------------

def jacobian_time(p: Tensor, q: Tensor) -> Tensor:
    """J_time: the 1x2 row containing the time component of the Jacobian.

    From eq. (9):
      t = v_0 + beta * r * (1 + c)
    so
      J_time = [ 0,  r*(1+c) ] = [ 0,  sqrt((1+c)/2) ].

    p, q: shape (..., 4).
    Returns: shape (..., 1, 2).
    """
    f = G.canonical_frame(p, q)
    # r * (1 + c) = (1 + c) / sqrt(2(1+c)) = sqrt((1+c)/2).
    time_scale = torch.sqrt((1.0 + f.c) * 0.5)     # shape (...,)
    zero = torch.zeros_like(time_scale)
    row = torch.stack([zero, time_scale], dim=-1)  # (..., 2)
    return row.unsqueeze(-2)                        # (..., 1, 2)


# ---- J_full for a STATIC camera (Proposition 6) -----------------------------

def jacobian_full_static(V: Tensor, p: Tensor, q: Tensor, cam: Camera) -> Tensor:
    """Assemble the full 3x2 Jacobian for a static camera.

    V: spatial mean of the Gaussian in WORLD coordinates, shape (..., 3).
    p, q: (..., 4), canonical plane identifiers.
    cam: the static camera (R_0, c_0, intrinsics).

    Returns: shape (..., 3, 2).

    This is eq. (13) of the Jacobian paper:
      J_full = [ J_pi @ R_0 @ J_embed ]
               [ J_time               ]
    where J_pi is the perspective Jacobian evaluated at X_cam = R_0 (V - c_0).
    """
    # Transform V to camera coordinates and get J_pi there.
    X_cam = world_to_camera(V, cam)                  # (..., 3)
    J_pi = perspective_jacobian(X_cam, cam)          # (..., 2, 3)

    # R_0 acts on J_embed's output (rotates the spatial directions into camera frame).
    J_e = jacobian_embed(p, q)                       # (..., 3, 2)
    # R_0 @ J_embed: broadcasting R_0 (3, 3) across batch.
    R0_J_e = cam.R @ J_e                             # works because cam.R is (3,3), J_e is (...,3,2)

    # Spatial block (2 x 2): J_pi @ R_0 @ J_embed
    J_spatial = J_pi @ R0_J_e                        # (..., 2, 2)

    # Time block (1 x 2): J_time
    J_t = jacobian_time(p, q)                        # (..., 1, 2)

    # Stack rows: 2 spatial rows on top, 1 time row on bottom -> 3x2
    J_full = torch.cat([J_spatial, J_t], dim=-2)     # (..., 3, 2)
    return J_full
