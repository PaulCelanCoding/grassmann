"""
Synthetic multi-camera, multi-time scene generator.

Produces a controlled ground-truth setup we can use to:
  (a) develop and test the initialization pipeline without needing real video,
  (b) later validate the full training loop by rendering the ground-truth
      scene through the Grassmann model and checking we can recover it.

The scene:
  - K cameras placed on a ring around the origin, all looking inward.
  - A few "scene points" moving smoothly through 3D space over time.
  - Each point has a color.
  - Rendered images are produced by simple Gaussian blobs at the projected
    point locations (NOT by the Grassmann model -- we want independent
    ground truth to compare against).

All returned quantities are torch tensors, float64 for testing convenience.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from .projection import Camera, project_static


DTYPE = torch.float64


@dataclass
class ScenePoint:
    """A moving 3D point with a color.

    trajectory: callable t -> (3,) torch tensor giving world position at time t.
    color: (3,) torch tensor in [0, 1].
    """
    trajectory: Callable[[float], Tensor]
    color: Tensor


@dataclass
class MultiCameraSyntheticScene:
    """A synthetic multi-camera dataset."""
    cameras: list[Camera]
    scene_points: list[ScenePoint]
    H: int
    W: int
    background: Tensor = field(default_factory=lambda: torch.zeros(3, dtype=DTYPE))


# ---- Camera placement helpers ------------------------------------------------

def cameras_on_ring(
    K: int,
    radius: float = 5.0,
    height: float = 0.0,
    look_at: Tensor | None = None,
    fx: float = 400.0, fy: float = 400.0,
    image_w: int = 200, image_h: int = 120,
    dtype=DTYPE,
) -> list[Camera]:
    """Place K cameras evenly on a horizontal ring, all looking at `look_at`.

    The resulting cameras are at positions (radius*cos(theta), height, radius*sin(theta))
    for theta = 0, 2pi/K, ..., looking toward the center of the ring.
    """
    if look_at is None:
        look_at = torch.zeros(3, dtype=dtype)

    cameras = []
    for k in range(K):
        theta = 2.0 * np.pi * k / K
        cam_pos = torch.tensor([
            radius * np.cos(theta),
            height,
            radius * np.sin(theta),
        ], dtype=dtype)

        # Build rotation so that camera looks FROM cam_pos TOWARD look_at.
        # Camera convention: +z_cam points into the scene (from the camera toward the point).
        forward = look_at - cam_pos
        forward = forward / forward.norm()
        # World up
        up_world = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
        # Right = forward x up
        right = torch.cross(forward, up_world, dim=-1)
        right = right / right.norm()
        # True up (orthogonalized)
        up_cam = torch.cross(right, forward, dim=-1)
        # Camera convention: X=right, Y=down (image convention) or Y=up? We use Y=down
        # (common computer vision convention: X right, Y down, Z forward).
        down = -up_cam

        # Rotation world->camera: rows of R are (right, down, forward).
        R = torch.stack([right, down, forward], dim=0)

        cameras.append(Camera(
            R=R, c=cam_pos,
            fx=fx, fy=fy, cx=image_w / 2, cy=image_h / 2,
        ))
    return cameras


def cameras_stereo_pair(
    baseline: float = 1.0,
    distance_to_scene: float = 5.0,
    fx: float = 400.0, fy: float = 400.0,
    image_w: int = 200, image_h: int = 120,
    dtype=DTYPE,
) -> list[Camera]:
    """Two parallel cameras looking down +z, separated by `baseline` along x."""
    look_at = torch.tensor([0.0, 0.0, distance_to_scene], dtype=dtype)
    cameras = []
    for sign in [-1.0, +1.0]:
        cam_pos = torch.tensor([sign * baseline / 2, 0.0, 0.0], dtype=dtype)
        forward = look_at - cam_pos
        forward = forward / forward.norm()
        # Both cameras look along approximately +z, slightly toed in.
        right = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
        # Orthogonalize right against forward
        right = right - (right @ forward) * forward
        right = right / right.norm()
        down = torch.cross(forward, right, dim=-1)
        R = torch.stack([right, down, forward], dim=0)
        cameras.append(Camera(
            R=R, c=cam_pos,
            fx=fx, fy=fy, cx=image_w / 2, cy=image_h / 2,
        ))
    return cameras


# ---- Scene point trajectories ------------------------------------------------

def trajectory_linear(x0: list, velocity: list) -> Callable[[float], Tensor]:
    x0_t = torch.tensor(x0, dtype=DTYPE)
    v_t = torch.tensor(velocity, dtype=DTYPE)
    def traj(t: float) -> Tensor:
        return x0_t + t * v_t
    return traj


def trajectory_circular(center: list, radius: float, axis: str = "y",
                        period: float = 4.0) -> Callable[[float], Tensor]:
    """Moves in a circle around `center`, in the plane perpendicular to `axis`."""
    c_t = torch.tensor(center, dtype=DTYPE)
    def traj(t: float) -> Tensor:
        theta = 2.0 * np.pi * t / period
        if axis == "y":
            return c_t + torch.tensor([radius * np.cos(theta), 0.0, radius * np.sin(theta)], dtype=DTYPE)
        elif axis == "z":
            return c_t + torch.tensor([radius * np.cos(theta), radius * np.sin(theta), 0.0], dtype=DTYPE)
        else:
            return c_t + torch.tensor([0.0, radius * np.cos(theta), radius * np.sin(theta)], dtype=DTYPE)
    return traj


def trajectory_static(position: list) -> Callable[[float], Tensor]:
    """A stationary point."""
    p_t = torch.tensor(position, dtype=DTYPE)
    def traj(t: float) -> Tensor:
        return p_t
    return traj


# ---- Frame rendering (simple Gaussian blobs, independent of Grassmann model) ---

def render_synthetic_frame(
    scene: MultiCameraSyntheticScene,
    cam_idx: int,
    t: float,
    blob_sigma: float = 3.0,
    min_depth: float = 0.1,
) -> Tensor:
    """Render one camera's view at time t, as a sum of Gaussian blobs.

    Returns (H, W, 3) image in [0, 1].
    """
    cam = scene.cameras[cam_idx]
    img = scene.background.expand(scene.H, scene.W, 3).clone()

    # Pixel grid
    uu, vv = torch.meshgrid(
        torch.arange(scene.W, dtype=DTYPE),
        torch.arange(scene.H, dtype=DTYPE),
        indexing="xy",
    )
    grid = torch.stack([uu, vv], dim=-1)   # (H, W, 2)

    # Sort scene points by depth (far to near -- simple alpha compositing, no occlusion).
    points_with_depth = []
    for pt in scene.scene_points:
        X_world = pt.trajectory(t)
        X_cam = cam.R @ (X_world - cam.c)
        depth = X_cam[2].item()
        if depth > min_depth:
            uv = project_static(X_world.unsqueeze(0), cam).squeeze(0)
            points_with_depth.append((depth, uv, pt.color))

    # Paint far-to-near.
    points_with_depth.sort(reverse=True, key=lambda x: x[0])
    for depth, uv, color in points_with_depth:
        diff = grid - uv
        mah = (diff * diff).sum(dim=-1) / (blob_sigma * blob_sigma)
        alpha = torch.exp(-0.5 * mah).clamp(max=1.0).unsqueeze(-1)
        img = (1.0 - alpha) * img + alpha * color

    return img.clamp(0.0, 1.0)


def render_all_views(scene: MultiCameraSyntheticScene, t: float,
                     blob_sigma: float = 3.0) -> list[Tensor]:
    """Render all K cameras at a single time. Returns list of (H, W, 3) images."""
    return [render_synthetic_frame(scene, k, t, blob_sigma=blob_sigma)
            for k in range(len(scene.cameras))]


def render_video(scene: MultiCameraSyntheticScene, cam_idx: int,
                 times: list[float], blob_sigma: float = 3.0) -> Tensor:
    """Render a video from one camera. Returns (T, H, W, 3)."""
    frames = [render_synthetic_frame(scene, cam_idx, t, blob_sigma=blob_sigma) for t in times]
    return torch.stack(frames)


# ---- A canonical test scene --------------------------------------------------

def make_default_scene(n_cams: int = 4, image_w: int = 200, image_h: int = 120) -> MultiCameraSyntheticScene:
    """A K-camera ring observing a few moving colored points.

    Parameters are tuned so that all scene points are visible from all cameras
    throughout the default time range [0, 1].
    """
    cams = cameras_on_ring(
        K=n_cams, radius=6.0, height=0.0,
        look_at=torch.tensor([0.0, 0.0, 0.0], dtype=DTYPE),
        # Wider FoV: smaller focal length relative to image size means scene
        # points near the center stay in-frame even at distance 6.
        fx=120.0, fy=120.0, image_w=image_w, image_h=image_h,
    )
    scene_points = [
        # Red point moving left to right (within a small range).
        ScenePoint(
            trajectory=trajectory_linear(x0=[-0.5, 0.2, 0.0], velocity=[0.3, 0.0, 0.0]),
            color=torch.tensor([1.0, 0.3, 0.3], dtype=DTYPE),
        ),
        # Green point moving in a small circle
        ScenePoint(
            trajectory=trajectory_circular(center=[0.0, 0.0, 0.0], radius=0.5, axis="y", period=4.0),
            color=torch.tensor([0.3, 1.0, 0.3], dtype=DTYPE),
        ),
        # Blue point stationary, near the center.
        ScenePoint(
            trajectory=trajectory_static(position=[0.3, -0.2, 0.1]),
            color=torch.tensor([0.3, 0.5, 1.0], dtype=DTYPE),
        ),
    ]
    return MultiCameraSyntheticScene(
        cameras=cams, scene_points=scene_points,
        H=image_h, W=image_w,
        background=torch.tensor([0.05, 0.05, 0.1], dtype=DTYPE),
    )
