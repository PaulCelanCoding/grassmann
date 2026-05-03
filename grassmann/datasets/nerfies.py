"""
NeRFies / HyperNeRF format loader.

On-disk layout (per the google/nerfies repo):

    <scene>/
      camera/
        ${item_id}.json          per-frame camera (intrinsics + extrinsics + distortion)
      rgb/
        ${scale}x/
          ${item_id}.png         image at given downscale factor
      dataset.json               { count, num_exemplars, ids, train_ids, val_ids }
      metadata.json              { ${item_id}: { warp_id, appearance_id, camera_id } }
      scene.json                 (scene-level: scale, near, far, ...) -- optional here
      points.npy                 (N, 3) array of background points

Camera JSON fields used:
  orientation         (3x3) world-to-camera rotation, list-of-lists
  position            (3,)  camera center in world coords
  focal_length        scalar (or [fx, fy] in some variants -- we accept both)
  principal_point     [cx, cy]
  image_size          [W, H]
  radial_distortion   [k1, k2, k3]      -- REJECTED if any nonzero
  tangential          [p1, p2]          -- REJECTED if any nonzero

Distortion handling: we reject scenes with non-zero distortion rather than
silently mismatching geometry. NeRFies scenes captured on rectified phone video
typically have all zeros; older datasets may not. Pre-rectify externally if you
hit this.

Observability: NeRFies' shipped points.npy has no per-point track data. We
compute a per-point observability heuristic: a point is "observed" by frame t
if the camera at t has a positive line-of-sight dot product with the direction
from c(t) to the point AND the point projects inside the image bounds at that
frame's calibration. This is a coarse approximation -- a real COLMAP track
would be sharper -- but it suffices for picking a representative frame at init.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from ..projection import Camera
from ..time_normalization import normalize_times
from . import MonocularDataset


_DISTORTION_TOL = 1e-8


def _is_zero_distortion(values) -> bool:
    if values is None:
        return True
    arr = np.asarray(values, dtype=np.float64).flatten()
    return bool(np.all(np.abs(arr) < _DISTORTION_TOL))


def _load_camera_json(path: Path) -> tuple[Camera, int, int]:
    """Parse one NeRFies camera JSON. Returns (Camera, H, W)."""
    with open(path, "r") as f:
        data = json.load(f)

    if not _is_zero_distortion(data.get("radial_distortion")):
        raise ValueError(
            f"{path.name}: nonzero radial_distortion={data.get('radial_distortion')}. "
            f"Pre-rectify the scene externally; the pinhole Camera in this repo "
            f"does not model distortion."
        )
    if not _is_zero_distortion(data.get("tangential") or data.get("tangential_distortion")):
        raise ValueError(
            f"{path.name}: nonzero tangential distortion. Pre-rectify the scene "
            f"externally."
        )

    R = torch.tensor(data["orientation"], dtype=torch.float64)        # (3, 3) world->cam
    c = torch.tensor(data["position"], dtype=torch.float64)           # (3,)   camera center
    f = data["focal_length"]
    if isinstance(f, (list, tuple)):
        fx, fy = float(f[0]), float(f[1])
    else:
        fx = fy = float(f)
    pp = data["principal_point"]
    cx, cy = float(pp[0]), float(pp[1])
    W, H = int(data["image_size"][0]), int(data["image_size"][1])
    return Camera(R=R, c=c, fx=fx, fy=fy, cx=cx, cy=cy), H, W


def _build_observability(
    cameras: list[Camera],
    points: Tensor,
    H: int,
    W: int,
) -> list[list[int]]:
    """Heuristic per-point visibility: a point is observed by frame t iff
    (a) it's in front of camera t (positive depth) and (b) it projects inside
    the image bounds.
    """
    observability: list[list[int]] = []
    for i in range(points.shape[0]):
        X = points[i]
        seen: list[int] = []
        for t, cam in enumerate(cameras):
            X_cam = cam.R @ (X - cam.c)
            depth = float(X_cam[2].item())
            if depth <= 1e-6:
                continue
            u = cam.fx * float(X_cam[0]) / depth + cam.cx
            v = cam.fy * float(X_cam[1]) / depth + cam.cy
            if 0.0 <= u < W and 0.0 <= v < H:
                seen.append(t)
        observability.append(seen)
    return observability


def load_nerfies(
    scene_dir: str | Path,
    *,
    image_scale: int = 4,
) -> MonocularDataset:
    """Load a NeRFies-format monocular scene.

    scene_dir: path to the scene directory.
    image_scale: which `rgb/${image_scale}x/` subdirectory to read frames from.
                  NeRFies typically ships {1, 2, 4, 8}.

    Returns: MonocularDataset.
    """
    scene_dir = Path(scene_dir)
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"NeRFies scene_dir does not exist: {scene_dir}")

    dataset_json = scene_dir / "dataset.json"
    cam_dir = scene_dir / "camera"
    rgb_dir = scene_dir / "rgb" / f"{image_scale}x"
    points_npy = scene_dir / "points.npy"

    for required in (dataset_json, cam_dir, rgb_dir, points_npy):
        if not required.exists():
            raise FileNotFoundError(f"NeRFies scene missing {required}")

    with open(dataset_json, "r") as f:
        ds = json.load(f)
    ids: list[str] = list(ds["ids"])
    train_ids = set(ds.get("train_ids", []))
    val_ids = set(ds.get("val_ids", []))

    # Load every camera in the listed order. Frame index t corresponds to ids[t].
    cameras: list[Camera] = []
    raw_H, raw_W = None, None
    for item_id in ids:
        cam, H_cam, W_cam = _load_camera_json(cam_dir / f"{item_id}.json")
        if raw_H is None:
            raw_H, raw_W = H_cam, W_cam
        cameras.append(cam)

    # The image at scale ${s}x has dims (raw_H/s, raw_W/s); intrinsics scale by 1/s.
    if raw_H is None or raw_W is None:
        raise ValueError(f"No frames found in {dataset_json}")
    H = raw_H // image_scale
    W = raw_W // image_scale
    if image_scale != 1:
        scaled = []
        for cam in cameras:
            scaled.append(Camera(
                R=cam.R, c=cam.c,
                fx=cam.fx / image_scale, fy=cam.fy / image_scale,
                cx=cam.cx / image_scale, cy=cam.cy / image_scale,
            ))
        cameras = scaled

    # Times: monocular -> one frame per timestamp, uniform in [0, 1].
    T = len(ids)
    times = normalize_times(range(T))

    # Points and per-point visibility.
    points = torch.tensor(np.load(points_npy), dtype=torch.float64)
    observability = _build_observability(cameras, points, H, W)

    train_indices = [t for t, item_id in enumerate(ids) if item_id in train_ids] or list(range(T))
    val_indices = [t for t, item_id in enumerate(ids) if item_id in val_ids]

    rgb_paths = [rgb_dir / f"{item_id}.png" for item_id in ids]

    cache: dict[int, Tensor] = {}

    def frame_loader(frame_idx: int) -> Tensor:
        if frame_idx in cache:
            return cache[frame_idx]
        img = Image.open(rgb_paths[frame_idx]).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr)
        if len(cache) >= 64:
            cache.pop(next(iter(cache)))
        cache[frame_idx] = tensor
        return tensor

    return MonocularDataset(
        cameras_per_frame=cameras,
        times=times,
        points3D=points,
        observability=observability,
        frame_loader=frame_loader,
        H=H,
        W=W,
        train_indices=train_indices,
        val_indices=val_indices,
    )
