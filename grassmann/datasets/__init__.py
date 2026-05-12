"""
Monocular video datasets for the Grassmann pipeline.

A `MonocularDataset` is the common contract every loader returns. Trainers,
init code, and visualizers consume this dataclass; they never touch a
dataset-specific format.

Conventions:
- Times are normalized to [0, 1] (see grassmann.time_normalization).
- One Camera per frame (monocular: a single moving camera over time).
- Observability is per-point list of frame indices that "see" the point. If
  the source has no per-point track data, loaders are responsible for
  computing or estimating it (e.g., line-of-sight heuristic).
- Frames are loaded lazily via the `frame_loader(frame_idx)` callable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import torch
from torch import Tensor

from ..projection import Camera


@dataclass
class MonocularDataset:
    """One monocular video scene: a moving camera + a point cloud + frames.

    cameras_per_frame: list of length T. cameras_per_frame[t] is the Camera at
                      frame t.
    times:            tensor of length T, in [0, 1]. times[t] is the
                      normalized timestamp of frame t.
    points3D:         (N, 3) float64 tensor of world-space points used for init.
    observability:    list of length N. observability[i] is the list of frame
                      indices in [0, T) that observe point i.
    frame_loader:     callable (frame_idx: int) -> (H, W, 3) float32 tensor in
                      [0, 1].
    H, W:             image dimensions (after any internal downscale).
    train_indices:    list of frame indices used for training.
    val_indices:      list of frame indices held out for validation. May be
                      empty if the dataset doesn't ship a split.
    """
    cameras_per_frame: List[Camera]
    times: Tensor
    points3D: Tensor
    observability: List[List[int]]
    frame_loader: Callable[[int], Tensor]
    H: int
    W: int
    train_indices: List[int]
    val_indices: List[int]

    @property
    def T(self) -> int:
        return len(self.cameras_per_frame)

    @property
    def N_points(self) -> int:
        return int(self.points3D.shape[0])


def load_monocular(
    name: str,
    scene_dir: Path,
    *,
    image_scale: int = 1,
    split: str | None = None,
    allow_distortion: bool = False,
) -> MonocularDataset:
    """Dispatch to nerfies or dycheck loader by name.

    `split` is only honored for dycheck (nerfies has no splits/ directory).
    """
    if name == "nerfies":
        from .nerfies import load_nerfies
        if split is not None:
            print("  [warning] --split is ignored for --dataset nerfies (no splits/ dir)")
        return load_nerfies(scene_dir, image_scale=image_scale,
                            allow_distortion=allow_distortion)
    if name == "dycheck":
        from .dycheck import load_dycheck
        return load_dycheck(scene_dir, image_scale=image_scale, split_name=split,
                            allow_distortion=allow_distortion)
    raise ValueError(f"Unknown --dataset {name!r}; expected nerfies|dycheck")
