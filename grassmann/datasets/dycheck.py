"""
DyCheck iPhone format loader.

DyCheck scenes ship in a NeRFies-compatible directory layout (camera/, rgb/,
dataset.json, metadata.json, points.npy, scene.json) with the following extras
that this loader currently ignores:

    depth/${s}x/${id}.npy        per-frame metric Lidar depth (training only)
    covisible/${s}x/${split}/    binary covisibility masks per split
    keypoint/${s}x/${split}/     2D keypoint annotations for eval
    splits/${split}.json         alternative train/val splits (DyCheck protocol)
    emf.json, extra.json         scene metadata

For the initial pivot we treat DyCheck as "NeRFies + optional splits override".
The depth/covisible/keypoint extras are loadable later as separate features
(e.g., depth-supervised loss); they are not part of the core MonocularDataset
contract.

DyCheck cameras typically have non-zero distortion -- the dataset is captured
on iPhone hardware. The NeRFies loader's distortion rejection will fire on
those scenes; rectify them externally with the DyCheck preprocessing tools
before loading.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch

from . import MonocularDataset
from .nerfies import load_nerfies


def load_dycheck(
    scene_dir: str | Path,
    *,
    image_scale: int = 4,
    split_name: Optional[str] = None,
    allow_distortion: bool = False,
) -> MonocularDataset:
    """Load a DyCheck iPhone scene.

    scene_dir:        path to the scene directory.
    image_scale:      which `rgb/${image_scale}x/` subdirectory to read frames from.
    split_name:       if given, override dataset.json's train/val split with the
                      contents of `splits/{split_name}.json`. Common DyCheck splits
                      are "train", "val", "common", "novel". If None, fall back to
                      the dataset.json split (NeRFies behavior).
    allow_distortion: forwarded to load_nerfies. DyCheck cameras are iPhone-
                      captured and typically have non-zero distortion; pass
                      True for code-path smokes only.

    Returns: MonocularDataset.
    """
    scene_dir = Path(scene_dir)
    ds = load_nerfies(scene_dir, image_scale=image_scale, allow_distortion=allow_distortion)
    if split_name is None:
        return ds

    split_file = scene_dir / "splits" / f"{split_name}.json"
    if not split_file.exists():
        raise FileNotFoundError(
            f"DyCheck split '{split_name}' not found at {split_file}. "
            f"Available splits should live under {scene_dir / 'splits'}/."
        )
    with open(split_file, "r") as f:
        split = json.load(f)
    # DyCheck split files: { "frame_names": [...], "time_ids": [...], "camera_ids": [...] }.
    # Map back to frame indices via dataset.json's `ids` order.
    dataset_json = scene_dir / "dataset.json"
    with open(dataset_json, "r") as f:
        ids = list(json.load(f)["ids"])
    name_to_idx = {name: t for t, name in enumerate(ids)}
    indices = [name_to_idx[name] for name in split["frame_names"] if name in name_to_idx]

    # Replace train/val on the loaded dataset with the override. The convention
    # we adopt: the named split = train_indices (e.g. split_name="train"); val
    # is everything not in this set.
    selected = set(indices)
    val_indices = [t for t in range(ds.T) if t not in selected]
    return MonocularDataset(
        cameras_per_frame=ds.cameras_per_frame,
        times=ds.times,
        points3D=ds.points3D,
        observability=ds.observability,
        frame_loader=ds.frame_loader,
        H=ds.H,
        W=ds.W,
        train_indices=indices,
        val_indices=val_indices,
    )
