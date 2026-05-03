"""Shared scene-builder helpers for tests. Imported by tests/conftest.py and
test files; not itself a pytest plugin."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _write_camera(
    path: Path,
    *,
    R,
    c,
    f,
    pp,
    image_size,
    radial=(0.0, 0.0, 0.0),
    tangential=(0.0, 0.0),
):
    data = {
        "orientation": [list(map(float, row)) for row in R],
        "position": [float(x) for x in c],
        "focal_length": float(f),
        "principal_point": [float(pp[0]), float(pp[1])],
        "skew": 0.0,
        "pixel_aspect_ratio": 1.0,
        "radial_distortion": list(radial),
        "tangential": list(tangential),
        "image_size": [int(image_size[0]), int(image_size[1])],
    }
    with open(path, "w") as f_out:
        json.dump(data, f_out)


def build_mini_nerfies_scene(
    root: Path,
    *,
    n_frames: int = 4,
    image_size=(16, 16),
    image_scale: int = 1,
    distortion_radial=(0.0, 0.0, 0.0),
    n_points: int = 5,
):
    """Create a NeRFies-format scene under `root` with synthetic data.

    Cameras: identity rotation looking +Z, positions slide along X.
    Frames: distinct solid-color RGB images.
    Points: a tight cluster in front of the cameras (so all project in-frame).
    """
    (root / "camera").mkdir(parents=True, exist_ok=True)
    (root / "rgb" / f"{image_scale}x").mkdir(parents=True, exist_ok=True)

    ids = [f"f{i:04d}" for i in range(n_frames)]

    R = np.eye(3, dtype=np.float64)
    f = 100.0
    pp = (image_size[0] / 2, image_size[1] / 2)
    for i, item_id in enumerate(ids):
        c = (float(i) * 0.1, 0.0, 0.0)
        _write_camera(
            root / "camera" / f"{item_id}.json",
            R=R, c=c, f=f, pp=pp, image_size=image_size,
            radial=distortion_radial,
        )
        arr = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        arr[:, :, 0] = (i * 60) % 256
        arr[:, :, 1] = (i * 90) % 256
        Image.fromarray(arr).save(root / "rgb" / f"{image_scale}x" / f"{item_id}.png")

    rng = np.random.default_rng(0)
    points = rng.normal(loc=[0.0, 0.0, 5.0], scale=[0.1, 0.1, 0.1], size=(n_points, 3)).astype(
        np.float64
    )
    np.save(root / "points.npy", points)

    n_train = max(1, int(0.8 * n_frames))
    train_ids = ids[:n_train]
    val_ids = ids[n_train:]
    with open(root / "dataset.json", "w") as f_out:
        json.dump({
            "count": n_frames,
            "num_exemplars": n_train,
            "ids": ids,
            "train_ids": train_ids,
            "val_ids": val_ids,
        }, f_out)

    with open(root / "metadata.json", "w") as f_out:
        json.dump({item_id: {"warp_id": i, "appearance_id": i, "camera_id": 0}
                    for i, item_id in enumerate(ids)}, f_out)

    return ids
