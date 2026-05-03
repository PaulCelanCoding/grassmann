"""Tests for grassmann.datasets.nerfies.

Builds an in-tmpdir mini scene that mirrors the NeRFies on-disk format and
verifies the loader produces a usable MonocularDataset, including:
  - times normalized to [0, 1]
  - cameras intrinsics scaled correctly with image_scale
  - observability list spans the points
  - distortion rejection raises a clear error
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from grassmann.datasets.nerfies import load_nerfies


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


def _build_mini_scene(
    root: Path,
    *,
    n_frames: int = 4,
    image_size=(16, 16),
    image_scale: int = 1,
    distortion_radial=(0.0, 0.0, 0.0),
    n_points: int = 5,
):
    """Create a NeRFies-format scene under `root` with synthetic data.

    Cameras: aligned with +Z, varying X position to simulate orbit.
    Frames: distinct solid-color RGB images.
    Points: a cluster in front of the cameras.
    """
    (root / "camera").mkdir(parents=True, exist_ok=True)
    (root / "rgb" / f"{image_scale}x").mkdir(parents=True, exist_ok=True)

    ids = [f"f{i:04d}" for i in range(n_frames)]

    # Cameras: identity rotation (looking +Z), positions slide along X.
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
        # RGB: solid color per frame (so a loaded frame is identifiable).
        arr = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        arr[:, :, 0] = (i * 60) % 256
        arr[:, :, 1] = (i * 90) % 256
        Image.fromarray(arr).save(root / "rgb" / f"{image_scale}x" / f"{item_id}.png")

    # Points cluster tightly in front (+Z) of the cameras so all project in-frame
    # under the synthetic intrinsics (f=100, 16x16 image -> visible X,Y in ~+/- 0.4).
    rng = np.random.default_rng(0)
    points = rng.normal(loc=[0.0, 0.0, 5.0], scale=[0.1, 0.1, 0.1], size=(n_points, 3)).astype(
        np.float64
    )
    np.save(root / "points.npy", points)

    # dataset.json: first 80% train, rest val.
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

    # metadata.json: minimal (loader doesn't currently consume it but the file
    # should exist in real scenes).
    with open(root / "metadata.json", "w") as f_out:
        json.dump({item_id: {"warp_id": i, "appearance_id": i, "camera_id": 0}
                    for i, item_id in enumerate(ids)}, f_out)

    return ids


@pytest.fixture
def mini_scene(tmp_path: Path) -> Path:
    _build_mini_scene(tmp_path)
    return tmp_path


def test_load_returns_dataset(mini_scene):
    ds = load_nerfies(mini_scene, image_scale=1)
    assert ds.T == 4
    assert len(ds.cameras_per_frame) == 4
    assert ds.points3D.shape == (5, 3)
    assert ds.H == 16 and ds.W == 16
    assert len(ds.observability) == 5


def test_times_in_unit_interval(mini_scene):
    ds = load_nerfies(mini_scene, image_scale=1)
    assert ds.times.min().item() == 0.0
    assert ds.times.max().item() == pytest.approx(1.0)


def test_train_val_split(mini_scene):
    ds = load_nerfies(mini_scene, image_scale=1)
    assert len(ds.train_indices) == 3       # 80% of 4
    assert len(ds.val_indices) == 1
    assert set(ds.train_indices) & set(ds.val_indices) == set()


def test_frame_loader_returns_image(mini_scene):
    ds = load_nerfies(mini_scene, image_scale=1)
    img = ds.frame_loader(0)
    assert img.shape == (16, 16, 3)
    assert img.dtype == torch.float32
    assert 0.0 <= img.min().item() <= img.max().item() <= 1.0


def test_observability_spans_points(mini_scene):
    """Points placed in front of cameras should be observed by all frames."""
    ds = load_nerfies(mini_scene, image_scale=1)
    # All synthetic points are at z~5 in front of every (z=0, looking +Z) camera,
    # so observability should be non-empty for each.
    for i, frames in enumerate(ds.observability):
        assert len(frames) > 0, f"Point {i} observed by 0 frames"


def test_image_scale_divides_intrinsics(tmp_path):
    """At image_scale=2, fx/fy/cx/cy and H/W should all halve."""
    _build_mini_scene(tmp_path, image_size=(16, 16), image_scale=2)
    ds = load_nerfies(tmp_path, image_scale=2)
    assert ds.H == 8 and ds.W == 8
    assert ds.cameras_per_frame[0].fx == pytest.approx(50.0)
    assert ds.cameras_per_frame[0].cx == pytest.approx(4.0)


def test_radial_distortion_rejected(tmp_path):
    """Loader must reject scenes with non-zero distortion."""
    _build_mini_scene(tmp_path, distortion_radial=(0.01, 0.0, 0.0))
    with pytest.raises(ValueError, match="radial_distortion"):
        load_nerfies(tmp_path, image_scale=1)


def test_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_nerfies(tmp_path / "does-not-exist", image_scale=1)


def test_missing_required_files_raises(tmp_path):
    """Even a partially populated dir should fail clearly."""
    (tmp_path / "camera").mkdir()
    with pytest.raises(FileNotFoundError, match="dataset.json"):
        load_nerfies(tmp_path, image_scale=1)


def test_trainer_from_monocular_dataset_runs(mini_scene):
    """End-to-end smoke: monocular Trainer with a NeRFies mini-scene runs N steps
    without error and the loss is finite."""
    from grassmann.datasets.nerfies import load_nerfies
    from grassmann.initialization import init_gaussians_from_points
    from grassmann.trainable import trainable_from_params
    from grassmann.training import Trainer, TrainerConfig

    ds = load_nerfies(mini_scene, image_scale=1)
    # Initialize 3 Gaussians from the scene points (using the first frame's camera).
    params = init_gaussians_from_points(
        ds.points3D[:3].to(torch.float32),
        torch.tensor([float(ds.times[0]), float(ds.times[1]), float(ds.times[2])]),
        ds.cameras_per_frame,
        sigma_aa=0.02, sigma_bb=0.05, opacity=0.5, sigma_k_pixel=1.0, sigma_k_temporal=0.0,
    )
    model = trainable_from_params(params, dtype=torch.float32)

    cfg = TrainerConfig(
        num_iters=3, log_every=3,
        background=torch.zeros(3, dtype=torch.float32),
    )
    trainer = Trainer.from_monocular_dataset(model, ds, cfg)
    assert trainer.config.monocular is True
    assert trainer.K == ds.T  # one camera per frame

    history = trainer.train()
    assert len(history["loss"]) >= 1
    assert all(torch.isfinite(torch.tensor(v)) for v in history["loss"])
