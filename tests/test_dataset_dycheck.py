"""Tests for grassmann.datasets.dycheck.

DyCheck on-disk format is NeRFies-compatible plus an optional `splits/` dir.
We reuse the NeRFies mini-fixture builder and add a splits override.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from grassmann.datasets.dycheck import load_dycheck
from _fixture_builders import build_mini_nerfies_scene


def _add_dycheck_split(root: Path, split_name: str, frame_names: list[str]) -> None:
    (root / "splits").mkdir(exist_ok=True)
    payload = {
        "frame_names": frame_names,
        "time_ids": list(range(len(frame_names))),
        "camera_ids": [0] * len(frame_names),
    }
    with open(root / "splits" / f"{split_name}.json", "w") as f:
        json.dump(payload, f)


def test_load_dycheck_without_split_falls_back_to_dataset_json(tmp_path):
    build_mini_nerfies_scene(tmp_path)
    ds = load_dycheck(tmp_path, image_scale=1)
    # Same as NeRFies loader: 80% train.
    assert len(ds.train_indices) == 3
    assert len(ds.val_indices) == 1


def test_load_dycheck_with_split_override(tmp_path):
    build_mini_nerfies_scene(tmp_path)
    # Override: pick frames f0001 and f0003 as the named split.
    _add_dycheck_split(tmp_path, "common", ["f0001", "f0003"])
    ds = load_dycheck(tmp_path, image_scale=1, split_name="common")
    assert ds.train_indices == [1, 3]
    assert ds.val_indices == [0, 2]


def test_unknown_split_raises(tmp_path):
    build_mini_nerfies_scene(tmp_path)
    with pytest.raises(FileNotFoundError, match="split 'novel' not found"):
        load_dycheck(tmp_path, image_scale=1, split_name="novel")


def test_dycheck_inherits_distortion_rejection(tmp_path):
    """Distortion check is inherited from the NeRFies loader."""
    build_mini_nerfies_scene(tmp_path, distortion_radial=(0.005, 0.0, 0.0))
    with pytest.raises(ValueError, match="radial_distortion"):
        load_dycheck(tmp_path, image_scale=1)
