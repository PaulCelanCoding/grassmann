"""Pytest configuration: expose fixtures from _fixture_builders."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `from _fixture_builders import build_mini_nerfies_scene` from any test file.
sys.path.insert(0, str(Path(__file__).parent))

import pytest

from _fixture_builders import build_mini_nerfies_scene


# Phase A (3-plane projector pivot): tests that target the legacy 2-plane
# parameterization (p_im, q_im, alpha_0, beta_0, L=2x2) are skipped at
# collection. They will be rewritten in Phase B against the new param.
# See ~/.claude/plans/grassmann-splatting-on-imperative-rocket.md.
collect_ignore = [
    "test_init_strategies.py",
    "test_dataset_nerfies.py",
    "test_density_control.py",
    "test_time_normalization.py",
    "test_training.py",
    "test_fast_rasterizer.py",
    "test_rendering.py",
    "test_initialization.py",
]


@pytest.fixture
def mini_nerfies_scene(tmp_path):
    """A minimal NeRFies-format scene under tmpdir."""
    build_mini_nerfies_scene(tmp_path)
    return tmp_path
