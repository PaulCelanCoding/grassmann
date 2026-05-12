"""Pytest configuration: expose fixtures from _fixture_builders."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `from _fixture_builders import build_mini_nerfies_scene` from any test file.
sys.path.insert(0, str(Path(__file__).parent))

import pytest

from _fixture_builders import build_mini_nerfies_scene


@pytest.fixture
def mini_nerfies_scene(tmp_path):
    """A minimal NeRFies-format scene under tmpdir."""
    build_mini_nerfies_scene(tmp_path)
    return tmp_path
