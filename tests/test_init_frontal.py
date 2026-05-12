"""Tests for the `frontal` init strategy: n = (0, d_hat) so the splat's
flat face after time conditioning is parallel to the init camera's image
plane.
"""
from __future__ import annotations

import pytest
import torch

from grassmann.gaussian import compute_derived, condition_on_time
from grassmann.initialization import (
    init_gaussian_from_point,
    init_gaussians_from_points,
)
from grassmann.projection import Camera


DTYPE = torch.float64


def _camera_at(x: float) -> Camera:
    """Identity-rotation camera at world position (x, 0, 0), looking +Z."""
    return Camera(
        R=torch.eye(3, dtype=DTYPE),
        c=torch.tensor([x, 0.0, 0.0], dtype=DTYPE),
        fx=100.0, fy=100.0, cx=8.0, cy=8.0,
    )


@pytest.fixture
def cams():
    return [_camera_at(i * 0.1) for i in range(5)]


def test_frontal_n_is_view_ray_with_zero_time_component(cams):
    """For `frontal`, n must satisfy n_t = 0 and n_xyz = (X - C) / ||X - C||."""
    X = torch.tensor([0.5, 0.2, 4.0], dtype=DTYPE)
    g = init_gaussian_from_point(
        X, t=0.5, cameras=cams,
        strategy="frontal", observability_idx=[1, 2, 3],   # median = cams[2]
        sigma_init_sq=0.02,
    )
    n = g.n[0]                                              # (4,)
    assert abs(float(n[0])) < 1e-12, f"n_t must be 0, got {float(n[0])}"
    expected = X - cams[2].c
    expected = expected / expected.norm()
    cos = float((n[1:] * expected).sum())
    assert abs(cos - 1.0) < 1e-9, f"n_xyz != view ray; cos={cos}"


def test_frontal_spatial_pure_cov_is_rank2_with_kernel_on_view_ray(cams):
    """After condition_on_time, Σ_spatial must be rank-2 with its near-zero
    eigenvector aligned with the view ray d_hat."""
    torch.manual_seed(0)
    X = torch.tensor([0.5, 0.2, 4.0], dtype=DTYPE)
    t_val = 0.5
    g = init_gaussian_from_point(
        X, t=t_val, cameras=cams,
        strategy="frontal", observability_idx=[2],
        sigma_init_sq=0.02,
    )
    derived = compute_derived(g)
    tc = condition_on_time(g, derived, t_0=t_val)
    Sigma_3D = tc.Sigma_3D_t[0]                             # (3, 3)
    ev, vec = torch.linalg.eigh(Sigma_3D)
    # Smallest eigenvalue (kernel direction) should be near zero, two
    # others non-negligible -- rank-2 disk.
    assert float(ev[0]) < 1e-9, f"expected near-zero smallest eigenvalue, got {float(ev[0])}"
    assert float(ev[1]) > 1e-4, f"second eigenvalue too small: {float(ev[1])}"
    # Kernel direction = view ray (up to sign).
    d = X - cams[2].c
    d = d / d.norm()
    cos = float((vec[:, 0] * d).sum())
    assert abs(abs(cos) - 1.0) < 1e-6, f"kernel not aligned with view ray; |cos|={abs(cos)}"


def test_frontal_batched_per_point_camera_selection(cams):
    """Batched init: each point's n must use its own median-observed camera."""
    pts = torch.tensor(
        [[0.0, 0.0, 5.0],
         [0.0, 0.0, 5.0]],
        dtype=DTYPE,
    )
    times = torch.tensor([0.2, 0.8], dtype=DTYPE)
    obs = [[0], [4]]                                        # point0 -> cams[0], point1 -> cams[4]
    g = init_gaussians_from_points(
        pts, times, cams,
        strategy="frontal", observability=obs,
        sigma_init_sq=0.02, seed=0,
    )
    for i, cam_idx in enumerate([0, 4]):
        expected = pts[i] - cams[cam_idx].c
        expected = expected / expected.norm()
        cos = float((g.n[i, 1:] * expected).sum())
        assert abs(cos - 1.0) < 1e-9, f"point {i}: n_xyz != view ray from cams[{cam_idx}]; cos={cos}"
        assert abs(float(g.n[i, 0])) < 1e-12


def test_frontal_requires_cameras():
    X = torch.tensor([0.0, 0.0, 5.0], dtype=DTYPE)
    with pytest.raises(ValueError, match="frontal"):
        init_gaussian_from_point(X, t=0.5, cameras=[], strategy="frontal")
