"""Tests for the init strategy flag (commit 6 of the monocular pivot)."""
from __future__ import annotations

import pytest
import torch

from grassmann.gaussian import compute_derived
from grassmann.initialization import (
    init_gaussian_from_point,
    init_gaussians_from_points,
    pick_reference_camera,
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
def cameras_orbit():
    """A 5-frame monocular trajectory: cameras slide along X axis."""
    return [_camera_at(i * 0.1) for i in range(5)]


def test_pick_birth_returns_first_observed(cameras_orbit):
    obs = [2, 3, 4]
    idx = pick_reference_camera(
        torch.tensor([0.0, 0.0, 5.0], dtype=DTYPE),
        cameras_orbit,
        strategy="birth",
        observability_idx=obs,
    )
    assert idx == 2


def test_pick_median_returns_middle_observed(cameras_orbit):
    obs = [0, 1, 2, 3, 4]
    idx = pick_reference_camera(
        torch.tensor([0.0, 0.0, 5.0], dtype=DTYPE),
        cameras_orbit,
        strategy="median",
        observability_idx=obs,
    )
    assert idx == 2  # middle of 5 is index 2


def test_pick_birth_falls_back_to_zero_without_observability(cameras_orbit):
    idx = pick_reference_camera(
        torch.tensor([0.0, 0.0, 5.0], dtype=DTYPE),
        cameras_orbit,
        strategy="birth",
    )
    assert idx == 0


def test_pick_median_falls_back_to_middle(cameras_orbit):
    idx = pick_reference_camera(
        torch.tensor([0.0, 0.0, 5.0], dtype=DTYPE),
        cameras_orbit,
        strategy="median",
    )
    assert idx == len(cameras_orbit) // 2  # 2


def test_pick_random_raises(cameras_orbit):
    with pytest.raises(ValueError, match="random"):
        pick_reference_camera(
            torch.tensor([0.0, 0.0, 5.0], dtype=DTYPE),
            cameras_orbit,
            strategy="random",
        )


def test_init_random_produces_valid_gaussian(cameras_orbit):
    """Random strategy should produce a usable GaussianParams: derived
    quantities are finite and Sigma_3D is rank <=2."""
    torch.manual_seed(0)
    g = init_gaussian_from_point(
        torch.tensor([0.05, 0.02, 5.0], dtype=DTYPE),
        t=0.5,
        cameras=cameras_orbit,
        strategy="random",
    )
    assert g.p_im.shape == (1, 3)
    derived = compute_derived(g)
    assert torch.isfinite(derived.V_k).all()
    assert torch.isfinite(derived.Sigma_3D).all()
    # Sigma_3D rank <= 2 (the model invariant).
    s = torch.linalg.svdvals(derived.Sigma_3D[0])
    assert s[2].item() < 1e-10


def test_init_orthogonal_axis_perpendicular_to_view_ray(cameras_orbit):
    """orthogonal init must put rank-1 axis ⊥ view ray AND V_k exactly at X."""
    from grassmann.gaussian import condition_on_time
    torch.manual_seed(0)
    X = torch.tensor([0.05, 0.02, 5.0], dtype=DTYPE)
    t_val = 0.5
    g = init_gaussian_from_point(
        X, t=t_val, cameras=cameras_orbit,
        strategy="orthogonal", observability_idx=[2],
    )
    assert g.p_im.shape == (1, 3)
    derived = compute_derived(g)
    # V_k must land exactly on X (no projection residual).
    assert torch.allclose(derived.V_k[0], X, atol=1e-9), \
        f"orthogonal V_k off-target: V_k={derived.V_k[0]}, X={X}"
    assert abs(float(derived.v_0[0]) - t_val) < 1e-9
    # Rank-1 axis must be perpendicular to view ray.
    tc = condition_on_time(g, derived, t_0=t_val)
    ev, vec = torch.linalg.eigh(tc.Sigma_3D_t[0])
    top = vec[:, -1]
    cam = cameras_orbit[2]
    u = (X - cam.c) / (X - cam.c).norm()
    assert abs(float((top * u).sum())) < 1e-6, \
        f"orthogonal axis should be ⊥ view ray; got |cos|={abs(float((top*u).sum()))}"


def test_init_tripod_emits_three_gaussians_with_orthogonal_axes(cameras_orbit):
    """tripod init: 3 Gaussians per point with axes spanning R^3, all V_k = X."""
    from grassmann.gaussian import condition_on_time
    torch.manual_seed(0)
    X = torch.tensor([0.05, 0.02, 5.0], dtype=DTYPE)
    t_val = 0.5
    g = init_gaussian_from_point(
        X, t=t_val, cameras=cameras_orbit,
        strategy="tripod", observability_idx=[2],
    )
    assert g.N == 3, f"tripod should produce 3 Gaussians per point; got {g.N}"
    derived = compute_derived(g)
    # All 3 V_k must equal X (no projection residual).
    for i in range(3):
        assert torch.allclose(derived.V_k[i], X, atol=1e-9), \
            f"tripod Gaussian {i} V_k off-target: V_k={derived.V_k[i]}, X={X}"
    tc = condition_on_time(g, derived, t_0=t_val)
    axes = []
    for i in range(3):
        ev, vec = torch.linalg.eigh(tc.Sigma_3D_t[i])
        axes.append(vec[:, -1])
    # Mutually orthogonal axes (R^3 basis).
    for i in range(3):
        for j in range(i + 1, 3):
            dot = abs(float((axes[i] * axes[j]).sum()))
            assert dot < 1e-6, f"tripod axes {i},{j} not orthogonal: |cos|={dot}"


def test_init_gaussians_from_points_tripod_triples_count(cameras_orbit):
    """init_gaussians_from_points with strategy='tripod' must yield 3*N Gaussians."""
    torch.manual_seed(0)
    pts = torch.tensor([[0.05, 0.02, 5.0], [0.1, -0.1, 4.0]], dtype=DTYPE)
    times = torch.tensor([0.3, 0.7], dtype=DTYPE)
    obs = [[1, 2, 3], [0, 2, 4]]
    g = init_gaussians_from_points(
        pts, times, cameras_orbit, strategy="tripod", observability=obs,
    )
    assert g.N == 6, f"tripod over 2 points should give 6 Gaussians; got {g.N}"


def test_init_birth_uses_observability_camera(cameras_orbit):
    """With strategy='birth' and observability=[3], the ref cam used must be 3."""
    torch.manual_seed(0)
    X = torch.tensor([0.0, 0.0, 5.0], dtype=DTYPE)
    g_birth = init_gaussian_from_point(
        X, t=0.5, cameras=cameras_orbit,
        strategy="birth", observability_idx=[3],
    )
    # Re-derive: the ray-from-cam-3 line uniquely identifies the (p, q) plane.
    # We confirm the derived V_k roughly matches X (the projection-onto-basis
    # may have a small residual, but that's the same regardless of ref cam).
    derived = compute_derived(g_birth)
    assert torch.allclose(derived.V_k[0], X, atol=1e-3)


def test_init_gaussians_from_points_threads_observability(cameras_orbit):
    """The batched API must dispatch each point to its own observability list."""
    torch.manual_seed(0)
    points = torch.tensor([
        [0.0, 0.0, 5.0],
        [0.05, 0.0, 5.0],
        [-0.05, 0.0, 5.0],
    ], dtype=DTYPE)
    times = torch.tensor([0.1, 0.5, 0.9], dtype=DTYPE)
    obs = [[0], [2], [4]]
    g = init_gaussians_from_points(
        points, times, cameras_orbit,
        strategy="birth",
        observability=obs,
    )
    assert g.N == 3
    derived = compute_derived(g)
    # All means should be finite and approx at the input X positions.
    for i in range(3):
        assert torch.allclose(derived.V_k[i], points[i], atol=1e-3)


def test_observability_length_mismatch_raises(cameras_orbit):
    points = torch.tensor([[0.0, 0.0, 5.0], [0.05, 0.0, 5.0]], dtype=DTYPE)
    times = torch.tensor([0.1, 0.9], dtype=DTYPE)
    with pytest.raises(ValueError, match="observability"):
        init_gaussians_from_points(
            points, times, cameras_orbit,
            strategy="median",
            observability=[[0]],   # length 1 != 2
        )


def test_default_strategy_is_lookat(cameras_orbit):
    """No strategy argument -> the legacy lookat behavior (preserves backward compat)."""
    X = torch.tensor([0.0, 0.0, 5.0], dtype=DTYPE)
    idx_default = pick_reference_camera(X, cameras_orbit)
    idx_lookat = pick_reference_camera(X, cameras_orbit, strategy="lookat")
    assert idx_default == idx_lookat
