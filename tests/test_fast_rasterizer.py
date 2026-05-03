"""Tests for Phase 7: fast rasterizer adapter.

On CPU-only systems (like this one), the CUDA kernel isn't available. We test:
  1. is_available() correctly reports False when there's no CUDA or no extension.
  2. sigma3d_to_cov6() packs a symmetric 3x3 into the 6-element form correctly.
  3. camera_to_view_matrix builds a 4x4 that transforms world -> camera correctly.
  4. fast_rasterize() falls back to the toy rasterizer and produces the same output
     as calling the toy path directly.
  5. Trainer with use_fast_rasterizer=True still works (transparent fallback).
"""
import pytest
import torch

from grassmann.projection import Camera, world_to_camera
from grassmann.gaussian import compute_derived, condition_on_time
from grassmann.rasterizer import project_to_screen, rasterize as toy_rasterize
from grassmann.synthetic import make_default_scene
from grassmann.initialization import init_gaussians_from_points
from grassmann.trainable import trainable_from_params
from grassmann.training import Trainer, TrainerConfig
from grassmann.fast_rasterizer import (
    is_available, sigma3d_to_cov6,
    camera_to_view_matrix, camera_to_proj_matrix, compute_tanfov,
    fast_rasterize, FastRasterConfig,
)


DTYPE = torch.float32
torch.manual_seed(42)


# =============================================================================
# Availability
# =============================================================================

def test_is_available_returns_bool():
    """is_available() should return a bool without raising."""
    result = is_available()
    assert isinstance(result, bool)


def test_not_available_on_cpu():
    """Without CUDA, is_available() should be False regardless of package presence."""
    if not torch.cuda.is_available():
        assert is_available() is False


# =============================================================================
# Geometry helpers
# =============================================================================

def test_sigma3d_to_cov6_layout():
    """Packing should match [xx, xy, xz, yy, yz, zz]."""
    S = torch.tensor([[
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 5.0],
        [3.0, 5.0, 6.0],
    ]], dtype=DTYPE)
    cov6 = sigma3d_to_cov6(S)
    expected = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=DTYPE)
    assert torch.allclose(cov6, expected)


def test_sigma3d_to_cov6_batch():
    """Batching works; shape correct."""
    S = torch.randn(5, 3, 3, dtype=DTYPE)
    # Make symmetric
    S = 0.5 * (S + S.transpose(-1, -2))
    cov6 = sigma3d_to_cov6(S)
    assert cov6.shape == (5, 6)


def test_view_matrix_transforms_world_to_camera():
    """Points transformed by the view matrix should match world_to_camera."""
    scene = make_default_scene(n_cams=2)
    cam = scene.cameras[0]

    # Convert cam to float32 for consistency
    cam_f = Camera(R=cam.R.to(DTYPE), c=cam.c.to(DTYPE),
                    fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy)

    V = camera_to_view_matrix(cam_f)  # (4, 4)
    assert V.shape == (4, 4)
    # V is built with the "point as row vector" glm convention:
    #   X_cam_hom = X_world_hom @ V
    # so we test:
    X_world = torch.tensor([[1.0, 2.0, 5.0]], dtype=DTYPE)
    X_world_hom = torch.cat([X_world, torch.ones(1, 1, dtype=DTYPE)], dim=-1)   # (1, 4)
    X_cam_hom_via_V = X_world_hom @ V                                            # (1, 4)
    X_cam_via_V = X_cam_hom_via_V[:, :3] / X_cam_hom_via_V[:, 3:]

    # Compare to our known-correct world_to_camera
    X_cam_direct = world_to_camera(X_world, cam_f)                                # (1, 3)

    assert torch.allclose(X_cam_via_V, X_cam_direct, atol=1e-5), \
        f"view matrix disagrees: V-path={X_cam_via_V}, direct={X_cam_direct}"


def test_tanfov_matches_definition():
    """tanfov = image_size / (2 * focal_length)."""
    cam = Camera(
        R=torch.eye(3, dtype=DTYPE), c=torch.zeros(3, dtype=DTYPE),
        fx=500.0, fy=400.0, cx=100.0, cy=75.0,
    )
    tx, ty = compute_tanfov(cam, H=150, W=200)
    assert abs(tx - 200 / (2 * 500)) < 1e-8
    assert abs(ty - 150 / (2 * 400)) < 1e-8


def test_proj_matrix_shape():
    """Projection matrix is 4x4."""
    cam = Camera(R=torch.eye(3, dtype=DTYPE), c=torch.zeros(3, dtype=DTYPE),
                 fx=500, fy=500, cx=100, cy=75)
    P = camera_to_proj_matrix(cam, H=150, W=200)
    assert P.shape == (4, 4)


# =============================================================================
# fast_rasterize fallback behavior
# =============================================================================

def test_fast_rasterize_falls_back_on_cpu():
    """On CPU, fast_rasterize should transparently fall back to the toy rasterizer."""
    scene = make_default_scene(n_cams=2, image_w=40, image_h=30)
    # Build a simple model.
    points = torch.tensor([[0.2, -0.1, 0.3], [-0.3, 0.2, 0.1]], dtype=torch.float64)
    times_t = torch.tensor([1.0, 1.0], dtype=torch.float64)
    colors = torch.tensor([[1.0, 0.2, 0.2], [0.2, 1.0, 0.2]], dtype=torch.float64)
    params_init = init_gaussians_from_points(
        points, times_t, scene.cameras, colors=colors,
        sigma_aa=0.02, sigma_bb=0.1, opacity=0.5, sigma_k_pixel=2.0,
    )
    model = trainable_from_params(params_init, dtype=DTYPE)
    params = model.forward()

    bg = torch.tensor([0.0, 0.0, 0.0], dtype=DTYPE)
    img = fast_rasterize(params, t_0=1.0, cam=scene.cameras[0],
                         H=scene.H, W=scene.W, background=bg)
    assert img.shape == (scene.H, scene.W, 3)
    assert not torch.isnan(img).any()
    assert img.min() >= 0.0 and img.max() <= 1.0 + 1e-5


def test_fast_rasterize_matches_toy_in_fallback():
    """When fast_rasterize falls back to toy, its output should equal direct toy."""
    scene = make_default_scene(n_cams=2, image_w=40, image_h=30)
    points = torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float64)
    times_t = torch.tensor([1.0], dtype=torch.float64)
    colors = torch.tensor([[1.0, 0.2, 0.2]], dtype=torch.float64)
    params_init = init_gaussians_from_points(
        points, times_t, scene.cameras, colors=colors,
        sigma_aa=0.02, sigma_bb=0.1, opacity=0.5, sigma_k_pixel=2.0,
    )
    model = trainable_from_params(params_init, dtype=DTYPE)
    params = model.forward()
    bg = torch.tensor([0.0, 0.0, 0.0], dtype=DTYPE)

    # Route A: fast_rasterize (forced fallback).
    img_fast = fast_rasterize(params, t_0=1.0, cam=scene.cameras[0],
                               H=scene.H, W=scene.W, background=bg,
                               force_fallback=True)

    # Route B: direct toy rasterize path.
    derived = compute_derived(params)
    tc = condition_on_time(params, derived, 1.0)
    sg = project_to_screen(params, tc, scene.cameras[0])
    img_toy = toy_rasterize(sg, H=scene.H, W=scene.W, background=bg)

    assert torch.allclose(img_fast, img_toy, atol=1e-6), \
        f"max diff = {(img_fast - img_toy).abs().max()}"


# =============================================================================
# Trainer with use_fast_rasterizer=True (should be transparent on CPU)
# =============================================================================

def test_trainer_with_fast_rasterizer_config():
    """Setting use_fast_rasterizer=True on CPU should still work (falls back)."""
    scene = make_default_scene(n_cams=2, image_w=30, image_h=20)
    target = torch.rand(scene.H, scene.W, 3, dtype=DTYPE)

    points = torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float64)
    times_t = torch.tensor([1.0], dtype=torch.float64)
    colors = torch.tensor([[1.0, 0.2, 0.2]], dtype=torch.float64)
    params_init = init_gaussians_from_points(
        points, times_t, scene.cameras, colors=colors,
        sigma_aa=0.02, sigma_bb=0.1, opacity=0.5, sigma_k_pixel=2.0,
    )
    model = trainable_from_params(params_init, dtype=DTYPE)

    frame_data = target.unsqueeze(0).unsqueeze(0)
    config = TrainerConfig(
        num_iters=5, log_every=5,
        background=torch.zeros(3, dtype=DTYPE),
        use_fast_rasterizer=True,       # request fast path; CPU will fall back
    )
    trainer = Trainer(
        model=model, cameras=[scene.cameras[0]],
        frame_data=frame_data, times=[1.0],
        H=scene.H, W=scene.W, config=config,
    )
    # Must run without error.
    trainer.train(num_iters=5, log_every=5)
    assert len(trainer.history["loss"]) == 1
    assert trainer.history["loss"][0] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
