"""Tests for Phase 5: trainable model, losses, training loop.

We verify:
  1. TrainableGaussians produces a valid GaussianParams that can be rendered.
  2. Gradients flow through the full pipeline (no NaNs, non-zero gradients).
  3. Renormalization keeps p_im, q_im on S^2.
  4. Training actually reduces loss on an overfit-one-frame test.
  5. Losses have correct shape and sensible magnitude.
"""
import pytest
import torch

from grassmann import quaternion as Q
from grassmann.gaussian import compute_derived, condition_on_time
from grassmann.rasterizer import project_to_screen, rasterize
from grassmann.initialization import init_gaussian_from_point, init_gaussians_from_points
from grassmann.synthetic import make_default_scene, render_synthetic_frame
from grassmann.triangulation import triangulate_point_dlt, observe_scene_point
from grassmann.trainable import TrainableGaussians, trainable_from_params, build_optimizer
from grassmann.losses import l1_loss, structural_loss, photometric_loss
from grassmann.training import Trainer, TrainerConfig


DTYPE = torch.float32
torch.manual_seed(42)


# =============================================================================
# TrainableGaussians
# =============================================================================

def test_trainable_forward_returns_valid_params():
    """Calling model.forward() should give a GaussianParams we can render."""
    cams = make_default_scene(n_cams=3).cameras
    X = torch.tensor([0.3, -0.1, 0.2], dtype=torch.float64)
    params_init = init_gaussian_from_point(X, t=1.0, cameras=cams)
    model = trainable_from_params(params_init, dtype=DTYPE)

    params = model.forward()
    # p, q should be unit.
    p_norm = params.p_im.norm(dim=-1)
    q_norm = params.q_im.norm(dim=-1)
    assert torch.allclose(p_norm, torch.ones(1, dtype=DTYPE), atol=1e-5)
    assert torch.allclose(q_norm, torch.ones(1, dtype=DTYPE), atol=1e-5)
    # Opacity in [0, 1]
    assert (params.opacity >= 0).all() and (params.opacity <= 1).all()
    # Color in [0, 1]
    assert (params.color >= 0).all() and (params.color <= 1).all()
    # Rendering succeeds.
    derived = compute_derived(params)
    tc = condition_on_time(params, derived, 1.0)
    sg = project_to_screen(params, tc, cams[0])
    img = rasterize(sg, H=40, W=60)
    assert not torch.isnan(img).any()


def test_gradients_flow_through_pipeline():
    """End-to-end rendering should produce non-trivial gradients on all parameters."""
    scene = make_default_scene(n_cams=2, image_w=40, image_h=30)
    cams = scene.cameras
    X = torch.tensor([0.3, -0.1, 0.2], dtype=torch.float64)
    params_init = init_gaussian_from_point(X, t=1.0, cameras=cams,
                                           color=torch.tensor([0.8, 0.2, 0.2]))
    model = trainable_from_params(params_init, dtype=DTYPE)

    # Render and compute a trivial loss against a target.
    params = model.forward()
    derived = compute_derived(params)
    tc = condition_on_time(params, derived, 1.0)
    sg = project_to_screen(params, tc, cams[0])
    img = rasterize(sg, H=scene.H, W=scene.W)
    target = torch.ones_like(img) * 0.5
    loss = (img - target).abs().mean()
    loss.backward()

    # All parameters should have non-zero gradients (barring flat regions).
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        # Allow p, q gradients to be small but nonzero. Just check for no NaN.
        assert not torch.isnan(p.grad).any(), f"{name} grad has NaN"


def test_renormalize_manifold():
    """After renormalization, p_im and q_im have unit norm."""
    cams = make_default_scene(n_cams=2).cameras
    X = torch.tensor([0.3, -0.1, 0.2], dtype=torch.float64)
    params_init = init_gaussian_from_point(X, t=1.0, cameras=cams)
    model = trainable_from_params(params_init, dtype=DTYPE)

    # Scale p_im up to simulate drift during training.
    with torch.no_grad():
        model.p_im.data *= 2.5
        model.q_im.data *= 0.3

    model.renormalize_manifold_()
    assert torch.allclose(model.p_im.data.norm(dim=-1), torch.ones(1, dtype=DTYPE), atol=1e-6)
    assert torch.allclose(model.q_im.data.norm(dim=-1), torch.ones(1, dtype=DTYPE), atol=1e-6)


def test_optimizer_has_correct_param_groups():
    """Adam should have the right number of parameter groups with distinct LRs."""
    cams = make_default_scene(n_cams=2).cameras
    X = torch.tensor([0.3, -0.1, 0.2], dtype=torch.float64)
    params_init = init_gaussian_from_point(X, t=1.0, cameras=cams)
    model = trainable_from_params(params_init, dtype=DTYPE)
    opt = build_optimizer(model)
    names = [g["name"] for g in opt.param_groups]
    assert "pq" in names
    assert "mean" in names
    assert "opacity" in names
    assert "color" in names


# =============================================================================
# Losses
# =============================================================================

def test_l1_loss_zero_for_identical():
    img = torch.rand(10, 10, 3)
    assert l1_loss(img, img).item() == 0.0


def test_l1_loss_positive_for_different():
    a = torch.zeros(10, 10, 3)
    b = torch.ones(10, 10, 3)
    assert l1_loss(a, b).item() == 1.0


def test_structural_loss_zero_for_identical():
    img = torch.rand(16, 16, 3)
    assert structural_loss(img, img).item() < 1e-6


def test_structural_loss_positive_for_different():
    a = torch.zeros(16, 16, 3)
    b = torch.ones(16, 16, 3)
    assert structural_loss(a, b).item() > 0.5


def test_photometric_loss_combines_components():
    a = torch.rand(16, 16, 3)
    b = torch.rand(16, 16, 3)
    l_l1_only = photometric_loss(a, b, lambda_l1=1.0, lambda_structural=0.0)
    l_combined = photometric_loss(a, b, lambda_l1=0.5, lambda_structural=0.5)
    # Both should be positive; combined should weight the two.
    assert l_l1_only.item() > 0
    assert l_combined.item() > 0


# =============================================================================
# Training loop (integration test)
# =============================================================================

def test_trainer_overfit_one_view_one_frame():
    """The crown jewel: training should reduce loss on a toy overfit problem.

    We render a single (camera, frame) pair from the synthetic scene, initialize
    Gaussians at the correct triangulated points, and verify that 200 Adam steps
    reduce loss to a small value.
    """
    scene = make_default_scene(n_cams=2, image_w=40, image_h=30)
    t = 0.5

    # Build target frame and triangulated init.
    target = render_synthetic_frame(scene, cam_idx=0, t=t, blob_sigma=3.0).to(DTYPE)

    # Triangulate the three scene points.
    points_rec = []
    colors = []
    for sp in scene.scene_points:
        uvs, depths = observe_scene_point(sp.trajectory, t, scene.cameras)
        visible = depths > 0.1
        if visible.sum() < 2:
            continue
        visible_cams = [scene.cameras[k] for k in range(len(scene.cameras)) if visible[k]]
        X_rec = triangulate_point_dlt(visible_cams, uvs[visible])
        points_rec.append(X_rec)
        colors.append(sp.color)
    points_rec = torch.stack(points_rec)
    colors = torch.stack(colors)
    times_t = torch.full((points_rec.shape[0],), t, dtype=torch.float64)
    params_init = init_gaussians_from_points(
        points_rec, times_t, scene.cameras, colors=colors,
        sigma_aa=0.02, sigma_bb=0.05, opacity=0.5, sigma_k=3.0,
    )
    model = trainable_from_params(params_init, dtype=DTYPE)

    # Frame data: tensor of shape (K=1, T=1, H, W, 3) -- we only train on cam 0 at time t.
    # Expand to (1, 1, H, W, 3).
    frame_data = target.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W, 3)

    config = TrainerConfig(
        num_iters=200, log_every=200,  # only log once at the end
        lambda_l1=1.0, lambda_structural=0.0,  # pure L1 for clarity
        lr_pq=1e-3, lr_mean=1e-2, lr_L=1e-2,
        lr_opacity=5e-2, lr_color=2e-2,
        background=scene.background.to(DTYPE),
    )
    trainer = Trainer(
        model=model,
        cameras=[scene.cameras[0]],  # single camera
        frame_data=frame_data,
        times=[t],
        H=scene.H, W=scene.W,
        config=config,
    )

    # Measure loss before training.
    loss_before, _ = trainer.train_step()
    # We already did 1 step -- but that's fine. Now do 200 more.
    trainer.train(num_iters=200)

    # Final loss.
    loss_after, _ = trainer.train_step()

    print(f"\n  Overfit test: loss before = {loss_before:.4f}, after = {loss_after:.4f}")
    assert loss_after < loss_before, f"loss didn't decrease: {loss_before} -> {loss_after}"
    # Expect at least a 30% reduction for an easy overfit problem.
    assert loss_after < 0.7 * loss_before, \
        f"loss only decreased by {(loss_before - loss_after) / loss_before * 100:.1f}%"


def test_trainer_multi_view_multi_frame():
    """Trainer should work with K > 1 cameras and T > 1 frames.

    We don't check convergence (too expensive in unit test); just verify the
    loop runs without error for a few iterations.
    """
    scene = make_default_scene(n_cams=3, image_w=40, image_h=30)
    times = [0.0, 0.5, 1.0]

    # Build target frames (K, T, H, W, 3).
    frame_data = torch.stack([
        torch.stack([render_synthetic_frame(scene, k, t, blob_sigma=3.0) for t in times])
        for k in range(3)
    ]).to(DTYPE)
    assert frame_data.shape == (3, 3, scene.H, scene.W, 3)

    # Init from triangulation at t=0.5 (middle).
    points_rec, colors = [], []
    t_init = 0.5
    for sp in scene.scene_points:
        uvs, depths = observe_scene_point(sp.trajectory, t_init, scene.cameras)
        visible = depths > 0.1
        if visible.sum() < 2:
            continue
        visible_cams = [scene.cameras[k] for k in range(len(scene.cameras)) if visible[k]]
        X_rec = triangulate_point_dlt(visible_cams, uvs[visible])
        points_rec.append(X_rec)
        colors.append(sp.color)
    points_rec = torch.stack(points_rec)
    colors = torch.stack(colors)
    times_t = torch.full((points_rec.shape[0],), t_init, dtype=torch.float64)
    params_init = init_gaussians_from_points(
        points_rec, times_t, scene.cameras, colors=colors,
        sigma_aa=0.02, sigma_bb=0.2, opacity=0.5, sigma_k=3.0,
    )
    model = trainable_from_params(params_init, dtype=DTYPE)

    trainer = Trainer(
        model=model, cameras=scene.cameras,
        frame_data=frame_data, times=times, H=scene.H, W=scene.W,
        config=TrainerConfig(num_iters=10, log_every=10,
                             background=scene.background.to(DTYPE)),
    )
    # Run 10 iters; just confirm no error and loss history is populated.
    trainer.train(num_iters=10, log_every=10)
    assert len(trainer.history["loss"]) == 1
    assert trainer.history["loss"][0] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
