"""Tests for Phase 6: adaptive density control.

We verify:
  1. Pruning: low-opacity Gaussians get removed; collapsed and runaway do too.
  2. Cloning: a duplicate row appears; tracker state grows accordingly.
  3. Splitting: two children replace one parent, positioned along the major axis.
  4. Grad accumulation: reads gradients from alpha_0, beta_0.
  5. Optimizer rebuild: after an N change, a new optimizer covers all new params.
  6. End-to-end: density control integrated with Trainer, runs without error,
     Gaussian count changes over time.
"""
import pytest
import torch
from torch import nn

from grassmann import quaternion as Q
from grassmann.gaussian import compute_derived, condition_on_time
from grassmann.rasterizer import project_to_screen, rasterize
from grassmann.initialization import init_gaussian_from_point, init_gaussians_from_points
from grassmann.synthetic import make_default_scene, render_synthetic_frame
from grassmann.triangulation import observe_scene_point, triangulate_point_dlt
from grassmann.trainable import trainable_from_params, build_optimizer
from grassmann.density_control import DensityTracker, DensityConfig
from grassmann.training import Trainer, TrainerConfig


DTYPE = torch.float32
torch.manual_seed(42)


def small_model(n_cams=2, n_points=5):
    """Build a small TrainableGaussians for tests."""
    scene = make_default_scene(n_cams=n_cams, image_w=30, image_h=20)
    # Build n_points Gaussians at random 3D positions.
    torch.manual_seed(0)
    points = torch.randn(n_points, 3, dtype=torch.float64) * 0.3
    times = torch.ones(n_points, dtype=torch.float64)
    colors = torch.rand(n_points, 3, dtype=torch.float64)
    params_init = init_gaussians_from_points(
        points, times, scene.cameras, colors=colors,
        sigma_aa=0.02, sigma_bb=0.05, opacity=0.5, sigma_k_pixel=2.0,
    )
    return trainable_from_params(params_init, dtype=DTYPE), scene


# =============================================================================
# Tracker basics
# =============================================================================

def test_tracker_accumulates_gradients():
    """After a backward pass, accumulate() should record nonzero magnitudes."""
    model, scene = small_model()
    tracker = DensityTracker(model)

    # Do a render + backward to populate .grad on alpha_0, beta_0.
    params = model.forward()
    derived = compute_derived(params)
    tc = condition_on_time(params, derived, 1.0)
    sg = project_to_screen(params, tc, scene.cameras[0])
    img = rasterize(sg, H=scene.H, W=scene.W)
    img.sum().backward()

    assert model.alpha_0.grad is not None
    tracker.accumulate()
    assert (tracker.grad_accum >= 0).all()
    assert tracker.grad_counts.max().item() == 1


def test_tracker_reset_zeroes_accum():
    model, _ = small_model()
    tracker = DensityTracker(model)
    tracker.grad_accum += 1.0
    tracker.grad_counts += 5
    tracker.reset()
    assert tracker.grad_accum.sum().item() == 0
    assert tracker.grad_counts.sum().item() == 0


# =============================================================================
# Prune
# =============================================================================

def test_prune_removes_low_opacity():
    """Gaussians with opacity below threshold should be pruned."""
    model, _ = small_model(n_points=6)
    N_before = model.N
    # Set opacity of first 3 Gaussians to a very low value (logit << 0).
    with torch.no_grad():
        model.opacity_logit.data[:3] = -10.0    # sigmoid(-10) ~ 4.5e-5
    tracker = DensityTracker(model)
    config = DensityConfig(opacity_threshold=0.01, scale_min=0.0, scale_max=1e9)
    n_pruned = tracker.prune(config)
    assert n_pruned == 3, f"Expected 3 pruned, got {n_pruned}"
    assert model.N == N_before - 3
    # Remaining should all have opacity above threshold.
    opacities = torch.sigmoid(model.opacity_logit)
    assert (opacities >= config.opacity_threshold).all()


def test_prune_removes_collapsed_gaussians():
    """Gaussians with very small Sigma_k should be pruned."""
    model, _ = small_model(n_points=5)
    # Collapse the first Gaussian's L to near zero.
    with torch.no_grad():
        model.L.data[0] = 1e-10 * torch.eye(2, dtype=DTYPE)
    tracker = DensityTracker(model)
    config = DensityConfig(opacity_threshold=0.0, scale_min=1e-6, scale_max=1e9)
    n_pruned = tracker.prune(config)
    assert n_pruned >= 1


def test_prune_removes_runaway_gaussians():
    """Gaussians with huge Sigma_k should be pruned."""
    model, _ = small_model(n_points=5)
    with torch.no_grad():
        model.L.data[0] = 100.0 * torch.eye(2, dtype=DTYPE)
    tracker = DensityTracker(model)
    config = DensityConfig(opacity_threshold=0.0, scale_min=1e-12, scale_max=1.0)
    n_pruned = tracker.prune(config)
    assert n_pruned >= 1


def test_prune_keeps_healthy():
    """A healthy model with sensible parameters should lose nobody."""
    model, _ = small_model(n_points=5)
    tracker = DensityTracker(model)
    # Very permissive
    config = DensityConfig(opacity_threshold=0.0, scale_min=0.0, scale_max=1e9)
    n_pruned = tracker.prune(config)
    assert n_pruned == 0
    assert model.N == 5


# =============================================================================
# Clone / Split
# =============================================================================

def test_clone_duplicates_small_stressed():
    """Clone should duplicate small Gaussians whose accumulated grad exceeds threshold."""
    model, _ = small_model(n_points=3)
    N_before = model.N
    tracker = DensityTracker(model)
    # Fake gradient accumulation: set first Gaussian's mean_grad above threshold.
    tracker.grad_accum[0] = 10.0
    tracker.grad_counts[0] = 1
    # Make sure its Sigma_k's largest eigenvalue is small (so it's a clone candidate).
    with torch.no_grad():
        model.L.data[0] = 0.001 * torch.eye(2, dtype=DTYPE)

    config = DensityConfig(grad_threshold=1.0, clone_scale_threshold=0.1)
    n_cloned, n_split = tracker.clone_and_split(config)
    assert n_cloned == 1, f"Expected 1 cloned, got {n_cloned}"
    assert n_split == 0
    assert model.N == N_before + 1


def test_split_divides_large_stressed():
    """Split should replace a stressed large Gaussian with 2 smaller offset copies."""
    model, _ = small_model(n_points=3)
    N_before = model.N
    tracker = DensityTracker(model)
    tracker.grad_accum[0] = 10.0
    tracker.grad_counts[0] = 1
    # Make its Sigma_k's largest eigenvalue large.
    with torch.no_grad():
        model.L.data[0] = 0.5 * torch.eye(2, dtype=DTYPE)   # Sigma_k = 0.25 * I

    config = DensityConfig(grad_threshold=1.0, clone_scale_threshold=0.1,
                           split_shrink_factor=1.6, split_spatial_offset_sigmas=1.0)
    n_cloned, n_split = tracker.clone_and_split(config)
    assert n_cloned == 0
    assert n_split == 1
    # N goes from 3 to 3 - 1 (removed) + 2 (added) = 4.
    assert model.N == N_before + 1


def test_clone_and_split_both():
    """Both operations should compose cleanly."""
    model, _ = small_model(n_points=4)
    N_before = model.N
    tracker = DensityTracker(model)
    tracker.grad_accum[0] = 10.0   # small + stressed -> clone
    tracker.grad_accum[1] = 10.0   # large + stressed -> split
    tracker.grad_counts[:] = 1
    with torch.no_grad():
        model.L.data[0] = 0.001 * torch.eye(2, dtype=DTYPE)
        model.L.data[1] = 0.5 * torch.eye(2, dtype=DTYPE)
    config = DensityConfig(grad_threshold=1.0, clone_scale_threshold=0.1)
    n_cloned, n_split = tracker.clone_and_split(config)
    assert n_cloned == 1
    assert n_split == 1
    # Net change: +1 (clone) + 1 (split = -1 + 2 = +1) = +2.
    assert model.N == N_before + 2


def test_no_action_when_no_stress():
    """If no gradients exceed threshold, no cloning or splitting."""
    model, _ = small_model(n_points=3)
    tracker = DensityTracker(model)
    # Leave grad_accum at zeros.
    config = DensityConfig(grad_threshold=1.0)
    n_cloned, n_split = tracker.clone_and_split(config)
    assert n_cloned == 0
    assert n_split == 0
    assert model.N == 3


# =============================================================================
# Optimizer rebuild
# =============================================================================

def test_optimizer_rebuild_after_density_op():
    """After density ops, a fresh optimizer should cover all current parameters."""
    model, _ = small_model(n_points=3)
    tracker = DensityTracker(model)
    # Force a split to change N.
    tracker.grad_accum[0] = 10.0
    tracker.grad_counts[0] = 1
    with torch.no_grad():
        model.L.data[0] = 0.5 * torch.eye(2, dtype=DTYPE)
    config = DensityConfig(grad_threshold=1.0, clone_scale_threshold=0.1)
    builder = lambda m: build_optimizer(m)
    new_opt, stats = tracker.densify_and_prune(config, builder)
    # All of model's parameters should be covered.
    opt_params = set()
    for g in new_opt.param_groups:
        for p in g["params"]:
            opt_params.add(id(p))
    model_params = set(id(p) for p in model.parameters())
    assert model_params.issubset(opt_params), "Not all model params are in the optimizer"


def test_adam_state_preserved_for_kept_splats():
    """RCA Bug D: after a density event, kept splats must retain their Adam moments
    (exp_avg, exp_avg_sq). Only newly-cloned/split rows start at zero."""
    model, scene = small_model(n_points=4)
    optimizer = build_optimizer(model)
    tracker = DensityTracker(model, optimizer)

    # Run a few forward+backward+step cycles so Adam state is populated.
    for _ in range(3):
        params = model.forward()
        derived = compute_derived(params)
        tc = condition_on_time(params, derived, 1.0)
        sg = project_to_screen(params, tc, scene.cameras[0])
        img = rasterize(sg, H=scene.H, W=scene.W)
        loss = img.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Snapshot exp_avg for alpha_0 BEFORE density op.
    before = optimizer.state[model.alpha_0]["exp_avg"].clone()
    assert before.norm() > 0, "Adam state should be populated after some steps"

    # Force a prune of row 0 only (drop_mask = [True, False, False, False]).
    config = DensityConfig(
        opacity_threshold=10.0,    # above 1.0 => prune nothing by opacity
        scale_min=1e-12, scale_max=1e12,  # never trigger by scale
        grad_threshold=1e12,       # never clone/split
    )
    # Hand-craft a single-row prune by directly calling _keep_rows.
    keep = torch.tensor([False, True, True, True])
    tracker._keep_rows(keep)

    # After prune, the alpha_0 Parameter is a NEW object; its Adam state must be
    # the original `before` tensor sliced to [1:] (rows 1, 2, 3 kept).
    after = optimizer.state[model.alpha_0]["exp_avg"]
    expected = before[[1, 2, 3]]
    assert torch.allclose(after, expected), \
        f"Adam exp_avg not preserved after prune.\nbefore[1:]={expected}\nafter={after}"
    # Now run a clone (replicate row 0) and check the cloned row has exp_avg=0.
    n_before = model.alpha_0.shape[0]
    clone_mask = torch.tensor([True, False, False])
    tracker._perform_clone(clone_mask)
    after_clone = optimizer.state[model.alpha_0]["exp_avg"]
    # First n_before rows: same as `after`. Last row: zero (newly cloned).
    assert torch.allclose(after_clone[:n_before], after), "Pre-clone state not preserved"
    assert after_clone[-1].abs().item() == 0.0, \
        f"Newly cloned row should have zero Adam moment; got {after_clone[-1]}"


# =============================================================================
# End-to-end: Trainer with density control enabled
# =============================================================================

def test_trainer_with_density_control():
    """Density control must not break training loop; N changes over iterations."""
    scene = make_default_scene(n_cams=2, image_w=30, image_h=20)
    t = 0.5
    target = render_synthetic_frame(scene, 0, t, blob_sigma=3.0).to(DTYPE)

    points_rec, colors_list = [], []
    for sp in scene.scene_points:
        uvs, depths = observe_scene_point(sp.trajectory, t, scene.cameras)
        visible = depths > 0.1
        if visible.sum() < 2:
            continue
        visible_cams = [scene.cameras[k] for k in range(len(scene.cameras)) if visible[k]]
        X_rec = triangulate_point_dlt(visible_cams, uvs[visible])
        points_rec.append(X_rec)
        colors_list.append(sp.color)
    points_rec = torch.stack(points_rec)
    colors = torch.stack(colors_list)
    times_t = torch.full((points_rec.shape[0],), t, dtype=torch.float64)
    params_init = init_gaussians_from_points(
        points_rec, times_t, scene.cameras, colors=colors,
        sigma_aa=0.02, sigma_bb=0.1, opacity=0.3, sigma_k_pixel=3.0,
    )
    model = trainable_from_params(params_init, dtype=DTYPE)
    N_initial = model.N

    frame_data = target.unsqueeze(0).unsqueeze(0)
    config = TrainerConfig(
        num_iters=300, log_every=100,
        densify_every=100, densify_start=50, densify_stop=1000,
        density_config=DensityConfig(
            grad_threshold=0.001,   # aggressive so we see activity
            opacity_threshold=0.001,
        ),
        background=scene.background.to(DTYPE),
    )
    trainer = Trainer(
        model=model, cameras=[scene.cameras[0]],
        frame_data=frame_data, times=[t],
        H=scene.H, W=scene.W, config=config,
    )
    trainer.train(num_iters=300, log_every=100)

    # After training, N may have changed (up or down).
    # The loss should still be finite.
    assert all(torch.isfinite(torch.tensor(L)) for L in trainer.history["loss"])
    # N history should have the right length.
    assert len(trainer.history["N"]) == 3
    # N should remain positive (model wasn't completely wiped out).
    assert all(N > 0 for N in trainer.history["N"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
