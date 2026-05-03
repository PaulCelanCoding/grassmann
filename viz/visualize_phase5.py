"""Phase 5 visualization: watch the model learn.

Two experiments:

  Demo A: Overfit one view + one frame. Simplest possible test. Show loss
          curve and rendered image at iters [0, 200, 500, 1000].

  Demo B: Multi-view multi-frame training. 3 cameras on a ring, 5 frames.
          Show loss curve and GT-vs-rendered grid (at a validation time/cam).
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from grassmann import quaternion as Q
from grassmann.gaussian import compute_derived, condition_on_time
from grassmann.rasterizer import project_to_screen, rasterize
from grassmann.initialization import init_gaussians_from_points
from grassmann.synthetic import make_default_scene, render_synthetic_frame
from grassmann.triangulation import triangulate_point_dlt, observe_scene_point
from grassmann.trainable import trainable_from_params
from grassmann.training import Trainer, TrainerConfig


DTYPE = torch.float32


def triangulate_all_points_at_time(scene, t):
    """Triangulate all scene points at time t. Returns (points, colors)."""
    points_rec, colors = [], []
    for sp in scene.scene_points:
        uvs, depths = observe_scene_point(sp.trajectory, t, scene.cameras)
        visible = depths > 0.1
        if visible.sum() < 2:
            continue
        visible_cams = [scene.cameras[k] for k in range(len(scene.cameras)) if visible[k]]
        X_rec = triangulate_point_dlt(visible_cams, uvs[visible])
        points_rec.append(X_rec)
        colors.append(sp.color)
    return torch.stack(points_rec), torch.stack(colors)


def render_current(model, cam, t, H, W, background):
    params = model.forward()
    derived = compute_derived(params)
    tc = condition_on_time(params, derived, t)
    sg = project_to_screen(params, tc, cam)
    bg = background.to(params.color.dtype)
    return rasterize(sg, H=H, W=W, background=bg).detach().cpu().numpy().clip(0, 1)


# ----------------------------------------------------------------------------
# Demo A: Overfit one (camera, frame) pair
# ----------------------------------------------------------------------------

def demo_overfit():
    """Train on a single frame from a single camera. Should converge fast."""
    print("--- Demo A: Overfit one view + one frame ---")
    scene = make_default_scene(n_cams=2, image_w=100, image_h=70)
    t = 0.5
    cam_idx = 0

    target = render_synthetic_frame(scene, cam_idx, t, blob_sigma=3.0).to(DTYPE)

    # Init from triangulation.
    points_rec, colors = triangulate_all_points_at_time(scene, t)
    times_t = torch.full((points_rec.shape[0],), t, dtype=torch.float64)
    # Initialize with wrong colors on purpose to force training to learn colors.
    wrong_colors = torch.full_like(colors, 0.5)
    params_init = init_gaussians_from_points(
        points_rec, times_t, scene.cameras, colors=wrong_colors,
        sigma_aa=0.02, sigma_bb=0.05, opacity=0.5, sigma_k_pixel=3.0, sigma_k_temporal=3.0,
    )
    model = trainable_from_params(params_init, dtype=DTYPE)

    # Snapshot renders at key iterations.
    frame_data = target.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W, 3)
    config = TrainerConfig(
        num_iters=1000, log_every=100,
        lambda_l1=1.0, lambda_structural=0.0,
        lr_pq=1e-3, lr_mean=1e-2, lr_L=1e-2,
        lr_opacity=5e-2, lr_color=5e-2,
        background=scene.background.to(DTYPE),
    )
    trainer = Trainer(
        model=model, cameras=[scene.cameras[cam_idx]],
        frame_data=frame_data, times=[t],
        H=scene.H, W=scene.W, config=config,
    )

    snapshots = {}
    snapshots[0] = render_current(model, scene.cameras[cam_idx], t, scene.H, scene.W, scene.background)

    for target_iter in [200, 500, 1000]:
        trainer.train(num_iters=target_iter - sum([k for k in snapshots if k < target_iter]) or target_iter,
                      log_every=max(target_iter, 1))
        snapshots[target_iter] = render_current(
            model, scene.cameras[cam_idx], t, scene.H, scene.W, scene.background)

    iters_logged = trainer.history["iter"]
    losses = trainer.history["loss"]

    # Plot
    fig = plt.figure(figsize=(15, 5))
    # Left: loss curve
    ax_loss = plt.subplot(1, 5, 1)
    ax_loss.plot(iters_logged, losses, "o-")
    ax_loss.set_xlabel("iteration")
    ax_loss.set_ylabel("loss")
    ax_loss.set_yscale("log")
    ax_loss.set_title("Training loss (log scale)")
    ax_loss.grid(alpha=0.3)

    # Panels 2-5: target + renders at iter 0, 200, 500, 1000
    for i, key in enumerate([0, 200, 500, 1000]):
        ax = plt.subplot(1, 5, i + 2)
        ax.imshow(snapshots[key])
        ax.set_title(f"iter {key}" if key > 0 else "iter 0 (init)")
        ax.axis("off")
    fig.suptitle("Demo A: Overfit single frame. Initial colors were wrong gray; training recovers them.",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig("docs/images/phase5_overfit.png", dpi=110, bbox_inches="tight")
    plt.close()

    # Final target vs rendered side-by-side.
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(target.cpu().numpy().clip(0, 1))
    axes[0].set_title("Target")
    axes[0].axis("off")
    axes[1].imshow(snapshots[1000])
    axes[1].set_title("Trained (1000 iters)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("docs/images/phase5_overfit_final.png", dpi=110, bbox_inches="tight")
    plt.close()

    print(f"  Final loss: {losses[-1]:.4f}")
    print("  Saved phase5_overfit.png, phase5_overfit_final.png")


# ----------------------------------------------------------------------------
# Demo B: Multi-view, multi-frame training
# ----------------------------------------------------------------------------

def demo_multiview_multiframe():
    """Train on 3 cameras x 5 frames."""
    print("\n--- Demo B: Multi-view, multi-frame ---")
    scene = make_default_scene(n_cams=3, image_w=100, image_h=70)
    times = [0.0, 0.25, 0.5, 0.75, 1.0]

    # Build target frames (K, T, H, W, 3).
    frame_data = torch.stack([
        torch.stack([render_synthetic_frame(scene, k, t, blob_sigma=3.0) for t in times])
        for k in range(3)
    ]).to(DTYPE)

    # Init: triangulate at EACH time and collect into one big model.
    all_points, all_times, all_colors = [], [], []
    for t in times:
        pts, cols = triangulate_all_points_at_time(scene, t)
        all_points.append(pts)
        all_times.append(torch.full((pts.shape[0],), t, dtype=torch.float64))
        all_colors.append(cols)
    points_rec = torch.cat(all_points, dim=0)
    times_t = torch.cat(all_times, dim=0)
    colors = torch.cat(all_colors, dim=0)
    print(f"  Initialized {points_rec.shape[0]} Gaussians "
          f"({len(scene.scene_points)} points x {len(times)} times)")

    params_init = init_gaussians_from_points(
        points_rec, times_t, scene.cameras, colors=colors,
        sigma_aa=0.02, sigma_bb=0.15, opacity=0.5, sigma_k_pixel=3.0, sigma_k_temporal=3.0,
    )
    model = trainable_from_params(params_init, dtype=DTYPE)

    config = TrainerConfig(
        num_iters=1500, log_every=100,
        lambda_l1=0.8, lambda_structural=0.2,
        lr_pq=1e-3, lr_mean=5e-3, lr_L=5e-3,
        lr_opacity=5e-2, lr_color=2e-2,
        background=scene.background.to(DTYPE),
    )
    trainer = Trainer(
        model=model, cameras=scene.cameras,
        frame_data=frame_data, times=times,
        H=scene.H, W=scene.W, config=config,
    )
    trainer.train(num_iters=1500, log_every=100)

    # Loss curve
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(trainer.history["iter"], trainer.history["loss"], "o-", label="total")
    axes[0].plot(trainer.history["iter"], trainer.history["l1"], "s-", alpha=0.6, label="L1 only")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("loss")
    axes[0].set_yscale("log")
    axes[0].set_title("Training loss (log scale)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    # Validation grid:
    # Render cam 1 at t = 0.25 (an off-init time to test interpolation).
    cam_for_val = 1
    t_for_val = 0.25
    gt = frame_data[cam_for_val, times.index(t_for_val)].cpu().numpy().clip(0, 1)
    rec = render_current(model, scene.cameras[cam_for_val], t_for_val,
                         scene.H, scene.W, scene.background)
    # Compact validation subplot
    axes[1].axis("off")
    axes[1].set_title(f"Validation: cam {cam_for_val} @ t={t_for_val}")
    # Inset two images side by side
    fig.delaxes(axes[1])
    ax_gt = fig.add_subplot(1, 4, 3)
    ax_gt.imshow(gt); ax_gt.set_title("GT"); ax_gt.axis("off")
    ax_rec = fig.add_subplot(1, 4, 4)
    ax_rec.imshow(rec); ax_rec.set_title("Trained"); ax_rec.axis("off")
    fig.suptitle("Demo B: 3 cams × 5 frames training", fontsize=11)
    plt.tight_layout()
    plt.savefig("docs/images/phase5_multiview.png", dpi=110, bbox_inches="tight")
    plt.close()

    # Final full grid: GT top, trained bottom, for all K cameras at the middle time.
    t_mid = 0.5
    t_mid_idx = times.index(t_mid)
    fig, axes = plt.subplots(2, len(scene.cameras), figsize=(3.5 * len(scene.cameras), 4.5))
    for k in range(len(scene.cameras)):
        axes[0, k].imshow(frame_data[k, t_mid_idx].cpu().numpy().clip(0, 1))
        axes[0, k].set_title(f"GT cam {k}")
        axes[0, k].axis("off")
        rec_img = render_current(model, scene.cameras[k], t_mid,
                                 scene.H, scene.W, scene.background)
        axes[1, k].imshow(rec_img)
        axes[1, k].set_title(f"Trained cam {k}")
        axes[1, k].axis("off")
    fig.suptitle(f"Demo B final: all cameras @ t={t_mid}", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig("docs/images/phase5_multiview_grid.png", dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Final loss: {trainer.history['loss'][-1]:.4f}")
    print("  Saved phase5_multiview.png, phase5_multiview_grid.png")


if __name__ == "__main__":
    demo_overfit()
    demo_multiview_multiframe()
    print("\nAll Phase 5 visualizations saved in /home/claude/grassmann/")
