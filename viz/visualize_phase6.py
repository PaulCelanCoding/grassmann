"""Phase 6 visualization.

The showcase: a scene initialized with FEWER Gaussians than needed. We train
twice:
  - Baseline (Phase 5): fixed 15 Gaussians, no density control.
  - With density control: start with 5, let the tracker clone/split/prune.

Compare loss curves, final Gaussian count, and rendered output.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from grassmann import quaternion as Q
from grassmann.gaussian import compute_derived, condition_on_time
from grassmann.rasterizer import project_to_screen, rasterize
from grassmann.initialization import init_gaussians_from_points
from grassmann.synthetic import make_default_scene, render_synthetic_frame
from grassmann.triangulation import observe_scene_point, triangulate_point_dlt
from grassmann.trainable import trainable_from_params
from grassmann.training import Trainer, TrainerConfig
from grassmann.density_control import DensityConfig


DTYPE = torch.float32


def render_current(model, cam, t, H, W, background):
    params = model.forward()
    derived = compute_derived(params)
    tc = condition_on_time(params, derived, t)
    sg = project_to_screen(params, tc, cam)
    bg = background.to(params.color.dtype)
    return rasterize(sg, H=H, W=W, background=bg).detach().cpu().numpy().clip(0, 1)


def triangulate_points(scene, t):
    pts, cols = [], []
    for sp in scene.scene_points:
        uvs, depths = observe_scene_point(sp.trajectory, t, scene.cameras)
        visible = depths > 0.1
        if visible.sum() < 2:
            continue
        visible_cams = [scene.cameras[k] for k in range(len(scene.cameras)) if visible[k]]
        X_rec = triangulate_point_dlt(visible_cams, uvs[visible])
        pts.append(X_rec)
        cols.append(sp.color)
    return torch.stack(pts), torch.stack(cols)


def build_training(scene, n_init_gaussians, densify=False):
    """Set up a trainer with either very few Gaussians (for density to grow)
    or full initialization (for baseline)."""
    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    frame_data = torch.stack([
        torch.stack([render_synthetic_frame(scene, k, t, blob_sigma=3.0) for t in times])
        for k in range(len(scene.cameras))
    ]).to(DTYPE)

    # Initialize Gaussians. If n_init_gaussians < 15, only pick some.
    all_pts, all_times, all_cols = [], [], []
    for t in times:
        pts, cols = triangulate_points(scene, t)
        all_pts.append(pts)
        all_times.append(torch.full((pts.shape[0],), t, dtype=torch.float64))
        all_cols.append(cols)
    pts_full = torch.cat(all_pts, dim=0)
    times_full = torch.cat(all_times, dim=0)
    cols_full = torch.cat(all_cols, dim=0)
    # Under-init: keep only the first n_init_gaussians rows.
    pts_init = pts_full[:n_init_gaussians]
    times_init = times_full[:n_init_gaussians]
    cols_init = cols_full[:n_init_gaussians]

    params_init = init_gaussians_from_points(
        pts_init, times_init, scene.cameras, colors=cols_init,
        sigma_aa=0.02, sigma_bb=0.15, opacity=0.5, sigma_k=3.0,
    )
    model = trainable_from_params(params_init, dtype=DTYPE)

    config = TrainerConfig(
        num_iters=800, log_every=100,
        lambda_l1=0.8, lambda_structural=0.2,
        lr_pq=1e-3, lr_mean=5e-3, lr_L=5e-3,
        lr_opacity=5e-2, lr_color=2e-2,
        background=scene.background.to(DTYPE),
        densify_every=150 if densify else 0,
        densify_start=200,
        densify_stop=700,
        density_config=DensityConfig(
            grad_threshold=1e-4,
            opacity_threshold=0.01,
            scale_min=1e-5,
            scale_max=2.0,
            clone_scale_threshold=0.08,
        ) if densify else None,
    )
    trainer = Trainer(
        model=model, cameras=scene.cameras,
        frame_data=frame_data, times=times,
        H=scene.H, W=scene.W, config=config,
    )
    return trainer, frame_data, times


def demo_density_control():
    print("--- Phase 6: Training with vs without density control ---")
    scene = make_default_scene(n_cams=3, image_w=60, image_h=45)

    # Run 1: baseline, plenty of Gaussians.
    print("\n[Baseline] 15 Gaussians, no density control")
    trainer_base, _, _ = build_training(scene, n_init_gaussians=15, densify=False)
    trainer_base.train(num_iters=800, log_every=100)
    print(f"  Final: N={trainer_base.model.N}, loss={trainer_base.history['loss'][-1]:.4f}")

    # Run 2: under-init, with density control.
    print("\n[Density control] 5 initial Gaussians, clone/split allowed")
    trainer_dc, frame_data, times = build_training(scene, n_init_gaussians=5, densify=True)
    trainer_dc.train(num_iters=800, log_every=100)
    print(f"  Final: N={trainer_dc.model.N}, loss={trainer_dc.history['loss'][-1]:.4f}")

    # Plot loss curves + N over time.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(trainer_base.history["iter"], trainer_base.history["loss"],
                 "o-", label="Baseline (15 fixed)", color="C0")
    axes[0].plot(trainer_dc.history["iter"], trainer_dc.history["loss"],
                 "s-", label="Density control (5 → growing)", color="C1")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("loss")
    axes[0].set_yscale("log")
    axes[0].set_title("Training loss (log scale)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(trainer_base.history["iter"], trainer_base.history["N"],
                 "o-", label="Baseline", color="C0")
    axes[1].plot(trainer_dc.history["iter"], trainer_dc.history["N"],
                 "s-", label="Density control", color="C1")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("Number of Gaussians")
    axes[1].set_title("Gaussian count over training")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle("Phase 6: Density control grows the model as needed", fontsize=11)
    plt.tight_layout()
    plt.savefig("docs/images/phase6_curves.png", dpi=110, bbox_inches="tight")
    plt.close()
    print("  Saved phase6_curves.png")

    # Grid: GT vs baseline vs density-control, per camera at t=0.5
    t_mid = 0.5
    t_mid_idx = times.index(t_mid)
    n_cams = len(scene.cameras)
    fig, axes = plt.subplots(3, n_cams, figsize=(3.5 * n_cams, 6.5))
    for k in range(n_cams):
        axes[0, k].imshow(frame_data[k, t_mid_idx].cpu().numpy().clip(0, 1))
        axes[0, k].set_title(f"GT cam {k}")
        axes[0, k].axis("off")
        axes[1, k].imshow(render_current(trainer_base.model, scene.cameras[k],
                                          t_mid, scene.H, scene.W, scene.background))
        axes[1, k].set_title(f"Baseline ({trainer_base.model.N} G)")
        axes[1, k].axis("off")
        axes[2, k].imshow(render_current(trainer_dc.model, scene.cameras[k],
                                          t_mid, scene.H, scene.W, scene.background))
        axes[2, k].set_title(f"Density control ({trainer_dc.model.N} G)")
        axes[2, k].axis("off")
    fig.suptitle(f"Phase 6 final renders @ t={t_mid}", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig("docs/images/phase6_grid.png", dpi=110, bbox_inches="tight")
    plt.close()
    print("  Saved phase6_grid.png")


if __name__ == "__main__":
    demo_density_control()
    print("\nPhase 6 visualizations saved in /home/claude/grassmann/")
