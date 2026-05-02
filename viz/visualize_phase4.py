"""Phase 4 end-to-end visualization.

Shows the full pipeline in action:
  1. Build a synthetic multi-camera scene with colored moving points.
  2. Render ground-truth views from all K cameras at time t.
  3. Extract pixel observations of each scene point in each camera.
  4. Triangulate each point back to 3D using DLT.
  5. Initialize Grassmann Gaussians from the triangulated 3D points.
  6. Render the reconstructed scene through the Grassmann rasterizer.
  7. Compare ground-truth vs reconstructed side-by-side.

Produces:
  - phase4_pipeline.png: grid showing GT (top) and reconstructed (bottom) views.
  - phase4_triangulation.png: scatter plot of true vs triangulated 3D points.
  - phase4_timelapse.png: GT vs reconstructed videos side-by-side over time.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from grassmann import quaternion as Q
from grassmann.gaussian import compute_derived, condition_on_time
from grassmann.rasterizer import project_to_screen, rasterize
from grassmann.synthetic import make_default_scene, render_synthetic_frame
from grassmann.triangulation import (
    observe_scene_point, triangulate_point_dlt, reprojection_error,
)
from grassmann.initialization import init_gaussians_from_points


DTYPE = torch.float64


def observe_all_points_at_time(scene, t, add_noise=0.0):
    """For each scene point, get (K, 2) pixel observations in each camera, plus visibility mask."""
    obs_list = []
    depth_list = []
    colors_list = []
    for pt in scene.scene_points:
        uvs, depths = observe_scene_point(pt.trajectory, t, scene.cameras, add_noise_std=add_noise)
        obs_list.append(uvs)
        depth_list.append(depths)
        colors_list.append(pt.color)
    return obs_list, depth_list, colors_list


def triangulate_scene(scene, t, add_noise=0.0):
    """Triangulate all scene points at time t. Returns (points, colors, true_points)."""
    obs_list, depth_list, colors_list = observe_all_points_at_time(scene, t, add_noise)

    points_rec = []
    points_true = []
    colors_out = []
    for i, (uvs, depths) in enumerate(zip(obs_list, depth_list)):
        visible = depths > 0.1
        if visible.sum() < 2:
            continue
        visible_cams = [scene.cameras[k] for k in range(len(scene.cameras)) if visible[k]]
        X_rec = triangulate_point_dlt(visible_cams, uvs[visible])
        points_rec.append(X_rec)
        points_true.append(scene.scene_points[i].trajectory(t))
        colors_out.append(colors_list[i])

    return (
        torch.stack(points_rec) if points_rec else torch.zeros(0, 3, dtype=DTYPE),
        torch.stack(colors_out) if colors_out else torch.zeros(0, 3, dtype=DTYPE),
        torch.stack(points_true) if points_true else torch.zeros(0, 3, dtype=DTYPE),
    )


def render_reconstruction(params, t, scene):
    """Render the Grassmann reconstruction from each camera at time t."""
    derived = compute_derived(params)
    tc = condition_on_time(params, derived, t)
    images = []
    for cam in scene.cameras:
        sg = project_to_screen(params, tc, cam)
        img = rasterize(sg, H=scene.H, W=scene.W, background=scene.background)
        images.append(img.numpy().clip(0, 1))
    return images


# ----------------------------------------------------------------------------
# Demo 1: Pipeline at a single time instant
# ----------------------------------------------------------------------------

def viz_pipeline_single_time():
    """Ground truth vs reconstruction at a single time, all K cameras."""
    scene = make_default_scene(n_cams=4, image_w=200, image_h=120)
    t = 0.5

    # GT frames
    gt_frames = [render_synthetic_frame(scene, k, t, blob_sigma=3.0).numpy()
                 for k in range(len(scene.cameras))]

    # Triangulate and init
    points_rec, colors, _ = triangulate_scene(scene, t)
    times_tensor = torch.full((points_rec.shape[0],), t, dtype=DTYPE)
    params = init_gaussians_from_points(
        points_rec, times_tensor, scene.cameras,
        colors=colors, sigma_aa=0.02, sigma_bb=0.2, opacity=0.95, sigma_k=3.0,
    )
    rec_frames = render_reconstruction(params, t, scene)

    # Plot grid: rows = {GT, Reconstructed}, cols = K cameras
    K = len(scene.cameras)
    fig, axes = plt.subplots(2, K, figsize=(3.5 * K, 4))
    for k in range(K):
        axes[0, k].imshow(gt_frames[k])
        axes[0, k].set_title(f"Camera {k} — GT")
        axes[0, k].axis("off")
        axes[1, k].imshow(rec_frames[k])
        axes[1, k].set_title(f"Camera {k} — Reconstructed")
        axes[1, k].axis("off")
    fig.suptitle(
        f"Phase 4: End-to-end pipeline at t={t}\n"
        f"Top: ground-truth synthetic scene.  Bottom: Grassmann reconstruction from triangulated points.",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig("docs/images/phase4_pipeline.png", dpi=110, bbox_inches="tight")
    plt.close()
    print("  Saved phase4_pipeline.png")


# ----------------------------------------------------------------------------
# Demo 2: Triangulation accuracy (with noise)
# ----------------------------------------------------------------------------

def viz_triangulation_accuracy():
    """Scatter plot: true vs triangulated 3D points, with varying noise levels."""
    scene = make_default_scene(n_cams=4, image_w=200, image_h=120)
    t = 0.5

    # Sample many noise levels, many trials per level.
    noise_levels = [0.0, 0.5, 1.0, 2.0]
    trials_per_level = 20

    fig, axes = plt.subplots(1, len(noise_levels), figsize=(4 * len(noise_levels), 4))
    for ax, noise in zip(axes, noise_levels):
        errs = []
        rec_pts = []
        true_pts = []
        torch.manual_seed(0)
        for _ in range(trials_per_level):
            rec, _, true = triangulate_scene(scene, t, add_noise=noise)
            for i in range(rec.shape[0]):
                e = (rec[i] - true[i]).norm().item()
                errs.append(e)
                rec_pts.append(rec[i].numpy())
                true_pts.append(true[i].numpy())

        rec_arr = np.array(rec_pts)
        true_arr = np.array(true_pts)

        # Plot true (gray circles) and reconstructed (colored by err) in x-z plane
        ax.scatter(true_arr[:, 0], true_arr[:, 2], c="lightgray", s=80,
                   edgecolors="black", label="true", zorder=2)
        sc = ax.scatter(rec_arr[:, 0], rec_arr[:, 2], c=errs, cmap="viridis",
                        s=20, alpha=0.7, zorder=3, label="reconstructed")
        ax.set_xlabel("world x")
        ax.set_ylabel("world z")
        ax.set_title(f"Pixel noise σ = {noise}\nMean err = {np.mean(errs):.4f}")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        plt.colorbar(sc, ax=ax, label="L2 error")
    fig.suptitle(
        "Phase 4: Triangulation accuracy vs pixel observation noise (K=4 cameras)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig("docs/images/phase4_triangulation.png", dpi=110, bbox_inches="tight")
    plt.close()
    print("  Saved phase4_triangulation.png")


# ----------------------------------------------------------------------------
# Demo 3: Time-lapse comparison (GT vs reconstruction over time)
# ----------------------------------------------------------------------------

def viz_timelapse():
    """GT vs reconstruction at multiple time instants, from one camera."""
    scene = make_default_scene(n_cams=4, image_w=200, image_h=120)
    ts = np.linspace(0.0, 1.0, 5)

    cam_idx = 0

    fig, axes = plt.subplots(2, len(ts), figsize=(3.0 * len(ts), 4))
    for col, t in enumerate(ts):
        # GT
        gt = render_synthetic_frame(scene, cam_idx, float(t), blob_sigma=3.0).numpy()
        axes[0, col].imshow(gt)
        axes[0, col].set_title(f"GT t={t:.2f}")
        axes[0, col].axis("off")

        # Triangulate + init + render
        points_rec, colors, _ = triangulate_scene(scene, float(t))
        times_tensor = torch.full((points_rec.shape[0],), float(t), dtype=DTYPE)
        params = init_gaussians_from_points(
            points_rec, times_tensor, scene.cameras,
            colors=colors, sigma_aa=0.02, sigma_bb=0.2, opacity=0.95, sigma_k=3.0,
        )
        rec = render_reconstruction(params, float(t), scene)[cam_idx]
        axes[1, col].imshow(rec)
        axes[1, col].set_title(f"Rec t={t:.2f}")
        axes[1, col].axis("off")
    fig.suptitle(
        f"Phase 4 time-lapse from camera {cam_idx}: GT (top) vs frame-by-frame reconstruction (bottom)\n"
        "(Each frame is independently triangulated and reinitialized — Phase 5 training will link them.)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig("docs/images/phase4_timelapse.png", dpi=110, bbox_inches="tight")
    plt.close()
    print("  Saved phase4_timelapse.png")


if __name__ == "__main__":
    viz_pipeline_single_time()
    viz_triangulation_accuracy()
    viz_timelapse()
    print("\nAll Phase 4 visualizations saved in /home/claude/grassmann/")
