"""
Diagnostic training: runs ONLY 100 iters, prints per-step diagnostics so we
can see EXACTLY when and why the model collapses.

Run from your grassmann/ directory:
    python diagnose_n3dv.py --scene_dir data/n3dv/flame_steak/

This script does the same setup as train_n3dv.py train but with much heavier
logging: per-iteration loss, per-iteration N, mean opacity, gradient norms,
and what fraction of Gaussians are visible (have non-trivial opacity).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from grassmann.projection import Camera
from grassmann.gaussian import compute_derived, condition_on_time
from grassmann.rasterizer import project_to_screen, rasterize
from grassmann.fast_rasterizer import fast_rasterize, is_available as fast_available
from grassmann.initialization import init_gaussians_from_points
from grassmann.trainable import trainable_from_params, build_optimizer
from grassmann.losses import l1_loss, photometric_loss

from train_n3dv import load_cameras, load_initial_points, get_image_dims, N3DVFrameLoader


DTYPE = torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir", required=True)
    p.add_argument("--downscale_factor", type=int, default=4)
    p.add_argument("--n_frames", type=int, default=300)
    p.add_argument("--num_iters", type=int, default=100)
    p.add_argument("--max_initial_points", type=int, default=20_000)
    args = p.parse_args()

    scene_dir = Path(args.scene_dir)
    cameras = load_cameras(scene_dir / "cameras.json")
    print(f"Loaded {len(cameras)} cameras")

    if args.downscale_factor > 1:
        for cam in cameras:
            cam.fx /= args.downscale_factor
            cam.fy /= args.downscale_factor
            cam.cx /= args.downscale_factor
            cam.cy /= args.downscale_factor

    H, W = get_image_dims(scene_dir, args.downscale_factor)
    print(f"Image size: {H} x {W}")

    points, colors = load_initial_points(scene_dir, cameras, n_points=30000)
    print(f"Initialized with {points.shape[0]} points")

    # Compute per-point distance to nearest neighbor (k=3 average), using a simple
    # pairwise distance computation on a CPU subset.
    print("Computing per-point local-density scales...")
    points_np = points.numpy()
    N = points_np.shape[0]
    # Use scipy's KDTree if available; otherwise a chunked numpy fallback
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(points_np)
        # Query 4 nearest (the first is itself)
        dists, _ = tree.query(points_np, k=4)
        nn_dist = dists[:, 1:].mean(axis=1)   # mean of 3 nearest neighbors
    except ImportError:
        print("  scipy not available, using uniform scale (less optimal)")
        nn_dist = np.full(N, 0.05)

    # Cap at sensible bounds to avoid huge or tiny Gaussians
    nn_dist = np.clip(nn_dist, 0.01, 0.3)
    print(f"  per-point NN dist: mean={nn_dist.mean():.3f}, "
        f"min={nn_dist.min():.3f}, max={nn_dist.max():.3f}")

    # `sigma_aa` should be ~ (NN distance)^2 for a Gaussian with std-dev = NN dist.
    # But this is a per-Gaussian value; init_gaussians_from_points takes a scalar.
    # We need to call it ONCE PER GAUSSIAN — slow but only at init.
    times_init = torch.linspace(0.0, float(args.n_frames - 1), points.shape[0], dtype=torch.float64)

    from grassmann.initialization import init_gaussian_from_point
    print("Initializing per-point Gaussians (this may take a minute)...")
    all_params = []
    for i in range(N):
        sigma_aa_i = float(nn_dist[i] ** 2)   # squared NN distance
        g = init_gaussian_from_point(
            points[i], float(times_init[i].item()), cameras,
            color=colors[i],
            sigma_aa=sigma_aa_i,
            sigma_bb=0.05,
            sigma_ab=0.0,
            opacity=0.3,
            sigma_k=20.0,
        )
        all_params.append(g)

    # Concatenate manually
    from grassmann.gaussian import GaussianParams
    params_init = GaussianParams(
        p_im=torch.cat([g.p_im for g in all_params]),
        q_im=torch.cat([g.q_im for g in all_params]),
        alpha_0=torch.cat([g.alpha_0 for g in all_params]),
        beta_0=torch.cat([g.beta_0 for g in all_params]),
        L=torch.cat([g.L for g in all_params]),
        opacity=torch.cat([g.opacity for g in all_params]),
        color=torch.cat([g.color for g in all_params]),
        sigma_k=20.0,
    )
    print(f"Built {params_init.p_im.shape[0]} Gaussians with per-point scales")


    model = trainable_from_params(params_init, dtype=DTYPE, device=DEVICE)
    print(f"\nModel: {model.N} Gaussians on {DEVICE}")

    # Initial render diagnostic — render one camera/frame BEFORE any training
    print("\n=== Initial render check ===")
    params = model.forward()
    derived = compute_derived(params)
    bg = torch.zeros(3, dtype=DTYPE, device=DEVICE)
    img0 = fast_rasterize(params, t_0=float(args.n_frames // 2), cam=cameras[0],
                          H=H, W=W, background=bg)
    print(f"  Render shape:  {img0.shape}")
    print(f"  Render range:  {img0.min().item():.4f} .. {img0.max().item():.4f}")
    print(f"  Render mean:   {img0.mean().item():.4f}")
    Image.fromarray((img0.detach().cpu().clamp(0, 1).numpy() * 255).astype("uint8")).save(
        "diagnose_render_iter0.png")
    print("  Saved diagnose_render_iter0.png")

    # Compare to target
    loader = N3DVFrameLoader(scene_dir, len(cameras), args.n_frames, args.downscale_factor)
    target0 = loader(0, float(args.n_frames // 2))
    print(f"  Target range:  {target0.min().item():.4f} .. {target0.max().item():.4f}")
    print(f"  Target mean:   {target0.mean().item():.4f}")
    Image.fromarray((target0.numpy() * 255).astype("uint8")).save("diagnose_target.png")
    print("  Saved diagnose_target.png")

    # Now run a few training iterations and watch what happens
    print("\n=== Training loop diagnostic (no density control) ===")
    optimizer = build_optimizer(model, lr_pq=1e-3, lr_mean=5e-3, lr_L=5e-3,
                                 lr_opacity=5e-2, lr_color=5e-2)

    times_list = list(range(args.n_frames))
    for it in range(1, args.num_iters + 1):
        cam_idx = torch.randint(0, len(cameras), (1,)).item()
        t_idx = torch.randint(0, args.n_frames, (1,)).item()
        t_value = float(times_list[t_idx])

        params = model.forward()
        bg = torch.zeros(3, dtype=DTYPE, device=DEVICE)
        rendered = fast_rasterize(params, t_0=t_value, cam=cameras[cam_idx],
                                   H=H, W=W, background=bg)
        target = loader(cam_idx, t_value).to(DEVICE).to(DTYPE)

        loss = photometric_loss(rendered, target, lambda_l1=0.8, lambda_structural=0.2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Periodic deep diagnostics
        if it == 1 or it % 10 == 0:
            with torch.no_grad():
                opacities = torch.sigmoid(model.opacity_logit)
                grads = {}
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        grads[name] = p.grad.norm().item()
                # Render diagnostic
                rendered_mean = rendered.detach().mean().item()
                rendered_max = rendered.detach().max().item()
                visible_frac = (opacities > 0.01).float().mean().item()

                print(f"  iter {it:4d}: loss={loss.item():.4f}  "
                      f"render(mean={rendered_mean:.3f}, max={rendered_max:.3f})  "
                      f"opacity(min={opacities.min():.4f}, mean={opacities.mean():.4f}, max={opacities.max():.4f})  "
                      f"visible={visible_frac:.2%}  "
                      f"grad_op={grads.get('opacity_logit', 0):.2e}  "
                      f"grad_color={grads.get('color_logit', 0):.2e}  "
                      f"grad_alpha={grads.get('alpha_0', 0):.2e}")

    # Final render
    print("\n=== Final render check ===")
    params = model.forward()
    img_final = fast_rasterize(params, t_0=float(args.n_frames // 2), cam=cameras[0],
                                H=H, W=W, background=bg)
    Image.fromarray((img_final.detach().cpu().clamp(0, 1).numpy() * 255).astype("uint8")).save(
        f"diagnose_render_iter{args.num_iters}.png")
    print(f"  Saved diagnose_render_iter{args.num_iters}.png")
    print(f"  Final render: mean={img_final.mean().item():.4f}, max={img_final.max().item():.4f}")


if __name__ == "__main__":
    main()
