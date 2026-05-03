"""
Single-Gaussian sanity test.

Purpose: bypass training entirely. Place exactly ONE red Gaussian at the
estimated scene center, then render it from every camera. If the rendering
pipeline is correct, the red blob should appear in roughly the middle of every
frame, since all cameras look at the scene center.

Outputs PNGs `sanity_camNN.png` for each camera. Open them and check:

    1. Does each render show a red blob (not black, not green, not blue)?
    2. Is the blob in roughly the center of the frame?
    3. Is it the right size (not tiny, not filling the whole frame)?
    4. Are the blobs in CONSISTENT positions across cameras?
       (Cameras at different angles should see the blob shifted along an
        epipolar line in a predictable way.)

Run from your grassmann/ directory:
    python scripts/sanity_one_gaussian.py --scene_dir data/n3dv/flame_steak/
"""
import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# scripts/ also needs to be importable for the `from train_n3dv import ...` line.
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import numpy as np
import torch
from PIL import Image

from grassmann import quaternion as Q
from grassmann import grassmann as G
from grassmann.gaussian import GaussianParams
from grassmann.fast_rasterizer import fast_rasterize, is_available as fast_available
from grassmann.rasterizer import project_to_screen, rasterize as toy_rasterize
from grassmann.gaussian import compute_derived, condition_on_time
from train_n3dv import load_cameras


DTYPE = torch.float32


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir", required=True)
    p.add_argument("--downscale_factor", type=int, default=4)
    p.add_argument("--depth", type=float, default=3.0,
                   help="Depth in front of mean camera position to place the Gaussian.")
    p.add_argument("--out_dir", default="sanity_out")
    args = p.parse_args()

    scene_dir = Path(args.scene_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    cameras = load_cameras(scene_dir / "cameras.json")
    if args.downscale_factor > 1:
        for cam in cameras:
            cam.fx /= args.downscale_factor
            cam.fy /= args.downscale_factor
            cam.cx /= args.downscale_factor
            cam.cy /= args.downscale_factor

    H = int(2 * cameras[0].cy)
    W = int(2 * cameras[0].cx)
    print(f"Image size: {H} x {W}")
    print(f"Cameras: {len(cameras)}")

    # Compute scene center
    cam_centers = np.stack([cam.c.numpy() for cam in cameras], axis=0)
    forwards = np.stack([cam.R[2].numpy() for cam in cameras], axis=0)
    centroid = cam_centers.mean(axis=0)
    avg_forward = forwards.mean(axis=0)
    avg_forward /= np.linalg.norm(avg_forward)
    scene_center = centroid + args.depth * avg_forward
    print(f"Scene center: {scene_center.tolist()}")

    # ============================================================
    # Build a single Gaussian by hand at the scene center.
    # We use the formula from grassmann/initialization.py but with explicit
    # values so nothing surprising can happen.
    # ============================================================
    X = torch.tensor(scene_center, dtype=torch.float64).unsqueeze(0)        # (1, 3)
    # Use camera 0 as the reference for the line.
    ref_cam = cameras[0]
    dir_world = X[0] - ref_cam.c.to(torch.float64)                          # (3,)
    u_hat = dir_world / dir_world.norm()                                     # (3,)
    t_target = 1.0   # arbitrary; using 1.0 so no t-scaling needed

    # Build line through ref camera in direction u_hat
    p_quat, q_quat = G.line_to_pq(
        ref_cam.c.to(torch.float64).unsqueeze(0),
        u_hat.unsqueeze(0),
    )
    e1_hat, e2_hat = G.orthonormal_basis(p_quat, q_quat)
    target = torch.cat([
        torch.tensor([[t_target]], dtype=torch.float64),
        X,
    ], dim=-1)                                                              # (1, 4)
    alpha_val = (target * e1_hat).sum(dim=-1)
    beta_val = (target * e2_hat).sum(dim=-1)

    # Big Gaussian: 50cm spatial, opacity 1.0, bright red.
    sigma_aa = 0.001       # ~50cm spatial std-dev (sqrt(0.25) = 0.5m)
    sigma_bb = 0.01
    sigma_ab = 0.0
    L = torch.tensor([[
        [np.sqrt(sigma_aa), 0.0],
        [sigma_ab / np.sqrt(sigma_aa), np.sqrt(sigma_bb - sigma_ab ** 2 / sigma_aa)],
    ]], dtype=torch.float64)

    params = GaussianParams(
        p_im=Q.imag(p_quat),
        q_im=Q.imag(q_quat),
        alpha_0=alpha_val,
        beta_0=beta_val,
        L=L,
        opacity=torch.tensor([0.99], dtype=torch.float64),
        color=torch.tensor([[1.0, 0.1, 0.1]], dtype=torch.float64),    # red
        sigma_k_pixel=1.0,
        sigma_k_temporal=1.0,
    )
    derived = compute_derived(params)
    print(f"\nSingle Gaussian:")
    print(f"  V_k (world):    {derived.V_k[0].tolist()}")
    print(f"  v_0 (time):     {derived.v_0[0].item():.3f}")
    print(f"  Sigma_3D eigvals: {torch.linalg.eigvalsh(derived.Sigma_3D[0]).tolist()}")
    err = (derived.V_k[0] - X[0]).norm().item()
    print(f"  Mean placement error vs target: {err:.4f}m")

    # ============================================================
    # Render from each camera using the toy rasterizer (CPU-safe, no fast path)
    # ============================================================
    print(f"\nRendering from {len(cameras)} cameras using toy rasterizer...")

    # Cast params to float32 for rendering speed
    params_f32 = GaussianParams(
        p_im=params.p_im.to(DTYPE),
        q_im=params.q_im.to(DTYPE),
        alpha_0=params.alpha_0.to(DTYPE),
        beta_0=params.beta_0.to(DTYPE),
        L=params.L.to(DTYPE),
        opacity=params.opacity.to(DTYPE),
        color=params.color.to(DTYPE),
        sigma_k_pixel=params.sigma_k_pixel,
        sigma_k_temporal=params.sigma_k_temporal,
    )
    derived_f32 = compute_derived(params_f32)
    tc = condition_on_time(params_f32, derived_f32, float(t_target))

    bg = torch.tensor([0.0, 0.0, 0.2], dtype=DTYPE)   # dark blue background to make red obvious

    print()
    for k, cam in enumerate(cameras):
        # Project the mean to see where it SHOULD appear
        cam_f32 = type(cam)(
            R=cam.R.to(DTYPE), c=cam.c.to(DTYPE),
            fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy,
        )
        X_cam = cam_f32.R @ (derived_f32.V_k[0] - cam_f32.c)
        if X_cam[2] < 0.01:
            print(f"  cam{k:02d}: scene center is BEHIND camera (depth={X_cam[2]:.3f})")
            continue
        u_pred = (cam_f32.fx * X_cam[0] / X_cam[2] + cam_f32.cx).item()
        v_pred = (cam_f32.fy * X_cam[1] / X_cam[2] + cam_f32.cy).item()
        in_bounds = (0 <= u_pred < W) and (0 <= v_pred < H)
        marker = "✓" if in_bounds else "✗"

        # Render
        sg = project_to_screen(params_f32, tc, cam_f32)
        img = toy_rasterize(sg, H=H, W=W, background=bg)
        img_np = (img.detach().cpu().clamp(0, 1).numpy() * 255).astype("uint8")
        Image.fromarray(img_np).save(out_dir / f"sanity_cam{k:02d}.png")

        rendered_max = img.max().item()
          # Find brightest pixel (using just the red channel since the Gaussian is red)
        red = img[..., 0]
        if red.max().item() > 0.05:
            flat_idx = red.flatten().argmax().item()
            actual_v = flat_idx // W
            actual_u = flat_idx % W
        else:
            actual_u, actual_v = -1, -1

        print(f"  cam{k:02d}: predicted pixel ({u_pred:6.1f}, {v_pred:6.1f}) {marker}, "
              f"render_max={rendered_max:.3f}, brightest_pixel=({actual_u}, {actual_v})")

    print(f"\nWrote {len(cameras)} PNGs to {out_dir}/")
    print("Open them and check:")
    print("  1. Each has a RED BLOB on a dark blue background (not all black, not gray)")
    print("  2. The blob is at the predicted pixel location (within ~10 pixels)")
    print("  3. The blob has a sensible size (not 1px, not full-frame)")


if __name__ == "__main__":
    main()
