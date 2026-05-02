"""
Train the Grassmann model on the Neural 3D Video dataset (N3DV / DyNeRF).

DATASET LAYOUT EXPECTED (matches the official N3DV release after extraction):

    n3dv/flame_steak/
        cam00/
            images/
                0000.png
                0001.png
                ...
        cam01/
            images/
                ...
        ...
        cam20/
        cameras.json       # we'll produce this from the calibration file
        points3D.txt       # COLMAP output (sparse point cloud at frame 0)

You'll need to convert the dataset's calibration to a cameras.json file with
the format below. See the helper at the bottom of this script.

USAGE:

    # Step 1: convert calibration (one-time per scene)
    python scripts/train_n3dv.py prepare --scene_dir n3dv/flame_steak

    # Step 2: train
    python scripts/train_n3dv.py train \
        --scene_dir n3dv/flame_steak \
        --num_iters 30000 \
        --use_fast_rasterizer

GPU memory: at full resolution (1352x1014) with thousands of Gaussians, expect
~10 GB. Drop H/W (downscale_factor) if you OOM.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from grassmann.projection import Camera
from grassmann.initialization import init_gaussians_from_points
from grassmann.trainable import trainable_from_params
from grassmann.training import Trainer, TrainerConfig
from grassmann.density_control import DensityConfig


DTYPE = torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_cameras(cameras_json_path: Path) -> list[Camera]:
    """Load cameras from a JSON file we created in `prepare`.

    Each entry has: { 'R': 3x3 (world->cam), 'c': 3, 'fx', 'fy', 'cx', 'cy' }
    """
    with open(cameras_json_path) as f:
        cams_data = json.load(f)
    cams = []
    for d in cams_data:
        cams.append(Camera(
            R=torch.tensor(d["R"], dtype=DTYPE),
            c=torch.tensor(d["c"], dtype=DTYPE),
            fx=float(d["fx"]),
            fy=float(d["fy"]),
            cx=float(d["cx"]),
            cy=float(d["cy"]),
        ))
    return cams


def load_initial_points(scene_dir: Path, cameras: list, n_points: int = 30000) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate initial points in a tight box centered in front of the cameras.

    For N3DV, the cameras roughly look down +Z toward the cooking scene which
    sits ~2-5 meters in front. We sample uniformly in a small region there.
    """
    rng = np.random.default_rng(42)
    # Centroid of camera positions
    cam_centers = np.stack([cam.c.numpy() for cam in cameras], axis=0)
    centroid = cam_centers.mean(axis=0)
    # Average forward direction
    forwards = np.stack([cam.R[2].numpy() for cam in cameras], axis=0)
    avg_forward = forwards.mean(axis=0)
    avg_forward /= np.linalg.norm(avg_forward)
    # Scene center: 3 meters in front of camera centroid along avg forward
    scene_center = centroid + 3.0 * avg_forward
    # Sample in a 3m cube around scene_center
    points = scene_center[None, :] + rng.uniform(-1.5, 1.5, (n_points, 3))
    points_t = torch.tensor(points, dtype=torch.float64)
    colors_t = torch.full((n_points, 3), 0.5, dtype=torch.float64)

    print(f"Scene center estimate: {scene_center.tolist()}")
    print(f"Generated {n_points} initial points in a 3m cube around it")
    print(f"Point cloud extent: "
          f"x={points_t[:, 0].min():.2f}..{points_t[:, 0].max():.2f}, "
          f"y={points_t[:, 1].min():.2f}..{points_t[:, 1].max():.2f}, "
          f"z={points_t[:, 2].min():.2f}..{points_t[:, 2].max():.2f}")
    return points_t, colors_t


class N3DVFrameLoader:
    """Lazy frame loader: returns the (H, W, 3) image for a given (cam_idx, time_idx).

    Avoids loading all frames into RAM (would be 30+ GB for a single scene at full res).
    Caches a small LRU window for fast access in the random sampler.
    """

    def __init__(self, scene_dir: Path, n_cams: int, n_frames: int,
                 downscale_factor: int = 1):
        self.scene_dir = scene_dir
        self.n_cams = n_cams
        self.n_frames = n_frames
        self.downscale_factor = downscale_factor
        self._cache: dict[tuple[int, int], torch.Tensor] = {}
        self._cache_capacity = 64

    def __call__(self, cam_idx: int, t_value: float) -> torch.Tensor:
        # t_value is a frame index expressed as a float (we use float times in
        # the Trainer; for N3DV they map 1:1 to frame indices).
        t_idx = int(round(t_value))
        key = (cam_idx, t_idx)
        if key in self._cache:
            return self._cache[key]

        img_path = self.scene_dir / f"cam{cam_idx:02d}" / "images" / f"{t_idx:04d}.png"
        img = Image.open(img_path).convert("RGB")
        if self.downscale_factor > 1:
            img = img.resize(
                (img.width // self.downscale_factor, img.height // self.downscale_factor),
                Image.LANCZOS,
            )
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).to(dtype=DTYPE)

        # Simple FIFO cache eviction
        if len(self._cache) >= self._cache_capacity:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = tensor
        return tensor


def get_image_dims(scene_dir: Path, downscale_factor: int) -> tuple[int, int]:
    """Read a sample image to get H, W."""
    img_path = next((scene_dir / "cam00" / "images").glob("*.png"))
    img = Image.open(img_path)
    W, H = img.width, img.height
    return H // downscale_factor, W // downscale_factor


# =============================================================================
# TRAINING
# =============================================================================

def train(args):
    scene_dir = Path(args.scene_dir)
    cameras = load_cameras(scene_dir / "cameras.json")
    print(f"Loaded {len(cameras)} cameras")

    # Adjust intrinsics for downscaling
    if args.downscale_factor > 1:
        for cam in cameras:
            cam.fx /= args.downscale_factor
            cam.fy /= args.downscale_factor
            cam.cx /= args.downscale_factor
            cam.cy /= args.downscale_factor

    H, W = get_image_dims(scene_dir, args.downscale_factor)
    print(f"Image size: {H} x {W} (downscale_factor = {args.downscale_factor})")

    # Initial points from COLMAP
    # points, colors = load_initial_points(scene_dir / "points3D.txt")
    
    
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


    times = list(range(args.n_frames))

    model = trainable_from_params(params_init, dtype=DTYPE, device=DEVICE)
    print(f"Model on {DEVICE}, {model.N} Gaussians")

    # Lazy loader (don't load all frames at once)
    frame_data = N3DVFrameLoader(scene_dir, len(cameras), args.n_frames,
                                  args.downscale_factor)

    # Background: black (or your scene's average color)
    bg = torch.zeros(3, dtype=DTYPE, device=DEVICE)

    config = TrainerConfig(
        num_iters=args.num_iters,
        log_every=200,
        lambda_l1=0.8,
        lambda_structural=0.2,
        lr_pq=1e-3,
        lr_mean=5e-3,
        lr_L=5e-3,
        lr_opacity=5e-2,
        lr_color=5e-2,
        background=bg,
        # Density control: enable after warmup, run for ~half of training
        densify_every=500,
        densify_start=3000,
        densify_stop=args.num_iters // 2,
        density_config=DensityConfig(
            opacity_threshold=0.001,
            scale_min=1e-5,
            scale_max=2.0,
            grad_threshold=2e-4,
            clone_scale_threshold=0.05,
        ),
        # Fast rasterizer
        use_fast_rasterizer=args.use_fast_rasterizer,
    )
    trainer = Trainer(
        model=model, cameras=cameras,
        frame_data=frame_data, times=times,
        H=H, W=W, config=config,
    )

    print(f"Training for {args.num_iters} iterations...")
    trainer.train()

    # Save the trained model.
    out_dir = Path(args.output_dir) if args.output_dir else scene_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "trained_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": trainer.history,
    }, out_path)
    print(f"Saved to {out_path}")


# =============================================================================
# DATA PREPARATION (one-time, scene-by-scene)
# =============================================================================

def prepare(args):
    """Convert the N3DV calibration into the cameras.json format we expect.

    The N3DV release ships a `poses_bounds.npy` file (a [N_cams, 17] numpy array
    in LLFF / NeRF format). We convert it to our (R, c, fx, fy, cx, cy) format.

    Format of poses_bounds row (LLFF convention):
        [pose_3x5_flattened, near, far]
    where pose_3x5 is [R | t | hwf] in camera-to-world coords.
    Specifically: R is 3x3, t is 3x1, hwf is [height, width, focal] as a 3-vector.
    """
    scene_dir = Path(args.scene_dir)
    poses_path = scene_dir / "poses_bounds.npy"
    if not poses_path.exists():
        raise FileNotFoundError(
            f"Expected {poses_path} (N3DV's standard calibration file)."
        )
    poses_bounds = np.load(poses_path)
    n_cams = poses_bounds.shape[0]
    print(f"Loaded poses for {n_cams} cameras")

    cameras_json = []
    for i in range(n_cams):
        row = poses_bounds[i]
        pose_3x5 = row[:15].reshape(3, 5)
        R_c2w = pose_3x5[:, :3]   # camera-to-world rotation, 3x3
        t_c2w = pose_3x5[:, 3]    # camera position in world (== camera center c)
        h, w, f = pose_3x5[:, 4]  # height, width, focal length

        # We need world-to-camera: R_w2c = R_c2w.T
        R_w2c = R_c2w.T

        # Note: LLFF/NeRF uses a different camera convention than ours
        # (X-right Y-up Z-back vs our X-right Y-down Z-forward). We need to
        # flip Y and Z. This is equivalent to multiplying R_w2c by a fix-up matrix:
        flip = np.diag([1.0, -1.0, -1.0])
        R_w2c = flip @ R_w2c

        # Principal point: typically the image center for these datasets.
        cx, cy = w / 2.0, h / 2.0

        cameras_json.append({
            "R": R_w2c.tolist(),
            "c": t_c2w.tolist(),
            "fx": float(f),
            "fy": float(f),
            "cx": float(cx),
            "cy": float(cy),
        })

    out_path = scene_dir / "cameras.json"
    with open(out_path, "w") as f:
        json.dump(cameras_json, f, indent=2)
    print(f"Wrote {out_path}")
    print()
    print("Now you need a points3D.txt file (initial 3D point cloud).")
    print("Easiest approach: extract frame 0 of each camera and run COLMAP on them:")
    print()
    print("  mkdir colmap_input")
    print(f"  for d in {scene_dir}/cam*; do")
    print('      cp "$d/images/0000.png" "colmap_input/$(basename $d).png"')
    print("  done")
    print("  colmap automatic_reconstructor --workspace_path colmap_workspace \\")
    print("                                 --image_path colmap_input")
    print(f"  cp colmap_workspace/sparse/0/points3D.txt {scene_dir}/points3D.txt")
    print()
    print("If you already have COLMAP output for this scene, just point it at points3D.txt.")


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="Convert calibration to cameras.json")
    p_prep.add_argument("--scene_dir", required=True)
    p_prep.set_defaults(func=prepare)

    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--scene_dir", required=True)
    p_train.add_argument("--num_iters", type=int, default=30000)
    p_train.add_argument("--n_frames", type=int, default=300)
    p_train.add_argument("--downscale_factor", type=int, default=2,
                         help="2 -> half resolution. Use 4 for low memory.")
    p_train.add_argument("--max_initial_points", type=int, default=100_000)
    p_train.add_argument("--use_fast_rasterizer", action="store_true",
                         help="Use diff-gaussian-rasterization (CUDA only).")
    p_train.add_argument("--output_dir", default=None,
                         help="Where to save trained_model.pt. Default: scene_dir.")
    p_train.set_defaults(func=train)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
