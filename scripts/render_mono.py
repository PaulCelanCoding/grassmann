"""
Render frames from a saved monocular Grassmann checkpoint.

Loads a NeRFies/DyCheck scene + a checkpoint produced by train_mono.py, then
renders one or more specified frames and saves PNGs to --output_dir.

Defaults to the CUDA fast rasterizer (diff-gaussian-rasterization) when the
device is cuda; falls back to the toy rasterizer on cpu (slow, smoke-only).

Intended invocation: from inside Modal (see scripts/train_modal.py --cmd render).
Local CPU runs work but are too slow for inspecting full scenes.

Usage:
    python scripts/render_mono.py \\
        --dataset nerfies \\
        --scene_dir /data/slice-banana \\
        --ckpt /checkpoints/<run>/trained_nerfies_median.pt \\
        --frames 0,50,100 \\
        --output_dir /checkpoints/<run>/renders \\
        --device cuda \\
        --side_by_side \\
        --allow_distortion
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from grassmann.datasets.dycheck import load_dycheck
from grassmann.datasets.nerfies import load_nerfies
from grassmann.fast_rasterizer import FastRasterConfig, fast_rasterize
from grassmann.surfel_rasterizer import SurfelRasterConfig, surfel_rasterize
from grassmann.initialization import init_gaussians_from_points
from grassmann.trainable import trainable_from_params


DTYPE = torch.float32


def _load_dataset(name, scene_dir, image_scale, split, allow_distortion):
    if name == "nerfies":
        return load_nerfies(scene_dir, image_scale=image_scale,
                             allow_distortion=allow_distortion)
    if name == "dycheck":
        return load_dycheck(scene_dir, image_scale=image_scale, split_name=split,
                             allow_distortion=allow_distortion)
    raise ValueError(name)


def _parse_frames(spec: str, T: int) -> list[int]:
    """'0,50,100' -> [0,50,100]; 'all' -> [0..T-1]; 'every:N' -> [0,N,2N,...]."""
    spec = spec.strip()
    if spec == "all":
        return list(range(T))
    if spec.startswith("every:"):
        step = int(spec.split(":", 1)[1])
        return list(range(0, T, step))
    return [int(x) for x in spec.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("nerfies", "dycheck"), required=True)
    ap.add_argument("--scene_dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--frames", type=str, default="0",
                    help="Comma list ('0,50,100'), 'all', or 'every:N'.")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--image_scale", type=int, default=4)
    ap.add_argument("--split", type=str, default=None)
    ap.add_argument("--allow_distortion", action="store_true")
    ap.add_argument("--device", type=str, default=None,
                    help="cuda | cpu. Defaults to cuda if available.")
    ap.add_argument("--side_by_side", action="store_true",
                    help="Also save GT|render side-by-side as sbs_NNNN.png.")
    ap.add_argument("--sigma_3d_blur", type=float, default=1e-4,
                    help="Isotropic numerical lift on Σ_3D(t_0) (rank-2 disk under the "
                         "3-plane parameterization) before the CUDA rasterizer. ε≈1e-4 "
                         "in scene units is enough for invertibility; larger values "
                         "become a meaningful blur.")
    ap.add_argument("--rasterizer", choices=("gaussian", "surfel"), default="gaussian",
                    help="Match the rasterizer used during training. 'surfel' routes "
                         "through diff_surfel_rasterization (rank-2 native).")
    ap.add_argument("--surfel_eigval_floor", type=float, default=1e-6)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.dataset} scene from {args.scene_dir} (image_scale={args.image_scale})")
    ds = _load_dataset(
        args.dataset, args.scene_dir, args.image_scale, args.split,
        allow_distortion=args.allow_distortion,
    )
    print(f"  T={ds.T}, points={ds.N_points}, H={ds.H}, W={ds.W}")

    frame_idxs = _parse_frames(args.frames, ds.T)
    if not frame_idxs:
        raise SystemExit(f"No frames parsed from --frames {args.frames!r}")
    if max(frame_idxs) >= ds.T:
        raise SystemExit(f"frame index {max(frame_idxs)} >= ds.T ({ds.T})")
    print(f"  Rendering {len(frame_idxs)} frame(s): {frame_idxs[:5]}{'...' if len(frame_idxs) > 5 else ''}")

    # Build a model with the same shape as the checkpoint, then overwrite from state.
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    n_saved = state["n_raw"].shape[0]
    # Detect SH-degree from the saved keys: sh_rest stores K-1 = (deg+1)^2 - 1 coeffs.
    if "sh_rest" in state:
        K_rest = state["sh_rest"].shape[1]
        ckpt_sh_degree = int(round(((K_rest + 1) ** 0.5))) - 1
    else:
        ckpt_sh_degree = 0
    print(f"  Checkpoint has N={n_saved} Gaussians, sh_degree={ckpt_sh_degree} "
          f"(vs {ds.N_points} init points)")

    # Build a placeholder model sized to the checkpoint, then load state_dict.
    # We use the dataset's first n_saved points (or pad if checkpoint is smaller)
    # purely to get the right tensor shapes; values are overwritten by load.
    pts_for_init = ds.points3D[:n_saved] if n_saved <= ds.N_points else ds.points3D
    if pts_for_init.shape[0] < n_saved:
        # Pad by repeating the last point until we hit n_saved (shape-only).
        pad = pts_for_init[-1:].repeat(n_saved - pts_for_init.shape[0], 1)
        pts_for_init = torch.cat([pts_for_init, pad], dim=0)
    times_init = torch.zeros(n_saved, dtype=torch.float64)
    params = init_gaussians_from_points(
        pts_for_init, times_init, ds.cameras_per_frame,
        sigma_init_sq=0.02,
        sigma_k_pixel=1.0, sigma_k_temporal=0.0,
    )
    model = trainable_from_params(
        params, dtype=DTYPE, device=device, sh_degree=ckpt_sh_degree,
    )
    model.load_state_dict(state, strict=True)
    print(f"  Loaded checkpoint -> N={model.N} Gaussians on {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bg = torch.zeros(3, dtype=DTYPE, device=device)
    raster_cfg = FastRasterConfig(
        sigma_3d_blur=args.sigma_3d_blur, sh_degree=ckpt_sh_degree,
    )
    surfel_cfg = SurfelRasterConfig(
        sh_degree=ckpt_sh_degree, eigval_floor=args.surfel_eigval_floor,
    )

    for f_idx in frame_idxs:
        cam = ds.cameras_per_frame[f_idx]
        t_value = float(ds.times[f_idx])
        params_now = model.forward()

        with torch.no_grad():
            if args.rasterizer == "surfel":
                img = surfel_rasterize(params_now, t_value, cam, ds.H, ds.W,
                                       background=bg, config=surfel_cfg)
            else:
                img = fast_rasterize(params_now, t_value, cam, ds.H, ds.W,
                                     background=bg, config=raster_cfg)

        arr = (img.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        out_path = args.output_dir / f"render_frame{f_idx:04d}.png"
        Image.fromarray(arr).save(out_path)
        print(f"  [{f_idx:04d}] -> {out_path.name}")

        if args.side_by_side:
            gt = ds.frame_loader(f_idx).to(DTYPE)
            gt_arr = (gt.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            sbs = np.concatenate([gt_arr, arr], axis=1)
            sbs_path = args.output_dir / f"sbs_frame{f_idx:04d}.png"
            Image.fromarray(sbs).save(sbs_path)

    print(f"Saved {len(frame_idxs)} render(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
