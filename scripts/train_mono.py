"""
Local entrypoint for monocular training (NeRFies / DyCheck).

Usage (from repo root):
  python scripts/train_mono.py \
      --dataset nerfies \
      --scene_dir data/nerfies/<scene> \
      --init_strategy median \
      --num_iters 5000 \
      --output_dir checkpoints/<scene>

Loads a MonocularDataset, initializes the per-frame Gaussian model from the
scene's point cloud + observability, and runs the standard Trainer in
monocular sampling mode.

This script is dataset-format-agnostic at the call-site level: it dispatches
to grassmann.datasets.{nerfies,dycheck} based on `--dataset`. Both loaders
return the same MonocularDataset contract.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/train_mono.py ...` from the repo root without installing
# the package. Modal sets this up via add_local_python_source; here we mirror it.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from grassmann.datasets.dycheck import load_dycheck
from grassmann.datasets.nerfies import load_nerfies
from grassmann.density_control import DensityConfig
from grassmann.initialization import init_gaussians_from_points
from grassmann.trainable import trainable_from_params
from grassmann.training import Trainer, TrainerConfig


DTYPE = torch.float32


def _load_dataset(name: str, scene_dir: Path, image_scale: int, split: str | None):
    if name == "nerfies":
        if split is not None:
            print(f"  [warning] --split is ignored for --dataset nerfies (no splits/ dir)")
        return load_nerfies(scene_dir, image_scale=image_scale)
    if name == "dycheck":
        return load_dycheck(scene_dir, image_scale=image_scale, split_name=split)
    raise ValueError(f"Unknown --dataset {name!r}; expected nerfies|dycheck")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("nerfies", "dycheck"), required=True)
    ap.add_argument("--scene_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, default=None,
                    help="Where to write the trained checkpoint. Defaults to scene_dir.")
    ap.add_argument("--image_scale", type=int, default=4)
    ap.add_argument("--split", type=str, default=None,
                    help="DyCheck split name (e.g. 'train', 'common'). Ignored for nerfies.")
    ap.add_argument("--init_strategy", choices=("lookat", "birth", "median", "random"),
                    default="median",
                    help="How to pick the per-point ref camera at init.")
    ap.add_argument("--num_iters", type=int, default=5000)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--use_fast_rasterizer", action="store_true")
    ap.add_argument("--device", type=str, default=None,
                    help="cpu | cuda. Defaults to cuda if available.")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.dataset} scene from {args.scene_dir} (image_scale={args.image_scale})")
    ds = _load_dataset(args.dataset, args.scene_dir, args.image_scale, args.split)
    print(f"  T={ds.T}, points={ds.N_points}, H={ds.H}, W={ds.W}, "
          f"train={len(ds.train_indices)}, val={len(ds.val_indices)}")

    # Initialize one Gaussian per scene point. Pick a temporal mean per point
    # from the median-observed frame's normalized time (sensible for static
    # background; dynamic objects will be re-organized by training).
    print(f"  Initializing {ds.N_points} Gaussians (strategy={args.init_strategy})...")
    times_for_init = []
    for obs in ds.observability:
        if obs:
            t_idx = obs[len(obs) // 2]
        else:
            t_idx = ds.T // 2
        times_for_init.append(float(ds.times[t_idx]))
    times_init = torch.tensor(times_for_init, dtype=torch.float64)

    params = init_gaussians_from_points(
        ds.points3D,
        times_init,
        ds.cameras_per_frame,
        strategy=args.init_strategy,
        observability=ds.observability,
        sigma_aa=0.02, sigma_bb=0.05, sigma_ab=0.0,
        opacity=0.5,
        sigma_k_pixel=1.0,
        sigma_k_temporal=0.0,
    )
    model = trainable_from_params(params, dtype=DTYPE, device=device)
    print(f"  Model: {model.N} Gaussians on {device}")

    config = TrainerConfig(
        num_iters=args.num_iters,
        log_every=args.log_every,
        lambda_l1=0.8,
        lambda_structural=0.2,
        lr_pq=1e-3, lr_mean=5e-3, lr_L=5e-3,
        lr_opacity=5e-2, lr_color=5e-2,
        background=torch.zeros(3, dtype=DTYPE, device=device),
        densify_every=500,
        densify_start=2000,
        densify_stop=max(args.num_iters // 2, 2000),
        density_config=DensityConfig(
            opacity_threshold=0.001,
            scale_min=1e-5, scale_max=2.0,
            grad_threshold=2e-4,
            clone_scale_threshold=0.05,
        ),
        use_fast_rasterizer=args.use_fast_rasterizer,
        validation_every=max(args.log_every, args.num_iters // 10),
    )
    trainer = Trainer.from_monocular_dataset(model, ds, config)

    print(f"Training for {args.num_iters} iterations...")
    history = trainer.train()

    out_dir = args.output_dir or args.scene_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"trained_{args.dataset}_{args.init_strategy}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
        "init_strategy": args.init_strategy,
        "dataset": args.dataset,
    }, out_path)
    print(f"Saved checkpoint to {out_path}")


if __name__ == "__main__":
    main()
