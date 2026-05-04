"""Standalone render script for Yang's 4DGS.

Upstream has no render.py: training only logs to TensorBoard and never saves
per-frame PNGs. We need PNGs on disk to compute apples-to-apples PSNR/SSIM
against D3DGS's render outputs.

Layout written:
  <model_path>/test/ours_<iter>/renders/<image_name>.png   (model render)
  <model_path>/test/ours_<iter>/gt/<image_name>.png        (GT)

Same convention as the original 3D-GS render.py and as Deformable-3D-Gaussians'
render outputs, so downstream eval scripts (metrics.py, our perframe-PSNR
loop) work without modification.
"""
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state


def _load_cfg_args(model_path: str) -> Namespace:
    cfg_path = os.path.join(model_path, "cfg_args")
    with open(cfg_path) as f:
        cfg_str = f.read()
    return eval(cfg_str)  # noqa: S307 -- file we wrote ourselves


def render_set(model_path, name, iteration, views_dataset, gaussians, pipeline, background):
    out_dir = Path(model_path) / name / f"ours_{iteration}"
    renders_dir = out_dir / "renders"
    gt_dir = out_dir / "gt"
    renders_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(len(views_dataset)), desc=f"Rendering {name}"):
        gt_image, viewpoint = views_dataset[idx]
        gt_image = gt_image.cuda()
        viewpoint = viewpoint.cuda()

        with torch.no_grad():
            render_pkg = render(viewpoint, gaussians, pipeline, background)
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        gt = torch.clamp(gt_image[:3, ...], 0.0, 1.0)

        # Use the camera's image_name so we can correlate frames across runs
        # (D3DGS uses the same convention).
        name_stub = viewpoint.image_name
        torchvision.utils.save_image(image, str(renders_dir / f"{name_stub}.png"))
        torchvision.utils.save_image(gt,    str(gt_dir / f"{name_stub}.png"))


def main():
    parser = ArgumentParser(description="Render script for Yang 4DGS")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    # Yang's prepare_output_and_logger writes only ModelParams to cfg_args, so
    # the 4D-specific knobs (gaussian_dim, rot_4d, time_duration, force_sh_3d,
    # eval_shfs_4d, num_pts) aren't recoverable from disk. Pass them via CLI.
    parser.add_argument("--gaussian_dim", type=int, default=4)
    parser.add_argument("--rot_4d", action="store_true", default=True)
    parser.add_argument("--force_sh_3d", action="store_true", default=False)
    parser.add_argument("--eval_shfs_4d", action="store_true", default=True)
    parser.add_argument("--time_duration", nargs=2, type=float, default=[0.0, 1.0])
    parser.add_argument("--num_pts", type=int, default=15000)
    parser.add_argument("--num_pts_ratio", type=float, default=1.0)
    # Override dataset.resolution. cfg_args persists the training-time value
    # (1, since we already pre-downsampled in the rectifier). To render at
    # scale-8 resolution from a model trained on scale-4 input, pass
    # --override_resolution 2 (cam_info's W=268, H=480 / 2 = 134, 240).
    parser.add_argument("--override_resolution", type=int, default=-1)
    args_cli = parser.parse_args()

    safe_state(args_cli.quiet)

    cfg = _load_cfg_args(args_cli.model_path)
    # cfg is a Namespace with everything train.py wrote, including the
    # gaussian_dim/time_duration/rot_4d/force_sh_3d/sh_degree/etc. we need
    # to reconstruct the model. Forward to ModelParams/PipelineParams via
    # a fake parser so we can re-extract typed groups.

    # ModelParams' field names use underscore prefixes for shorthand args
    # (_source_path -> --source_path). Build the parser then merge the cfg
    # values onto it; for fields not present we leave defaults.
    fp = ArgumentParser()
    lp = ModelParams(fp, sentinel=False)
    pp = PipelineParams(fp)
    base = fp.parse_args([])
    merged = vars(base)
    merged.update(vars(cfg))
    args = Namespace(**merged)

    dataset = lp.extract(args)
    pipeline = pp.extract(args)
    if args_cli.override_resolution > 0:
        dataset.resolution = args_cli.override_resolution
        print(f"override resolution -> {dataset.resolution}", flush=True)

    # Reconstruct GaussianModel using CLI flags for the 4D-specific knobs (the
    # cfg_args file written by Yang's training only contains ModelParams).
    gaussian_dim  = args_cli.gaussian_dim
    time_duration = args_cli.time_duration
    rot_4d        = args_cli.rot_4d
    force_sh_3d   = args_cli.force_sh_3d
    num_pts       = args_cli.num_pts
    num_pts_ratio = args_cli.num_pts_ratio
    sh_degree_t   = 2 if args_cli.eval_shfs_4d else 0
    prefilter_var = getattr(dataset, 'prefilter_var', -1.0)
    # Pipeline also needs eval_shfs_4d set (gaussian_renderer/__init__.py:47
    # reads pipe.eval_shfs_4d when assembling raster_settings).
    pipeline.eval_shfs_4d = args_cli.eval_shfs_4d

    gaussians = GaussianModel(
        dataset.sh_degree,
        gaussian_dim=gaussian_dim,
        time_duration=time_duration,
        rot_4d=rot_4d,
        force_sh_3d=force_sh_3d,
        sh_degree_t=sh_degree_t,
        prefilter_var=prefilter_var,
    )
    # Build the Scene WITHOUT load_iteration -- Yang's training saves
    # `chkpnt<iter>.pth` (model_params, iteration) tuples instead of the
    # `point_cloud/iteration_<iter>/point_cloud.ply` upstream-3DGS layout that
    # Scene.__init__'s load path expects. We restore the gaussians manually
    # from the .pth file below.
    scene = Scene(
        dataset, gaussians,
        load_iteration=None,
        shuffle=False,
        num_pts=num_pts, num_pts_ratio=num_pts_ratio,
        time_duration=time_duration,
    )
    chkpnt_path = os.path.join(args_cli.model_path, f"chkpnt{args_cli.iteration}.pth")
    if not os.path.isfile(chkpnt_path):
        raise FileNotFoundError(
            f"checkpoint {chkpnt_path} missing. Available: "
            f"{[f for f in os.listdir(args_cli.model_path) if f.startswith('chkpnt')]}"
        )
    print(f"loading {chkpnt_path}")
    model_params, _saved_iter = torch.load(chkpnt_path, map_location="cuda")
    # GaussianModel.restore handles the per-dim unpacking. Pass training_args=None
    # so it skips optimizer reconstruction (we don't need it for render-only).
    gaussians.restore(model_params, None)
    print(f"loaded {gaussians._xyz.shape[0]} gaussians", flush=True)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iteration = args_cli.iteration

    if not args_cli.skip_train:
        render_set(args_cli.model_path, "train", iteration,
                   scene.getTrainCameras(), gaussians, pipeline, background)
    if not args_cli.skip_test:
        render_set(args_cli.model_path, "test", iteration,
                   scene.getTestCameras(), gaussians, pipeline, background)


if __name__ == "__main__":
    main()
