"""
Modal entry for monocular GPU training on L4 (Phase A: 3-plane projector).

Volumes (created on first run):
  gs-mono         /data          NeRFies / DyCheck scenes
  gs-checkpoints  /checkpoints   per-run output

One-time data upload (from repo root):
  modal volume create gs-mono
  modal volume put gs-mono ./data/nerfies/<scene>  /<scene>
  modal volume put gs-mono ./data/dycheck/<scene>  /<scene>

Usage:
  modal run scripts/train_modal.py --cmd smoke --dataset nerfies --scene <scene>
  modal run scripts/train_modal.py --cmd train --dataset nerfies --scene <scene> --iters 30000
  modal run scripts/train_modal.py --cmd train --dataset dycheck --scene <scene> --split train
  modal run scripts/train_modal.py --cmd render --dataset nerfies --scene <scene> \
      --ckpt nerfies-slice-banana-random-30000it/trained_nerfies_random.pt \
      --frames 0,50,100 --side-by-side

Density control is disabled in Phase A (the legacy DC targets the 2-plane
parameterization; see the plan, Phase C, for re-introduction). The
`init_strategy` and `sigma_3d_blur` flags are the only meaningful knobs
exposed here.
"""
import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
        add_python="3.11",
    )
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install(
        "numpy",
        "matplotlib",
        "pillow",
        "tqdm",
        "pytest",
    )
    .pip_install(
        "git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git",
        gpu="L4",
    )
    .add_local_python_source("grassmann")
    .add_local_dir(str(REPO / "scripts"), remote_path="/root/scripts")
)

app = modal.App("grassmann-train", image=image)

mono_vol = modal.Volume.from_name("gs-mono", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("gs-checkpoints", create_if_missing=True)

VOLUMES = {"/data": mono_vol, "/checkpoints": ckpt_vol}
GPU = "L4"


def _run(argv: list[str]) -> None:
    print(f">>> {' '.join(argv)}", flush=True)
    subprocess.run(argv, check=True, cwd="/root")


def _ensure_scene_unpacked(scene: str) -> str:
    """Ensure /data/<scene> exists. If not, unzip /data/<scene>.zip onto the volume.

    Workaround for `modal volume put` being painfully slow when uploading many
    small files (we observed ~30+ min for 1500 files vs ~5 min for one zip).
    Standard NeRFies/HyperNeRF/DyCheck scenes have ~330 frames * (4 RGB scales
    + camera JSON + ...) -> easily into the thousands.

    Upload pattern:
        modal volume put gs-mono ./data/<dataset>/<scene>.zip /<scene>.zip
    """
    import os
    import zipfile

    scene_dir = f"/data/{scene}"
    if os.path.isdir(scene_dir):
        return scene_dir
    zip_path = f"/data/{scene}.zip"
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(
            f"Neither {scene_dir!r} (dir) nor {zip_path!r} (zip) exist on the gs-mono "
            f"volume. Upload one of them first."
        )
    print(f"  unpacking {zip_path} -> /data/...", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall("/data")
    if not os.path.isdir(scene_dir):
        # The zip's top-level dir name didn't match `scene`. Try to detect it.
        for entry in os.listdir("/data"):
            candidate = f"/data/{entry}"
            if os.path.isdir(candidate) and os.path.exists(f"{candidate}/dataset.json"):
                if entry != scene:
                    print(f"  [info] zip top-level was {entry!r}, not {scene!r}; "
                          f"using {candidate}", flush=True)
                return candidate
        raise RuntimeError(f"Unzipped {zip_path} but no scene dir found under /data/.")
    mono_vol.commit()
    return scene_dir


@app.function(gpu=GPU, volumes=VOLUMES, timeout=24 * 3600)
def train(
    dataset: str,
    scene: str,
    num_iters: int,
    image_scale: int,
    use_fast: bool,
    init_strategy: str,
    split: str | None,
    allow_distortion: bool,
    log_every: int,
    sigma_3d_blur: float,
    sigma_init_sq: float,
    run_tag: str,
    seed: int | None,
    static_baseline: bool,
    val_stride: int,
    split_convention: str,
    init_points_multiplier: int,
    diag_single_frame: int,
    lambda_frob: float,
    opacity_reset_every: int,
    lambda_aniso: float,
) -> None:
    scene_dir = _ensure_scene_unpacked(scene)
    suffix = f"-{run_tag}" if run_tag else ""
    out_dir = f"/checkpoints/{dataset}-{scene}-{init_strategy}-{num_iters}it{suffix}"
    argv = [
        "python", "scripts/train_mono.py",
        "--dataset", dataset,
        "--scene_dir", scene_dir,
        "--output_dir", out_dir,
        "--num_iters", str(num_iters),
        "--log_every", str(log_every),
        "--image_scale", str(image_scale),
        "--init_strategy", init_strategy,
        "--sigma_3d_blur", str(sigma_3d_blur),
        "--sigma_init_sq", str(sigma_init_sq),
    ]
    if split is not None:
        argv += ["--split", split]
    if use_fast:
        argv.append("--use_fast_rasterizer")
    if allow_distortion:
        argv.append("--allow_distortion")
    if seed is not None:
        argv += ["--seed", str(seed)]
    if static_baseline:
        argv.append("--static_baseline")
    argv += ["--val_stride", str(val_stride),
             "--split_convention", split_convention,
             "--init_points_multiplier", str(init_points_multiplier)]
    if diag_single_frame >= 0:
        argv += ["--diag_single_frame", str(diag_single_frame)]
    if lambda_frob > 0.0:
        argv += ["--lambda_frob", str(lambda_frob)]
    if opacity_reset_every > 0:
        argv += ["--opacity_reset_every", str(opacity_reset_every)]
    if lambda_aniso > 0.0:
        argv += ["--lambda_aniso", str(lambda_aniso)]
    _run(argv)
    ckpt_vol.commit()


@app.function(gpu=GPU, volumes=VOLUMES, timeout=2 * 3600)
def render(
    dataset: str,
    scene: str,
    ckpt: str,
    frames: str,
    image_scale: int,
    split: str | None,
    allow_distortion: bool,
    side_by_side: bool,
    sigma_3d_blur: float,
) -> None:
    import os
    scene_dir = _ensure_scene_unpacked(scene)
    ckpt_path = f"/checkpoints/{ckpt}"
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"{ckpt_path!r} not found on gs-checkpoints volume.")
    out_dir = os.path.join(os.path.dirname(ckpt_path), "renders")
    argv = [
        "python", "scripts/render_mono.py",
        "--dataset", dataset,
        "--scene_dir", scene_dir,
        "--ckpt", ckpt_path,
        "--frames", frames,
        "--output_dir", out_dir,
        "--image_scale", str(image_scale),
        "--device", "cuda",
        "--sigma_3d_blur", str(sigma_3d_blur),
    ]
    if split is not None:
        argv += ["--split", split]
    if allow_distortion:
        argv.append("--allow_distortion")
    if side_by_side:
        argv.append("--side_by_side")
    _run(argv)
    ckpt_vol.commit()
    rel = os.path.relpath(out_dir, "/checkpoints")
    print(f"\nPull renders locally:\n  modal volume get gs-checkpoints {rel} ./renders", flush=True)


@app.local_entrypoint()
def main(
    cmd: str = "smoke",
    dataset: str = "nerfies",
    scene: str = "slice-banana",
    iters: int = 500,
    log_every: int = 50,
    init_strategy: str = "random",
    split: str = "",
    allow_distortion: bool = True,
    ckpt: str = "",
    frames: str = "0",
    image_scale: int = 4,
    side_by_side: bool = False,
    sigma_3d_blur: float = 1e-4,
    sigma_init_sq: float = 0.02,
    run_tag: str = "",
    seed: int = -1,
    static_baseline: bool = False,
    val_stride: int = 4,
    split_convention: str = "val_stride",
    init_points_multiplier: int = 1,
    diag_single_frame: int = -1,
    lambda_frob: float = 0.0,
    opacity_reset_every: int = 0,
    lambda_aniso: float = 0.0,
):
    """
    --cmd smoke:  short run (--iters used; default 500) at scale 4. Validates
                  code path AND prints per-log_every loss to confirm convergence.
    --cmd train:  full run at scale 2 (default --iters 30000).
    --cmd render: load --ckpt (path under /checkpoints), render --frames via CUDA.
    """
    split_arg = split or None
    seed_arg = None if seed < 0 else seed
    if cmd == "smoke":
        train.remote(
            dataset=dataset, scene=scene,
            num_iters=iters, image_scale=4, use_fast=True,
            init_strategy=init_strategy, split=split_arg,
            allow_distortion=allow_distortion,
            log_every=log_every,
            sigma_3d_blur=sigma_3d_blur,
            sigma_init_sq=sigma_init_sq,
            run_tag=run_tag,
            seed=seed_arg,
            static_baseline=static_baseline,
            val_stride=val_stride,
            split_convention=split_convention,
            init_points_multiplier=init_points_multiplier,
            diag_single_frame=diag_single_frame,
            lambda_frob=lambda_frob,
            opacity_reset_every=opacity_reset_every,
            lambda_aniso=lambda_aniso,
        )
    elif cmd == "train":
        train.remote(
            dataset=dataset, scene=scene,
            num_iters=iters if iters != 500 else 30000,
            image_scale=2, use_fast=True,
            init_strategy=init_strategy, split=split_arg,
            allow_distortion=allow_distortion,
            log_every=log_every if log_every != 50 else 200,
            sigma_3d_blur=sigma_3d_blur,
            sigma_init_sq=sigma_init_sq,
            run_tag=run_tag,
            seed=seed_arg,
            static_baseline=static_baseline,
            val_stride=val_stride,
            split_convention=split_convention,
            init_points_multiplier=init_points_multiplier,
            diag_single_frame=diag_single_frame,
            lambda_frob=lambda_frob,
            opacity_reset_every=opacity_reset_every,
            lambda_aniso=lambda_aniso,
        )
    elif cmd == "render":
        if not ckpt:
            raise SystemExit("--cmd render requires --ckpt <path-under-/checkpoints>")
        render.remote(
            dataset=dataset, scene=scene,
            ckpt=ckpt, frames=frames, image_scale=image_scale,
            split=split_arg, allow_distortion=allow_distortion,
            side_by_side=side_by_side,
            sigma_3d_blur=sigma_3d_blur,
        )
    else:
        raise SystemExit(f"unknown --cmd {cmd!r}; expected smoke|train|render")
