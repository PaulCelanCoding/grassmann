"""
Modal entry for monocular GPU training on L4.

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

Smoke = train with --num_iters 100 --image_scale 4 to validate the
entire Modal + CUDA + data path before committing GPU-hours to a real run.

The legacy N3DV/multi-camera training scripts (incl. the single-Gaussian
sanity check) live under legacy/multi_camera/scripts/.
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
) -> None:
    scene_dir = _ensure_scene_unpacked(scene)
    out_dir = f"/checkpoints/{dataset}-{scene}-{init_strategy}-{num_iters}it"
    argv = [
        "python", "scripts/train_mono.py",
        "--dataset", dataset,
        "--scene_dir", scene_dir,
        "--output_dir", out_dir,
        "--num_iters", str(num_iters),
        "--image_scale", str(image_scale),
        "--init_strategy", init_strategy,
    ]
    if split is not None:
        argv += ["--split", split]
    if use_fast:
        argv.append("--use_fast_rasterizer")
    if allow_distortion:
        argv.append("--allow_distortion")
    _run(argv)
    ckpt_vol.commit()


@app.local_entrypoint()
def main(
    cmd: str = "smoke",
    dataset: str = "nerfies",
    scene: str = "slice-banana",
    iters: int = 30000,
    init_strategy: str = "median",
    split: str = "",
    allow_distortion: bool = True,  # default True: every shipped scene has it
):
    split_arg = split or None
    if cmd == "smoke":
        train.remote(
            dataset=dataset, scene=scene,
            num_iters=100, image_scale=4, use_fast=True,
            init_strategy=init_strategy, split=split_arg,
            allow_distortion=allow_distortion,
        )
    elif cmd == "train":
        train.remote(
            dataset=dataset, scene=scene,
            num_iters=iters, image_scale=2, use_fast=True,
            init_strategy=init_strategy, split=split_arg,
            allow_distortion=allow_distortion,
        )
    else:
        raise SystemExit(f"unknown --cmd {cmd!r}; expected smoke|train")
