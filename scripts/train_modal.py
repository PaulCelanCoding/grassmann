"""
Modal entry for GPU training/diagnostics on L4.

Volumes (created on first run):
  gs-n3dv         /data          N3DV scenes (cam??/, cameras.json, points3D.txt)
  gs-checkpoints  /checkpoints   per-run output

One-time data upload (from repo root):
  modal volume create gs-n3dv
  modal volume put gs-n3dv ./data/n3dv/flame_steak /flame_steak

Usage:
  modal run scripts/train_modal.py --cmd smoke
  modal run scripts/train_modal.py --cmd train --iters 30000
  modal run scripts/train_modal.py --cmd diagnose
  modal run scripts/train_modal.py --cmd sanity

Smoke = train with --num_iters 100 --downscale_factor 4 to validate the
entire Modal + CUDA + data path before committing GPU-hours to a real run.
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

n3dv_vol = modal.Volume.from_name("gs-n3dv", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("gs-checkpoints", create_if_missing=True)

VOLUMES = {"/data": n3dv_vol, "/checkpoints": ckpt_vol}
GPU = "L4"


def _run(argv: list[str]) -> None:
    print(f">>> {' '.join(argv)}", flush=True)
    subprocess.run(argv, check=True, cwd="/root")


@app.function(gpu=GPU, volumes=VOLUMES, timeout=24 * 3600)
def train(scene: str, num_iters: int, downscale: int, use_fast: bool) -> None:
    scene_dir = f"/data/{scene}"
    out_dir = f"/checkpoints/{scene}-{num_iters}it"
    argv = [
        "python", "scripts/train_n3dv.py", "train",
        "--scene_dir", scene_dir,
        "--output_dir", out_dir,
        "--num_iters", str(num_iters),
        "--downscale_factor", str(downscale),
    ]
    if use_fast:
        argv.append("--use_fast_rasterizer")
    _run(argv)
    ckpt_vol.commit()


@app.function(gpu=GPU, volumes=VOLUMES, timeout=3600)
def diagnose(scene: str) -> None:
    _run([
        "python", "scripts/diagnose_n3dv.py",
        "--scene_dir", f"/data/{scene}",
    ])
    ckpt_vol.commit()


@app.function(gpu=GPU, volumes=VOLUMES, timeout=3600)
def sanity(scene: str) -> None:
    _run([
        "python", "scripts/sanity_one_gaussian.py",
        "--scene_dir", f"/data/{scene}",
    ])
    ckpt_vol.commit()


@app.local_entrypoint()
def main(
    cmd: str = "smoke",
    scene: str = "flame_steak",
    iters: int = 30000,
):
    if cmd == "smoke":
        train.remote(scene=scene, num_iters=100, downscale=4, use_fast=True)
    elif cmd == "train":
        train.remote(scene=scene, num_iters=iters, downscale=1, use_fast=True)
    elif cmd == "diagnose":
        diagnose.remote(scene=scene)
    elif cmd == "sanity":
        sanity.remote(scene=scene)
    else:
        raise SystemExit(f"unknown --cmd {cmd!r}; expected smoke|train|diagnose|sanity")
