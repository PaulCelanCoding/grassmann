"""
Modal entry for running Deformable-3D-Gaussians (Yang et al., CVPR 2024)
on slice-banana for an iso-iter comparison against our 3-plane projector.

Setup:
  - Builds an image with Deformable3DGS's CUDA extensions
    (depth-diff-gaussian-rasterization, simple-knn). Repo cloned at
    image-build time so changes don't trigger rebuilds.
  - Mounts gs-mono volume at /data; before launch, symlinks
    /data/slice-banana -> /data/interp/slice-banana so their reader
    picks the `name.startswith('interp')` branch
    (train = ids[::4], val = ids[2::4] = 83/82 split for slice-banana).

Usage:
  modal run scripts/train_modal_deformable.py --scene slice-banana --iters 14000

Outputs land in the gs-checkpoints volume under
deformable-{scene}-{iters}it/.
"""
import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent

# Pin the Deformable3DGS commit so the image is reproducible.
DEFORMABLE_REPO = "https://github.com/ingra14m/Deformable-3D-Gaussians.git"
DEFORMABLE_REV = "main"

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
        add_python="3.11",
    )
    .apt_install(
        "git", "build-essential", "ninja-build", "wget",
        # opencv-python (pulled by Deformable3DGS deps) needs libGL + glib2.
        "libgl1", "libglib2.0-0",
    )
    .pip_install(
        "numpy",
        "matplotlib",
        "pillow",
        "tqdm",
        "plyfile==0.8.1",
        "imageio==2.27.0",
        "imageio-ffmpeg",
        "opencv-python",
        "scipy",
        "lpips",
        "torchvision",
    )
    .run_commands(
        f"cd /root && git clone --recursive {DEFORMABLE_REPO} && "
        f"cd Deformable-3D-Gaussians && git checkout {DEFORMABLE_REV}",
        gpu="L4",
    )
    .run_commands(
        "cd /root/Deformable-3D-Gaussians && "
        "pip install ./submodules/depth-diff-gaussian-rasterization "
        "./submodules/simple-knn",
        gpu="L4",
    )
)

app = modal.App("grassmann-deformable", image=image)

mono_vol = modal.Volume.from_name("gs-mono", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("gs-checkpoints", create_if_missing=True)

VOLUMES = {"/data": mono_vol, "/checkpoints": ckpt_vol}
GPU = "L4"


def _ensure_interp_layout(scene: str) -> str:
    """Ensure /data/interp/<scene> exists (their reader needs the 'interp' parent
    dir name for the [::4] / [2::4] split logic). We symlink the existing
    /data/<scene> into /data/interp/<scene>. Returns the path their loader expects.
    """
    import os
    src = f"/data/{scene}"
    if not os.path.isdir(src):
        raise FileNotFoundError(f"{src!r} does not exist on gs-mono volume.")
    dst_parent = "/tmp/interp"
    os.makedirs(dst_parent, exist_ok=True)
    dst = f"{dst_parent}/{scene}"
    if not os.path.islink(dst) and not os.path.isdir(dst):
        os.symlink(src, dst)
    return dst


@app.function(gpu=GPU, volumes=VOLUMES, timeout=4 * 3600)
def train(
    scene: str,
    num_iters: int,
    resolution: int,
    run_tag: str,
) -> None:
    import os
    scene_dir = _ensure_interp_layout(scene)
    suffix = f"-{run_tag}" if run_tag else ""
    out_dir = f"/checkpoints/deformable-{scene}-{num_iters}it{suffix}"
    os.makedirs(out_dir, exist_ok=True)
    cwd = "/root/Deformable-3D-Gaussians"

    # Train
    train_argv = [
        "python", "train.py",
        "-s", scene_dir,
        "-m", out_dir,
        "--eval",
        "--iterations", str(num_iters),
        "--resolution", str(resolution),
    ]
    print(f">>> {' '.join(train_argv)}", flush=True)
    subprocess.run(train_argv, check=True, cwd=cwd)

    # Render test set
    render_argv = ["python", "render.py", "-m", out_dir, "--mode", "original",
                    "--resolution", str(resolution)]
    print(f">>> {' '.join(render_argv)}", flush=True)
    subprocess.run(render_argv, check=True, cwd=cwd)

    # Compute metrics (PSNR / SSIM / LPIPS) on test set
    metrics_argv = ["python", "metrics.py", "-m", out_dir]
    print(f">>> {' '.join(metrics_argv)}", flush=True)
    subprocess.run(metrics_argv, check=True, cwd=cwd)

    ckpt_vol.commit()


@app.local_entrypoint()
def main(
    scene: str = "slice-banana",
    iters: int = 14000,
    resolution: int = 4,
    run_tag: str = "",
):
    train.remote(
        scene=scene,
        num_iters=iters,
        resolution=resolution,
        run_tag=run_tag,
    )
