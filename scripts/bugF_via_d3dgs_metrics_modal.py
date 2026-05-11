"""
Run Bug-F's rendered val/train frames through D3DGS's exact `metrics.py`
to get SSIM/PSNR/LPIPS computed by literally the same code that produced
D3DGS's published numbers. Eliminates any LPIPS-backbone / SSIM-impl /
GT-preprocessing differences.

Two-stage:
  Stage A (grassmann-train image): load Bug-F ckpt, render every train+val
          frame to PNG, lay out as D3DGS's metrics.py expects:
            <out_dir>/{train,test}/ours_<iter>/renders/<idx>.png
            <out_dir>/{train,test}/ours_<iter>/gt/<idx>.png  (copied from
                                                              D3DGS ckpt)

  Stage B (deformable image): cd into D3DGS repo, `python metrics.py
          -m <out_dir>`. Writes <out_dir>/results.json with the canonical
          numbers.

Usage:
  modal run scripts/bugF_via_d3dgs_metrics_modal.py \
    --bugf-ckpt nerfies-slice-banana-spatial_slice-16000it-bugF-iso5min-v2-2026-05-11/trained_nerfies_spatial_slice.pt \
    --d3dgs-dir deformable-slice-banana-7000it-iso5min-v3-2026-05-11 \
    --d3dgs-iters-tag ours_7000 \
    --out-dir bugF-via-d3dgs-metrics-iso5min
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent

# ---------------- Stage A image (render Bug-F → PNGs) -----------------------

render_image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
        add_python="3.11",
    )
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install("numpy", "matplotlib", "pillow", "tqdm")
    .pip_install(
        "git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git",
        gpu="L4",
    )
    .add_local_python_source("grassmann")
    .add_local_dir(str(REPO / "scripts"), remote_path="/root/scripts")
)

# ---------------- Stage B image (D3DGS metrics.py) --------------------------

DEFORMABLE_REPO = "https://github.com/ingra14m/Deformable-3D-Gaussians.git"
DEFORMABLE_REV = "main"

metrics_image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
        add_python="3.11",
    )
    .apt_install("git", "build-essential", "ninja-build", "wget",
                 "libgl1", "libglib2.0-0")
    .pip_install("numpy", "matplotlib", "pillow", "tqdm",
                 "plyfile==0.8.1", "imageio==2.27.0", "imageio-ffmpeg",
                 "opencv-python", "scipy", "lpips", "torchvision")
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

app = modal.App("grassmann-bugF-via-d3dgs-metrics")

mono_vol = modal.Volume.from_name("gs-mono", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("gs-checkpoints", create_if_missing=True)
VOLUMES = {"/data": mono_vol, "/checkpoints": ckpt_vol}


def _ensure_scene_unpacked(scene: str) -> str:
    import os, zipfile
    scene_dir = f"/data/{scene}"
    if os.path.isdir(scene_dir):
        return scene_dir
    zip_path = f"/data/{scene}.zip"
    if os.path.isfile(zip_path):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall("/data")
        return scene_dir
    raise FileNotFoundError(scene_dir)


@app.function(gpu="L4", image=render_image, volumes=VOLUMES, timeout=2 * 3600)
def render_stage(
    bugf_ckpt: str,
    d3dgs_dir: str,
    d3dgs_iters_tag: str,
    bugf_iter: int,
    out_dir_rel: str,
    scene: str = "slice-banana",
    image_scale: int = 4,
    sigma_3d_blur: float = 1e-4,
) -> None:
    """Render Bug-F at all train+val frames, save as PNGs in D3DGS layout.
    Also copy D3DGS's saved GT into the same dir so metrics.py finds it.
    """
    import os, sys, shutil
    import numpy as np
    import torch
    from PIL import Image as PILImage

    sys.path.insert(0, "/root")
    from grassmann.datasets.nerfies import load_nerfies
    from grassmann.fast_rasterizer import FastRasterConfig, fast_rasterize
    from grassmann.initialization import init_gaussians_from_points
    from grassmann.trainable import trainable_from_params

    DTYPE = torch.float32
    device = "cuda"
    scene_dir = _ensure_scene_unpacked(scene)
    ds = load_nerfies(scene_dir, image_scale=image_scale, allow_distortion=True)
    print(f"Loaded {scene} scale={image_scale} T={ds.T} H={ds.H} W={ds.W}")

    train_idx = list(range(0, ds.T, 4))
    val_idx   = list(range(2, ds.T, 4))
    print(f"Split: train={len(train_idx)} val={len(val_idx)}")

    ckpt_path = f"/checkpoints/{bugf_ckpt}"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
    n_saved = state["n_raw"].shape[0]
    sh_degree = 0
    if "sh_rest" in state:
        K_rest = state["sh_rest"].shape[1]
        sh_degree = int(round(((K_rest + 1) ** 0.5))) - 1
    pts = ds.points3D[:n_saved] if n_saved <= ds.N_points else ds.points3D
    if pts.shape[0] < n_saved:
        pad = pts[-1:].repeat(n_saved - pts.shape[0], 1)
        pts = torch.cat([pts, pad], dim=0)
    times0 = torch.zeros(n_saved, dtype=torch.float64)
    params0 = init_gaussians_from_points(
        pts, times0, ds.cameras_per_frame,
        sigma_init_sq=0.02, sigma_k_pixel=1.0, sigma_k_temporal=0.0,
    )
    model = trainable_from_params(params0, dtype=DTYPE, device=device, sh_degree=sh_degree)
    model.load_state_dict(state, strict=True)
    print(f"Loaded Bug-F: N={model.N} sh={sh_degree}")

    bg = torch.zeros(3, dtype=DTYPE, device=device)
    raster_cfg = FastRasterConfig(sigma_3d_blur=sigma_3d_blur, sh_degree=sh_degree)
    iter_tag_out = f"ours_{bugf_iter}"

    d3_root = f"/checkpoints/{d3dgs_dir}"
    out_root = f"/checkpoints/{out_dir_rel}"
    os.makedirs(out_root, exist_ok=True)

    for split_name, split_idx, d3_split in (
        ("train", train_idx, "train"),
        ("test",  val_idx,   "test"),
    ):
        ren_dir = f"{out_root}/{split_name}/{iter_tag_out}/renders"
        gt_dir  = f"{out_root}/{split_name}/{iter_tag_out}/gt"
        os.makedirs(ren_dir, exist_ok=True)
        os.makedirs(gt_dir,  exist_ok=True)

        d3_gt_src = f"{d3_root}/{d3_split}/{d3dgs_iters_tag}/gt"
        if not os.path.isdir(d3_gt_src):
            raise FileNotFoundError(f"D3DGS GT dir missing: {d3_gt_src}")

        print(f"\n=== {split_name} ({len(split_idx)} frames) ===")
        for k, f_idx in enumerate(split_idx):
            cam = ds.cameras_per_frame[f_idx]
            t   = float(ds.times[f_idx])
            with torch.no_grad():
                img = fast_rasterize(model.forward(), t, cam, ds.H, ds.W,
                                     background=bg, config=raster_cfg)
            # Save Bug-F render as uint8 PNG (symmetric with D3DGS output)
            arr = (img.detach().cpu().numpy().clip(0, 1) * 255.0).round().astype(np.uint8)
            PILImage.fromarray(arr).save(f"{ren_dir}/{k:05d}.png")
            # Copy D3DGS's saved GT for this index (apples-to-apples GT)
            shutil.copyfile(f"{d3_gt_src}/{k:05d}.png", f"{gt_dir}/{k:05d}.png")
            if (k + 1) % 20 == 0:
                print(f"  {split_name} {k+1}/{len(split_idx)}")

    ckpt_vol.commit()
    print(f"\nBug-F renders → {out_root}/{{train,test}}/{iter_tag_out}/renders/")
    print(f"GT copied from D3DGS into  {out_root}/{{train,test}}/{iter_tag_out}/gt/")


@app.function(gpu="L4", image=metrics_image, volumes=VOLUMES, timeout=1 * 3600)
def metrics_stage(out_dir_rel: str) -> None:
    """Run D3DGS's own metrics.py against the laid-out PNGs."""
    import os
    out_dir = f"/checkpoints/{out_dir_rel}"
    cwd = "/root/Deformable-3D-Gaussians"
    argv = ["python", "metrics.py", "-m", out_dir]
    print(f">>> cwd={cwd}  argv={argv}", flush=True)
    subprocess.run(argv, check=True, cwd=cwd)
    ckpt_vol.commit()
    # Print resulting JSON
    rj = f"{out_dir}/results.json"
    pj = f"{out_dir}/per_view.json"
    if os.path.isfile(rj):
        print("--- results.json ---")
        print(open(rj).read())
    if os.path.isfile(pj):
        print(f"--- per_view.json keys: {list(__import__('json').load(open(pj)).keys())}")


@app.local_entrypoint()
def main(
    bugf_ckpt: str = "nerfies-slice-banana-spatial_slice-16000it-bugF-iso5min-v2-2026-05-11/trained_nerfies_spatial_slice.pt",
    d3dgs_dir: str = "deformable-slice-banana-7000it-iso5min-v3-2026-05-11",
    d3dgs_iters_tag: str = "ours_7000",
    bugf_iter: int = 16000,
    out_dir: str = "bugF-via-d3dgs-metrics-iso5min",
    scene: str = "slice-banana",
    image_scale: int = 4,
):
    # Stage A: render Bug-F + place PNGs in D3DGS layout.
    render_stage.remote(
        bugf_ckpt=bugf_ckpt,
        d3dgs_dir=d3dgs_dir,
        d3dgs_iters_tag=d3dgs_iters_tag,
        bugf_iter=bugf_iter,
        out_dir_rel=out_dir,
        scene=scene,
        image_scale=image_scale,
    )
    # Stage B: run D3DGS's metrics.py on Bug-F PNGs.
    metrics_stage.remote(out_dir_rel=out_dir)
