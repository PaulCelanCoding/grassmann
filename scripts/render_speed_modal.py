"""
Modal entry: per-frame rendering speed benchmark for a checkpoint.

Renders every val frame K times (after warmup), with CUDA syncs, and
reports median/p50/p95 ms per frame plus throughput. The point is to
characterize *inference* speed (not training) for the recipe in
production (e.g. uncapped+SSIM ckpt at N=97k).

Usage:
    modal run scripts/render_speed_modal.py \
        --bugf-ckpt nerfies-slice-banana-spatial_slice-14000it-bugF-uncapped-ssim/trained_nerfies_spatial_slice.pt
"""
from __future__ import annotations

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
        add_python="3.11",
    )
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install("numpy", "pillow", "tqdm")
    .pip_install(
        "git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git",
        gpu="L4",
    )
    .add_local_python_source("grassmann")
    .add_local_dir(str(REPO / "scripts"), remote_path="/root/scripts")
)

app = modal.App("grassmann-render-speed", image=image)

mono_vol = modal.Volume.from_name("gs-mono", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("gs-checkpoints", create_if_missing=True)
VOLUMES = {"/data": mono_vol, "/checkpoints": ckpt_vol}


def _ensure_scene_unpacked(scene: str) -> str:
    import os, zipfile
    scene_dir = f"/data/{scene}"
    if os.path.isdir(scene_dir):
        return scene_dir
    zip_path = f"/data/{scene}.zip"
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall("/data")
    return scene_dir


@app.function(gpu="L4", volumes=VOLUMES, timeout=2 * 3600)
def bench(
    bugf_ckpt: str,
    scene: str = "slice-banana",
    image_scale: int = 4,
    warmup_iters: int = 30,
    repeats_per_frame: int = 10,
) -> None:
    import sys, time
    import numpy as np
    import torch

    sys.path.insert(0, "/root")
    from grassmann.datasets.nerfies import load_nerfies
    from grassmann.fast_rasterizer import FastRasterConfig, fast_rasterize
    from grassmann.initialization import init_gaussians_from_points
    from grassmann.trainable import trainable_from_params

    DTYPE = torch.float32
    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    scene_dir = _ensure_scene_unpacked(scene)
    ds = load_nerfies(scene_dir, image_scale=image_scale, allow_distortion=True)
    print(f"Loaded {scene} scale={image_scale}: T={ds.T} H={ds.H} W={ds.W}")
    val_idx = list(range(2, ds.T, 4))

    state = torch.load(f"/checkpoints/{bugf_ckpt}", map_location="cpu",
                       weights_only=False)["model_state_dict"]
    n_saved = state["n_raw"].shape[0]
    sh_degree = 0
    if "sh_rest" in state:
        K_rest = state["sh_rest"].shape[1]
        sh_degree = int(round(((K_rest + 1) ** 0.5))) - 1
    pts = ds.points3D[:n_saved] if n_saved <= ds.N_points else ds.points3D
    if pts.shape[0] < n_saved:
        pad = pts[-1:].repeat(n_saved - pts.shape[0], 1)
        pts = torch.cat([pts, pad], dim=0)
    params0 = init_gaussians_from_points(
        pts, torch.zeros(n_saved, dtype=torch.float64), ds.cameras_per_frame,
        sigma_init_sq=0.02, sigma_k_pixel=1.0, sigma_k_temporal=0.0,
    )
    model = trainable_from_params(params0, dtype=DTYPE, device=device, sh_degree=sh_degree)
    model.load_state_dict(state, strict=True)
    print(f"Loaded ckpt: N={model.N} sh={sh_degree} H={ds.H} W={ds.W}")

    bg = torch.zeros(3, dtype=DTYPE, device=device)
    cfg = FastRasterConfig(sigma_3d_blur=1e-4, sh_degree=sh_degree)

    # Pre-compute model params once (shared across all frames; mimics rendering loop).
    with torch.no_grad():
        params_now = model.forward()

    @torch.no_grad()
    def _render_one(f_idx: int):
        cam = ds.cameras_per_frame[f_idx]
        t = float(ds.times[f_idx])
        return fast_rasterize(params_now, t, cam, ds.H, ds.W, background=bg, config=cfg)

    # Warmup — JIT, allocator, kernel-compile.
    print(f"Warmup ({warmup_iters} iters)...")
    for _ in range(warmup_iters):
        _ = _render_one(val_idx[0])
    torch.cuda.synchronize()

    # Time each val frame K times.
    print(f"Timing {len(val_idx)} val frames × {repeats_per_frame} repeats ...")
    times_ms = []
    for f_idx in val_idx:
        for _ in range(repeats_per_frame):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = _render_one(f_idx)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    arr = np.array(times_ms)
    print(f"\n=== Render-speed benchmark ===")
    print(f"  N (Gaussians)     = {model.N}")
    print(f"  resolution        = {ds.W} × {ds.H}")
    print(f"  warmup iters      = {warmup_iters}")
    print(f"  total timed       = {len(arr)} renders ({len(val_idx)}×{repeats_per_frame})")
    print(f"  median ms/frame   = {np.median(arr):.3f}")
    print(f"  mean ms/frame     = {arr.mean():.3f}")
    print(f"  p05 / p95         = {np.percentile(arr, 5):.3f} / {np.percentile(arr, 95):.3f}")
    print(f"  throughput        = {1000.0 / np.median(arr):.1f} FPS (median)")
    print(f"==============================")


@app.local_entrypoint()
def main(
    bugf_ckpt: str = "nerfies-slice-banana-spatial_slice-14000it-bugF-uncapped-ssim/trained_nerfies_spatial_slice.pt",
    scene: str = "slice-banana",
    image_scale: int = 4,
    warmup_iters: int = 30,
    repeats_per_frame: int = 10,
):
    bench.remote(
        bugf_ckpt=bugf_ckpt,
        scene=scene,
        image_scale=image_scale,
        warmup_iters=warmup_iters,
        repeats_per_frame=repeats_per_frame,
    )
