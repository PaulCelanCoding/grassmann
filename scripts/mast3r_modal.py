"""
Modal app for MASt3R-SfM dense initialization.

Workflow:
  1. modal run scripts/mast3r_modal.py::run_sfm --scene slice-banana
       → loads scene frames from gs-mono volume,
         runs MASt3R-SfM (sparse_global_alignment),
         saves dense points + colors + per-frame poses to
         /checkpoints/mast3r/<scene>/{points3d,colors,poses_c2w}.npy
  2. The training pipeline (scripts/train_mono.py --init_points_path)
       loads points3d.npy as the init point cloud, replacing the
       dataset's bundled points.

The image install is the main risk: MASt3R requires a recursive clone (with
dust3r submodule), Cython-compiled ASMK retrieval module, and ~3 GB of model
checkpoints. We bake everything into the image so cold starts are cheap.

Compute scaling: a complete graph on 330 frames is 330*329 = ~109 k pairs at
~50 ms each ≈ 90 min. We use a windowed temporal pair scheme (`make_pairs`
with `scene_graph='swin-N'`) to stay sub-10-min.
"""
from __future__ import annotations

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent

# Model checkpoint URLs (Naver Labs).
WEIGHTS = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
RETR_W = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
RETR_C = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl"
NAVER_BASE = "https://download.europe.naverlabs.com/ComputerVision/MASt3R"

# MASt3R repo expects: pip deps, recursive submodule clone, ASMK install,
# checkpoints in /opt/mast3r/checkpoints/.
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
        add_python="3.11",
    )
    .apt_install("git", "build-essential", "wget", "ninja-build")
    .pip_install(
        # MASt3R + dust3r dependencies (consolidated).
        "numpy<2",          # mast3r/dust3r expect numpy 1.x
        "matplotlib",
        "tqdm",
        "pillow",
        "scipy",
        "trimesh",
        "einops",
        "roma",
        "huggingface_hub",
        "opencv-python-headless",
        "scikit-learn",
        "cython",
        # faiss is a build dep of ASMK's hamming extension — must be installed
        # *before* the asmk pip install below.
        "faiss-cpu",
    )
    .run_commands(
        # MASt3R + dust3r submodule.
        "git clone --recursive https://github.com/naver/mast3r /opt/mast3r",
        "cd /opt/mast3r && git checkout mast3r_sfm && git submodule update --init --recursive",
        # ASMK retrieval module (required for sparse_global_alignment).
        "git clone https://github.com/jenicek/asmk /opt/asmk",
        "cd /opt/asmk/cython && cythonize *.pyx",
        "cd /opt/asmk && pip install .",
    )
    .run_commands(
        # Checkpoints (~3 GB total — bake into image to avoid re-download per run).
        "mkdir -p /opt/mast3r/checkpoints",
        f"wget -q {NAVER_BASE}/{WEIGHTS} -O /opt/mast3r/checkpoints/{WEIGHTS}",
        f"wget -q {NAVER_BASE}/{RETR_W} -O /opt/mast3r/checkpoints/{RETR_W}",
        f"wget -q {NAVER_BASE}/{RETR_C} -O /opt/mast3r/checkpoints/{RETR_C}",
    )
    .env({
        "PYTHONPATH": "/opt/mast3r:/opt/mast3r/dust3r:/opt/mast3r/dust3r/croco",
    })
)

app = modal.App("grassmann-mast3r", image=image)

mono_vol = modal.Volume.from_name("gs-mono", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("gs-checkpoints", create_if_missing=True)
VOLUMES = {"/data": mono_vol, "/checkpoints": ckpt_vol}
GPU = "L4"


@app.function(gpu=GPU, volumes=VOLUMES, timeout=2 * 3600)
def import_check() -> None:
    """Smoke test: image builds, MASt3R imports cleanly, checkpoints exist."""
    import os
    import sys

    sys.path.insert(0, "/opt/mast3r")
    sys.path.insert(0, "/opt/mast3r/dust3r")

    from mast3r.model import AsymmetricMASt3R  # noqa: F401
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment  # noqa: F401
    from mast3r.image_pairs import make_pairs  # noqa: F401
    from dust3r.utils.image import load_images  # noqa: F401
    import asmk  # noqa: F401

    for f in (WEIGHTS, RETR_W, RETR_C):
        p = f"/opt/mast3r/checkpoints/{f}"
        assert os.path.isfile(p), f"missing checkpoint {p}"
        size_gb = os.path.getsize(p) / (1024 ** 3)
        print(f"  {p}: {size_gb:.2f} GB")
    print("MASt3R import + checkpoints OK.")


@app.function(gpu=GPU, volumes=VOLUMES, timeout=2 * 3600)
def run_sfm(
    scene: str,
    image_subdir: str = "rgb/4x",
    subsample: int = 1,
    pair_window: int = 8,
    niter1: int = 300,
    niter2: int = 300,
) -> None:
    """Run MASt3R-SfM on a scene's frames; save dense points to the volume.

    Args:
        scene: scene name (must exist as /data/<scene>/ on gs-mono).
        image_subdir: relative path to RGB frames inside /data/<scene>/.
        subsample: keep every Nth frame (default 1 = all).
        pair_window: temporal window size for `swin-N` pair graph; reduces the
                     pair count from O(F²) to O(F·N).
        niter1, niter2: SfM optimization iterations (defaults from MASt3R README).
    """
    import glob
    import os
    import sys
    import time

    import numpy as np
    import torch

    sys.path.insert(0, "/opt/mast3r")
    sys.path.insert(0, "/opt/mast3r/dust3r")

    from mast3r.model import AsymmetricMASt3R
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from mast3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images

    # Scene unpack (matches train_modal.py convention).
    scene_dir = _ensure_scene_unpacked(scene)
    image_dir = os.path.join(scene_dir, image_subdir)
    image_paths = sorted(
        glob.glob(f"{image_dir}/*.png") + glob.glob(f"{image_dir}/*.jpg")
    )
    if subsample > 1:
        image_paths = image_paths[::subsample]
    print(f"Loading {len(image_paths)} frames from {image_dir} ...", flush=True)
    if not image_paths:
        raise RuntimeError(f"no frames in {image_dir}")

    device = "cuda"
    weights = f"/opt/mast3r/checkpoints/{WEIGHTS}"
    model = AsymmetricMASt3R.from_pretrained(weights).to(device)

    images = load_images(image_paths, size=512)

    # Pair graph: 'swin-N' = sliding window of N frames (each frame paired with
    # next/prev N within the temporal sequence). For 330 frames + window 8,
    # ~330·16 = ~5280 pairs vs ~109k for the complete graph.
    pairs = make_pairs(
        images,
        scene_graph=f"swin-{pair_window}",
        prefilter=None,
        symmetrize=True,
    )
    print(f"  pair graph: {len(pairs)} pairs (swin-{pair_window})", flush=True)

    out_dir = f"/checkpoints/mast3r/{scene}"
    os.makedirs(out_dir, exist_ok=True)

    # Sparse global alignment (the SfM-style optimization).
    t0 = time.perf_counter()
    sga_scene = sparse_global_alignment(
        image_paths, pairs, out_dir, model,
        lr1=0.07, niter1=niter1, lr2=0.014, niter2=niter2,
        device=device, opt_depth=False,  # depth comes from the model directly
    )
    print(f"  SfM alignment: {time.perf_counter() - t0:.1f}s", flush=True)

    # Extract dense pointmap + colors.
    t0 = time.perf_counter()
    pts3d, depthmaps, confs = sga_scene.get_dense_pts3d(clean_depth=True)
    masks = [c > 1.5 for c in confs]                   # MASt3R confidence threshold

    points_world: list[np.ndarray] = []
    colors_all: list[np.ndarray] = []
    for img, p3d, mask in zip(images, pts3d, masks):
        # img['img'] is [-1, 1] in shape (1, 3, H, W) or (3, H, W) depending on
        # dust3r's load_images version. Squeeze any leading batch dim, then HWC.
        rgb_t = img["img"]
        if rgb_t.dim() == 4:                                # (1, 3, H, W)
            rgb_t = rgb_t.squeeze(0)
        rgb_chw = rgb_t.cpu().numpy() * 0.5 + 0.5           # (3, H, W) in [0, 1]
        rgb = rgb_chw.transpose(1, 2, 0).reshape(-1, 3)     # (HW, 3)
        p = p3d.detach().cpu().numpy().reshape(-1, 3)
        m = mask.cpu().numpy().reshape(-1).astype(bool)
        if p.shape[0] != rgb.shape[0]:
            # Pointmap may be downsampled vs the source RGB — fall back to gray.
            colors_all.append(np.full((m.sum(), 3), 0.5, dtype=np.float32))
        else:
            colors_all.append(rgb[m])
        points_world.append(p[m])

    points_world_arr = np.concatenate(points_world, axis=0).astype(np.float32)
    colors_arr = np.concatenate(colors_all, axis=0).astype(np.float32)
    print(f"  pointmap extract: {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"  total points: {points_world_arr.shape[0]:,}", flush=True)

    np.save(f"{out_dir}/points3d.npy", points_world_arr)
    np.save(f"{out_dir}/colors.npy", colors_arr)

    # Per-frame poses (cam-to-world) — useful for re-projection sanity later.
    cam2w = np.stack([sga_scene.get_im_poses().detach().cpu().numpy()], axis=0)
    np.save(f"{out_dir}/poses_c2w.npy", cam2w[0].astype(np.float32))

    print(f"Saved to {out_dir}/{{points3d,colors,poses_c2w}}.npy", flush=True)
    ckpt_vol.commit()


def _ensure_scene_unpacked(scene: str) -> str:
    """Reuse the same unzip-on-demand logic as train_modal.py."""
    import os
    import zipfile

    scene_dir = f"/data/{scene}"
    if os.path.isdir(scene_dir):
        return scene_dir
    zip_path = f"/data/{scene}.zip"
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(
            f"Neither {scene_dir!r} (dir) nor {zip_path!r} (zip) on gs-mono. Upload first."
        )
    print(f"  unpacking {zip_path} -> /data/...", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall("/data")
    if not os.path.isdir(scene_dir):
        for entry in os.listdir("/data"):
            cand = f"/data/{entry}"
            if os.path.isdir(cand) and os.path.exists(f"{cand}/dataset.json"):
                return cand
        raise RuntimeError(f"unzipped {zip_path} but no scene dir under /data/")
    mono_vol.commit()
    return scene_dir


@app.local_entrypoint()
def main(
    cmd: str = "import_check",
    scene: str = "slice-banana",
    image_subdir: str = "rgb/4x",
    subsample: int = 1,
    pair_window: int = 8,
    niter1: int = 300,
    niter2: int = 300,
):
    """
    --cmd import_check: build image + verify imports + checkpoint sizes (no SfM).
    --cmd run_sfm:      run MASt3R-SfM on the scene; save points to volume.
    """
    if cmd == "import_check":
        import_check.remote()
    elif cmd == "run_sfm":
        run_sfm.remote(
            scene=scene,
            image_subdir=image_subdir,
            subsample=subsample,
            pair_window=pair_window,
            niter1=niter1,
            niter2=niter2,
        )
    else:
        raise SystemExit(f"unknown --cmd {cmd!r}; expected import_check|run_sfm")
