"""
Modal entry: 4-panel side-by-side comparison (GT | Bug-F baseline |
maxAR200+SSIM | D3DGS) for a few selected val frames.

Renders both Bug-F ckpts fresh through our fast_rasterize, loads pre-rendered
D3DGS PNGs from the deformable-baseline volume dir, and saves a 4-panel
PNG per frame.

Usage:
    modal run scripts/quadtych_compare_modal.py \
        --bugf-baseline-ckpt nerfies-slice-banana-spatial_slice-14000it-bug-F-aniso/trained_nerfies_spatial_slice.pt \
        --bugf-best-ckpt    nerfies-slice-banana-spatial_slice-14000it-bugF-maxAR200-ssim/trained_nerfies_spatial_slice.pt \
        --d3dgs-dir         deformable-slice-banana-14000it-baseline-r2-scale4-2026-05-11 \
        --out-dir-rel       comparisons/quadtych_AR200_SSIM
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
    .pip_install("numpy", "matplotlib", "pillow", "tqdm", "lpips", "torchvision")
    .pip_install(
        "git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git",
        gpu="L4",
    )
    .add_local_python_source("grassmann")
    .add_local_dir(str(REPO / "scripts"), remote_path="/root/scripts")
)

app = modal.App("grassmann-quadtych-compare", image=image)

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
def quadtych(
    bugf_baseline_ckpt: str,
    bugf_best_ckpt: str,
    d3dgs_dir: str,
    scene: str = "slice-banana",
    image_scale: int = 4,
    iters_tag: str = "ours_14000",
    out_dir_rel: str = "comparisons/quadtych",
    n_worst: int = 6,
    n_best: int = 3,
) -> None:
    import json, os, sys
    import numpy as np
    import torch
    from PIL import Image as PILImage
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, "/root")
    from grassmann.datasets.nerfies import load_nerfies
    from grassmann.fast_rasterizer import FastRasterConfig, fast_rasterize
    from grassmann.initialization import init_gaussians_from_points
    from grassmann.trainable import trainable_from_params

    DTYPE = torch.float32
    device = "cuda"

    scene_dir = _ensure_scene_unpacked(scene)
    ds = load_nerfies(scene_dir, image_scale=image_scale, allow_distortion=True)
    print(f"Loaded {scene} scale={image_scale}: T={ds.T} H={ds.H} W={ds.W}")

    val_idx = list(range(2, ds.T, 4))   # deformable_interp val split

    def _load_ckpt(rel_path: str):
        full = f"/checkpoints/{rel_path}"
        state = torch.load(full, map_location="cpu", weights_only=False)["model_state_dict"]
        n_saved = state["n_raw"].shape[0]
        if "sh_rest" in state:
            K_rest = state["sh_rest"].shape[1]
            sh_degree = int(round(((K_rest + 1) ** 0.5))) - 1
        else:
            sh_degree = 0
        pts_for_init = ds.points3D[:n_saved] if n_saved <= ds.N_points else ds.points3D
        if pts_for_init.shape[0] < n_saved:
            pad = pts_for_init[-1:].repeat(n_saved - pts_for_init.shape[0], 1)
            pts_for_init = torch.cat([pts_for_init, pad], dim=0)
        params0 = init_gaussians_from_points(
            pts_for_init, torch.zeros(n_saved, dtype=torch.float64),
            ds.cameras_per_frame, sigma_init_sq=0.02,
            sigma_k_pixel=1.0, sigma_k_temporal=0.0,
        )
        model = trainable_from_params(params0, dtype=DTYPE, device=device, sh_degree=sh_degree)
        model.load_state_dict(state, strict=True)
        return model, sh_degree

    model_base, sh_base = _load_ckpt(bugf_baseline_ckpt)
    model_best, sh_best = _load_ckpt(bugf_best_ckpt)
    print(f"baseline N={model_base.N} sh={sh_base} | best N={model_best.N} sh={sh_best}")

    bg = torch.zeros(3, dtype=DTYPE, device=device)
    cfg_base = FastRasterConfig(sigma_3d_blur=1e-4, sh_degree=sh_base)
    cfg_best = FastRasterConfig(sigma_3d_blur=1e-4, sh_degree=sh_best)

    d3_test = f"/checkpoints/{d3dgs_dir}/test/{iters_tag}/renders"
    probe = PILImage.open(f"{d3_test}/00000.png")
    needs_resize = probe.size != (ds.W, ds.H)
    print(f"D3DGS probe size={probe.size}  ours=({ds.W},{ds.H})  resize={needs_resize}")

    def _load_d3(k):
        im = PILImage.open(f"{d3_test}/{k:05d}.png").convert("RGB")
        if needs_resize:
            im = im.resize((ds.W, ds.H), PILImage.BILINEAR)
        return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).to(device)

    import lpips
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    def _to(t): return (t.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0).clamp(-1.0, 1.0)
    @torch.no_grad()
    def _lp(a, b): return float(lpips_fn(_to(a), _to(b)).item())
    def _psnr(a, b): return float(-10.0 * torch.log10(((a - b) ** 2).mean().clamp_min(1e-12)))

    # First pass: per-frame metrics for the BEST ckpt (so we can rank worst/best).
    rows = []
    print("Scoring val frames ...")
    for k, f_idx in enumerate(val_idx):
        cam = ds.cameras_per_frame[f_idx]
        t = float(ds.times[f_idx])
        gt = ds.frame_loader(f_idx).to(device).float()
        if gt.max() > 1.5: gt = gt / 255.0
        with torch.no_grad():
            base_im = fast_rasterize(model_base.forward(), t, cam, ds.H, ds.W, background=bg, config=cfg_base)
            best_im = fast_rasterize(model_best.forward(), t, cam, ds.H, ds.W, background=bg, config=cfg_best)
        d3 = _load_d3(k)
        rows.append({
            "k": k, "f_idx": int(f_idx), "t": t,
            "base_psnr": _psnr(base_im, gt), "base_lpips": _lp(base_im, gt),
            "best_psnr": _psnr(best_im, gt), "best_lpips": _lp(best_im, gt),
            "d3_psnr":   _psnr(d3, gt),      "d3_lpips":   _lp(d3, gt),
        })

    # Rank by BEST-ckpt LPIPS gain over baseline (the most-improved frames),
    # plus worst BEST LPIPS frames (still-blurry frames the recipe helps least).
    delta = sorted(rows, key=lambda r: r["base_lpips"] - r["best_lpips"], reverse=True)
    most_improved = delta[:n_best]
    worst_best   = sorted(rows, key=lambda r: r["best_lpips"], reverse=True)[:n_worst]

    out_dir = f"/checkpoints/{out_dir_rel}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/per_frame_metrics.json", "w") as f:
        json.dump(rows, f, indent=2)

    def _save_4panel(row, tag):
        cam = ds.cameras_per_frame[row["f_idx"]]
        t = row["t"]
        gt = ds.frame_loader(row["f_idx"]).to(device).float()
        if gt.max() > 1.5: gt = gt / 255.0
        with torch.no_grad():
            base_im = fast_rasterize(model_base.forward(), t, cam, ds.H, ds.W, background=bg, config=cfg_base)
            best_im = fast_rasterize(model_best.forward(), t, cam, ds.H, ds.W, background=bg, config=cfg_best)
        d3 = _load_d3(row["k"])

        fig, ax = plt.subplots(1, 4, figsize=(18, 5.4))
        for a, im, title in zip(
            ax,
            [gt, base_im, best_im, d3],
            ["GT",
             f"Bug-F baseline\nPSNR={row['base_psnr']:.2f}  LPIPS={row['base_lpips']:.3f}",
             f"maxAR200+SSIM (ours)\nPSNR={row['best_psnr']:.2f}  LPIPS={row['best_lpips']:.3f}",
             f"Deformable-3DGS\nPSNR={row['d3_psnr']:.2f}  LPIPS={row['d3_lpips']:.3f}"],
        ):
            a.imshow(im.detach().cpu().numpy().clip(0, 1))
            a.set_title(title, fontsize=11); a.axis("off")
        plt.suptitle(f"{tag} | val frame {row['f_idx']}  t={row['t']:.3f}", fontsize=12)
        plt.tight_layout()
        out_path = f"{out_dir}/{tag}_f{row['f_idx']:04d}.png"
        plt.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  wrote {out_path}")

    for r in most_improved:
        _save_4panel(r, "most_improved")
    for r in worst_best:
        _save_4panel(r, "still_worst")

    ckpt_vol.commit()
    print(f"\nPull locally with:\n  modal volume get gs-checkpoints {out_dir_rel} ./out_quadtych")


@app.local_entrypoint()
def main(
    bugf_baseline_ckpt: str = "nerfies-slice-banana-spatial_slice-14000it-bug-F-aniso/trained_nerfies_spatial_slice.pt",
    bugf_best_ckpt: str    = "nerfies-slice-banana-spatial_slice-14000it-bugF-maxAR200-ssim/trained_nerfies_spatial_slice.pt",
    d3dgs_dir: str         = "deformable-slice-banana-14000it-baseline-r2-scale4-2026-05-11",
    scene: str             = "slice-banana",
    out_dir: str           = "comparisons/quadtych_AR200_SSIM",
):
    quadtych.remote(
        bugf_baseline_ckpt=bugf_baseline_ckpt,
        bugf_best_ckpt=bugf_best_ckpt,
        d3dgs_dir=d3dgs_dir,
        scene=scene,
        out_dir_rel=out_dir,
    )
