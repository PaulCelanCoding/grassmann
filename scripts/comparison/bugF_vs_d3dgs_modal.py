"""
Modal entry: side-by-side comparison of our Bug-F ckpt vs Deformable3DGS
ckpt on slice-banana (14k iters, scale 4, deformable_interp split).

What it does (single function, single GPU run):
  1. Loads NeRFies slice-banana at image_scale=4 (deformable_interp split:
     train=ids[::4]=83 frames, val=ids[2::4]=82 frames).
  2. Loads our Bug-F ckpt and renders every train+val frame via fast_rasterize.
  3. Reads D3DGS's pre-rendered PNGs from
       /checkpoints/<d3dgs_dir>/{train,test}/ours_14000/renders/<idx>.png
     (D3DGS test idx corresponds to position-in-val-list; train idx to
      position-in-train-list, both per the same deformable_interp split.)
     Resizes if D3DGS dims ≠ ours.
  4. Computes per-frame PSNR/L1 for both methods against the SAME GT
     (loaded by our reader → apples-to-apples vs our eval pipeline).
  5. Saves JSON with all per-frame stats + summary aggregates.
  6. Composes triptych PNGs (GT | Bug-F | D3DGS, with per-frame PSNR labels)
     for: 5 worst Bug-F val frames, 5 best Bug-F val frames, and 3 worst
     train frames.
  7. Reports train PSNR + val PSNR for both methods.

Usage:
  modal run scripts/comparison/bugF_vs_d3dgs_modal.py \
      --bugf-ckpt nerfies-slice-banana-spatial_slice-14000it-bug-F-aniso/trained_nerfies_spatial_slice.pt \
      --d3dgs-dir deformable-slice-banana-14000it-baseline-r2-scale4-2026-05-11

Outputs land in /checkpoints/comparisons/bugF_vs_d3dgs_14k/:
  - per_frame.json         (train + val per-frame stats)
  - summary.json           (aggregates)
  - figure_overview.png    (per-frame curve + Δ histogram + scatter)
  - triptychs/<split>/<idx>.png  (GT | Bug-F | D3DGS panels)

Pull locally with:
  modal volume get gs-checkpoints comparisons/bugF_vs_d3dgs_14k ./out
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent.parent

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

app = modal.App("grassmann-bugF-vs-d3dgs", image=image)

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
        raise FileNotFoundError(
            f"Neither {scene_dir!r} nor {zip_path!r} exists on the gs-mono volume."
        )
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall("/data")
    return scene_dir


@app.function(gpu="L4", volumes=VOLUMES, timeout=2 * 3600)
def compare(
    bugf_ckpt: str,
    d3dgs_dir: str,
    scene: str = "slice-banana",
    image_scale: int = 4,
    sigma_3d_blur: float = 1e-4,
    iters_tag: str = "ours_14000",
    out_dir_rel: str = "comparisons/bugF_vs_d3dgs_14k",
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
    print(f"Loaded {scene} at scale={image_scale}: T={ds.T} H={ds.H} W={ds.W} N_pts={ds.N_points}")

    # deformable_interp split: train=ids[::4], val=ids[2::4]
    train_idx = list(range(0, ds.T, 4))
    val_idx   = list(range(2, ds.T, 4))
    print(f"Split: train={len(train_idx)} val={len(val_idx)}")

    # ---- Load Bug-F ckpt ----
    ckpt_path = f"/checkpoints/{bugf_ckpt}"
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
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
    times0 = torch.zeros(n_saved, dtype=torch.float64)
    params0 = init_gaussians_from_points(
        pts_for_init, times0, ds.cameras_per_frame,
        sigma_init_sq=0.02, sigma_k_pixel=1.0, sigma_k_temporal=0.0,
    )
    model = trainable_from_params(params0, dtype=DTYPE, device=device, sh_degree=sh_degree)
    model.load_state_dict(state, strict=True)
    print(f"Loaded Bug-F: N={model.N} sh={sh_degree}")

    bg = torch.zeros(3, dtype=DTYPE, device=device)
    raster_cfg = FastRasterConfig(sigma_3d_blur=sigma_3d_blur, sh_degree=sh_degree)

    # ---- Sanity-check D3DGS layout + dims ----
    d3_root = f"/checkpoints/{d3dgs_dir}"
    d3_test = f"{d3_root}/test/{iters_tag}/renders"
    d3_train = f"{d3_root}/train/{iters_tag}/renders"
    for p in (d3_test, d3_train):
        if not os.path.isdir(p):
            raise FileNotFoundError(f"D3DGS dir missing: {p}")
    probe = PILImage.open(f"{d3_test}/00000.png")
    print(f"D3DGS probe (test/00000.png): size={probe.size}  ours (W,H)=({ds.W},{ds.H})")
    needs_resize = probe.size != (ds.W, ds.H)
    if needs_resize:
        print(f"  D3DGS images will be bilinearly resized to ({ds.W},{ds.H})")

    def _load_d3(idx: int, split: str) -> torch.Tensor:
        root = d3_test if split == "val" else d3_train
        p = f"{root}/{idx:05d}.png"
        im = PILImage.open(p).convert("RGB")
        if needs_resize:
            im = im.resize((ds.W, ds.H), PILImage.BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0       # (H, W, 3)
        return torch.from_numpy(a).to(device)

    def _load_d3_gt(idx: int, split: str) -> torch.Tensor:
        # D3DGS saved its own preprocessed GT alongside renders. Use it as
        # an alternative metric reference to reproduce their published PSNR.
        root_renders = d3_test if split == "val" else d3_train
        gt_root = root_renders.replace("/renders", "/gt")
        p = f"{gt_root}/{idx:05d}.png"
        im = PILImage.open(p).convert("RGB")
        if needs_resize:
            im = im.resize((ds.W, ds.H), PILImage.BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0
        return torch.from_numpy(a).to(device)

    def _psnr(img: torch.Tensor, gt: torch.Tensor) -> float:
        mse = ((img - gt) ** 2).mean().clamp_min(1e-12)
        return float(-10.0 * torch.log10(mse))

    def _l1(img: torch.Tensor, gt: torch.Tensor) -> float:
        return float((img - gt).abs().mean())

    # LPIPS — D3DGS uses VGG backbone (matches their published numbers).
    # We also compute AlexNet (3DGS-paper convention) for completeness.
    import lpips
    lpips_vgg  = lpips.LPIPS(net="vgg").to(device).eval()
    lpips_alex = lpips.LPIPS(net="alex").to(device).eval()
    def _to_lpips(img_hwc: torch.Tensor) -> torch.Tensor:
        return (img_hwc.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0).clamp(-1.0, 1.0)
    @torch.no_grad()
    def _lpips_vgg(img: torch.Tensor, gt: torch.Tensor) -> float:
        return float(lpips_vgg(_to_lpips(img), _to_lpips(gt)).item())
    @torch.no_grad()
    def _lpips_alex(img: torch.Tensor, gt: torch.Tensor) -> float:
        return float(lpips_alex(_to_lpips(img), _to_lpips(gt)).item())

    # ---- Loop over both splits ----
    per_frame = []
    for split_name, idx_list in (("train", train_idx), ("val", val_idx)):
        print(f"\n=== {split_name} ({len(idx_list)} frames) ===")
        for k, f_idx in enumerate(idx_list):
            cam = ds.cameras_per_frame[f_idx]
            t = float(ds.times[f_idx])
            gt = ds.frame_loader(f_idx).to(device).float()
            if gt.max() > 1.5:
                gt = gt / 255.0
            with torch.no_grad():
                bugf_img = fast_rasterize(model.forward(), t, cam, ds.H, ds.W,
                                          background=bg, config=raster_cfg)
            d3_img = _load_d3(k, split_name)
            d3_gt  = _load_d3_gt(k, split_name)
            entry = {
                "split": split_name,
                "k": k,
                "frame": int(f_idx),
                "t": t,
                # apples-to-apples vs OUR GT
                "bugf_psnr": _psnr(bugf_img, gt),
                "bugf_l1":   _l1(bugf_img, gt),
                "bugf_lpips_vgg": _lpips_vgg(bugf_img, gt),
                "bugf_lpips_alex": _lpips_alex(bugf_img, gt),
                "d3dgs_psnr": _psnr(d3_img, gt),
                "d3dgs_l1":   _l1(d3_img, gt),
                "d3dgs_lpips_vgg": _lpips_vgg(d3_img, gt),
                "d3dgs_lpips_alex": _lpips_alex(d3_img, gt),
                # apples-to-apples vs D3DGS-saved GT (reproduces their metric)
                "bugf_psnr_d3gt":  _psnr(bugf_img, d3_gt),
                "bugf_lpips_vgg_d3gt": _lpips_vgg(bugf_img, d3_gt),
                "bugf_lpips_alex_d3gt": _lpips_alex(bugf_img, d3_gt),
                "d3dgs_psnr_d3gt": _psnr(d3_img,  d3_gt),
                "d3dgs_lpips_vgg_d3gt": _lpips_vgg(d3_img, d3_gt),
                "d3dgs_lpips_alex_d3gt": _lpips_alex(d3_img, d3_gt),
                # GT divergence between the two preprocessing pipelines
                "gt_gt_psnr": _psnr(gt, d3_gt),
            }
            per_frame.append(entry)
            if (k + 1) % 20 == 0:
                print(f"  {split_name} {k+1}/{len(idx_list)}  "
                      f"bugF={entry['bugf_psnr']:.2f}  d3dgs={entry['d3dgs_psnr']:.2f}")

    # ---- Aggregates ----
    def _avg(rows, key):
        return float(np.mean([r[key] for r in rows]))
    train_rows = [r for r in per_frame if r["split"] == "train"]
    val_rows   = [r for r in per_frame if r["split"] == "val"]
    summary = {
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        # PSNR aggregates: vs OUR GT (apples-to-apples in our pipeline)
        "train_bugf_psnr_avg":  _avg(train_rows, "bugf_psnr"),
        "train_d3dgs_psnr_avg": _avg(train_rows, "d3dgs_psnr"),
        "val_bugf_psnr_avg":    _avg(val_rows,   "bugf_psnr"),
        "val_d3dgs_psnr_avg":   _avg(val_rows,   "d3dgs_psnr"),
        # PSNR aggregates: vs D3DGS-saved GT (reproduces their published metric)
        "train_bugf_psnr_d3gt_avg":  _avg(train_rows, "bugf_psnr_d3gt"),
        "train_d3dgs_psnr_d3gt_avg": _avg(train_rows, "d3dgs_psnr_d3gt"),
        "val_bugf_psnr_d3gt_avg":    _avg(val_rows,   "bugf_psnr_d3gt"),
        "val_d3dgs_psnr_d3gt_avg":   _avg(val_rows,   "d3dgs_psnr_d3gt"),
        # LPIPS-VGG aggregates (matches D3DGS published metric) — vs OUR GT
        "train_bugf_lpips_vgg_avg":  _avg(train_rows, "bugf_lpips_vgg"),
        "train_d3dgs_lpips_vgg_avg": _avg(train_rows, "d3dgs_lpips_vgg"),
        "val_bugf_lpips_vgg_avg":    _avg(val_rows,   "bugf_lpips_vgg"),
        "val_d3dgs_lpips_vgg_avg":   _avg(val_rows,   "d3dgs_lpips_vgg"),
        # LPIPS-VGG aggregates — vs D3DGS GT (reproduces D3DGS published LPIPS)
        "train_bugf_lpips_vgg_d3gt_avg":  _avg(train_rows, "bugf_lpips_vgg_d3gt"),
        "train_d3dgs_lpips_vgg_d3gt_avg": _avg(train_rows, "d3dgs_lpips_vgg_d3gt"),
        "val_bugf_lpips_vgg_d3gt_avg":    _avg(val_rows,   "bugf_lpips_vgg_d3gt"),
        "val_d3dgs_lpips_vgg_d3gt_avg":   _avg(val_rows,   "d3dgs_lpips_vgg_d3gt"),
        # LPIPS-Alex aggregates (3DGS-paper convention) — vs OUR GT
        "train_bugf_lpips_alex_avg":  _avg(train_rows, "bugf_lpips_alex"),
        "train_d3dgs_lpips_alex_avg": _avg(train_rows, "d3dgs_lpips_alex"),
        "val_bugf_lpips_alex_avg":    _avg(val_rows,   "bugf_lpips_alex"),
        "val_d3dgs_lpips_alex_avg":   _avg(val_rows,   "d3dgs_lpips_alex"),
        # LPIPS-Alex aggregates — vs D3DGS GT
        "train_bugf_lpips_alex_d3gt_avg":  _avg(train_rows, "bugf_lpips_alex_d3gt"),
        "train_d3dgs_lpips_alex_d3gt_avg": _avg(train_rows, "d3dgs_lpips_alex_d3gt"),
        "val_bugf_lpips_alex_d3gt_avg":    _avg(val_rows,   "bugf_lpips_alex_d3gt"),
        "val_d3dgs_lpips_alex_d3gt_avg":   _avg(val_rows,   "d3dgs_lpips_alex_d3gt"),
        # GT divergence diagnostic
        "val_gt_gt_psnr_avg":   _avg(val_rows,   "gt_gt_psnr"),
        "train_gt_gt_psnr_avg": _avg(train_rows, "gt_gt_psnr"),
        # L1 aggregates
        "train_bugf_l1_avg":  _avg(train_rows, "bugf_l1"),
        "train_d3dgs_l1_avg": _avg(train_rows, "d3dgs_l1"),
        "val_bugf_l1_avg":    _avg(val_rows,   "bugf_l1"),
        "val_d3dgs_l1_avg":   _avg(val_rows,   "d3dgs_l1"),
    }
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k} = {v:.4f}" if isinstance(v, float) else f"  {k} = {v}")

    out_dir = f"/checkpoints/{out_dir_rel}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/triptychs/train", exist_ok=True)
    os.makedirs(f"{out_dir}/triptychs/val",   exist_ok=True)
    with open(f"{out_dir}/per_frame.json", "w") as f:
        json.dump(per_frame, f, indent=2)
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Triptychs ----
    def _save_triptych(row, gt_t, bugf_t, d3_t, out_path):
        fig, axes = plt.subplots(1, 3, figsize=(12, 5))
        axes[0].imshow(gt_t.detach().cpu().numpy().clip(0, 1))
        axes[0].set_title(f"GT  (f{row['frame']}, t={row['t']:.3f})"); axes[0].axis("off")
        axes[1].imshow(bugf_t.detach().cpu().numpy().clip(0, 1))
        axes[1].set_title(f"Bug-F (ours)   PSNR={row['bugf_psnr']:.2f}"); axes[1].axis("off")
        axes[2].imshow(d3_t.detach().cpu().numpy().clip(0, 1))
        axes[2].set_title(f"Deformable3DGS  PSNR={row['d3dgs_psnr']:.2f}"); axes[2].axis("off")
        delta = row["bugf_psnr"] - row["d3dgs_psnr"]
        color = "green" if delta > 0 else "C3"
        plt.suptitle(f"{row['split']} frame {row['frame']}    "
                     f"Δ(ours−D3DGS) = {delta:+.2f} dB",
                     color=color, fontsize=12)
        plt.tight_layout()
        plt.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close()

    # pick K worst & best val by bugF PSNR, K worst train by bugF PSNR
    val_sorted = sorted(val_rows, key=lambda r: r["bugf_psnr"])
    train_sorted = sorted(train_rows, key=lambda r: r["bugf_psnr"])
    chosen = []
    for r in val_sorted[:5]:           chosen.append((r, "val",   "worst"))
    for r in val_sorted[-5:][::-1]:    chosen.append((r, "val",   "best"))
    for r in train_sorted[:3]:         chosen.append((r, "train", "worst"))

    for row, split, tag in chosen:
        cam = ds.cameras_per_frame[row["frame"]]
        gt = ds.frame_loader(row["frame"]).to(device).float()
        if gt.max() > 1.5:
            gt = gt / 255.0
        with torch.no_grad():
            bugf_img = fast_rasterize(model.forward(), row["t"], cam, ds.H, ds.W,
                                      background=bg, config=raster_cfg)
        d3_img = _load_d3(row["k"], split)
        out_p = f"{out_dir}/triptychs/{split}/{tag}_f{row['frame']:04d}.png"
        _save_triptych(row, gt, bugf_img, d3_img, out_p)
        print(f"  triptych → {out_p}")

    # ---- Overview figure: per-frame curves + Δ hist + scatter ----
    val_fr = np.array([r["frame"] for r in val_rows])
    val_t  = np.array([r["t"]     for r in val_rows])
    val_b  = np.array([r["bugf_psnr"]  for r in val_rows])
    val_d  = np.array([r["d3dgs_psnr"] for r in val_rows])
    tr_fr  = np.array([r["frame"] for r in train_rows])
    tr_t   = np.array([r["t"]     for r in train_rows])
    tr_b   = np.array([r["bugf_psnr"]  for r in train_rows])
    tr_d   = np.array([r["d3dgs_psnr"] for r in train_rows])

    fig, ax = plt.subplots(3, 1, figsize=(13, 11))
    ax[0].plot(tr_fr, tr_b, "o-", color="C0", ms=3, lw=1.0, label=f"Bug-F train avg={tr_b.mean():.2f}")
    ax[0].plot(tr_fr, tr_d, "s-", color="C3", ms=3, lw=1.0, label=f"D3DGS train avg={tr_d.mean():.2f}")
    ax[0].set_title("TRAIN per-frame PSNR"); ax[0].set_ylabel("PSNR (dB)"); ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(val_fr, val_b, "o-", color="C0", ms=4, lw=1.2, label=f"Bug-F val avg={val_b.mean():.2f}")
    ax[1].plot(val_fr, val_d, "s-", color="C3", ms=4, lw=1.2, label=f"D3DGS val avg={val_d.mean():.2f}")
    ax[1].set_title("VAL per-frame PSNR"); ax[1].set_ylabel("PSNR (dB)"); ax[1].legend()
    ax[1].grid(alpha=0.3)
    delta_val = val_b - val_d
    n_win = int((delta_val > 0).sum())
    ax[2].hist(delta_val, bins=20, edgecolor="black", color="lightsteelblue")
    ax[2].axvline(0, color="k", lw=1, ls="--")
    ax[2].axvline(delta_val.mean(), color="C3", lw=1.5,
                  label=f"mean Δ={delta_val.mean():+.2f}, ours-wins={n_win}/{len(val_rows)}")
    ax[2].set_xlabel("Δ PSNR = ours − D3DGS (val, dB)")
    ax[2].set_ylabel("# val frames"); ax[2].legend(); ax[2].grid(alpha=0.3)
    ax[2].set_title("VAL Δ-distribution")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/figure_overview.png", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  overview → {out_dir}/figure_overview.png")

    ckpt_vol.commit()
    print(f"\nPull locally with:")
    print(f"  modal volume get gs-checkpoints {out_dir_rel} ./out_bugF_vs_d3dgs")


@app.local_entrypoint()
def main(
    bugf_ckpt: str = "nerfies-slice-banana-spatial_slice-14000it-bug-F-aniso/trained_nerfies_spatial_slice.pt",
    d3dgs_dir: str = "deformable-slice-banana-14000it-baseline-r2-scale4-2026-05-11",
    scene: str = "slice-banana",
    image_scale: int = 4,
    out_dir: str = "comparisons/bugF_vs_d3dgs_14k",
    sigma_3d_blur: float = 1e-4,
):
    compare.remote(
        bugf_ckpt=bugf_ckpt,
        d3dgs_dir=d3dgs_dir,
        scene=scene,
        image_scale=image_scale,
        sigma_3d_blur=sigma_3d_blur,
        out_dir_rel=out_dir,
    )
