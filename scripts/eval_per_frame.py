"""Per-frame PSNR + aggregate spatial error map on a saved checkpoint.

Designed to run inside the Modal training image (CUDA fast rasterizer).
Loads a checkpoint, evaluates on val frames using the same split convention
as training, computes per-frame PSNR/L1 + an accumulated absolute-error
map, and saves PNGs + a JSON report. Outputs go to
/checkpoints/<ckpt_dir>/per_frame_diag/.

Usage (intended via `modal run train_modal.py --cmd eval_per_frame`):
    python scripts/eval_per_frame.py \\
        --dataset nerfies --scene_dir /data/slice-banana \\
        --ckpt /checkpoints/.../trained_nerfies.pt \\
        --output_dir /checkpoints/.../per_frame_diag \\
        --image_scale 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from grassmann.datasets import load_monocular  # noqa: E402
from grassmann.fast_rasterizer import FastRasterConfig, fast_rasterize  # noqa: E402
from grassmann.initialization import init_gaussians_from_points  # noqa: E402
from grassmann.trainable import trainable_from_params  # noqa: E402

DTYPE = torch.float32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("nerfies", "dycheck"), required=True)
    ap.add_argument("--scene_dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--image_scale", type=int, default=4)
    ap.add_argument("--split", type=str, default=None)
    ap.add_argument("--split_convention", choices=("val_stride", "deformable_interp"),
                    default="deformable_interp")
    ap.add_argument("--allow_distortion", action="store_true", default=True)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--sigma_3d_blur", type=float, default=1e-4)
    ap.add_argument("--lpips_net", default="alex",
                    help="LPIPS backbone (alex|vgg). 'none' disables (e.g. when "
                         "the `lpips` package isn't in the image).")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.dataset} from {args.scene_dir} (scale={args.image_scale})")
    ds = load_monocular(args.dataset, args.scene_dir,
                        image_scale=args.image_scale, split=args.split,
                        allow_distortion=args.allow_distortion)
    print(f"  T={ds.T}, points={ds.N_points}, H={ds.H}, W={ds.W}")

    # Construct val indices same way training does.
    if ds.val_indices:
        val_indices = list(ds.val_indices)
    elif args.split_convention == "deformable_interp":
        all_ids = list(range(ds.T))
        val_indices = all_ids[2::4]
    else:
        val_indices = list(range(0, ds.T, 4))
    print(f"  val frames: {len(val_indices)} (e.g. {val_indices[:5]}...)")

    # Load checkpoint.
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    n_saved = state["n_raw"].shape[0]
    if "sh_rest" in state:
        K_rest = state["sh_rest"].shape[1]
        sh_degree = int(round(((K_rest + 1) ** 0.5))) - 1
    else:
        sh_degree = 0
    print(f"  ckpt: N={n_saved}, sh_degree={sh_degree}")

    # Build placeholder model with matching shape.
    pts_for_init = ds.points3D[:n_saved] if n_saved <= ds.N_points else ds.points3D
    if pts_for_init.shape[0] < n_saved:
        pad = pts_for_init[-1:].repeat(n_saved - pts_for_init.shape[0], 1)
        pts_for_init = torch.cat([pts_for_init, pad], dim=0)
    times_init = torch.zeros(n_saved, dtype=torch.float64)
    params0 = init_gaussians_from_points(
        pts_for_init, times_init, ds.cameras_per_frame,
        sigma_init_sq=0.02, sigma_k_pixel=1.0, sigma_k_temporal=0.0,
    )
    model = trainable_from_params(params0, dtype=DTYPE, device=device, sh_degree=sh_degree)
    model.load_state_dict(state, strict=True)
    print(f"  loaded -> N={model.N}")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    bg = torch.zeros(3, dtype=DTYPE, device=device)
    raster_cfg = FastRasterConfig(sigma_3d_blur=args.sigma_3d_blur, sh_degree=sh_degree)

    # LPIPS instrumentation (added 2026-05-11 per results/rca/blur_rca.md so
    # every per-frame eval reports perceptual quality alongside PSNR/L1).
    lpips_model = None
    if args.lpips_net != "none":
        try:
            import lpips  # type: ignore
            lpips_model = lpips.LPIPS(net=args.lpips_net).to(device).eval()
            for p in lpips_model.parameters():
                p.requires_grad_(False)
            print(f"  LPIPS: net={args.lpips_net}")
        except ImportError:
            print("  LPIPS unavailable (lpips package not installed); "
                  "skipping perceptual metric. PSNR/L1 only.")
    def _lpips_score(img_hwc, gt_hwc) -> float:
        if lpips_model is None:
            return float("nan")
        with torch.no_grad():
            r = (img_hwc.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0).clamp(-1.0, 1.0)
            t = (gt_hwc.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0).clamp(-1.0, 1.0)
            return float(lpips_model(r, t).item())

    # Accumulators.
    err_sum = torch.zeros(ds.H, ds.W, dtype=torch.float32, device=device)
    err_count = 0
    per_frame: list[dict] = []
    worst_frame = (-1, 1e9, None, None)   # (idx, psnr, render, gt)
    best_frame = (-1, -1.0, None, None)

    print(f"Evaluating {len(val_indices)} val frames...")
    for f_idx in val_indices:
        cam = ds.cameras_per_frame[f_idx]
        t_value = float(ds.times[f_idx])
        gt = ds.frame_loader(f_idx).to(device).float()
        if gt.max() > 1.5:
            gt = gt / 255.0

        params_now = model.forward()
        with torch.no_grad():
            img = fast_rasterize(params_now, t_value, cam, ds.H, ds.W,
                                  background=bg, config=raster_cfg)

        diff = (img - gt).abs()
        per_pixel_l1 = diff.mean(dim=-1)            # (H, W)
        err_sum += per_pixel_l1
        err_count += 1

        l1 = float(per_pixel_l1.mean().item())
        psnr = float(-10.0 * torch.log10(((img - gt) ** 2).mean().clamp_min(1e-12)))
        lpips_val = _lpips_score(img, gt)
        per_frame.append({"frame": int(f_idx), "t": t_value, "l1": l1,
                          "psnr": psnr, "lpips": lpips_val})

        if psnr < worst_frame[1]:
            worst_frame = (f_idx, psnr, img.detach().cpu(), gt.detach().cpu())
        if psnr > best_frame[1]:
            best_frame = (f_idx, psnr, img.detach().cpu(), gt.detach().cpu())

    avg_l1 = err_sum / max(err_count, 1)
    avg_psnr = float(np.mean([p["psnr"] for p in per_frame]))
    avg_l1_scalar = float(np.mean([p["l1"] for p in per_frame]))
    lpips_vals = [p["lpips"] for p in per_frame if not np.isnan(p["lpips"])]
    avg_lpips = float(np.mean(lpips_vals)) if lpips_vals else float("nan")

    # Save raw avg-L1 array so we can compute regional stats locally.
    np.save(out / "avg_l1.npy", avg_l1.cpu().numpy())

    # Save aggregate spatial error map.
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    arr = avg_l1.detach().cpu().numpy()
    im = ax.imshow(arr, cmap="hot", vmax=np.quantile(arr, 0.99))
    plt.colorbar(im, ax=ax, label="L1 per pixel (avg over val frames)")
    ax.set_title(f"avg L1 spatial map | N val={err_count}, avg PSNR={avg_psnr:.3f}")
    plt.tight_layout()
    plt.savefig(out / "error_map.png", dpi=110); plt.close()

    # Per-frame PSNR curve.
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    fs = [p["frame"] for p in per_frame]
    ps = [p["psnr"] for p in per_frame]
    ax.plot(fs, ps, "o-", markersize=4)
    ax.axhline(avg_psnr, color="k", ls="--", alpha=0.5, label=f"avg={avg_psnr:.2f}")
    ax.set_xlabel("val frame index"); ax.set_ylabel("PSNR (dB)")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Per-frame val PSNR")
    plt.tight_layout()
    plt.savefig(out / "per_frame_psnr.png", dpi=110); plt.close()

    # Worst/best frame side-by-side.
    for label, (fi, psnr, ren, gt_img) in [("worst", worst_frame), ("best", best_frame)]:
        if ren is None: continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(gt_img.numpy().clip(0, 1)); axes[0].set_title("GT"); axes[0].axis("off")
        axes[1].imshow(ren.numpy().clip(0, 1)); axes[1].set_title(f"render  PSNR={psnr:.2f}"); axes[1].axis("off")
        diff = ((ren - gt_img).abs().mean(dim=-1)).numpy()
        im = axes[2].imshow(diff, cmap="hot", vmax=np.quantile(diff, 0.99))
        axes[2].set_title("|render - GT| (L1)"); axes[2].axis("off")
        plt.colorbar(im, ax=axes[2])
        plt.suptitle(f"{label}-PSNR val frame: idx={fi}")
        plt.tight_layout()
        plt.savefig(out / f"frame_{label}_{fi:04d}.png", dpi=110); plt.close()

    # Save summary JSON.
    with open(out / "summary.json", "w") as f:
        json.dump({
            "ckpt": str(args.ckpt),
            "n_val_frames": err_count,
            "avg_psnr": avg_psnr,
            "avg_l1": avg_l1_scalar,
            "avg_lpips": avg_lpips,
            "lpips_net": args.lpips_net if lpips_model is not None else None,
            "worst_frame": int(worst_frame[0]),
            "worst_psnr": float(worst_frame[1]),
            "best_frame": int(best_frame[0]),
            "best_psnr": float(best_frame[1]),
            "per_frame": per_frame,
        }, f, indent=2)
    print(f"avg val PSNR={avg_psnr:.4f}  L1={avg_l1_scalar:.5f}  "
          f"LPIPS={avg_lpips:.4f}  "
          f"worst frame {worst_frame[0]} @ {worst_frame[1]:.2f}  best {best_frame[0]} @ {best_frame[1]:.2f}")
    print(f"wrote outputs to {out}")


if __name__ == "__main__":
    main()
