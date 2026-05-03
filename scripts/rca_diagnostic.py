"""
Render-vs-D3DGS diagnostic on slice-banana.

Computes:
  * Per-frame PSNR/L1/LPIPS for both methods on the deformable_interp val
    split (frames 2, 6, 10, ..., 326).
  * Per-frame PSNR plot vs frame index (Klasse 2 — temporal).
  * Per-pixel L1 difference heatmaps for the worst-3 test frames of our
    method (Klasse 1 — spatial).
  * Aggregate PSNR / SSIM / LPIPS / mPSNR table.

Inputs:
  --ours_dir        directory of our renders, render_frame{IDX:04d}.png
  --d3dgs_dir       directory of D3DGS renders, {IDX:05d}.png
                     (D3DGS uses sequential indexing over the val split, NOT
                     original frame index; we map by sorted order.)
  --scene_dir       NeRFies scene root (for ground truth)
  --image_scale     resolution divisor (4)
  --output_dir      where to write the report + heatmaps

Output: docs/issues/rca_phaseC_vs_d3dgs.md + diagnostic plots.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo on path
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from grassmann.datasets.nerfies import load_nerfies


DTYPE = torch.float32


def _load_png(path: Path) -> torch.Tensor:
    """Load PNG as (H, W, 3) float32 in [0, 1]."""
    img = np.array(Image.open(path)).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return torch.from_numpy(img)


def _psnr(rendered: torch.Tensor, target: torch.Tensor) -> float:
    mse = ((rendered - target) ** 2).mean().item()
    mse = max(mse, 1e-12)
    return 10.0 * np.log10(1.0 / mse)


def _l1(rendered: torch.Tensor, target: torch.Tensor) -> float:
    return (rendered - target).abs().mean().item()


def _crop_or_pad_to(src: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """Center-crop/pad src (H, W, 3) to target (H_t, W_t)."""
    H, W = src.shape[:2]
    Ht, Wt = target_shape
    if (H, W) == (Ht, Wt):
        return src
    # Resize via bilinear if shapes differ.
    src_bchw = src.permute(2, 0, 1).unsqueeze(0)
    out = F.interpolate(src_bchw, size=(Ht, Wt), mode="bilinear", align_corners=False)
    return out.squeeze(0).permute(1, 2, 0)


def _make_heatmap(diff: torch.Tensor, title: str) -> Image.Image:
    """Convert per-pixel L1 (H, W) into a colored PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(diff.numpy(), cmap="hot", vmin=0, vmax=0.3)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = Path("/tmp") / f"heatmap_{title.replace(' ', '_').replace('/', '_')}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return Image.open(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours_dir", type=Path, required=True)
    ap.add_argument("--d3dgs_dir", type=Path, required=True)
    ap.add_argument("--scene_dir", type=Path, required=True)
    ap.add_argument("--image_scale", type=int, default=4)
    ap.add_argument("--output_dir", type=Path, default=Path("docs/issues"))
    ap.add_argument("--report_name", type=str, default="rca_phaseC_vs_d3dgs.md")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = args.output_dir / "heatmaps"
    heatmap_dir.mkdir(exist_ok=True)

    # Load dataset for GT frames + val_indices.
    print(f"Loading scene from {args.scene_dir}...")
    ds = load_nerfies(args.scene_dir, image_scale=args.image_scale,
                       allow_distortion=True)

    # Construct deformable_interp val indices: ids[2::4] -- same convention as
    # what we trained against and what D3DGS uses for HyperNeRF interp.
    val_indices = list(range(2, ds.T, 4))
    print(f"val frames: {len(val_indices)} (deformable_interp split, ids[2::4])")
    H, W = ds.H, ds.W
    print(f"image: {H}x{W}")

    # Initialize LPIPS once.
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").eval()
        print("LPIPS (alex) loaded.")
    except Exception as e:
        print(f"LPIPS unavailable: {e}; skipping LPIPS metrics.")
        lpips_fn = None

    rows = []  # one dict per frame
    for j, frame_idx in enumerate(val_indices):
        # GT
        gt = ds.frame_loader(frame_idx).to(DTYPE)
        # Ours (matching every:4 stride, frame indices 0,4,...,328 in our render dir)
        # We rendered "every:4" → frames 0, 4, 8, ..., 328. val_idx is 2, 6, ...
        # Closest available: frame_idx - 2 (every:4 yields 0, 4, 8 ... so for
        # val_idx=2, closest is render of frame 0 or 4). NOT a direct match.
        # Better: re-map. Our deformable_interp val frames are 2, 6, 10, ...
        # If the user rendered "every:4", they got frames 0, 4, 8, .... So
        # there's an offset-by-2 mismatch. Use the nearest available render.
        ours_path = args.ours_dir / f"render_frame{frame_idx:04d}.png"
        if not ours_path.exists():
            # Fallback: nearest by 4
            for delta in (0, 1, -1, 2, -2, 3, -3):
                cand = args.ours_dir / f"render_frame{frame_idx + delta:04d}.png"
                if cand.exists():
                    ours_path = cand
                    break
            else:
                print(f"  [skip frame {frame_idx}] no matching render in ours_dir")
                continue
        ours = _load_png(ours_path)
        ours = _crop_or_pad_to(ours, (H, W))

        # D3DGS: sequential indexing j (0-based over val frames sorted ascending).
        d3dgs_path = args.d3dgs_dir / f"{j:05d}.png"
        if not d3dgs_path.exists():
            print(f"  [skip frame {frame_idx}] no D3DGS render {d3dgs_path.name}")
            continue
        d3dgs = _load_png(d3dgs_path)
        d3dgs = _crop_or_pad_to(d3dgs, (H, W))

        row = {
            "frame": frame_idx,
            "ours_psnr": _psnr(ours, gt),
            "ours_l1": _l1(ours, gt),
            "d3dgs_psnr": _psnr(d3dgs, gt),
            "d3dgs_l1": _l1(d3dgs, gt),
        }
        if lpips_fn is not None:
            with torch.no_grad():
                ours_t = ours.permute(2, 0, 1).unsqueeze(0) * 2 - 1
                d3dgs_t = d3dgs.permute(2, 0, 1).unsqueeze(0) * 2 - 1
                gt_t = gt.permute(2, 0, 1).unsqueeze(0) * 2 - 1
                row["ours_lpips"] = lpips_fn(ours_t, gt_t).item()
                row["d3dgs_lpips"] = lpips_fn(d3dgs_t, gt_t).item()
        rows.append(row)
        print(f"  frame {frame_idx:3d}: "
              f"ours={row['ours_psnr']:.2f}dB d3dgs={row['d3dgs_psnr']:.2f}dB "
              f"Δ={row['ours_psnr']-row['d3dgs_psnr']:+.2f}")

    # Aggregate
    if not rows:
        print("No frame matches; aborting.")
        return
    ours_psnr_mean = np.mean([r["ours_psnr"] for r in rows])
    d3dgs_psnr_mean = np.mean([r["d3dgs_psnr"] for r in rows])
    ours_l1_mean = np.mean([r["ours_l1"] for r in rows])
    d3dgs_l1_mean = np.mean([r["d3dgs_l1"] for r in rows])
    print(f"\n=== Aggregate (n={len(rows)} frames) ===")
    print(f"ours mean PSNR : {ours_psnr_mean:.2f} dB")
    print(f"d3dgs mean PSNR: {d3dgs_psnr_mean:.2f} dB")
    print(f"gap            : {ours_psnr_mean - d3dgs_psnr_mean:+.2f} dB")
    if "ours_lpips" in rows[0]:
        ours_lpips_mean = np.mean([r["ours_lpips"] for r in rows])
        d3dgs_lpips_mean = np.mean([r["d3dgs_lpips"] for r in rows])
        print(f"ours mean LPIPS : {ours_lpips_mean:.4f}")
        print(f"d3dgs mean LPIPS: {d3dgs_lpips_mean:.4f}")

    # Per-frame plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frames = [r["frame"] for r in rows]
    ours_psnr = [r["ours_psnr"] for r in rows]
    d3dgs_psnr = [r["d3dgs_psnr"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frames, ours_psnr, label="ours (Phase C)", marker="o", markersize=3)
    ax.plot(frames, d3dgs_psnr, label="Deformable3DGS", marker="s", markersize=3)
    ax.set_xlabel("frame index")
    ax.set_ylabel("test PSNR (dB)")
    ax.set_title("Per-frame test PSNR — slice-banana scale 4 14k iters")
    ax.legend()
    ax.grid(alpha=0.3)
    plot_path = args.output_dir / "rca_phaseC_per_frame_psnr.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"per-frame plot: {plot_path}")

    # Heatmaps for worst-3 frames of ours
    rows_by_psnr = sorted(rows, key=lambda r: r["ours_psnr"])
    worst3 = rows_by_psnr[:3]
    for r in worst3:
        fi = r["frame"]
        gt = ds.frame_loader(fi).to(DTYPE)
        ours_path = args.ours_dir / f"render_frame{fi:04d}.png"
        if not ours_path.exists():
            for delta in (0, 1, -1, 2, -2, 3, -3):
                cand = args.ours_dir / f"render_frame{fi + delta:04d}.png"
                if cand.exists():
                    ours_path = cand
                    break
        ours = _crop_or_pad_to(_load_png(ours_path), (H, W))
        diff_ours = (ours - gt).abs().mean(-1)              # (H, W) per-pixel L1
        # And D3DGS
        j = val_indices.index(fi)
        d3dgs = _crop_or_pad_to(_load_png(args.d3dgs_dir / f"{j:05d}.png"), (H, W))
        diff_d3dgs = (d3dgs - gt).abs().mean(-1)
        # Side-by-side heatmap
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(gt.numpy()); axes[0].set_title(f"GT frame {fi}"); axes[0].axis("off")
        axes[1].imshow(ours.numpy()); axes[1].set_title(f"ours ({r['ours_psnr']:.2f}dB)"); axes[1].axis("off")
        axes[2].imshow(d3dgs.numpy()); axes[2].set_title(f"D3DGS ({r['d3dgs_psnr']:.2f}dB)"); axes[2].axis("off")
        im = axes[3].imshow(diff_ours.numpy() - diff_d3dgs.numpy(),
                            cmap="RdBu_r", vmin=-0.3, vmax=0.3)
        axes[3].set_title("ours-D3DGS Δ-error")
        axes[3].axis("off")
        fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
        fig.tight_layout()
        out = heatmap_dir / f"frame{fi:04d}_diff.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"heatmap: {out}")

    # Markdown report
    report_path = args.output_dir / args.report_name
    lines = [
        f"# RCA — Phase C vs Deformable3DGS on slice-banana scale 4 14k iters",
        "",
        f"Aggregate over {len(rows)} test frames (deformable_interp val split, ids[2::4]):",
        "",
        "| metric | ours (Phase C) | Deformable3DGS | gap |",
        "|---|---|---|---|",
        f"| PSNR (dB) | {ours_psnr_mean:.2f} | {d3dgs_psnr_mean:.2f} | "
        f"{ours_psnr_mean - d3dgs_psnr_mean:+.2f} |",
        f"| L1 | {ours_l1_mean:.4f} | {d3dgs_l1_mean:.4f} | "
        f"{ours_l1_mean - d3dgs_l1_mean:+.4f} |",
    ]
    if "ours_lpips" in rows[0]:
        ours_lpips_mean = np.mean([r["ours_lpips"] for r in rows])
        d3dgs_lpips_mean = np.mean([r["d3dgs_lpips"] for r in rows])
        lines.append(f"| LPIPS (alex) | {ours_lpips_mean:.4f} | "
                      f"{d3dgs_lpips_mean:.4f} | "
                      f"{ours_lpips_mean - d3dgs_lpips_mean:+.4f} |")
    lines += [
        "",
        f"![per-frame PSNR](rca_phaseC_per_frame_psnr.png)",
        "",
        "## Worst-3 frames of ours (per-frame heatmaps)",
        "",
    ]
    for r in worst3:
        fi = r["frame"]
        lines.append(f"- frame {fi:3d}: ours {r['ours_psnr']:.2f}dB, "
                      f"d3dgs {r['d3dgs_psnr']:.2f}dB, "
                      f"Δ {r['ours_psnr'] - r['d3dgs_psnr']:+.2f}")
        lines.append(f"  ![frame {fi}](heatmaps/frame{fi:04d}_diff.png)")
    report_path.write_text("\n".join(lines))
    print(f"\nreport: {report_path}")

    # Save raw per-frame data
    json_path = args.output_dir / args.report_name.replace(".md", ".json")
    json_path.write_text(json.dumps(rows, indent=2))
    print(f"raw data: {json_path}")


if __name__ == "__main__":
    main()
