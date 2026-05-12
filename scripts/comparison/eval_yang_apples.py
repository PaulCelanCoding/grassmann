"""Apples-to-apples per-frame PSNR/L1 for Yang 4DGS renders vs D3DGS GT.

Yang's render.py (our patched version) saves test renders as
<model_path>/test/ours_<iter>/renders/<image_name>.png where image_name is
the NeRFies item_id ("000001", "000002", ...). The val split (per our
HyperNeRF reader) uses ids[2::4], so val item_ids are
{"000003","000007",...,"000327"}.

D3DGS GT is at <gt>/{j:05d}.png for j = 0..81. Mapping:
  val_idx j  ->  ids index 2 + 4*j  ->  item_id "{(2+4*j)+1:06d}"
  (item_id starts at 000001, so absolute id = ids index + 1).

Yang trains/renders at scale 4 (480x270 area for slice-banana). D3DGS GT is
at scale 8 (240x134). We bilinear-downsample Yang's renders to D3DGS GT
shape before PSNR/L1 -- same convention as scripts/rca_diagnostic.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _load_rgb(p: Path) -> torch.Tensor:
    """Load PNG as (H, W, 3) float tensor in [0,1]."""
    a = np.array(Image.open(p)).astype(np.float32) / 255.0
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.shape[-1] == 4:
        a = a[..., :3]
    return torch.from_numpy(a)


def _resize_to(src: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    H, W = src.shape[:2]
    Ht, Wt = target_shape
    if (H, W) == (Ht, Wt):
        return src
    bchw = src.permute(2, 0, 1).unsqueeze(0)
    out = F.interpolate(bchw, size=(Ht, Wt), mode="bilinear", align_corners=False)
    return out.squeeze(0).permute(1, 2, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True, type=Path,
                    help="Directory with Yang's renders/<item_id>.png")
    ap.add_argument("--gt", type=Path, default=Path("/tmp/d3dgs_gt/gt"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tag", default="yang4dgs")
    ap.add_argument("--n_val", type=int, default=82)
    args = ap.parse_args()

    rows = []
    psnrs = []
    l1s = []
    missing = []
    # Determine target shape from the first available GT frame.
    gt0 = _load_rgb(args.gt / f"{0:05d}.png")
    Ht, Wt = gt0.shape[:2]
    print(f"GT shape: ({Ht}, {Wt})")

    for j in range(args.n_val):
        ids_index = 2 + 4 * j
        item_id = f"{ids_index + 1:06d}"
        rp = args.renders / f"{item_id}.png"
        gp = args.gt / f"{j:05d}.png"
        if not rp.exists() or not gp.exists():
            missing.append((j, item_id, str(rp), str(gp)))
            continue
        r = _load_rgb(rp)
        g = _load_rgb(gp)
        r_resized = _resize_to(r, (Ht, Wt))
        mse = float(((r_resized - g) ** 2).mean().item())
        psnr = 10.0 * float(np.log10(1.0 / max(mse, 1e-12)))
        l1 = float((r_resized - g).abs().mean().item())
        psnrs.append(psnr)
        l1s.append(l1)
        rows.append({
            "frame": ids_index,
            "item_id": item_id,
            f"{args.tag}_psnr": psnr,
            f"{args.tag}_l1": l1,
        })

    if missing:
        head = missing[:3]
        raise SystemExit(f"missing {len(missing)} frame(s); first: {head}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    mean_psnr = sum(psnrs) / len(psnrs)
    mean_l1 = sum(l1s) / len(l1s)
    print(f"  n={len(psnrs)}  mean PSNR={mean_psnr:.3f} dB  mean L1={mean_l1:.4f}  -> {args.out}")


if __name__ == "__main__":
    main()
