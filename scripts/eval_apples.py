"""Apples-to-apples per-frame PSNR/L1 vs D3DGS saved GT (slice-banana scale 8).

Pairs `<renders>/render_frame{F:04d}.png` against `<gt>/{j:05d}.png` for the
deformable_interp val split (frames [2, 6, 10, ..., 326] -> j = 0..81).

Usage:
    python scripts/eval_apples.py --renders ./renders/<run> \
        --tag 14k_opreset3k --out docs/issues/perframe_14k_opreset3k_apples.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

VAL_FRAMES = list(range(2, 327, 4))  # 82 frames; deformable_interp ids[2::4]


def _load(p: Path) -> np.ndarray:
    a = np.array(Image.open(p)).astype(np.float32) / 255.0
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.shape[-1] == 4:
        a = a[..., :3]
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True, type=Path)
    ap.add_argument("--gt", type=Path, default=Path("/tmp/d3dgs_gt/gt"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tag", default="ours")
    args = ap.parse_args()

    if len(VAL_FRAMES) != 82:
        raise SystemExit("VAL_FRAMES list must be 82 entries")

    rows = []
    psnrs = []
    l1s = []
    missing = []
    for j, f in enumerate(VAL_FRAMES):
        rp = args.renders / f"render_frame{f:04d}.png"
        gp = args.gt / f"{j:05d}.png"
        if not rp.exists() or not gp.exists():
            missing.append((f, str(rp), str(gp)))
            continue
        r = _load(rp)
        g = _load(gp)
        if r.shape != g.shape:
            raise SystemExit(f"shape mismatch frame {f}: ours {r.shape} vs gt {g.shape}")
        mse = float(((r - g) ** 2).mean())
        psnr = 10.0 * float(np.log10(1.0 / max(mse, 1e-12)))
        l1 = float(np.abs(r - g).mean())
        psnrs.append(psnr)
        l1s.append(l1)
        rows.append({"frame": f, f"{args.tag}_psnr": psnr, f"{args.tag}_l1": l1})

    if missing:
        raise SystemExit(f"missing {len(missing)} frame(s); first: {missing[0]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    mean_psnr = sum(psnrs) / len(psnrs)
    mean_l1 = sum(l1s) / len(l1s)
    print(f"  n={len(psnrs)}  mean PSNR={mean_psnr:.3f} dB  mean L1={mean_l1:.4f}  -> {args.out}")


if __name__ == "__main__":
    main()
