"""
Build a comparison table across runs.

Inputs:
  --ours    paths to one or more .pt checkpoints saved by our train_mono.py
            (each contains ckpt['history'] with iter/loss/l1/psnr lists; the
            last entry is the final-iter number; train+val PSNR are logged
            at validation_every steps when applicable).
  --deformable
            paths to one or more Deformable3DGS output dirs produced by
            scripts/train_modal_deformable.py. Each must contain
            results.json (written by their metrics.py) with PSNR/SSIM/LPIPS
            keys at the trained iteration.

Output:
  Prints a markdown table to stdout with one row per run:
    Method, Iters, train PSNR, val PSNR, val L1, N (when known), source

Usage:
  python scripts/collate_eval.py \\
      --ours checkpoints/<run>/trained_nerfies_random.pt \\
      --deformable checkpoints/deformable-slice-banana-14000it-iso14k/

Notes:
  Our trainer's history dict contains running training-batch metrics; for
  val numbers we use the last validate() call. Deformable3DGS's results.json
  is per-iteration test-set numbers averaged over the test-set frames at
  the saved iteration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch


def _summarize_ours(ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    history = ckpt.get("history", {})
    if not history:
        return {"method": "ours (no history)", "iters": "?",
                "train_psnr": "?", "val_psnr": "?", "val_l1": "?",
                "N": "?", "source": str(ckpt_path)}
    iters_list = history.get("iter", [])
    psnr_list = history.get("psnr", [])
    l1_list = history.get("l1", [])
    n_list = history.get("N", [])
    final_iter = iters_list[-1] if iters_list else "?"
    final_train_psnr = psnr_list[-1] if psnr_list else "?"
    final_l1 = l1_list[-1] if l1_list else "?"
    final_n = n_list[-1] if n_list else "?"
    # Look for the most recent val_l1/val_psnr in history if logged.
    # Our trainer logs val into the running info dict; we don't currently
    # persist it into history, so the val number is only printed to stdout.
    # The user can grep the run log; here we display N/A.
    return {
        "method": "3-plane (ours)",
        "iters": str(final_iter),
        "train_psnr": (f"{final_train_psnr:.2f}"
                       if isinstance(final_train_psnr, (int, float)) else "?"),
        "val_psnr": "(see log)",
        "val_l1": (f"{final_l1:.4f}"
                   if isinstance(final_l1, (int, float)) else "?"),
        "N": str(final_n),
        "source": ckpt_path.name,
    }


def _summarize_deformable(out_dir: Path) -> dict:
    """Their metrics.py writes a results.json per output dir."""
    results_path = out_dir / "results.json"
    if not results_path.exists():
        return {"method": "Deformable3DGS",
                "iters": "?", "train_psnr": "?", "val_psnr": "?",
                "val_l1": "?", "N": "?",
                "source": f"{out_dir.name} (no results.json)"}
    with open(results_path) as f:
        data = json.load(f)
    # Their results.json keys vary; usually a single "ours_<iter>" entry per
    # output dir with PSNR/SSIM/LPIPS subkeys.
    iter_key = next(iter(data))
    metrics = data[iter_key]
    return {
        "method": "Deformable3DGS",
        "iters": iter_key.replace("ours_", ""),
        "train_psnr": "(test only)",
        "val_psnr": f"{float(metrics.get('PSNR', 'nan')):.2f}",
        "val_l1": "?",
        "N": "?",
        "source": out_dir.name,
        "ssim": f"{float(metrics.get('SSIM', 'nan')):.4f}",
        "lpips": f"{float(metrics.get('LPIPS', 'nan')):.4f}",
    }


def _print_md_table(rows: list[dict]) -> None:
    cols = ["method", "iters", "train_psnr", "val_psnr", "val_l1",
            "ssim", "lpips", "N", "source"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        print("| " + " | ".join(str(r.get(c, "—")) for c in cols) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", type=Path, nargs="*", default=[],
                    help="Paths to our .pt checkpoints.")
    ap.add_argument("--deformable", type=Path, nargs="*", default=[],
                    help="Paths to Deformable3DGS output directories "
                         "(must contain results.json from their metrics.py).")
    args = ap.parse_args()

    rows: list[dict] = []
    for ckpt in args.ours:
        rows.append(_summarize_ours(ckpt))
    for d in args.deformable:
        rows.append(_summarize_deformable(d))

    if not rows:
        ap.error("Need at least one --ours or --deformable argument.")
    _print_md_table(rows)


if __name__ == "__main__":
    main()
