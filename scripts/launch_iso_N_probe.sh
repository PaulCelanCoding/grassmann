#!/usr/bin/env bash
# Iso-N capacity probe: same Bug-F recipe but with --densify_stop 5000 and
# --grad_threshold 2e-4 (3DGS defaults, bundled). Targets final N ≈ 30k
# (D3DGS scale) instead of 86k.
#
# Tests blur RCA mechanism #2 (over-densification → alpha-blend smoothing).
# K1 (SSIM swap) closed only ~12% of the LPIPS gap; capacity is the leading
# remaining hypothesis.

set -euo pipefail
cd "$(dirname "$0")/.."

BASE_FIXED=(
    --cmd smoke --dataset nerfies --scene slice-banana
    --iters 14000 --log-every 500 --seed 42
    --split-convention deformable_interp
    --sigma-init-sq 0.02 --lambda-frob 1e-4
    --densify-every 200 --densify-start 500
    --spatial-split-threshold 0.5
    --opacity-prune-threshold 1e-3 --sh-degree 3 --lr-decay 0.01
    --init-strategy spatial_slice --clamp-mode soft
    --grassmann-relax-start 1000 --grassmann-relax-end 8000
    --max-aspect-ratio 30 --random-background
    --opacity-reset-every 3000
    --scale-min-prune 5e-3
    --lambda-aniso 0
    --temporal-split-threshold 0.1
    --split-anisotropic-shrink
)

# ISO-N: lower densify-stop + raise grad-threshold to 3DGS defaults.
nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --densify-stop 5000 \
    --grad-threshold 2e-4 \
    --run-tag bugF-isoN-low \
    > /tmp/probe_bugF-isoN-low.log 2>&1 &
echo "Launched iso-N (low capacity) — PID $!"
echo ""
echo "After it completes, evaluate LPIPS with:"
echo "  modal run scripts/bugF_vs_d3dgs_modal.py \\"
echo "    --bugf-ckpt nerfies-slice-banana-spatial_slice-14000it-bugF-isoN-low/trained_nerfies_spatial_slice.pt \\"
echo "    --out-dir comparisons/isoN_low_lpips_check"
