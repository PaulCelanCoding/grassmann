#!/usr/bin/env bash
# Leave-one-out flag bisection on LPIPS (advisor recommendation,
# results/rca/blur_rca.md step 3). After K1, iso-N, and σ_3d_blur sweep all
# failed to close the LPIPS gap, this tests the three remaining recipe-level
# mechanisms before accepting the architectural ceiling conclusion.

set -euo pipefail
cd "$(dirname "$0")/.."

# All flags except the one varied per-probe. Each probe adds the
# differentiated knob explicitly to avoid bash array substitution pitfalls.
BASE_FIXED=(
    --cmd smoke --dataset nerfies --scene slice-banana
    --iters 14000 --log-every 500 --seed 42
    --split-convention deformable_interp
    --sigma-init-sq 0.02 --lambda-frob 1e-4
    --densify-every 200 --densify-start 500 --densify-stop 10000
    --grad-threshold 1e-5 --spatial-split-threshold 0.5
    --opacity-prune-threshold 1e-3 --sh-degree 3 --lr-decay 0.01
    --init-strategy spatial_slice --clamp-mode soft
    --grassmann-relax-start 1000
    --random-background
    --scale-min-prune 5e-3
    --lambda-aniso 0
    --temporal-split-threshold 0.1
    --split-anisotropic-shrink
)

# Bisect-1: drop opacity reset.   Baseline uses --opacity-reset-every 3000.
nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --opacity-reset-every 0 \
    --max-aspect-ratio 30 \
    --grassmann-relax-end 8000 \
    --run-tag bugF-bisect-noOPreset \
    > /tmp/probe_bugF-bisect-noOPreset.log 2>&1 &
echo "Launched bisect-noOPreset — PID $!"

# Bisect-2: allow elongated splats. Baseline uses --max-aspect-ratio 30.
nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --opacity-reset-every 3000 \
    --max-aspect-ratio 200 \
    --grassmann-relax-end 8000 \
    --run-tag bugF-bisect-maxAR200 \
    > /tmp/probe_bugF-bisect-maxAR200.log 2>&1 &
echo "Launched bisect-maxAR200 — PID $!"

# Bisect-3: ramp n_lr earlier.    Baseline uses --grassmann-relax-end 8000.
nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --opacity-reset-every 3000 \
    --max-aspect-ratio 30 \
    --grassmann-relax-end 1000 \
    --run-tag bugF-bisect-relaxEarly \
    > /tmp/probe_bugF-bisect-relaxEarly.log 2>&1 &
echo "Launched bisect-relaxEarly — PID $!"

echo ""
echo "All 3 launched. Each takes ~5 min on L4. After each completes, evaluate LPIPS with:"
echo "  modal run scripts/bugF_vs_d3dgs_modal.py \\"
echo "    --bugf-ckpt nerfies-slice-banana-spatial_slice-14000it-<run-tag>/trained_nerfies_spatial_slice.pt \\"
echo "    --out-dir comparisons/<run-tag>_lpips_check"
