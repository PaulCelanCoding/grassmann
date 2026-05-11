#!/usr/bin/env bash
# Phase-level profiling of the Bug-F training step on Modal/L4.
#
# Bug-F config mirrored from launch_bugF_isoN_controls.sh, shortened to
# 1500 iters. First 500 iters discarded (warmup + first density event +
# Adam state stabilization); profile stats accumulate over iter 501-1500.
#
# log-every 250 → 4 breakdown reports in the measure window.
# Density events fire at 500/700/.../1500 (8 events; ~5 inside measure window).
# Aspect-clip every 100 → 10 events in measure window.
#
# Bug-F steady-state reference: 26.5 ms/iter @ N≈86.6k on L4 (371s / 14k iters).
# Expected runtime here: ~50s training + ~30s setup.

set -euo pipefail
cd "$(dirname "$0")/.."

modal run scripts/train_modal.py \
    --cmd smoke --dataset nerfies --scene slice-banana \
    --iters 1500 --log-every 250 --seed 42 \
    --split-convention deformable_interp \
    --sigma-init-sq 0.02 --lambda-frob 1e-4 \
    --densify-every 200 --densify-start 500 --densify-stop 10000 \
    --grad-threshold 1e-5 \
    --opacity-prune-threshold 1e-3 --sh-degree 3 --lr-decay 0.01 \
    --init-strategy spatial_slice --clamp-mode soft \
    --grassmann-relax-start 1000 --grassmann-relax-end 8000 \
    --max-aspect-ratio 30 --random-background \
    --opacity-reset-every 3000 \
    --scale-min-prune 5e-3 \
    --lambda-aniso 0 \
    --temporal-split-threshold 0.1 \
    --spatial-split-threshold 0.5 \
    --split-anisotropic-shrink \
    --profile-breakdown --profile-warmup-iters 500 \
    --run-tag profile-bugF
