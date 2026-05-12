#!/usr/bin/env bash
# Phase-level profiling at Bug-F STEADY-STATE N (≈83k init via multiplier 6).
# Same Bug-F config; init_points_multiplier=6 boosts the initial Gaussian
# count from ~14k to ~83k so the profile matches the regime where Bug-F
# spends most of its 14k-iter run.
#
# 1500 iters; first 500 discarded; ~5 density events in measure window.
# Expected runtime: ~5-6 min on L4 (≈30 ms/iter × 1500 + setup).

set -euo pipefail
cd "$(dirname "$0")/.."

modal run scripts/train_modal.py \
    --cmd smoke --dataset nerfies --scene slice-banana \
    --iters 1500 --log-every 250 --seed 42 \
    --split-convention deformable_interp \
    --sigma-init-sq 0.02 --lambda-frob 1e-4 \
    --init-points-multiplier 6 \
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
    --run-tag profile-bugF-N86k
