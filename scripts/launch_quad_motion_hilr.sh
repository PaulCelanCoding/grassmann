#!/usr/bin/env bash
# Quadmotion probe at HIGHER LRs (current 5e-3 gave null; magnitude analysis
# says c2 needs ≥5e-2 to reach magnitude comparable to the linear shift).

set -euo pipefail
cd "$(dirname "$0")/.."

BASE_FIXED=(
    --cmd smoke --dataset nerfies --scene slice-banana
    --iters 14000 --log-every 500 --seed 42
    --split-convention deformable_interp
    --sigma-init-sq 0.02 --lambda-frob 1e-4
    --densify-every 200 --densify-start 500 --densify-stop 10000
    --grad-threshold 1e-5 --spatial-split-threshold 0.5
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

nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --use-quadratic-motion --lr-c2 5e-2 \
    --run-tag bugF-quadmotion-lr5e2 \
    > /tmp/probe_bugF-quadmotion-lr5e2.log 2>&1 &
echo "Launched quadmotion lr=5e-2 — PID $!"

nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --use-quadratic-motion --lr-c2 5e-1 \
    --run-tag bugF-quadmotion-lr5e1 \
    > /tmp/probe_bugF-quadmotion-lr5e1.log 2>&1 &
echo "Launched quadmotion lr=5e-1 — PID $!"
