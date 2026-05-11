#!/usr/bin/env bash
# Quadratic-motion probe: V_3D(t) = V_k + dt·c_world/σ_tt + dt²·c2.
# On Bug F baseline (val=24.93 dB, 86k Gaussians, wall=371s).

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

# Baseline lr_c2=5e-4
nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --use-quadratic-motion --lr-c2 5e-4 \
    --run-tag bugF-quadmotion-lr5e4 \
    > /tmp/probe_bugF-quadmotion-lr5e4.log 2>&1 &
echo "Launched quadmotion lr=5e-4 — PID $!"

# Bigger lr_c2 to test if c2 needs more learning headroom
nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --use-quadratic-motion --lr-c2 5e-3 \
    --run-tag bugF-quadmotion-lr5e3 \
    > /tmp/probe_bugF-quadmotion-lr5e3.log 2>&1 &
echo "Launched quadmotion lr=5e-3 — PID $!"

echo ""
echo "Bug F reference: 24.93 dB val, 86k Gaussians, 371s wall."
