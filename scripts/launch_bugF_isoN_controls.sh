#!/usr/bin/env bash
# Iso-N controls for Bug F (val=24.93 dB @ N=86.6k, wall=371s).
# Bug F's PSNR win could be:
#   (a) capacity-driven (more Gaussians → more PSNR), OR
#   (b) better-quality children (no zombies, alive splits).
# Tests:
#   ctrl-1: Bug-D + init_points_multiplier=2  → push baseline N up via init
#   ctrl-2: Bug-D + spatial_split_threshold=0.2  → push baseline N up via growth
#   ctrl-3: Bug F + max_split_per_event=200  → throttle F's growth back to ~Bug-D N
# If F still wins at iso-N → real quality fix.
# If F's win shrinks at iso-N → capacity confound.

set -euo pipefail
cd "$(dirname "$0")/.."

BASE_FIXED=(
    --cmd smoke --dataset nerfies --scene slice-banana
    --iters 14000 --log-every 500 --seed 42
    --split-convention deformable_interp
    --sigma-init-sq 0.02 --lambda-frob 1e-4
    --densify-every 200 --densify-start 500 --densify-stop 10000
    --grad-threshold 1e-5
    --opacity-prune-threshold 1e-3 --sh-degree 3 --lr-decay 0.01
    --init-strategy spatial_slice --clamp-mode soft
    --grassmann-relax-start 1000 --grassmann-relax-end 8000
    --max-aspect-ratio 30 --random-background
    --opacity-reset-every 3000
    --scale-min-prune 5e-3
    --lambda-aniso 0
    --temporal-split-threshold 0.1
)

# ---- ctrl-1: Bug-D + init_points_multiplier=2 (capacity from init) ----
nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --spatial-split-threshold 0.5 \
    --init-points-multiplier 2 \
    --run-tag ctrl1-bugD-init2x \
    > /tmp/probe_ctrl1-bugD-init2x.log 2>&1 &
echo "Launched ctrl-1 (Bug-D init 2x) — PID $!"

# ---- ctrl-2: Bug-D + spatial_split_threshold=0.2 (capacity from growth) ----
nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --spatial-split-threshold 0.2 \
    --run-tag ctrl2-bugD-thr02 \
    > /tmp/probe_ctrl2-bugD-thr02.log 2>&1 &
echo "Launched ctrl-2 (Bug-D thr 0.2) — PID $!"

# ---- ctrl-3: Bug F + max_split_per_event=200 (throttle F) ----
nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --spatial-split-threshold 0.5 \
    --split-anisotropic-shrink \
    --max-split-per-event 200 \
    --run-tag ctrl3-bugF-cap200 \
    > /tmp/probe_ctrl3-bugF-cap200.log 2>&1 &
echo "Launched ctrl-3 (Bug F cap 200) — PID $!"

echo ""
echo "All 3 iso-N controls launched."
