#!/usr/bin/env bash
# Per-frame frontal init experiment on the recipe winner.
#
# For each SfM point with observability list obs, emit one replica per
# every-20th observed frame. Each replica's --init-strategy frontal then
# picks THAT frame's camera, so the disk faces the camera at its init t.
# At render time t', the temporal w_t selects replicas near t' -- which
# were initialized frontal-to a nearby-in-time camera.

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
    --clamp-mode soft
    --grassmann-relax-start 1000 --grassmann-relax-end 8000
    --max-aspect-ratio 1000000 --random-background
    --opacity-reset-every 3000
    --scale-min-prune 5e-3
    --lambda-aniso 0
    --temporal-split-threshold 0.1
    --split-anisotropic-shrink
    --structural-kind ssim
)

nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --init-strategy frontal \
    --init-per-frame-stride 20 \
    --run-tag bugF-best-frontal-pf20 \
    > /tmp/probe_bugF-best-frontal-pf20.log 2>&1 &
echo "Launched PER-FRAME-FRONTAL (stride=20) — PID $!"

echo "ETA ~6-8 min on L4. Compare against:"
echo "  control (spatial_slice):   val_psnr=24.11 dB, N=98,890,  wall=396 s"
echo "  test    (frontal global):  val_psnr=23.98 dB, N=188,206, wall=475 s"
