#!/usr/bin/env bash
# Advisor's two remaining cheap suggestions on Bug F baseline:
#  K1: 3DGS-style 0.8*L1 + 0.2*DSSIM loss (--structural_kind ssim)
#  K2: --refine_poses (COLMAP poses may be imperfect, advisor flags ~0.5 dB)
# Both on top of Bug F.

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

# ---- K1: DSSIM structural loss ----
nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --structural-kind ssim \
    --run-tag bugF-K1-dssim \
    > /tmp/probe_bugF-K1-dssim.log 2>&1 &
echo "Launched K1 (DSSIM) — PID $!"

# ---- K2: refine_poses ----
nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --refine-poses --pose-warmup-iter 2000 \
    --run-tag bugF-K2-refinepose \
    > /tmp/probe_bugF-K2-refinepose.log 2>&1 &
echo "Launched K2 (refine_poses) — PID $!"

# ---- K1+K2: combined ----
nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --structural-kind ssim \
    --refine-poses --pose-warmup-iter 2000 \
    --run-tag bugF-K12-combo \
    > /tmp/probe_bugF-K12-combo.log 2>&1 &
echo "Launched K1+K2 combined — PID $!"

echo ""
echo "All 3 launched."
