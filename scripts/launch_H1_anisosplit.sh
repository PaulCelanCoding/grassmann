#!/usr/bin/env bash
# H1: geometric-only anisotropy split (no grad gate). On Bug F baseline.
# Diagnostic showed 14.75% of pop has aspect>20; thresholds 6/10/15 bracket
# where fragmentation starts helping vs hurting.

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
    --split-anisotropic-shrink   # Bug F (new baseline)
)

for thr in 15 10 6; do
    nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
        --aspect-split-threshold $thr \
        --run-tag bug-H1-aniso-thr$thr \
        > /tmp/probe_bug-H1-aniso-thr$thr.log 2>&1 &
    echo "Launched H1 aspect-split threshold=$thr — PID $!"
done

echo ""
echo "All 3 launched."
