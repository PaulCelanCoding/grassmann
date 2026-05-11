#!/usr/bin/env bash
# Bug F (anisotropic L-shrinkage) + Bug H (Kheradmand opacity-split) probes
# on top of the new Bug-D baseline (val=24.62 dB, N=46.8k, wall=210s).
#
# 3 probes, each varies one flag:
#   bug-F:  --split_anisotropic_shrink only
#   bug-H:  --split_opacity_correction only
#   bug-FH: both

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
)

# ---- Bug F: anisotropic L-shrinkage on split -----------------------
nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --split-anisotropic-shrink \
    --run-tag bug-F-aniso \
    > /tmp/probe_bug-F-aniso.log 2>&1 &
echo "Launched Bug F (aniso shrink) — PID $!"

# ---- Bug H: Kheradmand opacity-split correction --------------------
nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --split-opacity-correction \
    --run-tag bug-H-opsplit \
    > /tmp/probe_bug-H-opsplit.log 2>&1 &
echo "Launched Bug H (opacity-split) — PID $!"

# ---- Bug FH: combined ----------------------------------------------
nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --split-anisotropic-shrink --split-opacity-correction \
    --run-tag bug-FH \
    > /tmp/probe_bug-FH.log 2>&1 &
echo "Launched Bug F+H combined — PID $!"

echo ""
echo "All 3 launched. Tail with: tail -f /tmp/probe_bug-F-aniso.log /tmp/probe_bug-H-opsplit.log /tmp/probe_bug-FH.log"
