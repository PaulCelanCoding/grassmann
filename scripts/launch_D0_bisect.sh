#!/usr/bin/env bash
# D0 single-frame regression bisect: today's code at 27.84 dB vs Phase A 29.07 dB.
# Five one-flag-diff probes vs a bare D0 baseline. Plan from
# results/rca/static_baseline/SUMMARY.md §4.

set -euo pipefail
cd "$(dirname "$0")/.."

BASE=(
    --cmd smoke --dataset nerfies --scene slice-banana
    --iters 14000 --log-every 500 --seed 42
    --image-scale 4
    --diag-single-frame 100
    --init-strategy random
    --sigma-init-sq 0.02
    --densify-every 200 --densify-start 500 --densify-stop 10000
    --grad-threshold 1e-5 --spatial-split-threshold 0.5
    --opacity-prune-threshold 1e-3
    --split-convention deformable_interp
)

# ---- D0-bare: target match Phase A 29.07 dB ----
nohup modal run --detach scripts/train_modal.py "${BASE[@]}" \
    --sh-degree 0 \
    --run-tag d0-bisect-bare \
    > /tmp/probe_d0-bisect-bare.log 2>&1 &
echo "Launched D0-bare — PID $!"

# ---- +SH3 only ----
nohup modal run --detach scripts/train_modal.py "${BASE[@]}" \
    --sh-degree 3 \
    --run-tag d0-bisect-sh3 \
    > /tmp/probe_d0-bisect-sh3.log 2>&1 &
echo "Launched D0+SH3 — PID $!"

# ---- +lr_decay 0.01 only ----
nohup modal run --detach scripts/train_modal.py "${BASE[@]}" \
    --sh-degree 0 --lr-decay 0.01 \
    --run-tag d0-bisect-lrdecay \
    > /tmp/probe_d0-bisect-lrdecay.log 2>&1 &
echo "Launched D0+lr_decay — PID $!"

# ---- +opacity-reset 3000 only ----
nohup modal run --detach scripts/train_modal.py "${BASE[@]}" \
    --sh-degree 0 --opacity-reset-every 3000 \
    --run-tag d0-bisect-opreset \
    > /tmp/probe_d0-bisect-opreset.log 2>&1 &
echo "Launched D0+opreset — PID $!"

# ---- +scale_min_prune 5e-3 only ----
nohup modal run --detach scripts/train_modal.py "${BASE[@]}" \
    --sh-degree 0 --scale-min-prune 5e-3 \
    --run-tag d0-bisect-scalemin \
    > /tmp/probe_d0-bisect-scalemin.log 2>&1 &
echo "Launched D0+scalemin — PID $!"

# ---- +Bug-F (aniso shrink) only ----
nohup modal run --detach scripts/train_modal.py "${BASE[@]}" \
    --sh-degree 0 --split-anisotropic-shrink \
    --run-tag d0-bisect-bugF \
    > /tmp/probe_d0-bisect-bugF.log 2>&1 &
echo "Launched D0+BugF — PID $!"

echo ""
echo "All 6 launched."
