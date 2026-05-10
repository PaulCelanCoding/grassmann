#!/usr/bin/env bash
# Deeper RCA on opacity-driven death. The reset target (logit=-5 → opacity 0.0067)
# sits ABOVE the prune threshold (1e-3 → logit -6.9), so reset doesn't directly
# kill anyone. Test three fixes that align them.

set -euo pipefail
cd "$(dirname "$0")/.."

A1_FLAGS=(
    --cmd smoke
    --dataset nerfies
    --scene slice-banana
    --iters 14000
    --log-every 500
    --seed 42
    --split-convention deformable_interp
    --sigma-init-sq 0.02
    --lambda-frob 1e-4
    --lambda-aniso 1e-3
    --densify-every 200
    --densify-start 500
    --densify-stop 10000
    --grad-threshold 1e-5
    --spatial-split-threshold 0.5
    --sh-degree 3
    --lr-decay 0.01
    --init-strategy spatial_slice --clamp-mode soft
    --grassmann-relax-start 1000 --grassmann-relax-end 8000
    --max-aspect-ratio 30 --random-background
    --temporal-split-threshold 0.1
    --opacity-reset-every 3000
)

# ---- P-rca-2: raise prune threshold to 0.005 (default reset target ≈ 0.0067 still above)
nohup modal run --detach scripts/train_modal.py "${A1_FLAGS[@]}" \
    --opacity-prune-threshold 0.005 \
    --run-tag rca-thr5e3 \
    > /tmp/probe_rca-thr5e3.log 2>&1 &
echo "Launched P-rca-2 (threshold 0.005, reset 0.0067 above) — PID $!"

# ---- P-rca-3: raise prune threshold to 0.01 (above reset target → all post-reset below) ----
nohup modal run --detach scripts/train_modal.py "${A1_FLAGS[@]}" \
    --opacity-prune-threshold 0.01 \
    --run-tag rca-thr1e2 \
    > /tmp/probe_rca-thr1e2.log 2>&1 &
echo "Launched P-rca-3 (threshold 0.01, reset 0.0067 below) — PID $!"

# ---- P-rca-4: keep threshold 1e-3, lower reset target to logit=-8 (opacity 3.4e-4 below thr)
nohup modal run --detach scripts/train_modal.py "${A1_FLAGS[@]}" \
    --opacity-prune-threshold 1e-3 \
    --opacity-reset-logit -8 \
    --run-tag rca-resetm8 \
    > /tmp/probe_rca-resetm8.log 2>&1 &
echo "Launched P-rca-4 (reset logit -8 below threshold) — PID $!"

echo "Tail with: tail -f /tmp/probe_rca-*.log"
