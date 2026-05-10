#!/usr/bin/env bash
# Address each Bug A-E from results/rca/trigger_audit.md.
# Baseline: P-rca-7 (val=24.50 dB, N=45k, wall=287s).
# Each probe varies ONE flag from the new candidate baseline.

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
)

# ---- Bug A: tighten scale_max from 100 → 2.0 -----------------------
nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --lambda-aniso 1e-3 --temporal-split-threshold 0.1 \
    --scale-max-prune 2.0 \
    --run-tag bug-A-scalemax2 \
    > /tmp/probe_bug-A-scalemax2.log 2>&1 &
echo "Launched Bug A (scale_max 2.0) — PID $!"

# ---- Bug C: μ_t OOB prune at [-0.05, 1.05] -------------------------
nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --lambda-aniso 1e-3 --temporal-split-threshold 0.1 \
    --mu-t-min -0.05 --mu-t-max 1.05 \
    --run-tag bug-C-mutoob \
    > /tmp/probe_bug-C-mutoob.log 2>&1 &
echo "Launched Bug C (μ_t OOB prune) — PID $!"

# ---- Bug D: drop lambda_aniso (was 1e-3, set 0) --------------------
nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --lambda-aniso 0 --temporal-split-threshold 0.1 \
    --run-tag bug-D-anisooff \
    > /tmp/probe_bug-D-anisooff.log 2>&1 &
echo "Launched Bug D (lambda_aniso=0) — PID $!"

# ---- Bug E: lower temporal_split_threshold 0.1 → 0.03 --------------
nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --lambda-aniso 1e-3 --temporal-split-threshold 0.03 \
    --run-tag bug-E-tsplit003 \
    > /tmp/probe_bug-E-tsplit003.log 2>&1 &
echo "Launched Bug E (tsplit 0.03) — PID $!"

echo ""
echo "All 4 launched. Tail with: tail -f /tmp/probe_bug-*.log"
