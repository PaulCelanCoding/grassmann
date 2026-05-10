#!/usr/bin/env bash
# #4.1 MCMC probe set v2 — high-init-capacity variants.
# Hypothesis: pure-MCMC failed in v1 (val~19.5 dB) because density_strategy=mcmc
# only relocates (no growth), so N stays at the SfM count (~14k). Kheradmand 2024
# starts with much higher N. Match by --init-points-multiplier.

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
    --opacity-prune-threshold 1e-3
    --sh-degree 3
    --lr-decay 0.01
)

# ---- P-mcmc-5: pure MCMC + 4× init capacity (≈55k init) ---------------------
nohup modal run --detach scripts/train_modal.py "${A1_FLAGS[@]}" \
    --init-strategy random \
    --init-points-multiplier 4 \
    --density-strategy mcmc \
    --mcmc-noise-lr 5e-5 \
    --run-tag mcmc-init4x-noise5e5 \
    > /tmp/probe_mcmc-init4x-noise5e5.log 2>&1 &
echo "Launched P-mcmc-5 (mcmc + init 4× ≈55k + noise 5e-5) — PID $!"

# ---- P-mcmc-6: pure MCMC + 8× init capacity (≈110k init) --------------------
nohup modal run --detach scripts/train_modal.py "${A1_FLAGS[@]}" \
    --init-strategy random \
    --init-points-multiplier 8 \
    --density-strategy mcmc \
    --mcmc-noise-lr 5e-5 \
    --run-tag mcmc-init8x-noise5e5 \
    > /tmp/probe_mcmc-init8x-noise5e5.log 2>&1 &
echo "Launched P-mcmc-6 (mcmc + init 8× ≈110k + noise 5e-5) — PID $!"

echo ""
echo "P-mcmc-5/6 launched. Tail with: tail -f /tmp/probe_mcmc-init*.log"
