#!/usr/bin/env bash
# Static-baseline floor probes on the current best recipe (P-rca-7 + Bug-D + Bug-F).
# Current dynamic best: 24.93 dB @ N=86.6k, wall=371s.
# Plan: /home/xyz/.claude/plans/wir-w-rden-zu-debugging-tender-hartmanis.md
#
#   S0       = Bug-F recipe + --static_baseline
#   S0-clean = S0 minus motion-only knobs (no temporal-split-threshold, no grassmann-relax)
#   D0       = Bug-F recipe + --diag_single_frame 100 (forces static_baseline ON; upper-bound)

set -euo pipefail
cd "$(dirname "$0")/.."

# Shared BASE_FIXED from launch_bugF_isoN_controls.sh, MINUS the two motion-only knobs
# (added back per-probe so S0-clean can drop them).
BASE_COMMON=(
    --cmd smoke --dataset nerfies --scene slice-banana
    --iters 14000 --log-every 500 --seed 42
    --split-convention deformable_interp
    --sigma-init-sq 0.02 --lambda-frob 1e-4
    --densify-every 200 --densify-start 500 --densify-stop 10000
    --grad-threshold 1e-5 --spatial-split-threshold 0.5
    --opacity-prune-threshold 1e-3 --sh-degree 3 --lr-decay 0.01
    --init-strategy spatial_slice --clamp-mode soft
    --max-aspect-ratio 30 --random-background
    --opacity-reset-every 3000
    --scale-min-prune 5e-3
    --lambda-aniso 0
    --split-anisotropic-shrink
)

# ---- S0: Bug-F recipe + static_baseline ----------------------------
nohup modal run --detach scripts/train_modal.py "${BASE_COMMON[@]}" \
    --grassmann-relax-start 1000 --grassmann-relax-end 8000 \
    --temporal-split-threshold 0.1 \
    --static-baseline \
    --run-tag static-s0 \
    > /tmp/probe_static-s0.log 2>&1 &
echo "Launched S0 (Bug-F + static_baseline) — PID $!"

# ---- S0-clean: drop motion-only knobs (tsplit + grelax) ------------
nohup modal run --detach scripts/train_modal.py "${BASE_COMMON[@]}" \
    --static-baseline \
    --run-tag static-s0-clean \
    > /tmp/probe_static-s0-clean.log 2>&1 &
echo "Launched S0-clean (Bug-F + static, no tsplit/grelax) — PID $!"

# ---- D0: single-frame upper bound (forces static internally) -------
nohup modal run --detach scripts/train_modal.py "${BASE_COMMON[@]}" \
    --grassmann-relax-start 1000 --grassmann-relax-end 8000 \
    --temporal-split-threshold 0.1 \
    --diag-single-frame 100 \
    --run-tag static-d0-frame100 \
    > /tmp/probe_static-d0-frame100.log 2>&1 &
echo "Launched D0 (Bug-F + diag_single_frame=100) — PID $!"

echo ""
echo "All 3 launched. Tail with:"
echo "  tail -f /tmp/probe_static-s0.log /tmp/probe_static-s0-clean.log /tmp/probe_static-d0-frame100.log"
