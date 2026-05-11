#!/usr/bin/env bash
# Bug G (merge), I (post-Schur trigger), J (offset 1.6 + no shrink),
# H-inverse (brighter children) probes on top of Bug-D baseline.
#
# Bug-D baseline: val=24.62 dB, N=46.8k, wall=210s
# Bug F result:   val=24.93 dB, N=86.6k, wall=371s (capacity confound being verified)

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

# ---- Bug I: post-Schur Σ_3D_t for trigger ---------------------------
nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --trigger-post-schur \
    --run-tag bug-I-postschur \
    > /tmp/probe_bug-I-postschur.log 2>&1 &
echo "Launched Bug I (post-Schur trigger) — PID $!"

# ---- Bug J: split offset 1.6 + no shrink ----------------------------
nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --split-offset-sigmas 1.6 \
    --split-shrink-factor 1.0 \
    --run-tag bug-J-noshrink-off16 \
    > /tmp/probe_bug-J-noshrink-off16.log 2>&1 &
echo "Launched Bug J (offset 1.6 + no shrink) — PID $!"

# ---- Bug G: nearest-neighbor merge every 5 cycles -------------------
# densify_every=200 → merge every 1000 iters. Distance 0.02 in scene units
# (small enough that only true near-duplicates merge).
nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --merge-every 5 \
    --merge-distance 0.02 \
    --merge-normal-cos 0.95 \
    --run-tag bug-G-merge \
    > /tmp/probe_bug-G-merge.log 2>&1 &
echo "Launched Bug G (merge) — PID $!"

# ---- Bug H-inv: children BRIGHTER ----------------------------------
nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --split-opacity-brighter \
    --run-tag bug-Hinv-brighter \
    > /tmp/probe_bug-Hinv-brighter.log 2>&1 &
echo "Launched Bug H-inv (brighter children) — PID $!"

echo ""
echo "All 4 launched."
