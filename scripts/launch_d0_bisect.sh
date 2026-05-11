#!/usr/bin/env bash
# D0 bisection — isolate the 1.23 dB regression vs Phase A on
# `--diag_single_frame 100` (D0 = 27.84 dB now, Phase A = 29.07 dB).
# Plan: /home/xyz/.claude/plans/wir-w-rden-zu-debugging-tender-hartmanis.md
# RCA:  results/rca/static_baseline/SUMMARY.md "D0 single-frame regression"
#
# Probes:
#   D-A         = Phase A reproduction (random init, no DC, sh=0, lr=1) — control
#   D-noSH      = D0 recipe minus --sh_degree 3 (back to sh=0)
#   D-noLRdecay = D0 recipe minus --lr_decay 0.01 (back to 1.0)
#   D-noOPreset = D0 recipe minus --opacity_reset_every 3000 (off)
#   D-noBugF    = D0 recipe minus --split_anisotropic_shrink

set -euo pipefail
cd "$(dirname "$0")/.."

# ---- D-A: Phase A reproduction control ----------------------------
# random init, sh=0, no DC, no lr-decay, no opacity-reset, no Bug-F,
# no grelax, no aspect-clip, no random-bg, no scale-min-prune,
# no Frob/aniso regs, no Bug-F. Just init + train + diag_single_frame.
nohup modal run --detach scripts/train_modal.py \
    --cmd smoke --dataset nerfies --scene slice-banana \
    --iters 14000 --log-every 500 --seed 42 \
    --split-convention deformable_interp \
    --sigma-init-sq 0.02 \
    --init-strategy random \
    --diag-single-frame 100 \
    --run-tag d0bis-PhaseA-control \
    > /tmp/probe_d0bis-PhaseA.log 2>&1 &
echo "Launched D-A (Phase A reproduction control) — PID $!"

# ---- D-noSH: drop SH3 (sh=0) --------------------------------------
nohup modal run --detach scripts/train_modal.py \
    --cmd smoke --dataset nerfies --scene slice-banana \
    --iters 14000 --log-every 500 --seed 42 \
    --split-convention deformable_interp \
    --sigma-init-sq 0.02 --lambda-frob 1e-4 \
    --densify-every 200 --densify-start 500 --densify-stop 10000 \
    --grad-threshold 1e-5 --spatial-split-threshold 0.5 \
    --opacity-prune-threshold 1e-3 --lr-decay 0.01 \
    --init-strategy spatial_slice --clamp-mode soft \
    --grassmann-relax-start 1000 --grassmann-relax-end 8000 \
    --max-aspect-ratio 30 --random-background \
    --opacity-reset-every 3000 \
    --scale-min-prune 5e-3 \
    --lambda-aniso 0 \
    --temporal-split-threshold 0.1 \
    --split-anisotropic-shrink \
    --diag-single-frame 100 \
    --run-tag d0bis-noSH \
    > /tmp/probe_d0bis-noSH.log 2>&1 &
echo "Launched D-noSH (D0 minus SH3) — PID $!"

# ---- D-noLRdecay: drop --lr-decay (defaults to 1.0) ----------------
nohup modal run --detach scripts/train_modal.py \
    --cmd smoke --dataset nerfies --scene slice-banana \
    --iters 14000 --log-every 500 --seed 42 \
    --split-convention deformable_interp \
    --sigma-init-sq 0.02 --lambda-frob 1e-4 \
    --densify-every 200 --densify-start 500 --densify-stop 10000 \
    --grad-threshold 1e-5 --spatial-split-threshold 0.5 \
    --opacity-prune-threshold 1e-3 --sh-degree 3 \
    --init-strategy spatial_slice --clamp-mode soft \
    --grassmann-relax-start 1000 --grassmann-relax-end 8000 \
    --max-aspect-ratio 30 --random-background \
    --opacity-reset-every 3000 \
    --scale-min-prune 5e-3 \
    --lambda-aniso 0 \
    --temporal-split-threshold 0.1 \
    --split-anisotropic-shrink \
    --diag-single-frame 100 \
    --run-tag d0bis-noLRdecay \
    > /tmp/probe_d0bis-noLRdecay.log 2>&1 &
echo "Launched D-noLRdecay (D0 minus lr-decay) — PID $!"

# ---- D-noOPreset: drop opacity-reset ------------------------------
# Note: D0_FULL has --opacity-reset-every 3000. Modal CLI doesn't let
# us cleanly override to 0, so rebuild without that flag.
nohup modal run --detach scripts/train_modal.py \
    --cmd smoke --dataset nerfies --scene slice-banana \
    --iters 14000 --log-every 500 --seed 42 \
    --split-convention deformable_interp \
    --sigma-init-sq 0.02 --lambda-frob 1e-4 \
    --densify-every 200 --densify-start 500 --densify-stop 10000 \
    --grad-threshold 1e-5 --spatial-split-threshold 0.5 \
    --opacity-prune-threshold 1e-3 --sh-degree 3 --lr-decay 0.01 \
    --init-strategy spatial_slice --clamp-mode soft \
    --grassmann-relax-start 1000 --grassmann-relax-end 8000 \
    --max-aspect-ratio 30 --random-background \
    --scale-min-prune 5e-3 \
    --lambda-aniso 0 \
    --temporal-split-threshold 0.1 \
    --split-anisotropic-shrink \
    --diag-single-frame 100 \
    --run-tag d0bis-noOPreset \
    > /tmp/probe_d0bis-noOPreset.log 2>&1 &
echo "Launched D-noOPreset (D0 minus opacity-reset) — PID $!"

# ---- D-noBugF: drop --split-anisotropic-shrink --------------------
nohup modal run --detach scripts/train_modal.py \
    --cmd smoke --dataset nerfies --scene slice-banana \
    --iters 14000 --log-every 500 --seed 42 \
    --split-convention deformable_interp \
    --sigma-init-sq 0.02 --lambda-frob 1e-4 \
    --densify-every 200 --densify-start 500 --densify-stop 10000 \
    --grad-threshold 1e-5 --spatial-split-threshold 0.5 \
    --opacity-prune-threshold 1e-3 --sh-degree 3 --lr-decay 0.01 \
    --init-strategy spatial_slice --clamp-mode soft \
    --grassmann-relax-start 1000 --grassmann-relax-end 8000 \
    --max-aspect-ratio 30 --random-background \
    --opacity-reset-every 3000 \
    --scale-min-prune 5e-3 \
    --lambda-aniso 0 \
    --temporal-split-threshold 0.1 \
    --diag-single-frame 100 \
    --run-tag d0bis-noBugF \
    > /tmp/probe_d0bis-noBugF.log 2>&1 &
echo "Launched D-noBugF (D0 minus Bug-F) — PID $!"

echo ""
echo "All 5 launched. Tail with:"
echo "  tail -f /tmp/probe_d0bis-*.log"
