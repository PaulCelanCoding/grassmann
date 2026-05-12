#!/usr/bin/env bash
# Frontal init test on the current best baseline (Bug-F + uncapped max-aspect-ratio
# + DSSIM, the recipe winner from results/rca/blur_rca.md update 5 -- LPIPS 0.411,
# PSNR 24.38 dB, walltime 305s on slice-banana).
#
# Test: does initializing the 4D-plane normal n = (0, d_hat_view) — so the
# rank-2 splat disk faces the init camera at t=0 — help vs the current
# spatial_slice init (n = e_0)?

set -euo pipefail
cd "$(dirname "$0")/.."

# Mirrors the recipe winner (commit 208d24c) — every flag identical except
# --init-strategy is varied per run.
BASE_FIXED=(
    --cmd smoke --dataset nerfies --scene slice-banana
    --iters 14000 --log-every 500 --seed 42
    --split-convention deformable_interp
    --sigma-init-sq 0.02 --lambda-frob 1e-4
    --densify-every 200 --densify-start 500 --densify-stop 10000
    --grad-threshold 1e-5 --spatial-split-threshold 0.5
    --opacity-prune-threshold 1e-3 --sh-degree 3 --lr-decay 0.01
    --clamp-mode soft
    --grassmann-relax-start 1000 --grassmann-relax-end 8000
    --max-aspect-ratio 1000000 --random-background
    --opacity-reset-every 3000
    --scale-min-prune 5e-3
    --lambda-aniso 0
    --temporal-split-threshold 0.1
    --split-anisotropic-shrink
    --structural-kind ssim
)

# ---- Control: current best baseline (spatial_slice init) ----
nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --init-strategy spatial_slice \
    --run-tag bugF-best-spatial \
    > /tmp/probe_bugF-best-spatial.log 2>&1 &
echo "Launched CONTROL (spatial_slice init) — PID $!"

# ---- Test: frontal init (n_xyz = view-ray d_hat from init cam) ----
nohup modal run --detach scripts/train_modal.py "${BASE_FIXED[@]}" \
    --init-strategy frontal \
    --run-tag bugF-best-frontal \
    > /tmp/probe_bugF-best-frontal.log 2>&1 &
echo "Launched TEST   (frontal init)        — PID $!"

echo ""
echo "Both detached. Each ~5 min on L4."
echo "Evaluate LPIPS afterwards via:"
echo "  modal run scripts/bugF_vs_d3dgs_modal.py \\"
echo "    --bugf-ckpt nerfies-slice-banana-<init>-14000it-<tag>/trained_nerfies_<init>.pt \\"
echo "    --out-dir comparisons/<tag>_lpips"
