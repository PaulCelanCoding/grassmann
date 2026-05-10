#!/usr/bin/env bash
# Launch the #4.1 3DGS-MCMC probe set on Modal (slice-banana, 14k, scale 4).
# Matches the Wave A baseline: --cmd smoke gives image_scale=4; --iters 14000
# overrides the 500-iter smoke default. Logs → /tmp/probe_<tag>.log.

set -euo pipefail
cd "$(dirname "$0")/.."

# Wave A A1 anchor flags (verified against /tmp/probe_a1_anchor.log).
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

# ---- P-mcmc-1: pure MCMC (relocate + SGLD noise), bare A1 base --------------
nohup modal run --detach scripts/train_modal.py "${A1_FLAGS[@]}" \
    --init-strategy random \
    --density-strategy mcmc \
    --mcmc-noise-lr 5e-5 \
    --run-tag mcmc-pure-noise5e5 \
    > /tmp/probe_mcmc-pure-noise5e5.log 2>&1 &
echo "Launched P-mcmc-1 (pure MCMC + noise 5e-5) — PID $!"

# ---- P-mcmc-2: pure MCMC, relocation-only (no noise), bare A1 base ----------
nohup modal run --detach scripts/train_modal.py "${A1_FLAGS[@]}" \
    --init-strategy random \
    --density-strategy mcmc \
    --mcmc-noise-lr 0 \
    --run-tag mcmc-pure-noise0 \
    > /tmp/probe_mcmc-pure-noise0.log 2>&1 &
echo "Launched P-mcmc-2 (pure MCMC, no noise) — PID $!"

# ---- P-mcmc-3: pure MCMC + Combo-A components (grelax + aspect + random-bg) -
nohup modal run --detach scripts/train_modal.py "${A1_FLAGS[@]}" \
    --init-strategy spatial_slice --clamp-mode soft \
    --grassmann-relax-start 1000 --grassmann-relax-end 8000 \
    --max-aspect-ratio 30 \
    --random-background \
    --density-strategy mcmc \
    --mcmc-noise-lr 5e-5 \
    --run-tag mcmc-comboa-noise5e5 \
    > /tmp/probe_mcmc-comboa-noise5e5.log 2>&1 &
echo "Launched P-mcmc-3 (Combo-A + MCMC + noise 5e-5) — PID $!"

# ---- P-mcmc-4: heuristic + just SGLD noise on Combo-AA recipe ---------------
# Combo-AA = grelax(1k→8k) + aspect=30 + random-bg + tsplit=0.1
nohup modal run --detach scripts/train_modal.py "${A1_FLAGS[@]}" \
    --init-strategy spatial_slice --clamp-mode soft \
    --grassmann-relax-start 1000 --grassmann-relax-end 8000 \
    --max-aspect-ratio 30 \
    --random-background \
    --temporal-split-threshold 0.1 \
    --mcmc-noise-lr 5e-5 \
    --run-tag noise5e5-on-comboaa \
    > /tmp/probe_noise5e5-on-comboaa.log 2>&1 &
echo "Launched P-mcmc-4 (Combo-AA + SGLD noise 5e-5, heuristic densif) — PID $!"

echo ""
echo "All 4 launched. Tail logs with: tail -f /tmp/probe_mcmc-*.log /tmp/probe_noise5e5*.log"
