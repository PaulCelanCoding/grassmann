# Wave A probe results — 11 quality knobs vs A1 anchor

**Date:** 2026-05-10
**Branch:** monocular-init
**Anchor:** A1 reproduction at commit `e514bc9` — slice-banana, scale-4 train, deformable_interp, seed 42, SH3, 14k iters, LR-decay 0.01, λ_frob=1e-4, λ_aniso=1e-3, densify-every 200, grad-thr 1e-5.

## Anchor (A1) under current code

| metric | value |
|---|---|
| internal val PSNR (scale 4) | **23.50 dB** |
| train PSNR (final, scale 4) | 24.32 dB |
| final N | 22910 |
| wall (Modal L4) | 413 s |

**Note**: this run's val PSNR is 0.91 dB below the surfel-A/B-era reproduction (24.41 dB at commit `b958b68`). Wallclock is also 65% longer (413 s vs 249 s). The drift is unexplained — possibly a Modal infrastructure change or an upstream rasterizer-package version drift. **All Δ values below are vs THIS reproduction (23.50 dB), not the historical 24.41 / 25.82 dB anchor.**

The apples-to-apples scale-8 GT files at `/tmp/d3dgs_gt/gt/` were not present locally, so we report internal val PSNR (scale 4) only. To get scale-8 apples numbers, regenerate the GT (val frames at scale 8 from the dataset) and re-run `eval_apples.py`.

## Probe slate (single-flag flip vs A1)

All probes: identical A1 recipe + one new flag. 14k iters, seed 42.

| # | probe | new flag |
|---|---|---|
| 1 | **#5.2 color-LR warmup** | `--color_lr_warmup_iter 1000` |
| 2 | **#7.2 random background** | `--random_background` |
| 3 | **#3.1 k-NN σ_init** | `--sigma_init_knn_k 3 --sigma_init_alpha_t 0.1` |
| 4 | **#6.2 hard aspect-ratio clip** | `--max_aspect_ratio 30` |
| 5 | **#1.1 per-frame exposure** | `--exposure_per_frame --lambda_exposure_reg 1e-3` |
| 6 | **#4.2 temporal-axis split** | `--temporal_split_threshold 0.01` |
| 7 | **#5.3 time-coherence reg** | `--lambda_time_coherence 0.1 --time_coherence_dt 0.05` |
| 8 | **#7.1 Mip-Splatting 3D filter** | `--mip_filter_sigma_pixel 0.3` |
| 9 | **#2.1 pose refinement** | `--refine_poses --pose_warmup_iter 2000 --lr_pose_rot 1e-5 --lr_pose_trans 1e-4` |
| 10 | **#3.2 progressive Grassmann relax** | `--init_strategy spatial_slice --clamp_mode soft --grassmann_relax_start 1000 --grassmann_relax_end 5000` |
| 11 | **#8.1 floater multi-view pruning** | `--floater_min_views 5` (DEFERRED — 11th probe, queued after first batch) |

**DEFERRED**: #9.1 DepthAnythingV2 — needs Modal image rebuild (transformers + HF cache), depth-render path in fast_rasterizer, scale-shift alignment in losses. Out of scope for this session.

## Results

[FILLED IN ON COMPLETION — see /tmp/probes_summary.log]

| probe | val PSNR (s4) | Δ vs A1 (23.50) | train PSNR (s4) | final N | wall (s) | conclusion |
|---|---|---|---|---|---|---|
| A1 anchor | **23.50** | — | 24.32 | 22910 | 413 | reference |
| P-5.2 color-warmup | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| P-7.2 random-bg | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| P-3.1 knn-σ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| P-6.2 aspect30 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| P-1.1 exposure | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| P-4.2 tsplit | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| P-5.3 tcoh | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| P-7.1 mip | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| P-2.1 poseref | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| P-3.2 grelax | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |

## Implementation summary

All flags landed on branch `monocular-init` (uncommitted at the time of probe launch). Code touched:

- `grassmann/training.py` — TrainerConfig fields; LR warmup/relax schedules; pose+exposure params + optimizer groups; `_perturbed_camera()`; `clip_aspect_ratio_()` invocation; time-coherence loss term.
- `grassmann/trainable.py` — `clip_aspect_ratio_()` method on `TrainableGaussians`.
- `grassmann/initialization.py` — `compute_knn_sigma_init_sq()` + per-point σ²_init plumbing.
- `grassmann/density_control.py` — `temporal_split()` method; floater multi-view pruning in `prune()`.
- `grassmann/fast_rasterizer.py` — `mip_filter_sigma_pixel` field + per-Gaussian σ²·I addition.
- `scripts/train_mono.py` — CLI flags for all 11 items.
- `scripts/train_modal.py` — Modal-side parameter pass-through.

## Bugs encountered (and fixed) during launch

1. **`lr_R` / `lr_t` Modal CLI lowercasing** — Modal converts `--lr-R` to `lr_r` (kwarg), but Python signature had `lr_R` (uppercase). Renamed to `lr_pose_rot` / `lr_pose_trans`.
2. **`_perturbed_camera` recursion** — `replace_all` swap of `self.cameras[cam_idx]` → `self._perturbed_camera(cam_idx)` hit the helper itself. Fixed.

Both bugs only surfaced on Modal; local tests would catch them in a re-run.
