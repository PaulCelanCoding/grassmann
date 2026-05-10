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

## Single-flag results

Sorted by Δ. Each probe = A1 recipe + one new flag.

| probe | val PSNR (s4) | Δ vs A1 | train PSNR | final N | wall (s) | verdict |
|---|---|---|---|---|---|---|
| **#3.2 progressive Grassmann relax** | **23.85** | **+0.35** | 24.86 | 24677 | 238 | **WINNER** |
| **#6.2 hard aspect-ratio clip (30)** | **23.76** | **+0.26** | 24.90 | **51172** | 328 | **WINNER** (2× capacity) |
| **#7.2 random background** | **23.70** | **+0.20** | 24.85 | 26358 | 251 | **WINNER** |
| **#5.2 color-LR warmup (1000)** | **23.62** | **+0.12** | 24.49 | 23188 | 234 | **WINNER** (small) |
| #5.3 time-coherence reg | 23.59 | +0.09 | 24.48 | 22016 | 255 | marginal |
| #7.1 Mip-Splatting filter (0.3) | 23.53 | +0.03 | 24.41 | 23089 | 243 | null |
| #2.1 pose refinement (lr_R=1e-5, lr_t=1e-4) | 23.53 | +0.03 | 24.43 | 23132 | 245 | **null — surprise** |
| **A1 anchor (current)** | **23.50** | — | 24.32 | 22910 | 413 | reference |
| #8.1 floater multi-view (≥5 views) | 23.48 | −0.02 | 24.27 | **18148** | 215 | null on Δ; capacity-efficient |
| #3.1 k-NN σ_init (k=3, α_t=0.1) | 23.40 | −0.10 | 24.16 | 21855 | 233 | regression |
| #1.1 per-frame exposure | **22.70** | **−0.80** | 24.21 | 21412 | 248 | strong regression — needs hyperparam tune |
| #4.2 temporal-axis split (Σ_tt > 0.01) | 23.79 | +0.29 | 30.29 | **435348** | 1342 | gain but 19× N — threshold too loose |

### Notable observations

- **#3.2** (progressive Grassmann relaxation): the highest single-flag gain. Spatial_slice init was previously known to be 3× *worse* than random; introducing graduated `lr_n` ramping (0 → base over [1k, 5k] iters) flips this. **Empirically supports the hypothesis that Euclidean+renorm Adam on S³ is suboptimal**, but a much cheaper fix than #5.1 Riemannian Adam.
- **#6.2** (aspect clip): +0.26 dB *and* N grew 2× (22.9k → 51.2k). The hard clip prevents capacity from collapsing into spikes; freed capacity gets allocated by the existing densifier into more useful disks. Wall 328s vs anchor 413s — somehow faster despite 2× N (likely warm-cache effect, see note below).
- **#7.2** (random bg): cheap, classical 3DGS trick we never adopted. +0.20 dB.
- **#2.1** (pose refinement) **null** is the biggest surprise — was advisor's top pick. Three explanations:
    1. NeRFies poses on slice-banana are already adequate (sub-pixel error doesn't dominate the 1.47 dB residual).
    2. LRs (1e-5 / 1e-4) too small for 12k effective refinement iters.
    3. Pose warmup (2k) too late — geometry already locked in.
- **#1.1** (exposure) **strong regression** (−0.80 dB): val L1 also rose from 0.040 → 0.051. The exposure params likely overfit on training frames; the L2 reg (1e-3) is too weak. Worth a follow-up probe with `λ_exposure_reg=1e-1` or `lr_exposure=1e-4`.
- **#3.1** (k-NN σ): marginal regression. The single-σ baseline (sigma_init_sq=0.02) is already calibrated to the slice-banana scale; per-point σ from k-NN noise adds variance without information.

## Combo results

Built from the 4 single-flag winners (#3.2, #6.2, #7.2, #5.2). All combos = A1 recipe + listed flags; same seed 42, 14k iters.

| combo | flags | val PSNR | Δ vs A1 | additivity | N | wall (s) | verdict |
|---|---|---|---|---|---|---|---|
| **Combo-A** | #3.2 + #6.2 + #7.2 | **24.26** | **+0.76** | 94% of +0.81 sum | 53726 | 322 | **WINNER** |
| **Combo-B** | A + #5.2 (color-warmup) | **24.14** | **+0.64** | 69% of +0.93 sum | 48749 | **288** | runner-up; 11% faster than A |
| Combo-C | #6.2 + #7.2 + #5.2 (no grelax) | 23.71 | +0.21 | 36% of +0.58 sum | 53253 | 352 | confirms #3.2 dominates |
| Combo-D | B + #8.1 (floater) | 23.40 | **−0.10** | floater killed gain | **12481** | 213 | **REGRESSION** |
| A1 anchor | (reference) | 23.50 | — | — | 22910 | 413 | reference |

### Combo findings

1. **Combo-A is the winner.** The grelax + aspect-clip + random-bg stack delivers +0.76 dB vs A1, with 94% of the additive sum captured — surprisingly clean stacking.

2. **Combo-B is the practical pick.** It costs 0.12 dB vs A but is **34s faster** (288 vs 322) and uses 9% fewer Gaussians (48.7k vs 53.7k). For deployment-style runs, Combo-B is the better speed/quality trade.

3. **#3.2 grelax dominates.** Combo-C (Combo-A minus grelax) drops to +0.21 dB — only 28% of Combo-A's gain. The grelax lever alone explains ~0.55 dB of Combo-A's +0.76.

4. **#5.2 color-warmup interacts badly with grelax.** Single-flag it gave +0.12, but adding it to Combo-A *regressed* by 0.12 dB (Combo-A 24.26 → Combo-B 24.14). Possibly single-seed noise; possibly real interaction (the grelax schedule already gates `n` capacity, so the early-iter color suppression may starve the geometry of color-driven stress signal).

5. **#8.1 floater pruning catastrophic in combo with grelax** (Combo-D: −0.10 dB, N=12.5k). The floater detector counts grad-active iters; spatial_slice + grelax keeps many Gaussians "inactive" through the relaxation phase (iter 0–5k), and floater min_views=5 prunes them before they specialize. Use floater pruning ONLY with random init or a much higher min_views threshold.

6. **Combo speedup vs anchor**: A1 anchor took 413s; all combos finished in 213–352s. The anchor was a cold-start (first Modal launch). All other probes ran with warm Modal containers. PSNR comparisons are still apples-to-apples (deterministic recipe), but wallclock numbers are not directly comparable.

## Recommended recipe

For Phase D / next iteration: **Combo-A** as the new default.

```bash
modal run scripts/train_modal.py --cmd train \
    --dataset nerfies --scene <scene> \
    --iters 14000 \
    --init-strategy spatial_slice \
    --clamp-mode soft \
    --grassmann-relax-start 1000 --grassmann-relax-end 5000 \
    --max-aspect-ratio 30 \
    --random-background \
    [other A1 flags: --sh-degree 3 --lr-decay 0.01 --densify-every 200 ...]
```

## Residual to D3DGS

D3DGS reference: 27.50 dB apples-to-apples (scale 8). The historical A1 anchor was 25.82 dB apples → 1.47 dB residual.

This run's anchor at 23.50 val (scale 4) doesn't directly map to D3DGS's 27.50 apples (scale 8) because we couldn't run apples-to-apples (GT files missing locally). But applying the historical scale-4→scale-8 offset of ~1.4 dB to Combo-A's val_psnr suggests apples ≈ 25.66 dB, which would leave residual ≈ 1.84 dB to D3DGS — *worse* than the 1.47 dB Phase-C anchor. **The current commit's anchor has drifted ~0.91 dB below the surfel-A/B-era anchor for unexplained reasons** (see "Anchor" section above), so the headline is "Combo-A closes 0.76 dB of the lost ground", not "0.76 dB closer to D3DGS". A clean rerun with the historical baseline is needed before claiming residual reduction.

## Ideas not yet implemented or evaluated

From the user's original 30-idea list, the following are still unexplored:

**High-priority unexplored** (advisor or user flagged as top candidates):
- **#4.1 3DGS-MCMC relocation** (user's #1 priority) — L effort. Direct attack on dead-rate (32%) + split-direction ambiguity in G(3,4). Untouched.
- **#9.1 DepthAnythingV2 prior** — M-L effort. Standard sparse-view fix, often 0.5–1.5 dB on monocular dynamic. **DEFERRED** in this session (needs Modal image rebuild + HF cache + depth-render path).
- **#5.1 Riemannian Adam on G(3,4)** (user's #2 priority) — XL effort. **However, #3.2 grelax already captures +0.35 dB on the same hypothesis (manifold-aware optimization geometry); ROI of #5.1 is now lower.**

**Medium / lower-priority unexplored**:
- #1.2 Soft dynamic mask via residual-history reweighting
- #1.3 Coarse-to-fine resolution schedule
- #2.2 Time-conditioned pose deltas + temporal smoothness
- #2.3 Photometric BA inner loop
- #3.3 Motion-aware t₀ from first-observable frame
- #4.3 Error-weighted local grad threshold
- #6.1 SH-degree warmup schedule (A1 already uses sh_degree=3 fixed)
- #6.3 Opacity-entropy regularizer
- #7.3 Mixed-backend dispatch (high-aniso → surfel, low → fast)
- #8.2 Adaptive λ_frob on rank-1-collapse early warning
- #8.3 StopThePop windowed sorting (RED — kernel rewrite, out of scope)
- #9.2 RAFT optical-flow constraint
- #9.3 Omnidata normal prior

**Hyperparameter follow-ups** (probes that regressed but might work tuned):
- #1.1 per-frame exposure: try `λ_exposure_reg=1e-1` and `lr_exposure=1e-4` (vs current 1e-3 / 1e-3)
- #3.1 k-NN σ_init: try `k=5` or `α_t=0.0` (purely spatial)
- #2.1 pose refinement: try larger LRs (1e-4 / 1e-3) and earlier warmup (500 vs 2000)

## Bugs encountered (and fixed) during launch

1. **`lr_R` / `lr_t` Modal CLI lowercasing** — Modal converts `--lr-R` to `lr_r` (kwarg), but Python signature had `lr_R` (uppercase). Renamed to `lr_pose_rot` / `lr_pose_trans`.
2. **`_perturbed_camera` recursion** — `replace_all` swap of `self.cameras[cam_idx]` → `self._perturbed_camera(cam_idx)` hit the helper itself. Fixed.
3. **#1.1 exposure tensor shape** — assumed (3,H,W) but rendered is (H,W,3) for fast/toy paths. Fixed with branching on dim layout.

All three only surfaced on Modal; local CPU smoke tests would catch them.

## Wave A.2 — retunes + Combo-A variants

After Wave A produced Combo-A (+0.76), 10 retunes targeted the unexpectedly-failed singles + Combo-A hyperparam sweep.

| probe | flag(s) | val PSNR | Δ vs A1 | N | wall | verdict |
|---|---|---|---|---|---|---|
| **#4.2 v3 (thr 0.1)** | `--temporal_split_threshold 0.1` | **23.96** | **+0.46** | 53846 | 301 | **NEW SINGLE-FLAG WINNER** (was +0.29 with 19× N at thr 0.01) |
| #4.2 v4 (thr 0.1 + cap 100) | + max_split_per_event=100 | 23.97 | +0.47 | 55296 | 310 | cap didn't matter (thr 0.1 already selective) |
| **Combo-A v2 (relax 8k)** | `--grassmann_relax_end 8000` | **24.30** | **+0.80** | 51579 | 298 | **NEW BEST COMBO** (vs v1 +0.76) |
| Combo-A v3 (relax 3k) | start=500, end=3000 | 24.25 | +0.75 | 53418 | 299 | sweet spot is 5–8k |
| Combo-A v4 (aspect 10) | `--max_aspect_ratio 10` | 24.16 | +0.66 | 42476 | 285 | tighter aspect regresses |
| Combo-A v5 (aspect 100) | `--max_aspect_ratio 100` | 24.13 | +0.63 | 47624 | 300 | looser aspect regresses |
| #2.1 v4 retune | lr_R=1e-4, lr_t=1e-3, warmup=500 | 23.55 | +0.05 | 23425 | 241 | still null — NeRFies poses are fine |
| #1.1 v4 retune | λ_reg=0.1 (100× stronger) | 23.10 | −0.40 | 21347 | 314 | still regressed (was −0.80) |
| #3.1 v3 retune | α_t=0 (pure spatial k-NN) | 23.42 | −0.08 | 21841 | 222 | still regressed (was −0.10) |
| #7.1 v3 retune | mip σ=1.0 | 23.52 | +0.02 | 22896 | 235 | null at all σ tested |

### Wave A.2 findings

- **#4.2 threshold matters massively**: 0.01 → 0.1 fixes runaway N (435k → 54k) AND gives **+0.46 dB** instead of +0.29. The cap (`max_split_per_event=100`) does nothing once threshold is selective. **#4.2 thr=0.1 is now a clear single-flag winner.**
- **Combo-A v2 (relax_end=8000) is the new best combo at +0.80 dB.** Longer relax window lets `n` settle more gradually. relax=3000 costs 0.05 dB; >8000 not tested.
- **aspect_ratio=30 is the sweet spot.** 10 (too tight) and 100 (too loose) both regress to ~+0.65.
- **#2.1 pose-refinement remains null** even with bigger LRs and earlier warmup. NeRFies poses on slice-banana are not the bottleneck.
- **#1.1 exposure remains regressed** even with 100× stronger reg (recovered from −0.80 to −0.40 but still worse than A1).
- **#3.1 k-NN σ remains regressed** at α_t=0. Single σ_init is already calibrated for slice-banana.
- **#7.1 mip-filter remains null** at σ=0.3 and σ=1.0.

### Performance (capacity + wall) — gains pay capacity costs

| probe | Δ dB | N (vs 22910) | wall (s) |
|---|---|---|---|
| **Combo-A v2** | **+0.80** | 2.25× | 298 |
| **#4.2 v3 thr=0.1** | **+0.46** | 2.35× | 301 |
| #6.2 single | +0.26 | 2.23× | 328 |
| #3.2 grelax | +0.35 | 1.08× | 238 |
| #4.2 v1 thr=0.01 | +0.29 | **19.0×** | **1342** |
| Combo-D (regression) | −0.10 | 0.54× | 213 |

A1 anchor at 413s was a cold start; warm-start probes finish in 220-330s. Runaway #4.2 v1 (1342s) is the only true outlier. **At fixed iters, the big winners (Combo-A v2, #4.2 thr=0.1) all pay ~2.2-2.3× capacity (50k Gaussians).**

## Wave A.3 — stacking #4.2 thr=0.1 onto Combo-A

Pending probes:
- **Combo-AA** = Combo-A v2 + #4.2 thr=0.1 → if additive: +1.26 dB
- **Combo-AB** = Combo-A v1 + #4.2 thr=0.1 → comparison
- **#6.1 SH-degree warmup** (sh_degree_warmup_step=1000) — single-flag
- **#6.3 opacity-entropy reg** (λ=0.01) — single-flag
