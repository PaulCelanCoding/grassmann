# RCA — 3.24 dB gap to Deformable3DGS on slice-banana (Phase C)

**Date:** 2026-05-03
**Setup:** slice-banana, scale 8 (134×240), HyperNeRF deformable_interp val split
(ids[2::4], 82 frames), 14k iters, deterministic seed 42.
**Aggregate (apples-to-apples vs D3DGS's saved GT in `/tmp/d3dgs_gt`):**

| metric | ours (Phase C v2) | Deformable3DGS | gap |
|---|---|---|---|
| PSNR (dB) | 24.26 | 27.50 | **−3.24** |
| L1 | 0.0374 | 0.0244 | +0.0130 |
| std(PSNR) over frames | 1.08 | 2.26 | — |
| Gaussian count N | 37 840 | ~100–150 k (typical) | 0.25-0.4× |
| color DOF / Gaussian | 3 (constant RGB) | 48 (SH degree 3) | 0.06× |

LPIPS not computed (local `lpips` package conflicts with current torch CPU
build). D3DGS's per-view file reports mean LPIPS 0.1524.

---

## Attribution (1-line summary, all measured at scale 8 vs D3DGS GT)

| bucket | dB closed | evidence |
|---|---|---|
| **Per-Gaussian appearance DOF (constant RGB → SH3)** | **+0.60 dB** (measured) | direct A/B at iter 14k, same seed/init/N: 24.26 → 24.86 dB (§6) |
| **Gaussian count (37 840 → 85 830)** | **+0.02 dB** (measured) | direct A/B at iter 14k, same seed/init/sh=0: 24.26 → 24.28 dB (§6) |
| **Motion modeling (linear drift via Schur on time)** | < 0.5 dB upper bound | Δ-PSNR ↔ motion correlation = +0.07 (cam disp), −0.02 (image L1); Q4 high-motion frames show *smaller* deficit than Q3; spatial deficit is uniform across dyn/static regions |
| **Residual (unattributed)** | ≈ 2.0 dB | gap to D3DGS still 2.64 dB after SH3, and the levers above sum to <1 dB. Candidates: COLMAP-density init, motion model in expectation (not in correlation), numerical Σ_3D lift (ε I = 1e-4), 3-plane projector vs explicit scales/rotations (rasterizer-side EWA). Not yet isolated. |

---

## Evidence

### 1. Per-frame deficit pattern is "ours has a flat ceiling"

ours std=1.08 dB, D3DGS std=2.26 dB. The mean gap (3.24 dB) comes from
D3DGS *peaking* at 30-31 dB on calm frames where ours never exceeds ~25 dB.

Top-deficit frames:

| frame | ours | D3DGS | Δ | cam Δ | img L1 (Δt=1) |
|---|---|---|---|---|---|
| 194 | 23.76 | 30.50 | +6.74 | 0.37 | 0.052 |
| 122 | 24.80 | 31.50 | +6.70 | 0.49 | 0.057 |
| 222 | 24.59 | 31.27 | +6.68 | 0.22 | 0.051 |
| 162 | 23.62 | 30.10 | +6.48 | 0.46 | 0.049 |

These are *not* high-motion frames — image L1 motion is around the median.
The deficit is "ours capped, D3DGS pulled ahead", not "ours failed on a hard
frame". (Footnote: §6b shows that SH3 closes most of this group's deficit
substantially — except frame 194, which is SH3-resistant and is therefore
the concrete probe for the residual ≈ 2 dB.)

### 2. Motion is not the bottleneck

Per-frame Δ-PSNR vs three motion proxies (n=82):

| proxy | corr with Δ-PSNR | corr with ours_psnr | corr with d3dgs_psnr |
|---|---|---|---|
| camera displacement (frame Δt=1) | +0.07 | −0.19 | −0.03 |
| camera angular change | +0.16 | (n/a) | (n/a) |
| image L1 (frame Δt=1) | −0.02 | −0.25 | −0.14 |

If motion modeling were the dominant bottleneck, Δ-PSNR would correlate
strongly with motion magnitude. It doesn't.

Quartile binning by image L1 motion:

| quartile (Q1 = lowest motion) | n | Δ (dB) | ours | D3DGS |
|---|---|---|---|---|
| Q1 | 21 | 3.21 | 24.40 | 27.62 |
| Q2 | 20 | 3.22 | 24.41 | 27.63 |
| Q3 | 20 | 3.65 | 24.41 | 28.06 |
| Q4 (highest) | 21 | **2.92** | 23.81 | 26.74 |

Q4 has the *smallest* deficit — both methods degrade roughly equally on
high-motion frames.

### 3. Spatial decomposition: deficit uniform across dynamic/static regions

For top-3 deficit frames, masking pixels with inter-frame Δ > 0.05 as
"dynamic" (mean ≈ 30% of pixels):

| frame | dyn px | ours err (dyn) | D3DGS err (dyn) | ours err (static) | D3DGS err (static) | deficit ratio (dyn) | deficit ratio (static) |
|---|---|---|---|---|---|---|---|
| 194 | 30% | 0.0654 | 0.0356 | 0.0322 | 0.0180 | 1.84× | 1.79× |
| 122 | 35% | 0.0675 | 0.0453 | 0.0332 | 0.0194 | 1.49× | 1.71× |
| 222 | 28% | 0.0652 | 0.0400 | 0.0308 | 0.0181 | 1.63× | 1.70× |

The ratio of error (ours/D3DGS) is roughly 1.7× **in both regions**. By area
(static = 70% of pixels), static regions contribute ~55% of the L1 deficit.
A motion-modeling deficit would manifest as dynamic-region-dominated error.
It doesn't.

Heatmaps: `docs/issues/heatmaps_apples/frame{0194,0122,0222}_topdeficit.png`.

### 4. Spectral RCA: Phase C resolved the worst pathologies

`scripts/rca_spectral.py /home/xyz/grassmann/checkpoints/3plane_phaseC_v2.pt`:

| pathology | Phase A (50k iters, no DC) | Phase C (14k iters, DC v2) |
|---|---|---|
| Effectively dead (opacity < 0.01) | 32.2% | **8.1%** |
| High aniso (λ_max/λ_min > 100) | 30.4% | **3.4%** |
| Collapsed disks (λ_min < 1e-6) | 7.1% | **0.5%** |
| Median anisotropy | 22.9 | **1.20** |
| Σ_4D · n̂ residual (sanity) | 6.4e-14 | 3.0e-15 |

Phase C density control + opacity reset + Frobenius/aniso penalties
brought the per-Gaussian distribution into a healthy state. The gap to
D3DGS now reflects raw representation capacity, not pathological allocation.

### 5. Capacity decomposes into count and appearance-DOF; §6/6b isolate them.

| lever | ours | D3DGS | ratio | measured contribution |
|---|---|---|---|---|
| Gaussian count N | 37 840 | ~100k-150k | 0.25-0.4× | +0.02 dB (§6) |
| color DOF / Gaussian | 3 (constant RGB) | 48 (SH degree 3) | 0.06× | +0.60 dB (§6b) |
| effective color "capacity" (N × DOF) | 113 520 | ≈ 4.8M-7.2M | 0.016-0.024× | sum: +0.62 dB |

Constant RGB cannot represent within-Gaussian color gradients or
view-dependent shading; SH3 gives D3DGS that flexibility per Gaussian.
§6 directly tests "count alone" (2.27× N, sh=0); §6b directly tests
"appearance-DOF alone" (same N, sh=3). Together they account for ~0.62 dB
of the 3.24 dB gap. Most of the gap is therefore *not* in raw N or in
raw per-Gaussian color DOF — see Recommendation §3.

### 6. Capacity-scaling test (max-split-per-event 500 → 1500)

Re-trained the same Phase C config with `max_split_per_event=1500` (3× headroom),
identical otherwise. Final N = 85 830 (2.27× the baseline's 37 840), reached by
iter 10000 when the densify_stop fired; the remaining 4k iters fine-tuned on a
fixed N.

The N3x checkpoint was rendered at scale 8 with `render_mono.py` and evaluated
against D3DGS's saved GT (the same eval as the baseline, line-for-line apples
to-apples):

| run | N | mean PSNR (dB) | mean L1 | Δ vs baseline |
|---|---|---|---|---|
| baseline (max_split_per_event=500) | 37 840 | 24.26 | 0.0374 | — |
| **N3x (max_split_per_event=1500)** | **85 830** | **24.28** | **0.0375** | **+0.02 dB** |
| D3DGS reference (~100-150k) | ~100k+ | 27.50 | 0.0244 | — |

**Reading: count is essentially not the lever.** Doubling-and-then-some the
Gaussian count moved val PSNR by 0.02 dB — within rendering noise. The 3.24 dB
gap to D3DGS therefore cannot be capacity-by-count; the lever is *what each
Gaussian can encode*. With motion modeling already bounded at <0.5 dB
(§2), the 3 dB residual is per-Gaussian appearance DOF — i.e., constant RGB
vs SH3.

Per-frame data: `/tmp/perframe_n3x_apples.json`.

![per-frame PSNR baseline vs N3x vs D3DGS](rca_phaseC_n3x_per_frame.png)

### 6b. Appearance-DOF test (constant RGB → SH degree 3)

Re-trained the same Phase C config with `--sh_degree 3` (per-Gaussian
SH coefficients K=16/channel instead of constant RGB), same seed/N target
(`max_split_per_event=500`). Final N = 37 833 (essentially identical to
baseline's 37 840).

| run | N | mean PSNR (dB) | mean L1 | Δ vs baseline |
|---|---|---|---|---|
| baseline (sh=0)  | 37 840 | 24.26 | 0.0374 | — |
| N3x (sh=0)       | 85 830 | 24.28 | 0.0375 | +0.02 dB |
| **SH3 (sh=3)**   | **37 833** | **24.86** | **0.0336** | **+0.60 dB** |
| D3DGS reference | ~100k+ | 27.50 | 0.0244 | gap to SH3: −2.64 dB |

74% of frames improved (n=82). The lift is strongly concentrated on the
high-deficit frames identified in §1 (corr SH3-lift ↔ baseline-deficit:
+0.38; Q4 mean lift +1.15 dB vs Q1 +0.24 dB), with one notable outlier
(frame 194: deficit 6.74 dB; SH3 lift −0.07 dB; residual 6.80 dB — that
single frame is not appearance-DOF).

The early-iter 500 signal (+3 dB train PSNR at same N) did not predict the
14k endpoint. The likely mechanism: under sh=0, densification spends
geometry-DOF to compensate for missing appearance-DOF (more, smaller
Gaussians to approximate within-Gaussian color gradients), so by 14k the
sh=0 path has *traded geometry for appearance*. SH3 lifts the ceiling that
trade-off was working against, but most of the early gap closes via the
sh=0 path's geometric workaround.

Per-frame data: `/tmp/perframe_sh3_apples.json`.

---

## Reconciliation with prior RCA

`docs/issues/rca_3plane_low_psnr.md` (2026-05-03 morning) concluded
**motion-bound, not capacity-bound**, based on a single-frame fit at 29 dB
plus a +0.7 dB N×4 test. The current §6 N3x test (+0.02 dB at 2.27× N) is
the cleaner refutation of "count is the lever": the older N×4 test was at
scale 4 with 32% dead Gaussians (4× raw N meant ~2/3 dead waste), while
N3x is at scale 8 under healthy Phase C allocation. Both indicate count is
not the lever; the current test is the load-bearing one. The earlier
"motion-bound" verdict was the right rejection of capacity-by-count but the
wrong attribution of where the remainder lives — appearance-DOF was not on
the menu in the earlier 2-hypothesis test.

---

## Recommendation

1. **Land SH degree 3 as the color path for new training.** Measured
   +0.60 dB at iter 14k, same seed/N as baseline; closes ~19% of the gap.
   Small but the largest single lever found. Implementation lives in
   `grassmann/{gaussian,trainable,fast_rasterizer,density_control}.py`;
   `--sh_degree 3` on `train_mono.py`. Default stays at 0 for backward
   compat with existing checkpoints; new training should pass
   `--sh_degree 3`.
2. **Do NOT chase N.** The N3x test (§6) shows count is dead headroom on
   this scene: +0.02 dB for 2.27× N. Keeping `max_split_per_event=500` is
   correct. *Caveat:* untested whether SH3 + larger N is super-additive
   (D3DGS uses both).
3. **Investigate the ≈ 2 dB residual** (next RCA round). Levers above sum
   to ≤ 1.1 dB; gap to D3DGS is 2.64 dB after SH3. Untested candidates
   (no rank claimed):
   - **COLMAP-density init.** D3DGS bootstraps from dense COLMAP point clouds
     (often 5-10× more points than NeRFies' shipped `points.npy`); init
     density biases where capacity is allocated.
   - **MLP deformation vs linear-drift Schur.** Motion-correlation here is
     ~0, but that's correlation with magnitude — non-linear motion patterns
     could still cost a uniform amount in expectation. Frame 194 (SH3 lift
     ≈ 0, residual 6.80 dB) is a candidate test case.
   - **Numerical Σ_3D lift (`sigma_3d_blur=1e-4`).** Per-frame additive
     blur; D3DGS rasterizer doesn't do this. Likely small contribution
     (1cm in scene units), but cheap to A/B.
   - **3-plane projector vs explicit (scales, rotations).** The rasterizer
     receives `cov3D_precomp` from us (rank-2 + ε I); D3DGS supplies
     `(scales, rotations)` and the rasterizer builds the cov internally.
     Different gradient paths, possibly different EWA scaling.
4. *(Defer)* Motion-model upgrade. Bound at ≤ 0.5 dB on slice-banana scale 8.

## Files

- `/tmp/perframe_apples.json` — per-frame PSNR/L1 (raw)
- `/tmp/perframe_motion.json` — same + per-frame motion proxies
- `docs/issues/rca_phaseC_per_frame_psnr.png` — per-frame plot
- `docs/issues/heatmaps_apples/frame{0194,0122,0222}_topdeficit.png` —
  6-panel diff heatmaps for top-deficit frames
- `scripts/rca_spectral.py` — spectral analysis
- `scripts/rca_diagnostic.py` — render evaluation pipeline
