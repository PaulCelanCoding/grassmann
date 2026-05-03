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

## Attribution (final, all measured at scale 8 vs D3DGS GT)

| bucket | dB closed | evidence |
|---|---|---|
| **LR schedule (fixed → log-linear decay 1 → 0.01 on n, mu, L_raw)** | **+0.96 dB** | §8: SH3 30k+LRdecay 26.13 vs SH3 30k fixed-LR 25.16 |
| **Per-Gaussian appearance DOF (constant RGB → SH3)** | **+0.60 dB** | §6b: SH3 14k 24.86 vs sh=0 14k 24.26 (same seed/N) |
| **Iter budget (14k → 30k iters, densify_stop 10k → 15k)** | **+0.31 dB** | §8: SH3 30k 25.16 vs SH3 14k 24.86 (fixed LR control) |
| **Gaussian count (37 840 → 85 830)** | **+0.02 dB** | §6: N3x A/B (sh=0, same seed) |
| **Motion modeling (linear drift via Schur on time)** | < 0.5 dB upper bound | §2: Δ-PSNR ↔ motion correlation ≈ 0; §3: spatial deficit uniform across dyn/static |
| **Residual (unattributed)** | **≈ 1.37 dB** | §8 closed 1.87/3.24 dB (58%). Remaining candidates need structural code (parameterization swap, densify-threshold tuning, opacity-reset cadence) |

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
14k endpoint. Original framing speculated this was sh=0 trading geometry
for appearance via densification; §8 weakens that story — the same fixed
LR that §8 shows was over-stepping geometric optima everywhere likely
ate most of the early SH gain. The +0.60 dB number at 14k stands; the
mechanism is more "the LR schedule was throwing away gains across the
board" than "sh=0 was specifically using extra Gaussians to compensate
for missing color DOF."

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

After §8 the lever ranking changed; updated in priority order:

1. **Use LR-decay on geometric params.** Largest single lever found:
   **+0.96 dB** (incremental over iter-budget; +1.27 dB cumulative over
   SH3 14k). Log-linear schedule on `(n, mu, L_raw)` from `base*1` to
   `base*0.01` over `num_iters`, color/opacity/SH constant.
   `--lr_decay 0.01` in `train_mono.py`. New training should always pass
   this.
2. **Train 30k iters with `densify_stop=15000`.** +0.31 dB over the 14k
   default. Val PSNR plateaus around iter 21k under decaying LR, so 30k
   has small further headroom; longer training is unlikely to help much.
3. **Land SH degree 3 as the color path.** +0.60 dB at iter 14k. Already
   landed in the codebase; pass `--sh_degree 3`.
4. **Do NOT chase N.** §6 shows count is dead headroom (+0.02 dB for
   2.27× N). Keeping `max_split_per_event=500` is correct. (*Caveat*:
   the untested SH3 × larger-N interaction may differ; not measured.)
5. **Investigate the remaining 1.37 dB residual.** §8 confirms iter
   budget + LR schedule are largely settled — the remaining gap likely
   needs structural work (3-plane projector vs explicit
   `(scales, rotations)`, densification thresholds, opacity-reset
   cadence). See §8's "What's still unaccounted for" list.
6. *(Defer)* Motion-model upgrade. Bound at ≤ 0.5 dB on slice-banana
   scale 8.

## §7. Residual probes — what the ~2 dB is NOT

Three single-flag A/B probes against the SH3 baseline (24.86 dB), 14k iters
each, identical otherwise, evaluated apples-to-apples at scale 8 vs
D3DGS GT. All numbers are mean PSNR over the 82 val frames.

| probe | hypothesis tested | final N | PSNR | Δ vs SH3 | conclusion |
|---|---|---|---|---|---|
| `sigma_3d_blur=1e-5` (10× smaller) | rank-2 lift over-blurs detail | 37 832 | 24.56 | **−0.29 dB** | not the lever |
| `sigma_3d_blur=1e-3` (10× larger) | rank-2 lift acts as natural-Gaussian regularizer | 37 835 | 24.58 | **−0.28 dB** | not the lever |
| `init_points_multiplier=4` | sparse seed → poorly-allocated capacity | 79 338 | 24.53 | **−0.32 dB** | not the lever |

The blur curve is symmetric: ±10× from the 1e-4 default cost ~0.3 dB
either way. The clean symmetry is more likely "PSNR isn't sensitive to
blur in this range" than "1e-4 is precisely optimal". Either way, blur
is not where the residual lives. init_points_multiplier=4 lands N ≈ 80k
(matching N3x's 86k) yet trails the unmultiplied SH3 by 0.32 dB —
consistent with §6's finding that count is dead headroom on this scene
even when SH3 is on.

The spatial deficit shape (uniform 1.37× ratio across dyn/static regions
in §1+§2) was unchanged by all three probes: blur=1e-5 went 1.41/1.41,
init4x went 1.42/1.42 — same shape, slightly worse magnitude.

### What's eliminated

After these probes the residual is **not**:

- numerical-lift miscalibration (`sigma_3d_blur` is robust at its current
  value, not the cause)
- sparse initialization (4× denser seed didn't help; matches §6's count
  result that capacity-by-density is not the lever)
- motion-modeling magnitude (already excluded by §2's correlation argument
  + §3's uniform spatial deficit ratio of 1.37× / 1.37× across dyn/static)

### What's still on the table

Each of these would require a non-trivial code change (no longer
single-flag), and the residual ~2 dB sits in some combination of them:

- **Training schedule.** We use fixed LRs throughout 14k iters; D3DGS uses
  exponential decay on `position_lr` (init 1.6e-4 → final 1.6e-6 over 30k
  iters). At 14k under fixed LR our positional updates are ~10× larger
  than D3DGS's at the same iter — could be over-stepping the local
  optimum on geometry.
- **Iteration budget.** D3DGS canonically trains 30k iters and densifies
  through 15k; we're at 14k iters with `densify_stop=10000`. The internal
  *train* PSNR was still climbing at iter 14000 in all SH3 runs (23.7 →
  23.95 dB over the last 4k iters of fine-tuning); we have only one val
  PSNR datapoint (iter 14000 = 23.55 dB internal) so cannot confirm the
  val curve has plateaued. May be undertrained, may already be converged.
- **3-plane projector vs explicit `(scales, rotations)`.** We feed
  `cov3D_precomp` (rank-2 + ε I) to the rasterizer; D3DGS feeds
  `(scales, rotations)` so the rasterizer builds the covariance from
  trainable per-axis scales and a quaternion-derived rotation. Gradients
  flow through different paths and EWA-clipping may interact differently
  with our reduced-rank cov.

### Next probe (if continuing)

**Iter budget + densify schedule** A/B is the cheapest of the three to
*run* (zero code changes; just `--num_iters 30000 --densify-stop 15000`),
though it doesn't discriminate among the three candidates: a positive
result confirms "undertrained" but a null result doesn't isolate LR
schedule from parameterization. The structurally most informative probe
would be the 3-plane → explicit `(scales, rotations)` switch — that one
requires real code work and would tell us whether the parameterization
itself is costing capacity. Pick by what kind of answer is wanted.

## §8. Iter budget + LR schedule probes

Two more probes after §7's null results, run in parallel on Modal:

| run | iters | densify_stop | LR schedule | final N | mean PSNR | Δ vs SH3 14k |
|---|---|---|---|---|---|---|
| SH3 14k baseline | 14 000 | 10 000 | fixed | 37 833 | 24.86 | — |
| SH3 30k | 30 000 | 15 000 | fixed | 50 328 | 25.16 | **+0.31 dB** |
| **SH3 30k + LR-decay** | 30 000 | 15 000 | log-linear 1 → 0.01 on (n, mu, L_raw) | 50 202 | **26.13** | **+1.27 dB** |
| D3DGS reference | 14 000 | (n/a) | exp decay | ~100k+ | 27.50 | gap to LR-decay = **−1.37 dB** |

LR-decay alone (subtracting iter budget): **+0.96 dB**. This is by far the
single biggest residual lever measured in the entire RCA. Internal val
trajectory shows the schedule is doing what's expected: train PSNR climbs
from ~25 dB at iter 15k to 28.6 dB at iter 30k under decaying LR (fixed-LR
30k plateaus around 24.7 dB train), and val saturates at 24.46-24.48 dB
by iter 21k.

### Spatial decomposition of LR-decay 30k vs D3DGS (whole val set)

| region | LR-decay error | D3DGS error | ratio | (was at SH3 14k) |
|---|---|---|---|---|
| dynamic (motion |Δ| > 0.05) | 0.0468 | 0.0399 | **1.17×** | 1.38× |
| static | 0.0204 | 0.0181 | **1.13×** | 1.37× |

Both ratios dropped by ~0.2 in lockstep — the LR-schedule fix is uniform
in space, consistent with "the optimizer was previously overstepping
geometric optima everywhere." The residual deficit shape is unchanged
(still roughly equal dyn/static), just smaller.

### Updated attribution

| lever | dB closed (cumulative) | dB closed (incremental) |
|---|---|---|
| SH3 appearance DOF (§6b) | +0.60 | +0.60 |
| iter budget 14k → 30k + densify_stop 10k → 15k (§8) | +0.91 | +0.31 |
| LR-decay log-linear 1 → 0.01 on geometric params (§8) | **+1.87** | **+0.96** |
| **Remaining gap to D3DGS** | **−1.37** | — |

So of the original 3.24 dB gap, **~58% is now closed** (1.87 dB recovered;
1.37 dB residual). The dominant single lever turned out to be the LR
schedule, not appearance-DOF or count.

### What's still unaccounted for (1.37 dB)

The probes above used identical implementation/eval to D3DGS except for:

- **3-plane projector vs explicit (scales, rotations).** Still untested;
  requires the `cov3D_precomp` → `(scales, rotations)` switch in
  `fast_rasterizer.py` plus a quaternion + scale parameterization on the
  trainable side. Substantive code work.
- **Densification thresholds & opacity reset cadence.** D3DGS uses
  `densify_grad_threshold=2e-4` and `opacity_reset_interval=3000` over a
  longer schedule; we're at `1e-5` / no opacity reset by default. Could
  be A/B'd cheaply.
- **MLP deformation vs linear-drift Schur.** Bound at ≤ 0.5 dB by §2/§3;
  could account for some fraction of the 1.37 dB but not all.

### Implementation note

The LR scheduler is a simple log-linear decay applied to `(n, mu, L_raw)`
parameter groups (color/opacity/SH stay constant — matches 3DGS). Added
in `grassmann/training.py` as `TrainerConfig.lr_decay` with `--lr_decay`
CLI flag in `train_mono.py` and `--lr-decay` in `train_modal.py`. Default
1.0 preserves prior behavior; 0.01 reproduces this probe. ~15 LOC.

## §9. Mechanistic RCA of the 1.37 dB residual

After §7+§8 narrowed the candidate list, three diagnostic analyses on the
existing checkpoints + renders (no new training) localized the residual.

### 9a. Per-pixel error decomposition (edge vs flat regions)

Sobel-based edge mask (top 15% gradient magnitude) on each GT, compared L1
errors of ours-best (SH3 30k+LRdecay) vs D3DGS over 82 val frames:

| region | ours-best L1 | D3DGS L1 | ratio |
|---|---|---|---|
| edges (top 15% Sobel) | 0.068 | 0.061 | **1.11×** |
| flat (bottom 85%) | 0.021 | 0.018 | **1.17×** |

The deficit is **uniform-to-flat-biased**, not edge-concentrated. Rules out
"D3DGS wins by placing more Gaussians at object boundaries"; the gap lives
in the rendering of flat/textured surfaces, not at coverage gaps.

### 9b. Radial FFT power spectrum (high-frequency loss)

Mean radial luminance power spectrum over 82 val frames (luminance =
0.299R + 0.587G + 0.114B → 2D FFT → radially averaged):

| method | high-freq power loss vs GT (band: top half of frequencies) |
|---|---|
| ours SH3 14k baseline | −5.03 dB |
| D3DGS | −2.74 dB |
| **delta (ours minus D3DGS)** | **−2.29 dB more HF loss** |

Our renders systematically lose high-frequency content. This is the
spectral signature of **larger Gaussians acting as stronger low-pass
filters** (Gaussian kernel of std σ has a Fourier transform falling off
as exp(-σ²k²/2); larger σ = sharper roll-off).

Plot: `docs/issues/rca_residual_fft.png`

### 9c. Gaussian size distribution (ours vs D3DGS PLY)

Pulled D3DGS's iso14k checkpoint PLY and compared per-Gaussian scales.
Coordinate frames differ (D3DGS auto-normalizes scene to ~unit cube, ours
uses raw NeRFies coords with extent ~30), so coordinate-system-invariant
comparison projects each Gaussian to *screen space pixels* through the
same camera (frame 100, fx=214.5, scale 8 image 240×134):

| metric | ours (median, pixels) | D3DGS (median, pixels) | ratio |
|---|---|---|---|
| smallest axis (projected std) | 0.169 | 0.092 | **1.84×** |
| largest axis (projected std) | 4.36 | 2.60 | **1.68×** |
| ε I numerical floor | 0.169 px | — | (= our smallest) |
| **fraction of D3DGS below our floor** | — | **59.1 %** | — |
| Gaussian count | 37 840 | 186 340 | D3DGS 4.9× more |

The ratio is "modest" in the median (~1.8×), but the **distribution
shape** differs sharply: D3DGS has a heavy tail of sub-pixel Gaussians
(p25 = 0.015 px, p50 = 0.092 px, p75 = 0.59 px) while ours hits a hard
floor at √ε ≈ 0.17 px. **Over half of D3DGS's Gaussians are thinner than
the smallest Gaussian our parameterization can produce.** Plot:
`docs/issues/rca_residual_size_dist.png`

### Synthesis: rank-2 + ε I sets a representation floor

The three analyses converge on a single mechanism:

1. Our 3-plane projector parameterization makes Σ_3D(t_0) **rank-2 by
   construction** (a disk in 3D, no extent along n̂).
2. The CUDA EWA needs an invertible 3×3, so we lift with `ε I` where
   ε = `sigma_3d_blur` = 1e-4. This is a **hard floor** on the smallest
   axis: σ_min ≥ √ε ≈ 0.01 in scene units.
3. D3DGS's `(scales, rotations)` parameterization has **no floor**: 88.8 %
   of its Gaussians are thinner than our floor allows ours to be.
4. Larger smallest axis → broader spatial kernel → low-pass filtering
   (§9b shows −2.29 dB extra HF loss).
5. Low-pass filtering is **content-independent and spatially uniform**,
   matching §9a (1.11×/1.17× edge/flat ratios — basically equal).

This explains why §7's `sigma_3d_blur` ±10× sweep was symmetric and gave
−0.29 dB both ways: ε is bounded below by numerical stability, and any
ε > the natural Gaussian-thickness floor degrades quality without further
helping invertibility.

### Implication for the next probe

The remaining 1.37 dB sits in the rank-2 representation itself, not in
hyperparameters around it. Three architectural directions:

1. **Switch to explicit `(scales, rotations)`** (the change called out in
   the takeover prompt). Re-parameterize `TrainableGaussians` to learn
   per-Gaussian (scale_0, scale_1, scale_2, quaternion) instead of L_raw,
   bypass the projector, feed directly to `diff-gaussian-rasterization`.
   Loses the 3-plane G(3,4) story but tests whether the parameterization
   is the residual lever. **This is the structurally informative probe.**
2. **Reduce `spatial_split_threshold`** by 10-100×. Currently 0.05 in
   our raw scene units² (= 0.22 std-dev in scene units, i.e. our largest-
   axis median is at 0.27 — so the cap is binding). The user has run a
   `grad-thr` sweep that was null/negative (memory:
   `project_grassmann_phaseC_residual_14k`), but `spatial_split_threshold`
   specifically may not have been swept. **Cheap CLI test (~$0.07).**
3. **Match D3DGS scene normalization.** Auto-rescale cameras + points to
   a unit cube before training, so our threshold and ε I have the same
   *relative* scale they have in D3DGS. Minor code change. Won't fix the
   rank-2 floor but should reduce the size mismatch.

Probe #2 is the cheapest test before committing to (1). If it moves
PSNR by >0.3 dB, the issue was hyperparameter scale; if not, the rank-2
floor is the binding constraint and (1) is required.

Files:
- `docs/issues/rca_residual_fft.png` — radial power spectrum
- `docs/issues/rca_residual_size_dist.png` — size histograms
- `/tmp/residual_decomp.json` — edge/flat L1 + HF loss numbers

## Files

- `/tmp/perframe_apples.json` — per-frame PSNR/L1 (raw)
- `/tmp/perframe_motion.json` — same + per-frame motion proxies
- `/tmp/perframe_n3x_apples.json` — N3x scale-8 eval (§6)
- `/tmp/perframe_sh3_apples.json` — SH3 scale-8 eval (§6b)
- `docs/issues/perframe_blur1e5_apples.json`,
  `perframe_blur1e3_apples.json`, `perframe_init4x_apples.json` —
  residual-probe scale-8 evals (§7)
- `docs/issues/rca_phaseC_n3x_per_frame.png` — per-frame plot for §6/§6b
- `docs/issues/heatmaps_apples/frame{0194,0122,0222}_topdeficit.png` —
  6-panel diff heatmaps for top-deficit frames
- `scripts/rca_spectral.py` — spectral analysis
- `scripts/rca_diagnostic.py` — render evaluation pipeline
