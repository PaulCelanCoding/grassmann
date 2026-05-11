# Blur RCA — why our PSNR looks good but the renders are smoothed

**Date:** 2026-05-11
**Branch:** monocular-init
**Scope:** slice-banana, Bug-F recipe (current dynamic best, val PSNR 24.93 dB).
**Goal:** identify mechanisms in the current pipeline that trade high-frequency
sharpness for low-order pixel-statistic agreement — i.e. ways the recipe could
be reaching its PSNR by blurring.

## TL;DR — the signature is already in the data

`results/rca/bugF_vs_d3dgs_v2/bugF_vs_d3dgs_14k_v2/summary.json` (Bug-F vs
Deformable-3DGS, same scene, same split, apples-to-apples eval):

| metric | Bug-F (ours) | D3DGS | Δ |
|---|---|---|---|
| val PSNR | 25.10 | 25.38 | **−0.28 dB** (≈ noise) |
| val L1 | 0.0344 | 0.0321 | −0.0023 |
| **val LPIPS (AlexNet)** | **0.569** | **0.389** | **+0.180 (+46 %)** |
| train LPIPS | 0.553 | 0.374 | +0.179 (+48 %) |

ΔPSNR ≈ 0, ΔLPIPS +46 %. That asymmetry is *only* consistent with renders
that match low-order pixel statistics (mean, variance) while losing
high-frequency structure (edges, texture). The number to fix is LPIPS;
the PSNR target is already met.

## Update 4 (2026-05-11): leave-one-out flag bisection — **`--max_aspect_ratio` IS the lever**

Three single-flag retrains on the Bug-F recipe, each evaluated with LPIPS via
`bugF_vs_d3dgs_modal.py`:

| run | flag change | val PSNR | val LPIPS | Δ LPIPS | gap closed |
|---|---|---|---|---|---|
| Bug-F baseline | — | 25.10 | 0.569 | — | — |
| bisect-noOPreset | `--opacity_reset_every 0` | 24.68 | 0.566 | −0.003 | ~2 % (noise) |
| **bisect-maxAR200** | `--max_aspect_ratio 200` | **25.10** | **0.505** | **−0.064** | **~35 %** |
| bisect-relaxEarly | `--grassmann_relax_end 1000` | (in flight) | — | — | — |
| D3DGS reference | — | 25.38 | 0.389 | — | — |

**Lever identified**: raising the aspect-ratio cap from 30 → 200 closes
**~35 % of the LPIPS gap to D3DGS** at zero PSNR cost (25.10 → 25.10). The
3-plane Σ_3D(t_0) is a rank-2 disk; with aspect cap 30 the disk's two
in-plane eigenvalues are constrained to within 30× of each other, so the
disk cannot align as a thin sliver along sharp edges. Raising to 200
lets the disks elongate enough to act as edge-aligned splats.

This is the mechanism `phaseC_3db_gap.md` previously attributed to "rank-2
+ ε I lift" — but the lift wasn't the lever (Update 3 above); the aspect
constraint was. The Σ_3D shape *itself* (its in-plane anisotropy) is what
needed unlocking. **The recipe's blur was a result of constraining the
disks to be too circular.**

`noOPreset` is null at LPIPS (PSNR -0.42 dB without LPIPS gain). `relaxEarly`
result pending; will be appended below.

Follow-ups in flight (testing whether the lever saturates and stacks):
- `--max_aspect_ratio 500` (single flag, saturation test)
- `--max_aspect_ratio 200 --structural_kind ssim` (additive K1+maxAR test)

---

## Update 3 (2026-05-11): σ_3d_blur sweep is flat — blur is **baked into the trained Σ_3D**

Eval-time σ_3d_blur sweep on the same Bug-F ckpt (no retraining, identical
parameters and split):

| σ_3d_blur | val PSNR | val LPIPS |
|---|---|---|
| 1e-5 | 25.0988 | 0.5688 |
| **1e-4 (baseline)** | **25.0988** | **0.5688** |
| 1e-3 | 25.0987 | 0.5688 |
| 1e-1 (sanity) | **23.79** | **0.662** |

100× sweep around the baseline returns **bitwise-identical** PSNR/LPIPS to 4
decimal places. The 1e-1 sanity test (10000× baseline) **does** degrade both
metrics — so the test pipeline works; the lift is just numerically inactive
at the realistic range because the trained Σ_3D eigenvalues dominate it
(typical scene-scale ≫ 1e-6 variance).

**Mechanistic conclusion**: the blur signature lives in the trained Σ_3D
covariance itself, not in any rendering-pipeline knob. Recipe options to
modify it are: minimum scale floor, aspect cap, opacity reset cadence,
n-LR schedule (which controls how aggressively the rank-2 disk orientation
can move). Final bisection (Update 4 below) tests those.

---

## Update 2 (2026-05-11): iso-N refutes the **capacity** hypothesis too

Ran Bug-F + `--densify_stop 5000 --grad_threshold 2e-4` (3DGS defaults, bundled).
Final N = 5173 (vs Bug-F's 86k; far below D3DGS's 30k — flags throttled too hard).

| metric | Bug-F (N=86k) | iso-N (N=5.2k) | D3DGS (N=30k) | Δ iso-N vs Bug-F |
|---|---|---|---|---|
| val PSNR | 25.10 | 23.92 | 25.38 | **−1.18 dB** |
| val LPIPS | 0.569 | **0.718** | 0.389 | **+0.149 (worse)** |

**Same-algorithm comparison kills the over-densification story**: at much
*lower* N our recipe gets **worse** LPIPS (+0.15) and worse PSNR. The
earlier exclusion argument ("D3DGS 30k → 0.39, Bug-F 86k → 0.57, therefore
more N is worse") conflated algorithm with capacity. With algorithm held
fixed, **higher N is better for LPIPS** in our system — the recipe is
capacity-hungry.

Both Section #1 (boxstats) and Section #2 (over-densification) are now
disconfirmed as the primary mechanism. ~12 % of the gap is boxstats; the
rest is unattributed. The remaining best-supported hypothesis is
**architectural**: under the 3-plane G(3,4) projector, Σ_3D(t_0) is
rank-2 by construction (a disk in 3D). The Phase-C 3-dB-gap RCA already
attributed 1.37 dB PSNR residual to "rank-2 + ε I lift"
(`results/rca/phaseC_3db_gap.md`); the same constraint plausibly drives
LPIPS too. Capacity-hunger is the fingerprint: rank-2 disks need ~3×
more splats to approximate what D3DGS's rank-3 ellipsoids do natively.

Next eval-time experiment: `sigma_3d_blur` sweep on the same Bug-F ckpt
to test whether the rank-2 → rank-3 lift contributes to LPIPS. No
retraining.

---

## Update 1 (2026-05-11): K1 LPIPS measured — boxstats is **NOT** the primary lever

Re-ran `bugF_vs_d3dgs_modal.py` against the K1 ckpt (Bug-F + `--structural_kind ssim`):

| metric | Bug-F (boxstats) | K1 (ssim) | D3DGS | Δ K1 vs Bug-F | gap closed to D3DGS |
|---|---|---|---|---|---|
| val PSNR | 25.10 | 24.66 | 25.38 | **−0.44 dB** | — |
| val LPIPS (ours GT) | 0.569 | 0.549 | 0.389 | **−0.020 (−3.5 %)** | **~12 %** of 0.180 gap |
| val LPIPS (d3gt) | 0.481 | 0.457 | 0.287 | −0.024 | ~12 % of 0.194 gap |

The SSIM swap moves LPIPS marginally in the right direction at a non-trivial
PSNR cost. **It is a small-effect lever, not the primary mechanism.**
~88 % of the LPIPS gap remains unexplained. The Section #1 hypothesis below
is **partially supported but disconfirmed as the headline**.

The exclusion argument now elevates Section #2: same scene, same eval, D3DGS
at N ≈ 30k gets LPIPS 0.39; Bug-F at N ≈ 86k gets LPIPS 0.57. **More
Gaussians + worse sharpness** is consistent with over-densification → semi-
transparent overlap → alpha-blend smoothing.

The next decisive experiment is the iso-N capacity probe (Section "What to
do next" #3, promoted to step 1 below).

---

## Two mechanisms with mechanistic + empirical support

### 1. `structural_kind="boxstats"` — 20 % of the loss is edge-blind (MINOR LEVER, confirmed)

Default photometric loss is `lambda_l1=0.8 * L1 + lambda_structural=0.2 *
structural`, with `structural_kind="boxstats"` (`grassmann/training.py:193`,
`grassmann/losses.py:99–125`). `structural_loss` is

    |local_mean_r − local_mean_t| + |local_var_r − local_var_t|

over a 7×7 box. This term is **mathematically invariant to edge phase**: a
blurred render with matching local mean and variance has near-zero
structural loss. Real SSIM has a cross-covariance term
`(2σ_rt + C2) / (σ_r² + σ_t² + C2)` that *does* penalize phase
misalignment; boxstats does not. So 20 % of the gradient signal is
actively rewarding "match the average, ignore the structure", which is
exactly the behaviour the LPIPS gap reveals.

**Partial empirical evidence (incomplete):** `scripts/launch_advisor_cheap.sh`
launched K1 = Bug-F + `--structural_kind ssim`. The Modal run completed
(`/tmp/probe_bugF-K1-dssim.log` → val_psnr 24.49 dB, ckpt at
`/checkpoints/nerfies-slice-banana-spatial_slice-14000it-bugF-K1-dssim/`). K1
**loses 0.44 dB PSNR** vs Bug-F (24.93 → 24.49). This is consistent with the
hypothesis — SSIM forces the optimizer to align edges instead of just matching
means, which costs PSNR because edges are harder to register exactly than
smoothed regions are.

**Gap:** LPIPS was never measured on the K1 ckpt. The training-time log only
reports PSNR + L1; `scripts/eval_per_frame.py` measures only PSNR + L1;
`scripts/bugF_vs_d3dgs_modal.py` *does* compute LPIPS but was only run on the
Bug-F (boxstats) ckpt. **The discriminating measurement does not exist.**
Without LPIPS on K1 we cannot confirm SSIM-loss is the lever; we only know
SSIM gives up PSNR for *something*, and the "something" is plausibly
sharpness.

### 2. Bug-F over-densification — capacity-driven smoothing

`results/rca/static_baseline/SUMMARY.md` D0 bisection already showed
half of Bug-F's PSNR gain on multi-frame is **capacity-driven, not
quality-driven** — `--split-anisotropic-shrink` doubles split frequency.
On the full bundle our recipe uses `--grad-threshold 1e-5`, which is **20×
lower than 3DGS's 2e-4**, so densification fires almost everywhere, not
just at high-frequency seeds. The combination produces a large population
of small, overlapping Gaussians (final N = 86.6k vs D3DGS's ~30k).

Many small overlapping Gaussians composited with alpha-blending behave
*as a blur kernel*: each pixel sees a sum of overlapping low-opacity
contributions, smoothed across edges. PSNR is preserved because the per-pixel
mean is correct; LPIPS sees the smoothing.

iso-N controls (`scripts/launch_bugF_isoN_controls.sh`) already established
that Bug-F's gain is partially capacity. We have not measured *whether the
extra capacity blurs* — that would require LPIPS on iso-N ablations.

## Mechanisms that probably aren't load-bearing

- **`sigma_3d_blur = 1e-4` (`grassmann/fast_rasterizer.py:187`)** — added
  isotropic 3-D blur Σ_3D(t₀) += (1e-4)²·I. The variance is 1e-8 in scene
  units; Phase-C swept ±10× and saw only ±0.3 dB PSNR
  (`results/rca/phaseC_3db_gap.md:245–246`). LPIPS not measured at those
  settings, but the magnitude is small. Worth one cheap probe at 1e-5 +
  LPIPS, not the headline.
- **`sigma_k_pixel = 1.0`** — only the toy CPU rasterizer uses it
  (`grassmann/rasterizer.py:80`). Training/eval use `fast_rasterize`, which
  ignores `sigma_k_pixel`. Dead lever.
- **`mip_filter_sigma_pixel`** — disabled by default (=0). Not active.
- **`sh_degree=3`, `lr_decay=0.01`, `opacity_reset_every=3000`** — no
  mechanistic path to blur specifically. D0 bisection showed
  `lr_decay` is load-bearing for N control, not sharpness.

## What to do next (cheap, ordered by information value)

**Status after iso-N**: both knob-level mechanisms (boxstats, capacity) are
disconfirmed. Working hypothesis shifts to **architectural**: rank-2 Σ_3D(t_0)
under 3-plane projector parameterization.

1. **σ_3d_blur eval-time sweep on the same Bug-F ckpt** (NO retraining,
   ~5 min). Re-render at σ_3d_blur ∈ {1e-5, 1e-4, 1e-3} and measure LPIPS.
   - Sensitive → the rank-2 → rank-3 isotropic lift is part of the blur;
     there may be an LPIPS optimum elsewhere.
   - Flat → the lift is not the lever; conclusion strengthens toward
     architectural-ceiling.
2. **Instrument permanently** (DONE): LPIPS added to
   `scripts/eval_per_frame.py` so future training evals report it
   alongside PSNR/L1.
3. **If σ_3d_blur is flat — leave-one-out flag bisection on LPIPS**:
   - `--opacity_reset_every 0`: stop wiping opacity every 3k iters.
   - `--max_aspect_ratio 200` (or larger): allow elongated splats needed
     for sharp edges.
   - `--grassmann_relax_end 1000`: let n_lr ramp up earlier so the projector
     orientation has time to move.
   One flag per run, eval with LPIPS via the patched eval path.
4. **If everything is flat — accept the architectural ceiling.** The
   3-plane parameterization gives rank-2 disks where D3DGS has rank-3
   ellipsoids. iso-N data is the fingerprint: our recipe is capacity-hungry
   because rank-2 disks need ~3× more splats to approximate rank-3
   structure. Recipe knobs cannot close the gap. Next step is upstream
   (parameterization change), out of scope for an RCA.

## Files / evidence

- `results/rca/bugF_vs_d3dgs_v2/bugF_vs_d3dgs_14k_v2/summary.json` —
  source of the +46 % LPIPS gap finding.
- `grassmann/losses.py:99–125` — the boxstats structural loss.
- `grassmann/losses.py:64–96` — the SSIM alternative (already implemented).
- `grassmann/training.py:193` — `structural_kind` default = "boxstats".
- `scripts/launch_advisor_cheap.sh` — K1/K2/K12 launcher (already ran).
- `/tmp/probe_bugF-K1-dssim.log` — K1 result: val_psnr 24.49 dB
  (−0.44 dB vs Bug-F). LPIPS not measured.
- `/checkpoints/nerfies-slice-banana-spatial_slice-14000it-bugF-K1-dssim/trained_nerfies_spatial_slice.pt`
  — K1 ckpt, ready for LPIPS measurement.

## Caveat

The RCA is **not yet closed**: K1 ruled out boxstats as the headline
mechanism (only 12 % of the gap), but no intervention has yet closed the
remaining 88 %. The iso-N probe (Step 1 in "What to do next") is required
before treating any conclusion as established.
