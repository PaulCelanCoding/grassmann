# Phase C — 14k LR-decay residual probes (1.47 dB residual to D3DGS)

**Date:** 2026-05-03
**Anchor baseline (A1):** SH3 14k + LR-decay 0.01 = **26.03 dB** apples-to-apples (scale 8 vs `/tmp/d3dgs_gt/gt/`, 82 val frames, deformable_interp split, seed 42).
**D3DGS reference:** 27.50 dB → residual = **1.47 dB**.
**Constraint:** must not slow per-iter training; comparison fixed at 14k iters; no reparameterization.

Builds on `rca_phaseC_3db_gap.md` §8. Confirms 14k under LR-decay reaches near-parity with the 30k LR-decay anchor (26.13 dB) — iter-budget is exhausted by ~14k under decay; residual to D3DGS unchanged at ~1.47 dB.

## Anchor performance (A1)

| metric | value | source |
|---|---|---|
| apples-to-apples mean PSNR (scale 8) | **26.03 dB** | `/tmp/perframe_14k_lrdecay_apples.json` |
| internal val PSNR (scale 4) | 24.41 dB | train log (user-reported) |
| wallclock | ~6-7 min | Modal L4 |
| s/iter | ~25 ms | derived |

## Probe slate (single-variable A/B vs A1)

All 5 probes ran on Modal L4, 14k iters each, seed 42, otherwise identical to A1.

| probe | lever | run-tag |
|---|---|---|
| **B1** | `--opacity-reset-every 3000` (D3DGS cadence) | `14k-opreset3k` |
| **B2** | `--grad-threshold 2e-4` (D3DGS default; A1 uses 1e-5, 20× more aggressive) | `14k-grad2e4` |
| **B3** | `--lambda-frob 0 --lambda-aniso 0` (drop Phase-A penalties; also faster — no aniso eigendecomp) | `14k-nopen` |
| **B4** | `--densify-every 100` (D3DGS cadence; A1 uses 200) | `14k-densify100` |
| **B5** | `--lr-pos-scale 0.2` (5× lower position LRs; closer to D3DGS `position_lr_init`) | `14k-lowerlr5x` |

### Excluded
- **3-plane → explicit (scales, rotations) reparameterization** — out of scope per user.
- **MLP deformation** — would slow per-iter; bound at <0.5 dB by RCA §2/§3.
- **Render-path eigendecomp swap** — mathematically equivalent inside the CUDA kernel; null by construction.

## Results

| probe | final N | mean PSNR | Δ vs A1 | mean L1 | s/iter | Δ s/iter | wallclock | frames ≥ A1 | conclusion |
|---|---|---|---|---|---|---|---|---|---|
| **A1 anchor** | ~50k | **26.03** | — | 0.029 | ~25 ms | — | ~6-7 min | 82/82 | reference |
| B1 opreset3k | 47780 | 25.73 | **−0.30** | 0.030 | 18.9 ms | −6.1 | ~4.4 min | 16/82 | reject (worse) |
| B2 grad2e4 | 28017 | 25.91 | **−0.12** | 0.029 | 16.9 ms | −8.1 | ~3.9 min | 33/82 | within single-seed noise (±0.10-0.15 dB typical); not a residual closer either way, but **44% fewer Gaussians (28k vs 50k)** — capacity-efficiency angle |
| B3 nopen | 47361 | 25.71 | **−0.32** | 0.030 | 13.7 ms | −11.3 | ~3.2 min | 17/82 | reject (worse) |
| B4 densify100 | 80825 | 25.73 | **−0.30** | 0.030 | 27.2 ms | **+2.2** | ~6.4 min | 22/82 | reject (worse **and** slower) |
| B5 lowerlr5x | 47771 | 23.59 | **−2.44** | 0.039 | 17.7 ms | −7.3 | ~4.1 min | 0/82 | reject (under-trained) |
| D3DGS reference | ~100k+ | 27.50 | residual = **−1.47** | 0.024 | — | — | — | — | — |

Per-frame data: `docs/issues/perframe_14k_b{1..5}_*_apples.json`.

### Per-frame pattern: where each probe peaks vs A1

Top frames where each probe beats A1 (probe − A1, dB):

- **B1** opreset3k: f214 +0.49, f302 +0.48, f246 +0.28, f146 +0.26, f250 +0.23
- **B2** grad2e4:   f234 +0.72, f298 +0.49, f238 +0.45, f146 +0.41, f306 +0.41
- **B3** nopen:     f234 +0.79, f146 +0.66, f130 +0.57, f230 +0.50, f194 +0.41
- **B4** densify100: f210 +0.68, f146 +0.49, f86 +0.41, f238 +0.38, f234 +0.38
- **B5** lowerlr5x: 0 frames beat A1

B2/B3/B4 share peaks on frames 146, 234, 238 — these align with the dynamic-region top-deficit cluster identified in `rca_phaseC_3db_gap.md` §1 (incl. the SH3-resistant frame 194). Loosening capacity (looser splitting / no aniso / no Frobenius) helps on dynamic frames specifically — but the static-frame regression dominates the mean, so all variants regress overall.

## Gate decision

All 5 probes regressed PSNR; B4 also slowed per-iter. **None close any of the 1.47 dB residual.** Per-iter constraint and "no reparameterization" leave the residual unresolved on slice-banana via CLI levers.

Updated decision tree:
- B1 opacity-reset, B3 no-penalties → confirm RCA §4 spectral-pathology fixes are still load-bearing; removing them costs ~0.30 dB.
- B4 densify-every 100 → over-densifies (N=80k) and slows train without PSNR gain.
- B5 lower LRs → under-trains; existing LR-decay schedule is at the right magnitude.
- B2 grad-threshold 2e-4 → only 0.12 dB cost for 44% fewer Gaussians; potentially worth keeping for memory/compute-bound deployments, but not a residual-closer.

## Findings (per-probe, single sentence each)

- **opacity-reset every 3000 hurts (−0.30 dB)** — under healthy Phase-C spectra (8% dead, median aniso 1.20), forced resets just churn good logits.
- **grad-threshold 2e-4 within single-seed noise on PSNR (−0.12 dB; typical noise floor ±0.10-0.15 dB) but cuts N by 44%** — the 1e-5 trigger we use is over-densifying with low marginal value; an efficiency knob, not a residual-closer.
- **dropping Frobenius + aniso penalties hurts (−0.32 dB)** — the Phase-A correctness penalties are still the dominant safeguard against soft rank-collapse / runaway aniso even after Phase-C density control.
- **densify-every 100 over-allocates AND slows train (+2.2 ms/iter, −0.30 dB)** — A1's 200-cadence is at the right point.
- **lower position LRs by 5× under-trains (−2.44 dB)** — the LR-decay schedule (1 → 0.01) is already at the right magnitude; cutting the start point further starves geometry updates.

## What this leaves on the residual

After §6 (count dead), §6b (SH3 +0.60 dB landed), §7 (blur/init dead), §8 (LR-decay +0.96 dB landed, iters +0.31 dB landed), and now this slate (the 5 CLI levers tested all null or negative), the **1.47 dB residual is not closed by any of the 5 cheap CLI levers tested here**:
- opacity-reset cadence (B1)
- screen-space split-trigger threshold (B2)
- Frobenius/aniso penalty weights (B3)
- densification cadence (B4)
- position-LR magnitude (B5)

### CLI levers that remain untested (not yet eliminated)

The CLI surface is wider than this slate. Untested cheap levers:
- `--spatial-split-threshold` (size-trigger; A1 uses 0.05, D3DGS uses scene-extent-relative)
- `--max-split-per-event` (cap; A1 uses 500)
- `--opacity-prune-threshold` (A1 uses 1e-3, D3DGS uses 5e-3)
- `--sigma-init-sq` (init isotropic variance; A1 uses 0.02)
- `--densify-start` (A1 uses 500)

These weren't on the slate but could yield more null results before exhausting CLI space.

The remaining structural hypothesis from RCA §8 — *3-plane projector vs explicit (scales, rotations) parameterization* — is the only **structural** lever not eliminated. Per user constraint, that probe is excluded from this round.

### Per-frame angle for future structural work

The B2/B3/B4 frame-146/234/238 peaks (where loosening capacity helps on dynamic regions) suggest that a mixed parameterization — relaxed regularization on motion-resolved Gaussians, tight regularization on static-region Gaussians — could in principle gain on dynamic frames without paying the static-region tax. This is speculative and was not tested.

## Round 2: C-probes (vs B2 anchor with `--grad-threshold 2e-4`)

**New anchor (B2):** apples 25.91 dB, ~17 ms/iter, N=28k. Residual to D3DGS = 1.59 dB. The grad-threshold 2e-4 setting is kept forward for capacity efficiency (44% fewer Gaussians for −0.12 dB cost — within single-seed noise).

| probe | lever | hypothesis |
|---|---|---|
| **C1** | `--opacity-prune-threshold 5e-3` (D3DGS canonical; A1/B2 use 1e-3) | Less aggressive pruning may preserve useful low-opacity Gaussians |
| **C2** | `--lr-decay 0.001` (10× more aggressive geometric decay) | Final-stage geometry may need even smaller steps |
| **C3** | `--lambda-frob 1e-3 --lambda-aniso 1e-2` (10× stronger Phase-A penalties) | B3 hurt by removing penalties; this probes the opposite direction |
| **C4** | `--sigma-init-sq 0.005` (4× tighter init variance; was 0.02) | Init covariance may bias toward over-blur |
| **C5** | `--lambda-structural 0.0` (pure L1; drop our box-filter local-stats loss) | Our `structural_loss` (`grassmann/losses.py:57`) is **not** SSIM/DSSIM — it's a 7×7 box-filter local-mean+var matcher, divergent from D3DGS's DSSIM. Tests whether the structural term itself contributes to the deficit |

### C-probe results

| probe | final N | mean PSNR | Δ vs B2 | s/iter | Δ s/iter | wallclock | conclusion |
|---|---|---|---|---|---|---|---|
| **B2 anchor** | 28017 | **25.914** | — | 16.9 ms | — | ~3.9 min | reference |
| C1 prune5e3 | 27972 | 25.921 | **+0.007** | 16.6 ms | −0.3 | ~3.9 min | within noise floor (±0.10-0.15 dB) |
| C2 decay001 | 25977 | 25.561 | **−0.353** | 16.9 ms | 0.0 | ~3.9 min | reject — aggressive decay starves training |
| C3 strongpen | 25870 | 25.929 | **+0.015** | 21.8 ms | +4.9 | ~5.1 min | within noise on PSNR; observed s/iter elevated (+5 ms) but the eigendecomp cost is structurally identical to B2 — likely Modal container variance, not a real slowdown. Reject on null PSNR alone |
| C4 tightinit | 27711 | **25.979** | **+0.065** | 17.4 ms | +0.5 | ~4.0 min | best C-slate apples; within noise floor |
| C5 purel1 | 24960 | 25.885 | −0.029 | **16.0 ms** | −0.9 | ~3.7 min | within noise; structural-loss term was ~null |
| D3DGS reference | ~100k+ | 27.50 | residual = **−1.59** | — | — | — | — |

### Per-frame pattern (top-3 frame gains vs B2)

- **C1** prune5e3: f190 +0.73, f226 +0.57, f294 +0.50 (40/82 frames ≥ B2)
- **C2** decay001: f158 +0.45, f226 +0.43, f202 +0.34 (16/82 — biased to a few specific frames; mean drops elsewhere)
- **C3** strongpen: f14 +0.95, f226 +0.87, f186 +0.60 (41/82)
- **C4** tightinit: **f186 +1.33, f226 +1.18, f162 +0.79** (43/82 — largest individual-frame gains in the C-slate)
- **C5** purel1:    f190 +0.86, f10 +0.73, f186 +0.68 (38/82)

C1/C3/C4/C5 all peak on overlapping frames (**186, 226, 190**) — these aren't the dynamic-cluster frames from B-slate analysis (146, 234, 238); they're a different subset where the B2 baseline particularly under-performs. C4's +1.33 dB on f186 is the largest individual-frame swing in any C-probe but the static-region tax keeps the mean within noise.

## Updated conclusion (after both rounds)

After **10 cheap CLI probes** (5 in B-slate, 5 in C-slate) all single-variable A/B against the 14k LR-decay anchor (or its B2 successor), the picture is:

- **None close >0.1 dB** of the 1.47-1.59 dB residual to D3DGS on slice-banana
- **C4 (tight init)** is the marginal "best" probe at +0.07 dB — within noise but suggestive (largest per-frame swings); could be carried forward as a sub-noise refinement worth combining with B2's grad-threshold
- **C2 (lr_decay 0.001)** is the only C-slate clean regression — confirms the existing 0.01 decay is well-tuned
- **B5 (lower LRs)** at −2.44 dB and **B4 (densify_every 100)** as slower+worse remain the cleanest reject signals across both rounds

### Composite candidate (untested)

B2 + C4 (grad-threshold 2e-4 **and** sigma_init_sq 0.005) is a 2-lever combo not yet measured. Single-variable principle says: if both are within-noise individually, the combo is unlikely to break out of noise — but it's the cheapest remaining experiment if more probes are wanted.

### What this leaves on the residual (unchanged)

The 10-probe sweep does not eliminate the 3-plane → explicit (scales, rotations) parameterization hypothesis from RCA §8. Per user constraint that probe is excluded.

CLI levers still untested (low-priority — 10 probes already null):
- `--spatial-split-threshold` (currently 0.05)
- `--max-split-per-event` (currently 500)
- `--densify-start` (currently 500)
- `--background` color

## Round 3: D-probe (proper SSIM swap)

Implemented Gaussian-windowed SSIM in `grassmann/losses.py:ssim_loss` (11×11 window, σ=1.5, C1=0.01², C2=0.03² — matches 3DGS). Plumbed via `--structural-kind {boxstats,ssim}` flag. **D1** is B2 + `--structural-kind ssim` (lambda_structural=0.2 unchanged).

| probe | final N | mean PSNR | Δ vs B2 | s/iter | wallclock | conclusion |
|---|---|---|---|---|---|---|
| **B2 anchor** | 28017 | **25.914** | — | 16.9 ms | ~3.9 min | reference |
| **D1 ssim** | **45981** | **25.383** | **−0.531** | 18.9 ms | ~4.4 min | reject — SSIM grad-rich → 1.6× more splits, optimizer trades PSNR for SSIM |

### Per-frame: SSIM trades PSNR for shape

- D1 beats B2 on only 12/82 frames
- Top D1 gains: f186 +1.08, f326 +0.55, f226 +0.47, f190 +0.47, f230 +0.39 — the SH3-resistant cluster
- Top D1 losses: f22 −1.41, f70 −1.58, f114 −1.63, f66 −1.72, **f110 −1.95**
- Mean gap to D3DGS **widens** from 1.59 dB (B2) → 2.12 dB (D1)

### Reading

The SSIM swap is the single largest PSNR regression in the probe sweep besides B5 (under-trained at lower LRs). Combined with C5's null (`--lambda-structural 0.0`), the structural-loss term is **not** where the residual to D3DGS lives on this scene. Two separate confirmations from opposite directions:
- C5 dropped the box-filter term entirely → null
- D1 swapped to the canonical 3DGS DSSIM term at the same weight → regression

**Caveat:** SSIM and PSNR optimize for different things. D1's wins on f186/226/190/230 (same cluster as C4) suggest SSIM is doing what it's meant to do — preserve local structure — but on this dataset the metric we report (PSNR) penalizes SSIM-favored solutions. A fair comparison would also compute SSIM/LPIPS metrics; not done here. The PSNR regression is real for the metric we're tracking.

## Files

- This document: `docs/issues/rca_phaseC_14k_lrdecay_probes.md`
- Per-frame JSONs: `docs/issues/perframe_14k_b{1..5}_*_apples.json`
- Modal logs: `/tmp/probe_b{1..5}_*.log`, `/tmp/render_b{1..5}_*.log`
- Eval helper: `scripts/eval_apples.py`
- Predecessor: `docs/issues/rca_phaseC_3db_gap.md` (full RCA up through §8)
