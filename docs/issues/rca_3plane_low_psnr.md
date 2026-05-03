# RCA — 3-plane projector low PSNR ceiling (~22 dB)

**Date:** 2026-05-03
**Checkpoint:** `nerfies-slice-banana-random-50000it-3plane-50k-valstride4`
**Setup:** slice-banana, scale 4, 50k iters, random init, no DC, val_stride=4 split
**Final:** train PSNR 21.87 dB, val PSNR 21.82 dB, val L1 0.0516, N=13842

## Problem

Asymptotes at ~22 dB on slice-banana scale 4. Logarithmic convergence:
5k → 14k → 30k → 50k = 19.6 → 20.77 → 21.42 → 21.82 dB. Halving the
slope every ~15k iters — clean saturation, not a local-minimum stall.

For comparison, Deformable3DGS on the same data + split + scale reaches
**25.41 dB at iter 7000** (intermediate) and would land at ~26 dB by 14k.
A 5+ dB structural deficit.

## Spectral RCA on the trained 3-plane Gaussians

`scripts/rca_spectral.py` reconstructs the per-Gaussian Σ_3D(t_0)
eigenvalues, anisotropy, opacity, and orientation distributions. Key
distributions (percentiles 1 / 25 / 50 / 75 / 99):

| metric | p1 | p25 | p50 | p75 | p99 |
|---|---|---|---|---|---|
| Σ_3D(t_0) λ_min (minor axis) | 1.7e-9 | 4.3e-4 | 6.4e-3 | 0.047 | 1.23 |
| Σ_3D(t_0) λ_max (major axis) | 2.1e-3 | 0.044 | 0.20 | 0.73 | 13.45 |
| anisotropy λ_max/λ_min | 1.4 | 6.3 | **22.9** | 208.8 | 6.8e+7 |
| disk area π·√(λ_max·λ_min) | 2.6e-5 | 0.013 | 0.086 | 0.49 | 9.91 |
| Σ_tt_pure (temporal extent) | 9.1e-5 | 0.010 | 0.052 | 0.34 | 7.49 |
| \|c_world\| (space-time coupling) | 1.5e-3 | 0.027 | 0.12 | 0.42 | 6.56 |
| opacity | 2.6e-3 | 5.7e-3 | 0.038 | 0.45 | 0.9999 |
| \|n̂_t\| (time-axis alignment) | 5.4e-3 | 0.18 | 0.44 | 0.73 | 0.99 |

### Pathology counts (out of N = 13842)

- **Effectively dead (opacity < 0.01):** 4458 / 13842 = **32.2 %**
- **Extreme anisotropy (λ_max/λ_min > 100):** 4208 / 13842 = **30.4 %**
- **Runaway huge (λ_max > 1.0 in scene units²):** 2740 / 13842 = 19.8 %
- **Collapsed (λ_min < 1e-6 — effectively rank-1 again):** 976 = 7.1 %
- **Near time-axis n̂ (\|n_t\| > 0.95, degenerate temporally):** 867 = 6.3 %
- Σ_4D · n̂ residual: max 6.4e-14 (math invariant intact).

### Reading

Without density control, the optimizer redistributes the fixed
capacity by killing low-utility Gaussians (opacity → 0) and growing
the rest into elongated strips that cover regions cheaply. Net effect:

- Effective N ≈ 13842 − 4458 = **9 400 alive** Gaussians.
- For a 384×680 frame (≈ 261 k pixels), that's ~28 px/Gaussian.
- With median anisotropy 22.9, each "disk" is more like a 22:1 strip
  than a round patch.
- Detail-preserving capacity is much lower than N alone implies.

The 22 dB ceiling is consistent with this effective capacity. The
fixed-N + no-DC choice was an explicit Phase A simplification (DC
under the legacy 2-plane param was net-negative; Phase A delayed the
DC redesign to Phase C). What we've now learned is that Phase C is
**load-bearing** — without it, the model cannot reach Deformable3DGS
quality on this scene.

## Diagnostic: single-frame static fit (Diag 2)

`--diag_single_frame 100`: train and validate on slice-banana frame 100
only, with `static_baseline=True` (no time conditioning), N=13842,
scale 4, 14k iters.

**Result: val PSNR 29.07 dB, val L1 0.0168.**

Reading: N=13842 is **plenty** of capacity for a single image at scale 4.
The 22 dB ceiling on the full 330-frame run is therefore not "Gaussians
can't render the scene" — it's "13842 Gaussians cannot simultaneously
represent 330 frames with diverse dynamic content under our current
parameterization."

The 7 dB gap (29 dB single → 22 dB multi) is the **cost of multi-frame
sharing under the 3-plane projector with linear drift**. D3DGS pays
~3 dB of that cost (lands at ~26 dB) — its MLP-deformation buys back
~4 dB of the gap. Roughly:

- single-frame ceiling (this scene, scale 4, N=13842): 29 dB
- D3DGS multi-frame: 26 dB → motion model costs ~3 dB
- our multi-frame: 22 dB → linear-drift + fixed-N costs ~7 dB

The 5 dB gap to D3DGS is partly motion (~4 dB) + partly capacity
allocation (~1-3 dB; Diag 1 result will refine this).

## Two competing hypotheses for the ceiling, and their tests

The user's reviewer split the 22 dB ceiling into two clean hypotheses:

1. **Capacity-bound:** N=13842 fixed (with 32% dead and 30% high-aniso) is
   too few for slice-banana scale 4 surface detail. Test:
   `--init_points_multiplier 4` (N≈55k) at 14k iters; if ceiling moves
   meaningfully toward 25 dB the limiter is capacity, not motion.

2. **Motion-model-bound:** linear drift via the rank-2 disk doesn't
   capture slice-banana's banana-cutting motion; residuals appear as
   dynamic-region blur. Test: single-frame static fit (`--diag_single_frame
   100`) at scale 4 with N=13842; if it reaches ~25 dB on one frame, the
   gap to the temporal model on the same frame = motion residual.

### Diag 1 result: N×4 capacity test

`--init_points_multiplier 4 --split_convention deformable_interp`,
14k iters, scale 4, same seed. Final val PSNR **20.95 dB** vs **20.24 dB**
at N=1× on the same split. **Δ = +0.7 dB for 4× capacity.**

Caveat: at 14k iters the N×4 run is less converged than the N×1 run
(per-iter cost roughly linear in N). At full convergence the gap may
widen to 1-2 dB but not more.

### Synthesis: motion-bound, not capacity-bound

Combining Diag 1 + Diag 2:

| run | N | iters | val PSNR |
|---|---|---|---|
| 1-frame static fit | 13842 | 14k | **29.07** (per-frame ceiling) |
| 330-frame, N×1 | 13842 | 14k | 20.24 |
| 330-frame, N×4 | 55368 | 14k | 20.95 (+0.7 dB) |
| 330-frame, N×1, 50k iters | 13842 | 50k | 21.82 (asymptote) |
| D3DGS 330-frame | ~50k | 7k (intermediate) | 25.41 |

The single-frame ceiling of 29 dB tells us N=13842 is more than enough
*pixel capacity* for this scene at scale 4. The 8.8 dB drop from 29 →
20.24 when going from 1 frame to 330 frames at the same N is the cost
of compressing 330 dynamic frames into a single set of Gaussians under
linear drift. Quadrupling N recovers only 0.7 dB of that 8.8 dB cost.

**The dominant limiter is the motion model**, not capacity. The 3-plane
projector with linear drift via Schur on time cannot represent slice-
banana's banana-cutting motion accurately enough; residuals appear
across all dynamic regions and dominate the PSNR.

Implication for the next-step decision:

Both diagnostics are wired and currently running on Modal. Their
outcome decides whether the next investment is:

- (a) DC redesign for the 3-plane param (Phase C of the plan), if
  capacity-bound — ranked first by the reviewer.
- (b) Motion-model upgrade (acceleration term, MLP deformation), if
  motion-bound.
- (c) Some mix.

## Pathology mechanism (mechanistic, not phenomenological)

The 32% dead + 30% high-aniso pattern is a known dynamic-Gaussian-
splatting failure mode under fixed N: Gaussians that don't lower the
loss in *any* view shrink their opacity to escape the gradient
(opacity sigmoid pushed to logit ≈ −5), while Gaussians on flat
regions grow large and elongated to cheaply explain monotone areas.
Standard 3DGS counters this with periodic opacity reset + DC; our
Phase A explicitly disables both.

The 7% collapsed disks (λ_min < 1e-6) are the disturbing finding —
under the new param the rank-1 pathology was supposed to be impossible.
What's happening: the optimizer drives one column of `L_raw` close to
n̂, which the projector annihilates, leaving an effectively-rank-1
disk. The projector kills the *direction*, but the *magnitude* of
each column is unconstrained, so the optimizer can route capacity into
n̂ as a soft "delete this dimension" knob. A scale prior on `L_raw`
columns or a frobenius-norm penalty would prevent this; deferred to
Phase C.

## Files

- `scripts/rca_spectral.py` — produces the table above from any `.pt`
  checkpoint of the 3-plane param.
- `checkpoints/3plane_50k.pt` — the analyzed checkpoint (local copy
  pulled from Modal).
