# Surfel-rasterizer A/B + RCA — root cause is eigh chain-rule pathology

**Date:** 2026-05-04
**Branch:** monocular
**TL;DR:** Swapping the EWA rasterizer for `diff_surfel_rasterization` regresses
PSNR by 0.40 dB on slice-banana scale-8 vs the **current empirical baseline
(25.82 dB; reproduction of the A1 recipe under current code, intentionally
~0.21 dB lower than the historical 26.03 in exchange for ~40% faster wallclock
per the user's accepted speed/quality trade)**. Cross-render proves the
regression is in the surfel-trained checkpoint, not in rendering. The cause
is the **eigh chain-rule pathology at near-degenerate disk eigvals**: the
surfel path amplifies upstream gradients ~3000× vs the EWA cov6 path
(median, ~14% of Gaussians in the singular regime even at the EWA-converged
state). Eight probes narrowed the cause and ruled out simple knob fixes;
best surfel result is 25.50 dB (sign-canonicalized eigvecs), still 0.32 dB
short of the current baseline.

## Original A/B (D1, D2)

A1 anchor recipe: slice-banana, scale 4 train / scale 8 apples eval,
deformable_interp, seed 42, SH3, 14k iters, LR-decay 0.01, grad-thr 1e-5,
sigma_init_sq 0.02, densify-every 200, λ_frob=1e-4, λ_aniso=1e-3.

**Anchor reproduction note (2026-05-04):** Re-running the A1 recipe under
current code gives **25.815 dB** (vs the historical 26.03 logged when commit
`b958b68` was current). Wallclock dropped from ~360-420 s to 249 s
(~41% faster). The 0.21 dB drift is an intentional accepted trade for the
speed gain. **All Δ values below use the current 25.82 baseline.** For the
historical-baseline comparison, subtract another 0.21 dB.

| arm | rasterizer | 2DGS losses | apples PSNR | Δ vs current A1 (25.82) | s/iter | wallclock |
|---|---|---|---|---|---|---|
| **A1** anchor (current) | gaussian (Inria EWA, σ_lift²=1e-4) | OFF | **25.82** | — | ~18 ms | 249 s |
| A1 historical (b958b68) | gaussian | OFF | 26.03 | +0.21 | ~25 ms | ~360-420 s |
| **D1** surfel-OFF | surfel (Huang2024 ray-plane, no lift) | OFF | **25.42** | **−0.40** | 27.9 ms | 391 s |
| **D2** surfel-ON | surfel | ON (λ_n=0.05@7k, λ_d=100@3k) | **22.57** | **−3.25** | 26.5 ms | 371 s |

D2 cliffs at iter 7000 when normal-consistency loss activates → 2DGS losses
are catastrophic for our setup (sign-bug not ruled out, but uniformly bad
across all 82 frames). Parked.

## RCA (D1)

### Step 1 — Cross-render (rules out renderer side)

| | rendered EWA | rendered surfel |
|---|---|---|
| **A1 ckpt** (EWA-trained) | **26.032** | **26.024** (Δ −0.006) |
| **D1 ckpt** (surfel-trained) | **25.260** (Δ −0.77) | **25.421** (Δ −0.61) |

- Renderers nearly photometrically equivalent given same geometry (Δ 0.006 dB on A1 ckpt).
- D1 ckpt is worse with EITHER renderer. Per-frame correlation of (D1-surf vs A1)
  with (D1-EWA vs A1) deltas = **r = 0.940** → bad geometry travels with the
  ckpt regardless of renderer.
- D1 actually renders BETTER with surfel (its training rasterizer) than with
  EWA — Δ +0.16 dB. Surfel-trained disks are tuned to surfel's projection.
- **Conclusion:** The 0.61 dB regression is in the **D1 checkpoint**, i.e.
  in surfel **training dynamics**, not in surfel rendering.

### Step 2 — Local Jacobian-magnitude diagnostic

Compared chain-rule sensitivity of the two paths on synthetic rank-2 PSD inputs
(`A @ A.T`, N=1000, 20 random batches):

- **EWA path** (`Σ → cov6`): Jacobian magnitude ~190 (per 1e-5 perturbation)
- **Surfel path** (`eigh → scales[N,2] + R[N,3,3]`): Jacobian magnitude ~500,000
- **Median ratio: 3041× , max 4445×**

The eigh chain rule has 1/Δλ terms that explode at near-degenerate eigvals.

### Step 3 — Empirical degeneracy on real checkpoints

Loaded A1 and D1 checkpoints, computed eigvalsh on `Σ_3D(t_0)` for all Gaussians at t=0.5:

- **A1 ckpt:** median anisotropy `λ_max/λ_mid = 2.22`; **13.6%** of Gaussians
  have `(λ_max-λ_mid)/λ_max < 0.05` (in-plane eigvals nearly equal → 1/Δλ blowup)
- **D1 ckpt:** `eigvalsh` failed to converge on at least one matrix (error code 2,
  ill-conditioned). Surfel training drove some Σ into worse degeneracy than EWA.

So during surfel training, ~1 in 7 Gaussians per step has a chain-rule
singularity. These bursts inject pathological updates into `L_raw`.

### Step 4 — Probe slate (ruled out simple knob fixes)

| probe | knob | apples | Δ A1-current (25.82) | Δ D1 | conclusion |
|---|---|---|---|---|---|
| **D1** | (vanilla surfel) | 25.42 | −0.40 | — | base |
| **D-noPen** | drop λ_aniso, λ_frob | 25.25 | −0.57 | −0.17 | penalties HELP slightly; not the cause |
| **D-with-lift** | + σ_lift²=1e-4 pre-eigh | 25.43 | −0.39 | +0.01 | σ_lift² is NOT a hidden training regularizer |
| **D-signcanon** | + per-column eigvec sign-canon (max-abs-component positive) | **25.50** | **−0.32** | +0.08 | best knob fix; sign-instability accounts for ~0.08 dB |
| **D-signqcanon** | + qw≥0 quat canon on top of D-signcanon | (val=E) | — | 0 | quat canon is redundant once eigvec canon is in place |
| **D-lr0.1** | lr_pos_scale=0.1 (10× smaller geom lr) | (val 20.17, catastrophic) | — | — | **rules out** uniform lr scaling — under-trains within 14k |
| **D-jitter1e-4** | + ε(J+J^T) anisotropic jitter on Σ pre-eigh | 25.32 | −0.50 | −0.10 | jitter eliminates `loss=nan` but adds stochastic noise > eigh-deg benefit |

### Diagnostic byproduct — `loss=nan` source identified

The `loss=nan` printed every iter from D1, D-with-lift, D-signcanon, D-signqcanon
disappeared in D-noPen and D-jitter1e-4. Both fixes touch the same component:
**`lambda_aniso`'s `eigvalsh(Σ_3D_t)`** is the NaN source — it operates
independently of the surfel rasterizer adapter and produces NaN when input
matrices are degenerate. (PSNR/L1/params remain finite; the NaN never reaches
gradient updates because the Phase-A penalty's gradient is zeroed by clamps.)
Cosmetic, but worth fixing if the code is reused.

## Per-frame pattern

D1-surf vs A1: 10/82 frames beat A1; mean Δ −0.61 dB
- Top +: f178 +0.26, f274 +0.20, f302 +0.20, f158 +0.14, f138 +0.06
- Top −: f14 −2.25, f262 −2.23, f22 −2.20, f254 −2.09, f18 −1.89
- Worst regressions on early/late (extreme-time) frames; few wins on
  mid-trajectory dynamic-cluster frames.

D2-on vs A1: 0/82 frames beat A1; mean Δ −3.46 dB; universal regression.

## What a real fix would require

The 0.5 dB gap is **structural** to deriving (scales, rotations) via eigh from
our (n, μ, L) parameterization. Knob-level fixes can claw back ~0.1 dB at most.
To close the rest:

1. **Custom backward pass** for `sigma3d_to_disk` that detects degeneracy and
   returns a clipped/projected gradient instead of letting 1/Δλ blow up.
   Engineering work (~few hours), unclear how much it closes.
2. **Re-parameterize** to learn (scales, rotations) directly like 2DGS does —
   abandons the (n, μ, L) 3-plane structure that's load-bearing for our 4D
   conditioning. Significant.
3. **Decompose M' (3×3 rank-2 square root of Σ_3D) via SVD** instead of eigh
   on Σ — same eigvecs in theory but linearly conditioned at rank deficiency.
   Need to verify M' is exposed through `compute_derived` first.

## Caveats

- Single seed per arm; ±0.10-0.15 dB single-seed noise band per the existing slate.
- All probes at slice-banana scale-8 only. Other scenes / scales not tested.
- The `loss=nan` diagnosis above was inferred from the NaN's appearance/disappearance
  pattern across probes, not by direct tracing of which loss component first NaNs.

## Decision

- **Park surfel rasterizer; do not adopt as default.** Implementation stays
  behind `--rasterizer surfel`; sign-canon, σ_lift² lift, jitter, and 2DGS
  losses are all gated CLI flags so future revisits are cheap.
- The mechanistic claim from `phaseC_residual_14k` ("rank-2 + ε I floor explains
  the 1.37 dB residual") is **refuted** — the residual is NOT in the rank-2
  vs rank-3 lift; it would still be ≥1.0 dB even with a perfect rank-2 renderer.
- Next probe candidates (per `eins nach dem anderen` plan): Schur clamp /
  4D conditioning levers (v7 §5-6), init density at scale 8.
- Memory updated: `project_grassmann_surfel_rasterizer_ab.md`.

## Files

- This document: `results/rca/surfel_rasterizer_ab.md`
- Per-frame JSONs: `results/rca/perframe/perframe_14k_d{1,2,_surfel_*,A1ckpt_via_surfel,D1ckpt_via_ewa}_apples.json`
- Modal logs: `/tmp/probe_d_*.log`, `/tmp/render_*.log`
- Implementation: `grassmann/surfel_rasterizer.py`,
  `grassmann/losses.py:depth_distortion_loss/normal_consistency_loss/depth_to_world_normal`
- CLI: `--rasterizer surfel`, `--use_2dgs_losses`, `--lambda_normal`, `--lambda_dist`,
  `--surfel_eigval_floor`, `--surfel_sigma_3d_blur`, `--surfel_eigh_jitter`
  in `train_mono.py` and `render_mono.py`
- Modal image: `diff_surfel_rasterization` from `git+https://github.com/hbb1/diff-surfel-rasterization.git`
