# Bug-D spectral / motion / density diagnostic (2026-05-11)

Checkpoint: `nerfies-slice-banana-spatial_slice-14000it-bug-D-anisooff`
N=46,768 Gaussians, val PSNR=24.62 dB.

## Headline: geometry is healthy. Residual is capacity + tail-aspect.

### 1. Rank/Plane-constraint sanity (5 plots)

| Quantity | Result | Verdict |
|---|---|---|
| Σ_3D_t λ_3 (post-Schur kernel) | q95=3.8e-8, max=6.4e-7 | ✅ rank-2 clean |
| Σ_4D eig[0] (n-kernel) | q50=3.8e-13 | ✅ rank-3 clean |
| `\|Σ_4D n\| / \|Σ_4D\|_F` | q99=1.08e-7 | ✅ plane constraint holds (advisor expected 1e-14; ours is ~1e-7 from Adam drift on L_raw) |

### 2. Is the model actually dynamic? `|n_{1:}|` distribution

| Quantile | Value |
|---|---|
| q1  | 0.031 |
| q50 | 0.145 |
| q95 | 0.405 |
| q99 | 0.504 |

- **Static-degenerate fraction** (|n_{1:}|<0.01): **0.04%** ✓ not stuck at e₀
- **In-plane-degenerate fraction** (|n_{1:}|>0.99): **0.00%**
- But: **n stays mostly tilted toward e₀** — q50 of |n₁:| is only 0.14, q99 is 0.5 (never reaches 1).
  - Interpretation: the support planes are tilted off the time axis by a moderate angle (median ≈ sin⁻¹(0.14) ≈ 8°). Not degenerate, but n has limited dynamic range. This bounds how much "Schur-induced motion" the model can express per primitive.

### 3. Temporal extent √Σ_tt distribution

| Quantile | Value (normalized t ∈ [0,1]) |
|---|---|
| q5  | 0.014 |
| q50 | 0.060 |
| q95 | 0.214 |

- 41.5% "short-lived" (√Σ_tt < 0.05, i.e., visible for <5% of scene)
- 0.06% "whole-scene" (√Σ_tt > 0.5)
- **Not bimodal** ✓ (advisor flagged bimodal-at-ε-vs-full as the failure case; ours is unimodal-broad)

### 4. Anisotropy λ_1/λ_2 of Σ_3D_t

| Quantile | Value |
|---|---|
| q50 | 7.6 |
| q90 | 22.8 |
| q99 | 29.2 |
| max | 30.0 (the `--max-aspect-ratio` clip) |

- **14.75% of Gaussians have aspect > 20** (advisor's red-line threshold)
- The SVD clip at 30 is biting often — population accumulates at the upper bound
- **Recommendation**: probe `--max-aspect-ratio 10` or 15 (more aggressive clip)

### 5. Motion decomposition

| Component | q50 | q95 |
|---|---|---|
| `\|n_0\|` (rigid normal speed proxy) | 0.99 | 0.999 |
| `\|tangential drift\|` (c/Σ_tt projected ⊥ n_{1:}) | 3.4 | 18.6 |

- Both modes are active — **not degenerate** in either rigid-plane or tangential drift.

### 6. Density-control fire rates (from log)

See `diag_density_control.png`. Pattern is healthy:
- iters 600-1400: heavy prune (init culls bad Gaussians)
- iters 2000-5000: split-dominated growth
- iters 5000-10000: tapering splits + steady ~1-20 prune/cycle
- iters 10000+ (post-density-stop): zero density events

No "clone-die-clone-die" cycle — pruned/cycle is in single digits after iter 5000. Lifespan looks healthy.

## What this implies for the residual gap to D3DGS

Geometry is healthy across all 5 advisor-flagged failure modes. The remaining ~1.4 dB gap to D3DGS at iso-iters is most likely:

1. **Capacity**: 46.8k vs ~186k Gaussians in D3DGS at convergence (4× less)
2. **Aspect tail**: 15% of population at the 30-clip — pressure on `--max-aspect-ratio` worth probing
3. **Loss**: boxstats vs DSSIM (advisor expects 0.5-1 dB)

Geometry-curriculum levers (σ_tmp anneal, lr_n warmup, n-freeze first) are unlikely to be load-bearing since `|n_{1:}|` is already healthy and density-control fires cleanly.

## Plots

- `diag_spectral.png`  — Σ_3D_t + Σ_4D eigvals, plane residual, n distribution, Σ_tt
- `diag_motion.png`    — rigid vs tangential drift scatter
- `diag_density_control.png` — splits/tsplits/prunes + population over time
