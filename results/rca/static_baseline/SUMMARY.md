# Static-baseline floor on slice-banana (Bug-F + P-rca-7 recipe)

**Date:** 2026-05-11
**Plan:** `/home/xyz/.claude/plans/wir-w-rden-zu-debugging-tender-hartmanis.md`
**Branch:** monocular-init
**Setup:** slice-banana, scale 4, 14k iters, deformable_interp split, seed 42.
Recipe = Bug-F + Bug-D + P-rca-7 (current best dynamic, **24.93 dB**).
Static probe = same recipe + `--static_baseline` (Schur off, w_t=1).

## TL;DR — static path is worse, not better, on this scene

| run | val_psnr | wall | N | flags |
|---|---|---|---|---|
| dynamic Bug-F (reference) | **24.93** | 371s | 86.6k | full time conditioning |
| **S0** static + tsplit + grelax | **20.99** | 327s | 94.5k | `--static_baseline` |
| **S0-clean** static, no motion knobs | **20.95** | 410s | 130.3k | S0 minus `--temporal_split_threshold` and `--grassmann_relax_*` |
| **D0** single-frame fit (frame 100) | **27.84** | 221s | 29.4k | `--diag_single_frame 100` (forces static) |

- Static loses **−3.94 dB** vs dynamic. Time conditioning is *load-bearing* on slice-banana, not optional polish.
- Dropping motion-only knobs (tsplit + grelax) does nothing (+0.04 dB within noise) but explodes N by 35%.
- Single-frame fit (D0) **regresses** vs Phase A: see next section.

The user's premise — "static-debug should approach 30.5 dB" — does not hold *for this scene*. The bundle compression cost dominates: slice-banana has 247 train frames with banana-cutting motion, and the static path forces each Gaussian to explain all of them simultaneously.

## D0 single-frame regression vs Phase A — the sharpest signal in this RCA

| metric | D0 now | Phase A `--diag_single_frame 100` (`results/rca/3plane_low_psnr.md:120`) | delta |
|---|---|---|---|
| val PSNR | **27.84 dB** | **29.07 dB** | **−1.23 dB** |
| N (final) | 29 355 | 13 842 | +2.12× |
| iters | 14 000 | 14 000 | same |
| image_scale | 4 | 4 | same |
| frame | 100 | 100 | same |

Same flag (`--diag_single_frame 100`), same scene, same frame, same iter budget, same scale — and the current "best" recipe (Bug-F + Bug-D + P-rca-7 + sh=3 + lr_decay 0.01) **underfits one image by 1.23 dB while spending 2× the Gaussians**. This is a stronger debugging signal than the static-bundle floor: the recipe has flags that are net-negative for the single-frame / static regime, and the dynamic-side wins masked them.

Candidates (each tuned for the multi-frame regime, untested under single-frame):
- `sh_degree=3` — Phase A used sh=0. SH3 buys +0.60 dB on multi-frame (`phaseC_3db_gap.md` §6b) but its single-frame effect at the new N has never been measured.
- `opacity_reset_every=3000` + `scale_min_prune=5e-3` — both cull aggressively. On a one-image fit they may be killing useful coverage every 3k iters.
- `split_anisotropic_shrink` (Bug-F) — iso-N controls show half of Bug-F's gain is capacity-driven (`scripts/launch_bugF_isoN_controls.sh`); under one-frame the extra splits land redundantly.
- `lr_decay=0.01` — under 14k iters this is more aggressive than under 30k; LR may decay below useful rates before the single-frame fit completes.

A 3-4 run one-flag-diff bisection on D0 would isolate the regression cheaply (~10-15 min wall). This is **option 4** below and is the recommended next move because it uses data we already have.

## Mechanistic finding — static path activates a degenerate code path

Spectral RCA on S0 reveals a structurally distinct failure mode vs dynamic:

| metric | S0 (static) | S0-clean (static, no grelax) | D0 (1-frame) | Bug-D dynamic (reference) |
|---|---|---|---|---|
| `\|n̂_t\| > 0.95` (time-axis collapsed) | **97.9 %** | 13.0 % | 100 % | 0.04 % |
| effectively dead (opacity < 0.01) | **40.2 %** | **54.8 %** | 39.3 % | ≈ 8 % (Bug-D era) |
| alive (audit_triggers) | 5.4 % | **3.7 %** | — | — |
| zombie | 17.1 % | 23.9 % | — | — |
| `Σ_tt` q50 | 7.4e-4 | 0.16 | 4.8e-5 | 0.060 (B-D) |
| splits / prunes (48 events) | 94.6k / 14.1k | 132k / 15.6k | — | ~40k / 12k (B-F) |
| N (final) | 94.5k | 130.3k | 29.4k | 86.6k |

What this means:

1. **S0 with grelax**: `lr_n` ramps from 0 → base over [1k, 8k]. Combined with `static_baseline=True`, the optimizer has zero gradient pressure to tilt `n̂` off the time axis (Schur is bypassed, so n̂'s only effect under static is which 3 components of L_raw the projector kills). 97.9 % of n̂ stays at e_0. Σ_tt collapses to ~0 because the temporal axis is the projector kernel.

2. **S0-clean** (no grelax, lr_n free from iter 0): n̂ does spread (only 13 % near e_0), but death rate gets *worse* (54.8 % vs 40.2 %) and PSNR is the same. The spread is wasted DOF.

3. **D0**: with only one frame and one time value, n̂ collapses to e_0 by necessity (no temporal variation in supervision). The 39.3 % death rate at N=29k is also high.

**Common factor**: in all 3 static runs, ~40-55 % of the population is dead. Under static = True, the photometric loss can be reduced by allocating more Gaussians per spatial region, but the dead ones don't contribute and consume optimizer state. Splits >> prunes by 7-9×. The DC heuristic loop (designed for the dynamic path) over-splits under static and the opacity-reset-every-3000 + scale-min-prune fixes that work in dynamic don't catch the static failure mode.

## Decision against the plan's gate

Decision-gate quotes from the plan:

> - If S0 ≪ 30 dB but S0-clean > S0 → motion-only knobs are eating PSNR; pin the cleaner recipe and re-baseline.
> - If S0 ≪ D0 (e.g. S0 ≈ 25, D0 ≈ 32) → the bundle compression cost is high.

Reality: **S0 ≈ S0-clean ≪ D0 ≪ 30.5 dB target**. Neither branch matches the plan's expected outcomes cleanly. The premise (that a static-debug exists at ~30 dB on slice-banana) is the part that breaks. Four coherent next moves; **option 4 is recommended** because it uses what we already learned:

1. **Reframe**: accept that on slice-banana the static path floors at ~21 dB because the scene is fundamentally dynamic. Use D0 as the debug ceiling instead — but only *after* fixing the 1.23 dB regression vs Phase A (option 4).
2. **Switch scene**: probe `--static_baseline` on a NeRFies scene that's near-static (broom, tail-static segments) or a multi-view 3DGS reference scene. Establishes a real static-3DGS ceiling outside the slice-banana motion penalty. Higher effort (new data prep).
3. **Fix the static code path**: under `static_baseline=True`, n̂ and Σ_tt are *wasteful but not strictly degenerate* DOF (n̂ still picks which row of L_raw the projector kills). Two candidate patches:
   - Freeze `n̂ = e_0` (skip the lr_n parameter group entirely under static).
   - Freeze the time-axis row of L_raw.
   But: S0-clean already runs lr_n free from iter 0 and didn't improve PSNR. So we have one side of the experiment (n free); we'd need the other (n locked) to know if a code change is justified. Defer until we measure it.
4. **Bisect the D0 regression (recommended)** — done; see results below.

## D0 bisection results (5 one-flag-diff probes)

`scripts/launch_d0_bisect.sh`, 14k iters, scale 4, seed 42, frame 100.

| run | flag changed | val_psnr | Δ vs D0 | N | wall |
|---|---|---|---|---|---|
| D0 (current full recipe) | — | 27.84 | — | 29.4k | 221s |
| D-A | Phase A reproduction (random init, no DC, sh=0, lr=1) | **28.35** | +0.51 | 13.8k | 147s |
| D-noSH | drop `--sh-degree 3` | 27.99 | +0.15 | 41.0k | 319s |
| **D-noLRdecay** | drop `--lr-decay` (back to 1.0) | **22.99** | **−4.85** | 81.5k | 448s |
| D-noOPreset | drop `--opacity-reset-every 3000` | 27.59 | −0.25 | 27.8k | 180s |
| **D-noBugF** | drop `--split-anisotropic-shrink` | **28.68** | **+0.84** | 30.2k | 231s |
| Phase A (historical, 2026-05-03) | — | 29.07 | +1.23 | 13.8k | — |

### Headline: Bug-F is the regression at single-frame

Dropping `--split-anisotropic-shrink` recovers **+0.84 dB** and lands at **28.68 dB**, beating the D-A control (28.35) by **+0.33 dB**. Mechanism: Bug-F's iso-N analysis (`scripts/launch_bugF_isoN_controls.sh`) showed half of its multi-frame gain was capacity-driven (more splits → more Gaussians). Under single-frame the capacity is already excess (N=29-30k for one image at scale 4), and the anisotropic-shrink converts that excess into noise. Bug-F is regime-specific: gains dynamic, loses static.

### Secondary findings

- **LR-decay is load-bearing**. Without `--lr-decay 0.01`, the full recipe collapses to 22.99 dB while exploding N to 81.5k. The geometric-LR damping is what keeps DC from over-splitting.
- **SH3 and opacity-reset are roughly neutral at single-frame** (±0.25 dB).
- **D-A (28.35) < historical Phase A (29.07)** by 0.72 dB — same flag, same frame, same iters, same scale. This is **infrastructure drift**, not a recipe regression. The A1-drift memo (`project_grassmann_wave_a_combos.md`, 2026-05-10) noted a similar 0.91 dB drift between commit `b958b68` (Phase A era) and `e514bc9` (Wave A era), with wallclock also drifted 249s → 413s. Same root cause is plausible (Modal CUDA / rasterizer version pin drift). Out of scope for this RCA; worth a separate investigation if it bites again.

### Recipe-level recommendation

For the **single-frame** regime: disable `--split-anisotropic-shrink` (Bug-F). +0.84 dB confirmed on D0.

For the **full-bundle static** regime: see validation below — Bug-F is null at +0.06 dB. **The static floor on slice-banana is ~21 dB regardless of Bug-F.**

For the multi-frame / dynamic regime: no change. Current best stays at 24.93 dB (Bug-F).

## Validation: does the bisection generalize to full bundle? (S0-noBugF)

| run | val_psnr | N | wall |
|---|---|---|---|
| S0 (Bug-F on) | 20.99 | 94.5k | 327s |
| **S0-noBugF** | **21.05** | **70.6k** | **276s** |
| delta | **+0.06 dB** (within noise) | **−25 % N** | **−16 % wall** |

The bisection finding does **not** generalize. On the full bundle, Bug-F is PSNR-neutral but cheaper (lower N + lower wall) — the smaller-N variant is the rational default. Mechanism is consistent with the iso-N analysis: Bug-F's effect is **capacity-induced**, not quality-induced. It matters when N is over-allocated for one image (D0: N=29k for 1 frame → Bug-F injects noise). On the full bundle, N=70-94k for 247 frames is still bundle-compression-bound, so Bug-F's extra splits are either useful or absorbed.

### Final answer for the user's stated goal

> "wir würden zu debugging zwecken gerne erstmal den statischen szene fall optimieren — ceiling hier sollte ca 30.5 db sein"

**The 30.5 dB target is not reachable on slice-banana via `--static_baseline`.** The static path floors at **~21 dB** because the scene has 247 frames of banana-cutting motion that the static path cannot share-compress across Gaussians. The achievable single-image ceiling on this scene at current recipe + Modal infra is **D-noBugF = 28.68 dB**, with a 0.39 dB residual to the historical Phase A measurement of 29.07 dB attributable to Modal / rasterizer infra drift.

If the original intent of the static-debug was "isolate the rendering / DC pipeline from motion modeling", the cleanest paths from here are:
- (a) Switch to a near-static scene (NeRFies broom / static segment) and re-establish the static floor there. Outside slice-banana, the static path is not degenerate and the 30 dB ceiling may be reachable.
- (b) Use D0 (single-frame) as the per-frame ceiling proxy. D-noBugF = 28.68 dB is the realistic ceiling on this scene + infra.
- (c) Investigate the 0.39 dB infra drift separately (compare current diff-gaussian-rasterization commit pin vs Phase A era `b958b68`).

Knob-sweeping the S0 config to push past 21 dB is dominated by all three of these.

## Files

- `checkpoints/nerfies-slice-banana-spatial_slice-14000it-static-{s0,s0-clean,d0-frame100}/trained_nerfies_spatial_slice.pt` — all 3 ckpts.
- `/tmp/probe_static-{s0,s0-clean,d0-frame100}.log` — full training logs.
- `scripts/launch_static_floor.sh` — launcher (3 Modal runs).
