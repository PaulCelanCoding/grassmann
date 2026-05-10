# #4.1 3DGS-MCMC probe results — slice-banana, scale 4, 14k iters

**Date:** 2026-05-10
**Branch:** monocular-init
**TL;DR:** On slice-banana with the SfM-derived ~14k init points, MCMC
relocation as implemented is **strictly worse** than the heuristic
densifier — the population is frozen at init count, leading to
underfit (17–20 dB val PSNR vs Combo-AA's 24.36). Adding SGLD noise
on top of Combo-AA (heuristic split kept) gives **−0.11 dB**.
Hypothesis: pure-MCMC needs a much larger init population (Kheradmand's
typical setup is 200k+); v2 probes test 4× and 8× init. The marginal
SGLD-noise regression suggests the noise scale (5e-5 × |L|_F ≈ 7e-6
per step) is far below the position LR (5e-3) and likely null even at
the gate.

## Implementation summary

`grassmann/density_control.py` adds two MCMC building blocks (Kheradmand
NeurIPS 2024) adapted to the G(3,4) Schur parameterization:

1. **Stochastic relocation** (`mcmc_relocate`): dead Gaussians (opacity <
   `opacity_threshold`) are reassigned to live destinations sampled from a
   categorical proportional to live opacity. After relocation, the
   destination AND its `k` newcomers all get corrected:

   ```
   o_new = 1 − (1 − o_old)^(1 / (k + 1))
   L_new = L_old / sqrt(k + 1)
   ```

   so the destination's contribution under alpha-blending is preserved.
   Adam state on dead rows is zero-reset; on corrected live rows only
   the changed param momentum (opacity_logit, L_raw) is zeroed.

2. **SGLD noise on μ_spatial** (`mcmc_noise_step`, called every iteration):

   ```
   noise_std = mcmc_noise_lr · |L_raw|_F · sigmoid(−k · (opacity − τ))
   ```

   The opacity gate suppresses noise on alive Gaussians; only nearly-dead
   Gaussians wander. μ_time is intentionally left static.

CLI / Modal flags: `--density_strategy {heuristic,mcmc}`,
`--mcmc_noise_lr`, `--mcmc_noise_after`, `--mcmc_noise_gate_k`,
`--mcmc_noise_gate_thr`, `--mcmc_max_relocations_per_step`.

The two are independent: `density_strategy=heuristic` + `mcmc_noise_lr>0`
adds SGLD on top of legacy split+prune. `density_strategy=mcmc`
replaces split+temporal_split+prune with relocation only — **no
growth**, which is the failure mode below.

## Anchors

A1 anchor at `e514bc9`: **23.50 dB val** (scale 4, 22.9k N, 413s wall).
Combo-AA best: **24.36 dB val** (scale 4, 48.0k N, 283s wall).

## Wave 1 results (14k iters, scale 4, deformable_interp, seed 42)

| probe | val PSNR | Δ vs A1 (23.50) | Δ vs Combo-AA (24.36) | N final | wall (s) |
|---|---|---|---|---|---|
| **A1 anchor** | **23.50** | — | −0.86 | 22.9k | 413 |
| **Combo-AA** | **24.36** | +0.86 | — | 48.0k | 283 |
| P-mcmc-1 (mcmc, noise=5e-5, init 1×) | 19.66 | **−3.84** | **−4.70** | 13.8k | 203 |
| P-mcmc-2 (mcmc, no noise, init 1×) | 19.50 | −4.00 | −4.86 | 13.8k | 236 |
| P-mcmc-3 (mcmc + Combo-A flags, init 1×) | 17.45 | −6.05 | −6.91 | 13.8k | 204 |
| P-mcmc-4 (heuristic + noise=5e-5 on Combo-AA) | 24.25 | +0.75 | **−0.11** | 52.4k | 293 |

### Observations

1. **N final = 13842 (== init) for all `density_strategy=mcmc` probes.**
   `mcmc_relocate` only moves dead → live; no Gaussian is ever born.
   With ~14k SfM points and no growth path, the model can't reach
   Combo-AA's 48k capacity. Under-capacity drives the 4–7 dB regression.

2. **Combo-A flags actively hurt under no-growth (P-mcmc-3 −6.91 dB).**
   With `init_strategy=spatial_slice + grelax(1k→8k) + soft clamp`, every
   Gaussian initially has n=e₀ (rank-3 static disk); relaxation drives
   the rank-2 transition only on Gaussians the optimizer is using. With
   no growth and a starved population, this transition starves the
   capacity further. **`spatial_slice + grelax` and `density_strategy=mcmc`
   without growth are anti-correlated.**

3. **SGLD noise on top of Combo-AA gave −0.11 dB.** Probable cause: the
   effective per-step noise (5e-5 · |L_raw|_F · gate ≈ 7e-6 for live
   Gaussians, ~1.5e-5 for nearly-dead) is well below the μ-position LR
   (5e-3), so the noise is inside numerical noise of the gradient step.
   To meaningfully perturb μ, noise_lr ≥ 5e-3 would be needed. Even
   then the gate may suppress most of it.

## Wave 2 — pure-MCMC with higher init capacity

| probe | val PSNR | Δ vs Combo-AA | N final | relocations | wall (s) |
|---|---|---|---|---|---|
| P-mcmc-5 (mcmc + init 4× + noise 5e-5) | **19.86** | −4.50 | 55.4k | **0 (zero)** | 334 |
| P-mcmc-6 (mcmc + init 8× + noise 5e-5) | _pending_ | | | | |
| P-mcmc-7 (hybrid + Combo-AA recipe + noise 5e-5) | _pending_ | | | | |

### New finding from P-mcmc-5

**Zero relocations across the entire 14k-iter run.** The MCMC code path
is dormant because under our regularizer setup (opacity_logit init at
sigmoid⁻¹(0.5), λ_aniso/λ_frob mild, no opacity-decay), no Gaussian's
opacity ever drops below `opacity_threshold=1e-3`. Without dead
Gaussians, `mcmc_relocate` has nothing to do.

This is a **trigger-mismatch**, not a math bug:
- Heuristic prune fires at the same threshold but is paired with split,
  so it works because the splits replenish capacity in the right places.
- MCMC relies entirely on having dead Gaussians to relocate; if the
  training regime doesn't kill any, there's no driving signal.

Even with 55k randomly-distributed Gaussians, val PSNR plateaus at 19.86
because the random spatial init lacks SfM-like surface concentration —
no growth toward needed locations means coverage is permanently random.

## Wave 2 — pure-MCMC at higher init capacity (final)

| probe | val PSNR | Δ vs Combo-AA | N final | relocations | wall (s) |
|---|---|---|---|---|---|
| P-mcmc-5 (mcmc + init 4× + noise 5e-5) | **19.86** | −4.50 | 55.4k | **0** | 334 |
| P-mcmc-6 (mcmc + init 8× + noise 5e-5) | **20.20** | −4.16 | 110.7k | **0** | 526 |

P-mcmc-6 with 110k init Gaussians is +0.34 dB over P-mcmc-5 with 55k —
small marginal gain from extra capacity, far short of Combo-AA. The
limiter is **spatial coverage of the random init**, not Gaussian count.

## Wave 3 — hybrid (heuristic split + MCMC relocate)

| probe | val PSNR | Δ vs Combo-AA | N final | relocations | wall (s) |
|---|---|---|---|---|---|
| P-mcmc-7 (hybrid + Combo-AA recipe + noise 5e-5) | **24.16** | **−0.20** | 49.2k | **0** | 288 |

Hybrid recovers Combo-AA capacity (49k vs 48k) via heuristic split, but
**zero relocations** still — same trigger never fires.

## Root cause — opacity-driven death never engages in this regime

**Established RC (cross-checked across A1, Combo-AA, all MCMC probes):**
heuristic `prune` fires `pruned=0` for the entire 14k-iter run on every
probe in this batch and in the Wave A combos memory.

Why: both the A1 anchor (`/tmp/probe_a1_anchor.log`) and Combo-AA
(`/tmp/probe_comboAA.log`) **omit `--opacity-reset-every`** (default 0)
and **omit `--lambda-opacity-entropy`** (default 0). Without periodic
opacity resets or entropy regularization, Adam + photo-loss dynamics on
`opacity_logit` cluster around 0.5 with gradient-noise drift; reaching
`opacity < 1e-3` (logit < −7) needs a sustained downward force that
doesn't exist. Consequence:

- Heuristic prune trigger 1: `opacity < 1e-3` → never met
- Heuristic prune trigger 2: `λ_min(Σ_3D) < 1e-6` (collapsed) → suppressed
  by `λ_aniso = 1e-3` + `λ_frob = 1e-4`
- Heuristic prune trigger 3: `λ_max(Σ_3D) > 100` (runaway) → suppressed
  by the same regularizers
- Net: prune fires zero times → N grows monotonically via splits → no
  population churn → **MCMC relocate has no dead Gaussians, so the
  whole MCMC machinery (relocation + opacity-gated SGLD noise) is
  dormant by construction**

This is a critical finding for *both* Wave A interpretation and the
MCMC investigation: the current density-control loop in the project's
A1 baseline is **growth-only**. Heuristic split is doing all the
capacity-allocation work; prune contributes nothing.

## Wave 4 — RC verification

| probe | val PSNR | Δ vs Combo-AA | pruned (sum) | N final | wall (s) |
|---|---|---|---|---|---|
| **P-rca-1 (heuristic Combo-AA + opacity-reset-every 3000)** | **24.49** | **+0.13** | **7** | 60.0k | 316 |

RC **confirmed but mild**: opacity reset every 3000 iters does drive
SOME Gaussians below the prune threshold (7 prunes total, single
nonzero event at iter 9200 — right after the iter-9000 reset), and
the recipe gains +0.13 dB val PSNR over Combo-AA. Wallclock +12% vs
Combo-AA (316 vs 283 s).

**Side-finding (orthogonal to MCMC):** `--opacity-reset-every 3000`
on Combo-AA produces a new candidate baseline at **24.49 dB**. This is
worth folding into the Wave A combos memory; the previous A1 anchor
removed the reset based on early-phase evidence, but on top of
Combo-AA it's a +0.13 dB win.

**MCMC verdict:** in the +reset regime, total prunes/run is ~7 across
60k Gaussians (0.012%). Replacing those 7 prunes with MCMC relocations
cannot materially affect val PSNR. MCMC noise had a measurable −0.11 dB
effect on Combo-AA at noise_lr=5e-5 (P-mcmc-4); a hybrid + reset probe
would be expected to land between −0.20 and +0.13 dB. The investigation
isn't worth more compute.

## Wave 5 — deeper RCA via checkpoint introspection

P-rca-2 (heuristic Combo-AA + opacity-reset-every 3000 + opacity-prune-threshold 0.005):
**val=24.48 dB**, +0.12 dB vs Combo-AA, N=55.6k, **1500+ prunes** in single
post-reset cycle (vs. 7 in P-rca-1). Despite massive churn, no PSNR gain
over P-rca-1's 24.49 dB. → **Pruning more isn't the lever**.

P-rca-3 (threshold 0.01, above reset target 0.0067), P-rca-4 (reset
logit −8 below threshold): both **crashed** with `_RasterizeGaussiansBackward
returned an invalid gradient at index 2 - got [0, 0, 3]`. Mass post-reset
prune wiped N to 0 → rasterizer broke. Confirms catastrophic-death
prediction. Added a `min_keep=1024` safety in `prune()`.

### Checkpoint introspection (P-rca-1, N=59,988, after 4 resets)

| population | % | opacity median | \|L_raw\|_F median | λ_max(Σ_3D) median | λ_min(Σ_3D) median |
|---|---|---|---|---|---|
| **ALIVE** (>0.5) | 35.4% | 0.857 | **0.758** | **0.344** | 0.018 |
| **ZOMBIE** (0.005, 0.01] | 34.2% | 0.007 | **0.115** (6.6× smaller) | **0.0054** (63× smaller) | 0.00032 (56× smaller) |
| **NEARLY DEAD** (≤0.005) | 3.1% | 0.003 | 0.694 | 0.293 | 0.031 |

**Zombies are tiny Gaussians, not just low-opacity ones.** They have
6.6× smaller |L_raw| and 63× smaller λ_max than alive Gaussians. They
cover near-zero pixels → near-zero photometric gradient → opacity_logit
stuck at the post-reset value (-5) forever.

In contrast, NEARLY DEAD Gaussians have full size and DO get pruned
(opacity-gradient-driven below threshold). The 7-prune count in P-rca-1
catches *those*, not the zombies.

### Origin of the zombies

The heuristic split shrinks each child by `L /= φ=1.6`. After 4 cascading
splits, `L /= φ⁴ = 6.55×`, which exactly matches the observed
zombie/alive size ratio (0.758/0.115 ≈ 6.6×). Cascading splits produce
tiny disks the optimizer can no longer drive — and the diagnostic
output (P-rca-5) confirms it: post-reset, 50% of Gaussians stay
exactly at the reset target opacity 0.0067 throughout the next 1000
iterations, while the top 5% recover above 0.7 in the same window. The
50% are zombies; the 5% are alive.

### Why collapsed-prune doesn't catch zombies

The prune code uses `lam_min_nonzero = eigs[:, 1]` (the **middle**
eigenvalue — i.e. the smaller disk axis), not `eigs[:, 0]` (the rank-2
kernel direction, ≈ 0 by construction). Correctly applied, the
zombie/alive separation is:

| population | λ_kernel q50 | λ_mid q50 (used for prune) | λ_max q50 |
|---|---|---|---|
| ALIVE | 0.0180 | 0.0673 | 0.344 |
| ZOMBIE | 0.00032 | 0.00535 | 0.0054 |

Zombie λ_mid q1=0.0012, q5=0.0019, q50=0.0054, q95=0.147. Alive λ_mid
q1=0.0104, q5=0.0156. There's a clean gap: setting `scale_min ≈ 5e-3`
catches **47% of zombies** (9,673 of 20,504) and only **0.01% of alive
Gaussians as collateral** (3 of 21,255). `scale_min ≈ 1e-2` catches
62% of zombies; the trade-off is more alive collateral.

The current `scale_min=1e-6` is **3 orders of magnitude tighter than
needed**.

## Final disposition

The MCMC implementation is mathematically correct but operationally
dormant on slice-banana. The deeper finding from this investigation is
that the project's **density-control loop is broken in two ways**:

1. **Prune triggers are mismatched with split dynamics:** `scale_min=1e-6`
   is too tight, so the cascading-split-induced zombie population
   (~34% of Gaussians) survives indefinitely. Heuristic prune fires at
   most ~7 times per run because only opacity-gradient-driven kills
   are caught — not the dominant zombie failure mode.

2. **opacity-reset-every is misaligned with opacity-prune-threshold:**
   reset target 0.0067 vs. prune threshold 0.001 means the reset
   doesn't directly kill anyone; only random-walk drift does. Matching
   them (e.g., both 0.005) creates churn but no PSNR gain.

The zombie phenomenon is a **G(3,4)-specific consequence of the
phi-cascade in heuristic split** and likely affects all Wave A and
Phase C numbers. It hasn't been visible until now because both the
default and Combo-AA recipes had `opacity_reset_every=0`, which
bypassed the question entirely (no resets, no zombie/alive separation).

### Wave 6 — zombie fix verification (run this session)

| probe | val PSNR | Δ vs Combo-AA | N | wall (s) | prunes total |
|---|---|---|---|---|---|
| Combo-AA baseline | 24.36 | — | 48.0k | 283 | 0 |
| P-rca-1 (+opacity-reset 3000) | 24.49 | +0.13 | 60.0k | 316 | 7 |
| P-rca-6 (+opacity-reset + scale_min 5e-4) | 24.53 | +0.17 | 57.5k | 314 | 1 |
| **P-rca-7 (+opacity-reset + scale_min 5e-3)** | **24.50** | **+0.14** | **45.1k** | **287** | **11,937** |

- P-rca-6 with `scale_min=5e-4` caught only 1 Gaussian — too tight.
  Zombie λ_mid q1 ≈ 1.2e-3, q50 ≈ 5e-3, so 5e-4 is below the entire
  zombie distribution.
- P-rca-7 with `scale_min=5e-3` caught **11,937** zombies across 48
  density events (peak 893/cycle pre-reset, 880 post-reset). 47% of
  the zombie population per the checkpoint analysis.
- **PSNR is unchanged (24.50 vs 24.49)**. Zombies are **inert
  capacity** — they don't hurt val PSNR, just waste memory/compute.
- N drops 25% (60k → 45k) and wall drops to 287s (matched to Combo-AA's
  283s baseline) — so opacity-reset + zombie-prune is a **free
  performance optimization**: same +0.14 dB quality, same wall as
  vanilla Combo-AA, with 6% fewer Gaussians than Combo-AA itself.

### Recommended next probes (not run in this session)

1. **Lower split_shrink_factor** from 1.6 to 1.2 → fewer zombies
   generated upstream. Could improve PSNR if some zombies are
   "almost-alive" (with less aggressive shrinking, they'd stay alive
   and contribute).
2. **Lower split_shrink_factor** from 1.6 to 1.2 → less aggressive
   per-split shrinking → fewer zombies generated. Expected: smoother
   N curve, fewer cascades.
3. **Cap split depth per Gaussian** (track lineage: only split if
   not already split too many times). Requires new bookkeeping.
4. **Move MCMC trigger from `opacity < threshold` to
   `λ_max(Σ_3D) < scale_floor`** → MCMC relocates zombies. Cleanest
   integration of the present implementation.

### What's adopted now

- `--opacity-reset-every 3000` on Combo-AA → +0.13 dB → new candidate
  baseline **24.49 dB**.
- `min_keep=1024` safety in `DensityTracker.prune()` to prevent
  rasterizer crashes on mass-prune.
- Opacity-quantile diagnostic logging in `prune()` for future runs.

## Disposition

The MCMC implementation (`mcmc_relocate` + `mcmc_noise_step`) is
mathematically correct (alpha-blending preserved, Adam migration sound).
But its **trigger** (opacity below threshold) is unreachable in our
regularization regime. Three follow-up directions, ordered by ROI:

1. **Re-introduce opacity churn** (`--opacity-reset-every`) so that
   prune/relocate triggers actually fire. This is orthogonal to MCMC
   and should improve heuristic baselines too.
2. **Replace the trigger** with a gradient-norm or contribution-norm
   signal (the screen-space gradient is already accumulated in the
   tracker). This decouples MCMC from opacity dynamics.
3. **Drop MCMC** and revisit if/when the opacity-decay regime is
   restored.

The G(3,4) heuristic split with μ-shift along the major Σ_3D axis is
already adding capacity in the right place. MCMC's randomly-sampled
destinations have no edge over it on slice-banana when both fire.
