# Trigger-vs-distribution audit — 5 candidate bugs

**Date:** 2026-05-10
**Branch:** monocular-init
**TL;DR:** Following the φ-cascade-zombie RCA, ran a full audit
(`scripts/audit_triggers.py`) on the P-rca-1 checkpoint
(N=59,988, Combo-AA + opacity-reset-every-3000) cross-referenced with
log fire-rates across 5 probes. Found **5 additional trigger-vs-distribution
mismatches** that follow the same pattern as scale_min: the threshold
sits at the wrong percentile of the actual statistic.

## Method

Two cross-checked probes per trigger:
1. **Distribution check**: load the checkpoint, compute the trigger
   quantity (opacity, λ_max, Σ_tt, etc.) for every Gaussian, and look
   at where the threshold falls in the quantile distribution.
2. **Fire-rate audit**: parse density-cycle prints from `/tmp/probe_*.log`
   and count actual trigger fires per probe.

A "bug" here is any threshold that sits well outside its statistic's
support (so the mechanism never engages) or well inside the dense
part of the distribution (so it fires too aggressively, e.g. mass death).

## Findings

### Bug A — `scale_max=100` runaway prune is fully dormant

```
λ_max(Σ_3D)  max  = 5.61      (entire population)
threshold    = 100            ← 18× higher than max value
```

In every probe checked, `runaway=0` for every density cycle.

**Recommendation:** tighten `scale_max` to ~2.0 (q99 of λ_max = 1.16,
max = 5.6). Currently dead code that gives no signal.

### Bug B (already fixed) — `scale_min=1e-6` collapsed prune was 1500× too tight

This is the φ-cascade-zombie issue. Fixed in `e782f73` via
`--scale-min-prune 5e-3`. See `mcmc_probe_results.md` for the full RCA.

### Bug C — μ_t out-of-bounds (Gaussians that left the time domain)

```
scene time normalization: t ∈ [0, 1]
μ_t spread: min=−0.70  max=+1.62  mean=0.46
```

Some Gaussians have time-mean outside the scene's temporal support.
Their `w_t = exp(-(t-μ_t)² / 2Σ_tt)` weight is near-zero for all
rendered frames → zero photometric gradient → they become functionally
inert. This is a **separate failure mode that overlaps with the zombie
population** (both produce inert Gaussians, but for different reasons).

The current density-control loop has no μ_t clamp or out-of-bounds
prune. Worth probing: `--prune-mu-t-outside-scene` flag that drops
Gaussians with μ_t < -0.1 or > 1.1.

Empirical breakdown:

```
              μ_t q1     q5     q50    q95    q99    OOB (<-0.1 or >1.1)
ALIVE        -0.009   0.049   0.468  0.943  1.008    15  (0.07%)
ZOMBIE       -0.093  -0.022   0.492  0.831  1.075   220  (1.07%)
```

OOB is **15× more common in zombies than alive Gaussians** but only
1.07% of zombies are OOB — **not the dominant zombie driver**. 99% of
zombies have μ_t inside the scene range; their inertness comes from
the spatial+temporal cascade (tiny |L|_F + tiny Σ_tt), not from
out-of-domain μ_t.

Separate observation: zombie Σ_tt q50 is essentially **zero** (median
0.0000 to 5 decimals). Zombies have collapsed temporally — their w_t
blur radius is so small they only contribute at one specific instant.
This is a *consequence* of cascading splits with no μ_t / Σ_tt
refinement, not a separate bug.

**Recommendation:** add μ_t out-of-bounds prune for hygiene (drops 0–1%
of population) but it's not the lever. The zombie fix (`scale_min=5e-3`)
already catches these via the spatial trigger.

### Bug D — `lambda_aniso=1e-3` doesn't actually bound aspect ratio

```
aspect = λ_max / λ_mid       (Σ_3D in-plane aspect of the disk)
  q50  = 3.06
  q90  = 15.92
  q99  = 28.31
  max  = 29.99
```

20% of Gaussians (11,516 of 59,988) have aspect > 10. With
`max_aspect_ratio=30` (a Wave A flag that hard-clips via SVD), the
distribution stops exactly at 30 — the SVD clip is the only thing
keeping aspect bounded. The `lambda_aniso=1e-3` regularizer is too
weak to actually pressure aspect.

This is **not necessarily a bug** — the SVD aspect-clip was added in
Wave A precisely because the regularizer was insufficient. But the
overlap is wasted: the regularizer adds compute with no observable
effect. Either tighten λ_aniso (to 1e-2 or 1e-1) so it does the
clipping job and remove the SVD clip, or drop λ_aniso since it's
redundant with `max_aspect_ratio`.

### Bug E — `temporal_split_threshold=0.1` catches very little

```
Σ_tt distribution:
  q50  = 0.0016
  q90  = 0.017
  q99  = 0.085      ← below threshold
  max  = 0.333
```

Only 0.7% of Gaussians (418 of 59,988) have Σ_tt > 0.1. Per-cycle
tsplit fires 5–14 times in Combo-AA and 9–14 times in P-rca-7. This is
catching the very tail; could be lowered to 0.03 (q98) or 0.05 (q99)
to catch more Gaussians that have non-trivial temporal extent.

Wave A's `temporal_split_threshold=0.1` was tuned earlier; lowering
toward 0.03 would be a small probe to test whether more aggressive
temporal splitting helps PSNR.

## Aggregate fire rates (5 probes, 48 density events each)

| probe | splits | tsplits | prunes | notes |
|---|---|---|---|---|
| Combo-AA | 33,949 | 242 | **0** | growth-only |
| Combo-AA + opacity-reset 3000 | 45,704 | 449 | 7 | 7 prunes from random walk |
| Combo-AA + opacity-reset + threshold 0.005 | 45,048 | 651 | **3,966** | 1500/cycle post-reset |
| Combo-AA + opacity-reset + scale_min 5e-3 | 42,480 | 687 | **11,937** | the zombie fix; ~250/cycle steady |

## Disposition (verified by probes)

All 4 candidate fixes probed on top of P-rca-7 (val=24.50 dB, N=45.1k,
wall=287s). Results:

| Bug | Fix probed | val PSNR | Δ vs P-rca-7 | N | wall | verdict |
|---|---|---|---|---|---|---|
| A | `scale_max_prune=2.0` (was 100) | **24.08** | **−0.42** | 39.7k | 271s | **HURTS** — prunes useful large disks; original 100 was correctly dormant |
| C | `mu_t ∈ [-0.05, 1.05]` prune | 24.50 | +0.00 | 44.8k | 275s | **NULL** — only 1% of pop is OOB |
| **D** | **`lambda_aniso=0`** (was 1e-3) | **24.62** | **+0.12** | 46.8k | **210s** | **WIN +0.12 dB AND −27% wall** |
| E | `temporal_split_threshold=0.03` (was 0.1) | 24.53 | +0.03 | 57.3k | 307s | marginal: +0.03 dB, +N, +wall |

### Counter-finding: Bug A — dormant ≠ should-fire

`scale_max=100` was correctly dormant. Tightening to 2.0 prunes the top
~10% of Gaussians by λ_max — exactly the large disks that cover lots of
pixels. Result: −0.42 dB val PSNR, ~14% smaller N (the actually-useful
big Gaussians are gone). **Lesson:** a dormant trigger can be dormant
because the pathology it catches doesn't occur in healthy training. The
audit method should distinguish "dormant ∧ pathology present" (e.g.
zombies/scale_min) from "dormant ∧ pathology absent" (runaway/scale_max).

### Headline finding: Bug D — redundant `lambda_aniso` was net-negative

The Wave A `--max-aspect-ratio 30` flag adds a hard SVD aspect-clip every
100 iterations that bounds the disk aspect to ≤ 30. The
`lambda_aniso=1e-3` regularizer, originally added in Phase A to suppress
runaway anisotropy, was made redundant by the SVD clip but stayed in the
recipe. Empirical effect of removing it:

- **val PSNR: +0.12 dB** (24.50 → 24.62). Likely from removing
  gradient noise contributed by the eigh-backward pathology near
  near-degenerate disk eigenvalues (this is the same pathology
  documented in `results/rca/surfel_rasterizer_ab.md`).
- **wallclock: −27%** (287 → 210s). The regularizer requires:
    - Extra `compute_derived(params)` forward pass per iter
    - Extra `condition_on_time` call per iter
    - **Per-Gaussian 3×3 `torch.linalg.eigvalsh(Sigma_3D_t)`** per iter
      (60k eigvalsh ops/step at peak)
    - The unstable eigh backward chain
- **N: ~unchanged** (46.8k vs 45.1k).

**New candidate baseline:** Combo-AA + opacity-reset 3000 + scale_min 5e-3
+ lambda_aniso 0 → **val=24.62 dB, N=46.8k, wall=210s**. That's
**+0.26 dB and −26% wall vs Combo-AA's 24.36 / 283s**.

### Bug E — small marginal effect

`temporal_split_threshold=0.03` (was 0.1) makes temporal splits fire on
Σ_tt > 0.03 instead of > 0.1 — catches more Gaussians (Σ_tt q97 ≈ 0.03).
Net: +0.03 dB (within noise), +27% N (57k vs 45k), +7% wall. Not worth
adopting; the small PSNR gain doesn't justify the N/wall growth.

### Reusability

The audit script (`scripts/audit_triggers.py`) is reusable: drop a
new checkpoint into it for any future RCA round. The new baseline
recipe is in `scripts/launch_bug_probes.sh` (Bug D variant).
