# μ-DOF A/B test — does constraining μ ∈ n^⊥ help?

**Date:** 2026-05-04
**Setup:** slice-banana, 14k iters, seed 42, SH3, LR-decay 0.01, deformable_interp
split. Density control + Phase-A penalties OFF (intentional — keeps the test
focused on the μ-DOF lever; absolute PSNRs are not comparable to the A1 anchor
at 26.03 dB scale-8). Modal L4. 1 seed/arm.

## Question

`grassmann/gaussian.py:11` claims μ ∈ R⁴ has only "3 effective DOF" because the
n-component is invisible after projection. The claim fails: a shift μ → μ + λn
changes v_0 by λn_0 and V_k by λn_{1:}, which in turn affects:

1. **V_3D(t_0)** shifts by λ(n_{1:} − n_0·c_world/Σ_tt^pure). Invariance for
   all t_0 requires n_{1:} = (n_0/Σ_tt^pure)·c_world.
2. **w_t** depends on (t_0 − v_0)². Invariance for all t_0 requires n_0 = 0.

Substituting n_0 = 0 into (1) gives n_{1:} = 0. So n = (0, 0) = 0, contradicting
‖n‖ = 1. No shift along n leaves the render invariant; μ has **4 effective DOF**.

The test resolves this empirically: if μ truly has only 3 effective DOF, then
hard-constraining μ → P_n μ should be **PSNR-neutral**. If it has 4 effective
DOF, the constraint may help (regularization) or hurt (capacity loss).

## Arms

| arm | mu_constraint | impl |
|---|---|---|
| **free** (A) | `"free"` | legacy: μ unconstrained in R⁴ (~4 DOF) |
| project (B1) | `"project"` | `compute_derived` does μ ← (I − nn^T)μ before splitting v_0/V_k (3 DOF, hard) |
| reparam (B2) | `"reparam"` | `TrainableGaussians.forward()` returns the projected μ upstream of compute_derived (3 DOF, math-equivalent to B1) |
| penalty (B3) | `"penalty"` | μ free + soft loss term `λ · <n,μ>²`, λ=1.0 (4 DOF + bias) |

## Results

| arm | val PSNR | Δ vs free | wallclock | s/iter (with cold-start variance) |
|---|---|---|---|---|
| **free** (baseline) | **19.67 dB** | — | 369.3 s | 26.4 ms |
| project (B1) | 19.46 dB | **−0.21 dB** | 329.0 s | 23.5 ms |
| reparam (B2) | 19.57 dB | **−0.10 dB** | 431.0 s | 30.8 ms |
| penalty (B3, λ=1) | 19.47 dB | **−0.20 dB** | 340.7 s | 24.3 ms |

(Project and reparam are mathematically identical — both compute μ − ⟨n,μ⟩n,
just at different points in the autograd graph; the 0.11 dB spread is single-seed
floating-point-order noise.)

## Conclusion

All three constraint arms regress the unconstrained baseline by **~0.15-0.20 dB**.
Mean of the constrained arms = 19.50 dB, ~0.17 dB below free. At-or-just-above
typical single-seed noise (±0.10-0.15 dB), but **direction-consistent** across
three different implementations (hard-projection in two pipeline positions +
soft penalty).

**μ has 4 effective DOF.** The "3 effective DOF" claim in `gaussian.py:11`
is sloppy. The constraint isn't free regularization either — it mildly hurts.

Performance variance (329-431 s wallclock) is dominated by Modal cold-start /
GPU contention, not by the per-iter cost of the projection itself (which is one
4-vector dot + one rank-1 subtract per Gaussian, negligible vs the rasterizer
pass). No actionable s/iter signal.

## Decision

- **Total geometry DOF per Gaussian = 13**: 3 (n on S³/{±}) + 6 (L_raw modulo
  P_n kernel + column gauge) + 4 (μ ∈ R⁴).
- Math doc (`docs/maths/grassmanian_gradients.md`) records 4-DOF μ + 13 total.
- `gaussian.py:11` docstring corrected to drop the "3 effective DOF" claim.
- The `--mu_constraint` CLI flag stays in the codebase for future studies but
  defaults to `free`.
