# Grassmann Splatting on monocular video — RCA & open questions (2026-05-03)

## Setting

We're implementing the Grassmannian-Gaussian-Splatting framework
(`docs/maths/grassmann.pdf`, `jacobian_v6.pdf`) for monocular video
reconstruction (NeRFies / DyCheck format, scene = `slice-banana`,
~330 frames, 1 cam per frame). Library + tests = ~144 pytests passing
(Phases 1–7 complete: quaternions, jacobian, toy + CUDA rasterizer
adapter, trainer, density control, init strategies). GPU training runs
on Modal with the standard Inria `diff-gaussian-rasterization` CUDA
kernel via our adapter `grassmann/fast_rasterizer.py`.

## What's broken

Reconstruction quality plateaus at L1 ≈ 0.11 (≈ 17 dB PSNR) — vs. ~22 dB
typical of NeRFies / HyperNeRF on the same dataset. Renders show heavy
streaking in most configurations, and density control actively makes
things worse rather than better.

## RCA finding 1 — **Init geometry pins the rank-1 axis to the view ray**

Each Gaussian is parameterized by `(p, q, α₀, β₀, L)` with `(p, q) ∈ S²×S²`
selecting a 2-plane `E_{p,q} ⊂ R⁴ = R × R³` (time × space) and `Σ_k = LLᵀ`
the in-plane 2×2 covariance.

Per Remark 20 in the math docs, after time conditioning at `t = v₀`:

```
Σ_3D(t₀) = σ_aa · jₐ jₐᵀ          (rank 1, in 3D)
```

where `jₐ` is the spatial part of the orthonormal basis vector `ê₁`.

**The line-init `line_to_pq(x_line, û)`** (used by the legacy
`lookat`/`birth`/`median` strategies) constructs `(p, q)` such that
`p_im + q_im = 2λ·û` ⟹ `jₐ ∝ û`, **i.e. the rank-1 axis is exactly the
camera viewing ray**. From the init camera the Gaussian foreshortens to a
dot; from any other camera it projects to a screen-space line of length
`√σ_aa · sin(θ) · fx/depth`. Across ~330 frames this means streaks
everywhere — verified empirically: 100 % of init Gaussians have
`|cos(jₐ, û)| = 1.0`; after 5000 iters of training, 66 % still have
`|cos| > 0.99` (training cannot rotate out of the local minimum because
growing `σ_aa` along the bad axis is locally rewarded by L1 — more
pixels accidentally covered).

### Init-strategy ablation (5000 iters, slice-banana, scale 4)

| strategy | iter-0 axis ‖ view ray (>0.9) | final L1 | final N |
|---|---|---|---|
| `median` (legacy default) | 100 % | 0.42 | 3 318 |
| `birth` | 94 % | 0.41 | 2 591 |
| `lookat` | 92 % | 0.38 | 2 809 |
| `random` (uniform `(p, q)`) | 44 % | **0.13** | 14 980 |
| `orthogonal` (jₐ ⊥ û by construction) | 0 % | 0.46 | 9 362 |
| `tripod` (3 Gaussians/point, mutually ⊥ axes) | 32 % radial + 68 % tangent | 0.37 | 26 363 |

**`orthogonal` performed *worse* than `random`** because perpendicular
axes produce screen-aligned streaks at full projected length from every
view, while view-aligned ones at least foreshorten near the init view.
**Random init wins by averaging** — diverse axes give some Gaussians
"correct enough" geometry for some views.

## RCA finding 2 — **Density control is net-negative**

We fixed the streak issue with random init, but reconstruction still
saturates around L1 = 0.11. We expected density control to push this
lower (standard 3DGS uses it heavily). It doesn't — every variant we
tested makes things *worse* than turning DC off entirely.

### DC ablation (random init + sigma_3d_blur = 0.02, 5000 iters)

| variant | L1 | N |
|---|---|---|
| **no DC** | **0.108** | 13 842 |
| default DC | 0.131 | 14 980 |
| A — 3D streak-length threshold instead of `Σ_k λ_max` | 0.140 | 15 042 |
| B — split children get rotated `(p, q)` for orientation diversity | 0.166 | 16 637 |
| C — fix variance-shrinkage from `Σ/φ` to `Σ/φ²` (matches 3DGS) | 0.132 | 15 588 |
| A + B + C | 0.169 | 16 312 |
| **prune-only** (no clone, no split; `grad_threshold = ∞`) | **0.391** | 12 450 |

The prune-only result is the most surprising: pruning *alone* — without
the compensating splits/clones — is the worst variant of all. ~580
Gaussians pruned at iter 500 (the first event), continuing aggressively
thereafter. This means: in the default DC setting, splits weren't just
useless, they were *masking* an over-aggressive prune by re-supplying
capacity. The right pruning thresholds for monocular Grassmann are
likely much more conservative than the standard 3DGS values
(`opacity_threshold = 0.005`, `scale_min/max`).

### Mathematical reasons DC doesn't transfer cleanly

We identified four math mismatches between standard 3DGS DC and our
parameterization, all empirically confirmed by spectral analysis on the
30 k-iter random checkpoint (15 % of Gaussians have `|jb| < 0.1`,
median `|jb| = 0.49`, condition number max 5.8 × 10⁸, 51 % of splits
move children >50 % temporally instead of spatially):

1. **`Σ_k` eigvals don't measure 3D physical size.** The mapping
   `Σ_3D = J_e Σ_k J_eᵀ` has `|jₐ| ≡ 1` but `|j_b|² = (1−c)/2`
   ranging 0…1, so the same eigval can mean a 14 cm Gaussian or a 1 mm
   one depending on `(p, q)`.
2. **Splits along the (α, β) major axis often offset *temporally*, not
   spatially.** With `σ_bb > σ_aa` (the init), the major axis is `e_β`,
   which through `J_e` carries a `√((1+c)/2)` temporal component.
3. **`Σ_k` semantics depend on `(p, q)`** — children that inherit the
   parent's `L` but get a rotated `(p, q)` (our fix B) have a covariance
   shape unrelated to the parent's training history. Resetting child
   `L` to init values helps somewhat (0.30 → 0.17) but doesn't break
   even with no-DC.
4. **Shrinkage was off by a power** (Σ /= 1.6 vs. standard /= 2.56),
   making splits an essentially trivial perturbation.

Even fixing all four, the dominant problem stands: **any structural
mutation mid-training (clone / split / re-orient) disrupts the per-
Gaussian local optimum so much that the iters spent recovering exceed
the iters of progress added by extra capacity.** We saturate at the
same L1 with N = 14 k as we do with N = 23 k after splits.

## What we've ruled out

- It's not a CUDA-rasterizer bug: toy CPU rasterizer shows the same
  streaks (just slower).
- It's not an Adam state migration bug: we surgically migrate
  `exp_avg` / `exp_avg_sq` for kept rows, zero-init for new rows.
- It's not the rank-1-projection issue alone: adding 3D isotropic
  blur (`σ_3d²·I` on `Σ_3D(t_0)`) makes streaks fatter but doesn't fix
  the orientational problem.
- It's not the legacy ray-init bug alone: random init solves that, but
  only gets us to L1 = 0.11.
- It's not insufficient training: 30 000 iters with default DC ends at
  the same L1 as 5 000 iters (peak before DC kicks in was L1 = 0.12 at
  iter 2 500, then DC degrades it to 0.17, then it recovers to 0.13).

## Open questions for review

1. **Is the rank-1 `Σ_3D(t_0)` constraint (Remark 20) compatible with
   monocular reconstruction at all?** Each Gaussian renders as a 1D
   line in 3D regardless of axis choice. To get isotropic surface
   coverage, you need many Gaussians with diverse axes, and density
   control has to *both* increase N *and* diversify orientations
   simultaneously. Standard 3DGS DC does neither.

2. **Is there a principled way to design a DC for our `(p, q, α, β, L)`
   parameterization** that respects basis-dependent semantics? Splits
   need to pick a new `(p, q)` such that the child's covariance "means
   the same thing" as the parent's — but `Σ_k` is a 2-form on a basis
   that changes when `(p, q)` does, so this seems geometrically
   ill-posed.

3. **Is there a softer alternative to discrete clone/split events?**
   E.g., continuous gradient regularizers that reward "cover this region
   with more capacity" without surgically adding rows. Or randomly re-
   initialize dead Gaussians during training instead of pruning them.

4. **Would changing the gradient trigger from `‖∇(α₀, β₀)‖` to
   screen-space `‖∇μ_2d‖`** (standard 3DGS) help? We have means2D
   gradients available from the CUDA rasterizer; haven't wired them
   through yet. This was untested in the ablation above.

5. **Is the math itself negotiable?** The paper enforces rank-1
   `Σ_3D(t_0)` via the exact cancellation `Σ_3D − c c^T / σ_tt_pure`.
   Adding a 3D isotropic regularizer (σ_3d² · I) breaks this exactly
   but seems essential for renderable Gaussians. We currently use
   σ_3d_blur = 0.01–0.02 (1–2 cm in scene units).

## Repo pointers

- Init logic: `grassmann/initialization.py`
- Time conditioning + Σ_3D(t_0) math: `grassmann/gaussian.py`
- Density control: `grassmann/density_control.py`
- CUDA rasterizer adapter: `grassmann/fast_rasterizer.py`
- Trainer: `grassmann/training.py`
- Modal entrypoint for GPU runs: `scripts/train_modal.py`
- Math spec: `docs/maths/grassmann.pdf`, `jacobian_v6.pdf`
- Earlier RCA on a different bug: `streak_collapse.md`
