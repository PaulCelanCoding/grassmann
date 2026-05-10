# Quality knobs — evaluation of 30 architecture-specific ideas

**Date:** 2026-05-10
**Context:** Phase-C 14k probes on slice-banana left a **1.47 dB residual** to D3DGS that single-variable CLI levers (B1–B5) failed to close (`phaseC_14k_lrdecay_probes.md`). All 30 ideas below require implementation; almost none are pure-CLI. Each entry: **effort / probability / expected ΔdB / risk / verdict**.

Effort: **S** ≤ 2h · **M** ≤ 1 day · **L** ≤ 3 days · **XL** > 3 days
Probability: gut-feel chance it materially moves residual

## Anchor for all comparisons

A1 (current code): **25.82 dB** apples-to-apples (slice-banana scale-8, deformable_interp, seed 42, SH3, 14k iters, LR-decay 0.01, λ_frob=1e-4, λ_aniso=1e-3, densify-every 200, grad-thr 1e-5, sigma_init_sq 0.02). Wallclock ~250 s on Modal L4.

---

## 1. Input data

| # | idea | effort | prob | ΔdB est | verdict |
|---|---|---|---|---|---|
| 1.1 | Per-frame learnable exposure (log a_i, b_i + L2 reg) | **S** | Med-High | 0.1–0.3 | **GREEN** |
| 1.2 | Soft dynamic mask via residual-history EMA reweighting | M | Med | −0.2 to +0.3 | YELLOW |
| 1.3 | Coarse-to-fine resolution schedule (8 → 4 → 2) | M | Low-Med | 0–0.2 | YELLOW |

**1.1**: NeRFies/DyCheck has known AE/AWB drift. Cheap, standard. Likeliest small win.
**1.2**: Cuts both ways — downweighting high-residual pixels can *worsen* dynamic-frame coverage where the residual already clusters (per Phase-C per-frame: B2/B3/B4 peak on f146/f234/f238).
**1.3**: A1 already trains at scale-4; only marginal landscape benefit on top.

## 2. Poses / SfM

| # | idea | effort | prob | ΔdB est | verdict |
|---|---|---|---|---|---|
| 2.1 | Pose refinement during training (lr_R=1e-5, lr_t=1e-4, from iter 2k) | **M** | **High** | 0.3–1.0+ | **GREEN — top** |
| 2.2 | Time-conditioned pose deltas + temporal smoothness | M-L | Low-Med | 0–0.2 | YELLOW |
| 2.3 | Photometric BA inner loop every N iters | L | Med | 0.2–0.5 (after 2.1) | YELLOW |

**2.1** is the single highest-EV item. Per `surfel_rasterizer_ab.md` cross-render the regression is in the **checkpoint** (geometry/poses), not the renderer. NeRFies poses are sub-pixel inaccurate; gsplat-style joint refinement is the obvious thing we don't do. **If 2.1 closes >0.5 dB, the architectural items (#4.1, #5.1) become unnecessary.**
**2.2/2.3** only worthwhile if 2.1 underperforms.

## 3. Initialization

| # | idea | effort | prob | ΔdB est | verdict |
|---|---|---|---|---|---|
| 3.1 | k-NN-based σ_init per point (4D distance with α weighted Δt) | **S** | Med-High | 0.1–0.3 | **GREEN** |
| 3.2 | Progressive Grassmann relaxation (n clamped to e₀, relaxed 0→3k) | S-M | Med | 0.0–0.3 | YELLOW-GREEN |
| 3.3 | Motion-aware t₀ from first-observable frame (MASt3R only) | S | Low-Med | 0–0.2 | YELLOW |

**3.1**: real gap vs vanilla 3DGS; we run a single `sigma_init_sq=0.02` for all points. Cheap, well-known.
**3.2**: clever fix to the spatial_slice Sichtstrahl-collapse (memory: random > spatial_slice 3× on this scene). Risk: if relaxation rate is wrong, regress to either branch.
**3.3**: only meaningful with MASt3R observability; dataset-loader cost.

## 4. Densification & pruning

| # | idea | effort | prob | ΔdB est | verdict |
|---|---|---|---|---|---|
| 4.1 | 3DGS-MCMC relocation (Kheradmand et al.) replaces split | **L** | Med-High | 0.3–1.0 | GREEN — expensive |
| 4.2 | Temporal-axis split (Σ_tt large + ∂L/∂t high) | S-M | Med | 0.1–0.3 | **GREEN** |
| 4.3 | Error-weighted local grad threshold (per-patch normalization) | M | Med | 0.1–0.3 | YELLOW |

**4.1** is your top guess; it directly attacks the 32% dead-rate and the split-direction ambiguity in G(3,4). Risk: substantial impl, results may be **null** because the residual could be poses (#2.1), not DC.
**4.2** is a concrete gap — current density_control.py only triggers on screen-space ‖∇μ_2d‖, never on temporal stress. Low-effort, monocular-dynamic-specific.
**4.3** risks cancelling the underfit signal you actually want.

## 5. Loss / Optimizer

| # | idea | effort | prob | ΔdB est | verdict |
|---|---|---|---|---|---|
| 5.1 | Riemannian Adam on G(3,4) (S³ × Stiefel quotient) | **XL** | Med | −0.5 to +1.0 | YELLOW — high variance |
| 5.2 | Color-LR warmup (lr_color: 0 → 2e-2 over 1k iters) | S | Med | 0–0.2 | **GREEN** |
| 5.3 | Time-coherence regularizer ‖μ_3D(t+Δt) − μ_3D(t)‖² · w | S-M | Med | 0.1–0.3 | YELLOW-GREEN |

**5.1**: the architectural-purity bet. **Concern**: Euclidean+renormalize on S³ is empirically hard to beat; the literature has many "manifold-correct" optimizers that lose to naive baselines. ROI is unclear and the impl effort is real (retraction on S³, vector-transport on Stiefel quotient, μ-DOF interaction).
**5.2** matches your existing LR-decay pattern; trivial.
**5.3** explicitly uses your Schur conditioning — leverage rather than overhead.

## 6. Gaussian representation

| # | idea | effort | prob | ΔdB est | verdict |
|---|---|---|---|---|---|
| 6.1 | SH-degree warmup schedule (0 → 3 every 1k iters) | S-M | Med | 0–0.2 | YELLOW |
| 6.2 | Hard aspect-ratio clip on Σ_3D(t₀) (cap at 30 via eigval rescale) | **S** | Med | 0.1–0.3 | **GREEN** |
| 6.3 | Opacity-entropy regularizer (push α to {0,1}) | S | Low-Med | 0–0.2 | YELLOW |

**6.1**: A1 already runs SH3 fixed (per `surfel_rasterizer_ab.md`); the question is whether warmup beats fixed-SH3 by reducing early dynamic-shadow overfit. Plausible but small.
**6.2** is the right counter to "λ_aniso leaves p99=6.8e7 tail" — penalty alone is too soft. Cheap post-step op.
**6.3** likely subsumed by opacity-reset; small additional gain.

## 7. Rasterization

| # | idea | effort | prob | ΔdB est | verdict |
|---|---|---|---|---|---|
| 7.1 | Mip-splatting-style 3D smoothing filter (resolution-adaptive σ²·I added to Σ_3D) | M | Med-High | 0.1–0.3 | **GREEN** |
| 7.2 | Random background each iter | **S** | Med | 0.1–0.3 | **GREEN** |
| 7.3 | Mixed-backend dispatch (high-aniso → surfel, low → fast) | L | Low-Med | 0–0.3 | YELLOW |

**7.1** replaces the `σ_lift²=1e-4` hack with an honest filter. Should help tangential rank-2 disks specifically.
**7.2** classical 3DGS trick we never adopted; cheap.
**7.3** premature given surfel-A/B currently shows mean regression; only after surfel-on-edges hypothesis is verified.

## 8. Known artifacts → regularizer probes

| # | idea | effort | prob | ΔdB est | verdict |
|---|---|---|---|---|---|
| 8.1 | Floater pruning via multi-view consensus (active in <3 frames OR single-time-phase) | M | Med | 0.1–0.3 | YELLOW-GREEN |
| 8.2 | Adaptive λ_frob on rank-1-collapse early warning | S | Low-Med | 0–0.2 | YELLOW |
| 8.3 | StopThePop-style windowed per-pixel sorting | **XL** | ? | ? | **RED** (out of scope) |

**8.1** monocular-dynamic-specific; targets ghosts that 4D-coverage masks. Useful if floaters dominate residual frames.
**8.2** addresses the 7% tail; small absolute impact.
**8.3** kernel rewrite; defer.

## 9. Sparse-view priors

| # | idea | effort | prob | ΔdB est | verdict |
|---|---|---|---|---|---|
| 9.1 | DepthAnythingV2 monocular depth prior + scale-shift align | **M-L** | **High** | 0.5–1.5 | **GREEN** |
| 9.2 | RAFT optical-flow constraint on projected μ_3D(t) trajectory | L | Med | 0.2–0.5 | YELLOW |
| 9.3 | Omnidata normal prior (couples with surfel) | L | Low-Med | 0–0.3 | YELLOW |

**9.1**: the standard sparse-view fix. Empirically 0.5–2 dB on monocular dynamic. **Strong candidate to close the 1.47 dB residual on its own.**
**9.2** uses your Schur trajectory natively, but RAFT has its own failure modes on dynamic objects.
**9.3** depends on surfel which is currently regressed; coupled risk.

---

## Revised priority sequence (vs your proposal)

I disagree with putting #4.1 (MCMC) and #5.1 (Riemannian Adam) first. Both are XL-effort, high-variance, and the residual may evaporate after low-effort items. **Run cheap-and-standard first, then architectural.**

### Wave A — low effort, established track record (~3-5 days total)

1. **#2.1 Pose refinement** — single highest-EV item; if it closes >0.5 dB, half of #4 / #5 is moot.
2. **#9.1 DepthAnythingV2 prior** — standard sparse-view fix, often closes residuals on monocular dynamic alone.
3. **#3.1 k-NN σ_init** — cheap gap vs vanilla 3DGS.
4. **#6.2 Hard aspect-ratio clip** — addresses p99 tail directly.
5. **#7.2 Random background** — cheap trick.
6. **#7.1 Mip-splatting filter** — honest replacement for σ_lift hack.
7. **#1.1 Per-frame exposure** — cheap, AE/AWB-drift-targeted.
8. **#4.2 Temporal-axis split** — concrete monocular-DC gap.
9. **#5.2 Color-LR warmup** — trivial.
10. **#5.3 Time-coherence reg** — leverages Schur.

After Wave A: re-measure residual. If ≤0.5 dB, stop. Otherwise:

### Wave B — architectural, high cost

- **#5.1 Riemannian Adam** — only if residual persists and Wave A pinned the cause to optimization geometry (e.g. you see persistent S³ drift / renormalize churn).
- **#4.1 MCMC relocation** — only if Wave A pinned the cause to densification (residual concentrates on under-densified regions even after #4.2).

### Wave C — deferred / dataset-coupled

- 1.2, 1.3, 2.2, 2.3, 3.2, 3.3, 4.3, 6.1, 6.3, 7.3, 8.1, 8.2, 9.2, 9.3
- 8.3 explicitly RED.

---

## What "probing on Modal" looks like per item

Modal is parameterized by CLI flags only. None of Wave A is currently CLI-runnable — each requires code in `grassmann/` first. Per-item path:

| item | code touched | new CLI flag | added s/iter (est) |
|---|---|---|---|
| 2.1 | `training.py` (R, t to optimizer); per-frame nn.Parameter | `--refine_poses --lr_R --lr_t --pose_warmup_iter` | +1–3 ms |
| 9.1 | new `priors/depth_anything.py` (download + cache); `losses.py` (scale-shift + L1) | `--lambda_depth --depth_model` | +5–15 ms |
| 3.1 | `initialization.py` (k-NN over 4D) | `--sigma_init_knn_k --sigma_init_alpha_t` | 0 (init only) |
| 6.2 | `trainable.py` post-step hook (eigh + clip + reproject) | `--max_aspect_ratio` | +1–2 ms |
| 7.2 | `training.py` (random bg per step) | `--random_background` | 0 |
| 7.1 | `rasterizer.py` / `fast_rasterizer.py` (cov modification) | `--mip_filter_sigma_pixel` | +0 |
| 1.1 | `trainable.py` (per-frame log_a, log_b); `training.py` (apply before loss) | `--exposure_per_frame --lambda_exposure_reg` | +0.5 ms |
| 4.2 | `density_control.py` (temporal grad capture + split criterion) | `--temporal_split_threshold` | +0.5 ms |
| 5.2 | `training.py` (LR ramp on color group) | `--color_lr_warmup_iter` | 0 |
| 5.3 | `losses.py` (sample (t, t+Δt) pairs) | `--lambda_time_coherence --time_coherence_dt` | +1–2 ms |

**Each item gets one A1 anchor + one Δ probe at 14k.** Cost ≈ 250 s × 2 × 10 = 5000 s ≈ 90 min L4 wall in parallel batches of 4.

## Decision points before launching

1. Confirm A1 reproduces under current commit (`e514bc9`) — `surfel_rasterizer_ab.md` already noted 0.21 dB drift between commits. Without a stable anchor every Δ is meaningless.
2. Decide Wave A order — I recommend **2.1 → 9.1 first** (highest EV) before the cheap-but-small items, because if either closes the residual, the cheap items become noise-floor decoration.
3. Implementation budget — Wave A as drafted is 3-5 days of code. Need user buy-in on which subset to commit.
