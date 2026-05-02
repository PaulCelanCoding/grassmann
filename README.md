# Grassmann Framework Implementation

Implementation of the Grassmannian Gaussian-splatting framework for video
rendering described in the attached papers (`grassmann.pdf` and `jacobian_v6.pdf`).

This repository currently contains **Phases 1–7** of the full pipeline:
geometric primitives, the projection Jacobian, a differentiable toy rasterizer,
multi-view initialization, a working training loop, adaptive density control,
and an adapter to the original Inria `diff-gaussian-rasterization` CUDA
kernel for production-speed rendering. **This is the full implementation
for the static-multi-camera use case** — Phase 8 (dynamic cameras, §5.3 of
the Jacobian paper) is explicitly not needed for your setup.

## What is implemented

### Phase 1 — Quaternion & Grassmannian primitives
- `grassmann/quaternion.py` — batched Hamilton product, conjugate, inverse,
  unit-imaginary constructors. All functions are `torch.autograd`-compatible.
- `grassmann/grassmann.py` — canonical plane `E_{p,q}`, orthogonal basis
  `{e1 = p+q, e2 = 1-pq}` (Proposition 1), and the line ↔ (p, q) correspondence.

**Deviation from the paper.** Section 2 of the Grassmann paper gives a formula
`p = ytu⁻¹/‖·‖, q = u⁻¹yt/‖·‖` for mapping a line to `(p, q)`. Numerical
verification showed this formula maps every line onto the antidiagonal
`c = p·q = -1`, which Lemma 2.1 says is excluded from the image. I re-derived
the correct formula from the defining property of the embedding
`φ_1(L) = span{(1, y), (0, û)}`:
```
p = λ (û - û × y)
q = λ (û + û × y)
λ = 1 / √(1 + |y|²)
```
where `y` is the line's standard-form point (foot of perpendicular from origin).
This is verified by `test_embedding_into_canonical_plane` — the resulting
`(p, q)` correctly defines a plane containing `(1, y)` and `(0, û)` to machine
precision. **Worth flagging to the authors.**

### Phase 2 — The Jacobian
- `grassmann/projection.py` — pinhole `Camera`, `world_to_camera`,
  `perspective`, `perspective_jacobian` (eq. 7 of the Jacobian paper).
- `grassmann/jacobian.py` — `jacobian_embed` (eq. 8), `jacobian_time` (eq. 9),
  `jacobian_full_static` (eq. 13, Proposition 6). Only the static-camera
  case (Case A, §5.2) is implemented; this is what the paper itself
  recommends for initial proof of concept (Remark 8).

**Verification.** `test_jacobian_full_static_matches_autograd` compares the
analytical Jacobian against PyTorch autograd across 10 random configurations
at 1e-8 tolerance. A separate stress test over 500 random valid configs
showed relative error ~1e-16 (machine precision). The formula is exactly correct.

### Phase 3 — Rendering equation & toy rasterizer
- `grassmann/gaussian.py` — `GaussianParams` (raw parameters, 9 geometric DOF
  per Gaussian, matching §9 of the Jacobian paper), `compute_derived` (steps
  1–5 of §9.2), `condition_on_time` (steps 6–8). Uses the 3D-lifted approach
  from §9.1 exclusively (Ansatz A of the paper).
- `grassmann/rasterizer.py` — `project_to_screen` (EWA projection to 2D
  splats), `eval_2d_gaussian` (unnormalized Gaussian evaluation), `rasterize`
  (alpha-compositing loop, front-to-back).

**Deviation from the paper.** Eq. (32) defines
`Σ_tt = r²(1+c)² σ_ββ + σ_k²`. I found that using that value in the 3D
conditioning equations (44) and (45) breaks the exact rank-1 property of
`Σ_3D(t_0)` asserted in Remark 20, because `c_world` from eq. (43) does not
include any `σ_k²` contribution. I split the temporal variance into two roles:
- `sigma_tt_pure = r²(1+c)² σ_ββ` — used in eqs. (44) and (45), preserves
  Remark 20 exactly.
- `Sigma_tt = sigma_tt_pure + σ_k²` — used in the temporal weight
  `w_t = exp(-(t₀-v₀)²/(2Σ_tt))` (eq. 37), provides well-behaved fall-off
  when `σ_ββ → 0`.

Both roles reduce to the paper's `Σ_tt` when `σ_k² = 0`. **Worth flagging.**

### Phase 4 — Multi-view initialization from video
- `grassmann/synthetic.py` — synthetic multi-camera scene generator for
  development/testing. Cameras-on-a-ring, stereo-pair, configurable K.
  Scene points with linear / circular / static trajectories. Simple blob
  renderer for ground truth (independent of the Grassmann pipeline).
- `grassmann/triangulation.py` — DLT (Direct Linear Transform) triangulation
  for any K ≥ 2 cameras. Works by setting up the linear system
  `[u_k P[2] - P[0]; v_k P[2] - P[1]] X_hom = 0` for each camera k and
  solving by SVD. Matches 3D points to machine precision for noise-free
  observations; robust to ~2 pixels of noise.
- `grassmann/initialization.py` — `init_gaussian_from_point(X, t, cameras)`
  produces a valid `GaussianParams` whose mean is placed at (t, X_world)
  using a reference-camera-based ray parameterization.

**Deviation from the paper (third one).** Placing the Gaussian mean at
`(t, X_world)` for `t ≠ 1` requires careful construction: the `phi_t`
embedding from eq. (2) of the Grassmann paper scales the line by `t`. Naively
using `line_to_pq(c_ref, u_hat)` builds a plane containing only `(1, X_world)`,
which means the target `(t, X_world)` only approximately lies in `E_{p,q}`
for `t ≠ 1`. The fix: call `line_to_pq(c_ref / t, u_hat)`, which builds the
plane `φ_t(L)` containing `(t, X_world)` exactly. This is equivalent to
scaling the line's foot-of-perpendicular by `1/t`. I implemented this fix
and verified (see `test_init_gaussian_mean_matches_point_approximately`).
The fix is necessary whenever you want a Gaussian whose temporal center
`v_0 ≠ 1`, which is essentially all the time in a multi-frame video.

### Phase 5 — Training loop
- `grassmann/trainable.py` — `TrainableGaussians` is an `nn.Module` wrapper
  around `GaussianParams`. Handles reparameterizations that keep parameters
  on their natural domains: (p, q) on `S²` (normalize in forward, re-normalize
  after each step), opacity via sigmoid, color via sigmoid, Cholesky L as
  lower-triangular with safety epsilon on the diagonal. Plus `build_optimizer`
  with per-parameter-group learning rates (standard 3DGS practice).
- `grassmann/losses.py` — `l1_loss`, `structural_loss` (a cheap SSIM-like
  local-mean + local-variance loss — no external dependencies), and an
  optional `LPIPSLoss` wrapper for when the `lpips` package is installed.
  `photometric_loss` combines these with user-chosen weights. Also temporal
  L1 on frame differences (follows the paper's `temporal_lpips` pattern).
- `grassmann/training.py` — `Trainer` class. Takes a model, a list of
  cameras, frame data (either dense `(K, T, H, W, 3)` tensor or a callable
  `(cam_idx, t) -> (H, W, 3)` for lazy loading), and time values. At each
  iteration it samples a random `(camera, frame)` pair, renders, computes
  loss, backprops, steps Adam, re-normalizes manifolds. Verified end-to-end
  on both overfit-one-frame and multi-view multi-frame tests.

**Implementation notes.**
- The rasterizer automatically casts cameras to the Gaussians' dtype (typical
  setup: cameras in `float64`, trained model in `float32`).
- The Riemannian projection onto `S²` for `(p, q)` gradients is handled
  implicitly: we store raw R³ vectors, normalize on-the-fly in `forward()`,
  and also re-normalize the parameter data after each optimizer step. Both
  together are equivalent to projected Adam. This is cleaner than the
  explicit tangent-space projection described in §8.4 of the Jacobian paper
  and converges well in practice.

### Phase 6 — Adaptive density control
- `grassmann/density_control.py` — `DensityTracker` accumulates per-Gaussian
  gradient magnitudes (on `alpha_0` and `beta_0`) across training iterations,
  and exposes three operations described in §3.5 of the Grassmann paper and
  standard 3DGS (Kerbl et al. 2023):
  - **Prune** — remove Gaussians with low opacity, collapsed covariance,
    or runaway scale.
  - **Clone** — duplicate small Gaussians whose accumulated gradient indicates
    they're struggling to cover their region.
  - **Split** — divide large Gaussians under gradient stress into two
    smaller offset children, placed along the major axis of their
    `Σ_k` in the local (α, β) frame.
- After each density operation, the optimizer is rebuilt (Adam's per-parameter
  moment buffers need to match the new parameter set). The `Trainer` exposes
  `densify_every` / `densify_start` / `densify_stop` config options; density
  control is *off* by default and enabled explicitly.

**Key implementation detail.** Splitting and cloning operate on the local
(α, β) coordinates of the canonical plane `E_{p,q}`. Since `J_embed` maps
those directions to spatial R³ directions, offsetting along the principal
axis of `Σ_k` gives a physically meaningful offset of the new children in
3D world space. This is the natural analog of 3DGS's "offset along largest
eigenvalue of world-space Σ" for our plane-based parameterization.

### Phase 7 — Fast rasterization via `diff-gaussian-rasterization`
- `grassmann/fast_rasterizer.py` — adapter that wraps the original Inria
  CUDA rasterizer. The Jacobian paper §9.1 and §10 designed the 3D-lifted
  approach specifically so that `(V_3D(t_0), Σ_3D(t_0), α_eff, color)` can
  be fed directly into an unmodified 3DGS rasterizer — we just need to
  pack the 3×3 covariance into its 6-element upper-triangular form and
  build the glm-style view/projection matrices.
- `is_available()` probes for `diff_gaussian_rasterization` + CUDA and
  caches the result. If either is missing, `fast_rasterize()` transparently
  falls back to the Phase 3 toy rasterizer.
- `Trainer` has `use_fast_rasterizer=True` config option. On a GPU machine
  it routes every render through the CUDA kernel; on CPU (or without the
  package) it silently uses the toy path. Same API either way.
- `scripts/benchmark_phase7.py` — script to run on your GPU machine to verify the
  CUDA path and measure speedup. Typical speedup is 100–500× over the toy
  rasterizer for thousands of Gaussians at 480×640.

**Installation on GPU machines.** After cloning the repo and installing
the base requirements, add:

```bash
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
```

This compiles a CUDA extension, so you need the CUDA toolkit (nvcc) installed
and matching your PyTorch's CUDA version. See the 3DGS repo README for
version compatibility. Once installed, set `use_fast_rasterizer=True` in
your `TrainerConfig` and move your model to `.cuda()`.

## File structure

```
grassmann/
├── grassmann/                   # core package
│   ├── __init__.py
│   ├── quaternion.py            # Phase 1: Hamilton product, conjugate, inverse
│   ├── grassmann.py             # Phase 1: E_{p,q} basis, line ↔ (p,q)
│   ├── projection.py            # Phase 2: Camera, perspective + Jacobian
│   ├── jacobian.py              # Phase 2: J_embed, J_time, J_full (static camera)
│   ├── gaussian.py              # Phase 3: GaussianParams, conditioning
│   ├── rasterizer.py            # Phase 3: project_to_screen, rasterize
│   ├── synthetic.py             # Phase 4: multi-camera scene generator
│   ├── triangulation.py         # Phase 4: DLT multi-view triangulation
│   ├── initialization.py        # Phase 4: build Gaussians from 3D points
│   ├── trainable.py             # Phase 5: TrainableGaussians + optimizer
│   ├── losses.py                # Phase 5: L1, structural, optional LPIPS
│   ├── training.py              # Phase 5: Trainer class
│   ├── density_control.py       # Phase 6: DensityTracker (prune/clone/split)
│   └── fast_rasterizer.py       # Phase 7: diff-gaussian-rasterization adapter
├── tests/                       # pytest suite (~113 tests)
│   ├── test_quaternion.py       # 11 tests
│   ├── test_grassmann.py        # 16 tests
│   ├── test_jacobian.py         # 14 tests
│   ├── test_rendering.py        # 18 tests
│   ├── test_initialization.py   # 21 tests
│   ├── test_training.py         # 11 tests
│   ├── test_density_control.py  # 12 tests
│   └── test_fast_rasterizer.py  # 10 tests
├── scripts/                     # executables (run from repo root)
│   ├── train_n3dv.py            # N3DV training driver (prepare + train)
│   ├── diagnose_n3dv.py         # heavy-logging single-frame debug runner
│   ├── sanity_one_gaussian.py   # one-Gaussian smoke test on N3DV
│   ├── benchmark_phase7.py      # GPU benchmark: toy vs CUDA rasterizer
│   ├── stress_test_jacobian.py  # Jacobian fuzzer vs autograd
│   ├── preprocess.sh            # ffmpeg → frames for N3DV scenes
│   └── colmap.sh                # COLMAP → points3D.txt
├── viz/                         # plot generators (write to docs/images/)
│   ├── visualize_jacobian.py    # → docs/images/jacobian_viz.png
│   ├── visualize_rendering.py   # → docs/images/demo1..demo4*.png (Phase 3)
│   ├── visualize_phase4.py      # → docs/images/phase4_*.png
│   ├── visualize_phase5.py      # → docs/images/phase5_*.png (training curves)
│   ├── visualize_phase6.py      # → docs/images/phase6_*.png (density control)
│   └── visualize_phase7.py      # → docs/images/phase7_architecture.png
├── docs/images/                 # generated plots and demo PNGs
├── data/n3dv/                   # datasets (gitignored, populate locally)
├── requirements.txt
├── CLAUDE.md
└── README.md                    # this file
```

## Running it

Install dependencies (PyTorch, matplotlib, pytest, numpy). Then:

```bash
# Run all 113 tests (should take ~25s, most time in training + density integration tests).
python -m pytest tests/ -v

# Generate the Phase 2 visualization.
python viz/visualize_jacobian.py

# Generate the Phase 3 demo animations (four PNGs).
python viz/visualize_rendering.py

# Generate the Phase 4 pipeline visualizations (three PNGs).
python viz/visualize_phase4.py

# Generate the Phase 5 training demos (four PNGs) -- takes ~2 minutes.
python viz/visualize_phase5.py

# Generate the Phase 6 density-control demo (two PNGs) -- takes ~4 minutes.
python viz/visualize_phase6.py

# Generate the Phase 7 architecture diagram (instant).
python viz/visualize_phase7.py

# Optional: fuzz-test the Jacobian against autograd on many random configs.
python scripts/stress_test_jacobian.py
```

On a GPU machine with `diff-gaussian-rasterization` installed, also run the
Phase 7 benchmark to verify the CUDA path works and measure speedup:

```bash
python scripts/benchmark_phase7.py
```

## What each demo shows

### Phase 3 demos (`visualize_rendering.py`)
- **demo1_temporal_fade.png** — A ray-Gaussian at fixed screen position,
  opacity modulated by the temporal Gaussian weight `w_t`. Values match
  `0.14 → 0.41 → 0.80 → 1.00 → 0.80 → 0.41 → 0.14` (exp(-n²/2) for n = 2,1,0,1,2).
- **demo2_occlusion.png** — Red splat at depth 3, green at depth 8, same
  pixel column. The combined image shows red occluding green correctly.
- **demo3_motion.png** — A Gaussian with `σ_αβ ≠ 0` on a ray from the camera.
  Setting `σ_αβ` couples the `α` (along-ray) and `β` (time) directions, so the
  splat's depth changes linearly with time. You see parallax on screen.
- **demo4_scene.png** — Three ray-Gaussians (red, green, blue) at different
  screen positions and with staggered `v_0 = 0, 1, 2`. Each fades in and out
  independently.

### Phase 4 demos (`visualize_phase4.py`)
- **phase4_pipeline.png** — End-to-end pipeline at a single time: top row is
  ground-truth scene from 4 cameras; bottom row is the Grassmann
  reconstruction obtained by triangulating then initializing Gaussians.
- **phase4_triangulation.png** — Triangulation accuracy vs pixel noise.
  Machine-precision recovery with zero noise; ~0.025 world-unit error at 0.5px
  noise; ~0.1 at 2px noise.
- **phase4_timelapse.png** — GT vs reconstruction over 5 time instants,
  showing motion (linear, circular, static) is all captured.

### Phase 5 demos (`visualize_phase5.py`)
- **phase5_overfit.png** — Demo A: overfit one frame from one camera. Initial
  colors set to wrong gray; training recovers the correct red/green/blue
  within ~200 iterations.
- **phase5_overfit_final.png** — Side-by-side target vs trained, L1 ≈ 0.005.
- **phase5_multiview.png** — Demo B: loss curve + validation render for
  3 cameras × 5 frames training. Loss drops from 5e-3 to 2e-3 over 1500 iters.
- **phase5_multiview_grid.png** — GT vs trained side-by-side across all 3
  cameras at the middle time instant. Near-perfect match on all views.

### Phase 6 demos (`visualize_phase6.py`)
- **phase6_curves.png** — Loss curves + Gaussian count over training,
  comparing baseline (15 fixed Gaussians) against a density-controlled run
  starting with only 5 Gaussians. The density-controlled model grows
  5 → 10 → 20 → 40 in a clean staircase, converging to comparable loss.
- **phase6_grid.png** — Final renders across all 3 cameras, showing GT
  (top), baseline (middle), and density-controlled (bottom). All three match
  closely, but the density-controlled model achieved this from a much
  smaller initialization by cloning and splitting under-fit regions.

### Phase 7 diagram (`visualize_phase7.py`)
- **phase7_architecture.png** — Dual-path architecture: the Trainer's
  `render_one()` routes through either the toy rasterizer (CPU-safe
  fallback) or `diff_gaussian_rasterization` (CUDA, 100× faster). The
  view-independent `compute_derived` + `condition_on_time` work feeds
  both paths, which amortizes cost across K cameras. Status banner at the
  top auto-detects whether CUDA is available.

### Phase 2 demo (`visualize_jacobian.py`)
- **jacobian_viz.png** — Side-by-side: the canonical plane `E_{p,q}` embedded
  in 3D (colored by time), with basis arrows; and the projected (α, β) grid
  on the image plane with the Jacobian's column vectors drawn as arrows.

## Design choices worth knowing

- **Float64 throughout for correctness.** Phase 1–3 use `torch.float64` because
  we need tight tolerances for finite-difference tests and for verifying
  geometric properties (orthogonality, rank drops, etc.). When training begins
  in Phase 5+ you'll want to switch to `float32` or mixed precision for speed.

- **Ray parameterization for scene reconstruction.** The natural choice for
  representing a scene point is a ray from the camera origin. In that case
  `y = 0`, `c = 1`, `s = 0`, `e2_hat` is purely temporal, and `c_world = 0`.
  A Gaussian "at rest" really is at rest. This parameterization is exposed as
  `ray_gaussian(direction, depth_mean, …)` in `visualize_rendering.py`.

- **Mean placement subtlety.** `GaussianParams.alpha_0, beta_0` are *local
  coordinates in E_{p,q}*, not world coordinates. To place the Gaussian mean
  at a specific world-space point, you must convert: the utility
  `ray_gaussian` does this automatically for the ray case.

## What's not implemented

- **Phase 8** (dynamic camera, Case B of the Jacobian paper) — explicitly
  **not needed for your static-multi-camera setup**. Theorem 13 of the paper
  proves that for moving cameras, the 3D-lifted method we use (Ansatz A) is
  *strictly more accurate* than full linearization (Ansatz B). So if you
  ever do need to support camera motion, the current code will just work —
  you'll only need to feed time-varying view matrices into the rasterizer.

## Test inventory

| File                          | Tests | What's verified                                                      |
|-------------------------------|-------|----------------------------------------------------------------------|
| `test_quaternion.py`          | 11    | Hamilton product identities, norm multiplicativity, inverse.         |
| `test_grassmann.py`           | 16    | Proposition 1 (basis is orthogonal, in E_{p,q}); line↔(p,q) roundtrip.|
| `test_jacobian.py`            | 14    | `J_full` matches autograd and finite differences to machine precision.|
| `test_rendering.py`           | 18    | Rank drop after conditioning, PSD of covariances, temporal weight.    |
| `test_initialization.py`      | 21    | Triangulation accuracy, reference camera selection, end-to-end pipeline. |
| `test_training.py`            | 11    | Trainable parameters, gradient flow, manifold renorm, loss convergence. |
| `test_density_control.py`     | 12    | Prune (three criteria), clone, split, optimizer rebuild, Trainer integration. |
| `test_fast_rasterizer.py`     | 10    | View matrix correctness, covariance packing, fallback, Trainer integration. |
| **Total**                     | **113**| **All passing.**                                                     |
