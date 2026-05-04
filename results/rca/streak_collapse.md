# RCA — N3DV streak collapse

**TL;DR.** The streaks in `diagnose_render_iter*.png` are not a training pathology. They are the structural consequence of `init_gaussian_from_point` producing splats whose 3D shape is depth-aligned with **one reference camera**, rendered from a **different** camera. The Grassmann parameterization is rank-2 in space (a 2D ellipse in 3D), and ray-init commits that ellipse plane to one camera's frame. From any other camera, the ellipse is seen edge-on as a streak.

Three additional bugs compound but are not the root cause.

---

## Reproduction

All evidence below is on synthetic — no N3DV data on disk. Setup mimics N3DV camera topology: 21 cameras clustered along X axis near origin, all looking +Z, fx=fy=600, image 320×240.

| Run | What | Image | Result |
|---|---|---|---|
| 04_ref_cam | 500 splats, all ref_cam=0, render from cam 0 | `/tmp/rca/04_ref_cam.png` | Round dots, aspect 1:1 |
| 04_far_cam | Same splats, render from cam 20 (1m baseline) | `/tmp/rca/04_far_cam.png` | Horizontal streaks, aspect 617× median, 3347× max |
| 05 | Same as 04 but σ_αα=1e-8 | `/tmp/rca/05_falsify_sigma_aa_zero.png` | Round dots from cam 20 too. Aspect 1:1 |
| 07 | 200 splats, train 200 iter on cam 0 only | `/tmp/rca/07_after_train_cam0.png` (dots) `/tmp/rca/07_after_train_cam2.png` (streaks) | Trained view fine; held-out view shows the exact N3DV symptom |

## Mechanism

`init_gaussian_from_point(X, t, cameras)`:

1. Picks `ref_cam_idx` = camera most aligned with line-of-sight to X (`pick_reference_camera`)
2. Builds `(p, q)` from line through `c_ref` in direction `û = (X − c_ref)/|·|` (a sight ray)
3. With this construction:
   - `e1_hat_imag = r·(p+q)` is parallel to `û`
   - `e2_hat_imag = -r·(p×q)` is perpendicular to `û` (in the plane spanned by `û` and the foot-of-perpendicular `y`)
4. `J_embed = [e1_hat_imag | e2_hat_imag]` has columns spanning the plane `span(û, y)`, which **contains û** by construction
5. Σ_3D = J_embed · Σ_k · J_embed.T is supported on `span(û, y)` — a 2D plane that contains the sight ray to the reference camera
6. σ_αα contributes variance along `û` (depth from ref cam); σ_ββ contributes variance perpendicular to `û` (within `span(û, y)`)

Rendering from any camera C ≠ ref_cam:
- The 3D ellipse plane `span(û_ref, y)` is *not* perpendicular to C's view direction
- σ_αα-extent along `û_ref` is *transverse* from C's perspective (some component, not pure depth)
- Perspective Jacobian projects the depth-aligned-to-C₀ direction onto C's image as a 2D direction radiating from C's principal point
- Result: streak in the projected image

Confirmed by quantitative aspect ratio sweep across cameras (test 04): aspect ratio scales linearly with baseline distance from reference camera.

## Why σ_αα=0 fixes the streak

Σ_3D's eigenvalue along `û` becomes 0. The remaining eigenvalue along the perpendicular direction in `span(û, y)` is approximately perpendicular to the camera's optical axis for most rendering geometries, so it projects as a roughly round 2D Gaussian. Aspect ratio collapses to 1:1.

This is *diagnostic*, not *fix*. Setting σ_αα=0 means zero radial extent, which kills any ability to model depth uncertainty.

## Why this is structural, not tunable

The Grassmann rendering equation in this codebase produces **rank-2 Σ_3D always** (J_embed is 3×2 by construction). A rank-2 covariance in 3D is a flat ellipse, not an ellipsoid. To look round from camera C, the ellipse plane must be ⊥ C's view. Ray-init commits the plane to contain `û` from one specific camera. That commitment cannot be revoked by tuning σ_αα, σ_ββ, σ_αβ — the (p, q) themselves encode the plane.

To get a "billboard" ellipse facing C (perpendicular to C's view): need to choose (p, q) such that `span(p+q, p×q)` ⊥ C's view direction. Achievable for one C at a time. Not achievable simultaneously for 21 cameras with different view directions.

This is a fundamental limitation of the **rank-2 spatial parameterization** vs. standard 3DGS's rank-3 ellipsoids. The paper acknowledges Σ_3D is rank-2 (Theorem / Remark 20) but the multi-view rendering implications appear undocumented.

## Compounding bugs (not root cause, but contribute)

### Bug A — `pick_reference_camera` is greedy single-camera

For 30k random points, each splat picks ONE of 21 cameras as reference. From every other camera, that splat is a streak. There is no init that satisfies all cameras simultaneously with rank-2 splats.

### Bug B — σ_k double role

`sigma_k` is used as:
- Temporal variance in `gaussian.py:155`: `Σ_tt = sigma_tt_pure + params.sigma_k`
- 2D pixel variance in `rasterizer.py:80`: `cov2d += params.sigma_k * I_2`

With the train_n3dv default `sigma_k=20`:
- Temporal: w_t falloff is ~5 frames around v_0 (good — masks Bug C below)
- Pixel: minor axis floored at √20 = 4.5 px in 240-px image — large but not catastrophic
- Without σ_k=20: with sigma_bb=0.05, c≈0.5, Σ_tt = 0.038 → w_t(Δt=1) ≈ 0; every splat exists in exactly one frame (Bug C)

So σ_k=20 is actually a workaround masking Bug C. Splitting the API into `sigma_k_pixel` and `sigma_k_temporal` is correct but does not fix the streaks.

### Bug C — sigma_bb=0.05 default is way too small for time-encoded-as-frame-index

With v_0 ∈ [0, 299] and σ_ββ=0.05, the temporal weight kernel has std-dev √(σ_ββ·(1+c)/2) ≈ 0.16 frames (worst case). At Δt=1, w_t ≈ 0. Each splat is invisible in all but one frame.

Workaround in current code: σ_k=20 (Bug B) compensates. Real fix: σ_ββ should scale with the time range — for 300 frames either set σ_ββ ~ 10 (≈3-frame std), or normalize times to [0, 1].

### Bug D — Adam-reset on every density event

`density_control.py:densify_and_prune` calls `optimizer_builder(self.model)` which builds Adam from scratch. Standard 3DGS preserves moments for kept splats. With `densify_every=500, densify_stop=15000` this fires 30 times → roughly 30·~75 = 2250 wasted iterations of Adam re-warmup.

Not part of the streak symptom but matters for any subsequent training plan.

## What "fixing init" actually means

If the project's commitment is to keep the Grassmann rank-2 parameterization, the multi-view conflict is inherent. Two possible directions, both nontrivial:

1. **Multi-view billboard init**: pick (p, q) such that `span(p+q, p×q)` is some compromise plane (e.g., perpendicular to a representative view direction averaged over cameras). Splats look acceptable from all cameras but ideal from none. Possibly reasonable for clustered camera setups like N3DV; degrades for wide baselines.

2. **Train (p, q) aggressively from start**: accept ugly init, rely on density control + (p, q)-Adam to rotate splats toward something that works multi-view. Requires that the Riemannian-Euclidean-Adam-renorm strategy actually converges. Currently no evidence either way at scale (P5 stress test missing per `plan.md`).

A third direction breaks the rank-2 commitment:

3. **Bag of two splats per point**: pair each splat with a second perpendicular-orientation splat. Doubles count but cross-supports view directions. Standard 3DGS handles this via 3D ellipsoid — here it would be an explicit duplication.

I have no opinion on which is right for the project until the research goal is articulated.

## Confirmed claims

- ✅ Σ_3D is rank-2 with two specific spatial directions: `û` (sight ray) and y (foot-of-perp from origin). Verified numerically.
- ✅ Streaks appear from non-reference cameras and aspect ratio scales with baseline. Verified with sweep over 5 camera positions, ratios from 1:1 (ref) to 617:1 (max baseline).
- ✅ σ_αα → 0 collapses streaks to dots from all cameras. Falsification confirms σ_αα-along-ray is the elongation source.
- ✅ Same symptom reproduces in trained scenario (200 iter, single training view) — held-out cameras show streaks.

## Refuted / corrected claims from earlier analysis

- ❌ "Σ_3D is rank-1 along sight ray for all ray-init splats" — only true when the world origin lies on the sight line (degenerate case). Off-axis points give rank-2.
- ❌ "σ_αβ = 0.001 fixes it via conditioning" — the conditioning shrinkage is along û (parallel to e1_hat_imag), so it does not break the streak topology; only shifts the mean.
- ❌ "σ_k=20 is the bug" — it's masking a worse bug (σ_ββ=0.05 is too small for [0,299] time range). Both need fixing in coordination.
