# Grassmann GS — Grassmannian Gaussian Splatting on Monocular Video

Implementation of a Grassmannian-based Gaussian splatting pipeline for
monocular video reconstruction. Each Gaussian is parameterized by a
4-dimensional plane in R^4 (a point on the Grassmannian G(3,4)) plus a
covariance factor, so the model represents a thin "disk" in space-time
that becomes a moving 2D-Gaussian splat under perspective projection.

Math spec: [`docs/maths/grassmanian_gradients_v7.md`](docs/maths/grassmanian_gradients_v7.md).

## Layout

```
grassmann/        # library (geometry, rasterization, training, density control)
  datasets/       # NeRFies + DyCheck loaders (monocular video)
scripts/          # entry points (train, render, eval) + Modal wrappers
tests/            # pytest suite
docs/maths/       # math spec (v7)
legacy/           # archived (unmaintained) experimental code
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                # if you've added a pyproject.toml
# or:
pip install -r requirements.txt
```

GPU production training additionally requires the Inria diff-gaussian
rasterizer (compiles a CUDA extension; see the commented line in
`requirements.txt`):

```bash
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
```

## Datasets

Supported via `grassmann.datasets`:

- **NeRFies / HyperNeRF** — drop scene directories under `data/nerfies/<scene>/`.
- **DyCheck iPhone** — drop scenes under `data/dycheck/<scene>/`.

Both expose the same `MonocularDataset` interface (cameras-per-frame, frame
loader, normalized times in [0, 1], optional COLMAP point cloud +
per-point observability).

## Training — local

The canonical recipe (current best on the slice-banana scene):

```bash
python scripts/train_mono.py \
  --dataset nerfies \
  --scene_dir data/nerfies/slice-banana \
  --output_dir checkpoints/slice-banana \
  --num_iters 14000 \
  --image_scale 4 \
  --init_strategy spatial_slice \
  --clamp_mode soft \
  --sigma_init_sq 0.02 \
  --grassmann_relax_start 1000 --grassmann_relax_end 8000 \
  --structural_kind ssim --lambda_structural 0.2 \
  --max_aspect_ratio 1000000 \
  --random_background \
  --sh_degree 3 \
  --lr_decay 0.01 \
  --densify_every 200 --densify_start 500 --densify_stop 10000 \
  --grad_threshold 1e-5 --spatial_split_threshold 0.5 \
  --opacity_prune_threshold 1e-3 --scale_min_prune 5e-3 \
  --split_anisotropic_shrink \
  --temporal_split_threshold 0.1 \
  --lambda_frob 1e-4 \
  --opacity_reset_every 3000 \
  --use_fast_rasterizer
```

Recipe highlights:

- `--init_strategy spatial_slice` + `--clamp_mode soft`: start in the
  static-3DGS regime and let n tilt into a dynamic disk via the
  Prop 5.3 bridge of the v7 spec.
- `--split_anisotropic_shrink`: on split, shrink L_raw only along the
  major axis. Avoids the cascading-small-Gaussian pathology of
  isotropic /φ shrink.
- `--max_aspect_ratio 1000000`: effectively uncapped in-plane aspect.
  `--structural_kind ssim`: 1-SSIM (DSSIM) structural loss. Together
  these match the strongest mono baseline observed.

Quality numbers from this recipe on slice-banana (14k iters, image
scale 4): val PSNR around 24.5 dB, val LPIPS 0.41, walltime around
300 s on an L4.

## Training — Modal (L4)

```bash
modal volume create gs-mono
modal volume put gs-mono ./data/nerfies/slice-banana /slice-banana

modal run scripts/train_modal.py --cmd train \
  --dataset nerfies --scene slice-banana \
  --iters 14000 --init-strategy spatial-slice ...
```

`train_modal.py` is a thin wrapper around `train_mono.py` that prepares
the Modal image, mounts the `gs-mono` and `gs-checkpoints` volumes, and
shells out to the training script.

## Rendering / evaluation

- `scripts/render_mono.py` — render arbitrary frames from a saved checkpoint.
- `scripts/eval_apples.py` — apples-to-apples PSNR/SSIM/LPIPS against a
  pre-rendered GT directory.
- `scripts/eval_per_frame.py` — Modal-side full-pipeline eval (renders
  every train+val frame through the CUDA rasterizer + reports per-frame
  metrics).
- `scripts/collate_eval.py` — collate `*.json` summaries across runs
  into one markdown table.

For independent baselines (Deformable3DGS, Yang 4DGS) see the
comparison scripts under `scripts/bugF_vs_d3dgs_modal.py`,
`scripts/quadtych_compare_modal.py`,
`scripts/eval_yang_apples.py`, and the patches under
`scripts/yang_4dgs_patches/`. The "bugF" prefix in those filenames is a
historical checkpoint label.

## Tests

```bash
pytest tests/ -q
```

The active suite covers the surviving (3-plane G(3,4)) paths: numerical
correctness of `compute_derived` / `condition_on_time`, the projection
Jacobian against autograd, dataset loaders, and a numerical-cliff
fuzzer for the Jacobian (`scripts/stress_test_jacobian.py`).

## Agent contributors

Some changes in the git history were authored together with Claude
Code. Agent-team rules and conventions are in `AGENTS.md` (read by
Claude Code's harness).
