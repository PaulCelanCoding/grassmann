# Grassmann GS — Grassmannian Splatting on Monocular Video

Implementation of a Grassmannian splatting pipeline for monocular video
reconstruction. Each primitive is parameterized by a 3-dimensional plane
in R^4 (a point on the Grassmannian G(3, 4)) plus a covariance factor
over that plane, so it represents a thin "disk" in space–time that
becomes a moving 2D splat under perspective projection.

Math spec: [`docs/maths/grassmanian_gradients_v7.md`](docs/maths/grassmanian_gradients_v7.md).

## Layout

```
grassmann/         # library (geometry, rasterization, training, density control)
  datasets/        # NeRFies + DyCheck loaders (monocular video)
scripts/           # train + render + eval entry points
  comparison/      # Deformable3DGS and Yang 4DGS baseline drivers
tests/             # pytest suite
docs/maths/        # math spec (v7)
legacy/            # archived reference code (older parameterizations,
                   # triangulation, synthetic-scene visualizers)
```

## Install

For development and tests:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Training runs on Modal (CUDA) and uses an image defined inside
`scripts/train_modal.py` — no separate local CUDA install is required.

## Datasets

Supported via `grassmann.datasets`:

- **NeRFies / HyperNeRF** — drop scene directories under `data/nerfies/<scene>/`.
- **DyCheck iPhone** — drop scenes under `data/dycheck/<scene>/`.

Both expose the same `MonocularDataset` interface (one camera per frame,
lazy frame loader, normalized times in [0, 1], a COLMAP point cloud, and
per-point observability).

## Training

Training runs on [Modal](https://modal.com/) (L4 GPU). One-time setup:

```bash
pip install modal           # already in requirements.txt
modal token new             # opens a browser to mint API tokens; writes ~/.modal.toml
```

Modal credits are billed against the account that owns the token. The
training image is defined inside `scripts/train_modal.py`; the first
`modal run` will build it (a few minutes), subsequent runs reuse the
cached image. Two named volumes hold persistent state — create them
once and upload one scene:

```bash
modal volume create gs-mono
modal volume create gs-checkpoints
modal volume put gs-mono ./data/nerfies/slice-banana /slice-banana
```

Canonical recipe (current best on the slice-banana scene, 14k iters at
image scale 2):

```bash
modal run scripts/train_modal.py --cmd train \
  --dataset nerfies --scene slice-banana \
  --iters 14000 \
  --sigma-init-sq 0.02 \
  --grassmann-relax-start 1000 --grassmann-relax-end 8000 \
  --lambda-structural 0.2 \
  --max-aspect-ratio 1000000 \
  --random-background \
  --sh-degree 3 \
  --lr-decay 0.01 \
  --densify-every 200 --densify-start 500 --densify-stop 10000 \
  --grad-threshold 1e-5 --spatial-split-threshold 0.5 \
  --opacity-prune-threshold 1e-3 --scale-min-prune 5e-3 \
  --split-anisotropic-shrink \
  --temporal-split-threshold 0.1 \
  --lambda-frob 1e-4 \
  --opacity-reset-every 3000
```

Recipe highlights:

- Init places n = e_0 (every Gaussian's plane is the {t = 0} spatial
  slice), so the model starts in a static-3DGS regime. The
  `grassmann_relax_*` ramp keeps lr_n small early and unlocks the n
  tilt over the next several thousand iters. The Schur step uses the
  v7-doc Prop 5.3 soft clamp √(Σ_tt² + ε²), which makes that
  static-to-dynamic transition C^∞-smooth.
- `--split_anisotropic_shrink`: on split, shrink L_raw only along the
  axis being split. The default isotropic /φ shrinks all three
  eigenvalues per split, so cascading splits collapse Gaussians.
- `--max_aspect_ratio 1000000` is effectively uncapped in-plane aspect.
  Combined with the 1-SSIM (DSSIM) structural loss it matches the
  strongest mono baseline observed.

Reference numbers on slice-banana with this recipe (14k iters, scale 2):
val PSNR around 24.5 dB, val LPIPS ≈ 0.41, walltime ≈ 5 min on an L4.

`scripts/train_modal.py` builds the CUDA image, mounts the gs-mono and
gs-checkpoints volumes, and inside the container shells out to
`scripts/train_mono.py` — that script is the actual training driver and
holds the full CLI surface.

## Rendering / evaluation

```bash
modal run scripts/train_modal.py --cmd render \
  --dataset nerfies --scene slice-banana \
  --ckpt <run-dir>/trained_nerfies.pt --frames 0,50,100

modal run scripts/train_modal.py --cmd eval_per_frame \
  --dataset nerfies --scene slice-banana \
  --ckpt <run-dir>/trained_nerfies.pt
```

Local helpers (operate on already-rendered images):

- `scripts/eval_apples.py` — PSNR/SSIM/LPIPS against a pre-rendered GT directory.
- `scripts/collate_eval.py` — collate per-run `*.json` summaries into a markdown table.

Baselines for comparison (Deformable3DGS, Yang 4DGS) live under
`scripts/comparison/`.

## Tests

```bash
pytest tests/ -q
```

The active suite covers the 3-plane G(3, 4) path: numerical correctness
of `compute_derived` / `condition_on_time`, and the dataset loaders.

## Agent contributors

Some changes in the git history were authored together with Claude Code.
Agent-team rules and conventions for assisted edits live in `AGENTS.md`.
