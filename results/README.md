# results/

Experimental artifacts and root-cause analyses. Library code lives in `grassmann/`; this directory contains the *findings* produced by running it.

## Layout

- `rca/` — Root-cause analyses on the monocular branch (slice-banana, NeRFies/HyperNeRF format).
  - `*.md` — narrative RCA reports
  - `figures/` — plots (FFT spectrum, size distributions, per-frame PSNR curves)
  - `heatmaps/` — per-frame diff heatmaps
  - `perframe/` — per-frame PSNR/L1 JSON tables (one per probe / baseline)

## Reading order (monocular)

The reports trace the empirical path of the monocular pivot
chronologically. Each is self-contained and can be read in isolation
once you know the baseline (3-plane G(3,4) projector form, slice-banana
scene in NeRFies/HyperNeRF format).

1. `rca/streak_collapse.md` — earliest investigation: time-normalization
   bug + ray-init pathology under the legacy 2-plane parameterization.
2. `rca/monocular_streak_and_density_control.md` — design notes that
   motivated the pivot to the 3-plane (G(3,4)) projector.
3. `rca/3plane_low_psnr.md` — initial 22 dB PSNR ceiling on the
   3-plane parameterization, and why.
4. `rca/phaseC_vs_d3dgs.md` and `rca/phaseC_vs_d3dgs_scale8.md` —
   first apples-to-apples gap measurement against Deformable3DGS.
5. `rca/phaseC_3db_gap.md` — full attribution of the 3.24 dB gap;
   Secs. 1–9d trace SH degree / iter budget / LR-decay / floor-vs-
   densification mechanism. Sec. 10 adds the Yang 4DGS counter-baseline.
6. `rca/phaseC_14k_lrdecay_probes.md` — 14k iso-iter probe slate
   showing the residual is not closeable via any single-flag lever.
7. `rca/blur_rca.md` and `rca/blur_rca/uncapped_ssim_lpips/` —
   identification of the in-plane-aspect cap as the dominant residual
   driver. Uncapping aspect + switching to DSSIM closes 88 % of the
   remaining LPIPS gap and defines the canonical recipe in
   [`README.md`](../README.md).

## Reproducing the eval

All apples-to-apples PSNR / SSIM / LPIPS numbers in these reports were
computed against the saved Deformable3DGS GT directory
(`/tmp/d3dgs_gt/gt/`, 82 frames at 134×240) using the surviving eval
scripts under `scripts/`:

- `scripts/eval_apples.py` — pairs `render_frame{F:04d}.png` against
  `{j:05d}.png` (deformable_interp val split, ids[2::4]).
- `scripts/eval_per_frame.py` — Modal-side full-pipeline eval (renders
  each train+val frame, computes per-frame PSNR/SSIM/LPIPS, emits the
  `perframe_*.json` tables).
- `scripts/collate_eval.py` — combine multiple summary JSONs into one
  markdown comparison table.
- `scripts/eval_yang_apples.py` — same as `eval_apples.py` but for the
  Yang 4DGS per-frame item-id naming.

The bug-tracker filename prefix `bugF_*` in the Modal scripts
(`bugF_vs_d3dgs_modal.py`, `bugF_via_d3dgs_metrics_modal.py`) refers to
the saved-checkpoint label and is preserved so the on-disk artifacts
round-trip.

## Caveats

Two unflagged confounds in the aggregate `phaseC_3db_gap.md` numbers
(see its Sec. 10 review notes):

- Train-resolution asymmetry: ours and Yang train at scale 4
  (268×480) and downsample renders to scale 8 for eval; D3DGS trains
  at the eval resolution. Bilinear-downsample tends to lift PSNR
  0.1–0.3 dB.
- Gaussian-count imbalance (Yang 2.87 M vs ours 38 k) confounds the
  "representation cost" interpretation.
