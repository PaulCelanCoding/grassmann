# results/

Experimental artifacts and root-cause analyses. Library code lives in `grassmann/`; this directory contains the *findings* produced by running it.

## Layout

- `rca/` — Root-cause analyses on the monocular branch (slice-banana, NeRFies/HyperNeRF format).
  - `*.md` — narrative RCA reports
  - `figures/` — plots (FFT spectrum, size distributions, per-frame PSNR curves)
  - `heatmaps/` — per-frame diff heatmaps
  - `perframe/` — per-frame PSNR/L1 JSON tables (one per probe / baseline)

## Reading order (monocular branch)

1. `rca/streak_collapse.md` — earliest RCA: time-normalization bug + ray-init pathology under the legacy 2-plane parameterization.
2. `rca/monocular_streak_and_density_control.md` — design notes that motivated the pivot to the 3-plane (G(3,4)) projector.
3. `rca/3plane_low_psnr.md` — initial 22 dB PSNR ceiling on the 3-plane parameterization.
4. `rca/phaseC_vs_d3dgs.md` / `phaseC_vs_d3dgs_scale8.md` — first apples-to-apples gap to Deformable3DGS.
5. `rca/phaseC_3db_gap.md` — full attribution of the 3.24 dB gap; §1–§9d trace SH3 / iter budget / LR-decay / floor-vs-densification mechanism; §10 Yang 4DGS counter-baseline.
6. `rca/phaseC_14k_lrdecay_probes.md` — 14k iso-iter probe slate confirming the residual is not closeable via single-CLI-flag levers.

## Reproducing eval

All apples-to-apples PSNR numbers were computed against the saved D3DGS GT directory (`/tmp/d3dgs_gt/gt/`, 82 frames at 134×240) using:

- `scripts/eval_apples.py` — pairs ours' `render_frame{F:04d}.png` against `{j:05d}.png` (deformable_interp val split, ids[2::4]).
- `scripts/eval_yang_apples.py` — same, but maps Yang's per-frame item-id naming.
- `scripts/rca_diagnostic.py` — full RCA pipeline (renders → per-frame metrics → heatmaps → report).
- `scripts/rca_spectral.py` — spectral / per-Gaussian distribution analyses on a checkpoint.

## Caveats

The aggregate numbers in `phaseC_3db_gap.md` carry two unflagged confounds — see §10 review notes:
- Train-resolution asymmetry: ours and Yang train at scale 4 (268×480) and downsample renders to scale 8 for eval; D3DGS trained at the eval resolution. Bilinear-downsample tends to lift PSNR ~0.1–0.3 dB.
- Yang count imbalance (2.87 M vs ours 38 k) confounds the "representation cost" interpretation.
