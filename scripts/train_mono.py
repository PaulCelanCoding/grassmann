"""
Local entrypoint for monocular training (NeRFies / DyCheck) under the 3-plane
projector parameterization (Phase A).

Usage (from repo root):
  python scripts/train_mono.py \
      --dataset nerfies \
      --scene_dir data/nerfies/<scene> \
      --num_iters 5000 \
      --output_dir checkpoints/<scene>

Loads a MonocularDataset, initializes the per-frame Gaussian model from the
scene's point cloud + observability, and runs the standard Trainer in
monocular sampling mode.

Density control is disabled by default (Phase A: --densify_every defaults
to 0 because the legacy DC targeted the 2-plane param and is incompatible
with the new param; see the plan for Phase C re-introduction).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/train_mono.py ...` from the repo root without installing
# the package. Modal sets this up via add_local_python_source; here we mirror it.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from grassmann.datasets.dycheck import load_dycheck
from grassmann.datasets.nerfies import load_nerfies
from grassmann.density_control import DensityConfig
from grassmann.fast_rasterizer import FastRasterConfig
from grassmann.surfel_rasterizer import SurfelRasterConfig
from grassmann.initialization import init_gaussians_from_points
from grassmann.trainable import trainable_from_params
from grassmann.training import Trainer, TrainerConfig


DTYPE = torch.float32


def _load_dataset(
    name: str,
    scene_dir: Path,
    image_scale: int,
    split: str | None,
    allow_distortion: bool,
):
    if name == "nerfies":
        if split is not None:
            print(f"  [warning] --split is ignored for --dataset nerfies (no splits/ dir)")
        return load_nerfies(scene_dir, image_scale=image_scale,
                             allow_distortion=allow_distortion)
    if name == "dycheck":
        return load_dycheck(scene_dir, image_scale=image_scale, split_name=split,
                             allow_distortion=allow_distortion)
    raise ValueError(f"Unknown --dataset {name!r}; expected nerfies|dycheck")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("nerfies", "dycheck"), required=True)
    ap.add_argument("--scene_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, default=None,
                    help="Where to write the trained checkpoint. Defaults to scene_dir.")
    ap.add_argument("--image_scale", type=int, default=4)
    ap.add_argument("--split", type=str, default=None,
                    help="DyCheck split name (e.g. 'train', 'common'). Ignored for nerfies.")
    ap.add_argument("--init_strategy",
                    choices=("random", "spatial_slice"),
                    default="random",
                    help="'random' (legacy): n ~ Uniform(S^3), L_raw small isotropic. "
                         "'spatial_slice' (v7-doc §7.2): n = e_0 for every Gaussian, "
                         "starting in the static-3DGS regime; tilts emerge during training "
                         "via the bridge of Prop 5.3 (requires --clamp_mode=soft).")
    ap.add_argument("--num_iters", type=int, default=5000)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--use_fast_rasterizer", action="store_true")
    ap.add_argument("--sigma_3d_blur", type=float, default=1e-4,
                    help="Isotropic numerical lift (ε I) added to Σ_3D(t_0) before "
                         "feeding to the CUDA rasterizer. Σ_3D(t_0) is rank-2 under the "
                         "3-plane parameterization; the EWA needs an invertible 3x3, so "
                         "ε≈1e-4 (about 1cm² in scene units) is enough. Larger values "
                         "become a meaningful blur.")
    ap.add_argument("--sigma_init_sq", type=float, default=0.02,
                    help="In-plane init covariance scale σ²_init. L_raw entries ~ "
                         "N(0, σ²_init / 3) so E[L_raw L_raw^T] ≈ σ²_init · I_4. "
                         "After projection and Schur on time, Σ_3D(t_0) has eigenvalues "
                         "~ σ²_init in two of its three directions (the disk).")
    ap.add_argument("--device", type=str, default=None,
                    help="cpu | cuda. Defaults to cuda if available.")
    ap.add_argument("--allow_distortion", action="store_true",
                    help="Treat scenes with non-zero radial/tangential distortion "
                         "as pinhole. Geometry is approximate -- smoke runs only.")
    ap.add_argument("--densify_every", type=int, default=0,
                    help="Density-control event frequency. 0 disables. Phase C: "
                         "200 is a reasonable default; standard 3DGS uses 100.")
    ap.add_argument("--densify_start", type=int, default=500,
                    help="Earliest iter at which DC events fire. Lets the model "
                         "settle before mutating the Gaussian set.")
    ap.add_argument("--densify_stop", type=int, default=10000,
                    help="Latest iter at which DC events fire. Past this point "
                         "training is fixed-N to allow convergence.")
    ap.add_argument("--grad_threshold", type=float, default=2e-4,
                    help="Screen-space ‖∇μ_2d‖ threshold above which a Gaussian "
                         "is 'stressed' and eligible for split.")
    ap.add_argument("--spatial_split_threshold", type=float, default=0.5,
                    help="λ_max(Σ_3D) above which a stressed Gaussian SPLITS. "
                         "In scene units²; 0.5 ≈ (0.7m std) at scene scale 1m.")
    ap.add_argument("--max_split_per_event", type=int, default=0,
                    help="Cap on #splits per DC cycle (0 = unlimited). Useful "
                         "to prevent N-explosion in early iters.")
    ap.add_argument("--opacity_prune_threshold", type=float, default=1e-3,
                    help="Prune Gaussian if sigmoid(opacity_logit) < this. More "
                         "conservative than standard 3DGS (0.005).")
    ap.add_argument("--scale_min_prune", type=float, default=1e-6,
                    help="Prune Gaussian if λ_min(Σ_3D) < this (collapsed disk).")
    ap.add_argument("--scale_max_prune", type=float, default=100.0,
                    help="Prune Gaussian if λ_max(Σ_3D) > this (runaway disk).")
    ap.add_argument("--seed", type=int, default=None,
                    help="Optional seed for the random initialization (n, L_raw).")
    ap.add_argument("--static_baseline", action="store_true",
                    help="Disable time conditioning (Schur step skipped, w_t=1 always). "
                         "Establishes the static-3DGS-on-monocular-bundle floor within the "
                         "same pipeline; gap to the full temporal run = value of time conditioning.")
    ap.add_argument("--val_stride", type=int, default=4,
                    help="If the loaded dataset has no val_ids (e.g. HyperNeRF interp split "
                         "ships with val_ids=[]), construct a held-out split by taking every "
                         "Nth frame. DyGauBench convention for HyperNeRF interp = 4 (248 train "
                         "/ 82 val for 330-frame slice-banana). 0 disables val-split injection.")
    ap.add_argument("--split_convention", choices=("val_stride", "deformable_interp"),
                    default="val_stride",
                    help="How to construct train/test split when dataset ships val_ids=[]. "
                         "'val_stride' = every val_stride-th frame is val, rest is train "
                         "(247/83 for stride 4). 'deformable_interp' = ids[::4] is train, "
                         "ids[2::4] is val (83/82 -- matches Deformable3DGS HyperNeRF interp "
                         "convention; needed for iso-iter comparisons against published numbers).")
    ap.add_argument("--init_points_multiplier", type=int, default=1,
                    help="Replicate the init point cloud K times with small random "
                         "perturbations (~0.01 in scene units) before constructing Gaussians. "
                         "Used by the capacity-vs-motion diagnostic: if doubling/quadrupling N "
                         "lifts the PSNR ceiling, the limiter is capacity, not the motion "
                         "model. K=1 (default) is the original cloud.")
    ap.add_argument("--diag_single_frame", type=int, default=-1,
                    help="Capacity-vs-motion diagnostic 2: train and validate on a single "
                         "frame index (no time variation). Implies --static_baseline. "
                         "Final val PSNR = the ceiling for static 3DGS on this image at "
                         "current scale + N. If our full temporal run on the same frame "
                         "is much lower, the gap = motion residuals. -1 disables.")
    ap.add_argument("--lambda_frob", type=float, default=0.0,
                    help="Frobenius-norm penalty on L_raw (Phase-A-correctness). "
                         "Recommended ≈1e-4. Targets the 7%% rank-1-collapse pathology where "
                         "the optimizer routes capacity into n̂ (projector null-direction).")
    ap.add_argument("--opacity_reset_every", type=int, default=0,
                    help="Periodic opacity-logit reset (Phase-A-correctness). Every N "
                         "iters, opacity_logit -> opacity_reset_logit; Adam state for "
                         "opacity_logit is also zeroed. 0 disables. Standard 3DGS uses "
                         "3000. Targets the 32%%-dead pathology.")
    ap.add_argument("--opacity_reset_logit", type=float, default=-5.0,
                    help="Target opacity_logit at reset. -5 -> sigmoid(-5)≈0.007.")
    ap.add_argument("--lambda_aniso", type=float, default=0.0,
                    help="Bounded anisotropy penalty on Σ_3D(t_0). Trims the runaway "
                         "λ_max/λ_min tail. 0 disables. Recommended ≈1e-3 (small).")
    ap.add_argument("--lr_decay", type=float, default=1.0,
                    help="Log-linear LR decay factor for geometric params (n, mu, "
                         "L_raw). 1.0 disables. <1 decays from base*1 to base*lr_decay "
                         "over num_iters via lr(t)=base*lr_decay**t. 3DGS uses 0.01 "
                         "(100x decay) over 30k iters.")
    ap.add_argument("--lr_pos_scale", type=float, default=1.0,
                    help="Multiplier on (lr_n, lr_mu, lr_L) initial values. 1.0 keeps "
                         "defaults (1e-3, 5e-3, 5e-3). 0.2 -> 5x smaller; closer to "
                         "D3DGS position_lr_init magnitude. Color/opacity LRs unaffected.")
    ap.add_argument("--lambda_structural", type=float, default=0.2,
                    help="Weight on the structural loss (local-mean + local-variance "
                         "matching with a 7x7 box filter, see grassmann.losses). 0.2 "
                         "matches the historical L1=0.8/structural=0.2 split. 0.0 -> "
                         "pure L1.")
    ap.add_argument("--structural_kind", choices=("boxstats", "ssim"),
                    default="boxstats",
                    help="Choice of structural-loss term. 'boxstats' is the legacy "
                         "7x7 local-mean+var matcher (default, historical). 'ssim' is "
                         "1 - SSIM with a Gaussian window — matches the 3DGS DSSIM "
                         "structural term.")
    ap.add_argument("--clamp_mode", choices=("hard", "soft"), default="hard",
                    help="v7-doc §5.1 Schur denominator clamp. 'hard' (legacy): "
                         "max(Σ_tt, eps), discontinuous gradient. 'soft' (v7-doc): "
                         "√(Σ_tt² + eps²), C^∞-smooth — required for the n=e_0 "
                         "init bridge of Prop 5.3.")
    ap.add_argument("--eps_schur", type=float, default=-1.0,
                    help="Schur denominator floor. -1 (default) auto-selects: "
                         "1e-20 for clamp_mode=hard, 1e-8 for clamp_mode=soft "
                         "(v7-doc default). Override with explicit value if needed.")
    ap.add_argument("--mu_lr_split", action="store_true",
                    help="v7-doc §7.5: split μ into mu_time and mu_spatial as "
                         "two parameters with separate LRs (--lr_mu_time / "
                         "--lr_mu_spatial). Default off keeps single μ + lr_mu.")
    ap.add_argument("--lr_mu_spatial", type=float, default=1e-4,
                    help="LR for mu_spatial when --mu_lr_split (v7-doc default 1e-4).")
    ap.add_argument("--lr_mu_time", type=float, default=1e-3,
                    help="LR for mu_time when --mu_lr_split (v7-doc default 1e-3).")
    ap.add_argument("--mu_constraint",
                    choices=("free", "project", "reparam", "penalty"),
                    default="free",
                    help="μ-DOF probe (results/rca/mu_dof_ab_test.md). 'free' "
                         "(legacy): mu is unconstrained in R^4 (~4 effective DOF). "
                         "'project': compute_derived projects mu onto n^⊥ before "
                         "the time-split (3 DOF, hard constraint). 'reparam': "
                         "TrainableGaussians.forward() returns the projected mu "
                         "(same math as 'project', projection happens upstream). "
                         "'penalty': mu free + soft loss term λ·<n,μ>² with "
                         "λ=--lambda_mu_penalty.")
    ap.add_argument("--lambda_mu_penalty", type=float, default=1.0,
                    help="Strength of the soft <n,μ>² penalty when "
                         "--mu_constraint=penalty. 1.0 default; ignored otherwise.")
    # 2DGS A/B (results/rca/surfel_rasterizer_ab.md): swap diff-gaussian for
    # diff-surfel, optionally enable depth-distortion + normal-consistency.
    ap.add_argument("--rasterizer", choices=("gaussian", "surfel"), default="gaussian",
                    help="'gaussian' = Inria diff_gaussian_rasterization (current default, "
                         "needs σ_lift² rank-2→rank-3 lift). 'surfel' = Huang2024 "
                         "diff_surfel_rasterization (native rank-2 disk, no lift).")
    ap.add_argument("--use_2dgs_losses", action="store_true",
                    help="Enable 2DGS depth-distortion + normal-consistency regularizers. "
                         "Only meaningful with --rasterizer surfel. Schedule: dist@3000, normal@7000.")
    ap.add_argument("--lambda_normal", type=float, default=0.05,
                    help="2DGS normal-consistency lambda (paper default 0.05).")
    ap.add_argument("--lambda_dist", type=float, default=100.0,
                    help="2DGS depth-distortion lambda (paper: 100 indoor/object, 1000 unbounded).")
    ap.add_argument("--surfel_eigval_floor", type=float, default=1e-6,
                    help="Floor on smallest eigenvalue of Σ_3D(t₀) before sqrt; "
                         "stabilizes eigh backward at exact rank-2 degeneracy.")
    ap.add_argument("--surfel_sigma_3d_blur", type=float, default=0.0,
                    help="Optional pre-eigh σ_lift² lift in the surfel path; "
                         "tests whether σ_lift² acts as a hidden training regularizer "
                         "in the gaussian path. 0=honest rank-2 (default); 1e-4 matches A1.")
    ap.add_argument("--surfel_eigh_jitter", type=float, default=0.0,
                    help="Anisotropic random jitter on Σ_3D before eigh. Breaks "
                         "1/Δλ degeneracy in eigh backward at near-degenerate "
                         "in-plane eigvals (~14% of Gaussians by p99 anisotropy). "
                         "1e-5 to 1e-3 are reasonable.")
    ap.add_argument("--sh_degree", type=int, default=0,
                    help="Spherical-harmonics band for per-Gaussian color. 0 keeps the "
                         "legacy constant-RGB path (color_logit). >0 swaps in sh_dc + "
                         "sh_rest with K=(degree+1)^2 coefficients/channel; the CUDA "
                         "rasterizer evaluates SH against the per-Gaussian view direction. "
                         "Standard 3DGS default is 3 (16 coeffs/channel). Requires "
                         "--use_fast_rasterizer; the toy CPU rasterizer collapses to the "
                         "DC term.")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.dataset} scene from {args.scene_dir} (image_scale={args.image_scale})")
    ds = _load_dataset(
        args.dataset, args.scene_dir, args.image_scale, args.split,
        allow_distortion=args.allow_distortion,
    )
    print(f"  T={ds.T}, points={ds.N_points}, H={ds.H}, W={ds.W}, "
          f"train={len(ds.train_indices)}, val={len(ds.val_indices)}")

    # If the dataset ships with no val split (HyperNeRF interp ships with
    # val_ids=[]), construct one. Two conventions supported:
    #   * val_stride: every val_stride-th frame is val (rest is train).
    #     DyGauBench convention for HyperNeRF interp -> stride 4 (247/83).
    #   * deformable_interp: train = ids[::4], val = ids[2::4]. Matches
    #     Deformable3DGS scene/dataset_readers.py (interp branch). 83/82 for
    #     330-frame slice-banana. Use this for iso-iter comparison vs.
    #     Deformable3DGS / published interp numbers.
    val_indices_override = None
    train_indices_override = None
    if args.diag_single_frame >= 0:
        k = args.diag_single_frame
        if not (0 <= k < ds.T):
            raise SystemExit(f"--diag_single_frame {k} out of range [0, {ds.T})")
        train_indices_override = [k]
        val_indices_override = [k]
        args.static_baseline = True
        print(f"  [diag_single_frame={k}] training and validating on a single frame; "
              f"static_baseline forced ON (single-frame fit, no time variation).")
    elif args.val_stride > 0 and len(ds.val_indices) == 0:
        if args.split_convention == "deformable_interp":
            train_indices_override = list(range(0, ds.T, 4))            # ids[::4]
            val_indices_override = list(range(2, ds.T, 4))              # ids[2::4]
            print(f"  [split=deformable_interp] train={len(train_indices_override)}, "
                  f"val={len(val_indices_override)} (Deformable3DGS HyperNeRF interp split)")
        else:
            held_out = list(range(0, ds.T, args.val_stride))
            val_indices_override = held_out
            train_indices_override = [i for i in range(ds.T) if i not in set(held_out)]
            print(f"  [val_stride={args.val_stride}] dataset ships no val split; "
                  f"constructing held-out: train={len(train_indices_override)}, "
                  f"val={len(held_out)}")

    # Initialize one Gaussian per scene point. Pick a temporal mean per point
    # from the median-observed frame's normalized time (sensible for static
    # background; dynamic objects will be re-organized by training).
    times_for_init = []
    for obs in ds.observability:
        if obs:
            t_idx = obs[len(obs) // 2]
        else:
            t_idx = ds.T // 2
        times_for_init.append(float(ds.times[t_idx]))
    times_init = torch.tensor(times_for_init, dtype=torch.float64)

    # Capacity diagnostic: replicate the point cloud K times with small noise.
    points = ds.points3D
    times_used = times_init
    obs_used = ds.observability
    if args.init_points_multiplier > 1:
        K = args.init_points_multiplier
        rng = torch.Generator(); rng.manual_seed(args.seed if args.seed is not None else 0)
        repeated_points = points.repeat(K, 1)
        # Perturbation scale: 1% of scene bbox extent.
        bbox_extent = (points.max(0).values - points.min(0).values).norm().item()
        noise_scale = 0.01 * bbox_extent
        noise = torch.randn(repeated_points.shape, dtype=points.dtype,
                            generator=rng) * noise_scale
        # Don't perturb the original copy (first N rows).
        noise[:points.shape[0]] = 0
        points = repeated_points + noise
        times_used = times_init.repeat(K)
        obs_used = obs_used * K  # each replica reuses the same observability list
        print(f"  [init_points_multiplier={K}] cloud replicated: "
              f"N={points.shape[0]} (was {ds.N_points}), perturb scale={noise_scale:.4f}")
    print(f"  Initializing {points.shape[0]} Gaussians (strategy={args.init_strategy})...")

    params = init_gaussians_from_points(
        points,
        times_used,
        ds.cameras_per_frame,
        strategy=args.init_strategy,
        observability=obs_used,
        sigma_init_sq=args.sigma_init_sq,
        opacity=0.5,
        sigma_k_pixel=1.0,
        sigma_k_temporal=0.0,
        seed=args.seed,
    )
    eps_schur_resolved = (
        args.eps_schur if args.eps_schur > 0
        else (1e-8 if args.clamp_mode == "soft" else 1e-20)
    )
    model = trainable_from_params(
        params, dtype=DTYPE, device=device, sh_degree=args.sh_degree,
        mu_constraint=args.mu_constraint,
        mu_lr_split=args.mu_lr_split,
        clamp_mode=args.clamp_mode,
        eps_schur=eps_schur_resolved,
    )
    if args.sh_degree > 0:
        print(f"  Model: {model.N} Gaussians on {device} (SH degree {args.sh_degree}, "
              f"{(args.sh_degree + 1) ** 2} coeffs/channel)")
    else:
        print(f"  Model: {model.N} Gaussians on {device}")

    cfg_kwargs = dict(
        num_iters=args.num_iters,
        log_every=args.log_every,
        lambda_l1=0.8,
        lambda_structural=args.lambda_structural,
        structural_kind=args.structural_kind,
        lr_n=1e-3 * args.lr_pos_scale,
        lr_mu=5e-3 * args.lr_pos_scale,
        lr_L=5e-3 * args.lr_pos_scale,
        lr_mu_spatial=args.lr_mu_spatial * args.lr_pos_scale,
        lr_mu_time=args.lr_mu_time * args.lr_pos_scale,
        lr_opacity=5e-2, lr_color=5e-2,
        background=torch.zeros(3, dtype=DTYPE, device=device),
        densify_every=args.densify_every,
        densify_start=args.densify_start,
        densify_stop=args.densify_stop,
        density_config=DensityConfig(
            grad_threshold=args.grad_threshold,
            spatial_split_threshold=args.spatial_split_threshold,
            max_split_per_event=args.max_split_per_event,
            opacity_threshold=args.opacity_prune_threshold,
            scale_min=args.scale_min_prune,
            scale_max=args.scale_max_prune,
        ),
        use_fast_rasterizer=args.use_fast_rasterizer,
        fast_raster_config=FastRasterConfig(
            sigma_3d_blur=args.sigma_3d_blur,
            sh_degree=args.sh_degree,
        ),
        rasterizer=args.rasterizer,
        surfel_raster_config=SurfelRasterConfig(
            sh_degree=args.sh_degree,
            eigval_floor=args.surfel_eigval_floor,
            sigma_3d_blur=args.surfel_sigma_3d_blur,
            eigh_jitter=args.surfel_eigh_jitter,
        ),
        use_2dgs_losses=args.use_2dgs_losses,
        lambda_normal=args.lambda_normal,
        lambda_dist=args.lambda_dist,
        validation_every=max(args.log_every, args.num_iters // 10),
        static_baseline=args.static_baseline,
        lambda_frob=args.lambda_frob,
        opacity_reset_every=args.opacity_reset_every,
        opacity_reset_logit=args.opacity_reset_logit,
        lambda_aniso=args.lambda_aniso,
        lr_decay=args.lr_decay,
        lambda_mu_penalty=(
            args.lambda_mu_penalty if args.mu_constraint == "penalty" else 0.0
        ),
    )
    if val_indices_override is not None:
        cfg_kwargs["validation_frames"] = val_indices_override
        cfg_kwargs["train_frames"] = train_indices_override
    config = TrainerConfig(**cfg_kwargs)
    trainer = Trainer.from_monocular_dataset(model, ds, config)

    print(f"Training for {args.num_iters} iterations...")
    history = trainer.train()

    out_dir = args.output_dir or args.scene_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"trained_{args.dataset}_{args.init_strategy}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
        "init_strategy": args.init_strategy,
        "dataset": args.dataset,
    }, out_path)
    print(f"Saved checkpoint to {out_path}")


if __name__ == "__main__":
    main()
