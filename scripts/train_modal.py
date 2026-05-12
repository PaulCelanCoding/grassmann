"""
Modal entry for monocular GPU training on L4 (Phase A: 3-plane projector).

Volumes (created on first run):
  gs-mono         /data          NeRFies / DyCheck scenes
  gs-checkpoints  /checkpoints   per-run output

One-time data upload (from repo root):
  modal volume create gs-mono
  modal volume put gs-mono ./data/nerfies/<scene>  /<scene>
  modal volume put gs-mono ./data/dycheck/<scene>  /<scene>

Usage:
  modal run scripts/train_modal.py --cmd smoke --dataset nerfies --scene <scene>
  modal run scripts/train_modal.py --cmd train --dataset nerfies --scene <scene> --iters 30000
  modal run scripts/train_modal.py --cmd train --dataset dycheck --scene <scene> --split train
  modal run scripts/train_modal.py --cmd render --dataset nerfies --scene <scene> \
      --ckpt nerfies-slice-banana-random-30000it/trained_nerfies_random.pt \
      --frames 0,50,100 --side-by-side

Density control is disabled in Phase A (the legacy DC targets the 2-plane
parameterization; see the plan, Phase C, for re-introduction). The
`init_strategy` and `sigma_3d_blur` flags are the only meaningful knobs
exposed here.
"""
import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
        add_python="3.11",
    )
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install(
        "numpy",
        "matplotlib",
        "pillow",
        "tqdm",
        "pytest",
        "lpips",
        "torchvision",
    )
    .pip_install(
        "git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git",
        gpu="L4",
    )
    .add_local_python_source("grassmann")
    .add_local_dir(str(REPO / "scripts"), remote_path="/root/scripts")
)

app = modal.App("grassmann-train", image=image)

mono_vol = modal.Volume.from_name("gs-mono", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("gs-checkpoints", create_if_missing=True)

VOLUMES = {"/data": mono_vol, "/checkpoints": ckpt_vol}
GPU = "L4"


def _run(argv: list[str]) -> None:
    import time as _time
    print(f">>> {' '.join(argv)}", flush=True)
    _t0 = _time.perf_counter()
    subprocess.run(argv, check=True, cwd="/root")
    _wall = _time.perf_counter() - _t0
    print(f"TRAIN_WALL_S={_wall:.1f}", flush=True)


def _ensure_scene_unpacked(scene: str) -> str:
    """Ensure /data/<scene> exists. If not, unzip /data/<scene>.zip onto the volume.

    Workaround for `modal volume put` being painfully slow when uploading many
    small files (we observed ~30+ min for 1500 files vs ~5 min for one zip).
    Standard NeRFies/HyperNeRF/DyCheck scenes have ~330 frames * (4 RGB scales
    + camera JSON + ...) -> easily into the thousands.

    Upload pattern:
        modal volume put gs-mono ./data/<dataset>/<scene>.zip /<scene>.zip
    """
    import os
    import zipfile

    scene_dir = f"/data/{scene}"
    if os.path.isdir(scene_dir):
        return scene_dir
    zip_path = f"/data/{scene}.zip"
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(
            f"Neither {scene_dir!r} (dir) nor {zip_path!r} (zip) exist on the gs-mono "
            f"volume. Upload one of them first."
        )
    print(f"  unpacking {zip_path} -> /data/...", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall("/data")
    if not os.path.isdir(scene_dir):
        # The zip's top-level dir name didn't match `scene`. Try to detect it.
        for entry in os.listdir("/data"):
            candidate = f"/data/{entry}"
            if os.path.isdir(candidate) and os.path.exists(f"{candidate}/dataset.json"):
                if entry != scene:
                    print(f"  [info] zip top-level was {entry!r}, not {scene!r}; "
                          f"using {candidate}", flush=True)
                return candidate
        raise RuntimeError(f"Unzipped {zip_path} but no scene dir found under /data/.")
    mono_vol.commit()
    return scene_dir


@app.function(gpu=GPU, volumes=VOLUMES, timeout=24 * 3600)
def train(
    dataset: str,
    scene: str,
    num_iters: int,
    image_scale: int,
    use_fast: bool,
    init_strategy: str,
    split: str | None,
    allow_distortion: bool,
    log_every: int,
    sigma_3d_blur: float,
    sigma_init_sq: float,
    run_tag: str,
    seed: int | None,
    static_baseline: bool,
    val_stride: int,
    split_convention: str,
    init_points_multiplier: int,
    init_per_frame_stride: int,
    diag_single_frame: int,
    lambda_frob: float,
    opacity_reset_every: int,
    opacity_reset_logit: float,
    lambda_aniso: float,
    densify_every: int,
    densify_start: int,
    densify_stop: int,
    grad_threshold: float,
    spatial_split_threshold: float,
    max_split_per_event: int,
    opacity_prune_threshold: float,
    scale_min_prune: float,
    scale_max_prune: float,
    mu_t_min: float,
    mu_t_max: float,
    sh_degree: int,
    lr_decay: float,
    lr_pos_scale: float,
    lambda_structural: float,
    structural_kind: str,
    mu_constraint: str,
    lambda_mu_penalty: float,
    clamp_mode: str,
    eps_schur: float,
    mu_lr_split: bool,
    lr_mu_spatial: float,
    lr_mu_time: float,
    init_points_path: str,
    init_colors_path: str,
    # --- Wave A probe flags (quality_knobs_evaluation.md) ---
    color_lr_warmup_iter: int,
    random_background: bool,
    sigma_init_knn_k: int,
    sigma_init_alpha_t: float,
    max_aspect_ratio: float,
    exposure_per_frame: bool,
    lambda_exposure_reg: float,
    temporal_split_threshold: float,
    lambda_time_coherence: float,
    time_coherence_dt: float,
    mip_filter_sigma_pixel: float,
    refine_poses: bool,
    lr_pose_rot: float,
    lr_pose_trans: float,
    pose_warmup_iter: int,
    lambda_depth: float,
    depth_model: str,
    grassmann_relax_start: int,
    grassmann_relax_end: int,
    floater_min_views: int,
    floater_eps: float,
    sh_degree_warmup_step: int,
    lambda_opacity_entropy: float,
    density_strategy: str,
    mcmc_noise_lr: float,
    mcmc_noise_after: int,
    mcmc_noise_gate_k: float,
    mcmc_noise_gate_thr: float,
    mcmc_max_relocations_per_step: int,
    split_anisotropic_shrink: bool,
    split_opacity_correction: bool,
    split_opacity_brighter: bool,
    split_shrink_factor: float,
    split_offset_sigmas: float,
    trigger_post_schur: bool,
    merge_every: int,
    merge_distance: float,
    merge_normal_cos: float,
    aspect_split_threshold: float,
    use_quadratic_motion: bool,
    lr_c2: float,
    use_s3_motion: bool,
    lr_omega: float,
    profile_breakdown: bool,
    profile_warmup_iters: int,
) -> None:
    scene_dir = _ensure_scene_unpacked(scene)
    suffix = f"-{run_tag}" if run_tag else ""
    out_dir = f"/checkpoints/{dataset}-{scene}-{init_strategy}-{num_iters}it{suffix}"
    argv = [
        "python", "scripts/train_mono.py",
        "--dataset", dataset,
        "--scene_dir", scene_dir,
        "--output_dir", out_dir,
        "--num_iters", str(num_iters),
        "--log_every", str(log_every),
        "--image_scale", str(image_scale),
        "--init_strategy", init_strategy,
        "--sigma_3d_blur", str(sigma_3d_blur),
        "--sigma_init_sq", str(sigma_init_sq),
    ]
    if split is not None:
        argv += ["--split", split]
    if use_fast:
        argv.append("--use_fast_rasterizer")
    if allow_distortion:
        argv.append("--allow_distortion")
    if seed is not None:
        argv += ["--seed", str(seed)]
    if static_baseline:
        argv.append("--static_baseline")
    argv += ["--val_stride", str(val_stride),
             "--split_convention", split_convention,
             "--init_points_multiplier", str(init_points_multiplier)]
    if init_per_frame_stride > 0:
        argv += ["--init_per_frame_stride", str(init_per_frame_stride)]
    if diag_single_frame >= 0:
        argv += ["--diag_single_frame", str(diag_single_frame)]
    if lambda_frob > 0.0:
        argv += ["--lambda_frob", str(lambda_frob)]
    if opacity_reset_every > 0:
        argv += ["--opacity_reset_every", str(opacity_reset_every)]
    if opacity_reset_logit != -5.0:
        argv += ["--opacity_reset_logit", str(opacity_reset_logit)]
    if lambda_aniso > 0.0:
        argv += ["--lambda_aniso", str(lambda_aniso)]
    if densify_every > 0:
        argv += ["--densify_every", str(densify_every),
                 "--densify_start", str(densify_start),
                 "--densify_stop", str(densify_stop),
                 "--grad_threshold", str(grad_threshold),
                 "--spatial_split_threshold", str(spatial_split_threshold),
                 "--opacity_prune_threshold", str(opacity_prune_threshold)]
        if scale_min_prune != 1e-6:
            argv += ["--scale_min_prune", str(scale_min_prune)]
        if scale_max_prune != 100.0:
            argv += ["--scale_max_prune", str(scale_max_prune)]
        if mu_t_min > -1e9:
            argv += ["--mu_t_min", str(mu_t_min)]
        if mu_t_max < 1e9:
            argv += ["--mu_t_max", str(mu_t_max)]
        if max_split_per_event > 0:
            argv += ["--max_split_per_event", str(max_split_per_event)]
    if sh_degree > 0:
        argv += ["--sh_degree", str(sh_degree)]
    if lr_decay != 1.0:
        argv += ["--lr_decay", str(lr_decay)]
    if lr_pos_scale != 1.0:
        argv += ["--lr_pos_scale", str(lr_pos_scale)]
    if lambda_structural != 0.2:
        argv += ["--lambda_structural", str(lambda_structural)]
    if structural_kind != "boxstats":
        argv += ["--structural_kind", structural_kind]
    if mu_constraint != "free":
        argv += ["--mu_constraint", mu_constraint]
    if lambda_mu_penalty != 1.0:
        argv += ["--lambda_mu_penalty", str(lambda_mu_penalty)]
    if clamp_mode != "hard":
        argv += ["--clamp_mode", clamp_mode]
    if eps_schur > 0:
        argv += ["--eps_schur", str(eps_schur)]
    if mu_lr_split:
        argv += ["--mu_lr_split",
                 "--lr_mu_spatial", str(lr_mu_spatial),
                 "--lr_mu_time", str(lr_mu_time)]
    if init_points_path:
        argv += ["--init_points_path", init_points_path]
    if init_colors_path:
        argv += ["--init_colors_path", init_colors_path]
    # --- Wave A probe flags ---
    if color_lr_warmup_iter > 0:
        argv += ["--color_lr_warmup_iter", str(color_lr_warmup_iter)]
    if random_background:
        argv += ["--random_background"]
    if sigma_init_knn_k > 0:
        argv += ["--sigma_init_knn_k", str(sigma_init_knn_k),
                 "--sigma_init_alpha_t", str(sigma_init_alpha_t)]
    if max_aspect_ratio > 0:
        argv += ["--max_aspect_ratio", str(max_aspect_ratio)]
    if exposure_per_frame:
        argv += ["--exposure_per_frame",
                 "--lambda_exposure_reg", str(lambda_exposure_reg)]
    if temporal_split_threshold > 0:
        argv += ["--temporal_split_threshold", str(temporal_split_threshold)]
    if lambda_time_coherence > 0:
        argv += ["--lambda_time_coherence", str(lambda_time_coherence),
                 "--time_coherence_dt", str(time_coherence_dt)]
    if mip_filter_sigma_pixel > 0:
        argv += ["--mip_filter_sigma_pixel", str(mip_filter_sigma_pixel)]
    if refine_poses:
        argv += ["--refine_poses",
                 "--lr_pose_rot", str(lr_pose_rot),
                 "--lr_pose_trans", str(lr_pose_trans),
                 "--pose_warmup_iter", str(pose_warmup_iter)]
    if lambda_depth > 0:
        argv += ["--lambda_depth", str(lambda_depth),
                 "--depth_model", depth_model]
    if grassmann_relax_end > 0:
        argv += ["--grassmann_relax_start", str(grassmann_relax_start),
                 "--grassmann_relax_end", str(grassmann_relax_end)]
    if floater_min_views > 0:
        argv += ["--floater_min_views", str(floater_min_views),
                 "--floater_eps", str(floater_eps)]
    if sh_degree_warmup_step > 0:
        argv += ["--sh_degree_warmup_step", str(sh_degree_warmup_step)]
    if lambda_opacity_entropy > 0:
        argv += ["--lambda_opacity_entropy", str(lambda_opacity_entropy)]
    if density_strategy != "heuristic":
        argv += ["--density_strategy", density_strategy]
    if mcmc_noise_lr > 0:
        argv += ["--mcmc_noise_lr", str(mcmc_noise_lr),
                 "--mcmc_noise_after", str(mcmc_noise_after),
                 "--mcmc_noise_gate_k", str(mcmc_noise_gate_k),
                 "--mcmc_noise_gate_thr", str(mcmc_noise_gate_thr)]
    if mcmc_max_relocations_per_step > 0:
        argv += ["--mcmc_max_relocations_per_step", str(mcmc_max_relocations_per_step)]
    if split_anisotropic_shrink:
        argv.append("--split_anisotropic_shrink")
    if split_opacity_correction:
        argv.append("--split_opacity_correction")
    if split_opacity_brighter:
        argv.append("--split_opacity_brighter")
    if split_shrink_factor != 1.6:
        argv += ["--split_shrink_factor", str(split_shrink_factor)]
    if split_offset_sigmas != 1.0:
        argv += ["--split_offset_sigmas", str(split_offset_sigmas)]
    if trigger_post_schur:
        argv.append("--trigger_post_schur")
    if merge_every > 0:
        argv += ["--merge_every", str(merge_every),
                 "--merge_distance", str(merge_distance),
                 "--merge_normal_cos", str(merge_normal_cos)]
    if aspect_split_threshold > 0:
        argv += ["--aspect_split_threshold", str(aspect_split_threshold)]
    if use_quadratic_motion:
        argv += ["--use_quadratic_motion", "--lr_c2", str(lr_c2)]
    if use_s3_motion:
        argv += ["--use_s3_motion", "--lr_omega", str(lr_omega)]
    if profile_breakdown:
        argv += ["--profile_breakdown",
                 "--profile_warmup_iters", str(profile_warmup_iters)]
    _run(argv)
    ckpt_vol.commit()


@app.function(gpu=GPU, volumes=VOLUMES, timeout=2 * 3600)
def render(
    dataset: str,
    scene: str,
    ckpt: str,
    frames: str,
    image_scale: int,
    split: str | None,
    allow_distortion: bool,
    side_by_side: bool,
    sigma_3d_blur: float,
) -> None:
    import os
    scene_dir = _ensure_scene_unpacked(scene)
    ckpt_path = f"/checkpoints/{ckpt}"
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"{ckpt_path!r} not found on gs-checkpoints volume.")
    out_dir = os.path.join(os.path.dirname(ckpt_path), "renders")
    argv = [
        "python", "scripts/render_mono.py",
        "--dataset", dataset,
        "--scene_dir", scene_dir,
        "--ckpt", ckpt_path,
        "--frames", frames,
        "--output_dir", out_dir,
        "--image_scale", str(image_scale),
        "--device", "cuda",
        "--sigma_3d_blur", str(sigma_3d_blur),
    ]
    if split is not None:
        argv += ["--split", split]
    if allow_distortion:
        argv.append("--allow_distortion")
    if side_by_side:
        argv.append("--side_by_side")
    _run(argv)
    ckpt_vol.commit()
    rel = os.path.relpath(out_dir, "/checkpoints")
    print(f"\nPull renders locally:\n  modal volume get gs-checkpoints {rel} ./renders", flush=True)


@app.function(gpu=GPU, volumes=VOLUMES, timeout=2 * 3600)
def eval_per_frame(
    dataset: str,
    scene: str,
    ckpt: str,
    image_scale: int,
    split: str | None,
    split_convention: str,
    allow_distortion: bool,
    sigma_3d_blur: float,
) -> None:
    import os
    scene_dir = _ensure_scene_unpacked(scene)
    ckpt_path = f"/checkpoints/{ckpt}"
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"{ckpt_path!r} not found on gs-checkpoints volume.")
    out_dir = os.path.join(os.path.dirname(ckpt_path), "per_frame_diag")
    argv = [
        "python", "scripts/eval_per_frame.py",
        "--dataset", dataset,
        "--scene_dir", scene_dir,
        "--ckpt", ckpt_path,
        "--output_dir", out_dir,
        "--image_scale", str(image_scale),
        "--split_convention", split_convention,
        "--device", "cuda",
        "--sigma_3d_blur", str(sigma_3d_blur),
    ]
    if split is not None:
        argv += ["--split", split]
    if allow_distortion:
        argv.append("--allow_distortion")
    _run(argv)
    ckpt_vol.commit()
    rel = os.path.relpath(out_dir, "/checkpoints")
    print(f"\nPull artifacts locally:\n  modal volume get gs-checkpoints {rel} ./per_frame_diag", flush=True)


@app.local_entrypoint()
def main(
    cmd: str = "smoke",
    dataset: str = "nerfies",
    scene: str = "slice-banana",
    iters: int = 500,
    log_every: int = 50,
    init_strategy: str = "random",
    split: str = "",
    allow_distortion: bool = True,
    ckpt: str = "",
    frames: str = "0",
    image_scale: int = 4,
    side_by_side: bool = False,
    sigma_3d_blur: float = 1e-4,
    sigma_init_sq: float = 0.02,
    run_tag: str = "",
    seed: int = -1,
    static_baseline: bool = False,
    val_stride: int = 4,
    split_convention: str = "val_stride",
    init_points_multiplier: int = 1,
    init_per_frame_stride: int = 0,
    diag_single_frame: int = -1,
    lambda_frob: float = 0.0,
    opacity_reset_every: int = 0,
    opacity_reset_logit: float = -5.0,
    lambda_aniso: float = 0.0,
    densify_every: int = 0,
    densify_start: int = 500,
    densify_stop: int = 10000,
    grad_threshold: float = 2e-4,
    spatial_split_threshold: float = 0.5,
    max_split_per_event: int = 0,
    opacity_prune_threshold: float = 1e-3,
    scale_min_prune: float = 1e-6,
    scale_max_prune: float = 100.0,
    mu_t_min: float = -1e10,
    mu_t_max: float = 1e10,
    sh_degree: int = 0,
    lr_decay: float = 1.0,
    lr_pos_scale: float = 1.0,
    lambda_structural: float = 0.2,
    structural_kind: str = "boxstats",
    mu_constraint: str = "free",
    lambda_mu_penalty: float = 1.0,
    clamp_mode: str = "hard",
    eps_schur: float = -1.0,
    mu_lr_split: bool = False,
    lr_mu_spatial: float = 1e-4,
    lr_mu_time: float = 1e-3,
    init_points_path: str = "",
    init_colors_path: str = "",
    # --- Wave A probe flags (quality_knobs_evaluation.md) ---
    color_lr_warmup_iter: int = 0,
    random_background: bool = False,
    sigma_init_knn_k: int = 0,
    sigma_init_alpha_t: float = 0.1,
    max_aspect_ratio: float = 0.0,
    exposure_per_frame: bool = False,
    lambda_exposure_reg: float = 1e-3,
    temporal_split_threshold: float = 0.0,
    lambda_time_coherence: float = 0.0,
    time_coherence_dt: float = 0.05,
    mip_filter_sigma_pixel: float = 0.0,
    refine_poses: bool = False,
    lr_pose_rot: float = 1e-5,
    lr_pose_trans: float = 1e-4,
    pose_warmup_iter: int = 2000,
    lambda_depth: float = 0.0,
    depth_model: str = "depth_anything_v2_small",
    grassmann_relax_start: int = 0,
    grassmann_relax_end: int = 0,
    floater_min_views: int = 0,
    floater_eps: float = 1e-3,
    sh_degree_warmup_step: int = 0,
    lambda_opacity_entropy: float = 0.0,
    density_strategy: str = "heuristic",
    mcmc_noise_lr: float = 0.0,
    mcmc_noise_after: int = 0,
    mcmc_noise_gate_k: float = 100.0,
    mcmc_noise_gate_thr: float = 0.005,
    mcmc_max_relocations_per_step: int = 0,
    split_anisotropic_shrink: bool = False,
    split_opacity_correction: bool = False,
    split_opacity_brighter: bool = False,
    split_shrink_factor: float = 1.6,
    split_offset_sigmas: float = 1.0,
    trigger_post_schur: bool = False,
    merge_every: int = 0,
    merge_distance: float = 0.0,
    merge_normal_cos: float = 0.95,
    aspect_split_threshold: float = 0.0,
    use_quadratic_motion: bool = False,
    lr_c2: float = 5e-4,
    use_s3_motion: bool = False,
    lr_omega: float = 5e-4,
    profile_breakdown: bool = False,
    profile_warmup_iters: int = 200,
):
    """
    --cmd smoke:  short run (--iters used; default 500) at scale 4. Validates
                  code path AND prints per-log_every loss to confirm convergence.
    --cmd train:  full run at scale 2 (default --iters 30000).
    --cmd render: load --ckpt (path under /checkpoints), render --frames via CUDA.
    """
    split_arg = split or None
    seed_arg = None if seed < 0 else seed
    if cmd == "smoke":
        train.remote(
            dataset=dataset, scene=scene,
            num_iters=iters, image_scale=4, use_fast=True,
            init_strategy=init_strategy, split=split_arg,
            allow_distortion=allow_distortion,
            log_every=log_every,
            sigma_3d_blur=sigma_3d_blur,
            sigma_init_sq=sigma_init_sq,
            run_tag=run_tag,
            seed=seed_arg,
            static_baseline=static_baseline,
            val_stride=val_stride,
            split_convention=split_convention,
            init_points_multiplier=init_points_multiplier,
            init_per_frame_stride=init_per_frame_stride,
            diag_single_frame=diag_single_frame,
            lambda_frob=lambda_frob,
            opacity_reset_every=opacity_reset_every,
            opacity_reset_logit=opacity_reset_logit,
            lambda_aniso=lambda_aniso,
            densify_every=densify_every,
            densify_start=densify_start,
            densify_stop=densify_stop,
            grad_threshold=grad_threshold,
            spatial_split_threshold=spatial_split_threshold,
            max_split_per_event=max_split_per_event,
            opacity_prune_threshold=opacity_prune_threshold,
            scale_min_prune=scale_min_prune,
            scale_max_prune=scale_max_prune,
            mu_t_min=mu_t_min,
            mu_t_max=mu_t_max,
            sh_degree=sh_degree,
            lr_decay=lr_decay,
            lr_pos_scale=lr_pos_scale,
            lambda_structural=lambda_structural,
            structural_kind=structural_kind,
            mu_constraint=mu_constraint,
            lambda_mu_penalty=lambda_mu_penalty,
            clamp_mode=clamp_mode,
            eps_schur=eps_schur,
            mu_lr_split=mu_lr_split,
            lr_mu_spatial=lr_mu_spatial,
            lr_mu_time=lr_mu_time,
            init_points_path=init_points_path,
            init_colors_path=init_colors_path,
            color_lr_warmup_iter=color_lr_warmup_iter,
            random_background=random_background,
            sigma_init_knn_k=sigma_init_knn_k,
            sigma_init_alpha_t=sigma_init_alpha_t,
            max_aspect_ratio=max_aspect_ratio,
            exposure_per_frame=exposure_per_frame,
            lambda_exposure_reg=lambda_exposure_reg,
            temporal_split_threshold=temporal_split_threshold,
            lambda_time_coherence=lambda_time_coherence,
            time_coherence_dt=time_coherence_dt,
            mip_filter_sigma_pixel=mip_filter_sigma_pixel,
            refine_poses=refine_poses,
            lr_pose_rot=lr_pose_rot,
            lr_pose_trans=lr_pose_trans,
            pose_warmup_iter=pose_warmup_iter,
            lambda_depth=lambda_depth,
            depth_model=depth_model,
            grassmann_relax_start=grassmann_relax_start,
            grassmann_relax_end=grassmann_relax_end,
            floater_min_views=floater_min_views,
            floater_eps=floater_eps,
            sh_degree_warmup_step=sh_degree_warmup_step,
            lambda_opacity_entropy=lambda_opacity_entropy,
            density_strategy=density_strategy,
            mcmc_noise_lr=mcmc_noise_lr,
            mcmc_noise_after=mcmc_noise_after,
            mcmc_noise_gate_k=mcmc_noise_gate_k,
            mcmc_noise_gate_thr=mcmc_noise_gate_thr,
            mcmc_max_relocations_per_step=mcmc_max_relocations_per_step,
            split_anisotropic_shrink=split_anisotropic_shrink,
            split_opacity_correction=split_opacity_correction,
            split_opacity_brighter=split_opacity_brighter,
            split_shrink_factor=split_shrink_factor,
            split_offset_sigmas=split_offset_sigmas,
            trigger_post_schur=trigger_post_schur,
            merge_every=merge_every,
            merge_distance=merge_distance,
            merge_normal_cos=merge_normal_cos,
            aspect_split_threshold=aspect_split_threshold,
            use_quadratic_motion=use_quadratic_motion,
            lr_c2=lr_c2,
            use_s3_motion=use_s3_motion,
            lr_omega=lr_omega,
            profile_breakdown=profile_breakdown,
            profile_warmup_iters=profile_warmup_iters,
        )
    elif cmd == "train":
        train.remote(
            dataset=dataset, scene=scene,
            num_iters=iters if iters != 500 else 30000,
            image_scale=2, use_fast=True,
            init_strategy=init_strategy, split=split_arg,
            allow_distortion=allow_distortion,
            log_every=log_every if log_every != 50 else 200,
            sigma_3d_blur=sigma_3d_blur,
            sigma_init_sq=sigma_init_sq,
            run_tag=run_tag,
            seed=seed_arg,
            static_baseline=static_baseline,
            val_stride=val_stride,
            split_convention=split_convention,
            init_points_multiplier=init_points_multiplier,
            init_per_frame_stride=init_per_frame_stride,
            diag_single_frame=diag_single_frame,
            lambda_frob=lambda_frob,
            opacity_reset_every=opacity_reset_every,
            opacity_reset_logit=opacity_reset_logit,
            lambda_aniso=lambda_aniso,
            densify_every=densify_every,
            densify_start=densify_start,
            densify_stop=densify_stop,
            grad_threshold=grad_threshold,
            spatial_split_threshold=spatial_split_threshold,
            max_split_per_event=max_split_per_event,
            opacity_prune_threshold=opacity_prune_threshold,
            scale_min_prune=scale_min_prune,
            scale_max_prune=scale_max_prune,
            mu_t_min=mu_t_min,
            mu_t_max=mu_t_max,
            sh_degree=sh_degree,
            lr_decay=lr_decay,
            lr_pos_scale=lr_pos_scale,
            lambda_structural=lambda_structural,
            structural_kind=structural_kind,
            mu_constraint=mu_constraint,
            lambda_mu_penalty=lambda_mu_penalty,
            clamp_mode=clamp_mode,
            eps_schur=eps_schur,
            mu_lr_split=mu_lr_split,
            lr_mu_spatial=lr_mu_spatial,
            lr_mu_time=lr_mu_time,
            init_points_path=init_points_path,
            init_colors_path=init_colors_path,
            color_lr_warmup_iter=color_lr_warmup_iter,
            random_background=random_background,
            sigma_init_knn_k=sigma_init_knn_k,
            sigma_init_alpha_t=sigma_init_alpha_t,
            max_aspect_ratio=max_aspect_ratio,
            exposure_per_frame=exposure_per_frame,
            lambda_exposure_reg=lambda_exposure_reg,
            temporal_split_threshold=temporal_split_threshold,
            lambda_time_coherence=lambda_time_coherence,
            time_coherence_dt=time_coherence_dt,
            mip_filter_sigma_pixel=mip_filter_sigma_pixel,
            refine_poses=refine_poses,
            lr_pose_rot=lr_pose_rot,
            lr_pose_trans=lr_pose_trans,
            pose_warmup_iter=pose_warmup_iter,
            lambda_depth=lambda_depth,
            depth_model=depth_model,
            grassmann_relax_start=grassmann_relax_start,
            grassmann_relax_end=grassmann_relax_end,
            floater_min_views=floater_min_views,
            floater_eps=floater_eps,
            sh_degree_warmup_step=sh_degree_warmup_step,
            lambda_opacity_entropy=lambda_opacity_entropy,
            density_strategy=density_strategy,
            mcmc_noise_lr=mcmc_noise_lr,
            mcmc_noise_after=mcmc_noise_after,
            mcmc_noise_gate_k=mcmc_noise_gate_k,
            mcmc_noise_gate_thr=mcmc_noise_gate_thr,
            mcmc_max_relocations_per_step=mcmc_max_relocations_per_step,
            split_anisotropic_shrink=split_anisotropic_shrink,
            split_opacity_correction=split_opacity_correction,
            split_opacity_brighter=split_opacity_brighter,
            split_shrink_factor=split_shrink_factor,
            split_offset_sigmas=split_offset_sigmas,
            trigger_post_schur=trigger_post_schur,
            merge_every=merge_every,
            merge_distance=merge_distance,
            merge_normal_cos=merge_normal_cos,
            aspect_split_threshold=aspect_split_threshold,
            use_quadratic_motion=use_quadratic_motion,
            lr_c2=lr_c2,
            use_s3_motion=use_s3_motion,
            lr_omega=lr_omega,
            profile_breakdown=profile_breakdown,
            profile_warmup_iters=profile_warmup_iters,
        )
    elif cmd == "render":
        if not ckpt:
            raise SystemExit("--cmd render requires --ckpt <path-under-/checkpoints>")
        render.remote(
            dataset=dataset, scene=scene,
            ckpt=ckpt, frames=frames, image_scale=image_scale,
            split=split_arg, allow_distortion=allow_distortion,
            side_by_side=side_by_side,
            sigma_3d_blur=sigma_3d_blur,
        )
    elif cmd == "eval_per_frame":
        if not ckpt:
            raise SystemExit("--cmd eval_per_frame requires --ckpt <path-under-/checkpoints>")
        eval_per_frame.remote(
            dataset=dataset, scene=scene,
            ckpt=ckpt, image_scale=image_scale,
            split=split_arg, split_convention=split_convention,
            allow_distortion=allow_distortion,
            sigma_3d_blur=sigma_3d_blur,
        )
    else:
        raise SystemExit(f"unknown --cmd {cmd!r}; expected smoke|train|render|eval_per_frame")
