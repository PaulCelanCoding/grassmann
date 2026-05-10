"""
Training loop for the Grassmann model against multi-view multi-frame video.

Usage pattern:
    model = trainable_from_params(init_params)
    trainer = Trainer(model, cameras, frame_data, times)
    trainer.train(num_iters=2000, log_every=100)

frame_data can be:
  - a Tensor of shape (K, T, H, W, 3): K cameras, T frames each, sampled
    densely;
  - or a callable (cam_idx, t) -> (H, W, 3) tensor for on-the-fly loading.

times is a sequence of T time values.

The loop performs stochastic minibatching: at each iteration we sample a
random (camera, frame) pair, render it, compute loss, backprop, step Adam,
re-normalize manifolds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import torch
from torch import Tensor, nn

from .gaussian import compute_derived, condition_on_time
from .losses import l1_loss, mse_loss, photometric_loss, LPIPSLoss
from .projection import Camera
from .rasterizer import project_to_screen, rasterize
from .trainable import TrainableGaussians, build_optimizer
from .density_control import DensityTracker, DensityConfig
from .fast_rasterizer import fast_rasterize, FastRasterConfig, is_available as fast_available
from .surfel_rasterizer import (
    surfel_rasterize, SurfelRasterConfig, is_available as surfel_available,
)
from .losses import (
    depth_distortion_loss, normal_consistency_loss, depth_to_world_normal,
)


@dataclass
class TrainerConfig:
    """Training hyperparameters."""
    num_iters: int = 2000
    batch_size_views: int = 1       # cameras per step (1 = pure stochastic)
    log_every: int = 100
    renormalize_every: int = 1       # renormalize n_raw onto S^3 (cheap; do every step)
    lambda_l1: float = 0.8
    lambda_structural: float = 0.2
    lambda_lpips: float = 0.0
    use_lpips: bool = False           # requires lpips package + pretrained download
    lpips_net: str = "alex"           # 'alex' (fast) or 'vgg' (heavier)
    # Learning rates (see trainable.build_optimizer). Names track the 3-plane
    # parameterization: n is the unit-vector normal, mu is the R^4 mean,
    # L is the 4x3 raw factor.
    lr_n: float = 1e-3
    lr_mu: float = 5e-3
    lr_L: float = 5e-3
    lr_opacity: float = 5e-2
    lr_color: float = 2e-2
    # v7-doc §7.5: per-axis μ-LR (only used when model.mu_lr_split=True).
    lr_mu_spatial: float = 1e-4
    lr_mu_time: float = 1e-3
    # Log-linear LR decay applied to geometric params (n, mu, L_raw) only.
    # 1.0 disables; <1 decays geometric LRs from base*1 to base*lr_decay over
    # num_iters via lr(t) = base * lr_decay**t. Mirrors 3DGS's position_lr_final/
    # position_lr_init = 0.01 schedule over 30k iters. Color/opacity not scheduled
    # (3DGS does the same).
    lr_decay: float = 1.0
    # #5.2: warmup color LR linearly from 0 -> base over the first
    # color_lr_warmup_iter iterations. 0 disables (constant from step 1).
    color_lr_warmup_iter: int = 0
    # #6.2: hard cap on aspect ratio λ_max/λ_min of Σ_3D in-plane eigenvalues
    # (equivalently (s_max/s_min)² of P_n L_raw singular values). 0 disables.
    # Applied every `aspect_clip_every` iters as a no-grad SVD-based projection.
    max_aspect_ratio: float = 0.0
    aspect_clip_every: int = 100
    # Background color for rendering
    background: Tensor = field(default_factory=lambda: torch.tensor([0.05, 0.05, 0.1]))
    # #7.2: at each train_step, replace the constant `background` with a
    # uniform random RGB sample. Validation/render still uses `background`.
    random_background: bool = False
    # Validation
    validation_every: int = 0          # 0 = disabled
    validation_cams: Optional[list[int]] = None  # which camera indices to evaluate on
    validation_times: Optional[list[float]] = None  # which times
    validation_frames: Optional[list[int]] = None  # monocular: which frame indices
    train_frames: Optional[list[int]] = None       # monocular: restrict train sampling to this subset
                                                   # (defaults to all T frames if None)
    # Density control (Phase 6)
    densify_every: int = 0            # 0 = disabled
    densify_start: int = 500          # don't start densification before this iteration
    densify_stop: int = 15000         # stop densification after this iteration
    density_config: Optional[DensityConfig] = None  # hyperparams; defaults built if None
    # Fast rasterizer (Phase 7)
    use_fast_rasterizer: bool = False  # True -> use CUDA diff-gaussian-rasterization if available
    fast_raster_config: Optional[FastRasterConfig] = None  # defaults built if None
    # Sampling mode
    monocular: bool = False  # True -> cameras list is per-frame; t_idx == cam_idx; one sample per step
    # Static baseline: disable time conditioning entirely (Schur step skipped,
    # w_t=1 always). Used to measure the "static-3DGS-on-monocular-bundle" floor
    # within the same pipeline. The gap between this and the full temporal run
    # quantifies the value of time conditioning.
    static_baseline: bool = False
    # Phase-A-correctness penalties (RCA: 32% dead, 30% high-aniso, 7% rank-1-collapsed):
    #   lambda_frob: Frobenius-norm penalty on L_raw -- prevents the optimizer from
    #     soft-collapsing rank by routing capacity into the n̂ direction (which the
    #     projector annihilates). Recommended ~1e-4.
    #   opacity_reset_every: every N iters, reset opacity_logit to opacity_reset_logit
    #     (mirrors the standard 3DGS opacity-reset trick). 0 disables.
    #   opacity_reset_logit: target logit for the reset (sigmoid(-5)≈0.007).
    #   lambda_aniso: bounded anisotropy penalty on Σ_3D(t_0). Trims the runaway
    #     λ_max/λ_min tail (p99 ≈ 6.8e7 in the unpenalized 50k checkpoint).
    lambda_frob: float = 0.0
    opacity_reset_every: int = 0
    opacity_reset_logit: float = -5.0
    lambda_aniso: float = 0.0
    # μ-DOF probe: soft penalty on <n, mu>^2 (used when mu_constraint="penalty").
    # 0 disables. See results/rca/mu_dof_ab_test.md and the GaussianParams
    # mu_constraint docstring for the full A/B context.
    lambda_mu_penalty: float = 0.0
    # #5.3 time-coherence regularizer.
    # Penalize ‖μ_3D(t+dt) − μ_3D(t)‖² · w_t · w_{t+dt} for sampled (t, t+dt).
    # 0 disables. Uses already-sampled t per step + dt offset (symmetric ±dt/2).
    lambda_time_coherence: float = 0.0
    time_coherence_dt: float = 0.05
    # #1.1 per-frame learnable exposure: rendered ← exp(log_gain[t]) · rendered + bias[t].
    # Compensates AE/AWB drift in NeRFies/DyCheck. lambda_exposure_reg L2's params.
    exposure_per_frame: bool = False
    lambda_exposure_reg: float = 1e-3
    lr_exposure: float = 1e-3
    # #3.2 progressive Grassmann relaxation: scale lr_n from 0 → base over
    # [grassmann_relax_start, grassmann_relax_end]. Use with init_strategy=
    # spatial_slice (n=e₀ at init) so the geometry settles in the static-3DGS
    # regime before n is allowed to tilt. 0/0 disables.
    grassmann_relax_start: int = 0
    grassmann_relax_end: int = 0
    # #2.1 pose refinement: per-frame so3 + translation perturbation.
    # δR via exp(skew(dR)) @ R_orig, δc via c_orig + dt (world frame).
    # LR is held at 0 until iter `pose_warmup_iter`, then ramped to lr_R / lr_t.
    refine_poses: bool = False
    lr_pose_rot: float = 1e-5
    lr_pose_trans: float = 1e-4
    pose_warmup_iter: int = 2000
    # #6.1 SH-degree warmup: increase eff_sh_degree by 1 every N iters
    # (capped at model's max sh_degree). 0 disables (always max).
    sh_degree_warmup_step: int = 0
    # #6.3 opacity entropy regularizer: push α toward {0, 1}.
    # Loss term: -λ · mean(α log α + (1-α) log(1-α)). 0 disables.
    lambda_opacity_entropy: float = 0.0
    # Structural-loss kind: 'boxstats' (legacy 7x7 local-mean+var matcher) or
    # 'ssim' (1 - SSIM Gaussian-windowed, matches 3DGS DSSIM term). Only
    # active when lambda_structural > 0.
    structural_kind: str = "boxstats"
    # Rasterizer choice: 'gaussian' (Inria diff_gaussian_rasterization, with
    # σ_lift² rank-2→rank-3 lift) or 'surfel' (Huang2024 diff_surfel_rasterization,
    # native rank-2 disk via ray-plane intersection). See results/rca/surfel_*
    # for the A/B test.
    rasterizer: str = "gaussian"
    surfel_raster_config: Optional[SurfelRasterConfig] = None
    # 2DGS regularizers — only meaningful when rasterizer == 'surfel'. Lambdas
    # match 2DGS paper defaults; schedule activates after the listed iter.
    use_2dgs_losses: bool = False
    lambda_normal: float = 0.05
    lambda_dist: float = 100.0
    normal_after: int = 7000
    dist_after: int = 3000
    depth_ratio: float = 0.0  # 0 = expected depth (mip-NeRF-like), 1 = median (DTU)


FrameData = Union[Tensor, Callable[[int, float], Tensor]]


class Trainer:
    """Trainer for a TrainableGaussians model against multi-view multi-frame data.

    model:         TrainableGaussians
    cameras:       list of K Camera objects (static)
    frame_data:    either (K, T, H, W, 3) tensor OR callable (cam_idx, t) -> (H,W,3)
    times:         sequence of T time values (matching frame_data's T if tensor)
    H, W:          image dimensions
    config:        TrainerConfig
    """

    def __init__(
        self,
        model: TrainableGaussians,
        cameras: list[Camera],
        frame_data: FrameData,
        times: list[float],
        H: int,
        W: int,
        config: Optional[TrainerConfig] = None,
    ):
        self.model = model
        self.cameras = cameras
        self.K = len(cameras)
        self.frame_data = frame_data
        self.times = list(times)
        self.T = len(times)
        self.H = H
        self.W = W
        self.config = config or TrainerConfig()

        # Capture the learning-rate config as a callable so we can rebuild the
        # optimizer after density control changes the parameter set.
        self._build_opt = lambda m: build_optimizer(
            m,
            lr_n=self.config.lr_n,
            lr_mu=self.config.lr_mu,
            lr_L=self.config.lr_L,
            lr_opacity=self.config.lr_opacity,
            lr_color=self.config.lr_color,
            lr_mu_spatial=self.config.lr_mu_spatial,
            lr_mu_time=self.config.lr_mu_time,
        )
        self.optimizer = self._build_opt(model)
        # Migration: density tracker may rebuild the optimizer mid-training.
        # Capture base LRs from the FIRST optimizer build so the scheduler is
        # not perturbed by density-event re-instantiations.

        # Snapshot base LRs so the scheduler can multiply by decay**t. We only
        # schedule geometric params (n, mu, L_raw); color/opacity stay constant.
        self._base_lrs: dict[str, float] = {
            g["name"]: g["lr"] for g in self.optimizer.param_groups
            if g["name"] in ("n", "mu", "L_raw")
        }

        # Density control (Phase 6). The tracker holds a reference to the
        # optimizer so density events can migrate Adam state in place rather
        # than rebuilding from scratch (RCA Bug D fix).
        self.density_tracker: Optional[DensityTracker] = None
        if self.config.densify_every > 0:
            self.density_tracker = DensityTracker(model, self.optimizer)
            if self.config.density_config is None:
                self.config.density_config = DensityConfig()

        # LPIPS setup (optional).
        self.lpips_fn: Optional[LPIPSLoss] = None
        if self.config.use_lpips and self.config.lambda_lpips > 0:
            try:
                self.lpips_fn = LPIPSLoss(net=self.config.lpips_net, device="cpu")
            except ImportError as e:
                print(f"  [warning] LPIPS disabled: {e}")
                self.lpips_fn = None

        # #2.1 pose-refinement params (per-frame so3 vec + translation).
        self.pose_dR: Optional[nn.Parameter] = None
        self.pose_dt: Optional[nn.Parameter] = None
        if self.config.refine_poses:
            T = self.T
            dev = self.model.n_raw.device
            dt_ = self.model.n_raw.dtype
            self.pose_dR = nn.Parameter(torch.zeros(T, 3, dtype=dt_, device=dev))
            self.pose_dt = nn.Parameter(torch.zeros(T, 3, dtype=dt_, device=dev))
            # LR initially 0 (warmup); ramped in train loop.
            self.optimizer.add_param_group(
                {"params": [self.pose_dR], "lr": 0.0, "name": "pose_dR"}
            )
            self.optimizer.add_param_group(
                {"params": [self.pose_dt], "lr": 0.0, "name": "pose_dt"}
            )

        # #1.1 per-frame exposure (log_gain (T,), bias (T, 3)).
        self.exposure_log_gain: Optional[nn.Parameter] = None
        self.exposure_bias: Optional[nn.Parameter] = None
        if self.config.exposure_per_frame:
            T = self.T
            dev = self.model.n_raw.device
            dt = self.model.n_raw.dtype
            self.exposure_log_gain = nn.Parameter(torch.zeros(T, dtype=dt, device=dev))
            self.exposure_bias = nn.Parameter(torch.zeros(T, 3, dtype=dt, device=dev))
            self.optimizer.add_param_group(
                {"params": [self.exposure_log_gain],
                 "lr": self.config.lr_exposure, "name": "exposure_log_gain"}
            )
            self.optimizer.add_param_group(
                {"params": [self.exposure_bias],
                 "lr": self.config.lr_exposure, "name": "exposure_bias"}
            )

        # #6.1 SH-degree warmup state.
        self.current_iter: int = 0
        self.history: dict[str, list] = {"iter": [], "loss": [], "l1": [], "psnr": [], "N": []}

    # def get_frame(self, cam_idx: int, t_idx: int) -> Tensor:
    #     """Get the target frame for camera cam_idx at time index t_idx."""
    #     if isinstance(self.frame_data, torch.Tensor):
    #         return self.frame_data[cam_idx, t_idx]
    #     else:
    #         return self.frame_data(cam_idx, self.times[t_idx])

    def get_frame(self, cam_idx: int, t_idx: int) -> Tensor:
        """Get the target frame for camera cam_idx at time index t_idx.
        Always returned on the model's device so loss computations don't cross devices."""
        if isinstance(self.frame_data, torch.Tensor):
            frame = self.frame_data[cam_idx, t_idx]
        else:
            frame = self.frame_data(cam_idx, self.times[t_idx])
        return frame.to(device=self.model.n_raw.device)

    def _perturbed_camera(self, cam_idx: int):
        """#2.1: build a Camera with R_new = exp(skew(dR)) @ R, c_new = c + dt.

        Returns a new dataclass — keeps autograd live through the per-frame
        pose params. No-op when pose refinement is disabled.
        """
        cam = self.cameras[cam_idx]
        if self.pose_dR is None:
            return cam
        dR = self.pose_dR[cam_idx]                                # (3,)
        dt = self.pose_dt[cam_idx]                                # (3,)
        # Skew-symmetric matrix from so3 vec.
        z = torch.zeros_like(dR[0])
        K = torch.stack([
            torch.stack([z, -dR[2], dR[1]]),
            torch.stack([dR[2], z, -dR[0]]),
            torch.stack([-dR[1], dR[0], z]),
        ])
        delta_R = torch.linalg.matrix_exp(K)                      # (3, 3)
        R_new = delta_R @ cam.R.to(dtype=dR.dtype, device=dR.device)
        c_new = cam.c.to(dtype=dt.dtype, device=dt.device) + dt
        from .projection import Camera as _Cam
        return _Cam(R=R_new, c=c_new, fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy)

    def render_one(self, cam_idx: int, t_value: float,
                   means2d_capture: Optional[list] = None,
                   return_aux: bool = False,
                   bg_override: Optional[Tensor] = None):
        """Render the current model from camera cam_idx at time t_value.

        When `means2d_capture` is a list, the means2D dummy tensor (which gets
        screen-space gradients from the CUDA kernel after backward) is appended
        to it. Used by DensityTracker for the screen-space ‖∇μ_2d‖ trigger.

        When `return_aux=True` and the surfel rasterizer is active, returns
        (image, aux_dict) — aux contains 'rend_normal', 'rend_dist', etc. (see
        grassmann.surfel_rasterizer.RENDER_PKG_KEYS). Otherwise returns image.

        bg_override: optional (3,) tensor that replaces self.config.background
        for this call only — used by #7.2 random_background during train_step.
        """
        params = self.model.forward()
        bg_src = bg_override if bg_override is not None else self.config.background
        bg = bg_src.to(dtype=params.color.dtype, device=params.color.device)

        if self.config.rasterizer == "surfel":
            if not (surfel_available() and params.n.is_cuda):
                raise RuntimeError(
                    "rasterizer='surfel' requires diff_surfel_rasterization + CUDA"
                )
            sc = self.config.surfel_raster_config or SurfelRasterConfig()
            out = surfel_rasterize(
                params, t_value, self._perturbed_camera(cam_idx), self.H, self.W,
                background=bg, config=sc,
                static_baseline=self.config.static_baseline,
                means2d_capture=means2d_capture,
                return_aux=return_aux,
            )
            return out

        if self.config.use_fast_rasterizer and fast_available() and params.n.is_cuda:
            fc = self.config.fast_raster_config or FastRasterConfig()
            # #6.1 SH-degree warmup: cap effective sh_degree per current iter.
            sh_override = None
            warmup_step = self.config.sh_degree_warmup_step
            if warmup_step > 0:
                sh_override = min(fc.sh_degree, self.current_iter // warmup_step)
            img = fast_rasterize(
                params, t_value, self._perturbed_camera(cam_idx), self.H, self.W,
                background=bg, config=fc,
                static_baseline=self.config.static_baseline,
                means2d_capture=means2d_capture,
                sh_degree_override=sh_override,
            )
            return (img, None) if return_aux else img
        # Fallback: toy rasterizer path (also used when no GPU or no extension).
        if means2d_capture is not None:
            means2d_capture.append(None)
        derived = compute_derived(params)
        tc = condition_on_time(params, derived, t_value, static=self.config.static_baseline)
        sg = project_to_screen(params, tc, self._perturbed_camera(cam_idx))
        img = rasterize(sg, H=self.H, W=self.W, background=bg)
        return (img, None) if return_aux else img

    def train_step(self, iter_num: int = 0) -> tuple[float, float, float]:
        """One stochastic training step. Returns (total loss, L1, PSNR_dB).

        iter_num is the (1-based) current iteration; used to gate the 2DGS
        depth-distortion / normal-consistency lambdas per the paper schedule.
        """
        self.current_iter = iter_num
        # Sample a frame.
        if self.config.monocular:
            # Monocular: one camera per frame, sampled together. self.K must equal self.T.
            train_pool = self.config.train_frames if self.config.train_frames else list(range(self.T))
            pick = torch.randint(0, len(train_pool), (1,)).item()
            t_idx = int(train_pool[pick])
            cam_idx = t_idx
        else:
            cam_idx = torch.randint(0, self.K, (1,)).item()
            t_idx = torch.randint(0, self.T, (1,)).item()
        t_value = self.times[t_idx]

        # Target and rendered. If DC is enabled, capture the means2D dummy
        # tensor so the tracker can read screen-space gradients post-backward.
        means2d_capture: Optional[list] = [] if self.density_tracker is not None else None
        target = self.get_frame(cam_idx, t_idx).to(self.model.n_raw.dtype)
        # Surfel + 2DGS losses: pull aux maps from the rasterizer for
        # depth-distortion / normal-consistency regularization.
        want_aux = (self.config.rasterizer == "surfel"
                    and self.config.use_2dgs_losses)
        # #7.2 random background during training only.
        bg_override = None
        if self.config.random_background:
            bg_override = torch.rand(3, dtype=self.model.n_raw.dtype,
                                     device=self.model.n_raw.device)
        if want_aux:
            rendered, aux = self.render_one(cam_idx, t_value,
                                            means2d_capture=means2d_capture,
                                            return_aux=True,
                                            bg_override=bg_override)
        else:
            rendered = self.render_one(cam_idx, t_value,
                                       means2d_capture=means2d_capture,
                                       bg_override=bg_override)
            aux = None

        # #1.1 per-frame exposure: rendered ← exp(g_t)·rendered + b_t.
        # rendered may be (H, W, 3) (toy/fast paths) or (3, H, W) (surfel path).
        # Bias is per-channel (3,); broadcast against last or first axis.
        if self.exposure_log_gain is not None:
            g = torch.exp(self.exposure_log_gain[t_idx])                   # scalar
            b = self.exposure_bias[t_idx]                                  # (3,)
            if rendered.dim() == 3 and rendered.shape[-1] == 3:
                # (H, W, 3): bias broadcasts naturally on last dim.
                rendered = (g * rendered + b).clamp(0.0, 1.0)
            else:
                # (3, H, W) layout (e.g. surfel aux path).
                rendered = (g * rendered + b.view(3, 1, 1)).clamp(0.0, 1.0)

        # Loss.
        loss = photometric_loss(
            rendered, target,
            lambda_l1=self.config.lambda_l1,
            lambda_structural=self.config.lambda_structural,
            structural_kind=self.config.structural_kind,
            lpips_fn=self.lpips_fn,
            lambda_lpips=self.config.lambda_lpips,
        )
        # #1.1 exposure L2 reg.
        if (self.exposure_log_gain is not None
                and self.config.lambda_exposure_reg > 0.0):
            reg = (self.exposure_log_gain ** 2).mean() + (self.exposure_bias ** 2).mean()
            loss = loss + self.config.lambda_exposure_reg * reg

        # Phase-A-correctness penalties.
        if self.config.lambda_frob > 0.0:
            # Mean-squared L_raw entries; targets the soft-rank-collapse pathology
            # where the optimizer routes capacity into the projector's null direction.
            loss = loss + self.config.lambda_frob * (self.model.L_raw ** 2).mean()
        if self.config.lambda_mu_penalty > 0.0:
            # μ-DOF probe: soft penalty <n, μ>² (forward-pass n is unit-normalized).
            n_unit = self.model.n_raw / self.model.n_raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            if self.model.mu_lr_split:
                mu_full = torch.cat([self.model.mu_time, self.model.mu_spatial], dim=-1)
            else:
                mu_full = self.model.mu
            n_dot_mu = (n_unit * mu_full).sum(-1)
            loss = loss + self.config.lambda_mu_penalty * (n_dot_mu ** 2).mean()
        if aux is not None:
            # 2DGS regularizers (gated by iter — paper defaults: dist@3000, normal@7000).
            l_dist = self.config.lambda_dist if iter_num > self.config.dist_after else 0.0
            l_norm = self.config.lambda_normal if iter_num > self.config.normal_after else 0.0
            if l_dist > 0.0:
                loss = loss + l_dist * depth_distortion_loss(aux["rend_dist"])
            if l_norm > 0.0:
                rend_alpha = aux["rend_alpha"]                  # (1, H, W)
                # surf_depth = expected*(1-r) + median*r
                exp_d = aux["expected_depth"] / rend_alpha.detach().clamp_min(1e-6)
                exp_d = torch.nan_to_num(exp_d, nan=0.0, posinf=0.0, neginf=0.0)
                med_d = torch.nan_to_num(aux["median_depth"], nan=0.0,
                                         posinf=0.0, neginf=0.0)
                surf_d = exp_d * (1 - self.config.depth_ratio) + \
                         self.config.depth_ratio * med_d         # (1, H, W)
                # rend_normal in allmap is view-space; rotate to world via R^T.
                cam = self._perturbed_camera(cam_idx)
                R_w = cam.R.to(device=aux["rend_normal"].device,
                               dtype=aux["rend_normal"].dtype)
                rend_n_view = aux["rend_normal"].permute(1, 2, 0)  # (H, W, 3)
                rend_n_world = (rend_n_view @ R_w).permute(2, 0, 1) # (3, H, W)
                surf_n_world = depth_to_world_normal(surf_d, cam).permute(2, 0, 1)
                surf_n_world = surf_n_world * rend_alpha.detach()
                loss = loss + l_norm * normal_consistency_loss(rend_n_world, surf_n_world)
        # Aniso / time-coherence both need a fresh forward+derived; share one.
        _need_derived = (self.config.lambda_aniso > 0.0
                         or self.config.lambda_time_coherence > 0.0)
        if _need_derived:
            from .gaussian import compute_derived, condition_on_time
            params_now = self.model.forward()
            d = compute_derived(params_now)
        # #6.3 opacity entropy reg: push α toward {0, 1}.
        if self.config.lambda_opacity_entropy > 0.0:
            alpha = torch.sigmoid(self.model.opacity_logit).clamp(1e-6, 1 - 1e-6)
            ent = -(alpha * torch.log(alpha) + (1 - alpha) * torch.log(1 - alpha))
            loss = loss + self.config.lambda_opacity_entropy * ent.mean()

        if self.config.lambda_aniso > 0.0:
            # Bounded anisotropy penalty on Σ_3D(t_0). We recompute Σ_3D_t inside
            # the model's forward graph by re-doing the projector + Schur — this
            # is differentiable and adds modest cost (3x3 eigvalsh per Gaussian).
            tc = condition_on_time(params_now, d, t_0=t_value)
            eigs = torch.linalg.eigvalsh(tc.Sigma_3D_t)             # (N, 3) ascending
            lam_max = eigs[..., 2]
            lam_min = eigs[..., 1]                                  # smallest non-zero
            eps = 1e-8
            aniso_normed = ((lam_max - lam_min) / (lam_max + lam_min + eps)) ** 2
            loss = loss + self.config.lambda_aniso * aniso_normed.mean()
        if self.config.lambda_time_coherence > 0.0:
            # #5.3: ‖V_3D(t+dt/2) − V_3D(t-dt/2)‖² · w_t1 · w_t2.
            # Symmetric step keeps gradient roughly centered around t_value.
            dt = float(self.config.time_coherence_dt)
            tc1 = condition_on_time(params_now, d, t_0=t_value - 0.5 * dt)
            tc2 = condition_on_time(params_now, d, t_0=t_value + 0.5 * dt)
            diff = (tc2.V_3D_t - tc1.V_3D_t)                          # (N, 3)
            w = (tc1.w_t * tc2.w_t).detach()                          # (N,) gate, no grad
            tc_loss = (w * (diff * diff).sum(-1)).mean()
            loss = loss + self.config.lambda_time_coherence * tc_loss
        with torch.no_grad():
            l1_val = l1_loss(rendered, target).item()
            mse_val = mse_loss(rendered, target).item()
            psnr_val = 10.0 * float(torch.log10(torch.tensor(max(mse_val, 1e-12)).reciprocal()))

        # Backprop.
        self.optimizer.zero_grad()
        loss.backward()

        # Density-control gradient accumulation (reads .grad BEFORE optimizer.step()).
        if self.density_tracker is not None:
            means2d = means2d_capture[0] if means2d_capture else None
            self.density_tracker.accumulate(means2d)

        self.optimizer.step()

        return loss.item(), l1_val, psnr_val

    def renormalize_manifolds(self) -> None:
        """Re-project n_raw onto S^3 (cheap maintenance step)."""
        self.model.renormalize_manifold_()

    def validate(self) -> dict[str, float]:
        """Render held-out frames and compute mean L1.

        Monocular mode: iterate over `validation_frames`. Raises if no frames
        are configured -- silently falling back to all frames would report
        training-set L1 as val_l1 (fake-success signal; see CLAUDE.md).
        Multi-cam (legacy): iterate over `validation_cams x validation_times`.
        """
        total_l1 = 0.0
        total_mse = 0.0
        count = 0
        with torch.no_grad():
            if self.config.monocular:
                frames = self.config.validation_frames
                if not frames:
                    raise ValueError(
                        "Trainer.validate() called in monocular mode but "
                        "config.validation_frames is empty. Set validation_frames "
                        "to a held-out subset, or set validation_every=0."
                    )
                for t_idx in frames:
                    t_value = self.times[t_idx]
                    target = self.get_frame(t_idx, t_idx).to(self.model.n_raw.dtype)
                    rendered = self.render_one(t_idx, t_value)
                    total_l1 += l1_loss(rendered, target).item()
                    total_mse += mse_loss(rendered, target).item()
                    count += 1
            else:
                cam_indices = self.config.validation_cams or list(range(self.K))
                times = self.config.validation_times or self.times
                for cam_idx in cam_indices:
                    for t_idx, t_value in enumerate(times):
                        try:
                            t_frame_idx = self.times.index(t_value)
                        except ValueError:
                            continue
                        target = self.get_frame(cam_idx, t_frame_idx).to(self.model.n_raw.dtype)
                        rendered = self.render_one(cam_idx, t_value)
                        total_l1 += l1_loss(rendered, target).item()
                        total_mse += mse_loss(rendered, target).item()
                        count += 1
        n = max(count, 1)
        avg_mse = total_mse / n
        avg_psnr = 10.0 * float(torch.log10(torch.tensor(max(avg_mse, 1e-12)).reciprocal()))
        return {"val_l1": total_l1 / n, "val_psnr": avg_psnr}

    def train(self, num_iters: Optional[int] = None, log_every: Optional[int] = None,
              callback: Optional[Callable[[int, dict], None]] = None) -> dict:
        """Run the training loop.

        callback(iter, info_dict) is called every log_every iters after logging,
        so downstream code can plot / save intermediate renders.
        """
        n = num_iters if num_iters is not None else self.config.num_iters
        le = log_every if log_every is not None else self.config.log_every

        running_loss = 0.0
        running_l1 = 0.0
        running_psnr = 0.0
        decay = self.config.lr_decay
        warmup_color = self.config.color_lr_warmup_iter
        # Snapshot color-group base LR(s) once. Covers both sh_degree==0
        # ("color") and sh_degree>0 ("sh_dc", "sh_rest").
        _color_group_names = ("color", "sh_dc", "sh_rest")
        base_lr_color: dict[str, float] = {
            g["name"]: g["lr"] for g in self.optimizer.param_groups
            if g["name"] in _color_group_names
        }
        for i in range(1, n + 1):
            # Log-linear LR schedule on geometric params (mirrors 3DGS).
            if decay < 1.0:
                t = min(i / max(n, 1), 1.0)
                scale = decay ** t                       # 1 → decay over training
                for group in self.optimizer.param_groups:
                    name = group["name"]
                    if name in self._base_lrs:
                        group["lr"] = self._base_lrs[name] * scale
            # #5.2 color-LR warmup: linear 0 -> base over `warmup_color` iters.
            # Applies to whichever color group(s) the optimizer has.
            if warmup_color > 0 and base_lr_color:
                ramp = min(i / warmup_color, 1.0)
                for group in self.optimizer.param_groups:
                    if group["name"] in base_lr_color:
                        group["lr"] = base_lr_color[group["name"]] * ramp
            # #2.1 pose-refinement warmup: hold pose LRs at 0 until
            # pose_warmup_iter, then snap to (lr_R, lr_t).
            if self.pose_dR is not None:
                p_w = self.config.pose_warmup_iter
                tgt_R = self.config.lr_pose_rot if i >= p_w else 0.0
                tgt_t = self.config.lr_pose_trans if i >= p_w else 0.0
                for group in self.optimizer.param_groups:
                    if group["name"] == "pose_dR":
                        group["lr"] = tgt_R
                    elif group["name"] == "pose_dt":
                        group["lr"] = tgt_t
            # #3.2 progressive Grassmann relaxation: scale lr_n 0 → base
            # over [start, end]. Idle if both 0.
            r_start = self.config.grassmann_relax_start
            r_end = self.config.grassmann_relax_end
            if r_end > 0 and r_end > r_start and "n" in self._base_lrs:
                if i < r_start:
                    n_scale = 0.0
                elif i >= r_end:
                    n_scale = 1.0
                else:
                    n_scale = (i - r_start) / max(r_end - r_start, 1)
                # Compose with decay schedule already applied above.
                base_n = self._base_lrs["n"]
                if decay < 1.0:
                    t = min(i / max(n, 1), 1.0)
                    base_n = base_n * (decay ** t)
                for group in self.optimizer.param_groups:
                    if group["name"] == "n":
                        group["lr"] = base_n * n_scale

            loss_val, l1_val, psnr_val = self.train_step(iter_num=i)
            running_loss += loss_val
            running_l1 += l1_val
            running_psnr += psnr_val

            if i % self.config.renormalize_every == 0:
                self.renormalize_manifolds()

            # #6.2 hard aspect-ratio clip on Σ_3D in-plane eigenvalues.
            if (self.config.max_aspect_ratio > 0
                    and i % self.config.aspect_clip_every == 0):
                clipped = self.model.clip_aspect_ratio_(self.config.max_aspect_ratio)
                if clipped > 0:
                    # Wipe Adam momentum on L_raw to avoid stale-direction kicks
                    # right after the projection.
                    for group in self.optimizer.param_groups:
                        if group["name"] == "L_raw":
                            for p in group["params"]:
                                state = self.optimizer.state.get(p, {})
                                if "exp_avg" in state:
                                    state["exp_avg"].zero_()
                                if "exp_avg_sq" in state:
                                    state["exp_avg_sq"].zero_()

            # Periodic opacity reset (Phase-A-correctness: addresses 32%-dead pathology).
            if (self.config.opacity_reset_every > 0
                    and i % self.config.opacity_reset_every == 0):
                with torch.no_grad():
                    self.model.opacity_logit.data.fill_(self.config.opacity_reset_logit)
                # Also wipe Adam state for the opacity_logit param so the new logit
                # is not immediately overridden by stale momentum.
                for group in self.optimizer.param_groups:
                    for p in group["params"]:
                        if p is self.model.opacity_logit:
                            state = self.optimizer.state.get(p, {})
                            if "exp_avg" in state:
                                state["exp_avg"].zero_()
                            if "exp_avg_sq" in state:
                                state["exp_avg_sq"].zero_()
                print(f"  [opacity reset @ iter {i}] all logits -> "
                      f"{self.config.opacity_reset_logit}")

            # Density control
            if (self.density_tracker is not None
                    and self.config.densify_every > 0
                    and self.config.densify_start <= i <= self.config.densify_stop
                    and i % self.config.densify_every == 0):
                stats = self.density_tracker.densify_and_prune(
                    self.config.density_config,
                )
                print(f"  [density @ iter {i:5d}] split={stats['split']:4d} "
                      f"tsplit={stats.get('tsplit', 0):3d} "
                      f"pruned={stats['pruned']:4d} N={stats['final_N']}")

            if i % le == 0:
                avg_loss = running_loss / le
                avg_l1 = running_l1 / le
                avg_psnr = running_psnr / le
                self.history["iter"].append(i)
                self.history["loss"].append(avg_loss)
                self.history["l1"].append(avg_l1)
                self.history["psnr"].append(avg_psnr)
                self.history["N"].append(self.model.N)
                info = {"iter": i, "loss": avg_loss, "l1": avg_l1,
                        "psnr": avg_psnr, "N": self.model.N}

                # Optional validation.
                if self.config.validation_every > 0 and (i % self.config.validation_every == 0):
                    val = self.validate()
                    info.update(val)

                print(f"  iter {i:5d}: loss={avg_loss:.4f}  l1={avg_l1:.4f}  "
                      f"psnr={avg_psnr:.2f}dB  N={self.model.N}"
                      + (f"  val_l1={info.get('val_l1', float('nan')):.4f}" if "val_l1" in info else "")
                      + (f"  val_psnr={info.get('val_psnr', float('nan')):.2f}dB" if "val_psnr" in info else ""))
                if callback is not None:
                    callback(i, info)

                running_loss = 0.0
                running_l1 = 0.0
                running_psnr = 0.0

        return self.history

    @classmethod
    def from_monocular_dataset(
        cls,
        model: TrainableGaussians,
        dataset,                                 # MonocularDataset (avoid import cycle)
        config: Optional[TrainerConfig] = None,
    ) -> "Trainer":
        """Build a monocular Trainer from a MonocularDataset.

        Couples the per-frame Camera and per-frame target image so each step
        samples a single frame: cam_idx == t_idx == frame_idx.
        """
        cfg = config or TrainerConfig()
        cfg.monocular = True
        if cfg.validation_frames is None:
            if dataset.val_indices:
                cfg.validation_frames = list(dataset.val_indices)
            elif cfg.validation_every > 0:
                # No val split shipped: refuse to silently report training-set
                # L1 as val_l1 (would be a fake-success signal). Disable val.
                import warnings
                warnings.warn(
                    "from_monocular_dataset: dataset has no val_indices and no "
                    "validation_frames override; disabling validation to avoid "
                    "reporting training-set loss as val_l1.",
                    RuntimeWarning, stacklevel=2,
                )
                cfg.validation_every = 0

        # Adapter: in monocular mode train_step calls get_frame(cam_idx=frame, t_idx=frame),
        # which calls self.frame_data(cam_idx, t_value). Discard the t_value -- we have
        # the frame index already in cam_idx.
        loader = dataset.frame_loader

        def adapter(cam_idx: int, t_value: float) -> Tensor:
            return loader(cam_idx)

        return cls(
            model=model,
            cameras=dataset.cameras_per_frame,
            frame_data=adapter,
            times=dataset.times.tolist(),
            H=dataset.H,
            W=dataset.W,
            config=cfg,
        )
