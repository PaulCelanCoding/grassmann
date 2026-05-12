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

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import torch
from torch import Tensor, nn


class _PhaseTimer:
    """CUDA-synced wall-time accumulator for a named phase.

    Used only when TrainerConfig.profile_breakdown=True; otherwise __enter__
    and __exit__ are no-ops so the production path pays zero overhead.
    """
    __slots__ = ("trainer", "key", "active", "_t0")

    def __init__(self, trainer, key: str, active: bool):
        self.trainer = trainer
        self.key = key
        self.active = active

    def __enter__(self):
        if self.active:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if not self.active:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - self._t0
        self.trainer._phase_times[self.key] = (
            self.trainer._phase_times.get(self.key, 0.0) + dt
        )
        self.trainer._phase_counts[self.key] = (
            self.trainer._phase_counts.get(self.key, 0) + 1
        )

from .gaussian import compute_derived, condition_on_time
from .losses import l1_loss, mse_loss, photometric_loss, LPIPSLoss
from .projection import Camera
from .rasterizer import project_to_screen, rasterize
from .trainable import TrainableGaussians, build_optimizer
from .density_control import DensityTracker, DensityConfig
from .fast_rasterizer import fast_rasterize, FastRasterConfig, is_available as fast_available


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
    # Hard cap on aspect ratio λ_max/λ_min of Σ_3D in-plane eigenvalues
    # (equivalently (s_max/s_min)² of P_n L_raw singular values). 0 disables.
    # Applied every `aspect_clip_every` iters as a no-grad SVD-based projection.
    max_aspect_ratio: float = 0.0
    aspect_clip_every: int = 100
    # Background color for rendering
    background: Tensor = field(default_factory=lambda: torch.tensor([0.05, 0.05, 0.1]))
    # At each train_step, replace the constant `background` with a
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
    # Fast (CUDA) rasterizer
    use_fast_rasterizer: bool = False  # True -> use CUDA diff-gaussian-rasterization if available
    fast_raster_config: Optional[FastRasterConfig] = None  # defaults built if None
    # Sampling mode
    monocular: bool = False  # True -> cameras list is per-frame; t_idx == cam_idx; one sample per step
    # Static baseline: disable time conditioning entirely (Schur step skipped,
    # w_t=1 always). Used to measure the "static-3DGS-on-monocular-bundle" floor
    # within the same pipeline. The gap between this and the full temporal run
    # quantifies the value of time conditioning.
    static_baseline: bool = False
    # Correctness penalties (investigation found 32% dead, 30% high-aniso,
    # 7% rank-1-collapsed Gaussians in unpenalized runs):
    #   lambda_frob: Frobenius-norm penalty on L_raw -- prevents the optimizer from
    #     soft-collapsing rank by routing capacity into the n̂ direction (which the
    #     projector annihilates). Recommended ~1e-4.
    #   opacity_reset_every: every N iters, reset opacity_logit to opacity_reset_logit
    #     (mirrors the standard 3DGS opacity-reset trick). 0 disables.
    #   opacity_reset_logit: target logit for the reset (sigmoid(-5)≈0.007).
    lambda_frob: float = 0.0
    opacity_reset_every: int = 0
    opacity_reset_logit: float = -5.0
    # Progressive Grassmann relaxation: scale lr_n from 0 → base over
    # [grassmann_relax_start, grassmann_relax_end]. Use with init_strategy=
    # spatial_slice (n=e₀ at init) so the geometry settles in the static-3DGS
    # regime before n is allowed to tilt. 0/0 disables.
    grassmann_relax_start: int = 0
    grassmann_relax_end: int = 0
    # Structural-loss kind: 'boxstats' (legacy 7x7 local-mean+var matcher) or
    # 'ssim' (1 - SSIM Gaussian-windowed, matches 3DGS DSSIM term). Only
    # active when lambda_structural > 0.
    structural_kind: str = "boxstats"
    # Profiling: when True, train_step + train() bracket each phase with
    # CUDA-synced perf_counter() and dump per-phase ms/iter at log_every.
    # First profile_warmup_iters iters are discarded (JIT, allocator warmup).
    # Off by default; bit-identical to non-profile runs when off.
    profile_breakdown: bool = False
    profile_warmup_iters: int = 200


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
        # than rebuilding from scratch.
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

        self.current_iter: int = 0
        self.history: dict[str, list] = {"iter": [], "loss": [], "l1": [], "psnr": [], "N": []}
        # Profiling state (used only when config.profile_breakdown).
        self._phase_times: dict[str, float] = {}
        self._phase_counts: dict[str, int] = {}
        self._profile_iter_start: int = 0

    def _phase(self, key: str):
        return _PhaseTimer(self, key, self.config.profile_breakdown)

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
        """No-op pass-through (pose-refinement was removed)."""
        return self.cameras[cam_idx]

    def render_one(self, cam_idx: int, t_value: float,
                   means2d_capture: Optional[list] = None,
                   bg_override: Optional[Tensor] = None) -> Tensor:
        """Render the current model from camera cam_idx at time t_value.

        When `means2d_capture` is a list, the means2D dummy tensor (which gets
        screen-space gradients from the CUDA kernel after backward) is appended
        to it. Used by DensityTracker for the screen-space ‖∇μ_2d‖ trigger.

        bg_override: optional (3,) tensor that replaces self.config.background
        for this call only — used by random_background during train_step.
        """
        params = self.model.forward()
        bg_src = bg_override if bg_override is not None else self.config.background
        bg = bg_src.to(dtype=params.color.dtype, device=params.color.device)

        if self.config.use_fast_rasterizer and fast_available() and params.n.is_cuda:
            fc = self.config.fast_raster_config or FastRasterConfig()
            return fast_rasterize(
                params, t_value, self._perturbed_camera(cam_idx), self.H, self.W,
                background=bg, config=fc,
                static_baseline=self.config.static_baseline,
                means2d_capture=means2d_capture,
            )
        # Fallback: toy rasterizer path (also used when no GPU or no extension).
        if means2d_capture is not None:
            means2d_capture.append(None)
        derived = compute_derived(params)
        tc = condition_on_time(params, derived, t_value, static=self.config.static_baseline)
        sg = project_to_screen(params, tc, self._perturbed_camera(cam_idx))
        return rasterize(sg, H=self.H, W=self.W, background=bg)

    def train_step(self, iter_num: int = 0) -> tuple[float, float, float]:
        """One stochastic training step. Returns (total loss, L1, PSNR_dB)."""
        self.current_iter = iter_num
        # Sample a frame.
        with self._phase("data"):
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

            # Target. If DC is enabled, capture the means2D dummy
            # tensor so the tracker can read screen-space gradients post-backward.
            means2d_capture: Optional[list] = [] if self.density_tracker is not None else None
            target = self.get_frame(cam_idx, t_idx).to(self.model.n_raw.dtype)

        with self._phase("forward_render"):
            # Random background during training only.
            bg_override = None
            if self.config.random_background:
                bg_override = torch.rand(3, dtype=self.model.n_raw.dtype,
                                         device=self.model.n_raw.device)
            rendered = self.render_one(cam_idx, t_value,
                                       means2d_capture=means2d_capture,
                                       bg_override=bg_override)

        with self._phase("loss"):
            loss = photometric_loss(
                rendered, target,
                lambda_l1=self.config.lambda_l1,
                lambda_structural=self.config.lambda_structural,
                structural_kind=self.config.structural_kind,
                lpips_fn=self.lpips_fn,
                lambda_lpips=self.config.lambda_lpips,
            )

            # Correctness penalties.
            if self.config.lambda_frob > 0.0:
                # Mean-squared L_raw entries; targets the soft-rank-collapse pathology
                # where the optimizer routes capacity into the projector's null direction.
                loss = loss + self.config.lambda_frob * (self.model.L_raw ** 2).mean()
        with self._phase("log_metrics"):
            with torch.no_grad():
                l1_val = l1_loss(rendered, target).item()
                mse_val = mse_loss(rendered, target).item()
                psnr_val = 10.0 * float(torch.log10(torch.tensor(max(mse_val, 1e-12)).reciprocal()))

        # Backprop.
        with self._phase("backward"):
            self.optimizer.zero_grad()
            loss.backward()

        # Density-control gradient accumulation (reads .grad BEFORE optimizer.step()).
        with self._phase("accum"):
            if self.density_tracker is not None:
                means2d = means2d_capture[0] if means2d_capture else None
                self.density_tracker.accumulate(means2d)

        with self._phase("opt_step"):
            self.optimizer.step()
            loss_item = loss.item()

        return loss_item, l1_val, psnr_val

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

    def _print_profile_breakdown(self, current_iter: int) -> None:
        """Dump per-phase timing accumulated since warmup ended.

        ms/iter is amortized over (current_iter - warmup_end). ms/event divides
        each phase's total by its own call count — useful for sparse phases
        (e.g. 'density' fires every densify_every iters).
        """
        iters_in = current_iter - self._profile_iter_start
        if iters_in <= 0:
            return
        print(f"  [profile] phase breakdown over {iters_in} iters (since warmup):")
        ordered = [
            "sched", "data", "forward_render", "loss", "loss_extra",
            "log_metrics", "backward", "accum", "opt_step",
            "post_step", "density",
        ]
        total_ms = 0.0
        for k in ordered:
            if k not in self._phase_times:
                continue
            t = self._phase_times[k]
            n_calls = self._phase_counts.get(k, 1)
            ms_per_iter = 1000.0 * t / iters_in
            ms_per_event = 1000.0 * t / n_calls
            total_ms += ms_per_iter
            print(f"    {k:14s}: {ms_per_iter:7.3f} ms/iter   "
                  f"({n_calls:5d} calls, {ms_per_event:7.3f} ms/call)")
        print(f"    {'TOTAL':14s}: {total_ms:7.3f} ms/iter  (sum of phases)")

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
        # Profiling: track wall-clock + reset-iter so we can amortize correctly.
        for i in range(1, n + 1):
            # Reset timing accumulators after warmup so steady-state stats aren't
            # polluted by CUDA-JIT / allocator warmup costs.
            if (self.config.profile_breakdown
                    and i == self.config.profile_warmup_iters + 1):
                self._phase_times.clear()
                self._phase_counts.clear()
                self._profile_iter_start = i - 1
                print(f"  [profile] warmup done @ iter {i - 1}; timers reset", flush=True)

            with self._phase("sched"):
                # Log-linear LR schedule on geometric params (mirrors 3DGS).
                if decay < 1.0:
                    t = min(i / max(n, 1), 1.0)
                    scale = decay ** t                       # 1 → decay over training
                    for group in self.optimizer.param_groups:
                        name = group["name"]
                        if name in self._base_lrs:
                            group["lr"] = self._base_lrs[name] * scale
                # Progressive Grassmann relaxation: scale lr_n 0 → base
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

            with self._phase("post_step"):
                if i % self.config.renormalize_every == 0:
                    self.renormalize_manifolds()

                # Hard aspect-ratio clip on Σ_3D in-plane eigenvalues.
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

                # Periodic opacity reset (addresses the dead-Gaussian pathology).
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

            # Density control (timed separately — fires every densify_every iters).
            if (self.density_tracker is not None
                    and self.config.densify_every > 0
                    and self.config.densify_start <= i <= self.config.densify_stop
                    and i % self.config.densify_every == 0):
                with self._phase("density"):
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

                if (self.config.profile_breakdown
                        and i > self.config.profile_warmup_iters):
                    self._print_profile_breakdown(current_iter=i)

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
