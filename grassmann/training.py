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
from torch import Tensor

from .gaussian import compute_derived, condition_on_time
from .losses import l1_loss, photometric_loss, LPIPSLoss
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
    # Background color for rendering
    background: Tensor = field(default_factory=lambda: torch.tensor([0.05, 0.05, 0.1]))
    # Validation
    validation_every: int = 0          # 0 = disabled
    validation_cams: Optional[list[int]] = None  # which camera indices to evaluate on
    validation_times: Optional[list[float]] = None  # which times
    validation_frames: Optional[list[int]] = None  # monocular: which frame indices
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
        )
        self.optimizer = self._build_opt(model)

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

        self.history: dict[str, list] = {"iter": [], "loss": [], "l1": [], "N": []}

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

    def render_one(self, cam_idx: int, t_value: float) -> Tensor:
        """Render the current model from camera cam_idx at time t_value."""
        params = self.model.forward()
        bg = self.config.background.to(dtype=params.color.dtype, device=params.color.device)
        if self.config.use_fast_rasterizer and fast_available() and params.n.is_cuda:
            fc = self.config.fast_raster_config or FastRasterConfig()
            return fast_rasterize(
                params, t_value, self.cameras[cam_idx], self.H, self.W,
                background=bg, config=fc,
            )
        # Fallback: toy rasterizer path (also used when no GPU or no extension).
        derived = compute_derived(params)
        tc = condition_on_time(params, derived, t_value)
        sg = project_to_screen(params, tc, self.cameras[cam_idx])
        return rasterize(sg, H=self.H, W=self.W, background=bg)

    def train_step(self) -> tuple[float, float]:
        """One stochastic training step. Returns (total loss, L1 component)."""
        # Sample a frame.
        if self.config.monocular:
            # Monocular: one camera per frame, sampled together. self.K must equal self.T.
            t_idx = torch.randint(0, self.T, (1,)).item()
            cam_idx = t_idx
        else:
            cam_idx = torch.randint(0, self.K, (1,)).item()
            t_idx = torch.randint(0, self.T, (1,)).item()
        t_value = self.times[t_idx]

        # Target and rendered.
        target = self.get_frame(cam_idx, t_idx).to(self.model.n_raw.dtype)
        rendered = self.render_one(cam_idx, t_value)

        # Loss.
        loss = photometric_loss(
            rendered, target,
            lambda_l1=self.config.lambda_l1,
            lambda_structural=self.config.lambda_structural,
            lpips_fn=self.lpips_fn,
            lambda_lpips=self.config.lambda_lpips,
        )
        l1_val = l1_loss(rendered, target).item()

        # Backprop.
        self.optimizer.zero_grad()
        loss.backward()

        # Density-control gradient accumulation (reads .grad BEFORE optimizer.step()).
        if self.density_tracker is not None:
            self.density_tracker.accumulate()

        self.optimizer.step()

        return loss.item(), l1_val

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
                        count += 1
        return {"val_l1": total_l1 / max(count, 1)}

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
        for i in range(1, n + 1):
            loss_val, l1_val = self.train_step()
            running_loss += loss_val
            running_l1 += l1_val

            if i % self.config.renormalize_every == 0:
                self.renormalize_manifolds()

            # Density control
            if (self.density_tracker is not None
                    and self.config.densify_every > 0
                    and self.config.densify_start <= i <= self.config.densify_stop
                    and i % self.config.densify_every == 0):
                self.optimizer, stats = self.density_tracker.densify_and_prune(
                    self.config.density_config,
                )
                print(f"  [density @ iter {i:5d}] pruned={stats['pruned']:4d} "
                      f"cloned={stats['cloned']:4d} split={stats['split']:4d} "
                      f"N={stats['final_N']}")

            if i % le == 0:
                avg_loss = running_loss / le
                avg_l1 = running_l1 / le
                self.history["iter"].append(i)
                self.history["loss"].append(avg_loss)
                self.history["l1"].append(avg_l1)
                self.history["N"].append(self.model.N)
                info = {"iter": i, "loss": avg_loss, "l1": avg_l1, "N": self.model.N}

                # Optional validation.
                if self.config.validation_every > 0 and (i % self.config.validation_every == 0):
                    val = self.validate()
                    info.update(val)

                print(f"  iter {i:5d}: loss={avg_loss:.4f}  l1={avg_l1:.4f}  N={self.model.N}"
                      + (f"  val_l1={info.get('val_l1', float('nan')):.4f}" if "val_l1" in info else ""))
                if callback is not None:
                    callback(i, info)

                running_loss = 0.0
                running_l1 = 0.0

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
