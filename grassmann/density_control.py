"""
Adaptive density control for the 3-plane (G(3,4)) projector parameterization.

Phase C of the monocular pivot. Replaces the legacy 2-plane DC that was
empirically net-negative under the rank-1 Σ_3D(t_0) constraint. The new
DC follows the reviewer's recipe:

  * Trigger: screen-space ‖∇μ_2d‖ accumulated across views (the standard
    3DGS trigger). Captured from the CUDA rasterizer's means2D dummy
    tensor via fast_rasterize's `means2d_capture` arg.
  * Split: stressed Gaussians with large spatial extent
    (λ_max(Σ_3D) > spatial_split_threshold) get replaced by two children
    offset along the major spatial axis of Σ_3D, with L_raw shrunk by phi
    (variance /= phi²).
  * Prune: opacity < opacity_threshold (default 1e-3, more conservative
    than standard 3DGS 0.005), λ_min(Σ_3D) < scale_min (collapsed disk),
    or λ_max(Σ_3D) > scale_max (runaway disk).
  * Adam-state migration: kept rows preserve exp_avg / exp_avg_sq;
    new rows zero-init.

We do NOT clone (same-position duplication) in this revision -- the
empirical evidence says split (with µ-shift) is what adds capacity in
the right place; clone is no-op without subsequent divergence which
the optimizer often doesn't drive.

Public API:
    tracker = DensityTracker(model, optimizer)
    ... in training loop ...
    tracker.accumulate(means2d)         # after .backward(), before optimizer.step()
    if iter % densify_every == 0:
        stats = tracker.densify_and_prune(config)

The `means2d` arg is the dummy tensor that fast_rasterize() appended to
the `means2d_capture` list (its .grad gives the screen-space mean
gradient per Gaussian). When None (toy fallback path), the tracker
silently skips accumulation -- DC is GPU-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from .gaussian import GaussianParams, compute_derived
from .trainable import TrainableGaussians


# Geometry/opacity per-Gaussian params common to both color paths.
_GEOMETRY_PARAMS = ("n_raw", "L_raw", "mu", "opacity_logit")


def _per_gaussian_param_names(model: TrainableGaussians) -> tuple[str, ...]:
    """Per-Gaussian nn.Parameter names actually present on the model.

    Geometry/opacity are always present; the color appearance path is either
    `color_logit` (sh_degree=0) or `sh_dc` + `sh_rest` (sh_degree>0). Adam
    state on whichever set is present must be kept in lockstep with row-axis
    splits/prunes.
    """
    if getattr(model, "sh_degree", 0) > 0:
        return _GEOMETRY_PARAMS + ("sh_dc", "sh_rest")
    return _GEOMETRY_PARAMS + ("color_logit",)


@dataclass
class DensityConfig:
    """Hyperparameters for adaptive density control (3-plane param)."""
    # Split trigger
    grad_threshold: float = 2e-4         # accumulated screen-space ‖∇μ_2d‖
                                         # above which a Gaussian is "stressed".
    spatial_split_threshold: float = 0.5  # λ_max(Σ_3D) above which a stressed
                                         # Gaussian SPLITS. In scene units²; ≈
                                         # (0.7 m std) at scale 1m.
    split_shrink_factor: float = 1.6     # children L_raw /= phi (variance /= phi²).
    split_offset_sigmas: float = 1.0     # split children placed at ±N·σ_max.
    max_split_per_event: int = 0         # cap on splits per cycle (0 = unlimited).

    # Prune thresholds
    opacity_threshold: float = 1e-3      # prune if sigmoid(opacity_logit) < this.
    scale_min: float = 1e-6              # prune if λ_min(Σ_3D) < this (collapsed).
    scale_max: float = 100.0             # prune if λ_max(Σ_3D) > this (runaway).
    # #4.2 temporal-axis split: trigger when stressed AND Σ_tt > threshold.
    # Children offset by ±N·sqrt(Σ_tt) along the time axis (μ_t component).
    # 0 disables.
    temporal_split_threshold: float = 0.0
    temporal_split_offset_sigmas: float = 1.0
    # #8.1 floater multi-view consensus pruning: drop Gaussians active in
    # fewer than `floater_min_views` distinct accumulate() calls within the
    # current density-control window. Activity = grad_norm > floater_eps.
    # 0 disables.
    floater_min_views: int = 0
    floater_eps: float = 1e-3


class DensityTracker:
    """Tracks per-Gaussian screen-space gradient stats and runs split + prune.

    Call `accumulate(means2d)` after each `loss.backward()`, then
    `densify_and_prune(config)` on a schedule from the trainer.
    """

    def __init__(
        self,
        model: TrainableGaussians,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        device = model.n_raw.device
        dtype = model.n_raw.dtype
        N = model.N
        self.grad_accum = torch.zeros(N, dtype=dtype, device=device)
        self.grad_counts = torch.zeros(N, dtype=torch.long, device=device)
        # #8.1 floater detection: count active iters (grad_norm > eps) per Gaussian.
        self.active_counts = torch.zeros(N, dtype=torch.long, device=device)

    def accumulate(self, means2d: Optional[Tensor]) -> None:
        """Record screen-space gradient norms per Gaussian.

        means2d: dummy tensor returned by `fast_rasterize` via the
            `means2d_capture` list. After backward(), means2d.grad is the
            screen-space gradient per Gaussian -- we accumulate its 2D
            norm. None (toy fallback) is a no-op.
        """
        if means2d is None or means2d.grad is None:
            return
        # First two columns are screen-space (u, v); third is depth-related.
        g_uv = means2d.grad[..., :2]
        mag = g_uv.norm(dim=-1)                                       # (N,)
        # Defensive: if N changed since init (split/prune), resize the accum.
        if mag.shape[0] != self.grad_accum.shape[0]:
            return
        self.grad_accum += mag.detach()
        self.grad_counts += 1
        # #8.1: track which Gaussians are 'active' (any meaningful screen grad).
        self.active_counts += (mag.detach() > 1e-9).long()

    def mean_grad(self) -> Tensor:
        counts = self.grad_counts.clamp_min(1)
        return self.grad_accum / counts.to(self.grad_accum.dtype)

    def reset(self) -> None:
        self.grad_accum.zero_()
        self.grad_counts.zero_()
        self.active_counts.zero_()

    # --- Adam-state-aware row mutations -----------------------------------

    def _migrate_optimizer_state(
        self,
        old_param: nn.Parameter,
        new_param: nn.Parameter,
        *,
        slice_idx: Optional[Tensor] = None,
        append_zero_count: int = 0,
    ) -> None:
        opt = self.optimizer
        if opt is None:
            return
        for group in opt.param_groups:
            for i, p in enumerate(group["params"]):
                if p is not old_param:
                    continue
                state = opt.state.pop(old_param, None)
                if state is not None:
                    new_state: dict = {}
                    if "step" in state:
                        new_state["step"] = state["step"]
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key not in state:
                            continue
                        t = state[key]
                        if slice_idx is not None:
                            new_t = t[slice_idx].contiguous()
                        elif append_zero_count > 0:
                            pad_shape = (append_zero_count,) + tuple(t.shape[1:])
                            pad = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
                            new_t = torch.cat([t, pad], dim=0).contiguous()
                        else:
                            new_t = t
                        new_state[key] = new_t
                    opt.state[new_param] = new_state
                group["params"][i] = new_param
                return

    def _keep_rows(self, keep_mask: Tensor) -> None:
        idx = torch.nonzero(keep_mask, as_tuple=True)[0]
        with torch.no_grad():
            for name in _per_gaussian_param_names(self.model):
                old = getattr(self.model, name)
                new = nn.Parameter(old.data[idx].contiguous())
                setattr(self.model, name, new)
                self._migrate_optimizer_state(old, new, slice_idx=idx)
        self.grad_accum = self.grad_accum[idx].contiguous()
        self.grad_counts = self.grad_counts[idx].contiguous()
        self.active_counts = self.active_counts[idx].contiguous()

    def _append_rows(self, new_data: dict[str, Tensor]) -> None:
        names = _per_gaussian_param_names(self.model)
        missing = [n for n in names if n not in new_data]
        if missing:
            raise KeyError(f"_append_rows missing tensors for: {missing}")
        first = new_data[names[0]]
        n_new = first.shape[0]
        with torch.no_grad():
            for name in names:
                old = getattr(self.model, name)
                new = nn.Parameter(torch.cat([old.data, new_data[name]], dim=0).contiguous())
                setattr(self.model, name, new)
                self._migrate_optimizer_state(old, new, append_zero_count=n_new)
        device = self.grad_accum.device; dtype = self.grad_accum.dtype
        self.grad_accum = torch.cat(
            [self.grad_accum, torch.zeros(n_new, dtype=dtype, device=device)]
        )
        self.grad_counts = torch.cat(
            [self.grad_counts, torch.zeros(n_new, dtype=torch.long, device=device)]
        )
        self.active_counts = torch.cat(
            [self.active_counts, torch.zeros(n_new, dtype=torch.long, device=device)]
        )

    # --- Σ_3D spectral helpers --------------------------------------------

    def _sigma_3d_eigs(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute (Σ_3D, eigvals, λ_min_nonzero, λ_max) per Gaussian.

        Σ_3D here is the *pre-time-conditioning* spatial block of Σ_4D --
        view- and t₀-independent (per the reviewer's recommendation). Used
        for split / prune size thresholds.
        """
        with torch.no_grad():
            params = self.model.forward()
            d = compute_derived(params)
            Sigma_3D = d.Sigma_3D                                  # (N, 3, 3) rank ≤ 3
            eigs = torch.linalg.eigvalsh(Sigma_3D)                 # (N, 3) ascending
            lam_min_nonzero = eigs[:, 1]
            lam_max = eigs[:, 2]
        return Sigma_3D, eigs, lam_min_nonzero, lam_max

    # --- Operations -------------------------------------------------------

    def prune(self, config: DensityConfig) -> int:
        with torch.no_grad():
            opacity = torch.sigmoid(self.model.opacity_logit)
            _, _, lam_min, lam_max = self._sigma_3d_eigs()
            low_op_mask = opacity < config.opacity_threshold
            collapsed_mask = lam_min < config.scale_min
            runaway_mask = lam_max > config.scale_max
            drop_mask = low_op_mask | collapsed_mask | runaway_mask
            # #8.1 floater multi-view consensus pruning.
            if config.floater_min_views > 0:
                # Only prune Gaussians that have actually been seen this window
                # (grad_counts > 0); zero-count Gaussians are not floaters,
                # they're just unsampled.
                seen = self.grad_counts > 0
                floaters = seen & (self.active_counts < config.floater_min_views)
                drop_mask = drop_mask | floaters
            keep_mask = ~drop_mask
            n_pruned = int(drop_mask.sum().item())
        if n_pruned > 0:
            self._keep_rows(keep_mask)
        return n_pruned

    def split(self, config: DensityConfig) -> int:
        """Split stressed + spatially-large Gaussians along their major Σ_3D axis."""
        grads = self.mean_grad()
        with torch.no_grad():
            Sigma_3D, _, _, lam_max = self._sigma_3d_eigs()
            stressed = grads > config.grad_threshold
            big = lam_max > config.spatial_split_threshold
            split_mask = stressed & big
            n_split = int(split_mask.sum().item())
            if n_split == 0:
                return 0
            if config.max_split_per_event > 0 and n_split > config.max_split_per_event:
                # Cap: keep top-k by accumulated grad among the eligible set.
                eligible_grads = torch.where(split_mask, grads, torch.full_like(grads, -1.0))
                topk = torch.topk(eligible_grads, config.max_split_per_event).indices
                split_mask = torch.zeros_like(split_mask)
                split_mask[topk] = True
                n_split = int(split_mask.sum().item())

            idx = torch.nonzero(split_mask, as_tuple=True)[0]
            phi = config.split_shrink_factor

            # Major spatial axis of Σ_3D in WORLD coords.
            S_split = Sigma_3D[idx]                                     # (k, 3, 3)
            eigvals, eigvecs = torch.linalg.eigh(S_split)
            major_dir = eigvecs[..., :, 2]                              # (k, 3)
            major_std = eigvals[..., 2].sqrt()                          # (k,)
            offset = config.split_offset_sigmas * major_std.unsqueeze(-1) * major_dir

            mu_par = self.model.mu.data[idx]                            # (k, 4)
            mu_t = mu_par[..., 0]
            mu_x = mu_par[..., 1:]
            mu_x_plus = mu_x + offset
            mu_x_minus = mu_x - offset

            mu_plus = torch.cat([mu_t.unsqueeze(-1), mu_x_plus], dim=-1)
            mu_minus = torch.cat([mu_t.unsqueeze(-1), mu_x_minus], dim=-1)

            n_raw_par = self.model.n_raw.data[idx]
            L_raw_par = self.model.L_raw.data[idx] / phi
            op_par = self.model.opacity_logit.data[idx]

            new_rows: dict[str, Tensor] = {
                "n_raw": torch.cat([n_raw_par, n_raw_par], dim=0),
                "L_raw": torch.cat([L_raw_par, L_raw_par], dim=0),
                "mu": torch.cat([mu_plus, mu_minus], dim=0),
                "opacity_logit": torch.cat([op_par, op_par], dim=0),
            }
            if self.model.sh_degree == 0:
                col_par = self.model.color_logit.data[idx]
                new_rows["color_logit"] = torch.cat([col_par, col_par], dim=0)
            else:
                sh_dc_par = self.model.sh_dc.data[idx]
                sh_rest_par = self.model.sh_rest.data[idx]
                new_rows["sh_dc"] = torch.cat([sh_dc_par, sh_dc_par], dim=0)
                new_rows["sh_rest"] = torch.cat([sh_rest_par, sh_rest_par], dim=0)

            self._append_rows(new_rows)

            keep_mask = torch.ones(self.model.N, dtype=torch.bool, device=mu_par.device)
            keep_mask[idx] = False
        self._keep_rows(keep_mask)
        return n_split

    def temporal_split(self, config: DensityConfig) -> int:
        """#4.2: split stressed Gaussians with large Σ_tt along the time axis.

        Triggered when grad > grad_threshold AND Σ_tt > temporal_split_threshold.
        Children offset by ±N·sqrt(Σ_tt) on μ_t; L_raw shrunk by split_shrink_factor.
        Disabled when temporal_split_threshold == 0.
        """
        if config.temporal_split_threshold <= 0.0:
            return 0
        grads = self.mean_grad()
        with torch.no_grad():
            params = self.model.forward()
            d = compute_derived(params)
            sigma_tt = d.Sigma_tt                                       # (N,)
            stressed = grads > config.grad_threshold
            big_t = sigma_tt > config.temporal_split_threshold
            split_mask = stressed & big_t
            n_split = int(split_mask.sum().item())
            if n_split == 0:
                return 0
            if config.max_split_per_event > 0 and n_split > config.max_split_per_event:
                eligible = torch.where(split_mask, grads, torch.full_like(grads, -1.0))
                topk = torch.topk(eligible, config.max_split_per_event).indices
                split_mask = torch.zeros_like(split_mask)
                split_mask[topk] = True
                n_split = int(split_mask.sum().item())

            idx = torch.nonzero(split_mask, as_tuple=True)[0]
            phi = config.split_shrink_factor
            sigma_t_std = sigma_tt[idx].clamp_min(1e-12).sqrt()         # (k,)
            offset = config.temporal_split_offset_sigmas * sigma_t_std

            mu_par = self.model.mu.data[idx]                            # (k, 4)
            mu_t = mu_par[..., 0]
            mu_x = mu_par[..., 1:]
            mu_t_plus = (mu_t + offset).unsqueeze(-1)
            mu_t_minus = (mu_t - offset).unsqueeze(-1)
            mu_plus = torch.cat([mu_t_plus, mu_x], dim=-1)
            mu_minus = torch.cat([mu_t_minus, mu_x], dim=-1)

            n_raw_par = self.model.n_raw.data[idx]
            L_raw_par = self.model.L_raw.data[idx] / phi
            op_par = self.model.opacity_logit.data[idx]

            new_rows: dict[str, Tensor] = {
                "n_raw": torch.cat([n_raw_par, n_raw_par], dim=0),
                "L_raw": torch.cat([L_raw_par, L_raw_par], dim=0),
                "mu": torch.cat([mu_plus, mu_minus], dim=0),
                "opacity_logit": torch.cat([op_par, op_par], dim=0),
            }
            if self.model.sh_degree == 0:
                col_par = self.model.color_logit.data[idx]
                new_rows["color_logit"] = torch.cat([col_par, col_par], dim=0)
            else:
                sh_dc_par = self.model.sh_dc.data[idx]
                sh_rest_par = self.model.sh_rest.data[idx]
                new_rows["sh_dc"] = torch.cat([sh_dc_par, sh_dc_par], dim=0)
                new_rows["sh_rest"] = torch.cat([sh_rest_par, sh_rest_par], dim=0)
            self._append_rows(new_rows)

            keep_mask = torch.ones(self.model.N, dtype=torch.bool, device=mu_par.device)
            keep_mask[idx] = False
        self._keep_rows(keep_mask)
        return n_split

    def densify_and_prune(self, config: DensityConfig) -> dict:
        """Apply split + prune in one pass. Returns stats dict."""
        n_split = self.split(config)
        n_tsplit = self.temporal_split(config)
        n_pruned = self.prune(config)
        self.reset()
        return {"split": n_split, "tsplit": n_tsplit,
                "pruned": n_pruned, "final_N": self.model.N}
