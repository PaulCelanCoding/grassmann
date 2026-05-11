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

    When use_quadratic_motion is enabled, c2 is also a per-Gaussian (N, 3)
    parameter and must follow row mutations in lockstep.
    """
    base = (
        _GEOMETRY_PARAMS + ("sh_dc", "sh_rest")
        if getattr(model, "sh_degree", 0) > 0
        else _GEOMETRY_PARAMS + ("color_logit",)
    )
    if getattr(model, "use_quadratic_motion", False) and getattr(model, "c2", None) is not None:
        base = base + ("c2",)
    return base


@dataclass
class DensityConfig:
    """Hyperparameters for adaptive density control (3-plane param)."""
    # Split trigger
    grad_threshold: float = 2e-4         # accumulated screen-space ‖∇μ_2d‖
                                         # above which a Gaussian is "stressed".
    spatial_split_threshold: float = 0.5  # λ_max(Σ_3D) above which a stressed
                                         # Gaussian SPLITS. In scene units²; ≈
                                         # (0.7 m std) at scale 1m.
    # H1 (geometric-only anisotropy split): if aspect_split_threshold > 0, ADDITIONALLY
    # mark a Gaussian for split when λ_max/λ_mid > aspect_split_threshold, regardless
    # of grad-magnitude. The legacy (stressed & big) trigger is still active; this OR's
    # on a new geometric trigger. Useful when scene anisotropy (cables, thin edges)
    # accumulates streaks that the hard SVD clip can only cap, not fragment.
    aspect_split_threshold: float = 0.0  # 0 disables
    split_shrink_factor: float = 1.6     # children L_raw /= phi (variance /= phi²).
    split_offset_sigmas: float = 1.0     # split children placed at ±N·σ_max.
    max_split_per_event: int = 0         # cap on splits per cycle (0 = unlimited).
    # Bug F: anisotropic L-shrinkage on split. The default (isotropic /φ)
    # shrinks ALL three Σ_3D eigenvalues uniformly per split, generating
    # tiny "zombie" Gaussians after a few cascading splits. When True,
    # only the major-axis direction (the one being split along) shrinks by
    # 1/φ; in-plane orthogonal directions are preserved.
    split_anisotropic_shrink: bool = False
    # Bug H: Kheradmand opacity-split correction. Children inherit
    # o_child = 1 − √(1 − o_parent) so alpha-blending is preserved 1→2.
    # Default off (matches original 3DGS: children share parent opacity).
    split_opacity_correction: bool = False
    # Bug H-inv: opposite direction — children brighter via o_child = √(o_parent).
    # Compensates for split children being smaller (less screen-space coverage)
    # by boosting per-Gaussian alpha. Mutually exclusive with split_opacity_correction.
    split_opacity_brighter: bool = False
    # Bug I: trigger eigvals on POST-Schur Σ_3D_t = Σ_3D − cc^T/σ_tt instead
    # of pre-Schur Σ_3D. The rasterizer sees Σ_3D_t (rank-2), so trigger
    # thresholds should match what is actually rendered. When True, eigs[:,0]
    # ≈ 0 (kernel), eigs[:,1] is the true in-plane λ_min, eigs[:,2] the in-plane λ_max.
    trigger_post_schur: bool = False
    # Bug G: nearest-neighbor merge on a cadence. Every merge_every iters,
    # drop the lower-opacity member of any pair whose spatial means are within
    # merge_distance AND whose normals satisfy |n^T n'| > merge_normal_cos;
    # combine their opacities on the survivor via 1 - (1-o_a)(1-o_b).
    # 0 disables.
    merge_every: int = 0
    merge_distance: float = 0.0
    merge_normal_cos: float = 0.95

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
    # μ_t out-of-bounds prune (trigger_audit Bug C): drop Gaussians whose
    # time-mean is outside the scene's [mu_t_min, mu_t_max] domain. NeRFies
    # normalizes t to [0, 1]; defaults pad ±0.05 to avoid pruning Gaussians
    # at the boundary that are actively contributing to t∈[0, 1].
    # ±1e10 = effectively disabled.
    mu_t_min: float = -1e10
    mu_t_max: float = 1e10

    # ---- #4.1 3DGS-MCMC (Kheradmand NeurIPS 2024) ----------------------------
    # When density_strategy == "mcmc" the densify_and_prune() pass replaces
    # heuristic split+prune with stochastic relocation: dead (low-opacity)
    # Gaussians are sampled to live ones, and opacity / L_raw are corrected
    # so total scene alpha is preserved (Eq. 8 of the paper). SGLD-style
    # noise on μ is then applied per training step (mcmc_noise_step) to
    # encourage exploration. "heuristic" (default) keeps the legacy path.
    density_strategy: str = "heuristic"      # "heuristic" | "mcmc" | "hybrid"
    # SGLD noise scale on μ_spatial (per training step). 0 disables.
    # Effective std = mcmc_noise_lr * |L_raw|_F * sigmoid(-k*(o - τ))  where
    # the sigmoid gate suppresses noise on alive Gaussians (τ = mcmc_noise_gate_thr).
    mcmc_noise_lr: float = 0.0
    mcmc_noise_after: int = 0                # iter at which noise activates
    mcmc_noise_gate_k: float = 100.0
    mcmc_noise_gate_thr: float = 0.005
    # MCMC relocation cadence — how often the densify_and_prune() pass is
    # called from the trainer is governed by trainer.densify_every. Within
    # each call, mcmc strategy uses these:
    mcmc_max_relocations_per_step: int = 0   # 0 = unlimited (all dead are relocated)
    # When density_strategy == "mcmc", optional pure-growth via duplication
    # of high-opacity Gaussians (independent of dead-replacement). 0 disables.
    mcmc_grow_per_step: int = 0


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
        # Bug G: count densify_and_prune() calls so merge() can fire on a coarser cadence.
        self._density_call_count = 0

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

    def _sigma_3d_eigs(
        self,
        post_schur: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute (Σ_3D, eigvals, λ_min, λ_max) per Gaussian.

        post_schur=False (default): pre-Schur Σ_3D = Σ_4D[1:,1:] (rank ≤ 3).
            eigs[:,1] is the MIDDLE eigenvalue (legacy "lam_min_nonzero").

        post_schur=True (Bug I): Σ_3D_t = Σ_3D − cc^T/σ_tt (rank-2, t-invariant).
            This is exactly what the rasterizer sees as the in-plane disk.
            eigs[:,0] ≈ 0 (kernel direction), eigs[:,1] is the true in-plane
            λ_min, eigs[:,2] is the in-plane λ_max.
        """
        with torch.no_grad():
            params = self.model.forward()
            d = compute_derived(params)
            Sigma_3D = d.Sigma_3D                                  # (N, 3, 3) rank ≤ 3
            if post_schur:
                # Use the same eps_schur logic as condition_on_time. For the
                # trigger we only need a stable clamp — hard floor is fine here.
                eps_schur = float(getattr(params, "eps_schur", 1e-8))
                Stt_pure = getattr(d, "_sigma_tt_pure", d.Sigma_tt)
                inv_Stt = 1.0 / Stt_pure.clamp_min(eps_schur)
                cw = d.c_world
                outer = cw.unsqueeze(-1) * cw.unsqueeze(-2)        # (N, 3, 3)
                Sigma_3D = Sigma_3D - inv_Stt.unsqueeze(-1).unsqueeze(-1) * outer
            eigs = torch.linalg.eigvalsh(Sigma_3D)                 # (N, 3) ascending
            lam_min_nonzero = eigs[:, 1]
            lam_max = eigs[:, 2]
        return Sigma_3D, eigs, lam_min_nonzero, lam_max

    # --- Operations -------------------------------------------------------

    def prune(self, config: DensityConfig) -> int:
        with torch.no_grad():
            opacity = torch.sigmoid(self.model.opacity_logit)
            _, _, lam_min, lam_max = self._sigma_3d_eigs(post_schur=config.trigger_post_schur)
            low_op_mask = opacity < config.opacity_threshold
            collapsed_mask = lam_min < config.scale_min
            runaway_mask = lam_max > config.scale_max
            drop_mask = low_op_mask | collapsed_mask | runaway_mask
            # Bug C: μ_t out-of-bounds prune (default disabled via ±1e10 sentinels).
            if config.mu_t_min > -1e9 or config.mu_t_max < 1e9:
                mu_t = self.model.mu.data[..., 0] if self.model.mu is not None \
                       else self.model.mu_time.data.squeeze(-1)
                oob_mask = (mu_t < config.mu_t_min) | (mu_t > config.mu_t_max)
                drop_mask = drop_mask | oob_mask
            # RCA diagnostic: opacity distribution (cheap, ~1 ms per call).
            qs = torch.quantile(
                opacity,
                torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99],
                             device=opacity.device, dtype=opacity.dtype),
            )
            n_low = int(low_op_mask.sum().item())
            n_col = int(collapsed_mask.sum().item())
            n_run = int(runaway_mask.sum().item())
            print(
                f"  opacity q1/q5/q50/q95/q99 = "
                f"{qs[0]:.4f}/{qs[1]:.4f}/{qs[2]:.4f}/{qs[3]:.4f}/{qs[4]:.4f} "
                f"| low_op={n_low} collapsed={n_col} runaway={n_run}"
            )
            # #8.1 floater multi-view consensus pruning.
            if config.floater_min_views > 0:
                # Only prune Gaussians that have actually been seen this window
                # (grad_counts > 0); zero-count Gaussians are not floaters,
                # they're just unsampled.
                seen = self.grad_counts > 0
                floaters = seen & (self.active_counts < config.floater_min_views)
                drop_mask = drop_mask | floaters
            keep_mask = ~drop_mask
            n_kept = int(keep_mask.sum().item())
            n_pruned = int(drop_mask.sum().item())
            # Safety: never prune to N=0 (rasterizer breaks at empty set).
            # Keep at least the top-1024 by opacity if mass-prune triggered.
            min_keep = 1024
            if n_kept < min_keep and n_pruned > 0:
                k = min(min_keep, opacity.numel())
                topk = torch.topk(opacity, k).indices
                save_mask = torch.zeros_like(drop_mask)
                save_mask[topk] = True
                drop_mask = drop_mask & ~save_mask
                keep_mask = ~drop_mask
                n_pruned = int(drop_mask.sum().item())
                print(f"  [prune-safety] mass-prune averted; kept top-{k} by opacity")
        if n_pruned > 0:
            self._keep_rows(keep_mask)
        return n_pruned

    def split(self, config: DensityConfig) -> int:
        """Split stressed + spatially-large Gaussians along their major Σ_3D axis."""
        grads = self.mean_grad()
        with torch.no_grad():
            Sigma_3D, _, _, lam_max = self._sigma_3d_eigs(post_schur=config.trigger_post_schur)
            stressed = grads > config.grad_threshold
            big = lam_max > config.spatial_split_threshold
            split_mask = stressed & big
            if config.aspect_split_threshold > 0.0:
                # H1: geometric-only anisotropy split. Use middle eigenvalue
                # (which post_schur would call λ_mid_inplane; pre-schur it is
                # the middle of rank-3 Σ_3D). λ_max/λ_mid is the aspect.
                _, eigs_full, _, _ = self._sigma_3d_eigs(post_schur=config.trigger_post_schur)
                aspect = eigs_full[:, 2] / eigs_full[:, 1].clamp_min(1e-12)
                aniso_mask = aspect > config.aspect_split_threshold
                split_mask = split_mask | aniso_mask
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
            L_raw_old = self.model.L_raw.data[idx]                      # (k, 4, 3)

            if config.split_anisotropic_shrink:
                # Bug F: shrink only the major spatial direction by 1/φ.
                # A_4 = blockdiag(1, A_3) where A_3 = I − (1 − 1/φ) u u^T
                # acts on the OUTPUT (row) space of L_plane: A_3 u = u/φ,
                # A_3 v = v for v ⊥ u. Then Σ_3D' = A_3 Σ_3D A_3^T scales
                # the major eigenvalue by 1/φ² and leaves the other two
                # untouched. We reconstruct L_raw from L_plane' so the
                # n-component of L_raw is preserved.
                eps = 1e-12
                n_norm = n_raw_par.norm(dim=-1, keepdim=True).clamp_min(eps)
                n_unit = n_raw_par / n_norm                             # (k, 4)
                nL = torch.einsum("ki,kij->kj", n_unit, L_raw_old)      # (k, 3)
                L_plane_old = L_raw_old - n_unit.unsqueeze(-1) * nL.unsqueeze(-2)
                # A_4: (k, 4, 4); only the spatial 3×3 block is non-identity.
                alpha = 1.0 - 1.0 / phi
                uuT = major_dir.unsqueeze(-1) * major_dir.unsqueeze(-2) # (k, 3, 3)
                I3 = torch.eye(3, dtype=L_raw_old.dtype, device=L_raw_old.device)
                A3 = I3 - alpha * uuT                                   # (k, 3, 3)
                # Apply A_3 to the spatial rows of L_plane only.
                L_plane_new = L_plane_old.clone()
                L_plane_new[..., 1:, :] = torch.einsum(
                    "krs,ksc->krc", A3, L_plane_old[..., 1:, :]
                )
                # Recover L_raw: L_raw' = L_plane' + n (n^T L_raw_old).
                L_raw_par = L_plane_new + n_unit.unsqueeze(-1) * nL.unsqueeze(-2)
            else:
                L_raw_par = L_raw_old / phi

            op_par_raw = self.model.opacity_logit.data[idx]
            if config.split_opacity_correction:
                # Bug H: Kheradmand 1→2 alpha-preserving opacity split.
                # o_child = 1 − √(1 − o_parent), so blending two children at
                # the same depth recovers o_parent.
                o_parent = torch.sigmoid(op_par_raw).clamp(1e-6, 1.0 - 1e-6)
                o_child = (1.0 - (1.0 - o_parent).sqrt()).clamp(1e-6, 1.0 - 1e-6)
                op_par = torch.log(o_child / (1.0 - o_child))
            elif config.split_opacity_brighter:
                # Bug H-inv: opposite direction — boost children to compensate
                # for smaller per-Gaussian screen-space coverage post-shrink.
                # o_child = √(o_parent): parent 0.5 → child 0.71; parent 0.9 → 0.95.
                o_parent = torch.sigmoid(op_par_raw).clamp(1e-6, 1.0 - 1e-6)
                o_child = o_parent.sqrt().clamp(1e-6, 1.0 - 1e-6)
                op_par = torch.log(o_child / (1.0 - o_child))
            else:
                op_par = op_par_raw

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
            if getattr(self.model, "use_quadratic_motion", False) and self.model.c2 is not None:
                c2_par = self.model.c2.data[idx]
                new_rows["c2"] = torch.cat([c2_par, c2_par], dim=0)

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
            L_raw_old = self.model.L_raw.data[idx]                      # (k, 4, 3)

            if config.split_anisotropic_shrink:
                # Bug F (temporal variant): shrink ONLY the time row of L_plane
                # by 1/φ. This scales Σ_tt → Σ_tt/φ², c_world → c_world/φ, and
                # leaves Σ_3D unchanged. Post-Schur Σ_3D_t = Σ_3D − cc^T/σ_tt
                # is also invariant: (c/φ)(c/φ)^T / (σ_tt/φ²) = cc^T/σ_tt.
                eps = 1e-12
                n_norm = n_raw_par.norm(dim=-1, keepdim=True).clamp_min(eps)
                n_unit = n_raw_par / n_norm
                nL = torch.einsum("ki,kij->kj", n_unit, L_raw_old)
                L_plane_old = L_raw_old - n_unit.unsqueeze(-1) * nL.unsqueeze(-2)
                L_plane_new = L_plane_old.clone()
                L_plane_new[..., 0, :] = L_plane_old[..., 0, :] / phi
                L_raw_par = L_plane_new + n_unit.unsqueeze(-1) * nL.unsqueeze(-2)
            else:
                L_raw_par = L_raw_old / phi

            op_par_raw = self.model.opacity_logit.data[idx]
            if config.split_opacity_correction:
                o_parent = torch.sigmoid(op_par_raw).clamp(1e-6, 1.0 - 1e-6)
                o_child = (1.0 - (1.0 - o_parent).sqrt()).clamp(1e-6, 1.0 - 1e-6)
                op_par = torch.log(o_child / (1.0 - o_child))
            elif config.split_opacity_brighter:
                o_parent = torch.sigmoid(op_par_raw).clamp(1e-6, 1.0 - 1e-6)
                o_child = o_parent.sqrt().clamp(1e-6, 1.0 - 1e-6)
                op_par = torch.log(o_child / (1.0 - o_child))
            else:
                op_par = op_par_raw

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

    # --- #4.1 MCMC: Adam-state row-zero and relocation --------------------

    def _zero_optimizer_state_rows(self, param: nn.Parameter, idx: Tensor) -> None:
        """Zero exp_avg / exp_avg_sq at given row indices for `param`. No-op when
        no optimizer or no state for that param. `idx` is a 1D LongTensor.
        """
        opt = self.optimizer
        if opt is None or idx.numel() == 0:
            return
        state = opt.state.get(param, None)
        if state is None:
            return
        for key in ("exp_avg", "exp_avg_sq"):
            t = state.get(key, None)
            if t is None or t.shape[0] != param.shape[0]:
                continue
            t.index_fill_(0, idx, 0.0)

    def mcmc_relocate(self, config: DensityConfig) -> dict:
        """3DGS-MCMC relocation step (Kheradmand NeurIPS 2024).

        Dead (low-opacity) Gaussians are reassigned to destinations sampled
        from the live population with categorical(opacity[live]). Opacity
        and L_raw of the destination AND of the relocated copies are
        corrected so that the destination's total contribution under
        alpha-blending is preserved:

            o_new   = 1 − (1 − o_old)^(1/(k+1))
            L_new   = L_old / sqrt(k+1)

        where k+1 = (1 destination + k newcomers landing on it).

        The relocated dead rows inherit n_raw, μ, color/SH from the
        destination; their Adam state is reset to zero. Live rows that
        received copies have only their opacity_logit and L_raw modified;
        their Adam state on those tensors is also zeroed (the abrupt
        scale change makes prior momentum stale).

        Returns: {"relocated": k, "final_N": N (unchanged)}.
        """
        with torch.no_grad():
            opacity = torch.sigmoid(self.model.opacity_logit)            # (N,)
            dead_mask = opacity < config.opacity_threshold
            n_dead = int(dead_mask.sum().item())
            if n_dead == 0:
                return {"relocated": 0, "final_N": self.model.N}
            live_mask = ~dead_mask
            n_live = int(live_mask.sum().item())
            if n_live == 0:
                return {"relocated": 0, "final_N": self.model.N}

            cap = config.mcmc_max_relocations_per_step
            if cap > 0 and n_dead > cap:
                # Pick the lowest-opacity dead ones to relocate first.
                dead_idx_all = torch.nonzero(dead_mask, as_tuple=True)[0]
                dead_op = opacity[dead_idx_all]
                topk_local = torch.topk(-dead_op, cap).indices
                dead_idx = dead_idx_all[topk_local]
                n_dead = cap
            else:
                dead_idx = torch.nonzero(dead_mask, as_tuple=True)[0]

            live_idx = torch.nonzero(live_mask, as_tuple=True)[0]
            live_op = opacity[live_idx].to(torch.float32)                # multinomial wants fp32+

            # Sample destinations. 0-prob safety: clamp_min so multinomial doesn't NaN.
            probs = live_op.clamp_min(1e-12)
            dest_local = torch.multinomial(probs, n_dead, replacement=True)  # (n_dead,)
            # Count copies per destination (by live position).
            n_copies = torch.bincount(dest_local, minlength=n_live)          # (n_live,)
            kp1 = (n_copies + 1).to(self.model.opacity_logit.dtype)          # (n_live,)
            sqrt_kp1 = kp1.sqrt()                                            # (n_live,)

            # --- Step A: correct live destinations in-place ---
            # opacity_new = 1 − (1 − opacity)^(1/kp1); only those with k>0 change,
            # but applying to all is a no-op when kp1=1 → leave them.
            live_op_new = 1.0 - (1.0 - live_op.to(opacity.dtype)) ** (1.0 / kp1)
            live_op_new = live_op_new.clamp(1e-6, 1.0 - 1e-6)
            live_logit_new = torch.log(live_op_new / (1.0 - live_op_new))
            self.model.opacity_logit.data[live_idx] = live_logit_new

            self.model.L_raw.data[live_idx] = (
                self.model.L_raw.data[live_idx] / sqrt_kp1.view(-1, 1, 1)
            )

            # Zero Adam state on the LIVE rows that actually changed (k > 0).
            changed_live_local = torch.nonzero(n_copies > 0, as_tuple=True)[0]
            changed_live_global = live_idx[changed_live_local]
            self._zero_optimizer_state_rows(self.model.opacity_logit, changed_live_global)
            self._zero_optimizer_state_rows(self.model.L_raw, changed_live_global)

            # --- Step B: overwrite dead rows with corrected destinations ---
            dest_global = live_idx[dest_local]                               # (n_dead,)
            self.model.n_raw.data[dead_idx] = self.model.n_raw.data[dest_global]
            self.model.L_raw.data[dead_idx] = self.model.L_raw.data[dest_global]
            self.model.mu.data[dead_idx] = self.model.mu.data[dest_global]
            self.model.opacity_logit.data[dead_idx] = self.model.opacity_logit.data[dest_global]
            if self.model.sh_degree == 0:
                self.model.color_logit.data[dead_idx] = self.model.color_logit.data[dest_global]
            else:
                self.model.sh_dc.data[dead_idx] = self.model.sh_dc.data[dest_global]
                self.model.sh_rest.data[dead_idx] = self.model.sh_rest.data[dest_global]

            # Zero Adam state on dead rows for ALL per-Gaussian params (clean restart).
            for name in _per_gaussian_param_names(self.model):
                self._zero_optimizer_state_rows(getattr(self.model, name), dead_idx)

        return {"relocated": int(n_dead), "final_N": self.model.N}

    def mcmc_noise_step(self, config: DensityConfig, iter_num: int = 0) -> None:
        """Per-iter SGLD-like noise on μ_spatial, gated by opacity.

        std = mcmc_noise_lr * |L_raw|_F  (per-Gaussian scale)
        gate = sigmoid(-k * (opacity - τ))   ≈ 1 if dead, ≈ 0 if alive.

        Only μ_spatial (last 3 components of μ) is perturbed — μ_time is
        kept static so the temporal-axis split stays meaningful. n_raw and
        L_raw are not perturbed (matches Kheradmand Eq. 9, which adds noise
        only on means).
        """
        if config.mcmc_noise_lr <= 0.0 or iter_num < config.mcmc_noise_after:
            return
        with torch.no_grad():
            opacity = torch.sigmoid(self.model.opacity_logit)            # (N,)
            gate = torch.sigmoid(
                -config.mcmc_noise_gate_k * (opacity - config.mcmc_noise_gate_thr)
            )                                                            # (N,)
            # Per-Gaussian scale proxy: Frobenius norm of L_raw.
            L = self.model.L_raw.data
            scale = L.flatten(1).norm(dim=-1).clamp_min(1e-8)            # (N,)
            std = config.mcmc_noise_lr * scale * gate                    # (N,)
            if self.model.mu_lr_split:
                noise = torch.randn_like(self.model.mu_spatial.data)     # (N, 3)
                self.model.mu_spatial.data.add_(noise * std.unsqueeze(-1))
            else:
                noise = torch.randn(
                    self.model.mu.shape[0], 3,
                    dtype=self.model.mu.dtype, device=self.model.mu.device,
                )                                                        # (N, 3)
                # μ layout: [t, x, y, z] → perturb indices 1:.
                self.model.mu.data[..., 1:].add_(noise * std.unsqueeze(-1))

    # --- Bug G: nearest-neighbor merge ------------------------------------

    def merge(self, config: DensityConfig) -> int:
        """Drop the lower-opacity member of any pair within merge_distance
        whose normals satisfy |n^T n'| > merge_normal_cos; combine their
        opacities on the survivor via o_new = 1 − (1−o_a)(1−o_b).

        Pairwise distances are computed in chunks of 1024 to bound memory.
        For N=50k this is ~50 × 1024 × 50k × 4B = 10 GB peak — too much for
        L4. Chunk to 512 → 5 GB; 256 → 2.5 GB. Use 512.
        """
        if config.merge_distance <= 0.0:
            return 0
        with torch.no_grad():
            N = int(self.model.N)
            if N < 2:
                return 0
            mu_spatial = self.model.mu.data[..., 1:]                    # (N, 3)
            n_norm = self.model.n_raw.data.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            n_unit = self.model.n_raw.data / n_norm                     # (N, 4)
            opacity = torch.sigmoid(self.model.opacity_logit)           # (N,)
            drop = torch.zeros(N, dtype=torch.bool, device=mu_spatial.device)

            thr = config.merge_distance
            cos_thr = config.merge_normal_cos
            chunk = 512
            arange_full = torch.arange(N, device=mu_spatial.device)
            for start in range(0, N, chunk):
                end = min(start + chunk, N)
                chunk_mu = mu_spatial[start:end]                        # (c, 3)
                chunk_n = n_unit[start:end]                             # (c, 4)
                chunk_op = opacity[start:end]                           # (c,)
                d = torch.cdist(chunk_mu, mu_spatial)                   # (c, N)
                # mask self
                row_idx = torch.arange(end - start, device=d.device)
                col_idx = arange_full[start:end]
                d[row_idx, col_idx] = float("inf")
                # mask already-dropped
                d[:, drop] = float("inf")
                min_d, nn_idx = d.min(dim=1)                            # (c,)
                # normal alignment
                n_dot = (chunk_n * n_unit[nn_idx]).sum(-1).abs()        # (c,)
                close = (min_d < thr) & (n_dot > cos_thr)
                # absorb when chunk_op < nn_op AND survivor not already dropped
                nn_op = opacity[nn_idx]
                survivor_alive = ~drop[nn_idx]
                absorb = close & (chunk_op < nn_op) & survivor_alive & ~drop[start:end]
                ab_local = torch.nonzero(absorb, as_tuple=True)[0]
                if ab_local.numel() == 0:
                    continue
                # Sequential application — small set, fine in Python.
                for li in ab_local.tolist():
                    gi = start + li
                    tgt = int(nn_idx[li].item())
                    if drop[tgt]:
                        continue
                    o_src = float(opacity[gi].item())
                    o_dst = float(opacity[tgt].item())
                    o_new = 1.0 - (1.0 - o_src) * (1.0 - o_dst)
                    o_new = min(max(o_new, 1e-6), 1.0 - 1e-6)
                    self.model.opacity_logit.data[tgt] = torch.log(
                        torch.tensor(o_new / (1.0 - o_new),
                                     dtype=self.model.opacity_logit.dtype,
                                     device=self.model.opacity_logit.device)
                    )
                    drop[gi] = True

            n_merged = int(drop.sum().item())
        if n_merged > 0:
            self._keep_rows(~drop)
        return n_merged

    # ----------------------------------------------------------------------

    def densify_and_prune(self, config: DensityConfig) -> dict:
        """Apply the configured density-control pass. Returns stats dict.

        density_strategy:
          - "heuristic" (default): split + temporal_split + prune (legacy).
          - "mcmc": stochastic relocation per Kheradmand 2024 (no growth).
          - "hybrid": heuristic split + temporal_split, then mcmc_relocate
                     replaces low-opacity prune (dead → live with correction).
                     The collapsed/runaway prune still runs to drop pathological
                     Gaussians the relocator can't fix.
        """
        if config.density_strategy == "mcmc":
            stats = self.mcmc_relocate(config)
            self.reset()
            stats.setdefault("split", 0)
            stats.setdefault("tsplit", 0)
            stats.setdefault("pruned", 0)
            return stats
        if config.density_strategy == "hybrid":
            n_split = self.split(config)
            n_tsplit = self.temporal_split(config)
            relo = self.mcmc_relocate(config)             # dead → live
            # After relocation, every former-dead row inherits opacity from a
            # live destination, so prune.opacity_threshold no longer matches
            # them. The remaining prune call therefore only catches
            # collapsed/runaway Σ_3D pathologies.
            n_pruned = self.prune(config)
            self.reset()
            return {"split": n_split, "tsplit": n_tsplit,
                    "pruned": n_pruned,
                    "relocated": relo.get("relocated", 0),
                    "final_N": self.model.N}
        n_split = self.split(config)
        n_tsplit = self.temporal_split(config)
        n_pruned = self.prune(config)
        n_merged = 0
        self._density_call_count += 1
        if (config.merge_every > 0
                and config.merge_distance > 0.0
                and self._density_call_count % config.merge_every == 0):
            n_merged = self.merge(config)
        self.reset()
        return {"split": n_split, "tsplit": n_tsplit,
                "pruned": n_pruned, "merged": n_merged,
                "final_N": self.model.N}
