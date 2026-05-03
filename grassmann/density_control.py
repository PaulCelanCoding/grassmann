"""
Adaptive density control for Grassmann Gaussians.

Follows the standard 3DGS pattern (Kerbl et al., SIGGRAPH 2023) adapted to
our parameterization, plus the Grassmann paper's §3.5 suggestions:

  1. Prune: remove Gaussians with
       - low opacity (< opacity_threshold): capacity waste
       - small scale (Sigma_k eigenvalues tiny): collapsed splat
       - very large scale (eigenvalues huge): runaway splat that covers the whole image
       - too large Sigma_tt * (1+c)/2  (temporal extent collapses — a degenerate zero-time-extent Gaussian).

  2. Clone: duplicate a Gaussian if
       - its 2D projected-mean gradient norm (accumulated across iterations)
         is above a threshold, AND
       - its spatial scale is small (under-reconstructed small feature)

  3. Split: replace a Gaussian by two smaller ones if
       - its 2D projected-mean gradient norm is above the threshold, AND
       - its spatial scale is large (over-reconstructed large feature).
       The two new Gaussians' means are offset along the major axis of
       Sigma_3D; their Sigma_k is shrunk by a factor phi (usually 1.6).

We track the moving-average gradient of alpha_0 and beta_0 (the "projected
mean" in local plane coordinates) as a proxy for where the model is struggling.

The density control manipulates the TrainableGaussians' parameters in place by
slicing/extending each parameter tensor along axis 0. Crucially, the matching
Adam moments (`exp_avg`, `exp_avg_sq`) are sliced/extended in lockstep so that
optimizer state is preserved for kept splats and only zero-initialized for new
splats (RCA Bug D fix; standard 3DGS does the same).

Public API:
    tracker = DensityTracker(model, optimizer)
    ... in training loop ...
    tracker.accumulate()      # call after each .backward() (before optimizer.step())
    if iter % densify_every == 0:
        stats = tracker.densify_and_prune(config)   # mutates model + optimizer in place
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import Tensor, nn

from . import quaternion as Q
from .gaussian import GaussianParams, compute_derived
from .trainable import TrainableGaussians, build_optimizer


# Names of the per-Gaussian parameters on TrainableGaussians whose row-axis
# (axis 0, length N) must be kept in sync with density operations.
_PER_GAUSSIAN_PARAMS = (
    "p_im", "q_im", "alpha_0", "beta_0", "L", "opacity_logit", "color_logit",
)


@dataclass
class DensityConfig:
    """Hyperparameters for adaptive density control.

    Most defaults mirror standard 3DGS. Tune `opacity_threshold` and the
    `grad_threshold` for your scene — higher thresholds => fewer Gaussians.

    The flags `use_3d_threshold`, `correct_shrinkage`, and `diversify_split_pq`
    enable the three 2026-05 monocular-DC fixes. Off by default so legacy
    behavior is preserved unless explicitly opted into.
    """
    opacity_threshold: float = 0.005     # prune if sigmoid(opacity_logit) < this
    scale_min: float = 1e-4              # prune if smallest Sigma_k eigenvalue < this
    scale_max: float = 4.0               # prune if largest Sigma_k eigenvalue > this
    grad_threshold: float = 2e-4         # clone/split if accumulated grad magnitude > this
    clone_scale_threshold: float = 0.05  # if max(Sigma_k eigvals) < this -> CLONE
                                         # otherwise                      -> SPLIT
    split_shrink_factor: float = 1.6     # new splats have sigma / phi
    split_spatial_offset_sigmas: float = 1.0  # how many sigmas apart to place new splats
    # ----- 2026-05 fixes for monocular Grassmann DC (default off) -----
    use_3d_threshold: bool = False       # decide clone/split on Σ_3D(t_0) λ_max
                                         # (streak length in 3D meters), not Σ_k λ_max.
                                         # Threshold uses `streak_length_threshold`.
    streak_length_threshold: float = 0.05  # when use_3d_threshold=True, Gaussians
                                         # with √λ_max(Σ_3D(t_0)) < this metres CLONE.
    correct_shrinkage: bool = False      # split shrinks L by phi (variance / phi²)
                                         # to match standard 3DGS, instead of the
                                         # legacy L /= sqrt(phi) (variance / phi only).
    diversify_split_pq: bool = False     # split children get NEW (p, q) basis with
                                         # rank-1 axis rotated perpendicular to the
                                         # parent's j_a — the only way splits add
                                         # orientational coverage.


class DensityTracker:
    """Tracks per-Gaussian statistics during training for density control.

    Call `accumulate()` after each loss.backward() (before optimizer.step()).
    It reads gradients on alpha_0 / beta_0 and maintains a running norm.

    Later, call `densify_and_prune(config)` to apply the control operations.
    This mutates the model (adds/removes Gaussians) and surgically updates the
    optimizer state so that Adam moments are preserved for kept splats.

    The optimizer argument is optional only for unit tests that exercise the
    splat-manipulation logic in isolation. In a real training loop, always
    pass it so Bug D (Adam-state reset every density event) does not recur.
    """

    def __init__(
        self,
        model: TrainableGaussians,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        raise NotImplementedError(
            "DensityTracker targets the legacy 2-plane parameterization (p_im, q_im, "
            "alpha_0, beta_0, L). Under the 3-plane (G(3,4)) projector form it is "
            "disabled in Phase A — densify_every defaults to 0 in TrainerConfig and "
            "scripts/train_mono.py. See the plan in "
            "~/.claude/plans/grassmann-splatting-on-imperative-rocket.md, Phase C."
        )
        # --- legacy code below kept for reference; never executed in Phase A ---
        self.model = model
        self.optimizer = optimizer
        device = model.alpha_0.device
        dtype = model.alpha_0.dtype
        N = model.N
        # Accumulated gradient magnitude per Gaussian.
        self.grad_accum = torch.zeros(N, dtype=dtype, device=device)
        self.grad_counts = torch.zeros(N, dtype=torch.long, device=device)

    def accumulate(self) -> None:
        """Record the current (alpha_0, beta_0) gradient magnitudes.

        Must be called after .backward() and before any parameter reset.
        For Gaussians that did NOT contribute to the current render (e.g.,
        outside the sampled camera's frustum), the gradient is zero — we
        still count them but it doesn't affect the running max/mean.
        """
        if self.model.alpha_0.grad is None:
            return
        # magnitude = sqrt(d_alpha^2 + d_beta^2)
        g_alpha = self.model.alpha_0.grad
        g_beta = self.model.beta_0.grad
        mag = torch.sqrt(g_alpha * g_alpha + g_beta * g_beta)
        # Running mean: accum += mag; counts += 1
        self.grad_accum += mag.detach()
        self.grad_counts += 1

    def mean_grad(self) -> Tensor:
        """Mean per-Gaussian gradient magnitude since last reset."""
        counts = self.grad_counts.clamp_min(1)
        return self.grad_accum / counts.to(self.grad_accum.dtype)

    def reset(self) -> None:
        """Reset accumulated gradients (typically after a densification step)."""
        self.grad_accum.zero_()
        self.grad_counts.zero_()

    # --- Operations ---

    def _migrate_optimizer_state(
        self,
        old_param: nn.Parameter,
        new_param: nn.Parameter,
        *,
        slice_idx: Optional[Tensor] = None,
        append_zero_count: int = 0,
    ) -> None:
        """Replace `old_param` with `new_param` in the optimizer and migrate state.

        Behavior depends on the kwargs:
          - slice_idx given: state tensors are sliced along axis 0 by `slice_idx`
            (used for prune / split removal of original rows).
          - append_zero_count > 0: state tensors get `append_zero_count` zero rows
            appended along axis 0 (used for clone / split which append new rows).
          - both default: state is migrated as-is (no shape change).

        If self.optimizer is None, this is a no-op (unit-test path).
        """
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
        """Filter all per-Gaussian parameters to keep only rows where keep_mask is True.

        Adam state (exp_avg, exp_avg_sq) is sliced in lockstep so kept splats
        retain their moment history.
        """
        idx = torch.nonzero(keep_mask, as_tuple=True)[0]
        with torch.no_grad():
            for name in _PER_GAUSSIAN_PARAMS:
                old = getattr(self.model, name)
                new = nn.Parameter(old.data[idx].contiguous())
                setattr(self.model, name, new)
                self._migrate_optimizer_state(old, new, slice_idx=idx)
        # Re-index tracker state.
        self.grad_accum = self.grad_accum[idx].contiguous()
        self.grad_counts = self.grad_counts[idx].contiguous()

    def _append_rows(
        self,
        p_im: Tensor, q_im: Tensor, alpha_0: Tensor, beta_0: Tensor,
        L: Tensor, opacity_logit: Tensor, color_logit: Tensor,
    ) -> None:
        """Append new Gaussian rows to the model's parameters.

        Adam moments for the new rows are zero-initialized (standard 3DGS); the
        moments for existing rows are carried over verbatim.
        """
        new_data = {
            "p_im": p_im, "q_im": q_im, "alpha_0": alpha_0, "beta_0": beta_0,
            "L": L, "opacity_logit": opacity_logit, "color_logit": color_logit,
        }
        n_new = p_im.shape[0]
        with torch.no_grad():
            for name in _PER_GAUSSIAN_PARAMS:
                old = getattr(self.model, name)
                new = nn.Parameter(torch.cat([old.data, new_data[name]], dim=0).contiguous())
                setattr(self.model, name, new)
                self._migrate_optimizer_state(old, new, append_zero_count=n_new)
        # Extend tracker state with zeros for new rows.
        device = self.grad_accum.device
        dtype = self.grad_accum.dtype
        self.grad_accum = torch.cat([self.grad_accum, torch.zeros(n_new, dtype=dtype, device=device)])
        self.grad_counts = torch.cat([self.grad_counts, torch.zeros(n_new, dtype=torch.long, device=device)])

    def prune(self, config: DensityConfig) -> int:
        """Remove Gaussians with low opacity, zero scale, or runaway scale.

        Returns the number of pruned Gaussians.
        """
        with torch.no_grad():
            opacity = torch.sigmoid(self.model.opacity_logit)
            # Sigma_k eigenvalues. Sigma_k = L L^T is SPD 2x2.
            Sigma_k = self._sigma_k()
            eig = torch.linalg.eigvalsh(Sigma_k)        # (N, 2) ascending
            small_mask = (eig[:, 0] < config.scale_min)     # smallest eigval too tiny
            huge_mask = (eig[:, 1] > config.scale_max)      # largest eigval too big
            low_op_mask = opacity < config.opacity_threshold
            drop_mask = small_mask | huge_mask | low_op_mask
            keep_mask = ~drop_mask
            n_pruned = int(drop_mask.sum().item())
        if n_pruned > 0:
            self._keep_rows(keep_mask)
        return n_pruned

    def _sigma_k(self) -> Tensor:
        """Compute Sigma_k from the current (possibly not-lower-triangular) L."""
        L = torch.tril(self.model.L.data)
        return L @ L.transpose(-1, -2)

    def _streak_length_3d(self) -> Tensor:
        """λ_max(Σ_3D(t_0)).sqrt() per Gaussian — the rank-1 streak length in 3D.

        This is the physically meaningful "size" of each Gaussian as it would
        actually be rendered. Used by the use_3d_threshold density-control branch.
        Computed at each Gaussian's own v_0 so we get the actual streak the
        Gaussian renders at its peak temporal weight.
        """
        with torch.no_grad():
            params = self.model.forward()
            d = compute_derived(params)
            sigma_tt_pure = getattr(d, "_sigma_tt_pure", d.Sigma_tt)
            inv_stt = 1.0 / sigma_tt_pure.clamp_min(1e-20)
            cw = d.c_world                                          # (N, 3)
            outer = cw.unsqueeze(-1) * cw.unsqueeze(-2)             # (N, 3, 3)
            Sigma_3D_t = d.Sigma_3D - inv_stt.unsqueeze(-1).unsqueeze(-1) * outer
            ev = torch.linalg.eigvalsh(Sigma_3D_t)                  # (N, 3) ascending
            return ev[:, -1].clamp_min(0).sqrt()                    # (N,) streak length

    def clone_and_split(self, config: DensityConfig) -> tuple[int, int]:
        """Clone small, over-stressed Gaussians and split large, over-stressed ones.

        Returns (n_cloned, n_split).
        """
        grads = self.mean_grad()
        stressed = grads > config.grad_threshold
        if not stressed.any():
            return 0, 0

        with torch.no_grad():
            if config.use_3d_threshold:
                # Decide on the rank-1 axis length in 3D (the actual streak length
                # in scene metres), not on the (α, β) eigenvalues.
                streak_len = self._streak_length_3d()
                small_mask = stressed & (streak_len < config.streak_length_threshold)
                large_mask = stressed & ~small_mask
            else:
                Sigma_k = self._sigma_k()
                eig = torch.linalg.eigvalsh(Sigma_k)            # (N, 2)
                max_eigval = eig[:, 1]                          # largest eigenvalue
                small_mask = stressed & (max_eigval < config.clone_scale_threshold)
                large_mask = stressed & ~small_mask

        n_cloned = int(small_mask.sum().item())
        n_split = int(large_mask.sum().item())

        if n_cloned > 0:
            self._perform_clone(small_mask)
        if n_split > 0:
            self._perform_split(large_mask, config)

        return n_cloned, n_split

    def _perform_clone(self, clone_mask: Tensor) -> None:
        """Duplicate rows where clone_mask is True. New rows are identical to
        originals (training will separate them via gradient noise)."""
        idx = torch.nonzero(clone_mask, as_tuple=True)[0]
        with torch.no_grad():
            self._append_rows(
                p_im=self.model.p_im.data[idx].clone(),
                q_im=self.model.q_im.data[idx].clone(),
                alpha_0=self.model.alpha_0.data[idx].clone(),
                beta_0=self.model.beta_0.data[idx].clone(),
                L=self.model.L.data[idx].clone(),
                opacity_logit=self.model.opacity_logit.data[idx].clone(),
                color_logit=self.model.color_logit.data[idx].clone(),
            )

    def _perform_split(self, split_mask: Tensor, config: DensityConfig) -> None:
        """Replace each Gaussian with two, offset along its major spatial axis,
        with covariance shrunk by `split_shrink_factor`.

        Standard 3DGS does this in 3D world space. Here, since our covariance
        lives in the (alpha, beta) canonical-plane coordinates, we offset the
        mean along the PRINCIPAL direction of Sigma_k in (alpha, beta)-space.
        That direction, when projected through J_embed, is also the major
        spatial axis in R^3 — so the offset is physically meaningful.

        With config.correct_shrinkage=True, L is divided by phi (not sqrt(phi))
        to match the standard 3DGS variance-shrinkage of phi^2.

        With config.diversify_split_pq=True, each child gets a NEW (p, q) basis
        whose rank-1 axis is rotated perpendicular to the parent's. This
        introduces orientational diversity (parent's children render along
        different axes), which is the only way splits add coverage to scenes
        where the streaks are misoriented.
        """
        idx = torch.nonzero(split_mask, as_tuple=True)[0]
        n = idx.numel()
        phi = config.split_shrink_factor

        with torch.no_grad():
            L_orig = torch.tril(self.model.L.data[idx])            # (n, 2, 2)
            Sigma_k = L_orig @ L_orig.transpose(-1, -2)            # (n, 2, 2)
            eigvals, eigvecs = torch.linalg.eigh(Sigma_k)          # ascending; eigvecs is (n, 2, 2)
            # Major axis = eigenvector corresponding to largest eigenvalue = eigvecs[..., :, 1]
            major_dir = eigvecs[..., :, 1]                          # (n, 2)
            major_len = eigvals[..., 1].sqrt()                     # (n,) std-dev along major
            offset_mag = config.split_spatial_offset_sigmas * major_len  # (n,)
            # Offset along (alpha, beta) direction.
            d_alpha = major_dir[..., 0] * offset_mag
            d_beta = major_dir[..., 1] * offset_mag

            # New means: orig +/- offset
            new_alpha_plus = self.model.alpha_0.data[idx] + d_alpha
            new_alpha_minus = self.model.alpha_0.data[idx] - d_alpha
            new_beta_plus = self.model.beta_0.data[idx] + d_beta
            new_beta_minus = self.model.beta_0.data[idx] - d_beta

            # Shrinkage: legacy=L/sqrt(phi) -> Σ/phi; correct=L/phi -> Σ/phi^2 (matches 3DGS).
            shrink_div = phi if config.correct_shrinkage else (phi ** 0.5)
            new_L = L_orig / shrink_div
            new_L_plus = new_L.clone()
            new_L_minus = new_L.clone()

            opacity_copy = self.model.opacity_logit.data[idx].clone()
            color_copy = self.model.color_logit.data[idx].clone()

            if config.diversify_split_pq:
                # Compute the parent's rank-1 axis j_a_parent in 3D for each split.
                # Build NEW (p, q) for each child rotated perpendicular to j_a_parent.
                # The 2 children are placed along 2 orthogonal directions perp to j_a_parent.
                p_plus, q_plus, alpha_p, beta_p = self._build_diversified_children(
                    idx, new_alpha_plus, new_beta_plus, axis_seed=0,
                )
                p_minus, q_minus, alpha_m, beta_m = self._build_diversified_children(
                    idx, new_alpha_minus, new_beta_minus, axis_seed=1,
                )
                # Σ_k cannot be carried across a (p,q) basis change — its physical
                # meaning depends on (e1_hat, e2_hat). Reset to init values so the
                # child starts as a fresh Gaussian at the offset position with the
                # rotated orientation. (The shrinkage logic doesn't apply here since
                # we are not subdividing the parent's covariance.)
                init_L_row = torch.zeros(2, 2, dtype=new_L.dtype, device=new_L.device)
                init_L_row[0, 0] = math.sqrt(0.02)        # init σ_aa
                init_L_row[1, 1] = math.sqrt(0.05)        # init σ_bb
                fresh_L = init_L_row.expand(n, 2, 2).clone()
                self._append_rows(
                    p_im=torch.cat([p_plus, p_minus], dim=0),
                    q_im=torch.cat([q_plus, q_minus], dim=0),
                    alpha_0=torch.cat([alpha_p, alpha_m], dim=0),
                    beta_0=torch.cat([beta_p, beta_m], dim=0),
                    L=torch.cat([fresh_L, fresh_L.clone()], dim=0),
                    opacity_logit=torch.cat([opacity_copy, opacity_copy], dim=0),
                    color_logit=torch.cat([color_copy, color_copy], dim=0),
                )
            else:
                p_im_copy = self.model.p_im.data[idx].clone()
                q_im_copy = self.model.q_im.data[idx].clone()
                self._append_rows(
                    p_im=torch.cat([p_im_copy, p_im_copy], dim=0),
                    q_im=torch.cat([q_im_copy, q_im_copy], dim=0),
                    alpha_0=torch.cat([new_alpha_plus, new_alpha_minus], dim=0),
                    beta_0=torch.cat([new_beta_plus, new_beta_minus], dim=0),
                    L=torch.cat([new_L_plus, new_L_minus], dim=0),
                    opacity_logit=torch.cat([opacity_copy, opacity_copy], dim=0),
                    color_logit=torch.cat([color_copy, color_copy], dim=0),
                )

        # Now remove the originals (they're replaced by the two new splits).
        keep_mask = torch.ones(self.model.alpha_0.shape[0], dtype=torch.bool,
                                device=self.model.alpha_0.device)
        keep_mask[idx] = False
        self._keep_rows(keep_mask)

    def _build_diversified_children(
        self,
        parent_idx: Tensor,
        child_alpha_old: Tensor,   # (n,) — α₀ on parent's basis (used as throwaway,
                                   # we'll recompute via projection on the new basis)
        child_beta_old: Tensor,    # (n,)
        *,
        axis_seed: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Build new (p, q, α₀, β₀) for split children with rotated rank-1 axes.

        For each parent: compute parent's j_a (3D rank-1 axis), pick a perpendicular
        direction, and build a new basis whose rank-1 axis is along that perpendicular.
        Then project the parent's V_k onto the new basis to get (α₀, β₀).

        axis_seed=0 picks the first orthogonal-frame direction; seed=1 picks the
        cross-product (so the two children get mutually orthogonal new axes).

        Returns (p_im, q_im, alpha_0, beta_0) each (n, ...).
        """
        from .grassmann import line_to_pq, canonical_frame, orthonormal_basis
        import math as _m

        params = self.model.forward()
        # Parents only — slice via parent_idx.
        p_par = params.p()[parent_idx]                              # (n, 4)
        q_par = params.q()[parent_idx]                              # (n, 4)
        # j_a = spatial part of e1_hat = r·d. Direction = (p_im+q_im)/|p_im+q_im|.
        d_vec = (Q.imag(p_par) + Q.imag(q_par))                     # (n, 3)
        d_norm = d_vec.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        ja_dir = d_vec / d_norm                                     # (n, 3)
        # Pick orthogonal direction: use (axis_seed==0) → cross with global up
        #                                 (axis_seed==1) → cross with ja_dir × up
        device = ja_dir.device; dtype = ja_dir.dtype
        up = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device).expand_as(ja_dir).clone()
        # Where ja_dir is near-parallel to up, swap to right.
        right = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device).expand_as(ja_dir).clone()
        parallel = (ja_dir * up).sum(-1, keepdim=True).abs() > 0.99
        up = torch.where(parallel, right, up)
        t1 = torch.cross(ja_dir, up, dim=-1)
        t1 = t1 / t1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        t2 = torch.cross(ja_dir, t1, dim=-1)
        t2 = t2 / t2.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        new_axis = t1 if axis_seed == 0 else t2

        # Child V_k = parent V_k + offset (approx — same as legacy split, parent's basis).
        # We use the offset (α, β) on parent basis to compute child V_k directly.
        d_par = compute_derived(GaussianParams(
            p_im=Q.imag(p_par), q_im=Q.imag(q_par),
            alpha_0=params.alpha_0[parent_idx],
            beta_0=params.beta_0[parent_idx],
            L=params.L[parent_idx],
            opacity=params.opacity[parent_idx],
            color=params.color[parent_idx],
            sigma_k_pixel=params.sigma_k_pixel,
            sigma_k_temporal=params.sigma_k_temporal,
        ))
        # Map (α_child, β_child) back to V_k: V_k_child = α·j_a + β·j_b + temporal (skip).
        # Easier: compute the child V_k using the basis directly.
        e1_hat_par, e2_hat_par = orthonormal_basis(p_par, q_par)
        # v_child = α·e1_hat + β·e2_hat (in R^4 = (R, R^3))
        v_child = (child_alpha_old.unsqueeze(-1) * e1_hat_par
                   + child_beta_old.unsqueeze(-1) * e2_hat_par)    # (n, 4)
        V_k_child = Q.imag(v_child)                                 # (n, 3)
        v0_child = Q.real(v_child)                                  # (n,)

        # Build new (p, q) via line_to_pq through the child's V_k along new_axis.
        # Use the t-scaling trick for v0_child.
        v0_safe = torch.where(v0_child.abs() < 1e-8,
                              torch.full_like(v0_child, 1.0),
                              v0_child)
        x_line = V_k_child / v0_safe.unsqueeze(-1)                  # (n, 3)
        new_p, new_q = line_to_pq(x_line, new_axis)                 # (n, 4) each

        # Project (v0_child, V_k_child) onto the new basis to get α₀, β₀.
        e1_hat_new, e2_hat_new = orthonormal_basis(new_p, new_q)
        target = torch.cat([v0_child.unsqueeze(-1), V_k_child], dim=-1)   # (n, 4)
        new_alpha = (target * e1_hat_new).sum(dim=-1)
        new_beta = (target * e2_hat_new).sum(dim=-1)

        return Q.imag(new_p), Q.imag(new_q), new_alpha, new_beta

    # --- Convenience wrapper ---

    def densify_and_prune(
        self,
        config: DensityConfig,
        optimizer_builder: Optional[Callable[[TrainableGaussians], torch.optim.Optimizer]] = None,
    ) -> tuple[Optional[torch.optim.Optimizer], dict]:
        """Apply clone, split, prune in one call.

        The model and the registered optimizer (passed to __init__) are mutated
        in place: parameter tensors are sliced/extended along axis 0 and Adam
        moments follow in lockstep -- kept splats retain their `exp_avg` and
        `exp_avg_sq`; new splats start at zero. This is the RCA Bug D fix.

        `optimizer_builder` is accepted for backward compatibility: if provided
        and no optimizer was registered with the tracker, it is invoked once at
        the end to build a fresh optimizer (the legacy behavior, which loses
        Adam state). Prefer `DensityTracker(model, optimizer)` instead.

        Returns:
            (optimizer, stats). The returned optimizer is the same object that
            was registered (state migrated in place), or a freshly built one
            from the legacy `optimizer_builder` path, or None if neither was
            provided.
            stats: {'pruned', 'cloned', 'split', 'final_N'}.
        """
        n_cloned, n_split = self.clone_and_split(config)
        n_pruned = self.prune(config)
        self.reset()
        if self.optimizer is not None:
            opt_out = self.optimizer
        elif optimizer_builder is not None:
            opt_out = optimizer_builder(self.model)
        else:
            opt_out = None
        return opt_out, {
            "pruned": n_pruned,
            "cloned": n_cloned,
            "split": n_split,
            "final_N": self.model.N,
        }
