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

The density control manipulates the TrainableGaussians' parameters IN PLACE,
removing / adding rows. Because this changes the parameter count, we also
rebuild the optimizer afterward (Adam's moment buffers need re-alignment).

Public API:
    tracker = DensityTracker(model)
    ... in training loop ...
    tracker.accumulate()      # call after each .backward() (before optimizer.step())
    if iter % densify_every == 0:
        model, optimizer = tracker.densify_and_prune(config, optimizer_builder)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import Tensor, nn

from .gaussian import GaussianParams, compute_derived
from .trainable import TrainableGaussians, build_optimizer


@dataclass
class DensityConfig:
    """Hyperparameters for adaptive density control.

    Most defaults mirror standard 3DGS. Tune `opacity_threshold` and the
    `grad_threshold` for your scene — higher thresholds => fewer Gaussians.
    """
    opacity_threshold: float = 0.005     # prune if sigmoid(opacity_logit) < this
    scale_min: float = 1e-4              # prune if smallest Sigma_k eigenvalue < this
    scale_max: float = 4.0               # prune if largest Sigma_k eigenvalue > this
    grad_threshold: float = 2e-4         # clone/split if accumulated grad magnitude > this
    clone_scale_threshold: float = 0.05  # if max(Sigma_k eigvals) < this -> CLONE
                                         # otherwise                      -> SPLIT
    split_shrink_factor: float = 1.6     # new splats have sigma / phi
    split_spatial_offset_sigmas: float = 1.0  # how many sigmas apart to place new splats


class DensityTracker:
    """Tracks per-Gaussian statistics during training for density control.

    Call `accumulate()` after each loss.backward() (before optimizer.step()).
    It reads gradients on alpha_0 / beta_0 and maintains a running norm.

    Later, call `densify_and_prune(config, optimizer_builder)` to apply the
    control operations. This mutates the model (adds/removes Gaussians) and
    returns a fresh optimizer rebuilt over the new parameters.
    """

    def __init__(self, model: TrainableGaussians):
        self.model = model
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

    def _keep_rows(self, keep_mask: Tensor) -> None:
        """Filter all per-Gaussian parameters to keep only rows where keep_mask is True."""
        idx = torch.nonzero(keep_mask, as_tuple=True)[0]
        with torch.no_grad():
            self.model.p_im = nn.Parameter(self.model.p_im.data[idx].contiguous())
            self.model.q_im = nn.Parameter(self.model.q_im.data[idx].contiguous())
            self.model.alpha_0 = nn.Parameter(self.model.alpha_0.data[idx].contiguous())
            self.model.beta_0 = nn.Parameter(self.model.beta_0.data[idx].contiguous())
            self.model.L = nn.Parameter(self.model.L.data[idx].contiguous())
            self.model.opacity_logit = nn.Parameter(self.model.opacity_logit.data[idx].contiguous())
            self.model.color_logit = nn.Parameter(self.model.color_logit.data[idx].contiguous())
        # Re-index tracker state.
        self.grad_accum = self.grad_accum[idx].contiguous()
        self.grad_counts = self.grad_counts[idx].contiguous()

    def _append_rows(
        self,
        p_im: Tensor, q_im: Tensor, alpha_0: Tensor, beta_0: Tensor,
        L: Tensor, opacity_logit: Tensor, color_logit: Tensor,
    ) -> None:
        """Append new Gaussian rows to the model's parameters."""
        with torch.no_grad():
            self.model.p_im = nn.Parameter(torch.cat([self.model.p_im.data, p_im], dim=0).contiguous())
            self.model.q_im = nn.Parameter(torch.cat([self.model.q_im.data, q_im], dim=0).contiguous())
            self.model.alpha_0 = nn.Parameter(torch.cat([self.model.alpha_0.data, alpha_0], dim=0).contiguous())
            self.model.beta_0 = nn.Parameter(torch.cat([self.model.beta_0.data, beta_0], dim=0).contiguous())
            self.model.L = nn.Parameter(torch.cat([self.model.L.data, L], dim=0).contiguous())
            self.model.opacity_logit = nn.Parameter(torch.cat([self.model.opacity_logit.data, opacity_logit], dim=0).contiguous())
            self.model.color_logit = nn.Parameter(torch.cat([self.model.color_logit.data, color_logit], dim=0).contiguous())
        # Extend tracker state with zeros for new rows.
        n_new = p_im.shape[0]
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

    def clone_and_split(self, config: DensityConfig) -> tuple[int, int]:
        """Clone small, over-stressed Gaussians and split large, over-stressed ones.

        Returns (n_cloned, n_split).
        """
        grads = self.mean_grad()
        stressed = grads > config.grad_threshold
        if not stressed.any():
            return 0, 0

        with torch.no_grad():
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

            # New L: shrink by phi. Since Sigma = L L^T scales quadratically with L,
            # divide L by sqrt(phi) to reduce variance by phi.
            new_L = L_orig / (phi ** 0.5)
            # Copy for each of the two offset Gaussians.
            new_L_plus = new_L.clone()
            new_L_minus = new_L.clone()

            # Other parameters: copy directly.
            p_im_copy = self.model.p_im.data[idx].clone()
            q_im_copy = self.model.q_im.data[idx].clone()
            opacity_copy = self.model.opacity_logit.data[idx].clone()
            color_copy = self.model.color_logit.data[idx].clone()

            # Append both the plus-copy and the minus-copy (2n new rows).
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

    # --- Convenience wrapper: do it all + rebuild optimizer ---

    def densify_and_prune(
        self,
        config: DensityConfig,
        optimizer_builder: Callable[[TrainableGaussians], torch.optim.Optimizer],
    ) -> tuple[torch.optim.Optimizer, dict]:
        """Apply clone, split, prune in one call. Returns a fresh optimizer and stats.

        Returns:
            new_optimizer: an Adam rebuilt over the (possibly resized) parameters.
            stats: {'pruned', 'cloned', 'split', 'final_N'}.
        """
        n_cloned, n_split = self.clone_and_split(config)
        n_pruned = self.prune(config)
        self.reset()
        new_opt = optimizer_builder(self.model)
        return new_opt, {
            "pruned": n_pruned,
            "cloned": n_cloned,
            "split": n_split,
            "final_N": self.model.N,
        }
