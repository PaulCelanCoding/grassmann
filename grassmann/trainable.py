"""
Trainable Grassmann Gaussian model.

Wraps GaussianParams as torch.nn.Parameters so the whole pipeline is
differentiable end-to-end. Also handles:

  * The manifold constraint on (p, q): after each optimizer step, we must
    re-normalize p_im, q_im back to S^2. Equivalently, we can project
    gradients onto the tangent space of S^2 before stepping (Jacobian paper §8.4).
    In practice, "normalize after step" is a standard trick that works well
    for small learning rates and is what we implement.

  * Bounded opacity via sigmoid reparameterization: the raw parameter is an
    unbounded 'opacity_logit' and the value used by the rasterizer is
    sigmoid(opacity_logit) in [0, 1].

  * Bounded color via sigmoid reparameterization: same pattern for the RGB
    triples.

  * The Cholesky factor L of Sigma_k is stored directly as (N, 2, 2).
    We zero the upper triangle on every forward pass to enforce lower-triangular.
    An additional exp() or softplus() on the diagonal would guarantee
    positive definiteness; for now we rely on gradient descent not driving
    the diagonal entries to zero in practice, and add a tiny eps to the
    diagonal for numerical safety.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from . import quaternion as Q
from .gaussian import GaussianParams


DTYPE_DEFAULT = torch.float32   # float32 for training speed; float64 for correctness tests


class TrainableGaussians(nn.Module):
    """Trainable batch of Grassmann Gaussians.

    All parameters are torch.nn.Parameters. Call .forward() to get a
    GaussianParams snapshot that the rasterizer can consume.

    Reparameterizations used:
      * p_im, q_im: raw R^3 vectors, normalized on the fly (projection onto S^2).
      * opacity: stored as 'opacity_logit', exposed via sigmoid().
      * color: stored as 'color_logit', exposed via sigmoid().
      * L: lower-triangular, we zero the upper triangle on every forward.
      * sigma_k_pixel, sigma_k_temporal: scalars (non-trainable by default).
        Only sigma_k_pixel is exposed as learnable via `learn_sigma_k_pixel` --
        the temporal component is a config knob, not an optimization target.
    """

    def __init__(
        self,
        params: GaussianParams,
        *,
        dtype: torch.dtype = DTYPE_DEFAULT,
        device: str = "cpu",
        learn_sigma_k_pixel: bool = False,
    ):
        super().__init__()
        # Register parameters. We move everything to the requested dtype/device.
        self.p_im = nn.Parameter(params.p_im.to(dtype=dtype, device=device))
        self.q_im = nn.Parameter(params.q_im.to(dtype=dtype, device=device))
        self.alpha_0 = nn.Parameter(params.alpha_0.to(dtype=dtype, device=device))
        self.beta_0 = nn.Parameter(params.beta_0.to(dtype=dtype, device=device))
        self.L = nn.Parameter(params.L.to(dtype=dtype, device=device))

        # Reparameterize opacity and color to their logits.
        # opacity in [0,1] <-> logit in R,  logit = log(p / (1 - p)).
        # Clamp for numerical safety.
        opacity_clamped = params.opacity.clamp(1e-6, 1.0 - 1e-6)
        opacity_logit = torch.log(opacity_clamped / (1.0 - opacity_clamped))
        self.opacity_logit = nn.Parameter(opacity_logit.to(dtype=dtype, device=device))

        color_clamped = params.color.clamp(1e-6, 1.0 - 1e-6)
        color_logit = torch.log(color_clamped / (1.0 - color_clamped))
        self.color_logit = nn.Parameter(color_logit.to(dtype=dtype, device=device))

        # sigma_k_pixel: pixel-domain blur. Optionally trainable.
        sigma_k_pixel_t = torch.tensor(float(params.sigma_k_pixel), dtype=dtype, device=device)
        if learn_sigma_k_pixel:
            self.sigma_k_pixel_param = nn.Parameter(sigma_k_pixel_t)
        else:
            self.register_buffer("sigma_k_pixel_param", sigma_k_pixel_t)

        # sigma_k_temporal: temporal blur. Always a buffer (config-only).
        sigma_k_temporal_t = torch.tensor(float(params.sigma_k_temporal), dtype=dtype, device=device)
        self.register_buffer("sigma_k_temporal_param", sigma_k_temporal_t)

    @property
    def N(self) -> int:
        return self.p_im.shape[0]

    def forward(self) -> GaussianParams:
        """Build a GaussianParams snapshot with all reparameterizations applied."""
        # Normalize p_im, q_im onto S^2. This is where the manifold projection
        # happens implicitly -- gradients flow through the normalization so the
        # effective update is along the tangent space.
        eps = 1e-8
        p_norm = self.p_im.norm(dim=-1, keepdim=True).clamp_min(eps)
        q_norm = self.q_im.norm(dim=-1, keepdim=True).clamp_min(eps)
        p_im_unit = self.p_im / p_norm
        q_im_unit = self.q_im / q_norm

        # Ensure L is lower-triangular.
        L_tri = torch.tril(self.L)
        # Guard diagonal against collapsing to 0: add tiny epsilon to the diagonal.
        diag_safe = torch.diagonal(L_tri, dim1=-2, dim2=-1).abs().clamp_min(1e-6)
        L_tri = L_tri.clone()
        # Rewrite the diagonal with the safe version, preserving sign (take abs).
        idx = torch.arange(2, device=L_tri.device)
        L_tri[..., idx, idx] = diag_safe

        # Recover positive quantities.
        opacity = torch.sigmoid(self.opacity_logit)
        color = torch.sigmoid(self.color_logit)

        # If sigma_k_pixel is trainable, pass the tensor through so gradients flow;
        # otherwise unwrap to float to match the GaussianParams: float dataclass field.
        sigma_k_pixel_v = (
            self.sigma_k_pixel_param
            if isinstance(self.sigma_k_pixel_param, nn.Parameter)
            else float(self.sigma_k_pixel_param.item())
        )
        return GaussianParams(
            p_im=p_im_unit,
            q_im=q_im_unit,
            alpha_0=self.alpha_0,
            beta_0=self.beta_0,
            L=L_tri,
            opacity=opacity,
            color=color,
            sigma_k_pixel=sigma_k_pixel_v,
            sigma_k_temporal=float(self.sigma_k_temporal_param.item()),
        )

    def renormalize_manifold_(self) -> None:
        """Hard-normalize p_im and q_im onto S^2. Call after each optimizer step.

        This is redundant with the normalization in forward() but keeps the
        parameter values themselves bounded (helps numerical conditioning).
        """
        with torch.no_grad():
            eps = 1e-8
            self.p_im.data /= self.p_im.data.norm(dim=-1, keepdim=True).clamp_min(eps)
            self.q_im.data /= self.q_im.data.norm(dim=-1, keepdim=True).clamp_min(eps)


def trainable_from_params(
    params: GaussianParams,
    *,
    dtype: torch.dtype = DTYPE_DEFAULT,
    device: str = "cpu",
    learn_sigma_k_pixel: bool = False,
) -> TrainableGaussians:
    """Convenience constructor."""
    return TrainableGaussians(
        params, dtype=dtype, device=device, learn_sigma_k_pixel=learn_sigma_k_pixel,
    )


# ---- Per-parameter-group learning rate setup -------------------------------

def build_optimizer(
    model: TrainableGaussians,
    *,
    lr_pq: float = 1e-3,
    lr_mean: float = 5e-3,       # alpha_0, beta_0
    lr_L: float = 5e-3,
    lr_opacity: float = 5e-2,
    lr_color: float = 2e-2,
    lr_sigma_k_pixel: float = 1e-2,
) -> torch.optim.Optimizer:
    """Adam with separate learning rates per parameter type, following the
    standard 3DGS setup. Rates are in a scale-invariant-ish order: p, q are
    dimensionless unit vectors (small lr); colors and opacity are 1D sigmoid
    logits (larger lr); spatial/temporal means and covariance have intermediate rates.
    """
    param_groups = [
        {"params": [model.p_im, model.q_im], "lr": lr_pq, "name": "pq"},
        {"params": [model.alpha_0, model.beta_0], "lr": lr_mean, "name": "mean"},
        {"params": [model.L], "lr": lr_L, "name": "L"},
        {"params": [model.opacity_logit], "lr": lr_opacity, "name": "opacity"},
        {"params": [model.color_logit], "lr": lr_color, "name": "color"},
    ]
    if isinstance(model.sigma_k_pixel_param, nn.Parameter):
        param_groups.append(
            {"params": [model.sigma_k_pixel_param], "lr": lr_sigma_k_pixel, "name": "sigma_k_pixel"}
        )

    return torch.optim.Adam(param_groups)
