"""
Trainable Grassmann Gaussian model (3-plane projector parameterization).

Wraps GaussianParams as torch.nn.Parameters so the whole pipeline is
differentiable end-to-end. Reparameterizations:

  * n_raw in R^4: an unconstrained 4-vector. Normalized on the fly in
    forward() to give n in S^3 (the 3-plane normal). After each
    optimizer step, renormalize_manifold_() may be called to also pin
    the stored value to unit norm; this is a numerical-conditioning aid,
    not a correctness requirement (the in-forward normalization handles
    correctness via gradient-through-norm).

  * L_raw (4, 3): unconstrained Cholesky-like factor. The projector
    P_n = I - n n^T is applied inside compute_derived, so any column of
    L_raw aligned with n is automatically annihilated. No tril or
    diagonal-positivity constraint is needed.

  * mu in R^4: unconstrained 4-vector mean. The component along n is
    invisible after projection (3 effective DoF) -- gradient-descent
    naturally drives it to whatever value is consistent with the
    projection, so we keep all four components free.

  * opacity_logit -> sigmoid -> [0, 1].
  * color_logit   -> sigmoid -> [0, 1]^3.

  * sigma_k_pixel, sigma_k_temporal: scalars (config knobs). Only
    sigma_k_pixel is optionally trainable via learn_sigma_k_pixel.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .gaussian import GaussianParams


DTYPE_DEFAULT = torch.float32   # float32 for training speed; float64 for correctness tests


class TrainableGaussians(nn.Module):
    """Trainable batch of Grassmann Gaussians under the 3-plane projector
    parameterization.

    All parameters are torch.nn.Parameters. Call .forward() to get a
    GaussianParams snapshot that the rasterizer can consume.
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
        self.n_raw = nn.Parameter(params.n.to(dtype=dtype, device=device))
        self.L_raw = nn.Parameter(params.L_raw.to(dtype=dtype, device=device))
        self.mu = nn.Parameter(params.mu.to(dtype=dtype, device=device))

        opacity_clamped = params.opacity.clamp(1e-6, 1.0 - 1e-6)
        opacity_logit = torch.log(opacity_clamped / (1.0 - opacity_clamped))
        self.opacity_logit = nn.Parameter(opacity_logit.to(dtype=dtype, device=device))

        color_clamped = params.color.clamp(1e-6, 1.0 - 1e-6)
        color_logit = torch.log(color_clamped / (1.0 - color_clamped))
        self.color_logit = nn.Parameter(color_logit.to(dtype=dtype, device=device))

        sigma_k_pixel_t = torch.tensor(float(params.sigma_k_pixel), dtype=dtype, device=device)
        if learn_sigma_k_pixel:
            self.sigma_k_pixel_param = nn.Parameter(sigma_k_pixel_t)
        else:
            self.register_buffer("sigma_k_pixel_param", sigma_k_pixel_t)

        sigma_k_temporal_t = torch.tensor(float(params.sigma_k_temporal), dtype=dtype, device=device)
        self.register_buffer("sigma_k_temporal_param", sigma_k_temporal_t)

    @property
    def N(self) -> int:
        return self.n_raw.shape[0]

    def forward(self) -> GaussianParams:
        """Build a GaussianParams snapshot with all reparameterizations applied."""
        eps = 1e-8
        n_norm = self.n_raw.norm(dim=-1, keepdim=True).clamp_min(eps)
        n_unit = self.n_raw / n_norm
        opacity = torch.sigmoid(self.opacity_logit)
        color = torch.sigmoid(self.color_logit)

        sigma_k_pixel_v = (
            self.sigma_k_pixel_param
            if isinstance(self.sigma_k_pixel_param, nn.Parameter)
            else float(self.sigma_k_pixel_param.item())
        )
        return GaussianParams(
            n=n_unit,
            L_raw=self.L_raw,
            mu=self.mu,
            opacity=opacity,
            color=color,
            sigma_k_pixel=sigma_k_pixel_v,
            sigma_k_temporal=float(self.sigma_k_temporal_param.item()),
        )

    def renormalize_manifold_(self) -> None:
        """Hard-normalize n_raw onto S^3. Optional numerical-conditioning aid;
        the forward-pass normalization is what's load-bearing for correctness."""
        with torch.no_grad():
            eps = 1e-8
            self.n_raw.data /= self.n_raw.data.norm(dim=-1, keepdim=True).clamp_min(eps)


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
    lr_n: float = 1e-3,
    lr_mu: float = 5e-3,
    lr_L: float = 5e-3,
    lr_opacity: float = 5e-2,
    lr_color: float = 2e-2,
    lr_sigma_k_pixel: float = 1e-2,
) -> torch.optim.Optimizer:
    """Adam with separate learning rates per parameter type, following the
    standard 3DGS setup. n is a unit-vector manifold parameter (smaller lr),
    mu and L_raw are scale-aware (intermediate), opacity/color are sigmoid
    logits (larger).
    """
    param_groups = [
        {"params": [model.n_raw], "lr": lr_n, "name": "n"},
        {"params": [model.mu], "lr": lr_mu, "name": "mu"},
        {"params": [model.L_raw], "lr": lr_L, "name": "L_raw"},
        {"params": [model.opacity_logit], "lr": lr_opacity, "name": "opacity"},
        {"params": [model.color_logit], "lr": lr_color, "name": "color"},
    ]
    if isinstance(model.sigma_k_pixel_param, nn.Parameter):
        param_groups.append(
            {"params": [model.sigma_k_pixel_param], "lr": lr_sigma_k_pixel, "name": "sigma_k_pixel"}
        )

    return torch.optim.Adam(param_groups)
