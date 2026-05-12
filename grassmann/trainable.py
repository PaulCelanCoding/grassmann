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

  * Color (two paths, gated on `sh_degree`):
      - sh_degree = 0:   color_logit -> sigmoid -> RGB in [0, 1]^3 (constant-RGB
        path; matches legacy behavior).
      - sh_degree > 0:   sh_dc (N, 1, 3) + sh_rest (N, K-1, 3) where
        K = (sh_degree+1)^2. Concatenated to `sh: (N, K, 3)` and fed to
        diff-gaussian-rasterization; the CUDA kernel evaluates the SH
        expansion against the per-Gaussian view direction, so colors are
        view-dependent. `sh_dc` is initialized via `rgb_to_sh_dc(initial RGB)`;
        `sh_rest` is initialized to zeros (3DGS convention). For the toy
        CPU rasterizer fallback the DC term collapses back to constant RGB
        via `sh_dc_to_rgb`, populated as `params.color`.

  * sigma_k_pixel, sigma_k_temporal: scalars (config knobs). Only
    sigma_k_pixel is optionally trainable via learn_sigma_k_pixel.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .gaussian import GaussianParams, num_sh_coeffs, rgb_to_sh_dc, sh_dc_to_rgb


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
        sh_degree: int = 0,
        mu_lr_split: bool = False,
        eps_schur: float = 1e-8,
    ):
        super().__init__()
        self.sh_degree = int(sh_degree)
        self.mu_lr_split = bool(mu_lr_split)
        self.eps_schur = float(eps_schur)
        self.n_raw = nn.Parameter(params.n.to(dtype=dtype, device=device))
        self.L_raw = nn.Parameter(params.L_raw.to(dtype=dtype, device=device))
        if self.mu_lr_split:
            # v7-doc §7.5: per-axis LR for μ (spatial vs time). Stored as two
            # nn.Parameters so the optimizer can give them separate LRs.
            mu_init = params.mu.to(dtype=dtype, device=device)               # (N, 4)
            self.mu_time = nn.Parameter(mu_init[:, 0:1].contiguous())        # (N, 1)
            self.mu_spatial = nn.Parameter(mu_init[:, 1:].contiguous())      # (N, 3)
            self.mu = None  # type: ignore[assignment] — explicitly disabled
        else:
            self.mu = nn.Parameter(params.mu.to(dtype=dtype, device=device))

        opacity_clamped = params.opacity.clamp(1e-6, 1.0 - 1e-6)
        opacity_logit = torch.log(opacity_clamped / (1.0 - opacity_clamped))
        self.opacity_logit = nn.Parameter(opacity_logit.to(dtype=dtype, device=device))

        if self.sh_degree == 0:
            color_clamped = params.color.clamp(1e-6, 1.0 - 1e-6)
            color_logit = torch.log(color_clamped / (1.0 - color_clamped))
            self.color_logit = nn.Parameter(color_logit.to(dtype=dtype, device=device))
        else:
            K = num_sh_coeffs(self.sh_degree)
            sh_dc_init = rgb_to_sh_dc(params.color).to(dtype=dtype, device=device)  # (N, 1, 3)
            sh_rest_init = torch.zeros(
                params.color.shape[0], K - 1, 3, dtype=dtype, device=device,
            )
            self.sh_dc = nn.Parameter(sh_dc_init)
            self.sh_rest = nn.Parameter(sh_rest_init)

        sigma_k_pixel_t = torch.tensor(float(params.sigma_k_pixel), dtype=dtype, device=device)
        if learn_sigma_k_pixel:
            self.sigma_k_pixel_param = nn.Parameter(sigma_k_pixel_t)
        else:
            self.register_buffer("sigma_k_pixel_param", sigma_k_pixel_t)

        sigma_k_temporal_t = torch.tensor(float(params.sigma_k_temporal), dtype=dtype, device=device)
        self.register_buffer("sigma_k_temporal_param", sigma_k_temporal_t)

    @property
    def N(self) -> int:
        """Number of Gaussians currently in the model."""
        return self.n_raw.shape[0]

    def forward(self) -> GaussianParams:
        """Build a GaussianParams snapshot with all reparameterizations applied."""
        eps = 1e-8
        n_norm = self.n_raw.norm(dim=-1, keepdim=True).clamp_min(eps)
        n_unit = self.n_raw / n_norm
        opacity = torch.sigmoid(self.opacity_logit)

        if self.mu_lr_split:
            mu_eff = torch.cat([self.mu_time, self.mu_spatial], dim=-1)      # (N, 4)
        else:
            mu_eff = self.mu

        sh: torch.Tensor | None
        if self.sh_degree == 0:
            color = torch.sigmoid(self.color_logit)
            sh = None
        else:
            sh = torch.cat([self.sh_dc, self.sh_rest], dim=1)              # (N, K, 3)
            # color is the DC-only collapse, used by the toy CPU rasterizer
            # fallback (it can't evaluate view-dependent SH).
            color = sh_dc_to_rgb(self.sh_dc)

        sigma_k_pixel_v = (
            self.sigma_k_pixel_param
            if isinstance(self.sigma_k_pixel_param, nn.Parameter)
            else float(self.sigma_k_pixel_param.item())
        )
        return GaussianParams(
            n=n_unit,
            L_raw=self.L_raw,
            mu=mu_eff,
            opacity=opacity,
            color=color,
            sigma_k_pixel=sigma_k_pixel_v,
            sigma_k_temporal=float(self.sigma_k_temporal_param.item()),
            sh=sh,
            sh_degree=self.sh_degree,
            eps_schur=self.eps_schur,
        )

    def renormalize_manifold_(self) -> None:
        """Hard-normalize n_raw onto S^3. Optional numerical-conditioning aid;
        the forward-pass normalization is what's load-bearing for correctness."""
        with torch.no_grad():
            eps = 1e-8
            self.n_raw.data /= self.n_raw.data.norm(dim=-1, keepdim=True).clamp_min(eps)

    @torch.no_grad()
    def clip_aspect_ratio_(self, max_ratio: float) -> int:
        """#6.2: cap aspect ratio of in-plane covariance λ_max/λ_min ≤ max_ratio.

        Operates on the projected factor P_n L_raw (4×3). Singular values
        s = (s_1 ≥ s_2 ≥ s_3) of P_n L_raw correspond to eigenvalues λ_i = s_i²
        of the 4D covariance. Floor s_min so (s_max/s_min)² ≤ max_ratio, then
        rebuild L_raw = P_n L_raw_clipped + n (n^T L_raw)_orig (n-component is
        in the optimizer null direction; preserved unchanged).

        Returns the number of Gaussians that were actually clipped.
        """
        eps = 1e-8
        n_unit = self.n_raw / self.n_raw.norm(dim=-1, keepdim=True).clamp_min(eps)
        # n^T L_raw : (N, 3); n-direction component of L_raw to add back.
        nT_L = (n_unit.unsqueeze(-2) @ self.L_raw).squeeze(-2)             # (N, 3)
        n_comp = n_unit.unsqueeze(-1) * nT_L.unsqueeze(-2)                 # (N, 4, 3)
        PnL = self.L_raw - n_comp                                          # (N, 4, 3)
        # Batched SVD: PnL = U S V^T with U: (N, 4, 3), S: (N, 3), V^T: (N, 3, 3).
        U, S, Vh = torch.linalg.svd(PnL, full_matrices=False)
        s_max = S.max(dim=-1, keepdim=True).values                         # (N, 1)
        s_floor = s_max / float(max_ratio) ** 0.5
        S_new = torch.maximum(S, s_floor)
        clipped = (S_new > S).any(dim=-1).sum().item()
        if clipped == 0:
            return 0
        PnL_new = U @ torch.diag_embed(S_new) @ Vh
        self.L_raw.data.copy_(PnL_new + n_comp)
        return int(clipped)


def trainable_from_params(
    params: GaussianParams,
    *,
    dtype: torch.dtype = DTYPE_DEFAULT,
    device: str = "cpu",
    learn_sigma_k_pixel: bool = False,
    sh_degree: int = 0,
    mu_lr_split: bool = False,
    eps_schur: float = 1e-8,
) -> TrainableGaussians:
    """Convenience constructor."""
    return TrainableGaussians(
        params, dtype=dtype, device=device,
        learn_sigma_k_pixel=learn_sigma_k_pixel,
        sh_degree=sh_degree,
        mu_lr_split=mu_lr_split,
        eps_schur=eps_schur,
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
    lr_sh_dc: float = 2.5e-3,
    lr_sh_rest_ratio: float = 1.0 / 20.0,
    lr_sigma_k_pixel: float = 1e-2,
    lr_mu_spatial: float = 1e-4,
    lr_mu_time: float = 1e-3,
) -> torch.optim.Optimizer:
    """Adam with separate learning rates per parameter type, following the
    standard 3DGS setup. n is a unit-vector manifold parameter (smaller lr),
    mu and L_raw are scale-aware (intermediate), opacity/color are sigmoid
    logits (larger). When sh_degree > 0, color_logit is replaced by sh_dc and
    sh_rest, which use 3DGS-style LRs (rest = dc * 1/20).

    When `model.mu_lr_split=True`, μ is stored as two parameters (mu_time,
    mu_spatial) and the optimizer uses lr_mu_time / lr_mu_spatial for them
    (v7-doc §7.5 default 1e-3 / 1e-4); the lr_mu argument is ignored.
    """
    param_groups = [
        {"params": [model.n_raw], "lr": lr_n, "name": "n"},
        {"params": [model.L_raw], "lr": lr_L, "name": "L_raw"},
        {"params": [model.opacity_logit], "lr": lr_opacity, "name": "opacity"},
    ]
    if model.mu_lr_split:
        param_groups.append(
            {"params": [model.mu_time], "lr": lr_mu_time, "name": "mu_time"}
        )
        param_groups.append(
            {"params": [model.mu_spatial], "lr": lr_mu_spatial, "name": "mu_spatial"}
        )
    else:
        param_groups.append(
            {"params": [model.mu], "lr": lr_mu, "name": "mu"}
        )
    if model.sh_degree == 0:
        param_groups.append(
            {"params": [model.color_logit], "lr": lr_color, "name": "color"}
        )
    else:
        param_groups.append(
            {"params": [model.sh_dc], "lr": lr_sh_dc, "name": "sh_dc"}
        )
        param_groups.append(
            {"params": [model.sh_rest], "lr": lr_sh_dc * lr_sh_rest_ratio, "name": "sh_rest"}
        )
    if isinstance(model.sigma_k_pixel_param, nn.Parameter):
        param_groups.append(
            {"params": [model.sigma_k_pixel_param], "lr": lr_sigma_k_pixel, "name": "sigma_k_pixel"}
        )

    return torch.optim.Adam(param_groups)
