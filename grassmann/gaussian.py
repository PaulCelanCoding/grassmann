"""
Gaussian model: container for Grassmann Gaussian parameters and the
view-independent derived quantities the rasterizer needs.

**Parameterization (3-plane G(3,4), projector form).**

Each Gaussian is parameterized by:
  * n  in S^3                (3 DOF, plane normal in R^4)
  * L_raw  in R^(4x3)        (12 raw scalars; column space gets projected)
  * mu in R^4                (mean in space-time; 4 effective DOF -- a shift
                              μ -> μ + λn changes v_0 by λn_0 and V_k by
                              λn_{1:}; invariance of V_3D(t_0) requires
                              n_{1:} = (n_0/Σ_tt^pure) c_world while
                              invariance of w_t requires n_0 = 0; combined
                              they force n = 0, contradicting ‖n‖ = 1. A
                              14k slice-banana A/B confirmed empirically
                              that hard-projecting μ -> P_n μ regresses val
                              PSNR by ~0.2 dB; see
                              results/rca/mu_dof_ab_test.md.)
  * opacity in [0, 1]
  * color   in R^3           (constant RGB; used at sh_degree=0)
  * sh      in R^(K, 3)      (optional, K=(sh_degree+1)^2; used at sh_degree>0)
  * sigma_k_pixel    scalar  (rasterizer EWA blur)
  * sigma_k_temporal scalar  (additive temporal smoothing for w_t only)

The plane E_{n} subset R^4 is the orthogonal complement of n in R^4 -- a
3-dimensional subspace. The projector P_n = I - n n^T sends any vector
into E_{n}; the in-plane covariance is

    Sigma_4D = (P_n L_raw)(P_n L_raw)^T          (4x4 PSD, rank <= 3,
                                                  ker contains span(n))

Block-decomposing along the time axis e0,

    Sigma_4D = [[ sigma_tt   c^T  ],     mu = (mu_t, mu_x)
                [ c          Sigma_3D_full ]]

time-conditioning at t = t0 is the standard Schur complement, identical
to the legacy 2-plane code path:

    Sigma_3D(t0) = Sigma_3D_full - c c^T / sigma_tt    (rank <= 2: a disk)
    mu_3D(t0)   = mu_x + c (t0 - mu_t) / sigma_tt
    w_t         = exp(-(t0 - mu_t)^2 / (2 sigma_tt))

This module exposes the same DerivedQuantities and condition_on_time
contract that the legacy 2-plane parameterization used, so
`fast_rasterizer.py`, the trainer, and the means2D-grad wiring need no
changes when the parameterization is swapped under them.

History: the legacy 2-plane G(2,4) parameterization (p, q in S^2, alpha_0,
beta_0 in R, L in R^(2x2)) gave a rank-1 Sigma_3D(t0) which empirically
plateaued at L1 ~ 0.108 on slice-banana (see
results/rca/monocular_streak_and_density_control.md). The 3-plane
reformulation makes Sigma_3D(t0) rank-2 (a disk in 3D) and removes the
view-axis-pinning pathology by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


# SH band-0 normalization constant (Y_00 = 1 / (2*sqrt(pi))). The CUDA SH
# evaluator uses the convention `radiance = SH(coeffs, dir) + 0.5`, so the
# DC term `sh_dc` initialized from RGB is `(rgb - 0.5) / SH0`.
SH0 = 0.28209479177387814


def num_sh_coeffs(sh_degree: int) -> int:
    """Number of SH coefficients for a given band degree: (degree+1)^2."""
    return (sh_degree + 1) ** 2


def rgb_to_sh_dc(rgb: Tensor) -> Tensor:
    """Convert RGB in [0, 1] to the SH band-0 (DC) coefficient expected by
    the 3DGS-style CUDA rasterizer.

    Inverse of `radiance ~ SH0 * sh_dc + 0.5`, so `sh_dc = (rgb - 0.5) / SH0`.
    Input shape: (..., 3). Output shape: (..., 1, 3).
    """
    return ((rgb - 0.5) / SH0).unsqueeze(-2)


def sh_dc_to_rgb(sh_dc: Tensor) -> Tensor:
    """Inverse of rgb_to_sh_dc — map SH band-0 coefficient back to display RGB.

    Used by the toy CPU rasterizer fallback (which can't evaluate the full
    SH-vs-direction expansion) and for visualization. Clamps to [0, 1].
    """
    if sh_dc.dim() < 2:
        raise ValueError(f"sh_dc must have shape (..., 1, 3); got {tuple(sh_dc.shape)}")
    return (sh_dc[..., 0, :] * SH0 + 0.5).clamp(0.0, 1.0)


@dataclass
class GaussianParams:
    """Raw parameters of a batch of Grassmann Gaussians (3-plane form).

    All tensors are batched with leading shape (N,).

    n:        (N, 4)    Unit plane normal in S^3 (caller should pass already-
                        normalized values; TrainableGaussians.forward() does so).
    L_raw:    (N, 4, 3) Unconstrained Cholesky-like factor; the column space
                        is projected onto E_{n} on demand by compute_derived.
    mu:       (N, 4)    Mean in R^4 = (time, space).
    opacity:  (N,)      In [0, 1].
    color:    (N, 3)    RGB in [0, 1].
    sigma_k_pixel:    pixel-domain blur (rasterizer EWA), scalar.
    sigma_k_temporal: temporal blur added to Sigma_tt for the w_t weight only
                      (not for the Schur complement, so the rank-2 property
                      of Sigma_3D(t0) remains exact).
    """
    n: Tensor                       # (N, 4)
    L_raw: Tensor                   # (N, 4, 3)
    mu: Tensor                      # (N, 4)
    opacity: Tensor                 # (N,)
    color: Tensor                   # (N, 3)
    sigma_k_pixel: float = 1.0
    sigma_k_temporal: float = 0.0
    sh: Optional[Tensor] = None     # (N, K, 3) where K=(sh_degree+1)^2; None at sh_degree=0
    sh_degree: int = 0              # 0 → use color; >0 → use sh
    # v7-doc §5.1 soft-clamp probe: how to floor the temporal-axis denominators
    # in the Schur step + w_t. "hard" (default, legacy): max(Σ_tt, eps_schur)
    # via clamp_min, eps_schur=1e-20. "soft" (v7-doc Prop 5.3): replace Σ_tt
    # with √(Σ_tt² + eps_schur²), eps_schur=1e-8. The soft-clamp is what
    # makes the n=e_0 → tilted-disk transition C^∞-smooth at θ ~ √eps_schur.
    clamp_mode: str = "hard"        # "hard" | "soft"
    eps_schur: float = 1e-20        # default 1e-20 for hard; pass 1e-8 for soft

    @property
    def N(self) -> int:
        return self.n.shape[0]


# ---- Derived quantities: done in WORLD coordinates, per the projector recipe.

@dataclass
class DerivedQuantities:
    """View-independent quantities derived from GaussianParams.

    Computed once per forward pass. Same field names as the legacy 2-plane
    DerivedQuantities so condition_on_time / fast_rasterizer / training are
    parameterization-agnostic.

    V_k:       (N, 3)     Spatial mean in WORLD coords (= mu[..., 1:]).
    v_0:       (N,)       Temporal mean (= mu[..., 0]).
    Sigma_3D:  (N, 3, 3)  Spatial block of Sigma_4D (rank <= 3 -- pre-Schur).
    Sigma_tt:  (N,)       Temporal variance, BLURRED with sigma_k_temporal
                          (used for the w_t fall-off).
    c_world:   (N, 3)     Spatial-temporal cross-covariance vector.
    """
    V_k: Tensor
    v_0: Tensor
    Sigma_3D: Tensor
    Sigma_tt: Tensor
    c_world: Tensor


def compute_derived(params: GaussianParams) -> DerivedQuantities:
    """Compute view-independent derived quantities from the 3-plane projector
    parameterization.

    Steps:
      1. Project L_raw onto the 3-plane E_{n}:
            L_plane = (I - n n^T) L_raw
      2. Build the 4x4 PSD covariance:
            Sigma_4D = L_plane @ L_plane^T
      3. Block-decompose along time (axis 0):
            sigma_tt_pure = Sigma_4D[..., 0, 0]
            c_world       = Sigma_4D[..., 1:, 0]
            Sigma_3D      = Sigma_4D[..., 1:, 1:]
            v_0           = mu[..., 0]
            V_k           = mu[..., 1:]
      4. Sigma_tt (publicly exposed) = sigma_tt_pure + sigma_k_temporal,
         used only for the temporal weight w_t.
         The pure variance is stashed privately as `_sigma_tt_pure` so
         condition_on_time can use it for the Schur complement (the
         existing code does `getattr(derived, "_sigma_tt_pure", Sigma_tt)`).
    """
    n = params.n                                          # (N, 4) unit
    L_raw = params.L_raw                                  # (N, 4, 3)
    mu = params.mu                                        # (N, 4)

    # L_plane = (I - n n^T) L_raw = L_raw - n (n^T L_raw)
    nL = torch.einsum("...i,...ij->...j", n, L_raw)       # (N, 3)
    L_plane = L_raw - n.unsqueeze(-1) * nL.unsqueeze(-2)  # (N, 4, 3)

    # Sigma_4D = L_plane @ L_plane^T
    Sigma_4D = L_plane @ L_plane.transpose(-1, -2)        # (N, 4, 4)

    sigma_tt_pure = Sigma_4D[..., 0, 0]                   # (N,)
    c_world = Sigma_4D[..., 1:, 0]                        # (N, 3)
    Sigma_3D = Sigma_4D[..., 1:, 1:]                      # (N, 3, 3)

    v_0 = mu[..., 0]                                      # (N,)
    V_k = mu[..., 1:]                                     # (N, 3)

    # Public Sigma_tt (used by w_t) is blurred; private one (used by Schur) is pure.
    Sigma_tt = sigma_tt_pure + params.sigma_k_temporal

    derived = DerivedQuantities(
        V_k=V_k,
        v_0=v_0,
        Sigma_3D=Sigma_3D,
        Sigma_tt=Sigma_tt,
        c_world=c_world,
    )
    derived._sigma_tt_pure = sigma_tt_pure                # type: ignore[attr-defined]
    return derived


# ---- Time conditioning (3D-lifted, per §9.1) -------------------------------

@dataclass
class TimeConditioned:
    """Per-frame conditioned quantities for rendering.

    V_3D_t:        (N, 3)     Time-conditioned 3D mean.
    Sigma_3D_t:    (N, 3, 3)  Time-conditioned 3D covariance (rank <= 2 under
                              the 3-plane parameterization -- a disk in 3D).
    alpha_eff:     (N,)       Time-modulated effective opacity.
    w_t:           (N,)       Unnormalized temporal weight (for debugging/culling).
    """
    V_3D_t: Tensor
    Sigma_3D_t: Tensor
    alpha_eff: Tensor
    w_t: Tensor


def condition_on_time(
    params: GaussianParams,
    derived: DerivedQuantities,
    t_0: float,
    *,
    static: bool = False,
) -> TimeConditioned:
    """Per-frame conditioning at time t_0 (Schur complement on the time axis).

        V_3D(t_0)    = V_k + c_world * (t_0 - v_0) / sigma_tt_pure
        Sigma_3D(t_0) = Sigma_3D - c_world c_world^T / sigma_tt_pure
        w_t           = exp(-(t_0 - v_0)^2 / (2 Sigma_tt))     [BLURRED]
        alpha_eff     = opacity * w_t

    The "pure" sigma_tt_pure is used for the Schur (so Sigma_3D(t_0) is
    exactly rank-2 under the projector parameterization). The "blurred"
    Sigma_tt = sigma_tt_pure + sigma_k_temporal is used only for the
    temporal weight w_t to keep the fall-off well-behaved when n is
    near-aligned with the time axis (degenerate, sigma_tt_pure -> 0).

    `static=True` disables ALL temporal coupling: V_3D_t = V_k (no Schur
    shift), Sigma_3D_t = Sigma_3D (no Schur shrinkage), w_t = 1 (every
    Gaussian visible at every frame, regardless of v_0). This collapses
    the 3-plane model to a static-3DGS-on-monocular-bundle baseline --
    Gaussians must explain every frame simultaneously, with no
    per-frame parameters. The L1 floor of this run measures "what static
    3DGS achieves if you ignore the scene's motion"; the gap to the
    full temporal run measures the value of time conditioning.
    """
    if static:
        ones = torch.ones_like(derived.v_0)
        return TimeConditioned(
            V_3D_t=derived.V_k,
            Sigma_3D_t=derived.Sigma_3D,
            alpha_eff=params.opacity,           # w_t = 1
            w_t=ones,
        )

    dt = t_0 - derived.v_0                                      # (N,)

    sigma_tt_pure = getattr(derived, "_sigma_tt_pure", derived.Sigma_tt)
    # v7-doc §5.1 clamp: hard = max(x, eps); soft = √(x² + eps²).
    eps = float(params.eps_schur)
    if params.clamp_mode == "soft":
        # √(x² + ε²) ≥ ε always, smooth in x, identical to x for x ≫ ε.
        Stt_pure_safe = torch.sqrt(sigma_tt_pure ** 2 + eps ** 2)
        Stt_blur_safe = torch.sqrt(derived.Sigma_tt ** 2 + eps ** 2)
    else:                                                        # "hard"
        Stt_pure_safe = sigma_tt_pure.clamp_min(eps)
        Stt_blur_safe = derived.Sigma_tt.clamp_min(eps)
    inv_Stt_pure = 1.0 / Stt_pure_safe                           # (N,)

    shift = (dt * inv_Stt_pure).unsqueeze(-1) * derived.c_world  # (N, 3)
    V_3D_t = derived.V_k + shift                                 # (N, 3)

    cw = derived.c_world                                         # (N, 3)
    outer = cw.unsqueeze(-1) * cw.unsqueeze(-2)                  # (N, 3, 3)
    Sigma_3D_t = derived.Sigma_3D - inv_Stt_pure.unsqueeze(-1).unsqueeze(-1) * outer

    inv_Stt_blur = 1.0 / Stt_blur_safe
    w_t = torch.exp(-0.5 * dt * dt * inv_Stt_blur)               # (N,)
    alpha_eff = params.opacity * w_t                             # (N,)

    return TimeConditioned(V_3D_t=V_3D_t, Sigma_3D_t=Sigma_3D_t,
                           alpha_eff=alpha_eff, w_t=w_t)
