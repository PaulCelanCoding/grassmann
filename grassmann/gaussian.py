"""
Gaussian model: a container for the parameters of one or more Grassmann Gaussians,
together with the derived quantities needed for rendering.

Following the Jacobian paper §9 "Implementation Recipe", each Gaussian is
parameterized by:

  * p, q       in S^2         (2 DOF each)       -- identifies the plane E_{p,q}
  * alpha_0    in R             (1 DOF)          -- local coord of mean along e1_hat
  * beta_0     in R             (1 DOF)          -- local coord of mean along e2_hat
  * Sigma_k    in Sym+(2)       (3 DOF)          -- 2x2 covariance in (alpha, beta)
                                                    parameterized via L s.t. Sigma = L L^T
  * opacity    in [0, 1]        (1 DOF)          -- base opacity
  * color      in R^3           (3 DOF, simple)  -- RGB; spherical harmonics deferred
  * sigma_k    scalar           (pixel blur)     -- isotropic screen-space blur

Total geometry: 9 DOF (matches standard 3DGS). Plus opacity + color.

This module computes the derived quantities from the parameters:
  - V_k = spatial part of v = alpha_0 * e1_hat + beta_0 * e2_hat (world coords)
  - v_0 = time part of v
  - Sigma_3D = J_embed Sigma_k J_embed^T (rank-2, 3x3)
  - Sigma_tt = r^2 (1+c)^2 sigma_bb + sigma_k^2 (eq. 32)
  - c_world = r(1+c) J_embed @ (sigma_ab, sigma_bb)^T (eq. 43)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from . import quaternion as Q
from . import grassmann as G
from . import jacobian as Jac


@dataclass
class GaussianParams:
    """Raw parameters of a batch of Grassmann Gaussians.

    All tensors are batched with leading shape (N,) for N Gaussians.

    p_im, q_im: the imaginary parts of p, q (we store R^3 unit vectors for
                simplicity of optimization; .unit_imag() is applied on demand).
    alpha_0, beta_0: mean coords in E_{p,q} basis, shape (N,).
    L: lower-triangular 2x2 factor of Sigma_k so that Sigma_k = L L^T.
       Stored as shape (N, 2, 2). This parameterization makes Sigma_k SPD by construction.
    opacity: shape (N,), typically passed through sigmoid if trained.
    color: shape (N, 3).
    sigma_k: isotropic screen-space blur variance (scalar or shape (N,)).
    """
    p_im: Tensor           # (N, 3)
    q_im: Tensor           # (N, 3)
    alpha_0: Tensor        # (N,)
    beta_0: Tensor         # (N,)
    L: Tensor              # (N, 2, 2)  lower-triangular
    opacity: Tensor        # (N,)
    color: Tensor          # (N, 3)
    sigma_k: float = 1.0   # pixel^2 blur variance (for the spatial +sigma^2 I term)

    @property
    def N(self) -> int:
        return self.p_im.shape[0]

    def p(self) -> Tensor:
        """Unit imaginary quaternion p, shape (N, 4)."""
        return Q.unit_imag(self.p_im)

    def q(self) -> Tensor:
        """Unit imaginary quaternion q, shape (N, 4)."""
        return Q.unit_imag(self.q_im)

    def Sigma_k(self) -> Tensor:
        """Covariance in local (alpha, beta) coords, shape (N, 2, 2), SPD."""
        # Sigma_k = L L^T where L is lower-triangular.
        # Zero out the upper triangle of L first to ensure strict lower-triangular.
        L = torch.tril(self.L)
        return L @ L.transpose(-1, -2)


# ---- Derived quantities: done in WORLD coordinates, per Jacobian paper §9 ---

@dataclass
class DerivedQuantities:
    """View-independent quantities derived from GaussianParams.

    Computed once per Gaussian (or whenever p, q, alpha, beta, L change).
    All shapes have leading batch (N,).

    V_k:       (N, 3)     Spatial mean in WORLD coords.
    v_0:       (N,)       Temporal mean.
    Sigma_3D:  (N, 3, 3)  View-independent 3D covariance (rank 2).
    Sigma_tt:  (N,)       Temporal variance.
    c_world:   (N, 3)     Spatial-temporal cross-covariance vector.
    """
    V_k: Tensor
    v_0: Tensor
    Sigma_3D: Tensor
    Sigma_tt: Tensor
    c_world: Tensor


def compute_derived(params: GaussianParams) -> DerivedQuantities:
    """Compute all view-independent derived quantities from raw parameters.

    This implements steps 1-5 of §9.2 "The algorithm" in the Jacobian paper.

    NOTE on temporal variance.
    The paper's eq. (32) defines Sigma_tt = r^2(1+c)^2 sigma_bb + sigma_k^2.
    This value is used in two places:
      (a) the 3D conditioning eqs. (44) and (45) for V_3D(t_0) and Sigma_3D(t_0);
      (b) the temporal weight w_t = exp(-(t_0-v_0)^2 / (2 Sigma_tt))  [eq. 37].
    Using (a)+sigma_k^2 however breaks the exact rank-1 property from Remark 20,
    because c_world only encodes the "pure" spatial-temporal covariance
    r(1+c) * J_embed * (sigma_ab, sigma_bb)^T  (eq. 43), which has no sigma_k^2
    contribution.

    We therefore split the two roles:
      - sigma_tt_pure = r^2(1+c)^2 sigma_bb    (used for 3D conditioning, eqs. 44/45)
      - sigma_tt_blur = sigma_tt_pure + sigma_k^2   (used for the temporal weight, eq. 37)
    This preserves the rank-1 property exactly while keeping the temporal
    fall-off well-behaved. For typical operating conditions (sigma_tt_pure >> sigma_k^2)
    the two are indistinguishable; the distinction only matters when sigma_bb -> 0.
    """
    p = params.p()                                      # (N, 4)
    q = params.q()                                      # (N, 4)
    frame = G.canonical_frame(p, q)                     # c, d, s, r

    # Orthonormal basis of E_{p,q} as quaternions.
    e1_hat, e2_hat = G.orthonormal_basis(p, q)          # (N, 4) each

    # Mean v in E_{p,q}: alpha_0 * e1_hat + beta_0 * e2_hat
    alpha = params.alpha_0.unsqueeze(-1)                # (N, 1) for broadcasting
    beta = params.beta_0.unsqueeze(-1)
    v = alpha * e1_hat + beta * e2_hat                  # (N, 4)

    V_k = Q.imag(v)                                     # (N, 3) spatial mean
    v_0 = Q.real(v)                                     # (N,)   temporal mean

    # J_embed: 3x2 spatial embedding matrix.
    J_e = Jac.jacobian_embed(p, q)                      # (N, 3, 2)

    # Sigma_3D = J_embed @ Sigma_k @ J_embed^T  -> (N, 3, 3), rank <= 2.
    Sigma_k = params.Sigma_k()                          # (N, 2, 2)
    Sigma_3D = J_e @ Sigma_k @ J_e.transpose(-1, -2)    # (N, 3, 3)

    # Pure temporal variance r^2 (1+c)^2 * sigma_bb  (for exact rank-1 on conditioning).
    # r(1+c) = sqrt((1+c)/2), so r^2(1+c)^2 = (1+c)/2.
    sigma_bb = Sigma_k[..., 1, 1]                       # (N,)
    time_scale_sq = (1.0 + frame.c) * 0.5               # (N,)
    sigma_tt_pure = time_scale_sq * sigma_bb            # (N,)
    # Blurred temporal variance (for the unnormalized weight w_t). We ALSO use this
    # as the "Sigma_tt" exposed publicly, for consistency with the paper's eq. (32).
    Sigma_tt = sigma_tt_pure + params.sigma_k           # (N,)
    # Stash the pure one privately for the conditioning step.
    # We attach it as an extra field.

    # c_world = r(1+c) * J_embed @ (sigma_ab, sigma_bb)^T     (eq. 43)
    time_scale = torch.sqrt(time_scale_sq)              # (N,)
    sigma_ab = Sigma_k[..., 0, 1]                       # (N,)
    ab_bb = torch.stack([sigma_ab, sigma_bb], dim=-1)   # (N, 2)
    c_world = time_scale.unsqueeze(-1) * (J_e @ ab_bb.unsqueeze(-1)).squeeze(-1)   # (N, 3)

    derived = DerivedQuantities(
        V_k=V_k,
        v_0=v_0,
        Sigma_3D=Sigma_3D,
        Sigma_tt=Sigma_tt,
        c_world=c_world,
    )
    # Attach the pure variance for exact-rank conditioning.
    derived._sigma_tt_pure = sigma_tt_pure   # type: ignore[attr-defined]
    return derived


# ---- Time conditioning (3D-lifted, per §9.1) -------------------------------

@dataclass
class TimeConditioned:
    """Per-frame conditioned quantities for rendering.

    V_3D_t:        (N, 3)     Time-conditioned 3D mean.
    Sigma_3D_t:    (N, 3, 3)  Time-conditioned 3D covariance (rank 1 typically).
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
) -> TimeConditioned:
    """Per-frame conditioning at time t_0 (steps 6-8 of §9.2).

    Implements eqs. (44), (45), and (37):
        V_3D(t_0)   = V_k + c_world * Sigma_tt_pure^{-1} * (t_0 - v_0)
        Sigma_3D(t_0) = Sigma_3D - c_world c_world^T / Sigma_tt_pure
        w_t = exp(-(t_0 - v_0)^2 / (2 Sigma_tt))   [UNNORMALIZED! cf. Remark 18]
        alpha_eff = opacity * w_t

    NOTE: we use the PURE temporal variance (without sigma_k^2) for the 3D
    conditioning so that the rank-1 property of Sigma_3D(t_0) from Remark 20
    holds exactly. The BLURRED Sigma_tt (with sigma_k^2) is used only for the
    temporal weight w_t, where it provides a well-behaved fall-off even when
    sigma_bb -> 0. See the docstring of compute_derived for the rationale.
    """
    dt = t_0 - derived.v_0                                      # (N,)

    # Use pure variance for 3D conditioning (exact rank drop).
    sigma_tt_pure = getattr(derived, "_sigma_tt_pure", derived.Sigma_tt)
    # Guard against division by zero when sigma_bb == 0 (degenerate Gaussian,
    # no temporal extent); in that case the conditioning is ill-defined and we
    # fall back to "no shift, no shrinkage" (the Gaussian is a line at a fixed instant).
    eps = 1e-20
    inv_Stt_pure = 1.0 / sigma_tt_pure.clamp_min(eps)            # (N,)

    # Mean shift: V_3D(t_0) = V_k + c_world * (dt / Sigma_tt_pure)
    shift = (dt * inv_Stt_pure).unsqueeze(-1) * derived.c_world  # (N, 3)
    V_3D_t = derived.V_k + shift                                 # (N, 3)

    # Covariance shrinkage: Sigma_3D - (c_world c_world^T) / Sigma_tt_pure
    cw = derived.c_world                                         # (N, 3)
    outer = cw.unsqueeze(-1) * cw.unsqueeze(-2)                  # (N, 3, 3)
    Sigma_3D_t = derived.Sigma_3D - inv_Stt_pure.unsqueeze(-1).unsqueeze(-1) * outer

    # UNNORMALIZED temporal weight (eq. 37). Uses the blurred Sigma_tt.
    inv_Stt_blur = 1.0 / derived.Sigma_tt.clamp_min(eps)
    w_t = torch.exp(-0.5 * dt * dt * inv_Stt_blur)               # (N,)
    alpha_eff = params.opacity * w_t                             # (N,)

    return TimeConditioned(V_3D_t=V_3D_t, Sigma_3D_t=Sigma_3D_t,
                           alpha_eff=alpha_eff, w_t=w_t)
