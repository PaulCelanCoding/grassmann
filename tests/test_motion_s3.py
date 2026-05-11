"""Tests for the opt-in S³-geodesic centroid motion.

Properties verified:
  1. omega = 0  ⇒  condition_on_time output identical to legacy (omega=None) path.
  2. Finite omega, dt = 0  ⇒  R(0) = I, no rotation contribution.
  3. Analytic z-axis case: omega = (0, 0, π/2), V_k = (1, 0, 0), dt = 1
     ⇒  V_3D after S3 rotation = (0, 1, 0) (modulo the additive Schur shift,
     which is zero here when c_world = 0).
  4. _rodrigues is differentiable at θ = 0 (no NaN gradients).
"""
from __future__ import annotations

import math
import torch

from grassmann.gaussian import (
    GaussianParams,
    compute_derived,
    condition_on_time,
    _rodrigues,
)


DTYPE = torch.float64


def _params(N: int, *, with_omega: torch.Tensor | None = None, seed: int = 0) -> GaussianParams:
    g = torch.Generator()
    g.manual_seed(seed)
    n = torch.randn(N, 4, dtype=DTYPE, generator=g)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    L_raw = torch.randn(N, 4, 3, dtype=DTYPE, generator=g) * 0.1
    mu = torch.randn(N, 4, dtype=DTYPE, generator=g)
    return GaussianParams(
        n=n, L_raw=L_raw, mu=mu,
        opacity=torch.full((N,), 0.5, dtype=DTYPE),
        color=torch.full((N, 3), 0.5, dtype=DTYPE),
        sigma_k_pixel=1.0, sigma_k_temporal=0.0,
        omega=with_omega,
    )


def test_omega_zero_matches_legacy():
    N = 8
    legacy = _params(N, with_omega=None, seed=42)
    with_zero = _params(N, with_omega=torch.zeros(N, 3, dtype=DTYPE), seed=42)
    d_l = compute_derived(legacy)
    d_z = compute_derived(with_zero)
    out_l = condition_on_time(legacy, d_l, t_0=0.3)
    out_z = condition_on_time(with_zero, d_z, t_0=0.3)
    assert torch.allclose(out_l.V_3D_t, out_z.V_3D_t, atol=1e-12)
    assert torch.allclose(out_l.Sigma_3D_t, out_z.Sigma_3D_t, atol=1e-12)
    assert torch.allclose(out_l.alpha_eff, out_z.alpha_eff, atol=1e-12)


def test_dt_zero_R_is_identity():
    N = 4
    omega = torch.randn(N, 3, dtype=DTYPE)
    params = _params(N, with_omega=omega, seed=7)
    # Build a params whose v_0 equals t_0 so dt = 0 for all Gaussians.
    t_0 = float(params.mu[..., 0].mean().item())
    # Force v_0 to match t_0 exactly.
    mu_fixed = params.mu.clone()
    mu_fixed[..., 0] = t_0
    params = GaussianParams(
        n=params.n, L_raw=params.L_raw, mu=mu_fixed,
        opacity=params.opacity, color=params.color,
        sigma_k_pixel=params.sigma_k_pixel,
        sigma_k_temporal=params.sigma_k_temporal,
        omega=omega,
    )
    d = compute_derived(params)
    out = condition_on_time(params, d, t_0=t_0)
    # At dt = 0, R = I, so V_3D_t should equal V_k (Schur shift is dt·... = 0 too).
    assert torch.allclose(out.V_3D_t, d.V_k, atol=1e-10)


def test_analytic_z_rotation():
    """omega = (0, 0, π/2) at dt = 1 rotates (1, 0, 0) to (0, 1, 0)."""
    # Construct one Gaussian with V_k = (1, 0, 0), v_0 = 0, c_world = 0 so the
    # Schur shift vanishes and only the S3 rotation contributes.
    # Setting L_raw = 0 makes Sigma_4D = 0; but then sigma_tt_pure = 0 which the
    # soft clamp handles. Use a tiny L_raw orthogonal to e_0 to keep c_world = 0.
    n = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=DTYPE)  # n = e_0; plane = {v_0 = 0}
    L_raw = torch.zeros(1, 4, 3, dtype=DTYPE)
    L_raw[0, 1, 0] = 1.0  # one column in x direction; lives in plane (n·col = 0)
    L_raw[0, 2, 1] = 1.0
    L_raw[0, 3, 2] = 1.0
    mu = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=DTYPE)  # v_0 = 0, V_k = (1, 0, 0)
    omega = torch.tensor([[0.0, 0.0, math.pi / 2.0]], dtype=DTYPE)  # axis = z, rate = π/2
    params = GaussianParams(
        n=n, L_raw=L_raw, mu=mu,
        opacity=torch.tensor([0.5], dtype=DTYPE),
        color=torch.tensor([[0.5, 0.5, 0.5]], dtype=DTYPE),
        sigma_k_pixel=1.0, sigma_k_temporal=0.0,
        clamp_mode="soft", eps_schur=1e-8,
        omega=omega,
    )
    d = compute_derived(params)
    # Sanity: c_world should be ~0 because L_raw rows 1: are orthogonal to row 0.
    assert torch.allclose(d.c_world, torch.zeros_like(d.c_world), atol=1e-12)
    out = condition_on_time(params, d, t_0=1.0)
    # Schur shift = dt · c_world / σ_tt ~= 0; V_3D ≈ R(π/2)·(1, 0, 0) = (0, 1, 0).
    expected = torch.tensor([[0.0, 1.0, 0.0]], dtype=DTYPE)
    assert torch.allclose(out.V_3D_t, expected, atol=1e-10), out.V_3D_t


def test_rodrigues_zero_gradient_finite():
    """Gradient at θ = 0 must be finite (Taylor branch)."""
    theta = torch.zeros(2, 3, dtype=DTYPE, requires_grad=True)
    R = _rodrigues(theta)
    loss = R.sum()
    loss.backward()
    assert theta.grad is not None
    assert torch.isfinite(theta.grad).all(), theta.grad


def test_rodrigues_finite_angle_matches_analytic():
    """R(θ_z·π/2) applied to e_x gives e_y."""
    theta = torch.tensor([[0.0, 0.0, math.pi / 2.0]], dtype=DTYPE)
    R = _rodrigues(theta)
    ex = torch.tensor([[1.0, 0.0, 0.0]], dtype=DTYPE)
    ey = torch.einsum("nij,nj->ni", R, ex)
    expected = torch.tensor([[0.0, 1.0, 0.0]], dtype=DTYPE)
    assert torch.allclose(ey, expected, atol=1e-10), ey
