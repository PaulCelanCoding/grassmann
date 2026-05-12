"""Smoke test for the 3-plane (G(3,4)) projector parameterization.

Verifies the three load-bearing math properties of the new compute_derived /
condition_on_time:

  1. ker(Sigma_4D) contains span(n_hat): Sigma_4D @ n_hat ≈ 0.
  2. After Schur on time, rank(Sigma_3D(t_0)) ≤ 2 -- one eigenvalue is
     numerical zero (the "disk in 3D" property; analogue of legacy Remark 20
     but rank-2 instead of rank-1).
  3. Block-decomposition agrees with the publicly exposed fields:
     Sigma_tt_pure == Sigma_4D[0,0]; c_world == Sigma_4D[1:,0].

These three properties are what makes the 3-plane reformulation correct;
if any breaks, downstream rendering will be silently wrong.
"""
from __future__ import annotations

import torch

from grassmann.gaussian import GaussianParams, compute_derived, condition_on_time


DTYPE = torch.float64


def _random_params(N: int, seed: int = 0) -> GaussianParams:
    g = torch.Generator()
    g.manual_seed(seed)
    n = torch.randn(N, 4, dtype=DTYPE, generator=g)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    L_raw = torch.randn(N, 4, 3, dtype=DTYPE, generator=g) * 0.1
    mu = torch.randn(N, 4, dtype=DTYPE, generator=g)
    return GaussianParams(
        n=n,
        L_raw=L_raw,
        mu=mu,
        opacity=torch.full((N,), 0.5, dtype=DTYPE),
        color=torch.full((N, 3), 0.5, dtype=DTYPE),
        sigma_k_pixel=1.0,
        sigma_k_temporal=0.0,
    )


def _sigma_4D(params: GaussianParams) -> torch.Tensor:
    """Recompute Σ_4D from raw params for cross-checks."""
    n = params.n
    L_raw = params.L_raw
    nL = torch.einsum("...i,...ij->...j", n, L_raw)
    L_plane = L_raw - n.unsqueeze(-1) * nL.unsqueeze(-2)
    return L_plane @ L_plane.transpose(-1, -2)


def test_sigma4d_kernel_contains_n():
    """Property 1: Sigma_4D @ n_hat = 0 for every Gaussian."""
    params = _random_params(N=8)
    Sigma_4D = _sigma_4D(params)                       # (N, 4, 4)
    Sigma_4D_n = (Sigma_4D @ params.n.unsqueeze(-1)).squeeze(-1)  # (N, 4)
    max_err = Sigma_4D_n.abs().max().item()
    assert max_err < 1e-12, f"Sigma_4D @ n_hat had max abs {max_err}, expected ~0"


def test_sigma3D_t0_rank_at_most_2():
    """Property 2: after Schur on time, rank(Sigma_3D(t_0)) <= 2."""
    params = _random_params(N=8, seed=1)
    derived = compute_derived(params)
    tc = condition_on_time(params, derived, t_0=0.5)
    Sigma_3D_t = tc.Sigma_3D_t                          # (N, 3, 3)

    # Sigma_3D_t should be PSD (allowing tiny numerical drift toward negative).
    eigs = torch.linalg.eigvalsh(Sigma_3D_t)            # (N, 3) ascending
    # Three eigenvalues; the smallest should be numerically zero (rank <= 2).
    smallest = eigs[:, 0].abs()
    largest = eigs[:, 2].abs()
    rel = smallest / largest.clamp_min(1e-30)
    max_rel = rel.max().item()
    assert max_rel < 1e-10, (
        f"Smallest/largest Sigma_3D(t_0) eigenvalue ratio = {max_rel:g}; "
        f"expected ~0 for rank-2 disk. Eigs[0] sample: {eigs[0].tolist()}"
    )


def test_block_decomp_matches_public_fields():
    """Property 3: derived.Sigma_tt_pure == Sigma_4D[0,0]; c_world == Sigma_4D[1:,0]."""
    params = _random_params(N=4, seed=2)
    Sigma_4D = _sigma_4D(params)
    derived = compute_derived(params)

    sigma_tt_pure_expected = Sigma_4D[..., 0, 0]
    sigma_tt_pure_actual = getattr(derived, "_sigma_tt_pure", derived.Sigma_tt)
    assert torch.allclose(sigma_tt_pure_expected, sigma_tt_pure_actual, atol=1e-14), (
        f"Sigma_tt_pure mismatch: expected {sigma_tt_pure_expected.tolist()}, "
        f"got {sigma_tt_pure_actual.tolist()}"
    )

    c_world_expected = Sigma_4D[..., 1:, 0]
    assert torch.allclose(c_world_expected, derived.c_world, atol=1e-14), (
        f"c_world mismatch: expected {c_world_expected.tolist()}, "
        f"got {derived.c_world.tolist()}"
    )

    Sigma_3D_expected = Sigma_4D[..., 1:, 1:]
    assert torch.allclose(Sigma_3D_expected, derived.Sigma_3D, atol=1e-14)


def test_mu_split_into_v0_and_Vk():
    """Sanity: derived.v_0 == mu[..., 0]; derived.V_k == mu[..., 1:]."""
    params = _random_params(N=3, seed=3)
    derived = compute_derived(params)
    assert torch.allclose(derived.v_0, params.mu[..., 0])
    assert torch.allclose(derived.V_k, params.mu[..., 1:])


def test_temporal_blur_only_affects_w_t():
    """Sigma_tt_pure (used for Schur) is unaffected by sigma_k_temporal; the
    public Sigma_tt (used for w_t) IS shifted by sigma_k_temporal.
    """
    params = _random_params(N=2, seed=4)
    d_pure = compute_derived(params)

    params_blurred = GaussianParams(
        n=params.n, L_raw=params.L_raw, mu=params.mu,
        opacity=params.opacity, color=params.color,
        sigma_k_pixel=params.sigma_k_pixel,
        sigma_k_temporal=0.05,
    )
    d_blur = compute_derived(params_blurred)

    # Pure variance unchanged.
    assert torch.allclose(d_pure._sigma_tt_pure, d_blur._sigma_tt_pure)
    # Public Sigma_tt shifted by 0.05.
    diff = d_blur.Sigma_tt - d_pure.Sigma_tt
    assert torch.allclose(diff, torch.full_like(diff, 0.05), atol=1e-14)
