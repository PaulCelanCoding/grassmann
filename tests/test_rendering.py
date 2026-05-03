"""Tests for gaussian.py and rasterizer.py (Phase 3).

Key properties verified:
  1. Derived quantities (V_k, v_0, Sigma_3D, Sigma_tt, c_world) match their
     definitions from the Jacobian paper.
  2. Sigma_3D has rank <= 2 before conditioning; rank drops to <= 1 after conditioning
     (Remark 20 of the Jacobian paper).
  3. Temporal weight w_t is unnormalized (peak 1 at t_0 = v_0), NOT a density.
     This is the v5 fix described in Remark 18.
  4. Rasterizer produces reasonable images:
     - At t_0 = v_0, Gaussian has full opacity; blob centered at projected mean.
     - Far from v_0, Gaussian is invisible.
  5. Sanity: occlusion works (front Gaussian occludes back one).
"""
import pytest
import torch

from grassmann import quaternion as Q
from grassmann import grassmann as G
from grassmann import jacobian as Jac
from grassmann.projection import Camera, project_static, perspective_jacobian
from grassmann.gaussian import GaussianParams, compute_derived, condition_on_time
from grassmann.rasterizer import project_to_screen, rasterize, eval_2d_gaussian


DTYPE = torch.float64
torch.manual_seed(123)


def make_params(
    n=3,
    *,
    p_im=None, q_im=None,
    alpha_0=None, beta_0=None,
    L=None,
    opacity=None, color=None,
    sigma_k_pixel=1.0,
    sigma_k_temporal=1.0,
    dtype=DTYPE,
):
    """Helper to build GaussianParams with sensible defaults."""
    if p_im is None:
        p_im = torch.randn(n, 3, dtype=dtype)
        p_im = p_im / p_im.norm(dim=-1, keepdim=True)
    if q_im is None:
        q_im = torch.randn(n, 3, dtype=dtype)
        q_im = q_im / q_im.norm(dim=-1, keepdim=True)
        # Ensure p . q > -0.5 so we're away from the antidiagonal.
        dots = (p_im * q_im).sum(dim=-1)
        flip = dots < -0.5
        q_im[flip] = -q_im[flip]
    if alpha_0 is None:
        alpha_0 = torch.zeros(n, dtype=dtype)
    if beta_0 is None:
        beta_0 = torch.zeros(n, dtype=dtype)
    if L is None:
        # Default: small isotropic
        L = torch.zeros(n, 2, 2, dtype=dtype)
        L[..., 0, 0] = 0.05
        L[..., 1, 1] = 0.05
    if opacity is None:
        opacity = torch.full((n,), 0.9, dtype=dtype)
    if color is None:
        color = torch.rand(n, 3, dtype=dtype)
    return GaussianParams(
        p_im=p_im, q_im=q_im,
        alpha_0=alpha_0, beta_0=beta_0,
        L=L, opacity=opacity, color=color,
        sigma_k_pixel=sigma_k_pixel,
        sigma_k_temporal=sigma_k_temporal,
    )


# =============================================================================
# Derived quantities
# =============================================================================

def test_sigma_3d_has_rank_at_most_2():
    """Sigma_3D = J_embed Sigma_k J_embed^T is 3x3 but lives in a 2-plane."""
    params = make_params(n=5)
    derived = compute_derived(params)
    # Check each 3x3 covariance has rank <= 2: smallest singular value ~ 0.
    for i in range(5):
        s = torch.linalg.svdvals(derived.Sigma_3D[i])
        assert s[2] < 1e-10, f"Sigma_3D[{i}] has rank > 2: smallest sv = {s[2]}"


def test_sigma_3d_symmetric_psd():
    """Covariance matrices must be symmetric and PSD."""
    params = make_params(n=5)
    derived = compute_derived(params)
    for i in range(5):
        S = derived.Sigma_3D[i]
        assert torch.allclose(S, S.T, atol=1e-10)
        # PSD: all eigenvalues >= 0.
        evals = torch.linalg.eigvalsh(S)
        assert (evals >= -1e-10).all()


def test_sigma_tt_matches_formula():
    """Sigma_tt = r^2 (1+c)^2 sigma_bb + sigma_k_temporal  (eq. 32, with split sigma_k)."""
    params = make_params(n=4, sigma_k_temporal=0.5)
    derived = compute_derived(params)
    Sigma_k = params.Sigma_k()
    frame = G.canonical_frame(params.p(), params.q())
    # r^2 (1+c)^2 = (1+c)/2
    time_scale_sq = (1.0 + frame.c) * 0.5
    expected = time_scale_sq * Sigma_k[..., 1, 1] + params.sigma_k_temporal
    assert torch.allclose(derived.Sigma_tt, expected, atol=1e-10)


def test_V_k_is_spatial_part_of_v():
    """V_k = imag(alpha_0 e1 + beta_0 e2)."""
    n = 5
    params = make_params(n, alpha_0=torch.linspace(-0.5, 0.5, n, dtype=DTYPE),
                            beta_0=torch.linspace(-0.3, 0.3, n, dtype=DTYPE))
    derived = compute_derived(params)

    e1_hat, e2_hat = G.orthonormal_basis(params.p(), params.q())
    v = params.alpha_0.unsqueeze(-1) * e1_hat + params.beta_0.unsqueeze(-1) * e2_hat
    expected_V = Q.imag(v)
    expected_v0 = Q.real(v)

    assert torch.allclose(derived.V_k, expected_V, atol=1e-10)
    assert torch.allclose(derived.v_0, expected_v0, atol=1e-10)


# =============================================================================
# Time conditioning
# =============================================================================

def test_at_t_equals_v0_effective_opacity_equals_base():
    """When t_0 = v_0, w_t = 1 and alpha_eff = opacity."""
    params = make_params(n=3, beta_0=torch.tensor([0.0, 0.2, -0.3], dtype=DTYPE))
    derived = compute_derived(params)
    # Use the actual v_0 of each Gaussian as its own t_0 (one at a time).
    for i in range(3):
        t_0 = derived.v_0[i].item()
        tc = condition_on_time(params, derived, t_0)
        # Gaussian i should have w_t[i] ~ 1 and alpha_eff[i] ~ opacity[i].
        assert abs(tc.w_t[i].item() - 1.0) < 1e-10
        assert torch.allclose(tc.alpha_eff[i], params.opacity[i], atol=1e-10)


def test_w_t_is_unnormalized():
    """w_t must be the unnormalized Gaussian with peak 1, NOT a density.
    This is the fix in Remark 18 -- the normalized density would blow up as
    Sigma_tt -> 0 and drive alpha_eff > 1.
    """
    # Make Sigma_tt very small by making sigma_bb, sigma_k_temporal small.
    n = 1
    L = torch.tensor([[[0.01, 0.0], [0.0, 0.01]]], dtype=DTYPE)
    params = make_params(n=n, L=L, sigma_k_pixel=1e-6, sigma_k_temporal=1e-6)
    derived = compute_derived(params)
    assert derived.Sigma_tt.item() < 1e-4, f"Sigma_tt should be small; got {derived.Sigma_tt.item()}"

    # At t_0 = v_0 exactly, w_t should equal 1, not 1/sqrt(2 pi Stt) ~ 400.
    tc = condition_on_time(params, derived, derived.v_0[0].item())
    assert abs(tc.w_t.item() - 1.0) < 1e-10
    assert tc.alpha_eff.item() <= params.opacity.item() + 1e-10


def test_w_t_decays_gaussian():
    """At t_0 = v_0 +/- 3*sqrt(Sigma_tt), w_t should be ~ exp(-9/2) ~ 0.011."""
    params = make_params(n=1)
    derived = compute_derived(params)
    sigma = derived.Sigma_tt.sqrt().item()
    v0 = derived.v_0.item()
    tc = condition_on_time(params, derived, v0 + 3 * sigma)
    expected = torch.exp(torch.tensor(-4.5)).item()
    assert abs(tc.w_t.item() - expected) < 1e-6


def test_sigma_3d_rank_drops_after_conditioning():
    """After conditioning on t_0, Sigma_3D has rank <= 1 (Remark 20)."""
    params = make_params(n=3)
    derived = compute_derived(params)
    tc = condition_on_time(params, derived, t_0=0.0)
    for i in range(3):
        s = torch.linalg.svdvals(tc.Sigma_3D_t[i])
        # Smallest two singular values should be ~0 (rank 1).
        assert s[1] < 1e-10, f"Sigma_3D_t[{i}]: s[1] = {s[1]} should be ~0 (rank 1)"


def test_sigma_3d_t_psd():
    """Conditioned covariance must still be PSD."""
    params = make_params(n=5)
    derived = compute_derived(params)
    for t in [-1.0, 0.0, 0.5, 2.0]:
        tc = condition_on_time(params, derived, t_0=t)
        for i in range(5):
            S = tc.Sigma_3D_t[i]
            # Symmetric
            assert torch.allclose(S, S.T, atol=1e-10)
            # PSD
            evals = torch.linalg.eigvalsh(S)
            assert (evals >= -1e-8).all(), f"at t={t}, Gaussian {i}: min eval = {evals.min()}"


def test_mean_shift_linear_in_t():
    """V_3D(t_0) is linear in (t_0 - v_0): eq. (44)."""
    params = make_params(n=2)
    derived = compute_derived(params)
    V0 = condition_on_time(params, derived, 0.0).V_3D_t
    V1 = condition_on_time(params, derived, 1.0).V_3D_t
    V2 = condition_on_time(params, derived, 2.0).V_3D_t
    # Linear => V2 - V1 == V1 - V0
    diff1 = V1 - V0
    diff2 = V2 - V1
    assert torch.allclose(diff1, diff2, atol=1e-10)


# =============================================================================
# Splat projection and consistency
# =============================================================================

def test_project_to_screen_mean_matches_exact_projection():
    """The projected splat mean should be the exact perspective projection of V_3D(t_0),
    NOT a linearized approximation. This is the key benefit of the 3D-lifted method
    mentioned in §9.1.
    """
    cam = Camera(
        R=torch.eye(3, dtype=DTYPE), c=torch.zeros(3, dtype=DTYPE),
        fx=500, fy=500, cx=320, cy=240,
    )
    # Place Gaussians far in front of camera.
    params = make_params(n=3)
    derived = compute_derived(params)
    # Shift V_k into view by giving line positions in front of camera.
    # Easier: just override V_3D_t directly by manual construction.
    tc = condition_on_time(params, derived, 0.0)
    tc.V_3D_t[:] = torch.tensor([
        [0.5, 0.2, 5.0],
        [-0.3, 0.1, 8.0],
        [1.0, -0.5, 6.0],
    ], dtype=DTYPE)

    sg = project_to_screen(params, tc, cam)
    # sg.uv should be exactly project_static(V_3D_t).
    expected = project_static(tc.V_3D_t, cam)
    assert torch.allclose(sg.uv, expected, atol=1e-10)


def test_cov2d_is_spd():
    """2D screen-space covariance must be symmetric PSD."""
    cam = Camera(
        R=torch.eye(3, dtype=DTYPE), c=torch.zeros(3, dtype=DTYPE),
        fx=500, fy=500, cx=320, cy=240,
    )
    params = make_params(n=5, sigma_k_pixel=0.5)
    derived = compute_derived(params)
    tc = condition_on_time(params, derived, 0.0)
    # Place all in front.
    tc.V_3D_t[:] = torch.tensor([[0.0, 0.0, 5.0]] * 5, dtype=DTYPE)
    sg = project_to_screen(params, tc, cam)
    for i in range(5):
        C = sg.cov2d[i]
        assert torch.allclose(C, C.T, atol=1e-10)
        evals = torch.linalg.eigvalsh(C)
        assert (evals > 0).all(), f"cov2d[{i}] not PD: evals = {evals}"


# =============================================================================
# Rasterizer sanity tests
# =============================================================================

def test_eval_2d_gaussian_peak_at_mean():
    """A 2D Gaussian evaluated at its mean should equal 1 (unnormalized)."""
    uv = torch.zeros(1, 1, 2, dtype=DTYPE)
    mu = torch.zeros(2, dtype=DTYPE)
    cov = torch.eye(2, dtype=DTYPE) * 3.0
    val = eval_2d_gaussian(uv, mu, cov)
    assert abs(val.item() - 1.0) < 1e-10


def test_eval_2d_gaussian_decays_correctly():
    """At 2*sigma from center (isotropic), value should be exp(-2)."""
    uv = torch.tensor([[[2.0, 0.0]]], dtype=DTYPE)  # (1, 1, 2)
    mu = torch.zeros(2, dtype=DTYPE)
    cov = torch.eye(2, dtype=DTYPE) * 1.0   # sigma = 1
    val = eval_2d_gaussian(uv, mu, cov)
    expected = torch.exp(torch.tensor(-2.0, dtype=DTYPE)).item()
    assert abs(val.item() - expected) < 1e-12


def test_rasterize_single_static_gaussian():
    """Place one red Gaussian, render; verify pixel at projected mean is mostly red."""
    cam = Camera(
        R=torch.eye(3, dtype=DTYPE), c=torch.zeros(3, dtype=DTYPE),
        fx=500, fy=500, cx=40, cy=30,
    )
    # Build a Gaussian at V = (0, 0, 5) with opacity 1, color red.
    params = make_params(
        n=1,
        color=torch.tensor([[1.0, 0.0, 0.0]], dtype=DTYPE),
        opacity=torch.tensor([1.0], dtype=DTYPE),
        sigma_k_pixel=2.0,
    )
    derived = compute_derived(params)
    tc = condition_on_time(params, derived, t_0=derived.v_0[0].item())  # at mean time
    # Override mean to a convenient position.
    tc.V_3D_t[0] = torch.tensor([0.0, 0.0, 5.0], dtype=DTYPE)
    sg = project_to_screen(params, tc, cam)
    img = rasterize(sg, H=60, W=80)
    # Pixel at the projected mean (approximately 40, 30) should be reddish.
    u, v = sg.uv[0].round().long().tolist()
    # Clamp for safety
    u = max(0, min(u, 79))
    v = max(0, min(v, 59))
    pixel = img[v, u]
    # Should be red-dominated
    assert pixel[0] > pixel[1] and pixel[0] > pixel[2]
    assert pixel[0] > 0.5


def test_rasterize_temporal_fade():
    """The same Gaussian rendered far from its center time should be dim."""
    cam = Camera(
        R=torch.eye(3, dtype=DTYPE), c=torch.zeros(3, dtype=DTYPE),
        fx=500, fy=500, cx=40, cy=30,
    )
    params = make_params(
        n=1,
        color=torch.tensor([[1.0, 0.0, 0.0]], dtype=DTYPE),
        opacity=torch.tensor([1.0], dtype=DTYPE),
        sigma_k_pixel=2.0,
    )
    derived = compute_derived(params)
    sigma_tt = derived.Sigma_tt.sqrt().item()
    v0 = derived.v_0.item()

    # At t_0 = v_0: bright
    tc_on = condition_on_time(params, derived, v0)
    tc_on.V_3D_t[0] = torch.tensor([0.0, 0.0, 5.0], dtype=DTYPE)
    sg_on = project_to_screen(params, tc_on, cam)
    img_on = rasterize(sg_on, H=60, W=80)

    # At t_0 = v_0 + 10*sigma: essentially invisible
    tc_off = condition_on_time(params, derived, v0 + 10 * sigma_tt)
    tc_off.V_3D_t[0] = torch.tensor([0.0, 0.0, 5.0], dtype=DTYPE)
    sg_off = project_to_screen(params, tc_off, cam)
    img_off = rasterize(sg_off, H=60, W=80)

    assert img_on.max() > 0.5
    assert img_off.max() < 1e-3


def test_rasterize_occlusion():
    """A bright red Gaussian in front of a bright green one should produce mostly
    red at its projected location."""
    cam = Camera(
        R=torch.eye(3, dtype=DTYPE), c=torch.zeros(3, dtype=DTYPE),
        fx=500, fy=500, cx=40, cy=30,
    )
    # Build two Gaussians at same screen location but different depths.
    # Hack: construct manually.
    params = make_params(
        n=2,
        color=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=DTYPE),  # red, green
        opacity=torch.tensor([0.95, 0.95], dtype=DTYPE),
        sigma_k_pixel=3.0,
    )
    derived = compute_derived(params)
    tc = condition_on_time(params, derived, t_0=0.0)
    # Override means: red at depth 3, green at depth 8 -- same screen location.
    tc.V_3D_t[0] = torch.tensor([0.0, 0.0, 3.0], dtype=DTYPE)
    tc.V_3D_t[1] = torch.tensor([0.0, 0.0, 8.0], dtype=DTYPE)
    # Both should contribute at t_0 = their v_0. Use each Gaussian's own v_0 as t_0.
    # But they may have different v_0. Let's set both t_0 to a value where both are bright.
    # Simpler: override alpha_eff directly.
    tc.alpha_eff[:] = params.opacity   # ignore temporal fading
    sg = project_to_screen(params, tc, cam)
    img = rasterize(sg, H=60, W=80)
    pixel = img[30, 40]
    # Should be dominantly red (front occludes back).
    assert pixel[0] > 0.5
    assert pixel[0] > pixel[1]


def test_rasterize_batch_zero_gaussians():
    """Rendering with 0 valid Gaussians returns the background color."""
    cam = Camera(
        R=torch.eye(3, dtype=DTYPE), c=torch.zeros(3, dtype=DTYPE),
        fx=500, fy=500, cx=40, cy=30,
    )
    # Put the Gaussian behind the camera -> invalid.
    params = make_params(n=1)
    derived = compute_derived(params)
    tc = condition_on_time(params, derived, 0.0)
    tc.V_3D_t[0] = torch.tensor([0.0, 0.0, -5.0], dtype=DTYPE)   # behind!
    sg = project_to_screen(params, tc, cam)
    assert not sg.valid.any()
    bg = torch.tensor([0.1, 0.2, 0.3], dtype=DTYPE)
    img = rasterize(sg, H=60, W=80, background=bg)
    assert torch.allclose(img[0, 0], bg)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
