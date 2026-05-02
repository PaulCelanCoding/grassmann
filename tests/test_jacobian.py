"""Tests for projection.py and jacobian.py.

The crown jewel is test_jacobian_matches_finite_differences: it compares the
analytical Jacobian against a numerical one computed by torch.autograd.
If these agree to ~1e-5 for random inputs, Phase 2 is correct.
"""
import pytest
import torch

from grassmann import quaternion as Q
from grassmann import grassmann as G
from grassmann import jacobian as J
from grassmann.projection import (
    Camera,
    world_to_camera,
    perspective,
    project_static,
    perspective_jacobian,
)


# Use float64 throughout this phase for tight tolerances on finite differences.
DTYPE = torch.float64
torch.manual_seed(42)


def rand_camera(dtype=DTYPE) -> Camera:
    """Generate a random camera: random rotation, random center."""
    # Random rotation via QR decomposition of a random matrix.
    A = torch.randn(3, 3, dtype=dtype)
    Q_mat, _ = torch.linalg.qr(A)
    # Ensure det = +1 (proper rotation)
    if torch.det(Q_mat) < 0:
        Q_mat[:, 0] *= -1
    c = torch.randn(3, dtype=dtype) * 0.5
    return Camera(R=Q_mat, c=c, fx=800.0, fy=800.0, cx=320.0, cy=240.0)


def rand_pq(n=1, dtype=DTYPE, min_sep=0.3):
    """Random unit imaginary quaternion pairs, avoiding the antidiagonal."""
    ps, qs = [], []
    while len(ps) < n:
        pv = torch.randn(3, dtype=dtype)
        qv = torch.randn(3, dtype=dtype)
        pv = pv / pv.norm()
        qv = qv / qv.norm()
        if (pv * qv).sum() > -1 + min_sep:
            ps.append(Q.unit_imag(pv))
            qs.append(Q.unit_imag(qv))
    return torch.stack(ps), torch.stack(qs)


# =============================================================================
# Projection tests
# =============================================================================

def test_world_to_camera_identity():
    """Camera at origin with R = I should be the identity."""
    cam = Camera.at_origin(dtype=DTYPE)
    X = torch.randn(5, 3, dtype=DTYPE)
    assert torch.allclose(world_to_camera(X, cam), X)


def test_perspective_at_principal_point():
    """A point on the optical axis at any depth projects to the principal point."""
    cam = Camera.at_origin(fx=500, fy=500, cx=320, cy=240, dtype=DTYPE)
    # Points at (0, 0, Z) for various Z
    X = torch.tensor([[0, 0, 1], [0, 0, 5], [0, 0, 100]], dtype=DTYPE)
    uv = perspective(X, cam)
    expected = torch.tensor([[320, 240], [320, 240], [320, 240]], dtype=DTYPE)
    assert torch.allclose(uv, expected)


def test_perspective_jacobian_vs_autograd():
    """Compare analytic perspective Jacobian against autograd."""
    cam = rand_camera()
    X = torch.randn(3, dtype=DTYPE, requires_grad=True) + torch.tensor([0, 0, 5], dtype=DTYPE)

    # Analytic
    J_ana = perspective_jacobian(X.detach(), cam)

    # Autograd: compute each row of the Jacobian via grad.
    uv = perspective(X, cam)
    J_auto = torch.zeros(2, 3, dtype=DTYPE)
    for i in range(2):
        grad = torch.autograd.grad(uv[i], X, retain_graph=True)[0]
        J_auto[i] = grad

    assert torch.allclose(J_ana, J_auto, atol=1e-10), f"max err = {(J_ana - J_auto).abs().max()}"


def test_project_static_matches_composition():
    """project_static should equal perspective(world_to_camera(...))."""
    cam = rand_camera()
    X = torch.randn(7, 3, dtype=DTYPE) + torch.tensor([0, 0, 5], dtype=DTYPE)
    uv_direct = project_static(X, cam)
    uv_compose = perspective(world_to_camera(X, cam), cam)
    assert torch.allclose(uv_direct, uv_compose)


# =============================================================================
# J_embed and J_time
# =============================================================================

def test_jacobian_embed_columns_are_basis_spatial_parts():
    """J_embed columns should be spatial parts of (e1_hat, e2_hat)."""
    p, q = rand_pq(10)
    J_e = J.jacobian_embed(p, q)   # (10, 3, 2)

    e1_hat, e2_hat = G.orthonormal_basis(p, q)  # (10, 4) each
    # Spatial parts
    col1_expected = Q.imag(e1_hat)   # (10, 3)
    col2_expected = Q.imag(e2_hat)   # (10, 3)

    assert torch.allclose(J_e[..., 0], col1_expected, atol=1e-10)
    assert torch.allclose(J_e[..., 1], col2_expected, atol=1e-10)


def test_jacobian_time_structure():
    """J_time should be [0, sqrt((1+c)/2)]."""
    p, q = rand_pq(8)
    J_t = J.jacobian_time(p, q)     # (8, 1, 2)
    f = G.canonical_frame(p, q)
    expected_scale = torch.sqrt((1.0 + f.c) * 0.5)
    assert torch.allclose(J_t[..., 0, 0], torch.zeros_like(f.c), atol=1e-10)
    assert torch.allclose(J_t[..., 0, 1], expected_scale, atol=1e-10)


def test_jacobian_embed_columns_orthogonal():
    """The two columns of J_embed are spatial projections of (e1_hat, e2_hat),
    which are orthogonal in R^4. Are their SPATIAL parts orthogonal?

    e1_hat has time component 0, so its spatial dot product equals the full R^4
    dot product, which is 0 by orthogonality of e1_hat, e2_hat.
    So YES: the columns of J_embed should be orthogonal in R^3.
    """
    p, q = rand_pq(10)
    J_e = J.jacobian_embed(p, q)
    dots = (J_e[..., 0] * J_e[..., 1]).sum(dim=-1)
    assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-10)


# =============================================================================
# THE MAIN EVENT: J_full vs numerical Jacobian
# =============================================================================

def point_in_plane(alpha, beta, v_center_quat, e1_hat, e2_hat):
    """z(alpha, beta) = v_center + alpha * e1_hat + beta * e2_hat.

    All are quaternions of shape (..., 4). alpha, beta are scalars (or batched scalars).
    Returns shape (..., 4).
    """
    return v_center_quat + alpha.unsqueeze(-1) * e1_hat + beta.unsqueeze(-1) * e2_hat


def full_projection_static(alpha, beta, p, q, v_center_quat, cam):
    """The full map P: (alpha, beta) -> (u, v, t) for a STATIC camera.

    v_center_quat: the Gaussian mean v as a quaternion (time_component, spatial...), shape (4,).
    """
    e1_hat, e2_hat = G.orthonormal_basis(p, q)         # (4,) each
    z = v_center_quat + alpha * e1_hat + beta * e2_hat  # (4,)
    t = Q.real(z)                                       # scalar
    X_world = Q.imag(z)                                 # (3,)
    uv = project_static(X_world, cam)                   # (2,)
    return torch.stack([uv[0], uv[1], t])               # (3,)


def test_jacobian_full_static_matches_autograd():
    """The crown-jewel test. Compare analytical J_full (Proposition 6) against
    a numerical Jacobian computed by autograd, for random configurations."""
    for trial in range(10):
        cam = rand_camera()
        p, q = rand_pq(1)
        p, q = p[0], q[0]   # drop batch dim

        # Build a valid Gaussian mean v in E_{p,q}. Pick some (alpha_0, beta_0)
        # and set v = alpha_0 * e1_hat + beta_0 * e2_hat (starting from origin of plane,
        # which corresponds to the canonical point in E_{p,q}).
        # Actually v can be any point in E_{p,q}. The easiest: v = alpha_0 e1 + beta_0 e2.
        alpha_0 = torch.randn(1, dtype=DTYPE).squeeze() * 0.3
        beta_0 = torch.randn(1, dtype=DTYPE).squeeze() * 0.3
        e1_hat, e2_hat = G.orthonormal_basis(p, q)
        v_quat = alpha_0 * e1_hat + beta_0 * e2_hat        # (4,) in E_{p,q}
        # Make sure V (spatial part) is in front of the camera (positive Z in cam frame).
        V = Q.imag(v_quat)
        V_cam = cam.R @ (V - cam.c)
        if V_cam[2] < 0.5:
            # Nudge v so that the point is well in front of the camera.
            # We shift along e1_hat (pure spatial) so it stays in the plane.
            # We need to move V in the camera's +Z direction. Project that direction
            # onto the plane's spatial basis and shift v accordingly.
            # For simplicity: just retry the trial if bad.
            continue

        # --- Analytical Jacobian ---
        # Note: J.jacobian_full_static takes V (spatial part of v) as world coords.
        J_ana = J.jacobian_full_static(V, p, q, cam)       # (3, 2)

        # --- Numerical Jacobian via autograd ---
        # Treat the projection as P(alpha, beta) evaluated at (0, 0) (so the mean is v).
        alpha = torch.tensor(0.0, dtype=DTYPE, requires_grad=True)
        beta = torch.tensor(0.0, dtype=DTYPE, requires_grad=True)

        output = full_projection_static(alpha, beta, p, q, v_quat, cam)  # (3,)

        J_auto = torch.zeros(3, 2, dtype=DTYPE)
        for i in range(3):
            grads = torch.autograd.grad(output[i], (alpha, beta), retain_graph=True)
            J_auto[i, 0] = grads[0]
            J_auto[i, 1] = grads[1]

        err = (J_ana - J_auto).abs().max()
        assert err < 1e-8, f"Trial {trial}: analytical vs autograd Jacobian differs by {err}"


def test_jacobian_full_static_finite_difference():
    """Additional verification: compare analytical J_full against central finite differences."""
    cam = rand_camera()
    p, q = rand_pq(1)
    p, q = p[0], q[0]

    alpha_0 = torch.tensor(0.1, dtype=DTYPE)
    beta_0 = torch.tensor(0.05, dtype=DTYPE)
    e1_hat, e2_hat = G.orthonormal_basis(p, q)
    v_quat = alpha_0 * e1_hat + beta_0 * e2_hat

    V = Q.imag(v_quat)
    V_cam = cam.R @ (V - cam.c)
    # Keep retrying camera until V is in front.
    tries = 0
    while V_cam[2] < 0.5 and tries < 20:
        cam = rand_camera()
        V_cam = cam.R @ (V - cam.c)
        tries += 1
    assert V_cam[2] >= 0.5, "couldn't generate a valid camera"

    # Analytical
    J_ana = J.jacobian_full_static(V, p, q, cam)        # (3, 2)

    # Finite differences
    h = 1e-5
    alpha0 = torch.tensor(0.0, dtype=DTYPE)
    beta0 = torch.tensor(0.0, dtype=DTYPE)

    def eval_P(alpha, beta):
        return full_projection_static(alpha, beta, p, q, v_quat, cam)

    P_plus_a = eval_P(alpha0 + h, beta0)
    P_minus_a = eval_P(alpha0 - h, beta0)
    dP_dalpha = (P_plus_a - P_minus_a) / (2 * h)          # (3,)

    P_plus_b = eval_P(alpha0, beta0 + h)
    P_minus_b = eval_P(alpha0, beta0 - h)
    dP_dbeta = (P_plus_b - P_minus_b) / (2 * h)           # (3,)

    J_fd = torch.stack([dP_dalpha, dP_dbeta], dim=-1)     # (3, 2)

    err = (J_ana - J_fd).abs().max()
    # Central-difference truncation: O(h^2) = 1e-10. Roundoff: each projection involves
    # focal ~800, so error per output is ~800 * eps * chain ~ 1e-10, amplified by 1/(2h) = 5e4
    # gives ~5e-6 in pessimistic cases. The autograd test (above) checks to 1e-8;
    # this FD test is a cross-check only, so 1e-5 is plenty to catch real bugs.
    assert err < 1e-5, f"analytical vs finite-difference Jacobian differs by {err}"


def test_jacobian_time_row_is_independent_of_camera():
    """The time row of J_full should not depend on the camera. Verifying that
    changing the camera only changes the spatial rows."""
    p, q = rand_pq(1)
    p, q = p[0], q[0]
    V = torch.tensor([0.1, 0.2, 3.0], dtype=DTYPE)

    cam1 = rand_camera()
    cam2 = rand_camera()

    # Need V in front of both cameras.
    while (cam1.R @ (V - cam1.c))[2] < 0.5 or (cam2.R @ (V - cam2.c))[2] < 0.5:
        cam1 = rand_camera()
        cam2 = rand_camera()

    J1 = J.jacobian_full_static(V, p, q, cam1)
    J2 = J.jacobian_full_static(V, p, q, cam2)

    # Last row (time row) should be identical.
    assert torch.allclose(J1[2], J2[2], atol=1e-12)


def test_jacobian_static_equals_origin_case():
    """For cam = at_origin (R = I, c = 0), the static Jacobian should equal
    the 'camera-at-origin' form eq. (10) of the paper.
    """
    cam = Camera.at_origin(fx=500, fy=500, cx=320, cy=240, dtype=DTYPE)
    p, q = rand_pq(1)
    p, q = p[0], q[0]
    V = torch.tensor([0.1, 0.2, 5.0], dtype=DTYPE)

    J_full = J.jacobian_full_static(V, p, q, cam)

    # Manually build J_persp @ J_embed (no rotation) and stack J_time.
    J_pi = perspective_jacobian(V.unsqueeze(0), cam).squeeze(0)   # (2, 3)
    J_e = J.jacobian_embed(p, q)                                   # (3, 2)
    J_spatial_expected = J_pi @ J_e                                # (2, 2)
    J_time_expected = J.jacobian_time(p, q)                        # (1, 2)
    J_expected = torch.cat([J_spatial_expected, J_time_expected], dim=0)  # (3, 2)

    assert torch.allclose(J_full, J_expected, atol=1e-12)


# =============================================================================
# Batched shape propagation
# =============================================================================

def test_jacobian_embed_batched():
    p, q = rand_pq(5)
    J_e = J.jacobian_embed(p, q)
    assert J_e.shape == (5, 3, 2)


def test_jacobian_time_batched():
    p, q = rand_pq(5)
    J_t = J.jacobian_time(p, q)
    assert J_t.shape == (5, 1, 2)


def test_jacobian_full_static_batched():
    cam = rand_camera()
    p, q = rand_pq(5)
    V = torch.randn(5, 3, dtype=DTYPE) + torch.tensor([0, 0, 10], dtype=DTYPE)
    J_full = J.jacobian_full_static(V, p, q, cam)
    assert J_full.shape == (5, 3, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
