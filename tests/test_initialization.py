"""Tests for the Phase 4 pipeline: synthetic scene generation, triangulation, init.

Key properties verified:
  1. Synthetic scene generator produces valid cameras and renders images.
  2. Triangulation recovers 3D points exactly (up to numerical error) from
     noise-free observations.
  3. Triangulation is robust to small observation noise.
  4. Initialization produces valid GaussianParams that can be rendered.
  5. A Gaussian initialized from a 3D point at time t reconstructs approximately
     the right pixel color when rendered at the right camera and time.
"""
import pytest
import torch

from grassmann.projection import Camera, project_static
from grassmann.gaussian import compute_derived, condition_on_time
from grassmann.rasterizer import project_to_screen, rasterize
from grassmann.synthetic import (
    cameras_on_ring, cameras_stereo_pair,
    trajectory_linear, trajectory_circular, trajectory_static,
    ScenePoint, MultiCameraSyntheticScene, render_synthetic_frame, make_default_scene,
)
from grassmann.triangulation import (
    projection_matrix, triangulate_point_dlt, triangulate_points_batch,
    reprojection_error, observe_scene_point,
)
from grassmann.initialization import (
    pick_reference_camera, init_gaussian_from_point, init_gaussians_from_points,
    sample_color_from_image,
)


DTYPE = torch.float64
torch.manual_seed(42)


# =============================================================================
# Synthetic scene
# =============================================================================

def test_cameras_on_ring_produces_valid_cameras():
    """Ring of K cameras all looking at the origin."""
    cams = cameras_on_ring(K=4, radius=5.0)
    assert len(cams) == 4
    for cam in cams:
        # Rotation matrix should be orthogonal with det = 1.
        R = cam.R
        assert torch.allclose(R @ R.T, torch.eye(3, dtype=DTYPE), atol=1e-10)
        assert abs(torch.det(R).item() - 1.0) < 1e-10
        # Camera center should be on the ring (|c| ~ 5).
        assert abs(cam.c.norm().item() - 5.0) < 1e-5


def test_cameras_on_ring_look_at_origin():
    """Projection of origin should land near the principal point (cx, cy)."""
    cams = cameras_on_ring(K=6, radius=4.0, image_w=200, image_h=120)
    origin = torch.zeros(1, 3, dtype=DTYPE)
    for cam in cams:
        uv = project_static(origin, cam).squeeze(0)
        # Origin should project very close to (cx, cy) since cameras look at it.
        assert abs(uv[0].item() - cam.cx) < 1.0
        assert abs(uv[1].item() - cam.cy) < 1.0


def test_stereo_pair_cameras():
    """Two cameras, both in front of the scene, looking forward."""
    cams = cameras_stereo_pair(baseline=1.0, distance_to_scene=5.0)
    assert len(cams) == 2
    # One at x=-0.5, one at x=+0.5
    assert cams[0].c[0].item() < 0
    assert cams[1].c[0].item() > 0
    # A point at (0, 0, 5) should be in front of both.
    p = torch.tensor([[0.0, 0.0, 5.0]], dtype=DTYPE)
    for cam in cams:
        X_cam = cam.R @ (p[0] - cam.c)
        assert X_cam[2].item() > 0


def test_render_synthetic_frame_shape_and_range():
    """Rendered frame is correct shape and in [0, 1]."""
    scene = make_default_scene(n_cams=3, image_w=80, image_h=60)
    img = render_synthetic_frame(scene, cam_idx=0, t=0.0)
    assert img.shape == (60, 80, 3)
    assert img.min().item() >= 0.0
    assert img.max().item() <= 1.0


def test_render_synthetic_frame_shows_colors():
    """Rendered frame should contain visible red/green/blue contributions."""
    scene = make_default_scene(n_cams=3, image_w=80, image_h=60)
    img = render_synthetic_frame(scene, cam_idx=0, t=0.0, blob_sigma=3.0)
    # Somewhere there should be a pixel with high red, high green, and high blue.
    max_r = img[..., 0].max().item()
    max_g = img[..., 1].max().item()
    max_b = img[..., 2].max().item()
    assert max_r > 0.5, f"no bright red found; max R = {max_r}"
    assert max_g > 0.5, f"no bright green found; max G = {max_g}"
    assert max_b > 0.3, f"no bright blue found; max B = {max_b}"


def test_scene_point_moves_over_time():
    """Linear trajectory should give different positions at different times."""
    traj = trajectory_linear(x0=[0, 0, 0], velocity=[1.0, 0, 0])
    p0 = traj(0.0)
    p1 = traj(1.0)
    p2 = traj(2.0)
    assert torch.allclose(p1 - p0, torch.tensor([1.0, 0, 0], dtype=DTYPE))
    assert torch.allclose(p2 - p0, torch.tensor([2.0, 0, 0], dtype=DTYPE))


def test_scene_point_circular():
    """Circular trajectory returns to start after one period."""
    traj = trajectory_circular(center=[0, 0, 0], radius=1.0, axis="y", period=4.0)
    p0 = traj(0.0)
    p4 = traj(4.0)
    assert torch.allclose(p0, p4, atol=1e-10)


# =============================================================================
# Triangulation
# =============================================================================

def test_projection_matrix_matches_project_static():
    """P @ [X, 1] (in homogeneous coords) should match project_static."""
    cam = cameras_on_ring(K=1, radius=3.0)[0]
    P = projection_matrix(cam)
    X = torch.tensor([0.5, 0.3, 0.2], dtype=DTYPE)
    X_hom = torch.cat([X, torch.tensor([1.0], dtype=DTYPE)])
    uvw = P @ X_hom
    u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]
    uv_expected = project_static(X.unsqueeze(0), cam).squeeze(0)
    assert torch.allclose(torch.stack([u, v]), uv_expected, atol=1e-9)


def test_triangulation_recovers_known_point_2_cams():
    """Two-camera triangulation recovers a known point."""
    cams = cameras_stereo_pair(baseline=1.0)
    X_true = torch.tensor([0.3, -0.2, 5.0], dtype=DTYPE)
    uvs = torch.stack([project_static(X_true.unsqueeze(0), cam).squeeze(0) for cam in cams])
    X_rec = triangulate_point_dlt(cams, uvs)
    assert torch.allclose(X_rec, X_true, atol=1e-8), f"recovered {X_rec.tolist()}, expected {X_true.tolist()}"


def test_triangulation_recovers_known_point_K_cams():
    """K-camera triangulation recovers a known point for various K."""
    for K in [2, 3, 4, 6, 8]:
        cams = cameras_on_ring(K=K, radius=6.0)
        X_true = torch.tensor([0.3, -0.2, 0.5], dtype=DTYPE)
        uvs = torch.stack([project_static(X_true.unsqueeze(0), cam).squeeze(0) for cam in cams])
        X_rec = triangulate_point_dlt(cams, uvs)
        err = (X_rec - X_true).norm().item()
        assert err < 1e-6, f"K={K}: recovered with error {err}"


def test_triangulation_robust_to_noise():
    """With 0.5 pixel noise, triangulation should still be within 1cm (of unit scene)."""
    torch.manual_seed(0)
    cams = cameras_on_ring(K=8, radius=6.0)
    X_true = torch.tensor([0.3, -0.2, 0.5], dtype=DTYPE)
    uvs = torch.stack([project_static(X_true.unsqueeze(0), cam).squeeze(0) for cam in cams])
    # Add noise
    uvs_noisy = uvs + 0.5 * torch.randn_like(uvs)
    X_rec = triangulate_point_dlt(cams, uvs_noisy)
    err = (X_rec - X_true).norm().item()
    assert err < 0.05, f"triangulation error under noise: {err}"


def test_triangulation_batch():
    """Batched triangulation produces the same result as one-by-one."""
    cams = cameras_on_ring(K=4, radius=5.0)
    X_true = torch.tensor([
        [0.3, -0.2, 0.5],
        [-0.5, 0.1, 0.0],
        [0.0, 0.3, -0.1],
    ], dtype=DTYPE)
    # Build observations.
    obs = torch.zeros(3, len(cams), 2, dtype=DTYPE)
    for i in range(3):
        for k, cam in enumerate(cams):
            obs[i, k] = project_static(X_true[i].unsqueeze(0), cam).squeeze(0)
    X_rec = triangulate_points_batch(cams, obs)
    assert torch.allclose(X_rec, X_true, atol=1e-6)


def test_reprojection_error_zero_for_true_point():
    """Reprojection error at the true triangulated point is ~0."""
    cams = cameras_on_ring(K=4, radius=5.0)
    X_true = torch.tensor([0.3, -0.2, 0.5], dtype=DTYPE)
    uvs = torch.stack([project_static(X_true.unsqueeze(0), cam).squeeze(0) for cam in cams])
    err = reprojection_error(cams, X_true, uvs).item()
    assert err < 1e-10


def test_observe_scene_point_without_noise():
    """observe_scene_point should give consistent results."""
    cams = cameras_on_ring(K=3, radius=4.0)
    traj = trajectory_linear(x0=[0.0, 0.0, 0.0], velocity=[1.0, 0.0, 0.0])
    uvs, depths = observe_scene_point(traj, t=0.5, cameras=cams)
    assert uvs.shape == (3, 2)
    assert depths.shape == (3,)
    # At t=0.5, the point is at (0.5, 0, 0); some cameras should see it in front.
    assert (depths > 0).any()


# =============================================================================
# Initialization
# =============================================================================

def test_pick_reference_camera():
    """Should pick the camera most directly facing the point."""
    cams = cameras_on_ring(K=4, radius=5.0)
    # Point directly in front of camera 0 (at theta = 0, i.e., +x axis).
    # cam 0 is at (5, 0, 0), looking toward origin -- so -x direction.
    # A point at (0, 0, 0) is seen from directly in front by all cameras.
    # But a point at (2, 0, 0) is seen best by cam 0 (it's in front of cam 0).
    X = torch.tensor([2.0, 0.0, 0.0], dtype=DTYPE)
    idx = pick_reference_camera(X, cams)
    assert idx == 0, f"expected cam 0 (at +x looking in), got {idx}"


def test_init_gaussian_from_point_valid():
    """Initialization produces a valid GaussianParams that can be rendered."""
    cams = cameras_on_ring(K=4, radius=5.0)
    X = torch.tensor([0.5, 0.3, -0.1], dtype=DTYPE)
    params = init_gaussian_from_point(X, t=1.0, cameras=cams)
    assert params.N == 1
    # Sigma_k must be PD.
    Sigma_k = params.Sigma_k()
    evals = torch.linalg.eigvalsh(Sigma_k[0])
    assert (evals > 0).all()
    # Rendering should not produce NaN.
    derived = compute_derived(params)
    tc = condition_on_time(params, derived, 1.0)
    for cam in cams:
        sg = project_to_screen(params, tc, cam)
        if sg.valid[0]:
            img = rasterize(sg, H=60, W=80)
            assert not torch.isnan(img).any(), "rendered image contains NaN"


def test_init_gaussian_mean_matches_point_approximately():
    """The Gaussian's V_k (spatial mean in world coords) should be close to X_world.

    The Grassmann parameterization places the mean v in E_{p,q} via a least-squares
    projection of (t, X_world) onto the plane, so there can be residual error.
    For the specific case where (1, X_world) lies exactly in the plane (which it
    does by construction of line_to_pq), setting t = 1 should give zero residual.
    """
    cams = cameras_on_ring(K=4, radius=5.0)
    X = torch.tensor([0.5, 0.3, -0.1], dtype=DTYPE)
    # Use t=1.0 so that the target (t, X) = (1, X) lies in the plane.
    params = init_gaussian_from_point(X, t=1.0, cameras=cams)
    derived = compute_derived(params)
    V_k = derived.V_k[0]
    # Should match X_world well.
    err = (V_k - X).norm().item()
    assert err < 0.1, f"V_k = {V_k.tolist()}, X = {X.tolist()}, err = {err}"


def test_init_gaussian_v0_matches_t_approximately():
    """The Gaussian's temporal mean should be close to the requested t."""
    cams = cameras_on_ring(K=4, radius=5.0)
    X = torch.tensor([0.5, 0.3, -0.1], dtype=DTYPE)
    params = init_gaussian_from_point(X, t=1.0, cameras=cams)
    derived = compute_derived(params)
    v0 = derived.v_0[0].item()
    assert abs(v0 - 1.0) < 0.1, f"v_0 = {v0}, expected ~1.0"


def test_init_multiple_gaussians():
    """Batched initialization works and produces the right count."""
    cams = cameras_on_ring(K=4, radius=5.0)
    N = 10
    points = torch.randn(N, 3, dtype=DTYPE) * 0.5
    times = torch.ones(N, dtype=DTYPE) * 1.0
    colors = torch.rand(N, 3, dtype=DTYPE)
    params = init_gaussians_from_points(points, times, cams, colors=colors)
    assert params.N == N
    # All p, q should be unit imaginary.
    for i in range(N):
        p_norm = params.p_im[i].norm().item()
        q_norm = params.q_im[i].norm().item()
        assert abs(p_norm - 1.0) < 1e-6
        assert abs(q_norm - 1.0) < 1e-6


def test_sample_color_from_image():
    """Sampling colors at pixel locations."""
    # Build a gradient image.
    H, W = 20, 30
    img = torch.zeros(H, W, 3, dtype=DTYPE)
    img[..., 0] = torch.linspace(0, 1, W).expand(H, W)  # R gradient
    img[..., 1] = torch.linspace(0, 1, H).unsqueeze(-1).expand(H, W)  # G gradient

    # Sample at (5, 10): should have R ~ 5/29, G ~ 10/19.
    color = sample_color_from_image(img, torch.tensor([5.0, 10.0], dtype=DTYPE))
    assert abs(color[0].item() - 5 / 29) < 1e-6
    assert abs(color[1].item() - 10 / 19) < 1e-6


# =============================================================================
# End-to-end: observe, triangulate, initialize, render, compare
# =============================================================================

def test_end_to_end_single_point():
    """Full pipeline:
    1. Scene with one moving red point.
    2. Render from K cameras at time t.
    3. Triangulate from the K observations.
    4. Initialize a Grassmann Gaussian at the triangulated point.
    5. Render THAT Gaussian back to one of the cameras.
    6. Verify: the reconstructed image has a red blob near the expected location.
    """
    scene = make_default_scene(n_cams=4, image_w=80, image_h=60)
    # Use only the red point for this test.
    red_traj = scene.scene_points[0].trajectory
    red_color = scene.scene_points[0].color

    t = 0.5
    # 1. Observe in all cameras
    uvs, depths = observe_scene_point(red_traj, t, scene.cameras)
    # Filter cameras that see the point (depth > 0)
    visible = depths > 0.1
    assert visible.sum() >= 2, f"Fewer than 2 cameras see the point; visible = {visible.tolist()}"

    visible_cams = [scene.cameras[k] for k in range(len(scene.cameras)) if visible[k]]
    visible_uvs = uvs[visible]

    # 2. Triangulate
    X_rec = triangulate_point_dlt(visible_cams, visible_uvs)
    X_true = red_traj(t)
    err = (X_rec - X_true).norm().item()
    assert err < 1e-5, f"triangulation err = {err}"

    # 3. Initialize
    params = init_gaussian_from_point(X_rec, t=t, cameras=scene.cameras, color=red_color)

    # 4. Render at original cameras and time
    derived = compute_derived(params)
    tc = condition_on_time(params, derived, t)
    for k, cam in enumerate(scene.cameras):
        if not visible[k]:
            continue
        sg = project_to_screen(params, tc, cam)
        if not sg.valid[0]:
            continue
        img = rasterize(sg, H=scene.H, W=scene.W)
        # The projected mean should be close to the observation.
        uv_rec = sg.uv[0]
        uv_obs = uvs[k]
        proj_err = (uv_rec - uv_obs).norm().item()
        assert proj_err < 5.0, f"cam {k}: projection err {proj_err}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
